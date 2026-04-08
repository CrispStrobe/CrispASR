// qwen3-asr-main — CLI for Qwen/Qwen3-ASR-0.6B ggml runtime.
//
// Reads a 16 kHz mono WAV, runs the full audio→text pipeline (mel → encoder
// → splice into ChatML prompt → Qwen3 LLM forward with KV cache → greedy
// decode), and prints the transcript.
//
// Usage:
//   qwen3-asr-main -m qwen3-asr-0.6b.gguf -f audio.wav [-t 4]

#include "qwen3_asr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

// ---------------------------------------------------------------------------
// Tiny WAV reader: 16-bit PCM, mono, 16 kHz only.
// ---------------------------------------------------------------------------
static bool load_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    auto read32 = [&](uint32_t & v) { f.read((char*)&v, 4); };
    auto read16 = [&](uint16_t & v) { f.read((char*)&v, 2); };

    char riff[4]; f.read(riff, 4);
    uint32_t riff_size; read32(riff_size);
    char wave[4]; f.read(wave, 4);
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "%s: not a RIFF/WAVE file\n", path.c_str()); return false;
    }

    uint16_t audio_format = 0, n_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data;

    while (f) {
        char chunk_id[4]; f.read(chunk_id, 4);
        uint32_t chunk_sz; read32(chunk_sz);
        if (!f) break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            read16(audio_format); read16(n_channels);
            read32(sample_rate);
            uint32_t byte_rate; read32(byte_rate);
            uint16_t block_align; read16(block_align);
            read16(bits_per_sample);
            if (chunk_sz > 16) f.seekg(chunk_sz - 16, std::ios::cur);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data.resize(chunk_sz);
            f.read((char*)data.data(), chunk_sz);
            break;
        } else {
            f.seekg(chunk_sz, std::ios::cur);
        }
    }
    if (audio_format != 1 || bits_per_sample != 16) {
        fprintf(stderr, "%s: only 16-bit PCM supported (got format=%d bits=%d)\n",
                path.c_str(), audio_format, bits_per_sample); return false;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "%s: only 16 kHz supported (got %u). Pre-convert with:\n"
                        "  ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav\n",
                path.c_str(), sample_rate); return false;
    }
    const int16_t * pcm = (const int16_t *)data.data();
    size_t n_samples_total = data.size() / 2;
    if (n_channels == 1) {
        out.resize(n_samples_total);
        for (size_t i = 0; i < n_samples_total; i++) out[i] = pcm[i] / 32768.0f;
    } else {
        size_t n_frames = n_samples_total / n_channels;
        out.resize(n_frames);
        for (size_t i = 0; i < n_frames; i++) {
            float s = 0;
            for (int c = 0; c < n_channels; c++) s += pcm[i * n_channels + c] / 32768.0f;
            out[i] = s / n_channels;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Build the ChatML prompt token IDs by tokenising the template via the
// model's BPE encoder. This replaces the previous hardcoded ID list and
// makes it trivial to add an explicit language hint or system prompt.
// ---------------------------------------------------------------------------
static std::vector<int32_t> build_prompt_ids(qwen3_asr_context * ctx, int n_audio_pad) {
    // Build the chat-template string with N copies of <|audio_pad|> in the
    // user message. The BPE tokenizer recognises all the special tokens.
    std::string text =
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|>";
    text.reserve(text.size() + 13 * n_audio_pad + 64);
    for (int i = 0; i < n_audio_pad; i++) text += "<|audio_pad|>";
    text +=
        "<|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n";

    int n = 0;
    int32_t * raw = qwen3_asr_tokenize(ctx, text.c_str(), &n);
    std::vector<int32_t> ids;
    if (raw) {
        ids.assign(raw, raw + n);
        free(raw);
    }
    return ids;
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level decoder: maps a vocab string (made of unicode chars from
// the byte-encoder mapping) back to a UTF-8 byte sequence.
// The byte_encoder maps bytes 0-255 → safe unicode codepoints; we invert it.
// ---------------------------------------------------------------------------
static std::vector<int> & byte_decoder() {
    static std::vector<int> dec(0x200, -1);  // unicode codepoint → byte
    static bool initialized = false;
    if (initialized) return dec;
    // Same construction as GPT-2 / Qwen2 tokenizers:
    //   bs = list of "printable" bytes (33..126, 161..172, 174..255)
    //   then for each byte n in 0..255 not yet in bs, append to bs and assign
    //   unicode codepoint 256+offset
    std::vector<int> bs;
    std::vector<int> cs;
    for (int b = 0x21; b <= 0x7e; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xa1; b <= 0xac; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xae; b <= 0xff; b++) { bs.push_back(b); cs.push_back(b); }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool present = false;
        for (int x : bs) if (x == b) { present = true; break; }
        if (!present) { bs.push_back(b); cs.push_back(256 + n); n++; }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        if ((size_t)cs[i] < dec.size()) dec[cs[i]] = bs[i];
    }
    initialized = true;
    return dec;
}

// Decode UTF-8 string into codepoints, inverse-map each via byte_decoder, return raw bytes.
static std::string decode_token(const std::string & s) {
    auto & dec = byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int cp = 0;
        int len = 1;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i++; continue; }
        if (i + len > s.size()) break;
        for (int k = 1; k < len; k++) cp = (cp << 6) | (s[i + k] & 0x3F);
        i += len;
        if (cp >= 0 && cp < (int)dec.size() && dec[cp] >= 0) {
            out.push_back((char)dec[cp]);
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path;
    int n_threads = 4;
    int max_new = 256;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m" && i+1 < argc) model_path = argv[++i];
        else if (a == "-f" && i+1 < argc) audio_path = argv[++i];
        else if (a == "-t" && i+1 < argc) n_threads  = std::atoi(argv[++i]);
        else if (a == "-n" && i+1 < argc) max_new    = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            fprintf(stderr, "usage: %s -m model.gguf -f audio.wav [-t N] [-n MAX_NEW]\n", argv[0]);
            return 0;
        }
    }
    if (model_path.empty() || audio_path.empty()) {
        fprintf(stderr, "missing -m or -f. -h for help.\n"); return 1;
    }

    // ----- Load WAV -----
    std::vector<float> samples;
    if (!load_wav_16k_mono(audio_path, samples)) return 2;
    fprintf(stderr, "audio: %.2f s (%zu samples)\n", samples.size() / 16000.0, samples.size());

    // ----- Init context -----
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = n_threads;
    cp.verbosity = 1;
    auto * ctx = qwen3_asr_init_from_file(model_path.c_str(), cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 3; }

    auto t0 = std::chrono::steady_clock::now();

    // ----- Mel -----
    int n_mels = 0, T_mel = 0;
    float * mel = qwen3_asr_compute_mel(ctx, samples.data(), (int)samples.size(),
                                        &n_mels, &T_mel);
    if (!mel) { fprintf(stderr, "mel failed\n"); qwen3_asr_free(ctx); return 4; }
    auto t_mel = std::chrono::steady_clock::now();
    fprintf(stderr, "mel: %d × %d  (%.0f ms)\n", n_mels, T_mel,
            std::chrono::duration<double, std::milli>(t_mel - t0).count());

    // ----- Encoder -----
    int N_enc = 0, pdim = 0;
    float * audio_embeds = qwen3_asr_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) { fprintf(stderr, "encoder failed\n"); qwen3_asr_free(ctx); return 5; }
    auto t_enc = std::chrono::steady_clock::now();
    fprintf(stderr, "encoder: %d frames × %d dim  (%.0f ms)\n", N_enc, pdim,
            std::chrono::duration<double, std::milli>(t_enc - t_mel).count());

    // ----- Build prompt + embed text + splice audio -----
    auto ids = build_prompt_ids(ctx, N_enc);
    int T_prompt = (int)ids.size();
    fprintf(stderr, "prompt: %d tokens (incl. %d audio_pad)\n", T_prompt, N_enc);

    float * text_embeds = qwen3_asr_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { fprintf(stderr, "embed failed\n"); free(audio_embeds); qwen3_asr_free(ctx); return 6; }

    // Find audio_pad positions and splice
    const int AUDIO_PAD = 151676;
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == AUDIO_PAD) {
            std::memcpy(text_embeds + (size_t)i * pdim,
                        audio_embeds + (size_t)spliced * pdim,
                        pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);
    fprintf(stderr, "spliced %d audio frames\n", spliced);

    // ----- KV cache + prefill -----
    if (!qwen3_asr_kv_init(ctx, /*max_ctx*/ 4096)) {
        fprintf(stderr, "kv init failed\n"); free(text_embeds); qwen3_asr_free(ctx); return 7;
    }
    qwen3_asr_kv_reset(ctx);

    auto t_pf0 = std::chrono::steady_clock::now();
    int n_t = 0, vocab = 0;
    float * logits = qwen3_asr_run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_t, &vocab);
    auto t_pf1 = std::chrono::steady_clock::now();
    if (!logits) { fprintf(stderr, "prefill failed\n"); free(text_embeds); qwen3_asr_free(ctx); return 8; }
    fprintf(stderr, "prefill: %.0f ms\n",
            std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count());
    free(text_embeds);

    // First token from prefill
    int next = 0; float mx = -1e30f;
    for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; next = k; }
    free(logits);

    // ----- Greedy decode loop -----
    const int EOS = 151645;
    std::vector<int32_t> gen;
    gen.push_back(next);

    auto t_dec0 = std::chrono::steady_clock::now();
    int n_past = T_prompt;
    while ((int)gen.size() < max_new && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = qwen3_asr_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        int n_t2 = 0, v2 = 0;
        float * lg = qwen3_asr_run_llm_kv(ctx, tail, 1, n_past, &n_t2, &v2);
        free(tail);
        if (!lg) break;
        n_past += 1;
        int nx = 0; float lmx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > lmx) { lmx = lg[k]; nx = k; }
        free(lg);
        gen.push_back(nx);
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    fprintf(stderr, "decode: %zu tokens in %.0f ms (%.0f ms/token)\n",
            gen.size() - 1, dec_ms, dec_ms / std::max((size_t)1, gen.size() - 1));

    // ----- Decode token IDs to text -----
    // Qwen3-ASR emits a language-tag prefix before the transcript:
    //   "language" + " <LangName>" + <special-token, often 151704>
    // followed by the actual text. Capture the language name and strip the
    // prefix; also drop any "<|...|>" special-token names and "[PADxxx]"
    // placeholders for tokens beyond vocab.json's 151643 normal entries.
    std::string transcript;
    std::string detected_language;
    bool capture_language = false;
    for (auto id : gen) {
        if (id == EOS) break;
        std::string raw = qwen3_asr_token_text(ctx, id);
        if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') continue;
        if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D') continue;
        std::string txt = decode_token(raw);
        if (txt == "language") { capture_language = true; continue; }
        if (capture_language) {
            // Strip leading whitespace from the language name
            size_t s = 0;
            while (s < txt.size() && (txt[s] == ' ' || txt[s] == '\t')) s++;
            detected_language = txt.substr(s);
            capture_language = false;
            continue;
        }
        transcript += txt;
    }
    // Trim leading whitespace from transcript
    size_t start = 0;
    while (start < transcript.size() && (transcript[start] == ' ' || transcript[start] == '\t')) start++;
    transcript = transcript.substr(start);

    if (!detected_language.empty()) {
        fprintf(stderr, "language: %s\n", detected_language.c_str());
    }

    auto t_total = std::chrono::steady_clock::now();
    fprintf(stderr, "total: %.0f ms\n",
            std::chrono::duration<double, std::milli>(t_total - t0).count());

    fprintf(stdout, "%s\n", transcript.c_str());
    qwen3_asr_free(ctx);
    return 0;
}
