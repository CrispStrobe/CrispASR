// crispasr_backend_qwen3.cpp — adapter for Qwen/Qwen3-ASR-0.6B.
//
// Pipeline: mel -> encoder -> ChatML prompt (tokenized via BPE) with
// <|audio_pad|> placeholders -> embed + splice encoder frames -> KV
// prefill -> greedy decode -> GPT-2 byte-encoded detokenize.
//
// Qwen3's token_text() returns GPT-2 byte-encoded strings rather than
// raw bytes, so this backend has its own byte_decoder() helper (the
// standard GPT-2 mapping). When the src/core/ BPE helpers land, this
// will be factored out.
//
// Direct port of examples/qwen3-asr-main/main.cpp wrapped in the
// CrispasrBackend interface.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "core/greedy_decode.h"

#include "qwen3_asr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// GPT-2 byte encoder inverse: maps printable Unicode code points back to
// raw bytes 0..255. This is the standard GPT-2 tokenizer byte decoder and
// is shared by several BPE-based models (qwen3, parakeet, canary, whisper).
// It will move to src/core/bpe.{h,cpp} as part of the DRY refactor.
std::vector<int> & byte_decoder() {
    static std::vector<int> dec(0x200, -1);
    static bool initialized = false;
    if (initialized) return dec;
    std::vector<int> bs, cs;
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

std::string decode_token(const std::string & s) {
    auto & dec = byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int cp = 0, len = 1;
        if      (c < 0x80)           { cp = c;        len = 1; }
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

class Qwen3Backend : public CrispasrBackend {
public:
    Qwen3Backend() = default;
    ~Qwen3Backend() override { shutdown(); }

    const char * name() const override { return "qwen3"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_LANGUAGE_DETECT | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params & p) override {
        auto cp = qwen3_asr_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        ctx_ = qwen3_asr_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[qwen3]: failed to load model '%s'\n",
                    p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(
        const float * samples, int n_samples,
        int64_t t_offset_cs,
        const whisper_params & params) override
    {
        std::vector<crispasr_segment> out;
        if (!ctx_) return out;

        // ---- Mel ----
        int n_mels = 0, T_mel = 0;
        float * mel = qwen3_asr_compute_mel(ctx_, samples, n_samples, &n_mels, &T_mel);
        if (!mel) { fprintf(stderr, "crispasr[qwen3]: mel failed\n"); return out; }

        // ---- Encoder ----
        int N_enc = 0, pdim = 0;
        float * audio_embeds = qwen3_asr_run_encoder(ctx_, mel, n_mels, T_mel, &N_enc, &pdim);
        free(mel);
        if (!audio_embeds) {
            fprintf(stderr, "crispasr[qwen3]: encoder failed\n");
            return out;
        }

        // ---- ChatML prompt: <|im_start|>system\n<|im_end|>\n
        //                     <|im_start|>user\n<|audio_start|>
        //                     <|audio_pad|> x N
        //                     <|audio_end|><|im_end|>\n
        //                     <|im_start|>assistant\n
        std::string text =
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|>";
        text.reserve(text.size() + (size_t)N_enc * 13 + 64);
        for (int i = 0; i < N_enc; i++) text += "<|audio_pad|>";
        text +=
            "<|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n";

        int n_prompt = 0;
        int32_t * raw_ids = qwen3_asr_tokenize(ctx_, text.c_str(), &n_prompt);
        if (!raw_ids) {
            fprintf(stderr, "crispasr[qwen3]: tokenize failed\n");
            free(audio_embeds);
            return out;
        }
        std::vector<int32_t> ids(raw_ids, raw_ids + n_prompt);
        free(raw_ids);

        // Look up the audio_pad token id by tokenizing just the special token.
        int n_pad_id = 0;
        int32_t * pad_id_arr = qwen3_asr_tokenize(ctx_, "<|audio_pad|>", &n_pad_id);
        int audio_pad_id = -1;
        if (pad_id_arr && n_pad_id >= 1) audio_pad_id = pad_id_arr[0];
        free(pad_id_arr);
        if (audio_pad_id < 0) {
            fprintf(stderr, "crispasr[qwen3]: could not resolve <|audio_pad|> id\n");
            free(audio_embeds);
            return out;
        }

        // ---- Embed + splice ----
        float * text_embeds = qwen3_asr_embed_tokens(ctx_, ids.data(), (int)ids.size());
        if (!text_embeds) {
            fprintf(stderr, "crispasr[qwen3]: embed failed\n");
            free(audio_embeds);
            return out;
        }
        int spliced = 0;
        for (size_t i = 0; i < ids.size() && spliced < N_enc; i++) {
            if (ids[i] == audio_pad_id) {
                std::memcpy(text_embeds + i * pdim,
                            audio_embeds + (size_t)spliced * pdim,
                            pdim * sizeof(float));
                spliced++;
            }
        }
        free(audio_embeds);

        // ---- KV cache + prefill ----
        if (!qwen3_asr_kv_init(ctx_, 4096)) {
            free(text_embeds);
            fprintf(stderr, "crispasr[qwen3]: kv_init failed\n");
            return out;
        }
        qwen3_asr_kv_reset(ctx_);

        int n_t = 0, vocab = 0;
        float * logits = qwen3_asr_run_llm_kv(ctx_, text_embeds, (int)ids.size(), 0, &n_t, &vocab);
        free(text_embeds);
        if (!logits) {
            fprintf(stderr, "crispasr[qwen3]: prefill failed\n");
            return out;
        }

        // Last-token logits argmax.
        const int last_off = (n_t - 1) * vocab;
        int next = 0;
        {
            float mx = -1e30f;
            for (int k = 0; k < vocab; k++) {
                if (logits[last_off + k] > mx) { mx = logits[last_off + k]; next = k; }
            }
        }
        free(logits);

        // ---- Greedy decode via src/core/greedy_decode.h ----
        // Qwen3 EOS tokens: <|im_end|> (id unknown — look up via tokenize).
        int eos_id = -1;
        int n_eos = 0;
        int32_t * eos_arr = qwen3_asr_tokenize(ctx_, "<|im_end|>", &n_eos);
        if (eos_arr && n_eos >= 1) eos_id = eos_arr[0];
        free(eos_arr);

        core_greedy_decode::Config dec_cfg;
        dec_cfg.max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : 256;
        dec_cfg.eos_id         = eos_id;
        dec_cfg.vocab_size     = vocab;
        auto gen = core_greedy_decode::run(
            ctx_,
            /*first_token=*/next,
            /*initial_n_past=*/(int)ids.size(),
            qwen3_asr_embed_tokens,
            qwen3_asr_run_llm_kv,
            dec_cfg);

        // ---- Detokenize via GPT-2 byte decoder ----
        // Qwen3-ASR emits structured metadata tokens before the transcript:
        // special tokens like <|im_start|>, bracketed tags like <asr_text>,
        // and a "language <name>" prefix. Filter all of that out and keep
        // only the transcript itself.
        std::string transcript;
        std::string detected_language;
        bool capture_language = false;
        for (int32_t id : gen) {
            if (id == eos_id) break;
            const char * raw_piece = qwen3_asr_token_text(ctx_, id);
            if (!raw_piece || !*raw_piece) continue;
            std::string raw = raw_piece;

            // Skip Qwen3 special tokens: <|im_start|>, <|audio_pad|>, ...
            if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') continue;
            // Skip structured tags like <asr_text>, <punc>, ...
            if (raw.size() >= 2 && raw[0] == '<' && raw.back() == '>') continue;
            // Skip [PAD...] style placeholders if any leaked through.
            if (raw.size() >= 5 && raw[0] == '[' && raw[1] == 'P' && raw[2] == 'A' && raw[3] == 'D') continue;

            std::string txt = decode_token(raw);
            if (txt == "language") { capture_language = true; continue; }
            if (capture_language) {
                size_t s = 0;
                while (s < txt.size() && (txt[s] == ' ' || txt[s] == '\t')) s++;
                detected_language = txt.substr(s);
                capture_language = false;
                continue;
            }
            transcript += txt;
        }

        // Trim leading whitespace left over from the prompt template.
        while (!transcript.empty() &&
               (transcript.front() == ' ' || transcript.front() == '\n')) {
            transcript.erase(transcript.begin());
        }

        if (!params.no_prints && !detected_language.empty()) {
            fprintf(stderr, "crispasr[qwen3]: detected language: %s\n",
                    detected_language.c_str());
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = transcript;
        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            qwen3_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    qwen3_asr_context * ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_backend() {
    return std::unique_ptr<CrispasrBackend>(new Qwen3Backend());
}
