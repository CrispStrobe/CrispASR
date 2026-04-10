// voxtral-main — CLI for Mistral Voxtral-Mini-3B-2507.
//
// Reads a 16 kHz mono WAV, runs the full audio→text pipeline (mel → encoder
// → splice into [INST] prompt → Llama 3 LLM with KV cache → greedy decode),
// prints the transcript.
//
// Optional word-level timestamps via a CTC aligner second pass (-am flag).
//
// Usage:
//   voxtral-main -m voxtral-mini-3b-2507.gguf -f audio.wav [-t 4] [-l en]
//   voxtral-main -m model.gguf -f audio.wav -am aligner.gguf -timestamps

#include "voxtral.h"
#include "canary_ctc.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>

static bool load_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    auto read32 = [&](uint32_t & v) { f.read((char*)&v, 4); };
    auto read16 = [&](uint16_t & v) { f.read((char*)&v, 2); };
    char riff[4]; f.read(riff, 4); uint32_t riff_size; read32(riff_size);
    char wave[4]; f.read(wave, 4);
    if (memcmp(riff,"RIFF",4)!=0||memcmp(wave,"WAVE",4)!=0) return false;
    uint16_t afmt=0,nchan=0,bps=0; uint32_t sr=0; std::vector<uint8_t> data;
    while (f) {
        char cid[4]; f.read(cid,4); uint32_t csz; read32(csz); if(!f) break;
        if (!memcmp(cid,"fmt ",4)) {
            read16(afmt); read16(nchan); read32(sr);
            uint32_t br; read32(br); uint16_t ba; read16(ba); read16(bps);
            if (csz > 16) f.seekg(csz-16, std::ios::cur);
        } else if (!memcmp(cid,"data",4)) {
            data.resize(csz); f.read((char*)data.data(), csz); break;
        } else f.seekg(csz, std::ios::cur);
    }
    if (afmt!=1||bps!=16) { fprintf(stderr, "%s: only 16-bit PCM\n", path.c_str()); return false; }
    if (sr!=16000) { fprintf(stderr, "%s: need 16kHz (got %u)\n", path.c_str(), sr); return false; }
    const int16_t * pcm = (const int16_t *)data.data();
    size_t ns = data.size()/2;
    if (nchan == 1) {
        out.resize(ns); for (size_t i=0;i<ns;i++) out[i]=pcm[i]/32768.0f;
    } else {
        size_t nf = ns/nchan; out.resize(nf);
        for (size_t i=0;i<nf;i++) { float s=0; for(int c=0;c<nchan;c++) s+=pcm[i*nchan+c]/32768.0f; out[i]=s/nchan; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Word tokenizer for CTC alignment
// ---------------------------------------------------------------------------
static std::vector<std::string> tokenise_words(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// ---------------------------------------------------------------------------
// SRT / VTT output helpers
// ---------------------------------------------------------------------------
static std::string format_time_srt(int64_t cs) {
    int h = (int)(cs / 360000);
    int m = (int)((cs % 360000) / 6000);
    int s = (int)((cs % 6000) / 100);
    int ms = (int)(cs % 100) * 10;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", h, m, s, ms);
    return buf;
}

static std::string format_time_vtt(int64_t cs) {
    int h = (int)(cs / 360000);
    int m = (int)((cs % 360000) / 6000);
    int s = (int)((cs % 6000) / 100);
    int ms = (int)(cs % 100) * 10;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", h, m, s, ms);
    return buf;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s -m MODEL.gguf -f AUDIO.wav [options]\n\n"
        "options:\n"
        "  -h, --help       show this help\n"
        "  -m  FNAME        voxtral GGUF model (required)\n"
        "  -f  FNAME        input audio, 16 kHz mono WAV (required)\n"
        "  -t  N            threads (default: 4)\n"
        "  -l  LANG         language hint: en de fr es it pt nl hi (default: en)\n"
        "  -n  N            max new tokens (default: 512)\n"
        "  -am FNAME        CTC aligner GGUF (canary-ctc-aligner) for timestamps\n"
        "  -timestamps      enable word-level timestamps (requires -am)\n"
        "  -osrt            output SRT subtitle file (to stdout)\n"
        "  -ovtt            output VTT subtitle file (to stdout)\n"
        "  -np              no prints (suppress stderr info)\n"
        "\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path, aligner_path, lang = "en";
    int n_threads = 4, max_new = 512;
    bool timestamps = false;
    bool out_srt = false;
    bool out_vtt = false;
    bool no_prints = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "-m"  && i+1 < argc) model_path   = argv[++i];
        else if (a == "-f"  && i+1 < argc) audio_path   = argv[++i];
        else if (a == "-t"  && i+1 < argc) n_threads    = std::atoi(argv[++i]);
        else if (a == "-l"  && i+1 < argc) lang         = argv[++i];
        else if (a == "-n"  && i+1 < argc) max_new      = std::atoi(argv[++i]);
        else if (a == "-am" && i+1 < argc) aligner_path = argv[++i];
        else if (a == "-timestamps")       timestamps    = true;
        else if (a == "-osrt")           { timestamps = true; out_srt = true; }
        else if (a == "-ovtt")           { timestamps = true; out_vtt = true; }
        else if (a == "-np")               no_prints    = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (model_path.empty() || audio_path.empty()) {
        fprintf(stderr, "missing -m or -f. -h for help.\n"); return 1;
    }
    if (timestamps && aligner_path.empty()) {
        fprintf(stderr, "error: -timestamps / -osrt / -ovtt require -am ALIGNER.gguf\n");
        return 1;
    }

    // Auto-download: "-m auto" downloads the default Q4_K from HuggingFace
    if (model_path == "auto" || model_path == "default") {
        const char * home = getenv("HOME");
        std::string cache_dir = std::string(home ? home : "/tmp") + "/.cache/crispasr";
        std::string cached = cache_dir + "/voxtral-mini-3b-2507-q4_k.gguf";
        if (access(cached.c_str(), F_OK) == 0) {
            model_path = cached;
            if (!no_prints) fprintf(stderr, "using cached model: %s\n", cached.c_str());
        } else {
            fprintf(stderr, "downloading voxtral-mini-3b-2507-q4_k.gguf (~2.5 GB)...\n");
            std::string cmd = "mkdir -p '" + cache_dir + "' && "
                "hf download cstr/voxtral-mini-3b-2507-GGUF "
                "voxtral-mini-3b-2507-q4_k.gguf "
                "--local-dir '" + cache_dir + "' 2>&1";
            int rc = system(cmd.c_str());
            if (rc != 0) {
                fprintf(stderr, "download failed (rc=%d). Install hf CLI: pip install huggingface_hub\n", rc);
                return 1;
            }
            model_path = cached;
        }
    }

    std::vector<float> samples;
    if (!load_wav_16k_mono(audio_path, samples)) return 2;
    if (!no_prints)
        fprintf(stderr, "audio: %.2f s (%zu samples)\n", samples.size()/16000.0, samples.size());

    auto cp = voxtral_context_default_params(); cp.n_threads = n_threads;
    cp.verbosity = no_prints ? 0 : 1;
    auto * ctx = voxtral_init_from_file(model_path.c_str(), cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 3; }

    auto t0 = std::chrono::steady_clock::now();

    // Mel
    int n_mels = 0, T_mel = 0;
    float * mel = voxtral_compute_mel(ctx, samples.data(), (int)samples.size(), &n_mels, &T_mel);
    if (!mel) { fprintf(stderr, "mel failed\n"); voxtral_free(ctx); return 4; }
    auto t_mel = std::chrono::steady_clock::now();
    if (!no_prints)
        fprintf(stderr, "mel: %d × %d  (%.0f ms)\n", n_mels, T_mel,
                std::chrono::duration<double, std::milli>(t_mel - t0).count());

    // Encoder
    int N_enc = 0, pdim = 0;
    float * audio_embeds = voxtral_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) { fprintf(stderr, "encoder failed\n"); voxtral_free(ctx); return 5; }
    auto t_enc = std::chrono::steady_clock::now();
    if (!no_prints)
        fprintf(stderr, "encoder: %d frames × %d dim  (%.0f ms)\n", N_enc, pdim,
                std::chrono::duration<double, std::milli>(t_enc - t_mel).count());

    // Build prompt via the Tekken tokenizer — handles any language naturally.
    // Template: <s>[INST][BEGIN_AUDIO] <audio_pad>×N [/INST]lang:LANG[TRANSCRIBE]
    std::vector<int32_t> ids;
    {
        std::string prefix = "<s>[INST][BEGIN_AUDIO]";
        std::string suffix = "[/INST]lang:" + lang + "[TRANSCRIBE]";

        int n_prefix = 0, n_suffix = 0;
        int32_t * pid = voxtral_tokenize(ctx, prefix.c_str(), &n_prefix);
        int32_t * sid = voxtral_tokenize(ctx, suffix.c_str(), &n_suffix);

        ids.insert(ids.end(), pid, pid + n_prefix);
        for (int i = 0; i < N_enc; i++) ids.push_back(24);  // <audio_pad>
        ids.insert(ids.end(), sid, sid + n_suffix);

        free(pid);
        free(sid);
    }
    int T_prompt = (int)ids.size();
    if (!no_prints)
        fprintf(stderr, "prompt: %d tokens (incl. %d audio_pad)\n", T_prompt, N_enc);

    // Embed + splice
    float * text_embeds = voxtral_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { free(audio_embeds); voxtral_free(ctx); return 6; }
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == 24) {
            std::memcpy(text_embeds + (size_t)i * pdim,
                        audio_embeds + (size_t)spliced * pdim,
                        pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // KV cache + prefill
    if (!voxtral_kv_init(ctx, 4096)) { free(text_embeds); voxtral_free(ctx); return 7; }
    voxtral_kv_reset(ctx);
    auto t_pf0 = std::chrono::steady_clock::now();
    int n_t = 0, vocab = 0;
    float * logits = voxtral_run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_t, &vocab);
    auto t_pf1 = std::chrono::steady_clock::now();
    if (!logits) { free(text_embeds); voxtral_free(ctx); return 8; }
    free(text_embeds);
    if (!no_prints)
        fprintf(stderr, "prefill: %.0f ms\n",
                std::chrono::duration<double, std::milli>(t_pf1 - t_pf0).count());

    int next = 0; { float mx = -1e30f; for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; next = k; } }
    free(logits);

    // Greedy decode
    const int EOS = 2;
    std::vector<int32_t> gen; gen.push_back(next);
    auto t_dec0 = std::chrono::steady_clock::now();
    int n_past = T_prompt;
    while ((int)gen.size() < max_new && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = voxtral_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        float * lg = voxtral_run_llm_kv(ctx, tail, 1, n_past, nullptr, nullptr);
        free(tail); if (!lg) break;
        n_past++;
        int nx = 0; float mx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > mx) { mx = lg[k]; nx = k; }
        free(lg); gen.push_back(nx);
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
    if (!no_prints)
        fprintf(stderr, "decode: %zu tokens in %.0f ms (%.0f ms/token)\n",
                gen.size() - 1, dec_ms, dec_ms / std::max((size_t)1, gen.size() - 1));

    // Decode tokens to text
    std::string transcript;
    for (auto id : gen) {
        if (id == EOS) break;
        int len = 0;
        const uint8_t * bytes = voxtral_token_text(ctx, id, &len);
        if (bytes && len > 0) transcript.append((const char*)bytes, len);
    }

    auto t_total = std::chrono::steady_clock::now();
    if (!no_prints)
        fprintf(stderr, "total: %.0f ms\n",
                std::chrono::duration<double, std::milli>(t_total - t0).count());

    // ----- Timestamps via CTC aligner second pass -----
    if (timestamps) {
        auto t_align0 = std::chrono::steady_clock::now();

        canary_ctc_context_params acp = canary_ctc_context_default_params();
        acp.n_threads = n_threads;
        canary_ctc_context * actx = canary_ctc_init_from_file(aligner_path.c_str(), acp);
        if (!actx) { fprintf(stderr, "failed to load aligner model\n"); voxtral_free(ctx); return 9; }

        float * ctc_logits = nullptr;
        int T_ctc = 0, V_ctc = 0;
        int rc = canary_ctc_compute_logits(actx, samples.data(), (int)samples.size(),
                                           &ctc_logits, &T_ctc, &V_ctc);
        if (rc != 0) {
            fprintf(stderr, "CTC logits failed (rc=%d)\n", rc);
            canary_ctc_free(actx);
            voxtral_free(ctx);
            return 10;
        }

        auto words = tokenise_words(transcript);
        if (words.empty()) {
            if (!no_prints) fprintf(stderr, "no words to align\n");
            printf("%s\n", transcript.c_str());
        } else {
            std::vector<canary_ctc_word> out_words(words.size());
            std::vector<const char *> word_ptrs(words.size());
            for (size_t i = 0; i < words.size(); i++) word_ptrs[i] = words[i].c_str();

            rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc,
                                        word_ptrs.data(), (int)words.size(),
                                        out_words.data());
            auto t_align1 = std::chrono::steady_clock::now();
            if (!no_prints)
                fprintf(stderr, "align: %zu words in %.0f ms\n", words.size(),
                        std::chrono::duration<double, std::milli>(t_align1 - t_align0).count());

            if (rc != 0) {
                fprintf(stderr, "alignment failed (rc=%d), printing plain transcript\n", rc);
                printf("%s\n", transcript.c_str());
            } else if (out_srt) {
                for (size_t i = 0; i < out_words.size(); i++) {
                    printf("%zu\n%s --> %s\n%s\n\n",
                           i + 1,
                           format_time_srt(out_words[i].t0).c_str(),
                           format_time_srt(out_words[i].t1).c_str(),
                           out_words[i].text);
                }
            } else if (out_vtt) {
                printf("WEBVTT\n\n");
                for (size_t i = 0; i < out_words.size(); i++) {
                    printf("%s --> %s\n%s\n\n",
                           format_time_vtt(out_words[i].t0).c_str(),
                           format_time_vtt(out_words[i].t1).c_str(),
                           out_words[i].text);
                }
            } else {
                for (const auto & w : out_words) {
                    printf("[%8.2fs → %8.2fs]  %s\n",
                           w.t0 / 100.0, w.t1 / 100.0, w.text);
                }
            }
        }
        free(ctc_logits);
        canary_ctc_free(actx);
    } else {
        printf("%s\n", transcript.c_str());
    }

    voxtral_free(ctx);
    return 0;
}
