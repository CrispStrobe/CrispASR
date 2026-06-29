// crispasr_backend_ark_asr.cpp — AutoArk-AI/ARK-ASR-3B backend.
//
// Whisper-large-v3 encoder (partial RoPE) + MLP adapter + Qwen2.5-3B decoder
// with audio-token injection. Single self-contained GGUF (-m). See PLAN.md §ARK.

#include "ark_asr.h"
#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

class ArkAsrBackend : public CrispasrBackend {
public:
    ArkAsrBackend() = default;
    ~ArkAsrBackend() override { ArkAsrBackend::shutdown(); }

    const char* name() const override { return "ark-asr"; }

    uint32_t capabilities() const override {
        // CAP_UNBOUNDED_INPUT: the reference (modeling_arkasr.py) encodes the
        // whole clip in one RoPE encoder pass + one decode — no 30 s windowing.
        // Chunking into independent windows lets the promptless model re-detect
        // language per window (a German chunk gets translated to English), so we
        // default to full-audio like the reference. CAP_INTERNAL_CHUNKING makes
        // ark_asr_transcribe own the long-audio fallback (single-pass up to a
        // safety cap, else internal chunks) instead of crispasr_run's 30 s split.
        return CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT | CAP_INTERNAL_CHUNKING | CAP_FLASH_ATTN;
    }

    bool init(const whisper_params& params) override {
        auto cp = ark_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        cp.beam_size = params.beam_size > 0 ? params.beam_size : 1;

        ctx_ = ark_asr_init_from_file(params.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[ark-asr]: failed to load model '%s'\n", params.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        // EXPERIMENTAL language steering: ask > -l instruction > promptless.
        // mirrors crispasr_backend_mimo_asr.cpp.
        if (!params.ask.empty()) {
            ark_asr_set_ask(ctx_, params.ask.c_str());
        } else if (!params.language.empty() && params.language != "auto") {
            const std::string instr = "Transcribe the audio in " + crispasr_iso_to_english_lang(params.language) + ".";
            ark_asr_set_ask(ctx_, instr.c_str());
        } else {
            ark_asr_set_ask(ctx_, nullptr);
        }
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;
        char* text = ark_asr_transcribe(ctx_, samples, n_samples);
        if (text) {
            crispasr_segment seg;
            seg.text = text;
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)((int64_t)n_samples * 100 / 16000);
            out.push_back(std::move(seg));
            free(text);
        }
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            ark_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    ark_asr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_ark_asr_backend() {
    return std::make_unique<ArkAsrBackend>();
}
