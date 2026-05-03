// crispasr_backend_gemma4_e2b.cpp — Gemma-4-E2B ASR backend adapter.

#include "crispasr_backend.h"
#include "gemma4_e2b.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class Gemma4E2BBackend : public CrispasrBackend {
public:
    Gemma4E2BBackend() = default;

    const char* name() const override { return "gemma4-e2b"; }

    uint32_t capabilities() const override {
        // Verified against src/gemma4_e2b.{h,cpp} as of v0.5.7:
        //   CAP_LANGUAGE_DETECT      framework LID pre-step (no native API)
        //   CAP_AUTO_DOWNLOAD        registry entry in src/crispasr_model_registry.cpp
        //   CAP_DIARIZE              framework post-step on segment list
        //   CAP_TIMESTAMPS_CTC       framework post-step via -am aligner.gguf
        //   CAP_FLASH_ATTN           uses ggml_flash_attn_ext in attention graph
        //   CAP_PARALLEL_PROCESSORS  shared session-level dispatcher
        //   CAP_TEMPERATURE          params.temperature → ctx->temperature → decode cfg
        //
        // Not yet declared (would need code changes elsewhere):
        //   CAP_TOKEN_CONFIDENCE — gemma4_e2b_transcribe_with_probs exists in
        //     the C-ABI but transcribe() below only calls the plain text variant
        //   CAP_BEAM_SEARCH      — not implemented in the gemma4_e2b decode loop
        //   CAP_TRANSLATE        — no source/target plumbing
        //   CAP_PUNCTUATION_TOGGLE — no toggle exposed
        return CAP_LANGUAGE_DETECT | CAP_AUTO_DOWNLOAD | CAP_DIARIZE | CAP_TIMESTAMPS_CTC | CAP_FLASH_ATTN |
               CAP_PARALLEL_PROCESSORS | CAP_TEMPERATURE;
    }

    bool init(const whisper_params& params) override {
        gemma4_e2b_context_params cp = gemma4_e2b_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || getenv("GEMMA4_E2B_BENCH"))
            cp.verbosity = 2;
        cp.use_gpu = params.use_gpu;
        // Honor -tp / --temperature so CAP_TEMPERATURE is real, not just a
        // declaration — gemma4_e2b.cpp already plumbs ctx->temperature to
        // the decode config in run_llm_decode (line 2115).
        cp.temperature = params.temperature;
        ctx_ = gemma4_e2b_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = gemma4_e2b_transcribe(ctx_, samples, n_samples);
        if (!text || !text[0]) {
            free(text);
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        free(text);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            gemma4_e2b_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~Gemma4E2BBackend() override { Gemma4E2BBackend::shutdown(); }

private:
    gemma4_e2b_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_gemma4_e2b_backend() {
    return std::make_unique<Gemma4E2BBackend>();
}
