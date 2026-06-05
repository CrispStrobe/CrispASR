// crispasr_backend_granite_nle.cpp — granite-speech-4.1-2b-nar backend adapter.
//
// Mirrors the structure of crispasr_backend_gemma4_e2b.cpp (simple end-to-end
// transcribe call) rather than crispasr_backend_granite.cpp (which builds the
// LLM prompt and runs a KV-cached greedy decode by hand). The NAR runtime
// has no autoregressive sampling — granite_nle_transcribe handles the full
// pipeline (mel → encoder → BPE-CTC → projector → single non-causal LLM
// forward → slot decode → detokenize) and returns the final UTF-8 transcript.

#include "crispasr_backend.h"
#include "granite_nle.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class GraniteNleBackend : public CrispasrBackend {
public:
    GraniteNleBackend() = default;

    const char* name() const override { return "granite-4.1-nar"; }

    uint32_t capabilities() const override {
        // granite-4.1-nar produces text segments → framework -am + --diarize
        // post-steps work even though the runtime is encoder+projector only
        // with no LLM decode features.
        return CAP_AUTO_DOWNLOAD | CAP_FLASH_ATTN | CAP_TIMESTAMPS_CTC | CAP_DIARIZE | CAP_PUNCTUATION_NATIVE |
               CAP_UNBOUNDED_INPUT;
    }

    bool init(const whisper_params& params) override {
        granite_nle_context_params cp = granite_nle_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        if (getenv("CRISPASR_VERBOSE") || getenv("GRANITE_NLE_BENCH"))
            cp.verbosity = 2;
        cp.use_gpu = params.use_gpu;
        ctx_ = granite_nle_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = granite_nle_transcribe(ctx_, samples, n_samples);
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
            granite_nle_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~GraniteNleBackend() override { GraniteNleBackend::shutdown(); }

private:
    granite_nle_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_granite_nle_backend() {
    return std::make_unique<GraniteNleBackend>();
}
