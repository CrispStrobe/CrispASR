// crispasr_backend_omniasr.cpp — OmniASR-CTC backend adapter.

#include "crispasr_backend.h"
#include "omniasr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class OmniasrBackend : public CrispasrBackend {
public:
    OmniasrBackend() = default;

    const char* name() const override { return "omniasr"; }
    uint32_t capabilities() const override { return 0; }

    bool init(const whisper_params& params) override {
        omniasr_context_params cp = omniasr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        // OMNIASR_DEBUG=1 enables verbose debug output
        if (getenv("OMNIASR_DEBUG"))
            cp.verbosity = 2;
        ctx_ = omniasr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        (void)t_offset_cs;
        (void)params;
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = omniasr_transcribe(ctx_, samples, n_samples);
        if (text) {
            crispasr_segment seg;
            seg.text = text;
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)(n_samples * 100 / 16000);
            out.push_back(std::move(seg));
            free(text);
        }
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            omniasr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    omniasr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_omniasr_backend() {
    return std::make_unique<OmniasrBackend>();
}
