#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "sidon.h"
#include "whisper_params.h"

#include <cstdio>

class SidonBackend final : public CrispasrBackend {
public:
    const char* name() const override { return "sidon"; }
    uint32_t capabilities() const override { return CAP_S2S; }
    int tts_sample_rate() const override { return 48000; }

    bool init(const whisper_params& p) override {
        sidon_context_params cp = sidon_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        ctx_ = sidon_init_from_file(p.model.c_str(), cp);
        if (!ctx_)
            std::fprintf(stderr, "crispasr[sidon]: failed to load '%s'\n", p.model.c_str());
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

    std::vector<float> speech_to_speech(const float* samples, int n_samples, std::string* out_text,
                                        const whisper_params&) override {
        if (out_text)
            out_text->clear();
        return sidon_restore(ctx_, samples, n_samples);
    }

    void shutdown() override {
        if (ctx_) {
            sidon_free(ctx_);
            ctx_ = nullptr;
        }
    }
    ~SidonBackend() override { shutdown(); }

private:
    sidon_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_sidon_backend() {
    return std::make_unique<SidonBackend>();
}
