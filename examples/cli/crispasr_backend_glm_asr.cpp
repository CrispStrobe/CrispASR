// crispasr_backend_glm_asr.cpp — GLM-ASR-Nano backend adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "glm_asr.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class GlmAsrBackend : public CrispasrBackend {
public:
    GlmAsrBackend() = default;

    const char* name() const override { return "glm-asr"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_TEMPERATURE | CAP_LANGUAGE_DETECT | CAP_AUTO_DOWNLOAD;
    }

    bool init(const whisper_params& params) override {
        glm_asr_context_params cp = glm_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;
        ctx_ = glm_asr_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = glm_asr_transcribe(ctx_, samples, n_samples);
        if (!text)
            return out;

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        free(text);

        // Trim leading/trailing whitespace
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
            glm_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~GlmAsrBackend() override { GlmAsrBackend::shutdown(); }

private:
    glm_asr_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_glm_asr_backend() {
    return std::make_unique<GlmAsrBackend>();
}
