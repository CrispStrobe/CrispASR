// crispasr_backend_kyutai_stt.cpp — Kyutai STT backend adapter.

#include "crispasr_backend.h"
#include "kyutai_stt.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

class KyutaiSttBackend : public CrispasrBackend {
public:
    KyutaiSttBackend() = default;

    const char* name() const override { return "kyutai-stt"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD; }

    bool init(const whisper_params& params) override {
        kyutai_stt_context_params cp = kyutai_stt_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = params.use_gpu && params.gpu_backend != "cpu";
        ctx_ = kyutai_stt_init_from_file(params.model.c_str(), cp);
        return ctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = kyutai_stt_transcribe(ctx_, samples, n_samples);
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
            kyutai_stt_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~KyutaiSttBackend() override { KyutaiSttBackend::shutdown(); }

private:
    kyutai_stt_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_kyutai_stt_backend() {
    return std::make_unique<KyutaiSttBackend>();
}
