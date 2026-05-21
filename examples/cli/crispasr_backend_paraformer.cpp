// crispasr_backend_paraformer.cpp — FunASR Paraformer (NAR-ASR) adapter.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "paraformer.h"
#include "whisper_params.h"

#include <cstdlib>
#include <cstring>
#include <vector>

class ParaformerBackend : public CrispasrBackend {
public:
    ParaformerBackend() = default;

    const char* name() const override { return "paraformer"; }

    uint32_t capabilities() const override {
        return CAP_AUTO_DOWNLOAD | CAP_FLASH_ATTN | CAP_PUNCTUATION_TOGGLE | CAP_DIARIZE;
    }

    bool init(const whisper_params& p) override {
        paraformer_context_params cp = paraformer_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.flash_attn = p.flash_attn;
        ctx_ = paraformer_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[paraformer]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = paraformer_transcribe(ctx_, samples, n_samples);
        if (!text) {
            fprintf(stderr, "crispasr[paraformer]: transcribe failed\n");
            return out;
        }

        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples / 16000.0 * 100.0);
        seg.text = text;
        std::free(text);

        while (!seg.text.empty() && (seg.text.front() == ' ' || seg.text.front() == '\n'))
            seg.text.erase(seg.text.begin());
        while (!seg.text.empty() && (seg.text.back() == ' ' || seg.text.back() == '\n'))
            seg.text.pop_back();

        if (!params.punctuation) {
            crispasr_strip_ascii_punctuation(seg.text);
            crispasr_lowercase_ascii(seg.text);
        }

        if (!seg.text.empty())
            out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            paraformer_free(ctx_);
            ctx_ = nullptr;
        }
    }

    ~ParaformerBackend() override { ParaformerBackend::shutdown(); }

private:
    paraformer_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_make_paraformer_backend() {
    return std::make_unique<ParaformerBackend>();
}
