// crispasr_backend_dolphin.cpp — CLI adapter for Dolphin (DataoceanAI), #436.
//
// -l accepts Dolphin's two-level language: "zh" (region predicted) or
// "zh-CN" / "zh-SICHUAN" (both forced). Without -l the decoder predicts both,
// as upstream does. Decoding is CTC prefix beam + attention rescoring, beam
// from -bs (default 10, upstream's).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "dolphin.h"

#include <cstdio>
#include <string>

namespace {

class DolphinBackend : public CrispasrBackend {
public:
    DolphinBackend() = default;
    ~DolphinBackend() override { DolphinBackend::shutdown(); }

    const char* name() const override { return "dolphin"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD | CAP_BEAM_SEARCH | CAP_PUNCTUATION_NATIVE; }

    bool init(const whisper_params& p) override {
        dolphin_context_params cp = dolphin_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (p.beam_size > 0)
            cp.beam_size = p.beam_size;
        ctx_ = dolphin_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[dolphin]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;
        std::string lang, region;
        if (!params.language.empty() && params.language != "auto") {
            const size_t dash = params.language.find('-');
            lang = params.language.substr(0, dash);
            if (dash != std::string::npos)
                region = params.language.substr(dash + 1);
        }
        dolphin_result* r = dolphin_transcribe_ex(ctx_, samples, n_samples, lang.empty() ? nullptr : lang.c_str(),
                                                  region.empty() ? nullptr : region.c_str());
        if (!r)
            return out;
        crispasr_segment seg;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)n_samples * 100 / 16000;
        seg.text = r->text;
        if (!params.no_prints)
            fprintf(stderr, "crispasr[dolphin]: language %s, region %s\n", r->language, r->region);
        out.push_back(std::move(seg));
        dolphin_result_free(r);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            dolphin_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    dolphin_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_dolphin_backend() {
    return std::unique_ptr<CrispasrBackend>(new DolphinBackend());
}
