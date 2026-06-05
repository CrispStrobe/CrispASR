// crispasr_backend_csm.cpp -- adapter for Sesame CSM-1B TTS.
//
// Single-GGUF runtime: backbone (Llama-3.2 1B) + depth decoder
// (Llama-3.2 100M) + Mimi codec decoder, all in one file.
//
// Speaker conditioning: pass --voice <reference.wav> to clone a voice.
// The reference audio is encoded through Mimi and prepended as context.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "csm_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class CsmBackend : public CrispasrBackend {
public:
    CsmBackend() = default;
    ~CsmBackend() override { CsmBackend::shutdown(); }

    const char* name() const override { return "csm-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[csm-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        csm_tts_context_params cp = csm_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }
        cp.seed = p.seed;

        ctx_ = csm_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[csm-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        if (params.temperature > 0.0f)
            csm_tts_set_temperature(ctx_, params.temperature);
        csm_tts_set_seed(ctx_, params.seed);

        int n = 0;
        float* pcm = csm_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        csm_tts_pcm_free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return 24000; }

    void shutdown() override {
        if (ctx_) {
            csm_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    csm_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_csm_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new CsmBackend());
}
