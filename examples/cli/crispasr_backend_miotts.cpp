// crispasr_backend_miotts.cpp — CLI adapter for Aratako/MioTTS TTS.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "miotts.h"

#include <cstdio>
#include <string>
#include <vector>

class MioTtsBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto cp = miotts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        cp.use_gpu = p.use_gpu;
        cp.temperature = p.temperature;
        cp.max_tokens = 750;
        ctx_ = miotts_init_from_file(p.model.c_str(), cp);
        if (!ctx_)
            return false;
        if (!p.tts_voice.empty() && miotts_load_preset_embedding(ctx_, p.tts_voice.c_str()) != 0) {
            fprintf(stderr, "crispasr[miotts]: failed to load voice preset '%s'\n", p.tts_voice.c_str());
            shutdown();
            return false;
        }
        return true;
    }

    void shutdown() override {
        if (ctx_) {
            miotts_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "miotts"; }
    uint32_t capabilities() const override { return CAP_TTS; }
    int input_sample_rate() const override { return 16000; }
    int tts_sample_rate() const override { return miotts_get_sample_rate(ctx_); }

    std::vector<float> synthesize(const std::string& text, const whisper_params& /*p*/) override {
        if (!ctx_)
            return {};
        int n = 0;
        float* pcm = miotts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> result(pcm, pcm + n);
        miotts_free_audio(pcm);
        return result;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

private:
    miotts_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_create_miotts_backend() {
    return std::make_unique<MioTtsBackend>();
}
