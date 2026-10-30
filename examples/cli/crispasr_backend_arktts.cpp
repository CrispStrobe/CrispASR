// crispasr_backend_arktts.cpp — CLI adapter for Audio8-TTS (Arktts) backend.

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "arktts.h"
#include "core/wav_reader.h"

#include <cstdio>
#include <string>
#include <vector>

class ArkTtsBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto cp = arktts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        cp.use_gpu = p.use_gpu;
        cp.temperature = p.temperature;
        cp.seed = p.seed;
        cp.max_tokens = 512;
        ctx_ = arktts_init_from_file(p.model.c_str(), cp);
        if (!ctx_)
            return false;

        // Load reference audio if prompt wav provided
        if (!p.prompt.empty()) {
            std::vector<float> pcm;
            int sr = 0;
            if (crispasr::core::read_wav_mono_pcm16(p.prompt.c_str(), pcm, sr) && sr > 0) {
                arktts_set_reference(ctx_, pcm.data(), (int)pcm.size(), p.context.c_str());
            }
        }
        return true;
    }

    void shutdown() override {
        if (ctx_) {
            arktts_free(ctx_);
            ctx_ = nullptr;
        }
    }

    const char* name() const override { return "arktts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    int input_sample_rate() const override { return 16000; }

    int tts_sample_rate() const override { return arktts_get_sample_rate(ctx_); }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*p*/) override {
        return {};
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};
        arktts_set_temperature(ctx_, p.temperature);
        arktts_set_seed(ctx_, p.seed);
        if (p.max_new_tokens_explicit && p.max_new_tokens > 0) {
            arktts_set_max_new_tokens(ctx_, p.max_new_tokens);
        }
        int n = 0;
        float* pcm = arktts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> result(pcm, pcm + n);
        arktts_free_audio(pcm);
        return result;
    }

    ~ArkTtsBackend() override { shutdown(); }

private:
    struct arktts_context* ctx_ = nullptr;
};

std::unique_ptr<CrispasrBackend> crispasr_create_arktts_backend() {
    return std::make_unique<ArkTtsBackend>();
}
