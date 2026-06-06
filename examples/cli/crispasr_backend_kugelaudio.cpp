// crispasr_backend_kugelaudio.cpp — adapter for KugelAudio-0-Open TTS.
//
// KugelAudio is TTS-only: text → 24 kHz mono PCM.
// Architecture: Qwen2.5-7B + DiT diffusion head + acoustic VAE decoder.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "kugelaudio.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

class KugelAudioBackend : public CrispasrBackend {
public:
    KugelAudioBackend() = default;
    // cppcheck-suppress virtualCallInConstructor
    ~KugelAudioBackend() override { shutdown(); }

    const char* name() const override { return "kugelaudio"; }

    uint32_t capabilities() const override { return CAP_TTS; }

    bool init(const whisper_params& wparams) override {
        kugelaudio_context_params p = kugelaudio_context_default_params();
        p.n_threads = wparams.n_threads;
        p.verbosity = wparams.verbose ? 2 : 1;
        p.use_gpu = wparams.use_gpu;
        p.flash_attn = wparams.flash_attn;

        if (wparams.tts_steps > 0)
            p.tts_steps = wparams.tts_steps;
        if (wparams.tts_cfg_scale > 0.0f)
            p.cfg_scale = wparams.tts_cfg_scale;
        if (wparams.seed != 0)
            p.seed = wparams.seed;

        kctx_ = kugelaudio_init_from_file(wparams.model.c_str(), p);
        return kctx_ != nullptr;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        crispasr_segment seg;
        seg.text = "[kugelaudio is TTS-only — use --tts for synthesis]";
        seg.t0 = 0;
        seg.t1 = 0;
        return {seg};
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& wparams) override {
        if (!kctx_)
            return {};

        if (wparams.seed != 0)
            kugelaudio_set_seed(kctx_, wparams.seed);
        if (wparams.tts_steps > 0)
            kugelaudio_set_tts_steps(kctx_, wparams.tts_steps);
        if (wparams.tts_cfg_scale > 0.0f)
            kugelaudio_set_cfg_scale(kctx_, wparams.tts_cfg_scale);

        int n_samples = 0;
        float* pcm = kugelaudio_synthesize(kctx_, text.c_str(), &n_samples);
        if (!pcm || n_samples <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n_samples);
        free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return 24000; }

    void shutdown() override {
        if (kctx_) {
            kugelaudio_free(kctx_);
            kctx_ = nullptr;
        }
    }

private:
    kugelaudio_context* kctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_kugelaudio_backend() {
    return std::make_unique<KugelAudioBackend>();
}
