// crispasr_backend_voxcpm2_tts.cpp — adapter for openbmb/VoxCPM2 TTS.
//
// VoxCPM2: tokenizer-free diffusion autoregressive TTS with 30-language
// support, voice cloning, and 48kHz output. Based on MiniCPM-4 (2B params).
// Apache 2.0 license.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "core/audio_resample.h"
#include "core/wav_reader.h"

#include "voxcpm2_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class VoxCPM2TTSBackend : public CrispasrBackend {
public:
    VoxCPM2TTSBackend() = default;
    ~VoxCPM2TTSBackend() override { VoxCPM2TTSBackend::shutdown(); }

    const char* name() const override { return "voxcpm2-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING | CAP_AUTO_DOWNLOAD; }

    int tts_sample_rate() const override { return 48000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[voxcpm2-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        voxcpm2_context_params cp = voxcpm2_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);

        // CFM inference steps (quality vs speed tradeoff)
        // Default 10; can be lowered to 3 for real-time on slow hardware
        const char* env_steps = getenv("VOXCPM2_INFERENCE_STEPS");
        if (env_steps)
            cp.inference_steps = atoi(env_steps);

        const char* env_cfg = getenv("VOXCPM2_CFG_VALUE");
        if (env_cfg)
            cp.cfg_value = (float)atof(env_cfg);

        const char* env_max = getenv("VOXCPM2_MAX_LEN");
        if (env_max)
            cp.max_len = atoi(env_max);

        cp.seed = (uint32_t)p.seed;

        ctx_ = voxcpm2_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[voxcpm2-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        if (!p.tts_voice.empty()) {
            voice_path_ = p.tts_voice;
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        voxcpm2_set_seed(ctx_, (uint32_t)params.seed);

        // Voice cloning path: load WAV, resample to 16 kHz mono float32, hand to
        // voxcpm2_synthesize_clone. The encoder pads internally to the patch
        // length, so any ref duration ≥ ~2 s should work.
        std::vector<float> ref_pcm;
        if (!voice_path_.empty()) {
            int sr = 0;
            if (crispasr::core::read_wav_mono_pcm16(voice_path_, ref_pcm, sr)) {
                if (sr != 16000 && sr > 0) {
                    ref_pcm = core_audio::resample_polyphase(ref_pcm.data(), (int)ref_pcm.size(), sr, 16000);
                    if (!params.no_prints) {
                        fprintf(stderr, "crispasr[voxcpm2-tts]: resampled reference '%s' from %d Hz to 16000 Hz\n",
                                voice_path_.c_str(), sr);
                    }
                }
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr[voxcpm2-tts]: loaded reference audio '%s' (%zu samples @ 16 kHz)\n",
                            voice_path_.c_str(), ref_pcm.size());
                }
            } else {
                fprintf(stderr,
                        "crispasr[voxcpm2-tts]: failed to load reference audio '%s' — falling back to zero-shot\n",
                        voice_path_.c_str());
            }
        }

        int n = 0;
        float* pcm = nullptr;
        if (!ref_pcm.empty()) {
            pcm = voxcpm2_synthesize_clone(ctx_, text.c_str(), ref_pcm.data(), (int)ref_pcm.size(), &n);
        } else {
            pcm = voxcpm2_synthesize(ctx_, text.c_str(), &n);
        }

        if (!pcm || n <= 0) {
            fprintf(stderr, "crispasr[voxcpm2-tts]: synthesis returned no audio\n");
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        voxcpm2_pcm_free(pcm);
        return out;
    }

    // VoxCPM2 outputs 48kHz natively (the CLI output writer picks up the
    // sample rate from the backend's synthesize() output length + duration).

    void shutdown() override {
        if (ctx_) {
            voxcpm2_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    struct voxcpm2_context* ctx_ = nullptr;
    std::string voice_path_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_voxcpm2_tts_backend() {
    return std::make_unique<VoxCPM2TTSBackend>();
}
