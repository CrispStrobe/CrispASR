// crispasr_backend_bark.cpp -- adapter for Suno Bark TTS (MIT).
//
// 3-stage hierarchical TTS: text -> semantic -> coarse -> fine -> EnCodec PCM.
// All sub-models packed into a single GGUF (convert-bark-to-gguf.py).
// Speaker conditioning via .npz prompt files (--voice path.npz).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "bark_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class BarkBackend : public CrispasrBackend {
public:
    BarkBackend() = default;
    ~BarkBackend() override { BarkBackend::shutdown(); }

    const char* name() const override { return "bark"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_VOICE_CLONING; }

    int tts_sample_rate() const override { return 24000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[bark]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        bark_context_params cp = bark_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (p.temperature > 0.0f) {
            cp.temperature_semantic = p.temperature;
            cp.temperature_coarse = p.temperature;
            cp.temperature_fine = std::min(p.temperature, 0.5f);
        }
        cp.seed = p.seed;
        cp.flash_attn = p.flash_attn;

        ctx_ = bark_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[bark]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Load speaker prompt if specified
        if (!p.tts_voice.empty()) {
            load_speaker(p.tts_voice);
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        if (params.temperature > 0.0f) {
            bark_set_temperature_semantic(ctx_, params.temperature);
            bark_set_temperature_coarse(ctx_, params.temperature);
            bark_set_temperature_fine(ctx_, std::min(params.temperature, 0.5f));
        }
        bark_set_seed(ctx_, params.seed);

        // Update speaker if changed
        if (!params.tts_voice.empty() && params.tts_voice != last_voice_) {
            load_speaker(params.tts_voice);
        }

        int n = 0;
        float* pcm = bark_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n);
        bark_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            bark_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    bark_context* ctx_ = nullptr;
    std::string last_voice_;

    void load_speaker(const std::string& voice) {
        if (voice.empty()) {
            bark_clear_speaker(ctx_);
            last_voice_.clear();
            return;
        }

        // Check if it's an .npz file path
        if (voice.size() > 4 && voice.substr(voice.size() - 4) == ".npz") {
            if (bark_set_speaker_npz(ctx_, voice.c_str()) == 0) {
                last_voice_ = voice;
            } else {
                fprintf(stderr, "crispasr[bark]: failed to load speaker from '%s'\n", voice.c_str());
            }
        } else {
            // Could be a preset name like "v2/de_speaker_0"
            // Bark presets are .npz files bundled with the bark package;
            // for now we treat the voice as a direct path.
            fprintf(stderr, "crispasr[bark]: speaker presets not yet supported, "
                            "use --voice <path>.npz\n");
        }
    }
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_bark_backend() {
    return std::unique_ptr<CrispasrBackend>(new BarkBackend());
}
