// crispasr_backend_indextts.cpp -- adapter for IndexTTS-1.5 TTS.
//
// Two-GGUF runtime: the GPT AR model (loaded from --model) and the
// BigVGAN vocoder (loaded via --codec-model, or auto-discovered as
// a sibling file).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "core/audio_resample.h"
#include "core/wav_reader.h"

#include "indextts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

// Look for a sibling BigVGAN vocoder file next to the GPT model.
static std::string discover_vocoder(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "indextts-bigvgan.gguf",
        "indextts-bigvgan-f16.gguf",
        "indextts-bigvgan-q8_0.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class IndexttsBackend : public CrispasrBackend {
public:
    IndexttsBackend() = default;
    ~IndexttsBackend() override { IndexttsBackend::shutdown(); }

    const char* name() const override { return "indextts"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[indextts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        indextts_context_params cp = indextts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.temperature = p.temperature; // 0 = greedy argmax, >0 = sampling
        cp.seed = p.seed;
        ctx_ = indextts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[indextts]: failed to load GPT model '%s'\n", p.model.c_str());
            return false;
        }

        // BigVGAN vocoder
        std::string vocoder_path = p.tts_codec_model;
        if (!vocoder_path.empty() && vocoder_path != "auto" && vocoder_path != "default") {
            vocoder_path = crispasr_resolve_model_cli(vocoder_path, p.backend, p.no_prints, p.cache_dir,
                                                      p.auto_download, p.tts_codec_quant);
        } else {
            vocoder_path.clear();
        }
        if (vocoder_path.empty()) {
            vocoder_path = discover_vocoder(p.model);
        }
        if (vocoder_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                vocoder_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                          p.auto_download, p.tts_codec_quant);
            }
        }
        if (!vocoder_path.empty()) {
            indextts_set_vocoder_path(ctx_, vocoder_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[indextts]: vocoder path = '%s'\n", vocoder_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[indextts]: no BigVGAN vocoder found. Pass --codec-model PATH or place "
                            "indextts-bigvgan.gguf next to the GPT model.\n");
        }

        // Load reference voice if provided
        if (!p.tts_voice.empty()) {
            voice_path_ = p.tts_voice;
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }
        indextts_set_temperature(ctx_, params.temperature);
        indextts_set_seed(ctx_, params.seed);

        // Load reference audio if specified
        const float* ref_pcm = nullptr;
        int ref_n_samples = 0;
        std::vector<float> ref_audio;

        if (!voice_path_.empty()) {
            int sr = 0;
            if (crispasr::core::read_wav_mono_pcm16(voice_path_, ref_audio, sr)) {
                // IndexTTS API expects 24kHz mono float32 PCM. Resample if needed.
                if (sr != 24000 && sr > 0) {
                    ref_audio = core_audio::resample_polyphase(ref_audio.data(), (int)ref_audio.size(), sr, 24000);
                    if (!params.no_prints) {
                        fprintf(stderr, "crispasr[indextts]: resampled reference audio from %d Hz to 24000 Hz\n", sr);
                    }
                    sr = 24000;
                }
                ref_pcm = ref_audio.data();
                ref_n_samples = (int)ref_audio.size();
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr[indextts]: loaded reference audio '%s' (%d samples, %d Hz)\n",
                            voice_path_.c_str(), ref_n_samples, sr);
                }
            } else {
                fprintf(stderr, "crispasr[indextts]: failed to load reference audio '%s'\n", voice_path_.c_str());
            }
        }

        int n = 0;
        float* pcm = indextts_synthesize(ctx_, text.c_str(), ref_pcm, ref_n_samples, &n);
        if (!pcm || n <= 0) {
            fprintf(stderr, "crispasr[indextts]: synthesis failed\n");
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        indextts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            indextts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    indextts_context* ctx_ = nullptr;
    std::string voice_path_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_indextts_backend() {
    return std::unique_ptr<CrispasrBackend>(new IndexttsBackend());
}
