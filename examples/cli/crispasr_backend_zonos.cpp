// crispasr_backend_zonos.cpp -- adapter for Zyphra/Zonos-v0.1-transformer
// (AR transformer + DAC 44.1 kHz codec, Apache 2.0).
//
// Two-GGUF runtime: the AR transformer (--model) and the DAC 44 kHz
// decoder (--codec-model, or auto-discovered as sibling).
//
// Rich conditioning controls via CLI flags:
//   --pitch-std N      (0-400, default 20)
//   --speaking-rate N  (0-40, default 15)
//   --fmax N           (0-24000, default 22050)
//   --voice FILE.wav   (reference audio for speaker cloning)
//   --language CODE    (eSpeak language code, default en-us)

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "zonos_tts.h"

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

// Look for a sibling DAC codec file next to the Zonos model.
static std::string discover_codec(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "dac-44khz-f16.gguf",
        "dac-44khz.gguf",
        "dac_44khz.gguf",
        "zonos-dac-44khz.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class ZonosBackend : public CrispasrBackend {
public:
    ZonosBackend() = default;
    ~ZonosBackend() override { ZonosBackend::shutdown(); }

    const char* name() const override { return "zonos"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    int tts_sample_rate() const override { return 44100; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[zonos]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        zonos_tts_params zp = zonos_tts_default_params();
        zp.n_threads = p.n_threads;
        zp.verbosity = p.no_prints ? 0 : 1;
        zp.use_gpu = crispasr_backend_should_use_gpu(p);
        zp.flash_attn = p.flash_attn;
        zp.seed = p.seed;

        if (p.temperature > 0.0f) {
            zp.temperature = p.temperature;
        }

        ctx_ = zonos_tts_init_from_file(p.model.c_str(), zp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[zonos]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // DAC codec discovery
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty()) {
            codec_path = discover_codec(p.model);
        }
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (!codec_path.empty()) {
            zonos_tts_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[zonos]: codec path = '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[zonos]: no DAC codec found. Pass --codec-model PATH or place "
                            "dac-44khz-f16.gguf next to the model.\n");
        }

        // Set language if specified
        if (!p.language.empty()) {
            zonos_tts_set_language(ctx_, p.language.c_str());
        }

        // Load reference voice for speaker cloning
        if (!p.tts_voice.empty()) {
            // If it's a .wav file, extract speaker embedding
            const std::string& v = p.tts_voice;
            if (v.size() > 4 && (v.substr(v.size() - 4) == ".wav" || v.substr(v.size() - 4) == ".mp3" ||
                                 v.substr(v.size() - 5) == ".flac")) {
                if (zonos_tts_set_voice(ctx_, v.c_str()) != 0) {
                    fprintf(stderr, "crispasr[zonos]: warning: failed to load voice from '%s'\n", v.c_str());
                }
            }
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        if (params.temperature > 0.0f) {
            zonos_tts_set_temperature(ctx_, params.temperature);
        }
        zonos_tts_set_seed(ctx_, params.seed);

        int n = 0;
        float* pcm = zonos_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        zonos_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            zonos_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    zonos_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_zonos_backend() {
    return std::unique_ptr<CrispasrBackend>(new ZonosBackend());
}
