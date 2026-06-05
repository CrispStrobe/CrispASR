// crispasr_backend_outetts.cpp -- adapter for OuteAI/OuteTTS-0.3-1B
// (OLMo-1B talker + WavTokenizer 24 kHz codec).
//
// Two-GGUF runtime: the talker LM (--model) and the WavTokenizer
// decoder (--codec-model, or auto-discovered as sibling, or via
// the auto-download companion file).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "outetts.h"

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

// Look for a sibling WavTokenizer codec file next to the talker.
static std::string discover_codec(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "wavtokenizer-decoder-f16.gguf",
        "wavtokenizer-decoder.gguf",
        "wavtokenizer.gguf",
        "wavtok-decoder.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class OuteTTSBackend : public CrispasrBackend {
public:
    OuteTTSBackend() = default;
    ~OuteTTSBackend() override { OuteTTSBackend::shutdown(); }

    const char* name() const override { return "outetts"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    int tts_sample_rate() const override { return 24000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[outetts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        outetts_context_params cp = outetts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }
        cp.seed = p.seed;
        ctx_ = outetts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[outetts]: failed to load talker '%s'\n", p.model.c_str());
            return false;
        }

        // WavTokenizer codec
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
            outetts_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[outetts]: codec path = '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[outetts]: no WavTokenizer codec found. Pass --codec-model PATH or place "
                            "wavtokenizer-decoder-f16.gguf next to the talker.\n");
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }
        if (params.temperature > 0.0f)
            outetts_set_temperature(ctx_, params.temperature);
        outetts_set_seed(ctx_, params.seed);

        // Load speaker profile if --voice points to a .json file
        if (!params.tts_voice.empty() && params.tts_voice != last_voice_key_) {
            if (params.tts_voice.size() > 5 && params.tts_voice.substr(params.tts_voice.size() - 5) == ".json") {
                if (outetts_load_speaker(ctx_, params.tts_voice.c_str()) == 0) {
                    last_voice_key_ = params.tts_voice;
                } else {
                    fprintf(stderr, "crispasr[outetts]: failed to load speaker from '%s'\n", params.tts_voice.c_str());
                }
            }
        }

        int n = 0;
        float* pcm = outetts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        outetts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            outetts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    outetts_context* ctx_ = nullptr;
    std::string last_voice_key_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_outetts_backend() {
    return std::unique_ptr<CrispasrBackend>(new OuteTTSBackend());
}
