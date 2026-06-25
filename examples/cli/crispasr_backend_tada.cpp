// crispasr_backend_tada.cpp — adapter for HumeAI/tada-3b-ml
// (Llama-3.2-3B + VibeVoiceDiffusionHead + TADA codec decoder).
//
// Two-GGUF runtime: the main model (LLM + FM head, loaded from --model)
// and the codec decoder (loaded via --codec-model or auto-discovered).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "tada_tts.h"

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

static std::string discover_codec(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "tada-codec.gguf",
        "tada-codec-f16.gguf",
        "tada-codec-q8_0.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class TadaBackend : public CrispasrBackend {
public:
    TadaBackend() = default;
    ~TadaBackend() override { TadaBackend::shutdown(); }

    const char* name() const override { return "tada"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override {
        fprintf(stderr, "crispasr[tada]: transcription not supported\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        tada_context_params cp = tada_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        if (p.temperature > 0.0f)
            cp.temperature = p.temperature;
        cp.seed = p.seed;

        if (cp.use_gpu && !p.no_prints)
            fprintf(stderr, "crispasr[tada]: note: GPU acceleration is not yet implemented "
                            "for TADA; running on CPU\n");

        ctx_ = tada_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[tada]: failed to load '%s'\n", p.model.c_str());
            return false;
        }

        // Codec discovery
        std::string codec_path = p.tts_codec_model;
        if (!codec_path.empty() && codec_path != "auto" && codec_path != "default") {
            codec_path = crispasr_resolve_model_cli(codec_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                    p.tts_codec_quant);
        } else {
            codec_path.clear();
        }
        if (codec_path.empty())
            codec_path = discover_codec(p.model);
        if (codec_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                codec_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                        p.auto_download, p.tts_codec_quant);
            }
        }
        if (!codec_path.empty()) {
            tada_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints)
                fprintf(stderr, "crispasr[tada]: codec = '%s'\n", codec_path.c_str());
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[tada]: no codec found. "
                            "Pass --codec-model PATH or place tada-codec.gguf next to model.\n");
        }
        std::string prompt_path;
        if (!p.tts_voice.empty() && p.tts_voice != "default" && p.tts_voice != "auto") {
            // Check for .wav — not yet supported; user needs a tada-ref.gguf
            const std::string& v = p.tts_voice;
            bool is_wav = v.size() >= 4 &&
                          (v.substr(v.size() - 4) == ".wav" || v.substr(v.size() - 4) == ".WAV");
            if (is_wav) {
                fprintf(stderr, "crispasr[tada]: --voice with a .wav file is not yet supported directly.\n"
                                "  Convert your audio to a reference GGUF first:\n"
                                "    python models/convert-tada-ref-to-gguf.py \\\n"
                                "      --audio your_voice.wav \\\n"
                                "      --transcript \"Exact words spoken in the audio.\" \\\n"
                                "      --output tada-ref-custom.gguf\n"
                                "  For non-English audio add: --language fr  (ar/ch/de/es/fr/it/ja/pl/pt)\n"
                                "  Then: --voice tada-ref-custom.gguf\n");
            } else {
                prompt_path = crispasr_resolve_model_cli(p.tts_voice, p.backend, p.no_prints, p.cache_dir,
                                                         p.auto_download, p.model_quant);
                if (prompt_path.empty() && !p.no_prints)
                    fprintf(stderr, "crispasr[tada]: --voice '%s' not found. "
                                    "Pass the path to a tada-ref.gguf file.\n",
                            p.tts_voice.c_str());
            }
        } else if (const char* env = getenv("TADA_PROMPT_CACHE"); env && *env) {
            prompt_path = env;
        }
        if (!prompt_path.empty()) {
            if (tada_load_prompt(ctx_, prompt_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[tada]: failed to load prompt from '%s'\n", prompt_path.c_str());
            }
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_)
            return {};
        if (!params.language.empty() && params.language != "auto" && params.language != "en" &&
            !params.no_prints) {
            fprintf(stderr, "crispasr[tada]: note: -l / --language has no effect on TADA. "
                            "To speak a language other than English, pass a reference audio in "
                            "that language via --voice tada-ref-<lang>.gguf\n");
        }
        if (params.temperature > 0.0f)
            tada_set_temperature(ctx_, params.temperature);
        if (params.seed > 0)
            tada_set_seed(ctx_, params.seed);

        int n_samples = 0;
        float* pcm = tada_synthesize(ctx_, text.c_str(), &n_samples);
        if (!pcm)
            return {};

        std::vector<float> out(pcm, pcm + n_samples);
        tada_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            tada_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    tada_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_tada_backend() {
    return std::unique_ptr<CrispasrBackend>(new TadaBackend());
}
