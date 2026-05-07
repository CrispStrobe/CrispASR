// crispasr_backend_orpheus.cpp — adapter for canopylabs/orpheus-3b-0.1-ft
// (Llama-3.2-3B-Instruct talker + SNAC 24 kHz codec).
//
// Two-GGUF runtime: the talker LM (loaded from --model) and the SNAC
// codec (loaded via --codec-model, or auto-discovered as a sibling of
// the talker, or via the auto-download companion file).
//
// Speakers are baked-name strings (the Orpheus prompt convention is
// the literal `name: text` prefix); pass `--voice tara` etc. There is
// no embedding-table dispatch (in contrast to Qwen3-TTS-CustomVoice).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "orpheus.h"

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

// Look for a sibling SNAC codec file next to the talker.
static std::string discover_codec(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "snac-24khz.gguf",
        "snac_24khz.gguf",
        "orpheus-snac-24khz.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class OrpheusBackend : public CrispasrBackend {
public:
    OrpheusBackend() = default;
    ~OrpheusBackend() override { OrpheusBackend::shutdown(); }

    const char* name() const override { return "orpheus"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[orpheus]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        orpheus_context_params cp = orpheus_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // The CLI's global `--temperature` defaults to 0.0 (whisper-style
        // greedy ASR). For orpheus, greedy loops in a 7-slot pattern;
        // only override the orpheus default (0.6, engine_class.py) when
        // the user explicitly passes a non-zero value.
        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }
        ctx_ = orpheus_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[orpheus]: failed to load talker '%s'\n", p.model.c_str());
            return false;
        }

        // SNAC codec is required for synthesis. Without it, synthesize
        // returns nullptr; we still init the talker so callers using
        // orpheus_synthesize_codes (raw codec token IDs) can render via
        // the python reference SNAC decoder.
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
            orpheus_set_codec_path(ctx_, codec_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[orpheus]: codec path = '%s'\n", codec_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[orpheus]: no SNAC codec found. Pass --codec-model PATH or place "
                            "snac-24khz.gguf next to the talker. (synthesize() will return nullptr)\n");
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        if (!voice_loaded_) {
            std::string spk_name = params.tts_voice;
            if (orpheus_is_fixed_speaker(ctx_)) {
                if (spk_name.empty()) {
                    const char* first = orpheus_get_speaker_name(ctx_, 0);
                    if (!first) {
                        fprintf(stderr,
                                "crispasr[orpheus]: fixed_speaker variant has no speakers in the GGUF metadata\n");
                        return {};
                    }
                    spk_name = first;
                }
                if (orpheus_set_speaker_by_name(ctx_, spk_name.c_str()) != 0) {
                    fprintf(stderr, "crispasr[orpheus]: unknown speaker '%s' (available: ", spk_name.c_str());
                    int n = orpheus_n_speakers(ctx_);
                    for (int i = 0; i < n; i++) {
                        fprintf(stderr, "%s%s", i ? ", " : "", orpheus_get_speaker_name(ctx_, i));
                    }
                    fprintf(stderr, ")\n");
                    return {};
                }
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr[orpheus]: speaker = '%s'\n", spk_name.c_str());
                }
            }
            voice_loaded_ = true;
        }

        int n = 0;
        float* pcm = orpheus_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        orpheus_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            orpheus_free(ctx_);
            ctx_ = nullptr;
        }
        voice_loaded_ = false;
    }

private:
    orpheus_context* ctx_ = nullptr;
    bool voice_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_orpheus_backend() {
    return std::unique_ptr<CrispasrBackend>(new OrpheusBackend());
}
