// crispasr_backend_dots_tts.cpp — adapter for rednote-hilab/dots.tts
// (Qwen2.5-1.5B LLM + 18L DiT flow-matching + BigVGAN vocoder, 48 kHz).
//
// Continuous AR TTS — no discrete codec tokens. BPE text input (no phonemes).
// Vocoder GGUF loaded separately (auto-discovered next to main model).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "dots_tts.h"

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

// Look for a sibling vocoder GGUF next to the main model.
static std::string discover_vocoder(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "dots-tts-soar-vocoder-f16.gguf", "dots-tts-soar-vocoder-q8_0.gguf", "dots-tts-soar-vocoder-q4_k.gguf",
        "dots-tts-vocoder-f16.gguf",      "dots-tts-vocoder.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

// Look for a sibling CAM++ speaker-encoder GGUF next to the main model.
static std::string discover_speaker(const std::string& model_path) {
    auto sep = model_path.find_last_of("/\\");
    const std::string dir = (sep == std::string::npos) ? std::string(".") : model_path.substr(0, sep);
    static const char* candidates[] = {
        "dots-tts-soar-spk-f16.gguf",
        "dots-tts-soar-spk.gguf",
        "dots-tts-spk-f16.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class DotsTtsBackend : public CrispasrBackend {
public:
    DotsTtsBackend() = default;
    ~DotsTtsBackend() override { DotsTtsBackend::shutdown(); }

    const char* name() const override { return "dots-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    int tts_sample_rate() const override { return 48000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[dots-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        dots_tts_context_params cp = dots_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.seed = p.seed;

        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }

        ctx_ = dots_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[dots-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Vocoder GGUF: check --codec-model, sibling file, then registry
        std::string voc_path = p.tts_codec_model;
        if (!voc_path.empty() && voc_path != "auto" && voc_path != "default") {
            voc_path = crispasr_resolve_model_cli(voc_path, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                                  p.tts_codec_quant);
        } else {
            voc_path.clear();
        }
        if (voc_path.empty()) {
            voc_path = discover_vocoder(p.model);
        }
        if (voc_path.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                voc_path = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                      p.auto_download, p.tts_codec_quant);
            }
        }
        if (!voc_path.empty()) {
            dots_tts_set_vocoder_path(ctx_, voc_path.c_str());
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[dots-tts]: vocoder = '%s'\n", voc_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[dots-tts]: WARNING: vocoder GGUF not found — synthesis will fail\n");
        }

        // Voice cloning (optional): --voice <ref.wav> conditions synthesis on a
        // CAM++ speaker embedding. Needs the speaker-encoder GGUF (sibling, or
        // the registry companion). Without --voice, synthesis is text-only.
        if (!p.tts_voice.empty()) {
            // Speaker-encoder GGUF: --codec-model override (if it names a spk
            // file), else the sibling dots-tts-soar-spk-*.gguf next to the core.
            std::string spk_path = discover_speaker(p.model);
            if (spk_path.empty()) {
                fprintf(stderr, "crispasr[dots-tts]: WARNING: --voice given but speaker encoder GGUF not found "
                                "(expected dots-tts-soar-spk-f16.gguf beside the model) — ignoring voice prompt\n");
            } else if (dots_tts_set_speaker_path(ctx_, spk_path.c_str()) != 0 ||
                       dots_tts_set_voice_prompt(ctx_, p.tts_voice.c_str()) != 0) {
                fprintf(stderr, "crispasr[dots-tts]: WARNING: failed to apply voice prompt '%s'\n",
                        p.tts_voice.c_str());
            } else if (!p.no_prints) {
                fprintf(stderr, "crispasr[dots-tts]: voice = '%s' (encoder '%s')\n", p.tts_voice.c_str(),
                        spk_path.c_str());
            }
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        if (params.temperature > 0.0f) {
            dots_tts_set_temperature(ctx_, params.temperature);
        }
        dots_tts_set_seed(ctx_, params.seed);

        // Per-call voice gating: the spoken AI-disclaimer is synthesized with a
        // cleared params.tts_voice → neutral voice. Re-enable for the cloned
        // (non-empty tts_voice) call. No-op when no reference voice was loaded.
        dots_tts_set_voice_enabled(ctx_, params.tts_voice.empty() ? 0 : 1);

        int n = 0;
        float* pcm = dots_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        dots_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            dots_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    dots_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_dots_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new DotsTtsBackend());
}
