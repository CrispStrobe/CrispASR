// crispasr_backend_fireredtts3.cpp — adapter for FireRedTeam/FireRedTTS3
// (Qwen3-1.7B LLM + DiT flow head over continuous RedAE latents, 24 kHz).
//
// Zero-shot ICL voice cloning: --voice ref.wav + --ref-text "<transcript>"
// (auto-transcribed via the shared resolver when --ref-text is absent).
// Without --voice, the default English prompt baked into the core GGUF is
// used. The RedAE + CAM++ companion GGUF is auto-discovered next to the
// model or resolved from the registry companion entry.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_tts_ref_text.h"
#include "crispasr_voice_provenance.h"
#include "whisper_params.h"

#include "core/audio_resample.h"
#include "core/wav_reader.h"
#include "fireredtts3_tts.h"

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

std::string discover_redae(const std::string& model_path) {
    auto sep = model_path.find_last_of("/\\");
    const std::string dir = (sep == std::string::npos) ? std::string(".") : model_path.substr(0, sep);
    static const char* candidates[] = {
        "fireredtts3-redae-f16.gguf",
        "fireredtts3-redae-q8_0.gguf",
        "fireredtts3-redae-q4_k.gguf",
        "fireredtts3-redae.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

// ISO code / lowercase name → FireRed language tag (upstream MULTI_LANG_TAGS).
std::string frt_language_tag(const std::string& lang) {
    struct Row {
        const char* iso;
        const char* tag;
    };
    static const Row rows[] = {
        {"zh", "Chinese"},    {"en", "English"},    {"yue", "Cantonese"}, {"ja", "Japanese"}, {"ko", "Korean"},
        {"es", "Spanish"},    {"fr", "French"},     {"ru", "Russian"},    {"ar", "Arabic"},   {"tr", "Turkish"},
        {"id", "Indonesian"}, {"pt", "Portuguese"}, {"it", "Italian"},    {"nl", "Dutch"},    {"vi", "Vietnamese"},
        {"de", "German"},     {"uk", "Ukrainian"},  {"th", "Thai"},       {"pl", "Polish"},   {"ro", "Romanian"},
        {"el", "Greek"},      {"cs", "Czech"},      {"fi", "Finnish"},    {"hi", "Hindi"},
    };
    if (lang.empty() || lang == "auto")
        return "";
    for (const auto& r : rows) {
        if (lang == r.iso)
            return r.tag;
        // full name, case-insensitive first letter tolerated
        std::string t = r.tag;
        std::string l = lang;
        if (!l.empty())
            l[0] = (char)toupper((unsigned char)l[0]);
        if (l == t)
            return t;
    }
    return "";
}

class Fireredtts3Backend : public CrispasrBackend {
public:
    Fireredtts3Backend() = default;
    ~Fireredtts3Backend() override { Fireredtts3Backend::shutdown(); }

    const char* name() const override { return "fireredtts3"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD; }

    int tts_sample_rate() const override { return ctx_ ? fireredtts3_tts_sample_rate(ctx_) : 24000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[fireredtts3]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        fireredtts3_tts_context_params cp = fireredtts3_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.seed = p.seed;
        if (p.tts_steps > 0)
            cp.n_timesteps = p.tts_steps;
        if (p.tts_cfg_scale > 0.0f)
            cp.cfg_scale = p.tts_cfg_scale;

        ctx_ = fireredtts3_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[fireredtts3]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // RedAE + CAM++ companion: --codec-model, sibling file, registry.
        std::string redae = p.tts_codec_model;
        if (!redae.empty() && redae != "auto" && redae != "default") {
            redae = crispasr_resolve_model_cli(redae, p.backend, p.no_prints, p.cache_dir, p.auto_download,
                                               p.tts_codec_quant);
        } else {
            redae.clear();
        }
        if (redae.empty())
            redae = discover_redae(p.model);
        if (redae.empty()) {
            CrispasrRegistryEntry entry;
            if (crispasr_registry_lookup(p.backend, entry, p.tts_codec_quant) && !entry.companion_filename.empty()) {
                redae = crispasr_resolve_model_cli(entry.companion_filename, p.backend, p.no_prints, p.cache_dir,
                                                   p.auto_download, p.tts_codec_quant);
            }
        }
        if (redae.empty() || fireredtts3_tts_set_redae_path(ctx_, redae.c_str()) != 0) {
            fprintf(stderr, "crispasr[fireredtts3]: RedAE companion GGUF not found/loadable — synthesis will fail "
                            "(expected fireredtts3-redae-f16.gguf beside the model)\n");
            return true; // model loaded; synthesis will error out clearly
        }
        if (!p.no_prints)
            fprintf(stderr, "crispasr[fireredtts3]: redae = '%s'\n", redae.c_str());

        // Voice cloning: --voice ref.wav (+ --ref-text or auto-transcribe).
        if (!p.tts_voice.empty()) {
            const std::string voice = p.tts_voice_dir.empty()
                                          ? p.tts_voice
                                          : crispasr_voice::resolve_voice_path(p.tts_voice, p.tts_voice_dir);
            std::vector<float> pcm;
            int sr = 0;
            if (!crispasr::core::read_wav_mono_pcm16(voice, pcm, sr) || pcm.empty()) {
                fprintf(stderr, "crispasr[fireredtts3]: cannot read --voice '%s' — ignoring\n", voice.c_str());
            } else {
                if (sr != 16000)
                    pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
                std::string ref_text = p.tts_ref_text;
                if (ref_text.empty())
                    ref_text =
                        crispasr_ref_text::resolve_cached(voice, pcm, p, "crispasr[fireredtts3]", ".fireredtts3-ref");
                if (ref_text.empty()) {
                    fprintf(stderr, "crispasr[fireredtts3]: no --ref-text and auto-transcription failed — "
                                    "falling back to the baked default prompt\n");
                } else if (fireredtts3_tts_set_voice_pcm(ctx_, pcm.data(), (int)pcm.size()) != 0 ||
                           fireredtts3_tts_set_ref_text(ctx_, ref_text.c_str()) != 0) {
                    fprintf(stderr, "crispasr[fireredtts3]: WARNING: failed to apply voice prompt '%s'\n",
                            voice.c_str());
                } else if (!p.no_prints) {
                    fprintf(stderr, "crispasr[fireredtts3]: voice = '%s'\n", voice.c_str());
                }
            }
        }

        // Language: FireRed tag from --language (warn on unsupported).
        const std::string tag = frt_language_tag(p.language);
        if (!tag.empty())
            fireredtts3_tts_set_language(ctx_, tag.c_str());
        else if (!p.language.empty() && p.language != "auto" && p.language != "en")
            fprintf(stderr,
                    "crispasr[fireredtts3]: language '%s' is not a FireRedTTS3 tag — using the prompt "
                    "default\n",
                    p.language.c_str());
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};
        if (params.seed != 0)
            fireredtts3_tts_set_seed(ctx_, params.seed);
        int n = 0;
        float* pcm = fireredtts3_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        fireredtts3_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            fireredtts3_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    fireredtts3_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_fireredtts3_backend() {
    return std::unique_ptr<CrispasrBackend>(new Fireredtts3Backend());
}
