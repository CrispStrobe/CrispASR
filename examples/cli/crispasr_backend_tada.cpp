// crispasr_backend_tada.cpp — adapter for HumeAI/tada-3b-ml
// (Llama-3.2-3B + VibeVoiceDiffusionHead + TADA codec decoder).
//
// Two-GGUF runtime: the main model (LLM + FM head, loaded from --model)
// and the codec decoder (loaded via --codec-model or auto-discovered).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_cache.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "tada_tts.h"

#include <cctype>
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

static std::string tada_prompt_lang_suffix(std::string lang) {
    for (char& c : lang)
        c = (char)std::tolower((unsigned char)c);
    if (lang == "auto" || lang == "en" || lang == "eng")
        return "";
    if (lang == "zh" || lang == "zh-cn" || lang == "cn")
        return "ch";
    if (lang == "ar" || lang == "ch" || lang == "de" || lang == "es" || lang == "fr" || lang == "it" || lang == "ja" ||
        lang == "pl" || lang == "pt")
        return lang;
    return "";
}

static std::string discover_prompt(const std::string& model_path, const std::string& language) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    std::vector<std::string> candidates;
    const std::string lang = tada_prompt_lang_suffix(language);
    if (!lang.empty())
        candidates.push_back("tada-ref-" + lang + ".gguf");
    candidates.push_back("tada-ref.gguf");

    for (const std::string& name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

static std::string tada_lang_ref_url(const std::string& backend, const std::string& lang) {
    const std::string base = (backend == "tada-1b" || backend == "tada-tts-1b")
                                 ? "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/"
                                 : "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/";
    return base + "tada-ref-" + lang + ".gguf";
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

        // TADA's per-token flow-matching duration head is noise-sensitive: a
        // single unlucky noise draw can collapse token durations (rushed,
        // unintelligible speech). Generate several candidates per step and keep
        // the best by reconstruction score (Python num_acoustic_candidates).
        // Q4_K quantization amplifies the noise lottery — 4 candidates aren't
        // enough to reliably find a good sample (#192 "four hours" truncation).
        // 8 candidates make Q4_K produce correct output across seeds.
        // Override with TADA_NUM_CANDIDATES (1 = fastest/least robust).
        cp.num_acoustic_candidates = 8;
        if (const char* env = std::getenv("TADA_NUM_CANDIDATES"); env && *env) {
            int n = atoi(env);
            if (n >= 1)
                cp.num_acoustic_candidates = n;
        }

        // Talker text decoder: greedy AR with no repetition control loops and
        // hallucinates (#197). Sample by default with upstream InferenceOptions
        // values (do_sample=True, temp=0.6, top_k=0, top_p=0.9, rep_penalty=1.1).
        // Override via env vars. --temperature (above) still sets cp.temperature.
        cp.text_do_sample = true;
        if (const char* e = std::getenv("TADA_DO_SAMPLE"); e && *e)
            cp.text_do_sample = !(e[0] == '0' || e[0] == 'f' || e[0] == 'F' || e[0] == 'n' || e[0] == 'N');
        if (const char* e = std::getenv("TADA_TEMPERATURE"); e && *e)
            cp.temperature = (float)atof(e);
        if (const char* e = std::getenv("TADA_TOP_P"); e && *e)
            cp.text_top_p = (float)atof(e);
        if (const char* e = std::getenv("TADA_TOP_K"); e && *e)
            cp.text_top_k = atoi(e);
        if (const char* e = std::getenv("TADA_REPETITION_PENALTY"); e && *e)
            cp.text_repetition_penalty = (float)atof(e);

        // Acoustic flow-matching knobs (#197): the "quick and dirty" vs "slow and
        // accurate" axis. num_fm_steps is the primary quality lever (more ODE
        // steps = slower, higher fidelity). Defaults match upstream
        // InferenceOptions (10 / 1.6 / 0.9); override via env or per request.
        if (const char* e = std::getenv("TADA_NUM_FM_STEPS"); e && *e) {
            int n = atoi(e);
            if (n > 0)
                cp.num_fm_steps = n;
        }
        if (const char* e = std::getenv("TADA_ACOUSTIC_CFG"); e && *e)
            cp.acoustic_cfg = (float)atof(e);
        if (const char* e = std::getenv("TADA_NOISE_TEMP"); e && *e)
            cp.noise_temp = (float)atof(e);

        // Remember the resolved sampler defaults so a server can override them
        // per request and have unspecified knobs fall back here (rather than
        // leaking a previous request's value through the shared context).
        def_temperature_ = cp.temperature;
        def_top_p_ = cp.text_top_p;
        def_top_k_ = cp.text_top_k;
        def_rep_penalty_ = cp.text_repetition_penalty;
        def_do_sample_ = cp.text_do_sample;
        def_num_candidates_ = cp.num_acoustic_candidates;
        def_num_fm_steps_ = cp.num_fm_steps; // 0 → synth uses the 10-step default
        def_acoustic_cfg_ = cp.acoustic_cfg;
        def_noise_temp_ = cp.noise_temp;

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
            bool is_wav = v.size() >= 4 && (v.substr(v.size() - 4) == ".wav" || v.substr(v.size() - 4) == ".WAV");
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
                    fprintf(stderr,
                            "crispasr[tada]: --voice '%s' not found. "
                            "Pass the path to a tada-ref.gguf file.\n",
                            p.tts_voice.c_str());
            }
        } else if (const char* env = getenv("TADA_PROMPT_CACHE"); env && *env) {
            prompt_path = env;
        } else {
            prompt_path = discover_prompt(p.model, p.language);
            // Language-specific ref: check cache dir, then auto-download.
            // discover_prompt already prefers tada-ref-<lang>.gguf next to the model
            // file; this fallback finds refs that landed in the shared cache dir
            // (e.g. from a previous --auto-download run) and downloads missing ones.
            if (prompt_path.empty()) {
                const std::string lang = tada_prompt_lang_suffix(p.language);
                if (!lang.empty()) {
                    const std::string ref_name = "tada-ref-" + lang + ".gguf";
                    const std::string cached = crispasr_cache::dir(p.cache_dir) + "/" + ref_name;
                    if (crispasr_cache::file_present(cached)) {
                        prompt_path = cached;
                    } else if (p.auto_download) {
                        prompt_path = crispasr_cache::ensure_cached_file(ref_name, tada_lang_ref_url(p.backend, lang),
                                                                         p.no_prints, "crispasr", p.cache_dir);
                    } else if (!p.no_prints) {
                        fprintf(stderr,
                                "crispasr[tada]: no voice reference for language '%s' found. "
                                "Add --auto-download to fetch it, or pass --voice tada-ref-%s.gguf.\n",
                                p.language.c_str(), lang.c_str());
                    }
                }
            }
        }
        if (!prompt_path.empty()) {
            if (tada_load_prompt(ctx_, prompt_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[tada]: failed to load prompt from '%s'\n", prompt_path.c_str());
            } else {
                // Remember which voice is loaded so per-request switching (#201)
                // can skip a reload when the request asks for the same one. An
                // explicit --voice is keyed by its request string; a default /
                // discovered prompt is keyed empty ("keep current").
                const bool explicit_voice = !p.tts_voice.empty() && p.tts_voice != "default" && p.tts_voice != "auto";
                last_voice_key_ = explicit_voice ? p.tts_voice : "";
                if (!p.no_prints && !explicit_voice)
                    fprintf(stderr, "crispasr[tada]: using default voice prompt '%s'\n", prompt_path.c_str());
            }
        }
        return true;
    }

    // Switch the voice reference for this request (#201). Mirrors chatterbox's
    // per-call voice key: reload the prompt only when the request names a
    // different voice than the one currently loaded, so a long-running server
    // can swap voices without a restart. Returns false (and keeps the current
    // voice) when the requested ref can't be resolved.
    bool apply_request_voice(const whisper_params& p) {
        const std::string& v = p.tts_voice;
        if (v.empty() || v == "default" || v == "auto" || v == last_voice_key_)
            return true; // keep whatever is loaded
        const bool is_wav = v.size() >= 4 && (v.substr(v.size() - 4) == ".wav" || v.substr(v.size() - 4) == ".WAV");
        if (is_wav) {
            fprintf(stderr,
                    "crispasr[tada]: voice '%s' is a .wav — direct .wav voice cloning is not yet "
                    "supported at query time. Convert it to a tada-ref.gguf first "
                    "(models/convert-tada-ref-to-gguf.py or the CLI --make-ref pipeline) and pass that.\n",
                    v.c_str());
            return false;
        }
        std::string ref =
            crispasr_resolve_model_cli(v, p.backend, p.no_prints, p.cache_dir, p.auto_download, p.model_quant);
        if (ref.empty()) {
            fprintf(stderr, "crispasr[tada]: voice '%s' not found; keeping current voice.\n", v.c_str());
            return false;
        }
        if (tada_load_prompt(ctx_, ref.c_str()) != 0) {
            fprintf(stderr, "crispasr[tada]: failed to load voice '%s'; keeping current voice.\n", ref.c_str());
            return false;
        }
        last_voice_key_ = v;
        if (!p.no_prints)
            fprintf(stderr, "crispasr[tada]: switched voice → '%s'\n", ref.c_str());
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_)
            return {};
        // Switch the voice reference per request if a different one is named (#201)
        // — a long-running server can change voice without a restart.
        apply_request_voice(params);
        // Apply the talker sampler per request so a long-running server can tune
        // temperature / top-p / top-k / repetition-penalty at query time without a
        // restart (#197). The context is shared across requests, so every knob is
        // set on every call: a request value when provided (whisper_params uses
        // negative sentinels for "unset"), otherwise the init-time default — this
        // keeps requests isolated instead of leaking the previous request's value.
        tada_set_temperature(ctx_, params.temperature > 0.0f ? params.temperature : def_temperature_);
        tada_set_top_p(ctx_, params.tts_top_p >= 0.0f ? params.tts_top_p : def_top_p_);
        tada_set_top_k(ctx_, params.tts_top_k >= 0 ? params.tts_top_k : def_top_k_);
        tada_set_repetition_penalty(ctx_, params.tts_repetition_penalty > 0.0f ? params.tts_repetition_penalty
                                                                               : def_rep_penalty_);
        tada_set_num_candidates(ctx_, params.tts_num_candidates >= 1 ? params.tts_num_candidates : def_num_candidates_);
        tada_set_do_sample(ctx_, params.tts_do_sample >= 0 ? (params.tts_do_sample != 0) : def_do_sample_);
        // Acoustic-FM knobs (#197): same per-request-or-default isolation. These
        // reuse the cross-backend "num_steps"/"cfg_scale" HTTP fields (FM ODE steps
        // / CFG scale) plus a tada-specific noise_temp.
        tada_set_num_fm_steps(ctx_, params.tts_num_steps > 0 ? params.tts_num_steps : def_num_fm_steps_);
        tada_set_acoustic_cfg(ctx_, params.tts_cfg_scale >= 0.0f ? params.tts_cfg_scale : def_acoustic_cfg_);
        tada_set_noise_temp(ctx_, params.tts_noise_temp >= 0.0f ? params.tts_noise_temp : def_noise_temp_);
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
    // Resolved sampler defaults (from defaults + env), used when a request omits a knob.
    float def_temperature_ = 0.6f;
    float def_top_p_ = 0.9f;
    int def_top_k_ = 0;
    float def_rep_penalty_ = 1.1f;
    bool def_do_sample_ = true;
    int def_num_candidates_ = 4;
    // Resolved acoustic-FM defaults ("slow vs fast" axis, #197).
    int def_num_fm_steps_ = 10;
    float def_acoustic_cfg_ = 1.6f;
    float def_noise_temp_ = 0.9f;
    // Currently-loaded voice reference, so a server can switch voices per request
    // without a restart (#201). Reload only when the request names a different
    // voice. Empty/"default"/"auto" mean "keep whatever is loaded".
    std::string last_voice_key_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_tada_backend() {
    return std::unique_ptr<CrispasrBackend>(new TadaBackend());
}
