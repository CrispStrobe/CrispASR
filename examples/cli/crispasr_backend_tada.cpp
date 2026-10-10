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

// crispasr_audio_load_at_rate lives in the shared lib (crispasr_audio.cpp);
// forward-declare it so the #201 .wav-clone path can decode a reference clip
// straight to 24 kHz without pulling in the audio header.
extern "C" int crispasr_audio_load_at_rate(const char* path, int target_rate, float** out_pcm, int* out_samples,
                                           int* out_sample_rate);

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "core/crispasr_env.h"

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

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_VOICE_CLONING; }

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

        // num_acoustic_candidates INHERITS the library default (1) from
        // tada_context_default_params() above — do NOT re-hardcode it. Upstream
        // InferenceOptions default is 1; the reconstruction ("likelihood") scorer
        // used to rank >1 candidates scores acoustic dims only and can PREFER a
        // duration outlier, so best-of-N mangled "…four hours" → "…and forth"
        // (#192, N=4). A redundant override here is exactly how that 4 (and a
        // parallel session's 8) shipped; inheriting the tested library default
        // keeps one source of truth. Opt in to >1 with TADA_NUM_CANDIDATES.
        if (const char* env = crispasr_env::get("CRISPASR_TADA_NUM_CANDIDATES"); env && *env) {
            int n = atoi(env);
            if (n >= 1)
                cp.num_acoustic_candidates = n;
        }

        // Talker text decoder: greedy AR with no repetition control loops and
        // hallucinates (#197). Sample by default with upstream InferenceOptions
        // values (do_sample=True, temp=0.6, top_k=0, top_p=0.9, rep_penalty=1.1).
        // Override via env vars. --temperature (above) still sets cp.temperature.
        cp.text_do_sample = true;
        if (const char* e = crispasr_env::get("CRISPASR_TADA_DO_SAMPLE"); e && *e)
            cp.text_do_sample = !(e[0] == '0' || e[0] == 'f' || e[0] == 'F' || e[0] == 'n' || e[0] == 'N');
        if (const char* e = crispasr_env::get("CRISPASR_TADA_TEMPERATURE"); e && *e)
            cp.temperature = (float)atof(e);
        if (const char* e = crispasr_env::get("CRISPASR_TADA_TOP_P"); e && *e)
            cp.text_top_p = (float)atof(e);
        if (const char* e = crispasr_env::get("CRISPASR_TADA_TOP_K"); e && *e)
            cp.text_top_k = atoi(e);
        if (const char* e = crispasr_env::get("CRISPASR_TADA_REPETITION_PENALTY"); e && *e)
            cp.text_repetition_penalty = (float)atof(e);

        // Acoustic flow-matching knobs (#197): the "quick and dirty" vs "slow and
        // accurate" axis. num_fm_steps is the primary quality lever (more ODE
        // steps = slower, higher fidelity). Defaults match upstream
        // InferenceOptions (10 / 1.6 / 0.9); override via env or per request.
        if (const char* e = crispasr_env::get("CRISPASR_TADA_NUM_FM_STEPS"); e && *e) {
            int n = atoi(e);
            if (n > 0)
                cp.num_fm_steps = n;
        }
        if (const char* e = crispasr_env::get("CRISPASR_TADA_ACOUSTIC_CFG"); e && *e)
            cp.acoustic_cfg = (float)atof(e);
        if (const char* e = crispasr_env::get("CRISPASR_TADA_NOISE_TEMP"); e && *e)
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
        // In --make-ref / --align mode the --voice <wav> is the input to the
        // aligner pipeline, not a synth voice prompt — the make-ref/align handler
        // (crispasr_run.cpp) consumes it, so don't load it or fail on the .wav.
        if (!p.make_ref && !p.align && !p.tts_voice.empty() && p.tts_voice != "default" && p.tts_voice != "auto") {
            // Check for .wav — not yet supported; user needs a tada-ref.gguf
            const std::string& v = p.tts_voice;
            bool is_wav = v.size() >= 4 && (v.substr(v.size() - 4) == ".wav" || v.substr(v.size() - 4) == ".WAV");
            if (is_wav) {
                // #201: opt-in on-the-fly clone (CRISPASR_TADA_WAV_CLONE=1). On
                // success the prompt is set in-memory, so prompt_path stays empty
                // and the load block below is skipped. Off/failed → fail loudly
                // (clone_from_wav prints how to enable / build a tada-ref.gguf).
                if (!clone_from_wav(p))
                    return false;
            } else {
                prompt_path = crispasr_resolve_model_cli(p.tts_voice, p.backend, p.no_prints, p.cache_dir,
                                                         p.auto_download, p.model_quant);
                if (prompt_path.empty()) {
                    fprintf(stderr,
                            "crispasr[tada]: --voice '%s' could not be resolved. Pass the path to a "
                            "tada-ref.gguf (or build one from a .wav with --make-ref).\n",
                            p.tts_voice.c_str());
                    return false; // explicit voice couldn't be honored — fail loudly, don't use default
                }
            }
        } else if (const char* env = crispasr_env::get("CRISPASR_TADA_PROMPT_CACHE"); env && *env) {
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
                        prompt_path = crispasr_managed_download(
                            ref_name, tada_lang_ref_url(p.backend, lang),
                            "CC-BY-4.0 (FLEURS voice reference; see https://huggingface.co/datasets/google/fleurs)",
                            p.no_prints, "crispasr[tada]", p.cache_dir, p.accept_license);
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

    // #201: resolve a make-ref aux GGUF (encoder / aligner) — explicit path →
    // next to the model → cache dir → auto-download from the TADA HF repo.
    // Mirrors crispasr_run.cpp's resolve_aux so the adapter (server) path finds
    // the same files the CLI make-ref pipeline does.
    std::string resolve_makeref_aux(const whisper_params& p, const std::string& configured,
                                    const std::string& fname) const {
        auto present = [](const std::string& f) {
            struct stat st;
            return !f.empty() && stat(f.c_str(), &st) == 0;
        };
        if (present(configured))
            return configured;
        std::string mdir;
        {
            auto s = p.model.find_last_of("/\\");
            mdir = (s == std::string::npos) ? std::string(".") : p.model.substr(0, s);
        }
        std::string local = mdir + "/" + fname;
        if (present(local))
            return local;
        std::string cached = crispasr_cache::dir(p.cache_dir) + "/" + fname;
        if (crispasr_cache::file_present(cached))
            return cached;
        if (p.auto_download) {
            const std::string base = (p.backend == "tada-1b" || p.backend == "tada-tts-1b")
                                         ? "https://huggingface.co/cstr/tada-tts-1b-GGUF/resolve/main/"
                                         : "https://huggingface.co/cstr/tada-tts-3b-ml-GGUF/resolve/main/";
            return crispasr_managed_download(fname, base + fname,
                                             "llama3.2 (TADA-derived auxiliary model; see "
                                             "https://huggingface.co/HumeAI/tada-3b-ml/blob/main/LICENSE)",
                                             p.no_prints, "crispasr[tada]", p.cache_dir, p.accept_license);
        }
        return std::string();
    }

    // #201: on-the-fly clone from a reference WAV + its transcript
    // (p.tts_ref_text), applied to ctx_ in memory via tada_make_ref_from_pcm —
    // the no-temp-GGUF path the server needs. Opt-in, default OFF: without
    // CRISPASR_TADA_WAV_CLONE=1 a .wav voice is rejected as before. On success
    // sets last_voice_key_ and returns true.
    bool clone_from_wav(const whisper_params& p) {
        const std::string& v = p.tts_voice; // ends in .wav (callers gate on that)
        if (const char* g = crispasr_env::get("CRISPASR_TADA_WAV_CLONE"); !g || atoi(g) == 0) {
            fprintf(stderr, "crispasr[tada]: --voice with a .wav is opt-in — set CRISPASR_TADA_WAV_CLONE=1 to clone "
                            "at synth time, or build a tada-ref.gguf with --make-ref.\n");
            return false;
        }
        // Resolve the WAV + optional companion transcript, mirroring the server
        // voice-dir convention (qwen3-tts): a bare name that isn't an existing
        // path is taken relative to --voice-dir, with <stem>.txt as the ref-text
        // when the request didn't supply one. The server path-traversal guard
        // already rejected '..', absolute and '/'-containing names.
        std::string wav = v;
        std::string ref_text = p.tts_ref_text;
        {
            struct stat st;
            const bool bare = v.find('/') == std::string::npos && v.find('\\') == std::string::npos;
            if (stat(wav.c_str(), &st) != 0 && bare && !p.tts_voice_dir.empty() && v.find("..") == std::string::npos) {
                wav = p.tts_voice_dir + "/" + v;
                if (ref_text.empty()) {
                    const std::string stem = v.substr(0, v.size() - 4); // strip ".wav"
                    std::ifstream tf(p.tts_voice_dir + "/" + stem + ".txt");
                    if (tf.good()) {
                        std::stringstream ss;
                        ss << tf.rdbuf();
                        ref_text = ss.str();
                        while (!ref_text.empty() && isspace((unsigned char)ref_text.back()))
                            ref_text.pop_back();
                    }
                }
            }
        }
        if (ref_text.empty()) {
            fprintf(stderr, "crispasr[tada]: cloning from a .wav needs the transcript — pass --ref-text (CLI), the "
                            "ref_text field (server), or a companion <name>.txt in the voice dir.\n");
            return false;
        }
        const std::string enc = resolve_makeref_aux(p, p.make_ref_encoder, "tada-encoder-f16.gguf");
        const std::string lang = tada_prompt_lang_suffix(p.language);
        const std::string alang = lang.empty() ? std::string("en") : lang;
        std::string ali = resolve_makeref_aux(p, p.make_ref_aligner, "tada-aligner-" + alang + ".gguf");
        if (ali.empty() && alang != "en")
            ali = resolve_makeref_aux(p, p.make_ref_aligner, "tada-aligner-en.gguf");
        if (enc.empty() || ali.empty()) {
            fprintf(stderr, "crispasr[tada]: .wav cloning needs the encoder + aligner GGUFs (add --auto-download, "
                            "pass --make-ref-encoder/--make-ref-aligner, or place them next to the model).\n");
            return false;
        }
        float* pcm = nullptr;
        int n = 0, sr = 0;
        if (crispasr_audio_load_at_rate(wav.c_str(), 24000, &pcm, &n, &sr) != 0 || !pcm || n <= 0) {
            fprintf(stderr, "crispasr[tada]: .wav cloning: failed to decode '%s'\n", wav.c_str());
            if (pcm)
                free(pcm);
            return false;
        }
        int rc = tada_make_ref_from_pcm(ctx_, enc.c_str(), ali.c_str(), pcm, n, ref_text.c_str());
        free(pcm);
        if (rc != 0) {
            fprintf(stderr, "crispasr[tada]: .wav cloning failed (rc=%d)\n", rc);
            return false;
        }
        last_voice_key_ = v;
        if (!p.no_prints)
            fprintf(stderr, "crispasr[tada]: cloned voice in-memory from '%s'\n", wav.c_str());
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
            // #201: per-request on-the-fly clone (opt-in, default OFF). This is
            // the server's primary path for a .wav voice + ref_text.
            return clone_from_wav(p);
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
    int def_num_candidates_ = 1;
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
