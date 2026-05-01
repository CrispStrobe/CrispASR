// crispasr_backend_kokoro.cpp — adapter for hexgrad/Kokoro-82M and
// yl4579/StyleTTS2-LJSpeech (iSTFTNet TTS).
//
// Two-GGUF runtime: the talker GGUF (loaded from --model) and a
// per-voice GGUF (loaded via --voice). The talker carries 5 components
// (text_enc, BERT, predictor, decoder, generator); each voice GGUF
// stores one (style_pred, style_dec) reference vector indexed by
// phoneme length. Phonemizer: espeak-ng shell-out, with --tts-phonemes
// to bypass it.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "kokoro.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Per-language model + voice routing lives in src/kokoro.cpp behind the
// crispasr_kokoro_* C ABI so wrappers (Python, Rust, Go, ...) can reuse
// the same policy. The CLI just shells out to those helpers and prints
// friendly messages on the side.

std::string kokoro_resolve_model(const std::string& lang, const std::string& model_path) {
    char out[1024] = {0};
    int rc = crispasr_kokoro_resolve_model_for_lang(model_path.c_str(), lang.c_str(),
                                                    out, (int)sizeof(out));
    if (rc == 0) {
        fprintf(stderr,
                "crispasr[kokoro]: language 'de' — preferring German-trained "
                "backbone '%s' over '%s' (dida-80b/kokoro-german-hui-"
                "multispeaker-base, Apache-2.0; HUI corpus, CC0; see PLAN "
                "#56 opt 2b)\n",
                out, model_path.c_str());
        return out;
    }
    return model_path;
}

std::string kokoro_resolve_fallback_voice(const std::string& lang,
                                          const std::string& model_path,
                                          std::string* out_picked_name = nullptr) {
    char path[1024] = {0};
    char picked[64] = {0};
    int rc = crispasr_kokoro_resolve_fallback_voice(model_path.c_str(), lang.c_str(),
                                                    path, (int)sizeof(path),
                                                    picked, (int)sizeof(picked));
    if (rc != 0) return {};
    if (out_picked_name) *out_picked_name = picked;
    return path;
}

class KokoroBackend : public CrispasrBackend {
public:
    KokoroBackend() = default;
    ~KokoroBackend() override { KokoroBackend::shutdown(); }

    const char* name() const override { return "kokoro"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[kokoro]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        kokoro_context_params cp = kokoro_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // Map -l/--language to the espeak-ng voice. "auto" keeps the default
        // (en-us) since espeak has no auto-detect mode.
        if (!p.language.empty() && p.language != "auto") {
            std::strncpy(cp.espeak_lang, p.language.c_str(), sizeof(cp.espeak_lang) - 1);
            cp.espeak_lang[sizeof(cp.espeak_lang) - 1] = '\0';
        }
        std::string resolved_model = kokoro_resolve_model(p.language, p.model);
        is_german_backbone_ = (resolved_model != p.model);
        ctx_ = kokoro_init_from_file(resolved_model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[kokoro]: failed to load model '%s'\n", resolved_model.c_str());
            return false;
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        // Voice pack: load once on first call. Required — without it, synthesis
        // returns nullptr (predictor needs a (style_pred, style_dec) reference).
        // Resolution order: --voice (explicit) → ff_siwis fallback for non-native
        // languages → empty (synthesis will fail with a clear error).
        std::string voice_path = params.tts_voice;
        if (voice_path.empty() && !params.language.empty() && params.language != "auto"
            && !crispasr_kokoro_lang_has_native_voice(params.language.c_str())) {
            std::string picked;
            voice_path = kokoro_resolve_fallback_voice(params.language, params.model, &picked);
            if (!voice_path.empty() && crispasr_kokoro_lang_is_german(params.language.c_str())) {
                const bool is_kikiri = picked == "df_victoria";
                if (is_german_backbone_) {
                    fprintf(stderr,
                            "crispasr[kokoro]: language 'de' — using %s voicepack on the "
                            "dida-80b German-trained backbone (Apache-2.0%s) — fully native "
                            "German signal path. See PLAN #56 opt 2b.\n",
                            picked.c_str(),
                            is_kikiri ? "; voicepack from kikiri-tts/kikiri-german-victoria"
                                      : "; voicepack from Tundragoon German fine-tune");
                } else {
                    fprintf(stderr,
                            "crispasr[kokoro]: language 'de' — using %s voicepack on the "
                            "official Kokoro-82M backbone. Predictor weights are still the "
                            "English-trained ones; for fully native German prosody, drop "
                            "kokoro-de-hui-base-f16.gguf next to the model. See PLAN #56.\n",
                            picked.c_str());
                }
            } else if (!voice_path.empty()) {
                fprintf(stderr,
                        "crispasr[kokoro]: no native Kokoro-82M voice for language '%s'; "
                        "using %s fallback (French speaker — prosody will sound "
                        "French-accented). See PLAN #56.\n",
                        params.language.c_str(), picked.empty() ? "ff_siwis" : picked.c_str());
            } else {
                fprintf(stderr,
                        "crispasr[kokoro]: no native Kokoro-82M voice for language '%s' "
                        "and no fallback found at '<model_dir>/kokoro-voice-*.gguf'. "
                        "Pass --voice <path> or convert one via "
                        "models/convert-kokoro-voice-to-gguf.py. See PLAN #56.\n",
                        params.language.c_str());
            }
        }
        if (!voice_loaded_ && !voice_path.empty()) {
            if (kokoro_load_voice_pack(ctx_, voice_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[kokoro]: failed to load voice pack '%s'\n", voice_path.c_str());
                return {};
            }
            voice_loaded_ = true;
        }

        int n = 0;
        float* pcm = kokoro_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        kokoro_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            kokoro_free(ctx_);
            ctx_ = nullptr;
        }
        voice_loaded_ = false;
    }

private:
    kokoro_context* ctx_ = nullptr;
    bool voice_loaded_ = false;
    bool is_german_backbone_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_kokoro_backend() {
    return std::unique_ptr<CrispasrBackend>(new KokoroBackend());
}
