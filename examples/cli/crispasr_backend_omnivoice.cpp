// crispasr_backend_omnivoice.cpp — adapter for k2-fsa/OmniVoice TTS.
//
// Two-GGUF runtime: the main model (LLM + audio layers, loaded from
// --model) and a separate HiggsAudioV2 audio tokenizer (loaded via
// --codec-model, or auto-discovered as a sibling).
// Voice cloning: --voice ref.wav --ref-text "..."

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "omnivoice.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "core/crispasr_env.h"

namespace {

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static std::string dir_of(const std::string& p) {
    auto sep = p.find_last_of("/\\");
    return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
}

static std::string discover_tokenizer(const std::string& model_path) {
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "omnivoice-tokenizer.gguf",
        "omnivoice-tokenizer-f16.gguf",
        "omnivoice-audio-tokenizer.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class OmniVoiceBackend : public CrispasrBackend {
public:
    OmniVoiceBackend() = default;
    ~OmniVoiceBackend() override { OmniVoiceBackend::shutdown(); }

    const char* name() const override { return "omnivoice"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[omnivoice]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        omnivoice_context_params cp = omnivoice_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.flash_attn = p.flash_attn;
        cp.seed = p.seed;

        ctx_ = omnivoice_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[omnivoice]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Resolve audio tokenizer GGUF
        std::string tok_path = p.tts_codec_model;
        if (tok_path.empty() || tok_path == "auto" || tok_path == "default") {
            tok_path = discover_tokenizer(p.model);
        }
        if (!tok_path.empty()) {
            if (omnivoice_set_tokenizer_path(ctx_, tok_path.c_str()) != 0) {
                fprintf(stderr, "crispasr[omnivoice]: failed to load tokenizer '%s'\n", tok_path.c_str());
            } else if (!p.no_prints) {
                fprintf(stderr, "crispasr[omnivoice]: tokenizer loaded from '%s'\n", tok_path.c_str());
            }
        } else if (!p.no_prints) {
            fprintf(stderr, "crispasr[omnivoice]: no audio tokenizer found. Pass --codec-model PATH or place "
                            "omnivoice-tokenizer.gguf next to the model.\n");
            fprintf(stderr, "crispasr[omnivoice]: code generation will work; audio decode requires the tokenizer.\n");
        }

        // Diff-harness: OMNIVOICE_ENCODE_DIFF=<ref.gguf> runs the encode-path
        // stage diff and exits (#254 voice-clone port validation).
        if (const char* rp = crispasr_env::get("CRISPASR_OMNIVOICE_ENCODE_DIFF")) {
            int rc = omnivoice_encode_diff(ctx_, rp);
            exit(rc == 0 ? 0 : 1);
        }

        // Language
        if (!p.language.empty() && p.language != "auto") {
            omnivoice_set_language(ctx_, p.language.c_str());
        }

        // Voice cloning is intentionally NOT applied here (at startup): the
        // HTTP server owns ONE backend instance and passes a per-request
        // `voice=` in params.tts_voice, which only takes effect if re-applied
        // at synth time (see prepare_voice below) — the same shape as the
        // per-request language/seed/instruct handling in synthesize(). Baking
        // --voice at init alone made every /v1/audio/speech request serve the
        // startup voice (or the built-in default when no --voice was given).

        // Style instruct. Upstream rejects an unsupported item rather than
        // ignoring it, so a bad --tts-instruct must fail the run — silently
        // synthesising with no voice design is the bug, not the fallback.
        if (!p.tts_instruct.empty()) {
            if (omnivoice_set_instruct(ctx_, p.tts_instruct.c_str()) != 0) {
                fprintf(stderr, "crispasr[omnivoice]: --instruct was rejected (see above)\n");
                return false;
            }
        }

        // Speaking-rate multiplier (--tts-speed): scales the estimated target
        // length. >1 = faster/shorter, <1 = slower/longer. Handy to trim an
        // over-long estimate from a slow reference voice (#254).
        if (p.tts_speed > 0.0f && p.tts_speed != 1.0f) {
            omnivoice_set_speed(ctx_, p.tts_speed);
        }

        // Diffusion step count (--tts-steps): stage0 = num_steps × 2 backbone
        // forwards — the dominant cost. Default 32; lower trades refinement for
        // speed (ASR-clean to ~16). tts_num_steps is -1 unless the user set it.
        if (p.tts_num_steps >= 1) {
            omnivoice_set_num_steps(ctx_, p.tts_num_steps);
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        // Per-request voice cloning, same reasoning as the per-request knobs
        // below: the server copies the request's `voice` into
        // params.tts_voice, so re-apply the reference prompt here whenever it
        // changes (an empty value clears the prompt -> plain TTS).
        // Fail closed: if a NON-EMPTY voice was requested and could not be
        // applied, return an empty result so the request errors instead of
        // synthesising in whatever voice happens to be loaded. See
        // prepare_voice().
        if (!prepare_voice(params))
            return {};

        // Apply the diffusion step count PER CALL, not just at init: the server
        // reuses one backend instance and passes tts_num_steps per request, so a
        // per-request "num_steps" only takes effect if set here. Read live by the
        // runtime (no reload). tts_num_steps is -1 unless the caller set it. (#254)
        if (params.tts_num_steps >= 1) {
            omnivoice_set_num_steps(ctx_, params.tts_num_steps);
        }

        // #13273: and the target language, for the same reason. The server owns
        // ONE backend instance for the whole session and passes the per-request
        // language in `params`, so applying it only in init() meant
        // SubtitleEdit's language menu could never change anything after the
        // first line — the menu was decoration. Prefer -tl (TTS-explicit), else
        // -l; "auto" is the CLI's no-op sentinel and resolve() clears on it.
        // Same shape as crispasr_backend_cosyvoice3.cpp.
        const std::string tgt_lang = !params.target_lang.empty() ? params.target_lang : params.language;
        omnivoice_set_language(ctx_, tgt_lang.c_str());

        // Same omission, found while A/B-ing the above: /v1/audio/speech accepts
        // a per-request `seed` and omnivoice dropped it, so re-rendering one
        // subtitle line could not be made reproducible. 0 is the "leave it
        // alone" default (the runtime's own default is 42), so this changes
        // nothing unless a caller actually asks.
        if (params.seed != 0) {
            omnivoice_set_seed(ctx_, params.seed);
        }

        // And the instruct, for the third time the same reason: the server maps
        // a per-request "instructions" field onto params.tts_instruct, and
        // applying it only in init() left that field dead on every line after
        // the first. Rejection is already reported by the runtime; returning
        // empty here surfaces it as a failed synthesis rather than silently
        // dropping the voice design.
        if (omnivoice_set_instruct(ctx_, params.tts_instruct.c_str()) != 0) {
            return {};
        }

        int n_samples = 0;
        float* pcm = omnivoice_synthesize(ctx_, text.c_str(), &n_samples);
        if (pcm && n_samples > 0) {
            std::vector<float> out(pcm, pcm + n_samples);
            omnivoice_pcm_free(pcm);
            return out;
        }
        // Fall back to code-only output
        int n_codes = 0;
        int32_t* codes = omnivoice_synthesize_codes(ctx_, text.c_str(), &n_codes);
        if (codes && n_codes > 0) {
            if (!params.no_prints) {
                fprintf(stderr, "crispasr[omnivoice]: generated %d codes (audio decode requires tokenizer)\n", n_codes);
            }
            omnivoice_codes_free(codes);
        }
        return {};
    }

    void shutdown() override {
        if (ctx_) {
            omnivoice_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    // Re-apply the reference voice per request (mirror of moss-tts
    // prepare_voice, and of the per-request language/seed/instruct handling in
    // synthesize()). omnivoice_set_voice_prompt with an empty path clears the
    // reference (plain TTS); with a path it encodes the WAV through the audio
    // tokenizer into ref_audio_codes + ref_T. The encode is content-addressed
    // disk-cached inside omnivoice.cpp, so re-calling it for the same ref is
    // cheap. last_voice_ dedupes identical consecutive voices so a repeated
    // request never pays the encode again.
    // Returns false when a NON-EMPTY voice was requested and could not be
    // applied. The caller must then fail the request: the runtime has already
    // dropped the previous reference (fail-closed in omnivoice_set_voice_prompt),
    // so synthesising anyway would render in the wrong (default) voice — and
    // before that change, in the previous speaker's.
    bool prepare_voice(const whisper_params& params) {
        if (params.tts_voice == last_voice_)
            return true;
        last_voice_ = params.tts_voice;
        std::string ref_text = params.tts_ref_text;
        if (omnivoice_set_voice_prompt(ctx_, params.tts_voice.c_str(), ref_text.c_str()) != 0) {
            fprintf(stderr, "crispasr[omnivoice]: failed to set voice prompt '%s'\n", params.tts_voice.c_str());
            last_voice_.clear(); // force a re-apply on the next request
            return params.tts_voice.empty();
        }
        return true;
    }

    omnivoice_context* ctx_ = nullptr;
    std::string last_voice_; // cache key for the loaded reference voice
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_omnivoice_backend() {
    return std::make_unique<OmniVoiceBackend>();
}
