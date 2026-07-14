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
        if (const char* rp = getenv("OMNIVOICE_ENCODE_DIFF")) {
            int rc = omnivoice_encode_diff(ctx_, rp);
            exit(rc == 0 ? 0 : 1);
        }

        // Language
        if (!p.language.empty() && p.language != "auto") {
            omnivoice_set_language(ctx_, p.language.c_str());
        }

        // Voice cloning
        if (!p.tts_voice.empty()) {
            std::string ref_text = p.tts_ref_text;
            if (omnivoice_set_voice_prompt(ctx_, p.tts_voice.c_str(), ref_text.c_str()) != 0) {
                fprintf(stderr, "crispasr[omnivoice]: failed to set voice prompt '%s'\n", p.tts_voice.c_str());
            }
        }

        // Style instruct
        if (!p.tts_instruct.empty()) {
            omnivoice_set_instruct(ctx_, p.tts_instruct.c_str());
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

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
    omnivoice_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_omnivoice_backend() {
    return std::make_unique<OmniVoiceBackend>();
}
