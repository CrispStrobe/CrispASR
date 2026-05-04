// crispasr_backend_chatterbox.cpp — adapter for ResembleAI/chatterbox TTS
// (T3 AR text→speech-tokens model + S3Gen flow-matching tokens→24 kHz audio).
//
// Two-GGUF runtime: the T3 (loaded from --model) and the S3Gen (loaded
// via --codec-model, or auto-discovered as a sibling of the T3, or via
// the auto-download companion file).
//
// The same adapter handles four upstream variants — all keyed off the
// `chatterbox.t3.arch` GGUF metadata field that the runtime reads:
//   chatterbox          (English, Llama T3 + base S3Gen)
//   chatterbox-turbo    (English, GPT-2 T3 + meanflow S3Gen)
//   kartoffelbox-turbo  (German,  GPT-2 T3 + shared turbo S3Gen)
//   lahgtna-chatterbox  (Arabic,  Llama T3 + shared base  S3Gen)
//
// Voice prompt handling:
//   --voice <path-to-WAV>  → cloned voice via VE/CAMPPlus (chatterbox_set_voice_from_wav)
//   --voice ""             → built-in voice baked into the T3 GGUF (conds.*)
//
// CFM steps: --tts-steps overrides the default 10 (or 2 for the meanflow
// turbo variants). Pass 0 to keep the runtime default.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "chatterbox.h"

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

// Look for a sibling S3Gen GGUF next to the T3. Variants share companion
// names per the registry: base + lahgtna use "chatterbox-s3gen-*", turbo
// + kartoffelbox-turbo share "chatterbox-turbo-s3gen-*".
static std::string discover_s3gen(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "chatterbox-s3gen-q8_0.gguf",       "chatterbox-s3gen-f16.gguf",       "chatterbox-s3gen-q4_k.gguf",
        "chatterbox-turbo-s3gen-f16.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p)) {
            return p;
        }
    }
    return "";
}

class ChatterboxBackend : public CrispasrBackend {
public:
    ChatterboxBackend() = default;
    ~ChatterboxBackend() override { ChatterboxBackend::shutdown(); }

    const char* name() const override { return "chatterbox"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[chatterbox]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        chatterbox_context_params cp = chatterbox_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        // Global --temperature defaults to 0.0 (whisper-style greedy ASR);
        // chatterbox needs a non-zero AR sampling temperature. Only override
        // the 0.8 runtime default when the user explicitly passes >0.
        if (p.temperature > 0.0f) {
            cp.temperature = p.temperature;
        }
        if (p.tts_steps > 0) {
            cp.cfm_steps = p.tts_steps;
        }
        ctx_ = chatterbox_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[chatterbox]: failed to load T3 '%s'\n", p.model.c_str());
            return false;
        }

        std::string s3gen_path = p.tts_codec_model;
        if (s3gen_path.empty()) {
            s3gen_path = discover_s3gen(p.model);
        }
        if (s3gen_path.empty()) {
            fprintf(stderr, "crispasr[chatterbox]: no S3Gen GGUF found. Pass --codec-model PATH or place a "
                            "chatterbox-s3gen-*.gguf next to the T3.\n");
            return false;
        }
        if (chatterbox_set_s3gen_path(ctx_, s3gen_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[chatterbox]: failed to load S3Gen '%s'\n", s3gen_path.c_str());
            return false;
        }
        if (!p.no_prints) {
            fprintf(stderr, "crispasr[chatterbox]: S3Gen path = '%s'\n", s3gen_path.c_str());
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty()) {
            return {};
        }

        if (!voice_loaded_) {
            const std::string& v = params.tts_voice;
            if (!v.empty()) {
                if (chatterbox_set_voice_from_wav(ctx_, v.c_str()) != 0) {
                    fprintf(stderr, "crispasr[chatterbox]: failed to load reference voice from '%s'\n", v.c_str());
                    return {};
                }
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr[chatterbox]: cloned voice from '%s'\n", v.c_str());
                }
            } else if (!params.no_prints) {
                fprintf(stderr, "crispasr[chatterbox]: using built-in voice (conds.* baked into T3)\n");
            }
            voice_loaded_ = true;
        }

        int n = 0;
        float* pcm = chatterbox_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0) {
            return {};
        }
        std::vector<float> out(pcm, pcm + n);
        chatterbox_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            chatterbox_free(ctx_);
            ctx_ = nullptr;
        }
        voice_loaded_ = false;
    }

private:
    chatterbox_context* ctx_ = nullptr;
    bool voice_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_chatterbox_backend() {
    return std::unique_ptr<CrispasrBackend>(new ChatterboxBackend());
}
