// crispasr_backend_chatterbox.cpp — adapter for Chatterbox TTS (base, turbo,
// Kartoffelbox). Two-GGUF runtime: T3 AR model (--model) and S3Gen
// encoder+vocoder (--codec-model).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "chatterbox.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class ChatterboxBackend : public CrispasrBackend {
public:
    ChatterboxBackend() = default;
    ~ChatterboxBackend() override { ChatterboxBackend::shutdown(); }

    const char* name() const override { return "chatterbox"; }

    uint32_t capabilities() const override {
        return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_VOICE_CLONING;
    }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[chatterbox]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        chatterbox_context_params cp = chatterbox_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        ctx_ = chatterbox_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[chatterbox]: failed to load T3 model '%s'\n", p.model.c_str());
            return false;
        }
        // S3Gen companion: --codec-model <path>
        if (!p.tts_codec_model.empty()) {
            if (chatterbox_set_s3gen_path(ctx_, p.tts_codec_model.c_str()) != 0) {
                fprintf(stderr, "crispasr[chatterbox]: failed to load S3Gen '%s'\n", p.tts_codec_model.c_str());
                return false;
            }
            s3gen_loaded_ = true;
        } else {
            // Try auto-discovering S3Gen next to the T3 model
            std::string dir;
            auto slash = p.model.find_last_of("/\\");
            dir = (slash == std::string::npos) ? "." : p.model.substr(0, slash);
            // Try turbo variant first, then base
            for (const char* candidate : {"chatterbox-turbo-s3gen-f16.gguf", "chatterbox-s3gen-f16.gguf"}) {
                std::string path = dir + "/" + candidate;
                FILE* f = fopen(path.c_str(), "rb");
                if (f) {
                    fclose(f);
                    if (chatterbox_set_s3gen_path(ctx_, path.c_str()) == 0) {
                        s3gen_loaded_ = true;
                        if (!p.no_prints)
                            fprintf(stderr, "crispasr[chatterbox]: auto-loaded S3Gen '%s'\n", path.c_str());
                        break;
                    }
                }
            }
        }
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& /*params*/) override {
        if (!ctx_ || text.empty())
            return {};
        if (!s3gen_loaded_) {
            fprintf(stderr, "crispasr[chatterbox]: S3Gen not loaded. Pass --codec-model <s3gen.gguf>\n");
            return {};
        }
        int n = 0;
        float* pcm = chatterbox_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        chatterbox_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            chatterbox_free(ctx_);
            ctx_ = nullptr;
        }
        s3gen_loaded_ = false;
    }

private:
    chatterbox_context* ctx_ = nullptr;
    bool s3gen_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_chatterbox_backend() {
    return std::unique_ptr<CrispasrBackend>(new ChatterboxBackend());
}
