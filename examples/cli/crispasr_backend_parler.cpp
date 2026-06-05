// crispasr_backend_parler.cpp -- adapter for parler-tts/parler-tts-mini-v1.1
// (and parler-tts-large-v1). Prompt-conditioned TTS: describe the voice
// in natural language via --instruct, and the model generates speech
// matching that description.
//
// Single-GGUF runtime: the GGUF contains T5 encoder, MusicGen-style
// decoder, and DAC 44 kHz codec.  Produced by convert-parler-to-gguf.py.
//
// Usage:
//   crispasr -b parler-tts -m parler-tts-mini-v1.1-f16.gguf \
//     --instruct "A female speaker with a warm voice in a quiet room" \
//     --tts "Hello, this is a test of Parler TTS."

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "parler_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Default voice description when none is provided via --instruct.
static const char* DEFAULT_DESCRIPTION = "A female speaker delivers her words at a moderate pace with a clear "
                                         "and natural tone in a quiet environment.";

class ParlerTTSBackend : public CrispasrBackend {
public:
    ParlerTTSBackend() = default;
    ~ParlerTTSBackend() override { ParlerTTSBackend::shutdown(); }

    const char* name() const override { return "parler-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }

    int tts_sample_rate() const override { return 44100; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[parler-tts]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        parler_tts_context_params cp = parler_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.flash_attn = p.flash_attn;

        // Parler TTS needs stochastic sampling (temp=1.0) by default.
        // CLI default is 0.0 (greedy for ASR), so override unless user explicitly set -tp.
        cp.temperature = (p.temperature > 0.0f) ? p.temperature : 1.0f;
        cp.seed = p.seed;

        std::string model_path = p.model;
        model_path = crispasr_resolve_model_cli(model_path, p.backend, p.no_prints, p.cache_dir, p.auto_download, "");

        ctx_ = parler_tts_init_from_file(model_path.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[parler-tts]: failed to load model '%s'\n", model_path.c_str());
            return false;
        }

        // Set voice description from --instruct parameter
        const char* desc = DEFAULT_DESCRIPTION;
        if (!p.tts_instruct.empty()) {
            desc = p.tts_instruct.c_str();
        }
        int rc = parler_tts_set_description(ctx_, desc);
        if (rc != 0) {
            fprintf(stderr, "crispasr[parler-tts]: failed to encode description\n");
            return false;
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        parler_tts_set_temperature(ctx_, (params.temperature > 0.0f) ? params.temperature : 1.0f);
        parler_tts_set_seed(ctx_, params.seed);

        // If --instruct changed since last call, re-encode description
        if (!params.tts_instruct.empty() && params.tts_instruct != last_description_) {
            parler_tts_set_description(ctx_, params.tts_instruct.c_str());
            last_description_ = params.tts_instruct;
        }

        int n = 0;
        float* pcm = parler_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n);
        parler_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            parler_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    parler_tts_context* ctx_ = nullptr;
    std::string last_description_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_parler_tts_backend() {
    return std::unique_ptr<CrispasrBackend>(new ParlerTTSBackend());
}
