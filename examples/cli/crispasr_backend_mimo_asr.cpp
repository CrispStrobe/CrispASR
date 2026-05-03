// crispasr_backend_mimo_asr.cpp — XiaomiMiMo/MiMo-V2.5-ASR backend.
//
// MiMo-V2.5-ASR pairs a 6-layer "input_local_transformer" audio-token
// processor + 36-layer Qwen2 LM (the LM weights live in the GGUF passed
// via -m) with a SEPARATE audio tokenizer GGUF (the MiMo-Audio-Tokenizer
// encoder, ~395 MB Q4_K). The tokenizer path is resolved from
// --codec-model PATH or auto-discovered from candidate filenames in the
// same directory as the LM. PLAN #51 step 8/10.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "mimo_asr.h"
#include "whisper_params.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

std::string discover_audio_tokenizer(const std::string& model_path) {
    auto dir_of = [](const std::string& p) -> std::string {
        auto sep = p.find_last_of("/\\");
        return (sep == std::string::npos) ? std::string(".") : p.substr(0, sep);
    };
    const std::string dir = dir_of(model_path);
    static const char* candidates[] = {
        "mimo-tokenizer-q4_k.gguf",
        "mimo-tokenizer.gguf",
        "mimo-audio-tokenizer.gguf",
    };
    for (const char* name : candidates) {
        std::string p = dir + "/" + name;
        if (file_exists(p))
            return p;
    }
    return "";
}

class MimoAsrBackend : public CrispasrBackend {
public:
    MimoAsrBackend() = default;
    ~MimoAsrBackend() override { MimoAsrBackend::shutdown(); }

    const char* name() const override { return "mimo-asr"; }
    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD | CAP_TOKEN_CONFIDENCE | CAP_TIMESTAMPS_CTC; }

    bool init(const whisper_params& params) override {
        auto cp = mimo_asr_context_default_params();
        cp.n_threads = params.n_threads;
        cp.verbosity = params.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(params);
        cp.temperature = params.temperature;

        ctx_ = mimo_asr_init_from_file(params.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[mimo-asr]: failed to load LM model '%s'\n", params.model.c_str());
            return false;
        }

        std::string tok_path = params.tts_codec_model;
        if (tok_path.empty())
            tok_path = discover_audio_tokenizer(params.model);
        if (tok_path.empty()) {
            fprintf(stderr, "crispasr[mimo-asr]: no audio tokenizer GGUF found. Pass --codec-model PATH "
                            "or place mimo-tokenizer-q4_k.gguf next to the LM.\n");
            return false;
        }
        if (mimo_asr_set_tokenizer_path(ctx_, tok_path.c_str()) != 0) {
            fprintf(stderr, "crispasr[mimo-asr]: failed to register tokenizer path '%s'\n", tok_path.c_str());
            return false;
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr[mimo-asr]: tokenizer at '%s'\n", tok_path.c_str());
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        (void)params;
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;
        char* text = mimo_asr_transcribe(ctx_, samples, n_samples);
        if (text) {
            crispasr_segment seg;
            seg.text = text;
            seg.t0 = t_offset_cs;
            seg.t1 = t_offset_cs + (int64_t)((int64_t)n_samples * 100 / 16000);
            out.push_back(std::move(seg));
            free(text);
        }
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            mimo_asr_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    mimo_asr_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_mimo_asr_backend() {
    return std::make_unique<MimoAsrBackend>();
}
