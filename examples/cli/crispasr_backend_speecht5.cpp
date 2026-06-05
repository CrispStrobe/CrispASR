// crispasr_backend_speecht5.cpp -- adapter for Microsoft SpeechT5 TTS.
//
// Single-GGUF runtime containing the text encoder, speech decoder,
// post-net, and HiFi-GAN vocoder. Speaker conditioning via 512-dim
// x-vector passed with --voice <xvector.bin>.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "speecht5_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Load a binary file of float32 values (e.g. x-vector embedding).
static std::vector<float> load_float_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / sizeof(float));
    std::vector<float> data(n);
    size_t rd = fread(data.data(), sizeof(float), n, f);
    fclose(f);
    if ((int)rd != n)
        return {};
    return data;
}

class SpeechT5Backend : public CrispasrBackend {
public:
    SpeechT5Backend() = default;
    ~SpeechT5Backend() override { SpeechT5Backend::shutdown(); }

    const char* name() const override { return "speecht5"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD; }

    int tts_sample_rate() const override { return 16000; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[speecht5]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        speecht5_tts_params sp = speecht5_tts_default_params();
        sp.n_threads = p.n_threads;
        sp.verbosity = p.no_prints ? 0 : 1;
        sp.use_gpu = crispasr_backend_should_use_gpu(p);

        std::string model_path = p.model;
        ctx_ = speecht5_tts_init(model_path.c_str(), sp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[speecht5]: failed to load model '%s'\n", model_path.c_str());
            return false;
        }

        // Load speaker embedding from --voice
        if (!p.tts_voice.empty()) {
            load_speaker(p.tts_voice, p.no_prints);
        } else {
            // Use a default zero speaker embedding
            // (the model needs a speaker embedding to work)
            std::vector<float> default_spk(512, 0.0f);
            // Use a reasonable default: all zeros will produce generic voice
            speecht5_tts_set_speaker(ctx_, default_spk.data(), 512);
            if (!p.no_prints) {
                fprintf(stderr, "crispasr[speecht5]: using zero speaker embedding. "
                                "Pass --voice <xvector.bin> for better results.\n");
            }
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!ctx_ || text.empty())
            return {};

        // Load speaker if changed
        if (!params.tts_voice.empty() && params.tts_voice != last_voice_) {
            load_speaker(params.tts_voice, params.no_prints);
        }

        int n = 0;
        float* pcm = speecht5_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};

        std::vector<float> out(pcm, pcm + n);
        speecht5_tts_pcm_free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            speecht5_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    speecht5_tts_context* ctx_ = nullptr;
    std::string last_voice_;

    void load_speaker(const std::string& voice_path, bool quiet) {
        auto data = load_float_bin(voice_path);
        if (data.empty()) {
            if (!quiet) {
                fprintf(stderr, "crispasr[speecht5]: failed to load speaker from '%s'\n", voice_path.c_str());
            }
            return;
        }
        speecht5_tts_set_speaker(ctx_, data.data(), (int)data.size());
        last_voice_ = voice_path;
        if (!quiet) {
            fprintf(stderr, "crispasr[speecht5]: loaded speaker embedding (%d dims) from '%s'\n", (int)data.size(),
                    voice_path.c_str());
        }
    }
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_speecht5_backend() {
    return std::unique_ptr<CrispasrBackend>(new SpeechT5Backend());
}
