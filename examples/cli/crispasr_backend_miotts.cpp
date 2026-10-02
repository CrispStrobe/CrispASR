// crispasr_backend_miotts.cpp — CLI adapter for Aratako/MioTTS TTS.

#include "crispasr_backend.h"
#include "whisper_params.h"
#include "crispasr_voice_provenance.h"

#include "miotts.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {
std::string resolve_preset(const std::string& voice, const std::string& directory) {
    if (!voice.empty() && !directory.empty() && voice.find_first_of("/\\") == std::string::npos &&
        voice.find("..") == std::string::npos) {
        std::error_code ec;
        const std::string path = directory + "/" + voice;
        if (std::filesystem::exists(path, ec))
            return path;
        if (std::filesystem::exists(path + ".emb.gguf", ec))
            return path + ".emb.gguf";
    }
    return crispasr_voice::resolve_voice_path(voice, directory);
}
} // namespace

class MioTtsBackend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto cp = miotts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        cp.use_gpu = p.use_gpu;
        cp.temperature = p.temperature;
        cp.seed = p.seed;
        cp.max_tokens = 750;
        ctx_ = miotts_init_from_file(p.model.c_str(), cp);
        if (!ctx_)
            return false;
        const std::string voice = resolve_preset(p.tts_voice, p.tts_voice_dir);
        if (!voice.empty() && miotts_load_preset_embedding(ctx_, voice.c_str()) != 0) {
            fprintf(stderr, "crispasr[miotts]: failed to load voice preset '%s'\n", p.tts_voice.c_str());
            shutdown();
            return false;
        }
        default_voice_ = p.tts_voice;
        voice_ = voice;
        return true;
    }

    void shutdown() override {
        if (ctx_) {
            miotts_free(ctx_);
            ctx_ = nullptr;
        }
        default_voice_.clear();
        voice_.clear();
    }

    const char* name() const override { return "miotts"; }
    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE; }
    int input_sample_rate() const override { return 16000; }
    int tts_sample_rate() const override { return miotts_get_sample_rate(ctx_); }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};
        miotts_set_temperature(ctx_, p.temperature);
        miotts_set_seed(ctx_, p.seed);
        // A resident server passes voice overrides per request. An empty
        // override restores the startup preset rather than retaining another
        // request's speaker embedding.
        const std::string requested = p.tts_voice.empty() ? default_voice_ : p.tts_voice;
        const std::string voice = resolve_preset(requested, p.tts_voice_dir);
        if (voice != voice_) {
            const int rc = voice.empty() ? miotts_set_reference(ctx_, nullptr, 0)
                                         : miotts_load_preset_embedding(ctx_, voice.c_str());
            if (rc != 0) {
                fprintf(stderr, "crispasr[miotts]: failed to load voice preset '%s'\n", voice.c_str());
                return {};
            }
            voice_ = voice;
        }
        int n = 0;
        float* pcm = miotts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> result(pcm, pcm + n);
        miotts_free_audio(pcm);
        return result;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

private:
    miotts_context* ctx_ = nullptr;
    std::string default_voice_;
    std::string voice_;
};

std::unique_ptr<CrispasrBackend> crispasr_create_miotts_backend() {
    return std::make_unique<MioTtsBackend>();
}
