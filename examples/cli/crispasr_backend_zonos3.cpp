// crispasr_backend_zonos3.cpp — CLI adapter for Zyphra ZONOS3-Mini TTS (issue #603).
//
// --voice takes a 192-d speaker embedding: a path to <name>.embedding.npy, or a name looked up under --voice-dir
// (also inside per-language subdirectories, as in the HF repo's gguf/voices/<Language>/). No voice = the
// unconditioned prompt (no speaker row).

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "zonos3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {
std::string resolve_embedding(const std::string& voice, const std::string& directory) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (voice.empty())
        return {};
    if (fs::exists(voice, ec))
        return voice;
    if (directory.empty() || voice.find("..") != std::string::npos)
        return {};
    for (const std::string& name : {voice, voice + ".embedding.npy", voice + ".npy"}) {
        if (fs::exists(fs::path(directory) / name, ec))
            return (fs::path(directory) / name).string();
    }
    // gguf/voices/<Language>/<Name>_<tags>.embedding.npy: match the file name or its leading "<Name>_".
    for (auto it = fs::recursive_directory_iterator(directory, ec); !ec && it != fs::recursive_directory_iterator();
         it.increment(ec)) {
        const std::string fn = it->path().filename().string();
        const std::string suffix = ".embedding.npy";
        if (fn.size() <= suffix.size() || fn.compare(fn.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const std::string stem = fn.substr(0, fn.size() - suffix.size());
        if (stem == voice || stem.rfind(voice + "_", 0) == 0)
            return it->path().string();
    }
    return {};
}
} // namespace

class Zonos3Backend : public CrispasrBackend {
public:
    bool init(const whisper_params& p) override {
        auto cp = zonos3_tts_context_default_params();
        cp.n_threads = p.n_threads;
        cp.verbosity = p.no_prints ? 0 : (p.verbose ? 2 : 1);
        cp.use_gpu = p.use_gpu;
        cp.temperature = p.temperature > 0.0f ? p.temperature : -1.0f; // CLI 0 = model default (0.75)
        if (const char* g = getenv("CRISPASR_ZONOS3_GREEDY"))
            if (*g == '1')
                cp.temperature = 0.0f;
        cp.seed = (uint32_t)p.seed;
        if (const char* c = getenv("CRISPASR_ZONOS3_CFG"))
            cp.cfg_scale = (float)atof(c);
        if (!p.tts_codec_model.empty())
            cp.codec_path = p.tts_codec_model.c_str(); // --codec-model zac-keep2.gguf / zac.gguf
        if (const char* m = getenv("CRISPASR_ZONOS3_MAX_FRAMES"))
            cp.max_frames = atoi(m);
        ctx_ = zonos3_tts_init_from_file(p.model.c_str(), cp);
        if (!ctx_)
            return false;
        default_voice_ = p.tts_voice;
        if (!apply_voice(p.tts_voice, p.tts_voice_dir)) {
            shutdown();
            return false;
        }
        return true;
    }

    void shutdown() override {
        if (ctx_) {
            zonos3_tts_free(ctx_);
            ctx_ = nullptr;
        }
        voice_.clear();
    }

    const char* name() const override { return "zonos3"; }
    uint32_t capabilities() const override { return CAP_TTS | CAP_TEMPERATURE; }
    int input_sample_rate() const override { return 16000; }
    int tts_sample_rate() const override { return 44100; }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};
        zonos3_tts_set_seed(ctx_, (uint32_t)p.seed);
        if (!getenv("CRISPASR_ZONOS3_GREEDY"))
            zonos3_tts_set_temperature(ctx_, p.temperature > 0.0f ? p.temperature : -1.0f);
        if (!apply_voice(p.tts_voice.empty() ? default_voice_ : p.tts_voice, p.tts_voice_dir))
            return {};
        int n = 0;
        float* pcm = zonos3_tts_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        zonos3_tts_free_audio(pcm);
        return out;
    }

    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }

private:
    bool apply_voice(const std::string& voice, const std::string& dir) {
        if (voice == voice_)
            return true;
        const std::string path = resolve_embedding(voice, dir);
        if (!voice.empty() && path.empty()) {
            fprintf(stderr, "crispasr[zonos3]: voice '%s' not found (expected <name>.embedding.npy)\n", voice.c_str());
            return false;
        }
        const int rc = path.empty() ? zonos3_tts_set_speaker_embedding(ctx_, nullptr, 0)
                                    : zonos3_tts_load_speaker_embedding(ctx_, path.c_str());
        if (rc != 0)
            return false;
        voice_ = voice;
        return true;
    }

    zonos3_tts_context* ctx_ = nullptr;
    std::string default_voice_, voice_;
};

std::unique_ptr<CrispasrBackend> crispasr_create_zonos3_backend() {
    return std::make_unique<Zonos3Backend>();
}
