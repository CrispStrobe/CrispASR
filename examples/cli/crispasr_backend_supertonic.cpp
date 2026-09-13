// crispasr_backend_supertonic.cpp — adapter for Supertonic-3 TTS (#434).
//
// Non-autoregressive flow-matching TTS (Supertone/supertonic-3, OpenRAIL-M),
// 44.1 kHz, 31 languages, 10 preset voices embedded in the GGUF (F1..F5,
// M1..M5). No voice cloning — the open release ships fixed style presets.
//
// Usage:
//   crispasr --backend supertonic -m auto --tts "Hello world." -o out.wav
//   crispasr --backend supertonic -m auto --tts "Guten Tag." -l de \
//            --voice F2 --tts-speed 1.2 --tts-steps 8 -o out.wav

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "supertonic_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class SupertonicBackend : public CrispasrBackend {
public:
    SupertonicBackend() = default;
    ~SupertonicBackend() override { SupertonicBackend::shutdown(); }

    const char* name() const override { return "supertonic"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_AUTO_DOWNLOAD | CAP_TTS_SPEED; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n*/, int64_t /*t_off*/,
                                             const whisper_params& /*params*/) override {
        return {}; // TTS-only backend
    }

    bool init(const whisper_params& p) override {
        supertonic_context_params sp = supertonic_context_default_params();
        sp.n_threads = p.n_threads;
        sp.verbosity = p.no_prints ? 0 : 1;
        sp.use_gpu = p.use_gpu;
        sp.seed = (uint64_t)(p.seed > 0 ? p.seed : 0);
        if (p.tts_num_steps > 0)
            sp.total_steps = p.tts_num_steps;
        if (p.tts_speed > 0.0f && p.tts_speed != 1.0f) {
            // upstream default is 1.05; treat --tts-speed as a multiplier on it
            sp.speed = 1.05f * p.tts_speed;
        }
        ctx_ = supertonic_init_from_file(p.model.c_str(), sp);
        if (!ctx_) {
            std::fprintf(stderr, "crispasr[supertonic]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        apply_language(p);
        apply_voice(p);
        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_ || text.empty())
            return {};
        apply_language(p);
        apply_voice(p);
        if (p.tts_num_steps > 0)
            supertonic_set_total_steps(ctx_, p.tts_num_steps);
        if (p.tts_speed > 0.0f)
            supertonic_set_speed(ctx_, 1.05f * p.tts_speed);
        int n = 0;
        float* pcm = supertonic_synthesize(ctx_, text.c_str(), &n);
        if (!pcm || n <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n);
        std::free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? supertonic_sample_rate(ctx_) : 44100; }

    void shutdown() override {
        if (ctx_) {
            supertonic_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    void apply_language(const whisper_params& p) {
        if (p.language.empty() || p.language == "auto")
            return;
        if (supertonic_set_language(ctx_, p.language.c_str()) != 0 && !p.no_prints && p.language != last_warn_lang_) {
            std::fprintf(stderr,
                         "crispasr[supertonic]: language '%s' not supported (en ko ja ar bg cs da de el es et fi fr "
                         "hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi); keeping current\n",
                         p.language.c_str());
            last_warn_lang_ = p.language;
        }
    }
    // --voice selects a PRESET name (F1..F5, M1..M5); a path is rejected with
    // a hint since this backend has no cloning.
    void apply_voice(const whisper_params& p) {
        if (p.tts_voice.empty() || p.tts_voice == last_voice_)
            return;
        if (supertonic_set_voice(ctx_, p.tts_voice.c_str()) != 0) {
            if (!p.no_prints)
                std::fprintf(stderr,
                             "crispasr[supertonic]: unknown voice '%s' — expected a preset name F1..F5 / M1..M5 "
                             "(Supertonic-3 has no voice cloning); keeping '%s'\n",
                             p.tts_voice.c_str(), last_voice_.empty() ? "M1" : last_voice_.c_str());
        } else {
            last_voice_ = p.tts_voice;
        }
    }

    supertonic_context* ctx_ = nullptr;
    std::string last_voice_;
    std::string last_warn_lang_;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_supertonic_backend() {
    return std::make_unique<SupertonicBackend>();
}
