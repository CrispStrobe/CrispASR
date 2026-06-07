// crispasr_backend_piper.cpp — adapter for rhasspy/piper VITS TTS.
// Single GGUF, 250+ community voices across 30+ languages.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "crispasr_model_registry.h"
#include "whisper_params.h"

#include "piper_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class PiperBackend : public CrispasrBackend {
public:
    ~PiperBackend() override {
        if (ctx_)
            piper_tts_free(ctx_);
    }

    const char* name() const override { return "piper"; }

    uint32_t capabilities() const override { return CAP_TTS; }

    bool init(const whisper_params& p) override {
        std::string model_path = p.model;
        if (model_path.empty()) {
            fprintf(stderr, "piper: no model path specified (use -m <path.gguf>)\n");
            return false;
        }

        struct piper_tts_params pp = piper_tts_default_params();
        pp.n_threads = p.n_threads;
        pp.verbosity = p.print_progress ? 2 : 1;
        pp.noise_scale = -1.0f; // use model defaults
        pp.length_scale = -1.0f;
        pp.noise_w = -1.0f;

        ctx_ = piper_tts_init_from_file(model_path.c_str(), pp);
        if (!ctx_) {
            fprintf(stderr, "piper: failed to load model: %s\n", model_path.c_str());
            return false;
        }

        // Apply voice/language override (skip "auto" — not a valid espeak voice)
        if (!p.language.empty() && p.language != "auto") {
            piper_tts_set_language(ctx_, p.language.c_str());
        }

        // G2P dictionary source: "olaph" (MIT), "open-dict" (CC-BY-SA), or file path
        if (!p.g2p_dict.empty()) {
            piper_tts_set_g2p_dict(p.g2p_dict.c_str());
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};

        // Apply per-call parameters
        if (p.tts_speed > 0.0f) {
            piper_tts_set_length_scale(ctx_, 1.0f / p.tts_speed);
        }
        // 75c-opt-2: native piper knobs
        if (p.tts_noise_scale >= 0.0f)
            piper_tts_set_noise_scale(ctx_, p.tts_noise_scale);
        if (p.tts_noise_w >= 0.0f)
            piper_tts_set_noise_w(ctx_, p.tts_noise_w);
        if (p.tts_speaker_id >= 0)
            piper_tts_set_speaker_id(ctx_, p.tts_speaker_id);

        float* pcm = nullptr;
        int sr = 0;

        // Try full text→phoneme→synth (espeak-ng or built-in G2P).
        int n = piper_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm) {
            // Phonemization failed. If input looks like IPA (non-ASCII),
            // try direct phoneme synthesis.
            bool has_ipa_chars = false;
            for (unsigned char c : text) {
                if (c >= 0x80) {
                    has_ipa_chars = true;
                    break;
                }
            }
            if (has_ipa_chars) {
                n = piper_tts_synthesize_phonemes(ctx_, text.c_str(), &pcm, &sr);
            } else {
                fprintf(stderr, "piper: phonemization failed for '%s'\n", text.c_str());
            }
        }
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> out(pcm, pcm + n);
        free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? piper_tts_sample_rate(ctx_) : 22050; }

    std::vector<crispasr_segment> transcribe(const float* /*pcm*/, int /*n*/, int64_t /*t0*/,
                                             const whisper_params& /*p*/) override {
        return {}; // TTS-only backend
    }

    void shutdown() override {
        if (ctx_) {
            piper_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    piper_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_piper_backend() {
    return std::unique_ptr<CrispasrBackend>(new PiperBackend());
}
