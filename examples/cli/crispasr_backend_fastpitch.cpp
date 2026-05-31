// crispasr_backend_fastpitch.cpp -- adapter for NVIDIA FastPitch TTS.
//
// Non-autoregressive parallel TTS: single-pass mel generation + HiFi-GAN vocoder.
// German multi-speaker model (5 speakers), 22.05 kHz output.
// Section 133 in the CrispASR backend lineup.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "fastpitch_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class FastPitchBackend : public CrispasrBackend {
public:
    ~FastPitchBackend() override { FastPitchBackend::shutdown(); }

    const char* name() const override { return "fastpitch"; }

    uint32_t capabilities() const override { return CAP_TTS; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        fprintf(stderr, "crispasr[fastpitch]: transcription is not supported by this backend\n");
        return {};
    }

    bool init(const whisper_params& p) override {
        fastpitch_tts_params fp = fastpitch_tts_default_params();
        fp.n_threads = p.n_threads;
        fp.verbosity = p.no_prints ? 0 : 1;
        // Speaker selection via --voice <N> (integer string = speaker index).
        if (!p.tts_voice.empty()) {
            char* end = nullptr;
            long sid = strtol(p.tts_voice.c_str(), &end, 10);
            if (end != p.tts_voice.c_str() && *end == '\0' && sid >= 0) {
                fp.speaker_id = (int)sid;
            }
        }
        fp.pace = (p.tts_speed > 0.0f) ? (1.0f / p.tts_speed) : 1.0f;

        ctx_ = fastpitch_tts_init_from_file(p.model.c_str(), fp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[fastpitch]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& p) override {
        if (!ctx_)
            return {};

        // Update per-call parameters
        if (p.tts_speed > 0.0f) {
            fastpitch_tts_set_pace(ctx_, 1.0f / p.tts_speed);
        }
        if (!p.tts_voice.empty()) {
            char* end = nullptr;
            long sid = strtol(p.tts_voice.c_str(), &end, 10);
            if (end != p.tts_voice.c_str() && *end == '\0' && sid >= 0) {
                fastpitch_tts_set_speaker(ctx_, (int)sid);
            }
        }

        float* pcm = nullptr;
        int sr = 0;
        int n = fastpitch_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> result(pcm, pcm + n);
        free(pcm);
        return result;
    }

    int tts_sample_rate() const override { return ctx_ ? fastpitch_tts_sample_rate(ctx_) : 22050; }

    void shutdown() override {
        if (ctx_) {
            fastpitch_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    fastpitch_tts_context* ctx_ = nullptr;
};

} // namespace

// -- Factory registration --

std::unique_ptr<CrispasrBackend> crispasr_make_fastpitch_backend() {
    return std::make_unique<FastPitchBackend>();
}
