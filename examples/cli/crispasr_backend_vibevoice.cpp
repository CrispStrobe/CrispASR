// crispasr_backend_vibevoice.cpp — adapter for Microsoft VibeVoice-ASR.
//
// The runtime itself expects 24 kHz mono PCM. The unified CrispASR CLI
// standardizes on 16 kHz audio input, so this adapter performs the same
// simple linear 16k -> 24k upsample that other 24 kHz backends use.

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "vibevoice.h"

#include <cstdio>
#include <vector>

namespace {

static std::vector<float> resample_16k_to_24k(const float* in, int n_in) {
    std::vector<float> out;
    if (!in || n_in <= 0)
        return out;

    const int n_out = (int)((double)n_in * 24000.0 / 16000.0);
    out.resize((size_t)n_out);
    for (int i = 0; i < n_out; ++i) {
        const double pos = (double)i * 16000.0 / 24000.0;
        int i0 = (int)pos;
        int i1 = i0 + 1;
        if (i0 < 0)
            i0 = 0;
        if (i1 >= n_in)
            i1 = n_in - 1;
        const float frac = (float)(pos - (double)i0);
        out[(size_t)i] = in[i0] * (1.0f - frac) + in[i1] * frac;
    }
    return out;
}

class VibeVoiceBackend : public CrispasrBackend {
public:
    VibeVoiceBackend() = default;
    ~VibeVoiceBackend() override { VibeVoiceBackend::shutdown(); }

    const char* name() const override { return "vibevoice"; }

    uint32_t capabilities() const override {
        return CAP_TIMESTAMPS_CTC | CAP_AUTO_DOWNLOAD | CAP_TEMPERATURE | CAP_FLASH_ATTN | CAP_TTS;
    }

    bool init(const whisper_params& p) override {
        vibevoice_context_params cp = vibevoice_context_default_params();
        cp.n_threads = p.n_threads;
        cp.max_new_tokens = p.max_new_tokens > 0 ? p.max_new_tokens : cp.max_new_tokens;
        cp.verbosity = p.no_prints ? 0 : 1;
        cp.use_gpu = crispasr_backend_should_use_gpu(p);
        cp.tts_steps = p.tts_steps;
        ctx_ = vibevoice_init_from_file(p.model.c_str(), cp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[vibevoice]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& /*params*/) override {
        std::vector<crispasr_segment> out;
        if (!ctx_ || !samples || n_samples <= 0)
            return out;

        const std::vector<float> pcm24 = resample_16k_to_24k(samples, n_samples);
        char* text = vibevoice_transcribe(ctx_, pcm24.data(), (int)pcm24.size());
        if (!text)
            return out;

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples * 100.0 / 16000.0);
        out.push_back(std::move(seg));
        std::free(text);
        return out;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& params) override {
        if (!voice_loaded_ && !params.tts_voice.empty()) {
            if (vibevoice_load_voice(ctx_, params.tts_voice.c_str()) == 0)
                voice_loaded_ = true;
        }
        if (!ctx_ || text.empty())
            return {};
        int n_samples = 0;
        float* pcm = vibevoice_synthesize(ctx_, text.c_str(), &n_samples);
        if (!pcm || n_samples <= 0)
            return {};
        std::vector<float> out(pcm, pcm + n_samples);
        std::free(pcm);
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            vibevoice_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    vibevoice_context* ctx_ = nullptr;
    bool voice_loaded_ = false;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_vibevoice_backend() {
    return std::unique_ptr<CrispasrBackend>(new VibeVoiceBackend());
}
