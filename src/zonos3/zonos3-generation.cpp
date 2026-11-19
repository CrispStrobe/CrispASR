// Vendored from Zyphra/ZONOS3-Mini servers/ggml/src/zonos3-generation.cpp (MIT, see LICENSE.zyphra).
// CrispASR change: the registry is model/src/zonos3/registries/generation.json at upstream a50e141,
// inlined instead of embedded by CMake + parsed with nlohmann::json.
#include "zonos3-generation.h"

#include <cmath>

const zonos3_generation_registry& zonos3_generation() {
    static const zonos3_generation_registry r = [] {
        zonos3_generation_registry out{};
        out.sample_rate = 44100;
        out.sampling.temperature = 0.75f;
        out.sampling.topk = 80;
        out.sampling.top_p = 0.69f;
        out.sampling.min_p = 0.0f;
        out.sampling.repetition_penalty = 2.4f;
        out.sampling.repetition_window = 96;
        out.sampling.repetition_codebooks = 6;
        out.silence_prefill_default_frames = 3;
        out.silence_prefill_max_frames = 30;
        out.decode_context_frames = 30;
        out.reference_max_frames = 900;
        out.max_frames = 2000;
        out.stream_codebooks_default = 1;
        out.stream_codebooks_first_only = true;
        out.reference_audio_max_seconds = 60;
        out.upload_max_seconds = 600;
        out.edit_max_seconds = 150;
        out.output_limit_dbfs = -1.0;
        return out;
    }();
    return r;
}

void zonos3_soft_limit(float* pcm, size_t n) {
    static const float knee = (float)std::pow(10.0, zonos3_generation().output_limit_dbfs / 20.0);
    static const float width = 1.0f - knee;
    for (size_t i = 0; i < n; ++i) {
        const float a = std::fabs(pcm[i]);
        if (a > knee)
            pcm[i] = std::copysign(knee + width * std::tanh((a - knee) / width), pcm[i]);
    }
}
