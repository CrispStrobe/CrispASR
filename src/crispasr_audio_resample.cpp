// In-memory access to the shared CLI Kaiser polyphase resampler.
#include "crispasr.h"
#include "core/audio_resample.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

int crispasr_audio_resample(const float* pcm, int n_samples, int source_rate, int target_rate, float** out_pcm,
                            int* out_samples) {
    if (out_pcm)
        *out_pcm = nullptr;
    if (out_samples)
        *out_samples = 0;
    // Bound the filter as well as the output: co-prime arbitrary integer rates
    // can otherwise ask build_filter for billions of taps.
    if (!out_pcm || !out_samples || n_samples < 0 || (!pcm && n_samples != 0) || source_rate <= 0 || target_rate <= 0 ||
        source_rate > 384000 || target_rate > 384000)
        return -1;
    const int64_t count = (int64_t(n_samples) * target_rate + source_rate - 1) / source_rate;
    if (count > int64_t(n_samples) * 64 || count > INT_MAX || uint64_t(count) > SIZE_MAX / sizeof(float))
        return -1;
    if (n_samples == 0)
        return 0;
    try {
        const auto samples = core_audio::resample_polyphase(pcm, n_samples, source_rate, target_rate);
        if (samples.size() != size_t(count))
            return -1;
        auto* output = static_cast<float*>(std::malloc(samples.size() * sizeof(float)));
        if (!output)
            return -2;
        std::memcpy(output, samples.data(), samples.size() * sizeof(float));
        *out_pcm = output;
        *out_samples = static_cast<int>(samples.size());
        return 0;
    } catch (...) {
        // C/FFI callers must never receive a C++ allocation exception.
        return -2;
    }
}
