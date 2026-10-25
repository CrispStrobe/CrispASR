#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Original CSM uses Llama-3 frequency scaling in both transformer stacks.
// The pinned original-operator fixture is produced by
// tools/ci-heavy/csm_original_rope_reference.py. ggml expects a per-pair
// DIVISOR of its unscaled inverse frequency, rather than inverse frequencies.
namespace csm_rope {

inline bool valid_parameters(int dim, float theta, float factor, float low, float high, uint32_t original_context) {
    return dim > 0 && dim <= 4096 && dim % 2 == 0 && original_context > 0 && std::isfinite(theta) && theta > 0 &&
           std::isfinite(factor) && factor >= 1 && std::isfinite(low) && low > 0 && std::isfinite(high) && high > low;
}

inline std::vector<float> frequency_factors(int dim, float theta, float factor, float low, float high,
                                            uint32_t original_context) {
    if (!valid_parameters(dim, theta, factor, low, high, original_context))
        return {};
    std::vector<float> result((std::size_t)dim / 2, 1.0f);
    if (factor == 1.0f)
        return result;
    const float old_context = (float)original_context;
    const float low_wavelength = old_context / low;
    const float high_wavelength = old_context / high;
    if (!std::isfinite(low_wavelength) || !std::isfinite(high_wavelength))
        return {};
    for (int i = 0; i < dim / 2; ++i) {
        // Preserve the original reciprocal-of-positive-power convention.
        const float inverse = 1.0f / std::pow(theta, 2.0f * (float)i / (float)dim);
        const float wavelength = 6.2831853071795864769f / inverse;
        float scaled = inverse;
        if (wavelength > low_wavelength) {
            scaled = inverse / factor;
        } else if (wavelength >= high_wavelength) {
            const float smooth = (old_context / wavelength - low) / (high - low);
            scaled = (1.0f - smooth) * inverse / factor + smooth * inverse;
        }
        if (!std::isfinite(inverse) || inverse <= 0 || !std::isfinite(scaled) || scaled <= 0)
            return {};
        result[(std::size_t)i] = inverse / scaled;
        if (!std::isfinite(result[(std::size_t)i]) || result[(std::size_t)i] <= 0)
            return {};
    }
    return result;
}

} // namespace csm_rope
