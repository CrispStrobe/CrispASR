#pragma once

// Candidate for the pinned PyTorch 2.6 MKL CPU F32 multinomial path.
// Diagnostic only until native free-generation and decoded speech pass.
// PyTorch consumes one MT19937 word per contiguous probability tensor,
// then uses Intel's documented MCG31m1 stream and exponential ICDF.
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-vector-statistics-notes/2025-2/mcg31m1.html
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-vector-statistics-notes/2025-2/exponential-vsl-rng-method-exponential-icdf.html
// Scalar libm and MKL vector logarithms may differ in final F32 bits.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

namespace dia_rng {
inline void exponential_batch(float* noise, size_t count, std::mt19937& rng) {
    if (count == 0) {
        return;
    }
    constexpr uint64_t modulus = 2147483647u;
    uint64_t state = rng() % modulus;
    if (state == 0) {
        state = 1;
    }
    for (size_t i = 0; i < count; ++i) {
        const float uniform = static_cast<float>(state) / static_cast<float>(modulus);
        noise[i] = -std::log(uniform) + std::numeric_limits<float>::min();
        state = (state * 1132489760u) % modulus;
    }
}

inline uint32_t exponential_race(const float* probabilities, const float* noise, uint32_t vocabulary) {
    uint32_t winner = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocabulary; ++i) {
        const float score = probabilities[i] / noise[i];
        if (score > maximum) {
            maximum = score;
            winner = i;
        }
    }
    return winner;
}
} // namespace dia_rng
