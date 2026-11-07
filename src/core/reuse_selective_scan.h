#pragma once

// CPU reference for the RE-USE Mamba-1 selective-scan boundary. Delta is
// already softplus-transformed and A is already negative-exponentiated.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace reuse_selective_scan {
inline bool product(size_t a, size_t b, size_t& out) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        return false;
    out = a * b;
    return true;
}
inline bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

// u, delta and z: [batch, channel, time]; A: [channel, state];
// B and C: [batch, time, state]; D: [channel]. Initial state is zero.
// Output has the same layout as u. Failure leaves both outputs unchanged.
inline bool run(size_t batches, size_t channels, size_t times, size_t states, const std::vector<float>& u,
                const std::vector<float>& delta, const std::vector<float>& a, const std::vector<float>& b,
                const std::vector<float>& c, const std::vector<float>& d, const std::vector<float>& z,
                std::vector<float>& output, std::vector<float>& final_state) {
    size_t bc = 0, sequence = 0, state_count = 0, factors = 0, a_count = 0;
    if (&output == &final_state || !batches || !channels || !times || !states || !product(batches, channels, bc) ||
        !product(bc, times, sequence) || !product(bc, states, state_count) || !product(batches, times, factors) ||
        !product(factors, states, factors) || !product(channels, states, a_count) || u.size() != sequence ||
        delta.size() != sequence || z.size() != sequence || a.size() != a_count || b.size() != factors ||
        c.size() != factors || d.size() != channels || !finite(u) || !finite(delta) || !finite(a) || !finite(b) ||
        !finite(c) || !finite(d) || !finite(z) ||
        std::any_of(delta.begin(), delta.end(), [](float value) { return value < 0; }) ||
        std::any_of(a.begin(), a.end(), [](float value) { return value > 0; }))
        return false;
    std::vector<float> result(sequence), state(state_count, 0.0f);
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t t = 0; t < times; ++t) {
                const size_t idx = (batch * channels + channel) * times + t;
                float y = 0;
                for (size_t s = 0; s < states; ++s) {
                    const size_t state_idx = (batch * channels + channel) * states + s;
                    const size_t factor_idx = (batch * times + t) * states + s;
                    state[state_idx] = std::exp(delta[idx] * a[channel * states + s]) * state[state_idx] +
                                       (delta[idx] * u[idx]) * b[factor_idx];
                    y += state[state_idx] * c[factor_idx];
                }
                y += d[channel] * u[idx];
                const float gate = z[idx] / (1.0f + std::exp(-z[idx]));
                result[idx] = y * gate;
            }
        }
    }
    if (!finite(result) || !finite(state))
        return false;
    output = std::move(result);
    final_state = std::move(state);
    return true;
}
} // namespace reuse_selective_scan
