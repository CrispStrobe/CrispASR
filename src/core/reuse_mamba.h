#pragma once

// Model-free CPU reference for a single forward Mamba-1 block. This is not an
// optimized runtime or an end-to-end RE-USE backend.
#include "reuse_selective_scan.h"

namespace reuse_mamba {
struct Weights {
    size_t model = 0, inner = 0, rank = 0, states = 0, kernel = 0;
    // Linear matrices are row-major [output, input]. Convolution weights are
    // [inner, kernel], with the same order as PyTorch Conv1d.
    std::vector<float> in_proj, conv, conv_bias, x_proj, dt_proj, dt_bias, a_log, d, out_proj;
};

inline bool run(const Weights& w, size_t batches, size_t times, const std::vector<float>& input,
                std::vector<float>& output) {
    using reuse_selective_scan::finite;
    using reuse_selective_scan::product;
    if (!w.model || !w.inner || !w.rank || !w.states || !w.kernel || !batches || !times ||
        w.inner > (std::numeric_limits<size_t>::max() / 2) ||
        w.states > (std::numeric_limits<size_t>::max() - w.rank) / 2)
        return false;
    const size_t factors = w.rank + 2 * w.states;
    size_t seq = 0, input_count = 0, xz_count = 0, factor_count = 0, scan_count = 0, a_count = 0;
    size_t in_count = 0, conv_count = 0, x_count = 0, dt_count = 0, out_count = 0, bc_count = 0;
    if (!product(batches, times, seq) || !product(seq, w.model, input_count) || !product(seq, 2 * w.inner, xz_count) ||
        !product(seq, factors, factor_count) || !product(batches, w.inner, scan_count) ||
        !product(scan_count, times, scan_count) || !product(seq, w.states, bc_count) ||
        !product(w.inner, w.states, a_count) || !product(2 * w.inner, w.model, in_count) ||
        !product(w.inner, w.kernel, conv_count) || !product(factors, w.inner, x_count) ||
        !product(w.inner, w.rank, dt_count) || !product(w.model, w.inner, out_count) || input.size() != input_count ||
        w.in_proj.size() != in_count || w.conv.size() != conv_count || w.conv_bias.size() != w.inner ||
        w.x_proj.size() != x_count || w.dt_proj.size() != dt_count || w.dt_bias.size() != w.inner ||
        w.a_log.size() != a_count || w.d.size() != w.inner || w.out_proj.size() != out_count || !finite(input) ||
        !finite(w.in_proj) || !finite(w.conv) || !finite(w.conv_bias) || !finite(w.x_proj) || !finite(w.dt_proj) ||
        !finite(w.dt_bias) || !finite(w.a_log) || !finite(w.d) || !finite(w.out_proj))
        return false;
    std::vector<float> xz(xz_count), u(scan_count), z(scan_count), features(factor_count), delta(scan_count),
        b(bc_count), c(bc_count), a(a_count);
    for (size_t p = 0; p < seq; ++p)
        for (size_t o = 0; o < 2 * w.inner; ++o) {
            float sum = 0;
            for (size_t i = 0; i < w.model; ++i)
                sum += w.in_proj[o * w.model + i] * input[p * w.model + i];
            xz[p * (2 * w.inner) + o] = sum;
        }
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t channel = 0; channel < w.inner; ++channel)
            for (size_t t = 0; t < times; ++t) {
                float value = w.conv_bias[channel];
                for (size_t k = 0; k < w.kernel; ++k) {
                    // Conv1d padding is kernel-1 and the reference truncates
                    // to the original time length: only causal inputs remain.
                    const size_t lag = w.kernel - 1 - k;
                    if (t < lag)
                        continue;
                    const size_t source_t = t - lag;
                    value += w.conv[channel * w.kernel + k] * xz[(batch * times + source_t) * (2 * w.inner) + channel];
                }
                const size_t idx = (batch * w.inner + channel) * times + t;
                u[idx] = value / (1.0f + std::exp(-value));
                z[idx] = xz[(batch * times + t) * (2 * w.inner) + w.inner + channel];
            }
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t t = 0; t < times; ++t) {
            const size_t p = batch * times + t;
            for (size_t o = 0; o < factors; ++o) {
                float sum = 0;
                for (size_t channel = 0; channel < w.inner; ++channel)
                    sum += w.x_proj[o * w.inner + channel] * u[(batch * w.inner + channel) * times + t];
                features[p * factors + o] = sum;
            }
            for (size_t s = 0; s < w.states; ++s) {
                b[p * w.states + s] = features[p * factors + w.rank + s];
                c[p * w.states + s] = features[p * factors + w.rank + w.states + s];
            }
            for (size_t channel = 0; channel < w.inner; ++channel) {
                float value = w.dt_bias[channel];
                for (size_t r = 0; r < w.rank; ++r)
                    value += w.dt_proj[channel * w.rank + r] * features[p * factors + r];
                delta[(batch * w.inner + channel) * times + t] = value > 20.0f ? value : std::log1p(std::exp(value));
            }
        }
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = -std::exp(w.a_log[i]);
    std::vector<float> scan, state;
    if (!reuse_selective_scan::run(batches, w.inner, times, w.states, u, delta, a, b, c, w.d, z, scan, state))
        return false;
    std::vector<float> result(input_count);
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t t = 0; t < times; ++t)
            for (size_t o = 0; o < w.model; ++o) {
                float sum = 0;
                for (size_t channel = 0; channel < w.inner; ++channel)
                    sum += w.out_proj[o * w.inner + channel] * scan[(batch * w.inner + channel) * times + t];
                result[(batch * times + t) * w.model + o] = sum;
            }
    if (!finite(result))
        return false;
    output = std::move(result);
    return true;
}
} // namespace reuse_mamba
