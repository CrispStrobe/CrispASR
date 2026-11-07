#pragma once

// CPU reference for RE-USE's bidirectional Mamba block, including its inner
// residuals, projection and layer normalization. The outer T/F residual is
// applied by the caller.
#include "reuse_mamba.h"

namespace reuse_bimamba {
struct Weights {
    reuse_mamba::Weights forward, backward;
    std::vector<float> projection, projection_bias, norm_weight, norm_bias;
};

inline bool run(const Weights& w, size_t batches, size_t times, const std::vector<float>& input,
                std::vector<float>& output) {
    using reuse_selective_scan::finite;
    using reuse_selective_scan::product;
    const size_t model = w.forward.model;
    size_t sequence = 0, elements = 0, projection_size = 0;
    if (!batches || !times || !model || w.backward.model != model || model > std::numeric_limits<size_t>::max() / 2 ||
        !product(batches, times, sequence) || !product(sequence, model, elements) ||
        !product(model, 2 * model, projection_size) || input.size() != elements ||
        w.projection.size() != projection_size || w.projection_bias.size() != model || w.norm_weight.size() != model ||
        w.norm_bias.size() != model || !finite(input) || !finite(w.projection) || !finite(w.projection_bias) ||
        !finite(w.norm_weight) || !finite(w.norm_bias))
        return false;
    std::vector<float> reverse(elements);
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t t = 0; t < times; ++t)
            for (size_t i = 0; i < model; ++i)
                reverse[(batch * times + t) * model + i] = input[(batch * times + times - 1 - t) * model + i];
    std::vector<float> forward, backward;
    if (!reuse_mamba::run(w.forward, batches, times, input, forward) ||
        !reuse_mamba::run(w.backward, batches, times, reverse, backward))
        return false;
    std::vector<float> result(elements);
    for (size_t batch = 0; batch < batches; ++batch)
        for (size_t t = 0; t < times; ++t) {
            const size_t p = (batch * times + t) * model;
            const size_t reverse_p = (batch * times + times - 1 - t) * model;
            for (size_t o = 0; o < model; ++o) {
                float value = w.projection_bias[o];
                for (size_t i = 0; i < model; ++i) {
                    value += w.projection[o * 2 * model + i] * (forward[p + i] + input[p + i]);
                    value += w.projection[o * 2 * model + model + i] * (backward[reverse_p + i] + input[p + i]);
                }
                result[p + o] = value;
            }
            double mean = 0, variance = 0;
            for (size_t i = 0; i < model; ++i)
                mean += result[p + i];
            mean /= model;
            for (size_t i = 0; i < model; ++i) {
                const double delta = result[p + i] - mean;
                variance += delta * delta;
            }
            const double scale = 1.0 / std::sqrt(variance / model + 1e-5);
            for (size_t i = 0; i < model; ++i)
                result[p + i] = static_cast<float>((result[p + i] - mean) * scale * w.norm_weight[i] + w.norm_bias[i]);
        }
    if (!finite(result))
        return false;
    output = std::move(result);
    return true;
}
} // namespace reuse_bimamba
