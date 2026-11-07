#include "../src/core/qwen3_ultra_head.h"
#include <cmath>
#include <cstdio>
#include <limits>

using namespace qwen3_ultra_head;

static AttentionWeights zero_attention() {
    return {std::vector<float>(3 * dim * dim), std::vector<float>(3 * dim), std::vector<float>(dim * dim),
            std::vector<float>(dim)};
}

static Weights zero_weights() {
    Weights w;
    w.audio0_weight.resize(dim * feature_dim * 5);
    w.audio0_bias.resize(dim);
    w.audio2_weight.resize(dim * dim * 5);
    w.audio2_bias.resize(dim);
    w.context_weight.resize(dim * qwen_dim);
    w.context_bias.resize(dim);
    w.role_embedding.resize(roles * dim);
    w.endpoint_embedding.resize(2 * dim);
    w.context_attention = zero_attention();
    w.audio_attention = zero_attention();
    w.query_norm_weight.resize(dim, 1);
    w.query_norm_bias.resize(dim);
    w.audio_norm_weight.resize(dim, 1);
    w.audio_norm_bias.resize(dim);
    w.activity_weight.resize(dim);
    w.activity_bias.resize(1);
    w.boundary_weight.resize(dim);
    w.boundary_bias.resize(1);
    w.derivative_scale.resize(1, 1);
    for (int i = 0; i < offsets; ++i)
        w.prior.push_back(static_cast<float>(i - 6));
    return w;
}

int main() {
    std::vector<float> hidden(3 * 2 * qwen_dim);
    for (int i = 0; i < 6; ++i)
        std::fill(hidden.begin() + i * qwen_dim, hidden.begin() + (i + 1) * qwen_dim, static_cast<float>(i));
    std::array<float, roles * qwen_dim> role_context{};
    std::array<float, roles> role_mask{};
    if (!build_context(hidden.data(), hidden.size(), 3, 0, 1, role_context, role_mask) ||
        role_mask != std::array<float, roles>{1, 1, 0, 1} || role_context[0] != 1 || role_context[qwen_dim] != 0.5f ||
        role_context[2 * qwen_dim] != 0 || role_context[3 * qwen_dim] != 2.5f)
        return 7;
    if (!build_context(hidden.data(), hidden.size(), 3, 2, 0, role_context, role_mask) ||
        role_mask != std::array<float, roles>{1, 1, 1, 0} || role_context[0] != 4 || role_context[qwen_dim] != 4.5f ||
        role_context[2 * qwen_dim] != 2.5f || role_context[3 * qwen_dim] != 0)
        return 8;
    if (build_context(hidden.data(), hidden.size() - 1, 3, 0, 0, role_context, role_mask) ||
        build_context(hidden.data(), hidden.size(), 3, 3, 0, role_context, role_mask))
        return 9;
    auto w = zero_weights();
    std::array<float, frames * feature_dim> features{};
    std::array<float, roles * qwen_dim> context{};
    std::array<float, roles> mask{1, 1, 0, 0};
    std::array<float, offsets> output{};
    if (!run(w, features.data(), context.data(), mask.data(), 0, output))
        return 1;
    for (int i = 0; i < offsets; ++i)
        if (std::abs(output[i] - w.prior[i]) > 1e-6f)
            return 2;
    mask.fill(0);
    if (run(w, features.data(), context.data(), mask.data(), 0, output))
        return 3;
    mask[0] = 1;
    if (run(w, features.data(), context.data(), mask.data(), 2, output))
        return 4;
    features[0] = std::numeric_limits<float>::quiet_NaN();
    if (run(w, features.data(), context.data(), mask.data(), 0, output))
        return 5;
    features[0] = 0;
    w.audio2_weight.clear();
    if (run(w, features.data(), context.data(), mask.data(), 0, output))
        return 6;
    std::puts("Ultra local-head analytic controls passed");
    return 0;
}
