#pragma once

// CPU reference port of the qwen_fa_ultra 0.1.1 boundary head at model
// revision 765dc1d714e4c80196782a5df5bfbcb3e1416736. This isolated stage is
// not wired into an aligner or GPU backend. Weights use PyTorch row-major
// layouts so they can be compared directly against the original checkpoint.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace qwen3_ultra_head {
constexpr int dim = 128;
constexpr int qwen_dim = 1024;
constexpr int feature_dim = 91;
constexpr int frames = 41;
constexpr int roles = 4;
constexpr int offsets = 13;

// Build the timestamp/current/previous/next roles expected by the upstream
// head. Hidden states are [word, start/end, qwen_dim] in word-major order.
inline bool build_context(const float* hidden, size_t hidden_count, int n_words, int word, int endpoint,
                          std::array<float, roles * qwen_dim>& context, std::array<float, roles>& mask) {
    if (!hidden || n_words <= 0 || word < 0 || word >= n_words || endpoint < 0 || endpoint > 1 ||
        static_cast<size_t>(n_words) > hidden_count / qwen_dim / 2)
        return false;
    context.fill(0.0f);
    mask = {1.0f, 1.0f, word > 0 ? 1.0f : 0.0f, word + 1 < n_words ? 1.0f : 0.0f};
    const size_t base = static_cast<size_t>(word) * 2 * qwen_dim;
    for (int d = 0; d < qwen_dim; ++d) {
        context[d] = hidden[base + endpoint * qwen_dim + d];
        context[qwen_dim + d] = (hidden[base + d] + hidden[base + qwen_dim + d]) * 0.5f;
        if (word > 0)
            context[2 * qwen_dim + d] = (hidden[base - 2 * qwen_dim + d] + hidden[base - qwen_dim + d]) * 0.5f;
        if (word + 1 < n_words)
            context[3 * qwen_dim + d] = (hidden[base + 2 * qwen_dim + d] + hidden[base + 3 * qwen_dim + d]) * 0.5f;
    }
    return true;
}

struct AttentionWeights {
    std::vector<float> in_weight, in_bias, out_weight, out_bias;
    bool valid() const {
        return in_weight.size() == 3 * dim * dim && in_bias.size() == 3 * dim && out_weight.size() == dim * dim &&
               out_bias.size() == dim;
    }
};
struct Weights {
    std::vector<float> audio0_weight, audio0_bias, audio2_weight, audio2_bias;
    std::vector<float> context_weight, context_bias, role_embedding, endpoint_embedding;
    AttentionWeights context_attention, audio_attention;
    std::vector<float> query_norm_weight, query_norm_bias, audio_norm_weight, audio_norm_bias;
    std::vector<float> activity_weight, activity_bias, boundary_weight, boundary_bias, derivative_scale, prior;
    bool valid() const {
        return audio0_weight.size() == dim * feature_dim * 5 && audio0_bias.size() == dim &&
               audio2_weight.size() == dim * dim * 5 && audio2_bias.size() == dim &&
               context_weight.size() == dim * qwen_dim && context_bias.size() == dim &&
               role_embedding.size() == roles * dim && endpoint_embedding.size() == 2 * dim &&
               context_attention.valid() && audio_attention.valid() && query_norm_weight.size() == dim &&
               query_norm_bias.size() == dim && audio_norm_weight.size() == dim && audio_norm_bias.size() == dim &&
               activity_weight.size() == dim && activity_bias.size() == 1 && boundary_weight.size() == dim &&
               boundary_bias.size() == 1 && derivative_scale.size() == 1 && prior.size() == offsets;
    }
};

inline void linear(const float* input, int inputs, const float* weights, const float* bias, int outputs, float* out) {
    for (int o = 0; o < outputs; ++o) {
        float value = bias ? bias[o] : 0.0f;
        for (int i = 0; i < inputs; ++i)
            value += input[i] * weights[o * inputs + i];
        out[o] = value;
    }
}

inline void layer_norm(float* data, const std::vector<float>& weight, const std::vector<float>& bias) {
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i)
        mean += data[i];
    mean /= dim;
    float variance = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float v = data[i] - mean;
        variance += v * v;
    }
    const float scale = 1.0f / std::sqrt(variance / dim + 1e-5f);
    for (int i = 0; i < dim; ++i)
        data[i] = (data[i] - mean) * scale * weight[i] + bias[i];
}

inline bool attention(const float* query, const float* tokens, const float* mask, int count, const AttentionWeights& w,
                      float* result) {
    std::array<float, dim> q{};
    linear(query, dim, w.in_weight.data(), w.in_bias.data(), dim, q.data());
    std::vector<float> keys(count * dim), values(count * dim);
    for (int t = 0; t < count; ++t) {
        linear(tokens + t * dim, dim, w.in_weight.data() + dim * dim, w.in_bias.data() + dim, dim,
               keys.data() + t * dim);
        linear(tokens + t * dim, dim, w.in_weight.data() + 2 * dim * dim, w.in_bias.data() + 2 * dim, dim,
               values.data() + t * dim);
    }
    std::array<float, dim> combined{};
    for (int h = 0; h < 4; ++h) {
        std::vector<float> score(count);
        float maximum = -INFINITY;
        for (int t = 0; t < count; ++t) {
            if (mask && mask[t] == 0.0f) {
                score[t] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < 32; ++d)
                dot += q[h * 32 + d] * keys[t * dim + h * 32 + d];
            score[t] = dot / std::sqrt(32.0f);
            maximum = std::max(maximum, score[t]);
        }
        if (!std::isfinite(maximum))
            return false;
        float sum = 0.0f;
        for (auto& x : score) {
            x = std::exp(x - maximum);
            sum += x;
        }
        for (int t = 0; t < count; ++t)
            for (int d = 0; d < 32; ++d)
                combined[h * 32 + d] += score[t] / sum * values[t * dim + h * 32 + d];
    }
    linear(combined.data(), dim, w.out_weight.data(), w.out_bias.data(), dim, result);
    return true;
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

inline void conv(const float* input, int input_channels, const std::vector<float>& weight,
                 const std::vector<float>& bias, float* output) {
    // Input/output are time-major. The reference Conv1d weights are [out,in,5].
    for (int t = 0; t < frames; ++t) {
        for (int o = 0; o < dim; ++o) {
            float value = bias[o];
            for (int c = 0; c < input_channels; ++c) {
                for (int k = 0; k < 5; ++k) {
                    const int source = t + k - 2;
                    if (source >= 0 && source < frames)
                        value += input[source * input_channels + c] * weight[(o * input_channels + c) * 5 + k];
                }
            }
            output[t * dim + o] = gelu(value);
        }
    }
}

inline bool run(const Weights& w, const float* features, const float* context, const float* context_mask, int endpoint,
                std::array<float, offsets>& output) {
    if (!w.valid() || !features || !context || !context_mask || endpoint < 0 || endpoint > 1)
        return false;
    for (int i = 0; i < frames * feature_dim; ++i)
        if (!std::isfinite(features[i]))
            return false;
    for (int i = 0; i < roles * qwen_dim; ++i)
        if (!std::isfinite(context[i]))
            return false;
    for (int i = 0; i < roles; ++i)
        if (!std::isfinite(context_mask[i]))
            return false;
    std::array<float, roles * dim> tokens{};
    for (int r = 0; r < roles; ++r) {
        linear(context + r * qwen_dim, qwen_dim, w.context_weight.data(), w.context_bias.data(), dim,
               tokens.data() + r * dim);
        for (int d = 0; d < dim; ++d)
            tokens[r * dim + d] += w.role_embedding[r * dim + d];
    }
    std::array<float, dim> base{}, query{}, attended{};
    for (int d = 0; d < dim; ++d)
        base[d] = tokens[d] + w.endpoint_embedding[endpoint * dim + d];
    if (!attention(base.data(), tokens.data(), context_mask, roles, w.context_attention, query.data()))
        return false;
    for (int d = 0; d < dim; ++d)
        query[d] += base[d];
    layer_norm(query.data(), w.query_norm_weight, w.query_norm_bias);
    std::array<float, frames * dim> audio1{}, audio2{};
    conv(features, feature_dim, w.audio0_weight, w.audio0_bias, audio1.data());
    conv(audio1.data(), dim, w.audio2_weight, w.audio2_bias, audio2.data());
    if (!attention(query.data(), audio2.data(), nullptr, frames, w.audio_attention, attended.data()))
        return false;
    for (int d = 0; d < dim; ++d)
        query[d] += attended[d];
    layer_norm(query.data(), w.query_norm_weight, w.query_norm_bias);
    std::array<float, frames> activity{};
    for (int t = 0; t < frames; ++t) {
        float* a = audio2.data() + t * dim;
        layer_norm(a, w.audio_norm_weight, w.audio_norm_bias);
        float v = w.activity_bias[0];
        for (int d = 0; d < dim; ++d)
            v += a[d] * (query[d] / std::sqrt(static_cast<float>(dim)) + w.activity_weight[d]);
        activity[t] = v;
    }
    for (int k = 0; k < offsets; ++k) {
        const int t = 14 + k;
        const float* a = audio2.data() + t * dim;
        float v = w.boundary_bias[0];
        for (int d = 0; d < dim; ++d)
            v += a[d] * (query[d] / std::sqrt(static_cast<float>(dim)) + w.boundary_weight[d]);
        const float derivative = activity[t] - activity[t - 1];
        output[k] = v + w.derivative_scale[0] * (endpoint == 0 ? derivative : -derivative) + w.prior[k];
        if (!std::isfinite(output[k]))
            return false;
    }
    return true;
}
} // namespace qwen3_ultra_head
