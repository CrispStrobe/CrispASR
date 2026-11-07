#pragma once

// CPU reference for the RE-USE decoder's dense block and subpixel upsampling.
// Not an optimized runtime or a complete speech-enhancement backend.
#include "reuse_dense_encoder.h"

namespace reuse_decoder {
using reuse_dense_encoder::Layer;
using reuse_dense_encoder::Tensor;

struct UpLayer {
    size_t channels = 0, scale = 0;
    // Convolution: [scale * channels, channels, 1, 3]. Normalization and
    // activation are applied after pixel shuffle, with one value per channel.
    std::vector<float> weights, bias, norm_weight, norm_bias, slope;
};

inline bool dense(const Tensor& input, const std::vector<Layer>& layers, Tensor& output) {
    if (!reuse_dense_encoder::valid(input) || layers.size() != 4)
        return false;
    Tensor current, skip = input;
    for (size_t i = 0; i < 4; ++i) {
        const size_t dilation = size_t(1) << i;
        if (layers[i].kernel_time != 3 || layers[i].kernel_freq != 3 ||
            !reuse_dense_encoder::conv_norm(skip, layers[i], 1, 1, dilation, dilation, 1, current) ||
            current.channels != input.channels || current.time != input.time || current.freq != input.freq)
            return false;
        if (skip.channels > SIZE_MAX - current.channels || skip.values.size() > SIZE_MAX - current.values.size())
            return false;
        Tensor combined{skip.channels + current.channels, input.time, input.freq, {}};
        combined.values.reserve(current.values.size() + skip.values.size());
        combined.values.insert(combined.values.end(), current.values.begin(), current.values.end());
        combined.values.insert(combined.values.end(), skip.values.begin(), skip.values.end());
        skip = std::move(combined);
    }
    output = std::move(current);
    return true;
}

inline bool upsample(const Tensor& input, const UpLayer& layer, Tensor& output) {
    using reuse_dense_encoder::finite;
    using reuse_dense_encoder::product;
    if (!reuse_dense_encoder::valid(input) || !layer.channels || !layer.scale || layer.channels != input.channels)
        return false;
    size_t expanded_channels = 0, output_freq = 0, weight_count = 0, output_count = 0, plane = 0;
    if (!product(layer.channels, layer.scale, expanded_channels) || !product(input.freq, layer.scale, output_freq) ||
        !product(expanded_channels, layer.channels, weight_count) || !product(weight_count, 3, weight_count) ||
        !product(input.time, output_freq, plane) || !product(layer.channels, plane, output_count) ||
        layer.weights.size() != weight_count || layer.bias.size() != expanded_channels ||
        layer.norm_weight.size() != layer.channels || layer.norm_bias.size() != layer.channels ||
        layer.slope.size() != layer.channels || !finite(layer.weights) || !finite(layer.bias) ||
        !finite(layer.norm_weight) || !finite(layer.norm_bias) || !finite(layer.slope))
        return false;
    Tensor result{layer.channels, input.time, output_freq, std::vector<float>(output_count)};
    for (size_t channel = 0; channel < layer.channels; ++channel) {
        for (size_t t = 0; t < input.time; ++t) {
            for (size_t f = 0; f < input.freq; ++f) {
                for (size_t sub = 0; sub < layer.scale; ++sub) {
                    const size_t out_channel = sub * layer.channels + channel;
                    float value = layer.bias[out_channel];
                    for (size_t in = 0; in < layer.channels; ++in)
                        for (size_t k = 0; k < 3; ++k) {
                            if ((k == 0 && f == 0) || (k == 2 && f == input.freq - 1))
                                continue;
                            const size_t source = k == 0 ? f - 1 : k == 1 ? f : f + 1;
                            value += input.values[(in * input.time + t) * input.freq + source] *
                                     layer.weights[(out_channel * layer.channels + in) * 3 + k];
                        }
                    result.values[(channel * input.time + t) * output_freq + f * layer.scale + sub] = value;
                }
            }
        }
        double mean = 0, variance = 0;
        for (size_t i = 0; i < plane; ++i)
            mean += result.values[channel * plane + i];
        mean /= plane;
        for (size_t i = 0; i < plane; ++i) {
            const double delta = result.values[channel * plane + i] - mean;
            variance += delta * delta;
        }
        const double scale = layer.norm_weight[channel] / std::sqrt(variance / plane + 1e-5);
        for (size_t i = 0; i < plane; ++i) {
            float value =
                static_cast<float>((result.values[channel * plane + i] - mean) * scale + layer.norm_bias[channel]);
            result.values[channel * plane + i] = value >= 0 ? value : value * layer.slope[channel];
        }
    }
    if (!finite(result.values))
        return false;
    output = std::move(result);
    return true;
}

inline bool transpose(const Tensor& input, Tensor& output) {
    if (!reuse_dense_encoder::valid(input))
        return false;
    Tensor result{input.channels, input.freq, input.time, std::vector<float>(input.values.size())};
    for (size_t c = 0; c < input.channels; ++c)
        for (size_t t = 0; t < input.time; ++t)
            for (size_t f = 0; f < input.freq; ++f)
                result.values[(c * input.freq + f) * input.time + t] =
                    input.values[(c * input.time + t) * input.freq + f];
    output = std::move(result);
    return true;
}

inline bool decode_features(const Tensor& input, const std::vector<Layer>& layers, const UpLayer& up_freq,
                            const UpLayer& up_time, Tensor& output) {
    Tensor current, swapped, expanded;
    if (up_freq.scale != 2 || up_time.scale != 4 || !dense(input, layers, current) ||
        !upsample(current, up_freq, expanded) || !transpose(expanded, swapped) ||
        !upsample(swapped, up_time, expanded) || !transpose(expanded, current))
        return false;
    output = std::move(current);
    return true;
}

inline bool pointwise(const Tensor& input, const std::vector<float>& weights, float bias, Tensor& output) {
    if (!reuse_dense_encoder::valid(input) || weights.size() != input.channels ||
        !reuse_dense_encoder::finite(weights) || !std::isfinite(bias))
        return false;
    size_t plane = 0;
    if (!reuse_dense_encoder::product(input.time, input.freq, plane))
        return false;
    Tensor result{1, input.time, input.freq, std::vector<float>(plane)};
    for (size_t i = 0; i < plane; ++i) {
        float value = bias;
        for (size_t c = 0; c < input.channels; ++c)
            value += weights[c] * input.values[c * plane + i];
        result.values[i] = value;
    }
    if (!reuse_dense_encoder::finite(result.values))
        return false;
    output = std::move(result);
    return true;
}
} // namespace reuse_decoder
