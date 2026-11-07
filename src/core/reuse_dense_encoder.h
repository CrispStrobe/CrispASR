#pragma once

// Model-free CPU reference for RE-USE's dense encoder. This is a parity
// boundary, not a wired backend or an optimized inference path.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace reuse_dense_encoder {
struct Tensor {
    size_t channels = 0, time = 0, freq = 0;
    std::vector<float> values; // [channel, time, frequency]
};
struct Layer {
    size_t in_channels = 0, out_channels = 0, kernel_time = 0, kernel_freq = 0;
    std::vector<float> weights, bias, norm_weight, norm_bias, slope;
};

inline bool product(size_t a, size_t b, size_t& out) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        return false;
    out = a * b;
    return true;
}
inline bool finite(const std::vector<float>& x) {
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
inline bool valid(const Tensor& x) {
    size_t n = 0;
    return x.channels && x.time && x.freq && product(x.channels, x.time, n) && product(n, x.freq, n) &&
           x.values.size() == n && finite(x.values);
}

// Conv2d followed by per-channel InstanceNorm2d (biased variance, eps=1e-5)
// and channel-wise PReLU. Padding is symmetric and expressed in input cells.
inline bool conv_norm(const Tensor& input, const Layer& layer, size_t stride_t, size_t stride_f, size_t dilation_t,
                      size_t pad_t, size_t pad_f, Tensor& output) {
    if (!valid(input) || !stride_t || !stride_f || !dilation_t || layer.in_channels != input.channels ||
        !layer.out_channels || !layer.kernel_time || !layer.kernel_freq || !finite(layer.weights) ||
        !finite(layer.bias) || !finite(layer.norm_weight) || !finite(layer.norm_bias) || !finite(layer.slope))
        return false;
    size_t weight_count = 0, effective = 0;
    if (!product(layer.out_channels, layer.in_channels, weight_count) ||
        !product(weight_count, layer.kernel_time, weight_count) ||
        !product(weight_count, layer.kernel_freq, weight_count) || weight_count != layer.weights.size() ||
        layer.bias.size() != layer.out_channels || layer.norm_weight.size() != layer.out_channels ||
        layer.norm_bias.size() != layer.out_channels || layer.slope.size() != layer.out_channels ||
        !product(layer.kernel_time - 1, dilation_t, effective) || effective == SIZE_MAX ||
        pad_t > (SIZE_MAX - input.time) / 2 || pad_f > (SIZE_MAX - input.freq) / 2)
        return false;
    const size_t padded_t = input.time + 2 * pad_t, padded_f = input.freq + 2 * pad_f;
    ++effective;
    if (padded_t < effective || padded_f < layer.kernel_freq)
        return false;
    Tensor result;
    result.channels = layer.out_channels;
    result.time = 1 + (padded_t - effective) / stride_t;
    result.freq = 1 + (padded_f - layer.kernel_freq) / stride_f;
    size_t count = 0, plane = 0;
    if (!product(result.time, result.freq, plane) || !product(result.channels, plane, count))
        return false;
    result.values.resize(count);
    for (size_t o = 0; o < result.channels; ++o) {
        for (size_t t = 0; t < result.time; ++t) {
            for (size_t f = 0; f < result.freq; ++f) {
                float sum = layer.bias[o];
                for (size_t c = 0; c < input.channels; ++c)
                    for (size_t kt = 0; kt < layer.kernel_time; ++kt)
                        for (size_t kf = 0; kf < layer.kernel_freq; ++kf) {
                            const size_t it = t * stride_t + kt * dilation_t, iff = f * stride_f + kf;
                            if (it < pad_t || iff < pad_f || it - pad_t >= input.time || iff - pad_f >= input.freq)
                                continue;
                            const size_t wi =
                                ((o * input.channels + c) * layer.kernel_time + kt) * layer.kernel_freq + kf;
                            sum += input.values[(c * input.time + it - pad_t) * input.freq + iff - pad_f] *
                                   layer.weights[wi];
                        }
                result.values[(o * result.time + t) * result.freq + f] = sum;
            }
        }
        double mean = 0, variance = 0;
        for (size_t i = 0; i < plane; ++i)
            mean += result.values[o * plane + i];
        mean /= plane;
        for (size_t i = 0; i < plane; ++i) {
            const double delta = result.values[o * plane + i] - mean;
            variance += delta * delta;
        }
        const double scale = layer.norm_weight[o] / std::sqrt(variance / plane + 1e-5);
        for (size_t i = 0; i < plane; ++i) {
            float value = static_cast<float>((result.values[o * plane + i] - mean) * scale + layer.norm_bias[o]);
            result.values[o * plane + i] = value >= 0 ? value : value * layer.slope[o];
        }
    }
    if (!finite(result.values))
        return false;
    output = std::move(result);
    return true;
}

inline bool encode(const Tensor& input, const std::vector<Layer>& layers, Tensor& output,
                   std::vector<Tensor>* stages = nullptr) {
    if (!valid(input) || input.channels != 2 || layers.size() != 6 || input.time > SIZE_MAX - 2 ||
        input.freq > SIZE_MAX - 2)
        return false;
    Tensor padded;
    padded.channels = 2;
    padded.time = input.time + 2;
    padded.freq = input.freq + 2;
    size_t size = 0;
    if (!product(padded.channels, padded.time, size) || !product(size, padded.freq, size))
        return false;
    padded.values.assign(size, 0.0f);
    for (size_t c = 0; c < 2; ++c)
        for (size_t t = 0; t < input.time; ++t)
            for (size_t f = 0; f < input.freq; ++f)
                padded.values[(c * padded.time + t) * padded.freq + f] =
                    input.values[(c * input.time + t) * input.freq + f];
    Tensor current;
    if (!conv_norm(padded, layers[0], 1, 1, 1, 0, 0, current))
        return false;
    std::vector<Tensor> captured;
    if (stages)
        captured.push_back(current);
    Tensor skip = current;
    for (size_t i = 0; i < 4; ++i) {
        const size_t dilation = size_t(1) << i;
        if (layers[i + 1].kernel_time != 3 || layers[i + 1].kernel_freq != 3 ||
            !conv_norm(skip, layers[i + 1], 1, 1, dilation, dilation, 1, current) || current.time != skip.time ||
            current.freq != skip.freq)
            return false;
        if (stages)
            captured.push_back(current);
        if (skip.channels > SIZE_MAX - current.channels || skip.values.size() > SIZE_MAX - current.values.size())
            return false;
        Tensor combined;
        combined.channels = current.channels + skip.channels;
        combined.time = skip.time;
        combined.freq = skip.freq;
        combined.values.reserve(current.values.size() + skip.values.size());
        combined.values.insert(combined.values.end(), current.values.begin(), current.values.end());
        combined.values.insert(combined.values.end(), skip.values.begin(), skip.values.end());
        skip = std::move(combined);
    }
    if (layers[0].kernel_time != 1 || layers[0].kernel_freq != 1 || layers[5].kernel_time != 1 ||
        layers[5].kernel_freq != 3 || !conv_norm(current, layers[5], 4, 2, 1, 0, 0, padded))
        return false;
    if (stages) {
        captured.push_back(padded);
        *stages = std::move(captured);
    }
    output = std::move(padded);
    return true;
}
} // namespace reuse_dense_encoder
