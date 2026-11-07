#pragma once

// CPU reference for one RE-USE time/frequency Mamba block. The input and
// output are mono feature tensors [channel, time, frequency].
#include "reuse_bimamba.h"
#include "reuse_dense_encoder.h"

namespace reuse_tf_mamba {
struct Weights {
    reuse_bimamba::Weights time, frequency;
};

inline bool run(const Weights& w, const reuse_dense_encoder::Tensor& input, reuse_dense_encoder::Tensor& output,
                reuse_dense_encoder::Tensor* after_time = nullptr) {
    if (!reuse_dense_encoder::valid(input) || input.channels != w.time.forward.model ||
        input.channels != w.frequency.forward.model)
        return false;
    const size_t channels = input.channels, times = input.time, freqs = input.freq;
    std::vector<float> time_input(input.values.size());
    for (size_t f = 0; f < freqs; ++f)
        for (size_t t = 0; t < times; ++t)
            for (size_t c = 0; c < channels; ++c)
                time_input[(f * times + t) * channels + c] = input.values[(c * times + t) * freqs + f];
    std::vector<float> time_output;
    if (!reuse_bimamba::run(w.time, freqs, times, time_input, time_output))
        return false;
    std::vector<float> freq_input(input.values.size());
    for (size_t t = 0; t < times; ++t)
        for (size_t f = 0; f < freqs; ++f)
            for (size_t c = 0; c < channels; ++c) {
                const size_t in = (f * times + t) * channels + c;
                freq_input[(t * freqs + f) * channels + c] = time_input[in] + time_output[in];
            }
    std::vector<float> freq_output;
    if (!reuse_bimamba::run(w.frequency, times, freqs, freq_input, freq_output))
        return false;
    reuse_dense_encoder::Tensor result{channels, times, freqs, std::vector<float>(input.values.size())};
    for (size_t c = 0; c < channels; ++c)
        for (size_t t = 0; t < times; ++t)
            for (size_t f = 0; f < freqs; ++f) {
                const size_t in = (t * freqs + f) * channels + c;
                result.values[(c * times + t) * freqs + f] = freq_input[in] + freq_output[in];
            }
    if (!reuse_dense_encoder::finite(result.values))
        return false;
    if (after_time) {
        reuse_dense_encoder::Tensor captured{channels, times, freqs, std::vector<float>(input.values.size())};
        for (size_t c = 0; c < channels; ++c)
            for (size_t t = 0; t < times; ++t)
                for (size_t f = 0; f < freqs; ++f)
                    captured.values[(c * times + t) * freqs + f] = freq_input[(t * freqs + f) * channels + c];
        *after_time = std::move(captured);
    }
    output = std::move(result);
    return true;
}
} // namespace reuse_tf_mamba
