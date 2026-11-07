#pragma once

// Isolated CPU reference composition for RE-USE. It is deliberately not wired
// into CLI/session dispatch and is not optimized for practical long audio.
#include "reuse_frontend.h"
#include "reuse_model_loader.h"

namespace reuse_reference_network {
struct Trace {
    reuse_dense_encoder::Tensor encoder, block0, block14, block29;
};

inline bool run(const reuse_model_loader::Model& model, const std::vector<float>& audio, int sample_rate,
                std::vector<float>& output, Trace* trace = nullptr) {
    if (model.encoder.size() != 6 || model.blocks.size() != 30)
        return false;
    try {
        reuse_frontend::Spectrum spectrum;
        if (!reuse_frontend::analyze(audio, sample_rate, spectrum))
            return false;
        size_t size = 0;
        if (!reuse_dense_encoder::product(spectrum.frames, spectrum.bins, size) || size > SIZE_MAX / 2)
            return false;
        reuse_dense_encoder::Tensor input{2, spectrum.frames, spectrum.bins, std::vector<float>(2 * size)};
        std::copy(spectrum.magnitude.begin(), spectrum.magnitude.end(), input.values.begin());
        std::copy(spectrum.phase.begin(), spectrum.phase.end(), input.values.begin() + size);
        reuse_dense_encoder::Tensor state;
        if (!reuse_dense_encoder::encode(input, model.encoder, state))
            return false;
        Trace captured;
        if (trace)
            captured.encoder = state;
        for (size_t i = 0; i < model.blocks.size(); ++i) {
            reuse_dense_encoder::Tensor next;
            if (!reuse_tf_mamba::run(model.blocks[i], state, next))
                return false;
            state = std::move(next);
            if (trace && i == 0)
                captured.block0 = state;
            if (trace && i == 14)
                captured.block14 = state;
            if (trace && i == 29)
                captured.block29 = state;
        }
        reuse_dense_encoder::Tensor magnitude_features, phase_features, amplitude, real, imaginary;
        if (!reuse_decoder::decode_features(state, model.magnitude.dense, model.magnitude.frequency,
                                            model.magnitude.time, magnitude_features) ||
            !reuse_decoder::decode_features(state, model.phase.dense, model.phase.frequency, model.phase.time,
                                            phase_features) ||
            !reuse_decoder::pointwise(magnitude_features, model.magnitude.first_head, model.magnitude.first_bias,
                                      amplitude) ||
            !reuse_decoder::pointwise(phase_features, model.phase.first_head, model.phase.first_bias, real) ||
            !reuse_decoder::pointwise(phase_features, model.phase.second_head, model.phase.second_bias, imaginary) ||
            amplitude.time < spectrum.frames || amplitude.freq < spectrum.bins || real.time != amplitude.time ||
            real.freq != amplitude.freq || imaginary.time != amplitude.time || imaginary.freq != amplitude.freq)
            return false;
        const size_t stride = spectrum.bins + spectrum.bins % 2;
        if (!reuse_dense_encoder::product(spectrum.frames, stride, size))
            return false;
        std::vector<float> amp(size), re(size), im(size);
        for (size_t t = 0; t < spectrum.frames; ++t)
            for (size_t f = 0; f < spectrum.bins; ++f) {
                amp[t * stride + f] = amplitude.values[t * amplitude.freq + f];
                re[t * stride + f] = real.values[t * real.freq + f];
                im[t * stride + f] = imaginary.values[t * imaginary.freq + f];
            }
        std::vector<float> result;
        if (!reuse_frontend::synthesize(spectrum, amp, re, im, audio.size(), result))
            return false;
        if (trace)
            *trace = std::move(captured);
        output = std::move(result);
        return true;
    } catch (...) {
        return false;
    }
}
} // namespace reuse_reference_network
