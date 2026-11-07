#pragma once

// Pinned NVIDIA NemotronLabs RE-USE STFT frontend, isolated port stage.
// This is not yet connected to a model runtime or a user-facing backend.
#include <complex>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace reuse_frontend {
struct Spectrum {
    int sample_rate = 0;
    int n_fft = 0;
    int hop = 0;
    size_t frames = 0;
    size_t bins = 0;
    // Each array is time-major [frames, bins]. Magnitudes are log1p-compressed.
    std::vector<float> magnitude;
    std::vector<float> phase;
};

// Mixed-radix complex FFT. Computing the transform in double reduces error for
// non-power-of-two FFT sizes before matching the upstream float32 spectrum.
inline void fft(std::vector<std::complex<double>>& data) {
    const size_t n = data.size();
    if (n <= 1)
        return;
    size_t radix = 2;
    while (radix < n && n % radix != 0)
        ++radix;
    const size_t width = n / radix;
    std::vector<std::vector<std::complex<double>>> parts(radix, std::vector<std::complex<double>>(width));
    for (size_t r = 0; r < radix; ++r) {
        for (size_t j = 0; j < width; ++j)
            parts[r][j] = data[j * radix + r];
        fft(parts[r]);
    }
    const double pi = std::acos(-1.0);
    for (size_t k = 0; k < n; ++k) {
        std::complex<double> value(0.0, 0.0);
        for (size_t r = 0; r < radix; ++r) {
            const double angle = -2.0 * pi * static_cast<double>(r * k) / n;
            value += parts[r][k % width] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        data[k] = value;
    }
}

inline bool analyze(const std::vector<float>& audio, int sample_rate, Spectrum& output) {
    if (sample_rate < 8000 || sample_rate > 48000 || audio.size() > static_cast<size_t>(INT64_MAX / 2))
        return false;
    for (float sample : audio)
        if (!std::isfinite(sample))
            return false;
    int n_fft = 320 * sample_rate / 8000;
    int hop = 40 * sample_rate / 8000;
    n_fft += n_fft % 2;
    hop += hop % 2;
    if (audio.size() <= static_cast<size_t>(n_fft / 2))
        return false;
    const size_t frames = 1 + audio.size() / static_cast<size_t>(hop);
    const size_t bins = static_cast<size_t>(n_fft / 2 + 1);
    if (frames > std::numeric_limits<size_t>::max() / bins)
        return false;
    Spectrum result;
    result.sample_rate = sample_rate;
    result.n_fft = n_fft;
    result.hop = hop;
    result.frames = frames;
    result.bins = bins;
    result.magnitude.resize(frames * bins);
    result.phase.resize(frames * bins);
    std::vector<std::complex<double>> spectrum(static_cast<size_t>(n_fft));
    std::vector<float> window(n_fft);
    const float pi = std::acos(-1.0f);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5f - 0.5f * std::cos(2.0f * pi * i / n_fft);
    const int64_t length = static_cast<int64_t>(audio.size());
    for (size_t frame = 0; frame < frames; ++frame) {
        const int64_t first = static_cast<int64_t>(frame * hop) - n_fft / 2;
        for (int i = 0; i < n_fft; ++i) {
            int64_t index = first + i;
            if (index < 0)
                index = -index;
            else if (index >= length)
                index = 2 * length - 2 - index;
            const float windowed = audio[static_cast<size_t>(index)] * window[i];
            spectrum[i] = std::complex<double>(windowed, 0.0);
        }
        fft(spectrum);
        for (size_t k = 0; k < bins; ++k) {
            const float re = static_cast<float>(spectrum[k].real()), im = static_cast<float>(spectrum[k].imag());
            result.magnitude[frame * bins + k] = std::log1p(std::hypot(re, im));
            result.phase[frame * bins + k] = std::atan2(im, re);
        }
    }
    output = std::move(result);
    return true;
}
} // namespace reuse_frontend
