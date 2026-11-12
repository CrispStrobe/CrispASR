#pragma once

// Pinned NVIDIA NemotronLabs RE-USE STFT frontend, isolated port stage.
// This is not yet connected to a model runtime or a user-facing backend.
#include <algorithm>
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

// Optional observations before log-magnitude/atan2. These never replace the raw
// phase fed to the model: equivalent complex values can have different phases.
struct Diagnostics {
    std::vector<float> window;
    std::vector<float> real;
    std::vector<float> imaginary;
};

// PyTorch's periodic Hann computes the angular step once in float before
// multiplying the sample index. Rounding 2*pi*i before the division changes
// window values and can change raw phase near the atan2 branch cut.
inline std::vector<float> periodic_hann(int n_fft) {
    if (n_fft <= 0)
        return {};
    std::vector<float> window(static_cast<size_t>(n_fft));
    const float step = static_cast<float>(2.0 * std::acos(-1.0) / n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] = 0.5f - 0.5f * std::cos(step * static_cast<float>(i));
    return window;
}

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

inline bool analyze(const std::vector<float>& audio, int sample_rate, Spectrum& output,
                    Diagnostics* diagnostics = nullptr) {
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
    const auto window = periodic_hann(n_fft);
    Diagnostics observed;
    if (diagnostics) {
        observed.window = window;
        observed.real.resize(frames * bins);
        observed.imaginary.resize(frames * bins);
    }
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
            const float re = static_cast<float>(spectrum[k].real());
            // A real-to-complex FFT defines the DC and Nyquist imaginary parts
            // as positive zero. Tiny roundoff in a complex FFT changes atan2
            // from +pi to -pi when the endpoint real value is negative.
            const float im = (k == 0 || k + 1 == bins) ? 0.0f : static_cast<float>(spectrum[k].imag());
            if (diagnostics) {
                observed.real[frame * bins + k] = re;
                observed.imaginary[frame * bins + k] = im;
            }
            result.magnitude[frame * bins + k] = std::log1p(std::hypot(re, im));
            result.phase[frame * bins + k] = std::atan2(im, re);
        }
    }
    output = std::move(result);
    if (diagnostics)
        *diagnostics = std::move(observed);
    return true;
}

// Reconstruct the waveform from the three network heads. The network pads its
// frequency dimension to an even number; only the original STFT bins count.
inline bool synthesize(const Spectrum& spec, const std::vector<float>& amplitude, const std::vector<float>& real,
                       const std::vector<float>& imaginary, size_t samples, std::vector<float>& output) {
    if (spec.sample_rate < 8000 || spec.sample_rate > 48000 || spec.n_fft <= 0 || spec.n_fft % 2 || spec.hop <= 0 ||
        spec.bins != static_cast<size_t>(spec.n_fft / 2 + 1) || spec.frames < 2)
        return false;
    const size_t padded_bins = spec.bins + spec.bins % 2;
    if (spec.frames > std::numeric_limits<size_t>::max() / padded_bins ||
        spec.frames - 1 > (std::numeric_limits<size_t>::max() - spec.n_fft) / spec.hop ||
        amplitude.size() != spec.frames * padded_bins || real.size() != amplitude.size() ||
        imaginary.size() != amplitude.size())
        return false;
    const size_t reconstructed = (spec.frames - 1) * static_cast<size_t>(spec.hop);
    if (samples < reconstructed || samples - reconstructed >= static_cast<size_t>(spec.hop))
        return false;
    for (size_t i = 0; i < amplitude.size(); ++i)
        if (!std::isfinite(amplitude[i]) || !std::isfinite(real[i]) || !std::isfinite(imaginary[i]))
            return false;
    std::vector<double> sum(reconstructed + spec.n_fft), envelope(sum.size());
    std::vector<std::complex<double>> spectrum(spec.n_fft);
    const auto window = periodic_hann(spec.n_fft);
    for (size_t frame = 0; frame < spec.frames; ++frame) {
        size_t zeros = 0;
        for (size_t bin = 0; bin < spec.bins; ++bin)
            zeros += amplitude[frame * padded_bins + bin] <= 0.0f;
        for (size_t bin = 0; bin < spec.bins; ++bin) {
            const size_t index = frame * padded_bins + bin;
            const float magnitude = zeros * 2 > spec.bins ? 0.0f : std::expm1(std::max(0.0f, amplitude[index]));
            if (!std::isfinite(magnitude))
                return false;
            const float phase = std::atan2(imaginary[index], real[index]);
            spectrum[bin] = std::conj(std::complex<double>(magnitude * std::cos(phase), magnitude * std::sin(phase)));
        }
        for (size_t bin = spec.bins; bin < spectrum.size(); ++bin)
            spectrum[bin] = std::conj(spectrum[spectrum.size() - bin]);
        fft(spectrum);
        for (int i = 0; i < spec.n_fft; ++i) {
            const size_t index = frame * spec.hop + i;
            sum[index] += spectrum[i].real() * window[i] / spec.n_fft;
            envelope[index] += double(window[i]) * window[i];
        }
    }
    std::vector<float> result(samples, 1e-8f);
    for (size_t i = 0; i < reconstructed; ++i) {
        const size_t index = i + spec.n_fft / 2;
        if (envelope[index] <= 1e-11)
            return false;
        result[i] = static_cast<float>(sum[index] / envelope[index]);
    }
    output = std::move(result);
    return true;
}
} // namespace reuse_frontend
