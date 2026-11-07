#pragma once

// The 41 x 91 fixed-center acoustic feature contract of qwen_fa_ultra 0.1.1
// (model revision 765dc1d714e4c80196782a5df5bfbcb3e1416736). This standalone
// port stage is not wired into the aligner or a public model entrypoint.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace qwen3_ultra_features {
constexpr size_t frames = 41;
constexpr size_t channels = 91;
constexpr size_t bins = 257;
using Features = std::array<std::array<float, channels>, frames>;
using Normalizer = std::array<float, channels>;

inline const std::array<std::array<float, bins>, 80>& mel_bank() {
    static const auto bank = [] {
        std::array<std::array<float, bins>, 80> result{};
        std::array<int, 82> boundaries{};
        const double mel_max = 2595.0 * std::log10(1.0 + 8000.0 / 700.0);
        for (size_t i = 0; i < boundaries.size(); ++i) {
            const double mel = mel_max * i / 81.0;
            const double hz = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
            boundaries[i] = static_cast<int>(std::floor(513.0 * hz / 16000.0));
        }
        for (size_t i = 0; i < result.size(); ++i) {
            const int left = boundaries[i];
            const int center = std::max(boundaries[i + 1], left + 1);
            const int right = std::min(std::max(boundaries[i + 2], center + 1), 256);
            if (center >= static_cast<int>(bins))
                continue;
            float total = 0;
            for (int j = left; j < center; ++j) {
                result[i][j] = static_cast<float>(j - left) / (center - left);
                total += result[i][j];
            }
            for (int j = center; j < right; ++j) {
                result[i][j] = 1.0f - static_cast<float>(j - center) / (right - center);
                total += result[i][j];
            }
            if (total > 0)
                for (auto& v : result[i])
                    v /= total;
        }
        return result;
    }();
    return bank;
}

// NumPy's rfft promotes the float32 windowed input to float64. Preserve that
// precision here: float FFT roundoff is amplified by the log-spectrum features.
inline void fft512(std::array<double, 512>& re, std::array<double, 512>& im) {
    for (int i = 1, j = 0; i < 512; ++i) {
        int bit = 256;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    const double pi = std::acos(-1.0);
    for (int len = 2; len <= 512; len <<= 1) {
        const double wr = std::cos(-2.0 * pi / len), wi = std::sin(-2.0 * pi / len);
        for (int i = 0; i < 512; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int j = 0; j < len / 2; ++j) {
                const double tr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                const double ti = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                re[i + j + len / 2] = re[i + j] - tr;
                im[i + j + len / 2] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
                const double next = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next;
            }
        }
    }
}

inline bool extract(const std::vector<float>& audio, double center_ms, const Normalizer& mean, const Normalizer& stddev,
                    Features& output, bool validate_audio = true) {
    if (!std::isfinite(center_ms) || std::abs(center_ms) > 1.0e12)
        return false;
    if (validate_audio)
        for (float x : audio)
            if (!std::isfinite(x))
                return false;
    for (size_t i = 0; i < channels; ++i)
        if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || stddev[i] <= 0)
            return false;
    const int64_t center = static_cast<int64_t>(std::nearbyint(center_ms * 16.0));
    constexpr float epsilon = 1e-6f;
    const double pi = std::acos(-1.0);
    std::array<std::array<float, bins>, frames> magnitudes{}, normalized{};
    std::array<float, frames> rms{};
    for (size_t t = 0; t < frames; ++t) {
        std::array<double, 512> real{}, imag{};
        const int64_t first = center + (static_cast<int64_t>(t) - 20) * 160 - 200;
        float energy = 0;
        for (int n = 0; n < 400; ++n) {
            const int64_t index = first + n;
            const float sample = index >= 0 && index < static_cast<int64_t>(audio.size()) ? audio[index] : 0.0f;
            const float window = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * pi * n / 399.0));
            const float windowed = sample * window;
            real[n] = windowed;
            energy += windowed * windowed;
        }
        rms[t] = std::log(std::sqrt(energy / 400.0f + epsilon));
        fft512(real, imag);
        float spectrum_mean = 0;
        for (size_t k = 0; k < bins; ++k) {
            magnitudes[t][k] = static_cast<float>(std::hypot(real[k], imag[k]));
            normalized[t][k] = std::log(std::max(magnitudes[t][k], epsilon));
            spectrum_mean += normalized[t][k];
        }
        spectrum_mean /= static_cast<float>(bins);
        for (auto& v : normalized[t])
            v -= spectrum_mean;
        for (size_t m = 0; m < 80; ++m) {
            float power = 0;
            for (size_t k = 0; k < bins; ++k)
                power += mel_bank()[m][k] * magnitudes[t][k] * magnitudes[t][k];
            output[t][m] = std::log(std::max(power, epsilon));
        }
        output[t][80] = rms[t];
        output[t][81] = t ? rms[t] - rms[t - 1] : 0;
        float change = 0, previous = 0;
        if (t) {
            for (size_t k = 0; k < bins; ++k) {
                change += std::max(magnitudes[t][k] - magnitudes[t - 1][k], 0.0f);
                previous += magnitudes[t - 1][k];
            }
        }
        output[t][82] = t ? change / (previous + epsilon) : 0;
    }
    for (size_t t = 0; t < frames; ++t) {
        for (int b = 0; b < 4; ++b) {
            const int left = 1 + 64 * b, right = left + 64;
            float before = 0, after = 0;
            for (int k = left; k < right; ++k) {
                before += std::abs(normalized[t][k] - normalized[t ? t - 1 : t][k]);
                after += std::abs(normalized[t][k] - normalized[t + 1 < frames ? t + 1 : t][k]);
            }
            output[t][83 + b] = before / 64.0f;
            output[t][87 + b] = after / 64.0f;
        }
        for (size_t c = 0; c < channels; ++c) {
            output[t][c] = (output[t][c] - mean[c]) / stddev[c];
            if (!std::isfinite(output[t][c]))
                return false;
        }
    }
    return true;
}
} // namespace qwen3_ultra_features
