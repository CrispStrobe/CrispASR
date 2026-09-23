#pragma once
// torchaudio.functional.resample (default method "sinc_interp_hann",
// lowpass_filter_width=6, rolloff=0.99) for any rational rate pair.
//
// A port of _get_sinc_resample_kernel + _apply_sinc_resample_kernel, not an
// approximation: the kernel is built with the same float32 arithmetic
// torchaudio uses for a float32 waveform, the waveform is zero-padded
// (width, width + orig) and convolved with stride `orig`, and the output is
// cut to ceil(new * length / orig). Models whose reference pipeline resamples
// (Raon-Speech: 16k -> 24k in the processor, 24k -> 16k in the encoder) need
// this to match their mel features, not a generic resampler.

#include <cmath>
#include <numeric>
#include <vector>

namespace core_torchaudio {

struct SincKernel {
    int orig = 1;  // orig_freq / gcd
    int nw = 1;    // new_freq / gcd
    int width = 0; // one-sided filter support, in input samples
    // nw rows of (2 * width + orig) taps, row-major
    std::vector<float> taps;
    int n_taps() const { return 2 * width + orig; }
};

inline SincKernel sinc_resample_kernel(int orig_freq, int new_freq, int lowpass_filter_width = 6,
                                       float rolloff = 0.99f) {
    SincKernel k;
    const int g = std::gcd(orig_freq, new_freq);
    k.orig = orig_freq / g;
    k.nw = new_freq / g;
    const float base_freq = (float)std::min(k.orig, k.nw) * rolloff;
    k.width = (int)std::ceil((double)lowpass_filter_width * k.orig / base_freq);
    const int n = k.n_taps();
    k.taps.assign((size_t)k.nw * n, 0.0f);
    const float lw = (float)lowpass_filter_width;
    const float pi = 3.14159265358979323846f;
    const float scale = base_freq / (float)k.orig;
    for (int p = 0; p < k.nw; p++) {
        // t = arange(0, -new, -1)[p] / new + arange(-width, width + orig)[j] / orig
        const float tp = (float)(-p) / (float)k.nw;
        for (int j = 0; j < n; j++) {
            float t = tp + (float)(j - k.width) / (float)k.orig;
            t *= base_freq;
            t = std::fmax(-lw, std::fmin(lw, t));
            float w = std::cos(t * pi / lw / 2.0f);
            w *= w;
            t *= pi;
            const float s = (t == 0.0f) ? 1.0f : std::sin(t) / t;
            k.taps[(size_t)p * n + j] = s * w * scale;
        }
    }
    return k;
}

inline std::vector<float> resample(const float* x, int length, const SincKernel& k) {
    if (length <= 0)
        return {};
    if (k.orig == k.nw)
        return std::vector<float>(x, x + length);
    const int n = k.n_taps();
    const int padded = length + 2 * k.width + k.orig;
    std::vector<float> xp((size_t)padded, 0.0f);
    std::copy(x, x + length, xp.begin() + k.width);
    const int n_out_frames = (padded - n) / k.orig + 1;
    const long long target = (long long)std::ceil((double)k.nw * length / k.orig);
    std::vector<float> out((size_t)target, 0.0f);
    for (int i = 0; i < n_out_frames; i++) {
        const float* src = xp.data() + (size_t)i * k.orig;
        for (int p = 0; p < k.nw; p++) {
            const long long o = (long long)i * k.nw + p;
            if (o >= target)
                break;
            const float* tap = k.taps.data() + (size_t)p * n;
            float acc = 0.0f;
            for (int j = 0; j < n; j++)
                acc += tap[j] * src[j];
            out[(size_t)o] = acc;
        }
    }
    return out;
}

inline std::vector<float> resample(const float* x, int length, int orig_freq, int new_freq) {
    return resample(x, length, sinc_resample_kernel(orig_freq, new_freq));
}

} // namespace core_torchaudio
