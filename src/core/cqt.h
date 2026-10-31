// core/cqt.h — constant-Q transform (log-frequency spectrogram).
//
// A CQT bins the spectrum geometrically: bin k is centred at
//   f_k = fmin * 2^(k / bins_per_octave)
// so every bin spans the same number of cents. That is what pitch- and
// chord-oriented models want, and it is why a linear STFT/mel front end will
// not substitute for one.
//
// PRIMARY CONSUMER: BTC chord recognition (§251), which trains on librosa's CQT
// at n_bins=144, bins_per_octave=24, hop=2048, sr=22050. The model degrades if
// the front end does not track librosa closely, so `tools/cqt_librosa_parity.py`
// scores this implementation against librosa and MUST be re-run after any edit
// here.
//
// NOTE: `l1_normalize` also applies librosa's `scale=True` sqrt(N_k) factor —
// see the comment at the normalisation itself. `tools/cqt_librosa_parity.py`
// now asserts per-bin MAGNITUDE as well as shape; a scale-only regression is
// invisible to correlation and peak-bin match, and one shipped here once.
//
// MEASURED vs librosa 0.11.0 (three sustained tones an octave apart, BTC params):
//   per-frame shape correlation  median 0.9999, mean 0.9721, min 0.1136
//   peak-bin exact match         97.6%
// The mean/min are dragged down by exactly three TRANSITION frames plus the tail
// frame: at a tone boundary this reports the NEW pitch where librosa still
// reports the old one, because librosa's recursive per-octave downsampling gives
// each octave a different group delay while these direct kernels are centred
// uniformly. Steady-state frames agree to 0.9999. For 10-second BTC chord
// segments that boundary latency is immaterial; if a consumer ever needs
// frame-exact transition alignment, that is the known gap. Frame count also
// differs by one at the tail (we emit 1 + n/hop, librosa one more) -- align on
// the leading edge, not the trailing.
//
// ALGORITHM. Direct time-domain kernels (Brown 1991), not librosa's recursive
// downsampling. For each bin we build a complex exponential windowed to a
// bin-specific length
//   N_k = ceil(Q * sr / f_k),   Q = filter_scale / (2^(1/bins_per_octave) - 1)
// and correlate it with the signal at each hop. Chosen deliberately over the
// FFT/sparse-kernel route because it is transparent and exactly matches the
// defining equation — the cost is O(n_bins * N_k) per frame, which is fine at
// hop=2048 for offline chord work but is NOT a real-time path. If a hot path
// ever needs this, add the sparse spectral-kernel variant behind a flag rather
// than editing these kernels.
//
// Low bins dominate the cost: N_k for k=0 at fmin=32.7 Hz, bpo=24, sr=22050 is
// ~24k samples, versus ~380 at the top of a 6-octave range.

#pragma once

#include "core/parallel_for.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace core_cqt {

struct Params {
    int sample_rate = 22050;
    float fmin = 32.703195662574829f; // C1
    int n_bins = 144;
    int bins_per_octave = 24;
    int hop_length = 2048;
    float filter_scale = 1.0f;
    // librosa pads the signal by half the LONGEST filter and centres frame t on
    // sample t*hop. Keep true to line up with librosa frame-for-frame.
    bool center = true;
    // librosa's default `norm=1` scales each kernel by its L1 norm, making bin
    // magnitudes comparable across the (wildly different) filter lengths.
    bool l1_normalize = true;
};

// One complex kernel: a windowed complex exponential at the bin's centre freq.
struct Kernel {
    std::vector<float> re;
    std::vector<float> im;
    int length = 0;
};

inline float bin_frequency(const Params& p, int k) {
    return p.fmin * std::pow(2.0f, (float)k / (float)p.bins_per_octave);
}

// Q chosen so every bin spans the same relative bandwidth.
inline float quality_factor(const Params& p) {
    return p.filter_scale / (std::pow(2.0f, 1.0f / (float)p.bins_per_octave) - 1.0f);
}

inline int kernel_length(const Params& p, int k) {
    const float f = bin_frequency(p, k);
    if (f <= 0.0f)
        return 0;
    return (int)std::ceil(quality_factor(p) * (float)p.sample_rate / f);
}

// Build all n_bins kernels. Hann-windowed, matching librosa's default window.
inline std::vector<Kernel> build_kernels(const Params& p) {
    std::vector<Kernel> ks((size_t)p.n_bins);
    const double two_pi = 6.283185307179586476925286766559;
    for (int k = 0; k < p.n_bins; k++) {
        const int N = kernel_length(p, k);
        const double f = (double)bin_frequency(p, k);
        Kernel& kern = ks[(size_t)k];
        kern.length = N;
        kern.re.resize((size_t)N);
        kern.im.resize((size_t)N);
        double l1 = 0.0;
        for (int n = 0; n < N; n++) {
            // Periodic Hann, as torch/librosa use (NOT the symmetric variant --
            // see the mel pitfalls note in docs/contributing.md).
            const double w = 0.5 - 0.5 * std::cos(two_pi * (double)n / (double)N);
            const double ph = two_pi * f * (double)n / (double)p.sample_rate;
            const double re = w * std::cos(ph);
            const double im = w * std::sin(ph);
            kern.re[(size_t)n] = (float)re;
            kern.im[(size_t)n] = (float)im;
            l1 += std::sqrt(re * re + im * im);
        }
        if (p.l1_normalize && l1 > 0.0) {
            // librosa `norm=1` L1-normalises each filter, AND `scale=True`
            // (its default) then divides the response by sqrt(filter length):
            //   librosa/core/constantq.py: `V /= np.sqrt(lengths)`.
            //
            // librosa normalises its basis over a buffer padded to n_fft, so
            // the net difference against an L1 norm over the UNPADDED kernel
            // here is exactly a factor of sqrt(N_k). Measured on white noise
            // (all bins active) the ratio librosa/ours was sqrt(N_k) to within
            // 0.5% across the full range: bin 0 152.52 vs 151.69, bin 60 63.70
            // vs 63.78, bin 143 19.29 vs 19.26.
            //
            // Folding it into the kernel keeps magnitude() untouched. Without
            // it every bin was low by sqrt(N_k) — up to 152x at the bottom —
            // which pushed BTC's features out of the distribution its scalar
            // mean/std assume, so it predicted "no chord" everywhere while the
            // reference predicted real chords.
            const float inv = (float)(std::sqrt((double)N) / l1);
            for (int n = 0; n < N; n++) {
                kern.re[(size_t)n] *= inv;
                kern.im[(size_t)n] *= inv;
            }
        }
    }
    return ks;
}

inline int n_frames(const Params& p, int n_samples) {
    if (n_samples <= 0 || p.hop_length <= 0)
        return 0;
    return 1 + n_samples / p.hop_length;
}

// Compute |CQT|. Output is frame-major: out[t * n_bins + k], length
// n_frames * n_bins. Returns the frame count.
//
// `center=true` centres bin k's kernel on sample t*hop (the kernel spans
// [t*hop - N_k/2, t*hop + N_k/2)), zero-padding outside the signal. Note
// librosa's default pad_mode is 'constant' (zeros) for cqt, which this matches;
// if a consumer needs reflect padding, add it as a Params flag rather than
// changing this default.
inline int magnitude(const Params& p, const std::vector<Kernel>& kernels, const float* x, int n_samples,
                     std::vector<float>& out, int n_threads = 0) {
    const int T = n_frames(p, n_samples);
    if (T <= 0 || !x)
        return 0;
    out.assign((size_t)T * (size_t)p.n_bins, 0.0f);

    const char* env_serial = std::getenv("CRISPASR_CQT_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');

    if (force_serial || T <= 1) {
        for (int t = 0; t < T; t++) {
            const long centre = (long)t * (long)p.hop_length;
            for (int k = 0; k < p.n_bins; k++) {
                const Kernel& kern = kernels[(size_t)k];
                const int N = kern.length;
                if (N <= 0)
                    continue;
                // Correlate: conj(kernel) . signal, so the phase convention matches
                // exp(-i w n) even though the kernel stores exp(+i w n).
                const long start = p.center ? (centre - (long)N / 2) : centre;
                double acc_re = 0.0, acc_im = 0.0;
                for (int n = 0; n < N; n++) {
                    const long s = start + (long)n;
                    if (s < 0 || s >= (long)n_samples)
                        continue;
                    const double v = (double)x[(size_t)s];
                    acc_re += v * (double)kern.re[(size_t)n];
                    acc_im -= v * (double)kern.im[(size_t)n];
                }
                out[(size_t)t * (size_t)p.n_bins + (size_t)k] = (float)std::sqrt(acc_re * acc_re + acc_im * acc_im);
            }
        }
    } else {
        uint64_t work_per_frame = 0;
        for (int k = 0; k < p.n_bins; k++)
            work_per_frame += (uint64_t)std::max(0, std::min(kernels[(size_t)k].length, n_samples)) * 2;
        core_parallel::for_each_chunk_work(T, n_threads, work_per_frame, 65536, [&](int t_start, int t_end) {
            for (int t = t_start; t < t_end; t++) {
                const long centre = (long)t * (long)p.hop_length;
                float* out_frame = out.data() + (size_t)t * (size_t)p.n_bins;
                for (int k = 0; k < p.n_bins; k++) {
                    const Kernel& kern = kernels[(size_t)k];
                    const int N = kern.length;
                    if (N <= 0)
                        continue;
                    const long start = p.center ? (centre - (long)N / 2) : centre;
                    const long n_begin = std::max(0L, -start);
                    const long n_end = std::min((long)N, (long)n_samples - start);
                    double acc_re = 0.0, acc_im = 0.0;
                    if (n_begin < n_end) {
                        const float* xs = x + (start + n_begin);
                        const float* kre = kern.re.data() + n_begin;
                        const float* kim = kern.im.data() + n_begin;
                        const long count = n_end - n_begin;
                        for (long n = 0; n < count; n++) {
                            const double v = (double)xs[n];
                            acc_re += v * (double)kre[n];
                            acc_im -= v * (double)kim[n];
                        }
                    }
                    out_frame[k] = (float)std::sqrt(acc_re * acc_re + acc_im * acc_im);
                }
            }
        });
    }
    return T;
}

// Convenience: build kernels and transform in one call. Prefer the two-step
// form when transforming several signals — the kernels are the expensive part
// and are signal-independent.
inline int magnitude(const Params& p, const float* x, int n_samples, std::vector<float>& out, int n_threads = 0) {
    return magnitude(p, build_kernels(p), x, n_samples, out, n_threads);
}

} // namespace core_cqt
