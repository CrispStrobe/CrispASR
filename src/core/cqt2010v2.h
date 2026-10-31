// core/cqt2010v2.h — nnAudio's CQT2010v2 (Schörkhuber 2010 recursive-downsampling
// constant-Q), as ported to TensorFlow by Spotify in basic_pitch/layers/nnaudio.py.
//
// WHY THIS EXISTS ALONGSIDE core/cqt.h
// ------------------------------------
// `core/cqt.h` is the DIRECT time-domain kernel CQT (Brown 1991): one kernel per
// output bin, zero padding, `1 + n/hop` frames. It is validated against librosa
// for BTC chord recognition and must not change.
//
// Basic Pitch's front end is a different algorithm with a different group delay,
// and the network was trained on ITS output:
//
//   * a single 36-bin kernel bank covering only the TOP octave,
//   * reused across 9 octaves by recursively decimating the signal x2 with a
//     256-tap FIR and halving the hop each time,
//   * REFLECT padding of n_fft/2 before every octave's correlation (not zeros),
//   * a final per-bin `sqrt(ceil(Q*sr/f_k))` rescale so magnitudes land where
//     librosa's would.
//
// Feeding it `core/cqt.h` output instead gives per-frame cosines well under the
// 0.999 parity gate, so the two coexist rather than one wrapping the other.
//
// KERNELS ARE SUPPLIED, NOT DERIVED. The caller passes the kernel bank, the
// decimation FIR and the sqrt-length vector in. For Basic Pitch they are copied
// bit-for-bit out of the upstream ONNX initializers into the GGUF
// (models/convert-basic-pitch-to-gguf.py), which sidesteps reimplementing
// `scipy.signal.firwin2` — a frequency-sampling FIR design whose output would
// have to match to ~1e-9 for the ninth octave to survive eight passes of it.
//
// COST. n_octaves * n_filters * n_fft MACs per frame plus the decimation chain:
// for Basic Pitch's 2-second window that is ~14M MACs, a few milliseconds.

#pragma once

#include "core/parallel_for.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace core_cqt2010v2 {

struct Params {
    int n_bins = 309;         // output bins (fmin * 2^(k/bins_per_octave))
    int bins_per_octave = 36; // == n_filters for a full-octave kernel bank
    int n_octaves = 9;        // ceil(n_bins / bins_per_octave)
    int n_filters = 36;       // kernels in the top-octave bank
    int n_fft = 256;          // kernel length; reflect pad is n_fft/2
    int hop_length = 256;     // hop at the TOP octave; halves per decimation
};

// All pointers are borrowed and must outlive the call.
struct Kernels {
    const float* real = nullptr;    // [n_filters * n_fft]
    const float* imag = nullptr;    // [n_filters * n_fft]
    const float* lowpass = nullptr; // [lowpass_len] decimation FIR
    int lowpass_len = 256;
    const float* sqrt_lengths = nullptr; // [n_bins], applied after the octave concat
};

// numpy/torch 'reflect': the edge sample is NOT repeated. Requires n > pad.
inline void reflect_pad(const float* x, int n, int pad, std::vector<float>& out) {
    out.resize((size_t)n + 2 * (size_t)pad);
    for (int i = 0; i < pad; i++)
        out[(size_t)i] = x[pad - i]; // x[pad], x[pad-1], ... reversed
    for (int i = 0; i < n; i++)
        out[(size_t)pad + (size_t)i] = x[i];
    for (int i = 0; i < pad; i++)
        out[(size_t)pad + (size_t)n + (size_t)i] = x[n - 2 - i];
}

// nnaudio.downsampling_by_n with match_torch_exactly=True: zero-pad
// (L-1)/2 on BOTH sides, then a VALID stride-2 correlation with the FIR.
// Output length = floor((n + 2*((L-1)/2) - L) / 2) + 1.
inline void decimate_by_2(const float* x, int n, const float* fir, int L, std::vector<float>& out) {
    const int pad = (L - 1) / 2;
    const int padded = n + 2 * pad;
    if (padded < L) {
        out.clear();
        return;
    }
    const int T = (padded - L) / 2 + 1;
    out.assign((size_t)T, 0.0f);

    const char* env_serial = std::getenv("CRISPASR_CQT2010_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');

    if (force_serial || T <= 16) {
        for (int t = 0; t < T; t++) {
            const int base = 2 * t - pad; // index into x of tap 0
            double acc = 0.0;
            int k0 = 0, k1 = L;
            if (base < 0)
                k0 = -base;
            if (base + L > n)
                k1 = n - base;
            for (int k = k0; k < k1; k++)
                acc += (double)x[base + k] * (double)fir[k];
            out[(size_t)t] = (float)acc;
        }
    } else {
        core_parallel::for_each_chunk(T, 0, [&](int t_start, int t_end) {
            for (int t = t_start; t < t_end; t++) {
                const int base = 2 * t - pad;
                double acc = 0.0;
                int k0 = 0, k1 = L;
                if (base < 0)
                    k0 = -base;
                if (base + L > n)
                    k1 = n - base;
                for (int k = k0; k < k1; k++)
                    acc += (double)x[base + k] * (double)fir[k];
                out[(size_t)t] = (float)acc;
            }
        });
    }
}

// One octave: reflect-pad by n_fft/2, then a VALID stride-`hop` complex
// correlation with the whole kernel bank.
// Writes re/im as [n_filters * T], filter-major.
inline int octave_complex(const Params& p, const Kernels& kern, const float* x, int n, int hop, std::vector<float>& re,
                          std::vector<float>& im) {
    const int pad = p.n_fft / 2;
    std::vector<float> xp;
    reflect_pad(x, n, pad, xp);
    const int padded = (int)xp.size();
    if (padded < p.n_fft)
        return 0;
    const int T = (padded - p.n_fft) / hop + 1;
    re.assign((size_t)p.n_filters * (size_t)T, 0.0f);
    im.assign((size_t)p.n_filters * (size_t)T, 0.0f);

    const char* env_serial = std::getenv("CRISPASR_CQT2010_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');

    if (force_serial || p.n_filters <= 2) {
        for (int k = 0; k < p.n_filters; k++) {
            const float* kr = kern.real + (size_t)k * (size_t)p.n_fft;
            const float* ki = kern.imag + (size_t)k * (size_t)p.n_fft;
            for (int t = 0; t < T; t++) {
                const float* seg = xp.data() + (size_t)t * (size_t)hop;
                double ar = 0.0, ai = 0.0;
                for (int nn = 0; nn < p.n_fft; nn++) {
                    const double v = (double)seg[nn];
                    ar += v * (double)kr[nn];
                    ai += v * (double)ki[nn];
                }
                re[(size_t)k * (size_t)T + (size_t)t] = (float)ar;
                im[(size_t)k * (size_t)T + (size_t)t] = (float)ai;
            }
        }
    } else {
        core_parallel::for_each_chunk(p.n_filters, 0, [&](int k_start, int k_end) {
            for (int k = k_start; k < k_end; k++) {
                const float* kr = kern.real + (size_t)k * (size_t)p.n_fft;
                const float* ki = kern.imag + (size_t)k * (size_t)p.n_fft;
                float* re_k = re.data() + (size_t)k * (size_t)T;
                float* im_k = im.data() + (size_t)k * (size_t)T;
                for (int t = 0; t < T; t++) {
                    const float* seg = xp.data() + (size_t)t * (size_t)hop;
                    double ar = 0.0, ai = 0.0;
                    for (int nn = 0; nn < p.n_fft; nn++) {
                        const double v = (double)seg[nn];
                        ar += v * (double)kr[nn];
                        ai += v * (double)ki[nn];
                    }
                    re_k[t] = (float)ar;
                    im_k[t] = (float)ai;
                }
            }
        });
    }
    return T;
}

// |CQT|, frame-major: out[t * n_bins + k], k ascending in frequency.
// Returns the frame count, or 0 on a degenerate input.
//
// NOTE ON THE MISSING NEGATION: upstream computes `CQT_imag = -conv1d(...)`.
// The only consumer is sqrt(re^2 + im^2), so the sign is dropped here. If a
// caller ever needs the complex CQT, restore it at that point — do not "fix"
// it here, because the magnitude path is what the parity numbers were measured
// against.
inline int magnitude(const Params& p, const Kernels& kern, const float* x, int n_samples, std::vector<float>& out) {
    out.clear();
    if (!x || n_samples <= 0 || !kern.real || !kern.imag || !kern.lowpass || !kern.sqrt_lengths)
        return 0;
    if (p.n_octaves <= 0 || p.n_filters <= 0)
        return 0;

    // Octave 0 is the top; each subsequent level decimates x by 2 and halves
    // the hop, so level L covers an octave L below the top.
    std::vector<std::vector<float>> re_lv((size_t)p.n_octaves), im_lv((size_t)p.n_octaves);
    int T = 0;

    std::vector<float> cur(x, x + n_samples);
    int hop = p.hop_length;
    for (int L = 0; L < p.n_octaves; L++) {
        if (L > 0) {
            hop /= 2;
            std::vector<float> dn;
            decimate_by_2(cur.data(), (int)cur.size(), kern.lowpass, kern.lowpass_len, dn);
            cur.swap(dn);
            if (cur.empty() || hop < 1)
                return 0;
        }
        const int t = octave_complex(p, kern, cur.data(), (int)cur.size(), hop, re_lv[(size_t)L], im_lv[(size_t)L]);
        if (t == 0)
            return 0;
        if (L == 0)
            T = t;
        else if (t != T)
            // Upstream relies on every octave yielding the same frame count for
            // its 43844-sample window (see the length table in the port notes).
            // If a caller feeds a length where that breaks, fail loudly rather
            // than silently mis-stacking octaves.
            return 0;
    }

    // Upstream concatenates each newly computed (lower) octave in FRONT of the
    // accumulator, so the final bin order is level n_octaves-1 first (lowest
    // frequency) down to level 0 last, then the bottom
    // (n_octaves*n_filters - n_bins) bins are dropped.
    const int total = p.n_octaves * p.n_filters;
    const int drop = total - p.n_bins;
    if (drop < 0)
        return 0;

    out.assign((size_t)T * (size_t)p.n_bins, 0.0f);
    const char* env_serial = std::getenv("CRISPASR_CQT2010_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');
    if (force_serial || total <= 4) {
        for (int b = 0; b < total; b++) {
            const int f = b - drop;
            if (f < 0)
                continue;
            const int block = b / p.n_filters; // 0 = lowest octave
            const int k = b % p.n_filters;
            const int L = p.n_octaves - 1 - block;
            const float* rp = re_lv[(size_t)L].data() + (size_t)k * (size_t)T;
            const float* ip = im_lv[(size_t)L].data() + (size_t)k * (size_t)T;
            const double s = (double)kern.sqrt_lengths[f];
            for (int t = 0; t < T; t++) {
                const double r = (double)rp[t] * s;
                const double i = (double)ip[t] * s;
                out[(size_t)t * (size_t)p.n_bins + (size_t)f] = (float)std::sqrt(r * r + i * i);
            }
        }
    } else {
        core_parallel::for_each_chunk(total, 0, [&](int b_start, int b_end) {
            for (int b = b_start; b < b_end; b++) {
                const int f = b - drop;
                if (f < 0)
                    continue;
                const int block = b / p.n_filters; // 0 = lowest octave
                const int k = b % p.n_filters;
                const int L = p.n_octaves - 1 - block;
                const float* rp = re_lv[(size_t)L].data() + (size_t)k * (size_t)T;
                const float* ip = im_lv[(size_t)L].data() + (size_t)k * (size_t)T;
                const double s = (double)kern.sqrt_lengths[f];
                for (int t = 0; t < T; t++) {
                    const double r = (double)rp[t] * s;
                    const double i = (double)ip[t] * s;
                    out[(size_t)t * (size_t)p.n_bins + (size_t)f] = (float)std::sqrt(r * r + i * i);
                }
            }
        });
    }
    return T;
}

// signal.NormalizedLog: magnitude -> power -> dB -> min/max normalised to [0,1]
// over BOTH axes of the whole example. In place on a [T * n_bins] buffer.
//
// `div_no_nan`: a constant spectrogram (max == min) yields 0, not NaN.
inline void normalized_log(float* v, size_t n, float eps = 1e-10f) {
    if (!v || n == 0)
        return;
    double lo = 0.0, hi = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double p = (double)v[i] * (double)v[i];
        const double db = 10.0 * std::log10(p + (double)eps);
        v[i] = (float)db;
        if (i == 0 || db < lo)
            lo = db;
    }
    for (size_t i = 0; i < n; i++) {
        const double d = (double)v[i] - lo;
        if (i == 0 || d > hi)
            hi = d;
    }
    if (hi == 0.0) {
        for (size_t i = 0; i < n; i++)
            v[i] = 0.0f;
        return;
    }

    const char* env_serial = std::getenv("CRISPASR_CQT2010_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');
    if (force_serial || n <= 1024) {
        for (size_t i = 0; i < n; i++)
            v[i] = (float)(((double)v[i] - lo) / hi);
    } else {
        core_parallel::for_each_chunk((int)n, 0, [&](int i_start, int i_end) {
            for (int i = i_start; i < i_end; i++)
                v[i] = (float)(((double)v[i] - lo) / hi);
        });
    }
}

} // namespace core_cqt2010v2
