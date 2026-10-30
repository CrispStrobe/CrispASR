// src/core/audio_resample.cpp — see header for the contract.

#include "audio_resample.h"
#include "parallel_for.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace core_audio {

namespace {

// Modified Bessel function of the first kind, order 0 — for the Kaiser
// window. Series expansion converges quickly for the |x| ≤ β=8.6 range
// the resampler uses.
static double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double xx_4 = (x * x) / 4.0;
    for (int k = 1; k < 50; k++) {
        term *= xx_4 / (double)(k * k);
        sum += term;
        if (term < sum * 1e-15)
            break;
    }
    return sum;
}

// Build a Kaiser-windowed sinc filter for L:M polyphase resampling.
// The filter is sampled at `L * (2 * num_zeros + 1)` taps spanning
// `[-num_zeros, num_zeros]` cycles of the sinc; the polyphase loop
// picks the right phase for each output sample.
//
// Returned coefficients are scaled by `cutoff` so the filter passes DC
// at unity gain after the implicit factor-L upsample.
static std::vector<float> build_filter(int L, int M, int num_zeros, float kaiser_beta) {
    const int up = std::max(L, M);
    const int half_len = num_zeros * up;
    const int total_len = 2 * half_len + 1;
    // Cutoff in normalised input-rate units: pass band stops at
    // 1/max(L, M) of the post-upsample rate. The sinc argument scales
    // by `cutoff`; output is multiplied by `cutoff` so DC gain is 1.
    const double cutoff = 1.0 / (double)up;

    std::vector<float> h((size_t)total_len);
    const double i0_beta = bessel_i0((double)kaiser_beta);
    for (int n = 0; n < total_len; n++) {
        const double k = (double)(n - half_len);
        // Sinc.
        const double sinc_arg = k * cutoff;
        const double sinc = (k == 0.0) ? 1.0 : std::sin(M_PI * sinc_arg) / (M_PI * sinc_arg);
        // Kaiser window.
        const double t = (double)k / (double)half_len;
        const double w_arg = (double)kaiser_beta * std::sqrt(std::max(0.0, 1.0 - t * t));
        const double window = bessel_i0(w_arg) / i0_beta;
        h[(size_t)n] = (float)(sinc * window * cutoff);
    }
    return h;
}

static int gcd(int a, int b) {
    return std::gcd(a, b);
}

} // namespace

// Largest upsample this will perform. See the note in resample_polyphase.
static constexpr int kMaxResampleRatio = 64;

std::vector<float> resample_polyphase(const float* in, int n_in, int src_rate, int dst_rate, int num_zeros,
                                      float kaiser_beta) {
    if (!in || n_in <= 0 || src_rate <= 0 || dst_rate <= 0)
        return {};
    if (src_rate == dst_rate) {
        std::vector<float> out((size_t)n_in);
        std::memcpy(out.data(), in, (size_t)n_in * sizeof(float));
        return out;
    }

    // BOUND THE EXPANSION BEFORE COMPUTING ANYTHING.
    //
    // `sr` reaches most callers straight out of a decoded file header, so
    // dst_rate/src_rate is attacker-controlled. A WAV declaring sampleRate = 1
    // asks this function for a 16000x upsample: 176 000 input samples become
    // 2.816e9 output floats, 11.3 GB. (Same input, different code path, as the
    // crispasr_audio_load ceiling — that one bounds the DECODE, this bounds the
    // RESAMPLE, and the CLI reaches only this one.)
    //
    // THE PREVIOUS GUARD DID NOT GUARD. n_out was computed into an `int`, so
    // that case overflowed to -1478967296 and the `n_out <= 0` test below
    // caught it BY LUCK — the product happened to land negative. At 300 000
    // input samples the same expression overflows to +505032704 and at 400 000
    // to +2105032704: positive, past the test, allocating gigabytes and then
    // resampling with a length that has nothing to do with the data. A guard
    // whose input has already overflowed is not a guard.
    //
    // 64x against what real conversions need: 8k->48k is 6x, 16k->48k is 3x,
    // 8k->96k is 12x. Five times the headroom of the most extreme legitimate
    // ratio, and three orders of magnitude below the abusive one.
    const int64_t n_out64 = ((int64_t)n_in * dst_rate + src_rate - 1) / src_rate;
    if (n_out64 <= 0 || n_out64 > (int64_t)n_in * kMaxResampleRatio || n_out64 > (int64_t)INT_MAX) {
        // LOUD, because the caller cannot tell an empty return from silence.
        // read_audio_data assigned this straight into its PCM buffer, so the
        // CLI reported "0 samples ... no speech detected" and exited 0 — a
        // rejection rendered as a successful transcription of nothing.
        std::fprintf(stderr,
                     "[crispasr-resample] refusing %d samples at %d Hz -> %d Hz: %lld output samples exceeds the "
                     "%dx expansion limit. A sample rate this far from the target usually means a malformed "
                     "header.\n",
                     n_in, src_rate, dst_rate, (long long)n_out64, kMaxResampleRatio);
        return {};
    }

    const int g = gcd(src_rate, dst_rate);
    const int L = dst_rate / g;
    const int M = src_rate / g;

    // Total output length — round up so we never silently drop trailing
    // input samples.
    const int n_out = (int)n_out64;

    struct FilterKey {
        int L = 0, M = 0, num_zeros = 0;
        float kaiser_beta = 0.0f;
        bool operator==(const FilterKey& o) const {
            return L == o.L && M == o.M && num_zeros == o.num_zeros && kaiser_beta == o.kaiser_beta;
        }
    };
    static thread_local FilterKey cached_filter_key = {0, 0, 0, 0.0f};
    static thread_local std::vector<float> cached_filter;

    const FilterKey cur_key = {L, M, num_zeros, kaiser_beta};
    if (!(cached_filter_key == cur_key) || cached_filter.empty()) {
        cached_filter = build_filter(L, M, num_zeros, kaiser_beta);
        cached_filter_key = cur_key;
    }

    const float* h_ptr = cached_filter.data();
    const int total_len = (int)cached_filter.size();
    const int half_len = (total_len - 1) / 2;

    // For each output sample at index `i` (in dst_rate samples), the
    // corresponding fractional position in the upsampled stream
    // (rate = src_rate * L = dst_rate * M) is `i * M`. Each input
    // sample `j` sits at upsampled position `j * L`. So the filter
    // tap index for input sample `j` contributing to output `i` is
    // `(i * M - j * L) + half_len` — must be in [0, total_len).
    // Input-sample half-width of the filter. A tap exists for input `j`
    // while |i*M - j*L| <= half_len, i.e. within half_len/L input samples of
    // `j_center` — which is `num_zeros` only when L >= M (any upsample). For
    // a downsample (M > L) `up` is M, so the filter spans num_zeros*M/L input
    // samples and a fixed ±num_zeros window drops its outer taps: it cut the
    // Kaiser window off mid-slope, costing DC gain and adding ripple
    // (0.25 → 0.2495 ± 5.4e-4 at 24→16 kHz; the every-rate guard in
    // tests/test-audio-resample.cpp is the regression). The `tap` bounds
    // check below still discards anything genuinely outside the filter, so
    // the +1 is only there to cover the phase offset.
    const int j_span = half_len / L + 1;

    std::vector<float> out((size_t)n_out, 0.0f);
    float* out_ptr = out.data();

    auto process_range = [&](int i0, int i1) {
        for (int i = i0; i < i1; i++) {
            const int64_t i_up = (int64_t)i * M;      // position in upsampled stream
            const int j_center = (int)(i_up / L);     // nearest input sample (lower)
            const int phase_offset = (int)(i_up % L); // fractional component
            // Iterate over the input window that covers half_len upsampled
            // taps in each direction.
            const int j_min = std::max(0, j_center - j_span);
            const int j_max = std::min(n_in - 1, j_center + j_span);

            // Bounds on j where tap = (j_center - j)*L + phase_offset + half_len is in [0, total_len)
            const int val = phase_offset + half_len;
            const int j_start = std::max(j_min, j_center - (total_len - 1 - val) / L);
            const int j_end = std::min(j_max, j_center + val / L);

            double s = 0.0;
            int tap = (j_center - j_start) * L + val;
            for (int j = j_start; j <= j_end; j++, tap -= L) {
                s += (double)in[(size_t)j] * (double)h_ptr[(size_t)tap];
            }
            // Compensate for the implicit factor-L upsample (zero-stuffing
            // multiplies the spectrum by 1/L; a corresponding factor-L gain
            // restores unity DC).
            out_ptr[(size_t)i] = (float)(s * (double)L);
        }
    };

    if (n_out >= 4096) {
        core_parallel::for_each_chunk(n_out, 0, process_range);
    } else {
        process_range(0, n_out);
    }
    return out;
}

} // namespace core_audio
