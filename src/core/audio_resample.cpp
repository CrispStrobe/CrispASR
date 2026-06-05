// src/core/audio_resample.cpp — see header for the contract.

#include "audio_resample.h"

#include <algorithm>
#include <cmath>
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

std::vector<float> resample_polyphase(const float* in, int n_in, int src_rate, int dst_rate, int num_zeros,
                                      float kaiser_beta) {
    if (!in || n_in <= 0 || src_rate <= 0 || dst_rate <= 0)
        return {};
    if (src_rate == dst_rate) {
        std::vector<float> out((size_t)n_in);
        std::memcpy(out.data(), in, (size_t)n_in * sizeof(float));
        return out;
    }

    const int g = gcd(src_rate, dst_rate);
    const int L = dst_rate / g;
    const int M = src_rate / g;

    // Total output length — round up so we never silently drop trailing
    // input samples.
    const int n_out = (int)(((int64_t)n_in * dst_rate + src_rate - 1) / src_rate);
    if (n_out <= 0)
        return {};

    auto h = build_filter(L, M, num_zeros, kaiser_beta);
    const int half_len = ((int)h.size() - 1) / 2;
    const int up = std::max(L, M);

    // For each output sample at index `i` (in dst_rate samples), the
    // corresponding fractional position in the upsampled stream
    // (rate = src_rate * L = dst_rate * M) is `i * M`. Each input
    // sample `j` sits at upsampled position `j * L`. So the filter
    // tap index for input sample `j` contributing to output `i` is
    // `(i * M - j * L) + half_len` — must be in [0, total_len).
    std::vector<float> out((size_t)n_out, 0.0f);
    for (int i = 0; i < n_out; i++) {
        const int64_t i_up = (int64_t)i * M;      // position in upsampled stream
        const int j_center = (int)(i_up / L);     // nearest input sample (lower)
        const int phase_offset = (int)(i_up % L); // fractional component
        // Iterate over the input window that covers half_len upsampled
        // taps in each direction.
        const int j_min = std::max(0, j_center - num_zeros);
        const int j_max = std::min(n_in - 1, j_center + num_zeros + 1);
        double s = 0.0;
        for (int j = j_min; j <= j_max; j++) {
            // Filter tap index for this (i, j) pair:
            //   tap = i*M - j*L + half_len
            //       = (j_center - j) * L + phase_offset + half_len
            const int tap = (j_center - j) * L + phase_offset + half_len;
            if (tap < 0 || tap >= (int)h.size())
                continue;
            s += (double)in[(size_t)j] * (double)h[(size_t)tap];
        }
        // Compensate for the implicit factor-L upsample (zero-stuffing
        // multiplies the spectrum by 1/L; a corresponding factor-L gain
        // restores unity DC).
        out[(size_t)i] = (float)(s * (double)up);
    }
    return out;
}

} // namespace core_audio
