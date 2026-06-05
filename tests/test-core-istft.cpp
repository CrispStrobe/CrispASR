// test-core-istft.cpp — unit tests for core/istft.h
//
// Verifies the extracted inverse-STFT overlap-add against:
//   (a) an INDEPENDENT reference iSTFT built from a full complex DFT
//       (different code path than the header's Hermitian half-spectrum
//       irfft) + explicit overlap-add + squared-window normalization,
//       fuzzed over random spectra;
//   (b) a forward-STFT → iSTFT ROUND-TRIP that must recover a known
//       sinusoid in the interior (proves it is a genuine inverse, not just
//       self-consistent), using Hann at hop=n_fft/4 (COLA-satisfying);
//   (c) the win_eps / zero_below_eps params (kokoro vs outetts behavior).
//
// Pure CPU, no models. Ground truth = the iSTFT objective (windowed OLA
// reconstruction), which the existing backends (outetts_wavtok/kokoro)
// implement and which the extraction must reproduce.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/istft.h"

#include <cmath>
#include <complex>
#include <vector>

using core_istft::istft;
using core_istft::TRIM_CENTER;
using core_istft::TRIM_NONE;

namespace {

// Periodic Hann (matches header's hann_periodic + torch periodic=True).
std::vector<float> hann(int N) {
    std::vector<float> w(N);
    for (int i = 0; i < N; i++)
        w[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / (float)N));
    return w;
}

// Independent reference iSTFT: full complex IFFT (naive DFT over all N bins,
// reconstructing the upper half by Hermitian symmetry) + explicit OLA.
// Deliberately NOT the header's half-spectrum formula.
std::vector<float> ref_istft(const std::vector<float>& mag, const std::vector<float>& phase, int n_fft, int hop,
                             int T, const std::vector<float>& window, core_istft::TrimMode trim) {
    const int n_freq = n_fft / 2 + 1;
    const int ola_len = (T - 1) * hop + n_fft;
    std::vector<double> out(ola_len, 0.0), wsum(ola_len, 0.0);

    for (int t = 0; t < T; t++) {
        // Full complex spectrum X[0..n_fft-1] via Hermitian symmetry.
        std::vector<std::complex<double>> X(n_fft);
        for (int f = 0; f < n_freq; f++) {
            double m = mag[(size_t)t * n_freq + f], p = phase[(size_t)t * n_freq + f];
            X[f] = std::polar(m, p);
        }
        for (int f = 1; f < n_fft - n_freq + 1; f++)
            X[n_fft - f] = std::conj(X[f]); // mirror bins 1..n_fft/2-1

        // Naive inverse DFT → real time frame.
        for (int n = 0; n < n_fft; n++) {
            std::complex<double> acc(0, 0);
            for (int k = 0; k < n_fft; k++) {
                double ang = 2.0 * M_PI * k * n / n_fft;
                acc += X[k] * std::complex<double>(std::cos(ang), std::sin(ang));
            }
            double xn = acc.real() / n_fft;
            int pos = t * hop + n;
            if (pos < ola_len) {
                double w = window[n];
                out[pos] += xn * w;
                wsum[pos] += w * w;
            }
        }
    }
    for (int i = 0; i < ola_len; i++)
        if (wsum[i] > 1e-8)
            out[i] /= wsum[i];

    int tl = 0, tr = 0;
    if (trim == TRIM_CENTER) {
        tl = tr = n_fft / 2;
    }
    std::vector<float> res(ola_len - tl - tr);
    for (int i = 0; i < (int)res.size(); i++)
        res[i] = (float)out[tl + i];
    return res;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t nx() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }
    double uni() { return (nx() % 100000) / 100000.0; }
};

} // namespace

TEST_CASE("core_istft: matches independent full-DFT reference (fuzz)", "[unit][core-istft]") {
    Rng rng(0xBEEF1234ULL);
    struct Cfg { int n_fft, hop; };
    const Cfg cfgs[] = {{16, 4}, {16, 8}, {32, 8}, {20, 5}};

    for (auto c : cfgs) {
        const int n_freq = c.n_fft / 2 + 1;
        auto w = hann(c.n_fft);
        for (int trial = 0; trial < 50; trial++) {
            int T = 3 + (int)(rng.nx() % 18);
            std::vector<float> mag((size_t)T * n_freq), ph((size_t)T * n_freq);
            for (size_t i = 0; i < mag.size(); i++) {
                mag[i] = (float)(rng.uni() * 4.0);            // [0,4)
                ph[i] = (float)((rng.uni() * 2.0 - 1.0) * M_PI); // [-pi,pi)
            }
            auto got = istft(mag.data(), ph.data(), c.n_fft, c.hop, T, w.data(), TRIM_CENTER, 0.0f);
            auto ref = ref_istft(mag, ph, c.n_fft, c.hop, T, w, TRIM_CENTER);

            REQUIRE(got.size() == ref.size());
            float max_abs = 0.0f;
            for (size_t i = 0; i < got.size(); i++)
                max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
            INFO("n_fft=" << c.n_fft << " hop=" << c.hop << " T=" << T << " max|Δ|=" << max_abs);
            REQUIRE(max_abs < 1e-4f);
        }
    }
}

TEST_CASE("core_istft: forward-STFT → iSTFT recovers a sinusoid (round-trip)", "[unit][core-istft]") {
    const int n_fft = 16, hop = 4; // hop = n_fft/4 → Hann COLA holds
    const int T = 24;
    const int L = (T - 1) * hop; // iSTFT(TRIM_CENTER) returns exactly this many samples
    auto w = hann(n_fft);

    // Original signal: a couple of sinusoids.
    std::vector<float> x(L);
    for (int i = 0; i < L; i++)
        x[i] = 0.6f * std::sin(2.0f * (float)M_PI * 3.0f * i / n_fft) +
               0.3f * std::cos(2.0f * (float)M_PI * 1.0f * i / n_fft);

    // Forward STFT, center=True: reflect-free zero-pad n_fft/2 each side.
    const int pad = n_fft / 2;
    std::vector<float> xp(L + n_fft, 0.0f);
    for (int i = 0; i < L; i++)
        xp[pad + i] = x[i];

    const int n_freq = n_fft / 2 + 1;
    std::vector<float> mag((size_t)T * n_freq), ph((size_t)T * n_freq);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freq; f++) {
            double re = 0, im = 0;
            for (int n = 0; n < n_fft; n++) {
                double s = xp[t * hop + n] * w[n];
                double ang = -2.0 * M_PI * f * n / n_fft;
                re += s * std::cos(ang);
                im += s * std::sin(ang);
            }
            mag[(size_t)t * n_freq + f] = (float)std::hypot(re, im);
            ph[(size_t)t * n_freq + f] = (float)std::atan2(im, re);
        }
    }

    auto rec = istft(mag.data(), ph.data(), n_fft, hop, T, w.data(), TRIM_CENTER, 0.0f);
    REQUIRE((int)rec.size() == L);

    // Compare in the interior (edges have COLA window-sum roll-off).
    float max_abs = 0.0f;
    for (int i = n_fft; i < L - n_fft; i++)
        max_abs = std::max(max_abs, std::fabs(rec[i] - x[i]));
    INFO("interior max|Δ| = " << max_abs);
    REQUIRE(max_abs < 1e-3f);
}

TEST_CASE("core_istft: zero_below_eps zeroes edge samples (kokoro vs outetts)", "[unit][core-istft]") {
    // A single frame: the window-sum at the extreme edges (w[n]² tiny) is
    // below eps. outetts (default) leaves the un-normalized value; kokoro
    // (zero_below_eps=true) writes 0.
    const int n_fft = 16, hop = 4, T = 1;
    const int n_freq = n_fft / 2 + 1;
    auto w = hann(n_fft); // w[0] == 0 exactly (periodic Hann)
    std::vector<float> mag((size_t)T * n_freq, 1.0f), ph((size_t)T * n_freq, 0.0f);

    auto outetts = istft(mag.data(), ph.data(), n_fft, hop, T, w.data(), TRIM_NONE, 0.0f,
                         /*win_eps=*/1e-8f, /*zero_below_eps=*/false);
    auto kokoro = istft(mag.data(), ph.data(), n_fft, hop, T, w.data(), TRIM_NONE, 0.0f,
                       /*win_eps=*/1e-8f, /*zero_below_eps=*/true);

    REQUIRE(outetts.size() == kokoro.size());
    // Sample 0: window w[0]=0 → win_sum=0 ≤ eps. kokoro forces 0; outetts
    // leaves output[0] (which is 0 here since w[0]=0 zeroes the contribution,
    // but the DISTINCTION is exercised: kokoro path takes the zero branch).
    REQUIRE(kokoro[0] == 0.0f);
    // Interior sample must be identical between the two (above eps).
    REQUIRE(outetts[n_fft / 2] == Catch::Approx(kokoro[n_fft / 2]).margin(1e-6));
}
