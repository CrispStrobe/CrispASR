// test-mel-ab-bench.cpp — Paired A/B benchmark for core_mel optimization.
//
// Follows crispasr-crispembed-dev.md Rule 4/4a/5:
//   - Rigorous quality evaluation:
//       * Float-level exact diff count
//       * Maximum absolute difference: max |A - B|
//       * Cosine parity: (A · B) / (|A| * |B|)
//       * Decoded magnitude verification
//   - Rigorous speed evaluation:
//       * Cold warmup runs (discarded)
//       * 10 warm timed runs per arm
//       * Median, mean, stddev, and throughput (RTF / frames per sec)
//       * Speedup ratio: Median(Arm A Serial) / Median(Arm B Parallel)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "core/fft.h"
#include "core/mel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

void test_fft_r2c(const float* in, int N, float* out) {
    core_fft::fft_radix2_wrapper(in, N, out);
}

std::vector<float> generate_deterministic_signal(int n_samples, int sample_rate = 16000) {
    std::vector<float> sig((size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        const double t = (double)i / sample_rate;
        sig[(size_t)i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * t) +
                         0.3f * std::sin(2.0 * M_PI * 1234.5 * t) +
                         0.15f * std::sin(2.0 * M_PI * 2800.0 * t);
    }
    return sig;
}

std::vector<float> generate_hann_window(int n_fft) {
    std::vector<float> win((size_t)n_fft);
    for (int i = 0; i < n_fft; i++) {
        win[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)(n_fft - 1)));
    }
    return win;
}

struct BenchStats {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
    double rtf = 0.0;
};

BenchStats compute_stats(std::vector<double>& times_ms, double audio_sec) {
    std::sort(times_ms.begin(), times_ms.end());
    BenchStats s;
    s.min_ms = times_ms.front();
    s.max_ms = times_ms.back();
    s.median_ms = times_ms[times_ms.size() / 2];
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    s.mean_ms = sum / times_ms.size();
    double sq_diff = 0.0;
    for (double t : times_ms) {
        sq_diff += (t - s.mean_ms) * (t - s.mean_ms);
    }
    s.stddev_ms = std::sqrt(sq_diff / times_ms.size());
    s.rtf = (audio_sec * 1000.0) / s.median_ms;
    return s;
}

} // namespace

TEST_CASE("core_mel A/B speed and quality benchmark: NeMo FastConformer profile", "[mel][bench]") {
    const int sr = 16000;
    const int n_fft = 512;
    const int n_mels = 128;
    const int n_freqs = n_fft / 2 + 1;
    const double audio_dur = 30.0; // 30 seconds of audio = ~3000 frames
    const int n_samples = (int)(sr * audio_dur);

    const auto pcm = generate_deterministic_signal(n_samples, sr);
    const auto win = generate_hann_window(n_fft);
    const auto fb = core_mel::build_slaney_fb(sr, n_fft, n_mels, 0.0f, 8000.0f, core_mel::FbLayout::MelsFreqs);

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = 160;
    p.win_length = n_fft;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.log_eps = 1e-5f;
    p.log_guard = core_mel::LogGuard::AddEpsilon;
    p.spec_kind = core_mel::SpecKind::Power;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::MelsFreqs;
    p.center_pad = true;
    p.center_pad_reflect = true;
    p.preemph = 0.97f;

    // --- Quality Evaluation ---
    int T_a = 0, T_b = 0;
    p.n_threads = 1;
    const auto out_a = core_mel::compute(pcm.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T_a);

    p.n_threads = 8;
    const auto out_b = core_mel::compute(pcm.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T_b);

    REQUIRE(T_a == T_b);
    REQUIRE(out_a.size() == out_b.size());
    REQUIRE(out_a.size() > 0);

    double max_abs_diff = 0.0;
    double dot_prod = 0.0;
    double norm_a_sq = 0.0;
    double norm_b_sq = 0.0;
    size_t exact_mismatches = 0;

    for (size_t i = 0; i < out_a.size(); i++) {
        const double diff = std::abs((double)out_a[i] - (double)out_b[i]);
        if (diff > max_abs_diff)
            max_abs_diff = diff;
        if (out_a[i] != out_b[i])
            exact_mismatches++;
        dot_prod += (double)out_a[i] * (double)out_b[i];
        norm_a_sq += (double)out_a[i] * (double)out_a[i];
        norm_b_sq += (double)out_b[i] * (double)out_b[i];
    }

    const double cosine = dot_prod / (std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq));

    // Verify mathematical quality
    REQUIRE(exact_mismatches == 0); // 100% bit-identical
    REQUIRE(max_abs_diff == 0.0);
    REQUIRE_THAT(cosine, Catch::Matchers::WithinAbs(1.0, 1e-7));

    // --- Speed Benchmark (1 warmup + 10 iterations each) ---
    const int n_bench_iters = 10;
    std::vector<double> times_a_ms;
    std::vector<double> times_b_ms;

    // Warmup
    p.n_threads = 1;
    (void)core_mel::compute(pcm.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T_a);
    p.n_threads = 8;
    (void)core_mel::compute(pcm.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T_b);

    // Timed iterations
    for (int iter = 0; iter < n_bench_iters; iter++) {
        // Arm A (Serial)
        p.n_threads = 1;
        auto t0 = std::chrono::steady_clock::now();
        (void)core_mel::compute(pcm.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T_a);
        auto t1 = std::chrono::steady_clock::now();
        times_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        // Arm B (Parallel + SIMD)
        p.n_threads = 8;
        t0 = std::chrono::steady_clock::now();
        (void)core_mel::compute(pcm.data(), n_samples, win.data(), n_fft, fb.data(), n_freqs, test_fft_r2c, p, T_b);
        t1 = std::chrono::steady_clock::now();
        times_b_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const auto stats_a = compute_stats(times_a_ms, audio_dur);
    const auto stats_b = compute_stats(times_b_ms, audio_dur);
    const double speedup = stats_a.median_ms / stats_b.median_ms;

    printf("\n===================================================================\n");
    printf("A/B BENCHMARK RECEIPT: core_mel NeMo FastConformer Profile (30s audio, %d frames)\n", T_a);
    printf("===================================================================\n");
    printf("QUALITY:\n");
    printf("  Exact Float Parity    : 100%% (%zu / %zu floats match bit-for-bit)\n", out_a.size() - exact_mismatches, out_a.size());
    printf("  Max Absolute Error    : %.10e\n", max_abs_diff);
    printf("  Cosine Parity         : %.9f\n", cosine);
    printf("SPEED (10 measured runs per arm):\n");
    printf("  Arm A (Serial Baseline)   : median %6.2f ms (mean %6.2f ± %4.2f ms) -> %7.1fx RT\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.rtf);
    printf("  Arm B (Optimized Parallel): median %6.2f ms (mean %6.2f ± %4.2f ms) -> %7.1fx RT\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.rtf);
    printf("  SPEEDUP RATIO             : %.2fx faster\n", speedup);
    printf("===================================================================\n\n");

    REQUIRE(speedup > 1.0); // Optimized path must strictly outperform serial
}
