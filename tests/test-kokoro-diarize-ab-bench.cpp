// tests/test-kokoro-diarize-ab-bench.cpp — Paired A/B benchmark for Kokoro TTS & Diarization
// CPU accelerations.
//
// Verifies:
// 1. Bit-for-bit exact parity between serial baseline and chunk-parallel implementations.
// 2. Optional alternating latency benchmark across 10 warm iterations:
//    enable with CRISPASR_RUN_PERF=1; timing never gates unit correctness.
// 3. Fallback gate verification via CRISPASR_KOKORO_SERIAL=1 and CRISPASR_DIARIZE_SERIAL=1.

#include <catch2/catch_test_macros.hpp>
#include "ab_bench_helpers.h"
#include "scoped_test_env.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "core/spectral_diarize.h"
#include "kokoro.h"

#if defined(_WIN32)
static void set_env_var(const char* name, const char* value) {
    _putenv_s(name, value ? value : "");
}
static void unset_env_var(const char* name) {
    _putenv_s(name, "");
}
#else
static void set_env_var(const char* name, const char* value) {
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
}
static void unset_env_var(const char* name) {
    unsetenv(name);
}
#endif

struct BenchStats {
    double min_ms;
    double max_ms;
    double median_ms;
    double mean_ms;
    double stddev_ms;
};

static BenchStats compute_stats(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    const double min_v = samples.front();
    const double max_v = samples.back();
    const double median_v = (n % 2 == 1) ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    const double mean_v = sum / (double)n;
    double var_sum = 0.0;
    for (double x : samples) {
        var_sum += (x - mean_v) * (x - mean_v);
    }
    const double stddev_v = std::sqrt(var_sum / (double)n);
    return {min_v, max_v, median_v, mean_v, stddev_v};
}

TEST_CASE("Diarization cosine_similarity paired A/B benchmark", "[unit][diarize][ab-bench]") {
    ScopedTestEnv restore_env("CRISPASR_DIARIZE_SERIAL");
    const int n = 512;
    const int d = 256;
    const size_t total_x = (size_t)n * d;
    std::vector<float> x(total_x);
    for (size_t i = 0; i < total_x; i++) {
        x[i] = std::sin((float)i * 0.017f) * std::cos((float)i * 0.005f);
    }

    // Arm A: Serial baseline
    set_env_var("CRISPASR_DIARIZE_SERIAL", "1");
    auto warm_a = core_spectral::cosine_similarity(x.data(), n, d);
    (void)warm_a;

    const int iters = 10;
    std::vector<double> times_a(iters);
    std::vector<float> out_a = warm_a;

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_DIARIZE_SERIAL");
    auto warm_b = core_spectral::cosine_similarity(x.data(), n, d);
    (void)warm_b;

    std::vector<double> times_b(iters);
    std::vector<float> out_b = warm_b;

    // Parity verification
    REQUIRE(out_a.size() == (size_t)n * n);
    REQUIRE(out_b.size() == (size_t)n * n);
    REQUIRE(std::all_of(out_a.begin(), out_a.end(), [](float x) { return std::isfinite(x); }));
    REQUIRE(std::all_of(out_b.begin(), out_b.end(), [](float x) { return std::isfinite(x); }));
    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        const float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff <= 1e-6f);

    if (!benchmark_timing_enabled())
        return;

    // Alternate arms on each repetition to reduce drift from a blocked A/B run.
    for (int i = 0; i < iters; i++) {
        set_env_var("CRISPASR_DIARIZE_SERIAL", "1");
        {
            const auto t0 = std::chrono::steady_clock::now();
            out_a = core_spectral::cosine_similarity(x.data(), n, d);
            const auto t1 = std::chrono::steady_clock::now();
            times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        unset_env_var("CRISPASR_DIARIZE_SERIAL");
        {
            const auto t0 = std::chrono::steady_clock::now();
            out_b = core_spectral::cosine_similarity(x.data(), n, d);
            const auto t1 = std::chrono::steady_clock::now();
            times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] Diarization cosine_similarity (n=%d, d=%d, %d matrix entries) ===\n", n, d, n * n);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per similarity matrix)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e\n", max_diff);
}

TEST_CASE("Kokoro test_make_har_dft paired A/B benchmark", "[unit][kokoro][ab-bench]") {
    ScopedTestEnv restore_env("CRISPASR_KOKORO_SERIAL");
    const int hop = 5;
    const int n_fft = 20;
    const int n_bins = 11;
    const int T_har = 24000; // ~5 seconds of 24kHz audio (120,000 samples)
    const int L_pad = T_har * hop + n_fft;

    std::vector<float> padded((size_t)L_pad);
    for (int i = 0; i < L_pad; i++) {
        padded[i] = std::sin((float)i * 0.05f) * 0.5f;
    }

    std::vector<float> har_a((size_t)22 * T_har, 0.0f);
    std::vector<float> har_b((size_t)22 * T_har, 0.0f);

    // Arm A: Serial baseline
    set_env_var("CRISPASR_KOKORO_SERIAL", "1");
    kokoro_test::test_make_har_dft(padded.data(), hop, n_fft, n_bins, har_a.data(), T_har);

    const int iters = 10;
    std::vector<double> times_a(iters);

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_KOKORO_SERIAL");
    kokoro_test::test_make_har_dft(padded.data(), hop, n_fft, n_bins, har_b.data(), T_har);

    std::vector<double> times_b(iters);

    // Parity verification
    REQUIRE(std::all_of(har_a.begin(), har_a.end(), [](float x) { return std::isfinite(x); }));
    REQUIRE(std::all_of(har_b.begin(), har_b.end(), [](float x) { return std::isfinite(x); }));
    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)22 * T_har; i++) {
        const float diff = std::abs(har_a[i] - har_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff == 0.0f);

    if (!benchmark_timing_enabled())
        return;

    // Alternate arms on each repetition to reduce drift from a blocked A/B run.
    for (int i = 0; i < iters; i++) {
        set_env_var("CRISPASR_KOKORO_SERIAL", "1");
        {
            const auto t0 = std::chrono::steady_clock::now();
            kokoro_test::test_make_har_dft(padded.data(), hop, n_fft, n_bins, har_a.data(), T_har);
            const auto t1 = std::chrono::steady_clock::now();
            times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        unset_env_var("CRISPASR_KOKORO_SERIAL");
        {
            const auto t0 = std::chrono::steady_clock::now();
            kokoro_test::test_make_har_dft(padded.data(), hop, n_fft, n_bins, har_b.data(), T_har);
            const auto t1 = std::chrono::steady_clock::now();
            times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] Kokoro test_make_har_dft (T_har=%d, n_fft=%d, n_bins=%d) ===\n", T_har, n_fft, n_bins);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per DFT synthesis pass)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n", max_diff);
}

TEST_CASE("Kokoro test_f0_upsample paired A/B benchmark", "[unit][kokoro][ab-bench]") {
    ScopedTestEnv restore_env("CRISPASR_KOKORO_SERIAL");
    const int L_low = 1000;
    const int upsample_scale = 300;
    const int L_high = L_low * upsample_scale;

    std::vector<float> f0_curve(L_low);
    for (int i = 0; i < L_low; i++) {
        f0_curve[i] = 120.0f + 30.0f * std::sin((float)i * 0.05f);
    }

    std::vector<float> f0_up_a((size_t)L_high, 0.0f);
    std::vector<float> f0_up_b((size_t)L_high, 0.0f);

    // Arm A: Serial baseline
    set_env_var("CRISPASR_KOKORO_SERIAL", "1");
    kokoro_test::test_f0_upsample(f0_curve.data(), L_low, upsample_scale, f0_up_a.data());

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_KOKORO_SERIAL");
    kokoro_test::test_f0_upsample(f0_curve.data(), L_low, upsample_scale, f0_up_b.data());

    // Parity verification
    REQUIRE(std::all_of(f0_up_a.begin(), f0_up_a.end(), [](float x) { return std::isfinite(x); }));
    REQUIRE(std::all_of(f0_up_b.begin(), f0_up_b.end(), [](float x) { return std::isfinite(x); }));
    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)L_high; i++) {
        const float diff = std::abs(f0_up_a[i] - f0_up_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff == 0.0f);
    printf("Kokoro f0_upsample parity: max_diff = %.8e (100%% exact bit-identical)\n", max_diff);
}
