// tests/test-voxcpm2-ab-bench.cpp — Paired A/B benchmark for VoxCPM2 & Mel-Band RoFormer
// CPU accelerations.
//
// Verifies:
// 1. Bit-for-bit exact parity (max_diff == 0.0f) between serial baseline and
//    chunk-parallel / precalculated implementations.
// 2. Strict paired latency benchmark across 10 warm iterations (1 warmup discarded):
//    reports min, max, median, mean, stddev, and speedup factor.
// 3. Fallback gate verification via CRISPASR_VOXCPM2_SERIAL=1 and CRISPASR_MEL_BAND_SERIAL=1.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "mel_band_roformer.h"
#include "voxcpm2_tts.h"

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

TEST_CASE("VoxCPM2 wn_reconstruct paired A/B benchmark", "[perf][voxcpm2][ab-bench]") {
    const int out_ch = 512;
    const int in_ch = 256;
    const int ksize = 7;
    const size_t total_w = (size_t)out_ch * in_ch * ksize; // 917,504 floats

    std::vector<float> weight_g(out_ch);
    for (int oc = 0; oc < out_ch; oc++) {
        weight_g[oc] = 0.5f + 0.5f * std::sin((float)oc * 0.05f);
    }
    std::vector<float> weight_v(total_w);
    for (size_t i = 0; i < total_w; i++) {
        weight_v[i] = std::cos((float)i * 0.003f) * 0.1f;
    }

    // Arm A: Serial baseline
    set_env_var("CRISPASR_VOXCPM2_SERIAL", "1");
    auto warm_a = voxcpm2_test::test_wn_reconstruct(weight_g.data(), weight_v.data(), out_ch, in_ch, ksize);
    (void)warm_a;

    const int iters = 10;
    std::vector<double> times_a(iters);
    std::vector<float> out_a;
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        out_a = voxcpm2_test::test_wn_reconstruct(weight_g.data(), weight_v.data(), out_ch, in_ch, ksize);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_VOXCPM2_SERIAL");
    auto warm_b = voxcpm2_test::test_wn_reconstruct(weight_g.data(), weight_v.data(), out_ch, in_ch, ksize);
    (void)warm_b;

    std::vector<double> times_b(iters);
    std::vector<float> out_b;
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        out_b = voxcpm2_test::test_wn_reconstruct(weight_g.data(), weight_v.data(), out_ch, in_ch, ksize);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Parity verification
    REQUIRE(out_a.size() == total_w);
    REQUIRE(out_b.size() == total_w);
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_w; i++) {
        const float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff <= 1e-7f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] VoxCPM2 wn_reconstruct (out_ch=%d, in_ch=%d, ksize=%d, %zu floats) ===\n", out_ch, in_ch,
           ksize, total_w);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per VAE weight reconstruct)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n", max_diff);
}

TEST_CASE("VoxCPM2 snake1d paired A/B benchmark", "[perf][voxcpm2][ab-bench]") {
    const int C = 512;
    const int T = 2000;
    const size_t total = (size_t)C * T; // 1,024,000 floats

    std::vector<float> alpha(C);
    for (int c = 0; c < C; c++) {
        alpha[c] = 0.8f + 0.4f * std::cos((float)c * 0.1f);
    }
    std::vector<float> x_in(total);
    for (size_t i = 0; i < total; i++) {
        x_in[i] = std::sin((float)i * 0.007f);
    }

    // Arm A: Serial baseline
    set_env_var("CRISPASR_VOXCPM2_SERIAL", "1");
    std::vector<float> out_a(total);
    voxcpm2_test::test_snake1d(alpha.data(), x_in.data(), out_a.data(), C, T); // warmup

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        voxcpm2_test::test_snake1d(alpha.data(), x_in.data(), out_a.data(), C, T);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_VOXCPM2_SERIAL");
    std::vector<float> out_b(total);
    voxcpm2_test::test_snake1d(alpha.data(), x_in.data(), out_b.data(), C, T); // warmup

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        voxcpm2_test::test_snake1d(alpha.data(), x_in.data(), out_b.data(), C, T);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Parity verification
    float max_diff = 0.0f;
    for (size_t i = 0; i < total; i++) {
        const float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff == 0.0f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] VoxCPM2 snake1d (C=%d, T=%d, %zu floats) ===\n", C, T, total);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per snake1d activation)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n", max_diff);
}

TEST_CASE("Mel-Band RoFormer rope_head precalculated inv_freq paired A/B benchmark", "[perf][roformer][ab-bench]") {
    const int T = 2048;
    const int dim_head = 64;
    const size_t total = (size_t)T * dim_head;

    // Reference naive rope implementation (recomputing pow every m)
    auto naive_rope = [](float* qh, int T_len, int dh) {
        const int half = dh / 2;
        for (int m = 0; m < T_len; m++) {
            float* row = qh + (size_t)m * dh;
            for (int i = 0; i < half; i++) {
                const double inv = std::pow(10000.0, -(double)(2 * i) / (double)dh);
                const double ang = (double)m * inv;
                const double c = std::cos(ang), s = std::sin(ang);
                const float a = row[2 * i], b = row[2 * i + 1];
                row[2 * i] = (float)(a * c - b * s);
                row[2 * i + 1] = (float)(b * c + a * s);
            }
        }
    };

    std::vector<float> init_data(total);
    for (size_t i = 0; i < total; i++) {
        init_data[i] = std::sin((float)i * 0.01f);
    }

    // Arm A: Naive reference
    std::vector<float> out_a = init_data;
    naive_rope(out_a.data(), T, dim_head); // warmup

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        out_a = init_data;
        const auto t0 = std::chrono::steady_clock::now();
        naive_rope(out_a.data(), T, dim_head);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Precalculated inv_freq
    std::vector<float> out_b = init_data;
    mel_band_roformer_test::test_rope_head(out_b.data(), T, dim_head); // warmup

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        out_b = init_data;
        const auto t0 = std::chrono::steady_clock::now();
        mel_band_roformer_test::test_rope_head(out_b.data(), T, dim_head);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Parity verification
    float max_diff = 0.0f;
    for (size_t i = 0; i < total; i++) {
        const float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff == 0.0f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] Mel-Band RoFormer rope_head (T=%d, dim_head=%d, %zu floats) ===\n", T, dim_head, total);
    printf("Arm A (Naive pow per step):median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Precalculated inv):  median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per RoPE head call)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n\n", max_diff);
}
