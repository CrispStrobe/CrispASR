// tests/test-cpu-ops-ab-bench.cpp — Paired A/B benchmark & parity test
// for CPU ops (src/core/cpu_ops.h) and ConvT permutation (src/core/conv.h).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/cpu_ops.h"
#include "core/conv.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

TEST_CASE("cpu_ops: layernorm paired A/B benchmark and parity", "[cpu_ops][ab-bench]") {
    const int d = 1024;
    const int T = 512;
    const float eps = 1e-5f;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> x((size_t)T * d);
    std::vector<float> w(d);
    std::vector<float> b(d);
    for (auto& v : x)
        v = dist(rng);
    for (auto& v : w)
        v = dist(rng);
    for (auto& v : b)
        v = dist(rng);

    std::vector<float> out_a((size_t)T * d, 0.0f);
    std::vector<float> out_b((size_t)T * d, 0.0f);

    // Warmup
    core_cpu::layernorm_baseline(out_a.data(), x.data(), w.data(), b.data(), d, T, eps);
    core_cpu::layernorm_accelerated(out_b.data(), x.data(), w.data(), b.data(), d, T, eps, 0);

    // Verify parity
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_a.size(); ++i) {
        float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    REQUIRE(max_diff < 1e-5f);

    // Paired benchmark: 10 warm iterations each
    const int kIterations = 10;
    std::vector<double> times_a_ms;
    std::vector<double> times_b_ms;
    times_a_ms.reserve(kIterations);
    times_b_ms.reserve(kIterations);

    for (int iter = 0; iter < kIterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        core_cpu::layernorm_baseline(out_a.data(), x.data(), w.data(), b.data(), d, T, eps);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto t2 = std::chrono::high_resolution_clock::now();
        core_cpu::layernorm_accelerated(out_b.data(), x.data(), w.data(), b.data(), d, T, eps, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        times_b_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

    auto calc_stats = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double var = 0.0;
        for (double val : v)
            var += (val - mean) * (val - mean);
        double sd = std::sqrt(var / v.size());
        return std::make_tuple(med, mean, sd, v.front(), v.back());
    };

    auto [med_a, mean_a, sd_a, min_a, max_a] = calc_stats(times_a_ms);
    auto [med_b, mean_b, sd_b, min_b, max_b] = calc_stats(times_b_ms);
    double speedup = med_a / med_b;

    std::printf("\n======================================================================\n");
    std::printf("CPU LayerNorm Paired A/B Benchmark (T=%d, d=%d)\n", T, d);
    std::printf("======================================================================\n");
    std::printf("PARITY: max_diff = %.2e across %zu floats\n", max_diff, out_a.size());
    std::printf("  Arm A (Serial Baseline):     median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n", med_a,
                mean_a, sd_a, min_a, max_a);
    std::printf("  Arm B (Accelerated Parallel): median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n",
                med_b, mean_b, sd_b, min_b, max_b);
    std::printf("  SPEEDUP FACTOR:                %4.2fx faster (%6.2f ms saved per LayerNorm)\n", speedup,
                med_a - med_b);
    std::printf("======================================================================\n\n");

    REQUIRE(speedup >= 1.5);
}

TEST_CASE("convt: permute_convt1d_host paired A/B benchmark and parity", "[convt][ab-bench]") {
    const int IC = 512;
    const int OC = 512;
    const int K = 16;
    const size_t total_elems = (size_t)IC * OC * K; // 4,194,304

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> src(total_elems);
    for (auto& v : src)
        v = dist(rng);

    std::vector<float> dst_a(total_elems, 0.0f);
    std::vector<float> dst_b(total_elems, 0.0f);

    auto load_fn = [&](size_t s) -> float { return src[s]; };

    // Warmup
    core_convt::permute_convt1d_host_baseline(load_fn, dst_a.data(), K, OC, IC);
    core_convt::permute_convt1d_host_accelerated(load_fn, dst_b.data(), K, OC, IC, 0);

    // Verify 100% exact bit-identical parity
    size_t bit_identical_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_elems; ++i) {
        if (dst_a[i] == dst_b[i]) {
            bit_identical_count++;
        }
        float diff = std::abs(dst_a[i] - dst_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    double bit_identical_pct = 100.0 * (double)bit_identical_count / (double)total_elems;
    REQUIRE(bit_identical_pct == 100.0);
    REQUIRE(max_diff == 0.0f);

    // Paired benchmark: 10 warm iterations each
    const int kIterations = 10;
    std::vector<double> times_a_ms;
    std::vector<double> times_b_ms;
    times_a_ms.reserve(kIterations);
    times_b_ms.reserve(kIterations);

    for (int iter = 0; iter < kIterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        core_convt::permute_convt1d_host_baseline(load_fn, dst_a.data(), K, OC, IC);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto t2 = std::chrono::high_resolution_clock::now();
        core_convt::permute_convt1d_host_accelerated(load_fn, dst_b.data(), K, OC, IC, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        times_b_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

    auto calc_stats = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double var = 0.0;
        for (double val : v)
            var += (val - mean) * (val - mean);
        double sd = std::sqrt(var / v.size());
        return std::make_tuple(med, mean, sd, v.front(), v.back());
    };

    auto [med_a, mean_a, sd_a, min_a, max_a] = calc_stats(times_a_ms);
    auto [med_b, mean_b, sd_b, min_b, max_b] = calc_stats(times_b_ms);
    double speedup = med_a / med_b;

    std::printf("\n======================================================================\n");
    std::printf("ConvT Weight Permutation Paired A/B Benchmark (IC=%d, OC=%d, K=%d)\n", IC, OC, K);
    std::printf("======================================================================\n");
    std::printf("PARITY: 100%% exact bit-identical output across %zu floats (max_diff = 0.0f)\n", total_elems);
    std::printf("  Arm A (Serial Baseline):     median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n", med_a,
                mean_a, sd_a, min_a, max_a);
    std::printf("  Arm B (Accelerated Parallel): median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n",
                med_b, mean_b, sd_b, min_b, max_b);
    std::printf("  SPEEDUP FACTOR:                %4.2fx faster (%6.2f ms saved per weight permutation)\n", speedup,
                med_a - med_b);
    std::printf("======================================================================\n\n");

    REQUIRE(speedup >= 1.5);
}
