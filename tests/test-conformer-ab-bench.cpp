// tests/test-conformer-ab-bench.cpp — Paired A/B benchmark & parity test
// for IBM Macaron Conformer Shaw Block Attention (src/core/conformer_ibm.h).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/conformer_ibm.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

TEST_CASE("conformer_ibm: shaw block attention paired A/B benchmark and parity", "[conformer][ab-bench]") {
    const int T = 512;
    const int n_heads = 16;
    const int hd = 64;
    const int d = n_heads * hd; // 1024
    const int ctx_size = 64;
    const float scale = 1.0f / std::sqrt((float)hd);
    const int remainder = 0;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q((size_t)T * d);
    std::vector<float> K((size_t)T * d);
    std::vector<float> V((size_t)T * d);
    std::vector<float> rpe((size_t)ctx_size * ctx_size * hd);

    for (auto& x : Q)
        x = dist(rng);
    for (auto& x : K)
        x = dist(rng);
    for (auto& x : V)
        x = dist(rng);
    for (auto& x : rpe)
        x = dist(rng);

    std::vector<float> out_a((size_t)T * d, 0.0f);
    std::vector<float> out_b((size_t)T * d, 0.0f);

    // Warmup
    core_conformer_ibm::shaw_block_attention_cpu_baseline(out_a.data(), Q.data(), K.data(), V.data(), rpe.data(), T,
                                                          n_heads, hd, ctx_size, scale, remainder);
    core_conformer_ibm::shaw_block_attention_cpu_accelerated(out_b.data(), Q.data(), K.data(), V.data(), rpe.data(), T,
                                                             n_heads, hd, ctx_size, scale, remainder, 0);

    // Verify 100% bit-identical parity
    size_t bit_identical_count = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_a.size(); ++i) {
        if (out_a[i] == out_b[i]) {
            bit_identical_count++;
        }
        float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    double bit_identical_pct = 100.0 * (double)bit_identical_count / (double)out_a.size();
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
        core_conformer_ibm::shaw_block_attention_cpu_baseline(out_a.data(), Q.data(), K.data(), V.data(), rpe.data(), T,
                                                              n_heads, hd, ctx_size, scale, remainder);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto t2 = std::chrono::high_resolution_clock::now();
        core_conformer_ibm::shaw_block_attention_cpu_accelerated(out_b.data(), Q.data(), K.data(), V.data(), rpe.data(),
                                                                 T, n_heads, hd, ctx_size, scale, remainder, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        times_b_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

    auto calc_stats = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double var = 0.0;
        for (double x : v)
            var += (x - mean) * (x - mean);
        double sd = std::sqrt(var / v.size());
        return std::make_tuple(med, mean, sd, v.front(), v.back());
    };

    auto [med_a, mean_a, sd_a, min_a, max_a] = calc_stats(times_a_ms);
    auto [med_b, mean_b, sd_b, min_b, max_b] = calc_stats(times_b_ms);
    double speedup = med_a / med_b;

    std::printf("\n======================================================================\n");
    std::printf("Conformer Shaw Block Attention Paired A/B Benchmark (T=%d, d=%d, hd=%d)\n", T, d, hd);
    std::printf("======================================================================\n");
    std::printf("PARITY: 100%% exact bit-identical output across %zu floats (max_diff = 0.0f)\n", out_a.size());
    std::printf("  Arm A (Serial Baseline):     median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n", med_a,
                mean_a, sd_a, min_a, max_a);
    std::printf("  Arm B (Accelerated Parallel): median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n",
                med_b, mean_b, sd_b, min_b, max_b);
    std::printf("  SPEEDUP FACTOR:                %4.2fx faster (%6.2f ms saved per block attention)\n", speedup,
                med_a - med_b);
    std::printf("======================================================================\n\n");

    REQUIRE(speedup >= 1.5);
}
