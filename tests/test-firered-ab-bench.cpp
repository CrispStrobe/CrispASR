// tests/test-firered-ab-bench.cpp — Paired A/B benchmark & parity test
// for FireRed ASR CPU Matmul and F5-TTS Conv1D operations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/cpu_attention.h"
#include "core/cpu_model_kernels.h"
#include "ab_bench_helpers.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

// Independent serial oracle; the candidate below is the production helper.
static void cpu_matmul_bt_baseline(const float* A, const float* B, float* C, int M, int K, int N) {
    if (M == 1) {
        for (int n = 0; n < N; n++)
            C[n] = crispasr::cpu_attn::cpu_dot(A, B + (size_t)n * K, K);
    } else {
        for (int m = 0; m < M; m++) {
            const float* arow = A + (size_t)m * K;
            for (int n = 0; n < N; n++)
                C[(size_t)m * N + n] = crispasr::cpu_attn::cpu_dot(arow, B + (size_t)n * K, K);
        }
    }
}

static void cpu_matmul_bt_accelerated(const float* A, const float* B, float* C, int M, int K, int N) {
    core_cpu_kernels::matmul_bt(A, B, C, M, K, N, 4);
}

// Independent scalar oracle for same-padding dilated convolution.
static void cpu_conv1d_dil_baseline(const float* input, int C_in, int T, const float* weight, const float* bias,
                                    int C_out, int K, int dilation, float* output) {
    const int pad = dilation * (K - 1) / 2;
    for (int oc = 0; oc < C_out; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < C_in; ic++) {
                const float* in = input + (size_t)ic * T;
                const float* w = weight + ((size_t)oc * C_in + ic) * K;
                for (int k = 0; k < K; k++) {
                    int ti = t + k * dilation - pad;
                    if (ti >= 0 && ti < T)
                        sum += in[ti] * w[k];
                }
            }
            output[(size_t)oc * T + t] = sum;
        }
    }
}

static void cpu_conv1d_dil_accelerated(const float* input, int C_in, int T, const float* weight, const float* bias,
                                       int C_out, int K, int dilation, float* output) {
    core_cpu_kernels::conv1d_dil(input, C_in, T, weight, bias, C_out, K, dilation, output, 4);
}

TEST_CASE("FireRed production matmul covers batches and vector tails", "[unit][firered][cpu-kernels]") {
    for (int M : {1, 2, 7}) {
        for (int K : {1, 5, 65, 1280}) {
            const int N = 129;
            INFO("M=" << M << " K=" << K << " N=" << N);
            std::vector<float> a((size_t)M * K), b((size_t)N * K), reference((size_t)M * N), actual(reference.size());
            for (size_t i = 0; i < a.size(); ++i)
                a[i] = (float)((int)(i % 23) - 11) / 23;
            for (size_t i = 0; i < b.size(); ++i)
                b[i] = (float)((int)(i % 17) - 8) / 17;
            cpu_matmul_bt_baseline(a.data(), b.data(), reference.data(), M, K, N);
            for (int threads : {1, 4}) {
                core_cpu_kernels::matmul_bt(a.data(), b.data(), actual.data(), M, K, N, threads);
                REQUIRE(actual == reference);
                REQUIRE(std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }));
            }
        }
    }
}

TEST_CASE("F5 production dilated convolution covers boundaries and optional bias", "[unit][f5_tts][cpu-kernels]") {
    for (int T : {1, 5, 31}) {
        for (int dilation : {1, 3, 9}) {
            const int C_in = 5, C_out = 7, K = 7;
            INFO("T=" << T << " dilation=" << dilation);
            std::vector<float> input((size_t)C_in * T), weight((size_t)C_out * C_in * K), bias(C_out, 0.125f);
            std::vector<float> reference((size_t)C_out * T), actual(reference.size());
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = (float)((int)(i % 23) - 11) / 23;
            for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = (float)((int)(i % 17) - 8) / 17;
            for (bool with_bias : {false, true}) {
                const float* b = with_bias ? bias.data() : nullptr;
                cpu_conv1d_dil_baseline(input.data(), C_in, T, weight.data(), b, C_out, K, dilation, reference.data());
                for (int threads : {1, 4}) {
                    core_cpu_kernels::conv1d_dil(input.data(), C_in, T, weight.data(), b, C_out, K, dilation,
                                                 actual.data(), threads);
                    REQUIRE(actual == reference);
                    REQUIRE(
                        std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); }));
                }
            }
        }
    }
}

TEST_CASE("firered_asr: cpu_matmul_bt paired A/B benchmark and parity", "[unit][firered][ab-bench]") {
    const int M = 1;
    const int K = 1280;
    const int N = 1024;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A((size_t)M * K);
    std::vector<float> B((size_t)N * K);
    for (auto& v : A)
        v = dist(rng);
    for (auto& v : B)
        v = dist(rng);

    std::vector<float> C_a((size_t)M * N, 0.0f);
    std::vector<float> C_b((size_t)M * N, 0.0f);

    // Warmup
    cpu_matmul_bt_baseline(A.data(), B.data(), C_a.data(), M, K, N);
    cpu_matmul_bt_accelerated(A.data(), B.data(), C_b.data(), M, K, N);

    // Parity check: 100% exact bit-identical
    float max_diff = 0.0f;
    bool all_finite = true;
    for (size_t i = 0; i < C_a.size(); ++i) {
        all_finite &= std::isfinite(C_a[i]) && std::isfinite(C_b[i]);
        float diff = std::abs(C_a[i] - C_b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    REQUIRE(all_finite);
    REQUIRE(max_diff == 0.0f);
    if (!benchmark_timing_enabled())
        return;

    const int kIterations = 10;
    std::vector<double> times_a_ms;
    std::vector<double> times_b_ms;
    times_a_ms.reserve(kIterations);
    times_b_ms.reserve(kIterations);

    for (int iter = 0; iter < kIterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_matmul_bt_baseline(A.data(), B.data(), C_a.data(), M, K, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto t2 = std::chrono::high_resolution_clock::now();
        cpu_matmul_bt_accelerated(A.data(), B.data(), C_b.data(), M, K, N);
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
    std::printf("FireRed ASR cpu_matmul_bt Paired A/B Benchmark (M=%d, K=%d, N=%d)\n", M, K, N);
    std::printf("======================================================================\n");
    std::printf("PARITY: 100%% exact bit-identical output across %zu floats (max_diff = 0.0f)\n", C_a.size());
    std::printf("  Arm A (Serial Baseline):     median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n", med_a,
                mean_a, sd_a, min_a, max_a);
    std::printf("  Arm B (Accelerated Parallel): median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n",
                med_b, mean_b, sd_b, min_b, max_b);
    std::printf("  SPEEDUP FACTOR:                %4.2fx faster (%6.2f ms saved per matmul)\n", speedup, med_a - med_b);
    std::printf("======================================================================\n\n");

    // Timing is reported only; shared-worker speed is not a correctness gate.
}

TEST_CASE("f5_tts: cpu_conv1d_dil paired A/B benchmark and parity", "[unit][f5_tts][ab-bench]") {
    const int C_in = 64;
    const int C_out = 64;
    const int T = 128;
    const int K = 7;
    const int dilation = 3;

    std::mt19937 rng(888);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> input((size_t)C_in * T);
    std::vector<float> weight((size_t)C_out * C_in * K);
    std::vector<float> bias(C_out);
    for (auto& v : input)
        v = dist(rng);
    for (auto& v : weight)
        v = dist(rng);
    for (auto& v : bias)
        v = dist(rng);

    std::vector<float> output_a((size_t)C_out * T, 0.0f);
    std::vector<float> output_b((size_t)C_out * T, 0.0f);

    // Warmup
    cpu_conv1d_dil_baseline(input.data(), C_in, T, weight.data(), bias.data(), C_out, K, dilation, output_a.data());
    cpu_conv1d_dil_accelerated(input.data(), C_in, T, weight.data(), bias.data(), C_out, K, dilation, output_b.data());

    // Parity check: 100% exact bit-identical
    float max_diff = 0.0f;
    bool all_finite = true;
    for (size_t i = 0; i < output_a.size(); ++i) {
        all_finite &= std::isfinite(output_a[i]) && std::isfinite(output_b[i]);
        float diff = std::abs(output_a[i] - output_b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    REQUIRE(all_finite);
    REQUIRE(max_diff == 0.0f);
    if (!benchmark_timing_enabled())
        return;

    const int kIterations = 10;
    std::vector<double> times_a_ms;
    std::vector<double> times_b_ms;
    times_a_ms.reserve(kIterations);
    times_b_ms.reserve(kIterations);

    for (int iter = 0; iter < kIterations; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_conv1d_dil_baseline(input.data(), C_in, T, weight.data(), bias.data(), C_out, K, dilation, output_a.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        times_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        auto t2 = std::chrono::high_resolution_clock::now();
        cpu_conv1d_dil_accelerated(input.data(), C_in, T, weight.data(), bias.data(), C_out, K, dilation,
                                   output_b.data());
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
    std::printf("F5-TTS cpu_conv1d_dil Paired A/B Benchmark (Cin=%d, Cout=%d, T=%d, K=%d)\n", C_in, C_out, T, K);
    std::printf("======================================================================\n");
    std::printf("PARITY: 100%% exact bit-identical output across %zu floats (max_diff = 0.0f)\n", output_a.size());
    std::printf("  Arm A (Serial Baseline):     median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n", med_a,
                mean_a, sd_a, min_a, max_a);
    std::printf("  Arm B (Accelerated Parallel): median %6.2f ms (mean %6.2f ± %.2f ms) [min %6.2f, max %6.2f]\n",
                med_b, mean_b, sd_b, min_b, max_b);
    std::printf("  SPEEDUP FACTOR:                %4.2fx faster (%6.2f ms saved per dilated conv)\n", speedup,
                med_a - med_b);
    std::printf("======================================================================\n\n");

    // Timing is reported only; shared-worker speed is not a correctness gate.
}
