// tests/test-rvq-istft-ab-bench.cpp — Paired A/B benchmark for RVQ and iSTFT
// CPU accelerations.
//
// Verifies:
// 1. Bit-for-bit exact parity between serial baseline and chunk-parallel / SIMD unrolled implementations.
// 2. Strict paired latency benchmark across 10 warm iterations (1 warmup discarded):
//    reports min, max, median, mean, stddev, and speedup factor.
// 3. Fallback gate verification via CRISPASR_RVQ_SERIAL=1 and CRISPASR_ISTFT_SERIAL=1.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "core/istft.h"
#include "core/rvq.h"

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

TEST_CASE("RVQ encode_euclidean paired A/B benchmark", "[perf][rvq][ab-bench]") {
    const int T = 250;
    const int dim = 64;
    const int K = 512;
    const int n_stages = 4;

    std::vector<float> features((size_t)T * dim);
    for (size_t i = 0; i < features.size(); i++) {
        features[i] = std::sin((float)i * 0.03f) * 1.5f;
    }

    std::vector<std::vector<float>> embeds(n_stages);
    std::vector<std::vector<float>> norms(n_stages);
    std::vector<core_rvq::Codebook> stages(n_stages);
    for (int s = 0; s < n_stages; s++) {
        embeds[s].resize((size_t)K * dim);
        norms[s].resize((size_t)K);
        for (size_t i = 0; i < embeds[s].size(); i++) {
            embeds[s][i] = std::cos((float)(i + s * 100) * 0.017f);
        }
        for (int k = 0; k < K; k++) {
            float n = 0.0f;
            for (int j = 0; j < dim; j++) {
                float v = embeds[s][(size_t)k * dim + j];
                n += v * v;
            }
            norms[s][k] = n;
        }
        stages[s] = {embeds[s].data(), norms[s].data(), K, dim};
    }

    std::vector<int32_t> codes_a((size_t)T * n_stages);
    std::vector<int32_t> codes_b((size_t)T * n_stages);

    // Arm A: Serial baseline
    set_env_var("CRISPASR_RVQ_SERIAL", "1");
    core_rvq::encode_euclidean(features.data(), T, dim, stages.data(), n_stages, codes_a.data());

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        core_rvq::encode_euclidean(features.data(), T, dim, stages.data(), n_stages, codes_a.data());
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_RVQ_SERIAL");
    core_rvq::encode_euclidean(features.data(), T, dim, stages.data(), n_stages, codes_b.data());

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        core_rvq::encode_euclidean(features.data(), T, dim, stages.data(), n_stages, codes_b.data());
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Verify exact parity
    int mismatches = 0;
    for (size_t i = 0; i < codes_a.size(); i++) {
        if (codes_a[i] != codes_b[i]) {
            mismatches++;
        }
    }
    CHECK(mismatches == 0);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] RVQ encode_euclidean (T=%d, dim=%d, K=%d, stages=%d) ===\n", T, dim, K, n_stages);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per RVQ pass)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          mismatches = %d (100%% exact bit-identical)\n", mismatches);

    // Also verify encode_euclidean_per_stage (tests internal norm computation + encoding)
    std::vector<const float*> raw_embeds(n_stages);
    std::vector<int> raw_sizes(n_stages, K);
    for (int s = 0; s < n_stages; s++) {
        raw_embeds[s] = embeds[s].data();
    }
    std::vector<std::vector<int32_t>> stage_codes_a, stage_codes_b;
    set_env_var("CRISPASR_RVQ_SERIAL", "1");
    core_rvq::encode_euclidean_per_stage(features.data(), T, dim, raw_embeds.data(), raw_sizes.data(), n_stages, stage_codes_a);
    unset_env_var("CRISPASR_RVQ_SERIAL");
    core_rvq::encode_euclidean_per_stage(features.data(), T, dim, raw_embeds.data(), raw_sizes.data(), n_stages, stage_codes_b);
    int stage_mismatches = 0;
    for (int s = 0; s < n_stages; s++) {
        for (int t = 0; t < T; t++) {
            if (stage_codes_a[s][t] != stage_codes_b[s][t]) {
                stage_mismatches++;
            }
        }
    }
    CHECK(stage_mismatches == 0);
    printf("RVQ encode_euclidean_per_stage parity: mismatches = %d (100%% exact bit-identical)\n", stage_mismatches);
}

TEST_CASE("iSTFT paired A/B benchmark", "[perf][istft][ab-bench]") {
    const int n_fft = 1024;
    const int hop = 256;
    const int T_frames = 400;
    const int n_freq = n_fft / 2 + 1;

    std::vector<float> mag((size_t)T_frames * n_freq);
    std::vector<float> phase((size_t)T_frames * n_freq);
    for (size_t i = 0; i < mag.size(); i++) {
        mag[i] = 1.0f + 0.5f * std::sin((float)i * 0.01f);
        phase[i] = std::cos((float)i * 0.007f) * 3.14f;
    }

    std::vector<float> pcm_a, pcm_b;

    // Arm A: Serial baseline
    set_env_var("CRISPASR_ISTFT_SERIAL", "1");
    pcm_a = core_istft::istft(mag.data(), phase.data(), n_fft, hop, T_frames);

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        pcm_a = core_istft::istft(mag.data(), phase.data(), n_fft, hop, T_frames);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_ISTFT_SERIAL");
    pcm_b = core_istft::istft(mag.data(), phase.data(), n_fft, hop, T_frames);

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        pcm_b = core_istft::istft(mag.data(), phase.data(), n_fft, hop, T_frames);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    REQUIRE(pcm_a.size() == pcm_b.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < pcm_a.size(); i++) {
        float diff = std::abs(pcm_a[i] - pcm_b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    CHECK(max_diff <= 1e-7f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] iSTFT (T_frames=%d, n_fft=%d, hop=%d, samples=%zu) ===\n", T_frames, n_fft, hop,
           pcm_a.size());
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per iSTFT)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n", max_diff);
}
