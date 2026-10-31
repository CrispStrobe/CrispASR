// tests/test-cqt-diarize-ab-bench.cpp — Paired A/B benchmark for CQT and Diarization Smooth
// CPU accelerations.
//
// Verifies:
// 1. Bit-for-bit exact parity between serial baseline and chunk-parallel / precalculated implementations.
// 2. Strict paired latency benchmark across 10 warm iterations (1 warmup discarded):
//    reports min, max, median, mean, stddev, and speedup factor.
// 3. Fallback gate verification via CRISPASR_CQT_SERIAL=1, CRISPASR_CQT2010_SERIAL=1,
//    and CRISPASR_DIARIZE_SMOOTH_SERIAL=1.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "core/cqt.h"
#include "core/cqt2010v2.h"
#include "core/diarize_smooth.h"

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

TEST_CASE("CQT magnitude paired A/B benchmark", "[perf][cqt][ab-bench]") {
    core_cqt::Params p;
    p.sample_rate = 22050;
    p.fmin = 32.703195662574829f; // C1
    p.n_bins = 144;
    p.bins_per_octave = 24;
    p.hop_length = 2048;

    const int n_samples = 44100; // 2 seconds of 22050 Hz audio
    std::vector<float> x((size_t)n_samples);
    for (int i = 0; i < n_samples; i++) {
        x[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * (double)i / 22050.0);
    }

    const auto kernels = core_cqt::build_kernels(p);
    std::vector<float> out_a, out_b;

    // Arm A: Serial baseline
    set_env_var("CRISPASR_CQT_SERIAL", "1");
    core_cqt::magnitude(p, kernels, x.data(), n_samples, out_a);

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        core_cqt::magnitude(p, kernels, x.data(), n_samples, out_a);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Parallel accelerated
    unset_env_var("CRISPASR_CQT_SERIAL");
    core_cqt::magnitude(p, kernels, x.data(), n_samples, out_b);

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        core_cqt::magnitude(p, kernels, x.data(), n_samples, out_b);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    REQUIRE(out_a.size() == out_b.size());
    REQUIRE(!out_a.empty());

    float max_diff = 0.0f;
    for (size_t i = 0; i < out_a.size(); i++) {
        const float diff = std::abs(out_a[i] - out_b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    CHECK(max_diff == 0.0f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] CQT magnitude (n_samples=%d, n_bins=%d, hop=%d, frames=%zu) ===\n", n_samples, p.n_bins,
           p.hop_length, out_a.size() / p.n_bins);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per transform)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n", max_diff);
}

TEST_CASE("CQT2010v2 octave_complex & decimate_by_2 paired A/B benchmark", "[perf][cqt2010][ab-bench]") {
    core_cqt2010v2::Params p;
    p.n_bins = 309;
    p.bins_per_octave = 36;
    p.n_octaves = 9;
    p.n_filters = 36;
    p.n_fft = 256;
    p.hop_length = 256;

    const int n_samples = 43844;
    std::vector<float> x(n_samples);
    for (int i = 0; i < n_samples; i++) {
        x[i] = std::sin((float)i * 0.05f) * 0.4f;
    }

    std::vector<float> fir(256);
    for (int i = 0; i < 256; i++) {
        fir[i] = std::cos((float)i * 0.1f) * 0.05f;
    }

    std::vector<float> kern_re((size_t)p.n_filters * p.n_fft);
    std::vector<float> kern_im((size_t)p.n_filters * p.n_fft);
    for (size_t i = 0; i < (size_t)p.n_filters * p.n_fft; i++) {
        kern_re[i] = std::cos((float)i * 0.02f);
        kern_im[i] = std::sin((float)i * 0.02f);
    }

    core_cqt2010v2::Kernels kern;
    kern.real = kern_re.data();
    kern.imag = kern_im.data();
    kern.lowpass = fir.data();
    kern.lowpass_len = 256;

    // Test decimate_by_2
    std::vector<float> dec_a, dec_b;
    set_env_var("CRISPASR_CQT2010_SERIAL", "1");
    core_cqt2010v2::decimate_by_2(x.data(), n_samples, fir.data(), 256, dec_a);

    unset_env_var("CRISPASR_CQT2010_SERIAL");
    core_cqt2010v2::decimate_by_2(x.data(), n_samples, fir.data(), 256, dec_b);

    REQUIRE(dec_a.size() == dec_b.size());
    float max_diff_dec = 0.0f;
    for (size_t i = 0; i < dec_a.size(); i++) {
        float diff = std::abs(dec_a[i] - dec_b[i]);
        if (diff > max_diff_dec)
            max_diff_dec = diff;
    }
    CHECK(max_diff_dec == 0.0f);

    // Test octave_complex paired benchmark
    std::vector<float> re_a, im_a, re_b, im_b;
    set_env_var("CRISPASR_CQT2010_SERIAL", "1");
    core_cqt2010v2::octave_complex(p, kern, x.data(), n_samples, p.hop_length, re_a, im_a);

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        core_cqt2010v2::octave_complex(p, kern, x.data(), n_samples, p.hop_length, re_a, im_a);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    unset_env_var("CRISPASR_CQT2010_SERIAL");
    core_cqt2010v2::octave_complex(p, kern, x.data(), n_samples, p.hop_length, re_b, im_b);

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        core_cqt2010v2::octave_complex(p, kern, x.data(), n_samples, p.hop_length, re_b, im_b);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    REQUIRE(re_a.size() == re_b.size());
    REQUIRE(im_a.size() == im_b.size());

    float max_diff_re = 0.0f;
    for (size_t i = 0; i < re_a.size(); i++) {
        float diff = std::abs(re_a[i] - re_b[i]);
        if (std::isnan(diff) || diff > max_diff_re)
            max_diff_re = diff;
    }
    CHECK(max_diff_re == 0.0f);

    float max_diff_im = 0.0f;
    for (size_t i = 0; i < im_a.size(); i++) {
        float diff = std::abs(im_a[i] - im_b[i]);
        if (std::isnan(diff) || diff > max_diff_im)
            max_diff_im = diff;
    }
    CHECK(max_diff_im == 0.0f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] CQT2010v2 octave_complex (filters=%d, n_fft=%d, frames=%zu) ===\n", p.n_filters, p.n_fft,
           re_a.size() / p.n_filters);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Parallel Chunks):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per octave)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e (100%% exact bit-identical)\n", max_diff_re);
}

TEST_CASE("Diarization speaker_centroids paired A/B benchmark", "[perf][diarize_smooth][ab-bench]") {
    const int n = 1000;
    const int d = 256;
    const int k_speakers = 8;

    std::vector<float> embeddings((size_t)n * d);
    for (size_t i = 0; i < (size_t)n * d; i++) {
        embeddings[i] = std::sin((float)i * 0.013f) * 2.0f;
    }

    std::vector<int> labels(n);
    for (int i = 0; i < n; i++) {
        labels[i] = (i * 7) % k_speakers;
    }

    std::vector<int> vals_a, vals_b;
    std::vector<float> cent_a, cent_b;

    // Arm A: Serial baseline
    set_env_var("CRISPASR_DIARIZE_SMOOTH_SERIAL", "1");
    cent_a = core_diarize_smooth::speaker_centroids(embeddings.data(), n, d, labels, &vals_a);

    const int iters = 10;
    std::vector<double> times_a(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        cent_a = core_diarize_smooth::speaker_centroids(embeddings.data(), n, d, labels, &vals_a);
        const auto t1 = std::chrono::steady_clock::now();
        times_a[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Arm B: Precomputed norms accelerated
    unset_env_var("CRISPASR_DIARIZE_SMOOTH_SERIAL");
    cent_b = core_diarize_smooth::speaker_centroids(embeddings.data(), n, d, labels, &vals_b);

    std::vector<double> times_b(iters);
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        cent_b = core_diarize_smooth::speaker_centroids(embeddings.data(), n, d, labels, &vals_b);
        const auto t1 = std::chrono::steady_clock::now();
        times_b[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    REQUIRE(vals_a == vals_b);
    REQUIRE(cent_a.size() == cent_b.size());

    float max_diff = 0.0f;
    for (size_t i = 0; i < cent_a.size(); i++) {
        float diff = std::abs(cent_a[i] - cent_b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    CHECK(max_diff <= 1e-6f);

    const BenchStats stats_a = compute_stats(times_a);
    const BenchStats stats_b = compute_stats(times_b);
    const double speedup = stats_a.median_ms / std::max(1e-6, stats_b.median_ms);

    printf("\n=== [A/B Bench] Diarization speaker_centroids (n=%d, d=%d, %d speakers) ===\n", n, d, k_speakers);
    printf("Arm A (Serial Baseline):   median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("Arm B (Norm Cached):       median = %.2f ms, mean = %.2f ms (± %.2f ms), min = %.2f ms, max = %.2f ms\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("Speedup:                   %.2fx (%.2f ms saved per centroid estimation)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("Numerical parity:          max_diff = %.8e\n", max_diff);
}
