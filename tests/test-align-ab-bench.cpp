// test-align-ab-bench.cpp — Paired A/B benchmark for CTC forced alignment optimization.
//
// Follows crispasr-crispembed-dev.md Rule 4/4a/5:
//   - Rigorous quality evaluation:
//       * Word timestamp parity: t0 and t1 match bit-for-bit
//       * Character timestamp parity: character spans match bit-for-bit
//       * Exact word sequence preservation
//   - Rigorous speed evaluation:
//       * Cold warmup runs (discarded)
//       * 10 warm timed runs per arm
//       * Median, mean, stddev, and speedup factor: Median(Serial) / Median(Parallel)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "align.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct BenchStats {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
};

BenchStats compute_stats(std::vector<double>& times_ms) {
    REQUIRE(!times_ms.empty());
    std::sort(times_ms.begin(), times_ms.end());

    BenchStats s;
    s.min_ms = times_ms.front();
    s.max_ms = times_ms.back();

    double sum = 0.0;
    for (double v : times_ms)
        sum += v;
    s.mean_ms = sum / static_cast<double>(times_ms.size());

    const size_t n = times_ms.size();
    if (n % 2 == 1) {
        s.median_ms = times_ms[n / 2];
    } else {
        s.median_ms = 0.5 * (times_ms[n / 2 - 1] + times_ms[n / 2]);
    }

    double var_sum = 0.0;
    for (double v : times_ms) {
        const double diff = v - s.mean_ms;
        var_sum += diff * diff;
    }
    s.stddev_ms = std::sqrt(var_sum / static_cast<double>(times_ms.size()));
    return s;
}

std::vector<float> generate_deterministic_logits(int T, int V) {
    std::vector<float> logits(static_cast<size_t>(T) * static_cast<size_t>(V));
    uint32_t state = 42u;
    for (size_t i = 0; i < logits.size(); i++) {
        state = state * 1664525u + 1013904223u;
        logits[i] = static_cast<float>((state >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
    }
    return logits;
}

} // namespace

TEST_CASE("CTC forced alignment paired A/B benchmark (Serial vs Accelerated)", "[align][ab-bench]") {
    // 1. Setup realistic ASR CTC vocabulary and word transcript
    const int T = 1500;  // 30s audio at 50 fps (or 15s at 100 fps)
    const int V = 16384; // Typical Canary/Conformer subword CTC vocabulary
    const float frame_dur = 0.02f;
    const int blank_id = 0;

    std::vector<std::string> vocab(static_cast<size_t>(V));
    vocab[0] = "<blank>";
    vocab[1] = "|";
    for (int i = 0; i < 26; i++) {
        vocab[static_cast<size_t>(i + 2)] = std::string(1, static_cast<char>('a' + i));
    }
    for (int i = 28; i < V; i++) {
        vocab[static_cast<size_t>(i)] = "tok_" + std::to_string(i);
    }

    const std::vector<std::string> words = {"and",  "so",      "my",  "fellow", "americans", "ask",    "not", "what",
                                            "your", "country", "can", "do",     "for",       "you",    "ask", "what",
                                            "you",  "can",     "do",  "for",    "your",      "country"};

    const auto logits = generate_deterministic_logits(T, V);

    // Ensure CTC optimization gate is active
    setenv("CRISPASR_CTC_ALIGN_OPT", "1", 1);

    // Warmups (discarded)
    setenv("CRISPASR_CTC_ALIGN_SERIAL", "1", 1);
    auto warmup_a = ctc_forced_align(logits.data(), T, V, words, vocab, blank_id, frame_dur);
    unsetenv("CRISPASR_CTC_ALIGN_SERIAL");
    auto warmup_b = ctc_forced_align(logits.data(), T, V, words, vocab, blank_id, frame_dur);

    REQUIRE(warmup_a.size() == words.size());
    REQUIRE(warmup_b.size() == words.size());

    constexpr int N_RUNS = 10;

    // 2. Measure Arm A: Serial Baseline (CRISPASR_CTC_ALIGN_SERIAL=1)
    setenv("CRISPASR_CTC_ALIGN_SERIAL", "1", 1);
    std::vector<double> times_serial;
    times_serial.reserve(N_RUNS);
    std::vector<ctc_word_stamp> result_serial;

    for (int r = 0; r < N_RUNS; r++) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        result_serial = ctc_forced_align(logits.data(), T, V, words, vocab, blank_id, frame_dur);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_serial.push_back(ms);
    }

    // 3. Measure Arm B: Accelerated (Multithreaded chunk-parallel)
    unsetenv("CRISPASR_CTC_ALIGN_SERIAL");
    std::vector<double> times_parallel;
    times_parallel.reserve(N_RUNS);
    std::vector<ctc_word_stamp> result_parallel;

    for (int r = 0; r < N_RUNS; r++) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        result_parallel = ctc_forced_align(logits.data(), T, V, words, vocab, blank_id, frame_dur);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_parallel.push_back(ms);
    }

    // 4. Mathematical and Float Parity Verification
    REQUIRE(result_serial.size() == result_parallel.size());
    for (size_t wi = 0; wi < result_serial.size(); wi++) {
        CHECK(result_serial[wi].word == result_parallel[wi].word);
        CHECK(result_serial[wi].t0 == result_parallel[wi].t0);
        CHECK(result_serial[wi].t1 == result_parallel[wi].t1);
        REQUIRE(result_serial[wi].characters.size() == result_parallel[wi].characters.size());
        for (size_t ci = 0; ci < result_serial[wi].characters.size(); ci++) {
            CHECK(result_serial[wi].characters[ci].character == result_parallel[wi].characters[ci].character);
            CHECK(result_serial[wi].characters[ci].t0 == result_parallel[wi].characters[ci].t0);
            CHECK(result_serial[wi].characters[ci].t1 == result_parallel[wi].characters[ci].t1);
        }
    }

    // 5. Statistics & Report
    const auto stats_a = compute_stats(times_serial);
    const auto stats_b = compute_stats(times_parallel);
    const double speedup = stats_a.median_ms / stats_b.median_ms;

    printf("\n======================================================================\n");
    printf("CTC Forced Alignment Paired A/B Benchmark (T=%d frames, V=%d vocab)\n", T, V);
    printf("======================================================================\n");
    printf("PARITY: 100%% exact bit-identical timestamps across %zu words\n", result_serial.size());
    printf("  Arm A (Serial Baseline):     median %6.2f ms (mean %6.2f ± %4.2f ms) [min %6.2f, max %6.2f]\n",
           stats_a.median_ms, stats_a.mean_ms, stats_a.stddev_ms, stats_a.min_ms, stats_a.max_ms);
    printf("  Arm B (Accelerated Parallel): median %6.2f ms (mean %6.2f ± %4.2f ms) [min %6.2f, max %6.2f]\n",
           stats_b.median_ms, stats_b.mean_ms, stats_b.stddev_ms, stats_b.min_ms, stats_b.max_ms);
    printf("  SPEEDUP FACTOR:              %6.2fx faster (%6.2f ms saved per alignment)\n", speedup,
           stats_a.median_ms - stats_b.median_ms);
    printf("======================================================================\n");

    CHECK(speedup >= 1.5);
}
