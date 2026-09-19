// test-parakeet-strategy.cpp — unit tests for parakeet_pick_strategy, the pure
// long-audio routing decision hoisted into the shared orchestration
// (improvements Phase 1). Pinning it here dedups the decision that used to be
// written twice (CLI adapter + session C-ABI) and was the source of the JA /
// #257 routing bugs.

#include <catch2/catch_test_macros.hpp>

#include "parakeet_orchestrate.h"

namespace {
constexpr int SR = 16000;
parakeet_strategy_in mk(int secs, bool is_ja, bool chunk_explicit, int chunk_s, int thr, bool longform,
                        bool chunked_requested = false) {
    parakeet_strategy_in in;
    in.n_samples = secs * SR;
    in.sample_rate = SR;
    in.is_ja = is_ja;
    in.chunk_seconds_explicit = chunk_explicit;
    in.chunk_seconds = chunk_s;
    in.stream_threshold_s = thr;
    in.longform_enabled = longform;
    in.chunked_requested = chunked_requested;
    return in;
}

// What the orchestrator does: resolve the effective cap, then route.
parakeet_strategy route(parakeet_strategy_in in, bool threshold_from_env = false) {
    in.stream_threshold_s = parakeet_effective_single_pass_cap_s(in, threshold_from_env);
    return parakeet_pick_strategy(in);
}
} // namespace

TEST_CASE("non-JA explicit --chunk-seconds → CHUNK_SEGMENTED at any length",
          "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(20, false, true, 7, 300, true)) == parakeet_strategy::CHUNK_SEGMENTED);
    REQUIRE(parakeet_pick_strategy(mk(600, false, true, 7, 300, true)) == parakeet_strategy::CHUNK_SEGMENTED);
}

TEST_CASE("non-JA short (<= cap) → SINGLE_PASS", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(11, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(225, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(300, false, false, 0, 300, true)) == parakeet_strategy::SINGLE_PASS);
}

TEST_CASE("non-JA long (> cap) + longform → LONGFORM", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(301, false, false, 0, 300, true)) == parakeet_strategy::LONGFORM);
}

TEST_CASE("non-JA long + longform disabled → STREAMED", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(600, false, false, 0, 300, false)) == parakeet_strategy::STREAMED);
}

TEST_CASE("threshold 0 (always-streamed) → STREAMED", "[unit][parakeet-strategy][improvements]") {
    REQUIRE(parakeet_pick_strategy(mk(11, false, false, 0, 0, false)) == parakeet_strategy::STREAMED);
    REQUIRE(parakeet_pick_strategy(mk(600, false, false, 0, 0, false)) == parakeet_strategy::STREAMED);
}

TEST_CASE("JA never takes the CHUNK_SEGMENTED branch", "[unit][parakeet-strategy][improvements]") {
    // JA caller passes threshold=12, longform=false. Explicit chunk is ignored
    // (is_ja gate) → falls to the length logic.
    REQUIRE(parakeet_pick_strategy(mk(8, true, true, 7, 12, false)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_pick_strategy(mk(30, true, true, 7, 12, false)) == parakeet_strategy::STREAMED);
}

// ---- Issue #350: the chunked entry point must stay bounded ----

TEST_CASE("chunked request with default length is sliced, not one full pass", "[unit][parakeet-strategy][issue-350]") {
    // crispasr_session_transcribe_chunked(…, chunk_seconds = 0, …): "use
    // per-model defaults". The 300 s cap made 30-300 s files ONE decode — the
    // regression. Anything past the reliable window is LONGFORM now.
    REQUIRE(route(mk(231, false, false, 0, 300, true, /*chunked=*/true)) == parakeet_strategy::LONGFORM);
    REQUIRE(route(mk(60, false, false, 0, 300, true, true)) == parakeet_strategy::LONGFORM);
    REQUIRE(route(mk(31, false, false, 0, 300, true, true)) == parakeet_strategy::LONGFORM);
    // …and the slices are the reliable window, not the 300 s cap.
    REQUIRE(parakeet_effective_single_pass_cap_s(mk(231, false, false, 0, 300, true, true), false) ==
            kParakeetBoundedWindowS);
}

TEST_CASE("a chunked request short enough for one pass still takes it", "[unit][parakeet-strategy][issue-350]") {
    REQUIRE(route(mk(11, false, false, 0, 300, true, true)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(route(mk(30, false, false, 0, 300, true, true)) == parakeet_strategy::SINGLE_PASS);
}

TEST_CASE("no chunked request → the 300 s cap is untouched", "[unit][parakeet-strategy][issue-350]") {
    REQUIRE(route(mk(231, false, false, 0, 300, true, /*chunked=*/false)) == parakeet_strategy::SINGLE_PASS);
    REQUIRE(parakeet_effective_single_pass_cap_s(mk(231, false, false, 0, 300, true, false), false) == 300);
}

TEST_CASE("an explicit threshold or chunk length wins over the bounded cap", "[unit][parakeet-strategy][issue-350]") {
    // CRISPASR_PARAKEET_STREAM_THRESHOLD=200 stays 200 even for a chunked call.
    REQUIRE(parakeet_effective_single_pass_cap_s(mk(231, false, false, 0, 200, true, true), /*from_env=*/true) == 200);
    REQUIRE(route(mk(231, false, false, 0, 200, true, true), true) == parakeet_strategy::LONGFORM);
    // An explicit chunk length routes to CHUNK_SEGMENTED before any cap.
    REQUIRE(route(mk(231, false, true, 20, 300, true, true)) == parakeet_strategy::CHUNK_SEGMENTED);
    // Threshold 0 (always-streamed) is not "raised" to the bounded window.
    REQUIRE(parakeet_effective_single_pass_cap_s(mk(231, false, false, 0, 0, false, true), false) == 0);
    REQUIRE(route(mk(231, false, false, 0, 0, false, true)) == parakeet_strategy::STREAMED);
}

TEST_CASE("JA chunked request keeps the JA routing", "[unit][parakeet-strategy][issue-350]") {
    // JA passes longform=false, so above its 12 s window it streams as before.
    REQUIRE(route(mk(231, true, false, 0, 12, false, true)) == parakeet_strategy::STREAMED);
    REQUIRE(route(mk(8, true, false, 0, 12, false, true)) == parakeet_strategy::SINGLE_PASS);
}

// ---- Issue #350: gap detection (the repair pass's decision) ----

namespace {
using iv = std::pair<int64_t, int64_t>;
} // namespace

TEST_CASE("a hole in the word timeline is found, ordinary pauses are not", "[unit][parakeet-strategy][issue-350]") {
    // Words up to 100 cs, then nothing until 5000 cs, then words to 6000 cs.
    const std::vector<iv> covered = {{0, 50}, {50, 100}, {5000, 5500}, {5500, 6000}};
    const auto gaps = parakeet_find_gaps(covered, 0, 6000, /*min_gap*/ 300, /*slop*/ 30, /*max_window*/ 0);
    REQUIRE(gaps.size() == 1);
    CHECK(gaps[0].first == 100);
    CHECK(gaps[0].second == 5000);

    // The same timeline with a 2 s pause instead: below min_gap → no repair.
    const std::vector<iv> paused = {{0, 50}, {50, 100}, {300, 500}};
    CHECK(parakeet_find_gaps(paused, 0, 500, 300, 30, 3000).empty());
}

TEST_CASE("a long hole is split into reliable-window pieces", "[unit][parakeet-strategy][issue-350]") {
    // 90 s hole, 30 s repair window → 3 pieces that tile it exactly.
    const std::vector<iv> covered = {{0, 100}, {9100, 9200}};
    const auto gaps = parakeet_find_gaps(covered, 0, 9200, 300, 30, 3000);
    REQUIRE(gaps.size() == 3);
    CHECK(gaps.front().first == 100);
    CHECK(gaps.back().second == 9100);
    for (size_t i = 0; i < gaps.size(); i++) {
        CHECK(gaps[i].second - gaps[i].first <= 3000);
        if (i)
            CHECK(gaps[i].first == gaps[i - 1].second); // no seam left uncovered
    }
}

TEST_CASE("gaps at the head and tail of the span count too", "[unit][parakeet-strategy][issue-350]") {
    const std::vector<iv> covered = {{1000, 1100}};
    const auto gaps = parakeet_find_gaps(covered, 0, 5000, 300, 30, 0);
    REQUIRE(gaps.size() == 2);
    CHECK(gaps[0] == iv{0, 1000});
    CHECK(gaps[1] == iv{1100, 5000});
}

TEST_CASE("gap detection is safe on degenerate input", "[unit][parakeet-strategy][issue-350]") {
    CHECK(parakeet_find_gaps({}, 0, 0, 300, 30, 3000).empty());       // empty span
    CHECK(parakeet_find_gaps({}, 0, 5000, 0, 30, 3000).empty());      // min_gap 0 = off
    CHECK(parakeet_find_gaps({}, 0, 5000, 300, 30, 3000).size() > 0); // no words at all → all gap
    // Unsorted, overlapping input is normalized rather than trusted.
    const std::vector<iv> messy = {{5000, 5500}, {0, 100}, {40, 60}};
    const auto gaps = parakeet_find_gaps(messy, 0, 5500, 300, 30, 0);
    REQUIRE(gaps.size() == 1);
    CHECK(gaps[0] == iv{100, 5000});
}

// ---- Phase 2: memory policy ----

TEST_CASE("est_singlepass_peak_mb matches the reporter's ~1.9 GiB data point",
          "[unit][parakeet-strategy][improvements]") {
    // ~4 min clip → T_enc≈2800, 8 heads, default coeff 8.0 → ~1914 MiB.
    const double est = parakeet_est_singlepass_peak_mb(2800, 8, 8.0);
    REQUIRE(est > 1800.0);
    REQUIRE(est < 2100.0);
    // O(T^2): doubling T ~quadruples the estimate.
    REQUIRE(parakeet_est_singlepass_peak_mb(5600, 8, 8.0) > 3.9 * est);
    // degenerate inputs → 0
    REQUIRE(parakeet_est_singlepass_peak_mb(0, 8, 8.0) == 0.0);
    REQUIRE(parakeet_est_singlepass_peak_mb(2800, 0, 8.0) == 0.0);
}

TEST_CASE("singlepass_fits_budget gating", "[unit][parakeet-strategy][improvements]") {
    // Disabled: budget 0 or coeff 0 → always fits (historical behaviour).
    REQUIRE(parakeet_singlepass_fits_budget(9999, 8, 0.0, 8.0));
    REQUIRE(parakeet_singlepass_fits_budget(9999, 8, 100.0, 0.0));
    // ~4 min clip fits a 3.7 GiB card but not a 1.5 GiB budget.
    REQUIRE(parakeet_singlepass_fits_budget(2800, 8, 3700.0, 8.0));
    REQUIRE_FALSE(parakeet_singlepass_fits_budget(2800, 8, 1500.0, 8.0));
    // A short clip (T≈250, 20 s) fits even a tiny budget.
    REQUIRE(parakeet_singlepass_fits_budget(250, 8, 256.0, 8.0));
}

TEST_CASE("issue #441: the 47.5-minute input cannot pass a real machine's budget",
          "[unit][parakeet-strategy][issue441]") {
    // The reported crash: 2850.1 s @ 16 kHz on a 15 GB box, CPU build. hop=160,
    // subsampling=8, so T_enc = (n/160)/8. The single-pass estimate must exceed
    // half of that machine's memory by a wide margin, which is what makes the
    // policy switch to STREAMED before allocating anything.
    const int T_enc = (int)((2850.1 * 16000) / 160) / 8;
    REQUIRE(T_enc == 35626);

    const double budget_15gb_box = 0.5 * 15.0 * 1024.0; // the new default: half of MemAvailable
    REQUIRE_FALSE(parakeet_singlepass_fits_budget(T_enc, 8, budget_15gb_box, 8.0));

    // NEGATIVE CONTROL. The bug was that an UNSET budget meant "disabled", and
    // a disabled policy says "fits" for any input at all. If this ever starts
    // returning false, the guard above is no longer testing the thing that
    // actually changed -- it would be passing because the predicate refuses
    // everything, not because the budget is being applied.
    REQUIRE(parakeet_singlepass_fits_budget(T_enc, 8, 0.0, 8.0));

    // And the guard must NOT fire on ordinary input on the same machine: a
    // 60 s clip has to stay on the seamless single-pass path.
    const int T_60s = (int)((60.0 * 16000) / 160) / 8;
    REQUIRE(parakeet_singlepass_fits_budget(T_60s, 8, budget_15gb_box, 8.0));

    // The allocation-site guard is independent of routing knobs. Even a
    // caller that forces single-pass cannot request the reported graph on a
    // 15 GB machine; a normal 60 s graph remains allowed.
    REQUIRE_FALSE(parakeet_encoder_fits_available_memory(T_enc, 8, 15.0 * 1024.0, 8.0));
    REQUIRE(parakeet_encoder_fits_available_memory(T_60s, 8, 15.0 * 1024.0, 8.0));
}

TEST_CASE("issue #441: reporter's explicit chunk command selects the bounded route",
          "[unit][parakeet-strategy][issue441]") {
    parakeet_strategy_in in;
    in.n_samples = 45602304;
    in.sample_rate = 16000;
    in.is_ja = false;
    in.chunk_seconds_explicit = true;
    in.chunk_seconds = 419;
    in.stream_threshold_s = 300;
    in.longform_enabled = true;
    REQUIRE(parakeet_pick_strategy(in) == parakeet_strategy::CHUNK_SEGMENTED);
}
