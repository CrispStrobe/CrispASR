// test-audio-chunking.cpp — unit tests for audio_chunking::find_energy_min_split
//
// Covers the energy-minimum detection logic that find_energy_min_split exposes
// directly. Complements test_chunk_context.cpp which tests the higher-level
// split_at_energy_minima wrapper and the process_slice contract.
//
// All tests are pure CPU — no model load, no audio file, sub-millisecond.

#include <catch2/catch_test_macros.hpp>

#include "../src/core/audio_chunking.h"

#include <cmath>
#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// find_energy_min_split
// ---------------------------------------------------------------------------

TEST_CASE("find_energy_min_split: single quiet window chosen", "[unit][chunking]") {
    // 4 windows of 400 samples: loud, loud, QUIET, loud.
    // Quiet window starts at index 800.
    const size_t WIN = 400;
    std::vector<float> audio(4 * WIN, 1.0f);
    // Zero out window 2 (offset 800) so it has energy 0.
    for (size_t i = 2 * WIN; i < 3 * WIN; ++i)
        audio[i] = 0.0f;

    size_t cut = audio_chunking::find_energy_min_split(audio.data(), 0, audio.size(), WIN);
    // Must point to the start of the silent window.
    REQUIRE(cut == 2 * WIN);
}

TEST_CASE("find_energy_min_split: first window wins on tie", "[unit][chunking]") {
    // All windows have identical energy → the first one is returned.
    const size_t WIN = 200;
    std::vector<float> audio(4 * WIN, 0.5f);

    size_t cut = audio_chunking::find_energy_min_split(audio.data(), 0, audio.size(), WIN);
    // First window start is 0.
    REQUIRE(cut == 0);
}

TEST_CASE("find_energy_min_split: search range excludes quiet window", "[unit][chunking]") {
    // Audio: loud everywhere except window 2 (samples 800-1199).
    // Search only [0, 800) with win_samples=400 (2 windows: window 0 and window 1, both loud).
    // The quiet window 2 is outside the search range — so no window has energy 0.
    // The first loud window (index 0) wins on tie.
    const size_t WIN = 400;
    std::vector<float> audio(4 * WIN, 1.0f);
    for (size_t i = 2 * WIN; i < 3 * WIN; ++i)
        audio[i] = 0.0f;

    // Search [0, 800) — two loud windows. Expect the first (index 0).
    size_t cut = audio_chunking::find_energy_min_split(audio.data(), 0, 2 * WIN, WIN);
    REQUIRE(cut == 0); // first window wins on tie
}

TEST_CASE("find_energy_min_split: degenerate span < win_samples returns midpoint", "[unit][chunking]") {
    // span = 100 < win_samples = 400 → returns search_start + span/2.
    std::vector<float> audio(500, 1.0f);
    size_t cut = audio_chunking::find_energy_min_split(audio.data(), 100, 200, 400);
    REQUIRE(cut == 100 + 50); // start + span/2 = 100 + 50
}

TEST_CASE("find_energy_min_split: empty range returns search_start", "[unit][chunking]") {
    std::vector<float> audio(100, 1.0f);
    size_t cut = audio_chunking::find_energy_min_split(audio.data(), 50, 50, 400);
    REQUIRE(cut == 50);
}

TEST_CASE("find_energy_min_split: last window is quietest", "[unit][chunking]") {
    // Three windows; last one is silence.
    const size_t WIN = 320;
    std::vector<float> audio(3 * WIN, 0.8f);
    for (size_t i = 2 * WIN; i < 3 * WIN; ++i)
        audio[i] = 0.0f;

    size_t cut = audio_chunking::find_energy_min_split(audio.data(), 0, audio.size(), WIN);
    REQUIRE(cut == 2 * WIN);
}

// ---------------------------------------------------------------------------
// split_at_energy_minima: smoke checks not already in test_chunk_context.cpp
// ---------------------------------------------------------------------------

TEST_CASE("split_at_energy_minima: empty input returns no slices", "[unit][chunking]") {
    std::vector<float> empty;
    auto slices = audio_chunking::split_at_energy_minima(empty.data(), 0, 16000, 8000);
    REQUIRE(slices.empty());
}

TEST_CASE("split_at_energy_minima: cut lands in the quiet region", "[unit][chunking]") {
    // 3 seconds at 16 kHz.  max_chunk = 2 s.
    // The search window is the last 1 s of each chunk: [1 s, 2 s) = [SR, 2*SR).
    // Put the quiet window there so the chunker can pick it.
    const size_t SR = 16000;
    std::vector<float> audio(3 * SR, 0.7f);
    // Quiet zone: [SR, 2*SR) — exactly where the chunker's search range lands.
    for (size_t i = SR; i < 2 * SR; ++i)
        audio[i] = 0.0f;

    // max_chunk=2*SR, search_win=SR, win_samples=SR/4 (many windows inside the quiet zone).
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/2 * SR,
                                                         /*search_win=*/SR,
                                                         /*win=*/SR / 4);
    REQUIRE(slices.size() >= 2);
    // The cut must land inside the quiet zone [SR, 2*SR).
    REQUIRE(slices[0].second >= SR);
    REQUIRE(slices[0].second < 2 * SR);
}

TEST_CASE("split_at_energy_minima: slices are non-overlapping and contiguous", "[unit][chunking]") {
    const size_t SR = 16000;
    std::vector<float> audio(5 * SR, 0.4f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/2 * SR,
                                                         /*search_win=*/SR);
    REQUIRE(!slices.empty());
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == audio.size());
    for (size_t i = 1; i < slices.size(); ++i) {
        REQUIRE(slices[i].first == slices[i - 1].second);
        REQUIRE(slices[i].second > slices[i].first); // non-empty
    }
}
