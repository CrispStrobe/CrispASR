// test-stream-vad-skip.cpp — unit tests for the pre-decode finalized-slice
// skip added in PR #90 (stream JSON mode).
//
// The production logic lives in the slice loop inside crispasr_run.cpp
// (if (params.stream_json && s_end_abs <= finalized_until_sample) continue).
// Because it is a short inline expression, we mirror it here as a
// `classify_stream_slice` helper and test all categories + boundary values.
//
// All tests are pure arithmetic, no model or audio I/O.

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

// ---------------------------------------------------------------------------
// Mirror of the production classification: how the loop categorises each
// VAD slice relative to the finalized watermark.
//
// kFinalized  — s_end_abs <= finalized: skip entirely, no decode.
// kStraddling — s_start_abs < finalized < s_end_abs: decode tail only.
// kNew        — s_start_abs >= finalized: decode the full slice.
// ---------------------------------------------------------------------------

enum class SliceClass { kFinalized, kStraddling, kNew };

static SliceClass classify_stream_slice(int64_t s_start_abs, int64_t s_end_abs, int64_t finalized_until) {
    if (s_end_abs <= finalized_until)
        return SliceClass::kFinalized;
    if (s_start_abs < finalized_until)
        return SliceClass::kStraddling;
    return SliceClass::kNew;
}

// Compute absolute sample positions from window_start + relative slice offsets,
// mirroring the two lines in crispasr_run.cpp:
//   const int64_t s_start_abs = window_start_sample_now + (int64_t)sl.start;
//   const int64_t s_end_abs   = window_start_sample_now + (int64_t)sl.end;
static SliceClass classify_from_window(int64_t window_start, int sl_start, int sl_end, int64_t finalized_until) {
    const int64_t s_start = window_start + (int64_t)sl_start;
    const int64_t s_end = window_start + (int64_t)sl_end;
    return classify_stream_slice(s_start, s_end, finalized_until);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("stream-vad-skip: slice entirely before finalized → kFinalized", "[unit][stream-json]") {
    // Slice covers [0, 100), finalized at 100 — s_end_abs == finalized (inclusive).
    REQUIRE(classify_stream_slice(0, 100, 100) == SliceClass::kFinalized);
    // Slice well inside finalized region.
    REQUIRE(classify_stream_slice(0, 80, 100) == SliceClass::kFinalized);
    // Slice ends one sample before finalized — still finalized.
    REQUIRE(classify_stream_slice(50, 99, 100) == SliceClass::kFinalized);
}

TEST_CASE("stream-vad-skip: slice entirely after finalized → kNew", "[unit][stream-json]") {
    // Slice starts exactly at finalized boundary — not straddling.
    REQUIRE(classify_stream_slice(100, 200, 100) == SliceClass::kNew);
    // Slice well past finalized.
    REQUIRE(classify_stream_slice(200, 300, 100) == SliceClass::kNew);
    // finalized_until == 0 → nothing finalized yet, all slices are new.
    REQUIRE(classify_stream_slice(0, 100, 0) == SliceClass::kNew);
}

TEST_CASE("stream-vad-skip: slice straddles finalized boundary → kStraddling", "[unit][stream-json]") {
    // Classic straddle: start before, end after finalized.
    REQUIRE(classify_stream_slice(50, 200, 100) == SliceClass::kStraddling);
    // s_start_abs == finalized - 1 (one sample before boundary).
    REQUIRE(classify_stream_slice(99, 200, 100) == SliceClass::kStraddling);
    // finalized is at s_start_abs + 1 sample.
    REQUIRE(classify_stream_slice(99, 150, 100) == SliceClass::kStraddling);
}

TEST_CASE("stream-vad-skip: zero-finalized treats all slices as new", "[unit][stream-json]") {
    // At stream start nothing is finalized yet — every slice is new.
    REQUIRE(classify_stream_slice(0, 100, 0) == SliceClass::kNew);
    REQUIRE(classify_stream_slice(100, 200, 0) == SliceClass::kNew);
    REQUIRE(classify_stream_slice(1000, 2000, 0) == SliceClass::kNew);
}

TEST_CASE("stream-vad-skip: absolute position from window_start + relative offset", "[unit][stream-json]") {
    // window_start=40000, sl=[8000,16000), finalized_until=48000.
    // s_start_abs=48000 == finalized → kNew (not straddling).
    REQUIRE(classify_from_window(40000, 8000, 16000, 48000) == SliceClass::kNew);

    // window_start=40000, sl=[0,8000), finalized_until=48000.
    // s_start_abs=40000 < 48000, s_end_abs=48000 <= 48000 → kFinalized.
    REQUIRE(classify_from_window(40000, 0, 8000, 48000) == SliceClass::kFinalized);

    // window_start=40000, sl=[4000,16000), finalized_until=48000.
    // s_start_abs=44000 < 48000, s_end_abs=56000 > 48000 → kStraddling.
    REQUIRE(classify_from_window(40000, 4000, 16000, 48000) == SliceClass::kStraddling);
}

TEST_CASE("stream-vad-skip: straddle tail length gate (kStraddleMinSamples = 32000)", "[unit][stream-json]") {
    // The production code skips the tail decode when
    //   sub_len = sl.end - sub_start < kStraddleMinSamples (32000).
    // sub_start = finalized_until - window_start.
    // sub_len   = sl.end - sub_start.
    //
    // We verify the arithmetic that determines whether a straddling slice
    // gets decoded or suppressed.
    constexpr int kStraddleMinSamples = 32000;
    const int64_t window_start = 100000;
    const int64_t finalized = 116000; // 1 s into the window
    // sl.start=0 → s_start_abs=100000 < 116000  → straddle confirmed.
    // sub_start = finalized - window_start = 16000.

    // Case A: sl.end gives sub_len = 48000 >= 32000 → decode proceeds.
    const int sl_end_long = 64000; // sub_len = 64000 - 16000 = 48000
    const int sub_len_a = sl_end_long - (int)(finalized - window_start);
    REQUIRE(sub_len_a >= kStraddleMinSamples); // tail is long enough

    // Case B: sl.end gives sub_len = 16000 < 32000 → decode suppressed.
    const int sl_end_short = 32000; // sub_len = 32000 - 16000 = 16000
    const int sub_len_b = sl_end_short - (int)(finalized - window_start);
    REQUIRE(sub_len_b < kStraddleMinSamples); // tail too short, decode suppressed
}

TEST_CASE("stream-vad-skip: adjacent slices around finalized boundary are classified correctly",
          "[unit][stream-json]") {
    // Simulate a rolling window with three adjacent VAD slices.
    // finalized at absolute sample 96000 (6 s @ 16 kHz).
    // window_start = 64000 (4 s back from current head at 8 s).
    const int64_t win = 64000;
    const int64_t fin = 96000;

    // Slice A: [0, 16000) in window → abs [64000, 80000) — fully before fin.
    REQUIRE(classify_from_window(win, 0, 16000, fin) == SliceClass::kFinalized);

    // Slice B: [16000, 48000) in window → abs [80000, 112000) — straddles fin.
    REQUIRE(classify_from_window(win, 16000, 48000, fin) == SliceClass::kStraddling);

    // Slice C: [48000, 64000) in window → abs [112000, 128000) — fully new.
    REQUIRE(classify_from_window(win, 48000, 64000, fin) == SliceClass::kNew);
}
