// test_slice_timestamps.cpp — unit tests for crispasr_fixed_chunk_slices
// t0_cs / t1_cs timestamp math.
//
// PR #96 changed the server to route all audio through
// crispasr_compute_audio_slices (which falls back to
// crispasr_energy_chunk_slices / crispasr_fixed_chunk_slices when no VAD
// model is configured).  Each slice's t0_cs is passed as the timestamp base
// to backend->transcribe, so wrong t0_cs values silently shift every segment
// timestamp in the response.
//
// These tests pin the t0_cs / t1_cs math so a refactor of the slicer can't
// drift the offset without breaking the suite.  Pure CPU, no model load,
// sub-millisecond.

#include "crispasr_vad.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

static constexpr int SR = 16000; // 16 kHz

// ──────────────────────────────────────────────────────────────────────────────
// crispasr_fixed_chunk_slices
// ──────────────────────────────────────────────────────────────────────────────

TEST_CASE("fixed_chunk_slices: empty input returns no slices", "[unit][slice-ts]") {
    auto slices = crispasr_fixed_chunk_slices(0, SR, 30);
    REQUIRE(slices.empty());

    auto neg = crispasr_fixed_chunk_slices(-1, SR, 30);
    REQUIRE(neg.empty());
}

TEST_CASE("fixed_chunk_slices: audio shorter than chunk → single slice", "[unit][slice-ts]") {
    // 5 s audio, 30 s chunk → one slice covering everything.
    const int n = 5 * SR;
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].start == 0);
    REQUIRE(slices[0].end == n);
    REQUIRE(slices[0].t0_cs == 0);
    // t1_cs = floor(5 s * 100) = 500 cs
    REQUIRE(slices[0].t1_cs == 500);
}

TEST_CASE("fixed_chunk_slices: audio exactly equal to chunk → single slice", "[unit][slice-ts]") {
    const int n = 30 * SR;
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].start == 0);
    REQUIRE(slices[0].end == n);
    REQUIRE(slices[0].t0_cs == 0);
    REQUIRE(slices[0].t1_cs == 3000); // 30 s * 100 cs/s
}

TEST_CASE("fixed_chunk_slices: two-chunk audio has correct t0_cs", "[unit][slice-ts]") {
    // 60 s audio, 30 s chunk → 2 slices.
    // Slice 0: t0_cs=0,    t1_cs=3000
    // Slice 1: t0_cs=3000, t1_cs=6000
    const int n = 60 * SR;
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    REQUIRE(slices.size() == 2);

    REQUIRE(slices[0].start == 0);
    REQUIRE(slices[0].end == 30 * SR);
    REQUIRE(slices[0].t0_cs == 0);
    REQUIRE(slices[0].t1_cs == 3000);

    REQUIRE(slices[1].start == 30 * SR);
    REQUIRE(slices[1].end == 60 * SR);
    REQUIRE(slices[1].t0_cs == 3000);
    REQUIRE(slices[1].t1_cs == 6000);
}

TEST_CASE("fixed_chunk_slices: three chunks — t0_cs increments correctly", "[unit][slice-ts]") {
    const int n = 3 * 30 * SR; // 90 s
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    REQUIRE(slices.size() == 3);
    for (size_t i = 0; i < slices.size(); ++i) {
        int64_t expected_t0 = (int64_t)(i * 30) * 100;
        int64_t expected_t1 = (int64_t)((i + 1) * 30) * 100;
        REQUIRE(slices[i].t0_cs == expected_t0);
        REQUIRE(slices[i].t1_cs == expected_t1);
    }
}

TEST_CASE("fixed_chunk_slices: slices cover input without gaps or overlap", "[unit][slice-ts]") {
    const int n = 75 * SR; // 75 s — not a multiple of 30
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    REQUIRE(slices.size() == 3); // 30+30+15

    // First slice starts at 0, last ends at n.
    REQUIRE(slices.front().start == 0);
    REQUIRE(slices.back().end == n);

    // Contiguous: end[i] == start[i+1].
    for (size_t i = 1; i < slices.size(); ++i)
        REQUIRE(slices[i].start == slices[i - 1].end);
}

TEST_CASE("fixed_chunk_slices: t0_cs of slice N == t1_cs of slice N-1", "[unit][slice-ts]") {
    // The server passes sl.t0_cs to backend->transcribe as the absolute
    // timestamp base.  This test pins that adjacent slices have no gap in
    // their timestamp ranges — a gap would produce a jump in the VTT/SRT
    // output after PR #96 integrated slice-based timestamps.
    const int n = 90 * SR;
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    for (size_t i = 1; i < slices.size(); ++i)
        REQUIRE(slices[i].t0_cs == slices[i - 1].t1_cs);
}

TEST_CASE("fixed_chunk_slices: t0_cs equals start_sample / SR * 100", "[unit][slice-ts]") {
    // Explicit formula check: t0_cs = round(start / SR * 100).
    // Integer samples at 16 kHz → 30 s * 16000 = 480000 samples exactly,
    // so the double conversion is exact.
    const int n = 60 * SR;
    auto slices = crispasr_fixed_chunk_slices(n, SR, 30);

    for (const auto& sl : slices) {
        int64_t expected = (int64_t)((double)sl.start / SR * 100.0);
        REQUIRE(sl.t0_cs == expected);
    }
}
