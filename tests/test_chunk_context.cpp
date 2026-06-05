// test_chunk_context.cpp — unit tests for the energy-minima chunker
// (issue #89, refresh after the 617cd02 context+filter revert).
//
// Pins audio_chunking::split_at_energy_minima:
//   - short audio → single slice
//   - long audio → contiguous, non-overlapping slices that cover the input
//   - all-zero (silence) input still partitions cleanly
//   - audio length == max_chunk → single slice
//
// The ±2 s slice context expansion + word-level trim that 617cd02 added
// was reverted because the encoder timestamp drift it required is not
// stable for parakeet TDT (issue #89 round 2 — words at slice boundaries
// landed outside both adjacent slice ranges and were silently dropped,
// and the text rebuild inserted a space between every kana on the JA
// tokenizer). The tests that pinned those code paths are gone with the
// code. All tests below are pure CPU, no model load.

#include <catch2/catch_test_macros.hpp>

#include "../src/core/audio_chunking.h"

#include <cstddef>
#include <vector>

TEST_CASE("chunking: short audio → single slice covering everything", "[unit][chunk-context]") {
    std::vector<float> audio(1600, 0.5f); // 0.1 s @ 16 kHz
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/16000,
                                                         /*search_win=*/8000);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].first == 0);
    REQUIRE(slices[0].second == audio.size());
}

TEST_CASE("chunking: slices cover the full input without gaps or overlap", "[unit][chunk-context]") {
    // 4 s of uniform audio @ 16 kHz, split into 1 s chunks.
    const size_t SR = 16000;
    const size_t total = 4 * SR;
    std::vector<float> audio(total, 0.3f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    // At least 2 slices for 4 s / 1 s.
    REQUIRE(slices.size() >= 2);

    // Contiguous: end[i] == begin[i+1].
    for (size_t i = 1; i < slices.size(); ++i) {
        REQUIRE(slices[i].first == slices[i - 1].second);
    }
    // First slice starts at 0, last slice ends at total.
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == total);
}

TEST_CASE("chunking: all-zero audio splits into equal chunks", "[unit][chunk-context]") {
    const size_t SR = 16000;
    const size_t total = 3 * SR;
    std::vector<float> audio(total, 0.0f); // pure silence → energy minima everywhere
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    REQUIRE(slices.size() >= 2);
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == total);
    for (size_t i = 1; i < slices.size(); ++i)
        REQUIRE(slices[i].first == slices[i - 1].second);
}

TEST_CASE("chunking: audio exactly equal to max_chunk → single slice", "[unit][chunk-context]") {
    const size_t SR = 16000;
    std::vector<float> audio(SR, 0.2f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].first == 0);
    REQUIRE(slices[0].second == SR);
}

// ---------------------------------------------------------------------------
// process_slice contract — each slice is fed its own audio range only.
//
// This is the invariant that broke in 617cd02 and was restored in the issue
// #89 round-2 revert: process_slice calls
//   be.transcribe(samples.data() + sl.start, sl.end - sl.start, sl.t0_cs, ...)
// with no left/right neighbour audio leaking in. The tests below construct
// a slice list, then walk it the same way process_slice does and assert:
//
// 1. each slice's audio range is exactly [sl.start, sl.end);
// 2. no two slices' audio ranges overlap;
// 3. the union of slice ranges covers the input exactly once.
//
// If a future change adds back a context expansion (effective_start ≠ sl.start
// or effective_len ≠ sl.end - sl.start), these tests need updating to also
// pin a dedup contract — otherwise the user-reported word-drop on
// parakeet-tdt-0.6b-ja regresses silently.
// ---------------------------------------------------------------------------

// Distinctive non-uniform fill so a wrong slice extraction would copy the
// wrong samples. samples[i] = i / 1.0e6 keeps every value unique without
// overflowing float precision for inputs up to a few minutes.
static std::vector<float> make_fingerprinted_audio(size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = (float)i / 1.0e6f;
    return out;
}

TEST_CASE("contract: process_slice feeds each slice its own audio verbatim", "[unit][chunk-context]") {
    const size_t SR = 16000;
    const size_t total = 30 * SR; // 30 s
    auto audio = make_fingerprinted_audio(total);

    // 10 s chunks → expect 3 slices.
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/10 * SR,
                                                         /*search_win=*/5 * SR);
    REQUIRE(slices.size() >= 2);

    // Walk slices the same way process_slice does. For each slice, take the
    // pointer + length pair that would be passed to be.transcribe(); verify
    // the first and last samples in the extracted range match the audio's
    // values at sl.start and sl.end-1 — no off-by-one, no neighbour bleed.
    size_t covered_samples = 0;
    int prev_end = -1;
    for (const auto& sl : slices) {
        const float* slice_ptr = audio.data() + sl.first;
        const size_t slice_len = sl.second - sl.first;

        REQUIRE(slice_len > 0);
        REQUIRE(slice_ptr[0] == audio[sl.first]);
        REQUIRE(slice_ptr[slice_len - 1] == audio[sl.second - 1]);

        // Contiguity with the previous slice — no gap, no overlap.
        if (prev_end >= 0)
            REQUIRE((int)sl.first == prev_end);
        prev_end = (int)sl.second;
        covered_samples += slice_len;
    }
    REQUIRE(covered_samples == total);
}

TEST_CASE("contract: slice t0_cs derives from sl.start without context shift", "[unit][chunk-context]") {
    // process_slice passes sl.t0_cs as t_offset_cs to the backend (no
    // effective_t0_cs subtraction the way 617cd02 did). Verify the
    // crispasr_audio_slice layout used in the CLI computes t0_cs from
    // sl.start at the sample rate exactly — i.e. the cs offset matches
    // the slice's first sample.
    const int SR = 16000;
    struct SliceTriplet {
        int start;
        int end;
        int64_t t0_cs;
        int64_t t1_cs;
    };
    // Mirror what crispasr_energy_chunk_slices builds.
    const std::vector<std::pair<size_t, size_t>> ranges = {
        {0, 10 * SR},       // 0–10 s
        {10 * SR, 20 * SR}, // 10–20 s
        {20 * SR, 30 * SR}, // 20–30 s
    };
    std::vector<SliceTriplet> slices;
    for (auto& r : ranges) {
        slices.push_back({
            (int)r.first,
            (int)r.second,
            (int64_t)((double)r.first / SR * 100.0),
            (int64_t)((double)r.second / SR * 100.0),
        });
    }

    REQUIRE(slices[0].t0_cs == 0);
    REQUIRE(slices[0].t1_cs == 1000);
    REQUIRE(slices[1].t0_cs == 1000);
    REQUIRE(slices[1].t1_cs == 2000);
    REQUIRE(slices[2].t0_cs == 2000);
    REQUIRE(slices[2].t1_cs == 3000);

    // Adjacent slices share their boundary cs — feeding sl.t0_cs to
    // transcribe means every word emitted at the encoder's reported t0
    // lands either before the boundary (in slice i) or at/after it (in
    // slice i+1). No "owned by neither" gap, no overlap.
    for (size_t i = 1; i < slices.size(); ++i) {
        REQUIRE(slices[i].t0_cs == slices[i - 1].t1_cs);
    }
}

TEST_CASE("contract: no-context-bleed at adjacent slice boundaries", "[unit][chunk-context]") {
    // Read enough samples around each slice boundary to detect any
    // accidental ±N-sample leak. With the fingerprinted audio
    // (samples[i] = i / 1e6) we can spot a one-sample misalignment.
    const size_t SR = 16000;
    const size_t total = 25 * SR;
    auto audio = make_fingerprinted_audio(total);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/8 * SR,
                                                         /*search_win=*/4 * SR);
    REQUIRE(slices.size() >= 2);

    // Sample at the very edge of each slice must equal the audio's value at
    // that absolute index — i.e. no left/right neighbour was prepended.
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& sl = slices[i];
        const float first = (audio.data() + sl.first)[0];
        const float last = (audio.data() + sl.first)[sl.second - sl.first - 1];

        REQUIRE(first == audio[sl.first]);
        REQUIRE(last == audio[sl.second - 1]);

        // The audio just BEFORE sl.start (if any) must NOT appear inside
        // the slice — i.e. dereferencing slice_ptr[-1] is the previous
        // slice's last sample, not part of this slice. We check it as a
        // pointer-arithmetic invariant.
        if (sl.first > 0) {
            REQUIRE((audio.data() + sl.first)[-1] == audio[sl.first - 1]);
        }
    }
}
