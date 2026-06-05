// test-issue-89-long-audio-fallback.cpp — unit tests for the long-audio
// chunking fallback gate (issue #89).
//
// `a069018f fix(#89): always full-encode for CAP_UNBOUNDED_INPUT backends`
// set effective_chunk_seconds = 0 unconditionally for CAP_UNBOUNDED_INPUT
// backends without explicit `--chunk-seconds`. That was right for short
// audio (full-audio encoding gives the best quality) but wrong for long
// audio: on the reporter's 300 s YouTube clip parakeet-tdt-0.6b-ja
// collapsed to 35 tokens covering only the first 4.8 s. The FastConformer
// encoder + TDT decoder are not stable in a single pass past ~30-60 s.
//
// The fix: when no VAD and no explicit `--chunk-seconds` and the input
// audio is longer than the safe single-pass window, fall back to fixed
// chunking. The gate lives in `crispasr_long_audio_fallback.h`; these
// tests pin it so a refactor can't silently re-introduce the regression.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_long_audio_fallback.h"

using crispasr_long_audio::CAP_UNBOUNDED_INPUT_FLAG;
using crispasr_long_audio::should_auto_chunk_long;

namespace {
// Plausible bitmask for a non-CAP_UNBOUNDED_INPUT backend (whisper-ish).
constexpr uint32_t kBoundedBackendCaps = 0;
// CAP_UNBOUNDED_INPUT alone — what parakeet / canary / etc. declare.
constexpr uint32_t kUnboundedBackendCaps = CAP_UNBOUNDED_INPUT_FLAG;
constexpr int kSR = 16000;
constexpr int kThreshold = 30; // matches the 30 s fallback in crispasr_run.cpp
} // namespace

TEST_CASE("issue #89: 300s parakeet audio with no VAD/--chunk-seconds → fall back to chunked",
          "[unit][long-audio][issue-89]") {
    // The exact scenario the reporter hit:
    //   - parakeet (CAP_UNBOUNDED_INPUT)
    //   - no --vad, no --vad-model
    //   - no --chunk-seconds → effective_chunk_seconds = 0 after the
    //     CAP_UNBOUNDED_INPUT-default in crispasr_run.cpp
    //   - 300 s of audio at 16 kHz
    constexpr int n_samples = 300 * kSR;
    REQUIRE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, n_samples, kSR, kThreshold));
}

TEST_CASE("60s audio also triggers auto-chunking at 30s threshold", "[unit][long-audio][issue-89]") {
    // The reporter's 60 s test showed only 36 words (0-20 s) from a 60 s
    // file — the TDT decoder loses content past ~20 s even at 60 s. With
    // the new 30 s threshold, 60 s audio gets chunked into ~2 chunks.
    constexpr int n_samples = 60 * kSR;
    REQUIRE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, n_samples, kSR, kThreshold));
}

TEST_CASE("short audio stays on full-audio path", "[unit][long-audio][issue-89]") {
    // Below threshold → no fallback → the original full-audio encoding
    // gives the best quality (no chunk boundaries to stitch).
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, 20 * kSR, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, kThreshold * kSR, kSR, kThreshold));
}

TEST_CASE("explicit --vad/--vad-model bypasses the fallback", "[unit][long-audio][issue-89]") {
    // VAD-derived slicing is silence-bounded and already produces safe
    // input lengths for the encoder. Don't override the user's choice.
    REQUIRE_FALSE(should_auto_chunk_long(0, /*wants_vad=*/true, kUnboundedBackendCaps, 300 * kSR, kSR, kThreshold));
}

TEST_CASE("explicit --chunk-seconds bypasses the fallback", "[unit][long-audio][issue-89]") {
    // If the user already set a non-zero chunk size, respect it. The
    // overlap-save context in crispasr_run.cpp will cover the boundaries.
    REQUIRE_FALSE(should_auto_chunk_long(30, false, kUnboundedBackendCaps, 300 * kSR, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(120, false, kUnboundedBackendCaps, 300 * kSR, kSR, kThreshold));
}

TEST_CASE("non-CAP_UNBOUNDED_INPUT backends never trigger", "[unit][long-audio][issue-89]") {
    // Whisper-style backends have their own internal chunking (30 s
    // windows + seek); the auto-fallback would either be redundant
    // or actively harmful. Only apply to backends that have explicitly
    // declared CAP_UNBOUNDED_INPUT.
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kBoundedBackendCaps, 300 * kSR, kSR, kThreshold));
}

TEST_CASE("defensive zero/negative inputs return false", "[unit][long-audio][issue-89]") {
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, 0, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, 300 * kSR, 0, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kUnboundedBackendCaps, 300 * kSR, kSR, 0));
}
