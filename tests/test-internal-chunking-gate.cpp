// test-internal-chunking-gate.cpp — unit tests for the CAP_INTERNAL_CHUNKING
// gate in the long-audio fallback (issue #89 follow-up).
//
// Backends with CAP_INTERNAL_CHUNKING (parakeet, canary, fastconformer-ctc)
// handle their own long-audio chunking internally — the crispasr_run.cpp
// auto-chunk fallback must NOT fire for them. These tests pin that invariant.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_long_audio_fallback.h"

using crispasr_long_audio::CAP_INTERNAL_CHUNKING_FLAG;
using crispasr_long_audio::CAP_UNBOUNDED_INPUT_FLAG;
using crispasr_long_audio::should_auto_chunk_long;

namespace {
constexpr int kSR = 16000;
constexpr int kThreshold = 30;
constexpr uint32_t kUnbounded = CAP_UNBOUNDED_INPUT_FLAG;
constexpr uint32_t kInternal = CAP_UNBOUNDED_INPUT_FLAG | CAP_INTERNAL_CHUNKING_FLAG;
} // namespace

TEST_CASE("CAP_INTERNAL_CHUNKING prevents auto-chunk fallback", "[unit][internal-chunking][issue-89]") {
    // 300 s audio with no VAD and no explicit --chunk-seconds:
    // - plain CAP_UNBOUNDED_INPUT should trigger the 30 s fallback
    // - CAP_INTERNAL_CHUNKING should prevent it
    constexpr int n_samples = 300 * kSR;
    REQUIRE(should_auto_chunk_long(0, false, kUnbounded, n_samples, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kInternal, n_samples, kSR, kThreshold));
}

TEST_CASE("CAP_INTERNAL_CHUNKING: 60 s audio still skips fallback", "[unit][internal-chunking][issue-89]") {
    constexpr int n_samples = 60 * kSR;
    REQUIRE(should_auto_chunk_long(0, false, kUnbounded, n_samples, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kInternal, n_samples, kSR, kThreshold));
}

TEST_CASE("CAP_INTERNAL_CHUNKING: short audio stays on full-audio path regardless", "[unit][internal-chunking]") {
    constexpr int n_samples = 20 * kSR;
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kUnbounded, n_samples, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, false, kInternal, n_samples, kSR, kThreshold));
}

TEST_CASE("CAP_INTERNAL_CHUNKING: explicit --chunk-seconds bypasses regardless", "[unit][internal-chunking]") {
    constexpr int n_samples = 300 * kSR;
    REQUIRE_FALSE(should_auto_chunk_long(60, false, kInternal, n_samples, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(60, false, kUnbounded, n_samples, kSR, kThreshold));
}

TEST_CASE("CAP_INTERNAL_CHUNKING: VAD bypasses regardless", "[unit][internal-chunking]") {
    constexpr int n_samples = 300 * kSR;
    REQUIRE_FALSE(should_auto_chunk_long(0, true, kInternal, n_samples, kSR, kThreshold));
    REQUIRE_FALSE(should_auto_chunk_long(0, true, kUnbounded, n_samples, kSR, kThreshold));
}

TEST_CASE("CAP_INTERNAL_CHUNKING flag value matches crispasr_backend.h", "[unit][internal-chunking]") {
    // The flag is duplicated in the header for dependency-light unit testing.
    // This test pins the value so a refactor can't silently change one copy.
    REQUIRE(CAP_INTERNAL_CHUNKING_FLAG == (1u << 20));
}
