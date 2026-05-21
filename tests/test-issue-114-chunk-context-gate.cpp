// test-issue-114-chunk-context-gate.cpp - unit tests for the overlap-save
// gating logic (issue #114).
//
// The cad4c28a "feat(#89): overlap-save chunking with --chunk-overlap flag"
// change extended each transcribe() slice by +/-chunk_overlap_seconds on
// each side whenever slices.size() > 1. That gate was correct for
// explicit `--chunk-seconds N` runs (where the chunks are arbitrary cuts
// through continuous speech and the encoder genuinely needs left/right
// context to span boundaries), but wrong for VAD-derived multi-slice runs:
// VAD slices are separated by silence, so there is no boundary signal to
// recover. Adding 3 s of neighbour audio pulled the next utterance into the
// current encoder context window, shifted features, and caused short slices
// to be dropped.
//
// The fix gates use_chunk_context on both effective_chunk_seconds > 0 and
// whether the slices came from VAD, so the extension is applied only when
// chunking is the actual reason for having multiple slices.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_chunk_context_gate.h"

using crispasr_chunk_context::should_use_chunk_context;

TEST_CASE("issue #114: VAD-derived multi-slice run does NOT extend with context", "[unit][chunk-context][issue-114]") {
    constexpr int effective_chunk_seconds = 0;
    constexpr std::size_t n_slices = 56;
    constexpr float chunk_overlap_seconds = 3.0f;
    REQUIRE_FALSE(should_use_chunk_context(effective_chunk_seconds, n_slices, chunk_overlap_seconds, true));
}

TEST_CASE("issue #114: VAD with default fallback chunk_seconds still gets no context",
          "[unit][chunk-context][issue-114]") {
    // Non-CAP_UNBOUNDED backends such as Cohere keep the CLI default
    // effective_chunk_seconds=30 even when VAD produced the slices. That
    // value is only a later overlong-slice split limit; it must not make
    // VAD slices look like fixed chunks.
    REQUIRE_FALSE(should_use_chunk_context(30, 33, 3.0f, true));
}

TEST_CASE("explicit --chunk-seconds with multiple non-VAD slices uses overlap-save",
          "[unit][chunk-context][issue-114]") {
    constexpr int effective_chunk_seconds = 30;
    constexpr std::size_t n_slices = 10;
    constexpr float chunk_overlap_seconds = 3.0f;
    REQUIRE(should_use_chunk_context(effective_chunk_seconds, n_slices, chunk_overlap_seconds, false));
}

TEST_CASE("single slice never gets context", "[unit][chunk-context][issue-114]") {
    REQUIRE_FALSE(should_use_chunk_context(30, 1, 3.0f, false));
    REQUIRE_FALSE(should_use_chunk_context(0, 1, 3.0f, false));
}

TEST_CASE("--chunk-overlap 0 disables overlap-save", "[unit][chunk-context][issue-114]") {
    REQUIRE_FALSE(should_use_chunk_context(30, 10, 0.0f, false));
    REQUIRE_FALSE(should_use_chunk_context(30, 10, -1.0f, false));
}

TEST_CASE("gate is purely a function of its four inputs", "[unit][chunk-context][issue-114]") {
    for (int chunk_s : {0, 1, 5, 30, 120}) {
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{56}}) {
            for (float overlap : {-1.0f, 0.0f, 0.1f, 3.0f, 10.0f}) {
                const bool a = should_use_chunk_context(chunk_s, n, overlap, false);
                const bool b = should_use_chunk_context(chunk_s, n, overlap, false);
                const bool vad = should_use_chunk_context(chunk_s, n, overlap, true);
                REQUIRE(a == b);
                REQUIRE_FALSE(vad);

                const bool expected = (chunk_s > 0) && (n > 1) && (overlap > 0.0f);
                REQUIRE(a == expected);
            }
        }
    }
}
