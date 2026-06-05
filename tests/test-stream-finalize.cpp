// test-stream-finalize.cpp — unit tests for the streaming
// `final.text` stitcher in `examples/cli/crispasr_stream_finalize.h`.
//
// These cases lock the round-4 fix for issue #84
// (CKwasd, 2026-05-11): when `--stream-final-mode redecode` skips its
// extra backend pass (sub-2-s utterance) or it returns empty, the
// caller falls back to `stitch_partial_accumulator` rather than
// emitting an empty `final.text`. That fallback must always produce a
// non-empty string when at least one of the inputs is non-empty, and
// must reproduce the same six cases the previous in-place implementation
// handled (see commit history of crispasr_run.cpp::finalize_utterance).

#include <catch2/catch_test_macros.hpp>

#include "../examples/cli/crispasr_stream_finalize.h"

using crispasr::stitch_partial_accumulator;

TEST_CASE("stream-finalize: last_partial extends committed_prefix → last_partial wins", "[unit][stream-json]") {
    // The normal LCP-accumulator state on a clean rolling-window stream:
    // each new partial begins with everything we've already committed,
    // plus more. We return the longer one alone, no concatenation.
    REQUIRE(stitch_partial_accumulator("Hello", "Hello world") == "Hello world");
    REQUIRE(stitch_partial_accumulator("そういう", "そういうときはね") == "そういうときはね");
}

TEST_CASE("stream-finalize: empty committed_prefix → returns last_partial verbatim", "[unit][stream-json]") {
    // First-partial case: no prefix committed yet, so the last partial
    // is the whole hypothesis.
    REQUIRE(stitch_partial_accumulator("", "first partial") == "first partial");
    REQUIRE(stitch_partial_accumulator("", "") == "");
}

TEST_CASE("stream-finalize: empty last_partial → returns committed_prefix", "[unit][stream-json]") {
    // After a window-roll commit the previous last_partial has been
    // promoted to committed_prefix and last_partial may be empty until
    // the next decode arrives. The committed text must still surface.
    REQUIRE(stitch_partial_accumulator("Committed text", "") == "Committed text");
}

TEST_CASE("stream-finalize: divergent prefix+partial → concatenated with a space", "[unit][stream-json]") {
    // The rolling window evicted the prefix and the new partial doesn't
    // start with what we committed (model output drifted on the
    // overlap). Concatenate with a space to keep both visible.
    REQUIRE(stitch_partial_accumulator("First half.", "second half.") == "First half. second half.");
}

TEST_CASE("stream-finalize: partial that is a strict prefix of committed (window collapse)", "[unit][stream-json]") {
    // After a tail-only partial (window rolled past the start), the
    // new partial may be shorter than what we already committed.
    // The "last_partial starts with committed_prefix" test fails
    // (committed is longer); we fall through to the concatenation
    // case, keeping the committed text plus the short tail. This
    // documents the existing behaviour — a regression here would mean
    // we silently dropped already-emitted text.
    REQUIRE(stitch_partial_accumulator("Hello world", "Hello") == "Hello world Hello");
}

TEST_CASE("stream-finalize: round-4 sub-2-s utterance produces non-empty final", "[unit][stream-json][regression-84]") {
    // Reproduces the CKwasd round-3 retest case directly: a short
    // utterance (e.g. 1.18 s) where the redecode pass would be skipped
    // because `utterance_pcm.size() < kStreamRedecodeMinSamples`.
    // Before the fix, `final_text` stayed empty and the UI saw the
    // partial text vanish. After the fix the caller falls back to
    // stitch_partial_accumulator, which must return the same content
    // the wrapper has already seen in partials.
    //
    // The streaming loop's on_partial_text always updates
    // last_partial_text first, then maybe promotes prior content into
    // prefix_committed on rolling-window eviction. For a sub-2-s
    // utterance the eviction never fires (the window doesn't roll
    // before finalize), so the stitcher is called with
    // (prefix_committed="", last_partial_text="<observed partial>").
    const std::string observed_partial = "そういうときはね";
    REQUIRE_FALSE(stitch_partial_accumulator("", observed_partial).empty());
    REQUIRE(stitch_partial_accumulator("", observed_partial) == observed_partial);
}

TEST_CASE("stream-finalize: kStreamRedecodeMinSamples is 2 s at 16 kHz", "[unit][stream-json]") {
    // Threshold is structural — derived from moonshine/parakeet's
    // first conv kernel input requirement, not a tunable. Lock it so
    // a stray edit doesn't silently widen the empty-final window.
    REQUIRE(crispasr::kStreamRedecodeMinSamples == 32000);
}

TEST_CASE("stream-finalize: empty inputs produce empty output (utterance with no partials)", "[unit][stream-json]") {
    // An utterance can open and immediately finalize before any
    // partial fired (e.g. VAD slice text was empty under min_speech_sec).
    // The fallback must still terminate cleanly — empty final is OK in
    // this case because there's no UI text to blank.
    REQUIRE(stitch_partial_accumulator("", "").empty());
}
