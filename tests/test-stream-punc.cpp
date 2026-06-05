// Unit tests for the streaming-punctuation placement policy added in
// PR #112 (`--stream-punc off|final|partial`).
//
// The helpers in `examples/cli/crispasr_stream_punc.h` are pure
// string-to-bool predicates: no model load, no audio, no streaming
// runtime. Catch2 covers all 4×2 = 8 combinations of (mode × predicate)
// plus the validator, so any future change to the policy matrix
// fails loudly here even before someone wires up an integration test.

#include "../examples/cli/crispasr_stream_punc.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("stream-punc: partials_enabled is true ONLY for 'partial'", "[unit][stream-punc]") {
    REQUIRE_FALSE(crispasr_stream_punc_partials_enabled("off"));
    REQUIRE_FALSE(crispasr_stream_punc_partials_enabled("final"));
    REQUIRE(crispasr_stream_punc_partials_enabled("partial"));
}

TEST_CASE("stream-punc: finals_enabled is true for 'final' and 'partial', false for 'off'", "[unit][stream-punc]") {
    REQUIRE_FALSE(crispasr_stream_punc_finals_enabled("off"));
    REQUIRE(crispasr_stream_punc_finals_enabled("final"));
    REQUIRE(crispasr_stream_punc_finals_enabled("partial"));
}

TEST_CASE("stream-punc: mode_valid accepts the three documented values", "[unit][stream-punc]") {
    REQUIRE(crispasr_stream_punc_mode_valid("off"));
    REQUIRE(crispasr_stream_punc_mode_valid("final"));
    REQUIRE(crispasr_stream_punc_mode_valid("partial"));
}

TEST_CASE("stream-punc: mode_valid rejects unknown / case-mismatched / empty values", "[unit][stream-punc]") {
    REQUIRE_FALSE(crispasr_stream_punc_mode_valid(""));
    REQUIRE_FALSE(crispasr_stream_punc_mode_valid("FINAL")); // case-sensitive
    REQUIRE_FALSE(crispasr_stream_punc_mode_valid("Partial"));
    REQUIRE_FALSE(crispasr_stream_punc_mode_valid("on"));
    REQUIRE_FALSE(crispasr_stream_punc_mode_valid("yes"));
    REQUIRE_FALSE(crispasr_stream_punc_mode_valid("partials"));
}

// Regression pin: the documented default is `final`, which means
// finals get FireRedPunc but partials don't. If anyone flips the
// default (or the predicates), this case catches it.
TEST_CASE("stream-punc: default 'final' implies finals=on, partials=off", "[unit][stream-punc]") {
    const std::string default_mode = "final";
    REQUIRE(crispasr_stream_punc_mode_valid(default_mode));
    REQUIRE(crispasr_stream_punc_finals_enabled(default_mode));
    REQUIRE_FALSE(crispasr_stream_punc_partials_enabled(default_mode));
}

// Regression pin: 'partial' is the old pre-#112 behavior — punc on
// both partials and finals.
TEST_CASE("stream-punc: 'partial' implies both finals and partials punc'd", "[unit][stream-punc]") {
    REQUIRE(crispasr_stream_punc_finals_enabled("partial"));
    REQUIRE(crispasr_stream_punc_partials_enabled("partial"));
}

// Regression pin: 'off' means BOTH partials and finals skip the
// punc model. Some users might assume "off" means "off for partials,
// normal for finals" — the policy is intentionally more aggressive.
TEST_CASE("stream-punc: 'off' suppresses punc on BOTH partials and finals", "[unit][stream-punc]") {
    REQUIRE_FALSE(crispasr_stream_punc_finals_enabled("off"));
    REQUIRE_FALSE(crispasr_stream_punc_partials_enabled("off"));
}
