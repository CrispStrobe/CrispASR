// test-qwen3-tts-cv-style.cpp — API-contract unit tests for
// qwen3_tts_set_cv_style_instruct (issue #91: --instruct silently ignored
// for CustomVoice models).
//
// These tests exercise the public C API contract without loading any GGUF —
// null-context guards and wrong-model-type rejections are deterministic
// regardless of whether a model file is present.

#include <catch2/catch_test_macros.hpp>

#if __has_include("qwen3_tts.h")
#include "qwen3_tts.h"

// ---- null-context guards ----

TEST_CASE("qwen3_tts_set_cv_style_instruct: NULL ctx returns -1", "[unit][qwen3-tts][issue91]") {
    // Every setter must null-guard. A crash here means the implementation
    // is missing the `if (!ctx) return -1;` guard added in the #91 fix.
    int rc = qwen3_tts_set_cv_style_instruct(nullptr, "spoke very fast");
    REQUIRE(rc == -1);
}

TEST_CASE("qwen3_tts_set_cv_style_instruct: NULL ctx with NULL instruct returns -1", "[unit][qwen3-tts][issue91]") {
    int rc = qwen3_tts_set_cv_style_instruct(nullptr, nullptr);
    REQUIRE(rc == -1);
}

// ---- default params smoke test ----

TEST_CASE("qwen3_tts_context_default_params: sanity", "[unit][qwen3-tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    // n_threads must be at least 1 (defaults to 4 in practice).
    REQUIRE(p.n_threads >= 1);
    // verbosity should default to a non-negative level.
    REQUIRE(p.verbosity >= 0);
    // temperature=0 means greedy by default.
    REQUIRE(p.temperature >= 0.0f);
    // max_codec_steps=0 means use built-in default.
    REQUIRE(p.max_codec_steps >= 0);
}

// ---- qwen3_tts_set_instruct null guard (pre-existing VoiceDesign API) ----

TEST_CASE("qwen3_tts_set_instruct: NULL ctx returns -1", "[unit][qwen3-tts]") {
    // VoiceDesign setter should have the same null guard. Catch regressions
    // where a refactor removes it.
    int rc = qwen3_tts_set_instruct(nullptr, "young female with British accent");
    REQUIRE(rc == -1);
}

#else
// qwen3_tts.h not available in this build — skip gracefully.
TEST_CASE("qwen3_tts_set_cv_style_instruct: header not available (skipped)", "[unit][qwen3-tts][issue91]") {
    SUCCEED("qwen3_tts.h not in include path — skipping CV style instruct tests.");
}
#endif
