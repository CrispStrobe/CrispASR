// test-qwen3tts-params.cpp — unit tests for qwen3_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "qwen3_tts.h"

TEST_CASE("qwen3_tts_params: default values are sensible", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("qwen3_tts_init_from_file: null path returns nullptr", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    struct qwen3_tts_context* ctx = qwen3_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_tts_init_from_file: empty path returns nullptr", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    struct qwen3_tts_context* ctx = qwen3_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_tts_free: NULL context is a no-op", "[unit][qwen3_tts]") {
    qwen3_tts_free(nullptr);
    SUCCEED("qwen3_tts_free tolerated a NULL ctx.");
}
