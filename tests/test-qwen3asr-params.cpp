// test-qwen3asr-params.cpp — unit tests for qwen3_asr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "qwen3_asr.h"

TEST_CASE("qwen3_asr_params: default values are sensible", "[unit][qwen3_asr]") {
    struct qwen3_asr_context_params p = qwen3_asr_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("qwen3_asr_init_from_file: null path returns nullptr", "[unit][qwen3_asr]") {
    struct qwen3_asr_context_params p = qwen3_asr_context_default_params();
    struct qwen3_asr_context* ctx = qwen3_asr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_asr_init_from_file: empty path returns nullptr", "[unit][qwen3_asr]") {
    struct qwen3_asr_context_params p = qwen3_asr_context_default_params();
    struct qwen3_asr_context* ctx = qwen3_asr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_asr_free: NULL context is a no-op", "[unit][qwen3_asr]") {
    qwen3_asr_free(nullptr);
    SUCCEED("qwen3_asr_free tolerated a NULL ctx.");
}
