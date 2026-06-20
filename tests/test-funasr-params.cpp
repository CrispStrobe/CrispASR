// test-funasr-params.cpp — unit tests for funasr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "funasr.h"

TEST_CASE("funasr_params: default values are sensible", "[unit][funasr]") {
    struct funasr_context_params p = funasr_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("funasr_init_from_file: null path returns nullptr", "[unit][funasr]") {
    struct funasr_context_params p = funasr_context_default_params();
    struct funasr_context* ctx = funasr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("funasr_init_from_file: empty path returns nullptr", "[unit][funasr]") {
    struct funasr_context_params p = funasr_context_default_params();
    struct funasr_context* ctx = funasr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("funasr_free: NULL context is a no-op", "[unit][funasr]") {
    funasr_free(nullptr);
    SUCCEED("funasr_free tolerated a NULL ctx.");
}
