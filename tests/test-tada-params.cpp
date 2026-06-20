// test-tada-params.cpp — unit tests for tada_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "tada_tts.h"

TEST_CASE("tada_params: default values are sensible", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("tada_init_from_file: null path returns nullptr", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();
    struct tada_context* ctx = tada_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("tada_init_from_file: empty path returns nullptr", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();
    struct tada_context* ctx = tada_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("tada_free: NULL context is a no-op", "[unit][tada]") {
    tada_free(nullptr);
    SUCCEED("tada_free tolerated a NULL ctx.");
}
