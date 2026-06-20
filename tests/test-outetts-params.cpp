// test-outetts-params.cpp — unit tests for outetts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "outetts.h"

TEST_CASE("outetts_params: default values are sensible", "[unit][outetts]") {
    struct outetts_context_params p = outetts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("outetts_init_from_file: null path returns nullptr", "[unit][outetts]") {
    struct outetts_context_params p = outetts_context_default_params();
    struct outetts_context* ctx = outetts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("outetts_init_from_file: empty path returns nullptr", "[unit][outetts]") {
    struct outetts_context_params p = outetts_context_default_params();
    struct outetts_context* ctx = outetts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("outetts_free: NULL context is a no-op", "[unit][outetts]") {
    outetts_free(nullptr);
    SUCCEED("outetts_free tolerated a NULL ctx.");
}
