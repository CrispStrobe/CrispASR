// test-vibevoice-params.cpp — unit tests for vibevoice_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "vibevoice.h"

TEST_CASE("vibevoice_params: default values are sensible", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("vibevoice_init_from_file: null path returns nullptr", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();
    struct vibevoice_context* ctx = vibevoice_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("vibevoice_init_from_file: empty path returns nullptr", "[unit][vibevoice]") {
    struct vibevoice_context_params p = vibevoice_context_default_params();
    struct vibevoice_context* ctx = vibevoice_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("vibevoice_free: NULL context is a no-op", "[unit][vibevoice]") {
    vibevoice_free(nullptr);
    SUCCEED("vibevoice_free tolerated a NULL ctx.");
}
