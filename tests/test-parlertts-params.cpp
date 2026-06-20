// test-parlertts-params.cpp — unit tests for parler_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "parler_tts.h"

TEST_CASE("parler_tts_params: default values are sensible", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("parler_tts_init_from_file: null path returns nullptr", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();
    struct parler_tts_context* ctx = parler_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("parler_tts_init_from_file: empty path returns nullptr", "[unit][parler_tts]") {
    struct parler_tts_context_params p = parler_tts_context_default_params();
    struct parler_tts_context* ctx = parler_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("parler_tts_free: NULL context is a no-op", "[unit][parler_tts]") {
    parler_tts_free(nullptr);
    SUCCEED("parler_tts_free tolerated a NULL ctx.");
}
