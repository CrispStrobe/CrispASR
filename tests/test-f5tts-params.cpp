// test-f5tts-params.cpp — unit tests for f5_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "f5_tts.h"

TEST_CASE("f5tts_params: default values are sensible", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.speed > 0.0f);
}

TEST_CASE("f5tts_init_from_file: null path returns nullptr", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();
    struct f5_tts_context* ctx = f5_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("f5tts_init_from_file: empty path returns nullptr", "[unit][f5tts]") {
    struct f5_tts_params p = f5_tts_default_params();
    struct f5_tts_context* ctx = f5_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("f5tts_free: NULL context is a no-op", "[unit][f5tts]") {
    f5_tts_free(nullptr);
    SUCCEED("f5_tts_free tolerated a NULL ctx.");
}
