// test-pockettts-params.cpp — unit tests for pocket_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "pocket_tts.h"

TEST_CASE("pocket_tts_params: default values are sensible", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("pocket_tts_init_from_file: null path returns nullptr", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();
    struct pocket_tts_context* ctx = pocket_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("pocket_tts_init_from_file: empty path returns nullptr", "[unit][pocket_tts]") {
    struct pocket_tts_context_params p = pocket_tts_context_default_params();
    struct pocket_tts_context* ctx = pocket_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("pocket_tts_free: NULL context is a no-op", "[unit][pocket_tts]") {
    pocket_tts_free(nullptr);
    SUCCEED("pocket_tts_free tolerated a NULL ctx.");
}
