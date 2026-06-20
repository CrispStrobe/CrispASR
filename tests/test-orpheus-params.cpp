// test-orpheus-params.cpp — unit tests for orpheus_context_params defaults
// and null-guard coverage on public setters. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "orpheus.h"

TEST_CASE("orpheus_params: default values are sensible", "[unit][orpheus]") {
    struct orpheus_context_params p = orpheus_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature >= 0.0f);
    REQUIRE(p.max_audio_tokens >= 0);
    REQUIRE(p.flash_attn == true);
}

TEST_CASE("orpheus_init_from_file: null path returns nullptr", "[unit][orpheus]") {
    struct orpheus_context_params p = orpheus_context_default_params();
    struct orpheus_context* ctx = orpheus_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("orpheus_init_from_file: empty path returns nullptr", "[unit][orpheus]") {
    struct orpheus_context_params p = orpheus_context_default_params();
    struct orpheus_context* ctx = orpheus_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("orpheus setters: NULL context is a no-op", "[unit][orpheus]") {
    orpheus_set_n_threads(nullptr, 4);
    orpheus_set_temperature(nullptr, 0.8f);
    orpheus_set_seed(nullptr, 42);
    orpheus_set_speaker_by_name(nullptr, "tara");
    orpheus_set_codec_path(nullptr, "/dev/null");
    orpheus_free(nullptr);
    SUCCEED("All orpheus setters tolerated a NULL ctx.");
}
