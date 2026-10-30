// test-meanvc2-params.cpp — unit tests for meanvc2_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "meanvc2.h"

TEST_CASE("meanvc2_params: default values are sensible", "[unit][meanvc2]") {
    struct meanvc2_context_params p = meanvc2_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE((p.chunk_size == 12 || p.chunk_size == 4));
    REQUIRE(p.block_size == 4);
}

TEST_CASE("meanvc2_init_from_file: null path returns nullptr", "[unit][meanvc2]") {
    struct meanvc2_context_params p = meanvc2_context_default_params();
    struct meanvc2_context* ctx = meanvc2_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("meanvc2_init_from_file: empty path returns nullptr", "[unit][meanvc2]") {
    struct meanvc2_context_params p = meanvc2_context_default_params();
    struct meanvc2_context* ctx = meanvc2_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("meanvc2_null_guards: null ctx calls do not crash", "[unit][meanvc2]") {
    meanvc2_free(nullptr);
    meanvc2_free_audio(nullptr);

    float dummy[16] = {0};
    int out_samples = 0;
    REQUIRE(meanvc2_convert(nullptr, dummy, 16, &out_samples) == nullptr);
    REQUIRE(meanvc2_set_speaker_audio(nullptr, dummy, 16) == -1);
    REQUIRE(meanvc2_set_speaker_embedding(nullptr, dummy) == -1);
}
