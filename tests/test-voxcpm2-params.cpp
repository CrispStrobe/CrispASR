// test-voxcpm2-params.cpp — unit tests for voxcpm2_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "voxcpm2_tts.h"

TEST_CASE("voxcpm2_params: default values are sensible", "[unit][voxcpm2]") {
    struct voxcpm2_context_params p = voxcpm2_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("voxcpm2_init_from_file: null path returns nullptr", "[unit][voxcpm2]") {
    struct voxcpm2_context_params p = voxcpm2_context_default_params();
    struct voxcpm2_context* ctx = voxcpm2_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxcpm2_init_from_file: empty path returns nullptr", "[unit][voxcpm2]") {
    struct voxcpm2_context_params p = voxcpm2_context_default_params();
    struct voxcpm2_context* ctx = voxcpm2_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxcpm2_free: NULL context is a no-op", "[unit][voxcpm2]") {
    voxcpm2_free(nullptr);
    SUCCEED("voxcpm2_free tolerated a NULL ctx.");
}
