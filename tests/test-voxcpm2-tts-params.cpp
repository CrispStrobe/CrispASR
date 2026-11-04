// test-voxcpm2-tts-params.cpp — unit tests for voxcpm2_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "voxcpm2_tts.h"

TEST_CASE("voxcpm2_context_params: default values are sensible", "[unit][voxcpm2-tts]") {
    struct voxcpm2_context_params p = voxcpm2_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("voxcpm2_init_from_file: null path returns nullptr", "[unit][voxcpm2-tts]") {
    struct voxcpm2_context_params p = voxcpm2_context_default_params();
    struct voxcpm2_context* ctx = voxcpm2_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxcpm2_init_from_file: empty path returns nullptr", "[unit][voxcpm2-tts]") {
    struct voxcpm2_context_params p = voxcpm2_context_default_params();
    struct voxcpm2_context* ctx = voxcpm2_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxcpm2_free: NULL context is a no-op", "[unit][voxcpm2-tts]") {
    voxcpm2_free(nullptr);
    SUCCEED("voxcpm2_free tolerated a NULL ctx.");
}

TEST_CASE("voxcpm2: backend unlimited dispatch classification (#164)", "[unit][voxcpm2-tts]") {
    // Null and CPU backends are not unlimited GPUs
    CHECK_FALSE(voxcpm2_test::test_is_gpu_unlimited_backend_name(nullptr));
    CHECK_FALSE(voxcpm2_test::test_is_gpu_unlimited_backend_name("CPU"));

    // Vulkan backends are subject to maxComputeWorkGroupCount limit (65535)
    CHECK_FALSE(voxcpm2_test::test_is_gpu_unlimited_backend_name("Vulkan"));
    CHECK_FALSE(voxcpm2_test::test_is_gpu_unlimited_backend_name("Vulkan0"));

    // CUDA and Metal backends have >= 2^31 dispatch dimensions (unlimited)
    CHECK(voxcpm2_test::test_is_gpu_unlimited_backend_name("CUDA"));
    CHECK(voxcpm2_test::test_is_gpu_unlimited_backend_name("CUDA0"));
    CHECK(voxcpm2_test::test_is_gpu_unlimited_backend_name("MTL"));
    CHECK(voxcpm2_test::test_is_gpu_unlimited_backend_name("MTL0"));
    CHECK(voxcpm2_test::test_is_gpu_unlimited_backend_name("Metal"));
}
