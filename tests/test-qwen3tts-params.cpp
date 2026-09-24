// test-qwen3tts-params.cpp — unit tests for qwen3_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "qwen3_tts.h"
#include "qwen3_tts_hip_policy.h"

TEST_CASE("qwen3_tts_params: default values are sensible", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). qwen3-tts ships
// greedy talker decode (temperature 0) with GPU + flash-attn on by default; pin
// them so a drift fails CI.
TEST_CASE("qwen3_tts_params: value knobs match the shipped defaults", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.0f)); // 0 = greedy
    REQUIRE(p.max_codec_steps == 0);               // 0 = model default
    REQUIRE(p.seed == 0);
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true);
}

TEST_CASE("qwen3_tts_init_from_file: null path returns nullptr", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    struct qwen3_tts_context* ctx = qwen3_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_tts_init_from_file: empty path returns nullptr", "[unit][qwen3_tts]") {
    struct qwen3_tts_context_params p = qwen3_tts_context_default_params();
    struct qwen3_tts_context* ctx = qwen3_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("qwen3_tts_free: NULL context is a no-op", "[unit][qwen3_tts]") {
    qwen3_tts_free(nullptr);
    SUCCEED("qwen3_tts_free tolerated a NULL ctx.");
}

TEST_CASE("qwen3_tts HIP policy defaults known-bad paths to CPU", "[unit][qwen3_tts][hip]") {
    using namespace qwen3_tts_hip_policy;

    REQUIRE(codec_must_use_cpu("ROCm0", false));
    REQUIRE_FALSE(codec_must_use_cpu("ROCm0", true));
    REQUIRE_FALSE(codec_must_use_cpu("CUDA0", false));
    REQUIRE_FALSE(codec_must_use_cpu("Metal", false));

    REQUIRE(code_predictor_must_use_cpu("ROCm0", false, 5, 1024, true));
    REQUIRE_FALSE(code_predictor_must_use_cpu("ROCm0", true, 5, 1024, true));
    REQUIRE_FALSE(code_predictor_must_use_cpu("ROCm0", false, 5, 1024, false));
    REQUIRE_FALSE(code_predictor_must_use_cpu("ROCm0", false, 28, 2048, true));
    REQUIRE_FALSE(code_predictor_must_use_cpu("CUDA0", false, 5, 1024, true));
}

TEST_CASE("qwen3_tts F32 down-projection promotion follows the narrowing backends", "[unit][qwen3_tts][hip]") {
    using namespace qwen3_tts_hip_policy;

    for (const char* be : {"CUDA0", "ROCm0", "Vulkan0", "SYCL0"}) {
        INFO(be);
        REQUIRE(f16_matmul_narrows_activations(be));
        REQUIRE(promote_cp_down_to_f32(be, true, -1));
        REQUIRE_FALSE(promote_cp_down_to_f32(be, false, -1)); // quantized / F32 weights: nothing to promote
        REQUIRE_FALSE(promote_cp_down_to_f32(be, true, 0));   // explicit opt-out
    }
    for (const char* be : {"CPU", "Metal", "BLAS"}) {
        INFO(be);
        REQUIRE_FALSE(f16_matmul_narrows_activations(be));
        REQUIRE_FALSE(promote_cp_down_to_f32(be, true, -1));
        REQUIRE(promote_cp_down_to_f32(be, true, 1)); // forced for A/B
    }
    REQUIRE_FALSE(f16_matmul_narrows_activations(nullptr));
}

TEST_CASE("qwen3_tts Vulkan talker runs natively unless asked (#337)", "[unit][qwen3_tts][hip]") {
    using namespace qwen3_tts_hip_policy;
    REQUIRE_FALSE(vulkan_talker_on_cpu(nullptr, nullptr)); // new default: native
    REQUIRE_FALSE(vulkan_talker_on_cpu("0", nullptr));
    REQUIRE_FALSE(vulkan_talker_on_cpu(nullptr, "1")); // legacy opt-in is now a no-op
    REQUIRE(vulkan_talker_on_cpu("1", nullptr));       // escape hatch
    REQUIRE(vulkan_talker_on_cpu(nullptr, "0"));       // legacy explicit opt-out
    REQUIRE(vulkan_talker_on_cpu("1", "1"));           // the CPU request wins
}
