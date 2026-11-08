// test-qwen3tts-params.cpp — unit tests for qwen3_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "qwen3_tts.h"
#include "qwen3_tts_hip_policy.h"
#include "qwen3_tts_voice_pack.h"

TEST_CASE("Qwen3 default pack names follow the speaker dimension", "[unit][qwen3_tts]") {
    using qwen3_tts_voice_pack::default_filename;
    REQUIRE(std::string(default_filename(1024)) == "qwen3-tts-voice-0.6b-designed.gguf");
    REQUIRE(std::string(default_filename(2048)) == "qwen3-tts-voice-1.7b-designed.gguf");
    REQUIRE(default_filename(0) == nullptr);
    REQUIRE(default_filename(4096) == nullptr);
}

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

TEST_CASE("qwen3_tts codec can stay on Vulkan with the talker on CPU (#585)", "[unit][qwen3_tts][hip]") {
    using namespace qwen3_tts_hip_policy;
    REQUIRE_FALSE(vulkan_codec_stays_on_gpu(nullptr, nullptr, nullptr)); // VULKAN_CPU=1 alone: all CPU
    REQUIRE(vulkan_codec_stays_on_gpu("1", nullptr, nullptr));           // CODEC_GPU=1 is honoured now
    REQUIRE(vulkan_codec_stays_on_gpu(nullptr, "1", nullptr));           // CODEC_VULKAN=1 alias
    REQUIRE_FALSE(vulkan_codec_stays_on_gpu("0", "0", nullptr));
    REQUIRE_FALSE(vulkan_codec_stays_on_gpu("1", "1", "1")); // CODEC_CPU=1 wins
}

TEST_CASE("Qwen3 Base packs require model-sized F32 speakers and I32 codec frames (#555)", "[unit][qwen3_tts]") {
    ggml_init_params params = {8 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(params);
    REQUIRE(ctx);
    auto* speaker = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    auto* codes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 101);
    using qwen3_tts_voice_pack::valid_tensors;
    REQUIRE(valid_tensors(speaker, codes, 1024, 16));
    // The shipped 0.6B default pack triggered an 8192-byte read of 4096 bytes.
    REQUIRE_FALSE(valid_tensors(speaker, codes, 2048, 16));
    auto* large = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    REQUIRE(valid_tensors(large, codes, 2048, 16));
    REQUIRE_FALSE(valid_tensors(large, codes, 1024, 16));
    auto* half = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1024);
    REQUIRE_FALSE(valid_tensors(half, codes, 1024, 16));
    REQUIRE_FALSE(valid_tensors(nullptr, codes, 1024, 16));
    REQUIRE_FALSE(valid_tensors(speaker, nullptr, 1024, 16));
    REQUIRE_FALSE(valid_tensors(speaker, codes, 1024, 8));
    codes->type = GGML_TYPE_F32;
    REQUIRE_FALSE(valid_tensors(speaker, codes, 1024, 16));
    codes->type = GGML_TYPE_I32;
    codes->ne[2] = 2;
    REQUIRE_FALSE(valid_tensors(speaker, codes, 1024, 16));
    codes->ne[2] = 1;
    codes->ne[1] = 0;
    REQUIRE_FALSE(valid_tensors(speaker, codes, 1024, 16));
    ggml_free(ctx);
}

TEST_CASE("Qwen3 Base pack names must index complete unique voice records (#555)", "[unit][qwen3_tts]") {
    using qwen3_tts_voice_pack::valid_names;
    REQUIRE(valid_names({"first", "second"}, {"hello", "world"}));
    REQUIRE_FALSE(valid_names({}, {}));
    REQUIRE_FALSE(valid_names({"first"}, {}));
    REQUIRE_FALSE(valid_names({"first"}, {"hello", "world"}));
    REQUIRE_FALSE(valid_names({"first", "first"}, {"hello", "world"}));
    REQUIRE_FALSE(valid_names({""}, {"hello"}));
    REQUIRE_FALSE(valid_names({"first"}, {""}));
}
