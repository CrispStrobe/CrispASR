// test-granitespeech-params.cpp — unit tests for granite_speech_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "granite_speech.h"
#include "core/granite_llm.h"
#include "core/granite_prompt.h"
#include <limits>

TEST_CASE("granite_speech_params: default values are sensible", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197 + PLAN #89).
// flash_attn must stay on by default; a silent removal (the #89 regression class)
// fails CI here.
TEST_CASE("granite_speech_params: gpu/flash defaults are pinned", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true);
}

TEST_CASE("granite_speech_init_from_file: null path returns nullptr", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();
    struct granite_speech_context* ctx = granite_speech_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("granite_speech_init_from_file: empty path returns nullptr", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();
    struct granite_speech_context* ctx = granite_speech_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("granite_speech_free: NULL context is a no-op", "[unit][granite_speech]") {
    granite_speech_free(nullptr);
    SUCCEED("granite_speech_free tolerated a NULL ctx.");
}

TEST_CASE("granite_speech_rope: absent metadata preserves unscaled defaults", "[unit][granite_speech]") {
    core_granite_llm::Hparams hp = {};
    hp.n_heads = 16;
    hp.n_kv_heads = 4;
    hp.head_dim = 128;
    hp.rope_theta = 10000.0f;
    const auto p = core_granite_llm::attention_params(hp);
    REQUIRE(p.rope_freq_scale == 1.0f);
    REQUIRE(p.rope_ext_factor == 0.0f);
    REQUIRE(p.rope_attn_factor == 1.0f);
    REQUIRE(p.n_ctx_orig == 0);
    REQUIRE(p.rope_beta_fast == 0.0f);
    REQUIRE(p.rope_beta_slow == 0.0f);
}

TEST_CASE("granite_speech_rope: YaRN applies magnitude exactly once", "[unit][granite_speech]") {
    core_granite_llm::Hparams hp = {};
    hp.n_heads = 16;
    hp.n_kv_heads = 4;
    hp.head_dim = 128;
    hp.rope_theta = 10000.0f;
    REQUIRE(core_granite_llm::configure_yarn(hp.rope, 4.0f, 4096, 1.138629436111989f, 32.0f, 1.0f));
    const auto p = core_granite_llm::attention_params(hp);
    REQUIRE(p.n_ctx_orig == 4096);
    REQUIRE(p.rope_freq_scale == 0.25f);
    REQUIRE(p.rope_ext_factor == 1.0f);
    REQUIRE(p.rope_attn_factor == 1.0f);
    REQUIRE(p.rope_beta_fast == 32.0f);
    REQUIRE(p.rope_beta_slow == 1.0f);
    // A source with attention_factor=1 explicitly disables magnitude scaling.
    REQUIRE(core_granite_llm::configure_yarn(hp.rope, 4.0f, 4096, 1.0f, 32.0f, 1.0f));
    REQUIRE(std::fabs(hp.rope.attn_factor - 0.878248857f) < 1e-7f);
}

TEST_CASE("granite_speech_rope: invalid scaling is rejected transactionally", "[unit][granite_speech]") {
    core_granite_llm::RopeScaling rope;
    for (float factor : {0.0f, 0.5f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        REQUIRE_FALSE(core_granite_llm::configure_yarn(rope, factor, 4096, 1.0f, 32.0f, 1.0f));
        REQUIRE(rope.ext_factor == 0.0f);
    }
    REQUIRE_FALSE(core_granite_llm::configure_yarn(rope, 4.0f, 0, 1.0f, 32.0f, 1.0f));
    REQUIRE_FALSE(core_granite_llm::configure_yarn(rope, 4.0f, 4096, 0.0f, 32.0f, 1.0f));
    REQUIRE_FALSE(core_granite_llm::configure_yarn(rope, 4.0f, 4096, 1.0f, 0.0f, 1.0f));
    REQUIRE_FALSE(core_granite_llm::configure_yarn(rope, 4.0f, 4096, 1.0f, 32.0f, -1.0f));
}

TEST_CASE("granite_speech_prompt: shared legacy template matches original tokenizer", "[unit][granite_speech]") {
    // Verified with the original5 tokenizer at c075cf19; the C ABI previously
    // split ? + newline and dropped the space before ASSISTANT in custom asks.
    REQUIRE(core_granite_prompt::kNumPrefix4 == 3);
    REQUIRE(core_granite_prompt::kNumSuffix4 == 15);
    REQUIRE(core_granite_prompt::kSuffix4[10] == 5380);
    REQUIRE(core_granite_prompt::legacy_suffix("transcribe.") == "transcribe.\n ASSISTANT:");
    REQUIRE(core_granite_prompt::legacy_suffix("transcribe.", "The") == "transcribe.\n ASSISTANT:The");
}
