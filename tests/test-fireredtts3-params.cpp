// test-fireredtts3-params.cpp — unit tests for fireredtts3_tts_context_params
// defaults and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fireredtts3_tts.h"

TEST_CASE("fireredtts3_params: default values are sensible", "[unit][fireredtts3]") {
    struct fireredtts3_tts_context_params p = fireredtts3_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    // 0 / negative sentinels mean "use the upstream default" — the concrete
    // values are resolved at init (seed 1234, 10 timesteps, CFG 2.0,
    // stop 0.5, 400 patches). Pin the SENTINEL contract here so a drift to
    // eager concrete defaults has to update this test on purpose.
    REQUIRE(p.seed == 0);
    REQUIRE(p.n_timesteps == 0);
    REQUIRE(p.cfg_scale < 0.0f); // < 0 sentinel: 0.0 is meaningful (CFG off)
    REQUIRE(p.stop_threshold == 0.0f);
    REQUIRE(p.max_patches == 0);
}

TEST_CASE("fireredtts3_init_from_file: null path returns nullptr", "[unit][fireredtts3]") {
    struct fireredtts3_tts_context_params p = fireredtts3_tts_context_default_params();
    REQUIRE(fireredtts3_tts_init_from_file(nullptr, p) == nullptr);
    REQUIRE(fireredtts3_tts_init_from_file("", p) == nullptr);
}

TEST_CASE("fireredtts3 setters tolerate a NULL context", "[unit][fireredtts3]") {
    REQUIRE(fireredtts3_tts_set_redae_path(nullptr, "x.gguf") != 0);
    REQUIRE(fireredtts3_tts_set_voice_pcm(nullptr, nullptr, 0) != 0);
    REQUIRE(fireredtts3_tts_set_ref_text(nullptr, "x") != 0);
    REQUIRE(fireredtts3_tts_set_language(nullptr, "English") != 0);
    int n = -1;
    REQUIRE(fireredtts3_tts_synthesize(nullptr, "hi", &n) == nullptr);
    REQUIRE(fireredtts3_tts_sample_rate(nullptr) == 24000);
    fireredtts3_tts_free(nullptr); // must not crash
    fireredtts3_tts_pcm_free(nullptr);
}
