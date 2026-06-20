// test-zonos-params.cpp — unit tests for zonos_tts_params defaults
// and null-guard coverage on public setters. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "zonos_tts.h"

TEST_CASE("zonos_params: default values are sensible", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature >= 0.0f);
    REQUIRE(p.cfg_scale > 0.0f);
    REQUIRE(p.max_audio_tokens >= 0);
}

TEST_CASE("zonos_init_from_file: null path returns nullptr", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();
    struct zonos_tts_context* ctx = zonos_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("zonos_init_from_file: empty path returns nullptr", "[unit][zonos]") {
    struct zonos_tts_params p = zonos_tts_default_params();
    struct zonos_tts_context* ctx = zonos_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("zonos setters: NULL context is a no-op", "[unit][zonos]") {
    zonos_tts_set_n_threads(nullptr, 4);
    zonos_tts_set_temperature(nullptr, 0.5f);
    zonos_tts_set_seed(nullptr, 42);
    zonos_tts_set_cfg_scale(nullptr, 2.0f);
    zonos_tts_set_pitch_std(nullptr, 20.0f);
    zonos_tts_set_speaking_rate(nullptr, 15.0f);
    zonos_tts_set_fmax(nullptr, 22050.0f);
    zonos_tts_set_codec_path(nullptr, "/dev/null");
    zonos_tts_set_language(nullptr, "en-us");
    zonos_tts_set_speaker_embedding(nullptr, nullptr, 0);
    zonos_tts_free(nullptr);
    SUCCEED("All zonos setters tolerated a NULL ctx.");
}
