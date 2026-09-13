// test-supertonic-params.cpp — unit tests for supertonic_context_params
// defaults and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "supertonic_tts.h"

#include <cstring>

TEST_CASE("supertonic_params: default values are sensible", "[unit][supertonic]") {
    struct supertonic_context_params p = supertonic_context_default_params();
    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.total_steps > 0);
    REQUIRE(p.speed > 0.0f);
}

// Pin the exact upstream inference defaults (supertone-inc/supertonic
// py/example_onnx.py: total_step=8, speed=1.05, voice M1, lang en) so a
// drift that stays in-range still has to update this test on purpose.
TEST_CASE("supertonic_params: knobs match upstream inference defaults", "[unit][supertonic]") {
    struct supertonic_context_params p = supertonic_context_default_params();
    REQUIRE(p.total_steps == 8);
    REQUIRE(p.speed == Catch::Approx(1.05f));
    REQUIRE(std::strcmp(p.voice, "M1") == 0);
    REQUIRE(std::strcmp(p.language, "en") == 0);
    REQUIRE(p.seed == 0); // 0 = nondeterministic noise
}

TEST_CASE("supertonic_init_from_file: null path returns nullptr", "[unit][supertonic]") {
    struct supertonic_context_params p = supertonic_context_default_params();
    REQUIRE(supertonic_init_from_file(nullptr, p) == nullptr);
}

TEST_CASE("supertonic_init_from_file: missing file returns nullptr", "[unit][supertonic]") {
    struct supertonic_context_params p = supertonic_context_default_params();
    REQUIRE(supertonic_init_from_file("/nonexistent/supertonic.gguf", p) == nullptr);
}

TEST_CASE("supertonic setters: null context is safe", "[unit][supertonic]") {
    supertonic_set_n_threads(nullptr, 4);
    supertonic_set_speed(nullptr, 1.0f);
    supertonic_set_total_steps(nullptr, 8);
    supertonic_set_seed(nullptr, 1);
    REQUIRE(supertonic_set_voice(nullptr, "M1") != 0);
    REQUIRE(supertonic_set_language(nullptr, "en") != 0);
    REQUIRE(supertonic_sample_rate(nullptr) == 0);
    int n = -1;
    REQUIRE(supertonic_synthesize(nullptr, "hi", &n) == nullptr);
}
