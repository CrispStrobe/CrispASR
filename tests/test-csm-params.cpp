// test-csm-params.cpp — unit tests for csm_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "csm_tts.h"
#include "csm_rope.h"
#include "csm_rope_reference.h"
#include <cmath>
#include <limits>

TEST_CASE("csm_params: default values are sensible", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Pin the exact
// upstream Sesame CSM generate defaults so a drift fails CI.
TEST_CASE("csm_params: value knobs match upstream CSM defaults", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.9f));
    REQUIRE(p.topk == 50);
    REQUIRE(p.seed == 0);             // 0 = non-deterministic
    REQUIRE(p.max_audio_tokens == 0); // 0 = model default
    REQUIRE(p.use_gpu == false);
}

TEST_CASE("csm_init_from_file: null path returns nullptr", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();
    struct csm_tts_context* ctx = csm_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("csm_init_from_file: empty path returns nullptr", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();
    struct csm_tts_context* ctx = csm_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("csm_free: NULL context is a no-op", "[unit][csm]") {
    csm_tts_free(nullptr);
    SUCCEED("csm_tts_free tolerated a NULL ctx.");
}

TEST_CASE("csm_rope: both frequency tables match the pinned original operator", "[unit][csm][rope]") {
    auto check = [](const std::vector<float>& actual, const auto& expected) {
        REQUIRE(actual.size() == expected.size());
        double squared_error = 0, squared_reference = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            REQUIRE(std::isfinite(actual[i]));
            REQUIRE(std::abs(actual[i] - expected[i]) <= 1e-4f);
            const double delta = (double)actual[i] - expected[i];
            squared_error += delta * delta;
            squared_reference += (double)expected[i] * expected[i];
        }
        REQUIRE(std::sqrt(squared_error / squared_reference) <= 2e-6);
    };
    check(csm_rope::frequency_factors(64, 500000.0f, 32.0f, 0.125f, 0.5f, 1024), csm_rope_reference::backbone);
    check(csm_rope::frequency_factors(128, 500000.0f, 32.0f, 0.001953125f, 0.0078125f, 16), csm_rope_reference::depth);
}

TEST_CASE("csm_rope: unscaled config remains explicit and malformed inputs fail", "[unit][csm][rope]") {
    auto plain = csm_rope::frequency_factors(64, 500000.0f, 1.0f, 1.0f, 4.0f, 2048);
    REQUIRE(plain.size() == 32);
    for (float value : plain)
        REQUIRE(value == 1.0f);
    REQUIRE(csm_rope::frequency_factors(65, 500000, 32, 0.125f, 0.5f, 1024).empty());
    REQUIRE(csm_rope::frequency_factors(64, 500000, 0.5f, 0.125f, 0.5f, 1024).empty());
    REQUIRE(csm_rope::frequency_factors(64, 500000, 32, 0.125f, 0.125f, 1024).empty());
    REQUIRE(csm_rope::frequency_factors(64, 500000, 32, 0.125f, 0.5f, 0).empty());
    REQUIRE(csm_rope::frequency_factors(64, std::numeric_limits<float>::quiet_NaN(), 32, 0.125f, 0.5f, 1024).empty());
}
