// test-arktts-params.cpp — unit tests for arktts_context_params defaults and null guards.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "arktts.h"

TEST_CASE("arktts_params: default values are sensible", "[unit][arktts]") {
    struct arktts_context_params p = arktts_context_default_params();

    REQUIRE(p.n_threads >= 0);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature > 0.0f);
    REQUIRE(p.top_p > 0.0f);
    REQUIRE(p.top_p <= 1.0f);
    REQUIRE(p.top_k > 0);
    REQUIRE(p.max_tokens > 0);
}

TEST_CASE("arktts_init_from_file: null/empty path returns nullptr", "[unit][arktts]") {
    struct arktts_context_params p = arktts_context_default_params();
    REQUIRE(arktts_init_from_file(nullptr, p) == nullptr);
    REQUIRE(arktts_init_from_file("", p) == nullptr);
}

TEST_CASE("arktts null context guards", "[unit][arktts]") {
    REQUIRE(arktts_get_sample_rate(nullptr) == 44100);
    REQUIRE(arktts_set_reference(nullptr, nullptr, 0, nullptr) != 0);
    int n = 0;
    REQUIRE(arktts_synthesize(nullptr, "hello", &n) == nullptr);
    arktts_free(nullptr);
    arktts_free_audio(nullptr);
}
