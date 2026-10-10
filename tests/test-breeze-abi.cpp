// test-breeze-abi.cpp — the exported bt2-tts C entry points stay callable.
//
// breeze_tts_2.h is a C header other code compiles and links against, so a
// declaration that changes shape under the same symbol breaks callers at build
// time and a removed one breaks them at link time. Every call here passes a
// NULL context: the signatures and symbols are what is under test, and no
// weights are loaded.

#include <catch2/catch_test_macros.hpp>

#include "breeze_tts_2.h"

TEST_CASE("breeze abi — the two-scale guided entry point keeps its signature", "[unit][breeze-abi]") {
    float* (*guided)(breeze_tts_2_context*, const char*, const char*, const float*, int, const char*, float, float,
                     int*) = breeze_tts_2_synthesize_guided;
    int n = 7;
    REQUIRE(guided(nullptr, "hello", "calm", nullptr, 0, nullptr, 1.0f, 4.0f, &n) == nullptr);
    REQUIRE(n == 0);
}

TEST_CASE("breeze abi — the single-scale entry point is additive", "[unit][breeze-abi]") {
    int n = 7;
    REQUIRE(breeze_tts_2_synthesize_instructed(nullptr, "hello", "calm", nullptr, 0, nullptr, 4.0f, &n) == nullptr);
    REQUIRE(n == 0);
}

TEST_CASE("breeze abi — the capability query reports the modes this build runs", "[unit][breeze-abi]") {
    const uint32_t caps = breeze_tts_2_capabilities(nullptr);
    REQUIRE((caps & (1u << BREEZE_CFG_NONE)) != 0);
    REQUIRE((caps & (1u << BREEZE_CFG_INS)) != 0);
    REQUIRE((caps & (1u << BREEZE_CFG_BOTH)) != 0);
    // Upstream defines no negative prompt for a reference without an
    // instruction, so guidance toward the clip alone has nothing to run.
    REQUIRE((caps & (1u << BREEZE_CFG_REF)) == 0);
}
