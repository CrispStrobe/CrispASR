// test-kokoro-params.cpp — unit tests for kokoro_context_params defaults
// and null-guard coverage. No GGUF required.

#include <cstring>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "kokoro.h"

TEST_CASE("kokoro_params: default values are sensible", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #192/#197). Kokoro's one
// perceptual knob is speed, carried here as length_scale (1.0 = upstream default
// speed; >1 slower, <1 faster). Pin it plus the shipped defaults so a drift fails
// CI rather than silently changing every synthesis. espeak_lang seeds the G2P
// front-end and defaults to en-us upstream.
TEST_CASE("kokoro_params: value knobs match the shipped/upstream contract", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();

    REQUIRE(p.length_scale == Catch::Approx(1.0f)); // upstream Kokoro speed default 1.0
    REQUIRE(std::strcmp(p.espeak_lang, "en-us") == 0);
    REQUIRE(p.use_gpu == true);
    REQUIRE(p.flash_attn == true); // PLAN #89 plumbing — must stay on
}

TEST_CASE("kokoro_init_from_file: null path returns nullptr", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kokoro_init_from_file: empty path returns nullptr", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kokoro_free: NULL context is a no-op", "[unit][kokoro]") {
    kokoro_free(nullptr);
    SUCCEED("kokoro_free tolerated a NULL ctx.");
}

TEST_CASE("kokoro_length_scale: NULL context safety", "[unit][kokoro]") {
    kokoro_set_length_scale(nullptr, 0.5f);
    SUCCEED("kokoro_set_length_scale tolerated a NULL ctx.");
    REQUIRE(kokoro_get_length_scale(nullptr) == Catch::Approx(1.0f));
}

TEST_CASE("kokoro_length_scale: boundary clamping on context", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_context_create_for_testing(p);
    REQUIRE(ctx != nullptr);

    // Initial default is 1.0
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(1.0f));

    // Underflow clamping (< 0.25 -> 0.25)
    kokoro_set_length_scale(ctx, 0.1f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(0.25f));

    // Zero clamping (0.0 -> 0.25)
    kokoro_set_length_scale(ctx, 0.0f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(0.25f));

    // Negative clamping (-2.0 -> 0.25)
    kokoro_set_length_scale(ctx, -2.0f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(0.25f));

    // NaN -> 1.0, the NEUTRAL value, NOT the 0.25 clamp floor. length_scale is
    // 1/speed, so 0.25 is the fastest (4x) and most degraded setting; mapping
    // malformed input onto it would let the least-deliberate input path pick
    // the worst-sounding extreme. Garbage in must mean "no change", not "as
    // fast as possible". Guard the distinction explicitly: a regression to the
    // old behaviour lands on 0.25 and fails here.
    kokoro_set_length_scale(ctx, std::numeric_limits<float>::quiet_NaN());
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(1.0f));
    REQUIRE(kokoro_get_length_scale(ctx) != Catch::Approx(0.25f));

    // ...and the surrounding clamp still behaves: NaN is special-cased, not
    // treated as "any invalid value becomes neutral".
    kokoro_set_length_scale(ctx, 0.1f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(0.25f));

    // Overflow clamping (> 4.0 -> 4.0)
    kokoro_set_length_scale(ctx, 10.0f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(4.0f));

    // Normal values within [0.25, 4.0]
    kokoro_set_length_scale(ctx, 0.5f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(0.5f));

    kokoro_set_length_scale(ctx, 2.0f);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(2.0f));

    kokoro_free(ctx);
}

TEST_CASE("kokoro_length_scale: sequential request reset prevents state leakage", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_context_create_for_testing(p);
    REQUIRE(ctx != nullptr);

    // Simulate Request 1: speed = 2.0f -> length_scale = 0.5f
    float speed1 = 2.0f;
    float scale1 = (speed1 > 0.0f) ? (1.0f / speed1) : 1.0f;
    kokoro_set_length_scale(ctx, scale1);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(0.5f));

    // Simulate Request 2: speed = 1.0f (default) -> length_scale = 1.0f (must reset)
    float speed2 = 1.0f;
    float scale2 = (speed2 > 0.0f) ? (1.0f / speed2) : 1.0f;
    kokoro_set_length_scale(ctx, scale2);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(1.0f));

    // Simulate Request 3: speed <= 0.0f (sentinel / unset) -> length_scale = 1.0f
    float speed3 = -1.0f;
    float scale3 = (speed3 > 0.0f) ? (1.0f / speed3) : 1.0f;
    kokoro_set_length_scale(ctx, scale3);
    REQUIRE(kokoro_get_length_scale(ctx) == Catch::Approx(1.0f));

    kokoro_free(ctx);
}
