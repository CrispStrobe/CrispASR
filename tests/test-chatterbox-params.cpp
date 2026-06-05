// test-chatterbox-params.cpp — unit tests for chatterbox_context_params defaults
// + the runtime sampling-knob setters (PLAN #88 / #89, May 2026).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "chatterbox.h"

TEST_CASE("chatterbox_params: default values are sensible", "[unit][chatterbox]") {
    struct chatterbox_context_params p = chatterbox_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature == Catch::Approx(0.8f));
    REQUIRE(p.cfg_weight == Catch::Approx(0.5f));
    REQUIRE(p.exaggeration == Catch::Approx(0.5f));
    REQUIRE(p.repetition_penalty == Catch::Approx(1.2f));
    REQUIRE(p.min_p == Catch::Approx(0.05f));
    REQUIRE(p.top_p == Catch::Approx(1.0f));
    REQUIRE(p.max_speech_tokens == 1000);
    REQUIRE(p.cfm_steps == 10);
}

// PLAN #89 plumbing — every backend with `use_gpu` now also carries
// a `flash_attn` field defaulting to true. Catches accidental
// regressions where someone removes the line again.
TEST_CASE("chatterbox_params: flash_attn defaults to true (#89)", "[unit][chatterbox]") {
    struct chatterbox_context_params p = chatterbox_context_default_params();
    REQUIRE(p.flash_attn == true);
}

// Runtime setters added in commit 95e2fdf7 (later restored in this
// session — see HISTORY entry). Each clamps to a sensible range
// and silently no-ops on out-of-bound input rather than crashing.
// We can exercise the clamping logic without an actual chatterbox
// context: build a fake context struct that mimics the public
// header's *_set_* fields layout. The setters touch only
// `ctx->params`, so a stub struct with that prefix-compatible
// layout works.
//
// In practice, calling the setters with a NULL context is the only
// safe pure-data assertion we can make from a unit test (you can't
// construct a real chatterbox_context without loading a GGUF).
// Confirm the null-guard works — most runtime crashes were
// historically NULL-deref in early-init paths.
TEST_CASE("chatterbox setters: NULL context is a no-op", "[unit][chatterbox]") {
    // Each setter must accept NULL without crashing. Catches the
    // common "added a setter but forgot the null-guard" regression.
    chatterbox_set_temperature(nullptr, 0.5f);
    chatterbox_set_top_p(nullptr, 0.9f);
    chatterbox_set_min_p(nullptr, 0.05f);
    chatterbox_set_repetition_penalty(nullptr, 1.0f);
    chatterbox_set_max_speech_tokens(nullptr, 1000);
    chatterbox_set_cfg_weight(nullptr, 0.5f);
    chatterbox_set_exaggeration(nullptr, 0.5f);
    chatterbox_set_cfm_steps(nullptr, 10);
    SUCCEED("All chatterbox setters tolerated a NULL ctx.");
}
