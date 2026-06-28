// test-tada-params.cpp — unit tests for tada_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "tada_tts.h"

TEST_CASE("tada_params: default values are sensible", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

// Defaults-audit / config-parity guard (motivated by #197). crispasr-diff
// compares the C++ forward pass against a Python reference pinned to a
// non-default mode (text_do_sample=False) for determinism, so it cannot catch a
// production default diverging from upstream — the talker shipping greedy with a
// dead `temperature` field was invisible to the stage diff for exactly that
// reason. This encodes the upstream tada.py InferenceOptions contract directly,
// so a future regression (a knob reset to a non-upstream value, or removed)
// fails CI.
TEST_CASE("tada_params: value knobs match upstream InferenceOptions", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();

    REQUIRE(p.temperature == Catch::Approx(0.6f));             // text_temperature
    REQUIRE(p.text_top_p == Catch::Approx(0.9f));              // text_top_p
    REQUIRE(p.text_top_k == 0);                                // text_top_k
    REQUIRE(p.text_repetition_penalty == Catch::Approx(1.1f)); // text_repetition_penalty
    REQUIRE(p.acoustic_cfg == Catch::Approx(1.6f));            // acoustic_cfg_scale
    REQUIRE(p.noise_temp == Catch::Approx(0.9f));              // noise_temperature
    REQUIRE(p.num_fm_steps == 0);                              // 0 = the 10-step default (num_flow_matching_steps)
}

// The one deliberate lib-vs-production split: the LIBRARY default is
// deterministic (greedy text + single FM candidate) so crispasr-diff matches
// the greedy Python reference bit-exactly, while the CLI adapter and session
// C ABI flip these to upstream production values (do_sample=True,
// num_acoustic_candidates=4). Guarding both directions stops a future change
// from silently shipping greedy to production again [the #197 regression] or
// making the diff harness non-deterministic.
TEST_CASE("tada_params: library default is deterministic for diff parity", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();
    REQUIRE(p.text_do_sample == false);
    REQUIRE(p.num_acoustic_candidates == 1);
}

// Sampling setters must be NULL-safe (each guards `if (ctx)`).
TEST_CASE("tada setters: null context is a safe no-op", "[unit][tada]") {
    tada_set_do_sample(nullptr, true);
    tada_set_top_p(nullptr, 0.8f);
    tada_set_top_k(nullptr, 50);
    tada_set_repetition_penalty(nullptr, 1.3f);
    tada_set_temperature(nullptr, 0.7f);
    tada_set_num_candidates(nullptr, 4);
    SUCCEED("null-context setters did not crash");
}

TEST_CASE("tada_init_from_file: null path returns nullptr", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();
    struct tada_context* ctx = tada_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("tada_init_from_file: empty path returns nullptr", "[unit][tada]") {
    struct tada_context_params p = tada_context_default_params();
    struct tada_context* ctx = tada_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("tada_free: NULL context is a no-op", "[unit][tada]") {
    tada_free(nullptr);
    SUCCEED("tada_free tolerated a NULL ctx.");
}
