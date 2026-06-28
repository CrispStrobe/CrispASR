// test-session-setters.cpp — unit tests for the generation-control session
// setters added in the full-parity sweep (PLAN §88 / commit c88306fa).
//
// Strategy: call every setter with a null session handle and verify the
// null-guard path returns -1.  No model is loaded — sub-millisecond, no
// network, safe on every CI tier.  A regression here means a setter lost its
// null guard (silent crash risk for callers that open sessions lazily).
//
// Coverage: all 19 setters declared in include/crispasr.h under the
// "Unified session decode / sampling controls" block.

#include <catch2/catch_test_macros.hpp>

#include "crispasr.h"

// ─── single-float setters ──────────────────────────────────────────────────

TEST_CASE("session setter: set_temperature null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_temperature(nullptr, 0.8f, 42) == -1);
}

TEST_CASE("session setter: set_top_p null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_top_p(nullptr, 0.9f) == -1);
}

TEST_CASE("session setter: set_min_p null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_min_p(nullptr, 0.05f) == -1);
}

TEST_CASE("session setter: set_repetition_penalty null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_repetition_penalty(nullptr, 1.2f) == -1);
}

TEST_CASE("session setter: set_top_k null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_top_k(nullptr, 5) == -1);
}

TEST_CASE("session setter: set_do_sample null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_do_sample(nullptr, 1) == -1);
}

TEST_CASE("session setter: set_tts_num_candidates null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_num_candidates(nullptr, 4) == -1);
}

TEST_CASE("session setter: set_cfg_weight null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_cfg_weight(nullptr, 0.5f) == -1);
}

TEST_CASE("session setter: set_tts_noise_temp null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_noise_temp(nullptr, 0.9f) == -1);
}

TEST_CASE("session setter: set_exaggeration null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_exaggeration(nullptr, 0.5f) == -1);
}

TEST_CASE("session setter: set_length_scale null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_length_scale(nullptr, 1.0f) == -1);
}

TEST_CASE("session setter: set_frequency_penalty null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_frequency_penalty(nullptr, 0.4f) == -1);
}

// ─── single-int / uint64 setters ──────────────────────────────────────────

TEST_CASE("session setter: set_tts_seed null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_seed(nullptr, 12345) == -1);
}

TEST_CASE("session setter: set_tts_steps null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_tts_steps(nullptr, 20) == -1);
}

TEST_CASE("session setter: set_max_new_tokens null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_max_new_tokens(nullptr, 256) == -1);
}

TEST_CASE("session setter: set_max_speech_tokens null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_max_speech_tokens(nullptr, 1000) == -1);
}

TEST_CASE("session setter: set_best_of null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_best_of(nullptr, 5) == -1);
}

TEST_CASE("session setter: set_beam_size null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_beam_size(nullptr, 4) == -1);
}

TEST_CASE("session setter: set_alt_n null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_alt_n(nullptr, 3) == -1);
}

// ─── string / multi-param setters ─────────────────────────────────────────

TEST_CASE("session setter: set_ask null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_ask(nullptr, "hello") == -1);
}

TEST_CASE("session setter: set_grammar_text null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_grammar_text(nullptr, nullptr, nullptr, 100.0f) == -1);
}

TEST_CASE("session setter: set_fallback_thresholds null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_fallback_thresholds(nullptr, 2.4f, -1.0f, 0.6f, 0.2f) == -1);
}

TEST_CASE("session setter: set_whisper_decode_extras null-handle → -1", "[unit][setters]") {
    REQUIRE(crispasr_session_set_whisper_decode_extras(nullptr, 0, nullptr, 0) == -1);
}

// ─── last_synth_error ─────────────────────────────────────────────────────

TEST_CASE("session: last_synth_error null-handle → empty string", "[unit][setters]") {
    const char* err = crispasr_session_last_synth_error(nullptr);
    REQUIRE(err != nullptr);
    REQUIRE(err[0] == '\0');
}
