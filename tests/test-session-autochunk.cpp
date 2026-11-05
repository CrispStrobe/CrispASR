// test-session-autochunk.cpp — unit tests for the session long-audio
// auto-chunk applicability decision (fix/session-long-audio).

#include <catch2/catch_test_macros.hpp>

#include "session_autochunk.h"

using core_session::session_autochunk_applicable;

namespace {
constexpr int SR = 16000;
// enabled, backend, secs, chunk_s, return_logits, already_chunking
bool ac(bool en, const char* b, int secs, int chunk_s, bool logits, bool already) {
    return session_autochunk_applicable(en, b, secs * SR, SR, chunk_s, logits, already);
}
} // namespace

TEST_CASE("auto-chunk fires for long audio on a chunk-needing backend", "[unit][session-autochunk]") {
    REQUIRE(ac(true, "moonshine", 60, 30, false, false));
    REQUIRE(ac(true, "whisper", 45, 30, false, false));
}

TEST_CASE("short audio (<= window) is not chunked", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(true, "moonshine", 20, 30, false, false));
    REQUIRE_FALSE(ac(true, "moonshine", 30, 30, false, false)); // exactly at the window
}

TEST_CASE("self-chunking backends are skipped", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(true, "parakeet", 300, 30, false, false));
    REQUIRE_FALSE(ac(true, "reazonspeech", 300, 30, false, false));
    REQUIRE_FALSE(ac(true, "index-echo", 300, 30, false, false));
}

TEST_CASE("disabled gate / logits / explicit-chunk skip", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(false, "moonshine", 300, 30, false, false)); // gate off
    REQUIRE_FALSE(ac(true, "moonshine", 300, 30, true, false));   // return_logits
    REQUIRE_FALSE(ac(true, "moonshine", 300, 30, false, true));   // already chunking
}

TEST_CASE("custom window respected", "[unit][session-autochunk]") {
    REQUIRE_FALSE(ac(true, "moonshine", 40, 60, false, false)); // 40s <= 60s window
    REQUIRE(ac(true, "moonshine", 70, 60, false, false));       // 70s > 60s window
}

TEST_CASE("degenerate inputs are safe", "[unit][session-autochunk]") {
    REQUIRE_FALSE(session_autochunk_applicable(true, "moonshine", 999 * SR, 0, 30, false, false)); // sr=0
    REQUIRE_FALSE(session_autochunk_applicable(true, "moonshine", 999 * SR, SR, 0, false, false)); // chunk=0
}

// F4: per-backend default auto-chunk window (opt-in via CRISPASR_SESSION_PERBACKEND_CHUNK).
TEST_CASE("per-backend window off (default): flat 30 s for every backend", "[unit][session-autochunk]") {
    using core_session::session_default_chunk_seconds;
    REQUIRE(session_default_chunk_seconds("moonshine", false) == 30);
    REQUIRE(session_default_chunk_seconds("moonshine-streaming", false) == 30);
    REQUIRE(session_default_chunk_seconds("whisper", false) == 30);
    REQUIRE(session_default_chunk_seconds("qwen3", false) == 30);
    REQUIRE(session_default_chunk_seconds("", false) == 30);
}

TEST_CASE("per-backend window on: short-segment models chunk smaller", "[unit][session-autochunk]") {
    using core_session::session_default_chunk_seconds;
    // Only the short-segment models change; everything else keeps 30 s.
    REQUIRE(session_default_chunk_seconds("moonshine", true) == 20);
    REQUIRE(session_default_chunk_seconds("moonshine-streaming", true) == 20);
    REQUIRE(session_default_chunk_seconds("moonshine", true) < 30);
    REQUIRE(session_default_chunk_seconds("whisper", true) == 30);
    REQUIRE(session_default_chunk_seconds("qwen3", true) == 30);
    REQUIRE(session_default_chunk_seconds("nemotron", true) == 30);
    REQUIRE(session_default_chunk_seconds("", true) == 30);
}

TEST_CASE("piece-proportional token budget: disabled when rate <= 0", "[unit][session-autochunk]") {
    using core_session::compute_piece_max_tokens;
    REQUIRE(compute_piece_max_tokens(0.0f, 16, 10 * SR, SR, 256) == 256);
    REQUIRE(compute_piece_max_tokens(-1.0f, 16, 10 * SR, SR, 100) == 100);
    REQUIRE(compute_piece_max_tokens(0.0f, 16, 10 * SR, SR, 0) == 0);
}

TEST_CASE("piece-proportional token budget: scales with piece duration", "[unit][session-autochunk]") {
    using core_session::compute_piece_max_tokens;
    // 10 s @ 10 tokens/s = 100 tokens (> minimum of 16)
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 10 * SR, SR, 0) == 100);
    // 30 s @ 10 tokens/s = 300 tokens
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 30 * SR, SR, 0) == 300);
    // 5.5 s @ 10 tokens/s = 55 tokens
    REQUIRE(compute_piece_max_tokens(10.0f, 16, (int)(5.5 * SR), SR, 0) == 55);
}

TEST_CASE("piece-proportional token budget: floors at minimum", "[unit][session-autochunk]") {
    using core_session::compute_piece_max_tokens;
    // 1 s @ 10 tokens/s = 10 tokens, floored at minimum 16
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 1 * SR, SR, 0) == 16);
    // 0.5 s @ 10 tokens/s = 5 tokens, floored at minimum 16
    REQUIRE(compute_piece_max_tokens(10.0f, 16, (int)(0.5 * SR), SR, 0) == 16);
}

TEST_CASE("piece-proportional token budget: caps at explicit session max_new_tokens", "[unit][session-autochunk]") {
    using core_session::compute_piece_max_tokens;
    // 30 s @ 10 tokens/s = 300 tokens, capped at explicit session cap of 200
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 30 * SR, SR, 200) == 200);
    // 10 s @ 10 tokens/s = 100 tokens, below cap of 200 -> 100
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 10 * SR, SR, 200) == 100);
}

TEST_CASE("piece-proportional token budget: degenerate inputs", "[unit][session-autochunk]") {
    using core_session::compute_piece_max_tokens;
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 0, SR, 0) == 16);
    REQUIRE(compute_piece_max_tokens(10.0f, 16, 10 * SR, 0, 0) == 16);
    REQUIRE(compute_piece_max_tokens(10.0f, -5, 0, SR, 0) == 0);
}
