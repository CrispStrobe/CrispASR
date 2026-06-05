// test-params.cpp — unit tests for whisper_params struct defaults.
//
// Verifies that the crispasr-specific fields added to whisper_params carry
// the right default values so callers that don't set them get sensible
// behaviour without having to parse any command-line arguments.

#include <catch2/catch_test_macros.hpp>

#include "whisper_params.h"

// whisper_params is default-constructible; all fields use in-line initialisers.
// We capture the default-constructed instance once and then inspect it.
static const whisper_params kDefaults{};

// ─── new crispasr fields ──────────────────────────────────────────────────────

TEST_CASE("whisper_params: cache_dir defaults to empty (platform default used)", "[unit]") {
    REQUIRE(kDefaults.cache_dir.empty());
}

TEST_CASE("whisper_params: backend defaults to empty (whisper code path)", "[unit]") {
    REQUIRE(kDefaults.backend.empty());
}

TEST_CASE("whisper_params: punctuation defaults to true", "[unit]") {
    REQUIRE(kDefaults.punctuation == true);
}

TEST_CASE("whisper_params: max_new_tokens defaults to 512", "[unit]") {
    REQUIRE(kDefaults.max_new_tokens == 512);
}

TEST_CASE("whisper_params: frequency_penalty disabled by default", "[unit]") {
    REQUIRE(kDefaults.frequency_penalty == 0.0f);
}

TEST_CASE("whisper_params: chunk_seconds defaults to 30", "[unit]") {
    REQUIRE(kDefaults.chunk_seconds == 30);
}

// ─── VAD fields ───────────────────────────────────────────────────────────────

TEST_CASE("whisper_params: vad disabled by default", "[unit]") {
    REQUIRE(kDefaults.vad == false);
}

TEST_CASE("whisper_params: vad_threshold defaults to 0.5", "[unit]") {
    REQUIRE(kDefaults.vad_threshold == 0.5f);
}

// ─── stock whisper fields ────────────────────────────────────────────────────

TEST_CASE("whisper_params: use_gpu defaults to true", "[unit]") {
    REQUIRE(kDefaults.use_gpu == true);
}

TEST_CASE("whisper_params: temperature defaults to 0.0", "[unit]") {
    REQUIRE(kDefaults.temperature == 0.0f);
}

TEST_CASE("whisper_params: best_of defaults to sensible value", "[unit]") {
    REQUIRE(kDefaults.best_of >= 1);
}

TEST_CASE("whisper_params: beam_size defaults to sensible value", "[unit]") {
    REQUIRE(kDefaults.beam_size >= 1);
}

TEST_CASE("whisper_params: n_threads defaults to sensible value", "[unit]") {
    REQUIRE(kDefaults.n_threads >= 1);
}

TEST_CASE("whisper_params: language defaults to 'auto'", "[unit]") {
    REQUIRE(kDefaults.language == "auto");
}

TEST_CASE("whisper_params: print_progress defaults to false", "[unit]") {
    REQUIRE(kDefaults.print_progress == false);
}

TEST_CASE("whisper_params: progress_step defaults to 5", "[unit]") {
    REQUIRE(kDefaults.progress_step == 5);
}
