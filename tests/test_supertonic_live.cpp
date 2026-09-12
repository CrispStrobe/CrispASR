// tests/test_supertonic_live.cpp — live integration tests for Supertonic-3.
//
// Requires CRISPASR_MODEL_SUPERTONIC pointing at supertonic3-f16.gguf (single
// file — voices, unicode indexer and NFKD tables are embedded).

#include "supertonic_tts.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>

TEST_CASE("supertonic: init from GGUF", "[supertonic][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_SUPERTONIC");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_SUPERTONIC not set");
        return;
    }
    auto p = supertonic_context_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    auto* ctx = supertonic_init_from_file(model, p);
    REQUIRE(ctx != nullptr);
    REQUIRE(supertonic_sample_rate(ctx) == 44100);
    REQUIRE(supertonic_set_voice(ctx, "F1") == 0);
    REQUIRE(supertonic_set_voice(ctx, "no-such-voice") != 0);
    REQUIRE(supertonic_set_language(ctx, "de") == 0);
    REQUIRE(supertonic_set_language(ctx, "xx") != 0);
    supertonic_free(ctx);
}

TEST_CASE("supertonic: synthesize produces non-silent audio", "[supertonic][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_SUPERTONIC");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_SUPERTONIC not set");
        return;
    }
    auto p = supertonic_context_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    p.seed = 1234;
    p.total_steps = 8;
    auto* ctx = supertonic_init_from_file(model, p);
    REQUIRE(ctx != nullptr);
    int n = 0;
    float* pcm = supertonic_synthesize(ctx, "Hello there.", &n);
    REQUIRE(pcm != nullptr);
    REQUIRE(n > 44100 / 2); // at least half a second
    double e = 0;
    for (int i = 0; i < n; i++)
        e += (double)pcm[i] * pcm[i];
    REQUIRE(std::sqrt(e / n) > 1e-3); // non-silent RMS
    free(pcm);
    supertonic_free(ctx);
}
