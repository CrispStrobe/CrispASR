// tests/test_zonos3_live.cpp — live tests for the ZONOS3-Mini TTS backend (#603).
//
// CRISPASR_MODEL_ZONOS3 = Zyphra's gguf/backbone-ptq.gguf (zac-keep2.gguf beside it or CRISPASR_ZONOS3_CODEC).
// Optional CRISPASR_ZONOS3_VOICE = a gguf/voices/<Lang>/<name>.embedding.npy.
// Intelligibility is covered by the TTS->ASR roundtrip (tests/regression manifest entry zonos3-mini-en);
// these cases pin the invariants the roundtrip cannot see.

#include "zonos3_tts.h"
#include "zonos3/zonos3.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>

static const char* model_path() {
    const char* m = std::getenv("CRISPASR_MODEL_ZONOS3");
    return m && *m ? m : nullptr;
}

TEST_CASE("zonos3: condition buckets are bisect_right over the GGUF edges", "[zonos3][live]") {
    const char* m = model_path();
    if (!m) {
        SKIP("CRISPASR_MODEL_ZONOS3 not set");
        return;
    }
    zonos3_model model;
    REQUIRE(zonos3_model_load(model, m, false));
    REQUIRE(model.hp.n_cond == 3);
    for (uint32_t c = 0; c < model.hp.n_cond; ++c) {
        const auto& e = model.hp.cond_edges[c];
        REQUIRE(e.size() == model.hp.cond_bucket_counts[c] - 1);
        // Python bisect.bisect_right: a value equal to edge i lands in bucket i + 1, just below it in bucket i.
        for (size_t i = 0; i < e.size(); ++i) {
            REQUIRE(zonos3_condition_bucket(model, (int)c, e[i]) == (int)i + 1);
            REQUIRE(zonos3_condition_bucket(model, (int)c, std::nextafter((double)e[i], -INFINITY)) == (int)i);
        }
        REQUIRE(zonos3_condition_bucket(model, (int)c, NAN) == -1);
    }
    zonos3_model_free(model);
}

TEST_CASE("zonos3: greedy synthesis yields whole frames of non-silent 44.1 kHz audio", "[zonos3][live]") {
    const char* m = model_path();
    if (!m) {
        SKIP("CRISPASR_MODEL_ZONOS3 not set");
        return;
    }
    auto p = zonos3_tts_context_default_params();
    p.verbosity = 0;
    p.temperature = 0.0f;
    p.max_frames = 200;
    auto* ctx = zonos3_tts_init_from_file(m, p);
    REQUIRE(ctx != nullptr);
    if (const char* v = std::getenv("CRISPASR_ZONOS3_VOICE"))
        REQUIRE(zonos3_tts_load_speaker_embedding(ctx, v) == 0);
    int frames = 0, eos = -1, prefill = 0;
    int32_t* codes = zonos3_tts_generate_codes(
        ctx, "The quick brown fox jumps over the lazy dog. Today is a beautiful day for a walk in the park.", &frames,
        &eos, &prefill);
    REQUIRE(codes != nullptr);
    REQUIRE(prefill == 3);
    // Greedy needs real text to reach EOA (a two-word prompt can run to the budget); this one ends near frame 94.
    REQUIRE(eos > prefill);
    // Delay pattern: the first generated row carries codebook 0..prefill only; later codebooks are pad (4097).
    for (int cb = prefill + 1; cb < 10; ++cb)
        REQUIRE(codes[(size_t)prefill * 10 + cb] == 4097);
    int n = 0;
    float* pcm = zonos3_tts_decode_codes(ctx, codes, frames, eos, prefill, &n);
    REQUIRE(pcm != nullptr);
    REQUIRE(zonos3_tts_get_sample_rate(ctx) == 44100);
    // zac_decode emits (eos) frames x 2940 samples; the prefill frames' audio is dropped.
    REQUIRE(n == (eos - prefill) * 2940);
    double e2 = 0;
    for (int i = 0; i < n; ++i)
        e2 += (double)pcm[i] * pcm[i];
    REQUIRE(std::sqrt(e2 / n) > 0.01);
    zonos3_tts_free_audio(pcm);
    zonos3_tts_free_codes(codes);
    zonos3_tts_free(ctx);
}
