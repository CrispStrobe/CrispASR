// tests/test_miotts_live.cpp — live integration test for MioTTS backend.
//
// Requires CRISPASR_MODEL_MIOTTS env var pointing to a MioTTS GGUF.
// Generates speech from text and verifies non-empty PCM output.

#include "miotts.h"
#include "gguf.h"
#include "crispasr_backend.h"
#include "whisper_params.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

TEST_CASE("miotts: init from GGUF", "[miotts][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_MIOTTS");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_MIOTTS not set");
        return;
    }
    auto p = miotts_context_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    p.max_tokens = 50;
    p.temperature = 0.0f;
    auto* ctx = miotts_init_from_file(model, p);
    REQUIRE(ctx != nullptr);
    auto* gguf = gguf_init_from_file(model, {true, nullptr});
    REQUIRE(gguf != nullptr);
    const int64_t key = gguf_find_key(gguf, "miotts.codec.sample_rate");
    const int expected = key < 0 ? 24000 : static_cast<int>(gguf_get_val_u32(gguf, key));
    REQUIRE(miotts_get_sample_rate(ctx) == expected);
    gguf_free(gguf);
    miotts_free(ctx);
}

TEST_CASE("miotts: synthesize produces audio", "[miotts][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_MIOTTS");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_MIOTTS not set");
        return;
    }
    auto p = miotts_context_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    p.max_tokens = 50;
    p.temperature = 0.0f;
    auto* ctx = miotts_init_from_file(model, p);
    REQUIRE(ctx != nullptr);

    int n = 0;
    float* pcm = miotts_synthesize(ctx, "Hello", &n);
    // With zero embedding the audio may not be intelligible,
    // but PCM should be non-empty and non-silent.
    REQUIRE(pcm != nullptr);
    {
        REQUIRE(n > 0);
        // Check non-silent: at least one sample with abs > 0.001
        bool has_audio = false;
        for (int i = 0; i < n; i++) {
            if (pcm[i] > 0.001f || pcm[i] < -0.001f) {
                has_audio = true;
                break;
            }
        }
        REQUIRE(has_audio);
        miotts_free_audio(pcm);
    }
    miotts_free(ctx);
}

TEST_CASE("miotts: FSQ dequant exact", "[miotts][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_MIOTTS");
    if (!model || !*model) {
        SKIP("CRISPASR_MODEL_MIOTTS not set");
        return;
    }
    auto p = miotts_context_default_params();
    p.n_threads = 4;
    p.verbosity = 0;
    auto* ctx = miotts_init_from_file(model, p);
    REQUIRE(ctx != nullptr);

    // FSQ index 0 → codes [0,0,0,0,0] → normalized [-1,-1,-1,-1,-1]
    int32_t idx = 0;
    int dim = 0;
    float* emb = miotts_fsq_dequant(ctx, &idx, 1, &dim);
    REQUIRE(emb != nullptr);
    REQUIRE(dim > 0);
    miotts_free_audio(emb);
    miotts_free(ctx);
}

// Test the same adapter a resident HTTP server calls with per-request params.
std::unique_ptr<CrispasrBackend> crispasr_create_miotts_backend();

TEST_CASE("miotts: resident adapter restores startup voice", "[miotts][live]") {
    const char* model = std::getenv("CRISPASR_MODEL_MIOTTS");
    const char* voice_dir = std::getenv("CRISPASR_MIOTTS_VOICE_DIR");
    if (!model || !*model || !voice_dir || !*voice_dir) {
        SKIP("CRISPASR_MODEL_MIOTTS / CRISPASR_MIOTTS_VOICE_DIR not set");
        return;
    }
    whisper_params startup;
    startup.model = model;
    startup.use_gpu = false;
    startup.n_threads = 4;
    startup.temperature = 0;
    startup.tts_voice_dir = voice_dir;
    startup.tts_voice = "en_female.emb.gguf";
    auto backend = crispasr_create_miotts_backend();
    REQUIRE(backend->init(startup));
    REQUIRE(backend->tts_sample_rate() == 44100);
    const auto first = backend->synthesize("Hello world", startup);
    REQUIRE_FALSE(first.empty());
    auto request = startup;
    request.tts_voice = "en_male";
    const auto alternate = backend->synthesize("Hello world", request);
    REQUIRE_FALSE(alternate.empty());
    REQUIRE(alternate != first);
    request.tts_voice.clear();
    const auto restored = backend->synthesize("Hello world", request);
    REQUIRE(restored == first);
    request.tts_voice = "missing-preset.emb.gguf";
    REQUIRE(backend->synthesize("Hello world", request).empty());
    request.tts_voice.clear();
    REQUIRE(backend->synthesize("Hello world", request) == first);
    backend->shutdown();
}
