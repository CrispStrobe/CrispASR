// tests/test-espeak-phonemize.cpp — unit tests for espeak-ng phonemization
// used by piper and kokoro TTS backends.
//
// These tests verify:
//   1. piper_tts_has_espeak() reports availability correctly
//   2. piper_tts_synthesize() works with plain English text (requires espeak)
//   3. piper_tts_synthesize_phonemes() works with pre-phonemized IPA
//   4. Synthesized audio has sane properties (non-silent, correct sample rate)

#include <catch2/catch_test_macros.hpp>

#include "piper_tts.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

// Model path passed via -DPIPER_TEST_MODEL="..." at compile time,
// or via PIPER_TEST_MODEL env var at runtime.
static std::string get_model_path() {
    const char* env = std::getenv("PIPER_TEST_MODEL");
    if (env && *env) return env;
#ifdef PIPER_TEST_MODEL
    return PIPER_TEST_MODEL;
#else
    return "";
#endif
}

static float compute_rms(const float* pcm, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)pcm[i] * (double)pcm[i];
    return (float)std::sqrt(sum / n);
}

TEST_CASE("espeak-ng availability", "[espeak]") {
    // Reports whether espeak-ng is reachable via dlopen or popen.
    // May be false if espeak-ng is not installed — that's fine,
    // the test just documents the current state.
    bool has = piper_tts_has_espeak();
    INFO("piper_tts_has_espeak() = " << has);
    if (!has) {
        WARN("espeak-ng not available — text synthesis tests will be skipped");
    }
    SUCCEED();
}

TEST_CASE("piper phoneme synthesis (IPA)", "[piper][espeak]") {
    std::string model = get_model_path();
    if (model.empty()) {
        SKIP("PIPER_TEST_MODEL not set — skipping (set env var or -DPIPER_TEST_MODEL)");
    }

    struct piper_tts_params pp = piper_tts_default_params();
    pp.verbosity = 0;
    pp.noise_scale = 0.0f;
    pp.noise_w = 0.0f;

    auto* ctx = piper_tts_init_from_file(model.c_str(), pp);
    REQUIRE(ctx != nullptr);

    SECTION("IPA phonemes produce non-silent audio") {
        float* pcm = nullptr;
        int sr = 0;
        // "hello world" in IPA
        int n = piper_tts_synthesize_phonemes(ctx, "h\xc9\x99l\xcb\x88o\xca\x8a w\xcb\x88\xc9\x9c\xcb\x90ld", &pcm, &sr);
        REQUIRE(n > 0);
        REQUIRE(pcm != nullptr);
        REQUIRE(sr == 22050);

        float rms = compute_rms(pcm, n);
        INFO("RMS = " << rms << ", n_samples = " << n);
        // Audio should not be silent (RMS > 0.01)
        REQUIRE(rms > 0.01f);
        // Duration should be reasonable (0.3s - 5s for "hello world")
        float dur = (float)n / (float)sr;
        REQUIRE(dur > 0.3f);
        REQUIRE(dur < 5.0f);

        free(pcm);
    }

    SECTION("sample rate matches model") {
        REQUIRE(piper_tts_sample_rate(ctx) == 22050);
    }

    SECTION("espeak voice defaults to model config") {
        const char* voice = piper_tts_espeak_voice(ctx);
        REQUIRE(voice != nullptr);
        // lessac-medium is en-us
        REQUIRE(std::string(voice).find("en") != std::string::npos);
    }

    piper_tts_free(ctx);
}

TEST_CASE("piper text synthesis (espeak phonemization)", "[piper][espeak]") {
    std::string model = get_model_path();
    if (model.empty()) {
        SKIP("PIPER_TEST_MODEL not set");
    }
    if (!piper_tts_has_espeak()) {
        SKIP("espeak-ng not available");
    }

    struct piper_tts_params pp = piper_tts_default_params();
    pp.verbosity = 0;
    pp.noise_scale = 0.0f;
    pp.noise_w = 0.0f;

    auto* ctx = piper_tts_init_from_file(model.c_str(), pp);
    REQUIRE(ctx != nullptr);

    SECTION("plain English text produces non-silent audio") {
        float* pcm = nullptr;
        int sr = 0;
        int n = piper_tts_synthesize(ctx, "Hello world", &pcm, &sr);
        REQUIRE(n > 0);
        REQUIRE(pcm != nullptr);
        REQUIRE(sr == 22050);

        float rms = compute_rms(pcm, n);
        INFO("RMS = " << rms);
        REQUIRE(rms > 0.01f);

        free(pcm);
    }

    SECTION("longer text produces proportionally longer audio") {
        float* pcm_short = nullptr;
        float* pcm_long = nullptr;
        int sr = 0;
        int n_short = piper_tts_synthesize(ctx, "Hi.", &pcm_short, &sr);
        int n_long = piper_tts_synthesize(ctx, "The quick brown fox jumps over the lazy dog.", &pcm_long, &sr);
        REQUIRE(n_short > 0);
        REQUIRE(n_long > 0);
        // Longer text should produce more audio
        REQUIRE(n_long > n_short);

        free(pcm_short);
        free(pcm_long);
    }

    SECTION("language override works") {
        piper_tts_set_language(ctx, "en-us");
        float* pcm = nullptr;
        int sr = 0;
        int n = piper_tts_synthesize(ctx, "Testing", &pcm, &sr);
        REQUIRE(n > 0);
        free(pcm);
    }

    piper_tts_free(ctx);
}

TEST_CASE("piper rejects null/empty inputs", "[piper]") {
    std::string model = get_model_path();
    if (model.empty()) {
        SKIP("PIPER_TEST_MODEL not set");
    }

    struct piper_tts_params pp = piper_tts_default_params();
    pp.verbosity = 0;

    auto* ctx = piper_tts_init_from_file(model.c_str(), pp);
    REQUIRE(ctx != nullptr);

    float* pcm = nullptr;
    int sr = 0;

    SECTION("null text returns 0") {
        REQUIRE(piper_tts_synthesize(ctx, nullptr, &pcm, &sr) == 0);
    }

    SECTION("null pcm_out returns 0") {
        REQUIRE(piper_tts_synthesize(ctx, "hello", nullptr, &sr) == 0);
    }

    SECTION("null context returns 0") {
        REQUIRE(piper_tts_synthesize(nullptr, "hello", &pcm, &sr) == 0);
    }

    piper_tts_free(ctx);
}
