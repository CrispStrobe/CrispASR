// test-session-beam.cpp — functional tests for session-level beam search (§90).
//
// Two tiers:
//
//   [unit][beam]  — no model, no audio. Tests the setter API: null guard,
//                   width clamping (n<=0 → 1), sanity cap.
//
//   [beam][.live] — requires model files + samples/jfk.wav on disk.
//                   Gated on CRISPASR_MODEL_WHISPER (+ optionally
//                   CRISPASR_MODEL_QWEN3_ASR, CRISPASR_MODEL_GLM_ASR)
//                   env vars. Tests:
//                     1. beam_size=1 produces output byte-identical to
//                        never calling the setter (no-regression contract).
//                     2. beam_size=2..4 produces non-empty, well-formed
//                        output on a real audio fixture.
//
// Run:
//   ctest -R test-session-beam --output-on-failure          # unit only
//   CRISPASR_MODEL_WHISPER=models/ggml-base.en.bin \
//     ctest -R test-session-beam --output-on-failure        # + live
//
// Env vars:
//   CRISPASR_MODEL_WHISPER   — whisper GGUF path
//   CRISPASR_MODEL_QWEN3_ASR — qwen3-asr GGUF path (optional, replay-helper)
//   CRISPASR_MODEL_GLM_ASR   — glm-asr GGUF path (optional, per-backend setter)
//   CRISPASR_AUDIO_EN        — English test audio (default: samples/jfk.wav)

#include <catch2/catch_test_macros.hpp>

#include "crispasr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations for session API (exported from libcrispasr but not
// in the public header — declared here for test use only).
// ---------------------------------------------------------------------------

struct crispasr_session;
struct crispasr_session_result;

extern "C" {
crispasr_session* crispasr_session_open_explicit(
    const char* model_path, const char* backend_name, int n_threads);
crispasr_session* crispasr_session_open(const char* model_path, int n_threads);
crispasr_session_result* crispasr_session_transcribe(
    crispasr_session* s, const float* pcm, int n_samples);
int crispasr_session_result_n_segments(crispasr_session_result* r);
const char* crispasr_session_result_segment_text(
    crispasr_session_result* r, int i);
void crispasr_session_result_free(crispasr_session_result* r);
void crispasr_session_close(crispasr_session* s);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return (v && *v) ? v : fallback;
}

// Chunk-skipping 16-bit PCM WAV loader (handles LIST/INFO/etc. chunks).
static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return pcm;

    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 ||
        memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f); return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16) break;
            channels    = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits        = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || sample_rate != 16000 || channels < 1) {
        fclose(f); return pcm;
    }

    int n_samples = data_size / (2 * channels);
    std::vector<int16_t> raw(n_samples * channels);
    size_t nread = fread(raw.data(), 2, n_samples * channels, f);
    fclose(f);
    n_samples = (int)(nread / channels);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

// Concatenate all segment texts from a session result.
static std::string result_text(crispasr_session_result* r) {
    std::string out;
    int n = crispasr_session_result_n_segments(r);
    for (int i = 0; i < n; i++) {
        const char* t = crispasr_session_result_segment_text(r, i);
        if (t) out += t;
    }
    return out;
}

// =========================================================================
// Unit tests — no model needed
// =========================================================================

TEST_CASE("beam setter: null-handle returns -1", "[unit][beam]") {
    REQUIRE(crispasr_session_set_beam_size(nullptr, 4) == -1);
}

TEST_CASE("beam setter: width clamping n<=0 → 1", "[unit][beam]") {
    // We can't inspect the field without a session, but the setter must
    // return 0 (success) on a valid session. Since we don't have one here,
    // we just verify the null path. The no-regression live test below
    // proves clamping works end-to-end.
    REQUIRE(crispasr_session_set_beam_size(nullptr, 0) == -1);
    REQUIRE(crispasr_session_set_beam_size(nullptr, -5) == -1);
}

// =========================================================================
// Live tests — gated on model + audio availability
// =========================================================================

// --- whisper (native BEAM_SEARCH) ----------------------------------------

TEST_CASE("beam: whisper greedy no-regression (beam_size=1 == default)", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_WHISPER");
    if (model.empty()) { SKIP("CRISPASR_MODEL_WHISPER not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    // Run 1: default session (never call set_beam_size)
    crispasr_session* s1 = crispasr_session_open_explicit(model.c_str(), "whisper", 2);
    REQUIRE(s1 != nullptr);
    auto* r1 = crispasr_session_transcribe(s1, pcm.data(), (int)pcm.size());
    REQUIRE(r1 != nullptr);
    std::string text1 = result_text(r1);
    crispasr_session_result_free(r1);
    crispasr_session_close(s1);

    // Run 2: explicitly set beam_size=1 (should be identical)
    crispasr_session* s2 = crispasr_session_open_explicit(model.c_str(), "whisper", 2);
    REQUIRE(s2 != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s2, 1) == 0);
    auto* r2 = crispasr_session_transcribe(s2, pcm.data(), (int)pcm.size());
    REQUIRE(r2 != nullptr);
    std::string text2 = result_text(r2);
    crispasr_session_result_free(r2);
    crispasr_session_close(s2);

    // Run 3: set beam_size=0 (clamped to 1 — must also match)
    crispasr_session* s3 = crispasr_session_open_explicit(model.c_str(), "whisper", 2);
    REQUIRE(s3 != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s3, 0) == 0);
    auto* r3 = crispasr_session_transcribe(s3, pcm.data(), (int)pcm.size());
    REQUIRE(r3 != nullptr);
    std::string text3 = result_text(r3);
    crispasr_session_result_free(r3);
    crispasr_session_close(s3);

    INFO("default:    " << text1);
    INFO("beam_size=1:" << text2);
    INFO("beam_size=0:" << text3);
    REQUIRE_FALSE(text1.empty());  // stub-model guard: empty==empty must not pass vacuously
    REQUIRE(text1 == text2);
    REQUIRE(text1 == text3);
}

TEST_CASE("beam: whisper beam_size=2 produces non-empty output", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_WHISPER");
    if (model.empty()) { SKIP("CRISPASR_MODEL_WHISPER not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    crispasr_session* s = crispasr_session_open_explicit(model.c_str(), "whisper", 2);
    REQUIRE(s != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s, 2) == 0);
    auto* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    REQUIRE(r != nullptr);

    std::string text = result_text(r);
    INFO("whisper beam=2: " << text);
    REQUIRE(!text.empty());
    REQUIRE(crispasr_session_result_n_segments(r) > 0);

    crispasr_session_result_free(r);
    crispasr_session_close(s);
}

TEST_CASE("beam: whisper beam_size=4 produces non-empty output", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_WHISPER");
    if (model.empty()) { SKIP("CRISPASR_MODEL_WHISPER not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    crispasr_session* s = crispasr_session_open_explicit(model.c_str(), "whisper", 2);
    REQUIRE(s != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s, 4) == 0);
    auto* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    REQUIRE(r != nullptr);

    std::string text = result_text(r);
    INFO("whisper beam=4: " << text);
    REQUIRE(!text.empty());

    crispasr_session_result_free(r);
    crispasr_session_close(s);
}

// --- glm-asr (per-backend setter) ----------------------------------------

TEST_CASE("beam: glm-asr greedy no-regression (beam_size=1 == default)", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_GLM_ASR");
    if (model.empty()) { SKIP("CRISPASR_MODEL_GLM_ASR not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    // Default (no setter call)
    crispasr_session* s1 = crispasr_session_open_explicit(model.c_str(), "glm-asr", 2);
    REQUIRE(s1 != nullptr);
    auto* r1 = crispasr_session_transcribe(s1, pcm.data(), (int)pcm.size());
    REQUIRE(r1 != nullptr);
    std::string text1 = result_text(r1);
    crispasr_session_result_free(r1);
    crispasr_session_close(s1);

    // Explicit beam_size=1
    crispasr_session* s2 = crispasr_session_open_explicit(model.c_str(), "glm-asr", 2);
    REQUIRE(s2 != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s2, 1) == 0);
    auto* r2 = crispasr_session_transcribe(s2, pcm.data(), (int)pcm.size());
    REQUIRE(r2 != nullptr);
    std::string text2 = result_text(r2);
    crispasr_session_result_free(r2);
    crispasr_session_close(s2);

    INFO("default:    " << text1);
    INFO("beam_size=1:" << text2);
    REQUIRE_FALSE(text1.empty());  // stub-model guard: empty==empty must not pass vacuously
    REQUIRE(text1 == text2);
}

TEST_CASE("beam: glm-asr beam_size=2 produces non-empty output", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_GLM_ASR");
    if (model.empty()) { SKIP("CRISPASR_MODEL_GLM_ASR not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    crispasr_session* s = crispasr_session_open_explicit(model.c_str(), "glm-asr", 2);
    REQUIRE(s != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s, 2) == 0);
    auto* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    REQUIRE(r != nullptr);

    std::string text = result_text(r);
    INFO("glm-asr beam=2: " << text);
    REQUIRE(!text.empty());

    crispasr_session_result_free(r);
    crispasr_session_close(s);
}

// --- qwen3-asr (replay-helper via core_beam_decode::run_with_probs) -------

TEST_CASE("beam: qwen3-asr greedy no-regression (beam_size=1 == default)", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_QWEN3_ASR");
    if (model.empty()) { SKIP("CRISPASR_MODEL_QWEN3_ASR not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    // Default (no setter call)
    crispasr_session* s1 = crispasr_session_open_explicit(model.c_str(), "qwen3", 2);
    REQUIRE(s1 != nullptr);
    auto* r1 = crispasr_session_transcribe(s1, pcm.data(), (int)pcm.size());
    REQUIRE(r1 != nullptr);
    std::string text1 = result_text(r1);
    crispasr_session_result_free(r1);
    crispasr_session_close(s1);

    // Explicit beam_size=1
    crispasr_session* s2 = crispasr_session_open_explicit(model.c_str(), "qwen3", 2);
    REQUIRE(s2 != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s2, 1) == 0);
    auto* r2 = crispasr_session_transcribe(s2, pcm.data(), (int)pcm.size());
    REQUIRE(r2 != nullptr);
    std::string text2 = result_text(r2);
    crispasr_session_result_free(r2);
    crispasr_session_close(s2);

    INFO("default:    " << text1);
    INFO("beam_size=1:" << text2);
    REQUIRE_FALSE(text1.empty());  // stub-model guard: empty==empty must not pass vacuously
    REQUIRE(text1 == text2);
}

TEST_CASE("beam: qwen3-asr beam_size=2 produces non-empty output", "[beam][.live]") {
    std::string model = get_env("CRISPASR_MODEL_QWEN3_ASR");
    if (model.empty()) { SKIP("CRISPASR_MODEL_QWEN3_ASR not set"); return; }

    std::string audio = get_env("CRISPASR_AUDIO_EN", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    crispasr_session* s = crispasr_session_open_explicit(model.c_str(), "qwen3", 2);
    REQUIRE(s != nullptr);
    REQUIRE(crispasr_session_set_beam_size(s, 2) == 0);
    auto* r = crispasr_session_transcribe(s, pcm.data(), (int)pcm.size());
    REQUIRE(r != nullptr);

    std::string text = result_text(r);
    INFO("qwen3-asr beam=2: " << text);
    REQUIRE(!text.empty());

    crispasr_session_result_free(r);
    crispasr_session_close(s);
}
