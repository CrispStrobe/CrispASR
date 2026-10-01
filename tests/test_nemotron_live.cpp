// tests/test_nemotron_live.cpp — integration tests for nemotron backend.
//
// Live tests: require model + audio files on disk. Gated behind [.live].
// Run: ctest -R test-nemotron --output-on-failure
//
// Env vars:
//   CRISPASR_MODEL_NEMOTRON    — Q4_K GGUF path
//   CRISPASR_MODEL_NEMOTRON_F16 — F16 GGUF path (optional)

#include <catch2/catch_test_macros.hpp>
#include "nemotron.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

struct scoped_env {
    std::string name;
    std::string old_value;
    bool had_value;
    scoped_env(const char* key, const char* value) : name(key), had_value(std::getenv(key) != nullptr) {
        if (had_value)
            old_value = std::getenv(key);
#ifdef _WIN32
        _putenv_s(key, value);
#else
        setenv(key, value, 1);
#endif
    }
    ~scoped_env() {
#ifdef _WIN32
        _putenv_s(name.c_str(), had_value ? old_value.c_str() : "");
#else
        if (had_value)
            setenv(name.c_str(), old_value.c_str(), 1);
        else
            unsetenv(name.c_str());
#endif
    }
};

// Chunk-skipping 16-bit PCM WAV loader.
static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return pcm;

    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4)
            break;
        if (fread(&chunk_size, 4, 1, f) != 1)
            break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }

    int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw(n_samples * channels);
    fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

TEST_CASE("nemotron: init and free", "[nemotron][.live]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    }

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    nemotron_free(ctx);
}

TEST_CASE("nemotron: JFK English transcription", "[nemotron][.live]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* text = nemotron_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string got(text);
    std::free(text);
    INFO("Transcription: " << got);

    // Check key phrases are present (case-insensitive substring match)
    auto contains_ci = [](const std::string& hay, const char* needle) {
        std::string h = hay, n = needle;
        for (auto& c : h)
            c = (char)tolower((unsigned char)c);
        for (auto& c : n)
            c = (char)tolower((unsigned char)c);
        return h.find(n) != std::string::npos;
    };

    CHECK(contains_ci(got, "fellow americans"));
    CHECK(contains_ci(got, "country"));
    CHECK(got.size() > 30);

    nemotron_free(ctx);
}

TEST_CASE("nemotron: persistent stream matches one-shot chunked decode", "[nemotron][.live][streaming]") {
    std::string model = get_env("CRISPASR_MODEL_NEMOTRON");
    if (model.empty())
        SKIP("CRISPASR_MODEL_NEMOTRON not set");
    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());
    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    nemotron_context* ctx = nemotron_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    nemotron_set_context_preset(ctx, 0);
    scoped_env streaming_env("CRISPASR_NEMOTRON_STREAMING", "1");
    scoped_env step_env("CRISPASR_NEMOTRON_STREAM_CHUNKS_PER_STEP", "4");
    char* expected_raw = nemotron_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(expected_raw != nullptr);
    std::string expected(expected_raw);
    std::free(expected_raw);

    nemotron_stream* stream = nemotron_stream_create(ctx);
    REQUIRE(stream != nullptr);
    std::string actual;
    int tokens_before_flush = 0;
    auto cb = [](int id, float, void* userdata) {
        auto* pair = static_cast<std::pair<nemotron_context*, std::string*>*>(userdata);
        std::string piece = nemotron_token_to_str(pair->first, id);
        size_t pos = 0;
        while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos++;
        }
        *pair->second += piece;
    };
    std::pair<nemotron_context*, std::string*> state{ctx, &actual};
    const int append_samples = 320 * 16;
    int previous_frames = 0;
    for (size_t offset = 0; offset < pcm.size(); offset += append_samples) {
        int count = (int)std::min((size_t)append_samples, pcm.size() - offset);
        REQUIRE(nemotron_stream_append(stream, pcm.data() + offset, count, false, cb, &state));
        const int current_frames = nemotron_stream_processed_frames(stream);
        REQUIRE(current_frames >= previous_frames);
        // Four native chunks per step (pinned above). Growth is bounded; a
        // growing-prefix implementation would jump by total T.
        REQUIRE(current_frames - previous_frames <= 16);
        previous_frames = current_frames;
        if (offset + count < pcm.size())
            tokens_before_flush = (int)actual.size();
    }
    REQUIRE(tokens_before_flush > 0);
    REQUIRE(nemotron_stream_append(stream, nullptr, 0, true, cb, &state));
    while (!actual.empty() && actual.front() == ' ')
        actual.erase(actual.begin());
    INFO("one-shot: " << expected);
    INFO("streamed: " << actual);
    CHECK(actual == expected);

    nemotron_stream_reset(stream);
    CHECK(nemotron_stream_processed_frames(stream) == 0);
    nemotron_stream_free(stream);
    nemotron_free(ctx);
}

TEST_CASE("nemotron: F16 produces same text as Q4_K", "[nemotron][.live]") {
    std::string model_q4k = get_env("CRISPASR_MODEL_NEMOTRON");
    std::string model_f16 = get_env("CRISPASR_MODEL_NEMOTRON_F16");
    if (model_q4k.empty() || model_f16.empty()) {
        SKIP("CRISPASR_MODEL_NEMOTRON or CRISPASR_MODEL_NEMOTRON_F16 not set");
    }

    auto pcm = load_wav_16k_mono("samples/jfk.wav");
    REQUIRE(!pcm.empty());

    nemotron_context_params cp = nemotron_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;

    // Q4_K
    nemotron_context* ctx_q = nemotron_init_from_file(model_q4k.c_str(), cp);
    REQUIRE(ctx_q != nullptr);
    char* t_q = nemotron_transcribe(ctx_q, pcm.data(), (int)pcm.size());
    REQUIRE(t_q != nullptr);
    std::string text_q(t_q);
    std::free(t_q);
    nemotron_free(ctx_q);

    // F16
    nemotron_context* ctx_f = nemotron_init_from_file(model_f16.c_str(), cp);
    REQUIRE(ctx_f != nullptr);
    char* t_f = nemotron_transcribe(ctx_f, pcm.data(), (int)pcm.size());
    REQUIRE(t_f != nullptr);
    std::string text_f(t_f);
    std::free(t_f);
    nemotron_free(ctx_f);

    INFO("Q4_K: " << text_q);
    INFO("F16:  " << text_f);
    CHECK(text_q == text_f);
}
