// tests/test_paraformer.cpp — integration tests for paraformer backend.
//
// Live tests: require model + audio files on disk. Gated behind [.live].
// Run: ctest -R test-paraformer --output-on-failure
//
// Env vars:
//   PARAFORMER_MODEL     — GGUF path (default: /mnt/storage/paraformer-zh/paraformer-zh-f16.gguf)
//   PARAFORMER_MODEL_Q4K — Q4_K GGUF path (default: /mnt/storage/paraformer-zh/paraformer-zh-q4_k.gguf)

#include <catch2/catch_test_macros.hpp>
#include "paraformer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// Chunk-skipping 16-bit PCM WAV loader (handles LIST/INFO/etc. chunks).
static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return pcm;

    // Read RIFF header (12 bytes)
    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 ||
        memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f); return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    // Walk chunks
    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16) break;
            channels = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            // Skip remaining fmt bytes
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            // Skip unknown chunk
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

TEST_CASE("paraformer: init and free", "[paraformer][.live]") {
    std::string model = get_env("PARAFORMER_MODEL",
        "/mnt/storage/paraformer-zh/paraformer-zh-f16.gguf");

    paraformer_context_params cp = paraformer_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    paraformer_context* ctx = paraformer_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);
    paraformer_free(ctx);
}

TEST_CASE("paraformer: Chinese transcription matches reference", "[paraformer][.live]") {
    std::string model = get_env("PARAFORMER_MODEL",
        "/mnt/storage/paraformer-zh/paraformer-zh-f16.gguf");
    std::string audio = get_env("PARAFORMER_AUDIO_ZH",
        "/mnt/storage/paraformer-zh-upstream/example/asr_example.wav");

    auto pcm = load_wav_16k_mono(audio);
    REQUIRE(!pcm.empty());

    paraformer_context_params cp = paraformer_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    paraformer_context* ctx = paraformer_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* result = paraformer_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(result != nullptr);

    std::string got(result);
    std::free(result);
    paraformer_free(ctx);

    std::string expected =
        "\xe6\xad\xa3\xe6\x98\xaf\xe5\x9b\xa0\xe4\xb8\xba\xe5\xad\x98\xe5\x9c\xa8"
        "\xe7\xbb\x9d\xe5\xaf\xb9\xe6\xad\xa3\xe4\xb9\x89\xe6\x89\x80\xe4\xbb\xa5"
        "\xe6\x88\x91\xe4\xbb\xac\xe6\x8e\xa5\xe5\x8f\x97\xe7\x8e\xb0\xe5\xae\x9e"
        "\xe7\x9a\x84\xe7\x9b\xb8\xe5\xaf\xb9\xe6\xad\xa3\xe4\xb9\x89\xe4\xbd\x86"
        "\xe6\x98\xaf\xe4\xb8\x8d\xe8\xa6\x81\xe5\x9b\xa0\xe4\xb8\xba\xe7\x8e\xb0"
        "\xe5\xae\x9e\xe7\x9a\x84\xe7\x9b\xb8\xe5\xaf\xb9\xe6\xad\xa3\xe4\xb9\x89"
        "\xe6\x88\x91\xe4\xbb\xac\xe5\xb0\xb1\xe8\xae\xa4\xe4\xb8\xba\xe8\xbf\x99"
        "\xe4\xb8\xaa\xe4\xb8\x96\xe7\x95\x8c\xe6\xb2\xa1\xe6\x9c\x89\xe6\xad\xa3"
        "\xe4\xb9\x89\xe5\x9b\xa0\xe4\xb8\xba\xe5\xa6\x82\xe6\x9e\x9c\xe5\xbd\x93"
        "\xe4\xbd\xa0\xe8\xae\xa4\xe4\xb8\xba\xe8\xbf\x99\xe4\xb8\xaa\xe4\xb8\x96"
        "\xe7\x95\x8c\xe6\xb2\xa1\xe6\x9c\x89\xe6\xad\xa3\xe4\xb9\x89";
    // = 正是因为存在绝对正义所以我们接受现实的相对正义但是不要因为现实的相对正义
    //   我们就认为这个世界没有正义因为如果当你认为这个世界没有正义

    REQUIRE(got == expected);
}

TEST_CASE("paraformer: English JFK matches reference", "[paraformer][.live]") {
    std::string model = get_env("PARAFORMER_MODEL",
        "/mnt/storage/paraformer-zh/paraformer-zh-f16.gguf");
    std::string audio = get_env("PARAFORMER_AUDIO_EN",
        "/mnt/akademie_storage/whisper.cpp/samples/jfk.wav");

    auto pcm = load_wav_16k_mono(audio);
    if (pcm.empty()) {
        WARN("JFK audio not found, skipping");
        return;
    }

    paraformer_context_params cp = paraformer_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    paraformer_context* ctx = paraformer_init_from_file(model.c_str(), cp);
    REQUIRE(ctx != nullptr);

    char* result = paraformer_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(result != nullptr);

    std::string got(result);
    std::free(result);
    paraformer_free(ctx);

    std::string expected = "and so my fellow americans ask not what your country "
                           "can do for you ask what you can do for your country";
    REQUIRE(got == expected);
}

TEST_CASE("paraformer: Q4_K transcript matches F16", "[paraformer][.live]") {
    std::string model = get_env("PARAFORMER_MODEL_Q4K",
        "/mnt/storage/paraformer-zh/paraformer-zh-q4_k.gguf");
    std::string audio = get_env("PARAFORMER_AUDIO_ZH",
        "/mnt/storage/paraformer-zh-upstream/example/asr_example.wav");

    auto pcm = load_wav_16k_mono(audio);
    if (pcm.empty()) { WARN("Audio not found"); return; }

    paraformer_context_params cp = paraformer_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 0;
    paraformer_context* ctx = paraformer_init_from_file(model.c_str(), cp);
    if (!ctx) { WARN("Q4_K model not found"); return; }

    char* result = paraformer_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(result != nullptr);

    std::string got(result);
    std::free(result);
    paraformer_free(ctx);

    // Q4_K produces byte-identical transcript to F16 on this clip
    std::string expected =
        "\xe6\xad\xa3\xe6\x98\xaf\xe5\x9b\xa0\xe4\xb8\xba\xe5\xad\x98\xe5\x9c\xa8"
        "\xe7\xbb\x9d\xe5\xaf\xb9\xe6\xad\xa3\xe4\xb9\x89\xe6\x89\x80\xe4\xbb\xa5"
        "\xe6\x88\x91\xe4\xbb\xac\xe6\x8e\xa5\xe5\x8f\x97\xe7\x8e\xb0\xe5\xae\x9e"
        "\xe7\x9a\x84\xe7\x9b\xb8\xe5\xaf\xb9\xe6\xad\xa3\xe4\xb9\x89\xe4\xbd\x86"
        "\xe6\x98\xaf\xe4\xb8\x8d\xe8\xa6\x81\xe5\x9b\xa0\xe4\xb8\xba\xe7\x8e\xb0"
        "\xe5\xae\x9e\xe7\x9a\x84\xe7\x9b\xb8\xe5\xaf\xb9\xe6\xad\xa3\xe4\xb9\x89"
        "\xe6\x88\x91\xe4\xbb\xac\xe5\xb0\xb1\xe8\xae\xa4\xe4\xb8\xba\xe8\xbf\x99"
        "\xe4\xb8\xaa\xe4\xb8\x96\xe7\x95\x8c\xe6\xb2\xa1\xe6\x9c\x89\xe6\xad\xa3"
        "\xe4\xb9\x89\xe5\x9b\xa0\xe4\xb8\xba\xe5\xa6\x82\xe6\x9e\x9c\xe5\xbd\x93"
        "\xe4\xbd\xa0\xe8\xae\xa4\xe4\xb8\xba\xe8\xbf\x99\xe4\xb8\xaa\xe4\xb8\x96"
        "\xe7\x95\x8c\xe6\xb2\xa1\xe6\x9c\x89\xe6\xad\xa3\xe4\xb9\x89";
    REQUIRE(got == expected);
}
