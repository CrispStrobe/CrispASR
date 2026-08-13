// MOSS-Audio-4B integration test — exercises ASR transcription and
// audio-understanding (custom prompt).
//
// Requires CRISPASR_MODEL_MOSS_AUDIO env var pointing to the GGUF.
// SKIPs cleanly when not set.

#include <catch2/catch_test_macros.hpp>

#include "moss_audio.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef SAMPLES_DIR
#define SAMPLES_DIR "samples"
#endif

static std::vector<float> load_wav_16k(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f) - 44;
    fseek(f, 44, SEEK_SET);
    std::vector<int16_t> raw(sz / 2);
    size_t n = fread(raw.data(), 2, raw.size(), f);
    (void)n;
    fclose(f);
    std::vector<float> pcm(raw.size());
    for (size_t i = 0; i < raw.size(); i++)
        pcm[i] = raw[i] / 32768.0f;
    return pcm;
}

TEST_CASE("moss-audio transcribe JFK", "[integration][moss-audio]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MOSS_AUDIO");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MOSS_AUDIO not set");
    }

    auto params = moss_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = moss_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k(SAMPLES_DIR "/jfk.wav");
    REQUIRE(!pcm.empty());

    char* text = moss_audio_transcribe(ctx, pcm.data(), (int)pcm.size());
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    INFO("transcript: " << result);
    CHECK(result.find("country") != std::string::npos);
    CHECK(result.size() > 20);

    moss_audio_free(ctx);
}

TEST_CASE("moss-audio custom prompt", "[integration][moss-audio]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MOSS_AUDIO");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MOSS_AUDIO not set");
    }

    auto params = moss_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = moss_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k(SAMPLES_DIR "/jfk.wav");
    REQUIRE(!pcm.empty());

    // Audio understanding: ask a question about the audio
    char* text = moss_audio_process(ctx, pcm.data(), (int)pcm.size(), "What language is spoken in this audio?");
    REQUIRE(text != nullptr);

    std::string result(text);
    free(text);

    INFO("response: " << result);
    // Should mention English in some form
    CHECK(result.size() > 5);

    moss_audio_free(ctx);
}

TEST_CASE("moss-audio run_encoder_meta valid-frame metadata (issue #344)", "[integration][moss-audio]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_MOSS_AUDIO");
    if (!model_path || !*model_path) {
        SKIP("CRISPASR_MODEL_MOSS_AUDIO not set");
    }

    auto params = moss_audio_context_default_params();
    params.verbosity = 0;
    auto* ctx = moss_audio_init_from_file(model_path, params);
    REQUIRE(ctx != nullptr);

    auto pcm = load_wav_16k(SAMPLES_DIR "/jfk.wav");
    REQUIRE(!pcm.empty());

    // 1. mel with truthful pre-pad length
    int n_mels = 0, T_mel = 0, T_mel_actual = 0;
    float* mel = moss_audio_compute_mel_meta(ctx, pcm.data(), (int)pcm.size(), &n_mels, &T_mel, &T_mel_actual);
    REQUIRE(mel != nullptr);
    REQUIRE(n_mels == 128);
    REQUIRE(T_mel_actual > 0);
    // jfk.wav is ~10s, so the pre-pad length must be strictly below the pad.
    REQUIRE(T_mel_actual < 3000);
    REQUIRE(T_mel == 3000);

    // 2. metadata entrypoint: content-only frames + per-chunk valid counts
    std::vector<int> valid_counts((size_t)((T_mel_actual + 399) / 400), -1);
    int T_enc = 0, d_enc = 0, num_chunks = -1, echo = -1, total_valid = -1;
    float *ds0 = nullptr, *ds1 = nullptr, *ds2 = nullptr;
    float* enc = moss_audio_run_encoder_meta(ctx, mel, n_mels, T_mel, T_mel_actual, &T_enc, &d_enc, valid_counts.data(),
                                             &num_chunks, &echo, &total_valid, &ds0, &ds1, &ds2);
    REQUIRE(enc != nullptr);
    REQUIRE(d_enc == 1280);
    REQUIRE(echo == T_mel_actual);
    REQUIRE(num_chunks == (int)((T_mel_actual + 399) / 400));

    int sum_valid = 0;
    for (int c = 0; c < num_chunks; c++) {
        REQUIRE(valid_counts[c] > 0);
        sum_valid += valid_counts[c];
    }
    // core invariant: sum(per-chunk valid) == out_total_valid == out_T_enc
    REQUIRE(sum_valid == total_valid);
    REQUIRE(total_valid == T_enc);

    // taps are content-only too
    REQUIRE(ds0 != nullptr);
    REQUIRE(ds1 != nullptr);
    REQUIRE(ds2 != nullptr);

    // 3. differential vs the reference-faithful entrypoint: the padded path
    //    must expose MORE frames (pad chunks), never fewer.
    int T_enc_ref = 0, d_ref = 0;
    float* enc_ref = moss_audio_run_encoder(ctx, mel, n_mels, T_mel, &T_enc_ref, &d_ref, nullptr, nullptr, nullptr);
    REQUIRE(enc_ref != nullptr);
    REQUIRE(T_enc_ref > T_enc); // pad-only frames excluded from the meta path

    // 4. differential: no-pad input (T_mel_actual == T_mel) must be
    //    byte-identical to moss_audio_run_encoder.
    int T_enc_full = 0, d_full = 0, num_full = -1, total_full = -1;
    float* enc_full = moss_audio_run_encoder_meta(ctx, mel, n_mels, T_mel, T_mel, &T_enc_full, &d_full, nullptr,
                                                  &num_full, &echo, &total_full, nullptr, nullptr, nullptr);
    REQUIRE(enc_full != nullptr);
    REQUIRE(T_enc_full == T_enc_ref);
    REQUIRE(total_full == T_enc_full);
    REQUIRE(num_full == (int)((T_mel + 399) / 400));
    REQUIRE(memcmp(enc_full, enc_ref, (size_t)T_enc_full * d_full * sizeof(float)) == 0);

    // 5. adapter dims report from the GGUF (2560 for MOSS-Audio-4B) and stay
    //    row-consistent with the meta output.
    int adapt_T = 0, adapt_d = 0;
    float* adapted = moss_audio_run_adapter(ctx, enc, T_enc, d_enc, &adapt_T, &adapt_d);
    REQUIRE(adapted != nullptr);
    REQUIRE(adapt_T == T_enc);
    REQUIRE(adapt_d > 0);

    // 6. fail closed on out-of-range T_mel_actual (never inferred here)
    REQUIRE(moss_audio_run_encoder_meta(ctx, mel, n_mels, T_mel, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr) == nullptr);
    REQUIRE(moss_audio_run_encoder_meta(ctx, mel, n_mels, T_mel, T_mel + 1, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, nullptr, nullptr) == nullptr);

    free(mel);
    free(enc);
    free(enc_ref);
    free(enc_full);
    free(adapted);
    free(ds0);
    free(ds1);
    free(ds2);
    moss_audio_free(ctx);
}
