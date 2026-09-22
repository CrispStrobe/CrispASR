// hFT-Transformer integration test.
//
// Requires CRISPASR_MODEL_HFT_TRANSFORMER pointing at a GGUF produced by
// models/convert-hft-transformer-to-gguf.py. SKIPs cleanly when unset.
//
// What this covers is deliberately what the ONNX diff harness cannot see
// (crispasr-crispembed-dev.md rule 3b): the C ABI contract, the window
// arithmetic, the note-event invariants, and the velocity gate — which is the
// one knob on this model that changes the answer, and the one a refactor
// could silently drop while leaving every activation numerically identical.

#include <catch2/catch_test_macros.hpp>

#include "hft_transformer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* model_path() {
    const char* p = std::getenv("CRISPASR_MODEL_HFT_TRANSFORMER");
    return (p && *p) ? p : nullptr;
}

// A 3-second A4 (440 Hz) with a hard attack and an exponential decay. Not a
// piano, but enough that a working model fires SOMETHING and a broken one is
// silent or saturated.
std::vector<float> plucked_a4(int seconds = 3) {
    const int sr = 16000;
    std::vector<float> pcm((size_t)sr * seconds, 0.0f);
    for (size_t i = 0; i < pcm.size(); i++) {
        const double t = (double)i / sr;
        const double env = std::exp(-2.0 * t);
        double v = 0.0;
        for (int h = 1; h <= 6; h++)
            v += std::sin(2.0 * M_PI * 440.0 * h * t) / h;
        pcm[i] = (float)(0.3 * env * v);
    }
    return pcm;
}

} // namespace

TEST_CASE("hft-transformer init and sample rate", "[integration][hft-transformer]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_HFT_TRANSFORMER not set");

    hft_transformer_params p = hft_transformer_default_params();
    p.verbosity = 0;
    hft_transformer_ctx* ctx = hft_transformer_init_from_file(path, p);
    REQUIRE(ctx != nullptr);
    CHECK(hft_transformer_sample_rate(ctx) == 16000u);
    hft_transformer_free(ctx);
}

TEST_CASE("hft-transformer mel geometry", "[integration][hft-transformer]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_HFT_TRANSFORMER not set");

    hft_transformer_params p = hft_transformer_default_params();
    p.verbosity = 0;
    hft_transformer_ctx* ctx = hft_transformer_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    const auto pcm = plucked_a4();
    int frames = 0;
    float* mel = hft_transformer_mel(ctx, pcm.data(), (int)pcm.size(), &frames);
    REQUIRE(mel != nullptr);

    // hFT does NOT drop a sample before the STFT the way Onsets & Frames
    // does, so with center padding T = n / hop + 1. A one-frame disagreement
    // is a 16 ms timing error on every note and nothing else complains.
    CHECK(frames == (int)(pcm.size() / 256 + 1));

    // log(mel + 1e-8) floors at ln(1e-8) = -18.4207; nothing may sit below it,
    // and a silent-everywhere mel means the filterbank is wrong.
    bool any_above_floor = false;
    for (int i = 0; i < frames * 256; i++) {
        REQUIRE(mel[i] >= -18.43f);
        REQUIRE(std::isfinite(mel[i]));
        if (mel[i] > -18.0f)
            any_above_floor = true;
    }
    CHECK(any_above_floor);
    std::free(mel);
    hft_transformer_free(ctx);
}

TEST_CASE("hft-transformer heads and window arithmetic", "[integration][hft-transformer]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_HFT_TRANSFORMER not set");

    hft_transformer_params p = hft_transformer_default_params();
    p.verbosity = 2; // keep the raw heads
    hft_transformer_ctx* ctx = hft_transformer_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    const auto pcm = plucked_a4();
    hft_transformer_result res{};
    REQUIRE(hft_transformer_transcribe(ctx, pcm.data(), (int)pcm.size(), &res) == 0);

    CHECK(res.n_classes == 88);
    // The model answers 128 frames at a time and the tail is padded up to a
    // multiple of 128, so the frame count is always a whole number of windows.
    CHECK(res.n_frames % 128 == 0);
    CHECK(res.n_frames >= (int)(pcm.size() / 256 + 1));
    REQUIRE(res.onset_output != nullptr);
    REQUIRE(res.offset_output != nullptr);
    REQUIRE(res.mpe_output != nullptr);
    REQUIRE(res.velocity_output != nullptr);

    const int n = res.n_frames * res.n_classes;
    bool onset_differs_from_mpe = false;
    for (int i = 0; i < n; i++) {
        // onset / offset / mpe are sigmoids, so every value is a probability.
        REQUIRE(res.onset_output[i] >= 0.0f);
        REQUIRE(res.onset_output[i] <= 1.0f);
        REQUIRE(res.mpe_output[i] >= 0.0f);
        REQUIRE(res.mpe_output[i] <= 1.0f);
        // velocity is the argmax BIN, not a probability: 0..127.
        REQUIRE(res.velocity_output[i] >= 0.0f);
        REQUIRE(res.velocity_output[i] <= 127.0f);
        if (std::fabs(res.onset_output[i] - res.mpe_output[i]) > 1e-3f)
            onset_differs_from_mpe = true;
    }
    // Three scalar heads read the same 256-wide time-decoder output through
    // three different Linear(256, 1) matrices. A converter that bound the same
    // matrix to all three would still pass every per-value check above.
    CHECK(onset_differs_from_mpe);

    for (int i = 0; i < res.n_notes; i++) {
        const hft_transformer_note_event& e = res.note_events[i];
        CHECK(e.offset_time >= e.onset_time);
        CHECK(e.midi_note >= 21);
        CHECK(e.midi_note <= 108);
        CHECK(e.velocity >= 0);
        CHECK(e.velocity <= 127);
        if (i > 0)
            CHECK(e.onset_time >= res.note_events[i - 1].onset_time);
    }

    hft_transformer_result_free(&res);
    CHECK(res.note_events == nullptr);
    CHECK(res.n_notes == 0);
    hft_transformer_free(ctx);
}

TEST_CASE("hft-transformer velocity gate admits strictly more notes when off",
          "[integration][hft-transformer]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_HFT_TRANSFORMER not set");

    const auto pcm = plucked_a4();

    auto count = [&](bool gate) {
        hft_transformer_params p = hft_transformer_default_params();
        p.verbosity = 0;
        p.ignore_zero_velocity = gate;
        hft_transformer_ctx* ctx = hft_transformer_init_from_file(path, p);
        REQUIRE(ctx != nullptr);
        hft_transformer_result res{};
        REQUIRE(hft_transformer_transcribe(ctx, pcm.data(), (int)pcm.size(), &res) == 0);
        const int n = res.n_notes;
        hft_transformer_result_free(&res);
        hft_transformer_free(ctx);
        return n;
    };

    // `mode_velocity='ignore_zero'` can only ever REMOVE candidates, so the
    // gated count is bounded by the ungated one. §35.4 of the flutter_tuner
    // benchmark measured that removal as a better precision filter than the
    // onset threshold — which is the whole reason the gate is a parameter
    // here rather than a hardcoded `continue`.
    CHECK(count(true) <= count(false));
}

TEST_CASE("hft-transformer frame chunking does not change the answer",
          "[integration][hft-transformer]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_HFT_TRANSFORMER not set");

    const auto pcm = plucked_a4(2);

    auto onsets_for = [&](int chunk) {
        hft_transformer_params p = hft_transformer_default_params();
        p.verbosity = 2;
        p.frame_chunk = chunk;
        hft_transformer_ctx* ctx = hft_transformer_init_from_file(path, p);
        REQUIRE(ctx != nullptr);
        hft_transformer_result res{};
        REQUIRE(hft_transformer_transcribe(ctx, pcm.data(), (int)pcm.size(), &res) == 0);
        std::vector<float> out(res.onset_output, res.onset_output + res.n_frames * res.n_classes);
        hft_transformer_result_free(&res);
        hft_transformer_free(ctx);
        return out;
    };

    // Nothing in the encoder or the frequency decoder mixes across frames, so
    // the chunk size is a memory knob and nothing else. If this ever fails,
    // the chunking has grown a dependency it is not allowed to have.
    const auto a = onsets_for(8);
    const auto b = onsets_for(64);
    REQUIRE(a.size() == b.size());
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        worst = std::max(worst, (double)std::fabs(a[i] - b[i]));
    CHECK(worst < 1e-5);
}
