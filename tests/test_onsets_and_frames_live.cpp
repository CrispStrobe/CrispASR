// Onsets & Frames integration test.
//
// Requires CRISPASR_MODEL_ONSETS_AND_FRAMES pointing at a GGUF produced by
// models/convert-onsets-and-frames-to-gguf.py. SKIPs cleanly when unset.
//
// What this covers is deliberately what the ONNX diff harness cannot see
// (crispasr-crispembed-dev.md rule 3b): the C ABI contract, the note-event
// invariants, and the two properties a head-wiring typo would break while
// leaving every activation numerically plausible.

#include <catch2/catch_test_macros.hpp>

#include "onsets_and_frames.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* model_path() {
    const char* p = std::getenv("CRISPASR_MODEL_ONSETS_AND_FRAMES");
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

TEST_CASE("onsets-and-frames init and sample rate", "[integration][onsets-and-frames]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_ONSETS_AND_FRAMES not set");

    onsets_and_frames_params p = onsets_and_frames_default_params();
    p.verbosity = 0;
    onsets_and_frames_ctx* ctx = onsets_and_frames_init_from_file(path, p);
    REQUIRE(ctx != nullptr);
    CHECK(onsets_and_frames_sample_rate(ctx) == 16000u);
    onsets_and_frames_free(ctx);
}

TEST_CASE("onsets-and-frames mel geometry", "[integration][onsets-and-frames]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_ONSETS_AND_FRAMES not set");

    onsets_and_frames_params p = onsets_and_frames_default_params();
    p.verbosity = 0;
    onsets_and_frames_ctx* ctx = onsets_and_frames_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    const auto pcm = plucked_a4();
    int frames = 0;
    float* mel = onsets_and_frames_mel(ctx, pcm.data(), (int)pcm.size(), &frames);
    REQUIRE(mel != nullptr);

    // The reference forward drops the last sample before the STFT, so
    // T = (n - 1) / hop + 1. A one-frame disagreement is a 32 ms timing error
    // on every note and nothing else in the pipeline would complain about it.
    CHECK(frames == (int)((pcm.size() - 1) / 512 + 1));

    // log(clamp(mel, 1e-5)) floors at ln(1e-5) = -11.5129; nothing may sit
    // below it, and a silent-everywhere mel means the filterbank is wrong.
    bool any_above_floor = false;
    for (int i = 0; i < frames * 229; i++) {
        REQUIRE(mel[i] >= -11.52f);
        REQUIRE(std::isfinite(mel[i]));
        if (mel[i] > -11.0f)
            any_above_floor = true;
    }
    CHECK(any_above_floor);
    std::free(mel);
    onsets_and_frames_free(ctx);
}

TEST_CASE("onsets-and-frames note events are well formed", "[integration][onsets-and-frames]") {
    const char* path = model_path();
    if (!path)
        SKIP("CRISPASR_MODEL_ONSETS_AND_FRAMES not set");

    onsets_and_frames_params p = onsets_and_frames_default_params();
    p.verbosity = 2; // keep the raw heads
    onsets_and_frames_ctx* ctx = onsets_and_frames_init_from_file(path, p);
    REQUIRE(ctx != nullptr);

    const auto pcm = plucked_a4();
    onsets_and_frames_result res{};
    REQUIRE(onsets_and_frames_transcribe(ctx, pcm.data(), (int)pcm.size(), &res) == 0);

    CHECK(res.n_classes == 88);
    CHECK(res.n_frames > 0);
    REQUIRE(res.onset_output != nullptr);
    REQUIRE(res.frame_output != nullptr);
    REQUIRE(res.activation_output != nullptr);

    // Every head is a sigmoid, so every value is a probability. A head wired to
    // the wrong graph output would still satisfy this — which is the point of
    // the next check.
    for (int i = 0; i < res.n_frames * res.n_classes; i++) {
        REQUIRE(res.onset_output[i] >= 0.0f);
        REQUIRE(res.onset_output[i] <= 1.0f);
        REQUIRE(res.frame_output[i] >= 0.0f);
        REQUIRE(res.frame_output[i] <= 1.0f);
    }

    // frame and activation are DIFFERENT heads: activation is frame_stack's
    // output and frame is the combined stack's. The ONNX export's names are
    // shifted by one, so a converter or a runtime that believed them would put
    // the same tensor in both slots. That costs 9.1% -> 5.6% F1 with offsets
    // required and is invisible in any per-value check.
    double diff = 0.0;
    for (int i = 0; i < res.n_frames * res.n_classes; i++)
        diff += std::fabs(res.frame_output[i] - res.activation_output[i]);
    CHECK(diff > 1e-3);

    for (int i = 0; i < res.n_notes; i++) {
        const onsets_and_frames_note_event& e = res.note_events[i];
        CHECK(e.offset_time > e.onset_time);
        CHECK(e.midi_note >= 21);
        CHECK(e.midi_note <= 108);
        CHECK(e.velocity >= 0);
        CHECK(e.velocity <= 127);
        if (i > 0)
            CHECK(e.onset_time >= res.note_events[i - 1].onset_time);
    }

    onsets_and_frames_result_free(&res);
    CHECK(res.note_events == nullptr);
    CHECK(res.n_notes == 0);
    onsets_and_frames_free(ctx);
}
