// MT3 integration test (§250) — multi-instrument note events.
//
// Requires CRISPASR_MODEL_MT3 pointing at mt3-f16.gguf; SKIPs cleanly when
// unset. Stage-level parity lives in the `crispasr-diff mt3` harness against
// tools/reference_backends/mt3.py (encoder cos 0.999999879, first-step logits
// argmax identical, end-to-end notes an exact positional match); this file
// covers the contract the harness does not: the C ABI's shape, the decode
// budget knob, and the invariants a caller can rely on.

#include <catch2/catch_test_macros.hpp>

#include "mt3.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const char* model_path() {
    const char* p = std::getenv("CRISPASR_MODEL_MT3");
    return (p && *p) ? p : nullptr;
}

// A short polyphonic tone cluster. MT3 is trained on real instruments, so the
// point is NOT that it transcribes a synthetic signal accurately — it is that
// the pipeline runs, segments, decodes and returns a well-formed result.
std::vector<float> tone_cluster(int sr, double seconds) {
    const int n = (int)(sr * seconds);
    std::vector<float> pcm((size_t)n, 0.0f);
    const double freqs[3] = {261.63, 329.63, 392.00}; // C4 E4 G4
    for (int i = 0; i < n; i++) {
        double v = 0.0;
        for (double f : freqs)
            v += std::sin(2.0 * M_PI * f * i / sr);
        double env = 1.0;
        if (i < 512)
            env = i / 512.0;
        if (n - i < 512)
            env = (n - i) / 512.0;
        pcm[(size_t)i] = (float)(0.2 * env * v);
    }
    return pcm;
}

} // namespace

TEST_CASE("mt3 init reports the expected rates", "[integration][mt3]") {
    const char* path = model_path();
    if (!path) {
        SKIP("CRISPASR_MODEL_MT3 not set");
    }
    auto params = mt3_default_params();
    mt3_context* ctx = mt3_init_from_file(path, params);
    REQUIRE(ctx != nullptr);
    CHECK(mt3_sample_rate(ctx) == 16000);
    // 2.048 s segments at 16 kHz = 32768 samples (256 mel frames at hop 128).
    CHECK(mt3_segment_samples(ctx) == 32768);
    mt3_free(ctx);
}

TEST_CASE("mt3 returns a well-formed multi-instrument result", "[integration][mt3]") {
    const char* path = model_path();
    if (!path) {
        SKIP("CRISPASR_MODEL_MT3 not set");
    }
    auto params = mt3_default_params();
    // Keep the test quick: the budget only truncates, never corrupts.
    params.max_decode_steps = 64;
    mt3_context* ctx = mt3_init_from_file(path, params);
    REQUIRE(ctx != nullptr);

    const int sr = (int)mt3_sample_rate(ctx);
    auto pcm = tone_cluster(sr, 4.5); // > 2.048 s, so segmentation runs

    mt3_result res{};
    REQUIRE(mt3_transcribe(ctx, pcm.data(), (int)pcm.size(), &res) == 0);

    // Multi-segment input must be segmented, and decode health must be
    // reported rather than silently swallowed.
    CHECK(res.n_segments >= 2);
    CHECK(res.n_tokens > 0);
    CHECK(res.n_invalid >= 0);
    CHECK(res.n_dropped >= 0);

    for (int i = 0; i < res.n_notes; i++) {
        const mt3_note_event& n = res.notes[i];
        CHECK(n.end_time > n.start_time); // no zero-length or inverted notes
        CHECK(n.pitch >= 0);
        CHECK(n.pitch <= 127);
        CHECK(n.velocity >= 0);
        CHECK(n.velocity <= 127);
        CHECK(n.program >= 0);
        CHECK(n.program <= 127);
        // assign_instruments: drums always land on track 9 (GM channel 10).
        if (n.is_drum)
            CHECK(n.instrument == 9);
        else
            CHECK(n.instrument != 9);
    }

    mt3_result_free(&res);
    mt3_free(ctx);
}

TEST_CASE("mt3 rejects null and empty input without crashing", "[integration][mt3]") {
    const char* path = model_path();
    if (!path) {
        SKIP("CRISPASR_MODEL_MT3 not set");
    }
    auto params = mt3_default_params();
    mt3_context* ctx = mt3_init_from_file(path, params);
    REQUIRE(ctx != nullptr);

    mt3_result res{};
    CHECK(mt3_transcribe(ctx, nullptr, 16000, &res) != 0);
    CHECK(mt3_transcribe(nullptr, nullptr, 0, &res) != 0);
    mt3_free(ctx);

    CHECK(mt3_init_from_file("/nonexistent/mt3.gguf", params) == nullptr);
}
