// CREPE live integration test — model load + monophonic F0 estimation.
//
// Requires CRISPASR_MODEL_CREPE to point at a CREPE GGUF (see
// cstr/crepe-GGUF: crepe-{tiny,full}-{f16,q8_0,q4_k}.gguf). Skips cleanly
// when the model is not available, so the suite stays green without models.
//
// This is the ctest-registered gate. It is NOT a replacement for
// tests/test_crepe_parity.cpp — that one is a plain main() that dumps raw
// activations for tools/crepe_numpy_parity.py to score against torchcrepe
// (cos = 1.0 acceptance). This test needs no Python and no reference
// fixture: it asserts the properties that fall out of the geometry
// (frame count, activation range, decoded F0 on a synthetic tone) so a
// regression in the ggml graph fails ctest rather than waiting for a
// manual parity run.

#include <catch2/catch_test_macros.hpp>

#include "crepe.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

// A pure tone at `hz`, `secs` long, at CREPE's 16 kHz. Synthetic tones are
// exactly what CREPE is unambiguous on — the octave errors the model makes
// live on real archival speech, not on this (see PLAN.md).
std::vector<float> tone(double hz, double secs) {
    const int n = (int)(CREPE_SAMPLE_RATE * secs);
    std::vector<float> pcm((size_t)n);
    for (int i = 0; i < n; i++)
        pcm[i] = (float)(0.5 * std::sin(2.0 * M_PI * hz * i / CREPE_SAMPLE_RATE));
    return pcm;
}

// Median of the F0 over the frames whose confidence clears `min_conf`.
// Median, not mean, so the edge frames (half-window of zero padding, hence
// low confidence and arbitrary pitch) cannot drag the estimate.
double median_f0(const std::vector<crepe_frame>& frames, float min_conf) {
    std::vector<double> hz;
    for (const auto& f : frames)
        if (f.voiced_prob >= min_conf)
            hz.push_back(f.f0_hz);
    if (hz.empty())
        return 0.0;
    std::sort(hz.begin(), hz.end());
    return hz[hz.size() / 2];
}

} // namespace

TEST_CASE("crepe pitch estimation", "[integration][crepe]") {
    const char* model_path = std::getenv("CRISPASR_MODEL_CREPE");
    if (!model_path || !*model_path)
        SKIP("CRISPASR_MODEL_CREPE not set");

    crepe_context* ctx = crepe_init(model_path, 4);
    if (!ctx)
        SKIP("CRISPASR_MODEL_CREPE set but the model failed to load");

    const std::string capacity = crepe_capacity(ctx);
    CHECK((capacity == "tiny" || capacity == "full"));

    const double secs = 1.0;
    const float hop_ms = 10.0f;
    const auto pcm = tone(440.0, secs);

    SECTION("frame count follows the documented geometry") {
        // crepe_n_frames is 1 + n_samples/hop (torchcrepe pad=True), so it
        // scales linearly with the input. Checking two lengths catches an
        // off-by-a-batch in the kBatch=64 tail handling, which a single
        // length would not.
        const int hop = (int)std::lround(CREPE_SAMPLE_RATE * hop_ms / 1000.0);
        CHECK(crepe_n_frames(ctx, (int)pcm.size(), hop_ms) == 1 + (int)pcm.size() / hop);
        CHECK(crepe_n_frames(ctx, 3 * (int)pcm.size(), hop_ms) == 1 + 3 * (int)pcm.size() / hop);
        CHECK(crepe_n_frames(ctx, 0, hop_ms) == 0);
    }

    SECTION("raw activation is a well-formed probability grid") {
        const int max_frames = crepe_n_frames(ctx, (int)pcm.size(), hop_ms);
        REQUIRE(max_frames > 0);
        std::vector<float> act((size_t)max_frames * CREPE_PITCH_BINS);
        const int got = crepe_compute_activation(ctx, pcm.data(), (int)pcm.size(), hop_ms, act.data(), max_frames);
        REQUIRE(got == max_frames);

        // Sigmoid output: finite and inside [0, 1] everywhere. A graph that
        // reads uninitialized scratch (the gallocr input-reuse trap) shows up
        // here as NaN long before it shows up as a wrong pitch.
        for (int i = 0; i < got * CREPE_PITCH_BINS; i++) {
            REQUIRE(std::isfinite(act[(size_t)i]));
            REQUIRE(act[(size_t)i] >= 0.0f);
            REQUIRE(act[(size_t)i] <= 1.0f);
        }

        // A 440 Hz tone must light up ONE bin region confidently. cents =
        // 20*bin + 1997.3794084376191, f = 10 * 2^(cents/1200).
        int best = 0;
        for (int k = 1; k < CREPE_PITCH_BINS; k++)
            if (act[(size_t)(got / 2) * CREPE_PITCH_BINS + k] > act[(size_t)(got / 2) * CREPE_PITCH_BINS + best])
                best = k;
        const double cents = 20.0 * best + 1997.3794084376191;
        const double hz = 10.0 * std::pow(2.0, cents / 1200.0);
        CHECK(std::fabs(hz - 440.0) < 20.0);
    }

    SECTION("decoded F0 tracks a synthetic tone") {
        const int max_frames = crepe_n_frames(ctx, (int)pcm.size(), hop_ms);
        std::vector<crepe_frame> frames((size_t)max_frames);
        const int nf = crepe_compute_f0(ctx, pcm.data(), (int)pcm.size(), hop_ms, frames.data(), max_frames);
        REQUIRE(nf == max_frames);

        for (const auto& f : frames) {
            CHECK(std::isfinite(f.f0_hz));
            CHECK(f.voiced_prob >= 0.0f);
            CHECK(f.voiced_prob <= 1.0f);
        }
        // Frame i is centred on sample i*hop, so time_ms is monotone with a
        // constant hop_ms step.
        CHECK(frames[0].time_ms == 0.0f);
        CHECK(std::fabs(frames[1].time_ms - frames[0].time_ms - hop_ms) < 1e-3f);

        const double f0 = median_f0(frames, 0.5f);
        REQUIRE(f0 > 0.0);
        // Within a quarter-tone (~1.5 %) of 440 Hz. Deliberately tighter than
        // an octave: an octave error is exactly the failure this backend is
        // meant to catch, so a ±50 % band would assert nothing.
        CHECK(std::fabs(f0 - 440.0) / 440.0 < 0.02);
    }

    SECTION("bad arguments are rejected, not crashed on") {
        std::vector<crepe_frame> out(8);
        CHECK(crepe_compute_f0(ctx, pcm.data(), 0, hop_ms, out.data(), (int)out.size()) == 0);
        CHECK(crepe_compute_f0(ctx, nullptr, (int)pcm.size(), hop_ms, out.data(), (int)out.size()) == 0);
        CHECK(crepe_compute_f0(ctx, pcm.data(), (int)pcm.size(), hop_ms, out.data(), 0) == 0);
        // max_frames < the natural frame count must truncate, not overrun.
        const int n = crepe_compute_f0(ctx, pcm.data(), (int)pcm.size(), hop_ms, out.data(), (int)out.size());
        CHECK(n == (int)out.size());
    }

    crepe_free(ctx);
}
