// test-enhance-abi.cpp — model-free contracts for the selectable denoise
// pre-pass (crispasr_session_set_enhance / crispasr_enhance_audio_spec).
//
// Pins the spec grammar ("none"/"off"/"" disable), the same-length/same-rate
// output contract at several sample rates, the error codes, and that RNNoise
// output stays sample-aligned with its input (it used to lag one 10 ms frame,
// shifting every downstream timestamp).

#include <catch2/catch_test_macros.hpp>

#include "crispasr_session.h"

extern "C" {
#include "rnnoise.h"
}

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<float> tone_with_noise(int n, int sr) {
    std::vector<float> x(n);
    uint32_t seed = 12345;
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        const float noise = ((seed >> 9) / float(1u << 23) - 0.5f) * 0.02f;
        // A voiced-like harmonic burst in the middle third, noise elsewhere.
        const bool voiced = i > n / 3 && i < 2 * n / 3;
        const float t = (float)i / sr;
        x[i] = noise + (voiced ? 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * t) +
                                     0.15f * std::sin(2.0f * 3.14159265f * 440.0f * t)
                               : 0.0f);
    }
    return x;
}

} // namespace

TEST_CASE("enhance spec 'none' copies the input", "[unit][enhance]") {
    const auto x = tone_with_noise(1600, 16000);
    std::vector<float> y(x.size(), -9.0f);
    for (const char* spec : {"none", "off", "", "NONE"}) {
        REQUIRE(crispasr_enhance_audio_spec(spec, x.data(), (int32_t)x.size(), 16000, 1, y.data(), (int32_t)y.size()) ==
                0);
        REQUIRE(std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
    }
    REQUIRE(crispasr_enhance_audio_spec(nullptr, x.data(), (int32_t)x.size(), 16000, 1, y.data(), (int32_t)y.size()) ==
            0);
}

TEST_CASE("enhance rejects invalid arguments and unknown methods", "[unit][enhance]") {
    const auto x = tone_with_noise(1600, 16000);
    std::vector<float> y(x.size());
    REQUIRE(crispasr_enhance_audio_spec("none", nullptr, 10, 16000, 1, y.data(), 10) == -1);
    REQUIRE(crispasr_enhance_audio_spec("none", x.data(), 0, 16000, 1, y.data(), 10) == -1);
    REQUIRE(crispasr_enhance_audio_spec("none", x.data(), 100, 0, 1, y.data(), 100) == -1);
    REQUIRE(crispasr_enhance_audio_spec("none", x.data(), 100, 16000, 1, y.data(), 99) == -1);
    REQUIRE(crispasr_enhance_audio_spec("no-such-method", x.data(), 100, 16000, 1, y.data(), 100) == -2);
    REQUIRE(crispasr_enhance_audio_spec(CRISPASR_SOURCE_DIR "/README.md", x.data(), 100, 16000, 1, y.data(), 100) ==
            -2);
}

TEST_CASE("session enhance setter: NULL session, default, failures keep setting", "[unit][enhance]") {
    REQUIRE(crispasr_session_set_enhance(nullptr, "rnnoise") == -1);
    REQUIRE(std::string(crispasr_session_get_enhance(nullptr)) == "none");
    REQUIRE(crispasr_enhance_model_archs() != nullptr);
}

TEST_CASE("rnnoise keeps length, rate and sample alignment", "[unit][enhance]") {
    if (!crispasr_enhance_has_rnnoise())
        SKIP("RNNoise not compiled into this build");
    for (int sr : {8000, 16000, 22050, 48000}) {
        const auto x = tone_with_noise(sr, sr); // one second
        std::vector<float> y(x.size(), 0.0f);
        REQUIRE(crispasr_enhance_audio_spec("rnnoise", x.data(), (int32_t)x.size(), sr, 1, y.data(),
                                            (int32_t)y.size()) == 0);
        bool finite = true;
        for (float v : y)
            finite = finite && std::isfinite(v);
        REQUIRE(finite);
        // Best cross-correlation lag between input and output must be ~0:
        // a one-frame RNNoise delay would show up as sr/100 samples.
        int best_lag = 0;
        double best = -1e30;
        const int max_lag = sr / 50;
        for (int lag = -max_lag; lag <= max_lag; lag++) {
            double acc = 0.0;
            for (int i = sr / 3; i < 2 * sr / 3; i++) {
                const int j = i + lag;
                if (j >= 0 && j < (int)y.size())
                    acc += (double)x[i] * y[j];
            }
            if (acc > best) {
                best = acc;
                best_lag = lag;
            }
        }
        // Allow 1 ms of phase ambiguity on this tonal signal; the old bug was
        // a full 10 ms frame.
        INFO("sample rate " << sr << " best lag " << best_lag);
        REQUIRE(std::abs(best_lag) <= sr / 1000);
    }
}

TEST_CASE("rnnoise at 48 kHz is bit-identical to xiph's rnnoise_demo loop", "[unit][enhance]") {
    if (!crispasr_enhance_has_rnnoise())
        SKIP("RNNoise not compiled into this build");
    // Reference: the frame loop of xiph/rnnoise examples/rnnoise_demo.c
    // (6cbfd53 — the revision src/rnnoise vendors): short-range floats in,
    // first output frame dropped. The library's 48 kHz path must reproduce it
    // exactly; resampling only enters at other rates.
    const int F = 480, n_frames = 200, n = F * n_frames;
    const auto x = tone_with_noise(n, 48000);
    std::vector<float> ref;
    DenoiseState* st = rnnoise_create(nullptr);
    REQUIRE(st != nullptr);
    float frame[480];
    for (int f = 0; f < n_frames; f++) {
        for (int i = 0; i < F; i++)
            frame[i] = x[f * F + i] * 32768.0f;
        rnnoise_process_frame(st, frame, frame);
        if (f > 0)
            for (int i = 0; i < F; i++)
                ref.push_back(frame[i] / 32768.0f);
    }
    rnnoise_destroy(st);
    std::vector<float> y(n);
    REQUIRE(crispasr_enhance_audio_spec("rnnoise", x.data(), n, 48000, 1, y.data(), n) == 0);
    REQUIRE(ref.size() == (size_t)(n - F));
    REQUIRE(std::memcmp(ref.data(), y.data(), ref.size() * sizeof(float)) == 0);
}
