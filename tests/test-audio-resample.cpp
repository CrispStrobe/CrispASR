#include "core/audio_resample.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace {

float interior_mean(const std::vector<float>& samples) {
    constexpr size_t edge = 128;
    REQUIRE(samples.size() > 2 * edge);

    double sum = 0.0;
    for (size_t i = edge; i < samples.size() - edge; ++i) {
        sum += samples[i];
    }
    return static_cast<float>(sum / static_cast<double>(samples.size() - 2 * edge));
}

// Largest interior deviation from `level`. The mean alone cannot see ripple:
// a filter whose taps are half missing still averages close to the right
// value while every individual sample wobbles, so assert the worst sample.
float interior_max_dev(const std::vector<float>& samples, float level) {
    constexpr size_t edge = 128;
    REQUIRE(samples.size() > 2 * edge);

    double worst = 0.0;
    for (size_t i = edge; i < samples.size() - edge; ++i) {
        worst = std::max(worst, std::abs((double)samples[i] - (double)level));
    }
    return static_cast<float>(worst);
}

} // namespace

TEST_CASE("polyphase resampling preserves DC gain", "[unit][audio-resample]") {
    constexpr float level = 0.25f;
    const std::vector<float> input(48000, level);

    SECTION("48 kHz to 24 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 48000, 24000);
        REQUIRE(output.size() == 24000);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }

    SECTION("24 kHz to 16 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 24000, 16000);
        REQUIRE(output.size() == 32000);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }

    SECTION("24 kHz to 48 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 24000, 48000);
        REQUIRE(output.size() == 96000);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }

    // The source-separation surface (--separate) resamples the separator's
    // 44.1 kHz stems against 48 kHz host audio. This is the non-trivial
    // downsample (gcd 300 → L=147, M=160) that the up=max(L,M) bug silently
    // over-attenuated, so guard it explicitly.
    SECTION("48 kHz to 44.1 kHz") {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 48000, 44100);
        REQUIRE(output.size() == 44100);
        REQUIRE(interior_mean(output) == Catch::Approx(level).margin(5e-4f));
    }
}

// #334. A constant input must come out constant: the polyphase sum for every
// output phase is a slice of the SAME Kaiser-windowed sinc, so with the whole
// filter applied each phase sums to 1/L and the ×L compensation makes every
// output sample exactly `level` — no averaging, no window, exact arithmetic.
//
// The per-sample form is the point. `resample_polyphase` walked the input
// window as `j_center ± num_zeros`, but the filter spans `half_len/L =
// num_zeros * max(L, M) / L` INPUT samples, so whenever M > L (any
// downsample) the outer taps were never visited. The DC mean absorbed that
// (0.9984 vs 1.0, inside the 5e-4 margin above — a tolerance wider than the
// defect), while the per-phase sums each landed somewhere different, which is
// ripple. Assert the worst sample, at a margin an order of magnitude below
// the defect it is guarding.
TEST_CASE("polyphase resampling is ripple-free on a constant input", "[unit][audio-resample]") {
    constexpr float level = 0.25f;
    constexpr float kMaxDev = 5e-5f; // buggy build: 5.4e-4 at 24→16 kHz — 10x this margin
    const std::vector<float> input(48000, level);

    struct Case {
        int src;
        int dst;
    };
    // Downsamples (M > L) are the truncated ones; the upsamples are the
    // control — they were already exact and must stay that way.
    const Case cases[] = {
        {24000, 16000}, {48000, 16000}, {32000, 16000}, {44100, 16000}, {48000, 24000},
        {22050, 16000}, {16000, 24000}, {8000, 16000},  {24000, 48000},
    };
    for (const Case& c : cases) {
        CAPTURE(c.src, c.dst);
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), c.src, c.dst);
        REQUIRE(output.size() > 256);
        REQUIRE(interior_max_dev(output, level) < kMaxDev);
    }
}

// Regression: `sr` arrives from a decoded file header, so dst_rate/src_rate is
// attacker-controlled. A WAV declaring sampleRate = 1 asks for a 16000x
// upsample — 176 000 input samples become 2.816e9 output floats, 11.3 GB.
//
// THE OLD GUARD DID NOT GUARD. n_out was computed into an `int`, so that case
// overflowed to -1478967296 and the `n_out <= 0` test caught it BY LUCK. The
// cases below at 300 000 and 400 000 input samples overflow to +505032704 and
// +2105032704 — positive, past the test, allocating gigabytes and resampling
// with a length unrelated to the data. Those two are the point of this test:
// the committed fuzz WAV alone would pass against the broken code.
//
// The accepted arm is not decoration. Without it, "rejected" is equally
// explained by a resampler that rejects everything.
TEST_CASE("polyphase resampling bounds the expansion ratio", "[unit][audio-resample]") {
    auto tone = [](int n, int rate) {
        std::vector<float> v((size_t)n);
        for (int i = 0; i < n; i++)
            // Literal, not M_PI: MSVC does not define it without _USE_MATH_DEFINES,
            // and build.yml's ALL_BUILD compiles this whole tree on Windows.
            v[(size_t)i] = std::sin(2.0 * 3.14159265358979323846 * 100.0 * i / (double)rate);
        return v;
    };

    SECTION("real conversions are unaffected") {
        struct {
            int n, src, dst;
        } ok[] = {
            {176000, 44100, 16000}, // typical file -> model rate
            {32000, 16000, 24000},  // vibevoice / kyutai
            {16000, 8000, 48000},   // 6x, the widest real upsample
            {16000, 8000, 96000},   // 12x
            {1000, 1000, 64000},    // exactly at the limit
        };
        for (auto& c : ok) {
            const auto in = tone(c.n, c.src);
            const auto out = core_audio::resample_polyphase(in.data(), c.n, c.src, c.dst);
            INFO("" << c.src << " -> " << c.dst);
            REQUIRE_FALSE(out.empty());
            CHECK((int64_t)out.size() == ((int64_t)c.n * c.dst + c.src - 1) / c.src);
        }
    }

    SECTION("absurd expansion is refused, including the cases that overflowed positive") {
        struct {
            int n, src, dst;
        } bad[] = {
            {1000, 1000, 65000}, // one past the limit
            {176000, 1, 16000},  // the committed fuzz WAV; overflowed NEGATIVE
            {300000, 1, 16000},  // overflowed to +505032704
            {400000, 1, 16000},  // overflowed to +2105032704
        };
        for (auto& c : bad) {
            const auto in = tone(c.n, c.src > 0 ? c.src : 1);
            const auto out = core_audio::resample_polyphase(in.data(), c.n, c.src, c.dst);
            INFO("n=" << c.n << " " << c.src << " -> " << c.dst);
            CHECK(out.empty());
        }
    }
}

TEST_CASE("polyphase resampling benchmark: 10s audio", "[audio-resample][perf]") {
    const std::vector<float> input(480000, 0.25f); // 10 seconds at 48kHz
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 10; ++it) {
        const auto output = core_audio::resample_polyphase(input.data(), static_cast<int>(input.size()), 48000, 16000);
        REQUIRE(output.size() == 160000);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
    INFO("10s 48k->16k resample took " << ms << " ms per run");
    CHECK(ms > 0.0);
}

TEST_CASE("polyphase resampling bounds filter construction", "[unit][audio-resample]") {
    const float input[] = {0.25f, -0.125f};

    SECTION("extreme coprime header rates are refused") {
        // Output length is only one or two floats, so the expansion guard
        // cannot protect the filter. The original half-length calculation
        // overflowed for INT_MAX and requested over 1 GiB for 10000019.
        CHECK(core_audio::resample_polyphase(input, 2, INT_MAX, 16000).empty());
        CHECK(core_audio::resample_polyphase(input, 2, 10000019, 16000).empty());
        CHECK(core_audio::resample_polyphase(input, 2, INT_MAX, INT_MAX - 1).empty());
    }

    SECTION("invalid filter parameters are refused") {
        CHECK(core_audio::resample_polyphase(input, 2, 48000, 16000, 0).empty());
        CHECK(core_audio::resample_polyphase(input, 2, 48000, 16000, -1).empty());
        CHECK(core_audio::resample_polyphase(input, 2, 48000, 16000, INT_MAX).empty());
        CHECK(core_audio::resample_polyphase(input, 2, 48000, 16000, 14, std::numeric_limits<float>::quiet_NaN())
                  .empty());
        CHECK(
            core_audio::resample_polyphase(input, 2, 48000, 16000, 14, std::numeric_limits<float>::infinity()).empty());
    }

    SECTION("finite beta with overflowing normalization is refused") {
        CHECK(core_audio::resample_polyphase(input, 2, 48000, 16000, 14, std::numeric_limits<float>::max()).empty());
        CHECK(core_audio::resample_polyphase(input, 2, 48000, 16000, 14, -std::numeric_limits<float>::max()).empty());
    }

    SECTION("high and coprime audio rates remain valid") {
        const auto output = core_audio::resample_polyphase(input, 2, 192000, 96000);
        REQUIRE(output.size() == 1);
        CHECK(std::isfinite(output[0]));
        for (const auto& rates :
             {std::pair<int, int>{48001, 16000}, {44117, 48000}, {48000, 44117}, {383999, 384000}, {384000, 383999}}) {
            CAPTURE(rates.first, rates.second);
            const auto near = core_audio::resample_polyphase(input, 2, rates.first, rates.second);
            REQUIRE(near.size() == static_cast<size_t>((2LL * rates.second + rates.first - 1) / rates.first));
            for (const float sample : near)
                CHECK(std::isfinite(sample));
        }
    }

    SECTION("equal-rate copies do not require filter parameters") {
        const auto output =
            core_audio::resample_polyphase(input, 2, 16000, 16000, -1, std::numeric_limits<float>::infinity());
        REQUIRE(output == std::vector<float>{0.25f, -0.125f});
    }

    SECTION("rejection preserves a previously cached ordinary filter") {
        const auto before = core_audio::resample_polyphase(input, 2, 48000, 16000);
        CHECK(core_audio::resample_polyphase(input, 2, INT_MAX, 16000).empty());
        const auto after = core_audio::resample_polyphase(input, 2, 48000, 16000);
        CHECK(before == after);
    }
}
