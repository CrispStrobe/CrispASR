// CAM++ segment pooling parity (#377).
//
// The CAM layer's context is `x.mean(-1, keepdim=True) + seg_pooling(x)`, where
// seg_pooling is F.avg_pool1d(k=100, stride=100, ceil_mode=True). The port
// divided the PARTIAL TAIL window by the kernel size instead of by the frames
// actually in it. Nothing caught it: every CAM++ consumer (chatterbox,
// confucius4, cosyvoice3, dots, fireredtts3) was accepted end-to-end, and no
// acceptance test diffs this stage against upstream.
//
// Ground truth is torch, not a reading of count_include_pad:
//
//   F.avg_pool1d(torch.ones(1,1,551), kernel_size=100, stride=100,
//                ceil_mode=True)  ->  every one of the six segments is 1.0
//
// The all-ones input is what makes that decisive: under the wrong rule the
// 51-frame tail averages to 0.51, so the two rules render differently. A test
// on arbitrary data would not separate them nearly as clearly.
#include "core/campplus_segpool.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace campplus_segpool;

static bool close_to(float got, float want) {
    return (got > want ? got - want : want - got) <= 1e-5f;
}

TEST_CASE("campplus seg_pool segment count uses ceil_mode", "[unit][tts][campplus][segpool]") {
    REQUIRE(n_segments(551, 100) == 6); // 5 full + a 51-frame tail
    REQUIRE(n_segments(500, 100) == 5); // exact multiple, no tail
    REQUIRE(n_segments(1102, 100) == 12);
    REQUIRE(n_segments(1, 100) == 1);
    REQUIRE(n_segments(0, 100) == 0);
    REQUIRE(n_segments(10, 0) == 0);
}

TEST_CASE("campplus seg_pool divides the tail by its own width", "[unit][tts][campplus][segpool]") {
    // THE REGRESSION. All-ones, 551 frames: every segment must be exactly 1.0,
    // the 51-frame tail included. Dividing the tail by the kernel (100) yields
    // 0.51 — the bug this test exists for.
    const int C = 3, T = 551, k = 100;
    const int n_seg = n_segments(T, k);
    std::vector<float> in((size_t)C * T, 1.0f);
    std::vector<float> out((size_t)C * n_seg, -99.0f);
    avg(in.data(), C, T, k, out.data());

    for (int c = 0; c < C; c++)
        for (int s = 0; s < n_seg; s++)
            REQUIRE(close_to(out[(size_t)c * n_seg + s], 1.0f));

    // Named separately so a regression reads as the tail, not "some segment".
    INFO("tail segment: 1.0 is correct; 0.51 is the kernel-size-divisor bug");
    REQUIRE(close_to(out[(size_t)n_seg - 1], 1.0f));
}

TEST_CASE("campplus seg_pool is not simply returning the input value", "[unit][tts][campplus][segpool]") {
    // Positive control: a ramp must give per-segment means that differ from
    // each other, so the all-ones case above cannot pass for a trivial reason.
    const int T = 250, k = 100;
    const int n_seg = n_segments(T, k); // 3: 100, 100, 50
    std::vector<float> in((size_t)T);
    for (int t = 0; t < T; t++)
        in[(size_t)t] = (float)t;
    std::vector<float> out((size_t)n_seg, -99.0f);
    avg(in.data(), 1, T, k, out.data());

    REQUIRE(close_to(out[0], 49.5f));  // mean(0..99)
    REQUIRE(close_to(out[1], 149.5f)); // mean(100..199)
    // mean(200..249) = 224.5 with the correct divisor (50).
    // With the kernel-size divisor it would be 224.5 * 50/100 = 112.25.
    INFO("tail mean: 224.5 correct, 112.25 = kernel-size-divisor bug");
    REQUIRE(close_to(out[2], 224.5f));
}

TEST_CASE("campplus seg_pool keeps channels independent", "[unit][tts][campplus][segpool]") {
    const int C = 4, T = 130, k = 100;
    const int n_seg = n_segments(T, k); // 2: 100, 30
    std::vector<float> in((size_t)C * T);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            in[(size_t)c * T + t] = (float)c;
    std::vector<float> out((size_t)C * n_seg, -99.0f);
    avg(in.data(), C, T, k, out.data());

    for (int c = 0; c < C; c++)
        for (int s = 0; s < n_seg; s++)
            REQUIRE(close_to(out[(size_t)c * n_seg + s], (float)c));
}

TEST_CASE("campplus seg_pool tolerates degenerate inputs", "[unit][tts][campplus][segpool]") {
    std::vector<float> out(4, -99.0f);
    avg(nullptr, 1, 10, 100, out.data());
    avg(out.data(), 0, 10, 100, out.data());
    avg(out.data(), 1, 0, 100, out.data());
    avg(out.data(), 1, 10, 0, out.data());
    for (float v : out)
        REQUIRE(close_to(v, -99.0f)); // untouched
}

// ---------------------------------------------------------------------------
// CRISPASR_CAMPP_LEGACY_SEGPOOL — the A/B gate's own contract.
//
// The gate exists so the fix can be measured against the behaviour four shipped
// backends (chatterbox, confucius4, cosyvoice3, dots-tts) were accepted with.
// A measurement taken through it is only worth reading if the gate does exactly
// two things and no more:
//
//   * it must change the PARTIAL TAIL, or a per-backend "the two arms agree"
//     result means the gate never reached that backend rather than "the fix is
//     a no-op there" — vacuous, and it reads like a pass;
//   * it must change NOTHING ELSE, so a clip whose frame count is a whole
//     number of windows has to come out bit-identical in both arms. That case
//     is the one that separates "the divisor changed" from "something else
//     changed too", and it is arithmetic, so it belongs here rather than in a
//     Kaggle run.
//
// Asserted in code because both properties are invisible to every end-to-end
// acceptance test these backends have -- which is how the original divisor bug
// survived five consumers in the first place.
namespace {
struct ScopedLegacySegpool {
    bool had_prev = false;
    std::string prev;
    explicit ScopedLegacySegpool(const char* value) {
        if (const char* p = std::getenv("CRISPASR_CAMPP_LEGACY_SEGPOOL")) {
            had_prev = true;
            prev = p;
        }
        if (value)
            setenv("CRISPASR_CAMPP_LEGACY_SEGPOOL", value, 1);
        else
            unsetenv("CRISPASR_CAMPP_LEGACY_SEGPOOL");
    }
    ~ScopedLegacySegpool() {
        if (had_prev)
            setenv("CRISPASR_CAMPP_LEGACY_SEGPOOL", prev.c_str(), 1);
        else
            unsetenv("CRISPASR_CAMPP_LEGACY_SEGPOOL");
    }
};

// A ramp, so every window has a distinct mean and an accidental agreement
// cannot come from the input being degenerate.
std::vector<float> ramp(int C, int T) {
    std::vector<float> v((size_t)C * (size_t)T);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            v[(size_t)c * (size_t)T + (size_t)t] = (float)(t + 1) * (float)(c + 1) * 0.125f;
    return v;
}

std::vector<float> run_seg_pool(const char* legacy_env, int C, int T, int k) {
    ScopedLegacySegpool guard(legacy_env);
    std::vector<float> in = ramp(C, T);
    std::vector<float> out((size_t)C * (size_t)campplus_segpool::n_segments(T, k), -99.0f);
    campplus_segpool::avg(in.data(), C, T, k, out.data());
    return out;
}
} // namespace

TEST_CASE("campplus seg_pool legacy gate changes ONLY the partial tail", "[unit][tts][campplus][segpool]") {
    const int C = 3, k = 100;

    SECTION("exact multiple of the window: both arms must agree bit-for-bit") {
        // T = 400 -> 4 full windows, no tail. n_in_seg == seg_len everywhere, so
        // the two divisors are the same number and the gate has nothing to do.
        // If this ever differs, the gate is reaching something it must not.
        const int T = 400;
        REQUIRE(campplus_segpool::n_segments(T, k) == 4);
        const std::vector<float> fixed = run_seg_pool(nullptr, C, T, k);
        const std::vector<float> legacy = run_seg_pool("1", C, T, k);
        REQUIRE(fixed.size() == legacy.size());
        for (size_t i = 0; i < fixed.size(); i++)
            REQUIRE(fixed[i] == legacy[i]); // bit-for-bit: same divisor, same order
    }

    SECTION("partial tail present: the arms MUST differ, and only in the tail") {
        // T = 451 -> 4 full windows + a 51-frame tail. Anything that agrees in
        // the tail here means the gate is not reaching the code under test, and
        // every "identical" reading taken through it would be vacuous.
        const int T = 451;
        const int n_seg = campplus_segpool::n_segments(T, k);
        REQUIRE(n_seg == 5);
        const std::vector<float> fixed = run_seg_pool(nullptr, C, T, k);
        const std::vector<float> legacy = run_seg_pool("1", C, T, k);
        REQUIRE(fixed.size() == legacy.size());
        for (int c = 0; c < C; c++) {
            for (int sgi = 0; sgi < n_seg - 1; sgi++) {
                const size_t i = (size_t)c * (size_t)n_seg + (size_t)sgi;
                REQUIRE(fixed[i] == legacy[i]); // full windows are untouched
            }
            const size_t tail = (size_t)c * (size_t)n_seg + (size_t)(n_seg - 1);
            REQUIRE(fixed[tail] != legacy[tail]);
            // The legacy arm divided a 51-frame window by 100, so it is low by
            // exactly 51/100 -- a magnitude error, which cosine alone hides.
            REQUIRE(std::abs(legacy[tail] - fixed[tail] * (51.0f / 100.0f)) <= 1e-4f * std::abs(fixed[tail]));
        }
    }

    SECTION("the gate is off by default") {
        // Nothing in the suite may leave it set: a stray value would silently
        // put every later assertion on the known-wrong divisor.
        ScopedLegacySegpool guard(nullptr);
        REQUIRE(std::getenv("CRISPASR_CAMPP_LEGACY_SEGPOOL") == nullptr);
        const int T = 451, n_seg = campplus_segpool::n_segments(T, k);
        std::vector<float> in((size_t)T, 1.0f);
        std::vector<float> out((size_t)n_seg, -99.0f);
        campplus_segpool::avg(in.data(), 1, T, k, out.data());
        REQUIRE(close_to(out[(size_t)n_seg - 1], 1.0f)); // fixed divisor, not 0.51
    }
}
