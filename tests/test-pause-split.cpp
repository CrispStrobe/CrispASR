// Unit tests for src/core/pause_split.h — cutting audio at its pauses.
#include "core/pause_split.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

namespace {
constexpr int SR = 16000;
// `ms` of a 220 Hz tone at amplitude `a` appended to `x` (a = 0: silence).
void add(std::vector<float>& x, int ms, float a) {
    const int n = ms * SR / 1000;
    for (int i = 0; i < n; i++)
        x.push_back(a * (float)std::sin(2.0 * M_PI * 220.0 * (double)x.size() / SR));
}
} // namespace

TEST_CASE("pause split: cuts in the middle of a long enough pause", "[unit][pause-split]") {
    std::vector<float> x;
    add(x, 1500, 0.3f);
    add(x, 300, 0.0f);
    add(x, 1500, 0.3f);
    const auto p = core_pause_split::pieces(x.data(), (int)x.size(), SR, 200);
    REQUIRE(p.size() == 2);
    REQUIRE(p[0].first == 0);
    REQUIRE(p[1].second == (int)x.size());
    REQUIRE(p[0].second == p[1].first);
    const double cut_ms = 1000.0 * p[0].second / SR;
    REQUIRE(cut_ms > 1500.0);
    REQUIRE(cut_ms < 1800.0);
}

TEST_CASE("pause split: a pause shorter than the minimum is not a cut", "[unit][pause-split]") {
    std::vector<float> x;
    add(x, 1500, 0.3f);
    add(x, 100, 0.0f);
    add(x, 1500, 0.3f);
    REQUIRE(core_pause_split::pieces(x.data(), (int)x.size(), SR, 200).size() == 1);
}

TEST_CASE("pause split: silence at the edges is not a cut, short pieces merge", "[unit][pause-split]") {
    std::vector<float> x;
    add(x, 500, 0.0f);
    add(x, 1500, 0.3f);
    add(x, 300, 0.0f);
    add(x, 300, 0.3f); // with what follows the cut (150 + 300 + 100 ms) too short to stand alone
    add(x, 100, 0.0f);
    const auto p = core_pause_split::pieces(x.data(), (int)x.size(), SR, 200);
    REQUIRE(p.size() == 1);
    REQUIRE(p[0].first == 0);
    REQUIRE(p[0].second == (int)x.size());
}

TEST_CASE("pause split: works under a noise floor, not only on digital silence", "[unit][pause-split]") {
    std::vector<float> x;
    add(x, 1500, 0.3f);
    add(x, 400, 0.003f); // a quiet room, 40 dB down
    add(x, 1500, 0.3f);
    REQUIRE(core_pause_split::pieces(x.data(), (int)x.size(), SR, 200).size() == 2);
}

TEST_CASE("pause split: steady signal or empty input is one piece", "[unit][pause-split]") {
    std::vector<float> x;
    add(x, 3000, 0.3f);
    REQUIRE(core_pause_split::pieces(x.data(), (int)x.size(), SR, 200).size() == 1);
    REQUIRE(core_pause_split::pieces(nullptr, 0, SR, 200).empty());
}

TEST_CASE("pause split: a short pause in a long slice is still found", "[unit][pause-split]") {
    // 0.22 s in 10 s: 2 % of the frames (the case a percentile floor missed).
    std::vector<float> x;
    add(x, 6000, 0.3f);
    add(x, 220, 0.0f);
    add(x, 4000, 0.3f);
    REQUIRE(core_pause_split::pieces(x.data(), (int)x.size(), SR, 200).size() == 2);
}

TEST_CASE("pause split: digital silence elsewhere does not hide a noisy pause", "[unit][pause-split]") {
    std::vector<float> x;
    add(x, 300, 0.0f); // leading digital silence
    add(x, 2000, 0.3f);
    add(x, 300, 0.003f); // a pause 40 dB down, not zero
    add(x, 2000, 0.3f);
    REQUIRE(core_pause_split::pieces(x.data(), (int)x.size(), SR, 200).size() == 2);
}
