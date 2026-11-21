// test-haas-quiet-point-split.cpp — parakeet_quiet_point_bounds, the sample-
// exact port of Haas-ASR's haas_asr.longform.split_at_quiet_points (#605).
// Expected values were computed with the upstream Python function (acc614f).
// Hermetic: synthetic audio, no model.

#include "parakeet_orchestrate.h"

#include <catch2/catch_test_macros.hpp>

#include <utility>
#include <vector>

namespace {

constexpr int SR = 16000;

std::vector<float> tone(int seconds, float amp = 0.5f) {
    std::vector<float> a((size_t)seconds * SR);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (i % 2 ? amp : -amp);
    return a;
}

void silence(std::vector<float>& a, double t0_s, double t1_s) {
    for (int i = (int)(t0_s * SR); i < (int)(t1_s * SR); ++i)
        a[(size_t)i] = 0.0f;
}

void require_cover(const std::vector<std::pair<int, int>>& b, int n) {
    REQUIRE_FALSE(b.empty());
    REQUIRE(b.front().first == 0);
    REQUIRE(b.back().second == n);
    for (size_t i = 1; i < b.size(); ++i)
        REQUIRE(b[i].first == b[i - 1].second);
    for (const auto& c : b)
        REQUIRE(c.second - c.first <= (int)((kHaasChunkSeconds + kHaasSearchSeconds) * SR));
}

} // namespace

TEST_CASE("haas split: audio up to one chunk is a single piece", "[unit][haas-split]") {
    auto a = tone(30);
    auto b = parakeet_quiet_point_bounds(a.data(), (int)a.size());
    REQUIRE(b == std::vector<std::pair<int, int>>{{0, 480000}});
    a = tone(12);
    REQUIRE(parakeet_quiet_point_bounds(a.data(), (int)a.size()) == std::vector<std::pair<int, int>>{{0, 192000}});
}

// All-silent audio: every window ties, np.argmin takes the first, so each cut
// is low + window / 2. Upstream: [(0, 448800), (448800, 897600), (897600, 1120000)].
TEST_CASE("haas split: ties break to the earliest window like np.argmin", "[unit][haas-split]") {
    std::vector<float> a((size_t)70 * SR, 0.0f);
    const auto b = parakeet_quiet_point_bounds(a.data(), (int)a.size());
    REQUIRE(b == std::vector<std::pair<int, int>>{{0, 448800}, {448800, 897600}, {897600, 1120000}});
    require_cover(b, (int)a.size());
}

// A 0.2 s pause at 29.3-29.5 s lies inside the +-2 s search: the cut lands at
// the centre of the first all-silent 0.1 s window, 29.3 s + 0.05 s.
TEST_CASE("haas split: a cut moves to the pause near its target", "[unit][haas-split]") {
    auto a = tone(50);
    silence(a, 29.3, 29.5);
    const auto b = parakeet_quiet_point_bounds(a.data(), (int)a.size());
    REQUIRE(b.size() == 2);
    REQUIRE(b[0].second == (int)(29.3 * SR) + 800);
    require_cover(b, (int)a.size());
}

// A pause outside the search window is ignored; the cut stays within +-2 s.
TEST_CASE("haas split: a pause outside the search window is not used", "[unit][haas-split]") {
    auto a = tone(50);
    silence(a, 25.0, 25.5);
    const auto b = parakeet_quiet_point_bounds(a.data(), (int)a.size());
    REQUIRE(b.size() == 2);
    REQUIRE(b[0].second >= 28 * SR);
    REQUIRE(b[0].second <= 32 * SR);
    require_cover(b, (int)a.size());
}

TEST_CASE("haas split: long audio is covered without gaps", "[unit][haas-split]") {
    auto a = tone(181);
    silence(a, 61.0, 61.4);
    silence(a, 88.0, 88.3);
    const auto b = parakeet_quiet_point_bounds(a.data(), (int)a.size());
    require_cover(b, (int)a.size());
    // Upstream, same signal: the 61 s pause lies outside every search window,
    // the 88 s one is found from the 88.05 s target.
    REQUIRE(b == std::vector<std::pair<int, int>>{{0, 448800},
                                                  {448800, 897600},
                                                  {897600, 1408800},
                                                  {1408800, 1857600},
                                                  {1857600, 2306400},
                                                  {2306400, 2755200},
                                                  {2755200, 2896000}});
}

// Subtitle cues inside a chunk (#605): split at a pause of >= 0.6 s, and before
// a cue would run past 12 s; never an empty cue.
TEST_CASE("haas cues: split at pauses and at the length cap", "[unit][haas-split]") {
    // words: 0-50, 55-100 | pause 70 cs | 170-220
    REQUIRE(parakeet_cue_starts({0, 55, 170}, {50, 100, 220}) == std::vector<int>{0, 2});
    // a 59 cs gap is not a pause
    REQUIRE(parakeet_cue_starts({0, 109}, {50, 150}) == std::vector<int>{0});
    // continuous speech: cut before the cue would pass 12 s
    std::vector<int64_t> t0, t1;
    for (int i = 0; i < 40; ++i) {
        t0.push_back(i * 100);
        t1.push_back(i * 100 + 90);
    }
    const auto s = parakeet_cue_starts(t0, t1);
    REQUIRE(s.front() == 0);
    for (size_t k = 0; k < s.size(); ++k) {
        const int last = k + 1 < s.size() ? s[k + 1] - 1 : 39;
        REQUIRE(t1[(size_t)last] - t0[(size_t)s[k]] <= kHaasCueMaxCs);
        REQUIRE(last >= s[k]);
    }
    // one word longer than the cap still gets its own cue
    REQUIRE(parakeet_cue_starts({0, 10}, {5, 2000}) == std::vector<int>{0, 1});
    REQUIRE(parakeet_cue_starts({}, {}).empty());
}
