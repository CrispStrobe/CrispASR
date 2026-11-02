// test-core-logit-sampler.cpp — core_logit_sampler::sample (temperature/top-k/top-p).
//
// Properties a typo would break: top-k never returns a token outside the k
// best, top-p keeps exactly the smallest nucleus, non-finite logits are never
// chosen, a fixed seed reproduces the draw, and empirical frequencies follow
// softmax(logits / T).

#include <catch2/catch_test_macros.hpp>

#include "core/logit_sampler.h"

#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <vector>

using core_logit_sampler::Params;
using core_logit_sampler::sample;

TEST_CASE("top_k restricts draws to the k highest logits", "[unit]") {
    const std::vector<float> l = {0.1f, 3.0f, 2.0f, -1.0f, 2.5f, 0.0f};
    Params p;
    p.temperature = 5.0f; // flat enough that every kept token is drawn
    p.top_k = 3;
    std::mt19937_64 rng(1);
    std::set<int> seen;
    for (int i = 0; i < 4000; ++i)
        seen.insert(sample(l.data(), (int)l.size(), p, rng));
    CHECK(seen == std::set<int>{1, 4, 2});
}

TEST_CASE("top_p keeps the smallest nucleus and always the best token", "[unit]") {
    // softmax(log([0.5, 0.3, 0.15, 0.05])) == those probabilities.
    const std::vector<float> l = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
    Params p;
    p.top_p = 0.75f; // 0.5 < 0.75 <= 0.5 + 0.3 -> {0, 1} (off the float boundary)
    std::mt19937_64 rng(2);
    std::set<int> seen;
    for (int i = 0; i < 4000; ++i)
        seen.insert(sample(l.data(), (int)l.size(), p, rng));
    CHECK(seen == std::set<int>{0, 1});
    p.top_p = 0.1f; // below the best token's mass -> only token 0
    for (int i = 0; i < 200; ++i)
        CHECK(sample(l.data(), (int)l.size(), p, rng) == 0);
}

TEST_CASE("non-finite logits are never selected; all non-finite returns -1", "[unit]") {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> l = {nan, 0.0f, inf, -inf, 0.0f};
    Params p;
    std::mt19937_64 rng(3);
    for (int i = 0; i < 2000; ++i) {
        const int t = sample(l.data(), (int)l.size(), p, rng);
        CHECK((t == 1 || t == 4));
    }
    const std::vector<float> bad = {nan, inf, -inf};
    CHECK(sample(bad.data(), (int)bad.size(), p, rng) == -1);
}

TEST_CASE("same seed reproduces the same draws", "[unit]") {
    const std::vector<float> l = {0.3f, 0.2f, 0.1f, 0.0f, -0.1f};
    Params p;
    p.temperature = 0.7f;
    std::mt19937_64 a(42), b(42);
    for (int i = 0; i < 500; ++i)
        CHECK(sample(l.data(), (int)l.size(), p, a) == sample(l.data(), (int)l.size(), p, b));
}

TEST_CASE("draw frequencies follow softmax(logits / T)", "[unit]") {
    const std::vector<float> l = {1.0f, 0.0f};
    Params p;
    p.temperature = 0.5f; // P(0) = 1 / (1 + e^-2) = 0.8808
    std::mt19937_64 rng(7);
    const int n = 20000;
    int c0 = 0;
    for (int i = 0; i < n; ++i)
        c0 += sample(l.data(), (int)l.size(), p, rng) == 0;
    const double expect = 1.0 / (1.0 + std::exp(-2.0));
    CHECK(std::fabs((double)c0 / n - expect) < 0.01);
}
