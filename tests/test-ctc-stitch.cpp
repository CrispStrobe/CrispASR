// test-ctc-stitch.cpp - crispasr_ctc_stitch::stitch joins the CTC grids of
// overlapping windows without dropping or duplicating tokens (omniASR CTC
// chunking; the old path concatenated window TEXT and printed overlap words
// twice). Synthetic grids, no model.
#include "crispasr_ctc_stitch.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

constexpr int V = 4;     // 0 = blank, 1..3 = tokens
constexpr int SPF = 100; // samples per frame

// Window over [off, end) whose frame t emits emit[off/SPF + t] (global frame index).
crispasr_ctc_stitch::Window make(int off, int end, const std::vector<int>& emit) {
    crispasr_ctc_stitch::Window w;
    w.off = off;
    w.end = end;
    w.T = (end - off) / SPF;
    w.lg.assign((size_t)w.T * V, 0.0f);
    for (int t = 0; t < w.T; t++)
        w.lg[(size_t)t * V + emit[off / SPF + t]] = 1.0f;
    return w;
}

// Greedy CTC over a stitched grid -> token string ("1 2 3").
std::string decode(const std::vector<float>& g, int n) {
    std::string s;
    int prev = -1;
    for (int t = 0; t < n; t++) {
        int best = 0;
        for (int v = 1; v < V; v++)
            if (g[(size_t)t * V + v] > g[(size_t)t * V + best])
                best = v;
        if (best != 0 && best != prev)
            s += (s.empty() ? "" : " ") + std::to_string(best);
        prev = best;
    }
    return s;
}

} // namespace

TEST_CASE("identical windows stitch to the unsplit grid", "[ctc-stitch]") {
    std::vector<int> e(40, 0);
    e[5] = 1;
    e[25] = 2; // inside the overlap (frames 20..29), at the midpoint
    e[35] = 3;
    auto a = make(0, 3000, e), b = make(2000, 4000, e);
    int n = 0;
    auto g = crispasr_ctc_stitch::stitch({a, b}, V, 0, &n);
    CHECK(n == 40);
    CHECK(decode(g, n) == "1 2 3");
}

TEST_CASE("an emission the two windows place on opposite sides of the midpoint is kept once", "[ctc-stitch]") {
    // Window a (with left context) emits token 2 at frame 24, window b (without)
    // at 26. A midpoint cut (25) keeps both -> "1 2 2 3". The longest joint-blank
    // run lies in 27..29, so the cut lands there and token 2 comes from a only.
    std::vector<int> ea(40, 0), eb(40, 0);
    ea[5] = eb[5] = 1;
    ea[24] = 2;
    eb[26] = 2;
    ea[35] = eb[35] = 3;
    // make the runs unambiguous: both windows busy on 20..21 (a word before)
    ea[20] = ea[21] = eb[20] = eb[21] = 1;
    auto a = make(0, 3000, ea), b = make(2000, 4000, eb);
    int n = 0;
    auto g = crispasr_ctc_stitch::stitch({a, b}, V, 0, &n);
    CHECK(n == 40);
    CHECK(decode(g, n) == "1 1 2 3");
}

TEST_CASE("no joint blank in the overlap falls back to the midpoint", "[ctc-stitch]") {
    std::vector<int> ea(40, 1), eb(40, 2); // never blank, windows disagree everywhere
    auto a = make(0, 3000, ea), b = make(2000, 4000, eb);
    int n = 0;
    auto g = crispasr_ctc_stitch::stitch({a, b}, V, 0, &n);
    CHECK(n == 40); // a keeps 0..24, b keeps 25..39: every frame exactly once
    CHECK(decode(g, n) == "1 2");
}

TEST_CASE("three windows, single window, and empty input", "[ctc-stitch]") {
    std::vector<int> e(60, 0);
    e[10] = 1;
    e[30] = 2;
    e[50] = 3;
    int n = 0;
    auto g = crispasr_ctc_stitch::stitch({make(0, 2500, e), make(1500, 4500, e), make(3500, 6000, e)}, V, 0, &n);
    CHECK(n == 60);
    CHECK(decode(g, n) == "1 2 3");
    g = crispasr_ctc_stitch::stitch({make(0, 6000, e)}, V, 0, &n);
    CHECK(n == 60);
    g = crispasr_ctc_stitch::stitch({}, V, 0, &n);
    CHECK(n == 0);
    CHECK(g.empty());
}
