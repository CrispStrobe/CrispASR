// test-lcs-chunk-merge.cpp — unit tests for the LCS-based chunk-merge
// deduplication ported from NeMo's `longest_common_subsequence_merge`
// (`nemo/collections/asr/parts/utils/streaming_utils.py`).
//
// The Python algorithm operates on integer token ids, so its behaviour
// is purely a function of the input lists — no model, no audio. We pin
// the key cases that NeMo's docstring + tests rely on:
//
//   1. No overlap at all → no slicing.
//   2. Perfect alignment (LCS reaches the end of X) → eager slice of the
//      matched prefix of Y.
//   3. Partial alignment with leftmost-LCS heuristic.
//   4. Long silence (LCS shorter than min_lcs_length) → no slicing.
//   5. Identity (X == suffix of Y) → slice the whole prefix.
//
// These map directly to the cases the user-visible duplicate `なので`
// (issue #89 / #114 follow-up) lands in: chunk[i-1] and chunk[i] share a
// short token run at the boundary, and the dedup is supposed to slice it.

#include <catch2/catch_test_macros.hpp>

#include "core/crispasr_lcs.h"

#include <vector>

using crispasr_lcs::lcs_dedup_prefix_count;
using crispasr_lcs::longest_common_subsequence_merge;

TEST_CASE("LCS: empty inputs produce no slice", "[unit][lcs]") {
    REQUIRE(lcs_dedup_prefix_count({}, {1, 2, 3}) == 0);
    REQUIRE(lcs_dedup_prefix_count({1, 2, 3}, {}) == 0);
    REQUIRE(lcs_dedup_prefix_count({}, {}) == 0);
}

TEST_CASE("LCS: no overlap produces no slice", "[unit][lcs]") {
    // X = [10, 20, 30], Y = [40, 50, 60] — disjoint, so the longest LCS
    // is 0 and nothing should be sliced.
    REQUIRE(lcs_dedup_prefix_count({10, 20, 30}, {40, 50, 60}) == 0);
}

TEST_CASE("LCS: perfect alignment slices the matched prefix", "[unit][lcs]") {
    // X ends with [5, 6, 7]; Y starts with [5, 6, 7, ...]. The
    // 3-token overlap is exactly the chunk-boundary duplicate region.
    // Caller drops the first 3 tokens of Y to dedup.
    std::vector<int32_t> X = {1, 2, 3, 5, 6, 7};
    std::vector<int32_t> Y = {5, 6, 7, 8, 9, 10};
    REQUIRE(lcs_dedup_prefix_count(X, Y) == 3);
}

TEST_CASE("LCS: longer LCS is preferred", "[unit][lcs]") {
    // X tail and Y head share [42, 43, 44, 45] — slice 4.
    std::vector<int32_t> X = {1, 2, 42, 43, 44, 45};
    std::vector<int32_t> Y = {42, 43, 44, 45, 99, 100, 101};
    REQUIRE(lcs_dedup_prefix_count(X, Y) == 4);
}

TEST_CASE("LCS: subthreshold match leaves Y untouched", "[unit][lcs]") {
    // Single matched token with min_lcs_length=2 → no slice.
    std::vector<int32_t> X = {1, 2, 3, 99};
    std::vector<int32_t> Y = {99, 8, 9, 10};
    REQUIRE(lcs_dedup_prefix_count(X, Y, /*min_lcs_length=*/2) == 0);
    // Default min_lcs_length=1 → slice the 1-token match.
    REQUIRE(lcs_dedup_prefix_count(X, Y) == 1);
}

TEST_CASE("LCS: identity tail produces full-prefix slice", "[unit][lcs]") {
    // X tail is identical to Y → drop the whole Y prefix.
    std::vector<int32_t> X = {10, 20, 30, 40, 50};
    std::vector<int32_t> Y = {10, 20, 30, 40, 50, 60, 70};
    REQUIRE(lcs_dedup_prefix_count(X, Y) == 5);
}

TEST_CASE("LCS: partial alignment with one-token gap is recovered", "[unit][lcs]") {
    // X tail ends with [7, 8, 9]. Y starts with [7, 99, 8, 9, ...] —
    // there's a spurious extra token between the matching 7 and 8.
    // NeMo's partial-alignment path with diagonal expansion catches
    // this case; we expect a non-zero slice (the exact count depends on
    // the heuristic but must be ≥ 1).
    std::vector<int32_t> X = {1, 2, 7, 8, 9};
    std::vector<int32_t> Y = {7, 99, 8, 9, 10, 11};
    int slice = lcs_dedup_prefix_count(X, Y);
    REQUIRE(slice >= 1);
    REQUIRE(slice <= (int)Y.size());
}

TEST_CASE("LCS: leftmost match is selected over later ones", "[unit][lcs]") {
    // Two possible matches of the same length; NeMo's algorithm prefers
    // the LEFTMOST one in Y (to maximize the slice). With X = [a, b]
    // and Y = [a, b, c, a, b, d], the leftmost LCS is at j=2 (slice 2)
    // — not the later occurrence at j=5.
    std::vector<int32_t> X = {100, 1, 2};
    std::vector<int32_t> Y = {1, 2, 3, 1, 2, 4};
    REQUIRE(lcs_dedup_prefix_count(X, Y) == 2);
}

TEST_CASE("LCS: long silence (blank repeats) does not over-slice", "[unit][lcs]") {
    // The blank-padding edge case from NeMo's commentary: both chunks
    // have a run of blanks at the boundary. The default min_lcs_length=1
    // would slice; with min_lcs_length>=4 we leave it alone because the
    // common run is the blank itself, not real content.
    std::vector<int32_t> X = {1, 2, 3, 0, 0, 0};
    std::vector<int32_t> Y = {0, 0, 0, 5, 6, 7};
    REQUIRE(lcs_dedup_prefix_count(X, Y, /*min_lcs_length=*/4) == 0);
}

TEST_CASE("LCS: raw struct exposes both is_complete_merge and slice_count", "[unit][lcs]") {
    auto m = longest_common_subsequence_merge({1, 2, 3, 4, 5}, {4, 5, 6, 7});
    REQUIRE(m.slice_count == 2);
    REQUIRE(m.is_complete_merge); // LCS reaches end of X
}
