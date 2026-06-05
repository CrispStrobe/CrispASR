// test-lcs-c-abi.cpp — verify the public C ABI symbol
// `crispasr_lcs_dedup_prefix_count` is exported from `libcrispasr` and
// agrees with the in-tree algorithm on the same fixtures the algorithm
// test (`test-lcs-chunk-merge.cpp`) covers.
//
// Failure modes this is designed to catch:
//   - The symbol disappears from the dynamic library (CMake link order
//     regression, missing CA_EXPORT, name mangling slip).
//   - The C-ABI wrapper's input-validation or argument-order changes and
//     no longer matches the C++ algorithm.
//
// Includes only the public header — by design, no C++-internal types.

#include <catch2/catch_test_macros.hpp>

#include "crispasr.h"

#include <cstdint>
#include <vector>

TEST_CASE("C ABI: empty inputs return 0", "[unit][lcs][c-abi]") {
    const std::vector<int32_t> Y = {1, 2, 3};
    REQUIRE(crispasr_lcs_dedup_prefix_count(nullptr, 0, Y.data(), (int)Y.size(), 1) == 0);
    REQUIRE(crispasr_lcs_dedup_prefix_count(Y.data(), (int)Y.size(), nullptr, 0, 1) == 0);
}

TEST_CASE("C ABI: perfect alignment matches the C++ algorithm", "[unit][lcs][c-abi]") {
    const std::vector<int32_t> X = {1, 2, 3, 5, 6, 7};
    const std::vector<int32_t> Y = {5, 6, 7, 8, 9, 10};
    REQUIRE(crispasr_lcs_dedup_prefix_count(X.data(), (int)X.size(), Y.data(), (int)Y.size(), 1) == 3);
}

TEST_CASE("C ABI: min_lcs_length gates short matches", "[unit][lcs][c-abi]") {
    const std::vector<int32_t> X = {1, 2, 3, 99};
    const std::vector<int32_t> Y = {99, 8, 9, 10};
    // Single-token match — default min=1 slices, min=2 does not.
    REQUIRE(crispasr_lcs_dedup_prefix_count(X.data(), (int)X.size(), Y.data(), (int)Y.size(), 1) == 1);
    REQUIRE(crispasr_lcs_dedup_prefix_count(X.data(), (int)X.size(), Y.data(), (int)Y.size(), 2) == 0);
}

TEST_CASE("C ABI: min_lcs_length <= 0 falls back to the NeMo default", "[unit][lcs][c-abi]") {
    // Bindings that pass `min_lcs_length=0` (forgot to set it / sentinel)
    // get NeMo's default rather than "match anything" — defensive.
    const std::vector<int32_t> X = {1, 2, 42, 43};
    const std::vector<int32_t> Y = {42, 43, 99};
    const int slice = crispasr_lcs_dedup_prefix_count(X.data(), (int)X.size(), Y.data(), (int)Y.size(), 0);
    REQUIRE(slice == 2);
}

TEST_CASE("C ABI: leftmost match is preferred", "[unit][lcs][c-abi]") {
    const std::vector<int32_t> X = {100, 1, 2};
    const std::vector<int32_t> Y = {1, 2, 3, 1, 2, 4};
    REQUIRE(crispasr_lcs_dedup_prefix_count(X.data(), (int)X.size(), Y.data(), (int)Y.size(), 1) == 2);
}
