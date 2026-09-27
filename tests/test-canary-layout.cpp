// test-canary-layout.cpp - the pointwise Conv1d layouts the canary loader
// accepts (#470 follow-up). Hermetic: no model file.
//
// The loader validates every tensor's exact shape. Pointwise weights reach it
// flat [in,out] (cstr's quantised legacy GGUFs) or with the kernel axis
// [1,in,out] (cstr's F16/F32 canary-1b-v2.gguf AND the transcribe.cpp schema of
// Canary 180M Flash). PR #470 accepted only the flat form for legacy files and
// so rejected the unquantised 1B-v2 model the regression suite pins.
#include <catch2/catch_test_macros.hpp>

#include "canary_layout.h"

using canary_layout::Pointwise;
using canary_layout::pointwise_layout;

TEST_CASE("canary pointwise: both converter layouts are accepted", "[unit][canary][layout]") {
    const int64_t flat_pw1[4] = {1024, 2048, 1, 1};   // canary-1b-v2-q8_0.gguf pw1
    const int64_t single_pw1[4] = {1, 1024, 2048, 1}; // canary-1b-v2.gguf (F16) pw1 - the #470 regression
    const int64_t single_pw2[4] = {1, 1024, 1024, 1}; // canary-1b-v2.gguf (F16) pw2
    const int64_t flash_pw1[4] = {1, 512, 1024, 1};   // canary-180m-flash (transcribe.cpp schema)
    CHECK(pointwise_layout(flat_pw1, 1024, 2048) == Pointwise::Flat);
    CHECK(pointwise_layout(single_pw1, 1024, 2048) == Pointwise::Singleton);
    CHECK(pointwise_layout(single_pw2, 1024, 1024) == Pointwise::Singleton);
    CHECK(pointwise_layout(flash_pw1, 512, 1024) == Pointwise::Singleton);
}

TEST_CASE("canary pointwise: wrong shapes are still rejected", "[unit][canary][layout]") {
    const int64_t transposed[4] = {2048, 1024, 1, 1};
    const int64_t kernel_last[4] = {1024, 2048, 1, 1};
    const int64_t kernel3[4] = {3, 1024, 2048, 1};
    const int64_t extra_dim[4] = {1, 1024, 2048, 2};
    CHECK(pointwise_layout(transposed, 1024, 2048) == Pointwise::Invalid);
    CHECK(pointwise_layout(kernel_last, 1024, 1024) == Pointwise::Invalid); // right rank, wrong out
    CHECK(pointwise_layout(kernel3, 1024, 2048) == Pointwise::Invalid);     // not kernel size 1
    CHECK(pointwise_layout(extra_dim, 1024, 2048) == Pointwise::Invalid);
}
