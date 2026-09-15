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

TEST_CASE("campplus seg_pool: the two upstream tail conventions differ, and only on the tail",
          "[unit][tts][campplus][segpool]") {
    // cosyvoice3's upstream is campplus.onnx, NOT PyTorch, and the exported
    // graph divides the partial tail by the KERNEL size. Measured by running
    // the real graph: at T_cam=173 it matches the kernel-size arm exactly
    // (cos 1.000000) and the width arm only to 0.992663.
    const int T = 551, k = 100;
    const int n = n_segments(T, k);
    std::vector<float> in((size_t)T, 1.0f);
    std::vector<float> w((size_t)n, -99.0f), ks((size_t)n, -99.0f);
    avg(in.data(), 1, T, k, w.data(), tail_divisor::window_width);
    avg(in.data(), 1, T, k, ks.data(), tail_divisor::kernel_size);

    // Every FULL segment must agree — if they differ anywhere but the tail the
    // flag is changing something it has no business touching.
    for (int s = 0; s < n - 1; s++) {
        INFO("full segment " << s);
        REQUIRE(close_to(w[(size_t)s], ks[(size_t)s]));
    }
    // The tail is the whole difference: 51 frames of ones.
    REQUIRE(close_to(w[(size_t)n - 1], 1.0f));   // torch: own width
    REQUIRE(close_to(ks[(size_t)n - 1], 0.51f)); // onnx:  kernel size

    // CONTROL: with no partial tail the two conventions MUST be identical.
    // Without this the test above would pass equally on an implementation that
    // differed everywhere, and the ONNX no-tail arm (T_cam=100, cos 1.000000
    // across all three) is exactly what pinned the divergence to the tail.
    const int Tex = 500;
    const int nex = n_segments(Tex, k);
    std::vector<float> inex((size_t)Tex, 1.0f);
    std::vector<float> wex((size_t)nex, -99.0f), kex((size_t)nex, -99.0f);
    avg(inex.data(), 1, Tex, k, wex.data(), tail_divisor::window_width);
    avg(inex.data(), 1, Tex, k, kex.data(), tail_divisor::kernel_size);
    for (int s = 0; s < nex; s++)
        REQUIRE(close_to(wex[(size_t)s], kex[(size_t)s]));
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
