// test-basic-pitch-conv — the Basic Pitch convolution fast path must be
// BYTE-IDENTICAL to the reference loop it replaces.
//
// Hermetic: no GGUF, no ggml, no audio. The six shapes below are the six real
// call sites in bp_forward_window, so a change to any of them that perturbs a
// single float fails here rather than in a posteriorgram nobody reads.
//
// Why bit-identity and not a tolerance: the fast path is an execution-order
// rewrite, not an approximation. Per output element the accumulation order is
// unchanged (ic, then kh, then kw ascending), and vectorising across `wo`
// cannot reorder anything because distinct wo are distinct accumulators. A
// tolerance wider than the defect is not a test — if these ever stop being
// equal, something real changed.
//
// CRISPASR_BASIC_PITCH_CONV_ISA is read once into a function-local static, so
// the kernel is fixed for the life of the process; the ISA sweep therefore
// lives in tests/env-live-tests.sh / CI rather than in a loop here. This
// binary checks whatever kernel the host auto-selects, which is the one the
// host would actually ship with.

#include "core/basic_pitch_conv.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <vector>

using core_basic_pitch::bp_conv;

namespace {

struct Shape {
    const char* name;
    int IC, H, W, OC, KH, KW, stride_w, pad_h, pad_w;
};

// The six call sites in bp_forward_window, at the production window (T = 172).
const Shape SHAPES[] = {
    {"contour_conv", 8, 172, 264, 8, 3, 39, 1, 1, 19}, {"contour_out", 8, 172, 264, 1, 5, 5, 1, 2, 2},
    {"note_conv", 1, 172, 264, 32, 7, 7, 3, 3, 2},     {"note_out", 32, 172, 88, 1, 7, 3, 1, 3, 1},
    {"onset_conv", 8, 172, 264, 32, 5, 5, 3, 2, 1},    {"onset_out", 33, 172, 88, 1, 3, 3, 1, 1, 1},
};

// Deterministic, and deliberately NOT a smooth signal: a smooth input lets a
// wrong-but-close kernel pass by accident.
struct Rng {
    unsigned s = 12345u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return ((float)(s >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
    }
};

} // namespace

TEST_CASE("basic-pitch fast conv is byte-identical to the reference loop", "[unit][basic-pitch]") {
    for (const Shape& sh : SHAPES) {
        Rng rng;
        bp_conv c;
        c.oc = sh.OC;
        c.ic = sh.IC;
        c.kh = sh.KH;
        c.kw = sh.KW;
        c.w.resize((size_t)sh.OC * sh.IC * sh.KH * sh.KW);
        for (float& v : c.w)
            v = rng.next() * 0.1f;
        c.b.resize((size_t)sh.OC);
        for (float& v : c.b)
            v = rng.next() * 0.01f;
        // A zero weight exercises the `kv == 0.0f` skip both paths share.
        c.w[c.w.size() / 3] = 0.0f;

        std::vector<float> in((size_t)sh.IC * sh.H * sh.W);
        for (float& v : in)
            v = rng.next();

        int w_ref = 0, w_fast = 0;
        const std::vector<float> ref =
            core_basic_pitch::bp_conv2d_ref(in, sh.IC, sh.H, sh.W, c, sh.stride_w, sh.pad_h, sh.pad_w, w_ref);

        // bp_conv2d consults the gate; drive the fast path directly so the test
        // means the same thing whichever way the default is currently set.
        w_fast = (sh.W + 2 * sh.pad_w - c.kw) / sh.stride_w + 1;
        std::vector<float> fast((size_t)c.oc * sh.H * w_fast);
        const core_basic_pitch::bp_conv_fast::Kernel kern = core_basic_pitch::bp_conv_fast::pick_kernel();
        for (int threads : {1, 4}) {
            std::fill(fast.begin(), fast.end(), 0.0f);
            core_parallel::for_each_chunk(c.oc * sh.H, threads, [&](int begin, int end) {
                for (int idx = begin; idx < end; idx++)
                    core_basic_pitch::bp_conv_fast::conv_row(in.data(), fast.data(), sh.IC, sh.H, sh.W, c, sh.stride_w,
                                                             sh.pad_h, sh.pad_w, w_fast, idx / sh.H, idx % sh.H, kern);
            });

            INFO(sh.name << " threads=" << threads << " kernel=" << core_basic_pitch::bp_conv_fast::kernel_name(kern));
            REQUIRE(w_fast == w_ref);
            REQUIRE(fast.size() == ref.size());
            REQUIRE(std::memcmp(fast.data(), ref.data(), ref.size() * sizeof(float)) == 0);
        }
    }
}

TEST_CASE("basic-pitch conv shapes match the documented MMAC budget", "[unit][basic-pitch]") {
    // Guards the header comment that says contour_conv is 70% of the work, so
    // a shape edit cannot silently invalidate the reason the fast path exists.
    double total = 0, contour = 0;
    for (const Shape& sh : SHAPES) {
        const int W_out = (sh.W + 2 * sh.pad_w - sh.KW) / sh.stride_w + 1;
        const double mmac = (double)sh.OC * sh.IC * sh.KH * sh.KW * sh.H * W_out / 1e6;
        total += mmac;
        if (std::strcmp(sh.name, "contour_conv") == 0)
            contour = mmac;
    }
    REQUIRE(total > 480.0);
    REQUIRE(total < 490.0);
    REQUIRE(contour / total > 0.69);
}
