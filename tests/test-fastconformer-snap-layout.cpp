// test-fastconformer-snap-layout.cpp - the diff-harness snapshot of a 4-D
// pre-encoder conv output must be laid out like the reference dumper's
// (tools/reference_backends/canary.py): time-major rows, feature
// k = channel * Freq + freq (frequency fastest). Hermetic: tiny ggml graph.
//
// The nightly canary regression reported pre_enc_c0..c6 at cos -0.16..-0.32
// while pre_encode_output matched at 0.999999: snap_conv4d's permute put the
// CHANNEL fastest, so the harness compared the same numbers in a different
// order - a readout that could not detect anything.
#include <catch2/catch_test_macros.hpp>

#include "core/fastconformer.h"

#include <vector>

TEST_CASE("snap_conv4d flattens (OW,OH,OC) conv output as T x (C*F), freq fastest", "[unit][fastconformer]") {
    const int OW = 3, OH = 2, OC = 4; // ggml conv2d output (W=freq, H=time, C)
    ggml_init_params ip = {16 * 1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, OW, OH, OC, 1);
    float* d = (float*)t->data;
    for (int c = 0; c < OC; c++)
        for (int h = 0; h < OH; h++)
            for (int w = 0; w < OW; w++)
                d[(size_t)c * OH * OW + (size_t)h * OW + w] = (float)(c * 100 + h * 10 + w);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    core_conformer::snap_conv4d(ctx, gf, t, "snap");
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    ggml_tensor* s = ggml_graph_get_tensor(gf, "snap");
    REQUIRE(s != nullptr);
    REQUIRE(s->ne[0] == OC * OW);
    REQUIRE(s->ne[1] == OH);
    const float* o = (const float*)s->data;
    // canary.py: t4 (C, T, F) -> permute(1, 0, 2) -> reshape(T, C*F)
    for (int h = 0; h < OH; h++)
        for (int c = 0; c < OC; c++)
            for (int w = 0; w < OW; w++) {
                INFO("t=" << h << " c=" << c << " f=" << w);
                CHECK(o[(size_t)h * OC * OW + (size_t)c * OW + w] == (float)(c * 100 + h * 10 + w));
            }
    ggml_free(ctx);
}
