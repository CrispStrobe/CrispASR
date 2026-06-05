// test-core-seanet.cpp — unit test for core/seanet_decoder.h's convt1d_crop.
//
// seanet_decoder.h's only bug-fix was convt1d_crop: the old inline version
// cropped the CHANNEL axis instead of the TIME axis (interchanged ne0/stride/
// offset), corrupting audio on every DecoderBlock. The fix delegates to the
// production-validated core_convt::convt1d_crop.
//
// This test verifies the crop acts on TIME: the cropped ConvTranspose1d output
// must equal the UNCROPPED output's time-slice [crop_left : T_unpad-crop_right]
// for every channel. A channel-axis crop (the old bug) would fail this. No
// conv reference math needed — we compare the same op with/without crop.
//
// CPU backend, no models.

#include <catch2/catch_test_macros.hpp>

#include "core/seanet_decoder.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t nx() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }
    float sym() { return (float)((nx() % 20001) / 10000.0 - 1.0); }
};

// Run core_seanet::convt1d_crop and return the (Cout, T_out) output flattened.
std::vector<float> run_crop(const std::vector<float>& xv, const std::vector<float>& wv,
                           const std::vector<float>& bv, int Cin, int Cout, int K, int T_in, int stride,
                           int crop_left, int crop_right, int* out_Cout, int* out_T) {
    std::vector<uint8_t> meta(8 * 1024 * 1024);
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);

    // core_convt::convt1d_crop expects x as (Cin, T_in): ne[0]=Cin, ne[1]=T_in.
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cin, T_in);
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, Cout, Cin); // (K, Cout, Cin)
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Cout);
    for (auto* t : {x, w, b})
        ggml_set_input(t);
    ggml_set_name(x, "x"); ggml_set_name(w, "w"); ggml_set_name(b, "b");

    ggml_tensor* y = core_seanet::convt1d_crop(ctx, x, w, b, stride, crop_left, crop_right);
    ggml_set_output(y);
    ggml_set_name(y, "y");
    *out_Cout = (int)y->ne[0];
    *out_T = (int)y->ne[1];

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 512, false, false);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), xv.data(), 0, xv.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w"), wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "b"), bv.data(), 0, bv.size() * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    ggml_tensor* yt = ggml_graph_get_tensor(gf, "y");
    std::vector<float> out((size_t)yt->ne[0] * yt->ne[1]);
    ggml_backend_tensor_get(yt, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return out;
}

} // namespace

TEST_CASE("core_seanet: convt1d_crop crops TIME, not channels", "[unit][core-seanet]") {
    const int Cin = 2, Cout = 3, K = 4, T_in = 3, stride = 2;
    const int T_unpad = (T_in - 1) * stride + K; // 8

    Rng rng(0x5EA9EULL);
    std::vector<float> xv((size_t)Cin * T_in), wv((size_t)K * Cout * Cin), bv(Cout);
    for (auto* v : {&xv, &wv, &bv})
        for (auto& e : *v)
            e = rng.sym();

    // Uncropped output: (Cout, T_unpad).
    int c0, t0;
    auto full = run_crop(xv, wv, bv, Cin, Cout, K, T_in, stride, 0, 0, &c0, &t0);
    REQUIRE(c0 == Cout);
    REQUIRE(t0 == T_unpad);

    SECTION("symmetric crop matches the time-slice") {
        const int cl = 1, cr = 2, T_out = T_unpad - cl - cr; // 5
        int c1, t1;
        auto cropped = run_crop(xv, wv, bv, Cin, Cout, K, T_in, stride, cl, cr, &c1, &t1);
        REQUIRE(c1 == Cout);
        REQUIRE(t1 == T_out);
        // output layout: (Cout, T) → element (cout, t) at t*Cout + cout.
        float max_abs = 0;
        for (int cout = 0; cout < Cout; cout++)
            for (int t = 0; t < T_out; t++) {
                float got = cropped[(size_t)t * Cout + cout];
                float exp = full[(size_t)(cl + t) * Cout + cout]; // time-shifted slice of the full output
                max_abs = std::max(max_abs, std::fabs(got - exp));
            }
        INFO("max|Δ| cropped-vs-time-slice = " << max_abs);
        REQUIRE(max_abs < 1e-5f);
    }

    SECTION("left-only crop (causal) matches") {
        const int cl = 3, cr = 0, T_out = T_unpad - cl;
        int c1, t1;
        auto cropped = run_crop(xv, wv, bv, Cin, Cout, K, T_in, stride, cl, cr, &c1, &t1);
        REQUIRE(t1 == T_out);
        float max_abs = 0;
        for (int cout = 0; cout < Cout; cout++)
            for (int t = 0; t < T_out; t++)
                max_abs = std::max(max_abs, std::fabs(cropped[(size_t)t * Cout + cout] -
                                                      full[(size_t)(cl + t) * Cout + cout]));
        REQUIRE(max_abs < 1e-5f);
    }
}
