// test-core-cross-attn.cpp — unit test for core/cross_attn.h (manual path).
//
// The extraction's manual cross-attention path was bug-fixed (the V-multiply
// was dimensionally invalid; a spurious 1/sqrt(d) scale was applied — T5 is
// unscaled). This test builds the header's ggml graph and compares its output
// to an INDEPENDENT plain-C++ multi-head cross-attention reference (no ggml),
// proving the fixed graph both runs (valid dims) and is numerically correct
// for the T5 contract (apply_scale=false, no pre-norm, no rel_bias).
//
// CPU backend, no models. d_model=8, head_dim=4, n_heads=n_kv=2, T_enc=3, T_dec=1.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/cross_attn.h"

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
    float sym() { return (float)((nx() % 20001) / 10000.0 - 1.0); } // [-1,1]
};

} // namespace

TEST_CASE("core_cross_attn: manual path == plain-C++ MHA reference (T5 contract)",
          "[unit][core-cross-attn]") {
    const int d_model = 8, head_dim = 4, n_heads = 2, n_kv = 2, T_enc = 3, T_dec = 1;
    const int qd = n_heads * head_dim; // 8

    Rng rng(0x5151ULL);
    std::vector<float> dec(d_model * T_dec), q_proj(d_model * qd), o_proj(qd * d_model),
        cK(head_dim * T_enc * n_kv), cV(head_dim * T_enc * n_kv);
    for (auto* v : {&dec, &q_proj, &o_proj, &cK, &cV})
        for (auto& x : *v)
            x = rng.sym();

    // ---- ggml graph using the header ----
    std::vector<uint8_t> meta(4 * 1024 * 1024);
    ggml_init_params ip{meta.size(), meta.data(), true};
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* t_dec = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, T_dec);
    ggml_tensor* t_qp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, qd);
    ggml_tensor* t_op = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qd, d_model);
    ggml_tensor* t_cK = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, T_enc, n_kv);
    ggml_tensor* t_cV = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, T_enc, n_kv);
    for (auto* t : {t_dec, t_qp, t_op, t_cK, t_cV})
        ggml_set_input(t);
    ggml_set_name(t_dec, "dec");
    ggml_set_name(t_qp, "qp");
    ggml_set_name(t_op, "op");
    ggml_set_name(t_cK, "cK");
    ggml_set_name(t_cV, "cV");

    ggml_tensor* out = core_cross_attn::cross_attn_step_manual(
        ctx, t_dec, /*norm_w=*/nullptr, t_qp, t_op, t_cK, t_cV, head_dim, n_heads, n_kv, T_dec, T_enc,
        /*norm_eps=*/1e-5f, /*use_rms=*/true, /*rel_bias=*/nullptr, /*apply_scale=*/false);
    ggml_set_output(out);
    ggml_set_name(out, "out");

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 512, false, false);
    ggml_backend_sched_reset(sched);
    REQUIRE(ggml_backend_sched_alloc_graph(sched, gf)); // fails if dims are invalid

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec"), dec.data(), 0, dec.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qp"), q_proj.data(), 0, q_proj.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "op"), o_proj.data(), 0, o_proj.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cK"), cK.data(), 0, cK.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cV"), cV.data(), 0, cV.size() * sizeof(float));

    REQUIRE(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> got(d_model * T_dec);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), got.data(), 0, got.size() * sizeof(float));

    // ---- independent plain-C++ reference (unscaled MHA cross-attention) ----
    // ggml_mul_mat(A[k,m], B[k,n]) → R[m,n], R[i,j] = sum_k A[k,i]*B[k,j].
    // Layouts (ne0 innermost): dec[d,t]=dec[t*d_model+d]; qp[d,o]=qp[o*d_model+d];
    // cK[hd,te,h]=cK[h*T_enc*head_dim + te*head_dim + hd]; same for cV.
    auto QP = [&](int d, int o) { return q_proj[o * d_model + d]; };
    auto OP = [&](int i, int dm) { return o_proj[dm * qd + i]; };
    auto K = [&](int hd, int te, int h) { return cK[h * T_enc * head_dim + te * head_dim + hd]; };
    auto V = [&](int hd, int te, int h) { return cV[h * T_enc * head_dim + te * head_dim + hd]; };

    const int t = 0; // T_dec == 1
    std::vector<float> ref(d_model, 0.0f);
    std::vector<float> head_out(qd, 0.0f); // [h*head_dim + hd]
    for (int h = 0; h < n_heads; h++) {
        // Q[hd] for this head = sum_d qp[d, h*head_dim+hd] * dec[d,t]
        float Q[16];
        for (int hd = 0; hd < head_dim; hd++) {
            float acc = 0;
            for (int d = 0; d < d_model; d++)
                acc += QP(d, h * head_dim + hd) * dec[t * d_model + d];
            Q[hd] = acc;
        }
        // scores[te] = sum_hd K[hd,te,h]*Q[hd]   (NO scale — T5 contract)
        float sc[16];
        float mx = -1e30f;
        for (int te = 0; te < T_enc; te++) {
            float acc = 0;
            for (int hd = 0; hd < head_dim; hd++)
                acc += K(hd, te, h) * Q[hd];
            sc[te] = acc;
            mx = std::max(mx, acc);
        }
        float Z = 0;
        for (int te = 0; te < T_enc; te++) { sc[te] = std::exp(sc[te] - mx); Z += sc[te]; }
        for (int te = 0; te < T_enc; te++) sc[te] /= Z;
        // out[hd] = sum_te softmax[te]*V[hd,te,h]
        for (int hd = 0; hd < head_dim; hd++) {
            float acc = 0;
            for (int te = 0; te < T_enc; te++)
                acc += sc[te] * V(hd, te, h);
            head_out[h * head_dim + hd] = acc;
        }
    }
    // O projection: ref[dm] = sum_i op[i,dm]*head_out[i]
    for (int dm = 0; dm < d_model; dm++) {
        float acc = 0;
        for (int i = 0; i < qd; i++)
            acc += OP(i, dm) * head_out[i];
        ref[dm] = acc;
    }

    float max_abs = 0.0f;
    for (int dm = 0; dm < d_model; dm++)
        max_abs = std::max(max_abs, std::fabs(got[dm] - ref[dm]));
    INFO("max|Δ| ggml-vs-plain = " << max_abs);
    REQUIRE(max_abs < 1e-4f);

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
}
