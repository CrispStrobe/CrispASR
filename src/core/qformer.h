// src/core/qformer.h — windowed simplified Q-Former projector helpers.
//
// granite_nle.cpp ships a 2-layer "simplified" Q-Former projector — no
// self-attn, just cross-attn + MLP per layer, fed by a learned query +
// mean-pooled window plus a window-position embedding. This header lifts
// the two helper functions (pass A: per-layer LayerNorm + concat +
// linear-proj + GELU; pass B: per-window cross-attn + MLP + out_norm +
// out_linear graph) out of the TU so any future backend that wants the
// same simplified-windowed-Q-Former can reuse them.
//
// Note: granite_speech (base + plus) uses a STRUCTURALLY DIFFERENT
// projector — the full BLIP-2 Q-Former (self-attn + cross-attn + FFN per
// layer, no pass A, no window-mean-pool). It does not share these helpers.
// PLAN #55 step 5 was scoped to "lift `nle_proj_layer_proj` +
// `nle_proj_build_block` into core/qformer.h" so the lift here is NAR-only;
// granite_speech is left alone.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace core_qformer {

// Pass A — per-encoder-layer LayerNorm + concat across the K layers, then a
// linear projection (wide_d → hidden) followed by exact-erf GELU. Runs as a
// single graph dispatch and returns the post-GELU activations in `out`,
// resized to (hidden, T).
//
//   enc           : (wide_d = K*D, T) row-major F32
//   out           : output buffer (resized to hidden*T)
//   T             : time steps
//   wide_d        : input dim (== K*D)
//   hidden        : output dim after layer_proj
//   K             : number of encoder layers concatenated in `enc`
//   D             : per-layer dimension (== wide_d / K)
//   eps           : LayerNorm epsilon
//   layer_norm_w/b: K-element vectors of per-layer norm scale/bias
//   layer_proj_w/b: (wide_d → hidden) projection
static inline bool run_layer_proj(std::vector<uint8_t>& compute_meta, ggml_backend_sched_t sched,
                                  std::vector<float>& out, const float* enc, int T, int wide_d, int hidden, int K,
                                  int D, float eps, const std::vector<ggml_tensor*>& layer_norm_w,
                                  const std::vector<ggml_tensor*>& layer_norm_b, ggml_tensor* layer_proj_w,
                                  ggml_tensor* layer_proj_b) {
    if (K * D != wide_d)
        return false;

    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, wide_d, T);
    ggml_set_name(inp, "qformer_pa_in");
    ggml_set_input(inp);

    std::vector<ggml_tensor*> normed(K);
    for (int k = 0; k < K; k++) {
        ggml_tensor* slice = ggml_view_2d(ctx0, inp, D, T, inp->nb[1], (size_t)k * D * sizeof(float));
        ggml_tensor* s = ggml_cont(ctx0, slice);
        s = ggml_norm(ctx0, s, eps);
        if (k < (int)layer_norm_w.size() && layer_norm_w[k])
            s = ggml_mul(ctx0, s, layer_norm_w[k]);
        if (k < (int)layer_norm_b.size() && layer_norm_b[k])
            s = ggml_add(ctx0, s, layer_norm_b[k]);
        normed[k] = s;
    }
    ggml_tensor* cat = normed[0];
    for (int k = 1; k < K; k++)
        cat = ggml_concat(ctx0, cat, normed[k], 0);

    ggml_tensor* y = ggml_mul_mat(ctx0, layer_proj_w, cat);
    if (layer_proj_b)
        y = ggml_add(ctx0, y, layer_proj_b);
    y = ggml_gelu_erf(ctx0, y);

    ggml_set_name(y, "qformer_pa_out");
    ggml_build_forward_expand(gf, y);
    ggml_free(ctx0);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qformer_pa_in"), enc, 0, (size_t)wide_d * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    out.assign((size_t)hidden * T, 0.0f);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "qformer_pa_out"), out.data(), 0,
                            (size_t)hidden * T * sizeof(float));
    return true;
}

// Per-block weights for the simplified Q-Former (cross-attn + MLP only).
// `attn_*` is the cross-attention; `mlp_*` is the post-attn MLP.
struct BlockWeights {
    ggml_tensor* attn_norm_w;
    ggml_tensor* attn_norm_b;
    ggml_tensor* attn_q_w;
    ggml_tensor* attn_q_b;
    ggml_tensor* attn_k_w;
    ggml_tensor* attn_k_b;
    ggml_tensor* attn_v_w;
    ggml_tensor* attn_v_b;
    ggml_tensor* attn_o_w;
    ggml_tensor* attn_o_b;
    ggml_tensor* mlp_norm_w;
    ggml_tensor* mlp_norm_b;
    ggml_tensor* mlp_fc1_w;
    ggml_tensor* mlp_fc1_b;
    ggml_tensor* mlp_fc2_w;
    ggml_tensor* mlp_fc2_b;
};

// Outer (non-per-block) weights for the projector.
//   query        : (1, q_len, hidden) — learned query embedding
//   window_pos   : (1, block_size, hidden) — additive window-position embed
//   out_norm_w/b : final LayerNorm
//   out_linear_w/b: (hidden → llm_d) final linear
struct OuterWeights {
    ggml_tensor* query;
    ggml_tensor* window_pos;
    ggml_tensor* out_norm_w;
    ggml_tensor* out_norm_b;
    ggml_tensor* out_linear_w;
    ggml_tensor* out_linear_b;
};

struct Hparams {
    int hidden;     // Q-Former d_model
    int llm_d;      // final output dim (== granite-1B llm_d_model)
    int n_heads;    // attention heads
    int n_layers;   // # Q-Former layers (== blocks.size())
    int block_size; // encoder frames per window
    int downsample; // q_len = block_size / downsample
    float eps;      // LayerNorm epsilon
};

// Pass B — build one Q-Former graph for a single window.
//
//   * input tensor name : "qformer_blk_in"  shape (hidden, block_size)
//   * output tensor name: "qformer_blk_out" shape (llm_d, q_len)
//
// Caller dispatches the graph per-window with backend_sched_alloc_graph +
// graph_compute, copying its (block_size × hidden) window into the input
// tensor and reading (q_len × llm_d) tokens out.
static inline ggml_cgraph* build_block(std::vector<uint8_t>& compute_meta, const std::vector<BlockWeights>& blocks,
                                       const OuterWeights& outer, const Hparams& hp) {
    const int hidden = hp.hidden;
    const int n_heads = hp.n_heads;
    const int hd = hidden / n_heads;
    const int block_size = hp.block_size;
    const int q_len = block_size / hp.downsample;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, block_size);
    ggml_set_name(blk, "qformer_blk_in");
    ggml_set_input(blk);

    // mean_pool over downsample groups: (hidden, block_size) → (hidden, q_len)
    ggml_tensor* pooled = ggml_reshape_3d(ctx0, blk, hidden, hp.downsample, q_len);
    pooled = ggml_cont(ctx0, ggml_permute(ctx0, pooled, 1, 0, 2, 3));
    pooled = ggml_mean(ctx0, pooled);
    pooled = ggml_reshape_2d(ctx0, pooled, hidden, q_len);

    // query_embeds = query (1, q_len, hidden) + pooled
    ggml_tensor* qbase = ggml_reshape_2d(ctx0, outer.query, hidden, q_len);
    ggml_tensor* qcur = ggml_add(ctx0, qbase, pooled);

    // enc_kv = block + window_positions
    ggml_tensor* wpos = ggml_reshape_2d(ctx0, outer.window_pos, hidden, block_size);
    ggml_tensor* enc = ggml_add(ctx0, blk, wpos);

    for (int il = 0; il < hp.n_layers; il++) {
        const auto& b = blocks[il];

        // cross-attention
        {
            ggml_tensor* qn = ggml_norm(ctx0, qcur, hp.eps);
            if (b.attn_norm_w)
                qn = ggml_mul(ctx0, qn, b.attn_norm_w);
            if (b.attn_norm_b)
                qn = ggml_add(ctx0, qn, b.attn_norm_b);

            ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, qn);
            if (b.attn_q_b)
                Q = ggml_add(ctx0, Q, b.attn_q_b);
            ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, enc);
            if (b.attn_k_b)
                K = ggml_add(ctx0, K, b.attn_k_b);
            ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, enc);
            if (b.attn_v_b)
                V = ggml_add(ctx0, V, b.attn_v_b);

            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, q_len);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, block_size);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, block_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            ggml_tensor* a = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
            a = ggml_reshape_2d(ctx0, a, hidden, q_len);
            a = ggml_mul_mat(ctx0, b.attn_o_w, a);
            if (b.attn_o_b)
                a = ggml_add(ctx0, a, b.attn_o_b);

            qcur = ggml_add(ctx0, qcur, a);
        }

        // MLP (fc1 → SiLU → fc2)
        {
            ggml_tensor* xn = ggml_norm(ctx0, qcur, hp.eps);
            if (b.mlp_norm_w)
                xn = ggml_mul(ctx0, xn, b.mlp_norm_w);
            if (b.mlp_norm_b)
                xn = ggml_add(ctx0, xn, b.mlp_norm_b);

            ggml_tensor* h = ggml_mul_mat(ctx0, b.mlp_fc1_w, xn);
            if (b.mlp_fc1_b)
                h = ggml_add(ctx0, h, b.mlp_fc1_b);
            h = ggml_silu(ctx0, h);
            h = ggml_mul_mat(ctx0, b.mlp_fc2_w, h);
            if (b.mlp_fc2_b)
                h = ggml_add(ctx0, h, b.mlp_fc2_b);

            qcur = ggml_add(ctx0, qcur, h);
        }
    }

    // out_norm + out_linear
    ggml_tensor* xn = ggml_norm(ctx0, qcur, hp.eps);
    if (outer.out_norm_w)
        xn = ggml_mul(ctx0, xn, outer.out_norm_w);
    if (outer.out_norm_b)
        xn = ggml_add(ctx0, xn, outer.out_norm_b);
    ggml_tensor* y = ggml_mul_mat(ctx0, outer.out_linear_w, xn);
    if (outer.out_linear_b)
        y = ggml_add(ctx0, y, outer.out_linear_b);
    ggml_set_name(y, "qformer_blk_out");
    ggml_build_forward_expand(gf, y);

    ggml_free(ctx0);
    return gf;
}

} // namespace core_qformer
