// src/core/conformer_ibm.h — IBM-flavour Macaron Conformer block helpers.
//
// granite_speech.cpp and granite_nle.cpp share an identical 16-layer
// Macaron Conformer encoder (FFN1 → MHSA-with-Shaw-RPE → conv module →
// FFN2 → post-LN). The four block-internal helpers (FFN, conv module,
// fused norm + Q/KV matmul pair, Shaw block attention) and the per-layer
// Shaw RPE lookup-table builder were copy-pasted between the two TUs;
// they live here now so both runtimes share the same compiled code.
//
// Sibling of `core/fastconformer.h`, NOT a merge target — parakeet /
// canary use a different Conformer dialect (NeMo's: conv subsampling,
// MHA RPE) and forcing both into one header would muddy both. Keep them
// separate.
//
// Header-only `static inline` so the helpers inline at each call site
// and the existing numerical paths stay bit-identical (the lift is a
// pure rename).

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace core_conformer_ibm {

// FFN module (Macaron half-step): LayerNorm → up → SiLU → down. Both
// FFN1 and FFN2 use this — caller picks the per-block weights.
//
//   x   : (d, T) F32, row-major
//   out : (d, T) F32
static inline bool run_ffn(std::vector<uint8_t>& compute_meta, ggml_backend_sched_t sched, float* out, const float* x,
                           int d, int T, ggml_tensor* norm_w, ggml_tensor* norm_b, ggml_tensor* up_w, ggml_tensor* up_b,
                           ggml_tensor* down_w, ggml_tensor* down_b) {
    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(inp, "ffn_in");
    ggml_set_input(inp);

    ggml_tensor* cur = ggml_norm(ctx0, inp, 1e-5f);
    if (norm_w)
        cur = ggml_mul(ctx0, cur, norm_w);
    if (norm_b)
        cur = ggml_add(ctx0, cur, norm_b);
    cur = ggml_mul_mat(ctx0, up_w, cur);
    if (up_b)
        cur = ggml_add(ctx0, cur, up_b);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, down_w, cur);
    if (down_b)
        cur = ggml_add(ctx0, cur, down_b);

    ggml_set_name(cur, "ffn_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ffn_in"), x, 0, (size_t)d * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "ffn_out"), out, 0, (size_t)d * T * sizeof(float));
    return true;
}

// Fused LayerNorm + two parallel matmuls in a single graph dispatch.
// Used to apply attention norm then run Q (d → d) and KV (d → 2d) in
// one Metal/CPU round-trip. Shape contract:
//
//   x      : (d_in, T) F32
//   W_a    : (d_in, d_out_a) GGUF tensor
//   W_b    : (d_in, d_out_b) GGUF tensor
//   out_a  : (d_out_a, T) F32
//   out_b  : (d_out_b, T) F32
static inline bool run_norm_matmul_pair(std::vector<uint8_t>& compute_meta, ggml_backend_sched_t sched, float* out_a,
                                        ggml_tensor* W_a, int d_out_a, float* out_b, ggml_tensor* W_b, int d_out_b,
                                        const float* x, int d_in, int T, ggml_tensor* norm_w, ggml_tensor* norm_b,
                                        float eps) {
    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_in, T);
    ggml_set_name(inp, "nmm_pair_in");
    ggml_set_input(inp);

    ggml_tensor* normed = ggml_norm(ctx0, inp, eps);
    if (norm_w)
        normed = ggml_mul(ctx0, normed, norm_w);
    if (norm_b)
        normed = ggml_add(ctx0, normed, norm_b);

    ggml_tensor* r_a = ggml_mul_mat(ctx0, W_a, normed);
    ggml_set_name(r_a, "nmm_pair_out_a");
    ggml_set_output(r_a);

    ggml_tensor* r_b = ggml_mul_mat(ctx0, W_b, normed);
    ggml_set_name(r_b, "nmm_pair_out_b");
    ggml_set_output(r_b);

    ggml_build_forward_expand(gf, r_a);
    ggml_build_forward_expand(gf, r_b);
    ggml_free(ctx0);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "nmm_pair_in"), x, 0, (size_t)d_in * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "nmm_pair_out_a"), out_a, 0, (size_t)d_out_a * T * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "nmm_pair_out_b"), out_b, 0, (size_t)d_out_b * T * sizeof(float));
    return true;
}

// Conv module: LayerNorm → pointwise up → GLU → depthwise conv →
// folded BN → SiLU → pointwise down. BN is folded into bn_w / bn_b at
// load time so the runtime path is just multiply+add.
//
// The depthwise conv is dispatched as a (T, 1, inner, 1) batched dw
// conv with kernel K, stride 1, padding K/2.
//
//   x   : (d, T) F32
//   out : (d, T) F32
//
// Tensors are passed individually so the helper has no dependency on
// the caller's per-block struct shape (granite_enc_block vs
// granite_nle_enc_block — same fields but distinct types).
static inline bool run_conv_module(std::vector<uint8_t>& compute_meta, ggml_backend_sched_t sched, float* out,
                                   const float* x, int d, int T, ggml_tensor* norm_w, ggml_tensor* norm_b,
                                   ggml_tensor* up_w, ggml_tensor* up_b, ggml_tensor* dw_w, ggml_tensor* bn_w,
                                   ggml_tensor* bn_b, ggml_tensor* down_w, ggml_tensor* down_b) {
    const int inner = d * 2;
    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(inp, "conv_in");
    ggml_set_input(inp);

    ggml_tensor* cur = ggml_norm(ctx0, inp, 1e-5f);
    if (norm_w)
        cur = ggml_mul(ctx0, cur, norm_w);
    if (norm_b)
        cur = ggml_add(ctx0, cur, norm_b);

    if (up_w) {
        int in_ch = (int)up_w->ne[1], out_ch = (int)up_w->ne[2];
        cur = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, up_w, in_ch, out_ch), cur);
        if (up_b)
            cur = ggml_add(ctx0, cur, up_b);
    }

    int half = (int)cur->ne[0] / 2;
    ggml_tensor* x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, half, T, cur->nb[1], 0));
    ggml_tensor* x2 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, half, T, cur->nb[1], half * sizeof(float)));
    cur = ggml_mul(ctx0, x1, ggml_sigmoid(ctx0, x2));

    if (dw_w) {
        int K = (int)dw_w->ne[0];
        ggml_tensor* dw_w_f32 = ggml_cast(ctx0, dw_w, GGML_TYPE_F32);
        ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, inner);
        ggml_tensor* x_t = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        x_t = ggml_reshape_4d(ctx0, x_t, T, 1, inner, 1);
        x_t = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, x_t, 1, 1, K / 2, 0, 1, 1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, x_t, 1, 2, 0, 3));
        cur = ggml_reshape_2d(ctx0, cur, inner, T);
    }

    if (bn_w && bn_b) {
        cur = ggml_mul(ctx0, cur, bn_w);
        cur = ggml_add(ctx0, cur, bn_b);
    }
    cur = ggml_silu(ctx0, cur);

    if (down_w) {
        int in_ch = (int)down_w->ne[1], out_ch = (int)down_w->ne[2];
        cur = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, down_w, in_ch, out_ch), cur);
        if (down_b)
            cur = ggml_add(ctx0, cur, down_b);
    }

    ggml_set_name(cur, "conv_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "conv_in"), x, 0, (size_t)d * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "conv_out"), out, 0, (size_t)d * T * sizeof(float));
    return true;
}

// Block-local Shaw RPE attention on CPU.
//
//   For each block of ctx_size frames and each head:
//     attn[c,r] = (Q[c]·K[r] + Q[c]·RPE[c,r]) * scale,  softmax along r,
//     out[c]    = sum_r softmax * V[r]
//
// The last block may have `remainder` valid frames; positions beyond it
// are computed but their values are unused by the caller (residuals
// write out of bounds against zeroed tail data).
//
//   Q / K / V : (n_heads * hd, T) row-major
//   rpe       : (ctx_size, ctx_size, hd) precomputed lookup, or nullptr
//               (granite-nle's per-layer table may be empty if the
//               layer's attn_rel_pos_w had an unsupported quant type)
static inline void shaw_block_attention_cpu(float* out, const float* Q_data, const float* K_data, const float* V_data,
                                            const float* rpe, int T, int n_heads, int hd, int ctx_size, float scale,
                                            int remainder) {
    const int d = n_heads * hd;
    const int n_blocks = (T + ctx_size - 1) / ctx_size;

#pragma omp parallel for collapse(2) schedule(static)
    for (int blk = 0; blk < n_blocks; blk++) {
        for (int h = 0; h < n_heads; h++) {
            const int blk_start = blk * ctx_size;
            const int blk_len = (blk == n_blocks - 1 && remainder > 0) ? remainder : ctx_size;
            std::vector<float> scores((size_t)ctx_size * ctx_size);

            for (int c = 0; c < blk_len; c++) {
                for (int r = 0; r < blk_len; r++) {
                    float qk = 0.0f;
                    float pos = 0.0f;
                    for (int dd = 0; dd < hd; dd++) {
                        int q_idx = (h * hd + dd) + (blk_start + c) * d;
                        int k_idx = (h * hd + dd) + (blk_start + r) * d;
                        float q_val = Q_data[q_idx];
                        float k_val = K_data[k_idx];
                        qk += q_val * k_val;
                        if (rpe)
                            pos += q_val * rpe[(size_t)(c * ctx_size + r) * hd + dd];
                    }
                    scores[c * blk_len + r] = (qk + pos) * scale;
                }
            }

            for (int c = 0; c < blk_len; c++) {
                float max_val = -1e30f;
                for (int r = 0; r < blk_len; r++)
                    if (scores[c * blk_len + r] > max_val)
                        max_val = scores[c * blk_len + r];
                float sum = 0.0f;
                for (int r = 0; r < blk_len; r++) {
                    scores[c * blk_len + r] = std::exp(scores[c * blk_len + r] - max_val);
                    sum += scores[c * blk_len + r];
                }
                float inv_sum = 1.0f / (sum + 1e-10f);
                for (int r = 0; r < blk_len; r++)
                    scores[c * blk_len + r] *= inv_sum;
            }

            for (int c = 0; c < blk_len; c++) {
                for (int dd = 0; dd < hd; dd++) {
                    float sum = 0.0f;
                    for (int r = 0; r < blk_len; r++) {
                        int v_idx = (h * hd + dd) + (blk_start + r) * d;
                        sum += scores[c * blk_len + r] * V_data[v_idx];
                    }
                    out[(h * hd + dd) + (blk_start + c) * d] = sum;
                }
            }
        }
    }
}

// Build a Shaw RPE lookup table from a (max_pos*2+1, hd) embedding
// tensor.
//
//   out[c * ctx_size * hd + r * hd + d] = emb_table[dist[c,r] * hd + d]
//   dist[c, r] = clamp(c - r, -ctx_size, ctx_size) + max_pos
//
// `rpe_w` may be quantized; dequantization goes through the type's
// `to_float` trait (no scheduler needed). Returns false if the type has
// no CPU dequantizer (caller decides how to handle — usually skip the
// layer and run with rpe = nullptr).
//
// `out` is sized to ctx_size * ctx_size * hd by the call.
static inline bool build_shaw_rpe_lookup(ggml_tensor* rpe_w, int ctx_size, int hd, int max_pos,
                                         std::vector<float>& out) {
    if (!rpe_w)
        return false;
    const int emb_size = 2 * max_pos + 1;

    std::vector<float> emb_table((size_t)emb_size * hd);
    if (rpe_w->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(rpe_w, emb_table.data(), 0, emb_table.size() * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(rpe_w));
        ggml_backend_tensor_get(rpe_w, raw.data(), 0, raw.size());
        const struct ggml_type_traits* tt = ggml_get_type_traits(rpe_w->type);
        if (!tt || !tt->to_float)
            return false;
        tt->to_float(raw.data(), emb_table.data(), (int64_t)emb_table.size());
    }

    out.resize((size_t)ctx_size * ctx_size * hd);
    for (int c = 0; c < ctx_size; c++) {
        for (int r = 0; r < ctx_size; r++) {
            int dd = c - r;
            if (dd < -ctx_size)
                dd = -ctx_size;
            if (dd > ctx_size)
                dd = ctx_size;
            int idx = dd + max_pos;
            for (int e = 0; e < hd; e++)
                out[(size_t)(c * ctx_size + r) * hd + e] = emb_table[(size_t)idx * hd + e];
        }
    }
    return true;
}

} // namespace core_conformer_ibm
