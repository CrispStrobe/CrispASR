// src/core/fastconformer.h — shared FastConformer encoder helpers.
//
// Replaces the dw_striding pre-encode + rel-pos sinusoidal table + 24-32×
// macaron Conformer block body that parakeet, canary, and canary_ctc each
// have a near-identical copy of. The only per-model difference is whether
// Q/K/V/output projections, FFN linears, and pointwise convs carry biases
// — parakeet and canary_ctc don't, canary does.
//
// The helper is header-only so the compiler inlines it straight into each
// caller, producing the exact same ggml op sequence as the original inline
// code and preserving bit-identical graph execution on the regression sweep.
//
// Scope:
//   core_conformer::rel_shift        — (T-1)-shift view used by rel-pos attn
//   core_conformer::make_pos_enc     — sinusoidal rel-pos table builder
//   core_conformer::PreEncodeWeights — dw_striding subsampling weights
//   core_conformer::BlockWeights     — one Conformer block's tensors
//   core_conformer::build_pre_encode — conv front-end + linear → (d, T)
//   core_conformer::build_block      — one Conformer block
//
// Each model still owns its own ggml_context setup, input tensor creation,
// and final output head (CTC / RNN-T joint / transformer decoder). What
// moves into here is just the shared encoder body.

#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace core_conformer {

// ---------------------------------------------------------------------------
// Rel-pos "shift" trick from Transformer-XL / Conformer: rewrites the raw
// (Q @ R^T) matrix — which is indexed by (rel_pos, query_pos) with rel_pos
// running over 2T-1 positions — into a square (key_pos, query_pos) matrix
// of size (T, T). This is done as a strided view, no copy.
// Input:  [2T-1, T, H]
// Output: [T,   T, H]
// ---------------------------------------------------------------------------
static inline ggml_tensor* rel_shift(ggml_context* ctx, ggml_tensor* a) {
    const int T = (int)a->ne[1];
    const int H = (int)a->ne[2];
    return ggml_view_3d(ctx, a, T, T, H, a->nb[1] - a->nb[0], a->nb[2], (T - 1) * a->nb[0]);
}

// ---------------------------------------------------------------------------
// Sinusoidal rel-pos table, layout (d_model, 2T-1), with positions running
// descending from +(T-1) to -(T-1). Memory layout is pe[dim + pos*d] so
// that ne[0]=d (fast axis) and ne[1]=2T-1 (slow axis) matches the ggml
// tensor created as ggml_new_tensor_2d(F32, d, 2T-1).
//
// IMPORTANT: the tensor is created as ggml_new_tensor_2d(F32, d, 2T-1),
// giving ne[0]=d (fast) and ne[1]=2T-1 (slow). The CORRECT memory layout
// is therefore `pe[dim + pos*d]`, NOT `pe[(2*i)*K + j]` (which transposes
// the axes). An earlier version of parakeet/cohere shipped with the
// transposed layout — parakeet's TDT decoder is robust enough to mostly
// recover, but canary's encoder–decoder cross-attention is not. If you
// see word boundaries drifting on a new consumer, re-check this first.
// ---------------------------------------------------------------------------
static inline std::vector<float> make_pos_enc(int d_model, int T) {
    const int n_pos = 2 * T - 1;
    std::vector<float> pe((size_t)n_pos * d_model, 0.0f);
    for (int p = 0; p < n_pos; p++) {
        const float pos = (float)(T - 1 - p);
        for (int i = 0; i < d_model / 2; i++) {
            const float div = expf(-logf(10000.0f) * (float)(2 * i) / (float)d_model);
            pe[(size_t)p * d_model + 2 * i] = sinf(pos * div);
            pe[(size_t)p * d_model + 2 * i + 1] = cosf(pos * div);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Pre-encode (dw_striding 8× subsampling) weights.
//
//   Conv2d(1→C,  k=3, s=2, p=1) → ReLU
//   Conv2d_dw(C, k=3, s=2, p=1)
//   Conv2d(C→C,  k=1)            → ReLU
//   Conv2d_dw(C, k=3, s=2, p=1)
//   Conv2d(C→C,  k=1)            → ReLU
//   flatten(freq×channel) → Linear(W3*C → d_model)
// ---------------------------------------------------------------------------
struct PreEncodeWeights {
    ggml_tensor *conv0_w = nullptr, *conv0_b = nullptr; // first strided conv
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr; // dw
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr; // pw
    ggml_tensor *conv5_w = nullptr, *conv5_b = nullptr; // dw
    ggml_tensor *conv6_w = nullptr, *conv6_b = nullptr; // pw
    ggml_tensor *out_w = nullptr, *out_b = nullptr;     // Linear(W3*C → d_model)
};

// Snap a 4D conv output (OW, OH, OC, N) as a 2D named dup (OC*OW, OH) for
// staged comparison.  Feature ordering: k = oc*(OW) + ow  (matches Python's
// x.transpose(1,2).reshape(T, C*Freq) convention).
static inline void snap_conv4d(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* t, const char* name) {
    // permute(1, 2, 0, 3): (OW,OH,OC,N) → (OH,OC,OW,N)
    ggml_tensor* p = ggml_cont(ctx0, ggml_permute(ctx0, t, 1, 2, 0, 3));
    // reshape to (OC*OW, OH): ne[0]=OC*OW fastest, ne[1]=OH (T_enc)
    const int64_t C_Freq = t->ne[2] * t->ne[0]; // OC * OW
    const int64_t T_enc = t->ne[1];             // OH
    ggml_tensor* flat = ggml_reshape_2d(ctx0, p, C_Freq, T_enc);
    ggml_tensor* snap = ggml_dup(ctx0, flat);
    ggml_set_name(snap, name);
    ggml_build_forward_expand(gf, snap);
}

// Build the dw_striding pre-encoder. Input `mel` has shape (n_mels, T_mel).
// Returns a (d_model, T_enc) tensor where T_enc is read off the intermediate
// conv output via the caller (write it back through `out_T_enc`).
// When `gf` is non-null, named dup snaps are added after each conv step for
// staged comparison via the diff harness.
static inline ggml_tensor* build_pre_encode(ggml_context* ctx0, ggml_tensor* mel, const PreEncodeWeights& w,
                                            int subsampling_channels, int* out_T_enc, ggml_cgraph* gf = nullptr) {
    auto bias_4d = [&](ggml_tensor* b) {
        return ggml_cast(ctx0, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    ggml_tensor* cur = ggml_conv_2d(ctx0, w.conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv0_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c0");
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw(ctx0, w.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv2_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c2");
    cur = ggml_conv_2d(ctx0, w.conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv3_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c3");
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw(ctx0, w.conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv5_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c5");
    cur = ggml_conv_2d(ctx0, w.conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv6_b));
    if (gf)
        snap_conv4d(ctx0, gf, cur, "pre_enc_c6");
    cur = ggml_relu(ctx0, cur);

    const int H3 = (int)cur->ne[1];
    const int W3 = (int)cur->ne[0];
    const int C = subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, w.out_w, cur), w.out_b);

    if (out_T_enc)
        *out_T_enc = H3;
    return cur;
}

// ---------------------------------------------------------------------------
// One Conformer encoder block's weights. Bias tensors may be nullptr — when
// they are, the corresponding ggml_add is skipped. This accommodates the
// three FastConformer flavours we ship:
//
//   parakeet    — no biases on Q/K/V/out, ff linears, conv pw1/pw2
//   canary_ctc  — same as parakeet
//   canary      — biases on everything
//
// conv_dw_b is always present (parakeet/canary_ctc populate it synthetically
// via BN folding; canary has the PyTorch bias natively).
// ---------------------------------------------------------------------------
struct BlockWeights {
    // ---- FFN1 (macaron) ----
    ggml_tensor *norm_ff1_w = nullptr, *norm_ff1_b = nullptr;
    ggml_tensor *ff1_l1_w = nullptr, *ff1_l1_b = nullptr;
    ggml_tensor *ff1_l2_w = nullptr, *ff1_l2_b = nullptr;

    // ---- Self-attention (rel-pos with untied u/v biases) ----
    ggml_tensor *norm_attn_w = nullptr, *norm_attn_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor* attn_pos_w = nullptr; // no bias on rel-pos projection
    ggml_tensor* pos_bias_u = nullptr;
    ggml_tensor* pos_bias_v = nullptr;

    // ---- Conformer convolution module ----
    ggml_tensor *norm_conv_w = nullptr, *norm_conv_b = nullptr;
    ggml_tensor *conv_pw1_w = nullptr, *conv_pw1_b = nullptr; // (2d, d)
    ggml_tensor *conv_dw_w = nullptr, *conv_dw_b = nullptr;   // (d, 1, K)
    ggml_tensor *conv_pw2_w = nullptr, *conv_pw2_b = nullptr; // (d, d)

    // ---- FFN2 (macaron) ----
    ggml_tensor *norm_ff2_w = nullptr, *norm_ff2_b = nullptr;
    ggml_tensor *ff2_l1_w = nullptr, *ff2_l1_b = nullptr;
    ggml_tensor *ff2_l2_w = nullptr, *ff2_l2_b = nullptr;

    // ---- Block final LN ----
    ggml_tensor *norm_out_w = nullptr, *norm_out_b = nullptr;
};

struct BlockParams {
    int d; // d_model
    int n_heads;
    int head_dim; // d / n_heads
    int K;        // conv_kernel (usually 9)
    float ln_eps; // LayerNorm epsilon
};

// Build one Conformer block. `cur` must be (d, T). `pos_enc` is the shared
// sinusoidal rel-pos table (d, 2T-1). Returns the post-block (d, T) output.
static inline ggml_tensor* build_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* pos_enc, int T,
                                       const BlockWeights& e, const BlockParams& p) {
    const int d = p.d;
    const int n_heads = p.n_heads;
    const int head_dim = p.head_dim;
    const int K = p.K;
    const float eps = p.ln_eps;

    // Tiny helper: mul_mat + optional bias add.
    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    ggml_tensor* inpL = cur;

    // ---- FFN1 (macaron half) ----
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

    ggml_tensor* inpAttn = cur;

    // ---- Self-Attention (rel_pos with untied biases) ----
    x = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);

    ggml_tensor* Q = mm_bias(e.attn_q_w, x, e.attn_q_b);
    ggml_tensor* K_ = mm_bias(e.attn_k_w, x, e.attn_k_b);
    ggml_tensor* V = mm_bias(e.attn_v_w, x, e.attn_v_b);
    ggml_tensor* R = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc); // no bias

    ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
    ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

    Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
    Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
    K_ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T), 0, 2, 1, 3);
    R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

    // Compute the relative position bias BD = rel_shift(Q_v × R^T).
    // This is query-dependent so it can't be precomputed, but it CAN
    // be passed as the additive mask to ggml_flash_attn_ext, which
    // fuses AC (= Q_u × K^T) + BD + softmax + ×V into one kernel.
    ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
    ggml_tensor* BD = rel_shift(ctx0, BD_raw);

    // flash_attn_ext computes: softmax(Q_u × K^T * scale + mask) × V
    // We need:                 softmax((Q_u × K^T + BD) * scale)  × V
    // So pass mask = BD * scale to get equivalent semantics.
    const float scale = 1.0f / sqrtf((float)head_dim);
    // BD is a strided view from rel_shift — make contiguous before scale/cast.
    ggml_tensor* BD_c = ggml_cont(ctx0, BD);
    ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);
    // flash_attn_ext mask must be F16
    ggml_tensor* BD_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);

    // V needs [head_dim, T, n_heads] layout for flash_attn_ext (same as K)
    ggml_tensor* V_ = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, n_heads, T), 0, 2, 1, 3));

    ggml_tensor* attn_out =
        ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q_u), ggml_cont(ctx0, K_), V_, BD_mask, scale, 0.0f, 0.0f);
    attn_out = ggml_reshape_2d(ctx0, attn_out, d, T);

    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Conformer convolution module ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);

    // pw1: (d → 2d), then sigmoid GLU — fused into one op, avoids strided-view
    // CUDA fallback that plagued the manual sigmoid path (see issue #81 PR #05).
    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    cnv = ggml_siglu_swapped(ctx0, cnv);

    // dw conv (kernel K, padding K/2). BN was folded into conv_dw_w/b at load.
    ggml_tensor* dw_w_f32 = ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
    cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // (d, T) → (T, d)
    cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
    cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T);

    cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, e.conv_dw_b, d, 1));
    cnv = ggml_silu(ctx0, cnv);

    // pw2: (d → d)
    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 (macaron half) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

    // ---- Block final LN ----
    cur = ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);

    return cur;
}

} // namespace core_conformer
