// src/core/sanm.h — SANM (Self-Attention with Neural Memory) block.
//
// Implements the encoder block used by FunASR's SenseVoiceEncoderSmall, which
// fuses a standard multi-head self-attention with a parallel depthwise-conv
// "FSMN memory" branch operating on V. Adopted from CosyVoice / FunASR /
// FunAudioLLM and structurally distinct from the Conformer block helpers in
// `core_conformer` (no relative-position bias, no conv module, no Macaron
// FFN halves).
//
// Topology (matches funasr/models/sense_voice/model.py exactly):
//
//   x_norm = LayerNorm(x)                                # ne = (in_size, T)
//   qkv    = linear_q_k_v(x_norm) + b                    # ne = (3*n_feat, T)
//   q, k, v = split(qkv, n_feat, axis=ne[0])             # each ne = (n_feat, T)
//
//   # FSMN memory branch (depthwise 1D conv over V with residual to pre-conv V)
//   fsmn   = dw_conv1d(pad(v, left=(K-1)/2, right=K-1-(K-1)/2)) + v
//                                                         # ne = (n_feat, T)
//
//   # Standard MHA with q scaled by d_k^-0.5 (rather than dividing scores).
//   q_h, k_h, v_h = each reshape (head_dim, n_heads, T) and permute to
//                    (head_dim, T, n_heads)
//   scores = K_h^T · Q_h                                 # ne = (T, T, n_heads)
//   attn   = softmax_ext(scores, mask=null, scale=d_k^-0.5)
//   ctx    = V_perm · attn                               # ne = (head_dim, T, n_heads)
//   ctx    = reshape (n_feat, T) after permute back
//   att_out = linear_out(ctx) + b
//
//   return att_out + fsmn
//
// Encoder block (EncoderLayerSANM.forward), pre-normalize_before:
//   r = x
//   y = SANM(LN(r))
//   if in_size == size: x = r + y   else: x = y          # NB block-0 special
//   r = x
//   x = r + FFN(LN2(r))                                  # FFN is linear→ReLU→linear
//
// Block 0 has in_size=560, size=512 → attn residual is skipped (the
// caller flags this via `apply_attn_residual=false`). All other 69 blocks
// use the same `apply_attn_residual=true` path.
//
// Norm eps: SenseVoice's LayerNorm uses PyTorch nn.LayerNorm default
// (eps=1e-5). Mismatching this would silently degrade cosine similarity
// per block; we plumb it through `BlockParams.ln_eps`.

#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>

namespace core_sanm {

struct BlockWeights {
    // LayerNorms (LN_in_size, LN_size). w/b each.
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;

    // Fused Q/K/V linear with bias.
    //   qkv.w shape = (in_size, 3 * n_feat) in ggml convention
    //   qkv.b shape = (3 * n_feat,)
    ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;

    // Output projection with bias.
    //   out.w shape = (n_feat, n_feat); out.b shape = (n_feat,)
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;

    // Depthwise FSMN conv (no bias).
    //   fsmn.w shape = (K, n_feat) in ggml — the (n_feat, 1, K) PyTorch tensor
    //   stored with the singleton middle axis squeezed away by the converter.
    ggml_tensor* attn_fsmn_w = nullptr;

    // Positionwise FFN (idim, hidden_units) with biases. Activation is ReLU.
    //   l1.w = (in_size_or_size, hidden);  l1.b = (hidden,)
    //   l2.w = (hidden, in_size_or_size);  l2.b = (in_size_or_size,)
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_l2_w = nullptr, *ffn_l2_b = nullptr;
};

struct BlockParams {
    int in_size;  // input dim — equals size everywhere except block 0
    int size;     // n_feat — output dim
    int n_heads;  // attention heads
    int head_dim; // n_feat / n_heads
    int kernel;   // FSMN depthwise conv kernel (sense_voice = 11)
    float ln_eps;
    // When true, run the attention via ggml_flash_attn_ext (fuses scores +
    // softmax + ×V into one Metal/CUDA kernel; mask=null since this is a
    // bidirectional encoder with no per-utterance attn mask).
    // When false, use the unfused mul_mat + soft_max_ext + mul_mat path.
    bool flash_attn = false;
};

// Build one SANM encoder block. `cur` is (in_size, T). Returns the new
// (size, T) hidden state. The caller decides whether the attention branch
// gets a residual via `apply_attn_residual` — for block 0 (in_size != size)
// the upstream graph skips the attn residual entirely.
static inline ggml_tensor* build_block(ggml_context* ctx0, ggml_tensor* cur, int T, const BlockWeights& w,
                                       const BlockParams& p, bool apply_attn_residual) {
    const int n_feat = p.size;
    const int n_heads = p.n_heads;
    const int hd = p.head_dim;
    const int K = p.kernel;
    const float eps = p.ln_eps;
    const float scale = 1.0f / std::sqrt((float)hd);

    auto mm_bias = [&](ggml_tensor* W, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, W, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    // ---- norm1 + fused QKV ----
    ggml_tensor* residual = cur;
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, w.norm1_w, w.norm1_b, eps);
    ggml_tensor* qkv = mm_bias(w.attn_qkv_w, x, w.attn_qkv_b); // (3*n_feat, T)

    // Strided views into qkv — the row stride is `3*n_feat` floats so each
    // T-row leaves a 2*n_feat gap. The downstream reshape_3d asserts
    // contiguity, so make each split its own buffer with ggml_cont.
    const size_t row_bytes = qkv->nb[1];
    ggml_tensor* Q = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_feat, T, row_bytes, 0));
    ggml_tensor* K_ = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_feat, T, row_bytes, (size_t)n_feat * sizeof(float)));
    ggml_tensor* V = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, n_feat, T, row_bytes, (size_t)2 * n_feat * sizeof(float)));

    // ---- FSMN memory branch (depthwise conv1d on V) ----
    //
    // PyTorch: x = v.transpose(1, 2).contiguous(); pad(x); conv1d(x); transpose back; + v
    // ggml: V is (n_feat, T). Reshape to (T, 1, n_feat, 1) so ggml_conv_2d_dw_direct
    //       reads it as a 1D-over-time signal with n_feat channels. Weight
    //       reshaped to (K, 1, 1, n_feat). pad_w = (K-1)/2 on each side.
    ggml_tensor* fsmn;
    {
        ggml_tensor* w4 = ggml_cast(ctx0, w.attn_fsmn_w, GGML_TYPE_F32);
        w4 = ggml_reshape_4d(ctx0, w4, K, 1, 1, n_feat);

        ggml_tensor* v4 = ggml_cont(ctx0, ggml_transpose(ctx0, V)); // (T, n_feat)
        v4 = ggml_reshape_4d(ctx0, v4, T, 1, n_feat, 1);
        ggml_tensor* y = ggml_conv_2d_dw_direct(ctx0, w4, v4, 1, 1, (K - 1) / 2, 0, 1, 1);
        y = ggml_cont(ctx0, ggml_permute(ctx0, y, 1, 2, 0, 3)); // back to (n_feat, T, 1, 1)
        y = ggml_reshape_2d(ctx0, y, n_feat, T);
        fsmn = ggml_add(ctx0, y, V);
    }

    // ---- Multi-head self-attention ----
    Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
    K_ = ggml_reshape_3d(ctx0, K_, hd, n_heads, T);
    ggml_tensor* V_h = ggml_reshape_3d(ctx0, V, hd, n_heads, T);

    // Permute so head_dim stays fast, T becomes the "rows" axis per head.
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));   // (hd, T, n_heads)
    K_ = ggml_cont(ctx0, ggml_permute(ctx0, K_, 0, 2, 1, 3)); // (hd, T, n_heads)
    V_h = ggml_cont(ctx0, ggml_permute(ctx0, V_h, 0, 2, 1, 3));

    ggml_tensor* attn;
    if (p.flash_attn) {
        // flash_attn_ext fuses (scores + softmax + ×V) into one kernel.
        // Output is (hd, T, n_heads) which we reshape to (n_feat, T).
        attn = ggml_flash_attn_ext(ctx0, Q, K_, V_h, /*mask*/ nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, n_feat, T);
    } else {
        // scores = K^T · Q → (T, T, n_heads). Softmax along the key axis
        // (ne[0]), with the d_k^-0.5 scale fused in by soft_max_ext.
        ggml_tensor* scores = ggml_mul_mat(ctx0, K_, Q);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);
        // ctx = V_perm · scores where V_perm puts T fast for the dot product.
        ggml_tensor* V_p = ggml_cont(ctx0, ggml_permute(ctx0, V_h, 1, 0, 2, 3)); // (T, hd, n_heads)
        attn = ggml_mul_mat(ctx0, V_p, scores);                                  // (hd, T, n_heads)
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));            // (hd, n_heads, T)
        attn = ggml_reshape_2d(ctx0, attn, n_feat, T);
    }

    // Output projection + bias, then sum with the FSMN memory branch.
    attn = mm_bias(w.attn_out_w, attn, w.attn_out_b);
    ggml_tensor* attn_total = ggml_add(ctx0, attn, fsmn);

    // ---- Attention residual (skipped on the in_size != size block) ----
    cur = apply_attn_residual ? ggml_add(ctx0, residual, attn_total) : attn_total;

    // ---- FFN branch (always with residual) ----
    ggml_tensor* res2 = cur;
    ggml_tensor* y = ggml_norm_affine(ctx0, cur, w.norm2_w, w.norm2_b, eps);
    y = mm_bias(w.ffn_l1_w, y, w.ffn_l1_b);
    y = ggml_relu(ctx0, y);
    y = mm_bias(w.ffn_l2_w, y, w.ffn_l2_b);
    cur = ggml_add(ctx0, res2, y);

    return cur;
}

} // namespace core_sanm
