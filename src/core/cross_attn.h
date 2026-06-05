// src/core/cross_attn.h — encoder-decoder cross-attention helpers (header-only).
//
// Hoists the cross-attention pattern that encoder-decoder models repeat:
//
//   t5_translate.cpp       — T5 decoder cross-attention (manual matmul path)
//   moonshine_streaming.cpp — Moonshine dec cross-attn (flash_attn_ext path)
//
// Future consumers: SpeechT5, Dia, Parler, Pocket — 4 backends.
//
// The pattern:
//   1. Precompute cross-KV once from encoder output:
//        K = reshape+permute(K_proj @ enc_out)  → (head_dim, T_enc, n_kv_heads)
//        V = reshape+permute(V_proj @ enc_out)  → (head_dim, T_enc, n_kv_heads)
//      Store in a persistent buffer for reuse across decode steps.
//
//   2. At each decode step, compute Q from the decoder hidden state:
//        Q = reshape+permute(Q_proj @ dec_hidden) → (head_dim, T_dec, n_heads)
//      Then attend over the precomputed cross-KV:
//        attn = flash_attn_ext(Q, K_cross, V_cross)
//        out  = O_proj @ reshape(attn)
//
// This header provides graph-building helpers for both phases. The
// actual KV buffer management (allocation, lifetime) stays in each
// backend because the buffer strategy differs (T5 uses per-layer
// separate tensors, Moonshine uses a shared KV context with both
// self and cross slots).
//
// ---------------------------------------------------------------------------
// Per-source adoption verdict (audited 2026-05-31):
//
//   moonshine_streaming.cpp — FAITHFUL via cross_attn_step (flash path).
//                             Standard 1/sqrt(head_dim) scale, GQA-free.
//   t5_translate.cpp        — FAITHFUL-WITH-CONFIG via cross_attn_step_manual:
//                               * pass apply_scale=false (T5 folds the
//                                 attention scale into the weights — NO
//                                 ggml_scale; see t5_translate.cpp:823-825).
//                               * use_rms=true (T5 RMSNorm pre-norm).
//                               * rel_bias=nullptr (T5 cross-attn has no
//                                 position bias).
//                             The V-multiply uses the transpose-V
//                             formulation (cont(transpose(cross_V)) then
//                             mul_mat) matching t5_translate.cpp:827-828.
// ---------------------------------------------------------------------------

#pragma once

#include "ggml.h"

#include <cmath> // std::sqrt

namespace core_cross_attn {

// Build the ggml ops to project encoder output into cross-attention K and V
// for one decoder layer.
//
// Inputs:
//   ctx       — ggml context for graph building.
//   enc_out   — encoder output tensor, shape (d_model, T_enc).
//   k_proj    — K projection weight, shape (d_model, n_kv_heads * head_dim).
//   v_proj    — V projection weight, shape (d_model, n_kv_heads * head_dim).
//   head_dim  — dimension per attention head.
//   n_kv_heads — number of key/value heads (may differ from n_q_heads for GQA).
//   T_enc     — encoder sequence length.
//
// Returns K and V tensors in the layout expected by ggml_flash_attn_ext:
//   K: (head_dim, T_enc, n_kv_heads)
//   V: (head_dim, T_enc, n_kv_heads)
//
// The returned tensors are contiguous (ggml_cont after permute).
struct CrossKV {
    ggml_tensor* K;
    ggml_tensor* V;
};

static inline CrossKV project_cross_kv(ggml_context* ctx, ggml_tensor* enc_out, ggml_tensor* k_proj,
                                       ggml_tensor* v_proj, int head_dim, int n_kv_heads, int T_enc) {
    CrossKV result{};

    // K projection: (d_model, T_enc) → (n_kv_heads * head_dim, T_enc)
    //             → reshape (head_dim, n_kv_heads, T_enc)
    //             → permute (head_dim, T_enc, n_kv_heads)
    ggml_tensor* K = ggml_mul_mat(ctx, k_proj, enc_out);
    K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, T_enc);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    result.K = K;

    // V projection: same reshape + permute.
    ggml_tensor* V = ggml_mul_mat(ctx, v_proj, enc_out);
    V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, T_enc);
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
    result.V = V;

    return result;
}

// Build the ggml ops for one cross-attention step in the decoder.
//
// Inputs:
//   ctx        — ggml context for graph building.
//   dec_hidden — current decoder hidden state, shape (d_model, T_dec).
//                For autoregressive decoding, T_dec is typically 1.
//   norm_w     — LayerNorm / RMSNorm weight for pre-cross-attn norm, or
//                nullptr if the caller has already normalized dec_hidden.
//   q_proj     — Q projection weight, shape (d_model, n_heads * head_dim).
//   o_proj     — output projection weight, shape (n_heads * head_dim, d_model).
//   cross_K    — precomputed cross-K, shape (head_dim, T_enc, n_kv_heads).
//   cross_V    — precomputed cross-V, shape (head_dim, T_enc, n_kv_heads).
//   head_dim   — dimension per attention head.
//   n_heads    — number of query heads.
//   T_dec      — decoder sequence length (usually 1 in autoregressive mode).
//   T_enc      — encoder sequence length.
//   norm_eps   — epsilon for the pre-norm (only used if norm_w != nullptr).
//   use_rms    — if true, use RMSNorm (ggml_rms_norm); if false, use LayerNorm
//                (ggml_norm). Only relevant when norm_w is provided.
//
// Returns the output of the cross-attention block (d_model, T_dec),
// WITHOUT the residual add — the caller handles that because some
// models add a scale factor or use different residual sources.
static inline ggml_tensor* cross_attn_step(ggml_context* ctx, ggml_tensor* dec_hidden, ggml_tensor* norm_w,
                                           ggml_tensor* q_proj, ggml_tensor* o_proj, ggml_tensor* cross_K,
                                           ggml_tensor* cross_V, int head_dim, int n_heads, int T_dec, int T_enc,
                                           float norm_eps = 1e-5f, bool use_rms = false) {
    ggml_tensor* cur = dec_hidden;

    // Optional pre-norm.
    if (norm_w) {
        if (use_rms) {
            cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, norm_eps), norm_w);
        } else {
            cur = ggml_mul(ctx, ggml_norm(ctx, cur, norm_eps), norm_w);
        }
    }

    // Q projection + reshape + permute to (head_dim, T_dec, n_heads).
    ggml_tensor* Q = ggml_mul_mat(ctx, q_proj, cur);
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T_dec);
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);

    // Attention scale.
    float scale = 1.0f / std::sqrt((float)head_dim);

    // Flash attention against precomputed cross-KV.
    // ggml_flash_attn_ext expects:
    //   Q: (head_dim, T_dec, n_heads)
    //   K: (head_dim, T_enc, n_kv_heads)
    //   V: (head_dim, T_enc, n_kv_heads)
    //   mask: nullptr (cross-attention is typically unmasked)
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, cross_K, cross_V, nullptr, scale, 0.0f, 0.0f);

    // Reshape back to (n_heads * head_dim, T_dec) and project.
    attn = ggml_reshape_2d(ctx, attn, n_heads * head_dim, T_dec);
    return ggml_mul_mat(ctx, o_proj, attn);
}

// Variant of cross_attn_step that uses manual matmul+softmax+matmul
// instead of ggml_flash_attn_ext. Needed for backends that require
// F32 precision on the QK^T scores (e.g., T5 with relative position
// bias), or when the flash-attn op is not available.
//
// Same signature as cross_attn_step except:
//   - No mask support (cross-attention is always unmasked).
//   - rel_bias: optional (n_heads, T_dec, T_enc) relative position bias
//     added to QK^T before softmax. Pass nullptr to skip.
//   - apply_scale: if true, scale QK^T by 1/sqrt(head_dim). DEFAULT false
//     because the T5 source (t5_translate.cpp:823-825) applies NO scale —
//     the attention scale is folded into the T5 projection weights.
//     Set true for backends that need the explicit softmax temperature.
static inline ggml_tensor* cross_attn_step_manual(ggml_context* ctx, ggml_tensor* dec_hidden, ggml_tensor* norm_w,
                                                  ggml_tensor* q_proj, ggml_tensor* o_proj, ggml_tensor* cross_K,
                                                  ggml_tensor* cross_V, int head_dim, int n_heads, int n_kv_heads,
                                                  int T_dec, int T_enc, float norm_eps = 1e-5f, bool use_rms = true,
                                                  ggml_tensor* rel_bias = nullptr, bool apply_scale = false) {
    ggml_tensor* cur = dec_hidden;

    // Optional pre-norm.
    if (norm_w) {
        if (use_rms) {
            cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, norm_eps), norm_w);
        } else {
            cur = ggml_mul(ctx, ggml_norm(ctx, cur, norm_eps), norm_w);
        }
    }

    // Q projection: (d_model, T_dec) → (n_heads * head_dim, T_dec)
    //             → permute to (head_dim, T_dec, n_heads)
    ggml_tensor* Q = ggml_mul_mat(ctx, q_proj, cur);
    Q = ggml_permute(ctx, ggml_reshape_3d(ctx, Q, head_dim, n_heads, T_dec), 0, 2, 1, 3);

    // QK^T: (head_dim, T_dec, n_heads) @ (head_dim, T_enc, n_kv_heads)^T
    //     = (T_enc, T_dec, n_heads) — ggml_mul_mat transposes the first arg.
    ggml_tensor* qk = ggml_mul_mat(ctx, cross_K, Q);
    ggml_mul_mat_set_prec(qk, GGML_PREC_F32);

    // Scale — OFF by default (T5 folds the scale into its weights; see
    // t5_translate.cpp:823-825, which has no ggml_scale here).
    if (apply_scale) {
        float scale = 1.0f / std::sqrt((float)head_dim);
        qk = ggml_scale(ctx, qk, scale);
    }

    // Optional relative position bias.
    if (rel_bias) {
        qk = ggml_add(ctx, qk, rel_bias);
    }

    // Softmax over the K dimension (ne[0] = T_enc).
    qk = ggml_soft_max(ctx, qk); // (T_enc, T_dec, n_heads)

    // QKV: softmax(QK^T) @ V → (head_dim, T_dec, n_heads).
    // cross_V is (head_dim, T_enc, n_kv_heads). Transpose V to
    // (T_enc, head_dim, n_heads) so ggml_mul_mat contracts the T_enc axis
    // against qk's ne[0]=T_enc. This matches t5_translate.cpp:827-828
    // (cv_t = cont(transpose(CV)); ca_kqv = mul_mat(cv_t, ca_kq)).
    ggml_tensor* cv_t = ggml_cont(ctx, ggml_transpose(ctx, cross_V));
    ggml_tensor* kqv = ggml_mul_mat(ctx, cv_t, qk);

    // Reshape: (head_dim, T_dec, n_heads) → permute to (head_dim, n_heads, T_dec)
    //        → reshape to (n_heads * head_dim, T_dec).
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    kqv = ggml_reshape_2d(ctx, kqv, n_heads * head_dim, T_dec);

    // Output projection.
    return ggml_mul_mat(ctx, o_proj, kqv);
}

} // namespace core_cross_attn
