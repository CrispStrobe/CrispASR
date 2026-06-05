// src/core/adaln.h — AdaLN-Zero modulation helpers (header-only).
//
// Hoists the Adaptive Layer Normalization (AdaLN-Zero) pattern that
// multiple DiT-based TTS backends duplicate inline:
//
//   f5_tts.cpp          — F5-TTS DiT blocks (22 layers + final AdaLN)
//   cosyvoice3_tts.cpp  — CosyVoice3 flow DiT blocks (same architecture)
//
// ---------------------------------------------------------------------------
// Per-source adoption verdict (audited 2026-05-31):
//
//   f5_tts.cpp                        — FAITHFUL. modulate6/modulate2 bake
//                                       silu(t_emb) internally, matching F5.
//   cosyvoice3_tts.cpp (DEBUG path)   — FAITHFUL. Same internal-silu form.
//   cosyvoice3_tts.cpp (PRODUCTION,   — FAITHFUL-WITH-CONFIG. The
//     cv3_dit_block_apply :2778+)       production path pre-computes
//                                       silu(t_emb) ONCE upstream and passes
//                                       it in. Use the *_presilu variants
//                                       (modulate6_presilu / modulate2_presilu)
//                                       to AVOID a double-silu.
//   voxcpm2                           — DIVERGENT, do NOT adopt. voxcpm2 uses
//                                       RMSNorm + ungated residual, NOT
//                                       AdaLN-Zero. It is not a consumer of
//                                       this header.
//
// Future consumers: Zonos.
//
// The AdaLN-Zero pattern (Peebles & Xie, "Scalable Diffusion Models
// with Transformers", 2023):
//
//   1. Modulation: silu(t_emb) → Linear(dim, 6*dim) → chunk into 6 parts:
//      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
//
//   2. Pre-attention norm + modulation:
//      h = LayerNorm(x) * (1 + scale_msa) + shift_msa
//
//   3. After attention:
//      x = x + gate_msa * attn_output
//
//   4. Pre-FFN norm + modulation:
//      h = LayerNorm(x') * (1 + scale_mlp) + shift_mlp
//
//   5. After FFN:
//      x = x' + gate_mlp * ffn_output
//
// The "Final" variant uses only 2 chunks (scale, shift) for the
// output projection — no gating.
//
// These helpers build ggml graph ops. The caller provides the weight
// tensors and combines them with whatever attention/FFN implementation
// the model uses.

#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>

namespace core_adaln {

// The 6 modulation signals produced by a standard AdaLN-Zero block.
struct Modulation6 {
    ggml_tensor* shift_msa;
    ggml_tensor* scale_msa;
    ggml_tensor* gate_msa;
    ggml_tensor* shift_mlp;
    ggml_tensor* scale_mlp;
    ggml_tensor* gate_mlp;
};

// Compute the 6-way modulation from a time embedding.
//
//   t_emb     — (dim,) time/noise-level embedding.
//   adaln_w   — (dim, 6*dim) linear weight.
//   adaln_b   — (6*dim,) linear bias (may be nullptr for bias-free).
//   apply_silu — if true (default), apply silu(t_emb) internally
//                (f5_tts / cosyvoice3 DEBUG path). Pass false when the
//                caller has ALREADY applied silu upstream (cosyvoice3
//                PRODUCTION path) to avoid a double-silu.
//
// Returns 6 tensors of shape (dim,) each.
static inline Modulation6 modulate6(ggml_context* ctx, ggml_tensor* t_emb, ggml_tensor* adaln_w, ggml_tensor* adaln_b,
                                    bool apply_silu = true) {
    const int dim = (int)t_emb->ne[0];
    const size_t fs = sizeof(float);

    ggml_tensor* emb = apply_silu ? ggml_silu(ctx, t_emb) : t_emb;
    emb = ggml_mul_mat(ctx, adaln_w, emb);
    if (adaln_b) {
        emb = ggml_add(ctx, emb, adaln_b);
    }

    Modulation6 m{};
    m.shift_msa = ggml_view_1d(ctx, emb, dim, 0 * (size_t)dim * fs);
    m.scale_msa = ggml_view_1d(ctx, emb, dim, 1 * (size_t)dim * fs);
    m.gate_msa = ggml_view_1d(ctx, emb, dim, 2 * (size_t)dim * fs);
    m.shift_mlp = ggml_view_1d(ctx, emb, dim, 3 * (size_t)dim * fs);
    m.scale_mlp = ggml_view_1d(ctx, emb, dim, 4 * (size_t)dim * fs);
    m.gate_mlp = ggml_view_1d(ctx, emb, dim, 5 * (size_t)dim * fs);
    return m;
}

// Apply AdaLN modulation to a normalized input:
//   output = norm(x) * (1 + scale) + shift
//          = norm(x) + norm(x) * scale + shift
//
//   x      — (dim, T) input tensor.
//   scale  — (dim,) per-channel scale (broadcasts over T).
//   shift  — (dim,) per-channel shift (broadcasts over T).
//   eps    — LayerNorm epsilon (default 1e-6).
//
// Returns the modulated tensor (dim, T). Uses affine-free LayerNorm
// (ggml_norm, no learned gamma/beta), which is standard for DiT blocks.
static inline ggml_tensor* apply_norm_modulation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale,
                                                 ggml_tensor* shift, float eps = 1e-6f) {
    ggml_tensor* norm_x = ggml_norm(ctx, x, eps);
    ggml_tensor* scaled = ggml_mul(ctx, norm_x, scale);
    norm_x = ggml_add(ctx, norm_x, scaled);
    return ggml_add(ctx, norm_x, shift);
}

// Apply gated residual: x = residual + gate * sublayer_output.
//
// This is the standard AdaLN-Zero gating step applied after both
// the attention and FFN sub-layers.
static inline ggml_tensor* gated_residual(ggml_context* ctx, ggml_tensor* residual, ggml_tensor* sublayer_out,
                                          ggml_tensor* gate) {
    return ggml_add(ctx, residual, ggml_mul(ctx, sublayer_out, gate));
}

// The 2 modulation signals for AdaLN-Final (output norm before projection).
struct Modulation2 {
    ggml_tensor* scale;
    ggml_tensor* shift;
};

// Compute the 2-way modulation for AdaLN-Final.
//
//   t_emb     — (dim,) time embedding.
//   adaln_w   — (dim, 2*dim) linear weight.
//   adaln_b   — (2*dim,) linear bias (may be nullptr).
//   apply_silu — if true (default), apply silu(t_emb) internally. Pass
//                false when the caller already applied silu upstream
//                (cosyvoice3 PRODUCTION path) to avoid a double-silu.
//
// Note: F5-TTS uses chunk order (scale, shift) for AdaLN-Final,
// while CosyVoice3 also uses (scale, shift). This matches both.
// If a future model uses (shift, scale), the caller can swap the
// returned fields.
static inline Modulation2 modulate2(ggml_context* ctx, ggml_tensor* t_emb, ggml_tensor* adaln_w, ggml_tensor* adaln_b,
                                    bool apply_silu = true) {
    const int dim = (int)t_emb->ne[0];
    const size_t fs = sizeof(float);

    ggml_tensor* emb = apply_silu ? ggml_silu(ctx, t_emb) : t_emb;
    emb = ggml_mul_mat(ctx, adaln_w, emb);
    if (adaln_b) {
        emb = ggml_add(ctx, emb, adaln_b);
    }

    Modulation2 m{};
    m.scale = ggml_view_1d(ctx, emb, dim, 0);
    m.shift = ggml_view_1d(ctx, emb, dim, (size_t)dim * fs);
    return m;
}

// Apply AdaLN-Final modulation + optional projection.
//
//   x        — (dim, T) input tensor.
//   mod      — 2-way modulation from modulate2().
//   eps      — LayerNorm epsilon.
//
// Returns the modulated tensor (dim, T). The caller applies the
// final projection (Linear to mel_dim, etc.) separately.
static inline ggml_tensor* apply_final_norm(ggml_context* ctx, ggml_tensor* x, const Modulation2& mod,
                                            float eps = 1e-6f) {
    return apply_norm_modulation(ctx, x, mod.scale, mod.shift, eps);
}

} // namespace core_adaln
