// src/core/activation.h — non-FFN activation primitives for ggml graphs.
//
// FFN-shaped activations (gelu, swiglu, geglu, silu_ffn, gelu_erf_ffn)
// live in `ffn.h` because they're combined with their up/down projections.
// This header collects the standalone activations that get applied
// inside per-block residual paths or vocoder generator stacks.
//
// Currently:
//   snake_alpha  — Snake-α activation, used by BigVGAN-family vocoders
//                  (Kokoro generator, SNAC decoder, future iSTFTNet /
//                  ConvNeXt-vocoder ports). Per-channel learnable α, init=1.
//   snake_beta   — SnakeBeta activation, used by the qwen3-tts codec
//                  decoder (BigVGAN-style with separate α frequency and
//                  β amplitude scales).

#pragma once

#include "ggml.h"

namespace core_act {

// Snake-α activation: y = x + (1 / α) · sin²(α · x).
//
// Per-channel learnable scalar α with shape ne = (1, C, 1) F16 in the
// GGUF (matches PyTorch's `nn.Parameter(torch.ones(1, C, 1))`). We
// reshape to (C, 1) F32 so it broadcasts over the time axis of x.
//
// Input  x:     (C, T)     F32, channel-major.
// Input  alpha: (1, C, 1)  F16, the per-channel learnable scale.
// Output:       (C, T)     F32, same shape as x.
//
// α is initialised to 1 in training and stays bounded away from zero
// in practice, so the 1/α has no special-casing.
static inline ggml_tensor* snake_alpha(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    const int C = (int)x->ne[0];
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, C, 1);
    a = ggml_cast(ctx, a, GGML_TYPE_F32);  // (C, 1) F32 (Metal needs F32×F32)
    ggml_tensor* ax = ggml_mul(ctx, x, a); // α·x with α broadcast on T
    ggml_tensor* sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ax));
    ggml_tensor* div = ggml_div(ctx, sin_sq, a); // sin²(α·x) / α
    return ggml_add(ctx, x, div);
}

// SnakeBeta activation: y = x + exp(-β) · sin²(x · exp(α)).
//
// BigVGAN's two-parameter variant — α controls the frequency, β the
// amplitude, both per-channel. Used by the qwen3-tts codec decoder
// (and likely by Chatterbox / VoxCPM2 when those land).
//
// Input  x:     (C, T)  F32, channel-major.
// Input  alpha: (C,)    F32, per-channel frequency log-scale.
// Input  beta:  (C,)    F32, per-channel amplitude log-scale.
// Output:       (C, T)  F32.
static inline ggml_tensor* snake_beta(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta) {
    ggml_tensor* ea = ggml_exp(ctx, alpha);              // (C,)
    ggml_tensor* inv_eb = ggml_exp(ctx, ggml_neg(ctx, beta)); // (C,) = 1 / exp(β)
    ggml_tensor* xa = ggml_mul(ctx, x, ea);              // (C, T)
    ggml_tensor* s = ggml_sin(ctx, xa);
    ggml_tensor* s2 = ggml_mul(ctx, s, s);               // (C, T)
    ggml_tensor* scaled = ggml_mul(ctx, s2, inv_eb);
    return ggml_add(ctx, x, scaled);
}

} // namespace core_act
