// src/core/ffn.h — shared feed-forward network helpers (header-only).
//
// Encapsulates the SwiGLU and plain SiLU FFN patterns that several of our
// model implementations repeat inside their graph-building code. Using
// header-only inline helpers means zero runtime overhead (the compiler
// sees through to the underlying ggml_* calls) while giving each model a
// one-line call instead of a 6-8 line block.
//
// SwiGLU consumers (all LLM-based):
//   qwen3_asr  voxtral  voxtral4b  granite_speech
//
// Plain SiLU FFN (Linear -> SiLU -> Linear) is used by the FastConformer
// family (parakeet / canary / canary_ctc / cohere). A helper for that is
// also provided for future use but the parakeet-family migration is not
// in the current commit batch.
//
// These helpers build the graph ops for one FFN block. RMSNorm and the
// residual add remain in the caller, since models differ in whether they
// use pre-norm or post-norm, whether they multiply by a learned scale
// after norm, and what the residual source tensor is.

#pragma once

#include "ggml.h"

namespace core_ffn {

// SwiGLU without biases:
//   out = W_down @ (silu(W_gate @ x) * W_up @ x)
//
// Matches the pattern in qwen3_asr, voxtral, and granite. Call this
// between the pre-FFN RMSNorm and the residual add:
//
//   ggml_tensor * h   = pre_norm(cur);
//   ggml_tensor * mlp = core_ffn::swiglu(ctx, h, gate_w, up_w, down_w);
//   cur = ggml_add(ctx, residual, mlp);
//
// Returns the output of the down projection (no residual added).
static inline ggml_tensor* swiglu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gate_w, ggml_tensor* up_w,
                                  ggml_tensor* down_w) {
    ggml_tensor* gate = ggml_mul_mat(ctx, gate_w, x);
    ggml_tensor* up = ggml_mul_mat(ctx, up_w, x);
    ggml_tensor* mlp = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    return ggml_mul_mat(ctx, down_w, mlp);
}

// SwiGLU with an optional bias on the down projection. Used by voxtral4b
// when ffn_down_b is present in the GGUF (some checkpoints).
static inline ggml_tensor* swiglu_down_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gate_w, ggml_tensor* up_w,
                                            ggml_tensor* down_w, ggml_tensor* down_b) {
    ggml_tensor* out = swiglu(ctx, x, gate_w, up_w, down_w);
    if (down_b)
        out = ggml_add(ctx, out, down_b);
    return out;
}

// GeGLU without biases (the GELU-gated variant — same shape as SwiGLU
// but with `gelu_pytorch_tanh` as the gate activation). HF Gemma4TextMLP
// uses this:
//
//   down_proj = self.down_proj(act_fn(self.gate_proj(x)) * self.up_proj(x))
//   act_fn = ACT2FN["gelu_pytorch_tanh"]   # tanh-approximation GELU
//
// ggml_gelu is the tanh-approximation variant (matching pytorch's
// `gelu_pytorch_tanh`); the exact erf-based GELU is `ggml_gelu_erf`.
static inline ggml_tensor* geglu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gate_w, ggml_tensor* up_w,
                                 ggml_tensor* down_w) {
    ggml_tensor* gate = ggml_mul_mat(ctx, gate_w, x);
    ggml_tensor* up = ggml_mul_mat(ctx, up_w, x);
    ggml_tensor* mlp = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    return ggml_mul_mat(ctx, down_w, mlp);
}

// Plain two-linear SiLU FFN:
//   out = W2 @ silu(W1 @ x + b1) + b2
//
// Used by the FastConformer family (parakeet / canary / cohere). Biases
// are optional — pass nullptr to skip. The helper is provided here for
// completeness; the parakeet-family migration will happen in a follow-up.
static inline ggml_tensor* silu_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* b1,
                                    ggml_tensor* w2, ggml_tensor* b2) {
    ggml_tensor* h = ggml_mul_mat(ctx, w1, x);
    if (b1)
        h = ggml_add(ctx, h, b1);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, w2, h);
    if (b2)
        h = ggml_add(ctx, h, b2);
    return h;
}

// Plain two-linear GELU FFN with optional biases. Uses the exact
// erf-based GELU (matches PyTorch `nn.GELU()` default, i.e.
// `approximate='none'`):
//
//   out = W2 @ gelu_erf(W1 @ x + b1) + b2
//
// Used by Whisper-style audio encoders (voxtral 3B audio encoder) and
// the MiMo-Audio-Tokenizer encoder. Biases are optional.
static inline ggml_tensor* gelu_erf_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* b1,
                                        ggml_tensor* w2, ggml_tensor* b2) {
    ggml_tensor* h = ggml_mul_mat(ctx, w1, x);
    if (b1)
        h = ggml_add(ctx, h, b1);
    h = ggml_gelu_erf(ctx, h);
    h = ggml_mul_mat(ctx, w2, h);
    if (b2)
        h = ggml_add(ctx, h, b2);
    return h;
}

} // namespace core_ffn
