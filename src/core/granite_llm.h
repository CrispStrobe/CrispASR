// src/core/granite_llm.h — Granite-1B (40-layer) decoder backbone.
//
// granite_speech.cpp (causal, KV-cached) and granite_nle.cpp (non-causal,
// single-pass) ship the same 40-layer Granite-1B body: pre-RMSNorm + GQA(16/4)
// flash-attn with NEOX RoPE θ=10000, pre-RMSNorm + SwiGLU FFN, residual scaled
// by `residual_multiplier` on every add, and a final RMSNorm. The four µP
// multipliers (embedding=12, attention=0.0078125, residual=0.22, logits=8) are
// the same in both. Only the attention path (causal+KV vs non-causal) and the
// LM head plumbing (separate output_w with `/logits_scaling` vs tied
// token_embd_w with no scaling) differ — those stay at the call sites.
//
// This header lifts the embedding-multiplier scale + 40-block forward + final
// RMSNorm into one ggml_tensor-returning function. Callers build their own
// inputs_embeds (concat of audio + text or audio + prompt), allocate any KV
// cache they need, and apply their own LM head + slicing on the post-norm
// hidden returned here.

#pragma once

#include <cmath>
#include <vector>

#include "ggml.h"

#include "core/attention.h"
#include "core/ffn.h"

namespace core_granite_llm {

struct RopeScaling {
    int n_ctx_orig = 0;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
};

// Match transformers' GraniteRotaryEmbedding / _compute_yarn_parameters.
// ggml already multiplies cos/sin by 1 + .1*log(factor): do not apply it twice.
static inline bool configure_yarn(RopeScaling& out, float factor, int original_ctx, float attention_factor,
                                  float beta_fast, float beta_slow) {
    if (!std::isfinite(factor) || factor < 1.0f || original_ctx <= 0 || !std::isfinite(attention_factor) ||
        attention_factor <= 0.0f || !std::isfinite(beta_fast) || !std::isfinite(beta_slow) || beta_fast < beta_slow ||
        beta_slow <= 0.0f) {
        return false;
    }
    RopeScaling next;
    next.n_ctx_orig = original_ctx;
    next.freq_scale = 1.0f / factor;
    next.ext_factor = 1.0f;
    next.attn_factor = attention_factor / (1.0f + 0.1f * std::log(factor));
    next.beta_fast = beta_fast;
    next.beta_slow = beta_slow;
    out = next;
    return true;
}

struct Hparams {
    int n_layers;
    int d_model;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    float rms_eps;
    float rope_theta;
    float embedding_multiplier;
    float attention_multiplier;
    float residual_multiplier;
    RopeScaling rope;
};

static inline core_attn::KvSelfAttnParams attention_params(const Hparams& hp) {
    core_attn::KvSelfAttnParams p = {};
    p.n_heads = hp.n_heads;
    p.n_kv_heads = hp.n_kv_heads;
    p.head_dim = hp.head_dim;
    p.n_kv_grp = hp.n_heads / hp.n_kv_heads;
    p.n_ctx_orig = hp.rope.n_ctx_orig;
    p.rope_theta = hp.rope_theta;
    p.rope_beta_fast = hp.rope.beta_fast;
    p.rope_beta_slow = hp.rope.beta_slow;
    p.rope_freq_scale = hp.rope.freq_scale;
    p.rope_ext_factor = hp.rope.ext_factor;
    p.rope_attn_factor = hp.rope.attn_factor;
    p.attn_scale = hp.attention_multiplier;
    p.gqa_mode = core_attn::GQA_NATIVE;
    return p;
}

// Per-layer weights. NAR's `granite_nle_llm_block` uses `attn_o_w`;
// granite_speech's `granite_llm_block` uses `attn_out_w`. The caller
// fills the matching field — they are functionally identical.
struct LayerWeights {
    ggml_tensor* attn_norm_w;
    ggml_tensor* attn_q_w;
    ggml_tensor* attn_k_w;
    ggml_tensor* attn_v_w;
    ggml_tensor* attn_out_w;
    ggml_tensor* ffn_norm_w;
    ggml_tensor* ffn_gate_w;
    ggml_tensor* ffn_up_w;
    ggml_tensor* ffn_down_w;
};

// Non-causal flash attention, no KV cache. Mirrors `nle_llm_attn_noncausal`:
// flash_attn_ext over the full (T, T) tile with mask=nullptr; native GQA
// expansion is handled by flash. Used by granite-nle's editing pass.
static inline ggml_tensor* attn_noncausal(ggml_context* ctx0, ggml_tensor* x, const LayerWeights& b,
                                          ggml_tensor* positions, int n_q, int n_kv, int hd, float rope_theta,
                                          float attn_scale, const RopeScaling& rope) {
    const int T = (int)x->ne[1];

    ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x);
    ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x);
    ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, x);

    Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, rope.n_ctx_orig, rope_theta,
                      rope.freq_scale, rope.ext_factor, rope.attn_factor, rope.beta_fast, rope.beta_slow);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, rope.n_ctx_orig, rope_theta,
                      rope.freq_scale, rope.ext_factor, rope.attn_factor, rope.beta_fast, rope.beta_slow);

    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

    return ggml_mul_mat(ctx0, b.attn_out_w, attn);
}

// Build the 40-layer Granite-1B decoder forward.
//
//   inputs_embeds : (d, T) F32 — pre-built embeddings; the builder applies
//                   hp.embedding_multiplier × inputs_embeds as the first op.
//   positions     : (T,) I32 — absolute positions
//   causal_mask   : (n_past + T, T) FP16 mask, or nullptr. Required for
//                   prefill in the causal+KV path; nullptr for single-token
//                   decode (T == 1) and for the entire non-causal path.
//   kv_k, kv_v    : per-layer KV cache, ne = (hd, max_ctx, n_kv, n_layers).
//                   Pass nullptr for the non-causal path; n_past must then be 0.
//   n_past        : KV cache write offset (causal+KV path); 0 for non-causal.
//   blocks        : per-layer weight handles, length hp.n_layers.
//   output_norm_w : final RMSNorm scale.
//   is_causal     : true → KV-cached path via core_attn::kv_self_attn;
//                   false → non-causal flash, no cache.
//
// Returns the post-final-RMSNorm hidden tensor (d, T). Caller is responsible
// for slicing + applying its LM head + any logits scaling.
static inline ggml_tensor* build_decoder(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* inputs_embeds,
                                         ggml_tensor* positions, ggml_tensor* causal_mask, ggml_tensor* kv_k,
                                         ggml_tensor* kv_v, int n_past, const std::vector<LayerWeights>& blocks,
                                         ggml_tensor* output_norm_w, const Hparams& hp, bool is_causal,
                                         int fixed_kv_len = 0, ggml_tensor* kv_indices = nullptr) {
    const int n_q = hp.n_heads;
    const int n_kv = hp.n_kv_heads;
    const int hd = hp.head_dim;

    ggml_tensor* cur = ggml_scale(ctx0, inputs_embeds, hp.embedding_multiplier);
    ggml_set_name(cur, "emb_scaled");

    const auto kvp = attention_params(hp);

    for (int il = 0; il < hp.n_layers; il++) {
        const auto& b = blocks[il];
        ggml_tensor* residual = cur;

        cur = ggml_rms_norm(ctx0, cur, hp.rms_eps);
        cur = ggml_mul(ctx0, cur, b.attn_norm_w);

        ggml_tensor* attn;
        if (is_causal) {
            attn = core_attn::kv_self_attn(ctx0, gf, cur, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_out_w,
                                           /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask, kv_k,
                                           kv_v, il, n_past, kvp,
                                           /*qkv_w=*/nullptr, fixed_kv_len, kv_indices);
        } else {
            attn =
                attn_noncausal(ctx0, cur, b, positions, n_q, n_kv, hd, hp.rope_theta, hp.attention_multiplier, hp.rope);
        }

        cur = ggml_add(ctx0, residual, ggml_scale(ctx0, attn, hp.residual_multiplier));

        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, hp.rms_eps);
        cur = ggml_mul(ctx0, cur, b.ffn_norm_w);
        cur = core_ffn::swiglu(ctx0, cur, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, hp.residual_multiplier));
    }

    cur = ggml_rms_norm(ctx0, cur, hp.rms_eps);
    cur = ggml_mul(ctx0, cur, output_norm_w);
    return cur;
}

} // namespace core_granite_llm
