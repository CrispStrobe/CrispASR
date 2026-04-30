// src/core/attention.h — shared multi-head attention helpers (header-only).
//
// Replaces the Q/K/V-projection + reshape + RoPE + GQA-expand + flash-attn +
// output-projection block that every LLM-based model in src/ has 1–2 copies
// of. The helper is header-only so the compiler inlines it straight into
// each caller, producing the exact same ggml op sequence as the original
// inline code and preserving bit-identical graph execution.
//
// Scope of the initial version (this commit):
//
//   core_attn::llama_self_attn_kv()  — the classic Llama / Mistral LLM
//     attention block: RMSNorm weights applied by caller, no biases on
//     Q/K/V/O, NEOX RoPE, optional GQA expansion, ggml_flash_attn_ext with
//     a caller-supplied causal-or-sliding-window mask, reshape + output
//     projection. Used by voxtral, voxtral4b, qwen3 (without Q/K norm),
//     and granite LLM decoders.
//
// Follow-up variants (to be added when their first consumer migrates):
//
//   * post-projection Q/K RMSNorm (qwen3)
//   * separate audio-encoder variant with biases + no RoPE (voxtral audio)
//   * adaptive scale / residual_multiplier (granite µP)
//   * KV-cache lookup that returns a (K, V) pair instead of taking
//     pre-permuted inputs (needed when KV cache is stored in a different
//     layout than (head_dim, T, n_heads))
//
// This staged approach keeps the first commit narrow and verifiable. Every
// new caller either fits the existing helper or adds a new sibling helper
// with its own regression test.

#pragma once

#include "ggml.h"

#include <cstddef>

namespace core_attn {

// Parameters that differ from call to call. Everything here is a plain
// value type so the compiler can inline the caller's constants into the
// helper's ggml_* op chain.
struct LlamaSelfAttnParams {
    int n_heads;    // query heads (== n_kv_heads for MHA)
    int n_kv_heads; // key/value heads; with GQA, n_heads / n_kv_heads > 1
    int head_dim;   // per-head dimension
    int n_kv_grp;   // == n_heads / n_kv_heads (caller precomputes)
    int n_ctx_orig; // rope n_ctx_orig (usually llm_max_pos)
    float rope_theta;
    float attn_scale; // usually 1 / sqrt(head_dim); pass explicitly
};

// Llama / Mistral-style self-attention with optional GQA, NEOX RoPE, and
// ggml_flash_attn_ext.
//
// Inputs:
//   x              [d_model, T]  — RMSNormed input for this layer (the
//                                   caller does the norm + the learned
//                                   scale multiplication)
//   q_w,k_w,v_w    Q/K/V weight tensors (no biases in the LLM case)
//   o_w            output projection weight
//   positions      [T]           — RoPE position ids
//   mask           [ctx, T] F16  — causal / sliding-window mask or nullptr
//                                   for no-mask (voxtral audio case)
//
// Output:
//   attn           [d_model, T]  — post-output-projection tensor. The
//                                   caller adds it to the residual.
// Fused QKV variant: if qkv_w is non-null, do a single matmul and split.
// qkv_w shape: [d_model, n_q*hd + 2*n_kv*hd] — concatenated Q, K, V weights.
// Falls back to 3 separate matmuls when qkv_w is null (backward compat).
static inline ggml_tensor* llama_self_attn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* q_w, ggml_tensor* k_w,
                                           ggml_tensor* v_w, ggml_tensor* o_w, ggml_tensor* positions,
                                           ggml_tensor* mask, const LlamaSelfAttnParams& p,
                                           ggml_tensor* qkv_w = nullptr) {
    const int hd = p.head_dim;
    const int n_q = p.n_heads;
    const int n_kv = p.n_kv_heads;
    const int n_ctx = p.n_ctx_orig;
    const int grp = p.n_kv_grp;

    ggml_tensor* Q;
    ggml_tensor* K;
    ggml_tensor* V;

    if (qkv_w) {
        // Single fused matmul: one mul_mat instead of three.
        // qkv_w: [d_model, q_dim + k_dim + v_dim]
        // Output: [q_dim + k_dim + v_dim, T]
        ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, x);
        const int q_dim = n_q * hd;
        const int kv_dim = n_kv * hd;
        const int T = (int)x->ne[1];
        // Split along ne[0]: Q=[0..q_dim), K=[q_dim..q_dim+kv_dim), V=[q_dim+kv_dim..)
        Q = ggml_view_2d(ctx, qkv, q_dim, T, qkv->nb[1], 0);
        K = ggml_view_2d(ctx, qkv, kv_dim, T, qkv->nb[1], q_dim * ggml_type_size(qkv->type));
        V = ggml_view_2d(ctx, qkv, kv_dim, T, qkv->nb[1], (q_dim + kv_dim) * ggml_type_size(qkv->type));
    } else {
        // Standard 3 separate matmuls (backward compat).
        Q = ggml_mul_mat(ctx, q_w, x);
        K = ggml_mul_mat(ctx, k_w, x);
        V = ggml_mul_mat(ctx, v_w, x);
    }

    // T is the time dim of x; ggml stores [d_model, T] as ne = [d_model, T].
    const int T = (int)x->ne[1];

    Q = ggml_reshape_3d(ctx, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx, V, hd, n_kv, T);

    // NEOX RoPE on Q and K. Same args as the original inline code in
    // voxtral / voxtral4b / qwen3 / granite LLM blocks.
    Q = ggml_rope_ext(ctx, Q, positions, /*freq_factors*/ nullptr, hd, GGML_ROPE_TYPE_NEOX, n_ctx, p.rope_theta,
                      /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f,
                      /*attn_factor*/ 1.0f, /*beta_fast*/ 32.0f, /*beta_slow*/ 1.0f);
    K = ggml_rope_ext(ctx, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, n_ctx, p.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f,
                      1.0f);

    // GQA expansion: replicate each KV head `grp` times along a new dim so
    // K/V have n_heads rows instead of n_kv_heads, then flatten back.
    if (grp > 1) {
        ggml_tensor* K4 = ggml_reshape_4d(ctx, K, hd, 1, n_kv, T);
        ggml_tensor* V4 = ggml_reshape_4d(ctx, V, hd, 1, n_kv, T);
        K4 = ggml_repeat_4d(ctx, K4, hd, grp, n_kv, T);
        V4 = ggml_repeat_4d(ctx, V4, hd, grp, n_kv, T);
        K = ggml_cont(ctx, ggml_reshape_3d(ctx, K4, hd, n_q, T));
        V = ggml_cont(ctx, ggml_reshape_3d(ctx, V4, hd, n_q, T));
    }

    // Permute to flash-attention layout: (head_dim, T, n_heads).
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Flash attention. Output shape = (head_dim, n_heads, T, 1).
    ggml_tensor* attn =
        ggml_flash_attn_ext(ctx, Q, K, V, mask, p.attn_scale, /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);

    // Back to (d_model, T).
    attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);

    // Output projection (no bias).
    return ggml_mul_mat(ctx, o_w, attn);
}

// ---------------------------------------------------------------------------
// Encoder self-attention — biased Q/K/V/O projections, optional RoPE.
//
// Covers architectures like the Whisper audio encoder (voxtral 3B) and the
// causal RoPE+SwiGLU audio encoder (voxtral4b). Key differences from the
// LLM llama_self_attn():
//   - Q, K, V, O projections can each have an optional bias (nullptr = skip)
//   - RoPE is optional: pass positions == nullptr to skip
//   - GQA expansion is included for architectures that use it
//
// The caller still handles the pre-attention norm and post-attention
// residual add.
// ---------------------------------------------------------------------------

struct EncoderSelfAttnParams {
    int n_heads;    // query heads
    int n_kv_heads; // key/value heads (usually == n_heads for encoders)
    int head_dim;
    int n_kv_grp;     // n_heads / n_kv_heads (1 for MHA)
    float attn_scale; // usually 1/sqrt(head_dim)
    // RoPE params (only used when positions != nullptr)
    int n_ctx_orig;
    float rope_theta;
    // When true (default), wrap ggml_permute() in ggml_cont() before
    // flash_attn_ext. voxtral 3B needs this; voxtral4b does not (its
    // encoder was written without cont and changing it would alter the
    // ggml graph structure). Set to false for voxtral4b compatibility.
    bool permute_cont = true;
};

static inline ggml_tensor* encoder_self_attn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* q_w, ggml_tensor* q_b,
                                             ggml_tensor* k_w, ggml_tensor* k_b, ggml_tensor* v_w, ggml_tensor* v_b,
                                             ggml_tensor* o_w, ggml_tensor* o_b, ggml_tensor* positions,
                                             ggml_tensor* mask, const EncoderSelfAttnParams& p) {
    const int hd = p.head_dim;
    const int n_q = p.n_heads;
    const int n_kv = p.n_kv_heads;
    const int grp = p.n_kv_grp;
    const int T = (int)x->ne[1];

    // Q/K/V projections with optional biases.
    ggml_tensor* Q = ggml_mul_mat(ctx, q_w, x);
    if (q_b)
        Q = ggml_add(ctx, Q, q_b);
    ggml_tensor* K = ggml_mul_mat(ctx, k_w, x);
    if (k_b)
        K = ggml_add(ctx, K, k_b);
    ggml_tensor* V = ggml_mul_mat(ctx, v_w, x);
    if (v_b)
        V = ggml_add(ctx, V, v_b);

    Q = ggml_reshape_3d(ctx, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx, V, hd, n_kv, T);

    // Optional RoPE (skip for encoders with learned positional embeddings).
    if (positions) {
        Q = ggml_rope_ext(ctx, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, p.n_ctx_orig, p.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, p.n_ctx_orig, p.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
    }

    // GQA expansion (when n_kv_heads < n_heads).
    if (grp > 1) {
        ggml_tensor* K4 = ggml_reshape_4d(ctx, K, hd, 1, n_kv, T);
        ggml_tensor* V4 = ggml_reshape_4d(ctx, V, hd, 1, n_kv, T);
        K4 = ggml_repeat_4d(ctx, K4, hd, grp, n_kv, T);
        V4 = ggml_repeat_4d(ctx, V4, hd, grp, n_kv, T);
        K = ggml_cont(ctx, ggml_reshape_3d(ctx, K4, hd, n_q, T));
        V = ggml_cont(ctx, ggml_reshape_3d(ctx, V4, hd, n_q, T));
    }

    // Permute to flash-attention layout: (head_dim, T, n_heads).
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);
    if (p.permute_cont) {
        Q = ggml_cont(ctx, Q);
        K = ggml_cont(ctx, K);
        V = ggml_cont(ctx, V);
    }

    // Flash attention (bidirectional if mask==nullptr, causal/SWA otherwise).
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, mask, p.attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);

    // Output projection with optional bias.
    attn = ggml_mul_mat(ctx, o_w, attn);
    if (o_b)
        attn = ggml_add(ctx, attn, o_b);
    return attn;
}

// ---------------------------------------------------------------------------
// KV-cached self-attention for the LLM decoders (qwen3-asr, voxtral,
// voxtral4b, granite-speech).
//
// Replaces the Q/K/V-proj + [optional Q/K norm] + RoPE + persistent-KV-cache
// write + cache read + [manual GQA expansion] + flash-attn-ext + output-proj
// block that each of the four models has its own copy of. The helper does
// NOT do the pre-attention RMSNorm or the post-attention residual add —
// callers do those inline so the helper stays focused on the attention
// block proper (which is where the per-model knobs live).
//
// KV cache layout convention: ne = (head_dim, max_ctx, n_kv_heads, n_layers).
// Every consumer already stores its cache this way, which is why this helper
// is shareable in the first place.
// ---------------------------------------------------------------------------

// GQA expansion strategy.
//
// qwen3 and voxtral (3b) manually expand each KV head into `n_kv_grp` query
// heads via reshape_4d -> repeat_4d -> reshape_3d, and then wrap the final
// reshape in ggml_cont. voxtral4b also manually expands, but skips the final
// ggml_cont. granite skips the manual expansion entirely and relies on
// ggml_flash_attn_ext's native GQA support (pass Kfull/Vfull with n_kv heads
// directly, flash-attn handles the repeat internally).
//
// Each mode produces slightly different graph ops. Picking the wrong one
// breaks bit-identity on the regression sweep, so this is an explicit knob.
enum GqaMode {
    GQA_MANUAL_CONT = 0,   // reshape_4d / repeat_4d / reshape_3d + ggml_cont
    GQA_MANUAL_NOCONT = 1, // reshape_4d / repeat_4d / reshape_3d, no final cont
    GQA_NATIVE = 2,        // no expansion; flash_attn_ext handles GQA itself
};

struct KvSelfAttnParams {
    int n_heads;    // query heads
    int n_kv_heads; // key/value heads (== n_heads for MHA)
    int head_dim;   // per-head dimension
    int n_kv_grp;   // n_heads / n_kv_heads (caller precomputes)
    int n_ctx_orig; // RoPE n_ctx_orig (some models 0, some llm_max_pos)
    float rope_theta;
    float rope_beta_fast; // RoPE extrapolation beta_fast (qwen3/voxtral: 32, others: 0)
    float rope_beta_slow; // RoPE extrapolation beta_slow (qwen3/voxtral: 1,  others: 0)
    float attn_scale;     // usually 1/sqrt(head_dim); granite uses µP scale
    float qk_norm_eps;    // RMSNorm epsilon for optional Q/K norm (qwen3); unused otherwise
    GqaMode gqa_mode;
    int rope_type = GGML_ROPE_TYPE_NEOX; // NEOX for most models, NORMAL for fairseq2/omniasr
    // Partial-rotary RoPE: number of head-dim entries to rotate. The
    // remaining `head_dim - n_rot` entries pass through unchanged. Used
    // by Gemma4 full-attention layers (`partial_rotary_factor=0.25`,
    // i.e. n_rot = head_dim/4) and Phi-3-style models. Default 0 means
    // rotate the entire head_dim — matches every existing caller.
    int n_rot = 0;
    // Apply RMSNorm-without-learned-weight to V before the cache write.
    // Gemma4's `v_norm` is constructed with `with_scale=False`, i.e.
    // RMSNorm with no learned scale tensor — there is no weight to load,
    // we just need to run the normalisation op on V. Default false → no
    // op, matches every other consumer.
    bool v_rms_norm = false;
};

// KV-cached self-attention. Writes the new K/V into the persistent cache
// slice at [n_past, n_past + T) for layer `il`, then reads the full history
// [0, n_past + T) back out and runs flash-attention against Q.
//
// Inputs:
//   x            [d_model, T]  — pre-attention normalized activations
//   q_w,k_w,v_w  projection weights (no biases for the Llama case)
//   o_w          output projection weight (no bias)
//   q_norm_w     [head_dim] Q-norm weight, or nullptr to skip (non-qwen3)
//   k_norm_w     [head_dim] K-norm weight, or nullptr to skip
//   positions    [T] I32 — absolute positions n_past, n_past+1, ...
//   causal_mask  [Lk, T] F16 or nullptr (decode path uses nullptr)
//   kv_k, kv_v   persistent cache, ne = (hd, max_ctx, n_kv, n_layers)
//   il           layer index into the cache's trailing dim
//   n_past       number of tokens already in the cache
//
// Output:
//   attn         [d_model, T] — post-output-projection tensor. Caller adds
//                                it to the residual.
// fixed_kv_len > 0: override the KV-read length (Lk) to a constant, keeping
// topology identical across calls with different n_past.  Unwritten slots are
// masked to -inf by causal_mask so they never affect output.
static inline ggml_tensor* kv_self_attn(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* x, ggml_tensor* q_w,
                                        ggml_tensor* k_w, ggml_tensor* v_w, ggml_tensor* o_w, ggml_tensor* q_norm_w,
                                        ggml_tensor* k_norm_w, ggml_tensor* positions, ggml_tensor* causal_mask,
                                        ggml_tensor* kv_k, ggml_tensor* kv_v, int il, int n_past,
                                        const KvSelfAttnParams& p, ggml_tensor* qkv_w = nullptr,
                                        int fixed_kv_len = 0) {
    const int hd = p.head_dim;
    const int n_q = p.n_heads;
    const int n_kv = p.n_kv_heads;
    const int grp = p.n_kv_grp;
    const int T = (int)x->ne[1];
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    // ---- Q/K/V projections ----
    ggml_tensor* Q;
    ggml_tensor* K;
    ggml_tensor* V;

    if (qkv_w) {
        // Fused: one matmul, then split output. The 2D views below are
        // strided (each T-row leaves gaps for the other Q/K/V), so for T>1
        // the downstream ggml_reshape_3d would fail its contiguity assert.
        // ggml_cont materialises each into its own contiguous buffer; for
        // T=1 the cont is a no-op (single row is already contiguous).
        ggml_tensor* qkv = ggml_mul_mat(ctx0, qkv_w, x);
        const int q_dim = n_q * hd;
        const int kv_dim = n_kv * hd;
        const size_t ts = ggml_type_size(qkv->type);
        Q = ggml_view_2d(ctx0, qkv, q_dim, T, qkv->nb[1], 0);
        K = ggml_view_2d(ctx0, qkv, kv_dim, T, qkv->nb[1], q_dim * ts);
        V = ggml_view_2d(ctx0, qkv, kv_dim, T, qkv->nb[1], (q_dim + kv_dim) * ts);
        if (T > 1) {
            Q = ggml_cont(ctx0, Q);
            K = ggml_cont(ctx0, K);
            V = ggml_cont(ctx0, V);
        }
    } else {
        Q = ggml_mul_mat(ctx0, q_w, x);
        K = ggml_mul_mat(ctx0, k_w, x);
        V = ggml_mul_mat(ctx0, v_w, x);
    }

    Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

    // ---- Optional Q/K RMSNorm (qwen3) ----
    if (q_norm_w) {
        Q = ggml_rms_norm(ctx0, Q, p.qk_norm_eps);
        Q = ggml_mul(ctx0, Q, q_norm_w);
    }
    if (k_norm_w) {
        K = ggml_rms_norm(ctx0, K, p.qk_norm_eps);
        K = ggml_mul(ctx0, K, k_norm_w);
    }

    // ---- Optional V RMSNorm without learned weight (gemma4) ----
    // gemma4's v_norm is `Gemma4RMSNorm(head_dim, with_scale=False)`,
    // so there is no weight tensor — we just normalise V along its
    // last (head_dim) axis, exactly as ggml_rms_norm does.
    if (p.v_rms_norm) {
        V = ggml_rms_norm(ctx0, V, p.qk_norm_eps);
    }

    // ---- RoPE (NEOX for most models, NORMAL for fairseq2/omniasr) ----
    // p.n_rot > 0 selects partial-rotary mode (only the first n_rot
    // entries of each head are rotated; the rest pass through). 0
    // means rotate the entire head_dim, which matches every existing
    // caller's prior behaviour.
    const int n_rot = p.n_rot > 0 ? p.n_rot : hd;
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, n_rot, p.rope_type, p.n_ctx_orig, p.rope_theta,
                      /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f,
                      /*attn_factor*/ 1.0f, p.rope_beta_fast, p.rope_beta_slow);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, n_rot, p.rope_type, p.n_ctx_orig, p.rope_theta, 1.0f, 0.0f, 1.0f,
                      p.rope_beta_fast, p.rope_beta_slow);

    // ---- Permute new K/V to (hd, T, n_kv) for cache write ----
    ggml_tensor* K_new_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
    ggml_tensor* V_new_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

    // ---- Write into the persistent KV cache at [n_past, n_past+T) ----
    ggml_tensor* k_view = ggml_view_4d(ctx0, kv_k, hd, T, n_kv, 1, kv_k->nb[1], kv_k->nb[2], kv_k->nb[3],
                                       (size_t)il * kv_k->nb[3] + (size_t)n_past * kv_k->nb[1]);
    ggml_tensor* v_view = ggml_view_4d(ctx0, kv_v, hd, T, n_kv, 1, kv_v->nb[1], kv_v->nb[2], kv_v->nb[3],
                                       (size_t)il * kv_v->nb[3] + (size_t)n_past * kv_v->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));

    // ---- Read full K/V history from cache ----
    ggml_tensor* Kfull =
        ggml_cont(ctx0, ggml_view_3d(ctx0, kv_k, hd, Lk, n_kv, kv_k->nb[1], kv_k->nb[2], (size_t)il * kv_k->nb[3]));
    ggml_tensor* Vfull =
        ggml_cont(ctx0, ggml_view_3d(ctx0, kv_v, hd, Lk, n_kv, kv_v->nb[1], kv_v->nb[2], (size_t)il * kv_v->nb[3]));

    // ---- GQA expansion ----
    if (p.gqa_mode != GQA_NATIVE && grp > 1) {
        ggml_tensor* K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
        ggml_tensor* V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
        K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, grp, n_kv);
        V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, grp, n_kv);
        if (p.gqa_mode == GQA_MANUAL_CONT) {
            Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, Lk, n_q));
            Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, Lk, n_q));
        } else {
            Kfull = ggml_reshape_3d(ctx0, K4, hd, Lk, n_q);
            Vfull = ggml_reshape_3d(ctx0, V4, hd, Lk, n_q);
        }
    }

    // ---- Permute Q to (hd, T, n_q) for flash-attn ----
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

    // ---- Flash attention + reshape + output projection ----
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask, p.attn_scale, /*max_bias*/ 0.0f,
                                            /*logit_softcap*/ 0.0f);
    attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

    return ggml_mul_mat(ctx0, o_w, attn);
}

} // namespace core_attn
