// core/sdpa.h — scaled dot-product attention with a manual F32 default and
// fused ggml_flash_attn_ext as an opt-in.
//
// Why this exists: ggml's fused ggml_flash_attn_ext accumulates the Q·Kᵀ
// product in F16, and on flash kernels that ignore the GGML_PREC_F32 hint
// (observed on P100/sm_60) the hint is a no-op — the fused path then computes
// attention at lower precision than the code claims. That is invisible to
// robust ASR encoders but breaks precision-sensitive AR / flow-matching stacks
// (f5's DiT NaN'd; nemotron's RNNT flipped tokens). The fix, proven on f5_tts
// and nemotron, is to DEFAULT to a manual mul_mat / soft_max / mul_mat SDPA in
// F32 (correct on every backend) and make fused flash opt-in behind a per-call
// env gate. This header is that helper, so backends stop hand-rolling it.
//
// Header-only; depends on ggml.h alone.

#pragma once

#include "ggml.h"

#include <cstdlib>

namespace core_sdpa {

// Read a per-call "use fused flash" gate. NEVER cache in a function-local
// static: a cached read is fixed for the process, so flipping the env on a
// live context would change nothing and an A/B would silently compare a path
// against itself (a bug this repo has shipped). Default OFF → manual F32.
inline bool flash_enabled(const char* env_var) {
    const char* e = std::getenv(env_var);
    return e && e[0] && e[0] != '0';
}

// Scaled dot-product attention: returns softmax(scale·QKᵀ + mask)·V reshaped to
// (head_dim·n_heads, T_q), matching ggml_flash_attn_ext's head-major output so
// a caller's downstream out-projection is unchanged whichever path runs.
//
//   Q:    (head_dim, T_q,  n_heads)  — permuted views (as flash consumes) or contiguous
//   K, V: (head_dim, T_kv, n_heads)
//   mask: optional additive term, F16 or F32, shape (T_kv, T_q) broadcast over
//         heads OR (T_kv, T_q, n_heads) per-head; nullptr for no mask. Same
//         semantics as ggml_flash_attn_ext's mask: it is ADDED after QKᵀ is
//         scaled, so a pre-scaled bias (e.g. conformer rel-pos) is passed as-is.
//   d:    head_dim · n_heads (the flattened model dim of the output)
//
// use_flash: true → fused ggml_flash_attn_ext (F16 KQ accumulation; opt-in);
//            false → manual F32 SDPA (the correct default).
inline ggml_tensor* attn(ggml_context* ctx, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V, ggml_tensor* mask, int d,
                         int T_q, float scale, bool use_flash) {
    if (use_flash) {
        // flash requires an F16 mask; cast if the caller kept it F32.
        ggml_tensor* m = mask;
        if (m && m->type != GGML_TYPE_F16) {
            m = ggml_cast(ctx, m, GGML_TYPE_F16);
        }
        ggml_tensor* out = ggml_flash_attn_ext(ctx, Q, K, V, m, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32); // honoured only on some devices; harmless elsewhere
        return ggml_reshape_2d(ctx, out, d, T_q);
    }
    // Manual F32 SDPA. cont the (possibly permuted) Q/K so mul_mat is safe on
    // every backend; the fused path deliberately skips these copies, the
    // correctness path pays them.
    ggml_tensor* Kc = ggml_cont(ctx, K);
    ggml_tensor* Qc = ggml_cont(ctx, Q);
    ggml_tensor* scores = ggml_mul_mat(ctx, Kc, Qc); // (T_kv, T_q, n_heads) = QKᵀ per head
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_scale(ctx, scores, scale); // scale·QKᵀ
    if (mask) {
        // soft_max wants an F32 additive term for the correctness path; a
        // (T_kv, T_q) 2-D mask broadcasts over heads, a per-head 3-D one adds
        // elementwise. -inf entries mask to ~0 through the softmax.
        ggml_tensor* mf = (mask->type == GGML_TYPE_F32) ? mask : ggml_cast(ctx, mask, GGML_TYPE_F32);
        scores = ggml_add(ctx, scores, mf);
    }
    scores = ggml_soft_max(ctx, scores);                                    // F32 softmax over the key axis
    ggml_tensor* V_perm = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3)); // (T_kv, head_dim, n_heads)
    ggml_tensor* attn = ggml_mul_mat(ctx, V_perm, scores);                  // (head_dim, T_q, n_heads)
    ggml_mul_mat_set_prec(attn, GGML_PREC_F32);
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // (head_dim, n_heads, T_q)
    return ggml_reshape_2d(ctx, attn, d, T_q);
}

} // namespace core_sdpa
