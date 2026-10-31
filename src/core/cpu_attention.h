// src/core/cpu_attention.h — vectorizable CPU attention primitives (header-only).
//
// Hoisted from firered_asr.cpp (PR #229) so any backend that runs attention as
// hand-written CPU loops — rather than delegating to the ggml graph — can reuse
// one tested implementation:
//
//   firered_asr.cpp    — conformer rel_shift encoder + per-beam decoder attn
//   wav2vec2-ggml.cpp  — encoder self-attention (alignment path)
//
// Most backends express attention as ggml tensors (ggml_soft_max_ext /
// ggml_flash_attn_ext) and get SIMD + threading + GPU offload for free — they do
// NOT need this. These helpers exist only for the CPU-loop backends above.

#pragma once

#include <algorithm> // std::fill_n
#include <cmath>     // expf, INFINITY

namespace crispasr {
namespace cpu_attn {

// Dot product of two length-K float vectors using four independent accumulator
// chains so the compiler can vectorize the reduction even under strict FP (no
// -ffast-math / /fp:fast needed). Float accumulation over K~1280 is well within
// tolerance for ASR logits; the old double-accumulate form forced scalar,
// non-vectorized code and dominated the decoder.
static inline float cpu_dot(const float* a, const float* b, int K) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        s0 += a[k + 0] * b[k + 0];
        s1 += a[k + 1] * b[k + 1];
        s2 += a[k + 2] * b[k + 2];
        s3 += a[k + 3] * b[k + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; k < K; k++)
        s += a[k] * b[k];
    return s;
}

// Flash-style online softmax over T keys, fused with the value accumulation:
//
//   out[0..hd) = softmax_t( score_fn(t) ) · V[t]
//
// but without materializing the T-sized score/probability vector. `V` is the
// value matrix with row stride `stride_v` (V[t] = V + t*stride_v). `score_fn(t)`
// returns the (already scaled) attention logit for key t. `out` is fully written
// (zero-initialized internally), so no caller-side clear is needed.
//
// m tracks the running max logit; l tracks the running normalized exp-sum.
template <typename ScoreFn>
static inline void cpu_online_softmax_accumulate(int T, int hd, const float* __restrict V, int stride_v,
                                                 float* __restrict out, ScoreFn&& score_fn) {
    float m = -INFINITY;
    float l = 0.0f;
    std::fill_n(out, hd, 0.0f);
    for (int t = 0; t < T; t++) {
        const float* __restrict v = V + t * stride_v;

        float s = score_fn(t);
        float m_new = (s > m) ? s : m;
        float exp_scale = expf(m - m_new);
        float e = expf(s - m_new);

        // If a new maximum is found, previous accumulated values must be
        // rescaled. Otherwise exp_scale == 1 and the multiply can be skipped.
        if (m_new > m) {
            int dd = 0;
            for (; dd + 4 <= hd; dd += 4) {
                out[dd + 0] = out[dd + 0] * exp_scale + e * v[dd + 0];
                out[dd + 1] = out[dd + 1] * exp_scale + e * v[dd + 1];
                out[dd + 2] = out[dd + 2] * exp_scale + e * v[dd + 2];
                out[dd + 3] = out[dd + 3] * exp_scale + e * v[dd + 3];
            }
            for (; dd < hd; dd++)
                out[dd] = out[dd] * exp_scale + e * v[dd];
        } else {
            int dd = 0;
            for (; dd + 4 <= hd; dd += 4) {
                out[dd + 0] += e * v[dd + 0];
                out[dd + 1] += e * v[dd + 1];
                out[dd + 2] += e * v[dd + 2];
                out[dd + 3] += e * v[dd + 3];
            }
            for (; dd < hd; dd++)
                out[dd] += e * v[dd];
        }

        l = l * exp_scale + e;
        m = m_new;
    }

    float inv_l = 1.0f / l;
    int dd = 0;
    for (; dd + 4 <= hd; dd += 4) {
        out[dd + 0] *= inv_l;
        out[dd + 1] *= inv_l;
        out[dd + 2] *= inv_l;
        out[dd + 3] *= inv_l;
    }
    for (; dd < hd; dd++)
        out[dd] *= inv_l;
}

} // namespace cpu_attn
} // namespace crispasr
