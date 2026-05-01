// src/core/ctc.h — CTC-side helpers shared by the granite family.
//
// Two primitives that today live only in granite_nle but generalise to
// any CTC-tail backend:
//
//   * posterior_weighted_pool — the importance-weighted temporal pool
//     used by granite-nle's BPE auxiliary head. Each window's row is a
//     weighted sum of frame features where the weight is the per-frame
//     non-blank posterior. Windows that fall partly past T_in are
//     handled as zero-padded. See modeling_ctc.posterior_weighted_pool
//     upstream.
//
//   * greedy_decode_with_blank — argmax → unique_consecutive (collapse
//     repeats) → drop blanks → apply an additive shift. The shift exists
//     because CTC label IDs and the downstream LM / BPE token IDs are
//     usually offset by one (label 0 = blank, the rest correspond to
//     LM IDs starting at 0 → shift of -1). Sits next to
//     `core/greedy_decode.h` (which is the autoregressive LLM loop —
//     different shape, similar role).
//
// Header-only `static inline` so each caller's existing CTC tail keeps
// inlined codegen and stays bit-identical.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace core_ctc {

// ceil(T_in / pool_window): the number of pooled rows produced by
// posterior_weighted_pool. Exposed so the caller can size its output
// buffer before the call.
static inline int num_windows_for(int T_in, int pool_window) {
    const int pad_len = (pool_window - T_in % pool_window) % pool_window;
    return (T_in + pad_len) / pool_window;
}

// Posterior-weighted temporal pool.
//
//   pooled[w, :] = sum_{j in [0, pool_window)} (importance[t] / S_w) * hidden[t, :]
//   S_w          = sum_{j, t<T_in} importance[t] + 1e-8
//   t            = w * pool_window + j  (frames with t >= T_in are zero-padded)
//
//   hidden      : (T_in, d) row-major
//   importance  : (T_in,) — typically (1 - blank_prob)
//   T_in        : valid frame count
//   d           : per-frame feature dim
//   pool_window : window size in frames (>= 1)
//   out         : (num_windows, d) row-major, where
//                 num_windows = ceil(T_in / pool_window).
//                 Caller sizes the buffer.
//
// Returns num_windows.
static inline int posterior_weighted_pool(const float* hidden, const float* importance, int T_in, int d,
                                          int pool_window, float* out) {
    const int pad_len = (pool_window - T_in % pool_window) % pool_window;
    const int T_pad = T_in + pad_len;
    const int num_windows = T_pad / pool_window;
    std::memset(out, 0, (size_t)num_windows * d * sizeof(float));
    for (int w = 0; w < num_windows; w++) {
        float sum_imp = 0.0f;
        for (int j = 0; j < pool_window; j++) {
            int t = w * pool_window + j;
            if (t < T_in)
                sum_imp += importance[t];
        }
        const float denom = sum_imp + 1e-8f;
        float* dst = out + (size_t)w * d;
        for (int j = 0; j < pool_window; j++) {
            int t = w * pool_window + j;
            if (t >= T_in)
                break;
            const float weight = importance[t] / denom;
            const float* src = hidden + (size_t)t * d;
            for (int i = 0; i < d; i++)
                dst[i] += weight * src[i];
        }
    }
    return num_windows;
}

// CTC greedy decode: argmax → collapse repeats → drop blanks → apply shift.
//
//   logits   : (T, V) row-major
//   T        : time steps
//   V        : vocab (CTC label cardinality, including blank)
//   blank_id : the blank label index (typically 0)
//   shift    : added to every surviving id before it is emitted. Use
//              shift = -1 to convert CTC labels {0..V-1} where 0=blank
//              to LM token IDs starting at 0; use shift = 0 if labels
//              are already LM-aligned.
//
// Returns the decoded id sequence (may be empty).
static inline std::vector<int32_t> greedy_decode_with_blank(const float* logits, int T, int V, int blank_id,
                                                            int shift) {
    std::vector<int32_t> argmax((size_t)T);
    for (int t = 0; t < T; t++) {
        const float* row = logits + (size_t)t * V;
        int best = 0;
        float bestv = row[0];
        for (int v = 1; v < V; v++) {
            if (row[v] > bestv) {
                bestv = row[v];
                best = v;
            }
        }
        argmax[t] = best;
    }
    std::vector<int32_t> collapsed;
    collapsed.reserve(argmax.size());
    for (size_t i = 0; i < argmax.size(); i++) {
        if (i == 0 || argmax[i] != argmax[i - 1])
            collapsed.push_back(argmax[i]);
    }
    std::vector<int32_t> ids;
    ids.reserve(collapsed.size());
    for (int32_t id : collapsed) {
        if (id == blank_id)
            continue;
        ids.push_back(id + shift);
    }
    return ids;
}

} // namespace core_ctc
