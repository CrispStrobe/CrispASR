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

#include <algorithm>
#include <cmath>
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

// -----------------------------------------------------------------------
// CTC prefix beam search with optional gamma-threshold pruning.
//
// Standard CTC prefix beam search (Graves & Jaitly 2014) extended with
// MAES-style gamma pruning: at each frame, only keep hypotheses whose
// score is within `gamma` of the best. This gives beam-search quality
// at near-greedy speed for typical beam sizes (4–8).
//
// The algorithm maintains two probability channels per prefix:
//   p_b(y) — probability of y ending in blank at time t
//   p_nb(y) — probability of y ending in non-blank at time t
// Total prefix probability: p(y) = p_b(y) + p_nb(y)
//
// Parameters:
//   logprobs  : (T, V) row-major — log-softmax CTC output
//   T         : number of time steps
//   V         : vocab size including blank
//   blank_id  : blank label index
//   shift     : added to every surviving id (e.g. -1 to map CTC→LM ids)
//   beam_size : max hypotheses to keep per frame
//   gamma     : pruning threshold (0 = no pruning). Hypotheses with
//               score < best - gamma are dropped. 2.0–3.0 is typical.
//
// Returns the best hypothesis as a token id sequence.
// -----------------------------------------------------------------------

struct BeamResult {
    std::vector<int32_t> tokens; // decoded token ids (after shift)
    double score;                // log-probability of best hypothesis
};

static inline BeamResult prefix_beam_search(const float* logprobs, int T, int V, int blank_id, int shift,
                                            int beam_size = 4, float gamma = 0.0f) {
    // Hypothesis: a prefix and its blank/non-blank log-probs.
    // We use vectors as prefix keys (small — typically < 100 tokens).
    struct Hyp {
        std::vector<int32_t> prefix;
        double p_b;  // log-prob of paths ending in blank
        double p_nb; // log-prob of paths ending in non-blank
        double score() const {
            // log-sum-exp of p_b and p_nb
            double m = (p_b > p_nb) ? p_b : p_nb;
            return m + std::log(std::exp(p_b - m) + std::exp(p_nb - m));
        }
    };

    const double NEG_INF = -1e30;

    // Initial state: empty prefix, blank prob = 1 (log = 0), non-blank = 0
    std::vector<Hyp> beam(1);
    beam[0].p_b = 0.0;
    beam[0].p_nb = NEG_INF;

    for (int t = 0; t < T; t++) {
        const float* lp = logprobs + (size_t)t * V; // log-probs at frame t

        // Collect new hypotheses in a map keyed by prefix.
        // Using a flat vector + linear search since beam is small.
        struct NewHyp {
            std::vector<int32_t> prefix;
            double p_b;
            double p_nb;
            double score() const {
                double m = (p_b > p_nb) ? p_b : p_nb;
                return m + std::log(std::exp(p_b - m) + std::exp(p_nb - m));
            }
        };
        std::vector<NewHyp> next;
        next.reserve(beam.size() * 2); // rough estimate

        auto find_or_insert = [&](const std::vector<int32_t>& pfx) -> NewHyp& {
            for (auto& h : next)
                if (h.prefix == pfx)
                    return h;
            next.push_back({pfx, NEG_INF, NEG_INF});
            return next.back();
        };

        // log-sum-exp helper
        auto logaddexp = [](double a, double b) -> double {
            if (a == -1e30)
                return b;
            if (b == -1e30)
                return a;
            double m = (a > b) ? a : b;
            return m + std::log(std::exp(a - m) + std::exp(b - m));
        };

        for (auto& h : beam) {
            double p_total = h.score();

            // 1. Extend with blank
            {
                auto& nh = find_or_insert(h.prefix);
                nh.p_b = logaddexp(nh.p_b, p_total + (double)lp[blank_id]);
            }

            // 2. Extend with each non-blank token
            // For efficiency, only consider top-k tokens at this frame
            // when beam is small. For now, iterate all (V is typically
            // 1K–8K which is fine for CPU).
            for (int c = 0; c < V; c++) {
                if (c == blank_id)
                    continue;

                double lp_c = (double)lp[c];

                // Prefix extension
                auto new_pfx = h.prefix;
                bool is_repeat = (!h.prefix.empty() && h.prefix.back() == c);

                if (is_repeat) {
                    // Repeat of last char: only extend from blank-ending paths
                    // (non-blank ending with same char would be a collapsed repeat)
                    auto& nh = find_or_insert(new_pfx);
                    nh.p_nb = logaddexp(nh.p_nb, h.p_b + lp_c);

                    // Also allow extending from non-blank (which merges into same prefix)
                    // This handles the case: "aa" where both a's are separate emissions
                    new_pfx.push_back(c);
                    auto& nh2 = find_or_insert(new_pfx);
                    nh2.p_nb = logaddexp(nh2.p_nb, h.p_nb + lp_c);
                } else {
                    new_pfx.push_back(c);
                    auto& nh = find_or_insert(new_pfx);
                    nh.p_nb = logaddexp(nh.p_nb, p_total + lp_c);
                }
            }
        }

        // Prune: sort by score, keep top beam_size
        // Apply gamma pruning first if enabled
        if (gamma > 0.0f && !next.empty()) {
            double best_score = NEG_INF;
            for (auto& h : next) {
                double s = h.score();
                if (s > best_score)
                    best_score = s;
            }
            double threshold = best_score - (double)gamma;
            size_t write = 0;
            for (size_t i = 0; i < next.size(); i++) {
                if (next[i].score() >= threshold) {
                    if (write != i)
                        next[write] = std::move(next[i]);
                    write++;
                }
            }
            next.resize(write);
        }

        // Keep top beam_size
        if ((int)next.size() > beam_size) {
            std::partial_sort(next.begin(), next.begin() + beam_size, next.end(),
                              [](const NewHyp& a, const NewHyp& b) { return a.score() > b.score(); });
            next.resize(beam_size);
        }

        // Move to beam for next frame
        beam.resize(next.size());
        for (size_t i = 0; i < next.size(); i++) {
            beam[i].prefix = std::move(next[i].prefix);
            beam[i].p_b = next[i].p_b;
            beam[i].p_nb = next[i].p_nb;
        }
    }

    // Return best hypothesis
    if (beam.empty())
        return {{}, NEG_INF};
    int best = 0;
    for (int i = 1; i < (int)beam.size(); i++)
        if (beam[i].score() > beam[best].score())
            best = i;

    // Apply shift to token ids
    std::vector<int32_t> tokens;
    tokens.reserve(beam[best].prefix.size());
    for (int32_t id : beam[best].prefix)
        tokens.push_back(id + shift);
    return {std::move(tokens), beam[best].score()};
}

} // namespace core_ctc
