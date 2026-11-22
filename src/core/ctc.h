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
#include <unordered_map>
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
    // CTC prefix beam search (Hannun et al. 2014). For each hypothesis prefix
    // we track p_b / p_nb: the log-probability of all paths that produce the
    // prefix and end in a blank / in its last token. Extending with token c:
    //   c == blank            -> same prefix, p_b  += total
    //   c == last token       -> same prefix, p_nb += p_nb   (repeat collapses)
    //                            prefix + c,  p_nb += p_b    (blank separates)
    //   otherwise             -> prefix + c,  p_nb += total
    //
    // Prefixes are nodes of a trie (parent, token); a candidate is keyed by
    // its (parent, token) pair, so the same prefix reached by different paths
    // merges without copying or hashing token vectors. Only beam survivors
    // become trie nodes, so the trie grows by at most beam_size per frame.
    const double NEG_INF = -1e30;
    auto logaddexp = [&](double a, double b) -> double {
        if (a <= NEG_INF)
            return b;
        if (b <= NEG_INF)
            return a;
        const double m = (a > b) ? a : b;
        return m + std::log(std::exp(a - m) + std::exp(b - m));
    };

    struct Node {
        int parent;
        int32_t token;
    };
    std::vector<Node> trie = {{-1, -1}}; // node 0: the empty prefix
    std::unordered_map<uint64_t, int> children;
    auto key_of = [](int parent, int32_t token) -> uint64_t {
        return ((uint64_t)(uint32_t)(parent + 1) << 32) | (uint32_t)token;
    };

    struct Hyp {
        int node;
        double p_b, p_nb;
    };
    struct Cand {
        int parent; // trie node of the prefix without its last token (-1 for the empty prefix)
        int32_t token;
        double p_b, p_nb;
    };
    auto score = [&](double p_b, double p_nb) { return logaddexp(p_b, p_nb); };

    std::vector<Hyp> beam = {{0, 0.0, NEG_INF}};
    std::vector<Cand> next;
    std::unordered_map<uint64_t, size_t> index;

    for (int t = 0; t < T; t++) {
        const float* lp = logprobs + (size_t)t * V;
        next.clear();
        index.clear();
        index.reserve(beam.size() * (size_t)V);

        auto cand = [&](int parent, int32_t token) -> Cand& {
            const uint64_t k = key_of(parent, token);
            auto it = index.find(k);
            if (it != index.end())
                return next[it->second];
            index.emplace(k, next.size());
            next.push_back({parent, token, NEG_INF, NEG_INF});
            return next.back();
        };

        for (const Hyp& h : beam) {
            const Node& n = trie[h.node];
            const double total = score(h.p_b, h.p_nb);

            Cand& same = cand(n.parent, n.token);
            same.p_b = logaddexp(same.p_b, total + (double)lp[blank_id]);
            if (h.node != 0)
                same.p_nb = logaddexp(same.p_nb, h.p_nb + (double)lp[n.token]);

            for (int c = 0; c < V; c++) {
                if (c == blank_id)
                    continue;
                const double from = (h.node != 0 && c == n.token) ? h.p_b : total;
                if (from <= NEG_INF)
                    continue;
                Cand& ext = cand(h.node, c);
                ext.p_nb = logaddexp(ext.p_nb, from + (double)lp[c]);
            }
        }

        if (gamma > 0.0f && !next.empty()) {
            double best = NEG_INF;
            for (const Cand& c : next)
                best = std::max(best, score(c.p_b, c.p_nb));
            const double threshold = best - (double)gamma;
            next.erase(std::remove_if(next.begin(), next.end(),
                                      [&](const Cand& c) { return score(c.p_b, c.p_nb) < threshold; }),
                       next.end());
        }
        if ((int)next.size() > beam_size) {
            std::partial_sort(next.begin(), next.begin() + beam_size, next.end(), [&](const Cand& a, const Cand& b) {
                return score(a.p_b, a.p_nb) > score(b.p_b, b.p_nb);
            });
            next.resize(beam_size);
        }

        beam.clear();
        for (const Cand& c : next) {
            int node;
            if (c.parent < 0) {
                node = 0;
            } else {
                const uint64_t k = key_of(c.parent, c.token);
                auto it = children.find(k);
                if (it == children.end()) {
                    node = (int)trie.size();
                    trie.push_back({c.parent, c.token});
                    children.emplace(k, node);
                } else {
                    node = it->second;
                }
            }
            beam.push_back({node, c.p_b, c.p_nb});
        }
    }

    if (beam.empty())
        return {{}, NEG_INF};
    size_t best = 0;
    for (size_t i = 1; i < beam.size(); i++)
        if (score(beam[i].p_b, beam[i].p_nb) > score(beam[best].p_b, beam[best].p_nb))
            best = i;

    std::vector<int32_t> tokens;
    for (int node = beam[best].node; node > 0; node = trie[node].parent)
        tokens.push_back(trie[node].token + shift);
    std::reverse(tokens.begin(), tokens.end());
    return {std::move(tokens), score(beam[best].p_b, beam[best].p_nb)};
}

} // namespace core_ctc
