// logit_sampler.h — temperature / top-k / top-p token sampling over raw logits.
//
// Mirrors the order HF transformers' generate(do_sample=True) applies its
// logits warpers: temperature, then top-k, then top-p (nucleus). Header-only
// and weight-free so it is unit-tested directly (tests/test-core-logit-sampler.cpp).
// Non-finite logits are never selected. Returns -1 when no logit is finite so
// callers can abort the decode the same way their argmax path does.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace core_logit_sampler {

struct Params {
    float temperature = 1.0f; // must be > 0; callers keep greedy for <= 0
    int top_k = 0;            // <= 0 disables
    float top_p = 1.0f;       // >= 1 (or <= 0) disables
};

inline int sample(const float* logits, int vocab, const Params& p, std::mt19937_64& rng) {
    std::vector<int> ids;
    ids.reserve((size_t)vocab);
    for (int i = 0; i < vocab; ++i)
        if (std::isfinite(logits[i]))
            ids.push_back(i);
    if (ids.empty())
        return -1;
    const float inv_t = 1.0f / (p.temperature > 0.0f ? p.temperature : 1.0f);
    // Highest logit first; ties broken by token id so results are reproducible.
    auto by_logit = [logits](int a, int b) {
        const float la = logits[a];
        const float lb = logits[b];
        if (la != lb)
            return la > lb;
        return a < b;
    };
    size_t keep = ids.size();
    if (p.top_k > 0 && (size_t)p.top_k < keep) {
        std::partial_sort(ids.begin(), ids.begin() + p.top_k, ids.end(), by_logit);
        keep = (size_t)p.top_k;
    } else {
        std::sort(ids.begin(), ids.end(), by_logit);
    }
    const double max_l = (double)logits[ids[0]] * inv_t;
    std::vector<double> w(keep);
    double total = 0.0;
    for (size_t j = 0; j < keep; ++j) {
        w[j] = std::exp((double)logits[ids[j]] * inv_t - max_l);
        total += w[j];
    }
    if (p.top_p > 0.0f && p.top_p < 1.0f) {
        // Smallest prefix whose probability mass reaches top_p; always keeps
        // the most likely token (HF min_tokens_to_keep = 1).
        double cum = 0.0;
        size_t n = 0;
        while (n < keep) {
            cum += w[n] / total;
            ++n;
            if (cum >= (double)p.top_p)
                break;
        }
        keep = n;
        total = 0.0;
        for (size_t j = 0; j < keep; ++j)
            total += w[j];
    }
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng);
    for (size_t j = 0; j < keep; ++j) {
        r -= w[j];
        if (r <= 0.0)
            return ids[j];
    }
    return ids[keep - 1];
}

} // namespace core_logit_sampler
