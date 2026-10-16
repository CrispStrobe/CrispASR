#pragma once

// Private Dia sampler shared by the runtime and its probability regression tests.
#include "dia_exponential_rng.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

static uint32_t dia_sample_token(const float* logits, uint32_t vocab_size, float temperature, float top_p, int top_k,
                                 std::mt19937& rng, const float* exponential_noise = nullptr) {
    if (temperature <= 0.0f) {
        // Greedy
        return (uint32_t)(std::max_element(logits, logits + vocab_size) - logits);
    }

    // Apply temperature
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = (logits[i] - max_logit) / temperature;
    }

    // Softmax
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(probs[i]);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // Top-k filter
    if (top_k > 0 && top_k < (int)vocab_size) {
        // Select exactly k temperature-scaled logit indices, as the original
        // sampler does. Softmax rounding can collapse distinct logits into
        // equal probabilities; a probability threshold then retains extra tokens.
        std::vector<std::pair<float, uint32_t>> ranked_logits(vocab_size);
        for (uint32_t i = 0; i < vocab_size; i++) {
            ranked_logits[i] = {logits[i] / temperature, i};
        }
        const auto greater = [](const auto& a, const auto& b) { return a.first > b.first; };
        if (static_cast<uint64_t>(top_k) * 64 <= vocab_size) {
            std::partial_sort(ranked_logits.begin(), ranked_logits.begin() + top_k, ranked_logits.end(), greater);
        } else {
            std::nth_element(ranked_logits.begin(), ranked_logits.begin() + top_k - 1, ranked_logits.end(), greater);
        }
        std::vector<bool> keep(vocab_size, false);
        for (int i = 0; i < top_k; ++i)
            keep[ranked_logits[i].second] = true;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (!keep[i])
                probs[i] = 0.0f;
        }
        // Re-normalize
        sum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++)
            sum += probs[i];
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < vocab_size; i++)
                probs[i] /= sum;
        }
    }

    // Top-p (nucleus) filter
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, uint32_t>> sorted_probs(vocab_size);
        for (uint32_t i = 0; i < vocab_size; i++) {
            sorted_probs[i] = {probs[i], i};
        }
        std::sort(sorted_probs.begin(), sorted_probs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        float cumsum = 0.0f;
        for (auto& [p, idx] : sorted_probs) {
            // Keep the token that crosses top_p, including the most likely
            // token when its probability already exceeds top_p. The official
            // sampler shifts its cumulative-probability removal mask by one.
            const float prob = p;
            if (cumsum > top_p) {
                p = 0.0f;
            }
            cumsum += prob;
        }
        for (auto& [p, idx] : sorted_probs) {
            probs[idx] = p;
        }
        // Re-normalize
        sum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++)
            sum += probs[i];
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < vocab_size; i++)
                probs[i] /= sum;
        }
    }

    if (exponential_noise) {
        return dia_rng::exponential_race(probs.data(), exponential_noise, vocab_size);
    }

    // Sample
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}
