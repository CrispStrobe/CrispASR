// src/core/rvq.cpp — Euclidean RVQ encode (CPU). See core/rvq.h.

#include "rvq.h"

#include <cstdio>

namespace core_rvq {

bool encode_euclidean(const float* features, int T, int dim, const Codebook* stages, int n_stages, int32_t* codes_out) {
    if (!features || !stages || !codes_out || T <= 0 || dim <= 0 || n_stages <= 0)
        return false;
    for (int s = 0; s < n_stages; s++) {
        if (!stages[s].embed || !stages[s].embed_norm_sq || stages[s].dim != dim || stages[s].codebook_size <= 0) {
            fprintf(stderr, "core_rvq: stage %d malformed (embed=%p, norm=%p, dim=%d, K=%d)\n", s,
                    (const void*)stages[s].embed, (const void*)stages[s].embed_norm_sq, stages[s].dim,
                    stages[s].codebook_size);
            return false;
        }
    }

    // Working residual: copy of features, mutated stage-by-stage.
    std::vector<float> residual(features, features + (size_t)T * dim);

    for (int s = 0; s < n_stages; s++) {
        const auto& cb = stages[s];
        const int K = cb.codebook_size;
        const float* E = cb.embed;
        const float* En = cb.embed_norm_sq;

        for (int t = 0; t < T; t++) {
            float* x = residual.data() + (size_t)t * dim;
            // argmin_k ||x - E[k]||^2 = argmax_k (2 x·E[k] - ||E[k]||^2)
            int best = 0;
            float best_score = -1e30f;
            for (int k = 0; k < K; k++) {
                const float* e = E + (size_t)k * dim;
                float dot = 0.0f;
                for (int j = 0; j < dim; j++)
                    dot += x[j] * e[j];
                float score = 2.0f * dot - En[k];
                if (score > best_score) {
                    best_score = score;
                    best = k;
                }
            }
            codes_out[(size_t)t * n_stages + s] = best;
            const float* e = E + (size_t)best * dim;
            for (int j = 0; j < dim; j++)
                x[j] -= e[j];
        }
    }
    return true;
}

} // namespace core_rvq
