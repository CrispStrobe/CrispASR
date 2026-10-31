// src/core/rvq.cpp — Euclidean RVQ encode (CPU). See core/rvq.h.

#include "rvq.h"
#include "core/parallel_for.h"

#include <cstdio>
#include <cstdlib>

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

    const char* env_serial = std::getenv("CRISPASR_RVQ_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');

    if (force_serial || T <= 2) {
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
    } else {
        for (int s = 0; s < n_stages; s++) {
            const auto& cb = stages[s];
            const int K = cb.codebook_size;
            const float* E = cb.embed;
            const float* En = cb.embed_norm_sq;

            core_parallel::for_each_chunk(T, 0, [&](int t_start, int t_end) {
                for (int t = t_start; t < t_end; t++) {
                    float* x = residual.data() + (size_t)t * dim;
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
            });
        }
    }
    return true;
}

bool encode_euclidean_per_stage(const float* features, int T, int dim, const float* const* embeds, const int* sizes,
                                int n_stages, std::vector<std::vector<int32_t>>& out_codes) {
    if (!features || !embeds || !sizes || T <= 0 || dim <= 0 || n_stages <= 0)
        return false;

    const char* env_serial = std::getenv("CRISPASR_RVQ_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');

    // Precompute ‖E[k]‖² per stage + build Codebook views.
    std::vector<std::vector<float>> norms(n_stages);
    std::vector<Codebook> stages(n_stages);
    for (int s = 0; s < n_stages; s++) {
        if (!embeds[s] || sizes[s] <= 0)
            return false;
        const int K = sizes[s];
        norms[s].resize((size_t)K);
        if (force_serial || K <= 16) {
            for (int k = 0; k < K; k++) {
                const float* e = embeds[s] + (size_t)k * dim;
                float n = 0.0f;
                for (int j = 0; j < dim; j++)
                    n += e[j] * e[j];
                norms[s][k] = n;
            }
        } else {
            core_parallel::for_each_chunk(K, 0, [&](int k_start, int k_end) {
                for (int k = k_start; k < k_end; k++) {
                    const float* e = embeds[s] + (size_t)k * dim;
                    float n = 0.0f;
                    int j = 0;
                    for (; j + 3 < dim; j += 4)
                        n += e[j] * e[j] + e[j + 1] * e[j + 1] + e[j + 2] * e[j + 2] + e[j + 3] * e[j + 3];
                    for (; j < dim; j++)
                        n += e[j] * e[j];
                    norms[s][k] = n;
                }
            });
        }
        stages[s] = {embeds[s], norms[s].data(), sizes[s], dim};
    }

    std::vector<int32_t> flat((size_t)T * n_stages, 0);
    if (!encode_euclidean(features, T, dim, stages.data(), n_stages, flat.data()))
        return false;

    // Transpose (T, n_stages) → out_codes[s][t].
    out_codes.assign((size_t)n_stages, std::vector<int32_t>((size_t)T, 0));
    for (int t = 0; t < T; t++)
        for (int s = 0; s < n_stages; s++)
            out_codes[(size_t)s][(size_t)t] = flat[(size_t)t * n_stages + s];
    return true;
}

} // namespace core_rvq
