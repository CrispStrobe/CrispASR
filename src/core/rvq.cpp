// src/core/rvq.cpp — Euclidean RVQ encode (CPU). See core/rvq.h.

#include "rvq.h"
#include "core/parallel_for.h"

#include <cstdio>
#include <cstdlib>

namespace core_rvq {

bool encode_euclidean(const float* features, int T, int dim, const Codebook* stages, int n_stages, int32_t* codes_out,
                      int n_threads) {
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

    uint64_t work_per_frame = 0;
    for (int s = 0; s < n_stages; s++)
        work_per_frame = std::max(work_per_frame, (uint64_t)stages[s].codebook_size * dim);

    // Frames are independent across every stage: keep a frame on one worker
    // through all residual updates instead of creating a new team per stage.
    // The per-codebook dot products and first-wins argmax order are unchanged.
    core_parallel::for_each_chunk_work(T, force_serial ? 1 : n_threads, work_per_frame, 65536, [&](int begin, int end) {
        for (int t = begin; t < end; t++) {
            float* x = residual.data() + (size_t)t * dim;
            for (int s = 0; s < n_stages; s++) {
                const auto& cb = stages[s];
                int best = 0;
                float best_score = -1e30f;
                for (int k = 0; k < cb.codebook_size; k++) {
                    const float* e = cb.embed + (size_t)k * dim;
                    float dot = 0.0f;
                    for (int j = 0; j < dim; j++)
                        dot += x[j] * e[j];
                    const float score = 2.0f * dot - cb.embed_norm_sq[k];
                    if (score > best_score) {
                        best_score = score;
                        best = k;
                    }
                }
                codes_out[(size_t)t * n_stages + s] = best;
                const float* e = cb.embed + (size_t)best * dim;
                for (int j = 0; j < dim; j++)
                    x[j] -= e[j];
            }
        }
    });
    return true;
}

bool encode_euclidean_per_stage(const float* features, int T, int dim, const float* const* embeds, const int* sizes,
                                int n_stages, std::vector<std::vector<int32_t>>& out_codes, int n_threads) {
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
        core_parallel::for_each_chunk_work(K, force_serial ? 1 : n_threads, (uint64_t)dim, 65536,
                                           [&](int begin, int end) {
                                               for (int k = begin; k < end; k++) {
                                                   const float* e = embeds[s] + (size_t)k * dim;
                                                   float n = 0.0f;
                                                   // Preserve scalar accumulation order. Grouping four products
                                                   // changes F32 rounding and can change codes near a tie.
                                                   for (int j = 0; j < dim; j++)
                                                       n += e[j] * e[j];
                                                   norms[s][k] = n;
                                               }
                                           });
        stages[s] = {embeds[s], norms[s].data(), sizes[s], dim};
    }

    std::vector<int32_t> flat((size_t)T * n_stages, 0);
    if (!encode_euclidean(features, T, dim, stages.data(), n_stages, flat.data(), n_threads))
        return false;

    // Transpose (T, n_stages) → out_codes[s][t].
    out_codes.assign((size_t)n_stages, std::vector<int32_t>((size_t)T, 0));
    for (int t = 0; t < T; t++)
        for (int s = 0; s < n_stages; s++)
            out_codes[(size_t)s][(size_t)t] = flat[(size_t)t * n_stages + s];
    return true;
}

} // namespace core_rvq
