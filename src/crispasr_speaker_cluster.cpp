// crispasr_speaker_cluster.cpp — agglomerative single-linkage cosine
// clustering. See header for semantics.

#include "crispasr_speaker_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

float dot(const float* a, const float* b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++)
        s += a[i] * b[i];
    return s;
}

} // namespace

std::vector<int> crispasr_agglomerative_cluster(const std::vector<float>& embeddings, int n, int dim,
                                                float merge_threshold, int max_speakers) {
    std::vector<int> labels(n, -1);
    if (n <= 0 || dim <= 0 || (int)embeddings.size() < n * dim)
        return labels;
    if (n == 1) {
        labels[0] = 0;
        return labels;
    }
    if (max_speakers < 1)
        max_speakers = 1;

    // Each input starts as its own cluster represented by its index in
    // the embedding array. clusters[c] holds the list of input indices
    // currently assigned to cluster c.
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; i++)
        clusters[i].push_back(i);
    std::vector<bool> active(n, true);

    // Pairwise cosine sim cache, NxN upper triangle. Single-linkage
    // means a cluster-to-cluster sim is the MAX over all pairs of
    // member input indices, so we can keep operating on the original
    // sim matrix and look up the inter-cluster max on the fly.
    std::vector<float> sim((size_t)n * n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            const float s = dot(&embeddings[i * dim], &embeddings[j * dim], dim);
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }

    auto inter_cluster_sim = [&](int ca, int cb) -> float {
        float best = -1.0f;
        for (int i : clusters[ca]) {
            for (int j : clusters[cb]) {
                const float s = sim[i * n + j];
                if (s > best)
                    best = s;
            }
        }
        return best;
    };

    auto count_active = [&]() {
        int n_active = 0;
        for (bool b : active)
            if (b)
                n_active++;
        return n_active;
    };

    // Merge greedily. Two stopping conditions:
    //   * No active pair remains above merge_threshold (normal stop).
    //   * count drops to 1 (degenerate stop).
    // max_speakers is enforced separately: while count > max_speakers
    // we keep merging the closest pair even if it would otherwise have
    // fallen below threshold. This is the typical "hard cap" behavior
    // diarization pipelines expose.
    while (count_active() > 1) {
        float best = -1.0f;
        int best_a = -1, best_b = -1;
        for (int a = 0; a < n; a++) {
            if (!active[a])
                continue;
            for (int b = a + 1; b < n; b++) {
                if (!active[b])
                    continue;
                const float s = inter_cluster_sim(a, b);
                if (s > best) {
                    best = s;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        if (best_a < 0)
            break;
        const bool force_merge = count_active() > max_speakers;
        if (!force_merge && best < merge_threshold)
            break;
        // Merge b into a; deactivate b.
        for (int idx : clusters[best_b])
            clusters[best_a].push_back(idx);
        clusters[best_b].clear();
        active[best_b] = false;
    }

    // Assign cluster IDs in first-appearance order over the input
    // indices: the first input gets cluster 0, the next input not yet
    // in cluster 0 gets cluster 1, and so on. This keeps the labels
    // stable for callers that care about determinism.
    std::vector<int> input_to_cluster(n, -1);
    for (int c = 0; c < n; c++) {
        if (!active[c])
            continue;
        for (int idx : clusters[c])
            input_to_cluster[idx] = c;
    }
    std::vector<int> cluster_remap(n, -1);
    int next_id = 0;
    for (int i = 0; i < n; i++) {
        const int c = input_to_cluster[i];
        if (c < 0)
            continue;
        if (cluster_remap[c] < 0)
            cluster_remap[c] = next_id++;
        labels[i] = cluster_remap[c];
    }
    return labels;
}
