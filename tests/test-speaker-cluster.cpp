// Unit tests for crispasr_agglomerative_cluster (#107 P3b).
//
// We drive the clusterer with synthetic L2-normalized embeddings whose
// pairwise structure is known a priori, so the test is deterministic
// and needs no model load / no audio.

#include "../src/crispasr_speaker_cluster.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <set>
#include <vector>

namespace {

// L2-normalize a vector in place.
void l2norm(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v)
        s += (double)x * x;
    if (s <= 0.0)
        return;
    const float inv = (float)(1.0 / std::sqrt(s));
    for (float& x : v)
        x *= inv;
}

// Build N points around a center direction with controlled noise.
// `noise` ~ 0.0 means points lie exactly on the center; ~1.0 means
// orthogonal noise dominates.
std::vector<float> make_cluster_around(const std::vector<float>& center, int n, float noise, uint32_t seed) {
    const int dim = (int)center.size();
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> out;
    out.reserve((size_t)n * (size_t)dim);
    for (int i = 0; i < n; i++) {
        std::vector<float> v = center;
        for (int d = 0; d < dim; d++)
            v[d] += noise * g(rng);
        l2norm(v);
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// Concatenate cluster point sets.
std::vector<float> concat(std::vector<std::vector<float>> blocks) {
    std::vector<float> out;
    for (auto& b : blocks)
        out.insert(out.end(), b.begin(), b.end());
    return out;
}

// Count distinct labels (ignoring -1).
int distinct_labels(const std::vector<int>& labels) {
    std::set<int> s;
    for (int l : labels)
        if (l >= 0)
            s.insert(l);
    return (int)s.size();
}

constexpr int DIM = 8;

} // namespace

TEST_CASE("cluster: single embedding -> cluster 0", "[unit][diarize][cluster]") {
    std::vector<float> emb = {1, 0, 0, 0, 0, 0, 0, 0};
    auto labels = crispasr_agglomerative_cluster(emb, 1, DIM, 0.5f, 8);
    REQUIRE(labels.size() == 1);
    REQUIRE(labels[0] == 0);
}

TEST_CASE("cluster: two well-separated speakers -> 2 clusters", "[unit][diarize][cluster]") {
    std::vector<float> center_a = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> center_b = {0, 1, 0, 0, 0, 0, 0, 0};
    auto a = make_cluster_around(center_a, 5, 0.05f, 42);
    auto b = make_cluster_around(center_b, 5, 0.05f, 7);
    auto all = concat({a, b});

    auto labels = crispasr_agglomerative_cluster(all, 10, DIM, 0.5f, 8);
    REQUIRE(labels.size() == 10);
    REQUIRE(distinct_labels(labels) == 2);
    // Each block of 5 should share a label.
    for (int i = 0; i < 5; i++)
        REQUIRE(labels[i] == labels[0]);
    for (int i = 5; i < 10; i++)
        REQUIRE(labels[i] == labels[5]);
    REQUIRE(labels[0] != labels[5]);
}

TEST_CASE("cluster: three speakers -> 3 clusters", "[unit][diarize][cluster]") {
    std::vector<float> ca = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> cb = {0, 1, 0, 0, 0, 0, 0, 0};
    std::vector<float> cc = {0, 0, 1, 0, 0, 0, 0, 0};
    auto a = make_cluster_around(ca, 4, 0.05f, 11);
    auto b = make_cluster_around(cb, 4, 0.05f, 22);
    auto c = make_cluster_around(cc, 4, 0.05f, 33);
    auto all = concat({a, b, c});

    auto labels = crispasr_agglomerative_cluster(all, 12, DIM, 0.5f, 8);
    REQUIRE(distinct_labels(labels) == 3);
    REQUIRE(labels[0] == labels[1]);
    REQUIRE(labels[4] == labels[5]);
    REQUIRE(labels[8] == labels[9]);
    REQUIRE(labels[0] != labels[4]);
    REQUIRE(labels[4] != labels[8]);
}

TEST_CASE("cluster: max_speakers caps the cluster count", "[unit][diarize][cluster]") {
    std::vector<float> ca = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> cb = {0, 1, 0, 0, 0, 0, 0, 0};
    std::vector<float> cc = {0, 0, 1, 0, 0, 0, 0, 0};
    auto a = make_cluster_around(ca, 3, 0.05f, 11);
    auto b = make_cluster_around(cb, 3, 0.05f, 22);
    auto c = make_cluster_around(cc, 3, 0.05f, 33);
    auto all = concat({a, b, c});

    // High threshold would normally yield 3 clusters; max_speakers=2
    // forces a merge.
    auto labels = crispasr_agglomerative_cluster(all, 9, DIM, 0.5f, /*max_speakers=*/2);
    REQUIRE(distinct_labels(labels) <= 2);
}

TEST_CASE("cluster: low threshold merges noisy near-duplicates", "[unit][diarize][cluster]") {
    // Two centers fairly close together, plus moderate noise. Low
    // threshold should merge them; high threshold should keep them
    // apart.
    std::vector<float> ca = {0.9f, 0.1f, 0, 0, 0, 0, 0, 0};
    std::vector<float> cb = {0.7f, 0.3f, 0, 0, 0, 0, 0, 0};
    auto a = make_cluster_around(ca, 3, 0.05f, 1);
    auto b = make_cluster_around(cb, 3, 0.05f, 2);
    auto all = concat({a, b});

    // High threshold: should keep 2 clusters.
    auto labels_high = crispasr_agglomerative_cluster(all, 6, DIM, 0.95f, 8);
    // Low threshold: should merge.
    auto labels_low = crispasr_agglomerative_cluster(all, 6, DIM, 0.5f, 8);
    REQUIRE(distinct_labels(labels_low) <= distinct_labels(labels_high));
}

TEST_CASE("cluster: cluster IDs are first-appearance ordered", "[unit][diarize][cluster]") {
    // Inputs alternate between two speakers. The first input must
    // map to cluster 0, the next *different* speaker to cluster 1.
    std::vector<float> ca = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> cb = {0, 1, 0, 0, 0, 0, 0, 0};
    auto a1 = make_cluster_around(ca, 1, 0.02f, 100);
    auto b1 = make_cluster_around(cb, 1, 0.02f, 200);
    auto a2 = make_cluster_around(ca, 1, 0.02f, 300);
    auto b2 = make_cluster_around(cb, 1, 0.02f, 400);
    auto all = concat({a1, b1, a2, b2});

    auto labels = crispasr_agglomerative_cluster(all, 4, DIM, 0.5f, 8);
    REQUIRE(labels[0] == 0);
    REQUIRE(labels[1] == 1);
    REQUIRE(labels[2] == 0); // back to speaker A
    REQUIRE(labels[3] == 1);
}

TEST_CASE("cluster: invalid inputs are no-ops", "[unit][diarize][cluster]") {
    auto labels = crispasr_agglomerative_cluster({}, 0, DIM, 0.5f, 8);
    REQUIRE(labels.empty());

    std::vector<float> emb(DIM, 0.0f);
    auto labels2 = crispasr_agglomerative_cluster(emb, 1, /*dim=*/0, 0.5f, 8);
    REQUIRE(labels2.size() == 1);
    REQUIRE(labels2[0] == -1); // bogus dim -> no cluster assigned

    // Mismatched embedding count vs declared n.
    std::vector<float> emb_short(DIM, 0.0f);
    auto labels3 = crispasr_agglomerative_cluster(emb_short, 5, DIM, 0.5f, 8);
    REQUIRE(labels3.size() == 5);
    // All -1 because the function bails early when sizes don't add up.
    for (int l : labels3)
        REQUIRE(l == -1);
}
