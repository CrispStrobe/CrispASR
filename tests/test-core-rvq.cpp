// test-core-rvq.cpp — unit tests for core_rvq::encode_euclidean.
//
// core_rvq is the shared Euclidean RVQ encoder (kyutai_stt / mimo_tokenizer;
// §176l wants kyutai's scalar loop routed through it). It replaces the naive
// argmin_k ||x - E[k]||^2 with the algebraically-equal 2·x·E[k] - ||E[k]||^2
// shootout (drops the per-frame-constant ||x||^2). This test proves that
// rewrite selects the SAME codes as a direct full-distance reference — the
// correctness property a Kyutai/Mimi model round-trip would otherwise be needed
// to check, verified here deterministically with no model.

#include <catch2/catch_test_macros.hpp>

#include "core/rvq.h"
#include "scoped_test_env.h"

#include <cmath>
#include <random>
#include <vector>

namespace {

// Direct reference: argmin over the FULL distance in double precision, with the
// same frame-outer / stage-by-stage residual subtraction core_rvq uses.
std::vector<int32_t> reference_encode(const std::vector<float>& features, int T, int dim,
                                      const std::vector<core_rvq::Codebook>& stages) {
    const int n_stages = (int)stages.size();
    std::vector<int32_t> codes((size_t)T * n_stages, 0);
    std::vector<double> residual(features.begin(), features.end());
    for (int s = 0; s < n_stages; s++) {
        const auto& cb = stages[s];
        for (int t = 0; t < T; t++) {
            double* x = residual.data() + (size_t)t * dim;
            int best = 0;
            double best_d = 1e300;
            for (int k = 0; k < cb.codebook_size; k++) {
                const float* e = cb.embed + (size_t)k * dim;
                double d = 0.0;
                for (int j = 0; j < dim; j++) {
                    double diff = x[j] - (double)e[j];
                    d += diff * diff;
                }
                if (d < best_d) {
                    best_d = d;
                    best = k;
                }
            }
            codes[(size_t)t * n_stages + s] = best;
            const float* e = cb.embed + (size_t)best * dim;
            for (int j = 0; j < dim; j++)
                x[j] -= (double)e[j];
        }
    }
    return codes;
}

// True full-distance from a frame's residual to codebook entry k (double).
double true_dist(const double* x, const float* e, int dim) {
    double d = 0.0;
    for (int j = 0; j < dim; j++) {
        double diff = x[j] - (double)e[j];
        d += diff * diff;
    }
    return d;
}

} // namespace

TEST_CASE("core_rvq::encode_euclidean matches full-distance reference", "[unit][rvq]") {
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // A spread of shapes covering typical Mimi/Kyutai RVQ (dim 16–32, K up to
    // 512, several stages) plus small edge shapes.
    struct Shape {
        int T, dim, K, n_stages;
    };
    const Shape shapes[] = {
        {1, 4, 4, 1}, {8, 16, 32, 4}, {32, 32, 256, 8}, {4, 8, 512, 2}, {17, 24, 100, 3},
    };

    long total = 0, mismatches = 0, tie_mismatches = 0;

    for (const auto& sh : shapes) {
        std::vector<float> features((size_t)sh.T * sh.dim);
        for (auto& f : features)
            f = gauss(rng);

        // Random codebooks + cached ||E[k]||^2 (F32, exactly as callers stage them).
        std::vector<std::vector<float>> embeds(sh.n_stages), norms(sh.n_stages);
        std::vector<core_rvq::Codebook> stages(sh.n_stages);
        for (int s = 0; s < sh.n_stages; s++) {
            embeds[s].resize((size_t)sh.K * sh.dim);
            norms[s].resize(sh.K);
            for (auto& e : embeds[s])
                e = gauss(rng);
            for (int k = 0; k < sh.K; k++) {
                float n = 0.0f;
                for (int j = 0; j < sh.dim; j++) {
                    float v = embeds[s][(size_t)k * sh.dim + j];
                    n += v * v;
                }
                norms[s][k] = n;
            }
            stages[s] = {embeds[s].data(), norms[s].data(), sh.K, sh.dim};
        }

        std::vector<int32_t> got((size_t)sh.T * sh.n_stages, -1);
        REQUIRE(core_rvq::encode_euclidean(features.data(), sh.T, sh.dim, stages.data(), sh.n_stages, got.data()));

        std::vector<int32_t> want = reference_encode(features, sh.T, sh.dim, stages);

        // Re-walk with a double residual to classify any disagreement: a code
        // that differs is only acceptable if it is a genuine near-tie (the two
        // candidates' true distances are within a tiny epsilon — float vs the
        // shootout rounding), NOT a real wrong pick.
        std::vector<double> residual(features.begin(), features.end());
        for (int s = 0; s < sh.n_stages; s++) {
            const auto& cb = stages[s];
            for (int t = 0; t < sh.T; t++) {
                double* x = residual.data() + (size_t)t * sh.dim;
                int g = got[(size_t)t * sh.n_stages + s];
                int w = want[(size_t)t * sh.n_stages + s];
                total++;
                if (g != w) {
                    mismatches++;
                    double dg = true_dist(x, cb.embed + (size_t)g * sh.dim, sh.dim);
                    double dw = true_dist(x, cb.embed + (size_t)w * sh.dim, sh.dim);
                    double denom = std::max(1.0, std::abs(dw));
                    // Genuine tie: the picked entry is within 1e-4 relative of the
                    // reference's — a rounding-order flip, not a wrong answer.
                    REQUIRE(std::abs(dg - dw) / denom < 1e-4);
                    tie_mismatches++;
                }
                // Advance the residual along the code core_rvq actually chose so
                // subsequent stages compare against the same state.
                const float* e = cb.embed + (size_t)g * sh.dim;
                for (int j = 0; j < sh.dim; j++)
                    x[j] -= (double)e[j];
            }
        }
    }

    INFO("total codes=" << total << " mismatches=" << mismatches << " (all near-ties=" << tie_mismatches << ")");
    // Every disagreement must have been a certified near-tie.
    REQUIRE(mismatches == tie_mismatches);
}

TEST_CASE("core_rvq::encode_euclidean_per_stage matches the scalar reference (kyutai §176l layout)", "[unit][rvq]") {
    // Validates the convenience wrapper kyutai_stt routes through: internal
    // ||E[k]||^2 precompute + transpose of encode_euclidean's (T,n_stages) output
    // into per-stage out_codes[s][t]. Compared against the full-distance
    // reference (transposed), with the same near-tie certification.
    std::mt19937 rng(0xBADC0DE);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    struct Shape {
        int T, dim, K, n_stages;
    };
    const Shape shapes[] = {{1, 4, 4, 1}, {12, 16, 64, 4}, {20, 24, 128, 6}};

    for (const auto& sh : shapes) {
        std::vector<float> features((size_t)sh.T * sh.dim);
        for (auto& f : features)
            f = gauss(rng);

        std::vector<std::vector<float>> embeds(sh.n_stages);
        std::vector<std::vector<float>> norms(sh.n_stages);
        std::vector<const float*> embed_ptrs(sh.n_stages);
        std::vector<int> sizes(sh.n_stages);
        std::vector<core_rvq::Codebook> stages(sh.n_stages);
        for (int s = 0; s < sh.n_stages; s++) {
            embeds[s].resize((size_t)sh.K * sh.dim);
            norms[s].resize(sh.K);
            for (auto& e : embeds[s])
                e = gauss(rng);
            for (int k = 0; k < sh.K; k++) {
                float nn = 0.0f;
                for (int j = 0; j < sh.dim; j++) {
                    float v = embeds[s][(size_t)k * sh.dim + j];
                    nn += v * v;
                }
                norms[s][k] = nn;
            }
            embed_ptrs[s] = embeds[s].data();
            sizes[s] = sh.K;
            stages[s] = {embeds[s].data(), norms[s].data(), sh.K, sh.dim};
        }

        std::vector<std::vector<int32_t>> got;
        REQUIRE(core_rvq::encode_euclidean_per_stage(features.data(), sh.T, sh.dim, embed_ptrs.data(), sizes.data(),
                                                     sh.n_stages, got));
        REQUIRE((int)got.size() == sh.n_stages);

        std::vector<int32_t> want = reference_encode(features, sh.T, sh.dim, stages);

        // Certify any disagreement as a genuine near-tie (walk with a double
        // residual along the codes the wrapper chose).
        std::vector<double> residual(features.begin(), features.end());
        for (int s = 0; s < sh.n_stages; s++) {
            REQUIRE((int)got[s].size() == sh.T);
            const auto& cb = stages[s];
            for (int t = 0; t < sh.T; t++) {
                double* x = residual.data() + (size_t)t * sh.dim;
                int g = got[s][t];
                int w = want[(size_t)t * sh.n_stages + s];
                if (g != w) {
                    double dg = true_dist(x, cb.embed + (size_t)g * sh.dim, sh.dim);
                    double dw = true_dist(x, cb.embed + (size_t)w * sh.dim, sh.dim);
                    REQUIRE(std::abs(dg - dw) / std::max(1.0, std::abs(dw)) < 1e-4);
                }
                const float* e = cb.embed + (size_t)g * sh.dim;
                for (int j = 0; j < sh.dim; j++)
                    x[j] -= (double)e[j];
            }
        }
    }

    // Malformed: null embed pointer array entry.
    std::vector<float> feat(4, 0.0f);
    const float* nullembed[1] = {nullptr};
    int sz[1] = {2};
    std::vector<std::vector<int32_t>> oc;
    REQUIRE_FALSE(core_rvq::encode_euclidean_per_stage(feat.data(), 1, 4, nullembed, sz, 1, oc));
}

TEST_CASE("core_rvq::encode_euclidean rejects malformed input", "[unit][rvq]") {
    std::vector<float> feat(4, 0.0f);
    std::vector<float> emb(8, 0.0f), norm(2, 0.0f);
    core_rvq::Codebook good{emb.data(), norm.data(), 2, 4};
    std::vector<int32_t> codes(2, -1);

    REQUIRE_FALSE(core_rvq::encode_euclidean(nullptr, 1, 4, &good, 1, codes.data()));
    REQUIRE_FALSE(core_rvq::encode_euclidean(feat.data(), 0, 4, &good, 1, codes.data()));
    REQUIRE_FALSE(core_rvq::encode_euclidean(feat.data(), 1, 4, &good, 0, codes.data()));

    core_rvq::Codebook bad_dim{emb.data(), norm.data(), 2, 8}; // dim != feature dim
    REQUIRE_FALSE(core_rvq::encode_euclidean(feat.data(), 1, 4, &bad_dim, 1, codes.data()));

    core_rvq::Codebook null_norm{emb.data(), nullptr, 2, 4};
    REQUIRE_FALSE(core_rvq::encode_euclidean(feat.data(), 1, 4, &null_norm, 1, codes.data()));
}

TEST_CASE("core_rvq: per-stage norms preserve first-wins rounding near a tie", "[unit][rvq]") {
    ScopedTestEnv serial("CRISPASR_RVQ_SERIAL");
    // Adding four 2^-24 squares separately to 1 rounds to 1 each time;
    // grouping them first gives 1 + 2^-22. The latter silently changes argmin.
    for (const int dim : {8, 9, 12, 64, 65}) {
        for (const int K : {17, 16385}) {
            CAPTURE(dim, K);
            std::vector<float> embed((size_t)K * dim, 2.0f);
            std::fill(embed.begin(), embed.begin() + 2 * dim, 0.0f);
            embed[0] = 1.0f;
            for (int j = 4; j < 8; j++)
                embed[j] = std::ldexp(1.0f, -12);
            embed[dim + 1] = 1.0f;
            const float* books[] = {embed.data()};
            const int sizes[] = {K};
            std::vector<float> features((size_t)3 * dim, 0.0f);
            std::vector<std::vector<int32_t>> one, many;
            REQUIRE(core_rvq::encode_euclidean_per_stage(features.data(), 3, dim, books, sizes, 1, one, 1));
            REQUIRE(core_rvq::encode_euclidean_per_stage(features.data(), 3, dim, books, sizes, 1, many, 4));
            REQUIRE((one == std::vector<std::vector<int32_t>>{{0, 0, 0}}));
            REQUIRE(many == one);
        }
    }
}

TEST_CASE("core_rvq: multistage codes are independent of worker budget", "[unit][rvq]") {
    ScopedTestEnv serial("CRISPASR_RVQ_SERIAL");
    const int T = 33, dim = 32, K = 257, S = 4;
    std::vector<float> features((size_t)T * dim);
    for (size_t i = 0; i < features.size(); i++)
        features[i] = std::sin((float)i * 0.017f);
    std::vector<std::vector<float>> embeds(S, std::vector<float>((size_t)K * dim));
    std::vector<const float*> books(S);
    std::vector<int> sizes(S, K);
    for (int s = 0; s < S; s++) {
        for (size_t i = 0; i < embeds[s].size(); i++)
            embeds[s][i] = std::cos((float)(i + s * 103) * 0.013f) / (s + 1);
        books[s] = embeds[s].data();
    }
    std::vector<std::vector<int32_t>> one, many;
    REQUIRE(core_rvq::encode_euclidean_per_stage(features.data(), T, dim, books.data(), sizes.data(), S, one, 1));
    for (const int workers : {2, 4}) {
        REQUIRE(core_rvq::encode_euclidean_per_stage(features.data(), T, dim, books.data(), sizes.data(), S, many,
                                                     workers));
        REQUIRE(many == one);
    }
}
