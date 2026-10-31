// test-spectral-diarize.cpp — hermetic unit tests for the speaker-clustering
// numerics (#324). No model, no audio, no network.
//
// These exist because the clustering stack is WEIGHT-FREE: crispasr-diff ends
// at the logits and can see none of it (HARD RULE #3b). A transposed matrix, a
// biased-vs-unbiased covariance, or a drifted constant here produces perfectly
// plausible speaker labels and silently worse diarization.
//
// Everything is a KNOWN-ANSWER test on synthetic data — clusters are placed by
// construction, so the expected k and the expected partition are facts, not
// tolerances. Where a tuned constant decides a branch, the test brackets it
// from both sides so that moving the constant fails the test (HARD RULE #2c —
// a tolerance wider than the defect is not a test).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/spectral_diarize.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <random>
#include <set>
#include <vector>
#include "portable_env.h"
#include "scoped_test_env.h"

using namespace core_spectral;

namespace {

// Portable N(0,1). std::mt19937 is identical everywhere but
// std::normal_distribution is NOT — libc++ (macOS) and libstdc++ (Linux) draw
// different sequences from the same engine and seed, so these fixtures produced
// DIFFERENT data per platform and the assertions below silently tested different
// inputs. Box-Muller over the raw engine keeps the fixtures byte-identical
// everywhere. (src/core/spectral_diarize.cpp carries the same helpers for the
// same reason.)
inline double test_uniform01(std::mt19937& rng) {
    const uint64_t hi = (uint64_t)(rng() >> 5); // 27 bits
    const uint64_t lo = (uint64_t)(rng() >> 6); // 26 bits
    return (double)((hi << 26) | lo) * (1.0 / 9007199254740992.0);
}

inline float test_normal(std::mt19937& rng, float sigma = 1.0f) {
    double u1 = test_uniform01(rng);
    if (u1 < 1e-300)
        u1 = 1e-300;
    const double u2 = test_uniform01(rng);
    return (float)(sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2));
}


// `k` isotropic Gaussian blobs. Centres point in RANDOM directions rather than
// along coordinate axes: axis-aligned centres make the between-blob signal
// rank (k-1), so for small k the PCA-8 projection the GMM consumes is a couple
// of real dimensions plus six of near-degenerate noise — and a full-covariance
// GMM overfits that without bound. Measured on axis-aligned blobs: PC0/PC1
// variance 27.9, PC2..PC7 variance 0.03-0.09, and BIC fell monotonically to
// max_k. Random directions reproduce what real speaker embeddings look like.
std::vector<float> make_blobs(int k, int per, int d, float sigma, unsigned seed, float separation = 10.0f) {
    std::mt19937 rng(seed);
    auto g = [&rng]() { return test_normal(rng); };
    auto noise = [&rng, sigma]() { return test_normal(rng, sigma); };
    std::vector<float> cent((size_t)k * d);
    for (auto& v : cent)
        v = g();
    for (int c = 0; c < k; c++) {
        double nr = 0.0;
        for (int j = 0; j < d; j++)
            nr += (double)cent[(size_t)c * d + j] * cent[(size_t)c * d + j];
        nr = std::sqrt(std::max(nr, 1e-12));
        for (int j = 0; j < d; j++)
            cent[(size_t)c * d + j] = (float)(cent[(size_t)c * d + j] * separation / nr);
    }
    std::vector<float> x((size_t)k * per * d, 0.0f);
    for (int c = 0; c < k; c++)
        for (int i = 0; i < per; i++) {
            const size_t row = (size_t)(c * per + i);
            for (int j = 0; j < d; j++)
                x[row * d + j] = cent[(size_t)c * d + j] + noise();
        }
    return x;
}

// True iff `labels` partitions [0, k*per) into exactly the `k` contiguous
// blocks make_blobs() produced — i.e. every blob is one label and no label
// spans two blobs. Label VALUES are arbitrary; only the partition matters.
bool partition_matches_blobs(const std::vector<int>& labels, int k, int per) {
    if ((int)labels.size() != k * per)
        return false;
    std::map<int, int> blob_of_label;
    std::map<int, int> label_of_blob;
    for (int c = 0; c < k; c++)
        for (int i = 0; i < per; i++) {
            const int lab = labels[(size_t)(c * per + i)];
            auto b = blob_of_label.emplace(lab, c);
            if (!b.second && b.first->second != c)
                return false; // one label covering two blobs
            auto l = label_of_blob.emplace(c, lab);
            if (!l.second && l.first->second != lab)
                return false; // one blob split across labels
        }
    return (int)label_of_blob.size() == k;
}

} // namespace

// ── primitives ────────────────────────────────────────────────────────────

TEST_CASE("l2_normalize_rows: unit rows, and a zero row stays zero", "[unit][spectral]") {
    std::vector<float> x = {3.0f, 4.0f, 0.0f, 0.0f, -1.0f, 0.0f};
    l2_normalize_rows(x.data(), 3, 2);
    CHECK(x[0] == Catch::Approx(0.6f));
    CHECK(x[1] == Catch::Approx(0.8f));
    // An all-zero row must survive untouched — dividing by its norm would
    // emit NaN and poison every downstream cosine.
    CHECK(x[2] == 0.0f);
    CHECK(x[3] == 0.0f);
    CHECK(x[4] == Catch::Approx(-1.0f));
}

TEST_CASE("cosine_similarity: exact values on known vectors", "[unit][spectral]") {
    // rows: e0, e1 (orthogonal), -e0 (opposite), 2*e0 (parallel, scaled)
    std::vector<float> x = {1, 0, 0, 1, -1, 0, 2, 0};
    auto s = cosine_similarity(x.data(), 4, 2);
    CHECK(s[0 * 4 + 0] == Catch::Approx(1.0f));
    CHECK(s[0 * 4 + 1] == Catch::Approx(0.0f).margin(1e-6));
    CHECK(s[0 * 4 + 2] == Catch::Approx(-1.0f));
    CHECK(s[0 * 4 + 3] == Catch::Approx(1.0f)); // cosine is scale-invariant
    CHECK(s[1 * 4 + 0] == Catch::Approx(s[0 * 4 + 1]));
}

TEST_CASE("cosine_affinity: rescaled to [0,1], unit diagonal, symmetric", "[unit][spectral]") {
    std::vector<float> x = {1, 0, 0, 1, -1, 0};
    auto a = cosine_affinity(x.data(), 3, 2);
    for (int i = 0; i < 3; i++) {
        CHECK(a[(size_t)i * 3 + i] == Catch::Approx(1.0f));
        for (int j = 0; j < 3; j++) {
            CHECK(a[(size_t)i * 3 + j] >= 0.0f);
            CHECK(a[(size_t)i * 3 + j] <= 1.0f);
            CHECK(a[(size_t)i * 3 + j] == Catch::Approx(a[(size_t)j * 3 + i]));
        }
    }
    CHECK(a[0 * 3 + 1] == Catch::Approx(0.5f)); // orthogonal -> midpoint
    CHECK(a[0 * 3 + 2] == Catch::Approx(0.0f)); // opposite   -> 0
}

TEST_CASE("pca_project: preserves pairwise distances when k == rank", "[unit][spectral]") {
    // Points living on a 2-D plane embedded in 5-D. Projecting to 2 components
    // must be an isometry on that plane — an algebraic invariant, not a
    // tolerance (HARD RULE #2c). A transposed eigenvector matrix breaks it.
    const int n = 12, d = 5, k = 2;
    std::mt19937 rng(7);
    auto g = [&rng]() { return test_normal(rng); };
    std::vector<float> plane((size_t)n * d, 0.0f);
    for (int i = 0; i < n; i++) {
        const float u = g(), v = g();
        plane[(size_t)i * d + 0] = u;
        plane[(size_t)i * d + 1] = v;
        plane[(size_t)i * d + 2] = 0.5f * u - 0.25f * v; // dependent coords
        plane[(size_t)i * d + 3] = -u + 2.0f * v;
        plane[(size_t)i * d + 4] = 3.0f; // constant -> no variance
    }
    int got_k = 0;
    auto p = pca_project(plane.data(), n, d, k, &got_k);
    REQUIRE(got_k == k);
    REQUIRE(p.size() == (size_t)n * k);

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double d_hi = 0, d_lo = 0;
            for (int t = 0; t < d; t++) {
                const double e = plane[(size_t)i * d + t] - plane[(size_t)j * d + t];
                d_hi += e * e;
            }
            for (int t = 0; t < k; t++) {
                const double e = p[(size_t)i * k + t] - p[(size_t)j * k + t];
                d_lo += e * e;
            }
            CHECK(std::sqrt(d_lo) == Catch::Approx(std::sqrt(d_hi)).epsilon(1e-3));
        }
}

TEST_CASE("pca_project: is deterministic across calls", "[unit][spectral]") {
    auto x = make_blobs(3, 8, 6, 1.0f, 11);
    int k1 = 0, k2 = 0;
    auto a = pca_project(x.data(), 24, 6, 4, &k1);
    auto b = pca_project(x.data(), 24, 6, 4, &k2);
    REQUIRE(k1 == k2);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); i++)
        REQUIRE(a[i] == b[i]); // sign canonicalisation must make this exact
}

// ── GMM / BIC ─────────────────────────────────────────────────────────────

TEST_CASE("gmm_bic: BIC is minimised at the true component count", "[unit][spectral]") {
    // 3 well-separated tight blobs -> BIC must bottom out at k = 3.
    const int k_true = 3, per = 40, d = 3;
    auto x = make_blobs(k_true, per, d, 0.35f, 21, 12.0f);
    const int n = k_true * per;

    int best_k = 0;
    float best = std::numeric_limits<float>::infinity();
    for (int k = 1; k <= 6; k++) {
        float bic = 0.0f;
        if (!gmm_bic(x.data(), n, d, k, 3, 200, 42, &bic))
            continue;
        INFO("k=" << k << " bic=" << bic);
        if (bic < best) {
            best = bic;
            best_k = k;
        }
    }
    CHECK(best_k == k_true);
}

TEST_CASE("gmm_bic: rejects impossible fits instead of returning garbage", "[unit][spectral]") {
    std::vector<float> x = {0.0f, 0.0f, 1.0f, 1.0f};
    float bic = 0.0f;
    CHECK_FALSE(gmm_bic(x.data(), 2, 2, 5, 1, 10, 42, &bic)); // k > n
    CHECK_FALSE(gmm_bic(x.data(), 0, 2, 1, 1, 10, 42, &bic)); // empty
}

// ── spectral clustering ───────────────────────────────────────────────────

TEST_CASE("spectral_labels: recovers well-separated blobs exactly", "[unit][spectral]") {
    const int k = 3, per = 15, d = 8;
    auto x = make_blobs(k, per, d, 0.2f, 33, 8.0f);
    const int n = k * per;
    auto aff = cosine_affinity(x.data(), n, d);
    auto lab = spectral_labels(aff.data(), n, k, 42);
    CHECK(partition_matches_blobs(lab, k, per));
}

TEST_CASE("spectral_labels: k<=1 and degenerate sizes are handled", "[unit][spectral]") {
    auto x = make_blobs(1, 5, 4, 0.1f, 5);
    auto aff = cosine_affinity(x.data(), 5, 4);
    auto l1 = spectral_labels(aff.data(), 5, 1, 42);
    REQUIRE(l1.size() == 5);
    for (int v : l1)
        CHECK(v == 0);
    CHECK(spectral_labels(aff.data(), 0, 3, 42).empty());
    // k above n must clamp rather than index out of range.
    auto l2 = spectral_labels(aff.data(), 5, 99, 42);
    CHECK(l2.size() == 5);
}

TEST_CASE("spectral_labels: is deterministic", "[unit][spectral]") {
    const int k = 3, per = 12, d = 6;
    auto x = make_blobs(k, per, d, 0.3f, 77);
    auto aff = cosine_affinity(x.data(), k * per, d);
    auto a = spectral_labels(aff.data(), k * per, k, 42);
    auto b = spectral_labels(aff.data(), k * per, k, 42);
    REQUIRE(a == b);
}

// ── silhouette ────────────────────────────────────────────────────────────

TEST_CASE("silhouette_precomputed: near 1 for tight, far-apart clusters", "[unit][spectral]") {
    // Two clusters of 2, intra-distance 0, inter-distance 1 -> silhouette 1.
    const int n = 4;
    std::vector<float> dist((size_t)n * n, 1.0f);
    for (int i = 0; i < n; i++)
        dist[(size_t)i * n + i] = 0.0f;
    dist[0 * n + 1] = dist[1 * n + 0] = 0.0f;
    dist[2 * n + 3] = dist[3 * n + 2] = 0.0f;
    std::vector<int> lab = {0, 0, 1, 1};
    CHECK(silhouette_precomputed(dist.data(), n, lab) == Catch::Approx(1.0f));
}

TEST_CASE("silhouette_precomputed: 0 when there is only one cluster", "[unit][spectral]") {
    std::vector<float> dist(9, 1.0f);
    std::vector<int> lab = {0, 0, 0};
    CHECK(silhouette_precomputed(dist.data(), 3, lab) == 0.0f);
}

// ── spherical refinement ──────────────────────────────────────────────────

TEST_CASE("refine_spherical: fixes a mislabelled point and keeps k", "[unit][spectral]") {
    const int k = 2, per = 10, d = 5;
    auto x = make_blobs(k, per, d, 0.1f, 91, 9.0f);
    const int n = k * per;
    std::vector<int> lab((size_t)n);
    for (int i = 0; i < n; i++)
        lab[(size_t)i] = i < per ? 0 : 1;
    lab[0] = 1; // deliberate error

    auto refined = refine_spherical(x.data(), n, d, lab, 8);
    REQUIRE(refined.size() == (size_t)n);
    CHECK(partition_matches_blobs(refined, k, per));
    std::set<int> uniq(refined.begin(), refined.end());
    CHECK((int)uniq.size() == k); // refinement must not collapse a cluster
}

TEST_CASE("refine_spherical: single-cluster input yields all zeros", "[unit][spectral]") {
    auto x = make_blobs(1, 6, 3, 0.1f, 4);
    std::vector<int> lab(6, 7); // one cluster, arbitrary label value
    auto r = refine_spherical(x.data(), 6, 3, lab, 8);
    for (int v : r)
        CHECK(v == 0); // densely renumbered from an arbitrary label
}

// ── speaker-count estimation ──────────────────────────────────────────────

TEST_CASE("estimate_speakers: degenerate inputs report why", "[unit][spectral]") {
    auto x = make_blobs(1, 3, 4, 0.1f, 2);
    auto e0 = estimate_speakers(nullptr, 0, 4, 1, 10);
    CHECK(e0.best_k == 1);
    CHECK(std::string(e0.reason) == "no_embeddings");
    auto e1 = estimate_speakers(x.data(), 3, 4, 1, 10);
    CHECK(std::string(e1.reason) == "too_few_samples");
}

TEST_CASE("estimate_speakers: one voice trips the cosine veto", "[unit][spectral]") {
    // All embeddings in a tight cone -> high pairwise cosine -> k = 1 without
    // ever consulting BIC (which would over-split the cloud).
    const int n = 30, d = 8;
    std::mt19937 rng(3);
    auto jitter = [&rng]() { return test_normal(rng, 0.05f); };
    std::vector<float> x((size_t)n * d, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            x[(size_t)i * d + j] = jitter();
        x[(size_t)i * d + 0] += 1.0f;
    }
    auto e = estimate_speakers(x.data(), n, d, 1, 10);
    CHECK(e.best_k == 1);
    CHECK(std::string(e.reason) == "cosine_similarity_single_speaker");
    CHECK(e.cosine_sim_p10 >= kSingleSpeakerSimP10);
}

TEST_CASE("estimate_speakers: the cosine veto is bracketed by its threshold", "[unit][spectral]") {
    // Guard on the tuned constant itself: two configurations that straddle
    // kSingleSpeakerSimP10. If someone retunes the constant, one of these
    // flips and the test fails — which is the point.
    const int n = 40, d = 6;

    // (a) Antipodal pairs: half at +e0, half at -e0. p10 of the off-diagonal
    // cosines is about -1, far BELOW the threshold -> the veto must NOT fire.
    std::vector<float> split((size_t)n * d, 0.0f);
    for (int i = 0; i < n; i++)
        split[(size_t)i * d + 0] = (i < n / 2) ? 1.0f : -1.0f;
    auto e_split = estimate_speakers(split.data(), n, d, 1, 6);
    INFO("split p10 = " << e_split.cosine_sim_p10);
    CHECK(e_split.cosine_sim_p10 < kSingleSpeakerSimP10);
    CHECK(std::string(e_split.reason) != "cosine_similarity_single_speaker");

    // (b) Same direction throughout: p10 ~ 1, far ABOVE -> veto fires.
    std::vector<float> same((size_t)n * d, 0.0f);
    for (int i = 0; i < n; i++)
        same[(size_t)i * d + 0] = 1.0f + 0.001f * (float)i;
    auto e_same = estimate_speakers(same.data(), n, d, 1, 6);
    INFO("same p10 = " << e_same.cosine_sim_p10);
    CHECK(e_same.cosine_sim_p10 >= kSingleSpeakerSimP10);
    CHECK(std::string(e_same.reason) == "cosine_similarity_single_speaker");
}

TEST_CASE("estimate_speakers: min_k > 1 disables the single-speaker veto", "[unit][spectral]") {
    // The veto is gated on min_k <= 1; a caller demanding at least 2 speakers
    // must not be silently handed 1.
    const int n = 30, d = 6;
    std::vector<float> same((size_t)n * d, 0.0f);
    for (int i = 0; i < n; i++)
        same[(size_t)i * d + 0] = 1.0f + 0.001f * (float)i;
    auto e = estimate_speakers(same.data(), n, d, 2, 6);
    CHECK(std::string(e.reason) != "cosine_similarity_single_speaker");
}

// ── end to end ────────────────────────────────────────────────────────────

TEST_CASE("cluster_speakers: fixed count is honoured exactly", "[unit][spectral]") {
    const int k = 3, per = 14, d = 8;
    auto x = make_blobs(k, per, d, 0.25f, 55, 9.0f);
    const int n = k * per;
    auto lab = cluster_speakers(x.data(), n, d, 1, 10, /*num_speakers=*/k, nullptr);
    CHECK(partition_matches_blobs(lab, k, per));
}

TEST_CASE("cluster_speakers: auto-detects the blob count", "[unit][spectral]") {
    // 4-6 speakers: BIC under-counts (measured anchors 2/2/3) but the upstream
    // [k-2, k+3] silhouette window reaches the truth and recovers it exactly.
    for (int k : {4, 5, 6}) {
        const int per = 25, d = 32;
        auto x = make_blobs(k, per, d, 1.0f, 66, 6.0f);
        const int n = k * per;
        SpeakerEstimate est;
        auto lab = cluster_speakers(x.data(), n, d, 1, 10, 0, &est);
        INFO("true k = " << k << ", estimated " << est.best_k << " via " << est.reason);
        std::set<int> uniq(lab.begin(), lab.end());
        CHECK((int)uniq.size() == k);
        CHECK(partition_matches_blobs(lab, k, per));
    }
}

// Regression guard for a MEASURED weakness of the upstream recipe, and for the
// gate that fixes it. BIC over-counts on low-rank embedding sets; the
// [k-2, k+3] window can climb DOWN from a small over-count but not from a large
// one (true k=3 anchored at 8 leaves the window at [6,10]). Silhouette itself
// is reliable — it scored the true k at 1.0390 against 0.68/0.79 for its
// neighbours — so searching the full range recovers it.
//
// The default stays faithful to upstream; the DER harness decides whether to
// flip it. This test pins BOTH arms so neither can regress unnoticed.
TEST_CASE("cluster_speakers: default estimator is bic", "[unit][spectral]") {
    // The DEFAULT is the upstream BIC estimator. Eigengap looks better here —
    // it recovers this synthetic case, which BIC does not — but that is
    // exactly the trap: on real speech it under-counts and scores 11.4% DER
    // against BIC's 5.3% on VoxConverse. This test pins the default so a
    // synthetic win cannot quietly flip it again.
    const int k = 3, per = 25, d = 32;
    auto x = make_blobs(k, per, d, 1.0f, 66, 6.0f);
    const int n = k * per;

    unsetenv("CRISPASR_DIARIZE_COUNT");
    unsetenv("CRISPASR_DIARIZE_BIC_WINDOW");
    SpeakerEstimate est;
    cluster_speakers(x.data(), n, d, 1, 10, 0, &est);
    CHECK(std::string(est.reason) != "eigengap");

    // ...and eigengap remains selectable, and still solves this case.
    setenv("CRISPASR_DIARIZE_COUNT", "eigengap", 1);
    SpeakerEstimate est_eg;
    auto lab_eg = cluster_speakers(x.data(), n, d, 1, 10, 0, &est_eg);
    unsetenv("CRISPASR_DIARIZE_COUNT");
    CHECK(std::string(est_eg.reason) == "eigengap");
    CHECK(partition_matches_blobs(lab_eg, k, per));
}

TEST_CASE("cluster_speakers: full-k search rescues the legacy BIC over-count", "[unit][spectral]") {
    const int k = 3, per = 25, d = 32;
    auto x = make_blobs(k, per, d, 1.0f, 66, 6.0f);
    const int n = k * per;

    // The gate is INVERTED versus when this test was written: the full range is
    // now the default and CRISPASR_DIARIZE_BIC_WINDOW=1 opts back into the
    // anchored [k-2, k+3] window. Both arms are still pinned, and the claim is
    // unchanged — the window strands above the truth, the full range reaches it.
    setenv("CRISPASR_DIARIZE_COUNT", "bic", 1);

    setenv("CRISPASR_DIARIZE_BIC_WINDOW", "1", 1);
    SpeakerEstimate est_window;
    auto lab_window = cluster_speakers(x.data(), n, d, 1, 10, 0, &est_window);
    const size_t k_window = std::set<int>(lab_window.begin(), lab_window.end()).size();
    unsetenv("CRISPASR_DIARIZE_BIC_WINDOW");

    SpeakerEstimate est_full;
    auto lab_full = cluster_speakers(x.data(), n, d, 1, 10, 0, &est_full);
    unsetenv("CRISPASR_DIARIZE_COUNT");

    INFO("bic+window k = " << k_window << ", bic+full-search (default) k = " << est_full.best_k);
    CHECK(partition_matches_blobs(lab_full, k, per)); // the default reaches the truth
    CHECK(k_window != (size_t)k);                     // and the anchored window does not
}

TEST_CASE("cluster_speakers: is deterministic", "[unit][spectral]") {
    const int k = 2, per = 16, d = 6;
    auto x = make_blobs(k, per, d, 0.3f, 88, 8.0f);
    auto a = cluster_speakers(x.data(), k * per, d, 1, 8, 0, nullptr);
    auto b = cluster_speakers(x.data(), k * per, d, 1, 8, 0, nullptr);
    REQUIRE(a == b);
}

TEST_CASE("cluster_speakers: trivial sizes do not crash", "[unit][spectral]") {
    std::vector<float> x = {1.0f, 0.0f};
    CHECK(cluster_speakers(x.data(), 0, 2, 1, 5, 0, nullptr).empty());
    auto one = cluster_speakers(x.data(), 1, 2, 1, 5, 0, nullptr);
    REQUIRE(one.size() == 1);
    CHECK(one[0] == 0);
}

// #390 parallelised both k-sweeps (GMM/BIC and spectral+silhouette). Its whole
// safety argument is that tasks write only their own slot and the reductions
// stay SERIAL in ascending k, so tie-breaks land where the sequential loop put
// them. That claim is only worth as much as a test that varies the fan-out:
// the labels — and the estimated count — must not depend on how the work
// happened to land across threads. n_threads=1 is the sequential baseline the
// parallel arms have to reproduce exactly.
TEST_CASE("cluster_speakers: labels are independent of the thread count", "[unit][spectral]") {
    const int k = 4, per = 20, d = 24;
    auto x = make_blobs(k, per, d, 1.0f, 4242);
    const int n = k * per;

    SpeakerEstimate est1;
    const std::vector<int> serial = cluster_speakers(x.data(), n, d, 1, 10, 0, &est1, 42, /*n_threads=*/1);
    REQUIRE(serial.size() == (size_t)n);

    for (int nt : {2, 4, 8, 0}) { // 0 = "ask the machine"
        SpeakerEstimate est;
        const std::vector<int> got = cluster_speakers(x.data(), n, d, 1, 10, 0, &est, 42, nt);
        INFO("n_threads = " << nt << " (0 means hardware_concurrency)");
        CHECK(got == serial);
        CHECK(est.best_k == est1.best_k);
    }
}


TEST_CASE("cosine_similarity: zero and tiny self-dots survive parallel scheduling", "[unit][spectral]") {
    ScopedTestEnv serial("CRISPASR_DIARIZE_SERIAL");
    for (const int n : {5, 129}) {
        const int d = 256;
        std::vector<float> x((size_t)n * d, 0.0f);
        x[d] = 1e-13f; // below normalization's threshold, yet its square is representable
        for (int i = 2; i < n; i++) {
            x[(size_t)i * d + i % d] = 3.0f;
            x[(size_t)i * d + (i + 1) % d] = 4.0f;
        }
        serial.set("1");
        const auto one = cosine_similarity(x.data(), n, d, 1);
        serial.set(nullptr);
        const auto many = cosine_similarity(x.data(), n, d, 4);
        REQUIRE(one.size() == (size_t)n * n);
        REQUIRE(many.size() == one.size());
        REQUIRE(many[0] == 0.0f);
        REQUIRE(many[(size_t)n + 1] == (float)((double)x[d] * x[d]));
        REQUIRE(many[(size_t)n + 1] > 0.0f);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                REQUIRE(std::isfinite(many[(size_t)i * n + j]));
                REQUIRE(many[(size_t)i * n + j] == one[(size_t)i * n + j]);
                REQUIRE(many[(size_t)i * n + j] == many[(size_t)j * n + i]);
            }
        }
    }
}
