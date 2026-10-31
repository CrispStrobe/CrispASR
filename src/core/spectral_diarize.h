// src/core/spectral_diarize.h — speaker-count estimation + spectral clustering.
//
// The clustering half of the FoxNoseTech/diarize recipe (#324): estimate how
// many speakers are present with a GMM/BIC sweep, then group the speaker
// embeddings with spectral clustering on a cosine affinity.
//
// ── Provenance ────────────────────────────────────────────────────────────
// Clean-room. Every algorithm here is standard published numerical
// linear algebra / statistics (PCA, full-covariance GMM with BIC, the
// Ng-Jordan-Weiss normalised-Laplacian spectral method, silhouette scoring,
// spherical k-means). The upstream Python is a sequence of scikit-learn CALLS,
// not implementations, so there is nothing to translate: what is taken from it
// is the RECIPE — which algorithms run in what order — and the tuned
// CONSTANTS below. Those are facts and parameters, not copyrightable
// expression, which is what keeps this file MIT in an otherwise Apache-2.0
// lineage. See docs/architecture.md#foxnose-diarize.
//
// ── Parity expectations ───────────────────────────────────────────────────
// ⚠ This will NOT reproduce scikit-learn label-for-label, and no amount of
// care would. sklearn's k-means++ seeding, its GMM initialisation and its
// ARPACK/LOBPCG eigensolver all ride its own RNG stream and iteration order.
// Matching those bit-for-bit would mean reimplementing sklearn's RNG, not its
// mathematics. The acceptance gate for this module is therefore:
//   * hermetic unit tests on synthetic data with a KNOWN answer (this is the
//     HARD RULE #3b case — the diff harness cannot see weight-free code), and
//   * DER on labelled audio for the pipeline as a whole.
// Determinism across runs IS guaranteed: every routine here is seeded
// explicitly and does no wall-clock or thread-order-dependent work.

#pragma once

#include <cstddef>
#include <vector>

namespace core_spectral {

// ── Tuned constants (from the upstream recipe) ────────────────────────────

// If the 10th percentile of off-diagonal pairwise cosine similarity is at
// least this high, the whole set is treated as one speaker. This exists
// because BIC happily splits a single speaker's irregular embedding cloud
// into several Gaussians; the percentile is a cheap veto on that.
constexpr float kSingleSpeakerSimP10 = 0.16f;

// Silhouette prefers fewer, broader clusters on noisy speaker embeddings.
// Scoring candidates as `silhouette + kSilhouetteKBonus * log(k)` offsets that
// bias without swamping the separation term.
constexpr float kSilhouetteKBonus = 0.04f;

// PCA target dimensionality for the GMM sweep. Full-covariance GMMs need
// O(d^2) parameters per component, so the projection is what makes BIC
// meaningful at all on 256-dim embeddings.
constexpr int kPcaDim = 8;

// GMM fitting budget.
constexpr int kGmmNInit = 5;
constexpr int kGmmMaxIter = 300;

// ── Results ───────────────────────────────────────────────────────────────

struct SpeakerEstimate {
    int best_k = 1;
    // Why this k was chosen — "gmm_bic", "cosine_similarity_single_speaker",
    // "too_few_samples", "no_embeddings", "gmm_failed", "silhouette".
    const char* reason = "gmm_bic";
    int pca_dim = 0;
    float cosine_sim_p10 = 0.0f;
    std::vector<float> k_bics; // BIC per k, indexed from min_k; empty if unused
    int min_k_used = 1;
};

// ── Primitives (exposed so they can be unit-tested in isolation) ──────────

// Scale each row of `x` (n rows of `d`) to unit L2 norm, in place. Rows that
// are already zero are left alone rather than producing NaN.
void l2_normalize_rows(float* x, int n, int d, int n_threads = 0);

// Pairwise cosine similarity of the rows of `x`, as a dense (n, n) row-major
// matrix. `x` need not be normalised.
std::vector<float> cosine_similarity(const float* x, int n, int d, int n_threads = 0);

// The affinity the spectral step consumes: cosine rescaled from [-1, 1] to
// [0, 1], diagonal forced to 1, negatives clamped away.
std::vector<float> cosine_affinity(const float* x, int n, int d, int n_threads = 0);

// Project `x` (n, d) onto its top `k` principal components -> (n, k)
// row-major. Mean-centres first, exactly like sklearn's PCA. `k` is clamped to
// min(k, n-1, d). Deterministic: component signs are canonicalised so the
// largest-magnitude loading of each component is positive.
std::vector<float> pca_project(const float* x, int n, int d, int k, int* out_k);

// Bayesian Information Criterion of a `k`-component full-covariance Gaussian
// mixture fitted to `x` (n, d), using the same convention as scikit-learn:
//     BIC = -2 * total_log_likelihood + n_free_params * log(n)
// so LOWER is better. Returns false if the fit degenerated (a component
// collapsed, or the covariance became non-positive-definite).
bool gmm_bic(const float* x, int n, int d, int k, int n_init, int max_iter, unsigned seed, float* out_bic);

// Ng-Jordan-Weiss spectral clustering on a precomputed non-negative symmetric
// affinity (n, n): normalised Laplacian -> top-k eigenvectors -> row-normalise
// -> k-means. Returns labels in [0, k).
std::vector<int> spectral_labels(const float* affinity, int n, int k, unsigned seed);

// Mean silhouette score for `labels` under a precomputed distance matrix
// (n, n). Returns 0 when fewer than 2 clusters are populated.
float silhouette_precomputed(const float* distance, int n, const std::vector<int>& labels);

// Spherical centroid reassignment, the refinement the upstream applies after
// the spectral assignment. Operates on L2-normalised copies of `x`; stops
// early on convergence or if a cluster would empty (which would silently
// change k). Returns relabelled, densely-numbered labels.
std::vector<int> refine_spherical(const float* x, int n, int d, const std::vector<int>& labels, int max_iter = 8,
                                  int n_threads = 0);

// Estimate the speaker count from the EIGENGAP of the normalised Laplacian:
// build L_sym = D^-1/2 A D^-1/2, take its leading eigenvalues in descending
// order, and pick the k in [min_k, max_k] with the largest drop from
// lambda_k to lambda_{k+1}.
//
// This is the standard alternative to a GMM/BIC sweep for spectral speaker
// diarization, and it exists here because BIC + silhouette measurably fails
// on real speaker embeddings: silhouette saturates and climbs monotonically
// to whatever ceiling it is given, because within-speaker cosine (~0.6) is
// far below 1 while cross-speaker (~0.1) is far above 0, so splitting a real
// speaker keeps looking like an improvement. The eigengap reads cluster
// structure off the spectrum instead of scoring partitions, so saturation
// does not arise.
//
// `out_eigenvalues` optionally receives the leading eigenvalues (descending).
int estimate_speakers_eigengap(const float* affinity, int n, int min_k, int max_k, unsigned seed = 42,
                               std::vector<float>* out_eigenvalues = nullptr);

// Which speaker-count estimator cluster_speakers() uses.
// CRISPASR_DIARIZE_COUNT=bic|eigengap|nme-sc. DEFAULT IS BIC (the upstream
// estimator) — eigengap wins on synthetic data but under-counts on real
// speech and scores materially worse on VoxConverse. See the measurements at
// count_method_from_env() before changing this.
// Per-candidate trace of the NME-SC sweep, for diagnosis.
struct NmeScDiag {
    struct Point {
        int p;       // neighbours kept per row
        int k;       // k at that p's largest eigengap
        float gap;   // the gap itself
        float ratio; // p / gap — minimised
    };
    std::vector<Point> curve;
    int best_p = 0;
    int best_k = 1;
    float best_r = 0.0f;
};

// NME-SC speaker counting: sweep the affinity binarisation instead of fixing
// it, and take k from the p minimising p / max-eigengap. See the comment at the
// definition. `out` may be null.
int estimate_speakers_nme_sc(const float* affinity, int n, int min_k, int max_k, unsigned seed = 42,
                             NmeScDiag* out = nullptr);

enum class CountMethod { Bic, Eigengap, NmeSc };
CountMethod count_method_from_env();

// ── High-level entry points ───────────────────────────────────────────────

// Estimate the number of speakers in `x` (n, d) via the single-speaker
// cosine veto, then a PCA + full-covariance GMM BIC sweep.
SpeakerEstimate estimate_speakers(const float* x, int n, int d, int min_k, int max_k, unsigned seed = 42,
                                  int n_threads = 0);

// Cluster `x` into speakers. When `num_speakers > 0` that count is used
// directly and no estimation runs; otherwise the count is estimated and then
// refined by scoring a small neighbourhood of candidates on silhouette.
// `out_estimate` may be null.
// `n_threads` bounds the k-sweep fan-out; 0 means "ask the machine". Pass the
// caller's thread budget so a `-t N` run does not oversubscribe a shared box.
std::vector<int> cluster_speakers(const float* x, int n, int d, int min_speakers, int max_speakers, int num_speakers,
                                  SpeakerEstimate* out_estimate, unsigned seed = 42, int n_threads = 0);

} // namespace core_spectral
