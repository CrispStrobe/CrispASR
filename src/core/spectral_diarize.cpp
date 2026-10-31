// src/core/spectral_diarize.cpp — see spectral_diarize.h for provenance and
// parity expectations.

#include "spectral_diarize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <cstdlib>
#include <functional>
#include <atomic>
#include <random>
#include <string>
#include <thread>

#include "parallel_for.h"

namespace core_spectral {

namespace {

constexpr double kTiny = 1e-12;

// ── Portable RNG draws ────────────────────────────────────────────────────
//
// std::mt19937 is specified bit-for-bit by the standard, but the DISTRIBUTIONS
// are not: libc++ (macOS) and libstdc++ (Linux) yield different sequences from
// the same engine and the same seed. That made the whole diarizer
// platform-dependent — on four well-separated blobs the estimator returned
// k=4 on macOS and k=9 on Linux, because the random projection in
// top_eigenvectors and the k-means++ seeding both diverged, and the silhouette
// pass then scored a different winner.
//
// Drawing straight from the engine makes every result reproducible across
// platforms and standard libraries. Same seed in, same clustering out.
inline double rng_uniform01(std::mt19937& rng) {
    // 53 significant bits from two draws, in [0, 1).
    const uint64_t hi = (uint64_t)(rng() >> 5);                    // 27 bits
    const uint64_t lo = (uint64_t)(rng() >> 6);                    // 26 bits
    return (double)((hi << 26) | lo) * (1.0 / 9007199254740992.0); // 2^-53
}

// Inclusive [lo, hi]. Rejects the trailing partial block so the modulo is
// unbiased, which std::uniform_int_distribution also does — just not portably.
inline int rng_uniform_int(std::mt19937& rng, int lo, int hi) {
    if (hi <= lo)
        return lo;
    const uint32_t span = (uint32_t)(hi - lo) + 1u;
    const uint32_t limit = 0xFFFFFFFFu - (0xFFFFFFFFu % span);
    uint32_t v;
    do {
        v = (uint32_t)rng();
    } while (v >= limit);
    return lo + (int)(v % span);
}

// Box-Muller, so the normal draw is fully determined by the uniforms above.
inline double rng_normal(std::mt19937& rng) {
    double u1 = rng_uniform01(rng);
    if (u1 < 1e-300)
        u1 = 1e-300;
    const double u2 = rng_uniform01(rng);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}
// Ridge added to each covariance diagonal, mirroring sklearn's reg_covar.
// Without it a component that captures a near-degenerate cluster produces a
// singular covariance and the log-likelihood blows up to +inf, which BIC then
// happily reports as the best k.
constexpr double kRegCovar = 1e-6;

// ── Symmetric eigendecomposition (cyclic Jacobi) ──────────────────────────
// Used for the PCA covariance, which is d x d with d = embedding dim (256).
// O(d^3) per sweep is fine at that size and needs no external LAPACK.
// Returns eigenvalues ascending in `w`, eigenvectors as columns of `v`.
void jacobi_eigen(std::vector<double>& a, int n, std::vector<double>& w, std::vector<double>& v) {
    v.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        v[(size_t)i * n + i] = 1.0;

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                off += a[(size_t)p * n + q] * a[(size_t)p * n + q];
        if (off < 1e-22)
            break;
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                const double apq = a[(size_t)p * n + q];
                if (std::fabs(apq) < 1e-18)
                    continue;
                const double app = a[(size_t)p * n + p];
                const double aqq = a[(size_t)q * n + q];
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int k = 0; k < n; k++) {
                    const double akp = a[(size_t)k * n + p], akq = a[(size_t)k * n + q];
                    a[(size_t)k * n + p] = c * akp - s * akq;
                    a[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    const double apk = a[(size_t)p * n + k], aqk = a[(size_t)q * n + k];
                    a[(size_t)p * n + k] = c * apk - s * aqk;
                    a[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {
                    const double vkp = v[(size_t)k * n + p], vkq = v[(size_t)k * n + q];
                    v[(size_t)k * n + p] = c * vkp - s * vkq;
                    v[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    w.resize((size_t)n);
    for (int i = 0; i < n; i++)
        w[(size_t)i] = a[(size_t)i * n + i];
}

// ── Top-k eigenvectors of a large symmetric PSD matrix ────────────────────
// Subspace (orthogonal) iteration. The spectral step needs only the leading k
// eigenvectors of an n x n affinity where n is the number of windows, so a
// full O(n^3) Jacobi would be wasteful — this is O(n^2 k) per iteration.
// The matrix is shifted to guarantee positive semi-definiteness so the
// iteration converges to the ALGEBRAICALLY largest eigenvalues (the ones the
// normalised Laplacian method wants), not the largest in magnitude.
std::vector<double> top_eigenvectors(const std::vector<double>& m, int n, int k, unsigned seed) {
    k = std::min(k, n);
    std::vector<double> q((size_t)n * k);
    std::mt19937 rng(seed);
    for (auto& v : q)
        v = rng_normal(rng);

    // Gershgorin bound -> shift so every eigenvalue is >= 0.
    double shift = 0.0;
    for (int i = 0; i < n; i++) {
        double row = 0.0;
        for (int j = 0; j < n; j++)
            row += std::fabs(m[(size_t)i * n + j]);
        shift = std::max(shift, row);
    }

    auto orthonormalize = [&]() {
        for (int c = 0; c < k; c++) {
            for (int p = 0; p < c; p++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++)
                    dot += q[(size_t)i * k + c] * q[(size_t)i * k + p];
                for (int i = 0; i < n; i++)
                    q[(size_t)i * k + c] -= dot * q[(size_t)i * k + p];
            }
            double nrm = 0.0;
            for (int i = 0; i < n; i++)
                nrm += q[(size_t)i * k + c] * q[(size_t)i * k + c];
            nrm = std::sqrt(nrm);
            if (nrm > kTiny)
                for (int i = 0; i < n; i++)
                    q[(size_t)i * k + c] /= nrm;
        }
    };
    orthonormalize();

    std::vector<double> z((size_t)n * k);
    std::vector<double> prev;
    for (int it = 0; it < 300; it++) {
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                double acc = shift * q[(size_t)i * k + c];
                for (int j = 0; j < n; j++)
                    acc += m[(size_t)i * n + j] * q[(size_t)j * k + c];
                z[(size_t)i * k + c] = acc;
            }
        }
        q.swap(z);
        orthonormalize();
        if (!prev.empty()) {
            // Converged when the subspace stops moving. Compare |<q_c, prev_c>|
            // so an eigenvector flipping sign does not look like movement.
            double worst = 1.0;
            for (int c = 0; c < k; c++) {
                double dot = 0.0;
                for (int i = 0; i < n; i++)
                    dot += q[(size_t)i * k + c] * prev[(size_t)i * k + c];
                worst = std::min(worst, std::fabs(dot));
            }
            if (worst > 1.0 - 1e-9)
                break;
        }
        prev = q;
    }
    return q;
}

// ── k-means (k-means++ seeding, Lloyd iterations) ─────────────────────────
std::vector<int> kmeans(const std::vector<double>& x, int n, int d, int k, unsigned seed, int n_init, int max_iter) {
    std::vector<int> best_labels((size_t)n, 0);
    double best_inertia = std::numeric_limits<double>::infinity();
    std::mt19937 rng(seed);

    for (int init = 0; init < n_init; init++) {
        std::vector<double> cent((size_t)k * d);
        std::vector<double> d2((size_t)n, std::numeric_limits<double>::infinity());

        const int first = rng_uniform_int(rng, 0, n - 1);
        std::copy(x.begin() + (size_t)first * d, x.begin() + (size_t)(first + 1) * d, cent.begin());

        for (int c = 1; c < k; c++) {
            double total = 0.0;
            for (int i = 0; i < n; i++) {
                double dist = 0.0;
                for (int j = 0; j < d; j++) {
                    const double diff = x[(size_t)i * d + j] - cent[(size_t)(c - 1) * d + j];
                    dist += diff * diff;
                }
                d2[(size_t)i] = std::min(d2[(size_t)i], dist);
                total += d2[(size_t)i];
            }
            double target = rng_uniform01(rng) * std::max(total, kTiny);
            double run = 0.0;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) {
                run += d2[(size_t)i];
                if (run >= target) {
                    chosen = i;
                    break;
                }
            }
            std::copy(x.begin() + (size_t)chosen * d, x.begin() + (size_t)(chosen + 1) * d,
                      cent.begin() + (size_t)c * d);
        }

        std::vector<int> labels((size_t)n, 0);
        double inertia = 0.0;
        for (int it = 0; it < max_iter; it++) {
            bool moved = false;
            inertia = 0.0;
            for (int i = 0; i < n; i++) {
                double best = std::numeric_limits<double>::infinity();
                int arg = 0;
                for (int c = 0; c < k; c++) {
                    double dist = 0.0;
                    for (int j = 0; j < d; j++) {
                        const double diff = x[(size_t)i * d + j] - cent[(size_t)c * d + j];
                        dist += diff * diff;
                    }
                    if (dist < best) {
                        best = dist;
                        arg = c;
                    }
                }
                if (labels[(size_t)i] != arg) {
                    labels[(size_t)i] = arg;
                    moved = true;
                }
                inertia += best;
            }
            std::vector<double> sum((size_t)k * d, 0.0);
            std::vector<int> cnt((size_t)k, 0);
            for (int i = 0; i < n; i++) {
                const int c = labels[(size_t)i];
                cnt[(size_t)c]++;
                for (int j = 0; j < d; j++)
                    sum[(size_t)c * d + j] += x[(size_t)i * d + j];
            }
            for (int c = 0; c < k; c++)
                if (cnt[(size_t)c] > 0)
                    for (int j = 0; j < d; j++)
                        cent[(size_t)c * d + j] = sum[(size_t)c * d + j] / cnt[(size_t)c];
            if (!moved)
                break;
        }
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_labels = labels;
        }
    }
    return best_labels;
}

} // namespace

// ===========================================================================

void l2_normalize_rows(float* x, int n, int d, int n_threads) {
    const char* env_serial = std::getenv("CRISPASR_DIARIZE_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');
    core_parallel::for_each_chunk_work(n, force_serial ? 1 : n_threads, (uint64_t)d * 2, 65536,
                                       [&](int begin, int end) {
                                           for (int i = begin; i < end; i++) {
                                               double nrm = 0.0;
                                               for (int j = 0; j < d; j++)
                                                   nrm += (double)x[(size_t)i * d + j] * x[(size_t)i * d + j];
                                               nrm = std::sqrt(nrm);
                                               if (nrm <= kTiny)
                                                   continue; // preserve zero/tiny rows rather than emit NaN
                                               for (int j = 0; j < d; j++)
                                                   x[(size_t)i * d + j] = (float)(x[(size_t)i * d + j] / nrm);
                                           }
                                       });
}

std::vector<float> cosine_similarity(const float* x, int n, int d, int n_threads) {
    std::vector<float> norm((size_t)n * d);
    std::copy(x, x + (size_t)n * d, norm.begin());
    l2_normalize_rows(norm.data(), n, d, n_threads);

    std::vector<float> out((size_t)n * n, 0.0f);
    const char* env_serial = std::getenv("CRISPASR_DIARIZE_SERIAL");
    const bool force_serial = (env_serial && env_serial[0] == '1');
    // Compute each upper-triangle dot product once, then mirror it. Pair an
    // early (long) row with a late (short) row to balance each worker's work.
    // Every output element has exactly one owner, including the diagonal.
    const int row_pairs = n / 2 + n % 2;
    core_parallel::for_each_chunk_work(row_pairs, force_serial ? 1 : n_threads, ((uint64_t)n + 1) * d, 65536,
                                       [&](int begin, int end) {
                                           auto row = [&](int i) {
                                               const float* row_i = norm.data() + (size_t)i * d;
                                               for (int j = i; j < n; j++) {
                                                   const float* row_j = norm.data() + (size_t)j * d;
                                                   double dot = 0.0;
                                                   for (int t = 0; t < d; t++)
                                                       dot += (double)row_i[t] * row_j[t];
                                                   // A zero row has self-similarity zero; tiny preserved rows
                                                   // retain their actual self-dot. Neither has a unit diagonal.
                                                   out[(size_t)i * n + j] = out[(size_t)j * n + i] = (float)dot;
                                               }
                                           };
                                           for (int pair = begin; pair < end; pair++) {
                                               row(pair);
                                               const int other = n - 1 - pair;
                                               if (other != pair)
                                                   row(other);
                                           }
                                       });
    return out;
}

std::vector<float> cosine_affinity(const float* x, int n, int d, int n_threads) {
    std::vector<float> a = cosine_similarity(x, n, d, n_threads);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float v = (a[(size_t)i * n + j] + 1.0f) * 0.5f;
            a[(size_t)i * n + j] = std::max(v, 0.0f);
        }
    for (int i = 0; i < n; i++)
        a[(size_t)i * n + i] = 1.0f;
    return a;
}

std::vector<float> pca_project(const float* x, int n, int d, int k, int* out_k) {
    k = std::min({k, n - 1, d});
    if (out_k)
        *out_k = k;
    if (k <= 0 || n <= 0)
        return {};

    std::vector<double> mean((size_t)d, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            mean[(size_t)j] += x[(size_t)i * d + j];
    for (int j = 0; j < d; j++)
        mean[(size_t)j] /= n;

    std::vector<double> c((size_t)d * d, 0.0);
    std::vector<double> row((size_t)d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            row[(size_t)j] = x[(size_t)i * d + j] - mean[(size_t)j];
        for (int a = 0; a < d; a++)
            for (int b = a; b < d; b++)
                c[(size_t)a * d + b] += row[(size_t)a] * row[(size_t)b];
    }
    const double denom = std::max(1, n - 1);
    for (int a = 0; a < d; a++)
        for (int b = a; b < d; b++)
            c[(size_t)b * d + a] = c[(size_t)a * d + b] = c[(size_t)a * d + b] / denom;

    std::vector<double> w, v;
    jacobi_eigen(c, d, w, v);

    // Jacobi leaves eigenvalues unordered; take the k largest.
    std::vector<int> idx((size_t)d);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return w[(size_t)a] > w[(size_t)b]; });

    std::vector<float> out((size_t)n * k, 0.0f);
    for (int comp = 0; comp < k; comp++) {
        const int e = idx[(size_t)comp];
        // Canonicalise the sign so repeated runs agree: the largest-magnitude
        // loading is made positive. An eigenvector and its negation span the
        // same component, and the downstream GMM is affine-equivariant, but a
        // stable sign keeps the output reproducible.
        int arg = 0;
        double mx = 0.0;
        for (int j = 0; j < d; j++)
            if (std::fabs(v[(size_t)j * d + e]) > mx) {
                mx = std::fabs(v[(size_t)j * d + e]);
                arg = j;
            }
        const double sign = v[(size_t)arg * d + e] < 0 ? -1.0 : 1.0;
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < d; j++)
                acc += (x[(size_t)i * d + j] - mean[(size_t)j]) * v[(size_t)j * d + e] * sign;
            out[(size_t)i * k + comp] = (float)acc;
        }
    }
    return out;
}

bool gmm_bic(const float* x, int n, int d, int k, int n_init, int max_iter, unsigned seed, float* out_bic) {
    if (n <= 0 || d <= 0 || k <= 0 || k > n)
        return false;

    std::vector<double> xs((size_t)n * d);
    for (size_t i = 0; i < xs.size(); i++)
        xs[i] = x[i];

    double best_ll = -std::numeric_limits<double>::infinity();
    bool any = false;

    for (int init = 0; init < n_init; init++) {
        // sklearn initialises the responsibilities from k-means; same here.
        std::vector<int> lab = kmeans(xs, n, d, k, seed + (unsigned)init * 7919u, 1, 50);

        std::vector<double> weight((size_t)k, 0.0);
        std::vector<double> mu((size_t)k * d, 0.0);
        std::vector<double> cov((size_t)k * d * d, 0.0);
        std::vector<double> resp((size_t)n * k, 0.0);
        for (int i = 0; i < n; i++)
            resp[(size_t)i * k + lab[(size_t)i]] = 1.0;

        double ll = -std::numeric_limits<double>::infinity();
        bool ok = true;

        for (int it = 0; it < max_iter; it++) {
            // ---- M step ----
            std::fill(weight.begin(), weight.end(), 0.0);
            std::fill(mu.begin(), mu.end(), 0.0);
            std::fill(cov.begin(), cov.end(), 0.0);
            for (int c = 0; c < k; c++) {
                double nk = 0.0;
                for (int i = 0; i < n; i++)
                    nk += resp[(size_t)i * k + c];
                if (nk < 1e-8) {
                    ok = false;
                    break;
                }
                weight[(size_t)c] = nk / n;
                for (int i = 0; i < n; i++) {
                    const double r = resp[(size_t)i * k + c];
                    if (r == 0.0)
                        continue;
                    for (int j = 0; j < d; j++)
                        mu[(size_t)c * d + j] += r * xs[(size_t)i * d + j];
                }
                for (int j = 0; j < d; j++)
                    mu[(size_t)c * d + j] /= nk;
                std::vector<double> diff((size_t)d);
                for (int i = 0; i < n; i++) {
                    const double r = resp[(size_t)i * k + c];
                    if (r == 0.0)
                        continue;
                    for (int j = 0; j < d; j++)
                        diff[(size_t)j] = xs[(size_t)i * d + j] - mu[(size_t)c * d + j];
                    for (int a = 0; a < d; a++)
                        for (int b = a; b < d; b++)
                            cov[((size_t)c * d + a) * d + b] += r * diff[(size_t)a] * diff[(size_t)b];
                }
                for (int a = 0; a < d; a++) {
                    for (int b = a; b < d; b++) {
                        double v = cov[((size_t)c * d + a) * d + b] / nk;
                        if (a == b)
                            v += kRegCovar;
                        cov[((size_t)c * d + a) * d + b] = v;
                        cov[((size_t)c * d + b) * d + a] = v;
                    }
                }
            }
            if (!ok)
                break;

            // ---- E step (Cholesky per component) ----
            std::vector<double> logdet((size_t)k, 0.0);
            std::vector<double> chol((size_t)k * d * d, 0.0);
            for (int c = 0; c < k && ok; c++) {
                double* L = &chol[(size_t)c * d * d];
                const double* C = &cov[(size_t)c * d * d];
                for (int a = 0; a < d && ok; a++) {
                    for (int b = 0; b <= a; b++) {
                        double s = C[(size_t)a * d + b];
                        for (int t = 0; t < b; t++)
                            s -= L[(size_t)a * d + t] * L[(size_t)b * d + t];
                        if (a == b) {
                            if (s <= 0.0) {
                                ok = false; // covariance lost positive-definiteness
                                break;
                            }
                            L[(size_t)a * d + b] = std::sqrt(s);
                        } else {
                            L[(size_t)a * d + b] = s / L[(size_t)b * d + b];
                        }
                    }
                }
                if (!ok)
                    break;
                double ld = 0.0;
                for (int a = 0; a < d; a++)
                    ld += std::log(L[(size_t)a * d + a]);
                logdet[(size_t)c] = 2.0 * ld;
            }
            if (!ok)
                break;

            const double log2pi = std::log(2.0 * M_PI);
            double total = 0.0;
            std::vector<double> lp((size_t)k);
            std::vector<double> y((size_t)d);
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < k; c++) {
                    const double* L = &chol[(size_t)c * d * d];
                    double quad = 0.0;
                    for (int a = 0; a < d; a++) {
                        double s = xs[(size_t)i * d + a] - mu[(size_t)c * d + a];
                        for (int t = 0; t < a; t++)
                            s -= L[(size_t)a * d + t] * y[(size_t)t];
                        y[(size_t)a] = s / L[(size_t)a * d + a];
                        quad += y[(size_t)a] * y[(size_t)a];
                    }
                    lp[(size_t)c] =
                        std::log(std::max(weight[(size_t)c], kTiny)) - 0.5 * (d * log2pi + logdet[(size_t)c] + quad);
                }
                const double mx = *std::max_element(lp.begin(), lp.end());
                double se = 0.0;
                for (int c = 0; c < k; c++)
                    se += std::exp(lp[(size_t)c] - mx);
                const double lse = mx + std::log(se);
                total += lse;
                for (int c = 0; c < k; c++)
                    resp[(size_t)i * k + c] = std::exp(lp[(size_t)c] - lse);
            }
            const double prev = ll;
            ll = total;
            if (it > 0 && std::fabs(ll - prev) < 1e-6 * std::fabs(ll))
                break;
        }

        if (ok && std::isfinite(ll) && ll > best_ll) {
            best_ll = ll;
            any = true;
        }
    }

    if (!any)
        return false;

    // sklearn's convention: bic = -2 * log_likelihood + n_params * log(n),
    // with full-covariance parameter count k*d*(d+1)/2 + k*d + (k-1).
    const double n_params = (double)k * d * (d + 1) / 2.0 + (double)k * d + (k - 1);
    *out_bic = (float)(-2.0 * best_ll + n_params * std::log((double)n));
    return true;
}

// Row-wise threshold + symmetrise, the sparsification the eigengap needs.
//
// The cosine affinity is DENSE: (cos+1)/2 sits around 0.5 even for unrelated
// windows, so the graph is nearly complete, its spectrum has one dominant
// eigenvalue, and the largest gap is always at k=1 — the eigengap reports one
// speaker for everything. Keeping only each row's strongest links (the rest
// attenuated, not deleted, so the graph stays connected) restores the block
// structure the spectrum is supposed to expose. This is the row-thresholding
// step from the standard spectral-diarization refinement sequence.
std::vector<float> refine_affinity_rows(const float* affinity, int n, float keep_fraction, float attenuate) {
    std::vector<float> a((size_t)n * n);
    std::copy(affinity, affinity + (size_t)n * n, a.begin());
    const int keep = std::max(1, (int)std::lround(keep_fraction * n));
    std::vector<float> row((size_t)n);
    for (int i = 0; i < n; i++) {
        std::copy(a.begin() + (size_t)i * n, a.begin() + (size_t)(i + 1) * n, row.begin());
        std::nth_element(row.begin(), row.begin() + (n - keep), row.end());
        const float thresh = row[(size_t)(n - keep)];
        for (int j = 0; j < n; j++)
            if (a[(size_t)i * n + j] < thresh)
                a[(size_t)i * n + j] *= attenuate;
    }
    // Row thresholding breaks symmetry; restore it with the elementwise max
    // so a strong link surviving in either direction survives in both.
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            const float m = std::max(a[(size_t)i * n + j], a[(size_t)j * n + i]);
            a[(size_t)i * n + j] = a[(size_t)j * n + i] = m;
        }
    return a;
}

// Leading eigenvalues (descending) of the normalised Laplacian D^-1/2 A D^-1/2.
// Shared by the fixed-binarisation eigengap estimator and by the NME-SC sweep,
// which differ only in how A was binarised beforehand — duplicating it once per
// caller is how the powerset table ended up wrong in two places.
static std::vector<double> laplacian_eigenvalues(const float* aff, int n, int want, unsigned seed) {
    std::vector<double> dinv((size_t)n);
    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++)
            deg += aff[(size_t)i * n + j];
        dinv[(size_t)i] = deg > kTiny ? 1.0 / std::sqrt(deg) : 0.0;
    }
    std::vector<double> l((size_t)n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            l[(size_t)i * n + j] = dinv[(size_t)i] * aff[(size_t)i * n + j] * dinv[(size_t)j];

    const std::vector<double> q = top_eigenvectors(l, n, want, seed);
    const int k_have = std::min(want, n);

    // Rayleigh quotient per converged eigenvector: lambda_c = q_c^T L q_c
    // (q is orthonormal, so no normalisation term is needed).
    std::vector<double> lam((size_t)k_have, 0.0);
    std::vector<double> lq((size_t)n);
    for (int c = 0; c < k_have; c++) {
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++)
                acc += l[(size_t)i * n + j] * q[(size_t)j * k_have + c];
            lq[(size_t)i] = acc;
        }
        double num = 0.0;
        for (int i = 0; i < n; i++)
            num += q[(size_t)i * k_have + c] * lq[(size_t)i];
        lam[(size_t)c] = num;
    }
    std::sort(lam.begin(), lam.end(), std::greater<double>());
    return lam;
}

// ── NME-SC: auto-tuned binarisation + eigengap ────────────────────────────
//
// estimate_speakers_eigengap above binarises the affinity at a HARDCODED 15%
// of neighbours per row and reads the eigengap off that one graph. The choice
// of binarisation is not incidental — it decides how many clusters the
// spectrum appears to have — and no single value suits every recording. That
// is the whole premise of NME-SC (Park et al., "Auto-Tuning Spectral
// Clustering for Speaker Diarization Using Normalized Maximum Eigengap"): try
// several p, and pick the one whose graph gives the cleanest verdict.
//
// For each candidate p:
//     binarise to p neighbours per row -> normalised Laplacian eigenvalues
//     g_p = the largest eigengap, k_p = the k where it occurs
//     r(p) = p / g_p
// and take k from the p minimising r(p). Low r means "few neighbours needed to
// produce a large gap", i.e. structure the graph agrees on rather than
// structure a dense graph smeared into existence.
//
// Sweeping p costs one eigendecomposition per candidate, so the grid is
// geometric (~12 points) rather than exhaustive.
int estimate_speakers_nme_sc(const float* affinity, int n, int min_k, int max_k, unsigned seed, NmeScDiag* out) {
    min_k = std::max(1, min_k);
    if (out)
        *out = NmeScDiag{};
    if (n <= 2)
        return min_k;

    const int want = std::min(n, std::max(min_k + 1, max_k + 1));
    const int hi_p = std::max(2, n / 2);

    // Geometric grid over the neighbour count, deduplicated and clamped.
    std::vector<int> ps;
    for (double v = 2.0; v <= (double)hi_p + 0.5; v *= 1.6) {
        const int p = (int)std::lround(v);
        if (ps.empty() || ps.back() != p)
            ps.push_back(std::min(p, hi_p));
    }
    if (ps.empty())
        ps.push_back(std::min(2, hi_p));

    int best_k = min_k;
    double best_r = std::numeric_limits<double>::infinity();
    int best_p = ps.front();
    for (int p : ps) {
        // TRUE binarisation (attenuate = 0), not the 0.01 attenuation the fixed
        // eigengap path uses. With a dense weak background left in place the
        // maximum eigengap grows monotonically with p, so p/g_p is minimised at
        // the smallest p every time and the sweep degenerates to k=1. Zeroing
        // the non-neighbours is what makes r(p) have an interior minimum at
        // all — it is the method, not an implementation detail.
        const std::vector<float> aff = refine_affinity_rows(affinity, n, (float)p / (float)n, 0.0f);
        const std::vector<double> lam = laplacian_eigenvalues(aff.data(), n, want, seed);
        const int k_have = (int)lam.size();
        const int hi = std::min({max_k, k_have - 1, n - 1});
        double g = -1.0;
        int k_at = min_k;
        for (int k = min_k; k <= hi; k++) {
            const double gap = lam[(size_t)(k - 1)] - lam[(size_t)k];
            if (gap > g) {
                g = gap;
                k_at = k;
            }
        }
        if (g <= kTiny)
            continue; // a degenerate graph says nothing about k
        const double r = (double)p / g;
        if (out)
            out->curve.push_back({p, k_at, (float)g, (float)r});
        if (r < best_r) {
            best_r = r;
            best_k = k_at;
            best_p = p;
        }
    }
    if (out) {
        out->best_p = best_p;
        out->best_k = best_k;
        out->best_r = (float)best_r;
    }
    return best_k;
}

int estimate_speakers_eigengap(const float* affinity, int n, int min_k, int max_k, unsigned seed,
                               std::vector<float>* out_eigenvalues) {
    if (out_eigenvalues)
        out_eigenvalues->clear();
    min_k = std::max(1, min_k);
    if (n <= 1)
        return min_k;
    // Need one eigenvalue beyond max_k to see the gap AT max_k.
    const int want = std::min(n, std::max(min_k + 1, max_k + 1));

    const std::vector<float> aff = refine_affinity_rows(affinity, n, 0.15f, 0.01f);
    std::vector<double> lam = laplacian_eigenvalues(aff.data(), n, want, seed);
    const int k_have = (int)lam.size();
    if (out_eigenvalues)
        for (double v : lam)
            out_eigenvalues->push_back((float)v);

    int best_k = min_k;
    double best_gap = -1.0;
    const int hi = std::min({max_k, k_have - 1, n - 1});
    for (int k = std::max(1, min_k); k <= hi; k++) {
        const double gap = lam[(size_t)(k - 1)] - lam[(size_t)k];
        if (gap > best_gap) {
            best_gap = gap;
            best_k = k;
        }
    }
    return best_k;
}

CountMethod count_method_from_env() {
    // DEFAULT IS BIC — the upstream estimator.
    //
    // An earlier revision defaulted to eigengap on the strength of five
    // synthetic blob configurations (5/5 vs 4/5) and one 31.5 s clip. Both
    // were unrepresentative, and a real benchmark reversed the verdict.
    // Pooled DER over 8 VoxConverse dev files against HUMAN labels
    // (0.25 s collar, optimal 1:1 mapping):
    //
    //     upstream Python diarize 0.1.2   3.1 %
    //     this port, bic                  5.3 %
    //     this port, eigengap            11.4 %
    //
    // Eigengap systematically UNDER-counts on real speech — reference
    // 4/7/2/5/5/4/4/5 speakers against 3/5/2/3/3/2/2/3 — and the confusion
    // term triples. It is retained behind CRISPASR_DIARIZE_COUNT=eigengap
    // because it is genuinely better on well-separated data and costs less,
    // but it is not the default and synthetic evidence must not be used to
    // make it one again.
    // CRISPASR_DIARIZE_COUNT=nme-sc selects NME-SC, which is the same eigengap
    // read but over an AUTO-TUNED binarisation rather than a hardcoded 15%.
    // The fixed binarisation is a plausible reason plain eigengap under-counts
    // above, so this is the variant worth measuring — but it is opt-in until
    // it has been scored on speaker-count accuracy over a held-out split, not
    // on DER over the handful of files that happened to be to hand. See
    // tools/diarize_eval.py.
    const char* e = std::getenv("CRISPASR_DIARIZE_COUNT");
    if (e && *e) {
        const std::string v(e);
        if (v == "eigengap")
            return CountMethod::Eigengap;
        if (v == "nme-sc" || v == "nmesc")
            return CountMethod::NmeSc;
    }
    return CountMethod::Bic;
}

std::vector<int> spectral_labels(const float* affinity, int n, int k, unsigned seed) {
    if (n <= 0)
        return {};
    k = std::min(k, n);
    if (k <= 1)
        return std::vector<int>((size_t)n, 0);

    // Normalised Laplacian L_sym = D^-1/2 A D^-1/2; its top-k eigenvectors are
    // the embedding the clustering runs in.
    std::vector<double> dinv((size_t)n);
    for (int i = 0; i < n; i++) {
        double deg = 0.0;
        for (int j = 0; j < n; j++)
            deg += affinity[(size_t)i * n + j];
        dinv[(size_t)i] = deg > kTiny ? 1.0 / std::sqrt(deg) : 0.0;
    }
    std::vector<double> l((size_t)n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            l[(size_t)i * n + j] = dinv[(size_t)i] * affinity[(size_t)i * n + j] * dinv[(size_t)j];

    std::vector<double> emb = top_eigenvectors(l, n, k, seed);

    // Row-normalise (the Ng-Jordan-Weiss step) so k-means sees directions.
    for (int i = 0; i < n; i++) {
        double nrm = 0.0;
        for (int c = 0; c < k; c++)
            nrm += emb[(size_t)i * k + c] * emb[(size_t)i * k + c];
        nrm = std::sqrt(nrm);
        if (nrm > kTiny)
            for (int c = 0; c < k; c++)
                emb[(size_t)i * k + c] /= nrm;
    }
    return kmeans(emb, n, k, k, seed, 10, 300);
}

float silhouette_precomputed(const float* distance, int n, const std::vector<int>& labels) {
    if (n <= 1 || (int)labels.size() != n)
        return 0.0f;
    int k = 0;
    for (int v : labels)
        k = std::max(k, v + 1);
    if (k < 2)
        return 0.0f;

    std::vector<int> count((size_t)k, 0);
    for (int v : labels)
        count[(size_t)v]++;

    double total = 0.0;
    std::vector<double> sum((size_t)k);
    for (int i = 0; i < n; i++) {
        std::fill(sum.begin(), sum.end(), 0.0);
        for (int j = 0; j < n; j++) {
            if (j == i)
                continue;
            sum[(size_t)labels[(size_t)j]] += distance[(size_t)i * n + j];
        }
        const int own = labels[(size_t)i];
        if (count[(size_t)own] <= 1) {
            // A singleton cluster contributes 0 by convention.
            continue;
        }
        const double a = sum[(size_t)own] / (count[(size_t)own] - 1);
        double b = std::numeric_limits<double>::infinity();
        for (int c = 0; c < k; c++) {
            if (c == own || count[(size_t)c] == 0)
                continue;
            b = std::min(b, sum[(size_t)c] / count[(size_t)c]);
        }
        if (!std::isfinite(b))
            continue;
        const double denom = std::max(a, b);
        if (denom > kTiny)
            total += (b - a) / denom;
    }
    return (float)(total / n);
}

std::vector<int> refine_spherical(const float* x, int n, int d, const std::vector<int>& labels, int max_iter,
                                  int n_threads) {
    if (n <= 0)
        return {};
    std::vector<int> uniq(labels);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const int k = (int)uniq.size();
    if (k <= 1)
        return std::vector<int>((size_t)n, 0);

    std::vector<int> cur((size_t)n, 0);
    for (int i = 0; i < n; i++)
        cur[(size_t)i] = (int)(std::lower_bound(uniq.begin(), uniq.end(), labels[(size_t)i]) - uniq.begin());

    std::vector<float> emb((size_t)n * d);
    std::copy(x, x + (size_t)n * d, emb.begin());
    l2_normalize_rows(emb.data(), n, d, n_threads);

    for (int it = 0; it < max_iter; it++) {
        std::vector<double> cent((size_t)k * d, 0.0);
        std::vector<int> cnt((size_t)k, 0);
        for (int i = 0; i < n; i++) {
            const int c = cur[(size_t)i];
            cnt[(size_t)c]++;
            for (int j = 0; j < d; j++)
                cent[(size_t)c * d + j] += emb[(size_t)i * d + j];
        }
        bool valid = true;
        for (int c = 0; c < k && valid; c++) {
            if (cnt[(size_t)c] == 0) {
                valid = false;
                break;
            }
            double nrm = 0.0;
            for (int j = 0; j < d; j++)
                nrm += cent[(size_t)c * d + j] * cent[(size_t)c * d + j];
            nrm = std::sqrt(nrm);
            if (nrm <= kTiny) {
                valid = false;
                break;
            }
            for (int j = 0; j < d; j++)
                cent[(size_t)c * d + j] /= nrm;
        }
        if (!valid)
            break;

        std::vector<int> next((size_t)n, 0);
        for (int i = 0; i < n; i++) {
            double best = -std::numeric_limits<double>::infinity();
            int arg = 0;
            for (int c = 0; c < k; c++) {
                double dot = 0.0;
                for (int j = 0; j < d; j++)
                    dot += emb[(size_t)i * d + j] * cent[(size_t)c * d + j];
                if (dot > best) {
                    best = dot;
                    arg = c;
                }
            }
            next[(size_t)i] = arg;
        }
        // Refusing to drop a cluster keeps k stable — the caller chose it.
        std::vector<char> seen((size_t)k, 0);
        for (int v : next)
            seen[(size_t)v] = 1;
        if ((int)std::count(seen.begin(), seen.end(), 1) < k)
            break;
        if (next == cur)
            break;
        cur.swap(next);
    }
    return cur;
}

static SpeakerEstimate estimate_speakers_impl(const float* x, int n, int d, int min_k, int max_k, unsigned seed,
                                              const float* sim_precomputed, int n_threads) {
    SpeakerEstimate est;
    est.min_k_used = std::max(1, min_k);
    est.best_k = std::max(1, min_k);

    if (n <= 0) {
        est.reason = "no_embeddings";
        return est;
    }
    if (n < 4) {
        est.reason = "too_few_samples";
        return est;
    }

    // Single-speaker veto: if even the 10th percentile of off-diagonal cosine
    // similarity is high, everything is one voice and BIC would over-split it.
    {
        std::vector<float> sim_local;
        if (!sim_precomputed) {
            sim_local = cosine_similarity(x, n, d, n_threads);
            sim_precomputed = sim_local.data();
        }
        const float* sim = sim_precomputed;
        std::vector<float> off;
        off.reserve((size_t)n * (n - 1));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j)
                    off.push_back(sim[(size_t)i * n + j]);
        const size_t idx = (size_t)(0.10 * (double)(off.size() - 1));
        std::nth_element(off.begin(), off.begin() + (long)idx, off.end());
        est.cosine_sim_p10 = off[idx];
        if (est.cosine_sim_p10 >= kSingleSpeakerSimP10 && min_k <= 1) {
            est.best_k = 1;
            est.reason = "cosine_similarity_single_speaker";
            return est;
        }
    }

    // Projection dimensionality. kPcaDim (8) assumes enough embeddings that a
    // full covariance in 8 dims is estimable; the ceiling below allows only
    // n/(pca_dim+1) components, so on a SHORT recording that silently forces
    // the answer to one speaker. Measured: a VoxConverse file with 4 real
    // speakers yielded 17 embedding windows, giving 17/9 = 1 — k=1 was the
    // only candidate the sweep ever scored, and DER came out at 46.20%.
    //
    // So when the ceiling would bite, spend dimensions to buy components
    // instead: drop pca_dim until at least kMinReachableK components are
    // representable. This is deliberately a no-op whenever n is already large
    // enough (n >= 4 * (kPcaDim + 1) = 36), so the well-populated case that
    // #324 tuned is untouched — of the 8 VoxConverse dev files only the n=17
    // one changes.
    constexpr int kMinReachableK = 4;
    int want_dim = kPcaDim;
    if (n / (kPcaDim + 1) < kMinReachableK)
        want_dim = std::max(2, n / kMinReachableK - 1);

    int pca_dim = 0;
    std::vector<float> proj = pca_project(x, n, d, want_dim, &pca_dim);
    est.pca_dim = pca_dim;
    if (proj.empty()) {
        est.reason = "gmm_failed";
        return est;
    }

    // Component-count ceiling. The upstream recipe uses n/2 + 1 ("at least ~2
    // samples per component"), which is far too loose for a FULL covariance:
    // a d x d covariance is not estimable from fewer than d + 1 points, so
    // beyond n/(d+1) components every extra component fits a near-singular
    // Gaussian whose density — and therefore the likelihood — diverges. BIC
    // then falls monotonically and the sweep runs away to max_k.
    //
    // Measured on 3 well-separated blobs (n=60, pca_dim=8): with the n/2 bound
    // BIC dropped 881 -> 138 straight through k=10 and the anchor was 10;
    // the silhouette window [k-2, k+3] could then never reach the true k=3.
    // reg_covar does not save this — sklearn's 1e-6 default is negligible
    // against PCA noise-component variances of ~0.03.
    const int k_occupancy = std::max(1, n / (pca_dim + 1));
    const int k_upper = std::max(min_k + 1, std::min({max_k + 1, n / 2 + 1, k_occupancy + 1}));
    // Every k's GMM is seeded independently (seed + init*7919 inside gmm_bic)
    // and touches only its own slot, so the sweep parallelises without
    // changing a single BIC; the argmin below runs serially in ascending k,
    // preserving the old first-wins tie-break.
    const int k_lo = std::max(1, min_k);
    const int k_cnt = std::max(0, k_upper - k_lo);
    std::vector<float> bics((size_t)k_cnt, 0.0f);
    std::vector<char> bic_ok((size_t)k_cnt, 0);
    core_parallel::for_each_task(k_cnt, n_threads, [&](int i, int) {
        float bic = 0.0f;
        if (gmm_bic(proj.data(), n, pca_dim, k_lo + i, kGmmNInit, kGmmMaxIter, seed, &bic)) {
            bics[(size_t)i] = bic;
            bic_ok[(size_t)i] = 1;
        }
    });
    float best_bic = std::numeric_limits<float>::infinity();
    int best_k = min_k;
    bool any = false;
    for (int i = 0; i < k_cnt; i++) {
        if (!bic_ok[(size_t)i])
            continue;
        est.k_bics.push_back(bics[(size_t)i]);
        if (bics[(size_t)i] < best_bic) {
            best_bic = bics[(size_t)i];
            best_k = k_lo + i;
        }
        any = true;
    }
    if (!any) {
        est.reason = "gmm_failed";
        return est;
    }
    est.best_k = best_k;
    est.reason = "gmm_bic";
    return est;
}

SpeakerEstimate estimate_speakers(const float* x, int n, int d, int min_k, int max_k, unsigned seed, int n_threads) {
    return estimate_speakers_impl(x, n, d, min_k, max_k, seed, nullptr, n_threads);
}

std::vector<int> cluster_speakers(const float* x, int n, int d, int min_speakers, int max_speakers, int num_speakers,
                                  SpeakerEstimate* out_estimate, unsigned seed, int n_threads) {
    if (n <= 0)
        return {};
    if (n < 2)
        return std::vector<int>((size_t)n, 0);

    // ONE O(n^2 d) similarity pass for the whole function. Everything below —
    // the affinity every spectral run clusters on, the p10 single-speaker veto
    // inside the estimator, and the silhouette's distance matrix — is a cheap
    // transform of this one matrix; it used to be recomputed per spectral run.
    std::vector<float> sim = cosine_similarity(x, n, d, n_threads);
    std::vector<float> aff = sim;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float v = (aff[(size_t)i * n + j] + 1.0f) * 0.5f;
            aff[(size_t)i * n + j] = std::max(v, 0.0f);
        }
    for (int i = 0; i < n; i++)
        aff[(size_t)i * n + i] = 1.0f;

    auto run = [&](int k) {
        std::vector<int> lab = spectral_labels(aff.data(), n, k, seed);
        return refine_spherical(x, n, d, lab, 8, n_threads);
    };

    if (num_speakers > 0) {
        if (out_estimate) {
            out_estimate->best_k = num_speakers;
            out_estimate->reason = "fixed";
        }
        return run(num_speakers);
    }

    SpeakerEstimate est;
    if (count_method_from_env() == CountMethod::NmeSc) {
        NmeScDiag diag;
        est.best_k = estimate_speakers_nme_sc(aff.data(), n, min_speakers, max_speakers, seed, &diag);
        est.reason = "nme-sc";
        if (out_estimate)
            *out_estimate = est;
        if (std::getenv("CRISPASR_DIARIZE_DEBUG")) {
            fprintf(stderr, "  nme-sc: p*=%d k=%d r=%.4f\n", diag.best_p, diag.best_k, diag.best_r);
            for (const auto& pt : diag.curve)
                fprintf(stderr, "    p=%-4d k=%-2d gap=%.4f  r=%.4f%s\n", pt.p, pt.k, pt.gap, pt.ratio,
                        pt.p == diag.best_p ? "   <-- chosen" : "");
        }
        // Like eigengap, this reads structure off the spectrum, so the
        // silhouette pass (which is what over-merges) is skipped.
        return run(est.best_k);
    }
    if (count_method_from_env() == CountMethod::Eigengap) {
        est.best_k = estimate_speakers_eigengap(aff.data(), n, min_speakers, max_speakers, seed);
        est.reason = "eigengap";
        if (out_estimate)
            *out_estimate = est;
        // The eigengap already reads cluster structure off the spectrum, so
        // the silhouette pass — which is what saturates — is skipped.
        return run(est.best_k);
    }
    est = estimate_speakers_impl(x, n, d, min_speakers, max_speakers, seed, sim.data(), n_threads);
    if (out_estimate)
        *out_estimate = est;
    const int k = est.best_k;
    if (k < 2)
        return std::vector<int>((size_t)n, 0);

    // Silhouette refinement. The upstream recipe scores a small neighbourhood
    // of the BIC anchor, [k-2, k+3]. That recovers an UNDER-count (measured:
    // true k = 4/5/6 anchored at 2/2/3 and were all recovered exactly) but not
    // a large OVER-count, because the window cannot reach back far enough —
    // true k = 3 anchored at 8 leaves the window at [6, 10].
    //
    // Silhouette itself is the reliable half: on 3 well-separated blobs it
    // scored k=3 at 1.0390 against 0.6827 / 0.7926 for its neighbours.
    //
    // The full range is now the DEFAULT, and the [k-2, k+3] window is the opt-in.
    // That inverts the earlier gate, on measurement:
    //
    // On 4/5/6 well-separated blobs (per=25, d=32, sep=6.0) the BIC anchor came
    // out at 9/2/3 — errors of +5, -3, -3 — while the silhouette scored over the
    // FULL range peaked at exactly 4/5/6 every time (k=4: 0.4471 at the truth vs
    // 0.1050 at k=8). The anchor is unreliable in BOTH directions, and when it
    // over-counts the window is stranded above the answer and cannot climb back:
    // anchor 9 leaves [7,10], which simply does not contain 4.
    //
    // A guard on "anchor == max" is not enough — the same run anchored at 9 with
    // max 10, one short of the ceiling, and was stranded just the same. There is
    // no bound on how far BIC over-counts, so any lower bound derived from the
    // anchor can strand the search. Dropping that dependency removes the whole
    // failure mode instead of moving its threshold.
    //
    // Cost is (max-min) spectral runs instead of 6 — 9 vs 6 at the default
    // max_speakers=10. The upper bound stays anchored so absurdly high k are
    // still never scored.
    //
    // CRISPASR_DIARIZE_BIC_WINDOW=1 restores the old anchored window for A/B
    // work; the DER harness should confirm this on real meetings.
    const char* win_env = std::getenv("CRISPASR_DIARIZE_BIC_WINDOW");
    const bool anchored_window = win_env && *win_env && *win_env != '0';

    const int lower = anchored_window ? std::max({2, min_speakers, k - 2}) : std::max(2, min_speakers);
    const int upper = anchored_window ? std::min({max_speakers, n - 1, k + 3}) : std::min({max_speakers, n - 1});
    if (upper <= lower)
        return run(k);

    std::vector<float> dist((size_t)n * n);
    for (size_t i = 0; i < dist.size(); i++)
        dist[i] = std::max(1.0f - aff[i], 0.0f);

    // Every k's spectral run + silhouette is independent (read-only inputs,
    // per-call RNG seeded from the same `seed`), so the sweep parallelises
    // without moving a single score; the argmax below stays serial in
    // ascending k, preserving the old strictly-greater tie-break.
    const int n_cand = upper - lower + 1;
    std::vector<std::vector<int>> cand_labels((size_t)n_cand);
    std::vector<float> cand_sil((size_t)n_cand, 0.0f);
    core_parallel::for_each_task(n_cand, n_threads, [&](int i, int) {
        const int c = lower + i;
        // The k-sweep already owns the worker budget; normalization inside
        // each candidate must not create another team (including std::thread).
        cand_labels[(size_t)i] = refine_spherical(x, n, d, spectral_labels(aff.data(), n, c, seed), 8, 1);
        cand_sil[(size_t)i] = silhouette_precomputed(dist.data(), n, cand_labels[(size_t)i]);
    });

    int best_k = k;
    std::vector<int> best_labels;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_cand; i++) {
        const int c = lower + i;
        const float sil = cand_sil[(size_t)i];
        const float score = sil + kSilhouetteKBonus * (float)std::log((double)std::max(c, 1));
        if (std::getenv("CRISPASR_DIARIZE_DEBUG"))
            fprintf(stderr, "  spectral: k=%d silhouette=%.4f score=%.4f%s\n", c, sil, score,
                    c == k ? "   (bic anchor)" : "");
        if (score > best_score) {
            best_score = score;
            best_k = c;
            best_labels = std::move(cand_labels[(size_t)i]);
        }
    }
    if (out_estimate && best_k != k) {
        out_estimate->best_k = best_k;
        out_estimate->reason = "silhouette";
    }
    return best_labels;
}

} // namespace core_spectral
