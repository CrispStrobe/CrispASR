// core_cpu_linear — out[T,N] = x[T,K] @ w[N,K]^T + b[N] on the host CPU.
//
// One cblas_sgemm when the including target links crispasr-cpu-blas
// (HAVE_ACCELERATE on Apple, HAVE_BLAS with OpenBLAS/MKL elsewhere), else a
// single-threaded double-accumulate loop. Split out of firered_vad.cpp
// (PR #611) so the BLAS path can be checked against the scalar loop in a
// unit test without a model.
#pragma once

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define CORE_CPU_LINEAR_BLAS 1
#elif defined(HAVE_BLAS)
#include <cblas.h>
#define CORE_CPU_LINEAR_BLAS 1
#else
#define CORE_CPU_LINEAR_BLAS 0
#endif

namespace core_cpu_linear {

// True when this translation unit was compiled with a CBLAS path.
constexpr bool have_blas() {
    return CORE_CPU_LINEAR_BLAS != 0;
}

// w is row-major [N,K]; b may be null. force_scalar takes the loop even when
// BLAS is available (CRISPASR_*_FORCE_SCALAR validation switches).
inline void run(const float* x, const float* w, const float* b, float* out, int T, int K, int N,
                bool force_scalar = false) {
#if CORE_CPU_LINEAR_BLAS
    if (!force_scalar) {
        // w stored as [N,K] -> CblasTrans gives x[T,K] @ w^T[K,N] -> out[T,N]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, N, K, 1.0f, x, K, w, K, 0.0f, out, N);
        if (b) {
            for (int t = 0; t < T; t++)
                for (int n = 0; n < N; n++)
                    out[t * N + n] += b[n];
        }
        return;
    }
#else
    (void)force_scalar;
#endif
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += x[t * K + k] * w[n * K + k];
            out[t * N + n] = (float)s + (b ? b[n] : 0.0f);
        }
    }
}

} // namespace core_cpu_linear
