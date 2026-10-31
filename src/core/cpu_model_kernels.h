#pragma once

#include "cpu_attention.h"
#include "parallel_for.h"

#include <cstddef>
#include <cstdint>

namespace core_cpu_kernels {

// FireRed CPU fallback: A[M,K] @ transpose(B[N,K]). Each dot product is
// unchanged by scheduling; tests call this same production implementation.
inline void matmul_bt(const float* A, const float* B, float* C, int M, int K, int N, int n_threads) {
    if (M == 1) {
        core_parallel::for_each_chunk_work(N, n_threads, K, 65536, [&](int begin, int end) {
            for (int n = begin; n < end; ++n)
                C[n] = crispasr::cpu_attn::cpu_dot(A, B + (size_t)n * K, K);
        });
    } else {
        core_parallel::for_each_chunk_work(M, n_threads, (uint64_t)N * K, 65536, [&](int begin, int end) {
            for (int m = begin; m < end; ++m) {
                const float* row = A + (size_t)m * K;
                for (int n = 0; n < N; ++n)
                    C[(size_t)m * N + n] = crispasr::cpu_attn::cpu_dot(row, B + (size_t)n * K, K);
            }
        });
    }
}

// F5-TTS HiFi-GAN CPU fallback. Same-padding, groups=1; retains the original
// input-channel/kernel accumulation order, including the boundary skips.
inline void conv1d_dil(const float* input, int C_in, int T, const float* weight, const float* bias, int C_out, int K,
                       int dilation, float* output, int n_threads) {
    const int pad = dilation * (K - 1) / 2;
    core_parallel::for_each_chunk_work(C_out, n_threads, (uint64_t)T * C_in * K, 65536, [&](int begin, int end) {
        for (int oc = begin; oc < end; ++oc) {
            for (int t = 0; t < T; ++t) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < C_in; ++ic) {
                    const float* in = input + (size_t)ic * T;
                    const float* w = weight + ((size_t)oc * C_in + ic) * K;
                    for (int k = 0; k < K; ++k) {
                        const int ti = t + k * dilation - pad;
                        if (ti >= 0 && ti < T)
                            sum += in[ti] * w[k];
                    }
                }
                output[(size_t)oc * T + t] = sum;
            }
        }
    });
}

} // namespace core_cpu_kernels
