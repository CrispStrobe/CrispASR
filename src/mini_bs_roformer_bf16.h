// Backend-private BF16 operands / F32 accumulation GEMM for Mini-BS-RoFormer.
// Each packed weight vector holds two K values for sixteen output features.
// Large matrices reuse activation blocks across output panels with an 8x32
// microkernel; small matrices retain feature-parallel six-row microkernels.
#pragma once

#include "core/ggml_cpu_backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>

#if defined(__linux__) && defined(__x86_64__) &&                                                                       \
    ((defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 10) || (defined(__clang__) && __clang_major__ >= 12))
#define CRISPASR_MINI_BS_BF16_INTRINSICS 1
#include <immintrin.h>
#define CRISPASR_MINI_BS_BF16_TARGET __attribute__((target("avx512f,avx512bf16")))
#else
#define CRISPASR_MINI_BS_BF16_INTRINSICS 0
#define CRISPASR_MINI_BS_BF16_TARGET
#endif

namespace mini_bs_bf16 {

inline bool hardware_supported() {
#if CRISPASR_MINI_BS_BF16_INTRINSICS
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bf16");
#else
    return false;
#endif
}

inline bool supported(ggml_backend_t cpu) {
    return core_cpu_backend::is_cpu(cpu) && hardware_supported();
}

#if CRISPASR_MINI_BS_BF16_INTRINSICS
CRISPASR_MINI_BS_BF16_TARGET inline __m512bh broadcast_pair(const ggml_bf16_t* x) {
    uint32_t bits;
    // memcpy avoids reading BF16 storage through an aliased uint32_t pointer.
    std::memcpy(&bits, x, sizeof(bits));
    return (__m512bh)_mm512_set1_epi32(static_cast<int32_t>(bits));
}

template <int Row, int Rows, int Panels>
CRISPASR_MINI_BS_BF16_TARGET __attribute__((always_inline)) inline void initialize(__m512 (&acc)[Rows][Panels]) {
    for (int panel = 0; panel < Panels; ++panel)
        acc[Row][panel] = _mm512_setzero_ps();
    if constexpr (Row + 1 < Rows)
        initialize<Row + 1>(acc);
}

template <int Row, int Rows, int Panels>
CRISPASR_MINI_BS_BF16_TARGET __attribute__((always_inline)) inline void update(__m512 (&acc)[Rows][Panels],
                                                                               const __m512bh (&w)[Panels],
                                                                               const ggml_bf16_t* x, int64_t k) {
    const __m512bh value = broadcast_pair(x + Row * k);
    for (int panel = 0; panel < Panels; ++panel)
        acc[Row][panel] = _mm512_dpbf16_ps(acc[Row][panel], w[panel], value);
    if constexpr (Row + 1 < Rows)
        update<Row + 1>(acc, w, x, k);
}

template <int Row, int Rows, int Panels>
CRISPASR_MINI_BS_BF16_TARGET __attribute__((always_inline)) inline void store(const __m512 (&acc)[Rows][Panels],
                                                                              float* out, int64_t n) {
    for (int panel = 0; panel < Panels; ++panel)
        _mm512_storeu_ps(out + Row * n + panel * 16, acc[Row][panel]);
    if constexpr (Row + 1 < Rows)
        store<Row + 1>(acc, out, n);
}

template <int Rows, int Cols>
CRISPASR_MINI_BS_BF16_TARGET inline void kernel(const ggml_bf16_t* weights, const ggml_bf16_t* x, int64_t k, float* out,
                                                int64_t n) {
    constexpr int panels = Cols / 16;
    static_assert((Rows == 1 || Rows == 6 || Rows == 8) && (Cols == 16 || Cols == 32), "supported microkernel shape");
    __m512 acc[Rows][panels];
    initialize<0>(acc);
    for (int64_t pair = 0; pair < k / 2; ++pair) {
        __m512bh w[panels];
        for (int panel = 0; panel < panels; ++panel)
            w[panel] = (__m512bh)_mm512_loadu_si512(weights + panel * k * 16 + pair * 32);
        update<0>(acc, w, x + pair * 2, k);
    }
    store<0>(acc, out, n);
}

// Workers own disjoint feature tiles, including the final sixteen-feature
// panel. Inputs are read-only and each output element is assigned once.
template <int Cols>
CRISPASR_MINI_BS_BF16_TARGET inline void multiply_tiled(const ggml_bf16_t* weights, const ggml_bf16_t* x, float* out,
                                                        int64_t n, int64_t m, int64_t k, int ith, int nth) {
    for (int64_t tile = ith; tile < (n + Cols - 1) / Cols; tile += nth) {
        const int64_t col = tile * Cols;
        const ggml_bf16_t* w = weights + col * k;
        int64_t row = 0;
        if (col + Cols <= n) {
            for (; row + 6 <= m; row += 6)
                kernel<6, Cols>(w, x + row * k, k, out + row * n + col, n);
            for (; row < m; ++row)
                kernel<1, Cols>(w, x + row * k, k, out + row * n + col, n);
        } else {
            for (; row + 6 <= m; row += 6)
                kernel<6, 16>(w, x + row * k, k, out + row * n + col, n);
            for (; row < m; ++row)
                kernel<1, 16>(w, x + row * k, k, out + row * n + col, n);
        }
    }
}

// Workers own disjoint 96-row blocks. Visiting every output panel within a
// block reuses the activation rows without changing any dot product's K order.
CRISPASR_MINI_BS_BF16_TARGET inline void multiply_rows(const ggml_bf16_t* weights, const ggml_bf16_t* x, float* out,
                                                       int64_t n, int64_t m, int64_t k, int ith, int nth) {
    constexpr int64_t block = 96;
    for (int64_t begin = ith * block; begin < m; begin += nth * block) {
        const int64_t end = begin + block < m ? begin + block : m;
        for (int64_t col = 0; col < n; col += 32) {
            const ggml_bf16_t* w = weights + col * k;
            int64_t row = begin;
            if (col + 32 <= n) {
                for (; row + 8 <= end; row += 8)
                    kernel<8, 32>(w, x + row * k, k, out + row * n + col, n);
                for (; row < end; ++row)
                    kernel<1, 32>(w, x + row * k, k, out + row * n + col, n);
            } else {
                for (; row + 8 <= end; row += 8)
                    kernel<8, 16>(w, x + row * k, k, out + row * n + col, n);
                for (; row < end; ++row)
                    kernel<1, 16>(w, x + row * k, k, out + row * n + col, n);
            }
        }
    }
}

CRISPASR_MINI_BS_BF16_TARGET inline void multiply(const ggml_bf16_t* weights, const ggml_bf16_t* x, float* out,
                                                  int64_t n, int64_t m, int64_t k, int ith, int nth) {
    // Keep every worker occupied when there are too few activation blocks.
    if (m >= int64_t(96) * nth)
        multiply_rows(weights, x, out, n, m, k, ith, nth);
    else if (k <= 512)
        multiply_tiled<16>(weights, x, out, n, m, k, ith, nth);
    else
        multiply_tiled<32>(weights, x, out, n, m, k, ith, nth);
}

inline void compute(ggml_tensor* out, int ith, int nth, void*) {
    const ggml_tensor* weights = out->src[0];
    const ggml_tensor* x = out->src[1];
    GGML_ASSERT(weights->type == GGML_TYPE_BF16 && x->type == GGML_TYPE_BF16 && out->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(x) && ggml_is_contiguous(out));
    multiply(static_cast<const ggml_bf16_t*>(weights->data), static_cast<const ggml_bf16_t*>(x->data),
             static_cast<float*>(out->data), out->ne[0], out->ne[1], x->ne[0], ith, nth);
}
#endif

// Original BF16 weight shape is [K,N]. The packed graph value has shape
// [2,16,K/2,N/16], in ggml's dimension order. Rounding has already occurred.
inline ggml_tensor* pack(ggml_context* ctx, ggml_tensor* weight) {
    if (!hardware_supported() || !weight || weight->type != GGML_TYPE_BF16 || !ggml_is_contiguous(weight) ||
        weight->ne[0] <= 0 || weight->ne[0] % 2 || weight->ne[1] <= 0 || weight->ne[1] % 16 || weight->ne[2] != 1 ||
        weight->ne[3] != 1)
        return nullptr;
    auto* split = ggml_reshape_4d(ctx, weight, 2, weight->ne[0] / 2, 16, weight->ne[1] / 16);
    return ggml_cont(ctx, ggml_permute(ctx, split, 0, 2, 1, 3));
}

// A nullptr result requests the existing portable ggml matmul path. Callers
// also require supported(selected_backend) before selecting this CPU graph.
inline ggml_tensor* graph(ggml_context* ctx, ggml_tensor* packed_weight, ggml_tensor* x, int64_t original_n) {
#if CRISPASR_MINI_BS_BF16_INTRINSICS
    if (!hardware_supported() || !packed_weight || !x || original_n <= 0 || original_n % 16 ||
        packed_weight->type != GGML_TYPE_BF16 || x->type != GGML_TYPE_BF16 || !ggml_is_contiguous(packed_weight) ||
        !ggml_is_contiguous(x) || x->ne[0] <= 0 || x->ne[0] % 2 || x->ne[1] <= 0 || x->ne[2] != 1 || x->ne[3] != 1 ||
        packed_weight->ne[0] != 2 || packed_weight->ne[1] != 16 || packed_weight->ne[2] != x->ne[0] / 2 ||
        packed_weight->ne[3] != original_n / 16)
        return nullptr;
    ggml_tensor* inputs[] = {packed_weight, x};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, original_n, x->ne[1], 1, 1, inputs, 2, compute, GGML_N_TASKS_MAX,
                          nullptr);
#else
    (void)ctx;
    (void)packed_weight;
    (void)x;
    (void)original_n;
    return nullptr;
#endif
}

} // namespace mini_bs_bf16

#undef CRISPASR_MINI_BS_BF16_TARGET
#undef CRISPASR_MINI_BS_BF16_INTRINSICS
