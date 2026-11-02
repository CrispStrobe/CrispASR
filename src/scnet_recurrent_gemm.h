// Immutable F32 packing and independent-row recurrent matrix products.
// No provider calls, worker synchronization or allocation occur in a product.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SCNET_RECURRENT_X86 1
#define SCNET_RECURRENT_AVX2 __attribute__((target("avx2,fma")))
#define SCNET_RECURRENT_AVX512 __attribute__((target("avx512f,avx512dq,avx2,fma")))
#else
#define SCNET_RECURRENT_X86 0
#endif

namespace scnet_recurrent {

constexpr int GATE_BLOCK = 16;

struct PackedWeight {
    int hidden;
    std::vector<float> values;

    PackedWeight(const float* source, int width) : hidden(width) {
        if (!source || width <= 0 || width > std::numeric_limits<int>::max() / 4)
            throw std::runtime_error("invalid SCNet recurrent packing shape");
        const int gates = 4 * width;
        const size_t blocks = size_t(gates / GATE_BLOCK) + (gates % GATE_BLOCK != 0);
        if (blocks > std::numeric_limits<size_t>::max() / GATE_BLOCK / size_t(width))
            throw std::runtime_error("SCNet recurrent packing size overflow");
        values.resize(blocks * size_t(width) * GATE_BLOCK, 0.0f);
        // Original [4H,H] -> [ceil(4H/16),H,16]. Every stored coefficient is
        // copied without arithmetic; only an incomplete gate block is padded.
        // The validated model widths need no padding (15/60 MiB in total).
        for (size_t block = 0; block < blocks; ++block)
            for (int k = 0; k < width; ++k)
                for (int lane = 0; lane < GATE_BLOCK; ++lane) {
                    const size_t gate = block * GATE_BLOCK + size_t(lane);
                    if (gate < size_t(gates))
                        std::memcpy(values.data() + (block * width + k) * GATE_BLOCK + lane, source + gate * width + k,
                                    sizeof(float));
                }
    }
};

// hidden is [rows,H], output [rows,4H]. A caller owns every row it passes.
// SIMD lanes represent independent gates, not a reduction over hidden values.
// Each result uses the same increasing-k F32 FMA order on AVX2 and AVX512.
using Product = void (*)(const float*, const PackedWeight&, float*, int);
struct Kernel {
    Product product = nullptr;
    const char* name = "none";
};

#if SCNET_RECURRENT_X86
namespace detail {

template <int Rows>
SCNET_RECURRENT_AVX2 inline void avx2_tile(const float* hidden, const float* packed, float* output, int width,
                                           int valid_gates) {
    __m256 low[Rows], high[Rows];
    for (int row = 0; row < Rows; ++row)
        low[row] = high[row] = _mm256_setzero_ps();
    for (int k = 0; k < width; ++k) {
        const __m256 w0 = _mm256_loadu_ps(packed + size_t(k) * GATE_BLOCK);
        const __m256 w1 = _mm256_loadu_ps(packed + size_t(k) * GATE_BLOCK + 8);
        for (int row = 0; row < Rows; ++row) {
            const __m256 h = _mm256_set1_ps(hidden[size_t(row) * width + k]);
            low[row] = _mm256_fmadd_ps(h, w0, low[row]);
            high[row] = _mm256_fmadd_ps(h, w1, high[row]);
        }
    }
    for (int row = 0; row < Rows; ++row) {
        float* target = output + size_t(row) * 4 * width;
        if (valid_gates == GATE_BLOCK) {
            _mm256_storeu_ps(target, low[row]);
            _mm256_storeu_ps(target + 8, high[row]);
        } else {
            float tail[GATE_BLOCK];
            _mm256_storeu_ps(tail, low[row]);
            _mm256_storeu_ps(tail + 8, high[row]);
            std::memcpy(target, tail, size_t(valid_gates) * sizeof(float));
        }
    }
}

template <int Rows>
SCNET_RECURRENT_AVX512 inline void avx512_tile(const float* hidden, const float* packed, float* output, int width,
                                               int valid_gates) {
    __m512 sum[Rows];
    for (int row = 0; row < Rows; ++row)
        sum[row] = _mm512_setzero_ps();
    for (int k = 0; k < width; ++k) {
        const __m512 w = _mm512_loadu_ps(packed + size_t(k) * GATE_BLOCK);
        for (int row = 0; row < Rows; ++row) {
            const __m512 h = _mm512_set1_ps(hidden[size_t(row) * width + k]);
            sum[row] = _mm512_fmadd_ps(h, w, sum[row]);
        }
    }
    for (int row = 0; row < Rows; ++row) {
        float* target = output + size_t(row) * 4 * width;
        if (valid_gates == GATE_BLOCK) {
            _mm512_storeu_ps(target, sum[row]);
        } else {
            float tail[GATE_BLOCK];
            _mm512_storeu_ps(tail, sum[row]);
            std::memcpy(target, tail, size_t(valid_gates) * sizeof(float));
        }
    }
}
} // namespace detail

SCNET_RECURRENT_AVX2 inline void avx2_product(const float* hidden, const PackedWeight& weight, float* output,
                                              int rows) {
    if (rows <= 0)
        return;
    const int width = weight.hidden, gates = 4 * width;
    // Gate blocks remain cache-resident while all independent batch rows use
    // them. Specialize the final row group too: the common 28/29 and 42/43
    // worker partitions must not fall back to scalar gate dot products.
    for (int first = 0; first < gates; first += GATE_BLOCK) {
        const float* packed = weight.values.data() + size_t(first / GATE_BLOCK) * width * GATE_BLOCK;
        const int valid = std::min(GATE_BLOCK, gates - first);
        int row = 0;
        for (; row + 3 < rows; row += 4)
            detail::avx2_tile<4>(hidden + size_t(row) * width, packed, output + size_t(row) * gates + first, width,
                                 valid);
        if (row == rows)
            continue;
        const auto* source = hidden + size_t(row) * width;
        auto* target = output + size_t(row) * gates + first;
        switch (rows - row) {
        case 3:
            detail::avx2_tile<3>(source, packed, target, width, valid);
            break;
        case 2:
            detail::avx2_tile<2>(source, packed, target, width, valid);
            break;
        case 1:
            detail::avx2_tile<1>(source, packed, target, width, valid);
            break;
        }
    }
}

SCNET_RECURRENT_AVX512 inline void avx512_product(const float* hidden, const PackedWeight& weight, float* output,
                                                  int rows) {
    if (rows <= 0)
        return;
    const int width = weight.hidden, gates = 4 * width;
    for (int first = 0; first < gates; first += GATE_BLOCK) {
        const float* packed = weight.values.data() + size_t(first / GATE_BLOCK) * width * GATE_BLOCK;
        const int valid = std::min(GATE_BLOCK, gates - first);
        int row = 0;
        for (; row + 7 < rows; row += 8)
            detail::avx512_tile<8>(hidden + size_t(row) * width, packed, output + size_t(row) * gates + first, width,
                                   valid);
        if (row == rows)
            continue;
        const auto* source = hidden + size_t(row) * width;
        auto* target = output + size_t(row) * gates + first;
        switch (rows - row) {
        case 7:
            detail::avx512_tile<7>(source, packed, target, width, valid);
            break;
        case 6:
            detail::avx512_tile<6>(source, packed, target, width, valid);
            break;
        case 5:
            detail::avx512_tile<5>(source, packed, target, width, valid);
            break;
        case 4:
            detail::avx512_tile<4>(source, packed, target, width, valid);
            break;
        case 3:
            detail::avx512_tile<3>(source, packed, target, width, valid);
            break;
        case 2:
            detail::avx512_tile<2>(source, packed, target, width, valid);
            break;
        case 1:
            detail::avx512_tile<1>(source, packed, target, width, valid);
            break;
        }
    }
}

inline bool avx2_supported() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
inline bool avx512_supported() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") && avx2_supported();
}
#endif

inline Kernel selected_kernel() {
    static const Kernel selected = [] {
#if SCNET_RECURRENT_X86
        if (avx512_supported())
            return Kernel{avx512_product, "packed_avx512_fma"};
        if (avx2_supported())
            return Kernel{avx2_product, "packed_avx2_fma"};
#endif
        return Kernel{};
    }();
    return selected;
}

} // namespace scnet_recurrent

#undef SCNET_RECURRENT_AVX2
#undef SCNET_RECURRENT_AVX512
