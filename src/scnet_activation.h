// F32 gate activations for SCNet's native recurrent operation.
//
// The SIMD exponential below is adapted from the pinned ggml implementation:
// ggml/src/ggml-cpu/vec.h at e2ae679a563014f46c546ce6c6448e8601e4dea6,
// whose ggml_v_expf is adapted from Arm's optimized routine. Keep this
// provenance and the ggml MIT notice when moving or reusing these kernels.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SCNET_ACT_X86 1
#define SCNET_ACT_AVX2 __attribute__((target("avx2,fma")))
#define SCNET_ACT_AVX512 __attribute__((target("avx512f,avx512dq,avx2,fma")))
#else
#define SCNET_ACT_X86 0
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SCNET_ACT_NEON 1
#else
#define SCNET_ACT_NEON 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define SCNET_ACT_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define SCNET_ACT_NO_CONTRACT
#endif
#if defined(__clang__)
#define SCNET_ACT_PRECISE _Pragma("clang fp contract(off)")
#else
#define SCNET_ACT_PRECISE
#endif

namespace scnet_activation {

using Update = void (*)(const float*, float*, float*, float*, int);
struct Kernel {
    Update update;
    const char* name;
};

SCNET_ACT_NO_CONTRACT inline void scalar_range(const float* gates, float* cell, float* hidden, float* output, int width,
                                               int begin) {
    SCNET_ACT_PRECISE
    for (int k = begin; k < width; ++k) {
        const float i = 1.0f / (1.0f + std::exp(-gates[k]));
        const float f = 1.0f / (1.0f + std::exp(-gates[width + k]));
        const float g = std::tanh(gates[2 * width + k]);
        const float o = 1.0f / (1.0f + std::exp(-gates[3 * width + k]));
        cell[k] = f * cell[k] + i * g;
        hidden[k] = o * std::tanh(cell[k]);
        output[k] = hidden[k];
    }
}

inline void scalar_update(const float* gates, float* cell, float* hidden, float* output, int width) {
    scalar_range(gates, cell, hidden, output, width, 0);
}

#if SCNET_ACT_X86
// ggml's vector exp approximation, including its large-exponent handling.
SCNET_ACT_AVX2 inline __m256 exp_avx2(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b =
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f), _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c =
        _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
        return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                       _mm256_set1_epi32(0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d =
        _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                         _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

SCNET_ACT_AVX2 inline __m256 sigmoid_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_avx2(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

SCNET_ACT_AVX2 inline __m256 tanh_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f), sign = _mm256_set1_ps(-0.0f);
    const __m256 magnitude = _mm256_andnot_ps(sign, x);
    // Near zero, an odd Taylor polynomial avoids subtracting two almost equal
    // numbers. Through x^17 the next term is < 4.6e-10 on [-0.5, 0.5].
    const __m256 polynomial_magnitude = _mm256_min_ps(magnitude, _mm256_set1_ps(0.5f));
    const __m256 square = _mm256_mul_ps(polynomial_magnitude, polynomial_magnitude);
    __m256 p = _mm256_set1_ps(6404582.0f / 10854718875.0f);
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(-929569.0f / 638512875.0f));
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(21844.0f / 6081075.0f));
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(-1382.0f / 155925.0f));
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(62.0f / 2835.0f));
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(-17.0f / 315.0f));
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(2.0f / 15.0f));
    p = _mm256_fmadd_ps(p, square, _mm256_set1_ps(-1.0f / 3.0f));
    const __m256 small = _mm256_mul_ps(x, _mm256_fmadd_ps(p, square, one));
    // All |x| >= 20 already round to +/-1 in F32. Clamping before the
    // exponential also avoids overflowing -2*|x| for finite large inputs.
    const __m256 bounded = _mm256_min_ps(magnitude, _mm256_set1_ps(20.0f));
    const __m256 e = exp_avx2(_mm256_mul_ps(_mm256_set1_ps(-2.0f), bounded));
    const __m256 positive = _mm256_div_ps(_mm256_sub_ps(one, e), _mm256_add_ps(one, e));
    const __m256 large = _mm256_xor_ps(positive, _mm256_and_ps(sign, x));
    const __m256 result = _mm256_blendv_ps(large, small, _mm256_cmp_ps(magnitude, _mm256_set1_ps(0.5f), _CMP_LE_OQ));
    return _mm256_blendv_ps(result, x, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
}

SCNET_ACT_AVX2 SCNET_ACT_NO_CONTRACT inline void avx2_update(const float* gates, float* cell, float* hidden,
                                                             float* output, int width) {
    SCNET_ACT_PRECISE
    int k = 0;
    for (; k + 7 < width; k += 8) {
        const __m256 i = sigmoid_avx2(_mm256_loadu_ps(gates + k));
        const __m256 f = sigmoid_avx2(_mm256_loadu_ps(gates + width + k));
        const __m256 g = tanh_avx2(_mm256_loadu_ps(gates + 2 * width + k));
        const __m256 o = sigmoid_avx2(_mm256_loadu_ps(gates + 3 * width + k));
        const __m256 next_cell = _mm256_add_ps(_mm256_mul_ps(f, _mm256_loadu_ps(cell + k)), _mm256_mul_ps(i, g));
        const __m256 next_hidden = _mm256_mul_ps(o, tanh_avx2(next_cell));
        _mm256_storeu_ps(cell + k, next_cell);
        _mm256_storeu_ps(hidden + k, next_hidden);
        _mm256_storeu_ps(output + k, next_hidden);
    }
    scalar_range(gates, cell, hidden, output, width, k);
}

// A lane-width extension of the AVX2 equations above. Mask selection replaces
// the corresponding bit-mask blends; all F32 constants and FMA groupings stay
// unchanged. AVX512DQ is required for the floating-point bitwise intrinsics.
SCNET_ACT_AVX512 inline __m512 exp_avx512(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b =
        _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __m512i e = _mm512_slli_epi32(_mm512_castps_si512(z), 23);
    const __m512 k = _mm512_castsi512_ps(_mm512_add_epi32(e, _mm512_castps_si512(_mm512_set1_ps(1))));
    const __mmask16 c = _mm512_cmp_ps_mask(_mm512_andnot_ps(_mm512_set1_ps(-0.f), n), _mm512_set1_ps(126), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_mul_ps(_mm512_set1_ps(0x1.ffffecp-1f), b));
    if (!c)
        return _mm512_fmadd_ps(j, k, k);
    const __m512i g =
        _mm512_maskz_mov_epi32(_mm512_cmp_ps_mask(n, _mm512_setzero_ps(), _CMP_LE_OQ), _mm512_set1_epi32(0x82000000u));
    const __m512 s1 = _mm512_castsi512_ps(_mm512_add_epi32(g, _mm512_set1_epi32(0x7f000000u)));
    const __m512 s2 = _mm512_castsi512_ps(_mm512_sub_epi32(e, g));
    const __mmask16 d = _mm512_cmp_ps_mask(_mm512_andnot_ps(_mm512_set1_ps(-0.f), n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 expanded =
        _mm512_mask_blend_ps(c, _mm512_fmadd_ps(k, j, k), _mm512_mul_ps(_mm512_fmadd_ps(s2, j, s2), s1));
    return _mm512_mask_blend_ps(d, expanded, _mm512_mul_ps(s1, s1));
}

SCNET_ACT_AVX512 inline __m512 sigmoid_avx512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(one, _mm512_add_ps(one, exp_avx512(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

SCNET_ACT_AVX512 inline __m512 tanh_avx512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f), sign = _mm512_set1_ps(-0.0f);
    const __m512 magnitude = _mm512_andnot_ps(sign, x);
    const __m512 polynomial_magnitude = _mm512_min_ps(magnitude, _mm512_set1_ps(0.5f));
    const __m512 square = _mm512_mul_ps(polynomial_magnitude, polynomial_magnitude);
    __m512 p = _mm512_set1_ps(6404582.0f / 10854718875.0f);
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(-929569.0f / 638512875.0f));
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(21844.0f / 6081075.0f));
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(-1382.0f / 155925.0f));
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(62.0f / 2835.0f));
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(-17.0f / 315.0f));
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(2.0f / 15.0f));
    p = _mm512_fmadd_ps(p, square, _mm512_set1_ps(-1.0f / 3.0f));
    const __m512 small = _mm512_mul_ps(x, _mm512_fmadd_ps(p, square, one));
    const __m512 bounded = _mm512_min_ps(magnitude, _mm512_set1_ps(20.0f));
    const __m512 e = exp_avx512(_mm512_mul_ps(_mm512_set1_ps(-2.0f), bounded));
    const __m512 positive = _mm512_div_ps(_mm512_sub_ps(one, e), _mm512_add_ps(one, e));
    const __m512 large = _mm512_xor_ps(positive, _mm512_and_ps(sign, x));
    const __m512 result =
        _mm512_mask_blend_ps(_mm512_cmp_ps_mask(magnitude, _mm512_set1_ps(0.5f), _CMP_LE_OQ), large, small);
    return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q), result, x);
}

SCNET_ACT_AVX512 SCNET_ACT_NO_CONTRACT inline void avx512_update(const float* gates, float* cell, float* hidden,
                                                                 float* output, int width) {
    SCNET_ACT_PRECISE
    int k = 0;
    for (; k + 15 < width; k += 16) {
        const __m512 i = sigmoid_avx512(_mm512_loadu_ps(gates + k));
        const __m512 f = sigmoid_avx512(_mm512_loadu_ps(gates + width + k));
        const __m512 g = tanh_avx512(_mm512_loadu_ps(gates + 2 * width + k));
        const __m512 o = sigmoid_avx512(_mm512_loadu_ps(gates + 3 * width + k));
        const __m512 next_cell = _mm512_add_ps(_mm512_mul_ps(f, _mm512_loadu_ps(cell + k)), _mm512_mul_ps(i, g));
        const __m512 next_hidden = _mm512_mul_ps(o, tanh_avx512(next_cell));
        _mm512_storeu_ps(cell + k, next_cell);
        _mm512_storeu_ps(hidden + k, next_hidden);
        _mm512_storeu_ps(output + k, next_hidden);
    }
    // Preserve the existing AVX2/scalar split for every width, including 8..15.
    // No vector load may cross the end of an individual gate plane.
    if (k + 7 < width) {
        const __m256 i = sigmoid_avx2(_mm256_loadu_ps(gates + k));
        const __m256 f = sigmoid_avx2(_mm256_loadu_ps(gates + width + k));
        const __m256 g = tanh_avx2(_mm256_loadu_ps(gates + 2 * width + k));
        const __m256 o = sigmoid_avx2(_mm256_loadu_ps(gates + 3 * width + k));
        const __m256 next_cell = _mm256_add_ps(_mm256_mul_ps(f, _mm256_loadu_ps(cell + k)), _mm256_mul_ps(i, g));
        const __m256 next_hidden = _mm256_mul_ps(o, tanh_avx2(next_cell));
        _mm256_storeu_ps(cell + k, next_cell);
        _mm256_storeu_ps(hidden + k, next_hidden);
        _mm256_storeu_ps(output + k, next_hidden);
        k += 8;
    }
    scalar_range(gates, cell, hidden, output, width, k);
}

inline bool avx512_supported() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("fma");
}
#endif

#if SCNET_ACT_NEON
inline float32x4_t exp_neon(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j =
        vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
                  vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                            vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
                  u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, j, s2), s1), vfmaq_f32(k, j, k)));
}

inline float32x4_t sigmoid_neon(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, exp_neon(vnegq_f32(x))));
}

inline float32x4_t tanh_neon(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f), magnitude = vabsq_f32(x);
    const float32x4_t polynomial_magnitude = vminq_f32(magnitude, vdupq_n_f32(0.5f));
    const float32x4_t square = vmulq_f32(polynomial_magnitude, polynomial_magnitude);
    float32x4_t p = vdupq_n_f32(6404582.0f / 10854718875.0f);
    p = vfmaq_f32(vdupq_n_f32(-929569.0f / 638512875.0f), p, square);
    p = vfmaq_f32(vdupq_n_f32(21844.0f / 6081075.0f), p, square);
    p = vfmaq_f32(vdupq_n_f32(-1382.0f / 155925.0f), p, square);
    p = vfmaq_f32(vdupq_n_f32(62.0f / 2835.0f), p, square);
    p = vfmaq_f32(vdupq_n_f32(-17.0f / 315.0f), p, square);
    p = vfmaq_f32(vdupq_n_f32(2.0f / 15.0f), p, square);
    p = vfmaq_f32(vdupq_n_f32(-1.0f / 3.0f), p, square);
    const float32x4_t small = vmulq_f32(x, vfmaq_f32(one, p, square));
    const float32x4_t bounded = vminq_f32(magnitude, vdupq_n_f32(20.0f));
    const float32x4_t e = exp_neon(vmulq_n_f32(bounded, -2.0f));
    const float32x4_t positive = vdivq_f32(vsubq_f32(one, e), vaddq_f32(one, e));
    const uint32x4_t sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000u));
    const float32x4_t large = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(positive), sign));
    const float32x4_t result = vbslq_f32(vcleq_f32(magnitude, vdupq_n_f32(0.5f)), small, large);
    return vbslq_f32(vceqq_f32(x, x), result, x);
}

SCNET_ACT_NO_CONTRACT inline void neon_update(const float* gates, float* cell, float* hidden, float* output,
                                              int width) {
    SCNET_ACT_PRECISE
    int k = 0;
    for (; k + 3 < width; k += 4) {
        const float32x4_t i = sigmoid_neon(vld1q_f32(gates + k));
        const float32x4_t f = sigmoid_neon(vld1q_f32(gates + width + k));
        const float32x4_t g = tanh_neon(vld1q_f32(gates + 2 * width + k));
        const float32x4_t o = sigmoid_neon(vld1q_f32(gates + 3 * width + k));
        const float32x4_t next_cell = vaddq_f32(vmulq_f32(f, vld1q_f32(cell + k)), vmulq_f32(i, g));
        const float32x4_t next_hidden = vmulq_f32(o, tanh_neon(next_cell));
        vst1q_f32(cell + k, next_cell);
        vst1q_f32(hidden + k, next_hidden);
        vst1q_f32(output + k, next_hidden);
    }
    scalar_range(gates, cell, hidden, output, width, k);
}
#endif

inline Kernel fastest_kernel() {
    static const Kernel selected = [] {
#if SCNET_ACT_X86
        if (avx512_supported())
            return Kernel{avx512_update, "avx512_fma"};
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            return Kernel{avx2_update, "avx2_fma"};
#endif
#if SCNET_ACT_NEON
        return Kernel{neon_update, "neon"};
#endif
        return Kernel{scalar_update, "scalar"};
    }();
    return selected;
}

// Diagnostic override is read once per direction, outside all recurrent loops.
// It is independent of SCNET_SCALAR_LSTM, which controls matrix multiplication.
inline Kernel selected_kernel() {
    const char* value = std::getenv("CRISPASR_SCNET_SCALAR_GATES");
    if (value && *value && std::strcmp(value, "0") != 0)
        return {scalar_update, "scalar"};
    return fastest_kernel();
}
} // namespace scnet_activation
