// Exact-length FFTs for SCNet's normalized STFT and recurrent feature conversion.
// Smooth sizes use mixed-radix Cooley-Tukey; other sizes use Bluestein with a
// radix-2 convolution. No zero-padding substitution changes the transform size.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define SCNET_FFT_X86 1
#define SCNET_FFT_AVX2 __attribute__((target("avx2")))
#else
#define SCNET_FFT_X86 0
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define SCNET_FFT_NEON 1
#else
#define SCNET_FFT_NEON 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define SCNET_FFT_NO_CONTRACT __attribute__((optimize("fp-contract=off")))
#else
#define SCNET_FFT_NO_CONTRACT
#endif
#if defined(__clang__)
#define SCNET_FFT_PRECISE _Pragma("clang fp contract(off)")
#else
#define SCNET_FFT_PRECISE
#endif

namespace scnet_fft {

namespace detail {

// Each SIMD lane is an independent complex value or butterfly. Keep separate
// multiplies and additions: FMA or reassociation would change the scalar FFT's
// rounding tree, even though it computes the same mathematical transform.
SCNET_FFT_NO_CONTRACT inline void butterfly(float* re, float* im, int a, int b, float cr, float ci) {
    SCNET_FFT_PRECISE
    const float tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
    re[b] = re[a] - tr;
    im[b] = im[a] - ti;
    re[a] += tr;
    im[a] += ti;
}

SCNET_FFT_NO_CONTRACT inline void radix2_scalar(float* re, float* im, int size, const float* wr, const float* wi) {
    SCNET_FFT_PRECISE
    int offset = 0;
    for (int len = 2; len <= size; len <<= 1) {
        const int half = len / 2;
        for (int begin = 0; begin < size; begin += len)
            for (int k = 0; k < half; ++k)
                butterfly(re, im, begin + k, begin + k + half, wr[offset + k], wi[offset + k]);
        offset += half;
    }
}

SCNET_FFT_NO_CONTRACT inline void input_scalar(const float* re, const float* im, const float* cr, const float* ci,
                                               float* out_r, float* out_i, int size) {
    SCNET_FFT_PRECISE
    for (int k = 0; k < size; ++k) {
        out_r[k] = re[k] * cr[k] - im[k] * ci[k];
        out_i[k] = re[k] * ci[k] + im[k] * cr[k];
    }
}

SCNET_FFT_NO_CONTRACT inline void product_scalar(float* re, float* im, const float* kr, const float* ki, int size) {
    SCNET_FFT_PRECISE
    for (int k = 0; k < size; ++k) {
        const float a = re[k], b = im[k];
        re[k] = a * kr[k] - b * ki[k];
        im[k] = -(a * ki[k] + b * kr[k]);
    }
}

SCNET_FFT_NO_CONTRACT inline void output_scalar(const float* in_r, const float* in_i, const float* cr, const float* ci,
                                                float* re, float* im, int size, float scale) {
    SCNET_FFT_PRECISE
    for (int k = 0; k < size; ++k) {
        const float a = in_r[k] * scale, b = -in_i[k] * scale;
        re[k] = a * cr[k] - b * ci[k];
        im[k] = a * ci[k] + b * cr[k];
    }
}

#if SCNET_FFT_X86
SCNET_FFT_AVX2 SCNET_FFT_NO_CONTRACT inline void radix2_avx2(float* re, float* im, int size, const float* wr,
                                                             const float* wi) {
    SCNET_FFT_PRECISE
    int offset = 0;
    for (int len = 2; len <= size; len <<= 1) {
        const int half = len / 2;
        for (int begin = 0; begin < size; begin += len) {
            int k = 0;
            for (; k + 7 < half; k += 8) {
                const int a = begin + k, b = a + half;
                const auto cr = _mm256_loadu_ps(wr + offset + k), ci = _mm256_loadu_ps(wi + offset + k);
                const auto ar = _mm256_loadu_ps(re + a), ai = _mm256_loadu_ps(im + a);
                const auto br = _mm256_loadu_ps(re + b), bi = _mm256_loadu_ps(im + b);
                const auto tr = _mm256_sub_ps(_mm256_mul_ps(br, cr), _mm256_mul_ps(bi, ci));
                const auto ti = _mm256_add_ps(_mm256_mul_ps(br, ci), _mm256_mul_ps(bi, cr));
                _mm256_storeu_ps(re + b, _mm256_sub_ps(ar, tr));
                _mm256_storeu_ps(im + b, _mm256_sub_ps(ai, ti));
                _mm256_storeu_ps(re + a, _mm256_add_ps(ar, tr));
                _mm256_storeu_ps(im + a, _mm256_add_ps(ai, ti));
            }
            for (; k < half; ++k)
                butterfly(re, im, begin + k, begin + k + half, wr[offset + k], wi[offset + k]);
        }
        offset += half;
    }
}

SCNET_FFT_AVX2 SCNET_FFT_NO_CONTRACT inline void input_avx2(const float* re, const float* im, const float* cr,
                                                            const float* ci, float* out_r, float* out_i, int size) {
    SCNET_FFT_PRECISE
    int k = 0;
    for (; k + 7 < size; k += 8) {
        const auto a = _mm256_loadu_ps(re + k), b = _mm256_loadu_ps(im + k);
        const auto c = _mm256_loadu_ps(cr + k), d = _mm256_loadu_ps(ci + k);
        _mm256_storeu_ps(out_r + k, _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, d)));
        _mm256_storeu_ps(out_i + k, _mm256_add_ps(_mm256_mul_ps(a, d), _mm256_mul_ps(b, c)));
    }
    input_scalar(re + k, im + k, cr + k, ci + k, out_r + k, out_i + k, size - k);
}

SCNET_FFT_AVX2 SCNET_FFT_NO_CONTRACT inline void product_avx2(float* re, float* im, const float* kr, const float* ki,
                                                              int size) {
    SCNET_FFT_PRECISE
    const auto sign = _mm256_set1_ps(-0.0f);
    int k = 0;
    for (; k + 7 < size; k += 8) {
        const auto a = _mm256_loadu_ps(re + k), b = _mm256_loadu_ps(im + k);
        const auto c = _mm256_loadu_ps(kr + k), d = _mm256_loadu_ps(ki + k);
        _mm256_storeu_ps(re + k, _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, d)));
        _mm256_storeu_ps(im + k, _mm256_xor_ps(_mm256_add_ps(_mm256_mul_ps(a, d), _mm256_mul_ps(b, c)), sign));
    }
    product_scalar(re + k, im + k, kr + k, ki + k, size - k);
}

SCNET_FFT_AVX2 SCNET_FFT_NO_CONTRACT inline void output_avx2(const float* in_r, const float* in_i, const float* cr,
                                                             const float* ci, float* re, float* im, int size,
                                                             float scale) {
    SCNET_FFT_PRECISE
    const auto factor = _mm256_set1_ps(scale), sign = _mm256_set1_ps(-0.0f);
    int k = 0;
    for (; k + 7 < size; k += 8) {
        const auto a = _mm256_mul_ps(_mm256_loadu_ps(in_r + k), factor);
        const auto b = _mm256_mul_ps(_mm256_xor_ps(_mm256_loadu_ps(in_i + k), sign), factor);
        const auto c = _mm256_loadu_ps(cr + k), d = _mm256_loadu_ps(ci + k);
        _mm256_storeu_ps(re + k, _mm256_sub_ps(_mm256_mul_ps(a, c), _mm256_mul_ps(b, d)));
        _mm256_storeu_ps(im + k, _mm256_add_ps(_mm256_mul_ps(a, d), _mm256_mul_ps(b, c)));
    }
    output_scalar(in_r + k, in_i + k, cr + k, ci + k, re + k, im + k, size - k, scale);
}
#endif

#if SCNET_FFT_NEON
SCNET_FFT_NO_CONTRACT inline void radix2_neon(float* re, float* im, int size, const float* wr, const float* wi) {
    SCNET_FFT_PRECISE
    int offset = 0;
    for (int len = 2; len <= size; len <<= 1) {
        const int half = len / 2;
        for (int begin = 0; begin < size; begin += len) {
            int k = 0;
            for (; k + 3 < half; k += 4) {
                const int a = begin + k, b = a + half;
                const auto cr = vld1q_f32(wr + offset + k), ci = vld1q_f32(wi + offset + k);
                const auto ar = vld1q_f32(re + a), ai = vld1q_f32(im + a);
                const auto br = vld1q_f32(re + b), bi = vld1q_f32(im + b);
                const auto tr = vsubq_f32(vmulq_f32(br, cr), vmulq_f32(bi, ci));
                const auto ti = vaddq_f32(vmulq_f32(br, ci), vmulq_f32(bi, cr));
                vst1q_f32(re + b, vsubq_f32(ar, tr));
                vst1q_f32(im + b, vsubq_f32(ai, ti));
                vst1q_f32(re + a, vaddq_f32(ar, tr));
                vst1q_f32(im + a, vaddq_f32(ai, ti));
            }
            for (; k < half; ++k)
                butterfly(re, im, begin + k, begin + k + half, wr[offset + k], wi[offset + k]);
        }
        offset += half;
    }
}

SCNET_FFT_NO_CONTRACT inline void input_neon(const float* re, const float* im, const float* cr, const float* ci,
                                             float* out_r, float* out_i, int size) {
    SCNET_FFT_PRECISE
    int k = 0;
    for (; k + 3 < size; k += 4) {
        const auto a = vld1q_f32(re + k), b = vld1q_f32(im + k);
        const auto c = vld1q_f32(cr + k), d = vld1q_f32(ci + k);
        vst1q_f32(out_r + k, vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, d)));
        vst1q_f32(out_i + k, vaddq_f32(vmulq_f32(a, d), vmulq_f32(b, c)));
    }
    input_scalar(re + k, im + k, cr + k, ci + k, out_r + k, out_i + k, size - k);
}

SCNET_FFT_NO_CONTRACT inline void product_neon(float* re, float* im, const float* kr, const float* ki, int size) {
    SCNET_FFT_PRECISE
    int k = 0;
    for (; k + 3 < size; k += 4) {
        const auto a = vld1q_f32(re + k), b = vld1q_f32(im + k);
        const auto c = vld1q_f32(kr + k), d = vld1q_f32(ki + k);
        vst1q_f32(re + k, vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, d)));
        vst1q_f32(im + k, vnegq_f32(vaddq_f32(vmulq_f32(a, d), vmulq_f32(b, c))));
    }
    product_scalar(re + k, im + k, kr + k, ki + k, size - k);
}

SCNET_FFT_NO_CONTRACT inline void output_neon(const float* in_r, const float* in_i, const float* cr, const float* ci,
                                              float* re, float* im, int size, float scale) {
    SCNET_FFT_PRECISE
    const auto factor = vdupq_n_f32(scale);
    int k = 0;
    for (; k + 3 < size; k += 4) {
        const auto a = vmulq_f32(vld1q_f32(in_r + k), factor);
        const auto b = vmulq_f32(vnegq_f32(vld1q_f32(in_i + k)), factor);
        const auto c = vld1q_f32(cr + k), d = vld1q_f32(ci + k);
        vst1q_f32(re + k, vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, d)));
        vst1q_f32(im + k, vaddq_f32(vmulq_f32(a, d), vmulq_f32(b, c)));
    }
    output_scalar(in_r + k, in_i + k, cr + k, ci + k, re + k, im + k, size - k, scale);
}
#endif

struct Kernels {
    void (*radix2)(float*, float*, int, const float*, const float*);
    void (*input)(const float*, const float*, const float*, const float*, float*, float*, int);
    void (*product)(float*, float*, const float*, const float*, int);
    void (*output)(const float*, const float*, const float*, const float*, float*, float*, int, float);
    const char* name;
};

inline const Kernels& scalar_kernels() {
    static const Kernels kernels{radix2_scalar, input_scalar, product_scalar, output_scalar, "scalar"};
    return kernels;
}

inline const Kernels& fastest_kernels() {
#if SCNET_FFT_X86
    if (__builtin_cpu_supports("avx2")) {
        static const Kernels kernels{radix2_avx2, input_avx2, product_avx2, output_avx2, "avx2"};
        return kernels;
    }
#endif
#if SCNET_FFT_NEON
    static const Kernels kernels{radix2_neon, input_neon, product_neon, output_neon, "neon"};
    return kernels;
#else
    return scalar_kernels();
#endif
}

} // namespace detail

enum class Implementation { Automatic, Scalar };

class Plan {
public:
    struct Scratch {
        std::vector<float> ar, ai, br, bi;
    };

    explicit Plan(int size, Implementation implementation = Implementation::Automatic)
        : size_(size),
          kernels_(implementation == Implementation::Scalar ? detail::scalar_kernels() : detail::fastest_kernels()) {
        if (size <= 0 || size > (1 << 20))
            throw std::invalid_argument("invalid SCNet FFT length");
        const double pi = std::acos(-1.0);
        root_r_.resize(size);
        root_i_.resize(size);
        for (int k = 0; k < size; ++k) {
            root_r_[k] = static_cast<float>(std::cos(-2 * pi * k / size));
            root_i_[k] = static_cast<float>(std::sin(-2 * pi * k / size));
        }
        int remainder = size;
        for (int prime : {2, 3, 5, 7, 11, 13})
            while (remainder % prime == 0)
                remainder /= prime;
        if (remainder == 1) {
            if ((size & (size - 1)) == 0)
                prepare_radix2(size);
            return;
        }
        convolution_ = 1;
        while (convolution_ < 2 * size - 1)
            convolution_ <<= 1;
        prepare_radix2(convolution_);
        chirp_r_.resize(size);
        chirp_i_.resize(size);
        kernel_r_.assign(convolution_, 0);
        kernel_i_.assign(convolution_, 0);
        for (int k = 0; k < size; ++k) {
            const double angle = pi * static_cast<double>(int64_t(k) * k % (2 * size)) / size;
            chirp_r_[k] = static_cast<float>(std::cos(angle));
            chirp_i_[k] = static_cast<float>(-std::sin(angle));
            kernel_r_[k] = chirp_r_[k];
            kernel_i_[k] = -chirp_i_[k];
            if (k) {
                kernel_r_[convolution_ - k] = chirp_r_[k];
                kernel_i_[convolution_ - k] = -chirp_i_[k];
            }
        }
        radix2(kernel_r_.data(), kernel_i_.data());
    }

    int size() const { return size_; }
    const char* kernel_name() const { return kernels_.name; }

    // Unnormalized forward transform in place; caller owns reusable scratch.
    // The plan is immutable and can be shared by concurrent workers.
    void transform(float* re, float* im, Scratch& scratch) const {
        if (!swaps_.empty() && convolution_ == 0) {
            radix2(re, im);
            return;
        }
        if (convolution_) {
            scratch.ar.resize(convolution_);
            scratch.ai.resize(convolution_);
            // The chirp writes every prefix element. Only the convolution tail
            // needs clearing; it was overwritten by the previous transform.
            std::fill(scratch.ar.begin() + size_, scratch.ar.end(), 0.0f);
            std::fill(scratch.ai.begin() + size_, scratch.ai.end(), 0.0f);
            kernels_.input(re, im, chirp_r_.data(), chirp_i_.data(), scratch.ar.data(), scratch.ai.data(), size_);
            radix2(scratch.ar.data(), scratch.ai.data());
            kernels_.product(scratch.ar.data(), scratch.ai.data(), kernel_r_.data(), kernel_i_.data(), convolution_);
            radix2(scratch.ar.data(), scratch.ai.data());
            const float scale = 1.0f / convolution_;
            kernels_.output(scratch.ar.data(), scratch.ai.data(), chirp_r_.data(), chirp_i_.data(), re, im, size_,
                            scale);
            return;
        }
        scratch.ar.resize(size_);
        scratch.ai.resize(size_);
        scratch.br.resize(size_);
        scratch.bi.resize(size_);
        mixed(re, im, 1, scratch.ar.data(), scratch.ai.data(), scratch.br.data(), scratch.bi.data(), size_);
        std::copy(scratch.ar.begin(), scratch.ar.end(), re);
        std::copy(scratch.ai.begin(), scratch.ai.end(), im);
    }

private:
    void prepare_radix2(int n) {
        radix_size_ = n;
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                swaps_.emplace_back(i, j);
        }
        const double pi = std::acos(-1.0);
        for (int len = 2; len <= n; len <<= 1)
            for (int k = 0; k < len / 2; ++k) {
                twiddle_r_.push_back(static_cast<float>(std::cos(-2 * pi * k / len)));
                twiddle_i_.push_back(static_cast<float>(std::sin(-2 * pi * k / len)));
            }
    }

    void radix2(float* re, float* im) const {
        for (const auto& pair : swaps_) {
            std::swap(re[pair.first], re[pair.second]);
            std::swap(im[pair.first], im[pair.second]);
        }
        kernels_.radix2(re, im, radix_size_, twiddle_r_.data(), twiddle_i_.data());
    }

    void mixed(const float* re, const float* im, int stride, float* out_r, float* out_i, float* tmp_r, float* tmp_i,
               int n) const {
        if (n == 1) {
            out_r[0] = re[0];
            out_i[0] = im[0];
            return;
        }
        int radix = 2;
        while (n % radix)
            ++radix;
        const int m = n / radix;
        for (int j = 0; j < radix; ++j)
            mixed(re + j * stride, im + j * stride, stride * radix, out_r + j * m, out_i + j * m, tmp_r + j * m,
                  tmp_i + j * m, m);
        const int factor = size_ / n;
        for (int k = 0; k < m; ++k)
            for (int r = 0; r < radix; ++r) {
                float ar = 0, ai = 0;
                for (int j = 0; j < radix; ++j) {
                    const int twiddle = static_cast<int>(int64_t(j) * (k + r * m) * factor % size_);
                    const float xr = out_r[j * m + k], xi = out_i[j * m + k];
                    ar += xr * root_r_[twiddle] - xi * root_i_[twiddle];
                    ai += xr * root_i_[twiddle] + xi * root_r_[twiddle];
                }
                tmp_r[k + r * m] = ar;
                tmp_i[k + r * m] = ai;
            }
        std::copy(tmp_r, tmp_r + n, out_r);
        std::copy(tmp_i, tmp_i + n, out_i);
    }

    int size_, convolution_ = 0, radix_size_ = 0;
    const detail::Kernels& kernels_;
    std::vector<float> root_r_, root_i_, twiddle_r_, twiddle_i_, chirp_r_, chirp_i_, kernel_r_, kernel_i_;
    std::vector<std::pair<int, int>> swaps_;
};
} // namespace scnet_fft
