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

namespace scnet_fft {

class Plan {
public:
    struct Scratch {
        std::vector<float> ar, ai, br, bi;
    };

    explicit Plan(int size) : size_(size) {
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

    // Unnormalized forward transform in place; caller owns reusable scratch.
    // The plan is immutable and can be shared by concurrent workers.
    void transform(float* re, float* im, Scratch& scratch) const {
        if (!swaps_.empty() && convolution_ == 0) {
            radix2(re, im);
            return;
        }
        if (convolution_) {
            scratch.ar.assign(convolution_, 0);
            scratch.ai.assign(convolution_, 0);
            for (int k = 0; k < size_; ++k) {
                const float cr = chirp_r_[k], ci = chirp_i_[k];
                scratch.ar[k] = re[k] * cr - im[k] * ci;
                scratch.ai[k] = re[k] * ci + im[k] * cr;
            }
            radix2(scratch.ar.data(), scratch.ai.data());
            for (int k = 0; k < convolution_; ++k) {
                const float a = scratch.ar[k], b = scratch.ai[k];
                scratch.ar[k] = a * kernel_r_[k] - b * kernel_i_[k];
                scratch.ai[k] = -(a * kernel_i_[k] + b * kernel_r_[k]);
            }
            radix2(scratch.ar.data(), scratch.ai.data());
            const float scale = 1.0f / convolution_;
            for (int k = 0; k < size_; ++k) {
                const float a = scratch.ar[k] * scale, b = -scratch.ai[k] * scale;
                re[k] = a * chirp_r_[k] - b * chirp_i_[k];
                im[k] = a * chirp_i_[k] + b * chirp_r_[k];
            }
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
        int offset = 0;
        for (int len = 2; len <= radix_size_; len <<= 1) {
            const int half = len / 2;
            for (int begin = 0; begin < radix_size_; begin += len)
                for (int k = 0; k < half; ++k) {
                    const float cr = twiddle_r_[offset + k], ci = twiddle_i_[offset + k];
                    const int a = begin + k, b = a + half;
                    const float tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            offset += half;
        }
    }

    void mixed(const float* re, const float* im, int stride, float* out_r, float* out_i, float* tmp_r,
               float* tmp_i, int n) const {
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
            mixed(re + j * stride, im + j * stride, stride * radix, out_r + j * m, out_i + j * m,
                  tmp_r + j * m, tmp_i + j * m, m);
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
    std::vector<float> root_r_, root_i_, twiddle_r_, twiddle_i_, chirp_r_, chirp_i_, kernel_r_, kernel_i_;
    std::vector<std::pair<int, int>> swaps_;
};
} // namespace scnet_fft
