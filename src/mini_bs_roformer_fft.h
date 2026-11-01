// Cached radix-2 transforms for the Mini-BS-RoFormer frontend and inverse STFT.
// Independent double-precision trig avoids accumulated F32 twiddle drift in
// quiet bands, which the model's per-band normalization otherwise amplifies.
#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mini_bs_fft {

class Plan {
public:
    explicit Plan(int size) : size_(size) {
        if (size != 2048 && size != 4096)
            throw std::invalid_argument("Mini BS FFT size must be 2048 or 4096");
        for (int i = 1, j = 0; i < size; ++i) {
            int bit = size >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                swaps_.emplace_back(i, j);
        }
        real_.reserve(size - 1);
        imag_.reserve(size - 1);
        const double pi = std::acos(-1.0);
        for (int len = 2; len <= size; len <<= 1)
            for (int j = 0; j < len / 2; ++j) {
                const double angle = -2.0 * pi * j / len;
                real_.push_back((float)std::cos(angle));
                imag_.push_back((float)std::sin(angle));
            }
    }

    // Scratch belongs to the caller; the plan is immutable and may be shared
    // by concurrent frame/channel workers. Both arrays contain size_ floats.
    void transform(float* re, float* im) const {
        for (const auto& swap : swaps_) {
            std::swap(re[swap.first], re[swap.second]);
            std::swap(im[swap.first], im[swap.second]);
        }
        int offset = 0;
        for (int len = 2; len <= size_; len <<= 1) {
            const int half = len / 2;
            for (int i = 0; i < size_; i += len)
                for (int j = 0; j < half; ++j) {
                    const float cr = real_[offset + j], ci = imag_[offset + j];
                    const float tr = re[i + j + half] * cr - im[i + j + half] * ci;
                    const float ti = re[i + j + half] * ci + im[i + j + half] * cr;
                    re[i + j + half] = re[i + j] - tr;
                    im[i + j + half] = im[i + j] - ti;
                    re[i + j] += tr;
                    im[i + j] += ti;
                }
            offset += half;
        }
    }

private:
    int size_;
    std::vector<std::pair<int, int>> swaps_;
    std::vector<float> real_, imag_;
};

} // namespace mini_bs_fft
