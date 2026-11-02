// Python-compatible integer-seeded MT19937 for SCNet's per-chunk random shifts.
// Python random.seed(int) initializes by an unsigned little-endian key array;
// std::mt19937(seed) uses a different initialization and cannot reproduce it.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace scnet_detail {
class PythonRandom {
public:
    explicit PythonRandom(uint64_t seed) {
        state_[0] = 19650218U;
        for (uint32_t i = 1; i < 624; ++i)
            state_[i] = 1812433253U * (state_[i - 1] ^ (state_[i - 1] >> 30)) + i;
        const uint32_t key[] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
        const uint32_t key_size = key[1] ? 2 : 1;
        uint32_t i = 1, j = 0;
        for (uint32_t count = 0; count < 624; ++count) {
            state_[i] = (state_[i] ^ ((state_[i - 1] ^ (state_[i - 1] >> 30)) * 1664525U)) + key[j] + j;
            j = (j + 1) % key_size;
            if (++i == 624) {
                state_[0] = state_[623];
                i = 1;
            }
        }
        for (uint32_t count = 0; count < 623; ++count) {
            state_[i] = (state_[i] ^ ((state_[i - 1] ^ (state_[i - 1] >> 30)) * 1566083941U)) - i;
            if (++i == 624) {
                state_[0] = state_[623];
                i = 1;
            }
        }
        state_[0] = 0x80000000U;
    }

    int randint(int upper) {
        if (upper < 0 || upper >= 0x7fffffff)
            throw std::invalid_argument("invalid SCNet shift range");
        const uint32_t n = static_cast<uint32_t>(upper) + 1;
        uint32_t v = n;
        int bits = 0;
        while (v) {
            ++bits;
            v >>= 1;
        }
        do {
            v = next() >> (32 - bits);
        } while (v >= n);
        return static_cast<int>(v);
    }

private:
    uint32_t next() {
        if (position_ == 624) {
            for (int i = 0; i < 624; ++i) {
                const uint32_t y = (state_[i] & 0x80000000U) | (state_[(i + 1) % 624] & 0x7fffffffU);
                state_[i] = state_[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
            }
            position_ = 0;
        }
        uint32_t y = state_[position_++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        return y;
    }
    std::array<uint32_t, 624> state_{};
    int position_ = 624;
};
} // namespace scnet_detail
