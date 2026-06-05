// src/core/torch_rng.h — torch.randn-compatible RNG primitives for TTS/diffusion
// backends. Header-only so callers can drop in by including this file alone.
//
// Why this exists:
//
//   Several backends (vibevoice, chatterbox, chatterbox_s3gen, voxcpm2-tts) need
//   to reproduce a `torch.randn(...)` call bit-exactly — most often to seed
//   diffusion / CFM Euler loops where the same seed must yield the same noise
//   on Python (reference) and C++ (inference). Each backend had its own
//   byte-identical copy of MT19937 + Box-Muller. Lifting them here keeps the
//   contract single-sourced.
//
// Two PyTorch noise paths:
//
//   * `fill_gaussian_noise()`  — `torch.randn(dtype=float32)` for n>=16.
//     Uniform: `(mt19937() & 0xFFFFFF) / 2^24`.  Box-Muller in F32.
//
//   * `fill_gaussian_noise_bf16()` — `torch.randn(dtype=bfloat16)` for n>=16.
//     Uniform: `(mt19937() & 0xFF) / 2^8` (8-bit mantissa per
//     `std::numeric_limits<scalar_t>::digits` in PyTorch's `uniform_real`).
//     Box-Muller in BF16 (every intermediate cast to BF16). Critical for
//     CFM solvers whose Python reference dtype is bfloat16 — the F32 path
//     produces a completely different sequence for the same seed (cos≈0.03).
//
//   Both share the contiguous-tensor "tail recompute" quirk of PyTorch's
//   `normal_fill`: if `n % 16 != 0`, the final 16 outputs are recomputed
//   from a fresh batch of 16 uniforms (overwriting the partial tail).
//
// MT19937 state matches PyTorch's CPUGeneratorImpl seeding (the variant where
// `mt[0] = seed`, then the Knuth recurrence). `torch.manual_seed(s)` and
// `torch.Generator().manual_seed(s)` both produce this state on CPU.
//
// `bf16_round()` is exposed because BF16 emulation is useful beyond noise
// (sinusoidal time embeddings, t_span schedules, etc.) — anywhere C++ runs in
// F32 but the Python reference runs in BF16.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "ggml.h"

// MSVC's <cmath> does not define M_PI unless _USE_MATH_DEFINES is set
// before <math.h> is included — and that gate is fragile because <math.h>
// may have been pulled in transitively before us. Define the macro here if
// the compiler didn't, using the same value PyTorch's c10::pi<double> uses.
// We keep the exact `2.0f * (float)M_PI * u2` computation in callers below
// because this header's whole point is bit-equality with torch.randn.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace crispasr::core {

struct mt19937_state {
    uint32_t mt[624];
    int mti = 624;
};

inline void mt19937_seed(mt19937_state& s, uint32_t seed) {
    s.mt[0] = seed;
    for (int i = 1; i < 624; i++) {
        s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + (uint32_t)i;
    }
    s.mti = 624;
}

inline uint32_t mt19937_next(mt19937_state& s) {
    if (s.mti >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (s.mt[i] & 0x80000000u) | (s.mt[(i + 1) % 624] & 0x7FFFFFFFu);
            s.mt[i] = s.mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1) {
                s.mt[i] ^= 0x9908B0DFu;
            }
        }
        s.mti = 0;
    }
    uint32_t y = s.mt[s.mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= (y >> 18);
    return y;
}

// ── F32 path: torch.randn(dtype=float32) ───────────────────────────────────

inline float mt_uniform_torch_float(mt19937_state& rng) {
    return (float)(mt19937_next(rng) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

inline void torch_normal_fill_16(float* data) {
    for (int j = 0; j < 8; j++) {
        const float u1 = 1.0f - data[j];
        const float u2 = data[j + 8];
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = 2.0f * (float)M_PI * u2;
        data[j] = radius * std::cos(theta);
        data[j + 8] = radius * std::sin(theta);
    }
}

inline void fill_gaussian_noise(float* data, int n, mt19937_state& rng) {
    if (n <= 0) {
        return;
    }
    if (n < 16) {
        float tmp[16];
        for (int i = 0; i < 16; i++) {
            tmp[i] = mt_uniform_torch_float(rng);
        }
        torch_normal_fill_16(tmp);
        std::memcpy(data, tmp, (size_t)n * sizeof(float));
        return;
    }
    for (int i = 0; i < n; i++) {
        data[i] = mt_uniform_torch_float(rng);
    }
    int i = 0;
    for (; i <= n - 16; i += 16) {
        torch_normal_fill_16(data + i);
    }
    if (i < n) {
        // PyTorch recomputes the final overlapping block when numel isn't a
        // multiple of 16. Match that behaviour.
        float* tail = data + n - 16;
        for (int j = 0; j < 16; j++) {
            tail[j] = mt_uniform_torch_float(rng);
        }
        torch_normal_fill_16(tail);
    }
}

inline void fill_gaussian_noise(float* data, int n, uint32_t seed) {
    mt19937_state rng;
    mt19937_seed(rng, seed);
    fill_gaussian_noise(data, n, rng);
}

// ── BF16 emulation ─────────────────────────────────────────────────────────

// Round a float to BF16 precision and return as F32 (i.e. a representable
// BF16 value held in F32 storage). Useful for any C++ side that runs in F32
// but needs to match a BF16 PyTorch reference.
inline float bf16_round(float x) {
    ggml_bf16_t bf = ggml_fp32_to_bf16(x);
    return ggml_bf16_to_fp32(bf);
}

// ── BF16 path: torch.randn(dtype=bfloat16) ─────────────────────────────────

inline float mt_uniform_bf16(mt19937_state& rng) {
    return (float)(mt19937_next(rng) & 0xFFu) / 256.0f;
}

inline void bf16_normal_fill_16(float* data) {
    for (int j = 0; j < 8; j++) {
        float u1 = bf16_round(1.0f - data[j]);
        float u2 = data[j + 8];
        float radius = bf16_round(std::sqrt(bf16_round(-2.0f * bf16_round(std::log(u1)))));
        float theta = bf16_round(bf16_round(2.0f * (float)M_PI * u2));
        data[j] = bf16_round(radius * bf16_round(std::cos(theta)));
        data[j + 8] = bf16_round(radius * bf16_round(std::sin(theta)));
    }
}

inline void fill_gaussian_noise_bf16(float* data, int n, mt19937_state& rng) {
    if (n <= 0) {
        return;
    }
    if (n < 16) {
        float tmp[16];
        for (int i = 0; i < 16; i++) {
            tmp[i] = mt_uniform_bf16(rng);
        }
        bf16_normal_fill_16(tmp);
        std::memcpy(data, tmp, (size_t)n * sizeof(float));
        return;
    }
    for (int i = 0; i < n; i++) {
        data[i] = mt_uniform_bf16(rng);
    }
    int i = 0;
    for (; i <= n - 16; i += 16) {
        bf16_normal_fill_16(data + i);
    }
    if (i < n) {
        float* tail = data + n - 16;
        for (int j = 0; j < 16; j++) {
            tail[j] = mt_uniform_bf16(rng);
        }
        bf16_normal_fill_16(tail);
    }
}

inline void fill_gaussian_noise_bf16(float* data, int n, uint32_t seed) {
    mt19937_state rng;
    mt19937_seed(rng, seed);
    fill_gaussian_noise_bf16(data, n, rng);
}

} // namespace crispasr::core
