// src/core/fft.h — radix-2 Cooley-Tukey FFT shared by the granite family.
//
// Both `granite_speech.cpp` and `granite_nle.cpp` historically owned a
// byte-identical copy of the same recursive radix-2 FFT plus a
// const-input wrapper that satisfies `core_mel::FftR2C`. This header
// hosts the single shared implementation. Other backends (kokoro,
// mimo_tokenizer) ship their own near-identical copies; those can move
// here too in a follow-up — keeping this lift granite-only avoids
// touching their numerical paths in this commit.
//
// Header-only / `static inline` so each caller inlines the recursion
// and keeps the existing numerical behaviour bit-identical.

#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace core_fft {

// Recursive radix-2 Cooley-Tukey FFT.
//
//   in  : N real samples (mutated in place by the recursion's even/odd
//         splitting; callers that need const-input semantics should use
//         `fft_radix2_wrapper` below)
//   out : 2*N floats — interleaved (re, im) pairs of the complex DFT
//
// For odd N (only hit at the recursion root if a caller passes a non-
// power-of-two), falls back to an O(N^2) direct DFT. All real callers
// in this codebase pass N as a power of two (n_fft = 512 in granite,
// matching Whisper / HF feature extraction).
static inline void fft_radix2(float* in, int N, float* out) {
    if (N <= 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    if (N % 2 != 0) {
        for (int k = 0; k < N; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                double a = -2.0 * M_PI * k * n / N;
                re += in[n] * cos(a);
                im += in[n] * sin(a);
            }
            out[2 * k] = (float)re;
            out[2 * k + 1] = (float)im;
        }
        return;
    }
    int half = N / 2;
    std::vector<float> even(half), odd(half);
    for (int i = 0; i < half; i++) {
        even[i] = in[2 * i];
        odd[i] = in[2 * i + 1];
    }
    std::vector<float> E(2 * half), O(2 * half);
    fft_radix2(even.data(), half, E.data());
    fft_radix2(odd.data(), half, O.data());
    for (int k = 0; k < half; k++) {
        double a = -2.0 * M_PI * k / N;
        float wr = (float)cos(a), wi = (float)sin(a);
        float tre = wr * O[2 * k] - wi * O[2 * k + 1];
        float tim = wr * O[2 * k + 1] + wi * O[2 * k];
        out[2 * k] = E[2 * k] + tre;
        out[2 * k + 1] = E[2 * k + 1] + tim;
        out[2 * (k + half)] = E[2 * k] - tre;
        out[2 * (k + half) + 1] = E[2 * k + 1] - tim;
    }
}

// Const-input wrapper matching `core_mel::FftR2C`. Uses thread-local
// scratch buffers (~4N + 8N floats) so the recursion can mutate without
// disturbing the caller's input. `fft_radix2_wrapper` is what
// `core_mel::compute(...)` consumes via function pointer.
static inline void fft_radix2_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    fft_radix2(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

} // namespace core_fft
