// src/core/lfr.h — Low Frame Rate frame stacking.
//
// Stacks `m` consecutive frames into one super-frame and steps by `n`
// frames between super-frames. Matches FunASR's WavFrontend `apply_lfr`
// (funasr/frontends/wav_frontend.py).
//
// Input  : (T, D) row-major fbank features
// Output : (T_lfr, m*D) row-major stacked features
//          where T_lfr = ceil(T / n).
//
// Padding rule (matches FunASR exactly):
//   - prepend (m-1)/2 copies of frame[0] to give a half-window of left
//     context for the first super-frame
//   - append however many copies of frame[T-1] are needed so the last
//     super-frame's strided gather is in bounds
//     (i.e. T_padded ≥ (T_lfr - 1) * n + m)
//
// For SenseVoiceEncoderSmall: m=7, n=6, D=80 mel bins → (T_lfr, 560).

#pragma once

#include <cstddef>
#include <vector>

namespace core_lfr {

// Compute the LFR output time dimension. Pure helper — no allocation.
static inline int lfr_t_out(int T, int n) {
    if (T <= 0 || n <= 0)
        return 0;
    return (T + n - 1) / n;
}

// Stack `m` consecutive frames stepping by `n`, with FunASR-style padding.
//
//   feats : input frames (T, D) row-major float32
//   T     : input frame count
//   D     : input feature dim per frame
//   m     : stack size (rows per super-frame)
//   n     : stack stride
//
// Returns a row-major (T_lfr, m*D) float32 buffer. `T_lfr_out` is set
// on return.
inline std::vector<float> stack(const float* feats, int T, int D, int m, int n, int& T_lfr_out) {
    T_lfr_out = 0;
    if (!feats || T <= 0 || D <= 0 || m <= 0 || n <= 0)
        return {};

    const int T_lfr = lfr_t_out(T, n);
    const int left = (m - 1) / 2;
    // T_padded must satisfy: (T_lfr - 1) * n + m ≤ T_padded.
    const int needed_rows = (T_lfr - 1) * n + m;
    const int T_padded_min = T + left;
    const int right = (needed_rows > T_padded_min) ? (needed_rows - T_padded_min) : 0;
    const int T_padded = T_padded_min + right;

    // Build padded buffer.
    std::vector<float> padded((size_t)T_padded * (size_t)D);
    // Left: repeat frame[0].
    for (int i = 0; i < left; i++) {
        std::copy(feats, feats + D, padded.data() + (size_t)i * D);
    }
    // Original frames.
    std::copy(feats, feats + (size_t)T * D, padded.data() + (size_t)left * D);
    // Right: repeat frame[T-1].
    for (int i = 0; i < right; i++) {
        std::copy(feats + (size_t)(T - 1) * D, feats + (size_t)T * D, padded.data() + (size_t)(T_padded_min + i) * D);
    }

    std::vector<float> out((size_t)T_lfr * (size_t)m * (size_t)D);
    for (int t = 0; t < T_lfr; t++) {
        const float* src = padded.data() + (size_t)(t * n) * (size_t)D;
        float* dst = out.data() + (size_t)t * (size_t)m * (size_t)D;
        std::copy(src, src + (size_t)m * (size_t)D, dst);
    }
    T_lfr_out = T_lfr;
    return out;
}

} // namespace core_lfr
