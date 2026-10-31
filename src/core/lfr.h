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
#include <cstring>
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

    std::vector<float> out((size_t)T_lfr * (size_t)m * (size_t)D);
    float* dst = out.data();
    for (int t = 0; t < T_lfr; t++) {
        const int start_frame = t * n - left;
        for (int k = 0; k < m; k++) {
            int src_frame = start_frame + k;
            if (src_frame < 0)
                src_frame = 0;
            else if (src_frame >= T)
                src_frame = T - 1;
            std::memcpy(dst, feats + (size_t)src_frame * (size_t)D, (size_t)D * sizeof(float));
            dst += D;
        }
    }
    T_lfr_out = T_lfr;
    return out;
}

} // namespace core_lfr
