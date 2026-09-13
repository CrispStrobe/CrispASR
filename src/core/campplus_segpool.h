#pragma once
// CAM++ segment pooling (3D-Speaker CAMPPlus, CAMLayer.seg_pooling).
//
// Header-only so the parity test can exercise it without linking the backend,
// matching core/chatterbox_hift_simdconv.h.
//
// Mirrors, for stype='avg':
//     F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
//
// THE PARTIAL TAIL WINDOW DIVIDES BY ITS OWN WIDTH, NOT BY seg_len.
// ATen computes hend = min(hstart + k, L + pad) and takes pool_size from the
// CLAMPED window, so count_include_pad=True still divides the tail by its
// actual width. Settled by running torch rather than by reading the flag:
//
//     F.avg_pool1d(torch.ones(1,1,551), kernel_size=100, stride=100,
//                  ceil_mode=True)   ->  all six segments are exactly 1.0
//
// The all-ones input is what makes that decisive: under the kernel-size
// divisor the 51-frame tail averages to 0.51, so the two rules render
// differently. This was wrong for as long as the CAM++ port existed --
// every consumer (chatterbox, confucius4, cosyvoice3, dots, fireredtts3) was
// accepted end-to-end, and no acceptance test diffs this stage against
// upstream, so nothing could see it.

#include <algorithm>
#include <cstddef>

namespace campplus_segpool {

// ceil_mode=True: the trailing partial window still produces a segment.
inline int n_segments(int T, int seg_len) {
    if (T <= 0 || seg_len <= 0)
        return 0;
    return (T + seg_len - 1) / seg_len;
}

// `in` is (C, T) row-major; `out` is (C, n_segments(T, seg_len)) row-major.
inline void avg(const float* in, int C, int T, int seg_len, float* out) {
    if (!in || !out || C <= 0 || T <= 0 || seg_len <= 0)
        return;
    const int n_seg = n_segments(T, seg_len);
    for (int c = 0; c < C; c++) {
        const float* row = in + (size_t)c * (size_t)T;
        for (int s = 0; s < n_seg; s++) {
            const int t0 = s * seg_len;
            const int n_in_seg = std::min(seg_len, T - t0);
            float ss = 0.0f;
            for (int t = 0; t < n_in_seg; t++)
                ss += row[t0 + t];
            // Divide by the frames ACTUALLY in this window. n_in_seg ==
            // seg_len for every full segment, so only the tail differs.
            out[(size_t)c * (size_t)n_seg + (size_t)s] = ss / (float)n_in_seg;
        }
    }
}

} // namespace campplus_segpool
