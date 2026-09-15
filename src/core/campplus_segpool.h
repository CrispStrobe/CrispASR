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
#include <cstdlib>

namespace campplus_segpool {

// WHICH REFERENCE a consumer must match on the PARTIAL TAIL window.
//
// These are not "right" and "wrong" — they are two upstreams, and a backend is
// correct only against its own. Measured by RUNNING both, not by reading specs:
//
//   torch  F.avg_pool1d(ceil_mode=True) divides the tail by its OWN width.
//          Verified: all-ones, T=551 -> every segment exactly 1.0.
//
//   onnx   campplus.onnx as EXPORTED divides the tail by the KERNEL size.
//          Verified by running the real graph, with a no-tail control
//          (T_cam=100) where all arms agree at cos 1.000000 -- which rules out
//          any weight or front-end difference and pins it to the tail alone.
//          At T_cam=173 the graph matches the kernel-size arm EXACTLY
//          (cos 1.000000, |x| 14.1197) and the width arm only to 0.992663.
//          NOTE the isolated OPERATOR with those same attributes divides by
//          width, so the spec would have said "ONNX agrees with torch". The
//          exported graph does not. The mechanism inside the export was not
//          isolated; this is measured behaviour, not an explained cause.
enum class tail_divisor {
    window_width, // torch  — chatterbox, confucius4, dots, fireredtts3
    kernel_size,  // onnx   — cosyvoice3 ONLY
};


// ceil_mode=True: the trailing partial window still produces a segment.
inline int n_segments(int T, int seg_len) {
    if (T <= 0 || seg_len <= 0)
        return 0;
    return (T + seg_len - 1) / seg_len;
}

// `in` is (C, T) row-major; `out` is (C, n_segments(T, seg_len)) row-major.
inline void avg(const float* in, int C, int T, int seg_len, float* out,
                tail_divisor tail = tail_divisor::window_width) {
    if (!in || !out || C <= 0 || T <= 0 || seg_len <= 0)
        return;
    const int n_seg = n_segments(T, seg_len);
    // Read once per call, not per element.
    const char* lg = std::getenv("CRISPASR_CAMPP_LEGACY_SEGPOOL");
    const bool legacy = lg && lg[0] && lg[0] != '0';
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
            //
            // The caller names its upstream; the env var forces kernel_size
            // globally for A/B measurement.
            const bool use_kernel = legacy || tail == tail_divisor::kernel_size;
            const float divisor = use_kernel ? (float)seg_len : (float)n_in_seg;
            out[(size_t)c * (size_t)n_seg + (size_t)s] = ss / divisor;
        }
    }
}

} // namespace campplus_segpool
