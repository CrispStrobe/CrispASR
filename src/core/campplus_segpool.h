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
// ALL FIVE CAM++ CONSUMERS WANT window_width. That is measured, and it was
// briefly believed otherwise, so the evidence is recorded here rather than
// re-litigated.
//
// An earlier probe compared an ISOLATED AveragePool export at T_cam=173 and
// concluded that campplus.onnx — cosyvoice3's upstream — divides the tail by
// the kernel size, which would have made cosyvoice3 the one consumer needing
// kernel_size. Running the ACTUAL C++ arms against the ACTUAL campplus.onnx on
// real input (T_cam=549, tail=49) says the opposite, with the fbank pinned
// identical across arms so the pooling is the only variable:
//
//     cosyvoice3   cos_fixed 0.999999   cos_legacy 0.998052
//                  |fixed|   13.6296    |legacy|   14.0340   |onnx ref| 13.6330
//
// A second, independent reference (funasr) agrees to 7 decimals. Every other
// consumer reports the same direction against its own upstream, and fireredtts3
// reproduces its known signature (0.99999 fixed vs 0.264 legacy), so the
// instrument is trustworthy. All five: TOWARD_REFERENCE on the fixed arm.
//
// The lesson worth keeping: an isolated operator with the same attributes is
// NOT the graph. Measure the arms you ship against the reference you ship
// against, on the input you ship with.
//
// kernel_size therefore has no production caller. It is kept ONLY so the old
// divisor can be rebuilt for A/B — same purpose as
// CRISPASR_CAMPP_LEGACY_SEGPOOL — because "the fix moved it toward upstream"
// has to stay a measurement rather than becoming folklore.
enum class tail_divisor {
    window_width, // what ALL FIVE consumers match. The default.
    kernel_size,  // the old divisor, for A/B only — no production caller.
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
