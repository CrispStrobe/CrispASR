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
// FOUR of the five consumers are settled: chatterbox, confucius4, dots-tts and
// fireredtts3 all have PyTorch upstreams and all measure TOWARD_REFERENCE on
// window_width, with the fbank pinned identical across arms so pooling is the
// only variable, and fireredtts3 reproducing its known 0.99999-vs-0.264
// signature to prove the instrument:
//
//     chatterbox/confucius4  cos 0.999999 vs 0.998062   |ref| 13.6330
//     dots-tts               cos 0.999996 vs 0.999839   |ref| 20.4277
//     fireredtts3            cos 0.999997 vs 0.264410   |ref| 20.9117
//
// COSYVOICE3 IS NOT SETTLED. Its upstream is campplus.onnx, and two runs on the
// SAME clip (jfk.wav, T_cam=549, tail=49) disagree:
//
//     onnxruntime on Kaggle   |onnx| 13.6330  -> agrees with window_width
//     onnxruntime 1.23.2      |onnx| 14.0386  -> agrees with kernel_size exactly
//
// i.e. onnxruntime builds disagree with each other on AveragePool(ceil_mode=1),
// so "what does ONNX do" has no single answer to look up.
//
// One fact is independent of that, and it is the uncomfortable one: the eight
// speaker embeddings BAKED INTO the shipped cosyvoice3-voices.gguf match
// kernel_size exactly (cos 1.000000, |x| 14.1197 on zero_shot). That artifact
// was produced by this project's own converter on another machine. So with
// window_width the runtime `--voice ref.wav` path and the baked voice bank
// describe the same voice slightly differently (cos ~0.998).
//
// We ship window_width for cosyvoice3 anyway, for now: it is what the other
// four use, end-to-end passes 8/8 on BOTH arms, and flipping on a contested
// measurement is how this already produced one regression. Tracked, not closed.
// Resolving it needs the onnxruntime version pinned on both sides, and probably
// a re-bake of the voice bank against whichever convention wins.
//
// Two lessons paid for here:
//   * an ISOLATED operator with the same attributes is NOT the graph -- probing
//     an exported AveragePool alone gave the opposite answer and put a
//     regression on main;
//   * end-to-end cannot decide this at all. Every backend passes on both arms.
enum class tail_divisor {
    window_width, // settled for 4 of 5; the default, incl. cosyvoice3 for now.
    kernel_size,  // no production caller; A/B, and the cosyvoice3 candidate.
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
