// src/core/conv.h — convolution helpers that work around ggml limitations.
//
// ggml has no `groups` argument on `ggml_conv_1d` or
// `ggml_conv_transpose_1d`, so any depthwise / grouped conv has to be
// open-coded. This header collects the specific shapes that come up
// repeatedly across the BigVGAN-family vocoder ports (Kokoro, future
// iSTFTNet variants, possibly mimo codec).
//
// Currently:
//   convt1d_depthwise_2x_k3  — depthwise ConvTranspose1d with kernel=3,
//                              stride=2, padding=1, output_padding=1.
//                              Used for 2× upsamples in iSTFTNet-style
//                              vocoder pool layers.
//   convt1d_crop             — channels-first ConvTranspose1d wrapper
//                              that handles the (C,T) ↔ (T,C) transpose
//                              dance and lets the caller specify how
//                              many time samples to crop from each end
//                              (causal vs symmetric padding).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "parallel_for.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace core_convt {

// Depthwise ConvTranspose1d with parameters (k=3, s=2, p=1, op=1).
// Output length = 2 · T_in.
//
// PyTorch ConvTranspose1d emits `y[i] = sum input[j] · weight[k]` over
// (j, k) satisfying `j·stride + k − padding = i`. For our config:
//
//   y[c, 2t]   = w[c, 1] · x[c, t]                                  (j=t,   k=1)
//   y[c, 2t+1] = w[c, 2] · x[c, t] + w[c, 0] · x[c, t+1]            (j=t,k=2 + j=t+1,k=0)
//                                                                   (x[c, T]=0 boundary)
//
// **Critical**: `w[2]` and `w[0]` are NOT interchangeable in the odd
// case — getting the kernel ends swapped produces plausible-but-wrong
// audio that can survive informal QA. The Kokoro M11 diff harness
// caught exactly this bug (commit 448c1af); see LEARNINGS.md
// "Kokoro / StyleTTS2 lessons" Lesson 2.
//
// Inputs:
//   x        : (C, T)        F32, channel-major.
//   w_kernel : (K=3, 1, C)   F16, depthwise kernel (PyTorch
//              `nn.ConvTranspose1d(C, C, k=3, s=2, p=1, op=1, groups=C)`
//              stores weights as `(C, 1, K)` and the converter
//              transposes to `(K, 1, C)` for ggml).
//   w_bias   : (C,)          F32, optional per-channel bias (broadcast
//              over time). Pass nullptr to skip.
//
// Output: (C, 2·T) F32.
static inline ggml_tensor* convt1d_depthwise_2x_k3(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_kernel,
                                                   ggml_tensor* w_bias) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];

    // Permute kernel (K=3, 1, C) → (C, 3, 1), cast to F32 (F16 view + F32
    // mul fails on Metal at the kernel-dispatch level), reshape to
    // (C, 3), then take three column views w0/w1/w2.
    ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_kernel, 2, 0, 1, 3)); // (C, 3, 1) F16
    ggml_tensor* w_perm_f32 = ggml_cast(ctx, w_perm, GGML_TYPE_F32);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w_perm_f32, C, 3); // (C, 3) F32
    const size_t row_b = w_2d->nb[1];
    ggml_tensor* w0 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)0 * row_b);
    ggml_tensor* w1 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)1 * row_b);
    ggml_tensor* w2 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)2 * row_b);

    // x_shifted[c, t] = x[c, t+1] for t < T-1, 0 for t = T-1.
    // Take x[:, 1:] (C, T-1) and zero-pad on the right to (C, T).
    ggml_tensor* x_tail = ggml_view_2d(ctx, x, C, T - 1, x->nb[1], x->nb[1]);   // (C, T-1)
    x_tail = ggml_cont(ctx, x_tail);                                            // contiguous
    ggml_tensor* x_shifted = ggml_pad_ext(ctx, x_tail, 0, 0, 0, 1, 0, 0, 0, 0); // (C, T)

    // y_even (C, T) = w1 ⊙ x  (broadcast w1 over T)
    ggml_tensor* y_even = ggml_mul(ctx, x, w1);
    // y_odd (C, T) = w2 ⊙ x + w0 ⊙ x_shifted   (PyTorch ConvTranspose1d
    // kernel indexing — see derivation note above)
    ggml_tensor* y_odd = ggml_add(ctx, ggml_mul(ctx, x, w2), ggml_mul(ctx, x_shifted, w0));

    // Interleave: reshape both to (C, 1, T), concat dim=1 → (C, 2, T),
    // reshape to (C, 2T). Memory layout means consecutive time positions
    // alternate even/odd, which is the desired interleaving.
    ggml_tensor* even_3d = ggml_reshape_3d(ctx, y_even, C, 1, T);
    ggml_tensor* odd_3d = ggml_reshape_3d(ctx, y_odd, C, 1, T);
    ggml_tensor* stacked = ggml_concat(ctx, even_3d, odd_3d, /*dim=*/1);      // (C, 2, T)
    ggml_tensor* y = ggml_cont(ctx, ggml_reshape_2d(ctx, stacked, C, 2 * T)); // (C, 2T)

    if (w_bias)
        y = ggml_add(ctx, y, w_bias);
    return y;
}

// Channels-first ConvTranspose1d (groups=1) with caller-controlled
// time-axis cropping.
//
// ggml_conv_transpose_1d expects (T, Cin) input and emits T_unpad =
// (T_in - 1)·stride + K samples; it has no padding parameter. Most
// callers want a smaller T_out and crop the excess from the ends:
//
//   - **Causal upsamplers** (qwen3-tts codec) trim the right tail only:
//     `crop_left=0, crop_right=K-stride` so T_out = T_in · stride.
//   - **Symmetric-pad upsamplers** (SNAC, with k=2s, p=s/2) crop the
//     same amount from each end: `crop_left=crop_right=stride/2`,
//     giving T_out = T_in · stride.
//
// Inputs:
//   x         : (Cin, T_in)   F32, channel-major.
//   w         : (K, Cout, Cin) F16/F32, ggml weight layout (PyTorch
//               numpy `(Cin, Cout, K)` transposed by the converter).
//   b         : (Cout,)       F32 or nullptr.
//   stride    : positive integer.
//   crop_left : samples to crop from the start of the time axis (≥ 0).
//   crop_right: samples to crop from the end of the time axis (≥ 0).
//
// Output: (Cout, T_unpad - crop_left - crop_right) F32.
static inline ggml_tensor* convt1d_crop(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                        int crop_left, int crop_right) {
    const int Cout = (int)w->ne[1];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));          // (T_in, Cin)
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xT, stride, 0, 1); // (T_unpad, Cout, 1, 1)
    const int T_unpad = (int)y->ne[0];
    const int T_out = T_unpad - crop_left - crop_right;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout);
    if (crop_left > 0 || crop_right > 0) {
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_unpad * sizeof(float), (size_t)crop_left * sizeof(float));
        y = ggml_cont(ctx, y); // (T_out, Cout)
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// General-purpose decomposed ConvTranspose1d via mul_mat + col2im_1d.
// Uses pre-permuted weights w_perm [IC, K*OC] and the col2im_1d op.
//
// Supports both causal (crop_left=0, crop_right=K-stride) and symmetric
// (crop_left=crop_right=stride/2) cropping patterns used across all TTS
// decoder families.
//
// Inputs:
//   x          : (Cin, T_in)  F32, channel-major.
//   w_perm     : (IC, K*OC)   F32, weight pre-permuted at load time.
//   b          : (Cout,)      F32 or nullptr.
//   stride     : positive integer.
//   K          : kernel size.
//   crop_left  : samples to crop from the start of the time axis (≥ 0).
//   crop_right : samples to crop from the end of the time axis (≥ 0).
//
// Output: (Cout, T_unpad - crop_left - crop_right) F32.
static inline ggml_tensor* convt1d_decomp(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_perm, ggml_tensor* b,
                                          int stride, int K, int crop_left, int crop_right) {
    const int OC = (int)w_perm->ne[1] / K;

    // mul_mat contracts IC → col: [K*OC, T_in]
    ggml_tensor* col = ggml_mul_mat(ctx, w_perm, x);

    // col2im: [K*OC, T_in] → [T_full, OC]  (GATHER, overlap-add)
    //
    // p0 = 0 asks for the FULL uncropped signal, T_full = (T_in-1)*stride + K.
    // upstream's ggml_col2im_1d crops p0 from BOTH sides
    // (T_out = (T_in-1)*s0 + K - 2*p0) while the kernel treats p0 as a left
    // offset (t_abs = t_out + p0). We need ASYMMETRIC cropping — causal decoders
    // want crop_left=0, crop_right=K-stride — which that symmetric formula cannot
    // express. Taking the whole signal and cropping once in a view is both exact
    // and cheaper than the old two-step (op-side left offset + tail view).
    ggml_tensor* y = ggml_col2im_1d(ctx, col, stride, OC, /*p0=*/0);

    if (crop_left > 0 || crop_right > 0) {
        const int64_t T_keep = y->ne[0] - crop_left - crop_right;
        GGML_ASSERT(T_keep > 0);
        y = ggml_view_2d(ctx, y, T_keep, y->ne[1], y->nb[1], (size_t)crop_left * y->nb[0]);
        y = ggml_cont(ctx, y);
    }

    // [T_out, OC] → [OC, T_out]  (back to channels-first)
    y = ggml_cont(ctx, ggml_transpose(ctx, y));

    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// Causal ConvTranspose1d via decomposed mul_mat + col2im.
// Convenience wrapper: crop_left=0, crop_right=K-stride.
// Output: (Cout, T_in * stride) F32.
static inline ggml_tensor* convt1d_causal_decomp(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_perm, ggml_tensor* b,
                                                 int stride, int K) {
    return convt1d_decomp(ctx, x, w_perm, b, stride, K, /*crop_left=*/0, /*crop_right=*/K - stride);
}

// Time-first variant of convt1d_decomp for runtimes that use ggml's
// native (T, C) convention (e.g. IndExTTS BigVGAN, Chatterbox S3Gen).
// Input:  x = (T_in, Cin) F32.
// Output: (T_out, Cout) F32 where T_out = (T_in-1)*stride + K - crop_left - crop_right.
// Bias b = (Cout,) or nullptr — applied as (1, Cout) broadcast.
static inline ggml_tensor* convt1d_decomp_tf(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_perm, ggml_tensor* b,
                                             int stride, int K, int crop_left, int crop_right) {
    // (T, C) → (C, T) for the channels-first decomp path
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* y = convt1d_decomp(ctx, xt, w_perm, nullptr, stride, K, crop_left, crop_right);
    // (Cout, T_out) → (T_out, Cout) back to time-first
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]));
    }
    return y;
}

// ---------------------------------------------------------------------------
// Weight permutation utility (host-side, called at load time).
// ---------------------------------------------------------------------------

// Permute a ConvTranspose1d weight from ggml layout [K, OC, IC] to the
// decomposed layout [IC, K*OC] needed by convt1d_decomp / mul_mat.
//
// Supports F32 and F16 source tensors. Always produces F32 output.
// Returns a heap-allocated buffer; caller owns it.
//
// Usage at load time:
//   auto buf = permute_convt1d_weight(src_tensor);
//   ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, K*OC);
//   ggml_backend_tensor_set(dst, buf.get(), 0, ggml_nbytes(dst));

// Check if sequential baseline conv permutation is requested.
static inline bool crispasr_conv_serial() {
    const char* e = std::getenv("CRISPASR_CONV_SERIAL");
    return (e && *e && std::strcmp(e, "0") != 0);
}

template <typename Load> static inline void permute_convt1d_host_baseline(Load load, float* dp, int K, int OC, int IC) {
    const size_t J = (size_t)OC * K;
    const int B = 32;
    const int n_blk = (IC + B - 1) / B;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if ((size_t)IC * J > ((size_t)1 << 20))
#endif
    for (int bi = 0; bi < n_blk; bi++) {
        const int i0 = bi * B;
        const int i1 = i0 + B < IC ? i0 + B : IC;
        for (size_t j0 = 0; j0 < J; j0 += B) {
            const size_t j1 = j0 + B < J ? j0 + B : J;
            for (size_t j = j0; j < j1; j++)
                for (int i = i0; i < i1; i++)
                    dp[j * IC + i] = load((size_t)i * J + j);
        }
    }
}

template <typename Load>
static inline void permute_convt1d_host_accelerated(Load load, float* dp, int K, int OC, int IC, int n_threads = 0) {
    const size_t J = (size_t)OC * K;
    const int B = 32;
    const int n_blk = (IC + B - 1) / B;

    core_parallel::for_each_chunk(n_blk, n_threads, [&](int b_start, int b_end) {
        for (int bi = b_start; bi < b_end; bi++) {
            const int i0 = bi * B;
            const int i1 = i0 + B < IC ? i0 + B : IC;
            for (size_t j0 = 0; j0 < J; j0 += B) {
                const size_t j1 = j0 + B < J ? j0 + B : J;
                for (size_t j = j0; j < j1; j++)
                    for (int i = i0; i < i1; i++)
                        dp[j * IC + i] = load((size_t)i * J + j);
            }
        }
    });
}

// Host-side core of the permutation: a plain 2-D transpose of
// src[ic][j] (IC rows, J = OC*K cols) into dst[j][ic], where `load(s)` reads
// source element s as float. 32x32 tiles: the naive loop's stores stride by
// IC floats, and for the voxcpm2 VAE (41M elements) it took ~1.1 s of every
// process start (#461). Tile rows are split across OpenMP threads when the
// including target has it; every dst element is written exactly once from
// the same source value, so the result is byte-identical either way.
template <typename Load>
static inline void permute_convt1d_host(Load load, float* dp, int K, int OC, int IC, int n_threads = 0) {
    if (crispasr_conv_serial() || (size_t)IC * (size_t)OC * K <= (size_t)1 << 16) {
        permute_convt1d_host_baseline(load, dp, K, OC, IC);
    } else {
        permute_convt1d_host_accelerated(load, dp, K, OC, IC, n_threads);
    }
}

static inline std::unique_ptr<float[]> permute_convt1d_weight(ggml_tensor* src) {
    const int K = (int)src->ne[0];
    const int OC = (int)src->ne[1];
    const int IC = (int)src->ne[2];
    const size_t n_elems = (size_t)K * OC * IC;

    auto out = std::make_unique<float[]>((size_t)IC * K * OC);
    float* dp = out.get();

    // src layout: [K, OC, IC] → src[ic][oc][k] = tmp[ic * OC * K + oc * K + k]
    // dst layout: [IC, K*OC]  → dst[oc*K+k][ic] = dp[(oc * K + k) * IC + ic]
    if (src->type == GGML_TYPE_F32) {
        auto tmp = std::make_unique<float[]>(n_elems);
        ggml_backend_tensor_get(src, tmp.get(), 0, n_elems * sizeof(float));
        const float* tp = tmp.get();
        permute_convt1d_host([tp](size_t s) { return tp[s]; }, dp, K, OC, IC);
    } else {
        // F16 (most codec weights are F16)
        auto tmp = std::make_unique<ggml_fp16_t[]>(n_elems);
        ggml_backend_tensor_get(src, tmp.get(), 0, n_elems * sizeof(ggml_fp16_t));
        const ggml_fp16_t* tp = tmp.get();
        permute_convt1d_host([tp](size_t s) { return ggml_fp16_to_fp32(tp[s]); }, dp, K, OC, IC);
    }
    return out;
}

// Batch-permute helper: given an array of (src_tensor, dst_w_perm_ptr) pairs,
// creates a ggml context, permutes all weights, allocates a backend buffer,
// and uploads. Returns the ctx and buf (caller owns both).
//
// Usage:
//   ggml_context* ctx_perm = nullptr;
//   ggml_backend_buffer_t buf_perm = nullptr;
//   ggml_tensor* src_list[] = { ups[0].w, ups[1].w, ... };
//   ggml_tensor** dst_list[] = { &ups[0].w_perm, &ups[1].w_perm, ... };
//   permute_convt1d_weights_batch(src_list, dst_list, n, backend, &ctx_perm, &buf_perm);
static inline bool permute_convt1d_weights_batch(ggml_tensor** srcs, ggml_tensor*** dsts, int n, ggml_backend_t backend,
                                                 ggml_context** out_ctx, ggml_backend_buffer_t* out_buf) {
    // Publish borrowed tensor pointers only after the allocation succeeds.
    // Otherwise callers' fallback paths would dereference a freed context.
    for (int i = 0; i < n; i++)
        if (srcs[i] && srcs[i]->type != GGML_TYPE_F32 && srcs[i]->type != GGML_TYPE_F16)
            return false;
    std::vector<ggml_tensor*> pending(n, nullptr);
    const size_t meta_bytes = ggml_tensor_overhead() * (size_t)n + 4096;
    struct ggml_init_params pp = {meta_bytes, nullptr, true};
    ggml_context* ctx = ggml_init(pp);
    if (!ctx)
        return false;

    std::vector<std::unique_ptr<float[]>> bufs(n);
    for (int i = 0; i < n; i++) {
        if (!srcs[i])
            continue;
        bufs[i] = permute_convt1d_weight(srcs[i]);
        const int IC = (int)srcs[i]->ne[2];
        const int K = (int)srcs[i]->ne[0];
        const int OC = (int)srcs[i]->ne[1];
        pending[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, K * OC);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    for (int i = 0; i < n; i++) {
        if (pending[i] && bufs[i])
            ggml_backend_tensor_set(pending[i], bufs[i].get(), 0, ggml_nbytes(pending[i]));
    }
    for (int i = 0; i < n; i++) {
        *dsts[i] = pending[i];
    }
    *out_ctx = ctx;
    *out_buf = buf;
    return true;
}

} // namespace core_convt
