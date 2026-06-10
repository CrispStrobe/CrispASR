// src/core/hifigan.h -- Shared HiFi-GAN vocoder primitive for ggml graphs.
//
// Converts mel spectrograms to PCM waveforms using the standard HiFi-GAN
// architecture: conv_pre -> (upsample + MRF resblocks) x N -> conv_post.
//
// Used by:
//   - SpeechT5 TTS  (section 132) -- 80-bin mel, rates [4,4,4,4], 16 kHz
//   - FastPitch      (section 133) -- 80-bin mel, rates [8,8,2,2], 22 kHz
//
// Architecture:
//   1. Optional normalize: x = (x - mean) / scale
//   2. conv_pre: Conv1d(mel_dim, upsample_initial_ch, k=7, s=1, p=3)
//   3. For each upsample stage i:
//        h = LeakyReLU(h, slope)
//        h = ConvTranspose1d(ch_i, ch_{i+1}, k=K_i, s=R_i, p=(K-R)/2)
//        res = sum of resblocks / num_kernels
//      Each resblock has num_dilations sub-blocks:
//        for d in dilations:
//          residual = h
//          h = LeakyReLU(h, slope) -> Conv1d(ch, ch, k, d=d, p=d*(k-1)/2)
//          h = LeakyReLU(h, slope) -> Conv1d(ch, ch, k, d=1, p=(k-1)/2)
//          h = h + residual
//   4. h = LeakyReLU(h)
//   5. h = Conv1d(ch_last, 1, k=7, s=1, p=3)
//   6. h = tanh(h)
//
// All weight-norm is pre-fused in the GGUF tensors. Convolutions use
// ggml_conv_1d (stride-1 dilated) and ggml_conv_transpose_1d (upsampling).
// Weight tensor layout: GGUF stores (out_ch, in_ch, kernel) = ggml (ne0=kernel,
// ne1=in_ch, ne2=out_ch) which is what ggml_conv_1d expects.

#pragma once

#include "conv.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace core_hifigan {

// ── Hyperparameters (read from GGUF KV) ────────────────────────────

struct hparams {
    int model_in_dim = 80;
    int upsample_initial_ch = 512;
    float leaky_relu_slope = 0.1f;
    bool normalize_before = true;

    std::vector<int> upsample_rates;                       // e.g. [4,4,4,4]
    std::vector<int> upsample_kernel_sizes;                // e.g. [8,8,8,8]
    std::vector<int> resblock_kernel_sizes;                // e.g. [3,7,11]
    std::vector<std::vector<int>> resblock_dilation_sizes; // e.g. [[1,3,5],[1,3,5],[1,3,5]]

    int num_upsamples() const { return (int)upsample_rates.size(); }
    int num_kernels() const { return (int)resblock_kernel_sizes.size(); }

    // Channel count after upsample stage i (0-indexed).
    int channels_at(int i) const { return upsample_initial_ch >> (i + 1); }
};

// ── Tensor name lookup helper ──────────────────────────────────────

static inline ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

// ── LeakyReLU for ggml (alpha < 1) ──────────────────────────────

static inline ggml_tensor* leaky_relu(ggml_context* ctx, ggml_tensor* x, float slope) {
    // LeakyReLU(x) = max(x, slope * x) = x * (x > 0 ? 1 : slope)
    // Use ggml_leaky_relu if available, otherwise fallback.
    return ggml_leaky_relu(ctx, x, slope, false);
}

// ── Conv1d helper (stride=1, with dilation + padding) ──────────────
//
// weight: (out_ch, in_ch, kernel) stored as ggml (kernel, in_ch, out_ch)
// bias:   (out_ch,) or nullptr
// Input x: (in_ch, T)
// Output:  (out_ch, T') where T' depends on padding.

static inline ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias, int stride,
                                  int padding, int dilation) {
    // ggml_conv_1d signature: (a=weight, b=input, stride, padding, dilation)
    // Input x: (T, C_in), output y: (T_out, C_out)
    ggml_tensor* y = ggml_conv_1d(ctx, weight, x, stride, padding, dilation);
    if (bias) {
        // bias is (C_out,). Reshape to (1, C_out) for broadcast over T (ne[0]).
        ggml_tensor* b = ggml_reshape_2d(ctx, bias, 1, (int)bias->ne[0]);
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// ── ConvTranspose1d helper ─────────────────────────────────────────
//
// weight: (in_ch, out_ch, kernel) stored as ggml (kernel, out_ch, in_ch)
// w_perm: (in_ch, kernel*out_ch) pre-permuted weight for decomposed path,
//         or nullptr to use the old ggml_conv_transpose_1d path.
// For ggml_conv_transpose_1d: (a=weight, b=input, stride, padding, dilation)
// PyTorch padding = (kernel - stride) / 2

static inline ggml_tensor* conv_transpose_1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight,
                                             ggml_tensor* w_perm, ggml_tensor* bias, int stride, int padding) {
    if (w_perm) {
        const int K = (int)weight->ne[0];
        return core_convt::convt1d_decomp(ctx, x, w_perm, bias, stride, K, padding, padding);
    }
    // Old path — stable, works on CPU without the col2im op.
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, weight, x, stride, 0 /*p0*/, 1);
    // Trim padding from both ends: crop from [padding .. T_out - padding]
    if (padding > 0) {
        int64_t T_out = y->ne[0];
        int64_t C_out = y->ne[1];
        int64_t T_trimmed = T_out - 2 * padding;
        // Use ggml_view_2d to crop the time dimension
        y = ggml_view_2d(ctx, y, T_trimmed, C_out,
                         y->nb[1],                        // row stride (bytes per row)
                         padding * ggml_element_size(y)); // offset: skip first 'padding' elements
        y = ggml_cont(ctx, y);
    }
    if (bias) {
        ggml_tensor* b = ggml_reshape_2d(ctx, bias, 1, (int)bias->ne[0]);
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// ── Resblock forward ──────────────────────────────────────────────
//
// One HiFi-GAN resblock: num_dilations sub-blocks, each with 2 conv layers.
//   for d in dilations:
//     residual = x
//     x = LeakyReLU(x) -> Conv1d(x, k, d) -> LeakyReLU -> Conv1d(x, k, 1)
//     x = x + residual

static inline ggml_tensor* resblock_forward(ggml_context* ctx, ggml_tensor* x,
                                            const std::map<std::string, ggml_tensor*>& tensors,
                                            const std::string& prefix, int kernel_size,
                                            const std::vector<int>& dilations, float slope) {
    for (int d = 0; d < (int)dilations.size(); d++) {
        ggml_tensor* residual = x;

        // First conv: dilated
        x = leaky_relu(ctx, x, slope);
        std::string c1w = prefix + ".convs1." + std::to_string(d) + ".weight";
        std::string c1b = prefix + ".convs1." + std::to_string(d) + ".bias";
        int pad1 = dilations[d] * (kernel_size - 1) / 2;
        x = conv1d(ctx, x, T(tensors, c1w), T(tensors, c1b), 1, pad1, dilations[d]);

        // Second conv: dilation=1
        x = leaky_relu(ctx, x, slope);
        std::string c2w = prefix + ".convs2." + std::to_string(d) + ".weight";
        std::string c2b = prefix + ".convs2." + std::to_string(d) + ".bias";
        int pad2 = (kernel_size - 1) / 2;
        x = conv1d(ctx, x, T(tensors, c2w), T(tensors, c2b), 1, pad2, 1);

        // Residual connection
        x = ggml_add(ctx, x, residual);
    }
    return x;
}

// ── Full HiFi-GAN forward pass ──────────────────────────────────

// Build the ggml graph for HiFi-GAN vocoder inference.
//
// mel: (num_mel_bins, T_mel) F32, transposed mel spectrogram (channels-first)
//      The caller is responsible for transposing from (T, mel_bins) if needed.
// tensors: map of GGUF tensor name -> ggml_tensor*
// prefix: tensor name prefix (e.g. "voc" for "voc.conv_pre.weight")
// hp: hyperparameters
// ups_w_perm: pre-permuted upsample weights [IC, K*OC] for each stage,
//             or empty to use the old conv_transpose_1d path. When
//             provided, must have hp.num_upsamples() elements (nullptr
//             entries fall back to the old path for that stage).
//
// Returns: (1, T_audio) F32, mono waveform in [-1, 1].

static inline ggml_tensor* forward(ggml_context* ctx, ggml_tensor* mel,
                                   const std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix,
                                   const hparams& hp,
                                   const std::vector<ggml_tensor*>& ups_w_perm = {}) {
    ggml_tensor* x = mel;

    // Optional normalization: (mel - mean) / scale
    // Input x is (T, C) in ggml convention. mean/scale are (C,).
    if (hp.normalize_before) {
        ggml_tensor* mean = T(tensors, prefix + ".mean");
        ggml_tensor* scale = T(tensors, prefix + ".scale");
        if (mean && scale) {
            // Reshape to (1, C) so it broadcasts over T (ne[0]):
            // sub((T, C), (1, C)) → 1 divides T ✓, C == C ✓
            ggml_tensor* m = ggml_repeat(ctx, ggml_reshape_2d(ctx, mean, 1, (int)mean->ne[0]), x);
            ggml_tensor* s = ggml_repeat(ctx, ggml_reshape_2d(ctx, scale, 1, (int)scale->ne[0]), x);
            x = ggml_div(ctx, ggml_sub(ctx, x, m), s);
        }
    }

    // conv_pre: Conv1d(mel_dim, init_ch, k=7, s=1, p=3)
    x = conv1d(ctx, x, T(tensors, prefix + ".conv_pre.weight"), T(tensors, prefix + ".conv_pre.bias"), 1, 3, 1);

    // Upsample stages
    for (int i = 0; i < hp.num_upsamples(); i++) {
        x = leaky_relu(ctx, x, hp.leaky_relu_slope);

        int rate = hp.upsample_rates[i];
        int ksize = hp.upsample_kernel_sizes[i];
        int pad = (ksize - rate) / 2;

        std::string ups_w = prefix + ".ups." + std::to_string(i) + ".weight";
        std::string ups_b = prefix + ".ups." + std::to_string(i) + ".bias";
        ggml_tensor* wp = (i < (int)ups_w_perm.size()) ? ups_w_perm[i] : nullptr;
        x = conv_transpose_1d(ctx, x, T(tensors, ups_w), wp, T(tensors, ups_b), rate, pad);

        // MRF: sum of resblocks / num_kernels
        ggml_tensor* res_sum = nullptr;
        for (int j = 0; j < hp.num_kernels(); j++) {
            int rb_idx = i * hp.num_kernels() + j;
            std::string rb_prefix = prefix + ".resblocks." + std::to_string(rb_idx);
            int rb_kernel = hp.resblock_kernel_sizes[j];
            const std::vector<int>& rb_dilations = (j < (int)hp.resblock_dilation_sizes.size())
                                                       ? hp.resblock_dilation_sizes[j]
                                                       : hp.resblock_dilation_sizes[0];

            ggml_tensor* rb_out =
                resblock_forward(ctx, x, tensors, rb_prefix, rb_kernel, rb_dilations, hp.leaky_relu_slope);
            if (res_sum == nullptr) {
                res_sum = rb_out;
            } else {
                res_sum = ggml_add(ctx, res_sum, rb_out);
            }
        }
        // Average
        x = ggml_scale(ctx, res_sum, 1.0f / hp.num_kernels());
    }

    // Final activation + conv_post + tanh
    x = leaky_relu(ctx, x, hp.leaky_relu_slope);
    x = conv1d(ctx, x, T(tensors, prefix + ".conv_post.weight"), T(tensors, prefix + ".conv_post.bias"), 1, 3, 1);
    x = ggml_tanh(ctx, x);

    return x;
}

} // namespace core_hifigan
