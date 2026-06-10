// src/core/seanet_decoder.h — parameterized SEANet CNN decoder (header-only).
//
// Hoists the SEANet decoder architecture shared by the major neural audio
// codec families:
//
//   SNAC     — orpheus_snac.cpp   (hubertsiuzdak/snac_24khz)
//   EnCodec  — (future) Bark backend
//   DAC      — (future) Dia, Parler backends
//   Mimi     — (future) CSM, Pocket backends
//
// All four codec families descend from the same SEANet architecture
// (Défossez et al., "High Fidelity Neural Audio Compression", 2022):
//
//   Decode pipeline:
//     1. RVQ codebook dequantize → sum across codebook levels.
//     2. Conv1d pre-projection (latent_dim → decoder_dim).
//     3. N DecoderBlocks: activation → ConvTranspose1d upsample
//        → M ResidualUnits (activation → dilated Conv1d → activation → Conv1d → add).
//     4. Conv1d post-projection (final_dim → 1) → tanh → PCM.
//
// Differences between codec families are captured by configuration:
//   - Number and sizes of decoder blocks / residual units.
//   - Stride pattern (e.g., [8,8,4,2] for SNAC, [8,5,4,2] for EnCodec).
//   - Activation function (Snake for SNAC/DAC, ELU for EnCodec, SiLU for Mimi).
//   - Whether convolutions are depthwise or standard.
//   - Codebook count, size, and VQ strides.
//
// This header provides:
//   1. A configuration struct (SeanetConfig) for parameterizing the decoder.
//   2. Tensor slot structs (ResUnitSlots, BlockSlots, DecoderSlots) for
//      binding GGUF tensors.
//   3. Graph-building helpers that construct ggml ops for each stage.
//
// The helpers use the same (C, T) channels-innermost layout as the SNAC
// implementation. The caller is responsible for:
//   - Loading the GGUF and binding tensors into the slot structs.
//   - Providing the activation function (since it varies by codec).
//   - Managing the ggml context, backend, and compute lifecycle.

// ---------------------------------------------------------------------------
// Per-source adoption verdict (audited 2026-05-31):
//
//   SNAC (orpheus_snac.cpp)  — FAITHFUL target. This is the reference
//                              architecture; the (C,T) layout and crop
//                              convention match. NOTE: the actual SNAC
//                              backend has NOT yet been wired to this
//                              header (dead code), but the ops match.
//   EnCodec / DAC / Mimi      — FUTURE. Architecturally compatible via
//                              Config, but unverified — treat as
//                              FAITHFUL-WITH-CONFIG-pending-audit until a
//                              real backend diff confirms.
//
// ConvTranspose1d cropping is delegated to core_convt::convt1d_crop
// (src/core/conv.h), which is the proven Kokoro/SNAC implementation.
// Do NOT re-implement the crop inline — the naive (C,T)↔(T,C) view is
// easy to get wrong (it crops channels instead of time).
// ---------------------------------------------------------------------------

#pragma once

#include "conv.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace core_seanet {

// Configuration for a SEANet-family decoder.
struct Config {
    // Codebook parameters.
    int n_codebooks = 3;
    int codebook_size = 4096;
    int codebook_dim = 8;
    int latent_dim = 768;
    int decoder_dim = 1024;

    // VQ strides: how many time frames each codebook level spans.
    // Length = n_codebooks. Example: [4, 2, 1] for SNAC.
    std::vector<int> vq_strides;

    // Decoder block strides: upsample factor for each block.
    // Example: [8, 8, 4, 2] for SNAC → total upsample = 512.
    std::vector<int> decoder_strides;

    // Residual unit dilations within each block.
    // Example: [1, 3, 9] for SNAC.
    std::vector<int> residual_dilations;

    // Whether convolutions are depthwise (SNAC/DAC) or standard (EnCodec).
    bool depthwise = true;

    // Pre-projection configuration.
    // SNAC uses depthwise conv (k=7) + pointwise conv (k=1).
    // EnCodec uses a single Conv1d (k=7).
    int pre_conv_kernel = 7;

    // Post-projection kernel size.
    int post_conv_kernel = 7;
};

// Tensor slots for one residual unit.
struct ResUnitSlots {
    ggml_tensor* act0 = nullptr;    // activation param (e.g., Snake alpha), or nullptr
    ggml_tensor* conv0_w = nullptr; // first conv weight
    ggml_tensor* conv0_b = nullptr; // first conv bias
    ggml_tensor* act1 = nullptr;    // second activation param
    ggml_tensor* conv1_w = nullptr; // second conv weight
    ggml_tensor* conv1_b = nullptr; // second conv bias
};

// Tensor slots for one decoder block.
struct BlockSlots {
    ggml_tensor* act = nullptr;      // block activation param
    ggml_tensor* up_w = nullptr;     // ConvTranspose1d weight
    ggml_tensor* up_w_perm = nullptr;// pre-permuted [IC, K*OC] for decomposed path (or nullptr)
    ggml_tensor* up_b = nullptr;     // ConvTranspose1d bias
    std::vector<ResUnitSlots> res;   // residual units
};

// Tensor slots for the codebook quantizer (decode direction only).
struct QuantSlots {
    ggml_tensor* codebook = nullptr;   // (codebook_dim, codebook_size)
    ggml_tensor* out_proj_w = nullptr; // (1, codebook_dim, latent_dim)
    ggml_tensor* out_proj_b = nullptr; // (latent_dim,) — may be nullptr
};

// Tensor slots for the full decoder.
struct DecoderSlots {
    std::vector<QuantSlots> quantizers;

    // Pre-projection. For depthwise configs, two stages:
    ggml_tensor* pre_dw_w = nullptr; // depthwise conv weight (optional)
    ggml_tensor* pre_dw_b = nullptr;
    ggml_tensor* pre_pw_w = nullptr; // pointwise / standard conv weight
    ggml_tensor* pre_pw_b = nullptr;

    std::vector<BlockSlots> blocks;

    // Post-projection.
    ggml_tensor* post_act = nullptr; // final activation param
    ggml_tensor* post_w = nullptr;   // final conv weight
    ggml_tensor* post_b = nullptr;   // final conv bias
};

// Type alias for the activation function builder.
// Given (ctx, x, activation_param), returns the activated tensor.
// activation_param may be nullptr for parameterless activations (ELU, SiLU).
using ActivationFn = std::function<ggml_tensor*(ggml_context*, ggml_tensor*, ggml_tensor*)>;

// ---------------------------------------------------------------------------
// Graph-building helpers.
// ---------------------------------------------------------------------------

// Pointwise conv1d k=1: weight ne=[1, Cin, Cout].
// Input (Cin, T) → output (Cout, T).
static inline ggml_tensor* pw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int Cin = (int)w->ne[1];
    const int Cout = (int)w->ne[2];
    ggml_tensor* W = ggml_reshape_2d(ctx, w, Cin, Cout);
    ggml_tensor* y = ggml_mul_mat(ctx, W, x);
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// Standard conv1d with same-padding. Weight ne=[K, Cin, Cout].
// Input (Cin, T) → output (Cout, T).
static inline ggml_tensor* conv1d_same(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int pad) {
    const int T = (int)x->ne[1];
    const int Cout = (int)w->ne[2];
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, x));
    y = ggml_conv_1d(ctx, w, y, /*s*/ 1, pad, /*d*/ 1);
    y = ggml_reshape_2d(ctx, y, T, Cout);
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// Depthwise conv1d (groups = C). Weight ne=[K, 1, C].
// Input (C, T) → output (C, T).
static inline ggml_tensor* dw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K,
                                     int dil) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];
    const int p = dil * (K - 1) / 2;

    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, x));
    y = ggml_conv_1d_dw(ctx, w, y, /*s*/ 1, p, dil);
    y = ggml_reshape_2d(ctx, y, T, C);
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// ConvTranspose1d with symmetric cropping.
// Weight ne=[K, Cout, Cin] (ggml ConvTranspose1d format).
// Input (Cin, T) → output (Cout, T_out) where T_out = T*stride - crop_left - crop_right.
//
// Delegates to the proven core_convt::convt1d_crop (src/core/conv.h):
// it correctly crops the TIME axis of the (T_raw, Cout) transpose output
// (time-innermost) rather than the channel axis. The previous inline
// re-implementation here was BROKEN — it cropped channels instead of time.
static inline ggml_tensor* convt1d_crop(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                        int crop_left, int crop_right) {
    return core_convt::convt1d_crop(ctx, x, w, b, stride, crop_left, crop_right);
}

// Repeat each time step `factor` times (repeat_interleave along time).
// Input (C, T) → output (C, T*factor).
static inline ggml_tensor* repeat_interleave_time(ggml_context* ctx, ggml_tensor* x, int factor) {
    if (factor <= 1) {
        return x;
    }
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];
    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, C, 1, T);
    ggml_tensor* target = ggml_new_tensor_3d(ctx, x->type, C, factor, T);
    ggml_tensor* tiled = ggml_repeat(ctx, x3, target);
    return ggml_reshape_2d(ctx, tiled, C, T * factor);
}

// Build the RVQ dequantize + sum graph.
//
// codes_in   — array of n_codebooks input tensors, each 1D I32 with
//              T_q / vq_strides[k] entries.
// slots      — quantizer tensor slots.
// cfg        — decoder configuration.
//
// Returns z_q of shape (latent_dim, T_q).
static inline ggml_tensor* build_rvq_dequantize(ggml_context* ctx, ggml_tensor** codes_in, const QuantSlots* slots,
                                                const Config& cfg) {
    const int T_q = (int)(codes_in[0]->ne[0]) * cfg.vq_strides[0];
    ggml_tensor* z_q = nullptr;

    for (int k = 0; k < cfg.n_codebooks; k++) {
        // Codebook lookup.
        ggml_tensor* z = ggml_get_rows(ctx, slots[k].codebook, codes_in[k]);
        z = ggml_cont(ctx, ggml_cast(ctx, z, GGML_TYPE_F32));

        // Out projection (pointwise conv 1×1).
        z = pw_conv1d(ctx, z, slots[k].out_proj_w, slots[k].out_proj_b);

        // Upsample to T_q time steps.
        z = repeat_interleave_time(ctx, z, cfg.vq_strides[k]);

        if (k == 0) {
            z_q = z;
        } else {
            z_q = ggml_add(ctx, z_q, z);
        }
    }
    return ggml_cont(ctx, z_q);
}

// Build one residual unit graph.
//
// For depthwise mode:
//   y = act(x) → depthwise_conv(k=7, dilation=d) → act(y) → pointwise_conv(k=1)
//   return x + y
//
// For standard mode:
//   y = act(x) → conv1d(k=K, dilation=d) → act(y) → conv1d(k=1)
//   return x + y
static inline ggml_tensor* build_residual_unit(ggml_context* ctx, ggml_tensor* x, const ResUnitSlots& u, int dilation,
                                               bool depthwise, const ActivationFn& act_fn) {
    ggml_tensor* y = act_fn(ctx, x, u.act0);
    if (depthwise) {
        const int K = (int)u.conv0_w->ne[0];
        y = dw_conv1d(ctx, y, u.conv0_w, u.conv0_b, K, dilation);
    } else {
        const int K = (int)u.conv0_w->ne[0];
        const int p = dilation * (K - 1) / 2;
        y = conv1d_same(ctx, y, u.conv0_w, u.conv0_b, p);
    }
    y = act_fn(ctx, y, u.act1);
    y = pw_conv1d(ctx, y, u.conv1_w, u.conv1_b);
    return ggml_add(ctx, x, y);
}

// Build one decoder block graph.
//
//   act(x) → ConvTranspose1d(stride) → residual units
static inline ggml_tensor* build_decoder_block(ggml_context* ctx, ggml_tensor* x, const BlockSlots& blk, int stride,
                                               const std::vector<int>& dilations, bool depthwise,
                                               const ActivationFn& act_fn) {
    x = act_fn(ctx, x, blk.act);
    if (blk.up_w_perm) {
        const int K = (int)blk.up_w->ne[0];
        x = core_convt::convt1d_decomp(ctx, x, blk.up_w_perm, blk.up_b, stride, K,
                                       /*crop_left=*/stride / 2, /*crop_right=*/stride / 2);
    } else {
        x = convt1d_crop(ctx, x, blk.up_w, blk.up_b, stride, /*crop_left=*/stride / 2, /*crop_right=*/stride / 2);
    }

    for (int r = 0; r < (int)blk.res.size() && r < (int)dilations.size(); r++) {
        x = build_residual_unit(ctx, x, blk.res[r], dilations[r], depthwise, act_fn);
    }
    return x;
}

// Build the full decoder graph from dequantized latent z_q.
//
//   z_q           — (latent_dim, T_q) from build_rvq_dequantize.
//   slots         — decoder tensor slots.
//   cfg           — decoder configuration.
//   act_fn        — activation function builder.
//
// Returns the PCM output tensor (1D, T_q * product(decoder_strides)).
static inline ggml_tensor* build_decoder(ggml_context* ctx, ggml_tensor* z_q, const DecoderSlots& slots,
                                         const Config& cfg, const ActivationFn& act_fn) {
    ggml_tensor* h = z_q;

    // Pre-projection.
    if (cfg.depthwise && slots.pre_dw_w) {
        h = dw_conv1d(ctx, h, slots.pre_dw_w, slots.pre_dw_b, cfg.pre_conv_kernel, /*dil*/ 1);
    }
    if (slots.pre_pw_w) {
        if (slots.pre_pw_w->ne[0] == 1) {
            // Pointwise conv.
            h = pw_conv1d(ctx, h, slots.pre_pw_w, slots.pre_pw_b);
        } else {
            // Standard conv.
            h = conv1d_same(ctx, h, slots.pre_pw_w, slots.pre_pw_b, cfg.pre_conv_kernel / 2);
        }
    }

    // Decoder blocks.
    for (int b = 0; b < (int)cfg.decoder_strides.size() && b < (int)slots.blocks.size(); b++) {
        h = build_decoder_block(ctx, h, slots.blocks[b], cfg.decoder_strides[b], cfg.residual_dilations, cfg.depthwise,
                                act_fn);
        h = ggml_cont(ctx, h);
    }

    // Post-projection: activation → conv → tanh.
    h = act_fn(ctx, h, slots.post_act);
    const int post_pad = cfg.post_conv_kernel / 2;
    h = conv1d_same(ctx, h, slots.post_w, slots.post_b, post_pad);
    h = ggml_tanh(ctx, h);

    // Reshape to 1D PCM.
    const int T_pcm = (int)h->ne[1];
    return ggml_reshape_1d(ctx, h, (int64_t)T_pcm);
}

} // namespace core_seanet
