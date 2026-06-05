// src/core/dac_decoder.h -- Descript Audio Codec (DAC) decoder (header-only).
//
// DAC is a neural audio codec used by Zonos TTS (PLAN #130) and
// potentially by Dia (#136) and Parler (#137). This header provides
// the graph-building helpers for the decoder conv stack:
//
//   codes (9 codebooks) -> RVQ dequant -> decoder conv stack -> 44.1 kHz PCM
//
// Architecture (descript/dac_44khz):
//
//   Quantizer: 9 codebooks, each 1024 entries x 8-dim
//     For each codebook k:
//       z_k = embedding_lookup(codes[k])    -> (8, T)
//       z_k = out_proj_k(z_k)               -> (1024, T) linear 8->1024
//     z_q = sum_k(z_k)                      -> (1024, T)
//
//   Decoder:
//     [0] Conv1d(1024, 1536, k=7, p=3) - input conv
//     [1] DecoderBlock(1536, 768, stride=8) - 8x upsample
//     [2] DecoderBlock(768, 384, stride=8) - 8x upsample
//     [3] DecoderBlock(384, 192, stride=4) - 4x upsample
//     [4] DecoderBlock(192, 96, stride=2) - 2x upsample
//     [5] Snake1d(96)
//     [6] Conv1d(96, 1, k=7, p=3) - output conv
//     [7] Tanh
//
//   DecoderBlock(in_ch, out_ch, stride s):
//     [0] Snake1d(in_ch)
//     [1] ConvTranspose1d(in_ch, out_ch, k=2*s, stride=s, p=s/2)
//     [2] ResidualUnit(out_ch, dilation=1)
//     [3] ResidualUnit(out_ch, dilation=3)
//     [4] ResidualUnit(out_ch, dilation=9)
//
//   ResidualUnit(dim, dilation=d):
//     y = Snake1d(dim) -> Conv1d(dim, dim, k=7, p=3*d, dilation=d)
//         -> Snake1d(dim) -> Conv1d(dim, dim, k=1)
//     return x + y
//
//   Snake1d: y = x + (1/alpha) * sin^2(alpha * x)
//            alpha is per-channel learnable (1, C, 1)
//
// Total upsampling factor: 8*8*4*2 = 512.
// So ~86 tokens/s at 44.1 kHz.
//
// This header provides weight structures and graph-building functions.
// The actual GGUF loading is done by the consumer (zonos_tts.cpp or
// a standalone dac_decoder.cpp).
//
// Tensor naming convention for GGUF (from convert-dac-to-gguf.py):
//   dac.quant.K.*          - codebook K embeddings
//   dac.quant_proj.K.*     - codebook K out_proj (8 -> 1024)
//   dac.dec.in_conv.*      - input Conv1d
//   dac.dec.blk.B.0.*      - block B Snake1d alpha
//   dac.dec.blk.B.1.*      - block B ConvTranspose1d
//   dac.dec.blk.B.{2,3,4}.* - block B ResidualUnits (d=1,3,9)
//   dac.dec.out_snake.*    - output Snake1d alpha
//   dac.dec.out_conv.*     - output Conv1d (dim -> 1)

#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace core_dac {

// Configuration constants for descript/dac_44khz
struct DacConfig {
    int n_codebooks = 9;
    int codebook_size = 1024;
    int codebook_dim = 8;
    int hidden_size = 1024;         // quantizer output / decoder input
    int decoder_hidden_size = 1536; // decoder first conv output channels
    int sample_rate = 44100;
    int hop_length = 512; // total upsample factor
    int n_decoder_blocks = 4;
    int upsampling_ratios[4] = {8, 8, 4, 2};
    int decoder_channels[5] = {1536, 768, 384, 192, 96};
    int residual_dilations[3] = {1, 3, 9};
};

// Per-codebook quantizer weights
struct DacQuantizer {
    ggml_tensor* codebook = nullptr;   // (codebook_dim, codebook_size) or (codebook_size, codebook_dim)
    ggml_tensor* out_proj_w = nullptr; // (codebook_dim, hidden_size) linear weight
    ggml_tensor* out_proj_b = nullptr; // (hidden_size,) linear bias -- may be null
};

// ResidualUnit weights
struct DacResUnit {
    ggml_tensor* alpha0 = nullptr;  // Snake1d alpha (1, dim, 1)
    ggml_tensor* conv0_w = nullptr; // Conv1d k=7 weight
    ggml_tensor* conv0_b = nullptr; // Conv1d k=7 bias
    ggml_tensor* alpha1 = nullptr;  // Snake1d alpha (1, dim, 1)
    ggml_tensor* conv1_w = nullptr; // Conv1d k=1 weight
    ggml_tensor* conv1_b = nullptr; // Conv1d k=1 bias
};

// DecoderBlock weights
struct DacDecoderBlock {
    ggml_tensor* snake_alpha = nullptr; // Snake1d alpha (1, in_ch, 1)
    ggml_tensor* up_w = nullptr;        // ConvTranspose1d weight
    ggml_tensor* up_b = nullptr;        // ConvTranspose1d bias
    DacResUnit res[3];                  // dilation 1, 3, 9
};

// Full DAC decoder weight set
struct DacWeights {
    DacConfig config;
    std::vector<DacQuantizer> quantizers; // [n_codebooks]

    // Decoder
    ggml_tensor* in_conv_w = nullptr;       // Conv1d(hidden, decoder_hidden, k=7) weight
    ggml_tensor* in_conv_b = nullptr;       // Conv1d bias
    DacDecoderBlock blocks[4];              // 4 decoder blocks
    ggml_tensor* out_snake_alpha = nullptr; // final Snake1d alpha
    ggml_tensor* out_conv_w = nullptr;      // Conv1d(96, 1, k=7) weight
    ggml_tensor* out_conv_b = nullptr;      // Conv1d bias
};

// -----------------------------------------------------------------------
// Graph-building helpers
// -----------------------------------------------------------------------

// Snake activation: y = x + (1/alpha) * sin^2(alpha * x)
// x: (C, T) F32, alpha: (1, C, 1) F16 -> (C, T) F32
static inline ggml_tensor* snake(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    const int C = (int)x->ne[0];
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, C, 1);
    a = ggml_cast(ctx, a, GGML_TYPE_F32);
    ggml_tensor* ax = ggml_mul(ctx, x, a);
    ggml_tensor* s = ggml_sin(ctx, ax);
    ggml_tensor* s2 = ggml_sqr(ctx, s);
    ggml_tensor* div = ggml_div(ctx, s2, a);
    return ggml_add(ctx, x, div);
}

// Conv1d wrapper: x(C_in, T) * w(k, C_in, C_out) + b(C_out,) -> (C_out, T)
// Standard causal=false conv with padding = (kernel_size-1)/2 * dilation
// For the DAC decoder, all convs use symmetric padding.
// NOTE: This is a placeholder. The actual implementation needs to handle
// the ggml tensor layout conventions properly.
// See orpheus_snac.cpp for the working Conv1d pattern.

} // namespace core_dac
