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

#include "core/conv.h"
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
    ggml_tensor* up_w_perm = nullptr;   // pre-permuted [IC, K*OC] for decomposed path (or nullptr)
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

// Conv1d: x(C_in, T) * w(K, C_in, C_out) + b(C_out,) -> (C_out, T)
// Same-padding with optional dilation. Weight layout matches GGUF convention.
static inline ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K,
                                  int dil = 1) {
    const int T = (int)x->ne[1];
    const int Cout = (int)w->ne[2];
    const int p = dil * (K - 1) / 2;
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, Cin)
    y = ggml_conv_1d(ctx, w, y, /*stride=*/1, p, dil);       // (T, Cout, 1)
    y = ggml_reshape_2d(ctx, y, T, Cout);                    // (T, Cout)
    y = ggml_cont(ctx, ggml_transpose(ctx, y));              // (Cout, T)
    if (b)
        y = ggml_add(ctx, y, b);
    return y;
}

// ConvTranspose1d with symmetric cropping: T_out = T_in * stride
// DAC uses kernel=2*stride, pad=stride/2.
// Uses decomposed mul_mat + col2im_1d when w_perm is available.
static inline ggml_tensor* convt1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* w_perm,
                                   ggml_tensor* b, int stride) {
    const int pad = stride / 2;
    if (w_perm) {
        const int K = (int)w->ne[0];
        return core_convt::convt1d_decomp(ctx, x, w_perm, b, stride, K, pad, pad);
    }
    return core_convt::convt1d_crop(ctx, x, w, b, stride, /*crop_left=*/pad, /*crop_right=*/pad);
}

// ResidualUnit: Snake -> Conv1d(k=7,dil=d) -> Snake -> Conv1d(k=1) -> add
static inline ggml_tensor* res_unit(ggml_context* ctx, ggml_tensor* x, const DacResUnit& u, int dil) {
    ggml_tensor* y = snake(ctx, x, u.alpha0);
    y = conv1d(ctx, y, u.conv0_w, u.conv0_b, 7, dil);
    y = snake(ctx, y, u.alpha1);
    y = conv1d(ctx, y, u.conv1_w, u.conv1_b, 1); // k=1 pointwise
    return ggml_add(ctx, x, y);
}

// DecoderBlock: Snake -> ConvTranspose1d(stride=s) -> 3 x ResidualUnit(d=1,3,9)
static inline ggml_tensor* dec_block(ggml_context* ctx, ggml_tensor* x, const DacDecoderBlock& blk, int stride) {
    x = snake(ctx, x, blk.snake_alpha);
    x = convt1d(ctx, x, blk.up_w, blk.up_w_perm, blk.up_b, stride);
    x = res_unit(ctx, x, blk.res[0], 1);
    x = res_unit(ctx, x, blk.res[1], 3);
    x = res_unit(ctx, x, blk.res[2], 9);
    return x;
}

// Build full DAC decode graph: codes (n_codebooks x T) -> PCM (T*512,)
// Returns the PCM output tensor. Caller must create/alloc the graph.
//
// `codes_in` is an array of n_codebooks I32 tensors, each of length T.
// These must already be created (ggml_new_tensor_1d) and set as inputs.
static inline ggml_tensor* build_decode_graph(ggml_context* ctx, const DacWeights& w, ggml_tensor** codes_in, int /*T*/,
                                              ggml_cgraph* gf) {
    const auto& cfg = w.config;
    const int n_cb = cfg.n_codebooks;

    // RVQ dequantize: for each codebook, lookup + project + sum
    ggml_tensor* z_q = nullptr;
    for (int k = 0; k < n_cb; k++) {
        const auto& q = w.quantizers[k];
        // codebook lookup: (codebook_dim, T)
        ggml_tensor* z = ggml_get_rows(ctx, q.codebook, codes_in[k]);
        z = ggml_cont(ctx, ggml_cast(ctx, z, GGML_TYPE_F32));
        // out_proj: (codebook_dim, T) -> (hidden, T)
        // pw conv: weight is (1, codebook_dim, hidden)
        ggml_tensor* W2d =
            ggml_reshape_2d(ctx, q.out_proj_w, q.out_proj_w->ne[0] * q.out_proj_w->ne[1], q.out_proj_w->ne[2]);
        z = ggml_mul_mat(ctx, W2d, z); // (hidden, T)
        if (q.out_proj_b)
            z = ggml_add(ctx, z, q.out_proj_b);
        z_q = k == 0 ? z : ggml_add(ctx, z_q, z);
    }
    z_q = ggml_cont(ctx, z_q);

    // Input conv: Conv1d(hidden, 1536, k=7, p=3)
    ggml_tensor* h = conv1d(ctx, z_q, w.in_conv_w, w.in_conv_b, 7);

    // 4 decoder blocks: strides [8, 8, 4, 2]
    for (int b = 0; b < cfg.n_decoder_blocks; b++) {
        h = dec_block(ctx, h, w.blocks[b], cfg.upsampling_ratios[b]);
        h = ggml_cont(ctx, h);
    }

    // Final: Snake -> Conv1d(96, 1, k=7, p=3) -> Tanh
    h = snake(ctx, h, w.out_snake_alpha);
    h = conv1d(ctx, h, w.out_conv_w, w.out_conv_b, 7);
    h = ggml_tanh(ctx, h);

    // Output: (1, T_pcm) -> flatten to (T_pcm,)
    int T_pcm = (int)h->ne[1];
    h = ggml_reshape_1d(ctx, h, T_pcm);
    h = ggml_cont(ctx, h);
    ggml_set_name(h, "dac_pcm");
    ggml_set_output(h);
    if (gf)
        ggml_build_forward_expand(gf, h);

    return h;
}

} // namespace core_dac
