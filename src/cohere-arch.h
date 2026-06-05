#pragma once

// Cohere Transcribe GGUF tensor names.
// These match the names written by export_gguf.py.

// clang-format off

// --- Feature extraction ---
#define CT_FE_MEL_FB       "fe.mel_fb"       // [1, 128, 257]  mel filterbank
#define CT_FE_WINDOW       "fe.window"       // [400]          Hann window

// --- Pre-encode Conv2D subsampling ---
// 3× stride-2 depthwise-separable Conv2d: (1,128,T) → (256,16,T/8) → Linear(4096,1280)
#define CT_PRE_CONV0_W     "enc.pre.conv.0.weight"   // [256,  1, 3, 3]  regular conv2d
#define CT_PRE_CONV0_B     "enc.pre.conv.0.bias"
#define CT_PRE_CONV2_W     "enc.pre.conv.2.weight"   // [256,  1, 3, 3]  depthwise
#define CT_PRE_CONV2_B     "enc.pre.conv.2.bias"
#define CT_PRE_CONV3_W     "enc.pre.conv.3.weight"   // [256,256, 1, 1]  pointwise
#define CT_PRE_CONV3_B     "enc.pre.conv.3.bias"
#define CT_PRE_CONV5_W     "enc.pre.conv.5.weight"   // [256,  1, 3, 3]  depthwise
#define CT_PRE_CONV5_B     "enc.pre.conv.5.bias"
#define CT_PRE_CONV6_W     "enc.pre.conv.6.weight"   // [256,256, 1, 1]  pointwise
#define CT_PRE_CONV6_B     "enc.pre.conv.6.bias"
#define CT_PRE_OUT_W       "enc.pre.out.weight"      // [1280, 4096]
#define CT_PRE_OUT_B       "enc.pre.out.bias"

// --- Conformer encoder layer N  (use snprintf with "enc.blk.%d.*") ---
// Feed-forward 1 (Macaron pre-attention, scale 0.5)
#define CT_ENC_FF1_NORM_W  "enc.blk.%d.ff1.norm.weight"
#define CT_ENC_FF1_NORM_B  "enc.blk.%d.ff1.norm.bias"
#define CT_ENC_FF1_UP_W    "enc.blk.%d.ff1.up.weight"     // [ffn, d]
#define CT_ENC_FF1_UP_B    "enc.blk.%d.ff1.up.bias"
#define CT_ENC_FF1_DN_W    "enc.blk.%d.ff1.down.weight"   // [d, ffn]
#define CT_ENC_FF1_DN_B    "enc.blk.%d.ff1.down.bias"
// Self-attention (Transformer-XL relative position)
#define CT_ENC_ATN_NORM_W  "enc.blk.%d.attn.norm.weight"
#define CT_ENC_ATN_NORM_B  "enc.blk.%d.attn.norm.bias"
#define CT_ENC_ATN_Q_W     "enc.blk.%d.attn.q.weight"     // [d, d]
#define CT_ENC_ATN_Q_B     "enc.blk.%d.attn.q.bias"
#define CT_ENC_ATN_K_W     "enc.blk.%d.attn.k.weight"
#define CT_ENC_ATN_K_B     "enc.blk.%d.attn.k.bias"
#define CT_ENC_ATN_V_W     "enc.blk.%d.attn.v.weight"
#define CT_ENC_ATN_V_B     "enc.blk.%d.attn.v.bias"
#define CT_ENC_ATN_OUT_W   "enc.blk.%d.attn.out.weight"
#define CT_ENC_ATN_OUT_B   "enc.blk.%d.attn.out.bias"
#define CT_ENC_ATN_POS_W   "enc.blk.%d.attn.pos.weight"   // [d, d]  linear_pos
#define CT_ENC_ATN_POS_U   "enc.blk.%d.attn.pos_bias_u"   // [heads, head_dim]
#define CT_ENC_ATN_POS_V   "enc.blk.%d.attn.pos_bias_v"
// Convolution module
#define CT_ENC_CNV_NORM_W  "enc.blk.%d.conv.norm.weight"
#define CT_ENC_CNV_NORM_B  "enc.blk.%d.conv.norm.bias"
#define CT_ENC_CNV_PW1_W   "enc.blk.%d.conv.pw1.weight"   // [2d, d, 1]
#define CT_ENC_CNV_PW1_B   "enc.blk.%d.conv.pw1.bias"
#define CT_ENC_CNV_DW_W    "enc.blk.%d.conv.dw.weight"    // [d, 1, k]
#define CT_ENC_CNV_DW_B    "enc.blk.%d.conv.dw.bias"
#define CT_ENC_CNV_BN_W    "enc.blk.%d.conv.bn.weight"    // batch-norm scale
#define CT_ENC_CNV_BN_B    "enc.blk.%d.conv.bn.bias"
#define CT_ENC_CNV_BN_MEAN "enc.blk.%d.conv.bn.mean"
#define CT_ENC_CNV_BN_VAR  "enc.blk.%d.conv.bn.var"
#define CT_ENC_CNV_PW2_W   "enc.blk.%d.conv.pw2.weight"   // [d, d, 1]
#define CT_ENC_CNV_PW2_B   "enc.blk.%d.conv.pw2.bias"
// Feed-forward 2 (Macaron post-conv, scale 0.5)
#define CT_ENC_FF2_NORM_W  "enc.blk.%d.ff2.norm.weight"
#define CT_ENC_FF2_NORM_B  "enc.blk.%d.ff2.norm.bias"
#define CT_ENC_FF2_UP_W    "enc.blk.%d.ff2.up.weight"
#define CT_ENC_FF2_UP_B    "enc.blk.%d.ff2.up.bias"
#define CT_ENC_FF2_DN_W    "enc.blk.%d.ff2.down.weight"
#define CT_ENC_FF2_DN_B    "enc.blk.%d.ff2.down.bias"
// Output norm
#define CT_ENC_OUT_NORM_W  "enc.blk.%d.out_norm.weight"
#define CT_ENC_OUT_NORM_B  "enc.blk.%d.out_norm.bias"

// --- Encoder→decoder projection ---
#define CT_ENC_PROJ_W      "enc.proj.weight"   // [dec_d, enc_d]
#define CT_ENC_PROJ_B      "enc.proj.bias"

// --- Decoder embeddings ---
#define CT_DEC_EMB_W       "dec.emb.weight"       // [vocab, dec_d]
#define CT_DEC_POS_W       "dec.pos.weight"       // [max_ctx, dec_d]
#define CT_DEC_EMB_LN_W    "dec.emb_ln.weight"
#define CT_DEC_EMB_LN_B    "dec.emb_ln.bias"

// --- Decoder transformer layer N  (use snprintf with "dec.blk.%d.*") ---
#define CT_DEC_ATTN_LN_W   "dec.blk.%d.attn_ln.weight"
#define CT_DEC_ATTN_LN_B   "dec.blk.%d.attn_ln.bias"
#define CT_DEC_ATTN_Q_W    "dec.blk.%d.attn_q.weight"   // [d, d]
#define CT_DEC_ATTN_Q_B    "dec.blk.%d.attn_q.bias"
#define CT_DEC_ATTN_K_W    "dec.blk.%d.attn_k.weight"
#define CT_DEC_ATTN_K_B    "dec.blk.%d.attn_k.bias"
#define CT_DEC_ATTN_V_W    "dec.blk.%d.attn_v.weight"
#define CT_DEC_ATTN_V_B    "dec.blk.%d.attn_v.bias"
#define CT_DEC_ATTN_O_W    "dec.blk.%d.attn_o.weight"
#define CT_DEC_ATTN_O_B    "dec.blk.%d.attn_o.bias"
#define CT_DEC_XATTN_LN_W  "dec.blk.%d.cross_ln.weight"
#define CT_DEC_XATTN_LN_B  "dec.blk.%d.cross_ln.bias"
#define CT_DEC_XATTN_Q_W   "dec.blk.%d.cross_q.weight"
#define CT_DEC_XATTN_Q_B   "dec.blk.%d.cross_q.bias"
#define CT_DEC_XATTN_K_W   "dec.blk.%d.cross_k.weight"
#define CT_DEC_XATTN_K_B   "dec.blk.%d.cross_k.bias"
#define CT_DEC_XATTN_V_W   "dec.blk.%d.cross_v.weight"
#define CT_DEC_XATTN_V_B   "dec.blk.%d.cross_v.bias"
#define CT_DEC_XATTN_O_W   "dec.blk.%d.cross_o.weight"
#define CT_DEC_XATTN_O_B   "dec.blk.%d.cross_o.bias"
#define CT_DEC_FFN_LN_W    "dec.blk.%d.ffn_ln.weight"
#define CT_DEC_FFN_LN_B    "dec.blk.%d.ffn_ln.bias"
#define CT_DEC_FFN_UP_W    "dec.blk.%d.ffn_up.weight"   // [ffn, d]
#define CT_DEC_FFN_UP_B    "dec.blk.%d.ffn_up.bias"
#define CT_DEC_FFN_DN_W    "dec.blk.%d.ffn_down.weight" // [d, ffn]
#define CT_DEC_FFN_DN_B    "dec.blk.%d.ffn_down.bias"

// --- Decoder output ---
#define CT_DEC_OUT_LN_W    "dec.out_ln.weight"
#define CT_DEC_OUT_LN_B    "dec.out_ln.bias"
#define CT_DEC_HEAD_W      "dec.head.weight"   // [vocab, dec_d]
#define CT_DEC_HEAD_B      "dec.head.bias"

// --- GGUF metadata keys ---
#define CT_KEY_VOCAB_SIZE       "cohere_transcribe.vocab_size"
#define CT_KEY_ENC_N_LAYERS     "cohere_transcribe.encoder.n_layers"
#define CT_KEY_ENC_D_MODEL      "cohere_transcribe.encoder.d_model"
#define CT_KEY_ENC_N_HEADS      "cohere_transcribe.encoder.n_heads"
#define CT_KEY_ENC_HEAD_DIM     "cohere_transcribe.encoder.head_dim"
#define CT_KEY_ENC_FFN_DIM      "cohere_transcribe.encoder.ffn_dim"
#define CT_KEY_ENC_CONV_KERNEL  "cohere_transcribe.encoder.conv_kernel"
#define CT_KEY_DEC_N_LAYERS     "cohere_transcribe.decoder.n_layers"
#define CT_KEY_DEC_D_MODEL      "cohere_transcribe.decoder.d_model"
#define CT_KEY_DEC_N_HEADS      "cohere_transcribe.decoder.n_heads"
#define CT_KEY_DEC_HEAD_DIM     "cohere_transcribe.decoder.head_dim"
#define CT_KEY_DEC_FFN_DIM      "cohere_transcribe.decoder.ffn_dim"
#define CT_KEY_DEC_MAX_CTX      "cohere_transcribe.decoder.max_ctx"
#define CT_KEY_AUDIO_SR         "cohere_transcribe.audio.sample_rate"
#define CT_KEY_AUDIO_N_MELS     "cohere_transcribe.audio.n_mels"
#define CT_KEY_AUDIO_N_FFT      "cohere_transcribe.audio.n_fft"
#define CT_KEY_AUDIO_HOP        "cohere_transcribe.audio.hop_length"
#define CT_KEY_AUDIO_WIN        "cohere_transcribe.audio.win_length"

// clang-format on
