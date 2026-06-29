// chatterbox_campplus.h — internal CAMPPlus speaker encoder (Module 4).
//
// Module 4 of the chatterbox WAV→cond port, paired with chatterbox_ve
// (module 2 / VoiceEncoder LSTM) and chatterbox_s3tok (module 3 /
// S3Tokenizer V2). Produces the 192-d speaker x-vector that S3Gen's
// CFM denoiser consumes as `gen.embedding`.
//
// Pipeline (matches `chatterbox.models.s3gen.xvector.CAMPPlus.inference`):
//   1. Kaldi-compatible 80-bin fbank of the 16 kHz mono ref audio
//      (Povey window, 25 ms / 10 ms frames, HTK mel, log power)
//   2. Subtract the per-utterance mean of the fbank along time
//   3. FCM head: 2-D conv stack on (B, 1, n_mels=80, T) → (B, 320, T)
//      - conv1 (1→32, k=3) + BN + ReLU
//      - layer1 (2 BasicResBlocks, 1st stride=(2,1)) → (B, 32, 40, T)
//      - layer2 (2 BasicResBlocks, 1st stride=(2,1)) → (B, 32, 20, T)
//      - conv2 (32→32, k=3, s=(2,1)) + BN + ReLU → (B, 32, 10, T)
//      - reshape (B, 32*10, T) = (B, 320, T)
//   4. xvector chain on (B, 320, T):
//      - tdnn(320→128, k=5, s=2) + BN + ReLU                 → (B, 128, T/2)
//      - block1 (12 CAMDenseTDNNLayers, growth=32)            → (B, 512, T/2)
//      - transit1 (512→256) + BN + ReLU                      → (B, 256, T/2)
//      - block2 (24 CAMDenseTDNNLayers, growth=32, dilation=2)→ (B, 1024, T/2)
//      - transit2 (1024→512) + BN + ReLU                     → (B, 512, T/2)
//      - block3 (16 CAMDenseTDNNLayers, growth=32, dilation=2)→ (B, 1024, T/2)
//      - transit3 (1024→512) + BN + ReLU                     → (B, 512, T/2)
//      - out_nl (BN + ReLU)
//      - StatsPool: concat(mean(T), std(T))                  → (B, 1024)
//      - dense (1024→192) Conv1d(k=1) + BN(affine=False)     → (B, 192)
//
// Tensor names (s3.se.* in the chatterbox-s3gen GGUF) — alias-shortened
// at convert time vs the upstream module names:
//   head.{conv1, conv2}.weight                        F16  Conv2d (no bias)
//   head.{bn1, bn2}.{weight, bias, running_mean, running_var}    F32
//   head.layer{1,2}.{0,1}.conv{1,2}.weight            F16  Conv2d in 32→32
//   head.layer{1,2}.{0,1}.bn{1,2}.{...}               F32
//   head.layer{1,2}.0.shortcut.0.weight               F16  Conv2d (1×1)  when stride=2
//   head.layer{1,2}.0.shortcut.1.{...}                F32  matching BN
//   xv.tdnn.linear.weight                             F16  Conv1d
//   xv.tdnn.nl.bn.{...}                               F32
//   xv.block{1,2,3}.tdnnd{1..N}.{l1, cam.ll, cam.l1, cam.l2}.weight   F16
//   xv.block{1,2,3}.tdnnd{1..N}.cam.{l1, l2}.bias     F32
//   xv.block{1,2,3}.tdnnd{1..N}.{nonl1, nonl2}.bn.{...}              F32
//   xv.transit{1,2,3}.linear.weight                   F16  Conv1d 1×1, no bias
//   xv.transit{1,2,3}.nl.bn.{...}                     F32
//   xv.out_nl.bn.{...}                                F32  final BN
//   xv.dense.linear.weight                            F16  Conv1d 1×1
//   xv.dense.nl.bn.{running_mean, running_var}        F32  affine=False (no weight/bias)

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

// 2-D resblock weights (the FCM head's `BasicResBlock`). When the block's
// `stride!=1`, an additional 1×1 conv + BN shortcut is also bound; otherwise
// `shortcut_*` stay nullptr.
struct cb_campplus_resblock {
    ggml_tensor* conv1_w = nullptr; // F16  (k=3, k=3, in, out)
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* bn1_w = nullptr; // F32  (out,)
    ggml_tensor* bn1_b = nullptr;
    ggml_tensor* bn1_m = nullptr;   // running_mean
    ggml_tensor* bn1_v = nullptr;   // running_var
    ggml_tensor* conv2_w = nullptr; // F16  (k=3, k=3, out, out)
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* bn2_w = nullptr;
    ggml_tensor* bn2_b = nullptr;
    ggml_tensor* bn2_m = nullptr;
    ggml_tensor* bn2_v = nullptr;
    ggml_tensor* sc_w = nullptr; // F16  (1, 1, in, out)  — 1×1 shortcut
    ggml_tensor* sc_b = nullptr;
    ggml_tensor* sc_bn_w = nullptr;
    ggml_tensor* sc_bn_b = nullptr;
    ggml_tensor* sc_bn_m = nullptr;
    ggml_tensor* sc_bn_v = nullptr;
    int stride = 1; // first-block stride along H (2 for downsamplers)
};

// FCM head — 2-D conv stack on the mel-vs-time grid.
struct cb_campplus_fcm {
    ggml_tensor* conv1_w = nullptr;
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* bn1_w = nullptr;
    ggml_tensor* bn1_b = nullptr;
    ggml_tensor* bn1_m = nullptr;
    ggml_tensor* bn1_v = nullptr;
    std::vector<cb_campplus_resblock> layer1; // 2 blocks
    std::vector<cb_campplus_resblock> layer2; // 2 blocks
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* bn2_w = nullptr;
    ggml_tensor* bn2_b = nullptr;
    ggml_tensor* bn2_m = nullptr;
    ggml_tensor* bn2_v = nullptr;
};

// CAMDenseTDNNLayer weights — see xvector.py for the structure.
struct cb_campplus_dense_layer {
    // nonl1.bn — runs on the layer input (in_channels grows per layer).
    ggml_tensor* nonl1_bn_w = nullptr;
    ggml_tensor* nonl1_bn_b = nullptr;
    ggml_tensor* nonl1_bn_m = nullptr;
    ggml_tensor* nonl1_bn_v = nullptr;
    // l1: bottleneck Conv1d(in_channels → bn_channels, k=1, no bias)
    ggml_tensor* l1_w = nullptr;
    ggml_tensor* l1_b = nullptr;
    // nonl2.bn — runs on the bottleneck output (bn_channels=128).
    ggml_tensor* nonl2_bn_w = nullptr;
    ggml_tensor* nonl2_bn_b = nullptr;
    ggml_tensor* nonl2_bn_m = nullptr;
    ggml_tensor* nonl2_bn_v = nullptr;
    // CAM layer:
    //   linear_local: Conv1d(bn_channels → out_channels=growth_rate=32, k=3,
    //                        dilation=d, padding=(k-1)/2*d, no bias)
    //   linear1:      Conv1d(bn_channels → bn_channels/2, k=1)
    //   linear2:      Conv1d(bn_channels/2 → out_channels, k=1)
    ggml_tensor* cam_ll_w = nullptr;
    ggml_tensor* cam_l1_w = nullptr;
    ggml_tensor* cam_l1_b = nullptr;
    ggml_tensor* cam_l2_w = nullptr;
    ggml_tensor* cam_l2_b = nullptr;
};

struct cb_campplus_dense_block {
    int num_layers = 0;
    int dilation = 1;
    std::vector<cb_campplus_dense_layer> layers;
};

// Conv1d-with-BN unit used by `tdnn`, `transit*`, `out_nl`, `dense`.
struct cb_campplus_unit {
    ggml_tensor* lin_w = nullptr; // F16  Conv1d weight (kw, in, out)
    ggml_tensor* lin_b = nullptr; // optional bias
    ggml_tensor* bn_w = nullptr;
    ggml_tensor* bn_b = nullptr;
    ggml_tensor* bn_m = nullptr;
    ggml_tensor* bn_v = nullptr;
};

struct cb_campplus_model {
    cb_campplus_fcm head;
    cb_campplus_unit tdnn;
    cb_campplus_dense_block block1; // 12 layers, dilation=1
    cb_campplus_unit transit1;
    cb_campplus_dense_block block2; // 24 layers, dilation=2
    cb_campplus_unit transit2;
    cb_campplus_dense_block block3; // 16 layers, dilation=2
    cb_campplus_unit transit3;
    cb_campplus_unit out_nl; // BN-only (no linear)
    cb_campplus_unit dense;  // BN affine=False
};

namespace chatterbox_campplus {

// Per-context runtime cache for the CPU forward — holds the pre-folded
// BatchNorm params so subsequent calls don't recompute them. Allocate once
// per chatterbox_s3gen_context (the s3gen layer owns the lifetime). Lazy
// init on first compute_xvector call.
struct cb_campplus_runtime {
    bool initialised = false;
    void* impl = nullptr; // pimpl — see chatterbox_campplus.cpp
    cb_campplus_runtime();
    ~cb_campplus_runtime();
    cb_campplus_runtime(const cb_campplus_runtime&) = delete;
    cb_campplus_runtime& operator=(const cb_campplus_runtime&) = delete;
};

// Compute the 80-bin Kaldi fbank features for the given 16 kHz mono PCM
// buffer and subtract the per-utterance mean along time (matches
// `extract_feature` in xvector.py: `feature - feature.mean(0, keepdim=True)`).
// Returns row-major (T_frames, 80) float32 features. Sets T_frames_out=0
// on error. Does NOT scale to int16 — CAMPPlus consumes raw [-1, 1] floats.
std::vector<float> compute_fbank(const float* pcm_16k, int n_samples, int& T_frames_out);

// Run the CAMPPlus forward on a (T, 80) feature buffer (already
// mean-subtracted via `compute_fbank`) and return the 192-d speaker
// x-vector. Pure CPU — no ggml graph. Reads all weights into host F32 once
// per `cache` instance.
//
// Pipeline reproduces `CAMPPlus.forward`:
//   (T,80) → permute → (80,T) → unsqueeze(1) → (1,80,T) [Conv2d input]
//   FCM head: Conv2d + BN + ReLU + 4 BasicResBlocks + Conv2d + BN + ReLU
//             → reshape (320, T)
//   xv: tdnn(320→128, k=5, s=2) + BN + ReLU → (128, T/2)
//       block1 (12 layers, dilation=1) → 512 channels
//       transit1 (512→256)
//       block2 (24 layers, dilation=2) → 1024 channels
//       transit2 (1024→512)
//       block3 (16 layers, dilation=2) → 1024 channels
//       transit3 (1024→512)
//       out_nl (BN + ReLU)
//       StatsPool (concat mean+std along T) → 1024-d
//       dense (1024→192) Conv1d(k=1) + BN(affine=False) → 192-d
//
// Returns the (192,) f32 vector. Empty on error.
// `stats_var_floor` clamps the StatsPool variance before sqrt: 0 (default) =
// chatterbox/CosyVoice behaviour; 1e-2 = 3D-Speaker masked stats pooling (dots.tts).
// The embedding dimension (192 chatterbox / 512 dots) is inferred from the bound
// dense weight, so no separate parameter is needed.
std::vector<float> compute_xvector(const cb_campplus_model& m, cb_campplus_runtime& cache, const float* feat_t_80,
                                   int T, float stats_var_floor = 0.0f);

// Convenience: PCM → fbank → xvector.
std::vector<float> embed_speaker(const cb_campplus_model& m, cb_campplus_runtime& cache, const float* pcm_16k,
                                 int n_samples, float stats_var_floor = 0.0f);

// Module 4 phase 3 — 24 kHz Matcha-TTS prompt mel for `gen.prompt_feat`.
// Mirrors `chatterbox.models.s3gen.utils.mel.mel_spectrogram` with
// CosyVoice's defaults: n_fft=1920, hop=480, win=1920, n_mels=80,
// fmin=0, fmax=8000, sampling_rate=24000, center=False (manual reflect
// pad of (n_fft - hop)/2 = 720 samples each side), magnitude (NOT
// power), librosa Slaney mel, log compression `log(clamp(mel, 1e-5))`.
//
// Returns row-major (T_mel, n_mels=80) — the layout S3Gen consumes.
// Truncates the input audio to `max_samples` if non-zero
// (`prepare_conditionals` truncates to DEC_COND_LEN = 10 * S3GEN_SR =
// 240000 samples before computing the mel).
std::vector<float> compute_prompt_feat_24k(const float* pcm_24k, int n_samples, int max_samples, int& T_mel_out);

} // namespace chatterbox_campplus
