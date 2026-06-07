// src/tada_codec.h — TADA codec decoder (C ABI).
//
// Converts expanded acoustic feature sequences (512-d vectors at 50 Hz)
// into 24 kHz mono PCM via:
//   1. Linear projection (512 → 1024)
//   2. Local-attention transformer encoder (6 layers, 1024-d, 8 heads, RoPE)
//   3. DAC-style upsampler (WNConv1d + Snake1d, strides [4,4,5,6] → 480×)

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tada_codec_context;

struct tada_codec_context* tada_codec_init_from_file(const char* path, int n_threads);

// Decode expanded features to PCM.
// features: (n_frames, 512) float32 row-major
// token_masks: (n_frames,) int32 — 1 where features are non-zero
// Returns heap-allocated PCM, caller frees with tada_codec_pcm_free().
float* tada_codec_decode(struct tada_codec_context* ctx, const float* features, int n_frames,
                         const int32_t* token_masks, int* out_n_samples);

void tada_codec_pcm_free(float* pcm);
void tada_codec_free(struct tada_codec_context* ctx);

#ifdef __cplusplus
}
#endif
