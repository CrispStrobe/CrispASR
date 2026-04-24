// vibevoice.h — Microsoft VibeVoice-ASR (σ-VAE tokenizers + Qwen2 LM).
//
// Architecture: Two ConvNeXt-style tokenizer encoders (acoustic + semantic)
// → linear connectors → Qwen2-1.5B autoregressive decoder.
// Input: raw 24kHz mono PCM. Output: structured text with timestamps.
// 1.5B params (ASR path), 4.7 GB F16, MIT license.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vibevoice_context;

struct vibevoice_context_params {
    int n_threads;
    int max_new_tokens;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
};

struct vibevoice_context_params vibevoice_context_default_params(void);

struct vibevoice_context* vibevoice_init_from_file(const char* path_model, struct vibevoice_context_params params);

void vibevoice_free(struct vibevoice_context* ctx);

// Transcribe raw 24kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* vibevoice_transcribe(struct vibevoice_context* ctx, const float* samples, int n_samples);

// ── Stage-level API for differential testing ─────────────────────────────────

// Run the acoustic σ-VAE encoder. Returns a malloc'd float array of shape
// [*n_frames * *vae_dim] in row-major order (frame-major: data[t*vae_dim+c]).
// Caller frees with free(). Returns NULL on failure.
float* vibevoice_run_acoustic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                                      int* vae_dim);

// Run the semantic encoder. Same layout as acoustic. Returns NULL on failure.
float* vibevoice_run_semantic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                                      int* vae_dim);

// Run one SpeechConnector (FC1 → RMSNorm → FC2) on pre-computed encoder mean.
// prefix: "at_conn" (acoustic) or "se_conn" (semantic).
// encoder_mean: row-major [n_frames * vae_dim].
// Returns malloc'd float [n_frames * *d_lm]. Caller frees. Returns NULL on failure.
float* vibevoice_run_connector(struct vibevoice_context* ctx, const char* prefix, const float* encoder_mean,
                               int n_frames, int vae_dim, int* d_lm);

// Run both encoders + both connectors and return the combined speech features
// (element-wise sum of acoustic and semantic connector outputs).
// Returns malloc'd float [*n_frames * *d_lm]. Caller frees. Returns NULL on failure.
float* vibevoice_encode_speech(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                               int* d_lm);

#ifdef __cplusplus
}
#endif
