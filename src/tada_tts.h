// src/tada_tts.h — TADA-3B-ML TTS runtime (C ABI).
//
// HumeAI/tada-3b-ml: Llama-3.2-3B backbone with per-token flow-matching
// diffusion head (VibeVoiceDiffusionHead) and TADA codec decoder for
// text-to-speech synthesis at 24 kHz mono.
//
// Architecture:
//   1. Llama-3.2-3B (28L, 3072d, 24 heads, 8 KV heads) — same backbone as Orpheus
//      with added acoustic/time embeddings for speech conditioning
//   2. VibeVoiceDiffusionHead — per-token flow-matching ODE solver
//      (4-layer SwiGLU+AdaLN, sinusoidal time embedding, Euler integration)
//   3. TADA codec decoder — LocalAttentionEncoder + DAC upsampler → 24 kHz PCM
//
// Key innovation: 1:1 text-to-speech alignment — every text token maps to
// exactly one acoustic vector. No 7:1 expansion like Orpheus/SNAC.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tada_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // text sampling temperature (0 = greedy)
    uint64_t seed;
    int max_tokens; // max generation tokens (0 = 512 default)
    bool flash_attn;
    // FM solver params
    int num_fm_steps;   // ODE steps (0 = 10 default)
    float acoustic_cfg; // CFG scale for acoustic features (1.0 = no CFG)
    float noise_temp;   // noise temperature (0.0 = deterministic)
};

struct tada_context;

struct tada_context_params tada_context_default_params(void);

struct tada_context* tada_init_from_file(const char* path_model, struct tada_context_params params);

// Set the companion codec GGUF path (required before synthesize).
int tada_set_codec_path(struct tada_context* ctx, const char* path);

// Load a pre-computed voice prompt from a GGUF file containing
// prompt_token_values (N, 512) and prompt_token_positions (N,).
// This bypasses the Encoder and provides voice conditioning directly.
int tada_load_prompt(struct tada_context* ctx, const char* path);

// Set generation seed for reproducibility.
void tada_set_seed(struct tada_context* ctx, uint64_t seed);

// Set text sampling temperature.
void tada_set_temperature(struct tada_context* ctx, float temp);

// Synthesize text to 24 kHz mono PCM. Returns heap-allocated float array;
// caller must free with tada_pcm_free(). *out_n_samples is set to the
// number of float samples. Returns NULL on failure.
float* tada_synthesize(struct tada_context* ctx, const char* text, int* out_n_samples);

void tada_pcm_free(float* pcm);
void tada_free(struct tada_context* ctx);

// Test: run a single FM step with given inputs, return velocity.
// noisy_z: float[528], t_emb_sin: float[256], cond: float[3072], velocity_out: float[528]
void tada_test_fm_step(struct tada_context* ctx, const float* noisy_z, const float* t_emb_sin, const float* cond,
                       float* velocity_out);

#ifdef __cplusplus
}
#endif
