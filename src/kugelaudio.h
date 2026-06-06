// kugelaudio.h — KugelAudio-0-Open TTS runtime.
//
// Architecture: Qwen2.5-7B language model (28L, 3584d, GQA 28/4)
//   + 4-layer DiT diffusion head (AdaLN, SwiGLU, v-prediction)
//   + acoustic VAE decoder (ConvNeXt, 6-stage 3200x upsample)
//   → 24 kHz mono PCM output.
//
// 23 languages (en, de, fr, es, it, pt, nl, pl, ru, uk, cs, ro, hu,
// sv, da, fi, no, el, bg, sk, hr, sr, tr). MIT license.
// ~18.7 GB F16, ~5-6 GB Q4_K. Requires GPU (19 GB VRAM) for real-time.
//
// Inference pipeline:
//   1. Tokenize text with Qwen2.5 BPE
//   2. Embed text tokens + optional pre-encoded voice embeddings
//   3. AR decode with constrained token set {start, end, diffusion, eos}
//   4. On speech_diffusion token: run DPM-Solver++ SDE (20 steps)
//      with classifier-free guidance (cfg_scale=3.0)
//   5. Unscale latent → acoustic VAE decoder → 24 kHz PCM
//   6. Concatenate audio chunks until speech_end/eos

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kugelaudio_context;

struct kugelaudio_context_params {
    int n_threads;
    int max_new_tokens; // max AR tokens to generate (default 2048)
    int verbosity;      // 0=silent 1=normal 2=verbose
    bool use_gpu;
    int tts_steps;   // DPM-Solver++ inference steps (default 20, min 4)
    float cfg_scale; // classifier-free guidance scale (default 3.0)
    uint32_t seed;   // RNG seed for diffusion noise (0 = non-deterministic)
    bool flash_attn; // use flash attention if available
};

struct kugelaudio_context_params kugelaudio_context_default_params(void);

struct kugelaudio_context* kugelaudio_init_from_file(const char* path_model, struct kugelaudio_context_params params);

void kugelaudio_free(struct kugelaudio_context* ctx);

// Runtime setters
void kugelaudio_set_tts_steps(struct kugelaudio_context* ctx, int steps);
void kugelaudio_set_cfg_scale(struct kugelaudio_context* ctx, float scale);
void kugelaudio_set_seed(struct kugelaudio_context* ctx, uint32_t seed);

// Synthesize speech from text. Returns malloc'd 24 kHz mono PCM float array.
// n_samples is set to the number of output samples. Caller frees with free().
// Returns NULL on failure.
float* kugelaudio_synthesize(struct kugelaudio_context* ctx, const char* text, int* out_n_samples);

// Load a pre-encoded voice GGUF for speaker identity.
// Returns 0 on success, -1 on failure.
int kugelaudio_load_voice(struct kugelaudio_context* ctx, const char* voice_path);

// ── Stage-level API for differential testing ────────────────────────────────

// Run the diffusion head for one step. Returns predicted output [vae_dim].
// Caller frees with free(). Returns NULL on failure.
float* kugelaudio_run_diffusion_step(struct kugelaudio_context* ctx, const float* noisy_latent, int vae_dim,
                                     int timestep, const float* condition, int d_lm, int* out_dim);

// Run the acoustic decoder on a latent. Returns PCM audio.
// latent: [vae_dim] (single frame). Returns NULL on failure.
float* kugelaudio_run_acoustic_decoder(struct kugelaudio_context* ctx, const float* latent, int vae_dim,
                                       int* out_n_samples);

#ifdef __cplusplus
}
#endif
