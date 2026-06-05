// pocket_tts.h -- C API for Kyutai Pocket TTS (100M, 24 kHz).
//
// Architecture: continuous-latent AR at 12.5 Hz, NOT discrete tokens.
//   FlowLM backbone: causal transformer (1024D, 16H, 6L, RoPE, GELU)
//   -> consistency head (SimpleMLPAdaLN, 512D, 6 ResBlocks)
//   -> LSD one-step decode -> 32-dim continuous float vectors
//   -> Mimi VAE decoder (SEANet upsample conv + 2L transformer)
//   -> 24 kHz PCM
//
// Voice cloning: ref audio -> Mimi VAE encoder -> linear project
//   -> prepend to transformer KV cache.
//
// Text: SentencePiece (4000 vocab) + learned embedding (4001 x 1024).
//
// Key difference from every other TTS backend: no codebook, no sampling,
// no softmax. The AR loop emits continuous float vectors. The consistency
// head maps transformer hidden states to flow directions via AdaLN-MLP
// with timestep embedding, then one-step LSD decode produces the latent.
//
// Based on the CALM paper (arXiv:2509.06926).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pocket_tts_context;

struct pocket_tts_context_params {
    int n_threads;
    int verbosity;        // 0=silent 1=normal 2=verbose
    bool use_gpu;         // false => force CPU backend
    float temperature;    // noise temperature for LSD decode (default 0.7)
    uint64_t seed;        // RNG seed (0 = non-deterministic)
    int lsd_decode_steps; // LSD decode steps (default 1)
    float noise_clamp;    // max noise magnitude (0 = no clamp, default 3.0)
    float eos_threshold;  // EOS detection threshold (default 0.5)
    int max_audio_frames; // max AR frames (0 = auto from text length)
};

struct pocket_tts_context_params pocket_tts_context_default_params(void);

// Load from single GGUF (arch="pocket-tts").
struct pocket_tts_context* pocket_tts_init_from_file(const char* path_model, struct pocket_tts_context_params params);

void pocket_tts_free(struct pocket_tts_context* ctx);

// Synthesize text -> 24 kHz mono PCM.
// Returns malloc'd float array; caller frees with pocket_tts_pcm_free().
// *n_samples is set to the number of samples produced.
float* pocket_tts_synthesize(struct pocket_tts_context* ctx, const char* text, int* n_samples);

void pocket_tts_pcm_free(float* pcm);

// Set voice conditioning from a reference audio file (WAV, 16/24 kHz).
// The Mimi encoder weights must be present in the GGUF (--voice-cloning).
// Returns 0 on success.
int pocket_tts_set_voice(struct pocket_tts_context* ctx, const float* ref_pcm_24khz, int n_ref_samples);

// Clear voice conditioning (revert to unconditioned generation).
void pocket_tts_clear_voice(struct pocket_tts_context* ctx);

// Runtime parameter tweaks.
void pocket_tts_set_temperature(struct pocket_tts_context* ctx, float temp);
void pocket_tts_set_seed(struct pocket_tts_context* ctx, uint64_t seed);

// Query the sample rate (always 24000).
int pocket_tts_sample_rate(struct pocket_tts_context* ctx);

#ifdef __cplusplus
}
#endif
