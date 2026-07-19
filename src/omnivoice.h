#pragma once

// OmniVoice TTS public C ABI.
//
// k2-fsa/OmniVoice is a masked iterative TTS model built on a Qwen3-0.6B
// backbone. Unlike Qwen3-TTS (which uses autoregressive codebook-0 decode
// + a 15-codebook code predictor), OmniVoice predicts all 8 codebooks
// simultaneously via a single audio_heads projection, then iteratively
// unmasks tokens over N steps (SoundStorm-style).
//
// Architecture:
//   - Qwen3 LLM backbone: 28L, 1024 hidden, 16Q/8KV, head_dim 128
//   - audio_embeddings: Embedding(8*1025, 1024) with per-codebook offsets
//   - audio_heads: Linear(1024, 8*1025) — projects to 8 codebooks × 1025
//   - Generation: masked iterative with classifier-free guidance
//
// Two-GGUF runtime: the main model (LLM + audio layers) and a separate
// HiggsAudioV2 audio tokenizer (for encoding reference audio to codes
// and decoding generated codes to waveform).
//
// Supports k2-fsa/OmniVoice and finetunes (e.g. ModelsLab/omnivoice-singing).

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct omnivoice_context;

struct omnivoice_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    int num_steps;              // masked iterative steps; 0 = default (32)
    float guidance_scale;       // classifier-free guidance; 0 = default (1.0)
    float class_temperature;    // sampling temperature; 0 = greedy
    float position_temperature; // Gumbel noise for position selection
    float layer_penalty_factor; // penalty for higher codebook layers
    float t_shift;              // time-step shift; 0 = default (1.0)
    uint64_t seed;              // RNG seed; 0 = default 42
    bool flash_attn;
};

struct omnivoice_context_params omnivoice_context_default_params(void);

// Initialise from the main model GGUF file.
struct omnivoice_context* omnivoice_init_from_file(const char* path_model, struct omnivoice_context_params params);

// Point the runtime at the audio tokenizer GGUF.
// Required before the first synthesize call. Returns 0 on success.
int omnivoice_set_tokenizer_path(struct omnivoice_context* ctx, const char* path);

// Set a reference voice from a 24 kHz mono WAV plus its transcription.
// The audio is encoded via the HiggsAudioV2 tokenizer to produce
// reference audio codes for the voice-cloning prompt.
// Returns 0 on success.
int omnivoice_set_voice_prompt(struct omnivoice_context* ctx, const char* wav_path, const char* ref_text);

// Set the target language (e.g. "en", "zh", "ja"). Pass NULL for auto.
int omnivoice_set_language(struct omnivoice_context* ctx, const char* lang);

// Set a style instruction (for instruct-capable variants).
int omnivoice_set_instruct(struct omnivoice_context* ctx, const char* instruct);

// Set the speaking-rate multiplier used to scale the estimated target length:
// >1 = faster / shorter audio, <1 = slower / longer, 1.0 = model estimate.
// Useful when a reference voice yields an over- or under-long estimate.
int omnivoice_set_speed(struct omnivoice_context* ctx, float speed);

// Set the number of masked-iterative (diffusion) steps. Stage0 cost is
// num_steps × 2 backbone forwards, so this is the dominant speed/quality lever:
// lower = faster, fewer refinement passes. Default 32; ASR-clean down to ~16.
// Read live per synthesize (no reload). Values <1 are ignored. (#254)
int omnivoice_set_num_steps(struct omnivoice_context* ctx, int num_steps);

// Set the sampling seed for subsequent synthesis calls.
int omnivoice_set_seed(struct omnivoice_context* ctx, uint64_t seed);

// Run the masked iterative generation: text → 8-codebook audio codes.
// Returns malloc'd int32_t array of shape (n_codebooks * T) row-major
// [cb0_t0, cb1_t0, ..., cb7_t0, cb0_t1, ...].
// *out_n_codes is set to n_codebooks * T on success.
// Caller frees with omnivoice_codes_free.
int32_t* omnivoice_synthesize_codes(struct omnivoice_context* ctx, const char* text, int* out_n_codes);

void omnivoice_codes_free(int32_t* codes);

// Decode audio codes (from synthesize_codes) to 24 kHz mono PCM.
// Requires set_tokenizer_path to have been called first.
// codes is (n_codebooks * T) row-major. *out_n_samples is set on success.
// Caller frees with omnivoice_pcm_free.
float* omnivoice_decode_codes(struct omnivoice_context* ctx, const int32_t* codes, int n_codes, int* out_n_samples);

// End-to-end: text → 24 kHz mono PCM. Caller frees with omnivoice_pcm_free.
float* omnivoice_synthesize(struct omnivoice_context* ctx, const char* text, int* out_n_samples);

void omnivoice_pcm_free(float* pcm);

void omnivoice_free(struct omnivoice_context* ctx);

void omnivoice_sync(struct omnivoice_context* ctx);
void omnivoice_set_n_threads(struct omnivoice_context* ctx, int n_threads);

// Diff-harness entry: validate the voice-clone ENCODE path stage-by-stage against
// a Python reference archive (omnivoice-encode-ref.gguf). Requires the tokenizer
// loaded. Prints per-stage results to stderr. Returns 0 on success. (#254)
int omnivoice_encode_diff(struct omnivoice_context* ctx, const char* ref_gguf_path);

#ifdef __cplusplus
}
#endif
