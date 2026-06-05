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
    int tts_steps;   // DPM-Solver++ inference steps (default 20, min 4)
    uint32_t seed;   // RNG seed for TTS diffusion noise (0 = env/default)
    bool flash_attn; // PLAN #89 plumbing — σ-VAE encoder + Qwen2.5
                     // talker SA blocks.
};

struct vibevoice_context_params vibevoice_context_default_params(void);

struct vibevoice_context* vibevoice_init_from_file(const char* path_model, struct vibevoice_context_params params);

void vibevoice_free(struct vibevoice_context* ctx);

// Runtime setter for the DPM-Solver++ inference step count
// (default 20). Read on every vibevoice_synthesize call (line ~3422
// in vibevoice.cpp), so post-init mutation is safe. Clamps to
// [4, 100]: below 4 the schedule is degenerate, above 100 you're
// burning latency for inaudible quality gain.
void vibevoice_set_tts_steps(struct vibevoice_context* ctx, int steps);
void vibevoice_set_seed(struct vibevoice_context* ctx, uint32_t seed);

// Transcribe raw 24kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* vibevoice_transcribe(struct vibevoice_context* ctx, const float* samples, int n_samples);

// Variant that additionally returns per-emitted-token ids and softmax
// probabilities. Free with vibevoice_result_free.
struct vibevoice_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct vibevoice_result* vibevoice_transcribe_with_probs(struct vibevoice_context* ctx, const float* samples,
                                                         int n_samples);
void vibevoice_result_free(struct vibevoice_result* r);

// Token-id → vocab piece (raw, with Qwen2/GPT-2 byte-level BPE markers
// like Ġ / Ċ — caller may need to decode). Returns empty string for
// out-of-range ids.
const char* vibevoice_token_text(struct vibevoice_context* ctx, int id);

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

// ── TTS API (requires GGUF converted with --include-decoder) ─────────────────

// Synthesize speech from text. Returns malloc'd 24 kHz mono PCM float array.
// n_samples is set to the number of output samples. Caller frees with free().
// Returns NULL if the model lacks decoder tensors (vibevoice.has_decoder=0).
float* vibevoice_synthesize(struct vibevoice_context* ctx, const char* text, int* out_n_samples);

// Load a voice prompt GGUF for TTS. Returns 0 on success.
// The voice prompt pre-fills KV caches with speaker identity.
int vibevoice_load_voice(struct vibevoice_context* ctx, const char* voice_path);

#ifdef __cplusplus
}
#endif
