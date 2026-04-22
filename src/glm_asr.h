// glm_asr.h — C API for GLM-ASR-Nano (zai-org/GLM-ASR-Nano-2512).
//
// Architecture: Whisper-style encoder (1280d, 32L, partial RoPE)
//             + 4-frame-stack projector (5120→4096→2048)
//             + Llama-style LLM (2048d, 28L, GQA 16/4, SwiGLU)
//
// Very similar to voxtral — same building blocks, different sizes.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct glm_asr_context;

struct glm_asr_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
};

struct glm_asr_context_params glm_asr_context_default_params(void);

struct glm_asr_context* glm_asr_init_from_file(const char* path_model, struct glm_asr_context_params params);

void glm_asr_free(struct glm_asr_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* glm_asr_transcribe(struct glm_asr_context* ctx, const float* samples, int n_samples);

// Pipeline building blocks for differential testing:

// Compute mel spectrogram from raw PCM. Returns malloc'd (n_mels, T_mel) F32.
float* glm_asr_compute_mel(struct glm_asr_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                           int* out_T_mel);

// Run audio encoder on mel. Returns malloc'd (N, proj_dim) F32.
float* glm_asr_run_encoder(struct glm_asr_context* ctx, const float* mel, int n_mels, int T_mel, int* out_N,
                           int* out_dim);

// Embed token IDs. Returns malloc'd (n_tokens, d_model) F32.
float* glm_asr_embed_tokens(struct glm_asr_context* ctx, const int32_t* ids, int n_ids);

// KV cache management.
bool glm_asr_kv_init(struct glm_asr_context* ctx, int max_ctx);
void glm_asr_kv_reset(struct glm_asr_context* ctx);

// Run LLM forward with KV cache. Returns malloc'd logits (vocab_size,) F32.
float* glm_asr_run_llm_kv(struct glm_asr_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                          int* out_n_tokens, int* out_vocab_size);

// Token text lookup.
const char* glm_asr_token_text(struct glm_asr_context* ctx, int id);

// Tokenize text. Returns malloc'd array of int32_t, caller frees with free().
int32_t* glm_asr_tokenize(struct glm_asr_context* ctx, const char* text, int* out_n_tokens);

#ifdef __cplusplus
}
#endif
