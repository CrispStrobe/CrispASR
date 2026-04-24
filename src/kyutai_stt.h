// kyutai_stt.h — C API for Kyutai STT (stt-1b-en_fr, stt-2.6b-en).
//
// Architecture: Mimi audio codec encoder (SEANet CNN + transformer + RVQ)
//             + Causal transformer LM (2048d, 16L, RoPE, SwiGLU, RMSNorm)
//
// Audio flow: 24kHz PCM → SEANet encoder → transformer → downsample →
//             RVQ (32 codebooks) → LM → text tokens → SentencePiece decode

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kyutai_stt_context;

struct kyutai_stt_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
    bool use_gpu;      // false => force CPU backend
    float temperature; // 0 = greedy argmax, >0 = softmax sampling
};

struct kyutai_stt_context_params kyutai_stt_context_default_params(void);

struct kyutai_stt_context* kyutai_stt_init_from_file(const char* path_model, struct kyutai_stt_context_params params);

void kyutai_stt_free(struct kyutai_stt_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Internally resamples to 24 kHz for Mimi codec.
// Returns malloc'd UTF-8 string, caller frees with free().
char* kyutai_stt_transcribe(struct kyutai_stt_context* ctx, const float* samples, int n_samples);

// Token text lookup.
const char* kyutai_stt_token_text(struct kyutai_stt_context* ctx, int id);

#ifdef __cplusplus
}
#endif
