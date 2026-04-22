// firered_asr.h — C API for FireRedASR2-AED (FireRedTeam/FireRedASR2-AED).
//
// Architecture: Conformer encoder (16L, d=1280, 20 heads, relative PE, macaron FFN,
//               depthwise conv k=33, BatchNorm, Swish+GLU)
//             + Transformer decoder (16L, d=1280, cross-attention, GELU FFN)
//             + CTC head (for optional CTC decoding)
//             + Mixed BPE tokenizer (8667 tokens: Chinese chars + English BPE)
//
// Audio flow: 16kHz PCM → 80-dim log-mel → Conv2d subsampling (4x) →
//             Conformer encoder → Transformer decoder (beam search) → text

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct firered_asr_context;

struct firered_asr_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
};

struct firered_asr_context_params firered_asr_context_default_params(void);

struct firered_asr_context* firered_asr_init_from_file(const char* path_model,
                                                       struct firered_asr_context_params params);

void firered_asr_free(struct firered_asr_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* firered_asr_transcribe(struct firered_asr_context* ctx, const float* samples, int n_samples);

// Token text lookup.
const char* firered_asr_token_text(struct firered_asr_context* ctx, int id);

#ifdef __cplusplus
}
#endif
