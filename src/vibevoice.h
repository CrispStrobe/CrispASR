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

#ifdef __cplusplus
}
#endif
