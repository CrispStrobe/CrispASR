// firered_lid.h — FireRedLID (120-language identification).
//
// Architecture: same Conformer encoder as FireRedASR2-AED (16L, d=1280, 20 heads)
//             + 6-layer Transformer decoder (8 heads, cross-attention)
//             + 120-class language output
//
// The encoder is identical to FireRedASR — this file provides a simpler
// C API focused on language identification rather than full transcription.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct firered_lid_context;

struct firered_lid_context* firered_lid_init(const char* model_path, int n_threads);
void firered_lid_free(struct firered_lid_context* ctx);

// Detect language from 16kHz mono PCM audio.
// Returns ISO 639-1 language code (static string, valid until ctx is freed).
// Sets *confidence to the softmax probability if non-NULL.
// Returns NULL on failure.
const char* firered_lid_detect(struct firered_lid_context* ctx, const float* samples, int n_samples, float* confidence);

#ifdef __cplusplus
}
#endif
