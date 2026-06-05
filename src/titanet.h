// titanet.h — TitaNet-Large speaker embedding extraction.
//
// Architecture: NVIDIA TitaNet-Large (CC-BY-4.0, 23M params)
//   Input: 16 kHz mono PCM → 80-bin mel spectrogram (Hann, n_fft=512)
//   Encoder: 5 Jasper-style blocks with depthwise separable convolutions + SE
//     Block 0 (prolog):  DW-Conv(80, k=3) + PW-Conv(80→1024) + BN + SE
//     Block 1:           3× DW-Sep(1024, k=7)  + SE + residual
//     Block 2:           3× DW-Sep(1024, k=11) + SE + residual
//     Block 3:           3× DW-Sep(1024, k=15) + SE + residual
//     Block 4 (epilog):  DW-Conv(1024, k=1) + PW-Conv(1024→3072) + BN + SE
//   Decoder: ASP(3072 → 6144) + Linear(6144 → 192) + L2-normalize
//   Output: 192-d L2-normalized speaker embedding
//
// GGUF: titanet-large.gguf (~45 MB)

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct titanet_context;

// Initialize TitaNet from a GGUF model file. Returns NULL on failure.
struct titanet_context* titanet_init(const char* model_path, int n_threads);

// Free all resources.
void titanet_free(struct titanet_context* ctx);

// Extract a 192-d L2-normalized speaker embedding from 16 kHz mono PCM.
// Returns the number of floats written to `out` (192 on success, 0 on error).
// `out` must point to at least 192 floats.
int titanet_embed(struct titanet_context* ctx, const float* pcm_16k, int n_samples, float* out);

// Cosine similarity between two embeddings of dimension `dim`.
// For L2-normalized vectors this is just the dot product.
float titanet_cosine_sim(const float* a, const float* b, int dim);

#ifdef __cplusplus
}
#endif
