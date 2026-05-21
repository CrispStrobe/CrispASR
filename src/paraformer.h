// src/paraformer.h — FunASR Paraformer (NAR ASR) runtime.
//
// Paraformer is a non-autoregressive ASR model:
//   Kaldi-fbank + LFR + CMVN
//   → 50-block SANM encoder (1 entry block 560→512 + 49 main blocks)
//   → CIF predictor (Conv1d + Linear → sigmoid → accumulation)
//   → 16-block NAR decoder (FSMN self-attn + cross-attn + FFN)
//   → argmax over vocab (8404 characters)
//
// The encoder reuses core_sanm::build_block(). The decoder is new — each
// block has three sub-layers:
//   1. norm1 → FSMN depthwise conv (no Q/K/V self-attention)
//   2. norm2 → cross-attention (Q from decoder, fused K+V from encoder)
//   3. norm3 → FFN (w_1 → LayerNorm → w_2, note the internal LN)

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct paraformer_context;

struct paraformer_context_params {
    int n_threads;
    int verbosity;       // 0 = silent, 1 = normal, 2 = debug
    bool flash_attn;
};

paraformer_context_params paraformer_context_default_params();

paraformer_context* paraformer_init_from_file(const char* path, paraformer_context_params params);
void paraformer_free(paraformer_context* ctx);

// Transcribe audio (16 kHz mono PCM). Returns a malloc'd string; caller frees.
char* paraformer_transcribe(paraformer_context* ctx, const float* samples, int n_samples);

// Stage-capture API for the diff harness.
float* paraformer_extract_stage(paraformer_context* ctx, const float* samples, int n_samples,
                                const char* stage_name, int* n_out);

#ifdef __cplusplus
}
#endif
