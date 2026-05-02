#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gemma4_e2b_context;

struct gemma4_e2b_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
};

struct gemma4_e2b_context_params gemma4_e2b_context_default_params(void);

// Initialize from a GGUF file. Returns nullptr on failure.
struct gemma4_e2b_context* gemma4_e2b_init_from_file(const char* path_model, struct gemma4_e2b_context_params params);

// Transcribe PCM audio (16kHz mono float32). Returns malloc'd UTF-8 string (caller frees).
char* gemma4_e2b_transcribe(struct gemma4_e2b_context* ctx, const float* pcm, int n_samples);

// Variant that additionally returns per-emitted-token ids + softmax probs.
// Free with gemma4_e2b_result_free.
struct gemma4_e2b_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct gemma4_e2b_result* gemma4_e2b_transcribe_with_probs(struct gemma4_e2b_context* ctx, const float* pcm,
                                                           int n_samples);
void gemma4_e2b_result_free(struct gemma4_e2b_result* r);

// Single-id detokenize. Returns the raw vocab piece (no Gemma byte-level
// decode applied — caller may need to massage). Empty for out-of-range.
const char* gemma4_e2b_token_text(struct gemma4_e2b_context* ctx, int id);

// Free context and all associated memory.
void gemma4_e2b_free(struct gemma4_e2b_context* ctx);

// Set thread count after init.
void gemma4_e2b_set_n_threads(struct gemma4_e2b_context* ctx, int n_threads);

// ── Stage hooks for crispasr-diff ───────────────────────────────────────────
//
// These mirror the parakeet/voxtral/canary stage API: each one runs a
// well-defined slice of the forward pass and returns a malloc'd float
// buffer the caller frees. Used by examples/cli/crispasr_diff_main.cpp
// to compare each architectural boundary against tools/dump_reference.py
// activations.

// Compute mel spectrogram. Returns [n_mels, T_mel] in row-major.
float* gemma4_e2b_compute_mel(struct gemma4_e2b_context* ctx, const float* pcm, int n_samples, int* out_n_mels,
                              int* out_T_mel);

// Run audio encoder. Returns [d_model, T_enc] in row-major (matches Python
// reference's [T_enc, d_model] when transposed by the diff harness).
// `mel` is [n_mels, T_mel] from gemma4_e2b_compute_mel.
float* gemma4_e2b_run_encoder(struct gemma4_e2b_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                              int* out_d_model);

#ifdef __cplusplus
}
#endif
