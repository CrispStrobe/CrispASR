// omniasr.h — Facebook OmniASR (wav2vec2 encoder + CTC or LLM decoder).
//
// Two model types (auto-detected from GGUF metadata):
//   * CTC: encoder + linear CTC head (non-autoregressive, fastest)
//   * LLM: encoder + 12-layer LLaMA decoder (autoregressive, best quality)
//
// Input: raw 16kHz mono PCM. Output: text via SentencePiece tokenizer.
// 300M–7B params, 1600+ languages, Apache-2.0.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct omniasr_context;

struct omniasr_context_params {
    int n_threads;
    int max_new_tokens;   // LLM decoder generation cap
    int verbosity;        // 0=silent 1=normal 2=verbose
    const char* language; // ISO 639-3 lang code for LLM (e.g. "eng_Latn"), NULL for auto
    bool use_gpu;         // false => force CPU backend
    float temperature;    // 0 = greedy argmax, >0 = softmax sampling
};

struct omniasr_context_params omniasr_context_default_params(void);

struct omniasr_context* omniasr_init_from_file(const char* path_model, struct omniasr_context_params params);

void omniasr_free(struct omniasr_context* ctx);

// Transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* omniasr_transcribe(struct omniasr_context* ctx, const float* samples, int n_samples);

// Variant for the LLM model variant: returns text plus per-token ids and
// softmax probabilities (CPU-side argmax + softmax). Returns nullptr for the
// CTC variant — use omniasr_transcribe instead. Free with omniasr_result_free.
struct omniasr_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct omniasr_result* omniasr_transcribe_with_probs(struct omniasr_context* ctx, const float* samples, int n_samples);

void omniasr_result_free(struct omniasr_result* r);

// Token text lookup (id → vocab piece, no detokenisation).
const char* omniasr_token_text(struct omniasr_context* ctx, int id);

// Sticky per-call seed for the multinomial sampler. 0 (default) = derive
// deterministically from the encoder output. Non-zero values let
// best-of-N callers draw independent samples from the same audio.
void omniasr_set_seed(struct omniasr_context* ctx, uint64_t seed);

#ifdef __cplusplus
}
#endif
