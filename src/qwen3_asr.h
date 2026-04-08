// qwen3_asr.h — public C API for Qwen/Qwen3-ASR-0.6B ggml runtime
//
// Multilingual speech recognition (30 languages) using a 2D-conv subsampler
// + 18-layer Whisper-style encoder + 28-layer Qwen3 0.6B LLM with audio-token
// injection. Models are loaded from GGUF files produced by:
//   `python models/convert-qwen3-asr-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Reference: github.com/predict-woo/qwen3-asr.cpp (MIT) — used for
// architecture discovery only; no source vendored.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct qwen3_asr_context;

struct qwen3_asr_context_params {
    int  n_threads;
    int  verbosity;     // 0=silent 1=normal 2=verbose
};

struct qwen3_asr_context_params qwen3_asr_context_default_params(void);

// Load model from GGUF.
struct qwen3_asr_context * qwen3_asr_init_from_file(const char * path_model,
                                                    struct qwen3_asr_context_params params);

void qwen3_asr_free(struct qwen3_asr_context * ctx);

// Transcribe raw 16 kHz mono PCM. Returns malloc'd UTF-8 string (caller owns).
char * qwen3_asr_transcribe(struct qwen3_asr_context * ctx,
                            const float * samples, int n_samples);

// ---- Stage-1 helpers exposed for differential testing ----------------------
//
// These let a test driver feed pre-computed mel features (matching the
// PyTorch reference processor) and pull intermediate activations back out,
// so we can diff against ground-truth .npy files dumped by
// models/qwen3-asr-reference-dump.py.

// Run the conv front-end only on a (n_mels, T_mel) mel spectrogram.
// Output is a malloc'd float buffer of shape (num_chunks, T_chunk_out, 896)
// in row-major order. *out_n_chunks / *out_T_chunk_out / *out_d filled in.
// Caller frees with free().
float * qwen3_asr_run_conv(struct qwen3_asr_context * ctx,
                           const float * mel_features,  // F32, shape (n_mels, T_mel)
                           int n_mels, int T_mel,
                           int * out_n_chunks,
                           int * out_T_chunk_out,
                           int * out_d);

// Run the full audio encoder (conv front-end + pos embed + 18 encoder layers
// + ln_post + proj1/GELU/proj2) on a (n_mels, T_mel) mel spectrogram.
// Output: malloc'd float buffer of shape (N_total, audio_proj_dim=1024) in
// row-major order. N_total = sum of valid post-CNN frames across all chunks.
// Caller frees with free().
float * qwen3_asr_run_encoder(struct qwen3_asr_context * ctx,
                              const float * mel_features,
                              int n_mels, int T_mel,
                              int * out_N_total,
                              int * out_proj_dim);

// Run the Qwen3 0.6B LLM forward (text-only, no audio injection, no KV cache).
// Useful for differential testing the LLM forward in isolation against the
// HF reference. Returns malloc'd float buffer of shape (n_tokens, vocab_size)
// row-major. Caller frees with free().
float * qwen3_asr_run_llm(struct qwen3_asr_context * ctx,
                          const int32_t * input_ids,
                          int n_tokens,
                          int * out_n_tokens,
                          int * out_vocab_size);

// Get the vocab string for a token ID. Returns "" if id is out of range.
// The string is in GPT-2 byte-encoded form (apply byte_decoder to recover
// raw UTF-8 bytes).
const char * qwen3_asr_token_text(struct qwen3_asr_context * ctx, int id);

// Tokenize a UTF-8 text string with the model's GPT-2 byte-level BPE.
// Handles Qwen3 special tokens (`<|im_start|>`, `<|audio_pad|>`, etc.) by
// looking them up in the vocab table directly before BPE-merging the
// surrounding plain text. Returns a malloc'd int32 array of token IDs;
// caller must free() the returned pointer. *out_n_tokens set on return.
int32_t * qwen3_asr_tokenize(struct qwen3_asr_context * ctx,
                             const char * text, int * out_n_tokens);

// Compute the log-mel spectrogram for raw 16 kHz mono PCM samples, matching
// HuggingFace WhisperFeatureExtractor (n_fft=400, hop=160, 128 mel bins,
// log10 + clip-to-max-8 + (x+4)/4 normalization). Requires that the model
// GGUF includes audio.mel_filters + audio.mel_window (added by the latest
// converter). Returns a malloc'd float buffer of shape (n_mels=128, T_mel)
// row-major. *out_T_mel set on return. Caller frees with free().
float * qwen3_asr_compute_mel(struct qwen3_asr_context * ctx,
                              const float * samples, int n_samples,
                              int * out_n_mels, int * out_T_mel);

// Look up token embeddings via the model's token_embd table. Returns a
// malloc'd float buffer of shape (n_tokens, d_model=1024) row-major. Caller
// frees with free().
float * qwen3_asr_embed_tokens(struct qwen3_asr_context * ctx,
                               const int32_t * input_ids, int n_tokens);

// Run the Qwen3 0.6B LLM forward starting from precomputed inputs_embeds
// instead of input_ids. Used by the audio-injection path: caller computes
// text embeddings via qwen3_asr_embed_tokens(), splices in audio frames at
// the audio_pad positions, then calls this. Returns logits (n_tokens, vocab).
float * qwen3_asr_run_llm_from_embeds(struct qwen3_asr_context * ctx,
                                      const float * inputs_embeds,
                                      int n_tokens,
                                      int * out_n_tokens,
                                      int * out_vocab_size);

// ---- KV-cache LLM API (Stage 5) ---------------------------------------------
//
// Persistent K/V cache to enable O(N) per-step incremental decoding instead
// of O(N) full forwards. Use case:
//
//   qwen3_asr_kv_init(ctx, max_ctx);     // once per session, allocates cache
//   qwen3_asr_kv_reset(ctx);             // start of each utterance
//   logits = qwen3_asr_run_llm_kv(ctx, prompt_embeds, T_prompt, 0);
//   // logits[(T_prompt-1)*vocab .. ] = next-token logits
//   while (...) {
//     embed_one_token(...);
//     logits = qwen3_asr_run_llm_kv(ctx, &one_embed, 1, n_used);
//     // n_used auto-advances by ctx after each call
//   }
//
// `n_past` is the number of tokens already in the cache. The graph writes
// the new tokens at positions [n_past, n_past+n_tokens) and reads keys/values
// from positions [0, n_past+n_tokens) for attention. The cache is held in a
// dedicated backend buffer of shape (head_dim, max_ctx, n_kv_heads, n_layers)
// for both K and V.
//
// Returns logits (n_tokens, vocab) row-major. Caller frees with free().

// Allocate the KV cache. Call once per context, before the first kv call.
bool qwen3_asr_kv_init(struct qwen3_asr_context * ctx, int max_ctx);

// Reset the cache pointer to 0 (does NOT zero memory). Call at the start of
// each new utterance.
void qwen3_asr_kv_reset(struct qwen3_asr_context * ctx);

// Run the LLM forward writing into the persistent KV cache.
float * qwen3_asr_run_llm_kv(struct qwen3_asr_context * ctx,
                             const float * inputs_embeds,
                             int n_tokens,
                             int n_past,
                             int * out_n_tokens,
                             int * out_vocab_size);

#ifdef __cplusplus
}
#endif
