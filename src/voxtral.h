// voxtral.h — public C API for Mistral Voxtral-Mini-3B-2507 ggml runtime
//
// Audio-LLM combining a Whisper-large-v3 encoder + 4-frame stack projector +
// Llama 3 (Mistral) 3B LLM with audio-token injection. Models are loaded from
// GGUF files produced by:
//   `python models/convert-voxtral-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Architecture summary (verified against config.json + safetensors index):
//
//   Audio encoder: 32-layer Whisper-large-v3 encoder (d=1280, 20 heads,
//                  head_dim=64, FFN=5120, 128 mels, learned absolute pos
//                  embed, 2× Conv1D front-end, K-proj has no bias)
//   Projector:     stack-4-frames (1280×4=5120) → Linear(5120→3072) → GELU
//                  → Linear(3072→3072). 4× temporal downsampling: 50 fps
//                  Whisper output → 12.5 fps audio embeddings.
//   LLM:           30-layer Llama 3 / Mistral (d=3072, GQA 32/8, head_dim=128,
//                  FFN=8192, SwiGLU, RMSNorm, RoPE θ=1e8, NO Q/K-norm,
//                  NO biases, vocab=131072, max_pos=131072)
//   Tokenizer:     Mistral Tekken (tiktoken-style rank BPE, 150k vocab,
//                  1000 special tokens at ranks 0..999)
//   Audio inject:  audio_token_id=24 placeholder in the [INST] prompt; the
//                  LLM input embeddings at those positions get replaced with
//                  the projector output frames.
//
// This is structurally simpler than Qwen3-ASR (no Q-norm/K-norm, no chunked
// audio encoder) and should reuse most of the qwen3_asr.cpp infrastructure
// with parameter swaps.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct voxtral_context;

struct voxtral_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
};

struct voxtral_context_params voxtral_context_default_params(void);

// Load model from GGUF.
struct voxtral_context * voxtral_init_from_file(const char * path_model,
                                                struct voxtral_context_params params);

void voxtral_free(struct voxtral_context * ctx);

// Get the (raw bytes) text for a token ID. Tekken vocab entries are raw
// byte sequences, not byte-encoded unicode like GPT-2 BPE. Returns the
// length in *out_len. The pointer is owned by the context — do NOT free.
const uint8_t * voxtral_token_text(struct voxtral_context * ctx, int id, int * out_len);

// Tokenize a UTF-8 text string with the embedded Tekken tokenizer.
// Returns a malloc'd int32 array of token IDs; caller frees with free().
int32_t * voxtral_tokenize(struct voxtral_context * ctx,
                           const char * text, int * out_n_tokens);

// ---- Stage-1 helpers exposed for differential testing ----------------------
//
// These mirror the qwen3_asr_* test helpers — feed precomputed inputs and
// pull intermediate activations back out for diffing against PyTorch dumps.

// Run the Llama 3 LLM forward (text-only, no audio injection, no KV cache).
// Used for the LLM smoke test against models/voxtral-llm-dump.py.
// Returns a malloc'd float buffer of shape (n_tokens, vocab_size=131072)
// row-major. Caller frees with free().
float * voxtral_run_llm(struct voxtral_context * ctx,
                        const int32_t * input_ids, int n_tokens,
                        int * out_n_tokens, int * out_vocab_size);

#ifdef __cplusplus
}
#endif
