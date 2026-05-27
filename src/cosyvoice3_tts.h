#pragma once

// CosyVoice3-0.5B-2512 TTS — public C ABI.
//
// FunAudioLLM/Fun-CosyVoice3-0.5B-2512: multilingual TTS (9 langs +
// 18 Chinese dialects), Apache-2.0, 24 kHz output, zero-shot voice
// cloning. Pipeline:
//
//   text  → CosyVoice3LM (Qwen2-0.5B + speech_embd + speech_lm_head)
//         → speech tokens ∈ [0, 6561)
//        → Flow (DiT + CausalConditionalCFM)
//         → mel @ T_mel = 2 · T_tok
//        → CausalHiFTGenerator (HiFi-GAN + iSTFT)
//         → 24 kHz PCM
//
// Phase 2 (this header at first cut): the LLM side only. The flow and
// hift sub-models load through separate calls landed in later phases.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cosyvoice3_tts_context;

struct cosyvoice3_tts_context_params {
    int n_threads;
    int verbosity;       // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
    float temperature;   // 0 = greedy
    uint64_t seed;       // RNG seed; 0 = use default 42
    int max_tokens;      // upper bound on AR decode steps; 0 = use built-in default (1500)
    int ras_top_k;       // RAS sampler top-k (default 25; 0 → use default)
    float ras_top_p;     // RAS sampler top-p (default 0.8f; 0 → use default)
    int ras_win_size;    // RAS repetition window (default 10; 0 → use default)
    float ras_tau_r;     // RAS repetition threshold (default 0.1f)
};

struct cosyvoice3_tts_context_params cosyvoice3_tts_context_default_params(void);

// Initialise from the LLM GGUF file
// (e.g. cosyvoice3-llm-f16.gguf from `cstr/cosyvoice3-0.5b-2512-GGUF`).
// Returns nullptr on failure.
struct cosyvoice3_tts_context* cosyvoice3_tts_init_from_file(const char* path_model,
                                                             struct cosyvoice3_tts_context_params params);

void cosyvoice3_tts_free(struct cosyvoice3_tts_context* ctx);

void cosyvoice3_tts_set_n_threads(struct cosyvoice3_tts_context* ctx, int n_threads);
void cosyvoice3_tts_set_seed(struct cosyvoice3_tts_context* ctx, uint64_t seed);
void cosyvoice3_tts_set_temperature(struct cosyvoice3_tts_context* ctx, float temperature);

// Read out LLM hparams for the diff harness (each pointer may be NULL).
// d_model, n_layers, n_heads, n_kv_heads, head_dim, text_vocab,
// speech_vocab (the head dimension, 6761) and speech_codebook (the AR
// emit range, 6561).
int cosyvoice3_tts_get_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* d_model, uint32_t* n_layers,
                               uint32_t* n_heads, uint32_t* n_kv_heads, uint32_t* head_dim, uint32_t* text_vocab,
                               uint32_t* speech_vocab, uint32_t* speech_codebook);

// Build a text-mode input embedding (text_token_id -> token_embd[id])
// for `n_tokens` ids. Returns a malloc'd float buffer [n_tokens, d_model]
// in row-major order. Caller frees with free(). Useful for the diff
// harness and for the higher-level synth path.
float* cosyvoice3_tts_embed_text(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens);

// Build a speech-mode input embedding (speech_token_id ->
// speech_embd[id]) for `n_tokens` ids. Same return contract.
float* cosyvoice3_tts_embed_speech(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens);

// Run the 24-layer Qwen2 LM on caller-supplied [n_tokens, d_model]
// row-major float32 embeddings. Writes K/V into the persistent KV
// cache at positions [n_past, n_past + n_tokens). Returns the
// last-position logits over the speech codebook head (6761 entries).
// Caller frees with free(). Returns nullptr on failure.
//
// This is the diff-harness entry: feed in PyTorch-prebuilt embeds and
// expect bit-equivalent logits at the tail.
float* cosyvoice3_tts_prefill_with_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds, int n_tokens,
                                          int n_past);

// Single-step speech-token forward: speech_embd[id] -> Qwen2 ->
// speech_lm_head -> logits[6761]. Reads/writes the persistent KV cache
// at slot n_past. Caller frees with free().
float* cosyvoice3_tts_step_speech(struct cosyvoice3_tts_context* ctx, int32_t speech_id, int n_past);

// Reset the persistent KV cache so the next prefill starts from n_past=0.
void cosyvoice3_tts_reset_kv(struct cosyvoice3_tts_context* ctx);

// Diff-harness stage extractor. Returns malloc'd float[*out_n].
// Phase 2 supports:
//   "lm_step0_logits"   — single-step logits after prefilling on
//                          caller-supplied embeds; pass embeds via the
//                          `embeds_in` buffer of length n_tokens*d_model
//   "lm_token_embd"     — token_embd[ids] lookup verification
//   "lm_speech_embd"    — speech_embd[ids] lookup verification
// Other stages (flow / hift) land in their phases.
float* cosyvoice3_tts_extract_stage(struct cosyvoice3_tts_context* ctx, const char* stage_name, const int32_t* ids,
                                    int n_ids, const float* embeds_in, int n_embed_tokens, int* out_n);

#ifdef __cplusplus
}
#endif
