// granite_speech.h — public C API for ibm-granite/granite-4.0-1b-speech
//
// Speech-LLM: Conformer encoder + BLIP-2 Q-Former projector + Granite 1B LLM.
// Models loaded from GGUF files produced by:
//   `python models/convert-granite-speech-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Architecture:
//   Encoder:   16-layer Conformer (1024 dim, 8 heads, Macaron FFN, depthwise conv,
//              relative position embedding, batch norm). Input: 160-dim (80 mels × 2
//              stacked frames).
//   Projector: 2-layer BLIP-2 Q-Former with 3 learned query tokens. Cross-attends
//              to encoder output, producing 3 audio tokens for the LLM.
//   LLM:       Granite 4.0-1B (40 layers, 2048 dim, GQA 16/4, SwiGLU, RoPE θ=10000,
//              μP multipliers: embedding=12, attention=1/128, residual=0.22, logits=8).
//   Languages: en, fr, de, es, pt, ja

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct granite_speech_context;

struct granite_speech_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
};

struct granite_speech_context_params granite_speech_context_default_params(void);

struct granite_speech_context * granite_speech_init_from_file(
    const char * path_model, struct granite_speech_context_params params);

void granite_speech_free(struct granite_speech_context * ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char * granite_speech_transcribe(struct granite_speech_context * ctx,
                                 const float * samples, int n_samples);

// Pipeline building blocks for differential testing:

// Compute 80-bin log-mel spectrogram. Returns malloc'd (80, T) F32 row-major.
float * granite_speech_compute_mel(struct granite_speech_context * ctx,
                                   const float * samples, int n_samples,
                                   int * out_n_mels, int * out_T_mel);

// Run encoder on mel features. Returns malloc'd (N, 1024) F32.
float * granite_speech_run_encoder(struct granite_speech_context * ctx,
                                   const float * mel, int n_mels, int T_mel,
                                   int * out_N, int * out_dim);

// Run Q-Former projector on encoder output. Returns malloc'd (n_query, llm_dim) F32.
float * granite_speech_run_projector(struct granite_speech_context * ctx,
                                     const float * encoder_out, int enc_len, int enc_dim,
                                     int * out_N, int * out_dim);

// KV cache for LLM
bool granite_speech_kv_init(struct granite_speech_context * ctx, int max_ctx);
void granite_speech_kv_reset(struct granite_speech_context * ctx);

// Run LLM forward with KV cache. Returns last-token logits (vocab,) F32.
float * granite_speech_run_llm_kv(struct granite_speech_context * ctx,
                                   const float * inputs_embeds,
                                   int n_tokens, int n_past,
                                   int * out_n_tokens, int * out_vocab_size);

// Embed token IDs. Returns malloc'd (n_tokens, d_model) F32.
float * granite_speech_embed_tokens(struct granite_speech_context * ctx,
                                     const int32_t * input_ids, int n_tokens);

// Get text for a token ID (GPT-2 BPE vocab). Returns empty string if OOB.
const char * granite_speech_token_text(struct granite_speech_context * ctx, int id);

// Decode an array of token IDs to UTF-8 text. Caller frees with free().
char * granite_speech_decode_tokens(struct granite_speech_context * ctx,
                                     const int32_t * ids, int n_ids);

// Tokenize a UTF-8 text string into vocab IDs using the granite GPT-2-style
// byte-level BPE encoder. The result is a malloc'd int32 array of length
// `*out_n`; the caller owns it and must free() it. Returns NULL on failure.
//
// Requires the GGUF to include `tokenizer.ggml.merges` (the newer
// models/convert-granite-speech-to-gguf.py writes them; older GGUFs only
// had `tokenizer.ggml.tokens`). When merges are missing the helper still
// works for text that maps to single vocab entries, but it can't merge
// sub-words and falls back to per-byte tokens for unknown sequences.
int32_t * granite_speech_tokenize(struct granite_speech_context * ctx,
                                  const char * text, int * out_n);

// Per-model control token ids read from the GGUF metadata at load time.
// These vary between releases (granite-4.0 uses the 100k-vocab GPT-NeoX
// table, granite-3.x uses the smaller 49160-token Granite tokenizer) so
// callers must query them at runtime instead of hardcoding.
int granite_speech_audio_token_id(struct granite_speech_context * ctx);
int granite_speech_eos_token_id  (struct granite_speech_context * ctx);
int granite_speech_vocab_size    (struct granite_speech_context * ctx);

#ifdef __cplusplus
}
#endif
