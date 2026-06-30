// moss_transcribe.h — public C API for MOSS-Transcribe-preview-2B ggml runtime
//
// Speech recognition (ASR) using the stock Qwen3-Omni-MoE audio encoder +
// a Gated-MLP adapter + a Qwen3-1.7B LLM with audio-token injection. Models
// are loaded from GGUF files produced by:
//   `python models/convert-moss-transcribe-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Architecture: OpenMOSS-Team/MOSS-Transcribe-preview-2B (Apache-2.0)
//   Audio encoder (qwen3_omni_moe_audio_encoder): 128-mel → 3×Conv2d(stride 2,
//     480ch) → conv_out(7680→1280, no bias) → +sinusoidal pos → 32 pre-LN
//     Whisper-style layers (1280d, 20 heads, FFN 5120, WINDOWED attention) →
//     ln_post → proj1(1280→1280)+gelu → proj2(1280→2048).  output_dim 2048.
//   Audio adapter: MossGatedMLP(2048→8192→2048, SiLU gate, no bias).
//   LM: Qwen3-1.7B (28L, 2048d, 16Q/8KV, head_dim 128, QK-norm, SwiGLU 6144,
//     RoPE θ=1e6, vocab 151936, TIED embeddings → lm_head = embed_tokens).
//
// Differs from the sibling moss_audio (MOSS-Audio-4B) backend: no DeepStack,
// the conv_out/proj1/proj2 encoder head, and a smaller 1.7B LM.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moss_transcribe_context;

struct moss_transcribe_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct moss_transcribe_context_params moss_transcribe_context_default_params(void);

// Load model from GGUF.
struct moss_transcribe_context* moss_transcribe_init_from_file(const char* path_model,
                                                               struct moss_transcribe_context_params params);

void moss_transcribe_free(struct moss_transcribe_context* ctx);

// Transcribe raw 16 kHz mono PCM. Returns malloc'd UTF-8 string (caller owns).
char* moss_transcribe_transcribe(struct moss_transcribe_context* ctx, const float* samples, int n_samples);

// Per-token streaming callback. Fires once per generated token (id, prob, userdata).
typedef void (*moss_transcribe_token_cb)(int tok_id, float prob, void* userdata);

// Like moss_transcribe_transcribe() but fires cb(tok_id, prob, userdata) for each
// generated token. The final assembled text is NOT returned; output is via cb.
void moss_transcribe_transcribe_cb(struct moss_transcribe_context* ctx, const float* samples, int n_samples,
                                   moss_transcribe_token_cb cb, void* userdata);

// Beam search. 1 = greedy (default). >1 = beam search (core_beam_decode).
void moss_transcribe_set_beam_size(struct moss_transcribe_context* ctx, int beam_size);

// ---- Stage helpers for differential testing (crispasr-diff) ----

// Compute 128-bin log-mel spectrogram (Whisper-style, n_fft 400, hop 160).
// Output: malloc'd (n_mels, T_mel) F32 row-major. Caller frees.
float* moss_transcribe_compute_mel(struct moss_transcribe_context* ctx, const float* samples, int n_samples,
                                   int* out_n_mels, int* out_T_mel);

// Run the full audio encoder (conv stem + 32 layers + ln_post + proj1/proj2).
// Returns (T_enc, output_dim=2048) F32 row-major. Caller frees.
float* moss_transcribe_run_encoder(struct moss_transcribe_context* ctx, const float* mel, int n_mels, int T_mel,
                                   int* out_T_enc, int* out_d);

// Run the audio adapter on encoder output. Returns (T_enc, llm_dim=2048) F32.
float* moss_transcribe_run_adapter(struct moss_transcribe_context* ctx, const float* encoder_out, int T_enc, int d_enc,
                                   int* out_T, int* out_d);

// Embed tokens via the LM token table. Returns (n_tokens, llm_dim) F32.
float* moss_transcribe_embed_tokens(struct moss_transcribe_context* ctx, const int32_t* token_ids, int n_tokens);

// KV cache for LLM decode.
bool moss_transcribe_kv_init(struct moss_transcribe_context* ctx, int max_ctx);
void moss_transcribe_kv_reset(struct moss_transcribe_context* ctx);

// Run LLM with KV cache. Returns last-token logits (vocab_size,) F32.
float* moss_transcribe_run_llm_kv(struct moss_transcribe_context* ctx, const float* inputs_embeds, int n_tokens,
                                  int n_past, int* out_n_tokens, int* out_vocab_size);

// Tokenize text using GPT-2 byte-level BPE. Returns count written to out_tokens.
int moss_transcribe_tokenize(struct moss_transcribe_context* ctx, const char* text, int32_t* out_tokens,
                             int max_tokens);

// Token ID → string (GPT-2 byte-encoded form).
const char* moss_transcribe_token_text(struct moss_transcribe_context* ctx, int token_id);

// Build the inference prompt token IDs for N audio frames. Audio positions hold
// the placeholder id (0); fills *out_audio_mask[i] = 1 at those positions if
// out_audio_mask is non-null (caller-allocated, capacity >= return value).
// Returned via out_ids (caller-allocated, capacity >= n_audio + 8). Returns the
// number of prompt tokens, or -1 on error.
int moss_transcribe_build_prompt(struct moss_transcribe_context* ctx, int n_audio, int32_t* out_ids, int max_ids);

#ifdef __cplusplus
}
#endif
