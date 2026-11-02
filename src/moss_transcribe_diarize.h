// moss_transcribe_diarize.h — public C API for MOSS-Transcribe-Diarize-0.9B
//
// Joint ASR + speaker diarization + timestamps in a single 0.9B model.
// Input: 16 kHz mono PCM. Output: timestamped, speaker-labelled text segments.
//
// Architecture: OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B (Apache-2.0)
//   Whisper encoder: 80-mel → Conv1d stem → +pos embed → 24L pre-LN transformer
//     (1024d, 16 heads, global attention) → ln_post.
//   VQAdaptor: 4x temporal merge (reshape T/4 × 4096) → Linear(4096→1024) +
//     SiLU + Linear(1024→1024) + LayerNorm(1024).
//   Time markers: every 5s of audio, inject digit tokens into audio_pad sequence.
//   LM: Qwen3-0.6B (28L, 1024d, 16Q/8KV, head_dim 128, SwiGLU 3072,
//     RoPE θ=1e6, vocab 151936, TIED embeddings).
//   Output: text with [timestamp][Sxx] speaker labels.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moss_diarize_context;

struct moss_diarize_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct moss_diarize_params moss_diarize_default_params(void);

// A single diarized segment returned by the model.
struct moss_diarize_segment {
    int64_t t0_cs;   // start time in centiseconds
    int64_t t1_cs;   // end time in centiseconds
    int speaker_id;  // 1-based (from [S01], [S02], ...)
    char text[1024]; // transcribed text for this segment
};

// Load model from GGUF.
struct moss_diarize_context* moss_diarize_init_from_file(const char* path_model, struct moss_diarize_params params);

void moss_diarize_free(struct moss_diarize_context* ctx);

// Transcribe raw 16 kHz mono PCM. Returns malloc'd UTF-8 string with raw model
// output (timestamps + speaker labels). Caller owns and frees with free().
char* moss_diarize_transcribe(struct moss_diarize_context* ctx, const float* samples, int n_samples);

// Transcribe and parse into diarized segments. Returns number of segments
// written to out_segments (caller-allocated, capacity max_segments).
int moss_diarize_transcribe_segments(struct moss_diarize_context* ctx, const float* samples, int n_samples,
                                     struct moss_diarize_segment* out_segments, int max_segments);

// Set hotwords for prompt injection (comma-separated). Pass NULL to clear.
void moss_diarize_set_hotwords(struct moss_diarize_context* ctx, const char* hotwords);

// #292: forward --max-new-tokens. <= 0 keeps the backend's 1024 default (diarize
// output is long; the global 512 truncates single-pass long audio).
void moss_diarize_set_max_new_tokens(struct moss_diarize_context* ctx, int max_new_tokens);

// Beam search width. 1 = greedy (default); >1 = beam search.
void moss_diarize_set_beam_size(struct moss_diarize_context* ctx, int beam_size);

// Optional sampling for the greedy (beam_size == 1) decoder (#564). temperature
// <= 0 keeps greedy argmax (the default). With temperature > 0 each step draws
// from softmax(logits / temperature) after top_k (<= 0 disables) and top_p
// (>= 1 disables) filtering, in HF generate() order. seed 0 = nondeterministic;
// the RNG is re-seeded at the start of every transcribe call, so a fixed seed
// reproduces a call. Beam search (beam_size > 1) takes precedence.
void moss_diarize_set_sampling(struct moss_diarize_context* ctx, float temperature, int top_k, float top_p,
                               uint64_t seed);

// Override the system instruction. Pass NULL/"" to restore the default
// diarization prompt. Persists until changed.
void moss_diarize_set_ask(struct moss_diarize_context* ctx, const char* instruction);

// Set a language hint injected into the system prompt. Pass NULL/"" to clear.
void moss_diarize_set_language(struct moss_diarize_context* ctx, const char* lang);

// ---- Stage helpers for differential testing (crispasr-diff) ----

float* moss_diarize_compute_mel(struct moss_diarize_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                int* out_T_mel);

// Run the full audio encoder pipeline with 30s chunking.
// Input: raw 16 kHz mono PCM. Output: concatenated valid encoder frames.
float* moss_diarize_run_encoder(struct moss_diarize_context* ctx, const float* samples, int n_samples, int* out_T_total,
                                int* out_d);

float* moss_diarize_run_adaptor(struct moss_diarize_context* ctx, const float* encoder_out, int T_enc, int d_enc,
                                int* out_T, int* out_d);

float* moss_diarize_embed_tokens(struct moss_diarize_context* ctx, const int32_t* token_ids, int n_tokens);

bool moss_diarize_kv_init(struct moss_diarize_context* ctx, int max_ctx);
void moss_diarize_kv_reset(struct moss_diarize_context* ctx);

float* moss_diarize_run_llm_kv(struct moss_diarize_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                               int* out_n_tokens, int* out_vocab_size);

int moss_diarize_tokenize(struct moss_diarize_context* ctx, const char* text, int32_t* out_tokens, int max_tokens);

const char* moss_diarize_token_text(struct moss_diarize_context* ctx, int token_id);

#ifdef __cplusplus
}
#endif
