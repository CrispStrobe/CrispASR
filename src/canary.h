// canary.h — public C API for NVIDIA Canary AED ggml runtimes
//
// Supports both the legacy cstr/canary-1b-v2 GGUF layout and the
// transcribe.cpp Canary GGUF schema (including Canary 180M Flash's split
// 512-wide encoder, 1024-wide decoder, and trained projection).
//
// ASR mode:        source_lang == target_lang  (e.g. "de" → "de")
// Translation:     source_lang != target_lang  (e.g. "de" → "en")
//
// Models are loaded from GGUF files produced by:
//   python models/convert-canary-to-gguf.py --nemo X.nemo --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct canary_context;

struct canary_context_params {
    int n_threads;
    bool use_flash; // flash attention in encoder/decoder (default: false)
    int verbosity;  // 0=silent 1=normal 2=verbose
    bool use_gpu;   // false => force CPU backend
};

struct canary_context_params canary_context_default_params(void);

struct canary_context* canary_init_from_file(const char* path_model, struct canary_context_params params);

void canary_free(struct canary_context* ctx);

// ---- Per-token data returned by canary_transcribe_ex() ----

struct canary_token_data {
    int id;        // SentencePiece token id
    char text[64]; // decoded text (▁ → ' ')
    int64_t t0;    // centiseconds (absolute, includes t_offset_cs)
    int64_t t1;
    float p; // softmax probability of the emitted token [0, 1]
};

struct canary_word_data {
    char text[64];
    int64_t t0;
    int64_t t1;
};

struct canary_result {
    char* text;
    struct canary_token_data* tokens;
    int n_tokens;
    struct canary_word_data* words;
    int n_words;
};

void canary_result_free(struct canary_result* r);

// Transcribe (or translate) raw 16 kHz mono PCM.
//
// source_lang: ISO-639-1 code (e.g. "de", "en", "fr"). Required.
// target_lang: ISO-639-1 code. If equal to source_lang → ASR. Otherwise
//              → speech translation. Required.
// punctuation: enable punctuation + capitalisation in the output.
//
// Returns NULL on failure.
char* canary_transcribe(struct canary_context* ctx, const float* samples, int n_samples, const char* source_lang,
                        const char* target_lang, bool punctuation);

// Like canary_transcribe but returns per-token timing.
struct canary_result* canary_transcribe_ex(struct canary_context* ctx, const float* samples, int n_samples,
                                           const char* source_lang, const char* target_lang, bool punctuation,
                                           int64_t t_offset_cs);

// Variant-aware offline long-audio entry. Legacy canary-1b-v2 keeps its
// NeMo dynamic 30..40 s / 1 s-overlap token-LCS blueprint. New-schema
// canary-180m-flash uses direct inference through 40 s, then fixed 20 s
// windows / 6 s overlap with centered time-core stitching; every window
// receives a fresh task prompt. Positive chunk/overlap arguments override
// the variant defaults. This is external-style offline chunking, not native
// model streaming. Returned timings are runtime DTW estimates, not Canary's
// upstream auxiliary CTC timestamps.
struct canary_result* canary_transcribe_streamed(struct canary_context* ctx, const float* samples, int n_samples,
                                                 const char* source_lang, const char* target_lang, bool punctuation,
                                                 int64_t t_offset_cs, int chunk_seconds, int overlap_seconds);

// Vocabulary helpers
int canary_n_vocab(struct canary_context* ctx);
const char* canary_token_to_str(struct canary_context* ctx, int token_id);
int canary_str_to_token(struct canary_context* ctx, const char* str);

// Sampling: temperature > 0 makes the decoder sample over the per-step
// softmax instead of argmax. Default 0 keeps the bit-identical greedy
// path. Sticky on the context until the next call.
void canary_set_temperature(struct canary_context* ctx, float temperature, uint64_t seed);

// §90 beam-search width. n > 1 activates beam search; n <= 0 clamped to 1 (greedy).
void canary_set_beam_size(struct canary_context* ctx, int n);
void canary_set_max_new_tokens(struct canary_context* ctx, int n);

// Hyper-parameters
int canary_frame_dur_cs(struct canary_context* ctx);
int canary_n_mels(struct canary_context* ctx);
int canary_sample_rate(struct canary_context* ctx);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// Returns malloc'd F32 buffers the caller must free(). nullptr on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM, row-major (n_mels, T_mel).
float* canary_compute_mel(struct canary_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                          int* out_T_mel);

// Run just the native audio encoder on a mel spectrogram. Output layout
// row-major (T_enc, encoder_d_model). For split-width variants this is the
// pre-projection output; the normal transcription path applies the trained
// encoder-to-decoder projection before cross-attention.
float* canary_run_encoder(struct canary_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                          int* out_d_model);

// Staged encoder: runs the encoder with dup snapshots at every layer boundary
// and invokes `cb` once per named stage. Stage names: "pre_enc_out",
// "enc_L00".."enc_L31", "enc_out". T_enc and d_model are the actual shapes.
// Returns 0 on success, -1 on error. Used by crispasr-diff for per-layer
// cosine comparison against the Python reference.
typedef void (*canary_stage_cb)(const char* stage_name, const float* data, int T_enc, int d_model, void* userdata);
int canary_run_encoder_staged(struct canary_context* ctx, const float* mel, int n_mels, int T_mel, canary_stage_cb cb,
                              void* userdata);

// Internal smoke test: load and report all hparams. Returns 0 on success.
int canary_test_load(struct canary_context* ctx);

// Internal smoke test: build encoder graph on a zero mel of `T_mel` frames,
// run it, and report the output T_enc. Returns T_enc on success or -1.
int canary_test_encoder(struct canary_context* ctx, int T_mel);

#ifdef __cplusplus
}
#endif
