// canary.h — public C API for nvidia/canary-1b-v2 ggml runtime
//
// Multilingual ASR + speech translation across 25 European languages,
// with explicit source_lang / target_lang task tokens (the fix for the
// auto-language-ID problem we hit with parakeet on German audio).
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

// Vocabulary helpers
int canary_n_vocab(struct canary_context* ctx);
const char* canary_token_to_str(struct canary_context* ctx, int token_id);
int canary_str_to_token(struct canary_context* ctx, const char* str);

// Sampling: temperature > 0 makes the decoder sample over the per-step
// softmax instead of argmax. Default 0 keeps the bit-identical greedy
// path. Sticky on the context until the next call.
void canary_set_temperature(struct canary_context* ctx, float temperature, uint64_t seed);

// Hyper-parameters
int canary_frame_dur_cs(struct canary_context* ctx);
int canary_n_mels(struct canary_context* ctx);
int canary_sample_rate(struct canary_context* ctx);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// Returns malloc'd F32 buffers the caller must free(). nullptr on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM, row-major (n_mels, T_mel).
float* canary_compute_mel(struct canary_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                          int* out_T_mel);

// Run just the audio encoder on a mel spectrogram. Output layout
// row-major (T_enc, d_model). Same output as what the internal
// encoder pass produces before the decoder sees it.
float* canary_run_encoder(struct canary_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                          int* out_d_model);

// Internal smoke test: load and report all hparams. Returns 0 on success.
int canary_test_load(struct canary_context* ctx);

// Internal smoke test: build encoder graph on a zero mel of `T_mel` frames,
// run it, and report the output T_enc. Returns T_enc on success or -1.
int canary_test_encoder(struct canary_context* ctx, int T_mel);

#ifdef __cplusplus
}
#endif
