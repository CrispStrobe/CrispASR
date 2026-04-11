#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cohere_context;

struct cohere_context_params {
    int   n_threads;        // default: number of physical cores
    bool  use_flash;        // flash attention in decoder (default: false for now)
    bool  no_punctuation;   // use <|nopnc|> instead of <|pnc|> in prompt (default: false)
    bool  diarize;          // use <|diarize|> instead of <|nodiarize|>; model may emit
                            // <|spkchange|> and <|spk0|>..<|spk15|> tokens (experimental)
    // Output verbosity:
    //   0 = silent  — only hard errors (failed/cannot) go to stderr
    //   1 = normal  — model loading info printed (default)
    //   2 = verbose — per-inference timing, per-step tokens, performance report
    int   verbosity;
};

struct cohere_context_params cohere_context_default_params(void);

// Load model from GGUF file produced by export_gguf.py
struct cohere_context * cohere_init_from_file(const char * path_model,
                                              struct cohere_context_params params);

void cohere_free(struct cohere_context * ctx);

// Transcribe raw 16 kHz mono PCM.
// Returns a newly allocated UTF-8 string (caller must free()).
// lang: ISO-639-1 code e.g. "en", "fr", "de" (NULL → autodetect, not implemented yet)
char * cohere_transcribe(struct cohere_context * ctx,
                         const float * samples, int n_samples,
                         const char * lang);

// Vocabulary helpers
int         cohere_n_vocab(struct cohere_context * ctx);
const char* cohere_token_to_str(struct cohere_context * ctx, int token_id);
int         cohere_str_to_token(struct cohere_context * ctx, const char * str);

// Sampling: temperature > 0 enables stable softmax sampling in the
// transformer decoder. Default 0 keeps the bit-identical greedy path.
// Sticky on the context until the next call.
void        cohere_set_temperature(struct cohere_context * ctx,
                                   float temperature, uint64_t seed);

// ---- Extended API: per-token confidence and timing ----

// Per-token data returned by cohere_transcribe_ex().
struct cohere_token_data {
    int     id;        // vocabulary token ID
    char    text[48];  // decoded text (SentencePiece '▁' already converted to ' ')
    float   p;         // softmax probability [0, 1]
    int64_t t0;        // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;        // end time, centiseconds
};

// Result from cohere_transcribe_ex() — free with cohere_result_free().
struct cohere_result {
    char   * text;                     // full transcript (malloc'd)
    struct cohere_token_data * tokens; // per-token data (malloc'd)
    int      n_tokens;
};

void cohere_result_free(struct cohere_result * r);

// Like cohere_transcribe() but also returns per-token probability and timing.
//
// t_offset_cs: absolute start time of this audio slice, in centiseconds.
//   Token t0/t1 values equal (t_offset_cs + interpolated_offset_within_segment).
//   Pass 0 when processing a single file without VAD segmentation.
//   With VAD, pass (vad_segment_t0_seconds * 100).
//
// Token times are linearly interpolated across the segment duration,
// proportional to each token's decoded text length (best approximation
// without model-native timestamp tokens).
//
// Returns NULL on failure. Free result with cohere_result_free().
struct cohere_result * cohere_transcribe_ex(
    struct cohere_context * ctx,
    const float * samples,
    int           n_samples,
    const char  * lang,
    int64_t       t_offset_cs);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// Returns malloc'd F32 buffers the caller must free(). NULL on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM, row-major (n_mels, T_mel).
// Applies cohere's pre-emphasis (0.97) and NeMo-style per-feature log-mel
// exactly as the live encoder path does.
float *     cohere_compute_mel (struct cohere_context * ctx,
                                const float * samples, int n_samples,
                                int * out_n_mels, int * out_T_mel);

// Run just the audio encoder on a mel spectrogram. Takes (n_mels, T_mel)
// row-major mel as produced by cohere_compute_mel() and returns the
// encoder hidden state in row-major (T_enc, d_model) where T_enc is the
// mel frame count after the 8x conv subsampling.
float *     cohere_run_encoder (struct cohere_context * ctx,
                                const float * mel, int n_mels, int T_mel,
                                int * out_T_enc, int * out_d_model);

#ifdef __cplusplus
}
#endif
