#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cohere_context;

struct cohere_context_params {
    int n_threads;       // default: number of physical cores
    bool use_flash;      // flash attention in decoder (default: false for now)
    bool use_gpu;        // false => force CPU backend
    bool no_punctuation; // use <|nopnc|> instead of <|pnc|> in prompt (default: false)
    bool diarize;        // use <|diarize|> instead of <|nodiarize|>; model may emit
                         // <|spkchange|> and <|spk0|>..<|spk15|> tokens (experimental)
    // Output verbosity:
    //   0 = silent  — only hard errors (failed/cannot) go to stderr
    //   1 = normal  — model loading info printed (default)
    //   2 = verbose — per-inference timing, per-step tokens, performance report
    int verbosity;
};

struct cohere_context_params cohere_context_default_params(void);

// Load model from GGUF file produced by export_gguf.py
struct cohere_context* cohere_init_from_file(const char* path_model, struct cohere_context_params params);

void cohere_free(struct cohere_context* ctx);

// Transcribe raw 16 kHz mono PCM.
// Returns a newly allocated UTF-8 string (caller must free()).
// lang: ISO-639-1 code e.g. "en", "fr", "de" (NULL → autodetect, not implemented yet)
char* cohere_transcribe(struct cohere_context* ctx, const float* samples, int n_samples, const char* lang);

// ---- Supported languages ----
//
// Cohere Transcribe accepts a FIXED language set and performs no detection of
// its own; a wrong code yields a fluent wrong-language transcript, never an
// error. The set comes from the model's config.json (14 codes for the base
// model, {en, ar} for the Arabic finetune, which shares the base tokenizer —
// so the vocab cannot tell them apart) and rides in the GGUF.
//
// Returns 0 when the GGUF predates the metadata key. That means UNKNOWN, not
// "none": treat it as "no restriction available", never as an empty whitelist.
int cohere_n_supported_languages(struct cohere_context* ctx);

// ISO-639-1 code at index i, or NULL if out of range. Points into the context;
// valid until cohere_free().
const char* cohere_supported_language(struct cohere_context* ctx, int i);

// ---- Probe-based language identification ----
//
// Cohere Transcribe has no language-detection head, but it does answer a
// language prompt — so the model is its own detector: transcribe a short clip
// once per supported language and keep the candidate whose output looks most
// like real speech. This needs no second model, and it can only ever return a
// language this model actually supports, which an external detector (whisper-
// tiny knows 99) cannot promise.
//
// The cost is one encode + one short decode PER CANDIDATE, so it is a good
// trade for the Arabic finetune (2 candidates) and an expensive one for the
// 14-language base model. Callers decide; see CRISPASR_COHERE_PROBE_MAX_LANGS
// in the CLI.

// Optional text-language-detector hook. Given a probe transcript and the
// candidate code it was produced under, return that detector's probability in
// [0,1] that the text really is in `lang` (0 if it cannot tell). Agreement is
// the strongest single signal among same-script languages; scoring still works
// without it, just more weakly. NULL = unavailable.
typedef float (*cohere_text_lid_fn)(const char* text, const char* lang, void* user_data);

struct cohere_lid_params {
    float probe_seconds;         // leading clip length to probe (default 20)
    int max_new_tokens;          // cap per probe decode (default 48)
    cohere_text_lid_fn text_lid; // optional agreement hook (default NULL)
    void* text_lid_user;         // opaque, passed back to text_lid
    int verbosity;               // >0 prints per-candidate scores to stderr
};

struct cohere_lid_params cohere_lid_default_params(void);

// Detect the spoken language by probing the model over its own supported set.
// Writes an ISO-639-1 code into out_lang and, if non-NULL, the winner's share
// of the total score into out_confidence.
//
// Returns false when the GGUF carries no supported-language list (nothing to
// probe over — fall back to an external detector) or when every probe came
// back empty.
bool cohere_detect_language(struct cohere_context* ctx, const float* samples, int n_samples,
                            struct cohere_lid_params params, char* out_lang, int out_lang_size, float* out_confidence);

// Vocabulary helpers
int cohere_n_vocab(struct cohere_context* ctx);
const char* cohere_token_to_str(struct cohere_context* ctx, int token_id);
int cohere_str_to_token(struct cohere_context* ctx, const char* str);

// Sampling: temperature > 0 enables stable softmax sampling in the
// transformer decoder. Default 0 keeps the bit-identical greedy path.
// Sticky on the context until the next call.
void cohere_set_temperature(struct cohere_context* ctx, float temperature, uint64_t seed);
void cohere_set_max_new_tokens(struct cohere_context* ctx, int max_new_tokens);
void cohere_set_frequency_penalty(struct cohere_context* ctx, float frequency_penalty);

// §90 beam-search width. n > 1 activates beam search; n <= 0 clamped to 1 (greedy).
void cohere_set_beam_size(struct cohere_context* ctx, int n);

// ---- Extended API: per-token confidence and timing ----

// Per-token data returned by cohere_transcribe_ex().
struct cohere_token_data {
    int id;        // vocabulary token ID
    char text[64]; // decoded text (SentencePiece '▁' already converted to ' ')
    float p;       // softmax probability [0, 1]
    int64_t t0;    // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;    // end time, centiseconds
};

// Result from cohere_transcribe_ex() — free with cohere_result_free().
struct cohere_result {
    char* text;                       // full transcript (malloc'd)
    struct cohere_token_data* tokens; // per-token data (malloc'd)
    int n_tokens;
};

void cohere_result_free(struct cohere_result* r);

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
struct cohere_result* cohere_transcribe_ex(struct cohere_context* ctx, const float* samples, int n_samples,
                                           const char* lang, int64_t t_offset_cs);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// Returns malloc'd F32 buffers the caller must free(). NULL on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM, row-major (n_mels, T_mel).
// Applies cohere's pre-emphasis (0.97) and NeMo-style per-feature log-mel
// exactly as the live encoder path does.
float* cohere_compute_mel(struct cohere_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                          int* out_T_mel);

// Run just the audio encoder on a mel spectrogram. Takes (n_mels, T_mel)
// row-major mel as produced by cohere_compute_mel() and returns the
// encoder hidden state in row-major (T_enc, d_model) where T_enc is the
// mel frame count after the 8x conv subsampling.
float* cohere_run_encoder(struct cohere_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                          int* out_d_model);

// Staged encoder: runs the encoder with per-layer snapshots for crispasr-diff.
// Callback receives each snapshot: name, data, T_enc, d_model.
typedef void (*cohere_stage_cb)(const char* name, const float* data, int T_enc, int d_model, void* userdata);
int cohere_run_encoder_staged(struct cohere_context* ctx, const float* mel, int n_mels, int T_mel, cohere_stage_cb cb,
                              void* userdata);

// ---- Streaming delta-encoding support ----
//
// When transcribing sequential overlapping audio windows (e.g. in
// JSON+VAD streaming mode), the caller may enable delta encoding to
// avoid re-encoding the overlapping portion of the audio. Instead,
// the previous step's cross-KV cache is retained on the CPU, and only
// the NEW samples are encoded. The old + new cross-KV are spliced
// together using the same multi-chunk scatter-copy path already
// proven by the 30s+ long-audio chunker.
//
// Call cohere_set_stream_delta(ctx, new_samples) before every
// cohere_transcribe_ex() call. Pass 0 to disable delta encoding
// (full encode) for the next call.
void cohere_set_stream_delta(struct cohere_context* ctx, int delta_new_samples);

// Utterance-level delta: save/restore the utterance portion of the
// cached cross-KV so finalize redecode only encodes the new tail.
void cohere_save_utterance_cross_kv(struct cohere_context* ctx, int64_t utterance_start_sample, int64_t utterance_end_sample, int64_t window_start_sample, int sample_rate);
void cohere_restore_utterance_cross_kv(struct cohere_context* ctx, int n_new_samples);
void cohere_clear_utterance_cross_kv(struct cohere_context* ctx);

#ifdef __cplusplus
}
#endif
