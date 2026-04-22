// parakeet.h — public C API for nvidia/parakeet-tdt-0.6b-v3 ggml runtime
//
// Multilingual ASR (25 European languages) using FastConformer encoder +
// Token-and-Duration Transducer (TDT) decoder. Word-level timestamps come
// for free from the duration head — no separate CTC alignment needed.
//
// Models are loaded from GGUF files produced by:
//   python models/convert-parakeet-to-gguf.py --nemo X.nemo --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct parakeet_context;

struct parakeet_context_params {
    int n_threads;
    bool use_flash; // flash attention in encoder (default: false)
    int verbosity;  // 0=silent 1=normal 2=verbose
    bool use_gpu;   // false => force CPU backend
};

struct parakeet_context_params parakeet_context_default_params(void);

// Load model from GGUF (produced by convert-parakeet-to-gguf.py)
struct parakeet_context* parakeet_init_from_file(const char* path_model, struct parakeet_context_params params);

void parakeet_free(struct parakeet_context* ctx);

// ---- Per-token data returned by parakeet_transcribe_ex() ----

struct parakeet_token_data {
    int id;        // SentencePiece token id (0 .. vocab_size-1)
    char text[48]; // decoded text (SentencePiece '▁' converted to ' ')
    int64_t t0;    // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;    // end time,   centiseconds (start + duration*frame_dur_cs)
    float p;       // softmax probability of the emitted token [0,1]
};

// Word-level data: sub-word tokens grouped at SentencePiece '▁' boundaries.
struct parakeet_word_data {
    char text[64]; // word text (no leading space)
    int64_t t0;    // start time, centiseconds (from first sub-word)
    int64_t t1;    // end time,   centiseconds (from last sub-word)
};

struct parakeet_result {
    char* text;                         // full transcript (malloc'd, caller owns)
    struct parakeet_token_data* tokens; // per-token timing (malloc'd)
    int n_tokens;
    struct parakeet_word_data* words; // grouped word timings (malloc'd)
    int n_words;
};

void parakeet_result_free(struct parakeet_result* r);

// Transcribe raw 16 kHz mono PCM, returning a malloc'd UTF-8 string.
char* parakeet_transcribe(struct parakeet_context* ctx, const float* samples, int n_samples);

// Like parakeet_transcribe but returns per-token TDT timestamps.
//
// t_offset_cs: absolute start of this audio slice in centiseconds.
//   Token t0/t1 = t_offset_cs + (encoder_frame * frame_dur_cs).
//   For long audio with VAD, pass (vad_segment_t0_seconds * 100).
//
// Unlike Cohere's cross-attention DTW path, these timestamps come directly
// from the TDT decoder's duration head and are accurate to one encoder
// frame (~80 ms for parakeet-tdt-0.6b-v3).
struct parakeet_result* parakeet_transcribe_ex(struct parakeet_context* ctx, const float* samples, int n_samples,
                                               int64_t t_offset_cs);

// Vocabulary helpers
int parakeet_n_vocab(struct parakeet_context* ctx);
int parakeet_blank_id(struct parakeet_context* ctx);
const char* parakeet_token_to_str(struct parakeet_context* ctx, int token_id);

// Sampling: when temperature > 0, the TDT decoder draws each non-blank
// token via stable-softmax(logits / temperature) instead of argmax.
// Temperature == 0 (the default) keeps the bit-identical pure-greedy
// path. Set per-call as needed; the setting is sticky on the context
// until the next call. seed == 0 means time-based RNG.
void parakeet_set_temperature(struct parakeet_context* ctx, float temperature, uint64_t seed);

// Hyper-parameters needed by callers (frame duration for stamping etc.)
int parakeet_frame_dur_cs(struct parakeet_context* ctx); // centiseconds per encoder frame
int parakeet_n_mels(struct parakeet_context* ctx);
int parakeet_sample_rate(struct parakeet_context* ctx);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// These let the diff harness compare intermediate activations against
// a PyTorch reference. They are NOT needed by the normal transcribe
// path — use parakeet_transcribe(_ex) for inference.
//
// Returns a malloc'd F32 buffer that the caller must free(). Shape is
// reported via the out_* parameters. Returns nullptr on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM.
// Output layout: row-major (n_mels, T_mel).
float* parakeet_compute_mel(struct parakeet_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                            int* out_T_mel);

// Run just the audio encoder on a mel spectrogram. Takes the output of
// parakeet_compute_mel() (or any externally-produced reference mel with
// the same layout) and returns the encoder hidden state.
// Output layout: row-major (T_enc, d_model).
float* parakeet_run_encoder(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                            int* out_d_model);

// Internal smoke test: build encoder graph on a zero mel of `T_mel` frames,
// run it, and report the output T_enc. Returns T_enc on success or -1.
int parakeet_test_encoder(struct parakeet_context* ctx, int T_mel);

// Internal smoke test: take raw 16 kHz mono PCM, run mel + encoder, print
// encoder-output statistics. Returns T_enc on success or -1.
int parakeet_test_audio(struct parakeet_context* ctx, const float* samples, int n_samples);

#ifdef __cplusplus
}
#endif
