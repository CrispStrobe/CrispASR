// canary_ctc.h — runtime for canary-1b-v2's auxiliary CTC alignment model
//
// The auxiliary model is a 600M-param FastConformer + linear CTC head over
// canary's 16384-piece SentencePiece vocabulary. NeMo Forced Aligner (NFA)
// uses it to compute frame-aligned word timestamps for canary's transcript
// output, per the Canary paper section 5.
//
// We use it as a general-purpose multilingual subword CTC aligner: drop in
// any transcript text + audio pair, and get back per-word (t0, t1) at one
// encoder frame = 80 ms granularity. Works with cohere/parakeet/whisper
// transcripts too — just re-tokenise the words through the CTC model's
// SentencePiece vocab before running Viterbi.
//
// Models are loaded from GGUF files produced by:
//   python models/convert-canary-ctc-to-gguf.py --nemo X.nemo --output X.gguf

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct canary_ctc_context;

struct canary_ctc_context_params {
    int n_threads;
    int verbosity;
};

struct canary_ctc_context_params canary_ctc_context_default_params(void);

struct canary_ctc_context* canary_ctc_init_from_file(const char* path_model, struct canary_ctc_context_params params);

void canary_ctc_free(struct canary_ctc_context* ctx);

// Compute per-frame log-probabilities. Caller passes raw 16 kHz mono PCM.
// Returns a flat row-major float array of length T_enc * vocab_total written
// into `out_logits` (caller-allocated). T_enc and vocab_total are returned by
// reference. Returns 0 on success, non-zero on error.
//
// Layout: out_logits[t * vocab_total + v] = log P(token v | frame t).
int canary_ctc_compute_logits(struct canary_ctc_context* ctx, const float* samples, int n_samples,
                              float** out_logits, // newly malloc'd, caller frees
                              int* out_T_enc, int* out_vocab_total);

// One emitted aligned word.
struct canary_ctc_word {
    char text[64];
    int64_t t0; // centiseconds (caller may add an offset)
    int64_t t1;
};

// Forced-align a list of words against the CTC logits using subword Viterbi.
// `words` are arbitrary UTF-8 word strings (no leading space). The function
// internally re-tokenises each word through the CTC model's SentencePiece
// vocabulary (greedy longest-prefix match), builds the CTC label sequence,
// runs the Viterbi DP, and returns per-word (t0, t1) in centiseconds
// relative to the start of the audio.
//
// Returns 0 on success. The caller passes a pre-allocated array of
// `n_words` canary_ctc_word slots. The .text field is filled with the
// original word.
int canary_ctc_align_words(struct canary_ctc_context* ctx,
                           const float* logits, // [T_enc * vocab_total] (from canary_ctc_compute_logits)
                           int T_enc, int vocab_total,
                           const char** words, // array of n_words UTF-8 strings
                           int n_words,
                           struct canary_ctc_word* out_words); // pre-allocated, length n_words

// Convenience: greedy CTC decode (no forced alignment). Returns a malloc'd
// transcript string. Useful as a sanity check that the model loads correctly.
char* canary_ctc_greedy_decode(struct canary_ctc_context* ctx, const float* logits, int T_enc, int vocab_total);

// Greedy CTC decode that also returns per-emission token info: one entry
// per non-blank, non-special emitted symbol after blank-strip + repeat-collapse.
// `probs[i]` is the softmax probability of the picked token at frame
// `frame_starts[i]` (the first frame of the collapsed run).
// `frame_ends[i]` is the last frame (inclusive) of that run.
// `text_offsets[i]` / `text_lengths[i]` index into the result `text` field
// so callers can extract the substring corresponding to each emission
// without repeating the ▁→space mapping.
// Returns nullptr on failure. Caller frees with canary_ctc_decode_result_free.
struct canary_ctc_decode_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int* frame_starts;
    int* frame_ends;
    int* text_offsets;
    int* text_lengths;
    int n_tokens;
};

struct canary_ctc_decode_result* canary_ctc_greedy_decode_with_probs(struct canary_ctc_context* ctx,
                                                                     const float* logits, int T_enc, int vocab_total);

void canary_ctc_decode_result_free(struct canary_ctc_decode_result* r);

// Hyperparameters
int canary_ctc_n_vocab(struct canary_ctc_context* ctx);
int canary_ctc_blank_id(struct canary_ctc_context* ctx);
int canary_ctc_frame_dur_cs(struct canary_ctc_context* ctx);
int canary_ctc_n_mels(struct canary_ctc_context* ctx);
int canary_ctc_sample_rate(struct canary_ctc_context* ctx);

// Debug: compute the log-mel spectrogram the encoder would see, without
// running the encoder graph. Returns a malloc'd row-major F32 buffer of
// shape (n_mels, T_mel) — caller frees. Used for diffing our mel
// pipeline against a PyTorch / NeMo reference.
float* canary_ctc_compute_mel_debug(struct canary_ctc_context* ctx, const float* samples, int n_samples,
                                    int* out_n_mels, int* out_T_mel);

#ifdef __cplusplus
}
#endif
