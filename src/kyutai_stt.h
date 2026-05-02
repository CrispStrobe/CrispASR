// kyutai_stt.h — C API for Kyutai STT (stt-1b-en_fr, stt-2.6b-en).
//
// Architecture: Mimi audio codec encoder (SEANet CNN + transformer + RVQ)
//             + Causal transformer LM (2048d, 16L, RoPE, SwiGLU, RMSNorm)
//
// Audio flow: 24kHz PCM → SEANet encoder → transformer → downsample →
//             RVQ (32 codebooks) → LM → text tokens → SentencePiece decode

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kyutai_stt_context;

struct kyutai_stt_context_params {
    int n_threads;
    int verbosity;     // 0=silent 1=normal 2=verbose
    bool use_gpu;      // false => force CPU backend
    float temperature; // 0 = greedy argmax, >0 = softmax sampling
};

struct kyutai_stt_context_params kyutai_stt_context_default_params(void);

struct kyutai_stt_context* kyutai_stt_init_from_file(const char* path_model, struct kyutai_stt_context_params params);

void kyutai_stt_free(struct kyutai_stt_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Internally resamples to 24 kHz for Mimi codec.
// Returns malloc'd UTF-8 string, caller frees with free().
char* kyutai_stt_transcribe(struct kyutai_stt_context* ctx, const float* samples, int n_samples);

// Variant returning per-emitted-token ids + softmax probs alongside the text.
// All pointers are malloc'd; free with kyutai_stt_result_free.
struct kyutai_stt_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct kyutai_stt_result* kyutai_stt_transcribe_with_probs(struct kyutai_stt_context* ctx, const float* samples,
                                                           int n_samples);
void kyutai_stt_result_free(struct kyutai_stt_result* r);

// Token text lookup.
const char* kyutai_stt_token_text(struct kyutai_stt_context* ctx, int id);

// Sticky seed for the multinomial sampler (best-of-N). 0 = leave libc RNG
// state alone; non-zero = `srand(seed)`. Process-global; serialize best-of-N
// at the adapter level.
void kyutai_stt_set_seed(struct kyutai_stt_context* ctx, unsigned int seed);

// ---- Per-token + word-level timing (PLAN #61c) ----
//
// Kyutai's "delayed-streams" architecture aligns each emitted text
// token to the audio frame that produced it. Frame rate is 12.5 Hz
// (80 ms per frame); the LM emits one text token per Mimi frame.
// `audio_delay_seconds` (typically 0.5s) is the training-time
// lookahead the LM was given — we subtract it from the LM frame index
// to recover the audio time the token actually corresponds to.

struct kyutai_stt_token_data {
    int id;        // SentencePiece token id
    char text[48]; // decoded text (▁ → space)
    int64_t t0;    // start time, centiseconds (audio-relative, includes t_offset_cs)
    int64_t t1;    // end time,   centiseconds (= t0 + 8, one Mimi frame)
    float p;       // softmax probability of the emitted token [0,1]
};

// Word-level data: sub-tokens grouped at SentencePiece '▁' boundaries.
struct kyutai_stt_word_data {
    char text[64]; // word text (no leading space)
    int64_t t0;    // start time, centiseconds (from first sub-token)
    int64_t t1;    // end time,   centiseconds (from last sub-token)
    float p;       // mean softmax probability across the word's sub-tokens
};

struct kyutai_stt_result_ex {
    char* text;                           // full transcript (malloc'd)
    struct kyutai_stt_token_data* tokens; // per-token timing (malloc'd)
    int n_tokens;
    struct kyutai_stt_word_data* words; // grouped word timings (malloc'd)
    int n_words;
};

void kyutai_stt_result_ex_free(struct kyutai_stt_result_ex* r);

// Like kyutai_stt_transcribe but returns per-token + word-level timing.
//
// `t_offset_cs`: absolute start of this audio slice in centiseconds.
//   Token t0/t1 = t_offset_cs + (audio_frame * 8). For long audio with
//   VAD slicing, pass (vad_segment_t0_seconds * 100).
//
// Timestamps are accurate to one Mimi frame (80 ms) by construction —
// the kyutai LM's frame-aligned emission means no DTW or auxiliary
// duration head is needed.
struct kyutai_stt_result_ex* kyutai_stt_transcribe_ex(struct kyutai_stt_context* ctx, const float* samples,
                                                      int n_samples, int64_t t_offset_cs);

#ifdef __cplusplus
}
#endif
