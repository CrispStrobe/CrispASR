// firered_asr.h — C API for FireRedASR2-AED (FireRedTeam/FireRedASR2-AED).
//
// Architecture: Conformer encoder (16L, d=1280, 20 heads, relative PE, macaron FFN,
//               depthwise conv k=33, BatchNorm, Swish+GLU)
//             + Transformer decoder (16L, d=1280, cross-attention, GELU FFN)
//             + CTC head (for optional CTC decoding)
//             + Mixed BPE tokenizer (8667 tokens: Chinese chars + English BPE)
//
// Audio flow: 16kHz PCM → 80-dim log-mel → Conv2d subsampling (4x) →
//             Conformer encoder → Transformer decoder (beam search) → text

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct firered_asr_context;

struct firered_asr_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;  // false => force CPU backend
    int beam_size; // ASR beam width (ignored for LID; clamped to >= 1)
};

struct firered_asr_context_params firered_asr_context_default_params(void);

struct firered_asr_context* firered_asr_init_from_file(const char* path_model,
                                                       struct firered_asr_context_params params);

void firered_asr_free(struct firered_asr_context* ctx);

// High-level: transcribe raw 16 kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* firered_asr_transcribe(struct firered_asr_context* ctx, const float* samples, int n_samples);

// Token text lookup.
const char* firered_asr_token_text(struct firered_asr_context* ctx, int id);

// Runtime beam-size override (mutates ctx->params.beam_size). Used by
// the session API's crispasr_session_set_beam_size — PLAN §90. Pass
// 0 or negative to reset to the safe default (1 = greedy).
void firered_asr_set_beam_size(struct firered_asr_context* ctx, int beam_size);

// Result of `firered_asr_transcribe_with_probs`: full transcript + parallel
// arrays of token ids + per-token softmax probabilities (winning beam only).
// `n_tokens` excludes SOS and EOS. `text` is the post-processed UTF-8
// transcript (▁→space, EOS-stripped). All pointers are malloc'd; free with
// `firered_asr_result_free`.
struct firered_asr_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct firered_asr_result* firered_asr_transcribe_with_probs(struct firered_asr_context* ctx, const float* samples,
                                                             int n_samples);

void firered_asr_result_free(struct firered_asr_result* r);

// ---------------------------------------------------------------------------
// Stage API for diff regression (crispasr-diff)
// ---------------------------------------------------------------------------

// Compute Kaldi-compatible 80-dim log-mel fbank features + CMVN normalisation
// from raw 16 kHz mono PCM. Returns a malloc'd (n_frames, 80) F32 buffer in
// row-major order, or null on failure. Caller frees with free().
// *out_n_frames is set to the number of output frames.
float* firered_asr_compute_fbank(struct firered_asr_context* ctx, const float* samples, int n_samples,
                                 int* out_n_frames);

// Run the full encoder (Conv2d subsampling + Conformer hybrid_encoder) on
// pre-computed fbank+CMVN features. features: (n_frames, 80) row-major F32
// (e.g. from firered_asr_compute_fbank). Returns malloc'd (T_enc, d_model)
// F32 buffer, or null on failure. Caller frees with free().
// *out_T_enc and *out_d_model are set on success.
float* firered_asr_run_encoder(struct firered_asr_context* ctx, const float* features, int n_frames, int* out_T_enc,
                               int* out_d_model);

#ifdef __cplusplus
}
#endif
