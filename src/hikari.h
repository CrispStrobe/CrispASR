#pragma once

// hikari.h — sbintuitions/hikari-medium: simultaneous speech translation
// (EN -> DE/JA/RU) and streaming English ASR. A Whisper-medium encoder-decoder
// with a causal encoder; the decoder emits one token per 80 ms of audio, where
// token 93 ("~") means WAIT. See src/hikari.cpp for the policy port.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hikari_context;

struct hikari_context_params {
    int n_threads;
    int verbosity; // 0 silent, 1 normal, 2 per-step trace
    bool use_gpu;
};

struct hikari_context_params hikari_context_default_params(void);
struct hikari_context* hikari_init_from_file(const char* path_model, struct hikari_context_params params);
void hikari_free(struct hikari_context* ctx);

// Task. translate=0: English transcription (tgt_lang ignored).
// translate=1: EN -> tgt_lang ("de", "ja", "ru"; "en"/"zh" are accepted by the
// upstream server but untrained). Returns 0, or -1 for an unknown language.
// Resets the stream.
int hikari_set_task(struct hikari_context* ctx, int translate, const char* tgt_lang);

// Read/write policy (upstream hikari-client UI defaults).
struct hikari_policy {
    float baseline_wait_penalty; // 0.0  subtracted from the WAIT logit
    float wait_penalty_boost;    // 0.6  added after 10 WAITs in a row during speech
    float wait_penalty_decay;    // 0.3  pull toward baseline on every emitted token
    float repetition_penalty;    // 40   on the arg-max if it is among the last 5 ids
    int decoder_context;         // 337  tokens (= 337 * 80 ms of audio), clamped [50, 375]
    float speech_threshold;      // 0.8  VAD probability for "speech"
};
struct hikari_policy hikari_default_policy(void);
void hikari_set_policy(struct hikari_context* ctx, const struct hikari_policy* p);

// Speech probability of `n` samples (the newest 512 of the window), used by
// the wait-penalty boost. NULL (default) = never speech = boost disabled —
// and then the model mostly WAITs: the boost is what makes it emit.
typedef float (*hikari_speech_prob_fn)(const float* samples, int n, void* user);
void hikari_set_speech_prob_fn(struct hikari_context* ctx, hikari_speech_prob_fn fn, void* user);

// Zeros appended by hikari_transcribe() so the lagging decoder can finish
// (a live microphone keeps delivering audio; a file ends). Default 2000 ms.
void hikari_set_tail_silence_ms(struct hikari_context* ctx, int ms);

// ── Streaming ──────────────────────────────────────────────────────────────
void hikari_stream_reset(struct hikari_context* ctx);
// Push 16 kHz mono PCM. One decision per complete 80 ms chunk. Returns the
// number of non-WAIT tokens emitted during this call, -1 on error.
int hikari_stream_push(struct hikari_context* ctx, const float* pcm, int n);
// Zero-pad a partial trailing chunk (as hikari-client does) and decide on it.
int hikari_stream_flush(struct hikari_context* ctx);
int hikari_stream_n_steps(struct hikari_context* ctx);
int32_t hikari_stream_step_token(struct hikari_context* ctx, int i);
// Seconds of audio the decision at step i had seen.
double hikari_stream_step_time(struct hikari_context* ctx, int i);
// Speech probability the step used (0 when no speech_prob_fn is set).
float hikari_stream_step_speech_prob(struct hikari_context* ctx, int i);
// Decoded text of everything emitted since reset (malloc'd, caller frees).
char* hikari_stream_text(struct hikari_context* ctx);
// Decoded text of one token id (WAIT/specials -> ""). malloc'd.
char* hikari_token_text(struct hikari_context* ctx, int32_t id);

// Offline: reset, push all, append tail silence, flush. malloc'd text.
char* hikari_transcribe(struct hikari_context* ctx, const float* pcm, int n);

// ── Diff hooks (crispasr-diff) — state of the LAST step ────────────────────
// Normalised log-mel of the last window, zero-padded to 3000 frames, (80, 3000)
// row-major. malloc'd.
float* hikari_debug_last_mel(struct hikari_context* ctx, int* n_mels, int* n_frames);
// Full-window encoder (1500 frames, padded region included) on that mel,
// (1500, d_model) row-major. Clobbers the stream caches. malloc'd.
float* hikari_debug_encoder_full(struct hikari_context* ctx, int* n_frames, int* d_model);
// Decoder logits for every position of the last step's ids, (n, vocab).
// Call before hikari_debug_encoder_full. malloc'd; *out_ids gets the ids.
float* hikari_debug_last_logits(struct hikari_context* ctx, int* n_pos, int* n_vocab, int32_t** out_ids);

// Bench summary (HIKARI_BENCH=1 prints it after hikari_transcribe).
void hikari_print_bench(struct hikari_context* ctx);

#ifdef __cplusplus
}
#endif
