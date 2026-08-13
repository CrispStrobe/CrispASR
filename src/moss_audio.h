// moss_audio.h — public C API for the MOSS-Audio / MOSS-Music ggml runtime
//
// Audio understanding (ASR + audio QA + scene description) using a 32-layer
// Whisper-style encoder with DeepStack 3-tap cross-layer injection +
// Qwen3 LLM. Models loaded from GGUF files produced by:
//   `python models/convert-moss-audio-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Architecture: OpenMOSS-Team/MOSS-Audio-4B-Instruct (Apache-2.0);
// the same encoder/tap/adapter structure serves
// MOSS-Music-8B-Thinking (the adapter's llm.hidden_size is then 4096).
//   Audio encoder: 128-mel → 3×Conv2d(stride 2) → stem_proj → 32 WhisperEncoderLayer
//   DeepStack: taps at L8/L16/L24 → 3× GatedMLP → residual inject at LM L0/L1/L2
//   Audio adapter: GatedMLP(1280→8192→llm.hidden_size) for final encoder output
//   LM: 36-layer Qwen3 (d=llm.hidden_size, 32Q/8KV, head_dim=128, QK-norm,
//       SwiGLU, RoPE θ=1M)
//
// The GGUF kv `moss_audio.llm.hidden_size` is 2560 for MOSS-Audio-4B and 4096
// for MOSS-Music-8B-Thinking. It is never hardcoded here: load fails closed if
// it disagrees with the adapter's output-row shape.
//
// Ownership & threading contract (all stage helpers below):
//   * Returned buffers are malloc'd deep copies; the caller owns them and
//     must free() them. Nothing aliases ggml graph or backend memory.
//   * Copies are synchronous: the helper returns only after the stage is
//     fully computed and copied to the returned buffer.
//   * One context = one in-flight stage run. A context is not re-entrant and
//     not thread-safe: callbacks run synchronously on the caller thread and
//     must not re-enter the same context; the encoder's cached graph is
//     invalidated across invocations (see below), so a second call from a
//     callback or another thread corrupts state. Use one context per
//     concurrent consumer.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct moss_audio_context;

struct moss_audio_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
};

struct moss_audio_context_params moss_audio_context_default_params(void);

// Load model from GGUF.
struct moss_audio_context* moss_audio_init_from_file(const char* path_model, struct moss_audio_context_params params);

void moss_audio_free(struct moss_audio_context* ctx);

// Transcribe / understand audio. prompt is the text instruction (e.g.
// "Transcribe this audio." or "Describe the sounds in this clip.").
// Returns malloc'd UTF-8 string (caller owns).
char* moss_audio_process(struct moss_audio_context* ctx, const float* samples, int n_samples, const char* prompt);

// Convenience wrapper: transcribe with default prompt.
char* moss_audio_transcribe(struct moss_audio_context* ctx, const float* samples, int n_samples);

// ---- Stage helpers for differential testing ----

// Compute 128-bin log-mel spectrogram (Whisper-style).
// Output: malloc'd (n_mels, T_mel) F32 row-major. Caller frees.
// T_mel is the mel length the encoder consumes, padded to the 3000-frame
// (30s Whisper) convention.
float* moss_audio_compute_mel(struct moss_audio_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                              int* out_T_mel);

// Like moss_audio_compute_mel(), but also reports the PRE-PAD mel length.
// The returned tensor and *out_T_mel are identical to the plain variant;
// *out_T_mel_actual is the true frame count before the 3000-frame pad
// (0 means the input produced no frames). This is the only truthful way to
// distinguish real audio frames from frames created purely by the pad —
// T_mel_actual is observed during computation, never inferred from padded
// zeros or a global floor. Pass it straight to moss_audio_run_encoder_meta().
float* moss_audio_compute_mel_meta(struct moss_audio_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                                   int* out_T_mel, int* out_T_mel_actual);

// Run audio encoder only. Returns (T_enc, d_model=1280) F32 row-major.
// Also fills deepstack taps if ds_tap_0/1/2 are non-null (each T_enc × 1280).
// The encoder chunks the FULL mel buffer (typically the 3000-frame padded
// length passed as T_mel) into 400-frame pieces (3× stride-2 conv
// downsampling, 50 valid tokens per full chunk) and selects only valid
// output tokens — so for a padded input this includes the pad chunks, exactly
// as the Python reference does. Use moss_audio_run_encoder_meta() to get only
// real-content frames with truthful valid-frame metadata.
// On failure (NULL return) *ds_tap_0/1/2 are NOT modified: they never receive
// a dangling pointer. Check the return value before using or freeing taps.
float* moss_audio_run_encoder(struct moss_audio_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                              int* out_d, float** ds_tap_0, float** ds_tap_1, float** ds_tap_2);

// Pure valid-frame bookkeeping for the encoder chunk loop — no model state.
// chunk_frames = 400; each chunk's valid token count is 3× stride-2 conv
// downsampling of its real length. Returns the number of chunks (0 when
// T_mel <= 0). When valid_counts is non-null it must have capacity at least
// (T_mel + 399) / 400; the first num_chunks entries are filled with per-chunk
// valid counts. *out_total_valid receives sum(valid_counts). This is the ONLY
// source of per-chunk valid counts for both moss_audio_run_encoder and
// moss_audio_run_encoder_meta, so their metadata can never drift from the
// counts the chunk loop actually computes.
int moss_audio_plan_chunks(int T_mel, int* valid_counts, int* out_total_valid);

// Metadata-returning encoder entrypoint. Same 1280-D encoder and DeepStack
// taps 8/16/24 as moss_audio_run_encoder(), but it EXECUTES AND TRIMS using
// exactly T_mel_actual: only real-content frames are computed and returned.
// The 3000-frame Whisper pad never contributes a frame, an attention key, or
// a valid count — consumers need no post-hoc cutting.
//
//   mel, n_mels, T_mel : the padded (n_mels, T_mel) mel buffer as returned by
//                        moss_audio_compute_mel_meta() — T_mel is the buffer's
//                        row stride (typically the padded 3000).
//   T_mel_actual       : the PRE-PAD content length (the same value
//                        moss_audio_compute_mel_meta() reports). Validated to
//                        1 <= T_mel_actual <= T_mel; anything else fails closed
//                        (returns NULL). Never inferred from padded zeros or a
//                        global floor.
//   valid_counts       : caller-allocated int array with capacity at least
//                        ceil(T_mel_actual / 400). Filled with the per-chunk
//                        valid-frame counts of the REAL chunks; may be NULL.
//   *out_num_chunks    : == ceil(T_mel_actual / 400), the number of valid_counts
//                        entries written.
//   *out_total_valid   : == sum(valid_counts) == *out_T_enc.
//   *out_T_enc         : real-content frame count, == out_total_valid.
//   *out_d             : encoder dim (1280).
//   *out_T_mel_actual  : receives the validated T_mel_actual actually used.
//   ds_tap_0/1/2       : real-content DeepStack taps, each *out_T_enc × 1280.
//
// When the input is genuinely >= 30 s (T_mel_actual == T_mel, no pad), this is
// byte-identical to moss_audio_run_encoder(). For shorter inputs it differs by
// design: run_encoder preserves the reference's padded full-length behaviour,
// while this entrypoint returns only content.
// On failure (NULL return, including fail-closed T_mel_actual validation)
// *ds_tap_0/1/2 are NOT modified: they never receive a dangling pointer.
float* moss_audio_run_encoder_meta(struct moss_audio_context* ctx, const float* mel, int n_mels, int T_mel,
                                   int T_mel_actual, int* out_T_enc, int* out_d, int* valid_counts, int* out_num_chunks,
                                   int* out_T_mel_actual, int* out_total_valid, float** ds_tap_0, float** ds_tap_1,
                                   float** ds_tap_2);

// Run audio adapter on encoder output. Returns (T_enc, llm_dim) F32 where
// llm_dim is the adapter output-row count read from the GGUF weights
// (== moss_audio.llm.hidden_size; 2560 for MOSS-Audio-4B, 4096 for
// MOSS-Music-8B-Thinking). *out_d reports that same weight-derived value.
float* moss_audio_run_adapter(struct moss_audio_context* ctx, const float* encoder_out, int T_enc, int d_enc,
                              int* out_T, int* out_d);

// Embed tokens. Returns (n_tokens, llm_dim) F32.
float* moss_audio_embed_tokens(struct moss_audio_context* ctx, const int32_t* token_ids, int n_tokens);

// Initialize / reset KV cache for LLM decode.
bool moss_audio_kv_init(struct moss_audio_context* ctx, int max_ctx);
void moss_audio_kv_reset(struct moss_audio_context* ctx);

// Run LLM with KV cache. Returns logits (vocab_size,) F32 for last token.
float* moss_audio_run_llm_kv(struct moss_audio_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                             int* out_n_tokens, int* out_vocab_size);

// Tokenize text using BPE.
int moss_audio_tokenize(struct moss_audio_context* ctx, const char* text, int32_t* out_tokens, int max_tokens);

// Token ID → string.
const char* moss_audio_token_text(struct moss_audio_context* ctx, int token_id);

// Seed for sampling.
void moss_audio_set_seed(struct moss_audio_context* ctx, uint32_t seed);

// Beam search. 1 = greedy (default). >1 = beam search via
// core_beam_decode replay-from-prefix (§167g).
void moss_audio_set_beam_size(struct moss_audio_context* ctx, int beam_size);

// Per-token streaming callback. Fires once per generated token (id, softmax prob, userdata).
typedef void (*moss_audio_token_cb)(int tok_id, float prob, void* userdata);

// Like moss_audio_process() but fires cb(tok_id, prob, userdata) for each generated token.
// The final assembled text is NOT returned; all output is via the callback.
// Falls back to greedy decode (beam_size ignored).
void moss_audio_process_cb(struct moss_audio_context* ctx, const float* samples, int n_samples, const char* prompt,
                           moss_audio_token_cb cb, void* userdata);

#ifdef __cplusplus
}
#endif
