// vibevoice.h — Microsoft VibeVoice-ASR (σ-VAE tokenizers + Qwen2 LM).
//
// Architecture: Two ConvNeXt-style tokenizer encoders (acoustic + semantic)
// → linear connectors → Qwen2-1.5B autoregressive decoder.
// Input: raw 24kHz mono PCM. Output: structured text with timestamps.
// 1.5B params (ASR path), 4.7 GB F16, MIT license.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vibevoice_context;
struct vibevoice_stream;

struct vibevoice_context_params {
    int n_threads;
    int max_new_tokens;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    int tts_steps;   // DPM-Solver++ inference steps (default 20, min 4)
    uint32_t seed;   // RNG seed for TTS diffusion noise (0 = env/default)
    float cfg_scale; // TTS CFG guidance scale; 0 = model default (1.3 base,
                     // 3.0 realtime; upstream's realtime demo uses 1.5).
                     // Spontaneous BGM onsets are a documented model
                     // behavior (microsoft/VibeVoice FAQ, issue #171) —
                     // lowering cfg and/or changing --seed are the levers
                     // to re-roll them away.
    bool flash_attn; // PLAN #89 plumbing — σ-VAE encoder + Qwen2.5
                     // talker SA blocks.
};

struct vibevoice_context_params vibevoice_context_default_params(void);

struct vibevoice_context* vibevoice_init_from_file(const char* path_model, struct vibevoice_context_params params);

void vibevoice_free(struct vibevoice_context* ctx);

// Set an exact ASR generation cap. n <= 0 restores the duration-scaled
// default (at least 512 tokens, or 8 tokens per second of input audio).
void vibevoice_set_max_new_tokens(struct vibevoice_context* ctx, int n);
int vibevoice_resolve_max_new_tokens(int configured, int n_samples);

// Runtime setter for the DPM-Solver++ inference step count
// (default 20). Read on every vibevoice_synthesize call (line ~3422
// in vibevoice.cpp), so post-init mutation is safe. Clamps to
// [4, 100]: below 4 the schedule is degenerate, above 100 you're
// burning latency for inaudible quality gain.
void vibevoice_set_tts_steps(struct vibevoice_context* ctx, int steps);
void vibevoice_set_seed(struct vibevoice_context* ctx, uint32_t seed);
// CFG guidance scale for TTS synthesis (read per synthesize call).
// scale <= 0 restores the model default (1.3 base / 3.0 realtime).
void vibevoice_set_cfg_scale(struct vibevoice_context* ctx, float scale);

// Transcribe raw 24kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* vibevoice_transcribe(struct vibevoice_context* ctx, const float* samples, int n_samples);

// Variant of vibevoice_transcribe() that additionally splices free-form
// hotword/metadata text into the prompt. `context` may be NULL, or empty/
// whitespace-only, for the default prompt (identical output to
// vibevoice_transcribe()); matches `context_info` in microsoft/VibeVoice's
// vibevoice_asr_processor.py. A distinct symbol rather than a new parameter
// on vibevoice_transcribe() to avoid an ABI break: both are extern "C" and
// exported from libcrispasr.so, so changing an existing signature in place
// would silently corrupt any caller still built against the 3-arg form.
char* vibevoice_transcribe_with_context(struct vibevoice_context* ctx, const float* samples, int n_samples,
                                        const char* context);

// Variant that additionally returns per-emitted-token ids and softmax
// probabilities. Free with vibevoice_result_free.
struct vibevoice_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct vibevoice_result* vibevoice_transcribe_with_probs(struct vibevoice_context* ctx, const float* samples,
                                                         int n_samples);

// Context-aware counterpart of vibevoice_transcribe_with_probs(), see
// vibevoice_transcribe_with_context() above.
struct vibevoice_result* vibevoice_transcribe_with_probs_and_context(struct vibevoice_context* ctx,
                                                                     const float* samples, int n_samples,
                                                                     const char* context);
void vibevoice_result_free(struct vibevoice_result* r);

// Token-id → vocab piece (raw, with Qwen2/GPT-2 byte-level BPE markers
// like Ġ / Ċ — caller may need to decode). Returns empty string for
// out-of-range ids.
const char* vibevoice_token_text(struct vibevoice_context* ctx, int id);

// ── Stage-level API for differential testing ─────────────────────────────────

// Run the acoustic σ-VAE encoder. Returns a malloc'd float array of shape
// [*n_frames * *vae_dim] in row-major order (frame-major: data[t*vae_dim+c]).
// Caller frees with free(). Returns NULL on failure.
float* vibevoice_run_acoustic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                                      int* vae_dim);

// Run the semantic encoder. Same layout as acoustic. Returns NULL on failure.
float* vibevoice_run_semantic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                                      int* vae_dim);

// Run one SpeechConnector (FC1 → RMSNorm → FC2) on pre-computed encoder mean.
// prefix: "at_conn" (acoustic) or "se_conn" (semantic).
// encoder_mean: row-major [n_frames * vae_dim].
// Returns malloc'd float [n_frames * *d_lm]. Caller frees. Returns NULL on failure.
float* vibevoice_run_connector(struct vibevoice_context* ctx, const char* prefix, const float* encoder_mean,
                               int n_frames, int vae_dim, int* d_lm);

// Run both encoders + both connectors and return the combined speech features
// (element-wise sum of acoustic and semantic connector outputs).
// Returns malloc'd float [*n_frames * *d_lm]. Caller frees. Returns NULL on failure.
float* vibevoice_encode_speech(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                               int* d_lm);

// Returns true if the loaded model contains the acoustic (at_enc.*) and
// semantic (st_enc.*) tokenizer encoder tensors needed for transcription.
// TTS-only variants (vibevoice-realtime-0.5b etc.) return false.
bool vibevoice_has_asr(const struct vibevoice_context* ctx);

// Native VibeVoice-ASR-Streaming session. Input is 24 kHz mono PCM. The
// checkpoint-defined chunk and lookahead sizes are used; caller step/length
// settings do not change the model's receptive field. Each callback contains
// the newly completed text chunk. `flush` pads and processes a final partial
// window. Returns the number of chunks emitted, or a negative error code.
typedef void (*vibevoice_stream_callback)(const char* chunk_text, void* user_data);
struct vibevoice_stream* vibevoice_stream_open(struct vibevoice_context* ctx, const char* context);
int vibevoice_stream_feed(struct vibevoice_stream* stream, const float* samples, int n_samples, bool flush,
                          vibevoice_stream_callback callback, void* user_data);
void vibevoice_stream_reset(struct vibevoice_stream* stream);
void vibevoice_stream_free(struct vibevoice_stream* stream);
bool vibevoice_is_asr_streaming(const struct vibevoice_context* ctx);
int vibevoice_stream_chunk_samples(const struct vibevoice_context* ctx);
int vibevoice_stream_lookahead_samples(const struct vibevoice_context* ctx);

// ── TTS API (requires GGUF converted with --include-decoder) ─────────────────

// Synthesize speech from text. Returns malloc'd 24 kHz mono PCM float array.
// n_samples is set to the number of output samples. Caller frees with free().
// Returns NULL if the model lacks decoder tensors (vibevoice.has_decoder=0).
float* vibevoice_synthesize(struct vibevoice_context* ctx, const char* text, int* out_n_samples);

// Load a voice prompt GGUF for TTS. Returns 0 on success.
// The voice prompt pre-fills KV caches with speaker identity.
int vibevoice_load_voice(struct vibevoice_context* ctx, const char* voice_path);

#ifdef __cplusplus
}
#endif
