// voxtral4b.h — public C API for Mistral Voxtral-Mini-4B-Realtime-2602
//
// Audio-LLM with a RoPE+SwiGLU Whisper-style encoder + 4-frame stack
// projector + Llama 3 (Mistral) 3.4B LLM with adaptive RMSNorm and
// sliding window attention. Models loaded from GGUF files produced by:
//   `python models/convert-voxtral4b-to-gguf.py --input <hf_dir> --output X.gguf`
//
// Key differences from Voxtral-Mini-3B (voxtral.h):
//   - Encoder uses RoPE (not learned absolute pos embed)
//   - Encoder uses SwiGLU FFN (not GELU fc1/fc2)
//   - Encoder uses RMSNorm (not LayerNorm)
//   - Encoder + LLM use sliding window attention
//   - LLM has adaptive RMSNorm (time-conditioned)
//   - LLM uses tied embeddings (output = token_embd transposed)
//   - LLM: 26 layers, FFN=9216, RoPE θ=1e6

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct voxtral4b_context;

struct voxtral4b_context_params {
    int n_threads;
    int verbosity;   // 0=silent 1=normal 2=verbose
    bool use_gpu;    // false => force CPU backend
    bool flash_attn; // PLAN #89 plumbing — causal encoder + SWA
                     // decoder. Streaming-realtime path benefits
                     // most from kernel-level flash-attn (PLAN #86).
};

struct voxtral4b_context_params voxtral4b_context_default_params(void);

struct voxtral4b_context* voxtral4b_init_from_file(const char* path_model, struct voxtral4b_context_params params);

void voxtral4b_free(struct voxtral4b_context* ctx);

const uint8_t* voxtral4b_token_text(struct voxtral4b_context* ctx, int id, int* out_len);

int32_t* voxtral4b_tokenize(struct voxtral4b_context* ctx, const char* text, int* out_n_tokens);

float* voxtral4b_compute_mel(struct voxtral4b_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                             int* out_T_mel);

float* voxtral4b_run_encoder(struct voxtral4b_context* ctx, const float* mel_features, int n_mels, int T_mel,
                             int* out_N, int* out_dim);

float* voxtral4b_embed_tokens(struct voxtral4b_context* ctx, const int32_t* input_ids, int n_tokens);

bool voxtral4b_kv_init(struct voxtral4b_context* ctx, int max_ctx);
void voxtral4b_kv_reset(struct voxtral4b_context* ctx);

float* voxtral4b_run_llm_kv(struct voxtral4b_context* ctx, const float* inputs_embeds, int n_tokens, int n_past,
                            int* out_n_tokens, int* out_vocab_size);

// ── Native streaming (PLAN #7) ──────────────────────────────────────────────
// Synchronous-incremental streaming for the realtime model. Causal+SWA
// encoder + per-layer K/V cache + per-conv left-context state lets
// `feed()` process audio chunk-by-chunk without re-encoding. `flush()`
// runs the LLM prefill (prefix + accumulated audio embeds + suffix) and
// the greedy decode loop. Validated against single-shot `_transcribe`
// byte-for-byte (after SP `▁ → space` lstrip).
//
// PTT/dictation semantics: `feed` continuously, `flush` triggers the
// final decode and populates `get_text`. Live-during-speech captions
// are out of phase 1 scope (PLAN #7 phase 2).
//
// Audio length is bounded by the encoder's SWA window
// (`audio_swa = 750` frames = ~15s); attempts to feed beyond that are
// truncated at the front (oldest audio dropped).

struct voxtral4b_stream;

struct voxtral4b_stream* voxtral4b_stream_open(struct voxtral4b_context* ctx, int step_ms, int length_ms);

// `pcm` is float32 mono at 16 kHz. Returns 0 on success, <0 on error.
int voxtral4b_stream_feed(struct voxtral4b_stream* s, const float* pcm, int n_samples);

// Copies the un-read transcript text into `out` (NUL-terminated, truncated to `cap`-1).
// `out_t0_s` / `out_t1_s` give the cumulative window's audio time bounds.
// Returns 1 if there's text to read, 0 if not, <0 on error.
int voxtral4b_stream_get_text(struct voxtral4b_stream* s, char* out, int cap, double* out_t0_s, double* out_t1_s,
                              int64_t* out_decode_counter);

// Force a final decode over all audio fed so far. Idempotent. Returns 1
// on success (even if no new text), <0 on error.
int voxtral4b_stream_flush(struct voxtral4b_stream* s);

// Toggle live-captions decode-during-feed (PLAN #7 phase 3). When enabled,
// each new audio_embed produced during `feed` triggers one greedy decode
// step; tokens commit immediately to out_text and `get_text` returns
// progressive transcript. When disabled (default), decode is deferred to
// `flush`. Set BEFORE the first feed for clean semantics. Idempotent.
void voxtral4b_stream_set_live_decode(struct voxtral4b_stream* s, int enabled);

void voxtral4b_stream_close(struct voxtral4b_stream* s);

#ifdef __cplusplus
}
#endif
