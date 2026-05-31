#pragma once

// Dia 1.6B TTS — public C ABI.
//
// Nari Labs Dia 1.6B (Apache 2.0): dialogue-style TTS with [S1]/[S2]
// speaker tags. 1.6B params, 44.1 kHz output via DAC codec.
//
// Architecture:
//   Text encoder: 12-layer Llama-style transformer (byte-level, 1024-d)
//   AR decoder: 18-layer transformer with cross-attention (2048-d, GQA 16q/4kv)
//   DAC codec: 9 codebooks -> 44.1 kHz PCM (shared with Zonos #130)
//
// Delay pattern for multi-codebook generation:
//   [0, 8, 9, 10, 11, 12, 13, 14, 15]
//   Channel k is delayed by delay[k] steps; after EOS on channel 0,
//   generation continues for max_delay (15) more steps to flush.
//
// Classifier-Free Guidance (CFG):
//   Batch size is always 2 (conditional + unconditional).
//   logits = uncond + cfg_scale * (cond - uncond)
//
// The DAC codec weights can be in the same GGUF or a separate file
// (--codec-model). When separate, the DAC GGUF from the Zonos port
// is fully compatible.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dia_tts_context;

struct dia_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // sampling temperature (default 1.2)
    float cfg_scale;   // CFG guidance scale (default 3.0)
    float top_p;       // nucleus sampling threshold (default 0.95)
    int top_k;         // top-k filtering (default 45)
    uint64_t seed;     // RNG seed (0 = non-deterministic)
    int max_tokens;    // max generation steps (0 = use default 3072)
    bool flash_attn;   // enable flash attention if available
};

struct dia_tts_context_params dia_tts_context_default_params(void);

// Initialize from the Dia GGUF (produced by convert-dia-to-gguf.py).
// If the GGUF contains DAC weights (audio_encoder.*), they are loaded
// automatically. Otherwise, call dia_tts_set_codec_path before synthesis.
struct dia_tts_context* dia_tts_init_from_file(const char* path_model, struct dia_tts_context_params params);

// Point the runtime at a separate DAC codec GGUF. Returns 0 on success.
int dia_tts_set_codec_path(struct dia_tts_context* ctx, const char* path);

// Synthesize text -> 44.1 kHz mono float32 PCM. The text may contain
// [S1] and [S2] speaker tags for dialogue generation. If no speaker
// tag is present, [S1] is prepended automatically.
//
// Caller frees the returned buffer with dia_tts_pcm_free.
// *out_n_samples is set to the sample count on success.
// Returns nullptr on failure.
float* dia_tts_synthesize(struct dia_tts_context* ctx, const char* text, int* out_n_samples);

void dia_tts_pcm_free(float* pcm);
void dia_tts_free(struct dia_tts_context* ctx);

void dia_tts_set_n_threads(struct dia_tts_context* ctx, int n_threads);
void dia_tts_set_temperature(struct dia_tts_context* ctx, float temperature);
void dia_tts_set_cfg_scale(struct dia_tts_context* ctx, float cfg_scale);
void dia_tts_set_seed(struct dia_tts_context* ctx, uint64_t seed);

#ifdef __cplusplus
}
#endif
