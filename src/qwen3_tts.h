#pragma once

// Qwen3-TTS public C ABI.
//
// Qwen/Qwen3-TTS is a "discrete multi-codebook LM" — a Qwen3
// backbone (28 layers, 16Q/8KV, head_dim 128) with a 16-codebook
// RVQ output head. The codec that turns codes back into 24 kHz
// waveform lives in the SEPARATE Qwen/Qwen3-TTS-Tokenizer-12Hz
// repo and gets its own context (loaded via
// `qwen3_tts_set_codec_path`).

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct qwen3_tts_context;

struct qwen3_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature; // 0 = greedy
    int   tts_steps;   // for codec decode (codec is non-DiT, but the
                       // codebooks may benefit from rejection sampling)
};

struct qwen3_tts_context_params qwen3_tts_context_default_params(void);

// Initialise from the talker LM GGUF file.
struct qwen3_tts_context* qwen3_tts_init_from_file(const char* path_model,
                                                   struct qwen3_tts_context_params params);

// Point the runtime at the codec GGUF (cstr/qwen3-tts-tokenizer-12hz-GGUF).
// Required before the first synthesis call. Returns 0 on success.
int qwen3_tts_set_codec_path(struct qwen3_tts_context* ctx, const char* path);

// Optional: reference-audio path (16 kHz mono WAV) for voice cloning
// (Base / CustomVoice variants). Pass nullptr / "" to use the model's
// built-in default voice.
int qwen3_tts_set_voice_prompt(struct qwen3_tts_context* ctx, const char* wav_path);

// Synthesise text → 24 kHz mono float32 PCM. Caller frees with
// `qwen3_tts_pcm_free`. *out_n_samples is set on success.
float* qwen3_tts_synthesize(struct qwen3_tts_context* ctx,
                            const char* text,
                            int* out_n_samples);

void qwen3_tts_pcm_free(float* pcm);

void qwen3_tts_free(struct qwen3_tts_context* ctx);

void qwen3_tts_set_n_threads(struct qwen3_tts_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
