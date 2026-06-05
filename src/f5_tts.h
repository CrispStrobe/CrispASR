// f5_tts.h — F5-TTS native ggml runtime.
//
// SWivid/F5-TTS: DiT-based flow matching TTS with zero-shot voice cloning.
// Architecture: ConvNeXtV2 text encoder + 22-layer DiT with AdaLN-Zero +
// Euler ODE solver (32 steps, CFG) + Vocos vocoder (iSTFT-based).
// ~330M params DiT + ~13M Vocos, 24 kHz mono output.
//
// Character-level tokenization (2545 vocab, pinyin for Chinese).
// Voice cloning via reference audio conditioning (mel spectrogram).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct f5_tts_context;

struct f5_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    int seed;           // random seed for noise generation (0 = non-deterministic)
    int ode_steps;      // number of Euler ODE steps (default from GGUF, typically 32)
    float cfg_strength; // classifier-free guidance strength (default from GGUF, typically 2.0)
    float sway_coef;    // sway sampling coefficient (default from GGUF, typically -1.0)
    float speed;        // speech speed factor (>1 = faster, <1 = slower, 1.0 = default)
};

struct f5_tts_params f5_tts_default_params(void);

// Load an F5-TTS GGUF (containing both DiT + Vocos weights). Returns nullptr on failure.
struct f5_tts_context* f5_tts_init_from_file(const char* path_model, struct f5_tts_params params);

void f5_tts_free(struct f5_tts_context* ctx);

// Set reference audio for voice cloning. The reference audio is 24 kHz mono PCM.
// Must be called before synthesize. Returns 0 on success, -1 on failure.
int f5_tts_set_reference(struct f5_tts_context* ctx, const float* pcm_24k, int n_samples, const char* ref_text);

// Synthesize text to mono 24 kHz PCM.
// Returns number of samples written, 0 on failure.
// Caller owns the returned buffer (malloc'd; free with free()).
int f5_tts_synthesize(struct f5_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out);

// Runtime parameter setters.
void f5_tts_set_seed(struct f5_tts_context* ctx, int seed);
void f5_tts_set_ode_steps(struct f5_tts_context* ctx, int steps);
void f5_tts_set_cfg_strength(struct f5_tts_context* ctx, float strength);
void f5_tts_set_speed(struct f5_tts_context* ctx, float speed);

// Query model info.
int f5_tts_sample_rate(const struct f5_tts_context* ctx);
int f5_tts_vocab_size(const struct f5_tts_context* ctx);

// Diff harness: dump intermediates to this directory (empty = no dump).
void f5_tts_set_dump_dir(struct f5_tts_context* ctx, const char* dir);

#ifdef __cplusplus
}
#endif
