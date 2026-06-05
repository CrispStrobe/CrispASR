// parler_tts.h -- Parler TTS public C ABI.
//
// Parler TTS: prompt-conditioned text-to-speech from
// parler-tts/parler-tts-mini-v1.1 (Apache 2.0).  Two text inputs:
//
//   1. Voice description (T5 encoder): "A female speaker with a warm
//      voice in a quiet room" -- selects the voice characteristics.
//   2. Text prompt (decoder input): "Hello, world!" -- the speech to
//      synthesize.
//
// Architecture:
//   T5 encoder (flan-t5-large) encodes the voice description.
//   MusicGen-style causal decoder with cross-attention on T5 output
//   generates 9 codebooks of DAC audio tokens using a delay pattern.
//   DAC 44 kHz codec decoder converts tokens to 44.1 kHz mono PCM.
//
// Single GGUF contains all three components (T5 encoder, decoder, DAC).
// Produced by models/convert-parler-to-gguf.py.
//
// Status: PLAN #137, initial skeleton.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct parler_tts_context;

struct parler_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;    // 0 = greedy (default 1.0 per upstream)
    uint64_t seed;        // RNG seed (0 = default)
    int max_audio_tokens; // upper bound on AR decode steps; 0 = built-in default (2580)
    int top_k;            // top-k sampling; 0 = disabled (default 0)
    bool flash_attn;
};

struct parler_tts_context_params parler_tts_context_default_params(void);

// Load the Parler TTS all-in-one GGUF (arch="parler-tts", produced by
// models/convert-parler-to-gguf.py).  Contains T5 encoder, decoder,
// and DAC codec weights.
struct parler_tts_context* parler_tts_init_from_file(const char* path_model, struct parler_tts_context_params params);

// Set the voice description for conditioning.  This runs the T5
// encoder on the description text and caches the encoder hidden states
// for use in cross-attention during generation.
//
// Must be called at least once before parler_tts_synthesize().
// Can be called again to change the voice.
//
// Returns 0 on success, non-zero on failure.
int parler_tts_set_description(struct parler_tts_context* ctx, const char* description);

// Synthesize text -> 44.1 kHz mono float32 PCM.
// Caller frees with parler_tts_pcm_free.  Returns nullptr on failure.
// The voice description must have been set via parler_tts_set_description().
float* parler_tts_synthesize(struct parler_tts_context* ctx, const char* text, int* out_n_samples);

// Run the decoder only, return raw DAC codebook indices.
// Output: (num_codebooks * T) interleaved codes.
// Caller frees with parler_tts_codes_free.
int32_t* parler_tts_synthesize_codes(struct parler_tts_context* ctx, const char* text, int* out_n);

void parler_tts_codes_free(int32_t* codes);
void parler_tts_pcm_free(float* pcm);
void parler_tts_free(struct parler_tts_context* ctx);
void parler_tts_set_n_threads(struct parler_tts_context* ctx, int n_threads);
void parler_tts_set_temperature(struct parler_tts_context* ctx, float temperature);
void parler_tts_set_seed(struct parler_tts_context* ctx, uint64_t seed);

#ifdef __cplusplus
}
#endif
