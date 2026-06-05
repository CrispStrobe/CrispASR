// piper_tts.h — Piper VITS TTS native ggml runtime.
//
// rhasspy/piper VITS models: text encoder (6-layer transformer with
// relative position attention) + stochastic duration predictor +
// residual coupling flow (4 affine coupling blocks with WaveNet) +
// HiFi-GAN decoder (3 upsample stages + 9 resblocks).
//
// ~15-60 MB GGUF (Q4_K–F16) per voice, 22.05 kHz mono output.
// 250+ community voices across 30+ languages.
//
// Phonemizer: reuses the espeak-ng IPA integration from kokoro.cpp.
// The phoneme-to-id map is embedded in the GGUF as
// `piper.phoneme_id_map` JSON.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct piper_tts_context;

struct piper_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float noise_scale;  // variance of the noise in latent space (default from GGUF)
    float length_scale; // duration stretch factor (>1 = slower, <1 = faster)
    float noise_w;      // variance of the stochastic duration predictor noise
    int speaker_id;     // multi-speaker models: speaker index (0-based)
};

struct piper_tts_params piper_tts_default_params(void);

// Load a Piper VITS GGUF. Returns nullptr on failure.
struct piper_tts_context* piper_tts_init_from_file(const char* path_model, struct piper_tts_params params);

void piper_tts_free(struct piper_tts_context* ctx);

// Synthesize text to mono PCM at the model's sample rate.
// Returns number of samples written, 0 on failure.
// Caller owns the returned buffer (malloc'd; free with free()).
int piper_tts_synthesize(struct piper_tts_context* ctx, const char* text,
                         float** pcm_out,       // *pcm_out = malloc'd float32 buffer
                         int* sample_rate_out); // filled with the model's sample rate

// Synthesize from pre-phonemized IPA string (skip espeak-ng).
int piper_tts_synthesize_phonemes(struct piper_tts_context* ctx, const char* ipa_phonemes, float** pcm_out,
                                  int* sample_rate_out);

// Override the espeak-ng voice for phonemization.
void piper_tts_set_language(struct piper_tts_context* ctx, const char* espeak_voice);

// Runtime parameter setters.
void piper_tts_set_noise_scale(struct piper_tts_context* ctx, float noise_scale);
void piper_tts_set_length_scale(struct piper_tts_context* ctx, float length_scale);
void piper_tts_set_noise_w(struct piper_tts_context* ctx, float noise_w);
void piper_tts_set_speaker_id(struct piper_tts_context* ctx, int speaker_id);

// Query model info.
int piper_tts_sample_rate(const struct piper_tts_context* ctx);
int piper_tts_num_speakers(const struct piper_tts_context* ctx);
const char* piper_tts_espeak_voice(const struct piper_tts_context* ctx);

// Diff harness: dump all intermediates to this directory (empty = no dump).
void piper_tts_set_dump_dir(struct piper_tts_context* ctx, const char* dir);

#ifdef __cplusplus
}
#endif
