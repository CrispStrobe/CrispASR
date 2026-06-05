// fastpitch_tts.h -- NVIDIA FastPitch TTS native ggml runtime.
//
// FastPitch: parallel (non-autoregressive) text-to-speech model.
// Architecture:
//   - Text encoder: N-layer bidirectional Transformer (FFTransformerEncoder)
//   - Duration predictor: TemporalPredictor (Conv1d stack + Linear projection)
//   - Pitch predictor: TemporalPredictor (Conv1d stack + Linear projection)
//   - Pitch embedding: Conv1d (pitch -> embedding space)
//   - Length regulator: repeat_interleave by predicted durations
//   - Mel decoder: N-layer Transformer (FFTransformerDecoder)
//   - Output projection: Linear -> n_mel_channels
//   - Speaker embedding: Embedding lookup (multi-speaker)
//   - HiFi-GAN vocoder: conv_pre + upsample MRF resblocks + conv_post
//
// Key difference from all other TTS backends: NO autoregressive loop.
// The entire mel spectrogram is generated in a single forward pass.
// Very fast inference.
//
// German multi-speaker model (5 speakers) from:
//   inOXcrm/German_multispeaker_FastPitch_nemo (Apache 2.0)
// Vocoder: tts_de_hui_hifigan_ft_fastpitch_multispeaker_5
//
// Section 133 in the CrispASR backend lineup.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct fastpitch_tts_context;

struct fastpitch_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    int speaker_id;    // speaker index for multi-speaker (0-based, default 0)
    float pace;        // speech speed (>1 = faster, <1 = slower, 1.0 = default)
    float pitch_shift; // additive pitch shift in Hz (0.0 = no shift)
};

struct fastpitch_tts_params fastpitch_tts_default_params(void);

// Load a FastPitch GGUF (containing FastPitch + HiFi-GAN weights).
// Returns nullptr on failure.
struct fastpitch_tts_context* fastpitch_tts_init_from_file(const char* path_model, struct fastpitch_tts_params params);

void fastpitch_tts_free(struct fastpitch_tts_context* ctx);

// Synthesize text to mono PCM at the model's sample rate (typically 22050 Hz).
// The text is expected to be pre-phonemized or raw text (model handles G2P).
// Returns number of samples written, 0 on failure.
// Caller owns the returned buffer (malloc'd; free with free()).
int fastpitch_tts_synthesize(struct fastpitch_tts_context* ctx, const char* text, float** pcm_out,
                             int* sample_rate_out);

// Runtime parameter setters.
void fastpitch_tts_set_speaker(struct fastpitch_tts_context* ctx, int speaker_id);
void fastpitch_tts_set_pace(struct fastpitch_tts_context* ctx, float pace);
void fastpitch_tts_set_pitch_shift(struct fastpitch_tts_context* ctx, float shift);

// Query model info.
int fastpitch_tts_sample_rate(const struct fastpitch_tts_context* ctx);
int fastpitch_tts_n_speakers(const struct fastpitch_tts_context* ctx);

#ifdef __cplusplus
}
#endif
