// speecht5_tts.h -- SpeechT5 TTS public C ABI.
//
// Microsoft SpeechT5 (MIT license): text encoder-decoder architecture
// that generates 80-bin mel spectrograms autoregressively, refined by a
// 5-layer conv post-net, then converted to 16 kHz PCM via HiFi-GAN.
//
// Forward pass:
//   1. Text encoder: Embedding + ScaledPosEnc + 12-layer transformer
//      with relative position bias -> encoder hidden states
//   2. Speech decoder (AR): prenet (2x Linear+ReLU + final_layer +
//      ScaledPosEnc + speaker projection) -> 6-layer decoder with
//      self-attn + cross-attn -> feat_out (Linear -> mel)
//      + prob_out (Linear -> stop probability)
//   3. Post-net: 5-layer Conv1d+BN+Tanh stack refining mel
//   4. HiFi-GAN vocoder: mel -> 16 kHz waveform
//
// Speaker conditioning: 512-dim x-vector normalized, expanded,
// concatenated with hidden states, projected via Linear(1280->768)+ReLU.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct speecht5_tts_context;

struct speecht5_tts_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float threshold; // stop token threshold (default 0.5)
    int max_len;     // max output length in reduction frames (0=auto, default ~4000/reduction)
    uint64_t seed;   // RNG seed for prenet dropout (0=deterministic, no dropout at inference)
};

struct speecht5_tts_params speecht5_tts_default_params(void);

// Load SpeechT5 TTS + HiFi-GAN from a single GGUF file.
// Returns nullptr on failure.
struct speecht5_tts_context* speecht5_tts_init(const char* path, struct speecht5_tts_params params);

// Set speaker embedding (512-dim x-vector). Must be called before synthesize.
// Returns 0 on success, -1 on failure.
int speecht5_tts_set_speaker(struct speecht5_tts_context* ctx, const float* xvector, int dim);

// Synthesize text to 16 kHz mono float32 PCM.
// Caller frees result with speecht5_tts_pcm_free().
// Returns nullptr on failure.
float* speecht5_tts_synthesize(struct speecht5_tts_context* ctx, const char* text, int* out_n_samples);

void speecht5_tts_pcm_free(float* pcm);

void speecht5_tts_free(struct speecht5_tts_context* ctx);

#ifdef __cplusplus
}
#endif
