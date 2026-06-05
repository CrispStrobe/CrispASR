#pragma once

// IndexTTS-1.5 public C ABI.
//
// IndexTeam/IndexTTS (Apache-2.0) — a voice-cloning TTS pipeline:
//   1. Reference audio → 100-band log-mel → Conformer encoder → Perceiver resampler → 32 conditioning latents
//   2. Text → BPE tokenize (SentencePiece, 12000 vocab)
//   3. GPT-2 (24L, d=1280, 20 heads): [cond_latents | text_embs | start_mel] → autoregressive mel codes
//   4. Second GPT forward pass (return_latent) → hidden states for mel code positions
//   5. Reference mel → ECAPA-TDNN → 512-d speaker embedding
//   6. BigVGAN: GPT hidden states + speaker embedding → waveform @ 24kHz
//
// Two GGUFs required:
//   - indextts-gpt.gguf     (GPT-2 AR + Conformer encoder + Perceiver resampler)
//   - indextts-bigvgan.gguf (BigVGAN vocoder + ECAPA-TDNN speaker encoder)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct indextts_context;

struct indextts_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    float temperature;        // AR sampling temperature (default 0.8)
    float top_p;              // nucleus sampling (default 0.9)
    int top_k;                // top-k sampling (default 50)
    float repetition_penalty; // repetition penalty (default 1.2)
    int max_mel_tokens;       // upper bound on AR decode (default 1500)
    uint64_t seed;            // RNG seed (0 = use default)
};

struct indextts_context_params indextts_context_default_params(void);

// Initialise from the GPT GGUF (arch="indextts", produced by
// models/convert-indextts-to-gguf.py). Contains the GPT-2 AR model,
// Conformer conditioning encoder, and Perceiver resampler.
struct indextts_context* indextts_init_from_file(const char* path_model, struct indextts_context_params params);

// Point the runtime at the BigVGAN GGUF (arch="indextts-bigvgan").
// Contains the BigVGAN vocoder and ECAPA-TDNN speaker encoder.
// Required before the first indextts_synthesize call. Returns 0 on success.
int indextts_set_vocoder_path(struct indextts_context* ctx, const char* path);

// Synthesise text → 24 kHz mono float32 PCM using a reference audio clip
// for voice cloning. ref_pcm is mono 24kHz float32 PCM samples.
// Caller frees with indextts_pcm_free. *out_n_samples is set on success.
// Returns nullptr on error.
float* indextts_synthesize(struct indextts_context* ctx, const char* text, const float* ref_pcm, int ref_n_samples,
                           int* out_n_samples);

// Run only the GPT stage: text → mel codes. Caller frees with
// indextts_codes_free. *out_n is set to the number of mel code tokens.
// ref_pcm/ref_n_samples are optional (nullptr/0 → dummy conditioning).
int32_t* indextts_generate_mel_codes(struct indextts_context* ctx, const char* text, const float* ref_pcm,
                                     int ref_n_samples, int* out_n);

void indextts_codes_free(int32_t* codes);
void indextts_pcm_free(float* pcm);

void indextts_free(struct indextts_context* ctx);
void indextts_set_n_threads(struct indextts_context* ctx, int n_threads);
void indextts_set_seed(struct indextts_context* ctx, uint64_t seed);
void indextts_set_temperature(struct indextts_context* ctx, float temperature);

#ifdef __cplusplus
}
#endif
