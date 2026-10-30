#pragma once

// MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion.
// Architecture by ASLP@NPU (Apache-2.0).
//
// Pipeline:
//   1. Pretrained Streaming ASR (Fast-U2++ WeNet Conformer, 160ms/80ms):
//      Extracts 256-dim bottleneck features (BNFs) from 16 kHz input speech.
//   2. Speaker Embedding (WavLM Large + ECAPA-TDNN):
//      Extracts 256-dim speaker identity vector from reference voice.
//   3. Universal Timbre Token Encoder (UTTE):
//      - GlobalTimbreMemory (GTM): 32-slot memory bank with learnable priors.
//      - TemporalTimbreEncoder (TVT): frame-level cross-attention (BNF query x GTM KV).
//   4. Diffusion Transformer (DiT):
//      4 ChunkDiT blocks (dim=512, heads=2, FFN 1024) with Future-Receptive Chunking (FRC).
//      1-NFE Euler mean-flow solve: x_pred = x - u.
//   5. Neural Vocoder (Vocos):
//      8 ConvNeXt blocks (dim=320) + iSTFT head (640 n_fft, 160 hop) -> 16 kHz PCM waveform.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct meanvc2_context;

struct meanvc2_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
    int chunk_size; // 12 (120ms) or 4 (40ms)
    int block_size; // 4 (40ms lookahead)
    uint64_t seed;  // Flow matching noise RNG seed (default 42)
};

struct meanvc2_context_params meanvc2_context_default_params(void);

// Initialize context from unified GGUF model.
struct meanvc2_context* meanvc2_init_from_file(const char* path_model, struct meanvc2_context_params params);

// Set reference speaker voice from 16 kHz mono float PCM.
int meanvc2_set_speaker_audio(struct meanvc2_context* ctx, const float* audio_16k, int n_samples);

// Set 256-dim precomputed speaker embedding.
int meanvc2_set_speaker_embedding(struct meanvc2_context* ctx, const float* spk_emb_256);

// Convert source speech to target speaker voice. Returns 16 kHz mono float PCM. Caller frees with meanvc2_free_audio.
float* meanvc2_convert(struct meanvc2_context* ctx, const float* source_16k, int n_source_samples, int* out_n_samples);

// Diagnostic stage APIs for crispasr-diff harness
float* meanvc2_run_asr(struct meanvc2_context* ctx, const float* source_16k, int n_samples, int* out_frames,
                       int* out_dim);
int meanvc2_run_gtm(struct meanvc2_context* ctx, const float* spk_emb, float* out_k_mem, float* out_v_mem);
float* meanvc2_run_tvt(struct meanvc2_context* ctx, const float* bn_interp, int n_frames, const float* k_mem,
                       const float* v_mem);
float* meanvc2_run_dit(struct meanvc2_context* ctx, const float* bn_interp, int n_frames, const float* spk_emb,
                       const float* noise, int* out_mel_dim);
float* meanvc2_run_vocos(struct meanvc2_context* ctx, const float* mel_80_t, int n_frames, int* out_samples);

void meanvc2_free_audio(float* pcm);
void meanvc2_free(struct meanvc2_context* ctx);

#ifdef __cplusplus
}
#endif
