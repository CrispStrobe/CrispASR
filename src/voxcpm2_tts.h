// voxcpm2_tts.h — public C API for openbmb/VoxCPM2 TTS
//
// VoxCPM2 is a tokenizer-free diffusion autoregressive TTS model with:
//   - TSLM (Text-Semantic LM): 28-layer MiniCPM-4 with LongRoPE, GQA (16h/2kv)
//   - RALM (Residual Acoustic LM): 8-layer MiniCPM-4, no RoPE
//   - LocEnc (Local Encoder): 12-layer bidirectional transformer
//   - LocDiT (Local DiT): 12-layer bidirectional transformer + CFM Euler solver
//   - AudioVAE V2: causal conv encoder/decoder, 16kHz→48kHz with SR conditioning
//   - FSQ (Finite Scalar Quantization): bottleneck between TSLM and RALM
//
// Models loaded from GGUF files produced by:
//   `python models/convert-voxcpm2-to-gguf.py --input openbmb/VoxCPM2 --output X.gguf`
//
// Pipeline: text → tokenize → [ref_encode] → TSLM prefill → AR loop:
//   TSLM step → FSQ → RALM step → project → DiT CFM solve → next patch
//   → stop? → VAE decode → 48kHz PCM

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct voxcpm2_context;

struct voxcpm2_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    bool flash_attn;
    int inference_steps; // CFM Euler steps, default 10 (tradeoff: quality vs speed)
    float cfg_value;     // Classifier-free guidance scale, default 2.0
    int max_len;         // Max AR steps, default 2000
};

struct voxcpm2_context_params voxcpm2_context_default_params(void);

// Load VoxCPM2 GGUF. Returns nullptr on failure.
struct voxcpm2_context* voxcpm2_init_from_file(const char* path_model, struct voxcpm2_context_params params);

void voxcpm2_free(struct voxcpm2_context* ctx);

void voxcpm2_set_n_threads(struct voxcpm2_context* ctx, int n_threads);

// Synthesize text → 48 kHz mono float32 PCM (zero-shot, no voice cloning).
// Returns malloc'd buffer; caller frees with voxcpm2_pcm_free.
// *out_n_samples set on success; returns nullptr on failure.
float* voxcpm2_synthesize(struct voxcpm2_context* ctx, const char* text, int* out_n_samples);

// Synthesize with voice cloning from reference audio.
// ref_samples: 16 kHz mono float32 PCM reference audio.
// ref_n_samples: number of samples in ref_samples.
float* voxcpm2_synthesize_clone(struct voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                int ref_n_samples, int* out_n_samples);

// Streaming synthesis — generates and yields audio chunk by chunk.
struct voxcpm2_stream;

struct voxcpm2_stream* voxcpm2_stream_open(struct voxcpm2_context* ctx, const char* text, const float* ref_samples,
                                           int ref_n_samples);

// Get next audio chunk. Returns nullptr when done.
// Each chunk is one VAE decode unit (~decode_chunk_size samples at 48kHz).
// Caller does NOT free the returned pointer — it's owned by the stream.
const float* voxcpm2_stream_next(struct voxcpm2_stream* stream, int* out_n_samples);

void voxcpm2_stream_close(struct voxcpm2_stream* stream);

void voxcpm2_pcm_free(float* pcm);

// Diff-harness stage extractor. Returns malloc'd float[*out_n].
// Stage names: "text_input_ids", "locenc_out", "tslm_prefill_out",
// "ralm_prefill_out", "cfm_step0_result", "decoded_audio", etc.
float* voxcpm2_extract_stage(struct voxcpm2_context* ctx, const char* text, const float* ref_samples, int ref_n_samples,
                             const char* stage_name, int* out_n);

#ifdef __cplusplus
}
#endif
