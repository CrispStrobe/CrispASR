#pragma once

// MiMo-Audio-Tokenizer encoder (PCM → 8-channel RVQ codes) public C ABI.
//
// Loads cstr/mimo-tokenizer-GGUF (the upstream XiaomiMiMo/MiMo-Audio-Tokenizer
// stripped to encoder-only). Used by the MiMo-V2.5-ASR runtime in
// src/mimo_asr.cpp; not meant to be a standalone speech tokenizer model.
//
// Pipeline (see ref/mimo/github/src/mimo_audio_tokenizer/...):
//   16/24 kHz mono PCM
//     → log-mel spectrogram (n_fft=960, hop=240, n_mels=128, sr=24000)
//     → conv1 (Conv1d 128→1280, k=3, p=1) + GELU
//     → conv2 (Conv1d 1280→1280, k=3, s=2, p=1) + GELU
//     → 32-layer transformer (LN-pre, RoPE θ=10000, head_dim=64, GELU FFN,
//                             skip-add from layer 2 outputs after final layer)
//     → final LayerNorm
//     → down_sample (Conv1d 1280→1280, k=2, s=2, no bias) + GELU + LayerNorm
//     → ResidualVectorQuantizer (8-stage, codebook lookup) → (T, 8) int codes
//
// Output frame rate = 24000 / hop / stride / avg_pooler = 24000/240/2/2 = 25 fps.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mimo_tokenizer_context;

struct mimo_tokenizer_context_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose
    bool use_gpu;
};

struct mimo_tokenizer_context_params mimo_tokenizer_context_default_params(void);

// Initialise from the encoder GGUF file (cstr/mimo-tokenizer-GGUF).
// Returns nullptr on failure.
struct mimo_tokenizer_context* mimo_tokenizer_init_from_file(const char* path_model,
                                                             struct mimo_tokenizer_context_params params);

// Tokenise PCM (16 kHz mono float32) into 8-channel RVQ codes at 25 fps
// (after internal 16→24 kHz resample). On success returns a freshly-malloc'd
// int32 buffer of shape [n_frames * 8] (row-major, time-first), and writes
// `n_frames` to *n_frames_out. Caller must free() the returned pointer.
// Returns nullptr on failure.
int32_t* mimo_tokenizer_encode_pcm16k(struct mimo_tokenizer_context* ctx, const float* pcm, int n_samples,
                                      int* n_frames_out);

// Diff-harness stage extraction. Runs the encoder on the given 16 kHz PCM
// and returns a freshly-malloc'd float32 buffer holding the named stage
// tensor in the canonical (T, D) row-major layout matching the Python ref
// dump. Stage names match DEFAULT_STAGES in tools/reference_backends/mimo_tokenizer.py:
//   tok_mel       (T_mel,    128)
//   tok_conv1_out (T_mel,    1280)
//   tok_conv2_out (T_mel/2,  1280)
//   tok_xfmr_out  (T_mel/2,  1280)
//   tok_pool_out  (T_mel/4,  1280)
//   tok_codes     (T_mel/4,  8)     — int codes cast to float32
// Writes the total element count to *n_out. Caller must free().
// Returns nullptr if `stage` is unknown.
float* mimo_tokenizer_extract_stage(struct mimo_tokenizer_context* ctx, const float* pcm, int n_samples,
                                    const char* stage, int* n_out);

void mimo_tokenizer_free(struct mimo_tokenizer_context* ctx);

void mimo_tokenizer_set_n_threads(struct mimo_tokenizer_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif
