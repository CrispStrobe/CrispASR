// crisp_audio.h — shared C++ audio-encoder library.
//
// Models multimodal audio embedding paths used in many speech models —
// BidirLM-Omni (CrispEmbed), Qwen3-ASR + Voxtral + Whisper-derivatives
// (CrispASR), and most other Conv-stem + Transformer-encoder topologies
// in current use.
//
// Architectural variation is handled by a small config (stem type, pos
// type, norm placement, projection head shape) read from the GGUF, so the
// same forward-pass code can serve different model dialects without each
// caller reimplementing it. The first supported dialect is the Conv2D-3x
// stem + sinusoidal pos + pre-LN encoder + 2-layer-GELU projection used
// by Qwen3-ASR and BidirLM-Omni; more dialects (Whisper-style Conv1D
// stem, learned pos, ModernBERT-style encoder, etc.) plug into the same
// API as separate config combinations.
//
// Input contract (regardless of dialect):
//   raw 16 kHz mono float32 PCM    →    (n_frames, output_dim) features
//
// All functions are thread-unsafe per context — wrap with a mutex in
// multi-threaded callers.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct crisp_audio_context;

// Architectural dialect — selects which forward graph crisp_audio_encode
// builds. Pinned by GGUF metadata (`crisp_audio.dialect`); the enum here
// is just the runtime decode of that string.
//
// Adding a new dialect = (a) add an enum value, (b) implement the matching
// graph builder in src/. Callers don't need to change.
enum crisp_audio_dialect {
    CRISP_AUDIO_DIALECT_AUTO = 0,      // resolve from GGUF
    CRISP_AUDIO_DIALECT_QWEN_OMNI = 1, // Conv2D-3x s=2 + sinusoidal + pre-LN + proj1/GELU/proj2
    // Future: CRISPASR_CLASSIC, MOONSHINE, MODERNBERT_AUDIO, ...
};

// Hyper-parameters caller can override at runtime. Static model params
// (d_model, n_layers, dialect, etc.) live in GGUF — caller does not pass
// them here.
struct crisp_audio_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;

    // Tensor-name prefix in the GGUF (e.g. "audio." for qwen3-asr,
    // "audio_tower." for BidirLM). NULL → "audio.".
    const char* tensor_prefix;

    // Metadata-key prefix for hparams (e.g. "qwen3asr.audio." or
    // "bidirlm.audio."). NULL → "crisp_audio.".
    const char* meta_prefix;

    // Optional dialect override. If CRISP_AUDIO_DIALECT_AUTO (default),
    // crisp_audio_init_from_file reads the dialect from GGUF metadata
    // (`<meta_prefix>dialect`).
    enum crisp_audio_dialect dialect;
};

struct crisp_audio_params crisp_audio_params_default(void);

// Load the audio encoder from a GGUF.
// Returns NULL on failure. Caller must free with crisp_audio_free().
struct crisp_audio_context* crisp_audio_init_from_file(const char* gguf_path, const struct crisp_audio_params* params);

void crisp_audio_free(struct crisp_audio_context* ctx);

// Compute the log-mel spectrogram from raw 16 kHz mono float32 PCM.
// Mel parameters (n_fft, hop_length, n_mels, normalization) are read
// from GGUF — the dialect picks Whisper-v3-style log-mel for the
// Qwen-Omni dialect.
//
// Returns malloc'd (n_mels, T_mel) row-major, or NULL on failure.
// *out_n_mels and *out_T_mel are set on return. Caller frees with free().
float* crisp_audio_compute_mel(struct crisp_audio_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                               int* out_T_mel);

// Run the full audio encoder forward pass on a precomputed log-mel.
// Output: malloc'd (n_frames, output_dim) row-major, or NULL on failure.
// *out_n_frames and *out_dim are set on return. Caller frees with free().
float* crisp_audio_encode(struct crisp_audio_context* ctx, const float* mel, int n_mels, int T_mel, int* out_n_frames,
                          int* out_dim);

// Read scalar hparams of the loaded model. Useful for caller sanity
// checks. Returns 0 if the field is not meaningful for the dialect.
int crisp_audio_d_model(struct crisp_audio_context* ctx);
int crisp_audio_output_dim(struct crisp_audio_context* ctx);
int crisp_audio_n_layers(struct crisp_audio_context* ctx);
int crisp_audio_n_window(struct crisp_audio_context* ctx);
enum crisp_audio_dialect crisp_audio_dialect_of(struct crisp_audio_context* ctx);

#ifdef __cplusplus
}
#endif
