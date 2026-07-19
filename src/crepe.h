// crepe.h — CREPE monophonic F0 (pitch) estimation.
//
// Architecture: CREPE (Kim et al. 2018, MIT), via the torchcrepe weights.
//   Input:  16 kHz mono PCM -> 1024-sample frames, per-frame normalized
//   Body:   6 x [pad -> conv1d -> relu -> batchnorm -> maxpool(2)]
//           NOTE the relu is BEFORE the batchnorm; the converter therefore
//           ships BN as a standalone per-channel affine, never folded.
//   Head:   flatten (channel-fastest) -> Linear -> 360 bins -> sigmoid
//   Decode: bin -> cents (20 c/bin, offset 1997.3794084376191) -> Hz
//
// Two capacities: "full" (~44.5 MB f16) and "tiny" (~1.0 MB f16).
//
// GGUF: crepe-{full,tiny}-{f16,q8_0}.gguf, cstr/crepe-GGUF.
// Blueprint trace, geometry table and parity notes:
//   docs/music-transcription/PLAN.md
// Executable spec for this graph (keep in lockstep):
//   tools/crepe_numpy_parity.py

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct crepe_context;

// CREPE's fixed geometry — callers sizing buffers may rely on these.
#define CREPE_SAMPLE_RATE 16000
#define CREPE_WINDOW_SIZE 1024
#define CREPE_PITCH_BINS 360

// One frame of pitch output. Layout is deliberately identical to the
// CometBeat `PitchFrame` record ({double timeMs, double f0Hz, double
// voicedProb}) *in field order* so the Dart FFI binding is a straight
// reinterpret; note the C side is float, so the Dart side declares Float.
struct crepe_frame {
    float time_ms;     // frame centre, milliseconds from the start of the input
    float f0_hz;       // estimated fundamental
    float voiced_prob; // activation at the decoded bin, in [0, 1]
};

// Initialize CREPE from a GGUF model file. Returns NULL on failure.
struct crepe_context* crepe_init(const char* model_path, int n_threads);

// Free all resources.
void crepe_free(struct crepe_context* ctx);

// Number of frames `crepe_compute_f0` will produce for `n_samples` input at
// `hop_ms`. Lets the caller allocate exactly once. Returns 0 on bad args.
int crepe_n_frames(const struct crepe_context* ctx, int n_samples, float hop_ms);

// Estimate F0 over 16 kHz mono PCM.
//
// `hop_ms` defaults to 10.0 when <= 0 (CREPE's reference hop). The input is
// zero-padded by CREPE_WINDOW_SIZE/2 on both edges, matching torchcrepe's
// `pad=True`, so frame i is centred on sample i*hop.
//
// Writes at most `max_frames` entries to `out` and returns the number
// written, or 0 on error. Use `crepe_n_frames` to size `out`.
int crepe_compute_f0(struct crepe_context* ctx, const float* pcm_16k, int n_samples, float hop_ms,
                     struct crepe_frame* out, int max_frames);

// Raw 360-bin activation, for the diff harness and for callers that want to
// run their own decoder (e.g. Viterbi). Writes n_frames * CREPE_PITCH_BINS
// floats to `out` in frame-major order. Returns the frame count, or 0.
int crepe_compute_activation(struct crepe_context* ctx, const float* pcm_16k, int n_samples, float hop_ms, float* out,
                             int max_frames);

// Model capacity string baked into the GGUF ("full" or "tiny").
const char* crepe_capacity(const struct crepe_context* ctx);

#ifdef __cplusplus
}
#endif
