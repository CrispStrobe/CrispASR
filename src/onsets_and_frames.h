#pragma once

// onsets_and_frames.h — Onsets & Frames piano transcription backend
// (Hawthorne et al., "Onsets and Frames: Dual-Objective Piano Transcription",
//  ISMIR 2018; MIT-licensed reimplementation jongwook/onsets-and-frames)
//
// Architecture:
//   Input: 16 kHz mono → STFT(n_fft = win = 2048, hop 512, periodic Hann,
//          reflect pad) → |X| (MAGNITUDE, power 1.0) → mel(229, 30–8000 Hz,
//          HTK scale, slaney filter norm) → log(clamp(·, 1e-5))
//   4× ConvStack, all four reading the same mel:
//       Conv2d(1,48,3×3,p1)+ReLU, Conv2d(48,48,3×3,p1)+ReLU, MaxPool(1,2),
//       Conv2d(48,96,3×3,p1)+ReLU, MaxPool(1,2),
//       flatten [T, 96·57 = 5472] → Linear(5472, 768)
//   onset_stack   : ConvStack → BiLSTM(768→384) → Linear(768,88)
//   offset_stack  : ConvStack → BiLSTM(768→384) → Linear(768,88)
//   frame_stack   : ConvStack → Linear(768,88)                  (the ACTIVATION)
//   velocity_stack: ConvStack → Linear(768,88)
//   combined_stack: cat(onset, offset, activation) [264]
//                   → BiLSTM(264→384) → Linear(768,88)          (the FRAME head)
//
// Every head emits LOGITS. The reference `infer.py` thresholds them at 0.5 as
// though they were probabilities, which is really a sigmoid threshold of 0.62;
// this runtime applies the sigmoid and thresholds at 0.5, so
// `onset_threshold`/`frame_threshold` here mean what they say.
//
// The ConvStack + Linear parts run in a ggml computation graph; the BiLSTM is
// computed by hand outside it, because ggml has no fused RNN op and unrolling
// thousands of timesteps into a graph would exhaust its node budget. That is
// the same split `src/piano_transcription.cpp` makes for Kong's BiGRU, and
// this file follows its conventions.
//
// The conv stack is evaluated in time chunks with a 3-frame halo, which is
// bit-identical to the unchunked result (a 3×3 kernel with pad 1 reaches one
// frame per layer and MaxPool(1,2) does not mix time at all) and bounds the
// im2col working set independently of the length of the piece.
//
// GGUF produced by models/convert-onsets-and-frames-to-gguf.py, which reads
// the ONNX export — BatchNorm is already folded into the convolutions there —
// and which resolves the export's shifted output names structurally. See that
// script's docstring; believing the names costs 9.1% → 5.6% note F1 with
// offsets required.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct onsets_and_frames_ctx;

struct onsets_and_frames_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose (keeps raw head outputs)

    // Honoured since the Metal wiring: true tries CUDA > Metal > Vulkan and
    // falls back to the CPU backend, false goes straight to the CPU. Default
    // false; the CLI passes its own --no-gpu/--gpu-backend derived value.
    // CRISPASR_OAF_NO_GPU=1 forces CPU regardless (the A/B control). Note the
    // BiLSTM stays on the host either way, so a GPU moves only the ConvStack
    // and the two GEMMs -- measured, not assumed, in
    // docs/music-transcription/PIANO_METAL_AB.md.
    bool use_gpu;

    // Post-processing, on sigmoid-ed activations.
    float onset_threshold; // default 0.5
    float frame_threshold; // default 0.5

    // Length of one forward pass, in seconds. 0 = the whole clip in one pass,
    // which is what the reference implementation does and what the ONNX
    // numbers this port is validated against were measured with. A positive
    // value processes the audio in segments with 50% overlap, keeping the
    // middle half of each, which bounds peak memory on very long inputs at the
    // cost of a colder LSTM at the seams.
    float segment_seconds;
};

struct onsets_and_frames_params onsets_and_frames_default_params(void);

struct onsets_and_frames_note_event {
    float onset_time;  // seconds
    float offset_time; // seconds
    int midi_note;     // 21–108 (A0–C8)
    int velocity;      // 0–127
};

struct onsets_and_frames_result {
    onsets_and_frames_note_event* note_events;
    int n_notes;

    // Raw sigmoid-ed head outputs, (n_frames × 88) row-major.
    // Only populated when verbosity >= 2. Freed by _result_free().
    float* onset_output;
    float* offset_output;
    float* frame_output;      // the combined stack — the REAL frame head
    float* activation_output; // frame_stack — the pre-combination head
    float* velocity_output;
    int n_frames;
    int n_classes; // 88
};

// Initialize from a GGUF file. Returns nullptr on failure.
struct onsets_and_frames_ctx* onsets_and_frames_init_from_file(const char* path,
                                                               struct onsets_and_frames_params params);

void onsets_and_frames_free(struct onsets_and_frames_ctx* ctx);

// Transcribe float32 mono PCM at 16 kHz. Returns 0 on success.
int onsets_and_frames_transcribe(struct onsets_and_frames_ctx* ctx, const float* pcm, int n_samples,
                                 struct onsets_and_frames_result* result);

void onsets_and_frames_result_free(struct onsets_and_frames_result* result);

uint32_t onsets_and_frames_sample_rate(const struct onsets_and_frames_ctx* ctx);

// Diff-harness helper: the log-mel the model actually sees, (T, 229) row-major.
// Returns a malloc'd buffer the caller frees with free(); *out_frames receives T.
float* onsets_and_frames_mel(struct onsets_and_frames_ctx* ctx, const float* pcm, int n_samples, int* out_frames);

// Per-stage cosine parity against a reference GGUF from
// tools/reference_backends/onsets_and_frames.py. Returns 0 on PASS, 1 if any
// stage falls below cos 0.999, 2 on a harness error. See the implementation's
// comment block in src/onsets_and_frames.cpp for the procedure.
int onsets_and_frames_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_16k, int n_samples,
                           int verbosity);

#ifdef __cplusplus
}
#endif
