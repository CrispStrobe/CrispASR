#pragma once

// hft_transformer.h — hFT-Transformer piano transcription backend
// (Toyama, Akama, Ikemiya, Takida, Liao, Mitsufuji, "Automatic Piano
//  Transcription with Hierarchical Frequency-Time Transformer", ISMIR 2023;
//  MIT-licensed reference implementation sony/hFT-Transformer)
//
// Architecture:
//   Input: 16 kHz mono → STFT(n_fft = win = 2048, hop 256, periodic Hann,
//          CONSTANT (zero) pad) → |X|² (power 2.0) → mel(256 bins, 0–8000 Hz,
//          HTK scale, slaney filter norm) → log(mel + 1e-8)
//
//   The model answers 128 frames at a time from a 192-frame window: 128 new
//   frames plus 32 margin frames at each end. The margin outside the
//   recording is log(1e-8) and the tail is padded up to a multiple of 128.
//
//   ENCODER — one sequence per answered frame (batch 128, 256 freq tokens):
//       win = the 65 frames centred on the answered frame, per mel bin
//       Conv2d(1, 4, (1,5)) → flatten [4·61 = 244] → Linear(244, 256)
//         ... which this port FUSES into one Linear(65, 256); see below
//       × √256, + pos_embedding_freq[0:256]
//       3 × { x = LN(x + SelfAttn(x)); x = LN(x + FF(x)) }
//
//   DECODER-FREQ — 88 pitch tokens cross-attending to the 256 encoder tokens:
//       x = pos_embedding_freq[0:88]                    (no √256 here)
//       layer_zero : x = LN(x + CrossAttn(x, enc)); x = LN(x + FF(x))
//       2 ×        : x = LN(x + SelfAttn(x))
//                    x = LN(x + CrossAttn(x, enc))
//                    x = LN(x + FF(x))
//
//   DECODER-TIME — 128 time tokens (batch 88 pitches):
//       x = x·√256 + pos_embedding_time[0:128]
//       3 × { x = LN(x + SelfAttn(x)); x = LN(x + FF(x)) }
//       onset / offset / mpe = Linear(256, 1);  velocity = Linear(256, 128)
//
//   Attention is 4 heads × 64 with scale 1/8 throughout. Every `layer_norm`
//   is ONE module applied two or three times inside its layer — the
//   checkpoint carries a single set of gains per layer, not one per
//   application.
//
// Every head emits LOGITS; the sigmoid is applied here, and the velocity head
// is decoded by argmax over its 128 bins.
//
// THE FUSED FRONT END. The convolution and `tok_embedding_freq` are both
// linear in the 65-tap window with nothing between them, so the converter
// collapses them into a single Linear(65, 256). That is exact arithmetic
// (models/convert-hft-transformer-to-gguf.py --verify-fusion checks it), and
// it removes two im2col passes and 46k weights. The GGUF records
// `hft.front_end = "fused-conv-tok-embedding"` and this runtime refuses a
// file that says anything else.
//
// THE VELOCITY GATE IS NOT PLUMBING. The reference decoder runs
// `mode_velocity='ignore_zero'`, dropping any note whose velocity head reads
// zero at the onset frame. §35.4 of the flutter_tuner benchmark measured that
// gate as a BETTER precision filter than the onset threshold — 52.2% F1 with
// it against 52.1% for the best thresholded arm without it, while answering
// more often — and it is why hFT has no usable threshold lever at all.
// `ignore_zero_velocity` defaults to true and turning it off is a measurement,
// not a tuning knob.
//
// The whole model lives in ggml graphs; there is no hand-rolled recurrence
// here, unlike src/onsets_and_frames.cpp. The encoder and the frequency
// decoder are evaluated in frame chunks (HFT_FRAME_CHUNK) because nothing in
// either mixes across frames, so a chunk is bit-identical to the unchunked
// result and the attention score tensor — [256, 256, 4, chunk] — is what
// bounds peak memory.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hft_transformer_ctx;

struct hft_transformer_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose (keeps raw head outputs)

    // Honoured since the Metal wiring: true tries CUDA > Metal > Vulkan and
    // falls back to the CPU backend, false goes straight to the CPU. Default
    // false (see hft_transformer_default_params); the CLI passes its own
    // --no-gpu/--gpu-backend derived value. CRISPASR_HFT_NO_GPU=1 in the
    // environment forces CPU regardless, which is how the A/B arms are run.
    // n_threads is then ignored, since it only configures the CPU backend.
    bool use_gpu;

    // Post-processing, on sigmoid-ed activations. The reference defaults are
    // 0.5 for all three; §35.4 measured hFT as flat across 0.2–0.7 because of
    // the velocity gate below.
    float onset_threshold;
    float offset_threshold;
    float mpe_threshold;

    // `mode_velocity='ignore_zero'`. See the header comment: this is the
    // model's real precision filter.
    bool ignore_zero_velocity;

    // Frames per encoder/decoder-freq chunk. 0 = the built-in default (32).
    // Purely a memory/throughput knob; the result does not depend on it.
    int frame_chunk;
};

struct hft_transformer_params hft_transformer_default_params(void);

struct hft_transformer_note_event {
    float onset_time;  // seconds
    float offset_time; // seconds
    int midi_note;     // 21–108 (A0–C8)
    int velocity;      // 0–127, the velocity head's argmax bin
};

struct hft_transformer_result {
    hft_transformer_note_event* note_events;
    int n_notes;

    // Raw head outputs, (n_frames × 88) row-major. onset/offset/mpe are
    // post-sigmoid; velocity is the argmax bin as a float. Only populated
    // when verbosity >= 2. Freed by _result_free().
    float* onset_output;
    float* offset_output;
    float* mpe_output;
    float* velocity_output;
    int n_frames;
    int n_classes; // 88
};

// Initialize from a GGUF file. Returns nullptr on failure.
struct hft_transformer_ctx* hft_transformer_init_from_file(const char* path, struct hft_transformer_params params);

void hft_transformer_free(struct hft_transformer_ctx* ctx);

// Transcribe float32 mono PCM at 16 kHz. Returns 0 on success.
int hft_transformer_transcribe(struct hft_transformer_ctx* ctx, const float* pcm, int n_samples,
                               struct hft_transformer_result* result);

void hft_transformer_result_free(struct hft_transformer_result* result);

uint32_t hft_transformer_sample_rate(const struct hft_transformer_ctx* ctx);

// Diff-harness helper: the log-mel the model actually sees, (T, 256) row-major,
// BEFORE the margin padding. Returns a malloc'd buffer the caller frees with
// free(); *out_frames receives T.
float* hft_transformer_mel(struct hft_transformer_ctx* ctx, const float* pcm, int n_samples, int* out_frames);

#ifdef __cplusplus
}
#endif
