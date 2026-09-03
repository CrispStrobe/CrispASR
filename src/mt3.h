#pragma once

// mt3.h — Magenta MT3 "Multi-Task Multitrack Music Transcription" backend
// (ISMIR 2021 / ICLR 2022, Apache-2.0, https://github.com/magenta/mt3)
//
// Multi-instrument audio → MIDI note events with a General-MIDI program per
// note. A plain T5 1.1-style encoder-decoder (8+8 layers, d_model 512, 6 heads
// of 64, gated-GELU FFN 1024) that reads a 512-bin log-mel spectrogram and
// emits a token stream in Magenta's event codec.
//
//   audio (16 kHz mono)
//     → zero-pad to a multiple of hop_width (128), chunk into 256-frame
//       segments (2.048 s); the STFT runs PER SEGMENT on the flattened 32768
//       samples, never once over the whole file
//     → tf.signal-compatible log-mel: n_fft 2048, hop 128, Hann *periodic*,
//       pad_end (no centering), MAGNITUDE (not power), HTK mel with
//       lo 20 Hz / hi 7600 Hz, **DC bin zeroed**, **unnormalised triangles**,
//       safe_log clamping only x <= 0                       → (256, 512)
//     → Dense(512→512) continuous-inputs projection + FIXED SINUSOIDAL
//       ABSOLUTE positions (mt3/layers.py:51-82)            → encoder
//     → greedy decode (≤ targets_length tokens) per segment → event tokens
//     → tie-section state machine + cross-segment note assembly → notes
//
// Two things separate this from src/t5_translate.cpp and both are silent
// when got wrong:
//
//   * MT3 has **no relative attention bias**. Positions are the FixedEmbed
//     sinusoidal table shipped in the GGUF as `pos_embd.weight`. The loader
//     hard-fails if `mt3.pos_embed` is not "sinusoidal", if
//     `mt3.use_relative_attention_bias` is nonzero, or if a `*.rel_bias.weight`
//     tensor is present — a half-taken branch loads, runs, and is wrong only
//     at positions > 0, which is the bug kunato/mt3-pytorch shipped.
//   * There is **no 1/sqrt(head_dim) rescale** (mt3/layers.py:230-234). The
//     GGUF records `mt3.attn_logit_scale` so the runtime never has to guess.
//
// GGUF produced by models/convert-mt3-to-gguf.py (arch "mt3").
// Reference oracle: tools/reference_backends/mt3.py.
// Constants and the tie state table: docs/music-transcription/mt3-port-notes.md.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mt3_context;

struct mt3_params {
    int n_threads;
    int verbosity; // 0=silent, 1=normal, 2=verbose (per-segment token dumps)
    bool use_gpu;

    // Greedy decode budget per 2.048 s segment. <= 0 → mt3.targets_length
    // (1024, the upstream cap). Lowering it truncates long segments.
    int max_decode_steps;

    // Stop after this many segments (<= 0 → the whole file). Debug lever;
    // production callers leave it at 0.
    int max_segments;
};

struct mt3_params mt3_default_params(void);

// One decoded note. `program` is the General-MIDI program number the model
// emitted (0-127); for a drum hit `is_drum` is true and `program` is 0 by
// upstream convention. `instrument` is the track index assigned by
// mt3/note_sequences.py:assign_instruments — programs numbered in
// first-appearance order skipping 9, and every drum note on 9 (GM channel 10).
struct mt3_note_event {
    float start_time; // seconds
    float end_time;   // seconds
    int pitch;        // MIDI pitch 0-127
    int velocity;     // 0-127 (the mt3 variant has 1 velocity bin → 100)
    int program;      // GM program 0-127
    int instrument;   // track index, drums = 9
    bool is_drum;
};

struct mt3_result {
    struct mt3_note_event* notes;
    int n_notes;

    // Decoding health, aggregated over segments. `n_invalid` counts events the
    // upstream state machine rejects and SKIPS (a documented, exercised path —
    // e.g. a tie section declaring a pitch that was never active), `n_dropped`
    // counts tokens abandoned when a segment's shift ran past the next
    // segment's start_time (max_decode_time).
    int n_segments;
    int n_tokens;
    int n_invalid;
    int n_dropped;

    // Tie-section evidence. `n_tie_ends` counts `tie` tokens the state machine
    // accepted (one per segment whose tie section was closed), `n_tied_pitches`
    // counts pitch declarations ACCEPTED inside a tie section — i.e. notes
    // carried across a segment boundary instead of being closed at it. Both
    // zero on a single-segment input; nonzero is proof the cross-segment path
    // actually ran.
    int n_tie_ends;
    int n_tied_pitches;
};

// Initialize from a GGUF file. Returns nullptr on a load or validation error.
struct mt3_context* mt3_init_from_file(const char* path, struct mt3_params params);

void mt3_free(struct mt3_context* ctx);

// Transcribe float32 mono PCM at 16 kHz.
// Returns 0 on success, nonzero on error. Free with mt3_result_free().
int mt3_transcribe(struct mt3_context* ctx, const float* pcm, int n_samples, struct mt3_result* result);

void mt3_result_free(struct mt3_result* result);

// Expected input sample rate (16000).
uint32_t mt3_sample_rate(const struct mt3_context* ctx);

// Samples per model segment (inputs_length * hop_width = 32768).
uint32_t mt3_segment_samples(const struct mt3_context* ctx);

// Per-stage parity against tools/reference_backends/mt3.py's ref.gguf
// (mel / enc_input / enc_out / logits_step0 / logits_prefix). `pcm_16k` must be
// the SAME 16 kHz mono signal the reference ran on.
// Returns 0 when every stage passes, 1 on a parity failure, 2 on a load error.
int mt3_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_16k, int n_samples, int verbosity);

#ifdef __cplusplus
}
#endif
