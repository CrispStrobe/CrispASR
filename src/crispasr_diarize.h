// crispasr_diarize.h — shared speaker diarization post-step.
//
// Assigns a speaker index to each ASR segment, operating on either the
// original stereo L/R PCM (for methods that use channel cues) or a mono
// buffer (for methods that use time or acoustic cues only).
//
// Four in-process methods live here — the same ones the CLI's
// `--diarize-method` surface offered before this file existed:
//
//   * Energy    — stereo only. Compares |L| vs |R| per segment; the
//                 louder channel wins. Matches the historical
//                 crispasr `(speaker 0/1)` labelling.
//   * Xcorr     — stereo only. TDOA on L/R via cross-correlation with
//                 ±5 ms search window; the sign of the peak lag picks
//                 the channel.
//   * VadTurns  — mono-friendly. Alternates 0/1 every time the gap
//                 between adjacent ASR segments exceeds 600 ms.
//   * Pyannote  — mono-friendly, ML-based. Runs the GGUF-packed
//                 pyannote segmentation net from src/pyannote_seg.*
//                 and maps the 7-class posteriors onto up to three
//                 speakers per segment.
//
// Sherpa-ONNX diarization via subprocess still lives in the CLI
// (examples/cli/crispasr_diarize_cli.cpp). It shells out to an
// externally installed sherpa binary, which is CLI-shaped UX policy
// rather than a library responsibility.
//
// Shared by: the CLI, the C-ABI wrapper `crispasr_diarize_segments` in
// crispasr_c_api.cpp, and every language binding that calls through
// that wrapper.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class CrispasrDiarizeMethod {
    Energy,
    Xcorr,
    VadTurns,
    Pyannote,
};

// One ASR segment, in / out. Caller fills the centisecond range;
// diarization fills `speaker` with a zero-based index (-1 means
// "method couldn't decide — leave unlabelled").
struct CrispasrDiarizeSegment {
    int64_t t0_cs = 0; // in
    int64_t t1_cs = 0; // in
    int speaker = -1;  // out
};

struct CrispasrDiarizeOptions {
    CrispasrDiarizeMethod method = CrispasrDiarizeMethod::VadTurns;
    /// GGUF path for the Pyannote segmentation net. Ignored unless
    /// `method == Pyannote`. Must be a concrete file path — auto-
    /// download / cache is the caller's responsibility.
    std::string pyannote_model_path;
    /// Threads for pyannote inference. Ignored by the non-pyannote methods.
    int n_threads = 4;
    /// Absolute start (centiseconds) of the sample buffer within the
    /// original audio, so the lib can convert each segment's absolute
    /// t0/t1 into a buffer-relative sample index.
    int64_t slice_t0_cs = 0;
};

/// Run the selected diarizer over `segs`, mutating their `speaker` field.
/// `right` may alias `left` when `is_stereo == false`; methods that need
/// stereo data fall back to single-speaker labelling in that case.
///
/// Returns false only when the requested method needs a model that
/// failed to load (currently only Pyannote). All other methods always
/// succeed — they may leave `speaker = -1` when they have no information
/// to pick a label.
bool crispasr_diarize_segments(const float* left, const float* right, int n_samples, bool is_stereo,
                               std::vector<CrispasrDiarizeSegment>& segs, const CrispasrDiarizeOptions& opts);
