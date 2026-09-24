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
    /// #324: WeSpeaker embeddings + spectral clustering (the FoxNose recipe).
    /// Needs `foxnose_embedder_path`. Unlike the other methods this one
    /// derives speaker TURNS from the audio and then attributes each caller
    /// segment to the turn it overlaps most, so it can split a single ASR
    /// segment's speaker assignment only at segment granularity.
    FoxNose,
    /// #466: NVIDIA Nemotron-3-Diarization (streaming Sortformer v3) - an
    /// end-to-end model giving per-10-ms activity for up to 8 speakers in
    /// arrival order. Needs `sortformer_model_path`. Like FoxNose it derives
    /// turns from the audio; each caller segment gets the speaker with the most
    /// probability mass inside it.
    Sortformer,
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

    // ── FoxNose (#324) ────────────────────────────────────────────────
    /// GGUF path for the speaker-embedding model (WeSpeaker ResNet34-LM).
    /// Ignored unless `method == FoxNose`; required when it is.
    std::string foxnose_embedder_path;
    /// Speaker-count bounds for automatic estimation.
    int min_speakers = 1;
    int max_speakers = 20;
    /// > 0 pins the speaker count and skips estimation entirely.
    int num_speakers = 0;

    // ── Sortformer (#466) ─────────────────────────────────────────────
    /// GGUF of Nemotron-3-Diarization (`sortformer` architecture). Required
    /// when `method == Sortformer`.
    std::string sortformer_model_path;
    /// Speaker-activity threshold for turns (the model card / transformers use 0.5).
    float sortformer_threshold = 0.5f;
    /// Chunk schedule: "offline" / empty (default) or a streaming preset,
    /// "low_latency", "very_low_latency", "ultra_low_latency" — the output of
    /// a live session with that latency. Empty falls back to
    /// CRISPASR_SORTFORMER_MODE, which is how C-ABI callers select it.
    std::string sortformer_mode;
};

/// A speaker turn derived from the audio, independent of the caller's
/// segmentation. Only FoxNose and Sortformer produce these; the others label
/// caller segments directly and leave the vector empty.
struct CrispasrDiarizeTurn {
    double start_s = 0.0; ///< relative to the sample buffer, not absolute
    double end_s = 0.0;
    int speaker = 0;
};

/// Run the selected diarizer over `segs`, mutating their `speaker` field.
/// `right` may alias `left` when `is_stereo == false`; methods that need
/// stereo data fall back to single-speaker labelling in that case.
///
/// Returns false only when the requested method needs a model that
/// failed to load (currently only Pyannote). All other methods always
/// succeed — they may leave `speaker = -1` when they have no information
/// to pick a label.
/// `out_turns`, when non-null, receives the speaker turns the method derived
/// from the audio (FoxNose only). Callers with word timestamps can use them to
/// split a segment that spans several speakers — labelling alone is limited to
/// the caller's own segment granularity.
bool crispasr_diarize_segments(const float* left, const float* right, int n_samples, bool is_stereo,
                               std::vector<CrispasrDiarizeSegment>& segs, const CrispasrDiarizeOptions& opts,
                               std::vector<CrispasrDiarizeTurn>* out_turns = nullptr);

/// Free the cached pyannote segmentation context (§176e). Call at shutdown
/// or when the model is no longer needed.
void crispasr_diarize_free_pyannote_cache();
