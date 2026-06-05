// crispasr_diarize_cli.h — CLI-side diarization shim.
//
// The four in-process diarization methods (energy, xcorr, vad-turns,
// pyannote-native) live in `src/crispasr_diarize.h` so every CrispASR
// consumer reaches them through the shared library. This CLI-local
// header keeps the subprocess-based sherpa-onnx method plus the
// `--diarize-method` → method-enum translation that CLI callers rely
// on.
//
// CLI callers: use `crispasr_apply_diarize(..., whisper_params &)`.
// Library callers / wrappers: use
// `crispasr_diarize_segments(..., CrispasrDiarizeOptions &)` from
// `src/crispasr_diarize.h` directly.

#pragma once

#include "crispasr_diarize.h" // from src/ via whisper target's PUBLIC include dir
#include "crispasr_backend.h"

#include <vector>

struct whisper_params;         // fwd decl
class CrispasrSpeakerEmbedder; // fwd decl (crispasr_speaker_embedder.h)

/// Cached pyannote-seg posteriors over a full audio buffer. Built once
/// at the start of a run (issue #107 — avoids per-slice pyannote runs
/// that reset local track indices across slices and produce
/// inconsistent speaker labels).
///
/// `log_probs` is row-major [T, 7] log-softmax over the seven
/// pyannote-seg classes (silence, spk0, spk1, spk0+1, spk2, spk0+2,
/// spk1+2). `frame_dur_s` is the audio duration of one frame
/// (270 / 16000 = 16.875 ms for pyannote-seg-3.0). The cache implicitly
/// covers absolute time [0, T*frame_dur_s); pass `slice_t0_cs` when
/// applying it to translate segment cs into frame indices.
struct CrispasrPyannoteCache {
    std::vector<float> log_probs;
    int T = 0;
    double frame_dur_s = 0.0;
    bool valid() const { return T > 0 && (int)log_probs.size() == T * 7 && frame_dur_s > 0.0; }
};

/// Compute pyannote-seg posteriors over `full_audio` (mono, 16 kHz) so
/// the same buffer can be reused for every per-slice diarize call.
/// `params.sherpa_segment_model` selects (or auto-downloads) the GGUF.
/// Returns true on success; `out` is populated with shape [T, 7].
bool crispasr_compute_pyannote_cache(const float* full_audio, int n_samples, const whisper_params& params,
                                     CrispasrPyannoteCache& out);

/// Cached global sherpa-onnx speaker-diarization timeline. Built once
/// over the full audio at the start of a run (issue #110 — avoids
/// per-slice sherpa invocations that reset local speaker IDs across
/// slices and produce inconsistent labels).
///
/// `segments` holds the globally-parsed speaker regions with absolute
/// timestamps. When this cache is passed to `crispasr_apply_diarize`,
/// the sherpa subprocess is NOT re-invoked; instead the per-slice
/// ASR segments are scored against the pre-computed global timeline.
struct CrispasrSherpaCache {
    struct Segment {
        double t0_s;
        double t1_s;
        int speaker;
    };
    std::vector<Segment> segments;
    bool valid() const { return !segments.empty(); }
};

/// Run sherpa-onnx-offline-speaker-diarization once over the full mono
/// 16 kHz audio and parse the global speaker-turn timeline.
/// Returns true on success; `out` is populated with speaker regions.
bool crispasr_compute_sherpa_cache(const float* full_audio, int n_samples, const whisper_params& params,
                                   CrispasrSherpaCache& out);

/// Top-level CLI diarize post-step.
///
/// Routes `params.diarize_method` to either the shared library methods
/// (energy / xcorr / vad-turns / pyannote) or the CLI-local sherpa-ONNX
/// subprocess fallback. Handles auto-download of the pyannote GGUF
/// when `--diarize-method pyannote` was passed without `--sherpa-segment-model`.
/// Mutates each `seg.speaker` in-place, formatting the result as
/// `"(speaker N) "` to match the historical crispasr convention.
///
/// `left` and `right` are per-channel slice buffers when stereo is
/// available. For mono input, both vectors point at the same data and
/// `is_stereo` is false; the dispatcher should call this anyway so the
/// mono-friendly methods (vad-turns, pyannote, sherpa) can still run.
///
/// `pyannote_cache` is optional. When non-null and valid AND the method
/// is `pyannote`, the cached posteriors are used instead of running
/// pyannote-seg again on this slice — required for cross-slice speaker
/// ID consistency (#107).
bool crispasr_apply_diarize(const std::vector<float>& left, const std::vector<float>& right, bool is_stereo,
                            int64_t slice_t0_cs, std::vector<crispasr_segment>& segs, const whisper_params& params,
                            const CrispasrPyannoteCache* pyannote_cache = nullptr,
                            const CrispasrSherpaCache* sherpa_cache = nullptr);

/// Re-label each segment's speaker by clustering speaker embeddings
/// extracted from `full_audio`. Operates over the WHOLE finalized
/// segment list (after per-slice diarize + segment splitting), so it
/// gives globally stable speaker IDs across the entire file.
///
/// `embedder` is the pluggable model (TitaNet today; any other adapter
/// later). `full_audio` is the mono 16 kHz buffer; `params` carries
/// the merge threshold and max-speakers settings. Segments shorter
/// than ~250 ms are skipped (embedders generally produce noisy
/// vectors below that), so their existing (pyannote-local) speaker
/// label is left in place.
///
/// No-op when `embedder` is null, `segs` is empty, or `full_audio` is
/// empty / too short — the pyannote-only labels survive unchanged in
/// those cases, which is what makes the system "work sufficiently
/// well without an embedder" (#107 P3).
void crispasr_remap_speakers_via_embeddings(std::vector<crispasr_segment>& segs, const float* full_audio, int n_samples,
                                            CrispasrSpeakerEmbedder* embedder, const whisper_params& params);
