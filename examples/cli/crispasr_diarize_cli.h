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

struct whisper_params; // fwd decl

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
bool crispasr_apply_diarize(const std::vector<float>& left, const std::vector<float>& right, bool is_stereo,
                            int64_t slice_t0_cs, std::vector<crispasr_segment>& segs, const whisper_params& params);
