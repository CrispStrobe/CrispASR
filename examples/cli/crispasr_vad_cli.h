// crispasr_vad_cli.h — CLI-side VAD shim.
//
// The VAD algorithmic core (Silero segmentation, merge/split, stitching,
// timestamp remapping) lives in `src/crispasr_vad.h` so every CrispASR
// consumer reaches it through the shared library. This CLI-local header
// re-exports those types by transitive include and adds one wrapper
// that translates CLI `whisper_params` (including auto-download policy)
// into a library call.
//
// CLI callers: use `crispasr_compute_audio_slices(... whisper_params &)`.
// Library callers / wrappers: use `crispasr_compute_vad_slices` +
// `crispasr_fixed_chunk_slices` from `src/crispasr_vad.h` directly.

#pragma once

#include "crispasr_vad.h" // from src/ via whisper target's PUBLIC include dir

#include <string>
#include <vector>

struct whisper_params; // fwd decl

// Resolve the user-supplied VAD model path. When the user passed `--vad`
// without `--vad-model` (or with `--vad-model auto|default`), download
// the canonical ggml-silero-v6.3.0.bin into the crispasr cache dir on
// first use. Returns empty if VAD was not requested at all.
//
// Both the unified slicer and the whisper backend call this so users
// get the same auto-download UX whether their model uses whisper-internal
// VAD (crispasr's `wparams.vad_model_path`) or the unified slicer
// (which everything else uses).
std::string crispasr_resolve_vad_model(const whisper_params& p);

// The keyword values `--vad-model` accepts, in help-text order. SINGLE SOURCE
// for both the resolver's keyword test and the `--help` line: they are two
// lists that must agree, and they had already drifted — the help advertised
// only 'firered' and 'silero' while the resolver also accepted 'whisper-vad',
// 'marblenet', 'webrtc' and 'auto'/'default', so four working options were
// undiscoverable. Add a keyword here and in crispasr_resolve_vad_model().
inline const char* crispasr_vad_model_keywords() {
    return "auto, silero, firered, marblenet, webrtc, whisper-vad";
}

// Returns true if the resolved VAD model is a FireRedVAD model (not Silero).
// The whisper backend needs this to avoid passing FireRed GGUF to whisper's
// Silero-only VAD loader.
bool crispasr_vad_is_firered(const whisper_params& p);

// Returns true if the user selected WebRTC VAD (no model file, pure algorithmic).
// Like FireRed, this cannot be passed to whisper's internal Silero VAD loader.
bool crispasr_vad_is_webrtc(const whisper_params& p);

// Build the list of audio slices for a CLI invocation.
//
// If `params.vad` or `params.vad_model` is set, resolves the VAD model
// path (auto-downloading the canonical Silero GGUF into the CLI cache
// when the user passed `--vad` without `--vad-model`), then calls the
// library's `crispasr_compute_vad_slices`. Otherwise falls back to
// `crispasr_fixed_chunk_slices(chunk_seconds)`.
//
// Returns an empty vector if VAD was requested but detected no speech.
//
// When `out_vad_load_failed` is non-null it receives `true` iff a VAD model
// was requested but could not be loaded (vs. loaded-and-found-no-speech, which
// reports `false`). Issue #311 --strict-pipeline uses this to fail rather than
// silently fall through to fixed/energy chunking. It stays `false` when no VAD
// was requested.
std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(const float* samples, int n_samples, int sample_rate,
                                                                int chunk_seconds, const whisper_params& params,
                                                                bool* out_vad_load_failed = nullptr);
