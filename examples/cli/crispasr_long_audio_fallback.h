// crispasr_long_audio_fallback.h — gate for the issue #89 long-audio chunking
// fallback.
//
// CAP_UNBOUNDED_INPUT backends (parakeet, canary, wav2vec2, ...) are
// mathematically able to take arbitrarily long audio in one encoder pass, but
// in practice the FastConformer encoder + TDT decoder break down past
// ~30-60 s in a single pass — per-feature z-normalization stats are
// computed across the whole input, the position encodings move outside the
// training distribution, and the TDT decoder stops emitting after a few
// seconds. On the issue #89 reporter's 300 s YouTube clip with no `--vad`
// and no `--chunk-seconds` this collapsed to 35 tokens covering only the
// first 4.8 s, with the rest silently dropped.
//
// Fall back to fixed chunking when the user provided no VAD and no
// `--chunk-seconds` and the audio is longer than the model's safe single-
// pass window. The overlap-save context that `use_chunk_context` adds in
// the per-slice loop covers chunk boundaries — see the issue #114 gate in
// `crispasr_chunk_context_gate.h` for the matching invariant.
//
// The threshold is 30 s — the per-feature z-norm statistics computed across
// the entire input start drifting from the training distribution past ~30 s,
// causing the TDT decoder to emit blanks and silently lose content.  The
// original 60 s threshold (30b47952) still lost the first ~58 s of each chunk
// on the issue #89 reporter's Vulkan hardware (parakeet-tdt-0.6b-ja on a 300 s
// clip: 4 words in the first 60 s).  30 s keeps z-norm close to the ~10-15 s
// utterances the model was trained on while still being long enough that short
// audio goes through the full-audio path.  Users who really want full-audio
// encoding on a long file can pass `--chunk-seconds 0` explicitly to opt out,
// or `--vad` for finer slicing.

#pragma once

#include <cstddef>
#include <cstdint>

namespace crispasr_long_audio {

// Backend capability flag — duplicated from crispasr_backend.h (kept in
// sync via the static_assert at the bottom of this file when both headers
// participate in the build). Duplicated to keep this header dependency-
// light for the unit test build.
constexpr uint32_t CAP_UNBOUNDED_INPUT_FLAG = 1u << 19;

// Returns true iff the caller should set `effective_chunk_seconds =
// kLongAudioFallbackChunkSeconds` to avoid encoding more than ~1 minute of
// audio in one pass. Inputs:
//   - effective_chunk_seconds : current chunk decision (0 = full-audio)
//   - wants_vad               : user passed --vad or --vad-model
//   - capabilities            : backend.capabilities() bitmask
//   - n_samples / sample_rate : audio length in samples + Hz
//   - threshold_seconds       : auto-fallback when audio > this (default 60)
inline bool should_auto_chunk_long(int effective_chunk_seconds, bool wants_vad, uint32_t capabilities, int n_samples,
                                   int sample_rate, int threshold_seconds) {
    if (effective_chunk_seconds > 0)
        return false; // explicit chunking already chosen
    if (wants_vad)
        return false; // VAD-derived slices are silence-bounded; encoder is fine
    if (!(capabilities & CAP_UNBOUNDED_INPUT_FLAG))
        return false; // bounded-input backends are not subject to this bug
    if (sample_rate <= 0 || threshold_seconds <= 0)
        return false;
    return (std::size_t)n_samples > (std::size_t)threshold_seconds * (std::size_t)sample_rate;
}

} // namespace crispasr_long_audio
