// crispasr_vad.h — shared VAD segmentation + stitching helpers.
//
// Consumed by both the CLI (`examples/cli/`) and the C-ABI wrapper
// `crispasr_session_transcribe_vad` in crispasr_c_api.cpp. The Silero
// VAD pipeline from crispasr is the underlying engine; this header
// adds the downstream gluing every ASR pipeline ends up needing:
//
//   - compute slices at speech boundaries
//   - merge short/close slices into usable chunks (ASR needs context)
//   - split overlong slices to bound encoder O(T^2) cost
//   - stitch the retained slices into one contiguous PCM buffer with
//     0.1s silence gaps (crispasr-style), so any backend can be
//     driven by a single transcribe() call
//   - remap resulting timestamps from stitched-buffer space back to
//     original-audio positions via linear interpolation
//
// The intent of living in `src/` rather than `examples/cli/` is DRY:
// Dart/Python/Rust wrappers reach this through the C-ABI without
// re-implementing segmentation or stitching in each binding.

#pragma once

#include <cstdint>
#include <vector>

struct crispasr_audio_slice {
    int start, end;       // sample indices into the full PCM buffer
    int64_t t0_cs, t1_cs; // centiseconds, absolute start/end of the slice
};

// Stitched VAD result: VAD segments concatenated into a single buffer
// with 0.1s silence gaps (matching crispasr's approach). The mapping
// table allows remapping timestamps from stitched-buffer positions back
// to original-audio positions.
struct crispasr_vad_mapping {
    int64_t stitched_cs; // position in the stitched buffer (centiseconds)
    int64_t original_cs; // corresponding position in the original audio
};

struct crispasr_stitched_audio {
    std::vector<float> samples;                // stitched PCM buffer
    std::vector<crispasr_vad_mapping> mapping; // timestamp remapping table
    int64_t total_duration_cs = 0;             // total duration in centiseconds
};

// Plain options struct — no CLI dependency. Mirrors the crispasr
// VAD tunables plus the chunk fallback used by long-audio encoders.
// Zero-initialising yields a sensible default configuration.
struct crispasr_vad_options {
    // whisper_vad_params
    float threshold = 0.5f;
    int min_speech_duration_ms = 250;
    int min_silence_duration_ms = 100;
    int speech_pad_ms = 30;
    // Post-VAD clean-up
    int chunk_seconds = 30; // split any merged slice longer than this; 0 = no split
    int n_threads = 4;      // VAD inference threads
};

// Load Silero VAD, emit one slice per speech segment, then merge
// short/adjacent slices and split overlong ones. Returns an empty
// vector if no speech is detected or the VAD model fails to load.
//
// `vad_model_path` must be a concrete file path; auto-download /
// cache resolution is the caller's responsibility (CLI handles it
// via crispasr_cache; wrappers can ship the GGUF as an asset).
std::vector<crispasr_audio_slice> crispasr_compute_vad_slices(const float* samples, int n_samples, int sample_rate,
                                                              const char* vad_model_path,
                                                              const crispasr_vad_options& opts);

// Same shape as above but without VAD: returns fixed `chunk_seconds` windows
// (one slice covering the whole buffer when it's shorter than a chunk).
// Useful as a fallback when no VAD model is available.
std::vector<crispasr_audio_slice> crispasr_fixed_chunk_slices(int n_samples, int sample_rate, int chunk_seconds);

// Stitch VAD slices into one contiguous buffer with 0.1s silence gaps.
// Produces a mapping table for timestamp remapping. The stitched buffer
// is shorter than the original audio (silence removed), allowing backends
// to process longer recordings without OOM.
crispasr_stitched_audio crispasr_stitch_vad_slices(const float* samples, int n_samples, int sample_rate,
                                                   const std::vector<crispasr_audio_slice>& slices);

// Remap a centisecond timestamp from stitched-buffer space back to
// original-audio space using linear interpolation between mapping points.
int64_t crispasr_vad_remap_timestamp(const std::vector<crispasr_vad_mapping>& mapping, int64_t stitched_cs);
