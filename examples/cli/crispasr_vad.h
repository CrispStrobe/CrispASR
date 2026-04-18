// crispasr_vad.h — audio segmentation for the unified crispasr CLI.
//
// Given a mono 16 kHz PCM buffer, produces a list of "audio slices" that
// downstream backends transcribe one at a time. Uses Silero VAD via the
// whisper_vad_* API when a VAD model path is provided, otherwise falls back
// to fixed-size chunking.
//
// Extracted from examples/parakeet-main/main.cpp (which had the cleanest
// implementation of this pattern).

#pragma once

#include <cstdint>
#include <vector>

struct whisper_params; // fwd decl

struct crispasr_audio_slice {
    int start, end;       // sample indices into the full PCM buffer
    int64_t t0_cs, t1_cs; // centiseconds, absolute start/end of the slice
};

// Stitched VAD result: VAD segments concatenated into a single buffer
// with 0.1s silence gaps (matching whisper.cpp's approach). The mapping
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

// Remap a centisecond timestamp from stitched-buffer space back to
// original-audio space using linear interpolation between mapping points.
int64_t crispasr_vad_remap_timestamp(const std::vector<crispasr_vad_mapping>& mapping, int64_t stitched_cs);

// Build the list of audio slices to transcribe.
//
// If params.vad_model is non-empty, runs Silero VAD and emits one slice per
// speech segment. Otherwise, if the audio is longer than chunk_seconds,
// splits it into fixed-size windows (encoders are O(T^2) in frame count, so
// unbounded audio is a memory/latency wall). A single short audio returns a
// single slice covering the whole buffer.
//
// chunk_seconds is the fallback window size for the no-VAD path.
//
// Returns an empty vector if VAD was requested but detected no speech.
std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(const float* samples, int n_samples, int sample_rate,
                                                                int chunk_seconds, const whisper_params& params);
