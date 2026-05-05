// src/core/audio_chunking.h — boundary detection for long-form audio.
//
// Backends with bounded encoder windows (cohere = 30 s, parakeet/canary
// can also benefit) currently cut at exactly N * sample_rate samples,
// which slices mid-word and corrupts the transcript at chunk seams.
//
// `find_energy_min_split` scans a 1-D mono PCM segment in fixed-size
// non-overlapping windows and returns the start index of the lowest-RMS
// window — i.e. the quietest 100 ms within a search range. Cutting
// there avoids splitting a syllable in two and keeps the encoder /
// decoder operating on coherent acoustic boundaries.
//
// `split_at_energy_minima` wraps that into a chunker that yields
// [begin, end) sample ranges of length <= max_chunk_samples, choosing
// each cut from the last `boundary_context_samples` of the running
// window so we land in a quiet point near the cap. Mirrors the
// `_find_split_point_energy` / `split_audio_chunks_energy` helpers in
// nano-cohere-transcribe (Apache 2.0); ported to C++ for use across
// CrispASR's encoder backends.
#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace audio_chunking {

// Return the start index (relative to `samples[search_start]`'s base
// position in the caller's frame) of the lowest-RMS non-overlapping
// window of length `win_samples` inside [search_start, search_end).
//
// `search_end - search_start <= win_samples` is treated as a degenerate
// case and returns the midpoint of the search range.
inline size_t find_energy_min_split(const float* samples, size_t search_start, size_t search_end, size_t win_samples) {
    if (search_end <= search_start)
        return search_start;
    const size_t span = search_end - search_start;
    if (span <= win_samples)
        return search_start + span / 2;
    double min_e = std::numeric_limits<double>::infinity();
    size_t best = search_start;
    for (size_t i = 0; i + win_samples <= span; i += win_samples) {
        double s = 0.0;
        for (size_t j = 0; j < win_samples; ++j) {
            const float v = samples[search_start + i + j];
            s += (double)v * (double)v;
        }
        const double e = std::sqrt(s / (double)win_samples);
        if (e < min_e) {
            min_e = e;
            best = search_start + i;
        }
    }
    return best;
}

// Split `samples[0 .. n_samples)` into <= max_chunk_samples chunks,
// preferring cuts inside the last `search_window_samples` of each
// running window where RMS is lowest. Returns [begin, end) ranges in
// sample units. If the input is shorter than max_chunk_samples, returns
// a single [0, n_samples) range.
inline std::vector<std::pair<size_t, size_t>> split_at_energy_minima(const float* samples, size_t n_samples,
                                                                     size_t max_chunk_samples,
                                                                     size_t search_window_samples,
                                                                     size_t win_samples = 1600) {
    std::vector<std::pair<size_t, size_t>> out;
    if (n_samples == 0)
        return out;
    if (max_chunk_samples == 0 || n_samples <= max_chunk_samples) {
        out.emplace_back(0, n_samples);
        return out;
    }
    size_t idx = 0;
    while (idx < n_samples) {
        if (idx + max_chunk_samples >= n_samples) {
            out.emplace_back(idx, n_samples);
            break;
        }
        const size_t search_start =
            (search_window_samples >= max_chunk_samples) ? idx : idx + max_chunk_samples - search_window_samples;
        const size_t search_end = idx + max_chunk_samples;
        size_t cut = find_energy_min_split(samples, search_start, search_end, win_samples);
        // Defensive clamp: forward progress, never past the cap.
        if (cut <= idx)
            cut = idx + max_chunk_samples;
        if (cut > n_samples)
            cut = n_samples;
        out.emplace_back(idx, cut);
        idx = cut;
    }
    return out;
}

} // namespace audio_chunking
