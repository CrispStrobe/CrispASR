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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
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

// Peak absolute amplitude over a mono PCM span. 0 for an empty span.
inline float peak_abs(const float* samples, size_t n_samples) {
    float peak = 0.0f;
    for (size_t i = 0; i < n_samples; i++) {
        const float a = samples[i] < 0.0f ? -samples[i] : samples[i];
        if (a > peak)
            peak = a;
    }
    return peak;
}

// Is this span DIGITALLY silent — i.e. carrying no signal at all?
//
// Motivation: an encoder-decoder ASR model handed a span with nothing in it
// does not return nothing, it returns invented speech. Cohere Transcribe turns
// 10 s of zeros into "And I'm going to go ahead and do that.", and a long file
// whose trailing chunk is all zeros gets that sentence appended to an otherwise
// perfect transcript.
//
// The default epsilon is deliberately BELOW one int16 LSB (1/32768 = 3.05e-5),
// so a single non-zero int16 sample anywhere disables the gate. That is the
// point: this must never silence real audio. Measured for scale — the quietest
// real speech to hand (a FLEURS clip) peaks at 0.038, i.e. ~3800x this
// threshold, and low-level noise at ~0.0018 does not provoke the model anyway.
// So the gate covers exactly the observed failure (true digital silence) and
// declines to guess about anything else.
inline bool is_digitally_silent(const float* samples, size_t n_samples, float eps = 1e-5f) {
    if (n_samples == 0)
        return true;
    return peak_abs(samples, n_samples) < eps;
}

// ---------------------------------------------------------------------------
// Speechless-slice gate for the VAD-free energy chunker (issue #471).
//
// split_at_energy_minima cuts each window at its quietest 100 ms. When speech
// stops a few seconds before a window boundary and the audio runs past it, that
// quietest spot is the silence after the last word, so the remainder becomes a
// slice with no speech in it. An LLM backend handed that slice does not return
// nothing: qwen3 returned the --hotwords list verbatim, or invented a sentence
// ("Okay, so we're gonna talk about the benefits of having a pet."). The VAD
// path already drops speech-free audio (#213); this is the energy-path twin.
//
// Why not "don't cut, let the last slice run to the end" (the issue's other
// suggestion): that makes a slice longer than max_chunk_samples, and several
// callers pass a hard encoder cap there (the JA parakeet path: ~12 s). Dropping
// the speech-free slice keeps every slice within the cap and yields the same
// text, because the kept slice already holds all of the speech.
//
// Why not digital zero only (is_digitally_silent): the reporter saw a 0.3 s
// slice of real ROOM NOISE return the whole hotword list. So a slice is judged
// against the recording's OWN levels, measured over 100 ms windows:
//   floor = 10th percentile of window RMS (windows that carry any signal)
//   loud  = 95th percentile of window RMS
// A slice is speechless when its LOUDEST window stays within
//   min(4 * floor, floor + 0.2 * (loud - floor))
// i.e. both within 12 dB of the noise floor AND in the bottom fifth of the
// recording's dynamic range. Both must hold, which makes it conservative:
//   - noise-only audio (loud ~ floor) leaves the threshold just above the floor,
//     so noise fluctuations keep their slices — a file with nothing louder than
//     its noise is never emptied by this gate;
//   - a quiet talker in a loud file is > 12 dB above the floor, so kept;
//   - a speech slice's loudest 100 ms is near `loud`, far above either bound.
// A slice that is digitally silent (see is_digitally_silent) is always
// speechless. Slice boundaries are never moved, so audio with speech in every
// slice is chunked exactly as before.
// ---------------------------------------------------------------------------

struct energy_levels {
    float floor = 0.0f; // 10th-percentile 100 ms RMS over windows with signal
    float loud = 0.0f;  // 95th-percentile 100 ms RMS over windows with signal
};

inline float span_rms(const float* samples, size_t n_samples) {
    if (n_samples == 0)
        return 0.0f;
    double s = 0.0;
    for (size_t i = 0; i < n_samples; ++i)
        s += (double)samples[i] * (double)samples[i];
    return (float)std::sqrt(s / (double)n_samples);
}

// Per-window RMS over non-overlapping `win_samples` windows. A trailing partial
// window is kept when it is at least a quarter window long, or when it is the
// only window (a span shorter than one window).
inline std::vector<float> window_rms_list(const float* samples, size_t n_samples, size_t win_samples) {
    std::vector<float> out;
    if (n_samples == 0)
        return out;
    if (win_samples == 0 || n_samples <= win_samples) {
        out.push_back(span_rms(samples, n_samples));
        return out;
    }
    size_t i = 0;
    for (; i + win_samples <= n_samples; i += win_samples)
        out.push_back(span_rms(samples + i, win_samples));
    const size_t rest = n_samples - i;
    if (rest > 0 && rest * 4 >= win_samples)
        out.push_back(span_rms(samples + i, rest));
    return out;
}

inline energy_levels estimate_energy_levels(const float* samples, size_t n_samples, size_t win_samples,
                                            float digital_eps = 1e-5f) {
    energy_levels lv;
    std::vector<float> w = window_rms_list(samples, n_samples, win_samples);
    // Windows of digital silence (zero padding) say nothing about the room's
    // noise floor; with them included a padded file would get floor = 0 and the
    // gate could never fire on real-noise slices.
    w.erase(std::remove_if(w.begin(), w.end(), [&](float v) { return v < digital_eps; }), w.end());
    if (w.empty())
        return lv;
    std::sort(w.begin(), w.end());
    auto pct = [&](double p) { return w[std::min(w.size() - 1, (size_t)(p * (double)(w.size() - 1) + 0.5))]; };
    lv.floor = pct(0.10);
    lv.loud = pct(0.95);
    return lv;
}

// True when `samples[0..n)` holds no speech relative to `lv` (see block above).
inline bool is_speechless_span(const float* samples, size_t n_samples, const energy_levels& lv, size_t win_samples) {
    if (is_digitally_silent(samples, n_samples))
        return true;
    if (!(lv.floor > 0.0f))
        return false; // no measurable floor (all-zero elsewhere): only digital silence is judged
    const float thr = std::min(4.0f * lv.floor, lv.floor + 0.2f * (lv.loud - lv.floor));
    float peak = 0.0f;
    for (float v : window_rms_list(samples, n_samples, win_samples))
        peak = std::max(peak, v);
    return peak <= thr;
}

// The gate is on by default; CRISPASR_ENERGY_SILENCE_GATE=0 restores the old
// behaviour (every split range is transcribed).
inline bool speechless_gate_enabled() {
    const char* e = std::getenv("CRISPASR_ENERGY_SILENCE_GATE");
    return !(e && (e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F'));
}

// Remove speechless ranges from a multi-range split of `samples[0..n)`. A
// single range (audio not split at all) is returned untouched, so short input
// keeps its old behaviour; the gate only removes REMAINDERS the splitter made.
// Returns the kept ranges in order (they are no longer contiguous, like VAD
// slices); `n_dropped`, if given, receives how many were removed.
inline std::vector<std::pair<size_t, size_t>> drop_speechless_ranges(
    const float* samples, size_t n_samples, const std::vector<std::pair<size_t, size_t>>& ranges,
    size_t win_samples = 1600, size_t* n_dropped = nullptr) {
    if (n_dropped)
        *n_dropped = 0;
    if (ranges.size() <= 1 || !samples)
        return ranges;
    const energy_levels lv = estimate_energy_levels(samples, n_samples, win_samples);
    std::vector<std::pair<size_t, size_t>> kept;
    kept.reserve(ranges.size());
    for (const auto& r : ranges) {
        const size_t b = std::min(r.first, n_samples);
        const size_t e = std::min(std::max(r.second, b), n_samples);
        if (is_speechless_span(samples + b, e - b, lv, win_samples)) {
            if (n_dropped)
                ++*n_dropped;
            continue;
        }
        kept.push_back(r);
    }
    return kept;
}

} // namespace audio_chunking
