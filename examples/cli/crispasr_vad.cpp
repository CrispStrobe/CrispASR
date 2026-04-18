// crispasr_vad.cpp — implementation of Silero-VAD-backed audio slicing.
// See crispasr_vad.h for the interface contract.
//
// Extracted from examples/parakeet-main/main.cpp.

#include "crispasr_vad.h"
#include "crispasr_cache.h"
#include "whisper_params.h"

#include "whisper.h" // whisper_vad_* API

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>

namespace {

// Default Silero VAD model from the ggml-org/whisper-vad HF repo.
// ~885 KB. Auto-downloaded on first use to ~/.cache/crispasr so users
// can pass `--vad` without having to hunt down the GGUF.
constexpr const char* kVadDefaultUrl =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
constexpr const char* kVadDefaultFile = "ggml-silero-v5.1.2.bin";

// Resolve params.vad_model into a real file path. When empty (user passed
// --vad without --vad-model), or set to "auto"/"default", download the
// canonical Silero VAD GGUF into the crispasr cache dir on first use.
// Returns an empty string if no VAD was requested at all.
std::string resolve_vad_model(const whisper_params& p) {
    const std::string& v = p.vad_model;
    const bool want_vad = p.vad || !v.empty();
    if (!want_vad)
        return "";

    // Explicit path (not auto/default) — use as-is.
    if (!v.empty() && v != "auto" && v != "default")
        return v;

    // Auto path: delegate to the shared cache helper.
    return crispasr_cache::ensure_cached_file(kVadDefaultFile, kVadDefaultUrl, p.no_prints, "crispasr[vad]",
                                              p.cache_dir);
}

} // namespace

std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(const float* samples, int n_samples, int sample_rate,
                                                                int chunk_seconds, const whisper_params& params) {
    std::vector<crispasr_audio_slice> slices;

    whisper_vad_context* vctx = nullptr;
    const std::string vad_path = resolve_vad_model(params);
    const bool want_vad = !vad_path.empty();
    if (want_vad) {
        whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = params.n_threads;
        vctx = whisper_vad_init_from_file_with_params(vad_path.c_str(), vcp);
        if (!vctx) {
            fprintf(stderr,
                    "crispasr: warning: failed to load VAD model '%s', "
                    "falling back to fixed chunking\n",
                    vad_path.c_str());
        }
    }

    if (vctx) {
        whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold = params.vad_threshold;
        vp.min_speech_duration_ms = params.vad_min_speech_duration_ms;
        vp.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        vp.speech_pad_ms = (float)params.vad_speech_pad_ms;

        whisper_vad_segments* vseg = whisper_vad_segments_from_samples(vctx, vp, samples, n_samples);

        const int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
        for (int i = 0; i < nv; i++) {
            // The whisper VAD API returns timestamps in centiseconds,
            // not seconds. Convert to seconds for sample index computation.
            const float t0_cs = whisper_vad_segments_get_segment_t0(vseg, i);
            const float t1_cs = whisper_vad_segments_get_segment_t1(vseg, i);
            const float t0s = t0_cs / 100.0f;
            const float t1s = t1_cs / 100.0f;
            const int s = std::max(0, (int)(t0s * sample_rate));
            const int e = std::min(n_samples, (int)(t1s * sample_rate));
            if (e > s) {
                slices.push_back({
                    s,
                    e,
                    (int64_t)t0_cs,
                    (int64_t)t1_cs,
                });
            }
        }
        if (vseg)
            whisper_vad_free_segments(vseg);
        whisper_vad_free(vctx);

        // Post-merge: combine adjacent VAD segments that are too short
        // or too close together. ASR models need at least a few seconds
        // of audio context to produce reliable output; tiny VAD segments
        // (< 2s) and short inter-segment gaps (< 1s) degrade quality.
        if (slices.size() > 1) {
            const int min_dur_samples = 3 * sample_rate;   // 3 s minimum
            const int merge_gap_samples = 1 * sample_rate; // merge if gap < 1 s
            std::vector<crispasr_audio_slice> merged;
            merged.push_back(slices[0]);
            for (size_t i = 1; i < slices.size(); i++) {
                auto& prev = merged.back();
                const int gap = slices[i].start - prev.end;
                const int prev_dur = prev.end - prev.start;
                if (gap < merge_gap_samples || prev_dur < min_dur_samples) {
                    // Merge: extend prev to cover the current segment
                    prev.end = slices[i].end;
                    prev.t1_cs = slices[i].t1_cs;
                } else {
                    merged.push_back(slices[i]);
                }
            }
            slices = std::move(merged);
        }

        // Post-split: break any VAD segment that exceeds chunk_seconds
        // into sub-segments at the best gap within the merged segment.
        // This prevents OOM on very long continuous speech (e.g. 10+ min
        // lectures). We split at the original VAD boundary that falls
        // closest to each chunk_seconds multiple.
        if (chunk_seconds > 0) {
            const int max_samples = chunk_seconds * sample_rate;
            std::vector<crispasr_audio_slice> split;
            for (auto& sl : slices) {
                const int dur = sl.end - sl.start;
                if (dur <= max_samples) {
                    split.push_back(sl);
                } else {
                    // Split this oversized segment into roughly equal parts.
                    const int n_parts = (dur + max_samples - 1) / max_samples;
                    const int part_samples = dur / n_parts;
                    for (int p = 0; p < n_parts; p++) {
                        const int s = sl.start + p * part_samples;
                        const int e = (p == n_parts - 1) ? sl.end : sl.start + (p + 1) * part_samples;
                        split.push_back({
                            s,
                            e,
                            (int64_t)((double)s / sample_rate * 100.0),
                            (int64_t)((double)e / sample_rate * 100.0),
                        });
                    }
                }
            }
            slices = std::move(split);
        }

        return slices;
    }

    // No VAD: fall back to fixed chunking. Encoders scale O(T^2) in frame
    // count, so unbounded audio hits memory/latency walls. The default
    // chunk_seconds (30 s) is a conservative window most models handle well.
    //
    // Note: fixed chunking can cut mid-sentence at boundaries. For better
    // results, use --vad which segments at natural speech pauses.
    const int chunk_samples = chunk_seconds > 0 ? chunk_seconds * sample_rate : n_samples;

    if (n_samples <= chunk_samples) {
        const int64_t dur_cs = (int64_t)((double)n_samples / sample_rate * 100.0);
        slices.push_back({0, n_samples, 0, dur_cs});
    } else {
        for (int s = 0; s < n_samples; s += chunk_samples) {
            const int e = std::min(n_samples, s + chunk_samples);
            slices.push_back({
                s,
                e,
                (int64_t)((double)s / sample_rate * 100.0),
                (int64_t)((double)e / sample_rate * 100.0),
            });
        }
    }
    return slices;
}

// ---------------------------------------------------------------------------
// Timestamp remapping: stitched-buffer centiseconds → original-audio centiseconds.
// Uses linear interpolation between the mapping table entries.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Stitch multiple VAD slices into a single contiguous audio buffer with
// 0.1s silence gaps (matching whisper.cpp's approach). Builds a mapping
// table for timestamp remapping.
// ---------------------------------------------------------------------------
crispasr_stitched_audio crispasr_stitch_vad_slices(const float* samples, int /*n_samples*/, int sample_rate,
                                                   const std::vector<crispasr_audio_slice>& slices) {
    crispasr_stitched_audio result;
    if (slices.empty())
        return result;

    const int silence_samples = (int)(0.1f * sample_rate); // 0.1s silence gap

    // Compute total size.
    size_t total = 0;
    for (const auto& sl : slices)
        total += (size_t)(sl.end - sl.start);
    total += (size_t)(slices.size() - 1) * silence_samples; // silence gaps

    result.samples.resize(total, 0.0f);
    result.mapping.reserve(slices.size() * 2);

    int offset = 0;
    for (size_t i = 0; i < slices.size(); i++) {
        const auto& sl = slices[i];
        const int seg_len = sl.end - sl.start;

        // Map: stitched offset → original start.
        result.mapping.push_back({(int64_t)((double)offset / sample_rate * 100.0), sl.t0_cs});

        // Copy speech segment.
        std::memcpy(result.samples.data() + offset, samples + sl.start, (size_t)seg_len * sizeof(float));
        offset += seg_len;

        // Map: stitched end → original end.
        result.mapping.push_back({(int64_t)((double)offset / sample_rate * 100.0), sl.t1_cs});

        // Insert silence gap (already zeroed by resize).
        if (i + 1 < slices.size())
            offset += silence_samples;
    }

    result.total_duration_cs = (int64_t)((double)offset / sample_rate * 100.0);
    return result;
}

// ---------------------------------------------------------------------------
// Timestamp remapping: stitched-buffer centiseconds → original-audio centiseconds.
// Uses linear interpolation between the mapping table entries.
// ---------------------------------------------------------------------------
int64_t crispasr_vad_remap_timestamp(const std::vector<crispasr_vad_mapping>& mapping, int64_t stitched_cs) {
    if (mapping.empty())
        return stitched_cs;
    if (stitched_cs <= mapping.front().stitched_cs)
        return mapping.front().original_cs;
    if (stitched_cs >= mapping.back().stitched_cs)
        return mapping.back().original_cs;

    // Binary search for the bracket.
    size_t lo = 0, hi = mapping.size() - 1;
    while (lo + 1 < hi) {
        size_t mid = (lo + hi) / 2;
        if (mapping[mid].stitched_cs <= stitched_cs)
            lo = mid;
        else
            hi = mid;
    }
    // Linear interpolation between mapping[lo] and mapping[hi].
    const auto& a = mapping[lo];
    const auto& b = mapping[hi];
    if (b.stitched_cs == a.stitched_cs)
        return a.original_cs;
    const double frac = (double)(stitched_cs - a.stitched_cs) / (double)(b.stitched_cs - a.stitched_cs);
    return a.original_cs + (int64_t)(frac * (double)(b.original_cs - a.original_cs));
}
