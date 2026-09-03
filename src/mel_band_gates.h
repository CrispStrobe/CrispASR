// mel_band_gates.h — path selection for mel-band-roformer (Change 176),
// pure and unit-testable. Mirrors the htdemucs_gates.h pattern from
// PR #414 (src/htdemucs_gates.h): the same three coupled decisions,
// resolved ONCE per init from the environment, the caller's use_gpu intent,
// and whether a REAL (non-CPU) GPU backend exists.
//
//   use_graph   — run the ggml graph path instead of the legacy CPU path
//   use_fused   — single fused graph (band_split + transformer stack + mask
//                 estimator on-device, "no roundtrips"); only meaningful when
//                 use_graph is set
//   gpu_backend — place the graph on the GPU backend
//
// Measured facts the AUTO defaults encode (RTX 3090 Ti, 2026-09-02): the CPU
// path is RTF ~57 (10 s clip = 9m35s); per-layer graphs on GPU run the same
// clip in 4:47; the FUSED single graph in 2.6 s (~112x faster than per-layer,
// RTF ~0.09 on a 359 s song incl. iSTFT). Per-layer graphs still pay a
// host<->device roundtrip per layer and keep the CPU mask estimator, so AUTO
// picks graph+fused+GPU exactly when a real GPU is present and permitted,
// and the plain CPU path otherwise. Never a slower-than-before configuration.
//
// Env semantics (each var: unset = AUTO, "0" = force off, else force on):
//   CRISPASR_MELBAND_GPU    — GPU permission; explicit value beats the
//                              caller's use_gpu in BOTH directions (expert
//                              override; the CLI forwards params.use_gpu, so
//                              "params wins" would make GPU=0 dead — the
//                              #414 review catch, fix 208b2d59)
//   CRISPASR_MELBAND_GGML   — graph path (pre-176 opt-in A/B flag)
//   CRISPASR_MELBAND_FUSED  — fused graph (forced on implies graph unless
//                              graph is explicitly forced off)
#pragma once

// Shared decision table: core/backend_path_gates.h (this was byte-identical
// to htdemucs_gates.h). The segment-length logic below is mel-band specific.
#include "core/backend_path_gates.h"

#include <cstdlib>

namespace mel_band_gates {

using Resolved = core_backend_gates::Resolved;

inline Resolved resolve(const char* env_gpu, const char* env_ggml, const char* env_fused, bool caller_use_gpu,
                        bool have_real_gpu) {
    return core_backend_gates::resolve(env_gpu, env_ggml, env_fused, caller_use_gpu, have_real_gpu);
}

// Segment length in SAMPLES for the Demucs-style split (review #422).
//
// Four sources, in precedence order:
//   1. CRISPASR_MELBAND_SEG_S  (seconds, only when it parses > 0)
//   2. params.segment_seconds  (seconds, only when > 0)
//   3. the checkpoint's TRAINED chunk from GGUF metadata (SAMPLES, exact)
//   4. 8 s fallback, the Kim vocals chunk, for GGUFs predating that KV
//
// The trained chunk is applied in samples rather than round-tripped through
// seconds: 352800 / 44100 = 8 exactly for Kim, but a checkpoint trained at,
// say, 344400 samples (7.81 s) would floor to 7 and silently drop 0.81 s of
// trained context from every segment. Explicit overrides stay in seconds
// because they are a human-facing knob; the trained chunk is a property of the
// checkpoint and deserves to survive intact.
//
// Returns 0 when it cannot be determined (caller then takes the whole-buffer
// path), never a negative or overflowing length.
inline int resolve_segment_len(int param_seg_s, const char* env_seg_s, int chunk_size, int sample_rate) {
    if (sample_rate <= 0)
        return 0;

    int seg_s = param_seg_s;
    if (env_seg_s && *env_seg_s) {
        const int v = atoi(env_seg_s);
        if (v > 0) // a non-positive or unparseable value is ignored, not obeyed
            seg_s = v;
    }
    if (seg_s > 0) {
        // Guard the multiply: a huge --segment would otherwise wrap negative
        // and read as "no segmentation" instead of "absurd request".
        if (seg_s > (2147483647 / sample_rate))
            return 0;
        return seg_s * sample_rate;
    }
    if (chunk_size > 0)
        return chunk_size; // already samples — used exactly
    const int fallback_s = 8;
    if (fallback_s > (2147483647 / sample_rate))
        return 0;
    return fallback_s * sample_rate;
}

} // namespace mel_band_gates
