// crispasr_stream_finalize.h — helpers shared between the streaming
// finalize_utterance / EOF flush paths in crispasr_run.cpp and their
// unit tests.
//
// The logic here is intentionally pure (no I/O, no global state) so it
// can be tested in isolation. The streaming loop in crispasr_run.cpp
// owns the surrounding state (`prefix_committed`, `last_partial_text`,
// `utterance_pcm`, …) and feeds it into these helpers.

#pragma once

#include <string>

namespace crispasr {

// Build a `final.text` value by stitching the prefix-mode accumulator:
//   - `committed_prefix` is the longest-common-prefix of all observed
//     partials, captured before the rolling window evicted them.
//   - `last_partial` is the most recent partial for the open utterance.
//
// Cases (mirrors the historical inline implementation in
// `examples/cli/crispasr_run.cpp`):
//   * `last_partial` starts with `committed_prefix`  → `last_partial`
//     alone (it already extends the prefix).
//   * `committed_prefix` empty                       → `last_partial`.
//   * `last_partial` empty                           → `committed_prefix`.
//   * otherwise (divergent strings)                  → `committed_prefix + " " + last_partial`.
//
// This is the only place that decides what a finalized utterance's text
// reads as when no `redecode` pass produced one — either because the
// caller is in `--stream-final-mode prefix`, or because `redecode` was
// skipped (sub-2-s utterance) or returned empty in `redecode` mode (the
// fallback added in commit cf1de878, fixing #84 round 4).
inline std::string stitch_partial_accumulator(const std::string& committed_prefix, const std::string& last_partial) {
    if (!last_partial.empty() && last_partial.size() >= committed_prefix.size() &&
        last_partial.compare(0, committed_prefix.size(), committed_prefix) == 0) {
        return last_partial;
    }
    if (committed_prefix.empty()) {
        return last_partial;
    }
    if (last_partial.empty()) {
        return committed_prefix;
    }
    return committed_prefix + " " + last_partial;
}

// Threshold used by the streaming `finalize_utterance` to decide
// whether the `redecode`-mode extra backend pass is safe to invoke.
// Convolutional encoders (moonshine, parakeet, voxtral, …) abort with
// `OW > 0` from `ggml_im2col` when given less audio than the first
// conv kernel needs — empirically ~2 s at 16 kHz mono. When the
// VAD-trimmed `[t0..t1]` buffer is shorter than this, redecode is
// skipped and the caller falls back to `stitch_partial_accumulator`.
constexpr int kStreamRedecodeMinSamples = 32000;

} // namespace crispasr
