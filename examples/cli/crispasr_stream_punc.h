// crispasr_stream_punc.h — streaming punctuation placement policy
// for `--stream-json --vad --punc-model` (PR #112).
//
// Three modes selectable via `--stream-punc`:
//   "off"      — FireRedPunc runs on neither partials nor finals.
//   "final"    — FireRedPunc runs on finals only. Default; recommended
//                for realtime use because it keeps the high-frequency
//                partial path cheap while still restoring punctuation
//                on finalized utterances.
//   "partial"  — FireRedPunc runs on partials AND finals. Equivalent
//                to the historical pre-PR-#112 behavior.
//
// The helpers below answer "is FireRedPunc allowed on THIS path" for
// the two places `crispasr_run.cpp` calls into it. Extracted into a
// header so the policy is testable without pulling in the full
// streaming runner translation unit.

#pragma once

#include <string>

inline bool crispasr_stream_punc_partials_enabled(const std::string& mode) {
    return mode == "partial";
}

inline bool crispasr_stream_punc_finals_enabled(const std::string& mode) {
    return mode == "final" || mode == "partial";
}

/// Returns true when the mode string is one of the three accepted
/// values. The CLI argument parser uses an equivalent check and
/// exits with code 2 on a bad value; this is the shared
/// truth-source.
inline bool crispasr_stream_punc_mode_valid(const std::string& mode) {
    return mode == "off" || mode == "final" || mode == "partial";
}
