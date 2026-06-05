// crispasr_stream_partial_decode.h — partial-decode cadence policy for
// `--stream-json --vad --stream-partial-decode-ms` (PR #113).
//
// The throttle decides, per streaming step, whether to call
// `backend->transcribe()` for the live partial event. It runs only on
// the JSON+VAD path; the non-JSON streaming path always decodes every
// step. When throttled, VAD slice timing and silence-finalization
// checks still run every step, so utterance boundaries are unaffected.
//
// Rules (in order):
//   1. Non-JSON streaming                   -> always decode.
//   2. First step of the stream             -> always decode (so the
//                                              first partial fires
//                                              immediately).
//   3. `final_silence_due`                  -> bypass throttle. The
//                                              caller computes this
//                                              from
//                                              `--stream-final-on-silence-ms`
//                                              vs. `last_speech_end_sample`;
//                                              true means the next
//                                              step is about to
//                                              finalize anyway, so let
//                                              the short-utterance
//                                              fallback see a fresh
//                                              partial.
//   4. Interval ≤ 0                         -> defensive no-op
//                                              (degenerate config:
//                                              should not happen in
//                                              practice).
//   5. Otherwise                            -> decode iff samples
//                                              elapsed since the last
//                                              decode ≥ the configured
//                                              interval.
//
// Extracted from `crispasr_run.cpp` so the policy is testable without
// dragging in the full streaming runner translation unit.

#pragma once

#include <cstdint>

/// Convert `--stream-partial-decode-ms` to a sample-count interval.
/// When the flag is `0` (the default), the interval matches one
/// `--stream-step-ms` worth of audio — i.e. the throttle is conceptually
/// always on, just locked to the step cadence by default.
inline int64_t crispasr_stream_partial_decode_interval_samples(int ms, int step_ms, int sample_rate) {
    if (sample_rate <= 0)
        return 0;
    const int effective_ms = (ms > 0) ? ms : step_ms;
    if (effective_ms <= 0)
        return 0;
    return ((int64_t)effective_ms * (int64_t)sample_rate) / 1000;
}

/// Decide whether to run a partial ASR decode at this streaming step.
inline bool crispasr_stream_partial_decode_allow(bool stream_json, int64_t last_partial_decode_sample,
                                                 bool final_silence_due, int64_t cumulative_samples,
                                                 int64_t interval_samples) {
    if (!stream_json)
        return true;
    if (last_partial_decode_sample < 0)
        return true;
    if (final_silence_due)
        return true;
    if (interval_samples <= 0)
        return true;
    return (cumulative_samples - last_partial_decode_sample) >= interval_samples;
}
