// src/session_autochunk.h — pure applicability decision for the session
// long-audio auto-chunker (fix/session-long-audio).
//
// The raw session transcribe is a single pass; for short-segment models this
// degrades past ~30 s while the CLI dispatcher chunks. `transcribe_autochunk`
// slices long audio and transcribes each piece. This header holds only the pure
// "should we auto-chunk this call?" decision so it can be unit-tested without a
// model.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

namespace core_session {

// True iff the session should slice this buffer into chunks before transcribing.
//   enabled          — CRISPASR_SESSION_AUTOCHUNK (default on)
//   backend          — session backend id
//   n_samples, sr    — buffer length
//   chunk_seconds    — window (CRISPASR_SESSION_CHUNK_SECONDS, default 30)
//   return_logits    — session opted into per-frame CTC logits (can't merge)
//   already_chunking — an explicit chunked request is in flight (force_chunk>=0)
//
// Skips backends that already chunk internally (parakeet/reazonspeech
// self-chunk in transcribe_single), the logits path, an explicit chunk request,
// and audio at/under the window.
inline bool session_autochunk_applicable(bool enabled, const std::string& backend, int n_samples, int sr,
                                         int chunk_seconds, bool return_logits, bool already_chunking) {
    if (!enabled || return_logits || already_chunking)
        return false;
    if (backend == "parakeet" || backend == "reazonspeech" || backend == "index-echo")
        return false; // self-chunk in transcribe_single
    if (sr <= 0 || chunk_seconds <= 0)
        return false;
    return (long long)n_samples > (long long)chunk_seconds * sr;
}

// Per-backend default auto-chunk window in seconds, used when the caller did not
// pin one via CRISPASR_SESSION_CHUNK_SECONDS. The shipped default is a flat 30 s
// for every backend (perbackend_enabled=false) — that is the verified-best window
// for moonshine on the long-song A/B (see below), and the safe single-pass window
// for whisper/qwen3/nemotron.
//
// F4 (opt-in via CRISPASR_SESSION_PERBACKEND_CHUNK=1): short-segment models whose
// encoder is trained on short utterances (moonshine) get a smaller window, the
// session mirror of the CLI's vad_slice_cap_seconds(). This path is kept behind a
// gate — NOT deleted — because it MIGHT help on other clips/models and just needs
// more A/B. On the one long clip measured so far (moonshine / 60 s tiled song) it
// REGRESSED CLI-vs-session content overlap (30 s → 0.75, 20 s → 0.56) because more
// slices add chunk-boundary artifacts on that hard audio, so it stays opt-in until
// a clip is found where it wins; then flipping the default here is a one-liner.
// CRISPASR_SESSION_CHUNK_SECONDS remains the direct per-call override for any run.
inline int session_default_chunk_seconds(const std::string& backend, bool perbackend_enabled) {
    if (perbackend_enabled && (backend == "moonshine" || backend == "moonshine-streaming"))
        return 20;
    return 30;
}

// Issue #574: Piece-proportional token budget for chunked / long-audio decode.
// Each piece gets max(minimum, round(rate * duration_seconds)), optionally
// capped by session_max_new_tokens if an explicit flat limit was set (> 0).
// When rate <= 0, dynamic budget is disabled and session_max_new_tokens is returned.
inline int compute_piece_max_tokens(float rate, int minimum, int n_samples, int sr, int session_max_new_tokens = 0) {
    if (rate <= 0.0f)
        return session_max_new_tokens;
    if (sr <= 0 || n_samples <= 0)
        return std::max(0, minimum);
    const double secs = (double)n_samples / (double)sr;
    int limit = (int)std::lround(secs * (double)rate);
    if (limit < minimum)
        limit = minimum;
    if (session_max_new_tokens > 0 && limit > session_max_new_tokens)
        limit = session_max_new_tokens;
    return limit;
}

} // namespace core_session
