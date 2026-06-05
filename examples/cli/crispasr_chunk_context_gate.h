// crispasr_chunk_context_gate.h — gate for overlap-save context extension.
//
// Extracted from crispasr_run.cpp into a header so unit tests can exercise
// the decision without standing up the full processing pipeline.
//
// The gate decides whether each slice should be transcribed with ±
// chunk_overlap_seconds of acoustic context on each side ("overlap-save
// chunking"), or with just the bare slice samples.
//
// Issue #114: VAD-derived slices are separated by silence, so there is
// no boundary signal to recover. Adding context to a VAD slice pulls
// the *next* speech segment into the current encoder's context window,
// which shifts the FastConformer features and makes the TDT decoder
// pick a different — and worse — token path. The user-visible symptom
// on parakeet-tdt-0.6b-ja was kanji collapsing to bare hiragana plus
// entire short slices being dropped.
//
// The fix is to gate overlap-save on the actual slice source:
// it's a chunking-only mitigation, and VAD slicing is not chunking even
// when the CLI's fallback chunk_seconds value is non-zero.

#pragma once

#include <cstddef>
#include <cstring>

namespace crispasr_chunk_context {

// Backends whose own transcribe() does additional internal chunking near a
// ~30 s boundary. Wrapping their fallback chunks in extra acoustic context
// pushes the per-call input over that boundary, with backend-specific bad
// outcomes: cohere, voxtral, and qwen3 drop follow-up chunks via word-
// timestamp trimming (voxtral collapses a 5 min clip down to ~60 s of
// output; qwen3 truncates a 90 s clip mid-sentence in chunk 1);
// gemma4-e2b, glm-asr, and kyutai-stt blow past a 15 min wallclock on a
// 5 min clip (LLM-decode retry loop on the over-long buffer). voxtral4b
// is not affected — different model architecture despite the shared name.
// All six were caught by the A/B sweep in tools/check-overlap-save-bug.sh.
inline bool backend_allows_chunk_context(const char* backend_name) {
    if (backend_name == nullptr) {
        return true;
    }
    static const char* const kBlocked[] = {"cohere", "gemma4-e2b", "glm-asr", "kyutai-stt", "qwen3", "voxtral"};
    for (const char* b : kBlocked) {
        if (std::strcmp(backend_name, b) == 0) {
            return false;
        }
    }
    return true;
}

// Returns true iff the per-slice transcribe call should be wrapped in an
// overlap-save context window (`be.transcribe(samples + sl.start - ctx,
// sl.size + 2*ctx, ...)`), then trimmed back via word-level filtering.
//
// Returns false (no context, transcribe the bare slice) when:
//   - the slices came from VAD. VAD-derived slices are already speech
//     boundaries and must not pull neighbouring utterances into context.
//   - the backend does not safely support external overlap-save context.
//   - chunking is not in effect (effective_chunk_seconds == 0). The
//     CAP_UNBOUNDED_INPUT default for non-VAD runs.
//   - there is only one slice (no boundary to mitigate).
//   - the user has set --chunk-overlap 0 explicitly.
inline bool should_use_chunk_context(int effective_chunk_seconds, std::size_t n_slices, float chunk_overlap_seconds,
                                     bool vad_slicing, bool backend_allows_chunk_context = true) {
    return backend_allows_chunk_context && !vad_slicing && effective_chunk_seconds > 0 && n_slices > 1 &&
           chunk_overlap_seconds > 0.0f;
}

} // namespace crispasr_chunk_context
