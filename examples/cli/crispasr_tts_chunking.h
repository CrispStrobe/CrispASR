// crispasr_tts_chunking.h — long-form input helpers for /v1/audio/speech.
//
// The talker LM in every TTS backend has a finite training horizon
// (qwen3-tts-1.7b-base degrades past ~600 chars / 200 codec frames
// and silently truncates at MAX_FRAMES). Single-shot synthesis of
// long input therefore drifts in quality and eventually drops trailing
// text — neither failure surfaces to the API caller.
//
// The fix from issue #66 is to chunk on sentence boundaries before
// dispatching to backend->synthesize(), then concatenate the per-chunk
// float32 PCM with a brief silence pad. RTF stays flat from short to
// long input because each chunk synthesises within the model's healthy
// horizon, and voice consistency holds across chunk boundaries because
// the talker re-prefills with the same ICL ref every call (and our
// last_voice_key_ cache keeps that load-once cost amortised).
//
// Pure functions, no backend dependency. Unit-tested in
// tests/test_server_chunking.cpp.

#pragma once

#include <string>
#include <vector>

// Split text on sentence terminators (`.!?` ASCII; `。` U+3002 CJK
// ideographic full stop; `।` U+0964 Devanagari danda) when followed
// by whitespace or end-of-input. Any chunk exceeding `max_chars` is
// further split on whitespace boundaries so no piece can blow past
// the cap (protects against run-on input with no punctuation).
//
// Each returned chunk is trimmed of surrounding whitespace; empty
// pieces are dropped. Empty / whitespace-only input → empty vector.
//
// Known limitations (acceptable for v1):
//   - Over-splits English abbreviations like "Mr. Smith" (the period
//     is followed by whitespace). Adds an extra ~200 ms pause; doesn't
//     break audio. ICU BreakIterator would fix this — out of scope.
//   - Decimal numbers like "1.5" stay intact (the period is followed
//     by a digit, not whitespace).
//   - Quotation marks aren't pulled into the leading chunk: `"Hi."` then
//     `"Bye."` becomes `"Hi.` and `"Bye."` (the trailing close-quote
//     of one sentence belongs to the next chunk). Cosmetic — synthesis
//     ignores quote characters.
std::vector<std::string> crispasr_tts_split_sentences(const std::string& text, std::size_t max_chars = 600);

// Server policy wrapper around crispasr_tts_split_sentences. VibeVoice
// voice cloning intentionally stays single-shot because chunking breaks
// the continuous prompt + generated-text context that carries speaker
// identity and prosody.
std::vector<std::string> crispasr_tts_plan_chunks_for_backend(const std::string& text, const std::string& backend_name,
                                                              std::size_t max_chars = 600);

// Concatenate per-chunk PCM with `silence_samples` zeros inserted
// BETWEEN chunks. No pad before the first chunk or after the last —
// avoids audible click at output boundaries. silence_samples <= 0
// concatenates without pad. Empty input vector → empty output.
std::vector<float> crispasr_tts_concat_with_silence(const std::vector<std::vector<float>>& chunks, int silence_samples);
