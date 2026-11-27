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
// by whitespace or end-of-input. Any sentence exceeding `max_chars` is
// further split by crispasr_tts_split_long_sentence() so no piece can blow
// past the cap (protects against run-on input with no punctuation).
//
// Each returned chunk is trimmed of surrounding whitespace; empty
// pieces are dropped. Empty / whitespace-only input → empty vector.
//
// A '.' after an ordinal ("am 3. Oktober"), initial ("z. B.") or known
// abbreviation ("Mr.", "Dr.", "bzw.") does not end a sentence; the rule is
// shared with live translation (crispasr_sentence_end.h).
// CRISPASR_SENTENCE_ABBREV=0 restores the bare ". " rule.
//
// Known limitations (acceptable for v1):
//   - Abbreviations missing from the shared list still over-split.
//   - Decimal numbers like "1.5" stay intact (the period is followed
//     by a digit, not whitespace).
//   - Quotation marks aren't pulled into the leading chunk: `"Hi."` then
//     `"Bye."` becomes `"Hi.` and `"Bye."` (the trailing close-quote
//     of one sentence belongs to the next chunk). Cosmetic — synthesis
//     ignores quote characters.
std::vector<std::string> crispasr_tts_split_sentences(const std::string& text, std::size_t max_chars = 600);

// Split ONE sentence that is longer than max_chars into pieces of at most
// max_chars bytes (#609). Breaks go at the strongest clause boundary that
// keeps the pieces balanced: ; : ? ! or a dash first, then a comma, then any
// space, and a space after a '.' that did not end the sentence ("M.", "Mr.")
// only as a last resort; a text with no space at all is cut at a UTF-8
// character boundary. Pieces are trimmed; joined with single spaces they give
// back the trimmed input. A sentence within the cap comes back unchanged.
// CRISPASR_TTS_CLAUSE_SPLIT=0 restores the old split at the last space before
// the cap.
std::vector<std::string> crispasr_tts_split_long_sentence(const std::string& sentence, std::size_t max_chars);

// Server policy wrapper around crispasr_tts_split_sentences. VibeVoice
// voice cloning intentionally stays single-shot because chunking breaks
// the continuous prompt + generated-text context that carries speaker
// identity and prosody.
std::vector<std::string> crispasr_tts_plan_chunks_for_backend(const std::string& text, const std::string& backend_name,
                                                              std::size_t max_chars = 600);

// Greedily join consecutive sentences (space-separated) into chunks of at most
// max_chars bytes. A single sentence longer than max_chars stays on its own
// (the splitter already bounded it). Used for bt2-tts (#594).
std::vector<std::string> crispasr_tts_pack_sentences(const std::vector<std::string>& sentences, std::size_t max_chars);

// Concatenate per-chunk PCM with `silence_samples` zeros inserted
// BETWEEN chunks. No pad before the first chunk or after the last —
// avoids audible click at output boundaries. silence_samples <= 0
// concatenates without pad. Empty input vector → empty output.
std::vector<float> crispasr_tts_concat_with_silence(const std::vector<std::vector<float>>& chunks, int silence_samples);
