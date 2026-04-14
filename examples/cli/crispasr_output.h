// crispasr_output.h — output formatting shared across non-whisper backends.
//
// These writers consume std::vector<crispasr_segment> (the common result
// type) rather than whisper_context, so any backend can drive them.
//
// The whisper code path in cli.cpp continues to use its own writers
// (output_txt, output_srt, etc. defined there) because they have features
// like token-level WTS karaoke output and JSON metadata that are
// whisper-specific for now. A later refactor will unify the two.

#pragma once

#include "crispasr_backend.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

// Convert a centisecond timestamp to "HH:MM:SS.mmm" (VTT) or
// "HH:MM:SS,mmm" (SRT, when comma=true). Mirrors common-whisper's
// to_timestamp() but avoids a dependency on that library here.
std::string crispasr_to_timestamp(int64_t cs, bool comma = false);

// Derive an output path from an audio input path by stripping a known
// audio extension and appending the given extension (including the dot).
// "audio.wav" + ".srt" -> "audio.srt".
std::string crispasr_make_out_path(const std::string & audio, const std::string & ext);

// ---------------------------------------------------------------------------
// Display segments: what actually gets written to stdout and output files.
// Built from the crispasr_segment vector by splitting long segments on word
// boundaries when max_len > 0, or emitting one segment per word when
// max_len == 1.
// ---------------------------------------------------------------------------

struct crispasr_disp_segment {
    int64_t     t0, t1;        // centiseconds, absolute
    std::string text;
    std::string speaker;       // empty if none
};

// Build display segments from backend segments according to max_len.
//   max_len = 0 -> one display segment per input segment (no splitting)
//   max_len = 1 -> one display segment per word (requires words populated)
//   max_len > 1 -> split at word boundaries when accumulated text would
//                  exceed max_len characters
std::vector<crispasr_disp_segment> crispasr_make_disp_segments(
    const std::vector<crispasr_segment> & segments,
    int                                   max_len);

// ---------------------------------------------------------------------------
// Writers. All take a full file path; callers are expected to choose the
// path via crispasr_make_out_path().
// ---------------------------------------------------------------------------

bool crispasr_write_txt(const std::string & path,
                        const std::vector<crispasr_disp_segment> & segs);

bool crispasr_write_srt(const std::string & path,
                        const std::vector<crispasr_disp_segment> & segs);

bool crispasr_write_vtt(const std::string & path,
                        const std::vector<crispasr_disp_segment> & segs);

bool crispasr_write_csv(const std::string & path,
                        const std::vector<crispasr_disp_segment> & segs);

bool crispasr_write_json(const std::string & path,
                         const std::vector<crispasr_segment> & segs,
                         const std::string & backend_name,
                         const std::string & model_path,
                         const std::string & language,
                         bool full);

bool crispasr_write_lrc(const std::string & path,
                        const std::vector<crispasr_disp_segment> & segs);

// Print segments to stdout. If show_timestamps is true, each line is
// "[t0 --> t1] text"; otherwise the transcript is printed as one blob per
// segment separated by spaces.
void crispasr_print_stdout(const std::vector<crispasr_disp_segment> & segs,
                           bool show_timestamps);

// Print per-token alternatives (--alt mode). Shows each token with its
// confidence and top-N alternative candidates, inspired by antirez/voxtral.c.
void crispasr_print_alternatives(const std::vector<crispasr_segment> & segs,
                                  int n_alt);

// Remove punctuation from a segment in-place: from seg.text, each
// seg.words[i].text, and each seg.tokens[i].text. Called by the
// dispatch layer when --no-punctuation is set and the backend didn't
// strip punctuation natively. Targets ASCII punctuation plus a small
// set of common Unicode marks the LLM backends emit (smart quotes, em
// dash, ellipsis). Idempotent — running it twice is a no-op.
void crispasr_strip_punctuation(crispasr_segment & seg);
