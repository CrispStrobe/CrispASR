// crispasr_aligner.h — shared CTC / forced-alignment helper.
//
// LLM-based backends (qwen3, voxtral, voxtral4b, granite) emit plain text
// without per-word timestamps. A second pass through a CTC aligner
// produces frame-aligned word timings. Three model families are supported behind
// one entry point:
//
//   * canary-ctc-aligner   FastConformer + CTC head, 16k SentencePiece
//                          vocab covering 25+ European languages. The
//                          default; selected for any aligner model whose
//                          filename doesn't match the qwen3-fa pattern.
//
//   * qwen3-forced-aligner Qwen/Qwen3-ForcedAligner-0.6B. Same Qwen3-ASR
//                          architecture as the regular qwen3-asr backend
//                          but with a 5000-class lm_head that predicts
//                          per-token timestamps. Selected automatically
//                          when the aligner filename contains "forced-
//                          aligner" / "qwen3-fa" / "qwen3-forced".
//
//   * wav2vec2-aligner     Any GGUF accepted by the wav2vec2 backend
//                          (wav2vec2, HuBERT, data2vec CTC). Selected when
//                          the filename or GGUF architecture identifies that
//                          family. Uses the shared CTC Viterbi DP.
//
// Shared by the CLI, the C-ABI wrapper `crispasr_align_words_abi` in
// crispasr_c_api.cpp, and every language binding that reaches through
// that wrapper. `crispasr_word_aligned` is an in-library POD; the CLI
// adapts it to its own `crispasr_word` type.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CrispasrAlignedWord {
    std::string text;
    int64_t t0_cs = 0; // centiseconds, absolute (includes t_offset_cs)
    int64_t t1_cs = 0;
};

/// A re-timed segment: one input SRT cue (or one text line) with timings
/// derived from the word alignment that covers it.
struct CrispasrAlignedSegment {
    std::string text;
    int64_t t0_cs = 0;
    int64_t t1_cs = 0;
    size_t word_begin = 0; // [word_begin, word_end) into the aligned-word vector
    size_t word_end = 0;
};

// Absolute sample interval used when aligning one ASR segment.  Keeping this
// calculation beside the aligner prevents per-slice callers from accidentally
// feeding every segment the whole VAD slice (issue #444), which resets each
// segment's timestamps to the same slice origin.
struct CrispasrAlignmentAudioRange {
    int start = 0;         // absolute sample index, inclusive
    int end = 0;           // absolute sample index, exclusive
    int64_t offset_cs = 0; // timestamp corresponding to start

    bool valid() const { return end > start; }
};

CrispasrAlignmentAudioRange crispasr_alignment_audio_range(int64_t segment_t0_cs, int64_t segment_t1_cs,
                                                           int slice_start, int slice_end, int sample_rate);

/// Split text into alignment "words": whitespace-delimited for
/// space-delimited languages, per-character for CJK. This is the exact
/// splitter the aligner backends use — callers that map aligned words back
/// onto larger units (SRT cues, lines) must count with this, not with a
/// naive space count.
std::vector<std::string> crispasr_tokenise_align_words(const std::string& text);

/// The same alignment units with punctuation reattached to the neighbouring
/// unit for display. Alignment models must not receive punctuation timestamp
/// slots, but subtitle splitting still needs sentence-ending marks on the
/// returned words so it can use their measured times instead of interpolation.
std::vector<std::string> crispasr_tokenise_align_display_words(const std::string& text);

/// Parse SRT content into cue texts (indices and timestamps discarded,
/// multi-line cue text joined with spaces, whitespace-only cues dropped).
std::vector<std::string> crispasr_parse_srt_cues(const std::string& raw);

/// Parse CrispASR JSON output into segment texts. Extracts the "text" field
/// from each object in the "transcription" array. Works with both the full
/// JSON format (--output-json-full) and the plain format (--output-json).
/// Also accepts the align-only JSON format (array of {text, start, end}).
/// Returns empty on parse failure or missing transcription data.
std::vector<std::string> crispasr_parse_json_segments(const std::string& raw);

/// Group a flat word alignment back into the segment texts it was built
/// from. `segment_texts` joined with spaces must equal the transcript the
/// words were aligned against; each segment consumes its own word count
/// (per crispasr_tokenise_align_words). Segment timings are the first
/// word's t0 and the last word's t1; leftover words extend the last
/// segment; segments left without words (alignment ended early) are dropped.
std::vector<CrispasrAlignedSegment> crispasr_group_aligned_segments(const std::vector<std::string>& segment_texts,
                                                                    const std::vector<CrispasrAlignedWord>& words);

/// Run CTC forced alignment.
///
/// Dispatches to canary-ctc-aligner by default, to qwen3-fa when
/// `aligner_model` filename contains "forced-aligner" / "qwen3-fa" /
/// "qwen3-forced", and to wav2vec2/hubert/data2vec CTC when the filename
/// or GGUF architecture identifies that family. Models load and free inside the call —
/// cost is dominated by the ASR pass upstream, not the aligner load.
///
/// Returns an empty vector on any failure (error printed to stderr).
///
/// When `out_load_failed` is non-null it is set to `true` iff the aligner
/// MODEL could not be loaded (as opposed to loading fine but yielding no words),
/// so issue #311 --strict-pipeline can fail even when the backend's native word
/// timestamps would otherwise mask an explicitly-requested aligner that never
/// ran. Best-effort: covers the canary-CTC path (and any unrecognized/corrupt
/// GGUF, which routes there); it stays `false` otherwise and never
/// false-positives.
std::vector<CrispasrAlignedWord> crispasr_align_words(const std::string& aligner_model, const std::string& transcript,
                                                      const float* samples, int n_samples, int64_t t_offset_cs,
                                                      int n_threads, bool* out_load_failed = nullptr);

/// Free the cached aligner model context (§176e). Call at shutdown.
void crispasr_aligner_free_cache();
