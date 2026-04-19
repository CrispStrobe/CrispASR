// crispasr_aligner.h — shared CTC / forced-alignment helper.
//
// LLM-based backends (qwen3, voxtral, voxtral4b, granite) emit plain text
// without per-word timestamps. A second pass through a CTC aligner
// produces frame-aligned word timings. Two models are supported behind
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

/// Run CTC forced alignment.
///
/// Dispatches to canary-ctc-aligner by default and to qwen3-fa when
/// `aligner_model` filename contains "forced-aligner" / "qwen3-fa" /
/// "qwen3-forced". Both model types load and free inside the call —
/// cost is dominated by the ASR pass upstream, not the aligner load.
///
/// Returns an empty vector on any failure (error printed to stderr).
std::vector<CrispasrAlignedWord> crispasr_align_words(const std::string& aligner_model,
                                                     const std::string& transcript,
                                                     const float* samples, int n_samples,
                                                     int64_t t_offset_cs, int n_threads);
