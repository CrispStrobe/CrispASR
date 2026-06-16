// lid_fasttext.h — fastText supervised LID (GlotLID + LID-176).
//
// Architecture: hashed character-n-gram embedding bag → mean pool →
// linear → softmax. A tiny model in compute terms (~1 MFLOP per call)
// over a large embedding table (1.6M × 256 for GlotLID-V3, ~250k × 16
// for LID-176). Forward is pure manual F32/F16 — no ggml graph — matching
// the silero_lid pattern for small classifiers. Quants land later (the
// matmul will need ggml_mul_mat to handle Q8_0/Q5_K/Q4_K dequant).
//
// The two GGUF tensors:
//
//   ``lid_fasttext.embedding``  ``[n_words + bucket, dim]``  F32 or F16
//   ``lid_fasttext.output``     ``[n_labels, dim]``          F32 or F16
//
// Both GlotLID and LID-176 are produced by ``models/convert-glotlid-to-gguf.py``
// (variant string in metadata selects which); the C++ side reads
// ``lid_fasttext.variant`` and ``lid_fasttext.labels`` to know what
// language tagset it's predicting over.
//
// Critical detail: fastText supervised classifiers mean-pool an extra
// ``</s>`` (EOS) row at the end of every input — always
// ``input_matrix[0]`` — that ``tokenize()`` does not return. Missing it
// drops cosine to ~0.97 against the reference. See
// ``feedback_fasttext_eos_row`` memory.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lid_fasttext_context;

// Initialize from a GGUF produced by convert-glotlid-to-gguf.py.
// `n_threads` is reserved for future quant matmul; current manual path
// is single-threaded. Returns NULL on failure (errors logged to stderr).
struct lid_fasttext_context* lid_fasttext_init_from_file(const char* gguf_path, int n_threads);

void lid_fasttext_free(struct lid_fasttext_context* ctx);

// One-shot prediction: returns top-1 ISO 639-3 + script label string
// (e.g. "eng_Latn"). Pointer is valid until the next call into ctx or
// lid_fasttext_free; callers must copy if they need it longer.
// `confidence`, if non-NULL, receives the softmax score of the top-1.
// Returns NULL on failure.
const char* lid_fasttext_predict(struct lid_fasttext_context* ctx, const char* utf8_text, float* confidence);

// Top-k prediction. `out_labels` and `out_scores` must hold at least
// `k` entries; the function writes up to `k` results in score-descending
// order and returns the actual count written (≤ k, never zero on
// success). Each label pointer is valid until the next ctx call.
// Returns 0 on failure or empty input.
int lid_fasttext_predict_topk(struct lid_fasttext_context* ctx, const char* utf8_text, int k, const char** out_labels,
                              float* out_scores);

// Diff-harness helper: extract a named intermediate tensor as a
// malloc'd float buffer matching the reference dump's shape. Caller
// owns the returned pointer (free with `free`). Returns NULL on
// unknown stage or failure; sets *out_n to the element count.
//
// Stages exposed (matches tools/reference_backends/lid_glotlid.py
// DEFAULT_STAGES):
//
//   "input_ids"          int32-as-float, [n_input]  — concatenated row IDs
//   "embedding_bag_out"  float32, [dim]
//   "logits"             float32, [n_labels]
//   "softmax"            float32, [n_labels]
//   "top1_score"         float32, [1]
//
// The harness compares each via cosine + max-abs against the GGUF
// reference dump.
float* lid_fasttext_extract_stage(struct lid_fasttext_context* ctx, const char* utf8_text, const char* stage_name,
                                  int* out_n);

// Read-only metadata accessors — useful for the diff harness and CLI.
const char* lid_fasttext_variant(const struct lid_fasttext_context* ctx); // "glotlid-v3" | "fasttext-lid176"
int lid_fasttext_n_labels(const struct lid_fasttext_context* ctx);
int lid_fasttext_dim(const struct lid_fasttext_context* ctx);

#ifdef __cplusplus
}
#endif
