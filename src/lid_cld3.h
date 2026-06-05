// lid_cld3.h — C ABI for the CLD3 (Google compact language detector v3)
// text-LID backend.
//
// CLD3 is a tiny shallow classifier (~1.5 MB F32 / ~750 KB F16): six feature
// extractors emit (id, weight) pairs, six embedding tables mean-pool them
// into an 80-d concat, one hidden FC + ReLU produces a 208-d hidden vector,
// one output FC produces 109 logits, softmax picks the language.
//
// Architecture (verified against upstream embedding_network.cc and
// lang_id_nn_params.h):
//
//   feature 0 (cbog id_dim=1000 size=2 incl_terms=true) → bigrams      → 16-d
//   feature 1 (cbog id_dim=5000 size=4 incl_terms=true) → quadgrams    → 16-d
//   feature 2 (continuous-bag-of-relevant-scripts)      → rel-scripts  →  8-d
//   feature 3 (script)                                  → text-script  →  8-d
//   feature 4 (cbog id_dim=5000 size=3 incl_terms=true) → trigrams     → 16-d
//   feature 5 (cbog id_dim=100  size=1 incl_terms=true) → unigrams     → 16-d
//   concat[80] → FC + ReLU → hidden[208] → FC → logits[109] → softmax
//
// Mirrors lid_fasttext's API shape so the auto-routing text-LID dispatcher
// in examples/cli/text_lid_dispatch.cpp can pick between them at load time
// based on the GGUF's general.architecture field.
//
// Critical implementation notes (LEARNINGS.md "Text LID via CLD3"):
//
//   * Hash is MurmurHash2-32 with seed 0xBEEF (utils.cc:137-183 — m =
//     0x5BD1E995, r = 24). The cbog feature IDs use the raw UTF-8 bytes
//     of the ngram string as the hash input, so byte-for-byte hash
//     parity with upstream is mandatory.
//
//   * Text cleanup must do FULL-Unicode lowercasing (Cyrillic П→п, Greek
//     Α→α, Latin H→h). ASCII-only lowercasing flips Cyrillic-language
//     argmax (Привет мир → tg instead of ru — verified during the port).
//
//   * Hiragana, Katakana, and Hangul are NOT in the ULScript enum.
//     They all return ULScript_Hani (=24); a secondary Hangul-vs-Hani
//     codepoint count returns the NUM_ULSCRIPTS sentinel (=102) only
//     when Korean wins. Mis-mapping these (Hani=43, Devanagari=10) was
//     the dominant cause of early smoke failures in the Python port.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lid_cld3_context;

// Initialize from a GGUF produced by models/convert-cld3-to-gguf.py.
// `n_threads` is reserved (current forward path is single-threaded — the
// matmuls are 80×208 and 208×109, both well under the worth-threading
// threshold). Returns NULL on failure (errors logged to stderr).
struct lid_cld3_context* lid_cld3_init_from_file(const char* gguf_path, int n_threads);

void lid_cld3_free(struct lid_cld3_context* ctx);

// One-shot prediction. Returns the top-1 ISO 639-1 label (e.g. "en",
// "de", "zh-Latn"). Pointer is valid until the next call into ctx or
// until lid_cld3_free; callers must copy if they need it longer.
// `confidence`, if non-NULL, receives the softmax probability of the
// top-1 label.
const char* lid_cld3_predict(struct lid_cld3_context* ctx, const char* utf8_text, float* confidence);

// Top-k prediction. `out_labels` and `out_scores` must hold at least
// `k` entries; the function writes up to `k` results in score-descending
// order and returns the count actually written (≤ k, never zero on
// success). Each label pointer is valid until the next ctx call.
// Returns 0 on failure or empty input.
int lid_cld3_predict_topk(struct lid_cld3_context* ctx, const char* utf8_text, int k, const char** out_labels,
                          float* out_scores);

// Diff-harness helper: extract a named intermediate tensor as a malloc'd
// float buffer matching the reference dump's shape. Caller owns the
// returned pointer (free with `free`). Returns NULL on unknown stage
// or failure; sets *out_n to the element count.
//
// Stages exposed (matches DEFAULT_STAGES in
// tools/reference_backends/lid_cld3.py):
//
//   "embedding_bag_0" .. "embedding_bag_5"   F32  per-feature mean-pool
//                                                 (16/16/8/8/16/16 elems)
//   "concat"                                  F32  80 elems
//   "hidden_pre"                              F32  208 elems (no ReLU)
//   "hidden_out"                              F32  208 elems (post-ReLU)
//   "logits"                                  F32  109 elems
//   "softmax"                                 F32  109 elems
float* lid_cld3_extract_stage(struct lid_cld3_context* ctx, const char* utf8_text, const char* stage_name, int* out_n);

// Read-only metadata accessors.
const char* lid_cld3_variant(const struct lid_cld3_context* ctx); // "cld3" or model.name
int lid_cld3_n_labels(const struct lid_cld3_context* ctx);        // 109
int lid_cld3_dim_total(const struct lid_cld3_context* ctx);       // 80 (concat dim)
int lid_cld3_hidden_dim(const struct lid_cld3_context* ctx);      // 208

#ifdef __cplusplus
}
#endif
