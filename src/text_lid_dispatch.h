// text_lid_dispatch.h — backend-agnostic façade over the text-LID
// runtimes. Peeks the GGUF's `general.architecture` key at load time
// and routes to the appropriate backend (lid_fasttext for GlotLID-V3
// + LID-176; lid_cld3 for Google CLD3).
//
// The two backends already mirror each other's C ABI shape — this
// dispatcher is just a thin tag dispatch over them, so callers don't
// have to know which family the GGUF came from.
//
// Usage (replaces direct lid_{fasttext,cld3}_* calls):
//
//   text_lid_context* ctx = text_lid_init_from_file("model.gguf", 1);
//   float conf = 0.0f;
//   const char* lang = text_lid_predict(ctx, "Hallo Welt", &conf);
//   // lang = "de", conf = 0.99
//   text_lid_free(ctx);
//
// The same flag in cli.cpp's `--lid-on-transcript` and the standalone
// `crispasr-lid` binary go through this façade so any text-LID GGUF
// produced by `convert-glotlid-to-gguf.py` or `convert-cld3-to-gguf.py`
// works without re-running the user with a different flag.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct text_lid_context;

// Initialize from any text-LID GGUF. The `general.architecture` field
// chooses the backend (`lid-cld3` → CLD3, `lid-fasttext` → fastText
// supervised). Returns NULL on failure with an error logged to stderr.
struct text_lid_context* text_lid_init_from_file(const char* gguf_path, int n_threads);

#ifdef __cplusplus
} // extern "C"

#include <string>

// Resolve `arg` to a usable on-disk GGUF path, auto-downloading from the
// registry if needed. Accepted forms:
//
//   "auto"                       → cstr/cld3-GGUF (smallest, Apache-2.0)
//   "auto:cld3"                  → same as "auto"
//   "auto:glotlid"               → cstr/glotlid-GGUF (2102 ISO 639-3)
//   "auto:lid-fasttext176"       → cstr/fasttext-lid176-GGUF (CC-BY-SA-3.0)
//   "<filename>" or "<path>"     → if it exists, return as-is; else look up
//                                  by basename in the registry and download.
//
// Returns the absolute path on success, empty string on failure (with an
// error logged to stderr). `cache_dir_override` is forwarded to
// `crispasr_cache::ensure_cached_file`; pass an empty string for the
// default `~/.cache/crispasr/`.
std::string text_lid_resolve_path(const std::string& arg, const std::string& cache_dir_override = "",
                                  bool quiet = false);

extern "C" {
#endif

void text_lid_free(struct text_lid_context* ctx);

// Top-1 prediction. Returns the label string (pointer valid until the
// next ctx call), and writes the softmax probability to `confidence`
// if non-NULL.
const char* text_lid_predict(struct text_lid_context* ctx, const char* utf8_text, float* confidence);

// Top-k prediction. Writes up to `k` (label, score) pairs in
// score-descending order; returns the count actually written.
int text_lid_predict_topk(struct text_lid_context* ctx, const char* utf8_text, int k, const char** out_labels,
                          float* out_scores);

// Read-only metadata accessors. `text_lid_dim` returns the
// dispatcher's view of "feature width" — fastText's embedding dim or
// CLD3's concat dim — purely informational.
const char* text_lid_variant(const struct text_lid_context* ctx);
int text_lid_n_labels(const struct text_lid_context* ctx);
int text_lid_dim(const struct text_lid_context* ctx);

// Returns the resolved backend name as a stable string: "lid-cld3" or
// "lid-fasttext". Useful for log messages and the diff-harness branch
// when a caller wants to format output differently per backend.
const char* text_lid_backend(const struct text_lid_context* ctx);

#ifdef __cplusplus
}
#endif
