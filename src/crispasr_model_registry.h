// crispasr_model_registry.h — known-model registry lookup.
//
// Tiny table mapping a backend name (whisper, parakeet, canary, voxtral,
// voxtral4b, granite, granite-4.1, qwen3, cohere, wav2vec2) to the canonical GGUF
// filename + HuggingFace download URL + approximate size. Used by:
//
//   * the CLI's `-m auto` / `-m default` resolver
//   * file-not-found → offer-download flow (CLI shim adds TTY prompt)
//   * wrapper bindings that want to auto-download a stock model
//
// The interactive "prompt on a TTY" behaviour stays in the CLI shim
// (`examples/cli/crispasr_model_mgr_cli.cpp`). The library interface
// is non-interactive: callers decide policy, the library resolves.

#pragma once

#include <string>

struct CrispasrRegistryEntry {
    std::string backend;
    std::string filename;
    std::string url;         // direct HuggingFace resolve URL
    std::string approx_size; // human-readable (e.g. "~467 MB")
    std::string companion_filename;
    std::string companion_url;
};

/// Look up a registry entry by backend name. Returns true on hit.
bool crispasr_registry_lookup(const std::string& backend, CrispasrRegistryEntry& out,
                              const std::string& preferred_quant = "");

/// Number of entries in the static registry.
int crispasr_registry_count();

/// Get the i-th entry (0..count-1). Returns false on out-of-range.
/// Iterating from 0 to count-1 visits every entry in declaration order.
bool crispasr_registry_get_at(int i, CrispasrRegistryEntry& out, const std::string& preferred_quant = "");

/// Look up by filename. Exact match first, then fuzzy (substring) match.
/// Used by the file-not-found path to suggest the canonical URL for a
/// user-supplied filename.
bool crispasr_registry_lookup_by_filename(const std::string& filename, CrispasrRegistryEntry& out,
                                          const std::string& preferred_quant = "");

/// Scan the cache directory for any already-downloaded model from the
/// registry and return the first hit, preferring backends in this order:
/// whisper > parakeet > canary > cohere > voxtral > voxtral4b > granite
/// > granite-4.1 > qwen3 > wav2vec2. Populates `out` on success. Returns true if a
/// cached model was found, false if the cache is empty. Intended for the
/// `-m auto` path so a user who already has *any* model doesn't trigger
/// a fresh download of whisper-base.
bool crispasr_find_cached_model(CrispasrRegistryEntry& out, const std::string& cache_dir_override = "",
                                const std::string& preferred_quant = "");

/// Non-interactive resolve. If `model_arg` is a concrete file path that
/// exists, returns it unchanged. If it's "auto" / "default", downloads
/// the backend's canonical GGUF into the cache directory.
///
/// When the file is missing and `allow_download` is true, also downloads
/// it if the filename matches a registry entry. When `allow_download` is
/// false, returns `model_arg` untouched and leaves it to the caller to
/// decide what to do (prompt on TTY, raise an error, etc.).
///
/// Returns an empty string on unrecoverable failure.
std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override = "", bool allow_download = false,
                                   const std::string& preferred_quant = "");
