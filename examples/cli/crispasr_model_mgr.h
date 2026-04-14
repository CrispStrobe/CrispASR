// crispasr_model_mgr.h — model path resolution and auto-download.
//
// Handles the -m "auto" convenience: when the user passes "-m auto" or
// "-m default", the backend's canonical GGUF is downloaded into
// ~/.cache/crispasr/ on first use.
//
// Downloads shell out to `curl` (preferred) or `wget`. No Python, no libcurl
// link dependency. Works on Linux, macOS, and Windows 10+ where curl is part
// of the base system.

#pragma once

#include <string>

// Resolve a user-supplied -m argument to a concrete file path.
//
// If model_arg is "auto" or "default", this consults the per-backend
// registry to find the canonical URL+filename and downloads it to
// the cache directory if not already cached. Otherwise the argument is
// returned unchanged.
//
// cache_dir_override: when non-empty, overrides the default cache directory
// (~/.cache/crispasr). Corresponds to --cache-dir.
//
// Returns an empty string on failure.
std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override = "", bool auto_download = false);
