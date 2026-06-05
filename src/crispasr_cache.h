// crispasr_cache.h — shared cache directory + download helper.
//
// Three places in the unified CLI need to download a small companion file
// from HuggingFace on first use and cache it under ~/.cache/crispasr/:
//
//   * crispasr_model_mgr — `-m auto` model resolution
//   * crispasr_lid       — whisper-tiny LID model auto-download
//   * crispasr_vad       — Silero VAD model auto-download
//
// This header centralises the directory layout, the existence/zombie check,
// and the download logic (WinHTTP on Windows, curl/wget on POSIX) so each
// consumer is a one-liner over `crispasr_cache::ensure_cached_file(...)`.

#pragma once

#include <string>

namespace crispasr_cache {

// Return the cache directory (creating it if missing).
//   • If cache_dir_override is non-empty, use it directly (creating the leaf
//     directory if it does not exist; parents must already exist).
//   • Otherwise use the platform default:
//       - POSIX : $HOME/.cache/crispasr
//       - Windows: %USERPROFILE%/.cache/crispasr
//                  (%HOME% / %LOCALAPPDATA% as fallbacks if USERPROFILE unset)
std::string dir(const std::string& cache_dir_override = "");

// True iff `path` exists AND is non-zero bytes. Treats 0-byte zombies
// (left behind by an interrupted earlier download) as missing so the
// next attempt retries the fetch instead of handing a corrupted file
// to a model loader.
bool file_present(const std::string& path);

// Download `url` into `dest`.
//   Windows: tries WinHTTP first (built-in, handles HTTPS + redirects natively,
//            no shell-quoting issues), then falls back to curl, then wget.
//   POSIX  : tries curl, then wget.
// Returns true iff the file is present and non-empty after the download.
// `quiet=true` suppresses progress bars; failure messages always go to stderr.
bool fetch(const std::string& url, const std::string& dest, bool quiet);

// Search all well-known locations (canonical cache, CRISPASR_MODELS_DIR env,
// ~/.cache/huggingface/hub, and any compile-time extra dirs) for `filename`
// without downloading. Returns the first hit's absolute path, or "" if not
// found. Used by dry-run preview to match the "cached/local" status that
// ensure_cached_file() would report at runtime.
std::string probe_cached_file(const std::string& filename, const std::string& cache_dir_override = "");

// Composite helper: if `dest` (= dir(cache_dir_override) + "/" + filename)
// already satisfies file_present(), return its path immediately. Otherwise
// invoke fetch() to populate it. Returns the absolute path on success or an
// empty string on failure.
std::string ensure_cached_file(const std::string& filename, const std::string& url, bool quiet,
                               const char* pretty_label = "crispasr", const std::string& cache_dir_override = "");

} // namespace crispasr_cache
