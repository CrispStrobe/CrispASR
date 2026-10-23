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
#include <vector>

enum class CrispasrRegistryArtifactKind {
    Primary = 0,
    Companion = 1,
    Extra = 2,
};

struct CrispasrRegistryArtifact {
    CrispasrRegistryArtifactKind kind;
    std::string filename;
    std::string url;
    std::string approx_size; // empty when the registry does not record one
};

struct CrispasrRegistryBundle {
    std::string backend; // canonical registry key (aliases are resolved)
    std::string license;
    bool requires_license_acceptance = false;
    std::vector<CrispasrRegistryArtifact> artifacts;
};

struct CrispasrRegistryEntry {
    std::string backend;
    std::string filename;
    std::string url;         // direct HuggingFace resolve URL
    std::string approx_size; // human-readable (e.g. "~467 MB")
    std::string companion_filename;
    std::string companion_url;
    std::string companion_approx_size; // size of the companion; empty = same as approx_size
    std::string license;               // empty = permissive; non-empty = printed to stderr on download
};

/// Normalise a registry `license` string to its leading SPDX-ish tag,
/// lowercased (e.g. "CC-BY-NC-4.0 — NON-COMMERCIAL ..." -> "cc-by-nc-4.0").
std::string crispasr_license_tag(const std::string& license);

/// True when the licence designates a RESTRICTED model the user must
/// explicitly accept before download: cc-by-nc-*, gemma, llama*,
/// qwen-research, mistral-ai-research, lfm1.0, omnivoice-terms,
/// nvidia-open-model-license, other.
bool crispasr_license_requires_acceptance(const std::string& license);

/// True when `accepted` attests this licence: the exact tag, or "all" / "*".
/// Falls back to the CRISPASR_ACCEPT_LICENSE env var.
/// Acceptance affirms eligibility and compliance; it grants no extra rights.
bool crispasr_license_accepted(const std::string& license, const std::string& accepted);

/// Download a CrispASR-managed auxiliary artifact after applying the same
/// license gate and notice policy as registry models. `accepted_license` may
/// be empty to use CRISPASR_ACCEPT_LICENSE.
std::string crispasr_managed_download(const std::string& filename, const std::string& url, const std::string& license,
                                      bool quiet, const char* label, const std::string& cache_dir_override = "",
                                      const std::string& accepted_license = "");

/// Look up a registry entry by backend name. Returns true on hit.
bool crispasr_registry_lookup(const std::string& backend, CrispasrRegistryEntry& out,
                              const std::string& preferred_quant = "");

/// Return the default artifact bundle for a backend: primary model, inline
/// companion (if any), then extra companions.
///
/// This is the bundle `-m auto` downloads *with no quant suffix*. It
/// deliberately does not apply a preferred quant or infer a recommendation, so
/// it does NOT reproduce `-m auto:<quant>`: that path threads `preferred_quant`
/// through crispasr_registry_lookup(), which rewrites both the filename and the
/// URL of the primary and its companion. Callers mirroring `-m auto:<quant>`
/// must go through crispasr_registry_lookup() instead — the filenames here
/// would be wrong for them.
///
/// Returns false when the backend has no registry entry.
bool crispasr_registry_default_bundle(const std::string& backend, CrispasrRegistryBundle& out);

/// Number of entries in the static registry.
int crispasr_registry_count();

/// Get the i-th entry (0..count-1). Returns false on out-of-range or when
/// preferred_quant is not one of that model's explicitly qualified profiles.
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
/// `accepted_license` attests acceptance of a RESTRICTED licence (cc-by-nc-*,
/// gemma, llama*, lfm1.0, other): pass the exact SPDX-ish tag, or "all".
/// Falls back to the `CRISPASR_ACCEPT_LICENSE` env var. For such models
/// `allow_download` alone is NOT sufficient — without acceptance the download
/// is refused (non-TTY) or prompted (TTY), BEFORE any bytes are fetched.
/// Mirrors CrispEmbed's crispembed_mgr::resolve_model.
std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override = "", bool allow_download = false,
                                   const std::string& preferred_quant = "", const std::string& accepted_license = "");
