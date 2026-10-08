// crispasr_model_mgr_cli.cpp — CLI model-resolve with TTY prompting.
//
// Delegates non-interactive resolution to the shared library; layers a
// `Download now?` prompt on top when stdin is a TTY.

#include "crispasr_model_mgr_cli.h"
#include "crispasr_cache.h"
#include "crispasr_model_registry.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>

#if defined(_WIN32)
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

static bool parse_auto_quant_spec(const std::string& spec, std::string& base, std::string& quant) {
    const size_t pos = spec.find(':');
    if (pos == std::string::npos)
        return false;
    const std::string prefix = spec.substr(0, pos);
    if (prefix != "auto" && prefix != "default")
        return false;
    const std::string suffix = spec.substr(pos + 1);
    if (suffix.empty())
        return false;
    base = prefix;
    quant = suffix;
    return true;
}

// A bare explicit filename means "this exact file", but users commonly keep
// those files in --cache-dir / CRISPASR_MODELS_DIR rather than the shell's
// current directory.  Search the normal model roots before consulting the
// registry; otherwise an unregistered Piper voice can be silently replaced by
// the backend's unrelated default voice (#397).
static std::string probe_explicit_model_filename(const std::string& model_arg, const std::string& cache_dir_override) {
    if (model_arg.empty() || model_arg.find_first_of("/\\") != std::string::npos)
        return {};
    return crispasr_cache::probe_cached_file(model_arg, cache_dir_override);
}

static CrispasrResolvePreview build_preview(const std::string& model_arg, const std::string& backend_name,
                                            const std::string& cache_dir_override, const std::string& preferred_quant,
                                            bool ignore_cache) {
    std::string effective_model_arg = model_arg;
    std::string effective_quant = preferred_quant;
    std::string auto_base;
    std::string auto_quant;
    if (effective_quant.empty() && parse_auto_quant_spec(effective_model_arg, auto_base, auto_quant)) {
        effective_model_arg = auto_base;
        effective_quant = auto_quant;
    }

    CrispasrResolvePreview out;
    out.requested = model_arg;
    out.backend = backend_name;

    auto cache_path_for = [&](const std::string& filename) {
        return crispasr_cache::dir(cache_dir_override) + "/" + filename;
    };

    if (effective_model_arg != "auto" && effective_model_arg != "default") {
        FILE* f = fopen(effective_model_arg.c_str(), "rb");
        if (f) {
            fclose(f);
            out.exists_locally = true;
            out.resolved_path = effective_model_arg;
            out.filename = effective_model_arg;
            return out;
        }

        const std::string cached = probe_explicit_model_filename(effective_model_arg, cache_dir_override);
        if (!cached.empty()) {
            out.exists_locally = true;
            out.resolved_path = cached;
            out.filename = effective_model_arg;
            return out;
        }
    }

    CrispasrRegistryEntry match;
    bool have_match = false;
    if (effective_model_arg == "auto" || effective_model_arg == "default") {
        have_match = crispasr_registry_lookup(backend_name, match, effective_quant);
    } else {
        // Mirror crispasr_resolve_model's match priority exactly, or the preview
        // lies: (1) exact filename/companion, (2) backend-key match on the
        // literal -m arg (so sub-variant keys like parakeet-tdt_ctc-110m resolve
        // to their own entry rather than being shadowed by the filename-inferred
        // backend), (3) fallback to the --backend name.
        have_match = crispasr_registry_lookup_by_filename(effective_model_arg, match, effective_quant);
        if (!have_match)
            have_match = crispasr_registry_lookup(effective_model_arg, match, effective_quant);
        if (!have_match && !backend_name.empty())
            have_match = crispasr_registry_lookup(backend_name, match, effective_quant);
    }
    if (!have_match) {
        out.unresolved = true;
        out.resolved_path = effective_model_arg;
        return out;
    }

    out.matched_registry = true;
    out.filename = match.filename;
    out.url = match.url;
    out.approx_size = match.approx_size;
    out.resolved_path = cache_path_for(match.filename);

    if (!ignore_cache) {
        const std::string found = crispasr_cache::probe_cached_file(match.filename, cache_dir_override);
        if (!found.empty()) {
            out.exists_locally = true;
            out.resolved_path = found;
            return out;
        }
    }

    out.would_download = true;
    return out;
}

CrispasrResolvePreview crispasr_preview_model_cli(const std::string& model_arg, const std::string& backend_name,
                                                  const std::string& cache_dir_override,
                                                  const std::string& preferred_quant, bool ignore_cache) {
    return build_preview(model_arg, backend_name, cache_dir_override, preferred_quant, ignore_cache);
}

std::string crispasr_resolve_model_cli(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                       const std::string& cache_dir_override, bool auto_download,
                                       const std::string& preferred_quant, const std::string& accepted_license) {
    std::string effective_model_arg = model_arg;
    std::string effective_quant = preferred_quant;
    std::string auto_base;
    std::string auto_quant;
    if (effective_quant.empty() && parse_auto_quant_spec(effective_model_arg, auto_base, auto_quant)) {
        effective_model_arg = auto_base;
        effective_quant = auto_quant;
    }

    // "auto"/"default" and already-on-disk paths: library handles them.
    if (effective_model_arg == "auto" || effective_model_arg == "default") {
        return crispasr_resolve_model(effective_model_arg, backend_name, quiet, cache_dir_override, auto_download,
                                      effective_quant, accepted_license);
    }

    // Concrete path: check existence ourselves so we can interpose the
    // TTY prompt before asking the library to download.
    FILE* f = fopen(effective_model_arg.c_str(), "rb");
    if (f) {
        fclose(f);
        return effective_model_arg;
    }
    const int open_errno = errno;

    const std::string cached_explicit = probe_explicit_model_filename(effective_model_arg, cache_dir_override);
    if (!cached_explicit.empty()) {
        if (!quiet)
            fprintf(stderr, "crispasr: found explicit model in model search path: %s\n", cached_explicit.c_str());
        return cached_explicit;
    }

    // fopen failed. Distinguish "no such entry" (maybe a registry model *name*
    // to download — handled below) from "the entry exists but we can't open
    // it" (a dangling/unreadable symlink, a permission problem, an unmounted
    // volume). For the latter, do NOT silently substitute a downloadable
    // default: that masks a real "your file is unreadable" error as "loaded
    // some other model" (observed with a -m pointing at a symlink whose target
    // was offline — the whisper default ggml-base.bin was loaded instead).
    // Surface a clear error and return the path so the load layer fails on it.
    bool entry_exists = false;
#if !defined(_WIN32)
    struct stat lst;
    if (lstat(effective_model_arg.c_str(), &lst) == 0)
        entry_exists = true; // the path names a real filesystem entry
#endif
    if (entry_exists || open_errno == EACCES || open_errno == EPERM) {
        fprintf(stderr, "crispasr: cannot open model file '%s': %s\n", effective_model_arg.c_str(),
                std::strerror(open_errno));
#if !defined(_WIN32)
        if (entry_exists && S_ISLNK(lst.st_mode))
            fprintf(stderr, "  (it is a symlink — its target may be missing or on an unmounted/unreadable volume)\n");
#endif
        fprintf(stderr, "  Refusing to substitute a downloadable default for an explicit --model path.\n"
                        "  Fix the file/permissions, or pass -m auto to download a model.\n");
        return effective_model_arg;
    }

    // File missing — see whether the registry recognises it.
    // Match priority:
    //   1. exact filename / known-companion match (e.g. -m parakeet-tdt-0.6b-v2-q4_k.gguf)
    //   2. backend-key match on the literal -m arg (e.g. -m parakeet-v2 → the parakeet-v2 entry)
    //   3. fallback: backend name passed via --backend (or inferred from filename)
    // Step 2 must precede step 3, otherwise the CLI's filename-inferred
    // backend (always "parakeet" for any "parakeet*" arg) would shadow
    // sub-variant keys like "parakeet-v2" / "parakeet-tdt-1.1b" / etc.
    CrispasrRegistryEntry match;
    bool have_match = crispasr_registry_lookup_by_filename(effective_model_arg, match, effective_quant);
    if (!have_match)
        have_match = crispasr_registry_lookup(effective_model_arg, match, effective_quant);
    bool backend_fallback = false; // step 3: the -m arg itself matched NOTHING
    if (!have_match && !backend_name.empty()) {
        have_match = crispasr_registry_lookup(backend_name, match, effective_quant);
        backend_fallback = have_match;
    }

    if (backend_fallback) {
        // The user's -m matched nothing; only --backend did. Silently handing
        // back the backend's registry default here meant a typo'd path
        // transcribed with a DIFFERENT model at rc 0 (found 2026-09-02 while
        // execution-verifying the getting-started tutorial). A slash makes the
        // intent unambiguous — that is a PATH, and a missing path is an error,
        // same posture as the unreadable-entry branch above. A bare name keeps
        // the historical substitution but says so loudly.
        if (effective_model_arg.find('/') != std::string::npos || effective_model_arg.find('\\') != std::string::npos) {
            fprintf(stderr,
                    "crispasr: model file '%s' does not exist and matches no registry entry.\n"
                    "  Refusing to substitute the '%s' registry default for an explicit --model path.\n"
                    "  Fix the path, or pass -m auto to use the registry default.\n",
                    effective_model_arg.c_str(), backend_name.c_str());
            return effective_model_arg;
        }
        fprintf(stderr,
                "crispasr: WARNING: '-m %s' does not exist and matches no registry entry — "
                "substituting the '%s' registry default '%s'. Pass -m auto to silence this.\n",
                effective_model_arg.c_str(), backend_name.c_str(), match.filename.c_str());
    }

    if (!have_match) {
        // Nothing to download — return the arg and let the load layer
        // produce a real error.
        return effective_model_arg;
    }

    const std::string cached_path = crispasr_cache::dir(cache_dir_override) + "/" + match.filename;
    if (crispasr_cache::file_present(cached_path)) {
        if (!match.license.empty()) {
            const bool is_nc = crispasr_license_tag(match.license).rfind("cc-by-nc", 0) == 0;
            if (is_nc)
                fprintf(stderr,
                        "crispasr: WARNING: %s is licensed %s — NON-COMMERCIAL USE ONLY.\n"
                        "  By loading this model you confirm you will not use it for commercial purposes.\n",
                        match.filename.c_str(), match.license.c_str());
            else
                fprintf(stderr, "crispasr: note: %s is licensed %s — review and comply with the upstream terms.\n",
                        match.filename.c_str(), match.license.c_str());
        }
        if (auto_download)
            return crispasr_resolve_model(effective_model_arg, match.backend, quiet, cache_dir_override, true,
                                          effective_quant, accepted_license);
        return cached_path;
    }

    fprintf(stderr, "crispasr: model '%s' not found locally.\n", effective_model_arg.c_str());
    if (match.approx_size.empty())
        fprintf(stderr, "  Available for download: %s\n", match.filename.c_str());
    else
        fprintf(stderr, "  Available for download: %s (%s)\n", match.filename.c_str(), match.approx_size.c_str());
    if (!match.license.empty()) {
        const bool is_nc = crispasr_license_tag(match.license).rfind("cc-by-nc", 0) == 0;
        if (is_nc)
            fprintf(stderr, "  LICENSE: %s — NON-COMMERCIAL USE ONLY. Do not use for commercial purposes.\n",
                    match.license.c_str());
        else
            fprintf(stderr, "  License: %s\n", match.license.c_str());
    }

    // Restricted licences need explicit acceptance; --auto-download alone is
    // NOT sufficient (mirrors CrispEmbed).
    const bool restricted = !match.license.empty() && crispasr_license_requires_acceptance(match.license);
    const bool accepted = restricted && crispasr_license_accepted(match.license, accepted_license);
    if (restricted && !accepted) {
        fprintf(stderr, "  Refusing: pass --accept-license %s (or set CRISPASR_ACCEPT_LICENSE).\n",
                crispasr_license_tag(match.license).c_str());
        return effective_model_arg;
    }

    bool do_download = false;
    if (auto_download && (!restricted || accepted)) {
        do_download = true;
        fprintf(stderr, "  Auto-downloading (--auto-download is set)...\n");
    } else if (isatty(fileno(stdin))) {
        fprintf(stderr, "  Download now? [Y/n] ");
        fflush(stderr);
        char c = 'y';
        int ch = fgetc(stdin);
        if (ch != EOF && ch != '\n')
            c = (char)ch;
        do_download = (c == 'y' || c == 'Y' || c == '\n');
    } else {
        fprintf(stderr, "  Use --auto-download or -m auto to download automatically.\n");
    }

    if (do_download) {
        return crispasr_resolve_model(effective_model_arg, match.backend, quiet, cache_dir_override, true,
                                      effective_quant, accepted_license);
    }

    return effective_model_arg;
}
