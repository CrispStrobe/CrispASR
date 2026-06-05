// text_lid_dispatch.cpp — auto-routing façade for text-LID GGUFs.
//
// Peeks `general.architecture` once at load time, then dispatches each
// public call to the appropriate backend's C ABI. Both `lid_fasttext`
// and `lid_cld3` already expose the same shape (init / free / predict
// / predict_topk / variant / n_labels), so the dispatcher is a flat
// tag-switch — no function-pointer table, no virtual base, no
// per-call overhead beyond a single integer compare.

#include "text_lid_dispatch.h"

#include "core/gguf_loader.h"
#include "crispasr_cache.h"
#include "crispasr_model_registry.h"
#include "lid_cld3.h"
#include "lid_fasttext.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>

namespace {

enum Backend { kBackendUnknown = 0, kBackendFastText, kBackendCld3 };

// Stable, never-freed strings returned from text_lid_backend().
const char* backend_name(Backend b) {
    switch (b) {
    case kBackendFastText:
        return "lid-fasttext";
    case kBackendCld3:
        return "lid-cld3";
    default:
        return "unknown";
    }
}

} // namespace

struct text_lid_context {
    Backend backend = kBackendUnknown;
    lid_fasttext_context* ft = nullptr;
    lid_cld3_context* cld = nullptr;
};

extern "C" struct text_lid_context* text_lid_init_from_file(const char* gguf_path, int n_threads) {
    if (!gguf_path) {
        fprintf(stderr, "text_lid: null gguf_path\n");
        return nullptr;
    }

    // Peek `general.architecture` to pick the backend. We open the
    // metadata pass via core_gguf and free it before delegating to
    // the chosen backend (which will reopen the file itself — cheap,
    // and avoids passing a half-owned gguf_context around).
    gguf_context* meta = core_gguf::open_metadata(gguf_path);
    if (!meta) {
        fprintf(stderr, "text_lid: failed to open metadata for '%s'\n", gguf_path);
        return nullptr;
    }
    std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);

    Backend chosen = kBackendUnknown;
    if (arch == "lid-cld3") {
        chosen = kBackendCld3;
    } else if (arch == "lid-fasttext") {
        chosen = kBackendFastText;
    } else {
        fprintf(stderr,
                "text_lid: '%s' has unsupported architecture '%s'. "
                "Expected 'lid-cld3' (Google CLD3) or 'lid-fasttext' (GlotLID/LID-176). "
                "Re-run the appropriate converter (models/convert-cld3-to-gguf.py or "
                "models/convert-glotlid-to-gguf.py) and pass the resulting GGUF.\n",
                gguf_path, arch.c_str());
        return nullptr;
    }

    auto* ctx = new text_lid_context;
    ctx->backend = chosen;
    if (chosen == kBackendCld3) {
        ctx->cld = lid_cld3_init_from_file(gguf_path, n_threads);
        if (!ctx->cld) {
            delete ctx;
            return nullptr;
        }
    } else {
        ctx->ft = lid_fasttext_init_from_file(gguf_path, n_threads);
        if (!ctx->ft) {
            delete ctx;
            return nullptr;
        }
    }
    return ctx;
}

extern "C" void text_lid_free(struct text_lid_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cld)
        lid_cld3_free(ctx->cld);
    if (ctx->ft)
        lid_fasttext_free(ctx->ft);
    delete ctx;
}

extern "C" const char* text_lid_predict(struct text_lid_context* ctx, const char* utf8_text, float* confidence) {
    if (!ctx || !utf8_text)
        return nullptr;
    switch (ctx->backend) {
    case kBackendCld3:
        return lid_cld3_predict(ctx->cld, utf8_text, confidence);
    case kBackendFastText:
        return lid_fasttext_predict(ctx->ft, utf8_text, confidence);
    default:
        return nullptr;
    }
}

extern "C" int text_lid_predict_topk(struct text_lid_context* ctx, const char* utf8_text, int k,
                                     const char** out_labels, float* out_scores) {
    if (!ctx || !utf8_text || k <= 0 || !out_labels || !out_scores)
        return 0;
    switch (ctx->backend) {
    case kBackendCld3:
        return lid_cld3_predict_topk(ctx->cld, utf8_text, k, out_labels, out_scores);
    case kBackendFastText:
        return lid_fasttext_predict_topk(ctx->ft, utf8_text, k, out_labels, out_scores);
    default:
        return 0;
    }
}

extern "C" const char* text_lid_variant(const struct text_lid_context* ctx) {
    if (!ctx)
        return "";
    switch (ctx->backend) {
    case kBackendCld3:
        return lid_cld3_variant(ctx->cld);
    case kBackendFastText:
        return lid_fasttext_variant(ctx->ft);
    default:
        return "";
    }
}

extern "C" int text_lid_n_labels(const struct text_lid_context* ctx) {
    if (!ctx)
        return 0;
    switch (ctx->backend) {
    case kBackendCld3:
        return lid_cld3_n_labels(ctx->cld);
    case kBackendFastText:
        return lid_fasttext_n_labels(ctx->ft);
    default:
        return 0;
    }
}

extern "C" int text_lid_dim(const struct text_lid_context* ctx) {
    if (!ctx)
        return 0;
    switch (ctx->backend) {
    case kBackendCld3:
        return lid_cld3_dim_total(ctx->cld); // 80 (concat dim)
    case kBackendFastText:
        return lid_fasttext_dim(ctx->ft);
    default:
        return 0;
    }
}

extern "C" const char* text_lid_backend(const struct text_lid_context* ctx) {
    return ctx ? backend_name(ctx->backend) : backend_name(kBackendUnknown);
}

// ===========================================================================
// Path resolution + auto-download (C++-only helper, see header).
// ===========================================================================

namespace {

bool path_exists(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && st.st_size > 0;
}

std::string basename_of(const std::string& p) {
    auto pos = p.find_last_of("/\\");
    return pos == std::string::npos ? p : p.substr(pos + 1);
}

// Map `auto[:variant]` shorthand to a registry backend name. Default
// (bare `auto` or `auto:cld3`) is CLD3 — smallest, Apache-2.0.
const char* auto_variant_to_backend(const std::string& arg) {
    if (arg == "auto" || arg == "auto:cld3" || arg == "auto:lid-cld3")
        return "lid-cld3";
    if (arg == "auto:glotlid" || arg == "auto:lid-glotlid")
        return "lid-glotlid";
    if (arg == "auto:lid-fasttext176" || arg == "auto:fasttext176" || arg == "auto:lid176")
        return "lid-fasttext176";
    return nullptr;
}

} // namespace

std::string text_lid_resolve_path(const std::string& arg, const std::string& cache_dir_override, bool quiet) {
    if (arg.empty()) {
        fprintf(stderr, "text_lid: empty model path\n");
        return "";
    }

    // 1. `auto[:variant]` → registry lookup → download.
    if (arg.rfind("auto", 0) == 0 && (arg.size() == 4 || arg[4] == ':')) {
        const char* backend = auto_variant_to_backend(arg);
        if (!backend) {
            fprintf(stderr,
                    "text_lid: unknown auto variant '%s'. "
                    "Supported: auto, auto:cld3, auto:glotlid, auto:lid-fasttext176\n",
                    arg.c_str());
            return "";
        }
        CrispasrRegistryEntry entry;
        if (!crispasr_registry_lookup(backend, entry)) {
            fprintf(stderr, "text_lid: backend '%s' not in registry\n", backend);
            return "";
        }
        return crispasr_cache::ensure_cached_file(entry.filename, entry.url, quiet, "text-lid", cache_dir_override);
    }

    // 2. Path exists as-is — use it.
    if (path_exists(arg))
        return arg;

    // 3. Try registry lookup by basename — handy when the user pasted a
    //    canonical filename (`cld3-f16.gguf`, `lid-glotlid-f16.gguf`, …)
    //    without first downloading. Same fallback the main `-m` path uses.
    CrispasrRegistryEntry entry;
    if (crispasr_registry_lookup_by_filename(basename_of(arg), entry)) {
        std::string cached =
            crispasr_cache::ensure_cached_file(entry.filename, entry.url, quiet, "text-lid", cache_dir_override);
        if (!cached.empty())
            return cached;
    }

    fprintf(stderr,
            "text_lid: model '%s' not found and no registry hit. "
            "Pass an existing path, or use one of: auto, auto:cld3, auto:glotlid, auto:lid-fasttext176.\n",
            arg.c_str());
    return "";
}
