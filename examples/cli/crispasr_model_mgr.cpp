// crispasr_model_mgr.cpp — model registry + dispatch over the shared
// crispasr_cache helper.

#include "crispasr_model_mgr.h"
#include "crispasr_cache.h"

#include <cstdio>
#include <string>
#if defined(_WIN32)
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace {

// Per-backend canonical model. Extend as new backends are wired up.
struct registry_entry {
    const char* backend;
    const char* filename;
    const char* url; // direct download URL (HuggingFace resolve link)
    const char* approx_size;
};

constexpr registry_entry k_registry[] = {
    // Whisper base.en (default for the unified --backend whisper path).
    // Users who want a different size / language can either pass an
    // explicit path or fall back to the historical whisper-cli flow on
    // their own .bin files.
    {"whisper", "ggml-base.en.bin", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
     "~147 MB"},
    // Parakeet TDT 0.6B v3 quantised
    {"parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf", "~467 MB"},
    // Canary 1B v2 quantised
    {"canary", "canary-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf", "~600 MB"},
    // Voxtral Mini 3B 2507
    {"voxtral", "voxtral-mini-3b-2507-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF/resolve/main/voxtral-mini-3b-2507-q4_k.gguf", "~2.5 GB"},
    // Voxtral Mini 4B Realtime
    {"voxtral4b", "voxtral-mini-4b-realtime-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-4b-realtime-GGUF/resolve/main/voxtral-mini-4b-realtime-q4_k.gguf",
     "~3.3 GB"},
    // Granite 4.0 1B Speech (new converter — includes BPE merges so the
    // dispatcher's --translate path works end-to-end).
    {"granite", "granite-speech-4.0-1b-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.0-1b-GGUF/resolve/main/granite-speech-4.0-1b-q4_k.gguf", "~2.94 GB"},
    // Qwen3-ASR 0.6B (smaller, faster default for the dispatcher's
    // -m auto path; the 1.7B is much more accurate but ~3x larger)
    {"qwen3", "qwen3-asr-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf", "~500 MB"},
    // Cohere Transcribe 03-2026
    {"cohere", "cohere-transcribe-q4_k.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf", "~550 MB"},
    // Wav2Vec2 English (CTC)
    {"wav2vec2", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB"},
};

// Lookup by filename (for the "file not found → offer download" path).
const registry_entry* lookup_by_filename(const std::string& filename) {
    // Strip directory prefix if present
    std::string base = filename;
    auto slash = base.rfind('/');
    if (slash != std::string::npos)
        base = base.substr(slash + 1);
    auto bslash = base.rfind('\\');
    if (bslash != std::string::npos)
        base = base.substr(bslash + 1);

    for (const auto& e : k_registry) {
        if (base == e.filename)
            return &e;
    }
    // Fuzzy match: if the filename contains a registry entry's basename
    for (const auto& e : k_registry) {
        if (base.find(e.filename) != std::string::npos)
            return &e;
    }
    return nullptr;
}

const registry_entry* lookup(const std::string& backend) {
    for (const auto& e : k_registry) {
        if (backend == e.backend)
            return &e;
    }
    return nullptr;
}

} // namespace

std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override, bool auto_download) {
    // Pass-through for explicit paths that exist on disk.
    if (model_arg != "auto" && model_arg != "default") {
        // Check if file exists
        FILE* f = fopen(model_arg.c_str(), "rb");
        if (f) {
            fclose(f);
            return model_arg;
        }

        // File not found — try to match against our registry and offer download
        const registry_entry* match = lookup_by_filename(model_arg);
        if (!match) {
            // Also try backend-based lookup if we know the backend
            if (!backend_name.empty())
                match = lookup(backend_name);
        }

        if (match) {
            fprintf(stderr,
                    "crispasr: model '%s' not found locally.\n"
                    "  Available for download: %s (%s)\n",
                    model_arg.c_str(), match->filename, match->approx_size);

            // Check if auto-download is requested (model_arg == filename from registry)
            // or prompt the user on a TTY
            bool do_download = false;
            if (auto_download) {
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
                return crispasr_cache::ensure_cached_file(match->filename, match->url, quiet, "crispasr",
                                                          cache_dir_override);
            }
        }

        return model_arg; // return as-is, will fail at load time with a clear error
    }

    const registry_entry* e = lookup(backend_name);
    if (!e) {
        fprintf(stderr,
                "crispasr: error: -m auto is not supported for backend '%s' "
                "(no default model registered)\n",
                backend_name.c_str());
        return "";
    }

    if (!quiet) {
        fprintf(stderr, "crispasr: resolving %s (%s) via -m auto\n", e->filename, e->approx_size);
    }
    return crispasr_cache::ensure_cached_file(e->filename, e->url, quiet, "crispasr", cache_dir_override);
}
