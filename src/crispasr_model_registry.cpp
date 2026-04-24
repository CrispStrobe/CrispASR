// crispasr_model_registry.cpp — implementation. See the header.

#include "crispasr_model_registry.h"
#include "crispasr_cache.h"

#include <cstdio>
#include <vector>

namespace {

struct Entry {
    const char* backend;
    const char* filename;
    const char* url;
    const char* approx_size;
};

// Keep entries aligned with what the CLI-only registry used to ship.
// Adding a new backend: one row here + a PUBLIC-link in src/CMakeLists.txt.
constexpr Entry k_registry[] = {
    // Multilingual by default so `-m auto` works for non-English audio.
    // Users who want the smaller English-only build can pass
    // `-m models/ggml-base.en.bin` explicitly (once downloaded).
    {"whisper", "ggml-base.bin", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", "~147 MB"},
    {"parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf", "~467 MB"},
    {"canary", "canary-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf", "~600 MB"},
    {"voxtral", "voxtral-mini-3b-2507-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF/resolve/main/voxtral-mini-3b-2507-q4_k.gguf", "~2.5 GB"},
    {"voxtral4b", "voxtral-mini-4b-realtime-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-4b-realtime-GGUF/resolve/main/voxtral-mini-4b-realtime-q4_k.gguf",
     "~3.3 GB"},
    {"granite", "granite-speech-4.0-1b-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.0-1b-GGUF/resolve/main/granite-speech-4.0-1b-q4_k.gguf", "~2.94 GB"},
    {"qwen3", "qwen3-asr-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf", "~500 MB"},
    {"cohere", "cohere-transcribe-q4_k.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf", "~550 MB"},
    {"wav2vec2", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB"},
    {"omniasr", "omniasr-ctc-1b-q4_k.gguf",
     "https://huggingface.co/cstr/omniASR-CTC-1B-GGUF/resolve/main/omniasr-ctc-1b-q4_k.gguf", "~551 MB"},
    {"omniasr-llm", "omniasr-llm-300m-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF/resolve/main/omniasr-llm-300m-v2-q4_k.gguf", "~580 MB"},
    {"hubert", "hubert-large-ls960-ft-q4_k.gguf",
     "https://huggingface.co/cstr/hubert-large-ls960-ft-GGUF/resolve/main/hubert-large-ls960-ft-q4_k.gguf", "~200 MB"},
    {"data2vec", "data2vec-audio-base-960h-q4_k.gguf",
     "https://huggingface.co/cstr/data2vec-audio-960h-GGUF/resolve/main/data2vec-audio-base-960h-q4_k.gguf",
     "~60 MB"},
    {"vibevoice", "vibevoice-asr-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-asr-GGUF/resolve/main/vibevoice-asr-q4_k.gguf", "~4.5 GB"},
};

const Entry* find_by_backend(const std::string& backend) {
    for (const auto& e : k_registry)
        if (backend == e.backend)
            return &e;
    return nullptr;
}

std::string basename_of(const std::string& p) {
    std::string base = p;
    auto slash = base.rfind('/');
    if (slash != std::string::npos)
        base = base.substr(slash + 1);
    auto bslash = base.rfind('\\');
    if (bslash != std::string::npos)
        base = base.substr(bslash + 1);
    return base;
}

const Entry* find_by_filename(const std::string& filename) {
    const std::string base = basename_of(filename);
    for (const auto& e : k_registry)
        if (base == e.filename)
            return &e;
    for (const auto& e : k_registry)
        if (base.find(e.filename) != std::string::npos)
            return &e;
    return nullptr;
}

void fill(CrispasrRegistryEntry& out, const Entry& e) {
    out.backend = e.backend;
    out.filename = e.filename;
    out.url = e.url;
    out.approx_size = e.approx_size;
}

} // namespace

bool crispasr_registry_lookup(const std::string& backend, CrispasrRegistryEntry& out) {
    const Entry* e = find_by_backend(backend);
    if (!e)
        return false;
    fill(out, *e);
    return true;
}

bool crispasr_registry_lookup_by_filename(const std::string& filename, CrispasrRegistryEntry& out) {
    const Entry* e = find_by_filename(filename);
    if (!e)
        return false;
    fill(out, *e);
    return true;
}

bool crispasr_find_cached_model(CrispasrRegistryEntry& out, const std::string& cache_dir_override) {
    // k_registry is already ordered whisper > parakeet > canary > ... —
    // first entry wins, which matches the documented preference.
    const std::string dir = crispasr_cache::dir(cache_dir_override);
    for (const auto& e : k_registry) {
        const std::string path = dir + "/" + e.filename;
        if (crispasr_cache::file_present(path)) {
            fill(out, e);
            return true;
        }
    }
    return false;
}

std::string crispasr_resolve_model(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                   const std::string& cache_dir_override, bool allow_download) {
    // Concrete path that exists on disk — pass through.
    if (model_arg != "auto" && model_arg != "default") {
        FILE* f = fopen(model_arg.c_str(), "rb");
        if (f) {
            fclose(f);
            return model_arg;
        }

        // File not found — try registry-based download when permitted.
        const Entry* match = find_by_filename(model_arg);
        if (!match && !backend_name.empty())
            match = find_by_backend(backend_name);

        if (match && allow_download) {
            if (!quiet) {
                fprintf(stderr, "crispasr: model '%s' not found locally — downloading %s (%s)\n", model_arg.c_str(),
                        match->filename, match->approx_size);
            }
            return crispasr_cache::ensure_cached_file(match->filename, match->url, quiet, "crispasr",
                                                      cache_dir_override);
        }
        // Either no registry match or caller didn't authorise download —
        // return the arg untouched so the caller can decide (prompt,
        // error, etc.).
        return model_arg;
    }

    const Entry* e = find_by_backend(backend_name);
    if (!e) {
        fprintf(stderr, "crispasr: -m auto not supported for backend '%s' (no default model registered)\n",
                backend_name.c_str());
        return "";
    }

    if (!quiet)
        fprintf(stderr, "crispasr: resolving %s (%s) via -m auto\n", e->filename, e->approx_size);
    return crispasr_cache::ensure_cached_file(e->filename, e->url, quiet, "crispasr", cache_dir_override);
}
