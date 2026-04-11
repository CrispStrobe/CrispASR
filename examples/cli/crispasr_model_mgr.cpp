// crispasr_model_mgr.cpp — curl/wget-based model auto-download.

#include "crispasr_model_mgr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32)
#  include <direct.h>
#  define crispasr_mkdir(p) _mkdir(p)
#  define access _access
#  ifndef F_OK
#    define F_OK 0
#  endif
#else
#  include <unistd.h>
#  define crispasr_mkdir(p) mkdir((p), 0755)
#endif

namespace {

// Per-backend canonical model. Extend as new backends are wired up.
struct registry_entry {
    const char * backend;
    const char * filename;
    const char * url;       // direct download URL (HuggingFace resolve link)
    const char * approx_size;
};

constexpr registry_entry k_registry[] = {
    // Parakeet TDT 0.6B v3 quantised
    { "parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
      "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf",
      "~467 MB" },
    // Canary 1B v2 quantised
    { "canary", "canary-1b-v2-q4_k.gguf",
      "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf",
      "~600 MB" },
    // Voxtral Mini 3B 2507
    { "voxtral", "voxtral-mini-3b-2507-q4_k.gguf",
      "https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF/resolve/main/voxtral-mini-3b-2507-q4_k.gguf",
      "~2.5 GB" },
    // Voxtral Mini 4B Realtime
    { "voxtral4b", "voxtral-mini-4b-realtime-q4_k.gguf",
      "https://huggingface.co/cstr/voxtral-mini-4b-realtime-GGUF/resolve/main/voxtral-mini-4b-realtime-q4_k.gguf",
      "~3.3 GB" },
    // Granite 4.0 1B Speech
    { "granite", "granite-4.0-1b-speech-q4_k.gguf",
      "https://huggingface.co/cstr/granite-4.0-1b-speech-GGUF/resolve/main/granite-4.0-1b-speech-q4_k.gguf",
      "~900 MB" },
    // Qwen3-ASR 0.6B
    { "qwen3", "qwen3-asr-0.6b-q4_k.gguf",
      "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf",
      "~500 MB" },
    // Cohere Transcribe 03-2026
    { "cohere", "cohere-transcribe-q4_k.gguf",
      "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf",
      "~550 MB" },
};

const registry_entry * lookup(const std::string & backend) {
    for (const auto & e : k_registry) {
        if (backend == e.backend) return &e;
    }
    return nullptr;
}

bool file_exists(const std::string & p) {
    return access(p.c_str(), F_OK) == 0;
}

std::string cache_dir() {
    const char * home = std::getenv("HOME");
    if (!home || !*home) {
#if defined(_WIN32)
        const char * up = std::getenv("USERPROFILE");
        if (up && *up) home = up;
        else          home = "C:\\";
#else
        home = "/tmp";
#endif
    }
    std::string dir = home;
    dir += "/.cache";
    crispasr_mkdir(dir.c_str()); // ignore EEXIST
    dir += "/crispasr";
    crispasr_mkdir(dir.c_str());
    return dir;
}

// Shell-escape a value for use inside single-quoted shell arguments.
// Replaces any single quotes with '\'' the standard POSIX idiom.
std::string sh_single_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else           out += c;
    }
    out += "'";
    return out;
}

// Returns 0 on success. Tries curl first, then wget. No Python dependency.
int download(const std::string & url, const std::string & dest, bool quiet) {
    // curl: -fL = fail on HTTP error, follow redirects; -o = output file.
    std::string curl_cmd = "curl -fL ";
    if (quiet) curl_cmd += "-s ";
    else       curl_cmd += "--progress-bar ";
    curl_cmd += "-o " + sh_single_quote(dest) + " " + sh_single_quote(url);

    int rc = std::system(curl_cmd.c_str());
    if (rc == 0 && file_exists(dest)) return 0;

    // Fall back to wget.
    std::string wget_cmd = "wget ";
    if (quiet) wget_cmd += "-q ";
    wget_cmd += "-O " + sh_single_quote(dest) + " " + sh_single_quote(url);

    rc = std::system(wget_cmd.c_str());
    if (rc == 0 && file_exists(dest)) return 0;

    return -1;
}

} // namespace

std::string crispasr_resolve_model(const std::string & model_arg,
                                   const std::string & backend_name,
                                   bool quiet)
{
    // Pass-through for explicit paths.
    if (model_arg != "auto" && model_arg != "default") {
        return model_arg;
    }

    const registry_entry * e = lookup(backend_name);
    if (!e) {
        fprintf(stderr,
                "crispasr: error: -m auto is not supported for backend '%s' "
                "(no default model registered)\n",
                backend_name.c_str());
        return "";
    }

    const std::string dir    = cache_dir();
    const std::string target = dir + "/" + e->filename;

    if (file_exists(target)) {
        if (!quiet) {
            fprintf(stderr, "crispasr: using cached model: %s\n", target.c_str());
        }
        return target;
    }

    if (!quiet) {
        fprintf(stderr, "crispasr: downloading %s (%s)...\n",
                e->filename, e->approx_size);
        fprintf(stderr, "crispasr: source:      %s\n", e->url);
        fprintf(stderr, "crispasr: destination: %s\n", target.c_str());
    }

    if (download(e->url, target, quiet) != 0) {
        fprintf(stderr,
                "crispasr: error: download failed. Install curl or wget and try again, "
                "or download manually from:\n  %s\n  -> %s\n",
                e->url, target.c_str());
        return "";
    }

    return target;
}
