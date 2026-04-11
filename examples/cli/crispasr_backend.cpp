// crispasr_backend.cpp — backend factory, auto-detection, and helpers.

#include "crispasr_backend.h"
#include "whisper_params.h"

// Forward declarations of per-backend constructors. Each is implemented in
// its own crispasr_backend_X.cpp file and compiled only if the backend's
// library is linked in.
std::unique_ptr<CrispasrBackend> crispasr_make_parakeet_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_canary_backend();
std::unique_ptr<CrispasrBackend> crispasr_make_cohere_backend();
// Remaining constructors will land in subsequent steps:
// std::unique_ptr<CrispasrBackend> crispasr_make_qwen3_backend();
// std::unique_ptr<CrispasrBackend> crispasr_make_voxtral_backend();
// std::unique_ptr<CrispasrBackend> crispasr_make_voxtral4b_backend();
// std::unique_ptr<CrispasrBackend> crispasr_make_granite_backend();
// std::unique_ptr<CrispasrBackend> crispasr_make_whisper_backend();

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<CrispasrBackend> crispasr_create_backend(const std::string & name) {
    if (name == "parakeet") return crispasr_make_parakeet_backend();
    if (name == "canary")   return crispasr_make_canary_backend();
    if (name == "cohere")   return crispasr_make_cohere_backend();

    // Additional backends plug in here as they're added.
    // if (name == "cohere")    return crispasr_make_cohere_backend();
    // if (name == "qwen3")     return crispasr_make_qwen3_backend();
    // if (name == "voxtral")   return crispasr_make_voxtral_backend();
    // if (name == "voxtral4b") return crispasr_make_voxtral4b_backend();
    // if (name == "granite")   return crispasr_make_granite_backend();

    fprintf(stderr, "crispasr: error: unknown backend '%s'\n", name.c_str());
    return nullptr;
}

std::vector<std::string> crispasr_list_backends() {
    return {
        "whisper",   // via the unmodified cli.cpp path
        "parakeet",
        "canary",
        "cohere",
        // future: "qwen3", "voxtral", "voxtral4b", "granite"
    };
}

// ---------------------------------------------------------------------------
// GGUF auto-detection
// ---------------------------------------------------------------------------

// Read the "general.architecture" key from a GGUF file and map it to a
// backend name. Uses gguf_init_from_file() — which lives in ggml — so this
// is cheap: only the metadata is parsed, not the weight tensors.
//
// Mappings are based on the value that each model's converter writes into
// the GGUF file. When a converter doesn't write this key we fall back to
// filename heuristics.
std::string crispasr_detect_backend_from_gguf(const std::string & model_path) {
    if (model_path.empty()) return "";

    struct gguf_init_params gip = { /*.no_alloc=*/true, /*.ctx=*/nullptr };
    gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gip);

    if (gctx) {
        const int key = gguf_find_key(gctx, "general.architecture");
        if (key >= 0) {
            const char * arch = gguf_get_val_str(gctx, key);
            if (arch) {
                const std::string a = arch;
                gguf_free(gctx);

                if (a == "whisper")        return "whisper";
                if (a == "parakeet")       return "parakeet";
                if (a == "parakeet-tdt")   return "parakeet";
                if (a == "canary")         return "canary";
                if (a == "canary-ctc")     return "canary";
                if (a == "cohere")         return "cohere";
                if (a == "cohere-transcribe") return "cohere";
                if (a == "qwen3-asr")      return "qwen3";
                if (a == "qwen3_asr")      return "qwen3";
                if (a == "voxtral")        return "voxtral";
                if (a == "voxtral4b")      return "voxtral4b";
                if (a == "voxtral-4b")     return "voxtral4b";
                if (a == "granite-speech") return "granite";
                if (a == "granite_speech") return "granite";
                // Unknown arch; fall through to filename heuristics.
            }
        }
        gguf_free(gctx);
    }

    // Filename heuristics as a fallback. Case-insensitive substring match.
    auto contains_ci = [&](const char * needle) {
        std::string lo;
        lo.reserve(model_path.size());
        for (char c : model_path) lo += (char)std::tolower((unsigned char)c);
        return lo.find(needle) != std::string::npos;
    };

    if (contains_ci("voxtral") && contains_ci("4b"))    return "voxtral4b";
    if (contains_ci("voxtral"))                          return "voxtral";
    if (contains_ci("parakeet"))                         return "parakeet";
    if (contains_ci("canary"))                           return "canary";
    if (contains_ci("cohere"))                           return "cohere";
    if (contains_ci("qwen3") && contains_ci("asr"))      return "qwen3";
    if (contains_ci("granite") && contains_ci("speech")) return "granite";
    if (contains_ci("ggml-") && contains_ci(".bin"))     return "whisper";

    return "";
}
