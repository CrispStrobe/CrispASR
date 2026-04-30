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
    const char* companion_file; // optional extra file (e.g. tokenizer.bin), NULL if none
    const char* companion_url;
};

// Keep entries aligned with what the CLI-only registry used to ship.
// Adding a new backend: one row here + a PUBLIC-link in src/CMakeLists.txt.
// clang-format off
constexpr Entry k_registry[] = {
    {"whisper", "ggml-base.bin",
     "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", "~147 MB", nullptr, nullptr},
    {"parakeet", "parakeet-tdt-0.6b-v3-q4_k.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf", "~467 MB", nullptr, nullptr},
    {"canary", "canary-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/canary-1b-v2-GGUF/resolve/main/canary-1b-v2-q4_k.gguf", "~600 MB", nullptr, nullptr},
    {"voxtral", "voxtral-mini-3b-2507-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-3b-2507-GGUF/resolve/main/voxtral-mini-3b-2507-q4_k.gguf", "~2.5 GB", nullptr, nullptr},
    {"voxtral4b", "voxtral-mini-4b-realtime-q4_k.gguf",
     "https://huggingface.co/cstr/voxtral-mini-4b-realtime-GGUF/resolve/main/voxtral-mini-4b-realtime-q4_k.gguf",
     "~3.3 GB", nullptr, nullptr},
    {"granite", "granite-speech-4.0-1b-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.0-1b-GGUF/resolve/main/granite-speech-4.0-1b-q4_k.gguf", "~2.94 GB", nullptr, nullptr},
    {"granite-4.1", "granite-speech-4.1-2b-q4_k.gguf",
     "https://huggingface.co/cstr/granite-speech-4.1-2b-GGUF/resolve/main/granite-speech-4.1-2b-q4_k.gguf", "~2.94 GB", nullptr, nullptr},
    {"granite-4.1-plus", "granite-speech-4.1-2b-plus-f16.gguf",
     "https://huggingface.co/cstr/granite-speech-4.1-2b-plus-GGUF/resolve/main/granite-speech-4.1-2b-plus-f16.gguf",
     "~5.6 GB", nullptr, nullptr},
    {"qwen3", "qwen3-asr-0.6b-q4_k.gguf",
     "https://huggingface.co/cstr/qwen3-asr-0.6b-GGUF/resolve/main/qwen3-asr-0.6b-q4_k.gguf", "~500 MB", nullptr, nullptr},
    {"cohere", "cohere-transcribe-q4_k.gguf",
     "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf", "~550 MB", nullptr, nullptr},
    {"wav2vec2", "wav2vec2-xlsr-en-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-english-GGUF/resolve/main/wav2vec2-xlsr-en-q4_k.gguf",
     "~212 MB", nullptr, nullptr},
    {"omniasr", "omniasr-ctc-1b-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniASR-CTC-1B-v2-GGUF/resolve/main/omniasr-ctc-1b-v2-q4_k.gguf", "~580 MB", nullptr, nullptr},
    {"omniasr-llm", "omniasr-llm-300m-v2-q4_k.gguf",
     "https://huggingface.co/cstr/omniasr-llm-300m-v2-GGUF/resolve/main/omniasr-llm-300m-v2-q4_k.gguf", "~580 MB", nullptr, nullptr},
    {"hubert", "hubert-large-ls960-ft-q4_k.gguf",
     "https://huggingface.co/cstr/hubert-large-ls960-ft-GGUF/resolve/main/hubert-large-ls960-ft-q4_k.gguf", "~200 MB", nullptr, nullptr},
    {"data2vec", "data2vec-audio-base-960h-q4_k.gguf",
     "https://huggingface.co/cstr/data2vec-audio-960h-GGUF/resolve/main/data2vec-audio-base-960h-q4_k.gguf", "~60 MB", nullptr, nullptr},
    {"vibevoice", "vibevoice-asr-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-asr-GGUF/resolve/main/vibevoice-asr-q4_k.gguf", "~4.5 GB", nullptr, nullptr},
    {"vibevoice-tts", "vibevoice-realtime-0.5b-q4_k.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-realtime-0.5b-q4_k.gguf",
     "~636 MB",
     "vibevoice-voice-emma.gguf",
     "https://huggingface.co/cstr/vibevoice-realtime-0.5b-GGUF/resolve/main/vibevoice-voice-emma.gguf"},
    {"firered-asr", "firered-asr2-aed-q4_k.gguf",
     "https://huggingface.co/cstr/firered-asr2-aed-GGUF/resolve/main/firered-asr2-aed-q4_k.gguf", "~918 MB", nullptr, nullptr},
    {"kyutai-stt", "kyutai-stt-1b-q4_k.gguf",
     "https://huggingface.co/cstr/kyutai-stt-1b-GGUF/resolve/main/kyutai-stt-1b-q4_k.gguf", "~636 MB", nullptr, nullptr},
    {"glm-asr", "glm-asr-nano-q4_k.gguf",
     "https://huggingface.co/cstr/glm-asr-nano-GGUF/resolve/main/glm-asr-nano-q4_k.gguf", "~1.2 GB", nullptr, nullptr},
    {"moonshine", "moonshine-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/moonshine-tiny-q4_k.gguf", "~20 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-tiny-GGUF/resolve/main/tokenizer.bin"},
    {"wav2vec2-de", "wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "https://huggingface.co/cstr/wav2vec2-large-xlsr-53-german-GGUF/resolve/main/wav2vec2-large-xlsr-53-german-q4_k.gguf",
     "~222 MB", nullptr, nullptr},
    {"moonshine-streaming", "moonshine-streaming-tiny-q4_k.gguf",
     "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/moonshine-streaming-tiny-q4_k.gguf", "~31 MB",
     "tokenizer.bin", "https://huggingface.co/cstr/moonshine-streaming-tiny-GGUF/resolve/main/tokenizer.bin"},
    {"fastconformer-ctc", "stt-en-fastconformer-ctc-large-q4_k.gguf",
     "https://huggingface.co/cstr/stt-en-fastconformer-ctc-large-GGUF/resolve/main/stt-en-fastconformer-ctc-large-q4_k.gguf",
     "~83 MB", nullptr, nullptr},
    {"gemma4-e2b", "gemma4-e2b-it-q4_k.gguf",
     "https://huggingface.co/cstr/gemma4-e2b-it-GGUF/resolve/main/gemma4-e2b-it-q4_k.gguf",
     "~2.5 GB", nullptr, nullptr},
    // parakeet-ja: F16 is the auto-download default — Q4_K of this
    // model is quantisation-sensitive (joint.pred / decoder.embed
    // dimensions fall back to q4_0 inside q4_k mode) and the talker
    // enters a fixed-point loop after ~8 tokens. The Q4_K file is
    // available at the same repo for users who pin disk space, but
    // we'd rather have correct output by default.
    {"parakeet-ja", "parakeet-tdt-0.6b-ja.gguf",
     "https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF/resolve/main/parakeet-tdt-0.6b-ja.gguf",
     "~1.24 GB", nullptr, nullptr},
    // Qwen3-TTS: the talker LM and the codec live in two separate HF
    // repos. Default download is Q8_0 talker (the LEARNINGS-recommended
    // deployment quant — Q4_K drifts noticeably in strict diffs) paired
    // with F16 codec (quantising the codec hurts earlier than the talker
    // — the runtime_ref_codes path is sensitive). Q4_K talker is on the
    // same repo for users who pin disk space; pass `-m <path>` to use it.
    {"qwen3-tts", "qwen3-tts-12hz-0.6b-base-q8_0.gguf",
     "https://huggingface.co/cstr/qwen3-tts-0.6b-base-GGUF/resolve/main/qwen3-tts-12hz-0.6b-base-q8_0.gguf",
     "~986 MB",
     "qwen3-tts-tokenizer-12hz.gguf",
     "https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF/resolve/main/qwen3-tts-tokenizer-12hz.gguf"},
};
// clang-format on

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
            std::string dl =
                crispasr_cache::ensure_cached_file(match->filename, match->url, quiet, "crispasr", cache_dir_override);
            if (!dl.empty() && match->companion_file && match->companion_url)
                crispasr_cache::ensure_cached_file(match->companion_file, match->companion_url, quiet, "crispasr",
                                                   cache_dir_override);
            return dl;
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
    std::string result = crispasr_cache::ensure_cached_file(e->filename, e->url, quiet, "crispasr", cache_dir_override);

    // Download companion file (e.g. tokenizer.bin for moonshine) if needed
    if (!result.empty() && e->companion_file && e->companion_url) {
        crispasr_cache::ensure_cached_file(e->companion_file, e->companion_url, quiet, "crispasr", cache_dir_override);
    }

    return result;
}
