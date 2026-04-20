// crispasr_lid.h — shared language identification helpers.
//
// Two in-process backends:
//
//   * `CrispasrLidMethod::Whisper` — uses the whisper.cpp encoder +
//     language head on a multilingual ggml-*.bin model. Typically
//     `ggml-tiny.bin` (75 MB, fast, covers the languages whisper was
//     trained on). Process-wide caching keeps the context alive across
//     calls so batch jobs don't reload 75 MB per slice.
//
//   * `CrispasrLidMethod::Silero` — uses the GGUF-packed Silero 95-
//     language classifier through `src/silero_lid.*`. Smaller model
//     (~10 MB) and wider language coverage than whisper-tiny.
//
// Both paths take a concrete model file path; auto-download / cache
// resolution is the caller's responsibility (the CLI has a shim that
// handles it, wrappers can ship the GGUF as an asset).
//
// The sherpa-onnx subprocess backend stays in the CLI, same reason as
// the sherpa diarizer: it shells out to an external binary.
//
// Shared by the CLI, the C-ABI wrapper `crispasr_detect_language_pcm`
// in crispasr_c_api.cpp, and every language binding that calls through
// that wrapper.

#pragma once

#include <string>

enum class CrispasrLidMethod {
    Whisper = 0,
    Silero = 1,
    Firered = 2,
};

struct CrispasrLidResult {
    std::string lang_code;    // ISO 639-1 ("en", "de", "ja", ...) — empty on failure
    float confidence = -1.0f; // [0, 1], -1 if unknown
    std::string source;       // "whisper" | "silero"
};

struct CrispasrLidOptions {
    CrispasrLidMethod method = CrispasrLidMethod::Whisper;
    std::string model_path; // concrete file path; required
    int n_threads = 4;
    bool use_gpu = false;
    int gpu_device = 0;
    bool flash_attn = true;
    bool verbose = false; // print detection result to stderr
};

/// Run LID on a 16 kHz mono f32 PCM buffer using the method in `opts`.
/// Returns true on success; `out.lang_code` carries the ISO code.
/// On failure the reason is printed to stderr when `opts.verbose` is
/// true, and the function returns false.
bool crispasr_detect_language(const float* samples, int n_samples, const CrispasrLidOptions& opts,
                              CrispasrLidResult& out);
