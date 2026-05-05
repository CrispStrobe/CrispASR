// crispasr_vad_cli.cpp — CLI policy layer over the library VAD helpers.
//
// Auto-downloads the canonical Silero VAD GGUF into the CrispASR cache
// dir when the user passed `--vad` without `--vad-model`, then hands off
// to the shared algorithmic core in `src/crispasr_vad.cpp` via the
// exported `crispasr_compute_vad_slices` / `crispasr_fixed_chunk_slices`
// functions. Download / cache behaviour is CLI UX policy, not a library
// concern, so it lives here.

#include "crispasr_vad_cli.h"
#include "crispasr_cache.h"
#include "whisper_params.h"

#include <string>

// Default VAD models. Auto-downloaded on first use to ~/.cache/crispasr.
// `--vad` alone → Silero. `--vad -vm firered` → FireRedVAD. `--vad -vm whisper-vad` → Whisper-VAD-EncDec.
namespace {
constexpr const char* kVadSileroUrl = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
constexpr const char* kVadSileroFile = "ggml-silero-v5.1.2.bin";
constexpr const char* kVadFireredUrl = "https://huggingface.co/cstr/firered-vad-GGUF/resolve/main/firered-vad.gguf";
constexpr const char* kVadFireredFile = "firered-vad.gguf";
constexpr const char* kVadWhisperUrl =
    "https://huggingface.co/cstr/whisper-vad-encdec-asmr-GGUF/resolve/main/whisper-vad-asmr-q4_k.gguf";
constexpr const char* kVadWhisperFile = "whisper-vad-asmr-q4_k.gguf";
constexpr const char* kVadMarblenetUrl =
    "https://huggingface.co/cstr/marblenet-vad-GGUF/resolve/main/marblenet-vad.gguf";
constexpr const char* kVadMarblenetFile = "marblenet-vad.gguf";
} // namespace

// Check if a path refers to a FireRedVAD model (by filename pattern).
static bool is_firered_vad_path(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    std::string basename = (pos != std::string::npos) ? path.substr(pos + 1) : path;
    // Case-insensitive check for "firered" + "vad"
    std::string lo;
    lo.reserve(basename.size());
    for (char c : basename)
        lo += (char)std::tolower((unsigned char)c);
    return lo.find("firered") != std::string::npos && lo.find("vad") != std::string::npos;
}

std::string crispasr_resolve_vad_model(const whisper_params& p) {
    const std::string& v = p.vad_model;
    const bool want_vad = p.vad || !v.empty();
    if (!want_vad)
        return "";
    // Explicit path (not a keyword) — use as-is
    if (!v.empty() && v != "auto" && v != "default" && v != "silero" && v != "firered" && v != "whisper-vad" &&
        v != "marblenet")
        return v;
    // `--vad -vm firered` → auto-download FireRedVAD (2.4 MB, F1=97.57%)
    if (v == "firered")
        return crispasr_cache::ensure_cached_file(kVadFireredFile, kVadFireredUrl, p.no_prints, "crispasr[vad]",
                                                  p.cache_dir);
    // `--vad -vm marblenet` → auto-download MarbleNet VAD (439 KB, multilingual)
    if (v == "marblenet")
        return crispasr_cache::ensure_cached_file(kVadMarblenetFile, kVadMarblenetUrl, p.no_prints, "crispasr[vad]",
                                                  p.cache_dir);
    // `--vad -vm whisper-vad` → auto-download Whisper-VAD-EncDec (22 MB Q4_K, experimental)
    if (v == "whisper-vad")
        return crispasr_cache::ensure_cached_file(kVadWhisperFile, kVadWhisperUrl, p.no_prints, "crispasr[vad]",
                                                  p.cache_dir);
    // Default / auto / silero → Silero VAD (885 KB)
    return crispasr_cache::ensure_cached_file(kVadSileroFile, kVadSileroUrl, p.no_prints, "crispasr[vad]", p.cache_dir);
}

bool crispasr_vad_is_firered(const whisper_params& p) {
    std::string path = crispasr_resolve_vad_model(p);
    return !path.empty() && is_firered_vad_path(path);
}

std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(const float* samples, int n_samples, int sample_rate,
                                                                int chunk_seconds, const whisper_params& params) {
    const std::string vad_path = crispasr_resolve_vad_model(params);

    if (!vad_path.empty()) {
        crispasr_vad_options opts;
        opts.threshold = params.vad_threshold;
        opts.min_speech_duration_ms = params.vad_min_speech_duration_ms;
        opts.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        opts.speech_pad_ms = params.vad_speech_pad_ms;
        opts.chunk_seconds = chunk_seconds;
        opts.n_threads = params.n_threads;
        auto slices = crispasr_compute_vad_slices(samples, n_samples, sample_rate, vad_path.c_str(), opts);
        if (!slices.empty())
            return slices;
        // VAD model load failed or detected no speech — fall through
        // to fixed chunking so the CLI still produces output.
    }

    // VAD-free fallback: cut at lowest-RMS 100 ms within the last 5 s of
    // each `chunk_seconds` window so chunk boundaries don't slice
    // mid-word (PLAN #80b).
    return crispasr_energy_chunk_slices(samples, n_samples, sample_rate, chunk_seconds);
}
