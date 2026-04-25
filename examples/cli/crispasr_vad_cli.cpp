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

// Default Silero VAD model from the ggml-org/whisper-vad HF repo.
// ~885 KB. Auto-downloaded on first use to ~/.cache/crispasr so users
// can pass `--vad` without having to hunt down the GGUF.
namespace {
constexpr const char* kVadDefaultUrl =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
constexpr const char* kVadDefaultFile = "ggml-silero-v5.1.2.bin";
} // namespace

std::string crispasr_resolve_vad_model(const whisper_params& p) {
    const std::string& v = p.vad_model;
    const bool want_vad = p.vad || !v.empty();
    if (!want_vad)
        return "";
    if (!v.empty() && v != "auto" && v != "default")
        return v;
    return crispasr_cache::ensure_cached_file(kVadDefaultFile, kVadDefaultUrl, p.no_prints, "crispasr[vad]",
                                              p.cache_dir);
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

    return crispasr_fixed_chunk_slices(n_samples, sample_rate, chunk_seconds);
}
