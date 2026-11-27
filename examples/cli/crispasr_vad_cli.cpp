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
constexpr const char* kVadSileroUrl = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.3.0.bin";
constexpr const char* kVadSileroFile = "ggml-silero-v6.3.0.bin";
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
        v != "marblenet" && v != "webrtc")
        return v;
    // `--vad -vm webrtc` → WebRTC GMM VAD (no model file, pure algorithmic, BSD-3)
    if (v == "webrtc")
        return "webrtc";
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

bool crispasr_vad_is_webrtc(const whisper_params& p) {
    return p.vad_model == "webrtc";
}

std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(const float* samples, int n_samples, int sample_rate,
                                                                int chunk_seconds, const whisper_params& params,
                                                                bool* out_vad_load_failed) {
    if (out_vad_load_failed)
        *out_vad_load_failed = false;
    const std::string vad_path = crispasr_resolve_vad_model(params);

    // #311 follow-up: crispasr_resolve_vad_model() returns "" for TWO different
    // situations — "no VAD was requested" and "one was requested and the
    // download/resolve failed". Below, an empty path skips the VAD block
    // entirely, so the second case used to leave `out_vad_load_failed` false
    // and looked identical to the first. The strict guard in crispasr_run.cpp
    // keys off exactly that flag, so `--strict-pipeline` / `--require-vad`
    // could not fire on a failed download: the run exited 0 having quietly not
    // run VAD at all. Observed with a dangling cache dir — "download failed"
    // on stderr, rc=0, and a full-file chunk export as if VAD had run.
    //
    // Distinguish them here: the user asked for a VAD iff --vad was passed or
    // --vad-model names one.
    const bool vad_requested = params.vad || !params.vad_model.empty();
    if (vad_path.empty() && vad_requested) {
        if (out_vad_load_failed)
            *out_vad_load_failed = true;
    }

    if (!vad_path.empty()) {
        crispasr_vad_options opts;
        opts.threshold = params.vad_threshold;
        opts.threshold_explicit = params.vad_threshold_explicit;
        opts.min_speech_duration_ms = params.vad_min_speech_duration_ms;
        opts.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        opts.speech_pad_ms = params.vad_speech_pad_ms;
        opts.chunk_seconds = chunk_seconds;
        opts.n_threads = params.n_threads;
        bool load_failed = false;
        auto slices =
            crispasr_compute_vad_slices(samples, n_samples, sample_rate, vad_path.c_str(), opts, &load_failed);
        if (out_vad_load_failed)
            *out_vad_load_failed = load_failed;
        if (!slices.empty())
            return slices;
        // Issue #213: when the VAD model loaded successfully but detected
        // no speech, return empty (= no transcription). Previously we fell
        // through to energy-based chunking, which fed silent chunks to the
        // ASR model and caused hallucinated text.
        //
        // Only fall through to the energy-chunk fallback when the model
        // itself failed to load (the library logs a warning to stderr in
        // that case). We detect this by checking if the model file exists
        // and is readable — if it does, the VAD ran and "no speech" is the
        // correct answer.
        // WebRTC VAD has no model file — the sentinel "webrtc" means it ran.
        if (vad_path == "webrtc")
            return slices;
        FILE* f = fopen(vad_path.c_str(), "rb");
        if (f) {
            fclose(f);
            // Model file exists → VAD ran, no speech detected.
            return slices;
        }
        // Model file missing/unreadable → fall through to energy chunking.
        fprintf(stderr, "crispasr: VAD model '%s' not accessible, falling back to fixed chunking\n", vad_path.c_str());
    }

    // VAD-free fallback: cut at lowest-RMS 100 ms within the last 5 s of
    // each `chunk_seconds` window so chunk boundaries don't slice
    // mid-word (PLAN #80b).
    return crispasr_energy_chunk_slices(samples, n_samples, sample_rate, chunk_seconds);
}
