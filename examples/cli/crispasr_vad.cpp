// crispasr_vad.cpp — implementation of Silero-VAD-backed audio slicing.
// See crispasr_vad.h for the interface contract.
//
// Extracted from examples/parakeet-main/main.cpp.

#include "crispasr_vad.h"
#include "whisper_params.h"

#include "whisper.h" // whisper_vad_* API

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// Default Silero VAD model from the ggml-org/whisper-vad HF repo.
// ~885 KB. Auto-downloaded on first use to ~/.cache/crispasr so users
// can pass `--vad` without having to hunt down the GGUF.
constexpr const char * kVadDefaultUrl =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
constexpr const char * kVadDefaultFile = "ggml-silero-v5.1.2.bin";

bool vad_file_exists(const std::string & p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return false;
    // Treat 0-byte zombies (from a failed earlier download) as missing
    // so the next attempt retries the fetch instead of handing a
    // corrupted path to whisper_vad_init_from_file_with_params.
    return st.st_size > 0;
}

std::string vad_cache_dir() {
    const char * home = std::getenv("HOME");
    std::string dir = (home && *home) ? home : "/tmp";
    dir += "/.cache/crispasr";
    mkdir(dir.c_str(), 0755);
    return dir;
}

bool vad_fetch_url(const std::string & url, const std::string & dst,
                   bool no_prints) {
    if (!no_prints) {
        fprintf(stderr, "crispasr[vad]: downloading %s\n", url.c_str());
    }
    const std::string cmd_curl =
        "curl -fL --progress-bar -o " + dst + " " + url;
    if (std::system(cmd_curl.c_str()) == 0 && vad_file_exists(dst)) return true;
    const std::string cmd_wget =
        "wget -q --show-progress -O " + dst + " " + url;
    if (std::system(cmd_wget.c_str()) == 0 && vad_file_exists(dst)) return true;
    return false;
}

// Resolve params.vad_model into a real file path. When empty (user passed
// --vad without --vad-model), or set to "auto"/"default", download the
// canonical Silero VAD GGUF into the crispasr cache dir on first use.
// Returns an empty string if no VAD was requested at all.
std::string resolve_vad_model(const whisper_params & p) {
    const std::string & v = p.vad_model;
    const bool want_vad   = p.vad || !v.empty();
    if (!want_vad) return "";

    // Explicit path (not auto/default) — use as-is.
    if (!v.empty() && v != "auto" && v != "default") return v;

    // Auto path: check the cache, download on miss.
    const std::string dst = vad_cache_dir() + "/" + kVadDefaultFile;
    if (vad_file_exists(dst)) return dst;
    if (!vad_fetch_url(kVadDefaultUrl, dst, p.no_prints)) return "";
    return dst;
}

} // namespace

std::vector<crispasr_audio_slice> crispasr_compute_audio_slices(
    const float * samples,
    int           n_samples,
    int           sample_rate,
    int           chunk_seconds,
    const whisper_params & params)
{
    std::vector<crispasr_audio_slice> slices;

    whisper_vad_context * vctx = nullptr;
    const std::string vad_path = resolve_vad_model(params);
    const bool want_vad = !vad_path.empty();
    if (want_vad) {
        whisper_vad_context_params vcp = whisper_vad_default_context_params();
        vcp.n_threads = params.n_threads;
        vctx = whisper_vad_init_from_file_with_params(vad_path.c_str(), vcp);
        if (!vctx) {
            fprintf(stderr,
                    "crispasr: warning: failed to load VAD model '%s', "
                    "falling back to fixed chunking\n",
                    vad_path.c_str());
        }
    }

    if (vctx) {
        whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold               = params.vad_threshold;
        vp.min_speech_duration_ms  = params.vad_min_speech_duration_ms;
        vp.min_silence_duration_ms = params.vad_min_silence_duration_ms;
        vp.speech_pad_ms           = (float) params.vad_speech_pad_ms;

        whisper_vad_segments * vseg =
            whisper_vad_segments_from_samples(vctx, vp, samples, n_samples);

        const int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
        for (int i = 0; i < nv; i++) {
            const float t0s = whisper_vad_segments_get_segment_t0(vseg, i);
            const float t1s = whisper_vad_segments_get_segment_t1(vseg, i);
            const int s = std::max(0, (int)(t0s * sample_rate));
            const int e = std::min(n_samples, (int)(t1s * sample_rate));
            if (e > s) {
                slices.push_back({
                    s, e,
                    (int64_t)(t0s * 100.0f),
                    (int64_t)(t1s * 100.0f),
                });
            }
        }
        if (vseg) whisper_vad_free_segments(vseg);
        whisper_vad_free(vctx);
        return slices;
    }

    // No VAD: fall back to fixed chunking. Encoders scale O(T^2) in frame
    // count, so unbounded audio hits memory/latency walls. The default
    // chunk_seconds (30 s) is a conservative window most models handle well.
    const int chunk_samples = chunk_seconds > 0 ? chunk_seconds * sample_rate
                                                : n_samples;

    if (n_samples <= chunk_samples) {
        const int64_t dur_cs = (int64_t)((double)n_samples / sample_rate * 100.0);
        slices.push_back({0, n_samples, 0, dur_cs});
    } else {
        for (int s = 0; s < n_samples; s += chunk_samples) {
            const int e = std::min(n_samples, s + chunk_samples);
            slices.push_back({
                s, e,
                (int64_t)((double)s / sample_rate * 100.0),
                (int64_t)((double)e / sample_rate * 100.0),
            });
        }
    }
    return slices;
}
