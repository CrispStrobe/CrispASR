// crispasr_vad.cpp — Silero VAD segmentation + stitching for shared use.
// See crispasr_vad.h for the interface contract.
//
// Extracted from examples/cli/crispasr_vad.cpp so that the CLI, the C-ABI
// wrapper `crispasr_session_transcribe_vad`, and every language binding
// use the same implementation. Auto-download / cache resolution stays in
// the CLI (it's a UX policy, not a library responsibility) — this file
// operates on a concrete VAD model path supplied by the caller.

#include "crispasr_vad.h"

#include "core/audio_chunking.h"
#include "core/vad_failover.h" // §W4: detect an obviously-broken VAD result

#include "firered_vad.h" // FireRedVAD (DFSMN) — alternative to Silero
#include "crispasr.h"    // whisper_vad_* API (Silero VAD)
#if __has_include("marblenet_vad.h")
#include "marblenet_vad.h" // NVIDIA MarbleNet VAD (1D separable CNN)
#define CA_HAVE_MARBLENET_VAD 1
#endif
#if __has_include("crispasr_vad_encdec.h")
#include "crispasr_vad_encdec.h" // Whisper-encoder + decoder VAD (ONNX-converted)
#define CA_HAVE_WVAD_ENCDEC 1
#endif
#if __has_include("webrtc_vad.h")
#include "webrtc_vad.h" // WebRTC GMM-based VAD (no model file)
#define CA_HAVE_WEBRTC_VAD 1
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <cmath>
#include <string>

// ---- FireRed VAD context cache (§176e) ----
// Same pattern as MarbleNet/Silero: avoid init/free per call.
// firered_vad_context holds only immutable model weights + backend —
// no per-call mutable state — so the mutex is only needed to prevent
// concurrent GPU backend access, not for state isolation.
static std::mutex g_firered_cache_mtx;
static firered_vad_context* g_firered_cache_ctx = nullptr;
static std::string g_firered_cache_path;

static firered_vad_context* firered_vad_get_cached_locked(const char* path) {
    if (g_firered_cache_ctx && g_firered_cache_path == path)
        return g_firered_cache_ctx;
    if (g_firered_cache_ctx) {
        firered_vad_free(g_firered_cache_ctx);
        g_firered_cache_ctx = nullptr;
        g_firered_cache_path.clear();
    }
    g_firered_cache_ctx = firered_vad_init(path);
    if (g_firered_cache_ctx)
        g_firered_cache_path = path;
    return g_firered_cache_ctx;
}

// ---- MarbleNet VAD context cache (§176e) ----
// Same pattern as Silero: avoid init/free per call.
#ifdef CA_HAVE_MARBLENET_VAD
static std::mutex g_marblenet_cache_mtx;
static marblenet_vad_context* g_marblenet_cache_ctx = nullptr;
static std::string g_marblenet_cache_path;

static marblenet_vad_context* marblenet_vad_get_cached_locked(const char* path) {
    if (g_marblenet_cache_ctx && g_marblenet_cache_path == path)
        return g_marblenet_cache_ctx;
    if (g_marblenet_cache_ctx) {
        marblenet_vad_free(g_marblenet_cache_ctx);
        g_marblenet_cache_ctx = nullptr;
        g_marblenet_cache_path.clear();
    }
    g_marblenet_cache_ctx = marblenet_vad_init(path);
    if (g_marblenet_cache_ctx)
        g_marblenet_cache_path = path;
    return g_marblenet_cache_ctx;
}
#endif

// ---- WhisperEncDec VAD context cache (§176e) ----
#ifdef CA_HAVE_WVAD_ENCDEC
static std::mutex g_encdec_cache_mtx;
static whisper_vad_encdec_context* g_encdec_cache_ctx = nullptr;
static std::string g_encdec_cache_path;

static whisper_vad_encdec_context* encdec_vad_get_cached_locked(const char* path) {
    if (g_encdec_cache_ctx && g_encdec_cache_path == path)
        return g_encdec_cache_ctx;
    if (g_encdec_cache_ctx) {
        whisper_vad_encdec_free(g_encdec_cache_ctx);
        g_encdec_cache_ctx = nullptr;
        g_encdec_cache_path.clear();
    }
    g_encdec_cache_ctx = whisper_vad_encdec_init(path);
    if (g_encdec_cache_ctx)
        g_encdec_cache_path = path;
    return g_encdec_cache_ctx;
}
#endif

// ---- Silero VAD context cache (fixes #132) ----
// Creating and destroying the ggml scheduler + backend on every request
// fragments memory; after ~4 cycles the allocator hits a pathological
// path and VAD time jumps from ~2 s to ~140 s. Caching the context
// across calls avoids the repeated init/free cycle entirely.
static std::mutex g_silero_cache_mtx;
static whisper_vad_context* g_silero_cache_ctx = nullptr;
static std::string g_silero_cache_path;

// Return the cached Silero context (creating it on first use or when
// the model path changed). Caller must NOT free the returned pointer.
//
// The caller MUST already hold g_silero_cache_mtx and keep holding it for
// the whole duration it uses the returned context: the cached context owns
// mutable per-request state (LSTM h/c buffer, scheduler, probs) that is
// reset and rewritten on every detect, so two callers must not share it
// concurrently. The server runs VAD slicing outside its model_mutex
// (crispasr_server.cpp), so this mutex is the only thing serializing
// concurrent requests against the single cached context (#132).
static whisper_vad_context* silero_vad_get_cached_locked(const char* vad_model_path, int n_threads) {
    if (g_silero_cache_ctx && g_silero_cache_path == vad_model_path) {
        return g_silero_cache_ctx;
    }
    // Path changed or first call — (re)create.
    if (g_silero_cache_ctx) {
        whisper_vad_free(g_silero_cache_ctx);
        g_silero_cache_ctx = nullptr;
        g_silero_cache_path.clear();
    }
    whisper_vad_context_params vcp = whisper_vad_default_context_params();
    vcp.n_threads = n_threads;
    g_silero_cache_ctx = whisper_vad_init_from_file_with_params(vad_model_path, vcp);
    if (g_silero_cache_ctx) {
        g_silero_cache_path = vad_model_path;
    }
    return g_silero_cache_ctx;
}

void crispasr_vad_free_cache() {
    {
        std::lock_guard<std::mutex> lock(g_silero_cache_mtx);
        if (g_silero_cache_ctx) {
            whisper_vad_free(g_silero_cache_ctx);
            g_silero_cache_ctx = nullptr;
            g_silero_cache_path.clear();
        }
    }
#ifdef CA_HAVE_MARBLENET_VAD
    {
        std::lock_guard<std::mutex> lock(g_marblenet_cache_mtx);
        if (g_marblenet_cache_ctx) {
            marblenet_vad_free(g_marblenet_cache_ctx);
            g_marblenet_cache_ctx = nullptr;
            g_marblenet_cache_path.clear();
        }
    }
#endif
#ifdef CA_HAVE_WVAD_ENCDEC
    {
        std::lock_guard<std::mutex> lock(g_encdec_cache_mtx);
        if (g_encdec_cache_ctx) {
            whisper_vad_encdec_free(g_encdec_cache_ctx);
            g_encdec_cache_ctx = nullptr;
            g_encdec_cache_path.clear();
        }
    }
#endif
}

// Check if a model path is a FireRedVAD model (by filename pattern)
static bool is_firered_vad_model(const char* path) {
    std::string p(path);
    // basename
    auto pos = p.find_last_of("/\\");
    std::string basename = (pos != std::string::npos) ? p.substr(pos + 1) : p;
    return basename.find("firered") != std::string::npos && basename.find("vad") != std::string::npos;
}

// FireRedVAD path: uses the DFSMN-based VAD model
static std::vector<crispasr_audio_slice> compute_firered_vad_slices(const float* samples, int n_samples,
                                                                    int sample_rate, const char* vad_model_path,
                                                                    const crispasr_vad_options& opts) {
    std::vector<crispasr_audio_slice> slices;

    std::unique_lock<std::mutex> lock(g_firered_cache_mtx);
    firered_vad_context* vctx = firered_vad_get_cached_locked(vad_model_path);
    if (!vctx) {
        fprintf(stderr, "crispasr: warning: failed to load FireRedVAD model '%s'\n", vad_model_path);
        return slices;
    }

    firered_vad_segment* segs = nullptr;
    int n_segs = 0;
    float min_speech_sec = opts.min_speech_duration_ms / 1000.0f;
    float min_silence_sec = opts.min_silence_duration_ms / 1000.0f;
    int rc =
        firered_vad_detect(vctx, samples, n_samples, &segs, &n_segs, opts.threshold, min_speech_sec, min_silence_sec);
    if (rc == 0 && segs && n_segs > 0) {
        for (int i = 0; i < n_segs; i++) {
            int64_t t0_cs = (int64_t)(segs[i].start_sec * 100.0f);
            int64_t t1_cs = (int64_t)(segs[i].end_sec * 100.0f);
            int s = std::max(0, (int)(segs[i].start_sec * sample_rate));
            int e = std::min(n_samples, (int)(segs[i].end_sec * sample_rate));
            if (e > s)
                slices.push_back({s, e, t0_cs, t1_cs});
        }
    }
    free(segs);
    return slices; // vctx stays cached; lock released here
}

std::vector<crispasr_audio_slice> crispasr_compute_vad_slices(const float* samples, int n_samples, int sample_rate,
                                                              const char* vad_model_path,
                                                              const crispasr_vad_options& opts, bool* out_load_failed) {
    std::vector<crispasr_audio_slice> slices;
    // Assume the model loads; a branch below sets this true if it can't.
    if (out_load_failed)
        *out_load_failed = false;
    if (!vad_model_path || !*vad_model_path || n_samples <= 0)
        return slices;

    // Dispatch by filename pattern
    std::string vpath(vad_model_path);
    if (is_firered_vad_model(vad_model_path)) {
        slices = compute_firered_vad_slices(samples, n_samples, sample_rate, vad_model_path, opts);
    }
#ifdef CA_HAVE_MARBLENET_VAD
    else if (vpath.find("marblenet") != std::string::npos && vpath.find(".gguf") != std::string::npos) {
        std::lock_guard<std::mutex> vad_lock(g_marblenet_cache_mtx);
        marblenet_vad_context* vctx = marblenet_vad_get_cached_locked(vad_model_path);
        if (vctx) {
            marblenet_vad_segment* segs = nullptr;
            int n_segs = 0;
            float min_speech_sec = opts.min_speech_duration_ms / 1000.0f;
            float min_silence_sec = opts.min_silence_duration_ms / 1000.0f;
            marblenet_vad_detect(vctx, samples, n_samples, &segs, &n_segs, opts.threshold, min_speech_sec,
                                 min_silence_sec);
            for (int i = 0; i < n_segs; i++) {
                int64_t t0_cs = (int64_t)(segs[i].start_sec * 100.0f);
                int64_t t1_cs = (int64_t)(segs[i].end_sec * 100.0f);
                int s = std::max(0, (int)(segs[i].start_sec * sample_rate));
                int e = std::min(n_samples, (int)(segs[i].end_sec * sample_rate));
                if (e > s)
                    slices.push_back({s, e, t0_cs, t1_cs});
            }
            if (segs)
                free(segs);
            // Do NOT free vctx — it's cached.
        } else if (out_load_failed) {
            *out_load_failed = true; // marblenet model failed to load
        }
    }
#endif
#ifdef CA_HAVE_WVAD_ENCDEC
    else if (std::string(vad_model_path).find("whisper") != std::string::npos &&
             std::string(vad_model_path).find("vad") != std::string::npos &&
             std::string(vad_model_path).find(".gguf") != std::string::npos) {
        std::lock_guard<std::mutex> vad_lock(g_encdec_cache_mtx);
        whisper_vad_encdec_context* vctx = encdec_vad_get_cached_locked(vad_model_path);
        if (vctx) {
            whisper_vad_encdec_segment* segs = nullptr;
            int n_segs = 0;
            float min_speech_sec = opts.min_speech_duration_ms / 1000.0f;
            float min_silence_sec = opts.min_silence_duration_ms / 1000.0f;
            // Issue #83 follow-up: this VAD's frame classifier is calibrated
            // lower than Silero / FireRed — observed mean_prob ≈ 0.27 on
            // continuous Japanese speech (`2-min-Okayu.wav`) where firered
            // posts ≈ 0.49. With the global default threshold (0.5) most of
            // the audio falls below positive-thresh and the segment merger
            // collapses long stretches into nothing. Auto-lower to 0.30 when
            // the user didn't pass `-vt` explicitly; pass-through otherwise.
            float effective_threshold = opts.threshold;
            if (!opts.threshold_explicit && opts.threshold == 0.5f) {
                effective_threshold = 0.30f;
                fprintf(stderr,
                        "whisper_vad_encdec: using threshold=%.2f (default 0.5 is too "
                        "aggressive for this model — pass -vt to override)\n",
                        effective_threshold);
            }
            whisper_vad_encdec_detect(vctx, samples, n_samples, &segs, &n_segs, effective_threshold, min_speech_sec,
                                      min_silence_sec, nullptr, nullptr, nullptr, nullptr);
            for (int i = 0; i < n_segs; i++) {
                int64_t t0_cs = (int64_t)(segs[i].start_sec * 100.0f);
                int64_t t1_cs = (int64_t)(segs[i].end_sec * 100.0f);
                int s = std::max(0, (int)(segs[i].start_sec * sample_rate));
                int e = std::min(n_samples, (int)(segs[i].end_sec * sample_rate));
                if (e > s)
                    slices.push_back({s, e, t0_cs, t1_cs});
            }
            if (segs)
                free(segs);
            // Do NOT free vctx — it's cached.
        } else if (out_load_failed) {
            *out_load_failed = true; // whisper-vad-encdec model failed to load
        }
    }
#endif
#ifdef CA_HAVE_WEBRTC_VAD
    else if (vpath.find("webrtc") != std::string::npos) {
        // WebRTC VAD: pure algorithmic GMM, no model file needed.
        // The "model path" is just a sentinel (e.g. "webrtc" or "--vad-model webrtc").
        webrtc_vad_segment* segs = nullptr;
        int n_segs = 0;
        float min_speech_sec = opts.min_speech_duration_ms / 1000.0f;
        float min_silence_sec = opts.min_silence_duration_ms / 1000.0f;
        int rc =
            webrtc_vad_detect(samples, n_samples, &segs, &n_segs, opts.threshold, min_speech_sec, min_silence_sec, -1);
        if (rc == 0 && segs && n_segs > 0) {
            for (int i = 0; i < n_segs; i++) {
                int64_t t0_cs = (int64_t)(segs[i].start_sec * 100.0f);
                int64_t t1_cs = (int64_t)(segs[i].end_sec * 100.0f);
                int s = std::max(0, (int)(segs[i].start_sec * sample_rate));
                int e = std::min(n_samples, (int)(segs[i].end_sec * sample_rate));
                if (e > s)
                    slices.push_back({s, e, t0_cs, t1_cs});
            }
        }
        if (segs)
            free(segs);
    }
#endif
    else {
        // Default: Silero VAD via crispasr API.
        // Use a cached context to avoid the init/free cycle that causes
        // memory fragmentation and the 70x regression (#132). The cache is
        // shared mutable state, so hold g_silero_cache_mtx across the whole
        // detect — the lookup alone is not enough, the context's LSTM/probs
        // buffers are rewritten by whisper_vad_segments_from_samples and must
        // not be touched concurrently (the server slices VAD outside its
        // model_mutex, so concurrent requests would otherwise race here).
        std::lock_guard<std::mutex> vad_lock(g_silero_cache_mtx);
        whisper_vad_context* vctx = silero_vad_get_cached_locked(vad_model_path, opts.n_threads);
        if (!vctx) {
            fprintf(stderr, "crispasr: warning: failed to load VAD model '%s'\n", vad_model_path);
            if (out_load_failed)
                *out_load_failed = true;
            return slices;
        }

        whisper_vad_params vp = whisper_vad_default_params();
        vp.threshold = opts.threshold;
        vp.min_speech_duration_ms = opts.min_speech_duration_ms;
        vp.min_silence_duration_ms = opts.min_silence_duration_ms;
        vp.speech_pad_ms = (float)opts.speech_pad_ms;

        whisper_vad_segments* vseg = whisper_vad_segments_from_samples(vctx, vp, samples, n_samples);
        const int nv = vseg ? whisper_vad_segments_n_segments(vseg) : 0;
        for (int i = 0; i < nv; i++) {
            const float t0_cs = whisper_vad_segments_get_segment_t0(vseg, i);
            const float t1_cs = whisper_vad_segments_get_segment_t1(vseg, i);
            const float t0s = t0_cs / 100.0f;
            const float t1s = t1_cs / 100.0f;
            const int s = std::max(0, (int)(t0s * sample_rate));
            const int e = std::min(n_samples, (int)(t1s * sample_rate));
            if (e > s)
                slices.push_back({s, e, (int64_t)t0_cs, (int64_t)t1_cs});
        }
        if (vseg)
            whisper_vad_free_segments(vseg);
        // Do NOT free vctx — it's owned by the cache.
    }

    // Post-merge: offline/file callers keep the historical short/close merge.
    // JSON streaming can request a narrower close-gap-only policy so VAD never
    // hides a silence gap that should finalize an utterance.
    slices = crispasr_post_merge_vad_slices(slices, sample_rate, opts);

    // PLAN.md §W4: is this result obviously wrong?
    //
    // Assessed HERE — after the merge, before the re-chunk below. The re-chunk
    // splits long segments at `chunk_seconds`, so one 10-minute monologue
    // becomes ~20 slices; measuring segment COUNT after that would report a
    // healthy-looking number for any input and the few-segment signal could
    // never fire. This is the last point where the slice list still means
    // "what the VAD actually found".
    //
    // On failover we hand back fixed chunks over the whole buffer rather than
    // one giant slice: "transcribe everything" has to stay within the chunk
    // length the pipeline (and the backend's context) expects.
    // Disable with CRISPASR_VAD_FAILOVER=0.
    {
        const char* off = std::getenv("CRISPASR_VAD_FAILOVER");
        if (!(off && off[0] == '0') && sample_rate > 0) {
            std::vector<core_vad_failover::Span> spans;
            spans.reserve(slices.size());
            for (const auto& s : slices)
                spans.push_back({(double)s.t0_cs / 100.0, (double)s.t1_cs / 100.0});
            const double audio_sec = (double)n_samples / (double)sample_rate;
            const auto verdict = core_vad_failover::assess(spans, audio_sec);
            if (verdict.failover) {
                fprintf(stderr, "crispasr[vad]: %s — falling back to full-clip chunks\n", verdict.reason.c_str());
                const int chunk = opts.chunk_seconds > 0 ? opts.chunk_seconds : 30;
                return crispasr_fixed_chunk_slices(n_samples, sample_rate, chunk);
            }
        }
    }

    // Post-split: break any VAD segment that exceeds chunk_seconds into
    // sub-segments. See crispasr_rechunk_slices — the same step is reused on the
    // import path so a raw-segment export (issue #227) can be re-chunked to
    // whatever chunk length the importing run wants.
    if (opts.chunk_seconds > 0)
        slices = crispasr_rechunk_slices(slices, samples, n_samples, sample_rate, opts.chunk_seconds);

    return slices;
}

std::vector<crispasr_audio_slice> crispasr_rechunk_slices(const std::vector<crispasr_audio_slice>& in,
                                                          const float* samples, int n_samples, int sample_rate,
                                                          int chunk_seconds) {
    // Break any segment longer than chunk_seconds into sub-segments. Prevents
    // OOM on very long continuous speech (10+ min lectures). Cuts land on the
    // lowest-RMS 100 ms inside a ±2 s window around each target rather than at
    // equal parts, so a cut inside continuous speech does not slice mid-word
    // (issue #89: words spanning an arbitrary cut are lost by both adjacent
    // slices). Segments already within the limit pass through untouched, so a
    // raw-VAD-segment export re-chunked at any length reproduces exactly what a
    // fresh run at that length would have computed.
    if (chunk_seconds <= 0 || n_samples <= 0)
        return in;
    const int max_samples = chunk_seconds * sample_rate;
    const size_t search_window_samples = (size_t)(2.0 * sample_rate);
    const size_t energy_win_samples = (size_t)((double)sample_rate * 0.1); // 100 ms
    std::vector<crispasr_audio_slice> split;
    for (const auto& sl : in) {
        const int dur = sl.end - sl.start;
        if (dur <= max_samples || sl.start < 0 || sl.end > n_samples) {
            split.push_back(sl);
        } else {
            auto ranges = audio_chunking::split_at_energy_minima(samples + sl.start, (size_t)dur, (size_t)max_samples,
                                                                 search_window_samples, energy_win_samples);
            for (auto& r : ranges) {
                const int s = sl.start + (int)r.first;
                const int e = sl.start + (int)r.second;
                split.push_back({
                    s,
                    e,
                    (int64_t)((double)s / sample_rate * 100.0),
                    (int64_t)((double)e / sample_rate * 100.0),
                });
            }
        }
    }
    return split;
}

std::vector<crispasr_audio_slice> crispasr_fixed_chunk_slices(int n_samples, int sample_rate, int chunk_seconds) {
    std::vector<crispasr_audio_slice> slices;
    if (n_samples <= 0)
        return slices;

    const int chunk_samples = chunk_seconds > 0 ? chunk_seconds * sample_rate : n_samples;
    if (n_samples <= chunk_samples) {
        const int64_t dur_cs = (int64_t)((double)n_samples / sample_rate * 100.0);
        slices.push_back({0, n_samples, 0, dur_cs});
        return slices;
    }
    for (int s = 0; s < n_samples; s += chunk_samples) {
        const int e = std::min(n_samples, s + chunk_samples);
        slices.push_back({
            s,
            e,
            (int64_t)((double)s / sample_rate * 100.0),
            (int64_t)((double)e / sample_rate * 100.0),
        });
    }
    return slices;
}

std::vector<crispasr_audio_slice> crispasr_energy_chunk_slices(const float* samples, int n_samples, int sample_rate,
                                                               int chunk_seconds, float search_window_seconds) {
    std::vector<crispasr_audio_slice> slices;
    if (n_samples <= 0 || !samples)
        return slices;
    if (chunk_seconds <= 0 || search_window_seconds <= 0.0f)
        return crispasr_fixed_chunk_slices(n_samples, sample_rate, chunk_seconds);
    const size_t chunk_samples = (size_t)chunk_seconds * (size_t)sample_rate;
    const size_t search_window_samples = (size_t)(search_window_seconds * (float)sample_rate);
    const size_t energy_win_samples = (size_t)((double)sample_rate * 0.1); // 100 ms

    auto ranges = audio_chunking::split_at_energy_minima(samples, (size_t)n_samples, chunk_samples,
                                                         search_window_samples, energy_win_samples);
    // Issue #471: a cut placed in the silence after the last word leaves a
    // remainder with no speech; LLM backends hallucinate on it (qwen3 returns
    // the --hotwords list). Drop such slices, as the VAD path does (#213).
    if (audio_chunking::speechless_gate_enabled()) {
        size_t n_dropped = 0;
        ranges =
            audio_chunking::drop_speechless_ranges(samples, (size_t)n_samples, ranges, energy_win_samples, &n_dropped);
        if (n_dropped > 0)
            fprintf(stderr,
                    "crispasr: skipped %zu energy-chunk slice(s) with no speech above the noise floor "
                    "(CRISPASR_ENERGY_SILENCE_GATE=0 keeps them)\n",
                    n_dropped);
    }
    slices.reserve(ranges.size());
    for (auto& r : ranges) {
        const int s = (int)r.first;
        const int e = (int)r.second;
        slices.push_back({
            s,
            e,
            (int64_t)((double)s / sample_rate * 100.0),
            (int64_t)((double)e / sample_rate * 100.0),
        });
    }
    return slices;
}

crispasr_stitched_audio crispasr_stitch_vad_slices(const float* samples, int /*n_samples*/, int sample_rate,
                                                   const std::vector<crispasr_audio_slice>& slices) {
    crispasr_stitched_audio result;
    if (slices.empty())
        return result;

    const int silence_samples = (int)(0.1f * sample_rate); // 0.1s silence gap

    size_t total = 0;
    for (const auto& sl : slices)
        total += (size_t)(sl.end - sl.start);
    total += (size_t)(slices.size() - 1) * silence_samples;

    result.samples.resize(total, 0.0f);
    result.mapping.reserve(slices.size() * 2);

    int offset = 0;
    for (size_t i = 0; i < slices.size(); i++) {
        const auto& sl = slices[i];
        const int seg_len = sl.end - sl.start;

        result.mapping.push_back({(int64_t)((double)offset / sample_rate * 100.0), sl.t0_cs});
        std::memcpy(result.samples.data() + offset, samples + sl.start, (size_t)seg_len * sizeof(float));
        offset += seg_len;
        result.mapping.push_back({(int64_t)((double)offset / sample_rate * 100.0), sl.t1_cs});

        if (i + 1 < slices.size())
            offset += silence_samples;
    }

    result.total_duration_cs = (int64_t)((double)offset / sample_rate * 100.0);
    return result;
}

int64_t crispasr_vad_remap_timestamp(const std::vector<crispasr_vad_mapping>& mapping, int64_t stitched_cs) {
    if (mapping.empty())
        return stitched_cs;
    if (stitched_cs <= mapping.front().stitched_cs)
        return mapping.front().original_cs;
    if (stitched_cs >= mapping.back().stitched_cs)
        return mapping.back().original_cs;

    size_t lo = 0, hi = mapping.size() - 1;
    while (lo + 1 < hi) {
        size_t mid = (lo + hi) / 2;
        if (mapping[mid].stitched_cs <= stitched_cs)
            lo = mid;
        else
            hi = mid;
    }
    const auto& a = mapping[lo];
    const auto& b = mapping[hi];
    if (b.stitched_cs == a.stitched_cs)
        return a.original_cs;
    const double frac = (double)(stitched_cs - a.stitched_cs) / (double)(b.stitched_cs - a.stitched_cs);
    return a.original_cs + (int64_t)(frac * (double)(b.original_cs - a.original_cs));
}

// ---- VAD segment boundary export / import (issue #227) ----

bool crispasr_vad_chunk_mismatch(float imported_chunk, float requested_chunk) {
    const float req = requested_chunk > 0.0f ? requested_chunk : 30.0f;
    return imported_chunk > 0.0f && std::fabs(imported_chunk - req) > 0.01f;
}

std::string crispasr_serialize_vad_slices(const std::vector<crispasr_audio_slice>& slices, int sample_rate,
                                          float chunk_seconds, bool is_raw_segments) {
    std::string out;
    out.reserve(80 + slices.size() * 96);
    out += "{\n  \"crispasr_vad\": {\n";
    out += "    \"version\": 1,\n";
    // "kind" distinguishes the two exportable forms (issue #227):
    //   "chunks"       -- chunk boundaries; valid only for the chunk length that
    //                     produced them, hence chunk_cs and the import gate.
    //   "vad_segments" -- raw speech segments; chunk-length-independent, re-chunked
    //                     per run on import. chunk_cs is 0/absent for these.
    // Absent "kind" (files written before this field) is read as "chunks", which
    // is the historical behaviour.
    out += std::string("    \"kind\": \"") + (is_raw_segments ? "vad_segments" : "chunks") + "\",\n";
    out += "    \"sample_rate\": " + std::to_string(sample_rate) + ",\n";
    // The slices are CHUNK boundaries, not raw speech segments, so they depend
    // on the chunk length that produced them (issue #227: exporting at 30 s and
    // importing at 5 s silently reuses the wrong chunking). Recorded in
    // centiseconds so the existing integer field parser can read it back; 0
    // means "written by a version that did not record it".
    if (!is_raw_segments)
        out += "    \"chunk_cs\": " + std::to_string((long long)(chunk_seconds * 100.0f + 0.5f)) + ",\n";
    out += "    \"num_slices\": " + std::to_string(slices.size()) + ",\n";
    out += "    \"slices\": [";
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& s = slices[i];
        out += (i == 0) ? "\n" : ",\n";
        out += "      { \"start\": " + std::to_string(s.start) + ", \"end\": " + std::to_string(s.end) +
               ", \"t0_cs\": " + std::to_string((long long)s.t0_cs) +
               ", \"t1_cs\": " + std::to_string((long long)s.t1_cs) + " }";
    }
    out += slices.empty() ? "]\n" : "\n    ]\n";
    out += "  }\n}\n";
    return out;
}

namespace {

// Find the value of an integer field `"<key>"` starting at or after `from`
// within [begin, end) of `text`. Returns true and sets `value` + `next`
// (index just past the parsed number) on success. Tolerant of whitespace
// between the key, the colon, and the number.
bool ca_vad_find_int(const std::string& text, size_t from, size_t end, const char* key, int64_t& value, size_t& next) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t k = text.find(needle, from);
    if (k == std::string::npos || k >= end)
        return false;
    size_t p = k + needle.size();
    while (p < end && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r'))
        p++;
    if (p >= end || text[p] != ':')
        return false;
    p++;
    while (p < end && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r'))
        p++;
    if (p >= end)
        return false;
    size_t num_start = p;
    if (text[p] == '+' || text[p] == '-')
        p++;
    size_t digits = 0;
    while (p < end && text[p] >= '0' && text[p] <= '9') {
        p++;
        digits++;
    }
    if (digits == 0)
        return false;
    value = (int64_t)std::strtoll(text.c_str() + num_start, nullptr, 10);
    next = p;
    return true;
}

} // namespace

bool crispasr_parse_vad_slices(const std::string& text, std::vector<crispasr_audio_slice>& out, int* sample_rate_out,
                               float* chunk_seconds_out, bool* is_raw_segments_out) {
    out.clear();
    if (sample_rate_out)
        *sample_rate_out = 0;
    if (chunk_seconds_out)
        *chunk_seconds_out = 0.0f;
    if (is_raw_segments_out) {
        // Default "chunks": a file with no "kind" predates the field and is a
        // chunk export by definition.
        size_t arr0 = text.find("\"slices\"");
        *is_raw_segments_out = arr0 != std::string::npos && text.rfind("\"vad_segments\"", arr0) != std::string::npos;
    }

    // Optional top-level sample_rate (before the slices array).
    size_t arr = text.find("\"slices\"");
    if (arr == std::string::npos)
        return false;
    if (sample_rate_out) {
        int64_t sr = 0;
        size_t tmp = 0;
        if (ca_vad_find_int(text, 0, arr, "sample_rate", sr, tmp))
            *sample_rate_out = (int)sr;
    }
    if (chunk_seconds_out) {
        int64_t cs = 0;
        size_t tmp = 0;
        if (ca_vad_find_int(text, 0, arr, "chunk_cs", cs, tmp))
            *chunk_seconds_out = (float)cs / 100.0f;
    }

    size_t lb = text.find('[', arr);
    if (lb == std::string::npos)
        return false;
    size_t rb = text.find(']', lb);
    if (rb == std::string::npos)
        return false;

    // Walk each { ... } object inside the array.
    size_t p = lb + 1;
    while (p < rb) {
        size_t obj_lb = text.find('{', p);
        if (obj_lb == std::string::npos || obj_lb >= rb)
            break;
        size_t obj_rb = text.find('}', obj_lb);
        if (obj_rb == std::string::npos || obj_rb > rb)
            return false;

        int64_t start = 0, endv = 0, t0 = 0, t1 = 0;
        size_t nx = 0;
        if (!ca_vad_find_int(text, obj_lb, obj_rb, "start", start, nx) ||
            !ca_vad_find_int(text, obj_lb, obj_rb, "end", endv, nx) ||
            !ca_vad_find_int(text, obj_lb, obj_rb, "t0_cs", t0, nx) ||
            !ca_vad_find_int(text, obj_lb, obj_rb, "t1_cs", t1, nx)) {
            out.clear();
            return false;
        }
        crispasr_audio_slice s;
        s.start = (int)start;
        s.end = (int)endv;
        s.t0_cs = t0;
        s.t1_cs = t1;
        out.push_back(s);
        p = obj_rb + 1;
    }
    return true;
}
