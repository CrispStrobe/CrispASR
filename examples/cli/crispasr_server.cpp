// crispasr_server.cpp — HTTP server with persistent model for all backends.
//
// Keeps the model loaded in memory between requests. Accepts audio via
// POST /inference (multipart file upload) and returns JSON transcription.
//
// Usage:
//   crispasr --server -m model.gguf [--port 8080] [--host 127.0.0.1]
//
// Endpoints:
//   POST /inference                   — transcribe (native JSON)
//   POST /v1/audio/transcriptions     — OpenAI-compatible endpoint
//   POST /v1/audio/speech             — TTS (OpenAI-compatible; CAP_TTS only)
//   POST /v1/audio/speech-to-speech   — S2S audio→audio (CAP_S2S only)
//   POST /load                        — hot-swap model
//   GET  /health                      — server status
//   GET  /backends                    — list available backends
//   GET  /v1/models                   — OpenAI-compatible model list
//   GET  /v1/voices                   — list voices in --voice-dir (CAP_TTS only)
//   POST /v1/voices                   — upload voice file (multipart, CAP_TTS only)
//   DELETE /v1/voices/:name           — delete voice file (CAP_TTS only)
//
// Adapted from examples/server/server.cpp for multi-backend support.

#include "crispasr_backend.h"
#include "crispasr_diarize_cli.h"
#include "crispasr_speaker_embedder.h"
#include "crispasr_lid.h"
#include "crispasr_lid_cli.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_vad_cli.h"
#include "crispasr_aligner_cli.h"
#include "whisper_params.h"
#include "fireredpunc.h"                 // server-mode punctuation restoration (--punc-model)
#include "pcs.h"                         // PCS (punctuation + caps + segmentation) model
#include "crispasr_cache.h"              // ensure_cached_file() for resolving the punc model
#include "crispasr_punc_loader.h"        // shared --punc-model alias resolution (CLI parity)
#include "crispasr_truecase_loader.h"    // shared --truecase-model resolution + apply (CLI parity)
#include "crispasr_punctuation_policy.h" // crispasr_should_auto_enable_punctuation()

#include "common-crispasr.h"           // read_audio_data
#include "crispasr_chat.h"             // /v1/chat/completions
#include "../server/ws_stream.h"       // real-time WebSocket ASR streaming (--ws-port)
#include "../server/realtime_server.h" // vLLM Realtime API
#include "wyoming.h"                   // Wyoming protocol for Home Assistant Assist (--wyoming-port)
#include "crispasr_c2pa.h"
#include "crispasr_tts_chunking.h"
#include "crispasr_tts_disclaimer.h"
#include "crispasr_watermark.h"
#include "crispasr_watermark_dispatch.h"
#include "crispasr_wav_writer.h"
#include "../server/httplib.h"
#include "../json.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h> // _mktemp_s
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h> // mkstemp, close, unlink
#endif

// 75e: optional MP3/Opus output encoding (compile-time gated)
#ifdef CRISPASR_HAVE_LAME
#include <lame/lame.h>
#endif
#ifdef CRISPASR_HAVE_OPUS
#include <opus/opus.h>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string scratch_dir() {
    // Pick a writable scratch dir, preferring explicit overrides. Use the
    // non-throwing create_directories overload and fall back to the system temp
    // dir on failure: this runs on the per-request transcription path, so a
    // create_directories exception (e.g. a dangling cache symlink or read-only
    // HOME) must not blow up every request.
    std::string d;
    if (const char* env = std::getenv("CRISPASR_SCRATCH_DIR"); env && *env)
        return std::string(env); // explicit override: trust the caller, don't mkdir
    else if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache && *cache)
        d = std::string(cache) + "/crispasr/scratch";
    else if (const char* home = std::getenv("HOME"); home && *home)
        d = std::string(home) + "/.cache/crispasr/scratch";
    else
        d = ".cache/crispasr/scratch";

    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    if (ec || !std::filesystem::is_directory(d, ec)) {
        std::error_code tec;
        std::filesystem::path fallback = std::filesystem::temp_directory_path(tec) / "crispasr-scratch";
        std::filesystem::create_directories(fallback, tec);
        return fallback.string();
    }
    return d;
}

// Create a scratch file securely via mkstemp (POSIX) or _mktemp_s (Win).
// Writes `data` to it and returns the path. On failure returns "".
// The caller is responsible for calling std::remove() on the returned path.
// Preserve the original file extension so ffmpeg and miniaudio can detect
// the container format (critical for m4a/aac/opus/webm uploads).
static std::string write_temp_audio(const char* data, size_t size, const std::string& original_filename = "") {
    // Extract extension from original filename
    std::string ext;
    if (!original_filename.empty()) {
        auto dot = original_filename.rfind('.');
        if (dot != std::string::npos)
            ext = original_filename.substr(dot); // e.g. ".m4a"
    }
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, tmp_dir))
        return "";
    char tmp_path[MAX_PATH];
    if (!GetTempFileNameA(tmp_dir, "cra", 0, tmp_path))
        return "";
    std::string final_path = std::string(tmp_path) + ext;
    if (!ext.empty())
        MoveFileA(tmp_path, final_path.c_str());
    else
        final_path = tmp_path;
    std::ofstream f(final_path, std::ios::binary);
    if (!f)
        return "";
    f.write(data, (std::streamsize)size);
    f.close();
    return final_path;
#else
    std::string tmpl_s = scratch_dir() + "/crispasr-XXXXXX" + ext;
    // mkstemps requires the suffix length
    int suffix_len = (int)ext.size();
    std::vector<char> tmpl(tmpl_s.begin(), tmpl_s.end());
    tmpl.push_back('\0');
    int fd = suffix_len > 0 ? mkstemps(tmpl.data(), suffix_len) : mkstemp(tmpl.data());
    if (fd < 0)
        return "";
    // Write all data; retry on partial write.
    const char* p = data;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            ::close(fd);
            ::unlink(tmpl.data());
            return "";
        }
        p += n;
        remaining -= (size_t)n;
    }
    ::close(fd);
    return std::string(tmpl.data());
#endif
}

// Read a form field as a trimmed string, or return a default.
static std::string form_string(const httplib::Request& req, const std::string& key, const std::string& def = "") {
    std::string v;
    if (req.has_file(key)) {
        v = req.get_file_value(key).content;
    } else if (req.has_param(key)) {
        v = req.get_param_value(key);
    } else {
        return def;
    }
    // Trim whitespace.
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
        v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t'))
        v.pop_back();
    return v.empty() ? def : v;
}

static std::string trim_copy(std::string v) {
    while (!v.empty() && (v.front() == ' ' || v.front() == '\t' || v.front() == '\r' || v.front() == '\n'))
        v.erase(v.begin());
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r' || v.back() == '\n'))
        v.pop_back();
    return v;
}

static std::vector<std::string> split_api_keys(const std::string& csv) {
    std::vector<std::string> keys;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty())
            keys.push_back(item);
    }
    return keys;
}

static bool fixed_time_equal(const std::string& a, const std::string& b) {
    unsigned char diff = (unsigned char)(a.size() ^ b.size());
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i)
        diff |= (unsigned char)(a[i] ^ b[i]);
    return diff == 0 && a.size() == b.size();
}

static std::string request_api_key(const httplib::Request& req) {
    if (req.has_header("Authorization")) {
        const std::string value = trim_copy(req.get_header_value("Authorization"));
        const std::string prefix = "Bearer ";
        if (value.rfind(prefix, 0) == 0)
            return trim_copy(value.substr(prefix.size()));
    }
    if (req.has_header("X-API-Key"))
        return trim_copy(req.get_header_value("X-API-Key"));
    return "";
}

static bool is_authorized(const httplib::Request& req, const std::vector<std::string>& api_keys) {
    if (api_keys.empty())
        return true;
    const std::string key = request_api_key(req);
    if (key.empty())
        return false;
    for (const std::string& expected : api_keys)
        if (fixed_time_equal(key, expected))
            return true;
    return false;
}

// Parse a form field as float, returning `def` on missing or parse error.
static float form_float(const httplib::Request& req, const std::string& key, float def) {
    if (!req.has_file(key) && !req.has_param(key))
        return def;
    const std::string v = req.has_file(key) ? req.get_file_value(key).content : req.get_param_value(key);
    try {
        size_t pos = 0;
        float f = std::stof(v, &pos);
        // Reject trailing garbage like "0.5abc".
        if (pos != v.size())
            return def;
        return f;
    } catch (...) {
        return def;
    }
}

static int form_int(const httplib::Request& req, const std::string& key, int def) {
    if (!req.has_file(key) && !req.has_param(key))
        return def;
    const std::string v = req.has_file(key) ? req.get_file_value(key).content : req.get_param_value(key);
    try {
        size_t pos = 0;
        int n = std::stoi(v, &pos);
        if (pos != v.size())
            return def;
        return n;
    } catch (...) {
        return def;
    }
}

static uint64_t form_u64(const httplib::Request& req, const std::string& key, uint64_t def) {
    if (!req.has_file(key) && !req.has_param(key))
        return def;
    const std::string v = req.has_file(key) ? req.get_file_value(key).content : req.get_param_value(key);
    try {
        size_t pos = 0;
        uint64_t n = std::stoull(v, &pos);
        if (pos != v.size())
            return def;
        return n;
    } catch (...) {
        return def;
    }
}

static bool form_bool(const httplib::Request& req, const std::string& key, bool def) {
    std::string v = form_string(req, key, "");
    if (v.empty())
        return def;
    // Accept "true", "1", "yes" (case-insensitive) as truthy.
    for (auto& c : v)
        c = (char)std::tolower((unsigned char)c);
    if (v == "true" || v == "1" || v == "yes")
        return true;
    if (v == "false" || v == "0" || v == "no")
        return false;
    return def;
}

// JSON error response helper. Shape matches OpenAI's:
//   { "error": { "message": ..., "type": ..., "code": ..., "param": ... } }
// `code` is a stable machine-readable enum-string the client can switch on
// (e.g. "voice_not_found", "input_too_long"); `param` is the offending
// request field name (e.g. "voice", "input"). Both default to "" and are
// omitted from the JSON body when empty so the on-wire shape stays
// minimal for non-OpenAI consumers.
static void json_error(httplib::Response& res, int status, const std::string& message, const std::string& code = "",
                       const std::string& param = "") {
    res.status = status;
    std::string body =
        "{\"error\": {\"message\": \"" + crispasr_json_escape(message) + "\", \"type\": \"invalid_request_error\"";
    if (!code.empty())
        body += ", \"code\": \"" + crispasr_json_escape(code) + "\"";
    if (!param.empty())
        body += ", \"param\": \"" + crispasr_json_escape(param) + "\"";
    body += "}}";
    res.set_content(body, "application/json");
}

static void auth_error(httplib::Response& res) {
    res.status = 401;
    res.set_header("WWW-Authenticate", "Bearer");
    res.set_content("{\"error\": {\"message\": \"invalid or missing API key\", \"type\": \"invalid_api_key\"}}",
                    "application/json");
}

// Shared transcription result.
struct transcription_result {
    bool ok = false;
    std::string error;
    std::vector<crispasr_segment> segs;
    std::string language;
    double duration_s = 0.0;
    double elapsed_s = 0.0;
};

// Load audio from a multipart file upload, transcribe it, return result.
// Acquires model_mutex internally.
static transcription_result do_transcribe(const httplib::MultipartFormData& audio_file, CrispasrBackend* backend,
                                          std::mutex& model_mutex, whisper_params rp, bool need_timestamps,
                                          fireredpunc_context* punc_ctx = nullptr, pcs_context* pcs_ctx = nullptr,
                                          truecaser_context* tc_ctx = nullptr,
                                          truecaser_crf_context* tc_crf_ctx = nullptr,
                                          truecaser_lstm_context* tc_lstm_ctx = nullptr) {
    transcription_result result;
    result.language = rp.language;

    if (rp.verbose)
        fprintf(stderr, "crispasr-server: processing '%s' (%zu bytes)\n", audio_file.filename.c_str(),
                audio_file.content.size());

    // Write to a secure temporary file for audio decoding.
    std::string tmp_path = write_temp_audio(audio_file.content.data(), audio_file.content.size(), audio_file.filename);
    if (tmp_path.empty()) {
        result.error = "failed to create temporary file for audio";
        return result;
    }

    // Decode audio.
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(tmp_path, pcmf32, pcmf32s, rp.diarize)) {
        std::remove(tmp_path.c_str());
        result.error = "failed to decode audio (unsupported format or corrupt file)";
        return result;
    }
    std::remove(tmp_path.c_str());

    if (pcmf32.empty()) {
        result.error = "audio file contains no samples";
        return result;
    }

    result.duration_s = (double)pcmf32.size() / 16000.0;

    const bool want_auto_lang = rp.detect_language || rp.language == "auto";
    const bool has_native_lid = (backend->capabilities() & CAP_LANGUAGE_DETECT) != 0;
    const bool lid_disabled = rp.lid_backend == "off" || rp.lid_backend == "none";

    // Auto-chunk long audio to prevent OOM (#27).
    // Most backends have O(T²) attention in the encoder - VAD or 30s fixed chunks keep
    // memory bounded. The slice t0 values become the absolute timestamp base
    const int SR = 16000;
    const int n_samples = (int)pcmf32.size();
    // Mirror the CLI (crispasr_run.cpp): when VAD is active on a
    // CAP_UNBOUNDED_INPUT backend and chunk_seconds wasn't set explicitly, drop
    // the fixed max-split (0) so VAD slices aren't over-subdivided into 30 s
    // pieces — VAD already bounds them. Fewer, larger slices ⇒ less chunk-
    // boundary-overlap recompute (the server made ~2× the slices the CLI does on
    // the same VAD segments, #165). Without VAD the 30 s fixed chunking stays
    // (keeps parakeet et al. inside their safe single-pass window, #89).
    int effective_chunk_seconds = rp.chunk_seconds;
    if (rp.vad && !rp.chunk_seconds_explicit && (backend->capabilities() & CAP_UNBOUNDED_INPUT))
        effective_chunk_seconds = 0;
    const auto slices = crispasr_compute_audio_slices(pcmf32.data(), n_samples, SR, effective_chunk_seconds, rp);
    if (slices.empty()) {
        result.ok = true;
        return result;
    }

    // VAD (if any) already ran above — disable it for the backend so
    // whisper_full doesn't re-run Silero on every slice (#132).
    rp.vad = false;
    rp.vad_model.clear();

    {
        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::steady_clock::now();

        // Match file-mode `-l auto`: run LID once per uploaded audio sample
        // before dispatching chunks to backends that need explicit language.
        if (want_auto_lang && !has_native_lid && !lid_disabled) {
            crispasr_lid_result lid;
            if (crispasr_detect_language_cli(pcmf32.data(), (int)pcmf32.size(), rp, lid)) {
                rp.language = lid.lang_code;
                if (rp.source_lang.empty() || rp.source_lang == "auto")
                    rp.source_lang = lid.lang_code;
                if (!rp.no_prints) {
                    fprintf(stderr, "crispasr-server: LID -> language = '%s' (%s, p=%.3f)\n", lid.lang_code.c_str(),
                            lid.source.c_str(), lid.confidence);
                }
            } else if (rp.language == "auto") {
                if (!rp.no_prints) {
                    fprintf(stderr, "crispasr-server: LID failed and no -l was set — "
                                    "defaulting to 'en'. Pass `-l <code>` or a request language field to override.\n");
                }
                rp.language = "en";
                if (rp.source_lang.empty() || rp.source_lang == "auto")
                    rp.source_lang = "en";
            } else if (!rp.no_prints) {
                fprintf(stderr, "crispasr-server: LID failed, falling back to language='%s'\n", rp.language.c_str());
            }
            // Keep the LID model resident across requests (freed once at server
            // shutdown, like the VAD cache). Previously freed here per request,
            // which reloaded the whisper-LID model on every `language=auto`
            // transcription (#165). Set `language` explicitly to skip LID entirely.
        }
        result.language = rp.language;

        if (!rp.no_prints && slices.size() > 1) {
            fprintf(stderr, "crispasr-server: processing %zu slice(s)\n", slices.size());
        }

        const bool want_align = need_timestamps && !rp.aligner_model.empty() &&
                                ((backend->capabilities() & CAP_TIMESTAMPS_CTC) || rp.force_aligner);
        if (rp.verbose) {
            fprintf(stderr,
                    "crispasr-server[verbose]: align: need_ts=%d aligner='%s' caps_ctc=%d force=%d -> want=%d\n",
                    need_timestamps ? 1 : 0, rp.aligner_model.c_str(), !!(backend->capabilities() & CAP_TIMESTAMPS_CTC),
                    rp.force_aligner ? 1 : 0, want_align ? 1 : 0);
        }

        for (size_t i = 0; i < slices.size(); ++i) {
            const auto& sl = slices[i];
            auto tc0 = std::chrono::steady_clock::now();
            auto segs = backend->transcribe(pcmf32.data() + sl.start, sl.end - sl.start, sl.t0_cs, rp);

            if (want_align) {
                for (auto& seg : segs) {
                    if (!seg.words.empty() && !rp.force_aligner)
                        continue;
                    auto words = crispasr_ctc_align(rp.aligner_model, seg.text, pcmf32.data() + sl.start,
                                                    sl.end - sl.start, sl.t0_cs, rp.n_threads);
                    if (!words.empty()) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }

            for (auto& seg : segs)
                result.segs.push_back(std::move(seg));

            if (!rp.no_prints && slices.size() > 1) {
                auto tc1 = std::chrono::steady_clock::now();
                double slice_s = std::chrono::duration<double>(tc1 - tc0).count();
                fprintf(stderr, "crispasr-server: slice %zu/%zu done (%.1fs audio in %.1fs)\n", i + 1, slices.size(),
                        (sl.end - sl.start) / (double)SR, slice_s);
            }
        }

        // Diarization post-step (#143): assign speaker labels to segments.
        // Mirrors the CLI path in crispasr_run.cpp:732-743.
        if (rp.diarize && !result.segs.empty()) {
            const bool have_stereo = pcmf32s.size() == 2 && !pcmf32s[0].empty() && !pcmf32s[1].empty();

            // Pre-compute global caches for cross-slice consistency.
            CrispasrPyannoteCache pyannote_cache;
            if (rp.diarize_method == "pyannote" && !pcmf32.empty()) {
                crispasr_compute_pyannote_cache(pcmf32.data(), n_samples, rp, pyannote_cache);
            }
            CrispasrSherpaCache sherpa_cache;
            if ((rp.diarize_method == "sherpa" || rp.diarize_method == "sherpa-onnx" || rp.diarize_method == "ecapa") &&
                !pcmf32.empty()) {
                crispasr_compute_sherpa_cache(pcmf32.data(), n_samples, rp, sherpa_cache);
            }
            const CrispasrPyannoteCache* pya_ptr = pyannote_cache.valid() ? &pyannote_cache : nullptr;
            const CrispasrSherpaCache* shp_ptr = sherpa_cache.valid() ? &sherpa_cache : nullptr;

            // Apply diarize per-slice. We re-walk the slices and apply
            // diarize to the corresponding range of result.segs.
            size_t seg_offset = 0;
            for (size_t i = 0; i < slices.size(); ++i) {
                const auto& sl = slices[i];
                // Count how many segments belong to this slice (by timestamp range).
                size_t seg_count = 0;
                for (size_t j = seg_offset; j < result.segs.size(); ++j) {
                    // Segments from the next slice will have t0 >= next slice's t0_cs.
                    if (i + 1 < slices.size() && result.segs[j].t0 >= slices[i + 1].t0_cs)
                        break;
                    seg_count++;
                }

                if (seg_count > 0) {
                    std::vector<crispasr_segment> slice_segs(result.segs.begin() + (ptrdiff_t)seg_offset,
                                                             result.segs.begin() + (ptrdiff_t)(seg_offset + seg_count));
                    if (have_stereo) {
                        std::vector<float> sl_l(pcmf32s[0].begin() + sl.start, pcmf32s[0].begin() + sl.end);
                        std::vector<float> sl_r(pcmf32s[1].begin() + sl.start, pcmf32s[1].begin() + sl.end);
                        crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true, sl.t0_cs, slice_segs, rp, pya_ptr,
                                               shp_ptr);
                    } else {
                        std::vector<float> mono_slice(pcmf32.begin() + sl.start, pcmf32.begin() + sl.end);
                        crispasr_apply_diarize(mono_slice, mono_slice,
                                               /*is_stereo=*/false, sl.t0_cs, slice_segs, rp, pya_ptr, shp_ptr);
                    }
                    // Copy back the diarized segments.
                    for (size_t j = 0; j < seg_count; ++j)
                        result.segs[seg_offset + j] = std::move(slice_segs[j]);
                }
                seg_offset += seg_count;
            }

            // Global embedding-based re-clustering (issue #107 P3).
            if (!rp.diarize_embedder.empty() && !pcmf32.empty()) {
                auto embedder = crispasr_make_speaker_embedder(rp.diarize_embedder, rp.n_threads, rp.cache_dir);
                if (embedder) {
                    crispasr_remap_speakers_via_embeddings(result.segs, pcmf32.data(), n_samples, embedder.get(), rp);
                }
            }
        }

        // Punctuation stripping: when `--no-punctuation` / `punctuation=false`
        // is set and the backend doesn't natively toggle it, strip here.
        if (!rp.punctuation) {
            for (auto& seg : result.segs)
                crispasr_strip_punctuation(seg);
        }
        // Otherwise, when a punctuation model is loaded (--punc-model), restore
        // punctuation on each segment — the post-processor the CLI applies but
        // the server path previously skipped, so non-PnC backends (e.g. parakeet
        // RNNT/CTC) can return punctuated text. PCS (one model for punctuation +
        // capitalization + segmentation) takes precedence over FireRedPunc when
        // loaded; only one is ever resident. Serialized on its own mutex because
        // neither context is re-entrant.
        else if (punc_ctx || pcs_ctx) {
            static std::mutex punc_mtx;
            std::lock_guard<std::mutex> plk(punc_mtx);
            for (auto& seg : result.segs) {
                char* out =
                    pcs_ctx ? pcs_process(pcs_ctx, seg.text.c_str()) : fireredpunc_process(punc_ctx, seg.text.c_str());
                if (out) {
                    seg.text = out;
                    free(out);
                }
            }
        }

        // Truecasing post-step (--truecase-model), applied after punctuation —
        // mirrors the CLI, which the server path previously skipped entirely.
        // PCS already restores casing, so skip when PCS is active. Serialized:
        // the truecaser contexts aren't re-entrant.
        if (!pcs_ctx && (tc_ctx || tc_crf_ctx || tc_lstm_ctx)) {
            static std::mutex tc_mtx;
            std::lock_guard<std::mutex> tlk(tc_mtx);
            for (auto& seg : result.segs)
                crispasr_apply_truecase(tc_ctx, tc_crf_ctx, tc_lstm_ctx, seg.text);
        }

        auto t1 = std::chrono::steady_clock::now();
        result.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    }

    result.ok = true;
    return result;
}

// crispasr_make_wav_int16 lives in crispasr_wav_writer.h so the unit
// tests can exercise it without linking the server translation unit.

// ---------------------------------------------------------------------------
// 75e: MP3 / Opus encoding helpers (compile-time gated)
// ---------------------------------------------------------------------------

#ifdef CRISPASR_HAVE_LAME
// Encode float32 mono PCM to MP3 via libmp3lame. Returns empty on failure.
static std::string crispasr_encode_mp3(const float* pcm, int n_samples, int sample_rate, int bitrate_kbps = 128) {
    lame_t lame = lame_init();
    if (!lame)
        return {};
    lame_set_in_samplerate(lame, sample_rate);
    lame_set_num_channels(lame, 1);
    lame_set_out_samplerate(lame, sample_rate);
    lame_set_brate(lame, bitrate_kbps);
    lame_set_quality(lame, 2); // 2 = high quality
    lame_set_mode(lame, MONO);
    if (lame_init_params(lame) < 0) {
        lame_close(lame);
        return {};
    }

    // Convert float [-1,1] → int16
    std::vector<short> s16(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float v = pcm[i];
        if (v > 1.0f)
            v = 1.0f;
        else if (v < -1.0f)
            v = -1.0f;
        s16[i] = (short)(v * 32767.0f);
    }

    // Worst-case: 1.25 * n + 7200 (lame docs)
    size_t mp3_buf_size = (size_t)(1.25f * (float)n_samples) + 7200;
    std::vector<unsigned char> mp3_buf(mp3_buf_size);

    int written = lame_encode_buffer(lame, s16.data(), nullptr, n_samples, mp3_buf.data(), (int)mp3_buf_size);
    if (written < 0) {
        lame_close(lame);
        return {};
    }
    int flushed = lame_encode_flush(lame, mp3_buf.data() + written, (int)(mp3_buf_size - (size_t)written));
    lame_close(lame);
    if (flushed < 0)
        return {};

    // Prepend ID3v2 tag with AI-provenance metadata (TXXX frames)
    std::string id3 = crispasr_make_id3v2_ai_tag();
    id3.append((const char*)mp3_buf.data(), (size_t)(written + flushed));
    return id3;
}
#endif // CRISPASR_HAVE_LAME

#ifdef CRISPASR_HAVE_OPUS
// Encode float32 mono PCM to raw Opus frames concatenated with 2-byte
// little-endian length prefix per frame. Opus requires 48/24/16/12/8 kHz
// input; we resample to 48 kHz if needed using linear interpolation
// (good enough for speech; the Opus encoder does its own filtering).
static std::string crispasr_encode_opus(const float* pcm, int n_samples, int sample_rate, int bitrate = 64000) {
    // Resample to 48 kHz if needed
    std::vector<float> resampled;
    const float* src = pcm;
    int src_n = n_samples;
    int enc_rate = sample_rate;

    // Opus supports 8000, 12000, 16000, 24000, 48000
    if (sample_rate != 48000 && sample_rate != 24000 && sample_rate != 16000 && sample_rate != 12000 &&
        sample_rate != 8000) {
        enc_rate = 48000;
        int out_n = (int)((int64_t)n_samples * 48000 / sample_rate);
        resampled.resize(out_n);
        for (int i = 0; i < out_n; i++) {
            float pos = (float)i * (float)sample_rate / 48000.0f;
            int s0 = (int)pos;
            int s1 = std::min(s0 + 1, n_samples - 1);
            float frac = pos - (float)s0;
            resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
        }
        src = resampled.data();
        src_n = out_n;
    }

    int error = 0;
    OpusEncoder* enc = opus_encoder_create(enc_rate, 1, OPUS_APPLICATION_VOIP, &error);
    if (error != OPUS_OK || !enc)
        return {};
    opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate));

    // Encode in 20ms frames
    const int frame_samples = enc_rate / 50; // 20ms
    // Max opus frame is 1275 bytes per channel per frame
    std::vector<unsigned char> frame_buf(4000);
    std::string result;
    result.reserve((size_t)(src_n / 4)); // rough estimate

    for (int offset = 0; offset + frame_samples <= src_n; offset += frame_samples) {
        int encoded =
            opus_encode_float(enc, src + offset, frame_samples, frame_buf.data(), (opus_int32)frame_buf.size());
        if (encoded < 0)
            break;
        // Write 2-byte LE length prefix + frame data
        uint16_t len = (uint16_t)encoded;
        result.append((const char*)&len, 2);
        result.append((const char*)frame_buf.data(), (size_t)encoded);
    }

    opus_encoder_destroy(enc);
    return result;
}
#endif // CRISPASR_HAVE_OPUS

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

int crispasr_run_server(whisper_params& params, const std::string& host, int port) {
    using namespace httplib;

    crispasr_c2pa_startup_check();
    if (!params.watermark_model.empty()) {
        crispasr_wm_dispatch::init(params.watermark_model);
    }

    std::vector<std::string> api_keys = split_api_keys(params.server_api_keys);
    if (const char* env_keys = getenv("CRISPASR_API_KEYS")) {
        std::vector<std::string> more = split_api_keys(env_keys);
        api_keys.insert(api_keys.end(), more.begin(), more.end());
    }

    std::unique_ptr<CrispasrBackend> backend;
    std::mutex model_mutex;
    std::atomic<bool> ready{false};
    std::string backend_name = params.backend;

    // Initial model load
    {
        const bool model_is_auto = params.model == "auto" || params.model == "default";
        if (backend_name.empty() || backend_name == "auto") {
            if (model_is_auto) {
                backend_name = "whisper";
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr-server: -m auto with no backend — defaulting to whisper\n");
                }
            } else {
                backend_name = crispasr_detect_backend_from_gguf(params.model);
            }
        }
        if (backend_name.empty()) {
            fprintf(stderr, "crispasr-server: cannot detect backend from '%s'\n", params.model.c_str());
            return 1;
        }

        const std::string resolved =
            crispasr_resolve_model_cli(params.model, backend_name, params.no_prints, params.cache_dir,
                                       params.auto_download || model_is_auto, params.model_quant);
        if (resolved.empty()) {
            fprintf(stderr, "crispasr-server: failed to resolve model '%s' for backend '%s'\n", params.model.c_str(),
                    backend_name.c_str());
            return 1;
        }
        params.model = resolved;

        if (params.aligner_model == "auto" || params.aligner_model == "default") {
            const std::string resolved_aligner = crispasr_resolve_model_cli(
                params.aligner_model, "canary-ctc-aligner", params.no_prints, params.cache_dir, params.auto_download);
            if (resolved_aligner.empty()) {
                fprintf(stderr, "crispasr-server: failed to resolve aligner '%s'\n", params.aligner_model.c_str());
                return 1;
            }
            params.aligner_model = resolved_aligner;
        } else if (!params.aligner_model.empty()) {
            params.aligner_model = crispasr_resolve_model_cli(params.aligner_model, "", params.no_prints,
                                                              params.cache_dir, params.auto_download);
        }

        backend = crispasr_create_backend(backend_name);
        if (!backend || !backend->init(params)) {
            fprintf(stderr, "crispasr-server: failed to init backend '%s'\n", backend_name.c_str());
            return 1;
        }
        // #80e: warmup in server mode — on by default (amortized over many
        // requests). Opt out with --no-warmup (or CRISPASR_NO_WARMUP=1): some
        // GPU drivers crash or hang inside the warmup transcribe — e.g. the
        // parakeet warmup on certain Vulkan drivers (#165) — which would
        // otherwise prevent the server from ever reaching listen(). Guard the
        // call so a soft (throwing) warmup failure degrades to "no warmup"
        // instead of taking the whole server down before it can serve.
        const bool skip_warmup = params.no_warmup || [] {
            const char* e = std::getenv("CRISPASR_NO_WARMUP");
            return e && e[0] && e[0] != '0';
        }();
        if (skip_warmup) {
            fprintf(stderr, "crispasr-server: warmup skipped (--no-warmup)\n");
        } else {
            auto t0 = std::chrono::steady_clock::now();
            try {
                backend->warmup();
                auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "crispasr-server: warmup completed in %.0f ms\n", dt * 1000.0);
            } catch (const std::exception& e) {
                fprintf(stderr, "crispasr-server: warning: warmup failed (%s) — continuing without warmup\n", e.what());
            } catch (...) {
                fprintf(stderr, "crispasr-server: warning: warmup failed — continuing without warmup\n");
            }
        }
        ready.store(true);
        fprintf(stderr, "crispasr-server: backend '%s' loaded, model '%s'\n", backend_name.c_str(),
                params.model.c_str());
    }

    // Punctuation restoration post-processor, loaded once (resident). The
    // server path originally ignored --punc-model entirely (the step lived only
    // in the CLI layer), so non-PnC backends such as parakeet RNNT came back
    // unpunctuated. Mirror the CLI exactly:
    //   1. auto-enable FireRedPunc for backends that emit no punctuation and
    //      don't toggle it natively (CTC family), and
    //   2. resolve the same --punc-model aliases via the shared resolver,
    //      supporting both FireRedPunc and the PCS model.
    // The model auto-downloads on first use.
    if (crispasr_should_auto_enable_punctuation(backend->capabilities(), params)) {
        params.punc_model = "auto";
        fprintf(stderr, "crispasr-server: auto-enabling punctuation restoration for backend '%s'\n", backend->name());
    }

    std::unique_ptr<fireredpunc_context, decltype(&fireredpunc_free)> punc_ctx(nullptr, fireredpunc_free);
    std::unique_ptr<pcs_context, decltype(&pcs_free)> pcs_ctx(nullptr, pcs_free);
    {
        const crispasr_punc_spec spec = crispasr_resolve_punc_model(params.punc_model);
        std::string path = spec.direct_path;
        if (path.empty() && !spec.cache_filename.empty())
            path = crispasr_cache::ensure_cached_file(spec.cache_filename, spec.url, params.no_prints, "crispasr[punc]",
                                                      params.cache_dir);
        if (spec.kind == crispasr_punc_kind::fireredpunc && !path.empty()) {
            punc_ctx.reset(fireredpunc_init(path.c_str()));
            if (!punc_ctx)
                fprintf(stderr, "crispasr-server: warning: failed to load punc model '%s' — continuing without\n",
                        path.c_str());
            else
                fprintf(stderr, "crispasr-server: loaded punctuation model '%s'\n", path.c_str());
        } else if (spec.kind == crispasr_punc_kind::pcs && !path.empty()) {
            pcs_ctx.reset(pcs_init(path.c_str()));
            if (!pcs_ctx)
                fprintf(stderr, "crispasr-server: warning: failed to load PCS model '%s' — continuing without\n",
                        path.c_str());
            else
                fprintf(stderr, "crispasr-server: loaded PCS model '%s'\n", path.c_str());
        }
    }

    // Truecasing post-processor, loaded once (resident) when --truecase-model is
    // set. The server path previously skipped truecasing entirely; resolve the
    // same aliases the CLI does via the shared loader.
    std::unique_ptr<truecaser_context, decltype(&truecaser_free)> tc_ctx(nullptr, truecaser_free);
    std::unique_ptr<truecaser_crf_context, decltype(&truecaser_crf_free)> tc_crf_ctx(nullptr, truecaser_crf_free);
    std::unique_ptr<truecaser_lstm_context, decltype(&truecaser_lstm_free)> tc_lstm_ctx(nullptr, truecaser_lstm_free);
    crispasr_load_truecase(params.truecase_model, params.no_prints, params.cache_dir, tc_ctx, tc_crf_ctx, tc_lstm_ctx,
                           "crispasr-server");

    Server svr;

    // CORS support — opt-in via --cors-origin. Browser clients calling our
    // /v1/* endpoints from a different origin need:
    //   1. Access-Control-Allow-Origin on every response (set on each route)
    //   2. A 204 reply to OPTIONS preflights with Allow-{Methods,Headers}
    // The pre-routing handler runs on every request before route dispatch;
    // we use it to attach the response headers and short-circuit the
    // OPTIONS preflight without touching individual routes.
    if (!params.server_cors_origin.empty()) {
        const std::string cors_origin = params.server_cors_origin;
        svr.set_pre_routing_handler([cors_origin](const Request& req, Response& res) {
            res.set_header("Access-Control-Allow-Origin", cors_origin);
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key");
            res.set_header("Access-Control-Max-Age", "86400");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return Server::HandlerResponse::Handled;
            }
            return Server::HandlerResponse::Unhandled;
        });
        fprintf(stderr, "crispasr-server: CORS enabled (Allow-Origin: %s)\n", cors_origin.c_str());
    }

    auto require_auth = [&](const Request& req, Response& res) -> bool {
        if (is_authorized(req, api_keys))
            return true;
        auth_error(res);
        return false;
    };

    // -----------------------------------------------------------------------
    // POST /inference — native CrispASR transcription endpoint
    // -----------------------------------------------------------------------
    svr.Post("/inference", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!ready.load()) {
            json_error(res, 503, "model loading");
            return;
        }
        if (!req.has_file("file")) {
            json_error(res, 400, "no 'file' field in multipart upload");
            return;
        }

        auto audio_file = req.get_file_value("file");
        fprintf(stderr, "crispasr-server: /inference received '%s' (%zu bytes)\n", audio_file.filename.c_str(),
                audio_file.content.size());

        // Per-request parameter overrides.
        whisper_params rp = params;
        rp.language = form_string(req, "language", rp.language);
        rp.source_lang = form_string(req, "source_lang", rp.source_lang);
        rp.target_lang = form_string(req, "target_lang", rp.target_lang);
        rp.translate = form_bool(req, "translate", rp.translate);
        rp.punctuation = form_bool(req, "punctuation", rp.punctuation);
        rp.diarize = form_bool(req, "diarize", rp.diarize);
        if (rp.diarize && rp.diarize_method.empty())
            rp.diarize_method = form_string(req, "diarize_method", "energy");
        rp.diarize_embedder = form_string(req, "diarize_embedder", rp.diarize_embedder);
        rp.diarize_cluster_threshold = form_float(req, "diarize_cluster_threshold", rp.diarize_cluster_threshold);
        rp.diarize_max_speakers = form_int(req, "diarize_max_speakers", rp.diarize_max_speakers);
        rp.vad = form_bool(req, "vad", rp.vad);
        rp.vad_threshold = form_float(req, "vad_threshold", rp.vad_threshold);
        rp.vad_min_speech_duration_ms = form_int(req, "vad_min_speech_duration_ms", rp.vad_min_speech_duration_ms);
        rp.vad_min_silence_duration_ms = form_int(req, "vad_min_silence_duration_ms", rp.vad_min_silence_duration_ms);
        rp.vad_max_speech_duration_s = form_float(req, "vad_max_speech_duration_s", rp.vad_max_speech_duration_s);
        rp.vad_speech_pad_ms = form_int(req, "vad_speech_pad_ms", rp.vad_speech_pad_ms);
        rp.hotwords = form_string(req, "hotwords", rp.hotwords);
        rp.hotwords_boost = form_float(req, "hotwords_boost", rp.hotwords_boost);
        rp.temperature = form_float(req, "temperature", rp.temperature);
        rp.seed = form_u64(req, "seed", rp.seed);
        rp.max_new_tokens = form_int(req, "max_new_tokens", rp.max_new_tokens);
        rp.max_new_tokens = form_int(req, "max_tokens", rp.max_new_tokens);
        rp.frequency_penalty = form_float(req, "frequency_penalty", rp.frequency_penalty);
        rp.suppress_regex = form_string(req, "suppress_regex", rp.suppress_regex);
        rp.suppress_nst = form_bool(req, "suppress_nst", rp.suppress_nst);
        rp.grammar = form_string(req, "grammar", rp.grammar);
        rp.grammar_rule = form_string(req, "grammar_rule", rp.grammar_rule);
        rp.best_of = form_int(req, "best_of", rp.best_of);
        rp.beam_size = form_int(req, "beam_size", rp.beam_size);
        rp.entropy_thold = form_float(req, "entropy_thold", rp.entropy_thold);
        rp.logprob_thold = form_float(req, "logprob_thold", rp.logprob_thold);
        rp.no_speech_thold = form_float(req, "no_speech_thold", rp.no_speech_thold);
        rp.temperature_inc = form_float(req, "temperature_inc", rp.temperature_inc);
        rp.no_fallback = form_bool(req, "no_fallback", rp.no_fallback);
        rp.detect_language = form_bool(req, "detect_language", rp.detect_language);
        rp.lid_backend = form_string(req, "lid_backend", rp.lid_backend);
        rp.lid_model = form_string(req, "lid_model", rp.lid_model);
        if (req.has_file("chunk_seconds") || req.has_param("chunk_seconds"))
            rp.chunk_seconds_explicit = true;
        rp.chunk_seconds = form_int(req, "chunk_seconds", rp.chunk_seconds);
        rp.no_timestamps = form_bool(req, "no_timestamps", rp.no_timestamps);
        rp.split_on_word = form_bool(req, "split_on_word", rp.split_on_word);
        rp.max_len = form_int(req, "max_len", rp.max_len);
        rp.split_on_punct = form_bool(req, "split_on_punct", rp.split_on_punct);
        rp.offset_t_ms = form_int(req, "offset_t_ms", rp.offset_t_ms);
        rp.duration_ms = form_int(req, "duration_ms", rp.duration_ms);
        rp.max_context = form_int(req, "max_context", rp.max_context);
        rp.audio_ctx = form_int(req, "audio_ctx", rp.audio_ctx);
        rp.word_thold = form_float(req, "word_thold", rp.word_thold);
        rp.carry_initial_prompt = form_bool(req, "carry_initial_prompt", rp.carry_initial_prompt);
        rp.chunk_overlap_seconds = form_float(req, "chunk_overlap", rp.chunk_overlap_seconds);
        rp.lcs_dedup = form_string(req, "lcs_dedup", rp.lcs_dedup);
        rp.lcs_min_length = form_int(req, "lcs_min_length", rp.lcs_min_length);
        rp.parakeet_decoder = form_string(req, "parakeet_decoder", rp.parakeet_decoder);
        rp.no_auto_aligner = form_bool(req, "no_auto_aligner", rp.no_auto_aligner);
        rp.show_alternatives = form_bool(req, "show_alternatives", rp.show_alternatives);
        rp.n_alternatives = form_int(req, "alt_n", rp.n_alternatives);

        auto result = do_transcribe(audio_file, backend.get(), model_mutex, rp, /*need_timestamps=*/true,
                                    punc_ctx.get(), pcs_ctx.get(), tc_ctx.get(), tc_crf_ctx.get(), tc_lstm_ctx.get());
        if (!result.ok) {
            json_error(res, 400, result.error);
            return;
        }

        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", result.duration_s,
                result.elapsed_s, result.elapsed_s > 0 ? result.duration_s / result.elapsed_s : 0.0);

        std::string json = crispasr_segments_to_native_json(result.segs, backend_name, result.duration_s);
        res.set_content(json, "application/json");
    });

    // -----------------------------------------------------------------------
    // POST /v1/audio/transcriptions — OpenAI-compatible endpoint
    //
    // Accepts the same multipart fields as the OpenAI API:
    //   file             (required) — audio file
    //   model            (optional) — ignored (we use the loaded model)
    //   language         (optional) — ISO-639-1 code
    //   prompt           (optional) — initial prompt / context
    //   response_format  (optional) — json|verbose_json|text|srt|vtt
    //   temperature      (optional) — sampling temperature
    //   seed             (optional) — RNG seed for sampling
    //   max_tokens       (optional) — generated-token cap for AR backends
    //   max_new_tokens   (optional) — alias for max_tokens
    //   frequency_penalty (optional) — opt-in repeated-token penalty for AR backends
    //   timestamp_granularities[] (optional) — word|segment (verbose_json)
    //
    // CrispASR extension fields (ignored by vanilla OpenAI clients):
    //   translate        (optional) — true|false, translate to English
    //   source_lang      (optional) — source language for AST backends
    //   target_lang      (optional) — target language for AST backends
    //   punctuation      (optional) — true|false, enable punctuation restoration
    //   diarize          (optional) — true|false, enable speaker diarization
    //   diarize_method   (optional) — energy|xcorr|vad-turns|pyannote|sherpa
    //   vad              (optional) — true|false, enable VAD pre-processing
    //   vad_threshold    (optional) — VAD speech probability threshold
    //   hotwords         (optional) — comma-separated hotword list
    //   hotwords_boost   (optional) — log-prob boost for hotword matches
    //   suppress_regex   (optional) — regex pattern to suppress from output
    //   grammar          (optional) — GBNF grammar for constrained decoding
    //   grammar_rule     (optional) — root rule for grammar
    //   best_of          (optional) — whisper best-of-N sampling
    //   beam_size        (optional) — whisper beam search size
    //   entropy_thold    (optional) — entropy threshold for decoder fail
    //   no_speech_thold  (optional) — no-speech probability threshold
    //   chunk_seconds    (optional) — max chunk duration for long audio
    // -----------------------------------------------------------------------
    svr.Post("/v1/audio/transcriptions", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!ready.load()) {
            json_error(res, 503, "model is still loading");
            return;
        }
        if (!req.has_file("file")) {
            json_error(res, 400, "missing required field 'file'");
            return;
        }

        auto audio_file = req.get_file_value("file");
        fprintf(stderr, "crispasr-server: /v1/audio/transcriptions received '%s' (%zu bytes)\n",
                audio_file.filename.c_str(), audio_file.content.size());

        // Parse OpenAI form fields + CrispASR extensions.
        std::string response_format = form_string(req, "response_format", "json");
        std::string language = form_string(req, "language", params.language);
        std::string prompt = form_string(req, "prompt", "");
        float temperature = form_float(req, "temperature", params.temperature);
        uint64_t seed = form_u64(req, "seed", params.seed);
        int max_new_tokens = form_int(req, "max_new_tokens", params.max_new_tokens);
        max_new_tokens = form_int(req, "max_tokens", max_new_tokens);
        float frequency_penalty = form_float(req, "frequency_penalty", params.frequency_penalty);

        // Validate response_format early.
        if (response_format != "json" && response_format != "verbose_json" && response_format != "text" &&
            response_format != "srt" && response_format != "vtt") {
            json_error(res, 400,
                       "invalid response_format '" + response_format +
                           "'; must be one of: json, verbose_json, text, srt, vtt");
            return;
        }

        // CrispASR extension fields (ignored by vanilla OpenAI clients).
        bool translate = form_bool(req, "translate", params.translate);
        std::string source_lang = form_string(req, "source_lang", params.source_lang);
        std::string target_lang = form_string(req, "target_lang", params.target_lang);
        bool punctuation = form_bool(req, "punctuation", params.punctuation);
        bool diarize = form_bool(req, "diarize", params.diarize);
        std::string diarize_method = form_string(req, "diarize_method", params.diarize_method);
        // Build per-request params.
        whisper_params rp = params;
        rp.language = language;
        rp.temperature = temperature;
        rp.seed = seed;
        rp.max_new_tokens = max_new_tokens;
        rp.frequency_penalty = frequency_penalty;
        rp.translate = translate;
        rp.source_lang = source_lang;
        rp.target_lang = target_lang;
        rp.punctuation = punctuation;
        rp.diarize = diarize;
        if (diarize && !diarize_method.empty())
            rp.diarize_method = diarize_method;
        else if (diarize && rp.diarize_method.empty())
            rp.diarize_method = "energy";
        rp.diarize_embedder = form_string(req, "diarize_embedder", rp.diarize_embedder);
        rp.diarize_cluster_threshold = form_float(req, "diarize_cluster_threshold", rp.diarize_cluster_threshold);
        rp.diarize_max_speakers = form_int(req, "diarize_max_speakers", rp.diarize_max_speakers);
        rp.vad = form_bool(req, "vad", rp.vad);
        rp.vad_threshold = form_float(req, "vad_threshold", rp.vad_threshold);
        rp.vad_min_speech_duration_ms = form_int(req, "vad_min_speech_duration_ms", rp.vad_min_speech_duration_ms);
        rp.vad_min_silence_duration_ms = form_int(req, "vad_min_silence_duration_ms", rp.vad_min_silence_duration_ms);
        rp.vad_max_speech_duration_s = form_float(req, "vad_max_speech_duration_s", rp.vad_max_speech_duration_s);
        rp.vad_speech_pad_ms = form_int(req, "vad_speech_pad_ms", rp.vad_speech_pad_ms);
        rp.hotwords = form_string(req, "hotwords", rp.hotwords);
        rp.hotwords_boost = form_float(req, "hotwords_boost", rp.hotwords_boost);
        rp.suppress_regex = form_string(req, "suppress_regex", rp.suppress_regex);
        rp.suppress_nst = form_bool(req, "suppress_nst", rp.suppress_nst);
        rp.grammar = form_string(req, "grammar", rp.grammar);
        rp.grammar_rule = form_string(req, "grammar_rule", rp.grammar_rule);
        rp.best_of = form_int(req, "best_of", rp.best_of);
        rp.beam_size = form_int(req, "beam_size", rp.beam_size);
        rp.entropy_thold = form_float(req, "entropy_thold", rp.entropy_thold);
        rp.logprob_thold = form_float(req, "logprob_thold", rp.logprob_thold);
        rp.no_speech_thold = form_float(req, "no_speech_thold", rp.no_speech_thold);
        rp.temperature_inc = form_float(req, "temperature_inc", rp.temperature_inc);
        rp.no_fallback = form_bool(req, "no_fallback", rp.no_fallback);
        rp.detect_language = form_bool(req, "detect_language", rp.detect_language);
        rp.lid_backend = form_string(req, "lid_backend", rp.lid_backend);
        rp.lid_model = form_string(req, "lid_model", rp.lid_model);
        if (req.has_file("chunk_seconds") || req.has_param("chunk_seconds"))
            rp.chunk_seconds_explicit = true;
        rp.chunk_seconds = form_int(req, "chunk_seconds", rp.chunk_seconds);
        rp.no_timestamps = form_bool(req, "no_timestamps", rp.no_timestamps);
        rp.split_on_word = form_bool(req, "split_on_word", rp.split_on_word);
        rp.max_len = form_int(req, "max_len", rp.max_len);
        rp.split_on_punct = form_bool(req, "split_on_punct", rp.split_on_punct);
        rp.offset_t_ms = form_int(req, "offset_t_ms", rp.offset_t_ms);
        rp.duration_ms = form_int(req, "duration_ms", rp.duration_ms);
        rp.max_context = form_int(req, "max_context", rp.max_context);
        rp.audio_ctx = form_int(req, "audio_ctx", rp.audio_ctx);
        rp.word_thold = form_float(req, "word_thold", rp.word_thold);
        rp.carry_initial_prompt = form_bool(req, "carry_initial_prompt", rp.carry_initial_prompt);
        rp.chunk_overlap_seconds = form_float(req, "chunk_overlap", rp.chunk_overlap_seconds);
        rp.lcs_dedup = form_string(req, "lcs_dedup", rp.lcs_dedup);
        rp.lcs_min_length = form_int(req, "lcs_min_length", rp.lcs_min_length);
        rp.parakeet_decoder = form_string(req, "parakeet_decoder", rp.parakeet_decoder);
        rp.no_auto_aligner = form_bool(req, "no_auto_aligner", rp.no_auto_aligner);
        rp.show_alternatives = form_bool(req, "show_alternatives", rp.show_alternatives);
        rp.n_alternatives = form_int(req, "alt_n", rp.n_alternatives);
        if (!prompt.empty())
            rp.prompt = prompt;

        bool stream = form_bool(req, "stream", false);

        if (stream && (backend->capabilities() & CAP_STREAMING)) {
            std::string tmp_path =
                write_temp_audio(audio_file.content.data(), audio_file.content.size(), audio_file.filename);
            if (tmp_path.empty()) {
                json_error(res, 500, "failed to create temporary file for audio");
                return;
            }

            std::vector<float> pcmf32;
            std::vector<std::vector<float>> pcmf32s;
            if (!read_audio_data(tmp_path, pcmf32, pcmf32s, rp.diarize)) {
                std::remove(tmp_path.c_str());
                json_error(res, 400, "failed to decode audio (unsupported format or corrupt file)");
                return;
            }
            std::remove(tmp_path.c_str());

            if (pcmf32.empty()) {
                json_error(res, 400, "audio file contains no samples");
                return;
            }

            // Capture large vectors by value for the async provider
            res.set_chunked_content_provider(
                "text/event-stream", [pcmf32, rp, &backend, &model_mutex](size_t /*offset*/, httplib::DataSink& sink) {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    std::string last_sent_text;

                    backend->transcribe_streaming(
                        pcmf32.data(), pcmf32.size(), 0, rp, [&](const std::string& partial, bool is_final) {
                            if (!partial.empty() || is_final) {
                                std::string diff;
                                if (partial.size() > last_sent_text.size() &&
                                    partial.compare(0, last_sent_text.size(), last_sent_text) == 0) {
                                    diff = partial.substr(last_sent_text.size());
                                } else {
                                    diff = partial;
                                }

                                if (!diff.empty()) {
                                    std::ostringstream js;
                                    js << "data: {\"text\": \"" << crispasr_json_escape(diff) << "\"}\n\n";
                                    std::string chunk = js.str();
                                    sink.write(chunk.data(), chunk.size());
                                    last_sent_text = partial;
                                }
                            }
                            if (is_final) {
                                std::string done = "data: [DONE]\n\n";
                                sink.write(done.data(), done.size());
                            }
                        });
                    sink.done();
                    return false; // return false to signal end of stream
                });
            return;
        }

        const bool need_timestamps =
            response_format == "verbose_json" || response_format == "srt" || response_format == "vtt";
        auto result = do_transcribe(audio_file, backend.get(), model_mutex, rp, need_timestamps, punc_ctx.get(),
                                    pcs_ctx.get(), tc_ctx.get(), tc_crf_ctx.get(), tc_lstm_ctx.get());
        if (!result.ok) {
            json_error(res, 400, result.error);
            return;
        }

        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime), format=%s\n",
                result.duration_s, result.elapsed_s, result.elapsed_s > 0 ? result.duration_s / result.elapsed_s : 0.0,
                response_format.c_str());

        // Format response.
        if (response_format == "text") {
            res.set_content(crispasr_segments_to_text(result.segs), "text/plain; charset=utf-8");
        } else if (response_format == "srt") {
            res.set_content(crispasr_segments_to_srt(result.segs), "application/x-subrip; charset=utf-8");
        } else if (response_format == "vtt") {
            res.set_content(crispasr_segments_to_vtt(result.segs), "text/vtt; charset=utf-8");
        } else if (response_format == "verbose_json") {
            std::string task = rp.translate ? "translate" : "transcribe";
            res.set_content(crispasr_segments_to_openai_verbose_json(result.segs, result.duration_s, result.language,
                                                                     task, temperature),
                            "application/json");
        } else {
            // Default: json — {"text": "..."}
            res.set_content(crispasr_segments_to_openai_json(result.segs), "application/json");
        }
    });

    // -----------------------------------------------------------------------
    // POST /load — hot-swap model
    // -----------------------------------------------------------------------
    svr.Post("/load", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        std::lock_guard<std::mutex> lock(model_mutex);
        ready.store(false);

        std::string new_model = form_string(req, "model");
        std::string new_backend = form_string(req, "backend");

        if (new_model.empty()) {
            ready.store(true);
            json_error(res, 400, "no 'model' field");
            return;
        }

        if (new_backend.empty())
            new_backend = crispasr_detect_backend_from_gguf(new_model);

        const bool new_model_is_auto = new_model == "auto" || new_model == "default";
        if (new_backend.empty() && new_model_is_auto)
            new_backend = "whisper";
        if (new_backend.empty()) {
            ready.store(true);
            json_error(res, 400, "cannot detect backend for model '" + new_model + "'");
            return;
        }

        const std::string resolved_model =
            crispasr_resolve_model_cli(new_model, new_backend, params.no_prints, params.cache_dir,
                                       params.auto_download || new_model_is_auto, params.model_quant);
        if (resolved_model.empty()) {
            ready.store(true);
            json_error(res, 500, "failed to resolve model '" + new_model + "' for backend '" + new_backend + "'");
            return;
        }

        whisper_params np = params;
        np.model = resolved_model;
        np.backend = new_backend;

        auto nb = crispasr_create_backend(new_backend);
        if (!nb || !nb->init(np)) {
            ready.store(true); // keep old model
            json_error(res, 500, "failed to load model '" + resolved_model + "' with backend '" + new_backend + "'");
            return;
        }

        backend = std::move(nb);
        backend_name = new_backend;
        params.model = resolved_model;
        ready.store(true);

        fprintf(stderr, "crispasr-server: hot-swapped to '%s' backend, model '%s'\n", new_backend.c_str(),
                resolved_model.c_str());
        res.set_content("{\"status\": \"ok\", \"backend\": \"" + crispasr_json_escape(new_backend) + "\"}",
                        "application/json");
    });

    // -----------------------------------------------------------------------
    // GET /health
    // -----------------------------------------------------------------------
    svr.Get("/health", [&](const Request&, Response& res) {
        if (ready.load()) {
            res.set_content("{\"status\": \"ok\", \"backend\": \"" + crispasr_json_escape(backend_name) + "\"}",
                            "application/json");
        } else {
            res.status = 503;
            res.set_content("{\"status\": \"loading\"}", "application/json");
        }
    });

    // -----------------------------------------------------------------------
    // GET /backends
    // -----------------------------------------------------------------------
    svr.Get("/backends", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        auto names = crispasr_list_backends();
        std::ostringstream js;
        js << "{\"backends\": [";
        for (size_t i = 0; i < names.size(); i++) {
            if (i)
                js << ", ";
            js << "\"" << crispasr_json_escape(names[i]) << "\"";
        }
        js << "], \"active\": \"" << crispasr_json_escape(backend_name) << "\"}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // GET /v1/models — OpenAI-compatible model list
    // -----------------------------------------------------------------------
    svr.Get("/v1/models", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        std::ostringstream js;
        js << "{\"object\": \"list\", \"data\": [{";
        js << "\"id\": \"" << crispasr_json_escape(params.model) << "\", ";
        js << "\"object\": \"model\", ";
        js << "\"owned_by\": \"crispasr\", ";
        js << "\"backend\": \"" << crispasr_json_escape(backend_name) << "\"";
        js << "}]}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // POST /v1/translate — text-to-text translation (m2m100 / CAP_TRANSLATE)
    //
    // Body: application/json
    //   {
    //     "input":       "TEXT to translate",   (required; "text" also accepted)
    //     "source_lang": "en",                  (optional; falls back to server default)
    //     "target_lang": "de",                  (required unless a server default is set)
    //     "max_tokens":  256                    (optional)
    //   }
    // Returns 200 {"text": "..."}. The HTTP analogue of the CLI `--text` mode;
    // only meaningful when a translation backend (e.g. m2m100) is loaded.
    // -----------------------------------------------------------------------
    svr.Post("/v1/translate", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!ready.load()) {
            json_error(res, 503, "model is still loading");
            return;
        }
        if (!(backend->capabilities() & CAP_TRANSLATE)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support text translation (no CAP_TRANSLATE); load a "
                           "translation backend (e.g. m2m100) via POST /load");
            return;
        }
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (...) {
            json_error(res, 400, "invalid JSON body", "invalid_json");
            return;
        }
        std::string text = body.value("input", body.value("text", std::string()));
        if (text.empty()) {
            json_error(res, 400, "missing or empty 'input' field", "missing_required_field", "input");
            return;
        }
        // Prefer the dedicated translator-stage langs, then the generic src/tgt,
        // then the per-request override — same precedence as the CLI.
        std::string src = body.value(
            "source_lang", params.translate_source_lang.empty() ? params.source_lang : params.translate_source_lang);
        std::string tgt = body.value(
            "target_lang", params.translate_target_lang.empty() ? params.target_lang : params.translate_target_lang);
        if (src.empty() || tgt.empty()) {
            json_error(res, 400, "translation requires 'source_lang' and 'target_lang'", "missing_required_field");
            return;
        }
        whisper_params rp = params;
        rp.translate_max_tokens = body.value("max_tokens", rp.translate_max_tokens);
        std::string out;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            out = backend->translate_text(text, src, tgt, rp);
        }
        if (out.empty()) {
            json_error(res, 500, "translation failed");
            return;
        }
        std::ostringstream js;
        js << "{\"text\": \"" << crispasr_json_escape(out) << "\"}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // POST /v1/audio/speech — OpenAI-compatible TTS endpoint
    //
    // Body: application/json
    //   {
    //     "input":           "TEXT to synthesize",       (required)
    //     "model":           "<model id>",               (optional, ignored — we serve the loaded one)
    //     "voice":           "<name in --voice-dir>",    (optional)
    //     "instructions":    "<voice direction prose>",  (optional, applied via params.tts_instruct)
    //     "speed":           0.25 .. 4.0,                (optional, default 1.0)
    //     "response_format": "wav"|"pcm"|"f32"|"mp3"|"opus" (optional, default "wav")
    //   }
    //
    // Returns:
    //   200 audio/wav                 — 16-bit PCM int16 RIFF, 24 kHz mono (default)
    //   200 audio/pcm                 — raw int16 LE PCM, 24 kHz mono (OpenAI spec)
    //   200 application/octet-stream  — raw float32 PCM (crispasr-specific f32)
    //
    //   400 — backend lacks CAP_TTS, missing/empty input, input too long,
    //         malformed body, unknown response_format, speed out of range
    //   500 — backend->synthesize returned empty (e.g. unknown voice)
    //   503 — model still loading
    //
    // OpenAI compatibility notes:
    //   - `model` is read but not validated — clients always send it; we
    //     serve whatever was loaded via -m or POST /load. Surfaced in
    //     the synth log line for diagnostics.
    //   - `pcm` is OpenAI's 24 kHz signed 16-bit LE mono raw byte
    //     stream (no header). `f32` is the crispasr extension that
    //     emits raw float32 for downstream DSP.
    //   - `instructions` maps to params.tts_instruct (qwen3-tts
    //     VoiceDesign). On non-VoiceDesign backends it's silently
    //     ignored — OpenAI clients don't expect it to ever 4xx.
    //   - `speed` is applied as a post-synth linear resampler. Native
    //     backend duration knobs are a future improvement.
    //
    // Voice handling: the `voice` field is passed through to
    // params.tts_voice verbatim. Each backend interprets it on its
    // own terms — qwen3-tts CustomVoice resolves it as a speaker
    // name, orpheus resolves "tara"/"leah" as presets, qwen3-tts
    // Base resolves it as a path or (with --voice-dir set) as a
    // bare name relative to the voice-dir. When "voice" is omitted
    // the request inherits whatever was set at server startup via
    // --voice / --instruct.
    // -----------------------------------------------------------------------
    svr.Post("/v1/audio/speech", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!ready.load()) {
            json_error(res, 503, "model is still loading");
            return;
        }
        if (!(backend->capabilities() & CAP_TTS)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support TTS (no CAP_TTS); load a TTS backend "
                           "(e.g. qwen3-tts, kokoro, vibevoice, orpheus) via POST /load");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (...) {
            json_error(res, 400, "invalid JSON body", "invalid_json");
            return;
        }

        std::string text = body.value("input", "");
        if (text.empty()) {
            json_error(res, 400, "missing or empty 'input' field", "missing_required_field", "input");
            return;
        }
        if (params.tts_max_input_chars > 0 && (int)text.size() > params.tts_max_input_chars) {
            json_error(res, 400,
                       "'input' length " + std::to_string(text.size()) + " exceeds the configured limit of " +
                           std::to_string(params.tts_max_input_chars) +
                           " chars; raise --tts-max-input-chars or split the input client-side",
                       "input_too_long", "input");
            return;
        }

        // Read but don't validate `model` — we serve whatever was loaded.
        // Surfaced in the log line below for diagnostics.
        std::string requested_model = body.value("model", "");

        std::string voice_name = body.value("voice", "");
        std::string consent_attestation = body.value("consent_attestation", "");
        std::string instructions = body.value("instructions", "");
        // spoken_disclaimer defaults to true; set to false to skip the
        // audible AI-disclosure prefix (watermark + C2PA remain).
        const bool spoken_disclaimer = body.value("spoken_disclaimer", true);

        // Voice-cloning consent gate: when the voice is a .wav reference
        // (voice cloning), require an explicit consent_attestation field.
        const bool is_voice_clone =
            voice_name.size() >= 4 && (voice_name.compare(voice_name.size() - 4, 4, ".wav") == 0 ||
                                       voice_name.compare(voice_name.size() - 4, 4, ".WAV") == 0);
        if (is_voice_clone && consent_attestation.empty()) {
            json_error(res, 400,
                       "voice cloning requires a 'consent_attestation' field in the request body. "
                       "This field should contain a statement attesting that you have the consent "
                       "of the speaker whose voice is being cloned, or that it is your own voice. "
                       "Example: {\"consent_attestation\": \"I have the speaker's consent\"}",
                       "consent_required", "consent_attestation");
            return;
        }
        if (is_voice_clone) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
            fprintf(stderr, "[CONSENT] ts=%s voice=%s attestation=\"%s\" spoken_disclaimer=%s\n", ts,
                    voice_name.c_str(), consent_attestation.c_str(), spoken_disclaimer ? "yes" : "no");
        }
        std::string response_format = body.value("response_format", std::string("wav"));
        if (response_format != "wav" && response_format != "pcm" && response_format != "f32" &&
            response_format != "mp3" && response_format != "opus") {
            json_error(res, 400, "response_format must be one of 'wav', 'pcm', 'f32', 'mp3', or 'opus'",
                       "unsupported_response_format", "response_format");
            return;
        }
#ifndef CRISPASR_HAVE_LAME
        if (response_format == "mp3") {
            json_error(res, 400,
                       "mp3 output requires libmp3lame; rebuild with -DCMAKE_PREFIX_PATH pointing at lame, "
                       "or install libmp3lame-dev",
                       "codec_not_available", "response_format");
            return;
        }
#endif
#ifndef CRISPASR_HAVE_OPUS
        if (response_format == "opus") {
            json_error(res, 400,
                       "opus output requires libopus; rebuild with -DCMAKE_PREFIX_PATH pointing at opus, "
                       "or install libopus-dev",
                       "codec_not_available", "response_format");
            return;
        }
#endif

        float speed = body.value("speed", 1.0f);
        if (!(speed >= 0.25f && speed <= 4.0f)) {
            json_error(res, 400, "'speed' must be between 0.25 and 4.0 (got " + std::to_string(speed) + ")",
                       "invalid_speed", "speed");
            return;
        }

        // Per-request param overrides — copy then mutate. The voice
        // string is passed through verbatim; the backend adapter owns
        // the interpretation (speaker name, preset, path, or bare name
        // relative to --voice-dir). rp.tts_voice_dir already carries
        // the server's configured dir for adapters that want to do
        // bare-name resolution.
        //
        // `instructions` maps to params.tts_instruct (qwen3-tts
        // VoiceDesign). Non-VoiceDesign backends silently ignore it;
        // we don't 4xx because OpenAI clients always include the field
        // when they're using gpt-4o-mini-tts and shouldn't see errors
        // when pointed at a base TTS server.
        whisper_params rp = params;
        if (!voice_name.empty())
            rp.tts_voice = voice_name;
        if (!instructions.empty())
            rp.tts_instruct = instructions;
        if (body.contains("seed") && body["seed"].is_number_integer())
            rp.seed = body["seed"].get<uint64_t>();
        if (body.contains("temperature") && body["temperature"].is_number())
            rp.temperature = body["temperature"].get<float>();
        if (body.contains("max_new_tokens") && body["max_new_tokens"].is_number_integer())
            rp.max_new_tokens = body["max_new_tokens"].get<int>();
        if (body.contains("frequency_penalty") && body["frequency_penalty"].is_number())
            rp.frequency_penalty = body["frequency_penalty"].get<float>();

        // 75c-opt-2: native backend duration / sampling knobs.
        // All optional; negative sentinel = "use backend default".
        if (body.contains("top_p") && body["top_p"].is_number())
            rp.tts_top_p = body["top_p"].get<float>();
        if (body.contains("min_p") && body["min_p"].is_number())
            rp.tts_min_p = body["min_p"].get<float>();
        if (body.contains("top_k") && body["top_k"].is_number_integer())
            rp.tts_top_k = body["top_k"].get<int>();
        if (body.contains("repetition_penalty") && body["repetition_penalty"].is_number())
            rp.tts_repetition_penalty = body["repetition_penalty"].get<float>();
        if (body.contains("num_candidates") && body["num_candidates"].is_number_integer())
            rp.tts_num_candidates = body["num_candidates"].get<int>();
        if (body.contains("do_sample") && body["do_sample"].is_boolean())
            rp.tts_do_sample = body["do_sample"].get<bool>() ? 1 : 0;
        if (body.contains("cfg_scale") && body["cfg_scale"].is_number())
            rp.tts_cfg_scale = body["cfg_scale"].get<float>();
        if (body.contains("num_steps") && body["num_steps"].is_number_integer())
            rp.tts_num_steps = body["num_steps"].get<int>();
        if (body.contains("noise_scale") && body["noise_scale"].is_number())
            rp.tts_noise_scale = body["noise_scale"].get<float>();
        if (body.contains("noise_w") && body["noise_w"].is_number())
            rp.tts_noise_w = body["noise_w"].get<float>();
        if (body.contains("noise_temp") && body["noise_temp"].is_number())
            rp.tts_noise_temp = body["noise_temp"].get<float>();
        if (body.contains("exaggeration") && body["exaggeration"].is_number())
            rp.tts_exaggeration = body["exaggeration"].get<float>();
        if (body.contains("speaker_id") && body["speaker_id"].is_number_integer())
            rp.tts_speaker_id = body["speaker_id"].get<int>();
        if (body.contains("max_speech_tokens") && body["max_speech_tokens"].is_number_integer())
            rp.tts_max_speech_tokens = body["max_speech_tokens"].get<int>();

        // Wire speed into params so backends with native duration control
        // (e.g. melotts length_scale, piper noise_w) can use it directly.
        // The post-synth resampler below still applies as a fallback.
        rp.tts_speed = speed;

        bool stream = body.value("stream", false);

        // Long-form chunking (PLAN §75d / issue #66): split input on
        // sentence boundaries before dispatching to the backend so each
        // synth stays inside the talker's healthy training horizon.
        // Single-sentence input becomes a 1-element vector; the per-call
        // overhead is one std::vector<float> move.
        //
        // VibeVoice Base voice cloning relies on the continuous prompt +
        // generated-text context to maintain speaker identity and prosody.
        // Chunking degrades it, so keep the request as one synthesis call.
        auto t0 = std::chrono::steady_clock::now();
        const std::vector<std::string> sentences = crispasr_tts_plan_chunks_for_backend(text, backend->name());

        // Backend-declared output rate. Most TTS backends emit 24 kHz;
        // voxcpm2-tts emits 48 kHz. Hard-coding 24 kHz here is why
        // voxcpm2 output played at half speed before this fix (#122).
        const int sr_out = backend->tts_sample_rate();

        // 75e: streaming mode — synthesize per-sentence and push each
        // chunk to the client as raw PCM via chunked transfer encoding.
        // The client receives int16 LE mono PCM at sr_out. This is the
        // same binary format as response_format=pcm, but arrives
        // incrementally. Speed resampling is still applied per-chunk.
        if (stream) {
            // Streaming only supports PCM formats (wav/pcm/f32).
            // mp3/opus require full-file encoding — reject with 400.
            if (response_format == "mp3" || response_format == "opus") {
                json_error(res, 400,
                           "streaming is not supported with response_format='" + response_format +
                               "'; use 'pcm', 'wav', or 'f32', or set stream=false",
                           "invalid_request_error", "response_format");
                return;
            }
            // Pre-compute 200ms silence gap between chunks
            const int silence_n = sr_out / 5;
            std::vector<short> silence_s16(silence_n, 0);

            // Producer/consumer streaming: a worker thread synthesizes under
            // model_mutex and pushes int16 LE PCM chunks into a bounded queue;
            // the chunked-content-provider (which httplib runs on the request
            // thread) blocks on a condition variable and writes each chunk as
            // it arrives. This lets the first chunk reach the client as soon
            // as the backend emits it — for CAP_STREAMING backends that is
            // after the first ~chunk_frames codec frames, not after the whole
            // clip — so time-to-first-audio drops to roughly one chunk.
            struct StreamQueue {
                std::mutex m;
                std::condition_variable cv;
                std::deque<std::string> q;
                bool done = false;
                bool failed = false;
                // Set by the provider when the client socket write fails (the
                // client disconnected). The worker stops enqueuing so a detached
                // synth does not keep growing the queue / holding model_mutex for
                // audio nobody is reading.
                bool cancelled = false;
            };
            auto sq = std::make_shared<StreamQueue>();

            // Backpressure cap: the worker blocks once this many chunks are
            // queued, so a slow/stalled consumer can't make memory grow unbounded
            // (the previous "bounded queue" comment was aspirational — the deque
            // had no cap). At ~30 KB/streaming-chunk this caps the queue at ~2 MB.
            constexpr size_t kMaxQueueDepth = 64;

            // Enqueue one encoded chunk with backpressure: block while the queue
            // is at capacity, but bail immediately if the client has gone. Returns
            // false when cancelled (caller should stop producing).
            auto enqueue = [sq, kMaxQueueDepth](std::string&& data) -> bool {
                std::unique_lock<std::mutex> lk(sq->m);
                sq->cv.wait(lk, [&] { return sq->q.size() < kMaxQueueDepth || sq->cancelled; });
                if (sq->cancelled)
                    return false;
                sq->q.push_back(std::move(data));
                lk.unlock();
                sq->cv.notify_one(); // wake the consumer
                return true;
            };

            const bool true_streaming = (backend->capabilities() & CAP_STREAMING) != 0;

            // Post-process one float PCM buffer (speed resample + watermark) and
            // enqueue it as int16 LE. Runs on the worker thread.
            // Captures by value only (it is copied into the detached worker
            // thread, which outlives this handler scope — a `[&]` default would
            // dangle).
            auto push_pcm = [sq, sr_out, speed, enqueue](const float* pcm, int n_samples) {
                if (!pcm || n_samples <= 0)
                    return;
                std::vector<float> chunk(pcm, pcm + n_samples);
                if (speed != 1.0f) {
                    const int in_n = (int)chunk.size();
                    const int out_n = std::max(1, (int)((float)in_n / speed));
                    std::vector<float> rs((size_t)out_n);
                    for (int j = 0; j < out_n; j++) {
                        const float s = (float)j * speed;
                        const int s0 = (int)s;
                        const int s1 = std::min(s0 + 1, in_n - 1);
                        const float frac = s - (float)s0;
                        rs[j] = chunk[s0] * (1.0f - frac) + chunk[s1] * frac;
                    }
                    chunk = std::move(rs);
                }
                crispasr_wm_dispatch::embed(chunk.data(), (int)chunk.size(), sr_out);
                enqueue(crispasr_make_pcm_int16_le(chunk.data(), (int)chunk.size()));
            };

            // Worker thread: synthesize all sentences, enqueueing chunks as
            // they are produced. Captures by value the bits it needs so it
            // outlives the handler scope (the provider keeps `sq` alive).
            std::thread worker([&backend, &model_mutex, sentences, rp, is_voice_clone, silence_s16, true_streaming,
                                push_pcm, enqueue, sq, t0]() {
                auto is_cancelled = [&] {
                    std::lock_guard<std::mutex> lk(sq->m);
                    return sq->cancelled;
                };
                // The worker is detached: any exception escaping this lambda would
                // call std::terminate and kill the whole server (and leave `done`
                // unset → a hung request). Catch everything, mark the stream
                // failed+done, and let the provider end it cleanly. (model_mutex is
                // released by RAII during unwind.)
                try {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    // TEST-ONLY (CRISPASR_TEST_STREAM_THROW): force a worker
                    // exception to verify the server survives it. Requires both the
                    // env var AND the magic input, so it can never fire in prod.
                    if (std::getenv("CRISPASR_TEST_STREAM_THROW")) {
                        for (const auto& s : sentences)
                            if (s == "__throw_test__")
                                throw std::runtime_error("injected streaming worker exception (test)");
                    }
                    if (is_voice_clone) {
                        const auto& disc = crispasr_tts_get_disclaimer(backend.get(), rp);
                        if (!disc.empty()) {
                            push_pcm(disc.data(), (int)disc.size());
                            enqueue(std::string((const char*)silence_s16.data(), silence_s16.size() * sizeof(short)));
                        }
                    }
                    for (size_t i = 0; i < sentences.size(); i++) {
                        // Client gone — stop before the next (possibly long) synth.
                        if (is_cancelled())
                            break;
                        if (true_streaming) {
                            backend->synthesize_streaming(
                                sentences[i], rp,
                                [&](const float* pcm, int n_samples, bool /*is_final*/) { push_pcm(pcm, n_samples); });
                        } else {
                            std::vector<float> chunk = backend->synthesize(sentences[i], rp);
                            if (!chunk.empty())
                                push_pcm(chunk.data(), (int)chunk.size());
                        }
                        if (i + 1 < sentences.size()) {
                            enqueue(std::string((const char*)silence_s16.data(), silence_s16.size() * sizeof(short)));
                        }
                    }
                } catch (const std::exception& e) {
                    fprintf(stderr, "crispasr-server: streaming worker exception: %s\n", e.what());
                    std::lock_guard<std::mutex> lk(sq->m);
                    sq->failed = true;
                } catch (...) {
                    fprintf(stderr, "crispasr-server: streaming worker unknown exception\n");
                    std::lock_guard<std::mutex> lk(sq->m);
                    sq->failed = true;
                }
                {
                    std::lock_guard<std::mutex> lk(sq->m);
                    sq->done = true;
                }
                sq->cv.notify_one();
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "crispasr-server: streaming synthesis finished in %.2fs\n", el);
            });
            worker.detach();

            res.set_chunked_content_provider("audio/pcm", [sq](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                std::unique_lock<std::mutex> lk(sq->m);
                sq->cv.wait(lk, [&] { return !sq->q.empty() || sq->done || sq->cancelled; });
                if (sq->q.empty() && (sq->done || sq->cancelled)) {
                    lk.unlock();
                    sink.done();
                    return true;
                }
                std::string c = std::move(sq->q.front());
                sq->q.pop_front();
                lk.unlock();
                sq->cv.notify_one(); // release the worker's backpressure wait
                if (!sink.write(c.data(), c.size())) {
                    // Client disconnected — tell the worker to stop producing.
                    std::lock_guard<std::mutex> lk2(sq->m);
                    sq->cancelled = true;
                    sq->cv.notify_all();
                    return false;
                }
                return true;
            });
            return;
        }

        std::vector<std::vector<float>> chunks;
        chunks.reserve(sentences.size());
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            for (const auto& sent : sentences) {
                std::vector<float> chunk = backend->synthesize(sent, rp);
                if (!chunk.empty())
                    chunks.push_back(std::move(chunk));
            }
        }
        // 200 ms silence between chunks (scaled to the backend rate).
        // Inaudible click suppression at boundaries; long enough that
        // the listener perceives a natural sentence pause without dragging.
        std::vector<float> pcm = crispasr_tts_concat_with_silence(chunks, sr_out / 5);
        auto t1 = std::chrono::steady_clock::now();

        if (pcm.empty()) {
            json_error(res, 500, "synthesis failed (backend returned empty audio)", "synthesis_failed");
            return;
        }

        // Prepend spoken AI-disclosure for voice-cloned requests.
        // Skipped when "spoken_disclaimer": false; watermark + C2PA
        // provenance remain regardless.
        if (is_voice_clone && spoken_disclaimer) {
            crispasr_tts_prepend_disclaimer(pcm, backend.get(), rp);
        }

        // Apply speed via linear-interpolation resampler. speed=1.0 is a
        // no-op. Quality loss vs a sinc resampler is minimal at modest
        // speeds (0.5x .. 2.0x) for speech; backends that grow native
        // duration knobs will plumb through `rp.tts_speed` directly and
        // bypass this path.
        if (speed != 1.0f) {
            const int in_n = (int)pcm.size();
            const int out_n = std::max(1, (int)((float)in_n / speed));
            std::vector<float> resampled((size_t)out_n);
            for (int i = 0; i < out_n; i++) {
                const float src = (float)i * speed;
                const int s0 = (int)src;
                const int s1 = std::min(s0 + 1, in_n - 1);
                const float frac = src - (float)s0;
                resampled[i] = pcm[s0] * (1.0f - frac) + pcm[s1] * frac;
            }
            pcm = std::move(resampled);
        }

        // Embed spread-spectrum watermark marking audio as AI-generated.
        // Applied after speed resampling so the watermark is present in
        // the final signal regardless of speed setting.
        crispasr_wm_dispatch::embed(pcm.data(), (int)pcm.size(), sr_out);

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double audio_s = (double)pcm.size() / (double)sr_out;
        fprintf(stderr,
                "crispasr-server: synthesized %.1fs audio in %.2fs (RTF=%.2f) "
                "voice='%s' speed=%.2f format=%s model='%s' chunks=%zu sr=%dHz\n",
                audio_s, elapsed_s, elapsed_s > 0 ? elapsed_s / audio_s : 0.0,
                voice_name.empty() ? "<startup>" : voice_name.c_str(), speed, response_format.c_str(),
                requested_model.empty() ? "<unset>" : requested_model.c_str(), chunks.size(), sr_out);

        if (response_format == "f32") {
            std::string buf((const char*)pcm.data(), pcm.size() * sizeof(float));
            res.set_content(std::move(buf), "application/octet-stream");
        } else if (response_format == "pcm") {
            // OpenAI's pcm: signed 16-bit LE mono raw bytes, no header.
            // Spec is 24 kHz; we emit at the backend's native rate
            // (voxcpm2 = 48 kHz) — clients must know the rate
            // out-of-band. Use response_format=wav if the client needs
            // a self-describing container.
            std::string raw = crispasr_make_pcm_int16_le(pcm.data(), (int)pcm.size());
            res.set_content(std::move(raw), "audio/pcm");
#ifdef CRISPASR_HAVE_LAME
        } else if (response_format == "mp3") {
            std::string mp3 = crispasr_encode_mp3(pcm.data(), (int)pcm.size(), sr_out);
            if (mp3.empty()) {
                json_error(res, 500, "MP3 encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(mp3), "audio/mpeg");
#endif
#ifdef CRISPASR_HAVE_OPUS
        } else if (response_format == "opus") {
            std::string opus = crispasr_encode_opus(pcm.data(), (int)pcm.size(), sr_out);
            if (opus.empty()) {
                json_error(res, 500, "Opus encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(opus), "audio/opus");
#endif
        } else {
            std::string wav = crispasr_make_wav_int16(pcm.data(), (int)pcm.size(), sr_out);
            // C2PA Content Credentials signing (when c2pa-c is available
            // and --c2pa-cert / --c2pa-key are configured)
            crispasr_c2pa_sign_wav(wav, params.c2pa_cert, params.c2pa_key);
            res.set_content(std::move(wav), "audio/wav");
        }
    });

    // -----------------------------------------------------------------------
    // POST /v1/audio/speech-to-speech — S2S (audio in → audio out)
    //
    // Content-Type: multipart/form-data
    //   file:            <audio file>                    (required)
    //   language:        "ja"|"en"|...                   (optional)
    //   response_format: "wav"|"pcm"|"f32"               (optional, default "wav")
    //
    // Returns:
    //   200 audio/wav — output audio at backend's TTS sample rate (24 kHz mono)
    //   Header X-Transcript: <URL-encoded intermediate ASR transcript>
    //   400 — backend lacks CAP_S2S, missing file
    //   500 — S2S returned empty audio
    //   503 — model still loading
    //
    // Supported backends: lfm2-audio, mini-omni2
    // -----------------------------------------------------------------------
    svr.Post("/v1/audio/speech-to-speech", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!ready.load()) {
            json_error(res, 503, "model is still loading");
            return;
        }
        if (!(backend->capabilities() & CAP_S2S)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support speech-to-speech (no CAP_S2S); "
                           "load lfm2-audio or mini-omni2 via POST /load");
            return;
        }

        if (!req.has_file("file")) {
            json_error(res, 400, "missing 'file' field (multipart audio upload)", "missing_required_field", "file");
            return;
        }
        const auto& audio_file = req.get_file_value("file");

        // Decode input audio to 16 kHz mono PCM.
        std::string tmp_path =
            write_temp_audio(audio_file.content.data(), audio_file.content.size(), audio_file.filename);
        if (tmp_path.empty()) {
            json_error(res, 500, "failed to create temporary file for audio");
            return;
        }
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(tmp_path, pcmf32, pcmf32s, false)) {
            std::remove(tmp_path.c_str());
            json_error(res, 400, "failed to decode audio (unsupported format or corrupt file)");
            return;
        }
        std::remove(tmp_path.c_str());
        if (pcmf32.empty()) {
            json_error(res, 400, "audio file contains no samples");
            return;
        }

        std::string response_format = "wav";
        if (req.has_file("response_format"))
            response_format = req.get_file_value("response_format").content;

        whisper_params rp = params;
        if (req.has_file("language"))
            rp.language = req.get_file_value("language").content;

        const int sr_out = backend->tts_sample_rate();

        // Run S2S under model lock.
        std::string transcript;
        std::vector<float> pcm;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            auto t0 = std::chrono::steady_clock::now();
            pcm = backend->speech_to_speech(pcmf32.data(), (int)pcmf32.size(), &transcript, rp);
            auto t1 = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            double in_dur_s = (double)pcmf32.size() / 16000.0;
            double out_dur_s = pcm.empty() ? 0.0 : (double)pcm.size() / (double)sr_out;
            fprintf(stderr, "crispasr-server: S2S %.1fs in → %.1fs out in %.2fs transcript='%s'\n", in_dur_s, out_dur_s,
                    elapsed_s, transcript.empty() ? "<none>" : transcript.c_str());
        }

        if (pcm.empty()) {
            json_error(res, 500, "speech-to-speech returned empty audio", "s2s_failed");
            return;
        }

        // Watermark the output.
        crispasr_wm_dispatch::embed(pcm.data(), (int)pcm.size(), sr_out);

        // Return intermediate transcript as a header.
        if (!transcript.empty()) {
            // URL-encode for safe header transport.
            std::string encoded;
            for (char c : transcript) {
                if (std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~' || c == ' ') {
                    encoded += c;
                } else {
                    char hex[4];
                    snprintf(hex, sizeof(hex), "%%%02X", (unsigned char)c);
                    encoded += hex;
                }
            }
            res.set_header("X-Transcript", encoded);
        }

        // Encode output audio.
        if (response_format == "f32") {
            std::string buf((const char*)pcm.data(), pcm.size() * sizeof(float));
            res.set_content(std::move(buf), "application/octet-stream");
        } else if (response_format == "pcm") {
            std::string raw = crispasr_make_pcm_int16_le(pcm.data(), (int)pcm.size());
            res.set_content(std::move(raw), "audio/pcm");
#ifdef CRISPASR_HAVE_LAME
        } else if (response_format == "mp3") {
            std::string mp3 = crispasr_encode_mp3(pcm.data(), (int)pcm.size(), sr_out);
            if (mp3.empty()) {
                json_error(res, 500, "MP3 encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(mp3), "audio/mpeg");
#endif
#ifdef CRISPASR_HAVE_OPUS
        } else if (response_format == "opus") {
            std::string opus = crispasr_encode_opus(pcm.data(), (int)pcm.size(), sr_out);
            if (opus.empty()) {
                json_error(res, 500, "Opus encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(opus), "audio/opus");
#endif
        } else {
            std::string wav = crispasr_make_wav_int16(pcm.data(), (int)pcm.size(), sr_out);
            crispasr_c2pa_sign_wav(wav, params.c2pa_cert, params.c2pa_key);
            res.set_content(std::move(wav), "audio/wav");
        }
    });

    // -----------------------------------------------------------------------
    // GET /v1/voices — list voices in --voice-dir (CAP_TTS only)
    // Returns: {"voices": [{"name": "<stem>", "format": "wav"|"gguf"}, ...]}
    // -----------------------------------------------------------------------
    svr.Get("/v1/voices", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!(backend->capabilities() & CAP_TTS)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support TTS (no CAP_TTS); load a TTS backend "
                           "(e.g. qwen3-tts, kokoro, vibevoice, orpheus) via POST /load");
            return;
        }

        std::ostringstream js;
        js << "{\"voices\": [";
        bool first = true;
        if (!params.tts_voice_dir.empty()) {
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(params.tts_voice_dir, ec)) {
                if (ec)
                    break;
                if (!entry.is_regular_file())
                    continue;
                const auto& path = entry.path();
                const std::string ext = path.extension().string();
                if (ext != ".wav" && ext != ".gguf")
                    continue;
                const std::string stem = path.stem().string();
                const char* fmt = (ext == ".wav") ? "wav" : "gguf";
                if (!first)
                    js << ", ";
                js << "{\"name\": \"" << crispasr_json_escape(stem) << "\", \"format\": \"" << fmt << "\"}";
                first = false;
            }
        }
        js << "]}";
        res.set_content(js.str(), "application/json");
    });

    // -----------------------------------------------------------------------
    // POST /v1/voices — upload a voice file (multipart: "voice" file + optional "name" field)
    // Returns 201 on success: {"name": "...", "format": "wav", "size_bytes": N}
    // -----------------------------------------------------------------------
    svr.Post("/v1/voices", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (params.tts_voice_dir.empty()) {
            json_error(res, 400, "server has no --voice-dir configured; cannot store voice files");
            return;
        }
        if (!req.has_file("voice")) {
            json_error(res, 400, "missing multipart 'voice' file field");
            return;
        }
        const auto& voice_file = req.get_file_value("voice");
        if (voice_file.content.size() < 44) {
            json_error(res, 400, "uploaded file is too small to be a valid audio file");
            return;
        }

        // Derive voice name: from "name" form field, or from uploaded filename stem.
        std::string voice_name;
        if (req.has_file("name")) {
            voice_name = req.get_file_value("name").content;
        } else if (!voice_file.filename.empty()) {
            voice_name = std::filesystem::path(voice_file.filename).stem().string();
        }
        if (voice_name.empty()) {
            json_error(res, 400, "cannot derive voice name; provide a 'name' form field");
            return;
        }
        // Validate: alphanumeric, dash, underscore only
        for (char c : voice_name) {
            if (!std::isalnum((unsigned char)c) && c != '-' && c != '_') {
                json_error(res, 400, "voice name must match [a-zA-Z0-9_-]+");
                return;
            }
        }

        std::string dest = params.tts_voice_dir + "/" + voice_name + ".wav";
        if (std::filesystem::exists(dest) && !req.has_param("force")) {
            json_error(res, 409, "voice '" + voice_name + "' already exists; add ?force=true to overwrite");
            return;
        }

        // Write the file
        std::ofstream out(dest, std::ios::binary);
        if (!out) {
            json_error(res, 500, "failed to write voice file");
            return;
        }
        out.write(voice_file.content.data(), (std::streamsize)voice_file.content.size());
        out.close();

        // If a "transcript" text field is provided, write the paired .txt
        // (Qwen3-TTS ICL prefill format: <name>.wav + <name>.txt).
        if (req.has_file("transcript")) {
            const auto& txt = req.get_file_value("transcript");
            std::string txt_path = params.tts_voice_dir + "/" + voice_name + ".txt";
            std::ofstream txt_out(txt_path);
            if (txt_out) {
                txt_out.write(txt.content.data(), (std::streamsize)txt.content.size());
            }
        }

        nlohmann::json resp;
        resp["name"] = voice_name;
        resp["format"] = "wav";
        resp["size_bytes"] = voice_file.content.size();
        res.status = 201;
        res.set_content(resp.dump(), "application/json");
        fprintf(stderr, "crispasr-server: uploaded voice '%s' (%zu bytes)\n", voice_name.c_str(),
                voice_file.content.size());
    });

    // -----------------------------------------------------------------------
    // DELETE /v1/voices/:name — remove a voice file from --voice-dir
    // Returns 200: {"deleted": "<name>"}
    // -----------------------------------------------------------------------
    svr.Delete(R"(/v1/voices/([a-zA-Z0-9_-]+))", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (params.tts_voice_dir.empty()) {
            json_error(res, 400, "server has no --voice-dir configured");
            return;
        }

        const std::string voice_name = req.matches[1].str();
        // Try .wav then .gguf
        std::string wav_path = params.tts_voice_dir + "/" + voice_name + ".wav";
        std::string gguf_path = params.tts_voice_dir + "/" + voice_name + ".gguf";
        bool found = false;

        if (std::filesystem::exists(wav_path)) {
            std::remove(wav_path.c_str());
            // Also remove paired .txt if present
            std::string txt_path = params.tts_voice_dir + "/" + voice_name + ".txt";
            std::remove(txt_path.c_str());
            found = true;
        }
        if (std::filesystem::exists(gguf_path)) {
            std::remove(gguf_path.c_str());
            found = true;
        }

        if (!found) {
            json_error(res, 404, "voice '" + voice_name + "' not found in --voice-dir");
            return;
        }

        nlohmann::json resp;
        resp["deleted"] = voice_name;
        res.set_content(resp.dump(), "application/json");
        fprintf(stderr, "crispasr-server: deleted voice '%s'\n", voice_name.c_str());
    });

    // -----------------------------------------------------------------------
    // POST /v1/chat/completions — OpenAI-compatible chat endpoint
    //
    // Body: application/json
    //   {
    //     "model":            "<model id>",                (optional, ignored)
    //     "messages":         [{role, content}, ...],      (required)
    //     "temperature":      0.0 .. 2.0,                  (optional, default 0.8)
    //     "top_p":            0.0 .. 1.0,                  (optional, default 0.95)
    //     "top_k":            int,                          (optional, crispasr ext.)
    //     "max_tokens":       int,                          (optional, default 256)
    //     "seed":             int,                          (optional)
    //     "stop":             ["..."] | "...",              (optional)
    //     "stream":           bool                          (optional, default false)
    //   }
    //
    // stream=false  → 200 application/json, OpenAI ChatCompletion shape
    // stream=true   → 200 text/event-stream, SSE deltas + "data: [DONE]"
    //
    // Backed by the shared crispasr_chat_* C ABI (one process-wide session
    // for params.chat_model). The session's internal mutex serialises
    // overlapping requests; concurrent requests will queue, not crash.
    // -----------------------------------------------------------------------
    std::shared_ptr<crispasr_chat_session> chat_sess(nullptr, &crispasr_chat_close);
    std::mutex chat_init_mutex;
    auto ensure_chat_session = [&]() -> crispasr_chat_session_t {
        std::lock_guard<std::mutex> g(chat_init_mutex);
        if (chat_sess) {
            return chat_sess.get();
        }
        if (params.chat_model.empty()) {
            return nullptr;
        }
        crispasr_chat_open_params op;
        crispasr_chat_open_params_default(&op);
        op.n_ctx = params.chat_n_ctx;
        op.n_gpu_layers = params.chat_n_gpu_layers;
        crispasr_chat_error err{};
        crispasr_chat_session_t s = crispasr_chat_open(params.chat_model.c_str(), &op, &err);
        if (!s) {
            fprintf(stderr, "crispasr-server: chat session open failed: %s\n", err.message);
            return nullptr;
        }
        fprintf(stderr, "crispasr-server: /v1/chat/completions ready — model '%s', template '%s', ctx %d\n",
                params.chat_model.c_str(), crispasr_chat_template_name(s), crispasr_chat_n_ctx(s));
        chat_sess.reset(s, &crispasr_chat_close);
        return s;
    };

    // Build the GenerateParams from an OpenAI-compatible JSON body.
    // The `stop` field accepts either a string or an array of strings;
    // we normalise into the (vector<string>) `stops` out-param so the
    // const char* const* the ABI takes can point at stable storage.
    auto parse_generate_params = [](const nlohmann::json& body, crispasr_chat_generate_params& gp,
                                    std::vector<std::string>& stops, std::vector<const char*>& stops_cstr) {
        crispasr_chat_generate_params_default(&gp);
        if (body.contains("temperature") && body["temperature"].is_number()) {
            gp.temperature = body["temperature"].get<float>();
        }
        if (body.contains("top_p") && body["top_p"].is_number()) {
            gp.top_p = body["top_p"].get<float>();
        }
        if (body.contains("top_k") && body["top_k"].is_number_integer()) {
            gp.top_k = body["top_k"].get<int32_t>();
        }
        if (body.contains("max_tokens") && body["max_tokens"].is_number_integer()) {
            gp.max_tokens = body["max_tokens"].get<int32_t>();
        }
        if (body.contains("seed") && body["seed"].is_number_integer()) {
            gp.seed = body["seed"].get<uint32_t>();
        }
        if (body.contains("stop")) {
            if (body["stop"].is_string()) {
                stops.push_back(body["stop"].get<std::string>());
            } else if (body["stop"].is_array()) {
                for (const auto& s : body["stop"]) {
                    if (s.is_string()) {
                        stops.push_back(s.get<std::string>());
                    }
                }
            }
        }
        stops_cstr.reserve(stops.size());
        for (const auto& s : stops) {
            stops_cstr.push_back(s.c_str());
        }
        gp.stop = stops_cstr.empty() ? nullptr : stops_cstr.data();
        gp.n_stop = stops_cstr.size();
    };

    svr.Post("/v1/chat/completions", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (params.chat_model.empty()) {
            json_error(res, 503, "chat is not enabled on this server (start with --chat-model PATH)", "chat_disabled");
            return;
        }
        crispasr_chat_session_t s = ensure_chat_session();
        if (!s) {
            json_error(res, 500, "failed to initialise chat session", "chat_init_failed");
            return;
        }

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (...) {
            json_error(res, 400, "invalid JSON body", "invalid_json");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            json_error(res, 400, "missing or empty 'messages' array", "missing_required_field", "messages");
            return;
        }

        // Marshal messages into ABI-shaped POD. Backing strings stay
        // alive for the duration of this lambda (until generate returns).
        std::vector<std::string> roles_buf, contents_buf;
        roles_buf.reserve(body["messages"].size());
        contents_buf.reserve(body["messages"].size());
        for (const auto& m : body["messages"]) {
            if (!m.contains("role") || !m.contains("content") || !m["role"].is_string()) {
                json_error(res, 400, "each message needs string 'role' and 'content'", "invalid_message");
                return;
            }
            roles_buf.push_back(m["role"].get<std::string>());
            // OpenAI accepts string OR array of content parts; we
            // collapse multimodal arrays to their text-only joined form.
            if (m["content"].is_string()) {
                contents_buf.push_back(m["content"].get<std::string>());
            } else if (m["content"].is_array()) {
                std::string joined;
                for (const auto& part : m["content"]) {
                    if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                        if (!joined.empty())
                            joined += "\n";
                        joined += part["text"].get<std::string>();
                    }
                }
                contents_buf.push_back(joined);
            } else {
                contents_buf.push_back("");
            }
        }
        std::vector<crispasr_chat_message> msgs;
        msgs.reserve(roles_buf.size());
        for (size_t i = 0; i < roles_buf.size(); ++i) {
            msgs.push_back({roles_buf[i].c_str(), contents_buf[i].c_str()});
        }

        crispasr_chat_generate_params gp;
        std::vector<std::string> stops;
        std::vector<const char*> stops_cstr;
        parse_generate_params(body, gp, stops, stops_cstr);

        const bool stream = body.value("stream", false);
        const std::string model_id = params.chat_model; // for "model" field in response
        // Each session is multi-turn safe via reset; each /v1/chat/completions
        // call is treated as a stateless conversation, so flush KV cache.
        crispasr_chat_error rerr{};
        if (crispasr_chat_reset(s, &rerr) != 0) {
            json_error(res, 500, std::string("chat reset failed: ") + rerr.message, "chat_reset_failed");
            return;
        }

        const auto now_unix = []() -> int64_t {
            return (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                .count();
        };
        const std::string created_str = std::to_string(now_unix());
        // chat-cmpl-<random>; httplib doesn't ship UUIDs so an ms-resolution
        // timestamp + thread id is enough to disambiguate concurrent calls.
        const std::string completion_id =
            "chatcmpl-" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count());

        if (!stream) {
            crispasr_chat_error gerr{};
            char* out = crispasr_chat_generate(s, msgs.data(), msgs.size(), &gp, &gerr);
            if (!out) {
                json_error(res, 500, std::string("chat generate failed: ") + gerr.message, "chat_generate_failed");
                return;
            }
            const std::string reply = out;
            crispasr_chat_string_free(out);
            std::ostringstream js;
            js << "{\"id\": \"" << completion_id << "\", " << "\"object\": \"chat.completion\", "
               << "\"created\": " << created_str << ", " << "\"model\": \"" << crispasr_json_escape(model_id) << "\", "
               << "\"choices\": [{" << "\"index\": 0, " << "\"message\": {\"role\": \"assistant\", \"content\": \""
               << crispasr_json_escape(reply) << "\"}, " << "\"finish_reason\": \"stop\"" << "}]}";
            res.set_content(js.str(), "application/json");
            return;
        }

        // ---------- streaming (SSE) ----------
        // We can't stream from the chat ABI's on_token callback directly
        // into httplib's chunked sink because httplib's content provider
        // calls our lambda *after* the response is committed, asking us
        // to fill its sink. So: run generate synchronously into a queue
        // before the chunked provider drains it. For a one-call-per-
        // request server with a session-internal mutex, this is fine
        // and avoids needing a second thread + condvar dance.
        struct sse_state {
            std::vector<std::string> deltas;
            std::string error;
        };
        sse_state state;
        crispasr_chat_error gerr{};
        auto on_tok = +[](const char* utf8, void* user) { static_cast<sse_state*>(user)->deltas.emplace_back(utf8); };
        if (crispasr_chat_generate_stream(s, msgs.data(), msgs.size(), &gp, on_tok, &state, &gerr) != 0) {
            json_error(res, 500, std::string("chat stream failed: ") + gerr.message, "chat_stream_failed");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        // Build the full SSE body and ship it as one chunked response —
        // simpler than a content-provider closure since we already have
        // every delta in hand. Clients see proper SSE framing and can
        // parse incrementally.
        std::ostringstream sse;
        for (const auto& delta : state.deltas) {
            std::ostringstream js;
            js << "{\"id\": \"" << completion_id << "\", " << "\"object\": \"chat.completion.chunk\", "
               << "\"created\": " << created_str << ", " << "\"model\": \"" << crispasr_json_escape(model_id) << "\", "
               << "\"choices\": [{\"index\": 0, " << "\"delta\": {\"content\": \"" << crispasr_json_escape(delta)
               << "\"}, " << "\"finish_reason\": null}]}";
            sse << "data: " << js.str() << "\n\n";
        }
        // Final stop chunk + DONE marker.
        {
            std::ostringstream js;
            js << "{\"id\": \"" << completion_id << "\", " << "\"object\": \"chat.completion.chunk\", "
               << "\"created\": " << created_str << ", " << "\"model\": \"" << crispasr_json_escape(model_id) << "\", "
               << "\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}";
            sse << "data: " << js.str() << "\n\n";
        }
        sse << "data: [DONE]\n\n";
        res.set_content(sse.str(), "text/event-stream");
    });

    // -----------------------------------------------------------------------
    // Catch unmatched routes. cpp-httplib invokes the error handler for any
    // 4xx/5xx response, including ones our own route handlers produced via
    // json_error() — so guard on `res.body.empty()` to avoid clobbering the
    // structured error bodies the route handlers already set. Empty body
    // here means no route matched (or a matched route forgot to call
    // set_content), so falling back to the legacy "not found" payload is
    // safe.
    svr.set_error_handler([&](const Request& req, Response& res) {
        if (!res.body.empty())
            return;
        fprintf(stderr, "crispasr-server: %s %s → %d (no matching route)\n", req.method.c_str(), req.path.c_str(),
                res.status);
        res.set_content("{\"error\": \"not found. Use POST /v1/audio/transcriptions\"}", "application/json");
    });

    // When a route handler throws, cpp-httplib turns it into a bare 500 with an
    // empty body — which the error handler above then mislabels as "not found".
    // Surface the real reason instead: log it and return a structured 500 so an
    // exception in e.g. transcription isn't silently disguised as a 404.
    svr.set_exception_handler([&](const Request& req, Response& res, std::exception_ptr ep) {
        std::string what = "unknown error";
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
        }
        fprintf(stderr, "crispasr-server: %s %s → 500 (exception: %s)\n", req.method.c_str(), req.path.c_str(),
                what.c_str());
        res.status = 500;
        json_error(res, 500, std::string("internal error: ") + what);
    });

    // Start
    // -----------------------------------------------------------------------
    const bool tts = (backend->capabilities() & CAP_TTS) != 0;
    fprintf(stderr, "\ncrispasr-server: listening on %s:%d\n", host.c_str(), port);
    fprintf(stderr, "  POST /inference                  — upload audio (native JSON)\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions    — OpenAI-compatible API\n");
    if (tts) {
        fprintf(stderr, "  POST /v1/audio/speech            — TTS (OpenAI-compatible)\n");
    }
    if (backend->capabilities() & CAP_S2S) {
        fprintf(stderr, "  POST /v1/audio/speech-to-speech  — S2S audio→audio\n");
    }
    fprintf(stderr, "  POST /load                       — hot-swap model\n");
    fprintf(stderr, "  GET  /health                     — server status\n");
    fprintf(stderr, "  GET  /backends                   — list backends\n");
    fprintf(stderr, "  GET  /v1/models                  — model info\n");
    if (tts) {
        fprintf(stderr, "  GET  /v1/voices                  — list voices in --voice-dir\n");
        fprintf(stderr, "  POST /v1/voices                  — upload voice file (multipart)\n");
        fprintf(stderr, "  DELETE /v1/voices/:name          — delete voice file\n");
        if (params.tts_voice_dir.empty()) {
            fprintf(stderr, "crispasr-server: warning: --voice-dir not set; /v1/voices will return empty "
                            "and /v1/audio/speech will reject requests with a 'voice' field\n");
        }
    }
    if (!params.chat_model.empty()) {
        fprintf(stderr, "  POST /v1/chat/completions        — text-LLM chat (model '%s')\n", params.chat_model.c_str());
    }
    if (!api_keys.empty())
        fprintf(stderr, "crispasr-server: API key authentication enabled\n");

    // Real-time WebSocket ASR streaming on a second port (--ws-port). Opt-in:
    // -1 disables (default), 0 = main port + 1, N = port N. Reuses the streaming
    // session API (crispasr_session_stream_*); clients send binary 16 kHz mono
    // float32 PCM and receive JSON partial/final text events. Whisper-only today.
    bool ws_started = false;
    bool rt_started = false;
    bool wyoming_started = false;
    if (params.wyoming_port > 0) {
        if (wyoming_start(backend.get(), model_mutex, params, params.wyoming_port) == 0) {
            wyoming_started = true;
            fprintf(stderr, "  TCP  %s:%d                  — Wyoming protocol (Home Assistant Assist STT+TTS)\n",
                    host.c_str(), params.wyoming_port);
        } else {
            fprintf(stderr, "crispasr-server: warning: failed to start Wyoming server on port %d\n",
                    params.wyoming_port);
        }
    }
    if (params.server_ws_port >= 0) {
        const int ws_port = params.server_ws_port == 0 ? port + 1 : params.server_ws_port;
        if (ws_stream_start(params.model.c_str(), ws_port, params.n_threads) == 0) {
            ws_started = true;
            fprintf(stderr, "  WS   ws://%s:%d                 — real-time streaming ASR (binary PCM in, JSON out)\n",
                    host.c_str(), ws_port);
        } else {
            fprintf(stderr, "crispasr-server: warning: failed to start WebSocket streaming on port %d\n", ws_port);
        }
        const int rt_port = ws_port + 1; // vLLM Realtime API on ws_port + 1
        if (realtime_server_start(backend.get(), model_mutex, params, rt_port) == 0) {
            rt_started = true;
            fprintf(stderr, "  WS   ws://%s:%d/v1/realtime     — vLLM Realtime API (JSON WebSocket)\n", host.c_str(),
                    rt_port);
        } else {
            fprintf(stderr, "crispasr-server: warning: failed to start vLLM Realtime API on port %d\n", rt_port);
        }
    }
    fprintf(stderr, "\n");

    svr.listen(host, port);

    if (wyoming_started)
        wyoming_stop();
    if (ws_started)
        ws_stream_stop();
    if (rt_started)
        realtime_server_stop();
    crispasr_vad_free_cache();
    crispasr_lid_free_cache();

    return 0;
}
