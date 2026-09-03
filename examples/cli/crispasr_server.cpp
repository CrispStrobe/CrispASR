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
//   POST /v1/audio/separation          — source separation (--separate-model; §381)
//   POST /load                        — hot-swap model
//   GET  /health                      — server status
//   GET  /progress                    — poll the active job's progress (#408)
//   GET  /backends                    — list available backends
//   GET  /v1/models                   — OpenAI-compatible model list
//   GET  /v1/voices                   — list voices in --voice-dir (CAP_TTS only)
//        /voices, /v1/audio/voices    — aliases for llama-swap compatibility (#264)
//   POST /v1/voices                   — upload voice file (multipart, CAP_TTS only)
//   DELETE /v1/voices/:name           — delete voice file (CAP_TTS only)
//
// Adapted from examples/server/server.cpp for multi-backend support.

#include "crispasr_backend.h"
#include "core/asr_sensitivity.h"  // §W7 --sensitivity presets over the HTTP API
#include "core/ggml_cpu_backend.h" // CPU-backend probe — refuse to start when no module loads (#405)
#include "crispasr_diarize_cli.h"
#include "tiron_link.h" // #295: tiron cross-window speaker linking (shared with the CLI)
#include "crispasr_gap_fill.h"
#include "crispasr_speaker_embedder.h"
#include "crispasr_lid.h"
#include "crispasr_lid_cli.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_vad_cli.h"
#include "crispasr_cpu_isa.h" // #261: CPU instruction-set self-check (soft-degrade VAD/diarize)
#include "crispasr_aligner_cli.h"
#include "whisper_params.h"
#include "crispasr_strict.h"             // #311: shared strict-pipeline requirements (parity with the CLI)
#include "fireredpunc.h"                 // server-mode punctuation restoration (--punc-model)
#include "pcs.h"                         // PCS (punctuation + caps + segmentation) model
#include "crispasr_cache.h"              // ensure_cached_file() for resolving the punc model
#include "crispasr_punc_loader.h"        // shared --punc-model alias resolution (CLI parity)
#include "crispasr_truecase_loader.h"    // shared --truecase-model resolution + apply (CLI parity)
#include "crispasr_phonemes_policy.h"    // #316
#include "crispasr_punctuation_policy.h" // crispasr_should_auto_enable_punctuation()

#include "common-crispasr.h"           // read_audio_data
#include "core/separation_io.h"        // §381: separation result view + stem-to-WAV
#include "core/gguf_loader.h"          // §381: arch detection for --separate-model
#include "htdemucs.h"                  // §381: htdemucs separation backend
#include "mel_band_roformer.h"         // §381: mel-band-roformer separation backend
#include "crispasr_chat.h"             // /v1/chat/completions
#include "../server/ws_stream.h"       // real-time WebSocket ASR streaming (--ws-port)
#include "../server/realtime_server.h" // vLLM Realtime API
#include "wyoming.h"                   // Wyoming protocol for Home Assistant Assist (--wyoming-port)
#include "core/audio_window.h"
#include "core/crispasr_c2pa.h"
#include "crispasr_marking_policy.h" // #312: who may skip the spoken AI-disclaimer
#include "crispasr_tts_chunking.h"
#include "crispasr_tts_disclaimer.h"
#include "crispasr_consent_record.h"
#include "crispasr_voice_clone_policy.h"
#include "crispasr_voice_provenance.h"
#include "core/crispasr_watermark.h"
#include "core/omnivoice_instruct.h" // #13273: validate `instructions` at the edge
#include "crispasr_watermark_dispatch.h"
#include "core/crispasr_wav_writer.h"
#include "core/worker_pool.h"     // improvements Phase 4b: concurrent ASR worker pool
#include "crispasr_mp3_writer.h"  // MP3 output via in-tree glint encoder
#include "crispasr_aac_writer.h"  // AAC-LC (ADTS) output via in-tree glint encoder
#include "crispasr_opus_writer.h" // Ogg Opus output via in-tree glint encoder
#include "../server/httplib.h"
#include "../json.hpp"

#include <algorithm> // std::any_of — reaches us transitively today, which is
                     // exactly how #355 broke the Windows build
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
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

// 75e: optional Opus output encoding (compile-time gated). MP3 is
// always available via the in-tree glint encoder (crispasr_mp3_writer.h,
// with an optional libmp3lame fallback behind CRISPASR_HAVE_LAME).
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
// Sanitise an untrusted, request-supplied string before it goes into a stderr
// log line: cap the length and replace control bytes (incl. CR/LF and terminal
// escape sequences) so a crafted filename / voice / attestation can't forge log
// lines or inject escapes into an operator's terminal.
static std::string log_sanitize(const std::string& s, size_t cap = 256) {
    std::string o;
    o.reserve(s.size() < cap ? s.size() : cap);
    for (size_t i = 0; i < s.size() && i < cap; i++) {
        unsigned char c = (unsigned char)s[i];
        o += (c < 0x20 || c == 0x7f) ? '?' : (char)c;
    }
    if (s.size() > cap)
        o += "...";
    return o;
}

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
// Pre-dispatch language validation, mirroring the CLI's 0face104 fix plus the
// sole-language guard. The server parses its own request fields, so it carried
// the same hole the CLI had: an unknown `language` reached the backend and
// transcribed the wrong thing at HTTP 200. Whisper's 100-entry table governs
// only whisper (omnivoice alone legitimately takes fil/nan/arb/pes, so the
// check is gated on the effective backend, which at request time is the
// backend the server loaded); monolingual backends reject any other language
// outright, compared through whisper_lang_id() so "german" is caught like
// "de". Returns an error message, empty when the request is satisfiable.
static std::string validate_request_language(const std::string& backend, const std::string& lang) {
    if (lang.empty() || lang == "auto")
        return "";
    const bool is_whisper = backend.empty() || backend == "whisper";
    if (is_whisper && whisper_lang_id(lang.c_str()) == -1)
        return "unknown language '" + lang + "'";
    const char* sole = nullptr;
    if (backend == "moonshine" || backend == "moonshine-streaming")
        sole = "en";
    else if (backend == "gigaam")
        sole = "ru";
    if (sole) {
        const int want = whisper_lang_id(lang.c_str());
        const int have = whisper_lang_id(sole);
        const bool same = (want != -1 && have != -1) ? (want == have) : (lang == sole);
        if (!same)
            return "backend '" + backend + "' is " + std::string(sole) + "-only and cannot transcribe '" + lang +
                   "'; use '" + sole + "' or 'auto'";
    }
    return "";
}

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
    crispasr_ctc_logits logits;
    std::string language;
    double duration_s = 0.0;
    double elapsed_s = 0.0;
    // #227: serialized VAD/chunk boundaries, when the request asked for them
    // via `vad_export=true`. Empty otherwise.
    std::string vad_segments_json;
};

static void append_ctc_logits(crispasr_ctc_logits& dst, const crispasr_ctc_logits* src) {
    if (!src || src->data.empty() || src->n_frames <= 0 || src->n_vocab <= 0)
        return;
    if (dst.n_vocab == 0) {
        dst.n_vocab = src->n_vocab;
        dst.normalization = src->normalization;
        dst.vocab = src->vocab;
    }
    if (dst.n_vocab != src->n_vocab)
        return;
    dst.data.insert(dst.data.end(), src->data.begin(), src->data.end());
    dst.n_frames += src->n_frames;
}

static std::string add_ctc_logits_to_json(const std::string& base, const crispasr_ctc_logits& logits) {
    if (logits.data.empty())
        return base;
    try {
        auto obj = nlohmann::json::parse(base);
        obj["ctc_logits"] = nlohmann::json::parse(crispasr_ctc_logits_to_json(logits));
        return obj.dump();
    } catch (...) {
        return base;
    }
}

// #227: splice the serialized VAD/chunk boundaries into a JSON response under
// `vad_segments`, so a client can feed them back via `vad_import` on a later
// request (e.g. same audio, different backend) and skip re-running VAD.
static std::string add_vad_segments_to_json(const std::string& base, const std::string& vad_segments_json) {
    if (vad_segments_json.empty())
        return base;
    try {
        auto obj = nlohmann::json::parse(base);
        obj["vad_segments"] = nlohmann::json::parse(vad_segments_json);
        return obj.dump();
    } catch (...) {
        return base;
    }
}

// Per-thread breadcrumb of the current processing stage (#261). Synchronous
// fault signals (SIGILL/SIGSEGV/…) are delivered to the faulting thread, so the
// fatal-signal handler runs on that thread and reads its own TLS — the crashing
// stage. Reading a const char* is async-signal-safe. See the fatal-signal
// handler further down for how this is used.
static thread_local const char* g_current_stage = nullptr;

struct stage_scope {
    const char* prev;
    explicit stage_scope(const char* s) : prev(g_current_stage) { g_current_stage = s; }
    ~stage_scope() { g_current_stage = prev; }
    stage_scope(const stage_scope&) = delete;
    stage_scope& operator=(const stage_scope&) = delete;
};

// Load audio from a multipart file upload, transcribe it, return result.
// Acquires model_mutex internally.

// GET /progress state (#408). busy = any transcription job in flight (primary
// or pooled worker); progress = chunk-loop position of the most recent writer,
// 0..100, -1 when idle. A progress_scope is constructed AFTER the job's model
// mutex is acquired, so a request queued behind a running job neither resets
// the running job's progress nor flips the server idle when it was first to
// finish. With --server-workers > 1, concurrent pure-ASR jobs share the one
// progress slot (last writer wins); per-request scoping would need job ids,
// which the polling client in #408 does not need yet — the busy count stays
// honest either way.
static std::atomic<int> g_server_progress{-1};
static std::atomic<int> g_server_active{0};
struct progress_scope {
    progress_scope() {
        g_server_active.fetch_add(1, std::memory_order_relaxed);
        g_server_progress.store(0, std::memory_order_relaxed);
    }
    ~progress_scope() {
        if (g_server_active.fetch_sub(1, std::memory_order_acq_rel) == 1)
            g_server_progress.store(-1, std::memory_order_relaxed);
    }
    progress_scope(const progress_scope&) = delete;
    progress_scope& operator=(const progress_scope&) = delete;
};
static transcription_result do_transcribe(const httplib::MultipartFormData& audio_file, CrispasrBackend* backend,
                                          std::mutex& model_mutex, whisper_params rp, bool need_timestamps,
                                          fireredpunc_context* punc_ctx = nullptr, pcs_context* pcs_ctx = nullptr,
                                          truecaser_context* tc_ctx = nullptr,
                                          truecaser_crf_context* tc_crf_ctx = nullptr,
                                          truecaser_lstm_context* tc_lstm_ctx = nullptr) {
    transcription_result result;
    result.language = rp.language;

    if (rp.verbose)
        fprintf(stderr, "crispasr-server: processing '%s' (%zu bytes)\n", log_sanitize(audio_file.filename).c_str(),
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

    // #311: strict pipeline — resolve the required stages once. A required
    // punctuation model that the server never loaded is a hard error (the
    // server loads punc at startup, so this catches "required but absent").
    const crispasr_strict_reqs strict = crispasr_compute_strict_reqs(rp);
    bool aligner_load_failed = false; // #311: OR-accumulated across forced-aligner calls
    if (strict.punc && !punc_ctx && !pcs_ctx) {
        result.error = "punctuation required (require_punctuation/strict_pipeline) but the server has no punctuation "
                       "model loaded — start crispasr-server with --punc-model";
        return result;
    }

    // #91: offset_t_ms / duration_ms request params — restrict processing to a
    // time window of the upload (mirrors the CLI dispatcher crispasr_run.cpp,
    // which the server path previously skipped). Applied to the decoded 16 kHz
    // PCM before slicing so VAD/chunking/transcribe all operate on the window;
    // reported segment timestamps are shifted back by the offset at the end so
    // they stay in original-audio time. No backend adapter applies the offset
    // itself (only the legacy whisper CLI path did), so this is the sole
    // application — no double-shift.
    {
        const auto win = core_audio_window::compute((int64_t)pcmf32.size(), rp.offset_t_ms, rp.duration_ms, 16000);
        if (win.active && win.past_end) {
            // Window starts past end-of-audio: nothing to transcribe.
            result.ok = true;
            return result;
        }
        if (win.active) {
            core_audio_window::trim(pcmf32, win);
            for (auto& ch : pcmf32s)
                core_audio_window::trim(ch, win);
        }
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
    // Issue #89: backends with a bounded safe decode window (parakeet-ja,
    // ~12 s) get their slices capped — with VAD this re-splits merged
    // continuous-speech slices at energy minima, without it the fixed
    // chunking drops from 30 s to the cap. Mirrors crispasr_run.cpp.
    const int vad_cap = backend->vad_slice_cap_seconds();
    if (vad_cap > 0 && !rp.chunk_seconds_explicit &&
        (effective_chunk_seconds == 0 || effective_chunk_seconds > vad_cap)) {
        effective_chunk_seconds = vad_cap;
    }
    // Issue #257: backends that chunk internally (parakeet/canary — full-
    // attention FastConformer) are corrupted by the dispatcher's per-slice
    // transcribe + merge. Hand them the whole clip so their coherent internal
    // decode runs — and, for an explicit --chunk-seconds request, emits
    // ~N-second segments (the CLI adapter's issue #257 path) instead of one
    // blob. Mirrors the CLI dispatcher (crispasr_run.cpp
    // backend_self_chunks_on_explicit + CAP_INTERNAL_CHUNKING gate). VAD still
    // provides silence-bounded slices when the caller requested it.
    if ((backend->capabilities() & CAP_INTERNAL_CHUNKING) && !rp.vad) {
        effective_chunk_seconds = 0;
    }
    // #261: Silero/MarbleNet VAD runs on the CPU ggml backend inside slicing —
    // breadcrumb it so a native-instruction fault (SIGILL) is attributed
    // correctly by the fatal-signal handler.
    std::vector<crispasr_audio_slice> slices;
    if (!rp.vad_import_json.empty()) {
        // #227: caller supplied boundaries from an earlier response's
        // `vad_segments` — reuse them instead of running VAD again. Mirrors the
        // CLI's --vad-import (crispasr_run.cpp); same wire format.
        int imported_sr = 0;
        float imported_chunk = 0.0f;
        bool imported_raw = false;
        if (!crispasr_parse_vad_slices(rp.vad_import_json, slices, &imported_sr, &imported_chunk, &imported_raw)) {
            result.error = "malformed vad_import (expected the vad_segments object from a vad_export response)";
            return result;
        }
        // #227: same gate as the CLI. The boundaries are CHUNK boundaries, so a
        // payload exported under a different chunk length does not carry over --
        // reusing it silently re-chunks the audio wrongly. Rejecting here keeps
        // the server and CLI from disagreeing about what an import means.
        // #227: same policy as the CLI -- WARN by default, refuse only when the
        // caller opts in with vad_import_strict. Compare the REQUESTED chunk on
        // both sides: the exporter runs before backend init and cannot know the
        // effective value (see crispasr_run.cpp).
        const float requested_chunk = rp.chunk_seconds > 0 ? (float)rp.chunk_seconds : 30.0f;
        if (!imported_raw && crispasr_vad_chunk_mismatch(imported_chunk, requested_chunk)) {
            if (rp.vad_import_strict) {
                result.ok = false;
                result.error = "vad_import was exported at chunk_seconds " + std::to_string(imported_chunk) +
                               " but this request uses " + std::to_string(requested_chunk) +
                               "; re-export, match the chunk length, or drop vad_import_strict";
                return result;
            }
            fprintf(stderr,
                    "crispasr-server: warning: vad_import exported at chunk_seconds %.2f, request uses %.2f — "
                    "chunking will not match\n",
                    imported_chunk, requested_chunk);
        }
        if (imported_sr > 0 && imported_sr != SR) {
            for (auto& s : slices) {
                s.start = (int)((int64_t)s.start * SR / imported_sr);
                s.end = (int)((int64_t)s.end * SR / imported_sr);
            }
        }
        // Clamp to this request's buffer and drop out-of-range slices, so a
        // stale or hand-edited boundary set can't index out of bounds. Note the
        // boundaries are interpreted in the buffer actually being processed —
        // i.e. after any offset_t_ms/duration_ms window was applied above.
        std::vector<crispasr_audio_slice> clean;
        clean.reserve(slices.size());
        for (auto& s : slices) {
            if (s.start < 0)
                s.start = 0;
            if (s.end > n_samples)
                s.end = n_samples;
            if (s.end > s.start)
                clean.push_back(s);
        }
        slices = std::move(clean);
        // #227: a raw-segment payload carries no chunking; re-chunk it for THIS
        // request exactly as a fresh VAD pass would, so it is reusable across
        // requests with different chunk lengths (mirrors the CLI).
        if (imported_raw && effective_chunk_seconds > 0)
            slices = crispasr_rechunk_slices(slices, pcmf32.data(), n_samples, SR, effective_chunk_seconds);
    } else {
        // #261: Silero/MarbleNet VAD runs on the CPU ggml backend inside slicing —
        // breadcrumb it so a native-instruction fault (SIGILL) is attributed
        // correctly by the fatal-signal handler.
        stage_scope _stg(rp.vad ? "vad (silero/marblenet CPU inference)" : "audio slicing");
        bool vad_load_failed = false;
        slices =
            crispasr_compute_audio_slices(pcmf32.data(), n_samples, SR, effective_chunk_seconds, rp, &vad_load_failed);
        // #311: a required VAD model that failed to load is an error, not a
        // silent fall-through to fixed chunking (checked before the empty-slice
        // no-speech success path below).
        if (vad_load_failed && strict.vad) {
            result.error = "required VAD model '" + rp.vad_model +
                           "' failed to load (require_vad/strict_pipeline) — refusing to fall back to fixed chunking";
            return result;
        }
    }

    // #227: hand the boundaries back when asked, so the next request (same
    // audio, different backend) can skip VAD via `vad_import`.
    if (rp.vad_export_inline) {
        // A raw export makes the payload reusable across requests with different
        // chunk lengths; the default (chunk boundaries) matches the older
        // response shape, so this is additive.
        const float exp_chunk = rp.vad_export_raw ? 0.0f : (rp.chunk_seconds > 0 ? (float)rp.chunk_seconds : 30.0f);
        result.vad_segments_json = crispasr_serialize_vad_slices(slices, SR, exp_chunk, rp.vad_export_raw);
    }

    if (slices.empty()) {
        result.ok = true;
        return result;
    }

    // Keep this fact before disabling VAD below; it selects speech-only input
    // for the request-level LID pass.
    const bool have_vad_slices = rp.vad || !rp.vad_import_json.empty();

    // VAD (if any) already ran above — disable it for the backend so
    // whisper_full doesn't re-run Silero on every slice (#132).
    rp.vad = false;
    rp.vad_model.clear();

    {
        std::lock_guard<std::mutex> lock(model_mutex);
        progress_scope _progress; // #408: GET /progress reports this job now
        auto t0 = std::chrono::steady_clock::now();

        // Match file-mode `-l auto`: run LID once per uploaded audio sample
        // before dispatching chunks to backends that need explicit language.
        if (want_auto_lang && !has_native_lid && !lid_disabled) {
            std::vector<float> lid_speech;
            const float* lid_samples = pcmf32.data();
            int lid_n_samples = (int)pcmf32.size();
            if (have_vad_slices) {
                lid_speech = crispasr_lid_speech_prefix(pcmf32, slices);
                if (!lid_speech.empty()) {
                    lid_samples = lid_speech.data();
                    lid_n_samples = (int)lid_speech.size();
                }
            }
            const bool used_speech_lid = !lid_speech.empty();
            crispasr_lid_result lid;
            // Backend self-probe first (see crispasr_lid_cli.h); external LID
            // only when it declines.
            const bool probed = crispasr_backend_probe_language(*backend, lid_samples, lid_n_samples, rp, lid);
            if (probed || crispasr_detect_language_cli(lid_samples, lid_n_samples, rp, lid)) {
                rp.language = lid.lang_code;
                if (rp.source_lang.empty() || rp.source_lang == "auto")
                    rp.source_lang = lid.lang_code;
                if (!rp.no_prints) {
                    fprintf(stderr, "crispasr-server: LID -> language = '%s' (%s, p=%.3f)\n", lid.lang_code.c_str(),
                            lid.source.c_str(), lid.confidence);
                    if (used_speech_lid)
                        fprintf(stderr, "crispasr-server: LID input = %.2fs speech from %zu VAD slice(s)\n",
                                (double)lid_n_samples / 16000.0, slices.size());
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
            // #408: claim the chunk when it STARTS — (i+1) here would read 100
            // while the last chunk is still transcribing.
            g_server_progress = (int)(i * 100 / slices.size());
            auto tc0 = std::chrono::steady_clock::now();
            auto segs = backend->transcribe(pcmf32.data() + sl.start, sl.end - sl.start, sl.t0_cs, rp);
            if (rp.return_logits)
                append_ctc_logits(result.logits, backend->last_ctc_logits());

            // Issue #89 gap-fill second pass (bounded-window backends only) —
            // same policy as the CLI dispatcher (crispasr_gap_fill.h).
            if (vad_cap > 0) {
                const char* gf = getenv("CRISPASR_GAP_FILL");
                if (!gf || atoi(gf) != 0)
                    crispasr_gap_fill_slice(*backend, rp, pcmf32.data(), n_samples, SR, sl, segs);
            }

            if (want_align) {
                for (auto& seg : segs) {
                    if (!seg.words.empty() && !rp.force_aligner)
                        continue;
                    bool load_failed = false;
                    auto words = crispasr_ctc_align(rp.aligner_model, seg.text, pcmf32.data() + sl.start,
                                                    sl.end - sl.start, sl.t0_cs, rp.n_threads, &load_failed);
                    aligner_load_failed = aligner_load_failed || load_failed;
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

        // #408: all chunks decoded; the job stays busy at 100 through the
        // post-steps below (diarization/punctuation/truecasing) until the
        // scope exits.
        g_server_progress = 100;

        // Issue #356: same guard as the CLI's merge_segments. The server has no
        // merge step of its own — it appends each slice's segments straight into
        // the response — so the check belongs right after that loop.
        crispasr_warn_if_segments_backward(result.segs, "slice append");

        // Tiron (#295): if the transcript carries <|speakerN|> markers, link them
        // to global SPEAKER_NN with the library-shared linker (same as the CLI)
        // and skip the generic diarizer. Opt-in via diarize / diarize_embedder.
        bool tiron_handled = false;
        if (!result.segs.empty()) {
            std::vector<TironTranscriptSeg> ts(result.segs.size());
            for (size_t i = 0; i < result.segs.size(); i++) {
                ts[i].text = result.segs[i].text;
                ts[i].t0_cs = result.segs[i].t0;
                ts[i].t1_cs = result.segs[i].t1;
                ts[i].chunk_id = result.segs[i].chunk_id;
            }
            const std::string spec =
                !rp.diarize_embedder.empty() ? rp.diarize_embedder : (rp.diarize ? std::string("auto") : std::string());
            const int n_spk = crispasr_tiron_link_transcript(ts, pcmf32.data(), n_samples, spec.c_str(), rp.n_threads,
                                                             rp.cache_dir.c_str());
            if (n_spk >= 0) {
                tiron_handled = true;
                std::vector<crispasr_segment> kept;
                kept.reserve(result.segs.size());
                for (size_t i = 0; i < result.segs.size(); i++) {
                    if (ts[i].drop)
                        continue;
                    result.segs[i].text = ts[i].text;
                    if (!ts[i].speaker.empty())
                        result.segs[i].speaker = ts[i].speaker;
                    kept.push_back(std::move(result.segs[i]));
                }
                if (n_spk > 0)
                    result.segs = std::move(kept);
            }
        }

        // Diarization post-step (#143): assign speaker labels to segments.
        // Mirrors the CLI path in crispasr_run.cpp:732-743.
        if (!tiron_handled && rp.diarize && !result.segs.empty()) {
            // #261: diarization (pyannote-seg / sherpa / embedding) runs on the
            // CPU ggml backend. Breadcrumb the stage for the fatal-signal
            // handler, and wrap the whole step so a throwing failure (e.g.
            // std::bad_alloc on a very long clip) degrades to "transcription
            // without speaker labels" instead of failing the request — the
            // server must not lose a good ASR result over an optional post-step.
            stage_scope _stg("diarization (cpu inference)");
            try {
                const bool have_stereo = pcmf32s.size() == 2 && !pcmf32s[0].empty() && !pcmf32s[1].empty();

                // #324: foxnose diarizes in one global pass after transcription
                // (below) so speaker identities are consistent across slices —
                // the per-slice path must stand down. Mirrors crispasr_run.cpp.
                // Without this the server clustered each VAD slice on its own,
                // restarting the numbering every few seconds and reloading the
                // WeSpeaker embedder once per slice.
                if (rp.diarize_embedder_is_foxnose())
                    rp.diarize_foxnose_global = true;

                // Pre-compute global caches for cross-slice consistency.
                CrispasrPyannoteCache pyannote_cache;
                if (rp.diarize_method == "pyannote" && !pcmf32.empty()) {
                    crispasr_compute_pyannote_cache(pcmf32.data(), n_samples, rp, pyannote_cache);
                }
                CrispasrSherpaCache sherpa_cache;
                if ((rp.diarize_method == "sherpa" || rp.diarize_method == "sherpa-onnx" ||
                     rp.diarize_method == "ecapa") &&
                    !pcmf32.empty()) {
                    crispasr_compute_sherpa_cache(pcmf32.data(), n_samples, rp, sherpa_cache);
                }
                const CrispasrPyannoteCache* pya_ptr = pyannote_cache.valid() ? &pyannote_cache : nullptr;
                const CrispasrSherpaCache* shp_ptr = sherpa_cache.valid() ? &sherpa_cache : nullptr;

                // Apply diarize per-slice. The CLI does this inside its slice
                // loop, where the slice owns its segment vector; the server
                // transcribes first, so it re-walks the merged list and hands
                // each slice its own sub-range. #324: the turn splitter GROWS
                // that sub-range, so the shared helper rebuilds the list
                // instead of copying a fixed count back in place — writing back
                // only the original count dropped every sub-segment past it,
                // i.e. exactly the text around each speaker change.
                crispasr_diarize_merged_by_slice(
                    result.segs, slices,
                    [&](const crispasr_audio_slice& sl, std::vector<crispasr_segment>& slice_segs) {
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
                    });

                // #324: foxnose diarizes in ONE global pass over the whole
                // audio (the per-slice call above stood down via
                // diarize_foxnose_global), so its speaker numbering is
                // consistent across slices instead of restarting at 0 in every
                // VAD slice. Mirrors crispasr_apply_global_speaker_stages in
                // the CLI runner; foxnose owns the labels, so the TitaNet remap
                // below must not re-run.
                const bool foxnose_global_ran = crispasr_apply_foxnose_global(result.segs, pcmf32, rp);

                // Global embedding-based re-clustering (issue #107 P3).
                if (!foxnose_global_ran && !rp.diarize_embedder.empty() && !pcmf32.empty() &&
                    !rp.diarize_embedder_is_foxnose()) {
                    auto embedder = crispasr_make_speaker_embedder(rp.diarize_embedder, rp.n_threads, rp.cache_dir);
                    if (embedder) {
                        crispasr_remap_speakers_via_embeddings(result.segs, pcmf32.data(), n_samples, embedder.get(),
                                                               rp);
                    }
                }
            } catch (const std::exception& e) {
                fprintf(stderr,
                        "crispasr-server: warning: diarization failed (%s) — returning transcription "
                        "without speaker labels\n",
                        e.what());
            } catch (...) {
                fprintf(stderr, "crispasr-server: warning: diarization failed (unknown error) — returning "
                                "transcription without speaker labels\n");
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

    // #91: shift reported timestamps back into original-audio time when a
    // --offset-t window trimmed the buffer above. Done here, after all
    // slice-relative processing (diarize re-walk matches segments by the
    // unshifted slice t0_cs).
    if (rp.offset_t_ms > 0) {
        const int64_t off_cs = (int64_t)rp.offset_t_ms / 10;
        for (auto& seg : result.segs) {
            seg.t0 += off_cs;
            seg.t1 += off_cs;
            for (auto& w : seg.words) {
                w.t0 += off_cs;
                w.t1 += off_cs;
            }
            for (auto& tok : seg.tokens) {
                if (tok.t0 >= 0)
                    tok.t0 += off_cs;
                if (tok.t1 >= 0)
                    tok.t1 += off_cs;
            }
        }
    }

    // #311: word-timestamp requirement — an explicitly requested aligner must
    // have loaded, and every non-empty segment must carry word timestamps
    // (native or aligned). Parity with the CLI's crispasr_strict_check_words.
    if (strict.words) {
        if (aligner_load_failed) {
            result.error = "the explicitly requested forced aligner failed to load "
                           "(require_word_timestamps/strict_pipeline)";
            return result;
        }
        const int missing = crispasr_count_missing_word_ts(result.segs);
        if (missing > 0) {
            result.error = "word timestamps required (require_word_timestamps/strict_pipeline) but " +
                           std::to_string(missing) +
                           " non-empty segment(s) have none — the aligner produced no words, or the backend emitted "
                           "no native word timing";
            return result;
        }
    }

    result.ok = true;
    return result;
}

// crispasr_make_wav_int16 lives in crispasr_wav_writer.h so the unit
// tests can exercise it without linking the server translation unit.

// ---------------------------------------------------------------------------
// 75e: MP3 / Opus encoding helpers. MP3 (crispasr_make_mp3, glint) lives
// in crispasr_mp3_writer.h and is always available; Opus stays gated on
// libopus.
// ---------------------------------------------------------------------------

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

// Encode TTS PCM for response_format=opus. Default: a real, playable Ogg Opus
// file via the in-tree glint encoder (RFC 7845; no libopus needed, so opus
// output is now always available). CRISPASR_OPUS_ENCODER=libopus selects the
// legacy raw-packet libopus path (kept for A/B; it is NOT a standard container
// — length-prefixed frames — so only reachable when built with libopus).
// Sets content_type and returns the encoded bytes (empty on failure).
static std::string crispasr_opus_response(const float* pcm, int n_samples, int sample_rate, const char** content_type) {
#ifdef CRISPASR_HAVE_OPUS
    const char* pref = std::getenv("CRISPASR_OPUS_ENCODER");
    if (pref && std::strcmp(pref, "libopus") == 0) {
        *content_type = "audio/opus";
        return crispasr_encode_opus(pcm, n_samples, sample_rate);
    }
#endif
    *content_type = "audio/ogg";
    return crispasr_make_opus(pcm, n_samples, sample_rate);
}

// ---------------------------------------------------------------------------
// Fatal-signal diagnostics (#261)
//
// The CPU ggml backend runs Silero VAD and pyannote diarization (GPU ASR does
// not touch it). If an image is mistakenly built with `-march=native`
// (GGML_NATIVE=ON) on a host with wider CPU extensions than the deployment
// host, those CPU kernels raise SIGILL and the process dies with NO message —
// httplib's exception handler only catches C++ exceptions, not signals, so the
// container just restarts silently (the exact report in #261).
//
// The real fix is building with -DGGML_NATIVE=OFF (see .devops/*.Dockerfile).
// This handler is the safety net: it turns the silent death into an
// actionable, async-signal-safe diagnostic that names the stage that was
// running, then re-raises so the process still terminates with the correct
// signal (core-dump / restart semantics preserved). It uses only
// async-signal-safe calls (write(2); no printf/malloc). The g_current_stage
// breadcrumb and stage_scope RAII helper are defined earlier (before
// do_transcribe, which sets them).
// ---------------------------------------------------------------------------

static void crispasr_sig_write(const char* s) {
    if (!s)
        return;
    size_t n = 0;
    while (s[n])
        ++n;
#ifdef _WIN32
    (void)fwrite(s, 1, n, stderr);
#else
    ssize_t rc = write(STDERR_FILENO, s, n);
    (void)rc;
#endif
}

static void crispasr_fatal_signal_handler(int sig) {
    const char* name = "fatal signal";
    switch (sig) {
    case SIGILL:
        name = "SIGILL (illegal instruction)";
        break;
    case SIGSEGV:
        name = "SIGSEGV (segmentation fault)";
        break;
    case SIGFPE:
        name = "SIGFPE (floating-point exception)";
        break;
#ifdef SIGBUS
    case SIGBUS:
        name = "SIGBUS (bus error)";
        break;
#endif
    default:
        break;
    }
    crispasr_sig_write("\n*** crispasr-server: FATAL ");
    crispasr_sig_write(name);
    crispasr_sig_write(" during stage: ");
    crispasr_sig_write(g_current_stage ? g_current_stage : "unknown");
    crispasr_sig_write(" ***\n");
    if (sig == SIGILL) {
        crispasr_sig_write("This almost always means the binary uses CPU instructions this host does\n"
                           "not support (e.g. AVX-512 baked in by -march=native). The CPU ggml backend\n"
                           "used by VAD / diarization hit an unsupported instruction. Use an image built\n"
                           "with -DGGML_NATIVE=OFF (portable AVX2), or run plain ASR without vad/diarize\n"
                           "(that path executes on the GPU). See issue #261.\n");
    }
    // Restore the default disposition and re-raise so the process terminates
    // with the original signal (SA_NODEFER lets the re-raise re-enter).
    signal(sig, SIG_DFL);
    raise(sig);
}

static void crispasr_install_fatal_signal_handlers() {
#ifndef _WIN32
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = crispasr_fatal_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
#ifdef SIGBUS
    sigaction(SIGBUS, &sa, nullptr);
#endif
#else
    signal(SIGILL, crispasr_fatal_signal_handler);
    signal(SIGSEGV, crispasr_fatal_signal_handler);
    signal(SIGFPE, crispasr_fatal_signal_handler);
#endif
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

int crispasr_run_server(whisper_params& params, const std::string& host, int port) {
    using namespace httplib;

    crispasr_install_fatal_signal_handlers();

    // Consent-record sink. --consent-log already set this during arg parsing;
    // the env var is the route for a container where adding a flag to the
    // entrypoint is awkward. Explicit flag wins.
    if (!params.consent_log.empty())
        crispasr_consent::set_log_path(params.consent_log);
    crispasr_consent::init_log_path_from_env();
    if (!crispasr_consent::log_path().empty())
        fprintf(stderr,
                "crispasr-server: consent records -> %s (JSON Lines; append-only storage is yours to provide)\n",
                crispasr_consent::log_path().c_str());

    // #261: compare the build's CPU instruction-set baseline against this host.
    // If they mismatch, the CPU-only paths (VAD / diarization) would raise
    // SIGILL — log a loud banner now and refuse those requests below (soft
    // degrade) instead of crashing mid-request. Computed once; read by handlers.
    const crispasr_cpu_isa::IsaCheck cpu_isa = crispasr_cpu_isa::check();
    fprintf(stderr, "%s\n", crispasr_cpu_isa::banner(cpu_isa).c_str());

    // Issue #405: in a GGML_BACKEND_DL package the check above cannot see a CPU
    // module that REFUSED to load (host below every shipped variant's ISA
    // floor) — the registry then has no CPU device and the first model load
    // aborts the whole server on a bare GGML_ASSERT. Probe once and refuse to
    // start with the real story instead. Never fails in non-DL builds.
    {
        ggml_backend_t cpu_probe = core_cpu_backend::init();
        if (!cpu_probe) {
            fprintf(stderr, "crispasr-server: error: no CPU ggml backend could be initialised (see above) — "
                            "refusing to start. Use the '-cpu-legacy' release artifact on this machine, or "
                            "build from source.\n");
            return 1;
        }
        ggml_backend_free(cpu_probe);
    }

    crispasr_c2pa_startup_check();
    if (!params.watermark_model.empty()) {
        crispasr_wm_dispatch::init(params.watermark_model);
    }
    // Honor the --no-watermark opt-out (equivalent to CRISPASR_NO_WATERMARK).
    // A server operator that disables it takes on the AI-content marking duty
    // for every response the process serves — so, like the CLI, it requires the
    // explicit --accept-marking-responsibility attestation. Refuse to start
    // otherwise, rather than silently serving unmarked audio.
    // --no-c2pa is the same class of opt-out as --no-watermark (disables the
    // machine-readable C2PA manifest floor on every response), so it requires the
    // same attestation.
    if ((params.tts_no_watermark || params.tts_no_c2pa) && !params.tts_marking_responsibility_accepted) {
        fprintf(stderr,
                "crispasr: error: server launched with %s requires --accept-marking-responsibility "
                "(the operator accepts AI-content marking responsibility for every response served). "
                "Refusing to start.\n",
                params.tts_no_watermark ? "--no-watermark" : "--no-c2pa");
        return 1;
    }
    if (params.tts_no_watermark || params.tts_no_c2pa) {
        std::time_t t = std::time(nullptr);
        char ts[64];
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
        fprintf(stderr, "[MARKING] ts=%s scope=server no_watermark=%s no_c2pa=%s attestation=\"%s\"\n", ts,
                params.tts_no_watermark ? "yes" : "no", params.tts_no_c2pa ? "yes" : "no",
                params.tts_marking_attestation.c_str());
    }
    crispasr_wm_dispatch::set_disabled(params.tts_no_watermark);

    std::vector<std::string> api_keys = split_api_keys(params.server_api_keys);
    if (const char* env_keys = getenv("CRISPASR_API_KEYS")) {
        std::vector<std::string> more = split_api_keys(env_keys);
        api_keys.insert(api_keys.end(), more.begin(), more.end());
    }

    std::unique_ptr<CrispasrBackend> backend;
    std::mutex model_mutex;
    std::atomic<bool> ready{false};
    std::string backend_name = params.backend;

    // Improvements Phase 4b: concurrent ASR worker pool. The primary `backend`
    // above (+ model_mutex) still handles streaming/TTS/load and any request
    // that touches SHARED state (auto-LID model, aligner, punctuation/truecaser
    // contexts) — those must stay serialized. A "pure ASR" request (explicit
    // language, no aligner, no post-processing) touches only its backend, so it
    // can run on a pooled worker concurrently. Each worker owns its backend and
    // a private mutex; different workers never contend. Built only when
    // CRISPASR_SERVER_WORKERS>1 (default 1 = single instance, unchanged).
    struct AsrWorker {
        std::unique_ptr<CrispasrBackend> backend;
        std::mutex mtx;
    };
    std::unique_ptr<core_pool::WorkerPool<AsrWorker>> asr_pool;

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

        const std::string resolved = crispasr_resolve_model_cli(params.model, backend_name, params.no_prints,
                                                                params.cache_dir, params.auto_download || model_is_auto,
                                                                params.model_quant, params.accept_license);
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
        // Phase 4b: build the concurrent ASR worker pool. Each worker is an
        // independent backend used ONLY for pure-ASR requests (explicit
        // language, no aligner, no post-processing) so it shares no mutable
        // state with other workers. N=1 (default) leaves `asr_pool` null →
        // every request uses the primary backend + model_mutex (unchanged).
        // The --server-workers flag sets the default; CRISPASR_SERVER_WORKERS,
        // when present, overrides it (env stays the ultimate gate).
        int n_workers = std::max(1, params.server_workers);
        if (const char* e = std::getenv("CRISPASR_SERVER_WORKERS"))
            n_workers = std::max(1, atoi(e));

        // #358: the pool serves pure-ASR requests only (explicit language, no
        // aligner, no post-processing) — synthesis always runs serialised under
        // model_mutex. On a backend that can synthesise, someone raising this
        // flag to get concurrent TTS instead gets N full model instances and no
        // extra throughput; 4 workers with the 1.7B qwen3-tts is 4 x ~1.95 GiB
        // and overruns an M1's Metal working-set limit, which surfaces as an
        // allocation failure rather than as "that flag does not apply here".
        //
        // Note only, no behaviour change: there is no capability meaning "can
        // transcribe" (ASR is implicit), so a backend advertising CAP_TTS may
        // still be a genuine ASR backend for which the pool is doing its job.
        // Refusing to build it on that guess would be a throughput regression.
        if (n_workers > 1 && (backend->capabilities() & CAP_TTS)) {
            fprintf(stderr,
                    "crispasr-server: note — the worker pool serves pure-ASR requests only; "
                    "synthesis on '%s' stays serialised on one model instance regardless of "
                    "--server-workers %d, and each worker is a full model copy. For concurrent TTS, "
                    "run N processes behind a load balancer.\n",
                    backend_name.c_str(), n_workers);
        }

        if (n_workers > 1) {
            std::vector<std::unique_ptr<AsrWorker>> workers;
            for (int i = 0; i < n_workers; ++i) {
                auto w = std::make_unique<AsrWorker>();
                w->backend = crispasr_create_backend(backend_name);
                if (!w->backend || !w->backend->init(params)) {
                    fprintf(stderr, "crispasr-server: worker %d init failed — running single-instance\n", i);
                    workers.clear();
                    break;
                }
                if (!skip_warmup) {
                    try {
                        w->backend->warmup();
                    } catch (...) {
                    }
                }
                workers.push_back(std::move(w));
            }
            if (!workers.empty()) {
                asr_pool = std::make_unique<core_pool::WorkerPool<AsrWorker>>(std::move(workers));
                fprintf(stderr, "crispasr-server: ASR worker pool = %zu workers (pure-ASR requests run concurrently)\n",
                        asr_pool->size());
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

    // Security: bound the request body so an unbounded multipart upload cannot
    // buffer gigabytes into RAM (OOM DoS) before auth/routing even run — the
    // body is read (and, for multipart, fully accumulated) BEFORE the route
    // handler + require_auth. set_payload_max_length makes httplib's read_content
    // reject an over-cap Content-Length with 413 and skip the body WITHOUT
    // buffering it — the clean fix for the normal (Content-Length) upload path.
    const size_t max_upload_bytes = 512ull * 1024 * 1024; // 512 MB — far above any real audio upload
    svr.set_payload_max_length(max_upload_bytes);

    // A pre-routing handler runs before body read + route dispatch (httplib
    // routing() calls it before read_content). httplib's CHUNKED reader does
    // NOT honour payload_max_length (only the Content-Length path does), so a
    // Transfer-Encoding: chunked upload would bypass the cap and buffer
    // unbounded — reject chunked bodies on mutating requests here. When
    // --cors-origin is set this handler also attaches CORS + answers OPTIONS.
    const std::string cors_origin = params.server_cors_origin;
    svr.set_pre_routing_handler([cors_origin](const Request& req, Response& res) {
        if ((req.method == "POST" || req.method == "PUT") && req.has_header("Transfer-Encoding")) {
            // Chunked/streamed upload — not bounded by set_payload_max_length.
            res.status = 411; // Length Required — resend with a Content-Length.
            return Server::HandlerResponse::Handled;
        }
        if (!cors_origin.empty()) {
            res.set_header("Access-Control-Allow-Origin", cors_origin);
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key");
            res.set_header("Access-Control-Max-Age", "86400");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return Server::HandlerResponse::Handled;
            }
        }
        return Server::HandlerResponse::Unhandled;
    });
    if (!cors_origin.empty())
        fprintf(stderr, "crispasr-server: CORS enabled (Allow-Origin: %s)\n", cors_origin.c_str());

    auto require_auth = [&](const Request& req, Response& res) -> bool {
        if (is_authorized(req, api_keys))
            return true;
        auth_error(res);
        return false;
    };

    // Phase 4b: route a transcription to a concurrent pooled worker when it is
    // "pure ASR" — explicit language (no shared LID model), no aligner, and no
    // punctuation/truecaser post-processing (those contexts are shared and
    // non-re-entrant). Such a request touches only its own backend, so pooled
    // workers run without contending. Everything else stays on the primary
    // backend + model_mutex (serialized, unchanged). asr_pool is null unless
    // CRISPASR_SERVER_WORKERS>1.
    auto dispatch_transcribe = [&](const httplib::MultipartFormData& audio_file, const whisper_params& rp,
                                   bool need_ts) -> transcription_result {
        const bool lang_explicit = !rp.language.empty() && rp.language != "auto";
        const bool no_post = !punc_ctx && !pcs_ctx && !tc_ctx && !tc_crf_ctx && !tc_lstm_ctx;
        const bool pure_asr = lang_explicit && rp.aligner_model.empty() && no_post;
        if (asr_pool && pure_asr) {
            auto lease = asr_pool->acquire();
            return do_transcribe(audio_file, lease->backend.get(), lease->mtx, rp, need_ts, nullptr, nullptr, nullptr,
                                 nullptr, nullptr);
        }
        return do_transcribe(audio_file, backend.get(), model_mutex, rp, need_ts, punc_ctx.get(), pcs_ctx.get(),
                             tc_ctx.get(), tc_crf_ctx.get(), tc_lstm_ctx.get());
    };

    // -----------------------------------------------------------------------
    // §381: source separation — persistent context for --separate-model.
    // Loads the GGUF once at startup, auto-detects arch (htdemucs /
    // mel-band-roformer), and holds it resident. A std::mutex serialises
    // concurrent requests. Mirrors the --chat-model precedent.
    // -----------------------------------------------------------------------
    enum class SepArch { NONE, HTDEMUCS, MEL_BAND_ROFORMER };
    struct SepCtx {
        SepArch arch = SepArch::NONE;
        htdemucs_context* htd_ctx = nullptr;
        mel_band_roformer_context* mbr_ctx = nullptr;
        std::mutex mtx;
        int sample_rate = 0;

        int n_sources() const {
            if (arch == SepArch::HTDEMUCS && htd_ctx)
                return htdemucs_n_sources(htd_ctx);
            if (arch == SepArch::MEL_BAND_ROFORMER && mbr_ctx)
                return mel_band_roformer_n_sources(mbr_ctx);
            return 0;
        }
        const char* source_name(int i) const {
            if (arch == SepArch::HTDEMUCS && htd_ctx)
                return htdemucs_source_name(htd_ctx, i);
            if (arch == SepArch::MEL_BAND_ROFORMER && mbr_ctx)
                return mel_band_roformer_source_name(mbr_ctx, i);
            return "";
        }
    };

    std::unique_ptr<SepCtx> sep_ctx;

    if (!params.separate_model.empty()) {
        const std::string sep_resolved = crispasr_resolve_model_cli(
            params.separate_model, "" /*auto-detect*/, params.no_prints, params.cache_dir, params.auto_download);
        if (sep_resolved.empty()) {
            fprintf(stderr, "crispasr-server: cannot resolve --separate-model '%s'\n", params.separate_model.c_str());
            return 1;
        }
        gguf_context* meta = core_gguf::open_metadata(sep_resolved.c_str());
        if (!meta) {
            fprintf(stderr, "crispasr-server: cannot open --separate-model '%s'\n", sep_resolved.c_str());
            return 1;
        }
        const std::string sep_arch = core_gguf::kv_str(meta, "general.architecture", "");
        core_gguf::free_metadata(meta);

        sep_ctx = std::make_unique<SepCtx>();
        if (sep_arch == "htdemucs") {
            auto hp = htdemucs_default_params();
            hp.use_gpu = params.use_gpu;
            hp.n_threads = params.n_threads;
            sep_ctx->htd_ctx = htdemucs_init_from_file(sep_resolved.c_str(), hp);
            if (!sep_ctx->htd_ctx) {
                fprintf(stderr, "crispasr-server: failed to load htdemucs from '%s'\n", sep_resolved.c_str());
                return 1;
            }
            sep_ctx->arch = SepArch::HTDEMUCS;
            sep_ctx->sample_rate = htdemucs_sample_rate(sep_ctx->htd_ctx);
            fprintf(stderr, "crispasr-server: separation model loaded — htdemucs (%d Hz, %d stems)\n",
                    sep_ctx->sample_rate, sep_ctx->n_sources());
        } else if (sep_arch == "mel-band-roformer") {
            auto mp = mel_band_roformer_default_params();
            mp.use_gpu = params.use_gpu;
            mp.n_threads = params.n_threads;
            sep_ctx->mbr_ctx = mel_band_roformer_init_from_file(sep_resolved.c_str(), mp);
            if (!sep_ctx->mbr_ctx) {
                fprintf(stderr, "crispasr-server: failed to load mel-band-roformer from '%s'\n", sep_resolved.c_str());
                return 1;
            }
            sep_ctx->arch = SepArch::MEL_BAND_ROFORMER;
            sep_ctx->sample_rate = mel_band_roformer_sample_rate(sep_ctx->mbr_ctx);
            fprintf(stderr, "crispasr-server: separation model loaded — mel-band-roformer (%d Hz, %d stems)\n",
                    sep_ctx->sample_rate, sep_ctx->n_sources());
        } else {
            fprintf(stderr,
                    "crispasr-server: --separate-model: unsupported arch '%s' "
                    "(expected htdemucs or mel-band-roformer)\n",
                    sep_arch.c_str());
            return 1;
        }
    }

    // -----------------------------------------------------------------------
    // POST /v1/audio/separation — source separation (§381)
    // -----------------------------------------------------------------------
    svr.Post("/v1/audio/separation", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        if (!sep_ctx) {
            json_error(res, 503,
                       "source separation is not enabled on this server "
                       "(start with --separate-model PATH)",
                       "separation_disabled");
            return;
        }
        if (!req.has_file("file")) {
            json_error(res, 400, "no 'file' field in multipart upload");
            return;
        }

        const auto& audio_file = req.get_file_value("file");
        const std::string stems_csv = form_string(req, "stems", "");
        fprintf(stderr, "crispasr-server: /v1/audio/separation received '%s' (%zu bytes, stems='%s')\n",
                log_sanitize(audio_file.filename).c_str(), audio_file.content.size(), log_sanitize(stems_csv).c_str());

        // Validate that at least one requested stem exists.
        if (!stems_csv.empty() && stems_csv != "all") {
            bool any_match = false;
            for (int s = 0; s < sep_ctx->n_sources(); s++) {
                if (crispasr_stem_selected(stems_csv, sep_ctx->source_name(s))) {
                    any_match = true;
                    break;
                }
            }
            if (!any_match) {
                std::string avail;
                for (int s = 0; s < sep_ctx->n_sources(); s++) {
                    if (!avail.empty())
                        avail += ", ";
                    avail += sep_ctx->source_name(s);
                }
                json_error(res, 400, "no stems match the selection '" + stems_csv + "'; available: " + avail);
                return;
            }
        }

        // Write the upload to a temp file so read_audio_data can decode it.
        std::string tmp_path =
            write_temp_audio(audio_file.content.data(), audio_file.content.size(), audio_file.filename);
        if (tmp_path.empty()) {
            json_error(res, 500, "failed to write temporary audio file");
            return;
        }

        // Read and resample to the model's native rate (44100), stereo.
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(tmp_path, mono, stereo, /*stereo=*/true,
                             /*target_rate=*/sep_ctx->sample_rate)) {
            std::remove(tmp_path.c_str());
            json_error(res, 400, "cannot decode audio file");
            return;
        }
        std::remove(tmp_path.c_str());

        // Interleave to stereo (same logic as crispasr_separate_cli.cpp).
        const int n_channels = 2;
        const bool have_stereo = stereo.size() >= 2 && !stereo[0].empty();
        const int n_frames = have_stereo ? (int)stereo[0].size() : (int)mono.size();
        std::vector<float> pcm((size_t)n_frames * n_channels);
        for (int i = 0; i < n_frames; i++) {
            for (int c = 0; c < n_channels; c++) {
                float v;
                if (have_stereo)
                    v = stereo[c < (int)stereo.size() ? c : (int)stereo.size() - 1][i];
                else
                    v = mono[i];
                pcm[(size_t)i * n_channels + c] = v;
            }
        }

        // Run separation under the mutex.
        crispasr_separation_view view{};
        htdemucs_result* htd_r = nullptr;
        mel_band_roformer_result* mbr_r = nullptr;
        {
            std::lock_guard<std::mutex> lock(sep_ctx->mtx);
            if (sep_ctx->arch == SepArch::HTDEMUCS) {
                htd_r = htdemucs_separate(sep_ctx->htd_ctx, pcm.data(), n_frames);
                if (!htd_r) {
                    json_error(res, 500, "separation failed");
                    return;
                }
                view.n_sources = htd_r->n_sources;
                view.n_channels = htd_r->n_channels;
                view.n_frames = htd_r->n_samples;
                view.sample_rate = htd_r->sample_rate;
                view.sources = htd_r->sources;
                view.source_names = htd_r->source_names;
            } else {
                mbr_r = mel_band_roformer_separate(sep_ctx->mbr_ctx, pcm.data(), n_frames, n_channels);
                if (!mbr_r) {
                    json_error(res, 500, "separation failed");
                    return;
                }
                view.n_sources = mbr_r->n_sources;
                view.n_channels = mbr_r->n_channels;
                view.n_frames = mbr_r->n_samples;
                view.sample_rate = mbr_r->sample_rate;
                view.sources = mbr_r->sources;
                view.source_names = mbr_r->source_names;
            }
        }

        // Build multipart/mixed response: one WAV part per selected stem.
        // Boundary chosen to be unique enough for this use case.
        const std::string boundary = "----CrispASR_stem_boundary";
        std::string body;
        int n_parts = 0;
        for (int s = 0; s < view.n_sources; s++) {
            const std::string name = view.source_names[s] ? view.source_names[s] : ("stem" + std::to_string(s));
            if (!crispasr_stem_selected(stems_csv, name))
                continue;
            const std::string wav = crispasr_stem_to_wav(view, s);
            if (wav.empty())
                continue;
            body += "--" + boundary + "\r\n";
            body += "Content-Type: audio/wav\r\n";
            body += "Content-Disposition: attachment; filename=\"" + name + ".wav\"\r\n";
            body += "\r\n";
            body.append(wav);
            body += "\r\n";
            n_parts++;
        }

        // Free the backend result now that WAVs are serialised.
        if (htd_r)
            htdemucs_result_free(htd_r);
        if (mbr_r)
            mel_band_roformer_result_free(mbr_r);

        if (n_parts == 0) {
            json_error(res, 400, "no stems matched the selection");
            return;
        }

        body += "--" + boundary + "--\r\n";
        res.set_content(body, "multipart/mixed; boundary=" + boundary);
        fprintf(stderr, "crispasr-server: /v1/audio/separation → %d stem(s)\n", n_parts);
    });

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
        fprintf(stderr, "crispasr-server: /inference received '%s' (%zu bytes)\n",
                log_sanitize(audio_file.filename).c_str(), audio_file.content.size());

        // Per-request parameter overrides.
        whisper_params rp = params;
        rp.language = form_string(req, "language", rp.language);
        if (std::string lerr = validate_request_language(params.backend, rp.language); !lerr.empty()) {
            json_error(res, 400, lerr, "invalid_request_error", "language");
            return;
        }
        rp.source_lang = form_string(req, "source_lang", rp.source_lang);
        rp.target_lang = form_string(req, "target_lang", rp.target_lang);
        rp.translate = form_bool(req, "translate", rp.translate);
        rp.punctuation = form_bool(req, "punctuation", rp.punctuation);
        rp.diarize = form_bool(req, "diarize", rp.diarize);
        if (rp.diarize && rp.diarize_method.empty())
            rp.diarize_method = form_string(req, "diarize_method", "energy");
        rp.diarize_embedder = form_string(req, "diarize_embedder", rp.diarize_embedder);
        rp.diarize_cluster_threshold = form_float(req, "diarize_cluster_threshold", rp.diarize_cluster_threshold);
        rp.diarize_cluster_threshold_explicit = req.has_file("diarize_cluster_threshold");
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
        rp.return_logits = form_bool(req, "return_logits", rp.return_logits);
        // `sensitivity` is applied FIRST so an explicit entropy/logprob/
        // no_speech/temperature_inc field in the same request still wins —
        // same last-wins rule as the CLI, where --sensitivity is overridden by
        // a later -et/-lpt/-nth. An unknown preset is ignored with a warning
        // rather than failing the request, since the four fields below can
        // still fully specify the decode.
        {
            const std::string sens = form_string(req, "sensitivity", "");
            if (!sens.empty()) {
                core_sensitivity::Preset sp;
                if (core_sensitivity::parse_preset(sens, sp)) {
                    const auto st = core_sensitivity::preset(sp);
                    rp.entropy_thold = st.entropy_thold;
                    rp.logprob_thold = st.logprob_thold;
                    rp.no_speech_thold = st.no_speech_thold;
                    rp.temperature_inc = st.temperature_inc;
                } else {
                    fprintf(stderr, "crispasr[server]: ignoring unknown sensitivity '%s' (expected: %s)\n",
                            sens.c_str(), core_sensitivity::preset_list());
                }
            }
        }
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
        // #227: VAD boundary reuse. `vad_export=true` returns the computed
        // boundaries under `vad_segments`; `vad_import=<that object>` reuses
        // them and skips VAD entirely.
        rp.vad_export_inline = form_bool(req, "vad_export", rp.vad_export_inline);
        rp.vad_import_strict = form_bool(req, "vad_import_strict", rp.vad_import_strict);
        // #311: strict pipeline — a required stage that fails returns an HTTP error
        // instead of a degraded 200. Parity with the CLI's --strict-pipeline family.
        rp.strict_pipeline = form_bool(req, "strict_pipeline", rp.strict_pipeline);
        rp.require_vad = form_bool(req, "require_vad", rp.require_vad);
        rp.require_word_timestamps = form_bool(req, "require_word_timestamps", rp.require_word_timestamps);
        rp.require_punctuation = form_bool(req, "require_punctuation", rp.require_punctuation);
        rp.vad_export_raw = form_bool(req, "vad_export_raw", rp.vad_export_raw);
        rp.vad_import_json = form_string(req, "vad_import", rp.vad_import_json);
        rp.max_context = form_int(req, "max_context", rp.max_context);
        rp.audio_ctx = form_int(req, "audio_ctx", rp.audio_ctx);
        rp.word_thold = form_float(req, "word_thold", rp.word_thold);
        rp.carry_initial_prompt = form_bool(req, "carry_initial_prompt", rp.carry_initial_prompt);
        rp.chunk_overlap_seconds = form_float(req, "chunk_overlap", rp.chunk_overlap_seconds);
        rp.lcs_dedup = form_string(req, "lcs_dedup", rp.lcs_dedup);
        rp.lcs_min_length = form_int(req, "lcs_min_length", rp.lcs_min_length);
        rp.parakeet_decoder = form_string(req, "parakeet_decoder", rp.parakeet_decoder);
        {
            // Issue #257: parakeet/canary local-attention window "L,R" (encoder
            // frames) — NeMo rel_pos_local_attn, bounds long-audio VRAM.
            const std::string ac = form_string(req, "att_context", "");
            int l = INT_MIN, r = INT_MIN;
            if (!ac.empty() && std::sscanf(ac.c_str(), "%d,%d", &l, &r) == 2) {
                rp.att_context_left = l;
                rp.att_context_right = r;
            }
        }
        rp.no_auto_aligner = form_bool(req, "no_auto_aligner", rp.no_auto_aligner);
        rp.show_alternatives = form_bool(req, "show_alternatives", rp.show_alternatives);
        rp.n_alternatives = form_int(req, "alt_n", rp.n_alternatives);

        // #261: soft-degrade — refuse CPU-only VAD/diarize on an ISA-mismatched
        // host rather than SIGILL-crashing the server (see /v1/audio/transcriptions).
        if (!cpu_isa.ok && (rp.vad || rp.diarize)) {
            json_error(res, 400,
                       "this server build requires CPU instructions this host lacks (" + cpu_isa.missing +
                           "), so VAD/diarization cannot run without crashing. Retry without "
                           "vad/diarize, or deploy a portable image built with -DGGML_NATIVE=OFF (issue #261).");
            return;
        }

        auto result = dispatch_transcribe(audio_file, rp, /*need_timestamps=*/true);
        if (!result.ok) {
            json_error(res, 400, result.error);
            return;
        }

        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", result.duration_s,
                result.elapsed_s, result.elapsed_s > 0 ? result.duration_s / result.elapsed_s : 0.0);

        std::string json = crispasr_segments_to_native_json(result.segs, backend_name, result.duration_s);
        if (rp.return_logits)
            json = add_ctc_logits_to_json(json, result.logits);
        json = add_vad_segments_to_json(json, result.vad_segments_json);
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
    //   return_logits    (optional) — include dense CTC logits for supported CTC backends
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
    //   sensitivity      (optional) — conservative|balanced|aggressive: the four
    //                      threshold fields below as one bundle. Applied first,
    //                      so an explicit field in the same request still wins.
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
        if (std::string lerr = validate_request_language(params.backend, language); !lerr.empty()) {
            json_error(res, 400, lerr, "invalid_request_error", "language");
            return;
        }
        std::string prompt = form_string(req, "prompt", "");
        float temperature = form_float(req, "temperature", params.temperature);
        uint64_t seed = form_u64(req, "seed", params.seed);
        int max_new_tokens = form_int(req, "max_new_tokens", params.max_new_tokens);
        max_new_tokens = form_int(req, "max_tokens", max_new_tokens);
        float frequency_penalty = form_float(req, "frequency_penalty", params.frequency_penalty);

        // Validate response_format early.
        if (response_format != "json" && response_format != "verbose_json" && response_format != "text" &&
            response_format != "srt" && response_format != "vtt" && response_format != "diarized_json") {
            json_error(res, 400,
                       "invalid response_format '" + response_format +
                           "'; must be one of: json, verbose_json, text, srt, vtt, diarized_json");
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
        rp.diarize_cluster_threshold_explicit = req.has_file("diarize_cluster_threshold");
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
        rp.return_logits = form_bool(req, "return_logits", rp.return_logits);
        // `sensitivity` is applied FIRST so an explicit entropy/logprob/
        // no_speech/temperature_inc field in the same request still wins —
        // same last-wins rule as the CLI, where --sensitivity is overridden by
        // a later -et/-lpt/-nth. An unknown preset is ignored with a warning
        // rather than failing the request, since the four fields below can
        // still fully specify the decode.
        {
            const std::string sens = form_string(req, "sensitivity", "");
            if (!sens.empty()) {
                core_sensitivity::Preset sp;
                if (core_sensitivity::parse_preset(sens, sp)) {
                    const auto st = core_sensitivity::preset(sp);
                    rp.entropy_thold = st.entropy_thold;
                    rp.logprob_thold = st.logprob_thold;
                    rp.no_speech_thold = st.no_speech_thold;
                    rp.temperature_inc = st.temperature_inc;
                } else {
                    fprintf(stderr, "crispasr[server]: ignoring unknown sensitivity '%s' (expected: %s)\n",
                            sens.c_str(), core_sensitivity::preset_list());
                }
            }
        }
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
        // #227: VAD boundary reuse. `vad_export=true` returns the computed
        // boundaries under `vad_segments`; `vad_import=<that object>` reuses
        // them and skips VAD entirely.
        rp.vad_export_inline = form_bool(req, "vad_export", rp.vad_export_inline);
        rp.vad_import_strict = form_bool(req, "vad_import_strict", rp.vad_import_strict);
        // #311: strict pipeline — a required stage that fails returns an HTTP error
        // instead of a degraded 200. Parity with the CLI's --strict-pipeline family.
        rp.strict_pipeline = form_bool(req, "strict_pipeline", rp.strict_pipeline);
        rp.require_vad = form_bool(req, "require_vad", rp.require_vad);
        rp.require_word_timestamps = form_bool(req, "require_word_timestamps", rp.require_word_timestamps);
        rp.require_punctuation = form_bool(req, "require_punctuation", rp.require_punctuation);
        rp.vad_export_raw = form_bool(req, "vad_export_raw", rp.vad_export_raw);
        rp.vad_import_json = form_string(req, "vad_import", rp.vad_import_json);
        rp.max_context = form_int(req, "max_context", rp.max_context);
        rp.audio_ctx = form_int(req, "audio_ctx", rp.audio_ctx);
        rp.word_thold = form_float(req, "word_thold", rp.word_thold);
        rp.carry_initial_prompt = form_bool(req, "carry_initial_prompt", rp.carry_initial_prompt);
        rp.chunk_overlap_seconds = form_float(req, "chunk_overlap", rp.chunk_overlap_seconds);
        rp.lcs_dedup = form_string(req, "lcs_dedup", rp.lcs_dedup);
        rp.lcs_min_length = form_int(req, "lcs_min_length", rp.lcs_min_length);
        rp.parakeet_decoder = form_string(req, "parakeet_decoder", rp.parakeet_decoder);
        {
            // Issue #257: parakeet/canary local-attention window "L,R" (encoder
            // frames) — NeMo rel_pos_local_attn, bounds long-audio VRAM.
            const std::string ac = form_string(req, "att_context", "");
            int l = INT_MIN, r = INT_MIN;
            if (!ac.empty() && std::sscanf(ac.c_str(), "%d,%d", &l, &r) == 2) {
                rp.att_context_left = l;
                rp.att_context_right = r;
            }
        }
        rp.no_auto_aligner = form_bool(req, "no_auto_aligner", rp.no_auto_aligner);
        rp.show_alternatives = form_bool(req, "show_alternatives", rp.show_alternatives);
        rp.n_alternatives = form_int(req, "alt_n", rp.n_alternatives);
        if (!prompt.empty())
            rp.prompt = prompt;

        // #261: soft-degrade. When the build's CPU ISA baseline exceeds this
        // host, the CPU-only VAD / diarization kernels would SIGILL and kill
        // the server. Refuse those requests with a clear 400 instead; plain
        // ASR (GPU) still works, so a client can retry with vad=false&diarize=false.
        if (!cpu_isa.ok && (rp.vad || rp.diarize)) {
            json_error(res, 400,
                       "this server build requires CPU instructions this host lacks (" + cpu_isa.missing +
                           "), so VAD/diarization cannot run without crashing. Retry without "
                           "vad/diarize, or deploy a portable image built with -DGGML_NATIVE=OFF (issue #261).");
            return;
        }

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

        const bool need_timestamps = response_format == "verbose_json" || response_format == "srt" ||
                                     response_format == "vtt" || response_format == "diarized_json";
        auto result = dispatch_transcribe(audio_file, rp, need_timestamps);
        if (!result.ok) {
            json_error(res, 400, result.error);
            return;
        }

        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime), format=%s\n",
                result.duration_s, result.elapsed_s, result.elapsed_s > 0 ? result.duration_s / result.elapsed_s : 0.0,
                response_format.c_str());

        // Format response.
        // Diarization is expensive — on the 48-minute file in #326 it was the
        // dominant cost of the request. `text` and the default `json` have
        // nowhere to put a speaker label, so asking for both means paying for a
        // stage whose result is then thrown away. That was silent; say it.
        if (rp.diarize && (response_format == "text" || response_format == "json")) {
            const bool labelled = std::any_of(result.segs.begin(), result.segs.end(),
                                              [](const crispasr_segment& s) { return !s.speaker.empty(); });
            if (labelled) {
                fprintf(stderr,
                        "crispasr-server: diarization produced speaker labels but response_format='%s' "
                        "cannot carry them — use 'diarized_json', or 'verbose_json' / 'srt' / 'vtt'\n",
                        response_format.c_str());
            }
        }

        if (response_format == "text") {
            res.set_content(crispasr_segments_to_text(result.segs), "text/plain; charset=utf-8");
        } else if (response_format == "srt") {
            res.set_content(crispasr_segments_to_srt(result.segs), "application/x-subrip; charset=utf-8");
        } else if (response_format == "vtt") {
            res.set_content(crispasr_segments_to_vtt(result.segs), "text/vtt; charset=utf-8");
        } else if (response_format == "verbose_json") {
            std::string task = rp.translate ? "translate" : "transcribe";
            std::string json = crispasr_segments_to_openai_verbose_json(result.segs, result.duration_s, result.language,
                                                                        task, temperature);
            if (rp.return_logits)
                json = add_ctc_logits_to_json(json, result.logits);
            json = add_vad_segments_to_json(json, result.vad_segments_json);
            res.set_content(json, "application/json");
        } else if (response_format == "diarized_json") {
            std::string task = rp.translate ? "translate" : "transcribe";
            std::string json =
                crispasr_segments_to_diarized_json(result.segs, result.duration_s, result.language, task, temperature);
            if (rp.return_logits)
                json = add_ctc_logits_to_json(json, result.logits);
            json = add_vad_segments_to_json(json, result.vad_segments_json);
            res.set_content(json, "application/json");
        } else {
            // Default: json — {"text": "..."}
            std::string json = crispasr_segments_to_openai_json(result.segs);
            if (rp.return_logits)
                json = add_ctc_logits_to_json(json, result.logits);
            json = add_vad_segments_to_json(json, result.vad_segments_json);
            res.set_content(json, "application/json");
        }
    });

    // -----------------------------------------------------------------------
    // POST /load — hot-swap model
    // -----------------------------------------------------------------------
    svr.Post("/load", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        // Phase 4b: the worker pool is built once at startup; a runtime model
        // swap would leave pooled workers serving the old model. Reject the swap
        // rather than silently serve a stale model (restart to change models
        // when running with CRISPASR_SERVER_WORKERS>1).
        if (asr_pool) {
            json_error(res, 409, "/load is not supported with CRISPASR_SERVER_WORKERS>1 — restart to change models");
            return;
        }
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

        const std::string resolved_model = crispasr_resolve_model_cli(
            new_model, new_backend, params.no_prints, params.cache_dir, params.auto_download || new_model_is_auto,
            params.model_quant, params.accept_license);
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
    // GET /progress (#408) — poll the active transcription job:
    // {"busy": bool, "progress": -1..100}, -1 = idle. Auth-gated like the
    // other introspection routes; /health stays the public liveness probe.
    // -----------------------------------------------------------------------
    svr.Get("/progress", [&](const Request& req, Response& res) {
        if (!require_auth(req, res))
            return;
        char buf[96];
        snprintf(buf, sizeof(buf), "{\"busy\": %s, \"progress\": %d}", g_server_active.load() > 0 ? "true" : "false",
                 g_server_progress.load());
        res.set_content(buf, "application/json");
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
    //     "response_format": "wav"|"pcm"|"f32"|"mp3"|"aac"|"opus" (optional, default "wav")
    //     "consent_attestation":  "<text>",              (REQUIRED when `voice` is a .wav clone)
    //     "spoken_disclaimer":    true|false,            (optional, default true)
    //     "phonemes":        "<IPA>",              (optional; kokoro/piper — skips the G2P)
    //     "language":        "de",                       (optional; the language to SPEAK.
    //                                                     `target_lang` is an alias. cosyvoice3,
    //                                                     qwen3-tts and moss-tts act on it;
    //                                                     language-agnostic backends ignore it)
    //     "source_lang":     "en",                       (optional; the language the CLONING
    //                                                     REFERENCE is spoken in. `ref_lang` is an
    //                                                     alias. Only needed when it cannot be
    //                                                     inferred from the reference transcript —
    //                                                     see #329)
    //     "marking_attestation":  "<text>",              (required to HONOR spoken_disclaimer:false
    //                                                     on a clone; without it the opt-out is
    //                                                     denied, not the request — see below)
    //   }
    //
    // Returns:
    //   200 audio/wav                 — 16-bit PCM int16 RIFF, 24 kHz mono (default)
    //   200 audio/pcm                 — raw int16 LE PCM, 24 kHz mono (OpenAI spec)
    //   200 application/octet-stream  — raw float32 PCM (crispasr-specific f32)
    //
    //   Response headers on a voice clone:
    //     X-Crispasr-Spoken-Disclaimer: applied|skipped
    //     X-Crispasr-Marking-Warning:   <set only when an unattested opt-out was denied>
    //
    //   400 — backend lacks CAP_TTS, missing/empty input, input too long,
    //         malformed body, unknown response_format, speed out of range,
    //         voice clone without consent_attestation
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
        // Mint the per-request correlation id first, so every audit record this
        // handler emits carries it. Without it a [CONSENT] line and the response
        // it authorised are unlinkable on a server serving many requests.
        crispasr_consent::new_request_id();
        res.set_header("X-Crispasr-Request-Id", crispasr_consent::request_correlation_id());
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
        // Before anything logs, resolves or opens this name. A control character
        // in it forges log records at every downstream site that echoes it —
        // the GGUF loader, the kokoro adapter, ggml — none of which know the
        // string came off a socket. See crispasr_voice_clone_policy.h; the
        // path-traversal guard further down is a separate concern.
        if (crispasr_voice::voice_name_has_control_chars(voice_name)) {
            json_error(res, 400, "'voice' must not contain control characters", "invalid_voice", "voice");
            return;
        }
        std::string consent_attestation = body.value("consent_attestation", "");
        std::string instructions = body.value("instructions", "");
        // #13273: omnivoice's instruct is a CLOSED 48-item vocabulary and
        // upstream rejects anything else. Validate at the edge so a bad value
        // is a 400 naming the offending item, not a 500 from a synthesis that
        // failed three layers down. Free: the validator is a weight-free header.
        if (!instructions.empty() && backend_name == "omnivoice") {
            const auto parsed = core_omnivoice_instruct::parse(instructions);
            if (parsed.status != core_omnivoice_instruct::Status::ok &&
                parsed.status != core_omnivoice_instruct::Status::cleared) {
                json_error(res, 400, parsed.error, "invalid_instructions", "instructions");
                return;
            }
        }
        // #316: drive the acoustic model with these phonemes, skipping the G2P.
        // kokoro and piper only; refused elsewhere rather than silently
        // synthesizing `input`, which would make an A/B look like the phonemes
        // changed nothing.
        std::string tts_phonemes = body.value("phonemes", "");
        // #201: transcript of a .wav clone reference (TADA on-the-fly cloning,
        // gated by CRISPASR_TADA_WAV_CLONE). Passed through to the backend as
        // tts_ref_text; a companion <name>.txt in --voice-dir is the fallback.
        std::string ref_text = body.value("ref_text", "");
        // pad N ms of silence at the start of the output
        int tts_pad_silence_ms = body.value("pad_silence_ms", 0);
        // spoken_disclaimer defaults to true; set to false to skip the
        // audible AI-disclosure prefix (watermark + C2PA remain). The opt-out
        // is only honored when attested — see the marking gate below.
        const bool spoken_disclaimer = body.value("spoken_disclaimer", true);

        // Voice-cloning consent gate: a clone requires an explicit
        // consent_attestation field in the request body.
        // A clone is a recording reference OR a pack that declares it was baked from a
        // real recording — and the name is resolved against --voice-dir first,
        // because {"voice": "victim"} reaches the same file as
        // {"voice": "victim.wav"}. The old suffix test on the raw string missed
        // both, so a bare name and every .gguf-only cloning backend (chatterbox
        // has no .wav path at all) cleared this gate untouched.
        // See crispasr_voice_clone_policy.h.
        // ... and a name that resolves to no file at all may still be an entry
        // inside the backend's multi-voice bank, which is where cosyvoice3 keeps
        // every one of its voice clones.
        const crispasr_voice::CloneDecision clone_decision = crispasr_voice::classify_voice(
            voice_name, params.tts_voice_dir, /*baked_from_wav_this_run=*/false, backend->voice_bank_path());
        const bool is_voice_clone = clone_decision.is_clone;
        if (is_voice_clone && consent_attestation.empty()) {
            json_error(res, 400,
                       "voice cloning requires a 'consent_attestation' field in the request body. "
                       "This field should contain a statement attesting that you have the consent "
                       "of the speaker whose voice is being cloned, or that it is your own voice. "
                       "Example: {\"consent_attestation\": \"I have the speaker's consent\"}",
                       "consent_required", "consent_attestation");
            return;
        }
        // Marking-responsibility attestation (parallel to voice-clone consent):
        // opting out of the spoken AI-disclaimer on a voice clone requires an
        // explicit 'marking_attestation' field — the requester accepts the
        // disclosure duty.
        //
        // A server launched with --accept-marking-responsibility has already
        // accepted that duty for EVERY response the process serves, which
        // subsumes the per-request field — so it satisfies the gate too (the
        // operator attestation is the broader one; demanding the per-request
        // field on top of it refuses a request the operator already covered).
        //
        // #312: an UNATTESTED opt-out is DENIED, not refused — the decision, and
        // the reasoning behind it, live in crispasr_marking_policy.h so they can
        // be unit-tested (tests/test-marking-policy.cpp) rather than only through
        // a live server with a model loaded.
        // Whose voice is this? A PRESET voice that belongs to an identifiable
        // person needs the same audible disclosure a clone does — and does NOT
        // need consent_attestation, which is why this is resolved after the gate
        // above rather than folded into it. Per-request override, then the
        // pack/bank declaration, then the backend's.
        // See crispasr_speaker_identity.h.
        bool identity_recognised = true;
        const crispasr_voice::SpeakerIdentity identity_override =
            crispasr_voice::parse_speaker_identity(body.value("speaker_identity", std::string()), &identity_recognised);
        if (!identity_recognised) {
            json_error(res, 400, "unrecognised 'speaker_identity'. Expected one of: real_person, synthetic, unknown.",
                       "invalid_speaker_identity", "speaker_identity");
            return;
        }
        const crispasr_voice::SpeakerIdentity speaker_identity = crispasr_voice::resolve_speaker_identity(
            identity_override, clone_decision.pack_identity, backend->declared_speaker_identity(params.model),
            crispasr_voice::read_model_speaker_identity(params.model));
        const bool needs_spoken_disclosure =
            crispasr_voice::requires_spoken_disclosure(is_voice_clone, speaker_identity);
        if (crispasr_voice::should_warn_unknown_identity(is_voice_clone, speaker_identity) &&
            crispasr_voice::claim_unknown_identity_warning(backend_name)) {
            fprintf(stderr, "%s\n", crispasr_voice::unknown_identity_warning(backend_name).c_str());
        }

        const crispasr_marking::Decision marking =
            crispasr_marking::decide(needs_spoken_disclosure, spoken_disclaimer, body.value("marking_attestation", ""),
                                     params.tts_marking_responsibility_accepted, params.tts_marking_attestation);
        if (is_voice_clone) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
            // The audit line records what this response ACTUALLY carries, not
            // what was asked for — a denied opt-out reads spoken_disclaimer=yes.
            // ref_sha256 binds it to the bytes cloned (see
            // crispasr_consent_record.h); req is the per-request correlation id,
            // without which a consent line and the response it authorised are
            // unlinkable on a server serving many.
            const std::string resolved_ref = crispasr_voice::resolve_voice_path(voice_name, params.tts_voice_dir);
            std::string ref_hash = crispasr_consent::file_sha256(resolved_ref);
            crispasr_consent::emit("CONSENT", ts,
                                   {{"voice", log_sanitize(voice_name)},
                                    {"clone_reason", clone_decision.reason},
                                    {"attestation", log_sanitize(consent_attestation), /*quoted=*/true},
                                    {"spoken_disclaimer", marking.apply_spoken_disclaimer ? "yes" : "no"},
                                    {"ref_sha256", ref_hash.empty() ? "none" : ref_hash},
                                    {"req", crispasr_consent::request_correlation_id()}});
            if (marking.optout_denied) {
                fprintf(stderr,
                        "[MARKING] ts=%s scope=request no_spoken_disclaimer=DENIED "
                        "reason=\"no 'marking_attestation' field\" "
                        "action=\"served with the spoken AI-disclaimer\"\n",
                        ts);
            } else if (marking.optout_honored) {
                fprintf(stderr, "[MARKING] ts=%s scope=%s no_spoken_disclaimer=yes attestation=\"%s\"\n", ts,
                        marking.scope.c_str(), log_sanitize(marking.attestation).c_str());
            }
        } else if (needs_spoken_disclosure) {
            // A real-person PRESET: disclosed, not gated, so there is no
            // [CONSENT] line above to carry the record. Without this the only
            // trace of an Art. 50(4) disclosure on a non-clone would be the
            // audio itself.
            char ts[64];
            auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
            fprintf(stderr,
                    "[MARKING] ts=%s scope=request voice=%s speaker_identity=real_person spoken_disclaimer=%s\n", ts,
                    log_sanitize(voice_name).c_str(), marking.apply_spoken_disclaimer ? "yes" : "no");
        }
        const bool apply_spoken_disclaimer = marking.apply_spoken_disclaimer;
        // Tell the client what it got. Called once the response is committed to
        // being a 200 (buffered: after synthesis; streaming: before the first
        // chunk), so an error response never claims a disclaimer it never made.
        // Keyed on whether a disclosure was OWED, not on cloning: a real-person
        // preset gets one, and a client that reads these headers to decide
        // whether to show a visual label needs to hear about it.
        auto set_marking_headers = [&marking, needs_spoken_disclosure](Response& r) {
            if (!needs_spoken_disclosure)
                return;
            r.set_header("X-Crispasr-Spoken-Disclaimer", marking.apply_spoken_disclaimer ? "applied" : "skipped");
            if (marking.optout_denied)
                r.set_header("X-Crispasr-Marking-Warning",
                             "'spoken_disclaimer': false ignored - it requires a 'marking_attestation' field "
                             "(or a server launched with --accept-marking-responsibility); "
                             "served with the spoken AI-disclaimer");
        };

        std::string response_format = body.value("response_format", std::string("wav"));
        if (response_format != "wav" && response_format != "pcm" && response_format != "f32" &&
            response_format != "mp3" && response_format != "aac" && response_format != "opus") {
            json_error(res, 400, "response_format must be one of 'wav', 'pcm', 'f32', 'mp3', 'aac', or 'opus'",
                       "unsupported_response_format", "response_format");
            return;
        }
        // opus output is always available via the in-tree glint encoder
        // (a real Ogg Opus file); no libopus required.

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
        if (!voice_name.empty()) {
            // Path-traversal guard: `voice` is resolved by the backend as a
            // filesystem path (bare name relative to --voice-dir, or a path).
            // Over the network that must not escape the voice dir or read an
            // absolute path — reject `..`, absolute/home paths, NUL and
            // backslashes (a legitimate bare name / speaker / preset has none).
            if (voice_name.find("..") != std::string::npos || voice_name.front() == '/' || voice_name.front() == '~' ||
                voice_name.find('\\') != std::string::npos || voice_name.find('\0') != std::string::npos) {
                json_error(res, 400, "'voice' must not contain '..', a leading '/' or '~', or path separators",
                           "invalid_voice", "voice");
                return;
            }
            rp.tts_voice = voice_name;
        }
        if (!ref_text.empty())
            rp.tts_ref_text = ref_text;
        rp.tts_pad_silence_ms = tts_pad_silence_ms;
        if (!instructions.empty())
            rp.tts_instruct = instructions;
        if (!tts_phonemes.empty()) {
            if (!crispasr_phonemes_policy::backend_supports(backend_name)) {
                json_error(res, 400, crispasr_phonemes_policy::unsupported_message(backend_name),
                           "unsupported_phonemes", "phonemes");
                return;
            }
            rp.tts_phonemes = tts_phonemes;
        }
        // #249/#304: forward a target language to the TTS backend. MOSS-TTS sets
        // its "- Language:" prompt field from it; CosyVoice3 compares it to the
        // voice's own language to engage cross-lingual synthesis (dropping the
        // reference transcript so the clone speaks the target language, not the
        // reference's). Language-agnostic backends ignore it. Without this a
        // SubtitleEdit-style client could not pick the target language, so a
        // cross-lingual clone came out heavily accented. `target_lang` is an alias.
        {
            const std::string tts_lang = body.value("language", body.value("target_lang", std::string()));
            if (!tts_lang.empty())
                rp.language = tts_lang;
            // #329: and the language the REFERENCE clip is spoken in. Without it
            // CosyVoice3 has to infer that from the reference transcript, which
            // only ever answered for non-Latin scripts — so an English reference
            // plus "language": "de" silently stayed zero-shot and came back with
            // the English accent the caller was trying to get rid of.
            // `ref_lang` is an alias.
            const std::string ref_lang = body.value("source_lang", body.value("ref_lang", std::string()));
            if (!ref_lang.empty())
                rp.source_lang = ref_lang;
        }
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
        if (body.contains("min_speech_tokens") && body["min_speech_tokens"].is_number_integer())
            rp.tts_min_speech_tokens = body["min_speech_tokens"].get<int>();

        // Wire speed into params so backends with native duration control
        // (CAP_TTS_SPEED, e.g. kokoro) can use it directly.
        // The post-synth resampler below applies as a fallback for backends without native speed.
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
        const bool backend_handles_speed = backend->handles_tts_speed();

        // 75e: streaming mode — synthesize per-sentence and push each
        // chunk to the client as raw PCM via chunked transfer encoding.
        // The client receives int16 LE mono PCM at sr_out. This is the
        // same binary format as response_format=pcm, but arrives
        // incrementally. Speed resampling is still applied per-chunk.
        if (stream) {
            // Streaming only supports PCM formats (wav/pcm/f32).
            // mp3/aac/opus require full-file encoding — reject with 400.
            if (response_format == "mp3" || response_format == "aac" || response_format == "opus") {
                json_error(res, 400,
                           "streaming is not supported with response_format='" + response_format +
                               "'; use 'pcm', 'wav', or 'f32', or set stream=false",
                           "invalid_request_error", "response_format");
                return;
            }
            // Pre-compute 200ms silence gap between chunks
            const int silence_n = sr_out / 5;
            std::vector<short> silence_s16(silence_n, 0);

            // Report the marking decision before the first chunk goes out — same
            // headers as the buffered path (set_marking_headers is defined with
            // the decision above).
            set_marking_headers(res);

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
            auto push_pcm = [sq, sr_out, speed, backend_handles_speed, enqueue](const float* pcm, int n_samples) {
                if (!pcm || n_samples <= 0)
                    return;
                std::vector<float> chunk(pcm, pcm + n_samples);
                if (!backend_handles_speed && speed != 1.0f) {
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
                // force: a raw PCM stream has no container, so no manifest can
                // ride along and the watermark is the only mark available.
                // Matches the CLI's --tts-stream floor.
                crispasr_wm_dispatch::embed(chunk.data(), (int)chunk.size(), sr_out, /*force=*/true);
                enqueue(crispasr_make_pcm_int16_le(chunk.data(), (int)chunk.size()));
            };

            // Worker thread: synthesize all sentences, enqueueing chunks as
            // they are produced. Captures by value the bits it needs so it
            // outlives the handler scope (the provider keeps `sq` alive).
            std::thread worker([&backend, &model_mutex, sentences, rp, apply_spoken_disclaimer, silence_s16,
                                true_streaming, push_pcm, enqueue, sq, t0]() {
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
                    // Same rule as the buffered path: a clone is disclaimed
                    // unless the opt-out was attested. (Before #312 this keyed
                    // off is_voice_clone alone, so streaming ignored an
                    // honoured "spoken_disclaimer": false entirely.)
                    if (apply_spoken_disclaimer) {
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
        // Skipped when "spoken_disclaimer": false AND the opt-out was attested;
        // watermark + C2PA provenance remain regardless. #312: an unattested
        // opt-out lands here as apply_spoken_disclaimer=true (denied, not
        // refused) — the headers below tell the client which it got.
        if (apply_spoken_disclaimer) {
            crispasr_tts_prepend_disclaimer(pcm, backend.get(), rp);
        }
        set_marking_headers(res);

        // Apply speed via linear-interpolation resampler when the backend does
        // not natively scale duration (CAP_TTS_SPEED). Backends declaring
        // CAP_TTS_SPEED handle rp.tts_speed internally during synthesis,
        // preserving natural pitch and avoiding double-scaling.
        if (!backend_handles_speed && speed != 1.0f) {
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
        //
        // Watertight floor, per response: when this container cannot carry a
        // C2PA manifest the watermark is the only machine-readable mark there
        // is, so --no-watermark must not strip it. The CLI has always forced it
        // back on in that case; the server did not, so an attested
        // --no-watermark plus response_format=mp3/aac/opus/pcm/f32 returned
        // fully unmarked synthetic audio.
        const crispasr_marking::ContainerMarking cmark =
            crispasr_marking::container_marking_for_format(response_format);
        crispasr_wm_dispatch::embed(pcm.data(), (int)pcm.size(), sr_out, /*force=*/!cmark.carries_c2pa);

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double audio_s = (double)pcm.size() / (double)sr_out;
        fprintf(stderr,
                "crispasr-server: synthesized %.1fs audio in %.2fs (RTF=%.2f) "
                "voice='%s' speed=%.2f format=%s model='%s' chunks=%zu sr=%dHz\n",
                audio_s, elapsed_s, elapsed_s > 0 ? elapsed_s / audio_s : 0.0,
                voice_name.empty() ? "<startup>" : log_sanitize(voice_name).c_str(), speed, response_format.c_str(),
                requested_model.empty() ? "<unset>" : log_sanitize(requested_model).c_str(), chunks.size(), sr_out);

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
        } else if (response_format == "mp3") {
            std::string mp3 = crispasr_make_mp3(pcm.data(), (int)pcm.size(), sr_out);
            if (mp3.empty()) {
                json_error(res, 500, "MP3 encoding failed", "encoding_failed");
                return;
            }
            // MP3 carries a manifest (ID3v2.4 GEOB) and the native signer has
            // handled it all along — signing was just hardcoded to the WAV
            // branch, so every non-WAV response shipped without provenance.
            if (!params.tts_no_c2pa)
                crispasr_c2pa_sign_auto(mp3, cmark.c2pa_mime, params.c2pa_cert, params.c2pa_key, params.cache_dir);
            res.set_content(std::move(mp3), "audio/mpeg");
        } else if (response_format == "aac") {
            std::string aac = crispasr_make_aac(pcm.data(), (int)pcm.size(), sr_out);
            if (aac.empty()) {
                json_error(res, 500, "AAC encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(aac), "audio/aac");
        } else if (response_format == "opus") {
            const char* ct = "audio/ogg";
            std::string opus = crispasr_opus_response(pcm.data(), (int)pcm.size(), sr_out, &ct);
            if (opus.empty()) {
                json_error(res, 500, "Opus encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(opus), ct);
        } else {
            std::string wav = crispasr_make_wav_int16(pcm.data(), (int)pcm.size(), sr_out);
            // C2PA Content Credentials signing (when c2pa-c is available
            // and --c2pa-cert / --c2pa-key are configured)
            if (!params.tts_no_c2pa) // --no-c2pa: attested opt-out of the manifest floor
                crispasr_c2pa_sign_auto(wav, cmark.c2pa_mime, params.c2pa_cert, params.c2pa_key, params.cache_dir);
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
    //   200 audio/wav — output audio at backend's native TTS/S2S sample rate
    //   Header X-Transcript: <URL-encoded intermediate ASR transcript>
    //   400 — backend lacks CAP_S2S, missing file
    //   500 — S2S returned empty audio
    //   503 — model still loading
    //
    // Supported backends: lfm2-audio, mini-omni2, sidon, voxcpm2-vae
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
                           "load lfm2-audio, mini-omni2, sidon, or voxcpm2-vae via POST /load");
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

        // Watermark the output, with the same per-response watertight floor as
        // /v1/audio/speech: force the mark on when the requested container
        // cannot carry a C2PA manifest.
        const crispasr_marking::ContainerMarking cmark =
            crispasr_marking::container_marking_for_format(response_format);
        crispasr_wm_dispatch::embed(pcm.data(), (int)pcm.size(), sr_out, /*force=*/!cmark.carries_c2pa);

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
        } else if (response_format == "mp3") {
            std::string mp3 = crispasr_make_mp3(pcm.data(), (int)pcm.size(), sr_out);
            if (mp3.empty()) {
                json_error(res, 500, "MP3 encoding failed", "encoding_failed");
                return;
            }
            if (!params.tts_no_c2pa)
                crispasr_c2pa_sign_auto(mp3, cmark.c2pa_mime, params.c2pa_cert, params.c2pa_key, params.cache_dir);
            res.set_content(std::move(mp3), "audio/mpeg");
        } else if (response_format == "aac") {
            std::string aac = crispasr_make_aac(pcm.data(), (int)pcm.size(), sr_out);
            if (aac.empty()) {
                json_error(res, 500, "AAC encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(aac), "audio/aac");
        } else if (response_format == "opus") {
            const char* ct = "audio/ogg";
            std::string opus = crispasr_opus_response(pcm.data(), (int)pcm.size(), sr_out, &ct);
            if (opus.empty()) {
                json_error(res, 500, "Opus encoding failed", "encoding_failed");
                return;
            }
            res.set_content(std::move(opus), ct);
        } else {
            std::string wav = crispasr_make_wav_int16(pcm.data(), (int)pcm.size(), sr_out);
            if (!params.tts_no_c2pa) // --no-c2pa: attested opt-out of the manifest floor
                crispasr_c2pa_sign_auto(wav, cmark.c2pa_mime, params.c2pa_cert, params.c2pa_key, params.cache_dir);
            res.set_content(std::move(wav), "audio/wav");
        }
    });

    // -----------------------------------------------------------------------
    // GET /v1/voices — list voices in --voice-dir (CAP_TTS only)
    // Also aliased on /voices and /v1/audio/voices for llama-swap and
    // OpenAI-client compatibility (#264).
    // Returns: {"voices": [{"name": "<stem>", "format": "wav"|"gguf"}, ...]}
    // -----------------------------------------------------------------------
    auto handle_list_voices = [&](const Request& req, Response& res) {
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
    };
    svr.Get("/v1/voices", handle_list_voices);
    svr.Get("/voices", handle_list_voices);
    svr.Get("/v1/audio/voices", handle_list_voices);

    // -----------------------------------------------------------------------
    // POST /v1/voices — upload a voice file (multipart: "voice" file +
    // "consent_attestation" + optional "name" field)
    // Returns 201 on success: {"name": "...", "format": "wav", "size_bytes": N}
    //
    // This is the network equivalent of --make-ref: it takes a recording of a
    // real person and stores it as a reusable voiceprint the server will clone
    // from. The project's rule is that baking IS the cloning step, which is why
    // --make-ref and all three Python voice bakers demand --i-have-rights — so
    // the attestation is taken HERE, where the recording enters the system, not
    // only at /v1/audio/speech where it is replayed.
    //
    // It shipped without one: an upload was accepted from anyone who could
    // reach the endpoint, and the only trace it left was a byte count. The
    // synthesis gate did catch the resulting clone (a bare name resolves to
    // <voice-dir>/<name>.wav and scores as a recording-reference), so this was
    // never unmarked output — but consent for a third party's voice was never
    // asked for, and no [CONSENT] line existed to show it had been.
    // -----------------------------------------------------------------------
    svr.Post("/v1/voices", [&](const Request& req, Response& res) {
        // Mint the per-request correlation id first, so every audit record this
        // handler emits carries it. Without it a [CONSENT] line and the response
        // it authorised are unlinkable on a server serving many requests.
        crispasr_consent::new_request_id();
        res.set_header("X-Crispasr-Request-Id", crispasr_consent::request_correlation_id());
        if (!require_auth(req, res))
            return;
        // CAP_TTS gate (documented CAP_TTS-only; matches the GET /v1/voices guard).
        if (!(backend->capabilities() & CAP_TTS)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support TTS (no CAP_TTS); load a TTS backend via POST /load");
            return;
        }
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

        // Speaker-consent gate. Hard refusal, matching /v1/audio/speech rather
        // than #312's deny-the-opt-out rule: #312 is about a MARKING opt-out,
        // where serving the stronger default is always available. There is no
        // safe default for "may I keep a recording of this person's voice" —
        // either the attestation exists or the upload must not happen.
        const std::string upload_consent =
            req.has_file("consent_attestation") ? req.get_file_value("consent_attestation").content : std::string();
        if (upload_consent.empty()) {
            json_error(res, 400,
                       "uploading a voice reference requires a 'consent_attestation' form field. "
                       "Storing a recording as a reusable voiceprint is the cloning step itself. "
                       "This field should contain a statement attesting that you have the consent "
                       "of the speaker whose voice this is, or that it is your own voice. "
                       "Example: consent_attestation=\"I have the speaker's consent\"",
                       "consent_required", "consent_attestation");
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
        // Audit trail for the enrollment itself, in the same [CONSENT] format
        // the CLI and /v1/audio/speech emit — so "when was this voiceprint
        // created and on whose attestation" is answerable from the log, not
        // just "what was synthesized with it afterwards".
        {
            char ts[64];
            auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
            // Hash the UPLOADED BYTES, not the file that was written: this is
            // the recording the attestation is about, and hashing it here binds
            // the two before anything on disk can be swapped.
            crispasr_consent::emit(
                "CONSENT", ts,
                {{"scope", "voice-upload"},
                 {"voice", log_sanitize(voice_name)},
                 {"clone_reason", "recording-reference"},
                 {"bytes", std::to_string(voice_file.content.size())},
                 {"attestation", log_sanitize(upload_consent), /*quoted=*/true},
                 {"ref_sha256", crispasr_consent::bytes_sha256(voice_file.content.data(), voice_file.content.size())},
                 {"req", crispasr_consent::request_correlation_id()}});
        }
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
        // CAP_TTS gate (documented CAP_TTS-only; matches the GET /v1/voices guard).
        if (!(backend->capabilities() & CAP_TTS)) {
            json_error(res, 400,
                       "loaded backend '" + backend_name +
                           "' does not support TTS (no CAP_TTS); load a TTS backend via POST /load");
            return;
        }
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
    // for params.chat_model). Overlapping requests queue on chat_call_mutex
    // below — one whole request at a time, not one C call at a time.
    // -----------------------------------------------------------------------
    std::shared_ptr<crispasr_chat_session> chat_sess(nullptr, &crispasr_chat_close);
    std::mutex chat_init_mutex;
    // One /v1/chat/completions request at a time on the process-wide session.
    // The session's own mutex serialises each C call, but a request is two of
    // them — reset, then generate — and the KV cache they share is
    // session-global. A second request whose reset lands between this one's
    // reset and its generate makes this one prefill onto a cache it did not
    // clear, and answer with the other request's history in context. The whole
    // transaction takes one lock.
    std::mutex chat_call_mutex;
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
        std::lock_guard<std::mutex> chat_guard(chat_call_mutex);

        crispasr_chat_error rerr{};
        if (crispasr_chat_reset(s, &rerr) != 0) {
            json_error(res, 500, std::string("chat reset failed: ") + rerr.message, "chat_reset_failed");
            return;
        }

        // EU AI Act Art. 50(2) marking for synthetic TEXT. There is no
        // watermark-equivalent that survives a copy-paste of a sentence, so the
        // Commission's own guidance points at metadata travelling with the
        // response — which for an HTTP API means headers. Set on both the
        // buffered and the SSE branch, before either writes a body.
        //
        // This is weaker than the audio case and the docs say so (§6.6): a
        // client that drops the headers publishes unmarked text, and marking
        // what you then do with it stays the deployer's duty. Weak marking that
        // travels is still strictly better than none, and it costs two headers.
        res.set_header("X-Crispasr-Ai-Generated", "true");
        res.set_header("X-Crispasr-Ai-Disclosure", crispasr_chat_ai_disclosure_text());

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
        fprintf(stderr, "       /voices, /v1/audio/voices   — aliases (llama-swap compat)\n");
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
    if (sep_ctx) {
        const char* arch_name = sep_ctx->arch == SepArch::HTDEMUCS ? "htdemucs" : "mel-band-roformer";
        fprintf(stderr, "  POST /v1/audio/separation         — source separation (%s, %d stems)\n", arch_name,
                sep_ctx->n_sources());
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
        whisper_params realtime_params = params;
        if (params.vad || !params.vad_model.empty())
            realtime_params.vad_model = crispasr_resolve_vad_model(params);
        if (realtime_server_start(backend.get(), model_mutex, realtime_params, rt_port) == 0) {
            rt_started = true;
            fprintf(stderr, "  WS   ws://%s:%d/v1/realtime     — vLLM Realtime API (JSON WebSocket)\n", host.c_str(),
                    rt_port);
            if ((params.vad || !params.vad_model.empty()) && realtime_params.vad_model.empty())
                fprintf(stderr, "crispasr-server: warning: realtime VAD requested but its model could not be resolved; "
                                "/v1/realtime will require client commits\n");
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

    // §381: free the separation context.
    if (sep_ctx) {
        if (sep_ctx->htd_ctx)
            htdemucs_free(sep_ctx->htd_ctx);
        if (sep_ctx->mbr_ctx)
            mel_band_roformer_free(sep_ctx->mbr_ctx);
        sep_ctx.reset();
    }

    return 0;
}
