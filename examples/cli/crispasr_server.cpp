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
//   POST /load                        — hot-swap model
//   GET  /health                      — server status
//   GET  /backends                    — list available backends
//   GET  /v1/models                   — OpenAI-compatible model list
//   GET  /v1/voices                   — list voices in --voice-dir (CAP_TTS only)
//
// Adapted from examples/server/server.cpp for multi-backend support.

#include "crispasr_backend.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_vad_cli.h"
#include "whisper_params.h"

#include "common-crispasr.h" // read_audio_data
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
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h> // _mktemp_s
#include <windows.h>
#else
#include <unistd.h> // mkstemp, close, unlink
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Create a temporary file securely via mkstemp (POSIX) or _mktemp_s (Win).
// Writes `data` to it and returns the path. On failure returns "".
// The caller is responsible for calling std::remove() on the returned path.
static std::string write_temp_audio(const char* data, size_t size) {
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, tmp_dir))
        return "";
    char tmp_path[MAX_PATH];
    if (!GetTempFileNameA(tmp_dir, "cra", 0, tmp_path))
        return "";
    std::ofstream f(tmp_path, std::ios::binary);
    if (!f)
        return "";
    f.write(data, (std::streamsize)size);
    f.close();
    return std::string(tmp_path);
#else
    char tmpl[] = "/tmp/crispasr-XXXXXX";
    int fd = mkstemp(tmpl);
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
            ::unlink(tmpl);
            return "";
        }
        p += n;
        remaining -= (size_t)n;
    }
    ::close(fd);
    return std::string(tmpl);
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
    double duration_s = 0.0;
    double elapsed_s = 0.0;
};

// Load audio from a multipart file upload, transcribe it, return result.
// Acquires model_mutex internally.
static transcription_result do_transcribe(const httplib::MultipartFormData& audio_file, CrispasrBackend* backend,
                                          std::mutex& model_mutex, const whisper_params& rp) {
    transcription_result result;

    if (rp.verbose)
        fprintf(stderr, "crispasr-server: processing '%s' (%zu bytes)\n", audio_file.filename.c_str(),
                audio_file.content.size());

    // Write to a secure temporary file for audio decoding.
    std::string tmp_path = write_temp_audio(audio_file.content.data(), audio_file.content.size());
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

    // Auto-chunk long audio to prevent OOM (#27).
    // Most backends have O(T²) attention in the encoder — 30s chunks keep
    // memory bounded. The CLI does this via --vad / --chunk-seconds.
    const int SR = 16000;
    const int max_chunk_samples = rp.chunk_seconds * SR; // default 30s = 480000
    const int n_samples = (int)pcmf32.size();

    {
        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::steady_clock::now();

        if (n_samples <= max_chunk_samples) {
            // Short audio — single pass
            result.segs = backend->transcribe(pcmf32.data(), n_samples, 0, rp);
        } else {
            // Chunk long audio into fixed segments
            int n_chunks = (n_samples + max_chunk_samples - 1) / max_chunk_samples;
            fprintf(stderr, "crispasr-server: chunking %.1fs audio into %d × %ds segments\n", result.duration_s,
                    n_chunks, rp.chunk_seconds);
            int chunk_idx = 0;
            for (int offset = 0; offset < n_samples; offset += max_chunk_samples) {
                int chunk_len = std::min(max_chunk_samples, n_samples - offset);
                int64_t t_offset_cs = (int64_t)((double)offset / SR * 100.0);
                auto tc0 = std::chrono::steady_clock::now();
                auto chunk_segs = backend->transcribe(pcmf32.data() + offset, chunk_len, t_offset_cs, rp);
                auto tc1 = std::chrono::steady_clock::now();
                double chunk_s = std::chrono::duration<double>(tc1 - tc0).count();
                result.segs.insert(result.segs.end(), chunk_segs.begin(), chunk_segs.end());
                chunk_idx++;
                fprintf(stderr, "crispasr-server: chunk %d/%d done (%.1fs audio in %.1fs)\n", chunk_idx, n_chunks,
                        chunk_len / (double)SR, chunk_s);
            }
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
// Server entry point
// ---------------------------------------------------------------------------

int crispasr_run_server(whisper_params& params, const std::string& host, int port) {
    using namespace httplib;

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

        const std::string resolved = crispasr_resolve_model_cli(
            params.model, backend_name, params.no_prints, params.cache_dir, params.auto_download || model_is_auto);
        if (resolved.empty()) {
            fprintf(stderr, "crispasr-server: failed to resolve model '%s' for backend '%s'\n", params.model.c_str(),
                    backend_name.c_str());
            return 1;
        }
        params.model = resolved;

        backend = crispasr_create_backend(backend_name);
        if (!backend || !backend->init(params)) {
            fprintf(stderr, "crispasr-server: failed to init backend '%s'\n", backend_name.c_str());
            return 1;
        }
        ready.store(true);
        fprintf(stderr, "crispasr-server: backend '%s' loaded, model '%s'\n", backend_name.c_str(),
                params.model.c_str());
    }

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

        auto result = do_transcribe(audio_file, backend.get(), model_mutex, rp);
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
    //   timestamp_granularities[] (optional) — word|segment (verbose_json)
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

        // Parse OpenAI form fields.
        std::string response_format = form_string(req, "response_format", "json");
        std::string language = form_string(req, "language", params.language);
        std::string prompt = form_string(req, "prompt", "");
        float temperature = form_float(req, "temperature", params.temperature);

        // Validate response_format early.
        if (response_format != "json" && response_format != "verbose_json" && response_format != "text" &&
            response_format != "srt" && response_format != "vtt") {
            json_error(res, 400,
                       "invalid response_format '" + response_format +
                           "'; must be one of: json, verbose_json, text, srt, vtt");
            return;
        }

        // Build per-request params.
        whisper_params rp = params;
        rp.language = language;
        rp.temperature = temperature;
        if (!prompt.empty())
            rp.prompt = prompt;

        auto result = do_transcribe(audio_file, backend.get(), model_mutex, rp);
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
            res.set_content(
                crispasr_segments_to_openai_verbose_json(result.segs, result.duration_s, language, task, temperature),
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

        const std::string resolved_model = crispasr_resolve_model_cli(
            new_model, new_backend, params.no_prints, params.cache_dir, params.auto_download || new_model_is_auto);
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
    // POST /v1/audio/speech — OpenAI-compatible TTS endpoint
    //
    // Body: application/json
    //   {
    //     "input":           "TEXT to synthesize",       (required)
    //     "model":           "<model id>",               (optional, ignored — we serve the loaded one)
    //     "voice":           "<name in --voice-dir>",    (optional)
    //     "instructions":    "<voice direction prose>",  (optional, applied via params.tts_instruct)
    //     "speed":           0.25 .. 4.0,                (optional, default 1.0)
    //     "response_format": "wav" | "pcm" | "f32"       (optional, default "wav")
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
        std::string instructions = body.value("instructions", "");
        std::string response_format = body.value("response_format", std::string("wav"));
        if (response_format != "wav" && response_format != "pcm" && response_format != "f32") {
            json_error(res, 400, "response_format must be one of 'wav', 'pcm', or 'f32'", "unsupported_response_format",
                       "response_format");
            return;
        }

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

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> pcm;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            pcm = backend->synthesize(text, rp);
        }
        auto t1 = std::chrono::steady_clock::now();

        if (pcm.empty()) {
            json_error(res, 500, "synthesis failed (backend returned empty audio)", "synthesis_failed");
            return;
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

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double audio_s = (double)pcm.size() / 24000.0;
        fprintf(stderr,
                "crispasr-server: synthesized %.1fs audio in %.2fs (RTF=%.2f) "
                "voice='%s' speed=%.2f format=%s model='%s'\n",
                audio_s, elapsed_s, elapsed_s > 0 ? elapsed_s / audio_s : 0.0,
                voice_name.empty() ? "<startup>" : voice_name.c_str(), speed, response_format.c_str(),
                requested_model.empty() ? "<unset>" : requested_model.c_str());

        if (response_format == "f32") {
            std::string buf((const char*)pcm.data(), pcm.size() * sizeof(float));
            res.set_content(std::move(buf), "application/octet-stream");
        } else if (response_format == "pcm") {
            // OpenAI's pcm: 24 kHz signed 16-bit LE mono raw bytes, no
            // header. Content-Type is documented as audio/pcm; clients
            // know the rate out-of-band from the spec.
            std::string raw = crispasr_make_pcm_int16_le(pcm.data(), (int)pcm.size());
            res.set_content(std::move(raw), "audio/pcm");
        } else {
            std::string wav = crispasr_make_wav_int16(pcm.data(), (int)pcm.size(), 24000);
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

    // Start
    // -----------------------------------------------------------------------
    const bool tts = (backend->capabilities() & CAP_TTS) != 0;
    fprintf(stderr, "\ncrispasr-server: listening on %s:%d\n", host.c_str(), port);
    fprintf(stderr, "  POST /inference                  — upload audio (native JSON)\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions    — OpenAI-compatible API\n");
    if (tts) {
        fprintf(stderr, "  POST /v1/audio/speech            — TTS (OpenAI-compatible)\n");
    }
    fprintf(stderr, "  POST /load                       — hot-swap model\n");
    fprintf(stderr, "  GET  /health                     — server status\n");
    fprintf(stderr, "  GET  /backends                   — list backends\n");
    fprintf(stderr, "  GET  /v1/models                  — model info\n");
    if (tts) {
        fprintf(stderr, "  GET  /v1/voices                  — list voices in --voice-dir\n");
        if (params.tts_voice_dir.empty()) {
            fprintf(stderr, "crispasr-server: warning: --voice-dir not set; /v1/voices will return empty "
                            "and /v1/audio/speech will reject requests with a 'voice' field\n");
        }
    }
    fprintf(stderr, "\n");
    if (!api_keys.empty())
        fprintf(stderr, "crispasr-server: API key authentication enabled\n");

    svr.listen(host, port);
    return 0;
}
