// crispasr_server.cpp — HTTP server with persistent model for all backends.
//
// Keeps the model loaded in memory between requests. Accepts audio via
// POST /inference (multipart file upload) and returns JSON transcription.
//
// Usage:
//   crispasr --server -m model.gguf [--port 8080] [--host 127.0.0.1]
//
// Endpoints:
//   POST /inference  — transcribe uploaded audio file
//   POST /load       — hot-swap model (body: {"model": "path/to/model.gguf"})
//   GET  /health     — server status
//   GET  /backends   — list available backends
//
// Adapted from examples/server/server.cpp for multi-backend support.

#include "crispasr_backend.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr.h"
#include "crispasr_vad.h"
#include "whisper_params.h"

#include "common-whisper.h" // read_audio_data
#include "../server/httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <iomanip>

// Minimal JSON builder (avoid nlohmann dep for the server extension)
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
        }
    }
    return out;
}

static std::string segments_to_json(const std::vector<crispasr_segment>& segs, const std::string& backend_name,
                                    double duration_s) {
    std::ostringstream js;
    js << "{\n";
    js << "  \"backend\": \"" << json_escape(backend_name) << "\",\n";
    js << "  \"duration\": " << duration_s << ",\n";
    js << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& s = segs[i];
        js << "    {\n";
        js << "      \"t0\": " << (s.t0 * 10) << ",\n"; // ms
        js << "      \"t1\": " << (s.t1 * 10) << ",\n";
        js << "      \"text\": \"" << json_escape(s.text) << "\"";
        if (!s.speaker.empty()) {
            js << ",\n      \"speaker\": \"" << json_escape(s.speaker) << "\"";
        }
        if (!s.tokens.empty()) {
            js << ",\n      \"tokens\": [\n";
            for (size_t j = 0; j < s.tokens.size(); j++) {
                const auto& t = s.tokens[j];
                js << "        {\"text\": \"" << json_escape(t.text) << "\"";
                if (t.confidence >= 0)
                    js << ", \"confidence\": " << t.confidence;
                if (t.t0 >= 0)
                    js << ", \"t0\": " << (t.t0 * 10);
                if (t.t1 >= 0)
                    js << ", \"t1\": " << (t.t1 * 10);
                js << "}";
                if (j + 1 < s.tokens.size())
                    js << ",";
                js << "\n";
            }
            js << "      ]";
        }
        js << "\n    }";
        if (i + 1 < segs.size())
            js << ",";
        js << "\n";
    }
    js << "  ],\n";
    // Full text
    std::string full_text;
    for (const auto& s : segs) {
        if (!full_text.empty())
            full_text += " ";
        full_text += s.text;
    }
    js << "  \"text\": \"" << json_escape(full_text) << "\"\n";
    js << "}\n";
    return js.str();
}

// Helpers for OpenAI compatibility
static std::string format_timestamp_openai(int64_t t) {
    // t is in centiseconds (1/100th)
    double sec = t / 100.0;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << sec;
    return ss.str();
}

static std::string segments_to_srt(const std::vector<crispasr_segment>& segs) {
    std::ostringstream out;
    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];
        int64_t t0 = s.t0 * 10; // ms
        int64_t t1 = s.t1 * 10; // ms
        auto format_time = [](int64_t t) {
            int h = t / 3600000; t %= 3600000;
            int m = t / 60000; t %= 60000;
            int sec = t / 1000; t %= 1000;
            char buf[64];
            snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", h, m, sec, (int)t);
            return std::string(buf);
        };
        out << i + 1 << "\n" << format_time(t0) << " --> " << format_time(t1) << "\n" << s.text << "\n\n";
    }
    return out.str();
}

static std::string segments_to_vtt(const std::vector<crispasr_segment>& segs) {
    std::ostringstream out;
    out << "WEBVTT\n\n";
    for (const auto& s : segs) {
        int64_t t0 = s.t0 * 10;
        int64_t t1 = s.t1 * 10;
        auto format_time = [](int64_t t) {
            int h = t / 3600000; t %= 3600000;
            int m = t / 60000; t %= 60000;
            int sec = t / 1000; t %= 1000;
            char buf[64];
            snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", h, m, sec, (int)t);
            return std::string(buf);
        };
        out << format_time(t0) << " --> " << format_time(t1) << "\n" << s.text << "\n\n";
    }
    return out.str();
}

static std::string segments_to_openai_verbose(const std::vector<crispasr_segment>& segs, double duration_s) {
    std::ostringstream js;
    std::string full_text;
    for (const auto& s : segs) {
        if (!full_text.empty()) full_text += " ";
        full_text += s.text;
    }
    js << "{\n";
    js << "  \"task\": \"transcribe\",\n";
    js << "  \"language\": \"english\",\n";
    js << "  \"duration\": " << duration_s << ",\n";
    js << "  \"text\": \"" << json_escape(full_text) << "\",\n";
    js << "  \"segments\": [\n";
    for (size_t i = 0; i < segs.size(); i++) {
        const auto& s = segs[i];
        js << "    {\n";
        js << "      \"id\": " << i << ",\n";
        js << "      \"start\": " << (s.t0 / 100.0) << ",\n";
        js << "      \"end\": " << (s.t1 / 100.0) << ",\n";
        js << "      \"text\": \"" << json_escape(s.text) << "\"\n";
        js << "    }" << (i + 1 < segs.size() ? "," : "") << "\n";
    }
    js << "  ]\n";
    js << "}\n";
    return js.str();
}

int crispasr_run_server(whisper_params& params, const std::string& host, int port) {
    using namespace httplib;

    std::unique_ptr<CrispasrBackend> backend;
    std::mutex model_mutex;
    std::atomic<bool> ready{false};
    std::string backend_name = params.backend;

    // Initial model load
    {
        if (backend_name.empty() || backend_name == "auto") {
            backend_name = crispasr_detect_backend_from_gguf(params.model);
        }
        if (backend_name.empty()) {
            fprintf(stderr, "crispasr-server: cannot detect backend from '%s'\n", params.model.c_str());
            return 1;
        }
        backend = crispasr_create_backend(backend_name);
        if (!backend || !backend->init(params)) {
            fprintf(stderr, "crispasr-server: failed to init backend '%s'\n", backend_name.c_str());
            return 1;
        }
        ready.store(true);
        fprintf(stderr, "crispasr-server: backend '%s' loaded, model '%s'\n", backend_name.c_str(),
                params.model.c_str());
    }

    // Audio hash cache for encoder output reuse (same audio → skip re-encode)
    struct AudioCache {
        size_t hash = 0;
        std::vector<crispasr_segment> segs;
    };
    AudioCache last_cache;

    Server svr;

    // POST /inference — transcribe uploaded audio
    svr.Post("/inference", [&](const Request& req, Response& res) {
        if (!ready.load()) {
            res.status = 503;
            res.set_content("{\"error\": \"model loading\"}", "application/json");
            return;
        }

        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\": \"no 'file' field\"}", "application/json");
            return;
        }

        auto audio_file = req.get_file_value("file");
        fprintf(stderr, "crispasr-server: received '%s' (%zu bytes)\n", audio_file.filename.c_str(),
                audio_file.content.size());

        // Write to temp file for audio loading
        std::string tmp_path = "/tmp/crispasr-server-" +
                               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".wav";
        {
            std::ofstream f(tmp_path, std::ios::binary);
            f.write(audio_file.content.data(), audio_file.content.size());
        }

        // Load audio
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(tmp_path, pcmf32, pcmf32s, params.diarize)) {
            std::remove(tmp_path.c_str());
            res.status = 400;
            res.set_content("{\"error\": \"failed to read audio\"}", "application/json");
            return;
        }
        std::remove(tmp_path.c_str());

        double duration_s = (double)pcmf32.size() / 16000.0;

        // Override per-request params from form fields
        whisper_params rp = params;
        if (req.has_file("language"))
            rp.language = req.get_file_value("language").content;

        // Check audio cache (simple hash: size + first/middle/last samples)
        size_t audio_hash = pcmf32.size();
        if (!pcmf32.empty()) {
            union {
                float f;
                uint32_t u;
            } conv;
            conv.f = pcmf32[0];
            audio_hash ^= conv.u * 2654435761u;
            conv.f = pcmf32[pcmf32.size() / 2];
            audio_hash ^= conv.u * 40503u;
            conv.f = pcmf32.back();
            audio_hash ^= conv.u * 12345u;
        }

        // Transcribe (with cache check)
        std::lock_guard<std::mutex> lock(model_mutex);
        auto t0 = std::chrono::steady_clock::now();

        std::vector<crispasr_segment> segs;
        if (last_cache.hash == audio_hash && !last_cache.segs.empty()) {
            segs = last_cache.segs;
            fprintf(stderr, "crispasr-server: cache hit — reusing previous result\n");
        } else {
            segs = backend->transcribe(pcmf32.data(), (int)pcmf32.size(), 0, rp);
            last_cache.hash = audio_hash;
            last_cache.segs = segs;
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stderr, "crispasr-server: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", duration_s, elapsed,
                duration_s / elapsed);

        std::string json = segments_to_json(segs, backend_name, duration_s);
        res.set_content(json, "application/json");
    });

    // POST /v1/audio/transcriptions — OpenAI compatible endpoint
    svr.Post("/v1/audio/transcriptions", [&](const Request& req, Response& res) {
        if (!ready.load()) {
            res.status = 503;
            res.set_content("{\"error\": \"model loading\"}", "application/json");
            return;
        }
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content("{\"error\": \"no 'file' field\"}", "application/json");
            return;
        }

        auto audio_file = req.get_file_value("file");
        std::string response_format = req.has_file("response_format") ? req.get_file_value("response_format").content : "json";

        std::string tmp_path = "/tmp/crispasr-openai-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".wav";
        {
            std::ofstream f(tmp_path, std::ios::binary);
            f.write(audio_file.content.data(), audio_file.content.size());
        }

        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(tmp_path, pcmf32, pcmf32s, params.diarize)) {
            std::remove(tmp_path.c_str());
            res.status = 400;
            res.set_content("{\"error\": \"failed to read audio\"}", "application/json");
            return;
        }
        std::remove(tmp_path.c_str());
        double duration_s = (double)pcmf32.size() / 16000.0;

        whisper_params rp = params;
        if (req.has_file("language")) rp.language = req.get_file_value("language").content;
        if (req.has_file("prompt"))   rp.prompt = req.get_file_value("prompt").content;
        if (req.has_file("temperature")) rp.temperature = std::stof(req.get_file_value("temperature").content);

        std::vector<crispasr_segment> segs;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            segs = backend->transcribe(pcmf32.data(), (int)pcmf32.size(), 0, rp);
        }

        std::string full_text;
        for (const auto& s : segs) {
            if (!full_text.empty()) full_text += " ";
            full_text += s.text;
        }

        if (response_format == "verbose_json") {
            res.set_content(segments_to_openai_verbose(segs, duration_s), "application/json");
        } else if (response_format == "vtt") {
            res.set_content(segments_to_vtt(segs), "text/vtt");
        } else if (response_format == "srt") {
            res.set_content(segments_to_srt(segs), "text/plain");
        } else if (response_format == "text") {
            res.set_content(full_text, "text/plain");
        } else {
            // Default: json
            res.set_content("{\"text\": \"" + json_escape(full_text) + "\"}", "application/json");
        }
    });

    // POST /load — hot-swap model
    svr.Post("/load", [&](const Request& req, Response& res) {
        std::lock_guard<std::mutex> lock(model_mutex);
        ready.store(false);

        std::string new_model;
        std::string new_backend;
        if (req.has_file("model"))
            new_model = req.get_file_value("model").content;
        if (req.has_file("backend"))
            new_backend = req.get_file_value("backend").content;

        if (new_model.empty()) {
            ready.store(true);
            res.status = 400;
            res.set_content("{\"error\": \"no 'model' field\"}", "application/json");
            return;
        }

        if (new_backend.empty())
            new_backend = crispasr_detect_backend_from_gguf(new_model);

        whisper_params np = params;
        np.model = new_model;
        np.backend = new_backend;

        auto nb = crispasr_create_backend(new_backend);
        if (!nb || !nb->init(np)) {
            ready.store(true); // keep old model
            res.status = 500;
            res.set_content("{\"error\": \"failed to load new model\"}", "application/json");
            return;
        }

        backend = std::move(nb);
        backend_name = new_backend;
        params.model = new_model;
        ready.store(true);

        fprintf(stderr, "crispasr-server: hot-swapped to '%s' backend, model '%s'\n", new_backend.c_str(),
                new_model.c_str());
        res.set_content("{\"status\": \"ok\", \"backend\": \"" + new_backend + "\"}", "application/json");
    });

    // GET /health
    svr.Get("/health", [&](const Request&, Response& res) {
        if (ready.load()) {
            res.set_content("{\"status\": \"ok\", \"backend\": \"" + backend_name + "\"}", "application/json");
        } else {
            res.status = 503;
            res.set_content("{\"status\": \"loading\"}", "application/json");
        }
    });

    // GET /backends
    svr.Get("/backends", [&](const Request&, Response& res) {
        auto names = crispasr_list_backends();
        std::ostringstream js;
        js << "{\"backends\": [";
        for (size_t i = 0; i < names.size(); i++) {
            if (i)
                js << ", ";
            js << "\"" << names[i] << "\"";
        }
        js << "], \"active\": \"" << backend_name << "\"}";
        res.set_content(js.str(), "application/json");
    });

    fprintf(stderr, "\ncrispasr-server: listening on %s:%d\n", host.c_str(), port);
    fprintf(stderr, "  POST /inference  — upload audio file\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions — OpenAI API\n");
    fprintf(stderr, "  POST /load       — hot-swap model\n");
    fprintf(stderr, "  GET  /health     — server status\n");
    fprintf(stderr, "  GET  /backends   — list backends\n\n");

    svr.listen(host, port);
    return 0;
}
