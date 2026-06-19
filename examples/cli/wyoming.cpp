// wyoming.cpp — Wyoming protocol TCP server (issue #172).
//
// Implements the Wyoming peer-to-peer JSONL protocol over raw TCP so that
// one crispasr-server instance can serve as both a wyoming-faster-whisper
// replacement (STT) and a wyoming-piper replacement (TTS) for Home Assistant.
//
// Architecture: mirrors ws_stream.cpp — a single listener thread accepts
// connections; each connection runs on its own thread. Backend calls are
// serialised with the server's model_mutex (same lock as HTTP requests).
//
// Wyoming wire format (per message):
//   {"type":"...","data":{...},"payload_length":N}\n
//   [N bytes of binary payload when payload_length > 0]
//
// ASR path: buffer incoming audio-chunk payloads (int16 LE PCM at the
// announced rate), resample to 16 kHz float32 after audio-stop, then call
// backend->transcribe() once and emit transcript.
//
// TTS path: on synthesize, call backend->synthesize() under model_mutex,
// convert float32 PCM → int16 LE, stream back as audio-chunk events.
//
// Resampling: linear interpolation — good enough for TTS preview and HA
// playback; Wyoming clients receive whatever rate we announce in audio-start
// and handle presentation resampling themselves.

#include "wyoming.h"

#include "../json.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSE_SOCKET closesocket
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define SOCKET_ERRNO WSAGetLastError()
#define SHUTDOWN_BOTH SD_BOTH
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET_VAL (-1)
#define CLOSE_SOCKET close
#define SOCKET_ERRNO errno
#define SHUTDOWN_BOTH SHUT_RDWR
#endif

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Socket helpers
// ---------------------------------------------------------------------------

namespace {

static bool recv_exact(socket_t fd, void* buf, size_t n) {
    auto* p = (uint8_t*)buf;
    while (n > 0) {
        auto r = recv(fd, (char*)p, (int)n, 0);
        if (r <= 0)
            return false;
        p += (size_t)r;
        n -= (size_t)r;
    }
    return true;
}

static bool send_all(socket_t fd, const void* buf, size_t n) {
    auto* p = (const uint8_t*)buf;
    while (n > 0) {
        auto r = send(fd, (const char*)p, (int)n, 0);
        if (r <= 0)
            return false;
        p += (size_t)r;
        n -= (size_t)r;
    }
    return true;
}

// Read one Wyoming message header (JSON line, terminated by '\n').
// Fills *type, *data, and *payload_length. Returns false on error/EOF.
static bool wyoming_read_header(socket_t fd, std::string& type, json& data, int& payload_length) {
    std::string line;
    char c;
    while (true) {
        auto r = recv(fd, &c, 1, 0);
        if (r <= 0)
            return false;
        if (c == '\n')
            break;
        if (line.size() > 65536)
            return false; // runaway header
        line += c;
    }
    if (line.empty())
        return false;
    int data_length = 0;
    try {
        auto j = json::parse(line);
        type = j.value("type", "");
        data = j.value("data", json::object());
        payload_length = j.value("payload_length", 0);
        data_length = j.value("data_length", 0);
    } catch (...) {
        return false;
    }
    // Wyoming clients (HA's `wyoming` lib) send a non-empty `data` object as a
    // separate length-prefixed JSON blob AFTER the header line, advertised via
    // `data_length` — not inline. Read and merge it before the binary payload,
    // otherwise the payload read desyncs the stream. (issue #172)
    if (data_length > 0) {
        if (data_length > 16 * 1024 * 1024)
            return false; // runaway data blob
        std::string db;
        db.resize((size_t)data_length);
        if (!recv_exact(fd, &db[0], (size_t)data_length))
            return false;
        try {
            auto dj = json::parse(db);
            if (data.is_object() && dj.is_object()) {
                for (auto it = dj.begin(); it != dj.end(); ++it)
                    data[it.key()] = it.value();
            } else {
                data = dj;
            }
        } catch (...) {
            return false;
        }
    }
    return !type.empty();
}

// Send a Wyoming message with no binary payload.
static bool wyoming_send(socket_t fd, const std::string& type, const json& data) {
    json msg;
    msg["type"] = type;
    msg["data"] = data;
    msg["payload_length"] = 0;
    std::string line = msg.dump() + "\n";
    return send_all(fd, line.data(), line.size());
}

// Send a Wyoming message with a binary payload.
static bool wyoming_send_payload(socket_t fd, const std::string& type, const json& data, const void* payload,
                                 int payload_len) {
    json msg;
    msg["type"] = type;
    msg["data"] = data;
    msg["payload_length"] = payload_len;
    std::string line = msg.dump() + "\n";
    return send_all(fd, line.data(), line.size()) && send_all(fd, payload, (size_t)payload_len);
}

// Linear resample `src` (n_src float32 frames at src_rate Hz, mono)
// → output (mono float32 at dst_rate Hz).
static std::vector<float> resample_f32(const float* src, int n_src, int src_rate, int dst_rate) {
    if (src_rate == dst_rate) {
        return std::vector<float>(src, src + n_src);
    }
    const int n_dst = (int)((double)n_src * dst_rate / src_rate + 0.5);
    std::vector<float> out((size_t)n_dst);
    const double ratio = (double)(n_src - 1) / std::max(n_dst - 1, 1);
    for (int i = 0; i < n_dst; i++) {
        double pos = i * ratio;
        int i0 = (int)pos;
        int i1 = std::min(i0 + 1, n_src - 1);
        float alpha = (float)(pos - (double)i0);
        out[(size_t)i] = src[i0] * (1.0f - alpha) + src[i1] * alpha;
    }
    return out;
}

// Convert float32 [-1,1] PCM → int16 LE PCM (in-place into output vector).
static std::vector<int16_t> f32_to_s16(const float* src, int n) {
    std::vector<int16_t> out((size_t)n);
    for (int i = 0; i < n; i++) {
        float v = src[i];
        if (v > 1.0f)
            v = 1.0f;
        if (v < -1.0f)
            v = -1.0f;
        out[(size_t)i] = (int16_t)(v * 32767.0f);
    }
    return out;
}

// Convert int16 LE PCM → float32 [-1,1].
static std::vector<float> s16_to_f32(const int16_t* src, int n) {
    std::vector<float> out((size_t)n);
    for (int i = 0; i < n; i++)
        out[(size_t)i] = (float)src[i] / 32768.0f;
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static std::atomic<bool> g_running{false};
static std::thread g_listener_thread;
static socket_t g_listen_fd = INVALID_SOCKET_VAL;
static CrispasrBackend* g_backend = nullptr;
static std::mutex* g_mutex = nullptr;
static whisper_params g_params;

// Open connection tracking, so wyoming_stop() can wake blocked recv() calls and
// wait for in-flight handlers to finish before the backend is torn down. (H3)
static std::mutex g_clients_mtx;
static std::vector<socket_t> g_clients;
static std::atomic<int> g_active{0};

// ---------------------------------------------------------------------------
// Per-connection handler
// ---------------------------------------------------------------------------

static void wyoming_handle_connection(socket_t fd) {
    // Session state across messages
    std::string asr_language = g_params.language.empty() ? "en" : g_params.language;
    int audio_rate = 16000;
    int audio_width = 2; // bytes per sample (int16 = 2)
    int audio_ch = 1;
    std::vector<int16_t> audio_buf; // accumulated raw int16 samples

    g_active.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_clients_mtx);
        g_clients.push_back(fd);
    }

    while (g_running.load()) {
        std::string type;
        json data;
        int payload_length = 0;

        if (!wyoming_read_header(fd, type, data, payload_length))
            break;

        // Read optional binary payload
        std::vector<uint8_t> payload;
        if (payload_length > 0) {
            if (payload_length > 16 * 1024 * 1024) { // 16 MB guard
                fprintf(stderr, "wyoming: oversized payload %d bytes — closing\n", payload_length);
                break;
            }
            payload.resize((size_t)payload_length);
            if (!recv_exact(fd, payload.data(), (size_t)payload_length))
                break;
        }

        // ----------------------------------------------------------------
        // describe → info
        // ----------------------------------------------------------------
        if (type == "describe") {
            const bool has_tts = (g_backend->capabilities() & CAP_TTS) != 0;
            // Advertise the configured language instead of a hardcoded "en", so
            // HA routes non-English audio to this service. (C4)
            const std::string adv_lang = g_params.language.empty() ? std::string("en") : g_params.language;
            const json langs = json::array({adv_lang});

            json asr_model;
            asr_model["name"] = g_backend->name();
            asr_model["description"] = std::string("CrispASR ") + g_backend->name() + " backend";
            asr_model["attribution"] = {{"name", "CrispASR"}, {"url", "https://github.com/CrispStrobe/CrispASR"}};
            asr_model["installed"] = true;
            asr_model["languages"] = langs;

            json asr_service;
            asr_service["name"] = "crispasr";
            asr_service["description"] = "CrispASR multi-backend speech recognition";
            asr_service["attribution"] = {{"name", "CrispASR"}, {"url", "https://github.com/CrispStrobe/CrispASR"}};
            asr_service["installed"] = true;
            asr_service["languages"] = langs;
            asr_service["models"] = json::array({asr_model});

            json info_data;
            info_data["asr"] = json::array({asr_service});

            if (has_tts) {
                json voice;
                voice["name"] = g_params.tts_voice.empty() ? "default" : g_params.tts_voice;
                voice["description"] = "";
                voice["attribution"] = {{"name", "CrispASR"}, {"url", "https://github.com/CrispStrobe/CrispASR"}};
                voice["installed"] = true;
                voice["languages"] = langs;

                json tts_service;
                tts_service["name"] = "crispasr-tts";
                tts_service["description"] = std::string("CrispASR ") + g_backend->name() + " TTS";
                tts_service["attribution"] = {{"name", "CrispASR"}, {"url", "https://github.com/CrispStrobe/CrispASR"}};
                tts_service["installed"] = true;
                tts_service["voices"] = json::array({voice});

                info_data["tts"] = json::array({tts_service});
            } else {
                info_data["tts"] = json::array();
            }

            wyoming_send(fd, "info", info_data);

            // ----------------------------------------------------------------
            // transcribe — sets language for the following audio stream
            // ----------------------------------------------------------------
        } else if (type == "transcribe") {
            asr_language = data.value("language", asr_language);
            audio_buf.clear();

            // ----------------------------------------------------------------
            // audio-start — announces incoming audio format
            // ----------------------------------------------------------------
        } else if (type == "audio-start") {
            audio_rate = data.value("rate", 16000);
            audio_width = data.value("width", 2);
            audio_ch = data.value("channels", 1);
            audio_buf.clear();

            // ----------------------------------------------------------------
            // audio-chunk — accumulate PCM payload
            // ----------------------------------------------------------------
        } else if (type == "audio-chunk") {
            if (payload_length > 0 && audio_width == 2) {
                // payload is int16 LE, possibly stereo
                int n_int16 = payload_length / 2;
                const int16_t* src = (const int16_t*)payload.data();
                // Mix down to mono if stereo
                if (audio_ch == 2) {
                    for (int i = 0; i < n_int16 / 2; i++)
                        audio_buf.push_back((int16_t)(((int)src[i * 2] + (int)src[i * 2 + 1]) / 2));
                } else {
                    audio_buf.insert(audio_buf.end(), src, src + n_int16);
                }
            } else if (payload_length > 0 && audio_width == 4) {
                // float32 LE PCM (some Wyoming clients) → mono int16 (H1)
                int n_f32 = payload_length / 4;
                const float* src = (const float*)payload.data();
                int ch = std::max(audio_ch, 1);
                int frames = n_f32 / ch;
                for (int i = 0; i < frames; i++) {
                    float acc = 0.0f;
                    for (int c = 0; c < ch; c++)
                        acc += src[i * ch + c];
                    acc /= (float)ch;
                    if (acc > 1.0f) acc = 1.0f;
                    if (acc < -1.0f) acc = -1.0f;
                    audio_buf.push_back((int16_t)(acc * 32767.0f));
                }
            } else if (payload_length > 0) {
                static bool warned = false;
                if (!warned) {
                    fprintf(stderr, "wyoming: unsupported audio width=%d (expect 2 or 4) — dropping\n", audio_width);
                    warned = true;
                }
            }

            // ----------------------------------------------------------------
            // audio-stop → run transcription → emit transcript
            // ----------------------------------------------------------------
        } else if (type == "audio-stop") {
            std::string transcript_text;

            if (!audio_buf.empty()) {
                // Convert int16 → float32, resample to 16 kHz
                auto pcmf32 = s16_to_f32(audio_buf.data(), (int)audio_buf.size());
                if (audio_rate != 16000)
                    pcmf32 = resample_f32(pcmf32.data(), (int)pcmf32.size(), audio_rate, 16000);

                whisper_params rp = g_params;
                rp.language = asr_language;

                {
                    std::lock_guard<std::mutex> lock(*g_mutex);
                    auto segs = g_backend->transcribe(pcmf32.data(), (int)pcmf32.size(), 0, rp);
                    for (auto& seg : segs)
                        transcript_text += seg.text;
                }

                // Strip leading/trailing whitespace
                while (!transcript_text.empty() && transcript_text.front() == ' ')
                    transcript_text.erase(transcript_text.begin());
                while (!transcript_text.empty() && transcript_text.back() == ' ')
                    transcript_text.pop_back();
            }

            json t_data;
            t_data["text"] = transcript_text;
            wyoming_send(fd, "transcript", t_data);

            audio_buf.clear();

            // ----------------------------------------------------------------
            // synthesize → audio-start + audio-chunk* + audio-stop
            // ----------------------------------------------------------------
        } else if (type == "synthesize") {
            if (!(g_backend->capabilities() & CAP_TTS)) {
                // Backend doesn't support TTS; send empty audio stream
                json astart;
                astart["rate"] = 22050;
                astart["width"] = 2;
                astart["channels"] = 1;
                wyoming_send(fd, "audio-start", astart);
                wyoming_send(fd, "audio-stop", json::object());
                continue;
            }

            // Extract text — Wyoming nests it under "text": {"text": "..."}
            std::string synth_text;
            if (data.contains("text") && data["text"].is_object())
                synth_text = data["text"].value("text", "");
            else if (data.contains("text") && data["text"].is_string())
                synth_text = data["text"].get<std::string>();

            // Voice: use requested voice if set and non-empty
            std::string voice_name;
            if (data.contains("voice") && data["voice"].is_object())
                voice_name = data["voice"].value("name", "");

            whisper_params rp = g_params;
            if (!voice_name.empty() && voice_name != "default")
                rp.tts_voice = voice_name;

            std::vector<float> pcmf32;
            int tts_rate = 24000;

            if (!synth_text.empty()) {
                std::lock_guard<std::mutex> lock(*g_mutex);
                tts_rate = g_backend->tts_sample_rate();
                pcmf32 = g_backend->synthesize(synth_text, rp);
            }

            // Stream PCM back as Wyoming audio events
            const int chunk_frames = 4096;
            json astart;
            astart["rate"] = tts_rate;
            astart["width"] = 2;
            astart["channels"] = 1;
            wyoming_send(fd, "audio-start", astart);

            if (!pcmf32.empty()) {
                auto s16 = f32_to_s16(pcmf32.data(), (int)pcmf32.size());
                json achunk;
                achunk["rate"] = tts_rate;
                achunk["width"] = 2;
                achunk["channels"] = 1;
                for (int off = 0; off < (int)s16.size(); off += chunk_frames) {
                    int n = std::min(chunk_frames, (int)s16.size() - off);
                    wyoming_send_payload(fd, "audio-chunk", achunk, s16.data() + off, n * (int)sizeof(int16_t));
                }
            }

            wyoming_send(fd, "audio-stop", json::object());
        }
        // Unknown message types are silently ignored (protocol allows this).
    }

    {
        std::lock_guard<std::mutex> lk(g_clients_mtx);
        g_clients.erase(std::remove(g_clients.begin(), g_clients.end(), fd), g_clients.end());
    }
    CLOSE_SOCKET(fd);
    g_active.fetch_sub(1);
}

// ---------------------------------------------------------------------------
// Listener thread
// ---------------------------------------------------------------------------

static void wyoming_listener() {
    while (g_running.load()) {
        // select() with a timeout so the loop re-checks g_running and exits
        // promptly on shutdown — closesocket() alone does not reliably unblock
        // a blocking accept() on Windows. (C2)
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(g_listen_fd, &rfds);
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        int sel = select((int)g_listen_fd + 1, &rfds, nullptr, nullptr, &tv);
        if (sel <= 0)
            continue; // timeout or interrupted → re-check g_running
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        socket_t client = accept(g_listen_fd, (struct sockaddr*)&addr, &addr_len);
        if (client == INVALID_SOCKET_VAL) {
            if (g_running.load())
                fprintf(stderr, "wyoming: accept failed (errno=%d)\n", SOCKET_ERRNO);
            continue;
        }
        std::thread(wyoming_handle_connection, client).detach();
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int wyoming_start(CrispasrBackend* backend, std::mutex& model_mutex, const whisper_params& params, int port) {
    if (g_running.load())
        return 0;

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "wyoming: WSAStartup failed\n");
        return -1;
    }
#endif

    g_backend = backend;
    g_mutex = &model_mutex;
    g_params = params;

    g_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_fd == INVALID_SOCKET_VAL) {
        fprintf(stderr, "wyoming: socket() failed\n");
        return -1;
    }
    int opt = 1;
    setsockopt(g_listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(g_listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "wyoming: bind(%d) failed (errno=%d)\n", port, SOCKET_ERRNO);
        CLOSE_SOCKET(g_listen_fd);
        g_listen_fd = INVALID_SOCKET_VAL;
        return -1;
    }
    if (listen(g_listen_fd, 8) < 0) {
        fprintf(stderr, "wyoming: listen() failed\n");
        CLOSE_SOCKET(g_listen_fd);
        g_listen_fd = INVALID_SOCKET_VAL;
        return -1;
    }

    g_running.store(true);
    g_listener_thread = std::thread(wyoming_listener);

    fprintf(stderr, "wyoming: TCP server listening on 0.0.0.0:%d\n", port);
    return 0;
}

void wyoming_stop() {
    if (!g_running.load())
        return;
    g_running.store(false);

    // Wake any connection threads blocked in recv() so they exit instead of
    // touching the backend after teardown. (H3)
    {
        std::lock_guard<std::mutex> lk(g_clients_mtx);
        for (socket_t cfd : g_clients)
            shutdown(cfd, SHUTDOWN_BOTH);
    }

    if (g_listener_thread.joinable())
        g_listener_thread.join(); // select() loop exits within ~1s (C2)

    if (g_listen_fd != INVALID_SOCKET_VAL) {
        CLOSE_SOCKET(g_listen_fd);
        g_listen_fd = INVALID_SOCKET_VAL;
    }

    // Bounded wait for in-flight (detached) handlers to drain. (H3)
    for (int i = 0; i < 100 && g_active.load() > 0; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

#ifdef _WIN32
    WSACleanup(); // (M1)
#endif
}
