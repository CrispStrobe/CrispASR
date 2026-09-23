#include "realtime_server.h"
#include "core/realtime_turn_buffer.h"
#include "crispasr_vad.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../json.hpp"
#include "crispasr.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSE_SOCKET closesocket
#define SOCKET_ERRNO WSAGetLastError()
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET (-1)
#define CLOSE_SOCKET close
#define SOCKET_ERRNO errno
#endif

static std::atomic<bool> g_rt_running{false};
static std::thread g_rt_thread;
static socket_t g_rt_listen_fd = INVALID_SOCKET;

// SHA-1 for WebSocket handshake (minimal, RFC 3174 compliant)
namespace {
struct sha1_ctx {
    uint32_t h[5];
    uint64_t len;
    uint8_t buf[64];
    int buf_len;
};
static void sha1_init(sha1_ctx& c) {
    c.h[0] = 0x67452301;
    c.h[1] = 0xEFCDAB89;
    c.h[2] = 0x98BADCFE;
    c.h[3] = 0x10325476;
    c.h[4] = 0xC3D2E1F0;
    c.len = 0;
    c.buf_len = 0;
}
static uint32_t rotl(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}
static void sha1_block(sha1_ctx& c, const uint8_t* d) {
    uint32_t w[80], a = c.h[0], b = c.h[1], cc = c.h[2], dd = c.h[3], e = c.h[4];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)d[i * 4] << 24 | (uint32_t)d[i * 4 + 1] << 16 | (uint32_t)d[i * 4 + 2] << 8 | d[i * 4 + 3];
    for (int i = 16; i < 80; i++)
        w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & dd);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ cc ^ dd;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & cc) | (b & dd) | (cc & dd);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ cc ^ dd;
            k = 0xCA62C1D6;
        }
        uint32_t t = rotl(a, 5) + f + e + k + w[i];
        e = dd;
        dd = cc;
        cc = rotl(b, 30);
        b = a;
        a = t;
    }
    c.h[0] += a;
    c.h[1] += b;
    c.h[2] += cc;
    c.h[3] += dd;
    c.h[4] += e;
}
static void sha1_update(sha1_ctx& c, const void* data, size_t len) {
    auto* p = (const uint8_t*)data;
    c.len += len;
    while (len > 0) {
        int space = 64 - c.buf_len;
        int take = (int)len < space ? (int)len : space;
        memcpy(c.buf + c.buf_len, p, take);
        c.buf_len += take;
        p += take;
        len -= take;
        if (c.buf_len == 64) {
            sha1_block(c, c.buf);
            c.buf_len = 0;
        }
    }
}
static void sha1_final(sha1_ctx& c, uint8_t out[20]) {
    uint64_t bits = c.len * 8;
    uint8_t pad = 0x80;
    sha1_update(c, &pad, 1);
    pad = 0;
    while (c.buf_len != 56)
        sha1_update(c, &pad, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; i++)
        be[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha1_update(c, be, 8);
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c.h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c.h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c.h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c.h[i];
    }
}

static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string base64_encode(const uint8_t* d, int n) {
    std::string r;
    for (int i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)d[i] << 16;
        if (i + 1 < n)
            v |= (uint32_t)d[i + 1] << 8;
        if (i + 2 < n)
            v |= d[i + 2];
        r += b64[(v >> 18) & 63];
        r += b64[(v >> 12) & 63];
        r += (i + 1 < n) ? b64[(v >> 6) & 63] : '=';
        r += (i + 2 < n) ? b64[v & 63] : '=';
    }
    return r;
}

static std::string ws_accept_key(const std::string& client_key) {
    std::string cat = client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    sha1_ctx c;
    sha1_init(c);
    sha1_update(c, cat.data(), cat.size());
    uint8_t hash[20];
    sha1_final(c, hash);
    return base64_encode(hash, 20);
}

static bool recv_exact(socket_t fd, void* buf, size_t n) {
    auto* p = (uint8_t*)buf;
    while (n > 0) {
        auto r = recv(fd, (char*)p, (int)n, 0);
        if (r <= 0)
            return false;
        p += r;
        n -= r;
    }
    return true;
}

static bool send_all(socket_t fd, const void* buf, size_t n) {
    auto* p = (const uint8_t*)buf;
    while (n > 0) {
        auto r = send(fd, (const char*)p, (int)n, 0);
        if (r <= 0)
            return false;
        p += r;
        n -= r;
    }
    return true;
}

static bool ws_send_text(socket_t fd, const std::string& text) {
    size_t n = text.size();
    uint8_t head[10];
    int hlen = 0;
    head[0] = 0x81; // FIN + text
    if (n < 126) {
        head[1] = (uint8_t)n;
        hlen = 2;
    } else if (n <= 65535) {
        head[1] = 126;
        head[2] = (uint8_t)(n >> 8);
        head[3] = (uint8_t)n;
        hlen = 4;
    } else {
        head[1] = 127;
        for (int i = 0; i < 8; i++)
            head[2 + i] = (uint8_t)(n >> (56 - 8 * i));
        hlen = 10;
    }
    if (!send_all(fd, head, hlen))
        return false;
    return send_all(fd, text.data(), n);
}

static void ws_send_close(socket_t fd) {
    uint8_t f[2] = {0x88, 0x00};
    send_all(fd, f, 2);
}

static int ws_read_frame(socket_t fd, std::vector<uint8_t>& payload, uint8_t* out_opcode) {
    uint8_t h[2];
    if (!recv_exact(fd, h, 2))
        return -1;
    bool fin = (h[0] & 0x80) != 0;
    uint8_t opcode = h[0] & 0x0F;
    bool masked = (h[1] & 0x80) != 0;
    uint64_t len = h[1] & 0x7F;
    if (len == 126) {
        uint8_t e[2];
        if (!recv_exact(fd, e, 2))
            return -1;
        len = ((uint64_t)e[0] << 8) | e[1];
    } else if (len == 127) {
        uint8_t e[8];
        if (!recv_exact(fd, e, 8))
            return -1;
        len = 0;
        for (int i = 0; i < 8; i++)
            len = (len << 8) | e[i];
    }
    uint8_t mask[4] = {0, 0, 0, 0};
    if (masked) {
        if (!recv_exact(fd, mask, 4))
            return -1;
    }
    if (len > 1024 * 1024 * 10)
        return -1; // 10MB limit
    payload.resize((size_t)len);
    if (len > 0) {
        if (!recv_exact(fd, payload.data(), (size_t)len))
            return -1;
        if (masked) {
            for (size_t i = 0; i < len; i++)
                payload[i] ^= mask[i % 4];
        }
    }
    if (out_opcode)
        *out_opcode = opcode;
    return (int)len;
}

static std::vector<uint8_t> base64_decode(const std::string& in) {
    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++)
        T[(int)b64[i]] = i;

    int val = 0, valb = -8;
    for (char c : in) {
        if (T[(unsigned char)c] == -1)
            break;
        val = (val << 6) + T[(unsigned char)c];
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

} // namespace

struct rt_session {
    socket_t client_fd;
    CrispasrBackend* backend;
    std::mutex* model_mutex;
    whisper_params rp;
    core_realtime::TurnBuffer turn{16000 * 30};
    std::unique_ptr<CrispasrRealtimeSession> realtime;
    std::string text_sent;
    double turn_processing_ms = 0.0;
    double turn_queue_wait_ms = 0.0;
    bool server_vad = false;
    bool speech_active = false;
    std::vector<float> vad_window;
    size_t vad_samples_since_eval = 0;
    size_t audio_received_samples = 0;
    size_t vad_dropped_samples = 0;

    rt_session(socket_t fd, CrispasrBackend* b, std::mutex* m, whisper_params p)
        : client_fd(fd), backend(b), model_mutex(m), rp(std::move(p)), realtime(backend->create_realtime_session(rp)) {
        server_vad = rp.vad && !rp.vad_model.empty();
    }

    void send_simple_event(const char* type) {
        nlohmann::json evt;
        evt["type"] = type;
        ws_send_text(client_fd, evt.dump());
    }

    void emit_partial(const std::string& partial) {
        std::string diff;
        if (partial.size() > text_sent.size() && partial.compare(0, text_sent.size(), text_sent) == 0)
            diff = partial.substr(text_sent.size());
        else if (partial != text_sent)
            diff = partial;
        if (!diff.empty()) {
            nlohmann::json evt;
            evt["type"] = "conversation.item.input_audio_transcription.delta";
            evt["delta"] = diff;
            ws_send_text(client_fd, evt.dump());
            text_sent = partial;
        }
    }

    bool append_realtime(const float* samples, int n_samples, bool flush) {
        if (!realtime)
            return false;
        const auto queued = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(*model_mutex);
        const auto started = std::chrono::steady_clock::now();
        turn_queue_wait_ms += std::chrono::duration<double, std::milli>(started - queued).count();
        const bool ok = realtime->append(samples, n_samples, flush,
                                         [&](const std::string& partial, bool) { emit_partial(partial); });
        turn_processing_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        return ok;
    }

    void append_asr(const float* samples, size_t count) {
        size_t offset = 0;
        while (offset < count) {
            const auto appended = turn.append(samples + offset, count - offset);
            audio_received_samples += appended.consumed;
            if (realtime && appended.consumed > 0 &&
                !append_realtime(samples + offset, (int)appended.consumed, false)) {
                realtime.reset();
                text_sent.clear();
                nlohmann::json evt;
                evt["type"] = "session.updated";
                evt["partial_transcription"] = false;
                evt["reason"] = "native_stream_failed_commit_fallback";
                ws_send_text(client_fd, evt.dump());
            }
            offset += appended.consumed;
            if (appended.full)
                handle_commit();
            else if (appended.consumed == 0)
                break;
        }
    }

    void append_with_vad(const float* samples, size_t count) {
        if (!server_vad) {
            append_asr(samples, count);
            return;
        }
        vad_window.insert(vad_window.end(), samples, samples + count);
        vad_samples_since_eval += count;
        if (speech_active)
            append_asr(samples, count);
        const size_t eval_step = 1600; // 100 ms
        if (vad_samples_since_eval < eval_step)
            return;
        vad_samples_since_eval = 0;

        crispasr_vad_options opts;
        opts.threshold = rp.vad_threshold;
        opts.threshold_explicit = rp.vad_threshold_explicit;
        opts.min_speech_duration_ms = rp.vad_min_speech_duration_ms;
        opts.min_silence_duration_ms = rp.vad_min_silence_duration_ms;
        opts.speech_pad_ms = rp.vad_speech_pad_ms;
        opts.chunk_seconds = 0;
        opts.n_threads = rp.n_threads;
        bool load_failed = false;
        auto slices = crispasr_compute_vad_slices(vad_window.data(), (int)vad_window.size(), 16000,
                                                  rp.vad_model.c_str(), opts, &load_failed);
        if (load_failed) {
            server_vad = false;
            send_simple_event("input_audio_buffer.vad_failed");
            append_asr(vad_window.data(), vad_window.size());
            vad_window.clear();
            return;
        }
        if (!speech_active && !slices.empty()) {
            speech_active = true;
            send_simple_event("input_audio_buffer.speech_started");
            // The inactive probe is capped at one second, so forwarding it all
            // gives the model a robust onset/pre-roll without unbounded silence.
            append_asr(vad_window.data(), vad_window.size());
        } else if (speech_active) {
            const int silence_samples = rp.vad_min_silence_duration_ms * 16;
            const int trailing = slices.empty() ? (int)vad_window.size() : (int)vad_window.size() - slices.back().end;
            if (trailing >= silence_samples) {
                send_simple_event("input_audio_buffer.speech_stopped");
                handle_commit();
                speech_active = false;
                vad_window.clear();
                return;
            }
        }
        const size_t keep = speech_active ? 16000 * 3 : 16000;
        if (vad_window.size() > keep) {
            if (!speech_active)
                vad_dropped_samples += vad_window.size() - keep;
            vad_window.erase(vad_window.begin(), vad_window.end() - keep);
        }
    }

    double process_audio() {
        if (turn.empty())
            return 0.0;
        const auto queued = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(*model_mutex);
        const auto started = std::chrono::steady_clock::now();
        turn_queue_wait_ms += std::chrono::duration<double, std::milli>(started - queued).count();
        backend->transcribe_streaming(turn.audio().data(), (int)turn.size(), 0, rp,
                                      [&](const std::string& partial, bool /*is_final*/) { emit_partial(partial); });
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }

    void handle_commit() {
        const size_t committed_samples = turn.size();
        double processing_ms = 0.0;
        if (realtime && append_realtime(nullptr, 0, true)) {
            processing_ms = turn_processing_ms;
        } else {
            if (realtime) {
                realtime.reset();
                text_sent.clear();
            }
            processing_ms = turn_processing_ms + process_audio();
        }
        nlohmann::json evt;
        evt["type"] = "conversation.item.input_audio_transcription.completed";
        evt["transcript"] = text_sent;
        evt["audio_duration_ms"] = committed_samples * 1000 / 16000;
        evt["audio_received_duration_ms"] = (audio_received_samples + vad_dropped_samples) * 1000 / 16000;
        evt["audio_processed_duration_ms"] = committed_samples * 1000 / 16000;
        evt["queue_backlog_duration_ms"] = 0;
        evt["model_queue_wait_ms"] = turn_queue_wait_ms;
        evt["processing_ms"] = processing_ms;
        evt["end_to_end_processing_ms"] = processing_ms + turn_queue_wait_ms;
        evt["realtime_factor"] = processing_ms > 0.0 ? (committed_samples / 16.0) / processing_ms : 0.0;
        ws_send_text(client_fd, evt.dump());

        // Reset for next utterance
        turn.clear();
        if (realtime)
            realtime->reset();
        text_sent.clear();
        turn_processing_ms = 0.0;
        turn_queue_wait_ms = 0.0;
        audio_received_samples = 0;
        vad_dropped_samples = 0;
    }
};

static void rt_handle_connection(rt_session* sess) {
    char req_buf[8192];
    int req_len = 0;
    while (req_len < (int)sizeof(req_buf) - 1) {
        int n = recv(sess->client_fd, req_buf + req_len, sizeof(req_buf) - 1 - req_len, 0);
        if (n <= 0)
            break;
        req_len += n;
        req_buf[req_len] = '\0';
        if (strstr(req_buf, "\r\n\r\n"))
            break;
    }
    if (req_len <= 0) {
        CLOSE_SOCKET(sess->client_fd);
        delete sess;
        return;
    }
    req_buf[req_len] = '\0';

    std::string req(req_buf);
    std::string ws_key;
    auto pos = req.find("Sec-WebSocket-Key:");
    if (pos == std::string::npos)
        pos = req.find("sec-websocket-key:");
    if (pos != std::string::npos) {
        auto start = req.find_first_not_of(" \t", pos + 18);
        auto end = req.find("\r\n", start);
        if (start != std::string::npos && end != std::string::npos)
            ws_key = req.substr(start, end - start);
    }
    if (ws_key.empty()) {
        const char* bad = "HTTP/1.1 400 Bad Request\r\n\r\n";
        send_all(sess->client_fd, bad, strlen(bad));
        CLOSE_SOCKET(sess->client_fd);
        delete sess;
        return;
    }

    std::string accept = ws_accept_key(ws_key);
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\n"
                       "Upgrade: websocket\r\n"
                       "Connection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: " +
                       accept + "\r\n\r\n";
    if (!send_all(sess->client_fd, resp.data(), resp.size())) {
        CLOSE_SOCKET(sess->client_fd);
        delete sess;
        return;
    }

    nlohmann::json created;
    created["type"] = "session.created";
    created["turn_detection"] = sess->server_vad ? "server_vad" : "client_commit";
    created["server_vad"] = sess->server_vad;
    created["partial_transcription"] = sess->realtime != nullptr;
    created["max_turn_seconds"] = 30;
    ws_send_text(sess->client_fd, created.dump());

    std::vector<uint8_t> payload;
    while (g_rt_running.load()) {
        uint8_t opcode = 0;
        int len = ws_read_frame(sess->client_fd, payload, &opcode);
        if (len < 0)
            break;
        if (opcode == 0x08)
            break; // close

        if (opcode == 0x01 && len > 0) { // text
            std::string msg(payload.begin(), payload.end());
            try {
                auto j = nlohmann::json::parse(msg);
                std::string type = j.value("type", "");
                if (type == "input_audio_buffer.append") {
                    std::string b64 = j.value("audio", "");
                    if (!b64.empty()) {
                        auto pcm16 = base64_decode(b64);
                        int n_samples = pcm16.size() / 2;
                        const int16_t* p = (const int16_t*)pcm16.data();
                        std::vector<float> decoded((size_t)n_samples);
                        for (int i = 0; i < n_samples; i++)
                            decoded[(size_t)i] = (float)p[i] / 32768.0f;
                        sess->append_with_vad(decoded.data(), decoded.size());
                    }
                } else if (type == "input_audio_buffer.commit") {
                    sess->vad_window.clear();
                    sess->speech_active = false;
                    sess->handle_commit();
                }
            } catch (...) {
            }
        }
    }

    ws_send_close(sess->client_fd);
    CLOSE_SOCKET(sess->client_fd);
    delete sess;
}

static void rt_listener_thread(CrispasrBackend* backend, std::mutex* model_mutex, whisper_params base_params) {
    while (g_rt_running.load()) {
        // select() with a timeout, not a bare accept(): closing a listening socket
        // does not wake a blocked accept() on Linux, so realtime_server_stop()'s
        // join would hang and the server could not shut down.
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(g_rt_listen_fd, &rfds);
        struct timeval tv = {1, 0};
        if (select((int)g_rt_listen_fd + 1, &rfds, nullptr, nullptr, &tv) <= 0)
            continue;
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        socket_t client = accept(g_rt_listen_fd, (struct sockaddr*)&addr, &addr_len);
        if (client == INVALID_SOCKET) {
            continue;
        }
        auto* sess = new rt_session(client, backend, model_mutex, base_params);
        std::thread(rt_handle_connection, sess).detach();
    }
}

int realtime_server_start(CrispasrBackend* backend, std::mutex& model_mutex, const whisper_params& base_params,
                          int port) {
    if (g_rt_running.load())
        return 0;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    g_rt_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_rt_listen_fd == INVALID_SOCKET)
        return -1;

    int opt = 1;
    setsockopt(g_rt_listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(g_rt_listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        CLOSE_SOCKET(g_rt_listen_fd);
        g_rt_listen_fd = INVALID_SOCKET;
        return -1;
    }

    if (listen(g_rt_listen_fd, 4) < 0) {
        CLOSE_SOCKET(g_rt_listen_fd);
        g_rt_listen_fd = INVALID_SOCKET;
        return -1;
    }

    g_rt_running.store(true);
    g_rt_thread = std::thread(rt_listener_thread, backend, &model_mutex, base_params);

    fprintf(stderr, "realtime: vLLM Realtime WebSocket listening on ws://0.0.0.0:%d/v1/realtime\n", port);
    return 0;
}

void realtime_server_stop() {
    if (!g_rt_running.load())
        return;
    g_rt_running.store(false);
    if (g_rt_thread.joinable())
        g_rt_thread.join(); // the select() loop exits within ~1 s
    if (g_rt_listen_fd != INVALID_SOCKET) {
        CLOSE_SOCKET(g_rt_listen_fd);
        g_rt_listen_fd = INVALID_SOCKET;
    }
}
