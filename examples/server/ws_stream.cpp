// ws_stream.cpp — minimal RFC 6455 WebSocket server for real-time ASR.
//
// One listener thread accepts connections; each connection gets its own
// thread that creates a crispasr session + streaming decoder, feeds
// incoming binary frames as PCM, and sends back JSON text updates.
//
// Limitations (intentional simplicity):
//   - No TLS (use a reverse proxy for wss://)
//   - No fragmented frames (each client message must be a single frame)
//   - Max frame payload 1 MB (plenty for PCM chunks)
//   - No ping/pong (connection drops on timeout)

#include "ws_stream.h"
// CrispasrSession/CrispasrStream types and functions are forward-declared
// in ws_stream.h and resolved at link time from libcrispasr.

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET socket_t;
#  define CLOSE_SOCKET closesocket
#  define SOCKET_ERRNO WSAGetLastError()
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
   typedef int socket_t;
#  define INVALID_SOCKET (-1)
#  define CLOSE_SOCKET close
#  define SOCKET_ERRNO errno
#endif

// SHA-1 for WebSocket handshake (minimal, RFC 3174 compliant)
namespace {

struct sha1_ctx { uint32_t h[5]; uint64_t len; uint8_t buf[64]; int buf_len; };

static void sha1_init(sha1_ctx& c) {
    c.h[0]=0x67452301; c.h[1]=0xEFCDAB89; c.h[2]=0x98BADCFE;
    c.h[3]=0x10325476; c.h[4]=0xC3D2E1F0; c.len=0; c.buf_len=0;
}

static uint32_t rotl(uint32_t v, int n) { return (v<<n)|(v>>(32-n)); }

static void sha1_block(sha1_ctx& c, const uint8_t* d) {
    uint32_t w[80], a=c.h[0],b=c.h[1],cc=c.h[2],dd=c.h[3],e=c.h[4];
    for (int i=0;i<16;i++) w[i]=(uint32_t)d[i*4]<<24|(uint32_t)d[i*4+1]<<16|(uint32_t)d[i*4+2]<<8|d[i*4+3];
    for (int i=16;i<80;i++) w[i]=rotl(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
    for (int i=0;i<80;i++) {
        uint32_t f,k;
        if      (i<20) { f=(b&cc)|((~b)&dd); k=0x5A827999; }
        else if (i<40) { f=b^cc^dd;           k=0x6ED9EBA1; }
        else if (i<60) { f=(b&cc)|(b&dd)|(cc&dd); k=0x8F1BBCDC; }
        else           { f=b^cc^dd;           k=0xCA62C1D6; }
        uint32_t t=rotl(a,5)+f+e+k+w[i]; e=dd; dd=cc; cc=rotl(b,30); b=a; a=t;
    }
    c.h[0]+=a; c.h[1]+=b; c.h[2]+=cc; c.h[3]+=dd; c.h[4]+=e;
}

static void sha1_update(sha1_ctx& c, const void* data, size_t len) {
    auto* p = (const uint8_t*)data;
    c.len += len;
    while (len > 0) {
        int space = 64 - c.buf_len;
        int take = (int)len < space ? (int)len : space;
        memcpy(c.buf + c.buf_len, p, take);
        c.buf_len += take; p += take; len -= take;
        if (c.buf_len == 64) { sha1_block(c, c.buf); c.buf_len = 0; }
    }
}

static void sha1_final(sha1_ctx& c, uint8_t out[20]) {
    uint64_t bits = c.len * 8;
    uint8_t pad = 0x80;
    sha1_update(c, &pad, 1);
    pad = 0;
    while (c.buf_len != 56) sha1_update(c, &pad, 1);
    uint8_t be[8];
    for (int i=0;i<8;i++) be[i] = (uint8_t)(bits >> (56-8*i));
    sha1_update(c, be, 8);
    for (int i=0;i<5;i++) { out[i*4]=(uint8_t)(c.h[i]>>24); out[i*4+1]=(uint8_t)(c.h[i]>>16);
                            out[i*4+2]=(uint8_t)(c.h[i]>>8); out[i*4+3]=(uint8_t)c.h[i]; }
}

// Base64 encode
static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::string base64_encode(const uint8_t* d, int n) {
    std::string r;
    for (int i=0;i<n;i+=3) {
        uint32_t v = (uint32_t)d[i]<<16;
        if (i+1<n) v |= (uint32_t)d[i+1]<<8;
        if (i+2<n) v |= d[i+2];
        r += b64[(v>>18)&63]; r += b64[(v>>12)&63];
        r += (i+1<n) ? b64[(v>>6)&63] : '=';
        r += (i+2<n) ? b64[v&63] : '=';
    }
    return r;
}

// WebSocket accept key: SHA1(client_key + magic) → base64
static std::string ws_accept_key(const std::string& client_key) {
    std::string cat = client_key + "258EAFA5-E914-47DA-95CA-5AB5DC11D585";
    sha1_ctx c; sha1_init(c);
    sha1_update(c, cat.data(), cat.size());
    uint8_t hash[20]; sha1_final(c, hash);
    return base64_encode(hash, 20);
}

// Read exactly n bytes from socket
static bool recv_exact(socket_t fd, void* buf, size_t n) {
    auto* p = (uint8_t*)buf;
    while (n > 0) {
        auto r = recv(fd, (char*)p, (int)n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

// Send exactly n bytes
static bool send_all(socket_t fd, const void* buf, size_t n) {
    auto* p = (const uint8_t*)buf;
    while (n > 0) {
        auto r = send(fd, (const char*)p, (int)n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

// Send a WebSocket text frame
static bool ws_send_text(socket_t fd, const std::string& msg) {
    std::vector<uint8_t> frame;
    frame.push_back(0x81); // FIN + text opcode
    if (msg.size() < 126) {
        frame.push_back((uint8_t)msg.size());
    } else if (msg.size() < 65536) {
        frame.push_back(126);
        frame.push_back((uint8_t)(msg.size() >> 8));
        frame.push_back((uint8_t)(msg.size() & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--)
            frame.push_back((uint8_t)(msg.size() >> (8*i)));
    }
    frame.insert(frame.end(), msg.begin(), msg.end());
    return send_all(fd, frame.data(), frame.size());
}

// Send a WebSocket close frame
static void ws_send_close(socket_t fd) {
    uint8_t frame[] = {0x88, 0x00}; // FIN + close, 0 payload
    send_all(fd, frame, 2);
}

// Read one WebSocket frame. Returns payload length, -1 on error/close.
// Handles masking. Stores opcode in *opcode_out.
static int ws_read_frame(socket_t fd, std::vector<uint8_t>& payload, uint8_t* opcode_out) {
    uint8_t hdr[2];
    if (!recv_exact(fd, hdr, 2)) return -1;
    uint8_t opcode = hdr[0] & 0x0F;
    bool masked = (hdr[1] & 0x80) != 0;
    uint64_t len = hdr[1] & 0x7F;
    if (len == 126) {
        uint8_t ext[2];
        if (!recv_exact(fd, ext, 2)) return -1;
        len = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (len == 127) {
        uint8_t ext[8];
        if (!recv_exact(fd, ext, 8)) return -1;
        len = 0;
        for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
    }
    if (len > 1048576) return -1; // 1 MB max
    uint8_t mask[4] = {};
    if (masked && !recv_exact(fd, mask, 4)) return -1;
    payload.resize((size_t)len);
    if (len > 0 && !recv_exact(fd, payload.data(), (size_t)len)) return -1;
    if (masked) {
        for (size_t i = 0; i < (size_t)len; i++)
            payload[i] ^= mask[i % 4];
    }
    if (opcode_out) *opcode_out = opcode;
    return (int)len;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static std::atomic<bool> g_ws_running{false};
static std::thread g_ws_thread;
static socket_t g_ws_listen_fd = INVALID_SOCKET;
static std::string g_model_path;
static int g_n_threads = 4;

// Per-connection handler
static void ws_handle_connection(socket_t client_fd) {
    // 1. Read HTTP upgrade request
    char req_buf[4096];
    int req_len = recv(client_fd, req_buf, sizeof(req_buf) - 1, 0);
    if (req_len <= 0) { CLOSE_SOCKET(client_fd); return; }
    req_buf[req_len] = '\0';

    // Extract Sec-WebSocket-Key
    std::string req(req_buf);
    std::string ws_key;
    auto pos = req.find("Sec-WebSocket-Key:");
    if (pos == std::string::npos) pos = req.find("sec-websocket-key:");
    if (pos != std::string::npos) {
        auto start = req.find_first_not_of(" \t", pos + 18);
        auto end = req.find("\r\n", start);
        if (start != std::string::npos && end != std::string::npos)
            ws_key = req.substr(start, end - start);
    }
    if (ws_key.empty()) {
        const char* bad = "HTTP/1.1 400 Bad Request\r\n\r\n";
        send_all(client_fd, bad, strlen(bad));
        CLOSE_SOCKET(client_fd);
        return;
    }

    // 2. Send upgrade response
    std::string accept = ws_accept_key(ws_key);
    std::string resp = "HTTP/1.1 101 Switching Protocols\r\n"
                       "Upgrade: websocket\r\n"
                       "Connection: Upgrade\r\n"
                       "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
    if (!send_all(client_fd, resp.data(), resp.size())) {
        CLOSE_SOCKET(client_fd);
        return;
    }

    // 3. Create session + streaming decoder
    auto* session = crispasr_session_open(g_model_path.c_str(), g_n_threads);
    if (!session) {
        ws_send_text(client_fd, "{\"error\":\"failed to open session\"}");
        ws_send_close(client_fd);
        CLOSE_SOCKET(client_fd);
        return;
    }

    auto* stream = crispasr_session_stream_open(session, g_n_threads, 3000, 10000, 200, "en", 0);
    if (!stream) {
        ws_send_text(client_fd, "{\"error\":\"failed to open stream (whisper-only today)\"}");
        ws_send_close(client_fd);
        crispasr_session_close(session);
        CLOSE_SOCKET(client_fd);
        return;
    }

    ws_send_text(client_fd, "{\"status\":\"ready\"}");

    // 4. Frame loop: binary = PCM, text = control, close = bye
    std::vector<uint8_t> payload;
    int64_t last_counter = -1;
    while (g_ws_running.load()) {
        uint8_t opcode = 0;
        int len = ws_read_frame(client_fd, payload, &opcode);
        if (len < 0) break;

        if (opcode == 0x08) break; // close

        if (opcode == 0x02 && len > 0) { // binary = PCM float32
            int n_samples = len / (int)sizeof(float);
            if (n_samples > 0) {
                int rc = crispasr_stream_feed(stream, (const float*)payload.data(), n_samples);
                if (rc == 1) {
                    // New text available
                    char text_buf[8192] = {};
                    double t0 = 0, t1 = 0;
                    long long counter = 0;
                    crispasr_stream_get_text(stream, text_buf, sizeof(text_buf), &t0, &t1, &counter);
                    if (counter != last_counter) {
                        last_counter = counter;
                        char json_buf[8320];
                        snprintf(json_buf, sizeof(json_buf),
                                 "{\"text\":\"%s\",\"t0\":%.3f,\"t1\":%.3f,\"counter\":%lld}",
                                 text_buf, t0, t1, counter);
                        // Escape quotes in text for valid JSON
                        std::string json_str(json_buf);
                        if (!ws_send_text(client_fd, json_str)) break;
                    }
                }
            }
        }

        if (opcode == 0x01) { // text = control command
            std::string cmd(payload.begin(), payload.end());
            if (cmd == "flush") {
                crispasr_stream_flush(stream);
                char text_buf[8192] = {};
                double t0 = 0, t1 = 0;
                long long counter = 0;
                crispasr_stream_get_text(stream, text_buf, sizeof(text_buf), &t0, &t1, &counter);
                char json_buf[8320];
                snprintf(json_buf, sizeof(json_buf),
                         "{\"text\":\"%s\",\"t0\":%.3f,\"t1\":%.3f,\"counter\":%lld,\"final\":true}",
                         text_buf, t0, t1, counter);
                ws_send_text(client_fd, json_buf);
            }
        }
    }

    // 5. Cleanup
    crispasr_stream_close(stream);
    crispasr_session_close(session);
    ws_send_close(client_fd);
    CLOSE_SOCKET(client_fd);
}

static void ws_listener_thread() {
    while (g_ws_running.load()) {
        struct sockaddr_in addr;
        socklen_t addr_len = sizeof(addr);
        socket_t client = accept(g_ws_listen_fd, (struct sockaddr*)&addr, &addr_len);
        if (client == INVALID_SOCKET) {
            if (g_ws_running.load())
                fprintf(stderr, "ws: accept failed (errno=%d)\n", SOCKET_ERRNO);
            continue;
        }
        // One thread per connection (ASR streaming is CPU-bound per session)
        std::thread(ws_handle_connection, client).detach();
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" int ws_stream_start(const char* model_path, int port, int n_threads) {
    if (g_ws_running.load()) return 0;

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    g_model_path = model_path;
    g_n_threads = n_threads;

    g_ws_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_ws_listen_fd == INVALID_SOCKET) {
        fprintf(stderr, "ws: socket() failed\n");
        return -1;
    }
    int opt = 1;
    setsockopt(g_ws_listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(g_ws_listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "ws: bind(%d) failed (errno=%d)\n", port, SOCKET_ERRNO);
        CLOSE_SOCKET(g_ws_listen_fd);
        g_ws_listen_fd = INVALID_SOCKET;
        return -1;
    }

    if (listen(g_ws_listen_fd, 4) < 0) {
        fprintf(stderr, "ws: listen() failed\n");
        CLOSE_SOCKET(g_ws_listen_fd);
        g_ws_listen_fd = INVALID_SOCKET;
        return -1;
    }

    g_ws_running.store(true);
    g_ws_thread = std::thread(ws_listener_thread);

    fprintf(stderr, "ws: WebSocket streaming server listening on ws://0.0.0.0:%d\n", port);
    return 0;
}

extern "C" void ws_stream_stop(void) {
    if (!g_ws_running.load()) return;
    g_ws_running.store(false);
    if (g_ws_listen_fd != INVALID_SOCKET) {
        CLOSE_SOCKET(g_ws_listen_fd);
        g_ws_listen_fd = INVALID_SOCKET;
    }
    if (g_ws_thread.joinable())
        g_ws_thread.join();
}
