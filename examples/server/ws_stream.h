// ws_stream.h — minimal RFC 6455 WebSocket server for real-time ASR streaming.
//
// Runs on a separate port (default: HTTP port + 1). Clients connect,
// send binary PCM chunks (16kHz mono float32), receive JSON text updates:
//   {"text": "...", "t0": 0.0, "t1": 1.5, "counter": 1}
//
// Uses raw POSIX/Winsock sockets. No external dependencies beyond the
// C standard library and the crispasr Session API.
//
// Usage from the HTTP server main():
//   ws_stream_start(ctx, ws_port, params);  // spawns listener thread
//   ws_stream_stop();                        // joins and cleans up

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declare Session API (exported by libcrispasr)
struct CrispasrSession;
struct CrispasrStream;
struct CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
void crispasr_session_close(struct CrispasrSession* s);
struct CrispasrStream* crispasr_session_stream_open(struct CrispasrSession* s, int n_threads,
                                                     int step_ms, int length_ms, int keep_ms,
                                                     const char* language, int translate);
int crispasr_stream_feed(struct CrispasrStream* s, const float* pcm, int n_samples);
int crispasr_stream_get_text(struct CrispasrStream* s, char* out_text, int out_cap,
                              double* out_t0_s, double* out_t1_s, long long* out_counter);
int crispasr_stream_flush(struct CrispasrStream* s);
void crispasr_stream_close(struct CrispasrStream* s);

// Start the WebSocket listener thread on `port`. `model_path` is used
// to create per-connection crispasr sessions. Returns 0 on success.
int ws_stream_start(const char* model_path, int port, int n_threads);

// Stop the listener and join the thread. Safe to call multiple times.
void ws_stream_stop(void);

#ifdef __cplusplus
}
#endif
