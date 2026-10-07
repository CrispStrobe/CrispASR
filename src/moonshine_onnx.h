#pragma once
#include <cstdint>
#include <string>

// Optional ONNX Runtime CPU backend. Open a five-graph streaming_config.json
// or an older two-graph config.json; all companions are relative to that file.
struct moonshine_onnx_context;
struct moonshine_onnx_stream;
moonshine_onnx_context* moonshine_onnx_open(const char* config_path, int threads);
bool moonshine_onnx_is_model(const char* path);
void moonshine_onnx_close(moonshine_onnx_context* ctx);
bool moonshine_onnx_incremental(const moonshine_onnx_context* ctx);
std::string moonshine_onnx_transcribe(moonshine_onnx_context* ctx, const float* pcm, int count);
moonshine_onnx_stream* moonshine_onnx_stream_open(moonshine_onnx_context* ctx, int step_ms);
int moonshine_onnx_stream_feed(moonshine_onnx_stream* stream, const float* pcm, int count);
int moonshine_onnx_stream_flush(moonshine_onnx_stream* stream);
int moonshine_onnx_stream_get_text(moonshine_onnx_stream* stream, char* out, int cap, double* t0, double* t1,
                                   int64_t* counter);
void moonshine_onnx_stream_close(moonshine_onnx_stream* stream);
