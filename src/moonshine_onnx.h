#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Optional ONNX Runtime CPU backend. Open a five-graph streaming_config.json
// or an older two-graph config.json; all companions are relative to that file.
struct moonshine_onnx_context;
struct moonshine_onnx_stream;
moonshine_onnx_context* moonshine_onnx_open(const char* config_path, int threads);
bool moonshine_onnx_is_model(const char* path);
void moonshine_onnx_close(moonshine_onnx_context* ctx);
// <= 0 restores the model's duration-derived decoding budget.
void moonshine_onnx_set_max_new_tokens(moonshine_onnx_context* ctx, int count);
bool moonshine_onnx_incremental(const moonshine_onnx_context* ctx);
std::string moonshine_onnx_transcribe(moonshine_onnx_context* ctx, const float* pcm, int count);
moonshine_onnx_stream* moonshine_onnx_stream_open(moonshine_onnx_context* ctx, int step_ms);
int moonshine_onnx_stream_feed(moonshine_onnx_stream* stream, const float* pcm, int count);
int moonshine_onnx_stream_flush(moonshine_onnx_stream* stream);
int moonshine_onnx_stream_get_text(moonshine_onnx_stream* stream, char* out, int cap, double* t0, double* t1,
                                   int64_t* counter);
void moonshine_onnx_stream_close(moonshine_onnx_stream* stream);

// Diagnostic capture of the actual batch forward. Copies the first invocation
// of each graph while its outputs are live, before decoder/cache reuse.
struct moonshine_onnx_stage {
    std::string name;
    std::vector<int64_t> shape; // row-major ONNX shape, unit axes retained
    std::vector<float> data;
};
struct moonshine_onnx_capture {
    std::vector<moonshine_onnx_stage> stages;
    std::string text;
};
moonshine_onnx_capture moonshine_onnx_debug_forward(moonshine_onnx_context* ctx, const float* pcm, int count);
