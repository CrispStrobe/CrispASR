#pragma once
#include "zac.h"
#include <functional>

struct ssl_model;
// Encoder weights live separately from the decode-only codec. A model is single-threaded;
// callers sharing a model must serialize encoding (backend allocator is reused).
struct zac_encoder_model {
    zac_model codec;
    float window_ratio = 2.0f;
    struct quantizer {
        int input_dim = 0, code_dim = 0, size = 0;
        std::vector<float> in_weight, in_bias, out_weight, out_bias, codebook, normalized, norm_squared;
    };
    std::vector<quantizer> rvq;
};
// Longest input the encoder takes: zonos3.tokenizer.dual_reference.prepare_reference's limit. Audio runs in
// independent 20-second lanes (10 s of left context on continuations), so model memory does not grow with length;
// callers enforce their own, shorter limits.
constexpr int ZAC_ENCODER_MAX_SECONDS = 600;
bool zac_encoder_load(zac_encoder_model& m, const char* gguf, bool use_gpu);
void zac_encoder_free(zac_encoder_model& m);
bool zac_encode(const zac_encoder_model& m, const ssl_model& teacher, const float* audio, int n_samples,
                std::vector<int32_t>& codes);
// Diagnostic callback uses row-major [time,channels], with an independent padded lane prefix.
// Return false to abort on an output error. Outputs include full-file internal_audio,
// lane.N.audio/teacher_audio/acoustic/ssl/pooled/semantic/q1/semantic_prediction/residual,
// lane.N.rvq.C.projected/quantized/codes and final ordinary codes.
using zac_encoder_dump = std::function<bool(const std::string&, const float*, int, int)>;
bool zac_encode_diagnostic(const zac_encoder_model& m, const ssl_model& teacher, const float* audio, int n_samples,
                           std::vector<int32_t>& codes, const zac_encoder_dump& dump);

// Codec graph internals shared with the streaming decoder; not an application interface.
namespace zac_encoder_detail {
ggml_tensor* conv(ggml_context*, ggml_cgraph*, const zac_model&, ggml_tensor*, const std::string&, int dilation = 1,
                  int stride = 1);
ggml_tensor* snake(ggml_context*, ggml_cgraph*, const zac_model&, ggml_tensor*, const std::string&);
ggml_tensor* residual(ggml_context*, ggml_cgraph*, const zac_model&, ggml_tensor*, const std::string&, int);
ggml_tensor* convnext(ggml_context*, ggml_cgraph*, const zac_model&, ggml_tensor*, const std::string&, bool);
ggml_tensor* moss(ggml_context*, ggml_cgraph*, const zac_model&, ggml_tensor*, const std::string&, const zac_stage_hp&,
                  ggml_tensor* positions, ggml_tensor* mask, bool first);
} // namespace zac_encoder_detail
