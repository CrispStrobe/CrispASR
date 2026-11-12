#pragma once

// Experimental, strict F32 loader for the pinned RE-USE reference model.
// This does not expose a product backend.
#include "reuse_decoder.h"
#include "reuse_tf_mamba.h"
#include <ggml-backend.h>

namespace reuse_model_loader {
struct Decoder {
    std::vector<reuse_dense_encoder::Layer> dense;
    reuse_decoder::UpLayer frequency, time;
    std::vector<float> first_head, second_head;
    float first_bias = 0, second_bias = 0;
};
struct Model {
    std::vector<reuse_dense_encoder::Layer> encoder;
    std::vector<reuse_tf_mamba::Weights> blocks;
    Decoder magnitude, phase;
};

// Rejects incompatible architecture, revision, tensor count/name/dtype/shape,
// and nonfinite values. On failure leaves output unchanged.
bool load(const char* path, ggml_backend_t backend, Model& output);
} // namespace reuse_model_loader
