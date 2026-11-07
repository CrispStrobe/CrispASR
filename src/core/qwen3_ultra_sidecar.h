#pragma once

#include "qwen3_ultra_features.h"
#include "qwen3_ultra_head.h"
#include <ggml-backend.h>

namespace qwen3_ultra_sidecar {
struct Package {
    qwen3_ultra_head::Weights weights;
    qwen3_ultra_features::Normalizer mean{};
    qwen3_ultra_features::Normalizer stddev{};
};

// Load a strictly shaped F32 sidecar. On failure `out` remains unchanged.
// The caller owns `backend`; the returned package owns its copied CPU weights.
bool load(const char* path, ggml_backend_t backend, Package& out);
} // namespace qwen3_ultra_sidecar
