#pragma once

#include <cstring>

namespace qwen3_tts_hip_policy {

inline bool is_rocm_backend(const char* backend_name) {
    return backend_name != nullptr && std::strncmp(backend_name, "ROCm", 4) == 0;
}

// #337: gfx1100 produced content-dependent wrong codec-encoder codes. Keep the
// native path available for parity work, but do not ship it by default until a
// real AMD run passes the public Daphne reproduction.
inline bool codec_must_use_cpu(const char* backend_name, bool native_override) {
    return is_rocm_backend(backend_name) && !native_override;
}

// The second #337 defect was narrower: the 5-layer, 1024-wide predictor
// from the 0.6B F16 artifact overflowed when HIP narrowed a large SwiGLU
// activation for an F16 down-projection matmul. The native override now bakes
// F32 down weights; retain the conservative CPU default pending validation on
// more GPUs. Quantized 0.6B and both 1.7B variants retain native HIP.
inline bool code_predictor_must_use_cpu(const char* backend_name, bool native_override, int n_layers, int d_model,
                                        bool weights_are_f16) {
    return is_rocm_backend(backend_name) && !native_override && n_layers == 5 && d_model == 1024 && weights_are_f16;
}

} // namespace qwen3_tts_hip_policy
