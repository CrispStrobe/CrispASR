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

// #337/#457: an F16 x F32 matmul narrows the F32 activation to half on
// backends whose GEMM runs in F16 — cuBLAS / hipBLAS (CUDA and ROCm share the
// ggml-cuda path) and Vulkan's mul_mm (shared-memory tiles in F16). A SwiGLU
// intermediate above 65504 (the 0.6B-F16 code predictor reaches ~156000) then
// becomes Inf and every output of that projection NaN. ggml-cpu keeps F32
// activations for F16 weights (CrispASR patch #38: vec_dot_type F32) and
// Metal's f16 x f32 kernels read src1 as float, so neither narrows.
inline bool f16_matmul_narrows_activations(const char* backend_name) {
    if (backend_name == nullptr)
        return false;
    return std::strncmp(backend_name, "CUDA", 4) == 0 || is_rocm_backend(backend_name) ||
           std::strncmp(backend_name, "Vulkan", 6) == 0 || std::strncmp(backend_name, "SYCL", 4) == 0;
}

// Promote the code predictor's F16 FFN down-projection weights to F32 when the
// predictor runs on a narrowing backend. env_override: -1 = policy, 0 = never,
// 1 = always (A/B testing on any backend).
inline bool promote_cp_down_to_f32(const char* backend_name, bool weights_are_f16, int env_override) {
    if (!weights_are_f16 || env_override == 0)
        return false;
    return env_override == 1 || f16_matmul_narrows_activations(backend_name);
}

// #337: pin the talker to CPU on Vulkan only on request. `vulkan_cpu` is
// CRISPASR_QWEN3_TTS_VULKAN_CPU; `vulkan_native` is the legacy
// CRISPASR_QWEN3_TTS_VULKAN_NATIVE (an explicit "0" there also asks for the
// old CPU pin, so scripts written against the old default keep working).
inline bool vulkan_talker_on_cpu(const char* vulkan_cpu, const char* vulkan_native) {
    if (vulkan_cpu && vulkan_cpu[0] && vulkan_cpu[0] != '0')
        return true;
    if (vulkan_native && vulkan_native[0] == '0')
        return true;
    return false;
}

} // namespace qwen3_tts_hip_policy
