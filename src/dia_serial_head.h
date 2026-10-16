#pragma once
// Opt-in CPU graph diagnostic. No external tensor, token or step-dependent math.
#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace dia_serial_head {
inline int mode(const char* value) {
    if (!value || !*value || std::strcmp(value, "off") == 0)
        return 0;
    if (std::strcmp(value, "identity") == 0)
        return 1;
    if (std::strcmp(value, "serial") == 0)
        return 2;
    return -1;
}
// Reject unsupported opt-in configurations before the encoder or graph runs.
// OFF preserves the existing path even for GPU or quantized models.
inline bool supported(int selected_mode, bool cpu, const std::vector<ggml_tensor*>& heads, int64_t hidden,
                      int64_t vocabulary) {
    if (selected_mode == 0)
        return true;
    if (selected_mode < 1 || selected_mode > 2 || !cpu || heads.empty())
        return false;
    for (const auto* head : heads) {
        if (!head || head->type != GGML_TYPE_F32 || head->ne[0] != hidden || head->ne[1] != vocabulary ||
            head->ne[2] != 1 || head->ne[3] != 1 || head->nb[0] != sizeof(float))
            return false;
    }
    return true;
}
inline void compute(ggml_tensor* dst, const ggml_tensor* original, const ggml_tensor* weight, const ggml_tensor* hidden,
                    int ith, int nth, void* userdata) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && original->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32 && hidden->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(original));
    GGML_ASSERT(dst->ne[1] == 1 && dst->ne[3] == 1 && hidden->ne[1] == 1 && hidden->ne[3] == 1);
    GGML_ASSERT(weight->ne[0] == hidden->ne[0] && weight->ne[1] == dst->ne[0]);
    GGML_ASSERT(weight->ne[2] == 1 && weight->ne[3] == 1 && hidden->ne[2] == dst->ne[2]);
    GGML_ASSERT(weight->nb[0] == sizeof(float) && hidden->nb[0] == sizeof(float));
    const bool serial = userdata != nullptr;
    const int64_t count = ggml_nelements(dst);
    auto* output = static_cast<float*>(dst->data);
    const auto* copy = static_cast<const float*>(original->data);
    for (int64_t index = ith; index < count; index += nth) {
        if (!serial) {
            output[index] = copy[index];
            continue;
        }
        const int64_t token = index % dst->ne[0], batch = index / dst->ne[0];
        const auto* w = reinterpret_cast<const float*>(static_cast<const char*>(weight->data) + token * weight->nb[1]);
        const auto* x = reinterpret_cast<const float*>(static_cast<const char*>(hidden->data) + batch * hidden->nb[2]);
        float sum = 0.0f;
        for (int64_t k = 0; k < hidden->ne[0]; ++k) {
            // An observable F32 product prevents contraction into FMA, including
            // under compilers that contract a plain multiply followed by add.
            const volatile float product = w[k] * x[k];
            sum += product;
        }
        output[index] = sum;
    }
}
inline ggml_tensor* apply(ggml_context* ctx, ggml_tensor* original, ggml_tensor* weight, ggml_tensor* hidden,
                          int selected_mode) {
    GGML_ASSERT(selected_mode >= 0 && selected_mode <= 2);
    if (selected_mode == 0)
        return original;
    // The original matmul remains a dependency for an exact identity control.
    // This deliberately pays both paths; it is a correctness experiment.
    static int serial_marker;
    return ggml_map_custom3(ctx, original, weight, hidden, compute, GGML_N_TASKS_MAX,
                            selected_mode == 2 ? &serial_marker : nullptr);
}
} // namespace dia_serial_head
