#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vibevoice_semantic_weights {

// Quantized matmul also quantizes its activations. The semantic encoder's
// repeated residual cancellation amplifies that second rounding (issue #589).
// Recover legacy packs once at load, preserving their dequantized weight values
// in F16 so subsequent matmuls retain floating-point activations on every backend.
struct Copies {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    void clear() {
        if (buf)
            ggml_backend_buffer_free(buf);
        if (ctx)
            ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

inline bool needs_copy(const std::string& name, const ggml_tensor* t) {
    const auto ends_with = [&](const char* suffix) {
        const size_t n = std::strlen(suffix);
        return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
    };
    return t && ggml_is_quantized(t->type) && ggml_n_dims(t) == 2 && name.rfind("st_enc.s.", 0) == 0 &&
           (ends_with(".ffn.up.weight") || ends_with(".ffn.down.weight"));
}

inline bool recover(std::map<std::string, ggml_tensor*>& tensors, ggml_backend_t backend, Copies& copies) {
    std::vector<std::pair<std::string, ggml_tensor*>> selected;
    for (const auto& entry : tensors) {
        if (needs_copy(entry.first, entry.second)) {
            const ggml_type_traits* traits = ggml_get_type_traits(entry.second->type);
            if (!traits || !traits->to_float || !ggml_is_contiguous(entry.second))
                return false;
            selected.push_back(entry);
        }
    }
    if (selected.empty())
        return true;

    ggml_init_params params = {selected.size() * ggml_tensor_overhead(), nullptr, true};
    copies.ctx = ggml_init(params);
    if (!copies.ctx)
        return false;
    std::vector<ggml_tensor*> restored;
    for (const auto& entry : selected) {
        const ggml_tensor* src = entry.second;
        auto* dst = ggml_new_tensor_2d(copies.ctx, GGML_TYPE_F16, src->ne[0], src->ne[1]);
        ggml_set_name(dst, entry.first.c_str());
        restored.push_back(dst);
    }
    copies.buf = ggml_backend_alloc_ctx_tensors(copies.ctx, backend);
    if (!copies.buf) {
        copies.clear();
        return false;
    }
    ggml_backend_buffer_set_usage(copies.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // At most 1 MiB of F32 scratch, rather than another full encoder copy.
    // Batch rows to avoid thousands of tiny CPU/GPU transfers during loading.
    for (size_t i = 0; i < selected.size(); ++i) {
        const ggml_tensor* src = selected[i].second;
        ggml_tensor* dst = restored[i];
        const size_t width = (size_t)src->ne[0];
        const size_t row_bytes = ggml_row_size(src->type, src->ne[0]);
        const size_t chunk_rows = std::max<size_t>(1, (1024 * 1024) / (width * sizeof(float)));
        std::vector<uint8_t> raw(chunk_rows * row_bytes);
        std::vector<float> fp32(chunk_rows * width);
        std::vector<ggml_fp16_t> fp16(chunk_rows * width);
        const auto* traits = ggml_get_type_traits(src->type);
        for (size_t row = 0; row < (size_t)src->ne[1]; row += chunk_rows) {
            const size_t count = std::min(chunk_rows, (size_t)src->ne[1] - row);
            const size_t elements = count * width;
            ggml_backend_tensor_get(src, raw.data(), row * src->nb[1], count * row_bytes);
            traits->to_float(raw.data(), fp32.data(), elements);
            ggml_fp32_to_fp16_row(fp32.data(), fp16.data(), elements);
            ggml_backend_tensor_set(dst, fp16.data(), row * dst->nb[1], elements * sizeof(ggml_fp16_t));
        }
    }
    // Publish only after every copy is complete; existing graphs never point at
    // partially initialized weights. Copies live until the owning model is freed.
    for (size_t i = 0; i < selected.size(); ++i)
        tensors[selected[i].first] = restored[i];
    return true;
}

} // namespace vibevoice_semantic_weights
