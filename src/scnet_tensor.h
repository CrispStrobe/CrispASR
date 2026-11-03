// CPU layout and pointwise operations for SCNet's F32 activations.
// These nodes only copy values, insert positive zero padding, or perform one
// F32 add/multiply per output. They introduce no reduction or fused arithmetic.
// Callers must pass cpu_only=false whenever a GPU backend is selected: that
// path, unsupported types, and unsupported layouts retain the standard ggml op.
#pragma once

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#define SCNET_TENSOR_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define SCNET_TENSOR_RESTRICT __restrict__
#else
#define SCNET_TENSOR_RESTRICT
#endif

namespace scnet_tensor {

// cppcheck 2.7 flags char*->float* reinterpret_casts as invalidPointerCast; route via void*.
inline const float* as_f32(const void* p) {
    return static_cast<const float*>(p);
}
namespace detail {

struct Range {
    int64_t begin, end;
};

// The scheduler can provide fewer workers than requested. Partition using its
// actual nth, including the case where there are fewer work items than workers.
inline Range worker_range(int64_t count, int ith, int nth) {
    const int64_t quotient = count / nth, remainder = count % nth;
    const int64_t begin = quotient * ith + std::min<int64_t>(ith, remainder);
    return {begin, begin + quotient + (ith < remainder ? 1 : 0)};
}

inline int64_t rows(const ggml_tensor* tensor) {
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

inline std::array<int64_t, 4> row_coordinates(const ggml_tensor* tensor, int64_t row) {
    const int64_t i1 = row % tensor->ne[1];
    row /= tensor->ne[1];
    return {0, i1, row % tensor->ne[2], row / tensor->ne[2]};
}

inline size_t row_offset(const ggml_tensor* tensor, const std::array<int64_t, 4>& index) {
    return size_t(index[1]) * tensor->nb[1] + size_t(index[2]) * tensor->nb[2] + size_t(index[3]) * tensor->nb[3];
}

inline void cont_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const auto* src = dst->src[0];
    auto* output = static_cast<float*>(dst->data);
    const auto* input = static_cast<const char*>(src->data);
    if (ggml_is_contiguous(src)) {
        const auto range = worker_range(dst->ne[0] * rows(dst), ith, nth);
        if (range.begin != range.end)
            std::memcpy(output + range.begin, input + size_t(range.begin) * sizeof(float),
                        size_t(range.end - range.begin) * sizeof(float));
        return;
    }
    if (src->nb[0] == sizeof(float)) {
        const auto range = worker_range(rows(dst), ith, nth);
        for (int64_t row = range.begin; row < range.end; ++row) {
            const auto index = row_coordinates(dst, row);
            std::memcpy(output + row * dst->ne[0], input + row_offset(src, index), size_t(dst->ne[0]) * sizeof(float));
        }
        return;
    }

    // A permuted dense tensor has a unit-stride source axis. Tile that axis
    // against destination axis zero so both reads and writes use short,
    // contiguous cache-line runs. Other axes (including batch) remain explicit.
    int unit_axis = -1;
    for (int axis = 1; axis < 4; ++axis)
        if (src->nb[axis] == sizeof(float) && src->ne[axis] > 1) {
            unit_axis = axis;
            break;
        }
    if (unit_axis != -1) {
        constexpr int TILE = 32;
        const int64_t tiles0 = (src->ne[0] + TILE - 1) / TILE;
        const int64_t tilesu = (src->ne[unit_axis] + TILE - 1) / TILE;
        int64_t outer = 1;
        for (int axis = 1; axis < 4; ++axis)
            if (axis != unit_axis)
                outer *= src->ne[axis];
        const auto range = worker_range(outer * tiles0 * tilesu, ith, nth);
        alignas(64) float tile[TILE][TILE];
        for (int64_t job = range.begin; job < range.end; ++job) {
            int64_t index = job;
            const int64_t start0 = (index % tiles0) * TILE;
            index /= tiles0;
            const int64_t startu = (index % tilesu) * TILE;
            index /= tilesu;
            size_t source_offset = size_t(start0) * src->nb[0] + size_t(startu) * src->nb[unit_axis];
            size_t target_offset = size_t(start0) * dst->nb[0] + size_t(startu) * dst->nb[unit_axis];
            for (int axis = 1; axis < 4; ++axis) {
                if (axis == unit_axis)
                    continue;
                const int64_t coordinate = index % src->ne[axis];
                index /= src->ne[axis];
                source_offset += size_t(coordinate) * src->nb[axis];
                target_offset += size_t(coordinate) * dst->nb[axis];
            }
            const int count0 = int(std::min<int64_t>(TILE, src->ne[0] - start0));
            const int countu = int(std::min<int64_t>(TILE, src->ne[unit_axis] - startu));
            for (int i0 = 0; i0 < count0; ++i0) {
                const auto* source = reinterpret_cast<const float*>(input + source_offset + size_t(i0) * src->nb[0]);
                std::memcpy(tile[i0], source, size_t(countu) * sizeof(float));
            }
            for (int iu = 0; iu < countu; ++iu) {
                auto* target = reinterpret_cast<float*>(static_cast<char*>(dst->data) + target_offset +
                                                        size_t(iu) * dst->nb[unit_axis]);
                for (int i0 = 0; i0 < count0; ++i0)
                    target[i0] = tile[i0][iu];
            }
        }
        return;
    }

    // Arbitrary strided views are uncommon here, but keep exact copy semantics
    // when no source axis is contiguous. memcpy also preserves NaN payloads.
    const auto range = worker_range(rows(dst), ith, nth);
    for (int64_t row = range.begin; row < range.end; ++row) {
        const auto index = row_coordinates(dst, row);
        const char* source = input + row_offset(src, index);
        for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0)
            std::memcpy(output + row * dst->ne[0] + i0, source + size_t(i0) * src->nb[0], sizeof(float));
    }
}

inline void concat_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const auto* a = dst->src[0];
    const auto* b = dst->src[1];
    int axis = 0;
    while (axis < 3 && dst->ne[axis] == a->ne[axis])
        ++axis;
    const auto range = worker_range(rows(dst), ith, nth);
    auto* output = static_cast<float*>(dst->data);
    for (int64_t row = range.begin; row < range.end; ++row) {
        auto index = row_coordinates(dst, row);
        auto* target = output + row * dst->ne[0];
        if (axis == 0) {
            std::memcpy(target, static_cast<const char*>(a->data) + row_offset(a, index),
                        size_t(a->ne[0]) * sizeof(float));
            std::memcpy(target + a->ne[0], static_cast<const char*>(b->data) + row_offset(b, index),
                        size_t(b->ne[0]) * sizeof(float));
        } else {
            const auto* source = a;
            if (index[axis] >= a->ne[axis]) {
                index[axis] -= a->ne[axis];
                source = b;
            }
            std::memcpy(target, static_cast<const char*>(source->data) + row_offset(source, index),
                        size_t(dst->ne[0]) * sizeof(float));
        }
    }
}

inline void pad_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const auto* src = dst->src[0];
    std::array<int64_t, 4> left{};
    for (int axis = 0; axis < 4; ++axis)
        left[axis] = (dst->ne[axis] - src->ne[axis]) / 2;
    const auto range = worker_range(rows(dst), ith, nth);
    auto* output = static_cast<float*>(dst->data);
    for (int64_t row = range.begin; row < range.end; ++row) {
        auto index = row_coordinates(dst, row);
        bool inside = true;
        for (int axis = 1; axis < 4; ++axis) {
            index[axis] -= left[axis];
            inside = inside && index[axis] >= 0 && index[axis] < src->ne[axis];
        }
        auto* target = output + row * dst->ne[0];
        if (!inside) {
            std::memset(target, 0, size_t(dst->ne[0]) * sizeof(float));
            continue;
        }
        std::memset(target, 0, size_t(left[0]) * sizeof(float));
        std::memcpy(target + left[0], static_cast<const char*>(src->data) + row_offset(src, index),
                    size_t(src->ne[0]) * sizeof(float));
        const int64_t end = left[0] + src->ne[0];
        std::memset(target + end, 0, size_t(dst->ne[0] - end) * sizeof(float));
    }
}

template <bool Multiply>
inline void binary_vector(float* SCNET_TENSOR_RESTRICT output, const float* SCNET_TENSOR_RESTRICT a,
                          const float* SCNET_TENSOR_RESTRICT b, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        if constexpr (Multiply)
            output[i] = a[i] * b[i];
        else
            output[i] = a[i] + b[i];
    }
}

template <bool Multiply>
inline void binary_scalar(float* SCNET_TENSOR_RESTRICT output, const float* SCNET_TENSOR_RESTRICT a, float b,
                          int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        if constexpr (Multiply)
            output[i] = a[i] * b;
        else
            output[i] = a[i] + b;
    }
}

template <bool Multiply> inline void binary_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const auto* a = dst->src[0];
    const auto* b = dst->src[1];
    auto* output = static_cast<float*>(dst->data);
    if (ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_are_same_shape(a, b)) {
        const auto range = worker_range(dst->ne[0] * rows(dst), ith, nth);
        binary_vector<Multiply>(output + range.begin, static_cast<const float*>(a->data) + range.begin,
                                static_cast<const float*>(b->data) + range.begin, range.end - range.begin);
        return;
    }
    const auto range = worker_range(rows(dst), ith, nth);
    for (int64_t row = range.begin; row < range.end; ++row) {
        auto index = row_coordinates(dst, row);
        const auto* x = as_f32(static_cast<const char*>(a->data) + row_offset(a, index));
        for (int axis = 1; axis < 4; ++axis)
            index[axis] %= b->ne[axis];
        const char* y = static_cast<const char*>(b->data) + row_offset(b, index);
        auto* target = output + row * dst->ne[0];
        if (b->ne[0] == 1) {
            binary_scalar<Multiply>(target, x, *as_f32(y), dst->ne[0]);
        } else if (b->nb[0] == sizeof(float)) {
            for (int64_t i0 = 0; i0 < dst->ne[0]; i0 += b->ne[0])
                binary_vector<Multiply>(target + i0, x + i0, as_f32(y), b->ne[0]);
        } else {
            for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                const float value = *as_f32(y + size_t(i0 % b->ne[0]) * b->nb[0]);
                if constexpr (Multiply)
                    target[i0] = x[i0] * value;
                else
                    target[i0] = x[i0] + value;
            }
        }
    }
}

inline ggml_tensor* custom(ggml_context* ctx, const int64_t* shape, ggml_tensor** inputs, int count,
                           ggml_custom_op_t operation, const char* name) {
    auto* result = ggml_custom_4d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3], inputs, count, operation,
                                  GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(result, name);
    return result;
}

} // namespace detail

// Always creates an independent output, even for an already contiguous input.
// Capture code relies on that copy/name/lifetime behavior.
inline ggml_tensor* cont(ggml_context* ctx, ggml_tensor* a, bool cpu_only) {
    if (!cpu_only || a->type != GGML_TYPE_F32)
        return ggml_cont(ctx, a);
    ggml_tensor* inputs[] = {a};
    return detail::custom(ctx, a->ne, inputs, 1, detail::cont_compute, "scnet_cpu_cont");
}

inline ggml_tensor* concat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int axis, bool cpu_only) {
    bool supported = cpu_only && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) &&
                     b->nb[0] == sizeof(float) && axis >= 0 && axis < 4;
    for (int dim = 0; dim < 4; ++dim)
        supported = supported && (dim == axis || a->ne[dim] == b->ne[dim]);
    if (!supported)
        return ggml_concat(ctx, a, b, axis);
    std::array<int64_t, 4> shape{a->ne[0], a->ne[1], a->ne[2], a->ne[3]};
    shape[axis] += b->ne[axis];
    ggml_tensor* inputs[] = {a, b};
    return detail::custom(ctx, shape.data(), inputs, 2, detail::concat_compute, "scnet_cpu_concat");
}

inline ggml_tensor* pad_ext(ggml_context* ctx, ggml_tensor* a, int lp0, int rp0, int lp1, int rp1, int lp2, int rp2,
                            int lp3, int rp3, bool cpu_only) {
    const std::array<int, 8> padding{lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3};
    bool supported = cpu_only && a->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float);
    for (int axis = 0; axis < 4; ++axis) {
        const int left = padding[2 * axis], right = padding[2 * axis + 1];
        supported = supported && left >= 0 && right >= left && int64_t(right) - left <= 1;
    }
    if (!supported)
        return ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    std::array<int64_t, 4> shape{};
    for (int axis = 0; axis < 4; ++axis)
        shape[axis] = a->ne[axis] + padding[2 * axis] + padding[2 * axis + 1];
    ggml_tensor* inputs[] = {a};
    return detail::custom(ctx, shape.data(), inputs, 1, detail::pad_compute, "scnet_cpu_pad");
}

inline ggml_tensor* add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, bool cpu_only) {
    if (!cpu_only || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) ||
        !ggml_can_repeat(b, a))
        return ggml_add(ctx, a, b);
    ggml_tensor* inputs[] = {a, b};
    return detail::custom(ctx, a->ne, inputs, 2, detail::binary_compute<false>, "scnet_cpu_add");
}

inline ggml_tensor* mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, bool cpu_only) {
    if (!cpu_only || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) ||
        !ggml_can_repeat(b, a))
        return ggml_mul(ctx, a, b);
    ggml_tensor* inputs[] = {a, b};
    return detail::custom(ctx, a->ne, inputs, 2, detail::binary_compute<true>, "scnet_cpu_mul");
}

} // namespace scnet_tensor

#undef SCNET_TENSOR_RESTRICT
