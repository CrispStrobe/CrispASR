// src/core/cpu_ops.h — small CPU-side primitives shared by the granite family.
//
// `granite_speech.cpp` and `granite_nle.cpp` ship byte-identical copies
// of two helpers:
//
//   * `cpu_layernorm` — pure CPU LayerNorm in (d, T) layout, parallelised
//     over time frames; supports in-place use because each frame's stats
//     and output rows are disjoint.
//
//   * `run_matmul`    — out = W @ x [+ bias] computed via a tiny single-op
//     ggml graph dispatched through the caller's `ggml_backend_sched_t`.
//     Used at every place a single GGUF tensor needs to be applied to an
//     F32 activation buffer (mel input projection, attention output,
//     CTC head, BPE head, etc.).
//
// Lifting both to a shared header keeps the granite TUs honest — same
// math, same numerics — without forcing a context type onto the helper.
// The matmul takes `compute_meta` and `sched` directly, so any backend
// that has those two pieces (every one of ours does) can call through.
//
// Header-only `static inline` so each call site keeps inlined codegen
// identical to the original hand-rolled helper — pure rename, no LSB
// drift expected on the diff harness.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "parallel_for.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace core_cpu {

// FP16/quantized → F32 dequantization (GPU-safe). Reads a whole tensor to a
// host F32 buffer via ggml_backend_tensor_get() (works whether the weight lives
// in a CPU or GPU buffer). Correct for F32, F16, AND any block-quantized type —
// unlike a raw `ggml_backend_tensor_get(t, buf, 0, n*sizeof(float))`, which
// over-reads a quantized tensor (its ggml_nbytes is far smaller than n*4) and
// asserts, or an F16-only reader that leaves the buffer as garbage on a
// quantized weight. Use this for every CPU-side weight read; never size the copy
// by n*sizeof(float) on a tensor whose type you don't control.
static inline std::vector<float> to_f32(const ggml_tensor* t) {
    if (!t)
        return {};
    int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t)n);
        ggml_backend_tensor_get(t, tmp.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
    } else {
        // Quantized: read the tensor's native bytes, then dequantize via the
        // type's to_float trait (block-aligned rows — ne[0] is a multiple of the
        // block size for any quantizable tensor, so a whole-buffer pass matches
        // per-row layout).
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const ggml_type_traits* tr = ggml_get_type_traits(t->type);
        if (tr && tr->to_float)
            tr->to_float(raw.data(), out.data(), n);
        else
            std::fill(out.begin(), out.end(), 0.0f);
    }
    return out;
}

// Check if sequential baseline CPU ops is requested.
static inline bool crispasr_cpu_ops_serial() {
    const char* e = std::getenv("CRISPASR_CPU_OPS_SERIAL");
    return (e && *e && std::strcmp(e, "0") != 0);
}

static inline void layernorm_baseline(float* out, const float* x, const float* w, const float* b, int d, int T,
                                      float eps) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * d;
        float* ot = out + (size_t)t * d;
        float mean = 0, var = 0;
        for (int i = 0; i < d; i++)
            mean += xt[i];
        mean /= d;
        for (int i = 0; i < d; i++) {
            float v = xt[i] - mean;
            var += v * v;
        }
        var /= d;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < d; i++)
            ot[i] = (xt[i] - mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

static inline void layernorm_accelerated(float* out, const float* x, const float* w, const float* b, int d, int T,
                                         float eps, int n_threads = 0) {
    core_parallel::for_each_chunk(T, n_threads, [&](int t_start, int t_end) {
        for (int t = t_start; t < t_end; t++) {
            const float* xt = x + (size_t)t * d;
            float* ot = out + (size_t)t * d;

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int i = 0;
            for (; i + 4 <= d; i += 4) {
                s0 += xt[i + 0];
                s1 += xt[i + 1];
                s2 += xt[i + 2];
                s3 += xt[i + 3];
            }
            float sum = (s0 + s1) + (s2 + s3);
            for (; i < d; i++) {
                sum += xt[i];
            }
            const float mean = sum / (float)d;

            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            i = 0;
            for (; i + 4 <= d; i += 4) {
                float d0 = xt[i + 0] - mean;
                float d1 = xt[i + 1] - mean;
                float d2 = xt[i + 2] - mean;
                float d3 = xt[i + 3] - mean;
                v0 += d0 * d0;
                v1 += d1 * d1;
                v2 += d2 * d2;
                v3 += d3 * d3;
            }
            float var_sum = (v0 + v1) + (v2 + v3);
            for (; i < d; i++) {
                float di = xt[i] - mean;
                var_sum += di * di;
            }
            const float var = var_sum / (float)d;
            const float inv = 1.0f / std::sqrt(var + eps);

            i = 0;
            if (w && b) {
                for (; i + 4 <= d; i += 4) {
                    ot[i + 0] = (xt[i + 0] - mean) * inv * w[i + 0] + b[i + 0];
                    ot[i + 1] = (xt[i + 1] - mean) * inv * w[i + 1] + b[i + 1];
                    ot[i + 2] = (xt[i + 2] - mean) * inv * w[i + 2] + b[i + 2];
                    ot[i + 3] = (xt[i + 3] - mean) * inv * w[i + 3] + b[i + 3];
                }
                for (; i < d; i++) {
                    ot[i] = (xt[i] - mean) * inv * w[i] + b[i];
                }
            } else if (w) {
                for (; i + 4 <= d; i += 4) {
                    ot[i + 0] = (xt[i + 0] - mean) * inv * w[i + 0];
                    ot[i + 1] = (xt[i + 1] - mean) * inv * w[i + 1];
                    ot[i + 2] = (xt[i + 2] - mean) * inv * w[i + 2];
                    ot[i + 3] = (xt[i + 3] - mean) * inv * w[i + 3];
                }
                for (; i < d; i++) {
                    ot[i] = (xt[i] - mean) * inv * w[i];
                }
            } else if (b) {
                for (; i + 4 <= d; i += 4) {
                    ot[i + 0] = (xt[i + 0] - mean) * inv + b[i + 0];
                    ot[i + 1] = (xt[i + 1] - mean) * inv + b[i + 1];
                    ot[i + 2] = (xt[i + 2] - mean) * inv + b[i + 2];
                    ot[i + 3] = (xt[i + 3] - mean) * inv + b[i + 3];
                }
                for (; i < d; i++) {
                    ot[i] = (xt[i] - mean) * inv + b[i];
                }
            } else {
                for (; i + 4 <= d; i += 4) {
                    ot[i + 0] = (xt[i + 0] - mean) * inv;
                    ot[i + 1] = (xt[i + 1] - mean) * inv;
                    ot[i + 2] = (xt[i + 2] - mean) * inv;
                    ot[i + 3] = (xt[i + 3] - mean) * inv;
                }
                for (; i < d; i++) {
                    ot[i] = (xt[i] - mean) * inv;
                }
            }
        }
    });
}

// CPU LayerNorm in (d, T) layout: out = (x - mean) / sqrt(var + eps) * w + b.
// Parallel over time frames. Supports in-place (out == x) because each
// iteration reads and writes disjoint rows. `w` and `b` may be null.
static inline void layernorm(float* out, const float* x, const float* w, const float* b, int d, int T, float eps,
                             int n_threads = 0) {
    if (crispasr_cpu_ops_serial() || T <= 1) {
        layernorm_baseline(out, x, w, b, d, T, eps);
    } else {
        layernorm_accelerated(out, x, w, b, d, T, eps, n_threads);
    }
}

// out = W @ x [+ bias], computed via a one-op ggml graph dispatched through
// `sched`. `compute_meta` is the caller's pre-sized scratch buffer used as
// `ggml_init_params::mem_buffer` (the same buffer the model reuses for
// every per-step graph it builds — keeping it caller-owned lets the
// backend pool that allocation across calls).
//
//   x    : (d_in, T) F32, row-major
//   W    : GGUF tensor of any quant type
//   bias : optional (may be null)
//   out  : (d_out, T) F32
static inline bool matmul(std::vector<uint8_t>& compute_meta, ggml_backend_sched_t sched, float* out, const float* x,
                          int d_in, int T, ggml_tensor* W, ggml_tensor* bias, int d_out) {
    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_in, T);
    ggml_set_name(inp, "mm_in");
    ggml_set_input(inp);

    ggml_tensor* r = ggml_mul_mat(ctx0, W, inp);
    if (bias)
        r = ggml_add(ctx0, r, bias);
    ggml_set_name(r, "mm_out");
    ggml_build_forward_expand(gf, r);
    ggml_free(ctx0);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mm_in"), x, 0, (size_t)d_in * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "mm_out"), out, 0, (size_t)d_out * T * sizeof(float));
    return true;
}

} // namespace core_cpu
