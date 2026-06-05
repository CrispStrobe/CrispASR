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

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace core_cpu {

// CPU LayerNorm in (d, T) layout: out = (x - mean) / sqrt(var + eps) * w + b.
// Parallel over time frames. Supports in-place (out == x) because each
// iteration reads and writes disjoint rows. `w` and `b` may be null.
static inline void layernorm(float* out, const float* x, const float* w, const float* b, int d, int T, float eps) {
#pragma omp parallel for schedule(static)
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
