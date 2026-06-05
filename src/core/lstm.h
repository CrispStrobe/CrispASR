// src/core/lstm.h — bidirectional LSTM helpers for ggml graphs.
//
// PyTorch nn.LSTM forward (single layer, single direction):
//
//   pre_t = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh   ∈ R^{4H}
//   i, f, g, o = pre_t.split(H)                          (PyTorch gate order)
//   c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
//   h_t = sigmoid(o) * tanh(c_t)
//
// Storage convention used by every Kokoro / StyleTTS2 LSTM in the GGUF:
//
//   weight_ih_l0          ne = (input_size,  4 * hidden_size)   F16
//   weight_hh_l0          ne = (hidden_size, 4 * hidden_size)   F16
//   bias_ih_l0            ne = (4 * hidden_size,)               F32
//   bias_hh_l0            ne = (4 * hidden_size,)               F32
//   weight_ih_l0_reverse  same shape as weight_ih_l0 (bidir layer 0)
//   ... etc for hh + biases
//
// Both biases are added (PyTorch keeps them separate even though their
// sum is mathematically equivalent — we follow the same pattern so the
// graph is bit-equivalent to the reference dump).
//
// We optimise the per-step graph by hoisting the input projection out of
// the loop:  proj_x = (W_ih @ X) + b_ih, computed once for all T at
// once.  Per timestep we still need W_hh @ h_{t-1} (the recurrence is
// fundamentally sequential).  At t=0 the helper skips the W_hh matmul
// entirely (h_{-1} = 0), which matches PyTorch's default zero initial
// state and avoids needing an externally-zeroed buffer.
//
// Output assembly: each step's h_t is written into a column of a
// pre-allocated `output` tensor (shape (H, T)) via ggml_view_2d +
// ggml_cpy + ggml_build_forward_expand.  This is the same pattern used
// by core_attn::kv_self_attn for the persistent KV cache write — the
// scheduler's view-tracking sequences the cpys before any downstream
// read of the output.

#pragma once

#include "ggml.h"

#include <cstddef>

namespace core_lstm {

// Single-direction LSTM forward over T timesteps.
//
//   ctx          per-graph ggml context (no_alloc=true)
//   gf           graph being built — cpy ops are appended via
//                ggml_build_forward_expand so the writes are guaranteed
//                to run before any downstream read of the returned tensor
//   X            input, ne = (input_size, T)  F32
//   W_ih         ne = (input_size,  4H)        F16/F32
//   W_hh         ne = (hidden_size, 4H)        F16/F32
//   b_ih, b_hh   ne = (4H,)                     F32
//   H            hidden_size
//   reverse      iterate t = T-1 .. 0 (use the *_reverse weights)
//
// Returns the LSTM output as a contiguous (H, T) F32 tensor.
static inline ggml_tensor* lstm_unidir(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* X, ggml_tensor* W_ih,
                                       ggml_tensor* W_hh, ggml_tensor* b_ih, ggml_tensor* b_hh, int H, bool reverse) {
    const int T = (int)X->ne[1];
    const int H4 = 4 * H;

    // Input projection over all T timesteps at once.
    // proj_x ne = (4H, T)
    ggml_tensor* proj_x = ggml_mul_mat(ctx, W_ih, X);
    proj_x = ggml_add(ctx, proj_x, b_ih);

    // Pre-allocated output container (H, T) F32.
    ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);

    ggml_tensor* h = nullptr;
    ggml_tensor* c = nullptr;

    const size_t row_stride_4h = proj_x->nb[1];
    const size_t row_stride_h = output->nb[1];
    const size_t f32_size = ggml_type_size(GGML_TYPE_F32);

    const int t0 = reverse ? T - 1 : 0;
    const int dt = reverse ? -1 : 1;
    for (int step = 0; step < T; step++) {
        const int t = t0 + step * dt;

        // proj_x slice for this timestep: (4H, 1) F32, contiguous.
        ggml_tensor* px = ggml_view_2d(ctx, proj_x, H4, 1, row_stride_4h, (size_t)t * row_stride_4h);

        // pre = px + (W_hh @ h + b_hh) — at step 0 the W_hh@0 term vanishes.
        ggml_tensor* pre;
        if (h) {
            ggml_tensor* ph = ggml_mul_mat(ctx, W_hh, h); // (4H, 1)
            ph = ggml_add(ctx, ph, b_hh);
            pre = ggml_add(ctx, px, ph);
        } else {
            pre = ggml_add(ctx, px, b_hh);
        }

        // Split (4H, 1) into 4 gates of shape (H, 1).  Gate order is
        // PyTorch's (i, f, g, o), each H-wide stripe along ne[0].
        const size_t pre_stride = pre->nb[1];
        ggml_tensor* gi = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)0 * H * f32_size);
        ggml_tensor* gf_gate = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)1 * H * f32_size);
        ggml_tensor* gg = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)2 * H * f32_size);
        ggml_tensor* go = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)3 * H * f32_size);

        ggml_tensor* sig_i = ggml_sigmoid(ctx, gi);
        ggml_tensor* sig_f = ggml_sigmoid(ctx, gf_gate);
        ggml_tensor* sig_o = ggml_sigmoid(ctx, go);
        ggml_tensor* tanh_g = ggml_tanh(ctx, gg);

        // c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
        if (c) {
            c = ggml_add(ctx, ggml_mul(ctx, sig_f, c), ggml_mul(ctx, sig_i, tanh_g));
        } else {
            c = ggml_mul(ctx, sig_i, tanh_g);
        }
        // h_t = sigmoid(o) * tanh(c_t)
        h = ggml_mul(ctx, sig_o, ggml_tanh(ctx, c));

        // Write h_t into column t of output.
        ggml_tensor* slot = ggml_view_2d(ctx, output, H, 1, row_stride_h, (size_t)t * row_stride_h);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h, slot));
    }
    return output;
}

// Bidirectional LSTM. Runs two unidirectional passes (forward over the
// natural order, backward over the reversed order using the
// *_reverse weights), then concatenates along the feature dim:
//   output ne = (2H, T)  with forward H first, then backward H.
static inline ggml_tensor* lstm_bidir(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* X, ggml_tensor* W_ih_f,
                                      ggml_tensor* W_hh_f, ggml_tensor* b_ih_f, ggml_tensor* b_hh_f,
                                      ggml_tensor* W_ih_r, ggml_tensor* W_hh_r, ggml_tensor* b_ih_r,
                                      ggml_tensor* b_hh_r, int H) {
    ggml_tensor* fwd = lstm_unidir(ctx, gf, X, W_ih_f, W_hh_f, b_ih_f, b_hh_f, H, /*reverse=*/false);
    ggml_tensor* bwd = lstm_unidir(ctx, gf, X, W_ih_r, W_hh_r, b_ih_r, b_hh_r, H, /*reverse=*/true);
    // Concat along feature axis (ne[0] == H), result (2H, T).
    return ggml_concat(ctx, fwd, bwd, /*dim=*/0);
}

} // namespace core_lstm
