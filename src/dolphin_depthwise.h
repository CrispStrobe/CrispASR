#pragma once

#include "ggml.h"

namespace dolphin_detail {

// WeNet CSGU normalizes explicitly padded frames only in causal mode.
// Noncausal Conv1d padding happens after normalization, so its edge frames
// must stay zero even when LayerNorm has a nonzero learned bias.
template <typename F>
ggml_tensor* depthwise_conv(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K, bool causal,
                            F pre) {
    const int C = (int)x->ne[0], T = (int)x->ne[1];
    const int left = causal ? K - 1 : (K - 1) / 2;
    const int right = K - 1 - left;
    ggml_tensor* xp = causal ? pre(ggml_pad_ext(g, ggml_cont(g, x), 0, 0, left, right, 0, 0, 0, 0))
                             : ggml_pad_ext(g, ggml_cont(g, pre(ggml_cont(g, x))), 0, 0, left, right, 0, 0, 0, 0);
    ggml_tensor* xt = ggml_cont(g, ggml_transpose(g, xp));
    ggml_tensor* y = ggml_ssm_conv(g, ggml_reshape_3d(g, xt, T + K - 1, C, 1), w);
    return ggml_add(g, ggml_reshape_2d(g, y, C, T), b);
}

} // namespace dolphin_detail
