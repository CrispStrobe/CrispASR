#pragma once

#include "ggml.h"

#include <cfloat>
#include <climits>

namespace core_attention {

// Flash attention accepts an F16 additive mask. Relative-position scores can
// have large row offsets: casting those offsets directly loses differences
// between keys before softmax sees them. A constant per query/head/batch has
// no effect on softmax, so remove the row maximum while still in F32.
static inline ggml_tensor* centered_flash_mask(ggml_context* ctx, ggml_tensor* mask) {
    GGML_ASSERT(mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->ne[0] > 0 && mask->ne[0] <= INT_MAX);
    mask = ggml_cont(ctx, mask);
    // Pool only keys (ne[0]); queries, heads and batches remain independent.
    auto* maximum = ggml_pool_2d(ctx, mask, GGML_OP_POOL_MAX, (int)mask->ne[0], 1, (int)mask->ne[0], 1, 0, 0);
    // An all-masked row must stay -inf rather than becoming -inf - -inf.
    // Keep invalid +inf/NaN inputs nonfinite as well; this is not sanitization.
    maximum = ggml_clamp(ctx, maximum, -FLT_MAX, FLT_MAX);
    return ggml_cast(ctx, ggml_sub(ctx, mask, maximum), GGML_TYPE_F16);
}

} // namespace core_attention
