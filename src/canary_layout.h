// canary_layout.h - tensor-layout rules the canary loader validates against
// (#470 follow-up). Header-only and dependency-free so the rules can be unit
// tested without a model file.
#pragma once

#include <cstdint>

namespace canary_layout {

// A Conformer pointwise Conv1d weight (kernel size 1) mapping `in` -> `out`
// channels reaches the runtime in one of two layouts, both consumed unchanged by
// the reshape-based compute path:
//   Flat      [in, out]      - the cstr QUANTISED legacy GGUFs (q4_k/q5_0/q8_0)
//   Singleton [1, in, out]   - the cstr F16/F32 legacy GGUF (canary-1b-v2.gguf)
//                              and the transcribe.cpp (Canary 180M Flash) schema
// PR #470 accepted only Flat for legacy files, so the unquantised canary-1b-v2
// (what the regression suite pins) failed to load: "has shape [1,1024,2048,1],
// expected [1024,2048]". ne[] is ggml's 4-dim shape.
enum class Pointwise { Invalid, Flat, Singleton };

inline Pointwise pointwise_layout(const int64_t ne[4], int64_t in, int64_t out) {
    if (ne[0] == in && ne[1] == out && ne[2] == 1 && ne[3] == 1)
        return Pointwise::Flat;
    if (ne[0] == 1 && ne[1] == in && ne[2] == out && ne[3] == 1)
        return Pointwise::Singleton;
    return Pointwise::Invalid;
}

} // namespace canary_layout
