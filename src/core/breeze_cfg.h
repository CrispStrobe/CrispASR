// core/breeze_cfg.h -- classifier-free guidance for Breeze TTS 2 (bt2-tts).
//
// Breeze guides by running INDEPENDENT prompts through the backbone -- not one
// prompt with different conditioning vectors -- and combining their logits at
// every backbone step and again per codebook inside the depth decoder
// (models/generation_breeze.py). Which prompts exist is decided by the scale
// alone (breeze_infer/templates.py::prepare_inputs):
//
//   scale == 1   the positive prompt alone; the combine would reduce to it
//   scale == 0   the negative prompt alone
//   otherwise    negative + scale * (positive - negative)
//
// The negative prompt is the positive one minus the instruction. A reference
// clip, when given, sits in BOTH, so it is not part of the plan. Only the
// instruction templates define a negative prompt, which is why a request
// without one is a single branch at any scale.
//
// Header-only and model-free so tests/test-breeze-cfg.cpp can pin it.

#pragma once

#include <vector>

namespace breeze_cfg {

// One prompt branch. Branch 0 is the base of the combine and its scale is
// unused.
struct Branch {
    bool instruction = false;
    float scale = 0.0f;
};

inline std::vector<Branch> plan(bool has_instruction, float scale) {
    if (!has_instruction || scale == 0.0f)
        return {{false, 0.0f}};
    if (scale == 1.0f)
        return {{true, 0.0f}};
    return {{false, 0.0f}, {true, scale}};
}

// rows is [branches.size(), n], one logits row per branch; out is [n].
inline void combine(const float* rows, int n, const std::vector<Branch>& branches, float* out) {
    for (int i = 0; i < n; i++) {
        float v = rows[i];
        for (size_t b = 1; b < branches.size(); b++)
            v += branches[b].scale * (rows[b * (size_t)n + i] - rows[i]);
        out[i] = v;
    }
}

} // namespace breeze_cfg
