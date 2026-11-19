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

#include <cstdint>
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

// Repetition penalty over the generated first-codebook history (#594).
// Applied ONCE per distinct token, exactly as upstream's
// GeneratedTokenRepetitionPenaltyLogitsProcessor (HF gather/scatter:
// duplicates read the same original score and write the same penalised
// value). Applying it per OCCURRENCE compounds it to penalty^k: after a few
// hundred frames every common code (pauses, steady vowels) is suppressed by
// orders of magnitude, the backbone loses its text alignment and, on long
// chunks, starts speaking the <ins_bos> instruction aloud.
inline void apply_repetition_penalty(float* logits, int n, const std::vector<int32_t>& history, float penalty) {
    if (penalty == 1.0f || history.empty())
        return;
    std::vector<uint8_t> seen((size_t)n, 0);
    for (int32_t tok : history) {
        if (tok < 0 || tok >= n || seen[(size_t)tok])
            continue;
        seen[(size_t)tok] = 1;
        // The HF convention: multiply negatives, divide the rest.
        logits[tok] = logits[tok] < 0.0f ? logits[tok] * penalty : logits[tok] / penalty;
    }
}

} // namespace breeze_cfg
