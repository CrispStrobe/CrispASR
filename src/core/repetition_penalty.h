// core/repetition_penalty.h -- HF-style repetition penalty, shared by the AR
// TTS backends (bt2-tts #594, outetts).
//
// Applied ONCE per distinct token in `history`, exactly as HF
// RepetitionPenaltyLogitsProcessor (gather/scatter: duplicates read the same
// original score and write the same penalised value). Applying it per
// OCCURRENCE compounds it to penalty^k: over a few hundred audio frames every
// common code (pauses, steady vowels) is suppressed by orders of magnitude and
// sampling leaves the model's distribution.

#pragma once

#include <cstdint>
#include <vector>

namespace core_sampling {

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

} // namespace core_sampling
