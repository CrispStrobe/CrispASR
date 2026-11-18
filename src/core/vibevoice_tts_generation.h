#pragma once

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cstddef>
#include <cstdlib>

namespace core_vibevoice_tts {
inline int diffusion_steps(int configured) {
    return configured > 0 ? configured : 20;
}

// Match the existing realtime precedence: an explicit parameter wins over env.
inline float guidance_scale(bool base, float configured, const char* environment) {
    if (configured > 0.0f)
        return configured;
    if (environment) {
        const float value = static_cast<float>(std::atof(environment));
        if (value > 0.0f)
            return value;
    }
    return base ? 1.3f : 3.0f;
}

// Legacy GGUFs predate context metadata. These are the original base model
// decoder_config values (1.5B: hidden_size=1536; 7B: hidden_size=3584).
inline int legacy_context_length(int hidden_size) {
    return hidden_size == 3584 ? 32768 : 65536;
}

// Upstream modeling_vibevoice_inference.py generate(), revision 952326ddb264:
// default max_length_times=2 applies to
// the entire processor prompt, including speaker-reference placeholders.
inline int generation_budget(int configured, int prefix_len, int context_length) {
    if (prefix_len <= 0 || prefix_len >= context_length)
        return 0;
    const int64_t prompt_budget = 2LL * prefix_len;
    const int64_t context_budget = int64_t(context_length) - prefix_len;
    const int64_t requested = configured > 0 ? configured : context_budget;
    return static_cast<int>(std::min({prompt_budget, context_budget, requested}));
}
// Base serving tokenizer: BOS is None, EOS is <|endoftext|>. Upstream
// VibeVoiceTokenConstraintProcessor masks every other vocabulary row before
// greedy selection. Scan IDs in order to preserve torch.argmax tie-breaking.
inline int select_base_token(const float* logits, size_t count, bool constrain) {
    if (count == 0)
        return -1;
    auto score = [&](size_t i) {
        const bool allowed = i == 151643 || i == 151652 || i == 151653 || i == 151654;
        return constrain && !allowed ? logits[i] + -std::numeric_limits<float>::infinity() : logits[i];
    };
    int best = 0;
    float best_score = score(0);
    for (size_t i = 1; i < count; ++i) {
        const float value = score(i);
        if (value > best_score || (constrain && std::isnan(value) && !std::isnan(best_score))) {
            best = static_cast<int>(i);
            best_score = value;
        }
    }
    return best;
}
} // namespace core_vibevoice_tts
