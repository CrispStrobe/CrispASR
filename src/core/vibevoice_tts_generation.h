#pragma once

#include <algorithm>
#include <cstdint>

namespace core_vibevoice_tts {
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
} // namespace core_vibevoice_tts
