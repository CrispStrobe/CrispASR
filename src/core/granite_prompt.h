#pragma once

#include <cstdint>
#include <string>

namespace core_granite_prompt {

// Original USER:/ASSISTANT: template, including GPT-2's merged "?\n" token.
// Share the constants and suffix spelling between CLI and C ABI.
static constexpr int32_t kPrefix4[] = {6584, 25, 220};
static constexpr int kNumPrefix4 = 3;
static constexpr int32_t kSuffix4[] = {4919, 499,  1380, 3191,  279,  8982, 1139, 264,
                                       5439, 3645, 5380, 36660, 3931, 2891, 25};
static constexpr int kNumSuffix4 = 15;

static inline std::string legacy_suffix(const std::string& instruction, const std::string& prefix_text = "") {
    return instruction + "\n ASSISTANT:" + prefix_text;
}

} // namespace core_granite_prompt
