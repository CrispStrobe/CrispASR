#pragma once

#include <string>

namespace core_qwen3_prompt {

// Hotwords are context, independent of an explicit question and language bias.
inline std::string build(int audio_tokens, std::string system, const std::string& question, const std::string& prefill,
                         const std::string& hotwords) {
    if (!hotwords.empty()) {
        if (!system.empty() && system.back() != ' ')
            system += ' ';
        system += "The following words may appear in the audio: " + hotwords + ".";
    }
    std::string text = "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
    text.reserve(text.size() + (size_t)audio_tokens * 13 + question.size() + prefill.size() + 64);
    for (int i = 0; i < audio_tokens; ++i)
        text += "<|audio_pad|>";
    text += "<|audio_end|>";
    if (!question.empty())
        text += '\n' + question;
    text += "<|im_end|>\n<|im_start|>assistant\n" + prefill;
    return text;
}

} // namespace core_qwen3_prompt
