#pragma once

#include <string>

namespace core_qwen3_prompt {

// Raw context and wrapped hotwords are independent of questions and language bias.
// #602 Rumble-2-Flash chat_template.jinja: the system text is prefixed with
// the control tokens unless it already carries them ("<digits>"/"<nodigits>").
inline std::string with_system_prefix(const char* prefix, const std::string& system) {
    if (!prefix || !*prefix)
        return system;
    if (system.find("<digits>") != std::string::npos || system.find("<nodigits>") != std::string::npos)
        return system;
    return std::string(prefix) + system;
}

inline std::string build(int audio_tokens, std::string system, const std::string& question, const std::string& prefill,
                         const std::string& hotwords, const std::string& context = "") {
    if (!context.empty()) {
        if (!system.empty() && system.back() != ' ')
            system += ' ';
        system += context;
    }
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
