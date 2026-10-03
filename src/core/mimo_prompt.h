#pragma once

#include <cstring>
#include <string>

namespace core_mimo_prompt {

inline const char* language_tag(const char* language) {
    if (language && strcmp(language, "en") == 0)
        return "<english>";
    if (language && strcmp(language, "zh") == 0)
        return "<chinese>";
    return "";
}

inline std::string instruction(const std::string& language, const std::string& ask) {
    if (!ask.empty())
        return ask;
    return language == "zh" ? "请将这段语音转换为文字" : "Please transcribe this audio file";
}

} // namespace core_mimo_prompt
