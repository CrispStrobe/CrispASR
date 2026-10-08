#pragma once
#include <string>

namespace core_nemotron_text {
// The model closes each sentence with its language tag as an ordinary token
// ("… Besprechung. <de-DE> Wir haben …"). That is the model talking about the
// audio, not transcript: it reached SRTs, word lists and the realtime stream
// verbatim. Remove every `<xx-XX>` / `<xxx-XX>` and the space it leaves.
inline bool is_lang_tag(const std::string& s, size_t i, size_t* len) {
    if (i >= s.size() || s[i] != '<')
        return false;
    size_t j = i + 1;
    size_t lower = 0;
    while (j < s.size() && s[j] >= 'a' && s[j] <= 'z' && lower < 3) {
        ++j;
        ++lower;
    }
    if (lower < 2 || j + 3 >= s.size() || s[j] != '-')
        return false;
    if (!(s[j + 1] >= 'A' && s[j + 1] <= 'Z' && s[j + 2] >= 'A' && s[j + 2] <= 'Z' && s[j + 3] == '>'))
        return false;
    *len = j + 4 - i;
    return true;
}

inline std::string strip_lang_tags(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size();) {
        size_t len = 0;
        if (is_lang_tag(in, i, &len)) {
            i += len;
            // One separator is enough where the tag stood.
            if (!out.empty() && out.back() == ' ' && i < in.size() && in[i] == ' ')
                ++i;
            continue;
        }
        out += in[i++];
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

} // namespace core_nemotron_text
