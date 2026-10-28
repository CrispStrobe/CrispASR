// Native Parakeet word ownership. Shared by grouping and window metadata.
// Token indices, not timestamp/text matching: adjacent words may share frames.
#pragma once
#include <utility>
#include <vector>

namespace core_parakeet_words {
inline bool is_punct_only(const char* s) {
    if (!s || !*s)
        return false;
    const unsigned char* p = (const unsigned char*)s;
    while (*p) {
        unsigned char c = *p;
        if (c < 0x80) {
            if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '\'' || c == '"' ||
                  c == '(' || c == ')' || c == '-'))
                return false;
            p++;
        } else if (c == 0xE3 && p[1] == 0x80 && p[2] >= 0x80 && p[2] <= 0xBF) {
            p += 3; // U+3000–U+303F CJK Symbols and Punctuation
        } else if (c == 0xE3 && p[1] == 0x83 && p[2] == 0xBB) {
            p += 3; // U+30FB ・
        } else if (c == 0xEF && p[1] == 0xBC && ((p[2] >= 0x81 && p[2] <= 0x8F) || p[2] == 0x9F)) {
            p += 3; // U+FF01–U+FF0F, U+FF1F
        } else {
            return false;
        }
    }
    return true;
}

// Ranges partition the original token stream when any word exists. Empty/BOS
// tokens before the first word belong to it; skipped tokens between words and
// after the last word belong to the preceding word. Grouping still ignores
// them for word text/times/probability, while selected windows retain raw data.
// A standalone SentencePiece space still starts the next non-punctuation word.
template <class TextAt> std::vector<std::pair<int, int>> token_ranges(int count, TextAt text_at) {
    int starts = 0;
    for (int i = 0; i < count; ++i) {
        const char* t = text_at(i);
        if (t[0] == ' ' && t[1] != '\0')
            ++starts;
    }
    const bool space_prefix_style = starts >= 2;
    std::vector<std::pair<int, int>> out;
    int begin = 0;
    bool have_word = false;
    bool pending_space = false;
    for (int i = 0; i < count; ++i) {
        const char* t = text_at(i);
        if (!t[0])
            continue;
        if (t[0] == ' ' && !t[1]) {
            pending_space = have_word;
            continue;
        }
        const bool is_punct = is_punct_only(t);
        if (have_word && !is_punct && (pending_space || t[0] == ' ' || !space_prefix_style)) {
            out.push_back({begin, i});
            begin = i;
        }
        if (!is_punct)
            pending_space = false;
        have_word = true;
    }
    if (have_word)
        out.push_back({begin, count});
    return out;
}
} // namespace core_parakeet_words
