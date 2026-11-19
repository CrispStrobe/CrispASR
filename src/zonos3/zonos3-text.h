// zonos3-text.h — the one UTF-8 codec of the C++ tree. Decoding follows Python's errors="replace": each maximal
// subpart of an ill-formed sequence (Unicode Table 3-7) decodes as one U+FFFD.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct utf8_cp {
    uint32_t cp;
    size_t b0, b1; // byte range of the code point
};

// Decode one code point at s[i] and advance i (never past s.size()); requires i < s.size().
uint32_t utf8_next(std::string_view s, size_t& i);
// Code points of s[b, e) with their byte ranges.
std::vector<utf8_cp> utf8_spans(std::string_view s, size_t b, size_t e);
// Number of code points.
size_t utf8_length(std::string_view s);
// Bytes utf8_append writes for cp (surrogate / > U+10FFFF: 3, the U+FFFD written instead).
size_t utf8_encoded_length(uint32_t cp);
// Append cp as UTF-8 (surrogate / > U+10FFFF: U+FFFD).
void utf8_append(std::string& out, uint32_t cp);
std::u32string utf8_to_u32(std::string_view s);
std::string utf8_from_u32(std::u32string_view s);
// Python str.isspace() for one code point (also what re's \s matches in a str pattern).
bool is_py_space(uint32_t cp);

inline bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
