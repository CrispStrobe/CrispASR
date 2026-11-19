#include "zonos3-text.h"

namespace {

constexpr uint32_t REPLACEMENT = 0xFFFD;

bool encodable(uint32_t cp) {
    return cp <= 0x10FFFF && (cp < 0xD800 || cp > 0xDFFF);
}

} // namespace

uint32_t utf8_next(std::string_view s, size_t& i) {
    const auto c = (unsigned char)s[i];
    if (c < 0x80) {
        ++i;
        return c;
    }
    int len = 0;
    unsigned char lo = 0x80, hi = 0xBF; // allowed range of the second byte
    if (c >= 0xC2 && c <= 0xDF)
        len = 2;
    else if (c == 0xE0)
        len = 3, lo = 0xA0;
    else if (c == 0xED)
        len = 3, hi = 0x9F;
    else if (c >= 0xE1 && c <= 0xEF)
        len = 3;
    else if (c == 0xF0)
        len = 4, lo = 0x90;
    else if (c == 0xF4)
        len = 4, hi = 0x8F;
    else if (c >= 0xF1 && c <= 0xF3)
        len = 4;
    if (len == 0) {
        ++i;
        return REPLACEMENT;
    }
    uint32_t cp = c & (0x7F >> len);
    for (int k = 1; k < len; ++k) {
        if (i + k >= s.size()) {
            i += k;
            return REPLACEMENT;
        }
        const auto d = (unsigned char)s[i + k];
        if (d < (k == 1 ? lo : 0x80) || d > (k == 1 ? hi : 0xBF)) {
            i += k;
            return REPLACEMENT;
        }
        cp = (cp << 6) | (d & 0x3F);
    }
    i += len;
    return cp;
}

std::vector<utf8_cp> utf8_spans(std::string_view s, size_t b, size_t e) {
    const std::string_view seg = s.substr(0, e);
    std::vector<utf8_cp> out;
    for (size_t i = b; i < e;) {
        const size_t at = i;
        const uint32_t cp = utf8_next(seg, i);
        out.push_back({cp, at, i});
    }
    return out;
}

size_t utf8_length(std::string_view s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ++n)
        utf8_next(s, i);
    return n;
}

size_t utf8_encoded_length(uint32_t cp) {
    if (!encodable(cp))
        return 3;
    return cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
}

void utf8_append(std::string& out, uint32_t cp) {
    if (!encodable(cp))
        cp = REPLACEMENT;
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

std::u32string utf8_to_u32(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();)
        out.push_back((char32_t)utf8_next(s, i));
    return out;
}

std::string utf8_from_u32(std::u32string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t cp : s)
        utf8_append(out, (uint32_t)cp);
    return out;
}

bool is_py_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}
