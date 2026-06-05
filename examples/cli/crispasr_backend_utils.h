#pragma once

#include "whisper_params.h"

#include <cctype>
#include <string>

inline bool crispasr_backend_should_use_gpu(const whisper_params& params) {
    return params.use_gpu && params.gpu_backend != "cpu";
}

// Strip ASCII punctuation characters from `s` in-place. Keeps ASCII letters,
// digits, whitespace, and any byte ≥ 0x80 (so multi-byte UTF-8 sequences pass
// through untouched). Used by adapters whose runtime produces punctuated
// output but the user passed --no-punctuation.
inline void crispasr_strip_ascii_punctuation(std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char u = (unsigned char)c;
        if ((u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') || (u >= '0' && u <= '9') || u == ' ' || u == '\t' ||
            u == '\n' || u >= 0x80) {
            out += c;
        }
    }
    s = std::move(out);
    // Collapse runs of multiple spaces left over by stripped punctuation
    // (e.g. "Hi, there" → "Hi  there" → "Hi there").
    std::string collapsed;
    collapsed.reserve(s.size());
    bool prev_space = false;
    for (char c : s) {
        const bool is_space = (c == ' ' || c == '\t' || c == '\n');
        if (is_space) {
            if (!prev_space)
                collapsed += ' ';
            prev_space = true;
        } else {
            collapsed += c;
            prev_space = false;
        }
    }
    s = std::move(collapsed);
    // Trim leading / trailing whitespace.
    auto lo = s.find_first_not_of(' ');
    auto hi = s.find_last_not_of(' ');
    s = (lo == std::string::npos) ? "" : s.substr(lo, hi - lo + 1);
}

// ASCII-lowercase pass. Multi-byte UTF-8 bytes (≥ 0x80) pass through
// unchanged — matches the historical CTC convention of "lowercase Latin
// transcript", without breaking non-Latin scripts.
inline void crispasr_lowercase_ascii(std::string& s) {
    for (char& c : s) {
        unsigned char u = (unsigned char)c;
        if (u >= 'A' && u <= 'Z')
            c = (char)(u + 32);
    }
}
