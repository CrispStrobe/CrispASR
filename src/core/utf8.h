// core/utf8.h — UTF-8 codepoint counting, in one place.
//
// The same four-line loop had been written out four separate times before this
// header existed: `core_lid_probe::utf8_length` (core/lid_probe.h),
// a file-static `utf8_len` in crispasr.cpp, a lambda in kokoro.cpp, and — as of
// the f5-tts duration fix — a fourth in f5_tts.cpp. They agree, which is luck
// rather than design: each is a hand-rolled continuation-byte test.
//
// Counting matters because a BYTE count is not a character count outside ASCII,
// and the two get mixed. f5-tts derived a speech rate as chars/sec from the
// reference and then multiplied it by a byte count of the target, so Devanagari
// (3 bytes/char) inflated the estimate ~3x; since the ODE solve scales with the
// mel sequence length, a two-second line became minutes of synthesis.
#pragma once

#include <cstddef>
#include <string>

namespace core_utf8 {

// Number of UTF-8 codepoints in [s, s + n_bytes).
//
// Counts NON-CONTINUATION bytes, i.e. every byte that is not 10xxxxxx. That is
// the leading byte of each sequence, so well-formed input gives the codepoint
// count exactly.
//
// On MALFORMED input it degrades rather than rejecting: a stray continuation
// byte is not counted, and an over-long or truncated sequence still counts its
// leading byte once. Callers here are estimating a duration or a length ratio,
// where a slightly-off count on invalid input beats throwing; nothing in this
// header validates UTF-8, and it should not be used as a validity check.
inline size_t length(const char* s, size_t n_bytes) {
    size_t n = 0;
    for (size_t i = 0; i < n_bytes; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80)
            n++;
    return n;
}

inline size_t length(const std::string& s) {
    return length(s.data(), s.size());
}

// NUL-terminated overload.
inline size_t length(const char* s) {
    if (!s)
        return 0;
    size_t n = 0;
    for (; *s; ++s)
        if (((unsigned char)*s & 0xC0) != 0x80)
            n++;
    return n;
}

} // namespace core_utf8
