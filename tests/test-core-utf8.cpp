// test-core-utf8.cpp — core/utf8.h codepoint counting.
//
// Pins the property the f5-tts duration bug turned on: a BYTE count is not a
// character count outside ASCII. f5 derived a speech rate as chars/sec from the
// reference and multiplied it by strlen() of the target, so a Devanagari line
// (3 bytes/char) was estimated ~3x too long; the ODE solve scales with the mel
// sequence length, so a ~2 s line took minutes.
//
// The ASCII cases are the ones that make the bug invisible in CI: for ASCII the
// two counts are identical, so an ASCII-only fixture cannot fail. That is why
// the non-ASCII cases below exist, and why they carry explicit byte-vs-codepoint
// assertions rather than just a count.
#include "core/utf8.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("core_utf8: ascii counts equal byte counts", "[unit][core-utf8]") {
    const std::string s = "The quick brown fox";
    REQUIRE(core_utf8::length(s) == s.size());
    REQUIRE(core_utf8::length(s.c_str()) == s.size());
    REQUIRE(core_utf8::length(s.data(), s.size()) == s.size());
}

TEST_CASE("core_utf8: multi-byte scripts count characters, not bytes", "[unit][core-utf8]") {
    // Devanagari: 3 bytes per codepoint. This is the f5-tts case.
    const std::string hi = "नमस्ते";
    REQUIRE(hi.size() == 18);            // 6 codepoints x 3 bytes
    REQUIRE(core_utf8::length(hi) == 6); // ...but 6 characters
    REQUIRE(core_utf8::length(hi) * 3 == hi.size());

    // CJK: also 3 bytes per codepoint.
    const std::string zh = "你好世界";
    REQUIRE(zh.size() == 12);
    REQUIRE(core_utf8::length(zh) == 4);

    // 2-byte (Cyrillic) and 4-byte (emoji, outside the BMP).
    REQUIRE(core_utf8::length(std::string("привет")) == 6);
    REQUIRE(core_utf8::length(std::string("\U0001F600\U0001F601")) == 2);
}

TEST_CASE("core_utf8: mixed scripts add up per codepoint", "[unit][core-utf8]") {
    const std::string mixed = "abc नमस्ते 123";
    // 3 ascii + space + 6 devanagari + space + 3 ascii = 14 codepoints,
    // but 3 + 1 + 18 + 1 + 3 = 26 bytes.
    REQUIRE(mixed.size() == 26);
    REQUIRE(core_utf8::length(mixed) == 14);
}

TEST_CASE("core_utf8: degrades on malformed input rather than throwing", "[unit][core-utf8]") {
    // These callers estimate durations and length ratios; a slightly-off count
    // on invalid bytes beats an exception. Pinning the documented behaviour so
    // nobody "fixes" it into a validator.
    const std::string lone_continuation("\x80\x80", 2); // continuation bytes with no lead
    REQUIRE(core_utf8::length(lone_continuation) == 0);

    // A truncated 3-byte sequence still counts its leading byte once.
    const std::string truncated("\xE0\xA4", 2);
    REQUIRE(core_utf8::length(truncated) == 1);

    // Embedded NUL: the sized overload sees past it, the C-string one stops.
    const std::string with_nul("ab\0cd", 5);
    REQUIRE(core_utf8::length(with_nul.data(), with_nul.size()) == 5);
    REQUIRE(core_utf8::length(with_nul.c_str()) == 2);
}

TEST_CASE("core_utf8: empty and null", "[unit][core-utf8]") {
    REQUIRE(core_utf8::length(std::string()) == 0);
    REQUIRE(core_utf8::length("") == 0);
    REQUIRE(core_utf8::length((const char*)nullptr) == 0);
    REQUIRE(core_utf8::length(nullptr, 0) == 0);
}

TEST_CASE("core_utf8: agrees with the copies it replaces", "[unit][core-utf8]") {
    // core_lid_probe::utf8_length, crispasr.cpp's static utf8_len and kokoro's
    // lambda are all the same non-continuation-byte loop. Reproduce it here and
    // assert agreement, so consolidating onto this header is provably a no-op
    // for existing callers rather than assumed to be.
    auto legacy = [](const std::string& s) {
        size_t n = 0;
        for (size_t i = 0; i < s.size(); i++)
            if (((unsigned char)s[i] & 0xC0) != 0x80)
                n++;
        return n;
    };
    for (const std::string& s : {std::string("hello"), std::string("नमस्ते"), std::string("你好"),
                                 std::string("\U0001F600"), std::string("mixed नमस्ते 42"), std::string()}) {
        INFO("input bytes = " << s.size());
        REQUIRE(core_utf8::length(s) == legacy(s));
    }
}
