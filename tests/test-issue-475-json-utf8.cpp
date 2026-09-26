// test-issue-475-json-utf8.cpp - -ojf must be valid UTF-8 JSON when a byte-level
// BPE token holds part of a character (issue #475, qwen3-asr on Korean).
// Compiles crispasr_output.cpp directly - no model.
#include "crispasr_output.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size())
            return false;
        for (size_t k = 1; k < n; k++)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80)
                return false;
        i += n;
    }
    return true;
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("#475 json_escape replaces invalid UTF-8 and keeps valid text", "[issue-475]") {
    CHECK(crispasr_json_escape("하나, 둘") == "하나, 둘");
    CHECK(crispasr_json_escape(" \xeb") == " \xEF\xBF\xBD");                               // truncated 3-byte sequence
    CHECK(crispasr_json_escape("\x91\x98") == "\xEF\xBF\xBD\xEF\xBF\xBD");                 // stray continuation bytes
    CHECK(crispasr_json_escape("\xC0\xAF") == "\xEF\xBF\xBD\xEF\xBF\xBD");                 // overlong '/'
    CHECK(crispasr_json_escape("\xED\xA0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"); // surrogate
    CHECK(crispasr_json_escape("q\"\\\n") == "q\\\"\\\\\\n");
}

TEST_CASE("#475 -ojf token texts land on character boundaries", "[issue-475]") {
    // the issue's tokens: "하" "," " \xeb" "\x91" "\x98" "," " \xec" "\x85" "\x8b" "."
    const std::vector<std::string> toks = {"하", ",", " \xeb", "\x91", "\x98", ",", " \xec", "\x85", "\x8b", "."};
    crispasr_segment seg;
    seg.t0 = 0;
    seg.t1 = 100;
    std::string joined;
    for (size_t i = 0; i < toks.size(); i++) {
        crispasr_token t;
        t.text = toks[i];
        t.t0 = (int64_t)i * 10;
        t.t1 = (int64_t)i * 10 + 10;
        t.confidence = 0.9f;
        seg.tokens.push_back(t);
        joined += toks[i];
    }
    seg.text = joined;
    REQUIRE(joined == "하, 둘, 셋.");
    const std::string path = "test-issue-475.json";
    REQUIRE(crispasr_write_json(path, {seg}, "qwen3", "m.gguf", "ko", /*full=*/true));
    const std::string doc = slurp(path);
    std::remove(path.c_str());
    CHECK(valid_utf8(doc));
    CHECK(doc.find("\xEF\xBF\xBD") == std::string::npos);      // carried, not replaced
    CHECK(doc.find("\"text\": \" 둘\"") == std::string::npos); // the " " stays on its own token...
    CHECK(doc.find("\"text\": \"둘\"") != std::string::npos);  // ...and the syllable is whole
    CHECK(doc.find("\"text\": \"셋\"") != std::string::npos);
    size_t n_tok = 0;
    for (size_t p = doc.find("\"p\": "); p != std::string::npos; p = doc.find("\"p\": ", p + 1))
        n_tok++;
    CHECK(n_tok == toks.size()); // token count and timings unchanged
}
