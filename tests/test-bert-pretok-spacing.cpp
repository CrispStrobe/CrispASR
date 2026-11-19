// test-bert-pretok-spacing.cpp — core_bert::pretokenize's whitespace flags.
//
// fireredpunc rebuilds the user's text from the pre-token WORDS. Its blueprint
// spacing rule (a space only between two ASCII-alphanumeric tokens whose
// predecessor predicted no mark) deleted every space between Cyrillic words
// ("Онсказал,чтосоздал...") and the space after any comma ("so,my") in the
// 2026-11 extended sweep. The fix keeps the input's own spacing, which needs
// `out_space_before` to say exactly where the input had whitespace — and to
// line up 1:1 with the returned words, or the rebuild silently mis-spaces.
#include <catch2/catch_test_macros.hpp>

#include "core/bert_pretok.h"

#include <string>
#include <vector>

static std::vector<uint8_t> flags(const std::string& s, std::vector<std::string>* words = nullptr) {
    std::vector<uint8_t> sp;
    auto w = core_bert::pretokenize(s, &sp);
    REQUIRE(w.size() == sp.size());
    REQUIRE(w == core_bert::pretokenize(s)); // the flags never change the split
    if (words)
        *words = w;
    return sp;
}

TEST_CASE("space flags follow the input", "[pretokenize]") {
    std::vector<std::string> w;
    auto sp = flags("And so, my fellow", &w);
    REQUIRE(w == std::vector<std::string>{"And", "so", ",", "my", "fellow"});
    REQUIRE(sp == std::vector<uint8_t>{0, 1, 0, 1, 1});
}

TEST_CASE("cyrillic words keep their separators", "[pretokenize]") {
    std::vector<std::string> w;
    auto sp = flags("Он сказал, что", &w);
    REQUIRE(w.size() == 4);
    REQUIRE(sp == std::vector<uint8_t>{0, 1, 0, 1});
}

TEST_CASE("CJK ideographs split without whitespace", "[pretokenize]") {
    auto sp = flags("今天 好");
    REQUIRE(sp == std::vector<uint8_t>{0, 0, 1});
}

TEST_CASE("leading and repeated whitespace", "[pretokenize]") {
    auto sp = flags("  a \t b");
    REQUIRE(sp == std::vector<uint8_t>{1, 1});
}
