// test-ctc-punct-space — NeMo's space-before-punctuation decode step
// (src/core/ctc_punct_space.h), checked against the behaviour of
// tokenizer_utils.extract_punctuation_from_vocab + AbstractCTCDecoding's
// re.sub(r'(\s)(<marks>)', r'\2', text). Weight-free.

#include <catch2/catch_test_macros.hpp>

#include "core/ctc_punct_space.h"

#include <set>
#include <string>
#include <vector>

using core_ctc_punct::strip_space_before;
using core_ctc_punct::supported_punctuation;

// Head of the stt_ru_fastconformer_hybrid_large_pc vocabulary plus the bare
// marks it also carries (',' is id 1021) and a few special shapes.
static const std::vector<std::string> kVocab = {
    "<unk>",
    "\xE2\x96\x81.",
    "\xE2\x96\x81,",
    "е",
    "а",
    "\xE2\x96\x81в",
    "\xE2\x96\x81?",
    ",",
    ".",
    "?",
    "-",
    "[PAD]",
    "<pad>",
    "##,",
    "+",
    "$",
    " ",
};

TEST_CASE("supported punctuation follows extract_punctuation_from_vocab", "[ctc-punct]") {
    const std::set<uint32_t> p = supported_punctuation(kVocab);
    // Bare marks count; ▁-prefixed, [..], <..> and ## tokens do not; S* ('+', '$') is not P*.
    CHECK(p == std::set<uint32_t>{',', '.', '?', '-'});
    // A vocabulary whose marks only exist with ▁ has no supported punctuation.
    CHECK(supported_punctuation({"\xE2\x96\x81,", "\xE2\x96\x81.", "a"}).empty());
}

TEST_CASE("one space before a supported mark is removed", "[ctc-punct]") {
    const std::set<uint32_t> p = supported_punctuation(kVocab);
    CHECK(strip_space_before(" Он сказал , что создал звонок , работающий от вай-фай .", p) ==
          " Он сказал, что создал звонок, работающий от вай-фай.");
    // Only ONE whitespace goes (NeMo keeps the rest as model error).
    CHECK(strip_space_before("a  ,", p) == "a ,");
    // Unsupported marks and symbols keep their space.
    CHECK(strip_space_before("a ! b + c", p) == "a ! b + c");
    // Nothing to do without supported marks.
    CHECK(strip_space_before("a , b", {}) == "a , b");
}

TEST_CASE("offset map follows the removed bytes", "[ctc-punct]") {
    const std::set<uint32_t> p = {','};
    const std::string in = " да , нет";
    std::vector<int> kept;
    const std::string out = strip_space_before(in, p, &kept);
    REQUIRE(out == " да, нет");
    REQUIRE(kept.size() == in.size() + 1);
    CHECK(kept[0] == 0);
    CHECK(kept[in.size()] == (int)out.size());
    // The piece " ," (offset 5, the space before ',') maps to "," alone.
    const size_t comma = in.find(',');
    CHECK(kept[comma - 1] == kept[comma]);
    CHECK(out.substr(kept[comma], kept[comma + 1] - kept[comma]) == ",");
    // Every later byte shifts left by exactly one.
    for (size_t i = comma; i <= in.size(); i++)
        CHECK(kept[i] == (int)i - 1);
}
