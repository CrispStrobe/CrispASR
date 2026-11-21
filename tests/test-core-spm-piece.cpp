// test-core-spm-piece.cpp — core_spm::piece_to_text, SentencePiece piece →
// visible text. Hermetic.
//
// Byte fallback (#605): a SentencePiece model trained with byte_fallback spells
// a character outside its vocabulary as one "<0xHH>" piece per UTF-8 byte.
// Haas-ja (3,072 pieces) does this for rare kanji: 董 is <0xE8><0x91><0xA3>.
// Treating those like "<unk>" dropped the character — on the asr-ja-bench FLEURS
// subset our output read 骨品 for 骨董品, and lost 娯, 徘徊, 焉, 竣, 踵, 癌, 塵, ...

#include "core/spm_piece.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

std::string decode(const std::vector<std::string>& pieces) {
    std::string out;
    for (const auto& p : pieces)
        out += core_spm::piece_to_text(p);
    return out;
}

} // namespace

TEST_CASE("spm piece: word-start marker and plain pieces", "[unit][spm-piece]") {
    REQUIRE(core_spm::piece_to_text("\xE2\x96\x81"
                                    "foo") == " foo");
    REQUIRE(core_spm::piece_to_text("\xE2\x96\x81") == " ");
    REQUIRE(core_spm::piece_to_text("bar") == "bar");
    REQUIRE(core_spm::piece_to_text("猫") == "猫");
    REQUIRE(core_spm::piece_to_text("") == "");
}

TEST_CASE("spm piece: control pieces are dropped", "[unit][spm-piece]") {
    for (const char* p : {"<unk>", "<s>", "</s>", "<pad>", "<blk>", "<0x>", "<0xZZ>", "<0x123>"}) {
        INFO(p);
        REQUIRE(core_spm::piece_to_text(p) == "");
    }
}

TEST_CASE("spm piece: byte-fallback pieces become their byte", "[unit][spm-piece]") {
    REQUIRE(core_spm::piece_to_text("<0x41>") == "A");
    REQUIRE(core_spm::piece_to_text("<0x0A>") == "\n");
    REQUIRE(core_spm::piece_to_text("<0xe8>") == std::string(1, (char)0xE8));
    // 骨董品: 骨 and 品 are in the vocabulary, 董 is three byte pieces.
    REQUIRE(decode({"\xE2\x96\x81"
                    "骨",
                    "<0xE8>", "<0x91>", "<0xA3>", "品"}) == " 骨董品");
    // 徘徊: both characters byte-encoded.
    REQUIRE(decode({"<0xE5>", "<0xBE>", "<0x98>", "<0xE5>", "<0xBE>", "<0x8A>"}) == "徘徊");
}
