// test-core-wordpiece.cpp — unit tests for core/wordpiece.h BERT WordPiece tokenizer.
//
// Verifies: build_map, lookup, cased/uncased detection, BasicTokenizer
// (whitespace + punctuation splitting), WordPiece greedy longest-match,
// [UNK] fallback, UTF-8 handling. No model load, pure CPU.

#include <catch2/catch_test_macros.hpp>

#include "core/wordpiece.h"

#include <string>
#include <vector>

// Helper: build a minimal BERT-like vocab for testing.
static core_wordpiece::Tokenizer make_vocab(bool cased) {
    core_wordpiece::Tokenizer tok;
    // Standard BERT special tokens
    tok.id_to_token = {
        "[PAD]",   // 0
        "[UNK]",   // 1 — note: unk_id default is 100, we override below
        "[CLS]",   // 2
        "[SEP]",   // 3
        "hello",   // 4
        "world",   // 5
        "##ing",   // 6
        "test",    // 7
        "##s",     // 8
        "##ed",    // 9
        "go",      // 10
        "##od",    // 11
        ".",       // 12
        ",",       // 13
        "good",    // 14
        "morning", // 15
    };
    if (cased) {
        tok.id_to_token.push_back("Hello"); // 16
    }
    tok.unk_id = 1;
    tok.cls_id = 2;
    tok.sep_id = 3;
    tok.pad_id = 0;
    tok.build_map();
    return tok;
}

TEST_CASE("wordpiece — build_map populates token_to_id", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    REQUIRE(tok.loaded);
    REQUIRE(tok.token_to_id.size() == tok.id_to_token.size());
    REQUIRE(tok.token_to_id["hello"] == 4);
    REQUIRE(tok.token_to_id["##ing"] == 6);
}

TEST_CASE("wordpiece — lookup known token", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    REQUIRE(tok.lookup("hello") == 4);
    REQUIRE(tok.lookup("##ing") == 6);
}

TEST_CASE("wordpiece — lookup unknown returns unk_id", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    REQUIRE(tok.lookup("xyzzy") == tok.unk_id);
}

TEST_CASE("wordpiece — uncased detection (no 'Hello' in vocab)", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    // Vocab has "hello" but not "Hello" → should auto-detect as uncased
    REQUIRE(tok.do_lower == true);
}

TEST_CASE("wordpiece — cased detection ('Hello' in vocab)", "[unit][wordpiece]") {
    auto tok = make_vocab(true);
    // Vocab has "Hello" → should auto-detect as cased
    REQUIRE(tok.do_lower == false);
}

TEST_CASE("wordpiece — single known word", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    auto ids = tok.tokenize("hello");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 4); // "hello"
}

TEST_CASE("wordpiece — two words separated by space", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    auto ids = tok.tokenize("hello world");
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == 4); // "hello"
    REQUIRE(ids[1] == 5); // "world"
}

TEST_CASE("wordpiece — subword splitting with ## prefix", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    // "testing" should split as "test" + "##ing"
    auto ids = tok.tokenize("testing");
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == 7); // "test"
    REQUIRE(ids[1] == 6); // "##ing"
}

TEST_CASE("wordpiece — punctuation splits into own token", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    auto ids = tok.tokenize("hello.");
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == 4);  // "hello"
    REQUIRE(ids[1] == 12); // "."
}

TEST_CASE("wordpiece — unknown word produces [UNK]", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    // "xyz" has no vocab coverage at all
    auto ids = tok.tokenize("xyz");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == tok.unk_id);
}

TEST_CASE("wordpiece — uncased lowering", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    REQUIRE(tok.do_lower == true);
    // "Hello" → lowered to "hello" → id 4
    // (Only ASCII lowering — the tokenizer handles A-Z → a-z)
    auto ids = tok.tokenize("Hello");
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 4); // "hello"
}

TEST_CASE("wordpiece — multiple subword splits", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    // "goods" → greedy longest-match: "good" (14) + "##s" (8)
    auto ids = tok.tokenize("goods");
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == 14); // "good"
    REQUIRE(ids[1] == 8);  // "##s"
}

TEST_CASE("wordpiece — empty input produces empty output", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    auto ids = tok.tokenize("");
    REQUIRE(ids.empty());
}

TEST_CASE("wordpiece — whitespace-only input produces empty output", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    auto ids = tok.tokenize("   \t\n");
    REQUIRE(ids.empty());
}

TEST_CASE("wordpiece — comma splits correctly", "[unit][wordpiece]") {
    auto tok = make_vocab(false);
    auto ids = tok.tokenize("hello, world");
    REQUIRE(ids.size() == 3);
    REQUIRE(ids[0] == 4);  // "hello"
    REQUIRE(ids[1] == 13); // ","
    REQUIRE(ids[2] == 5);  // "world"
}
