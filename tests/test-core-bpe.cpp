// test-core-bpe.cpp — unit tests for core/bpe.h GPT-2 byte-level BPE.
//
// Verifies: byte_encoder table properties, utf8_encode, bytes_to_unicode,
// bpe_one (greedy merge), token_bytes_to_utf8, detokenize, per-byte
// fallback. No model load, pure CPU.

#include <catch2/catch_test_macros.hpp>

#include "core/bpe.h"

#include <string>
#include <unordered_map>
#include <vector>

// ── byte_encoder table ─────────────────────────────────────────────────────

TEST_CASE("bpe — byte_encoder has 256 entries", "[unit][bpe]") {
    const auto& enc = core_bpe::byte_encoder();
    REQUIRE(enc.size() == 256);
}

TEST_CASE("bpe — byte_encoder maps all 256 values to unique codepoints", "[unit][bpe]") {
    const auto& enc = core_bpe::byte_encoder();
    std::unordered_map<int, int> seen;
    for (int b = 0; b < 256; b++) {
        REQUIRE(seen.find(enc[b]) == seen.end()); // must be unique
        seen[enc[b]] = b;
    }
    REQUIRE(seen.size() == 256);
}

TEST_CASE("bpe — printable ASCII bytes map to themselves", "[unit][bpe]") {
    const auto& enc = core_bpe::byte_encoder();
    // GPT-2 table: bytes 0x21..0x7E map to themselves (identity)
    for (int b = 0x21; b <= 0x7E; b++) {
        REQUIRE(enc[b] == b);
    }
}

// ── utf8_encode ────────────────────────────────────────────────────────────

TEST_CASE("bpe — utf8_encode ASCII", "[unit][bpe]") {
    std::string out;
    core_bpe::utf8_encode('A', out);
    REQUIRE(out == "A");
}

TEST_CASE("bpe — utf8_encode 2-byte", "[unit][bpe]") {
    std::string out;
    core_bpe::utf8_encode(0x00E9, out); // e-acute
    REQUIRE(out.size() == 2);
    REQUIRE((unsigned char)out[0] == 0xC3);
    REQUIRE((unsigned char)out[1] == 0xA9);
}

TEST_CASE("bpe — utf8_encode 3-byte", "[unit][bpe]") {
    std::string out;
    core_bpe::utf8_encode(0x4E16, out); // CJK character '世'
    REQUIRE(out.size() == 3);
}

// ── bytes_to_unicode ───────────────────────────────────────────────────────

TEST_CASE("bpe — bytes_to_unicode for ASCII printable", "[unit][bpe]") {
    // Printable ASCII maps to itself in the byte_encoder, so bytes_to_unicode
    // of "Hi" should produce "Hi" (identity)
    std::string result = core_bpe::bytes_to_unicode("Hi", 2);
    REQUIRE(result == "Hi");
}

TEST_CASE("bpe — bytes_to_unicode for null byte", "[unit][bpe]") {
    // Byte 0x00 is NOT in the printable range, so it maps to a non-ASCII
    // codepoint (256 or higher), producing a multi-byte UTF-8 string.
    char zero = '\0';
    std::string result = core_bpe::bytes_to_unicode(&zero, 1);
    REQUIRE(result.size() >= 2); // must be multi-byte UTF-8
}

// ── byte_decoder round-trip ────────────────────────────────────────────────

TEST_CASE("bpe — byte_decoder inverts byte_encoder", "[unit][bpe]") {
    const auto& enc = core_bpe::byte_encoder();
    const auto& dec = core_bpe::byte_decoder();
    REQUIRE(dec.size() == 256);
    for (int b = 0; b < 256; b++) {
        auto it = dec.find((uint32_t)enc[b]);
        REQUIRE(it != dec.end());
        REQUIRE(it->second == (uint8_t)b);
    }
}

// ── token_bytes_to_utf8 ────────────────────────────────────────────────────

TEST_CASE("bpe — token_bytes_to_utf8 round-trips ASCII", "[unit][bpe]") {
    // "hello" in byte-encoded form = "hello" (ASCII maps to itself)
    std::string result = core_bpe::token_bytes_to_utf8("hello");
    REQUIRE(result == "hello");
}

// ── bpe_one ────────────────────────────────────────────────────────────────

TEST_CASE("bpe — bpe_one whole-token lookup", "[unit][bpe]") {
    // Build a tiny vocab where "hello" is a single token
    std::unordered_map<std::string, int32_t> token_to_id = {{"h", 0}, {"e", 1}, {"l", 2}, {"o", 3}, {"hello", 10}};
    std::unordered_map<std::string, int32_t> merge_rank = {
        {"h e", 0},
        {"he l", 1},
        {"hel l", 2},
        {"hell o", 3},
    };

    // Byte-encode "hello" first (all printable ASCII → identity)
    std::string be = core_bpe::bytes_to_unicode("hello", 5);
    std::vector<int32_t> ids;
    core_bpe::bpe_one(token_to_id, merge_rank, be, ids);
    REQUIRE(ids.size() == 1);
    REQUIRE(ids[0] == 10); // merged all the way to "hello"
}

TEST_CASE("bpe — bpe_one partial merge", "[unit][bpe]") {
    // Vocab has "ab" and "c" but not "abc"
    std::unordered_map<std::string, int32_t> token_to_id = {{"a", 0}, {"b", 1}, {"c", 2}, {"ab", 10}};
    std::unordered_map<std::string, int32_t> merge_rank = {
        {"a b", 0},
    };

    std::string be = core_bpe::bytes_to_unicode("abc", 3);
    std::vector<int32_t> ids;
    core_bpe::bpe_one(token_to_id, merge_rank, be, ids);
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == 10); // "ab"
    REQUIRE(ids[1] == 2);  // "c"
}

TEST_CASE("bpe — bpe_one no merges (empty merge table)", "[unit][bpe]") {
    // Without merges, each byte-encoded codepoint is its own symbol
    std::unordered_map<std::string, int32_t> token_to_id = {{"h", 0}, {"i", 1}};
    std::unordered_map<std::string, int32_t> merge_rank; // empty

    std::string be = core_bpe::bytes_to_unicode("hi", 2);
    std::vector<int32_t> ids;
    core_bpe::bpe_one(token_to_id, merge_rank, be, ids);
    REQUIRE(ids.size() == 2);
    REQUIRE(ids[0] == 0); // "h"
    REQUIRE(ids[1] == 1); // "i"
}

TEST_CASE("bpe — bpe_one empty input", "[unit][bpe]") {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    std::vector<int32_t> ids;
    core_bpe::bpe_one(token_to_id, merge_rank, "", ids);
    REQUIRE(ids.empty());
}

// ── detokenize ─────────────────────────────────────────────────────────────

TEST_CASE("bpe — detokenize simple ASCII tokens", "[unit][bpe]") {
    // Token strings in the GGUF are stored in byte-encoded form. For
    // printable ASCII (0x21..0x7E), byte-encoded == raw text. But space
    // (0x20) is NOT in the printable range and maps to a non-identity
    // codepoint. So we byte-encode the space first.
    std::string space_encoded = core_bpe::bytes_to_unicode(" ", 1);
    std::vector<std::string> id_to_token = {"Hello", space_encoded + "world"};
    std::vector<int32_t> ids = {0, 1};
    std::string result = core_bpe::detokenize(id_to_token, ids.data(), ids.size());
    REQUIRE(result == "Hello world");
}

TEST_CASE("bpe — detokenize skips out-of-range IDs", "[unit][bpe]") {
    std::vector<std::string> id_to_token = {"ok"};
    std::vector<int32_t> ids = {0, 999, -1};
    std::string result = core_bpe::detokenize(id_to_token, ids.data(), ids.size());
    REQUIRE(result == "ok");
}
