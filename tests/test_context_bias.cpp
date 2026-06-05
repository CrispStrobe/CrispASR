// tests/test_context_bias.cpp — Catch2 unit tests for CTC-WS phrase-boost trie.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/asr_context_bias.h"

#include <numeric>
#include <vector>
#include <string>

using namespace core_context_bias;

// ── Helpers ──────────────────────────────────────────────────────────

// Trivial "tokenizer": each character maps to its ASCII value.
static std::vector<int32_t> char_tokenize(const std::string& s) {
    std::vector<int32_t> ids;
    for (char c : s) ids.push_back((int32_t)(unsigned char)c);
    return ids;
}

// ── Trie construction ────────────────────────────────────────────────

TEST_CASE("empty trie", "[context_bias]") {
    Trie trie;
    REQUIRE(trie.empty());

    MatchState st;
    std::vector<float> logits(256, 0.0f);
    apply_bias(trie, st, logits.data(), 256, 2.0f);
    // All logits should remain zero — no bias applied
    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    REQUIRE(sum == 0.0f);
}

TEST_CASE("single hotword insert + match", "[context_bias]") {
    Trie trie;
    // Insert "cat" as token IDs [99, 97, 116]
    trie.insert({99, 97, 116}, 5.0f);
    trie.build_failure_links();
    REQUIRE_FALSE(trie.empty());

    MatchState st;
    const int V = 256;
    const float boost = 2.0f;

    // At root: only token 99 ('c') should be boosted
    std::vector<float> logits(V, 0.0f);
    apply_bias(trie, st, logits.data(), V, boost);
    REQUIRE(logits[99] > 0.0f); // 'c' boosted
    REQUIRE(logits[97] == 0.0f); // 'a' not boosted yet
    REQUIRE(logits[0] == 0.0f);  // unrelated token

    // Advance with 'c' (99)
    advance(trie, st, 99);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost);
    REQUIRE(logits[97] > 0.0f); // 'a' boosted (continues "ca")
    // 'c' is also boosted via failure link → root → 'c' (new "cat" start)
    REQUIRE(logits[99] > 0.0f);

    // Advance with 'a' (97)
    advance(trie, st, 97);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost);
    // 't' (116) should get boost + endpoint boost (5.0)
    REQUIRE(logits[116] >= boost + 5.0f);

    // Advance with 't' — hotword complete, back to root-ish
    advance(trie, st, 116);
}

TEST_CASE("multiple hotwords with overlap", "[context_bias]") {
    Trie trie;
    // "ab" and "abc"
    trie.insert({1, 2}, 3.0f);
    trie.insert({1, 2, 3}, 5.0f);
    trie.build_failure_links();

    MatchState st;
    const int V = 10;
    const float boost = 1.0f;

    // Root: token 1 ('a') should be boosted
    std::vector<float> logits(V, 0.0f);
    apply_bias(trie, st, logits.data(), V, boost);
    REQUIRE(logits[1] > 0.0f);

    advance(trie, st, 1);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost);
    // Token 2 ('b') should be boosted
    REQUIRE(logits[2] > 0.0f);

    advance(trie, st, 2);
    std::fill(logits.begin(), logits.end(), 0.0f);
    apply_bias(trie, st, logits.data(), V, boost);
    // Token 3 ('c') should be boosted (continues "abc")
    REQUIRE(logits[3] > 0.0f);
}

TEST_CASE("mismatch resets via failure links", "[context_bias]") {
    Trie trie;
    trie.insert({1, 2, 3}, 5.0f);
    trie.build_failure_links();

    MatchState st;

    // Match prefix "1, 2"
    advance(trie, st, 1);
    advance(trie, st, 2);

    // Now mismatch with token 9 — should reset
    advance(trie, st, 9);
    REQUIRE(st.node == 0); // back at root

    // Token 1 should still be boostable from root
    std::vector<float> logits(10, 0.0f);
    apply_bias(trie, st, logits.data(), 10, 1.0f);
    REQUIRE(logits[1] > 0.0f);
}

TEST_CASE("failure link suffix match", "[context_bias]") {
    Trie trie;
    // "abc" and "bc"
    trie.insert({1, 2, 3}, 5.0f);
    trie.insert({2, 3}, 3.0f);
    trie.build_failure_links();

    MatchState st;
    // Start matching "a, b"
    advance(trie, st, 1); // in "a" prefix
    advance(trie, st, 2); // in "ab" prefix

    // Now mismatch with 9 — should fall back.
    // But "b" is a prefix of "bc", so via failure links we might land
    // at the "b" node of the "bc" pattern.
    advance(trie, st, 9);
    // After mismatch, we should be back at root
    // (9 is not a child of any node in the "bc" path either)
}

// ── parse_hotwords ───────────────────────────────────────────────────

TEST_CASE("parse_hotwords basic", "[context_bias]") {
    auto hw = parse_hotwords("Acme Corp, Sandra Berenz, GPU-PB");
    REQUIRE(hw.size() == 3);
    REQUIRE(hw[0] == "Acme Corp");
    REQUIRE(hw[1] == "Sandra Berenz");
    REQUIRE(hw[2] == "GPU-PB");
}

TEST_CASE("parse_hotwords with boost suffix", "[context_bias]") {
    auto hw = parse_hotwords("Berenz^5.0, NVIDIA^3.0, plain");
    REQUIRE(hw.size() == 3);
    REQUIRE(hw[0] == "Berenz^5.0");
    REQUIRE(hw[1] == "NVIDIA^3.0");
    REQUIRE(hw[2] == "plain");
}

TEST_CASE("parse_hotwords empty and whitespace", "[context_bias]") {
    REQUIRE(parse_hotwords("").empty());
    REQUIRE(parse_hotwords("  ,  ,  ").empty());
    auto hw = parse_hotwords("  foo  ,  bar  ");
    REQUIRE(hw.size() == 2);
    REQUIRE(hw[0] == "foo");
    REQUIRE(hw[1] == "bar");
}

// ── build_trie convenience ───────────────────────────────────────────

TEST_CASE("build_trie with char tokenizer", "[context_bias]") {
    auto trie = build_trie({"cat", "car"}, char_tokenize, 2.0f);
    REQUIRE_FALSE(trie.empty());
    // Should have nodes for: root → c → a → t, and c → a → r
    // At least 5 nodes: root, c, a, t, r
    REQUIRE(trie.nodes.size() >= 5);
}

TEST_CASE("build_trie with boost suffix", "[context_bias]") {
    auto trie = build_trie({"cat^10.0"}, char_tokenize, 2.0f);
    REQUIRE_FALSE(trie.empty());

    MatchState st;
    advance(trie, st, 'c');
    advance(trie, st, 'a');

    // At "ca" node, 't' should get boost + 10.0 endpoint
    std::vector<float> logits(256, 0.0f);
    apply_bias(trie, st, logits.data(), 256, 1.5f);
    REQUIRE(logits['t'] >= 11.5f); // 1.5 prefix + 10.0 endpoint
}

// ── edge cases ───────────────────────────────────────────────────────

TEST_CASE("single-token hotword", "[context_bias]") {
    Trie trie;
    trie.insert({42}, 5.0f);
    trie.build_failure_links();

    MatchState st;
    std::vector<float> logits(100, 0.0f);
    apply_bias(trie, st, logits.data(), 100, 1.0f);
    // Token 42 should be boosted with prefix_boost + endpoint_boost
    REQUIRE(logits[42] >= 6.0f); // 1.0 + 5.0
}

TEST_CASE("token ID out of vocab range ignored", "[context_bias]") {
    Trie trie;
    trie.insert({999}, 5.0f); // token 999 > vocab size 100
    trie.build_failure_links();

    MatchState st;
    std::vector<float> logits(100, 0.0f);
    apply_bias(trie, st, logits.data(), 100, 1.0f);
    // Should not crash, and no logit should be modified
    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    REQUIRE(sum == 0.0f);
}

TEST_CASE("repeated advance and reset", "[context_bias]") {
    Trie trie;
    trie.insert({1, 2, 3}, 5.0f);
    trie.build_failure_links();

    MatchState st;

    // Match full sequence twice
    for (int rep = 0; rep < 2; rep++) {
        advance(trie, st, 1);
        advance(trie, st, 2);
        advance(trie, st, 3);
    }

    // Manual reset
    st.reset();
    REQUIRE(st.node == 0);
}
