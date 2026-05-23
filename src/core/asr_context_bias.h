// src/core/asr_context_bias.h — CTC-WS phrase-boost trie (PLAN #98 Phase A).
//
// Aho-Corasick multi-pattern matcher over token-ID sequences. When the
// CTC/TDT decoder is scoring frame logits, the trie adds a configurable
// boost to tokens that continue an active hotword prefix match. This is
// the same "shallow fusion" approach described in NeMo's CTC-WS word
// boosting and the TurboBias paper (arxiv.org/abs/2508.07014).
//
// Usage:
//   1. Build a trie from hotword strings + a tokenizer function.
//   2. Before each frame's argmax, call apply_bias() with the current
//      trie state and the logits buffer — it adds the boost in-place.
//   3. After argmax, call advance() with the chosen token to move the
//      trie state forward (or reset on mismatch).
//
// Header-only, no ggml dependency. Pure CPU, ~200 LOC.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace core_context_bias {

// A single node in the Aho-Corasick trie.
struct TrieNode {
    std::unordered_map<int32_t, int> children; // token_id → child node index
    int fail = 0;                              // failure link (index into nodes[])
    float boost = 0.0f;                        // accumulated boost at this node (>0 if a hotword ends here)
    int depth = 0;                             // number of tokens from root to this node
};

struct Trie {
    std::vector<TrieNode> nodes;

    Trie() { nodes.emplace_back(); } // root node at index 0

    // Insert a hotword as a sequence of token IDs with a given boost score.
    void insert(const std::vector<int32_t>& token_ids, float boost) {
        int cur = 0;
        for (int32_t tid : token_ids) {
            auto it = nodes[cur].children.find(tid);
            if (it == nodes[cur].children.end()) {
                int next = (int)nodes.size();
                nodes[cur].children[tid] = next;
                nodes.emplace_back();
                nodes.back().depth = nodes[cur].depth + 1;
                cur = next;
            } else {
                cur = it->second;
            }
        }
        nodes[cur].boost += boost; // accumulate if duplicate
    }

    // Build Aho-Corasick failure links. Must be called after all inserts
    // and before any matching.
    void build_failure_links() {
        std::queue<int> q;
        // Root's children fail to root
        for (auto& [tid, child] : nodes[0].children) {
            nodes[child].fail = 0;
            q.push(child);
        }
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (auto& [tid, v] : nodes[u].children) {
                int f = nodes[u].fail;
                while (f != 0 && nodes[f].children.find(tid) == nodes[f].children.end())
                    f = nodes[f].fail;
                auto it = nodes[f].children.find(tid);
                nodes[v].fail = (it != nodes[f].children.end() && it->second != v) ? it->second : 0;
                // Propagate boost from suffix matches
                nodes[v].boost += nodes[nodes[v].fail].boost;
                q.push(v);
            }
        }
    }

    bool empty() const { return nodes.size() <= 1; }
};

// Runtime state for one ongoing decode. One per utterance.
struct MatchState {
    int node = 0; // current position in the trie (0 = root)

    void reset() { node = 0; }
};

// Advance the match state by one emitted token. Call this AFTER argmax
// with the chosen token_id to update the trie position.
inline void advance(const Trie& trie, MatchState& st, int32_t token_id) {
    int cur = st.node;
    while (cur != 0 && trie.nodes[cur].children.find(token_id) == trie.nodes[cur].children.end())
        cur = trie.nodes[cur].fail;
    auto it = trie.nodes[cur].children.find(token_id);
    st.node = (it != trie.nodes[cur].children.end()) ? it->second : 0;
}

// Apply bias to logits for the current frame. For each token_id that is
// a valid next step in the trie (from the current state or any suffix
// via failure links), add `boost` to logits[token_id].
//
// This is the "shallow fusion" step: tokens that continue a hotword
// prefix get a log-prob boost, making them more likely to win argmax.
inline void apply_bias(const Trie& trie, const MatchState& st, float* logits, int n_vocab, float default_boost) {
    if (trie.empty())
        return;

    // Walk from current node + all failure-chain nodes, collect valid next tokens
    int cur = st.node;
    while (true) {
        for (const auto& [tid, child_idx] : trie.nodes[cur].children) {
            if (tid >= 0 && tid < n_vocab) {
                // Boost = default_boost (for prefix continuation) + any
                // endpoint boost if a hotword completes at the child
                float b = default_boost;
                if (trie.nodes[child_idx].boost > 0.0f)
                    b += trie.nodes[child_idx].boost;
                logits[tid] += b;
            }
        }
        if (cur == 0)
            break;
        cur = trie.nodes[cur].fail;
    }
}

// Convenience: build a trie from a list of hotword strings using a
// tokenizer function. The tokenizer converts a string to token IDs.
using Tokenizer = std::function<std::vector<int32_t>(const std::string&)>;

inline Trie build_trie(const std::vector<std::string>& hotwords, const Tokenizer& tokenize,
                       float boost_per_word = 2.0f) {
    Trie trie;
    for (const auto& hw : hotwords) {
        if (hw.empty())
            continue;
        // Parse optional boost suffix: "word^5.0"
        float boost = boost_per_word;
        std::string word = hw;
        auto caret = hw.rfind('^');
        if (caret != std::string::npos && caret + 1 < hw.size()) {
            try {
                boost = std::stof(hw.substr(caret + 1));
                word = hw.substr(0, caret);
            } catch (...) {
                // Not a valid float suffix, treat the whole thing as the word
            }
        }
        auto ids = tokenize(word);
        if (!ids.empty()) {
            trie.insert(ids, boost);
        }
    }
    trie.build_failure_links();
    return trie;
}

// Parse a comma-separated hotword string into a vector.
inline std::vector<std::string> parse_hotwords(const std::string& s) {
    std::vector<std::string> result;
    if (s.empty())
        return result;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos)
            end = s.size();
        // Trim whitespace
        size_t a = start, b = end;
        while (a < b && (s[a] == ' ' || s[a] == '\t'))
            a++;
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t'))
            b--;
        if (b > a)
            result.push_back(s.substr(a, b - a));
        start = end + 1;
    }
    return result;
}

} // namespace core_context_bias
