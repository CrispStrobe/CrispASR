// test-core-sentencepiece.cpp — unit tests for core/sentencepiece.h
//
// Verifies the extracted SentencePiece unigram Viterbi tokenizer against:
//   (a) hand-computed expected segmentations (proving it is TRUE Viterbi,
//       not greedy-longest — the discriminating property);
//   (b) an INDEPENDENT brute-force max-score segmenter (different code),
//       fuzzed over random vocabs/strings;
//   (c) the single-byte unk fallback, merge_consecutive_unk, and
//       oov_score_default behaviors documented as the t5/indextts contract.
//
// Pure CPU, no models. Ground truth = the unigram Viterbi objective
// (maximize sum of piece log-scores), which is what t5_translate.cpp's
// tokenize_sp implements and what the extraction must reproduce.

#include <catch2/catch_test_macros.hpp>

#include "core/sentencepiece.h"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using core_spm::Config;
using core_spm::tokenize;

namespace {

// Independent reference: brute-force best-score segmentation over the SAME
// scoring rules as the header (real piece score, else single-byte unk
// penalty). Written from scratch (recursive DP) so a shared bug can't hide.
// Operates on the ALREADY-PREPROCESSED string (no ▁ logic) to isolate the
// DP from the space-handling; tests call it with prepend_space=false.
struct RefResult {
    double score;
    std::vector<int32_t> ids;
};

RefResult ref_viterbi(const std::string& s, const std::unordered_map<std::string, int32_t>& vocab,
                      const std::vector<float>& scores, const Config& cfg) {
    const int n = (int)s.size();
    std::vector<double> best(n + 1, -1e300);
    std::vector<int32_t> tok(n + 1, -1);
    std::vector<int> prev(n + 1, -1);
    best[0] = 0.0;
    for (int i = 0; i < n; i++) {
        if (best[i] <= -1e299)
            continue;
        int max_j = std::min(n, i + cfg.max_piece_len);
        for (int j = i + 1; j <= max_j; j++) {
            if (cfg.utf8_aligned && j < n && (((unsigned char)s[j]) & 0xC0) == 0x80)
                continue;
            std::string sub = s.substr(i, j - i);
            auto it = vocab.find(sub);
            double cand;
            int32_t id;
            if (it != vocab.end()) {
                int32_t tid = it->second;
                double sc = (tid >= 0 && tid < (int32_t)scores.size()) ? (double)scores[tid]
                                                                       : (double)cfg.oov_score_default;
                cand = best[i] + sc;
                id = tid;
            } else if (j == i + 1) {
                cand = best[i] + (double)cfg.unk_penalty;
                id = cfg.unk_id;
            } else {
                continue;
            }
            if (cand > best[j]) {
                best[j] = cand;
                tok[j] = id;
                prev[j] = i;
            }
        }
    }
    RefResult r;
    r.score = best[n];
    int pos = n;
    while (pos > 0 && tok[pos] >= -1) {
        if (prev[pos] < 0 && tok[pos] < 0)
            break;
        r.ids.push_back(tok[pos]);
        pos = prev[pos];
        if (pos < 0)
            break;
    }
    std::reverse(r.ids.begin(), r.ids.end());
    return r;
}

// Simple deterministic PRNG (no <random> nondeterminism across libs).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (uint32_t)(s >> 11);
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint32_t)(hi - lo + 1)); }
};

} // namespace

TEST_CASE("core_spm: true Viterbi, not greedy-longest", "[unit][core-spm]") {
    // "abc": greedy-longest would grab "abc" (score -10); Viterbi prefers
    // "ab"+"c" (-1 + -1 = -2). The extraction must pick the Viterbi answer.
    std::unordered_map<std::string, int32_t> v = {
        {"ab", 0}, {"a", 1}, {"b", 2}, {"c", 3}, {"abc", 4},
    };
    std::vector<float> sc = {-1.0f, -5.0f, -5.0f, -1.0f, -10.0f};
    Config cfg;
    cfg.unk_id = 99;
    cfg.merge_consecutive_unk = false;

    auto ids = tokenize("abc", v, sc, cfg, /*prepend_space=*/false);
    REQUIRE(ids == std::vector<int32_t>{0, 3}); // "ab","c"
}

TEST_CASE("core_spm: single-byte unk fallback + merge", "[unit][core-spm]") {
    std::unordered_map<std::string, int32_t> v = {{"x", 0}};
    std::vector<float> sc = {-1.0f};
    Config cfg;
    cfg.unk_id = 99;
    cfg.unk_penalty = -100.0f;

    SECTION("no merge: each unknown byte is its own unk") {
        cfg.merge_consecutive_unk = false;
        auto ids = tokenize("xyz", v, sc, cfg, false);
        REQUIRE(ids == std::vector<int32_t>{0, 99, 99}); // x, <unk>, <unk>
    }
    SECTION("merge: consecutive unks collapse") {
        cfg.merge_consecutive_unk = true;
        auto ids = tokenize("xyz", v, sc, cfg, false);
        REQUIRE(ids == std::vector<int32_t>{0, 99}); // x, <unk>
    }
}

TEST_CASE("core_spm: oov_score_default is honored (indextts -20 vs t5 0)", "[unit][core-spm]") {
    // The big piece "zz" has id 5, which is out of range of `scores`, so its
    // score is cfg.oov_score_default. It competes with covering "zz" as two
    // "z" pieces (id 0, score -1 each → total -2). The OOV default decides.
    std::unordered_map<std::string, int32_t> v = {{"zz", 5}, {"z", 0}};
    std::vector<float> sc = {-1.0f}; // only id 0 has a score; id 5 is OOR
    Config cfg;
    cfg.unk_id = 99;
    cfg.merge_consecutive_unk = false;

    SECTION("oov better than the two sub-pieces → picks the big piece") {
        cfg.oov_score_default = -1.5f; // > -2 (two z's)
        auto ids = tokenize("zz", v, sc, cfg, false);
        REQUIRE(ids == std::vector<int32_t>{5});
    }
    SECTION("oov worse than the two sub-pieces → splits") {
        cfg.oov_score_default = -50.0f; // < -2
        auto ids = tokenize("zz", v, sc, cfg, false);
        REQUIRE(ids == std::vector<int32_t>{0, 0});
    }
}

TEST_CASE("core_spm: leading-space / ▁ handling", "[unit][core-spm]") {
    // With prepend_space, input "a b" → "▁a▁b". Vocab has the ▁-pieces.
    const std::string U = "\xE2\x96\x81"; // ▁
    std::unordered_map<std::string, int32_t> v = {
        {U + "a", 0}, {U + "b", 1}, {"a", 2}, {"b", 3},
    };
    std::vector<float> sc = {-1.0f, -1.0f, -5.0f, -5.0f};
    Config cfg;
    cfg.unk_id = 99;
    cfg.merge_consecutive_unk = false;

    auto ids = tokenize("a b", v, sc, cfg, /*prepend_space=*/true);
    REQUIRE(ids == std::vector<int32_t>{0, 1}); // ▁a, ▁b
}

TEST_CASE("core_spm: matches independent brute-force Viterbi (fuzz)", "[unit][core-spm]") {
    Rng rng(0xC0FFEEULL);
    const std::string alphabet = "abcde";
    int mismatches = 0;
    int checked = 0;

    for (int trial = 0; trial < 2000; trial++) {
        // Random vocab: all 1-char + some random 2-3 char pieces, random scores.
        std::unordered_map<std::string, int32_t> v;
        std::vector<float> sc;
        int32_t next_id = 0;
        for (char c : alphabet) {
            v[std::string(1, c)] = next_id++;
            sc.push_back(-(float)rng.range(1, 9));
        }
        int extra = rng.range(0, 8);
        for (int e = 0; e < extra; e++) {
            int len = rng.range(2, 3);
            std::string piece;
            for (int k = 0; k < len; k++)
                piece.push_back(alphabet[rng.range(0, (int)alphabet.size() - 1)]);
            if (v.find(piece) == v.end()) {
                v[piece] = next_id++;
                sc.push_back(-(float)rng.range(1, 5)); // 2-3grams cheaper → tempting
            }
        }
        // Random input string.
        int slen = rng.range(1, 10);
        std::string s;
        for (int k = 0; k < slen; k++)
            s.push_back(alphabet[rng.range(0, (int)alphabet.size() - 1)]);

        Config cfg;
        cfg.unk_id = 99;
        cfg.unk_penalty = -100.0f;
        cfg.merge_consecutive_unk = false;

        auto got = tokenize(s, v, sc, cfg, /*prepend_space=*/false);
        auto ref = ref_viterbi(s, v, sc, cfg);

        checked++;
        // Compare achieved scores (ties on different-but-equal-score paths
        // are allowed: assert the header's path scores == the optimum).
        double got_score = 0.0;
        {
            // Recompute score of the header's chosen ids by re-walking s.
            // All pieces here are pure ASCII so we can reconstruct lengths.
            size_t pos = 0;
            for (int32_t id : got) {
                if (id == cfg.unk_id) {
                    got_score += cfg.unk_penalty;
                    pos += 1;
                } else {
                    // find the piece string for id
                    std::string piece;
                    for (auto& kv : v)
                        if (kv.second == id)
                            piece = kv.first;
                    got_score += (id < (int32_t)sc.size()) ? sc[id] : cfg.oov_score_default;
                    pos += piece.size();
                }
            }
            REQUIRE(pos == s.size()); // segmentation must cover the string
        }
        if (std::abs(got_score - ref.score) > 1e-4) {
            mismatches++;
            if (mismatches <= 3) {
                WARN("string='" << s << "' header_score=" << got_score << " optimum=" << ref.score);
            }
        }
    }
    INFO("checked " << checked << " random cases");
    REQUIRE(mismatches == 0);
}
