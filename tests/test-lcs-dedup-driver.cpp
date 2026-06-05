// test-lcs-dedup-driver.cpp — unit tests for the segment-aware LCS dedup
// driver that wraps `crispasr_lcs::lcs_dedup_prefix_count` and physically
// removes duplicate leading tokens / words / segment-text from adjacent
// overlap-save chunks (issue #89 / #114 follow-up).
//
// These tests exercise the driver on synthetic `crispasr_segment`
// fixtures so a model load is not required.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_backend.h"
#include "crispasr_lcs_dedup.h"

#include <vector>

using crispasr_lcs::apply_lcs_chunk_dedup;
using crispasr_lcs::collect_tail_token_ids;
using crispasr_lcs::drop_leading_tokens;

namespace {
// Build a one-segment chunk with the given token ids + texts. Word list
// is derived 1:1 from tokens for simplicity (the driver tolerates
// looser word/token alignment in real data).
crispasr_segment make_seg(const std::vector<std::pair<int32_t, const char*>>& toks, int64_t t0_cs = 0) {
    crispasr_segment s;
    int64_t t = t0_cs;
    for (const auto& [id, text] : toks) {
        crispasr_token tok;
        tok.id = id;
        tok.text = text;
        tok.t0 = t;
        tok.t1 = t + 8; // one 80 ms parakeet encoder frame per token
        s.tokens.push_back(tok);
        crispasr_word w;
        w.text = text;
        w.t0 = t;
        w.t1 = t + 8;
        s.words.push_back(w);
        s.text += text;
        t += 8;
    }
    if (!s.tokens.empty()) {
        s.t0 = s.tokens.front().t0;
        s.t1 = s.tokens.back().t1;
    }
    return s;
}
} // namespace

TEST_CASE("collect_tail_token_ids returns all ids when tail<=0", "[unit][lcs-dedup]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg({{1, "a"}, {2, "b"}, {3, "c"}}));
    segs.push_back(make_seg({{4, "d"}, {5, "e"}}, /*t0_cs=*/24));
    auto ids = collect_tail_token_ids(segs, /*tail=*/-1);
    REQUIRE(ids == std::vector<int32_t>{1, 2, 3, 4, 5});
}

TEST_CASE("collect_tail_token_ids clips to tail size", "[unit][lcs-dedup]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg({{10, "x"}, {11, "y"}, {12, "z"}, {13, "w"}, {14, "v"}}));
    auto ids = collect_tail_token_ids(segs, /*tail=*/2);
    REQUIRE(ids == std::vector<int32_t>{13, 14});
}

TEST_CASE("drop_leading_tokens: drop whole segment when n covers it", "[unit][lcs-dedup]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg({{1, "a"}, {2, "b"}}));
    segs.push_back(make_seg({{3, "c"}, {4, "d"}}, /*t0_cs=*/16));
    int dropped = drop_leading_tokens(segs, /*n=*/2);
    REQUIRE(dropped == 2);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].text == "cd");
}

TEST_CASE("drop_leading_tokens: partial slice rebuilds text", "[unit][lcs-dedup]") {
    std::vector<crispasr_segment> segs;
    segs.push_back(make_seg({{1, "a"}, {2, "b"}, {3, "c"}, {4, "d"}}));
    int dropped = drop_leading_tokens(segs, /*n=*/2);
    REQUIRE(dropped == 2);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].text == "cd");
    REQUIRE(segs[0].tokens.size() == 2);
    REQUIRE(segs[0].tokens[0].id == 3);
    REQUIRE(segs[0].tokens[1].id == 4);
}

TEST_CASE("apply_lcs_chunk_dedup: 3-token boundary dup is sliced", "[unit][lcs-dedup]") {
    // Reconstructs the issue #89 scenario: chunk[0] ends with [5, 6, 7]
    // and chunk[1] begins with [5, 6, 7, 8, 9, 10]. After dedup chunk[1]
    // should start at token id 8.
    std::vector<std::vector<crispasr_segment>> per_slice;
    per_slice.push_back({make_seg({{1, "a"}, {5, "x"}, {6, "y"}, {7, "z"}})});
    per_slice.push_back({make_seg({{5, "x"}, {6, "y"}, {7, "z"}, {8, "p"}, {9, "q"}, {10, "r"}}, /*t0_cs=*/24)});

    apply_lcs_chunk_dedup(per_slice, /*delay_tokens=*/8, /*min_lcs_length=*/1);

    REQUIRE(per_slice.size() == 2);
    REQUIRE(per_slice[0].size() == 1);
    REQUIRE(per_slice[0][0].text == "axyz");
    REQUIRE(per_slice[1].size() == 1);
    REQUIRE(per_slice[1][0].text == "pqr");
    REQUIRE(per_slice[1][0].tokens.front().id == 8);
}

TEST_CASE("apply_lcs_chunk_dedup: no overlap leaves chunks unchanged", "[unit][lcs-dedup]") {
    std::vector<std::vector<crispasr_segment>> per_slice;
    per_slice.push_back({make_seg({{1, "a"}, {2, "b"}})});
    per_slice.push_back({make_seg({{100, "p"}, {101, "q"}}, /*t0_cs=*/16)});
    apply_lcs_chunk_dedup(per_slice, /*delay_tokens=*/8, /*min_lcs_length=*/1);
    REQUIRE(per_slice[0][0].text == "ab");
    REQUIRE(per_slice[1][0].text == "pq");
}

TEST_CASE("apply_lcs_chunk_dedup: min_lcs_length gates short matches", "[unit][lcs-dedup]") {
    // 1-token match would normally slice; with min_lcs_length=3 it must not.
    std::vector<std::vector<crispasr_segment>> per_slice;
    per_slice.push_back({make_seg({{1, "a"}, {99, "ok"}})});
    per_slice.push_back({make_seg({{99, "ok"}, {100, "p"}, {101, "q"}}, /*t0_cs=*/16)});
    apply_lcs_chunk_dedup(per_slice, /*delay_tokens=*/8, /*min_lcs_length=*/3);
    REQUIRE(per_slice[1][0].text == "okpq");
}

TEST_CASE("apply_lcs_chunk_dedup is a no-op for a single chunk", "[unit][lcs-dedup]") {
    std::vector<std::vector<crispasr_segment>> per_slice;
    per_slice.push_back({make_seg({{1, "a"}, {2, "b"}})});
    apply_lcs_chunk_dedup(per_slice, /*delay_tokens=*/8, /*min_lcs_length=*/1);
    REQUIRE(per_slice.size() == 1);
    REQUIRE(per_slice[0][0].text == "ab");
}
