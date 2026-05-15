// test_chunk_context.cpp — unit tests for chunk boundary context (issue #89).
//
// Pins four things:
//   1. audio_chunking::split_at_energy_minima produces correct, gap-free slices.
//   2. The ctx_l / ctx_r expansion math stays correct at boundaries (first/
//      last/single slice, audio shorter than the context window, etc.).
//   3. The word-level filter (primary path) that assigns each word to the slice
//      where it STARTS (w.t0 < t0_cs || w.t0 >= t1_cs) and rebuilds text.
//   4. The segment-level filter (fallback, no-words backends).
//
// All tests are pure CPU, no model load.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include "../src/core/audio_chunking.h"
#include "../examples/cli/crispasr_backend.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static crispasr_segment make_seg(int64_t t0, int64_t t1, const std::string& text = "x") {
    crispasr_segment s;
    s.t0 = t0;
    s.t1 = t1;
    s.text = text;
    return s;
}

// Word-level trim — mirrors the primary path in process_slice.
// Assigns each word to the slice it STARTS in; rebuilds segment text/t0/t1.
static void trim_words_to_window(std::vector<crispasr_segment>& segs, int64_t t0_cs, int64_t t1_cs) {
    for (auto& seg : segs) {
        seg.words.erase(
            std::remove_if(seg.words.begin(), seg.words.end(),
                           [t0_cs, t1_cs](const crispasr_word& w) { return w.t0 < t0_cs || w.t0 >= t1_cs; }),
            seg.words.end());
        if (seg.words.empty()) {
            seg.text = "";
        } else {
            std::string rebuilt;
            for (const auto& w : seg.words) {
                if (!rebuilt.empty() && !w.text.empty() && w.text[0] != ' ')
                    rebuilt += ' ';
                rebuilt += w.text;
            }
            seg.text = rebuilt;
            seg.t0 = seg.words.front().t0;
            seg.t1 = seg.words.back().t1;
        }
    }
    segs.erase(std::remove_if(segs.begin(), segs.end(), [](const crispasr_segment& s) { return s.text.empty(); }),
               segs.end());
}

// Segment-level trim — fallback path (no word timestamps).
static void trim_to_window(std::vector<crispasr_segment>& segs, int64_t t0_cs, int64_t t1_cs) {
    segs.erase(std::remove_if(segs.begin(), segs.end(),
                              [t0_cs, t1_cs](const crispasr_segment& s) { return s.t0 < t0_cs || s.t0 >= t1_cs; }),
               segs.end());
}

static crispasr_word make_word(int64_t t0, int64_t t1, const std::string& text) {
    crispasr_word w;
    w.t0 = t0;
    w.t1 = t1;
    w.text = text;
    return w;
}

// Compute context bounds as process_slice does.
struct SliceCtx {
    int start;     // expanded start sample
    int len;       // expanded length (samples)
    int64_t t0_cs; // expanded t_offset_cs
};

static SliceCtx compute_ctx(int sl_start, int sl_end, int64_t sl_t0_cs, bool has_prev, bool has_next, int total_samples,
                            int ctx_samples, int SR) {
    const int ctx_l = has_prev ? std::min(ctx_samples, sl_start) : 0;
    const int ctx_r = has_next ? std::min(ctx_samples, total_samples - sl_end) : 0;
    SliceCtx r;
    r.start = sl_start - ctx_l;
    r.len = sl_end - r.start + ctx_r;
    r.t0_cs = sl_t0_cs - (int64_t)((double)ctx_l / SR * 100.0);
    return r;
}

// ---------------------------------------------------------------------------
// 1. split_at_energy_minima — basic correctness
// ---------------------------------------------------------------------------

TEST_CASE("chunking: short audio → single slice covering everything", "[unit][chunk-context]") {
    std::vector<float> audio(1600, 0.5f); // 0.1 s @ 16 kHz
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/16000,
                                                         /*search_win=*/8000);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].first == 0);
    REQUIRE(slices[0].second == audio.size());
}

TEST_CASE("chunking: slices cover the full input without gaps or overlap", "[unit][chunk-context]") {
    // 4 s of uniform audio @ 16 kHz, split into 1 s chunks.
    const size_t SR = 16000;
    const size_t total = 4 * SR;
    std::vector<float> audio(total, 0.3f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    // At least 2 slices for 4 s / 1 s.
    REQUIRE(slices.size() >= 2);

    // Contiguous: end[i] == begin[i+1].
    for (size_t i = 1; i < slices.size(); ++i) {
        REQUIRE(slices[i].first == slices[i - 1].second);
    }
    // First slice starts at 0, last slice ends at total.
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == total);
}

TEST_CASE("chunking: all-zero audio splits into equal chunks", "[unit][chunk-context]") {
    const size_t SR = 16000;
    const size_t total = 3 * SR;
    std::vector<float> audio(total, 0.0f); // pure silence → energy minima everywhere
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), total,
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    REQUIRE(slices.size() >= 2);
    REQUIRE(slices.front().first == 0);
    REQUIRE(slices.back().second == total);
    for (size_t i = 1; i < slices.size(); ++i)
        REQUIRE(slices[i].first == slices[i - 1].second);
}

TEST_CASE("chunking: audio exactly equal to max_chunk → single slice", "[unit][chunk-context]") {
    const size_t SR = 16000;
    std::vector<float> audio(SR, 0.2f);
    auto slices = audio_chunking::split_at_energy_minima(audio.data(), audio.size(),
                                                         /*max_chunk=*/SR,
                                                         /*search_win=*/SR / 2);
    REQUIRE(slices.size() == 1);
    REQUIRE(slices[0].first == 0);
    REQUIRE(slices[0].second == SR);
}

// ---------------------------------------------------------------------------
// 2. Context expansion math
// ---------------------------------------------------------------------------

TEST_CASE("ctx: first slice gets no left context", "[unit][chunk-context]") {
    const int SR = 16000;
    const int ctx = 2 * SR; // 2 s
    const int total = 10 * SR;
    // Slice 0: [0, 3*SR)
    auto r = compute_ctx(0, 3 * SR, 0, /*has_prev=*/false, /*has_next=*/true, total, ctx, SR);
    REQUIRE(r.start == 0); // no left expansion
    REQUIRE(r.t0_cs == 0);
    REQUIRE(r.len == 3 * SR + ctx); // full right ctx
}

TEST_CASE("ctx: last slice gets no right context", "[unit][chunk-context]") {
    const int SR = 16000;
    const int ctx = 2 * SR;
    const int total = 10 * SR;
    // Slice 2: [7*SR, 10*SR)
    int sl_start = 7 * SR, sl_end = total;
    int64_t t0_cs = (int64_t)((double)sl_start / SR * 100.0);
    auto r = compute_ctx(sl_start, sl_end, t0_cs, /*has_prev=*/true, /*has_next=*/false, total, ctx, SR);
    REQUIRE(r.start == sl_start - ctx); // left ctx added
    REQUIRE(r.len == sl_end - r.start); // no right ctx
    REQUIRE(r.t0_cs == t0_cs - 200);    // 2 s = 200 cs
}

TEST_CASE("ctx: middle slice gets both contexts", "[unit][chunk-context]") {
    const int SR = 16000;
    const int ctx = 2 * SR;
    const int total = 10 * SR;
    int sl_start = 3 * SR, sl_end = 6 * SR;
    int64_t t0_cs = 300; // 3 s = 300 cs
    auto r = compute_ctx(sl_start, sl_end, t0_cs, /*has_prev=*/true, /*has_next=*/true, total, ctx, SR);
    REQUIRE(r.start == sl_start - ctx);
    REQUIRE(r.len == sl_end - r.start + ctx);
    REQUIRE(r.t0_cs == t0_cs - 200);
}

TEST_CASE("ctx: single slice gets no context at all", "[unit][chunk-context]") {
    const int SR = 16000;
    const int ctx = 2 * SR;
    const int total = 5 * SR;
    int sl_start = 0, sl_end = total;
    auto r = compute_ctx(sl_start, sl_end, 0, /*has_prev=*/false, /*has_next=*/false, total, ctx, SR);
    REQUIRE(r.start == 0);
    REQUIRE(r.len == total);
    REQUIRE(r.t0_cs == 0);
}

TEST_CASE("ctx: left context clamped when slice is near start", "[unit][chunk-context]") {
    const int SR = 16000;
    const int ctx = 2 * SR; // wants 2 s
    const int total = 10 * SR;
    // Slice starts at only 0.5 s → can only pull 0.5 s of left ctx.
    int sl_start = SR / 2;
    int sl_end = 3 * SR;
    int64_t t0_cs = 50; // 0.5 s = 50 cs
    auto r = compute_ctx(sl_start, sl_end, t0_cs, /*has_prev=*/true, /*has_next=*/true, total, ctx, SR);
    REQUIRE(r.start == 0);          // clamped to 0
    REQUIRE(r.t0_cs == 0);          // clamped to 0 cs
    REQUIRE(r.len == sl_end + ctx); // right ctx unclamped
}

TEST_CASE("ctx: right context clamped when slice is near end", "[unit][chunk-context]") {
    const int SR = 16000;
    const int ctx = 2 * SR;
    const int total = 10 * SR;
    // Slice ends 0.5 s before total → only 0.5 s of right ctx available.
    int sl_start = 7 * SR;
    int sl_end = total - SR / 2;
    int64_t t0_cs = 700;
    auto r = compute_ctx(sl_start, sl_end, t0_cs, /*has_prev=*/true, /*has_next=*/true, total, ctx, SR);
    REQUIRE(r.start + r.len == total);       // expanded end clamped to total
    REQUIRE(r.len < sl_end - r.start + ctx); // didn't get the full 2 s right ctx
}

// ---------------------------------------------------------------------------
// 3. Segment filtering
// ---------------------------------------------------------------------------

TEST_CASE("filter: segments inside window are kept", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs = {
        make_seg(100, 120),
        make_seg(200, 250),
        make_seg(290, 300),
    };
    trim_to_window(segs, /*t0_cs=*/100, /*t1_cs=*/300);
    REQUIRE(segs.size() == 3);
}

TEST_CASE("filter: segment with t0 < t0_cs is removed (left-ctx region)", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs = {
        make_seg(50, 80),   // left-ctx region → remove
        make_seg(100, 150), // inside → keep
    };
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].t0 == 100);
}

TEST_CASE("filter: segment with t0 >= t1_cs is removed (right-ctx region)", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs = {
        make_seg(200, 250), // inside → keep
        make_seg(300, 350), // right-ctx start → remove (t0 == t1_cs)
        make_seg(320, 360), // well past end → remove
    };
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].t0 == 200);
}

TEST_CASE("filter: segment starting exactly at t0_cs is kept", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs = {make_seg(100, 150)};
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
}

TEST_CASE("filter: segment starting at t1_cs-1 is kept", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs = {make_seg(299, 310)};
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].t0 == 299);
}

TEST_CASE("filter: all segments outside window → empty result", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs = {
        make_seg(0, 50),
        make_seg(50, 99),
        make_seg(300, 400),
    };
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.empty());
}

TEST_CASE("filter: empty input → empty output", "[unit][chunk-context]") {
    std::vector<crispasr_segment> segs;
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.empty());
}

TEST_CASE("filter: segments with t1 past window end but t0 inside are kept", "[unit][chunk-context]") {
    // A word that STARTS in this chunk but ENDS in the next chunk's territory.
    // We keep it — the next chunk's left-ctx filter will exclude duplicates.
    std::vector<crispasr_segment> segs = {make_seg(290, 320)}; // t0=290 < t1_cs=300
    trim_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].t1 == 320); // t1 preserved unchanged
}

// ---------------------------------------------------------------------------
// 4. No-duplication contract
// ---------------------------------------------------------------------------

TEST_CASE("filter: adjacent slices never produce the same t0 in both outputs", "[unit][chunk-context]") {
    // Simulate two adjacent slices each emitting a segment at t0 = 300 cs
    // (the boundary between them, where slice A ends and slice B starts).
    // Slice A window [0, 300), slice B window [300, 600).
    // Both decoders emit a segment at t0=300.

    std::vector<crispasr_segment> segs_a = {make_seg(300, 340)};
    std::vector<crispasr_segment> segs_b = {make_seg(300, 340)};

    trim_to_window(segs_a, 0, 300);   // [0, 300) → t0=300 is removed from A
    trim_to_window(segs_b, 300, 600); // [300, 600) → t0=300 stays in B

    REQUIRE(segs_a.empty());     // not in A
    REQUIRE(segs_b.size() == 1); // only in B
}

// ---------------------------------------------------------------------------
// 5. Word-level filter (primary path — word timestamps available)
// ---------------------------------------------------------------------------

TEST_CASE("word-filter: words starting in window are kept", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(100, 500, "");
    seg.words = {make_word(100, 130, "And"), make_word(150, 200, "so,"), make_word(250, 310, "my"),
                 make_word(310, 390, "fellow")};
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 400);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].words.size() == 4);
}

TEST_CASE("word-filter: words with t0 < t0_cs removed (left-ctx region)", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(50, 400, "");
    seg.words = {
        make_word(50, 80, "left-ctx"),      // before window
        make_word(100, 150, "inside"),      // in window
        make_word(200, 250, "also-inside"), // in window
    };
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].words.size() == 2);
    REQUIRE(segs[0].t0 == 100);
}

TEST_CASE("word-filter: words with t0 >= t1_cs removed (right-ctx region)", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(100, 500, "");
    seg.words = {
        make_word(100, 180, "keep"), make_word(300, 360, "boundary"), // t0 == t1_cs → removed
        make_word(350, 400, "after"),                                 // t0 > t1_cs → removed
    };
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 300);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].words.size() == 1);
    REQUIRE(segs[0].t0 == 100);
}

TEST_CASE("word-filter: boundary word spanning t1_cs kept in starting slice", "[unit][chunk-context]") {
    // 'Americans,' starts at 290 (< 300 = t1_cs) but ends at 340 (> 300).
    // It belongs to the CURRENT slice (assigned by start), not the next.
    crispasr_segment seg = make_seg(200, 400, "");
    seg.words = {make_word(200, 250, "my"), make_word(290, 340, "Americans,")};
    std::vector<crispasr_segment> segs_a = {seg};
    trim_words_to_window(segs_a, 0, 300);
    REQUIRE(segs_a.size() == 1);
    REQUIRE(segs_a[0].words.size() == 2); // both kept; next slice will not re-include
}

TEST_CASE("word-filter: all words outside window → segment dropped", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(0, 200, "");
    seg.words = {make_word(0, 50, "before"), make_word(50, 99, "still-before")};
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 300);
    REQUIRE(segs.empty());
}

TEST_CASE("word-filter: text rebuilt correctly with plain tokens (no leading space)", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(100, 400, "");
    seg.words = {make_word(100, 150, "ask"), make_word(200, 240, "not"), make_word(300, 360, "what")};
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 400);
    REQUIRE(segs[0].text == "ask not what");
}

TEST_CASE("word-filter: text rebuilt correctly with space-prefix tokens", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(100, 400, "");
    seg.words = {make_word(100, 150, " ask"), make_word(200, 240, " not"), make_word(300, 360, " what")};
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 400);
    REQUIRE(segs[0].text == " ask not what"); // no double-space inserted
}

TEST_CASE("word-filter: adjacent slices assign boundary word to correct slice", "[unit][chunk-context]") {
    // 'Americans,' starts at 176, t1_cs for slice A = 300.
    // 'ask' starts at 328, t0_cs for slice B = 300.
    // After filtering each slice, no word appears in both.
    crispasr_segment base = make_seg(0, 600, "");
    base.words = {make_word(176, 328, "Americans,"), make_word(328, 392, "ask")};

    auto segs_a = std::vector<crispasr_segment>{base};
    auto segs_b = std::vector<crispasr_segment>{base};
    trim_words_to_window(segs_a, 0, 300);   // slice A: [0, 300)
    trim_words_to_window(segs_b, 300, 600); // slice B: [300, 600)

    REQUIRE(segs_a.size() == 1);
    REQUIRE(segs_a[0].words.size() == 1);
    REQUIRE(segs_a[0].words[0].text == "Americans,");

    REQUIRE(segs_b.size() == 1);
    REQUIRE(segs_b[0].words.size() == 1);
    REQUIRE(segs_b[0].words[0].text == "ask");
}

TEST_CASE("word-filter: t0/t1 of segment updated from surviving words", "[unit][chunk-context]") {
    crispasr_segment seg = make_seg(50, 600, "");
    seg.words = {make_word(50, 80, "left"), make_word(200, 280, "middle"), make_word(400, 480, "right")};
    std::vector<crispasr_segment> segs = {seg};
    trim_words_to_window(segs, 100, 350);
    REQUIRE(segs.size() == 1);
    REQUIRE(segs[0].t0 == 200); // updated from first surviving word
    REQUIRE(segs[0].t1 == 280); // updated from last surviving word
}
