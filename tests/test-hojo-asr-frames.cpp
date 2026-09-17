// test-hojo-asr-frames.cpp — hermetic guards for the Hojo-ASR frame schedule.
//
// crispasr-diff compares VALUES; it cannot see that a stage computed the right
// values over the wrong frames, nor that a tile schedule silently dropped the
// left context of every other block. Both produce fluent, confident, wrong
// transcripts. These tests pin the arithmetic instead, and each one was written
// to FAIL against a deliberately broken variant before the real code existed:
//
//  * `feat_output_len` at an exact multiple of n_window_infer. Python's `//`
//    floors, C's `/` truncates, and the two disagree exactly once — at
//    (leave - 1) // 2 with leave == 0. A naive transcription is one frame too
//    long for every 30 s / 60 s / 90 s input and correct everywhere else.
//  * the conv tile halo. The whole point of tiling is that a kept output never
//    reads a padded edge; that is an algebraic containment, so it is asserted
//    as one (HARD RULE 2c) rather than as a tolerance on a signal.

#include <catch2/catch_test_macros.hpp>

#include "core/hojo_asr_frames.h"

#include <cstdint>
#include <vector>

using namespace core_hojo_frames;

namespace {

// The reference, transcribed with Python's floor division so the C/Python
// disagreement cannot hide inside the thing we are checking against.
int py_floordiv(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        q--;
    return q;
}

int reference_feat_output_len(int L, int n_window) {
    const int leave = L % n_window;
    const int feat = py_floordiv(leave - 1, 2) + 1;
    const int per_window = (n_window + 7) / 8;
    return py_floordiv(py_floordiv(feat - 1, 2) + 1 - 1, 2) + 1 + py_floordiv(L, n_window) * per_window;
}

} // namespace

TEST_CASE("hojo frame count matches the reference at every length", "[hojo]") {
    const int nwi = 3000;
    for (int L = 1; L <= 9100; L++) {
        REQUIRE(feat_output_len(L, nwi) == reference_feat_output_len(L, nwi));
    }
}

TEST_CASE("hojo frame count is right at the chunk boundary", "[hojo]") {
    // 3000 mel frames = 30 s = one full window = 3000/8 encoder frames.
    REQUIRE(feat_output_len(3000, 3000) == 375);
    REQUIRE(feat_output_len(6000, 3000) == 750);
    // One frame either side of the boundary must differ by no more than one
    // frame — the exact-multiple case is where truncating division adds a
    // phantom frame.
    REQUIRE(feat_output_len(2999, 3000) == 375);
    REQUIRE(feat_output_len(3001, 3000) == 376);
    // 10 s of audio.
    REQUIRE(feat_output_len(1000, 3000) == 125);
    // The 8x downsample holds for short inputs too.
    REQUIRE(feat_output_len(8, 3000) == 1);
    REQUIRE(feat_output_len(16, 3000) == 2);
}

TEST_CASE("hojo positive control: a truncating-division port fails the boundary", "[hojo]") {
    // The bug this suite exists to catch, spelled out. If this ever stops
    // differing from the real implementation, the test above has gone blind.
    auto truncating = [](int L, int n_window) {
        const int leave = L % n_window;
        const int feat = (leave - 1) / 2 + 1;
        const int per_window = (n_window + 7) / 8;
        return ((feat - 1) / 2 + 1 - 1) / 2 + 1 + (L / n_window) * per_window;
    };
    REQUIRE(truncating(3000, 3000) != feat_output_len(3000, 3000));
    REQUIRE(truncating(1000, 3000) == feat_output_len(1000, 3000)); // agrees off the boundary
}

TEST_CASE("hojo chunk plan tiles the mel exactly once", "[hojo]") {
    const int chunk_T = 3000, nwi = 3000;
    for (int T_mel : {1, 99, 1000, 2999, 3000, 3001, 6000, 7321}) {
        const ChunkPlan p = plan_chunks(T_mel, chunk_T, nwi);
        REQUIRE(!p.chunks.empty());
        int covered = 0;
        int expect_t0 = 0;
        for (const auto& ch : p.chunks) {
            REQUIRE(ch.t0 == expect_t0); // contiguous, no gaps, no overlap
            REQUIRE(ch.len > 0);
            REQUIRE(ch.len <= chunk_T);
            REQUIRE(ch.valid > 0);
            covered += ch.len;
            expect_t0 += ch.len;
        }
        REQUIRE(covered == T_mel);           // every mel frame used once
        REQUIRE(p.win_T == p.chunks[0].len); // pad_sequence pads to the longest
        REQUIRE(p.T_enc == [&] {
            int n = 0;
            for (const auto& ch : p.chunks)
                n += ch.valid;
            return n;
        }());
    }
}

TEST_CASE("hojo single short utterance is convolved at its own width", "[hojo]") {
    // pad_sequence over a one-element batch pads to that element, so a 10 s clip
    // must NOT be widened to 3000 frames — doing so is harmless numerically but
    // 3x the conv work and 3x the transient memory.
    const ChunkPlan p = plan_chunks(1000, 3000, 3000);
    REQUIRE(p.chunks.size() == 1u);
    REQUIRE(p.win_T == 1000);
    REQUIRE(p.T_enc == 125);
}

TEST_CASE("hojo conv tiles cover every input a kept output reads", "[hojo]") {
    // Output frame o reads input frames [8o-7, 8o+7]. A tile's window must
    // contain that whole span for every output it keeps, at a LOCAL index that
    // is never 0 (index 0 is the frame contaminated by the conv's zero pad).
    for (int tile : {1, 3, 8, 64, 375}) {
        for (int n_out : {1, 2, 7, 64, 65, 125, 375}) {
            std::vector<bool> produced((size_t)n_out, false);
            for (int o0 = 0; o0 < n_out; o0 += tile) {
                const int o1 = (o0 + tile < n_out) ? (o0 + tile) : n_out;
                const TileWindow w = tile_window(o0, o1);

                REQUIRE(w.keep_from >= 1);
                REQUIRE(w.keep_count == o1 - o0);
                REQUIRE(w.out_offset == o0);
                // The window must actually produce keep_from + keep_count frames.
                REQUIRE(conv_stem_out_len(w.width) >= w.keep_from + w.keep_count);

                for (int o = o0; o < o1; o++) {
                    const int local = w.keep_from + (o - o0);
                    // local index -> the same global output frame
                    REQUIRE(w.mel_offset + 8 * local == 8 * o);
                    // containment: [8o-7, 8o+7] inside [mel_offset, mel_offset+width)
                    REQUIRE(8 * o - 7 >= w.mel_offset);
                    REQUIRE(8 * o + 7 < w.mel_offset + w.width);
                    REQUIRE(!produced[(size_t)o]);
                    produced[(size_t)o] = true;
                }
            }
            for (int o = 0; o < n_out; o++)
                REQUIRE(produced[(size_t)o]);
        }
    }
}

TEST_CASE("hojo repetition penalty matches transformers", "[hojo]") {
    const int V = 8;
    std::vector<float> logits = {2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 3.0f, -3.0f, 1.0f};
    const std::vector<int32_t> gen = {0, 1, 4};
    std::vector<float> before = logits;

    apply_repetition_penalty(logits.data(), V, gen.data(), (int)gen.size(), 2.0f);

    REQUIRE(logits[0] == 1.0f);  // positive -> divided
    REQUIRE(logits[1] == -4.0f); // negative -> multiplied
    REQUIRE(logits[4] == 0.0f);  // zero counts as non-negative -> divided, unchanged
    for (int t : {2, 3, 5, 6, 7})
        REQUIRE(logits[(size_t)t] == before[(size_t)t]); // untouched tokens are untouched
}

TEST_CASE("hojo repetition penalty of 1.0 is the identity", "[hojo]") {
    const int V = 4;
    std::vector<float> logits = {1.5f, -1.5f, 0.0f, 9.0f};
    const std::vector<int32_t> gen = {0, 1, 2, 3};
    const std::vector<float> before = logits;
    apply_repetition_penalty(logits.data(), V, gen.data(), (int)gen.size(), 1.0f);
    REQUIRE(logits == before);
    // ... and the penalty from the checkpoint is NOT 1.0, so the identity above
    // is a property of the guard, not a description of the shipped decode.
    apply_repetition_penalty(logits.data(), V, gen.data(), (int)gen.size(), 2.0f);
    REQUIRE(logits != before);
}

TEST_CASE("hojo repetition penalty ignores out-of-range ids", "[hojo]") {
    const int V = 3;
    std::vector<float> logits = {1.0f, 1.0f, 1.0f};
    const std::vector<int32_t> gen = {-1, 3, 99};
    const std::vector<float> before = logits;
    apply_repetition_penalty(logits.data(), V, gen.data(), (int)gen.size(), 2.0f);
    REQUIRE(logits == before);
}

TEST_CASE("hojo max_new_tokens follows the reference cap", "[hojo]") {
    // max(10, min(cfg, T_enc * 2 + 10))
    REQUIRE(max_new_for_frames(200, 125) == 200); // 260 -> capped by cfg
    REQUIRE(max_new_for_frames(200, 10) == 30);   // 30 < 200 -> frame-bound
    REQUIRE(max_new_for_frames(200, 0) == 10);    // floor
    REQUIRE(max_new_for_frames(5, 100) == 10);    // floor beats a tiny cfg
    REQUIRE(max_new_for_frames(200, 95) == 200);
    REQUIRE(max_new_for_frames(200, 94) == 198);
}
