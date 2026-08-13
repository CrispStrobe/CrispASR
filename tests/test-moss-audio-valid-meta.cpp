// Hermetic CPU test for the issue #344 valid-frame metadata.
//
// moss_audio_plan_chunks() is pure arithmetic (no model state): per-chunk
// valid-frame counts for the MOSS encoder's 400-frame chunk loop, where each
// chunk's valid token count is 3× stride-2 conv downsampling of its real
// length. These tests pin the invariants the downstream MOSS-Music feature
// pipeline relies on:
//
//   1. sum(per-chunk valid) == out_total_valid (metadata is internally
//      consistent with what the encoder loop actually computes).
//   2. num_chunks == ceil(T_mel / 400) and per-chunk values == conv_len³
//      of the chunk's real length.
//   3. Sub-30s pad exclusion: a content length under 3000 yields strictly
//      fewer valid frames than the same mel run through the 3000-frame
//      Whisper pad — the pad never fabricates frames that a consumer would
//      mistake for content. The valid count is derived from the real chunk
//      lengths, never from a padded-zeros heuristic or a global floor.

#include <catch2/catch_test_macros.hpp>

#include "moss_audio.h"

#include <numeric>

namespace {

// 3× stride-2 conv downsampling (conv_len(L) = (L-1)/2 + 1), replicated here
// ONLY to independently predict plan_chunks' output — the function under test
// is the production source of truth, this is the oracle.
int conv_len(int L) {
    return (L - 1) / 2 + 1;
}
int conv_len3(int L) {
    return conv_len(conv_len(conv_len(L)));
}

} // namespace

TEST_CASE("plan_chunks: sum(valid) == total and per-chunk math", "[moss-audio-valid-meta]") {
    const int kChunk = 400;
    const int cases[] = {1, 2, 50, 399, 400, 401, 799, 800, 801, 1199, 1200, 3000, 3001, 3999, 4000, 4001, 12345};

    for (int T_mel : cases) {
        const int expect_chunks = (T_mel + kChunk - 1) / kChunk;
        std::vector<int> counts(expect_chunks, -1);
        int total = -1;
        CAPTURE(T_mel);
        REQUIRE(moss_audio_plan_chunks(T_mel, counts.data(), &total) == expect_chunks);

        int sum = 0;
        for (int c = 0; c < expect_chunks; c++) {
            const int chunk_len = std::min(kChunk, T_mel - c * kChunk);
            INFO("chunk " << c << " len " << chunk_len);
            REQUIRE(counts[c] == conv_len3(chunk_len));
            sum += counts[c];
        }
        REQUIRE(sum == total);
    }
}

TEST_CASE("plan_chunks: invalid input fails closed, null counts allowed", "[moss-audio-valid-meta]") {
    int total = 999;
    int counts[4] = {-1, -1, -1, -1};

    REQUIRE(moss_audio_plan_chunks(0, counts, &total) == 0);
    REQUIRE(total == 0);
    REQUIRE(moss_audio_plan_chunks(-5, nullptr, &total) == 0);
    REQUIRE(total == 0);

    // null valid_counts is permitted: only num_chunks / total matter.
    REQUIRE(moss_audio_plan_chunks(800, nullptr, &total) == 2);
    REQUIRE(total == 100);
}

TEST_CASE("sub-30s pad exclusion: plan_chunks over T_mel_actual, never the pad", "[moss-audio-valid-meta]") {
    // The encoder mel buffer is padded to 3000 frames. Planning over the
    // PADDED length reports valid frames that include zero-content pad chunks.
    std::vector<int> padded_counts(8, -1);
    int padded_total = -1;
    REQUIRE(moss_audio_plan_chunks(3000, padded_counts.data(), &padded_total) == 8);
    REQUIRE(padded_total == 375);

    // conv_len³ collapses like ceil(L/8), so near 30s a 1-7 frame pad tail can
    // land on the same token count as the real tail (e.g. 2999 -> 375, equal
    // to the padded path). The universal invariant is therefore "never more
    // than the padded path"; a strictly-lower count is guaranteed whenever a
    // whole 400-frame pad chunk exists (T_actual <= 2800).
    for (int T_actual : {1, 2, 100, 399, 400, 401, 799, 800, 801, 1001, 1500, 2400, 2800, 2999}) {
        const int n = (T_actual + 399) / 400;
        std::vector<int> counts(n, -1);
        int total = -1;
        INFO("T_actual=" << T_actual);
        REQUIRE(moss_audio_plan_chunks(T_actual, counts.data(), &total) == n);
        REQUIRE(total > 0);
        REQUIRE(total <= padded_total); // pad must never fabricate a frame the meta path counts
        if (T_actual <= 2800)
            REQUIRE(total < padded_total); // whole pad chunks exist and are excluded
        if (T_actual >= 400)
            REQUIRE(counts[0] == 50); // every full chunk is 50 valid tokens
    }

    // >= 30 s (no pad) must NOT fabricate pad-only frames.
    int full = -1;
    REQUIRE(moss_audio_plan_chunks(3000, nullptr, &full) == 8);
    REQUIRE(full == 375);
    REQUIRE(full == padded_total);
}

TEST_CASE("encoder wrappers: ds_tap out-pointers untouched on failure (issue #344 B1)", "[moss-audio-valid-meta]") {
    // B1 regression guard. The run_encoder/run_encoder_meta wrappers must NEVER
    // write *ds_tap_x when the encoder impl fails: pre-fix, both wrappers
    // published the impl's (possibly freed) tap slots on EVERY failure path,
    // leaving callers with dangling non-NULL pointers on a NULL return.
    //
    // The deep failure paths (graph alloc / graph compute / missing
    // encoder_output) require a loaded model and a ggml backend and cannot be
    // forced hermetically. This pins the wrapper invariant at the observable
    // boundary instead: any NULL-returning call must leave caller-provided tap
    // out-pointers byte-for-byte untouched. Pre-fix this test fails (the
    // sentinels are overwritten with NULL); post-fix it passes.
    float dummy0 = 0.0f, dummy1 = 0.0f, dummy2 = 0.0f;
    int T_enc = -1, d = -1;

    // run_encoder: NULL ctx => impl early-validation failure => NULL return.
    float* ds0 = &dummy0;
    float* ds1 = &dummy1;
    float* ds2 = &dummy2;
    float* r = moss_audio_run_encoder(nullptr, nullptr, 0, 0, &T_enc, &d, &ds0, &ds1, &ds2);
    REQUIRE(r == nullptr);
    REQUIRE(ds0 == &dummy0);
    REQUIRE(ds1 == &dummy1);
    REQUIRE(ds2 == &dummy2);

    // Same, with tap requests nominally enabled and garbage mel args.
    ds0 = &dummy0;
    ds1 = &dummy1;
    ds2 = &dummy2;
    T_enc = -1;
    d = -1;
    float sample = 0.0f;
    r = moss_audio_run_encoder(nullptr, &sample, 1, 400, &T_enc, &d, &ds0, &ds1, &ds2);
    REQUIRE(r == nullptr);
    REQUIRE(ds0 == &dummy0);
    REQUIRE(ds1 == &dummy1);
    REQUIRE(ds2 == &dummy2);

    // run_encoder_meta: fail-closed NULL return => taps untouched.
    ds0 = &dummy0;
    ds1 = &dummy1;
    ds2 = &dummy2;
    T_enc = -1;
    d = -1;
    int nc = -1, tot = -1, echo = -1;
    r = moss_audio_run_encoder_meta(nullptr, &sample, 1, 3000, 100, &T_enc, &d, nullptr, &nc, &echo, &tot, &ds0, &ds1,
                                    &ds2);
    REQUIRE(r == nullptr);
    REQUIRE(ds0 == &dummy0);
    REQUIRE(ds1 == &dummy1);
    REQUIRE(ds2 == &dummy2);
}
