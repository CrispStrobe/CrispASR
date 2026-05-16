// test-stream-vad-merge-policy.cpp -- pure unit tests for the JSON streaming
// VAD post-merge policy. No VAD model or audio I/O is required.

#include <catch2/catch_test_macros.hpp>

#include "../src/crispasr_vad.h"

#include <vector>

static crispasr_audio_slice slice(int start, int end) {
    return crispasr_audio_slice{start, end, start / 160, end / 160};
}

static crispasr_vad_options stream_opts(int merge_gap_ms, int final_silence_ms = 800) {
    crispasr_vad_options opts;
    opts.post_merge_policy = crispasr_vad_post_merge_policy::streaming_json;
    opts.stream_close_gap_ms = merge_gap_ms;
    opts.stream_final_silence_ms = final_silence_ms;
    return opts;
}

TEST_CASE("stream-vad-merge-policy: effective gap clamps below final silence", "[unit][stream-json]") {
    REQUIRE(crispasr_stream_vad_effective_merge_gap_ms(250, 800) == 250);
    REQUIRE(crispasr_stream_vad_effective_merge_gap_ms(500, 300) == 299);
    REQUIRE(crispasr_stream_vad_effective_merge_gap_ms(250, 1) == 0);
    REQUIRE(crispasr_stream_vad_effective_merge_gap_ms(250, 0) == 250);
    REQUIRE(crispasr_stream_vad_effective_merge_gap_ms(0, 800) == 0);
}

TEST_CASE("stream-vad-merge-policy: close jitter gap merges below threshold", "[unit][stream-json]") {
    const std::vector<crispasr_audio_slice> in = {
        slice(0, 16000),
        slice(16000 + 3999, 32000),
    };
    const auto out = crispasr_post_merge_vad_slices(in, 16000, stream_opts(250));

    REQUIRE(out.size() == 1);
    REQUIRE(out[0].start == 0);
    REQUIRE(out[0].end == 32000);
}

TEST_CASE("stream-vad-merge-policy: gap equal to threshold does not merge", "[unit][stream-json]") {
    const std::vector<crispasr_audio_slice> in = {
        slice(0, 16000),
        slice(16000 + 4000, 32000),
    };
    const auto out = crispasr_post_merge_vad_slices(in, 16000, stream_opts(250));

    REQUIRE(out.size() == 2);
    REQUIRE(out[0].end == 16000);
    REQUIRE(out[1].start == 20000);
}

TEST_CASE("stream-vad-merge-policy: hard final silence boundary is not hidden", "[unit][stream-json]") {
    const std::vector<crispasr_audio_slice> in = {
        slice(0, 16000),
        slice(16000 + 4800, 32000),
    };
    const auto out = crispasr_post_merge_vad_slices(in, 16000, stream_opts(500, 300));

    REQUIRE(crispasr_stream_vad_effective_merge_gap_ms(500, 300) == 299);
    REQUIRE(out.size() == 2);
}

TEST_CASE("stream-vad-merge-policy: no duration-based short-slice merge", "[unit][stream-json]") {
    const std::vector<crispasr_audio_slice> in = {
        slice(0, 8000),
        slice(8000 + 20000, 44000),
    };
    const auto out = crispasr_post_merge_vad_slices(in, 16000, stream_opts(250));

    REQUIRE(out.size() == 2);
}

TEST_CASE("stream-vad-merge-policy: offline policy still merges short previous slice", "[unit][stream-json]") {
    const std::vector<crispasr_audio_slice> in = {
        slice(0, 8000),
        slice(8000 + 20000, 44000),
    };
    crispasr_vad_options opts;
    const auto out = crispasr_post_merge_vad_slices(in, 16000, opts);

    REQUIRE(out.size() == 1);
    REQUIRE(out[0].start == 0);
    REQUIRE(out[0].end == 44000);
}

TEST_CASE("stream-vad-merge-policy: final silence disabled uses configured gap", "[unit][stream-json]") {
    const std::vector<crispasr_audio_slice> in = {
        slice(0, 16000),
        slice(16000 + 7999, 40000),
    };
    const auto out = crispasr_post_merge_vad_slices(in, 16000, stream_opts(500, 0));

    REQUIRE(out.size() == 1);
}
