// Unit tests for the streaming partial-decode cadence predicate added
// in PR #113 (`--stream-partial-decode-ms`).
//
// The two helpers in `examples/cli/crispasr_stream_partial_decode.h`
// are pure arithmetic: no model load, no audio, no streaming runtime.
// Cover the interval-conversion math, the five branches of the gate
// predicate, and a "simulate a 500 ms step + 1000 ms throttle" scenario
// that mirrors the smoke configuration the PR shipped with.

#include "../examples/cli/crispasr_stream_partial_decode.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

constexpr int kSR = 16000;

TEST_CASE("partial-decode: interval = ms * sample_rate / 1000", "[unit][stream-partial-decode]") {
    REQUIRE(crispasr_stream_partial_decode_interval_samples(1000, /*step_ms=*/500, kSR) == 16000);
    REQUIRE(crispasr_stream_partial_decode_interval_samples(750, /*step_ms=*/500, kSR) == 12000);
    REQUIRE(crispasr_stream_partial_decode_interval_samples(500, /*step_ms=*/500, kSR) == 8000);
}

TEST_CASE("partial-decode: ms=0 falls back to step_ms (the documented default)", "[unit][stream-partial-decode]") {
    // The whole point of `0` as the default: the throttle is always
    // conceptually present but locked to the step cadence when the
    // user doesn't override it.
    REQUIRE(crispasr_stream_partial_decode_interval_samples(0, /*step_ms=*/500, kSR) == 8000);
    REQUIRE(crispasr_stream_partial_decode_interval_samples(0, /*step_ms=*/3000, kSR) == 48000);
    REQUIRE(crispasr_stream_partial_decode_interval_samples(0, /*step_ms=*/1, kSR) == 16);
}

TEST_CASE("partial-decode: degenerate intervals return 0 safely", "[unit][stream-partial-decode]") {
    // Both ms and step_ms are 0 -> the gate's interval_samples<=0
    // branch fires, which always allows decode. Document the math.
    REQUIRE(crispasr_stream_partial_decode_interval_samples(0, /*step_ms=*/0, kSR) == 0);
    REQUIRE(crispasr_stream_partial_decode_interval_samples(-100, /*step_ms=*/-1, kSR) == 0);
    REQUIRE(crispasr_stream_partial_decode_interval_samples(500, /*step_ms=*/500, /*sample_rate=*/0) == 0);
}

TEST_CASE("partial-decode: non-JSON streaming never throttles", "[unit][stream-partial-decode]") {
    // Even with a tiny interval and a barely-advanced sample count,
    // the non-JSON path must always decode.
    REQUIRE(crispasr_stream_partial_decode_allow(/*stream_json=*/false,
                                                 /*last_partial_decode_sample=*/16000,
                                                 /*final_silence_due=*/false,
                                                 /*cumulative_samples=*/16001,
                                                 /*interval_samples=*/16000));
}

TEST_CASE("partial-decode: first step always decodes (last_partial_decode_sample == -1)",
          "[unit][stream-partial-decode]") {
    REQUIRE(crispasr_stream_partial_decode_allow(true, -1, false, 0, 16000));
    REQUIRE(crispasr_stream_partial_decode_allow(true, -1, false, 0, 0)); // even with degenerate interval
}

TEST_CASE("partial-decode: final-silence-due bypasses the throttle", "[unit][stream-partial-decode]") {
    // We're well below the interval but the caller signals that the
    // next step will finalize the utterance — the bypass fires.
    REQUIRE(crispasr_stream_partial_decode_allow(/*stream_json=*/true,
                                                 /*last_partial_decode_sample=*/100000,
                                                 /*final_silence_due=*/true,
                                                 /*cumulative_samples=*/100001,
                                                 /*interval_samples=*/16000));
}

TEST_CASE("partial-decode: interval <= 0 is treated as 'always allow'", "[unit][stream-partial-decode]") {
    // Defensive branch — happens only with both ms=0 AND step_ms=0
    // upstream, which the CLI parser shouldn't allow but the gate
    // tolerates anyway.
    REQUIRE(crispasr_stream_partial_decode_allow(true, 100000, false, 100001, 0));
    REQUIRE(crispasr_stream_partial_decode_allow(true, 100000, false, 100001, -1));
}

TEST_CASE("partial-decode: gate trips only after `interval_samples` of new audio", "[unit][stream-partial-decode]") {
    constexpr int64_t interval = 16000; // 1 s at 16 kHz
    constexpr int64_t last = 50000;

    // delta < interval: throttled.
    REQUIRE_FALSE(crispasr_stream_partial_decode_allow(true, last, false, last + 1, interval));
    REQUIRE_FALSE(crispasr_stream_partial_decode_allow(true, last, false, last + interval - 1, interval));

    // delta == interval: allowed.
    REQUIRE(crispasr_stream_partial_decode_allow(true, last, false, last + interval, interval));

    // delta > interval: allowed.
    REQUIRE(crispasr_stream_partial_decode_allow(true, last, false, last + interval + 1, interval));
}

// Scenario test pinning the smoke configuration from the PR body:
// --stream-step 500 + --stream-partial-decode-ms 1000 should let only
// every OTHER step decode, with the first step always allowed.
TEST_CASE("partial-decode: 500 ms step + 1000 ms throttle -> decode every other step",
          "[unit][stream-partial-decode]") {
    const int64_t interval = crispasr_stream_partial_decode_interval_samples(/*ms=*/1000, /*step_ms=*/500, kSR);
    REQUIRE(interval == 16000);

    // Each step delivers ~8000 samples (500 ms at 16 kHz). Simulate
    // 8 streaming steps; assume `final_silence_due` is false throughout
    // (no utterance has finalized yet).
    int64_t last_decode = -1;
    int64_t cumulative = 0;
    int decoded = 0, throttled = 0;
    for (int step = 0; step < 8; step++) {
        cumulative += 8000;
        const bool allow = crispasr_stream_partial_decode_allow(
            /*stream_json=*/true, last_decode, /*final_silence_due=*/false, cumulative, interval);
        if (allow) {
            decoded++;
            last_decode = cumulative;
        } else {
            throttled++;
        }
    }

    // First step always decodes; then every other step thereafter
    // (steps 1, 3, 5, 7 decode; steps 2, 4, 6 throttle). 4 decoded
    // / 4 throttled is the expected ratio from the PR's smoke
    // (205 -> 170 partials over 150 s is roughly -17%, consistent
    // with this cadence over a much longer stream).
    REQUIRE(decoded == 4);
    REQUIRE(throttled == 4);
}

// Pins the regression risk: someone changes the default behaviour
// from "0 means follow step_ms" to "0 means disable throttling".
TEST_CASE("partial-decode: default 0 + step 500 still throttles every step (no-op default)",
          "[unit][stream-partial-decode]") {
    const int64_t interval = crispasr_stream_partial_decode_interval_samples(/*ms=*/0, /*step_ms=*/500, kSR);
    REQUIRE(interval == 8000);

    // 6 consecutive 500 ms steps should all decode — the gate trips
    // exactly when the step cadence does.
    int64_t last_decode = -1;
    int64_t cumulative = 0;
    int decoded = 0;
    for (int step = 0; step < 6; step++) {
        cumulative += 8000;
        if (crispasr_stream_partial_decode_allow(true, last_decode, false, cumulative, interval)) {
            decoded++;
            last_decode = cumulative;
        }
    }
    REQUIRE(decoded == 6);
}
