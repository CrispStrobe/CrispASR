// test-core-tts-voice-policy.cpp — per-request voice decisions (#371).
//
// The bug: the HTTP server reuses one backend for the session and passes each
// request's `voice=` in params.tts_voice, but omnivoice and f5-tts only read it
// at init(), so /v1/audio/speech always served the startup voice — or the
// built-in default when no --voice was given — no matter what was uploaded.
#include "core/tts_voice_policy.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

using core_tts_voice::Action;
using core_tts_voice::decide;
using core_tts_voice::is_builtin_request;

TEST_CASE("tts voice: a new reference is applied", "[unit][tts-voice][issue-371]") {
    // The reported case: session started with one voice (or none), request asks
    // for another.
    REQUIRE(decide("", "alice.wav") == Action::Apply);
    REQUIRE(decide("alice.wav", "bob.wav") == Action::Apply);
}

TEST_CASE("tts voice: an identical repeat is not re-encoded", "[unit][tts-voice][issue-371]") {
    // Why the dedupe exists: applying a reference loads a WAV, resamples, and
    // may run a whole ASR model to auto-transcribe it. Doing that per request
    // for an unchanged voice is the cost the last_voice_ check avoids.
    REQUIRE(decide("alice.wav", "alice.wav") == Action::Unchanged);
    REQUIRE(decide("", "") == Action::Unchanged);
    REQUIRE(decide("default", "default") == Action::Unchanged);
}

TEST_CASE("tts voice: sentinels select the built-in reference", "[unit][tts-voice][issue-371]") {
    REQUIRE(is_builtin_request(""));
    REQUIRE(is_builtin_request("default"));
    REQUIRE(is_builtin_request("auto"));

    // Switching AWAY from a real voice back to the built-in must act, not dedupe.
    REQUIRE(decide("alice.wav", "") == Action::Builtin);
    REQUIRE(decide("alice.wav", "default") == Action::Builtin);
    REQUIRE(decide("alice.wav", "auto") == Action::Builtin);
}

TEST_CASE("tts voice: sentinels are exact, not normalised", "[unit][tts-voice][issue-371]") {
    // A file really can be called Default.wav, or live in a directory named
    // "auto". Case-folding or substring-matching the sentinel would make those
    // unreachable, so the match is exact and case-sensitive — matching what
    // --voice already did.
    REQUIRE_FALSE(is_builtin_request("Default"));
    REQUIRE_FALSE(is_builtin_request("AUTO"));
    REQUIRE_FALSE(is_builtin_request("default.wav"));
    REQUIRE_FALSE(is_builtin_request("voices/auto/ref.wav"));
    REQUIRE(decide("", "Default") == Action::Apply);
}

TEST_CASE("tts voice: a failed load must not dedupe the retry away", "[unit][tts-voice][issue-371]") {
    // The subtle one. If a load fails and the caller leaves `last` set to the
    // voice it FAILED to apply, the next identical request returns Unchanged and
    // is served with whatever reference happens to be resident — silently the
    // wrong voice, with no second attempt. Resetting `last` to "" on failure is
    // what makes the retry reachable.
    REQUIRE(decide("broken.wav", "broken.wav") == Action::Unchanged); // the trap
    REQUIRE(decide("", "broken.wav") == Action::Apply);               // after reset
}
