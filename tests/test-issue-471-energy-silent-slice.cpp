// test-issue-471-energy-silent-slice.cpp — regression guard for issue #471.
//
// Without VAD, audio longer than --chunk-seconds is split by
// crispasr_energy_chunk_slices -> audio_chunking::split_at_energy_minima, which
// cuts each window at its quietest 100 ms. When speech ends a few seconds
// before a window boundary and the audio runs past it, that quietest spot is
// the silence after the last word, and the remainder became a slice with no
// speech in it — which qwen3 answered with the --hotwords list (or an invented
// sentence). The fix drops such slices (audio_chunking::drop_speechless_ranges).
//
// These tests synthesize the issue's audio in C++ (no model, no file) and run
// exactly the pipeline crispasr_energy_chunk_slices runs (split, then drop),
// with its parameters: 30 s chunks, 5 s search window, 100 ms energy windows.
// Every "no speech-free slice" assertion is paired with a positive control
// showing the UNGATED split really produces one, so the test cannot pass by
// synthesizing audio that never triggered the bug.

#include <catch2/catch_test_macros.hpp>

#include "../src/core/audio_chunking.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr size_t SR = 16000;
constexpr size_t CHUNK = 30 * SR; // --chunk-seconds default
constexpr size_t SEARCH = 5 * SR; // crispasr_energy_chunk_slices default
constexpr size_t WIN = SR / 10;   // 100 ms energy window

using Ranges = std::vector<std::pair<size_t, size_t>>;

// Deterministic uniform noise in [-1, 1).
struct Lcg {
    uint32_t s;
    explicit Lcg(uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (float)((s >> 8) & 0xFFFFFF) / (float)0x800000 - 1.0f;
    }
};

// Speech-like signal: noise shaped by a ~4 Hz syllable envelope, with a short
// pause (~0.4 s) roughly every 3 s. `amp` is the peak amplitude.
void add_speech(std::vector<float>& a, size_t begin, size_t end, float amp, uint32_t seed) {
    Lcg rng(seed);
    const double pi = 3.14159265358979;
    for (size_t i = begin; i < end && i < a.size(); ++i) {
        const double t = (double)(i - begin) / SR;
        const double in_phrase = std::fmod(t, 3.1);
        if (in_phrase > 2.7) // inter-phrase pause
            continue;
        const double syl = 0.5 - 0.5 * std::cos(2.0 * pi * 4.3 * t); // 0..1
        const double env = 0.15 + 0.85 * syl;
        a[i] += (float)(amp * env * rng.next());
    }
}

// Low-level room noise with a slow +/-30 % level wobble (HVAC-like), so the
// floor is not perfectly stationary.
void add_room_noise(std::vector<float>& a, float rms, uint32_t seed) {
    Lcg rng(seed);
    const double pi = 3.14159265358979;
    // uniform[-1,1) has RMS 1/sqrt(3)
    const float k = rms * std::sqrt(3.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        const double wob = 1.0 + 0.3 * std::sin(2.0 * pi * 0.37 * (double)i / SR);
        a[i] += (float)(k * wob) * rng.next();
    }
}

Ranges split_raw(const std::vector<float>& a) {
    return audio_chunking::split_at_energy_minima(a.data(), a.size(), CHUNK, SEARCH, WIN);
}

// What crispasr_energy_chunk_slices now does.
Ranges split_gated(const std::vector<float>& a, size_t* n_dropped = nullptr) {
    return audio_chunking::drop_speechless_ranges(a.data(), a.size(), split_raw(a), WIN, n_dropped);
}

// Ground truth: does [b, e) overlap the speech region [sb, se) by >= 100 ms?
bool has_speech(const std::pair<size_t, size_t>& r, size_t sb, size_t se) {
    const size_t lo = std::max(r.first, sb), hi = std::min(r.second, se);
    return hi > lo && hi - lo >= WIN;
}

size_t count_speechless(const Ranges& rs, size_t sb, size_t se) {
    size_t n = 0;
    for (const auto& r : rs)
        n += has_speech(r, sb, se) ? 0 : 1;
    return n;
}

// Every speech sample [sb, se) lies inside some kept range.
bool covers(const Ranges& rs, size_t sb, size_t se) {
    size_t pos = sb;
    for (const auto& r : rs) {
        if (r.first <= pos && r.second > pos)
            pos = r.second;
        if (pos >= se)
            return true;
    }
    return pos >= se;
}

} // namespace

TEST_CASE("#471: speech then digital silence past the chunk boundary — no silent slice", "[unit][chunking][issue471]") {
    // The issue's repro shape: ~23 s of speech, then 9 s of digital zeros (32 s).
    const size_t speech_end = 23 * SR;
    std::vector<float> a(32 * SR, 0.0f);
    add_speech(a, 0, speech_end, 0.3f, 1);

    // Positive control: the ungated split reproduces the bug.
    const Ranges raw = split_raw(a);
    REQUIRE(raw.size() == 2);
    REQUIRE(count_speechless(raw, 0, speech_end) == 1);

    size_t dropped = 0;
    const Ranges kept = split_gated(a, &dropped);
    REQUIRE(dropped == 1);
    REQUIRE(kept.size() == 1);
    REQUIRE(count_speechless(kept, 0, speech_end) == 0);
    REQUIRE(covers(kept, 0, speech_end)); // no speech lost
    REQUIRE(kept[0] == raw[0]);           // the speech slice is untouched
}

TEST_CASE("#471: speech then ROOM NOISE past the chunk boundary — no noise-only slice", "[unit][chunking][issue471]") {
    // The reporter's harder case: real room noise, not zeros. A remainder of a
    // fraction of a second still returned the whole hotword list (that short
    // case is the next test). Two lengths: 32 s and 30.3 s total.
    for (size_t total_cs : {3200u, 3030u}) {
        const size_t total = total_cs * SR / 100;
        const size_t speech_end = 23 * SR;
        std::vector<float> a(total, 0.0f);
        add_room_noise(a, 0.003f, 7); // ~-50 dBFS
        add_speech(a, 0, speech_end, 0.3f, 2);

        const Ranges raw = split_raw(a);
        REQUIRE(raw.size() == 2);
        REQUIRE(count_speechless(raw, 0, speech_end) == 1); // bug reproduced

        const Ranges kept = split_gated(a);
        REQUIRE(count_speechless(kept, 0, speech_end) == 0);
        REQUIRE(covers(kept, 0, speech_end));
        REQUIRE(kept.size() == 1);
        REQUIRE(kept[0] == raw[0]);
    }
}

TEST_CASE("#471: a 0.3 s room-noise remainder is judged speechless", "[unit][chunking][issue471]") {
    // The issue's shortest failing slice: 0.3 s of room noise after the cut.
    // Where exactly the splitter cuts depends on the audio, so hand the gate
    // that range list directly: speech [0, 29.9 s), noise-only [29.9, 30.2 s).
    const size_t cut = 299 * SR / 10;
    std::vector<float> a(302 * SR / 10, 0.0f);
    add_room_noise(a, 0.003f, 19);
    add_speech(a, 0, cut, 0.3f, 20);
    const Ranges ranges = {{0, cut}, {cut, a.size()}};
    size_t dropped = 0;
    const Ranges kept = audio_chunking::drop_speechless_ranges(a.data(), a.size(), ranges, WIN, &dropped);
    REQUIRE(dropped == 1);
    REQUIRE(kept.size() == 1);
    REQUIRE(kept[0] == ranges[0]);
}

TEST_CASE("#471: a speech-free slice in the MIDDLE is dropped too", "[unit][chunking][issue471]") {
    // speech 0-20 s, 45 s of room noise, speech 65-80 s.
    std::vector<float> a(80 * SR, 0.0f);
    add_room_noise(a, 0.003f, 11);
    add_speech(a, 0, 20 * SR, 0.3f, 3);
    add_speech(a, 65 * SR, 80 * SR, 0.3f, 4);

    auto speech = [](const std::pair<size_t, size_t>& r) {
        return has_speech(r, 0, 20 * SR) || has_speech(r, 65 * SR, 80 * SR);
    };
    const Ranges raw = split_raw(a);
    size_t raw_silent = 0;
    for (const auto& r : raw)
        raw_silent += speech(r) ? 0 : 1;
    REQUIRE(raw_silent >= 1);

    const Ranges kept = split_gated(a);
    for (const auto& r : kept)
        REQUIRE(speech(r));
    REQUIRE(covers(kept, 0, 20 * SR));
    REQUIRE(covers(kept, 65 * SR, 80 * SR));
}

TEST_CASE("#471 control: ordinary speech with pauses is split exactly as before", "[unit][chunking][issue471]") {
    // 95 s of continuous talk (pauses every ~3 s) over room noise: every slice
    // has speech, so the gate must change nothing — same count, same cuts.
    for (float noise : {0.0f, 0.003f, 0.02f}) {
        std::vector<float> a(95 * SR, 0.0f);
        if (noise > 0.0f)
            add_room_noise(a, noise, 5);
        add_speech(a, 0, a.size(), 0.3f, 6);
        const Ranges raw = split_raw(a);
        REQUIRE(raw.size() >= 4);
        size_t dropped = 99;
        const Ranges kept = split_gated(a, &dropped);
        REQUIRE(dropped == 0);
        REQUIRE(kept == raw);
    }
}

TEST_CASE("#471 control: a quiet talker in a loud recording is kept", "[unit][chunking][issue471]") {
    // Loud speaker 0-28 s, then a talker 20 dB quieter 35-58 s (own slice),
    // over room noise 40 dB below the loud speaker.
    std::vector<float> a(60 * SR, 0.0f);
    add_room_noise(a, 0.001f, 13);
    add_speech(a, 0, 28 * SR, 0.3f, 8);
    add_speech(a, 35 * SR, 58 * SR, 0.03f, 9);
    const Ranges raw = split_raw(a);
    REQUIRE(raw.size() >= 2);
    const Ranges kept = split_gated(a);
    REQUIRE(covers(kept, 0, 28 * SR));
    REQUIRE(covers(kept, 35 * SR, 58 * SR));
}

TEST_CASE("#471 control: noise-only and short audio are left alone", "[unit][chunking][issue471]") {
    SECTION("noise-only recording: nothing louder than the noise, nothing dropped") {
        std::vector<float> a(65 * SR, 0.0f);
        add_room_noise(a, 0.003f, 17);
        const Ranges raw = split_raw(a);
        REQUIRE(raw.size() >= 2);
        REQUIRE(split_gated(a) == raw);
    }
    SECTION("single range (audio within the chunk) is never touched, even if silent") {
        std::vector<float> a(10 * SR, 0.0f);
        const Ranges raw = split_raw(a);
        REQUIRE(raw.size() == 1);
        REQUIRE(split_gated(a) == raw);
    }
    SECTION("long all-zero audio: every slice is digital silence, all dropped") {
        std::vector<float> a(65 * SR, 0.0f);
        REQUIRE(split_raw(a).size() >= 2);
        REQUIRE(split_gated(a).empty());
    }
}

TEST_CASE("#471: is_speechless_span thresholds are relative to the recording", "[unit][chunking][issue471]") {
    audio_chunking::energy_levels lv;
    lv.floor = 0.003f;
    lv.loud = 0.1f;
    // threshold = min(4 * 0.003, 0.003 + 0.2 * 0.097) = 0.012
    std::vector<float> quiet(SR, 0.0f), loud(SR, 0.0f);
    add_room_noise(quiet, 0.004f, 21); // noise a bit above the floor
    add_room_noise(loud, 0.02f, 22);   // ~16 dB above the floor
    REQUIRE(audio_chunking::is_speechless_span(quiet.data(), quiet.size(), lv, WIN));
    REQUIRE_FALSE(audio_chunking::is_speechless_span(loud.data(), loud.size(), lv, WIN));

    // No measurable floor -> only digital silence counts.
    audio_chunking::energy_levels none;
    REQUIRE_FALSE(audio_chunking::is_speechless_span(quiet.data(), quiet.size(), none, WIN));
    const std::vector<float> zeros(SR, 0.0f);
    REQUIRE(audio_chunking::is_speechless_span(zeros.data(), zeros.size(), none, WIN));
}

TEST_CASE("#471: CRISPASR_ENERGY_SILENCE_GATE=0 disables the gate", "[unit][chunking][issue471]") {
#if defined(_WIN32)
    _putenv_s("CRISPASR_ENERGY_SILENCE_GATE", "0");
    REQUIRE_FALSE(audio_chunking::speechless_gate_enabled());
    _putenv_s("CRISPASR_ENERGY_SILENCE_GATE", "");
#else
    setenv("CRISPASR_ENERGY_SILENCE_GATE", "0", 1);
    REQUIRE_FALSE(audio_chunking::speechless_gate_enabled());
    unsetenv("CRISPASR_ENERGY_SILENCE_GATE");
#endif
    REQUIRE(audio_chunking::speechless_gate_enabled());
}
