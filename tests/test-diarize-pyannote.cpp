// Unit tests for the pyannote-seg → ASR-segment speaker scoring logic,
// the meat of --diarize-method pyannote (issue #107).
//
// We don't load the model or any audio here; we drive
// `assign_speakers_from_log_posteriors` with hand-crafted log-softmax
// posteriors and check the resulting speaker labels.
//
// pyannote-seg-3.0 class layout:
//   0 = silence
//   1 = spk0,        2 = spk1,        3 = spk0 + spk1,
//   4 = spk2,        5 = spk0 + spk2, 6 = spk1 + spk2.

#include "../src/crispasr_diarize.h"
#include "../src/crispasr_diarize_internal.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

using crispasr_diarize_internal::assign_speakers_from_log_posteriors;

namespace {

// Build a [T,7] log-softmax frame from a target class. The target class
// gets ~unit probability (logp ≈ 0), others get a tiny floor (logp = -8
// → p ≈ 3e-4). Realistic enough for the scoring logic — true pyannote
// outputs are not one-hot but the dominant class is usually >0.9.
void push_frame(std::vector<float>& probs, int target_class, float target_logp = -0.001f) {
    const float floor_logp = -8.0f;
    for (int c = 0; c < 7; c++)
        probs.push_back(c == target_class ? target_logp : floor_logp);
}

// pyannote-seg-3.0 frame duration. 270 samples / 16 kHz = 16.875 ms.
constexpr double kFrameDurS = 270.0 / 16000.0;

// Convert a frame count to cs. Half-open: frame [0, n) covers cs [0, ceil(n*frame_dur_s*100)).
int64_t frames_to_cs(int n_frames) {
    return (int64_t)std::ceil((double)n_frames * kFrameDurS * 100.0);
}

} // namespace

TEST_CASE("apply_pyannote: silence segment leaves speaker = -1", "[unit][diarize][pyannote]") {
    std::vector<float> probs;
    const int T = 200;
    for (int f = 0; f < T; f++)
        push_frame(probs, 0); // pure silence

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == -1);
}

TEST_CASE("apply_pyannote: pure-spk0 frames assign segment to spk 0", "[unit][diarize][pyannote]") {
    std::vector<float> probs;
    const int T = 100;
    for (int f = 0; f < T; f++)
        push_frame(probs, 1); // spk0

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 0);
}

TEST_CASE("apply_pyannote: pure-spk1 frames assign segment to spk 1", "[unit][diarize][pyannote]") {
    std::vector<float> probs;
    const int T = 100;
    for (int f = 0; f < T; f++)
        push_frame(probs, 2); // spk1

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 1);
}

TEST_CASE("apply_pyannote: pure-spk2 frames assign segment to spk 2", "[unit][diarize][pyannote]") {
    std::vector<float> probs;
    const int T = 100;
    for (int f = 0; f < T; f++)
        push_frame(probs, 4); // spk2

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 2);
}

TEST_CASE("apply_pyannote: minority-speaker turn within a segment can still surface", "[unit][diarize][pyannote]") {
    // First 30 frames spk0, next 70 frames spk1. Old logic argmaxed per
    // frame then counted classes → spk1 wins with 70 vs 30 (correct
    // here, just on counts). Make sure the new posterior-weighted
    // accumulator also lands on spk1.
    std::vector<float> probs;
    const int T_spk0 = 30, T_spk1 = 70;
    for (int f = 0; f < T_spk0; f++)
        push_frame(probs, 1); // spk0
    for (int f = 0; f < T_spk1; f++)
        push_frame(probs, 2); // spk1
    const int T = T_spk0 + T_spk1;

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 1);
}

TEST_CASE("apply_pyannote: overlap class spk0+spk1 with extra spk1 frames tips to spk1 (regression vs. old LUT)",
          "[unit][diarize][pyannote]") {
    // Old behavior: class_to_speaker[]={...,0,1,0,...} maps class 3
    // (spk0+spk1) to spk0 ONLY, ignoring spk1's coexistence. So 30
    // frames of overlap + 25 frames of pure spk1 would tally as
    // spk0=30, spk1=25 → spk0 wins. New behavior: overlap class
    // contributes to both spk0 and spk1, so spk1's total is
    // 30 (overlap) + 25 (pure) = 55 while spk0 is 30 (overlap) →
    // spk1 wins.
    std::vector<float> probs;
    const int T_overlap = 30, T_spk1_only = 25;
    for (int f = 0; f < T_overlap; f++)
        push_frame(probs, 3); // spk0 + spk1
    for (int f = 0; f < T_spk1_only; f++)
        push_frame(probs, 2); // spk1
    const int T = T_overlap + T_spk1_only;

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 1);
}

TEST_CASE("apply_pyannote: multiple ASR segments get independent labels by frame range", "[unit][diarize][pyannote]") {
    // Frame 0..49 = spk0, 50..99 = spk1, 100..149 = spk2.
    // Three ASR segments aligned to each block.
    std::vector<float> probs;
    for (int f = 0; f < 50; f++)
        push_frame(probs, 1); // spk0
    for (int f = 0; f < 50; f++)
        push_frame(probs, 2); // spk1
    for (int f = 0; f < 50; f++)
        push_frame(probs, 4); // spk2
    const int T = 150;

    std::vector<CrispasrDiarizeSegment> segs = {
        {0, frames_to_cs(50), -1},
        {frames_to_cs(50), frames_to_cs(100), -1},
        {frames_to_cs(100), frames_to_cs(150), -1},
    };
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 0);
    REQUIRE(segs[1].speaker == 1);
    REQUIRE(segs[2].speaker == 2);
}

TEST_CASE("apply_pyannote: slice_t0_cs offset is honored", "[unit][diarize][pyannote]") {
    std::vector<float> probs;
    const int T = 100;
    for (int f = 0; f < T; f++)
        push_frame(probs, 2); // spk1

    // Segment expressed in absolute cs; slice starts at cs=500. So the
    // segment cs=[500, 500+frames_to_cs(T)) maps to local frames [0, T).
    const int64_t slice_t0 = 500;
    std::vector<CrispasrDiarizeSegment> segs = {{slice_t0, slice_t0 + frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, slice_t0, segs);

    REQUIRE(segs[0].speaker == 1);
}

TEST_CASE("apply_pyannote: empty/invalid inputs are no-ops", "[unit][diarize][pyannote]") {
    std::vector<CrispasrDiarizeSegment> segs = {{0, 1000, -1}};

    // Null probs.
    assign_speakers_from_log_posteriors(nullptr, 100, kFrameDurS, 0, segs);
    REQUIRE(segs[0].speaker == -1);

    // T = 0.
    float p = -0.0f;
    assign_speakers_from_log_posteriors(&p, 0, kFrameDurS, 0, segs);
    REQUIRE(segs[0].speaker == -1);

    // Bogus frame duration.
    assign_speakers_from_log_posteriors(&p, 1, 0.0, 0, segs);
    REQUIRE(segs[0].speaker == -1);
}

TEST_CASE("apply_pyannote: posterior weight beats one-hot count when overlap is sustained but uncertain",
          "[unit][diarize][pyannote]") {
    // Build 100 frames where the per-frame argmax is class 1 (spk0)
    // but class 2 (spk1) is only marginally lower — so a one-hot
    // count assigns 100 to spk0 and 0 to spk1, while a posterior-
    // weighted sum gives spk0 ~ 100*0.55 = 55 and spk1 ~ 100*0.45 = 45.
    // Then add 60 frames where class 2 dominates strongly. The
    // posterior-weighted total becomes spk1 ~ 45 + 60 = 105 > spk0 ~ 55.
    // (We mostly assert that the new logic doesn't break the obvious
    // case: 100 frames where spk0 is consistently the top class but
    // never with full confidence still loses to 60 frames of strong
    // spk1.)
    std::vector<float> probs;
    // 100 frames: class 1 logp = log(0.55) ≈ -0.598, class 2 logp = log(0.45) ≈ -0.799.
    for (int f = 0; f < 100; f++) {
        probs.push_back(-8.0f);   // 0
        probs.push_back(-0.598f); // 1 (spk0)
        probs.push_back(-0.799f); // 2 (spk1)
        probs.push_back(-8.0f);   // 3
        probs.push_back(-8.0f);   // 4
        probs.push_back(-8.0f);   // 5
        probs.push_back(-8.0f);   // 6
    }
    // 60 frames of strong spk1.
    for (int f = 0; f < 60; f++)
        push_frame(probs, 2);
    const int T = 160;

    std::vector<CrispasrDiarizeSegment> segs = {{0, frames_to_cs(T), -1}};
    assign_speakers_from_log_posteriors(probs.data(), T, kFrameDurS, 0, segs);

    REQUIRE(segs[0].speaker == 1);
}
