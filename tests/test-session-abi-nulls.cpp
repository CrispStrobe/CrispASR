// test-session-abi-nulls.cpp — model-free contracts of the session C-ABI (#332).
//
// Two families, both runnable with no model and no audio fixture:
//
//   1. NULL-session getters return their documented "no data" values instead
//      of dereferencing. These are the first calls every binding makes; a
//      crash here takes down the host VM (JVM / Dart isolate / Ruby), so the
//      NULL contract is load-bearing, not defensive decoration.
//   2. crispasr_diarize_segments_abi argument validation and the model-free
//      diarize methods (Energy / VadTurns), exercised through the SAME
//      48-byte options mirror the language bindings hand-maintain. This is
//      deliberately a fourth mirror of the append-only layout: if the C
//      struct grows and this file is not updated in the same commit, the
//      layout static_asserts in crispasr_c_api.cpp fail the build before
//      this test can even lie about passing (#324's miss, #332's fix).

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

extern "C" {
struct crispasr_session;
int crispasr_session_input_sample_rate(crispasr_session* s);
int crispasr_session_output_sample_rate(crispasr_session* s);
int crispasr_session_input_channels(crispasr_session* s);
int crispasr_session_output_channels(crispasr_session* s);
int crispasr_session_set_pcm_sample_rate(crispasr_session* s, int rate);

// Hand-written mirror of the append-only ABI structs, byte-identical to the
// ones in crispasr_c_api.cpp (pinned there by static_asserts) and to the
// mirrors in crispasr-sys, flutter and the Go cgo preamble.
struct diarize_seg_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker;
    int32_t _pad;
};
struct diarize_opts_abi {
    int32_t method;
    int32_t n_threads;
    int64_t slice_t0_cs;
    const char* pyannote_model_path;
    const char* foxnose_embedder_path;
    int32_t min_speakers;
    int32_t max_speakers;
    int32_t num_speakers;
    int32_t _pad2;
};
// #395: the turn struct is NOT append-only-shared with the opts struct — it is
// its own 24-byte POD, mirrored here for the same reason as the others.
struct diarize_turn_abi {
    int64_t t0_cs;
    int64_t t1_cs;
    int32_t speaker;
    int32_t _pad;
};
int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples, int32_t is_stereo,
                                  diarize_seg_abi* segs, int32_t n_segs, const diarize_opts_abi* opts);
int crispasr_diarize_segments_turns_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                        int32_t is_stereo, diarize_seg_abi* segs, int32_t n_segs,
                                        const diarize_opts_abi* opts, diarize_turn_abi* out_turns, int32_t n_turns_cap,
                                        int32_t* out_n_turns);
}

namespace {

diarize_opts_abi default_opts(int method) {
    diarize_opts_abi o = {};
    o.method = method;
    o.n_threads = 1;
    return o;
}

} // namespace

TEST_CASE("NULL session: every #332 getter returns its no-data value", "[unit][abi]") {
    CHECK(crispasr_session_input_sample_rate(nullptr) == 0);
    CHECK(crispasr_session_output_sample_rate(nullptr) == 0);
    CHECK(crispasr_session_input_channels(nullptr) == 0);
    CHECK(crispasr_session_output_channels(nullptr) == 0);
    CHECK(crispasr_session_set_pcm_sample_rate(nullptr, 16000) == -1);
}

TEST_CASE("diarize ABI rejects invalid arguments with -1", "[unit][diarize]") {
    std::vector<float> pcm(16000, 0.01f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(2); // VadTurns

    CHECK(crispasr_diarize_segments_abi(nullptr, nullptr, 16000, 0, &seg, 1, &opts) == -1);
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, nullptr, 1, &opts) == -1);
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 0, &opts) == -1);
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, nullptr) == -1);

    diarize_opts_abi bad_method = default_opts(6); // one past Sortformer
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &bad_method) == -1);
    bad_method.method = -1;
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &bad_method) == -1);
}

TEST_CASE("VadTurns alternates speakers across a >600 ms gap", "[unit][diarize]") {
    std::vector<float> pcm(16000 * 4, 0.01f);
    diarize_seg_abi segs[2] = {
        {0, 100, -1, 0},   // 0.0 – 1.0 s
        {200, 300, -1, 0}, // 2.0 – 3.0 s: 1 s gap
    };
    diarize_opts_abi opts = default_opts(2);
    REQUIRE(crispasr_diarize_segments_abi(pcm.data(), nullptr, (int32_t)pcm.size(), 0, segs, 2, &opts) == 0);
    CHECK(segs[0].speaker != segs[1].speaker);
    CHECK(segs[0].speaker >= 0);
    CHECK(segs[1].speaker >= 0);
}

TEST_CASE("Energy picks the louder channel; ties stay unlabelled", "[unit][diarize]") {
    // Segment 0 (0–1 s): left loud, right quiet. Segment 1 (1–2 s): reversed.
    // Segment 2 (2–3 s): equal energy — inside the 1.1x margin, so -1.
    const int sr = 16000;
    std::vector<float> left(sr * 3, 0.0f), right(sr * 3, 0.0f);
    for (int i = 0; i < sr; i++) {
        left[i] = 0.5f;
        right[i] = 0.05f;
        left[sr + i] = 0.05f;
        right[sr + i] = 0.5f;
        left[2 * sr + i] = 0.3f;
        right[2 * sr + i] = 0.3f;
    }
    diarize_seg_abi segs[3] = {{0, 100, -1, 0}, {100, 200, -1, 0}, {200, 300, -1, 0}};
    diarize_opts_abi opts = default_opts(0); // Energy
    REQUIRE(crispasr_diarize_segments_abi(left.data(), right.data(), sr * 3, 1, segs, 3, &opts) == 0);
    CHECK(segs[0].speaker == 0);
    CHECK(segs[1].speaker == 1);
    CHECK(segs[2].speaker == -1);
}

TEST_CASE("Energy on mono input succeeds but labels nothing", "[unit][diarize]") {
    std::vector<float> pcm(16000, 0.2f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(0);
    REQUIRE(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &opts) == 0);
    CHECK(seg.speaker == -1);
}

TEST_CASE("slice_t0 maps absolute segment times into the buffer", "[unit][diarize]") {
    // The buffer starts at absolute t = 10 s. A segment at 10–11 s must land
    // on samples [0, 16000) — labelled from the left-loud first second — and
    // one at 11–12 s on the right-loud second.
    const int sr = 16000;
    std::vector<float> left(sr * 2, 0.0f), right(sr * 2, 0.0f);
    for (int i = 0; i < sr; i++) {
        left[i] = 0.5f;
        right[i] = 0.05f;
        left[sr + i] = 0.05f;
        right[sr + i] = 0.5f;
    }
    diarize_seg_abi segs[2] = {{1000, 1100, -1, 0}, {1100, 1200, -1, 0}};
    diarize_opts_abi opts = default_opts(0);
    opts.slice_t0_cs = 1000;
    REQUIRE(crispasr_diarize_segments_abi(left.data(), right.data(), sr * 2, 1, segs, 2, &opts) == 0);
    CHECK(segs[0].speaker == 0);
    CHECK(segs[1].speaker == 1);
}

TEST_CASE("FoxNose with a missing embedder fails with 1, not a crash", "[unit][diarize]") {
    std::vector<float> pcm(16000, 0.01f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(4); // FoxNose
    opts.foxnose_embedder_path = "/nonexistent/wespeaker.gguf";
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &opts) == 1);
}

TEST_CASE("Sortformer with a missing model fails with 1, not a crash", "[unit][diarize]") {
    std::vector<float> pcm(16000, 0.01f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(5); // Sortformer (#466); model path rides in pyannote_model_path
    opts.pyannote_model_path = "/nonexistent/nemotron3-diar.gguf";
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &opts) == 1);
    opts.pyannote_model_path = nullptr; // no path at all is the same failure
    CHECK(crispasr_diarize_segments_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &opts) == 1);
}

// ── #395: the turn-forwarding entry point ────────────────────────────────────
//
// The turns themselves need the FoxNose embedder, so what is model-free here
// is the CONTRACT: the same argument validation, the same labelling as the
// older symbol when no turn buffer is asked for, and 0 turns (not an error)
// from the methods that don't derive any.

TEST_CASE("turns ABI rejects the same invalid arguments, plus a negative cap", "[unit][diarize]") {
    std::vector<float> pcm(16000, 0.01f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(2); // VadTurns
    diarize_turn_abi turns[4] = {};
    int32_t n_turns = -7;

    CHECK(crispasr_diarize_segments_turns_abi(nullptr, nullptr, 16000, 0, &seg, 1, &opts, turns, 4, &n_turns) == -1);
    CHECK(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, 16000, 0, nullptr, 1, &opts, turns, 4, &n_turns) ==
          -1);
    CHECK(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, 16000, 0, &seg, 0, &opts, turns, 4, &n_turns) == -1);
    CHECK(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, nullptr, turns, 4, &n_turns) ==
          -1);
    CHECK(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &opts, turns, -1, &n_turns) ==
          -1);

    diarize_opts_abi bad_method = default_opts(6); // one past Sortformer
    CHECK(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &bad_method, turns, 4,
                                              &n_turns) == -1);

    // A rejected call writes nothing — not even the count.
    CHECK(n_turns == -7);
}

TEST_CASE("turns ABI with no turn buffer is the older symbol", "[unit][diarize]") {
    // Same input as "VadTurns alternates speakers across a >600 ms gap", run
    // through both entry points: the labels must agree exactly.
    std::vector<float> pcm(16000 * 4, 0.01f);
    diarize_seg_abi old_segs[2] = {{0, 100, -1, 0}, {200, 300, -1, 0}};
    diarize_seg_abi new_segs[2] = {{0, 100, -1, 0}, {200, 300, -1, 0}};
    diarize_opts_abi opts = default_opts(2);

    REQUIRE(crispasr_diarize_segments_abi(pcm.data(), nullptr, (int32_t)pcm.size(), 0, old_segs, 2, &opts) == 0);
    REQUIRE(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, (int32_t)pcm.size(), 0, new_segs, 2, &opts,
                                                nullptr, 0, nullptr) == 0);
    CHECK(new_segs[0].speaker == old_segs[0].speaker);
    CHECK(new_segs[1].speaker == old_segs[1].speaker);
}

TEST_CASE("a method that derives no turns reports 0, not an error", "[unit][diarize]") {
    // Energy labels caller segments directly; only FoxNose derives turns.
    const int sr = 16000;
    std::vector<float> left(sr, 0.5f), right(sr, 0.05f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(0); // Energy
    diarize_turn_abi turns[4] = {};
    int32_t n_turns = -7;

    // With a buffer: 0 turns can never overflow it, so no truncation code.
    REQUIRE(crispasr_diarize_segments_turns_abi(left.data(), right.data(), sr, 1, &seg, 1, &opts, turns, 4, &n_turns) ==
            0);
    CHECK(n_turns == 0);
    CHECK(seg.speaker == 0); // labelling still happened

    // Count-only query (no buffer) is legal and returns 0 as well.
    seg.speaker = -1;
    n_turns = -7;
    REQUIRE(crispasr_diarize_segments_turns_abi(left.data(), right.data(), sr, 1, &seg, 1, &opts, nullptr, 0,
                                                &n_turns) == 0);
    CHECK(n_turns == 0);
    CHECK(seg.speaker == 0);

    // A zero cap is not truncation when there is nothing to truncate.
    REQUIRE(crispasr_diarize_segments_turns_abi(left.data(), right.data(), sr, 1, &seg, 1, &opts, turns, 0, &n_turns) ==
            0);
    CHECK(n_turns == 0);
}

TEST_CASE("turns ABI: FoxNose with a missing embedder still fails with 1", "[unit][diarize]") {
    std::vector<float> pcm(16000, 0.01f);
    diarize_seg_abi seg = {0, 100, -1, 0};
    diarize_opts_abi opts = default_opts(4); // FoxNose
    opts.foxnose_embedder_path = "/nonexistent/wespeaker.gguf";
    diarize_turn_abi turns[4] = {};
    int32_t n_turns = -7;
    CHECK(crispasr_diarize_segments_turns_abi(pcm.data(), nullptr, 16000, 0, &seg, 1, &opts, turns, 4, &n_turns) == 1);
    CHECK(n_turns == -7); // a failed run writes no count
}
