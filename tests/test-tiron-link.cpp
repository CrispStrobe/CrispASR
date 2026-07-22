// test-tiron-link.cpp — Tiron cross-window speaker linking (#295).
//
// Uses a deterministic fake embedder so the grouping + clustering logic is
// exercised without a real TitaNet/ECAPA model. The fake maps a PCM range to a
// one-hot voiceprint keyed on its first sample, so audio filled with constant
// `k` embeds to basis vector e_k and cosine(e_i, e_j) = [i == j].

#include "tiron_link.h"

#include "crispasr_speaker_embedder.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

namespace {

// dim-4 fake: voiceprint = e_k where k = round(pcm[0]) clamped to [0,3].
class FakeEmbedder : public CrispasrSpeakerEmbedder {
public:
    int dim() const override { return 4; }
    bool embed(const float* pcm, int n, float* out) override {
        if (!pcm || n <= 0)
            return false;
        int k = (int)std::lround(pcm[0]);
        if (k < 0)
            k = 0;
        if (k > 3)
            k = 3;
        for (int i = 0; i < 4; i++)
            out[i] = (i == k) ? 1.0f : 0.0f;
        return true;
    }
    const char* name() const override { return "fake"; }
};

// Fill [t0_cs, t1_cs) of a 16 kHz buffer with a constant voice code.
void paint(std::vector<float>& pcm, int64_t t0_cs, int64_t t1_cs, float code) {
    for (int64_t s = t0_cs * 160; s < t1_cs * 160 && s < (int64_t)pcm.size(); s++)
        pcm[s] = code;
}

} // namespace

TEST_CASE("tiron link: voice, not local index, drives global ids", "[unit][tiron]") {
    // Two physical speakers A (code 1) and B (code 2), arranged so the
    // window-local index is DELIBERATELY misleading:
    //   turn0  w0 local1  = A
    //   turn1  w0 local2  = B
    //   turn2  w1 local1  = B   (local1 again, but a different voice than w0.local1)
    //   turn3  w1 local2  = A   (A reappears under a different local index)
    std::vector<float> pcm(512000, 0.0f);
    paint(pcm, 0, 100, 1.0f);     // turn0  A
    paint(pcm, 100, 200, 2.0f);   // turn1  B
    paint(pcm, 3000, 3100, 2.0f); // turn2  B
    paint(pcm, 3100, 3200, 1.0f); // turn3  A

    std::vector<TironTurn> turns = {
        {0, 100, 0, 1},
        {100, 200, 0, 2},
        {3000, 3100, 1, 1},
        {3100, 3200, 1, 2},
    };

    FakeEmbedder emb;
    TironLinkResult r = crispasr_tiron_link_speakers(turns, pcm.data(), (int)pcm.size(), &emb);

    REQUIRE(r.turn_speaker.size() == 4);
    REQUIRE(r.n_speakers == 2);
    // A's two turns share an id; B's two turns share an id; A != B.
    CHECK(r.turn_speaker[0] == r.turn_speaker[3]); // both A, despite different local index
    CHECK(r.turn_speaker[1] == r.turn_speaker[2]); // both B, despite local1 in w1
    CHECK(r.turn_speaker[0] != r.turn_speaker[1]);
}

TEST_CASE("tiron link: within-window must-link rescues a too-short turn", "[unit][tiron]") {
    // A short second turn of (w0, local1) can't embed on its own, but it is in
    // the same must-link group as a long turn of the same window-speaker, so it
    // inherits that group's cluster.
    std::vector<float> pcm(64000, 0.0f);
    paint(pcm, 0, 100, 1.0f);   // turn0  A, 1.0 s
    paint(pcm, 100, 200, 2.0f); // turn1  B, 1.0 s
    paint(pcm, 200, 205, 1.0f); // turn2  A, 0.05 s  (too short alone)

    std::vector<TironTurn> turns = {
        {0, 100, 0, 1},   // A
        {100, 200, 0, 2}, // B
        {200, 205, 0, 1}, // A (short) — same (w0, local1) group as turn0
    };

    FakeEmbedder emb;
    TironLinkResult r = crispasr_tiron_link_speakers(turns, pcm.data(), (int)pcm.size(), &emb);

    REQUIRE(r.turn_speaker.size() == 3);
    CHECK(r.turn_speaker[2] == r.turn_speaker[0]); // short A turn linked to long A turn
    CHECK(r.turn_speaker[0] != r.turn_speaker[1]);
}

TEST_CASE("tiron link: no embedder degrades to per-group ids", "[unit][tiron]") {
    std::vector<float> pcm(512000, 0.0f);
    paint(pcm, 0, 100, 1.0f);
    paint(pcm, 3000, 3100, 1.0f);
    std::vector<TironTurn> turns = {
        {0, 100, 0, 1}, {3000, 3100, 1, 1}, // same local index, but no acoustic info to link
    };

    TironLinkResult r = crispasr_tiron_link_speakers(turns, pcm.data(), (int)pcm.size(), nullptr);

    REQUIRE(r.turn_speaker.size() == 2);
    // Without embeddings the two windows can't be linked → distinct ids.
    CHECK(r.turn_speaker[0] != r.turn_speaker[1]);
    CHECK(r.n_speakers == 2);
}

TEST_CASE("tiron link: empty input is safe", "[unit][tiron]") {
    FakeEmbedder emb;
    TironLinkResult r = crispasr_tiron_link_speakers({}, nullptr, 0, &emb);
    CHECK(r.turn_speaker.empty());
    CHECK(r.n_speakers == 0);
}
