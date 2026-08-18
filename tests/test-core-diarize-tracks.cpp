// test-core-diarize-tracks.cpp — lower-bound speaker count from local tracks.
//
// Guards the clamp added in #368: the GMM/BIC estimator collapses to k=1 on
// short inputs and merges two speakers into one label, so the clusterer is given
// a floor derived from the pyannote local tracks already on the segments.
//
// The case worth pinning is the SPARSE one. The obvious implementation is
// max_track + 1, and it is wrong: a file where only tracks 1 and 2 are active
// has two speakers, and max+1 claims three — forcing a split that does not
// exist. Distinct-count and max+1 agree on every dense input, so a test that
// only used 0,1,2 would pass against the buggy version too.
#include "core/diarize_tracks.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using core_diarize_tracks::distinct_track_count;
using core_diarize_tracks::min_speakers_from_labels;

TEST_CASE("diarize tracks: distinct count, not max+1", "[unit][diarize-tracks][issue-368]") {
    // Sparse ids: only tracks 1 and 2 are active. Two speakers, not three.
    const std::vector<std::string> sparse = {"(speaker 1) ", "(speaker 2) ", "(speaker 1) "};
    REQUIRE(distinct_track_count(sparse) == 2);
    REQUIRE(min_speakers_from_labels(sparse) == 2);

    // A max+1 implementation would say 3 here; assert we do not.
    REQUIRE(min_speakers_from_labels(sparse) != 3);

    // Wider gap, same principle.
    const std::vector<std::string> gapped = {"(speaker 0) ", "(speaker 7) "};
    REQUIRE(min_speakers_from_labels(gapped) == 2);
}

TEST_CASE("diarize tracks: a single speaker is not split", "[unit][diarize-tracks][issue-368]") {
    // The clamp must never FORCE a split — one track means the floor stays 1 and
    // the estimator is free to choose k=1.
    const std::vector<std::string> one = {"(speaker 0) ", "(speaker 0) ", "(speaker 0) "};
    REQUIRE(distinct_track_count(one) == 1);
    REQUIRE(min_speakers_from_labels(one) == 1);
}

TEST_CASE("diarize tracks: the reported two-speaker case", "[unit][diarize-tracks][issue-368]") {
    // samples/multispeaker.wav: 4 embeddable segments alternating between two
    // pyannote tracks. Reported as n_emb=4 -> k=1 before the clamp.
    const std::vector<std::string> segs = {"(speaker 0) ", "(speaker 1) ", "(speaker 0) ", "(speaker 1) "};
    REQUIRE(min_speakers_from_labels(segs) == 2);
}

TEST_CASE("diarize tracks: unlabelled and malformed input under-counts, never invents",
          "[unit][diarize-tracks][issue-368]") {
    // Over-counting forces a wrong split; under-counting merely forgoes the
    // clamp. So anything unparseable must be ignored, not guessed at.
    REQUIRE(distinct_track_count({}) == 0);
    REQUIRE(min_speakers_from_labels({}) == 1);

    const std::vector<std::string> mixed = {
        "(speaker 0) ", "", "hello", "(speaker )", "(speaker x) ", "speaker 5", "(speaker 1) ",
    };
    REQUIRE(distinct_track_count(mixed) == 2); // only the two real ones
    REQUIRE(min_speakers_from_labels(mixed) == 2);
}

TEST_CASE("diarize tracks: multi-digit track ids parse whole", "[unit][diarize-tracks][issue-368]") {
    // atoi from the wrong offset would read "1" out of "(speaker 12) " and
    // silently collide with track 1.
    const std::vector<std::string> many = {"(speaker 1) ", "(speaker 12) ", "(speaker 120) "};
    REQUIRE(distinct_track_count(many) == 3);
}
