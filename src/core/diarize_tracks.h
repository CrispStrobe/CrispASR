// core/diarize_tracks.h — lower-bound speaker count from pyannote local tracks.
//
// The GMM/BIC speaker-count estimator collapses to k=1 on short inputs (few
// segments, near-duplicate embeddings), silently merging two speakers into one
// label. The pyannote pass that produced the segments already knows better: its
// local track ids come from a single forward pass over the whole file (the #107
// full-audio cache), so within one pass they are globally consistent and their
// COUNT is a sound lower bound on the number of speakers.
//
// Count DISTINCT tracks, never max+1: track ids are not guaranteed dense. A file
// where only tracks 1 and 2 are active has two speakers, and max+1 would claim
// three — forcing a split that does not exist. That distinction is the whole
// reason this is a function with tests rather than one inline expression.
#pragma once

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

namespace core_diarize_tracks {

// The label format written by the diarize CLI, e.g. "(speaker 3) ".
inline constexpr const char* kSpeakerPrefix = "(speaker ";

// Distinct local-track ids across `labels`. Labels that do not carry the
// prefix (unlabelled or already-remapped segments) are ignored rather than
// counted as a track, so a partially-labelled input under-counts instead of
// inventing speakers — under-counting only forgoes the clamp, over-counting
// would force a wrong split.
inline int distinct_track_count(const std::vector<std::string>& labels) {
    std::set<int> tracks;
    const size_t plen = std::string(kSpeakerPrefix).size();
    for (const std::string& s : labels) {
        if (s.rfind(kSpeakerPrefix, 0) != 0)
            continue;
        const char* p = s.c_str() + plen;
        if (*p < '0' || *p > '9') // "(speaker )" or "(speaker x)" — not a track
            continue;
        const int n = std::atoi(p);
        if (n >= 0)
            tracks.insert(n);
    }
    return (int)tracks.size();
}

// Lower bound to hand the clusterer: never below 1, never below the number of
// distinct tracks actually observed.
inline int min_speakers_from_labels(const std::vector<std::string>& labels) {
    const int n = distinct_track_count(labels);
    return n > 1 ? n : 1;
}

} // namespace core_diarize_tracks
