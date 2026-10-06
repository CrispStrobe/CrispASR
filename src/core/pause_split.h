#pragma once
// core/pause_split.h — cut audio at its pauses, without a VAD model.
//
// For decoders that end their output at the first pause they hear
// (moonshine-de: trained on single short sentences, it emits EOS after a
// sentence-final pause and drops whatever follows). Splitting the input at
// pauses and decoding each piece gives them one sentence at a time, which is
// what they were trained on.
//
// A pause is a run of 20 ms frames whose level is low *for this clip*: below
// the noise floor plus a third of the way up to the speech level (the
// quietest frame and the 90th percentile of the frame levels in dB), so it adapts to a noisy
// microphone and to clean synthetic audio alike. Cuts go to the middle of
// each pause of at least `min_pause_ms`; a piece shorter than `min_piece_ms`
// is merged into its neighbour rather than decoded on its own.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace core_pause_split {

// Returns [start, end) sample ranges covering [0, n) without gaps.
inline std::vector<std::pair<int, int>> pieces(const float* x, int n, int sample_rate, int min_pause_ms,
                                               int min_piece_ms = 600) {
    std::vector<std::pair<int, int>> out;
    if (!x || n <= 0)
        return out;
    const int frame = std::max(1, sample_rate / 50); // 20 ms
    const int n_frames = n / frame;
    const int min_pause_frames = std::max(1, (min_pause_ms * sample_rate / 1000 + frame - 1) / frame);
    if (n_frames < 2 * min_pause_frames + 2) {
        out.push_back({0, n});
        return out;
    }
    std::vector<float> db(n_frames);
    for (int f = 0; f < n_frames; f++) {
        double e = 0.0;
        for (int i = 0; i < frame; i++) {
            const double v = x[(size_t)f * frame + i];
            e += v * v;
        }
        db[f] = (float)(10.0 * std::log10(e / frame + 1e-12));
    }
    // Floor: the quietest frame after a 3-frame median (one odd frame is not
    // a pause). Not a low percentile: a 0.2 s pause in a 10 s slice is 2 % of
    // its frames, and a 10th-percentile floor then sits on speech and finds
    // no pause at all.
    float floor_db = db[0];
    for (int f = 0; f < n_frames; f++) {
        float w[3] = {db[std::max(0, f - 1)], db[f], db[std::min(n_frames - 1, f + 1)]};
        std::sort(w, w + 3);
        floor_db = std::min(floor_db, w[1]);
    }
    std::vector<float> sorted = db;
    std::sort(sorted.begin(), sorted.end());
    const float speech_db = sorted[(size_t)(0.90 * (n_frames - 1))];
    if (speech_db - floor_db < 12.0f) { // no clear pauses: speech throughout, or noise throughout
        out.push_back({0, n});
        return out;
    }
    // A third of the way up from the floor, but never below 30 dB under the
    // speech level: one stretch of digital silence would otherwise pull the
    // threshold under the level of every real (noisy) pause.
    const float thr = std::max(floor_db + (speech_db - floor_db) / 3.0f, speech_db - 30.0f);

    // Cut points: middle of every quiet run long enough, never at the edges.
    std::vector<int> cuts;
    for (int f = 0; f < n_frames;) {
        if (db[f] >= thr) {
            ++f;
            continue;
        }
        int g = f;
        while (g < n_frames && db[g] < thr)
            ++g;
        if (g - f >= min_pause_frames && f > 0 && g < n_frames)
            cuts.push_back(((f + g) / 2) * frame);
        f = g;
    }
    int start = 0;
    const int min_piece = min_piece_ms * sample_rate / 1000;
    for (int c : cuts) {
        if (c - start < min_piece)
            continue; // too short on its own: keep it with what follows
        out.push_back({start, c});
        start = c;
    }
    if (!out.empty() && n - start < min_piece)
        out.back().second = n; // a short tail joins the last piece
    else
        out.push_back({start, n});
    return out;
}

} // namespace core_pause_split
