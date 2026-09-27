// crispasr_ctc_stitch.h — join the CTC logit grids of overlapping audio windows.
//
// Each window is transcribed on its own; its grid is frame-major
// (lg[t * V + v]). Consecutive windows overlap in samples. We keep every
// audio position exactly once by cutting each overlap at one sample position:
// the centre of the longest run of frames BOTH windows call blank. A cut
// through speech can drop or duplicate a token, or lose the word-boundary
// piece ("country" + "and" -> "countryand" with a plain midpoint cut). With no
// joint-blank frame in the overlap, the cut falls back to the midpoint.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace crispasr_ctc_stitch {

struct Window {
    int off = 0; // first sample
    int end = 0; // one past the last sample
    int T = 0;   // frames in lg
    std::vector<float> lg;
};

inline int argmax_at(const Window& w, int t, int V) {
    const float* row = w.lg.data() + (size_t)t * V;
    int best = 0;
    for (int v = 1; v < V; v++)
        if (row[v] > row[best])
            best = v;
    return best;
}

inline int frame_of(const Window& w, double sample) {
    const double fps = (double)w.T / (double)(w.end - w.off);
    int f = (int)std::lround((sample - w.off) * fps);
    return f < 0 ? 0 : (f > w.T ? w.T : f);
}

// Returns the stitched grid; *n_frames receives its frame count.
inline std::vector<float> stitch(const std::vector<Window>& ws, int V, int blank_id, int* n_frames) {
    std::vector<float> out;
    *n_frames = 0;
    const size_t n = ws.size();
    // cut[c] = sample position separating window c from window c+1
    std::vector<double> cut(n > 0 ? n - 1 : 0);
    for (size_t c = 0; c + 1 < n; c++) {
        const Window& a = ws[c];
        const Window& b = ws[c + 1];
        const double lo = b.off, hi = a.end; // overlap [lo, hi)
        const double mid = 0.5 * (lo + hi);
        cut[c] = mid;
        if (hi <= lo)
            continue;
        // Scan the overlap one frame at a time; cut at the centre of the LONGEST
        // run where both windows call blank (ties: nearest the midpoint). A
        // lone joint-blank frame is not enough: when the two windows place one
        // emission on either side of it, the token is kept twice.
        const double spf = (double)(a.end - a.off) / (double)a.T; // samples per frame
        int best_len = 0;
        double best_s = mid;
        int run = 0;
        double run_start = lo;
        for (double s = lo; s < hi; s += spf) {
            const int fa = frame_of(a, s), fb = frame_of(b, s);
            const bool joint =
                fa < a.T && fb < b.T && argmax_at(a, fa, V) == blank_id && argmax_at(b, fb, V) == blank_id;
            if (joint) {
                if (run == 0)
                    run_start = s;
                run++;
                const double centre = run_start + 0.5 * (run - 1) * spf;
                if (run > best_len || (run == best_len && std::fabs(centre - mid) < std::fabs(best_s - mid))) {
                    best_len = run;
                    best_s = centre;
                }
            } else {
                run = 0;
            }
        }
        if (best_len > 0)
            cut[c] = best_s;
    }
    for (size_t c = 0; c < n; c++) {
        const Window& w = ws[c];
        const int f0 = c == 0 ? 0 : frame_of(w, cut[c - 1]);
        const int f1 = c + 1 == n ? w.T : frame_of(w, cut[c]);
        if (f1 > f0) {
            out.insert(out.end(), w.lg.begin() + (size_t)f0 * V, w.lg.begin() + (size_t)f1 * V);
            *n_frames += f1 - f0;
        }
    }
    return out;
}

} // namespace crispasr_ctc_stitch
