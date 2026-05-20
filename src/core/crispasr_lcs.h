// crispasr_lcs_merge.h — Longest Common Subsequence merge for adjacent
// overlap-save chunks (#114 / #89 follow-up).
//
// When the CLI splits long audio into overlapping fixed chunks (each chunk
// extended by ±chunk_overlap_seconds on each side so the bidirectional
// encoder has context across the cut), adjacent chunks transcribe the
// shared region twice. The pre-existing word-level boundary filter trimmed
// most of the duplicates but left a 2 × kBoundaryToleranceCs (200 ms)
// window where the same word could appear in both outputs because the TDT
// decoder's emission frame can drift by ±1-2 encoder frames between the
// two passes.
//
// Upstream NeMo's `BatchedFrameASRTDT` solves this with the LCS-based
// hypothesis stitching in
// `nemo/collections/asr/parts/utils/streaming_utils.py:longest_common_subsequence_merge`.
// It operates on the abstract notion of integer token ids — independent of
// text or character encoding, which matters for parakeet-ja where the
// SentencePiece vocab makes byte-level dedup unreliable.
//
// This header ports the algorithm verbatim modulo language idioms so the
// behaviour matches NeMo's reference implementation. The driver lives in
// crispasr_run.cpp; the algorithm is split out so a unit test can pin it
// without a model load.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace crispasr_lcs {

// Minimum LCS length below which no deduplication is performed.
// Matches NeMo's MIN_MERGE_SUBSEQUENCE_LEN default.
constexpr int kMinMergeSubsequenceLen = 1;

// Result of `longest_common_subsequence_merge`. `length` is the LCS length;
// `i` and `j` are the chosen start positions in X (previous-chunk tail) and
// Y (current-chunk tokens) respectively. `slice_count` is the number of
// leading tokens in Y that should be removed to deduplicate against X — it
// matches NeMo's `result_idx[-1]` after the partial-alignment expansion.
struct LcsMerge {
    int i = 0;
    int j = 0;
    int slice_count = 0;
    bool is_complete_merge = false;
};

// Port of NeMo's `longest_common_subsequence_merge`. X is the tail of the
// previous chunk's emitted token ids, Y is the full set of tokens emitted
// by the current chunk.
//
// Returns a slice point. The caller should drop the first
// `LcsMerge::slice_count` tokens from Y (or pass to
// `lcs_dedup_prefix_count` which wraps this with the min-length check).
inline LcsMerge longest_common_subsequence_merge(const std::vector<int32_t>& X, const std::vector<int32_t>& Y) {
    LcsMerge out;
    const int m = (int)X.size();
    const int n = (int)Y.size();
    if (m == 0 || n == 0)
        return out;

    // LCSuff[i][j] = length of longest common SUFFIX of X[0..i) and Y[0..j).
    // Same recurrence as NeMo: LCSuff[i][j] = LCSuff[i-1][j-1]+1 if equal, else 0.
    std::vector<std::vector<int>> LCSuff(m + 1, std::vector<int>(n + 1, 0));
    int best = 0;
    int best_i = 0, best_j = 0;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[(size_t)i - 1] == Y[(size_t)j - 1]) {
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1;
                if (LCSuff[i][j] >= best) {
                    best = LCSuff[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }

    int i = best_i;
    int j = best_j;
    int length = best;
    out.is_complete_merge = (i == m);
    out.slice_count = length;

    if (out.is_complete_merge) {
        // Perfect alignment found — backtrack to LCS origin so we know how
        // many tokens to slice from the head of Y. The backtrack is over
        // already; the slice count is `j` (everything up to and including
        // the matched region in Y is duplicated with X's tail).
        while (length >= 0 && i > 0 && j > 0) {
            if (LCSuff[i - 1][j - 1] > 0) {
                length--;
                i--;
                j--;
            } else {
                i--;
                j--;
                length--;
                break;
            }
        }
        out.i = i;
        out.j = j;
        // slice_count = LCS length (we want to drop the matched prefix of Y)
        out.slice_count = best;
        return out;
    }

    // Partial alignment path (3-step NeMo heuristic):
    //   (1) backward search for leftmost LCS,
    //   (2) greedy diagonal expansion downward,
    //   (3) backtrack with diagonal-skip accounting.
    int max_j = 0;
    int max_j_idx = n;
    int i_partial = m;
    int j_partial = -1;

    for (int ii = m; ii >= 0; ii--) {
        for (int jj = 0; jj <= n; jj++) {
            if (LCSuff[ii][jj] > max_j && jj <= max_j_idx) {
                max_j = LCSuff[ii][jj];
                max_j_idx = jj;
                i_partial = ii;
                j_partial = jj;
            }
        }
    }

    // EARLY EXIT: if the longest leftmost run is below the threshold, do
    // not slice. Long silence regions can produce a small spurious LCS in
    // blank tokens that we should not act on.
    if (max_j <= kMinMergeSubsequenceLen) {
        out.i = i_partial;
        out.j = 0;
        out.slice_count = 0;
        return out;
    }

    // (2) diagonal expansion downward from (i_partial, j_partial)
    int i_temp = i_partial + 1;
    int j_temp = j_partial + 1;
    int j_exp = 0;
    int j_skip = 0;

    for (int ii = i_temp; ii <= m; ii++) {
        int j_any_skip = 0;
        for (int jj = j_temp; jj <= j_temp + j_skip; jj++) {
            if (jj < n + 1) {
                if (LCSuff[ii][jj] == 0)
                    j_any_skip = 1;
                else
                    j_exp = 1 + j_skip + j_any_skip;
            }
        }
        j_skip += j_any_skip;
        j_temp += 1;
    }

    j_skip = 0;
    j_partial += j_exp;

    // (3) partial backward trace
    int slice_count = 0;
    while (i_partial > 0 && j_partial > 0) {
        if (LCSuff[i_partial][j_partial] == 0) {
            j_partial -= 1;
            j_skip += 1;
        }
        if (j_partial > 0) {
            slice_count += 1;
            i_partial -= 1;
            j_partial -= 1;
        }
    }

    out.i = std::max(0, i_partial);
    out.j = std::max(0, j_partial);
    out.slice_count = slice_count + j_skip;
    return out;
}

// Returns the number of leading tokens in `curr_tokens` to drop so they
// don't duplicate `prev_tail_tokens`. Returns 0 if no usable LCS exists
// (below `min_lcs_length`). The caller picks `prev_tail_tokens` —
// typically the last `delay * max_steps_per_timestep` tokens of the
// previous chunk's emission, where `delay = chunk_overlap_seconds /
// encoder_frame_seconds`.
inline int lcs_dedup_prefix_count(const std::vector<int32_t>& prev_tail_tokens, const std::vector<int32_t>& curr_tokens,
                                  int min_lcs_length = kMinMergeSubsequenceLen) {
    if (prev_tail_tokens.empty() || curr_tokens.empty())
        return 0;
    LcsMerge m = longest_common_subsequence_merge(prev_tail_tokens, curr_tokens);
    if (m.slice_count < min_lcs_length)
        return 0;
    return m.slice_count;
}

} // namespace crispasr_lcs
