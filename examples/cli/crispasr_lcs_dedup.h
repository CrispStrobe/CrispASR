// crispasr_lcs_dedup.h — LCS-based deduplication driver for overlap-save
// chunk boundaries (issue #89 / #114 follow-up).
//
// Wraps the pure-algorithm `crispasr_lcs::lcs_dedup_prefix_count` with a
// segment-aware driver that knows how to walk the per-slice
// `std::vector<crispasr_segment>` we get from the parakeet (and other
// CAP_UNBOUNDED_INPUT) backends and physically drop the first N
// duplicate tokens from a chunk while keeping the segment / word /
// text bookkeeping consistent.
//
// The function is split into this header from `crispasr_run.cpp` so the
// unit test in `tests/test-lcs-dedup-driver.cpp` can exercise it
// against synthetic segment fixtures without a model load.

#pragma once

#include "crispasr_backend.h" // crispasr_segment / _word / _token
#include "core/crispasr_lcs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace crispasr_lcs {

// Flatten the last `tail_tokens` token ids of a single chunk's segments,
// preserving order. tail_tokens <= 0 → all tokens.
inline std::vector<int32_t> collect_tail_token_ids(const std::vector<crispasr_segment>& segs, int tail_tokens) {
    std::vector<int32_t> ids;
    for (const auto& s : segs)
        for (const auto& t : s.tokens)
            ids.push_back(t.id);
    if (tail_tokens > 0 && (int)ids.size() > tail_tokens)
        ids.erase(ids.begin(), ids.begin() + (ids.size() - (size_t)tail_tokens));
    return ids;
}

// Drop the first `n` tokens from a chunk's segments in place, removing
// segments that become empty, peeling matching words from the head where
// the timestamps line up, and rebuilding `seg.text` from the surviving
// tokens. Returns the number of tokens actually removed (≤ n; can be less
// if the chunk has fewer tokens than requested).
inline int drop_leading_tokens(std::vector<crispasr_segment>& segs, int n) {
    if (n <= 0)
        return 0;
    int dropped = 0;

    auto rebuild_text_from_tokens = [](crispasr_segment& s) {
        s.text.clear();
        for (const auto& t : s.tokens)
            s.text += t.text;
        if (!s.text.empty() && s.text.front() == ' ')
            s.text.erase(0, 1);
    };

    while (dropped < n && !segs.empty()) {
        auto& s = segs.front();
        if ((int)s.tokens.size() <= n - dropped) {
            // Whole segment is duplicate — remove.
            dropped += (int)s.tokens.size();
            segs.erase(segs.begin());
            continue;
        }
        // Partial — slice off the head tokens.
        const int peel = n - dropped;
        const int64_t cutoff_cs = s.tokens[(size_t)peel - 1].t1;
        s.tokens.erase(s.tokens.begin(), s.tokens.begin() + peel);
        dropped += peel;

        // Peel words whose t1 is at or before cutoff_cs (best effort —
        // words and tokens don't have a strict 1:1 mapping but tokens
        // are always emitted before the closing word boundary).
        auto first_keep =
            std::find_if(s.words.begin(), s.words.end(), [&](const crispasr_word& w) { return w.t0 > cutoff_cs - 1; });
        s.words.erase(s.words.begin(), first_keep);

        rebuild_text_from_tokens(s);
        s.t0 = s.tokens.empty() ? cutoff_cs : s.tokens.front().t0;
        if (s.text.empty()) {
            segs.erase(segs.begin());
        }
        break;
    }
    return dropped;
}

// Run LCS dedup across adjacent chunks. Each pair (per_slice[i-1],
// per_slice[i]) is checked: if the tail of i-1 and the head of i share a
// common token-id subsequence of length ≥ `min_lcs_length`, the matched
// prefix of i is dropped. `delay_tokens` bounds how far back into i-1 we
// look — set to roughly `chunk_overlap_seconds / encoder_frame_seconds *
// max_tokens_per_frame`. The wrapper is no-op for chunks with empty
// tokens (e.g. silence regions); the surface for false-positive dedup is
// controlled entirely by `min_lcs_length`.
inline void apply_lcs_chunk_dedup(std::vector<std::vector<crispasr_segment>>& per_slice, int delay_tokens,
                                  int min_lcs_length = kMinMergeSubsequenceLen) {
    for (size_t i = 1; i < per_slice.size(); i++) {
        const auto& prev = per_slice[i - 1];
        const auto prev_tail = collect_tail_token_ids(prev, delay_tokens);
        if (prev_tail.empty())
            continue;
        const auto curr_ids = collect_tail_token_ids(per_slice[i], /*tail=*/-1);
        if (curr_ids.empty())
            continue;
        const int slice = lcs_dedup_prefix_count(prev_tail, curr_ids, min_lcs_length);
        if (slice > 0)
            drop_leading_tokens(per_slice[i], slice);
    }
}

} // namespace crispasr_lcs
