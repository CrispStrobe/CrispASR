// core/asr_overlap_trim.h — drop the part of a chunk whose audio is already
// transcribed.
//
// Backends that decode long audio in OVERLAPPING chunks (canary's
// canary_transcribe_streamed: 8 s chunks, 2 s overlap) transcribe the overlap
// twice, once at the tail of chunk N and again at the head of chunk N+1. The
// second copy has to go.
//
// The usual tool is an LCS over the token ids: find the longest common
// subsequence between the tail of what has been accepted and the head of the
// new chunk, and drop the matched prefix. That works when the decoder emits the
// same words twice. It does nothing when the decoder emits DIFFERENT words for
// the same audio, and an attention decoder given a different amount of
// right-context routinely does exactly that (#365):
//
//   ... Ça c'est quand elle s'est rendue compte
//       qu'elle n'avait pas dormi.
//       J'arrive sur le...
//   ...      rendu compte qu'elle n'avait pas dormi.   <- same audio, re-worded
//
// No common subsequence worth the name, so nothing was dropped, and because the
// second chunk's tokens carry their own chunk offset the timestamps also ran
// BACKWARDS — 00:20:27,060 followed by 00:20:26,820.
//
// Time is the reliable signal here, because the overlap is a property of the
// chunking, not of what the decoder said. Anything ending at or before the last
// accepted token's end covers audio already spoken for, whatever words came out
// this time. This is deliberately a floor UNDER the LCS result rather than a
// replacement: LCS still catches re-emission that spills past the nominal
// overlap, and this catches the re-wording LCS cannot see.
//
// canary already applied exactly this rule to its WORD list and not to its
// token list, which is why word timings looked sane while the text and the SRT
// duplicated.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace core_overlap_trim {

// Number of leading items to drop: those whose end time is at or before
// `accepted_end_cs`, i.e. entirely inside audio already accepted.
//
// `t1_of(i)` returns item i's end time in the same units as accepted_end_cs
// (centiseconds throughout CrispASR). Items are assumed ordered by time within
// the chunk, which is what a decoder emits; the scan stops at the first item
// that extends past the boundary, so a stray out-of-order item later in the
// chunk cannot cause the whole chunk to be dropped.
//
// Never returns `n` when `n > 0` unless every item is covered — callers that
// must keep at least one token should clamp, and canary does.
template <typename F> inline int leading_covered(int n, int64_t accepted_end_cs, F t1_of) {
    if (n <= 0)
        return 0;
    int skip = 0;
    while (skip < n && t1_of(skip) <= accepted_end_cs)
        skip++;
    return skip;
}

// Conservative streamed-seam variant. A single early token can be a genuine
// first word whose decoder timestamp straddles the seam; require at least two
// covered tokens before treating the prefix as re-transcribed overlap.
template <typename F> inline int leading_covered_multi(int n, int64_t accepted_end_cs, F t1_of) {
    const int skip = leading_covered(n, accepted_end_cs, t1_of);
    return skip >= 2 && skip < n ? skip : 0;
}


// ---------------------------------------------------------------------------
// Fuzzy seam dedup
// ---------------------------------------------------------------------------
//
// Time alone cannot separate "this chunk re-said the overlap in different
// words" from "this chunk starts with new speech that happens to carry an
// early timestamp". Trimming the whole overlap window by time was measured
// doing both: it removed a real duplication ("versions are kept on versions
// are kept online") and also ate the leading "Many" from "Many people don't
// think about them as dinosaurs".
//
// What actually distinguishes them is whether the words were already said.
// Exact token LCS misses re-wording ("s'est rendue compte" vs "rendu compte");
// comparing NORMALISED words with a one-edit tolerance does not — "rendue" and
// "rendu" are the same word to a listener, and to this.
//
// So: find the longest prefix of the new chunk that also appears at the tail of
// what was accepted, and drop exactly that. A prefix with no counterpart in the
// tail is new speech and is kept, which is what protects "Many".

inline std::string normalise_word(const std::string& w) {
    std::string out;
    out.reserve(w.size());
    for (unsigned char c : w) {
        if (c >= 0x80)
            out.push_back((char)c); // keep UTF-8 bytes; accents differ, edit distance absorbs it
        else if (std::isalnum(c))
            out.push_back((char)std::tolower(c));
    }
    return out;
}

// Levenshtein, capped: returns >max_d as soon as it is certain, so the common
// "completely different words" case exits without filling a table.
inline int edit_distance_upto(const std::string& a, const std::string& b, int max_d) {
    const int n = (int)a.size(), m = (int)b.size();
    if (std::abs(n - m) > max_d)
        return max_d + 1;
    std::vector<int> prev((size_t)m + 1), cur((size_t)m + 1);
    for (int j = 0; j <= m; j++)
        prev[(size_t)j] = j;
    for (int i = 1; i <= n; i++) {
        cur[0] = i;
        int row_min = cur[0];
        for (int j = 1; j <= m; j++) {
            const int cost = (a[(size_t)i - 1] == b[(size_t)j - 1]) ? 0 : 1;
            cur[(size_t)j] = std::min({prev[(size_t)j] + 1, cur[(size_t)j - 1] + 1, prev[(size_t)j - 1] + cost});
            row_min = std::min(row_min, cur[(size_t)j]);
        }
        if (row_min > max_d)
            return max_d + 1;
        prev.swap(cur);
    }
    return prev[(size_t)m];
}

// Two words are "the same word" if they match after normalisation, allowing one
// edit for short words and two for longer ones — enough for rendue/rendu and
// for a dropped accent, not enough to conflate distinct words.
inline bool words_equivalent(const std::string& a, const std::string& b) {
    const std::string na = normalise_word(a), nb = normalise_word(b);
    if (na.empty() || nb.empty())
        return na == nb;
    if (na == nb)
        return true;
    const int longest = (int)std::max(na.size(), nb.size());
    if (longest < 4)
        return false; // too short to allow slack without merging real words
    return edit_distance_upto(na, nb, longest >= 8 ? 2 : 1) <= (longest >= 8 ? 2 : 1);
}

// How many leading words of `head` repeat the tail of `tail_words`.
//
// Returns the length of the longest prefix of `head` that occurs as a
// contiguous run inside `tail_words`, requiring at least `min_run` words so a
// single common short word ("the", "de") cannot trigger a drop.
inline int leading_repeat_len(const std::vector<std::string>& tail_words, const std::vector<std::string>& head,
                              int min_run = 2) {
    const int H = (int)head.size(), T = (int)tail_words.size();
    if (H <= 0 || T <= 0)
        return 0;
    int best = 0;
    for (int start = 0; start < T; start++) {
        int k = 0;
        while (k < H && start + k < T && words_equivalent(tail_words[(size_t)(start + k)], head[(size_t)k]))
            k++;
        if (k > best)
            best = k;
    }
    return best >= min_run ? best : 0;
}

} // namespace core_overlap_trim
