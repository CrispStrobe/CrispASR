// crispasr_gap_fill.h — issue #89 gap-fill second pass, shared between the
// CLI dispatcher (crispasr_run.cpp) and the HTTP server (crispasr_server.cpp).
//
// Bounded-window backends (parakeet-ja) sometimes emit nothing for a
// multi-second span *inside* a slice — the encoder blanks an utterance
// whenever enough context follows it, even though the same span transcribes
// verbatim in isolation (measured: the issue #89 reporter's 60 s clip's
// first 4.6 s transcribe perfectly alone but are skipped inside any ≥8 s
// window). Second pass: find spans ≥ min_gap with no emitted words, run
// each through the backend in isolation, keep words that land inside the
// gap (and don't restate covered content), and merge them back. Silence
// gaps re-transcribe to nothing and cost little.
//
// Gate at the call site on backend.vad_slice_cap_seconds() > 0;
// CRISPASR_GAP_FILL=0 disables, CRISPASR_GAP_FILL_MIN_CS tunes the trigger
// (60 measured worse than the default 100: more variant noise, slower).
//
// Issue #356: a recovery lands *inside* the span of the segment it was
// recovered from — the first pass emitted one segment whose sparse words
// straddle the hole, so its t0..t1 covers the recovered range. Appending the
// recovery and sorting by t0 alone leaves the covering segment first and every
// recovery after it reading as a jump backwards in time (SRT/VTT cues overlap
// by up to the hole length, 10+ s on the reporter's file). After each round
// the covering segment is now split at the recovered range so the output is
// monotone: head words, recovery, tail words.

//
// Issue #605: on noisy audio (a game stream) most empty spans are pauses,
// music and sound effects, and refilling them made the decoder invent words
// that each became a short subtitle cue. A recovery is now kept only if its
// mean token confidence clears core_gap_fill::min_run_confidence(), and a
// recovered word that repeats the first-pass word at the hole's edge is
// dropped. Whether the pass runs at all is the backend's call
// (gap_fill_by_default(); Haas-ja shows no #89 drop and defaults off), with
// CRISPASR_GAP_FILL forcing it either way. See src/core/gap_fill_policy.h.

#pragma once

#include "crispasr_backend.h"
#include "crispasr_vad.h"
#include "core/gap_fill_policy.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// Rebuild display text from a word list. Two word conventions coexist:
// whisper/parakeet carry a leading space in word.text (" on") while
// granite's [T:N]-parsed words do not; CJK boundaries never get a space
// (#205 / 617cd02).
inline std::string crispasr_rebuild_text_from_words(const std::vector<crispasr_word>& words) {
    std::string rebuilt;
    for (const auto& w : words) {
        if (w.text.empty())
            continue;
        if (!rebuilt.empty()) {
            const unsigned char prev_last = (unsigned char)rebuilt.back();
            const unsigned char cur_first = (unsigned char)w.text[0];
            const bool already_spaced = (cur_first == ' ');
            const bool cjk_boundary = (prev_last >= 0xE0) || (cur_first >= 0xE0);
            if (!already_spaced && !cjk_boundary)
                rebuilt += ' ';
        }
        rebuilt += w.text;
    }
    if (!rebuilt.empty() && rebuilt[0] == ' ')
        rebuilt = rebuilt.substr(1);
    return rebuilt;
}

// Issue #356: make a gap-filled segment list monotone. The first pass emits a
// segment whose sparse words straddle the recovered hole, so its span encloses
// the recoveries; sorting by t0 alone leaves that segment first and the
// recoveries overlap it. Resolve by splitting the covering segment at each
// recovery: words (and tokens) that start at or after the recovery's end move
// into a new tail segment, the head keeps the rest, and both rebuild their
// display text from their words — the same convention as the overlap-save trim
// in crispasr_run.cpp. A covering segment with no words on the far side of the
// recovery is simply clamped at the recovery's start. Runs to a fixpoint so a
// span enclosing several recoveries is split at each one; every split strictly
// reduces the word count of the segment being split, so it terminates.
//
// The property that has to hold is about WORDS, not segment spans:
// crispasr_make_disp_segments builds every SRT/VTT cue from the word
// timestamps of a segment that has words and never reads seg.t0/seg.t1 there,
// so what must come out non-decreasing is the word stream read in segment
// order. Splitting delivers that because it moves words; clamping alone does
// not, which is why an overlap that cannot be split is merged rather than
// clamped (see the interleave branch below).
inline void crispasr_gap_fill_resolve_overlaps(std::vector<crispasr_segment>& segs) {
    const auto by_time = [](const crispasr_segment& a, const crispasr_segment& b) {
        return a.t0 != b.t0 ? a.t0 < b.t0 : a.t1 < b.t1;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        std::sort(segs.begin(), segs.end(), by_time);
        for (size_t i = 1; i < segs.size(); i++) {
            crispasr_segment& prev = segs[i - 1];
            crispasr_segment& cur = segs[i];
            if (cur.t0 >= prev.t1)
                continue;
            // prev owns a word that STARTS inside cur. The two genuinely
            // interleave, so no split of prev can put them in reading order,
            // and the clamp below would strand that word's text in a cue that
            // has already ended while leaving the word stream non-monotone —
            // #356 all over again, just invisible to a check on segment spans.
            // Reachable whenever a recovered word runs up to kEdgePadCs past
            // the next first-pass word's start, which the refill window
            // explicitly allows. Merge instead: absorb cur's words and tokens
            // into prev and re-sort by time. That is the resolution the other
            // two gap-fill implementations already use (the library's
            // parakeet_orchestrate.cpp gap_fill_segments and the session's
            // transcribe_ja_sliced both insert recovered words INTO a segment),
            // which is exactly why neither of them can produce this defect.
            // Segment count strictly decreases, so this terminates.
            //
            // Only when cur carries words: a text-only cur has no word list to
            // fold in and merging would have to invent a position for its text,
            // so that case falls through to the clamp. It needs two first-pass
            // segments to overlap each other, which gap-fill does not produce.
            bool interleaved = false;
            for (const auto& w : prev.words)
                if (w.t0 >= cur.t0 && w.t0 < cur.t1) {
                    interleaved = true;
                    break;
                }
            if (interleaved && !cur.words.empty()) {
                for (auto& w : cur.words)
                    prev.words.push_back(std::move(w));
                for (auto& t : cur.tokens)
                    prev.tokens.push_back(std::move(t));
                std::stable_sort(prev.words.begin(), prev.words.end(),
                                 [](const crispasr_word& a, const crispasr_word& b) { return a.t0 < b.t0; });
                std::stable_sort(prev.tokens.begin(), prev.tokens.end(),
                                 [](const crispasr_token& a, const crispasr_token& b) { return a.t0 < b.t0; });
                prev.text = crispasr_rebuild_text_from_words(prev.words);
                prev.t0 = prev.words.front().t0;
                prev.t1 = std::max(prev.t1, cur.t1);
                for (const auto& w : prev.words)
                    prev.t1 = std::max(prev.t1, w.t1);
                segs.erase(segs.begin() + (std::ptrdiff_t)i);
                changed = true;
                break; // re-sort and re-scan
            }
            // prev spans past cur's start. Split when prev still has words on
            // the far side of cur; otherwise just cap prev's span.
            if (!prev.words.empty() && prev.words.back().t0 >= cur.t1) {
                crispasr_segment tail = prev; // copy metadata (speaker, chunk_id, ...)
                std::vector<crispasr_word> head_words, tail_words;
                for (auto& w : prev.words)
                    (w.t0 >= cur.t1 ? tail_words : head_words).push_back(std::move(w));
                // A token with no timestamp (t0 == -1) joins no word; it rides
                // with whichever half survives rather than being dropped.
                std::vector<crispasr_token> head_tokens, tail_tokens, untimed_tokens;
                for (auto& t : prev.tokens) {
                    if (t.t0 < 0)
                        untimed_tokens.push_back(std::move(t));
                    else
                        (t.t0 >= cur.t1 ? tail_tokens : head_tokens).push_back(std::move(t));
                }
                tail.words = std::move(tail_words);
                tail.tokens = std::move(tail_tokens);
                tail.text = crispasr_rebuild_text_from_words(tail.words);
                tail.t0 = tail.words.front().t0;
                tail.t1 = tail.words.back().t1;
                if (head_words.empty()) {
                    // Nothing left of the recovery — prev *is* the tail, so the
                    // head's tokens have no other home.
                    for (auto& t : head_tokens)
                        untimed_tokens.push_back(std::move(t));
                    tail.tokens.insert(tail.tokens.begin(), std::make_move_iterator(untimed_tokens.begin()),
                                       std::make_move_iterator(untimed_tokens.end()));
                    prev = std::move(tail);
                } else {
                    prev.words = std::move(head_words);
                    prev.tokens = std::move(head_tokens);
                    prev.tokens.insert(prev.tokens.begin(), std::make_move_iterator(untimed_tokens.begin()),
                                       std::make_move_iterator(untimed_tokens.end()));
                    prev.text = crispasr_rebuild_text_from_words(prev.words);
                    prev.t0 = prev.words.front().t0;
                    prev.t1 = prev.words.back().t1;
                    // tinydiarize: the flag means "a speaker turn follows this
                    // segment". The tail is what now follows the head, so only
                    // the tail may keep it — copying it onto both halves plants
                    // a spurious turn marker mid-utterance.
                    prev.speaker_turn_next = false;
                    segs.push_back(std::move(tail));
                }
                changed = true;
                break; // re-sort and re-scan
            }
            // No words past the recovery (or no words at all): clamp. A head
            // word ending inside cur (boundary tolerance) resolves here too —
            // the interleave branch above already took every case where a whole
            // word of prev would have been stranded on the far side.
            prev.t1 = std::max(prev.t0, cur.t0);
        }
    }
}

// Whether the #89 gap-fill pass runs for this backend: it needs bounded VAD
// slices, and then follows CRISPASR_GAP_FILL or, unset, the backend's default.
// One predicate for every call site (CLI slice loop, split-pipeline gate,
// server), so they cannot disagree.
inline bool crispasr_gap_fill_active(const CrispasrBackend& be) {
    return be.vad_slice_cap_seconds() > 0 && core_gap_fill::enabled(be.gap_fill_by_default());
}

inline void crispasr_gap_fill_slice(CrispasrBackend& be, const whisper_params& params, const float* samples,
                                    int n_samples_total, int sample_rate, const crispasr_audio_slice& sl,
                                    std::vector<crispasr_segment>& segs) {
    int64_t min_gap_cs = 100; // 1.0 s of missing speech triggers a refill
    if (const char* e = getenv("CRISPASR_GAP_FILL_MIN_CS"))
        min_gap_cs = std::max((int64_t)30, (int64_t)atoi(e));
    constexpr int64_t kEdgePadCs = 20;   // extend the refill window slightly
    constexpr int64_t kCoverSlopCs = 30; // words this close bridge a gap
    const float min_conf = core_gap_fill::min_run_confidence();

    const int max_rounds = 2;
    for (int round = 0; round < max_rounds; round++) {
        // Covered intervals from emitted words (fallback: segment spans), and
        // the comparison form of every kept word for the #605 seam guard.
        std::vector<std::pair<int64_t, int64_t>> covered;
        std::vector<std::pair<int64_t, std::string>> seam;
        for (const auto& seg : segs) {
            if (seg.words.empty()) {
                if (seg.t1 > seg.t0)
                    covered.push_back({seg.t0, seg.t1});
                continue;
            }
            for (const auto& w : seg.words) {
                covered.push_back({w.t0, std::max(w.t1, w.t0 + 1)});
                seam.push_back({w.t0, core_gap_fill::norm_word(w.text)});
            }
        }
        std::sort(covered.begin(), covered.end());
        std::vector<std::pair<int64_t, int64_t>> merged;
        for (auto& iv : covered) {
            if (!merged.empty() && iv.first <= merged.back().second + kCoverSlopCs)
                merged.back().second = std::max(merged.back().second, iv.second);
            else
                merged.push_back(iv);
        }
        // Uncovered gaps within the slice.
        std::vector<std::pair<int64_t, int64_t>> gaps;
        int64_t cursor = sl.t0_cs;
        for (auto& iv : merged) {
            if (iv.first - cursor >= min_gap_cs)
                gaps.push_back({cursor, iv.first});
            cursor = std::max(cursor, iv.second);
        }
        if (sl.t1_cs - cursor >= min_gap_cs)
            gaps.push_back({cursor, sl.t1_cs});
        if (gaps.empty())
            return;

        bool recovered_any = false;
        for (auto& g : gaps) {
            const int64_t win0_cs = std::max(sl.t0_cs, g.first - kEdgePadCs);
            const int64_t win1_cs = std::min(sl.t1_cs, g.second + kEdgePadCs);
            const int s0 = std::max(0, (int)(win0_cs * sample_rate / 100));
            const int s1 = std::min(n_samples_total, (int)(win1_cs * sample_rate / 100));
            if (s1 - s0 < sample_rate / 4)
                continue;
            auto fill = be.transcribe(samples + s0, s1 - s0, win0_cs, params);
            crispasr_segment rec;
            rec.t0 = g.first;
            rec.t1 = g.second;
            // Keep only words inside the gap that don't restate content the
            // first pass (or an earlier refill) already emitted — refill
            // windows overlap covered speech at their edges and would
            // otherwise duplicate the boundary words.
            auto mid_is_covered = [&merged](int64_t mid) {
                for (const auto& iv : merged)
                    if (mid >= iv.first && mid < iv.second)
                        return true;
                return false;
            };
            for (auto& fseg : fill) {
                const size_t first_kept = rec.words.size();
                for (auto& w : fseg.words) {
                    const int64_t mid = (w.t0 + w.t1) / 2;
                    if (mid < g.first - kCoverSlopCs || mid >= g.second + kCoverSlopCs || mid_is_covered(mid))
                        continue;
                    // #605: the refill re-heard the first-pass word at the
                    // hole's edge and timed it inside the hole.
                    if (core_gap_fill::is_seam_duplicate(w.t0, core_gap_fill::norm_word(w.text), seam))
                        continue;
                    rec.words.push_back(std::move(w));
                }
                // The fill's tokens ride along with the kept words — the
                // token-level outputs (JSON full, karaoke) read seg.tokens and
                // a recovery with an empty list vanishes from them. A token
                // joins at most one word, so a shared word boundary doesn't
                // duplicate it; a token with no timestamp (t0 == -1) has no
                // word to join and stays out.
                for (auto& tk : fseg.tokens) {
                    for (size_t i = first_kept; i < rec.words.size(); i++) {
                        if (tk.t0 >= rec.words[i].t0 && tk.t0 <= rec.words[i].t1) {
                            rec.tokens.push_back(std::move(tk));
                            break;
                        }
                    }
                }
            }
            if (rec.words.empty())
                continue;
            // #605: keep the recovery only if the decoder was sure of it. A
            // refill of a pause or of game noise decodes at low confidence; a
            // real #89 drop decodes verbatim. Judged over the whole run, so one
            // hesitant token does not discard a confident recovery.
            {
                std::vector<float> conf;
                for (const auto& tk : rec.tokens)
                    if (!core_gap_fill::is_blank(tk.text))
                        conf.push_back(tk.confidence);
                if (!core_gap_fill::run_is_confident(conf, min_conf))
                    continue;
            }
            rec.text = crispasr_rebuild_text_from_words(rec.words);
            if (rec.text.empty())
                continue;
            for (const auto& w : rec.words)
                seam.push_back({w.t0, core_gap_fill::norm_word(w.text)});
            rec.t0 = rec.words.front().t0;
            rec.t1 = rec.words.back().t1;
            segs.push_back(std::move(rec));
            recovered_any = true;
        }
        if (!recovered_any)
            return;
        // Issue #356: a plain t0 sort left each recovery inside the span of
        // the segment it was recovered from — split/clamp instead so the
        // list comes out monotone.
        crispasr_gap_fill_resolve_overlaps(segs);
    }
}
