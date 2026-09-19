// src/parakeet_orchestrate.cpp — shared parakeet transcription orchestration
// (improvements Phase 1). Faithful hoist of the CLI backend adapter's
// transcribe() path selection + long-audio + segmentation into the library so
// the session C-ABI can call the SAME code. Byte-identical to the pre-hoist
// adapter path (verified by the surface-parity harness + CLI A/B).

#include "parakeet_orchestrate.h"

#include "core/asr_segment_group.h"
#include "core/sys_mem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// parakeet_result → one neutral segment (mirrors the adapter's result_to_segment).
parakeet_seg result_to_seg(const parakeet_result* r, int64_t fallback_t0_cs) {
    parakeet_seg seg;
    seg.t0 = fallback_t0_cs;
    seg.t1 = fallback_t0_cs;
    seg.text = r->text ? r->text : "";

    seg.words.reserve(r->n_words);
    for (int i = 0; i < r->n_words; i++) {
        const auto& w = r->words[i];
        parakeet_seg::word cw;
        cw.text = w.text;
        cw.t0 = w.t0;
        cw.t1 = w.t1;
        cw.p = w.p;
        seg.words.push_back(std::move(cw));
    }
    seg.tokens.reserve(r->n_tokens);
    for (int i = 0; i < r->n_tokens; i++) {
        const auto& t = r->tokens[i];
        parakeet_seg::token ct;
        ct.text = t.text;
        ct.id = t.id;
        ct.t0 = t.t0;
        ct.t1 = t.t1;
        ct.p = t.p;
        seg.tokens.push_back(std::move(ct));
    }
    if (!seg.words.empty()) {
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
    } else if (!seg.tokens.empty()) {
        seg.t0 = seg.tokens.front().t0;
        seg.t1 = seg.tokens.back().t1;
    }
    return seg;
}

// Split ONE coherent decode into ~seg_seconds segments (mirrors the adapter's
// split_result_into_segments; issue #257).
std::vector<parakeet_seg> split_result_to_segs(const parakeet_result* r, int64_t fallback_t0_cs, int seg_seconds) {
    std::vector<parakeet_seg> out;
    if (!r)
        return out;
    if (r->n_words <= 0) {
        parakeet_seg seg = result_to_seg(r, fallback_t0_cs);
        if (!seg.text.empty() || !seg.tokens.empty())
            out.push_back(std::move(seg));
        return out;
    }
    std::vector<int64_t> word_t0;
    word_t0.reserve(r->n_words);
    for (int i = 0; i < r->n_words; i++)
        word_t0.push_back(r->words[i].t0);
    const std::vector<int> starts = core_segment::group_by_window(word_t0, (int64_t)seg_seconds * 100);

    int ti = 0;
    for (size_t si = 0; si < starts.size(); si++) {
        const int w_begin = starts[si];
        const int w_end = (si + 1 < starts.size()) ? starts[si + 1] : r->n_words;
        parakeet_seg seg;
        std::string text;
        for (int wi = w_begin; wi < w_end; wi++) {
            const auto& w = r->words[wi];
            if (!text.empty())
                text += ' ';
            text += w.text;
            parakeet_seg::word cw;
            cw.text = w.text;
            cw.t0 = w.t0;
            cw.t1 = w.t1;
            cw.p = w.p;
            seg.words.push_back(std::move(cw));
        }
        seg.text = std::move(text);
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
        const int64_t next_start = (w_end < r->n_words) ? r->words[w_end].t0 : INT64_MAX;
        for (; ti < r->n_tokens && r->tokens[ti].t0 < next_start; ti++) {
            const auto& t = r->tokens[ti];
            parakeet_seg::token ct;
            ct.text = t.text;
            ct.id = t.id;
            ct.t0 = t.t0;
            ct.t1 = t.t1;
            ct.p = t.p;
            seg.tokens.push_back(std::move(ct));
        }
        out.push_back(std::move(seg));
    }
    for (; ti < r->n_tokens && !out.empty(); ti++) {
        const auto& t = r->tokens[ti];
        parakeet_seg::token ct;
        ct.text = t.text;
        ct.id = t.id;
        ct.t0 = t.t0;
        ct.t1 = t.t1;
        ct.p = t.p;
        out.back().tokens.push_back(std::move(ct));
    }
    return out;
}

// Lowest-RMS 100 ms frame in [target-window, target] (mirrors the adapter's
// find_silence_cut) — keep chunk boundaries off mid-word.
int find_silence_cut(const float* s, int n, int target, int window, int sr) {
    const int lo = std::max(1, target - window);
    const int hi = std::min(n - 1, target);
    if (hi <= lo)
        return std::min(std::max(target, 1), n);
    const int win = std::max(1, sr / 10);
    double best = 1e30;
    int best_pos = target;
    for (int c = lo; c <= hi; c += win / 2) {
        const int a = std::max(0, c - win / 2);
        const int b = std::min(n, c + win / 2);
        double e = 0.0;
        for (int i = a; i < b; i++)
            e += (double)s[i] * (double)s[i];
        e /= std::max(1, b - a);
        if (e < best) {
            best = e;
            best_pos = c;
        }
    }
    return best_pos;
}

// One silence-split longform window. The boundaries depend only on the PCM, so
// the whole plan can be computed up front — which is what lets the encoder run
// ahead of the decoder (see transcribe_longform).
struct lf_window {
    int ext_s, ext_e;          // encoder input range (window ± 2 s context)
    int end;                   // logical end sample (pre-extension) — what progress reports as processed (#385)
    int64_t ext_t0;            // absolute start of ext_s, centiseconds
    int64_t left_cs, right_cs; // keep words with left_cs <= t0 < right_cs
};

// Reproduces the original serial loop's window sequence exactly.
std::vector<lf_window> plan_longform_windows(const float* samples, int n_samples, int64_t t_offset_cs,
                                             int cap_samples) {
    std::vector<lf_window> plan;
    const int SR = 16000;
    const int search = 5 * SR;
    const int ctxs = 2 * SR;
    int pos = 0;
    while (pos < n_samples) {
        int end;
        if (n_samples - pos <= cap_samples) {
            end = n_samples;
        } else {
            end = find_silence_cut(samples, n_samples, pos + cap_samples, search, SR);
            if (end <= pos)
                end = std::min(n_samples, pos + cap_samples);
        }
        lf_window w;
        w.ext_s = std::max(0, pos - ctxs);
        w.ext_e = std::min(n_samples, end + ctxs);
        w.end = end;
        w.ext_t0 = t_offset_cs + (int64_t)((double)w.ext_s / SR * 100.0);
        w.left_cs = (pos == 0) ? INT64_MIN : t_offset_cs + (int64_t)((double)pos / SR * 100.0);
        w.right_cs = (end == n_samples) ? INT64_MAX : t_offset_cs + (int64_t)((double)end / SR * 100.0);
        plan.push_back(w);
        pos = end;
    }
    return plan;
}

// Normalize a word for boundary-dedup comparison: lowercase + drop ASCII
// punctuation. Non-ASCII bytes (JA / accented text) are kept verbatim. Mirrors
// parakeet_norm_word in the session merge path.
std::string norm_word(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c < 0x80) {
            if (std::isalnum(c))
                o += (char)std::tolower(c);
        } else {
            o += (char)c;
        }
    }
    return o;
}

// Issue #350: re-transcribe the spans the first pass left empty and merge the
// recovered words back. The TDT decoder silently drops whole sections of long
// audio — a 230 s file came back with a contiguous 45 s hole that decodes
// verbatim when handed to the same model on its own. A length cap alone cannot
// rule that out (it is not a hard threshold), so repair what the timeline shows
// is missing, whichever strategy produced it.
//
// This is the non-JA generalisation of the #89 gap-fill: same trigger (a span
// >= min_gap with no words), same guards (recovered words are kept only if they
// land in the hole and nothing already covers them), so a pass that dropped
// nothing is left EXACTLY as it was — including its text, which is only rebuilt
// when a word is actually inserted.
//
// Deliberately NOT a reuse of the two #89 implementations
// (examples/cli/crispasr_gap_fill.h's crispasr_gap_fill_slice; the
// transcribe_ja_sliced lambda in crispasr_c_api.cpp): those are JA-only in
// practice (gated on vad_slice_cap / the legacy inline path), live above this
// library layer on CLI/session segment types, and probe each gap as ONE window
// via the full backend dispatch — re-entrant from here, and wrong for holes
// longer than the reliable window. This one calls parakeet_transcribe_ex
// directly and splits long holes (parakeet_find_gaps carries the shared,
// unit-tested interval arithmetic).
//
// `min_gap_cs` is deliberately coarser than the #89 JA default (1 s): a 1-2 s
// hole in a non-JA transcript is an ordinary pause far more often than a drop,
// and handing the decoder a pause makes it invent a filler ("Um", "M"). The
// spans this defect produces are whole sentences. CRISPASR_GAP_FILL_MIN_CS
// overrides it; CRISPASR_GAP_FILL=0 turns the repair off entirely.
//
// Returns the number of words recovered.
int gap_fill_segments(parakeet_context* ctx, const float* samples, int n_samples, int64_t t_offset_cs,
                      std::vector<parakeet_seg>& segs, int repair_window_s, int64_t min_gap_cs, bool no_prints) {
    const int SR = 16000;
    if (getenv("CRISPASR_GAP_FILL") && atoi(getenv("CRISPASR_GAP_FILL")) == 0)
        return 0;
    if (const char* e = getenv("CRISPASR_GAP_FILL_MIN_CS"))
        min_gap_cs = std::max((int64_t)30, (int64_t)atoi(e));
    // The repair reasons entirely in word timestamps, and rebuilds a changed
    // segment's text from its word list. A segment carrying text but no words
    // (a decode that produced no timings) would lose that text, so leave the
    // whole result alone rather than half-repair it.
    for (const auto& s : segs)
        if (s.words.empty() && !s.text.empty())
            return 0;
    constexpr int64_t kCoverSlopCs = 30; // pauses this short stay part of one run
    constexpr int64_t kEdgePadCs = 20;   // decode a little around the hole for context
    constexpr int kMaxRounds = 2;

    const int64_t span_t0 = t_offset_cs;
    const int64_t span_t1 = t_offset_cs + (int64_t)((double)n_samples / SR * 100.0);
    const int64_t max_window_cs = (int64_t)std::max(2, repair_window_s) * 100;

    int total = 0;
    for (int round = 0; round < kMaxRounds; ++round) {
        std::vector<std::pair<int64_t, int64_t>> covered;
        std::vector<std::pair<int64_t, std::string>> seam; // (t0, normalized) of every kept word
        for (const auto& s : segs)
            for (const auto& w : s.words) {
                covered.push_back({w.t0, std::max(w.t1, w.t0 + 1)});
                seam.push_back({w.t0, norm_word(w.text)});
            }
        const auto gaps = parakeet_find_gaps(covered, span_t0, span_t1, min_gap_cs, kCoverSlopCs, max_window_cs);
        if (gaps.empty())
            break;
        // Same merged view the gap search used, to reject a recovered word that
        // duplicates one the first pass already emitted next to the hole.
        std::sort(covered.begin(), covered.end());
        std::vector<std::pair<int64_t, int64_t>> merged;
        for (auto& iv : covered) {
            if (!merged.empty() && iv.first <= merged.back().second + kCoverSlopCs)
                merged.back().second = std::max(merged.back().second, iv.second);
            else
                merged.push_back({iv.first, std::max(iv.second, iv.first + 1)});
        }

        int recovered = 0;
        std::vector<bool> dirty(segs.size(), false);
        for (const auto& g : gaps) {
            const int64_t win0_cs = std::max(span_t0, g.first - kEdgePadCs);
            const int64_t win1_cs = std::min(span_t1, g.second + kEdgePadCs);
            const int w0 = std::max(0, (int)((win0_cs - t_offset_cs) * SR / 100));
            const int w1 = std::min(n_samples, (int)((win1_cs - t_offset_cs) * SR / 100));
            if (w1 - w0 < SR / 4) // < 0.25 s is not worth a decode
                continue;
            parakeet_result* r = parakeet_transcribe_ex(ctx, samples + w0, w1 - w0, win0_cs);
            if (!r)
                continue;
            parakeet_seg got = result_to_seg(r, win0_cs);
            parakeet_result_free(r);

            for (auto& w : got.words) {
                const int64_t mid = (w.t0 + w.t1) / 2;
                if (mid < g.first - kCoverSlopCs || mid >= g.second + kCoverSlopCs)
                    continue; // context word from outside the hole
                bool dup = false;
                for (const auto& iv : merged)
                    if (mid >= iv.first && mid < iv.second) {
                        dup = true;
                        break;
                    }
                if (dup)
                    continue;
                // Seam guard: the repair window re-hears the speech either side
                // of the hole, and the decoder times a boundary word a few
                // hundred ms earlier there — which lands it INSIDE the hole and
                // doubles it ("...and And the target..."). Drop a recovered word
                // that repeats a kept word within kSeamCs. Only pairs that
                // straddle the seam are compared, so a genuine "the the" inside
                // one decode is left alone.
                constexpr int64_t kSeamCs = 50;
                bool seam_dup = false;
                for (const auto& sw : seam)
                    if (llabs(sw.first - w.t0) < kSeamCs && sw.second == norm_word(w.text)) {
                        seam_dup = true;
                        break;
                    }
                if (seam_dup)
                    continue;
                // Attach to the segment this timestamp belongs in (the last one
                // starting at or before it), so LONGFORM / CHUNK_SEGMENTED keep
                // their segmentation instead of growing a trailing blob.
                size_t si = 0;
                bool placed = false;
                for (size_t i = 0; i < segs.size(); ++i)
                    if (segs[i].t0 <= w.t0) {
                        si = i;
                        placed = true;
                    }
                if (!placed && segs.empty()) {
                    segs.push_back(parakeet_seg());
                    dirty.push_back(false);
                    si = 0;
                }
                dirty[si] = true;
                seam.push_back({w.t0, norm_word(w.text)}); // guard the next window's edge too
                std::vector<parakeet_seg::token> keep;
                for (auto& tk : got.tokens)
                    if (tk.t0 >= w.t0 && tk.t0 <= w.t1)
                        keep.push_back(tk);
                for (auto& tk : keep)
                    segs[si].tokens.push_back(std::move(tk));
                segs[si].words.push_back(std::move(w));
                ++recovered;
            }
        }
        if (recovered == 0)
            break;
        total += recovered;
        if (!no_prints)
            fprintf(stderr, "crispasr[parakeet]: gap-fill recovered %d word(s) the first pass dropped\n", recovered);
        // Re-sort and rebuild only the segments a word was inserted into. Text
        // is regenerated from the word list, so a segment nothing landed in
        // keeps the decoder's own detokenized string byte for byte.
        for (size_t i = 0; i < segs.size(); ++i) {
            if (!dirty[i])
                continue;
            parakeet_seg& s = segs[i];
            std::stable_sort(s.words.begin(), s.words.end(),
                             [](const parakeet_seg::word& a, const parakeet_seg::word& b) { return a.t0 < b.t0; });
            std::stable_sort(s.tokens.begin(), s.tokens.end(),
                             [](const parakeet_seg::token& a, const parakeet_seg::token& b) { return a.t0 < b.t0; });
            std::string text;
            for (const auto& w : s.words) {
                if (w.text.empty())
                    continue;
                if (!text.empty()) {
                    const unsigned char prev_last = (unsigned char)text.back();
                    const unsigned char cur_first = (unsigned char)w.text[0];
                    // CJK boundaries take no space (>= 0xE0 lead byte = 3-byte
                    // UTF-8); latin words carry a leading space from the
                    // tokenizer already.
                    if (cur_first != ' ' && prev_last != ' ' && prev_last < 0xE0 && cur_first < 0xE0)
                        text += ' ';
                }
                text += w.text;
            }
            if (!text.empty() && text[0] == ' ')
                text.erase(0, 1);
            s.text = std::move(text);
            if (!s.words.empty()) {
                s.t0 = s.words.front().t0;
                s.t1 = s.words.back().t1;
            }
        }
    }
    return total;
}

// Trim one window's decode to its non-overlapping span and append it. Shared by
// the serial and pipelined paths so both produce byte-identical segments.
// Consumes `r` (frees it).
void append_window_seg(parakeet_result* r, const lf_window& w, std::vector<parakeet_seg>& out) {
    if (!r)
        return;
    parakeet_seg full = result_to_seg(r, w.ext_t0);
    parakeet_result_free(r);

    parakeet_seg seg;
    seg.t0 = w.left_cs == INT64_MIN ? full.t0 : w.left_cs;
    seg.t1 = seg.t0;
    std::string text;
    for (auto& word : full.words) {
        if (word.t0 >= w.left_cs && word.t0 < w.right_cs) {
            if (!text.empty())
                text += ' ';
            text += word.text;
            seg.words.push_back(std::move(word));
        }
    }
    for (auto& tk : full.tokens) {
        if (tk.t0 >= w.left_cs && tk.t0 < w.right_cs)
            seg.tokens.push_back(std::move(tk));
    }
    seg.text = std::move(text);
    if (!seg.words.empty()) {
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
    }
    if (!seg.text.empty() || !seg.words.empty())
        out.push_back(std::move(seg));
}

// Silence-split single-pass longform (mirrors the adapter's transcribe_longform).
//
// The windows are independent: each is encoded from raw PCM and the merge is a
// pure timestamp filter. The encoder runs on ctx->backend (GPU) and the TDT
// decoder on the CPU (cblas), so running them strictly in sequence left one of
// the two idle at all times — measured 0.42 cores on an 11.2 min file, with the
// stage sum equal to wall time (encoder 67 %, decode 32 %).
//
// So: a producer thread encodes window N+1 while this thread decodes window N.
// Order is preserved (single producer, single consumer, FIFO), and the queue is
// bounded so at most two encoder buffers are resident (~4.5 MB each at 90 s).
//
// Safety: the two threads touch DISJOINT parakeet_context state — the encoder
// owns sched/compute_meta/cached_enc_*, the decoder owns pred_w/joint_w and
// reads the model. That stops being true when the decoder is itself a ggml
// graph on ctx->backend (CUDA/Vulkan default), so pipelining is disabled when
// parakeet_decode_uses_backend() says so. CRISPASR_PARAKEET_PIPELINE=0/1
// forces it off/on.
std::vector<parakeet_seg> transcribe_longform(parakeet_context* ctx, const float* samples, int n_samples,
                                              int64_t t_offset_cs, int cap_samples,
                                              const parakeet_orchestrate_opts& opts) {
    std::vector<parakeet_seg> out;
    const std::vector<lf_window> plan = plan_longform_windows(samples, n_samples, t_offset_cs, cap_samples);
    if (plan.empty())
        return out;

    // Issue #385: report each finished window. `w.end` is the window's logical
    // end sample, so the sequence is monotonic and the last fires (n, n).
    auto report = [&](const lf_window& w) {
        if (opts.on_progress)
            opts.on_progress(w.end, n_samples);
    };

    bool pipeline = plan.size() > 1 && parakeet_decode_uses_backend(ctx) == 0;
    if (const char* e = getenv("CRISPASR_PARAKEET_PIPELINE"))
        pipeline = atoi(e) != 0 && plan.size() > 1;

    if (!pipeline) {
        for (const auto& w : plan) {
            append_window_seg(parakeet_transcribe_ex(ctx, samples + w.ext_s, w.ext_e - w.ext_s, w.ext_t0), w, out);
            report(w);
        }
        return out;
    }

    struct enc_item {
        float* buf = nullptr; // malloc'd by parakeet_encode; nullptr = encode failed
        int T_enc = 0;
        int d_model = 0;
    };

    std::deque<enc_item> q;
    std::mutex m;
    std::condition_variable cv_full, cv_empty;
    size_t produced = 0;
    const size_t kQueueCap = 2;

    std::thread producer([&] {
        for (const auto& w : plan) {
            enc_item it;
            it.buf = parakeet_encode(ctx, samples + w.ext_s, w.ext_e - w.ext_s, &it.T_enc, &it.d_model);
            std::unique_lock<std::mutex> lk(m);
            cv_full.wait(lk, [&] { return q.size() < kQueueCap; });
            q.push_back(it);
            ++produced;
            cv_empty.notify_one();
        }
    });

    for (size_t i = 0; i < plan.size(); ++i) {
        enc_item it;
        {
            std::unique_lock<std::mutex> lk(m);
            cv_empty.wait(lk, [&] { return !q.empty(); });
            it = q.front();
            q.pop_front();
            cv_full.notify_one();
        }
        if (it.buf) {
            append_window_seg(parakeet_decode_frames(ctx, it.buf, it.T_enc, it.d_model, plan[i].ext_t0), plan[i], out);
            free(it.buf);
        } // else: encode failed for this window; keep going, order intact
        report(plan[i]);
    }

    producer.join();
    return out;
}

struct resolved_strategy {
    parakeet_strategy strat = parakeet_strategy::SINGLE_PASS;
    int stream_threshold_s = 300;                     // single-pass cap
    int longform_window_s = kParakeetLongformWindowS; // LONGFORM piece size
    int stream_chunk_s = 0;
    int stream_overlap_s = 2;
};

// Single source of truth for "which long-audio route does this input take", and
// with what window. Split out of parakeet_transcribe_segments() so the cap and
// the LONGFORM window can be reasoned about (and overridden) independently.
// `quiet` suppresses the memory-policy notice for predicate-only callers.
resolved_strategy resolve_strategy(parakeet_context* ctx, int n_samples, bool is_ja,
                                   const parakeet_orchestrate_opts& opts, bool quiet) {
    const int SR = 16000;
    resolved_strategy rs;
    // Single-pass cap. JA keeps its own 12 s cap (issue #89 — the JA models
    // degenerate past their trained window); non-JA keeps the historical 300 s
    // so mid-length audio stays on the seamless single-pass path.
    rs.stream_threshold_s = is_ja ? kParakeetBoundedWindowJaS : 300;
    bool longform_enabled = !is_ja;
    bool threshold_from_env = false;
    if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_THRESHOLD")) {
        rs.stream_threshold_s = std::max(0, atoi(e));
        threshold_from_env = true;
    }
    if (const char* e = getenv("CRISPASR_PARAKEET_LONGFORM"))
        longform_enabled = atoi(e) != 0;
    if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
        rs.stream_chunk_s = std::max(2, atoi(e));
    if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_OVERLAP"))
        rs.stream_overlap_s = std::max(0, atoi(e));

    parakeet_strategy_in sin;
    sin.n_samples = n_samples;
    sin.sample_rate = SR;
    sin.is_ja = is_ja;
    sin.chunk_seconds_explicit = opts.chunk_seconds_explicit;
    sin.chunk_seconds = opts.chunk_seconds;
    sin.stream_threshold_s = rs.stream_threshold_s;
    sin.longform_enabled = longform_enabled;
    sin.chunked_requested = opts.chunked_requested;
    // Issue #350: a chunked-entry-point caller that left the length to the
    // per-model default gets the bounded cap, not the 300 s single-pass one.
    rs.stream_threshold_s = parakeet_effective_single_pass_cap_s(sin, threshold_from_env);
    sin.stream_threshold_s = rs.stream_threshold_s;

    // LONGFORM window — independent of the cap (see kParakeetLongformWindowS).
    // Clamped to the effective cap so a lowered CRISPASR_PARAKEET_STREAM_THRESHOLD
    // still shrinks the windows exactly as it did before the two were split.
    if (const char* e = getenv("CRISPASR_PARAKEET_LONGFORM_WINDOW"))
        rs.longform_window_s = std::max(4, atoi(e));
    if (rs.stream_threshold_s > 0)
        rs.longform_window_s = std::min(rs.longform_window_s, rs.stream_threshold_s);

    rs.strat = parakeet_pick_strategy(sin);

    // Phase 2: proactive encoder memory policy. When single-pass is chosen but
    // its estimated O(T^2) rel-pos bias would exceed a user-set VRAM budget,
    // switch to the streamed (bounded-window) encoder BEFORE allocating —
    // instead of allocate → OOM → reactive fallback. The allocation-site hard
    // guard in parakeet.cpp remains active even when this routing policy is off.
    //   CRISPASR_PARAKEET_MEM_POLICY = auto (default) | off | streamed
    //   CRISPASR_PARAKEET_VRAM_BUDGET_MB : budget (MiB); explicit 0 disables routing policy
    //   CRISPASR_PARAKEET_MEM_COEFF : O(T^2) estimate coefficient (default 8.0)
    if (rs.strat == parakeet_strategy::SINGLE_PASS) {
        const char* pol = getenv("CRISPASR_PARAKEET_MEM_POLICY");
        const bool mode_off = pol && strcmp(pol, "off") == 0;
        const bool mode_force_streamed = pol && strcmp(pol, "streamed") == 0;
        if (mode_force_streamed) {
            rs.strat = parakeet_strategy::STREAMED;
        } else if (!mode_off) {
            double budget = 0.0, coeff = 8.0;
            // #441: an UNSET budget used to mean "policy disabled", which left a
            // ~123 GB single-pass allocation to be attempted on a 15 GB machine.
            // Linux overcommit granted it about half the time (RSS stayed ~1 GB,
            // because the buffer is never fully written) and refused it the rest,
            // and the refusal was dereferenced — so the symptom was an
            // INTERMITTENT SIGSEGV rather than a diagnosable error. The estimate
            // and the switch to STREAMED were both already implemented; only the
            // budget was missing. Default it to what the machine can give.
            //
            // Half of MemAvailable, because the encoder bias is not the only live
            // allocation — weights, activations and the decoder share the pool.
            // An EXPLICIT 0 still disables the policy, so that escape hatch
            // survives; only "unset" changes meaning.
            const char* bud_env = getenv("CRISPASR_PARAKEET_VRAM_BUDGET_MB");
            if (bud_env) {
                budget = atof(bud_env);
            } else {
                const double avail = core_sys_mem::available_mb();
                // -1 means "could not determine". Treating that as 0 would turn
                // an unreadable /proc into a policy that refuses every input.
                if (avail > 0.0)
                    budget = avail * 0.5;
            }
            if (const char* e = getenv("CRISPASR_PARAKEET_MEM_COEFF"))
                coeff = atof(e);
            const int T_enc = parakeet_est_enc_frames(ctx, n_samples);
            const int H = parakeet_n_heads(ctx);
            if (!parakeet_singlepass_fits_budget(T_enc, H, budget, coeff)) {
                if (!opts.no_prints && !quiet)
                    fprintf(stderr,
                            "crispasr[parakeet]: single-pass est %.0f MiB > budget %.0f MiB (T=%d, H=%d); "
                            "using streamed encoding (MEM_POLICY=off disables proactive routing; the physical-memory "
                            "guard remains active)\n",
                            parakeet_est_singlepass_peak_mb(T_enc, H, coeff), budget, T_enc, H);
                rs.strat = parakeet_strategy::STREAMED;
            }
        }
    }
    return rs;
}

} // namespace

int parakeet_repair_segments(parakeet_context* ctx, const float* samples, int n_samples, int64_t t_offset_cs,
                             std::vector<parakeet_seg>& segs, bool no_prints) {
    if (!ctx || !samples || n_samples <= 0)
        return 0;
    // Same constants the SINGLE_PASS branch of parakeet_transcribe_segments()
    // passes, so a split pass repairs identically to a whole one.
    return gap_fill_segments(ctx, samples, n_samples, t_offset_cs, segs, kParakeetBoundedWindowS, kParakeetGapFillMinCs,
                             no_prints);
}

bool parakeet_slice_is_single_pass(parakeet_context* ctx, int n_samples, bool is_ja,
                                   const parakeet_orchestrate_opts& opts) {
    if (!ctx || n_samples <= 0)
        return false;
    // The simulated-OOM hook makes single-pass fail on purpose; the split path
    // has no streamed retry of its own, so leave those runs on transcribe().
    if (getenv("CRISPASR_PARAKEET_SIMULATE_ENCODE_OOM"))
        return false;
    return resolve_strategy(ctx, n_samples, is_ja, opts, /*quiet=*/true).strat == parakeet_strategy::SINGLE_PASS;
}

std::vector<parakeet_seg> parakeet_decode_frames_to_segments(parakeet_context* ctx, const float* enc_frames, int T_enc,
                                                             int d_model, int64_t t_offset_cs) {
    std::vector<parakeet_seg> out;
    if (!ctx || !enc_frames || T_enc <= 0)
        return out;
    parakeet_result* r = parakeet_decode_frames(ctx, enc_frames, T_enc, d_model, t_offset_cs);
    if (!r)
        return out;
    // Same tail as the SINGLE_PASS branch of parakeet_transcribe_segments().
    out.push_back(result_to_seg(r, t_offset_cs));
    parakeet_result_free(r);
    return out;
}

// Issue #385: bridge the C progress hook of parakeet_transcribe_streamed_progress
// to opts.on_progress, for the routes that are ONE decode over a
// chunk-encoded input (CHUNK_SEGMENTED / STREAMED / the single-pass OOM
// fallback). Those have observable ENCODER windows even though they have only
// one decode, so #385 does not have to leave them silent.
//
// The terminal tick is suppressed here on purpose. The encoder's last window
// means "encoding done", but the single TDT decode and the #350 gap-fill both
// still run after it, and they are not the short part — a bar that reached
// 100 % there would read "finished" through the longest remaining step. Each
// call site re-emits (n_samples, n_samples) itself once the work really is
// done, so the sequence still ends at `total` exactly as the session header
// documents.
namespace {
struct enc_progress_bridge {
    const parakeet_orchestrate_opts* opts;
    static void thunk(int processed, int total, void* ud) {
        auto* b = static_cast<enc_progress_bridge*>(ud);
        if (b->opts->on_progress && processed < total)
            b->opts->on_progress(processed, total);
    }
};
} // namespace

std::vector<parakeet_seg> parakeet_transcribe_segments(parakeet_context* ctx, const float* samples, int n_samples,
                                                       int64_t t_offset_cs, bool is_ja,
                                                       const parakeet_orchestrate_opts& opts) {
    std::vector<parakeet_seg> out;
    if (!ctx || !samples || n_samples <= 0)
        return out;

    const int SR = 16000;

    // Resolve EVERY route before dispatch. #441's first fix guarded only the
    // SINGLE_PASS branch below an early explicit-chunk return, which meant its
    // unit test did not exercise the reporter's actual command.
    const resolved_strategy rs = resolve_strategy(ctx, n_samples, is_ja, opts, /*quiet=*/false);
    const parakeet_strategy strat = rs.strat;

    // Issue #257: explicit --chunk-seconds (non-JA) → coherent quality-window
    // decode grouped into ~N-second segments (see parakeet_orchestrate.h).
    if (strat == parakeet_strategy::CHUNK_SEGMENTED) {
        const int seg_seconds = std::max(2, opts.chunk_seconds);
        int enc_window = 0;
        if (const char* e = getenv("CRISPASR_PARAKEET_STREAM_CHUNK"))
            enc_window = std::max(2, atoi(e));
        const int ov = std::max(0, (int)(opts.chunk_overlap_seconds + 0.5f));
        if (!opts.no_prints)
            fprintf(stderr,
                    "crispasr[parakeet]: route=chunk-segmented output-window=%ds encoder-window=%ds "
                    "(bounded encoder allocation)\n",
                    seg_seconds, enc_window > 0 ? enc_window : 30);
        enc_progress_bridge bridge{&opts};
        parakeet_result* rc =
            parakeet_transcribe_streamed_progress(ctx, samples, n_samples, t_offset_cs, enc_window, ov,
                                                  opts.on_progress ? &enc_progress_bridge::thunk : nullptr, &bridge);
        if (!rc)
            return out;
        out = split_result_to_segs(rc, t_offset_cs, seg_seconds);
        parakeet_result_free(rc);
        // Issue #350: CHUNK_SEGMENTED bounds the ENCODER but still runs one TDT
        // decode over the concatenation, so it drops spans like any other single
        // decode (measured 88 % coverage where per-slice paths reach 93-94 %).
        gap_fill_segments(ctx, samples, n_samples, t_offset_cs, out, kParakeetBoundedWindowS, kParakeetGapFillMinCs,
                          opts.no_prints);
        if (opts.on_progress)
            opts.on_progress(n_samples, n_samples); // #385: 100 % once the decode + repair are done
        return out;
    }

    const int stream_chunk_s = rs.stream_chunk_s;
    const int stream_overlap_s = rs.stream_overlap_s;

    // Issue #350: JA is left alone — it has its own #89 machinery (VAD/energy
    // slices capped at 12 s plus a 1 s-threshold gap-fill) on the paths that
    // drive it, and none of that issue's measurements cover it.
    const bool repair = !is_ja;

    // The LONGFORM piece size is the WINDOW, not the single-pass cap. The two
    // are independent: the window is a throughput knob, gap_fill_segments is
    // what makes coverage robust.
    if (strat == parakeet_strategy::LONGFORM) {
        out = transcribe_longform(ctx, samples, n_samples, t_offset_cs, rs.longform_window_s * SR, opts);
        if (repair)
            gap_fill_segments(ctx, samples, n_samples, t_offset_cs, out, kParakeetBoundedWindowS, kParakeetGapFillMinCs,
                              opts.no_prints);
        return out;
    }

    parakeet_result* r = nullptr;
    enc_progress_bridge bridge{&opts};
    if (strat == parakeet_strategy::SINGLE_PASS) {
        // Issue #257: single-pass full attention is O(T^2); a VRAM-limited GPU
        // can fail the encode alloc → null → empty transcript. Fall back to the
        // streamed (bounded-window) encoder. Simulate with
        // CRISPASR_PARAKEET_SIMULATE_ENCODE_OOM=1.
        const bool simulate_oom = getenv("CRISPASR_PARAKEET_SIMULATE_ENCODE_OOM") != nullptr;
        r = simulate_oom ? nullptr : parakeet_transcribe_ex(ctx, samples, n_samples, t_offset_cs);
        if (!r) {
            if (!opts.no_prints)
                fprintf(stderr,
                        "crispasr[parakeet]: single-pass encode failed (likely VRAM OOM at %.0fs); "
                        "falling back to streamed encoding — pass --chunk-seconds N for segmented "
                        "output or --att-context L,R for bounded-memory single-pass\n",
                        (double)n_samples / SR);
            r = parakeet_transcribe_streamed_progress(
                ctx, samples, n_samples, t_offset_cs, stream_chunk_s, stream_overlap_s,
                opts.on_progress ? &enc_progress_bridge::thunk : nullptr, &bridge);
        }
    } else { // STREAMED
        r = parakeet_transcribe_streamed_progress(ctx, samples, n_samples, t_offset_cs, stream_chunk_s,
                                                  stream_overlap_s,
                                                  opts.on_progress ? &enc_progress_bridge::thunk : nullptr, &bridge);
    }
    if (!r)
        return out;
    out.push_back(result_to_seg(r, t_offset_cs));
    parakeet_result_free(r);
    if (repair)
        gap_fill_segments(ctx, samples, n_samples, t_offset_cs, out, kParakeetBoundedWindowS, kParakeetGapFillMinCs,
                          opts.no_prints);
    // #385: SINGLE_PASS has no windows and stays silent (nothing fired above, so
    // firing only the terminal tick here would be a lone 100 % out of nowhere);
    // the streamed routes did report, so they get their real end.
    if (opts.on_progress && strat != parakeet_strategy::SINGLE_PASS)
        opts.on_progress(n_samples, n_samples);
    return out;
}
