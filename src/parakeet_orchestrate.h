// src/parakeet_orchestrate.h — shared parakeet transcription orchestration
// (improvements Phase 1: collapse the CLI-adapter vs session-C-ABI dual dispatch).
//
// The path-selection + long-audio + segmentation logic that decides HOW to run
// parakeet on a buffer used to be written twice — once in the CLI backend
// adapter (examples/cli/crispasr_backend_parakeet.cpp) and once inline in the
// session C-ABI (src/crispasr_c_api.cpp). That divergence is why issue #257 (and
// the JA-detection fix before it) had to be applied in multiple places.
//
// This header hoists that orchestration into the library so BOTH surfaces call
// one implementation. Callers set sticky per-call state on the context first
// (temperature/beam/att-context/hotwords/ctc via the existing setters), then
// call parakeet_transcribe_segments() and convert the neutral `parakeet_seg`
// list to their own segment type. The routing DECISION is a pure function
// (parakeet_pick_strategy) so it is unit-tested without a model.
#pragma once

#include "parakeet.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Neutral, surface-agnostic segment (subset both crispasr_segment and
// crispasr_session_seg can be built from).
struct parakeet_seg {
    struct word {
        std::string text;
        int64_t t0 = 0; // centiseconds, absolute
        int64_t t1 = 0;
        float p = 1.0f;
    };
    struct token {
        std::string text;
        int32_t id = -1;
        int64_t t0 = -1;
        int64_t t1 = -1;
        float p = -1.0f;
    };
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
    std::vector<word> words;
    std::vector<token> tokens;
};

struct parakeet_orchestrate_opts {
    bool chunk_seconds_explicit = false;
    int chunk_seconds = 0;
    float chunk_overlap_seconds = 2.0f;
    bool no_prints = false;
    // Issue #350: the caller came through a chunked long-form entry point
    // (crispasr_session_transcribe_chunked[_lang]) but left the length to the
    // per-model default (`chunk_seconds == 0`, documented as "use per-model
    // defaults"). That is an explicit request for BOUNDED long-form, so it must
    // never collapse to one unbounded full-length pass — see
    // parakeet_effective_single_pass_cap_s.
    bool chunked_requested = false;
    // Issue #385: per-window progress on the LONGFORM route — the #208 session
    // contract, which the #350 hoist left behind on the legacy inline path.
    // Invoked on the calling thread after each finished window with (input
    // samples processed so far, total samples); `processed` is monotonically
    // non-decreasing and reaches `total` on the last window, including a
    // window whose encode failed, so a caller's progress bar cannot stall
    // short of 100 %. The single-decode routes (SINGLE_PASS / STREAMED /
    // CHUNK_SEGMENTED) have no observable windows and do not invoke it, per
    // the session header's documented contract; the #350 gap-fill repair runs
    // after the last window and stays silent too.
    std::function<void(int processed, int total)> on_progress;
};

// The decoder's reliable single-pass window, in seconds. Past roughly this much
// audio the TDT decoder loses track and silently drops whole spans of speech
// (LEARNINGS §"Parakeet single-pass DROPS whole sections of long audio"; issue
// #350 measured a contiguous 45 s hole in a 230 s file). Used as the cap for
// callers that asked for bounded long-form, and as the width of a gap-fill
// repair window.
inline constexpr int kParakeetBoundedWindowS = 30;
inline constexpr int kParakeetBoundedWindowJaS = 12;

// Smallest hole in the word timeline worth re-transcribing (centiseconds).
// Coarser than the #89 JA gap-fill's 1 s: a 1-2 s hole in a non-JA transcript is
// an ordinary pause far more often than a dropped section, and re-decoding a
// pause makes the model invent a filler word.
inline constexpr int kParakeetGapFillMinCs = 300;

// Long-audio / chunking strategy. Pure decision, unit-tested.
enum class parakeet_strategy {
    CHUNK_SEGMENTED, // explicit --chunk-seconds (non-JA): quality encode → N-second segments
    LONGFORM,        // non-JA above the single-pass cap: silence-split single-pass pieces
    SINGLE_PASS,     // fits the single-pass cap: one full-attention pass (+ OOM fallback)
    STREAMED,        // JA / cap disabled: overlapping streamed encoder, one segment
};

// Inputs mirror the adapter/session env-tunable knobs so the choice is
// reproducible and testable. `is_ja` is content-based JA detection (kana/kanji).
struct parakeet_strategy_in {
    int n_samples = 0;
    int sample_rate = 16000;
    bool is_ja = false;
    bool chunk_seconds_explicit = false;
    int chunk_seconds = 0;
    int stream_threshold_s = 300;   // single-pass cap (s); 0 = always streamed
    bool longform_enabled = true;   // silence-split above the cap
    bool chunked_requested = false; // caller asked for bounded long-form (#350)
};

// Issue #350: the cap that actually applies to this call. A caller that came
// through the chunked entry point asked for bounded long-form; honouring that
// with the 300 s default cap means a 30-300 s file takes ONE full-length pass —
// too long for the decoder to hold (spans vanish), too short to reach LONGFORM.
// Such calls are capped at the reliable window instead, so anything longer is
// sliced. An explicit CRISPASR_PARAKEET_STREAM_THRESHOLD still wins (it is
// passed in as `stream_threshold_s` with `threshold_from_env` set), as does an
// explicit chunk length (that routes to CHUNK_SEGMENTED before the cap is read).
inline int parakeet_effective_single_pass_cap_s(const parakeet_strategy_in& in, bool threshold_from_env) {
    if (threshold_from_env || !in.chunked_requested || in.chunk_seconds_explicit)
        return in.stream_threshold_s;
    const int bounded = in.is_ja ? kParakeetBoundedWindowJaS : kParakeetBoundedWindowS;
    return in.stream_threshold_s > 0 ? std::min(in.stream_threshold_s, bounded) : in.stream_threshold_s;
}

// Default LONGFORM window (s) — the size of each silence-split piece once audio
// exceeds the effective single-pass cap. DISTINCT from that cap: it only affects
// audio that was going to be split anyway, so lowering it costs no seamless
// single-pass coverage. Effective window is min(this, cap), which keeps
// CRISPASR_PARAKEET_STREAM_THRESHOLD=N behaving as before for N <= 90.
//
// The two used to be one number and could not be tuned independently: lowering
// it to get cheaper LONGFORM windows also pushed mid-length audio off the
// seamless single-pass path (127 s file: 0.91 % WER as one pass vs 1.21 % once
// split in two). Hence the split — the cap stays 300.
//
// This is a THROUGHPUT knob only. The FastConformer encoder materialises an
// O(T^2) relative-position bias, so covering the same audio in k windows costs
// ~T^2/k instead of T^2. Smaller windows are strictly less encoder work and the
// direction is hardware-independent — measured 300 -> 90: 1.85x on Metal, 1.53x
// on CPU (-ng), same M1 Pro. It saturates around 60-90 s where the linear
// conv/FFN terms take over, so 90 sits at the knee.
//
// It is deliberately NOT a coverage knob. Window size used to change how much
// speech went missing (a 300 s window dropped 233 of 2175 words on one 12 min
// file while being word-exact on another), which made the number look
// load-bearing. Issue #350's gap_fill_segments repairs dropped spans whichever
// strategy produced them, and with it in place WER is flat across window sizes —
// two ~12 min LibriSpeech concatenations, M1 Pro / parakeet-tdt-0.6b-v3 Q4_K /
// Metal:
//          300 s    150 s    90 s     60 s
//   A      1.87 %   1.81 %   1.87 %   1.93 %
//   B      1.24 %   1.56 %   1.38 %   1.24 %
// Non-monotonic and inside run-to-run noise. So tune this for speed and memory;
// coverage is gap_fill_segments' job.
// Override with CRISPASR_PARAKEET_LONGFORM_WINDOW.
constexpr int kParakeetLongformWindowS = 90;

// Pure routing decision — no model, no side effects. Mirrors the adapter:
//   - non-JA + explicit --chunk-seconds>0            → CHUNK_SEGMENTED
//   - longform on + threshold>0 + n > threshold       → LONGFORM
//   - threshold>0 + n <= threshold                    → SINGLE_PASS
//   - else                                            → STREAMED
// `stream_threshold_s` is expected to already be the EFFECTIVE cap
// (parakeet_effective_single_pass_cap_s).
inline parakeet_strategy parakeet_pick_strategy(const parakeet_strategy_in& in) {
    if (!in.is_ja && in.chunk_seconds_explicit && in.chunk_seconds > 0)
        return parakeet_strategy::CHUNK_SEGMENTED;
    const long long cap = (long long)in.stream_threshold_s * in.sample_rate;
    if (in.longform_enabled && in.stream_threshold_s > 0 && (long long)in.n_samples > cap)
        return parakeet_strategy::LONGFORM;
    if (in.stream_threshold_s > 0 && (long long)in.n_samples <= cap)
        return parakeet_strategy::SINGLE_PASS;
    return parakeet_strategy::STREAMED;
}

// ---- Issue #350: gap-fill repair ----
//
// The failure mode a length cap cannot rule out: the TDT decoder emits NOTHING
// for a contiguous span it did hear (the same span decodes verbatim in
// isolation). It shows up as a hole in the word timeline. Finding those holes is
// pure interval arithmetic, so it lives here and is unit-tested without a model.
//
// `covered` is the (t0, t1) centisecond span of every word emitted, in any
// order. Words closer together than `slop_cs` count as one covered run — natural
// pauses inside speech must not read as drops. Returns the spans of at least
// `min_gap_cs` inside [span_t0_cs, span_t1_cs) that no word covers, each split
// into pieces no longer than `max_window_cs` so a repair decode stays inside the
// decoder's reliable window.
inline std::vector<std::pair<int64_t, int64_t>> parakeet_find_gaps(std::vector<std::pair<int64_t, int64_t>> covered,
                                                                   int64_t span_t0_cs, int64_t span_t1_cs,
                                                                   int64_t min_gap_cs, int64_t slop_cs,
                                                                   int64_t max_window_cs) {
    std::vector<std::pair<int64_t, int64_t>> gaps;
    if (span_t1_cs <= span_t0_cs || min_gap_cs <= 0)
        return gaps;
    std::sort(covered.begin(), covered.end());
    std::vector<std::pair<int64_t, int64_t>> merged;
    for (auto& iv : covered) {
        const int64_t a = iv.first, b = std::max(iv.second, iv.first + 1);
        if (!merged.empty() && a <= merged.back().second + slop_cs)
            merged.back().second = std::max(merged.back().second, b);
        else
            merged.push_back({a, b});
    }
    auto push = [&](int64_t a, int64_t b) {
        a = std::max(a, span_t0_cs);
        b = std::min(b, span_t1_cs);
        if (b - a < min_gap_cs)
            return;
        if (max_window_cs <= 0) {
            gaps.push_back({a, b});
            return;
        }
        // Split a long hole into reliable-window pieces rather than handing the
        // decoder back the same too-long input that dropped it.
        const int64_t n = (b - a + max_window_cs - 1) / max_window_cs;
        const int64_t step = (b - a + n - 1) / n;
        for (int64_t p = a; p < b; p += step)
            gaps.push_back({p, std::min(b, p + step)});
    };
    int64_t cursor = span_t0_cs;
    for (auto& iv : merged) {
        if (iv.first > cursor)
            push(cursor, iv.first);
        cursor = std::max(cursor, iv.second);
    }
    push(cursor, span_t1_cs);
    return gaps;
}

// ---- Phase 2: proactive encoder memory policy ----
//
// Single-pass full attention materializes an O(T^2) relative-position bias
// (several T×(2T) tensors live at once), which is what OOM'd the reporter's
// 3.7 GiB card at ~4 min (issue #257). Rather than allocate-and-fail-and-retry
// (the reactive fallback), estimate the peak up front and proactively pick the
// streamed (bounded-window) encoder when it would exceed a user-set budget.
//
// `coeff` folds the "how many T×2T F32 tensors coexist" constant; the default
// (8.0) is calibrated so a ~4 min clip (T≈2800, 8 heads) estimates ~2 GiB,
// matching the reporter's 1911 MiB, and is deliberately a slight OVER-estimate
// so the gate errs toward the safe streamed path. Env-tunable via
// CRISPASR_PARAKEET_MEM_COEFF. This is a heuristic gate, NOT an exact allocator
// model — the reactive OOM fallback still backstops a wrong estimate.
inline double parakeet_est_singlepass_peak_mb(int T_enc, int n_heads, double coeff) {
    if (T_enc <= 0 || n_heads <= 0)
        return 0.0;
    const double T = (double)T_enc;
    return coeff * T * T * (double)n_heads * 4.0 / (1024.0 * 1024.0);
}

// Does single-pass fit `budget_mb`? A non-positive budget means "no budget set"
// → always fits (policy disabled, historical behaviour). A non-positive coeff
// disables the estimate → always fits.
inline bool parakeet_singlepass_fits_budget(int T_enc, int n_heads, double budget_mb, double coeff) {
    if (budget_mb <= 0.0 || coeff <= 0.0)
        return true;
    return parakeet_est_singlepass_peak_mb(T_enc, n_heads, coeff) <= budget_mb;
}

// Full orchestration: mel → path selection → decode → segmentation, returning
// the neutral segment list. `is_ja` is passed in (callers already detect it, or
// pass parakeet_vocab_is_japanese(ctx)). Reads the same CRISPASR_PARAKEET_*
// env knobs the adapter did, so behaviour is byte-identical to the pre-hoist
// adapter path.
// True when parakeet_transcribe_segments() would take the plain SINGLE_PASS
// route for this input — exactly one encode + one decode, so the caller may
// substitute parakeet_encode() + parakeet_decode_frames_to_segments() and get
// an identical result. False for LONGFORM / STREAMED / CHUNK_SEGMENTED inputs,
// which do their own multi-window merging and must go through transcribe().
bool parakeet_slice_is_single_pass(struct parakeet_context* ctx, int n_samples, bool is_ja,
                                   const parakeet_orchestrate_opts& opts);

// Second half of the SINGLE_PASS path: decode already-encoded frames into the
// neutral segment shape. Pairs with parakeet_encode() so a caller processing
// many short slices can overlap the encode of slice N+1 (GPU) with the decode
// of slice N (CPU). Produces exactly what parakeet_transcribe_segments() would
// have returned for a slice that took SINGLE_PASS, so the two are
// interchangeable. Does NOT free `enc_frames`.
// The post-decode span repair that parakeet_transcribe_segments() applies at the
// end of its SINGLE_PASS branch (issue #350: the TDT decoder silently drops
// whole sections of audio). Exposed because a caller that split that branch into
// parakeet_encode() + parakeet_decode_frames_to_segments() would otherwise lose
// it — the repair is not part of the decode, it is the tail of the pass.
//
// ⚠ It RE-ENCODES the gaps it finds, so it must run single-threaded: never
// inside a decode that is overlapped with an encode on another thread. Returns
// the number of words recovered (0 = `segs` untouched).
int parakeet_repair_segments(struct parakeet_context* ctx, const float* samples, int n_samples, int64_t t_offset_cs,
                             std::vector<parakeet_seg>& segs, bool no_prints);

std::vector<parakeet_seg> parakeet_decode_frames_to_segments(struct parakeet_context* ctx, const float* enc_frames,
                                                             int T_enc, int d_model, int64_t t_offset_cs);

std::vector<parakeet_seg> parakeet_transcribe_segments(struct parakeet_context* ctx, const float* samples,
                                                       int n_samples, int64_t t_offset_cs, bool is_ja,
                                                       const parakeet_orchestrate_opts& opts);
