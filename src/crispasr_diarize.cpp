// crispasr_diarize.cpp — shared speaker diarization post-step.
// See crispasr_diarize.h for the interface contract.
//
// Extracted from examples/cli/crispasr_diarize.cpp so the CLI, the
// C-ABI wrapper `crispasr_diarize_segments`, and every language
// binding use the same implementation. The sherpa-ONNX subprocess
// path stays in the CLI (external binary, CLI-shaped UX).

#include "crispasr_diarize.h"
#include "crispasr_diarize_internal.h"
#include "pyannote_seg.h"
#include "wespeaker.h"

#include "core/crispasr_env.h"
#include "core/foxnose_pipeline.h"
#include "nemotron3_diar.h"
#include "core/powerset.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

// Convert an absolute centisecond timestamp into the per-channel sample
// index inside the current slice. Slice-relative because the energy
// arrays cover only [slice_t0_cs, slice_t0_cs + n_samples), not the
// whole input.
inline int64_t cs_to_sample_in_slice(int64_t cs_abs, int64_t slice_t0_cs) {
    int64_t cs_local = cs_abs - slice_t0_cs;
    if (cs_local < 0)
        cs_local = 0;
    return (cs_local * 16000) / 100;
}

// -----------------------------------------------------------------------
// Method 1: energy-based comparison (stereo only)
// -----------------------------------------------------------------------
//
// For each segment, sum |L[i]| and |R[i]| across the segment's sample
// range and pick the louder channel. Margin = 1.1× to avoid flapping
// on near-equal energy. Same threshold the historical crispasr
// `--diarize` path uses.
void apply_energy(const float* left, const float* right, int n_samples, int64_t slice_t0_cs,
                  std::vector<CrispasrDiarizeSegment>& segs) {
    for (auto& seg : segs) {
        int64_t is0 = cs_to_sample_in_slice(seg.t0_cs, slice_t0_cs);
        int64_t is1 = cs_to_sample_in_slice(seg.t1_cs, slice_t0_cs);
        if (is0 < 0)
            is0 = 0;
        if (is1 > n_samples)
            is1 = n_samples;
        if (is0 >= is1)
            continue;

        double e_l = 0.0, e_r = 0.0;
        for (int64_t j = is0; j < is1; j++) {
            e_l += std::fabs((double)left[j]);
            e_r += std::fabs((double)right[j]);
        }
        if (e_l > 1.1 * e_r)
            seg.speaker = 0;
        else if (e_r > 1.1 * e_l)
            seg.speaker = 1;
        else
            seg.speaker = -1;
    }
}

// -----------------------------------------------------------------------
// Method 2: cross-correlation lag (TDOA-style, stereo only)
// -----------------------------------------------------------------------
//
// Compute the cross-correlation of L and R within each segment over a
// short search window (±5 ms = ±80 samples at 16 kHz). The lag at the
// correlation peak's sign tells us which channel the voice is closest
// to. Falls back to energy on short segments.
void apply_xcorr(const float* left, const float* right, int n_samples, int64_t slice_t0_cs,
                 std::vector<CrispasrDiarizeSegment>& segs) {
    constexpr int MAX_LAG = 80; // ±5 ms at 16 kHz
    for (auto& seg : segs) {
        int64_t is0 = cs_to_sample_in_slice(seg.t0_cs, slice_t0_cs);
        int64_t is1 = cs_to_sample_in_slice(seg.t1_cs, slice_t0_cs);
        if (is0 < 0)
            is0 = 0;
        if (is1 > n_samples)
            is1 = n_samples;
        if (is1 - is0 < 2 * MAX_LAG) {
            double e_l = 0.0, e_r = 0.0;
            for (int64_t j = is0; j < is1; j++) {
                e_l += std::fabs((double)left[j]);
                e_r += std::fabs((double)right[j]);
            }
            seg.speaker = (e_l >= e_r) ? 0 : 1;
            continue;
        }

        const int64_t hi = is1 - MAX_LAG;
        const int64_t lo = is0 + MAX_LAG;
        double best = -1e30;
        int best_lag = 0;
        for (int lag = -MAX_LAG; lag <= MAX_LAG; lag++) {
            double sum = 0.0;
            for (int64_t j = lo; j < hi; j++) {
                sum += (double)left[j] * (double)right[j + lag];
            }
            if (sum > best) {
                best = sum;
                best_lag = lag;
            }
        }
        if (best_lag < 0)
            seg.speaker = 0;
        else if (best_lag > 0)
            seg.speaker = 1;
        else
            seg.speaker = -1;
    }
}

// -----------------------------------------------------------------------
// Method 3: VAD-turn segmentation (mono-friendly, timing only)
// -----------------------------------------------------------------------
//
// Walk segments in time order, toggling the speaker every time the gap
// to the previous segment exceeds 600 ms (conventional pause threshold
// used by pyannote / NeMo for natural conversation turns). Not real
// speaker ID — a proxy for turn boundaries that works on any input.
constexpr int64_t MIN_TURN_GAP_CS = 60;

void apply_vad_turns(std::vector<CrispasrDiarizeSegment>& segs) {
    if (segs.empty())
        return;
    int speaker = 0;
    int64_t prev_t1 = -1;
    for (auto& seg : segs) {
        if (prev_t1 >= 0 && (seg.t0_cs - prev_t1) > MIN_TURN_GAP_CS) {
            speaker = 1 - speaker;
        }
        seg.speaker = speaker;
        prev_t1 = seg.t1_cs;
    }
}

// -----------------------------------------------------------------------
// Method 4: native pyannote segmentation (no subprocess)
// -----------------------------------------------------------------------
//
// EXPERIMENTAL — segmentation only, NOT full diarization. See issue #107.
//
// Runs the GGUF-packed pyannote segmentation net from src/pyannote_seg.*
// over the mono buffer. Output is 7 powerset class posteriors per frame —
// see core/powerset.h for the layout; do not restate it here, a second copy
// of that table is what caused the class 3/4 swap.
// For each ASR segment, count the dominant speaker across its frames
// and assign the most-frequent one.
//
// What this path DOES handle correctly (post-#107 within-pass fixes):
//   * Overlap classes (4 = spk0+spk1, 5 = spk0+spk2, 6 = spk1+spk2) now
//     contribute activity to BOTH speakers they cover, instead of being
//     collapsed onto one via a class→single-speaker LUT.
//   * Per-frame, per-speaker activity is posterior-weighted (exp the
//     log-softmax, sum the per-speaker masks) rather than one-hot
//     argmax-then-count. Confident frames count more than uncertain
//     ones.
//   * Silence-dominated frames (p[silence] > 0.5) are gated out, so a
//     pure-silence segment leaves speaker = -1 rather than spuriously
//     picking spk0 from residual non-silence mass.
//
// What this path STILL does NOT handle (remaining work tracked in #107):
//   * Cross-slice / cross-window stitching. pyannote-seg emits LOCAL
//     speaker activity tracks (spk0/spk1/spk2 within ONE forward pass).
//     The CLI runs apply_pyannote per ASR slice, so the local track
//     indices reset / swap across slices and the "speaker 0" in slice
//     A is unrelated to "speaker 0" in slice B. On multi-slice
//     long-form audio you can therefore see consistent labels within
//     a slice but inconsistent labels across slices.
//   * ASR segment splitting at speaker-turn boundaries. Each ASR
//     segment is assigned exactly ONE speaker; if a turn change occurs
//     inside a long ASR span (e.g. cohere produces a single 27 s
//     segment containing both speakers), the minority speaker is lost.
// Fixing both requires (a) running pyannote ONCE over the whole audio
// instead of per slice, and (b) a speaker embedding + clustering step
// (TitaNet is already wired up; see src/titanet.h) to anchor IDs
// globally. For reliable multi-speaker diarization on long-form audio
// today, use --diarize-method sherpa (full segmentation + embedding +
// clustering pipeline).
// §176e: static cache for pyannote segmentation context. Avoids init/free
// per diarize call — same pattern as Silero VAD (#132) and MarbleNet/EncDec.
static std::mutex g_pyannote_cache_mtx;
static pyannote_seg_context* g_pyannote_cache_ctx = nullptr;
static std::string g_pyannote_cache_path;
static int g_pyannote_cache_threads = 0;

static pyannote_seg_context* pyannote_get_cached_locked(const char* path, int n_threads) {
    if (g_pyannote_cache_ctx && g_pyannote_cache_path == path && g_pyannote_cache_threads == n_threads)
        return g_pyannote_cache_ctx;
    if (g_pyannote_cache_ctx) {
        pyannote_seg_free(g_pyannote_cache_ctx);
        g_pyannote_cache_ctx = nullptr;
        g_pyannote_cache_path.clear();
    }
    g_pyannote_cache_ctx = pyannote_seg_init(path, n_threads);
    if (g_pyannote_cache_ctx) {
        g_pyannote_cache_path = path;
        g_pyannote_cache_threads = n_threads;
    }
    return g_pyannote_cache_ctx;
}

void crispasr_diarize_free_pyannote_cache() {
    std::lock_guard<std::mutex> lock(g_pyannote_cache_mtx);
    if (g_pyannote_cache_ctx) {
        pyannote_seg_free(g_pyannote_cache_ctx);
        g_pyannote_cache_ctx = nullptr;
        g_pyannote_cache_path.clear();
    }
}

bool apply_pyannote(const float* mono, int n_samples, int64_t slice_t0_cs, std::vector<CrispasrDiarizeSegment>& segs,
                    const std::string& model_path, int n_threads) {
    if (model_path.empty())
        return false;

    std::lock_guard<std::mutex> lock(g_pyannote_cache_mtx);
    pyannote_seg_context* pctx = pyannote_get_cached_locked(model_path.c_str(), n_threads);
    if (!pctx)
        return false;

    int T_seg = 0;
    float* probs = pyannote_seg_run(pctx, mono, n_samples, &T_seg);
    // Do NOT free pctx — it's cached.
    if (!probs || T_seg <= 0) {
        if (probs)
            std::free(probs);
        return false;
    }

    // Frame duration: sinc(stride=10) × 3 maxpools(stride=3) = 270 samples = 16.875 ms
    const double frame_dur_s = 270.0 / 16000.0;
    crispasr_diarize_internal::assign_speakers_from_log_posteriors(probs, T_seg, frame_dur_s, slice_t0_cs, segs);
    std::free(probs);
    return true;
}

} // namespace

namespace crispasr_diarize_internal {

// Class layout of pyannote-seg-3.0. The head is a POWERSET over 3 local
// speakers, and pyannote builds it with itertools.combinations ordered by
// increasing subset size — every singleton first, then every pair:
//   0 = silence
//   1 = spk0 only
//   2 = spk1 only
//   3 = spk2 only
//   4 = spk0 + spk1
//   5 = spk0 + spk2
//   6 = spk1 + spk2
//
// ⚠ This table used to have 3 and 4 swapped (3 read as "spk0+spk1" and 4 as
// "spk2 only"), which silently mis-attributed every frame of the THIRD local
// speaker to the first two as if they were talking over each other. Measured
// against VoxConverse dev ground truth, the implied overlap fraction gives the
// layout away — the old table only agrees with reality if a speaker can be
// active most of a file yet never once alone:
//
//   file    GT overlap   as {4,5,6} (this table)   as {3,5,6} (old table)
//   fsaal        0.42%                     0.17%                   62.65%
//   jyirt        0.09%                     0.00%                   28.32%
//   mesob       28.84%                    34.05%                    0.00%
//   nnqfq       14.40%                    10.14%                   48.40%
//
// Per-speaker activity mask: which output classes include each speaker.
// Used to sum per-frame activity probability across all classes that
// involve a given speaker — overlap classes contribute to BOTH speakers
// they cover, fixing the previous LUT collapse (#107).
//
// Derived from core/powerset.h rather than written out, because writing it out
// is exactly how 3 and 4 got swapped in the first place.
struct SpkMask {
    float m[core_powerset::kSpeakers][core_powerset::kClasses];
    constexpr SpkMask() : m() {
        for (int s = 0; s < core_powerset::kSpeakers; s++)
            for (int c = 0; c < core_powerset::kClasses; c++)
                m[s][c] = core_powerset::covers(c, s) ? 1.0f : 0.0f;
    }
};
static constexpr SpkMask kSpkMask{};
static const auto& SPK_MASK = kSpkMask.m;

int score_speaker_for_range(const float* log_probs, int T, double frame_dur_s, int64_t start_cs, int64_t end_cs) {
    if (!log_probs || T <= 0 || frame_dur_s <= 0.0)
        return -1;
    const double a0 = (double)start_cs / 100.0;
    const double a1 = (double)end_cs / 100.0;
    const int f0 = std::max(0, (int)(a0 / frame_dur_s));
    const int f1 = std::min(T, (int)(a1 / frame_dur_s) + 1);
    if (f0 >= f1)
        return -1;

    // Posterior-weighted per-speaker activity sum. For each non-
    // silence-dominated frame: exp the 7 log-probs once, then add to
    // each speaker's accumulator the sum of class probs covering that
    // speaker (via SPK_MASK).
    //
    // Silence gating: when p[silence] > 0.5 the frame is dropped
    // entirely (no speaker gets credit). This is what makes a pure-
    // silence range return -1 rather than picking spk0 from residual
    // non-silence mass. Without the gate, a 200-frame silence range
    // with p[0]≈0.99 and the other six classes each at ~3e-4 still
    // tallies ~0.12 per speaker.
    double act[3] = {0.0, 0.0, 0.0};
    int n_speech_frames = 0;
    for (int f = f0; f < f1; f++) {
        const float* lv = log_probs + f * 7;
        float p[7];
        for (int c = 0; c < 7; c++)
            p[c] = std::exp(lv[c]);
        if (p[0] > 0.5f)
            continue;
        for (int s = 0; s < 3; s++) {
            double sum = 0.0;
            for (int c = 0; c < 7; c++)
                sum += SPK_MASK[s][c] * p[c];
            act[s] += sum;
        }
        n_speech_frames++;
    }
    if (n_speech_frames == 0)
        return -1;

    int best_spk = 0;
    for (int s = 1; s < 3; s++)
        if (act[s] > act[best_spk])
            best_spk = s;
    // Tiny floor so an early-exit "speech_frames == 1 but no speaker
    // class fired" can't pick spk0 by index tie-break.
    return (act[best_spk] > 0.05) ? best_spk : -1;
}

void assign_speakers_from_log_posteriors(const float* log_probs, int T, double frame_dur_s, int64_t slice_t0_cs,
                                         std::vector<CrispasrDiarizeSegment>& segs) {
    if (!log_probs || T <= 0 || frame_dur_s <= 0.0)
        return;
    for (auto& seg : segs) {
        const int spk =
            score_speaker_for_range(log_probs, T, frame_dur_s, seg.t0_cs - slice_t0_cs, seg.t1_cs - slice_t0_cs);
        if (spk >= 0)
            seg.speaker = spk;
    }
}

std::vector<SpeakerRun> group_words_into_speaker_runs(const std::vector<int>& word_spk,
                                                      const std::vector<int64_t>& word_t0_cs,
                                                      const std::vector<int64_t>& word_t1_cs, int64_t min_run_cs) {
    std::vector<SpeakerRun> runs;
    const std::size_t n = word_spk.size();
    if (n == 0)
        return runs;

    // Build maximal same-speaker runs.
    std::size_t rs = 0;
    while (rs < n) {
        const int rspk = word_spk[rs];
        std::size_t re = rs + 1;
        while (re < n && word_spk[re] == rspk)
            re++;
        runs.push_back({rs, re, rspk});
        rs = re;
    }

    if (min_run_cs <= 0)
        return runs;

    auto run_span_cs = [&](const SpeakerRun& r) -> int64_t {
        if (r.end <= r.start || r.end > n)
            return 0;
        const int64_t t0 = word_t0_cs[r.start];
        const int64_t t1 = word_t1_cs[r.end - 1];
        return t1 > t0 ? (t1 - t0) : 0;
    };

    // Fold every run shorter than min_run_cs into its longer neighbour;
    // repeat until stable. A folded run inherits the neighbour's speaker.
    bool changed = true;
    while (changed && runs.size() >= 2) {
        changed = false;
        for (std::size_t i = 0; i < runs.size(); i++) {
            if (run_span_cs(runs[i]) >= min_run_cs)
                continue;
            std::size_t merge_into;
            if (i == 0) {
                merge_into = i + 1;
            } else if (i == runs.size() - 1) {
                merge_into = i - 1;
            } else {
                const int64_t prev_dur = run_span_cs(runs[i - 1]);
                const int64_t next_dur = run_span_cs(runs[i + 1]);
                merge_into = (prev_dur >= next_dur) ? (i - 1) : (i + 1);
            }
            if (merge_into == i + 1)
                runs[i + 1].start = runs[i].start;
            else // merge_into == i - 1
                runs[i - 1].end = runs[i].end;
            runs.erase(runs.begin() + (std::ptrdiff_t)i);
            changed = true;
            break;
        }
    }
    return runs;
}

} // namespace crispasr_diarize_internal

namespace {

// #324 FoxNose: embed sliding windows of each caller segment with WeSpeaker,
// cluster them, smooth the label sequence, then attribute every caller
// segment to the speaker turn it overlaps most.
//
// The caller's segments ARE the speech regions — they come from ASR/VAD
// upstream — so this method deliberately does not run a VAD of its own.
// Re-segmenting would both duplicate work and desynchronise the labels from
// the segments the caller is going to attach them to.
// One wespeaker context per concurrent worker. wespeaker_embed builds and
// computes a graph against context-owned scratch, so contexts cannot be
// shared across threads; the pipeline guarantees one live call per worker
// index, so no locking is needed either.
struct FoxnoseEmbedder {
    std::vector<wespeaker_context*> ctx;
};

int foxnose_embed_windows_cb(void* ud, int worker, const float* pcm, int n, const int* ws, const int* we, int n_win,
                             float* out) {
    auto* e = static_cast<FoxnoseEmbedder*>(ud);
    if (worker < 0 || worker >= (int)e->ctx.size())
        return -1;
    return wespeaker_embed_windows(e->ctx[worker], pcm, n, ws, we, n_win, out);
}

int foxnose_embed_cb(void* ud, int worker, const float* pcm, int n, float* out) {
    auto* e = static_cast<FoxnoseEmbedder*>(ud);
    if (worker < 0 || worker >= (int)e->ctx.size())
        return -1;
    return wespeaker_embed(e->ctx[worker], pcm, n, out);
}

int foxnose_embed_batch_cb(void* ud, int worker, const float* pcm, int64_t n, const int64_t* offs, const int* lens,
                           int n_win, float* out) {
    auto* e = static_cast<FoxnoseEmbedder*>(ud);
    if (worker < 0 || worker >= (int)e->ctx.size())
        return -1;
    return wespeaker_embed_batch(e->ctx[worker], pcm, n, offs, lens, n_win, out);
}

bool apply_foxnose(const float* left, int n_samples, const CrispasrDiarizeOptions& opts,
                   std::vector<CrispasrDiarizeSegment>& segs, std::vector<CrispasrDiarizeTurn>* out_turns) {
    if (opts.foxnose_embedder_path.empty()) {
        fprintf(stderr, "crispasr_diarize: foxnose needs --diarize-embedder <wespeaker.gguf>\n");
        return false;
    }
    if (segs.empty() || !left || n_samples <= 0)
        return true;

    // ONE ggml thread per context, and parallelism across windows instead.
    // A 1.2 s window is far too small to amortise ggml's per-graph thread
    // sync: measured on this model, ResNet34 took 57 ms at 1 thread and
    // 320 ms at 8, so raising -t made diarization 4.4x SLOWER end to end.
    // See core_foxnose::Params::n_workers.
    wespeaker_context_params cp = wespeaker_context_default_params();
    cp.n_threads = 1; // one thread per worker; parallelism comes from n_workers
    // #324 B1: opt-in GPU embedder (htdemucs pattern). Workers stay CPU-pinned
    // (wespeaker_init_worker), so pair this with CRISPASR_DIARIZE_EMBED_WORKERS=1
    // unless a mixed GPU-main + CPU-workers schedule is being measured.
    if (const char* e = std::getenv("CRISPASR_WESPEAKER_GPU"))
        if (*e && *e != '0')
            cp.use_gpu = true;
    // Same override the pluggable embedders honour, so the pre-#324-perf
    // behaviour (all of -t inside one graph, one window at a time) stays
    // reachable for an interleaved A/B.
    if (const char* e = std::getenv("CRISPASR_SPEAKER_EMBED_THREADS")) {
        const int v = std::atoi(e);
        if (v > 0)
            cp.n_threads = v;
    }
    cp.verbosity = 0;

    wespeaker_context* ctx = wespeaker_init_from_file(opts.foxnose_embedder_path.c_str(), cp);
    if (!ctx) {
        fprintf(stderr, "crispasr_diarize: failed to load embedder '%s'\n", opts.foxnose_embedder_path.c_str());
        return false;
    }
    FoxnoseEmbedder emb;
    emb.ctx.push_back(ctx);
    // Extra workers SHARE these weights — one GGUF read, one copy in RAM.
    int want_workers = std::max(1, opts.n_threads);
    if (const char* e = std::getenv("CRISPASR_DIARIZE_EMBED_WORKERS")) {
        const int v = std::atoi(e);
        if (v > 0)
            want_workers = v;
    }
    // GPU embedder => single context. Workers borrow the main context's
    // weights, and with use_gpu those live in a Metal buffer a CPU worker
    // cannot touch (measured: ggml_abort from the first worker thread).
    // One-GPU-many-CPU needs dual-resident weights; until someone builds
    // that, the GPU path runs all windows through the one GPU context.
    if (cp.use_gpu)
        want_workers = 1;
    for (int i = 1; i < want_workers; i++) {
        wespeaker_context* c = wespeaker_init_worker(ctx);
        if (!c)
            break; // fewer workers is slower, not wrong
        emb.ctx.push_back(c);
    }
    const int sr = wespeaker_sample_rate(ctx);
    const int dim = wespeaker_embed_dim(ctx);

    // Caller timestamps are absolute; the buffer starts at slice_t0_cs.
    std::vector<core_foxnose::Speech> speech;
    speech.reserve(segs.size());
    for (const auto& g : segs) {
        const double a = (double)(g.t0_cs - opts.slice_t0_cs) / 100.0;
        const double b = (double)(g.t1_cs - opts.slice_t0_cs) / 100.0;
        speech.push_back({std::max(0.0, a), std::max(0.0, b)});
    }

    core_foxnose::Params p;
    p.min_speakers = opts.min_speakers;
    p.max_speakers = opts.max_speakers;
    p.num_speakers = opts.num_speakers;
    p.n_workers = (int)emb.ctx.size();
    // Shared-trunk windowing: one trunk pass per span instead of per window.
    // Measured 1.78x less diarization CPU (66.0 s -> 37.0 s on an 85 s file,
    // 135 trunk passes -> 18) for +0.30 mean DER on the VoxConverse dev shard
    // (7.32% -> 7.62%; 6 of 8 files identical, one loses a speaker). Default
    // OFF because accuracy is the better default for a diarizer; opt in with
    // CRISPASR_DIARIZE_SPAN_EMBED=1.
    //
    // The cost is intrinsic, not a tuning problem: sharing a trunk pass means
    // adjacent windows share convolutional context, so the embedding space
    // smooths and a speaker with little airtime merges into a neighbour. Raw
    // silhouette then prefers the smaller k by a wide margin. See PLAN.md
    // "#324 shared-trunk" for the score curves.
    core_foxnose::EmbedWindowsFn span_fn = nullptr;
    if (const char* e = std::getenv("CRISPASR_DIARIZE_SPAN_EMBED"))
        if (*e && *e != '0')
            span_fn = foxnose_embed_windows_cb;
    // Batched independent-window embedding (#324 perf). Arithmetic-identical
    // to the per-window path, but MEASURED SLOWER on CPU (esrit.wav, -t 8:
    // 12.1 s -> 14.9 s wall) — ggml's CPU conv loops ne[3], so fusing buys no
    // GEMM shape and the 32-window chunks starve the 8-worker schedule. On the
    // GPU it collapses ~350 dispatches into ~11 and is the schedule that made
    // Metal reach CPU parity, so it is the DEFAULT there (single context, no
    // worker balance to lose). CRISPASR_DIARIZE_BATCH_EMBED=1/0 overrides
    // either way; the span path above, an accuracy-trading opt-in, wins when
    // both are set.
    core_foxnose::EmbedBatchFn batch_fn = (span_fn || !cp.use_gpu) ? nullptr : foxnose_embed_batch_cb;
    if (const char* e = std::getenv("CRISPASR_DIARIZE_BATCH_EMBED")) {
        if (*e && *e == '0')
            batch_fn = nullptr;
        else if (*e && *e != '0' && !span_fn)
            batch_fn = foxnose_embed_batch_cb;
    }
    core_foxnose::Result res =
        core_foxnose::diarize(left, n_samples, sr, speech, foxnose_embed_cb, &emb, dim, p, span_fn, batch_fn);
    if (std::getenv("CRISPASR_DIARIZE_DEBUG"))
        fprintf(stderr, "crispasr[diarize]: foxnose windows=%d skipped=%d -> %d speakers (%s), %zu turns\n",
                res.n_windows, res.n_skipped, res.n_speakers, res.reason.c_str(), res.turns.size());
    // Workers borrow ctx's weights, so they must go first.
    for (size_t i = emb.ctx.size(); i-- > 0;)
        wespeaker_free(emb.ctx[i]);

    if (out_turns) {
        out_turns->clear();
        out_turns->reserve(res.turns.size());
        for (const auto& t : res.turns)
            out_turns->push_back({t.start, t.end, t.speaker});
    }

    if (res.turns.empty())
        return true; // nothing to say; leave speaker = -1

    for (size_t i = 0; i < segs.size(); i++) {
        const double a = speech[i].start, b = speech[i].end;
        double best = 0.0;
        int best_spk = -1;
        for (const auto& t : res.turns) {
            const double ov = std::min(b, t.end) - std::max(a, t.start);
            if (ov > best) {
                best = ov;
                best_spk = t.speaker;
            }
        }
        segs[i].speaker = best_spk;
    }
    return true;
}

// ── Sortformer (#466) ─────────────────────────────────────────────────────
static std::mutex g_sortformer_mtx;
static nemotron3_diar_context* g_sortformer_ctx = nullptr;
static std::string g_sortformer_path;

bool apply_sortformer(const float* mono, int n_samples, const CrispasrDiarizeOptions& opts,
                      std::vector<CrispasrDiarizeSegment>& segs, std::vector<CrispasrDiarizeTurn>* out_turns) {
    if (opts.sortformer_model_path.empty()) {
        fprintf(stderr, "crispasr_diarize: sortformer needs a Nemotron-3-Diarization GGUF (--diarize-model)\n");
        return false;
    }
    std::lock_guard<std::mutex> lock(g_sortformer_mtx);
    if (!g_sortformer_ctx || g_sortformer_path != opts.sortformer_model_path) {
        if (g_sortformer_ctx)
            nemotron3_diar_free(g_sortformer_ctx);
        auto p = nemotron3_diar_default_params();
        p.n_threads = opts.n_threads;
        p.verbosity = 0;
        g_sortformer_ctx = nemotron3_diar_init_from_file(opts.sortformer_model_path.c_str(), p);
        g_sortformer_path = g_sortformer_ctx ? opts.sortformer_model_path : std::string();
        if (!g_sortformer_ctx)
            return false;
    }
    std::string mode = opts.sortformer_mode;
    if (mode.empty()) {
        const char* e = crispasr_env::get("CRISPASR_SORTFORMER_MODE");
        mode = (e && *e) ? e : "offline";
    }
    if (nemotron3_diar_set_mode(g_sortformer_ctx, mode.c_str()) != 0) {
        fprintf(stderr,
                "crispasr_diarize: unknown sortformer mode '%s' (offline, low_latency, very_low_latency, "
                "ultra_low_latency)\n",
                mode.c_str());
        return false;
    }
    int T = 0, S = 0;
    float* probs = nemotron3_diar_probs(g_sortformer_ctx, mono, n_samples, &T, &S);
    if (!probs || T <= 0) {
        std::free(probs);
        return false;
    }
    // Rows past the valid-frame count are padding (transformers masks them).
    T = std::min(T, nemotron3_diar_n_valid_frames(g_sortformer_ctx, n_samples));
    const double frame_s = 0.01; // one row per 10 ms
    const float thr = opts.sortformer_threshold;
    // Turns: runs of p > thr per speaker (Nemotron3DiarizationProcessor.extract_speaker_dict).
    std::vector<CrispasrDiarizeTurn> turns;
    for (int s = 0; s < S; s++) {
        int start = -1;
        for (int t = 0; t <= T; t++) {
            const bool on = t < T && probs[(size_t)t * S + s] > thr;
            if (on && start < 0)
                start = t;
            if (!on && start >= 0) {
                turns.push_back({start * frame_s, t * frame_s, s});
                start = -1;
            }
        }
    }
    std::sort(turns.begin(), turns.end(), [](const CrispasrDiarizeTurn& a, const CrispasrDiarizeTurn& b) {
        return a.start_s < b.start_s || (a.start_s == b.start_s && a.speaker < b.speaker);
    });
    // Label each caller segment by the speaker with the most probability mass inside it.
    for (auto& seg : segs) {
        // one centisecond == one 10 ms frame
        const int f0 = std::max(0, (int)(seg.t0_cs - opts.slice_t0_cs));
        const int f1 = std::min(T, (int)(seg.t1_cs - opts.slice_t0_cs));
        int best = -1;
        double best_mass = 0.0;
        for (int s = 0; s < S; s++) {
            double mass = 0.0;
            for (int t = f0; t < f1; t++)
                mass += probs[(size_t)t * S + s];
            if (mass > best_mass) {
                best_mass = mass;
                best = s;
            }
        }
        // no speaker ever above threshold in the segment -> leave it unlabelled
        bool any = false;
        for (int t = f0; t < f1 && !any && best >= 0; t++)
            any = probs[(size_t)t * S + best] > thr;
        seg.speaker = any ? best : -1;
    }
    std::free(probs);
    if (out_turns)
        *out_turns = std::move(turns);
    return true;
}

} // namespace

bool crispasr_diarize_segments(const float* left, const float* right, int n_samples, bool is_stereo,
                               std::vector<CrispasrDiarizeSegment>& segs, const CrispasrDiarizeOptions& opts,
                               std::vector<CrispasrDiarizeTurn>* out_turns) {
    if (segs.empty() || !left || n_samples <= 0)
        return true; // nothing to do, but not an error

    switch (opts.method) {
    case CrispasrDiarizeMethod::Energy:
        if (!is_stereo || !right)
            return true; // can't energy-diarize mono; leave speakers untouched
        apply_energy(left, right, n_samples, opts.slice_t0_cs, segs);
        return true;
    case CrispasrDiarizeMethod::Xcorr:
        if (!is_stereo || !right)
            return true;
        apply_xcorr(left, right, n_samples, opts.slice_t0_cs, segs);
        return true;
    case CrispasrDiarizeMethod::VadTurns:
        apply_vad_turns(segs);
        return true;
    case CrispasrDiarizeMethod::Pyannote:
        return apply_pyannote(left, n_samples, opts.slice_t0_cs, segs, opts.pyannote_model_path, opts.n_threads);
    case CrispasrDiarizeMethod::FoxNose:
        return apply_foxnose(left, n_samples, opts, segs, out_turns);
    case CrispasrDiarizeMethod::Sortformer:
        return apply_sortformer(left, n_samples, opts, segs, out_turns);
    }
    return false;
}
