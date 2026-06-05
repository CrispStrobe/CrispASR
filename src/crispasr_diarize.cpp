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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
// over the mono buffer. Output is 7 class posteriors per frame:
//   0 = silence, 1 = spk0, 2 = spk1, 3 = spk0+1,
//   4 = spk2,    5 = spk0+2, 6 = spk1+2
// For each ASR segment, count the dominant speaker across its frames
// and assign the most-frequent one.
//
// What this path DOES handle correctly (post-#107 within-pass fixes):
//   * Overlap classes (3 = spk0+spk1, 5 = spk0+spk2, 6 = spk1+spk2) now
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
bool apply_pyannote(const float* mono, int n_samples, int64_t slice_t0_cs, std::vector<CrispasrDiarizeSegment>& segs,
                    const std::string& model_path, int n_threads) {
    if (model_path.empty())
        return false;

    pyannote_seg_context* pctx = pyannote_seg_init(model_path.c_str(), n_threads);
    if (!pctx)
        return false;

    int T_seg = 0;
    float* probs = pyannote_seg_run(pctx, mono, n_samples, &T_seg);
    pyannote_seg_free(pctx);
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

// Class layout of pyannote-seg-3.0:
//   0 = silence
//   1 = spk0 only
//   2 = spk1 only
//   3 = spk0 + spk1
//   4 = spk2 only
//   5 = spk0 + spk2
//   6 = spk1 + spk2
// Per-speaker activity mask: which output classes include each speaker.
// Used to sum per-frame activity probability across all classes that
// involve a given speaker — overlap classes contribute to BOTH speakers
// they cover, fixing the previous LUT collapse (#107).
static const float SPK_MASK[3][7] = {
    {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}, // spk0 active: classes 1, 3, 5
    {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f}, // spk1 active: classes 2, 3, 6
    {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, // spk2 active: classes 4, 5, 6
};

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

} // namespace crispasr_diarize_internal

bool crispasr_diarize_segments(const float* left, const float* right, int n_samples, bool is_stereo,
                               std::vector<CrispasrDiarizeSegment>& segs, const CrispasrDiarizeOptions& opts) {
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
    }
    return false;
}
