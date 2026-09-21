// basic_pitch.cpp — Spotify Basic Pitch backend (Apache-2.0)
//
// Port of basic_pitch/{models,nn,inference,note_creation}.py and
// basic_pitch/layers/{nnaudio,signal}.py. See src/basic_pitch.h for the
// architecture and models/convert-basic-pitch-to-gguf.py for the weights.
//
// The whole network is six convolutions, so everything runs in plain C++ loops
// rather than a ggml graph. ggml is still used for GGUF loading, which is what
// every other backend here does.
//
// WARNING: this header used to say the six convolutions were "small" and that
// "the expensive part is the CQT front end, not the network". Both are wrong,
// and the second shaped this file for a long time. Per 43844-sample window
// (T=172) the six call sites in bp_forward_window cost 484.4 MMAC, of which
// contour_conv (8->8, 3x39, out 172x264) alone is 340.0 MMAC = 70%, against
// roughly 6 MMAC for the CQT. Run with CRISPASR_BASIC_PITCH_TIMING=1 to print
// the split. See bp_conv_fast below for what follows from that.

#include "basic_pitch.h"

#include "core/basic_pitch_conv.h"
#include "core/cqt2010v2.h"
#include "core/gguf_loader.h"
#include "core/ggml_cpu_backend.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

// ─── Hyperparameters ────────────────────────────────────────────────────────

struct bp_hparams {
    uint32_t sample_rate = 22050;
    uint32_t fft_hop = 256;
    uint32_t audio_n_samples = 43844; // 22050*2 - 256
    uint32_t annot_n_frames = 172;    // (22050/256) * 2
    uint32_t annotations_fps = 86;
    uint32_t overlapping_frames = 30;
    uint32_t midi_offset = 21;
    uint32_t n_freq_bins_notes = 88;
    uint32_t n_freq_bins_contours = 264;
    uint32_t contours_bins_per_semitone = 3;

    uint32_t cqt_n_bins = 309;
    uint32_t cqt_bins_per_octave = 36;
    uint32_t cqt_n_octaves = 9;
    uint32_t cqt_n_filters = 36;
    uint32_t cqt_n_fft = 256;
    uint32_t cqt_hop = 256;
    float cqt_fmin = 27.5f;
    float cqt_bn_scale = 2.480741f;
    float cqt_bn_shift = -0.8769183f;
    float norm_log_eps = 1e-10f;

    uint32_t n_harmonics = 8;
    std::vector<int> harmonic_shifts{-36, 0, 36, 57, 72, 84, 93, 101};
};

using core_basic_pitch::bp_conv;
using core_basic_pitch::bp_conv2d;
using core_basic_pitch::bp_fastconv_on;
using core_basic_pitch::bp_timing_on;
namespace bp_conv_fast = core_basic_pitch::bp_conv_fast;

// ─── Weights ────────────────────────────────────────────────────────────────
//
// Materialised as float vectors at load: the whole model is ~40k weights, so
// keeping a second F32 copy costs 160 kB and removes an F16 conversion from
// every inner loop.

struct bp_weights {
    std::vector<float> cqt_real;     // [36 * 256]
    std::vector<float> cqt_imag;     // [36 * 256]
    std::vector<float> cqt_lowpass;  // [256]
    std::vector<float> sqrt_lengths; // [309]

    bp_conv contour_conv; // (8, 8, 3, 39)   stride (1,1) pads (1,19)
    bp_conv contour_out;  // (1, 8, 5, 5)    stride (1,1) pads (2,2)
    bp_conv note_conv;    // (32, 1, 7, 7)   stride (1,3) pads (3,2)
    bp_conv note_out;     // (1, 32, 7, 3)   stride (1,1) pads (3,1)
    bp_conv onset_conv;   // (32, 8, 5, 5)   stride (1,3) pads (2,1)
    bp_conv onset_out;    // (1, 33, 3, 3)   stride (1,1) pads (1,1)
};

struct basic_pitch_ctx {
    bp_hparams hp;
    bp_weights w;
    basic_pitch_params params{};

    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    ggml_backend_t backend = nullptr;
};

// ─── Small helpers ──────────────────────────────────────────────────────────

static std::vector<float> bp_tensor_f32(ggml_tensor* t) {
    std::vector<float> out;
    if (!t)
        return out;
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(src[i]);
    } else {
        fprintf(stderr, "basic-pitch: unsupported tensor type %d for '%s'\n", (int)t->type, ggml_get_name(t));
        out.clear();
    }
    return out;
}

static void bp_relu(std::vector<float>& v) {
    for (float& x : v)
        if (x < 0.0f)
            x = 0.0f;
}

static void bp_sigmoid(std::vector<float>& v) {
    for (float& x : v)
        x = 1.0f / (1.0f + std::exp(-x));
}

// ─── Front end: CQT → NormalizedLog → BN → HarmonicStacking ─────────────────

// Output is channel-major [n_harmonics][T][264], which is what bp_conv2d wants.
static std::vector<float> bp_harmonic_stack(const bp_hparams& hp, const std::vector<float>& cqt, int T) {
    const int NB = (int)hp.cqt_n_bins;
    const int NF = (int)hp.n_freq_bins_contours;
    const int NH = (int)hp.n_harmonics;
    std::vector<float> out((size_t)NH * (size_t)T * (size_t)NF, 0.0f);
    for (int c = 0; c < NH; c++) {
        const int shift = hp.harmonic_shifts[(size_t)c];
        float* dst = out.data() + (size_t)c * (size_t)T * (size_t)NF;
        for (int t = 0; t < T; t++) {
            const float* src = cqt.data() + (size_t)t * (size_t)NB;
            float* row = dst + (size_t)t * (size_t)NF;
            for (int f = 0; f < NF; f++) {
                // shift > 0: x[:, :, shift:] then right zero-pad  → src bin f+shift
                // shift < 0: x[:, :, :shift] then left  zero-pad  → src bin f+shift
                // shift == 0: identity.
                const int s = f + shift;
                row[f] = (s >= 0 && s < NB) ? src[s] : 0.0f;
            }
        }
    }
    return out;
}

struct bp_window_out {
    int T = 0;
    std::vector<float> cqt;     // [T * 309] magnitude
    std::vector<float> normlog; // [T * 309] after NormalizedLog
    std::vector<float> hstack;  // [8 * T * 264] channel-major, after BN
    std::vector<float> contour; // [T * 264]
    std::vector<float> note;    // [T * 88]
    std::vector<float> onset;   // [T * 88]
};

// One 43844-sample window through the whole model.
static bool bp_forward_window(const basic_pitch_ctx* ctx, const float* window, int n, bp_window_out& out) {
    const bp_hparams& hp = ctx->hp;
    const bp_weights& w = ctx->w;
    const int nth = ctx->params.n_threads > 0 ? ctx->params.n_threads : 1;

    // CRISPASR_BASIC_PITCH_TIMING=1 prints the per-stage split. It exists to
    // settle where this backend actually spends its time.
    const bool timing = bp_timing_on();
    using bp_clock = std::chrono::steady_clock;
    bp_clock::time_point t_mark = bp_clock::now();
    double ms_cqt = 0, ms_stack = 0, ms_conv = 0, ms_act = 0;
    auto lap = [&](double& acc) {
        const bp_clock::time_point now = bp_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - t_mark).count();
        t_mark = now;
    };

    core_cqt2010v2::Params cp;
    cp.n_bins = (int)hp.cqt_n_bins;
    cp.bins_per_octave = (int)hp.cqt_bins_per_octave;
    cp.n_octaves = (int)hp.cqt_n_octaves;
    cp.n_filters = (int)hp.cqt_n_filters;
    cp.n_fft = (int)hp.cqt_n_fft;
    cp.hop_length = (int)hp.cqt_hop;

    core_cqt2010v2::Kernels ck;
    ck.real = w.cqt_real.data();
    ck.imag = w.cqt_imag.data();
    ck.lowpass = w.cqt_lowpass.data();
    ck.lowpass_len = (int)w.cqt_lowpass.size();
    ck.sqrt_lengths = w.sqrt_lengths.data();

    const int T = core_cqt2010v2::magnitude(cp, ck, window, n, out.cqt);
    if (T <= 0) {
        fprintf(stderr, "basic-pitch: CQT failed (n=%d)\n", n);
        return false;
    }
    out.T = T;

    if (timing)
        lap(ms_cqt);

    out.normlog = out.cqt;
    core_cqt2010v2::normalized_log(out.normlog.data(), out.normlog.size(), hp.norm_log_eps);

    // The single-channel BatchNorm after the CQT, folded to scale/shift by the
    // ONNX export.
    std::vector<float> bn = out.normlog;
    for (float& v : bn)
        v = v * hp.cqt_bn_scale + hp.cqt_bn_shift;

    out.hstack = bp_harmonic_stack(hp, bn, T);
    if (timing)
        lap(ms_stack);

    const int NF = (int)hp.n_freq_bins_contours; // 264
    const int NN = (int)hp.n_freq_bins_notes;    // 88
    const int NH = (int)hp.n_harmonics;

    // ── contour head ───────────────────────────────────────────────────────
    int w1 = 0, w2 = 0;
    std::vector<float> h = bp_conv2d(out.hstack, NH, T, NF, w.contour_conv, 1, 1, 19, w1, nth);
    if (timing)
        lap(ms_conv);
    bp_relu(h);
    if (timing)
        lap(ms_act);
    std::vector<float> contour = bp_conv2d(h, w.contour_conv.oc, T, w1, w.contour_out, 1, 2, 2, w2, nth);
    if (timing)
        lap(ms_conv);
    bp_sigmoid(contour);
    if (timing)
        lap(ms_act);
    out.contour = contour; // [1][T][264] == [T][264]

    // ── note head (takes the contour map as a single channel) ──────────────
    std::vector<float> nh = bp_conv2d(contour, 1, T, w2, w.note_conv, 3, 3, 2, w1, nth);
    if (timing)
        lap(ms_conv);
    bp_relu(nh);
    if (timing)
        lap(ms_act);
    std::vector<float> note_pre = bp_conv2d(nh, w.note_conv.oc, T, w1, w.note_out, 1, 3, 1, w2, nth);
    if (timing)
        lap(ms_conv);
    bp_sigmoid(note_pre); // x_notes_pre — the sigmoid IS part of this layer
    out.note = note_pre;

    // ── onset head ─────────────────────────────────────────────────────────
    std::vector<float> oh = bp_conv2d(out.hstack, NH, T, NF, w.onset_conv, 3, 2, 1, w1, nth);
    if (timing)
        lap(ms_conv);
    bp_relu(oh);
    if (timing)
        lap(ms_act);
    // Concat([x_notes_pre, oh], axis=channels) — note map FIRST, matching the
    // ONNX Concat input order.
    const int oc = w.onset_conv.oc;
    std::vector<float> cat((size_t)(oc + 1) * (size_t)T * (size_t)NN);
    std::memcpy(cat.data(), note_pre.data(), (size_t)T * (size_t)NN * sizeof(float));
    std::memcpy(cat.data() + (size_t)T * (size_t)NN, oh.data(), (size_t)oc * (size_t)T * (size_t)NN * sizeof(float));
    std::vector<float> onset = bp_conv2d(cat, oc + 1, T, NN, w.onset_out, 1, 1, 1, w2, nth);
    if (timing)
        lap(ms_conv);
    bp_sigmoid(onset);
    if (timing)
        lap(ms_act);
    out.onset = onset;
    if (timing) {
        lap(ms_act);
        fprintf(stderr,
                "basic-pitch timing: cqt %.1f ms  hstack %.1f ms  conv %.1f ms  act %.1f ms  "
                "(conv %s, threads %d, 484.4 MMAC -> %.2f GMAC/s)\n",
                ms_cqt, ms_stack, ms_conv, ms_act,
                bp_fastconv_on() ? bp_conv_fast::kernel_name(bp_conv_fast::pick_kernel()) : "reference", nth,
                ms_conv > 0 ? 0.4844 / (ms_conv / 1000.0) : 0.0);
    }
    return true;
}

// ─── Windowing / stitching (inference.py) ───────────────────────────────────

struct bp_stitched {
    int T = 0;
    std::vector<float> contour; // [T * 264]
    std::vector<float> note;    // [T * 88]
    std::vector<float> onset;   // [T * 88]
};

static bool bp_run_file(const basic_pitch_ctx* ctx, const float* pcm, int n_samples, bp_stitched& out) {
    const bp_hparams& hp = ctx->hp;
    const int W = (int)hp.audio_n_samples;               // 43844
    const int n_overlap = (int)hp.overlapping_frames;    // 30
    const int overlap_len = n_overlap * (int)hp.fft_hop; // 7680
    const int hop_size = W - overlap_len;                // 36164
    const int n_olap = n_overlap / 2;                    // 15
    const int keep = (int)hp.annot_n_frames - n_overlap; // 142
    const int NF = (int)hp.n_freq_bins_contours;
    const int NN = (int)hp.n_freq_bins_notes;

    // get_audio_input(): a LEADING zero pad of overlap_len/2 samples and
    // nothing else. No trailing pad beyond what each short window gets.
    std::vector<float> padded((size_t)overlap_len / 2, 0.0f);
    padded.insert(padded.end(), pcm, pcm + n_samples);

    std::vector<float> window((size_t)W);
    int n_windows = 0;
    for (size_t i = 0; i < padded.size(); i += (size_t)hop_size) {
        const size_t avail = std::min((size_t)W, padded.size() - i);
        std::memcpy(window.data(), padded.data() + i, avail * sizeof(float));
        if (avail < (size_t)W)
            std::memset(window.data() + avail, 0, ((size_t)W - avail) * sizeof(float));

        bp_window_out wo;
        if (!bp_forward_window(ctx, window.data(), W, wo))
            return false;
        if (wo.T <= n_overlap)
            return false;

        // unwrap_output(): drop n_olap frames from each end of every window.
        for (int t = n_olap; t < wo.T - n_olap; t++) {
            out.contour.insert(out.contour.end(), wo.contour.begin() + (size_t)t * NF,
                               wo.contour.begin() + (size_t)(t + 1) * NF);
            out.note.insert(out.note.end(), wo.note.begin() + (size_t)t * NN, wo.note.begin() + (size_t)(t + 1) * NN);
            out.onset.insert(out.onset.end(), wo.onset.begin() + (size_t)t * NN,
                             wo.onset.begin() + (size_t)(t + 1) * NN);
        }
        n_windows++;
    }
    if (n_windows == 0)
        return false;

    // Trim to the number of frames the ORIGINAL (unpadded) audio accounts for.
    // Upstream uses a float window count, so this is a truncation of a product,
    // not a frame count derived from n_samples directly.
    const double n_expected_windows = (double)n_samples / (double)hop_size;
    int T = (int)(n_expected_windows * (double)keep);
    T = std::max(0, std::min(T, n_windows * keep));
    out.contour.resize((size_t)T * NF);
    out.note.resize((size_t)T * NN);
    out.onset.resize((size_t)T * NN);
    out.T = T;
    return true;
}

// ─── Note creation (note_creation.py) ───────────────────────────────────────

// note_creation.py::model_frames_to_time. The window offset is subtracted once
// per ANNOT_N_FRAMES-sized block even though only `keep` frames survive per
// window — that is upstream's formula, and the note times were tuned with it.
static double bp_frame_time(const bp_hparams& hp, int i) {
    const double hop_s = (double)hp.fft_hop / (double)hp.sample_rate;
    const double magic = 0.0018;
    const double window_offset =
        hop_s * ((double)hp.annot_n_frames - (double)hp.audio_n_samples / (double)hp.fft_hop) + magic;
    const double wn = std::floor((double)i / (double)hp.annot_n_frames);
    return (double)i * hop_s - window_offset * wn;
}

// note_creation.py::get_infered_onsets(n_diff=2), in place on `onsets`.
static void bp_infer_onsets(std::vector<float>& onsets, const std::vector<float>& frames, int T, int F) {
    if (T <= 0)
        return;
    std::vector<float> fd((size_t)T * (size_t)F, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < F; f++) {
            // diffs[n][t] = frames[t] - frames[t-n], with frames[<0] = 0.
            float mn = 0.0f;
            for (int n = 1; n <= 2; n++) {
                const float prev = (t - n >= 0) ? frames[(size_t)(t - n) * F + f] : 0.0f;
                const float d = frames[(size_t)t * F + f] - prev;
                if (n == 1 || d < mn)
                    mn = d;
            }
            fd[(size_t)t * F + f] = mn < 0.0f ? 0.0f : mn;
        }
    }
    for (int t = 0; t < std::min(2, T); t++)
        for (int f = 0; f < F; f++)
            fd[(size_t)t * F + f] = 0.0f;

    float max_on = 0.0f, max_fd = 0.0f;
    for (size_t i = 0; i < onsets.size(); i++)
        max_on = std::max(max_on, onsets[i]);
    for (size_t i = 0; i < fd.size(); i++)
        max_fd = std::max(max_fd, fd[i]);
    if (max_fd <= 0.0f)
        return; // upstream would produce NaN here; a silent clip has no onsets
    const float k = max_on / max_fd;
    for (size_t i = 0; i < onsets.size(); i++)
        onsets[i] = std::max(onsets[i], fd[i] * k);
}

struct bp_frame_note {
    int start = 0;
    int end = 0;
    int midi = 0;
    float amplitude = 0.0f;
};

// note_creation.py::output_to_notes_polyphonic. Every mutation of
// `remaining_energy` is order dependent, so the traversal order below is
// deliberately identical to numpy's: peaks descending in time then in
// frequency (`np.where(...)[::-1]`), and the melodia argmax taking the FIRST
// maximum in row-major order.
static std::vector<bp_frame_note> bp_output_to_notes(const basic_pitch_ctx* ctx, const bp_stitched& s) {
    const bp_hparams& hp = ctx->hp;
    const basic_pitch_params& P = ctx->params;
    const int T = s.T;
    const int F = (int)hp.n_freq_bins_notes;
    const int max_freq_idx = F - 1; // 87
    const int energy_tol = 11;
    std::vector<bp_frame_note> notes;
    if (T <= 1)
        return notes;

    const int min_note_len =
        (int)std::lround((double)P.minimum_note_length_ms / 1000.0 * ((double)hp.sample_rate / (double)hp.fft_hop));

    std::vector<float> frames = s.note;
    std::vector<float> onsets = s.onset;

    // constrain_frequency(): idx = round(hz_to_midi(f)) - 21.
    int min_idx = 0, max_idx = F;
    if (P.minimum_frequency > 0.0f)
        min_idx = (int)std::lround(69.0 + 12.0 * std::log2((double)P.minimum_frequency / 440.0)) - (int)hp.midi_offset;
    if (P.maximum_frequency > 0.0f)
        max_idx = (int)std::lround(69.0 + 12.0 * std::log2((double)P.maximum_frequency / 440.0)) - (int)hp.midi_offset;
    min_idx = std::max(0, std::min(min_idx, F));
    max_idx = std::max(0, std::min(max_idx, F));
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < min_idx; f++) {
            onsets[(size_t)t * F + f] = 0.0f;
            frames[(size_t)t * F + f] = 0.0f;
        }
        for (int f = max_idx; f < F; f++) {
            onsets[(size_t)t * F + f] = 0.0f;
            frames[(size_t)t * F + f] = 0.0f;
        }
    }

    if (P.infer_onsets)
        bp_infer_onsets(onsets, frames, T, F);

    std::vector<float> rem = frames;

    // scipy.signal.argrelmax(onsets, axis=0) with mode='clip': a strict local
    // maximum in time. Clipping makes both endpoints compare against
    // themselves, so they never qualify.
    auto is_peak = [&](int t, int f) {
        const float v = onsets[(size_t)t * F + f];
        const int lo = std::max(t - 1, 0);
        const int hi = std::min(t + 1, T - 1);
        return v > onsets[(size_t)lo * F + f] && v > onsets[(size_t)hi * F + f];
    };

    for (int t = T - 1; t >= 0; t--) {
        for (int f = F - 1; f >= 0; f--) {
            if (!is_peak(t, f) || onsets[(size_t)t * F + f] < P.onset_threshold)
                continue;
            if (t >= T - 1)
                continue;
            int i = t + 1;
            int k = 0;
            while (i < T - 1 && k < energy_tol) {
                if (rem[(size_t)i * F + f] < P.frame_threshold)
                    k++;
                else
                    k = 0;
                i++;
            }
            i -= k;
            if (i - t <= min_note_len)
                continue;
            for (int j = t; j < i; j++) {
                rem[(size_t)j * F + f] = 0.0f;
                if (f < max_freq_idx)
                    rem[(size_t)j * F + f + 1] = 0.0f;
                if (f > 0)
                    rem[(size_t)j * F + f - 1] = 0.0f;
            }
            double acc = 0.0;
            for (int j = t; j < i; j++)
                acc += frames[(size_t)j * F + f];
            notes.push_back({t, i, f + (int)hp.midi_offset, (float)(acc / (double)(i - t))});
        }
    }

    if (P.melodia_trick) {
        for (;;) {
            int bt = -1, bf = -1;
            float best = 0.0f;
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    const float v = rem[(size_t)t * F + f];
                    if (bt < 0 || v > best) {
                        best = v;
                        bt = t;
                        bf = f;
                    }
                }
            }
            if (bt < 0 || best <= P.frame_threshold)
                break;
            rem[(size_t)bt * F + bf] = 0.0f;

            int i = bt + 1;
            int k = 0;
            while (i < T - 1 && k < energy_tol) {
                if (rem[(size_t)i * F + bf] < P.frame_threshold)
                    k++;
                else
                    k = 0;
                rem[(size_t)i * F + bf] = 0.0f;
                if (bf < max_freq_idx)
                    rem[(size_t)i * F + bf + 1] = 0.0f;
                if (bf > 0)
                    rem[(size_t)i * F + bf - 1] = 0.0f;
                i++;
            }
            const int i_end = i - 1 - k;

            i = bt - 1;
            k = 0;
            while (i > 0 && k < energy_tol) {
                if (rem[(size_t)i * F + bf] < P.frame_threshold)
                    k++;
                else
                    k = 0;
                rem[(size_t)i * F + bf] = 0.0f;
                if (bf < max_freq_idx)
                    rem[(size_t)i * F + bf + 1] = 0.0f;
                if (bf > 0)
                    rem[(size_t)i * F + bf - 1] = 0.0f;
                i--;
            }
            const int i_start = i + 1 + k;

            if (i_end - i_start <= min_note_len)
                continue;
            if (i_start < 0 || i_end > T)
                continue;
            double acc = 0.0;
            for (int j = i_start; j < i_end; j++)
                acc += frames[(size_t)j * F + bf];
            const int len = i_end - i_start;
            notes.push_back({i_start, i_end, bf + (int)hp.midi_offset, len > 0 ? (float)(acc / (double)len) : 0.0f});
        }
    }

    return notes;
}

// ─── Load ───────────────────────────────────────────────────────────────────

static bool bp_bind_conv(core_gguf::tensor_map& tm, const char* prefix, bp_conv& c, int oc, int ic, int kh, int kw) {
    const std::string wname = std::string(prefix) + ".weight";
    const std::string bname = std::string(prefix) + ".bias";
    ggml_tensor* wt = core_gguf::require(tm, wname.c_str(), "basic-pitch");
    ggml_tensor* bt = core_gguf::require(tm, bname.c_str(), "basic-pitch");
    if (!wt || !bt)
        return false;
    c.w = bp_tensor_f32(wt);
    c.b = bp_tensor_f32(bt);
    c.oc = oc;
    c.ic = ic;
    c.kh = kh;
    c.kw = kw;
    const size_t want = (size_t)oc * (size_t)ic * (size_t)kh * (size_t)kw;
    if (c.w.size() != want || c.b.size() != (size_t)oc) {
        fprintf(stderr, "basic-pitch: '%s' has %zu weights / %zu biases, expected %zu / %d\n", prefix, c.w.size(),
                c.b.size(), want, oc);
        return false;
    }
    return true;
}

struct basic_pitch_params basic_pitch_default_params(void) {
    basic_pitch_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.onset_threshold = 0.5f;
    p.frame_threshold = 0.3f;
    p.minimum_note_length_ms = 127.7f;
    p.minimum_frequency = 0.0f;
    p.maximum_frequency = 0.0f;
    p.infer_onsets = true;
    p.melodia_trick = true;
    return p;
}

struct basic_pitch_ctx* basic_pitch_init_from_file(const char* path, struct basic_pitch_params params) {
    if (!path)
        return nullptr;
    basic_pitch_ctx* ctx = new basic_pitch_ctx();
    ctx->params = params;

    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete ctx;
        return nullptr;
    }
    bp_hparams& hp = ctx->hp;
    hp.sample_rate = core_gguf::kv_u32(meta, "basic_pitch.sample_rate", hp.sample_rate);
    hp.fft_hop = core_gguf::kv_u32(meta, "basic_pitch.fft_hop", hp.fft_hop);
    hp.audio_n_samples = core_gguf::kv_u32(meta, "basic_pitch.audio_n_samples", hp.audio_n_samples);
    hp.annot_n_frames = core_gguf::kv_u32(meta, "basic_pitch.annot_n_frames", hp.annot_n_frames);
    hp.annotations_fps = core_gguf::kv_u32(meta, "basic_pitch.annotations_fps", hp.annotations_fps);
    hp.overlapping_frames = core_gguf::kv_u32(meta, "basic_pitch.overlapping_frames", hp.overlapping_frames);
    hp.midi_offset = core_gguf::kv_u32(meta, "basic_pitch.midi_offset", hp.midi_offset);
    hp.n_freq_bins_notes = core_gguf::kv_u32(meta, "basic_pitch.n_freq_bins_notes", hp.n_freq_bins_notes);
    hp.n_freq_bins_contours = core_gguf::kv_u32(meta, "basic_pitch.n_freq_bins_contours", hp.n_freq_bins_contours);
    hp.contours_bins_per_semitone =
        core_gguf::kv_u32(meta, "basic_pitch.contours_bins_per_semitone", hp.contours_bins_per_semitone);
    hp.cqt_n_bins = core_gguf::kv_u32(meta, "basic_pitch.cqt.n_bins", hp.cqt_n_bins);
    hp.cqt_bins_per_octave = core_gguf::kv_u32(meta, "basic_pitch.cqt.bins_per_octave", hp.cqt_bins_per_octave);
    hp.cqt_n_octaves = core_gguf::kv_u32(meta, "basic_pitch.cqt.n_octaves", hp.cqt_n_octaves);
    hp.cqt_n_filters = core_gguf::kv_u32(meta, "basic_pitch.cqt.n_filters", hp.cqt_n_filters);
    hp.cqt_n_fft = core_gguf::kv_u32(meta, "basic_pitch.cqt.n_fft", hp.cqt_n_fft);
    hp.cqt_hop = core_gguf::kv_u32(meta, "basic_pitch.cqt.hop", hp.cqt_hop);
    hp.cqt_fmin = core_gguf::kv_f32(meta, "basic_pitch.cqt.fmin", hp.cqt_fmin);
    hp.cqt_bn_scale = core_gguf::kv_f32(meta, "basic_pitch.cqt.bn_scale", hp.cqt_bn_scale);
    hp.cqt_bn_shift = core_gguf::kv_f32(meta, "basic_pitch.cqt.bn_shift", hp.cqt_bn_shift);
    hp.norm_log_eps = core_gguf::kv_f32(meta, "basic_pitch.norm_log_eps", hp.norm_log_eps);
    hp.n_harmonics = core_gguf::kv_u32(meta, "basic_pitch.n_harmonics", hp.n_harmonics);
    {
        // The shifts are the only harmonic parameter the runtime needs; the
        // float harmonics list is carried in the GGUF for provenance.
        const std::vector<float> shifts = core_gguf::kv_f32_array(meta, "basic_pitch.harmonic_shifts");
        if (!shifts.empty()) {
            hp.harmonic_shifts.assign(shifts.size(), 0);
            for (size_t i = 0; i < shifts.size(); i++)
                hp.harmonic_shifts[i] = (int)std::lround(shifts[i]);
        }
    }
    core_gguf::free_metadata(meta);

    if (hp.harmonic_shifts.size() != (size_t)hp.n_harmonics) {
        fprintf(stderr, "basic-pitch: %zu harmonic shifts but n_harmonics=%u\n", hp.harmonic_shifts.size(),
                hp.n_harmonics);
        delete ctx;
        return nullptr;
    }

    ctx->backend = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(ctx->backend, params.n_threads > 0 ? params.n_threads : 1);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "basic-pitch", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;

    bp_weights& w = ctx->w;
    ggml_tensor* t;
    bool ok = true;
    if ((t = core_gguf::require(wl.tensors, "basic_pitch.cqt.kernel_real", "basic-pitch")))
        w.cqt_real = bp_tensor_f32(t);
    else
        ok = false;
    if ((t = core_gguf::require(wl.tensors, "basic_pitch.cqt.kernel_imag", "basic-pitch")))
        w.cqt_imag = bp_tensor_f32(t);
    else
        ok = false;
    if ((t = core_gguf::require(wl.tensors, "basic_pitch.cqt.lowpass", "basic-pitch")))
        w.cqt_lowpass = bp_tensor_f32(t);
    else
        ok = false;
    if ((t = core_gguf::require(wl.tensors, "basic_pitch.cqt.sqrt_lengths", "basic-pitch")))
        w.sqrt_lengths = bp_tensor_f32(t);
    else
        ok = false;

    ok = ok && bp_bind_conv(wl.tensors, "basic_pitch.contour_conv", w.contour_conv, 8, (int)hp.n_harmonics, 3, 39);
    ok = ok && bp_bind_conv(wl.tensors, "basic_pitch.contour_out", w.contour_out, 1, 8, 5, 5);
    ok = ok && bp_bind_conv(wl.tensors, "basic_pitch.note_conv", w.note_conv, 32, 1, 7, 7);
    ok = ok && bp_bind_conv(wl.tensors, "basic_pitch.note_out", w.note_out, 1, 32, 7, 3);
    ok = ok && bp_bind_conv(wl.tensors, "basic_pitch.onset_conv", w.onset_conv, 32, (int)hp.n_harmonics, 5, 5);
    ok = ok && bp_bind_conv(wl.tensors, "basic_pitch.onset_out", w.onset_out, 1, 33, 3, 3);

    const size_t kern = (size_t)hp.cqt_n_filters * (size_t)hp.cqt_n_fft;
    if (ok && (w.cqt_real.size() != kern || w.cqt_imag.size() != kern ||
               w.sqrt_lengths.size() != (size_t)hp.cqt_n_bins || w.cqt_lowpass.empty())) {
        fprintf(stderr, "basic-pitch: CQT front-end tensors have unexpected sizes\n");
        ok = false;
    }
    if (!ok) {
        basic_pitch_free(ctx);
        return nullptr;
    }

    if (params.verbosity >= 1)
        fprintf(stderr, "basic-pitch: loaded %s (sr=%u, window=%u, %u CQT bins, %u harmonics)\n", path, hp.sample_rate,
                hp.audio_n_samples, hp.cqt_n_bins, hp.n_harmonics);
    return ctx;
}

void basic_pitch_free(struct basic_pitch_ctx* ctx) {
    if (!ctx)
        return;
    if (ctx->w_buf)
        core_gguf::release_weight_buffer(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t basic_pitch_sample_rate(const struct basic_pitch_ctx* ctx) {
    return ctx ? ctx->hp.sample_rate : 22050u;
}

uint32_t basic_pitch_window_samples(const struct basic_pitch_ctx* ctx) {
    return ctx ? ctx->hp.audio_n_samples : 43844u;
}

int basic_pitch_transcribe(struct basic_pitch_ctx* ctx, const float* pcm, int n_samples,
                           struct basic_pitch_result* result) {
    if (!ctx || !pcm || n_samples <= 0 || !result)
        return 1;
    std::memset(result, 0, sizeof(*result));

    bp_stitched s;
    if (!bp_run_file(ctx, pcm, n_samples, s))
        return 1;

    const auto frame_notes = bp_output_to_notes(ctx, s);
    result->n_notes = (int)frame_notes.size();
    if (result->n_notes > 0) {
        result->notes = (basic_pitch_note_event*)std::malloc(sizeof(basic_pitch_note_event) * frame_notes.size());
        if (!result->notes)
            return 1;
        for (size_t i = 0; i < frame_notes.size(); i++) {
            const bp_frame_note& fn = frame_notes[i];
            basic_pitch_note_event& ev = result->notes[i];
            ev.start_time = (float)bp_frame_time(ctx->hp, fn.start);
            ev.end_time = (float)bp_frame_time(ctx->hp, fn.end);
            ev.midi_note = fn.midi;
            ev.amplitude = fn.amplitude;
            ev.velocity = (int)std::lround(127.0 * (double)fn.amplitude);
        }
        std::sort(result->notes, result->notes + result->n_notes,
                  [](const basic_pitch_note_event& a, const basic_pitch_note_event& b) {
                      if (a.start_time != b.start_time)
                          return a.start_time < b.start_time;
                      return a.midi_note < b.midi_note;
                  });
    }

    result->n_frames = s.T;
    result->n_freq_contours = (int)ctx->hp.n_freq_bins_contours;
    result->n_freq_notes = (int)ctx->hp.n_freq_bins_notes;
    if (ctx->params.verbosity >= 2 && s.T > 0) {
        auto dup = [](const std::vector<float>& v) -> float* {
            float* p = (float*)std::malloc(v.size() * sizeof(float));
            if (p)
                std::memcpy(p, v.data(), v.size() * sizeof(float));
            return p;
        };
        result->contour = dup(s.contour);
        result->note_head = dup(s.note);
        result->onset_head = dup(s.onset);
    }
    return 0;
}

void basic_pitch_result_free(struct basic_pitch_result* result) {
    if (!result)
        return;
    std::free(result->notes);
    std::free(result->contour);
    std::free(result->note_head);
    std::free(result->onset_head);
    std::memset(result, 0, sizeof(*result));
}

// ─── Diff harness ───────────────────────────────────────────────────────────

static double bp_cosine(const float* a, const float* b, int64_t n) {
    double num = 0.0, na = 0.0, nb = 0.0;
    for (int64_t i = 0; i < n; i++) {
        num += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0)
        return 0.0;
    return num / (std::sqrt(na) * std::sqrt(nb));
}

static double bp_max_abs(const float* a, const float* b, int64_t n) {
    double m = 0.0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

static bool bp_ref_get(const core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end())
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

int basic_pitch_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_22k, int n_samples, int verbosity) {
    basic_pitch_params p = basic_pitch_default_params();
    p.verbosity = 0;
    basic_pitch_ctx* ctx = basic_pitch_init_from_file(model_gguf, p);
    if (!ctx) {
        fprintf(stderr, "basic_pitch_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "basic_pitch_ref", rw)) {
        fprintf(stderr, "basic_pitch_diff: failed to load reference %s\n", ref_gguf);
        basic_pitch_free(ctx);
        return 2;
    }

    const bp_hparams& hp = ctx->hp;
    int n_fail = 0;
    const double COS_MIN = 0.999;

    auto report = [&](const char* stage, const std::vector<float>& mine, const std::vector<float>& ref) {
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = bp_cosine(mine.data(), ref.data(), n);
        const double mad = bp_max_abs(mine.data(), ref.data(), n);
        const bool ok = cos >= COS_MIN && mine.size() == ref.size();
        if (!ok)
            n_fail++;
        if (verbosity >= 1 || !ok)
            fprintf(stderr, "  %-16s %s cos=%.6f max_abs=%.3e  (mine=%zu ref=%zu)\n", stage, ok ? "PASS" : "FAIL", cos,
                    mad, mine.size(), ref.size());
    };

    // The reference dumps window 0. Reproduce the same window here: the leading
    // zero pad from get_audio_input() is part of it.
    const int W = (int)hp.audio_n_samples;
    const int lead = (int)hp.overlapping_frames * (int)hp.fft_hop / 2; // 3840
    std::vector<float> window((size_t)W, 0.0f);
    const int copy = std::min(W - lead, n_samples);
    if (copy > 0)
        std::memcpy(window.data() + lead, pcm_22k, (size_t)copy * sizeof(float));

    fprintf(stderr, "basic-pitch diff (n_samples=%d, window=%d, cqt_bins=%u, harmonics=%u):\n", n_samples, W,
            hp.cqt_n_bins, hp.n_harmonics);

    {
        std::vector<float> r;
        if (bp_ref_get(rw, "audio_window0", r))
            report("audio_window0", window, r);
        else
            fprintf(stderr, "  audio_window0    SKIP (absent from reference)\n");
    }

    bp_window_out wo;
    if (!bp_forward_window(ctx, window.data(), W, wo)) {
        fprintf(stderr, "basic_pitch_diff: forward pass failed\n");
        core_gguf::free_weights(rw);
        basic_pitch_free(ctx);
        return 2;
    }

    std::vector<float> r;
    if (bp_ref_get(rw, "cqt_magnitude", r))
        report("cqt_magnitude", wo.cqt, r);
    if (bp_ref_get(rw, "normalized_log", r))
        report("normalized_log", wo.normlog, r);
    if (bp_ref_get(rw, "harmonic_stack", r)) {
        // Reference layout is (T, 264, 8); ours is channel-major.
        const int T = wo.T, NF = (int)hp.n_freq_bins_contours, NH = (int)hp.n_harmonics;
        std::vector<float> tfc((size_t)T * (size_t)NF * (size_t)NH);
        for (int c = 0; c < NH; c++)
            for (int t = 0; t < T; t++)
                for (int f = 0; f < NF; f++)
                    tfc[((size_t)t * NF + f) * NH + c] = wo.hstack[((size_t)c * T + t) * NF + f];
        report("harmonic_stack", tfc, r);
    }
    if (bp_ref_get(rw, "head_contour", r))
        report("head_contour", wo.contour, r);
    if (bp_ref_get(rw, "head_note", r))
        report("head_note", wo.note, r);
    if (bp_ref_get(rw, "head_onset", r))
        report("head_onset", wo.onset, r);

    // Full-file stitched heads, which also exercise the windowing and the
    // unwrap trim — a per-window pass says nothing about those.
    bp_stitched st;
    if (bp_run_file(ctx, pcm_22k, n_samples, st)) {
        if (bp_ref_get(rw, "unwrapped_contour", r))
            report("unwrap_contour", st.contour, r);
        if (bp_ref_get(rw, "unwrapped_note", r))
            report("unwrap_note", st.note, r);
        if (bp_ref_get(rw, "unwrapped_onset", r))
            report("unwrap_onset", st.onset, r);
    }

    core_gguf::free_weights(rw);
    basic_pitch_free(ctx);
    fprintf(stderr, "basic-pitch diff: %s (%d failing stage%s)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail,
            n_fail == 1 ? "" : "s");
    return n_fail == 0 ? 0 : 1;
}
