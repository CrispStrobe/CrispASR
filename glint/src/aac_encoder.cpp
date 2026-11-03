// glint - AAC-LC encoder core (long + short blocks, CBR-average, ADTS)
// MIT License - Clean-room implementation
//
// Window scheduling needs one block of lookahead (a LONG_START must be
// emitted BEFORE the transient frame), so the encoder holds one block:
// glint_aac_encode(block t) emits the frame covering blocks (t-2, t-1),
// total encoder delay 2048 samples. The first call emits a silence frame.
// glint_aac_flush emits the two tail frames (held block + final overlap).
//
// Rate control is a bit-debt controller: each frame's budget is the running
// target minus bits already spent, clamped to the decoder's per-channel input
// buffer (6144 bits). Long-run average hits the requested bitrate exactly;
// individual ADTS frames vary (the format is variable-length by design —
// there is no MP3-style back-pointer, so no desync hazard).

#include "glint/glint.h"

#include "aac_coder.hpp"
#include "aac_mdct.hpp"
#include "aac_psy.hpp"
#include "aac_tables.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

using namespace glint::aac;
using namespace glint::aac_tables;

constexpr int kAdtsHeaderBits = 56;   // protection_absent = 1
#ifdef GLINT_SMALL_BUFFERS
// Two frames at the 6144-bit/ch cap: 2 * (7 + 2*6144/8) = 3086 bytes.
constexpr int kMaxOutBytes = 3328;
#else
constexpr int kMaxOutBytes = 8192;    // flush emits two frames back-to-back
#endif
constexpr int kDecoderBufBits = 6144; // per-channel input buffer cap (ISO)

int samplerate_index(int sr) {
    for (int i = 0; i < kNumSampleRates; i++) {
        if (kSampleRates[i] == sr) return i;
    }
    return -1;
}

// Bandwidth cutoff in Hz for a given per-channel bitrate (phase-1 heuristic,
// to be replaced by the psy model; roughly tracks fdk-aac's LC defaults).
double cutoff_hz(int bits_per_sec_per_ch) {
    if (bits_per_sec_per_ch >= 96000) return 20000.0;
    if (bits_per_sec_per_ch >= 64000) return 16000.0;
    if (bits_per_sec_per_ch >= 48000) return 14000.0;
    if (bits_per_sec_per_ch >= 32000) return 12000.0;
    return bits_per_sec_per_ch * 0.375;
}

}  // namespace

struct glint_aac_context {
    int sample_rate;
    int sr_index;
    int num_channels;
    int bitrate_bps;
    int max_sfb_long;
    int max_sfb_short;
    int quality;                    // glint_quality
    int vbr;                        // constant-quality VBR
    int vbr_gain_floor;             // anchor-gain floor from vbr_quality
    int force_short;                // GLINT_AAC_FORCE_SHORT diagnostic

    // Block pipeline: the frame emitted on each call covers (blk0, blk1).
    PcmT blk0[2][kAacFrameLen];     // older block
    PcmT blk1[2][kAacFrameLen];     // newer block
    int attack0, attack1;           // transient flags for blk0 / blk1
    int pos0, pos1;                 // attack sub-block (0..7) within the block
    int prev_short;
    int frames_since_short;         // conservative-path window after transients

    // Transient detector state (per channel)
    double hp_last[2];              // last input sample of the previous block
    double base_e[2];               // rolling sub-block energy baseline

    AacBandLayout layout;           // current frame's common band layout
    SpecT spec[2][kAacFrameLen];    // coded-order spectra (M/S bands transformed)
    AacChannelPlan plan[2];
    uint8_t ms_used[kMaxSfb];       // per-band M/S flags (stereo)
    int ms_present;                 // 0 = none, 1 = per-band flags, 2 = all
    // Masks are ENERGIES (huge in the Q-scaled integer domain) — always
    // float storage, never SpecT (int32 storage truncated them to garbage
    // and froze the shaping loop on the no-FPU profile).
    float mask[2][kMaxSfb];

    double bits_per_frame;
    double target_acc;
    double bits_spent;

    double emax_run;                // running max band energy (ATH calibration)
    double emax_run_short;          // separate domain for short-frame masks

    int frames_in;                  // input blocks consumed
    int flushed;

    uint8_t out[kMaxOutBytes];
};

namespace {

// NMR-driven noise shaping. Ported from the MP3 nmr_outer_loop but with
// AAC-specific structure found by measurement (see PLAN.md A1.1): shape to
// 0.125x the mask, bidirectional scalefactor offsets, per-band ceiling at
// the MASK, two amplification tiers, 2.5x total-noise backstop, and
// shape_bits = unshaped spend + HALF the leftover budget.
constexpr double kShapeTarget = 0.125;
constexpr double kNoiseGuard = 2.5;
// Offsets are bidirectional (+ = finer, - = coarser than the anchor); the
// +-30 bound keeps any two coded bands' scalefactors within the +-60 dpcm
// range of the scalefactor codebook.
constexpr int kMaxSfOffset = 30;

void shape_channel(glint_aac_context* c, int chn, int budget);

// Band energy in the (possibly scaled) spec domain; per-coefficient work is
// integer on the no-FPU profile.
double band_energy(const SpecT* spec, const AacBandLayout& L, int b) {
#ifdef GLINT_AAC_INT
    int64_t acc = 0;
    for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
        int64_t v = spec[i];
        acc += v * v;
    }
    return static_cast<double>(acc);
#else
    double acc = 0.0;
    for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
        acc += spec[i] * spec[i];
    }
    return acc;
#endif
}

// Per-band M/S decision + in-place transform of c->spec, and the coded-domain
// shaping masks for both channels. maskL/maskR may be null (short frames):
// the product rule then degenerates to the pure energy rule eM*eS < eL*eR.
// In M/S bands BOTH channels shape against t = min(maskL, maskR): noise in M
// or S lands in decoded L and R (l = m+s, r = m-s), so the side channel must
// never be shaped against a mask the quieter original channel does not have.
void decide_ms(glint_aac_context* c, const double* maskL, const double* maskR,
               bool masked_rule) {
    const AacBandLayout& L = c->layout;
    int n_ms = 0;
    for (int b = 0; b < L.num_bands; b++) {
#ifdef GLINT_AAC_INT
        // Integer energies (spec is Q3 int32; >>4 pre-shift keeps the
        // 96-line band sums inside int64), double only for the final
        // 4-value product rule (per band, not per coefficient).
        // Full-scale int64 energies (Parseval bounds the frame total at
        // ~2^59); m/s energies via the identity eM+eS=(eL+eR)/2 pattern is
        // not needed — direct sums stay exact.
        int64_t iL = 0, iR = 0, iM = 0, iS = 0;
        for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
            int64_t l = c->spec[0][i], r = c->spec[1][i];
            int64_t m = (l + r) / 2, s2 = (l - r) / 2;
            iL += l * l;
            iR += r * r;
            iM += m * m;
            iS += s2 * s2;
        }
        double eL = static_cast<double>(iL), eR = static_cast<double>(iR);
        double eM = static_cast<double>(iM), eS = static_cast<double>(iS);
        double mask_to_e = 1.0;  // masks and energies share the Q3^2 domain
#else
        double eL = 0, eR = 0, eM = 0, eS = 0;
        for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
            double l = c->spec[0][i], r = c->spec[1][i];
            double m = 0.5 * (l + r), s = 0.5 * (l - r);
            eL += l * l;
            eR += r * r;
            eM += m * m;
            eS += s * s;
        }
        double mask_to_e = 1.0;
#endif
        // tm stays in the mask (spec^2) domain for storage; t is rescaled
        // into the energy domain only for the product-rule comparison.
        // masked_rule=false (short frames): the energy-only rule decides —
        // the masked rule measured -0.10 ODG on transient-dense music —
        // but the masks still flow to the allocator via c->mask.
        double tm = 0.0;
        if (maskL && maskR) tm = maskL[b] < maskR[b] ? maskL[b] : maskR[b];
        double t = masked_rule ? tm / mask_to_e : 0.0;
        if ((t + eM) * (t + eS) < (t + eL) * (t + eR)) {
            c->ms_used[b] = 1;
            n_ms++;
            for (int i = L.offset[b]; i < L.offset[b + 1]; i++) {
#ifdef GLINT_AAC_INT
                int32_t l = c->spec[0][i], r = c->spec[1][i];
                c->spec[0][i] = static_cast<int32_t>((static_cast<int64_t>(l) + r) / 2);
                c->spec[1][i] = static_cast<int32_t>((static_cast<int64_t>(l) - r) / 2);
#else
                double l = c->spec[0][i], r = c->spec[1][i];
                c->spec[0][i] = 0.5 * (l + r);
                c->spec[1][i] = 0.5 * (l - r);
#endif
            }
            c->mask[0][b] = static_cast<float>(tm);
            c->mask[1][b] = static_cast<float>(tm);
        } else {
            c->ms_used[b] = 0;
            c->mask[0][b] = static_cast<float>(maskL ? maskL[b] : 0.0);
            c->mask[1][b] = static_cast<float>(maskR ? maskR[b] : 0.0);
        }
    }
    c->ms_present = (n_ms == 0) ? 0 : (n_ms == L.num_bands) ? 2 : 1;
    static const bool msdbg = getenv("GLINT_AAC_MS_DEBUG") != nullptr;
    if (msdbg) {
        static long total = 0, frames = 0;
        total += n_ms;
        frames++;
        fprintf(stderr, "msdbg frames=%ld total_ms_bands=%ld\n", frames, total);
    }
}

// Transient detector: 8 sub-block energies of the first-difference signal;
// attack when one jumps 8x over the rolling baseline (with a silence floor).
bool detect_attack(const PcmT* x, double* hp_last, double* base_e, int* pos) {
#ifdef GLINT_AAC_INT
    // Integer form: identical structure, int64 energies; base update
    // (7*base + 3*e)/10 approximates the 0.7/0.3 EMA.
    int64_t prev = static_cast<int64_t>(*hp_last);
    int64_t base = static_cast<int64_t>(*base_e);
    bool attack = false;
    int p = 0;
    for (int w = 0; w < 8; w++) {
        int64_t e = 0;
        for (int n = 0; n < 128; n++) {
            int64_t d = static_cast<int64_t>(x[128 * w + n]) - prev;
            prev = x[128 * w + n];
            e += d * d;
        }
        if (!attack && e > 8 * base && e > 1000000) {
            attack = true;
            p = w;
        }
        base = base > e ? (7 * base + 3 * e) / 10 : e;
    }
    *hp_last = static_cast<double>(prev);
    *base_e = static_cast<double>(base);
    *pos = p;
    return attack;
#else
    constexpr double kRatio = 8.0;
    constexpr double kFloor = 1e6;  // diff-energy floor at int16 scale
    double prev = *hp_last;
    double base = *base_e;
    bool attack = false;
    int p = 0;
    for (int w = 0; w < 8; w++) {
        double e = 0.0;
        for (int n = 0; n < 128; n++) {
            double d = x[128 * w + n] - prev;
            prev = x[128 * w + n];
            e += d * d;
        }
        if (!attack && e > kRatio * base && e > kFloor) {
            attack = true;
            p = w;
        }
        base = base > e ? 0.7 * base + 0.3 * e : e;  // fast rise, slow fall
    }
    *hp_last = prev;
    *base_e = base;
    *pos = p;
    return attack;
#endif
}

// One frame: window-decide, MDCT, M/S, fit, shape, emit ADTS at out. Returns
// frame bytes. next_attack = transient flag of the block AFTER blk1.
int emit_frame(glint_aac_context* c, int next_attack, uint8_t* out) {
    const int ch = c->num_channels;

    // ---- window decision ----
    int short_f = c->attack0 || c->attack1 || c->force_short;
    int next_short = c->attack1 || next_attack;
    if (!short_f && c->prev_short && next_short) short_f = 1;  // bridge
    int seq = short_f ? kSeqShort
              : next_short ? kSeqStart
              : c->prev_short ? kSeqStop
                              : kSeqLong;

    if (seq == kSeqShort) {
        // Group split at the attack window: [0,s) [s,s+2) [s+2,8)
        int s = c->attack1 ? (c->pos1 + 4 > 7 ? 7 : c->pos1 + 4)
                : c->attack0 ? (c->pos0 - 4 < 0 ? 0 : c->pos0 - 4)
                             : 0;
        uint8_t glen[3];
        int ng = 0;
        if (s > 0) glen[ng++] = static_cast<uint8_t>(s);
        int mid = (s + 2 > 8) ? 8 - s : 2;
        glen[ng++] = static_cast<uint8_t>(mid);
        if (s + mid < 8) glen[ng++] = static_cast<uint8_t>(8 - s - mid);
        aac_make_layout(c->sr_index, kSeqShort, c->max_sfb_short, glen, ng, &c->layout);
    } else {
        aac_make_layout(c->sr_index, seq, c->max_sfb_long, nullptr, 1, &c->layout);
    }
    c->prev_short = short_f;
    c->frames_since_short = short_f ? 0 : c->frames_since_short + 1;
    const AacBandLayout& L = c->layout;

    // ---- MDCT into coded order ----
    for (int chn = 0; chn < ch; chn++) {
        if (seq == kSeqShort) {
            SpecT natural[kAacFrameLen];
            aac_mdct_frame(seq, c->blk0[chn], c->blk1[chn], natural);
            aac_reorder_short(natural, L, c->sr_index, c->spec[chn]);
        } else {
            aac_mdct_frame(seq, c->blk0[chn], c->blk1[chn], c->spec[chn]);
        }
    }

    // ---- budget ----
    // VBR: no debt controller — every frame may spend up to the decoder
    // buffer cap; the constant-quality gain floor is what limits spend.
    double avail;
    if (c->vbr) {
        avail = static_cast<double>(kDecoderBufBits) * ch;
    } else {
        avail = c->target_acc - c->bits_spent;
        double cap = static_cast<double>(kDecoderBufBits) * ch;
        if (avail > cap) avail = cap;
        double floor_bits = 0.35 * c->bits_per_frame;
        if (avail < floor_bits) avail = floor_bits;
    }

    const int info_bits = (seq == kSeqShort) ? 15 : 11;  // ics_info
    int fixed_overhead = kAdtsHeaderBits + 3 /*END*/ + 7 /*worst-case align*/;
    if (ch == 2) {
        fixed_overhead += 3 + 4 + 1 + info_bits + 2;  // CPE hdr + ics_info + ms_mask
    } else {
        fixed_overhead += 3 + 4 + info_bits;
    }

    // ---- masks + M/S ----
    // Long-family frames get psy masks; short frames use the energy-only M/S
    // rule and are not shaped (the MP3 lesson: attack-dominated masks
    // mislead on transition frames; short windows already localize noise).
    const bool shape = (c->quality >= GLINT_QUALITY_NORMAL) && seq == kSeqLong;
    double maskL[kMaxSfb], maskR[kMaxSfb];
    const double* ml = nullptr;
    const double* mr = nullptr;
    // Tonality-aware mask offsets at low per-channel rates (the MP3 gate:
    // <= 96 kbps/ch), where the flat -14 dB offset over-protects noise-like
    // bands and under-protects tonal ones.
    const bool tonal = (c->bitrate_bps / ch) <= 96000;
    if (seq != kSeqShort) {
        double e0 = aac_compute_masks(c->spec[0], c->sr_index, L.num_bands,
                                      c->emax_run, maskL, tonal);
        if (e0 > c->emax_run) c->emax_run = e0;
        ml = maskL;
        if (ch == 2) {
            double e1 = aac_compute_masks(c->spec[1], c->sr_index, L.num_bands,
                                          c->emax_run, maskR, tonal);
            if (e1 > c->emax_run) c->emax_run = e1;
            mr = maskR;
        }
    } else if (c->quality >= GLINT_QUALITY_NORMAL) {
        // Short-frame masks (per-group spreading): enables the masked M/S
        // rule and per-(group,band) allocation on transient frames.
        double e0 = aac_compute_masks_short(c->spec[0], L, c->sr_index,
                                            c->emax_run_short, maskL);
        if (e0 > c->emax_run_short) c->emax_run_short = e0;
        ml = maskL;
        if (ch == 2) {
            double e1 = aac_compute_masks_short(c->spec[1], L, c->sr_index,
                                                c->emax_run_short, maskR);
            if (e1 > c->emax_run_short) c->emax_run_short = e1;
            mr = maskR;
        }
    }

    // TNS BEFORE M/S: the decoder recombines mid/side first and then runs
    // each channel's TNS synthesis on the OUTPUT channel, so the encoder
    // must analyze/filter the original L/R (masks above are pre-filter).
    // Its bits are part of the exact ICS cost the rate search sees.
    //
    // Mask compensation: the fit/shaping run on the FILTERED spectrum, but
    // the masks describe the unfiltered domain. The decoder's 1/A synthesis
    // amplifies filtered-domain noise by the same local gain it restores to
    // the signal, so scale each in-region band's mask by the band's energy
    // ratio E_filtered/E_unfiltered (the filter is short against the band
    // width, so the local approximation holds). Without this, TNS measured
    // as a regression on every metric (ODG -0.87 -> -1.09 at 128k).
    for (int chn = 0; chn < ch; chn++) {
        c->plan[chn].tns.active = 0;
        static const bool no_tns = getenv("GLINT_AAC_NO_TNS") != nullptr;
        if (seq != kSeqShort && !no_tns) {
            double e_unf[kMaxSfb];
            for (int b = 0; b < L.num_bands; b++) {
                e_unf[b] = band_energy(c->spec[chn], L, b);
            }
            aac_tns_analyze(c->spec[chn], L, c->sr_index, &c->plan[chn].tns);
            if (c->plan[chn].tns.active) {
                double* mask = (chn == 0) ? maskL : maskR;
                for (int b = 0; b < L.num_bands; b++) {
                    if (e_unf[b] <= 0.0) continue;
                    double ratio = band_energy(c->spec[chn], L, b) / e_unf[b];
                    if (ratio < 1e-4) ratio = 1e-4;
                    if (ratio > 1e4) ratio = 1e4;
                    mask[b] *= ratio;
                }
            }
        }
    }

    if (ch == 2) {
        decide_ms(c, ml, mr, seq != kSeqShort);
        if (c->ms_present == 1) fixed_overhead += L.num_bands;  // ms_used bits
    } else {
        for (int b = 0; b < L.num_bands; b++) {
            c->mask[0][b] = static_cast<float>(ml ? maskL[b] : 0.0);
        }
    }

    int spend = static_cast<int>(avail) - fixed_overhead;
    if (spend < 64) spend = 64;

    // ---- fit (+shape) ----
    if (ch == 2) {
        double e0 = 0, e1 = 0;
        for (int i = 0; i < L.num_lines; i++) {
            e0 += c->spec[0][i] * c->spec[0][i];
            e1 += c->spec[1][i] * c->spec[1][i];
        }
        double share = (e0 + e1 > 0) ? e0 / (e0 + e1) : 0.5;
        if (share < 0.3) share = 0.3;
        if (share > 0.7) share = 0.7;
        const int gf = c->vbr ? c->vbr_gain_floor : 0;
        int budget0 = c->vbr ? spend / 2 : static_cast<int>(spend * share);
        // Distortion-controlled allocation at normal/best CBR: per-band
        // noise targets ~ mask^alpha * k with k bisected to the budget.
        // Conservative walk instead when (a) close after a transient — the
        // masks mislead on decay tails (castanets ODG -0.01 -> -0.23 when
        // allocated directly there; the START/STOP lesson again), or (b)
        // below ~56 kbps/ch where the walk measured better (64k mono).
        const bool rate_ok = (c->bitrate_bps / ch) >= 56000;
        // Hard post-transient gate: for < 26 frames after a short the WALK
        // is the better tool (a graduated alpha derate was tried and lost
        // everywhere — a derated allocator does nothing, the walk works).
        const bool direct = shape && !c->vbr && rate_ok &&
                            c->frames_since_short >= 26;
        // Short frames: per-group masked allocation (no walk exists there;
        // the alternative is the flat budget fit).
        const bool direct_s = (c->quality >= GLINT_QUALITY_NORMAL) &&
                              !c->vbr && seq == kSeqShort && rate_ok;
        if (direct || direct_s) {
            aac_fit_channel_masked(c->spec[0], L, c->mask[0], budget0, 1.0,
                                   &c->plan[0]);
        } else {
            aac_fit_channel(c->spec[0], L, budget0, nullptr, -1, gf, &c->plan[0]);
            if (shape) shape_channel(c, 0, budget0);
        }
        int budget1 = spend - c->plan[0].ics_bits;  // leftover flows to ch 1
        if (direct || direct_s) {
            aac_fit_channel_masked(c->spec[1], L, c->mask[1], budget1, 1.0,
                                   &c->plan[1]);
        } else {
            aac_fit_channel(c->spec[1], L, budget1, nullptr, -1, gf, &c->plan[1]);
            if (shape) shape_channel(c, 1, budget1);
        }
    } else {
        if (!c->vbr && (c->bitrate_bps / ch) >= 56000 &&
            ((shape && c->frames_since_short >= 26) ||
             (c->quality >= GLINT_QUALITY_NORMAL && seq == kSeqShort))) {
            aac_fit_channel_masked(c->spec[0], L, c->mask[0], spend, 1.0,
                                   &c->plan[0]);
        } else {
            aac_fit_channel(c->spec[0], L, spend, nullptr, -1,
                            c->vbr ? c->vbr_gain_floor : 0, &c->plan[0]);
            if (shape) shape_channel(c, 0, spend);
        }
    }

    // ---- emit raw_data_block at out+7, then prepend the ADTS header ----
    AacBitWriter bw(out + 7, kMaxOutBytes / 2 - 7);
    if (ch == 2) {
        bw.put(1, 3);                       // id_syn_ele CPE
        bw.put(0, 4);                       // element_instance_tag
        bw.put(1, 1);                       // common_window
        aac_write_ics_info(bw, L);
        bw.put(c->ms_present, 2);           // ms_mask_present
        if (c->ms_present == 1) {
            for (int b = 0; b < L.num_bands; b++) bw.put(c->ms_used[b], 1);
        }
        aac_write_ics_body(bw, c->plan[0], L, false);
        aac_write_ics_body(bw, c->plan[1], L, false);
    } else {
        bw.put(0, 3);                       // id_syn_ele SCE
        bw.put(0, 4);                       // element_instance_tag
        aac_write_ics_body(bw, c->plan[0], L, true);
    }
    bw.put(7, 3);                           // id_syn_ele END
    bw.byte_align();

    int frame_len = bw.bytes() + 7;

    // ADTS header (MPEG-4, AAC-LC, protection absent)
    int profile = 1;  // AAC-LC = audio object type 2, coded as 2-1
    out[0] = 0xFF;
    out[1] = 0xF1;
    out[2] = static_cast<uint8_t>((profile << 6) | (c->sr_index << 2) |
                                  ((ch >> 2) & 1));
    out[3] = static_cast<uint8_t>(((ch & 3) << 6) | ((frame_len >> 11) & 3));
    out[4] = static_cast<uint8_t>((frame_len >> 3) & 0xFF);
    out[5] = static_cast<uint8_t>(((frame_len & 7) << 5) | 0x1F);
    out[6] = 0xFC;

    c->target_acc += c->bits_per_frame;
    c->bits_spent += 8.0 * frame_len;
    return frame_len;
}

// Advance the block pipeline with `cur` (or silence when null).
void push_block(glint_aac_context* c, const PcmT cur[2][kAacFrameLen],
                int attack_cur, int pos_cur) {
    for (int chn = 0; chn < c->num_channels; chn++) {
        std::memcpy(c->blk0[chn], c->blk1[chn], sizeof(c->blk0[chn]));
        if (cur) {
            std::memcpy(c->blk1[chn], cur[chn], sizeof(c->blk1[chn]));
        } else {
            std::memset(c->blk1[chn], 0, sizeof(c->blk1[chn]));
        }
    }
    c->attack0 = c->attack1;
    c->pos0 = c->pos1;
    c->attack1 = attack_cur;
    c->pos1 = pos_cur;
}

const uint8_t* encode_common(glint_aac_context* c,
                             const PcmT cur[2][kAacFrameLen], int* out_size) {
    int attack_cur = 0, pos_cur = 0;
    for (int chn = 0; chn < c->num_channels; chn++) {
        int p = 0;
        bool a = detect_attack(cur[chn], &c->hp_last[chn], &c->base_e[chn], &p);
        if (a && (!attack_cur || p < pos_cur)) pos_cur = p;
        attack_cur |= a;
    }
    int n = emit_frame(c, attack_cur, c->out);
    push_block(c, cur, attack_cur, pos_cur);
    c->frames_in++;
    *out_size = n;
    return c->out;
}

}  // namespace

extern "C" {

int glint_version(void) {
    // 0.8.1: glint_decode_audio's MP3 path is gapless (Xing/Info frame
    // skipped, LAME delay/padding trimmed). Callers that line decoded audio
    // up against anything else should require >= 0.8.1 — an older glint
    // returns every tagged MP3 a frame plus ~1105 samples late.
    return (0 << 16) | (10 << 8) | 0;  // 0.10.0
}

glint_aac_t glint_aac_create(const struct glint_aac_config* cfg) {
    if (!cfg) return nullptr;
    int sri = samplerate_index(cfg->sample_rate);
    if (sri < 0) return nullptr;
    if (cfg->num_channels < 1 || cfg->num_channels > 2) return nullptr;
    if (cfg->bitrate < 8 || cfg->bitrate > 800) return nullptr;

    glint_aac_context* c = new (std::nothrow) glint_aac_context();
    if (!c) return nullptr;
    std::memset(c, 0, sizeof(*c));
    c->sample_rate = cfg->sample_rate;
    c->sr_index = sri;
    c->num_channels = cfg->num_channels;
    c->bitrate_bps = cfg->bitrate * 1000;
    c->quality = (cfg->quality >= GLINT_QUALITY_BEST) ? GLINT_QUALITY_BEST
                 : (cfg->quality == GLINT_QUALITY_NORMAL) ? GLINT_QUALITY_NORMAL
                                                          : GLINT_QUALITY_SPEED;
    c->vbr = cfg->vbr ? 1 : 0;
    if (c->vbr) {
        // Constant-quality anchor-gain floors, V0 (finest) .. V9 (coarsest).
        // Each step is ~1.5 dB of quantization noise; calibrated against the
        // CBR anchors the rate search settles on for the reference clips
        // (see PLAN.md A4). GLINT_AAC_VBR_GAIN overrides for experiments.
        // Calibrated on the reference speech clip (V0 ~300k, V4 ~160k,
        // V9 ~50k stereo; music runs lower/higher with content — that's VBR).
        static const int kVbrGain[10] = {
            133, 137, 140, 143, 146, 149, 152, 155, 158, 162
        };
        int q = cfg->vbr_quality;
        if (q < 0) q = 0;
        if (q > 9) q = 9;
        c->vbr_gain_floor = kVbrGain[q];
        if (const char* e = getenv("GLINT_AAC_VBR_GAIN")) {
            int v = atoi(e);
            if (v > 0 && v < 256) c->vbr_gain_floor = v;
        }
    }
    c->force_short = getenv("GLINT_AAC_FORCE_SHORT") != nullptr;
    c->bits_per_frame = static_cast<double>(c->bitrate_bps) * kAacFrameLen / c->sample_rate;

    // max_sfb from the bandwidth cutoff: first band whose start line reaches
    // the cutoff ends the coded region.
    double cut = cutoff_hz(c->bitrate_bps / c->num_channels);
    double hz_per_line = 0.5 * c->sample_rate / kAacFrameLen;
    int cut_line = static_cast<int>(cut / hz_per_line);
    {
        const uint16_t* swb = kSwbOffsetLong[sri];
        int nswb = kNumSwbLong[sri];
        int msfb = nswb;
        for (int b = 1; b <= nswb; b++) {
            if (swb[b] >= cut_line) { msfb = b; break; }
        }
        if (msfb < 4) msfb = 4;
        c->max_sfb_long = msfb;
    }
    {
        const uint16_t* swb = kSwbOffsetShort[sri];
        int nswb = kNumSwbShort[sri];
        int cut_line_s = (cut_line + 7) / 8;
        int msfb = nswb;
        for (int b = 1; b <= nswb; b++) {
            if (swb[b] >= cut_line_s) { msfb = b; break; }
        }
        if (msfb < 3) msfb = 3;
        c->max_sfb_short = msfb;
    }
    return c;
}

int glint_aac_samples_per_frame(glint_aac_t c) {
    (void)c;
    return kAacFrameLen;
}

const uint8_t* glint_aac_encode(glint_aac_t c, const int16_t** channel_data, int* out_size) {
    if (!c || !channel_data || !out_size || c->flushed) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
    PcmT cur[2][kAacFrameLen];
    for (int chn = 0; chn < c->num_channels; chn++) {
        for (int i = 0; i < kAacFrameLen; i++) {
            cur[chn][i] = static_cast<PcmT>(channel_data[chn][i]);
        }
    }
    return encode_common(c, cur, out_size);
}

const uint8_t* glint_aac_encode_float(glint_aac_t c, const float** channel_data, int* out_size) {
    if (!c || !channel_data || !out_size || c->flushed) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
    PcmT cur[2][kAacFrameLen];
    for (int chn = 0; chn < c->num_channels; chn++) {
        for (int i = 0; i < kAacFrameLen; i++) {
            double v = static_cast<double>(channel_data[chn][i]) * 32768.0;
#ifdef GLINT_SMALL_BUFFERS
            if (v > 32767.0) v = 32767.0;
            if (v < -32768.0) v = -32768.0;
#endif
            cur[chn][i] = static_cast<PcmT>(v);
        }
    }
    return encode_common(c, cur, out_size);
}

const uint8_t* glint_aac_flush(glint_aac_t c, int* out_size) {
    if (!c || !out_size || c->flushed || c->frames_in == 0) {
        if (out_size) *out_size = 0;
        return nullptr;
    }
    c->flushed = 1;
    // Two tail frames: (blk0, blk1) and (blk1, silence).
    int n1 = emit_frame(c, 0, c->out);
    push_block(c, nullptr, 0, 0);
    int n2 = emit_frame(c, 0, c->out + n1);
    *out_size = n1 + n2;
    return c->out;
}

void glint_aac_destroy(glint_aac_t c) {
    delete c;
}

}  // extern "C"

namespace {

void shape_channel(glint_aac_context* c, int chn, int budget) {
    using namespace glint::aac;
    const AacBandLayout& L = c->layout;
    AacChannelPlan& plan = c->plan[chn];
    const SpecT* spec = c->spec[chn];
    const int nb = L.num_bands;

    const float* mask = c->mask[chn];  // coded-domain masks from emit_frame
    double noise[kMaxSfb];
    aac_band_noise(plan, spec, L, noise);

    double r0[kMaxSfb];
    double j_best = 0.0, total0 = 0.0, worst = 0.0;
    for (int b = 0; b < nb; b++) {
        total0 += noise[b];
        r0[b] = (mask[b] > 0.0) ? noise[b] / mask[b] : 0.0;
        j_best += r0[b];
        if (r0[b] > worst) worst = r0[b];
    }
    if (worst <= kShapeTarget) return;

    // CBR: spent + HALF the leftover (the MP3 reservoir lesson). VBR: the
    // budget is the buffer cap, not a rate signal — give shaping spent +25%
    // like MP3's VBR shaping.
    int shape_bits = c->vbr ? plan.ics_bits + plan.ics_bits / 4
                            : plan.ics_bits + (budget - plan.ics_bits) / 2;
    if (shape_bits > budget) shape_bits = budget;

    int off[kMaxSfb] = {0};
    bool was_amped[kMaxSfb] = {false};
    AacChannelPlan best = plan;
    AacChannelPlan cur = plan;
    double cur_noise[kMaxSfb];
    std::memcpy(cur_noise, noise, sizeof(double) * nb);

    const int max_iters = (c->quality >= GLINT_QUALITY_BEST) ? 40 : 16;
    int stall = 0;
    for (int iter = 0; iter < max_iters; iter++) {
        double w = 0.0;
        for (int b = 0; b < nb; b++) {
            if (mask[b] > 0.0) {
                double r = cur_noise[b] / mask[b];
                if (r > w) w = r;
            }
        }
        if (w <= kShapeTarget) break;
        // Narrow rounds: AAC CBR has no reservoir slack, so a wide
        // amplification front forces the gain anchor up several steps at
        // once and every iterate gets rejected.
        double thresh = kShapeTarget > w * 0.5 ? kShapeTarget : w * 0.5;

        bool changed = false;
        for (int b = 0; b < nb; b++) {
            if (mask[b] <= 0.0) continue;
            double r = cur_noise[b] / mask[b];
            // Tier A: any band over the MASK is audible and must be fixed
            // every round. Tier B: below the mask, only near-worst bands.
            if ((r > 1.0 || r >= thresh) && off[b] < kMaxSfOffset) {
                off[b] += 2;  // 2 sf steps = ~3 dB finer
                was_amped[b] = true;
                changed = true;
            } else if (r < kShapeTarget * 0.25 && off[b] > -kMaxSfOffset &&
                       !was_amped[b]) {
                // Over-coded band donates bits by coarsening (hysteresis:
                // never coarsen a band the loop had to rescue).
                off[b] -= 2;
                changed = true;
            }
        }
        if (!changed) break;

        AacChannelPlan cand;
        cand.tns = cur.tns;  // fit reads the caller's TNS decision from the plan
        aac_fit_channel(spec, L, shape_bits, off, cur.fit_gain,
                        c->vbr ? c->vbr_gain_floor : 0, &cand);
        double cand_noise[kMaxSfb];
        aac_band_noise(cand, spec, L, cand_noise);
        double j = 0.0, total = 0.0;
        bool band_ok = true;
        for (int b = 0; b < nb; b++) {
            total += cand_noise[b];
            if (mask[b] > 0.0) {
                double r = cand_noise[b] / mask[b];
                j += r;
                // No band may become audible that wasn't: below the mask
                // noise placement is free — that freedom IS the shaping.
                double ceiling = 1.0 > r0[b] ? 1.0 : r0[b];
                if (r > ceiling * 1.05) band_ok = false;
            }
        }
        cur = cand;
        std::memcpy(cur_noise, cand_noise, sizeof(double) * nb);
        if (j < j_best && band_ok && total <= total0 * kNoiseGuard) {
            j_best = j;
            best = cand;
            stall = 0;
        } else if (++stall >= 5) {
            break;
        }
        static const bool dbg = getenv("GLINT_AAC_DEBUG") != nullptr;
        if (dbg) {
            fprintf(stderr, "  it%d: j=%.3g best=%.3g total/0=%.2f g=%d bits=%d/%d w=%.2f\n",
                    iter, j, j_best, total / total0, cand.fit_gain,
                    cand.ics_bits, shape_bits, w);
        }
    }
    plan = best;
}

}  // namespace
