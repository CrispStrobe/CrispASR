// glint - MPEG-1/2 Layer III decoder (PLAN § D1)
// MIT License - Clean-room implementation from ISO 11172-3 / 13818-3.

#include "mp3_decoder.hpp"

#include <cmath>
#include <cstring>

#include "tables.hpp"

namespace glint {
namespace mp3 {

namespace {

// ---------------------------------------------------------------- bits

struct BitReader {
    const uint8_t* data;
    int len;       // bytes
    int pos = 0;   // bit position

    BitReader(const uint8_t* d, int n) : data(d), len(n) {}

    int bits_left() const { return len * 8 - pos; }

    uint32_t get(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; i++) {
            int byte = pos >> 3;
            int bit = 7 - (pos & 7);
            v = (v << 1) | (byte < len ? ((data[byte] >> bit) & 1) : 0);
            pos++;
        }
        return v;
    }

    int get1() {
        int byte = pos >> 3;
        int bit = 7 - (pos & 7);
        int v = byte < len ? ((data[byte] >> bit) & 1) : 0;
        pos++;
        return v;
    }
};

// ------------------------------------------------------------- huffman
// Decode trees built from the ENCODER's code/length tables. Leaves are
// stored as ~value; inner nodes hold two child indices.

struct HuffTree {
    // child[i][b]: next node, or ~packed_value when negative.
    int16_t child[1024][2];
    int nodes = 1;

    HuffTree() { child[0][0] = child[0][1] = 0; }

    void insert(uint32_t code, int nbits, int value) {
        int at = 0;
        for (int i = nbits - 1; i >= 0; i--) {
            int b = (code >> i) & 1;
            if (i == 0) {
                child[at][b] = static_cast<int16_t>(~value);
                return;
            }
            if (child[at][b] == 0) {
                child[at][b] = static_cast<int16_t>(nodes);
                child[nodes][0] = child[nodes][1] = 0;
                nodes++;
            }
            at = child[at][b];
        }
    }

    // Returns the packed value, or -1 on a broken stream.
    int decode(BitReader& br) const {
        int at = 0;
        for (int guard = 0; guard < 24; guard++) {
            int16_t next = child[at][br.get1()];
            if (next < 0) return ~next;
            if (next == 0 && at != 0) return -1;
            at = next;
            if (at == 0) return -1;
        }
        return -1;
    }
};

struct HuffSet {
    HuffTree pair[34];   // indexed by table id (shared code tables filled
                         // once per unique id; see build())
    HuffTree quad_a;     // count1 table A (id 32)
    HuffTree quad_b;     // count1 table B (id 33): 4-bit straight codes
    bool built = false;
};

HuffSet g_huff;

void build_huffman() {
    if (g_huff.built) return;
    // Unique pair-code tables; ids 16-23 share table 16's codes, 24-31
    // share 24 (they differ only in linbits).
    static const int kUnique[] = { 1, 2,  3,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 15, 16, 24 };
    for (int u : kUnique) {
        tables::HuffTable t = tables::get_huff_table(u);
        for (int x = 0; x < t.xlen; x++) {
            for (int y = 0; y < t.xlen; y++) {
                int idx = x * t.xlen + y;
                int nbits = t.hlen[idx];
                if (nbits == 0) continue;
                g_huff.pair[u].insert(tables::get_huff_code(u, idx), nbits,
                                      (x << 5) | y);
            }
        }
    }
    for (int a = 16; a < 24; a++) g_huff.pair[a] = g_huff.pair[16];
    for (int a = 25; a < 32; a++) g_huff.pair[a] = g_huff.pair[24];
    // count1 tables: 16 quads (v,w,x,y packed as 4 bits).
    {
        tables::HuffTable t = tables::get_huff_table(32);
        for (int v = 0; v < 16; v++) {
            int nbits = t.hlen[v];
            g_huff.quad_a.insert(tables::get_huff_code(32, v), nbits, v);
        }
        for (int v = 0; v < 16; v++)
            g_huff.quad_b.insert(static_cast<uint32_t>(v ^ 0xF), 4, v);
    }
    g_huff.built = true;
}

// -------------------------------------------------------------- tables

const int kBitrateV1[15] = { 0,   32,  40,  48,  56,  64,  80, 96,
                             112, 128, 160, 192, 224, 256, 320 };
const int kBitrateV2[15] = { 0,  8,  16, 24, 32,  40,  48, 56,
                             64, 80, 96, 112, 128, 144, 160 };
const int kSampleRateV1[3] = { 44100, 48000, 32000 };

// MPEG-1 scalefactor bit lengths from scalefac_compress.
const int kSlen1[16] = { 0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4 };
const int kSlen2[16] = { 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3 };

// LSF scalefactor band-group sizes [blocknumber][blocktype][group]
// (blocktype 0 = long, 1 = short, 2 = mixed), ISO 13818-3.
const int kLsfNsfb[6][3][4] = {
    { { 6, 5, 5, 5 }, { 9, 9, 9, 9 }, { 6, 9, 9, 9 } },
    { { 6, 5, 7, 3 }, { 9, 9, 12, 6 }, { 6, 9, 12, 6 } },
    { { 11, 10, 0, 0 }, { 18, 18, 0, 0 }, { 15, 18, 0, 0 } },
    { { 7, 7, 7, 0 }, { 12, 12, 12, 0 }, { 6, 15, 12, 0 } },
    { { 6, 6, 6, 3 }, { 12, 9, 9, 6 }, { 6, 12, 9, 6 } },
    { { 8, 8, 5, 0 }, { 15, 12, 9, 0 }, { 6, 18, 9, 0 } },
};

// Alias-reduction butterflies.
const double kCi[8] = { -0.6,    -0.535,  -0.33,   -0.185,
                        -0.095,  -0.041,  -0.0142, -0.0037 };

// ---------------------------------------------------------- transforms

// The transforms below used to call std::cos/std::sin/std::pow inside their
// inner loops — ~230k cos() per stereo frame, which made decode ~30x slower
// than ffmpeg (80% of the time in libm). Every value is now taken from a table
// built ONCE from the very same double expression, and the sums keep their
// order, so the output is bit-identical to the direct formulas.
struct DecodeTables {
    double imdct36[36][18];
    double imdct12[12][6];
    double win_long[36];
    double win_short[12];
    double synth[64][32];
    // |ix|^(4/3) for every value a Layer III stream can code (big_values
    // 15 + linbits 13 -> 8206). Larger (malformed) values fall back to pow.
    static constexpr int kPow43 = 8207;
    double pow43[kPow43];

    DecodeTables() {
        for (int i = 0; i < 36; i++)
            for (int k = 0; k < 18; k++)
                imdct36[i][k] =
                    std::cos(M_PI / 72.0 * (2 * i + 1 + 18) * (2 * k + 1));
        for (int i = 0; i < 12; i++)
            for (int k = 0; k < 6; k++)
                imdct12[i][k] =
                    std::cos(M_PI / 24.0 * (2 * i + 1 + 6) * (2 * k + 1));
        for (int i = 0; i < 36; i++) win_long[i] = std::sin(M_PI / 36.0 * (i + 0.5));
        for (int i = 0; i < 12; i++) win_short[i] = std::sin(M_PI / 12.0 * (i + 0.5));
        for (int i = 0; i < 64; i++)
            for (int k = 0; k < 32; k++)
                synth[i][k] = std::cos((16 + i) * (2 * k + 1) * M_PI / 64.0);
        for (int v = 0; v < kPow43; v++) pow43[v] = std::pow(double(v), 4.0 / 3.0);
    }
};

const DecodeTables& tabs() {
    static const DecodeTables t;  // thread-safe one-time init (C++11)
    return t;
}

// |v|^(4/3), from the table when in range.
inline double pow43(int v) {
    int a = v < 0 ? -v : v;
    return a < DecodeTables::kPow43 ? tabs().pow43[a]
                                    : std::pow(double(a), 4.0 / 3.0);
}

// 36-point IMDCT: x[i] = sum_k X[k] cos(pi/72 (2i+1+18)(2k+1)).
void imdct36(const double* X, double* x) {
    const auto& t = tabs().imdct36;
    for (int i = 0; i < 36; i++) {
        double s = 0;
        for (int k = 0; k < 18; k++) s += X[k] * t[i][k];
        x[i] = s;
    }
}

// 12-point IMDCT for one short window.
void imdct12(const double* X, double* x) {
    const auto& t = tabs().imdct12;
    for (int i = 0; i < 12; i++) {
        double s = 0;
        for (int k = 0; k < 6; k++) s += X[k] * t[i][k];
        x[i] = s;
    }
}

double win_long(int i) { return tabs().win_long[i]; }
double win_short(int i) { return tabs().win_short[i]; }

}  // namespace

// -------------------------------------------------------------- header

int mp3_frame_info(const uint8_t* data, int len, Mp3FrameInfo* info) {
    if (len < 4) return -1;
    if (data[0] != 0xFF || (data[1] & 0xE0) != 0xE0) return -1;
    int version = (data[1] >> 3) & 3;  // 3=MPEG1 2=MPEG2 0=MPEG2.5
    int layer = (data[1] >> 1) & 3;    // 1 = Layer III
    if (version == 1 || layer != 1) return -1;
    int bit_idx = (data[2] >> 4) & 0xF;
    int sr_idx = (data[2] >> 2) & 3;
    if (bit_idx == 0 || bit_idx == 15 || sr_idx == 3) return -1;
    int pad = (data[2] >> 1) & 1;
    int mode = (data[3] >> 6) & 3;

    int mpeg1 = version == 3;
    int sr = kSampleRateV1[sr_idx];
    if (version == 2) sr /= 2;
    if (version == 0) sr /= 4;
    int kbps = mpeg1 ? kBitrateV1[bit_idx] : kBitrateV2[bit_idx];

    info->sample_rate = sr;
    info->channels = mode == 3 ? 1 : 2;
    info->bitrate_kbps = kbps;
    info->mpeg2 = !mpeg1;
    info->samples = mpeg1 ? 1152 : 576;
    info->frame_bytes = (mpeg1 ? 144000 : 72000) * kbps / sr + pad;
    return 0;
}

// --------------------------------------------------------------- class

void Mp3Decoder::init() {
    build_huffman();
    store_len_ = 0;
    std::memset(overlap_, 0, sizeof(overlap_));
    std::memset(synth_v_, 0, sizeof(synth_v_));
    std::memset(scalefac_l_, 0, sizeof(scalefac_l_));
    std::memset(scalefac_s_, 0, sizeof(scalefac_s_));
    synth_off_[0] = synth_off_[1] = 0;
    first_ = 1;
}

void Mp3Decoder::requantize(const GranuleInfo& g, int ch, int sr_unified) {
    const int* sfb_l = tables::get_sfb_long_by_unified(sr_unified);
    const int* sfb_s = tables::get_sfb_short_by_unified(sr_unified);
    const double global = std::exp2(0.25 * (g.global_gain - 210));
    const double sf_mult = g.scalefac_scale ? 1.0 : 0.5;

    if (g.window_switching && g.block_type == 2) {
        // Short (or mixed) blocks.
        int i = 0;
        if (g.mixed_block) {
            // First 36 lines are long bands 0..7 (MPEG-1; 6 for LSF is
            // handled by the band table itself reaching 36 at band 8).
            int band = 0;
            while (i < 36 && band < 21) {
                int end = sfb_l[band + 1];
                double m = global *
                           std::exp2(-sf_mult *
                                     (scalefac_l_[ch][band] +
                                      (g.preflag
                                           ? tables::preemphasis[band]
                                           : 0)));
                for (; i < end && i < 36; i++) {
                    double v = pow43(ix_[ch][i]);
                    xr_[ch][i] = (ix_[ch][i] < 0 ? -v : v) * m;
                }
                band++;
            }
        }
        // Short region: coded order is [sfb][window][line].
        int band = g.mixed_block ? 3 : 0;
        i = g.mixed_block ? 36 : 0;
        for (; band < 13 && i < 576; band++) {
            int width = sfb_s[band + 1] - sfb_s[band];
            for (int w = 0; w < 3; w++) {
                double m =
                    global *
                    std::exp2(-2.0 * g.subblock_gain[w] -
                              sf_mult * scalefac_s_[ch][band][w]);
                for (int k = 0; k < width && i < 576; k++, i++) {
                    double v = pow43(ix_[ch][i]);
                    xr_[ch][i] = (ix_[ch][i] < 0 ? -v : v) * m;
                }
            }
        }
        for (; i < 576; i++) xr_[ch][i] = 0;

        // Reorder short-region lines from [sfb][window][line] to
        // [line-triplet] order the IMDCT consumes.
        double tmp[576];
        int start = g.mixed_block ? 36 : 0;
        std::memcpy(tmp, xr_[ch], sizeof(tmp));
        band = g.mixed_block ? 3 : 0;
        int src = start;
        for (; band < 13; band++) {
            int b0 = sfb_s[band] * 3;
            int width = sfb_s[band + 1] - sfb_s[band];
            if (b0 >= 576) break;
            for (int w = 0; w < 3; w++)
                for (int k = 0; k < width; k++, src++)
                    if (b0 + 3 * k + w < 576 && src < 576)
                        xr_[ch][b0 + 3 * k + w] = tmp[src];
        }
    } else {
        // Long blocks.
        int band = 0;
        for (int i = 0; i < 576; i++) {
            while (band < 21 && i >= sfb_l[band + 1]) band++;
            int sf = band < 21 ? scalefac_l_[ch][band] : 0;
            int pre =
                (g.preflag && band < 21) ? tables::preemphasis[band] : 0;
            double m = global * std::exp2(-sf_mult * (sf + pre));
            double v = pow43(ix_[ch][i]);
            xr_[ch][i] = (ix_[ch][i] < 0 ? -v : v) * m;
        }
    }
}

void Mp3Decoder::stereo_process(const GranuleInfo (*gi)[2], int gr,
                                int mode_ext, int mpeg2, int sr_unified,
                                int intensity_scale) {
    const bool ms = (mode_ext & 2) != 0;
    const bool is = (mode_ext & 1) != 0;
    const GranuleInfo& g = gi[gr][1];
    const int* sfb_l = tables::get_sfb_long_by_unified(sr_unified);
    const int* sfb_s = tables::get_sfb_short_by_unified(sr_unified);

    int is_bound = 576;
    if (is) {
        // Intensity region starts at the RIGHT channel's zero bound,
        // rounded up to a band boundary.
        int nz = nonzero_[1];
        while (nz > 0 && xr_[1][nz - 1] == 0.0) nz--;
        if (g.window_switching && g.block_type == 2) {
            is_bound = nz;  // per-window handling below uses bands
        } else {
            int band = 0;
            while (band < 21 && sfb_l[band] < nz) band++;
            is_bound = sfb_l[band];
        }
    }

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    if (ms) {
        int end = is ? is_bound : 576;
        for (int i = 0; i < end; i++) {
            double m = xr_[0][i], s = xr_[1][i];
            xr_[0][i] = (m + s) * inv_sqrt2;
            xr_[1][i] = (m - s) * inv_sqrt2;
        }
    }

    if (!is) return;

    // Intensity stereo above is_bound: position from the right
    // channel's scalefactors.
    auto apply_is = [&](int i0, int i1, int is_pos) {
        if (!mpeg2) {
            if (is_pos >= 7) {
                if (ms)
                    for (int i = i0; i < i1; i++) {
                        double m = xr_[0][i], s = xr_[1][i];
                        xr_[0][i] = (m + s) * inv_sqrt2;
                        xr_[1][i] = (m - s) * inv_sqrt2;
                    }
                return;
            }
            double r = std::tan(is_pos * M_PI / 12.0);
            for (int i = i0; i < i1; i++) {
                double v = xr_[0][i];
                xr_[0][i] = v * (r / (1 + r));
                xr_[1][i] = v * (1 / (1 + r));
            }
        } else {
            // LSF intensity: io^((is_pos+1)/2) scaling (13818-3).
            if (is_pos == 0) {
                for (int i = i0; i < i1; i++) xr_[1][i] = xr_[0][i];
                return;
            }
            double io = std::exp2(-0.5 * (1 + intensity_scale));
            double k = std::pow(io, (is_pos + 1) / 2);
            for (int i = i0; i < i1; i++) {
                double v = xr_[0][i];
                if (is_pos & 1) {
                    xr_[0][i] = v * k;
                    xr_[1][i] = v;
                } else {
                    xr_[1][i] = v * k;
                }
            }
        }
    };

    if (g.window_switching && g.block_type == 2) {
        // Short blocks: per band/window (post-reorder layout: lines are
        // interleaved by window; process whole bands).
        int start_band = 0;
        // Find first band whose START is >= the right zero bound.
        int nz = is_bound;
        while (start_band < 13 && sfb_s[start_band] * 3 < nz)
            start_band++;
        for (int band = start_band; band < 13; band++) {
            int b0 = sfb_s[band] * 3;
            int b1 = sfb_s[band + 1] * 3;
            if (b0 >= 576) break;
            for (int w = 0; w < 3; w++) {
                int is_pos = band < 12 ? scalefac_s_[1][band][w]
                                       : scalefac_s_[1][11][w];
                // Post-reorder, window w's lines are at b0+3k+w; apply
                // per-line with stride 3.
                if (!mpeg2 && is_pos >= 7) continue;  // keep as-is
                for (int i = b0 + w; i < b1 && i < 576; i += 3)
                    apply_is(i, i + 1, is_pos);
            }
        }
    } else {
        int band = 0;
        while (band < 21 && sfb_l[band + 1] <= is_bound) band++;
        for (; band < 21; band++)
            apply_is(sfb_l[band], sfb_l[band + 1],
                     scalefac_l_[1][band]);
        // sfb21 tail: reuse the last band's position.
        apply_is(sfb_l[21], 576, scalefac_l_[1][20]);
    }
}

void Mp3Decoder::antialias(const GranuleInfo& g, int ch) {
    int sblimit = 32;
    if (g.window_switching && g.block_type == 2)
        sblimit = g.mixed_block ? 2 : 0;
    for (int sb = 1; sb < sblimit; sb++) {
        for (int i = 0; i < 8; i++) {
            double cs = 1.0 / std::sqrt(1.0 + kCi[i] * kCi[i]);
            double ca = kCi[i] * cs;
            double lo = xr_[ch][sb * 18 - 1 - i];
            double hi = xr_[ch][sb * 18 + i];
            xr_[ch][sb * 18 - 1 - i] = lo * cs - hi * ca;
            xr_[ch][sb * 18 + i] = hi * cs + lo * ca;
        }
    }
}

void Mp3Decoder::imdct_granule(const GranuleInfo& g, int ch) {
    for (int sb = 0; sb < 32; sb++) {
        double X[18], out[36];
        for (int k = 0; k < 18; k++) X[k] = xr_[ch][sb * 18 + k];

        int bt = g.window_switching ? g.block_type : 0;
        if (bt == 2 && g.mixed_block && sb < 2) bt = 0;

        if (bt == 2) {
            // Three short IMDCTs, windowed and overlapped at +6 steps.
            std::memset(out, 0, sizeof(out));
            for (int w = 0; w < 3; w++) {
                double Xs[6], xs[12];
                for (int k = 0; k < 6; k++) Xs[k] = X[3 * k + w];
                imdct12(Xs, xs);
                for (int i = 0; i < 12; i++)
                    out[6 + 6 * w + i] += xs[i] * win_short(i);
            }
        } else {
            imdct36(X, out);
            for (int i = 0; i < 36; i++) {
                double w;
                if (bt == 0) {
                    w = win_long(i);
                } else if (bt == 1) {  // start
                    w = i < 18 ? win_long(i)
                               : (i < 24 ? 1.0
                                         : (i < 30 ? win_short(i - 18)
                                                   : 0.0));
                } else {  // stop
                    w = i < 6 ? 0.0
                              : (i < 12 ? win_short(i - 6)
                                        : (i < 18 ? 1.0 : win_long(i)));
                }
                out[i] *= w;
            }
        }

        // Overlap-add + frequency inversion for odd subbands.
        for (int i = 0; i < 18; i++) {
            double v = out[i] + overlap_[ch][sb * 18 + i];
            overlap_[ch][sb * 18 + i] = out[18 + i];
            xr_[ch][sb * 18 + i] =
                (sb & 1) && (i & 1) ? -v : v;
        }
    }
}

void Mp3Decoder::synth_granule(int ch, int n_gran, float* pcm, int nch) {
    // 18 slots of 32 subband samples -> 32 PCM samples each.
    for (int slot = 0; slot < 18; slot++) {
        double s[32];
        for (int sb = 0; sb < 32; sb++)
            s[sb] = xr_[ch][sb * 18 + slot];

        synth_off_[ch] = (synth_off_[ch] - 64) & 1023;
        double* v = synth_v_[ch];
        const auto& cs = tabs().synth;
        for (int i = 0; i < 64; i++) {
            double sum = 0;
            for (int k = 0; k < 32; k++) sum += s[k] * cs[i][k];
            v[(synth_off_[ch] + i) & 1023] = sum;
        }

        for (int j = 0; j < 32; j++) {
            double sum = 0;
            for (int i = 0; i < 16; i++) {
                // U-buffer per ISO: u[64m + j] = v[128m + j],
                // u[64m + 32 + j] = v[128m + 96 + j] (m = i/2).
                int vidx = (synth_off_[ch] + 128 * (i / 2) +
                            ((i & 1) ? 96 : 0) + j) &
                           1023;
                // The table IS the ISO D (synthesis) window.
                sum += tables::analysis_window_d[j + 32 * i] * v[vidx];
            }
            int out_idx = (n_gran * 18 + slot) * 32 + j;
            pcm[out_idx * nch + ch] = static_cast<float>(sum);
        }
    }
}

int Mp3Decoder::decode_frame(const uint8_t* data, int len, float* pcm,
                             Mp3FrameInfo* info_out) {
    Mp3FrameInfo info;
    if (mp3_frame_info(data, len, &info) < 0) return -1;
    if (info.frame_bytes > len) return -1;
    if (info_out) *info_out = info;

    const int mpeg1 = !info.mpeg2;
    const int nch = info.channels;
    const int ngr = mpeg1 ? 2 : 1;
    const int version = (data[1] >> 3) & 3;
    const int sr_idx = (data[2] >> 2) & 3;
    const int mode_ext = (data[3] >> 4) & 3;
    const int mode = (data[3] >> 6) & 3;
    const int crc = !(data[1] & 1);
    // Unified sample-rate index for the encoder's band tables:
    // MPEG-1 -> 0..2 (44.1/48/32), MPEG-2 -> 3..5, 2.5 -> also LSF rows.
    int sr_unified;
    {
        static const int kMap1[3] = { 0, 1, 2 };
        sr_unified = version == 3 ? kMap1[sr_idx] : 3 + sr_idx;
    }

    int side_len = mpeg1 ? (nch == 1 ? 17 : 32) : (nch == 1 ? 9 : 17);
    int header_len = 4 + (crc ? 2 : 0);
    if (info.frame_bytes < header_len + side_len) return -1;

    BitReader sb(data + header_len, side_len);
    int main_data_begin = static_cast<int>(sb.get(mpeg1 ? 9 : 8));
    sb.get(mpeg1 ? (nch == 1 ? 5 : 3) : (nch == 1 ? 1 : 2));  // private
    int scfsi[2][4] = {};
    if (mpeg1)
        for (int ch = 0; ch < nch; ch++)
            for (int b = 0; b < 4; b++) scfsi[ch][b] = sb.get1();

    GranuleInfo gi[2][2];
    for (int gr = 0; gr < ngr; gr++) {
        for (int ch = 0; ch < nch; ch++) {
            GranuleInfo& g = gi[gr][ch];
            g.part2_3_length = static_cast<int>(sb.get(12));
            g.big_values = static_cast<int>(sb.get(9));
            g.global_gain = static_cast<int>(sb.get(8));
            g.scalefac_compress = static_cast<int>(sb.get(mpeg1 ? 4 : 9));
            g.window_switching = sb.get1();
            if (g.window_switching) {
                g.block_type = static_cast<int>(sb.get(2));
                g.mixed_block = sb.get1();
                g.table_select[0] = static_cast<int>(sb.get(5));
                g.table_select[1] = static_cast<int>(sb.get(5));
                g.table_select[2] = 0;
                for (int w = 0; w < 3; w++)
                    g.subblock_gain[w] = static_cast<int>(sb.get(3));
                // Implied regions (ISO): region0 ends at 36 for short,
                // 54 for LSF start/stop, 36 for MPEG-1 start/stop.
                g.region0_count =
                    (!mpeg1 && g.block_type != 2) ? 54 : 36;  // in LINES
                g.region1_count = 576;                        // in LINES
                if (g.block_type == 0) return -2;  // invalid
            } else {
                g.block_type = 0;
                g.mixed_block = 0;
                for (int r = 0; r < 3; r++)
                    g.table_select[r] = static_cast<int>(sb.get(5));
                g.region0_count = static_cast<int>(sb.get(4));
                g.region1_count = static_cast<int>(sb.get(3));
                g.subblock_gain[0] = g.subblock_gain[1] =
                    g.subblock_gain[2] = 0;
            }
            g.preflag = mpeg1 ? sb.get1() : 0;
            g.scalefac_scale = sb.get1();
            g.count1table_select = sb.get1();
        }
    }

    // ------- bit reservoir: append this frame's main data ------------
    int main_len = info.frame_bytes - header_len - side_len;
    if (main_len < 0) return -1;
    // The frame's payload begins main_data_begin bytes BACK in the
    // reservoir. Check availability before consuming.
    if (main_data_begin > store_len_) {
        // Not enough history (stream start / after seek): stash and
        // report zero samples.
        if (store_len_ + main_len > static_cast<int>(sizeof(store_)))
            store_len_ = 0;  // resync fallback
        std::memcpy(store_ + store_len_, data + header_len + side_len,
                    static_cast<size_t>(main_len));
        store_len_ += main_len;
        if (store_len_ > 511) {  // keep at most 511 bytes of history
            std::memmove(store_, store_ + store_len_ - 511, 511);
            store_len_ = 511;
        }
        return 0;
    }

    uint8_t body[4096 + 2048];
    int hist = main_data_begin;
    std::memcpy(body, store_ + store_len_ - hist,
                static_cast<size_t>(hist));
    std::memcpy(body + hist, data + header_len + side_len,
                static_cast<size_t>(main_len));
    int body_len = hist + main_len;

    // Update reservoir history for the NEXT frame.
    if (store_len_ + main_len > static_cast<int>(sizeof(store_))) {
        std::memmove(store_, store_ + store_len_ - 511, 511);
        store_len_ = 511;
    }
    std::memcpy(store_ + store_len_, data + header_len + side_len,
                static_cast<size_t>(main_len));
    store_len_ += main_len;
    if (store_len_ > 511) {
        std::memmove(store_, store_ + store_len_ - 511, 511);
        store_len_ = 511;
    }

    BitReader br(body, body_len);

    const int* sfb_l = tables::get_sfb_long_by_unified(sr_unified);
    const int* sfb_s = tables::get_sfb_short_by_unified(sr_unified);

    for (int gr = 0; gr < ngr; gr++) {
        for (int ch = 0; ch < nch; ch++) {
            const GranuleInfo& g = gi[gr][ch];
            int part2_start = br.pos;

            // ---------------- scalefactors ----------------
            if (mpeg1) {
                int s1 = kSlen1[g.scalefac_compress];
                int s2 = kSlen2[g.scalefac_compress];
                if (g.window_switching && g.block_type == 2) {
                    if (g.mixed_block) {
                        for (int b = 0; b < 8; b++)
                            scalefac_l_[ch][b] =
                                static_cast<int>(br.get(s1));
                        for (int b = 3; b < 6; b++)
                            for (int w = 0; w < 3; w++)
                                scalefac_s_[ch][b][w] =
                                    static_cast<int>(br.get(s1));
                    } else {
                        for (int b = 0; b < 6; b++)
                            for (int w = 0; w < 3; w++)
                                scalefac_s_[ch][b][w] =
                                    static_cast<int>(br.get(s1));
                    }
                    for (int b = 6; b < 12; b++)
                        for (int w = 0; w < 3; w++)
                            scalefac_s_[ch][b][w] =
                                static_cast<int>(br.get(s2));
                    scalefac_s_[ch][12][0] = scalefac_s_[ch][12][1] =
                        scalefac_s_[ch][12][2] = 0;
                } else {
                    static const int kScfBands[5] = { 0, 6, 11, 16, 21 };
                    for (int grp = 0; grp < 4; grp++) {
                        int slen = grp < 2 ? s1 : s2;
                        if (gr == 1 && scfsi[ch][grp]) continue;  // keep
                        for (int b = kScfBands[grp]; b < kScfBands[grp + 1];
                             b++)
                            scalefac_l_[ch][b] =
                                static_cast<int>(br.get(slen));
                    }
                    scalefac_l_[ch][21] = scalefac_l_[ch][22] = 0;
                }
            } else {
                // LSF scalefactors (13818-3): 4 slens over band groups.
                int sfc = g.scalefac_compress;
                int slen[4] = { 0, 0, 0, 0 };
                int bn;
                int preflag = 0;
                const bool is_ch =
                    (mode_ext & 1) && ch == 1;  // intensity position ch
                int int_scale = 0;
                if (!is_ch) {
                    if (sfc < 400) {
                        slen[0] = (sfc >> 4) / 5;
                        slen[1] = (sfc >> 4) % 5;
                        slen[2] = (sfc & 15) >> 2;
                        slen[3] = sfc & 3;
                        bn = 0;
                    } else if (sfc < 500) {
                        int t = sfc - 400;
                        slen[0] = (t >> 2) / 5;
                        slen[1] = (t >> 2) % 5;
                        slen[2] = t & 3;
                        bn = 1;
                    } else {
                        int t = sfc - 500;
                        slen[0] = t / 3;
                        slen[1] = t % 3;
                        bn = 2;
                        preflag = 1;
                    }
                } else {
                    int_scale = sfc & 1;
                    int isc = sfc >> 1;
                    if (isc < 180) {
                        slen[0] = isc / 36;
                        slen[1] = (isc % 36) / 6;
                        slen[2] = isc % 6;
                        bn = 3;
                    } else if (isc < 244) {
                        int t = isc - 180;
                        slen[0] = (t & 63) >> 4;
                        slen[1] = (t & 15) >> 2;
                        slen[2] = t & 3;
                        bn = 4;
                    } else {
                        int t = isc - 244;
                        slen[0] = t / 3;
                        slen[1] = t % 3;
                        bn = 5;
                    }
                }
                (void)int_scale;
                const_cast<GranuleInfo&>(g).preflag = preflag;
                int btype = !g.window_switching || g.block_type != 2
                                ? 0
                                : (g.mixed_block ? 2 : 1);
                int vals[54];
                int nv = 0;
                for (int grp = 0; grp < 4; grp++)
                    for (int k = 0; k < kLsfNsfb[bn][btype][grp]; k++)
                        vals[nv++] =
                            static_cast<int>(br.get(slen[grp]));
                if (btype == 0) {
                    for (int b = 0; b < 21 && b < nv; b++)
                        scalefac_l_[ch][b] = vals[b];
                    scalefac_l_[ch][21] = scalefac_l_[ch][22] = 0;
                } else if (btype == 1) {
                    int k = 0;
                    for (int b = 0; b < 12; b++)
                        for (int w = 0; w < 3; w++)
                            scalefac_s_[ch][b][w] =
                                k < nv ? vals[k++] : 0;
                    scalefac_s_[ch][12][0] = scalefac_s_[ch][12][1] =
                        scalefac_s_[ch][12][2] = 0;
                } else {
                    int k = 0;
                    for (int b = 0; b < 6 && k < nv; b++)
                        scalefac_l_[ch][b] = vals[k++];
                    for (int b = 3; b < 12; b++)
                        for (int w = 0; w < 3; w++)
                            scalefac_s_[ch][b][w] =
                                k < nv ? vals[k++] : 0;
                }
            }

            // ---------------- spectrum ----------------
            std::memset(ix_[ch], 0, sizeof(ix_[ch]));
            int part2_3_end = part2_start + g.part2_3_length;

            // Region boundaries in LINES.
            int reg_end[3];
            if (g.window_switching) {
                reg_end[0] = g.region0_count;  // 36 or 54 (lines)
                reg_end[1] = 576;
                reg_end[2] = 576;
            } else {
                int r0 = g.region0_count + 1;
                int r1 = r0 + g.region1_count + 1;
                if (r0 > 22) r0 = 22;
                if (r1 > 22) r1 = 22;
                reg_end[0] = sfb_l[r0];
                reg_end[1] = sfb_l[r1];
                reg_end[2] = 576;
            }

            int nlines = g.big_values * 2;
            if (nlines > 576) return -2;
            int i = 0;
            for (int r = 0; r < 3 && i < nlines; r++) {
                int tsel = g.table_select[r];
                tables::HuffTable ht = tables::get_huff_table(tsel);
                for (; i < nlines && i < reg_end[r]; i += 2) {
                    if (tsel == 0 || tsel == 4 || tsel == 14) {
                        ix_[ch][i] = ix_[ch][i + 1] = 0;
                        continue;
                    }
                    int v = g_huff.pair[tsel].decode(br);
                    if (v < 0) return -3;
                    int x = v >> 5, y = v & 31;
                    if (x == 15 && ht.linbits)
                        x += static_cast<int>(br.get(ht.linbits));
                    if (x) x = br.get1() ? -x : x;
                    if (y == 15 && ht.linbits)
                        y += static_cast<int>(br.get(ht.linbits));
                    if (y) y = br.get1() ? -y : y;
                    ix_[ch][i] = x;
                    ix_[ch][i + 1] = y;
                }
            }

            // count1 region: quads until the bit budget is exhausted.
            const HuffTree& qt =
                g.count1table_select ? g_huff.quad_b : g_huff.quad_a;
            while (br.pos < part2_3_end && i <= 572) {
                int v = qt.decode(br);
                if (v < 0) break;
                int q[4] = { (v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1,
                             v & 1 };
                for (int k = 0; k < 4; k++) {
                    if (q[k]) {
                        if (br.pos >= part2_3_end) { q[k] = 0; }
                        else if (br.get1()) q[k] = -1;
                    }
                    ix_[ch][i + k] = q[k];
                }
                i += 4;
            }
            if (br.pos > part2_3_end) {
                // Overshoot: the last quad straddled the boundary — drop
                // it (standard decoder behavior).
                for (int k = i - 4; k < i && k >= 0; k++) ix_[ch][k] = 0;
                i -= 4;
            }
            nonzero_[ch] = i;
            br.pos = part2_3_end;
        }

        // -------- reconstruction for this granule --------
        for (int ch = 0; ch < nch; ch++)
            requantize(gi[gr][ch], ch, sr_unified);
        if (nch == 2 && mode == 1)
            stereo_process(gi, gr, mode_ext, !mpeg1, sr_unified,
                           gi[gr][1].scalefac_compress & 1);
        for (int ch = 0; ch < nch; ch++) {
            antialias(gi[gr][ch], ch);
            imdct_granule(gi[gr][ch], ch);
            synth_granule(ch, gr, pcm, nch);
        }
    }

    return info.samples;
}

}  // namespace mp3
}  // namespace glint
