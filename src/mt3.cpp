// mt3.cpp — Magenta MT3 multi-instrument transcription via ggml.
//
// See mt3.h for the architecture and docs/music-transcription/mt3-port-notes.md
// for every constant with its upstream line citation. The graph is deliberately
// shaped like src/t5_translate.cpp's (RMSNorm, gated-GELU FFN, cross-attention,
// KV cache, greedy decode) with the two MT3-specific divergences called out at
// their sites:
//
//   1. fixed sinusoidal ABSOLUTE positions, never relative-attention buckets;
//   2. no 1/sqrt(head_dim) attention rescale — `mt3.attn_logit_scale`.
//
// Everything downstream of the token stream (event codec, tie sections,
// cross-segment note assembly) is a line-for-line port of
// mt3/run_length_encoding.py + mt3/note_sequences.py + mt3/metrics_utils.py,
// invalid-event skip-and-continue included: that recovery path is a documented,
// exercised behaviour, not an error branch.

#include "mt3.h"

#include "core/crispasr_env.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Env gating (repo rule: every debug/feature path behind a CRISPASR_ var)
// ===========================================================================

static bool mt3_env_on(const char* name) {
    const char* v = crispasr_env::get(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

// ===========================================================================
// Event codec — data driven from mt3.codec.event_{types,min_values,max_values}
// ===========================================================================

enum mt3_event_type {
    MT3_EV_SHIFT = 0,
    MT3_EV_PITCH,
    MT3_EV_VELOCITY,
    MT3_EV_TIE,
    MT3_EV_PROGRAM,
    MT3_EV_DRUM,
    MT3_EV_UNKNOWN,
};

static mt3_event_type mt3_event_type_from_name(const std::string& s) {
    if (s == "shift")
        return MT3_EV_SHIFT;
    if (s == "pitch")
        return MT3_EV_PITCH;
    if (s == "velocity")
        return MT3_EV_VELOCITY;
    if (s == "tie")
        return MT3_EV_TIE;
    if (s == "program")
        return MT3_EV_PROGRAM;
    if (s == "drum")
        return MT3_EV_DRUM;
    return MT3_EV_UNKNOWN;
}

struct mt3_codec_range {
    std::string name;
    mt3_event_type type = MT3_EV_UNKNOWN;
    int min_value = 0;
    int max_value = 0;
    int offset = 0; // first codec id of this range
};

// Port of mt3/event_codec.py. `shift` is forced first and starts at codec id 0
// (event_codec.py:57-59), which the converter already guarantees by writing the
// ranges in build_codec() order.
struct mt3_codec {
    std::vector<mt3_codec_range> ranges;
    int num_classes = 0;
    int steps_per_second = 100;
    int num_velocity_bins = 1;

    void build() {
        int off = 0;
        for (auto& r : ranges) {
            r.offset = off;
            r.type = mt3_event_type_from_name(r.name);
            off += r.max_value - r.min_value + 1;
        }
        num_classes = off;
    }

    // mt3/event_codec.py:103-112. false when the id lands outside every range.
    bool decode(int index, mt3_event_type& type, int& value) const {
        for (const auto& r : ranges) {
            if (index >= r.offset && index <= r.offset + (r.max_value - r.min_value)) {
                type = r.type;
                value = r.min_value + index - r.offset;
                return true;
            }
        }
        return false;
    }
};

// ===========================================================================
// tf.signal-compatible log-mel front end
// ===========================================================================
//
// Four divergences from src/core/mel.h / librosa, each silent-but-wrong if
// missed (port-notes §1.3):
//   hi_hz = 7600 (not Nyquist), the DC spectrogram bin is zeroed, triangles are
//   unnormalised (peak exactly 1.0, no Slaney area norm), and safe_log clamps
//   ONLY x <= 0 so magnitudes in (0, 1e-5) pass through and produce values
//   below log(1e-5).

static double mt3_hertz_to_mel(double f) {
    // TF's _hertz_to_mel: 1127.0 * ln(1 + f/700) == 2595 * log10(1 + f/700).
    return 1127.0 * std::log(1.0 + f / 700.0);
}

// Sparse triangular filterbank. Dense it would be 1025x512 with ~2k nonzeros;
// the row range of each triangle is contiguous because mel(f) is monotonic.
struct mt3_melfb {
    int n_spec_bins = 0;    // 1025
    int n_mel_bins = 0;     // 512
    std::vector<int> start; // per mel bin: first spectrogram bin
    std::vector<int> count; // per mel bin: number of bins
    std::vector<int> woff;  // per mel bin: offset into `w`
    std::vector<float> w;   // concatenated weights

    void build(int num_mel_bins, int num_spec_bins, double sample_rate, double lo_hz, double hi_hz) {
        n_spec_bins = num_spec_bins;
        n_mel_bins = num_mel_bins;
        const double nyquist = sample_rate / 2.0;
        // np.linspace(0, nyquist, num_spec_bins)[1:] — bands_to_zero = 1, i.e.
        // spectrogram bin 0 (DC) gets an all-zero filterbank row.
        std::vector<double> bins_mel(num_spec_bins);
        bins_mel[0] = 0.0;
        for (int b = 1; b < num_spec_bins; b++) {
            const double hz = (double)b * nyquist / (double)(num_spec_bins - 1);
            bins_mel[b] = mt3_hertz_to_mel(hz);
        }
        // np.linspace(mel(lo), mel(hi), num_mel_bins + 2), framed (3, 1).
        const double mlo = mt3_hertz_to_mel(lo_hz);
        const double mhi = mt3_hertz_to_mel(hi_hz);
        const int n_edge = num_mel_bins + 2;
        std::vector<double> edges(n_edge);
        for (int i = 0; i < n_edge; i++)
            edges[i] = mlo + (mhi - mlo) * (double)i / (double)(n_edge - 1);
        edges[n_edge - 1] = mhi;

        start.assign(num_mel_bins, 0);
        count.assign(num_mel_bins, 0);
        woff.assign(num_mel_bins, 0);
        w.clear();
        std::vector<float> col(num_spec_bins);
        for (int j = 0; j < num_mel_bins; j++) {
            const double lower = edges[j], center = edges[j + 1], upper = edges[j + 2];
            int first = -1, last = -1;
            for (int b = 1; b < num_spec_bins; b++) { // row 0 stays zero
                const double ls = (bins_mel[b] - lower) / (center - lower);
                const double us = (upper - bins_mel[b]) / (upper - center);
                double v = std::min(ls, us);
                if (v < 0.0)
                    v = 0.0;
                col[b] = (float)v;
                if (col[b] != 0.0f) {
                    if (first < 0)
                        first = b;
                    last = b;
                }
            }
            if (first < 0) {
                start[j] = 0;
                count[j] = 0;
                woff[j] = (int)w.size();
                continue;
            }
            start[j] = first;
            count[j] = last - first + 1;
            woff[j] = (int)w.size();
            for (int b = first; b <= last; b++)
                w.push_back(col[b]);
        }
    }
};

// Double-precision iterative radix-2 FFT with exact per-stage twiddles (no
// recurrence drift): the reference computes magnitudes in float64 and a
// single-precision recurrence would move the smallest magnitudes enough for the
// natural log to see it.
struct mt3_fft {
    int n = 0;
    std::vector<int> rev;
    std::vector<double> wr, wi;

    void init(int size) {
        n = size;
        rev.assign(n, 0);
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            rev[i] = j;
        }
        wr.resize(n / 2);
        wi.resize(n / 2);
        for (int k = 0; k < n / 2; k++) {
            const double a = -2.0 * M_PI * (double)k / (double)n;
            wr[k] = std::cos(a);
            wi[k] = std::sin(a);
        }
    }

    void run(double* re, double* im) const {
        for (int i = 1; i < n; i++)
            if (i < rev[i]) {
                std::swap(re[i], re[rev[i]]);
                std::swap(im[i], im[rev[i]]);
            }
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; j++) {
                    const double cr = wr[j * step], ci = wi[j * step];
                    const double xr = re[i + j + half], xi = im[i + j + half];
                    const double tr = xr * cr - xi * ci;
                    const double ti = xr * ci + xi * cr;
                    re[i + j + half] = re[i + j] - tr;
                    im[i + j + half] = im[i + j] - ti;
                    re[i + j] += tr;
                    im[i + j] += ti;
                }
            }
        }
    }
};

struct mt3_frontend {
    mt3_fft fft;
    mt3_melfb fb;
    std::vector<float> window; // Hann PERIODIC (tf.signal.stft default)
    int n_fft = 2048;
    int hop = 128;
    int n_mel = 512;
    float log_eps = 1e-5f;

    void init(int n_fft_, int hop_, int n_mel_, double sr, double lo_hz, double hi_hz, float log_eps_) {
        n_fft = n_fft_;
        hop = hop_;
        n_mel = n_mel_;
        log_eps = log_eps_;
        fft.init(n_fft);
        fb.build(n_mel, n_fft / 2 + 1, sr, lo_hz, hi_hz);
        window.resize(n_fft);
        for (int k = 0; k < n_fft; k++)
            window[k] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * (double)k / (double)n_fft));
    }

    // mt3/spectral_ops.py:29-32 — clamp ONLY non-positive values.
    float safe_log(float x) const { return std::log(x <= 0.0f ? log_eps : x); }

    // tf.signal.stft(pad_end=True) → magnitude → mel → safe_log.
    // `n_frames` = ceil(n/hop); frame f reads [f*hop, f*hop+n_fft) of the
    // segment's OWN samples and zero beyond its end. No centering, no reflect.
    // Output is row-major (n_frames, n_mel) — MelsTime, which is exactly the
    // layout ggml_mul_mat wants for the continuous-inputs projection.
    void logmel(const float* samples, int n, std::vector<float>& out, int& n_frames) const {
        n_frames = (n + hop - 1) / hop;
        out.assign((size_t)n_frames * (size_t)n_mel, 0.0f);
        std::vector<double> re(n_fft), im(n_fft);
        std::vector<float> mag(n_fft / 2 + 1);
        for (int f = 0; f < n_frames; f++) {
            const int base = f * hop;
            for (int k = 0; k < n_fft; k++) {
                const int idx = base + k;
                const float s = (idx < n) ? samples[idx] : 0.0f;
                re[k] = (double)(s * window[k]); // float32 product, then widened
                im[k] = 0.0;
            }
            fft.run(re.data(), im.data());
            for (int b = 0; b <= n_fft / 2; b++)
                mag[b] = (float)std::sqrt(re[b] * re[b] + im[b] * im[b]);
            float* dst = out.data() + (size_t)f * (size_t)n_mel;
            for (int j = 0; j < n_mel; j++) {
                const int c = fb.count[j];
                const float* wp = fb.w.data() + fb.woff[j];
                const float* mp = mag.data() + fb.start[j];
                double acc = 0.0;
                for (int t = 0; t < c; t++)
                    acc += (double)mp[t] * (double)wp[t];
                dst[j] = safe_log((float)acc);
            }
        }
    }
};

// ===========================================================================
// Segmentation (port-notes §1.4)
// ===========================================================================

struct mt3_segment {
    double start_time = 0.0; // seconds, floored to 1/steps_per_second
    int sample_begin = 0;
    int sample_end = 0; // exclusive, into the hop-padded stream
};

// mt3 colab `_audio_to_frames` + t5 `split_tokens_to_inputs_length`: pad the
// audio up to a whole number of hop-sized frames, chunk the frame stream into
// inputs_length-frame segments (the last one short), and floor each start time
// to 10 ms — that floor is not cosmetic, it is the offset every event time of
// the segment is added to AND the max_decode_time clamp of the previous one.
static std::vector<mt3_segment> mt3_split_segments(int n_samples, int hop, int inputs_length, int sample_rate,
                                                   int steps_per_second) {
    std::vector<mt3_segment> segs;
    if (n_samples <= 0)
        return segs;
    const int padded = ((n_samples + hop - 1) / hop) * hop;
    const int num_frames = padded / hop;
    const double fps = (double)sample_rate / (double)hop;
    for (int f0 = 0; f0 < num_frames; f0 += inputs_length) {
        const int f1 = std::min(f0 + inputs_length, num_frames);
        double st = (double)f0 / fps;
        st -= std::fmod(st, 1.0 / (double)steps_per_second);
        mt3_segment s;
        s.start_time = st;
        s.sample_begin = f0 * hop;
        s.sample_end = f1 * hop;
        segs.push_back(s);
    }
    return segs;
}

// ===========================================================================
// Model
// ===========================================================================

struct mt3_hparams {
    int vocab_size = 1536;
    int d_model = 512;
    int d_kv = 64;
    int d_ff = 1024;
    int n_heads = 6;
    int enc_layers = 8;
    int dec_layers = 8;
    float ln_eps = 1e-6f;
    float attn_logit_scale = 1.0f;
    int pad_id = 0;
    int eos_id = 1;
    int dec_start_id = 0;

    int sample_rate = 16000;
    int hop_width = 128;
    int n_fft = 2048;
    int num_mel_bins = 512;
    float mel_lo_hz = 20.0f;
    float mel_hi_hz = 7600.0f;
    float log_eps = 1e-5f;

    int inputs_length = 256;
    int targets_length = 1024;
    int num_special_tokens = 3;
    int extra_ids = 100;
    int pos_max_length = 2048;

    std::string pos_embed = "sinusoidal";
    int use_rel_bias = 0;
};

struct mt3_enc_layer {
    ggml_tensor* attn_rms = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* ffn_rms = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct mt3_dec_layer {
    ggml_tensor* attn_rms = nullptr;
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* cross_rms = nullptr;
    ggml_tensor* cross_q = nullptr;
    ggml_tensor* cross_k = nullptr;
    ggml_tensor* cross_v = nullptr;
    ggml_tensor* cross_o = nullptr;
    ggml_tensor* ffn_rms = nullptr;
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
};

struct mt3_model {
    mt3_hparams hp;
    ggml_tensor* pos_embd = nullptr;   // (d_model, pos_max_length) F32, FixedEmbed
    ggml_tensor* token_embd = nullptr; // (d_model, vocab)
    ggml_tensor* lm_head = nullptr;
    ggml_tensor* enc_inp_proj = nullptr; // continuous_inputs_projection
    ggml_tensor* enc_final_rms = nullptr;
    ggml_tensor* dec_final_rms = nullptr;
    std::vector<mt3_enc_layer> enc;
    std::vector<mt3_dec_layer> dec;
};

struct mt3_context {
    mt3_params params;
    mt3_model model;
    mt3_codec codec;
    mt3_frontend fe;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Decoder self-attention KV cache (reset per segment).
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    int kv_max_ctx = 0;

    // Cross-attention KV cache (recomputed per segment; T_enc is constant).
    std::vector<ggml_tensor*> cross_kv_k;
    std::vector<ggml_tensor*> cross_kv_v;
    ggml_context* cross_kv_ctx = nullptr;
    ggml_backend_buffer_t cross_kv_buf = nullptr;
    int cross_T_enc = 0;

    // Extra graph outputs for the diff harness. Off by default: marking a
    // tensor as an output perturbs buffer elision, so the shipping graph stays
    // byte-for-byte what it was.
    bool capture = false;
    std::vector<float> cap_enc_input;
};

static ggml_tensor* mt3_T(mt3_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return (it != c->tensors.end()) ? it->second : nullptr;
}

static ggml_tensor* mt3_TR(mt3_context* c, const char* name) {
    auto* t = mt3_T(c, name);
    if (!t)
        fprintf(stderr, "mt3: required tensor '%s' not found\n", name);
    return t;
}

// ── Metadata ─────────────────────────────────────────────────────

static bool mt3_load_metadata(mt3_context* c, gguf_context* g) {
    auto& hp = c->model.hp;
    auto u32 = [&](const char* k, int d) { return (int)core_gguf::kv_u32(g, k, (uint32_t)d); };
    auto f32 = [&](const char* k, float d) { return core_gguf::kv_f32(g, k, d); };

    hp.vocab_size = u32("mt3.vocab_size", 1536);
    hp.d_model = u32("mt3.d_model", 512);
    hp.d_kv = u32("mt3.d_kv", 64);
    hp.d_ff = u32("mt3.d_ff", 1024);
    hp.n_heads = u32("mt3.n_heads", 6);
    hp.enc_layers = u32("mt3.encoder.n_layers", 8);
    hp.dec_layers = u32("mt3.decoder.n_layers", 8);
    hp.ln_eps = f32("mt3.layer_norm_epsilon", 1e-6f);
    hp.pad_id = u32("mt3.pad_token_id", 0);
    hp.eos_id = u32("mt3.eos_token_id", 1);
    hp.dec_start_id = u32("mt3.decoder_start_token_id", 0);

    hp.sample_rate = u32("mt3.spectrogram.sample_rate", 16000);
    hp.hop_width = u32("mt3.spectrogram.hop_width", 128);
    hp.n_fft = u32("mt3.spectrogram.n_fft", 2048);
    hp.num_mel_bins = u32("mt3.spectrogram.num_mel_bins", 512);
    hp.mel_lo_hz = f32("mt3.spectrogram.mel_lo_hz", 20.0f);
    hp.mel_hi_hz = f32("mt3.spectrogram.mel_hi_hz", 7600.0f);
    hp.log_eps = f32("mt3.spectrogram.log_eps", 1e-5f);

    hp.inputs_length = u32("mt3.inputs_length", 256);
    hp.targets_length = u32("mt3.targets_length", 1024);
    hp.num_special_tokens = u32("mt3.codec.num_special_tokens", 3);
    hp.extra_ids = u32("mt3.codec.extra_ids", 100);
    hp.pos_max_length = u32("mt3.pos_embed_max_length", 2048);

    hp.pos_embed = core_gguf::kv_str(g, "mt3.pos_embed", "");
    hp.use_rel_bias = u32("mt3.use_relative_attention_bias", 1);
    hp.attn_logit_scale = f32("mt3.attn_logit_scale", 0.0f);

    // ── Risk #3 guard rail ────────────────────────────────────────
    // A model that claims relative-attention buckets must NOT quietly take the
    // absolute-position path (or vice versa): the result loads, runs, and is
    // wrong only at positions > 0 — kunato/mt3-pytorch's shipped bug. Fail loud.
    if (hp.pos_embed != "sinusoidal") {
        fprintf(stderr,
                "mt3: mt3.pos_embed = '%s' but this runtime implements only the "
                "fixed sinusoidal (FixedEmbed) path. Refusing to load.\n",
                hp.pos_embed.c_str());
        return false;
    }
    if (hp.use_rel_bias != 0) {
        fprintf(stderr,
                "mt3: mt3.use_relative_attention_bias = %d. This runtime has no "
                "relative-attention-bias path; a silent fallback would be wrong "
                "at every position > 0. Refusing to load.\n",
                hp.use_rel_bias);
        return false;
    }
    // The scale is written by the converter precisely so it never comes from a
    // default (or stack garbage — cf. the KvSelfAttnParams lesson).
    if (!(hp.attn_logit_scale > 0.0f)) {
        fprintf(stderr, "mt3: mt3.attn_logit_scale missing or non-positive (%g); refusing to guess.\n",
                (double)hp.attn_logit_scale);
        return false;
    }

    // ── Codec ranges, data driven ─────────────────────────────────
    auto& cd = c->codec;
    cd.steps_per_second = u32("mt3.codec.steps_per_second", 100);
    cd.num_velocity_bins = u32("mt3.codec.num_velocity_bins", 1);
    std::vector<std::string> types = core_gguf::kv_str_array(g, "mt3.codec.event_types");
    std::vector<int> mins, maxs;
    {
        const int i_min = gguf_find_key(g, "mt3.codec.event_min_values");
        const int i_max = gguf_find_key(g, "mt3.codec.event_max_values");
        if (i_min >= 0 && i_max >= 0) {
            const int n_min = (int)gguf_get_arr_n(g, i_min);
            const int n_max = (int)gguf_get_arr_n(g, i_max);
            const int32_t* pmin = (const int32_t*)gguf_get_arr_data(g, i_min);
            const int32_t* pmax = (const int32_t*)gguf_get_arr_data(g, i_max);
            for (int i = 0; i < n_min; i++)
                mins.push_back((int)pmin[i]);
            for (int i = 0; i < n_max; i++)
                maxs.push_back((int)pmax[i]);
        }
    }
    if (types.empty() || types.size() != mins.size() || types.size() != maxs.size()) {
        fprintf(stderr, "mt3: mt3.codec.event_{types,min_values,max_values} missing or inconsistent\n");
        return false;
    }
    for (size_t i = 0; i < types.size(); i++) {
        mt3_codec_range r;
        r.name = types[i];
        r.min_value = mins[i];
        r.max_value = maxs[i];
        cd.ranges.push_back(r);
    }
    cd.build();
    if (cd.ranges[0].type != MT3_EV_SHIFT || cd.ranges[0].offset != 0) {
        fprintf(stderr, "mt3: codec range 0 must be 'shift' at codec id 0 (event_codec.py:57-59)\n");
        return false;
    }
    const int declared = u32("mt3.codec.num_classes", cd.num_classes);
    if (declared != cd.num_classes) {
        fprintf(stderr, "mt3: codec num_classes mismatch: metadata %d, ranges sum to %d\n", declared, cd.num_classes);
        return false;
    }
    return true;
}

// ── Bind ─────────────────────────────────────────────────────────

static bool mt3_bind(mt3_context* c) {
    auto& m = c->model;
    const auto& hp = m.hp;

    // Risk #3, second half: assert the rel-bias tensors really are absent, so a
    // GGUF that carries them can never be run through the absolute-position
    // graph by accident.
    for (const auto& kv : c->tensors) {
        if (kv.first.find("rel_bias") != std::string::npos || kv.first.find("attn_rel_b") != std::string::npos) {
            fprintf(stderr,
                    "mt3: tensor '%s' present but mt3.use_relative_attention_bias = 0. "
                    "The model and the metadata disagree; refusing to load.\n",
                    kv.first.c_str());
            return false;
        }
    }

    m.pos_embd = mt3_TR(c, "pos_embd.weight");
    m.token_embd = mt3_TR(c, "token_embd.weight");
    m.lm_head = mt3_TR(c, "lm_head.weight");
    m.enc_inp_proj = mt3_TR(c, "enc.inp_proj.weight");
    m.enc_final_rms = mt3_TR(c, "enc.final_rms.weight");
    m.dec_final_rms = mt3_TR(c, "dec.final_rms.weight");
    if (!m.pos_embd || !m.token_embd || !m.lm_head || !m.enc_inp_proj || !m.enc_final_rms || !m.dec_final_rms)
        return false;

    if (m.pos_embd->ne[0] != hp.d_model) {
        fprintf(stderr, "mt3: pos_embd.weight has ne[0]=%lld, expected d_model=%d\n", (long long)m.pos_embd->ne[0],
                hp.d_model);
        return false;
    }
    if (m.pos_embd->type != GGML_TYPE_F32) {
        fprintf(stderr, "mt3: pos_embd.weight must be F32 (the FixedEmbed table is not quantisable)\n");
        return false;
    }

    char buf[128];
    m.enc.resize(hp.enc_layers);
    for (int i = 0; i < hp.enc_layers; i++) {
        auto& l = m.enc[i];
        auto w = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "enc.blk.%d.%s", i, suffix);
            return mt3_TR(c, buf);
        };
        l.attn_rms = w("attn_rms.weight");
        l.attn_q = w("attn_q.weight");
        l.attn_k = w("attn_k.weight");
        l.attn_v = w("attn_v.weight");
        l.attn_o = w("attn_o.weight");
        l.ffn_rms = w("ffn_rms.weight");
        l.ffn_gate = w("ffn_gate.weight");
        l.ffn_up = w("ffn_up.weight");
        l.ffn_down = w("ffn_down.weight");
        if (!l.attn_rms || !l.attn_q || !l.attn_k || !l.attn_v || !l.attn_o || !l.ffn_rms || !l.ffn_gate || !l.ffn_up ||
            !l.ffn_down)
            return false;
    }

    m.dec.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        auto& l = m.dec[i];
        auto w = [&](const char* suffix) {
            snprintf(buf, sizeof(buf), "dec.blk.%d.%s", i, suffix);
            return mt3_TR(c, buf);
        };
        l.attn_rms = w("attn_rms.weight");
        l.attn_q = w("attn_q.weight");
        l.attn_k = w("attn_k.weight");
        l.attn_v = w("attn_v.weight");
        l.attn_o = w("attn_o.weight");
        l.cross_rms = w("cross_rms.weight");
        l.cross_q = w("cross_q.weight");
        l.cross_k = w("cross_k.weight");
        l.cross_v = w("cross_v.weight");
        l.cross_o = w("cross_o.weight");
        l.ffn_rms = w("ffn_rms.weight");
        l.ffn_gate = w("ffn_gate.weight");
        l.ffn_up = w("ffn_up.weight");
        l.ffn_down = w("ffn_down.weight");
        if (!l.attn_rms || !l.attn_q || !l.attn_k || !l.attn_v || !l.attn_o || !l.cross_rms || !l.cross_q ||
            !l.cross_k || !l.cross_v || !l.cross_o || !l.ffn_rms || !l.ffn_gate || !l.ffn_up || !l.ffn_down)
            return false;
    }
    return true;
}

// ===========================================================================
// Graphs
// ===========================================================================

// Graph node budget. The decoder graph is ~50 nodes per layer (8 layers) and
// the encoder fewer; 2048 leaves ample margin. It is deliberately not larger:
// the per-step cost of ggml_backend_sched_reset / alloc_graph scales with it,
// and greedy decode rebuilds the graph once per token.
static const int MT3_GRAPH_NODES = 2048;

static ggml_tensor* mt3_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// Softmax with the model's own logit scale. MT3 folds 1/sqrt(depth) into the
// initializers (mt3/layers.py:230-234) so the scale is 1.0 — but it comes from
// `mt3.attn_logit_scale`, never from an implicit default.
static ggml_tensor* mt3_softmax(ggml_context* ctx, ggml_tensor* kq, ggml_tensor* mask, float scale) {
    if (scale != 1.0f)
        kq = ggml_scale(ctx, kq, scale);
    if (mask)
        kq = ggml_add(ctx, kq, mask);
    return ggml_soft_max(ctx, kq);
}

static ggml_cgraph* mt3_build_encoder_graph(mt3_context* c, int T) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int nh = hp.n_heads;
    const int hd = hp.d_kv;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, MT3_GRAPH_NODES, false);

    // Continuous input: the log-mel frame itself, (num_mel_bins, T).
    // MelsTime IS the layout ggml_mul_mat wants here — never transpose it.
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.num_mel_bins, T);
    ggml_set_name(mel, "enc_mel");
    ggml_set_input(mel);

    // Absolute sinusoidal positions, arange(T) (network.py:171/180). No pos
    // buckets, no rel-bias get_rows, no ggml_add(kq, pos_bias) anywhere below.
    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos, "enc_pos");
    ggml_set_input(pos);

    ggml_tensor* cur = ggml_mul_mat(ctx0, m.enc_inp_proj, mel);
    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, m.pos_embd, pos));

    if (c->capture) {
        ggml_set_name(cur, "enc_input");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }

    for (int il = 0; il < hp.enc_layers; il++) {
        const auto& l = m.enc[il];
        ggml_tensor* residual = cur;

        cur = mt3_rms_norm(ctx0, cur, l.attn_rms, hp.ln_eps);

        ggml_tensor* Q = ggml_mul_mat(ctx0, l.attn_q, cur);
        ggml_tensor* K = ggml_mul_mat(ctx0, l.attn_k, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, l.attn_v, cur);

        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3);
        K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3);
        V = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3);

        ggml_tensor* kq = ggml_mul_mat(ctx0, K, Q); // (T_k, T_q, nh)
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        // The encoder padding mask is all-ones (network.py:286-289): the
        // zero-padded tail of a short final segment is DELIBERATELY attended to.
        kq = mt3_softmax(ctx0, kq, nullptr, hp.attn_logit_scale);

        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);

        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, T);
        cur = ggml_mul_mat(ctx0, l.attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        residual = cur;
        cur = mt3_rms_norm(ctx0, cur, l.ffn_rms, hp.ln_eps);
        // flax nn.gelu(approximate=True) == ggml's tanh GELU. Not gelu_erf.
        ggml_tensor* gate = ggml_gelu(ctx0, ggml_mul_mat(ctx0, l.ffn_gate, cur));
        ggml_tensor* up = ggml_mul_mat(ctx0, l.ffn_up, cur);
        cur = ggml_mul_mat(ctx0, l.ffn_down, ggml_mul(ctx0, gate, up));
        cur = ggml_add(ctx0, cur, residual);
    }

    cur = mt3_rms_norm(ctx0, cur, m.enc_final_rms, hp.ln_eps);
    ggml_set_name(cur, "enc_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

static bool mt3_alloc_kv_cache(mt3_context* c, int max_ctx) {
    const auto& hp = c->model.hp;
    if (c->kv_buf) {
        ggml_backend_buffer_free(c->kv_buf);
        c->kv_buf = nullptr;
    }
    if (c->kv_ctx) {
        ggml_free(c->kv_ctx);
        c->kv_ctx = nullptr;
    }
    ggml_init_params params = {ggml_tensor_overhead() * 2 + 64, nullptr, true};
    c->kv_ctx = ggml_init(params);
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hp.d_kv, max_ctx, hp.n_heads, hp.dec_layers);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hp.d_kv, max_ctx, hp.n_heads, hp.dec_layers);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");
    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf)
        return false;
    ggml_backend_buffer_clear(c->kv_buf, 0);
    c->kv_max_ctx = max_ctx;
    return true;
}

static bool mt3_alloc_cross_kv(mt3_context* c, int T_enc) {
    const auto& hp = c->model.hp;
    if (c->cross_T_enc == T_enc && c->cross_kv_buf)
        return true;
    if (c->cross_kv_buf) {
        ggml_backend_buffer_free(c->cross_kv_buf);
        c->cross_kv_buf = nullptr;
    }
    if (c->cross_kv_ctx) {
        ggml_free(c->cross_kv_ctx);
        c->cross_kv_ctx = nullptr;
    }
    ggml_init_params params = {ggml_tensor_overhead() * hp.dec_layers * 2 + 64, nullptr, true};
    c->cross_kv_ctx = ggml_init(params);
    c->cross_kv_k.resize(hp.dec_layers);
    c->cross_kv_v.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        c->cross_kv_k[i] = ggml_new_tensor_3d(c->cross_kv_ctx, GGML_TYPE_F16, hp.d_kv, T_enc, hp.n_heads);
        c->cross_kv_v[i] = ggml_new_tensor_3d(c->cross_kv_ctx, GGML_TYPE_F16, hp.d_kv, T_enc, hp.n_heads);
    }
    c->cross_kv_buf = ggml_backend_alloc_ctx_tensors(c->cross_kv_ctx, c->backend);
    if (!c->cross_kv_buf)
        return false;
    c->cross_T_enc = T_enc;
    return true;
}

static bool mt3_compute_cross_kv(mt3_context* c, const float* enc_out, int T_enc) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int nh = hp.n_heads, hd = hp.d_kv, D = hp.d_model;

    if (!mt3_alloc_cross_kv(c, T_enc))
        return false;

    for (int il = 0; il < hp.dec_layers; il++) {
        const auto& l = m.dec[il];
        ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

        ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        ggml_set_name(enc_inp, "enc_for_cross");
        ggml_set_input(enc_inp);

        ggml_tensor* K = ggml_mul_mat(ctx0, l.cross_k, enc_inp);
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(K, "cross_k");
        ggml_tensor* V = ggml_mul_mat(ctx0, l.cross_v, enc_inp);
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(V, "cross_v");
        ggml_build_forward_expand(gf, K);
        ggml_build_forward_expand(gf, V);

        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
            ggml_free(ctx0);
            return false;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_for_cross"), enc_out, 0,
                                (size_t)D * T_enc * sizeof(float));
        if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx0);
            return false;
        }

        const size_t n_elem = (size_t)hd * T_enc * nh;
        std::vector<float> buf(n_elem);
        std::vector<ggml_fp16_t> buf16(n_elem);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "cross_k"), buf.data(), 0, n_elem * sizeof(float));
        ggml_fp32_to_fp16_row(buf.data(), buf16.data(), (int)n_elem);
        ggml_backend_tensor_set(c->cross_kv_k[il], buf16.data(), 0, n_elem * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "cross_v"), buf.data(), 0, n_elem * sizeof(float));
        ggml_fp32_to_fp16_row(buf.data(), buf16.data(), (int)n_elem);
        ggml_backend_tensor_set(c->cross_kv_v[il], buf16.data(), 0, n_elem * sizeof(ggml_fp16_t));
        ggml_free(ctx0);
    }
    return true;
}

static ggml_cgraph* mt3_build_decoder_graph(mt3_context* c, int n_tokens, int offset) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model, nh = hp.n_heads, hd = hp.d_kv;
    const int Lk = offset + n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, MT3_GRAPH_NODES, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp, "dec_tokens");
    ggml_set_input(inp);

    // Decoder positions are arange(T) too (network.py:214/225 — the
    // decoder_positions argument is overwritten and ignored); in cached
    // single-step decode that is just the step index (layers.py:589-596).
    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos, "dec_pos");
    ggml_set_input(pos);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = ggml_get_rows(ctx0, m.token_embd, inp);
    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, m.pos_embd, pos));

    for (int il = 0; il < hp.dec_layers; il++) {
        const auto& l = m.dec[il];
        ggml_tensor* residual = cur;

        cur = mt3_rms_norm(ctx0, cur, l.attn_rms, hp.ln_eps);

        ggml_tensor* Q = ggml_mul_mat(ctx0, l.attn_q, cur);
        ggml_tensor* K = ggml_mul_mat(ctx0, l.attn_k, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, l.attn_v, cur);

        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, n_tokens), 0, 2, 1, 3);
        ggml_tensor* K_new = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, n_tokens), 0, 2, 1, 3);
        ggml_tensor* V_new = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, n_tokens), 0, 2, 1, 3);

        ggml_tensor* k_view =
            ggml_view_4d(ctx0, c->kv_k, hd, n_tokens, nh, 1, c->kv_k->nb[1], c->kv_k->nb[2], c->kv_k->nb[3],
                         (size_t)il * c->kv_k->nb[3] + (size_t)offset * c->kv_k->nb[1]);
        ggml_tensor* v_view =
            ggml_view_4d(ctx0, c->kv_v, hd, n_tokens, nh, 1, c->kv_v->nb[1], c->kv_v->nb[2], c->kv_v->nb[3],
                         (size_t)il * c->kv_v->nb[3] + (size_t)offset * c->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new, v_view));

        ggml_tensor* Kfull = ggml_cast(
            ctx0, ggml_view_3d(ctx0, c->kv_k, hd, Lk, nh, c->kv_k->nb[1], c->kv_k->nb[2], (size_t)il * c->kv_k->nb[3]),
            GGML_TYPE_F32);
        ggml_tensor* Vfull = ggml_cast(
            ctx0, ggml_view_3d(ctx0, c->kv_v, hd, Lk, nh, c->kv_v->nb[1], c->kv_v->nb[2], (size_t)il * c->kv_v->nb[3]),
            GGML_TYPE_F32);

        ggml_tensor* kq = ggml_mul_mat(ctx0, Kfull, Q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = mt3_softmax(ctx0, kq, causal_mask, hp.attn_logit_scale);

        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, Vfull));
        ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);

        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, n_tokens);
        cur = ggml_mul_mat(ctx0, l.attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // ---- Cross-attention (no positions of any kind, no mask) ----
        residual = cur;
        cur = mt3_rms_norm(ctx0, cur, l.cross_rms, hp.ln_eps);

        ggml_tensor* CQ = ggml_mul_mat(ctx0, l.cross_q, cur);
        CQ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, CQ, hd, nh, n_tokens), 0, 2, 1, 3);

        ggml_tensor* ca_kq = ggml_mul_mat(ctx0, c->cross_kv_k[il], CQ);
        ggml_mul_mat_set_prec(ca_kq, GGML_PREC_F32);
        ca_kq = mt3_softmax(ctx0, ca_kq, nullptr, hp.attn_logit_scale);

        ggml_tensor* cv_t = ggml_cont(ctx0, ggml_transpose(ctx0, c->cross_kv_v[il]));
        ggml_tensor* ca_kqv = ggml_mul_mat(ctx0, cv_t, ca_kq);

        cur = ggml_cont(ctx0, ggml_permute(ctx0, ca_kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, n_tokens);
        cur = ggml_mul_mat(ctx0, l.cross_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // ---- FFN ----
        residual = cur;
        cur = mt3_rms_norm(ctx0, cur, l.ffn_rms, hp.ln_eps);
        ggml_tensor* gate = ggml_gelu(ctx0, ggml_mul_mat(ctx0, l.ffn_gate, cur));
        ggml_tensor* up = ggml_mul_mat(ctx0, l.ffn_up, cur);
        cur = ggml_mul_mat(ctx0, l.ffn_down, ggml_mul(ctx0, gate, up));
        cur = ggml_add(ctx0, cur, residual);
    }

    cur = mt3_rms_norm(ctx0, cur, m.dec_final_rms, hp.ln_eps);
    if (n_tokens > 1)
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    cur = ggml_mul_mat(ctx0, m.lm_head, cur); // untied logits_dense
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ── Runners ──────────────────────────────────────────────────────

static bool mt3_run_encoder(mt3_context* c, const float* mel, int T, std::vector<float>& enc_out) {
    const auto& hp = c->model.hp;
    ggml_cgraph* gf = mt3_build_encoder_graph(c, T);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return false;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_mel"), mel, 0,
                            (size_t)hp.num_mel_bins * (size_t)T * sizeof(float));
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++)
        pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return false;

    enc_out.resize((size_t)hp.d_model * (size_t)T);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "enc_out"), enc_out.data(), 0, enc_out.size() * sizeof(float));
    if (c->capture) {
        ggml_tensor* ei = ggml_graph_get_tensor(gf, "enc_input");
        if (ei) {
            c->cap_enc_input.resize((size_t)hp.d_model * (size_t)T);
            ggml_backend_tensor_get(ei, c->cap_enc_input.data(), 0, c->cap_enc_input.size() * sizeof(float));
        }
    }
    return true;
}

static bool mt3_run_decoder_step(mt3_context* c, const int32_t* tokens, int n_tokens, int offset,
                                 std::vector<float>& logits) {
    const auto& hp = c->model.hp;
    const int Lk = offset + n_tokens;
    ggml_cgraph* gf = mt3_build_decoder_graph(c, n_tokens, offset);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return false;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_tokens"), tokens, 0, (size_t)n_tokens * sizeof(int32_t));
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = offset + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));

    if (n_tokens > 1) {
        std::vector<float> mask((size_t)n_tokens * (size_t)Lk, 0.0f);
        for (int q = 0; q < n_tokens; q++)
            for (int k = offset + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = -INFINITY;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0, mask.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return false;

    logits.resize(hp.vocab_size);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), logits.data(), 0, logits.size() * sizeof(float));
    return true;
}

// Greedy decode of one segment. Mirrors the reference exactly: the decoder
// INPUT starts at [decoder_start_id]; the argmax is appended to the output and
// fed back unless it is EOS.
static bool mt3_greedy(mt3_context* c, int max_steps, std::vector<int>& out_tokens,
                       std::vector<float>* all_logits /*optional, (steps, vocab)*/) {
    const auto& hp = c->model.hp;
    out_tokens.clear();
    if (all_logits)
        all_logits->clear();

    int32_t tok = (int32_t)hp.dec_start_id;
    std::vector<float> logits;
    for (int step = 0; step < max_steps; step++) {
        if (!mt3_run_decoder_step(c, &tok, 1, step, logits))
            return false;
        if (all_logits)
            all_logits->insert(all_logits->end(), logits.begin(), logits.end());
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < (int)logits.size(); i++)
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        out_tokens.push_back(best);
        if (best == hp.eos_id)
            break;
        tok = (int32_t)best;
    }
    return true;
}

// ===========================================================================
// Note assembly: mt3/note_sequences.py + run_length_encoding.py + metrics_utils
// ===========================================================================

static const double MT3_MIN_NOTE_DURATION = 0.01;     // note_sequences.py:32
static const double MT3_DEFAULT_NOTE_DURATION = 0.01; // note_sequences.py:29
static const int MT3_DEFAULT_VELOCITY = 100;          // note_sequences.py:28

struct mt3_active {
    int pitch = 0;
    int program = 0;
    double onset = 0.0;
    int velocity = 0;
};

struct mt3_note {
    double start_s = 0.0;
    double end_s = 0.0;
    int pitch = 0;
    int velocity = 0;
    int program = 0;
    bool is_drum = false;
};

// note_sequences.py:262-281. NOTE what does NOT reset between segments:
// current_time, current_velocity, current_program, active_pitches and the
// accumulated note list. Only tied_pitches / is_tie_section do.
//
// `active_pitches` is a std::vector rather than a map on purpose: upstream is a
// Python dict, whose iteration order is INSERTION order, and both the tie-close
// loop and flush() iterate it. A std::map would silently reorder the emitted
// notes relative to the reference.
struct mt3_decode_state {
    double current_time = 0.0;
    int current_velocity = MT3_DEFAULT_VELOCITY;
    int current_program = 0;
    std::vector<mt3_active> active_pitches;
    std::vector<std::pair<int, int>> tied_pitches; // (pitch, program)
    bool is_tie_section = false;
    std::vector<mt3_note> notes;
    int n_tie_ends = 0;     // `tie` events accepted
    int n_tied_pitches = 0; // pitch declarations accepted inside a tie section

    int find_active(int pitch, int program) const {
        for (size_t i = 0; i < active_pitches.size(); i++)
            if (active_pitches[i].pitch == pitch && active_pitches[i].program == program)
                return (int)i;
        return -1;
    }
    bool is_tied(int pitch, int program) const {
        for (const auto& p : tied_pitches)
            if (p.first == pitch && p.second == program)
                return true;
        return false;
    }
    // note_sequences.py:301-310 — MIN_NOTE_DURATION floor on the end time.
    void add_note(double start, double end, int pitch, int velocity, int program, bool is_drum) {
        if (end < start + MT3_MIN_NOTE_DURATION)
            end = start + MT3_MIN_NOTE_DURATION;
        mt3_note n;
        n.start_s = start;
        n.end_s = end;
        n.pitch = pitch;
        n.velocity = velocity;
        n.program = program;
        n.is_drum = is_drum;
        notes.push_back(n);
    }
};

// mt3/vocabularies.py:70-74
static int mt3_bin_to_velocity(int b, int num_bins) {
    if (b == 0)
        return 0;
    return (int)(127 * b / num_bins);
}

// mt3/note_sequences.py:313-387. Returns false where upstream raises
// ValueError — the caller counts it invalid and CONTINUES (this is a spec'd,
// exercised path, not an error branch).
static bool mt3_decode_note_event(mt3_decode_state& st, double time, mt3_event_type type, int value,
                                  int num_velocity_bins) {
    if (time < st.current_time)
        return false; // non-monotonic time
    st.current_time = time;

    switch (type) {
    case MT3_EV_PITCH: {
        const int prog = st.current_program;
        const int idx = st.find_active(value, prog);
        if (st.is_tie_section) {
            if (idx < 0)
                return false; // inactive pitch/program declared in a tie section
            if (st.is_tied(value, prog))
                return false; // already tied
            st.tied_pitches.emplace_back(value, prog);
            st.n_tied_pitches++;
            return true;
        }
        if (st.current_velocity == 0) {
            if (idx < 0)
                return false; // note-off for an inactive pitch/program
            const mt3_active a = st.active_pitches[idx];
            st.active_pitches.erase(st.active_pitches.begin() + idx);
            st.add_note(a.onset, time, value, a.velocity, prog, false);
            return true;
        }
        if (idx >= 0) {
            // Re-onset of a still-sounding note: close the old one first.
            const mt3_active a = st.active_pitches[idx];
            st.active_pitches.erase(st.active_pitches.begin() + idx);
            st.add_note(a.onset, time, value, a.velocity, prog, false);
        }
        mt3_active na;
        na.pitch = value;
        na.program = prog;
        na.onset = time;
        na.velocity = st.current_velocity;
        st.active_pitches.push_back(na); // dict insert → back of the order
        return true;
    }
    case MT3_EV_DRUM:
        if (st.current_velocity == 0)
            return false; // velocity cannot be zero for a drum event
        st.add_note(time, time + MT3_DEFAULT_NOTE_DURATION, value, st.current_velocity, 0, true);
        return true;
    case MT3_EV_VELOCITY:
        st.current_velocity = mt3_bin_to_velocity(value, num_velocity_bins);
        return true;
    case MT3_EV_PROGRAM:
        // Deliberately carries across segment boundaries, and re-declares the
        // program inside a tie section (the tie list is (program, pitch) pairs).
        st.current_program = value;
        return true;
    case MT3_EV_TIE: {
        if (!st.is_tie_section)
            return false;
        // End of section: every active (pitch, program) NOT declared tied is
        // closed at current_time. A tie declaration that was REJECTED above is
        // therefore closed here — that is the intended behaviour, and getting
        // it wrong yields correct-looking notes with wrong durations.
        for (size_t i = 0; i < st.active_pitches.size();) {
            const mt3_active& a = st.active_pitches[i];
            if (!st.is_tied(a.pitch, a.program)) {
                const mt3_active cp = a;
                st.active_pitches.erase(st.active_pitches.begin() + i);
                st.add_note(cp.onset, st.current_time, cp.pitch, cp.velocity, cp.program, false);
            } else {
                i++;
            }
        }
        st.is_tie_section = false;
        st.n_tie_ends++;
        return true;
    }
    default:
        return false;
    }
}

struct mt3_segment_stats {
    int invalid = 0;
    int dropped = 0;
};

// mt3/run_length_encoding.py:371-423 over one segment's tokens.
//
// TIME SEMANTICS: `cur_steps` accumulates across CONSECUTIVE shift tokens and
// is reset to 0 by any non-shift event; cur_time = start_time + cur_steps/sps.
// Shifts are ABSOLUTE within the segment, not deltas.
static mt3_segment_stats mt3_decode_segment(mt3_decode_state& st, const std::vector<int>& tokens, double start_time,
                                            bool has_max_time, double max_time, const mt3_codec& codec,
                                            int num_special_tokens, int padded_vocab, int eos_id) {
    mt3_segment_stats stats;
    long long cur_steps = 0;
    double cur_time = start_time;

    for (size_t idx = 0; idx < tokens.size(); idx++) {
        const int tid = tokens[idx];
        // vocabularies.py:211-219 GenericTokenVocabulary._decode_id
        int cid;
        if (tid == eos_id)
            break;
        if (tid < num_special_tokens || tid >= padded_vocab)
            cid = -2;
        else
            cid = tid - num_special_tokens;
        if (cid < 0 || cid >= codec.num_classes) {
            stats.invalid++;
            continue;
        }
        mt3_event_type type;
        int value;
        if (!codec.decode(cid, type, value)) {
            stats.invalid++;
            continue;
        }
        if (type == MT3_EV_SHIFT) {
            cur_steps += value;
            cur_time = start_time + (double)cur_steps / (double)codec.steps_per_second;
            if (has_max_time && cur_time > max_time) {
                stats.dropped = (int)(tokens.size() - idx);
                break;
            }
        } else {
            cur_steps = 0;
            if (!mt3_decode_note_event(st, cur_time, type, value, codec.num_velocity_bins))
                stats.invalid++;
        }
    }
    return stats;
}

// mt3/note_sequences.py:396-408 — push current_time past every dangling onset,
// then close everything there. Insertion order is preserved (see the comment on
// mt3_decode_state::active_pitches).
static void mt3_flush(mt3_decode_state& st) {
    for (const auto& a : st.active_pitches)
        st.current_time = std::max(st.current_time, a.onset + MT3_MIN_NOTE_DURATION);
    while (!st.active_pitches.empty()) {
        const mt3_active a = st.active_pitches.front();
        st.active_pitches.erase(st.active_pitches.begin());
        st.add_note(a.onset, st.current_time, a.pitch, a.velocity, a.program, false);
    }
    std::stable_sort(st.notes.begin(), st.notes.end(), [](const mt3_note& x, const mt3_note& y) {
        if (x.start_s != y.start_s)
            return x.start_s < y.start_s;
        if (x.program != y.program)
            return x.program < y.program;
        return x.pitch < y.pitch;
    });
}

// mt3/note_sequences.py:72-84 assign_instruments — programs get track indices in
// first-appearance order SKIPPING 9, drums always land on 9 (GM channel 10).
static void mt3_assign_instruments(std::vector<mt3_note>& notes, std::vector<int>& instrument) {
    instrument.assign(notes.size(), 0);
    std::map<int, int> prog_to_inst;
    int next_inst = 0;
    for (size_t i = 0; i < notes.size(); i++) {
        if (notes[i].is_drum) {
            instrument[i] = 9;
            continue;
        }
        auto it = prog_to_inst.find(notes[i].program);
        if (it == prog_to_inst.end()) {
            if (next_inst == 9)
                next_inst = 10;
            prog_to_inst[notes[i].program] = next_inst;
            it = prog_to_inst.find(notes[i].program);
            next_inst++;
        }
        instrument[i] = it->second;
    }
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct mt3_params mt3_default_params(void) {
    mt3_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false; // CPU by default; CRISPASR_MT3_GPU=1 opts in
    p.max_decode_steps = 0;
    p.max_segments = 0;
    return p;
}

extern "C" struct mt3_context* mt3_init_from_file(const char* path, struct mt3_params params) {
    auto* c = new mt3_context();
    c->params = params;

    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            delete c;
            return nullptr;
        }
        const std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "mt3") {
            fprintf(stderr, "mt3: '%s' has architecture '%s', expected 'mt3'\n", path, arch.c_str());
            core_gguf::free_metadata(g);
            delete c;
            return nullptr;
        }
        const bool ok = mt3_load_metadata(c, g);
        core_gguf::free_metadata(g);
        if (!ok) {
            delete c;
            return nullptr;
        }
    }

    const auto& hp = c->model.hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "mt3: d=%d d_kv=%d enc=%dL dec=%dL heads=%d ff=%d vocab=%d pos=%s scale=%.3f\n", hp.d_model,
                hp.d_kv, hp.enc_layers, hp.dec_layers, hp.n_heads, hp.d_ff, hp.vocab_size, hp.pos_embed.c_str(),
                (double)hp.attn_logit_scale);
        fprintf(stderr, "mt3: mel %d bins %d..%.0f Hz n_fft=%d hop=%d, %d-frame segments (%.3f s), codec %d classes\n",
                hp.num_mel_bins, (int)hp.mel_lo_hz, (double)hp.mel_hi_hz, hp.n_fft, hp.hop_width, hp.inputs_length,
                (double)hp.inputs_length * hp.hop_width / hp.sample_rate, c->codec.num_classes);
    }

    c->backend_cpu = core_cpu_backend::init();
    c->backend = c->backend_cpu;
    {
        const char* gpu_env = crispasr_env::get("CRISPASR_MT3_GPU");
        const bool force_gpu = gpu_env && std::atoi(gpu_env) != 0;
        const bool force_cpu = gpu_env && std::atoi(gpu_env) == 0;
        if (!force_cpu && (force_gpu || params.use_gpu)) {
            ggml_backend_t gpu = crispasr_init_gpu_backend();
            // On a box with no GPU this hands back a *second* CPU backend.
            // Taking it would put two CPU backends in the scheduler and print
            // a misleading "GPU backend enabled (CPU)" line, so drop it.
            // core_cpu_backend::is_cpu, NOT ggml_backend_is_cpu: under
            // GGML_BACKEND_DL (#355) the CPU backend is a dlopen-ed module and
            // that symbol is not linkable, so a direct call builds everywhere
            // except the linux-backend-dl job — which is exactly where it
            // failed (undefined reference, run 33712251884).
            if (gpu && core_cpu_backend::is_cpu(gpu)) {
                ggml_backend_free(gpu);
                gpu = nullptr;
            }
            if (gpu) {
                c->backend = gpu;
                if (params.verbosity >= 1)
                    fprintf(stderr, "mt3: GPU backend enabled (%s)\n", ggml_backend_name(c->backend));
            }
        }
    }

    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path, c->backend, "mt3", wl)) {
            mt3_free(c);
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    if (!mt3_bind(c)) {
        mt3_free(c);
        return nullptr;
    }

    {
        ggml_backend_t backends[] = {c->backend, c->backend_cpu};
        const int n_be = (c->backend != c->backend_cpu) ? 2 : 1;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, MT3_GRAPH_NODES, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * MT3_GRAPH_NODES +
                               ggml_graph_overhead_custom(MT3_GRAPH_NODES, false));
    }

    c->fe.init(hp.n_fft, hp.hop_width, hp.num_mel_bins, (double)hp.sample_rate, (double)hp.mel_lo_hz,
               (double)hp.mel_hi_hz, hp.log_eps);

    c->capture = mt3_env_on("CRISPASR_MT3_DIFF");

    // Parity levers. The numpy reference decoder is non-cached (O(T^2) re-run
    // of the whole prefix every step), so a 1024-step reference run over a
    // multi-segment file is hours; a gate run caps BOTH sides at the same
    // budget with these. Production leaves them unset and gets
    // mt3.targets_length / the whole file.
    if (const char* v = crispasr_env::get("CRISPASR_MT3_MAX_DECODE_STEPS")) {
        const int n = std::atoi(v);
        if (n > 0)
            c->params.max_decode_steps = n;
    }
    if (const char* v = crispasr_env::get("CRISPASR_MT3_MAX_SEGMENTS")) {
        const int n = std::atoi(v);
        if (n > 0)
            c->params.max_segments = n;
    }
    return c;
}

extern "C" void mt3_free(struct mt3_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cross_kv_buf)
        ggml_backend_buffer_free(ctx->cross_kv_buf);
    if (ctx->cross_kv_ctx)
        ggml_free(ctx->cross_kv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" uint32_t mt3_sample_rate(const struct mt3_context* ctx) {
    return ctx ? (uint32_t)ctx->model.hp.sample_rate : 16000u;
}

extern "C" uint32_t mt3_segment_samples(const struct mt3_context* ctx) {
    return ctx ? (uint32_t)(ctx->model.hp.inputs_length * ctx->model.hp.hop_width) : 32768u;
}

extern "C" void mt3_result_free(struct mt3_result* result) {
    if (!result)
        return;
    free(result->notes);
    result->notes = nullptr;
    result->n_notes = 0;
}

// Compute one segment's log-mel, zero-padded to inputs_length rows.
//
// seqio pads the `inputs` feature AFTER the preprocessors have run, so the pad
// is literal 0.0 MEL ROWS (log-magnitude 0, i.e. magnitude 1.0 — not silence),
// and network.py:286-289 deliberately leaves them UNMASKED. Truncating instead
// changes the output; phase 1 measured a 113-frame tail going from 2 tokens to
// 9 when padded correctly.
static void mt3_segment_mel(const mt3_context* c, const float* pcm, int n_samples, const mt3_segment& seg,
                            std::vector<float>& mel) {
    const auto& hp = c->model.hp;
    const int T = hp.inputs_length;
    const int n_mel = hp.num_mel_bins;
    const int want = seg.sample_end - seg.sample_begin;
    std::vector<float> chunk((size_t)want, 0.0f);
    for (int i = 0; i < want; i++) {
        const int idx = seg.sample_begin + i;
        chunk[i] = (idx < n_samples) ? pcm[idx] : 0.0f; // hop zero-pad
    }
    int n_frames = 0;
    std::vector<float> m;
    c->fe.logmel(chunk.data(), want, m, n_frames);
    mel.assign((size_t)T * (size_t)n_mel, 0.0f);
    const int copy = std::min(T, n_frames);
    std::memcpy(mel.data(), m.data(), (size_t)copy * (size_t)n_mel * sizeof(float));
}

extern "C" int mt3_transcribe(struct mt3_context* ctx, const float* pcm, int n_samples, struct mt3_result* result) {
    if (!ctx || !pcm || n_samples <= 0 || !result)
        return 1;
    const auto& hp = ctx->model.hp;
    *result = mt3_result{};

    std::vector<mt3_segment> segs =
        mt3_split_segments(n_samples, hp.hop_width, hp.inputs_length, hp.sample_rate, ctx->codec.steps_per_second);
    if (segs.empty())
        return 0;
    int n_run = (int)segs.size();
    if (ctx->params.max_segments > 0)
        n_run = std::min(n_run, ctx->params.max_segments);

    int max_steps = ctx->params.max_decode_steps > 0 ? ctx->params.max_decode_steps : hp.targets_length;
    if (!mt3_alloc_kv_cache(ctx, max_steps + 4))
        return 2;

    const int padded_vocab = hp.vocab_size;
    mt3_decode_state st;
    int total_tokens = 0, total_invalid = 0, total_dropped = 0;

    std::vector<float> mel, enc_out;
    std::vector<int> tokens;
    for (int i = 0; i < n_run; i++) {
        mt3_segment_mel(ctx, pcm, n_samples, segs[i], mel);
        if (!mt3_run_encoder(ctx, mel.data(), hp.inputs_length, enc_out))
            return 2;
        if (!mt3_compute_cross_kv(ctx, enc_out.data(), hp.inputs_length))
            return 2;
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        if (!mt3_greedy(ctx, max_steps, tokens, nullptr))
            return 2;

        // metrics_utils.py:92-116 — ONE decoding state for the whole file. Only
        // the tie bookkeeping is reset per segment (note_sequences.py:390-393).
        st.tied_pitches.clear();
        st.is_tie_section = true;
        const bool has_max = (i + 1 < n_run);
        const double max_time = has_max ? segs[i + 1].start_time : 0.0;
        mt3_segment_stats s = mt3_decode_segment(st, tokens, segs[i].start_time, has_max, max_time, ctx->codec,
                                                 hp.num_special_tokens, padded_vocab, hp.eos_id);
        total_tokens += (int)tokens.size();
        total_invalid += s.invalid;
        total_dropped += s.dropped;
        if (ctx->params.verbosity >= 2) {
            fprintf(stderr, "mt3: segment %d t=%.2f: %zu tokens, %d invalid, %d dropped\n", i, segs[i].start_time,
                    tokens.size(), s.invalid, s.dropped);
            fprintf(stderr, "  tokens:");
            for (int t : tokens)
                fprintf(stderr, " %d", t);
            fprintf(stderr, "\n");
        }
    }
    mt3_flush(st);

    std::vector<int> instrument;
    mt3_assign_instruments(st.notes, instrument);

    result->n_notes = (int)st.notes.size();
    result->n_segments = n_run;
    result->n_tokens = total_tokens;
    result->n_invalid = total_invalid;
    result->n_dropped = total_dropped;
    result->n_tie_ends = st.n_tie_ends;
    result->n_tied_pitches = st.n_tied_pitches;
    result->notes = nullptr;
    if (result->n_notes > 0) {
        result->notes = (mt3_note_event*)calloc((size_t)result->n_notes, sizeof(mt3_note_event));
        if (!result->notes)
            return 2;
        for (int i = 0; i < result->n_notes; i++) {
            result->notes[i].start_time = (float)st.notes[i].start_s;
            result->notes[i].end_time = (float)st.notes[i].end_s;
            result->notes[i].pitch = st.notes[i].pitch;
            result->notes[i].velocity = st.notes[i].velocity;
            result->notes[i].program = st.notes[i].program;
            result->notes[i].is_drum = st.notes[i].is_drum;
            result->notes[i].instrument = instrument[i];
        }
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr,
                "mt3: %d segments, %d tokens, %d invalid, %d dropped, %d tie-section ends, "
                "%d tied pitches -> %d notes\n",
                n_run, total_tokens, total_invalid, total_dropped, result->n_tie_ends, result->n_tied_pitches,
                result->n_notes);
    return 0;
}

// ===========================================================================
// Diff harness (#333)
// ===========================================================================

static double mt3_cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0)
        return (na == nb) ? 1.0 : 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static double mt3_max_abs(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

static bool mt3_ref_get(const core_gguf::WeightLoad& wl, const char* name, std::vector<float>& out) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end())
        return false;
    ggml_tensor* t = it->second;
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return true;
}

extern "C" int mt3_diff(const char* model_gguf, const char* ref_gguf, const float* pcm_16k, int n_samples,
                        int verbosity) {
    mt3_params p = mt3_default_params();
    p.verbosity = verbosity;
    mt3_context* ctx = mt3_init_from_file(model_gguf, p);
    if (!ctx) {
        fprintf(stderr, "mt3_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    ctx->capture = true;

    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "mt3_ref", rw)) {
        fprintf(stderr, "mt3_diff: failed to load reference %s\n", ref_gguf);
        mt3_free(ctx);
        return 2;
    }

    const auto& hp = ctx->model.hp;
    int n_fail = 0;
    const double COS_MIN = 0.999;
    auto report = [&](const char* stage, const std::vector<float>& mine, const std::vector<float>& ref,
                      double cos_min) {
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = mt3_cosine(mine.data(), ref.data(), n);
        const double mad = mt3_max_abs(mine.data(), ref.data(), n);
        const bool ok = cos >= cos_min && mine.size() == ref.size();
        if (!ok)
            n_fail++;
        fprintf(stderr, "  %-14s %s cos=%.9f max_abs=%.3e  (mine=%zu ref=%zu)\n", stage, ok ? "PASS" : "FAIL", cos, mad,
                mine.size(), ref.size());
    };

    std::vector<mt3_segment> segs =
        mt3_split_segments(n_samples, hp.hop_width, hp.inputs_length, hp.sample_rate, ctx->codec.steps_per_second);
    if (segs.empty()) {
        fprintf(stderr, "mt3_diff: empty audio\n");
        core_gguf::free_weights(rw);
        mt3_free(ctx);
        return 2;
    }
    fprintf(stderr, "mt3 diff (n_samples=%d, %zu segments of %d frames):\n", n_samples, segs.size(), hp.inputs_length);

    std::vector<float> r;

    // Segment 0's exact samples — a resampler or segmentation difference then
    // shows up as itself instead of shifting every downstream cosine.
    {
        const int want = segs[0].sample_end - segs[0].sample_begin;
        std::vector<float> chunk((size_t)want, 0.0f);
        for (int i = 0; i < want; i++) {
            const int idx = segs[0].sample_begin + i;
            chunk[i] = (idx < n_samples) ? pcm_16k[idx] : 0.0f;
        }
        if (mt3_ref_get(rw, "audio_segment0", r))
            report("audio_seg0", chunk, r, 0.9999);
    }

    std::vector<float> mel;
    mt3_segment_mel(ctx, pcm_16k, n_samples, segs[0], mel);
    if (mt3_ref_get(rw, "mel", r))
        report("mel", mel, r, 0.9999); // highest-risk stage: tighter gate

    // mel_all: every segment, short final one zero-padded to inputs_length.
    if (mt3_ref_get(rw, "mel_all", r)) {
        std::vector<float> all;
        const size_t per = (size_t)hp.inputs_length * (size_t)hp.num_mel_bins;
        const size_t n_seg_ref = r.size() / per;
        std::vector<float> m;
        for (size_t i = 0; i < segs.size() && i < n_seg_ref; i++) {
            mt3_segment_mel(ctx, pcm_16k, n_samples, segs[i], m);
            all.insert(all.end(), m.begin(), m.end());
        }
        report("mel_all", all, r, 0.9999);
    }

    std::vector<float> enc_out;
    if (!mt3_run_encoder(ctx, mel.data(), hp.inputs_length, enc_out)) {
        fprintf(stderr, "mt3_diff: encoder failed\n");
        core_gguf::free_weights(rw);
        mt3_free(ctx);
        return 2;
    }
    if (mt3_ref_get(rw, "enc_input", r))
        report("enc_input", ctx->cap_enc_input, r, COS_MIN);
    if (mt3_ref_get(rw, "enc_out", r))
        report("enc_out", enc_out, r, COS_MIN);

    // Decoder: greedy over segment 0, collecting every step's logits.
    std::vector<float> ref_prefix;
    const bool have_prefix = mt3_ref_get(rw, "logits_prefix", ref_prefix);
    int steps = have_prefix ? (int)(ref_prefix.size() / hp.vocab_size) : 64;
    if (steps <= 0)
        steps = 64;

    if (!mt3_compute_cross_kv(ctx, enc_out.data(), hp.inputs_length) || !mt3_alloc_kv_cache(ctx, steps + 4)) {
        fprintf(stderr, "mt3_diff: decoder setup failed\n");
        core_gguf::free_weights(rw);
        mt3_free(ctx);
        return 2;
    }
    std::vector<int> tokens;
    std::vector<float> all_logits;
    if (!mt3_greedy(ctx, steps, tokens, &all_logits)) {
        fprintf(stderr, "mt3_diff: greedy decode failed\n");
        core_gguf::free_weights(rw);
        mt3_free(ctx);
        return 2;
    }

    if (mt3_ref_get(rw, "logits_step0", r)) {
        std::vector<float> mine(all_logits.begin(),
                                all_logits.begin() + std::min((size_t)hp.vocab_size, all_logits.size()));
        report("logits_step0", mine, r, COS_MIN);
        int am_mine = 0, am_ref = 0;
        for (int i = 1; i < (int)mine.size(); i++)
            if (mine[i] > mine[am_mine])
                am_mine = i;
        for (int i = 1; i < (int)r.size(); i++)
            if (r[i] > r[am_ref])
                am_ref = i;
        const bool ok = (am_mine == am_ref);
        if (!ok)
            n_fail++;
        fprintf(stderr, "  %-14s %s argmax mine=%d ref=%d\n", "argmax_step0", ok ? "PASS" : "FAIL", am_mine, am_ref);
    }

    if (have_prefix) {
        const int n_cmp = std::min(steps, (int)(all_logits.size() / hp.vocab_size));
        std::vector<float> mine(all_logits.begin(), all_logits.begin() + (size_t)n_cmp * hp.vocab_size);
        std::vector<float> refp(ref_prefix.begin(), ref_prefix.begin() + (size_t)n_cmp * hp.vocab_size);
        report("logits_prefix", mine, refp, COS_MIN);
        // The real signal for a decoder: identical greedy tokens, step by step.
        int agree = 0;
        for (int s = 0; s < n_cmp; s++) {
            int am = 0, ar = 0;
            const float* mp = mine.data() + (size_t)s * hp.vocab_size;
            const float* rp = refp.data() + (size_t)s * hp.vocab_size;
            for (int i = 1; i < hp.vocab_size; i++) {
                if (mp[i] > mp[am])
                    am = i;
                if (rp[i] > rp[ar])
                    ar = i;
            }
            if (am != ar)
                break;
            agree++;
        }
        const bool ok = (agree == n_cmp);
        if (!ok)
            n_fail++;
        fprintf(stderr, "  %-14s %s greedy tokens identical for %d/%d steps\n", "greedy_tokens", ok ? "PASS" : "FAIL",
                agree, n_cmp);
    }

    core_gguf::free_weights(rw);
    mt3_free(ctx);
    fprintf(stderr, "mt3 diff: %s (%d failing stage%s)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail,
            n_fail == 1 ? "" : "s");
    return n_fail == 0 ? 0 : 1;
}
