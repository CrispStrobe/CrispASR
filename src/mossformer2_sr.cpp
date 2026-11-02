// Native ggml port of ClearerVoice-Studio MossFormer2_SR_48K (see
// mossformer2_sr.h).  Network: models/mossformer2_sr/generator.py (Mossformer ->
// MossFormer_MaskNet, Generator); the MossFormerM stack is shared with
// MossFormer2_SS_16K / SE_48K (mossformer2_blocks.{h,cpp}).  Front/back end:
// utils/decode.py decode_one_audio_mossformer2_sr_48k + get_mel,
// dataloader/meldataset.py mel_spectrogram and utils/bandwidth_sub.py.
#include "mossformer2_sr.h"
#include "mossformer2_blocks.h"
#include "mossformer2_se.h" // mf2se::segment_plan (identical window plan)

#include "core/activation.h" // core_act::snake_alpha
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/mel.h" // core_mel::build_slaney_fb
#include "core/parallel_for.h"
#include "crispasr_enhancer.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
constexpr size_t GRAPH_SIZE = 32768;
// Fixed by MossFormer2_SR_48K.json / get_mel; read_hparams() rejects GGUFs that
// disagree, so the model-free helpers can use them directly.
constexpr int SR = 48000, N_MELS = 80, NFFT = 1024, HOP = 256, WIN = 1024, BINS = NFFT / 2 + 1;
constexpr float MEL_FMAX = 8000.0f, LOG_CLIP = 1e-5f;
constexpr int GEN_CHUNK_DEFAULT = 320; // generator frames per chunk (core, margins excluded)

bool env_flag(const char* key, bool fallback = false) {
    const char* v = std::getenv(key);
    return v && *v ? std::strcmp(v, "0") != 0 : fallback;
}

struct StageTimer {
    const char* name;
    bool enabled = env_flag("CRISPASR_MF2SR_BENCH");
    std::chrono::steady_clock::time_point start;
    explicit StageTimer(const char* n) : name(n) {
        if (enabled)
            start = std::chrono::steady_clock::now();
    }
    ~StageTimer() {
        if (enabled)
            std::fprintf(stderr, "mossformer2_sr_bench: %s %.3f ms\n", name,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
};

struct HParams {
    int sample_rate = 0, num_mels = 0, n_fft = 0, hop = 0, win = 0, fmin = -1, fmax = 0, num_spks = 0;
    int up_initial = 0, gen_kernel = 0;
    std::vector<int> up_rates, up_kernels, rb_kernels, rb_dilations;
    mf2::BlockHParams blk;
    float one_time_decode_length = 0, decode_window = 0, decode_stride_ratio = 0, log_clip = 0;
    float bw_threshold = 0, crossfade_ms = 0;
};

std::vector<int> kv_ints(gguf_context* g, const std::string& key) {
    std::vector<int> out;
    const int64_t k = gguf_find_key(g, key.c_str());
    if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY)
        return out;
    const enum gguf_type t = gguf_get_arr_type(g, k);
    const size_t n = gguf_get_arr_n(g, k);
    const void* data = gguf_get_arr_data(g, k);
    for (size_t i = 0; i < n; ++i) {
        if (t == GGUF_TYPE_INT32)
            out.push_back(((const int32_t*)data)[i]);
        else if (t == GGUF_TYPE_UINT32)
            out.push_back((int)((const uint32_t*)data)[i]);
        else if (t == GGUF_TYPE_INT64)
            out.push_back((int)((const int64_t*)data)[i]);
        else
            return {};
    }
    return out;
}

bool read_hparams(gguf_context* meta, HParams& h) {
    if (core_gguf::kv_str(meta, "general.architecture", "") != "mossformer2-sr")
        return false;
    auto u = [&](const char* key, uint32_t fallback = 0) {
        return (int)core_gguf::kv_u32(meta, (std::string("mossformer2_sr.") + key).c_str(), fallback);
    };
    auto f = [&](const char* key, float fallback = 0) {
        return core_gguf::kv_f32(meta, (std::string("mossformer2_sr.") + key).c_str(), fallback);
    };
    h.sample_rate = u("sample_rate");
    h.num_mels = u("num_mels");
    h.n_fft = u("n_fft");
    h.hop = u("hop_size");
    h.win = u("win_size");
    h.fmin = u("mel_fmin", 1);
    h.fmax = u("mel_fmax");
    h.num_spks = u("num_spks");
    h.blk.dim = u("mossformer_sequence_dim");
    h.blk.n_layers = u("num_mossformer_layer");
    h.blk.group = u("group_size");
    h.blk.qk = u("query_key_dim");
    h.blk.hidden = u("hidden_dim");
    h.blk.fsmn_inner = u("fsmn_inner_channels");
    h.blk.fsmn_lorder = u("fsmn_lorder");
    h.blk.conv_kernel = u("conv_module_kernel");
    h.blk.rotary_dim = u("rotary_dim");
    h.blk.rotary_theta = f("rotary_theta");
    h.blk.fsmn_dilated = false; // Gated_FSMN_Block / UniDeepFsmn
    h.up_initial = u("upsample_initial_channel");
    h.gen_kernel = u("gen_conv_kernel");
    h.up_rates = kv_ints(meta, "mossformer2_sr.upsample_rates");
    h.up_kernels = kv_ints(meta, "mossformer2_sr.upsample_kernel_sizes");
    h.rb_kernels = kv_ints(meta, "mossformer2_sr.resblock_kernel_sizes");
    h.rb_dilations = kv_ints(meta, "mossformer2_sr.resblock_dilations");
    h.one_time_decode_length = f("one_time_decode_length");
    h.decode_window = f("decode_window");
    h.decode_stride_ratio = f("decode_stride_ratio");
    h.log_clip = f("mel_log_clip");
    h.bw_threshold = f("bandwidth_energy_threshold");
    h.crossfade_ms = f("bandwidth_crossfade_ms", -1);
    int total_up = 1;
    bool gen_ok = h.up_rates.size() == h.up_kernels.size() && !h.up_rates.empty() && !h.rb_kernels.empty() &&
                  !h.rb_dilations.empty() && h.up_initial > 0 && (h.up_initial >> h.up_rates.size()) > 0 &&
                  h.gen_kernel % 2 == 1;
    for (size_t i = 0; gen_ok && i < h.up_rates.size(); ++i) {
        gen_ok = h.up_rates[i] > 0 && h.up_kernels[i] >= h.up_rates[i] && (h.up_kernels[i] - h.up_rates[i]) % 2 == 0;
        total_up *= h.up_rates[i];
    }
    for (int k : h.rb_kernels)
        gen_ok = gen_ok && k > 0 && k % 2 == 1;
    for (int d : h.rb_dilations)
        gen_ok = gen_ok && d > 0;
    // Versioned architecture: reject metadata this runtime was not validated on
    // before a ggml shape assertion can abort the host process.
    return gen_ok && total_up == HOP && h.sample_rate == SR && h.num_mels == N_MELS && h.n_fft == NFFT &&
           h.hop == HOP && h.win == WIN && h.fmin == 0 && h.fmax == (int)MEL_FMAX && h.num_spks == 1 &&
           h.blk.dim == 512 && h.blk.n_layers >= 1 && h.blk.n_layers <= 64 && h.blk.group == 256 && h.blk.qk == 128 &&
           h.blk.hidden == 2048 && h.blk.fsmn_inner == 256 && h.blk.fsmn_lorder == 20 && h.blk.conv_kernel == 17 &&
           h.blk.rotary_dim == 32 && h.blk.rotary_theta == 10000.0f && h.one_time_decode_length > 0 &&
           h.decode_window > 0 && h.decode_stride_ratio > 0 && h.decode_stride_ratio <= 1 && h.log_clip == LOG_CLIP &&
           h.bw_threshold > 0 && h.bw_threshold < 1 && h.crossfade_ms >= 0;
}

std::vector<float> tensor_f32(ggml_tensor* t) {
    std::vector<float> out((size_t)ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(out.size());
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)out.size());
    } else {
        throw std::runtime_error(std::string("expected F32/F16 tensor: ") + ggml_get_name(t));
    }
    return out;
}

using cpx = std::complex<double>;

// Iterative radix-2 complex FFT (double) with a precomputed twiddle table.
struct Fft2 {
    int N = 0;
    std::vector<cpx> w;
    std::vector<int> rev;
    explicit Fft2(int n) : N(n), w((size_t)n / 2), rev((size_t)n) {
        for (int k = 0; k < n / 2; ++k)
            w[(size_t)k] = std::polar(1.0, -2.0 * M_PI * (double)k / (double)n);
        int bits = 0;
        while ((1 << bits) < n)
            ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b)
                r |= ((i >> b) & 1) << (bits - 1 - b);
            rev[(size_t)i] = r;
        }
    }
    // Real input `x` (N samples) -> bins 0..N/2.
    void rfft(const double* x, cpx* out) const {
        std::vector<cpx> a((size_t)N);
        for (int i = 0; i < N; ++i)
            a[(size_t)rev[(size_t)i]] = cpx(x[i], 0.0);
        for (int len = 2; len <= N; len <<= 1) {
            const int half = len / 2, step = N / len;
            for (int i = 0; i < N; i += len)
                for (int j = 0; j < half; ++j) {
                    const cpx t = w[(size_t)j * step] * a[(size_t)(i + j + half)];
                    a[(size_t)(i + j + half)] = a[(size_t)(i + j)] - t;
                    a[(size_t)(i + j)] += t;
                }
        }
        std::copy_n(a.begin(), N / 2 + 1, out);
    }
};

const Fft2& fft_of(int n) {
    static const Fft2 f1024(NFFT), f256(256);
    if (n == NFFT)
        return f1024;
    if (n == 256)
        return f256;
    throw std::runtime_error("unsupported FFT size");
}

using Captures = std::map<std::string, std::vector<float>>;

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *in = nullptr, *pe = nullptr, *positions = nullptr, *out = nullptr;
    std::map<std::string, ggml_tensor*> captures;
    int T = 0;
    std::set<std::string> wanted;
    ~Graph() { clear(); }
    void clear() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (ctx)
            ggml_free(ctx);
        sched = nullptr;
        ctx = nullptr;
        graph = nullptr;
        in = pe = positions = out = nullptr;
        captures.clear();
        T = 0;
        wanted.clear();
    }
};
} // namespace

struct mossformer2_sr_context {
    HParams hp;
    int n_threads = 4;
    bool verbose = false;
    int gen_chunk = GEN_CHUNK_DEFAULT; // frames; 0 = one pass
    int gen_margin = 0;                // frames of context on each side of a chunk
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    ggml_context* derived_ctx = nullptr;
    ggml_backend_buffer_t derived_buf = nullptr;
    std::map<std::string, ggml_tensor*> derived;
    std::vector<float> inv_freq, mel_fb;
    float pos_scale = 1.0f;
    bool quantized = false;
    bool portable_dwconv = false;
    Graph masknet, generator;
};

// ---------------------------------------------------------------------------
// Model-free front / back end
// ---------------------------------------------------------------------------
namespace mf2sr {

int log_mel(const float* x, int n, std::vector<float>& mel, int n_threads) {
    mel.clear();
    const int pad = (NFFT - HOP) / 2; // 384
    if (n <= pad)
        return 0; // torch reflect padding needs pad < length
    static const std::vector<float> fb = core_mel::build_slaney_fb(SR, NFFT, N_MELS, 0.0f, MEL_FMAX);
    static const std::vector<double> hann = [] {
        std::vector<double> v(WIN);
        for (int i = 0; i < WIN; ++i) // torch.hann_window(periodic=True)
            v[(size_t)i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i / (double)WIN);
        return v;
    }();
    // F.pad(mode='reflect'): x[pad], ..., x[1], x[0..n), x[n-2], ..., x[n-1-pad]
    std::vector<float> xp((size_t)n + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        xp[(size_t)i] = x[pad - i];
        xp[(size_t)(n + pad + i)] = x[n - 2 - i];
    }
    std::copy_n(x, n, xp.begin() + pad);
    const int T = (int)((xp.size() - NFFT) / HOP) + 1;
    mel.assign((size_t)T * N_MELS, 0.0f);
    const Fft2& fft = fft_of(NFFT);
    core_parallel::for_each_chunk(T, n_threads, [&](int a, int b) {
        std::vector<double> buf(NFFT);
        std::vector<cpx> spec(BINS);
        std::vector<float> mag(BINS);
        for (int t = a; t < b; ++t) {
            const float* src = xp.data() + (size_t)t * HOP;
            for (int i = 0; i < NFFT; ++i)
                buf[(size_t)i] = (double)src[i] * hann[(size_t)i];
            fft.rfft(buf.data(), spec.data());
            for (int k = 0; k < BINS; ++k) { // sqrt(re^2 + im^2 + 1e-9) in float32
                const float re = (float)spec[(size_t)k].real(), im = (float)spec[(size_t)k].imag();
                mag[(size_t)k] = std::sqrt(re * re + im * im + 1e-9f);
            }
            float* row = mel.data() + (size_t)t * N_MELS;
            for (int m = 0; m < N_MELS; ++m) {
                const float* w = fb.data() + (size_t)m * BINS;
                float acc = 0.0f;
                for (int k = 0; k < BINS; ++k)
                    acc += w[k] * mag[(size_t)k];
                row[m] = std::log(std::max(acc, LOG_CLIP));
            }
        }
    });
    return T;
}

namespace {
// Coefficients of prod_i (x - r_i), highest power first (numpy.poly).
std::vector<cpx> poly(const std::vector<cpx>& roots) {
    std::vector<cpx> c{cpx(1.0, 0.0)};
    for (const cpx& r : roots) {
        std::vector<cpx> next(c.size() + 1, cpx(0.0, 0.0));
        for (size_t i = 0; i < c.size(); ++i) {
            next[i] += c[i];
            next[i + 1] -= c[i] * r;
        }
        c.swap(next);
    }
    return c;
}

// scipy.signal.lfilter (direct form II transposed) with initial state zi.
std::vector<double> lfilter(const Filter& f, const double* x, size_t n, std::vector<double> z) {
    const size_t order = f.a.size() - 1;
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        const double yi = f.b[0] * x[i] + z[0];
        for (size_t j = 0; j + 1 < order; ++j)
            z[j] = f.b[j + 1] * x[i] + z[j + 1] - f.a[j + 1] * yi;
        z[order - 1] = f.b[order] * x[i] - f.a[order] * yi;
        y[i] = yi;
    }
    return y;
}
} // namespace

Filter butter4(double wn, bool highpass) {
    if (!(wn > 0.0 && wn < 1.0))
        throw std::invalid_argument("butter4: critical frequency must be in (0, 1)");
    constexpr int N = 4;
    const double fs = 2.0, warped = 2.0 * fs * std::tan(M_PI * wn / fs);
    // buttap: p = -exp(i pi m / (2N)), m = -N+1, -N+3, ..., N-1; no zeros, k = 1.
    std::vector<cpx> p, z;
    for (int m = -N + 1; m < N; m += 2)
        p.push_back(-std::exp(cpx(0.0, M_PI * m / (2.0 * N))));
    double k = 1.0;
    if (highpass) { // lp2hp_zpk
        cpx prod_neg_p(1.0, 0.0);
        for (const cpx& pi : p)
            prod_neg_p *= -pi;
        k = (cpx(1.0, 0.0) / prod_neg_p).real();
        for (cpx& pi : p)
            pi = warped / pi;
        z.assign(N, cpx(0.0, 0.0));
    } else { // lp2lp_zpk
        for (cpx& pi : p)
            pi *= warped;
        k = std::pow(warped, (double)N);
    }
    // bilinear_zpk(fs = 2)
    const double fs2 = 2.0 * fs;
    cpx num(1.0, 0.0), den(1.0, 0.0);
    std::vector<cpx> zz, pz;
    for (const cpx& zi : z) {
        zz.push_back((fs2 + zi) / (fs2 - zi));
        num *= fs2 - zi;
    }
    for (const cpx& pi : p) {
        pz.push_back((fs2 + pi) / (fs2 - pi));
        den *= fs2 - pi;
    }
    while (zz.size() < pz.size())
        zz.push_back(cpx(-1.0, 0.0));
    k *= (num / den).real();
    const auto bc = poly(zz), ac = poly(pz);
    Filter f;
    for (const cpx& c : bc)
        f.b.push_back(k * c.real());
    for (const cpx& c : ac)
        f.a.push_back(c.real());
    return f;
}

std::vector<double> lfilter_zi(const Filter& f) {
    // zi = solve(I - companion(a).T, b[1:] - a[1:] * b[0])   (a[0] == 1)
    const int n = (int)f.a.size() - 1;
    std::vector<double> m((size_t)n * n, 0.0), rhs((size_t)n);
    for (int i = 0; i < n; ++i) {
        m[(size_t)i * n + i] = 1.0;
        m[(size_t)i * n] += f.a[(size_t)i + 1]; // -(-a[i+1]) in column 0
        if (i + 1 < n)
            m[(size_t)i * n + i + 1] -= 1.0;
        rhs[(size_t)i] = f.b[(size_t)i + 1] - f.a[(size_t)i + 1] * f.b[0];
    }
    for (int c = 0; c < n; ++c) { // Gaussian elimination, partial pivoting
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::abs(m[(size_t)r * n + c]) > std::abs(m[(size_t)piv * n + c]))
                piv = r;
        if (piv != c) {
            for (int j = 0; j < n; ++j)
                std::swap(m[(size_t)c * n + j], m[(size_t)piv * n + j]);
            std::swap(rhs[(size_t)c], rhs[(size_t)piv]);
        }
        for (int r = c + 1; r < n; ++r) {
            const double s = m[(size_t)r * n + c] / m[(size_t)c * n + c];
            for (int j = c; j < n; ++j)
                m[(size_t)r * n + j] -= s * m[(size_t)c * n + j];
            rhs[(size_t)r] -= s * rhs[(size_t)c];
        }
    }
    std::vector<double> zi((size_t)n);
    for (int r = n - 1; r >= 0; --r) {
        double acc = rhs[(size_t)r];
        for (int j = r + 1; j < n; ++j)
            acc -= m[(size_t)r * n + j] * zi[(size_t)j];
        zi[(size_t)r] = acc / m[(size_t)r * n + r];
    }
    return zi;
}

std::vector<double> filtfilt(const Filter& f, const std::vector<double>& x) {
    const size_t padlen = 3 * std::max(f.a.size(), f.b.size());
    const size_t n = x.size();
    if (n <= padlen)
        throw std::invalid_argument("filtfilt: input too short");
    // odd_ext
    std::vector<double> ext(n + 2 * padlen);
    for (size_t i = 0; i < padlen; ++i) {
        ext[i] = 2.0 * x[0] - x[padlen - i];
        ext[n + padlen + i] = 2.0 * x[n - 1] - x[n - 2 - i];
    }
    std::copy(x.begin(), x.end(), ext.begin() + (std::ptrdiff_t)padlen);
    const auto zi = lfilter_zi(f);
    std::vector<double> z(zi.size());
    for (size_t i = 0; i < zi.size(); ++i)
        z[i] = zi[i] * ext[0];
    auto y = lfilter(f, ext.data(), ext.size(), z);
    std::reverse(y.begin(), y.end());
    for (size_t i = 0; i < zi.size(); ++i)
        z[i] = zi[i] * y[0];
    auto yb = lfilter(f, y.data(), y.size(), z);
    std::reverse(yb.begin(), yb.end());
    return std::vector<double>(yb.begin() + (std::ptrdiff_t)padlen, yb.end() - (std::ptrdiff_t)padlen);
}

double detect_bandwidth(const double* x, int64_t n, int fs, double threshold) {
    // scipy.signal.stft(x, fs): Hann(256, periodic), hop 128, boundary='zeros'
    // (128 zeros each side), padded=True; |Z|^2 summed over frames (the
    // 1 / sum(win) scaling cancels in the normalised cumulative energy).
    constexpr int NP = 256, STEP = 128;
    const int64_t len = n + NP;
    const int64_t nadd = ((-(len - NP)) % STEP + STEP) % STEP;
    std::vector<double> xp((size_t)(len + nadd), 0.0);
    std::copy_n(x, n, xp.begin() + NP / 2);
    const int64_t frames = ((int64_t)xp.size() - NP) / STEP + 1;
    static const std::vector<double> win = [] {
        std::vector<double> v(NP);
        for (int i = 0; i < NP; ++i)
            v[(size_t)i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i / (double)NP);
        return v;
    }();
    const Fft2& fft = fft_of(NP);
    std::vector<double> psd(NP / 2 + 1, 0.0), buf(NP);
    std::vector<cpx> spec(NP / 2 + 1);
    for (int64_t t = 0; t < frames; ++t) {
        const double* src = xp.data() + t * STEP;
        for (int i = 0; i < NP; ++i)
            buf[(size_t)i] = src[i] * win[(size_t)i];
        fft.rfft(buf.data(), spec.data());
        for (int k = 0; k <= NP / 2; ++k)
            psd[(size_t)k] += std::norm(spec[(size_t)k]);
    }
    double total = 0.0;
    for (double v : psd)
        total += v;
    if (!(total > 0.0))
        return -1.0;
    double cum = 0.0;
    for (int k = 0; k <= NP / 2; ++k) {
        cum += psd[(size_t)k];
        if (cum / total >= threshold)
            return (double)k * fs / NP;
    }
    return (double)fs / 2;
}

std::vector<double> bandwidth_sub(const std::vector<double>& low, const std::vector<double>& high, int fs,
                                  double threshold, double crossfade_ms, double* f_high_out) {
    const size_t m = std::min(low.size(), high.size());
    const double f_high = detect_bandwidth(low.data(), (int64_t)low.size(), fs, threshold);
    if (f_high_out)
        *f_high_out = f_high;
    const double wn = f_high / (0.5 * fs);
    if (!(wn > 0.0 && wn < 1.0) || low.size() <= 15 || high.size() <= 15) // see mossformer2_sr.h
        return std::vector<double>(low.begin(), low.begin() + (std::ptrdiff_t)m);
    // replace_bandwidth: lowpass(low) + highpass(high), truncated to the shorter
    const auto lo = filtfilt(butter4(wn, false), low);
    const auto hi = filtfilt(butter4(wn, true), high);
    // smooth_transition(substituted, low): linspace(0, 1, L) ramp over the first
    // L = int(crossfade_ms * fs / 1000) samples, from `low` to `substituted`.
    const int64_t L = (int64_t)(crossfade_ms * fs / 1000.0);
    std::vector<double> out(m);
    for (size_t i = 0; i < m; ++i) {
        const double cf = (int64_t)i < L ? (L > 1 ? (double)i / (double)(L - 1) : 0.0) : 1.0;
        out[i] = (1.0 - cf) * low[i] + cf * (hi[i] + lo[i]);
    }
    return out;
}

} // namespace mf2sr

namespace {
ggml_tensor* weight(mossformer2_sr_context& m, const std::string& n) {
    const auto it = m.weights.tensors.find(n);
    if (it == m.weights.tensors.end())
        throw std::runtime_error("missing tensor: " + n);
    return it->second;
}

void expect(mossformer2_sr_context& m, const std::string& n, int64_t d0, int64_t d1 = 1, int64_t d2 = 1) {
    const ggml_tensor* w = weight(m, n);
    if (w->ne[0] != d0 || w->ne[1] != d1 || w->ne[2] != d2 || w->ne[3] != 1)
        throw std::runtime_error("incompatible tensor shape: " + n);
}

int n_resblocks(const HParams& h) {
    return (int)h.rb_kernels.size();
}

std::string rb_name(int stage, int j, const HParams& h) {
    return "gen.rb." + std::to_string(stage * n_resblocks(h) + j) + ".";
}

void validate_weights(mossformer2_sr_context& m) {
    const auto& h = m.hp;
    const auto& b = h.blk;
    const int k = b.conv_kernel, lo = 2 * b.fsmn_lorder - 1;
    expect(m, "mn.norm.weight", h.num_mels);
    expect(m, "mn.norm.bias", h.num_mels);
    expect(m, "mn.conv_enc.weight", h.num_mels, b.dim);
    expect(m, "mn.pos_enc.inv_freq", b.dim / 2);
    expect(m, "mn.pos_enc.scale", 1);
    expect(m, "mn.prelu", 1);
    expect(m, "mn.conv_out.weight", b.dim, b.dim);
    expect(m, "mn.conv_out.bias", b.dim);
    expect(m, "mn.output.weight", b.dim, b.dim);
    expect(m, "mn.output_gate.weight", b.dim, b.dim);
    expect(m, "mn.conv_dec.weight", b.dim, h.num_mels);
    for (int i = 0; i < b.n_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        expect(m, p + "flash.hidden.linear.weight", b.dim, b.hidden);
        expect(m, p + "flash.hidden.dw.weight", k, b.hidden);
        expect(m, p + "flash.qk.linear.weight", b.dim, b.qk);
        expect(m, p + "flash.qk.dw.weight", k, b.qk);
        expect(m, p + "flash.out.linear.weight", b.hidden / 2, b.dim);
        expect(m, p + "flash.out.dw.weight", k, b.dim);
        expect(m, p + "flash.qk_os.gamma", b.qk, 4);
        expect(m, p + "flash.qk_os.beta", b.qk, 4);
        expect(m, p + "fsmn.conv1.weight", b.dim, b.fsmn_inner);
        expect(m, p + "fsmn.conv1.prelu", 1);
        expect(m, p + "fsmn.to_u.dw.weight", k, b.fsmn_inner);
        expect(m, p + "fsmn.to_v.dw.weight", k, b.fsmn_inner);
        expect(m, p + "fsmn.fsmn.project.weight", b.fsmn_inner, b.fsmn_inner);
        expect(m, p + "fsmn.mem.weight", lo, b.fsmn_inner);
        expect(m, p + "fsmn.conv2.weight", b.fsmn_inner, b.dim);
    }
    const int ch = h.up_initial, gk = h.gen_kernel;
    expect(m, "gen.conv_pre.weight", h.num_mels, (int64_t)gk * ch);
    expect(m, "gen.conv_pre.bias", ch);
    for (size_t i = 0; i < h.up_rates.size(); ++i) {
        const int cin = ch >> i, cout = ch >> (i + 1);
        const std::string s = std::to_string(i);
        expect(m, "gen.snakes." + s + ".alpha", cin);
        expect(m, "gen.ups." + s + ".weight", cin, (int64_t)cout * h.up_kernels[i]);
        expect(m, "gen.ups." + s + ".bias", cout);
        for (int j = 0; j < n_resblocks(h); ++j) {
            const std::string r = rb_name((int)i, j, h);
            for (size_t d = 0; d < h.rb_dilations.size(); ++d)
                for (const char* c : {"1", "2"}) {
                    const std::string ds = std::to_string(d);
                    expect(m, r + "c" + c + "." + ds + ".weight", cout, (int64_t)h.rb_kernels[(size_t)j] * cout);
                    expect(m, r + "c" + c + "." + ds + ".bias", cout);
                    expect(m, r + "a" + c + "." + ds, cout);
                }
        }
    }
    const int last = ch >> h.up_rates.size();
    expect(m, "gen.snake_post.alpha", last);
    expect(m, "gen.conv_post.weight", last, gk);
    expect(m, "gen.conv_post.bias", 1);
    for (const auto& kv : m.weights.tensors)
        if (ggml_is_quantized(kv.second->type))
            m.quantized = true;
}

// Receptive half-width of the generator in input frames (rounded up): the
// chunk margin that makes chunked generation equal one-pass generation.
int generator_margin(const HParams& h) {
    double r = (h.gen_kernel / 2) / (double)HOP; // conv_post at the output rate
    int rate = HOP;
    for (int i = (int)h.up_rates.size() - 1; i >= 0; --i) {
        int mrf = 0;
        for (int k : h.rb_kernels) {
            int s = 0;
            for (int d : h.rb_dilations)
                s += d * (k - 1) / 2 + (k - 1) / 2;
            mrf = std::max(mrf, s);
        }
        r += mrf / (double)rate;
        rate /= h.up_rates[(size_t)i];
        r += (double)h.up_kernels[(size_t)i] / h.up_rates[(size_t)i] / rate + 1.0 / rate; // transposed conv reach
    }
    r += h.gen_kernel / 2; // conv_pre at the frame rate
    return (int)std::ceil(r) + 2;
}

// Load-time F32 depthwise kernels on the compute backend: tap-major copies for
// the CPU custom op (the portable path reads the k=17/39 kernels directly).
void prepare_derived(mossformer2_sr_context& m) {
    if (m.portable_dwconv)
        return;
    const auto kernels = mf2::dw_kernels(m.hp.blk);
    ggml_init_params ip{ggml_tensor_overhead() * (kernels.size() + 4), nullptr, true};
    m.derived_ctx = ggml_init(ip);
    if (!m.derived_ctx)
        throw std::runtime_error("cannot allocate derived tensor metadata");
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> data;
    for (const auto& k : kernels) {
        ggml_tensor* w = weight(m, k.name);
        const int K = (int)w->ne[0], C = (int)ggml_nelements(w) / (K * k.gin);
        const auto values = tensor_f32(w); // PyTorch order [C, gin, K]
        const std::string name = mf2::tap_name(k.name);
        ggml_tensor* t = ggml_new_tensor_2d(m.derived_ctx, GGML_TYPE_F32, (int64_t)C * k.gin, K);
        ggml_set_name(t, name.c_str());
        m.derived[name] = t;
        data.emplace_back(t, mf2::tap_kernel(values.data(), C, k.gin, K));
    }
    m.derived_buf = ggml_backend_alloc_ctx_tensors(m.derived_ctx, m.backend);
    if (!m.derived_buf)
        throw std::runtime_error("cannot allocate derived kernels");
    for (auto& d : data)
        ggml_backend_tensor_set(d.first, d.second.data(), 0, d.second.size() * sizeof(float));
}

void begin_graph(mossformer2_sr_context& m, Graph& gr) {
    gr.clear();
    ggml_init_params ip{ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false), nullptr,
                        true};
    gr.ctx = ggml_init(ip);
    if (!gr.ctx)
        throw std::runtime_error("cannot allocate graph metadata");
    ggml_backend_t backends[] = {m.backend, m.cpu};
    gr.sched = ggml_backend_sched_new(backends, nullptr, m.backend == m.cpu ? 1 : 2, GRAPH_SIZE, false, true);
    if (!gr.sched)
        throw std::runtime_error("cannot create graph scheduler");
    gr.graph = ggml_new_graph_custom(gr.ctx, GRAPH_SIZE, false);
}

std::function<void(const std::string&, ggml_tensor*)> capture_fn(Graph& gr, const std::set<std::string>& wanted) {
    return [&gr, &wanted](const std::string& name, ggml_tensor* x) {
        if (!wanted.count(name))
            return;
        x = ggml_cont(gr.ctx, x);
        ggml_set_name(x, name.c_str());
        ggml_set_output(x);
        ggml_build_forward_expand(gr.graph, x);
        gr.captures[name] = x;
    };
}

void finish_graph(Graph& gr, int T, const std::set<std::string>& wanted) {
    ggml_set_output(gr.out);
    ggml_build_forward_expand(gr.graph, gr.out);
    if (!ggml_backend_sched_alloc_graph(gr.sched, gr.graph))
        throw std::runtime_error("graph allocation failed");
    gr.T = T;
    gr.wanted = wanted;
}

// MaskNet: [80, T] log-mel -> [80, T] generator input.
void build_masknet(mossformer2_sr_context& m, int T, const std::set<std::string>& wanted) {
    auto& gr = m.masknet;
    if (gr.T == T && gr.wanted == wanted)
        return;
    begin_graph(m, gr);
    const auto& h = m.hp;
    ggml_context* g = gr.ctx;
    gr.in = ggml_new_tensor_2d(g, GGML_TYPE_F32, h.num_mels, T);
    gr.pe = ggml_new_tensor_2d(g, GGML_TYPE_F32, h.blk.dim, T);
    gr.positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    for (ggml_tensor* t : {gr.in, gr.pe, gr.positions})
        ggml_set_input(t);

    mf2::Ctx c;
    c.g = g;
    c.weight = [&m](const std::string& name) { return weight(m, name); };
    c.derived = &m.derived;
    c.positions = gr.positions;
    c.portable_dwconv = m.portable_dwconv;
    c.capture = capture_fn(gr, wanted);

    // GroupNorm(1, 80) -> Conv1d(80, D, 1) -> + ScaledSinuEmbedding -> MossFormerM
    ggml_tensor* x = mf2::global_group_norm(c, gr.in, "mn.norm", 1e-8f);
    c.capture("mn_norm", x);
    x = ggml_add(g, mf2::linear(c, x, "mn.conv_enc", false), gr.pe);
    c.capture("mn_in", x);
    x = mf2::computation_block(c, x, h.blk, "mn.intra_ln_norm", "mn.intra_norm");
    c.capture("mdl_out", x);
    x = mf2::prelu(g, x, weight(m, "mn.prelu"));
    x = mf2::linear(c, x, "mn.conv_out"); // one speaker: the whole head
    ggml_tensor* o = ggml_tanh(g, mf2::linear(c, x, "mn.output"));
    ggml_tensor* gate = ggml_sigmoid(g, mf2::linear(c, x, "mn.output_gate"));
    gr.out = ggml_relu(g, mf2::linear(c, ggml_mul(g, o, gate), "mn.conv_dec", false)); // [80, T]
    c.capture("mf_out", gr.out);
    finish_graph(gr, T, wanted);
}

// Conv1d (stride 1, zero padding) on channel-major x [Cin, T] as a sum of
// per-tap matmuls: the GGUF stores the kernel as [Cin, K * Cout] (tap k is
// rows k * Cout .. (k + 1) * Cout), so no im2col buffer is needed and the
// matrices can be quantized.
ggml_tensor* gen_conv(mossformer2_sr_context& m, ggml_context* g, ggml_tensor* x, const std::string& prefix, int K,
                      int dil) {
    ggml_tensor* w = weight(m, prefix + ".weight");
    const int64_t cin = x->ne[0], T = x->ne[1], cout = w->ne[1] / K;
    const int pad = dil * (K - 1) / 2;
    ggml_tensor* xp = pad ? ggml_pad_ext(g, x, 0, 0, pad, pad, 0, 0, 0, 0) : x;
    ggml_tensor* y = nullptr;
    for (int k = 0; k < K; ++k) {
        ggml_tensor* wk = ggml_view_2d(g, w, cin, cout, w->nb[1], (size_t)k * cout * w->nb[1]);
        ggml_tensor* xk = ggml_view_2d(g, xp, cin, T, xp->nb[1], (size_t)k * dil * xp->nb[1]);
        ggml_tensor* p = ggml_mul_mat(g, wk, xk);
        y = y ? ggml_add(g, y, p) : p;
    }
    return ggml_add(g, y, mf2::as_f32(g, weight(m, prefix + ".bias")));
}

// ConvTranspose1d(stride u, kernel K, padding (K - u) / 2): matmul against the
// stored [Cin, Cout * K] matrix, then ggml_col2im_1d overlap-add (with the
// symmetric crop) -> [T * u, Cout] -> channel-major.
ggml_tensor* gen_convt(mossformer2_sr_context& m, ggml_context* g, ggml_tensor* x, int i) {
    const auto& h = m.hp;
    const int u = h.up_rates[(size_t)i], K = h.up_kernels[(size_t)i];
    const std::string p = "gen.ups." + std::to_string(i);
    ggml_tensor* w = weight(m, p + ".weight");
    const int cout = (int)(w->ne[1] / K);
    ggml_tensor* col = ggml_mul_mat(g, w, x);                      // [Cout * K, T]
    ggml_tensor* y = ggml_col2im_1d(g, col, u, cout, (K - u) / 2); // [T * u, Cout]
    y = ggml_cont(g, ggml_transpose(g, y));
    return ggml_add(g, y, mf2::as_f32(g, weight(m, p + ".bias")));
}

// Generator: [80, T] -> [256 T] waveform.
// sin^2(z) for the fused Snake op: Cody-Waite reduction by pi (sin^2 has
// period pi, so the quadrant sign drops out) and the degree-11 Taylor sine on
// [-pi/2, pi/2] (truncation error < 6e-8, i.e. float rounding).  Branch-free so
// the loop vectorises; |z| >= 1e5 (never seen) is redone with std::sin.
inline float sin_sq(float z) {
    // round-to-nearest via the 1.5 * 2^23 trick (|z / pi| < 2^22 here)
    const float k = (z * 0.31830988618379067f + 12582912.0f) - 12582912.0f;
    const float r = (z - k * 3.140625f) - k * 9.676535897932e-4f;
    const float r2 = r * r;
    const float p =
        r * (1.0f + r2 * (-1.6666667e-1f +
                          r2 * (8.3333333e-3f + r2 * (-1.9841270e-4f + r2 * (2.7557319e-6f + r2 * -2.5052108e-8f)))));
    return p * p;
}

// Snake (y = x + sin^2(alpha x) / (alpha + 1e-9), per channel) fused into one
// pass on the CPU: x [C, T] channel-major, b = alpha [C].
void snake_cpu_op(ggml_tensor* dst, const ggml_tensor* x, const ggml_tensor* alpha, int ith, int nth, void*) {
    const int64_t C = x->ne[0], T = ggml_nrows(x);
    const float* a = (const float*)alpha->data;
    std::vector<float> inv((size_t)C);
    for (int64_t c = 0; c < C; ++c)
        inv[(size_t)c] = 1.0f / (a[c] + 1e-9f);
    const int64_t per = (T + nth - 1) / nth, t0 = per * ith, t1 = std::min(T, t0 + per);
    for (int64_t t = t0; t < t1; ++t) {
        const float* xr = (const float*)((const char*)x->data + t * x->nb[1]);
        float* yr = (float*)((char*)dst->data + t * dst->nb[1]);
        for (int64_t c = 0; c < C; ++c)
            yr[c] = xr[c] + inv[(size_t)c] * sin_sq(a[c] * xr[c]);
        for (int64_t c = 0; c < C; ++c) {
            const float z = a[c] * xr[c];
            if (!(std::fabs(z) < 1e5f))
                yr[c] = xr[c] + inv[(size_t)c] * std::sin(z) * std::sin(z);
        }
    }
}

ggml_tensor* snake(mossformer2_sr_context& m, ggml_context* g, ggml_tensor* x, const std::string& alpha) {
    ggml_tensor* a = weight(m, alpha);
    if (m.backend != m.cpu || a->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x))
        return core_act::snake_alpha(g, x, a);
    return ggml_map_custom2(g, x, a, snake_cpu_op, GGML_N_TASKS_MAX, nullptr);
}

void build_generator(mossformer2_sr_context& m, int T, const std::set<std::string>& wanted) {
    auto& gr = m.generator;
    if (gr.T == T && gr.wanted == wanted)
        return;
    begin_graph(m, gr);
    const auto& h = m.hp;
    ggml_context* g = gr.ctx;
    gr.in = ggml_new_tensor_2d(g, GGML_TYPE_F32, h.num_mels, T);
    ggml_set_input(gr.in);
    auto capture = capture_fn(gr, wanted);

    ggml_tensor* x = gen_conv(m, g, gr.in, "gen.conv_pre", h.gen_kernel, 1);
    capture("g_pre", x);
    const int nk = n_resblocks(h);
    for (size_t i = 0; i < h.up_rates.size(); ++i) {
        const std::string s = std::to_string(i);
        x = snake(m, g, x, "gen.snakes." + s + ".alpha");
        x = gen_convt(m, g, x, (int)i);
        capture("g_ups" + s, x);
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < nk; ++j) { // ResBlock1
            const std::string r = rb_name((int)i, j, h);
            const int K = h.rb_kernels[(size_t)j];
            ggml_tensor* y = x;
            for (size_t d = 0; d < h.rb_dilations.size(); ++d) {
                const std::string ds = std::to_string(d);
                ggml_tensor* t = snake(m, g, y, r + "a1." + ds);
                t = gen_conv(m, g, t, r + "c1." + ds, K, h.rb_dilations[d]);
                t = snake(m, g, t, r + "a2." + ds);
                t = gen_conv(m, g, t, r + "c2." + ds, K, 1);
                y = ggml_add(g, t, y);
            }
            xs = xs ? ggml_add(g, xs, y) : y;
        }
        x = ggml_scale(g, xs, 1.0f / (float)nk);
        capture("g_mrf" + s, x);
    }
    x = snake(m, g, x, "gen.snake_post.alpha");
    x = gen_conv(m, g, x, "gen.conv_post", h.gen_kernel, 1); // [1, 256 T]
    gr.out = ggml_tanh(g, x);
    finish_graph(gr, T, wanted);
}

void read_captures(Graph& gr, Captures* caps) {
    if (caps)
        for (const auto& item : gr.captures)
            (*caps)[item.first] = tensor_f32(item.second);
}

// MaskNet forward on [T, 80] log-mel; returns [T, 80].
std::vector<float> run_masknet(mossformer2_sr_context& m, const std::vector<float>& mel, int T, Captures* caps,
                               const std::set<std::string>& wanted) {
    StageTimer timer("masknet");
    build_masknet(m, T, wanted);
    auto& gr = m.masknet;
    const auto pe = mf2::sinu_embedding(T, m.inv_freq, m.pos_scale);
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t)
        pos[t] = t;
    ggml_backend_tensor_set(gr.in, mel.data(), 0, mel.size() * sizeof(float));
    ggml_backend_tensor_set(gr.pe, pe.data(), 0, pe.size() * sizeof(float));
    ggml_backend_tensor_set(gr.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(gr.sched, gr.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph execution failed");
    read_captures(gr, caps);
    return tensor_f32(gr.out);
}

// Generator forward on [T, 80]; returns 256 T samples.  Chunked (see
// mossformer2_sr.h) unless captures are requested or chunk == 0.
std::vector<float> run_generator(mossformer2_sr_context& m, const std::vector<float>& feats, int T, Captures* caps,
                                 const std::set<std::string>& wanted) {
    StageTimer timer("generator");
    const int C = m.hp.num_mels;
    const bool gen_caps =
        std::any_of(wanted.begin(), wanted.end(), [](const std::string& s) { return s.rfind("g_", 0) == 0; });
    const bool one_pass = m.gen_chunk <= 0 || gen_caps || T <= m.gen_chunk + 2 * m.gen_margin;
    const int core = one_pass ? T : m.gen_chunk, margin = one_pass ? 0 : m.gen_margin;
    std::vector<float> out((size_t)T * HOP);
    for (int a = 0; a < T; a += core) {
        const int b = std::min(T, a + core);
        const int lo = std::max(0, a - margin), hi = std::min(T, b + margin);
        build_generator(m, hi - lo, gen_caps ? wanted : std::set<std::string>{});
        auto& gr = m.generator;
        ggml_backend_tensor_set(gr.in, feats.data() + (size_t)lo * C, 0, (size_t)(hi - lo) * C * sizeof(float));
        if (ggml_backend_sched_graph_compute(gr.sched, gr.graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph execution failed");
        if (gen_caps)
            read_captures(gr, caps);
        const auto y = tensor_f32(gr.out);
        std::copy(y.begin() + (size_t)(a - lo) * HOP, y.begin() + (size_t)(b - lo) * HOP,
                  out.begin() + (size_t)a * HOP);
    }
    return out;
}

// One decode window: mel -> MaskNet -> generator, 256 * T samples (T = n / 256).
std::vector<float> run_window(mossformer2_sr_context& m, const float* x, int n, Captures* caps = nullptr,
                              const std::set<std::string>& wanted = {}) {
    std::vector<float> mel;
    int T = 0;
    {
        StageTimer timer("mel");
        T = mf2sr::log_mel(x, n, mel, m.n_threads);
    }
    if (T <= 0)
        throw std::runtime_error("no analysis frames");
    if (caps)
        (*caps)["mel"] = mel;
    const auto mf = run_masknet(m, mel, T, caps, wanted);
    auto wave = run_generator(m, mf, T, caps, wanted);
    if (caps)
        (*caps)["gen_wave"] = wave;
    return wave;
}

std::vector<double> bandwidth_sub(const mossformer2_sr_context& m, const std::vector<double>& low,
                                  const std::vector<double>& high) {
    StageTimer timer("bandwidth_sub");
    double f_high = 0;
    auto out = mf2sr::bandwidth_sub(low, high, m.hp.sample_rate, m.hp.bw_threshold, m.hp.crossfade_ms, &f_high);
    if (m.verbose)
        std::fprintf(stderr, "mossformer2_sr: input bandwidth %.1f Hz\n", f_high);
    return out;
}

std::vector<float> decode(mossformer2_sr_context& m, const float* in, int n) {
    const auto& h = m.hp;
    if (n < WIN) { // too short for a frame: zero-extend, then trim
        std::vector<float> ext(WIN, 0.0f);
        std::copy_n(in, n, ext.begin());
        auto y = decode(m, ext.data(), WIN);
        y.resize((size_t)n);
        return y;
    }
    std::vector<double> low, high;
    if ((double)n <= (double)h.sample_rate * h.one_time_decode_length) {
        const auto wave = run_window(m, in, n);
        low.assign(in, in + n);
        high.assign(wave.begin(), wave.end());
    } else {
        const int window = (int)(h.sample_rate * h.decode_window);
        const int stride = (int)(window * h.decode_stride_ratio);
        const auto plan = mf2se::segment_plan(n, window, stride);
        std::vector<float> padded(in, in + n);
        padded.resize((size_t)plan.padded, 0.0f);
        high.assign((size_t)plan.padded, 0.0);
        const int g = plan.give_up;
        for (int64_t idx : plan.starts) {
            const int64_t lo = idx == 0 ? 0 : g, hi = window - g;
            if (idx + lo >= n)
                break; // only end padding left; upstream truncates these samples anyway
            const auto seg = run_window(m, padded.data() + idx, window);
            if ((int64_t)seg.size() < hi)
                throw std::runtime_error("decode window shorter than its kept span");
            for (int64_t i = lo; i < hi; ++i)
                high[(size_t)(idx + i)] = seg[(size_t)i];
        }
        low.assign(padded.begin(), padded.end());
    }
    const auto sub = bandwidth_sub(m, low, high);
    std::vector<float> out((size_t)n, 0.0f); // one-pass: upstream returns 256 * (n / 256) samples
    for (size_t i = 0; i < std::min(sub.size(), out.size()); ++i)
        out[i] = (float)sub[i];
    return out;
}
} // namespace

mossformer2_sr_context* mossformer2_sr_init(const char* path, int n_threads, bool verbose) {
    if (!path || !*path)
        return nullptr;
    std::unique_ptr<mossformer2_sr_context, decltype(&mossformer2_sr_free)> m(new mossformer2_sr_context,
                                                                              mossformer2_sr_free);
    try {
        StageTimer timer("load");
        gguf_context* meta = core_gguf::open_metadata(path);
        if (!meta)
            return nullptr;
        const bool valid = read_hparams(meta, m->hp);
        core_gguf::free_metadata(meta);
        if (!valid)
            throw std::runtime_error("unsupported architecture or incomplete MossFormer2_SR_48K metadata");
        m->n_threads = n_threads > 0 ? n_threads : std::min(8, core_parallel::resolve_threads(8, 0));
        m->verbose = verbose;
        m->gen_margin = generator_margin(m->hp);
        if (const char* c = std::getenv("CRISPASR_MF2SR_GEN_CHUNK"))
            m->gen_chunk = std::max(0, std::atoi(c));
        m->cpu = core_cpu_backend::init();
        if (!m->cpu)
            throw std::runtime_error("CPU backend unavailable");
        core_cpu_backend::set_n_threads(m->cpu, m->n_threads);
        if (env_flag("CRISPASR_MOSSFORMER2_SR_GPU")) {
            ggml_backend_load_all();
            m->backend = crispasr_init_gpu_backend();
            if (m->backend && core_cpu_backend::is_cpu(m->backend)) {
                ggml_backend_free(m->backend);
                m->backend = nullptr;
            }
        }
        if (!m->backend)
            m->backend = m->cpu;
        m->portable_dwconv = m->backend != m->cpu || env_flag("CRISPASR_MF2_PORTABLE_DWCONV");
        const bool loaded = m->backend == m->cpu
                                ? core_gguf::load_weights(path, m->cpu, "mossformer2_sr", m->weights)
                                : core_gguf::load_weights_fit(path, m->backend, m->cpu, "mossformer2_sr", m->weights);
        if (!loaded)
            throw std::runtime_error("weight load failed");
        validate_weights(*m);
        m->inv_freq = tensor_f32(weight(*m, "mn.pos_enc.inv_freq"));
        m->pos_scale = tensor_f32(weight(*m, "mn.pos_enc.scale"))[0];
        prepare_derived(*m);
        if (verbose)
            std::fprintf(stderr, "mossformer2_sr: backend=%s threads=%d layers=%d gen_chunk=%d margin=%d%s%s\n",
                         ggml_backend_name(m->backend), m->n_threads, m->hp.blk.n_layers, m->gen_chunk, m->gen_margin,
                         m->quantized ? " quantized" : "", m->portable_dwconv ? " dwconv=split" : "");
        return m.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_sr: %s\n", e.what());
        return nullptr;
    }
}

void mossformer2_sr_free(mossformer2_sr_context* ctx) {
    if (!ctx)
        return;
    ctx->masknet.clear();
    ctx->generator.clear();
    if (ctx->derived_buf)
        ggml_backend_buffer_free(ctx->derived_buf);
    if (ctx->derived_ctx)
        ggml_free(ctx->derived_ctx);
    core_gguf::free_weights(ctx->weights);
    if (ctx->backend && ctx->backend != ctx->cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->cpu)
        ggml_backend_free(ctx->cpu);
    delete ctx;
}

int mossformer2_sr_sample_rate(const mossformer2_sr_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}

const char* mossformer2_sr_backend_name(const mossformer2_sr_context* ctx) {
    return ctx && ctx->backend ? ggml_backend_name(ctx->backend) : nullptr;
}

bool mossformer2_sr_process(mossformer2_sr_context* ctx, const float* in, int n, std::vector<float>& out) {
    if (!ctx || !in || n <= 0)
        return false;
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(in[i]))
            return false;
    try {
        StageTimer timer("process");
        out = decode(*ctx, in, n);
        if ((int)out.size() != n || !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("nonfinite output waveform");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_sr: super-resolution failed: %s\n", e.what());
        return false;
    }
}

int mossformer2_sr_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // the reference carries the exact input waveform
    int threads = 0;
    if (const char* t = std::getenv("CRISPASR_MF2SR_THREADS"))
        threads = std::max(1, std::atoi(t));
    std::unique_ptr<mossformer2_sr_context, decltype(&mossformer2_sr_free)> m(
        mossformer2_sr_init(model_gguf, threads, verbosity > 0), mossformer2_sr_free);
    if (!m)
        return 2;
    core_gguf::WeightLoad ref;
    if (!core_gguf::load_weights(ref_gguf, m->cpu, "mossformer2_sr_reference", ref))
        return 2;
    int failures = 0, compared = 0;
    try {
        const auto it = ref.tensors.find("input_audio");
        if (it == ref.tensors.end() || ggml_n_dims(it->second) != 1)
            throw std::runtime_error("reference needs a 1-D input_audio");
        const auto audio = tensor_f32(it->second);
        if (!std::all_of(audio.begin(), audio.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("nonfinite reference waveform");
        const int n = (int)audio.size();
        std::set<std::string> wanted;
        bool any_forward = false, any_e2e = false;
        for (const auto& kv : ref.tensors) {
            if (kv.first == "input_audio")
                continue;
            if (kv.first.rfind("e2e_", 0) == 0)
                any_e2e = true;
            else {
                any_forward = true;
                if (kv.first != "mel" && kv.first != "gen_wave")
                    wanted.insert(kv.first); // graph captures (mel / gen_wave are host-side)
            }
        }
        Captures actual;
        if (any_forward) {
            // Single forward over the whole clip with the native front end (the
            // reference stages come from one forward too).
            StageTimer timer("diff_forward");
            run_window(*m, audio.data(), n, &actual, wanted);
        }
        if (any_e2e) {
            StageTimer timer("diff_e2e");
            actual["e2e_wave"] = decode(*m, audio.data(), n);
        }
        const double threshold = m->quantized ? 0.99 : 0.999;
        for (const auto& kv : ref.tensors) {
            if (kv.first == "input_audio")
                continue;
            const auto a = actual.find(kv.first);
            if (a == actual.end() || (int64_t)a->second.size() != ggml_nelements(kv.second)) {
                std::printf("[FAIL] %s missing native stage or shape mismatch\n", kv.first.c_str());
                ++failures;
                continue;
            }
            ++compared;
            const auto truth = tensor_f32(kv.second);
            const auto& data = a->second;
            const size_t width = ggml_n_dims(kv.second) == 1 ? data.size() : (size_t)kv.second->ne[0];
            double dot = 0, aa = 0, bb = 0, max_abs = 0, cos_min = 1;
            size_t nonfinite = 0;
            for (size_t row = 0; row < data.size(); row += width) {
                double rd = 0, ra = 0, rb = 0;
                for (size_t j = row; j < row + width; ++j) {
                    const double x = data[j], y = truth[j];
                    if (!std::isfinite(x) || !std::isfinite(y)) {
                        ++nonfinite;
                        continue;
                    }
                    rd += x * y;
                    ra += x * x;
                    rb += y * y;
                    max_abs = std::max(max_abs, std::abs(x - y));
                }
                dot += rd;
                aa += ra;
                bb += rb;
                const double rc = ra > 0 && rb > 0 ? rd / std::sqrt(ra * rb) : ra == rb ? 1 : 0;
                cos_min = std::min(cos_min, rc);
            }
            const double cosine = aa > 0 && bb > 0 ? dot / std::sqrt(aa * bb) : aa == bb ? 1 : 0;
            const double ratio = bb > 0 ? std::sqrt(aa / bb) : aa == 0 ? 1 : std::numeric_limits<double>::infinity();
            const bool pass = nonfinite == 0 && cosine > threshold && ratio >= 0.9 && ratio <= 1.1;
            if (!pass)
                ++failures;
            if (verbosity > 0 || !pass)
                std::printf("[%s] %-14s cos=%.9f cos_min_row=%.9f norm_ratio=%.6f max_abs=%.4g n=%zu\n",
                            pass ? "PASS" : "FAIL", kv.first.c_str(), cosine, cos_min, ratio, max_abs, data.size());
        }
        std::printf("mossformer2_sr diff: %d stages compared, %d failed, cosine > %.3f (%s)\n", compared, failures,
                    threshold, m->quantized ? "quantized" : "F32/F16");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_sr diff: %s\n", e.what());
        failures = 1;
    }
    core_gguf::free_weights(ref);
    return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// crispasr_enhancer registration
// ---------------------------------------------------------------------------

static void* mf2sr_ops_init(const char* path, int n_threads, bool verbose) {
    return mossformer2_sr_init(path, n_threads, verbose);
}
static int mf2sr_ops_rate(void* ctx) {
    return mossformer2_sr_sample_rate((mossformer2_sr_context*)ctx);
}
static bool mf2sr_ops_process(void* ctx, const float* in, int n, std::vector<float>& out) {
    return mossformer2_sr_process((mossformer2_sr_context*)ctx, in, n, out);
}
static void mf2sr_ops_free(void* ctx) {
    mossformer2_sr_free((mossformer2_sr_context*)ctx);
}

const CrispasrEnhanceModelOps* mossformer2_sr_enhance_ops() {
    static const CrispasrEnhanceModelOps ops = {"mossformer2-sr", "mossformer2-sr",  mf2sr_ops_init,
                                                mf2sr_ops_rate,   mf2sr_ops_process, mf2sr_ops_free};
    return &ops;
}
