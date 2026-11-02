// Native ggml port of ClearerVoice-Studio MossFormer2_SE_48K (see
// mossformer2_se.h).  Network: models/mossformer2_se/mossformer2.py
// MossFormer_MaskNet; the MossFormerM stack is shared with MossFormer2_SS_16K
// (mossformer2_blocks.{h,cpp}).  Front/back end: utils/decode.py
// decode_one_audio_mossformer2_se_48k and utils/misc.py compute_fbank / stft /
// istft.
#include "mossformer2_se.h"
#include "mossformer2_blocks.h"

#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/kaldi_fbank.h"
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
// Fixed by MossFormer2_SE_48K.yaml / compute_fbank; read_hparams() rejects GGUFs
// that disagree, so the model-free helpers can use them directly.
constexpr int SR = 48000, N_MELS = 60, IN_CH = 180, WIN = 1920, HOP = 384, NFFT = 1920, BINS = 961;
constexpr float WAV_SCALE = 32768.0f;

bool env_flag(const char* key, bool fallback = false) {
    const char* v = std::getenv(key);
    return v && *v ? std::strcmp(v, "0") != 0 : fallback;
}

struct StageTimer {
    const char* name;
    bool enabled = env_flag("CRISPASR_MF2SE_BENCH");
    std::chrono::steady_clock::time_point start;
    explicit StageTimer(const char* n) : name(n) {
        if (enabled)
            start = std::chrono::steady_clock::now();
    }
    ~StageTimer() {
        if (enabled)
            std::fprintf(stderr, "mossformer2_se_bench: %s %.3f ms\n", name,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
};

struct HParams {
    int sample_rate = 0, num_mels = 0, in_channels = 0, win_len = 0, win_inc = 0, fft_len = 0, mask_bins = 0;
    int num_spks = 0, delta_win = 0;
    mf2::BlockHParams blk;
    float one_time_decode_length = 0, decode_window = 0, decode_stride_ratio = 0, wav_scale = 0;
    float preemph = 0, low_freq = 0, high_freq = -1, dither = 0;
    std::string window;
};

bool read_hparams(gguf_context* meta, HParams& h) {
    if (core_gguf::kv_str(meta, "general.architecture", "") != "mossformer2-se")
        return false;
    auto u = [&](const char* key) {
        return (int)core_gguf::kv_u32(meta, (std::string("mossformer2_se.") + key).c_str(), 0);
    };
    auto f = [&](const char* key, float fallback = 0) {
        return core_gguf::kv_f32(meta, (std::string("mossformer2_se.") + key).c_str(), fallback);
    };
    h.sample_rate = u("sample_rate");
    h.num_mels = u("num_mels");
    h.in_channels = u("in_channels");
    h.win_len = u("win_len");
    h.win_inc = u("win_inc");
    h.fft_len = u("fft_len");
    h.mask_bins = u("mask_bins");
    h.num_spks = u("num_spks");
    h.delta_win = u("delta_win_length");
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
    h.one_time_decode_length = f("one_time_decode_length");
    h.decode_window = f("decode_window");
    h.decode_stride_ratio = f("decode_stride_ratio");
    h.wav_scale = f("wav_scale");
    h.preemph = f("fbank_preemphasis");
    h.low_freq = f("fbank_low_freq");
    h.high_freq = f("fbank_high_freq", -1);
    h.dither = f("fbank_dither", 1.0f);
    h.window = core_gguf::kv_str(meta, "mossformer2_se.window", "");
    // Versioned architecture: reject metadata this runtime was not validated on
    // before a ggml shape assertion can abort the host process.
    return h.sample_rate == SR && h.num_mels == N_MELS && h.in_channels == IN_CH && h.win_len == WIN &&
           h.win_inc == HOP && h.fft_len == NFFT && h.mask_bins == BINS && h.num_spks == 2 && h.delta_win == 5 &&
           h.blk.dim == 512 && h.blk.n_layers >= 1 && h.blk.n_layers <= 64 && h.blk.group == 256 && h.blk.qk == 128 &&
           h.blk.hidden == 2048 && h.blk.fsmn_inner == 256 && h.blk.fsmn_lorder == 20 && h.blk.conv_kernel == 17 &&
           h.blk.rotary_dim == 32 && h.blk.rotary_theta == 10000.0f && h.one_time_decode_length > 0 &&
           h.decode_window > 0 && h.decode_stride_ratio > 0 && h.decode_stride_ratio <= 1 && h.wav_scale == WAV_SCALE &&
           h.preemph == 0.97f && h.low_freq == 20.0f && h.high_freq == 0.0f && h.window == "hamming" &&
           h.dither >= 0.0f;
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

using cpx = std::complex<float>;

// Mixed-radix complex FFT for N = 2^a * m (m odd, direct DFT at the leaves)
// with one precomputed twiddle table; N = 1920 = 2^7 * 15.
struct Fft {
    int N = 0;
    std::vector<cpx> w; // exp(-2 pi i k / N)
    explicit Fft(int n) : N(n), w((size_t)n) {
        for (int k = 0; k < n; ++k) {
            const double a = -2.0 * M_PI * (double)k / (double)n;
            w[(size_t)k] = cpx((float)std::cos(a), (float)std::sin(a));
        }
    }
    void rec(const cpx* in, int stride, cpx* out, int n, int step) const {
        if (n % 2) {
            for (int k = 0; k < n; ++k) {
                cpx acc(0.0f, 0.0f);
                for (int j = 0; j < n; ++j)
                    acc += in[(size_t)j * stride] * w[(size_t)((j * k) % n) * step];
                out[k] = acc;
            }
            return;
        }
        const int h = n / 2;
        rec(in, 2 * stride, out, h, 2 * step);
        rec(in + stride, 2 * stride, out + h, h, 2 * step);
        for (int k = 0; k < h; ++k) {
            const cpx e = out[k], t = w[(size_t)k * step] * out[k + h];
            out[k] = e + t;
            out[k + h] = e - t;
        }
    }
    void forward(const cpx* in, cpx* out) const { rec(in, 1, out, N, 1); }
};

const Fft& fft1920() {
    static const Fft f(NFFT);
    return f;
}

// torch.hamming_window(N, periodic=False)
const std::vector<float>& hamming1920() {
    static const std::vector<float> w = [] {
        std::vector<float> v(WIN);
        for (int i = 0; i < WIN; ++i)
            v[(size_t)i] = (float)(0.54 - 0.46 * std::cos(2.0 * M_PI * (double)i / (double)(WIN - 1)));
        return v;
    }();
    return w;
}

using Captures = std::map<std::string, std::vector<float>>;

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *feats = nullptr, *pe = nullptr, *positions = nullptr, *out = nullptr;
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
        feats = pe = positions = out = nullptr;
        captures.clear();
        T = 0;
        wanted.clear();
    }
};
} // namespace

struct mossformer2_se_context {
    HParams hp;
    int n_threads = 4;
    bool verbose = false;
    float dither = 1.0f; // Kaldi fbank dither at int16 scale (upstream 1.0)
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    ggml_context* derived_ctx = nullptr;
    ggml_backend_buffer_t derived_buf = nullptr;
    std::map<std::string, ggml_tensor*> derived;
    std::vector<float> inv_freq;
    float pos_scale = 1.0f;
    bool quantized = false;
    bool portable_dwconv = false;
    Graph graph;
};

// ---------------------------------------------------------------------------
// Model-free front / back end
// ---------------------------------------------------------------------------
namespace mf2se {

std::vector<float> compute_deltas(const std::vector<float>& x, int T, int C, int win_length) {
    // torchaudio.functional.compute_deltas: replicate-pad n = (win - 1) / 2 frames,
    // out[t] = sum_{j=-n..n} j * x[t + j] / (n (n + 1) (2n + 1) / 3).
    const int n = (win_length - 1) / 2;
    const float denom = (float)(n * (n + 1) * (2 * n + 1)) / 3.0f;
    std::vector<float> out((size_t)T * C, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) {
            float acc = 0.0f;
            for (int j = -n; j <= n; ++j) {
                const int s = std::min(std::max(t + j, 0), T - 1);
                acc += (float)j * x[(size_t)s * C + c];
            }
            out[(size_t)t * C + c] = acc / denom;
        }
    return out;
}

int features(const float* in, int n, std::vector<float>& fbank, std::vector<float>& feats, float dither,
             uint64_t seed) {
    core_kaldi::FbankParams p;
    p.sample_rate = SR;
    p.n_mels = N_MELS;
    p.frame_length_ms = WIN * 1000 / SR; // 40
    p.frame_shift_ms = HOP * 1000 / SR;  // 8
    p.low_freq = 20.0f;
    p.high_freq = 0.0f;
    p.preemph = 0.97f;
    p.remove_dc_offset = true;
    p.int16_scale = true; // decode.py: inputs * MAX_WAV_VALUE before compute_fbank
    p.window_type = core_kaldi::WindowType::Hamming;
    p.snip_edges = true;
    p.mel_domain_triangles = true;
    p.dither = dither;
    p.dither_seed = seed;
    int T = 0;
    fbank = core_kaldi::compute_fbank(in, n, p, T);
    feats.clear();
    if (T <= 0)
        return 0;
    const auto d1 = compute_deltas(fbank, T, N_MELS);
    const auto d2 = compute_deltas(d1, T, N_MELS);
    feats.resize((size_t)T * IN_CH);
    for (int t = 0; t < T; ++t) {
        float* row = feats.data() + (size_t)t * IN_CH;
        std::copy_n(fbank.data() + (size_t)t * N_MELS, N_MELS, row);
        std::copy_n(d1.data() + (size_t)t * N_MELS, N_MELS, row + N_MELS);
        std::copy_n(d2.data() + (size_t)t * N_MELS, N_MELS, row + 2 * N_MELS);
    }
    return T;
}

std::vector<float> apply_mask(const float* x, int n, const float* mask, int n_frames, int n_threads) {
    const auto& win = hamming1920();
    const Fft& fft = fft1920();
    const int T = n_frames;
    const int64_t cover = T > 0 ? (int64_t)(T - 1) * HOP + WIN : 0;
    if (T <= 0 || cover > n)
        throw std::runtime_error("mossformer2_se: frame count does not fit the signal");
    // Per frame: windowed FFT, mask, Hermitian iFFT, window again (torch.istft
    // multiplies by the synthesis window before the overlap-add).
    std::vector<float> frames((size_t)T * WIN);
    core_parallel::for_each_chunk(T, n_threads, [&](int a, int b) {
        std::vector<cpx> buf(NFFT), spec(NFFT);
        for (int t = a; t < b; ++t) {
            const float* src = x + (size_t)t * HOP;
            for (int i = 0; i < NFFT; ++i)
                buf[(size_t)i] = cpx(src[i] * win[(size_t)i], 0.0f);
            fft.forward(buf.data(), spec.data());
            const float* m = mask + (size_t)t * BINS;
            // Inverse via conj(FFT(conj(Y))) / N over the Hermitian extension;
            // the imaginary parts of DC and Nyquist are dropped (C2R semantics).
            for (int k = 0; k < BINS; ++k) {
                cpx y = spec[(size_t)k] * m[k];
                if (k == 0 || k == BINS - 1)
                    y = cpx(y.real(), 0.0f);
                buf[(size_t)k] = std::conj(y);
                if (k > 0 && k < BINS - 1)
                    buf[(size_t)(NFFT - k)] = y;
            }
            fft.forward(buf.data(), spec.data());
            float* dst = frames.data() + (size_t)t * WIN;
            for (int i = 0; i < WIN; ++i)
                dst[i] = spec[(size_t)i].real() / (float)NFFT * win[(size_t)i];
        }
    });
    std::vector<double> acc((size_t)cover, 0.0), env((size_t)cover, 0.0);
    for (int t = 0; t < T; ++t) {
        const float* f = frames.data() + (size_t)t * WIN;
        double* a = acc.data() + (size_t)t * HOP;
        double* e = env.data() + (size_t)t * HOP;
        for (int i = 0; i < WIN; ++i) {
            a[i] += f[i];
            e[i] += (double)win[(size_t)i] * win[(size_t)i];
        }
    }
    std::vector<float> y((size_t)n, 0.0f); // torch.istft zero-fills past the last frame
    for (int64_t i = 0; i < cover; ++i)
        y[(size_t)i] = env[(size_t)i] > 1e-11 ? (float)(acc[(size_t)i] / env[(size_t)i]) : 0.0f;
    return y;
}

SegmentPlan segment_plan(int64_t n, int window, int stride) {
    SegmentPlan p;
    p.window = window;
    p.stride = stride;
    p.give_up = (window - stride) / 2;
    int64_t t = n;
    // decode_one_audio_mossformer2_se_48k padding, verbatim (including the
    // third branch, which appends window + remainder samples).
    if (t < window)
        t = window;
    else if (t < window + stride)
        t = window + stride;
    else if ((t - window) % stride != 0)
        t += t - (t - window) / stride * stride;
    p.padded = t;
    for (int64_t idx = 0; idx + window <= t; idx += stride)
        p.starts.push_back(idx);
    return p;
}

} // namespace mf2se

namespace {
ggml_tensor* weight(mossformer2_se_context& m, const std::string& n) {
    const auto it = m.weights.tensors.find(n);
    if (it == m.weights.tensors.end())
        throw std::runtime_error("missing tensor: " + n);
    return it->second;
}

void expect(mossformer2_se_context& m, const std::string& n, int64_t d0, int64_t d1 = 1, int64_t d2 = 1) {
    const ggml_tensor* w = weight(m, n);
    if (w->ne[0] != d0 || w->ne[1] != d1 || w->ne[2] != d2 || w->ne[3] != 1)
        throw std::runtime_error("incompatible tensor shape: " + n);
}

void validate_weights(mossformer2_se_context& m) {
    const auto& h = m.hp;
    const auto& b = h.blk;
    const int k = b.conv_kernel, lo = 2 * b.fsmn_lorder - 1;
    expect(m, "mn.norm.weight", h.in_channels);
    expect(m, "mn.norm.bias", h.in_channels);
    expect(m, "mn.conv_enc.weight", h.in_channels, b.dim);
    expect(m, "mn.pos_enc.inv_freq", b.dim / 2);
    expect(m, "mn.pos_enc.scale", 1);
    expect(m, "mn.prelu", 1);
    expect(m, "mn.conv_out.weight", b.dim, b.dim * h.num_spks);
    expect(m, "mn.conv_out.bias", b.dim * h.num_spks);
    expect(m, "mn.output.weight", b.dim, b.dim);
    expect(m, "mn.output_gate.weight", b.dim, b.dim);
    expect(m, "mn.conv_dec.weight", b.dim, h.mask_bins);
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
    for (const auto& kv : m.weights.tensors)
        if (ggml_is_quantized(kv.second->type))
            m.quantized = true;
}

// Load-time F32 depthwise kernels on the compute backend: tap-major copies for
// the CPU custom op (the portable path reads the k=17/39 kernels directly).
void prepare_derived(mossformer2_se_context& m) {
    if (m.portable_dwconv)
        return; // no grouped / dilated kernels in this model
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

void build_graph(mossformer2_se_context& m, int T, const std::set<std::string>& wanted) {
    auto& gr = m.graph;
    if (gr.T == T && gr.wanted == wanted)
        return;
    gr.clear();
    const auto& h = m.hp;
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
    ggml_context* g = gr.ctx;

    gr.feats = ggml_new_tensor_2d(g, GGML_TYPE_F32, h.in_channels, T);
    gr.pe = ggml_new_tensor_2d(g, GGML_TYPE_F32, h.blk.dim, T);
    gr.positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    for (ggml_tensor* t : {gr.feats, gr.pe, gr.positions})
        ggml_set_input(t);

    mf2::Ctx c;
    c.g = g;
    c.weight = [&m](const std::string& name) { return weight(m, name); };
    c.derived = &m.derived;
    c.positions = gr.positions;
    c.portable_dwconv = m.portable_dwconv;
    c.capture = [&gr, &wanted](const std::string& name, ggml_tensor* x) {
        if (!wanted.count(name))
            return;
        x = ggml_cont(gr.ctx, x);
        ggml_set_name(x, name.c_str());
        ggml_set_output(x);
        ggml_build_forward_expand(gr.graph, x);
        gr.captures[name] = x;
    };

    // MaskNet: GroupNorm(1, 180) -> Conv1d(180, D, 1) -> + ScaledSinuEmbedding
    ggml_tensor* x = mf2::global_group_norm(c, gr.feats, "mn.norm", 1e-8f);
    c.capture("mn_norm", x);
    x = ggml_add(g, mf2::linear(c, x, "mn.conv_enc", false), gr.pe);
    c.capture("mn_in", x);
    x = mf2::computation_block(c, x, h.blk, "mn.intra_ln_norm", "mn.intra_norm");
    c.capture("mdl_out", x);
    x = mf2::prelu(g, x, weight(m, "mn.prelu"));
    // conv1d_out -> view(B * spks, D, S); the decode path keeps speaker 0 only,
    // i.e. output channels [0, D) of conv1d_out.
    const int D = h.blk.dim;
    ggml_tensor* w_out = weight(m, "mn.conv_out.weight");
    ggml_tensor* b_out = weight(m, "mn.conv_out.bias");
    ggml_tensor* w0 = ggml_view_2d(g, w_out, D, D, w_out->nb[1], 0);
    ggml_tensor* b0 = mf2::as_f32(g, ggml_view_1d(g, b_out, D, 0));
    x = ggml_add(g, ggml_mul_mat(g, w0, x), b0);
    ggml_tensor* o = ggml_tanh(g, mf2::linear(c, x, "mn.output"));
    ggml_tensor* gate = ggml_sigmoid(g, mf2::linear(c, x, "mn.output_gate"));
    gr.out = ggml_relu(g, mf2::linear(c, ggml_mul(g, o, gate), "mn.conv_dec", false)); // [961, T]
    c.capture("mask", gr.out);
    ggml_set_output(gr.out);
    ggml_build_forward_expand(gr.graph, gr.out);
    if (!ggml_backend_sched_alloc_graph(gr.sched, gr.graph))
        throw std::runtime_error("graph allocation failed");
    gr.T = T;
    gr.wanted = wanted;
}

// Network forward on [T, 180] features; returns the mask [T, 961].
std::vector<float> forward(mossformer2_se_context& m, const std::vector<float>& feats, int T, Captures* caps = nullptr,
                           const std::set<std::string>& wanted = {}) {
    StageTimer timer("forward");
    build_graph(m, T, wanted);
    auto& gr = m.graph;
    const auto pe = mf2::sinu_embedding(T, m.inv_freq, m.pos_scale);
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t)
        pos[t] = t;
    ggml_backend_tensor_set(gr.feats, feats.data(), 0, feats.size() * sizeof(float));
    ggml_backend_tensor_set(gr.pe, pe.data(), 0, pe.size() * sizeof(float));
    ggml_backend_tensor_set(gr.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(gr.sched, gr.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph execution failed");
    if (caps)
        for (const auto& item : gr.captures)
            (*caps)[item.first] = tensor_f32(item.second);
    return tensor_f32(gr.out);
}

// One decode window (the body of decode_one_audio_mossformer2_se_48k's loop or
// its one-pass branch) on exactly `n` samples.
// `seed` selects the dither stream (the window's start sample).
std::vector<float> run_window(mossformer2_se_context& m, const float* x, int n, uint64_t seed = 0,
                              Captures* caps = nullptr, const std::set<std::string>& wanted = {}) {
    if (n < WIN) { // too short for one analysis frame: zero-extend, then trim
        std::vector<float> ext(WIN, 0.0f);
        std::copy_n(x, n, ext.begin());
        auto y = run_window(m, ext.data(), WIN, seed, caps, wanted);
        y.resize((size_t)n);
        return y;
    }
    std::vector<float> fbank, feats;
    int T = 0;
    {
        StageTimer timer("features");
        T = mf2se::features(x, n, fbank, feats, m.dither, seed);
    }
    if (T <= 0)
        throw std::runtime_error("no analysis frames");
    if (caps) {
        (*caps)["fbank"] = fbank;
        (*caps)["feats"] = feats;
    }
    const auto mask = forward(m, feats, T, caps, wanted);
    StageTimer timer("istft");
    return mf2se::apply_mask(x, n, mask.data(), T, m.n_threads);
}

std::vector<float> decode(mossformer2_se_context& m, const float* in, int n) {
    const auto& h = m.hp;
    if ((double)n <= (double)h.sample_rate * h.one_time_decode_length)
        return run_window(m, in, n);
    const int window = (int)(h.sample_rate * h.decode_window);
    const int stride = (int)(window * h.decode_stride_ratio);
    const auto plan = mf2se::segment_plan(n, window, stride);
    std::vector<float> padded(in, in + n);
    padded.resize((size_t)plan.padded, 0.0f);
    std::vector<float> out((size_t)plan.padded, 0.0f);
    const int g = plan.give_up;
    for (int64_t idx : plan.starts) {
        const int64_t lo = idx == 0 ? 0 : g, hi = window - g;
        if (idx + lo >= n)
            break; // only end padding left; upstream truncates these samples anyway
        const auto seg = run_window(m, padded.data() + idx, window, (uint64_t)idx);
        std::copy(seg.begin() + lo, seg.begin() + hi, out.begin() + idx + lo);
    }
    out.resize((size_t)n);
    return out;
}
} // namespace

mossformer2_se_context* mossformer2_se_init(const char* path, int n_threads, bool verbose) {
    if (!path || !*path)
        return nullptr;
    std::unique_ptr<mossformer2_se_context, decltype(&mossformer2_se_free)> m(new mossformer2_se_context,
                                                                              mossformer2_se_free);
    try {
        StageTimer timer("load");
        gguf_context* meta = core_gguf::open_metadata(path);
        if (!meta)
            return nullptr;
        const bool valid = read_hparams(meta, m->hp);
        core_gguf::free_metadata(meta);
        if (!valid)
            throw std::runtime_error("unsupported architecture or incomplete MossFormer2_SE_48K metadata");
        m->n_threads = n_threads > 0 ? n_threads : std::min(8, core_parallel::resolve_threads(8, 0));
        m->verbose = verbose;
        m->dither = m->hp.dither;
        if (const char* d = std::getenv("CRISPASR_MF2SE_DITHER"))
            m->dither = std::max(0.0f, (float)std::atof(d));
        m->cpu = core_cpu_backend::init();
        if (!m->cpu)
            throw std::runtime_error("CPU backend unavailable");
        core_cpu_backend::set_n_threads(m->cpu, m->n_threads);
        if (env_flag("CRISPASR_MOSSFORMER2_SE_GPU")) {
            ggml_backend_load_all();
            m->backend = crispasr_init_gpu_backend();
            if (m->backend && core_cpu_backend::is_cpu(m->backend)) {
                ggml_backend_free(m->backend);
                m->backend = nullptr;
            }
        }
        if (!m->backend)
            m->backend = m->cpu;
        // ggml_ssm_conv on GPUs only implements a few kernel widths; split the
        // k=17/39 kernels there (CRISPASR_MF2_PORTABLE_DWCONV=1 forces it on CPU).
        m->portable_dwconv = m->backend != m->cpu || env_flag("CRISPASR_MF2_PORTABLE_DWCONV");
        const bool loaded = m->backend == m->cpu
                                ? core_gguf::load_weights(path, m->cpu, "mossformer2_se", m->weights)
                                : core_gguf::load_weights_fit(path, m->backend, m->cpu, "mossformer2_se", m->weights);
        if (!loaded)
            throw std::runtime_error("weight load failed");
        validate_weights(*m);
        m->inv_freq = tensor_f32(weight(*m, "mn.pos_enc.inv_freq"));
        m->pos_scale = tensor_f32(weight(*m, "mn.pos_enc.scale"))[0];
        prepare_derived(*m);
        if (verbose)
            std::fprintf(stderr, "mossformer2_se: backend=%s threads=%d layers=%d%s%s\n", ggml_backend_name(m->backend),
                         m->n_threads, m->hp.blk.n_layers, m->quantized ? " quantized" : "",
                         m->portable_dwconv ? " dwconv=split" : "");
        return m.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_se: %s\n", e.what());
        return nullptr;
    }
}

void mossformer2_se_free(mossformer2_se_context* ctx) {
    if (!ctx)
        return;
    ctx->graph.clear();
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

int mossformer2_se_sample_rate(const mossformer2_se_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}

const char* mossformer2_se_backend_name(const mossformer2_se_context* ctx) {
    return ctx && ctx->backend ? ggml_backend_name(ctx->backend) : nullptr;
}

bool mossformer2_se_enhance(mossformer2_se_context* ctx, const float* in, int n, std::vector<float>& out) {
    if (!ctx || !in || n <= 0)
        return false;
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(in[i]))
            return false;
    try {
        StageTimer timer("enhance");
        out = decode(*ctx, in, n);
        if ((int)out.size() != n || !std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("nonfinite enhanced waveform");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_se: enhancement failed: %s\n", e.what());
        return false;
    }
}

int mossformer2_se_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // the reference carries the exact input waveform
    int threads = 0;
    if (const char* t = std::getenv("CRISPASR_MF2SE_THREADS"))
        threads = std::max(1, std::atoi(t));
    std::unique_ptr<mossformer2_se_context, decltype(&mossformer2_se_free)> m(
        mossformer2_se_init(model_gguf, threads, verbosity > 0), mossformer2_se_free);
    if (!m)
        return 2;
    m->dither = 0.0f; // references are dumped with dither 0 (upstream's is unseeded)
    core_gguf::WeightLoad ref;
    if (!core_gguf::load_weights(ref_gguf, m->cpu, "mossformer2_se_reference", ref))
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
                wanted.insert(kv.first);
            }
        }
        Captures actual;
        if (any_forward) {
            // Single forward over the whole clip with the native front end
            // (the reference stages come from one forward too).
            StageTimer timer("diff_forward");
            run_window(*m, audio.data(), n, 0, &actual, wanted);
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
            const size_t width = (size_t)kv.second->ne[0];
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
        std::printf("mossformer2_se diff: %d stages compared, %d failed, cosine > %.3f (%s)\n", compared, failures,
                    threshold, m->quantized ? "quantized" : "F32/F16");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_se diff: %s\n", e.what());
        failures = 1;
    }
    core_gguf::free_weights(ref);
    return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// crispasr_enhancer registration
// ---------------------------------------------------------------------------

static void* mf2se_ops_init(const char* path, int n_threads, bool verbose) {
    return mossformer2_se_init(path, n_threads, verbose);
}
static int mf2se_ops_rate(void* ctx) {
    return mossformer2_se_sample_rate((mossformer2_se_context*)ctx);
}
static bool mf2se_ops_process(void* ctx, const float* in, int n, std::vector<float>& out) {
    return mossformer2_se_enhance((mossformer2_se_context*)ctx, in, n, out);
}
static void mf2se_ops_free(void* ctx) {
    mossformer2_se_free((mossformer2_se_context*)ctx);
}

const CrispasrEnhanceModelOps* mossformer2_se_enhance_ops() {
    static const CrispasrEnhanceModelOps ops = {"mossformer2-se", "mossformer2-se",  mf2se_ops_init,
                                                mf2se_ops_rate,   mf2se_ops_process, mf2se_ops_free};
    return &ops;
}
