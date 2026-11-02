// Native ggml port of ClearerVoice-Studio MossFormer2_SS_16K
// (models/mossformer2_ss/mossformer2.py: Encoder -> MossFormer_MaskNet ->
// Decoder) plus utils/decode.py decode_one_audio_mossformer2_ss_16k.
// The MossFormerM stack itself is shared: mossformer2_blocks.{h,cpp}.
#include "mossformer2_ss.h"
#include "mossformer2_blocks.h"

#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/parallel_for.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

namespace {
constexpr size_t GRAPH_SIZE = 32768;
const char* SOURCES[] = {"spk1", "spk2"};

bool env_flag(const char* key, bool fallback = false) {
    const char* v = std::getenv(key);
    return v && *v ? std::strcmp(v, "0") != 0 : fallback;
}

struct StageTimer {
    const char* name;
    bool enabled = env_flag("CRISPASR_MF2SS_BENCH");
    std::chrono::steady_clock::time_point start;
    explicit StageTimer(const char* n) : name(n) {
        if (enabled)
            start = std::chrono::steady_clock::now();
    }
    ~StageTimer() {
        if (enabled)
            std::fprintf(stderr, "mossformer2_ss_bench: %s %.3f ms\n", name,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
};

struct HParams {
    int sample_rate = 0, num_spks = 0, enc_kernel = 0, enc_dim = 0;
    mf2::BlockHParams blk;
    float one_time_decode_length = 0, decode_window = 0, decode_stride_ratio = 0;
};

bool read_hparams(gguf_context* meta, HParams& h) {
    if (core_gguf::kv_str(meta, "general.architecture", "") != "mossformer2-ss")
        return false;
    auto u = [&](const char* key) {
        return (int)core_gguf::kv_u32(meta, (std::string("mossformer2_ss.") + key).c_str(), 0);
    };
    auto f = [&](const char* key) {
        return core_gguf::kv_f32(meta, (std::string("mossformer2_ss.") + key).c_str(), 0);
    };
    h.sample_rate = u("sample_rate");
    h.num_spks = u("num_spks");
    h.enc_kernel = u("encoder_kernel_size");
    h.enc_dim = u("encoder_embedding_dim");
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
    h.one_time_decode_length = f("one_time_decode_length");
    h.decode_window = f("decode_window");
    h.decode_stride_ratio = f("decode_stride_ratio");
    // Versioned architecture: reject metadata this runtime was not validated on
    // before a ggml shape assertion can abort the host process.
    return h.sample_rate == 16000 && h.num_spks == 2 && h.enc_kernel == 16 && h.enc_dim == 512 && h.blk.dim == 512 &&
           h.blk.n_layers >= 1 && h.blk.n_layers <= 64 && h.blk.group == 256 && h.blk.qk == 128 &&
           h.blk.hidden == 2048 && h.blk.fsmn_inner == 256 && h.blk.fsmn_lorder == 20 && h.blk.conv_kernel == 17 &&
           h.blk.rotary_dim == 32 && h.blk.rotary_theta == 10000.0f && h.one_time_decode_length > 0 &&
           h.decode_window > 0 && h.decode_stride_ratio > 0 && h.decode_stride_ratio <= 1;
}

using Captures = std::map<std::string, std::vector<float>>;

// CRISPASR_MF2SS_PROFILE=1: per-op-type wall time (node-by-node execution).
struct OpProfile {
    std::chrono::steady_clock::time_point last;
    std::map<std::string, std::pair<double, int>> by_op;
    static bool callback(ggml_tensor* t, bool ask, void* user) {
        auto* p = static_cast<OpProfile*>(user);
        if (ask)
            return true;
        const auto now = std::chrono::steady_clock::now();
        auto& e = p->by_op[ggml_op_desc(t)];
        e.first += std::chrono::duration<double, std::milli>(now - p->last).count();
        e.second++;
        p->last = now;
        return true;
    }
    void report() const {
        std::vector<std::pair<double, std::string>> rows;
        for (const auto& kv : by_op)
            rows.emplace_back(kv.second.first, kv.first + " x" + std::to_string(kv.second.second));
        std::sort(rows.rbegin(), rows.rend());
        for (const auto& r : rows)
            std::fprintf(stderr, "mossformer2_ss_profile: %10.1f ms  %s\n", r.first, r.second.c_str());
    }
};

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

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *frames = nullptr, *pe = nullptr, *positions = nullptr, *out = nullptr;
    std::map<std::string, ggml_tensor*> captures;
    int n_samples = 0;
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
        frames = pe = positions = out = nullptr;
        captures.clear();
        n_samples = 0;
        wanted.clear();
    }
};
} // namespace

struct mossformer2_ss_context {
    HParams hp;
    mossformer2_ss_params params{};
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

namespace {
ggml_tensor* weight(mossformer2_ss_context& m, const std::string& n) {
    const auto it = m.weights.tensors.find(n);
    if (it == m.weights.tensors.end())
        throw std::runtime_error("missing tensor: " + n);
    return it->second;
}

void expect(mossformer2_ss_context& m, const std::string& n, int64_t d0, int64_t d1 = 1, int64_t d2 = 1) {
    const ggml_tensor* w = weight(m, n);
    if (w->ne[0] != d0 || w->ne[1] != d1 || w->ne[2] != d2 || w->ne[3] != 1)
        throw std::runtime_error("incompatible tensor shape: " + n);
}

void validate_weights(mossformer2_ss_context& m) {
    const auto& h = m.hp;
    const auto& b = h.blk;
    const int k = b.conv_kernel, lo = 2 * b.fsmn_lorder - 1;
    expect(m, "enc.weight", h.enc_kernel, h.enc_dim);
    expect(m, "dec.weight", h.enc_kernel, h.enc_dim);
    expect(m, "mn.norm.weight", h.enc_dim);
    expect(m, "mn.conv_enc.weight", h.enc_dim, b.dim);
    expect(m, "mn.pos_enc.inv_freq", b.dim / 2);
    expect(m, "mn.pos_enc.scale", 1);
    expect(m, "mn.prelu", 1);
    expect(m, "mn.conv_out.weight", b.dim, b.dim * h.num_spks);
    expect(m, "mn.output.weight", b.dim, b.dim);
    expect(m, "mn.output_gate.weight", b.dim, b.dim);
    expect(m, "mn.conv_dec.weight", b.dim, h.enc_dim);
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
        expect(m, p + "fsmn.ddn.conv1.weight", lo, b.fsmn_inner);
        expect(m, p + "fsmn.ddn.conv2.weight", lo, 2, b.fsmn_inner);
        expect(m, p + "fsmn.ddn.prelu1", b.fsmn_inner);
        expect(m, p + "fsmn.conv2.weight", b.fsmn_inner, b.dim);
    }
    for (const auto& kv : m.weights.tensors)
        if (ggml_is_quantized(kv.second->type))
            m.quantized = true;
}

// Load-time F32 depthwise kernels on the compute backend: tap-major copies for
// the CPU custom op, or zero-inserted dilated kernels for the portable path.
void prepare_derived(mossformer2_ss_context& m) {
    const auto kernels = mf2::dw_kernels(m.hp.blk);
    ggml_init_params ip{ggml_tensor_overhead() * (kernels.size() * 2 + 4), nullptr, true};
    m.derived_ctx = ggml_init(ip);
    if (!m.derived_ctx)
        throw std::runtime_error("cannot allocate derived tensor metadata");
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> data;
    auto add = [&](const std::string& name, int64_t ne0, int64_t ne1, std::vector<float> values) {
        ggml_tensor* t = ggml_new_tensor_2d(m.derived_ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(t, name.c_str());
        m.derived[name] = t;
        data.emplace_back(t, std::move(values));
    };
    for (const auto& k : kernels) {
        ggml_tensor* w = weight(m, k.name);
        const int K = (int)w->ne[0], C = (int)ggml_nelements(w) / (K * k.gin);
        const auto values = tensor_f32(w); // PyTorch order [C, gin, K]
        if (!m.portable_dwconv)
            add(mf2::tap_name(k.name), (int64_t)C * k.gin, K, mf2::tap_kernel(values.data(), C, k.gin, K));
        else if (k.gin > 1 || k.dil > 1) {
            auto dil = mf2::dilate_kernel(values.data(), C, k.gin, K, k.dil);
            for (int j = 0; j < k.gin; ++j)
                add(mf2::dilated_name(k.name, j), (int64_t)(K - 1) * k.dil + 1, C, std::move(dil[j]));
        }
    }
    if (data.empty())
        return;
    m.derived_buf = ggml_backend_alloc_ctx_tensors(m.derived_ctx, m.backend);
    if (!m.derived_buf)
        throw std::runtime_error("cannot allocate derived kernels");
    for (auto& d : data)
        ggml_backend_tensor_set(d.first, d.second.data(), 0, d.second.size() * sizeof(float));
}

int encoder_frames(const HParams& h, int n) {
    const int stride = h.enc_kernel / 2;
    return n < h.enc_kernel ? 0 : (n - h.enc_kernel) / stride + 1;
}

void build_graph(mossformer2_ss_context& m, int n, const std::set<std::string>& wanted) {
    auto& gr = m.graph;
    if (gr.n_samples == n && gr.wanted == wanted)
        return;
    gr.clear();
    const auto& h = m.hp;
    const int T = encoder_frames(h, n), K = h.enc_kernel;
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

    gr.frames = ggml_new_tensor_2d(g, GGML_TYPE_F32, K, T);
    gr.pe = ggml_new_tensor_2d(g, GGML_TYPE_F32, h.blk.dim, T);
    gr.positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    for (ggml_tensor* t : {gr.frames, gr.pe, gr.positions})
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

    // Encoder: Conv1d(1, N, K, stride K/2, bias=False) + ReLU on host-framed input.
    ggml_tensor* enc = ggml_relu(g, ggml_mul_mat(g, weight(m, "enc.weight"), gr.frames)); // [N, T]
    c.capture("enc_out", enc);
    // MaskNet: GroupNorm(1, N) -> Conv1d(N, D, 1) -> + ScaledSinuEmbedding
    ggml_tensor* x = mf2::global_group_norm(c, enc, "mn.norm", 1e-8f);
    c.capture("mn_norm", x);
    x = ggml_add(g, mf2::linear(c, x, "mn.conv_enc", false), gr.pe);
    c.capture("mn_in", x);
    x = mf2::computation_block(c, x, h.blk, "mn.intra_ln_norm", "mn.intra_norm");
    c.capture("mdl_out", x);
    x = mf2::prelu(g, x, weight(m, "mn.prelu"));
    x = mf2::linear(c, x, "mn.conv_out"); // [D * spks, T]
    // view(B * spks, D, S): speaker s owns channels [s * D, (s + 1) * D).
    const int D = h.blk.dim, S = h.num_spks;
    x = ggml_view_3d(g, x, D, S, T, (size_t)D * sizeof(float), x->nb[1], 0);
    x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3)); // [D, T, S]
    x = ggml_reshape_2d(g, x, D, (int64_t)T * S);
    ggml_tensor* o = ggml_tanh(g, mf2::linear(c, x, "mn.output"));
    ggml_tensor* gate = ggml_sigmoid(g, mf2::linear(c, x, "mn.output_gate"));
    x = ggml_relu(g, mf2::linear(c, ggml_mul(g, o, gate), "mn.conv_dec", false)); // [N, T * S]
    ggml_tensor* masks = ggml_reshape_3d(g, x, h.enc_dim, T, S);
    c.capture("masks", masks);
    // Decoder (ConvTranspose1d(N, 1, K, K/2, bias=False)) as per-frame filter
    // bank projections; the stride-K/2 overlap-add runs on the host.
    ggml_tensor* sep = ggml_mul(g, masks, enc);
    ggml_tensor* dec_t = ggml_cont(g, ggml_transpose(g, weight(m, "dec.weight")));       // [N, K]
    gr.out = ggml_mul_mat(g, dec_t, ggml_reshape_2d(g, sep, h.enc_dim, (int64_t)T * S)); // [K, T * S]
    ggml_set_output(gr.out);
    ggml_build_forward_expand(gr.graph, gr.out);
    if (!ggml_backend_sched_alloc_graph(gr.sched, gr.graph))
        throw std::runtime_error("graph allocation failed");
    gr.n_samples = n;
    gr.wanted = wanted;
}

// MossFormer.forward on exactly `n` samples: returns num_spks waveforms of n.
std::vector<std::vector<float>> forward(mossformer2_ss_context& m, const float* pcm, int n, Captures* caps = nullptr,
                                        const std::set<std::string>& wanted = {}) {
    StageTimer timer("forward");
    const auto& h = m.hp;
    const int K = h.enc_kernel, stride = K / 2, T = encoder_frames(h, n), S = h.num_spks;
    if (T <= 0)
        throw std::runtime_error("input shorter than the encoder kernel");
    build_graph(m, n, wanted);
    auto& gr = m.graph;
    std::vector<float> frames((size_t)K * T);
    for (int t = 0; t < T; ++t)
        std::memcpy(frames.data() + (size_t)t * K, pcm + (size_t)t * stride, K * sizeof(float));
    const auto pe = mf2::sinu_embedding(T, m.inv_freq, m.pos_scale);
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t)
        pos[t] = t;
    ggml_backend_tensor_set(gr.frames, frames.data(), 0, frames.size() * sizeof(float));
    ggml_backend_tensor_set(gr.pe, pe.data(), 0, pe.size() * sizeof(float));
    ggml_backend_tensor_set(gr.positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    std::unique_ptr<OpProfile> prof;
    if (env_flag("CRISPASR_MF2SS_PROFILE")) {
        prof = std::make_unique<OpProfile>();
        prof->last = std::chrono::steady_clock::now();
        ggml_backend_sched_set_eval_callback(gr.sched, OpProfile::callback, prof.get());
    }
    const bool ok = ggml_backend_sched_graph_compute(gr.sched, gr.graph) == GGML_STATUS_SUCCESS;
    if (prof) {
        ggml_backend_sched_set_eval_callback(gr.sched, nullptr, nullptr);
        prof->report();
    }
    if (!ok)
        throw std::runtime_error("graph execution failed");
    const auto proj = tensor_f32(gr.out); // [K, T, S]
    const int n_est = (T - 1) * stride + K;
    std::vector<std::vector<float>> out(S, std::vector<float>(n, 0.0f));
    for (int s = 0; s < S; ++s) {
        std::vector<float> y(n_est, 0.0f);
        for (int t = 0; t < T; ++t) {
            const float* f = proj.data() + ((size_t)s * T + t) * K;
            for (int k = 0; k < K; ++k)
                y[(size_t)t * stride + k] += f[k];
        }
        // Pad or trim to the input length (MossFormer.forward).
        std::copy_n(y.begin(), std::min(n, n_est), out[s].begin());
    }
    if (caps) {
        for (const auto& item : gr.captures)
            (*caps)[item.first] = tensor_f32(item.second);
        for (int s = 0; s < S; ++s)
            (*caps)["dec_spk" + std::to_string(s + 1)] = out[s];
    }
    return out;
}

// utils/decode.py decode_one_audio_mossformer2_ss_16k, truncated to the input
// length like ClearVoice.decode().
std::vector<std::vector<float>> decode(mossformer2_ss_context& m, const std::vector<float>& input) {
    const auto& h = m.hp;
    const int S = h.num_spks;
    const int64_t t0 = (int64_t)input.size();
    const int window = (int)(h.sample_rate * h.decode_window);
    const int stride = (int)(window * h.decode_stride_ratio);
    double sq = 0;
    for (float v : input)
        sq += (double)v * v;
    const double rms_input = std::sqrt(sq / (double)t0);
    const bool segment = !m.params.no_segment && t0 > (int64_t)(h.sample_rate * h.one_time_decode_length);
    int64_t t = t0;
    if (t < window)
        t = window;
    else if (t < window + stride)
        t = window + stride;
    else if ((t - window) % stride != 0)
        t += t - (t - window) / stride * stride; // upstream's padding expression, verbatim
    std::vector<float> padded(input);
    padded.resize((size_t)t, 0.0f);
    std::vector<std::vector<double>> acc(S, std::vector<double>((size_t)t, 0.0));
    if (segment) {
        const int give_up = (window - stride) / 2;
        for (int64_t idx = 0; idx + window <= t; idx += stride) {
            const auto out = forward(m, padded.data() + idx, window);
            for (int s = 0; s < S; ++s) {
                const int64_t lo = idx == 0 ? 0 : give_up, hi = window - give_up;
                for (int64_t i = lo; i < hi; ++i)
                    acc[s][(size_t)(idx + i)] = out[s][(size_t)i];
            }
        }
    } else {
        if (t > std::numeric_limits<int>::max())
            throw std::runtime_error("input too long");
        const auto out = forward(m, padded.data(), (int)t);
        for (int s = 0; s < S; ++s)
            std::copy(out[s].begin(), out[s].end(), acc[s].begin());
    }
    std::vector<std::vector<float>> result(S, std::vector<float>((size_t)t0));
    for (int s = 0; s < S; ++s) {
        double so = 0;
        for (double v : acc[s])
            so += v * v;
        const double rms_out = std::sqrt(so / (double)t);
        // Upstream divides unconditionally (NaN on an all-zero output); keep
        // silence silent instead.
        const double gain = rms_out > 0 ? rms_input / rms_out : 0.0;
        for (int64_t i = 0; i < t0; ++i)
            result[s][(size_t)i] = (float)(acc[s][(size_t)i] * gain);
    }
    return result;
}
} // namespace

mossformer2_ss_params mossformer2_ss_default_params(void) {
    return {0, true, 0, false};
}

mossformer2_ss_context* mossformer2_ss_init_from_file(const char* path, mossformer2_ss_params params) {
    if (!path || !*path)
        return nullptr;
    std::unique_ptr<mossformer2_ss_context, decltype(&mossformer2_ss_free)> m(new mossformer2_ss_context,
                                                                              mossformer2_ss_free);
    try {
        StageTimer timer("load");
        gguf_context* meta = core_gguf::open_metadata(path);
        if (!meta)
            return nullptr;
        const bool valid = read_hparams(meta, m->hp);
        core_gguf::free_metadata(meta);
        if (!valid)
            throw std::runtime_error("unsupported architecture or incomplete MossFormer2_SS_16K metadata");
        if (params.gpu_device < 0)
            throw std::runtime_error("invalid runtime parameters");
        params.n_threads = params.n_threads > 0 ? params.n_threads : std::min(8, core_parallel::resolve_threads(8, 0));
        m->params = params;
        if (params.use_gpu)
            ggml_backend_load_all();
        m->cpu = core_cpu_backend::init();
        if (!m->cpu)
            throw std::runtime_error("CPU backend unavailable");
        core_cpu_backend::set_n_threads(m->cpu, params.n_threads);
        if (params.use_gpu) {
            crispasr_set_gpu_device_pref(params.gpu_device);
            m->backend = crispasr_init_gpu_backend();
            if (m->backend && core_cpu_backend::is_cpu(m->backend)) {
                ggml_backend_free(m->backend);
                m->backend = nullptr;
            }
        }
        if (!m->backend)
            m->backend = m->cpu;
        // ggml_ssm_conv on GPUs only implements a few kernel widths; split the
        // 17/39/77-tap kernels there (CRISPASR_MF2_PORTABLE_DWCONV=1 forces it
        // on CPU for verification).
        m->portable_dwconv = m->backend != m->cpu || env_flag("CRISPASR_MF2_PORTABLE_DWCONV");
        const bool loaded = m->backend == m->cpu
                                ? core_gguf::load_weights(path, m->cpu, "mossformer2_ss", m->weights)
                                : core_gguf::load_weights_fit(path, m->backend, m->cpu, "mossformer2_ss", m->weights);
        if (!loaded)
            throw std::runtime_error("weight load failed");
        validate_weights(*m);
        m->inv_freq = tensor_f32(weight(*m, "mn.pos_enc.inv_freq"));
        m->pos_scale = tensor_f32(weight(*m, "mn.pos_enc.scale"))[0];
        prepare_derived(*m);
        std::fprintf(stderr, "mossformer2_ss: backend=%s threads=%d layers=%d%s%s\n", ggml_backend_name(m->backend),
                     params.n_threads, m->hp.blk.n_layers, m->quantized ? " quantized" : "",
                     m->portable_dwconv ? " dwconv=split" : "");
        return m.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_ss: %s\n", e.what());
        return nullptr;
    }
}

void mossformer2_ss_free(mossformer2_ss_context* ctx) {
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

mossformer2_ss_result* mossformer2_ss_separate(mossformer2_ss_context* ctx, const float* pcm, int n_samples,
                                               int in_channels) {
    if (!ctx || !pcm || n_samples <= 0 || (in_channels != 1 && in_channels != 2))
        return nullptr;
    try {
        StageTimer timer("separate");
        std::vector<float> mono((size_t)n_samples);
        for (int i = 0; i < n_samples; ++i) {
            const float v = in_channels == 1 ? pcm[i] : 0.5f * (pcm[2 * (size_t)i] + pcm[2 * (size_t)i + 1]);
            if (!std::isfinite(v))
                return nullptr;
            mono[i] = v;
        }
        const auto sources = decode(*ctx, mono);
        const int S = ctx->hp.num_spks;
        std::unique_ptr<mossformer2_ss_result, decltype(&mossformer2_ss_result_free)> r(new mossformer2_ss_result{},
                                                                                        mossformer2_ss_result_free);
        r->n_sources = S;
        r->n_channels = 1;
        r->n_samples = n_samples;
        r->sample_rate = ctx->hp.sample_rate;
        r->sources = new float* [S] {};
        r->source_names = new const char* [S] {};
        for (int s = 0; s < S; ++s) {
            if (!std::all_of(sources[s].begin(), sources[s].end(), [](float x) { return std::isfinite(x); }))
                throw std::runtime_error("nonfinite separated waveform");
            r->sources[s] = new float[sources[s].size()];
            std::copy(sources[s].begin(), sources[s].end(), r->sources[s]);
            r->source_names[s] = SOURCES[s];
        }
        return r.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_ss: separation failed: %s\n", e.what());
        return nullptr;
    }
}

void mossformer2_ss_result_free(mossformer2_ss_result* r) {
    if (!r)
        return;
    if (r->sources)
        for (int s = 0; s < r->n_sources; ++s)
            delete[] r->sources[s];
    delete[] r->sources;
    delete[] r->source_names;
    delete r;
}

int mossformer2_ss_sample_rate(const mossformer2_ss_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}
int mossformer2_ss_n_sources(const mossformer2_ss_context* ctx) {
    return ctx ? ctx->hp.num_spks : 0;
}
const char* mossformer2_ss_source_name(const mossformer2_ss_context* ctx, int idx) {
    return ctx && idx >= 0 && idx < ctx->hp.num_spks && idx < 2 ? SOURCES[idx] : nullptr;
}
const char* mossformer2_ss_backend_name(const mossformer2_ss_context* ctx) {
    return ctx && ctx->backend ? ggml_backend_name(ctx->backend) : nullptr;
}

int mossformer2_ss_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // the reference carries the exact input waveform
    auto params = mossformer2_ss_default_params();
    params.use_gpu = env_flag("CRISPASR_MF2SS_GPU", false);
    if (const char* threads = std::getenv("CRISPASR_MF2SS_THREADS"))
        params.n_threads = std::max(1, std::atoi(threads));
    std::unique_ptr<mossformer2_ss_context, decltype(&mossformer2_ss_free)> m(
        mossformer2_ss_init_from_file(model_gguf, params), mossformer2_ss_free);
    if (!m)
        return 2;
    core_gguf::WeightLoad ref;
    if (!core_gguf::load_weights(ref_gguf, m->cpu, "mossformer2_ss_reference", ref))
        return 2;
    int failures = 0, compared = 0;
    try {
        const auto it = ref.tensors.find("input_audio");
        if (it == ref.tensors.end() || ggml_n_dims(it->second) != 1 || it->second->ne[0] < m->hp.enc_kernel)
            throw std::runtime_error("reference needs a 1-D input_audio of at least one encoder frame");
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
            StageTimer timer("diff_forward");
            forward(*m, audio.data(), n, &actual, wanted);
        }
        if (any_e2e) {
            StageTimer timer("diff_e2e");
            const auto e2e = decode(*m, audio);
            for (int s = 0; s < m->hp.num_spks; ++s)
                actual["e2e_spk" + std::to_string(s + 1)] = e2e[s];
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
        std::printf("mossformer2_ss diff: %d stages compared, %d failed, cosine > %.3f (%s)\n", compared, failures,
                    threshold, m->quantized ? "quantized" : "F32/F16");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mossformer2_ss diff: %s\n", e.what());
        failures = 1;
    }
    core_gguf::free_weights(ref);
    return failures ? 1 : 0;
}
