// voice_isolator.cpp — Inworld Voice Isolator v1 runtime (see voice_isolator.h).
//
// Every numbered comment below names the upstream line it mirrors
// (inworld/voice_isolation/model.py unless stated). The STFT/iSTFT, the band
// packing and the bounded correction run on the host in float32, as upstream
// keeps them in float32 under every precision; the network runs as one ggml
// graph per stage (encoder, each block, decoder) so the peak activation memory
// is one block's, not the whole network's.

#include "voice_isolator.h"

#include "core/crispasr_env.h"
#include "core/fft.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/parallel_for.h"
#include "core/torchaudio_resample.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kLnEps = 1e-5f; // torch.nn.LayerNorm default

struct Hparams {
    int width = 0, heads = 0, n_fft = 0, hop = 0, sample_rate = 0;
    int d_state = 0, d_conv = 0, expand = 0, dt_rank = 0;
    int n_blocks = 0, n_bands = 0;
    std::vector<int> is_mamba;
    std::vector<int> band_lo, band_hi;
    float correction_bound = 0.5f, chunk_seconds = 8.0f, overlap_seconds = 2.0f;
    bool causal = false;
    int lookahead_default = 0, attention_window = 0;
    std::vector<int> lookahead_choices;
    int bins() const { return n_fft / 2 + 1; }
};

struct MambaW {
    ggml_tensor *in_proj = nullptr, *conv_w = nullptr, *conv_b = nullptr, *x_proj = nullptr;
    ggml_tensor *dt_w = nullptr, *dt_b = nullptr, *a_log = nullptr, *d = nullptr, *out_proj = nullptr;
};
struct BiMambaW {
    MambaW fwd, bwd;
    bool bidirectional = true;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr, *norm_w = nullptr, *norm_b = nullptr;
};
struct AttnW {
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr, *proj_w = nullptr, *proj_b = nullptr;
    ggml_tensor *n1_w = nullptr, *n1_b = nullptr, *n2_w = nullptr, *n2_b = nullptr;
    ggml_tensor *f1_w = nullptr, *f1_b = nullptr, *f2_w = nullptr, *f2_b = nullptr;
};
struct BlockW {
    bool mamba = false;
    AttnW time, band;
    BiMambaW tm, fm;
};
struct BandW {
    ggml_tensor *ln_w = nullptr, *ln_b = nullptr, *lin_w = nullptr, *lin_b = nullptr;
    ggml_tensor *h_ln_w = nullptr, *h_ln_b = nullptr, *h1_w = nullptr, *h1_b = nullptr, *h2_w = nullptr,
                *h2_b = nullptr;
};

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

ggml_tensor* layer_norm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    x = ggml_norm(c, x, kLnEps);
    x = ggml_mul(c, x, w);
    return ggml_add(c, x, b);
}

ggml_tensor* linear(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* y = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, y, b) : y;
}

} // namespace

struct voice_isolator_context {
    Hparams hp;
    voice_isolator_params params{};
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    std::vector<BandW> bands;
    std::vector<BlockW> blocks;
    ggml_tensor* band_embedding = nullptr;
    std::vector<float> dec_scale1p, dec_shift, lookahead_row; // host-precomputed constants
    int lookahead = 0;
    std::vector<float> window; // periodic Hann, n_fft
    bool f32_weights = true;
    std::string weight_type = "F32";
};

namespace {

ggml_tensor* need(voice_isolator_context& m, const std::string& name) {
    return core_gguf::require(m.weights.tensors, name.c_str(), "voice_isolator");
}

ggml_tensor* opt(voice_isolator_context& m, const std::string& name) {
    auto it = m.weights.tensors.find(name);
    return it == m.weights.tensors.end() ? nullptr : it->second;
}

bool read_hparams(gguf_context* g, Hparams& h) {
    if (core_gguf::kv_str(g, "general.architecture", "") != "voice-isolator")
        return false;
    const std::string p = "voice_isolator.";
    auto u32 = [&](const char* k) { return (int)core_gguf::kv_u32(g, (p + k).c_str(), 0); };
    h.width = u32("width");
    h.heads = u32("num_heads");
    h.n_fft = u32("n_fft");
    h.hop = u32("hop_length");
    h.sample_rate = u32("sample_rate");
    h.d_state = u32("mamba_state");
    h.d_conv = u32("mamba_conv");
    h.expand = u32("mamba_expand");
    h.n_blocks = u32("n_blocks");
    h.n_bands = u32("n_bands");
    h.dt_rank = (h.width + 15) / 16; // model.py:389 math.ceil(width / 16)
    h.correction_bound = core_gguf::kv_f32(g, (p + "correction_bound").c_str(), 0.5f);
    h.chunk_seconds = core_gguf::kv_f32(g, (p + "chunk_seconds").c_str(), 8.0f);
    h.overlap_seconds = core_gguf::kv_f32(g, (p + "overlap_seconds").c_str(), 2.0f);
    h.causal = core_gguf::kv_bool(g, (p + "causal").c_str(), false);
    h.lookahead_default = u32("lookahead_frames");
    h.attention_window = u32("attention_window_frames");
    const int n_choices = u32("n_lookahead_choices");
    auto ints = [&](const char* k, std::vector<int>& out) {
        const int id = gguf_find_key(g, (p + k).c_str());
        if (id < 0)
            return false;
        const size_t n = gguf_get_arr_n(g, id);
        const auto t = gguf_get_arr_type(g, id);
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            if (t == GGUF_TYPE_INT32)
                out[i] = ((const int32_t*)gguf_get_arr_data(g, id))[i];
            else if (t == GGUF_TYPE_UINT32)
                out[i] = (int)((const uint32_t*)gguf_get_arr_data(g, id))[i];
            else if (t == GGUF_TYPE_INT64)
                out[i] = (int)((const int64_t*)gguf_get_arr_data(g, id))[i];
            else
                return false;
        }
        return true;
    };
    std::vector<int> edges;
    if (!ints("block_is_mamba", h.is_mamba) || !ints("band_edges", edges))
        return false;
    if (n_choices > 0 && !ints("lookahead_choices", h.lookahead_choices))
        return false;
    h.lookahead_choices.resize(n_choices);
    if ((int)edges.size() != 2 * h.n_bands || (int)h.is_mamba.size() != h.n_blocks)
        return false;
    for (int b = 0; b < h.n_bands; b++) {
        h.band_lo.push_back(edges[2 * b]);
        h.band_hi.push_back(edges[2 * b + 1]);
    }
    return h.width > 0 && h.heads > 0 && h.width % (2 * h.heads) == 0 && h.n_fft == 2048 && h.hop > 0 &&
           h.d_state > 0 && h.d_conv > 0 && h.expand > 0 && h.n_bands > 0 && h.band_lo[0] == 0 &&
           h.band_hi.back() == h.bins();
}

void bind_weights(voice_isolator_context& m) {
    const Hparams& h = m.hp;
    m.band_embedding = need(m, "band_embedding");
    for (int b = 0; b < h.n_bands; b++) {
        BandW w;
        const std::string e = "encoders." + std::to_string(b) + ".";
        w.ln_w = need(m, e + "0.weight");
        w.ln_b = need(m, e + "0.bias");
        w.lin_w = need(m, e + "1.weight");
        w.lin_b = need(m, e + "1.bias");
        const std::string d = "denoise_decoder.heads." + std::to_string(b) + ".";
        w.h_ln_w = need(m, d + "0.weight");
        w.h_ln_b = need(m, d + "0.bias");
        w.h1_w = need(m, d + "1.weight");
        w.h1_b = need(m, d + "1.bias");
        w.h2_w = need(m, d + "3.weight");
        w.h2_b = need(m, d + "3.bias");
        const int nb = h.band_hi[b] - h.band_lo[b];
        if (w.lin_w->ne[0] != 2 * nb || w.h2_w->ne[1] != 4 * nb)
            throw std::runtime_error("band " + std::to_string(b) + " shape mismatch");
        m.bands.push_back(w);
    }
    auto mamba = [&](const std::string& p) {
        MambaW w;
        w.in_proj = need(m, p + "in_proj.weight");
        w.conv_w = need(m, p + "conv1d.weight");
        w.conv_b = need(m, p + "conv1d.bias");
        w.x_proj = need(m, p + "x_proj.weight");
        w.dt_w = need(m, p + "dt_proj.weight");
        w.dt_b = need(m, p + "dt_proj.bias");
        w.a_log = need(m, p + "A_log");
        w.d = need(m, p + "D");
        w.out_proj = need(m, p + "out_proj.weight");
        return w;
    };
    auto bimamba = [&](const std::string& p) {
        BiMambaW w;
        w.fwd = mamba(p + "forward_blocks.");
        w.bidirectional = opt(m, p + "backward_blocks.in_proj.weight") != nullptr;
        if (w.bidirectional)
            w.bwd = mamba(p + "backward_blocks.");
        w.proj_w = need(m, p + "output_proj.weight");
        w.proj_b = need(m, p + "output_proj.bias");
        w.norm_w = need(m, p + "norm.weight");
        w.norm_b = need(m, p + "norm.bias");
        return w;
    };
    for (int i = 0; i < h.n_blocks; i++) {
        BlockW b;
        const std::string p = "blocks." + std::to_string(i) + ".";
        b.mamba = h.is_mamba[i] != 0;
        if (b.mamba) {
            b.tm = bimamba(p + "time_mamba.");
            b.fm = bimamba(p + "freq_mamba.");
            if (b.tm.bidirectional == h.causal || !b.fm.bidirectional)
                throw std::runtime_error("Mamba direction does not match the checkpoint's causality");
        } else {
            const std::string t = p + "time_transformer.";
            b.time.qkv_w = need(m, t + "attn.qkv.weight");
            b.time.qkv_b = need(m, t + "attn.qkv.bias");
            b.time.proj_w = need(m, t + "attn.proj.weight");
            b.time.proj_b = need(m, t + "attn.proj.bias");
            b.time.n1_w = need(m, t + "norm1.weight");
            b.time.n1_b = need(m, t + "norm1.bias");
            b.time.n2_w = need(m, t + "norm2.weight");
            b.time.n2_b = need(m, t + "norm2.bias");
            b.time.f1_w = need(m, t + "ffn.0.weight");
            b.time.f1_b = need(m, t + "ffn.0.bias");
            b.time.f2_w = need(m, t + "ffn.3.weight");
            b.time.f2_b = need(m, t + "ffn.3.bias");
            const std::string f = p + "band_transformer.";
            b.band.qkv_w = need(m, f + "self_attn.in_proj_weight");
            b.band.qkv_b = need(m, f + "self_attn.in_proj_bias");
            b.band.proj_w = need(m, f + "self_attn.out_proj.weight");
            b.band.proj_b = need(m, f + "self_attn.out_proj.bias");
            b.band.n1_w = need(m, f + "norm1.weight");
            b.band.n1_b = need(m, f + "norm1.bias");
            b.band.n2_w = need(m, f + "norm2.weight");
            b.band.n2_b = need(m, f + "norm2.bias");
            b.band.f1_w = need(m, f + "linear1.weight");
            b.band.f1_b = need(m, f + "linear1.bias");
            b.band.f2_w = need(m, f + "linear2.weight");
            b.band.f2_b = need(m, f + "linear2.bias");
        }
        m.blocks.push_back(b);
    }
    // _Decoder.forward: scale, shift = condition(controls=[1, 0]).chunk(2)
    // -> weight[:, 0] + bias. Constant per checkpoint: computed once here.
    ggml_tensor* cw = need(m, "denoise_decoder.condition.weight"); // ggml [2, 2W]
    ggml_tensor* cb = need(m, "denoise_decoder.condition.bias");   // [2W]
    const int W = h.width;
    if (cw->type != GGML_TYPE_F32 || cb->type != GGML_TYPE_F32 || cw->ne[0] != 2 || cw->ne[1] != 2 * W)
        throw std::runtime_error("decoder condition must be F32 [2W, 2]");
    std::vector<float> wv(4 * W), bv(2 * W);
    ggml_backend_tensor_get(cw, wv.data(), 0, wv.size() * sizeof(float));
    ggml_backend_tensor_get(cb, bv.data(), 0, bv.size() * sizeof(float));
    m.dec_scale1p.resize(W);
    m.dec_shift.resize(W);
    for (int i = 0; i < W; i++) {
        m.dec_scale1p[i] = 1.0f + (wv[2 * i] + bv[i]);
        m.dec_shift[i] = wv[2 * (W + i)] + bv[W + i];
    }
    if (h.causal && !h.lookahead_choices.empty()) {
        ggml_tensor* le = need(m, "lookahead_embedding.weight"); // [W, n_choices]
        std::vector<float> all((size_t)ggml_nelements(le));
        ggml_backend_tensor_get(le, all.data(), 0, all.size() * sizeof(float));
        const auto it = std::find(h.lookahead_choices.begin(), h.lookahead_choices.end(), m.lookahead);
        if (it == h.lookahead_choices.end())
            throw std::runtime_error("look-ahead " + std::to_string(m.lookahead) + " is not a trained position");
        const int idx = (int)(it - h.lookahead_choices.begin());
        m.lookahead_row.assign(all.begin() + (size_t)idx * W, all.begin() + (size_t)(idx + 1) * W);
    }
}

// --- one ggml graph: build, allocate, set inputs, compute, read outputs -----

struct Run {
    voice_isolator_context& m;
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph* gf = nullptr;
    std::vector<std::pair<ggml_tensor*, const void*>> inputs;
    std::vector<ggml_tensor*> zeros;
    explicit Run(voice_isolator_context& model) : m(model) {
        ggml_init_params ip = {ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr,
                               true};
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, 16384, false);
    }
    ~Run() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (ctx)
            ggml_free(ctx);
    }
    ggml_tensor* input(ggml_type t, std::initializer_list<int64_t> ne, const void* data) {
        int64_t d[4] = {1, 1, 1, 1};
        int i = 0;
        for (int64_t v : ne)
            d[i++] = v;
        ggml_tensor* x = ggml_new_tensor(ctx, t, (int)ne.size(), d);
        ggml_set_input(x);
        if (data)
            inputs.push_back({x, data});
        else
            zeros.push_back(x);
        return x;
    }
    void compute(const std::vector<ggml_tensor*>& outs) {
        for (ggml_tensor* o : outs) {
            ggml_set_output(o);
            ggml_build_forward_expand(gf, o);
        }
        ggml_backend_t bk[2] = {m.backend, m.cpu};
        const int n = m.backend == m.cpu ? 1 : 2;
        sched = ggml_backend_sched_new(bk, nullptr, n, 16384, false, true);
        if (!sched || !ggml_backend_sched_alloc_graph(sched, gf))
            throw std::runtime_error("graph allocation failed");
        // An input the graph never reads (e.g. the reverse index of a causal,
        // forward-only Mamba) is left unallocated by the scheduler.
        for (auto& in : inputs)
            if (in.first->buffer)
                ggml_backend_tensor_set(in.first, in.second, 0, ggml_nbytes(in.first));
        for (ggml_tensor* z : zeros)
            if (z->buffer)
                ggml_backend_tensor_memset(z, 0, 0, ggml_nbytes(z));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph compute failed");
    }
};

void get(ggml_tensor* t, std::vector<float>& out) {
    out.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
}

// _TimeAttention / _BandTransformer attention core over x [W, L, S]:
// qkv -> (rope) -> SDPA -> (exclusive) -> out projection.
ggml_tensor* attention(Run& r, ggml_tensor* x, const AttnW& w, ggml_tensor* positions, bool exclusive,
                       ggml_tensor* mask) {
    ggml_context* c = r.ctx;
    const Hparams& h = r.m.hp;
    const int W = h.width, H = h.heads, hd = W / H;
    const int64_t L = x->ne[1], S = x->ne[2];
    ggml_tensor* qkv = linear(c, x, w.qkv_w, w.qkv_b); // [3W, L, S]; 3W is (3, H, hd)
    auto part = [&](int k) {
        return ggml_cont(c, ggml_view_4d(c, qkv, hd, H, L, S, (size_t)hd * 4, (size_t)3 * W * 4, (size_t)3 * W * L * 4,
                                         (size_t)k * W * 4));
    };
    ggml_tensor *q = part(0), *k = part(1), *v = part(2); // [hd, H, L, S]
    if (positions) {
        // apply_rope: halves rotated (NEOX), inv = 10000^(-i/half)
        q = ggml_rope_ext(c, q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3); // [hd, L, H, S]
    ggml_tensor* kp = ggml_permute(c, k, 0, 2, 1, 3);
    ggml_tensor* vp = ggml_permute(c, v, 0, 2, 1, 3);
    ggml_tensor* o = ggml_flash_attn_ext(c, qp, kp, vp, mask, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32); // o: [hd, H, L, S]; F32 accumulation (P100 lesson)
    if (exclusive) {
        // exclusive(): o - (o.v / max(v.v, 1e-6)) v, per head and token
        ggml_tensor* dot = ggml_sum_rows(c, ggml_mul(c, o, v));
        ggml_tensor* vv = ggml_clamp(c, ggml_sum_rows(c, ggml_sqr(c, v)), 1e-6f, 3.0e38f);
        o = ggml_sub(c, o, ggml_mul(c, v, ggml_div(c, dot, vv)));
    }
    o = ggml_reshape_3d(c, o, W, L, S);
    return linear(c, o, w.proj_w, w.proj_b);
}

// Pre-norm transformer layer (_TimeBlock / _BandTransformer) over x [W, L, S].
ggml_tensor* transformer(Run& r, ggml_tensor* x, const AttnW& w, ggml_tensor* positions, bool exclusive,
                         ggml_tensor* mask) {
    ggml_context* c = r.ctx;
    x = ggml_add(c, x, attention(r, layer_norm(c, x, w.n1_w, w.n1_b), w, positions, exclusive, mask));
    ggml_tensor* f = linear(c, layer_norm(c, x, w.n2_w, w.n2_b), w.f1_w, w.f1_b);
    f = ggml_gelu_erf(c, f); // torch.nn.GELU / F.gelu default: exact erf
    return ggml_add(c, x, linear(c, f, w.f2_w, w.f2_b));
}

// Mamba.forward (no carried state) over x [W, L, S] -> [W, L, S].
ggml_tensor* mamba(Run& r, ggml_tensor* x, const MambaW& w, ggml_tensor* seq_ids) {
    ggml_context* c = r.ctx;
    const Hparams& h = r.m.hp;
    const int64_t I = (int64_t)h.expand * h.width, L = x->ne[1], S = x->ne[2];
    const int R = h.dt_rank, N = h.d_state;
    ggml_tensor* xz = ggml_mul_mat(c, w.in_proj, x); // [2I, L, S]
    ggml_tensor* u = ggml_view_3d(c, xz, I, L, S, xz->nb[1], xz->nb[2], 0);
    ggml_tensor* z = ggml_cont(c, ggml_view_3d(c, xz, I, L, S, xz->nb[1], xz->nb[2], (size_t)I * 4));
    // depthwise Conv1d, padding k-1, first L outputs = causal conv
    ggml_tensor* ut = ggml_cont(c, ggml_permute(c, u, 1, 0, 2, 3)); // [L, I, S]
    ut = ggml_pad_ext(c, ut, h.d_conv - 1, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor* cv = ggml_ssm_conv(c, ut, w.conv_w); // [I, L, S]
    cv = ggml_silu(c, ggml_add(c, cv, w.conv_b));
    ggml_tensor* xdb = ggml_mul_mat(c, w.x_proj, cv); // [R + 2N, L, S]
    ggml_tensor* dt = ggml_cont(c, ggml_view_3d(c, xdb, R, L, S, xdb->nb[1], xdb->nb[2], 0));
    dt = ggml_add(c, ggml_mul_mat(c, w.dt_w, dt), w.dt_b); // softplus is applied inside ssm_scan
    ggml_tensor* B = ggml_view_4d(c, xdb, N, 1, L, S, (size_t)N * 4, xdb->nb[1], xdb->nb[2], (size_t)R * 4);
    ggml_tensor* C = ggml_view_4d(c, xdb, N, 1, L, S, (size_t)N * 4, xdb->nb[1], xdb->nb[2], (size_t)(R + N) * 4);
    ggml_tensor* A = ggml_neg(c, ggml_exp(c, w.a_log)); // [N, I]
    ggml_tensor* s0 = r.input(GGML_TYPE_F32, {N, 1, I, S}, nullptr);
    ggml_tensor* y = ggml_ssm_scan(c, s0, ggml_reshape_4d(c, cv, 1, I, L, S), dt, A, B, C, seq_ids, 1);
    y = ggml_view_3d(c, y, I, L, S, (size_t)I * 4, (size_t)I * L * 4, 0);
    y = ggml_add(c, y, ggml_mul(c, cv, w.d)); // + u * D
    y = ggml_mul(c, y, ggml_silu(c, z));      // * silu(z)
    return ggml_mul_mat(c, w.out_proj, y);
}

// BidirectionalMamba over x [W, L, S]; rev reverses L within each sequence.
ggml_tensor* bimamba(Run& r, ggml_tensor* x, const BiMambaW& w, ggml_tensor* seq_ids, ggml_tensor* rev) {
    ggml_context* c = r.ctx;
    ggml_tensor* f = ggml_add(c, mamba(r, x, w.fwd, seq_ids), x);
    ggml_tensor* y;
    if (!w.bidirectional) {
        y = linear(c, f, w.proj_w, w.proj_b); // causal: output_proj(forward + x)
    } else {
        ggml_tensor* xr = ggml_get_rows(c, x, rev);
        ggml_tensor* b = ggml_get_rows(c, mamba(r, xr, w.bwd, seq_ids), rev);
        y = linear(c, ggml_concat(c, f, ggml_add(c, b, x), 0), w.proj_w, w.proj_b);
    }
    return layer_norm(c, y, w.norm_w, w.norm_b);
}

std::vector<int32_t> reverse_ids(int L, int S) {
    std::vector<int32_t> v((size_t)L * S);
    for (int s = 0; s < S; s++)
        for (int t = 0; t < L; t++)
            v[(size_t)s * L + t] = L - 1 - t;
    return v;
}

std::vector<int32_t> iota(int n) {
    std::vector<int32_t> v(n);
    for (int i = 0; i < n; i++)
        v[i] = i;
    return v;
}

// Run one block on tokens [W, T, NB] in place.
// Host [W, A, B] <-> [W, B, A] (swap the two sequence axes).
void swap_axes(const std::vector<float>& in, int W, int A, int B, std::vector<float>& out) {
    out.resize(in.size());
    for (int b = 0; b < B; b++)
        for (int a = 0; a < A; a++)
            std::memcpy(&out[((size_t)a * B + b) * W], &in[((size_t)b * A + a) * W], (size_t)W * sizeof(float));
}

// Sequences are independent along both paths, so one axis pass runs as
// several graphs over groups of sequences: identical arithmetic, bounded
// activation memory (~kTokensPerGraph tokens per graph).
constexpr int64_t kTokensPerGraph = 8192;

// One axis pass over x [W, L, S] (host), in place: the time path (L = frames,
// S = bands) or the band path (L = bands, S = frames) of block `bi`.
void run_axis(voice_isolator_context& m, int bi, bool time_axis, int L, int S, std::vector<float>& x) {
    const Hparams& h = m.hp;
    const int W = h.width;
    const BlockW& b = m.blocks[bi];
    const int group = (int)std::max<int64_t>(1, std::min<int64_t>(S, kTokensPerGraph / std::max(1, L)));
    std::vector<int32_t> pos = iota(L);
    std::vector<ggml_fp16_t> mask;
    if (!b.mamba && time_axis && h.causal) {
        // window_mask / is_causal: key <= query and query - key < window
        mask.assign((size_t)L * L, ggml_fp32_to_fp16(-INFINITY));
        for (int q = 0; q < L; q++)
            for (int k = 0; k <= q; k++)
                if (h.attention_window <= 0 || q - k < h.attention_window)
                    mask[(size_t)q * L + k] = ggml_fp32_to_fp16(0.0f);
    }
    std::vector<float> chunk;
    for (int s0 = 0; s0 < S; s0 += group) {
        const int n = std::min(group, S - s0);
        float* base = x.data() + (size_t)s0 * L * W;
        Run r(m);
        ggml_context* c = r.ctx;
        ggml_tensor* in = r.input(GGML_TYPE_F32, {W, L, n}, base);
        ggml_tensor* out;
        std::vector<int32_t> ids, rev;
        if (!b.mamba) {
            if (time_axis) {
                ggml_tensor* positions = r.input(GGML_TYPE_I32, {L}, pos.data());
                ggml_tensor* kq = mask.empty() ? nullptr : r.input(GGML_TYPE_F16, {L, L}, mask.data());
                out = transformer(r, in, b.time, positions, /*exclusive=*/true, kq); // exclusive_axes == "time"
            } else {
                out = transformer(r, in, b.band, nullptr, /*exclusive=*/false, nullptr);
            }
        } else {
            ids = iota(n);
            rev = reverse_ids(L, n);
            ggml_tensor* sid = r.input(GGML_TYPE_I32, {n}, ids.data());
            ggml_tensor* rv = r.input(GGML_TYPE_I32, {L, n}, rev.data());
            out = ggml_add(c, in, bimamba(r, in, time_axis ? b.tm : b.fm, sid, rv)); // DualPathMamba residual
        }
        r.compute({out});
        get(out, chunk);
        std::memcpy(base, chunk.data(), chunk.size() * sizeof(float));
    }
}

// Run one block on tokens [W, T, NB] in place: time path, then band path.
void run_block(voice_isolator_context& m, int bi, int T, std::vector<float>& tokens) {
    const int W = m.hp.width, NB = m.hp.n_bands;
    run_axis(m, bi, /*time_axis=*/true, T, NB, tokens);
    std::vector<float> by_frame;
    swap_axes(tokens, W, T, NB, by_frame); // [W, NB, T]
    run_axis(m, bi, /*time_axis=*/false, NB, T, by_frame);
    swap_axes(by_frame, W, NB, T, tokens);
}

// encode(): per-band LayerNorm + Linear of interleaved (re, im) bins, then
// band_embedding (+ lookahead_embedding for causal checkpoints).
void run_encoder(voice_isolator_context& m, int T, const std::vector<float>& packed, std::vector<float>& tokens) {
    const Hparams& h = m.hp;
    const int W = h.width, NB = h.n_bands;
    Run r(m);
    ggml_context* c = r.ctx;
    ggml_tensor* in = r.input(GGML_TYPE_F32, {(int64_t)packed.size()}, packed.data());
    std::vector<ggml_tensor*> parts;
    size_t off = 0;
    for (int b = 0; b < NB; b++) {
        const int f2 = 2 * (h.band_hi[b] - h.band_lo[b]);
        ggml_tensor* v = ggml_view_2d(c, in, f2, T, (size_t)f2 * 4, off * 4);
        off += (size_t)f2 * T;
        ggml_tensor* y =
            linear(c, layer_norm(c, v, m.bands[b].ln_w, m.bands[b].ln_b), m.bands[b].lin_w, m.bands[b].lin_b);
        parts.push_back(ggml_reshape_3d(c, y, W, T, 1));
    }
    while (parts.size() > 1) { // balanced concat along the band axis
        std::vector<ggml_tensor*> next;
        for (size_t i = 0; i + 1 < parts.size(); i += 2)
            next.push_back(ggml_concat(c, parts[i], parts[i + 1], 2));
        if (parts.size() % 2)
            next.push_back(parts.back());
        parts.swap(next);
    }
    ggml_tensor* x = ggml_add(c, parts[0], ggml_reshape_3d(c, m.band_embedding, W, 1, NB));
    if (!m.lookahead_row.empty())
        x = ggml_add(c, x, r.input(GGML_TYPE_F32, {W}, m.lookahead_row.data()));
    r.compute({x});
    get(x, tokens);
}

// _Decoder: FiLM by the constant controls, then per-band heads -> raw [bins][T][4].
void run_decoder(voice_isolator_context& m, int T, const std::vector<float>& tokens, std::vector<float>& raw) {
    const Hparams& h = m.hp;
    const int W = h.width, NB = h.n_bands, F = h.bins();
    Run r(m);
    ggml_context* c = r.ctx;
    ggml_tensor* x = r.input(GGML_TYPE_F32, {W, T, NB}, tokens.data());
    x = ggml_add(c, ggml_mul(c, x, r.input(GGML_TYPE_F32, {W}, m.dec_scale1p.data())),
                 r.input(GGML_TYPE_F32, {W}, m.dec_shift.data()));
    std::vector<ggml_tensor*> outs;
    for (int b = 0; b < NB; b++) {
        const BandW& w = m.bands[b];
        ggml_tensor* v = ggml_view_2d(c, x, W, T, (size_t)W * 4, (size_t)b * W * T * 4);
        ggml_tensor* y = linear(c, layer_norm(c, v, w.h_ln_w, w.h_ln_b), w.h1_w, w.h1_b);
        y = linear(c, ggml_gelu_erf(c, y), w.h2_w, w.h2_b); // [4 nb, T]
        outs.push_back(y);
    }
    r.compute(outs);
    raw.assign((size_t)F * T * 4, 0.0f);
    std::vector<float> o;
    for (int b = 0; b < NB; b++) {
        get(outs[b], o);
        const int lo = h.band_lo[b], nb = h.band_hi[b] - lo;
        for (int t = 0; t < T; t++)
            for (int j = 0; j < nb; j++)
                for (int k = 0; k < 4; k++)
                    raw[((size_t)(lo + j) * T + t) * 4 + k] = o[((size_t)t * nb + j) * 4 + k];
    }
}

// torch.stft(center=True, reflect (or constant when n <= n_fft/2), periodic
// Hann, onesided) -> re/im [bins][T].
void stft(const voice_isolator_context& m, const float* x, int n, std::vector<float>& re, std::vector<float>& im,
          int& T) {
    const int N = m.hp.n_fft, hop = m.hp.hop, F = N / 2 + 1, pad = N / 2;
    T = 1 + n / hop;
    std::vector<float> xp((size_t)n + 2 * pad, 0.0f);
    const bool reflect = n > pad;
    for (int i = 0; i < (int)xp.size(); i++) {
        int j = i - pad;
        if (j < 0 || j >= n) {
            if (!reflect)
                continue;
            j = j < 0 ? -j : 2 * (n - 1) - j;
        }
        xp[i] = x[j];
    }
    re.assign((size_t)F * T, 0.0f);
    im.assign((size_t)F * T, 0.0f);
    core_parallel::for_each_chunk(T, m.params.n_threads, [&](int t0, int t1) {
        std::vector<float> br(N), bi(N);
        for (int t = t0; t < t1; t++) {
            for (int i = 0; i < N; i++) {
                br[i] = xp[(size_t)t * hop + i] * m.window[i];
                bi[i] = 0.0f;
            }
            core_fft::fft_radix2_inplace(br.data(), bi.data(), N);
            for (int f = 0; f < F; f++) {
                re[(size_t)f * T + t] = br[f];
                im[(size_t)f * T + t] = bi[f];
            }
        }
    });
}

// torch.istft(center=True, length=n): windowed overlap-add / sum(window^2).
void istft(const voice_isolator_context& m, const std::vector<float>& re, const std::vector<float>& im, int T, int n,
           std::vector<float>& y) {
    const int N = m.hp.n_fft, hop = m.hp.hop, F = N / 2 + 1, pad = N / 2;
    const size_t total = (size_t)N + (size_t)hop * (T - 1);
    std::vector<float> acc(total, 0.0f), env(total, 0.0f), frames((size_t)T * N);
    core_parallel::for_each_chunk(T, m.params.n_threads, [&](int t0, int t1) {
        std::vector<float> br(N), bi(N);
        for (int t = t0; t < t1; t++) {
            // inverse real FFT via conj(FFT(conj(X_full))) / N
            for (int f = 0; f < F; f++) {
                br[f] = re[(size_t)f * T + t];
                bi[f] = -im[(size_t)f * T + t];
            }
            for (int f = F; f < N; f++) {
                br[f] = re[(size_t)(N - f) * T + t];
                bi[f] = im[(size_t)(N - f) * T + t];
            }
            bi[0] = 0.0f;     // irfft ignores the imaginary DC
            bi[N / 2] = 0.0f; // ... and Nyquist parts
            core_fft::fft_radix2_inplace(br.data(), bi.data(), N);
            for (int i = 0; i < N; i++)
                frames[(size_t)t * N + i] = br[i] / (float)N * m.window[i];
        }
    });
    for (int t = 0; t < T; t++)
        for (int i = 0; i < N; i++) {
            acc[(size_t)t * hop + i] += frames[(size_t)t * N + i];
            env[(size_t)t * hop + i] += m.window[i] * m.window[i];
        }
    y.assign(n, 0.0f);
    for (int i = 0; i < n; i++) {
        const size_t j = (size_t)i + pad;
        if (j < total && env[j] > 1e-11f)
            y[i] = acc[j] / env[j];
    }
}

struct Stages {
    bool keep = false;
    std::vector<float> stft_ri; // [bins][T][2]
    std::vector<float> encode;
    std::vector<std::vector<float>> blocks;
    std::vector<float> raw;
};

// VoiceIsolatorNetwork.forward over one chunk.
std::vector<float> forward(voice_isolator_context& m, const float* audio, int n, Stages* st = nullptr,
                           const std::vector<std::vector<float>>* teacher = nullptr) {
    const Hparams& h = m.hp;
    const int F = h.bins();
    int T = 0;
    std::vector<float> re, im;
    stft(m, audio, n, re, im, T);
    if (st && st->keep) {
        st->stft_ri.resize((size_t)F * T * 2);
        for (size_t i = 0; i < (size_t)F * T; i++) {
            st->stft_ri[2 * i] = re[i];
            st->stft_ri[2 * i + 1] = im[i];
        }
    }
    // components [T, bins, 2] -> per band [T, 2 nb] (flatten(2): bin-major, re/im inner)
    std::vector<float> packed;
    for (int b = 0; b < h.n_bands; b++)
        for (int t = 0; t < T; t++)
            for (int f = h.band_lo[b]; f < h.band_hi[b]; f++) {
                packed.push_back(re[(size_t)f * T + t]);
                packed.push_back(im[(size_t)f * T + t]);
            }
    std::vector<float> tokens;
    double t0 = now_ms();
    run_encoder(m, T, packed, tokens);
    if (st && st->keep)
        st->encode = tokens;
    for (int i = 0; i < h.n_blocks; i++) {
        if (teacher && i > 0)
            tokens = (*teacher)[i - 1];
        run_block(m, i, T, tokens);
        if (st && st->keep)
            st->blocks.push_back(tokens);
    }
    std::vector<float> raw;
    run_decoder(m, T, tokens, raw);
    if (h.causal && m.lookahead > 0) { // read_ahead: frame t reads t + lookahead
        std::vector<float> shifted(raw.size());
        for (int f = 0; f < F; f++)
            for (int t = 0; t < T; t++) {
                const int s = std::min(t + m.lookahead, T - 1);
                std::memcpy(&shifted[((size_t)f * T + t) * 4], &raw[((size_t)f * T + s) * 4], 4 * sizeof(float));
            }
        raw.swap(shifted);
    }
    if (crispasr_env::truthy("CRISPASR_VOICE_ISOLATOR_VERBOSE"))
        fprintf(stderr, "voice_isolator: chunk %d samples, %d frames: network %.0f ms\n", n, T, now_ms() - t0);
    if (st && st->keep)
        st->raw = raw;
    // correction(): spectrum * 2 tanh(mask) + bounded residual
    const float bound = h.correction_bound;
    std::vector<float> dre((size_t)F * T), dim((size_t)F * T);
    for (size_t i = 0; i < (size_t)F * T; i++) {
        const float* r = &raw[i * 4];
        const float mr = 2.0f * std::tanh(r[0]), mi = 2.0f * std::tanh(r[1]);
        const float cr = std::tanh(r[2]), ci = std::tanh(r[3]);
        const float norm = std::max(1.0f, std::sqrt(cr * cr + ci * ci));
        const float mag = bound * std::sqrt(re[i] * re[i] + im[i] * im[i]);
        dre[i] = re[i] * mr - im[i] * mi + cr / norm * mag;
        dim[i] = re[i] * mi + im[i] * mr + ci / norm * mag;
    }
    std::vector<float> y;
    istft(m, dre, dim, T, n, y);
    for (int i = 0; i < n; i++)
        y[i] += audio[i];
    return y;
}

// VoiceIsolatorNetwork.enhance: full windows with linear crossfades; the last
// window ends at the clip's end.
std::vector<float> enhance(voice_isolator_context& m, const std::vector<float>& audio) {
    const Hparams& h = m.hp;
    const int n = (int)audio.size();
    float cs = m.params.chunk_seconds > 0 ? m.params.chunk_seconds : 8.0f;
    float os = m.params.overlap_seconds >= 0 ? m.params.overlap_seconds : 2.0f;
    // Debug/tuning overrides of the render window.
    if (const char* v = crispasr_env::get("CRISPASR_VOICE_ISOLATOR_CHUNK_SECONDS"))
        cs = (float)atof(v);
    if (const char* v = crispasr_env::get("CRISPASR_VOICE_ISOLATOR_OVERLAP_SECONDS"))
        os = (float)atof(v);
    const int chunk = (int)(cs * h.sample_rate);
    const int overlap = std::min((int)(os * h.sample_rate), chunk / 2);
    if (chunk < h.hop)
        throw std::runtime_error("chunk shorter than one hop");
    if (n <= chunk)
        return forward(m, audio.data(), n);
    std::vector<int> starts;
    for (int s = 0; s < n - chunk; s += chunk - overlap)
        starts.push_back(s);
    starts.push_back(n - chunk);
    std::vector<double> out(n, 0.0), weight(n, 0.0);
    std::vector<float> w(chunk);
    for (size_t idx = 0; idx < starts.size(); idx++) {
        const int start = starts[idx], end = start + chunk;
        std::fill(w.begin(), w.end(), 1.0f);
        auto linspace = [&](int at, float a, float b) {
            for (int i = 0; i < overlap; i++)
                w[at + i] = overlap > 1 ? a + (b - a) * (float)i / (float)(overlap - 1) : a;
        };
        if (idx > 0) {
            const int fade_in = starts[idx - 1] + chunk - overlap - start;
            for (int i = 0; i < fade_in; i++)
                w[i] = 0.0f;
            if (overlap > 0)
                linspace(fade_in, 0.0f, 1.0f);
        }
        if (overlap > 0 && end < n)
            linspace(chunk - overlap, 1.0f, 0.0f);
        const std::vector<float> y = forward(m, audio.data() + start, chunk);
        for (int i = 0; i < chunk; i++) {
            out[start + i] += (double)y[i] * w[i];
            weight[start + i] += w[i];
        }
    }
    std::vector<float> res(n);
    for (int i = 0; i < n; i++)
        res[i] = (float)(out[i] / std::max(weight[i], 1e-6));
    return res;
}

const char* kSources[2] = {"voice", "background"};

} // namespace

extern "C" {

voice_isolator_params voice_isolator_default_params(void) {
    voice_isolator_params p{};
    p.n_threads = 0;
    p.use_gpu = false;
    p.gpu_device = 0;
    p.lookahead_frames = -1;
    p.chunk_seconds = 0.0f;
    p.overlap_seconds = -1.0f;
    return p;
}

voice_isolator_context* voice_isolator_init_from_file(const char* path, voice_isolator_params params) {
    std::unique_ptr<voice_isolator_context, decltype(&voice_isolator_free)> m(new voice_isolator_context,
                                                                              voice_isolator_free);
    try {
        gguf_context* meta = core_gguf::open_metadata(path);
        if (!meta)
            return nullptr;
        const bool ok = read_hparams(meta, m->hp);
        core_gguf::free_metadata(meta);
        if (!ok)
            throw std::runtime_error("not a supported voice-isolator GGUF");
        params.n_threads = params.n_threads > 0 ? params.n_threads : std::min(8, core_parallel::resolve_threads(8, 0));
        m->params = params;
        m->lookahead = params.lookahead_frames >= 0 ? params.lookahead_frames : m->hp.lookahead_default;
        if (!m->hp.causal)
            m->lookahead = 0;
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
        const bool loaded = m->backend == m->cpu
                                ? core_gguf::load_weights(path, m->cpu, "voice_isolator", m->weights)
                                : core_gguf::load_weights_fit(path, m->backend, m->cpu, "voice_isolator", m->weights);
        if (!loaded)
            throw std::runtime_error("weight load failed");
        bind_weights(*m);
        for (auto& kv : m->weights.tensors) {
            if (kv.second->type != GGML_TYPE_F32) {
                m->f32_weights = false;
                m->weight_type = ggml_type_name(kv.second->type);
                if (kv.second->type != GGML_TYPE_F16)
                    break;
            }
        }
        const int N = m->hp.n_fft;
        m->window.resize(N);
        for (int i = 0; i < N; i++) // torch.hann_window(N), periodic
            m->window[i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / N));
        fprintf(stderr, "voice_isolator: %s backend=%s threads=%d weights=%s look-ahead=%d frames\n",
                m->hp.causal ? "realtime (causal)" : "offline", ggml_backend_name(m->backend), params.n_threads,
                m->weight_type.c_str(), m->lookahead);
        return m.release();
    } catch (const std::exception& e) {
        fprintf(stderr, "voice_isolator: %s\n", e.what());
        return nullptr;
    }
}

void voice_isolator_free(voice_isolator_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->weights);
    if (ctx->backend && ctx->backend != ctx->cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->cpu)
        ggml_backend_free(ctx->cpu);
    delete ctx;
}

voice_isolator_result* voice_isolator_separate(voice_isolator_context* ctx, const float* pcm, int n_samples,
                                               int in_channels) {
    if (!ctx || !pcm || n_samples <= 0 || in_channels <= 0)
        return nullptr;
    try {
        std::vector<float> mono(n_samples);
        for (int i = 0; i < n_samples; i++) {
            float s = 0.0f;
            for (int c = 0; c < in_channels; c++)
                s += pcm[(size_t)i * in_channels + c];
            mono[i] = s / (float)in_channels;
            if (!std::isfinite(mono[i]))
                throw std::runtime_error("audio contains NaN or infinite samples");
        }
        const std::vector<float> voice = enhance(*ctx, mono);
        auto* r = new voice_isolator_result;
        r->n_sources = 2;
        r->n_channels = 1;
        r->n_samples = n_samples;
        r->sample_rate = ctx->hp.sample_rate;
        r->sources = new float*[2];
        r->source_names = kSources;
        r->sources[0] = new float[n_samples];
        r->sources[1] = new float[n_samples];
        for (int i = 0; i < n_samples; i++) {
            r->sources[0][i] = voice[i];
            r->sources[1][i] = mono[i] - voice[i];
        }
        return r;
    } catch (const std::exception& e) {
        fprintf(stderr, "voice_isolator: %s\n", e.what());
        return nullptr;
    }
}

void voice_isolator_result_free(voice_isolator_result* r) {
    if (!r)
        return;
    for (int s = 0; s < r->n_sources; s++)
        delete[] r->sources[s];
    delete[] r->sources;
    delete r;
}

int voice_isolator_isolate_mono(voice_isolator_context* ctx, const float* pcm, int n_samples, int sample_rate,
                                float* out) {
    if (!ctx || !pcm || !out || n_samples <= 0 || sample_rate <= 0)
        return -1;
    try {
        const int sr = ctx->hp.sample_rate;
        std::vector<float> a =
            sample_rate == sr
                ? std::vector<float>(pcm, pcm + n_samples)
                : core_torchaudio::resample(pcm, n_samples, core_torchaudio::kaiser_best_kernel(sample_rate, sr));
        for (float v : a)
            if (!std::isfinite(v))
                return -1;
        std::vector<float> y = enhance(*ctx, a);
        if (sample_rate != sr)
            y = core_torchaudio::resample(y.data(), (int)y.size(),
                                          core_torchaudio::kaiser_best_kernel(sr, sample_rate));
        for (int i = 0; i < n_samples; i++) // fit_length
            out[i] = i < (int)y.size() ? y[i] : 0.0f;
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "voice_isolator: %s\n", e.what());
        return -2;
    }
}

int voice_isolator_sample_rate(const voice_isolator_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 0;
}
int voice_isolator_n_sources(const voice_isolator_context* ctx) {
    return ctx ? 2 : 0;
}
const char* voice_isolator_source_name(const voice_isolator_context* ctx, int idx) {
    return ctx && idx >= 0 && idx < 2 ? kSources[idx] : nullptr;
}
int voice_isolator_is_causal(const voice_isolator_context* ctx) {
    return ctx && ctx->hp.causal ? 1 : 0;
}
const char* voice_isolator_backend_name(const voice_isolator_context* ctx) {
    return ctx && ctx->backend ? ggml_backend_name(ctx->backend) : nullptr;
}

int voice_isolator_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    voice_isolator_params p = voice_isolator_default_params();
    if (const char* la = crispasr_env::get("CRISPASR_VOICE_ISOLATOR_LOOKAHEAD"))
        p.lookahead_frames = atoi(la);
    voice_isolator_context* m = voice_isolator_init_from_file(model_gguf, p);
    if (!m)
        return 2;
    core_gguf::WeightLoad ref;
    if (!core_gguf::load_weights(ref_gguf, m->cpu, "voice_isolator_ref", ref)) {
        voice_isolator_free(m);
        return 2;
    }
    auto ref_get = [&](const char* name, std::vector<float>& out) {
        auto it = ref.tensors.find(name);
        if (it == ref.tensors.end())
            return false;
        get(it->second, out);
        return true;
    };
    const float gate = m->f32_weights ? 0.9999f : m->weight_type == "f16" ? 0.999f : 0.99f;
    int failures = 0;
    auto compare = [&](const std::string& name, const std::vector<float>& mine, const std::vector<float>& want) {
        if (mine.size() != want.size()) {
            fprintf(stderr, "  %-22s SHAPE MISMATCH mine=%zu ref=%zu\n", name.c_str(), mine.size(), want.size());
            failures++;
            return;
        }
        double dot = 0, a2 = 0, b2 = 0, maxd = 0;
        for (size_t i = 0; i < mine.size(); i++) {
            dot += (double)mine[i] * want[i];
            a2 += (double)mine[i] * mine[i];
            b2 += (double)want[i] * want[i];
            maxd = std::max(maxd, (double)std::fabs(mine[i] - want[i]));
        }
        const double cs = dot / std::max(1e-30, std::sqrt(a2 * b2));
        double e2 = 0;
        for (size_t i = 0; i < mine.size(); i++)
            e2 += ((double)mine[i] - want[i]) * ((double)mine[i] - want[i]);
        const double rel = std::sqrt(e2 / std::max(1e-30, b2));
        const bool ok = std::isfinite(cs) && cs > gate;
        failures += ok ? 0 : 1;
        fprintf(stderr, "  %-22s cos=%.6f |mine|=%.4f |ref|=%.4f rel_l2=%.2e max_abs=%.3e %s\n", name.c_str(), cs,
                std::sqrt(a2), std::sqrt(b2), rel, maxd, ok ? "PASS" : "FAIL");
    };
    int rc = 0;
    try {
        std::vector<float> input;
        if (!ref_get("input_48k", input))
            throw std::runtime_error("reference lacks input_48k");
        std::vector<std::vector<float>> ref_blocks(m->hp.n_blocks);
        for (int i = 0; i < m->hp.n_blocks; i++)
            ref_get(("block" + std::to_string(i)).c_str(), ref_blocks[i]);
        fprintf(stderr, "voice_isolator diff: %zu samples, weights %s, gate cos > %.4f\n", input.size(),
                m->weight_type.c_str(), gate);
        Stages st;
        st.keep = true;
        std::vector<float> voice =
            m->hp.causal ? forward(*m, input.data(), (int)input.size(), &st) : std::vector<float>();
        if (!m->hp.causal) {
            if ((int)input.size() > (int)(8.0f * m->hp.sample_rate))
                throw std::runtime_error("reference input longer than one chunk");
            voice = forward(*m, input.data(), (int)input.size(), &st);
        }
        std::vector<float> want;
        fprintf(stderr, "[chained]\n");
        if (ref_get("stft", want))
            compare("stft", st.stft_ri, want);
        if (ref_get("encode", want))
            compare("encode", st.encode, want);
        for (int i = 0; i < m->hp.n_blocks; i++)
            if (!ref_blocks[i].empty())
                compare("block" + std::to_string(i), st.blocks[i], ref_blocks[i]);
        if (ref_get("decoder_raw", want))
            compare("decoder_raw", st.raw, want);
        if (ref_get("voice", want)) {
            compare("voice", voice, want);
            std::vector<float> bg_m(input.size()), bg_r(input.size());
            for (size_t i = 0; i < input.size(); i++) {
                bg_m[i] = input[i] - voice[i];
                bg_r[i] = input[i] - want[i];
            }
            compare("background", bg_m, bg_r);
            // SI-SDR of mine vs the reference voice
            double dot = 0, rr = 0;
            for (size_t i = 0; i < want.size(); i++) {
                dot += (double)voice[i] * want[i];
                rr += (double)want[i] * want[i];
            }
            const double a = dot / std::max(1e-30, rr);
            double sig = 0, noise = 0;
            for (size_t i = 0; i < want.size(); i++) {
                const double s = a * want[i], e = voice[i] - s;
                sig += s * s;
                noise += e * e;
            }
            fprintf(stderr, "  voice SI-SDR vs reference: %.2f dB\n", 10.0 * std::log10(sig / std::max(1e-30, noise)));
        }
        if (verbosity > 0) {
            fprintf(stderr, "[teacher-forced: block i fed reference block i-1]\n");
            Stages tf;
            tf.keep = true;
            forward(*m, input.data(), (int)input.size(), &tf, &ref_blocks);
            for (int i = 0; i < m->hp.n_blocks; i++)
                if (!ref_blocks[i].empty())
                    compare("block" + std::to_string(i) + "(tf)", tf.blocks[i], ref_blocks[i]);
        }
        rc = failures ? 1 : 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "voice_isolator diff: %s\n", e.what());
        rc = 2;
    }
    core_gguf::free_weights(ref);
    voice_isolator_free(m);
    return rc;
}

} // extern "C"
