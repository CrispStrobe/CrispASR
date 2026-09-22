// src/btc_chords.cpp — BTC chord recognition runtime.
//
// Must match tools/btc_torch_parity.py, which is scored against the PyTorch
// reference at cos >= 0.99999. If this graph changes, change that spec FIRST
// and re-run it. The ten non-obvious details it encodes are in
// docs/music-transcription/BTC_BLUEPRINT.md; the four that bite hardest:
//
//   * Two attention blocks per layer over the SAME input — forward masked
//     causally, backward with that mask TRANSPOSED — concatenated to 256 and
//     projected back to 128. mask=nullptr would silently give full attention.
//   * LayerNorm is gamma*(x-mu)/(std + eps) + beta with eps OUTSIDE the sqrt
//     and UNBIASED std (n-1). ggml_norm does (x-mu)/sqrt(var_biased + eps),
//     so it CANNOT be used here — the norm is built from primitives below.
//   * The FFN is Conv(k=3) -> ReLU -> Conv(k=3) -> ReLU, symmetric (1,1)
//     padding. The TRAILING ReLU is an upstream bug baked into the weights.
//   * Positional encoding is concat([sin, cos]), not interleaved.

#include "btc_chords.h"

#include "btc_chord_vocab.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "core/audio_resample.h"
#include "core/cqt.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ---------------------------------------------------------------------------
// Chord vocabularies (utils/mir_eval_modules.py)
// ---------------------------------------------------------------------------
namespace {

// Chord vocabulary and positional encoding live in btc_chord_vocab.h so they
// can be unit-tested without the weights (tests/test-btc-vocab.cpp). Thin
// aliases here keep the call sites below unchanged.
using btc_vocab::maj_min_name;
using btc_vocab::voca_name;
using btc_vocab::voca_to_maj_min;

} // namespace

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
struct btc_hparams {
    int feature_size = 144;
    int hidden_size = 128;
    int n_layers = 8;
    int n_heads = 4;
    int filter_size = 128;
    int n_chords = 170;
    int timestep = 108;
    float norm_mean = 0.0f;
    float norm_std = 1.0f;
    float eps = 1e-6f;
    int cqt_n_bins = 144;
    int cqt_bins_per_octave = 24;
    int cqt_hop_length = 2048;
    int sample_rate = 22050;
    // Chunk length the reference splits audio into before CQT (mp3.inst_len in
    // run_config.yaml). Also sets the frame duration together with timestep.
    float inst_len_sec = 10.0f;
};

struct btc_attn_block {
    ggml_tensor* q = nullptr;
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
    ggml_tensor* o = nullptr;
    ggml_tensor* ffn0_w = nullptr;
    ggml_tensor* ffn0_b = nullptr;
    ggml_tensor* ffn1_w = nullptr;
    ggml_tensor* ffn1_b = nullptr;
    ggml_tensor* norm_mha_g = nullptr;
    ggml_tensor* norm_mha_b = nullptr;
    ggml_tensor* norm_ffn_g = nullptr;
    ggml_tensor* norm_ffn_b = nullptr;
};

struct btc_layer {
    btc_attn_block fwd;
    btc_attn_block bwd;
    ggml_tensor* proj_w = nullptr;
    ggml_tensor* proj_b = nullptr;
};

struct btc_model {
    btc_hparams hp;
    ggml_tensor* embed_w = nullptr;
    std::vector<btc_layer> layers;
    ggml_tensor* final_g = nullptr;
    ggml_tensor* final_b = nullptr;
    ggml_tensor* out_w = nullptr;
    ggml_tensor* out_b = nullptr;
};

struct btc_chords_context {
    btc_model model;
    btc_chords_params params;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_w = nullptr;
    std::vector<std::string> names; // vocabulary, filled at load
    std::vector<const char*> name_ptrs;

    // Per-stage capture for the parity diff. Off in the normal path.
    bool capture = false;
    std::map<std::string, std::vector<float>> captures;

    // One allocator for the one graph shape, kept for the life of the context.
    // btc_forward_block used to ggml_gallocr_new/_free per call, and it is
    // called once per block from the chunk loop in btc_chords_recognize -- the
    // #132 pattern src/crispasr.cpp:197-203 documents, where per-call
    // disposable allocators fragment memory and cost 2-5x. T is fixed by the
    // chunk geometry, so a single allocator is a direct substitution: a
    // ggml_gallocr keeps its buffer whenever the next graph fits, and the arena
    // grows monotonically (src/hft_transformer.cpp:108-115).
    ggml_gallocr_t alloc_block = nullptr;
};

// Record an intermediate. Graph tensors MUST be ggml_set_output before being
// read back, or gallocr will have recycled the buffer by the time we look.
static void btc_capture(btc_chords_context* ctx, const char* name, ggml_tensor* t) {
    if (!ctx || !ctx->capture || !t)
        return;
    std::vector<float> v((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    ctx->captures[name] = std::move(v);
}

static bool btc_forward_block(btc_chords_context* ctx, const float* feat, int T, std::vector<float>& logits_out);

static bool btc_debug() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_BTC_DEBUG");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

// Reduce the 170-class output to maj/min. Opt-in via CRISPASR_BTC_MAJ_MIN=1;
// the richer vocabulary is the default (see voca_to_maj_min).
static bool btc_maj_min() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_BTC_MAJ_MIN");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

// ---------------------------------------------------------------------------
// Graph pieces
// ---------------------------------------------------------------------------

// BTC LayerNorm: gamma * (x - mu) / (std_unbiased + eps) + beta.
// NOT ggml_norm — that is (x - mu)/sqrt(var_biased + eps), which differs both
// in the eps placement and the denominator (n vs n-1). At width 128 the
// biased/unbiased gap alone is max_abs 1.5e-2 vs 5.3e-7 against the reference.
static ggml_tensor* btc_layer_norm(ggml_context* g, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta,
                                   ggml_tensor* eps) {
    const int n = (int)x->ne[0];
    ggml_tensor* mu = ggml_mean(g, x); // (1, T)
    ggml_tensor* d = ggml_sub(g, x, ggml_repeat(g, mu, x));
    ggml_tensor* var = ggml_mean(g, ggml_sqr(g, d));     // biased
    var = ggml_scale(g, var, (float)n / (float)(n - 1)); // -> unbiased
    // eps arrives as a 1-element INPUT tensor: the graph context is no_alloc,
    // so ggml_new_f32 (which writes data immediately) would abort.
    ggml_tensor* sq = ggml_sqrt(g, var);
    ggml_tensor* sd = ggml_add(g, sq, ggml_repeat(g, eps, sq)); // ggml_add1 is deprecated
    ggml_tensor* y = ggml_div(g, d, ggml_repeat(g, sd, d));
    y = ggml_mul(g, y, ggml_repeat(g, ggml_reshape_2d(g, gamma, n, 1), y));
    return ggml_add(g, y, ggml_repeat(g, ggml_reshape_2d(g, beta, n, 1), y));
}

// Multi-head self-attention with an additive mask. x is (hidden, T).
static ggml_tensor* btc_attention(ggml_context* g, ggml_tensor* x, const btc_attn_block& b, ggml_tensor* mask,
                                  int hidden, int heads, int T) {
    const int hd = hidden / heads;
    ggml_tensor* q = ggml_mul_mat(g, b.q, x);
    ggml_tensor* k = ggml_mul_mat(g, b.k, x);
    ggml_tensor* v = ggml_mul_mat(g, b.v, x);

    q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, heads, T), 0, 2, 1, 3)); // (hd, T, heads)
    k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, heads, T), 0, 2, 1, 3));

    // Upstream scales Q rather than the logits; soft_max_ext's `scale` applies
    // the same factor to the logits, which is equivalent.
    ggml_tensor* scores = ggml_mul_mat(g, k, q); // (T, T, heads)
    scores = ggml_soft_max_ext(g, scores, mask, 1.0f / sqrtf((float)hd), 0.0f);

    v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, heads, T), 1, 2, 0, 3)); // (T, hd, heads)
    ggml_tensor* out = ggml_mul_mat(g, v, scores);                                      // (hd, T, heads)
    out = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)), hidden, T);
    return ggml_mul_mat(g, b.o, out);
}

// Conv(k=3) -> ReLU -> Conv(k=3) -> ReLU, symmetric (1,1) padding.
// ggml_conv_1d wants data as (T, C), so transpose in and out; the attention
// path works in (C, T).
static ggml_tensor* btc_ffn(ggml_context* g, ggml_tensor* x, const btc_attn_block& b) {
    ggml_tensor* t = ggml_cont(g, ggml_transpose(g, x)); // (T, C)
    ggml_tensor* h = ggml_conv_1d(g, b.ffn0_w, t, 1, 1, 1);
    h = ggml_add(g, h, ggml_reshape_2d(g, b.ffn0_b, 1, (int)b.ffn0_b->ne[0]));
    h = ggml_relu(g, h);
    h = ggml_conv_1d(g, b.ffn1_w, h, 1, 1, 1);
    h = ggml_add(g, h, ggml_reshape_2d(g, b.ffn1_b, 1, (int)b.ffn1_b->ne[0]));
    // TRAILING ReLU — upstream's loop guard `i < len(self.layers)` is always
    // true, so ReLU fires after the last conv too. Baked into the weights;
    // omitting it scores cos 0.461 on the block.
    h = ggml_relu(g, h);
    return ggml_cont(g, ggml_transpose(g, h)); // back to (C, T)
}

static ggml_tensor* btc_block(ggml_context* g, ggml_tensor* x, const btc_attn_block& b, ggml_tensor* mask,
                              const btc_hparams& hp, int T, ggml_tensor* eps) {
    ggml_tensor* xn = btc_layer_norm(g, x, b.norm_mha_g, b.norm_mha_b, eps);
    x = ggml_add(g, x, btc_attention(g, xn, b, mask, hp.hidden_size, hp.n_heads, T));
    xn = btc_layer_norm(g, x, b.norm_ffn_g, b.norm_ffn_b, eps);
    return ggml_add(g, x, btc_ffn(g, xn, b));
}

// concat([sin(t*inv), cos(t*inv)]) — two contiguous halves, NOT interleaved.
static void btc_timing_signal(int length, int channels, std::vector<float>& out) {
    btc_vocab::timing_signal(length, channels, out);
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
static bool btc_bind(btc_model& m, const core_gguf::tensor_map& t) {
    auto get = [&](const std::string& n) { return core_gguf::try_get(t, n.c_str()); };
    m.embed_w = get("embedding_proj.weight");
    m.final_g = get("final_norm.gamma");
    m.final_b = get("final_norm.beta");
    m.out_w = get("output.proj.weight");
    m.out_b = get("output.proj.bias");
    if (!m.embed_w || !m.out_w)
        return false;

    m.layers.resize(m.hp.n_layers);
    for (int i = 0; i < m.hp.n_layers; i++) {
        const std::string p = "layers." + std::to_string(i);
        for (int dir = 0; dir < 2; dir++) {
            btc_attn_block& b = dir == 0 ? m.layers[i].fwd : m.layers[i].bwd;
            const std::string d = p + (dir == 0 ? ".fwd" : ".bwd");
            b.q = get(d + ".attn.q.weight");
            b.k = get(d + ".attn.k.weight");
            b.v = get(d + ".attn.v.weight");
            b.o = get(d + ".attn.o.weight");
            b.ffn0_w = get(d + ".ffn.0.weight");
            b.ffn0_b = get(d + ".ffn.0.bias");
            b.ffn1_w = get(d + ".ffn.1.weight");
            b.ffn1_b = get(d + ".ffn.1.bias");
            b.norm_mha_g = get(d + ".norm_mha.gamma");
            b.norm_mha_b = get(d + ".norm_mha.beta");
            b.norm_ffn_g = get(d + ".norm_ffn.gamma");
            b.norm_ffn_b = get(d + ".norm_ffn.beta");
            if (!b.q || !b.ffn0_w || !b.norm_mha_g)
                return false;
        }
        m.layers[i].proj_w = get(p + ".proj.weight");
        m.layers[i].proj_b = get(p + ".proj.bias");
        if (!m.layers[i].proj_w)
            return false;
    }
    return true;
}

btc_chords_params btc_chords_default_params(void) {
    btc_chords_params p;
    p.n_threads = 0;
    p.use_gpu = true;
    p.gpu_device = 0;
    return p;
}

btc_chords_context* btc_chords_init_from_file(const char* model_path, btc_chords_params params) {
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "btc: failed to read metadata from %s\n", model_path);
        return nullptr;
    }
    auto* ctx = new btc_chords_context();
    ctx->params = params;
    btc_hparams& hp = ctx->model.hp;
    hp.feature_size = core_gguf::kv_u32(meta, "btc.feature_size", 144);
    hp.hidden_size = core_gguf::kv_u32(meta, "btc.hidden_size", 128);
    hp.n_layers = core_gguf::kv_u32(meta, "btc.n_layers", 8);
    hp.n_heads = core_gguf::kv_u32(meta, "btc.n_heads", 4);
    hp.filter_size = core_gguf::kv_u32(meta, "btc.filter_size", 128);
    hp.n_chords = core_gguf::kv_u32(meta, "btc.n_chords", 170);
    hp.timestep = core_gguf::kv_u32(meta, "btc.timestep", 108);
    hp.norm_mean = core_gguf::kv_f32(meta, "btc.norm_mean", 0.0f);
    hp.norm_std = core_gguf::kv_f32(meta, "btc.norm_std", 1.0f);
    hp.eps = core_gguf::kv_f32(meta, "btc.layer_norm_eps", 1e-6f);
    hp.cqt_n_bins = core_gguf::kv_u32(meta, "btc.cqt_n_bins", 144);
    hp.cqt_bins_per_octave = core_gguf::kv_u32(meta, "btc.cqt_bins_per_octave", 24);
    hp.cqt_hop_length = core_gguf::kv_u32(meta, "btc.cqt_hop_length", 2048);
    hp.sample_rate = core_gguf::kv_u32(meta, "btc.sample_rate", 22050);
    hp.inst_len_sec = core_gguf::kv_f32(meta, "btc.inst_len_sec", 10.0f);
    gguf_free(meta);

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : nullptr;
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "btc", wl)) {
        fprintf(stderr, "btc: failed to load weights from %s\n", model_path);
        btc_chords_free(ctx);
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    if (!btc_bind(ctx->model, wl.tensors)) {
        fprintf(stderr, "btc: missing tensors in %s\n", model_path);
        btc_chords_free(ctx);
        return nullptr;
    }

    const bool reduce = hp.n_chords > 25 && btc_maj_min();
    const int n_out = reduce ? 25 : hp.n_chords;
    ctx->names.reserve(n_out);
    for (int i = 0; i < n_out; i++)
        ctx->names.push_back(n_out == 25 ? maj_min_name(i) : voca_name(i));
    ctx->name_ptrs.reserve(n_out);
    for (auto& s : ctx->names)
        ctx->name_ptrs.push_back(s.c_str());

    fprintf(stderr, "btc: %d chords%s, %d layers, backend %s\n", hp.n_chords, reduce ? " (reduced to 25 maj/min)" : "",
            hp.n_layers, ggml_backend_name(ctx->backend));
    return ctx;
}

void btc_chords_free(btc_chords_context* ctx) {
    if (!ctx)
        return;
    if (ctx->alloc_block)
        ggml_gallocr_free(ctx->alloc_block);
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int btc_chords_vocab_size(const btc_chords_context* ctx) {
    return ctx ? (int)ctx->names.size() : 0;
}

const char* btc_chords_label_name(const btc_chords_context* ctx, int label) {
    if (!ctx || label < 0 || label >= (int)ctx->names.size())
        return "N";
    return ctx->names[label].c_str();
}

// ---------------------------------------------------------------------------
// Forward: one block of `timestep` frames
// ---------------------------------------------------------------------------
static bool btc_forward_block(btc_chords_context* ctx, const float* feat, int T, std::vector<float>& logits_out) {
    const btc_hparams& hp = ctx->model.hp;
    btc_model& m = ctx->model;

    const size_t n_nodes = 8192;
    ggml_init_params gp = {ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr,
                           true};
    ggml_context* g = ggml_init(gp);
    if (!g)
        return false;

    ggml_tensor* X = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.feature_size, T);
    ggml_set_input(X);
    ggml_tensor* POS = ggml_new_tensor_2d(g, GGML_TYPE_F32, hp.hidden_size, T);
    ggml_set_input(POS);
    // Two masks: causal for the forward blocks, its transpose for the backward
    // ones. F16 is what soft_max_ext expects for the mask.
    ggml_tensor* MF = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T);
    ggml_tensor* MB = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T);
    ggml_set_input(MF);
    ggml_set_input(MB);
    ggml_tensor* EPS = ggml_new_tensor_1d(g, GGML_TYPE_F32, 1);
    ggml_set_input(EPS);

    std::vector<std::pair<std::string, ggml_tensor*>> taps;
    const bool cap = ctx->capture;

    ggml_tensor* x = ggml_mul_mat(g, m.embed_w, X); // (hidden, T), no bias
    x = ggml_add(g, x, POS);
    if (cap) {
        ggml_set_output(x);
        taps.emplace_back("embed_posenc", x);
    }

    for (int i = 0; i < hp.n_layers; i++) {
        ggml_tensor* f = btc_block(g, x, m.layers[i].fwd, MF, hp, T, EPS);
        ggml_tensor* b = btc_block(g, x, m.layers[i].bwd, MB, hp, T, EPS);
        if (cap && i == 0) {
            ggml_set_output(f);
            ggml_set_output(b);
            taps.emplace_back("layer0_fwd", f);
            taps.emplace_back("layer0_bwd", b);
        }
        ggml_tensor* cat = ggml_concat(g, f, b, 0); // (256, T)
        x = ggml_mul_mat(g, m.layers[i].proj_w, cat);
        x = ggml_add(g, x, ggml_reshape_2d(g, m.layers[i].proj_b, hp.hidden_size, 1));
        if (cap) {
            ggml_set_output(x);
            taps.emplace_back("layer" + std::to_string(i) + "_out", x);
        }
    }

    if (m.final_g)
        x = btc_layer_norm(g, x, m.final_g, m.final_b, EPS);
    if (cap) {
        ggml_set_output(x);
        taps.emplace_back("final_norm", x);
    }
    ggml_tensor* out = ggml_mul_mat(g, m.out_w, x);
    out = ggml_add(g, out, ggml_reshape_2d(g, m.out_b, hp.n_chords, 1));
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_nodes, false);
    ggml_build_forward_expand(gf, out);
    for (auto& kv : taps)
        ggml_build_forward_expand(gf, kv.second);
    if (!ctx->alloc_block)
        ctx->alloc_block = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->alloc_block || !ggml_gallocr_alloc_graph(ctx->alloc_block, gf)) {
        fprintf(stderr, "btc: graph alloc failed\n");
        ggml_free(g);
        return false;
    }
    // Every ggml_set_input tensor below is re-set on every call, including the
    // masks and the timing signal, which are constant for a given T: gallocr
    // may hand an input's slot to a later intermediate once the allocator is
    // reused across calls (playbook §6.6). They already were; this comment is
    // the reason they must stay that way.

    // Normalise with the checkpoint's own scalar stats.
    std::vector<float> norm((size_t)T * hp.feature_size);
    for (size_t i = 0; i < norm.size(); i++)
        norm[i] = (feat[i] - hp.norm_mean) / hp.norm_std;
    ggml_backend_tensor_set(X, norm.data(), 0, norm.size() * sizeof(float));

    std::vector<float> pos;
    btc_timing_signal(T, hp.hidden_size, pos);
    ggml_backend_tensor_set(POS, pos.data(), 0, pos.size() * sizeof(float));

    std::vector<ggml_fp16_t> mf((size_t)T * T), mb((size_t)T * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++) {
            // mask[q][k] laid out with k fastest. Causal: forbid k > q.
            mf[(size_t)q * T + k] = (k > q) ? ninf : zero;
            mb[(size_t)q * T + k] = (k < q) ? ninf : zero;
        }
    ggml_backend_tensor_set(MF, mf.data(), 0, mf.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(MB, mb.data(), 0, mb.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(EPS, &hp.eps, 0, sizeof(float));

    ggml_backend_graph_compute(ctx->backend, gf);

    for (auto& kv : taps)
        btc_capture(ctx, kv.first.c_str(), kv.second);
    btc_capture(ctx, "logits", out);

    logits_out.assign((size_t)ggml_nelements(out), 0.0f);
    ggml_backend_tensor_get(out, logits_out.data(), 0, logits_out.size() * sizeof(float));

    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
btc_chords_result* btc_chords_recognize(btc_chords_context* ctx, const float* pcm, int n_samples, int sample_rate) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;
    const btc_hparams& hp = ctx->model.hp;

    std::vector<float> mono;
    const float* src = pcm;
    int n = n_samples;
    if (sample_rate != hp.sample_rate) {
        // librosa.load(sr=22050) is what the model was trained through, and
        // resample_polyphase's default num_zeros=14 matches its kaiser_fast.
        mono = core_audio::resample_polyphase(pcm, n_samples, sample_rate, hp.sample_rate);
        src = mono.data();
        n = (int)mono.size();
    }

    core_cqt::Params cp;
    cp.sample_rate = hp.sample_rate;
    cp.n_bins = hp.cqt_n_bins;
    cp.bins_per_octave = hp.cqt_bins_per_octave;
    cp.hop_length = hp.cqt_hop_length;

    // CHUNKED CQT — this reproduces the reference PIPELINE, not just the
    // reference transform. utils/mir_eval_modules.audio_file_to_features CQTs
    // each `inst_len` (10 s) segment INDEPENDENTLY and concatenates the
    // results; because librosa centres each call, every chunk carries its own
    // edge padding, so a continuous transform of the whole signal is NOT the
    // same thing. Measured on the upstream test clip (257 s): chunked gives
    // 2778 frames, continuous 2770, and our continuous output scored cos 0.8815
    // against the reference pipeline while scoring 0.9993 against a continuous
    // librosa CQT. In other words the transform was right and the pipeline was
    // wrong -- and the per-stage diff harness could not see it, because its
    // reference dump replays `input_feat` by design.
    const int chunk = (int)((double)hp.inst_len_sec * (double)hp.sample_rate);
    const auto kernels = core_cqt::build_kernels(cp);
    std::vector<float> mag;
    int n_frames = 0;
    if (chunk > 0) {
        int cur = 0;
        std::vector<float> part;
        while (n > cur + chunk) {
            const int got = core_cqt::magnitude(cp, kernels, src + cur, chunk, part);
            mag.insert(mag.end(), part.begin(), part.end());
            n_frames += got;
            cur += chunk;
        }
        // Trailing partial chunk -- the reference always emits this final call,
        // even when the remainder is empty.
        const int got = core_cqt::magnitude(cp, kernels, src + cur, n - cur, part);
        mag.insert(mag.end(), part.begin(), part.end());
        n_frames += got;
    } else {
        n_frames = core_cqt::magnitude(cp, kernels, src, n, mag);
    }
    if (n_frames <= 0)
        return nullptr;

    // BTC trains on log(|CQT| + 1e-6) — epsilon ADDED, not a floor
    // (audio_file_to_features: np.log(np.abs(feature) + 1e-6)).
    for (auto& v : mag)
        v = std::log(v + 1e-6f);

    if (const char* dp = getenv("CRISPASR_BTC_DUMP_FEAT")) {
        // Front-end comparison hook: the per-stage diff replays the reference's
        // own input_feat by design, so it cannot catch a CQT mismatch. This
        // dumps what the front end actually produced, to be scored against
        // librosa directly.
        if (FILE* f = fopen(dp, "wb")) {
            fwrite(mag.data(), sizeof(float), mag.size(), f);
            fclose(f);
            fprintf(stderr, "btc: dumped %d x %d features to %s\n", n_frames, hp.feature_size, dp);
        }
    }

    const bool reduce = hp.n_chords > 25 && btc_maj_min();
    // Frame duration is inst_len/timestep, NOT hop/sample_rate. The reference
    // derives it from the chunk geometry (feature_per_second = inst_len /
    // timestep in audio_file_to_features), and the two differ by 0.31 % --
    // 2048/22050 = 0.0928798 s vs 10/108 = 0.0925926 s. That is 0.79 s of
    // accumulated drift over a 4-minute song, i.e. every chord boundary lands
    // progressively late.
    const double frame_ms =
        1000.0 * btc_vocab::frame_seconds(hp.inst_len_sec, hp.timestep, hp.cqt_hop_length, hp.sample_rate);

    std::vector<int> labels;
    std::vector<float> confs;
    labels.reserve(n_frames);
    confs.reserve(n_frames);

    // Fixed `timestep` blocks, matching test.py's
    // feature[:, 108*t : 108*(t+1), :] — NOT a sliding window; the bias mask is
    // built for exactly that length. A short tail block is run at its own
    // length, which the mask construction handles.
    for (int start = 0; start < n_frames; start += hp.timestep) {
        const int T = std::min(hp.timestep, n_frames - start);
        std::vector<float> logits;
        if (!btc_forward_block(ctx, mag.data() + (size_t)start * hp.feature_size, T, logits))
            return nullptr;
        for (int t = 0; t < T; t++) {
            const float* row = logits.data() + (size_t)t * hp.n_chords;
            int best = 0;
            float mx = row[0];
            for (int c = 1; c < hp.n_chords; c++)
                if (row[c] > mx) {
                    mx = row[c];
                    best = c;
                }
            double sum = 0.0;
            for (int c = 0; c < hp.n_chords; c++)
                sum += std::exp((double)row[c] - mx);
            labels.push_back(reduce ? voca_to_maj_min(best) : best);
            confs.push_back((float)(1.0 / sum));
        }
    }

    // Merge runs of identical labels into spans.
    std::vector<btc_chord_span> spans;
    for (size_t i = 0; i < labels.size();) {
        size_t j = i;
        double conf = 0.0;
        while (j < labels.size() && labels[j] == labels[i]) {
            conf += confs[j];
            j++;
        }
        btc_chord_span s;
        s.start_ms = (double)i * frame_ms;
        s.end_ms = (double)j * frame_ms;
        s.label = labels[i];
        s.confidence = (float)(conf / (double)(j - i));
        spans.push_back(s);
        i = j;
    }

    auto* r = new btc_chords_result();
    r->n_spans = (int)spans.size();
    r->spans = new btc_chord_span[spans.size()];
    std::copy(spans.begin(), spans.end(), r->spans);
    r->n_chords = (int)ctx->names.size();
    r->chord_names = ctx->name_ptrs.data();
    if (btc_debug())
        fprintf(stderr, "btc: %d frames -> %d spans\n", n_frames, r->n_spans);
    return r;
}

void btc_chords_result_free(btc_chords_result* r) {
    if (!r)
        return;
    delete[] r->spans;
    delete r;
}

// ---------------------------------------------------------------------------
// Parity diff against tools/btc_torch_parity.py's reference dump
// ---------------------------------------------------------------------------
namespace {

double btc_cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0)
        return (na == 0 && nb == 0) ? 1.0 : 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double btc_max_abs(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

bool btc_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    const int64_t n = ggml_nelements(it->second);
    out.resize((size_t)n);
    ggml_backend_tensor_get(it->second, out.data(), 0, (size_t)n * sizeof(float));
    return true;
}

} // namespace

int btc_chords_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    btc_chords_params p = btc_chords_default_params();
    p.use_gpu = false; // structural diff on CPU first
    btc_chords_context* ctx = btc_chords_init_from_file(model_gguf, p);
    if (!ctx) {
        fprintf(stderr, "btc_diff: failed to load %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "btc_ref", rw)) {
        fprintf(stderr, "btc_diff: failed to load reference %s\n", ref_gguf);
        btc_chords_free(ctx);
        return 2;
    }

    // Replay the reference's OWN features so a CQT front-end difference can
    // never masquerade as a model parity failure.
    std::vector<float> feat;
    if (!btc_ref_get(rw, "input_feat", feat)) {
        fprintf(stderr, "btc_diff: reference has no input_feat — re-dump with the updated spec\n");
        core_gguf::free_weights(rw);
        btc_chords_free(ctx);
        return 2;
    }
    const btc_hparams& hp = ctx->model.hp;
    const int T = (int)(feat.size() / (size_t)hp.feature_size);

    ctx->capture = true;
    std::vector<float> logits;
    if (!btc_forward_block(ctx, feat.data(), T, logits)) {
        fprintf(stderr, "btc_diff: forward failed\n");
        core_gguf::free_weights(rw);
        btc_chords_free(ctx);
        return 2;
    }

    const double COS_MIN = 0.999;
    int n_fail = 0, n_run = 0;
    std::string first_fail;

    auto report = [&](const char* stage) {
        std::vector<float> ref;
        if (!btc_ref_get(rw, stage, ref))
            return;
        auto it = ctx->captures.find(stage);
        if (it == ctx->captures.end()) {
            fprintf(stderr, "  %-16s MISSING (no C++ capture)\n", stage);
            return;
        }
        const std::vector<float>& mine = it->second;
        const int64_t n = (int64_t)std::min(mine.size(), ref.size());
        const double cos = btc_cosine(mine.data(), ref.data(), n);
        const bool ok = cos >= COS_MIN && mine.size() == ref.size();
        n_run++;
        if (!ok) {
            n_fail++;
            if (first_fail.empty())
                first_fail = stage;
        }
        // Print both magnitudes: an outlier on either side means "same name,
        // wrong data" (a harness bug), not a runtime bug.
        double na = 0, nb = 0;
        for (int64_t i = 0; i < n; i++) {
            na += (double)mine[i] * mine[i];
            nb += (double)ref[i] * ref[i];
        }
        fprintf(stderr, "  %-16s %s cos=%.6f max_abs=%.3e  mine=%zu ref=%zu  |mine|=%.4f |ref|=%.4f%s\n", stage,
                ok ? "PASS" : "FAIL", cos, btc_max_abs(mine.data(), ref.data(), n), mine.size(), ref.size(),
                std::sqrt(na), std::sqrt(nb), mine.size() == ref.size() ? "" : "  *** SHAPE MISMATCH ***");
        (void)verbosity;
    };

    fprintf(stderr, "\n=== btc per-stage parity (cos_min >= %.3f, T=%d) ===\n", COS_MIN, T);
    report("embed_posenc");
    report("layer0_fwd");
    report("layer0_bwd");
    for (int i = 0; i < hp.n_layers; i++)
        report(("layer" + std::to_string(i) + "_out").c_str());
    report("final_norm");
    report("logits");

    fprintf(stderr, "\n%d/%d stages passed", n_run - n_fail, n_run);
    if (!first_fail.empty())
        fprintf(stderr, " — FIRST DIVERGENCE: %s", first_fail.c_str());
    fprintf(stderr, "\n");

    core_gguf::free_weights(rw);
    btc_chords_free(ctx);
    return n_fail > 0 ? 1 : 0;
}
