// titanet.cpp — TitaNet-Large speaker embedding extraction (pure CPU).
//
// See titanet.h for the full architecture description. This implementation
// reads all weights into host F32 once at init time and runs the forward
// pass entirely on the CPU. The model is 23M params / ~45 MB — small enough
// that the single-shot embedding extraction (one per speaker cluster in the
// diarization pipeline) finishes in well under a second.
//
// Preprocessing matches NeMo's AudioToMelSpectrogramPreprocessor:
//   - Hann window (embedded in GGUF), n_fft=512, hop=160, win=400
//   - Mel filterbank from GGUF (80 bins, 257 FFT bins)
//   - log(max(power, 1e-10))
//   - Per-feature normalization: (x - mean) / (std + 1e-5)

#include "titanet.h"
#include "core/gguf_loader.h"
#if defined(CRISPASR_USE_ONNXRUNTIME)
#include "core/ort_session.h"
#endif
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/crispasr_env.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

// The TitaNet speaker-embedding forward is entirely hand-rolled CPU scalar
// math. The pointwise convs (1×1 matmuls) + the ASP TDNN/attention matmuls are
// the dominant cost. PLAN: Accelerate cblas_sgemm; set TITANET_FORCE_SCALAR=1
// to validate scalar == GEMM or run on non-Apple.
static bool titanet_use_scalar() {
#if defined(HAVE_ACCELERATE)
    static const bool force_scalar = crispasr_env::get("CRISPASR_TITANET_FORCE_SCALAR") != nullptr;
    return force_scalar;
#else
    return true;
#endif
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `TITANET_BENCH=1` for per-stage timings.
// ===========================================================================

static bool titanet_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_TITANET_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct titanet_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit titanet_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~titanet_bench_stage() {
        if (!titanet_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  titanet_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ============================================================================
// Helpers
// ============================================================================

// NeMo TitaNet uses eps=0.001 for encoder BN, eps=1e-5 for decoder BN
static constexpr float kBnEpsEncoder = 1e-3f;
static constexpr float kBnEpsDecoder = 1e-5f;

static std::vector<float> read_f32(ggml_tensor* t) {
    if (!t)
        return {};
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
    } else {
        const size_t raw = ggml_nbytes(t);
        std::vector<uint8_t> tmp(raw);
        ggml_backend_tensor_get(t, tmp.data(), 0, raw);
        const ggml_type_traits* tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float)
            tt->to_float(tmp.data(), out.data(), (int64_t)n);
    }
    return out;
}

struct BNFolded {
    std::vector<float> gamma; // (C,)
    std::vector<float> beta;  // (C,)
};

static BNFolded fold_bn(ggml_tensor* m, ggml_tensor* v, ggml_tensor* w, ggml_tensor* b, int C,
                        float eps = kBnEpsEncoder) {
    BNFolded out;
    if (!m || !v) {
        out.gamma.assign((size_t)C, 1.0f);
        out.beta.assign((size_t)C, 0.0f);
        return out;
    }
    auto mean = read_f32(m);
    auto var = read_f32(v);
    std::vector<float> weight, bias;
    if (w)
        weight = read_f32(w);
    if (b)
        bias = read_f32(b);
    out.gamma.resize((size_t)C);
    out.beta.resize((size_t)C);
    for (int c = 0; c < C; c++) {
        float inv_std = 1.0f / std::sqrt(var[c] + eps);
        float g = w ? weight[c] * inv_std : inv_std;
        float bv = (b ? bias[c] : 0.0f) - mean[c] * g;
        out.gamma[c] = g;
        out.beta[c] = bv;
    }
    return out;
}

// ============================================================================
// Model structure — cached host weights
// ============================================================================

struct SubBlockCache {
    std::vector<float> dw_w; // depthwise: (C, K) — one kernel per channel
    int dw_C = 0;
    int dw_K = 0;
    BNFolded dw_bn;
    std::vector<float> pw_w; // pointwise: (C_out, C_in)
    int pw_in = 0;
    int pw_out = 0;
    BNFolded pw_bn;
};

struct SECache {
    std::vector<float> fc1_w; // (se_ch, C)
    int se_ch = 0;
    int C = 0;
    std::vector<float> fc2_w; // (C, se_ch)
};

struct ResCache {
    std::vector<float> conv_w; // (C_out, C_in)
    int in_ch = 0;
    int out_ch = 0;
    BNFolded bn;
};

struct BlockCache {
    std::vector<SubBlockCache> subs;
    SECache se;
    bool has_residual = false;
    ResCache res;
};

struct ASPCache {
    std::vector<float> tdnn_w; // (128, 9216)
    std::vector<float> tdnn_b; // (128,)
    BNFolded tdnn_bn;          // (128,)
    std::vector<float> conv_w; // (3072, 128)
    std::vector<float> conv_b; // (3072,)
    BNFolded pool_bn;          // (6144,)
    std::vector<float> fc_w;   // (192, 6144)
    std::vector<float> fc_b;   // (192,)
};

struct titanet_model_cache {
    bool initialised = false;
    int n_mels = 80;
    int n_fft = 512;
    int emb_dim = 192;
    int channels = 1024;
    int epilog_channels = 3072;
    int n_blocks = 5;
    int se_channels = 128;
    std::vector<int> block_repeats;
    std::vector<int> block_kernels;

    // Embedded preprocessor
    std::vector<float> mel_fb;      // (n_mels, n_fft/2+1)
    std::vector<float> hann_window; // (400,) — GGUF has symmetric, we override with periodic

    std::vector<BlockCache> blocks;
    ASPCache asp;
};

// ggml-path weight handles — folded/preprocessed weights uploaded once at
// init into a CPU backend buffer (layouts chosen so conv_1d/conv_1d_dw/
// mul_mat consume them without transposes; see titanet_upload_ggml_weights).
struct SubBlockGgml {
    ggml_tensor* dw = nullptr;   // (K, 1, C)      — ggml_conv_1d_dw kernel
    ggml_tensor* pw = nullptr;   // (1, Cin, Cout) — ggml_conv_1d k=1 kernel
    ggml_tensor* bn_g = nullptr; // (Cout,)
    ggml_tensor* bn_b = nullptr; // (Cout,)
};
struct BlockGgml {
    std::vector<SubBlockGgml> subs;
    ggml_tensor* se_fc1 = nullptr;   // (C, se_ch)
    ggml_tensor* se_fc2 = nullptr;   // (se_ch, C)
    ggml_tensor* res_pw = nullptr;   // (1, Cin, Cout)
    ggml_tensor* res_bn_g = nullptr; // (Cout,)
    ggml_tensor* res_bn_b = nullptr;
};
struct TitanetGgml {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<BlockGgml> blocks;
    ggml_tensor* tdnn = nullptr; // (1, 9216, 128)
    ggml_tensor* tdnn_b = nullptr;
    ggml_tensor* tdnn_bn_g = nullptr;
    ggml_tensor* tdnn_bn_b = nullptr;
    ggml_tensor* aconv = nullptr; // (1, 128, 3072)
    ggml_tensor* aconv_b = nullptr;
    ggml_tensor* pool_bn_g = nullptr; // (6144,)
    ggml_tensor* pool_bn_b = nullptr;
    ggml_tensor* fc = nullptr; // (6144, 192)
    ggml_tensor* fc_b = nullptr;
};

struct titanet_context {
    titanet_model_cache cache;
    int n_threads = 4;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* weight_ctx = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_t backend_cpu = nullptr; // ggml-path compute backend
    TitanetGgml g;

#if defined(CRISPASR_USE_ONNXRUNTIME)
    Ort::Session* ort_session = nullptr; // onnxruntime path (model_path ends in ".onnx")
#endif
};

// ============================================================================
// Init — load GGUF, pre-fold all BN params
// ============================================================================

static ggml_tensor* G(const std::map<std::string, ggml_tensor*>& ts, const std::string& name) {
    auto it = ts.find(name);
    return it != ts.end() ? it->second : nullptr;
}

static void init_cache(titanet_context* ctx) {
    auto& c = ctx->cache;
    auto& ts = ctx->tensors;

    // Load preprocessor
    {
        auto* fb = G(ts, "mel_filterbank");
        if (fb)
            c.mel_fb = read_f32(fb);
        // NeMo uses periodic Hann window (torch.hann_window(N, periodic=True))
        // which is hann(i) = 0.5 * (1 - cos(2*pi*i / N)) for i=0..N-1.
        // The GGUF has the SYMMETRIC window (2*pi*i / (N-1)) from the
        // preprocessor checkpoint. We override with the periodic version.
        int win_len = 400; // 25ms * 16000 Hz
        c.hann_window.resize(win_len);
        for (int i = 0; i < win_len; i++)
            c.hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / win_len));
    }

    // Encoder blocks
    c.blocks.resize(c.n_blocks);
    for (int bi = 0; bi < c.n_blocks; bi++) {
        auto& blk = c.blocks[bi];
        std::string bp = "enc.b" + std::to_string(bi);
        int R = c.block_repeats[bi];
        int K = c.block_kernels[bi];
        int C_in = (bi == 0) ? c.n_mels : c.channels;
        int C_out = (bi == c.n_blocks - 1) ? c.epilog_channels : c.channels;

        blk.subs.resize(R);
        for (int si = 0; si < R; si++) {
            auto& sub = blk.subs[si];
            std::string sp = bp + ".s" + std::to_string(si);

            // Depthwise: weight shape in GGUF is (C, K) after squeeze
            auto* dw_t = G(ts, sp + ".dw.w");
            if (dw_t) {
                sub.dw_w = read_f32(dw_t);
                // GGUF stores [C, 1, K] squeezed to [C, K] or kept as [C, 1, K]
                // The converter squeezes k=1 but not k>1
                sub.dw_C = (si == 0) ? C_in : c.channels;
                sub.dw_K = K;
                if (dw_t->ne[0] > 1 && ggml_n_dims(dw_t) >= 2) {
                    // [C, 1, K] in NeMo → GGUF may store as (K, 1, C) due to gguf col-major
                    // Actually the converter didn't squeeze since K > 1
                    // The shape in NeMo is (C_out, 1, K) → GGUF stores row-major
                    sub.dw_K = (int)dw_t->ne[0];
                    sub.dw_C = (int)dw_t->ne[ggml_n_dims(dw_t) - 1];
                }
            }

            sub.dw_bn = fold_bn(G(ts, sp + ".bn.m"), G(ts, sp + ".bn.v"), G(ts, sp + ".bn.w"), G(ts, sp + ".bn.b"),
                                (si == 0 && bi == c.n_blocks - 1) ? C_out : c.channels);

            // Pointwise: weight shape (C_out, C_in) after squeeze
            auto* pw_t = G(ts, sp + ".pw.w");
            if (pw_t) {
                sub.pw_w = read_f32(pw_t);
                sub.pw_in = (si == 0) ? C_in : c.channels;
                sub.pw_out = (bi == c.n_blocks - 1) ? C_out : c.channels;
            }

            sub.pw_bn = fold_bn(nullptr, nullptr, nullptr, nullptr, 0); // placeholder
        }

        // SE block
        auto* se1 = G(ts, bp + ".se.fc1.w");
        auto* se2 = G(ts, bp + ".se.fc2.w");
        if (se1 && se2) {
            blk.se.fc1_w = read_f32(se1);
            blk.se.fc2_w = read_f32(se2);
            blk.se.se_ch = c.se_channels;
            blk.se.C = C_out;
            // Adjust for blocks where epilog has different SE
            if (bi == c.n_blocks - 1) {
                blk.se.se_ch = (int)se1->ne[ggml_n_dims(se1) - 1];
                if (ggml_n_dims(se1) == 2)
                    blk.se.se_ch = (int)se1->ne[1];
            }
        }

        // Residual connection (only mega blocks have this)
        auto* res_conv = G(ts, bp + ".res.conv.w");
        if (res_conv) {
            blk.has_residual = true;
            blk.res.conv_w = read_f32(res_conv);
            blk.res.in_ch = C_in;
            blk.res.out_ch = C_out;
            blk.res.bn = fold_bn(G(ts, bp + ".res.bn.m"), G(ts, bp + ".res.bn.v"), G(ts, bp + ".res.bn.w"),
                                 G(ts, bp + ".res.bn.b"), C_out);
        }
    }

    // ASP decoder
    {
        auto& a = c.asp;
        a.tdnn_w = read_f32(G(ts, "dec.asp.tdnn.w"));
        a.tdnn_b = read_f32(G(ts, "dec.asp.tdnn.b"));
        a.tdnn_bn = fold_bn(G(ts, "dec.asp.bn.m"), G(ts, "dec.asp.bn.v"), G(ts, "dec.asp.bn.w"), G(ts, "dec.asp.bn.b"),
                            128, kBnEpsDecoder);
        a.conv_w = read_f32(G(ts, "dec.asp.conv.w"));
        a.conv_b = read_f32(G(ts, "dec.asp.conv.b"));
        a.pool_bn = fold_bn(G(ts, "dec.pool_bn.m"), G(ts, "dec.pool_bn.v"), G(ts, "dec.pool_bn.w"),
                            G(ts, "dec.pool_bn.b"), 6144, kBnEpsDecoder);
        a.fc_w = read_f32(G(ts, "dec.fc.w"));
        a.fc_b = read_f32(G(ts, "dec.fc.b"));
    }

    c.initialised = true;
}

// ===========================================================================
// ggml forward (default) — the whole encoder + ASP as one CPU-backend graph
// (threaded SIMD matmuls). CRISPASR_TITANET_LEGACY=1 selects the original
// scalar/Accelerate path (A/B ground truth; the ggml path is validated
// embedding-for-embedding against it).
// ===========================================================================

// Inverse-default regime (see crispasr-crispembed-dev.md): the ggml path is
// embedding-identical (cos=1.000000 vs legacy) and ~10× faster than the
// scalar loops on non-Apple CPUs, but on Apple the legacy path's Accelerate
// GEMMs run on the AMX (~5× faster than ggml's F32 NEON matmul: 0.7 s vs
// 3.4 s per embed on M1). So: default ggml WITHOUT Accelerate, default
// legacy WITH it. CRISPASR_TITANET_GGML=1 / CRISPASR_TITANET_LEGACY=1
// force either way.
static bool titanet_use_legacy() {
    static const bool v = [] {
        if (const char* e = std::getenv("CRISPASR_TITANET_LEGACY"); e && *e && *e != '0')
            return true;
        if (const char* e = std::getenv("CRISPASR_TITANET_GGML"); e && *e && *e != '0')
            return false;
#if defined(HAVE_ACCELERATE)
        return true;
#else
        return false;
#endif
    }();
    return v;
}

// TITANET_DUMP=<path>: append each L2-normalized embedding as raw F32 —
// A/B comparison between the ggml and legacy paths.
static void titanet_dump_emb(const float* emb, int dim) {
    const char* p = crispasr_env::get("CRISPASR_TITANET_DUMP");
    if (!p)
        return;
    if (FILE* f = fopen(p, "ab")) {
        fwrite(emb, sizeof(float), (size_t)dim, f);
        fclose(f);
    }
}

// Upload the pre-folded host cache into ggml tensors on the CPU backend.
// Layout notes (host flat data → ggml ne, no reordering needed):
//   dw   (C,K) row-major        → ne (K, 1, C)      for ggml_conv_1d_dw
//   pw   (Cout,Cin) row-major   → ne (1, Cin, Cout) for ggml_conv_1d k=1
//   fc   (Rows,Cols) row-major  → ne (Cols, Rows)   for ggml_mul_mat direct
static bool titanet_upload_ggml_weights(titanet_context* ctx) {
    auto& c = ctx->cache;
    auto& g = ctx->g;

    size_t n_tensors = 16;
    for (auto& blk : c.blocks)
        n_tensors += blk.subs.size() * 4 + 5;
    ggml_init_params ip = {n_tensors * ggml_tensor_overhead(), nullptr, true};
    g.ctx = ggml_init(ip);
    if (!g.ctx)
        return false;

    std::vector<std::pair<ggml_tensor*, const float*>> uploads;
    auto mk1 = [&](const std::vector<float>& v) {
        if (v.empty())
            return (ggml_tensor*)nullptr;
        ggml_tensor* t = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, (int64_t)v.size());
        uploads.emplace_back(t, v.data());
        return t;
    };
    auto mk = [&](const std::vector<float>& v, int64_t ne0, int64_t ne1, int64_t ne2) {
        if (v.empty())
            return (ggml_tensor*)nullptr;
        ggml_tensor* t = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, ne0, ne1, ne2);
        uploads.emplace_back(t, v.data());
        return t;
    };

    g.blocks.resize(c.blocks.size());
    for (size_t bi = 0; bi < c.blocks.size(); bi++) {
        auto& blk = c.blocks[bi];
        auto& gb = g.blocks[bi];
        gb.subs.resize(blk.subs.size());
        for (size_t si = 0; si < blk.subs.size(); si++) {
            auto& sub = blk.subs[si];
            auto& gs = gb.subs[si];
            gs.dw = mk(sub.dw_w, sub.dw_K, 1, sub.dw_C);
            gs.pw = mk(sub.pw_w, 1, sub.pw_in, sub.pw_out);
            gs.bn_g = mk1(sub.dw_bn.gamma);
            gs.bn_b = mk1(sub.dw_bn.beta);
        }
        if (!blk.se.fc1_w.empty()) {
            gb.se_fc1 = mk(blk.se.fc1_w, blk.se.C, blk.se.se_ch, 1);
            gb.se_fc2 = mk(blk.se.fc2_w, blk.se.se_ch, blk.se.C, 1);
        }
        if (blk.has_residual) {
            gb.res_pw = mk(blk.res.conv_w, 1, blk.res.in_ch, blk.res.out_ch);
            gb.res_bn_g = mk1(blk.res.bn.gamma);
            gb.res_bn_b = mk1(blk.res.bn.beta);
        }
    }
    auto& a = c.asp;
    const int asp_in = c.epilog_channels * 3;
    g.tdnn = mk(a.tdnn_w, 1, asp_in, 128);
    g.tdnn_b = mk1(a.tdnn_b);
    g.tdnn_bn_g = mk1(a.tdnn_bn.gamma);
    g.tdnn_bn_b = mk1(a.tdnn_bn.beta);
    g.aconv = mk(a.conv_w, 1, 128, c.epilog_channels);
    g.aconv_b = mk1(a.conv_b);
    g.pool_bn_g = mk1(a.pool_bn.gamma);
    g.pool_bn_b = mk1(a.pool_bn.beta);
    g.fc = mk(a.fc_w, c.epilog_channels * 2, c.emb_dim, 1);
    g.fc_b = mk1(a.fc_b);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, ctx->backend_cpu);
    if (!g.buf)
        return false;
    for (auto& [t, data] : uploads)
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    return true;
}

// Encoder + ASP as one graph. `mel` is row-major (T, n_mels); writes the
// un-normalized embedding (emb_dim floats) to `emb`.
static bool titanet_embed_ggml(titanet_context* ctx, const float* mel, int T, float* emb) {
    titanet_bench_stage _bs("encoder_ggml");
    auto& c = ctx->cache;
    auto& g = ctx->g;

    const size_t GRAPH_NODES = 1024;
    ggml_init_params ip = {GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false),
                           nullptr, true};
    ggml_context* c0 = ggml_init(ip);
    if (!c0)
        return false;
    ggml_cgraph* gf = ggml_new_graph_custom(c0, GRAPH_NODES, false);

    ggml_tensor* mel_in = ggml_new_tensor_2d(c0, GGML_TYPE_F32, c.n_mels, T); // row-major (T, n_mels)
    ggml_set_input(mel_in);
    ggml_tensor* x = ggml_cont(c0, ggml_transpose(c0, mel_in)); // (T, C) time-fastest

    auto bn = [&](ggml_tensor* t, ggml_tensor* gam, ggml_tensor* bet) {
        const int64_t C = t->ne[1];
        return ggml_add(c0, ggml_mul(c0, t, ggml_reshape_2d(c0, gam, 1, C)), ggml_reshape_2d(c0, bet, 1, C));
    };

    for (size_t bi = 0; bi < g.blocks.size(); bi++) {
        auto& gb = g.blocks[bi];
        ggml_tensor* blk_in = x;
        for (size_t si = 0; si < gb.subs.size(); si++) {
            auto& gs = gb.subs[si];
            const int K = (int)gs.dw->ne[0];
            ggml_tensor* h = ggml_conv_1d_dw(c0, gs.dw, x, 1, (K - 1) / 2, 1);
            h = ggml_reshape_2d(c0, h, x->ne[0], gs.dw->ne[2]);
            h = ggml_conv_1d(c0, gs.pw, h, 1, 0, 1); // (T, Cout)
            h = bn(h, gs.bn_g, gs.bn_b);
            if (si + 1 < gb.subs.size())
                h = ggml_relu(c0, h);
            x = h;
        }
        if (gb.se_fc1) {
            ggml_tensor* pooled = ggml_mean(c0, x); // over T → (1, C)
            const int64_t C = x->ne[1];
            ggml_tensor* hse = ggml_relu(c0, ggml_mul_mat(c0, gb.se_fc1, ggml_reshape_2d(c0, pooled, C, 1)));
            ggml_tensor* s = ggml_sigmoid(c0, ggml_mul_mat(c0, gb.se_fc2, hse)); // (C, 1)
            x = ggml_mul(c0, x, ggml_reshape_2d(c0, s, 1, C));
        }
        if (gb.res_pw) {
            ggml_tensor* r = ggml_conv_1d(c0, gb.res_pw, blk_in, 1, 0, 1);
            r = bn(r, gb.res_bn_g, gb.res_bn_b);
            x = ggml_add(c0, x, r);
        }
        x = ggml_relu(c0, x);
    }

    // ---- ASP ----
    // x is (T, 3072). Stats over time; softmax over time per channel.
    ggml_tensor* mean = ggml_mean(c0, x); // (1, C)
    ggml_tensor* var = ggml_sub(c0, ggml_mean(c0, ggml_sqr(c0, x)), ggml_sqr(c0, mean));
    ggml_tensor* stdv = ggml_sqrt(c0, ggml_clamp(c0, var, 1e-10f, 3.4e38f));
    ggml_tensor* xcat = ggml_concat(c0, x, ggml_repeat(c0, mean, x), 1);
    xcat = ggml_concat(c0, xcat, ggml_repeat(c0, stdv, x), 1); // (T, 9216)

    ggml_tensor* h = ggml_conv_1d(c0, g.tdnn, xcat, 1, 0, 1); // (T, 128)
    h = ggml_add(c0, h, ggml_reshape_2d(c0, g.tdnn_b, 1, 128));
    h = ggml_tanh(c0, bn(h, g.tdnn_bn_g, g.tdnn_bn_b));

    ggml_tensor* attn = ggml_conv_1d(c0, g.aconv, h, 1, 0, 1); // (T, 3072)
    attn = ggml_add(c0, attn, ggml_reshape_2d(c0, g.aconv_b, 1, (int64_t)c.epilog_channels));
    attn = ggml_soft_max(c0, attn); // over ne[0]=T per channel

    ggml_tensor* wm = ggml_sum_rows(c0, ggml_mul(c0, attn, x));                // (1, C)
    ggml_tensor* wm2 = ggml_sum_rows(c0, ggml_mul(c0, attn, ggml_sqr(c0, x))); // (1, C)
    ggml_tensor* wstd = ggml_sqrt(c0, ggml_clamp(c0, ggml_sub(c0, wm2, ggml_sqr(c0, wm)), 1e-10f, 3.4e38f));

    ggml_tensor* pool = ggml_concat(c0, wm, wstd, 1); // (1, 6144)
    const int64_t P = (int64_t)c.epilog_channels * 2;
    pool = ggml_add(c0, ggml_mul(c0, pool, ggml_reshape_2d(c0, g.pool_bn_g, 1, P)),
                    ggml_reshape_2d(c0, g.pool_bn_b, 1, P));

    ggml_tensor* out = ggml_mul_mat(c0, ggml_reshape_2d(c0, g.fc, P, c.emb_dim), ggml_reshape_2d(c0, pool, P, 1));
    out = ggml_add(c0, out, ggml_reshape_2d(c0, g.fc_b, (int64_t)c.emb_dim, 1)); // (emb_dim, 1)
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend_cpu));
    bool ok = ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        ggml_backend_tensor_set(mel_in, mel, 0, (size_t)T * c.n_mels * sizeof(float));
        ok = ggml_backend_graph_compute(ctx->backend_cpu, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok)
        ggml_backend_tensor_get(out, emb, 0, (size_t)c.emb_dim * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(c0);
    return ok;
}

// ============================================================================
// onnxruntime path (CRISPASR_USE_ONNXRUNTIME) — model_path ending in ".onnx"
// ============================================================================

#if defined(CRISPASR_USE_ONNXRUNTIME)

// NeMo/librosa-style mel filterbank with slaney area normalization.
// 80 triangular filters over [0, sr/2], matching AudioToMelSpectrogramPreprocessor.
static std::vector<float> compute_mel_filterbank_slaney(int sr, int n_fft, int n_mels) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> fb((size_t)n_mels * n_bins, 0.0f);
    auto hz_to_mel = [](float f) { return 2595.0f * std::log10(1.0f + f / 700.0f); };
    auto mel_to_hz = [](float m) { return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f); };
    const float f_min = 0.0f, f_max = (float)sr / 2.0f;
    const float m_min = hz_to_mel(f_min), m_max = hz_to_mel(f_max);
    std::vector<float> mels(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        mels[i] = m_min + (m_max - m_min) * (float)i / (float)(n_mels + 1);
    for (int m = 0; m < n_mels; m++) {
        const float f_l = mel_to_hz(mels[m]);
        const float f_c = mel_to_hz(mels[m + 1]);
        const float f_r = mel_to_hz(mels[m + 2]);
        const float norm = (f_r - f_l) / 2.0f; // slaney: normalize by triangle area
        for (int k = 0; k < n_bins; k++) {
            const float f = (float)k * (float)sr / (float)n_fft;
            float w = 0.0f;
            if (f >= f_l && f <= f_c)
                w = (f - f_l) / (f_c - f_l);
            else if (f >= f_c && f <= f_r)
                w = (f_r - f) / (f_r - f_c);
            if (w > 0.0f)
                fb[(size_t)m * n_bins + k] = w * 2.0f / norm;
        }
    }
    return fb;
}

// Fill the model cache with TitaNet-Large constants and the embedded
// preprocessor (periodic Hann + slaney mel), then open the ORT session.
static struct titanet_context* titanet_init_ort(struct titanet_context* ctx, const char* model_path, int n_threads) {
    auto& c = ctx->cache;
    c.n_mels = 80;
    c.n_fft = 512;
    c.emb_dim = 192;
    c.channels = 1024;
    c.epilog_channels = 3072;
    c.n_blocks = 5;
    c.se_channels = 128;
    c.block_repeats = {1, 3, 3, 3, 1};
    c.block_kernels = {3, 7, 11, 15, 1};

    // Periodic Hann window (400) — the GGUF path overrides the embedded
    // symmetric window with periodic; keep identical preprocessing here.
    c.hann_window.resize(400);
    for (int i = 0; i < 400; i++)
        c.hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / 400.0f));

    c.mel_fb = compute_mel_filterbank_slaney(16000, c.n_fft, c.n_mels);

    const bool force_cpu = crispasr_env::get("CRISPASR_ORT_FORCE_CPU") != nullptr;
    std::string provider;
    try {
        ctx->ort_session =
            new Ort::Session(crispasr_ort::create_session(model_path, n_threads > 0 ? n_threads : 4, force_cpu, provider));
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "titanet[ort]: failed to create session: %s\n", e.what());
        delete ctx;
        return nullptr;
    }
    c.initialised = true;
    fprintf(stderr, "titanet[ort]: %s (%s EP)\n", model_path, provider.c_str());
    return ctx;
}

static int titanet_embed_ort(struct titanet_context* ctx, const float* mel, int T, int n_samples, float* out) {
    auto& c = ctx->cache;
    // Transpose (T, 80) → (1, 80, T), the NeMo ONNX input layout.
    std::vector<float> x((size_t)c.n_mels * T);
    for (int m = 0; m < c.n_mels; m++)
        for (int t = 0; t < T; t++)
            x[(size_t)m * T + t] = mel[(size_t)t * c.n_mels + m];

    std::vector<int64_t> audio_shape = {1, c.n_mels, T};
    std::vector<int64_t> len_shape = {1};
    int64_t len_val = n_samples;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_audio = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), audio_shape.data(), 3);
    Ort::Value in_len = Ort::Value::CreateTensor<int64_t>(mem, &len_val, 1, len_shape.data(), 1);

    const char* in_names[] = {"audio_signal", "length"};
    const char* out_names[] = {"embs"};
    Ort::RunOptions ro;
    // Run() expects a contiguous array of Ort::Value — build one.
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(in_audio));
    inputs.push_back(std::move(in_len));
    auto outs = ctx->ort_session->Run(ro, in_names, inputs.data(), 2, out_names, 1);
    if (outs.size() != 1)
        return 0;

    const float* data = outs[0].GetTensorData<float>();
    float norm = 0.0f;
    for (int i = 0; i < c.emb_dim; i++) {
        out[i] = data[i];
        norm += out[i] * out[i];
    }
    norm = 1.0f / (std::sqrt(norm) + 1e-12f);
    for (int i = 0; i < c.emb_dim; i++)
        out[i] *= norm;
    titanet_dump_emb(out, c.emb_dim);
    return c.emb_dim;
}

#endif // CRISPASR_USE_ONNXRUNTIME

extern "C" struct titanet_context* titanet_init(const char* model_path, int n_threads) {
    auto* ctx = new titanet_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    auto& c = ctx->cache;

#if defined(CRISPASR_USE_ONNXRUNTIME)
    if (crispasr_ort::is_onnx_path(model_path))
        return titanet_init_ort(ctx, model_path, n_threads);
#endif

    // Phase 1: metadata
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }
    c.n_mels = (int)core_gguf::kv_u32(gctx, "titanet.n_mels", 80);
    c.n_fft = (int)core_gguf::kv_u32(gctx, "titanet.n_fft", 512);
    c.channels = (int)core_gguf::kv_u32(gctx, "titanet.channels", 1024);
    c.epilog_channels = (int)core_gguf::kv_u32(gctx, "titanet.epilog_channels", 3072);
    c.emb_dim = (int)core_gguf::kv_u32(gctx, "titanet.emb_dim", 192);
    c.n_blocks = (int)core_gguf::kv_u32(gctx, "titanet.n_blocks", 5);
    c.se_channels = (int)core_gguf::kv_u32(gctx, "titanet.se_channels", 128);

    // Read block_repeats and block_kernels arrays
    {
        int key = gguf_find_key(gctx, "titanet.block_repeats");
        if (key >= 0) {
            int n = gguf_get_arr_n(gctx, key);
            c.block_repeats.resize(n);
            for (int i = 0; i < n; i++)
                c.block_repeats[i] = ((const int32_t*)gguf_get_arr_data(gctx, key))[i];
        }
        key = gguf_find_key(gctx, "titanet.block_kernels");
        if (key >= 0) {
            int n = gguf_get_arr_n(gctx, key);
            c.block_kernels.resize(n);
            for (int i = 0; i < n; i++)
                c.block_kernels[i] = ((const int32_t*)gguf_get_arr_data(gctx, key))[i];
        }
    }
    core_gguf::free_metadata(gctx);

    if ((int)c.block_repeats.size() != c.n_blocks || (int)c.block_kernels.size() != c.n_blocks) {
        fprintf(stderr, "titanet: block_repeats/kernels array size mismatch\n");
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "titanet: %d mels, %d channels, %d emb_dim, %d blocks\n", c.n_mels, c.channels, c.emb_dim,
            c.n_blocks);

    // Phase 2: load weights
    ctx->backend = crispasr_init_gpu_backend();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "titanet", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // Pre-fold BN and cache all weights
    init_cache(ctx);

    // ggml compute path (default): upload the folded cache to a CPU backend
    // buffer once. On failure fall back to the legacy scalar path.
    if (!titanet_use_legacy()) {
        ctx->backend_cpu = core_cpu_backend::init();
        if (ctx->backend_cpu) {
            core_cpu_backend::set_n_threads(ctx->backend_cpu, ctx->n_threads);
            if (!titanet_upload_ggml_weights(ctx)) {
                fprintf(stderr, "titanet: ggml weight upload failed, using legacy path\n");
                if (ctx->g.buf)
                    ggml_backend_buffer_free(ctx->g.buf);
                if (ctx->g.ctx)
                    ggml_free(ctx->g.ctx);
                ctx->g = TitanetGgml{};
                ggml_backend_free(ctx->backend_cpu);
                ctx->backend_cpu = nullptr;
            }
        }
    }

    return ctx;
}

extern "C" void titanet_free(struct titanet_context* ctx) {
    if (!ctx)
        return;
#if defined(CRISPASR_USE_ONNXRUNTIME)
    delete ctx->ort_session;
    ctx->ort_session = nullptr;
#endif
    if (ctx->g.buf)
        ggml_backend_buffer_free(ctx->g.buf);
    if (ctx->g.ctx)
        ggml_free(ctx->g.ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->weight_ctx)
        ggml_free(ctx->weight_ctx);
    if (ctx->buf)
        core_gguf::release_weight_buffer(ctx->buf);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ============================================================================
// Forward pass — pure CPU
// ============================================================================

// Radix-2 FFT (same as core/kaldi_fbank.cpp)
static void fft_radix2(float* re, float* im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = std::cos(ang), wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float tr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                float ti = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                re[i + j + len / 2] = re[i + j] - tr;
                im[i + j + len / 2] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
                float nr = cr * wre - ci * wim;
                ci = cr * wim + ci * wre;
                cr = nr;
            }
        }
    }
}

// NeMo-compatible mel spectrogram matching FilterbankFeatures.forward():
//   1. Pre-emphasis: x[t] = x[t] - 0.97 * x[t-1]  (NeMo default)
//   2. STFT: center=True with pad_mode="constant" (zero-pad), periodic Hann,
//      n_fft=512, hop=160, win=400, window centered in frame
//   3. Power spectrum: |STFT|^2
//   4. Mel filterbank: fb @ power
//   5. Log: log(x + 2^-24)
//   6. Per-feature normalization: (x - mean) / (std + 1e-5)
static std::vector<float> compute_mel_spectrogram(const titanet_model_cache& c, const float* pcm, int n_samples,
                                                  int& T_out) {
    T_out = 0;
    const int n_fft = c.n_fft;                     // 512
    const int win_len = (int)c.hann_window.size(); // 400
    const int hop = 160;
    const int n_mels = c.n_mels;      // 80
    const int n_bins = n_fft / 2 + 1; // 257

    // Pre-emphasis: x[t] = x[t] - 0.97 * x[t-1], x[0] = x[0] - 0.97 * x[0]
    // NeMo masks beyond seq_len but for single utterance all samples are valid.
    std::vector<float> preemph(n_samples);
    preemph[0] = pcm[0] - 0.97f * pcm[0]; // = pcm[0] * 0.03
    for (int i = 1; i < n_samples; i++)
        preemph[i] = pcm[i] - 0.97f * pcm[i - 1];

    // Center=True with pad_mode="constant" (zero-pad, NeMo default)
    const int pad = n_fft / 2; // 256
    const int n_padded = n_samples + 2 * pad;
    std::vector<float> padded(n_padded, 0.0f);
    std::memcpy(padded.data() + pad, preemph.data(), n_samples * sizeof(float));

    int T = (n_padded - n_fft) / hop + 1;
    if (T <= 0)
        return {};

    std::vector<float> features((size_t)T * (size_t)n_mels);
    std::vector<float> fre(n_fft), fim(n_fft);

    for (int t = 0; t < T; t++) {
        int off = t * hop;
        std::fill(fre.begin(), fre.end(), 0.0f);
        std::fill(fim.begin(), fim.end(), 0.0f);

        // PyTorch torch.stft centers the window when win_length < n_fft:
        //   left_pad = (n_fft - win_length) / 2
        //   window_padded[left_pad : left_pad + win_length] = window
        // Then: frame = signal[off : off+n_fft] * window_padded
        int win_start = (n_fft - win_len) / 2; // 56 for n_fft=512, win=400
        for (int i = 0; i < win_len; i++)
            fre[win_start + i] = padded[off + win_start + i] * c.hann_window[i];

        fft_radix2(fre.data(), fim.data(), n_fft);

        // Power spectrum → mel filterbank → log
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_bins; k++) {
                float pw = fre[k] * fre[k] + fim[k] * fim[k];
                sum += pw * c.mel_fb[m * n_bins + k];
            }
            // NeMo: log(x + 2^-24) with log_zero_guard_type="add"
            features[t * n_mels + m] = std::log(sum + 5.960464477539063e-8f);
        }
    }

    // Per-feature normalization: (x - mean) / (std + 1e-5)
    // NeMo's normalize="per_feature"
    for (int m = 0; m < n_mels; m++) {
        double sum = 0, sum2 = 0;
        for (int t = 0; t < T; t++) {
            double v = features[t * n_mels + m];
            sum += v;
            sum2 += v * v;
        }
        float mean = (float)(sum / T);
        float var = (float)(sum2 / T - (double)mean * mean);
        float inv_std = 1.0f / (std::sqrt(std::max(var, 0.0f)) + 1e-5f);
        for (int t = 0; t < T; t++)
            features[t * n_mels + m] = (features[t * n_mels + m] - mean) * inv_std;
    }

    // NeMo's AudioToMelSpectrogramPreprocessor reports feat_len = ceil(n_samples / hop).
    // Our STFT may produce 1 extra frame. Truncate to the valid length to match NeMo's
    // MaskedConv1d behavior (which zeroes frames beyond feat_len).
    int T_valid = (n_samples + hop - 1) / hop; // ceil(n_samples / hop)
    if (T_valid < T) {
        features.resize((size_t)T_valid * (size_t)n_mels);
        T = T_valid;
    }

    T_out = T;
    return features;
}

// Apply BatchNorm in-place on [C, T] data using pre-folded gamma/beta
static void apply_bn(float* data, int C, int T, const BNFolded& bn) {
    for (int c = 0; c < C; c++) {
        float g = bn.gamma[c], b = bn.beta[c];
        float* row = data + (size_t)c * T;
        for (int t = 0; t < T; t++)
            row[t] = row[t] * g + b;
    }
}

// ReLU in-place
static void apply_relu(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0)
            data[i] = 0;
}

// Depthwise Conv1d: each channel has its own kernel of size K.
// Input: [C, T], kernel: [C, K], output: [C, T] (same padding)
// Weight layout from NeMo: (C, 1, K) stored row-major → flat (C*K)
static std::vector<float> depthwise_conv1d(const float* input, int C, int T, const float* kernel, int K) {
    int pad = (K - 1) / 2;
    std::vector<float> out((size_t)C * T, 0.0f);
    for (int c = 0; c < C; c++) {
        const float* kc = kernel + (size_t)c * K;
        const float* ic = input + (size_t)c * T;
        float* oc = out.data() + (size_t)c * T;
        for (int t = 0; t < T; t++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                int ti = t + k - pad;
                if (ti >= 0 && ti < T)
                    s += ic[ti] * kc[k];
            }
            oc[t] = s;
        }
    }
    return out;
}

// Pointwise Conv1d (k=1): matmul. Input [C_in, T], weight [C_out, C_in] → [C_out, T]
static std::vector<float> pointwise_conv1d(const float* input, int C_in, int T, const float* weight, int C_out) {
    std::vector<float> out((size_t)C_out * T, 0.0f);
#if defined(HAVE_ACCELERATE)
    if (!titanet_use_scalar()) {
        // out[C_out, T] = W[C_out, C_in] @ input[C_in, T]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_out, T, C_in, 1.0f, weight, C_in, input, T, 0.0f,
                    out.data(), T);
        return out;
    }
#endif
    for (int co = 0; co < C_out; co++) {
        const float* wrow = weight + (size_t)co * C_in;
        float* orow = out.data() + (size_t)co * T;
        for (int t = 0; t < T; t++) {
            float s = 0.0f;
            for (int ci = 0; ci < C_in; ci++)
                s += input[(size_t)ci * T + t] * wrow[ci];
            orow[t] = s;
        }
    }
    return out;
}

// SE block: global avg pool → FC1 → ReLU → FC2 → sigmoid → scale
static void apply_se(float* data, int C, int T, const SECache& se) {
    // Global average pool over time
    std::vector<float> pooled(C, 0.0f);
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int t = 0; t < T; t++)
            s += data[(size_t)c * T + t];
        pooled[c] = s / T;
    }

    // FC1: (se_ch, C) @ pooled → (se_ch,)
    std::vector<float> h(se.se_ch, 0.0f);
    for (int i = 0; i < se.se_ch; i++) {
        float s = 0.0f;
        for (int c = 0; c < C; c++)
            s += se.fc1_w[(size_t)i * C + c] * pooled[c];
        h[i] = std::max(s, 0.0f); // ReLU
    }

    // FC2: (C, se_ch) @ h → (C,)
    std::vector<float> scale(C, 0.0f);
    for (int c = 0; c < C; c++) {
        float s = 0.0f;
        for (int i = 0; i < se.se_ch; i++)
            s += se.fc2_w[(size_t)c * se.se_ch + i] * h[i];
        scale[c] = 1.0f / (1.0f + std::exp(-s)); // sigmoid
    }

    // Scale: data[c,t] *= scale[c]
    for (int c = 0; c < C; c++) {
        float sc = scale[c];
        for (int t = 0; t < T; t++)
            data[(size_t)c * T + t] *= sc;
    }
}

extern "C" int titanet_embed(struct titanet_context* ctx, const float* pcm_16k, int n_samples, float* out) {
    if (!ctx || !pcm_16k || n_samples <= 0 || !out)
        return 0;

    auto& c = ctx->cache;
    if (!c.initialised)
        return 0;

    titanet_bench_stage _bs_total("embed_total");

    // 1. Mel spectrogram
    int T = 0;
    auto mel = compute_mel_spectrogram(c, pcm_16k, n_samples, T);
    if (mel.empty() || T <= 0)
        return 0;

    // Debug: optionally load reference mel features
    const char* ref_path = crispasr_env::get("CRISPASR_TITANET_REF_MEL");
    if (ref_path && *ref_path) {
        FILE* f = fopen(ref_path, "rb");
        if (f) {
            fread(mel.data(), sizeof(float), (size_t)T * c.n_mels, f);
            fclose(f);
            fprintf(stderr, "titanet: LOADED ref mel from %s (%d frames)\n", ref_path, T);
        }
    }

#if defined(CRISPASR_USE_ONNXRUNTIME)
    // onnxruntime path: the ONNX graph replaces the hand-rolled encoder.
    if (ctx->ort_session)
        return titanet_embed_ort(ctx, mel.data(), T, n_samples, out);
#endif

    // ggml path: encoder + ASP as one CPU-backend graph, then shared L2 norm.
    if (ctx->backend_cpu && ctx->g.ctx) {
        if (!titanet_embed_ggml(ctx, mel.data(), T, out))
            return 0;
        float norm = 0.0f;
        for (int i = 0; i < c.emb_dim; i++)
            norm += out[i] * out[i];
        norm = 1.0f / (std::sqrt(norm) + 1e-12f);
        for (int i = 0; i < c.emb_dim; i++)
            out[i] *= norm;
        titanet_dump_emb(out, c.emb_dim);
        return c.emb_dim;
    }

    // Transpose mel from (T, n_mels) to (n_mels, T) for channel-first processing
    std::vector<float> x((size_t)c.n_mels * T);
    for (int m = 0; m < c.n_mels; m++)
        for (int t = 0; t < T; t++)
            x[(size_t)m * T + t] = mel[(size_t)t * c.n_mels + m];

    // 2. Encoder blocks
    for (int bi = 0; bi < c.n_blocks; bi++) {
        auto& blk = c.blocks[bi];
        int R = (int)blk.subs.size();
        int C_cur = (bi == 0) ? c.n_mels : c.channels;

        // Save input for residual
        std::vector<float> residual;
        if (blk.has_residual)
            residual = x;

        for (int si = 0; si < R; si++) {
            auto& sub = blk.subs[si];
            int C_in = (si == 0) ? C_cur : c.channels;

            // Depthwise conv1d
            auto dw_out = depthwise_conv1d(x.data(), C_in, T, sub.dw_w.data(), sub.dw_K);

            // Pointwise conv1d
            int C_out = sub.pw_out > 0 ? sub.pw_out : c.channels;
            auto pw_out = pointwise_conv1d(dw_out.data(), C_in, T, sub.pw_w.data(), C_out);

            // BatchNorm
            apply_bn(pw_out.data(), C_out, T, sub.dw_bn);
            // ReLU only for non-last sub-blocks (NeMo JasperBlock: last sub-block
            // goes straight to SE without activation)
            if (si < R - 1)
                apply_relu(pw_out.data(), C_out * T);

            x = std::move(pw_out);
        }

        // SE block
        int C_blk = (bi == c.n_blocks - 1) ? c.epilog_channels : c.channels;
        if (!blk.se.fc1_w.empty())
            apply_se(x.data(), C_blk, T, blk.se);

        // Residual
        if (blk.has_residual && !residual.empty()) {
            // Apply residual conv + BN to the original input
            auto res_out = pointwise_conv1d(residual.data(), blk.res.in_ch, T, blk.res.conv_w.data(), blk.res.out_ch);
            apply_bn(res_out.data(), blk.res.out_ch, T, blk.res.bn);

            // Add: x += residual
            int n = C_blk * T;
            for (int i = 0; i < n; i++)
                x[i] += res_out[i];
        }

        // mout: ReLU after SE + residual (NeMo JasperBlock always has this)
        apply_relu(x.data(), C_blk * T);
    }

    // 3. ASP (Attentive Statistics Pooling)
    int C_enc = c.epilog_channels; // 3072

    // Compute mean and std over time
    std::vector<float> h_mean(C_enc), h_std(C_enc);
    for (int c_ = 0; c_ < C_enc; c_++) {
        double sum = 0, sum2 = 0;
        for (int t = 0; t < T; t++) {
            float v = x[(size_t)c_ * T + t];
            sum += v;
            sum2 += (double)v * v;
        }
        h_mean[c_] = (float)(sum / T);
        h_std[c_] = std::sqrt(std::max((float)(sum2 / T) - h_mean[c_] * h_mean[c_], 1e-10f));
    }

    // ASP TDNN: input is cat(features, mean_expanded, std_expanded) = 9216 × T
    // Output: 128 × T
    auto& asp = c.asp;
    int asp_in = C_enc * 3; // 9216
    std::vector<float> asp_h(128 * T, 0.0f);
#if defined(HAVE_ACCELERATE)
    if (!titanet_use_scalar()) {
        // asp_h[128, T] = W[128, 9216] @ X[9216, T], X = [x; mean⊗1; std⊗1].
        std::vector<float> X((size_t)asp_in * T);
        std::memcpy(X.data(), x.data(), (size_t)C_enc * T * sizeof(float));
        for (int c_ = 0; c_ < C_enc; c_++) {
            float* rm = X.data() + (size_t)(C_enc + c_) * T;
            float* rs = X.data() + (size_t)(2 * C_enc + c_) * T;
            float vm = h_mean[c_], vs = h_std[c_];
            for (int t = 0; t < T; t++) {
                rm[t] = vm;
                rs[t] = vs;
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128, T, asp_in, 1.0f, asp.tdnn_w.data(), asp_in,
                    X.data(), T, 0.0f, asp_h.data(), T);
        if (!asp.tdnn_b.empty())
            for (int i = 0; i < 128; i++) {
                float b = asp.tdnn_b[i];
                float* row = asp_h.data() + (size_t)i * T;
                for (int t = 0; t < T; t++)
                    row[t] += b;
            }
    } else
#endif
    {
        for (int i = 0; i < 128; i++) {
            for (int t = 0; t < T; t++) {
                double s = asp.tdnn_b.empty() ? 0.0 : asp.tdnn_b[i];
                for (int c_ = 0; c_ < C_enc; c_++) {
                    s += x[(size_t)c_ * T + t] * asp.tdnn_w[(size_t)i * asp_in + c_];
                    s += h_mean[c_] * asp.tdnn_w[(size_t)i * asp_in + C_enc + c_];
                    s += h_std[c_] * asp.tdnn_w[(size_t)i * asp_in + 2 * C_enc + c_];
                }
                asp_h[(size_t)i * T + t] = (float)s;
            }
        }
    }

    // BN + tanh
    for (int c_ = 0; c_ < 128; c_++) {
        float g = asp.tdnn_bn.gamma[c_], b = asp.tdnn_bn.beta[c_];
        for (int t = 0; t < T; t++)
            asp_h[(size_t)c_ * T + t] = std::tanh(asp_h[(size_t)c_ * T + t] * g + b);
    }

    // Attention conv: 128 → 3072
    std::vector<float> attn((size_t)C_enc * T, 0.0f);
#if defined(HAVE_ACCELERATE)
    if (!titanet_use_scalar()) {
        // attn[C_enc, T] = W[C_enc, 128] @ asp_h[128, T]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C_enc, T, 128, 1.0f, asp.conv_w.data(), 128,
                    asp_h.data(), T, 0.0f, attn.data(), T);
        if (!asp.conv_b.empty())
            for (int c_ = 0; c_ < C_enc; c_++) {
                float b = asp.conv_b[c_];
                float* row = attn.data() + (size_t)c_ * T;
                for (int t = 0; t < T; t++)
                    row[t] += b;
            }
    } else
#endif
    {
        for (int c_ = 0; c_ < C_enc; c_++) {
            for (int t = 0; t < T; t++) {
                double s = asp.conv_b.empty() ? 0.0 : asp.conv_b[c_];
                for (int k = 0; k < 128; k++)
                    s += asp_h[(size_t)k * T + t] * asp.conv_w[(size_t)c_ * 128 + k];
                attn[(size_t)c_ * T + t] = (float)s;
            }
        }
    }

    // Softmax over time
    for (int c_ = 0; c_ < C_enc; c_++) {
        float mx = attn[(size_t)c_ * T];
        for (int t = 1; t < T; t++)
            mx = std::max(mx, attn[(size_t)c_ * T + t]);
        float sum = 0;
        for (int t = 0; t < T; t++) {
            attn[(size_t)c_ * T + t] = std::exp(attn[(size_t)c_ * T + t] - mx);
            sum += attn[(size_t)c_ * T + t];
        }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t < T; t++)
            attn[(size_t)c_ * T + t] *= inv_sum;
    }

    // Weighted mean + std
    std::vector<float> w_mean(C_enc), w_std(C_enc);
    for (int c_ = 0; c_ < C_enc; c_++) {
        float wm = 0, wm2 = 0;
        for (int t = 0; t < T; t++) {
            float a = attn[(size_t)c_ * T + t], v = x[(size_t)c_ * T + t];
            wm += a * v;
            wm2 += a * v * v;
        }
        w_mean[c_] = wm;
        w_std[c_] = std::sqrt(std::max(wm2 - wm * wm, 1e-10f));
    }

    // Pool: cat(w_mean, w_std) → 6144
    std::vector<float> pool(6144);
    for (int c_ = 0; c_ < C_enc; c_++) {
        pool[c_] = w_mean[c_];
        pool[C_enc + c_] = w_std[c_];
    }

    // Pool BN
    for (int c_ = 0; c_ < 6144; c_++)
        pool[c_] = pool[c_] * asp.pool_bn.gamma[c_] + asp.pool_bn.beta[c_];

    // FC: Linear(6144 → 192)
    int emb_dim = c.emb_dim;
    for (int i = 0; i < emb_dim; i++) {
        double s = asp.fc_b.empty() ? 0.0 : asp.fc_b[i];
        for (int k = 0; k < 6144; k++)
            s += pool[k] * asp.fc_w[(size_t)i * 6144 + k];
        out[i] = (float)s;
    }

    // L2 normalize
    float norm = 0.0f;
    for (int i = 0; i < emb_dim; i++)
        norm += out[i] * out[i];
    norm = 1.0f / (std::sqrt(norm) + 1e-12f);
    for (int i = 0; i < emb_dim; i++)
        out[i] *= norm;

    titanet_dump_emb(out, emb_dim);
    return emb_dim;
}

extern "C" float titanet_cosine_sim(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0)
        return 0.0f;
    float dot = 0.0f;
    for (int i = 0; i < dim; i++)
        dot += a[i] * b[i];
    return dot;
}
