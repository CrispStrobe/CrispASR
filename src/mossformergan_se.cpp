// mossformergan_se.cpp — native MossFormerGAN_SE_16K (see mossformergan_se.h).
//
// Upstream blueprint: modelscope/ClearerVoice-Studio @ 6b3774dc,
// clearvoice/models/mossformer_gan_se/generator.py (SyncANet) and
// clearvoice/utils/decode.py (decode_one_audio_mossformergan_se_16k).
//
// Tensor layout. Activations are kept channels-fastest ("CL"): a torch
// [C, T, F] feature map is ggml ne [C, F, T]. Then every 1x1 conv / Linear is a
// plain mul_mat, every per-position norm is ggml_norm over ne0, the
// sub-pixel shuffle is a reshape, and 2-D convs are per-tap mul_mats over
// flat shifted views of a zero-padded input (no im2col, which would be ~2 GB
// for the 256-channel dense layer on a 10 s window). The converter stores
// conv weights per tap ([taps, Cout, Cin]) and splits the dilated-dense input
// into 64-channel parts so the dense concatenation is never materialised.
// Depthwise sequence convs (FSMN, ConvModule) run as ggml_ssm_conv on a
// transposed, symmetrically padded copy.
//
// The MossFormer gated attention has two T x T relu^2 products per block
// (the "cross" term of the intra pass and the quadratic term of the inter
// pass); they are evaluated in row chunks bounded to ~256 MiB and
// accumulated with ggml_acc so peak memory stays modest on a 10 s window.

#include "mossformergan_se.h"

#include "crispasr_enhancer.h"

#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct mfg_hparams {
    int sample_rate = 16000;
    int n_fft = 400;
    int hop = 100;
    int win = 400;
    int C = 64;  // num_channel
    int F = 201; // num_features (n_fft/2 + 1)
    int n_layers = 6;
    int n_head = 4;
    int E = 6;    // attention qk channels per head
    int qk = 128; // MossFormer query_key_dim
    int hidden = 256;
    int rot = 32;
    int decode_window = 10;
    int one_time = 10;
    float compress = 0.3f;
    float eps = 1e-5f;
    float theta = 10000.0f;
};

struct ffconvm_w {
    ggml_tensor *ln_w = nullptr, *ln_b = nullptr, *w = nullptr, *b = nullptr, *dw = nullptr;
};
struct fsmn_w { // fsmn.py UniDeepFsmn
    ggml_tensor *lin_w = nullptr, *lin_b = nullptr, *proj_w = nullptr, *conv = nullptr;
    int lorder = 0;
};
struct dense_layer_w {
    ggml_tensor *conv_w = nullptr, *conv_b = nullptr, *norm_w = nullptr, *norm_b = nullptr, *prelu = nullptr;
    fsmn_w fsmn;
};
struct mossformer_w {
    ffconvm_w hidden, qk, out;
    ggml_tensor *os_gamma = nullptr, *os_beta = nullptr; // [128, 4]
};
struct se_w {
    ggml_tensor *a0w, *a0b, *a2w, *a2b, *m0w, *m0b, *m2w, *m2b;
};
struct branch_w {
    ggml_tensor *norm_g = nullptr, *norm_b = nullptr;
    ffconvm_w to_u, to_v;
    fsmn_w rnn;
    mossformer_w mf;
    ggml_tensor *lin_w = nullptr, *lin_b = nullptr; // [K=2, 64, 128] taps
    se_w se;
};
struct attn_head_w {
    ggml_tensor *w, *b, *prelu, *g, *beta;
};
struct block_w {
    ggml_tensor *fconv_w = nullptr, *fconv_b = nullptr; // [2, 128] / [128]
    branch_w intra, inter;
    std::vector<attn_head_w> q, k, v;
    attn_head_w proj;
};

struct mfg_model {
    mfg_hparams hp;
    core_gguf::WeightLoad wl;
    // encoder
    ggml_tensor *enc_c1_w, *enc_c1_b, *enc_c1_nw, *enc_c1_nb, *enc_c1_a;
    dense_layer_w enc_dense[4];
    ggml_tensor *enc_c2_w, *enc_c2_b, *enc_c2_nw, *enc_c2_nb, *enc_c2_a;
    std::vector<block_w> blocks;
    // mask decoder
    dense_layer_w mask_dense[4];
    ggml_tensor *mask_sp_w, *mask_sp_b, *mask_c1_w, *mask_c1_b, *mask_nw, *mask_nb, *mask_a, *mask_fw, *mask_fb,
        *mask_out_a;
    // complex decoder
    dense_layer_w cplx_dense[4];
    ggml_tensor *cplx_sp_w, *cplx_sp_b, *cplx_a, *cplx_nw, *cplx_nb, *cplx_c_w, *cplx_c_b;
};

} // namespace

struct mossformergan_se_context {
    mfg_model m;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 4;
    bool verbose = false;
    // STFT tables
    std::vector<float> window, cos_t, sin_t; // cos_t/sin_t: [F][n_fft]
};

namespace {

constexpr int kGraphSize = 32768;

// MOSSFORMERGAN_SE_PROFILE=1: per-op wall time (forces node-by-node evaluation).
struct op_profile {
    std::map<std::string, double> ms;
    std::map<std::string, int> count;
    std::chrono::steady_clock::time_point t0;
};
bool profile_cb(ggml_tensor* t, bool ask, void* user) {
    auto* p = (op_profile*)user;
    if (ask) {
        p->t0 = std::chrono::steady_clock::now();
        return true;
    }
    std::string key = ggml_op_desc(t);
    if (t->op == GGML_OP_MUL_MAT)
        key += "[" + std::to_string(t->src[0]->ne[0]) + "x" + std::to_string(t->src[0]->ne[1]) + "]";
    p->ms[key] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p->t0).count();
    p->count[key]++;
    return true;
}
constexpr size_t kChunkBytes = 256ull << 20;

ggml_tensor* req(mfg_model& m, const std::string& name) {
    auto it = m.wl.tensors.find(name);
    if (it == m.wl.tensors.end()) {
        std::fprintf(stderr, "mossformergan_se: missing tensor '%s'\n", name.c_str());
        throw std::runtime_error("missing tensor");
    }
    return it->second;
}

void bind_ffconvm(mfg_model& m, ffconvm_w& f, const std::string& p) {
    f.ln_w = req(m, p + ".mdl.0.weight");
    f.ln_b = req(m, p + ".mdl.0.bias");
    f.w = req(m, p + ".mdl.1.weight");
    f.b = req(m, p + ".mdl.1.bias");
    f.dw = req(m, p + ".mdl.3.sequential.1.conv.weight");
}

void bind_fsmn(mfg_model& m, fsmn_w& f, const std::string& p) {
    f.lin_w = req(m, p + ".linear.weight");
    f.lin_b = req(m, p + ".linear.bias");
    f.proj_w = req(m, p + ".project.weight");
    f.conv = req(m, p + ".conv1.weight");
    f.lorder = (int)(f.conv->ne[1] + 1) / 2;
}

// p: "<module>.dilated_dense." / "<module>.dense_block."
void bind_dense(mfg_model& m, dense_layer_w* d, const std::string& p) {
    for (int i = 0; i < 4; i++) {
        const std::string n = std::to_string(i + 1);
        d[i].conv_w = req(m, p + "conv" + n + ".weight");
        d[i].conv_b = req(m, p + "conv" + n + ".bias");
        d[i].norm_w = req(m, p + "norm" + n + ".weight");
        d[i].norm_b = req(m, p + "norm" + n + ".bias");
        d[i].prelu = req(m, p + "prelu" + n + ".weight");
        bind_fsmn(m, d[i].fsmn, p + "fsmn" + n + ".fsmn");
    }
}

void bind_se(mfg_model& m, se_w& s, const std::string& p) {
    s.a0w = req(m, p + ".avg_pool_layer.0.weight");
    s.a0b = req(m, p + ".avg_pool_layer.0.bias");
    s.a2w = req(m, p + ".avg_pool_layer.2.weight");
    s.a2b = req(m, p + ".avg_pool_layer.2.bias");
    s.m0w = req(m, p + ".max_pool_layer.0.weight");
    s.m0b = req(m, p + ".max_pool_layer.0.bias");
    s.m2w = req(m, p + ".max_pool_layer.2.weight");
    s.m2b = req(m, p + ".max_pool_layer.2.bias");
}

void bind_branch(mfg_model& m, branch_w& b, const std::string& p, const std::string& kind) {
    b.norm_g = req(m, p + kind + "_norm.gamma");
    b.norm_b = req(m, p + kind + "_norm.beta");
    bind_ffconvm(m, b.to_u, p + kind + "_to_u");
    bind_ffconvm(m, b.to_v, p + kind + "_to_v");
    bind_fsmn(m, b.rnn, p + kind + "_rnn.0");
    const std::string mf = p + kind + "_mossformer";
    bind_ffconvm(m, b.mf.hidden, mf + ".to_hidden");
    bind_ffconvm(m, b.mf.qk, mf + ".to_qk");
    bind_ffconvm(m, b.mf.out, mf + ".to_out");
    b.mf.os_gamma = req(m, mf + ".qk_offset_scale.gamma");
    b.mf.os_beta = req(m, mf + ".qk_offset_scale.beta");
    b.lin_w = req(m, p + kind + "_linear.weight");
    b.lin_b = req(m, p + kind + "_linear.bias");
    bind_se(m, b.se, p + kind + "_se");
}

attn_head_w bind_head(mfg_model& m, const std::string& p) {
    return {req(m, p + ".0.weight"), req(m, p + ".0.bias"), req(m, p + ".1.weight"), req(m, p + ".2.gamma"),
            req(m, p + ".2.beta")};
}

bool load_model(mfg_model& m, const char* path, ggml_backend_t backend) {
    gguf_context* g = core_gguf::open_metadata(path);
    if (!g)
        return false;
    const std::string arch = core_gguf::kv_str(g, "general.architecture", "");
    auto& hp = m.hp;
    auto u32 = [&](const char* k, int d) {
        return (int)core_gguf::kv_u32(g, (std::string("mossformergan_se.") + k).c_str(), (uint32_t)d);
    };
    auto f32 = [&](const char* k, float d) {
        return core_gguf::kv_f32(g, (std::string("mossformergan_se.") + k).c_str(), d);
    };
    hp.sample_rate = u32("sample_rate", hp.sample_rate);
    hp.n_fft = u32("n_fft", hp.n_fft);
    hp.hop = u32("hop_length", hp.hop);
    hp.win = u32("win_length", hp.win);
    hp.C = u32("num_channel", hp.C);
    hp.F = u32("num_features", hp.F);
    hp.n_layers = u32("n_layers", hp.n_layers);
    hp.n_head = u32("n_head", hp.n_head);
    hp.E = u32("attn_qk_dim", hp.E);
    hp.qk = u32("mossformer_qk_dim", hp.qk);
    hp.hidden = u32("mossformer_hidden", hp.hidden);
    hp.rot = u32("rotary_dim", hp.rot);
    hp.decode_window = u32("decode_window_seconds", hp.decode_window);
    hp.one_time = u32("one_time_decode_seconds", hp.one_time);
    const int layout = u32("layout", 0);
    hp.compress = f32("compress_factor", hp.compress);
    hp.eps = f32("norm_eps", hp.eps);
    hp.theta = f32("rope_theta", hp.theta);
    core_gguf::free_metadata(g);
    if (arch != "mossformergan-se" || layout != 2 || hp.C != 64 || hp.n_fft != hp.win || hp.F != hp.n_fft / 2 + 1 ||
        hp.hop <= 0 || hp.n_layers <= 0 || hp.n_layers > 64 || hp.n_head <= 0 || hp.qk <= 0 || hp.hidden != 2 * hp.qk ||
        hp.rot <= 0 || hp.rot > hp.qk || hp.decode_window <= 0 || hp.compress <= 0.0f) {
        std::fprintf(stderr,
                     "mossformergan_se: unsupported GGUF metadata (arch '%s', layout %d; reconvert with "
                     "models/convert-mossformergan-se-to-gguf.py)\n",
                     arch.c_str(), layout);
        return false;
    }
    if (!core_gguf::load_weights(path, backend, "mossformergan_se", m.wl))
        return false;
    try {
        const std::string e = "dense_encoder.";
        m.enc_c1_w = req(m, e + "conv_1.0.weight");
        m.enc_c1_b = req(m, e + "conv_1.0.bias");
        m.enc_c1_nw = req(m, e + "conv_1.1.weight");
        m.enc_c1_nb = req(m, e + "conv_1.1.bias");
        m.enc_c1_a = req(m, e + "conv_1.2.weight");
        bind_dense(m, m.enc_dense, e + "dilated_dense.");
        m.enc_c2_w = req(m, e + "conv_2.0.weight");
        m.enc_c2_b = req(m, e + "conv_2.0.bias");
        m.enc_c2_nw = req(m, e + "conv_2.1.weight");
        m.enc_c2_nb = req(m, e + "conv_2.1.bias");
        m.enc_c2_a = req(m, e + "conv_2.2.weight");
        m.blocks.resize(hp.n_layers);
        for (int i = 0; i < hp.n_layers; i++) {
            auto& b = m.blocks[i];
            const std::string p = "blocks." + std::to_string(i) + ".";
            b.fconv_w = req(m, p + "Fconv.weight");
            b.fconv_b = req(m, p + "Fconv.bias");
            bind_branch(m, b.intra, p, "intra");
            bind_branch(m, b.inter, p, "inter");
            for (int h = 0; h < hp.n_head; h++) {
                b.q.push_back(bind_head(m, p + "attn_conv_Q_" + std::to_string(h)));
                b.k.push_back(bind_head(m, p + "attn_conv_K_" + std::to_string(h)));
                b.v.push_back(bind_head(m, p + "attn_conv_V_" + std::to_string(h)));
            }
            b.proj = bind_head(m, p + "attn_concat_proj");
        }
        bind_dense(m, m.mask_dense, "mask_decoder.dense_block.");
        m.mask_sp_w = req(m, "mask_decoder.sub_pixel.conv.weight");
        m.mask_sp_b = req(m, "mask_decoder.sub_pixel.conv.bias");
        m.mask_c1_w = req(m, "mask_decoder.conv_1.weight");
        m.mask_c1_b = req(m, "mask_decoder.conv_1.bias");
        m.mask_nw = req(m, "mask_decoder.norm.weight");
        m.mask_nb = req(m, "mask_decoder.norm.bias");
        m.mask_a = req(m, "mask_decoder.prelu.weight");
        m.mask_fw = req(m, "mask_decoder.final_conv.weight");
        m.mask_fb = req(m, "mask_decoder.final_conv.bias");
        m.mask_out_a = req(m, "mask_decoder.prelu_out.weight");
        bind_dense(m, m.cplx_dense, "complex_decoder.dense_block.");
        m.cplx_sp_w = req(m, "complex_decoder.sub_pixel.conv.weight");
        m.cplx_sp_b = req(m, "complex_decoder.sub_pixel.conv.bias");
        m.cplx_a = req(m, "complex_decoder.prelu.weight");
        m.cplx_nw = req(m, "complex_decoder.norm.weight");
        m.cplx_nb = req(m, "complex_decoder.norm.bias");
        m.cplx_c_w = req(m, "complex_decoder.conv.weight");
        m.cplx_c_b = req(m, "complex_decoder.conv.bias");
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph helpers. All activations are CL layout (channels in ne0).
// ---------------------------------------------------------------------------

struct graph_builder {
    ggml_context* ctx;
    const mfg_hparams& hp;
    ggml_tensor* pos_q = nullptr;     // I32 [Q]   rope positions along frequency
    ggml_tensor* pos_t = nullptr;     // I32 [T]   rope positions along time
    ggml_tensor* offdiag_q = nullptr; // [Q, Q]  1 - eye
    ggml_tensor* offdiag_t = nullptr; // [T, T]
    std::vector<std::pair<std::string, ggml_tensor*>>* taps = nullptr;

    void tap(const char* name, ggml_tensor* t) {
        if (!taps)
            return;
        ggml_set_output(t);
        taps->emplace_back(name, t);
    }

    ggml_tensor* linear(ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx, w, x);
        return b ? ggml_add(ctx, y, b) : y;
    }

    ggml_tensor* layer_norm(ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, hp.eps), w), b);
    }

    // PReLU with per-channel (ne0) or scalar slope.
    ggml_tensor* prelu(ggml_tensor* x, ggml_tensor* a) {
        ggml_tensor* r = ggml_relu(ctx, x);
        return ggml_add(ctx, r, ggml_mul(ctx, ggml_sub(ctx, x, r), a));
    }

    // Depthwise conv along ne1 of x [D, L, R] with kernel k (ggml ne [D, K]), symmetric
    // pad (K-1)/2. ggml_conv_2d_dw_direct's channels-contiguous (CWHN) path consumes
    // the CL activation directly (no transpose / pad copies) and vectorises over D.
    ggml_tensor* dwconv_seq(ggml_tensor* x, ggml_tensor* k) {
        const int64_t D = x->ne[0], L = x->ne[1], R = x->ne[2];
        const int64_t K = k->ne[1];
        const int pad = (int)(K - 1) / 2;
        ggml_tensor* xin = ggml_permute(ctx, ggml_reshape_4d(ctx, x, D, L, R, 1), 2, 0, 1, 3); // [L, R, D, 1]
        ggml_tensor* kk = ggml_permute(ctx, ggml_reshape_4d(ctx, k, D, K, 1, 1), 3, 0, 1, 2);  // [K, 1, 1, D]
        kk->nb[1] = kk->nb[0]; // size-1 dims: any stride; satisfy the CWHN kernel layout check
        kk->nb[2] = kk->nb[0];
        ggml_tensor* y = ggml_conv_2d_dw_direct(ctx, kk, xin, 1, 1, pad, 0, 1, 1); // [L, R, D, 1] (CWHN memory)
        return ggml_permute(ctx, y, 1, 2, 0, 3);                                   // [D, L, R], contiguous
    }

    // FFConvM: LayerNorm -> Linear -> SiLU -> ConvModule (x + dwconv31(x)).
    ggml_tensor* ffconvm(ggml_tensor* x, const ffconvm_w& f) {
        ggml_tensor* h = ggml_silu(ctx, linear(layer_norm(x, f.ln_w, f.ln_b), f.w, f.b));
        return ggml_add(ctx, h, dwconv_seq(h, f.dw));
    }

    // fsmn.py UniDeepFsmn: x + p + dwconv(p), p = project(relu(linear(x))).
    ggml_tensor* fsmn(ggml_tensor* x, const fsmn_w& f) {
        ggml_tensor* p = ggml_mul_mat(ctx, f.proj_w, ggml_relu(ctx, linear(x, f.lin_w, f.lin_b)));
        return ggml_add(ctx, x, ggml_add(ctx, p, dwconv_seq(p, f.conv)));
    }

    // InstanceNorm2d(affine) over (F, T) per channel of x [C, F, T].
    ggml_tensor* instance_norm(ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        const int64_t C = x->ne[0], F = x->ne[1], T = x->ne[2];
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, C, F * T))); // [FT, C]
        xt = ggml_norm(ctx, xt, hp.eps);
        ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, xt)); // [C, FT]
        y = ggml_reshape_3d(ctx, y, C, F, T);
        return ggml_add(ctx, ggml_mul(ctx, y, w), b);
    }

    // Zero-pad x [C, F, T] for the tap convolution: F (lp, rp), T (top, 1 spare row).
    ggml_tensor* pad_for_taps(ggml_tensor* x, int lp, int rp, int top) {
        return ggml_pad_ext(ctx, x, 0, 0, lp, rp, top, 1, 0, 0);
    }

    // Accumulate tap matmuls for one padded part xp [Cin, Fp, Tp]. Output rows
    // cover the full padded width Fp (columns >= F_out are discarded later).
    // W: [Cin, Cout, taps]; tap index = base + kh*KW + kw; row offset per kh is
    // row0 + kh*dil.
    ggml_tensor* taps_part(ggml_tensor* acc, ggml_tensor* xp, ggml_tensor* W, int base, int KH, int KW, int row0,
                           int dil, int T) {
        const int64_t Cin = xp->ne[0], Fp = xp->ne[1];
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                const int row = row0 + kh * dil;
                const size_t off = ((size_t)row * Fp + kw) * xp->nb[1];
                ggml_tensor* v = ggml_view_2d(ctx, xp, Cin, Fp * T, xp->nb[1], off);
                const int idx = base + kh * KW + kw;
                ggml_tensor* w = ggml_view_2d(ctx, W, W->ne[0], W->ne[1], W->nb[1], (size_t)idx * W->nb[2]);
                ggml_tensor* y = ggml_mul_mat(ctx, w, v);
                acc = acc ? ggml_add(ctx, acc, y) : y;
            }
        }
        return acc;
    }

    // Single-part 1xKW conv on x [Cin, F, T], pad (lp, rp), stride s along F. Returns [Cout, F_out, T] + bias.
    ggml_tensor* conv_1xk(ggml_tensor* x, ggml_tensor* W, ggml_tensor* b, int KW, int lp, int rp, int stride) {
        const int64_t F = x->ne[1], T = x->ne[2];
        const int64_t Fp = F + lp + rp;
        const int64_t Ffull = Fp - KW + 1;
        const int64_t Fout = (Ffull - 1) / stride + 1;
        ggml_tensor* xp = pad_for_taps(x, lp, rp, 0);
        ggml_tensor* acc = taps_part(nullptr, xp, W, 0, 1, KW, 0, 1, (int)T);
        const int64_t Cout = acc->ne[0];
        ggml_tensor* v = ggml_view_3d(ctx, acc, Cout, Fout, T, acc->nb[1] * stride, acc->nb[1] * Fp, 0);
        return ggml_add(ctx, ggml_cont(ctx, v), b);
    }

    // DilatedDenseNet (depth 4) on x [64, F, T].
    ggml_tensor* dense_net(ggml_tensor* x, const dense_layer_w* L) {
        const int64_t C = x->ne[0], F = x->ne[1], T = x->ne[2];
        constexpr int kTop = 8;           // largest dilation
        std::vector<ggml_tensor*> hist_p; // padded [x, out0, out1, out2]
        hist_p.push_back(pad_for_taps(x, 1, 1, kTop));
        ggml_tensor* out = nullptr;
        for (int i = 0; i < 4; i++) {
            const int dil = 1 << i;
            ggml_tensor* acc = nullptr;
            for (int p = 0; p <= i; p++) {
                // weight part p covers channels of hist[i - p] (cat([out_{i-1}, ..., out_0, x]))
                acc = taps_part(acc, hist_p[i - p], L[i].conv_w, p * 6, 2, 3, kTop - dil, dil, (int)T);
            }
            const int64_t Fp = F + 2;
            ggml_tensor* y = ggml_view_3d(ctx, acc, C, F, T, acc->nb[1], acc->nb[1] * Fp, 0);
            y = ggml_add(ctx, ggml_cont(ctx, y), L[i].conv_b);
            y = instance_norm(y, L[i].norm_w, L[i].norm_b);
            y = prelu(y, L[i].prelu);
            out = fsmn(y, L[i].fsmn); // sequence along F (ne1), rows T
            if (i < 3)
                hist_p.push_back(pad_for_taps(out, 1, 1, kTop));
        }
        return out;
    }

    // relu^2 attention: q, k [d, L, B]; vals [L, Dv, B] (keys in ne0); optional mask [L, L].
    // Adds scale * relu(k.q)^2 @ vals into `acc` ([Dv, L, B] when !permuted, else
    // acc is [Dv, B, L] logically and chunk outputs are permuted [Dv, L, B] -> [Dv, B, L]).
    ggml_tensor* relu2_attn_acc(ggml_tensor* acc, ggml_tensor* q, ggml_tensor* k, ggml_tensor* vals, ggml_tensor* mask,
                                float scale, bool permuted) {
        const int64_t d = q->ne[0], L = q->ne[1], B = q->ne[2], Dv = vals->ne[1];
        int64_t chunk = (int64_t)(kChunkBytes / ((size_t)L * (size_t)L * sizeof(float)));
        chunk = std::max<int64_t>(1, std::min<int64_t>(chunk, B));
        for (int64_t b0 = 0; b0 < B; b0 += chunk) {
            const int64_t bc = std::min(chunk, B - b0);
            ggml_tensor* qc = ggml_view_3d(ctx, q, d, L, bc, q->nb[1], q->nb[2], b0 * q->nb[2]);
            ggml_tensor* kc = ggml_view_3d(ctx, k, d, L, bc, k->nb[1], k->nb[2], b0 * k->nb[2]);
            ggml_tensor* vc = ggml_view_3d(ctx, vals, L, Dv, bc, vals->nb[1], vals->nb[2], b0 * vals->nb[2]);
            ggml_tensor* s = ggml_mul_mat(ctx, kc, qc); // [Lk, Lq, bc]
            s = ggml_relu_inplace(ctx, s);
            s = ggml_sqr_inplace(ctx, s);
            if (mask)
                s = ggml_mul_inplace(ctx, s, mask);
            ggml_tensor* o = ggml_mul_mat(ctx, vc, s); // [Dv, Lq, bc]
            o = ggml_scale(ctx, o, scale);
            if (!permuted) {
                acc = ggml_acc_inplace(ctx, acc, o, acc->nb[1], acc->nb[2], acc->nb[3], b0 * acc->nb[2]);
            } else {
                // o is [Dv, Lq, bc] over (seq = Lq, rows = bc); acc is [Dv, rows_total(L of acc), seq]
                // here: acc logical [Dv, n, R] with o's Lq = R (rows) and bc = n-chunk.
                ggml_tensor* op = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)); // [Dv, bc, Lq]
                acc = ggml_acc_inplace(ctx, acc, op, acc->nb[1], acc->nb[2], acc->nb[3], b0 * acc->nb[1]);
            }
        }
        return acc;
    }

    ggml_tensor* rope(ggml_tensor* x, ggml_tensor* pos) {
        const int64_t d = x->ne[0], n = x->ne[1], R = x->ne[2];
        ggml_tensor* x4 = ggml_reshape_4d(ctx, x, d, 1, n, R);
        x4 = ggml_rope_ext(ctx, x4, pos, nullptr, hp.rot, GGML_ROPE_TYPE_NORMAL, 0, hp.theta, 1.0f, 0.0f, 1.0f, 0.0f,
                           0.0f);
        return ggml_reshape_3d(ctx, x4, d, n, R);
    }

    // MossFormer (gated attention unit) on x [64, n, R]: sequence n, rows R.
    ggml_tensor* mossformer(ggml_tensor* x, const mossformer_w& w, ggml_tensor* pos, ggml_tensor* offdiag_rows,
                            bool dbg) {
        const int64_t C = x->ne[0], n = x->ne[1], R = x->ne[2];
        const int64_t half = C / 2;
        // token shift of the first half of the channels by one step along n
        ggml_tensor* xs = ggml_view_3d(ctx, x, half, n, R, x->nb[1], x->nb[2], 0);
        xs = ggml_pad_ext(ctx, xs, 0, 0, 1, 0, 0, 0, 0, 0); // [half, n+1, R]
        xs = ggml_view_3d(ctx, xs, half, n, R, xs->nb[1], xs->nb[2], 0);
        ggml_tensor* xp = ggml_view_3d(ctx, x, C - half, n, R, x->nb[1], x->nb[2], half * x->nb[0]);
        ggml_tensor* normed = ggml_concat(ctx, ggml_cont(ctx, xs), ggml_cont(ctx, xp), 0);

        ggml_tensor* hid = ffconvm(normed, w.hidden); // [256, n, R] = (v | u)
        ggml_tensor* qk = ffconvm(normed, w.qk);      // [128, n, R]
        if (dbg) {
            tap("stage_b0_intra_mf_hidden", hid);
            tap("stage_b0_intra_mf_qk", qk);
        }
        const int64_t D = qk->ne[0], H2 = hid->ne[0], H = H2 / 2;
        ggml_tensor* qkh[4];
        for (int h = 0; h < 4; h++) {
            ggml_tensor* g = ggml_view_1d(ctx, w.os_gamma, D, h * w.os_gamma->nb[1]);
            ggml_tensor* b = ggml_view_1d(ctx, w.os_beta, D, h * w.os_beta->nb[1]);
            qkh[h] = rope(ggml_add(ctx, ggml_mul(ctx, qk, g), b), pos);
        }
        ggml_tensor *quad_q = qkh[0], *lin_q = qkh[1], *quad_k = qkh[2], *lin_k = qkh[3];

        ggml_tensor* hT = ggml_cont(ctx, ggml_transpose(ctx, hid)); // [n, 256, R]
        // linear attention: kv[d, e] = sum_n k[d, n] h[e, n] / n ; out[e, i] = sum_d kv[d, e] q[d, i]
        ggml_tensor* kT = ggml_cont(ctx, ggml_transpose(ctx, lin_k)); // [n, 128, R]
        ggml_tensor* kv = ggml_mul_mat(ctx, kT, hT);                  // [128, 256, R]
        ggml_tensor* att = ggml_mul_mat(ctx, kv, lin_q);              // [256, n, R]
        att = ggml_scale(ctx, att, 1.0f / (float)n);
        // quadratic attention along n within each row
        att = relu2_attn_acc(att, quad_q, quad_k, hT, nullptr, 1.0f / ((float)n * (float)n), false);
        // cross attention across rows for each position (diagonal masked)
        ggml_tensor* qc = ggml_cont(ctx, ggml_permute(ctx, quad_q, 0, 2, 1, 3)); // [128, R, n]
        ggml_tensor* kc = ggml_cont(ctx, ggml_permute(ctx, quad_k, 0, 2, 1, 3));
        ggml_tensor* hc = ggml_cont(ctx, ggml_permute(ctx, hid, 1, 2, 0, 3)); // [R, 256, n]
        att = relu2_attn_acc(att, qc, kc, hc, offdiag_rows, 1.0f / ((float)R * (float)R), true);

        ggml_tensor* att_v = ggml_view_3d(ctx, att, H, n, R, att->nb[1], att->nb[2], 0);
        ggml_tensor* att_u = ggml_view_3d(ctx, att, H, n, R, att->nb[1], att->nb[2], H * att->nb[0]);
        ggml_tensor* v = ggml_view_3d(ctx, hid, H, n, R, hid->nb[1], hid->nb[2], 0);
        ggml_tensor* u = ggml_view_3d(ctx, hid, H, n, R, hid->nb[1], hid->nb[2], H * hid->nb[0]);
        ggml_tensor* gate = ggml_sigmoid(ctx, ggml_mul(ctx, att_v, u));
        ggml_tensor* out = ggml_mul(ctx, ggml_mul(ctx, att_u, v), gate);
        return ggml_add(ctx, x, ffconvm(out, w.out));
    }

    // SELayer: x * (mlp_avg(mean) + mlp_max(max)), x [C, Q, T].
    ggml_tensor* se(ggml_tensor* x, const se_w& s) {
        const int64_t C = x->ne[0], N = x->ne[1] * x->ne[2];
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, C, N))); // [N, C]
        ggml_tensor* avg = ggml_reshape_1d(ctx, ggml_mean(ctx, xt), C);
        ggml_tensor* mx =
            ggml_pool_2d(ctx, ggml_reshape_3d(ctx, xt, N, 1, C), GGML_OP_POOL_MAX, (int)N, 1, (int)N, 1, 0.0f, 0.0f);
        mx = ggml_reshape_1d(ctx, mx, C);
        ggml_tensor* sa = ggml_sigmoid(ctx, linear(ggml_relu(ctx, linear(avg, s.a0w, s.a0b)), s.a2w, s.a2b));
        ggml_tensor* sm = ggml_sigmoid(ctx, linear(ggml_relu(ctx, linear(mx, s.m0w, s.m0b)), s.m2w, s.m2b));
        return ggml_mul(ctx, x, ggml_add(ctx, sa, sm));
    }

    // Shared intra/inter branch body after the pair unfold: u/v FFConvM, FSMN,
    // gating, ConvTranspose1d(k=2) back to L+1 positions, MossFormer.
    ggml_tensor* branch_core(ggml_tensor* y, const branch_w& b, ggml_tensor* pos, ggml_tensor* offdiag_rows,
                             bool dbg_intra, bool dbg_inter) {
        ggml_tensor* u = ffconvm(y, b.to_u);
        ggml_tensor* v = ffconvm(y, b.to_v);
        if (dbg_intra) {
            tap("stage_b0_intra_to_u", u);
            tap("stage_b0_intra_to_v", v);
        }
        if (dbg_inter)
            tap("stage_b0_inter_to_u", u);
        u = fsmn(u, b.rnn);
        if (dbg_intra)
            tap("stage_b0_intra_rnn", u);
        if (dbg_inter)
            tap("stage_b0_inter_rnn", u);
        ggml_tensor* z = ggml_mul(ctx, v, u); // [128, L, R]
        ggml_tensor* w0 = ggml_view_2d(ctx, b.lin_w, b.lin_w->ne[0], b.lin_w->ne[1], b.lin_w->nb[1], 0);
        ggml_tensor* w1 = ggml_view_2d(ctx, b.lin_w, b.lin_w->ne[0], b.lin_w->ne[1], b.lin_w->nb[1], b.lin_w->nb[2]);
        ggml_tensor* y0 = ggml_mul_mat(ctx, w0, z); // [64, L, R]
        ggml_tensor* y1 = ggml_mul_mat(ctx, w1, z);
        y0 = ggml_pad_ext(ctx, y0, 0, 0, 0, 1, 0, 0, 0, 0);
        y1 = ggml_pad_ext(ctx, y1, 0, 0, 1, 0, 0, 0, 0, 0);
        ggml_tensor* o = ggml_add(ctx, ggml_add(ctx, y0, y1), b.lin_b); // [64, L+1, R]
        if (dbg_intra)
            tap("stage_b0_intra_linear", o);
        if (dbg_inter)
            tap("stage_b0_inter_linear", o);
        o = mossformer(o, b.mf, pos, offdiag_rows, dbg_intra);
        if (dbg_intra)
            tap("stage_b0_intra_mossformer", o);
        if (dbg_inter)
            tap("stage_b0_inter_mossformer", o);
        return o;
    }

    // Pair "unfold" along ne1: [C, L, R] -> [2C, L-1, R], channel 2c+j = x[c, l+j].
    ggml_tensor* pair_unfold(ggml_tensor* x) {
        const int64_t C = x->ne[0], L = x->ne[1], R = x->ne[2];
        ggml_tensor* x0 = ggml_cont(ctx, ggml_view_3d(ctx, x, C, L - 1, R, x->nb[1], x->nb[2], 0));
        ggml_tensor* x1 = ggml_cont(ctx, ggml_view_3d(ctx, x, C, L - 1, R, x->nb[1], x->nb[2], x->nb[1]));
        ggml_tensor* c = ggml_concat(ctx, ggml_reshape_4d(ctx, x0, 1, C, L - 1, R),
                                     ggml_reshape_4d(ctx, x1, 1, C, L - 1, R), 0); // [2, C, L-1, R]
        return ggml_reshape_3d(ctx, c, 2 * C, L - 1, R);
    }

    // LayerNormalization4DCF over (E, Q) per frame; x [E, Q, T], g/b [E, Q].
    ggml_tensor* ln4dcf(ggml_tensor* x, ggml_tensor* g, ggml_tensor* b) {
        const int64_t E = x->ne[0], Q = x->ne[1], T = x->ne[2];
        ggml_tensor* y = ggml_norm(ctx, ggml_reshape_2d(ctx, x, E * Q, T), hp.eps);
        y = ggml_reshape_3d(ctx, y, E, Q, T);
        return ggml_add(ctx, ggml_mul(ctx, y, g), b);
    }

    ggml_tensor* attn_conv(ggml_tensor* x, const attn_head_w& h) {
        return ln4dcf(prelu(linear(x, h.w, h.b), h.prelu), h.g, h.beta);
    }

    ggml_tensor* sync_block(ggml_tensor* x, const block_w& b, bool dbg) {
        const int64_t C = x->ne[0], Q = x->ne[1], T = x->ne[2];
        // ---- intra (sequence along frequency, rows = frames)
        ggml_tensor* h = layer_norm(x, b.intra.norm_g, b.intra.norm_b);
        if (dbg)
            tap("stage_b0_intra_norm", h);
        {
            // Fconv: grouped (1,2) conv, out channel 2c+j = w[2c+j,0] x[c,q] + w[2c+j,1] x[c,q+1] + b
            ggml_tensor* x0 = ggml_cont(ctx, ggml_view_3d(ctx, h, C, Q - 1, T, h->nb[1], h->nb[2], 0));
            ggml_tensor* x1 = ggml_cont(ctx, ggml_view_3d(ctx, h, C, Q - 1, T, h->nb[1], h->nb[2], h->nb[1]));
            ggml_tensor* d0 = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, x0, 1, C, Q - 1, T), 2, C, Q - 1, T);
            ggml_tensor* d1 = ggml_repeat_4d(ctx, ggml_reshape_4d(ctx, x1, 1, C, Q - 1, T), 2, C, Q - 1, T);
            ggml_tensor* w0 = ggml_reshape_2d(ctx, ggml_view_1d(ctx, b.fconv_w, 2 * C, 0), 2, C);
            ggml_tensor* w1 = ggml_reshape_2d(ctx, ggml_view_1d(ctx, b.fconv_w, 2 * C, b.fconv_w->nb[1]), 2, C);
            ggml_tensor* y = ggml_add(ctx, ggml_mul(ctx, d0, w0), ggml_mul(ctx, d1, w1));
            y = ggml_add(ctx, ggml_reshape_3d(ctx, y, 2 * C, Q - 1, T), b.fconv_b);
            ggml_tensor* o = branch_core(y, b.intra, pos_q, offdiag_t, dbg, false); // [64, Q, T]
            o = se(o, b.intra.se);
            if (dbg)
                tap("stage_b0_intra_se", o);
            x = ggml_add(ctx, o, x);
        }
        // ---- inter (sequence along time, rows = frequencies)
        h = layer_norm(x, b.inter.norm_g, b.inter.norm_b);
        if (dbg)
            tap("stage_b0_inter_norm", h);
        {
            ggml_tensor* ht = ggml_cont(ctx, ggml_permute(ctx, h, 0, 2, 1, 3));     // [64, T, Q]
            ggml_tensor* y = pair_unfold(ht);                                       // [128, T-1, Q]
            ggml_tensor* o = branch_core(y, b.inter, pos_t, offdiag_q, false, dbg); // [64, T, Q]
            o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                   // [64, Q, T]
            o = se(o, b.inter.se);
            if (dbg)
                tap("stage_b0_inter_se", o);
            x = ggml_add(ctx, o, x);
        }
        // ---- full-band attention across frames
        std::vector<ggml_tensor*> heads;
        for (int hh = 0; hh < hp.n_head; hh++) {
            ggml_tensor* qh = attn_conv(x, b.q[hh]); // [E, Q, T]
            ggml_tensor* kh = attn_conv(x, b.k[hh]);
            ggml_tensor* vh = attn_conv(x, b.v[hh]); // [Cv, Q, T]
            const int64_t E = qh->ne[0], Cv = vh->ne[0];
            ggml_tensor* s = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, kh, E * Q, T), ggml_reshape_2d(ctx, qh, E * Q, T));
            s = ggml_soft_max_ext(ctx, s, nullptr, 1.0f / std::sqrt((float)(E * Q)), 0.0f);             // [Tk, Tq]
            ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, vh, Cv * Q, T))); // [T, CvQ]
            ggml_tensor* o = ggml_mul_mat(ctx, vt, s);                                                  // [CvQ, Tq]
            heads.push_back(ggml_reshape_3d(ctx, o, Cv, Q, T));
        }
        ggml_tensor* cat = heads[0];
        for (size_t i = 1; i < heads.size(); i++)
            cat = ggml_concat(ctx, cat, heads[i], 0);
        ggml_tensor* p = attn_conv(cat, b.proj);
        if (dbg)
            tap("stage_b0_attn_concat_proj", p);
        return ggml_add(ctx, p, x);
    }
};

struct graph_io {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor* input = nullptr; // [3, F, T]
    ggml_tensor* pos_q = nullptr;
    ggml_tensor* pos_t = nullptr;
    ggml_tensor* offdiag_q = nullptr;
    ggml_tensor* offdiag_t = nullptr;
    ggml_tensor* mask = nullptr; // [F, T]
    ggml_tensor* cplx = nullptr; // [2, F, T]
    std::vector<std::pair<std::string, ggml_tensor*>> taps;
};

void build_graph(mossformergan_se_context* c, graph_io& io, int T, bool debug) {
    const mfg_model& m = c->m;
    const mfg_hparams& hp = m.hp;
    const size_t mem = ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false);
    ggml_init_params ip = {mem, nullptr, true};
    io.ctx = ggml_init(ip);
    ggml_context* ctx = io.ctx;
    io.gf = ggml_new_graph_custom(ctx, kGraphSize, false);

    const int F = hp.F;
    const int Q = (F + 2 - 3) / 2 + 1; // conv_2 stride-2 output
    io.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, F, T);
    ggml_set_input(io.input);
    io.pos_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Q);
    ggml_set_input(io.pos_q);
    io.pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(io.pos_t);
    io.offdiag_q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Q, Q);
    ggml_set_input(io.offdiag_q);
    io.offdiag_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_input(io.offdiag_t);

    graph_builder g{ctx, hp};
    g.pos_q = io.pos_q;
    g.pos_t = io.pos_t;
    g.offdiag_q = io.offdiag_q;
    g.offdiag_t = io.offdiag_t;
    if (debug)
        g.taps = &io.taps;

    // ---- DenseEncoder
    ggml_tensor* w1 = ggml_view_2d(ctx, m.enc_c1_w, m.enc_c1_w->ne[0], m.enc_c1_w->ne[1], m.enc_c1_w->nb[1], 0);
    ggml_tensor* x = ggml_mul_mat(ctx, w1, ggml_reshape_2d(ctx, io.input, 3, (int64_t)F * T));
    x = ggml_add(ctx, ggml_reshape_3d(ctx, x, hp.C, F, T), m.enc_c1_b);
    x = g.prelu(g.instance_norm(x, m.enc_c1_nw, m.enc_c1_nb), m.enc_c1_a);
    g.tap("stage_enc_conv1", x);
    x = g.dense_net(x, m.enc_dense);
    g.tap("stage_enc_dense", x);
    x = g.conv_1xk(x, m.enc_c2_w, m.enc_c2_b, 3, 1, 1, 2); // [64, Q, T]
    x = g.prelu(g.instance_norm(x, m.enc_c2_nw, m.enc_c2_nb), m.enc_c2_a);
    g.tap("stage_enc", x);

    // ---- SyncANet blocks
    static const char* block_names[] = {"stage_block0", "stage_block1", "stage_block2", "stage_block3",
                                        "stage_block4", "stage_block5", "stage_block6", "stage_block7"};
    for (int i = 0; i < hp.n_layers; i++) {
        x = g.sync_block(x, m.blocks[i], debug && i == 0);
        if (i < 8)
            g.tap(block_names[i], x);
    }

    // ---- MaskDecoder
    ggml_tensor* md = g.dense_net(x, m.mask_dense);
    g.tap("stage_mask_dense", md);
    md = g.conv_1xk(md, m.mask_sp_w, m.mask_sp_b, 3, 1, 1, 1); // [128, Q, T]
    md = ggml_reshape_3d(ctx, md, hp.C, 2 * Q, T);             // sub-pixel shuffle
    md = g.conv_1xk(md, m.mask_c1_w, m.mask_c1_b, 2, 0, 0, 1); // [1, F, T]
    {
        ggml_tensor* flat = ggml_reshape_1d(ctx, md, (int64_t)F * T);
        flat = ggml_norm(ctx, flat, hp.eps);
        flat = ggml_add(ctx, ggml_mul(ctx, flat, m.mask_nw), m.mask_nb);
        flat = g.prelu(flat, m.mask_a);
        ggml_tensor* fw = ggml_reshape_1d(ctx, m.mask_fw, 1);
        flat = ggml_add(ctx, ggml_mul(ctx, flat, fw), m.mask_fb);
        md = g.prelu(ggml_reshape_2d(ctx, flat, F, T), m.mask_out_a);
    }
    io.mask = md;
    ggml_set_output(io.mask);

    // ---- ComplexDecoder
    ggml_tensor* cd = g.dense_net(x, m.cplx_dense);
    cd = g.conv_1xk(cd, m.cplx_sp_w, m.cplx_sp_b, 3, 1, 1, 1);
    cd = ggml_reshape_3d(ctx, cd, hp.C, 2 * Q, T);
    cd = g.prelu(g.instance_norm(cd, m.cplx_nw, m.cplx_nb), m.cplx_a);
    cd = g.conv_1xk(cd, m.cplx_c_w, m.cplx_c_b, 2, 0, 0, 1); // [2, F, T]
    io.cplx = cd;
    ggml_set_output(io.cplx);

    for (auto& t : io.taps)
        ggml_build_forward_expand(io.gf, t.second);
    ggml_build_forward_expand(io.gf, io.mask);
    ggml_build_forward_expand(io.gf, io.cplx);
}

// Run the network on CL input [3, F, T]. Returns mask [F, T] and complex [2, F, T].
bool run_network(mossformergan_se_context* c, const std::vector<float>& input, int T, std::vector<float>& mask,
                 std::vector<float>& cplx, std::vector<std::pair<std::string, std::vector<float>>>* taps,
                 std::vector<std::vector<int64_t>>* tap_ne) {
    const mfg_hparams& hp = c->m.hp;
    graph_io io;
    build_graph(c, io, T, taps != nullptr);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, io.gf)) {
        std::fprintf(stderr, "mossformergan_se: graph allocation failed (T=%d)\n", T);
        ggml_free(io.ctx);
        return false;
    }
    const int F = hp.F;
    const int Q = (int)io.pos_q->ne[0];
    ggml_backend_tensor_set(io.input, input.data(), 0, input.size() * sizeof(float));
    std::vector<int32_t> pq(Q), pt(T);
    for (int i = 0; i < Q; i++)
        pq[i] = i;
    for (int i = 0; i < T; i++)
        pt[i] = i;
    ggml_backend_tensor_set(io.pos_q, pq.data(), 0, pq.size() * sizeof(int32_t));
    ggml_backend_tensor_set(io.pos_t, pt.data(), 0, pt.size() * sizeof(int32_t));
    std::vector<float> od((size_t)std::max(Q, T) * std::max(Q, T));
    for (int i = 0; i < Q; i++)
        for (int j = 0; j < Q; j++)
            od[(size_t)i * Q + j] = i == j ? 0.0f : 1.0f;
    ggml_backend_tensor_set(io.offdiag_q, od.data(), 0, (size_t)Q * Q * sizeof(float));
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
            od[(size_t)i * T + j] = i == j ? 0.0f : 1.0f;
    ggml_backend_tensor_set(io.offdiag_t, od.data(), 0, (size_t)T * T * sizeof(float));

    const char* prof_env = std::getenv("MOSSFORMERGAN_SE_PROFILE");
    op_profile prof;
    if (prof_env && prof_env[0] == '1')
        ggml_backend_sched_set_eval_callback(c->sched, profile_cb, &prof);
    const bool ok = ggml_backend_sched_graph_compute(c->sched, io.gf) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_set_eval_callback(c->sched, nullptr, nullptr);
    if (prof_env && prof_env[0] == '1') {
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& kv : prof.ms) {
            v.emplace_back(kv.second, kv.first);
            tot += kv.second;
        }
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "mossformergan_se profile (T=%d): %.1f ms\n", T, tot);
        for (size_t i = 0; i < v.size() && i < 25; i++)
            std::fprintf(stderr, "  %8.1f ms %5d  %s\n", v[i].first, prof.count[v[i].second], v[i].second.c_str());
    }
    if (!ok) {
        std::fprintf(stderr, "mossformergan_se: graph compute failed\n");
        ggml_free(io.ctx);
        return false;
    }
    mask.resize((size_t)F * T);
    cplx.resize((size_t)2 * F * T);
    ggml_backend_tensor_get(io.mask, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_get(io.cplx, cplx.data(), 0, cplx.size() * sizeof(float));
    if (taps) {
        for (auto& t : io.taps) {
            std::vector<float> d(ggml_nelements(t.second));
            ggml_backend_tensor_get(t.second, d.data(), 0, d.size() * sizeof(float));
            taps->emplace_back(t.first, std::move(d));
            tap_ne->push_back({t.second->ne[0], t.second->ne[1], t.second->ne[2], t.second->ne[3]});
        }
    }
    ggml_free(io.ctx);
    return true;
}

// ---------------------------------------------------------------------------
// STFT front/back end (torch.stft/istft, center=True reflect, periodic Hamming)
// ---------------------------------------------------------------------------

void init_tables(mossformergan_se_context* c) {
    const int N = c->m.hp.n_fft, F = c->m.hp.F;
    c->window.resize(N);
    const double pi = std::acos(-1.0);
    for (int n = 0; n < N; n++)
        c->window[n] = (float)(0.54 - 0.46 * std::cos(2.0 * pi * n / N));
    c->cos_t.resize((size_t)F * N);
    c->sin_t.resize((size_t)F * N);
    for (int k = 0; k < F; k++)
        for (int n = 0; n < N; n++) {
            const long long kn = ((long long)k * n) % N;
            const double a = 2.0 * pi * (double)kn / N;
            c->cos_t[(size_t)k * N + n] = (float)std::cos(a);
            c->sin_t[(size_t)k * N + n] = (float)std::sin(a);
        }
}

template <typename Fn> void parallel_for(int n, int n_threads, Fn fn) {
    n_threads = std::max(1, std::min(n_threads, n));
    if (n_threads == 1) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> th;
    const int per = (n + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        const int a = t * per, b = std::min(n, a + per);
        if (a < b)
            th.emplace_back([=] { fn(a, b); });
    }
    for (auto& t : th)
        t.join();
}

// One upstream _decode_one_audio_mossformergan_se_16k call.
bool decode_window(mossformergan_se_context* c, const float* in, int len, double norm_factor, std::vector<float>& out,
                   std::vector<float>* model_input = nullptr) {
    const mfg_hparams& hp = c->m.hp;
    const int N = hp.n_fft, hop = hp.hop, F = hp.F, half = N / 2;
    const int nframe = (len + hop - 1) / hop;
    const int padded_len = nframe * hop;
    // pad with the beginning of the input (torch.cat([x, x[:, :pad]]))
    std::vector<float> x(padded_len);
    for (int i = 0; i < len; i++)
        x[i] = (float)((double)in[i] * norm_factor);
    for (int i = len; i < padded_len; i++)
        x[i] = x[std::min(i - len, len - 1)];
    // reflect pad by n_fft/2 (needs len > half; short inputs are zero-extended by the caller)
    const int L = padded_len + 2 * half;
    std::vector<float> xp(L);
    for (int i = 0; i < L; i++) {
        int j = i - half;
        if (j < 0)
            j = -j;
        if (j >= padded_len)
            j = 2 * (padded_len - 1) - j;
        xp[i] = x[std::max(0, std::min(j, padded_len - 1))];
    }
    const int T = padded_len / hop + 1;
    std::vector<float> re((size_t)F * T), im((size_t)F * T); // compressed spectrum [T][F]
    parallel_for(T, c->n_threads, [&](int t0, int t1) {
        std::vector<float> fr(N);
        for (int t = t0; t < t1; t++) {
            for (int n = 0; n < N; n++)
                fr[n] = xp[(size_t)t * hop + n] * c->window[n];
            for (int k = 0; k < F; k++) {
                const float* ct = &c->cos_t[(size_t)k * N];
                const float* st = &c->sin_t[(size_t)k * N];
                double sr = 0.0, si = 0.0;
                for (int n = 0; n < N; n++) {
                    sr += fr[n] * ct[n];
                    si -= fr[n] * st[n];
                }
                const double mag = std::sqrt(sr * sr + si * si);
                double s = 0.0;
                if (mag > 0.0)
                    s = std::pow(mag, (double)hp.compress) / mag;
                re[(size_t)t * F + k] = (float)(sr * s);
                im[(size_t)t * F + k] = (float)(si * s);
            }
        }
    });
    // model input CL [3, F, T]: (|X_c|, Re, Im)
    std::vector<float> inp((size_t)3 * F * T);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < F; k++) {
            const float r = re[(size_t)t * F + k], i = im[(size_t)t * F + k];
            float* d = &inp[((size_t)t * F + k) * 3];
            d[0] = std::sqrt(r * r + i * i);
            d[1] = r;
            d[2] = i;
        }
    if (model_input) {
        *model_input = inp;
        return true;
    }
    std::vector<float> mask, cplx;
    if (!run_network(c, inp, T, mask, cplx, nullptr, nullptr))
        return false;

    // combine, uncompress, inverse DFT, overlap-add
    const double inv_c = 1.0 / (double)hp.compress;
    std::vector<float> yr((size_t)T * N, 0.0f);
    parallel_for(T, c->n_threads, [&](int t0, int t1) {
        std::vector<double> sr(F), si(F);
        for (int t = t0; t < t1; t++) {
            for (int k = 0; k < F; k++) {
                const size_t ci = (size_t)t * F + k;
                const float* d = &inp[ci * 3];
                const double mag = d[0];
                double cs = 1.0, sn = 0.0;
                if (mag > 0.0) {
                    cs = d[1] / mag;
                    sn = d[2] / mag;
                }
                const double om = (double)mask[ci] * mag;
                const double fr = om * cs + cplx[ci * 2 + 0];
                const double fi = om * sn + cplx[ci * 2 + 1];
                const double m = std::sqrt(fr * fr + fi * fi);
                double s = 0.0;
                if (m > 0.0)
                    s = std::pow(m, inv_c) / m;
                sr[k] = fr * s;
                si[k] = fi * s;
            }
            float* y = &yr[(size_t)t * N];
            for (int n = 0; n < N; n++) {
                double acc = sr[0] + ((n & 1) ? -sr[F - 1] : sr[F - 1]);
                for (int k = 1; k < F - 1; k++) {
                    const size_t ti = (size_t)k * N + n;
                    acc += 2.0 * (sr[k] * c->cos_t[ti] - si[k] * c->sin_t[ti]);
                }
                y[n] = (float)(acc / N) * c->window[n];
            }
        }
    });
    std::vector<double> ola((size_t)(T - 1) * hop + N, 0.0), env((size_t)(T - 1) * hop + N, 0.0);
    for (int t = 0; t < T; t++)
        for (int n = 0; n < N; n++) {
            ola[(size_t)t * hop + n] += yr[(size_t)t * N + n];
            env[(size_t)t * hop + n] += (double)c->window[n] * c->window[n];
        }
    out.resize(len);
    for (int i = 0; i < len; i++) {
        const size_t j = (size_t)i + half;
        const double e = env[j];
        const double v = e > 1e-11 ? ola[j] / e : 0.0;
        out[i] = (float)(v / norm_factor);
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

mossformergan_se_context* mossformergan_se_init(const char* path, int n_threads, bool verbose) {
    auto* c = new mossformergan_se_context();
    c->n_threads = n_threads > 0 ? n_threads : 4;
    c->verbose = verbose;
    c->cpu = core_cpu_backend::init();
    if (!c->cpu) {
        delete c;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(c->cpu, c->n_threads);
    c->backend = c->cpu;
    const char* gpu = std::getenv("CRISPASR_MOSSFORMERGAN_SE_GPU");
    if (gpu && gpu[0] && gpu[0] != '0') {
        ggml_backend_t b = crispasr_init_gpu_backend();
        if (b && !core_cpu_backend::is_cpu(b))
            c->backend = b;
        else if (b && b != c->cpu)
            ggml_backend_free(b);
    }
    if (!load_model(c->m, path, c->backend)) {
        mossformergan_se_free(c);
        return nullptr;
    }
    ggml_backend_t backends[2] = {c->backend, c->cpu};
    const int nb = c->backend == c->cpu ? 1 : 2;
    c->sched = ggml_backend_sched_new(backends, nullptr, nb, kGraphSize, false, false);
    if (!c->sched) {
        mossformergan_se_free(c);
        return nullptr;
    }
    init_tables(c);
    if (verbose)
        std::fprintf(stderr, "mossformergan_se: loaded %s (%zu tensors, backend %s, %d threads)\n", path,
                     c->m.wl.tensors.size(), ggml_backend_name(c->backend), c->n_threads);
    return c;
}

void mossformergan_se_free(mossformergan_se_context* c) {
    if (!c)
        return;
    if (c->sched)
        ggml_backend_sched_free(c->sched);
    core_gguf::free_weights(c->m.wl);
    if (c->backend && c->backend != c->cpu)
        ggml_backend_free(c->backend);
    if (c->cpu)
        ggml_backend_free(c->cpu);
    delete c;
}

int mossformergan_se_sample_rate(const mossformergan_se_context* c) {
    return c ? c->m.hp.sample_rate : 16000;
}

bool mossformergan_se_enhance(mossformergan_se_context* c, const float* in, int n, std::vector<float>& out) {
    if (!c || !in || n <= 0)
        return false;
    const mfg_hparams& hp = c->m.hp;
    // norm_factor = sqrt(N / sum(x^2)) over the whole input (upstream decode wrapper)
    double ss = 0.0;
    for (int i = 0; i < n; i++)
        ss += (double)in[i] * in[i];
    out.assign(n, 0.0f);
    if (!(ss > 0.0) || !std::isfinite(ss))
        return true; // digital silence (upstream would produce NaN)
    const double norm_factor = std::sqrt((double)n / ss);

    // Inputs too short for the centred reflect-padded STFT are zero-extended.
    const int min_len = hp.n_fft / 2 + 1;
    auto run = [&](const float* x, int len, std::vector<float>& y) -> bool {
        if (len >= min_len)
            return decode_window(c, x, len, norm_factor, y);
        std::vector<float> ext(min_len, 0.0f);
        std::copy(x, x + len, ext.begin());
        if (!decode_window(c, ext.data(), min_len, norm_factor, y))
            return false;
        y.resize(len);
        return true;
    };

    const int window = hp.sample_rate * hp.decode_window;
    const int stride = (int)(window * 0.75);
    if (n <= hp.sample_rate * hp.one_time) {
        std::vector<float> y;
        if (!run(in, n, y))
            return false;
        std::copy(y.begin(), y.begin() + n, out.begin());
        return true;
    }
    const int give_up = (window - stride) / 2;
    int cur = 0;
    std::vector<float> y;
    while (cur + window <= n) {
        if (!run(in + cur, window, y))
            return false;
        if (cur == 0)
            std::copy(y.begin(), y.begin() + (window - give_up), out.begin());
        else
            std::copy(y.begin() + give_up, y.begin() + (window - give_up), out.begin() + cur + give_up);
        cur += stride;
    }
    if (cur < n) {
        const int last = cur - give_up;
        if (!run(in + last, n - last, y))
            return false;
        std::copy(y.begin() + give_up, y.end(), out.begin() + cur);
    }
    return true;
}

bool mossformergan_se_forward_stages(mossformergan_se_context* c, const float* spec, int T,
                                     std::vector<mossformergan_se_stage>& stages) {
    if (!c || !spec || T <= 1)
        return false;
    const int F = c->m.hp.F;
    // torch [2, T, F] -> CL [3, F, T]
    std::vector<float> inp((size_t)3 * F * T);
    for (int t = 0; t < T; t++)
        for (int k = 0; k < F; k++) {
            const float r = spec[(size_t)t * F + k], i = spec[(size_t)T * F + (size_t)t * F + k];
            float* d = &inp[((size_t)t * F + k) * 3];
            d[0] = std::sqrt(r * r + i * i);
            d[1] = r;
            d[2] = i;
        }
    std::vector<float> mask, cplx;
    std::vector<std::pair<std::string, std::vector<float>>> taps;
    std::vector<std::vector<int64_t>> ne;
    if (!run_network(c, inp, T, mask, cplx, &taps, &ne))
        return false;
    stages.clear();
    // CL [C, Q, T] -> torch [C, T, Q]
    auto cqt = [](const std::vector<float>& d, int64_t C, int64_t Q, int64_t TT) {
        std::vector<float> o(d.size());
        for (int64_t c0 = 0; c0 < C; c0++)
            for (int64_t t = 0; t < TT; t++)
                for (int64_t q = 0; q < Q; q++)
                    o[(c0 * TT + t) * Q + q] = d[(t * Q + q) * C + c0];
        return o;
    };
    for (size_t i = 0; i < taps.size(); i++) {
        const std::string& name = taps[i].first;
        const auto& n = ne[i];
        mossformergan_se_stage s;
        s.name = name;
        // MossFormer/FFConvM internals are stored in torch's folded [rows, seq, ch] layout already.
        const bool folded = name.find("_to_") != std::string::npos || name.find("_rnn") != std::string::npos ||
                            name.find("_mossformer") != std::string::npos || name.find("_mf_") != std::string::npos;
        if (folded) {
            s.shape = {n[2], n[1], n[0]};
            s.data = taps[i].second;
        } else if (name.find("_linear") != std::string::npos) {
            // ConvTranspose1d output torch [rows, C, L]; ours [C, L, rows]
            s.shape = {n[2], n[0], n[1]};
            s.data.resize(taps[i].second.size());
            for (int64_t r = 0; r < n[2]; r++)
                for (int64_t c0 = 0; c0 < n[0]; c0++)
                    for (int64_t l = 0; l < n[1]; l++)
                        s.data[(r * n[0] + c0) * n[1] + l] = taps[i].second[(r * n[1] + l) * n[0] + c0];
        } else {
            s.shape = {n[0], n[2], n[1]};
            s.data = cqt(taps[i].second, n[0], n[1], n[2]);
        }
        stages.push_back(std::move(s));
    }
    mossformergan_se_stage sm{"stage_mask", {T, F}, mask};
    stages.push_back(sm);
    mossformergan_se_stage sc{"stage_complex", {2, T, F}, cqt(cplx, 2, F, T)};
    stages.push_back(sc);
    // final (compressed-domain) real/imag
    mossformergan_se_stage sp{"stage_pred", {2, T, F}, std::vector<float>((size_t)2 * T * F)};
    for (int t = 0; t < T; t++)
        for (int k = 0; k < F; k++) {
            const size_t ci = (size_t)t * F + k;
            const float* d = &inp[ci * 3];
            float cs = 1.0f, sn = 0.0f;
            if (d[0] > 0.0f) {
                cs = d[1] / d[0];
                sn = d[2] / d[0];
            }
            const float om = mask[ci] * d[0];
            sp.data[ci] = om * cs + cplx[ci * 2];
            sp.data[(size_t)T * F + ci] = om * sn + cplx[ci * 2 + 1];
        }
    stages.push_back(sp);
    return true;
}

// ---------------------------------------------------------------------------
// crispasr_enhancer registration
// ---------------------------------------------------------------------------

static void* mfg_ops_init(const char* path, int n_threads, bool verbose) {
    return mossformergan_se_init(path, n_threads, verbose);
}
static int mfg_ops_rate(void* ctx) {
    return mossformergan_se_sample_rate((mossformergan_se_context*)ctx);
}
static bool mfg_ops_process(void* ctx, const float* in, int n, std::vector<float>& out) {
    return mossformergan_se_enhance((mossformergan_se_context*)ctx, in, n, out);
}
static void mfg_ops_free(void* ctx) {
    mossformergan_se_free((mossformergan_se_context*)ctx);
}

const CrispasrEnhanceModelOps* mossformergan_se_enhance_ops() {
    static const CrispasrEnhanceModelOps ops = {"mossformergan-se", "mossformergan-se", mfg_ops_init,
                                                mfg_ops_rate,       mfg_ops_process,    mfg_ops_free};
    return &ops;
}
