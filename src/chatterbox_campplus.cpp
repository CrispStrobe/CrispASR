// chatterbox_campplus.cpp — see chatterbox_campplus.h for the full pipeline
// contract. Implements both the Kaldi fbank front-end (Module 4 phase 1)
// and the CAMPPlus TDNN forward (phase 2) — pure CPU float math, no ggml
// graph. The forward runs on the order of seconds for an 11 s clip (one-shot
// at voice-clone time, not on the synthesis hot path).

#include "chatterbox_campplus.h"

#include "core/fft.h"
#include "core/kaldi_fbank.h"
#include "core/mel.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace chatterbox_campplus {

// ---------------------------------------------------------------------------
// Phase 1 — Kaldi fbank + per-utterance mean subtract
// ---------------------------------------------------------------------------

std::vector<float> compute_fbank(const float* pcm_16k, int n_samples, int& T_frames_out) {
    T_frames_out = 0;
    if (!pcm_16k || n_samples <= 0)
        return {};

    core_kaldi::FbankParams p;
    p.sample_rate = 16000;
    p.n_mels = 80;
    p.frame_length_ms = 25;
    p.frame_shift_ms = 10;
    p.low_freq = 20.0f;
    p.high_freq = 0.0f;
    p.preemph = 0.97f;
    p.remove_dc_offset = true;
    p.int16_scale = false;

    int T = 0;
    auto feats = core_kaldi::compute_fbank(pcm_16k, n_samples, p, T);
    if (feats.empty() || T <= 0)
        return {};

    const int n_mels = p.n_mels;
    std::vector<double> mean((size_t)n_mels, 0.0);
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++)
            mean[(size_t)m] += (double)feats[(size_t)t * (size_t)n_mels + (size_t)m];
    }
    const double inv_T = 1.0 / (double)T;
    for (int m = 0; m < n_mels; m++)
        mean[(size_t)m] *= inv_T;
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++)
            feats[(size_t)t * (size_t)n_mels + (size_t)m] -= (float)mean[(size_t)m];
    }

    T_frames_out = T;
    return feats;
}

// ---------------------------------------------------------------------------
// Phase 2 — CAMPPlus TDNN forward (pure CPU)
// ---------------------------------------------------------------------------

namespace {

constexpr float kBnEps = 1e-5f; // PyTorch nn.BatchNorm{1,2}d default

// Read any tensor (F32 / F16 / quantized) into a freshly-sized host F32
// vector. Mirrors the same helper used by chatterbox_ve.cpp.
static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return out;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
        return out;
    }
    const size_t raw = ggml_nbytes(t);
    std::vector<uint8_t> tmp(raw);
    ggml_backend_tensor_get(t, tmp.data(), 0, raw);
    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (tt && tt->to_float)
        tt->to_float(tmp.data(), out.data(), (int64_t)n);
    else
        std::fill(out.begin(), out.end(), 0.0f);
    return out;
}

// Pre-fold a BN's (mean, var, weight, bias) into (gamma_eff, bias_eff) so
// each BN reduces to a per-channel `y = x * gamma_eff + bias_eff`.
//
//   gamma_eff = weight / sqrt(var + eps)         (or 1/sqrt(var+eps) if affine=False)
//   bias_eff  = bias - mean * gamma_eff          (or -mean*gamma_eff if affine=False)
struct BNFolded {
    std::vector<float> gamma; // (C,)
    std::vector<float> beta;  // (C,)
};

static BNFolded fold_bn(ggml_tensor* m, ggml_tensor* v, ggml_tensor* w, ggml_tensor* b, int C, float eps = kBnEps) {
    BNFolded out;
    if (!m || !v) {
        // Affine-only or empty BN — should be unreachable; return identity.
        out.gamma.assign((size_t)C, 1.0f);
        out.beta.assign((size_t)C, 0.0f);
        return out;
    }
    auto mean = read_tensor_f32(m);
    auto var = read_tensor_f32(v);
    std::vector<float> weight, bias;
    if (w)
        weight = read_tensor_f32(w);
    if (b)
        bias = read_tensor_f32(b);

    out.gamma.assign((size_t)C, 0.0f);
    out.beta.assign((size_t)C, 0.0f);
    for (int c = 0; c < C; c++) {
        const float inv_std = 1.0f / std::sqrt(var[(size_t)c] + eps);
        const float g = w ? weight[(size_t)c] * inv_std : inv_std;
        const float bv = (b ? bias[(size_t)c] : 0.0f) - mean[(size_t)c] * g;
        out.gamma[(size_t)c] = g;
        out.beta[(size_t)c] = bv;
    }
    return out;
}

// Cached host weights and folded BN params per layer. Keeps all of
// CAMPPlus's ~815 source tensors in fast-access form.
struct ResBlockCache {
    std::vector<float> conv1_w; // (kH=3, kW=3, in, out) PyTorch (out, in, kH, kW) — stored as
                                // GGUF flat (kw, kh, in, out) row-major; we'll re-interpret
                                // for the CPU conv2d below
    int conv1_in = 0;
    int conv1_out = 0;
    std::vector<float> conv1_b;
    BNFolded bn1;
    std::vector<float> conv2_w;
    int conv2_in = 0;
    int conv2_out = 0;
    std::vector<float> conv2_b;
    BNFolded bn2;
    bool has_shortcut = false;
    std::vector<float> sc_w; // (1, 1, in, out)
    int sc_in = 0;
    int sc_out = 0;
    std::vector<float> sc_b;
    BNFolded sc_bn;
    int stride = 1;
};

struct DenseLayerCache {
    int in_channels = 0;
    BNFolded nonl1_bn;       // (in_channels,)
    std::vector<float> l1_w; // (1, in, bn) row-major in GGUF; stored flat
    std::vector<float> l1_b;
    int bn_channels = 0;
    BNFolded nonl2_bn;           // (bn_channels,)
    std::vector<float> cam_ll_w; // (k=3, bn, out=growth) row-major
    int cam_out = 0;
    int cam_k = 3;
    int dilation = 1;
    int padding = 0;
    std::vector<float> cam_l1_w; // (1, bn, bn/2)
    std::vector<float> cam_l1_b;
    std::vector<float> cam_l2_w; // (1, bn/2, growth)
    std::vector<float> cam_l2_b;
};

struct UnitCache {
    std::vector<float> lin_w; // (kw, in, out)
    std::vector<float> lin_b;
    int kw = 1;
    int in_dim = 0;
    int out_dim = 0;
    BNFolded bn;
    bool has_bn = false;
};

struct CampplusCache {
    bool initialised = false;
    // FCM head
    std::vector<float> head_conv1_w; // (3, 3, 1, 32)
    std::vector<float> head_conv1_b;
    BNFolded head_bn1;
    std::vector<float> head_conv2_w; // (3, 3, 32, 32)
    std::vector<float> head_conv2_b;
    BNFolded head_bn2;
    std::vector<ResBlockCache> head_layer1; // 2
    std::vector<ResBlockCache> head_layer2; // 2

    // xv
    UnitCache xv_tdnn; // 320→128, k=5, s=2
    std::vector<DenseLayerCache> block1;
    UnitCache xv_transit1;
    std::vector<DenseLayerCache> block2;
    UnitCache xv_transit2;
    std::vector<DenseLayerCache> block3;
    UnitCache xv_transit3;
    UnitCache xv_out_nl; // BN-only
    UnitCache xv_dense;  // 1024→192, k=1, BN affine=False (no weight/bias on BN)

    // Channel counts (validated post-bind for shape sanity).
    int n_mels = 80;
    int after_fcm_C = 320;
    int after_tdnn_C = 128;
    int block1_out_C = 0; // 128 + 12*32 = 512
    int block2_out_C = 0; // 256 + 24*32 = 1024
    int block3_out_C = 0; // 512 + 16*32 = 1024
    int after_transit3_C = 512;
};

// Fold a 4-D Conv2d weight (PyTorch (out, in, kH, kW) flattened in
// GGUF as ne=(kW, kH, in, out), row-major) into our preferred CPU
// layout: (out, in, kH, kW) row-major — i.e. just match PyTorch's
// natural in-memory arrangement.
//
// `weight_flat` is the raw F32 buffer (already dequantized from F16/Q8_0
// at read time), and the GGUF stored it as (kW, kH, in, out) which
// in row-major bytes is exactly the (out, in, kH, kW) PyTorch order
// — they mirror each other thanks to ggml's reverse-indexing convention.
// So we don't need to permute, just read indices appropriately.

// Helper: index into a Conv2d weight stored ne=(kW, kH, in, out). The
// PyTorch order (o, i, kh, kw) maps to GGUF's flat index
// `((o*in + i)*kH + kh)*kW + kw`.
static inline float w2d(const float* w, int o, int i, int kh, int kw, int n_in, int kH, int kW) {
    return w[((size_t)o * (size_t)n_in + (size_t)i) * (size_t)kH * (size_t)kW + (size_t)kh * (size_t)kW + (size_t)kw];
}

// Index into a Conv1d weight stored ne=(kW, in, out) → PyTorch (out, in, kw)
// → flat row-major (o, i, kw) → `(o*in + i)*kw + kw`.
static inline float w1d(const float* w, int o, int i, int kw, int n_in, int kW) {
    return w[((size_t)o * (size_t)n_in + (size_t)i) * (size_t)kW + (size_t)kw];
}

static void add_channel_bias(float* x, int C, int N, const std::vector<float>& bias) {
    if (bias.empty())
        return;
    for (int c = 0; c < C; c++) {
        const float b = bias[(size_t)c];
        float* row = x + (size_t)c * (size_t)N;
        for (int n = 0; n < N; n++)
            row[n] += b;
    }
}

// 2-D conv for the FCM head. Input layout: (C_in, H, W) row-major
// (C_in slowest). Output: (C_out, H_out, W_out) row-major. Stride and
// padding are (sH, sW) / (pH, pW).
static void conv2d_forward(const float* in, int C_in, int H_in, int W_in, const float* w, int C_out, int kH, int kW,
                           int sH, int sW, int pH, int pW, float* out) {
    const int H_out = (H_in + 2 * pH - kH) / sH + 1;
    const int W_out = (W_in + 2 * pW - kW) / sW + 1;
    for (int co = 0; co < C_out; co++) {
        for (int yh = 0; yh < H_out; yh++) {
            const int yh_in = yh * sH - pH;
            for (int xw = 0; xw < W_out; xw++) {
                const int xw_in = xw * sW - pW;
                float s = 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kh = 0; kh < kH; kh++) {
                        const int hpos = yh_in + kh;
                        if (hpos < 0 || hpos >= H_in)
                            continue;
                        for (int kw = 0; kw < kW; kw++) {
                            const int wpos = xw_in + kw;
                            if (wpos < 0 || wpos >= W_in)
                                continue;
                            const float xv =
                                in[((size_t)ci * (size_t)H_in + (size_t)hpos) * (size_t)W_in + (size_t)wpos];
                            const float wv = w2d(w, co, ci, kh, kw, C_in, kH, kW);
                            s += xv * wv;
                        }
                    }
                }
                out[((size_t)co * (size_t)H_out + (size_t)yh) * (size_t)W_out + (size_t)xw] = s;
            }
        }
    }
}

// 1-D conv for the xvector chain. Input: (C_in, T) row-major. Output:
// (C_out, T_out) row-major.
static void conv1d_forward(const float* in, int C_in, int T_in, const float* w, int C_out, int kW, int s, int p, int d,
                           float* out) {
    const int T_out = (T_in + 2 * p - d * (kW - 1) - 1) / s + 1;
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < T_out; t++) {
            const int t0 = t * s - p;
            float ssum = 0.0f;
            for (int ci = 0; ci < C_in; ci++) {
                for (int k = 0; k < kW; k++) {
                    const int tpos = t0 + k * d;
                    if (tpos < 0 || tpos >= T_in)
                        continue;
                    const float xv = in[(size_t)ci * (size_t)T_in + (size_t)tpos];
                    const float wv = w1d(w, co, ci, k, C_in, kW);
                    ssum += xv * wv;
                }
            }
            out[(size_t)co * (size_t)T_out + (size_t)t] = ssum;
        }
    }
}

// Apply per-channel BN (gamma, beta) to a (C, N) buffer in place where
// N is the flat product of all non-channel dims (H*W for 2D, T for 1D).
static void apply_bn_inplace(float* x, int C, int N, const BNFolded& f) {
    for (int c = 0; c < C; c++) {
        const float g = f.gamma[(size_t)c];
        const float b = f.beta[(size_t)c];
        float* row = x + (size_t)c * (size_t)N;
        for (int n = 0; n < N; n++)
            row[n] = row[n] * g + b;
    }
}

static void relu_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (x[i] < 0.0f)
            x[i] = 0.0f;
}

// FCM BasicResBlock. `in` shape (C_in, H_in, W_in) → `out` shape
// (C_out, H_out, W_out). H may be downsampled by `stride` (W stays).
static void resblock_forward(const float* in, int C_in, int H_in, int W_in, const ResBlockCache& b,
                             std::vector<float>& out, int& C_out, int& H_out, int& W_out) {
    // conv1: stride=(stride, 1), p=(1,1)
    H_out = (H_in + 2 - 3) / b.stride + 1;
    W_out = W_in;
    C_out = b.conv1_out;
    out.assign((size_t)C_out * (size_t)H_out * (size_t)W_out, 0.0f);
    conv2d_forward(in, C_in, H_in, W_in, b.conv1_w.data(), C_out, 3, 3, b.stride, 1, 1, 1, out.data());
    add_channel_bias(out.data(), C_out, H_out * W_out, b.conv1_b);
    apply_bn_inplace(out.data(), C_out, H_out * W_out, b.bn1);
    relu_inplace(out.data(), out.size());

    // conv2: stride=1, p=1, channels stay
    std::vector<float> tmp((size_t)C_out * (size_t)H_out * (size_t)W_out, 0.0f);
    conv2d_forward(out.data(), C_out, H_out, W_out, b.conv2_w.data(), C_out, 3, 3, 1, 1, 1, 1, tmp.data());
    add_channel_bias(tmp.data(), C_out, H_out * W_out, b.conv2_b);
    apply_bn_inplace(tmp.data(), C_out, H_out * W_out, b.bn2);

    // Shortcut branch: 1×1 Conv2d + BN if (stride!=1 || in!=out), else identity.
    std::vector<float> sc;
    const float* sc_ptr = nullptr;
    if (b.has_shortcut) {
        sc.assign((size_t)C_out * (size_t)H_out * (size_t)W_out, 0.0f);
        conv2d_forward(in, C_in, H_in, W_in, b.sc_w.data(), C_out, 1, 1, b.stride, 1, 0, 0, sc.data());
        add_channel_bias(sc.data(), C_out, H_out * W_out, b.sc_b);
        apply_bn_inplace(sc.data(), C_out, H_out * W_out, b.sc_bn);
        sc_ptr = sc.data();
    } else {
        sc_ptr = in;
    }

    // out = relu(tmp + sc)
    const size_t n = tmp.size();
    for (size_t i = 0; i < n; i++)
        tmp[i] = tmp[i] + sc_ptr[i];
    relu_inplace(tmp.data(), n);
    out = std::move(tmp);
}

static void init_unit(UnitCache& u, const cb_campplus_unit& src, int default_in, int default_out, int default_kw) {
    if (src.lin_w) {
        // GGUF ne=(kw, in, out) → flat == PyTorch (out, in, kw) row-major.
        u.lin_w = read_tensor_f32(src.lin_w);
        // Reshape semantics — ne[0]=kw, ne[1]=in, ne[2]=out.
        u.kw = (int)src.lin_w->ne[0];
        u.in_dim = (int)src.lin_w->ne[1];
        u.out_dim = (int)src.lin_w->ne[2];
    } else {
        u.kw = default_kw;
        u.in_dim = default_in;
        u.out_dim = default_out;
    }
    if (src.lin_b)
        u.lin_b = read_tensor_f32(src.lin_b);
    if (src.bn_m && src.bn_v) {
        u.bn = fold_bn(src.bn_m, src.bn_v, src.bn_w, src.bn_b, u.out_dim);
        u.has_bn = true;
    }
}

static void init_resblock(ResBlockCache& b, const cb_campplus_resblock& src) {
    b.stride = src.stride;
    if (src.conv1_w) {
        b.conv1_w = read_tensor_f32(src.conv1_w);
        b.conv1_in = (int)src.conv1_w->ne[2];
        b.conv1_out = (int)src.conv1_w->ne[3];
    }
    if (src.conv1_b)
        b.conv1_b = read_tensor_f32(src.conv1_b);
    b.bn1 = fold_bn(src.bn1_m, src.bn1_v, src.bn1_w, src.bn1_b, b.conv1_out);
    if (src.conv2_w) {
        b.conv2_w = read_tensor_f32(src.conv2_w);
        b.conv2_in = (int)src.conv2_w->ne[2];
        b.conv2_out = (int)src.conv2_w->ne[3];
    }
    if (src.conv2_b)
        b.conv2_b = read_tensor_f32(src.conv2_b);
    b.bn2 = fold_bn(src.bn2_m, src.bn2_v, src.bn2_w, src.bn2_b, b.conv2_out);
    if (src.sc_w) {
        b.has_shortcut = true;
        b.sc_w = read_tensor_f32(src.sc_w);
        b.sc_in = (int)src.sc_w->ne[2];
        b.sc_out = (int)src.sc_w->ne[3];
        if (src.sc_b)
            b.sc_b = read_tensor_f32(src.sc_b);
        b.sc_bn = fold_bn(src.sc_bn_m, src.sc_bn_v, src.sc_bn_w, src.sc_bn_b, b.sc_out);
    }
}

static void init_dense_layer(DenseLayerCache& l, const cb_campplus_dense_layer& src, int dilation) {
    l.dilation = dilation;
    if (src.l1_w) {
        l.l1_w = read_tensor_f32(src.l1_w);
        l.in_channels = (int)src.l1_w->ne[1];
        l.bn_channels = (int)src.l1_w->ne[2];
    }
    if (src.l1_b)
        l.l1_b = read_tensor_f32(src.l1_b);
    l.nonl1_bn = fold_bn(src.nonl1_bn_m, src.nonl1_bn_v, src.nonl1_bn_w, src.nonl1_bn_b, l.in_channels);
    l.nonl2_bn = fold_bn(src.nonl2_bn_m, src.nonl2_bn_v, src.nonl2_bn_w, src.nonl2_bn_b, l.bn_channels);
    if (src.cam_ll_w) {
        l.cam_ll_w = read_tensor_f32(src.cam_ll_w);
        l.cam_k = (int)src.cam_ll_w->ne[0];
        // ne[1] should equal bn_channels, ne[2] is out (growth_rate=32).
        l.cam_out = (int)src.cam_ll_w->ne[2];
    }
    l.padding = (l.cam_k - 1) / 2 * l.dilation;
    if (src.cam_l1_w) {
        l.cam_l1_w = read_tensor_f32(src.cam_l1_w);
        if (src.cam_l1_b)
            l.cam_l1_b = read_tensor_f32(src.cam_l1_b);
    }
    if (src.cam_l2_w) {
        l.cam_l2_w = read_tensor_f32(src.cam_l2_w);
        if (src.cam_l2_b)
            l.cam_l2_b = read_tensor_f32(src.cam_l2_b);
    }
}

static void init_cache(CampplusCache& cache, const cb_campplus_model& m) {
    if (cache.initialised)
        return;

    // FCM head.
    if (m.head.conv1_w) {
        cache.head_conv1_w = read_tensor_f32(m.head.conv1_w);
        if (m.head.conv1_b)
            cache.head_conv1_b = read_tensor_f32(m.head.conv1_b);
        cache.head_bn1 = fold_bn(m.head.bn1_m, m.head.bn1_v, m.head.bn1_w, m.head.bn1_b, (int)m.head.conv1_w->ne[3]);
    }
    if (m.head.conv2_w) {
        cache.head_conv2_w = read_tensor_f32(m.head.conv2_w);
        if (m.head.conv2_b)
            cache.head_conv2_b = read_tensor_f32(m.head.conv2_b);
        cache.head_bn2 = fold_bn(m.head.bn2_m, m.head.bn2_v, m.head.bn2_w, m.head.bn2_b, (int)m.head.conv2_w->ne[3]);
    }
    cache.head_layer1.assign(m.head.layer1.size(), ResBlockCache{});
    cache.head_layer2.assign(m.head.layer2.size(), ResBlockCache{});
    for (size_t i = 0; i < m.head.layer1.size(); i++)
        init_resblock(cache.head_layer1[i], m.head.layer1[i]);
    for (size_t i = 0; i < m.head.layer2.size(); i++)
        init_resblock(cache.head_layer2[i], m.head.layer2[i]);

    // xv units.
    init_unit(cache.xv_tdnn, m.tdnn, 320, 128, 5);
    init_unit(cache.xv_transit1, m.transit1, 512, 256, 1);
    init_unit(cache.xv_transit2, m.transit2, 1024, 512, 1);
    init_unit(cache.xv_transit3, m.transit3, 1024, 512, 1);
    // out_nl is BN-only (no linear). The unit's `out_dim` we infer from
    // the BN running_mean length.
    if (m.out_nl.bn_m) {
        cache.xv_out_nl.bn =
            fold_bn(m.out_nl.bn_m, m.out_nl.bn_v, m.out_nl.bn_w, m.out_nl.bn_b, (int)m.out_nl.bn_m->ne[0]);
        cache.xv_out_nl.has_bn = true;
        cache.xv_out_nl.out_dim = (int)m.out_nl.bn_m->ne[0];
    }
    // dense: BN affine=False (no weight/bias). bn_w and bn_b will be nullptr.
    init_unit(cache.xv_dense, m.dense, 1024, 192, 1);

    // Dense blocks.
    auto init_block = [&](std::vector<DenseLayerCache>& dst, const cb_campplus_dense_block& blk) {
        dst.assign(blk.layers.size(), DenseLayerCache{});
        for (size_t i = 0; i < blk.layers.size(); i++)
            init_dense_layer(dst[i], blk.layers[i], blk.dilation);
    };
    init_block(cache.block1, m.block1);
    init_block(cache.block2, m.block2);
    init_block(cache.block3, m.block3);

    cache.block1_out_C = 128 + 12 * 32; // 512
    cache.block2_out_C = 256 + 24 * 32; // 1024
    cache.block3_out_C = 512 + 16 * 32; // 1024

    cache.initialised = true;
}

// ---------------------------------------------------------------------------
// Forward pieces
// ---------------------------------------------------------------------------

// FCM head: (1, 80, T) → (320, T)
static void fcm_forward(const CampplusCache& cache, const float* feat_t_80, int T, std::vector<float>& fcm_out,
                        int& C_out_final, int& T_out_final) {
    // Permute (T, 80) → (80, T) channel-first. Then unsqueeze to (1, 80, T).
    // Stored as (C_in=1, H=80, W=T).
    const int C0 = 1;
    const int H0 = 80;
    const int W0 = T;
    std::vector<float> x((size_t)C0 * (size_t)H0 * (size_t)W0, 0.0f);
    for (int h = 0; h < H0; h++) {
        for (int w = 0; w < W0; w++) {
            x[(size_t)h * (size_t)W0 + (size_t)w] = feat_t_80[(size_t)w * (size_t)H0 + (size_t)h];
        }
    }

    // conv1: 1→32, k=3, s=1, p=1
    int H1 = H0;
    int W1 = W0;
    int C1 = 32;
    std::vector<float> y1((size_t)C1 * (size_t)H1 * (size_t)W1, 0.0f);
    conv2d_forward(x.data(), C0, H0, W0, cache.head_conv1_w.data(), C1, 3, 3, 1, 1, 1, 1, y1.data());
    add_channel_bias(y1.data(), C1, H1 * W1, cache.head_conv1_b);
    apply_bn_inplace(y1.data(), C1, H1 * W1, cache.head_bn1);
    relu_inplace(y1.data(), y1.size());

    // layer1 (2 BasicResBlocks) — first downsamples H by 2.
    int Ca = C1, Ha = H1, Wa = W1;
    std::vector<float> ya = std::move(y1);
    for (size_t i = 0; i < cache.head_layer1.size(); i++) {
        int Cn = 0, Hn = 0, Wn = 0;
        std::vector<float> yn;
        resblock_forward(ya.data(), Ca, Ha, Wa, cache.head_layer1[i], yn, Cn, Hn, Wn);
        ya = std::move(yn);
        Ca = Cn;
        Ha = Hn;
        Wa = Wn;
    }
    // layer2 (2 BasicResBlocks) — first downsamples H by 2 again.
    for (size_t i = 0; i < cache.head_layer2.size(); i++) {
        int Cn = 0, Hn = 0, Wn = 0;
        std::vector<float> yn;
        resblock_forward(ya.data(), Ca, Ha, Wa, cache.head_layer2[i], yn, Cn, Hn, Wn);
        ya = std::move(yn);
        Ca = Cn;
        Ha = Hn;
        Wa = Wn;
    }

    // conv2: 32→32, k=3, s=(2,1), p=1
    int H2 = (Ha + 2 - 3) / 2 + 1;
    int W2 = Wa;
    int C2 = 32;
    std::vector<float> y2((size_t)C2 * (size_t)H2 * (size_t)W2, 0.0f);
    conv2d_forward(ya.data(), Ca, Ha, Wa, cache.head_conv2_w.data(), C2, 3, 3, 2, 1, 1, 1, y2.data());
    add_channel_bias(y2.data(), C2, H2 * W2, cache.head_conv2_b);
    apply_bn_inplace(y2.data(), C2, H2 * W2, cache.head_bn2);
    relu_inplace(y2.data(), y2.size());

    // Reshape (C2=32, H2=10, W2=T) → (320, T) by flattening C2 and H2.
    // Stored channel-major: y2[c*H2*W2 + h*W2 + t]. The flatten is
    // `(c, h) → c*H2 + h` so the new channel index is c*H2+h.
    fcm_out.assign((size_t)(C2 * H2) * (size_t)W2, 0.0f);
    for (int c = 0; c < C2; c++) {
        for (int h = 0; h < H2; h++) {
            const int new_c = c * H2 + h;
            for (int t = 0; t < W2; t++) {
                fcm_out[(size_t)new_c * (size_t)W2 + (size_t)t] =
                    y2[((size_t)c * (size_t)H2 + (size_t)h) * (size_t)W2 + (size_t)t];
            }
        }
    }
    C_out_final = C2 * H2;
    T_out_final = W2;
}

// CAMDenseTDNNLayer forward. `in` is (C_in, T) where C_in is the running
// concatenation channel count; `out` ends up appended (concat) onto in
// to form the next input.
static std::vector<float> dense_layer_forward(const float* in, int C_in_actual, int T, const DenseLayerCache& l) {
    // nonl1.bn(in_channels) → ReLU → l1 1×1 conv → nonl2.bn → ReLU →
    // CAM layer. nonl1's BN expects the full running C_in which equals
    // l.in_channels. (Per the dense block construction in xvector.py
    // the in_channels of layer i = base + i * growth.)
    if (C_in_actual != l.in_channels) {
        fprintf(stderr, "campplus: dense_layer_forward C_in mismatch (%d vs %d)\n", C_in_actual, l.in_channels);
        return {};
    }
    std::vector<float> a((size_t)C_in_actual * (size_t)T);
    std::memcpy(a.data(), in, a.size() * sizeof(float));
    apply_bn_inplace(a.data(), C_in_actual, T, l.nonl1_bn);
    relu_inplace(a.data(), a.size());

    // l1: 1×1 Conv1d (in_channels → bn_channels)
    std::vector<float> bo((size_t)l.bn_channels * (size_t)T, 0.0f);
    conv1d_forward(a.data(), C_in_actual, T, l.l1_w.data(), l.bn_channels, 1, 1, 0, 1, bo.data());
    if (!l.l1_b.empty())
        add_channel_bias(bo.data(), l.bn_channels, T, l.l1_b);

    // nonl2.bn(bn_channels) → ReLU
    apply_bn_inplace(bo.data(), l.bn_channels, T, l.nonl2_bn);
    relu_inplace(bo.data(), bo.size());

    // CAM layer:
    //   y = linear_local(x)           # (out, T)
    //   ctx = mean(x, T) + seg_pool(x)  # (in_channels=bn, 1) + broadcast
    //   ctx = relu(linear1(ctx))       # (bn/2,)
    //   m = sigmoid(linear2(ctx))      # (out,)
    //   return y * m
    //
    // The seg_pooling uses kernel=100, stride=100, ceil_mode=True →
    // hold-and-broadcast along T. For `mean(-1, keepdim=True)` plus
    // seg-pool then expanded back to T, the result is per-time scalar
    // attention weights times the per-channel output. Since linear1 /
    // linear2 are applied AFTER summing the global mean and the
    // seg-pooled view, the time dimension flows through seg_pool. For
    // a single utterance we can simplify by recognising the global
    // mean is (bn, 1) and seg-pool gives (bn, T) via the
    // unsqueeze-expand-reshape dance. Their sum is also (bn, T).
    std::vector<float> y((size_t)l.cam_out * (size_t)T, 0.0f);
    conv1d_forward(bo.data(), l.bn_channels, T, l.cam_ll_w.data(), l.cam_out, l.cam_k, 1, l.padding, l.dilation,
                   y.data());

    // Compute global mean over T per channel.
    std::vector<float> gmean((size_t)l.bn_channels, 0.0f);
    for (int c = 0; c < l.bn_channels; c++) {
        float s = 0.0f;
        const float* row = bo.data() + (size_t)c * (size_t)T;
        for (int t = 0; t < T; t++)
            s += row[t];
        gmean[(size_t)c] = s / (float)T;
    }
    // seg_pool: avg_pool1d(k=100, s=100, ceil_mode=True). Number of
    // segments = ceil(T / 100). For each segment, mean of the values
    // in that segment per channel; then expand back to (bn, T) by
    // tiling each segment value across the 100 (or remainder) frames
    // it covered. ceil_mode behaviour: when T % 100 != 0, the last
    // segment covers the partial tail; pytorch's avg_pool1d with
    // ceil_mode=True averages over the actual frames present
    // (count_include_pad=True default uses 0-padding, but kernel doesn't
    // extend past T when ceil_mode shrinks it) — actually
    // F.avg_pool1d with ceil_mode=True INCLUDES partial windows; the
    // divisor is the kernel size unless count_include_pad=False.
    //
    // Let's keep it simple: average the actual values in each segment
    // (matches `count_include_pad=True` default with kernel=100 stride=100;
    // when the last segment is shorter, the divisor is still kernel=100
    // — torch's behaviour). Then broadcast each segment value to all
    // its frames.
    constexpr int kSegLen = 100;
    const int n_seg = (T + kSegLen - 1) / kSegLen;
    std::vector<float> seg((size_t)l.bn_channels * (size_t)n_seg, 0.0f);
    for (int c = 0; c < l.bn_channels; c++) {
        const float* row = bo.data() + (size_t)c * (size_t)T;
        for (int s = 0; s < n_seg; s++) {
            const int t0 = s * kSegLen;
            float ss = 0.0f;
            const int n_in_seg = std::min(kSegLen, T - t0);
            for (int t = 0; t < n_in_seg; t++)
                ss += row[t0 + t];
            // PyTorch's avg_pool1d(ceil_mode=True) divides by the kernel
            // size (count_include_pad=True default).
            seg[(size_t)c * (size_t)n_seg + (size_t)s] = ss / (float)kSegLen;
        }
    }
    // ctx = mean + seg_pool(x) broadcast back to (bn, T)
    std::vector<float> ctx((size_t)l.bn_channels * (size_t)T, 0.0f);
    for (int c = 0; c < l.bn_channels; c++) {
        for (int t = 0; t < T; t++) {
            const int s = t / kSegLen;
            ctx[(size_t)c * (size_t)T + (size_t)t] = gmean[(size_t)c] + seg[(size_t)c * (size_t)n_seg + (size_t)s];
        }
    }

    // linear1: 1×1 Conv1d (bn → bn/2) + bias + ReLU
    const int bn_half = l.bn_channels / 2;
    std::vector<float> ctx2((size_t)bn_half * (size_t)T, 0.0f);
    conv1d_forward(ctx.data(), l.bn_channels, T, l.cam_l1_w.data(), bn_half, 1, 1, 0, 1, ctx2.data());
    if (!l.cam_l1_b.empty()) {
        for (int c = 0; c < bn_half; c++) {
            const float bias = l.cam_l1_b[(size_t)c];
            float* row = ctx2.data() + (size_t)c * (size_t)T;
            for (int t = 0; t < T; t++)
                row[t] += bias;
        }
    }
    relu_inplace(ctx2.data(), ctx2.size());

    // linear2: 1×1 Conv1d (bn/2 → out) + bias + sigmoid
    std::vector<float> mask((size_t)l.cam_out * (size_t)T, 0.0f);
    conv1d_forward(ctx2.data(), bn_half, T, l.cam_l2_w.data(), l.cam_out, 1, 1, 0, 1, mask.data());
    if (!l.cam_l2_b.empty()) {
        for (int c = 0; c < l.cam_out; c++) {
            const float bias = l.cam_l2_b[(size_t)c];
            float* row = mask.data() + (size_t)c * (size_t)T;
            for (int t = 0; t < T; t++)
                row[t] += bias;
        }
    }
    for (size_t i = 0; i < mask.size(); i++)
        mask[i] = 1.0f / (1.0f + std::exp(-mask[i]));

    // y *= mask
    for (size_t i = 0; i < y.size(); i++)
        y[i] *= mask[i];

    return y;
}

// CAMDenseTDNNBlock.forward — sequentially concat each layer's output
// onto the running input.
static std::vector<float> dense_block_forward(const float* in, int C_in, int T, const std::vector<DenseLayerCache>& blk,
                                              int& C_out) {
    std::vector<float> running((size_t)C_in * (size_t)T);
    std::memcpy(running.data(), in, running.size() * sizeof(float));
    int C_running = C_in;

    for (size_t li = 0; li < blk.size(); li++) {
        const auto& l = blk[li];
        auto delta = dense_layer_forward(running.data(), C_running, T, l);
        if (delta.empty())
            return {};
        std::vector<float> next((size_t)(C_running + l.cam_out) * (size_t)T, 0.0f);
        std::memcpy(next.data(), running.data(), (size_t)C_running * (size_t)T * sizeof(float));
        std::memcpy(next.data() + (size_t)C_running * (size_t)T, delta.data(), delta.size() * sizeof(float));
        running = std::move(next);
        C_running += l.cam_out;
    }
    C_out = C_running;
    return running;
}

// TDNNLayer / TransitLayer forward (Conv1d + BN + ReLU). For the tdnn
// stage the kernel is k=5 stride=2 padding=2; for transit*/dense it's
// k=1 stride=1 padding=0. We infer from the cached unit.
static std::vector<float> unit_forward(const UnitCache& u, const float* in, int C_in, int T, int& T_out, int s, int p) {
    if (C_in != u.in_dim) {
        fprintf(stderr, "campplus: unit_forward C_in mismatch (%d vs %d)\n", C_in, u.in_dim);
        return {};
    }
    T_out = (T + 2 * p - (u.kw - 1) - 1) / s + 1;
    std::vector<float> out((size_t)u.out_dim * (size_t)T_out, 0.0f);
    conv1d_forward(in, C_in, T, u.lin_w.data(), u.out_dim, u.kw, s, p, 1, out.data());
    if (!u.lin_b.empty())
        add_channel_bias(out.data(), u.out_dim, T_out, u.lin_b);
    if (u.has_bn) {
        apply_bn_inplace(out.data(), u.out_dim, T_out, u.bn);
    }
    return out;
}

// Helper: BN+ReLU+Conv1d (the order TransitLayer / DenseLayer use).
// `bn` is the BN folded against C_in.
static std::vector<float> bn_relu_conv1d(const BNFolded& bn, const float* in, int C_in, int T,
                                         const std::vector<float>& lin_w, int kw, int C_out, int s, int p) {
    std::vector<float> a((size_t)C_in * (size_t)T);
    std::memcpy(a.data(), in, a.size() * sizeof(float));
    apply_bn_inplace(a.data(), C_in, T, bn);
    relu_inplace(a.data(), a.size());
    const int T_out = (T + 2 * p - (kw - 1) - 1) / s + 1;
    std::vector<float> out((size_t)C_out * (size_t)T_out, 0.0f);
    conv1d_forward(a.data(), C_in, T, lin_w.data(), C_out, kw, s, p, 1, out.data());
    return out;
}

// StatsPool: concat(mean, std) along T → (2*C,)
static std::vector<float> stats_pool(const float* in, int C, int T) {
    std::vector<float> out((size_t)(2 * C), 0.0f);
    for (int c = 0; c < C; c++) {
        const float* row = in + (size_t)c * (size_t)T;
        double mean = 0.0;
        for (int t = 0; t < T; t++)
            mean += (double)row[t];
        mean /= (double)T;
        double sumsq = 0.0;
        for (int t = 0; t < T; t++) {
            const double d = (double)row[t] - mean;
            sumsq += d * d;
        }
        // PyTorch tensor.std defaults to unbiased=True → divide by (n-1).
        const double var = (T > 1) ? sumsq / (double)(T - 1) : 0.0;
        const double std_ = std::sqrt(var);
        out[(size_t)c] = (float)mean;
        out[(size_t)C + (size_t)c] = (float)std_;
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// pimpl + lifecycle
// ---------------------------------------------------------------------------

cb_campplus_runtime::cb_campplus_runtime() {
    impl = new CampplusCache();
}

cb_campplus_runtime::~cb_campplus_runtime() {
    delete static_cast<CampplusCache*>(impl);
    impl = nullptr;
}

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

std::vector<float> compute_xvector(const cb_campplus_model& m, cb_campplus_runtime& cache, const float* feat_t_80,
                                   int T) {
    if (!feat_t_80 || T <= 0)
        return {};
    if (!m.head.conv1_w || !m.tdnn.lin_w || !m.dense.lin_w || m.block1.layers.empty()) {
        fprintf(stderr, "chatterbox_campplus: model not bound\n");
        return {};
    }
    auto* state = static_cast<CampplusCache*>(cache.impl);
    if (!state) {
        cache.impl = new CampplusCache();
        state = static_cast<CampplusCache*>(cache.impl);
    }
    init_cache(*state, m);
    cache.initialised = true;

    const bool dbg = std::getenv("CHATTERBOX_DEBUG") != nullptr;

    // FCM head: (T, 80) → (320, T)
    int C_fcm = 0, T_fcm = 0;
    std::vector<float> fcm;
    fcm_forward(*state, feat_t_80, T, fcm, C_fcm, T_fcm);
    if (dbg)
        fprintf(stderr, "campplus: post-FCM C=%d T=%d\n", C_fcm, T_fcm);

    // tdnn: 320→128, k=5, s=2, p=2 → (128, T/2)
    constexpr int kTdnnPad = 2;
    int T_tdnn = 0;
    auto post_tdnn = unit_forward(state->xv_tdnn, fcm.data(), C_fcm, T_fcm, T_tdnn, /*s*/ 2, /*p*/ kTdnnPad);
    if (post_tdnn.empty())
        return {};
    relu_inplace(post_tdnn.data(), post_tdnn.size());
    if (dbg)
        fprintf(stderr, "campplus: post-tdnn C=%d T=%d\n", state->xv_tdnn.out_dim, T_tdnn);

    // block1: 12 layers, dilation=1 → 128 + 12*32 = 512
    int C_blk1 = 0;
    auto post_blk1 = dense_block_forward(post_tdnn.data(), 128, T_tdnn, state->block1, C_blk1);
    if (dbg)
        fprintf(stderr, "campplus: post-block1 C=%d T=%d\n", C_blk1, T_tdnn);

    // transit1: BN(512) + ReLU + Conv1d 1×1 (512→256), bias=False
    BNFolded bn_t1 = fold_bn(m.transit1.bn_m, m.transit1.bn_v, m.transit1.bn_w, m.transit1.bn_b, C_blk1);
    auto post_t1 = bn_relu_conv1d(bn_t1, post_blk1.data(), C_blk1, T_tdnn, state->xv_transit1.lin_w, 1, 256, 1, 0);

    // block2: 24 layers, dilation=2 → 256 + 24*32 = 1024
    int C_blk2 = 0;
    auto post_blk2 = dense_block_forward(post_t1.data(), 256, T_tdnn, state->block2, C_blk2);

    // transit2: BN(1024) + ReLU + Conv1d 1×1 (1024→512)
    BNFolded bn_t2 = fold_bn(m.transit2.bn_m, m.transit2.bn_v, m.transit2.bn_w, m.transit2.bn_b, C_blk2);
    auto post_t2 = bn_relu_conv1d(bn_t2, post_blk2.data(), C_blk2, T_tdnn, state->xv_transit2.lin_w, 1, 512, 1, 0);

    // block3: 16 layers, dilation=2 → 512 + 16*32 = 1024
    int C_blk3 = 0;
    auto post_blk3 = dense_block_forward(post_t2.data(), 512, T_tdnn, state->block3, C_blk3);

    if (dbg)
        fprintf(stderr, "campplus: post-block3 C=%d T=%d\n", C_blk3, T_tdnn);

    // transit3: BN(1024) + ReLU + Conv1d 1×1 (1024→512)
    BNFolded bn_t3 = fold_bn(m.transit3.bn_m, m.transit3.bn_v, m.transit3.bn_w, m.transit3.bn_b, C_blk3);
    auto post_t3 = bn_relu_conv1d(bn_t3, post_blk3.data(), C_blk3, T_tdnn, state->xv_transit3.lin_w, 1, 512, 1, 0);

    // out_nl: BN(512) + ReLU. Note: out_nl is a bare get_nonlinear, so its
    // GGUF tensors are at `out_nl.bn.*` (not `out_nl.nl.bn.*`); the bind
    // step has a special case for this.
    if (state->xv_out_nl.has_bn)
        apply_bn_inplace(post_t3.data(), 512, T_tdnn, state->xv_out_nl.bn);
    relu_inplace(post_t3.data(), post_t3.size());

    // StatsPool → (1024,)
    auto stats = stats_pool(post_t3.data(), 512, T_tdnn);

    // dense: Conv1d(1024→192, k=1) + BN(affine=False)
    std::vector<float> dense_in((size_t)1024, 0.0f);
    std::memcpy(dense_in.data(), stats.data(), 1024 * sizeof(float));
    int T_dense = 1;
    int T_dense_out = 0;
    auto dense_pre =
        unit_forward(state->xv_dense, dense_in.data(), 1024, T_dense, T_dense_out, /*s*/ 1, /*p*/ 0); // BN folded
    if (dense_pre.empty())
        return {};
    // The unit_forward above used u.bn folded against `out_dim=192`; for
    // affine=False the gamma reduces to 1/sqrt(var+eps) and beta to
    // -mean/sqrt(var+eps), so it's correct.
    // Squeeze the trailing T=1.
    std::vector<float> emb((size_t)192);
    for (int i = 0; i < 192; i++)
        emb[(size_t)i] = dense_pre[(size_t)i];
    return emb;
}

std::vector<float> embed_speaker(const cb_campplus_model& m, cb_campplus_runtime& cache, const float* pcm_16k,
                                 int n_samples) {
    int T = 0;
    auto fb = compute_fbank(pcm_16k, n_samples, T);
    if (fb.empty() || T <= 0)
        return {};
    return compute_xvector(m, cache, fb.data(), T);
}

// ---------------------------------------------------------------------------
// Phase 3 — 24 kHz Matcha-TTS prompt mel for `gen.prompt_feat`
// ---------------------------------------------------------------------------

namespace {

// Periodic Hann window — `torch.hann_window(N)` default `periodic=True`.
static void make_hann_periodic_24k(int N, std::vector<float>& out) {
    out.resize((size_t)N);
    const float twopi = 2.0f * (float)M_PI;
    for (int i = 0; i < N; i++)
        out[(size_t)i] = 0.5f * (1.0f - std::cos(twopi * (float)i / (float)N));
}

// Reflect-pad a (n,) buffer to (n + 2*pad,) — matches
// `F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")` in mel.py.
// PyTorch reflect mirrors around the EDGE samples (excludes the edge),
// so for input [a, b, c, d] with pad=2 → [c, b, a, b, c, d, c, b].
static std::vector<float> reflect_pad(const float* x, int n, int pad) {
    std::vector<float> out((size_t)n + 2 * (size_t)pad, 0.0f);
    if (n <= 0)
        return out;
    for (int i = 0; i < pad; i++) {
        // Index `pad - 1 - i` in the output is `i + 1` in input (PyTorch reflect).
        const int src = std::min(i + 1, n - 1);
        out[(size_t)(pad - 1 - i)] = x[src];
    }
    std::memcpy(out.data() + pad, x, (size_t)n * sizeof(float));
    for (int i = 0; i < pad; i++) {
        const int src = std::max(n - 2 - i, 0);
        out[(size_t)(pad + n + i)] = x[src];
    }
    return out;
}

} // namespace

std::vector<float> compute_prompt_feat_24k(const float* pcm_24k, int n_samples, int max_samples, int& T_mel_out) {
    T_mel_out = 0;
    if (!pcm_24k || n_samples <= 0)
        return {};

    constexpr int kSr = 24000;
    constexpr int kNFft = 1920;
    constexpr int kHop = 480;
    constexpr int kWin = 1920;
    constexpr int kNMels = 80;
    constexpr float kFmin = 0.0f;
    constexpr float kFmax = 8000.0f;
    constexpr float kClipVal = 1e-5f;
    constexpr float kStftEps = 1e-9f;

    // Truncate to max_samples (= DEC_COND_LEN = 10 * 24000 = 240000 in
    // `prepare_conditionals`) — `s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]`.
    if (max_samples > 0 && n_samples > max_samples)
        n_samples = max_samples;

    // Manual reflect pad of (n_fft - hop) / 2 = 720 samples each side
    // BEFORE the STFT (since the upstream `mel_spectrogram` passes
    // `center=False` to `torch.stft`).
    const int outer_pad = (kNFft - kHop) / 2;
    auto padded = reflect_pad(pcm_24k, n_samples, outer_pad);

    // Hann window.
    static thread_local std::vector<float> hann;
    if ((int)hann.size() != kWin)
        make_hann_periodic_24k(kWin, hann);

    // librosa Slaney mel basis (htk=False default, norm='slaney').
    static thread_local std::vector<float> mel_fb;
    if (mel_fb.empty()) {
        mel_fb = core_mel::build_slaney_fb(kSr, kNFft, kNMels, kFmin, kFmax, core_mel::FbLayout::MelsFreqs);
    }

    // STFT parameters: stride kHop, win kWin == n_fft so no inner zero-padding.
    // Frame count = (padded.size() - kNFft) / kHop + 1 — center=False
    // semantics applied to the already-reflect-padded input.
    const int n_padded = (int)padded.size();
    if (n_padded < kNFft) {
        return {};
    }
    const int T = (n_padded - kNFft) / kHop + 1;
    if (T <= 0)
        return {};

    // STFT + magnitude + mel projection + log compression. core_mel
    // doesn't quite fit the Matcha shape (which uses sqrt(power+1e-9)
    // — magnitude with an additive eps inside the sqrt — and natural
    // log with a clip-min of 1e-5, NOT log10 + max-clip(max-8) +
    // (x+4)/4). So we do it inline, reusing core_fft for the FFT and
    // core_mel::build_slaney_fb for the basis.
    const int n_freqs = kNFft / 2 + 1;
    std::vector<float> features((size_t)T * (size_t)kNMels, 0.0f);
    std::vector<float> frame((size_t)kNFft, 0.0f);
    std::vector<float> spec((size_t)2 * kNFft, 0.0f);
    std::vector<float> mag((size_t)n_freqs, 0.0f);

    for (int t = 0; t < T; t++) {
        const int offset = t * kHop;
        for (int i = 0; i < kNFft; i++)
            frame[(size_t)i] = padded[(size_t)(offset + i)] * hann[(size_t)i];
        core_fft::fft_radix2_wrapper(frame.data(), kNFft, spec.data());
        for (int k = 0; k < n_freqs; k++) {
            const float re = spec[(size_t)2 * k];
            const float im = spec[(size_t)2 * k + 1];
            // sqrt(power + 1e-9) — matches `torch.sqrt(spec.pow(2).sum(-1) + 1e-9)`.
            mag[(size_t)k] = std::sqrt(re * re + im * im + kStftEps);
        }
        // mel projection: mel_basis @ magnitudes. mel_fb is row-major
        // (n_mels × n_freqs), so each mel bin is a contiguous row.
        for (int m = 0; m < kNMels; m++) {
            const float* row = mel_fb.data() + (size_t)m * (size_t)n_freqs;
            double s = 0.0;
            for (int k = 0; k < n_freqs; k++)
                s += (double)row[(size_t)k] * (double)mag[(size_t)k];
            // log(clamp(mel, 1e-5)) — natural log, NOT log10.
            const float v = (float)s;
            features[(size_t)t * (size_t)kNMels + (size_t)m] = std::log(std::max(v, kClipVal));
        }
    }

    T_mel_out = T;
    return features;
}

} // namespace chatterbox_campplus
