// src/tada_codec.cpp — TADA codec decoder runtime.
//
// Architecture (from HumeAI/tada-codec decoder subfolder):
//
//   1. decoder_proj: Linear(512, 1024)
//   2. local_attention_decoder: 6 × TransformerEncoderLayer
//      - LocalSelfAttention: fused QKV(1024→3072), RoPE(θ=10000), 8 heads
//        post-norm: LayerNorm(x + attn_out)
//      - FFN: Linear(1024,4096) → GELU → Linear(4096,1024)
//        post-norm: LayerNorm(x + ffn_out)
//      - final_norm: LayerNorm(1024)
//   3. wav_decoder (DACDecoder): WNConv1d upsampler
//      - Conv1d(1024, 1536, k=7) → 4× DecoderBlock(strides [4,4,5,6])
//      - DecoderBlock: Snake1d → WNConvT1d → 3× ResidualUnit(d=1,3,9)
//      - Snake1d → Conv1d(96, 1, k=7) → Tanh
//
//   Total upsample: 4×4×5×6 = 480. 50 Hz features → 24000 Hz audio.

#include "tada_codec.h"
#include "core/conv.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <chrono>
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
// Bench instrumentation — `TADA_CODEC_BENCH=1` for per-stage timings.
// ===========================================================================

static bool tada_codec_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("TADA_CODEC_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct tada_codec_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit tada_codec_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~tada_codec_bench_stage() {
        if (!tada_codec_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  tada_codec_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ─────────────────────────── internal types ─────────────────────────

namespace {

// Conv weight (pre-materialized from weight-norm g*v/||v|| by the converter)
struct wn_conv {
    ggml_tensor* w = nullptr;      // materialized weight
    ggml_tensor* b = nullptr;      // bias
    ggml_tensor* w_perm = nullptr; // pre-permuted for decomposed col2im path
};

struct tada_codec_attn_layer {
    ggml_tensor* qkv_w;       // (1024, 3072)
    ggml_tensor* qkv_b;       // (3072,)
    ggml_tensor* out_w;       // (1024, 1024)
    ggml_tensor* out_b;       // (1024,)
    ggml_tensor* attn_norm_w; // LayerNorm weight
    ggml_tensor* attn_norm_b; // LayerNorm bias
    ggml_tensor* ffn_up_w;    // (1024, 4096)
    ggml_tensor* ffn_up_b;    // (4096,)
    ggml_tensor* ffn_down_w;  // (4096, 1024)
    ggml_tensor* ffn_down_b;  // (1024,)
    ggml_tensor* ffn_norm_w;  // LayerNorm weight
    ggml_tensor* ffn_norm_b;  // LayerNorm bias
    ggml_tensor* rope_freqs;  // (2, head_dim/2, max_seq_len)
};

struct tada_codec_res_unit {
    ggml_tensor* alpha0;     // Snake1d alpha
    ggml_tensor* inv_alpha0; // 1/alpha0 (precomputed)
    wn_conv conv0;           // WNConv1d k=7
    ggml_tensor* alpha1;     // Snake1d alpha
    ggml_tensor* inv_alpha1; // 1/alpha1 (precomputed)
    wn_conv conv1;           // WNConv1d k=1
};

struct tada_codec_dec_block {
    ggml_tensor* snake_alpha;     // Snake1d alpha
    ggml_tensor* inv_snake_alpha; // 1/alpha (precomputed)
    wn_conv up_conv;              // WNConvTranspose1d
    tada_codec_res_unit res[3];   // dilation 1, 3, 9
};

} // namespace

struct tada_codec_context {
    int n_threads = 4;

    // Projection
    ggml_tensor* proj_w = nullptr;
    ggml_tensor* proj_b = nullptr;

    // Attention encoder
    tada_codec_attn_layer attn_layers[6];
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* final_norm_b = nullptr;

    // DAC decoder
    wn_conv in_conv;                // Conv1d(1024, 1536, k=7)
    tada_codec_dec_block blocks[4]; // strides [4,4,5,6]
    ggml_tensor* out_snake_alpha = nullptr;
    ggml_tensor* out_inv_snake_alpha = nullptr;
    wn_conv out_conv; // Conv1d(96, 1, k=7)

    // Config
    int embed_dim = 512;
    int hidden_dim = 1024;
    int n_attn_layers = 6;
    int n_attn_heads = 8;
    int head_dim = 128; // 1024/8
    int wav_channels = 1536;
    int strides[4] = {4, 4, 5, 6};
    int channels[5] = {1536, 768, 384, 192, 96};

    // ggml state
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    bool owns_backend = true;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;
    ggml_context* ctx_inv = nullptr;
    ggml_backend_buffer_t buf_inv = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;
};

// ──────────────────────── weight loading ────────────────────────────

static void bind_wn(tada_codec_context* c, wn_conv& wn, const char* prefix) {
    char key[256];
    // Try materialized weight first, fall back to raw direction tensor
    snprintf(key, sizeof(key), "%s.weight", prefix);
    wn.w = core_gguf::try_get(c->tensors, key);
    if (!wn.w) {
        snprintf(key, sizeof(key), "%s.parametrizations.weight.original1", prefix);
        wn.w = core_gguf::try_get(c->tensors, key);
    }
    snprintf(key, sizeof(key), "%s.bias", prefix);
    wn.b = core_gguf::try_get(c->tensors, key);
}

static bool bind_weights(tada_codec_context* c) {
    auto& m = c->tensors;

    c->proj_w = core_gguf::require(m, "codec.proj.weight", "tada-codec");
    c->proj_b = core_gguf::try_get(m, "codec.proj.bias");
    c->final_norm_w = core_gguf::require(m, "codec.attn.final_norm.weight", "tada-codec");
    c->final_norm_b = core_gguf::try_get(m, "codec.attn.final_norm.bias");

    char key[256];
    for (int i = 0; i < 6; i++) {
        auto& l = c->attn_layers[i];
#define BIND_ATTN(fld, suffix)                                                                                         \
    snprintf(key, sizeof(key), "codec.attn.blk.%d." suffix, i);                                                        \
    l.fld = core_gguf::try_get(m, key);
        BIND_ATTN(qkv_w, "attn_qkv.weight")
        BIND_ATTN(qkv_b, "attn_qkv.bias")
        BIND_ATTN(out_w, "attn_output.weight")
        BIND_ATTN(out_b, "attn_output.bias")
        BIND_ATTN(attn_norm_w, "attn_norm.weight")
        BIND_ATTN(attn_norm_b, "attn_norm.bias")
        BIND_ATTN(ffn_up_w, "ffn_up.weight")
        BIND_ATTN(ffn_up_b, "ffn_up.bias")
        BIND_ATTN(ffn_down_w, "ffn_down.weight")
        BIND_ATTN(ffn_down_b, "ffn_down.bias")
        BIND_ATTN(ffn_norm_w, "ffn_norm.weight")
        BIND_ATTN(ffn_norm_b, "ffn_norm.bias")
        BIND_ATTN(rope_freqs, "self_attn.rope_freqs")
#undef BIND_ATTN
    }

    // DAC decoder
    bind_wn(c, c->in_conv, "codec.dac.0");

    // Blocks: codec.dac.{1,2,3,4}
    for (int b = 0; b < 4; b++) {
        auto& blk = c->blocks[b];
        int dac_idx = b + 1; // codec.dac.1..4

        snprintf(key, sizeof(key), "codec.dac.%d.block.0.alpha", dac_idx);
        blk.snake_alpha = core_gguf::try_get(m, key);

        snprintf(key, sizeof(key), "codec.dac.%d.block.1", dac_idx);
        bind_wn(c, blk.up_conv, key);

        // ResidualUnits at codec.dac.{dac_idx}.block.{2,3,4}
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];
            int res_idx = r + 2; // block.{2,3,4}

            snprintf(key, sizeof(key), "codec.dac.%d.block.%d.block.0.alpha", dac_idx, res_idx);
            ru.alpha0 = core_gguf::try_get(m, key);

            snprintf(key, sizeof(key), "codec.dac.%d.block.%d.block.1", dac_idx, res_idx);
            bind_wn(c, ru.conv0, key);

            snprintf(key, sizeof(key), "codec.dac.%d.block.%d.block.2.alpha", dac_idx, res_idx);
            ru.alpha1 = core_gguf::try_get(m, key);

            snprintf(key, sizeof(key), "codec.dac.%d.block.%d.block.3", dac_idx, res_idx);
            bind_wn(c, ru.conv1, key);
        }
    }

    // Output: codec.dac.5 = Snake1d, codec.dac.6 = WNConv1d
    c->out_snake_alpha = core_gguf::try_get(m, "codec.dac.5.alpha");
    bind_wn(c, c->out_conv, "codec.dac.6");

    return true;
}

// ──────────────────── graph building helpers ────────────────────────

// Collect all alpha tensors that need inv_alpha, create the tensors
// in inv_ctx, alloc buffer, then fill with 1/alpha data.
static void precompute_all_inv_alphas(tada_codec_context* c) {
    // Count how many inv_alpha tensors we need
    struct alpha_pair {
        ggml_tensor* src;
        ggml_tensor** dst;
    };
    std::vector<alpha_pair> pairs;

    for (int b = 0; b < 4; b++) {
        auto& blk = c->blocks[b];
        if (blk.snake_alpha)
            pairs.push_back({blk.snake_alpha, &blk.inv_snake_alpha});
        for (int r = 0; r < 3; r++) {
            if (blk.res[r].alpha0)
                pairs.push_back({blk.res[r].alpha0, &blk.res[r].inv_alpha0});
            if (blk.res[r].alpha1)
                pairs.push_back({blk.res[r].alpha1, &blk.res[r].inv_alpha1});
        }
    }
    if (c->out_snake_alpha)
        pairs.push_back({c->out_snake_alpha, &c->out_inv_snake_alpha});

    if (pairs.empty())
        return;

    // Create ggml context for inv_alpha tensors — use no_alloc=true
    // and backend_alloc_ctx_tensors to avoid inline data allocation issues.
    size_t ctx_size = ggml_tensor_overhead() * (pairs.size() + 4) + 4096;
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* inv_ctx = ggml_init(ip);

    for (auto& p : pairs) {
        *p.dst = ggml_new_tensor(inv_ctx, GGML_TYPE_F32, ggml_n_dims(p.src), p.src->ne);
    }

    // Allocate buffer on backend
    ggml_backend_buffer_t inv_buf = ggml_backend_alloc_ctx_tensors(inv_ctx, c->backend);

    // Fill each with 1/alpha
    for (auto& p : pairs) {
        int64_t n = ggml_nelements(p.src);
        std::vector<float> a(n), inv(n);
        if (p.src->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(p.src, tmp.data(), 0, n * sizeof(ggml_fp16_t));
            for (int64_t i = 0; i < n; i++)
                a[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(p.src, a.data(), 0, n * sizeof(float));
        }
        for (int64_t i = 0; i < n; i++)
            inv[i] = 1.0f / (a[i] + 1e-12f);
        ggml_backend_tensor_set(*p.dst, inv.data(), 0, n * sizeof(float));
    }

    c->ctx_inv = inv_ctx;
    c->buf_inv = inv_buf;
}

// Get the materialized weight (pre-computed g*v/||v|| by the converter).
static inline ggml_tensor* wn_weight(const wn_conv& wn) {
    return wn.w;
}

// Snake1d: y = x + sin²(alpha * x) * inv_alpha
// Uses pre-computed inv_alpha to avoid ggml_div (which triggers fused aa_snake).
// Computes sin, then element-wise square (not ggml_sqr which also triggers it).
static ggml_tensor* snake1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* inv_alpha) {
    if (!alpha || !inv_alpha)
        return x;
    const int C = (int)x->ne[0];

    ggml_tensor* a = ggml_cast(ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, alpha, C, 1)), GGML_TYPE_F32);
    ggml_tensor* ia = ggml_cast(ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, inv_alpha, C, 1)), GGML_TYPE_F32);

    // sin(alpha * x)
    ggml_tensor* ax = ggml_mul(ctx, x, a);
    ggml_tensor* sin_ax = ggml_sin(ctx, ax);

    // sin²(alpha * x) = sin * sin (element-wise, avoids ggml_sqr pattern)
    ggml_tensor* sin2 = ggml_mul(ctx, sin_ax, sin_ax);

    // sin² * inv_alpha — broadcast (C,1) over (C,T) via ggml_mul
    ggml_tensor* term = ggml_mul(ctx, sin2, ia);

    return ggml_add(ctx, x, term);
}

// Conv1d with weight-normed weights. x: (C_in, T) → (C_out, T).
// Symmetric padding: p = (K-1)*dil/2.
static ggml_tensor* wn_conv1d(ggml_context* ctx, ggml_tensor* x, const wn_conv& wn, int dilation) {
    ggml_tensor* w = wn_weight(wn);
    if (!w)
        return x;
    const int K = (int)w->ne[0];
    const int Cout = (int)w->ne[2];
    const int T = (int)x->ne[1];
    const int p = dilation * (K - 1) / 2;

    // ggml_conv_1d expects input (T, C_in) and weight (K, C_in, C_out)
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    ggml_tensor* y = ggml_conv_1d(ctx, w, xt, /*stride*/ 1, p, dilation);
    // Result shape: (T_out, C_out, 1)
    y = ggml_reshape_2d(ctx, y, T, Cout);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C_out, T)
    if (wn.b) {
        y = ggml_add(ctx, y, ggml_cast(ctx, wn.b, GGML_TYPE_F32));
    }
    return y;
}

// ConvTranspose1d with weight-normed weights.
// PyTorch: ConvTranspose1d(Cin, Cout, kernel_size=2*s, stride=s, padding=ceil(s/2))
// Output length: (T-1)*s - 2*ceil(s/2) + 2*s = (T-1)*s + 2*s - 2*ceil(s/2)
static ggml_tensor* wn_convt1d(ggml_context* ctx, ggml_tensor* x, const wn_conv& wn, int stride) {
    ggml_tensor* w = wn_weight(wn);
    if (!w)
        return x;

    const int K = (int)w->ne[0];      // kernel_size = 2*stride
    const int pad = (stride + 1) / 2; // ceil(stride/2), matches Python

    if (wn.w_perm) {
        ggml_tensor* b = wn.b ? ggml_cast(ctx, wn.b, GGML_TYPE_F32) : nullptr;
        return core_convt::convt1d_decomp(ctx, x, wn.w_perm, b, stride, K, pad, pad);
    }

    const int Cout = (int)w->ne[1];
    const int T = (int)x->ne[1];
    // PyTorch output: (T-1)*stride - 2*pad + K
    const int T_out = (T - 1) * stride - 2 * pad + K;

    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    // ggml_conv_transpose_1d with p0=0 produces raw output without cropping
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xt, stride, 0, 1);
    // Raw output length from ggml: (T-1)*stride + K
    // Crop `pad` from left, rest from right to get T_out
    int crop_left = pad;
    int crop_right = (T - 1) * stride + K - T_out - crop_left;
    (void)crop_right;
    // View into y starting at offset crop_left, taking T_out elements
    y = ggml_view_2d(ctx, y, T_out, Cout, y->nb[1], (size_t)crop_left * sizeof(float));
    y = ggml_cont(ctx, y);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C_out, T_out)
    if (wn.b) {
        y = ggml_add(ctx, y, ggml_cast(ctx, wn.b, GGML_TYPE_F32));
    }
    return y;
}

// ResidualUnit: Snake → Conv(k=7,dil) → Snake → Conv(k=1)
static ggml_tensor* res_unit(ggml_context* ctx, ggml_tensor* x, const tada_codec_res_unit& ru, int dilation) {
    ggml_tensor* y = snake1d(ctx, x, ru.alpha0, ru.inv_alpha0);
    y = wn_conv1d(ctx, y, ru.conv0, dilation);
    y = snake1d(ctx, y, ru.alpha1, ru.inv_alpha1);
    y = wn_conv1d(ctx, y, ru.conv1, /*dilation*/ 1);
    return ggml_add(ctx, x, y);
}

// DecoderBlock: Snake → ConvT(stride) → 3× ResUnit(d=1,3,9)
static ggml_tensor* dec_block(ggml_context* ctx, ggml_tensor* x, const tada_codec_dec_block& blk, int stride) {
    x = snake1d(ctx, x, blk.snake_alpha, blk.inv_snake_alpha);
    x = wn_convt1d(ctx, x, blk.up_conv, stride);
    static const int dilations[3] = {1, 3, 9};
    for (int r = 0; r < 3; r++) {
        x = res_unit(ctx, x, blk.res[r], dilations[r]);
    }
    return x;
}

// ──────────────────── full decode graph ─────────────────────────────

static ggml_cgraph* build_decode_graph(tada_codec_context* c, int n_frames) {
    const int d = c->hidden_dim;    // 1024
    const int ed = c->embed_dim;    // 512
    const int nh = c->n_attn_heads; // 8
    const int hd = c->head_dim;     // 128
    const float eps = 1e-5f;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    // Large graph — attention + DAC decoder has many ops
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input: (embed_dim, n_frames) = (512, T)
    ggml_tensor* features = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ed, n_frames);
    ggml_set_name(features, "features");
    ggml_set_input(features);

    // 1. Linear projection: (512, T) → (1024, T)
    ggml_tensor* cur = ggml_mul_mat(ctx0, c->proj_w, features);
    if (c->proj_b)
        cur = ggml_add(ctx0, cur, c->proj_b);

    // Dump: codec_proj
    ggml_tensor* dump_proj = ggml_cont(ctx0, cur);
    ggml_set_name(dump_proj, "dump_proj");
    ggml_build_forward_expand(gf, dump_proj);

    // 2. Local attention encoder (6 layers) with v2 block attention mask
    // Mask input: (T, T) F16 — True=masked, False=attend. Built from token_masks on CPU.
    ggml_tensor* attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_frames, n_frames);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // Create position tensor for RoPE ONCE (shared across all layers)
    ggml_tensor* codec_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_name(codec_pos, "codec_pos");
    ggml_set_input(codec_pos);

    for (int il = 0; il < c->n_attn_layers; il++) {
        const auto& l = c->attn_layers[il];

        // Self-attention (post-norm)
        // QKV projection
        ggml_tensor* qkv = ggml_mul_mat(ctx0, l.qkv_w, cur); // (3*d, T)
        if (l.qkv_b)
            qkv = ggml_add(ctx0, qkv, l.qkv_b);

        // Split into Q, K, V — each (d, T)
        ggml_tensor* q = ggml_view_2d(ctx0, qkv, d, n_frames, qkv->nb[1], 0);
        ggml_tensor* k = ggml_view_2d(ctx0, qkv, d, n_frames, qkv->nb[1], (size_t)d * sizeof(float));
        ggml_tensor* v = ggml_view_2d(ctx0, qkv, d, n_frames, qkv->nb[1], (size_t)2 * d * sizeof(float));
        q = ggml_cont(ctx0, q);
        k = ggml_cont(ctx0, k);
        v = ggml_cont(ctx0, v);

        // Reshape for multi-head: (d, T) → (hd, nh, T)
        q = ggml_reshape_3d(ctx0, q, hd, nh, n_frames);
        k = ggml_reshape_3d(ctx0, k, hd, nh, n_frames);
        v = ggml_reshape_3d(ctx0, v, hd, nh, n_frames);

        // RoPE — attn_factor=1.0 is critical (0.0 would zero the output)
        q = ggml_rope_ext(ctx0, q, codec_pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        k = ggml_rope_ext(ctx0, k, codec_pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);

        // Permute to (hd, T, nh) for attention
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);

        // Attention: softmax(Q @ K^T / sqrt(hd)) @ V
        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, q, k, v, attn_mask, scale, 0, 0);
        // Result: (hd, T, nh) → reshape to (d, T)
        attn = ggml_cont(ctx0, attn);
        attn = ggml_reshape_2d(ctx0, attn, d, n_frames);

        // Output projection
        attn = ggml_mul_mat(ctx0, l.out_w, attn);
        if (l.out_b)
            attn = ggml_add(ctx0, attn, l.out_b);

        // Post-norm residual: cur = LayerNorm(cur + attn)
        cur = ggml_add(ctx0, cur, attn);
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, l.attn_norm_w);
        if (l.attn_norm_b)
            cur = ggml_add(ctx0, cur, l.attn_norm_b);

        // FFN: Linear(1024,4096) → GELU → Linear(4096,1024)
        ggml_tensor* ffn = ggml_mul_mat(ctx0, l.ffn_up_w, cur);
        if (l.ffn_up_b)
            ffn = ggml_add(ctx0, ffn, l.ffn_up_b);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = ggml_mul_mat(ctx0, l.ffn_down_w, ffn);
        if (l.ffn_down_b)
            ffn = ggml_add(ctx0, ffn, l.ffn_down_b);

        // Post-norm residual: cur = LayerNorm(cur + ffn)
        cur = ggml_add(ctx0, cur, ffn);
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, l.ffn_norm_w);
        if (l.ffn_norm_b)
            cur = ggml_add(ctx0, cur, l.ffn_norm_b);

        if (il == 0) {
            ggml_tensor* d_l0 = ggml_cont(ctx0, cur);
            ggml_set_name(d_l0, "dump_layer0");
            ggml_build_forward_expand(gf, d_l0);
        }
    }

    // Final norm
    cur = ggml_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->final_norm_w);
    if (c->final_norm_b)
        cur = ggml_add(ctx0, cur, c->final_norm_b);

    // Dump: codec_attn_out
    ggml_tensor* dump_attn = ggml_cont(ctx0, cur);
    ggml_set_name(dump_attn, "dump_attn");
    ggml_build_forward_expand(gf, dump_attn);

    // 3. DAC decoder
    // Input conv: (1024, T) → (1536, T)
    cur = wn_conv1d(ctx0, cur, c->in_conv, /*dilation*/ 1);
    ggml_tensor* dump_dac_in = ggml_cont(ctx0, cur);
    ggml_set_name(dump_dac_in, "dump_dac_in");
    ggml_build_forward_expand(gf, dump_dac_in);

    // 4 decoder blocks with strides [4,4,5,6]
    // Inline block 0 for debugging
    {
        auto& blk = c->blocks[0];
        cur = snake1d(ctx0, cur, blk.snake_alpha, blk.inv_snake_alpha);
        ggml_tensor* d_s = ggml_cont(ctx0, cur);
        ggml_set_name(d_s, "dump_b0_snake");
        ggml_build_forward_expand(gf, d_s);
        cur = wn_convt1d(ctx0, cur, blk.up_conv, c->strides[0]);
        ggml_tensor* d_c = ggml_cont(ctx0, cur);
        ggml_set_name(d_c, "dump_b0_convt");
        ggml_build_forward_expand(gf, d_c);
        static const int dils[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            cur = res_unit(ctx0, cur, blk.res[r], dils[r]);
        }
    }
    for (int b = 1; b < 4; b++) {
        cur = dec_block(ctx0, cur, c->blocks[b], c->strides[b]);
        char dname[32];
        snprintf(dname, sizeof(dname), "dump_blk%d", b);
        ggml_tensor* dblk = ggml_cont(ctx0, cur);
        ggml_set_name(dblk, dname);
        ggml_build_forward_expand(gf, dblk);
    }
    ggml_tensor* dump_dac_out = ggml_cont(ctx0, cur);
    ggml_set_name(dump_dac_out, "dump_dac_out");
    ggml_build_forward_expand(gf, dump_dac_out);

    // Output: Snake → Conv1d(96, 1, k=7) → Tanh
    cur = snake1d(ctx0, cur, c->out_snake_alpha, c->out_inv_snake_alpha);
    cur = wn_conv1d(ctx0, cur, c->out_conv, /*dilation*/ 1);
    cur = ggml_tanh(ctx0, cur);

    // cur is (1, T_audio) — flatten to 1D
    int64_t T_audio = cur->ne[1];
    cur = ggml_reshape_1d(ctx0, cur, T_audio);
    ggml_set_name(cur, "pcm");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// ──────────── DAC-only graph (bypass attention) ────────────────────

static ggml_cgraph* build_dac_only_graph(tada_codec_context* c, int n_frames) {
    const int d = c->hidden_dim; // 1024

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input: (hidden_dim, n_frames) = (1024, T) — directly from attention output
    ggml_tensor* cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_frames);
    ggml_set_name(cur, "hidden_input");
    ggml_set_input(cur);

    // DAC decoder (same as in build_decode_graph, starting from in_conv)
    cur = wn_conv1d(ctx0, cur, c->in_conv, 1);
    ggml_tensor* dump_dac_in = ggml_cont(ctx0, cur);
    ggml_set_name(dump_dac_in, "dump_dac_in");
    ggml_build_forward_expand(gf, dump_dac_in);

    {
        auto& blk = c->blocks[0];
        cur = snake1d(ctx0, cur, blk.snake_alpha, blk.inv_snake_alpha);
        ggml_tensor* d_s = ggml_cont(ctx0, cur);
        ggml_set_name(d_s, "dump_b0_snake");
        ggml_build_forward_expand(gf, d_s);
        cur = wn_convt1d(ctx0, cur, blk.up_conv, c->strides[0]);
        ggml_tensor* d_c = ggml_cont(ctx0, cur);
        ggml_set_name(d_c, "dump_b0_convt");
        ggml_build_forward_expand(gf, d_c);
        static const int dils[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++)
            cur = res_unit(ctx0, cur, blk.res[r], dils[r]);
    }
    for (int b = 1; b < 4; b++) {
        cur = dec_block(ctx0, cur, c->blocks[b], c->strides[b]);
        char dname[32];
        snprintf(dname, sizeof(dname), "dump_blk%d", b);
        ggml_tensor* dblk = ggml_cont(ctx0, cur);
        ggml_set_name(dblk, dname);
        ggml_build_forward_expand(gf, dblk);
    }

    cur = snake1d(ctx0, cur, c->out_snake_alpha, c->out_inv_snake_alpha);
    cur = wn_conv1d(ctx0, cur, c->out_conv, 1);
    cur = ggml_tanh(ctx0, cur);
    int64_t T_audio = cur->ne[1];
    cur = ggml_reshape_1d(ctx0, cur, T_audio);
    ggml_set_name(cur, "pcm");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// ──────────────────────── public API ────────────────────────────────

static tada_codec_context* tada_codec_init_from_file_impl(const char* path, int n_threads, bool use_gpu,
                                                          ggml_backend_t shared_backend,
                                                          ggml_backend_t shared_backend_cpu) {
    auto* c = new tada_codec_context();
    c->n_threads = n_threads;
    c->compute_meta.resize(32 * 1024 * 1024); // 32 MB for large graph

    // Metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete c;
        return nullptr;
    }
    c->embed_dim = (int)core_gguf::kv_u32(meta, "tada_codec.embed_dim", 512);
    c->hidden_dim = (int)core_gguf::kv_u32(meta, "tada_codec.hidden_dim", 1024);
    c->n_attn_layers = (int)core_gguf::kv_u32(meta, "tada_codec.num_attn_layers", 6);
    c->n_attn_heads = (int)core_gguf::kv_u32(meta, "tada_codec.num_attn_heads", 8);
    c->head_dim = c->hidden_dim / c->n_attn_heads;
    c->wav_channels = (int)core_gguf::kv_u32(meta, "tada_codec.wav_channels", 1536);
    core_gguf::free_metadata(meta);

    // Channel config: 1536, 768, 384, 192, 96
    for (int i = 0; i < 5; i++) {
        c->channels[i] = c->wav_channels / (1 << i);
    }

    fprintf(stderr, "tada-codec: %dd, %d attn layers, %d heads, strides [4,4,5,6]\n", c->hidden_dim, c->n_attn_layers,
            c->n_attn_heads);

    // Backend
    if (shared_backend && shared_backend_cpu) {
        c->backend = shared_backend;
        c->backend_cpu = shared_backend_cpu;
        c->owns_backend = false;
    } else {
        c->backend_cpu = ggml_backend_cpu_init();
        if (!c->backend_cpu) {
            fprintf(stderr, "tada-codec: failed to init CPU backend\n");
            delete c;
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(c->backend_cpu, n_threads);
        c->backend = use_gpu ? ggml_backend_init_best() : c->backend_cpu;
        if (!c->backend)
            c->backend = c->backend_cpu;
    }

    // #192: the codec miscomputes on Vulkan/MoltenVK for non-trivial sequence
    // lengths, so run the whole codec on CPU when the GPU backend is Vulkan.
    // Diagnosis (per-stage TADA_CODEC_DUMP, Vulkan vs CPU on identical 522-frame
    // features): the attention encoder MATCHES across backends (dump_attn,
    // dump_layer0 identical), but the DAC decoder front-end explodes — dump_dac_in
    // (the in_conv Conv1d → im2col+mul_mat) jumps to rms ~37 / range ±800 on
    // Vulkan vs rms ~0.85 / ±8 on CPU, a ~43× structural blowup that propagates to
    // the output; the final Tanh masks the range but the audio is distorted →
    // empty/garbled ASR. It is NOT a precision issue (storing F32 weights changes
    // nothing — MoltenVK downconverts) and NOT a ggml_backend_sched cross-backend
    // bug (the codec runs as a single Vulkan split, no CPU offload). It is
    // size-dependent: short inputs ("Hello world") render fine on Vulkan; the
    // im2col/conv path breaks only past some sequence length — a MoltenVK conv
    // kernel bug. The CPU codec is bit-faithful (Metal, which renders these same
    // features correctly, confirms the features are good). Talker/FM keep their
    // native Vulkan path. Opt back into the broken native codec with
    // CRISPASR_TADA_CODEC_VULKAN_NATIVE=1 for debugging.
    if (c->backend != c->backend_cpu && std::strstr(ggml_backend_name(c->backend), "Vulkan")) {
        const char* keep = std::getenv("CRISPASR_TADA_CODEC_VULKAN_NATIVE");
        if (!(keep && keep[0] == '1')) {
            fprintf(stderr, "tada-codec: Vulkan backend detected — running codec on CPU (#192 Vulkan "
                            "conv miscompute at length; set CRISPASR_TADA_CODEC_VULKAN_NATIVE=1 to override)\n");
            c->backend = c->backend_cpu;
        }
    }

    fprintf(stderr, "tada-codec: backend=%s%s\n", ggml_backend_name(c->backend),
            (c->backend == c->backend_cpu) ? " (CPU)" : " + CPU fallback");

    // Weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "tada-codec", wl)) {
        fprintf(stderr, "tada-codec: failed to load weights\n");
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_weights(c)) {
        fprintf(stderr, "tada-codec: failed to bind tensors\n");
        delete c;
        return nullptr;
    }

    // Pre-compute 1/alpha for all Snake1d activations (avoids fused snake op)
    precompute_all_inv_alphas(c);

    // Permute ConvTranspose1d weights for decomposed col2im path
    {
        const int n = 4;
        ggml_tensor* srcs[4];
        ggml_tensor** dsts[4];
        for (int i = 0; i < n; i++) {
            srcs[i] = wn_weight(c->blocks[i].up_conv);
            dsts[i] = &c->blocks[i].up_conv.w_perm;
        }
        core_convt::permute_convt1d_weights_batch(srcs, dsts, n, c->backend, &c->ctx_perm, &c->buf_perm);
    }

    fprintf(stderr, "tada-codec: loaded OK (%zu tensors)\n", c->tensors.size());
    return c;
}

extern "C" {

struct tada_codec_context* tada_codec_init_from_file_ex(const char* path, int n_threads, bool use_gpu) {
    return tada_codec_init_from_file_impl(path, n_threads, use_gpu, nullptr, nullptr);
}

struct tada_codec_context* tada_codec_init_from_file_with_backend(const char* path, int n_threads,
                                                                  ggml_backend_t backend, ggml_backend_t backend_cpu) {
    return tada_codec_init_from_file_impl(path, n_threads, false, backend, backend_cpu);
}

struct tada_codec_context* tada_codec_init_from_file(const char* path, int n_threads) {
    return tada_codec_init_from_file_ex(path, n_threads, false);
}

float* tada_codec_decode(struct tada_codec_context* ctx, const float* features, int n_frames,
                         const int32_t* token_masks, int* out_n_samples) {
    if (!ctx || !features || n_frames <= 0)
        return nullptr;
    tada_codec_bench_stage _bs_total("decode_total");

    const int ed = ctx->embed_dim;

    ggml_cgraph* gf = build_decode_graph(ctx, n_frames);

    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    int n_be = (ctx->backend && ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "tada-codec: failed to alloc graph\n");
        ggml_backend_sched_free(sched);
        return nullptr;
    }

    // Set input: features (embed_dim, n_frames) column-major
    ggml_tensor* inp = ggml_graph_get_tensor(gf, "features");
    ggml_backend_tensor_set(inp, features, 0, (size_t)ed * n_frames * sizeof(float));

    // Set position IDs for RoPE in codec attention layers
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "codec_pos");
    if (pos_t) {
        std::vector<int32_t> positions(n_frames);
        for (int i = 0; i < n_frames; i++)
            positions[i] = i;
        ggml_backend_tensor_set(pos_t, positions.data(), 0, (size_t)n_frames * sizeof(int32_t));
    }
    // Build v2 block attention mask from token_masks
    ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "attn_mask");
    if (mask_t) {
        // Compute block IDs: cumsum(token_masks) - token_masks
        std::vector<int32_t> block_ids(n_frames);
        int block = 0;
        for (int i = 0; i < n_frames; i++) {
            if (token_masks[i])
                block++;
            block_ids[i] = block - (token_masks[i] ? 1 : 0);
        }

        // Build mask: can_attend[i][j] = (block_ids[j] == block_ids[i]) || (block_ids[j] == block_ids[i]-1)
        // ggml_flash_attn_ext mask: 0.0 = attend, -inf = masked
        std::vector<ggml_fp16_t> mask_data((size_t)n_frames * n_frames);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n_frames; i++) {
            for (int j = 0; j < n_frames; j++) {
                bool same = (block_ids[j] == block_ids[i]);
                bool prev = (block_ids[j] == block_ids[i] - 1);
                mask_data[(size_t)i * n_frames + j] = (same || prev) ? zero : neg_inf;
            }
        }
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tada-codec: graph compute failed\n");
        ggml_backend_sched_free(sched);
        return nullptr;
    }

    const bool dump_stats = [] {
        const char* e = std::getenv("TADA_CODEC_DUMP");
        return e && *e && *e != '0';
    }();
    if (dump_stats) {
        auto dump = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t)
                return;
            int n = (int)ggml_nelements(t);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, (size_t)n * sizeof(float));
            float rms = 0;
            for (int i = 0; i < n; i++)
                rms += buf[i] * buf[i];
            rms = std::sqrt(rms / n);
            float mn = *std::min_element(buf.begin(), buf.end());
            float mx = *std::max_element(buf.begin(), buf.end());
            fprintf(stderr, "  DUMP %s: n=%d range=[%.4f, %.4f] rms=%.4f\n", name, n, mn, mx, rms);
        };
        dump("dump_proj");
        dump("dump_attn");
        dump("dump_layer0");
        dump("dump_dac_in");
        dump("dump_b0_snake");
        dump("dump_b0_convt");
        dump("dump_blk0");
        dump("dump_blk1");
        dump("dump_blk2");
        dump("dump_blk3");
        dump("dump_dac_out");
    }

    ggml_tensor* pcm_t = ggml_graph_get_tensor(gf, "pcm");
    int n_samples = (int)ggml_nelements(pcm_t);
    float* pcm = (float*)malloc((size_t)n_samples * sizeof(float));
    ggml_backend_tensor_get(pcm_t, pcm, 0, (size_t)n_samples * sizeof(float));

    if (dump_stats) {
        float rms = 0, mn = pcm[0], mx = pcm[0];
        for (int i = 0; i < n_samples; i++) {
            rms += pcm[i] * pcm[i];
            if (pcm[i] < mn)
                mn = pcm[i];
            if (pcm[i] > mx)
                mx = pcm[i];
        }
        rms = std::sqrt(rms / n_samples);
        fprintf(stderr, "  DUMP pcm: n=%d range=[%.6f, %.6f] rms=%.6f\n", n_samples, mn, mx, rms);
    }

    ggml_backend_sched_free(sched);
    *out_n_samples = n_samples;
    return pcm;
}

float* tada_codec_extract_stage(struct tada_codec_context* ctx, const float* features, int n_frames,
                                const int32_t* token_masks, const char* stage, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !features || !stage || !*stage || n_frames <= 0)
        return nullptr;

    const int ed = ctx->embed_dim;
    ggml_cgraph* gf = build_decode_graph(ctx, n_frames);

    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    int n_be = (ctx->backend && ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "tada-codec: failed to alloc graph for stage '%s'\n", stage);
        ggml_backend_sched_free(sched);
        return nullptr;
    }

    ggml_tensor* inp = ggml_graph_get_tensor(gf, "features");
    ggml_backend_tensor_set(inp, features, 0, (size_t)ed * n_frames * sizeof(float));

    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "codec_pos");
    if (pos_t) {
        std::vector<int32_t> positions(n_frames);
        for (int i = 0; i < n_frames; i++)
            positions[i] = i;
        ggml_backend_tensor_set(pos_t, positions.data(), 0, (size_t)n_frames * sizeof(int32_t));
    }

    ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "attn_mask");
    if (mask_t) {
        std::vector<int32_t> block_ids(n_frames);
        int block = 0;
        for (int i = 0; i < n_frames; i++) {
            if (token_masks && token_masks[i])
                block++;
            block_ids[i] = block - ((token_masks && token_masks[i]) ? 1 : 0);
        }

        std::vector<ggml_fp16_t> mask_data((size_t)n_frames * n_frames);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n_frames; i++) {
            for (int j = 0; j < n_frames; j++) {
                bool same = (block_ids[j] == block_ids[i]);
                bool prev = (block_ids[j] == block_ids[i] - 1);
                mask_data[(size_t)i * n_frames + j] = (same || prev) ? zero : neg_inf;
            }
        }
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tada-codec: graph compute failed for stage '%s'\n", stage);
        ggml_backend_sched_free(sched);
        return nullptr;
    }

    ggml_tensor* t = ggml_graph_get_tensor(gf, stage);
    if (!t) {
        ggml_backend_sched_free(sched);
        return nullptr;
    }
    const int n = (int)ggml_nelements(t);
    float* out = (float*)malloc((size_t)n * sizeof(float));
    if (!out) {
        ggml_backend_sched_free(sched);
        return nullptr;
    }
    ggml_backend_tensor_get(t, out, 0, (size_t)n * sizeof(float));
    if (out_n)
        *out_n = n;
    ggml_backend_sched_free(sched);
    return out;
}

float* tada_codec_decode_dac(struct tada_codec_context* ctx, const float* hidden, int n_frames, int* out_n_samples) {
    if (!ctx || !hidden || n_frames <= 0)
        return nullptr;
    const int d = ctx->hidden_dim;

    ggml_cgraph* gf = build_dac_only_graph(ctx, n_frames);
    ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
    int n_be = (ctx->backend && ctx->backend != ctx->backend_cpu) ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "tada-codec: failed to alloc DAC graph\n");
        ggml_backend_sched_free(sched);
        return nullptr;
    }

    // Set input: hidden (hidden_dim, n_frames) — column-major, same as ggml internal
    ggml_tensor* inp = ggml_graph_get_tensor(gf, "hidden_input");
    ggml_backend_tensor_set(inp, hidden, 0, (size_t)d * n_frames * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tada-codec: DAC graph compute failed\n");
        ggml_backend_sched_free(sched);
        return nullptr;
    }

    // Dump intermediates
    auto dump = [&](const char* name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t)
            return;
        int n = (int)ggml_nelements(t);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, (size_t)n * sizeof(float));
        float rms = 0;
        for (int i = 0; i < n; i++)
            rms += buf[i] * buf[i];
        rms = std::sqrt(rms / n);
        fprintf(stderr, "  DAC %s: n=%d rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n", name, n, rms, buf[0], buf[1],
                buf[2], buf[3], buf[4]);
    };
    dump("dump_dac_in");
    dump("dump_b0_snake");
    dump("dump_b0_convt");
    dump("dump_blk1");
    dump("dump_blk2");
    dump("dump_blk3");

    ggml_tensor* pcm_t = ggml_graph_get_tensor(gf, "pcm");
    int n_samples = (int)ggml_nelements(pcm_t);
    float* pcm = (float*)malloc((size_t)n_samples * sizeof(float));
    ggml_backend_tensor_get(pcm_t, pcm, 0, (size_t)n_samples * sizeof(float));

    float rms = 0;
    for (int i = 0; i < n_samples; i++)
        rms += pcm[i] * pcm[i];
    rms = std::sqrt(rms / n_samples);
    fprintf(stderr, "  DAC pcm: n=%d rms=%.6f\n", n_samples, rms);

    ggml_backend_sched_free(sched);
    *out_n_samples = n_samples;
    return pcm;
}

void tada_codec_pcm_free(float* pcm) {
    free(pcm);
}

void tada_codec_free(struct tada_codec_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_inv)
        ggml_backend_buffer_free(ctx->buf_inv);
    if (ctx->ctx_inv)
        ggml_free(ctx->ctx_inv);
    if (ctx->buf_perm)
        ggml_backend_buffer_free(ctx->buf_perm);
    if (ctx->ctx_perm)
        ggml_free(ctx->ctx_perm);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->owns_backend) {
        if (ctx->backend && ctx->backend != ctx->backend_cpu)
            ggml_backend_free(ctx->backend);
        if (ctx->backend_cpu)
            ggml_backend_free(ctx->backend_cpu);
    }
    delete ctx;
}

} // extern "C"
