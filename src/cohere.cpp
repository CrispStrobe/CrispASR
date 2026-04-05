// cohere.cpp — Cohere Transcribe inference via ggml
//
// Architecture:
//   Encoder: Conv2D subsampling (×8) + 48-layer Conformer (Transformer-XL rel-pos attention)
//   Decoder: 8-layer causal transformer with cross-attention + KV cache
//   Features: on-the-fly preemphasis → STFT → mel filterbank → log → per-feature norm
//
// Tensor naming follows export_gguf.py / cohere-arch.h.

#include "cohere.h"
#include "cohere-arch.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdarg>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fftw3.h>
#include <cblas.h>
#if defined(__F16C__) && defined(__AVX2__)
#include <immintrin.h>
#define CT_HAVE_F16C 1
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Model hyperparams
// ---------------------------------------------------------------------------

struct cohere_hparams {
    int vocab_size   = 16384;
    // encoder
    int enc_n_layers = 48;
    int enc_d_model  = 1280;
    int enc_n_heads  = 8;
    int enc_head_dim = 160;
    int enc_ffn_dim  = 5120;
    int enc_conv_k   = 9;
    // decoder
    int dec_n_layers = 8;
    int dec_d_model  = 1024;
    int dec_n_heads  = 8;
    int dec_head_dim = 128;
    int dec_ffn_dim  = 4096;
    int dec_max_ctx  = 1024;
    // audio
    int sample_rate  = 16000;
    int n_mels       = 128;
    int n_fft        = 512;
    int hop_length   = 160;
    int win_length   = 400;
    // derived
    int n_freqs() const { return n_fft / 2 + 1; }            // 257
    int pre_conv_ch  = 256;
    int pre_sub_fac  = 8;  // 3 × stride-2 → ×8 downsampling
};

// ---------------------------------------------------------------------------
// Conformer layer weights
// ---------------------------------------------------------------------------

struct cohere_enc_layer {
    // FF1
    ggml_tensor * ff1_norm_w, * ff1_norm_b;
    ggml_tensor * ff1_up_w, * ff1_up_b;
    ggml_tensor * ff1_dn_w, * ff1_dn_b;
    // Self-attention (relative pos)
    ggml_tensor * attn_norm_w, * attn_norm_b;
    ggml_tensor * attn_q_w, * attn_q_b;
    ggml_tensor * attn_k_w, * attn_k_b;
    ggml_tensor * attn_v_w, * attn_v_b;
    ggml_tensor * attn_out_w, * attn_out_b;
    ggml_tensor * attn_pos_w;          // linear_pos  [d,d]
    ggml_tensor * attn_pos_bias_u;     // [heads, head_dim]
    ggml_tensor * attn_pos_bias_v;
    // Convolution module
    ggml_tensor * conv_norm_w, * conv_norm_b;
    ggml_tensor * conv_pw1_w, * conv_pw1_b; // pointwise1: [2d, d, 1]
    ggml_tensor * conv_dw_w,  * conv_dw_b;  // depthwise:  [d, 1, k]
    ggml_tensor * conv_bn_w,  * conv_bn_b;  // batch-norm scale/bias
    ggml_tensor * conv_bn_mean, * conv_bn_var;
    ggml_tensor * conv_pw2_w, * conv_pw2_b; // pointwise2: [d, d, 1]
    // FF2
    ggml_tensor * ff2_norm_w, * ff2_norm_b;
    ggml_tensor * ff2_up_w, * ff2_up_b;
    ggml_tensor * ff2_dn_w, * ff2_dn_b;
    // Output norm
    ggml_tensor * out_norm_w, * out_norm_b;

    // BatchNorm-folded depthwise conv weights (precomputed at model load time).
    // w_fused[d] = dw_w[d] * bn_gamma[d] / sqrt(bn_var[d] + 1e-5)
    // b_fused[d] = (dw_b[d] - bn_mean[d]) * bn_gamma[d] / sqrt(bn_var[d] + 1e-5) + bn_beta[d]
    // These are F32 host vectors used by both the legacy and ggml-graph paths.
    std::vector<float> conv_dw_w_fused; // [d * conv_k]
    std::vector<float> conv_dw_b_fused; // [d]
};

// ---------------------------------------------------------------------------
// Decoder layer weights
// ---------------------------------------------------------------------------

struct cohere_dec_layer {
    ggml_tensor * attn_ln_w, * attn_ln_b;
    ggml_tensor * attn_q_w,  * attn_q_b;
    ggml_tensor * attn_k_w,  * attn_k_b;
    ggml_tensor * attn_v_w,  * attn_v_b;
    ggml_tensor * attn_o_w,  * attn_o_b;
    ggml_tensor * cross_ln_w, * cross_ln_b;
    ggml_tensor * cross_q_w,  * cross_q_b;
    ggml_tensor * cross_k_w,  * cross_k_b;
    ggml_tensor * cross_v_w,  * cross_v_b;
    ggml_tensor * cross_o_w,  * cross_o_b;
    ggml_tensor * ffn_ln_w,  * ffn_ln_b;
    ggml_tensor * ffn_up_w,  * ffn_up_b;
    ggml_tensor * ffn_dn_w,  * ffn_dn_b;
};

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

struct cohere_model {
    cohere_hparams hparams;

    // Feature extraction
    ggml_tensor * fe_mel_fb; // [1, n_mels, n_freqs]
    ggml_tensor * fe_window; // [win_length]

    // Pre-encode subsampling
    ggml_tensor * pre_conv0_w, * pre_conv0_b;
    ggml_tensor * pre_conv2_w, * pre_conv2_b;
    ggml_tensor * pre_conv3_w, * pre_conv3_b;
    ggml_tensor * pre_conv5_w, * pre_conv5_b;
    ggml_tensor * pre_conv6_w, * pre_conv6_b;
    ggml_tensor * pre_out_w,   * pre_out_b;

    // Encoder layers
    std::vector<cohere_enc_layer> enc_layers;

    // Encoder→decoder projection
    ggml_tensor * enc_proj_w, * enc_proj_b;

    // Decoder
    ggml_tensor * dec_emb_w;
    ggml_tensor * dec_pos_w;
    ggml_tensor * dec_emb_ln_w, * dec_emb_ln_b;
    std::vector<cohere_dec_layer> dec_layers;
    ggml_tensor * dec_out_ln_w, * dec_out_ln_b;
    ggml_tensor * dec_head_w,   * dec_head_b;

    // ggml bookkeeping
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

struct cohere_vocab {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;

    int n_vocab() const { return (int)id_to_token.size(); }

    int token_id(const std::string & s) const {
        auto it = token_to_id.find(s);
        return it == token_to_id.end() ? -1 : it->second;
    }
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct cohere_context {
    cohere_model  model;
    cohere_vocab  vocab;
    cohere_context_params params;

    // Persistent KV cache tensors and their context
    struct ggml_context * kv_ctx = nullptr;
    struct ggml_tensor  * kv_k   = nullptr;
    struct ggml_tensor  * kv_v   = nullptr;

    // Cross-attention KV cache: computed once per utterance from encoder output.
    // Shape per layer: (T_enc, dec_d_model), row-major.
    std::vector<std::vector<float>> cross_kv_k; // [n_dec_layers][T_enc * dec_d_model]
    std::vector<std::vector<float>> cross_kv_v;

    // ggml backend for compute graph execution (CPU; GPU backends slot in here later)
    ggml_backend_t       ggml_backend = nullptr;
    ggml_backend_sched_t ggml_alloc   = nullptr;

    // Metadata context for graph node descriptors (no_alloc=true; actual buffers via gallocr)
    // Sized generously; only holds ggml_tensor_overhead() * N_NODES bytes.
    std::vector<uint8_t> compute_meta;

    // Cached T_enc from last encode call, needed by decode graph builder.
    int cached_T_enc = 0;

    // Mel spectrogram buffer
    std::vector<float> mel_buf;

    // Constants
    struct ggml_context * ctx_const = nullptr;
    struct ggml_tensor * eps = nullptr;
};

static void cohere_log_tensor(const char * name, const struct ggml_tensor * t);
static struct ggml_cgraph * cohere_build_graph_encoder(struct cohere_context * ctx, int T_mel);
static struct ggml_cgraph * cohere_build_graph_decoder(struct cohere_context * ctx, const int * tokens, int n_tokens, int offset);

// ---------------------------------------------------------------------------
// Encoder Graph Builder
// ---------------------------------------------------------------------------

static struct ggml_tensor * cohere_rel_shift(struct ggml_context * ctx, struct ggml_tensor * a) {
    const int T = a->ne[1];
    const int n_heads = a->ne[2];
    struct ggml_tensor * out = ggml_view_3d(ctx, a,
        T, T, n_heads,
        a->nb[1] - a->nb[0],
        a->nb[2],
        (T - 1) * a->nb[0]
    );
    return out;
}

static struct ggml_cgraph * cohere_build_graph_encoder(struct cohere_context * ctx, int T_mel) {
    const auto & model = ctx->model;
    const auto & hp    = model.hparams;
    const int d        = hp.enc_d_model;
    const int n_heads  = hp.enc_n_heads;
    const int head_dim = hp.enc_head_dim;
    const int n_mels   = hp.n_mels;

    struct ggml_init_params params = {
        .mem_size   = ctx->compute_meta.size(),
        .mem_buffer = ctx->compute_meta.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);

    // mel: [n_mels, T_mel]
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // --- Subsampling ---
    // conv.0: [n_mels, T_mel, 1] -> [64, T/2, 256]
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model.pre_conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv0_b, 1, 1, model.pre_conv0_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // conv.2: depthwise [64, T/2, 256] -> [32, T/4, 256]
    cur = ggml_conv_2d_dw(ctx0, model.pre_conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv2_b, 1, 1, model.pre_conv2_b->ne[0], 1), GGML_TYPE_F32));
    // conv.3: pointwise [32, T/4, 256] -> [32, T/4, 256]
    cur = ggml_conv_2d(ctx0, model.pre_conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv3_b, 1, 1, model.pre_conv3_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // conv.5: depthwise [32, T/4, 256] -> [16, T/8, 256]
    cur = ggml_conv_2d_dw(ctx0, model.pre_conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv5_b, 1, 1, model.pre_conv5_b->ne[0], 1), GGML_TYPE_F32));
    // conv.6: pointwise [16, T/8, 256] -> [16, T/8, 256]
    cur = ggml_conv_2d(ctx0, model.pre_conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv6_b, 1, 1, model.pre_conv6_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // Flatten: [W3, H3, 256] -> [W3, 256, H3] -> [W3*256, H3]
    int H3 = cur->ne[1];
    int W3 = cur->ne[0];
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3)); // [W3, C, H3]
    cur = ggml_reshape_2d(ctx0, cur, W3 * 256, H3);
    cohere_log_tensor("after_flatten", cur);

    // pre_out: [4096, H3] -> [d, H3]
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.pre_out_w, cur), model.pre_out_b);

    const int T = H3;

    // Positional encodings sinusoidal: [d, 2*T-1]
    struct ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto & layer = model.enc_layers[il];
        struct ggml_tensor * inpL = cur;

        // FF1
        struct ggml_tensor * ff1 = ggml_norm(ctx0, cur, 1e-5f);
        ff1 = ggml_add(ctx0, ggml_mul(ctx0, ff1, layer.ff1_norm_w), layer.ff1_norm_b);
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_up_w, ff1), layer.ff1_up_b);
        ff1 = ggml_mul(ctx0, ff1, ggml_sigmoid(ctx0, ff1));
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_dn_w, ff1), layer.ff1_dn_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, ff1, 0.5f));

        struct ggml_tensor * inpAtn = cur;

        // Self-Attention
        struct ggml_tensor * attn = ggml_norm(ctx0, cur, 1e-5f);
        attn = ggml_add(ctx0, ggml_mul(ctx0, attn, layer.attn_norm_w), layer.attn_norm_b);
        
        struct ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, attn), layer.attn_q_b);
        struct ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, attn), layer.attn_k_b);
        struct ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, attn), layer.attn_v_b);
        
        struct ggml_tensor * R = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_enc); // [d, 2T-1]
        
        struct ggml_tensor * Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_u, d));
        struct ggml_tensor * Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_v, d));
        
        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3); // [hd, T, H]
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3); // [hd, T, H]
        K   = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K,   head_dim, n_heads, T), 0, 2, 1, 3); // [hd, T, H]
        R   = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R,   head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3); // [hd, 2T-1, H]
        
        struct ggml_tensor * AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K), ggml_cont(ctx0, Q_u)); // [T, T, H]
        struct ggml_tensor * BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), ggml_cont(ctx0, Q_v)); // [2T-1, T, H]

        struct ggml_tensor * BD = cohere_rel_shift(ctx0, BD_raw); // [T, T, H]
        
        struct ggml_tensor * scores = ggml_add(ctx0, AC, BD);
        scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max(ctx0, scores);

        struct ggml_tensor * V_trans = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, n_heads, T), 1, 2, 0, 3); // [T, hd, H]
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_trans), scores); // [hd, T, H]

        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);

        attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_out_w, attn_out), layer.attn_out_b);
        cur = ggml_add(ctx0, inpAtn, attn_out);

        struct ggml_tensor * inpCnv = cur;

        // Convolution module
        struct ggml_tensor * cnv = ggml_norm(ctx0, cur, 1e-5f);
        cnv = ggml_add(ctx0, ggml_mul(ctx0, cnv, layer.conv_norm_w), layer.conv_norm_b);
        
        struct ggml_tensor * pw1_w = ggml_reshape_2d(ctx0, layer.conv_pw1_w, d, 2 * d);
        cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw1_w, cnv), layer.conv_pw1_b);

        struct ggml_tensor * cnv_gate = ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], d * sizeof(float));
        cnv = ggml_mul(ctx0, 
            ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], 0),
            ggml_sigmoid(ctx0, cnv_gate)
        );
        
        // cnv is [C, T] initially.
        // ggml_conv_1d_dw handles [T, C] input (ne0=T, ne1=C).
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv)); // [C, T] -> [T, C]
        cnv = ggml_conv_1d_dw(ctx0, layer.conv_dw_w, cnv, 1, (hp.enc_conv_k - 1) / 2, 1);
        
        // ggml_conv_1d_dw returns [T, 1, C, 1] (ne0=T, ne1=1, ne2=C)
        // Permute to [C, T, 1, 1] to match the rest of the layer expectation.
        // p0 is where old_ne0 (T) goes -> 1
        // p1 is where old_ne1 (1) goes -> 2
        // p2 is where old_ne2 (C) goes -> 0
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        
        // Add bias: reshape to [C, 1, 1, 1] then broadcast to [C, T, 1, 1]
        struct ggml_tensor * b = ggml_reshape_4d(ctx0, layer.conv_dw_b, d, 1, 1, 1);
        cnv = ggml_add(ctx0, cnv, b);
        
        // batchnorm: y = (x - mean) / sqrt(var + eps) * scale + bias
        struct ggml_tensor * bn_mean = ggml_reshape_4d(ctx0, layer.conv_bn_mean, d, 1, 1, 1);
        struct ggml_tensor * bn_var  = ggml_reshape_4d(ctx0, layer.conv_bn_var,  d, 1, 1, 1);
        struct ggml_tensor * bn_w    = ggml_reshape_4d(ctx0, layer.conv_bn_w,    d, 1, 1, 1);
        struct ggml_tensor * bn_b    = ggml_reshape_4d(ctx0, layer.conv_bn_b,    d, 1, 1, 1);

        cnv = ggml_sub(ctx0, cnv, bn_mean);
        cnv = ggml_mul(ctx0, cnv, ggml_div(ctx0, bn_w, ggml_sqrt(ctx0, ggml_add(ctx0, bn_var, ctx->eps))));
        cnv = ggml_add(ctx0, cnv, bn_b);
        
        cnv = ggml_mul(ctx0, cnv, ggml_sigmoid(ctx0, cnv)); // swish
        
        struct ggml_tensor * pw2_w = ggml_reshape_2d(ctx0, layer.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw2_w, cnv), layer.conv_pw2_b);
        cur = ggml_add(ctx0, inpCnv, cnv);

        struct ggml_tensor * inpFF2 = cur;

        // FF2
        struct ggml_tensor * ff2 = ggml_norm(ctx0, cur, 1e-5f);
        ff2 = ggml_add(ctx0, ggml_mul(ctx0, ff2, layer.ff2_norm_w), layer.ff2_norm_b);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_up_w, ff2), layer.ff2_up_b);
        ff2 = ggml_mul(ctx0, ff2, ggml_sigmoid(ctx0, ff2));
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_dn_w, ff2), layer.ff2_dn_b);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, ff2, 0.5f));

        // Final output norm for the layer
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.out_norm_w), layer.out_norm_b);
    }

    // Encoder-decoder projection
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.enc_proj_w, cur), model.enc_proj_b);
    ggml_set_name(cur, "enc_out");

    ggml_build_forward_expand(gf, cur);

    return gf;
}

static struct ggml_cgraph * cohere_build_graph_decoder(struct cohere_context * ctx, const int * tokens, int n_tokens, int offset);

#define CT_CHECK(x) do { if (!(x)) { fprintf(stderr, "CT_CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); abort(); } } while(0)

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float swish(float x)   { return x * sigmoid(x); }

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

#include "gguf.h"

static ggml_tensor * ct_get_tensor(cohere_model & model, const std::string & name) {
    auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
        fprintf(stderr, "cohere: tensor '%s' not found in GGUF\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

static ggml_tensor * ct_get_tensor_fmt(cohere_model & model, const char * fmt, int idx) {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt, idx);
    return ct_get_tensor(model, buf);
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static bool cohere_load_model(cohere_model & model,
                               cohere_vocab  & vocab,
                               const char * path,
                               ggml_backend_t backend) {
    // First pass: read metadata
    struct ggml_init_params meta_params = {
        .mem_size   = 4 * 1024 * 1024, // 4 MB for metadata
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * meta_ctx = ggml_init(meta_params);
    struct gguf_init_params load_params_meta = { .no_alloc = true, .ctx = &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path, load_params_meta);
    if (!gguf_ctx) {
        fprintf(stderr, "cohere: failed to open '%s'\n", path);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    auto & hp = model.hparams;
    auto kv_i = [&](const char * key) -> int {
        int ki = gguf_find_key(gguf_ctx, key);
        if (ki < 0) { fprintf(stderr, "cohere: missing key '%s'\n", key); return 0; }
        return (int)gguf_get_val_u32(gguf_ctx, ki);
    };

    hp.vocab_size   = kv_i(CT_KEY_VOCAB_SIZE);
    hp.enc_n_layers = kv_i(CT_KEY_ENC_N_LAYERS);
    hp.enc_d_model  = kv_i(CT_KEY_ENC_D_MODEL);
    hp.enc_n_heads  = kv_i(CT_KEY_ENC_N_HEADS);
    hp.enc_head_dim = kv_i(CT_KEY_ENC_HEAD_DIM);
    hp.enc_ffn_dim  = kv_i(CT_KEY_ENC_FFN_DIM);
    hp.enc_conv_k   = kv_i(CT_KEY_ENC_CONV_KERNEL);
    hp.dec_n_layers = kv_i(CT_KEY_DEC_N_LAYERS);
    hp.dec_d_model  = kv_i(CT_KEY_DEC_D_MODEL);
    hp.dec_n_heads = kv_i(CT_KEY_DEC_N_HEADS);
    hp.dec_head_dim = kv_i(CT_KEY_DEC_HEAD_DIM);
    hp.dec_ffn_dim  = kv_i(CT_KEY_DEC_FFN_DIM);
    hp.dec_max_ctx  = kv_i(CT_KEY_DEC_MAX_CTX);
    hp.n_mels       = kv_i(CT_KEY_AUDIO_N_MELS);
    hp.n_fft        = kv_i(CT_KEY_AUDIO_N_FFT);
    hp.hop_length   = kv_i(CT_KEY_AUDIO_HOP);
    hp.win_length   = kv_i(CT_KEY_AUDIO_WIN);

    // Load vocabulary
    {
        int ki = gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens");
        if (ki >= 0) {
            int n = gguf_get_arr_n(gguf_ctx, ki);
            cohere_debug("cohere: loading %d tokens from GGUF\n", n);
            vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                vocab.id_to_token[i] = gguf_get_arr_str(gguf_ctx, ki, i);
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
            const char * specials[] = {"<|startoftranscript|>", "<|en|>", "<|endoftext|>"};
            for (const char * s : specials) {
                cohere_debug("cohere: token '%s' -> ID %d\n", s, vocab.token_id(s));
            }
        } else {
            cohere_debug("cohere: WARNING: tokenizer.ggml.tokens not found in GGUF\n");
        }
    }

    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);

    // Second pass: load all tensor metadata (no_alloc=true)
    struct ggml_context * weight_ctx = nullptr;
    {
        struct gguf_init_params load_params = { .no_alloc = true, .ctx = &weight_ctx };
        gguf_ctx = gguf_init_from_file(path, load_params);
        if (!gguf_ctx || !weight_ctx) {
            fprintf(stderr, "cohere: failed to load tensor metadata from '%s'\n", path);
            return false;
        }

        // Create backend buffer for all tensors
        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        FILE * file = fopen(path, "rb");
        if (!file) {
            fprintf(stderr, "cohere: failed to open '%s' for reading\n", path);
            return false;
        }

        size_t data_offset = gguf_get_data_offset(gguf_ctx);

        for (ggml_tensor * t = ggml_get_first_tensor(weight_ctx); t;
             t = ggml_get_next_tensor(weight_ctx, t)) {
            model.tensors[ggml_get_name(t)] = t;

            // Read data from GGUF into backend tensor
            int64_t tensor_id = gguf_find_tensor(gguf_ctx, ggml_get_name(t));
            if (tensor_id < 0) continue;

            size_t t_offset = gguf_get_tensor_offset(gguf_ctx, tensor_id);
            fseek(file, data_offset + t_offset, SEEK_SET);
            std::vector<uint8_t> data(ggml_nbytes(t));
            if (fread(data.data(), 1, ggml_nbytes(t), file) != ggml_nbytes(t)) {
                fprintf(stderr, "cohere: failed to read data for tensor '%s'\n", ggml_get_name(t));
                continue;
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }

        fclose(file);

        model.ctx = weight_ctx;
        gguf_free(gguf_ctx);

        struct ggml_tensor * tp = model.tensors["enc.proj.weight"];
        if (tp) {
            std::vector<uint8_t> data(ggml_nbytes(tp));
            ggml_backend_tensor_get(tp, data.data(), 0, data.size());
            if (tp->type == GGML_TYPE_F16) {
                float v0 = ggml_fp16_to_fp32(((ggml_fp16_t*)data.data())[0]);
                cohere_debug("cohere: enc.proj.weight [0] (F16): %.4f\n", v0);
            } else if (tp->type == GGML_TYPE_F32) {
                cohere_debug("cohere: enc.proj.weight [0] (F32): %.4f\n", ((float*)data.data())[0]);
            }
        }
    }

    // Wire up model fields
    auto & m = model;
    auto T = [&](const char * name) { return ct_get_tensor(m, name); };
    auto TF = [&](const char * fmt, int i) { return ct_get_tensor_fmt(m, fmt, i); };

    m.fe_mel_fb = T(CT_FE_MEL_FB);
    m.fe_window = T(CT_FE_WINDOW);

    m.pre_conv0_w = T(CT_PRE_CONV0_W);  m.pre_conv0_b = T(CT_PRE_CONV0_B);
    m.pre_conv2_w = T(CT_PRE_CONV2_W);  m.pre_conv2_b = T(CT_PRE_CONV2_B);
    m.pre_conv3_w = T(CT_PRE_CONV3_W);  m.pre_conv3_b = T(CT_PRE_CONV3_B);
    m.pre_conv5_w = T(CT_PRE_CONV5_W);  m.pre_conv5_b = T(CT_PRE_CONV5_B);
    m.pre_conv6_w = T(CT_PRE_CONV6_W);  m.pre_conv6_b = T(CT_PRE_CONV6_B);
    m.pre_out_w   = T(CT_PRE_OUT_W);    m.pre_out_b   = T(CT_PRE_OUT_B);

    m.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        auto & l = m.enc_layers[i];
        l.ff1_norm_w  = TF(CT_ENC_FF1_NORM_W, i); l.ff1_norm_b  = TF(CT_ENC_FF1_NORM_B, i);
        l.ff1_up_w    = TF(CT_ENC_FF1_UP_W,   i); l.ff1_up_b    = TF(CT_ENC_FF1_UP_B,   i);
        l.ff1_dn_w    = TF(CT_ENC_FF1_DN_W,   i); l.ff1_dn_b    = TF(CT_ENC_FF1_DN_B,   i);
        l.attn_norm_w = TF(CT_ENC_ATN_NORM_W, i); l.attn_norm_b = TF(CT_ENC_ATN_NORM_B, i);
        l.attn_q_w    = TF(CT_ENC_ATN_Q_W,    i); l.attn_q_b    = TF(CT_ENC_ATN_Q_B,    i);
        l.attn_k_w    = TF(CT_ENC_ATN_K_W,    i); l.attn_k_b    = TF(CT_ENC_ATN_K_B,    i);
        l.attn_v_w    = TF(CT_ENC_ATN_V_W,    i); l.attn_v_b    = TF(CT_ENC_ATN_V_B,    i);
        l.attn_out_w  = TF(CT_ENC_ATN_OUT_W,  i); l.attn_out_b  = TF(CT_ENC_ATN_OUT_B,  i);
        l.attn_pos_w  = TF(CT_ENC_ATN_POS_W,  i);
        l.attn_pos_bias_u = TF(CT_ENC_ATN_POS_U, i);
        l.attn_pos_bias_v = TF(CT_ENC_ATN_POS_V, i);
        l.conv_norm_w = TF(CT_ENC_CNV_NORM_W, i); l.conv_norm_b = TF(CT_ENC_CNV_NORM_B, i);
        l.conv_pw1_w  = TF(CT_ENC_CNV_PW1_W,  i); l.conv_pw1_b  = TF(CT_ENC_CNV_PW1_B,  i);
        l.conv_dw_w   = TF(CT_ENC_CNV_DW_W,   i); l.conv_dw_b   = TF(CT_ENC_CNV_DW_B,   i);
        l.conv_bn_w   = TF(CT_ENC_CNV_BN_W,   i); l.conv_bn_b   = TF(CT_ENC_CNV_BN_B,   i);
        l.conv_bn_mean = TF(CT_ENC_CNV_BN_MEAN, i);
        l.conv_bn_var  = TF(CT_ENC_CNV_BN_VAR,  i);
        l.conv_pw2_w  = TF(CT_ENC_CNV_PW2_W,  i); l.conv_pw2_b  = TF(CT_ENC_CNV_PW2_B,  i);
        l.ff2_norm_w  = TF(CT_ENC_FF2_NORM_W, i); l.ff2_norm_b  = TF(CT_ENC_FF2_NORM_B, i);
        l.ff2_up_w    = TF(CT_ENC_FF2_UP_W,   i); l.ff2_up_b    = TF(CT_ENC_FF2_UP_B,   i);
        l.ff2_dn_w    = TF(CT_ENC_FF2_DN_W,   i); l.ff2_dn_b    = TF(CT_ENC_FF2_DN_B,   i);
        l.out_norm_w  = TF(CT_ENC_OUT_NORM_W, i); l.out_norm_b  = TF(CT_ENC_OUT_NORM_B, i);
    }

    m.enc_proj_w = T(CT_ENC_PROJ_W);  m.enc_proj_b = T(CT_ENC_PROJ_B);

    m.dec_emb_w    = T(CT_DEC_EMB_W);
    m.dec_pos_w    = T(CT_DEC_POS_W);
    m.dec_emb_ln_w = T(CT_DEC_EMB_LN_W);  m.dec_emb_ln_b = T(CT_DEC_EMB_LN_B);

    m.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; i++) {
        auto & l = m.dec_layers[i];
        l.attn_ln_w = TF(CT_DEC_ATTN_LN_W, i); l.attn_ln_b = TF(CT_DEC_ATTN_LN_B, i);
        l.attn_q_w  = TF(CT_DEC_ATTN_Q_W,  i); l.attn_q_b  = TF(CT_DEC_ATTN_Q_B,  i);
        l.attn_k_w  = TF(CT_DEC_ATTN_K_W,  i); l.attn_k_b  = TF(CT_DEC_ATTN_K_B,  i);
        l.attn_v_w  = TF(CT_DEC_ATTN_V_W,  i); l.attn_v_b  = TF(CT_DEC_ATTN_V_B,  i);
        l.attn_o_w  = TF(CT_DEC_ATTN_O_W,  i); l.attn_o_b  = TF(CT_DEC_ATTN_O_B,  i);
        l.cross_ln_w = TF(CT_DEC_XATTN_LN_W, i); l.cross_ln_b = TF(CT_DEC_XATTN_LN_B, i);
        l.cross_q_w  = TF(CT_DEC_XATTN_Q_W,  i); l.cross_q_b  = TF(CT_DEC_XATTN_Q_B,  i);
        l.cross_k_w  = TF(CT_DEC_XATTN_K_W,  i); l.cross_k_b  = TF(CT_DEC_XATTN_K_B,  i);
        l.cross_v_w  = TF(CT_DEC_XATTN_V_W,  i); l.cross_v_b  = TF(CT_DEC_XATTN_V_B,  i);
        l.cross_o_w  = TF(CT_DEC_XATTN_O_W,  i); l.cross_o_b  = TF(CT_DEC_XATTN_O_B,  i);
        l.ffn_ln_w  = TF(CT_DEC_FFN_LN_W,  i); l.ffn_ln_b  = TF(CT_DEC_FFN_LN_B,  i);
        l.ffn_up_w  = TF(CT_DEC_FFN_UP_W,  i); l.ffn_up_b  = TF(CT_DEC_FFN_UP_B,  i);
        l.ffn_dn_w  = TF(CT_DEC_FFN_DN_W,  i); l.ffn_dn_b  = TF(CT_DEC_FFN_DN_B,  i);
    }

    m.dec_out_ln_w = T(CT_DEC_OUT_LN_W);  m.dec_out_ln_b = T(CT_DEC_OUT_LN_B);
    m.dec_head_w   = T(CT_DEC_HEAD_W);    m.dec_head_b   = T(CT_DEC_HEAD_B);

    return true;
}

// ---------------------------------------------------------------------------
// Utility: get F32 value from a ggml tensor (any dtype)
// ---------------------------------------------------------------------------

static float ct_f32(const ggml_tensor * t, int i0, int i1 = 0, int i2 = 0, int i3 = 0) {
    return ggml_get_f32_nd(const_cast<ggml_tensor*>(t), i0, i1, i2, i3);
}

// ---------------------------------------------------------------------------
// Layer norm (on float buffer, in-place)
// ---------------------------------------------------------------------------

static void ct_layer_norm(float * out, const float * in, int n,
                          const float * w, const float * b, float eps = 1e-5f) {
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = in[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < n; i++) out[i] = (in[i] - (float)mean) * inv * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// SILU / Swish activation
// ---------------------------------------------------------------------------

static void ct_swish_inplace(float * x, int n) {
    for (int i = 0; i < n; i++) x[i] = swish(x[i]);
}

// ---------------------------------------------------------------------------
// Fast F16→F32 conversion using AVX2/F16C hardware instructions.
// Writes n floats into dst. Falls back to scalar bit-manipulation if no F16C.
// ---------------------------------------------------------------------------
static void ct_f16_to_f32(const uint16_t * src, float * dst, int n) {
#ifdef CT_HAVE_F16C
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
        __m256  f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t r;
        if      (exp == 0 && mant == 0) r = sign;
        else if (exp == 31)             r = sign | 0x7F800000u | (mant << 13);
        else                            r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        memcpy(dst + i, &r, sizeof(float));
    }
#else
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t r;
        if      (exp == 0 && mant == 0) r = sign;
        else if (exp == 31)             r = sign | 0x7F800000u | (mant << 13);
        else                            r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        memcpy(dst + i, &r, sizeof(float));
    }
#endif
}

// Thread-local F32 buffer for on-the-fly F16→F32 weight conversion.
// Grows to max weight tensor size (~26 MB for enc ffn); never freed.
static thread_local std::vector<float> tl_w_buf;

// Convert an F16 ggml_tensor's data to F32 in tl_w_buf; return pointer.
// Valid until the next call to ct_tensor_f32() on the same thread.
static const float * ct_tensor_f32(const ggml_tensor * t) {
    int n = (int)ggml_nelements(t);
    if ((int)tl_w_buf.size() < n) tl_w_buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        ct_f16_to_f32((const uint16_t *)t->data, tl_w_buf.data(), n);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(tl_w_buf.data(), t->data, (size_t)n * sizeof(float));
    } else {
        for (int i = 0; i < n; i++) tl_w_buf[i] = ggml_get_f32_1d(const_cast<ggml_tensor *>(t), i);
    }
    return tl_w_buf.data();
}

// ---------------------------------------------------------------------------
// Linear layer: out[m, T] = w[m, n] × in[n, T]  (weight in row-major out×in)
// Returns newly allocated buffer (caller frees).
// ---------------------------------------------------------------------------

// ct_linear: out (T × n_out) = in (T × n_in) @ w^T (n_in × n_out) + b
// Uses OpenBLAS SGEMM for ~10-30× speedup over scalar loops.
static std::vector<float> ct_linear(const float * in, int n_in, int T,
                                    const float * w, int n_out,
                                    const float * b = nullptr) {
    std::vector<float> out(n_out * T, 0.0f);
    // out (T, n_out) = in (T, n_in) * w^T (n_in, n_out)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, n_out, n_in,
                1.0f, in, n_in, w, n_in,
                0.0f, out.data(), n_out);
    if (b) {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < n_out; o++)
                out[t * n_out + o] += b[o];
    }
    return out;
}

// ct_linear_into: like ct_linear but writes into caller-supplied buffer (no malloc if already sized).
static void ct_linear_into(std::vector<float>& out, const float * in, int n_in, int T,
                            const float * w, int n_out, const float * b = nullptr) {
    out.resize((size_t)T * n_out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, n_out, n_in,
                1.0f, in, n_in, w, n_in,
                0.0f, out.data(), n_out);
    if (b) {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < n_out; o++)
                out[t * n_out + o] += b[o];
    }
}

// Convert a ggml tensor's data to a float32 std::vector (host copy).
// Uses direct memory access for F16/F32 tensors to avoid dependence on the
// ggml_table_f32_f16 lookup table (which requires ggml_init() to be called).
static std::vector<float> ct_to_f32(const ggml_tensor * t) {
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        const float * src = (const float *)t->data;
        for (int i = 0; i < n; i++) out[i] = src[i];
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t * src = (uint16_t *)t->data;
        for (int i = 0; i < n; i++) {
            uint16_t h = src[i];
            uint32_t sign = (uint32_t)(h >> 15) << 31;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t r;
            if (exp == 0 && mant == 0) {
                r = sign;  // +/- zero
            } else if (exp == 31) {
                r = sign | 0x7F800000u | (mant << 13);  // inf or NaN
            } else {
                r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);  // normal
            }
            memcpy(&out[i], &r, sizeof(float));
        }
    } else {
        // fallback for other types (e.g. quantised) — requires ggml_init
        for (int i = 0; i < n; i++) out[i] = ggml_get_f32_1d(const_cast<ggml_tensor*>(t), i);
    }
    return out;
}

// Memoized ct_to_f32: converts once at model load time, returns const ref forever.
// Populated eagerly by cohere_model_warm_cache(); safe because weights are immutable.
static const std::vector<float> & ct_to_f32_ref(const ggml_tensor * t) {
    static std::unordered_map<const ggml_tensor *, std::vector<float>> s_f32_cache;
    auto it = s_f32_cache.find(t);
    if (it != s_f32_cache.end()) return it->second;
    return s_f32_cache.emplace(t, ct_to_f32(t)).first->second;
}

// ---------------------------------------------------------------------------
// Feature extraction: raw PCM → log-mel spectrogram
// Returns float array of shape (n_mels, T_mel), row-major.
// ---------------------------------------------------------------------------

static std::vector<float> cohere_compute_features(const cohere_hparams & hp,
                                                   const float * fe_mel_fb_data,
                                                   const float * fe_window_data,
                                                   const float * samples, int n_samples,
                                                   int & T_out) {
    const int n_fft     = hp.n_fft;
    const int hop       = hp.hop_length;
    const int win       = hp.win_length;
    const int n_freqs   = hp.n_freqs();
    const int n_mels    = hp.n_mels;
    const float preemph = 0.97f;
    const float log_grd = (float)(1.0 / (1 << 24));

    // Pre-emphasis
    std::vector<float> pe(n_samples);
    pe[0] = samples[0];
    for (int i = 1; i < n_samples; i++) pe[i] = samples[i] - preemph * samples[i-1];

    // Center-pad
    int pad = n_fft / 2;
    std::vector<float> padded(pad + n_samples + pad, 0.0f);
    memcpy(padded.data() + pad, pe.data(), n_samples * sizeof(float));

    // Number of frames
    int n_pad = (int)padded.size();
    int T = (n_pad - n_fft) / hop + 1;
    T_out = T;

    // Hann window (from fe_window tensor, length win_length, padded to n_fft)
    std::vector<float> window(n_fft, 0.0f);
    int lpad = (n_fft - win) / 2;
    for (int i = 0; i < win; i++) window[lpad + i] = fe_window_data[i];

    // STFT → power spectrum → mel → log → normalize
    // Use FFTW3f real-to-complex FFT: O(n_fft·log n_fft) vs O(n_fft²) for direct DFT.
    std::vector<float> power(n_freqs * T, 0.0f);
    {
        std::vector<float>           fft_in(n_fft);
        std::vector<fftwf_complex>   fft_out(n_freqs);
        fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft, fft_in.data(), fft_out.data(), FFTW_ESTIMATE);
        for (int t = 0; t < T; t++) {
            const float * frame = padded.data() + t * hop;
            for (int n = 0; n < n_fft; n++) fft_in[n] = frame[n] * window[n];
            fftwf_execute(plan);
            for (int k = 0; k < n_freqs; k++) {
                float re = fft_out[k][0], im = fft_out[k][1];
                power[t * n_freqs + k] = re*re + im*im;
            }
        }
        fftwf_destroy_plan(plan);
    }

    // mel filterbank: fe_mel_fb shape [1, n_mels, n_freqs]
    std::vector<float> mel(n_mels * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            float v = 0.0f;
            for (int f = 0; f < n_freqs; f++) {
                v += fe_mel_fb_data[m * n_freqs + f] * power[t * n_freqs + f];
            }
            mel[t * n_mels + m] = logf(v + log_grd);
        }
    }

    // Per-feature normalization: biased std (matches ONNX: std = sqrt(mean(diff²)))
    for (int m = 0; m < n_mels; m++) {
        double mean = 0.0, var = 0.0;
        for (int t = 0; t < T; t++) mean += mel[t * n_mels + m];
        mean /= T;
        for (int t = 0; t < T; t++) { double d = mel[t * n_mels + m] - mean; var += d*d; }
        float std = sqrtf((float)(var / T + 1e-5));
        for (int t = 0; t < T; t++) mel[t * n_mels + m] = (mel[t * n_mels + m] - (float)mean) / std;
    }

    cohere_debug("cohere: mel m=0, t=0..4: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           mel[0*n_mels + 0], mel[1*n_mels + 0], mel[2*n_mels + 0], mel[3*n_mels + 0], mel[4*n_mels + 0]);
    cohere_debug("cohere: mel t=0, m=0..4: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           mel[0*n_mels + 0], mel[0*n_mels + 1], mel[0*n_mels + 2], mel[0*n_mels + 3], mel[0*n_mels + 4]);

    return mel; // shape: [n_mels, T], time-major
}

// ---------------------------------------------------------------------------
// Conformer: sinusoidal relative positional encoding
// Returns (2T-1, d_model) array, positions from T-1 to -(T-1)
// ---------------------------------------------------------------------------

static std::vector<float> ct_rel_pos_enc(int T, int d_model) {
    int n_pos = 2 * T - 1;
    std::vector<float> pe(n_pos * d_model, 0.0f);
    for (int i = 0; i < n_pos; i++) {
        float pos = (float)(T - 1 - i); // T-1 down to -(T-1)
        for (int j = 0; j < d_model / 2; j++) {
            float div = powf(10000.0f, 2.0f * j / d_model);
            pe[i * d_model + 2*j]   = sinf(pos / div);
            pe[i * d_model + 2*j+1] = cosf(pos / div);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Pre-compute cross-attention K and V for all decoder layers.
// Call once per utterance after encoding; results stored in cross_kv_k/v.
// Layout: cross_kv_k[li] has shape (T_enc, dec_d_model), row-major.
// ---------------------------------------------------------------------------

static void cohere_precompute_cross_kv(
    const cohere_model & m,
    const float * enc_out, int T_enc,
    std::vector<std::vector<float>> & cross_kv_k,
    std::vector<std::vector<float>> & cross_kv_v
) {
    const auto & hp = m.hparams;
    const int d = hp.dec_d_model;
    cross_kv_k.resize(hp.dec_n_layers);
    cross_kv_v.resize(hp.dec_n_layers);
    for (int li = 0; li < hp.dec_n_layers; li++) {
        const auto & l = m.dec_layers[li];
        const float * ck_w = ct_tensor_f32(l.cross_k_w);
        cross_kv_k[li] = ct_linear(enc_out, d, T_enc, ck_w, d, ct_to_f32_ref(l.cross_k_b).data());
        const float * cv_w = ct_tensor_f32(l.cross_v_w);
        cross_kv_v[li] = ct_linear(enc_out, d, T_enc, cv_w, d, ct_to_f32_ref(l.cross_v_b).data());
    }
}

// ---------------------------------------------------------------------------
// Logging & Debugging
// ---------------------------------------------------------------------------

static void cohere_debug(const char * fmt, ...) {
    if (!getenv("COHERE_DEBUG")) return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

static void cohere_log_tensor(const char * name, const struct ggml_tensor * t) {
    if (!getenv("COHERE_DEBUG")) return;
    if (!t) {
        cohere_debug("%-25s: NULL\n", name);
        return;
    }
    cohere_debug("%-25s: shape [%4ld, %4ld, %4ld, %4ld] type %d\n",
        name,
        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3],
        (int)t->type);
}

// ---------------------------------------------------------------------------
// Decoder Graph Builder
// ---------------------------------------------------------------------------

static struct ggml_cgraph * cohere_build_graph_decoder(
    struct cohere_context * ctx,
    const int * tokens, int n_tokens, int offset
) {
    const auto & model = ctx->model;
    const auto & hp    = model.hparams;
    const int d        = hp.dec_d_model;
    const int n_heads  = hp.dec_n_heads;
    const int head_dim = hp.dec_head_dim;

    cohere_debug("\n--- cohere_build_graph_decoder: n_tokens=%d, offset=%d ---\n", n_tokens, offset);

    struct ggml_init_params params = {
        .mem_size   = ctx->compute_meta.size(),
        .mem_buffer = ctx->compute_meta.data(),
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // [d, n_tokens]
    struct ggml_tensor * cur = ggml_add(ctx0,
        ggml_get_rows(ctx0, model.dec_emb_w, embd),
        ggml_get_rows(ctx0, model.dec_pos_w, position)
    );
    cohere_log_tensor("input_embd", cur);

    // emb_ln
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.dec_emb_ln_w), model.dec_emb_ln_b);
    cohere_log_tensor("emb_ln", cur);

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto & layer = model.dec_layers[il];
        struct ggml_tensor * inpL = cur;

        // self-attention norm
        cur = ggml_norm(ctx0, inpL, 1e-5f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.attn_ln_w), layer.attn_ln_b);

        // self-attention Q, K, V
        struct ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        struct ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        struct ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, cur), layer.attn_v_b);

        if (il == 0) {
            cohere_log_tensor("Qcur", Qcur);
            cohere_log_tensor("Kcur", Kcur);
            cohere_log_tensor("Vcur", Vcur);
        }

        // Store current K, V into cache
        {
            struct ggml_tensor * k_view = ggml_view_4d(ctx0, ctx->kv_k,
                head_dim, n_tokens, n_heads, 1,
                ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
                il * ctx->kv_k->nb[3] + offset * ctx->kv_k->nb[1]);
            
            struct ggml_tensor * v_view = ggml_view_4d(ctx0, ctx->kv_v,
                head_dim, n_tokens, n_heads, 1,
                ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
                il * ctx->kv_v->nb[3] + offset * ctx->kv_v->nb[1]);

            struct ggml_tensor * K_perm = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            struct ggml_tensor * V_perm = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));
        }

        struct ggml_tensor * Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3); // [hd, n_tok, n_heads]
        
        struct ggml_tensor * K = ggml_view_3d(ctx0, ctx->kv_k,
            head_dim, offset + n_tokens, n_heads,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2],
            il * ctx->kv_k->nb[3]); // [hd, offset+n_tok, n_heads]
        K = ggml_cont(ctx0, K);

        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q); // [n_tok, offset+n_tok, n_heads]
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf((float)head_dim));
        KQ = ggml_diag_mask_inf(ctx0, KQ, offset);
        KQ = ggml_soft_max(ctx0, KQ);

        struct ggml_tensor * V = ggml_view_3d(ctx0, ctx->kv_v,
            head_dim, offset + n_tokens, n_heads,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2],
            il * ctx->kv_v->nb[3]); // [hd, offset+n_tok, n_heads]

        struct ggml_tensor * V_trans = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // [offset+n_tok, hd, n_heads]
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ); // [hd, n_tok, n_heads]
        
        if (il == 0) {
            cohere_log_tensor("Q", Q);
            cohere_log_tensor("K", K);
            cohere_log_tensor("KQ", KQ);
            cohere_log_tensor("V_trans", V_trans);
            cohere_log_tensor("KQV", KQV);
        }

        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3); // [hd, n_heads, n_tok]
        KQV = ggml_cont(ctx0, KQV);
        cur = ggml_reshape_2d(ctx0, KQV, d, n_tokens);

        // out projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_o_w, cur), layer.attn_o_b);
        cur = ggml_add(ctx0, cur, inpL);
        if (il == 0) ggml_set_name(cur, "h0_after_sa");

        struct ggml_tensor * inpCA = cur;

        // cross-attention
        cur = ggml_norm(ctx0, inpCA, 1e-5f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.cross_ln_w), layer.cross_ln_b);

        struct ggml_tensor * CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_q_w, cur), layer.cross_q_b);
        CQ = ggml_reshape_3d(ctx0, CQ, head_dim, n_heads, n_tokens);
        CQ = ggml_permute(ctx0, CQ, 0, 2, 1, 3); // [hd, n_tok, n_heads]

        // CK and CV inputs
        char ck_name[32], cv_name[32];
        snprintf(ck_name, sizeof(ck_name), "CK_%d", il);
        snprintf(cv_name, sizeof(cv_name), "CV_%d", il);
        
        struct ggml_tensor * CK_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, ctx->cached_T_enc);
        struct ggml_tensor * CV_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, ctx->cached_T_enc);
        ggml_set_name(CK_in, ck_name); ggml_set_input(CK_in);
        ggml_set_name(CV_in, cv_name); ggml_set_input(CV_in);

        struct ggml_tensor * CK = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, CK_in, head_dim, n_heads, ctx->cached_T_enc), 0, 2, 1, 3));

        struct ggml_tensor * CV = ggml_reshape_3d(ctx0, CV_in, head_dim, n_heads, ctx->cached_T_enc);
        CV = ggml_permute(ctx0, CV, 0, 2, 1, 3); // [hd, T_enc, n_heads]
        struct ggml_tensor * CV_trans = ggml_cont(ctx0, ggml_permute(ctx0, CV, 1, 0, 2, 3)); // [T_enc, hd, n_heads]

        struct ggml_tensor * C_KQ = ggml_mul_mat(ctx0, CK, CQ); // [T_enc, n_tokens, n_heads]
        C_KQ = ggml_scale(ctx0, C_KQ, 1.0f / sqrtf((float)head_dim));
        C_KQ = ggml_soft_max(ctx0, C_KQ);

        struct ggml_tensor * C_KQV = ggml_mul_mat(ctx0, CV_trans, C_KQ); // [hd, n_tok, n_heads]
        
        if (il == 0) {
            cohere_log_tensor("CQ", CQ);
            cohere_log_tensor("CK", CK);
            cohere_log_tensor("C_KQ", C_KQ);
            cohere_log_tensor("C_KQV", C_KQV);
        }

        C_KQV = ggml_permute(ctx0, C_KQV, 0, 2, 1, 3); // [hd, n_heads, n_tok]
        C_KQV = ggml_cont(ctx0, C_KQV);
        cur = ggml_reshape_2d(ctx0, C_KQV, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_o_w, cur), layer.cross_o_b);
        cur = ggml_add(ctx0, cur, inpCA);
        if (il == 0) ggml_set_name(cur, "h0_after_ca");

        struct ggml_tensor * inpFFN = cur;

        // FFN
        cur = ggml_norm(ctx0, inpFFN, 1e-5f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, layer.ffn_ln_w), layer.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up_w, cur), layer.ffn_up_b);
        cur = ggml_relu(ctx0, cur);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_dn_w, cur), layer.ffn_dn_b);

        cur = ggml_add(ctx0, cur, inpFFN);

        if (il == 0) {
            cohere_log_tensor("layer_0_out", cur);
        }
    }

    // final norm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.dec_out_ln_w), model.dec_out_ln_b);

    // logits
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.dec_head_w, cur), model.dec_head_b);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ---------------------------------------------------------------------------
// Decoder: one step (auto-regressive)
// tokens:   [offset .. offset+n_tok-1]
// Returns logits: (n_tok, vocab_size)
// ---------------------------------------------------------------------------

static std::vector<float> cohere_decode_step(
    struct cohere_context * ctx,
    int T_enc,
    const int * tokens, int n_tok, int offset,
    const std::vector<std::vector<float>> & cross_kv_k,
    const std::vector<std::vector<float>> & cross_kv_v
) {
    const auto & hp = ctx->model.hparams;
    const int vocab_size = hp.vocab_size;

    ctx->cached_T_enc = T_enc;

    struct ggml_cgraph * gf = cohere_build_graph_decoder(ctx, tokens, n_tok, offset);

    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        fprintf(stderr, "cohere: failed to allocate decoder graph\n");
        return {};
    }

    // Set inputs
    struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, tokens, 0, n_tok * sizeof(int));

    struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");
    std::vector<int> pos_data(n_tok);
    for (int i = 0; i < n_tok; i++) pos_data[i] = offset + i;
    ggml_backend_tensor_set(position, pos_data.data(), 0, n_tok * sizeof(int));

    // Set cross-attention inputs
    for (int il = 0; il < hp.dec_n_layers; il++) {
        char ck_name[32], cv_name[32];
        snprintf(ck_name, sizeof(ck_name), "CK_%d", il);
        snprintf(cv_name, sizeof(cv_name), "CV_%d", il);
        struct ggml_tensor * CK_t = ggml_graph_get_tensor(gf, ck_name);
        struct ggml_tensor * CV_t = ggml_graph_get_tensor(gf, cv_name);
        
        if (il == 0 && offset == 0) {
            cohere_debug("cohere: setting CK_0, size %zu, buffer=%p, first 5: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                   cross_kv_k[il].size(), (void*)(CK_t ? CK_t->buffer : nullptr),
                   cross_kv_k[il][0], cross_kv_k[il][1], cross_kv_k[il][2], cross_kv_k[il][3], cross_kv_k[il][4]);
        }

        if (CK_t) {
            if (!CK_t->buffer) { fprintf(stderr, "error: %s buffer is NULL\n", ck_name); }
            else ggml_backend_tensor_set(CK_t, cross_kv_k[il].data(), 0, cross_kv_k[il].size() * sizeof(float));
        }
        if (CV_t) {
            if (!CV_t->buffer) { fprintf(stderr, "error: %s buffer is NULL\n", cv_name); }
            else ggml_backend_tensor_set(CV_t, cross_kv_v[il].data(), 0, cross_kv_v[il].size() * sizeof(float));
        }
    }

    // Execute
    if (ggml_backend_sched_graph_compute(ctx->ggml_alloc, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cohere: failed to compute decoder graph\n");
        return {};
    }

    // Extract logits (last node in graph)
    struct ggml_tensor * logits_t = ggml_graph_node(gf, -1);
    std::vector<float> logits(n_tok * vocab_size);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));

    if (offset == 0) {
        // Log some interesting hidden states for the first step (prompt)
        struct ggml_tensor * emb_ln = ggml_graph_get_tensor(gf, "emb_ln");
        if (emb_ln) {
            std::vector<float> h(ggml_nelements(emb_ln));
            ggml_backend_tensor_get(emb_ln, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after emb_ln: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                h[0], h[1], h[2], h[3], h[4]);
        }

        struct ggml_tensor * h0_sa = ggml_graph_get_tensor(gf, "h0_after_sa");
        if (h0_sa) {
            std::vector<float> h(ggml_nelements(h0_sa));
            ggml_backend_tensor_get(h0_sa, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after sa_li0: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                h[0], h[1], h[2], h[3], h[4]);
        }

        struct ggml_tensor * h0_ca = ggml_graph_get_tensor(gf, "h0_after_ca");
        if (h0_ca) {
            std::vector<float> h(ggml_nelements(h0_ca));
            ggml_backend_tensor_get(h0_ca, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after ca_li0: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                h[0], h[1], h[2], h[3], h[4]);
        }
        
        // Log top logits for last token of prompt
        const float * last_logits = logits.data() + (n_tok - 1) * vocab_size;
        int top_id = 0;
        for (int i = 1; i < vocab_size; i++) if (last_logits[i] > last_logits[top_id]) top_id = i;
        cohere_debug("cohere: step 0 (prompt) top_tok: %d, logit: %.4f, tok_749_logit: %.4f\n", 
               top_id, last_logits[top_id], last_logits[749]);
    }

    return logits;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

// Pre-populate the ct_to_f32_ref cache for every model weight tensor.
// Called once at init so all inferences pay zero conversion cost.
static void cohere_model_warm_cache(const cohere_model & m) {
    auto w = [](const ggml_tensor * t){ if (t) ct_to_f32_ref(t); };

    // Pre-encode conv subsampling
    w(m.pre_conv0_w); w(m.pre_conv0_b);
    w(m.pre_conv2_w); w(m.pre_conv2_b);
    w(m.pre_conv3_w); w(m.pre_conv3_b);
    w(m.pre_conv5_w); w(m.pre_conv5_b);
    w(m.pre_conv6_w); w(m.pre_conv6_b);
    w(m.pre_out_w);   w(m.pre_out_b);

    // Encoder layers
    for (const auto & l : m.enc_layers) {
        w(l.ff1_norm_w); w(l.ff1_norm_b);
        w(l.ff1_up_w);   w(l.ff1_up_b);
        w(l.ff1_dn_w);   w(l.ff1_dn_b);
        w(l.attn_norm_w); w(l.attn_norm_b);
        w(l.attn_q_w);   w(l.attn_q_b);
        w(l.attn_k_w);   w(l.attn_k_b);
        w(l.attn_v_w);   w(l.attn_v_b);
        w(l.attn_out_w); w(l.attn_out_b);
        w(l.attn_pos_w); w(l.attn_pos_bias_u); w(l.attn_pos_bias_v);
        w(l.conv_norm_w); w(l.conv_norm_b);
        w(l.conv_pw1_w);  w(l.conv_pw1_b);
        w(l.conv_dw_w);   w(l.conv_dw_b);
        w(l.conv_bn_w);   w(l.conv_bn_b);
        w(l.conv_bn_mean); w(l.conv_bn_var);
        w(l.conv_pw2_w);  w(l.conv_pw2_b);
        w(l.ff2_norm_w); w(l.ff2_norm_b);
        w(l.ff2_up_w);   w(l.ff2_up_b);
        w(l.ff2_dn_w);   w(l.ff2_dn_b);
        w(l.out_norm_w); w(l.out_norm_b);
    }

    // Encoder→decoder projection
    w(m.enc_proj_w); w(m.enc_proj_b);

    // Decoder top-level
    w(m.dec_emb_w); w(m.dec_pos_w);
    w(m.dec_emb_ln_w); w(m.dec_emb_ln_b);
    w(m.dec_out_ln_w); w(m.dec_out_ln_b);
    w(m.dec_head_w);   w(m.dec_head_b);

    // Decoder layers
    for (const auto & l : m.dec_layers) {
        w(l.attn_ln_w);  w(l.attn_ln_b);
        w(l.attn_q_w);   w(l.attn_q_b);
        w(l.attn_k_w);   w(l.attn_k_b);
        w(l.attn_v_w);   w(l.attn_v_b);
        w(l.attn_o_w);   w(l.attn_o_b);
        w(l.cross_ln_w); w(l.cross_ln_b);
        w(l.cross_q_w);  w(l.cross_q_b);
        w(l.cross_k_w);  w(l.cross_k_b);
        w(l.cross_v_w);  w(l.cross_v_b);
        w(l.cross_o_w);  w(l.cross_o_b);
        w(l.ffn_ln_w);   w(l.ffn_ln_b);
        w(l.ffn_up_w);   w(l.ffn_up_b);
        w(l.ffn_dn_w);   w(l.ffn_dn_b);
    }
}

struct cohere_context_params cohere_context_default_params(void) {
    return { .n_threads = 4, .use_flash = false };
}

struct cohere_context * cohere_init_from_file(const char * path_model,
                                              struct cohere_context_params params) {
    auto * ctx = new cohere_context;
    ctx->params = params;

    // Initialize ggml backend
    ctx->ggml_backend = ggml_backend_cpu_init();
    if (!ctx->ggml_backend) {
        fprintf(stderr, "cohere: failed to initialize ggml CPU backend\n");
        delete ctx;
        return nullptr;
    }

    if (!cohere_load_model(ctx->model, ctx->vocab, path_model, ctx->ggml_backend)) {
        cohere_free(ctx);
        return nullptr;
    }

    const auto & hp = ctx->model.hparams;

    // Allocate persistent KV cache
    {
        // Shape: [head_dim, max_ctx, n_heads, n_layers]
        struct ggml_init_params kv_params = {
            .mem_size   = ggml_tensor_overhead() * 2 + 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        ctx->kv_ctx = ggml_init(kv_params);
        ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads, hp.dec_n_layers);
        ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads, hp.dec_n_layers);
        
        ggml_backend_buffer_t kv_buf = ggml_backend_alloc_buffer(ctx->ggml_backend, ggml_nbytes(ctx->kv_k) * 2);
        char * base = (char *)ggml_backend_buffer_get_base(kv_buf);
        ggml_backend_tensor_alloc(kv_buf, ctx->kv_k, (void *)(base));
        ggml_backend_tensor_alloc(kv_buf, ctx->kv_v, (void *)(base + ggml_nbytes(ctx->kv_k)));
    }

    // Initialize scheduler
    ggml_backend_t backends[] = { ctx->ggml_backend };
    ctx->ggml_alloc = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);

    // Sized generously for graph nodes
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + 1024);

    // Create constants context
    ctx->ctx_const = ggml_init({ .mem_size = 1024, .mem_buffer = nullptr, .no_alloc = false });
    ctx->eps = ggml_new_tensor_1d(ctx->ctx_const, GGML_TYPE_F32, 1);
    float eps_val = 1e-5f;
    memcpy(ctx->eps->data, &eps_val, sizeof(float));

    return ctx;
}

void cohere_free(struct cohere_context * ctx) {
    if (!ctx) return;
    if (ctx->ggml_alloc)    ggml_backend_sched_free(ctx->ggml_alloc);
    if (ctx->ggml_backend)  ggml_backend_free(ctx->ggml_backend);
    if (ctx->kv_ctx)        ggml_free(ctx->kv_ctx);
    if (ctx->ctx_const)     ggml_free(ctx->ctx_const);
    if (ctx->model.buf)     ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)     ggml_free(ctx->model.ctx);
    delete ctx;
}

int cohere_n_vocab(struct cohere_context * ctx) {
    return ctx->vocab.n_vocab();
}

const char * cohere_token_to_str(struct cohere_context * ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size()) return "<unk>";
    return ctx->vocab.id_to_token[id].c_str();
}

int cohere_str_to_token(struct cohere_context * ctx, const char * s) {
    return ctx->vocab.token_id(s);
}

// ---------------------------------------------------------------------------
// IMPERATIVE ENCODER (from working commit)
// ---------------------------------------------------------------------------

static std::vector<float> cohere_encode_imperative(const cohere_model & m, const float * mel, int T_mel) {
    const auto & hp = m.hparams;
    int ch = hp.pre_conv_ch;
    int d = hp.enc_d_model;

    // Subsampling
    // ... skipping detailed implementation for brevity, 
    // but I need it to be complete to match the reference.
    // Actually, I'll just use the GGML one for now but print its output.
    return {}; 
}

char * cohere_transcribe(struct cohere_context * ctx,
                         const float * samples, int n_samples,
                         const char * lang) {
    const auto & hp  = ctx->model.hparams;
    const auto & voc = ctx->vocab;

    // --- Feature extraction ---
    const auto & mel_fb = ct_to_f32_ref(ctx->model.fe_mel_fb);
    const auto & window = ct_to_f32_ref(ctx->model.fe_window);

    int T_mel = 0;
    auto mel = cohere_compute_features(hp,
        mel_fb.data() + 0, // skip batch dim → shape (n_mels, n_freqs) stored as [1,128,257]
        window.data(),
        samples, n_samples, T_mel);

    // --- Encoder Graph ---
    struct ggml_cgraph * gf_enc = cohere_build_graph_encoder(ctx, T_mel);

    cohere_debug("cohere: allocating encoder graph...\n");
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf_enc)) {
        fprintf(stderr, "cohere: failed to allocate encoder graph\n");
        return nullptr;
    }

    cohere_debug("cohere: setting encoder inputs...\n");
    // Set mel input
    struct ggml_tensor * mel_t = ggml_graph_get_tensor(gf_enc, "mel");
    if (!mel_t) { fprintf(stderr, "error: mel tensor not found\n"); return nullptr; }
    if (!mel_t->buffer) { fprintf(stderr, "error: mel_t buffer is NULL\n"); }
    else ggml_backend_tensor_set(mel_t, mel.data(), 0, mel.size() * sizeof(float));

    // Set pos_enc input
    int H1 = (T_mel + 2 - 3) / 2 + 1;
    int H2 = (H1 + 2 - 3) / 2 + 1;
    int H3 = (H2 + 2 - 3) / 2 + 1;
    int T_enc = H3;
    auto pos_enc_data = ct_rel_pos_enc(T_enc, hp.enc_d_model);
    
    struct ggml_tensor * pos_enc_t = ggml_graph_get_tensor(gf_enc, "pos_enc");
    if (!pos_enc_t) { fprintf(stderr, "error: pos_enc tensor not found\n"); return nullptr; }
    if (!pos_enc_t->buffer) { fprintf(stderr, "error: pos_enc_t buffer is NULL\n"); }
    else ggml_backend_tensor_set(pos_enc_t, pos_enc_data.data(), 0, pos_enc_data.size() * sizeof(float));

    cohere_debug("cohere: computing encoder graph...\n");
    if (ggml_backend_sched_graph_compute(ctx->ggml_alloc, gf_enc) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cohere: failed to compute encoder graph\n");
        return nullptr;
    }

    cohere_debug("cohere: encoder done.\n");
    struct ggml_tensor * enc_out_t = ggml_graph_get_tensor(gf_enc, "enc_out");
    if (!enc_out_t) { fprintf(stderr, "error: enc_out tensor not found\n"); return nullptr; }
    
    if (!enc_out_t->buffer) {
        fprintf(stderr, "error: enc_out_t buffer is NULL! type=%d, name=%s\n", enc_out_t->type, enc_out_t->name);
    }

    std::vector<float> enc_out(ggml_nelements(enc_out_t));
    ggml_backend_tensor_get(enc_out_t, enc_out.data(), 0, enc_out.size() * sizeof(float));

    cohere_debug("cohere: enc_out shape [%d, %d]\n", T_enc, (int)ggml_nelements(enc_out_t) / T_enc);
    cohere_debug("cohere: enc_out[0, :5] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           enc_out[0], enc_out[1], enc_out[2], enc_out[3], enc_out[4]);

    // --- Decoder: build prompt ---
    // Special token IDs from vocab
    auto tid = [&](const std::string & s) { return voc.token_id(s); };
    const char * lang_tok = lang ? lang : "en";
    char lang_tok_str[32];
    snprintf(lang_tok_str, sizeof(lang_tok_str), "<|%s|>", lang_tok);

    std::vector<int> prompt = {
        tid("<|startofcontext|>"),
        tid("<|startoftranscript|>"),
        tid("<|emo:undefined|>"),
        tid(lang_tok_str),
        tid(lang_tok_str),  // second occurrence (language + task)
        tid("<|pnc|>"),
        tid("<|noitn|>"),
        tid("<|notimestamp|>"),
        tid("<|nodiarize|>"),
    };
    // Filter out any missing tokens
    prompt.erase(std::remove_if(prompt.begin(), prompt.end(), [](int t){ return t == -1; }), prompt.end());

    cohere_debug("cohere: prompt IDs: ");
    for (int id : prompt) cohere_debug("%d ", id);
    cohere_debug("\n");

    // Pre-compute cross KV cache once for this utterance
    cohere_precompute_cross_kv(ctx->model, enc_out.data(), T_enc,
                               ctx->cross_kv_k, ctx->cross_kv_v);

    // Reset persistent KV cache
    {
        ggml_backend_buffer_clear(ctx->kv_k->buffer, 0);
        ggml_backend_buffer_clear(ctx->kv_v->buffer, 0);
    }

    const int eos_id  = tid("<|endoftext|>");
    const int max_gen = 100; // debug cap; was: hp.dec_max_ctx - (int)prompt.size() - 4

    // --- Run prompt through decoder ---
    auto logits = cohere_decode_step(ctx, T_enc,
                                      prompt.data(), (int)prompt.size(), 0,
                                      ctx->cross_kv_k, ctx->cross_kv_v);
    int offset = (int)prompt.size();

    // Greedy decode
    std::vector<int> generated;
    for (int step = 0; step < max_gen; step++) {
        // Argmax over last token's logits
        const float * last_logits = logits.data() + ((offset - (step > 0 ? 0 : 0)) - 1) * hp.vocab_size;
        // After prompt pass: last token logits are at offset prompt.size()-1
        int vocab = hp.vocab_size;
        if (step == 0) last_logits = logits.data() + ((int)prompt.size() - 1) * vocab;
        else           last_logits = logits.data();  // n_tok=1

        int next_tok = (int)(std::max_element(last_logits, last_logits + vocab) - last_logits);
        fprintf(stderr, "cohere: step %d → tok %d (%s)\n", step, next_tok,
                next_tok >= 0 && next_tok < (int)voc.id_to_token.size()
                    ? voc.id_to_token[next_tok].c_str() : "?");
        if (next_tok == eos_id || next_tok < 0) break;

        generated.push_back(next_tok);
        offset++;

        // Next step: decode single token
        logits = cohere_decode_step(ctx, T_enc,
                                    &next_tok, 1, offset - 1,
                                    ctx->cross_kv_k, ctx->cross_kv_v);
    }

    // --- Decode tokens to text ---
    std::string text;
    for (int id : generated) {
        if (id < 0 || id >= (int)voc.id_to_token.size()) continue;
        const std::string & tok = voc.id_to_token[id];
        if (tok.front() == '<' && tok.back() == '>') continue; // skip special tokens
        // SentencePiece: ▁ (U+2581) = word boundary
        std::string t = tok;
        size_t pos;
        while ((pos = t.find("\xe2\x96\x81")) != std::string::npos) t.replace(pos, 3, " ");
        text += t;
    }
    // Trim leading space
    if (!text.empty() && text[0] == ' ') text = text.substr(1);

    char * result = (char *)malloc(text.size() + 1);
    memcpy(result, text.c_str(), text.size() + 1);
    return result;
}
