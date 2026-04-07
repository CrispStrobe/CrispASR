// parakeet.cpp — nvidia/parakeet-tdt-0.6b-v3 ggml runtime
//
// First iteration: loader + public C API skeleton.
// Encoder forward (FastConformer), predictor LSTM, joint head, and the
// TDT greedy decode loop will land in subsequent commits.
//
// Architecture summary (see parakeet-todo.md for the full plan):
//   Mel:       128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann window)
//   Encoder:   24× FastConformer block, d_model=1024, 8 heads, head_dim=128,
//              ff_dim=4096, conv kernel=9, 8× temporal subsampling via dw_striding
//   Predictor: embed(8193, 640) + 2-layer LSTM(640, 640)
//   Joint:     enc(1024→640) + pred(640→640) → tanh → linear(640 → 8198)
//              8198 = 8192 vocab + 1 blank + 5 TDT durations {0,1,2,3,4}

#include "parakeet.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif
#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct parakeet_hparams {
    uint32_t sample_rate          = 16000;
    uint32_t n_mels               = 128;
    uint32_t n_fft                = 512;
    uint32_t win_length           = 400;
    uint32_t hop_length           = 160;
    uint32_t d_model              = 1024;
    uint32_t n_layers             = 24;
    uint32_t n_heads              = 8;
    uint32_t head_dim             = 128;
    uint32_t ff_dim               = 4096;
    uint32_t subsampling_factor   = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel          = 9;
    uint32_t pred_hidden          = 640;
    uint32_t pred_layers          = 2;
    uint32_t joint_hidden         = 640;
    uint32_t vocab_size           = 8192;
    uint32_t blank_id             = 8192;
    uint32_t n_tdt_durations      = 5;
    uint32_t frame_dur_cs         = 8;     // 80 ms per encoder frame
    std::vector<int32_t> tdt_durations = {0, 1, 2, 3, 4};
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct parakeet_pre_encode {
    // dw_striding subsampling: Conv2d 0 → Conv2d 2 (dw) + 3 (pw) → Conv2d 5 (dw) + 6 (pw)
    // followed by linear out (4096 → 1024)
    ggml_tensor * conv0_w = nullptr, * conv0_b = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;  // dw
    ggml_tensor * conv3_w = nullptr, * conv3_b = nullptr;  // pw
    ggml_tensor * conv5_w = nullptr, * conv5_b = nullptr;  // dw
    ggml_tensor * conv6_w = nullptr, * conv6_b = nullptr;  // pw
    ggml_tensor * out_w   = nullptr, * out_b   = nullptr;
};

struct parakeet_enc_layer {
    // Pre-FFN1 LN, FFN1 (macaron half)
    ggml_tensor * norm_ff1_w = nullptr, * norm_ff1_b = nullptr;
    ggml_tensor * ff1_l1_w   = nullptr;
    ggml_tensor * ff1_l2_w   = nullptr;

    // Pre-Attn LN, MHA (rel_pos with untied biases)
    ggml_tensor * norm_attn_w = nullptr, * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr;
    ggml_tensor * attn_k_w    = nullptr;
    ggml_tensor * attn_v_w    = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_pos_w  = nullptr;
    ggml_tensor * pos_bias_u  = nullptr;  // (n_heads, head_dim)
    ggml_tensor * pos_bias_v  = nullptr;

    // Pre-Conv LN, depthwise sep conv (Conformer convolution module)
    ggml_tensor * norm_conv_w = nullptr, * norm_conv_b = nullptr;
    ggml_tensor * conv_pw1_w  = nullptr;  // (2 * d_model, d_model, 1) — followed by GLU
    ggml_tensor * conv_dw_w   = nullptr;  // (d_model, 1, K)
    ggml_tensor * conv_dw_b   = nullptr;  // (d_model,) — synthetic, populated by BN fold
    ggml_tensor * conv_bn_w   = nullptr, * conv_bn_b   = nullptr;
    ggml_tensor * conv_bn_rm  = nullptr, * conv_bn_rv  = nullptr;
    ggml_tensor * conv_pw2_w  = nullptr;  // (d_model, d_model, 1)

    // Pre-FFN2 LN, FFN2 (macaron half)
    ggml_tensor * norm_ff2_w = nullptr, * norm_ff2_b = nullptr;
    ggml_tensor * ff2_l1_w   = nullptr;
    ggml_tensor * ff2_l2_w   = nullptr;

    // Final block LN
    ggml_tensor * norm_out_w = nullptr, * norm_out_b = nullptr;
};

struct parakeet_predictor {
    ggml_tensor * embed_w = nullptr;                  // (vocab+1, pred_hidden)
    // 2-layer LSTM (PyTorch convention: w_ih [4H, in], w_hh [4H, H], b_ih [4H], b_hh [4H])
    ggml_tensor * lstm0_w_ih = nullptr, * lstm0_w_hh = nullptr;
    ggml_tensor * lstm0_b_ih = nullptr, * lstm0_b_hh = nullptr;
    ggml_tensor * lstm1_w_ih = nullptr, * lstm1_w_hh = nullptr;
    ggml_tensor * lstm1_b_ih = nullptr, * lstm1_b_hh = nullptr;
};

struct parakeet_joint {
    ggml_tensor * enc_w  = nullptr, * enc_b  = nullptr;   // (joint_hidden, d_model)
    ggml_tensor * pred_w = nullptr, * pred_b = nullptr;   // (joint_hidden, pred_hidden)
    ggml_tensor * out_w  = nullptr, * out_b  = nullptr;   // (vocab+1+n_dur, joint_hidden)
};

// ===========================================================================
// Model and vocabulary
// ===========================================================================

struct parakeet_model {
    parakeet_hparams hparams;

    // Mel preprocessor weights (baked into the .nemo checkpoint)
    ggml_tensor * mel_fb     = nullptr;   // (1, n_mels, n_fft/2+1)
    ggml_tensor * mel_window = nullptr;   // (win_length,)

    parakeet_pre_encode             pre_encode;
    std::vector<parakeet_enc_layer> enc;
    parakeet_predictor              predictor;
    parakeet_joint                  joint;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor *> tensors;
};

struct parakeet_vocab {
    std::vector<std::string>            id_to_token;
    std::unordered_map<std::string,int> token_to_id;
};

struct parakeet_context {
    parakeet_context_params params;

    parakeet_model model;
    parakeet_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    std::vector<uint8_t> compute_meta;     // metadata buffer for graph allocation

    int n_threads = 4;
};

// ---------------------------------------------------------------------------
// Transformer-XL relative-position shift.
// Same trick as cohere.cpp: a single ggml_view_3d that walks the BD matrix
// with stride (nb[1] - nb[0]) along the time axis, dropping the upper
// triangle so each row picks up the rel-pos score r_{j-i}. Zero-cost view.
// ---------------------------------------------------------------------------
static ggml_tensor * parakeet_rel_shift(ggml_context * ctx, ggml_tensor * a) {
    const int T = (int)a->ne[1];
    const int H = (int)a->ne[2];
    return ggml_view_3d(ctx, a,
        T, T, H,
        a->nb[1] - a->nb[0],
        a->nb[2],
        (T - 1) * a->nb[0]);
}

// ===========================================================================
// Loader helpers
// ===========================================================================

static ggml_tensor * try_get(parakeet_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it != m.tensors.end() ? it->second : nullptr;
}

static ggml_tensor * require(parakeet_model & m, const char * name) {
    auto t = try_get(m, name);
    if (!t) {
        fprintf(stderr, "parakeet: required tensor '%s' not found in GGUF\n", name);
    }
    return t;
}

static uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t def = 0) {
    int ki = gguf_find_key(gctx, key);
    return ki >= 0 ? (uint32_t)gguf_get_val_u32(gctx, ki) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool parakeet_load_model(parakeet_model & model,
                                parakeet_vocab  & vocab,
                                const char      * path,
                                ggml_backend_t    backend) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        ggml_init_params meta_params = {
            /*mem_size=*/   4 * 1024 * 1024,
            /*mem_buffer=*/ nullptr,
            /*no_alloc=*/   true,
        };
        ggml_context * meta_ctx = ggml_init(meta_params);
        gguf_init_params load_params_meta = { /*no_alloc=*/true, /*ctx=*/&meta_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params_meta);
        if (!gctx) {
            fprintf(stderr, "parakeet: failed to open '%s'\n", path);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }

        auto & hp = model.hparams;
        hp.sample_rate          = kv_u32(gctx, "parakeet.sample_rate",          hp.sample_rate);
        hp.n_mels               = kv_u32(gctx, "parakeet.n_mels",               hp.n_mels);
        hp.n_fft                = kv_u32(gctx, "parakeet.n_fft",                hp.n_fft);
        hp.win_length           = kv_u32(gctx, "parakeet.win_length",           hp.win_length);
        hp.hop_length           = kv_u32(gctx, "parakeet.hop_length",           hp.hop_length);
        hp.d_model              = kv_u32(gctx, "parakeet.d_model",              hp.d_model);
        hp.n_layers             = kv_u32(gctx, "parakeet.n_layers",             hp.n_layers);
        hp.n_heads              = kv_u32(gctx, "parakeet.n_heads",              hp.n_heads);
        hp.head_dim             = kv_u32(gctx, "parakeet.head_dim",             hp.head_dim);
        hp.ff_dim               = kv_u32(gctx, "parakeet.ff_dim",               hp.ff_dim);
        hp.subsampling_factor   = kv_u32(gctx, "parakeet.subsampling_factor",   hp.subsampling_factor);
        hp.subsampling_channels = kv_u32(gctx, "parakeet.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel          = kv_u32(gctx, "parakeet.conv_kernel",          hp.conv_kernel);
        hp.pred_hidden          = kv_u32(gctx, "parakeet.pred_hidden",          hp.pred_hidden);
        hp.pred_layers          = kv_u32(gctx, "parakeet.pred_layers",          hp.pred_layers);
        hp.joint_hidden         = kv_u32(gctx, "parakeet.joint_hidden",         hp.joint_hidden);
        hp.vocab_size           = kv_u32(gctx, "parakeet.vocab_size",           hp.vocab_size);
        hp.blank_id             = kv_u32(gctx, "parakeet.blank_id",             hp.blank_id);
        hp.n_tdt_durations      = kv_u32(gctx, "parakeet.n_tdt_durations",      hp.n_tdt_durations);
        hp.frame_dur_cs         = kv_u32(gctx, "parakeet.frame_dur_cs",         hp.frame_dur_cs);

        int ki = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (ki >= 0) {
            int n = gguf_get_arr_n(gctx, ki);
            vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                vocab.id_to_token[i] = gguf_get_arr_str(gctx, ki, i);
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        gguf_free(gctx);
        ggml_free(meta_ctx);
    }

    // ---- pass 2: load tensor metadata + bind into a backend buffer ----
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params load_params = { /*no_alloc=*/true, /*ctx=*/&weight_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params);
        if (!gctx || !weight_ctx) {
            fprintf(stderr, "parakeet: failed to load tensor metadata\n");
            return false;
        }

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        // mmap the GGUF, then ggml_backend_tensor_set into each tensor's slot
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "parakeet: open failed\n"); return false; }
        struct stat st; fstat(fd, &st);
        size_t file_size = (size_t)st.st_size;
        void * mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_base == MAP_FAILED) {
            fprintf(stderr, "parakeet: mmap failed\n");
            return false;
        }

        size_t data_offset = gguf_get_data_offset(gctx);

        for (ggml_tensor * t = ggml_get_first_tensor(weight_ctx); t;
             t = ggml_get_next_tensor(weight_ctx, t)) {
            model.tensors[ggml_get_name(t)] = t;

            int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;

            size_t off    = gguf_get_tensor_offset(gctx, tid);
            size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char *)mmap_base + data_offset + off, 0, nbytes);
        }

        munmap(mmap_base, file_size);
        model.ctx = weight_ctx;
        gguf_free(gctx);
    }

    // ---- bind named tensors into the per-layer structs ----

    // Mel preprocessor (optional — may be absent if recomputed at runtime)
    model.mel_fb     = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode (subsampling)
    model.pre_encode.conv0_w = require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv2_w = require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv2_b = require(model, "encoder.pre.conv.2.bias");
    model.pre_encode.conv3_w = require(model, "encoder.pre.conv.3.weight");
    model.pre_encode.conv3_b = require(model, "encoder.pre.conv.3.bias");
    model.pre_encode.conv5_w = require(model, "encoder.pre.conv.5.weight");
    model.pre_encode.conv5_b = require(model, "encoder.pre.conv.5.bias");
    model.pre_encode.conv6_w = require(model, "encoder.pre.conv.6.weight");
    model.pre_encode.conv6_b = require(model, "encoder.pre.conv.6.bias");
    model.pre_encode.out_w   = require(model, "encoder.pre.out.weight");
    model.pre_encode.out_b   = require(model, "encoder.pre.out.bias");

    // Encoder layers
    model.enc.resize(model.hparams.n_layers);
    for (uint32_t i = 0; i < model.hparams.n_layers; i++) {
        char buf[128];
        auto & e = model.enc[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };
        auto try_ = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return try_get(model, buf);
        };

        e.norm_ff1_w  = get("norm_ff1.weight");
        e.norm_ff1_b  = get("norm_ff1.bias");
        e.ff1_l1_w    = get("ff1.linear1.weight");
        e.ff1_l2_w    = get("ff1.linear2.weight");

        e.norm_attn_w = get("norm_attn.weight");
        e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w    = get("attn.q.weight");
        e.attn_k_w    = get("attn.k.weight");
        e.attn_v_w    = get("attn.v.weight");
        e.attn_out_w  = get("attn.out.weight");
        e.attn_pos_w  = get("attn.pos.weight");
        e.pos_bias_u  = get("attn.pos_bias_u");
        e.pos_bias_v  = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight");
        e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w  = get("conv.pw1.weight");
        e.conv_dw_w   = get("conv.dw.weight");
        e.conv_dw_b   = get("conv.dw.bias");          // synthetic — populated by BN fold
        e.conv_pw2_w  = get("conv.pw2.weight");
        e.conv_bn_w   = get("conv.bn.weight");
        e.conv_bn_b   = get("conv.bn.bias");
        e.conv_bn_rm  = get("conv.bn.running_mean");
        e.conv_bn_rv  = get("conv.bn.running_var");

        e.norm_ff2_w  = get("norm_ff2.weight");
        e.norm_ff2_b  = get("norm_ff2.bias");
        e.ff2_l1_w    = get("ff2.linear1.weight");
        e.ff2_l2_w    = get("ff2.linear2.weight");

        e.norm_out_w  = get("norm_out.weight");
        e.norm_out_b  = get("norm_out.bias");
        (void)try_;
    }

    // Predictor
    auto & p = model.predictor;
    p.embed_w    = require(model, "decoder.embed.weight");
    p.lstm0_w_ih = require(model, "decoder.lstm.0.w_ih");
    p.lstm0_w_hh = require(model, "decoder.lstm.0.w_hh");
    p.lstm0_b_ih = require(model, "decoder.lstm.0.b_ih");
    p.lstm0_b_hh = require(model, "decoder.lstm.0.b_hh");
    p.lstm1_w_ih = require(model, "decoder.lstm.1.w_ih");
    p.lstm1_w_hh = require(model, "decoder.lstm.1.w_hh");
    p.lstm1_b_ih = require(model, "decoder.lstm.1.b_ih");
    p.lstm1_b_hh = require(model, "decoder.lstm.1.b_hh");

    // Joint
    auto & j = model.joint;
    j.enc_w  = require(model, "joint.enc.weight");
    j.enc_b  = require(model, "joint.enc.bias");
    j.pred_w = require(model, "joint.pred.weight");
    j.pred_b = require(model, "joint.pred.bias");
    j.out_w  = require(model, "joint.out.weight");
    j.out_b  = require(model, "joint.out.bias");

    fprintf(stderr,
            "parakeet: vocab=%u  d_model=%u  n_layers=%u  n_heads=%u  ff=%u  pred=%u  joint=%u\n",
            model.hparams.vocab_size, model.hparams.d_model, model.hparams.n_layers,
            model.hparams.n_heads, model.hparams.ff_dim, model.hparams.pred_hidden,
            model.hparams.joint_hidden);
    return true;
}

// ===========================================================================
// BatchNorm folding (load-time, once)
//
// Inference-time BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
//                     = x * s + (beta - mean * s)   where s = gamma/sqrt(var+eps)
//
// Since mean / var are fixed after training, we fold s into the depthwise conv
// weights and absorb the bias shift into the synthetic conv_dw_b tensor (which
// the converter pre-allocated as zeros). After this the encoder graph drops
// the BN block entirely.
// ===========================================================================

static void parakeet_fold_batchnorm(parakeet_model & model) {
    const int d   = (int)model.hparams.d_model;
    const int K   = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.n_layers; il++) {
        auto & e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b ||
            !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv) {
            fprintf(stderr, "parakeet: BN fold: missing tensor on layer %u\n", il);
            return;
        }

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w,  bn_w   .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b,  bn_b   .data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        // Fold s into conv_dw_w (F16, ggml shape [K, 1, d]).
        {
            std::vector<float> w_f32((size_t)K * d);
            // dw_w is stored as F16; read+convert via ggml helpers.
            std::vector<ggml_fp16_t> w_f16((size_t)K * d);
            ggml_backend_tensor_get(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < w_f16.size(); i++)
                w_f32[i] = ggml_fp16_to_fp32(w_f16[i]);
            for (int c = 0; c < d; c++)
                for (int ki = 0; ki < K; ki++)
                    w_f32[ki + c * K] *= s[c];
            for (size_t i = 0; i < w_f16.size(); i++)
                w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
            ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        }

        // Fold into the synthetic conv_dw_b: b[c] = (0 - mean[c]) * s[c] + bn_b[c]
        std::vector<float> dw_b(d);
        for (int c = 0; c < d; c++)
            dw_b[c] = -bn_mean[c] * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    fprintf(stderr, "parakeet: BN folded into conv_dw weights for %u layers\n",
            model.hparams.n_layers);
}

// ===========================================================================
// Encoder graph builder
//
// Input:  mel [n_mels, T_mel]
// Output: enc_out [d_model, T_enc]   where T_enc = T_mel / subsampling_factor
// ===========================================================================

static const float kLayerNormEps = 1e-5f;
static const float kBatchNormEps = 1e-5f;

static ggml_cgraph * parakeet_build_graph_encoder(parakeet_context * ctx, int T_mel) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d        = (int)hp.d_model;
    const int n_heads  = (int)hp.n_heads;
    const int head_dim = (int)hp.head_dim;
    const int n_mels   = (int)hp.n_mels;
    const int K        = (int)hp.conv_kernel;

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ----- Input -----
    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ----- Pre-encode (dw_striding subsampling 8×) -----
    // Identical structure to cohere.cpp: Conv2d → ReLU → DwConv → PwConv → ReLU
    // → DwConv → PwConv → ReLU → flatten → linear(d_model).
    auto bias_4d = [&](ggml_context * ctx0, ggml_tensor * b) {
        return ggml_cast(ctx0,
            ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1),
            GGML_TYPE_F32);
    };

    ggml_tensor * cur = ggml_conv_2d(ctx0, m.pre_encode.conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.pre_encode.conv0_b));
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw(ctx0, m.pre_encode.conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.pre_encode.conv2_b));
    cur = ggml_conv_2d   (ctx0, m.pre_encode.conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.pre_encode.conv3_b));
    cur = ggml_relu(ctx0, cur);

    cur = ggml_conv_2d_dw(ctx0, m.pre_encode.conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.pre_encode.conv5_b));
    cur = ggml_conv_2d   (ctx0, m.pre_encode.conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(ctx0, m.pre_encode.conv6_b));
    cur = ggml_relu(ctx0, cur);

    // Flatten freq×channel: [W3, H3, C] → [W3, C, H3] → [W3*C, H3]
    const int H3 = (int)cur->ne[1];
    const int W3 = (int)cur->ne[0];
    const int C  = (int)hp.subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    // out: linear(W3*C → d_model)
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.pre_encode.out_w, cur), m.pre_encode.out_b);

    const int T = H3;  // encoder time frames after 8× subsampling

    // ----- Sinusoidal rel-pos table [d, 2T-1] -----
    ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- 24× Conformer block -----
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto & e = m.enc[il];
        ggml_tensor * inpL = cur;

        // ---- FFN1 (macaron half) ----
        ggml_tensor * x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_ff1_w);
        x = ggml_add(ctx0, x, e.norm_ff1_b);
        x = ggml_mul_mat(ctx0, e.ff1_l1_w, x);
        x = ggml_silu(ctx0, x);
        x = ggml_mul_mat(ctx0, e.ff1_l2_w, x);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

        ggml_tensor * inpAttn = cur;

        // ---- Self-Attention (rel_pos with untied biases) ----
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_attn_w);
        x = ggml_add(ctx0, x, e.norm_attn_b);

        ggml_tensor * Q = ggml_mul_mat(ctx0, e.attn_q_w, x);
        ggml_tensor * K_ = ggml_mul_mat(ctx0, e.attn_k_w, x);
        ggml_tensor * V  = ggml_mul_mat(ctx0, e.attn_v_w, x);
        ggml_tensor * R  = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc);

        ggml_tensor * Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
        ggml_tensor * Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
        K_  = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_,  head_dim, n_heads, T), 0, 2, 1, 3);
        R   = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R,   head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

        ggml_tensor * AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K_), Q_u); // [T, T, H]
        ggml_tensor * BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v); // [2T-1, T, H]
        ggml_tensor * BD = parakeet_rel_shift(ctx0, BD_raw);                // [T, T, H]

        ggml_tensor * scores = ggml_add(ctx0, AC, BD);
        scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max(ctx0, scores);

        ggml_tensor * V3 = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T);
        ggml_tensor * V_t = ggml_permute(ctx0, V3, 1, 2, 0, 3); // [T, hd, H]
        ggml_tensor * attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_t), scores); // [hd, T, H]
        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);

        attn_out = ggml_mul_mat(ctx0, e.attn_out_w, attn_out);
        cur = ggml_add(ctx0, inpAttn, attn_out);

        // ---- Convolution module ----
        ggml_tensor * inpConv = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_conv_w);
        x = ggml_add(ctx0, x, e.norm_conv_b);

        // pw1: [d → 2d], then GLU
        ggml_tensor * pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
        ggml_tensor * cnv = ggml_mul_mat(ctx0, pw1_w, x);
        ggml_tensor * cnv_gate = ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], d * sizeof(float));
        cnv = ggml_mul(ctx0,
            ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], 0),
            ggml_sigmoid(ctx0, cnv_gate));

        // dw conv (kernel 9, padding K/2). Same direct path as cohere.cpp.
        ggml_tensor * dw_w_f32 = ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
        ggml_tensor * dw_w_4d  = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));    // [d, T] → [T, d]
        cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3)); // → [d, T, 1, 1]
        cnv = ggml_reshape_2d(ctx0, cnv, d, T);

        // BN was folded into conv_dw_w/b at load time (parakeet_fold_batchnorm).
        // The fused bias is now in e.conv_dw_b.
        cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, e.conv_dw_b, d, 1));

        cnv = ggml_silu(ctx0, cnv); // swish

        // pw2
        ggml_tensor * pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
        cnv = ggml_mul_mat(ctx0, pw2_w, cnv);
        cur = ggml_add(ctx0, inpConv, cnv);

        // ---- FFN2 (macaron half) ----
        ggml_tensor * inpFF2 = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_ff2_w);
        x = ggml_add(ctx0, x, e.norm_ff2_b);
        x = ggml_mul_mat(ctx0, e.ff2_l1_w, x);
        x = ggml_silu(ctx0, x);
        x = ggml_mul_mat(ctx0, e.ff2_l2_w, x);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

        // ---- Final block LN ----
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, e.norm_out_w);
        cur = ggml_add(ctx0, cur, e.norm_out_b);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Helper: build the sinusoidal rel-pos table [d, 2T-1] expected by the encoder.
// pos[i] = i - (T-1), then standard sinusoidal embedding (Transformer-XL style).
static std::vector<float> parakeet_make_pos_enc(int d_model, int T) {
    const int K = 2 * T - 1;
    std::vector<float> pe((size_t)d_model * K, 0.0f);
    for (int j = 0; j < K; j++) {
        const float p = (float)(T - 1 - j);     // descending: [T-1, T-2, ..., -(T-1)]
        for (int i = 0; i < d_model / 2; i++) {
            const float div = expf(-logf(10000.0f) * (float)(2 * i) / (float)d_model);
            pe[(size_t)(2 * i)     * K + j] = sinf(p * div);
            pe[(size_t)(2 * i + 1) * K + j] = cosf(p * div);
        }
    }
    return pe;
}

// Run the encoder once. Returns enc_out as a flat row-major [T_enc, d_model].
// Caller computes T_enc as ceil(T_mel / subsampling_factor) (approximately —
// the actual value depends on the conv arithmetic and is reported back).
static std::vector<float> parakeet_encode_mel(parakeet_context * ctx,
                                              const float * mel, int n_mels, int T_mel,
                                              int * out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "parakeet: mel feature mismatch (%d vs %d)\n",
                n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(
            ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    ggml_cgraph * gf = parakeet_build_graph_encoder(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parakeet: failed to alloc encoder graph\n");
        return {};
    }

    // Set inputs
    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;  // pos_enc has 2T-1 columns; recover T
    auto pe = parakeet_make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    // Compute
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parakeet: encoder graph compute failed\n");
        return {};
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out) {
        fprintf(stderr, "parakeet: missing enc_out tensor\n");
        return {};
    }
    const int d  = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc) *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t pick_backend() {
#ifdef GGML_USE_METAL
    if (ggml_backend_t b = ggml_backend_metal_init()) return b;
#endif
#ifdef GGML_USE_CUDA
    if (ggml_backend_t b = ggml_backend_cuda_init(0))  return b;
#endif
    return ggml_backend_cpu_init();
}

// ===========================================================================
// Public C API
// ===========================================================================

extern "C" struct parakeet_context_params parakeet_context_default_params(void) {
    parakeet_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    return p;
}

extern "C" struct parakeet_context * parakeet_init_from_file(
    const char * path_model, struct parakeet_context_params params)
{
    auto * ctx = new parakeet_context();
    ctx->params    = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend     = pick_backend();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend) ctx->backend = ctx->backend_cpu;

    if (!parakeet_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "parakeet: failed to load '%s'\n", path_model);
        parakeet_free(ctx);
        return nullptr;
    }

    parakeet_fold_batchnorm(ctx->model);
    return ctx;
}

extern "C" void parakeet_free(struct parakeet_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)             ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)         ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)         ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)       ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// Internal C++ entry point for tests — declared in parakeet.h via a different
// linkage section to avoid polluting the public C API.
extern std::vector<float> parakeet_encode_mel(parakeet_context * ctx,
                                              const float * mel, int n_mels, int T_mel,
                                              int * out_T_enc);

extern "C" int parakeet_test_encoder(struct parakeet_context * ctx, int T_mel) {
    int n_mels = (int)ctx->model.hparams.n_mels;
    std::vector<float> mel((size_t)n_mels * T_mel, 0.0f);
    int T_enc = 0;
    auto out = parakeet_encode_mel(ctx, mel.data(), n_mels, T_mel, &T_enc);
    if (out.empty()) return -1;
    fprintf(stderr, "parakeet: encoder OK — T_mel=%d → T_enc=%d  d=%d  out[0..3]=%g %g %g %g\n",
            T_mel, T_enc, (int)ctx->model.hparams.d_model,
            (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    return T_enc;
}

extern "C" void parakeet_result_free(struct parakeet_result * r) {
    if (!r) return;
    free(r->text);
    free(r->tokens);
    free(r);
}

extern "C" int parakeet_n_vocab    (struct parakeet_context * ctx) { return (int)ctx->model.hparams.vocab_size; }
extern "C" int parakeet_blank_id   (struct parakeet_context * ctx) { return (int)ctx->model.hparams.blank_id; }
extern "C" int parakeet_frame_dur_cs(struct parakeet_context * ctx){ return (int)ctx->model.hparams.frame_dur_cs; }
extern "C" int parakeet_n_mels     (struct parakeet_context * ctx) { return (int)ctx->model.hparams.n_mels; }
extern "C" int parakeet_sample_rate(struct parakeet_context * ctx) { return (int)ctx->model.hparams.sample_rate; }

extern "C" const char * parakeet_token_to_str(struct parakeet_context * ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size()) return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// Forward-pass entry points — stubs for now. Full implementation in next commit.
extern "C" char * parakeet_transcribe(struct parakeet_context * /*ctx*/,
                                      const float * /*samples*/, int /*n_samples*/) {
    fprintf(stderr, "parakeet: transcribe() not yet implemented\n");
    return nullptr;
}

extern "C" struct parakeet_result * parakeet_transcribe_ex(
    struct parakeet_context * /*ctx*/, const float * /*samples*/, int /*n_samples*/,
    int64_t /*t_offset_cs*/) {
    fprintf(stderr, "parakeet: transcribe_ex() not yet implemented\n");
    return nullptr;
}
