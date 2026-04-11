// canary.cpp — nvidia/canary-1b-v2 ggml runtime
//
// First iteration: loader + public C API skeleton.
// Encoder forward (FastConformer with biases), Transformer decoder
// (self-attn + cross-attn + FFN with KV cache), task-token prompt and
// greedy decode loop will land in subsequent commits.
//
// Architecture:
//   Mel:           128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann)
//   Encoder:       32× FastConformer block (use_bias=True), d_model=1024,
//                  8 heads, head_dim=128, ff_dim=4096, conv kernel=9,
//                  8× temporal subsampling via dw_striding
//   Decoder:       8× pre-LN Transformer block (SA + CA + FFN),
//                  d_model=1024, 8 heads, head_dim=128, ff_dim=4096
//   Embedding:     token (16384, 1024) + learned pos_enc (1024, 1024) + LN
//   Output head:   linear (1024 → 16384)
//
// Decoder prompt format (mirrors Cohere — same vocab):
//   <|startoftranscript|> <|src|> <|tgt|> <|pnc|> <|notimestamp|> <|nodiarize|> ...

#include "canary.h"

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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct canary_hparams {
    uint32_t sample_rate          = 16000;
    uint32_t n_mels               = 128;
    uint32_t n_fft                = 512;
    uint32_t win_length           = 400;
    uint32_t hop_length           = 160;
    uint32_t d_model              = 1024;
    uint32_t enc_n_layers         = 32;
    uint32_t dec_n_layers         = 8;
    uint32_t n_heads              = 8;
    uint32_t head_dim             = 128;
    uint32_t ff_dim               = 4096;
    uint32_t subsampling_factor   = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel          = 9;
    uint32_t vocab_size           = 16384;
    uint32_t max_dec_ctx          = 1024;
    uint32_t frame_dur_cs         = 8;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct canary_pre_encode {
    ggml_tensor * conv0_w = nullptr, * conv0_b = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    ggml_tensor * conv3_w = nullptr, * conv3_b = nullptr;
    ggml_tensor * conv5_w = nullptr, * conv5_b = nullptr;
    ggml_tensor * conv6_w = nullptr, * conv6_b = nullptr;
    ggml_tensor * out_w   = nullptr, * out_b   = nullptr;
};

struct canary_enc_layer {
    ggml_tensor * norm_ff1_w = nullptr, * norm_ff1_b = nullptr;
    ggml_tensor * ff1_l1_w   = nullptr, * ff1_l1_b   = nullptr;
    ggml_tensor * ff1_l2_w   = nullptr, * ff1_l2_b   = nullptr;

    ggml_tensor * norm_attn_w = nullptr, * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr, * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr, * attn_k_b    = nullptr;
    ggml_tensor * attn_v_w    = nullptr, * attn_v_b    = nullptr;
    ggml_tensor * attn_out_w  = nullptr, * attn_out_b  = nullptr;
    ggml_tensor * attn_pos_w  = nullptr;     // no bias on the rel-pos projection
    ggml_tensor * pos_bias_u  = nullptr;
    ggml_tensor * pos_bias_v  = nullptr;

    ggml_tensor * norm_conv_w = nullptr, * norm_conv_b = nullptr;
    ggml_tensor * conv_pw1_w  = nullptr, * conv_pw1_b  = nullptr;
    ggml_tensor * conv_dw_w   = nullptr, * conv_dw_b   = nullptr;
    ggml_tensor * conv_bn_w   = nullptr, * conv_bn_b   = nullptr;
    ggml_tensor * conv_bn_rm  = nullptr, * conv_bn_rv  = nullptr;
    ggml_tensor * conv_pw2_w  = nullptr, * conv_pw2_b  = nullptr;

    ggml_tensor * norm_ff2_w = nullptr, * norm_ff2_b = nullptr;
    ggml_tensor * ff2_l1_w   = nullptr, * ff2_l1_b   = nullptr;
    ggml_tensor * ff2_l2_w   = nullptr, * ff2_l2_b   = nullptr;

    ggml_tensor * norm_out_w = nullptr, * norm_out_b = nullptr;
};

struct canary_dec_layer {
    // Pre-LN block layout:
    //   x = x + sa_out @ SA(norm_sa(x))
    //   x = x + ca_out @ CA(norm_ca(x), enc_kv)
    //   x = x + ff_out @ activation(ff_in @ norm_ff(x))
    ggml_tensor * norm_sa_w = nullptr, * norm_sa_b = nullptr;
    ggml_tensor * sa_q_w    = nullptr, * sa_q_b    = nullptr;
    ggml_tensor * sa_k_w    = nullptr, * sa_k_b    = nullptr;
    ggml_tensor * sa_v_w    = nullptr, * sa_v_b    = nullptr;
    ggml_tensor * sa_out_w  = nullptr, * sa_out_b  = nullptr;

    ggml_tensor * norm_ca_w = nullptr, * norm_ca_b = nullptr;
    ggml_tensor * ca_q_w    = nullptr, * ca_q_b    = nullptr;
    ggml_tensor * ca_k_w    = nullptr, * ca_k_b    = nullptr;
    ggml_tensor * ca_v_w    = nullptr, * ca_v_b    = nullptr;
    ggml_tensor * ca_out_w  = nullptr, * ca_out_b  = nullptr;

    ggml_tensor * norm_ff_w = nullptr, * norm_ff_b = nullptr;
    ggml_tensor * ff_in_w   = nullptr, * ff_in_b   = nullptr;
    ggml_tensor * ff_out_w  = nullptr, * ff_out_b  = nullptr;
};

// ===========================================================================
// Model
// ===========================================================================

struct canary_model {
    canary_hparams hparams;

    ggml_tensor * mel_fb     = nullptr;
    ggml_tensor * mel_window = nullptr;

    canary_pre_encode               pre_encode;
    std::vector<canary_enc_layer>   enc;
    std::vector<canary_dec_layer>   dec;

    // Decoder embeddings + final norm + output head
    ggml_tensor * dec_embed_w     = nullptr;   // (vocab, d_model)
    ggml_tensor * dec_pos_enc     = nullptr;   // (max_ctx, d_model) — learned
    ggml_tensor * dec_embed_ln_w  = nullptr;
    ggml_tensor * dec_embed_ln_b  = nullptr;
    ggml_tensor * dec_final_ln_w  = nullptr;
    ggml_tensor * dec_final_ln_b  = nullptr;
    ggml_tensor * dec_head_w      = nullptr;   // (vocab, d_model)
    ggml_tensor * dec_head_b      = nullptr;

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor *> tensors;
};

struct canary_vocab {
    std::vector<std::string>            id_to_token;
    std::unordered_map<std::string,int> token_to_id;
};

struct canary_context {
    canary_context_params params;

    canary_model model;
    canary_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    std::vector<uint8_t> compute_meta;

    // Self-attention KV cache for the decoder
    // shape: [head_dim, max_ctx, n_heads, dec_n_layers]
    ggml_context        * kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor         * kv_k   = nullptr;
    ggml_tensor         * kv_v   = nullptr;

    // Cross-attention K/V — pre-computed once per slice from encoder output.
    // One tensor per decoder layer, shape [head_dim, T_enc, n_heads].
    ggml_context        * cross_ctx = nullptr;
    ggml_backend_buffer_t cross_buf = nullptr;
    std::vector<ggml_tensor *> cross_k;
    std::vector<ggml_tensor *> cross_v;

    // Per-step cross-attention weights from the last decoder layer, captured
    // when collect_attn=true and n_tokens==1. Used for DTW timestamp alignment.
    // step_attn[step_idx] has T_enc * n_heads floats (head h occupies the
    // contiguous slice [h*T_enc, h*T_enc + T_enc)).
    bool collect_attn = false;
    int  attn_T_enc   = 0;
    int  attn_n_heads = 0;
    std::vector<std::vector<float>> step_attn;

    int n_threads = 4;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

static ggml_tensor * try_get(canary_model & m, const char * name) {
    auto it = m.tensors.find(name);
    return it != m.tensors.end() ? it->second : nullptr;
}

static ggml_tensor * require(canary_model & m, const char * name) {
    auto t = try_get(m, name);
    if (!t) fprintf(stderr, "canary: required tensor '%s' not found\n", name);
    return t;
}

static uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t def = 0) {
    int ki = gguf_find_key(gctx, key);
    return ki >= 0 ? (uint32_t)gguf_get_val_u32(gctx, ki) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool canary_load_model(canary_model & model,
                              canary_vocab  & vocab,
                              const char    * path,
                              ggml_backend_t  backend) {
    // ---- pass 1: hparams + vocab ----
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
            fprintf(stderr, "canary: failed to open '%s'\n", path);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }

        auto & hp = model.hparams;
        hp.sample_rate          = kv_u32(gctx, "canary.sample_rate",          hp.sample_rate);
        hp.n_mels               = kv_u32(gctx, "canary.n_mels",               hp.n_mels);
        hp.n_fft                = kv_u32(gctx, "canary.n_fft",                hp.n_fft);
        hp.win_length           = kv_u32(gctx, "canary.win_length",           hp.win_length);
        hp.hop_length           = kv_u32(gctx, "canary.hop_length",           hp.hop_length);
        hp.d_model              = kv_u32(gctx, "canary.d_model",              hp.d_model);
        hp.enc_n_layers         = kv_u32(gctx, "canary.enc_n_layers",         hp.enc_n_layers);
        hp.dec_n_layers         = kv_u32(gctx, "canary.dec_n_layers",         hp.dec_n_layers);
        hp.n_heads              = kv_u32(gctx, "canary.n_heads",              hp.n_heads);
        hp.head_dim             = kv_u32(gctx, "canary.head_dim",             hp.head_dim);
        hp.ff_dim               = kv_u32(gctx, "canary.ff_dim",               hp.ff_dim);
        hp.subsampling_factor   = kv_u32(gctx, "canary.subsampling_factor",   hp.subsampling_factor);
        hp.subsampling_channels = kv_u32(gctx, "canary.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel          = kv_u32(gctx, "canary.conv_kernel",          hp.conv_kernel);
        hp.vocab_size           = kv_u32(gctx, "canary.vocab_size",           hp.vocab_size);
        hp.max_dec_ctx          = kv_u32(gctx, "canary.max_dec_ctx",          hp.max_dec_ctx);
        hp.frame_dur_cs         = kv_u32(gctx, "canary.frame_dur_cs",         hp.frame_dur_cs);

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

    // ---- pass 2: tensor metadata + mmap into backend buffer ----
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params load_params = { /*no_alloc=*/true, /*ctx=*/&weight_ctx };
        gguf_context * gctx = gguf_init_from_file(path, load_params);
        if (!gctx || !weight_ctx) {
            fprintf(stderr, "canary: failed to load tensor metadata\n");
            return false;
        }

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "canary: open failed\n"); return false; }
        struct stat st; fstat(fd, &st);
        size_t file_size = (size_t)st.st_size;
        void * mmap_base = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_base == MAP_FAILED) {
            fprintf(stderr, "canary: mmap failed\n");
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

    // ---- bind named tensors ----

    // Mel preprocessor
    model.mel_fb     = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode
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
    model.enc.resize(model.hparams.enc_n_layers);
    for (uint32_t i = 0; i < model.hparams.enc_n_layers; i++) {
        char buf[128];
        auto & e = model.enc[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight"); e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w   = get("ff1.linear1.weight"); e.ff1_l1_b = get("ff1.linear1.bias");
        e.ff1_l2_w   = get("ff1.linear2.weight"); e.ff1_l2_b = get("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight"); e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w    = get("attn.q.weight");    e.attn_q_b    = get("attn.q.bias");
        e.attn_k_w    = get("attn.k.weight");    e.attn_k_b    = get("attn.k.bias");
        e.attn_v_w    = get("attn.v.weight");    e.attn_v_b    = get("attn.v.bias");
        e.attn_out_w  = get("attn.out.weight");  e.attn_out_b  = get("attn.out.bias");
        e.attn_pos_w  = get("attn.pos.weight");
        e.pos_bias_u  = get("attn.pos_bias_u");
        e.pos_bias_v  = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight"); e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w  = get("conv.pw1.weight");  e.conv_pw1_b  = get("conv.pw1.bias");
        e.conv_dw_w   = get("conv.dw.weight");   e.conv_dw_b   = get("conv.dw.bias");
        e.conv_pw2_w  = get("conv.pw2.weight");  e.conv_pw2_b  = get("conv.pw2.bias");
        e.conv_bn_w   = get("conv.bn.weight");   e.conv_bn_b   = get("conv.bn.bias");
        e.conv_bn_rm  = get("conv.bn.running_mean");
        e.conv_bn_rv  = get("conv.bn.running_var");

        e.norm_ff2_w = get("norm_ff2.weight"); e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w   = get("ff2.linear1.weight"); e.ff2_l1_b = get("ff2.linear1.bias");
        e.ff2_l2_w   = get("ff2.linear2.weight"); e.ff2_l2_b = get("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight"); e.norm_out_b = get("norm_out.bias");
    }

    // Decoder
    model.dec.resize(model.hparams.dec_n_layers);
    for (uint32_t i = 0; i < model.hparams.dec_n_layers; i++) {
        char buf[128];
        auto & d = model.dec[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "decoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };

        d.norm_sa_w = get("norm_sa.weight"); d.norm_sa_b = get("norm_sa.bias");
        d.sa_q_w    = get("sa_q.weight");    d.sa_q_b    = get("sa_q.bias");
        d.sa_k_w    = get("sa_k.weight");    d.sa_k_b    = get("sa_k.bias");
        d.sa_v_w    = get("sa_v.weight");    d.sa_v_b    = get("sa_v.bias");
        d.sa_out_w  = get("sa_out.weight");  d.sa_out_b  = get("sa_out.bias");

        d.norm_ca_w = get("norm_ca.weight"); d.norm_ca_b = get("norm_ca.bias");
        d.ca_q_w    = get("ca_q.weight");    d.ca_q_b    = get("ca_q.bias");
        d.ca_k_w    = get("ca_k.weight");    d.ca_k_b    = get("ca_k.bias");
        d.ca_v_w    = get("ca_v.weight");    d.ca_v_b    = get("ca_v.bias");
        d.ca_out_w  = get("ca_out.weight");  d.ca_out_b  = get("ca_out.bias");

        d.norm_ff_w = get("norm_ff.weight"); d.norm_ff_b = get("norm_ff.bias");
        d.ff_in_w   = get("ff_in.weight");   d.ff_in_b   = get("ff_in.bias");
        d.ff_out_w  = get("ff_out.weight");  d.ff_out_b  = get("ff_out.bias");
    }

    // Decoder embeddings + output head
    model.dec_embed_w    = require(model, "decoder.embed.weight");
    model.dec_pos_enc    = require(model, "decoder.pos_enc");
    model.dec_embed_ln_w = require(model, "decoder.embed_ln.weight");
    model.dec_embed_ln_b = require(model, "decoder.embed_ln.bias");
    model.dec_final_ln_w = require(model, "decoder.final_norm.weight");
    model.dec_final_ln_b = require(model, "decoder.final_norm.bias");
    model.dec_head_w     = require(model, "decoder.head.weight");
    model.dec_head_b     = require(model, "decoder.head.bias");

    fprintf(stderr,
        "canary: vocab=%u  d_model=%u  enc_layers=%u  dec_layers=%u  heads=%u  ff=%u  max_ctx=%u\n",
        model.hparams.vocab_size, model.hparams.d_model,
        model.hparams.enc_n_layers, model.hparams.dec_n_layers,
        model.hparams.n_heads, model.hparams.ff_dim, model.hparams.max_dec_ctx);
    return true;
}

// ===========================================================================
// FFT (iterative Cooley-Tukey, real-input, N must be a power of 2)
// ===========================================================================

static void canary_fft_r2c(const float * in, int N, float * out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1) bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++) rev = (rev << 1) | ((i >> b) & 1);
        out[2*rev]   = in[i];
        out[2*rev+1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float are = out[2*a], aim = out[2*a+1];
                float bre = out[2*b], bim = out[2*b+1];
                float tre = ure*bre - uim*bim, tim = ure*bim + uim*bre;
                out[2*a]   = are + tre; out[2*a+1] = aim + tim;
                out[2*b]   = are - tre; out[2*b+1] = aim - tim;
                float new_ure = ure*wre - uim*wim;
                uim = ure*wim + uim*wre;
                ure = new_ure;
            }
        }
    }
}

// NeMo-style mel: same as parakeet (128 mel, 16 kHz, n_fft=512, win=400, hop=160).
// Delegates to core_mel::compute() with the NeMo cluster's parameters; only
// the FFT function pointer differs between parakeet/canary/canary_ctc/cohere.
#include "core/mel.h"

static std::vector<float> canary_compute_mel(canary_context * ctx,
                                             const float * samples, int n_samples,
                                             int & T_out) {
    const auto & hp = ctx->model.hparams;
    const int n_fft   = (int)hp.n_fft;
    const int hop     = (int)hp.hop_length;
    const int win     = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels  = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "canary: missing preprocessor.fb / window\n");
        return {};
    }

    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft      = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels     = n_mels;
    p.log_base   = core_mel::LogBase::Ln;
    p.norm       = core_mel::Normalization::PerFeatureZ;
    p.layout     = core_mel::Layout::TimeMels;
    p.log_eps    = (float)(1.0 / (1 << 24));
    p.center_pad = true;

    return core_mel::compute(
        samples, n_samples,
        window_raw.data(), win,
        mel_fb.data(), n_freqs,
        canary_fft_r2c,
        p,
        T_out);
}

// ===========================================================================
// rel-pos shift (Transformer-XL trick): single zero-cost ggml_view_3d
// ===========================================================================

static ggml_tensor * canary_rel_shift(ggml_context * ctx, ggml_tensor * a) {
    const int T = (int)a->ne[1];
    const int H = (int)a->ne[2];
    return ggml_view_3d(ctx, a,
        T, T, H,
        a->nb[1] - a->nb[0],
        a->nb[2],
        (T - 1) * a->nb[0]);
}

// Sinusoidal rel-pos table [d, 2T-1], descending positions [T-1 .. -(T-1)].
// IMPORTANT layout note: the tensor is created as ggml_new_tensor_2d(F32, d, 2T-1)
// which has ne[0]=d (fast) and ne[1]=2T-1 (slow). So the correct memory layout
// is `pe[dim + pos*d]`, NOT `pe[(2*i)*K + j]` (which transposes the axes and was
// the bug copied from parakeet/cohere — parakeet's TDT decoder is robust to it,
// but canary's encoder–decoder cross-attention is not).
static std::vector<float> canary_make_pos_enc(int d_model, int T) {
    const int n_pos = 2 * T - 1;
    std::vector<float> pe((size_t)n_pos * d_model, 0.0f);
    for (int p = 0; p < n_pos; p++) {
        const float pos = (float)(T - 1 - p);   // descending: [T-1, T-2, ..., -(T-1)]
        for (int i = 0; i < d_model / 2; i++) {
            const float div = expf(-logf(10000.0f) * (float)(2 * i) / (float)d_model);
            pe[(size_t)p * d_model + 2 * i    ] = sinf(pos * div);
            pe[(size_t)p * d_model + 2 * i + 1] = cosf(pos * div);
        }
    }
    return pe;
}

// ===========================================================================
// Encoder graph
//
// Same structure as parakeet but with biases on every linear/conv. The
// encoder block is the standard pre-LN Conformer (FFN1/2 macaron, MHA
// with rel-pos untied biases, depthwise sep conv with GLU, final block LN).
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph * canary_build_graph_encoder(canary_context * ctx, int T_mel) {
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
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ----- Inputs -----
    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    auto bias_4d = [&](ggml_context * ctx0, ggml_tensor * b) {
        return ggml_cast(ctx0,
            ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1),
            GGML_TYPE_F32);
    };

    // ----- Pre-encode (dw_striding 8×) -----
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

    const int H3 = (int)cur->ne[1];
    const int W3 = (int)cur->ne[0];
    const int C  = (int)hp.subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.pre_encode.out_w, cur), m.pre_encode.out_b);

    const int T = H3;

    // ----- Sinusoidal rel-pos table -----
    ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- 32× Conformer block -----
    for (uint32_t il = 0; il < hp.enc_n_layers; il++) {
        const auto & e = m.enc[il];
        ggml_tensor * inpL = cur;

        // ---- FFN1 (macaron half) ----
        ggml_tensor * x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_ff1_w);
        x = ggml_add(ctx0, x, e.norm_ff1_b);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.ff1_l1_w, x), e.ff1_l1_b);
        x = ggml_silu(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.ff1_l2_w, x), e.ff1_l2_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

        ggml_tensor * inpAttn = cur;

        // ---- Self-Attention (rel_pos with untied biases) ----
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_attn_w);
        x = ggml_add(ctx0, x, e.norm_attn_b);

        ggml_tensor * Q  = ggml_add(ctx0, ggml_mul_mat(ctx0, e.attn_q_w, x), e.attn_q_b);
        ggml_tensor * K_ = ggml_add(ctx0, ggml_mul_mat(ctx0, e.attn_k_w, x), e.attn_k_b);
        ggml_tensor * V  = ggml_add(ctx0, ggml_mul_mat(ctx0, e.attn_v_w, x), e.attn_v_b);
        ggml_tensor * R  = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc);  // no bias on rel-pos proj

        ggml_tensor * Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
        ggml_tensor * Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
        K_  = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_,  head_dim, n_heads, T), 0, 2, 1, 3);
        R   = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R,   head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

        ggml_tensor * AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K_), Q_u);
        ggml_tensor * BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
        ggml_tensor * BD = canary_rel_shift(ctx0, BD_raw);

        ggml_tensor * scores = ggml_add(ctx0, AC, BD);
        scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max(ctx0, scores);

        ggml_tensor * V3 = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T);
        ggml_tensor * V_t = ggml_permute(ctx0, V3, 1, 2, 0, 3);
        ggml_tensor * attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_t), scores);
        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);

        attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, e.attn_out_w, attn_out), e.attn_out_b);
        cur = ggml_add(ctx0, inpAttn, attn_out);

        // ---- Convolution module ----
        ggml_tensor * inpConv = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_conv_w);
        x = ggml_add(ctx0, x, e.norm_conv_b);

        // pw1: [d → 2d] + bias, then GLU
        ggml_tensor * pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
        ggml_tensor * cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw1_w, x), e.conv_pw1_b);
        ggml_tensor * cnv_gate = ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], d * sizeof(float));
        cnv = ggml_mul(ctx0,
            ggml_view_2d(ctx0, cnv, d, T, cnv->nb[1], 0),
            ggml_sigmoid(ctx0, cnv_gate));

        // dw conv (kernel 9, padding K/2)
        ggml_tensor * dw_w_f32 = ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
        ggml_tensor * dw_w_4d  = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
        cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        cnv = ggml_reshape_2d(ctx0, cnv, d, T);

        // BN was folded into conv_dw_w/b at load time → just add the folded bias
        cnv = ggml_add(ctx0, cnv, ggml_reshape_2d(ctx0, e.conv_dw_b, d, 1));

        cnv = ggml_silu(ctx0, cnv);

        // pw2: [d → d] + bias
        ggml_tensor * pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw2_w, cnv), e.conv_pw2_b);
        cur = ggml_add(ctx0, inpConv, cnv);

        // ---- FFN2 (macaron half) ----
        ggml_tensor * inpFF2 = cur;
        x = ggml_norm(ctx0, cur, kLayerNormEps);
        x = ggml_mul(ctx0, x, e.norm_ff2_w);
        x = ggml_add(ctx0, x, e.norm_ff2_b);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.ff2_l1_w, x), e.ff2_l1_b);
        x = ggml_silu(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.ff2_l2_w, x), e.ff2_l2_b);
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

// Run the encoder once. Returns enc_out as a flat row-major [T_enc * d_model].
static std::vector<float> canary_encode_mel(canary_context * ctx,
                                            const float * mel, int n_mels, int T_mel,
                                            int * out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "canary: mel feature mismatch (%d vs %d)\n",
                n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = { ctx->backend, ctx->backend_cpu };
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(
            ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    ggml_cgraph * gf = canary_build_graph_encoder(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: failed to alloc encoder graph\n");
        return {};
    }

    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = canary_make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: encoder compute failed\n");
        return {};
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out) return {};
    const int d  = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc) *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Decoder KV cache + cross-KV allocation
// ===========================================================================

static void canary_alloc_kv(canary_context * ctx) {
    if (ctx->kv_buf) return;
    const auto & hp = ctx->model.hparams;
    const int head_dim = (int)hp.head_dim;
    const int n_heads  = (int)hp.n_heads;
    const int max_ctx  = (int)hp.max_dec_ctx;
    const int n_layers = (int)hp.dec_n_layers;

    ggml_init_params p = {
        /*mem_size=*/   ggml_tensor_overhead() * 4,
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/   true,
    };
    ctx->kv_ctx = ggml_init(p);

    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32,
                                   head_dim, max_ctx, n_heads, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F32,
                                   head_dim, max_ctx, n_heads, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
}

// Build the cross-K/V tensors from encoder output. Called once per slice.
// enc: flat [T_enc * d_model] (row-major, ne[0]=d_model fastest)
static void canary_build_cross_kv(canary_context * ctx,
                                  const float * enc_data, int T_enc) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d        = (int)hp.d_model;
    const int head_dim = (int)hp.head_dim;
    const int n_heads  = (int)hp.n_heads;
    const int n_layers = (int)hp.dec_n_layers;

    // Allocate cross_kv tensors on a CPU buffer (small, ~1 MB per layer for T_enc≈100)
    if (ctx->cross_ctx) {
        ggml_backend_buffer_free(ctx->cross_buf);
        ggml_free(ctx->cross_ctx);
        ctx->cross_ctx = nullptr;
        ctx->cross_buf = nullptr;
        ctx->cross_k.clear();
        ctx->cross_v.clear();
    }

    ggml_init_params p = {
        /*mem_size=*/   ggml_tensor_overhead() * (n_layers * 2 + 8),
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/   true,
    };
    ctx->cross_ctx = ggml_init(p);
    ctx->cross_k.resize(n_layers);
    ctx->cross_v.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        // ggml_flash_attn_ext requires F16 K/V on the CPU backend (the F32 path
        // is not implemented and falls through to broken behaviour). Match the
        // pattern cohere.cpp uses.
        ctx->cross_k[il] = ggml_new_tensor_3d(ctx->cross_ctx, GGML_TYPE_F16, head_dim, T_enc, n_heads);
        ctx->cross_v[il] = ggml_new_tensor_3d(ctx->cross_ctx, GGML_TYPE_F16, head_dim, T_enc, n_heads);
    }
    // Allocate on the SAME backend the decoder graph runs on. Allocating on
    // a separate CPU backend instance breaks the scheduler's cross-graph
    // tensor references — even when both backends happen to be CPU.
    ctx->cross_buf = ggml_backend_alloc_ctx_tensors(ctx->cross_ctx, ctx->backend);

    // Compute K/V projections per layer using a tiny graph
    for (int il = 0; il < n_layers; il++) {
        const auto & dl = m.dec[il];

        ggml_init_params gp = {
            /*mem_size=*/   ctx->compute_meta.size(),
            /*mem_buffer=*/ ctx->compute_meta.data(),
            /*no_alloc=*/   true,
        };
        ggml_context * gctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(gctx, 64, false);

        ggml_tensor * enc = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d, T_enc);
        ggml_set_name(enc, "enc_in");
        ggml_set_input(enc);

        // CK: linear(enc) → bias → reshape [hd, n_heads, T_enc] → permute [hd, T_enc, n_heads]
        ggml_tensor * CK = ggml_add(gctx, ggml_mul_mat(gctx, dl.ca_k_w, enc), dl.ca_k_b);
        CK = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, CK, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        ggml_set_name(CK, "CK");

        ggml_tensor * CV = ggml_add(gctx, ggml_mul_mat(gctx, dl.ca_v_w, enc), dl.ca_v_b);
        CV = ggml_cont(gctx, ggml_permute(gctx,
            ggml_reshape_3d(gctx, CV, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        ggml_set_name(CV, "CV");

        ggml_build_forward_expand(gf, CK);
        ggml_build_forward_expand(gf, CV);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "canary: cross-kv alloc failed\n");
            ggml_free(gctx);
            return;
        }

        ggml_tensor * enc_in = ggml_graph_get_tensor(gf, "enc_in");
        ggml_backend_tensor_set(enc_in, enc_data, 0, (size_t)d * T_enc * sizeof(float));

        ggml_backend_sched_graph_compute(ctx->sched, gf);

        ggml_tensor * CK_out = ggml_graph_get_tensor(gf, "CK");
        ggml_tensor * CV_out = ggml_graph_get_tensor(gf, "CV");
        std::vector<float> kbuf((size_t)head_dim * T_enc * n_heads);
        std::vector<float> vbuf((size_t)head_dim * T_enc * n_heads);
        ggml_backend_tensor_get(CK_out, kbuf.data(), 0, kbuf.size() * sizeof(float));
        ggml_backend_tensor_get(CV_out, vbuf.data(), 0, vbuf.size() * sizeof(float));

        // Convert F32 → F16 before uploading into the F16 cross-KV slots.
        std::vector<ggml_fp16_t> kbuf16(kbuf.size());
        std::vector<ggml_fp16_t> vbuf16(vbuf.size());
        ggml_fp32_to_fp16_row(kbuf.data(), kbuf16.data(), kbuf.size());
        ggml_fp32_to_fp16_row(vbuf.data(), vbuf16.data(), vbuf.size());
        ggml_backend_tensor_set(ctx->cross_k[il], kbuf16.data(), 0, kbuf16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(ctx->cross_v[il], vbuf16.data(), 0, vbuf16.size() * sizeof(ggml_fp16_t));

        ggml_free(gctx);
    }
}

// ===========================================================================
// Decoder per-step graph (autoregressive, with self-attn KV cache + pre-built cross-KV)
// ===========================================================================

static ggml_cgraph * canary_build_graph_decoder(canary_context * ctx,
                                                int n_tokens, int offset)
{
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int d        = (int)hp.d_model;
    const int n_heads  = (int)hp.n_heads;
    const int head_dim = (int)hp.head_dim;

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    ggml_tensor * position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // Token embed + positional embed (learned)
    ggml_tensor * cur = ggml_add(ctx0,
        ggml_get_rows(ctx0, m.dec_embed_w, embd),
        ggml_get_rows(ctx0, m.dec_pos_enc, position));

    // Embedding LayerNorm
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.dec_embed_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_embed_ln_b);

    for (uint32_t il = 0; il < hp.dec_n_layers; il++) {
        const auto & dl = m.dec[il];
        ggml_tensor * inpL = cur;

        // ---- Self-attention (PRE-LN: x = x + SA(LN1(x)), causal, with KV cache) ----
        // Confirmed via NeMo source (transformer_decoders.py forward_preln):
        //   residual = decoder_query
        //   decoder_query = layer_norm_1(decoder_query)
        //   decoder_keys  = layer_norm_1(decoder_keys)
        //   self_attn_output = first_sub_layer(decoder_query, decoder_keys, decoder_keys, mask)
        //   self_attn_output += residual
        cur = ggml_norm(ctx0, inpL, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_sa_w);
        cur = ggml_add(ctx0, cur, dl.norm_sa_b);

        ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_q_w, cur), dl.sa_q_b);
        ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_k_w, cur), dl.sa_k_b);
        ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_v_w, cur), dl.sa_v_b);

        // Store K, V into cache slot at [il, offset .. offset+n_tokens]
        {
            ggml_tensor * k_view = ggml_view_4d(ctx0, ctx->kv_k,
                head_dim, n_tokens, n_heads, 1,
                ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
                il * ctx->kv_k->nb[3] + offset * ctx->kv_k->nb[1]);
            ggml_tensor * v_view = ggml_view_4d(ctx0, ctx->kv_v,
                head_dim, n_tokens, n_heads, 1,
                ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
                il * ctx->kv_v->nb[3] + offset * ctx->kv_v->nb[1]);

            ggml_tensor * K_perm = ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Kcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            ggml_tensor * V_perm = ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, Vcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));
        }

        // Q/K/V attention
        ggml_tensor * Q = ggml_permute(ctx0,
            ggml_reshape_3d(ctx0, Qcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);

        const int L = offset + n_tokens;
        ggml_tensor * K_full = ggml_view_3d(ctx0, ctx->kv_k,
            head_dim, L, n_heads,
            ctx->kv_k->nb[1], ctx->kv_k->nb[2],
            il * ctx->kv_k->nb[3]);
        K_full = ggml_cont(ctx0, K_full);

        ggml_tensor * KQ = ggml_mul_mat(ctx0, K_full, Q);
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf((float)head_dim));
        if (n_tokens > 1) {
            KQ = ggml_diag_mask_inf(ctx0, KQ, offset);
        }
        KQ = ggml_soft_max(ctx0, KQ);

        ggml_tensor * V_full = ggml_view_3d(ctx0, ctx->kv_v,
            head_dim, L, n_heads,
            ctx->kv_v->nb[1], ctx->kv_v->nb[2],
            il * ctx->kv_v->nb[3]);
        ggml_tensor * V_trans = ggml_cont(ctx0, ggml_permute(ctx0, V_full, 1, 0, 2, 3));
        ggml_tensor * sa_out = ggml_mul_mat(ctx0, V_trans, KQ);

        sa_out = ggml_permute(ctx0, sa_out, 0, 2, 1, 3);
        sa_out = ggml_cont(ctx0, sa_out);
        cur    = ggml_reshape_2d(ctx0, sa_out, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.sa_out_w, cur), dl.sa_out_b);
        cur = ggml_add(ctx0, cur, inpL);

        // ---- Cross-attention (PRE-LN: x = x + CA(LN2(x)), no causal mask) ----
        ggml_tensor * inpCA = cur;
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_ca_w);
        cur = ggml_add(ctx0, cur, dl.norm_ca_b);

        ggml_tensor * CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ca_q_w, cur), dl.ca_q_b);
        CQ = ggml_reshape_3d(ctx0, CQ, head_dim, n_heads, n_tokens);
        CQ = ggml_permute(ctx0, CQ, 0, 2, 1, 3);

        ggml_tensor * CK = ctx->cross_k[il];
        ggml_tensor * CV = ctx->cross_v[il];

        ggml_tensor * ca_out = ggml_flash_attn_ext(ctx0, CQ, CK, CV, nullptr,
                                                    1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);

        // Capture cross-attention weights for the last decoder layer in
        // single-token generation mode. Used by the DTW timestamp pass.
        // CK shape: [head_dim, T_enc, n_heads] (F16)
        // CQ shape: [head_dim, n_tokens=1, n_heads]
        // ca_w = softmax( CK^T @ CQ / sqrt(head_dim) ) → [T_enc, 1, n_heads]
        if (ctx->collect_attn && (int)il == (int)hp.dec_n_layers - 1 && n_tokens == 1) {
            ggml_tensor * ca_w = ggml_mul_mat(ctx0, CK, CQ);   // [T_enc, 1, n_heads]
            ca_w = ggml_scale(ctx0, ca_w, 1.0f / sqrtf((float)head_dim));
            ca_w = ggml_soft_max(ctx0, ca_w);
            ggml_set_name(ca_w, "ca_attn_w");
            ggml_build_forward_expand(gf, ca_w);
        }

        cur = ggml_reshape_2d(ctx0, ca_out, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ca_out_w, cur), dl.ca_out_b);
        cur = ggml_add(ctx0, cur, inpCA);

        // ---- FFN (PRE-LN: x = x + FFN(LN3(x)), ReLU activation) ----
        ggml_tensor * inpFF = cur;
        cur = ggml_norm(ctx0, cur, kLayerNormEps);
        cur = ggml_mul(ctx0, cur, dl.norm_ff_w);
        cur = ggml_add(ctx0, cur, dl.norm_ff_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ff_in_w, cur), dl.ff_in_b);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, dl.ff_out_w, cur), dl.ff_out_b);
        cur = ggml_add(ctx0, cur, inpFF);
    }

    // Final layer norm + output head
    cur = ggml_norm(ctx0, cur, kLayerNormEps);
    cur = ggml_mul(ctx0, cur, m.dec_final_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_final_ln_b);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.dec_head_w, cur), m.dec_head_b);
    ggml_set_name(cur, "logits");

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run one decoder step (or batch). Returns logits for the LAST token.
static std::vector<float> canary_decode_step(canary_context * ctx,
                                             const int * tokens, int n_tokens, int offset)
{
    canary_alloc_kv(ctx);
    ggml_cgraph * gf = canary_build_graph_decoder(ctx, n_tokens, offset);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "canary: decoder alloc failed\n");
        return {};
    }

    ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, tokens, 0, n_tokens * sizeof(int));

    std::vector<int> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++) pos[i] = offset + i;
    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "position");
    ggml_backend_tensor_set(pos_in, pos.data(), 0, n_tokens * sizeof(int));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "canary: decoder compute failed\n");
        return {};
    }

    // Capture cross-attention for DTW (only on single-token generation steps).
    if (ctx->collect_attn && n_tokens == 1) {
        ggml_tensor * ca_w_t = ggml_graph_get_tensor(gf, "ca_attn_w");
        if (ca_w_t) {
            const int T_enc   = (int)ca_w_t->ne[0];
            const int n_heads = (int)ca_w_t->ne[2];
            ctx->attn_T_enc   = T_enc;
            ctx->attn_n_heads = n_heads;
            std::vector<float> w((size_t)T_enc * n_heads);
            ggml_backend_tensor_get(ca_w_t, w.data(), 0, w.size() * sizeof(float));
            ctx->step_attn.push_back(std::move(w));
        }
    }

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    const int V = (int)logits->ne[0];
    std::vector<float> all((size_t)V * n_tokens);
    ggml_backend_tensor_get(logits, all.data(), 0, all.size() * sizeof(float));

    // Return logits for the last position only
    return std::vector<float>(all.begin() + (size_t)(n_tokens - 1) * V, all.end());
}

// ===========================================================================
// Greedy decode: build prompt, run encoder, run decoder steps until EOS
// ===========================================================================

static std::vector<int> canary_build_prompt(canary_context * ctx,
                                            const std::string & src,
                                            const std::string & tgt,
                                            bool punctuation)
{
    auto tok = [&](const char * s) { return canary_str_to_token(ctx, s); };
    std::string src_tok = "<|" + src + "|>";
    std::string tgt_tok = "<|" + tgt + "|>";

    // NeMo CanaryBPETokenizer (shared with Cohere) prompt format:
    //   <|startofcontext|>
    //   <|startoftranscript|>
    //   <|emo:undefined|>
    //   <|src_lang|>
    //   <|target_lang|>
    //   <|pnc|>  or  <|nopnc|>
    //   <|notimestamp|>
    //   <|nodiarize|>
    std::vector<int> p;
    p.push_back(tok("<|startofcontext|>"));
    p.push_back(tok("<|startoftranscript|>"));
    p.push_back(tok("<|emo:undefined|>"));
    p.push_back(canary_str_to_token(ctx, src_tok.c_str()));
    p.push_back(canary_str_to_token(ctx, tgt_tok.c_str()));
    p.push_back(tok(punctuation ? "<|pnc|>" : "<|nopnc|>"));
    p.push_back(tok("<|notimestamp|>"));
    p.push_back(tok("<|nodiarize|>"));

    for (int t : p) {
        if (t < 0) {
            fprintf(stderr, "canary: prompt contains an unknown token\n");
            return {};
        }
    }
    return p;
}

static std::string spiece_to_text(const std::string & piece) {
    if (piece.empty()) return "";
    if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>') return "";
    if (piece.size() >= 3 &&
        (unsigned char)piece[0] == 0xE2 &&
        (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81) {
        return std::string(" ") + piece.substr(3);
    }
    return piece;
}

// ===========================================================================
// BatchNorm folding (load-time, once) — same trick as parakeet/cohere
// ===========================================================================

static void canary_fold_batchnorm(canary_model & model) {
    const int d   = (int)model.hparams.d_model;
    const int K   = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto & e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b ||
            !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv) continue;

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d), dw_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w,  bn_w   .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b,  bn_b   .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_dw_b,  dw_b   .data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++) s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        std::vector<ggml_fp16_t> w_f16((size_t)K * d);
        ggml_backend_tensor_get(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        std::vector<float> w_f32((size_t)K * d);
        for (size_t i = 0; i < w_f16.size(); i++) w_f32[i] = ggml_fp16_to_fp32(w_f16[i]);
        for (int c = 0; c < d; c++)
            for (int ki = 0; ki < K; ki++)
                w_f32[ki + c * K] *= s[c];
        for (size_t i = 0; i < w_f16.size(); i++) w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
        ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));

        // Fold into existing dw_b: b'[c] = (dw_b[c] - mean[c]) * s[c] + bn_b[c]
        for (int c = 0; c < d; c++)
            dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    fprintf(stderr, "canary: BN folded into conv_dw weights for %u layers\n",
            model.hparams.enc_n_layers);
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

extern "C" struct canary_context_params canary_context_default_params(void) {
    canary_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    return p;
}

extern "C" struct canary_context * canary_init_from_file(
    const char * path_model, struct canary_context_params params)
{
    auto * ctx = new canary_context();
    ctx->params    = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend     = pick_backend();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend) ctx->backend = ctx->backend_cpu;

    if (!canary_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        canary_free(ctx);
        return nullptr;
    }
    canary_fold_batchnorm(ctx->model);
    return ctx;
}

extern "C" void canary_free(struct canary_context * ctx) {
    if (!ctx) return;
    if (ctx->cross_buf)         ggml_backend_buffer_free(ctx->cross_buf);
    if (ctx->cross_ctx)         ggml_free(ctx->cross_ctx);
    if (ctx->kv_buf)            ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)            ggml_free(ctx->kv_ctx);
    if (ctx->sched)             ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)         ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)         ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)       ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void canary_result_free(struct canary_result * r) {
    if (!r) return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" int canary_n_vocab    (struct canary_context * ctx) { return (int)ctx->model.hparams.vocab_size; }
extern "C" int canary_n_mels     (struct canary_context * ctx) { return (int)ctx->model.hparams.n_mels; }
extern "C" int canary_sample_rate(struct canary_context * ctx) { return (int)ctx->model.hparams.sample_rate; }
extern "C" int canary_frame_dur_cs(struct canary_context * ctx){ return (int)ctx->model.hparams.frame_dur_cs; }

extern "C" const char * canary_token_to_str(struct canary_context * ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size()) return "";
    return ctx->vocab.id_to_token[id].c_str();
}

extern "C" int canary_str_to_token(struct canary_context * ctx, const char * str) {
    auto it = ctx->vocab.token_to_id.find(str);
    return it != ctx->vocab.token_to_id.end() ? it->second : -1;
}

extern "C" int canary_test_load(struct canary_context * ctx) {
    fprintf(stderr,
        "canary: load test OK\n"
        "  vocab_size  = %d\n"
        "  d_model     = %d\n"
        "  enc_layers  = %d\n"
        "  dec_layers  = %d\n"
        "  n_heads     = %d\n"
        "  head_dim    = %d\n"
        "  ff_dim      = %d\n"
        "  max_dec_ctx = %d\n"
        "  n_mels      = %d\n"
        "  sample_rate = %d\n"
        "  frame_dur_cs= %d\n",
        (int)ctx->model.hparams.vocab_size,
        (int)ctx->model.hparams.d_model,
        (int)ctx->model.hparams.enc_n_layers,
        (int)ctx->model.hparams.dec_n_layers,
        (int)ctx->model.hparams.n_heads,
        (int)ctx->model.hparams.head_dim,
        (int)ctx->model.hparams.ff_dim,
        (int)ctx->model.hparams.max_dec_ctx,
        (int)ctx->model.hparams.n_mels,
        (int)ctx->model.hparams.sample_rate,
        (int)ctx->model.hparams.frame_dur_cs);

    // Confirm a few special tokens resolve
    const char * specials[] = {
        "<|startoftranscript|>", "<|endoftext|>", "<|en|>", "<|de|>",
        "<|fr|>", "<|es|>", "<|nopnc|>", "<|notimestamp|>", "<|nodiarize|>",
    };
    for (const char * s : specials) {
        int id = canary_str_to_token(ctx, s);
        fprintf(stderr, "  token %-22s = %d\n", s, id);
    }
    return 0;
}

extern "C" int canary_test_encoder(struct canary_context * ctx, int T_mel) {
    int n_mels = (int)ctx->model.hparams.n_mels;
    std::vector<float> mel((size_t)n_mels * T_mel, 0.0f);
    int T_enc = 0;
    auto out = canary_encode_mel(ctx, mel.data(), n_mels, T_mel, &T_enc);
    if (out.empty()) return -1;
    fprintf(stderr,
        "canary: encoder OK — T_mel=%d → T_enc=%d  d=%d  out[0..3]=%g %g %g %g\n",
        T_mel, T_enc, (int)ctx->model.hparams.d_model,
        (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    return T_enc;
}

extern "C" struct canary_result * canary_transcribe_ex(
    struct canary_context * ctx, const float * samples, int n_samples,
    const char * source_lang, const char * target_lang, bool punctuation,
    int64_t t_offset_cs)
{
    if (!ctx || !samples || n_samples <= 0 || !source_lang || !target_lang) return nullptr;

    // 1. Mel
    int T_mel = 0;
    auto mel = canary_compute_mel(ctx, samples, n_samples, T_mel);
    if (mel.empty()) return nullptr;

    // 2. Encoder
    int T_enc = 0;
    auto enc = canary_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    if (enc.empty()) return nullptr;

    // 3. Pre-compute cross-attention K/V
    canary_build_cross_kv(ctx, enc.data(), T_enc);

    // 4. Build prompt
    std::vector<int> prompt = canary_build_prompt(ctx, source_lang, target_lang, punctuation);
    if (prompt.empty()) return nullptr;

    // Reset DTW state and enable cross-attn capture for the upcoming greedy loop.
    // Each per-step decode call will append one entry to ctx->step_attn (one
    // T_enc * n_heads vector per emitted token).
    ctx->collect_attn = true;
    ctx->step_attn.clear();

    // 5. Greedy decode
    const int eos = canary_str_to_token(ctx, "<|endoftext|>");
    const int max_steps = 256;
    const int max_ctx   = (int)ctx->model.hparams.max_dec_ctx;

    std::vector<int> generated = prompt;
    int offset = 0;

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "canary: prompt =");
        for (int t : prompt) fprintf(stderr, " %d(%s)", t, ctx->vocab.id_to_token[t].c_str());
        fprintf(stderr, "\n");
    }


    // First call: feed the entire prompt at once
    auto logits = canary_decode_step(ctx, prompt.data(), (int)prompt.size(), 0);
    if (logits.empty()) return nullptr;
    offset = (int)prompt.size();

    for (int step = 0; step < max_steps && offset < max_ctx - 1; step++) {
        // Argmax
        int best = 0;
        float best_lp = logits[0];
        for (int v = 1; v < (int)logits.size(); v++) {
            if (logits[v] > best_lp) { best_lp = logits[v]; best = v; }
        }
        if (ctx->params.verbosity >= 2 && step < 30) {
            const char * pc = (best >= 0 && best < (int)ctx->vocab.id_to_token.size())
                ? ctx->vocab.id_to_token[best].c_str() : "?";
            fprintf(stderr, "  step %3d  tok=%5d  '%s'  logp=%.3f\n", step, best, pc, (double)best_lp);
        }
        if (best == eos) break;
        generated.push_back(best);

        // Decode next step with the new token
        int tok = best;
        logits = canary_decode_step(ctx, &tok, 1, offset);
        if (logits.empty()) break;
        offset++;
    }

    // 6. Build result (skip the prompt prefix)
    auto * r = (canary_result *)calloc(1, sizeof(canary_result));
    int n_emitted = (int)generated.size() - (int)prompt.size();
    r->n_tokens = n_emitted;
    r->tokens = (canary_token_data *)calloc(n_emitted > 0 ? n_emitted : 1, sizeof(canary_token_data));

    // ----- Per-token timestamps via cross-attention DTW -----
    //
    // For each emitted token we have ctx->step_attn[i] = cross-attn weights from
    // the LAST decoder layer, layout [T_enc * n_heads]. We:
    //   1. Average across heads → A[n_emitted × T_enc]
    //   2. Subtract per-frame mean (column normalisation) so globally "hot"
    //      frames don't dominate
    //   3. Run prefix-max DTW: D[i][j] = A[i][j] + max_{k≤j} D[i-1][k]
    //   4. Traceback from argmax of last row → path[i] = best frame for token i
    //   5. t0/t1 from path[i] × frame_dur_cs
    //
    // Same approach as cohere-main, gives ~360 ms MAE on word boundaries.
    // (Approach copied from src/cohere.cpp; see that file for the discussion of
    // hot-frame collapse and the column-normalisation fix.)
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    const int64_t total_cs = (int64_t)T_enc * frame_dur_cs;
    const int64_t seg_end_cs = t_offset_cs + total_cs;

    const bool have_attn =
        n_emitted > 0 &&
        ctx->step_attn.size() == (size_t)n_emitted &&
        ctx->attn_T_enc > 0 &&
        ctx->attn_n_heads > 0;

    std::vector<int> path;
    if (have_attn) {
        const int T_e   = ctx->attn_T_enc;
        const int n_h   = ctx->attn_n_heads;

        // Step 1 + temporal offset correction (same trick as cohere): use
        // step_attn[i-1] for token i because the attention is collected while
        // the decoder processes generated[i] as input to predict generated[i+1].
        std::vector<float> A((size_t)n_emitted * T_e, 0.0f);
        for (int i = 0; i < n_emitted; i++) {
            int src = (i > 0) ? i - 1 : 0;
            const float * w = ctx->step_attn[src].data();
            float * Ai = A.data() + (size_t)i * T_e;
            for (int h = 0; h < n_h; h++)
                for (int t = 0; t < T_e; t++)
                    Ai[t] += w[h * T_e + t];
            for (int t = 0; t < T_e; t++) Ai[t] /= n_h;
        }

        // Step 2: column normalisation
        {
            std::vector<float> col_mean(T_e, 0.f);
            for (int i = 0; i < n_emitted; i++)
                for (int t = 0; t < T_e; t++)
                    col_mean[t] += A[(size_t)i * T_e + t];
            for (int t = 0; t < T_e; t++) col_mean[t] /= n_emitted;
            for (int i = 0; i < n_emitted; i++)
                for (int t = 0; t < T_e; t++)
                    A[(size_t)i * T_e + t] -= col_mean[t];
        }

        // Step 3: forward DP with prefix-max predecessors
        std::vector<float> D((size_t)n_emitted * T_e);
        std::vector<int>   P((size_t)n_emitted * T_e);
        for (int j = 0; j < T_e; j++) D[j] = A[j];
        for (int i = 1; i < n_emitted; i++) {
            const float * Di_1 = D.data() + (size_t)(i-1) * T_e;
            const float * Ai   = A.data() + (size_t)i     * T_e;
            float       * Di   = D.data() + (size_t)i     * T_e;
            int         * Pi   = P.data() + (size_t)i     * T_e;
            float pm_val = Di_1[0]; int pm_idx = 0;
            for (int j = 0; j < T_e; j++) {
                if (Di_1[j] > pm_val) { pm_val = Di_1[j]; pm_idx = j; }
                Di[j] = Ai[j] + pm_val;
                Pi[j] = pm_idx;
            }
        }

        // Step 4: traceback
        path.resize(n_emitted);
        const float * Dlast = D.data() + (size_t)(n_emitted-1) * T_e;
        path[n_emitted-1] = (int)(std::max_element(Dlast, Dlast + T_e) - Dlast);
        for (int i = n_emitted-2; i >= 0; i--)
            path[i] = P[(size_t)(i+1) * T_e + path[i+1]];
    }

    std::string text;
    for (int i = 0; i < n_emitted; i++) {
        int tid = generated[(int)prompt.size() + i];
        const std::string & piece = (tid >= 0 && tid < (int)ctx->vocab.id_to_token.size())
            ? ctx->vocab.id_to_token[tid] : std::string("");
        std::string vis = spiece_to_text(piece);

        canary_token_data & td = r->tokens[i];
        td.id  = tid;
        if (have_attn) {
            int64_t a = t_offset_cs + (int64_t)path[i] * frame_dur_cs;
            int64_t b = (i+1 < n_emitted)
                ? (t_offset_cs + (int64_t)path[i+1] * frame_dur_cs)
                : seg_end_cs;
            // Guarantee at least one frame of duration so adjacent tokens that
            // collapse to the same DTW frame still have a non-zero span.
            if (b <= a) b = a + frame_dur_cs;
            if (b > seg_end_cs) b = seg_end_cs;
            td.t0 = a;
            td.t1 = b;
        } else {
            // Fallback: linear interpolation
            td.t0 = t_offset_cs + (int64_t)((double)i       / std::max(1, n_emitted) * total_cs);
            td.t1 = t_offset_cs + (int64_t)((double)(i + 1) / std::max(1, n_emitted) * total_cs);
        }
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    if (!text.empty() && text[0] == ' ') text = text.substr(1);
    r->text = strdup(text.c_str());

    // Disable attn collection so the next call starts fresh
    ctx->collect_attn = false;

    // Word grouping (same as parakeet's)
    {
        std::vector<canary_word_data> words;
        canary_word_data cur = {};
        bool have_cur = false;
        auto is_punct = [](const char * s) {
            if (!s || !*s) return false;
            for (const char * p = s; *p; p++) {
                unsigned char c = (unsigned char)*p;
                if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' ||
                      c == '\'' || c == '"' || c == '(' || c == ')' || c == '-')) return false;
            }
            return true;
        };
        for (int i = 0; i < r->n_tokens; i++) {
            const auto & td = r->tokens[i];
            if (!td.text[0]) continue;
            const bool is_word_start = (td.text[0] == ' ');
            const bool is_p = is_punct(td.text);
            if (is_word_start && !is_p && have_cur) { words.push_back(cur); cur = {}; have_cur = false; }
            if (!have_cur) { cur.t0 = td.t0; have_cur = true; }
            cur.t1 = td.t1;
            const char * src = td.text + (is_word_start ? 1 : 0);
            size_t cl = strlen(cur.text);
            size_t cap = sizeof(cur.text) - cl - 1;
            size_t add = strlen(src);
            if (add > cap) add = cap;
            memcpy(cur.text + cl, src, add);
            cur.text[cl + add] = '\0';
        }
        if (have_cur) words.push_back(cur);
        r->n_words = (int)words.size();
        r->words = (canary_word_data *)calloc(r->n_words > 0 ? r->n_words : 1, sizeof(canary_word_data));
        for (int i = 0; i < r->n_words; i++) r->words[i] = words[i];
    }

    return r;
}

extern "C" char * canary_transcribe(struct canary_context * ctx,
                                    const float * samples, int n_samples,
                                    const char * source_lang,
                                    const char * target_lang,
                                    bool          punctuation) {
    canary_result * r = canary_transcribe_ex(ctx, samples, n_samples,
                                             source_lang, target_lang, punctuation, 0);
    if (!r) return nullptr;
    char * out = strdup(r->text ? r->text : "");
    canary_result_free(r);
    return out;
}
