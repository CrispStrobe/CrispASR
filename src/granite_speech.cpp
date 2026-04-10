// granite_speech.cpp — ibm-granite/granite-4.0-1b-speech ggml runtime
//
// Three-module speech-LLM:
//   1. 16-layer Conformer encoder (Macaron FFN, depthwise conv, rel pos emb)
//   2. 2-layer BLIP-2 Q-Former projector (3 learned query tokens → 3 LLM tokens)
//   3. 40-layer Granite 1B LLM (GQA 16/4, μP multipliers, RoPE)

#include "granite_speech.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

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
#include <unistd.h>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct granite_speech_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;

    // Encoder
    uint32_t enc_n_layers = 16;
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_heads = 8;
    uint32_t enc_head_dim = 128;
    uint32_t enc_input_dim = 160;
    uint32_t enc_conv_kernel = 15;
    uint32_t enc_ff_dim = 4096;

    // Projector (Q-Former)
    uint32_t proj_n_layers = 2;
    uint32_t proj_d_model = 1024;
    uint32_t proj_n_heads = 16;
    uint32_t proj_ff_dim = 4096;

    // LLM
    uint32_t llm_n_layers = 40;
    uint32_t llm_d_model = 2048;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 4;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 4096;
    float    llm_rope_theta = 10000.0f;
    float    llm_rms_eps = 1e-5f;
    uint32_t llm_vocab_size = 100353;

    // μP multipliers
    float embedding_multiplier = 12.0f;
    float attention_multiplier = 0.0078125f;  // 1/128
    float residual_multiplier = 0.22f;
    float logits_scaling = 8.0f;

    uint32_t downsample_rate = 5;
    uint32_t window_size = 15;
    uint32_t audio_token_index = 100352;
};

// ===========================================================================
// Model tensors
// ===========================================================================

struct granite_enc_block {
    // Attention
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_kv_w = nullptr;  // combined K+V: (2*head_dim*n_heads, d_model)
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_out_b = nullptr;
    ggml_tensor * attn_rel_pos_w = nullptr;  // (max_pos*2+1, head_dim)

    // Conv module
    ggml_tensor * conv_up_w = nullptr;
    ggml_tensor * conv_up_b = nullptr;
    ggml_tensor * conv_dw_w = nullptr;  // depthwise: (2*d_model, 1, kernel)
    ggml_tensor * conv_bn_w = nullptr;
    ggml_tensor * conv_bn_b = nullptr;
    ggml_tensor * conv_bn_mean = nullptr;
    ggml_tensor * conv_bn_var = nullptr;
    ggml_tensor * conv_down_w = nullptr;
    ggml_tensor * conv_down_b = nullptr;
    ggml_tensor * conv_norm_w = nullptr;
    ggml_tensor * conv_norm_b = nullptr;

    // FFN1 (Macaron pre)
    ggml_tensor * ff1_norm_w = nullptr;
    ggml_tensor * ff1_norm_b = nullptr;
    ggml_tensor * ff1_up_w = nullptr;
    ggml_tensor * ff1_up_b = nullptr;
    ggml_tensor * ff1_down_w = nullptr;
    ggml_tensor * ff1_down_b = nullptr;

    // FFN2 (Macaron post)
    ggml_tensor * ff2_norm_w = nullptr;
    ggml_tensor * ff2_norm_b = nullptr;
    ggml_tensor * ff2_up_w = nullptr;
    ggml_tensor * ff2_up_b = nullptr;
    ggml_tensor * ff2_down_w = nullptr;
    ggml_tensor * ff2_down_b = nullptr;

    // Post-norm
    ggml_tensor * post_norm_w = nullptr;
    ggml_tensor * post_norm_b = nullptr;
};

struct granite_proj_block {
    // Self-attention
    ggml_tensor * sa_q_w = nullptr, * sa_q_b = nullptr;
    ggml_tensor * sa_k_w = nullptr, * sa_k_b = nullptr;
    ggml_tensor * sa_v_w = nullptr, * sa_v_b = nullptr;
    ggml_tensor * sa_out_w = nullptr, * sa_out_b = nullptr;
    ggml_tensor * sa_norm_w = nullptr, * sa_norm_b = nullptr;

    // Cross-attention
    ggml_tensor * ca_q_w = nullptr, * ca_q_b = nullptr;
    ggml_tensor * ca_k_w = nullptr, * ca_k_b = nullptr;
    ggml_tensor * ca_v_w = nullptr, * ca_v_b = nullptr;
    ggml_tensor * ca_out_w = nullptr, * ca_out_b = nullptr;
    ggml_tensor * ca_norm_w = nullptr, * ca_norm_b = nullptr;

    // FFN
    ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;
    ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
    ggml_tensor * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
};

struct granite_llm_block {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_gate_w = nullptr;
    ggml_tensor * ffn_up_w = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
};

struct granite_speech_model {
    granite_speech_hparams hparams;

    struct {
        ggml_tensor * input_w = nullptr;
        ggml_tensor * input_b = nullptr;
        ggml_tensor * mel_filters = nullptr;
        ggml_tensor * mel_window = nullptr;
        // Mid-CTC residual (applied after layer 8)
        ggml_tensor * ctc_out_w = nullptr;   // (1024, 348)
        ggml_tensor * ctc_out_b = nullptr;   // (348,)
        ggml_tensor * ctc_mid_w = nullptr;   // (348, 1024)
        ggml_tensor * ctc_mid_b = nullptr;   // (1024,)
        std::vector<granite_enc_block> blocks;
    } encoder;

    struct {
        ggml_tensor * query = nullptr;       // (1, n_query, d_model)
        ggml_tensor * ln_w = nullptr;
        ggml_tensor * ln_b = nullptr;
        ggml_tensor * linear_w = nullptr;    // (llm_d, proj_d)
        ggml_tensor * linear_b = nullptr;
        std::vector<granite_proj_block> blocks;
    } projector;

    struct {
        ggml_tensor * token_embd_w = nullptr;
        ggml_tensor * output_norm_w = nullptr;
        ggml_tensor * output_w = nullptr;   // separate lm_head (not tied)
        std::vector<granite_llm_block> blocks;
    } llm;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

struct granite_speech_context {
    granite_speech_context_params params;
    granite_speech_model model;

    ggml_backend_t       backend = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ggml_context *        kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor *         kv_k = nullptr;
    ggml_tensor *         kv_v = nullptr;

    int n_threads = 4;

    // Precomputed relative position embedding lookup (200, 200, 128) F32
    // rpe_lookup[c][r][d] = rel_pos_emb(attention_dists[c][r])[d]
    std::vector<float> rpe_lookup;
};

// ===========================================================================
// GGUF loader helpers
// ===========================================================================

static uint32_t kv_u32(gguf_context * g, const char * key, uint32_t def) {
    int i = gguf_find_key(g, key);
    return i >= 0 ? (uint32_t)gguf_get_val_u32(g, i) : def;
}
static float kv_f32(gguf_context * g, const char * key, float def) {
    int i = gguf_find_key(g, key);
    return i >= 0 ? gguf_get_val_f32(g, i) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool granite_speech_load_model(granite_speech_model & model, const char * path,
                                      ggml_backend_t backend) {
    // Pass 1: metadata
    {
        gguf_init_params mp = { true, nullptr };
        gguf_context * g = gguf_init_from_file(path, mp);
        if (!g) return false;
        auto & hp = model.hparams;

        hp.enc_n_layers = kv_u32(g, "granite_speech.enc.n_layers", hp.enc_n_layers);
        hp.enc_d_model  = kv_u32(g, "granite_speech.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads  = kv_u32(g, "granite_speech.enc.n_heads", hp.enc_n_heads);
        hp.enc_head_dim = kv_u32(g, "granite_speech.enc.head_dim", hp.enc_head_dim);
        hp.enc_input_dim = kv_u32(g, "granite_speech.enc.input_dim", hp.enc_input_dim);
        hp.enc_conv_kernel = kv_u32(g, "granite_speech.enc.conv_kernel", hp.enc_conv_kernel);
        hp.enc_ff_dim   = kv_u32(g, "granite_speech.enc.ff_dim", hp.enc_ff_dim);

        hp.proj_n_layers = kv_u32(g, "granite_speech.proj.n_layers", hp.proj_n_layers);
        hp.proj_d_model  = kv_u32(g, "granite_speech.proj.d_model", hp.proj_d_model);
        hp.proj_n_heads  = kv_u32(g, "granite_speech.proj.n_heads", hp.proj_n_heads);
        hp.proj_ff_dim   = kv_u32(g, "granite_speech.proj.ff_dim", hp.proj_ff_dim);

        hp.llm_n_layers   = kv_u32(g, "granite_speech.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model    = kv_u32(g, "granite_speech.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads    = kv_u32(g, "granite_speech.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = kv_u32(g, "granite_speech.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim   = kv_u32(g, "granite_speech.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim     = kv_u32(g, "granite_speech.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = kv_f32(g, "granite_speech.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps    = kv_f32(g, "granite_speech.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = kv_u32(g, "granite_speech.llm.vocab_size", hp.llm_vocab_size);

        hp.embedding_multiplier = kv_f32(g, "granite_speech.llm.embedding_multiplier", hp.embedding_multiplier);
        hp.attention_multiplier = kv_f32(g, "granite_speech.llm.attention_multiplier", hp.attention_multiplier);
        hp.residual_multiplier  = kv_f32(g, "granite_speech.llm.residual_multiplier", hp.residual_multiplier);
        hp.logits_scaling       = kv_f32(g, "granite_speech.llm.logits_scaling", hp.logits_scaling);

        hp.downsample_rate    = kv_u32(g, "granite_speech.downsample_rate", hp.downsample_rate);
        hp.window_size        = kv_u32(g, "granite_speech.window_size", hp.window_size);
        hp.audio_token_index  = kv_u32(g, "granite_speech.audio_token_index", hp.audio_token_index);

        gguf_free(g);
    }

    // Pass 2: load tensors
    ggml_context * weight_ctx = nullptr;
    {
        gguf_init_params lp = { true, &weight_ctx };
        gguf_context * g = gguf_init_from_file(path, lp);
        if (!g || !weight_ctx) return false;

        model.buf = ggml_backend_alloc_ctx_tensors(weight_ctx, backend);

        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st; fstat(fd, &st);
        void * mmap_base = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);

        int n_tensors = gguf_get_n_tensors(g);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(weight_ctx, name);
            if (!t) continue;
            size_t offset = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
            ggml_backend_tensor_set(t, (const char *)mmap_base + offset, 0, ggml_nbytes(t));
            model.tensors[name] = t;
        }
        munmap(mmap_base, (size_t)st.st_size);
        gguf_free(g);
    }
    model.ctx = weight_ctx;

    // Bind tensors
    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = model.tensors.find(n);
        return it != model.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string & n) -> ggml_tensor * {
        auto * t = get(n);
        if (!t) fprintf(stderr, "granite_speech: missing '%s'\n", n.c_str());
        return t;
    };

    // Encoder
    auto & e = model.encoder;
    e.input_w = require("enc.input.weight");
    e.input_b = require("enc.input.bias");
    e.mel_filters = get("audio.mel_filters");
    e.mel_window = get("audio.mel_window");
    e.ctc_out_w = get("enc.ctc_out.weight");
    e.ctc_out_b = get("enc.ctc_out.bias");
    e.ctc_mid_w = get("enc.ctc_mid.weight");
    e.ctc_mid_b = get("enc.ctc_mid.bias");

    e.blocks.resize(model.hparams.enc_n_layers);
    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto p = "enc.blk." + std::to_string(il) + ".";
        auto & b = e.blocks[il];
        b.attn_norm_w = get(p + "attn_norm.weight");
        b.attn_norm_b = get(p + "attn_norm.bias");
        b.attn_q_w = require(p + "attn_q.weight");
        b.attn_kv_w = require(p + "attn_kv.weight");
        b.attn_out_w = require(p + "attn_out.weight");
        b.attn_out_b = get(p + "attn_out.bias");
        b.attn_rel_pos_w = get(p + "attn_rel_pos.weight");

        b.conv_up_w = get(p + "conv_up.weight");
        b.conv_up_b = get(p + "conv_up.bias");
        b.conv_dw_w = get(p + "conv_dw.weight");
        b.conv_bn_w = get(p + "conv_bn.weight");
        b.conv_bn_b = get(p + "conv_bn.bias");
        b.conv_bn_mean = get(p + "conv_bn.running_mean");
        b.conv_bn_var = get(p + "conv_bn.running_var");
        b.conv_down_w = get(p + "conv_down.weight");
        b.conv_down_b = get(p + "conv_down.bias");
        b.conv_norm_w = get(p + "conv_norm.weight");
        b.conv_norm_b = get(p + "conv_norm.bias");

        b.ff1_norm_w = get(p + "ff1_norm.weight");
        b.ff1_norm_b = get(p + "ff1_norm.bias");
        b.ff1_up_w = require(p + "ff1_up.weight");
        b.ff1_up_b = get(p + "ff1_up.bias");
        b.ff1_down_w = require(p + "ff1_down.weight");
        b.ff1_down_b = get(p + "ff1_down.bias");

        b.ff2_norm_w = get(p + "ff2_norm.weight");
        b.ff2_norm_b = get(p + "ff2_norm.bias");
        b.ff2_up_w = require(p + "ff2_up.weight");
        b.ff2_up_b = get(p + "ff2_up.bias");
        b.ff2_down_w = require(p + "ff2_down.weight");
        b.ff2_down_b = get(p + "ff2_down.bias");

        b.post_norm_w = get(p + "post_norm.weight");
        b.post_norm_b = get(p + "post_norm.bias");
    }

    // Projector
    auto & pr = model.projector;
    pr.query = require("proj.query");
    pr.ln_w = get("proj.ln.weight");
    pr.ln_b = get("proj.ln.bias");
    pr.linear_w = require("proj.linear.weight");
    pr.linear_b = get("proj.linear.bias");

    pr.blocks.resize(model.hparams.proj_n_layers);
    for (uint32_t il = 0; il < model.hparams.proj_n_layers; il++) {
        auto p = "proj.blk." + std::to_string(il) + ".";
        auto & b = pr.blocks[il];
        b.sa_q_w = require(p + "sa_query.weight"); b.sa_q_b = get(p + "sa_query.bias");
        b.sa_k_w = require(p + "sa_key.weight"); b.sa_k_b = get(p + "sa_key.bias");
        b.sa_v_w = require(p + "sa_value.weight"); b.sa_v_b = get(p + "sa_value.bias");
        b.sa_out_w = require(p + "sa_out.weight"); b.sa_out_b = get(p + "sa_out.bias");
        b.sa_norm_w = get(p + "sa_norm.weight"); b.sa_norm_b = get(p + "sa_norm.bias");

        b.ca_q_w = require(p + "ca_query.weight"); b.ca_q_b = get(p + "ca_query.bias");
        b.ca_k_w = require(p + "ca_key.weight"); b.ca_k_b = get(p + "ca_key.bias");
        b.ca_v_w = require(p + "ca_value.weight"); b.ca_v_b = get(p + "ca_value.bias");
        b.ca_out_w = require(p + "ca_out.weight"); b.ca_out_b = get(p + "ca_out.bias");
        b.ca_norm_w = get(p + "ca_norm.weight"); b.ca_norm_b = get(p + "ca_norm.bias");

        b.ffn_up_w = require(p + "ffn_up.weight"); b.ffn_up_b = get(p + "ffn_up.bias");
        b.ffn_down_w = require(p + "ffn_down.weight"); b.ffn_down_b = get(p + "ffn_down.bias");
        b.ffn_norm_w = get(p + "ffn_norm.weight"); b.ffn_norm_b = get(p + "ffn_norm.bias");
    }

    // LLM
    auto & l = model.llm;
    l.token_embd_w = require("token_embd.weight");
    l.output_norm_w = require("output_norm.weight");
    l.output_w = require("output.weight");

    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        auto p = "blk." + std::to_string(il) + ".";
        auto & b = l.blocks[il];
        b.attn_norm_w = require(p + "attn_norm.weight");
        b.attn_q_w = require(p + "attn_q.weight");
        b.attn_k_w = require(p + "attn_k.weight");
        b.attn_v_w = require(p + "attn_v.weight");
        b.attn_out_w = require(p + "attn_output.weight");
        b.ffn_norm_w = require(p + "ffn_norm.weight");
        b.ffn_gate_w = require(p + "ffn_gate.weight");
        b.ffn_up_w = require(p + "ffn_up.weight");
        b.ffn_down_w = require(p + "ffn_down.weight");
    }

    return true;
}

// ===========================================================================
// Public API (stubs — to be implemented)
// ===========================================================================

extern "C" struct granite_speech_context_params granite_speech_context_default_params(void) {
    return { /*n_threads=*/4, /*verbosity=*/1 };
}

extern "C" struct granite_speech_context * granite_speech_init_from_file(
    const char * path, struct granite_speech_context_params params)
{
    auto * ctx = new granite_speech_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!granite_speech_load_model(ctx->model, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    // Fold batch norm into scale+shift tensors (load-time, once)
    // BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    //       = x * scale + shift
    // where scale = gamma/sqrt(var+eps), shift = beta - mean*scale
    {
        const float eps = 1e-5f;
        const int inner = 2048;  // conv expansion = 2 * d_model
        int folded = 0;
        for (uint32_t il = 0; il < ctx->model.hparams.enc_n_layers; il++) {
            auto & b = ctx->model.encoder.blocks[il];
            if (!b.conv_bn_w || !b.conv_bn_b || !b.conv_bn_mean || !b.conv_bn_var) continue;

            std::vector<float> gamma(inner), beta(inner), mean(inner), var(inner);
            ggml_backend_tensor_get(b.conv_bn_w, gamma.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_get(b.conv_bn_b, beta.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_get(b.conv_bn_mean, mean.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_get(b.conv_bn_var, var.data(), 0, inner * sizeof(float));

            std::vector<float> scale(inner), shift(inner);
            for (int c = 0; c < inner; c++) {
                scale[c] = gamma[c] / std::sqrt(var[c] + eps);
                shift[c] = beta[c] - mean[c] * scale[c];
            }

            // Write precomputed scale/shift back into bn_w/bn_b tensors
            ggml_backend_tensor_set(b.conv_bn_w, scale.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_set(b.conv_bn_b, shift.data(), 0, inner * sizeof(float));
            folded++;
        }
        if (params.verbosity >= 1)
            fprintf(stderr, "granite_speech: BN folded for %d encoder layers\n", folded);
    }

    // Precompute relative position embedding lookup table
    // attention_dists[c][r] = clamp(c - r, -ctx_size, ctx_size) + max_pos_emb
    // RPE lookup: (ctx_size, ctx_size, head_dim) F32
    {
        const int C = 200;   // context_size
        const int max_pos = 512; // max_pos_emb
        const int hd = (int)ctx->model.hparams.enc_head_dim; // 128
        const int emb_size = 2 * max_pos + 1; // 1025

        // Compute attention_dists indices
        std::vector<int> dists(C * C);
        for (int c = 0; c < C; c++)
            for (int r = 0; r < C; r++) {
                int d = c - r;
                if (d < -C) d = -C;
                if (d > C) d = C;
                dists[c * C + r] = d + max_pos;
            }

        // Read the rel_pos_emb weight from layer 0 (same for all layers)
        // rel_pos_emb.weight: ne[0]=128 (head_dim), ne[1]=1025 (2*max_pos+1)
        ggml_tensor * rpe_w = ctx->model.encoder.blocks[0].attn_rel_pos_w;
        if (rpe_w) {
            std::vector<float> emb_table((size_t)emb_size * hd);
            ggml_backend_tensor_get(rpe_w, emb_table.data(), 0, emb_table.size() * sizeof(float));

            // Build lookup: rpe_lookup[c * C * hd + r * hd + d] = emb_table[dists[c,r] * hd + d]
            ctx->rpe_lookup.resize((size_t)C * C * hd);
            for (int c = 0; c < C; c++)
                for (int r = 0; r < C; r++) {
                    int idx = dists[c * C + r];
                    for (int d = 0; d < hd; d++)
                        ctx->rpe_lookup[(size_t)(c * C + r) * hd + d] = emb_table[(size_t)idx * hd + d];
                }

            if (params.verbosity >= 1)
                fprintf(stderr, "granite_speech: RPE lookup precomputed (%d × %d × %d)\n", C, C, hd);
        }
    }

    // Create scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(
        ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        const auto & hp = ctx->model.hparams;
        fprintf(stderr, "granite_speech: loaded %s (enc %u layers, proj %u layers, llm %u layers, vocab %u)\n",
                path, hp.enc_n_layers, hp.proj_n_layers, hp.llm_n_layers, hp.llm_vocab_size);
    }
    return ctx;
}

extern "C" void granite_speech_free(struct granite_speech_context * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->model.buf) ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Mel spectrogram (80 bins, n_fft=512, hop=160, per-utterance max norm)
// ===========================================================================

static void granite_fft(float * in, int N, float * out) {
    if (N <= 1) { out[0] = in[0]; out[1] = 0; return; }
    if (N % 2 != 0) {
        // DFT fallback for odd sizes
        for (int k = 0; k < N; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                double a = -2.0 * M_PI * k * n / N;
                re += in[n] * cos(a); im += in[n] * sin(a);
            }
            out[2*k] = (float)re; out[2*k+1] = (float)im;
        }
        return;
    }
    int half = N / 2;
    std::vector<float> even(half), odd(half);
    for (int i = 0; i < half; i++) { even[i] = in[2*i]; odd[i] = in[2*i+1]; }
    std::vector<float> E(2*half), O(2*half);
    granite_fft(even.data(), half, E.data());
    granite_fft(odd.data(), half, O.data());
    for (int k = 0; k < half; k++) {
        double a = -2.0 * M_PI * k / N;
        float wr = (float)cos(a), wi = (float)sin(a);
        float tre = wr*O[2*k] - wi*O[2*k+1];
        float tim = wr*O[2*k+1] + wi*O[2*k];
        out[2*k] = E[2*k] + tre; out[2*k+1] = E[2*k+1] + tim;
        out[2*(k+half)] = E[2*k] - tre; out[2*(k+half)+1] = E[2*k+1] - tim;
    }
}

extern "C" float * granite_speech_compute_mel(struct granite_speech_context * ctx,
                                              const float * samples, int n_samples,
                                              int * out_n_mels, int * out_T_mel) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    const int n_fft = 512, win_length = 400, hop = 160, n_mels = 80, n_freqs = n_fft / 2 + 1;

    // Load mel filters from GGUF
    std::vector<float> filt((size_t)n_freqs * n_mels);
    if (ctx->model.encoder.mel_filters) {
        ggml_backend_tensor_get(ctx->model.encoder.mel_filters, filt.data(), 0,
                                filt.size() * sizeof(float));
    } else {
        return nullptr;
    }

    // Hann window: win_length=400, zero-padded to n_fft=512
    // torchaudio uses periodic Hann window of length win_length
    std::vector<float> hann(n_fft, 0.0f);
    for (int i = 0; i < win_length; i++)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / win_length));
    // Remaining n_fft - win_length = 112 entries stay zero (zero-padding)

    // Center-pad (zero padding)
    const int pad = n_fft / 2;
    std::vector<float> padded((size_t)n_samples + 2 * pad, 0.0f);
    std::memcpy(padded.data() + pad, samples, n_samples * sizeof(float));

    int T_full = (int)((padded.size() - n_fft) / hop + 1);
    // torchaudio MelSpectrogram does NOT drop the last frame
    int T = T_full;
    if (T <= 0) return nullptr;

    // STFT → power → mel → log10
    std::vector<float> mel((size_t)n_mels * T, 0.0f);
    float mel_max = -1e30f;
    {
        std::vector<float> fi(n_fft * 4, 0.0f), fo(n_fft * 8, 0.0f);
        std::vector<float> power(n_freqs);
        for (int t = 0; t < T; t++) {
            const float * frame = padded.data() + (size_t)t * hop;
            for (int n = 0; n < n_fft; n++) fi[n] = frame[n] * hann[n];
            granite_fft(fi.data(), n_fft, fo.data());
            for (int k = 0; k < n_freqs; k++) {
                float re = fo[2*k], im = fo[2*k+1];
                power[k] = re * re + im * im;
            }
            // mel @ power → log10
            for (int m = 0; m < n_mels; m++) {
                double s = 0.0;
                for (int k = 0; k < n_freqs; k++)
                    s += (double)filt[(size_t)k * n_mels + m] * power[k];
                float lv = std::log10(std::max((float)s, 1e-10f));
                mel[(size_t)t * n_mels + m] = lv;  // (T, n_mels) layout for stacking
                if (lv > mel_max) mel_max = lv;
            }
        }
    }

    // Normalize: max(logmel, mx - 8) / 4 + 1
    const float floor_v = mel_max - 8.0f;
    for (size_t i = 0; i < mel.size(); i++) {
        float v = mel[i];
        if (v < floor_v) v = floor_v;
        mel[i] = v / 4.0f + 1.0f;
    }

    // Drop last frame if odd
    if (T % 2 == 1) T--;

    // Stack 2 adjacent frames: (T, 80) → (T/2, 160)
    int T_stacked = T / 2;
    std::vector<float> stacked((size_t)T_stacked * 160);
    for (int t = 0; t < T_stacked; t++) {
        std::memcpy(stacked.data() + (size_t)t * 160,
                     mel.data() + (size_t)(2*t) * n_mels, n_mels * sizeof(float));
        std::memcpy(stacked.data() + (size_t)t * 160 + n_mels,
                     mel.data() + (size_t)(2*t+1) * n_mels, n_mels * sizeof(float));
    }

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (size_t i = 0; i < stacked.size(); i++) { if(stacked[i]<mn)mn=stacked[i]; if(stacked[i]>mx)mx=stacked[i]; sum+=stacked[i]; }
        fprintf(stderr, "  mel: (%d, 160) min=%.4f max=%.4f mean=%.6f\n",
                T_stacked, mn, mx, sum/stacked.size());
    }

    if (out_n_mels) *out_n_mels = 160;
    if (out_T_mel)  *out_T_mel  = T_stacked;

    float * result = (float *)malloc(stacked.size() * sizeof(float));
    std::memcpy(result, stacked.data(), stacked.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Encoder, Projector, LLM — stubs (to be implemented in next step)
// ===========================================================================

// ===========================================================================
// Depthwise conv helper (CPU, applied between ggml graph segments)
// Each channel c has its own K-tap filter: out[c,t] = sum_k w[k,c] * in[c,t+k-pad]
// ===========================================================================

static void depthwise_conv_1d_cpu(float * out, const float * in,
                                  const float * weight, int channels, int T,
                                  int kernel_size, int pad) {
    for (int c = 0; c < channels; c++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int ti = t + k - pad;
                if (ti >= 0 && ti < T) {
                    // weight layout from GGUF: ne[0]=K, ne[1]=1, ne[2]=C
                    // Data: weight[k + 0*K + c*1*K] = weight[k + c*K]
                    // But GGUF ne ordering: element at [k, 0, c] = data[c * 1 * K + 0 * K + k] = data[c*K + k]
                    sum += weight[(size_t)c * kernel_size + k] * in[(size_t)c * T + ti];
                }
            }
            out[(size_t)c * T + t] = sum;
        }
    }
}

// ===========================================================================
// CPU-based blocked attention with Shaw relative position embeddings
//
// For each block of ctx_size frames:
//   attn_weights = (Q @ K^T) * scale + pos_attn
//   pos_attn[c,r] = sum_d Q[c,d] * RPE[c,r,d] * scale
//   out = softmax(attn_weights) @ V
// ===========================================================================

static void shaw_block_attention_cpu(
    float * out,          // (n_heads * hd, T) output — same layout as input
    const float * Q_data, // (n_heads * hd, T) — ne[0]=n_heads*hd, ne[1]=T
    const float * K_data, // same
    const float * V_data, // same
    const float * rpe,    // (ctx_size, ctx_size, hd) — precomputed RPE lookup for this layer
    int T, int n_heads, int hd, int ctx_size, float scale,
    int remainder)  // frames in last block (0 = no padding)
{
    const int d = n_heads * hd;
    const int n_blocks = (T + ctx_size - 1) / ctx_size;

    // Working buffers
    std::vector<float> scores((size_t)ctx_size * ctx_size);
    std::vector<float> attn_out((size_t)ctx_size * hd);

    for (int blk = 0; blk < n_blocks; blk++) {
        int blk_start = blk * ctx_size;
        int blk_len = (blk == n_blocks - 1 && remainder > 0) ? remainder : ctx_size;

        for (int h = 0; h < n_heads; h++) {
            // Q_block[c, d] = Q_data[(h * hd + d) + (blk_start + c) * d_full]
            // where d_full = n_heads * hd
            // Layout: Q_data is (d, T) in ggml — element [dim, time] = Q_data[dim + time * d]

            // Compute QK^T * scale + pos_attn for this block and head
            for (int c = 0; c < blk_len; c++) {
                for (int r = 0; r < blk_len; r++) {
                    float qk = 0.0f;
                    float pos = 0.0f;
                    for (int dd = 0; dd < hd; dd++) {
                        int q_idx = (h * hd + dd) + (blk_start + c) * d;
                        int k_idx = (h * hd + dd) + (blk_start + r) * d;
                        float q_val = Q_data[q_idx];
                        float k_val = K_data[k_idx];
                        qk += q_val * k_val;
                        // pos_attn: Q[c,d] * RPE[c,r,d]
                        pos += q_val * rpe[(size_t)(c * ctx_size + r) * hd + dd];
                    }
                    scores[c * blk_len + r] = (qk + pos) * scale;
                }
            }

            // Softmax per row
            for (int c = 0; c < blk_len; c++) {
                float max_val = -1e30f;
                for (int r = 0; r < blk_len; r++)
                    if (scores[c * blk_len + r] > max_val) max_val = scores[c * blk_len + r];
                float sum = 0.0f;
                for (int r = 0; r < blk_len; r++) {
                    scores[c * blk_len + r] = std::exp(scores[c * blk_len + r] - max_val);
                    sum += scores[c * blk_len + r];
                }
                float inv_sum = 1.0f / (sum + 1e-10f);
                for (int r = 0; r < blk_len; r++)
                    scores[c * blk_len + r] *= inv_sum;
            }

            // Compute output: out[c, d] = sum_r scores[c, r] * V[r, d]
            for (int c = 0; c < blk_len; c++) {
                for (int dd = 0; dd < hd; dd++) {
                    float sum = 0.0f;
                    for (int r = 0; r < blk_len; r++) {
                        int v_idx = (h * hd + dd) + (blk_start + r) * d;
                        sum += scores[c * blk_len + r] * V_data[v_idx];
                    }
                    // Write to output: out[(h*hd + dd) + (blk_start + c) * d]
                    out[(h * hd + dd) + (blk_start + c) * d] = sum;
                }
            }
        }
    }
}

// ===========================================================================
// CPU-based linear layer helper (builds tiny ggml graph per matmul)
// ===========================================================================

// Apply: out = W @ x + b, where x is (d_in, T), W is (d_in, d_out), out is (d_out, T)
static void cpu_linear(granite_speech_context * ctx, float * out,
                       const float * x, ggml_tensor * W, ggml_tensor * bias,
                       int d_in, int d_out, int T) {
    // Simple CPU matmul (no ggml graph overhead for small ops)
    // W in ggml: ne[0]=d_in, ne[1]=d_out. W[i,j] = data[j * d_in + i]
    // ggml_mul_mat(W, x) computes W^T @ x
    // For F32 weights, do it directly:
    std::vector<float> w_f32((size_t)d_in * d_out);
    ggml_backend_tensor_get(W, w_f32.data(), 0, w_f32.size() * sizeof(float));

    std::vector<float> b_f32;
    if (bias) {
        b_f32.resize(d_out);
        ggml_backend_tensor_get(bias, b_f32.data(), 0, d_out * sizeof(float));
    }

    // out[o, t] = sum_i W[i, o] * x[i, t] + bias[o]
    // = sum_i w_f32[o * d_in + i] * x[i + t * d_in] + b[o]
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < d_out; o++) {
            float sum = 0.0f;
            for (int i = 0; i < d_in; i++)
                sum += w_f32[(size_t)o * d_in + i] * x[(size_t)i + (size_t)t * d_in];
            if (bias) sum += b_f32[o];
            out[(size_t)o + (size_t)t * d_out] = sum;
        }
    }
}

// ===========================================================================
// Conformer encoder — CPU-based forward (not ggml graph)
//
// The encoder uses block-local attention (context_size=200) with Shaw
// relative position embeddings. This is hard to express as a single ggml
// graph, so we implement the encoder as a hybrid: ggml graphs for the
// per-block matmuls, and CPU loops for the block chunking and relative
// position logic.
//
// For simplicity in V1, we use a simplified global attention encoder
// (same as before) and note that block-local attention + rel pos +
// depthwise conv are needed for accuracy. These will be added when
// we have ground truth to compare against.
// ===========================================================================

static ggml_cgraph * granite_build_encoder(granite_speech_context * ctx, int T) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int d = (int)hp.enc_d_model;      // 1024
    const int n_heads = (int)hp.enc_n_heads; // 8
    const int hd = (int)hp.enc_head_dim;     // 128
    const int ff = (int)hp.enc_ff_dim;       // 4096
    const int n_layers = (int)hp.enc_n_layers;
    const int input_dim = (int)hp.enc_input_dim;  // 160
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: (T, input_dim=160) stacked mel frames
    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, input_dim, T);
    ggml_set_name(inp, "enc_input"); ggml_set_input(inp);

    // Input linear: (160 → 1024)
    ggml_tensor * cur = ggml_mul_mat(ctx0, m.encoder.input_w, inp);
    if (m.encoder.input_b) cur = ggml_add(ctx0, cur, m.encoder.input_b);

    // Block-local attention with Shaw relative position embeddings
    const int ctx_size = 200;  // context_size
    const int n_blocks_attn = (T + ctx_size - 1) / ctx_size;
    const int T_padded = n_blocks_attn * ctx_size;

    // Block-diagonal mask: (T, T) F16
    ggml_tensor * block_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
    ggml_set_name(block_mask, "block_mask");
    ggml_set_input(block_mask);

    // RPE lookup as input: (ctx_size * head_dim, ctx_size) = (200*128, 200) F32
    // Layout: rpe[r * hd + d, c] = RPE_lookup[c, r, d] for matmul with Q
    ggml_tensor * rpe_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ctx_size * hd, ctx_size);
    ggml_set_name(rpe_tensor, "rpe_lookup");
    ggml_set_input(rpe_tensor);

    // 16 × Conformer blocks
    for (int il = 0; il < n_layers; il++) {
        const auto & b = m.encoder.blocks[il];

        // --- FFN1 (Macaron half-step) ---
        {
            ggml_tensor * x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.ff1_norm_w) x = ggml_mul(ctx0, x, b.ff1_norm_w);
            if (b.ff1_norm_b) x = ggml_add(ctx0, x, b.ff1_norm_b);
            x = ggml_mul_mat(ctx0, b.ff1_up_w, x);
            if (b.ff1_up_b) x = ggml_add(ctx0, x, b.ff1_up_b);
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, b.ff1_down_w, x);
            if (b.ff1_down_b) x = ggml_add(ctx0, x, b.ff1_down_b);
            cur = ggml_add(ctx0, cur, ggml_scale(ctx0, x, 0.5f));  // half-step residual
        }

        // --- MHSA (Block-local attention, context_size=200, Shaw rel pos) ---
        {
            ggml_tensor * x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.attn_norm_w) x = ggml_mul(ctx0, x, b.attn_norm_w);
            if (b.attn_norm_b) x = ggml_add(ctx0, x, b.attn_norm_b);

            // Q: (d → d), KV: (d → 2d) combined
            ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, x);   // (d, T)
            ggml_tensor * KV = ggml_mul_mat(ctx0, b.attn_kv_w, x); // (2*d, T)

            // Split KV
            int kv_dim = n_heads * hd;
            ggml_tensor * K = ggml_cont(ctx0, ggml_view_2d(ctx0, KV, kv_dim, T, KV->nb[1], 0));
            ggml_tensor * V = ggml_cont(ctx0, ggml_view_2d(ctx0, KV, kv_dim, T, KV->nb[1],
                                           kv_dim * ggml_type_size(KV->type)));

            // Reshape to (hd, nh, T)
            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, T);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, T);

            // Shaw relative position attention — manual blocked implementation
            // flash_attn_ext can't include query-dependent pos_attn bias,
            // so we implement attention manually using ggml_mul_mat + soft_max.
            //
            // Q: (hd, nh, T), K: (hd, nh, T), V: (hd, nh, T)
            // Permute to (hd, T, nh) then reshape into blocks (hd, C, nh*nblocks)
            // for batched matmul.

            // For this layer's RPE: load from the weight tensor
            // rpe_lookup is precomputed for layer 0; for other layers we'd need
            // per-layer lookup. For now, use the precomputed one (approximate).
            // TODO: precompute RPE per layer at init time

            // Permute Q/K/V: (hd, nh, T) → (hd, T, nh)
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            // Use flash_attn_ext with block mask (pos_attn omitted for now)
            // The block mask provides the main structural constraint.
            // Full Shaw RPE would improve accuracy but requires manual attention.
            ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K, V, block_mask,
                                                      attn_scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, n_heads * hd, T);

            // Output projection
            attn = ggml_mul_mat(ctx0, b.attn_out_w, attn);
            if (b.attn_out_b) attn = ggml_add(ctx0, attn, b.attn_out_b);
            cur = ggml_add(ctx0, cur, attn);
        }

        // --- Conv module ---
        // LayerNorm → pointwise up → GLU → depthwise conv → BN → SiLU → pointwise down
        {
            // LayerNorm at the start (the HF ConformerConvModule.norm)
            ggml_tensor * x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.conv_norm_w) x = ggml_mul(ctx0, x, b.conv_norm_w);
            if (b.conv_norm_b) x = ggml_add(ctx0, x, b.conv_norm_b);
            if (b.conv_up_w) {
                // conv_up_w ne=(1, 1024, 4096) — 1×1 conv, reshape to (1024, 4096) for matmul
                int in_ch = (int)b.conv_up_w->ne[1];
                int out_ch = (int)b.conv_up_w->ne[2];
                x = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, b.conv_up_w, in_ch, out_ch), x);
                if (b.conv_up_b) x = ggml_add(ctx0, x, b.conv_up_b);
            }

            // GLU: split into two halves, first half * sigmoid(second half)
            int half_dim = (int)x->ne[0] / 2;
            ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, x, half_dim, T, x->nb[1], 0));
            ggml_tensor * x2 = ggml_cont(ctx0, ggml_view_2d(ctx0, x, half_dim, T, x->nb[1],
                                            half_dim * ggml_type_size(x->type)));
            x = ggml_mul(ctx0, x1, ggml_sigmoid(ctx0, x2));

            // Depthwise conv (kernel=15, groups=2048, pad=7) + batch norm + SiLU
            // conv_dw_w ne=(15, 1, 2048) — each of 2048 channels has its own 15-tap filter
            // ggml doesn't support grouped conv, so we use ggml_conv_1d per-channel
            // by treating x as (T, 2048) and convolving with (15, 1, 2048) with pad=7
            //
            // For a depthwise conv: output[c, t] = sum_k weight[k, 0, c] * input[c, t+k-pad]
            // This is equivalent to a standard conv with C_in=1 applied independently per channel.
            // ggml_conv_1d(kernel=(15, 1, 2048), input=(T, 2048)) should work IF ggml
            // treats ne[1]=1 as the input channel dim.
            //
            // Actually, ggml_conv_1d does: output = kernel^T @ im2col(input)
            // With kernel (K=15, C_in=1, C_out=2048) and input (T, C_in=2048):
            //   This is a 15-tap conv with 2048 input channels and 2048 output channels,
            //   NOT depthwise. It would produce cross-channel mixing.
            //
            // For true depthwise, we'd need to loop over channels or use a diagonal weight.
            // WORKAROUND: skip the depthwise conv and just apply batch_norm + SiLU.
            // The batch norm at least preserves the per-channel statistics.
            // Batch norm (applied to x of shape (inner_dim=2048, T)):
            // BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
            //       = x * scale + shift
            // where scale = gamma / sqrt(var + eps), shift = beta - gamma * mean / sqrt(var + eps)
            // These 1D (2048,) tensors broadcast over T via ggml_mul/ggml_add
            // Depthwise conv (kernel=15, groups=inner_dim, pad=7)
            // Uses ggml_conv_2d_dw_direct (same pattern as Parakeet encoder)
            // Re-enabled for testing — using same pattern as Parakeet
            if (b.conv_dw_w) {
                int K = (int)b.conv_dw_w->ne[0];  // 15
                int inner = half_dim;  // 2048
                int dw_pad = K / 2;   // 7
                ggml_tensor * dw_w = ggml_cast(ctx0, b.conv_dw_w, GGML_TYPE_F32);
                ggml_tensor * dw_w_4d = ggml_reshape_4d(ctx0, dw_w, K, 1, 1, inner);
                ggml_tensor * x_t = ggml_cont(ctx0, ggml_transpose(ctx0, x));
                x_t = ggml_reshape_4d(ctx0, x_t, T, 1, inner, 1);
                x_t = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, x_t, 1, 1, dw_pad, 0, 1, 1);
                x = ggml_cont(ctx0, ggml_permute(ctx0, x_t, 1, 2, 0, 3));
                x = ggml_reshape_2d(ctx0, x, inner, T);
            }

            // BN (precomputed at load time): bn_w = scale, bn_b = shift
            if (b.conv_bn_w && b.conv_bn_b) {
                x = ggml_mul(ctx0, x, b.conv_bn_w);
                x = ggml_add(ctx0, x, b.conv_bn_b);
            }
            x = ggml_silu(ctx0, x);

            // Pointwise down: conv_down_w ne=(1, 2048, 1024)
            if (b.conv_down_w) {
                int in_ch = (int)b.conv_down_w->ne[1];
                int out_ch = (int)b.conv_down_w->ne[2];
                x = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, b.conv_down_w, in_ch, out_ch), x);
                if (b.conv_down_b) x = ggml_add(ctx0, x, b.conv_down_b);
            }

            cur = ggml_add(ctx0, cur, x);
        }

        // --- FFN2 (Macaron half-step) ---
        {
            ggml_tensor * x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.ff2_norm_w) x = ggml_mul(ctx0, x, b.ff2_norm_w);
            if (b.ff2_norm_b) x = ggml_add(ctx0, x, b.ff2_norm_b);
            x = ggml_mul_mat(ctx0, b.ff2_up_w, x);
            if (b.ff2_up_b) x = ggml_add(ctx0, x, b.ff2_up_b);
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, b.ff2_down_w, x);
            if (b.ff2_down_b) x = ggml_add(ctx0, x, b.ff2_down_b);
            cur = ggml_add(ctx0, cur, ggml_scale(ctx0, x, 0.5f));
        }

        // --- Post LayerNorm ---
        {
            cur = ggml_norm(ctx0, cur, 1e-5f);
            if (b.post_norm_w) cur = ggml_mul(ctx0, cur, b.post_norm_w);
            if (b.post_norm_b) cur = ggml_add(ctx0, cur, b.post_norm_b);
        }

        // Mid-CTC residual at layer 8 (after 8th layer = index 7)
        if (il == n_layers / 2 - 1 && m.encoder.ctc_out_w && m.encoder.ctc_mid_w) {
            // out: (1024 → 348) → softmax → out_mid: (348 → 1024) → add to hidden
            ggml_tensor * mid = ggml_mul_mat(ctx0, m.encoder.ctc_out_w, cur);
            if (m.encoder.ctc_out_b) mid = ggml_add(ctx0, mid, m.encoder.ctc_out_b);
            mid = ggml_soft_max(ctx0, mid);  // softmax over last dim (348)
            mid = ggml_mul_mat(ctx0, m.encoder.ctc_mid_w, mid);
            if (m.encoder.ctc_mid_b) mid = ggml_add(ctx0, mid, m.encoder.ctc_mid_b);
            cur = ggml_add(ctx0, cur, mid);
        }
    }

    ggml_set_name(cur, "enc_output");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Build a tiny ggml graph for a single matmul: out = W @ x [+ bias]
// x: (d_in, T), W: ggml tensor, out: (d_out, T)
static bool run_matmul(granite_speech_context * ctx, float * out,
                       const float * x, int d_in, int T,
                       ggml_tensor * W, ggml_tensor * bias, int d_out) {
    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_in, T);
    ggml_set_name(inp, "mm_in"); ggml_set_input(inp);

    ggml_tensor * r = ggml_mul_mat(ctx0, W, inp);
    if (bias) r = ggml_add(ctx0, r, bias);
    ggml_set_name(r, "mm_out");
    ggml_build_forward_expand(gf, r);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mm_in"), x, 0, (size_t)d_in * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "mm_out"), out, 0, (size_t)d_out * T * sizeof(float));
    return true;
}

// Build and run the conv module as a ggml graph
static bool run_conv_module(granite_speech_context * ctx, float * out,
                            const float * x, int d, int T,
                            const granite_enc_block & b) {
    const int inner = d * 2;  // conv_expansion_factor = 2
    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 256, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(inp, "conv_in"); ggml_set_input(inp);

    // LayerNorm
    ggml_tensor * cur = ggml_norm(ctx0, inp, 1e-5f);
    if (b.conv_norm_w) cur = ggml_mul(ctx0, cur, b.conv_norm_w);
    if (b.conv_norm_b) cur = ggml_add(ctx0, cur, b.conv_norm_b);

    // Pointwise up
    if (b.conv_up_w) {
        int in_ch = (int)b.conv_up_w->ne[1], out_ch = (int)b.conv_up_w->ne[2];
        cur = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, b.conv_up_w, in_ch, out_ch), cur);
        if (b.conv_up_b) cur = ggml_add(ctx0, cur, b.conv_up_b);
    }

    // GLU
    int half = (int)cur->ne[0] / 2;
    ggml_tensor * x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, half, T, cur->nb[1], 0));
    ggml_tensor * x2 = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, half, T, cur->nb[1], half * sizeof(float)));
    cur = ggml_mul(ctx0, x1, ggml_sigmoid(ctx0, x2));

    // Depthwise conv
    if (b.conv_dw_w) {
        int K = (int)b.conv_dw_w->ne[0];
        ggml_tensor * dw_w = ggml_cast(ctx0, b.conv_dw_w, GGML_TYPE_F32);
        ggml_tensor * dw_w_4d = ggml_reshape_4d(ctx0, dw_w, K, 1, 1, inner);
        ggml_tensor * x_t = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        x_t = ggml_reshape_4d(ctx0, x_t, T, 1, inner, 1);
        x_t = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, x_t, 1, 1, K/2, 0, 1, 1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, x_t, 1, 2, 0, 3));
        cur = ggml_reshape_2d(ctx0, cur, inner, T);
    }

    // BN (folded) + SiLU
    if (b.conv_bn_w && b.conv_bn_b) {
        cur = ggml_mul(ctx0, cur, b.conv_bn_w);
        cur = ggml_add(ctx0, cur, b.conv_bn_b);
    }
    cur = ggml_silu(ctx0, cur);

    // Pointwise down
    if (b.conv_down_w) {
        int in_ch = (int)b.conv_down_w->ne[1], out_ch = (int)b.conv_down_w->ne[2];
        cur = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, b.conv_down_w, in_ch, out_ch), cur);
        if (b.conv_down_b) cur = ggml_add(ctx0, cur, b.conv_down_b);
    }

    ggml_set_name(cur, "conv_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "conv_in"), x, 0, (size_t)d * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "conv_out"), out, 0, (size_t)d * T * sizeof(float));
    return true;
}

// Build and run FFN as a ggml graph: LayerNorm → up → SiLU → down
static bool run_ffn(granite_speech_context * ctx, float * out,
                    const float * x, int d, int T,
                    ggml_tensor * norm_w, ggml_tensor * norm_b,
                    ggml_tensor * up_w, ggml_tensor * up_b,
                    ggml_tensor * down_w, ggml_tensor * down_b) {
    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(inp, "ffn_in"); ggml_set_input(inp);

    ggml_tensor * cur = ggml_norm(ctx0, inp, 1e-5f);
    if (norm_w) cur = ggml_mul(ctx0, cur, norm_w);
    if (norm_b) cur = ggml_add(ctx0, cur, norm_b);
    cur = ggml_mul_mat(ctx0, up_w, cur);
    if (up_b) cur = ggml_add(ctx0, cur, up_b);
    cur = ggml_silu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, down_w, cur);
    if (down_b) cur = ggml_add(ctx0, cur, down_b);

    ggml_set_name(cur, "ffn_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ffn_in"), x, 0, (size_t)d * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "ffn_out"), out, 0, (size_t)d * T * sizeof(float));
    return true;
}

// CPU LayerNorm: out = (x - mean) / sqrt(var + eps) * w + b
static void cpu_layernorm(float * out, const float * x, const float * w, const float * b,
                          int d, int T, float eps) {
    for (int t = 0; t < T; t++) {
        const float * xt = x + (size_t)t * d;
        float * ot = out + (size_t)t * d;
        float mean = 0, var = 0;
        for (int i = 0; i < d; i++) mean += xt[i];
        mean /= d;
        for (int i = 0; i < d; i++) { float v = xt[i] - mean; var += v * v; }
        var /= d;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < d; i++)
            ot[i] = (xt[i] - mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

extern "C" float * granite_speech_run_encoder(struct granite_speech_context * ctx,
                                              const float * mel, int n_mels, int T_mel,
                                              int * out_N, int * out_dim) {
    if (!ctx || !mel || n_mels != 160) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;        // 1024
    const int n_heads = (int)hp.enc_n_heads;  // 8
    const int hd = (int)hp.enc_head_dim;      // 128
    const int n_layers = (int)hp.enc_n_layers; // 16
    const int ctx_size = 200;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = T_mel;
    const int remainder = T % ctx_size;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  encoder: per-layer processing T=%d d=%d layers=%d ctx=%d\n",
                T, d, n_layers, ctx_size);

    // Input linear: mel (160, T) → hidden (d, T)
    std::vector<float> hidden((size_t)d * T);
    run_matmul(ctx, hidden.data(), mel, n_mels, T,
               ctx->model.encoder.input_w, ctx->model.encoder.input_b, d);

    // Buffers
    std::vector<float> ffn_out((size_t)d * T);
    std::vector<float> Q((size_t)d * T), KV((size_t)d * 2 * T);
    std::vector<float> attn_out((size_t)d * T);
    std::vector<float> conv_out((size_t)d * T);

    // Precompute per-layer RPE lookups
    std::vector<std::vector<float>> rpe_per_layer(n_layers);
    {
        const int max_pos = 512;
        std::vector<int> dists(ctx_size * ctx_size);
        for (int c = 0; c < ctx_size; c++)
            for (int r = 0; r < ctx_size; r++) {
                int dd = std::max(-ctx_size, std::min(ctx_size, c - r));
                dists[c * ctx_size + r] = dd + max_pos;
            }
        for (int il = 0; il < n_layers; il++) {
            ggml_tensor * rpe_w = ctx->model.encoder.blocks[il].attn_rel_pos_w;
            if (!rpe_w) continue;
            std::vector<float> emb((size_t)(2 * max_pos + 1) * hd);
            ggml_backend_tensor_get(rpe_w, emb.data(), 0, emb.size() * sizeof(float));
            rpe_per_layer[il].resize((size_t)ctx_size * ctx_size * hd);
            for (int c = 0; c < ctx_size; c++)
                for (int r = 0; r < ctx_size; r++) {
                    int idx = dists[c * ctx_size + r];
                    for (int dd = 0; dd < hd; dd++)
                        rpe_per_layer[il][(size_t)(c * ctx_size + r) * hd + dd] = emb[(size_t)idx * hd + dd];
                }
        }
    }

    for (int il = 0; il < n_layers; il++) {
        const auto & b = ctx->model.encoder.blocks[il];

        // --- FFN1 (Macaron half-step) ---
        run_ffn(ctx, ffn_out.data(), hidden.data(), d, T,
                b.ff1_norm_w, b.ff1_norm_b, b.ff1_up_w, b.ff1_up_b, b.ff1_down_w, b.ff1_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++) hidden[i] += 0.5f * ffn_out[i];

        // --- Attention: norm → Q/KV projections → Shaw attention on CPU ---
        // LayerNorm
        std::vector<float> normed((size_t)d * T);
        {
            std::vector<float> nw(d), nb(d);
            if (b.attn_norm_w) ggml_backend_tensor_get(b.attn_norm_w, nw.data(), 0, d * sizeof(float));
            if (b.attn_norm_b) ggml_backend_tensor_get(b.attn_norm_b, nb.data(), 0, d * sizeof(float));
            cpu_layernorm(normed.data(), hidden.data(), b.attn_norm_w ? nw.data() : nullptr,
                          b.attn_norm_b ? nb.data() : nullptr, d, T, 1e-5f);
        }

        // Q projection: (d → d)
        run_matmul(ctx, Q.data(), normed.data(), d, T, b.attn_q_w, nullptr, d);

        // KV projection: (d → 2d)
        run_matmul(ctx, KV.data(), normed.data(), d, T, b.attn_kv_w, nullptr, d * 2);

        // Split KV: KV layout is (2*d, T) — ne[0]=2*d per frame
        // First d values = K, next d values = V
        std::vector<float> K((size_t)d * T), V((size_t)d * T);
        for (int t = 0; t < T; t++) {
            std::memcpy(K.data() + (size_t)t * d, KV.data() + (size_t)t * 2 * d, d * sizeof(float));
            std::memcpy(V.data() + (size_t)t * d, KV.data() + (size_t)t * 2 * d + d, d * sizeof(float));
        }

        // Shaw block attention on CPU
        shaw_block_attention_cpu(attn_out.data(), Q.data(), K.data(), V.data(),
                                 rpe_per_layer[il].empty() ? nullptr : rpe_per_layer[il].data(),
                                 T, n_heads, hd, ctx_size, attn_scale, remainder);

        if (ctx->params.verbosity >= 2 && il == 0) {
            fprintf(stderr, "  L0 after attn[0][:4] = %.4f %.4f %.4f %.4f\n",
                    attn_out[0], attn_out[1], attn_out[2], attn_out[3]);
        }

        // Output projection + residual
        std::vector<float> proj_out((size_t)d * T);
        run_matmul(ctx, proj_out.data(), attn_out.data(), d, T, b.attn_out_w, b.attn_out_b, d);
        for (size_t i = 0; i < (size_t)d * T; i++) hidden[i] += proj_out[i];

        // --- Conv module ---
        run_conv_module(ctx, conv_out.data(), hidden.data(), d, T, b);
        for (size_t i = 0; i < (size_t)d * T; i++) hidden[i] += conv_out[i];

        // --- FFN2 (Macaron half-step) ---
        run_ffn(ctx, ffn_out.data(), hidden.data(), d, T,
                b.ff2_norm_w, b.ff2_norm_b, b.ff2_up_w, b.ff2_up_b, b.ff2_down_w, b.ff2_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++) hidden[i] += 0.5f * ffn_out[i];

        // --- Post LayerNorm ---
        {
            std::vector<float> nw(d), nb(d);
            if (b.post_norm_w) ggml_backend_tensor_get(b.post_norm_w, nw.data(), 0, d * sizeof(float));
            if (b.post_norm_b) ggml_backend_tensor_get(b.post_norm_b, nb.data(), 0, d * sizeof(float));
            cpu_layernorm(hidden.data(), hidden.data(), b.post_norm_w ? nw.data() : nullptr,
                          b.post_norm_b ? nb.data() : nullptr, d, T, 1e-5f);
        }

        // --- Mid-CTC residual at layer 8 ---
        if (il == n_layers / 2 - 1 && ctx->model.encoder.ctc_out_w && ctx->model.encoder.ctc_mid_w) {
            std::vector<float> mid_out(348 * T), mid_back((size_t)d * T);
            run_matmul(ctx, mid_out.data(), hidden.data(), d, T,
                       ctx->model.encoder.ctc_out_w, ctx->model.encoder.ctc_out_b, 348);
            // Softmax per frame
            for (int t = 0; t < T; t++) {
                float * row = mid_out.data() + t * 348;
                float mx = -1e30f;
                for (int i = 0; i < 348; i++) if (row[i] > mx) mx = row[i];
                float sum = 0;
                for (int i = 0; i < 348; i++) { row[i] = std::exp(row[i] - mx); sum += row[i]; }
                for (int i = 0; i < 348; i++) row[i] /= sum;
            }
            run_matmul(ctx, mid_back.data(), mid_out.data(), 348, T,
                       ctx->model.encoder.ctc_mid_w, ctx->model.encoder.ctc_mid_b, d);
            for (size_t i = 0; i < (size_t)d * T; i++) hidden[i] += mid_back[i];
        }

        if (ctx->params.verbosity >= 2 && (il % 4 == 3 || il == n_layers - 1))
            fprintf(stderr, "  encoder layer %d/%d\n", il + 1, n_layers);
    }

    size_t total = (size_t)T * d;
    float * result = (float *)malloc(total * sizeof(float));
    std::memcpy(result, hidden.data(), total * sizeof(float));

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (size_t i = 0; i < total; i++) { if(result[i]<mn)mn=result[i]; if(result[i]>mx)mx=result[i]; sum+=result[i]; }
        fprintf(stderr, "  encoder output: (%d, %d) min=%.4f max=%.4f mean=%.6f\n",
                T_mel, d, mn, mx, sum/total);
        fprintf(stderr, "  encoder[0][:4] = %.4f %.4f %.4f %.4f\n", result[0], result[1], result[2], result[3]);
    }

    if (out_N) *out_N = T_mel;
    if (out_dim) *out_dim = d;
    return result;
}

// ===========================================================================
// Q-Former projector (2 layers: self-attn + cross-attn + FFN per layer)
// ===========================================================================

static ggml_cgraph * granite_build_projector(granite_speech_context * ctx, int enc_len) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int d = (int)hp.proj_d_model;       // 1024
    const int n_heads = (int)hp.proj_n_heads;  // 16
    const int hd = d / n_heads;               // 64
    const int ff = (int)hp.proj_ff_dim;       // 4096
    const int n_layers = (int)hp.proj_n_layers;
    const int llm_d = (int)hp.llm_d_model;    // 2048
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    // Query tokens: (1, n_query=3, 1024)
    int n_query = (int)m.projector.query->ne[1];  // 3

    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Encoder output as input
    ggml_tensor * enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, enc_len);
    ggml_set_name(enc, "proj_enc_input"); ggml_set_input(enc);

    // Learned query tokens: extract from (1, n_query, d) → (d, n_query)
    ggml_tensor * query = ggml_reshape_2d(ctx0, m.projector.query, d, n_query);

    // Apply input LayerNorm to encoder features
    ggml_tensor * enc_normed = enc;
    if (m.projector.ln_w) {
        enc_normed = ggml_norm(ctx0, enc, 1e-12f);
        enc_normed = ggml_mul(ctx0, enc_normed, m.projector.ln_w);
        if (m.projector.ln_b) enc_normed = ggml_add(ctx0, enc_normed, m.projector.ln_b);
    }

    ggml_tensor * cur = query;  // (d, n_query)

    for (int il = 0; il < n_layers; il++) {
        const auto & b = m.projector.blocks[il];

        // --- Self-attention among query tokens ---
        {
            ggml_tensor * Q = ggml_mul_mat(ctx0, b.sa_q_w, cur);
            if (b.sa_q_b) Q = ggml_add(ctx0, Q, b.sa_q_b);
            ggml_tensor * K = ggml_mul_mat(ctx0, b.sa_k_w, cur);
            if (b.sa_k_b) K = ggml_add(ctx0, K, b.sa_k_b);
            ggml_tensor * V = ggml_mul_mat(ctx0, b.sa_v_w, cur);
            if (b.sa_v_b) V = ggml_add(ctx0, V, b.sa_v_b);

            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, n_query);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, n_query);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, n_query);
            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            V = ggml_permute(ctx0, V, 0, 2, 1, 3);

            ggml_tensor * sa = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr,
                                                    attn_scale, 0.0f, 0.0f);
            sa = ggml_reshape_2d(ctx0, sa, d, n_query);
            sa = ggml_mul_mat(ctx0, b.sa_out_w, sa);
            if (b.sa_out_b) sa = ggml_add(ctx0, sa, b.sa_out_b);

            cur = ggml_add(ctx0, cur, sa);
            // LayerNorm
            if (b.sa_norm_w) {
                cur = ggml_norm(ctx0, cur, 1e-12f);
                cur = ggml_mul(ctx0, cur, b.sa_norm_w);
                if (b.sa_norm_b) cur = ggml_add(ctx0, cur, b.sa_norm_b);
            }
        }

        // --- Cross-attention: queries attend to encoder output ---
        {
            ggml_tensor * Q = ggml_mul_mat(ctx0, b.ca_q_w, cur);
            if (b.ca_q_b) Q = ggml_add(ctx0, Q, b.ca_q_b);
            ggml_tensor * K = ggml_mul_mat(ctx0, b.ca_k_w, enc_normed);
            if (b.ca_k_b) K = ggml_add(ctx0, K, b.ca_k_b);
            ggml_tensor * V = ggml_mul_mat(ctx0, b.ca_v_w, enc_normed);
            if (b.ca_v_b) V = ggml_add(ctx0, V, b.ca_v_b);

            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, n_query);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, enc_len);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, enc_len);
            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            V = ggml_permute(ctx0, V, 0, 2, 1, 3);

            ggml_tensor * ca = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr,
                                                    attn_scale, 0.0f, 0.0f);
            ca = ggml_reshape_2d(ctx0, ca, d, n_query);
            ca = ggml_mul_mat(ctx0, b.ca_out_w, ca);
            if (b.ca_out_b) ca = ggml_add(ctx0, ca, b.ca_out_b);

            cur = ggml_add(ctx0, cur, ca);
            if (b.ca_norm_w) {
                cur = ggml_norm(ctx0, cur, 1e-12f);
                cur = ggml_mul(ctx0, cur, b.ca_norm_w);
                if (b.ca_norm_b) cur = ggml_add(ctx0, cur, b.ca_norm_b);
            }
        }

        // --- FFN ---
        {
            ggml_tensor * x = ggml_mul_mat(ctx0, b.ffn_up_w, cur);
            if (b.ffn_up_b) x = ggml_add(ctx0, x, b.ffn_up_b);
            x = ggml_gelu_erf(ctx0, x);
            x = ggml_mul_mat(ctx0, b.ffn_down_w, x);
            if (b.ffn_down_b) x = ggml_add(ctx0, x, b.ffn_down_b);

            cur = ggml_add(ctx0, cur, x);
            if (b.ffn_norm_w) {
                cur = ggml_norm(ctx0, cur, 1e-12f);
                cur = ggml_mul(ctx0, cur, b.ffn_norm_w);
                if (b.ffn_norm_b) cur = ggml_add(ctx0, cur, b.ffn_norm_b);
            }
        }
    }

    // Final linear projection: (1024 → 2048)
    cur = ggml_mul_mat(ctx0, m.projector.linear_w, cur);
    if (m.projector.linear_b) cur = ggml_add(ctx0, cur, m.projector.linear_b);

    ggml_set_name(cur, "proj_output");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float * granite_speech_run_projector(struct granite_speech_context * ctx,
                                                const float * enc_out, int enc_len, int enc_dim,
                                                int * out_N, int * out_dim) {
    if (!ctx || !enc_out || enc_dim != (int)ctx->model.hparams.proj_d_model) return nullptr;
    const int d = enc_dim;                     // 1024
    const int llm_d = (int)ctx->model.hparams.llm_d_model;  // 2048
    const int window_size = (int)ctx->model.hparams.window_size;      // 15
    const int downsample = (int)ctx->model.hparams.downsample_rate;   // 5
    const int n_query = window_size / downsample;  // 3
    const int nblocks = (enc_len + window_size - 1) / window_size;    // ceil
    const int total_tokens = nblocks * n_query;

    if (ctx->params.verbosity >= 2)
        fprintf(stderr, "  projector: enc_len=%d window=%d nblocks=%d n_query=%d total_tokens=%d\n",
                enc_len, window_size, nblocks, n_query, total_tokens);

    // Pad encoder output to multiple of window_size
    int padded_len = nblocks * window_size;
    std::vector<float> padded((size_t)padded_len * d, 0.0f);
    std::memcpy(padded.data(), enc_out, (size_t)enc_len * d * sizeof(float));

    // Run Q-Former per window
    std::vector<float> all_proj((size_t)total_tokens * llm_d);

    for (int blk = 0; blk < nblocks; blk++) {
        const float * window_data = padded.data() + (size_t)blk * window_size * d;

        // Build Q-Former graph for this window
        ggml_cgraph * gf = granite_build_projector(ctx, window_size);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "proj_enc_input"), window_data, 0,
                                (size_t)window_size * d * sizeof(float));

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;

        ggml_tensor * out = ggml_graph_get_tensor(gf, "proj_output");
        ggml_backend_tensor_get(out, all_proj.data() + (size_t)blk * n_query * llm_d, 0,
                                (size_t)n_query * llm_d * sizeof(float));
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  projector: %d windows × %d queries = %d audio tokens (dim=%d)\n",
                nblocks, n_query, total_tokens, llm_d);

    float * result = (float *)malloc(all_proj.size() * sizeof(float));
    std::memcpy(result, all_proj.data(), all_proj.size() * sizeof(float));
    if (out_N) *out_N = total_tokens;
    if (out_dim) *out_dim = llm_d;
    return result;
}

extern "C" bool granite_speech_kv_init(struct granite_speech_context * ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0) return false;
    const auto & hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int nl = (int)hp.llm_n_layers;
    size_t k_size = (size_t)ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * n_kv * nl;

    ggml_init_params ip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ctx->kv_ctx = ggml_init(ip);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, k_size * 2);
    char * base = (char *)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + ggml_nbytes(ctx->kv_k));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "granite_speech: kv cache %.0f MiB\n", (k_size * 2) / 1048576.0);
    return true;
}

extern "C" void granite_speech_kv_reset(struct granite_speech_context * ctx) {
    if (ctx && ctx->kv_buf) ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

// ===========================================================================
// Granite LLM graph (40 layers, GQA 16/4, μP multipliers)
// ===========================================================================

static ggml_cgraph * granite_build_llm_kv(granite_speech_context * ctx,
                                          int n_past, int n_tokens) {
    const auto & m = ctx->model;
    const auto & hp = m.hparams;
    const int d = (int)hp.llm_d_model;      // 2048
    const int n_q = (int)hp.llm_n_heads;    // 16
    const int n_kv = (int)hp.llm_n_kv_heads;// 4
    const int hd = (int)hp.llm_head_dim;    // 128
    const int n_layers = (int)hp.llm_n_layers;
    const int vocab = (int)hp.llm_vocab_size;

    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds, "inputs_embeds"); ggml_set_input(embeds);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions"); ggml_set_input(positions);

    ggml_tensor * causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask"); ggml_set_input(causal_mask);
    }

    ggml_tensor * cur = embeds;

    for (int il = 0; il < n_layers; il++) {
        const auto & b = m.llm.blocks[il];
        ggml_tensor * residual = cur;

        // Pre-RMSNorm
        cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        cur = ggml_mul(ctx0, cur, b.attn_norm_w);

        // GQA self-attention
        ggml_tensor * Q = ggml_mul_mat(ctx0, b.attn_q_w, cur);
        ggml_tensor * K = ggml_mul_mat(ctx0, b.attn_k_w, cur);
        ggml_tensor * V = ggml_mul_mat(ctx0, b.attn_v_w, cur);

        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, n_tokens);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, n_tokens);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, n_tokens);

        // RoPE
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, 0, 0,
                          hp.llm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, 0, 0,
                          hp.llm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute K/V for cache write
        ggml_tensor * K_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor * V_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // KV cache write
        ggml_tensor * k_dst = ggml_view_4d(ctx0, ctx->kv_k, hd, n_tokens, n_kv, 1,
                                           ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
                                           (size_t)il * ctx->kv_k->nb[3] +
                                           (size_t)n_past * ctx->kv_k->nb[1]);
        ggml_tensor * v_dst = ggml_view_4d(ctx0, ctx->kv_v, hd, n_tokens, n_kv, 1,
                                           ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
                                           (size_t)il * ctx->kv_v->nb[3] +
                                           (size_t)n_past * ctx->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_dst));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_dst));

        // KV cache read
        int Lk = n_past + n_tokens;
        ggml_tensor * Kfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_k, hd, Lk, n_kv,
                                        ctx->kv_k->nb[1], ctx->kv_k->nb[2],
                                        (size_t)il * ctx->kv_k->nb[3]));
        ggml_tensor * Vfull = ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_v, hd, Lk, n_kv,
                                        ctx->kv_v->nb[1], ctx->kv_v->nb[2],
                                        (size_t)il * ctx->kv_v->nb[3]));

        // GQA expansion
        const int n_kv_grp = n_q / n_kv;
        if (n_kv_grp > 1) {
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
            K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, n_kv_grp, n_kv);
            V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, n_kv_grp, n_kv);
            Kfull = ggml_reshape_3d(ctx0, K4, hd, Lk, n_q);
            Vfull = ggml_reshape_3d(ctx0, V4, hd, Lk, n_q);
        }

        // Flash attention (μP: attention_multiplier replaces 1/sqrt(hd))
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask,
                                                  hp.attention_multiplier, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, n_q * hd, n_tokens);
        attn = ggml_mul_mat(ctx0, b.attn_out_w, attn);

        // μP: residual_multiplier scales the residual addition
        cur = ggml_add(ctx0, residual, ggml_scale(ctx0, attn, hp.residual_multiplier));

        // FFN: Pre-RMSNorm + SwiGLU
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        cur = ggml_mul(ctx0, cur, b.ffn_norm_w);

        ggml_tensor * gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, b.ffn_gate_w, cur));
        ggml_tensor * up = ggml_mul_mat(ctx0, b.ffn_up_w, cur);
        cur = ggml_mul_mat(ctx0, b.ffn_down_w, ggml_mul(ctx0, gate, up));
        cur = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, hp.residual_multiplier));
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // LM head (separate, not tied) with μP logits scaling
    if (n_tokens > 1) {
        cur = ggml_view_1d(ctx0, cur, d, (size_t)(n_tokens - 1) * d * sizeof(float));
        cur = ggml_reshape_2d(ctx0, cur, d, 1);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    cur = ggml_scale(ctx0, cur, 1.0f / hp.logits_scaling);  // μP logits scaling

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float * granite_speech_run_llm_kv(struct granite_speech_context * ctx,
                                             const float * inputs_embeds,
                                             int n_tokens, int n_past,
                                             int * out_n_tokens, int * out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0) return nullptr;
    const auto & hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        int Lk = n_past + n_tokens;
        mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++)
            for (int k = 0; k < Lk; k++)
                if (k > n_past + q) mask[(size_t)q * Lk + k] = neg_inf;
    }

    ggml_cgraph * gf = granite_build_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;

    ggml_tensor * logits_t = ggml_graph_get_tensor(gf, "logits");
    float * result = (float *)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    if (out_n_tokens) *out_n_tokens = 1;
    if (out_vocab_size) *out_vocab_size = vocab;
    return result;
}

extern "C" float * granite_speech_embed_tokens(struct granite_speech_context * ctx,
                                               const int32_t * input_ids, int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0) return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;

    ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids"); ggml_set_input(ids);
    ggml_tensor * out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);

    // Apply embedding multiplier (μP)
    out = ggml_scale(ctx0, out, ctx->model.hparams.embedding_multiplier);

    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), input_ids, 0,
                            (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;
    ggml_tensor * emb = ggml_graph_get_tensor(gf, "embeds");
    float * result = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    ggml_backend_tensor_get(emb, result, 0, (size_t)n_tokens * d * sizeof(float));
    return result;
}

extern "C" char * granite_speech_transcribe(struct granite_speech_context *, const float *, int) {
    fprintf(stderr, "granite_speech: full transcribe not yet implemented\n");
    return nullptr;
}
