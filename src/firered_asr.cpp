// firered_asr.cpp — FireRedASR2-AED runtime.
//
// Architecture: Conformer encoder (16L, d=1280, 20 heads, rel-PE, macaron FFN)
//             + Transformer decoder (16L, d=1280, cross-attention, GELU FFN)
//
// The encoder is a standard Conformer with:
//   - Conv2d subsampling (2x 3x3 stride-2 → 4x temporal reduction)
//   - Macaron-style FFN (half-step pre+post around attention)
//   - Relative positional encoding with learnable pos_bias_u/v
//   - Depthwise separable convolution (kernel=33, GLU gating, BatchNorm, Swish)
//
// The decoder is a standard Transformer with:
//   - Sinusoidal positional encoding
//   - Masked self-attention + cross-attention + GELU FFN
//   - Pre-norm (LayerNorm before each sub-layer)

#include "firered_asr.h"

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// Model structures
// ===========================================================================

struct firered_hparams {
    int d_model = 1280;
    int n_head = 20;
    int d_inner = 5120;
    int n_layers_enc = 16;
    int n_layers_dec = 16;
    int idim = 80;   // mel bins
    int odim = 8667; // vocab size
    int subsample = 4;
    int kernel_size = 33;
    int pe_maxlen = 5000;
    int sos_id = 3;
    int eos_id = 4;
    int blank_id = 0;
    int pad_id = 2;
    // Derived
    int head_dim = 64; // d_model / n_head
};

// --- Encoder ---

struct firered_enc_ffn {
    // Macaron FFN: LayerNorm → Linear(d→4d) → Swish → Dropout → Linear(4d→d)
    ggml_tensor* ln_w = nullptr;   // net.0.weight [d_model]
    ggml_tensor* ln_b = nullptr;   // net.0.bias
    ggml_tensor* up_w = nullptr;   // net.1.weight [d_inner, d_model]
    ggml_tensor* up_b = nullptr;   // net.1.bias
    ggml_tensor* down_w = nullptr; // net.4.weight [d_model, d_inner]
    ggml_tensor* down_b = nullptr; // net.4.bias
};

struct firered_enc_mhsa {
    // Relative-position multi-head self-attention
    ggml_tensor* ln_q_w = nullptr; // layer_norm_q
    ggml_tensor* ln_q_b = nullptr;
    ggml_tensor* ln_k_w = nullptr; // layer_norm_k
    ggml_tensor* ln_k_b = nullptr;
    ggml_tensor* ln_v_w = nullptr; // layer_norm_v
    ggml_tensor* ln_v_b = nullptr;
    ggml_tensor* w_qs = nullptr; // [d_model, d_model]
    ggml_tensor* w_ks = nullptr;
    ggml_tensor* w_vs = nullptr;
    ggml_tensor* fc_w = nullptr;       // output projection
    ggml_tensor* lin_pos = nullptr;    // linear_pos [d_model, d_model]
    ggml_tensor* pos_bias_u = nullptr; // [n_head, head_dim]
    ggml_tensor* pos_bias_v = nullptr;
};

struct firered_enc_conv {
    // Conformer conv module
    ggml_tensor* pre_ln_w = nullptr;
    ggml_tensor* pre_ln_b = nullptr;
    ggml_tensor* pw1_w = nullptr; // pointwise_conv1 [2*d_model, d_model, 1]
    ggml_tensor* dw_w = nullptr;  // depthwise [2*d_model, 1, kernel_size]
    ggml_tensor* bn_w = nullptr;  // batch_norm weight (gamma)
    ggml_tensor* bn_b = nullptr;  // batch_norm bias (beta)
    // BatchNorm running stats would be needed at inference if not in eval mode,
    // but PyTorch .eval() uses running_mean/running_var which should be in the checkpoint
    ggml_tensor* bn_mean = nullptr;
    ggml_tensor* bn_var = nullptr;
    ggml_tensor* pw2_w = nullptr; // pointwise_conv2 [d_model, 2*d_model, 1]
};

struct firered_enc_block {
    firered_enc_ffn ffn1;
    firered_enc_mhsa mhsa;
    firered_enc_conv conv;
    firered_enc_ffn ffn2;
    ggml_tensor* ln_w = nullptr; // final layer_norm
    ggml_tensor* ln_b = nullptr;
};

// --- Decoder ---

struct firered_dec_attn {
    ggml_tensor* w_qs = nullptr;
    ggml_tensor* w_qs_b = nullptr;
    ggml_tensor* w_ks = nullptr;
    ggml_tensor* w_vs = nullptr;
    ggml_tensor* w_vs_b = nullptr;
    ggml_tensor* fc_w = nullptr;
    ggml_tensor* fc_b = nullptr;
};

struct firered_dec_block {
    // Self-attention
    ggml_tensor* sattn_norm_w = nullptr;
    ggml_tensor* sattn_norm_b = nullptr;
    firered_dec_attn sattn;
    // Cross-attention
    ggml_tensor* xattn_norm_w = nullptr;
    ggml_tensor* xattn_norm_b = nullptr;
    firered_dec_attn xattn;
    // MLP
    ggml_tensor* mlp_norm_w = nullptr;
    ggml_tensor* mlp_norm_b = nullptr;
    ggml_tensor* mlp_w1 = nullptr; // [d_inner, d_model]
    ggml_tensor* mlp_b1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr; // [d_model, d_inner]
    ggml_tensor* mlp_b2 = nullptr;
};

struct firered_model {
    firered_hparams hp;

    // Encoder
    struct {
        // Input preprocessor: 2x Conv2d(3x3, stride 2) + Linear
        ggml_tensor* conv0_w = nullptr; // [32, 1, 3, 3]
        ggml_tensor* conv0_b = nullptr;
        ggml_tensor* conv1_w = nullptr; // [32, 32, 3, 3]
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* proj_w = nullptr; // [d_model, 608]
        ggml_tensor* proj_b = nullptr;
        // Relative positional encoding
        ggml_tensor* pe = nullptr; // [1, 9999, d_model]
        // Conformer blocks
        std::vector<firered_enc_block> blocks;
    } enc;

    // Decoder
    struct {
        ggml_tensor* emb_w = nullptr; // [odim, d_model]
        ggml_tensor* pe = nullptr;    // [1, pe_maxlen, d_model]
        ggml_tensor* norm_out_w = nullptr;
        ggml_tensor* norm_out_b = nullptr;
        ggml_tensor* prj_w = nullptr; // [odim, d_model] — output projection
        std::vector<firered_dec_block> blocks;
    } dec;

    // CTC
    ggml_tensor* ctc_w = nullptr; // [odim, d_model]
    ggml_tensor* ctc_b = nullptr;

    // Tokenizer
    std::vector<std::string> vocab;

    // Weight memory
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct firered_asr_context {
    firered_asr_context_params params;
    firered_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for decoder
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_self_k = nullptr;
    ggml_tensor* kv_self_v = nullptr;
    ggml_tensor* kv_cross_k = nullptr;
    ggml_tensor* kv_cross_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Implementation
// ===========================================================================

extern "C" struct firered_asr_context_params firered_asr_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1};
}

// --- Tensor loading helpers ---

static void load_ffn(const std::map<std::string, ggml_tensor*>& ts, const char* prefix, firered_enc_ffn& ffn) {
    char buf[128];
    auto get = [&](const char* suffix) -> ggml_tensor* {
        snprintf(buf, sizeof(buf), "%s%s", prefix, suffix);
        auto it = ts.find(buf);
        return it != ts.end() ? it->second : nullptr;
    };
    ffn.ln_w = get(".net.0.weight");
    ffn.ln_b = get(".net.0.bias");
    ffn.up_w = get(".net.1.weight");
    ffn.up_b = get(".net.1.bias");
    ffn.down_w = get(".net.4.weight");
    ffn.down_b = get(".net.4.bias");
}

static void load_enc_mhsa(const std::map<std::string, ggml_tensor*>& ts, const char* prefix, firered_enc_mhsa& mhsa) {
    char buf[128];
    auto get = [&](const char* suffix) -> ggml_tensor* {
        snprintf(buf, sizeof(buf), "%s%s", prefix, suffix);
        auto it = ts.find(buf);
        return it != ts.end() ? it->second : nullptr;
    };
    mhsa.ln_q_w = get(".ln_q.weight");
    mhsa.ln_q_b = get(".ln_q.bias");
    mhsa.ln_k_w = get(".ln_k.weight");
    mhsa.ln_k_b = get(".ln_k.bias");
    mhsa.ln_v_w = get(".ln_v.weight");
    mhsa.ln_v_b = get(".ln_v.bias");
    mhsa.w_qs = get(".w_qs.weight");
    mhsa.w_ks = get(".w_ks.weight");
    mhsa.w_vs = get(".w_vs.weight");
    mhsa.fc_w = get(".fc.weight");
    mhsa.lin_pos = get(".lin_pos.weight");
    mhsa.pos_bias_u = get(".pos_bias_u");
    mhsa.pos_bias_v = get(".pos_bias_v");
}

// ===========================================================================
// Model loading
// ===========================================================================

extern "C" struct firered_asr_context* firered_asr_init_from_file(const char* path_model,
                                                                  struct firered_asr_context_params params) {
    auto* ctx = new firered_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    auto& m = ctx->model;
    auto& hp = m.hp;

    // ---- pass 1: read hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "firered_asr: failed to open '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
        hp.d_model = core_gguf::kv_u32(gctx, "firered.d_model", hp.d_model);
        hp.n_head = core_gguf::kv_u32(gctx, "firered.n_head", hp.n_head);
        hp.d_inner = core_gguf::kv_u32(gctx, "firered.d_inner", hp.d_inner);
        hp.n_layers_enc = core_gguf::kv_u32(gctx, "firered.n_layers_enc", hp.n_layers_enc);
        hp.n_layers_dec = core_gguf::kv_u32(gctx, "firered.n_layers_dec", hp.n_layers_dec);
        hp.idim = core_gguf::kv_u32(gctx, "firered.idim", hp.idim);
        hp.odim = core_gguf::kv_u32(gctx, "firered.odim", hp.odim);
        hp.subsample = core_gguf::kv_u32(gctx, "firered.subsample", hp.subsample);
        hp.kernel_size = core_gguf::kv_u32(gctx, "firered.kernel_size", hp.kernel_size);
        hp.pe_maxlen = core_gguf::kv_u32(gctx, "firered.pe_maxlen", hp.pe_maxlen);
        hp.sos_id = core_gguf::kv_u32(gctx, "firered.sos_id", hp.sos_id);
        hp.eos_id = core_gguf::kv_u32(gctx, "firered.eos_id", hp.eos_id);
        hp.blank_id = core_gguf::kv_u32(gctx, "firered.blank_id", hp.blank_id);
        hp.pad_id = core_gguf::kv_u32(gctx, "firered.pad_id", hp.pad_id);
        hp.head_dim = hp.d_model / hp.n_head;

        // Tokenizer
        m.vocab.resize(hp.odim);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < hp.odim; i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "firered_asr", wl)) {
        fprintf(stderr, "firered_asr: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ts.find(name);
        if (it == ts.end()) {
            if (params.verbosity >= 2)
                fprintf(stderr, "firered_asr: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };

    // --- Encoder input preprocessor ---
    m.enc.conv0_w = get("enc.preproc.conv.0.weight");
    m.enc.conv0_b = get("enc.preproc.conv.0.bias");
    m.enc.conv1_w = get("enc.preproc.conv.2.weight");
    m.enc.conv1_b = get("enc.preproc.conv.2.bias");
    m.enc.proj_w = get("enc.preproc.out.weight");
    m.enc.proj_b = get("enc.preproc.out.bias");
    m.enc.pe = get("enc.pe.pe");

    // --- Encoder Conformer blocks ---
    m.enc.blocks.resize(hp.n_layers_enc);
    for (int i = 0; i < hp.n_layers_enc; i++) {
        auto& b = m.enc.blocks[i];
        char prefix[64];

        // FFN1 (macaron)
        snprintf(prefix, sizeof(prefix), "enc.%d.ffn1", i);
        load_ffn(ts, prefix, b.ffn1);

        // MHSA
        snprintf(prefix, sizeof(prefix), "enc.%d.mhsa", i);
        load_enc_mhsa(ts, prefix, b.mhsa);

        // Conv module
        char buf[128];
        snprintf(buf, sizeof(buf), "enc.%d.conv.pre_ln.weight", i);
        b.conv.pre_ln_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.pre_ln.bias", i);
        b.conv.pre_ln_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.pw1.weight", i);
        b.conv.pw1_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.dw.weight", i);
        b.conv.dw_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.weight", i);
        b.conv.bn_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.bias", i);
        b.conv.bn_b = get(buf);
        // BatchNorm running_mean and running_var
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.running_mean", i);
        b.conv.bn_mean = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.running_var", i);
        b.conv.bn_var = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.pw2.weight", i);
        b.conv.pw2_w = get(buf);

        // FFN2 (macaron)
        snprintf(prefix, sizeof(prefix), "enc.%d.ffn2", i);
        load_ffn(ts, prefix, b.ffn2);

        // Final layer norm
        snprintf(buf, sizeof(buf), "enc.%d.ln.weight", i);
        b.ln_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.ln.bias", i);
        b.ln_b = get(buf);
    }

    // --- Decoder ---
    m.dec.emb_w = get("dec.emb.weight");
    m.dec.pe = get("dec.pe.pe");
    m.dec.norm_out_w = get("dec.norm_out.weight");
    m.dec.norm_out_b = get("dec.norm_out.bias");
    m.dec.prj_w = get("dec.prj.weight");

    m.dec.blocks.resize(hp.n_layers_dec);
    for (int i = 0; i < hp.n_layers_dec; i++) {
        auto& b = m.dec.blocks[i];
        char buf[128];

        // Self-attention
        snprintf(buf, sizeof(buf), "dec.%d.sattn_norm.weight", i);
        b.sattn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn_norm.bias", i);
        b.sattn_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_qs.weight", i);
        b.sattn.w_qs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_qs.bias", i);
        b.sattn.w_qs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_ks.weight", i);
        b.sattn.w_ks = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_vs.weight", i);
        b.sattn.w_vs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_vs.bias", i);
        b.sattn.w_vs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.fc.weight", i);
        b.sattn.fc_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.fc.bias", i);
        b.sattn.fc_b = get(buf);

        // Cross-attention
        snprintf(buf, sizeof(buf), "dec.%d.xattn_norm.weight", i);
        b.xattn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn_norm.bias", i);
        b.xattn_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_qs.weight", i);
        b.xattn.w_qs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_qs.bias", i);
        b.xattn.w_qs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_ks.weight", i);
        b.xattn.w_ks = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_vs.weight", i);
        b.xattn.w_vs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_vs.bias", i);
        b.xattn.w_vs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.fc.weight", i);
        b.xattn.fc_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.fc.bias", i);
        b.xattn.fc_b = get(buf);

        // MLP
        snprintf(buf, sizeof(buf), "dec.%d.mlp_norm.weight", i);
        b.mlp_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp_norm.bias", i);
        b.mlp_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_1.weight", i);
        b.mlp_w1 = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_1.bias", i);
        b.mlp_b1 = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_2.weight", i);
        b.mlp_w2 = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_2.bias", i);
        b.mlp_b2 = get(buf);
    }

    // CTC
    m.ctc_w = get("ctc.weight");
    m.ctc_b = get("ctc.bias");

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "firered_asr: loaded %d enc + %d dec layers, vocab %d, d_model %d\n", hp.n_layers_enc,
                hp.n_layers_dec, hp.odim, hp.d_model);
    }

    return ctx;
}

extern "C" void firered_asr_free(struct firered_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Forward pass (stub — to be implemented with diff-testing)
// ===========================================================================

extern "C" char* firered_asr_transcribe(struct firered_asr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    // TODO: implement full forward pass
    // 1. Compute 80-dim log-mel spectrogram
    // 2. Conv2d subsampling (4x temporal reduction)
    // 3. 16-layer Conformer encoder
    // 4. Autoregressive Transformer decoder (beam search)
    // 5. Token decoding

    fprintf(stderr, "firered_asr: model loaded successfully, forward pass not yet implemented\n");
    fprintf(stderr, "firered_asr: %d encoder + %d decoder layers, %d-dim, %d vocab\n", ctx->model.hp.n_layers_enc,
            ctx->model.hp.n_layers_dec, ctx->model.hp.d_model, ctx->model.hp.odim);
    return nullptr;
}

extern "C" const char* firered_asr_token_text(struct firered_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->model.vocab.size())
        return nullptr;
    return ctx->model.vocab[id].c_str();
}
