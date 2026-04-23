// omniasr.cpp — Facebook OmniASR runtime (CTC + LLM variants).
//
// CTC:  CNN frontend → Transformer encoder → CTC head.
// LLM:  CNN frontend → Transformer encoder → enc_proj → LLaMA decoder.
// Input: raw 16kHz PCM (no mel features needed).

#include "omniasr.h"
#include "core/attention.h"
#include "core/ffn.h"
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
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Model
// ===========================================================================

struct omniasr_hparams {
    // Encoder
    int d_model = 1024;
    int d_ffn = 4096;
    int n_heads = 16;
    int n_enc = 24;
    int n_cnn = 7;
    int vocab_size = 9812;
    int bos_id = 0;
    int eos_id = 2;
    int pad_id = 1;
    int unk_id = 3;
    int head_dim = 64;
    // Decoder (LLM only)
    int model_type = 0; // 0=CTC, 1=LLM
    int d_dec = 4096;
    int d_ffn_dec = 2816;
    int n_heads_dec = 8;
    int n_dec = 12;
    int head_dim_dec = 512;
    int n_langs = 0;
};

struct omniasr_model {
    omniasr_hparams hp;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<std::string> vocab;
    std::vector<int> cnn_strides;
};

// Decoder layer weight pointers for ggml graph building
struct omniasr_dec_block {
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* q_w = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* gate_w = nullptr;
    ggml_tensor* up_w = nullptr;
    ggml_tensor* down_w = nullptr;
};

struct omniasr_context {
    omniasr_model model;
    omniasr_context_params params;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;
    // LLM decoder
    std::vector<omniasr_dec_block> dec_blocks;
    ggml_tensor* dec_ln_w = nullptr;
    ggml_tensor* lm_head_w = nullptr;
    ggml_tensor* enc_proj_w = nullptr;
    ggml_tensor* enc_proj_b = nullptr;
    ggml_tensor* tok_emb_w = nullptr;
    ggml_tensor* lang_emb_w = nullptr;
    // KV cache for LLM decoder
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // [head_dim, max_ctx, n_heads, n_layers]
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;
    // Compute buffer for ggml graph building
    std::vector<uint8_t> compute_meta;
};

struct omniasr_perf {
    int64_t t_total_us = 0;
    int64_t t_norm_us = 0;
    int64_t t_enc_alloc_us = 0;
    int64_t t_enc_compute_us = 0;
    int64_t t_enc_read_us = 0;
    int64_t t_prefill_alloc_us = 0;
    int64_t t_prefill_compute_us = 0;
    int64_t t_decode_alloc_us = 0;
    int64_t t_decode_compute_us = 0;
    int64_t t_decode_logits_us = 0;
    int n_dec_steps = 0;
    int enc_nodes = 0;
    int prefill_nodes = 0;
    int decode_nodes = 0;
};

static void omniasr_perf_print(const omniasr_perf& p, int n_samples, int verbosity) {
    const char* bench = getenv("OMNIASR_BENCH");
    if (verbosity < 2 && (!bench || !bench[0]))
        return;
    const double audio_s = n_samples / 16000.0;
    fprintf(stderr, "omniasr: =========== performance report ===========\n");
    fprintf(stderr, "omniasr:  audio          %7.2f s\n", audio_s);
    fprintf(stderr, "omniasr:  normalize      %7.1f ms\n", p.t_norm_us / 1e3);
    fprintf(stderr, "omniasr:  enc alloc      %7.1f ms  nodes=%d\n", p.t_enc_alloc_us / 1e3, p.enc_nodes);
    fprintf(stderr, "omniasr:  enc compute    %7.1f ms\n", p.t_enc_compute_us / 1e3);
    fprintf(stderr, "omniasr:  enc read       %7.1f ms\n", p.t_enc_read_us / 1e3);
    fprintf(stderr, "omniasr:  prefill alloc  %7.1f ms  nodes=%d\n", p.t_prefill_alloc_us / 1e3, p.prefill_nodes);
    fprintf(stderr, "omniasr:  prefill compute%7.1f ms\n", p.t_prefill_compute_us / 1e3);
    fprintf(stderr, "omniasr:  decode alloc   %7.1f ms  steps=%d nodes=%d\n", p.t_decode_alloc_us / 1e3, p.n_dec_steps,
            p.decode_nodes);
    fprintf(stderr, "omniasr:  decode compute %7.1f ms\n", p.t_decode_compute_us / 1e3);
    fprintf(stderr, "omniasr:  decode logits  %7.1f ms\n", p.t_decode_logits_us / 1e3);
    fprintf(stderr, "omniasr:  total measured %7.1f ms\n", p.t_total_us / 1e3);
    if (p.t_total_us > 0)
        fprintf(stderr, "omniasr:  realtime       %7.2fx\n", audio_s / (p.t_total_us / 1e6));
}

// ===========================================================================
// Defaults
// ===========================================================================

extern "C" struct omniasr_context_params omniasr_context_default_params(void) {
    omniasr_context_params p;
    p.n_threads = 4;
    p.max_new_tokens = 512;
    p.verbosity = 1;
    p.language = nullptr;
    p.use_gpu = true;
    return p;
}

// ===========================================================================
// Debug: dump ggml tensor to binary file for comparison with Python reference
// ===========================================================================

static void dump_tensor(ggml_tensor* t, const char* name, const char* dir) {
    if (!dir || !dir[0] || !t)
        return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    int n = (int)ggml_nelements(t);
    std::vector<float> data(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(uint16_t));
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp.data(), data.data(), n);
    }
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(data.data(), sizeof(float), n, f);
        fclose(f);
        fprintf(stderr, "  DUMP: %s [%lld,%lld] → %s\n", name, (long long)t->ne[0], (long long)t->ne[1], path);
    }
}

static void dump_cpu(const float* data, int n, const char* name, const char* dir) {
    if (!dir || !dir[0])
        return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
        fprintf(stderr, "  DUMP: %s [%d] → %s\n", name, n, path);
    }
}

// ===========================================================================
// ggml graph helpers
// ===========================================================================

// LayerNorm: (x - mean) / sqrt(var + eps) * w + b
// x: [C, T] col-major (ne[0]=C). w,b: [C].
static ggml_tensor* build_ln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    if (b)
        x = ggml_add(ctx, x, b);
    return x;
}

// OmniASR/Wav2Vec2 positional convolution: grouped Conv1d + GELU + residual.
// ggml has no grouped conv op, so build it as one conv branch per group.
// h: [C, T], w: [K, C_per_group, C], b: [C].
static ggml_tensor* build_grouped_pos_conv(ggml_context* ctx, ggml_tensor* h, ggml_tensor* w, ggml_tensor* b) {
    const int64_t K = w->ne[0];
    const int64_t C_per_group = w->ne[1];
    const int64_t C = w->ne[2];
    const int64_t T = h->ne[1];
    const int64_t groups = C / C_per_group;

    GGML_ASSERT(h->ne[0] == C);
    GGML_ASSERT(groups > 0 && groups * C_per_group == C);

    ggml_tensor* pos = nullptr;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t c0 = g * C_per_group;

        ggml_tensor* h_g = ggml_view_2d(ctx, h, C_per_group, T, h->nb[1], (size_t)c0 * h->nb[0]);
        h_g = ggml_cont(ctx, ggml_transpose(ctx, h_g)); // [T, C_per_group]

        ggml_tensor* w_g = ggml_view_3d(ctx, w, K, C_per_group, C_per_group, w->nb[1], w->nb[2], (size_t)c0 * w->nb[2]);
        w_g = ggml_cont(ctx, w_g);

        ggml_tensor* y = ggml_conv_1d(ctx, w_g, h_g, 1, (int)(K / 2), 1); // [T + 1, C_per_group] for even K
        y = ggml_view_2d(ctx, y, T, C_per_group, y->nb[1], 0);            // fairseq trims the final frame
        y = ggml_cont(ctx, ggml_transpose(ctx, y));                       // [C_per_group, T]

        ggml_tensor* b_g = ggml_view_1d(ctx, b, C_per_group, (size_t)c0 * b->nb[0]);
        y = ggml_add(ctx, y, b_g);
        pos = pos ? ggml_concat(ctx, pos, y, 0) : y;
    }

    pos = ggml_gelu(ctx, pos);
    return ggml_add(ctx, h, pos);
}

// Transformer encoder layer (pre-norm)
// x: [d_model, T] col-major
static ggml_tensor* build_enc_layer(ggml_context* ctx, ggml_tensor* x, ggml_tensor* attn_ln_w, ggml_tensor* attn_ln_b,
                                    ggml_tensor* q_w, ggml_tensor* q_b, ggml_tensor* k_w, ggml_tensor* k_b,
                                    ggml_tensor* v_w, ggml_tensor* v_b, ggml_tensor* o_w, ggml_tensor* o_b,
                                    ggml_tensor* ffn_ln_w, ggml_tensor* ffn_ln_b, ggml_tensor* up_w, ggml_tensor* up_b,
                                    ggml_tensor* down_w, ggml_tensor* down_b, int n_heads, int head_dim) {
    int d = (int)x->ne[0];
    int T = (int)x->ne[1];

    // Self-attention with pre-norm
    ggml_tensor* residual = x;
    ggml_tensor* h = build_ln(ctx, x, attn_ln_w, attn_ln_b);

    // Q, K, V projections
    ggml_tensor* Q = ggml_mul_mat(ctx, q_w, h);
    if (q_b)
        Q = ggml_add(ctx, Q, q_b);
    ggml_tensor* K = ggml_mul_mat(ctx, k_w, h);
    if (k_b)
        K = ggml_add(ctx, K, k_b);
    ggml_tensor* V = ggml_mul_mat(ctx, v_w, h);
    if (v_b)
        V = ggml_add(ctx, V, v_b);

    // Reshape for multi-head: [d, T] → [head_dim, n_heads, T]
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, T);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, T);

    // Permute for flash_attn: [head_dim, T, n_heads]
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);

    // Reshape back: [head_dim, T, n_heads] → [d, T]
    attn = ggml_reshape_2d(ctx, attn, d, T);

    // Output projection
    attn = ggml_mul_mat(ctx, o_w, attn);
    if (o_b)
        attn = ggml_add(ctx, attn, o_b);

    // Residual
    x = ggml_add(ctx, residual, attn);

    // FFN with pre-norm
    residual = x;
    h = build_ln(ctx, x, ffn_ln_w, ffn_ln_b);
    h = ggml_mul_mat(ctx, up_w, h);
    if (up_b)
        h = ggml_add(ctx, h, up_b);
    h = ggml_gelu(ctx, h);
    h = ggml_mul_mat(ctx, down_w, h);
    if (down_b)
        h = ggml_add(ctx, h, down_b);

    return ggml_add(ctx, residual, h);
}

// ===========================================================================
// Init
// ===========================================================================

extern "C" struct omniasr_context* omniasr_init_from_file(const char* path_model,
                                                          struct omniasr_context_params params) {
    auto* ctx = new omniasr_context();
    ctx->params = params;
    auto& m = ctx->model;
    auto& hp = m.hp;

    // Read metadata
    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    hp.d_model = core_gguf::kv_u32(gctx, "omniasr.d_model", 1024);
    hp.d_ffn = core_gguf::kv_u32(gctx, "omniasr.d_ffn", 4096);
    hp.n_heads = core_gguf::kv_u32(gctx, "omniasr.n_heads", 16);
    hp.n_enc = core_gguf::kv_u32(gctx, "omniasr.n_enc_layers", 24);
    hp.n_cnn = core_gguf::kv_u32(gctx, "omniasr.n_cnn_layers", 7);
    hp.vocab_size = core_gguf::kv_u32(gctx, "omniasr.vocab_size", 9812);
    hp.bos_id = core_gguf::kv_u32(gctx, "omniasr.bos_id", 0);
    hp.eos_id = core_gguf::kv_u32(gctx, "omniasr.eos_id", 2);
    hp.pad_id = core_gguf::kv_u32(gctx, "omniasr.pad_id", 1);
    hp.unk_id = core_gguf::kv_u32(gctx, "omniasr.unk_id", 3);
    hp.head_dim = hp.d_model / hp.n_heads;
    hp.model_type = core_gguf::kv_u32(gctx, "omniasr.model_type", 0);
    hp.d_dec = core_gguf::kv_u32(gctx, "omniasr.d_dec", 4096);
    hp.d_ffn_dec = core_gguf::kv_u32(gctx, "omniasr.d_ffn_dec", 2816);
    hp.n_heads_dec = core_gguf::kv_u32(gctx, "omniasr.n_heads_dec", 8);
    hp.n_dec = core_gguf::kv_u32(gctx, "omniasr.n_dec_layers", 12);
    hp.head_dim_dec = core_gguf::kv_u32(gctx, "omniasr.head_dim_dec", 512);
    hp.n_langs = core_gguf::kv_u32(gctx, "omniasr.n_langs", 0);

    // CNN strides
    int stride_key = gguf_find_key(gctx, "omniasr.cnn_strides");
    if (stride_key >= 0) {
        int n = gguf_get_arr_n(gctx, stride_key);
        m.cnn_strides.resize(n);
        for (int i = 0; i < n; i++)
            m.cnn_strides[i] = ((const int32_t*)gguf_get_arr_data(gctx, stride_key))[i];
    } else {
        m.cnn_strides = {5, 2, 2, 2, 2, 2, 2};
    }

    // Vocab
    const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        m.vocab.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s)
                m.vocab[i] = s;
        }
    }
    gguf_free(gctx);

    if (params.verbosity >= 1) {
        const char* type_str = hp.model_type == 1 ? "LLM" : "CTC";
        fprintf(stderr, "omniasr-%s: enc=%dL d=%d, ", type_str, hp.n_enc, hp.d_model);
        if (hp.model_type == 1)
            fprintf(stderr, "dec=%dL d=%d ffn=%d heads=%d, ", hp.n_dec, hp.d_dec, hp.d_ffn_dec, hp.n_heads_dec);
        fprintf(stderr, "cnn=%d, vocab=%d\n", hp.n_cnn, hp.vocab_size);
    }

    // Load weights
    if (params.use_gpu) {
        ctx->backend = ggml_backend_init_best();
    }
    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
    }
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        if (ctx->backend) {
            ggml_backend_free(ctx->backend);
        }
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads > 0 ? params.n_threads : 4);
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, params.n_threads > 0 ? params.n_threads : 4);
    }

    core_gguf::WeightLoad wl;
    const char* arch = hp.model_type == 1 ? "omniasr-llm" : "omniasr-ctc";
    if (!core_gguf::load_weights(path_model, ctx->backend, arch, wl)) {
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            ggml_backend_free(ctx->backend_cpu);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    m.tensors = wl.tensors;

    // Backend scheduler: ggml requires the last backend to be CPU when a GPU
    // backend is present so host-side fallbacks have somewhere to land.
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    int n_backends = 1;
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_backends++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_backends, 65536, false, false);

    // LLM decoder: populate block pointers + allocate compute buffer
    if (hp.model_type == 1) {
        auto G = [&](const std::string& name) -> ggml_tensor* {
            auto it = m.tensors.find(name);
            return it != m.tensors.end() ? it->second : nullptr;
        };
        ctx->dec_blocks.resize(hp.n_dec);
        for (int i = 0; i < hp.n_dec; i++) {
            std::string p = "dec." + std::to_string(i);
            auto& b = ctx->dec_blocks[i];
            b.attn_ln_w = G(p + ".attn_ln.weight");
            b.q_w = G(p + ".attn.q_proj.weight");
            b.k_w = G(p + ".attn.k_proj.weight");
            b.v_w = G(p + ".attn.v_proj.weight");
            b.o_w = G(p + ".attn.out.weight");
            b.ffn_ln_w = G(p + ".ffn_ln.weight");
            b.gate_w = G(p + ".ffn.gate.weight");
            b.up_w = G(p + ".ffn.up.weight");
            b.down_w = G(p + ".ffn.down.weight");
        }
        ctx->dec_ln_w = G("dec_ln.weight");
        ctx->lm_head_w = G("lm_head.weight");
        ctx->enc_proj_w = G("enc_proj.weight");
        ctx->enc_proj_b = G("enc_proj.bias");
        ctx->tok_emb_w = G("tok_emb.weight");
        ctx->lang_emb_w = G("lang_emb.weight");
        // Compute meta for graph building (generous size for 12-layer decoder)
        ctx->compute_meta.resize(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(32768, false));
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "omniasr: loaded %zu tensors, %zu vocab\n", m.tensors.size(), m.vocab.size());
    }

    return ctx;
}

extern "C" void omniasr_free(struct omniasr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->weight_ctx)
        ggml_free(ctx->weight_ctx);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// LLM KV cache
// ===========================================================================

static bool omniasr_alloc_kv_cache(omniasr_context* ctx, int max_ctx) {
    auto& hp = ctx->model.hp;
    int hd = hp.head_dim_dec;
    int nh = hp.n_heads_dec;
    int nl = hp.n_dec;

    // Create context for KV tensors
    size_t mem = 2 * ggml_tensor_overhead() + 64;
    struct ggml_init_params kv_params = {mem, nullptr, true};
    ctx->kv_ctx = ggml_init(kv_params);

    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    size_t kbytes = ggml_nbytes(ctx->kv_k);
    size_t vbytes = ggml_nbytes(ctx->kv_v);
    ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, kbytes + vbytes + 64);
    if (!ctx->kv_buf) {
        fprintf(stderr, "omniasr-llm: failed to allocate KV cache (%zu MB)\n", (kbytes + vbytes) / (1024 * 1024));
        return false;
    }

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);

    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr-llm: KV cache: %d ctx, %zu MB\n", max_ctx, (kbytes + vbytes) / (1024 * 1024));
    return true;
}

// (ggml decoder graph builder — TODO: optimize with ggml after CPU version verified)

// ===========================================================================
// LLM transcribe
// ===========================================================================

static char* omniasr_transcribe_llm(omniasr_context* ctx, const std::vector<float>& encoder_out, int d_enc, int T_enc,
                                    omniasr_perf* perf);

// ===========================================================================
// Transcribe
// ===========================================================================

extern "C" char* omniasr_transcribe(struct omniasr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    omniasr_perf perf;
    const int64_t t_total0 = ggml_time_us();

    auto& m = ctx->model;
    auto& hp = m.hp;
    auto& ts = m.tensors;

    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    // Build ggml graph for full forward pass
    size_t mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(65536, false);
    std::vector<uint8_t> meta(mem);
    struct ggml_init_params gp = {mem, meta.data(), true};
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    const char* dump_dir = getenv("OMNIASR_DUMP_DIR");

    // Input normalization: layer_norm(waveform) — zero mean, unit variance
    // This is a wav2vec2 convention, required for OmniASR.
    std::vector<float> pcm_norm(n_samples);
    {
        const int64_t t0 = ggml_time_us();
        double mean = 0;
        for (int i = 0; i < n_samples; i++)
            mean += samples[i];
        mean /= n_samples;
        double var = 0;
        for (int i = 0; i < n_samples; i++)
            var += (samples[i] - mean) * (samples[i] - mean);
        var /= n_samples;
        float inv_std = 1.0f / (sqrtf((float)var + 1e-5f));
        for (int i = 0; i < n_samples; i++)
            pcm_norm[i] = (samples[i] - (float)mean) * inv_std;
        perf.t_norm_us += ggml_time_us() - t0;
    }
    dump_cpu(pcm_norm.data(), n_samples, "pcm_norm", dump_dir);

    // Input: normalized PCM [n_samples, 1]
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(inp, "pcm");
    ggml_set_input(inp);

    // CNN Feature Extractor: 7 layers of Conv1d + LayerNorm + GELU
    ggml_tensor* h = inp;
    for (int i = 0; i < hp.n_cnn; i++) {
        std::string prefix = "cnn." + std::to_string(i);
        ggml_tensor* conv_w = G(prefix + ".conv.weight");
        ggml_tensor* conv_b = G(prefix + ".conv.bias");
        ggml_tensor* ln_w = G(prefix + ".ln.weight");
        ggml_tensor* ln_b = G(prefix + ".ln.bias");

        int stride = (i < (int)m.cnn_strides.size()) ? m.cnn_strides[i] : 2;

        // Conv1d: no padding (wav2vec2 convention)
        h = ggml_conv_1d(ctx0, conv_w, h, stride, 0, 1);
        if (conv_b) {
            // Bias: output is [T, C]. Transpose to [C, T], add bias [C], transpose back.
            ggml_tensor* ht = ggml_cont(ctx0, ggml_transpose(ctx0, h));
            ht = ggml_add(ctx0, ht, conv_b);
            h = ggml_cont(ctx0, ggml_transpose(ctx0, ht));
        }

        // LayerNorm: output [T, C]. ggml_norm operates on ne[0].
        // After conv1d: ne[0]=T, ne[1]=C. ggml_norm normalizes over ne[0]=T.
        // But we need to normalize over C (channels).
        // Transpose to [C, T], norm, transpose back.
        h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [C, T]
        h = build_ln(ctx0, h, ln_w, ln_b);
        // Keep in [C, T] format — the GELU and next conv need [T, C] though
        // Actually, ggml_norm normalizes over ne[0]. For [C, T]: normalizes over C. That's correct!
        // So h is [C, T] with LN over C ✓

        h = ggml_gelu(ctx0, h);

        // Next conv expects [T, C] input. Transpose back.
        h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [T, C]

        // Debug: mark last CNN layer output
        if (i == hp.n_cnn - 1) {
            ggml_set_name(h, "cnn_out");
            ggml_set_output(h);
        }
    }

    // h is [T_cnn, 512] after CNN. Transpose to [512, T] for LN + projection.
    h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [512, T]

    // Post-extract LayerNorm (on 512-dim CNN output)
    // Try both shortened and long names (LLM converter shortens, CTC keeps long)
    ggml_tensor* pe_ln_w = G("post_extract_ln.weight");
    ggml_tensor* pe_ln_b = G("post_extract_ln.bias");
    if (!pe_ln_w) {
        pe_ln_w = G("encoder_frontend.post_extract_ln.weight");
        pe_ln_b = G("encoder_frontend.post_extract_ln.bias");
    }
    if (pe_ln_w)
        h = build_ln(ctx0, h, pe_ln_w, pe_ln_b);

    // Pre-lookup CTC tensors (needed in both single-graph and two-graph paths)
    ggml_tensor* ctc_w = G("ctc.weight");
    ggml_tensor* ctc_b = G("ctc.bias");

    // Linear projection: 512 → d_model
    ggml_tensor* proj_w = G("proj.weight");
    ggml_tensor* proj_b = G("proj.bias");
    h = ggml_mul_mat(ctx0, proj_w, h); // [d_model, T]
    if (proj_b)
        h = ggml_add(ctx0, h, proj_b);

    ggml_set_name(h, "proj_out");
    ggml_set_output(h);

    // Convolutional positional encoding: grouped Conv1d + GELU + residual.
    // Weight normalization pre-computed in converter → stored as pos_conv.weight.
    // Groups=16, kernel=128, channels=1024.
    {
        ggml_tensor* wv_t = G("pos_conv.weight");
        ggml_tensor* pb_t = G("pos_conv.bias");
        if (wv_t && pb_t)
            h = build_grouped_pos_conv(ctx0, h, wv_t, pb_t);
    }

    // h is now [d_model, T] — correct format for transformer layers

    // Transformer encoder layers
    for (int i = 0; i < hp.n_enc; i++) {
        std::string p = "enc." + std::to_string(i);
        h = build_enc_layer(ctx0, h, G(p + ".attn_ln.weight"), G(p + ".attn_ln.bias"), G(p + ".attn.q_proj.weight"),
                            G(p + ".attn.q_proj.bias"), G(p + ".attn.k_proj.weight"), G(p + ".attn.k_proj.bias"),
                            G(p + ".attn.v_proj.weight"), G(p + ".attn.v_proj.bias"), G(p + ".attn.out.weight"),
                            G(p + ".attn.out.bias"), G(p + ".ffn_ln.weight"), G(p + ".ffn_ln.bias"),
                            G(p + ".ffn.up.weight"), G(p + ".ffn.up.bias"), G(p + ".ffn.down.weight"),
                            G(p + ".ffn.down.bias"), hp.n_heads, hp.head_dim);
    }

    // Final LayerNorm
    h = build_ln(ctx0, h, G("enc_ln.weight"), G("enc_ln.bias"));

    if (hp.model_type == 1) {
        ggml_set_name(h, "enc_out");
    } else {
        // CTC head: linear projection to vocab
        h = ggml_mul_mat(ctx0, ctc_w, h); // [vocab_size, T]
        if (ctc_b)
            h = ggml_add(ctx0, h, ctc_b);
        ggml_set_name(h, "logits");
    }
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);
    perf.enc_nodes = ggml_graph_n_nodes(gf);

    // Allocate and compute
    ggml_backend_sched_reset(ctx->sched);
    int64_t t0 = ggml_time_us();
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "omniasr: graph alloc failed\n");
        ggml_free(ctx0);
        return nullptr;
    }
    perf.t_enc_alloc_us += ggml_time_us() - t0;

    // Set input
    ggml_backend_tensor_set(inp, pcm_norm.data(), 0, n_samples * sizeof(float));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr: %d samples, computing graph...\n", n_samples);

    t0 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "omniasr: graph compute failed\n");
        ggml_free(ctx0);
        return nullptr;
    }
    perf.t_enc_compute_us += ggml_time_us() - t0;

    if (hp.model_type == 1) {
        ggml_tensor* enc_out_t = ggml_graph_get_tensor(gf, "enc_out");
        int d_e = (int)enc_out_t->ne[0];
        int T_e = (int)enc_out_t->ne[1];
        std::vector<float> enc_out_data((size_t)d_e * T_e);
        t0 = ggml_time_us();
        ggml_backend_tensor_get(enc_out_t, enc_out_data.data(), 0, (size_t)d_e * T_e * sizeof(float));
        perf.t_enc_read_us += ggml_time_us() - t0;
        dump_cpu(enc_out_data.data(), d_e * T_e, "encoder_output", dump_dir);
        ggml_free(ctx0);

        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "omniasr-llm: encoder done [%d, %d], running decoder...\n", d_e, T_e);

        char* out = omniasr_transcribe_llm(ctx, enc_out_data, d_e, T_e, &perf);
        perf.t_total_us = ggml_time_us() - t_total0;
        omniasr_perf_print(perf, n_samples, ctx->params.verbosity);
        return out;
    }

read_logits:
    // Debug: read intermediate outputs
    if (ctx->params.verbosity >= 2) {
        auto dump = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t) {
                fprintf(stderr, "  %s: NOT FOUND\n", name);
                return;
            }
            float buf[10];
            int n = std::min(10, (int)ggml_nelements(t));
            ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
            fprintf(stderr, "  %s: ne=[%lld,%lld], data[:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", name, (long long)t->ne[0],
                    (long long)t->ne[1], buf[0], n > 1 ? buf[1] : 0, n > 2 ? buf[2] : 0, n > 3 ? buf[3] : 0,
                    n > 4 ? buf[4] : 0);
        };
        dump("cnn_out");
        dump("proj_out");
        dump("logits");
        // Also show logit stats at first frame
        ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
        if (lt) {
            int V_dbg = (int)lt->ne[0];
            std::vector<float> frame0(V_dbg);
            ggml_backend_tensor_get(lt, frame0.data(), 0, V_dbg * sizeof(float));
            int best = 0;
            for (int i = 1; i < V_dbg; i++)
                if (frame0[i] > frame0[best])
                    best = i;
            fprintf(stderr, "  logits frame 0: argmax=%d (%.4f), blank(%d)=%.4f\n", best, frame0[best], hp.pad_id,
                    frame0[hp.pad_id]);
        }
    }

    // Read logits
    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    int V = (int)logits_t->ne[0]; // vocab_size
    int T = (int)logits_t->ne[1]; // time steps
    std::vector<float> logits(V * T);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, V * T * sizeof(float));
    ggml_free(ctx0);

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr: logits [%d, %d], CTC decoding...\n", V, T);

    // Greedy CTC decode: argmax per frame, collapse repeats, remove blanks
    // Blank token = pad_id = 1 (SentencePiece convention for OmniASR)
    int blank_id = hp.bos_id; // CTC blank = <s> = 0 (fairseq2 convention)
    std::vector<int> tokens;
    int prev_id = -1;
    for (int t = 0; t < T; t++) {
        // logits layout: [V, T] col-major → logits[t * V + v]
        int best = 0;
        float best_val = logits[t * V];
        for (int v = 1; v < V; v++) {
            if (logits[t * V + v] > best_val) {
                best_val = logits[t * V + v];
                best = v;
            }
        }
        if (best != blank_id && best != prev_id) {
            tokens.push_back(best);
        }
        prev_id = best;
    }

    // Detokenize: SentencePiece convention — ▁ (U+2581) = space
    std::string result;
    for (int tid : tokens) {
        if (tid == hp.bos_id || tid == hp.eos_id || tid == hp.pad_id || tid == hp.unk_id)
            continue;
        if (tid < (int)m.vocab.size()) {
            std::string piece = m.vocab[tid];
            for (size_t i = 0; i < piece.size(); i++) {
                if ((unsigned char)piece[i] == 0xE2 && i + 2 < piece.size() && (unsigned char)piece[i + 1] == 0x96 &&
                    (unsigned char)piece[i + 2] == 0x81) {
                    result += ' ';
                    i += 2;
                } else {
                    result += piece[i];
                }
            }
        }
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr: decoded %d tokens → %zu chars\n", (int)tokens.size(), result.size());

    // Trim
    while (!result.empty() && result.front() == ' ')
        result.erase(result.begin());
    while (!result.empty() && result.back() == ' ')
        result.pop_back();

    if (result.empty())
        return nullptr;

    char* out = (char*)malloc(result.size() + 1);
    if (!out)
        return nullptr;
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    perf.t_total_us = ggml_time_us() - t_total0;
    omniasr_perf_print(perf, n_samples, ctx->params.verbosity);
    return out;
}

// ===========================================================================
// LLM decoder — ggml graph with KV cache (like voxtral4b)
// ===========================================================================

static ggml_tensor* omniasr_build_dec_body(omniasr_context* ctx, ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* cur,
                                           ggml_tensor* positions, ggml_tensor* causal_mask, int n_past, int n_tokens) {
    auto& hp = ctx->model.hp;
    int dd = hp.d_dec;
    int nh = hp.n_heads_dec;
    int hd = hp.head_dim_dec;
    int n_layers = hp.n_dec;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ nh,
        /*n_kv_heads*/ nh, // MHA (same as query heads)
        /*head_dim*/ hd,
        /*n_kv_grp*/ 1, // no GQA
        /*n_ctx_orig*/ 0,
        /*rope_theta*/ 10000.0f,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ 1.0f / sqrtf((float)hd),
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_NATIVE,
        /*rope_type*/ GGML_ROPE_TYPE_NORMAL, // fairseq2 interleaved
    };

    for (int il = 0; il < n_layers; il++) {
        auto& b = ctx->dec_blocks[il];
        ggml_tensor* residual = cur;

        cur = ggml_rms_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, b.attn_ln_w);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, cur, b.q_w, b.k_w, b.v_w, b.o_w, nullptr, nullptr,
                                                    positions, causal_mask, ctx->kv_k, ctx->kv_v, il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, b.ffn_ln_w);
        ggml_tensor* ffn = core_ffn::swiglu(ctx0, cur, b.gate_w, b.up_w, b.down_w);
        cur = ggml_add(ctx0, residual, ffn);
    }

    cur = ggml_rms_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul(ctx0, cur, ctx->dec_ln_w);

    if (n_tokens > 1) {
        cur = ggml_view_1d(ctx0, cur, dd, (size_t)(n_tokens - 1) * dd * sizeof(float));
        cur = ggml_reshape_2d(ctx0, cur, dd, 1);
    }

    return ggml_mul_mat(ctx0, ctx->lm_head_w, cur);
}

// Build decoder graph for n_tokens at position n_past. Prefill uses F32 embeddings;
// single-token decode can use token IDs and backend get_rows to avoid CPU embedding copies.
static ggml_cgraph* omniasr_build_dec_graph(omniasr_context* ctx, int n_past, int n_tokens, bool input_ids_mode) {
    auto& hp = ctx->model.hp;
    int dd = hp.d_dec;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* cur = nullptr;
    if (input_ids_mode) {
        ggml_tensor* input_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_name(input_ids, "input_ids");
        ggml_set_input(input_ids);
        cur = ggml_get_rows(ctx0, ctx->tok_emb_w, input_ids);
    } else {
        ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dd, n_tokens);
        ggml_set_name(embeds, "dec_input");
        ggml_set_input(embeds);
        cur = embeds;
    }

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    cur = omniasr_build_dec_body(ctx, ctx0, gf, cur, positions, causal_mask, n_past, n_tokens);

    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

static bool omniasr_run_dec_token(omniasr_context* ctx, int token_id, int n_past, std::vector<float>& logits,
                                  omniasr_perf* perf) {
    int32_t input_id = token_id;
    int32_t position = n_past;

    ggml_cgraph* gf = omniasr_build_dec_graph(ctx, n_past, 1, true);
    if (perf && perf->decode_nodes == 0)
        perf->decode_nodes = ggml_graph_n_nodes(gf);
    ggml_backend_sched_reset(ctx->sched);
    int64_t t0 = ggml_time_us();
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "omniasr-llm: token decoder graph alloc failed\n");
        return false;
    }
    if (perf)
        perf->t_decode_alloc_us += ggml_time_us() - t0;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), &input_id, 0, sizeof(input_id));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), &position, 0, sizeof(position));

    t0 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "omniasr-llm: token decoder graph compute failed\n");
        return false;
    }
    if (perf) {
        perf->t_decode_compute_us += ggml_time_us() - t0;
        perf->n_dec_steps++;
    }

    ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
    int V = (int)lt->ne[0];
    logits.resize(V);
    t0 = ggml_time_us();
    ggml_backend_tensor_get(lt, logits.data(), 0, V * sizeof(float));
    if (perf)
        perf->t_decode_logits_us += ggml_time_us() - t0;
    return true;
}

static ggml_cgraph* omniasr_build_prefill_graph(omniasr_context* ctx, int d_enc, int T_enc, bool use_lang,
                                                int prefix_len) {
    auto& hp = ctx->model.hp;
    int dd = hp.d_dec;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, T_enc);
    ggml_set_name(enc, "encoder_out");
    ggml_set_input(enc);

    ggml_tensor* cur = ggml_mul_mat(ctx0, ctx->enc_proj_w, enc);
    if (ctx->enc_proj_b)
        cur = ggml_add(ctx0, cur, ctx->enc_proj_b);

    if (use_lang) {
        ggml_tensor* lid_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(lid_id, "lid_id");
        ggml_set_input(lid_id);
        cur = ggml_concat(ctx0, cur, ggml_get_rows(ctx0, ctx->tok_emb_w, lid_id), 1);

        ggml_tensor* lang_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(lang_id, "lang_id");
        ggml_set_input(lang_id);
        cur = ggml_concat(ctx0, cur, ggml_get_rows(ctx0, ctx->lang_emb_w, lang_id), 1);
    }

    ggml_tensor* bos_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(bos_id, "bos_id");
    ggml_set_input(bos_id);
    cur = ggml_concat(ctx0, cur, ggml_get_rows(ctx0, ctx->tok_emb_w, bos_id), 1);
    GGML_ASSERT(cur->ne[0] == dd && cur->ne[1] == prefix_len);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prefix_len);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, prefix_len, prefix_len);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    cur = omniasr_build_dec_body(ctx, ctx0, gf, cur, positions, causal_mask, 0, prefix_len);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

static bool omniasr_run_prefill(omniasr_context* ctx, const std::vector<float>& encoder_out, int d_enc, int T_enc,
                                bool use_lang, int lang_id, int lid_marker_id, int prefix_len,
                                std::vector<float>& logits, omniasr_perf* perf) {
    std::vector<int32_t> positions(prefix_len);
    for (int i = 0; i < prefix_len; i++)
        positions[i] = i;

    std::vector<ggml_fp16_t> mask((size_t)prefix_len * prefix_len, ggml_fp32_to_fp16(0.0f));
    ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < prefix_len; q++)
        for (int k = 0; k < prefix_len; k++)
            if (k > q)
                mask[(size_t)q * prefix_len + k] = neg_inf;

    int32_t lid = lid_marker_id;
    int32_t lang = lang_id;
    int32_t bos = ctx->model.hp.bos_id;

    ggml_cgraph* gf = omniasr_build_prefill_graph(ctx, d_enc, T_enc, use_lang, prefix_len);
    if (perf)
        perf->prefill_nodes = ggml_graph_n_nodes(gf);
    ggml_backend_sched_reset(ctx->sched);
    int64_t t0 = ggml_time_us();
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "omniasr-llm: prefill graph alloc failed\n");
        return false;
    }
    if (perf)
        perf->t_prefill_alloc_us += ggml_time_us() - t0;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "encoder_out"), encoder_out.data(), 0,
                            (size_t)d_enc * T_enc * sizeof(float));
    if (use_lang) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "lid_id"), &lid, 0, sizeof(lid));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "lang_id"), &lang, 0, sizeof(lang));
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "bos_id"), &bos, 0, sizeof(bos));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));

    t0 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "omniasr-llm: prefill graph compute failed\n");
        return false;
    }
    if (perf)
        perf->t_prefill_compute_us += ggml_time_us() - t0;

    ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
    int V = (int)lt->ne[0];
    logits.resize(V);
    t0 = ggml_time_us();
    ggml_backend_tensor_get(lt, logits.data(), 0, V * sizeof(float));
    if (perf)
        perf->t_decode_logits_us += ggml_time_us() - t0;
    return true;
}

static char* omniasr_transcribe_llm(omniasr_context* ctx, const std::vector<float>& encoder_out, int d_enc, int T_enc,
                                    omniasr_perf* perf) {
    auto& m = ctx->model;
    auto& hp = m.hp;

    int dd = hp.d_dec; // 4096

    const int tok_emb_size = ctx->tok_emb_w ? (int)ctx->tok_emb_w->ne[1] : 0;
    int lang_id = 417; // eng_Latn default (parquet_index=416, +1 per factory.py)
    // Factory: lang_mapping = {lang.lower(): parquet_index + 1}
    // Index 0 reserved for no-language/dropout
    // From languges_lookup_table.parquet:
    //   eng_Latn=417, deu_Latn=367, fra_Latn=448, spa_Latn=1355, jpn_Jpan=632, kor_Hang=734
    // TODO: embed parquet mapping in GGUF and parse ctx->params.language

    bool use_lang = (hp.n_langs > 0 && ctx->lang_emb_w);
    // Sequence: [audio_embs...] [lid_marker_emb] [lang_emb] [BOS_emb] [generated...]
    // lid_marker is special token at index vocab_size (9812) in text_frontend
    int lid_marker_id = hp.vocab_size; // 9812 — the extra token in tok_emb (size 9813)
    if (lang_id < 0 || lang_id >= hp.n_langs || lid_marker_id >= tok_emb_size)
        use_lang = false;
    int n_lang_tokens = use_lang ? 2 : 0;       // lid_marker + lang_emb
    int prefix_len = T_enc + n_lang_tokens + 1; // audio + [lid_marker + lang] + BOS

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr-llm: prefix len=%d (%d audio%s + BOS), lang_id=%d, d=%d\n", prefix_len, T_enc,
                use_lang ? " + lid + lang" : "", lang_id, dd);

    // 3. Allocate KV cache and run decoder via ggml graph
    int max_gen = ctx->params.max_new_tokens > 0 ? ctx->params.max_new_tokens : 512;
    int max_ctx = prefix_len + max_gen;
    // Allocate KV cache
    if (!ctx->kv_k) {
        omniasr_alloc_kv_cache(ctx, max_ctx);
    } else if (ctx->kv_max_ctx < max_ctx) {
        // Reallocate if needed
        if (ctx->kv_ctx)
            ggml_free(ctx->kv_ctx);
        if (ctx->kv_buf)
            ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_k = ctx->kv_v = nullptr;
        omniasr_alloc_kv_cache(ctx, max_ctx);
    }
    // Clear KV cache for new transcription
    if (ctx->kv_buf)
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_n_used = 0;
    // 4. Prefill decoder with entire prefix
    std::vector<float> logits;
    if (!omniasr_run_prefill(ctx, encoder_out, d_enc, T_enc, use_lang, lang_id, lid_marker_id, prefix_len, logits,
                             perf)) {
        fprintf(stderr, "omniasr-llm: prefill failed\n");
        return nullptr;
    }
    ctx->kv_n_used = prefix_len;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr-llm: prefill done (%d tokens)\n", prefix_len);

    // 5. Greedy argmax from prefill logits → first generated token
    std::vector<int> output_tokens;
    auto argmax = [&](const std::vector<float>& lg) -> int {
        int best = 0;
        float best_val = lg[0];
        for (int i = 1; i < (int)lg.size(); i++)
            if (lg[i] > best_val) {
                best_val = lg[i];
                best = i;
            }
        return best;
    };

    int cur_token = argmax(logits);
    if (ctx->params.verbosity >= 2)
        fprintf(stderr, "  prefill → token=%d (%s)\n", cur_token,
                cur_token < (int)m.vocab.size() ? m.vocab[cur_token].c_str() : "?");

    if (cur_token != hp.eos_id)
        output_tokens.push_back(cur_token);

    // 6. Autoregressive generation: one token at a time
    for (int step = 0; (int)output_tokens.size() < max_gen && cur_token != hp.eos_id; step++) {
        if (cur_token < 0 || cur_token >= tok_emb_size) {
            break; // Invalid token
        }

        int n_past = prefix_len + step;
        if (!omniasr_run_dec_token(ctx, cur_token, n_past, logits, perf)) {
            fprintf(stderr, "omniasr-llm: decode step %d failed\n", step);
            break;
        }

        cur_token = argmax(logits);
        if (cur_token == hp.eos_id)
            break;
        output_tokens.push_back(cur_token);

        if (ctx->params.verbosity >= 2 && step < 5)
            fprintf(stderr, "  gen %d: token=%d (%s)\n", step, cur_token,
                    cur_token < (int)m.vocab.size() ? m.vocab[cur_token].c_str() : "?");
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr-llm: generated %d tokens\n", (int)output_tokens.size());

    // Detokenize (same as CTC)
    std::string result;
    for (int tid : output_tokens) {
        if (tid == hp.bos_id || tid == hp.eos_id || tid == hp.pad_id || tid == hp.unk_id)
            continue;
        if (tid < (int)m.vocab.size()) {
            std::string piece = m.vocab[tid];
            for (size_t i = 0; i < piece.size(); i++) {
                if ((unsigned char)piece[i] == 0xE2 && i + 2 < piece.size() && (unsigned char)piece[i + 1] == 0x96 &&
                    (unsigned char)piece[i + 2] == 0x81) {
                    result += ' ';
                    i += 2;
                } else {
                    result += piece[i];
                }
            }
        }
    }

    // Trim
    while (!result.empty() && result.front() == ' ')
        result.erase(result.begin());
    while (!result.empty() && result.back() == ' ')
        result.pop_back();

    if (result.empty())
        return nullptr;

    char* out = (char*)malloc(result.size() + 1);
    if (!out)
        return nullptr;
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    return out;
}
