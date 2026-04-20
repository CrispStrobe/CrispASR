// omniasr.cpp — Facebook OmniASR-CTC runtime.
//
// Full ggml graph: CNN frontend → Transformer encoder → CTC head.
// No mel features needed — processes raw 16kHz PCM directly.

#include "omniasr.h"
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
    int d_model    = 1024;
    int d_ffn      = 4096;
    int n_heads    = 16;
    int n_enc      = 24;
    int n_cnn      = 7;
    int vocab_size = 9812;
    int bos_id     = 0;
    int eos_id     = 2;
    int pad_id     = 1;
    int unk_id     = 3;
    int head_dim   = 64;
};

struct omniasr_model {
    omniasr_hparams hp;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<std::string> vocab;
    std::vector<int> cnn_strides;
};

struct omniasr_context {
    omniasr_model model;
    omniasr_context_params params;
    ggml_backend_t backend   = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;
};

// ===========================================================================
// Defaults
// ===========================================================================

extern "C" struct omniasr_context_params omniasr_context_default_params(void) {
    return {/*.n_threads=*/4, /*.verbosity=*/1};
}

// ===========================================================================
// ggml graph helpers
// ===========================================================================

// LayerNorm: (x - mean) / sqrt(var + eps) * w + b
// x: [C, T] col-major (ne[0]=C). w,b: [C].
static ggml_tensor* build_ln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// Transformer encoder layer (pre-norm)
// x: [d_model, T] col-major
static ggml_tensor* build_enc_layer(ggml_context* ctx, ggml_tensor* x,
                                     ggml_tensor* attn_ln_w, ggml_tensor* attn_ln_b,
                                     ggml_tensor* q_w, ggml_tensor* q_b,
                                     ggml_tensor* k_w, ggml_tensor* k_b,
                                     ggml_tensor* v_w, ggml_tensor* v_b,
                                     ggml_tensor* o_w, ggml_tensor* o_b,
                                     ggml_tensor* ffn_ln_w, ggml_tensor* ffn_ln_b,
                                     ggml_tensor* up_w, ggml_tensor* up_b,
                                     ggml_tensor* down_w, ggml_tensor* down_b,
                                     int n_heads, int head_dim) {
    int d = (int)x->ne[0];
    int T = (int)x->ne[1];

    // Self-attention with pre-norm
    ggml_tensor* residual = x;
    ggml_tensor* h = build_ln(ctx, x, attn_ln_w, attn_ln_b);

    // Q, K, V projections
    ggml_tensor* Q = ggml_mul_mat(ctx, q_w, h);
    if (q_b) Q = ggml_add(ctx, Q, q_b);
    ggml_tensor* K = ggml_mul_mat(ctx, k_w, h);
    if (k_b) K = ggml_add(ctx, K, k_b);
    ggml_tensor* V = ggml_mul_mat(ctx, v_w, h);
    if (v_b) V = ggml_add(ctx, V, v_b);

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
    if (o_b) attn = ggml_add(ctx, attn, o_b);

    // Residual
    x = ggml_add(ctx, residual, attn);

    // FFN with pre-norm
    residual = x;
    h = build_ln(ctx, x, ffn_ln_w, ffn_ln_b);
    h = ggml_mul_mat(ctx, up_w, h);
    if (up_b) h = ggml_add(ctx, h, up_b);
    h = ggml_gelu(ctx, h);
    h = ggml_mul_mat(ctx, down_w, h);
    if (down_b) h = ggml_add(ctx, h, down_b);

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
    if (!gctx) { delete ctx; return nullptr; }

    hp.d_model    = core_gguf::kv_u32(gctx, "omniasr.d_model", 1024);
    hp.d_ffn      = core_gguf::kv_u32(gctx, "omniasr.d_ffn", 4096);
    hp.n_heads    = core_gguf::kv_u32(gctx, "omniasr.n_heads", 16);
    hp.n_enc      = core_gguf::kv_u32(gctx, "omniasr.n_enc_layers", 24);
    hp.n_cnn      = core_gguf::kv_u32(gctx, "omniasr.n_cnn_layers", 7);
    hp.vocab_size = core_gguf::kv_u32(gctx, "omniasr.vocab_size", 9812);
    hp.bos_id     = core_gguf::kv_u32(gctx, "omniasr.bos_id", 0);
    hp.eos_id     = core_gguf::kv_u32(gctx, "omniasr.eos_id", 2);
    hp.pad_id     = core_gguf::kv_u32(gctx, "omniasr.pad_id", 1);
    hp.unk_id     = core_gguf::kv_u32(gctx, "omniasr.unk_id", 3);
    hp.head_dim   = hp.d_model / hp.n_heads;

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
            if (s) m.vocab[i] = s;
        }
    }
    gguf_free(gctx);

    if (params.verbosity >= 1) {
        fprintf(stderr, "omniasr: d=%d, ffn=%d, heads=%d, enc=%d, cnn=%d, vocab=%d\n",
                hp.d_model, hp.d_ffn, hp.n_heads, hp.n_enc, hp.n_cnn, hp.vocab_size);
    }

    // Load weights
    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) { delete ctx; return nullptr; }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "omniasr-ctc", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    m.tensors = wl.tensors;

    // Backend scheduler
    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 65536, false, false);

    if (params.verbosity >= 1) {
        fprintf(stderr, "omniasr: loaded %zu tensors, %zu vocab\n", m.tensors.size(), m.vocab.size());
    }

    return ctx;
}

extern "C" void omniasr_free(struct omniasr_context* ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->weight_ctx) ggml_free(ctx->weight_ctx);
    if (ctx->buf) ggml_backend_buffer_free(ctx->buf);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Transcribe
// ===========================================================================

extern "C" char* omniasr_transcribe(struct omniasr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;

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

    // Input: raw PCM [n_samples, 1] — ggml col-major: ne[0]=n_samples, ne[1]=1
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(inp, "pcm");
    ggml_set_input(inp);

    // CNN Feature Extractor: 7 layers of Conv1d + LayerNorm + GELU
    ggml_tensor* h = inp;
    for (int i = 0; i < hp.n_cnn; i++) {
        std::string prefix = "cnn." + std::to_string(i);
        ggml_tensor* conv_w = G(prefix + ".conv.weight");
        ggml_tensor* conv_b = G(prefix + ".conv.bias");
        ggml_tensor* ln_w   = G(prefix + ".ln.weight");
        ggml_tensor* ln_b   = G(prefix + ".ln.bias");

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
    ggml_tensor* pe_ln_w = G("encoder_frontend.post_extract_ln.weight");
    ggml_tensor* pe_ln_b = G("encoder_frontend.post_extract_ln.bias");
    if (pe_ln_w)
        h = build_ln(ctx0, h, pe_ln_w, pe_ln_b);

    // Pre-lookup CTC tensors (needed in both single-graph and two-graph paths)
    ggml_tensor* ctc_w = G("ctc.weight");
    ggml_tensor* ctc_b = G("ctc.bias");

    // Linear projection: 512 → d_model
    ggml_tensor* proj_w = G("proj.weight");
    ggml_tensor* proj_b = G("proj.bias");
    h = ggml_mul_mat(ctx0, proj_w, h); // [d_model, T]
    if (proj_b) h = ggml_add(ctx0, h, proj_b);

    ggml_set_name(h, "proj_out");
    ggml_set_output(h);

    // Convolutional positional encoding: grouped Conv1d + GELU + residual.
    // Weight normalization pre-computed in converter → stored as pos_conv.weight.
    // Groups=16, kernel=128, channels=1024.
    // ggml lacks grouped conv, so we split: graph 1 (CNN+proj), CPU (pos_conv), graph 2 (transformer+CTC).
    {
        ggml_tensor* wv_t = G("pos_conv.weight");
        ggml_tensor* pb_t = G("pos_conv.bias");

        if (wv_t && pb_t) {
            // Read weights to CPU
            int K_pos = (int)wv_t->ne[0]; // 128
            int IC_g  = (int)wv_t->ne[1]; // 64 (input channels per group)
            int OC    = (int)wv_t->ne[2]; // 1024
            int groups = OC / IC_g;        // 16
            int pad_pos = (K_pos - 1) / 2; // 63

            // Read the projection output to CPU for pos conv computation
            // h is [d_model=1024, T] in ggml. Mark as output to read after graph.
            // But we haven't computed the graph yet! We need to split:
            // Graph 1: CNN + LN + projection → read h to CPU
            // CPU: pos_encoder grouped conv
            // Graph 2: h + pos → transformer → CTC

            // Mark h as output for Graph 1
            ggml_set_name(h, "h_pre_pos");
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);

            // Compute Graph 1
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
                fprintf(stderr, "omniasr: graph 1 alloc failed\n");
                ggml_free(ctx0);
                return nullptr;
            }
            ggml_backend_tensor_set(inp, samples, 0, n_samples * sizeof(float));
            if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "omniasr: graph 1 compute failed\n");
                ggml_free(ctx0);
                return nullptr;
            }

            // Read h_pre_pos: [d_model, T] col-major = data[t * d + c]... no:
            // ggml [d_model, T] with ne[0]=d_model, ne[1]=T: data[t * d_model + c]
            ggml_tensor* h_cpu_t = ggml_graph_get_tensor(gf, "h_pre_pos");
            int d = (int)h_cpu_t->ne[0]; // 1024
            int T_pos = (int)h_cpu_t->ne[1]; // 549
            std::vector<float> h_cpu(d * T_pos);
            ggml_backend_tensor_get(h_cpu_t, h_cpu.data(), 0, d * T_pos * sizeof(float));

            if (ctx->params.verbosity >= 2)
                fprintf(stderr, "  pos_encoder: d=%d, T=%d, K=%d, groups=%d\n", d, T_pos, K_pos, groups);

            // Read pre-computed pos conv weight (weight normalization done in converter)
            // Layout: ggml [K, IC_g, OC] col-major = data[oc * IC_g * K + ic * K + k]
            std::vector<float> pos_w(OC * IC_g * K_pos), bias(OC);
            if (wv_t->type == GGML_TYPE_F16) {
                std::vector<uint16_t> tmp(OC * IC_g * K_pos);
                ggml_backend_tensor_get(wv_t, tmp.data(), 0, tmp.size() * 2);
                pos_w.resize(tmp.size());
                ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp.data(), pos_w.data(), tmp.size());
            } else {
                ggml_backend_tensor_get(wv_t, pos_w.data(), 0, pos_w.size() * sizeof(float));
            }
            ggml_backend_tensor_get(pb_t, bias.data(), 0, OC * sizeof(float));

            // Grouped Conv1d on CPU: h [d=1024, T] → pos [d=1024, T]
            // h layout: data[t * d + c] (ggml col-major)
            // Groups=16: group g processes channels [g*64, (g+1)*64) with conv [64, 64, 128]
            int C_per_g = IC_g; // 64
            std::vector<float> pos(d * T_pos, 0);
            for (int g = 0; g < groups; g++) {
                int c_start = g * C_per_g;
                for (int oc_local = 0; oc_local < C_per_g; oc_local++) {
                    int oc = c_start + oc_local;
                    for (int t = 0; t < T_pos; t++) {
                        float s = bias[oc];
                        for (int ic_local = 0; ic_local < C_per_g; ic_local++) {
                            int ic = c_start + ic_local;
                            for (int k = 0; k < K_pos; k++) {
                                int ti = t + k - pad_pos;
                                if (ti >= 0 && ti < T_pos) {
                                    // h at [ic, ti]: h_cpu[ti * d + ic]
                                    // weight at [oc, ic_local, k]: pos_w[oc * IC_g * K_pos + ic_local * K_pos + k]
                                    s += h_cpu[ti * d + ic] * pos_w[oc * IC_g * K_pos + ic_local * K_pos + k];
                                }
                            }
                        }
                        // GELU
                        float v = s;
                        v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                        // pos at [oc, t]: pos[t * d + oc]
                        pos[t * d + oc] = v;
                    }
                }
            }

            // Add pos to h: h = h + pos
            for (int i = 0; i < d * T_pos; i++)
                h_cpu[i] += pos[i];

            // Now rebuild Graph 2: transformer + CTC using h_cpu as input
            ggml_free(ctx0);
            mem = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(65536, false);
            meta.resize(mem);
            gp = {mem, meta.data(), true};
            ctx0 = ggml_init(gp);
            gf = ggml_new_graph_custom(ctx0, 65536, false);

            // New input: h with pos encoding [d_model, T]
            h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_pos);
            ggml_set_name(h, "h_with_pos");
            ggml_set_input(h);

            // Continue with transformer layers below (h is the input)
            // We'll set the tensor data before computing Graph 2
            // Store h_cpu for later
            // (the ggml_backend_tensor_set will happen after graph alloc)

            // We need to store h_cpu and set it after alloc
            // Use a static/member variable or lambda capture
            // For simplicity: store in a local and set after alloc below

            // Build transformer layers onto h
            for (int i = 0; i < hp.n_enc; i++) {
                std::string p = "enc." + std::to_string(i);
                h = build_enc_layer(ctx0, h,
                                    G(p + ".attn_ln.weight"), G(p + ".attn_ln.bias"),
                                    G(p + ".attn.q_proj.weight"), G(p + ".attn.q_proj.bias"),
                                    G(p + ".attn.k_proj.weight"), G(p + ".attn.k_proj.bias"),
                                    G(p + ".attn.v_proj.weight"), G(p + ".attn.v_proj.bias"),
                                    G(p + ".attn.out.weight"), G(p + ".attn.out.bias"),
                                    G(p + ".ffn_ln.weight"), G(p + ".ffn_ln.bias"),
                                    G(p + ".ffn.up.weight"), G(p + ".ffn.up.bias"),
                                    G(p + ".ffn.down.weight"), G(p + ".ffn.down.bias"),
                                    hp.n_heads, hp.head_dim);
            }

            // Final LayerNorm + CTC head
            h = build_ln(ctx0, h, G("enc_ln.weight"), G("enc_ln.bias"));
            h = ggml_mul_mat(ctx0, ctc_w, h);
            if (ctc_b) h = ggml_add(ctx0, h, ctc_b);

            ggml_set_name(h, "logits");
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);

            // Allocate Graph 2
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
                fprintf(stderr, "omniasr: graph 2 alloc failed\n");
                ggml_free(ctx0);
                return nullptr;
            }

            // Set h_with_pos input
            ggml_tensor* h_inp = ggml_graph_get_tensor(gf, "h_with_pos");
            ggml_backend_tensor_set(h_inp, h_cpu.data(), 0, d * T_pos * sizeof(float));

            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "omniasr: %d samples, graph 2 computing (%d enc layers)...\n", n_samples, hp.n_enc);

            if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "omniasr: graph 2 compute failed\n");
                ggml_free(ctx0);
                return nullptr;
            }

            // Skip the original single-graph path below
            goto read_logits;
        }
    }

    // h is now [d_model, T] — correct format for transformer layers

    // Transformer encoder layers
    for (int i = 0; i < hp.n_enc; i++) {
        std::string p = "enc." + std::to_string(i);
        h = build_enc_layer(ctx0, h,
                            G(p + ".attn_ln.weight"), G(p + ".attn_ln.bias"),
                            G(p + ".attn.q_proj.weight"), G(p + ".attn.q_proj.bias"),
                            G(p + ".attn.k_proj.weight"), G(p + ".attn.k_proj.bias"),
                            G(p + ".attn.v_proj.weight"), G(p + ".attn.v_proj.bias"),
                            G(p + ".attn.out.weight"), G(p + ".attn.out.bias"),
                            G(p + ".ffn_ln.weight"), G(p + ".ffn_ln.bias"),
                            G(p + ".ffn.up.weight"), G(p + ".ffn.up.bias"),
                            G(p + ".ffn.down.weight"), G(p + ".ffn.down.bias"),
                            hp.n_heads, hp.head_dim);
    }

    // Final LayerNorm
    h = build_ln(ctx0, h, G("enc_ln.weight"), G("enc_ln.bias"));

    // CTC head: linear projection to vocab
    h = ggml_mul_mat(ctx0, ctc_w, h); // [vocab_size, T]
    if (ctc_b) h = ggml_add(ctx0, h, ctc_b);

    ggml_set_name(h, "logits");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    // Allocate and compute
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "omniasr: graph alloc failed\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // Set input
    ggml_backend_tensor_set(inp, samples, 0, n_samples * sizeof(float));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "omniasr: %d samples, computing graph...\n", n_samples);

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "omniasr: graph compute failed\n");
        ggml_free(ctx0);
        return nullptr;
    }

read_logits:
    // Debug: read intermediate outputs
    if (ctx->params.verbosity >= 2) {
        auto dump = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t) { fprintf(stderr, "  %s: NOT FOUND\n", name); return; }
            float buf[10];
            int n = std::min(10, (int)ggml_nelements(t));
            ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
            fprintf(stderr, "  %s: ne=[%lld,%lld], data[:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    name, (long long)t->ne[0], (long long)t->ne[1],
                    buf[0], n>1?buf[1]:0, n>2?buf[2]:0, n>3?buf[3]:0, n>4?buf[4]:0);
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
            for (int i = 1; i < V_dbg; i++) if (frame0[i] > frame0[best]) best = i;
            fprintf(stderr, "  logits frame 0: argmax=%d (%.4f), blank(%d)=%.4f\n",
                    best, frame0[best], hp.pad_id, frame0[hp.pad_id]);
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
    int blank_id = hp.pad_id; // CTC blank = <pad> = 1
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
                if ((unsigned char)piece[i] == 0xE2 && i + 2 < piece.size() &&
                    (unsigned char)piece[i + 1] == 0x96 && (unsigned char)piece[i + 2] == 0x81) {
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
    while (!result.empty() && result.front() == ' ') result.erase(result.begin());
    while (!result.empty() && result.back() == ' ') result.pop_back();

    if (result.empty()) return nullptr;

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    return out;
}
