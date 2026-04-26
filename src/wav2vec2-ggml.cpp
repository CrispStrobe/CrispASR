/**
 * wav2vec2-ggml.cpp  —  Wav2Vec2ForCTC forward pass using ggml.
 *
 * Implements wav2vec2_load(), wav2vec2_compute_logits(), and
 * wav2vec2_greedy_decode() declared in wav2vec2-ggml.h.
 *
 * Architecture (do_stable_layer_norm = true, which is what HF exports):
 *   CNN feature extractor (7 strided conv layers)
 *   Feature projection:  LayerNorm(C_cnn) → Linear(C_cnn → H)
 *   Positional conv:     grouped Conv1d(H, H, K=128, G=16) + GELU, residual
 *   L × Transformer layer (pre-norm):
 *       LN → MHA(n_heads) → residual
 *       LN → FFN(H → I → H, GELU) → residual
 *   Global LayerNorm(H)
 *   LM head: Linear(H → V)
 *
 * All large linear layers use ggml_mul_mat so quantised weights (Q4_K_M etc.)
 * work transparently.  CNN, norms, pos-conv and attention scores use manual F32.
 *
 * Adapted from nabil6391/wav2vec2.cpp (MIT licence).
 */

#include "wav2vec2-ggml.h"
#include "core/gguf_loader.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ===========================================================================
// GGUF loading helpers
// ===========================================================================

static ggml_tensor* require_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "[wav2vec2] required tensor '%s' not found in GGUF\n", name);
        exit(1);
    }
    return t;
}

static uint32_t gguf_u32(gguf_context* gctx, const char* key, uint32_t def = 0) {
    int idx = gguf_find_key(gctx, key);
    return idx >= 0 ? (uint32_t)gguf_get_val_u32(gctx, idx) : def;
}

static float gguf_f32(gguf_context* gctx, const char* key, float def = 0.f) {
    int idx = gguf_find_key(gctx, key);
    return idx >= 0 ? gguf_get_val_f32(gctx, idx) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

bool wav2vec2_load(const char* fname, wav2vec2_model& model) {
    // Phase 1: metadata pass (no tensor allocation)
    gguf_init_params p_meta = {/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* gctx_meta = gguf_init_from_file(fname, p_meta);
    if (!gctx_meta) {
        fprintf(stderr, "[wav2vec2] cannot open: %s\n", fname);
        return false;
    }

    auto& hp = model.hparams;
    hp.vocab_size = gguf_u32(gctx_meta, "wav2vec2.vocab_size", hp.vocab_size);
    hp.hidden_size = gguf_u32(gctx_meta, "wav2vec2.hidden_size", hp.hidden_size);
    hp.num_hidden_layers = gguf_u32(gctx_meta, "wav2vec2.num_hidden_layers", hp.num_hidden_layers);
    hp.num_attention_heads = gguf_u32(gctx_meta, "wav2vec2.num_attention_heads", hp.num_attention_heads);
    hp.intermediate_size = gguf_u32(gctx_meta, "wav2vec2.intermediate_size", hp.intermediate_size);
    hp.num_feat_extract_layers = gguf_u32(gctx_meta, "wav2vec2.num_feat_extract_layers", hp.num_feat_extract_layers);
    hp.num_conv_pos_embeddings = gguf_u32(gctx_meta, "wav2vec2.num_conv_pos_embeddings", hp.num_conv_pos_embeddings);
    hp.num_conv_pos_embedding_groups =
        gguf_u32(gctx_meta, "wav2vec2.num_conv_pos_embedding_groups", hp.num_conv_pos_embedding_groups);
    hp.layer_norm_eps = gguf_f32(gctx_meta, "wav2vec2.layer_norm_eps", hp.layer_norm_eps);
    hp.pad_token_id = gguf_u32(gctx_meta, "wav2vec2.pad_token_id", hp.pad_token_id);
    hp.feat_extract_norm_type = gguf_u32(gctx_meta, "wav2vec2.feat_extract_norm_type", hp.feat_extract_norm_type);
    hp.do_stable_layer_norm = gguf_u32(gctx_meta, "wav2vec2.do_stable_layer_norm", hp.do_stable_layer_norm);
    hp.global_ln_before_encoder = gguf_u32(gctx_meta, "wav2vec2.global_ln_before_encoder", hp.global_ln_before_encoder);

    for (uint32_t i = 0; i < hp.num_feat_extract_layers; i++) {
        char key[64];
        snprintf(key, sizeof(key), "wav2vec2.conv_dim_%u", i);
        hp.conv_dim[i] = gguf_u32(gctx_meta, key, hp.conv_dim[i]);
        snprintf(key, sizeof(key), "wav2vec2.conv_kernel_%u", i);
        hp.conv_kernel[i] = gguf_u32(gctx_meta, key, hp.conv_kernel[i]);
        snprintf(key, sizeof(key), "wav2vec2.conv_stride_%u", i);
        hp.conv_stride[i] = gguf_u32(gctx_meta, key, hp.conv_stride[i]);
    }

    // Vocabulary
    {
        int idx = gguf_find_key(gctx_meta, "tokenizer.ggml.tokens");
        if (idx >= 0) {
            uint32_t n = gguf_get_arr_n(gctx_meta, idx);
            model.vocab.resize(n);
            for (uint32_t i = 0; i < n; i++) {
                const char* s = gguf_get_arr_str(gctx_meta, idx, i);
                model.vocab[i] = s ? s : "";
            }
        }
    }
    gguf_free(gctx_meta);

    // Phase 2: load tensors via core_gguf (backend-buffer-backed)
    // This puts all weights in a ggml_backend_buffer so the graph
    // scheduler can reference them without cross-context issues.
    // Try GPU first (CUDA/Metal/Vulkan), fall back to CPU.
    model.backend = ggml_backend_init_best();
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }
    if (!model.backend) {
        fprintf(stderr, "[wav2vec2] failed to init any backend\n");
        return false;
    }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(fname, model.backend, "wav2vec2", wl)) {
        fprintf(stderr, "[wav2vec2] failed to load weights from: %s\n", fname);
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            fprintf(stderr, "[wav2vec2] required tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };
    auto try_get = [&](const char* name) -> ggml_tensor* {
        auto it = model.tensors.find(name);
        return it != model.tensors.end() ? it->second : nullptr;
    };

    model.enc.resize(hp.num_hidden_layers);
    uint32_t L = hp.num_feat_extract_layers;

    for (uint32_t i = 0; i < L; i++) {
        char buf[80];
        snprintf(buf, sizeof(buf), "cnn.%u.conv.weight", i);
        model.cnn[i].conv_w = get(buf);
        snprintf(buf, sizeof(buf), "cnn.%u.conv.bias", i);
        model.cnn[i].conv_b = try_get(buf);
        snprintf(buf, sizeof(buf), "cnn.%u.norm.weight", i);
        model.cnn[i].norm_w = try_get(buf);
        if (model.cnn[i].norm_w) {
            snprintf(buf, sizeof(buf), "cnn.%u.norm.bias", i);
            model.cnn[i].norm_b = get(buf);
            model.cnn[i].has_norm = true;
        }
    }

    model.fp_ln_w = get("feat_proj.ln.weight");
    model.fp_ln_b = get("feat_proj.ln.bias");
    model.fp_w = get("feat_proj.weight");
    model.fp_b = get("feat_proj.bias");
    model.pos_conv_w = get("pos_conv.weight");
    model.pos_conv_b = get("pos_conv.bias");
    // Multi-layer pos_conv (Data2Vec/HuBERT)
    model.n_pos_conv_layers = 1;
    for (int li = 0; li < 8; li++) {
        char wn[64], bn[64];
        snprintf(wn, sizeof(wn), "pos_conv.%d.weight", li);
        snprintf(bn, sizeof(bn), "pos_conv.%d.bias", li);
        model.pos_conv_layers[li].w = try_get(wn);
        model.pos_conv_layers[li].b = try_get(bn);
        if (model.pos_conv_layers[li].w)
            model.n_pos_conv_layers = li + 1;
    }
    model.enc_ln_w = get("enc.ln.weight");
    model.enc_ln_b = get("enc.ln.bias");

    for (uint32_t i = 0; i < hp.num_hidden_layers; i++) {
        char buf[80];
        auto& e = model.enc[i];
        snprintf(buf, sizeof(buf), "enc.%u.ln1.weight", i);
        e.ln1_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ln1.bias", i);
        e.ln1_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.q.weight", i);
        e.q_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.q.bias", i);
        e.q_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.k.weight", i);
        e.k_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.k.bias", i);
        e.k_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.v.weight", i);
        e.v_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.v.bias", i);
        e.v_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.out.weight", i);
        e.o_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.out.bias", i);
        e.o_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ln2.weight", i);
        e.ln2_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ln2.bias", i);
        e.ln2_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc1.weight", i);
        e.fc1_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc1.bias", i);
        e.fc1_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc2.weight", i);
        e.fc2_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc2.bias", i);
        e.fc2_b = get(buf);
    }

    model.lm_w = get("lm_head.weight");
    model.lm_b = try_get("lm_head.bias");

    fprintf(stderr, "[wav2vec2] vocab=%u  hidden=%u  layers=%u  heads=%u  ffn=%u\n", hp.vocab_size, hp.hidden_size,
            hp.num_hidden_layers, hp.num_attention_heads, hp.intermediate_size);
    return true;
}

// ===========================================================================
// Manual compute primitives (F32)
// ===========================================================================

// LayerNorm over last dim: x[T, C] → y[T, C] (in-place OK)
static void layer_norm(const float* x, float* y, const float* w, const float* b, int T, int C, float eps) {
    for (int t = 0; t < T; t++) {
        const float* xt = x + t * C;
        float* yt = y + t * C;
        double sum = 0.0, sq = 0.0;
        for (int c = 0; c < C; c++) {
            sum += xt[c];
            sq += (double)xt[c] * xt[c];
        }
        float mean = (float)(sum / C);
        float var = (float)(sq / C) - mean * mean;
        float inv = 1.f / sqrtf(var + eps);
        for (int c = 0; c < C; c++)
            yt[c] = (xt[c] - mean) * inv * w[c] + b[c];
    }
}

// GELU (exact tanh approximation)
static inline float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Softmax in-place over n elements
static void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float s = 0.f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= s;
}

// InstanceNorm (GroupNorm with num_groups=C) on channel-first data [C, L].
// Used by feat_extract_norm="group", layer 0.
static void instance_norm_1d(const float* x, float* y, const float* w, const float* b, int C, int L, float eps) {
    for (int c = 0; c < C; c++) {
        const float* xc = x + c * L;
        float* yc = y + c * L;
        double sum = 0.0, sq = 0.0;
        for (int l = 0; l < L; l++) {
            sum += xc[l];
            sq += (double)xc[l] * xc[l];
        }
        float mean = (float)(sum / L);
        float var = (float)(sq / L) - mean * mean;
        float inv = 1.f / sqrtf(var + eps);
        for (int l = 0; l < L; l++)
            yc[l] = (xc[l] - mean) * inv * w[c] + b[c];
    }
}

// LayerNorm on channel-first [C, L] (norm across C at each L).
// Used by feat_extract_norm="layer", all CNN layers.
static void layer_norm_cf(const float* x, float* y, const float* w, const float* b, int C, int L, float eps) {
    for (int l = 0; l < L; l++) {
        double sum = 0.0, sq = 0.0;
        for (int c = 0; c < C; c++) {
            float v = x[c * L + l];
            sum += v;
            sq += (double)v * v;
        }
        float mean = (float)(sum / C);
        float var = (float)(sq / C) - mean * mean;
        float inv = 1.f / sqrtf(var + eps);
        for (int c = 0; c < C; c++)
            y[c * L + l] = (x[c * L + l] - mean) * inv * w[c] + b[c];
    }
}

// Conv1d (dilation=1), left_pad zero-pads on the left.
// Weight: w[Cout][Cin][K]  Input: x[Cin][L_in]  Output: y[Cout][L_out]
static void conv1d(const float* x, const float* w, const float* b, float* y, int Cin, int Cout, int K, int stride,
                   int L_in, int left_pad = 0) {
    int L_pad = L_in + left_pad;
    int L_out = (L_pad - K) / stride + 1;

    std::vector<float> padded(Cin * L_pad, 0.f);
    for (int c = 0; c < Cin; c++)
        std::memcpy(padded.data() + c * L_pad + left_pad, x + c * L_in, L_in * sizeof(float));

    for (int oc = 0; oc < Cout; oc++) {
        float bv = b ? b[oc] : 0.f;
        for (int t = 0; t < L_out; t++) {
            float sum = bv;
            int t0 = t * stride;
            for (int ic = 0; ic < Cin; ic++) {
                const float* wrow = w + (oc * Cin + ic) * K;
                const float* xrow = padded.data() + ic * L_pad + t0;
                for (int k = 0; k < K; k++)
                    sum += wrow[k] * xrow[k];
            }
            y[oc * L_out + t] = sum;
        }
    }
}

// Grouped Conv1d with symmetric padding so output_len == input_len.
// Used for the positional conv embedding (K=128, groups=16 by default).
// Weight: w[Cout][Cin/groups][K]
static void grouped_conv1d_same(const float* x, const float* w, const float* b, float* y, int C_in, int C_out, int K,
                                int groups, int L) {
    assert(C_in % groups == 0 && C_out % groups == 0);
    int cin_pg = C_in / groups;
    int cout_pg = C_out / groups;
    int pad_total = K - 1; // "same" padding
    int pad_l = pad_total / 2;
    int pad_r = pad_total - pad_l;
    int L_pad = L + pad_l + pad_r;

    std::vector<float> padded(C_in * L_pad, 0.f);
    for (int c = 0; c < C_in; c++)
        std::memcpy(padded.data() + c * L_pad + pad_l, x + c * L, L * sizeof(float));

// Parallelize over output channels (groups × cout_pg = C_out)
#pragma omp parallel for schedule(static) collapse(2)
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < cout_pg; oc++) {
            int ic0 = g * cin_pg, oc0 = g * cout_pg;
            int og = oc0 + oc;
            float bv = b ? b[og] : 0.f;
            for (int t = 0; t < L; t++) {
                float sum = bv;
                for (int ic = 0; ic < cin_pg; ic++) {
                    const float* wrow = w + (og * cin_pg + ic) * K;
                    const float* xrow = padded.data() + (ic0 + ic) * L_pad + t;
                    for (int k = 0; k < K; k++)
                        sum += wrow[k] * xrow[k];
                }
                y[og * L + t] = sum;
            }
        }
    }
}

// ggml-based quantised linear: y[T, n_out] = W[n_out, n_in] * x[T, n_in] + b
// Creates a fresh ggml context per call so scratch-buffer growth is bounded.
static void ggml_linear_f32(std::vector<uint8_t>& scratch, ggml_tensor* W, const float* bias, const float* x, float* y,
                            int n_in, int n_out, int T, int n_threads = 1) {
    size_t ctx_size = (size_t)(n_in + n_out) * T * sizeof(float) * 8 + ggml_tensor_overhead() * 8 +
                      ggml_graph_overhead() + 4 * 1024 * 1024;
    scratch.resize(ctx_size);

    ggml_init_params p = {ctx_size, scratch.data(), /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(p);

    ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, T);
    std::memcpy(xt->data, x, (size_t)n_in * T * sizeof(float));

    ggml_tensor* out = ggml_mul_mat(ctx, W, xt);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float* od = (const float*)out->data;
    if (bias) {
        for (int t = 0; t < T; t++)
            for (int i = 0; i < n_out; i++)
                y[t * n_out + i] = od[t * n_out + i] + bias[i];
    } else {
        std::memcpy(y, od, (size_t)n_out * T * sizeof(float));
    }
    ggml_free(ctx);
}

// ===========================================================================
// ggml graph-based forward pass (new, GPU-ready)
// ===========================================================================

// Build a ggml graph for the FULL wav2vec2 forward pass.
// Returns a graph whose output tensor "logits" has shape (V, T).
//
// This replaces the manual C++ forward pass with a single ggml graph
// that can run on any ggml backend (CPU, CUDA, Metal, Vulkan).
//
// NOTE: The CNN feature extractor stays as manual C++ (CPU-only) because
// ggml_conv_1d with varying strides + LayerNorm/InstanceNorm is fiddly
// to get right in graph form and the CNN is <5% of total compute.
// Only the transformer + feature projection + LM head are in the graph.
static ggml_cgraph* wav2vec2_build_transformer_graph(const wav2vec2_model& m,
                                                     int T, // number of CNN output frames (computed by caller)
                                                     std::vector<uint8_t>& compute_meta) {
    const auto& hp = m.hparams;
    const int H = (int)hp.hidden_size;
    const int n_heads = (int)hp.num_attention_heads;
    const int head_dim = H / n_heads;
    const int I = (int)hp.intermediate_size;
    const int L = (int)hp.num_hidden_layers;
    const int V = (int)hp.vocab_size;
    const float ln_eps = hp.layer_norm_eps;

    size_t ctx_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    compute_meta.resize(ctx_size);
    ggml_init_params ip = {ctx_size, compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: feature-projected hidden states [H, T] from the CNN + proj
    ggml_tensor* cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, H, T);
    ggml_set_name(cur, "hidden_in");
    ggml_set_input(cur);

    const bool pre_norm = (hp.do_stable_layer_norm != 0);

    // Data2Vec: applies global encoder LN BEFORE transformer layers.
    // wav2vec2/HuBERT: global LN is applied AFTER all layers instead.
    if (hp.global_ln_before_encoder) {
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_mul(ctx0, cur, m.enc_ln_w);
        cur = ggml_add(ctx0, cur, m.enc_ln_b);
        ggml_set_name(cur, "after_global_ln");
        ggml_set_output(cur);
    }

    // ---- L × Transformer layers ----
    // pre_norm  (stable/large): LN → attn → add → LN → FFN → add
    // post_norm (standard/base): attn → add → LN → FFN → add → LN
    for (int il = 0; il < L; il++) {
        const auto& e = m.enc[il];
        ggml_tensor* residual = cur;

        // Attention input: pre-norm applies LN before, post-norm after.
        ggml_tensor* attn_in = cur;
        if (pre_norm) {
            attn_in = ggml_norm(ctx0, cur, ln_eps);
            attn_in = ggml_mul(ctx0, attn_in, e.ln1_w);
            attn_in = ggml_add(ctx0, attn_in, e.ln1_b);
        }

        // Q/K/V projections
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, e.q_w, attn_in), e.q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, e.k_w, attn_in), e.k_b);
        ggml_tensor* V_t = ggml_add(ctx0, ggml_mul_mat(ctx0, e.v_w, attn_in), e.v_b);

        // Reshape to (head_dim, n_heads, T)
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T);
        V_t = ggml_reshape_3d(ctx0, V_t, head_dim, n_heads, T);

        // Permute for matmul: (head_dim, n_heads, T) → (head_dim, T, n_heads)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V_t = ggml_cont(ctx0, ggml_permute(ctx0, V_t, 0, 2, 1, 3));

        // scores = K^T @ Q → (T, T, n_heads), scaled + softmax
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        // attn_out = V_for_attn^T @ scores
        ggml_tensor* V_for_attn = ggml_cont(ctx0, ggml_permute(ctx0, V_t, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V_for_attn, scores);

        // Reshape back to (H, T)
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, H, T);

        // Output projection + residual
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, e.o_w, attn), e.o_b);
        cur = ggml_add(ctx0, residual, attn);

        // Post-attention LayerNorm (post-norm only)
        if (!pre_norm) {
            cur = ggml_norm(ctx0, cur, ln_eps);
            cur = ggml_mul(ctx0, cur, e.ln1_w);
            cur = ggml_add(ctx0, cur, e.ln1_b);
        }

        // FFN
        residual = cur;
        ggml_tensor* ffn_in = cur;
        if (pre_norm) {
            ffn_in = ggml_norm(ctx0, cur, ln_eps);
            ffn_in = ggml_mul(ctx0, ffn_in, e.ln2_w);
            ffn_in = ggml_add(ctx0, ffn_in, e.ln2_b);
        }
        ggml_tensor* x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.fc1_w, ffn_in), e.fc1_b);
        x = ggml_gelu(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.fc2_w, x), e.fc2_b);
        cur = ggml_add(ctx0, residual, x);

        // Post-FFN LayerNorm (post-norm only)
        if (!pre_norm) {
            cur = ggml_norm(ctx0, cur, ln_eps);
            cur = ggml_mul(ctx0, cur, e.ln2_w);
            cur = ggml_add(ctx0, cur, e.ln2_b);
        }
    }

    // ---- Global LayerNorm (skip if already applied before the loop for data2vec) ----
    if (!hp.global_ln_before_encoder) {
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_mul(ctx0, cur, m.enc_ln_w);
        cur = ggml_add(ctx0, cur, m.enc_ln_b);
    }

    // ---- LM head: Linear(H → V) ----
    cur = ggml_mul_mat(ctx0, m.lm_w, cur);
    if (m.lm_b)
        cur = ggml_add(ctx0, cur, m.lm_b);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Debug: 1-layer graph vs manual — compare every intermediate
// ===========================================================================
static void wav2vec2_debug_attention(const wav2vec2_model& m, const float* hidden_th, int T, int H) {
    const auto& hp = m.hparams;
    const auto& e = m.enc[0]; // layer 0 only
    int n_heads = (int)hp.num_attention_heads;
    int head_dim = H / n_heads; // 64
    int I = (int)hp.intermediate_size;
    float ln_eps = hp.layer_norm_eps;
    float scale = 1.f / sqrtf((float)head_dim);

    auto cmp = [](const char* name, const float* a, const float* b, int n) {
        float mx = 0;
        for (int i = 0; i < n; i++)
            mx = std::max(mx, fabsf(a[i] - b[i]));
        fprintf(stderr, "  %-20s max_diff=%.6f first4: graph[%.4f %.4f %.4f %.4f] manual[%.4f %.4f %.4f %.4f] %s\n",
                name, mx, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], mx < 0.01f ? "OK" : "FAIL");
    };

    // ---- Build graph: LN → Q/K/V → reshape → permute → matmul → softmax → attn → out_proj ----
    // Using compute_with_ctx (no_alloc=false) to get correct results
    size_t mem = (size_t)H * T * 4 * 30 + (size_t)T * T * n_heads * 4 * 2 + (size_t)I * T * 4 +
                 ggml_tensor_overhead() * 500 + ggml_graph_overhead_custom(1024, false) + 128 * 1024 * 1024;
    fprintf(stderr, "[wav2vec2-dbg] allocating %.0f MB for 1-layer graph\n", mem / (1024.0 * 1024.0));
    std::vector<uint8_t> buf;
    try {
        buf.resize(mem);
    } catch (...) {
        fprintf(stderr, "[wav2vec2-dbg] OOM\n");
        return;
    }
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 1024, false);

    // Input [H, T] — transpose hidden_th [T, H] → [H, T]
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            ((float*)cur->data)[h + t * H] = hidden_th[t * H + h];

    // LN1
    ggml_tensor* x = ggml_norm(ctx, cur, ln_eps);
    x = ggml_mul(ctx, x, e.ln1_w);
    x = ggml_add(ctx, x, e.ln1_b);
    ggml_set_name(x, "ln1");

    // Q projection
    ggml_tensor* Q = ggml_add(ctx, ggml_mul_mat(ctx, e.q_w, x), e.q_b);
    ggml_set_name(Q, "Q_proj");

    // K, V projections
    ggml_tensor* K = ggml_add(ctx, ggml_mul_mat(ctx, e.k_w, x), e.k_b);
    ggml_tensor* V = ggml_add(ctx, ggml_mul_mat(ctx, e.v_w, x), e.v_b);

    // Reshape + permute
    ggml_tensor* Qr = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
    ggml_tensor* Qp = ggml_cont(ctx, ggml_permute(ctx, Qr, 0, 2, 1, 3));
    ggml_set_name(Qp, "Q_perm"); // [head_dim, T, n_heads]
    ggml_tensor* Kr = ggml_reshape_3d(ctx, K, head_dim, n_heads, T);
    ggml_tensor* Kp = ggml_cont(ctx, ggml_permute(ctx, Kr, 0, 2, 1, 3));
    ggml_tensor* Vr = ggml_reshape_3d(ctx, V, head_dim, n_heads, T);
    ggml_tensor* Vp = ggml_cont(ctx, ggml_permute(ctx, Vr, 0, 2, 1, 3));

    // Attention scores
    ggml_tensor* scores = ggml_mul_mat(ctx, Kp, Qp); // [T, T, n_heads]
    scores = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
    ggml_set_name(scores, "attn_scores");

    // attn = scores @ V
    ggml_tensor* Vp2 = ggml_cont(ctx, ggml_permute(ctx, Vp, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(ctx, Vp2, scores);
    ggml_set_name(attn, "attn_raw"); // [head_dim, T, n_heads]

    // Reshape back to [H, T]
    ggml_tensor* ap = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    ggml_tensor* ar = ggml_reshape_2d(ctx, ap, H, T);
    ggml_set_name(ar, "attn_merged");

    // Output projection + residual
    ggml_tensor* out = ggml_add(ctx, ggml_mul_mat(ctx, e.o_w, ar), e.o_b);
    ggml_tensor* res = ggml_add(ctx, cur, out);
    ggml_set_name(res, "after_attn");

    ggml_build_forward_expand(gf, res);
    fprintf(stderr, "[wav2vec2-dbg] graph built, computing...\n");
    fflush(stderr);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    fprintf(stderr, "[wav2vec2-dbg] graph computed, comparing...\n");
    fflush(stderr);

    // ---- Manual path (matching wav2vec2_compute_logits exactly) ----
    std::vector<float> normed(T * H), Q_m(T * H), K_m(T * H), V_m(T * H);
    layer_norm(hidden_th, normed.data(), (const float*)e.ln1_w->data, (const float*)e.ln1_b->data, T, H, ln_eps);

    std::vector<uint8_t> scratch;
    ggml_linear_f32(scratch, e.q_w, (const float*)e.q_b->data, normed.data(), Q_m.data(), H, H, T, 1);
    ggml_linear_f32(scratch, e.k_w, (const float*)e.k_b->data, normed.data(), K_m.data(), H, H, T, 1);
    ggml_linear_f32(scratch, e.v_w, (const float*)e.v_b->data, normed.data(), V_m.data(), H, H, T, 1);

    // Manual attention
    std::vector<float> attn_m(T * H, 0.f);
    for (int h = 0; h < n_heads; h++) {
        int off = h * head_dim;
        std::vector<float> sc(T);
        for (int tq = 0; tq < T; tq++) {
            for (int tk = 0; tk < T; tk++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += Q_m[tq * H + off + d] * K_m[tk * H + off + d];
                sc[tk] = dot * scale;
            }
            float mx = *std::max_element(sc.begin(), sc.end());
            float sm = 0;
            for (int j = 0; j < T; j++) {
                sc[j] = expf(sc[j] - mx);
                sm += sc[j];
            }
            for (int j = 0; j < T; j++)
                sc[j] /= sm;
            for (int tv = 0; tv < T; tv++) {
                float s = sc[tv];
                for (int d = 0; d < head_dim; d++)
                    attn_m[tq * H + off + d] += s * V_m[tv * H + off + d];
            }
        }
    }
    std::vector<float> oproj_m(T * H);
    ggml_linear_f32(scratch, e.o_w, (const float*)e.o_b->data, attn_m.data(), oproj_m.data(), H, H, T, 1);
    std::vector<float> res_m(T * H);
    for (int i = 0; i < T * H; i++)
        res_m[i] = hidden_th[i] + oproj_m[i];

    // ---- Compare ----
    fprintf(stderr, "[wav2vec2-dbg] 1-layer attention comparison (T=%d H=%d heads=%d):\n", T, H, n_heads);

    // LN1: graph is [H, T] ggml = data[h + t*H]. Manual is [T, H] = data[t*H + h]. Same layout!
    ggml_tensor* gln1 = ggml_graph_get_tensor(gf, "ln1");
    // Convert graph [H,T] to [T,H] for comparison
    std::vector<float> gln1_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gln1_th[t * H + h] = ((float*)gln1->data)[h + t * H];
    cmp("LN1", gln1_th.data(), normed.data(), T * H);

    // Q projection
    ggml_tensor* gQ = ggml_graph_get_tensor(gf, "Q_proj");
    std::vector<float> gQ_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gQ_th[t * H + h] = ((float*)gQ->data)[h + t * H];
    cmp("Q_proj", gQ_th.data(), Q_m.data(), T * H);

    // Q after permute — graph has [head_dim, T, n_heads]
    ggml_tensor* gQp = ggml_graph_get_tensor(gf, "Q_perm");
    // Q_perm data layout: data[d + t*head_dim + h*head_dim*T]
    // Manual Q_m is [T, H] where H = n_heads * head_dim
    // Q_m[t, h*head_dim + d] should == Q_perm[d, t, h]
    float qp_diff = 0;
    for (int h = 0; h < n_heads; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < head_dim; d++) {
                float gv = ((float*)gQp->data)[d + t * head_dim + h * head_dim * T];
                float mv = Q_m[t * H + h * head_dim + d];
                qp_diff = std::max(qp_diff, fabsf(gv - mv));
            }
    fprintf(stderr, "  %-20s max_diff=%.6f %s\n", "Q_perm_vs_Q_manual", qp_diff, qp_diff < 0.01f ? "OK" : "FAIL");

    // Attention output
    ggml_tensor* gar = ggml_graph_get_tensor(gf, "attn_merged");
    std::vector<float> gar_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gar_th[t * H + h] = ((float*)gar->data)[h + t * H];
    cmp("attn_merged", gar_th.data(), attn_m.data(), T * H);

    // After attention residual
    ggml_tensor* gres = ggml_graph_get_tensor(gf, "after_attn");
    std::vector<float> gres_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gres_th[t * H + h] = ((float*)gres->data)[h + t * H];
    cmp("after_attn", gres_th.data(), res_m.data(), T * H);

    ggml_free(ctx);
}

#if 0 // Debug function preserved for reference
static void wav2vec2_debug_single_op(const wav2vec2_model & m,
                                      const float * hidden_th, int T, int H) {
    // Test: does ggml_mul_mat(fc1_w, x) in a gallocr graph match
    // ggml_linear_f32(fc1_w, bias, x, y, H, I, T)?
    const auto & e = m.enc[0];
    int I = (int)m.hparams.intermediate_size;

    // Check if weight tensor has buffer and data
    fprintf(stderr, "[dbg] fc1_w: buffer=%p data=%p type=%d ne=(%lld,%lld)\n",
            (void*)e.fc1_w->buffer, e.fc1_w->data, (int)e.fc1_w->type,
            (long long)e.fc1_w->ne[0], (long long)e.fc1_w->ne[1]);
    // Test JUST mul_mat — no LN, just raw matmul with the hidden state
    std::vector<float> fc1_manual(T * I);
    std::vector<uint8_t> scratch;
    ggml_linear_f32(scratch, e.fc1_w, (const float *)e.fc1_b->data,
                    hidden_th, fc1_manual.data(), H, I, T, 1);

    fprintf(stderr, "[dbg] manual fc1(raw)[t=0, 0:8]:");
    for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", fc1_manual[i]);
    fprintf(stderr, "\n");

    // Build a mini graph: just fc1 on raw input (no LN)
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    std::vector<uint8_t> meta(ctx_size);
    ggml_init_params ip = { ctx_size, meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Input: [H, T] in ggml order
    ggml_tensor * x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, H, T);
    ggml_set_name(x, "x_in");
    ggml_set_input(x);

    // fc1 only (no LN)
    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, e.fc1_w, x), e.fc1_b);
    ggml_set_name(fc1, "fc1_out");
    ggml_build_forward_expand(gf, fc1);
    ggml_free(ctx0);

    // Use simple ggml_graph_compute_with_ctx (same as ggml_linear_f32 uses)
    // to eliminate any backend-related differences
    {
        // Rebuild graph with no_alloc=false so tensors get memory
        size_t ctx2_size = (size_t)H * T * sizeof(float) * 2 +
                          (size_t)I * T * sizeof(float) * 2 +
                          ggml_tensor_overhead() * 16 + ggml_graph_overhead() +
                          16 * 1024 * 1024;
        std::vector<uint8_t> buf2(ctx2_size);
        ggml_init_params ip2 = { ctx2_size, buf2.data(), false };
        ggml_context * ctx2 = ggml_init(ip2);
        ggml_cgraph * gf2 = ggml_new_graph(ctx2);

        ggml_tensor * x2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, H, T);

        // Transpose input: hidden_th is [T, H] row-major → [H, T] ggml
        // ggml 2D layout: data[h + t * H]
        float * x2d = (float *)x2->data;
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                x2d[h + t * H] = hidden_th[t * H + h];

        ggml_tensor * fc1_2 = ggml_add(ctx2, ggml_mul_mat(ctx2, e.fc1_w, x2), e.fc1_b);
        ggml_build_forward_expand(gf2, fc1_2);
        ggml_graph_compute_with_ctx(ctx2, gf2, 1);

        const float * fc1_data = (const float *)fc1_2->data;
        fprintf(stderr, "[dbg] simple fc1[t=0, 0:8]:");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", fc1_data[i]);
        fprintf(stderr, "\n");

        ggml_free(ctx2);
    }

    // Now test with gallocr to see if that path differs
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(alloc, gf);
    ggml_gallocr_alloc_graph(alloc, gf);

    // Transpose input
    std::vector<float> x_ht(H * T);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            x_ht[h * T + t] = hidden_th[t * H + h];

    ggml_tensor * x_inp = ggml_graph_get_tensor(gf, "x_in");
    ggml_backend_tensor_set(x_inp, x_ht.data(), 0, H * T * sizeof(float));

    ggml_backend_graph_compute(m.backend, gf);

    ggml_tensor * fc1_out = ggml_graph_get_tensor(gf, "fc1_out");
    std::vector<float> fc1_graph(I);
    ggml_backend_tensor_get(fc1_out, fc1_graph.data(), 0, I * sizeof(float));

    fprintf(stderr, "[dbg] graph  fc1[t=0, 0:8]:");
    for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", fc1_graph[i]);
    fprintf(stderr, "\n");

    // Compare
    float max_diff = 0;
    for (int i = 0; i < I; i++) {
        float diff = fabsf(fc1_manual[i] - fc1_graph[i]);
        if (diff > max_diff) max_diff = diff;
    }
    fprintf(stderr, "[dbg] fc1 max_diff = %.6f (pass=%s)\n",
            max_diff, max_diff < 0.01f ? "YES" : "NO");

    ggml_gallocr_free(alloc);
}
#endif

// ===========================================================================
// Graph-based forward: CNN (manual) → transformer (ggml graph)
// ===========================================================================

std::vector<float> wav2vec2_compute_logits_graph(const wav2vec2_model& m, const float* raw_audio, int n_samples,
                                                 int n_threads) {
    const auto& hp = m.hparams;
    const int H = (int)hp.hidden_size;
    const bool bench = (getenv("WAV2VEC2_BENCH") && getenv("WAV2VEC2_BENCH")[0]);
    int64_t t_cnn = 0, t_proj = 0, t_pos = 0, t_enc = 0, t_lm = 0, t0;

    // ---- 0. Normalize ----
    std::vector<float> audio(raw_audio, raw_audio + n_samples);
    {
        double sum = 0.0, sq = 0.0;
        for (float v : audio) {
            sum += v;
            sq += (double)v * v;
        }
        float mean = (float)(sum / n_samples);
        float std_ = sqrtf(std::max(0.f, (float)(sq / n_samples) - mean * mean) + 1e-7f);
        for (float& v : audio)
            v = (v - mean) / std_;
    }

    // ---- 1. CNN feature extractor via per-layer ggml graphs ----
    // Previously manual C++ loops (88% of runtime). Now uses ggml_conv_1d
    // per layer with scheduler reuse. Each layer is a small graph:
    //   conv1d → transpose → bias → norm → gelu → transpose_back
    // ggml_conv_1d output is [L_out, OC]; we transpose to [OC, L_out]
    // for bias/norm (which operate on ne[0]), then transpose back for
    // the next conv1d input [L, Cin].
    t0 = ggml_time_us();

    // Persistent scheduler for CNN layers — reused across all 7 conv layers.
    ggml_backend_t cnn_bks[1] = {m.backend};
    ggml_backend_sched_t cnn_sc = ggml_backend_sched_new(cnn_bks, nullptr, 1, 128, false, false);

    // cnn_buf holds data in ggml [L, C] format = memory layout data[l + c*L]
    // which is channel-first in C row-major terms: data[c * L + l].
    std::vector<float> cnn_buf(audio.begin(), audio.end());
    uint32_t L_cur = (uint32_t)n_samples, C_cur = 1;

    ggml_backend_cpu_set_n_threads(m.backend, n_threads);

    for (uint32_t li = 0; li < hp.num_feat_extract_layers; li++) {
        uint32_t C_out = hp.conv_dim[li];
        uint32_t K = hp.conv_kernel[li];
        uint32_t S = hp.conv_stride[li];
        uint32_t L_out = (L_cur - K) / S + 1;

        size_t gmem = ggml_tensor_overhead() * 30 + ggml_graph_overhead() + 2 * 1024 * 1024;
        struct ggml_init_params gp = {gmem, nullptr, true};
        ggml_context* cctx = ggml_init(gp);

        // Input: [L_cur, C_cur] in ggml terms (ne[0]=L, ne[1]=C)
        // Memory: data[l + c*L] = channel-first C array data[c*L + l]
        ggml_tensor* inp = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, (int64_t)L_cur, (int64_t)C_cur);
        ggml_set_name(inp, "cnn_in");
        ggml_set_input(inp);

        // F32 im2col + mul_mat (cache-friendly for large C_in × C_out matmuls).
        // im2col output: [IC*K, L_out]; kernel reshaped: [IC*K, OC]
        // mul_mat gives [OC, L_out] — already channels-first, no transpose needed!
        ggml_tensor* im2c = ggml_im2col(cctx, m.cnn[li].conv_w, inp, (int)S, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
        ggml_tensor* kw = m.cnn[li].conv_w;
        int64_t IC_K = kw->ne[0] * kw->ne[1]; // K * Cin
        int64_t OC = kw->ne[2];
        ggml_tensor* kw_2d = ggml_reshape_2d(cctx, kw, IC_K, OC);
        ggml_tensor* cur = ggml_mul_mat(cctx, kw_2d, im2c); // [OC, L_out]
        // ^^^ This is already channels-first [OC, L_out] — no transpose needed!

        // Bias: [OC] adds to [OC, L_out] along ne[0]=OC
        if (m.cnn[li].conv_b)
            cur = ggml_add(cctx, cur, m.cnn[li].conv_b);

        // Norm: ggml_norm over ne[0]=OC
        if (m.cnn[li].has_norm) {
            cur = ggml_norm(cctx, cur, hp.layer_norm_eps);
            if (m.cnn[li].norm_w)
                cur = ggml_mul(cctx, cur, m.cnn[li].norm_w);
            if (m.cnn[li].norm_b)
                cur = ggml_add(cctx, cur, m.cnn[li].norm_b);
        }

        cur = ggml_gelu(cctx, cur);

        // Output is [OC, L_out] in ggml (ne[0]=OC). For next layer's im2col,
        // we need [L_out, OC] (ne[0]=L_out). Transpose once.
        cur = ggml_cont(cctx, ggml_transpose(cctx, cur));

        ggml_set_name(cur, "cnn_out");
        ggml_set_output(cur);

        ggml_cgraph* gf_l = ggml_new_graph(cctx);
        ggml_build_forward_expand(gf_l, cur);

        ggml_backend_sched_t sc = cnn_sc;
        auto asgn = [&](ggml_tensor* t) {
            if (t)
                ggml_backend_sched_set_tensor_backend(sc, t, m.backend);
        };
        asgn(m.cnn[li].conv_w);
        asgn(m.cnn[li].conv_b);
        asgn(m.cnn[li].norm_w);
        asgn(m.cnn[li].norm_b);

        ggml_backend_sched_reset(sc);
        bool ok = ggml_backend_sched_alloc_graph(sc, gf_l);
        if (ok) {
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf_l, "cnn_in"), cnn_buf.data(), 0,
                                    L_cur * C_cur * sizeof(float));
            ok = (ggml_backend_sched_graph_compute(sc, gf_l) == GGML_STATUS_SUCCESS);
        }
        if (ok) {
            ggml_tensor* out_t = ggml_graph_get_tensor(gf_l, "cnn_out");
            cnn_buf.resize((size_t)L_out * C_out);
            ggml_backend_tensor_get(out_t, cnn_buf.data(), 0, cnn_buf.size() * sizeof(float));
        } else {
            // Fallback to manual C++ (shouldn't happen on CPU backend)
            if (bench)
                fprintf(stderr, "wav2vec2: CNN layer %u ggml failed, manual fallback\n", li);
            std::vector<float> cf(C_cur * L_cur), co(C_out * ((L_cur - K) / S + 1));
            for (uint32_t t2 = 0; t2 < L_cur; t2++)
                for (uint32_t c = 0; c < C_cur; c++)
                    cf[c * L_cur + t2] = cnn_buf[t2 * C_cur + c];
            std::vector<float> w_buf;
            const float* wdata;
            if (m.cnn[li].conv_w->type == GGML_TYPE_F16) {
                size_t n = C_out * C_cur * K;
                w_buf.resize(n);
                const ggml_fp16_t* w16 = (const ggml_fp16_t*)m.cnn[li].conv_w->data;
                for (size_t i = 0; i < n; i++)
                    w_buf[i] = ggml_fp16_to_fp32(w16[i]);
                wdata = w_buf.data();
            } else {
                wdata = (const float*)m.cnn[li].conv_w->data;
            }
            conv1d(cf.data(), wdata, m.cnn[li].conv_b ? (const float*)m.cnn[li].conv_b->data : nullptr, co.data(),
                   (int)C_cur, (int)C_out, (int)K, (int)S, (int)L_cur, 0);
            if (m.cnn[li].has_norm) {
                const float* nw = (const float*)m.cnn[li].norm_w->data;
                const float* nb = (const float*)m.cnn[li].norm_b->data;
                if (hp.feat_extract_norm_type == 1)
                    layer_norm_cf(co.data(), co.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
                else
                    instance_norm_1d(co.data(), co.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
            }
            for (float& v : co)
                v = gelu(v);
            cnn_buf.resize(L_out * C_out);
            for (uint32_t t2 = 0; t2 < L_out; t2++)
                for (uint32_t c = 0; c < C_out; c++)
                    cnn_buf[t2 * C_out + c] = co[c * L_out + t2];
        }
        ggml_free(cctx);
        L_cur = L_out;
        C_cur = C_out;
    }
    ggml_backend_sched_free(cnn_sc);

    int T = (int)L_cur;
    int C_cnn = (int)C_cur;

    // cnn_buf data from ggml [L_out, OC] tensor is in memory as data[l + c*L],
    // which is channel-first C array layout data[c*L + l].
    // Transpose to [T, C] row-major for downstream transformer code.
    {
        std::vector<float> tc(T * C_cnn);
        for (int t2 = 0; t2 < T; t2++)
            for (int c = 0; c < C_cnn; c++)
                tc[t2 * C_cnn + c] = cnn_buf[c * T + t2];
        cnn_buf = std::move(tc);
    }

    // Dump CNN output for reference comparison
    {
        const char* dump = getenv("WAV2VEC2_DUMP_DIR");
        if (dump && dump[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s/cnn_out.bin", dump);
            FILE* f = fopen(path, "wb");
            if (f) {
                fwrite(cnn_buf.data(), sizeof(float), C_cnn * T, f);
                fclose(f);
                fprintf(stderr, "  DUMP: cnn_out [%d,%d] → %s\n", C_cnn, T, path);
            }
        }
    }

    std::vector<float>& feat = cnn_buf; // alias — already [T, C_cnn]

    t_cnn = ggml_time_us() - t0;
    // ---- 2. Feature projection: LN + Linear ----
    t0 = ggml_time_us();
    layer_norm(feat.data(), feat.data(), (const float*)m.fp_ln_w->data, (const float*)m.fp_ln_b->data, T, C_cnn,
               hp.layer_norm_eps);

    std::vector<float> hidden(T * H);
    std::vector<uint8_t> scratch;
    ggml_linear_f32(scratch, m.fp_w, (const float*)m.fp_b->data, feat.data(), hidden.data(), C_cnn, H, T, n_threads);

    // Dump feature projection output
    {
        const char* dump = getenv("WAV2VEC2_DUMP_DIR");
        if (dump && dump[0]) {
            char path[512];
            snprintf(path, sizeof(path), "%s/feat_proj.bin", dump);
            FILE* f = fopen(path, "wb");
            if (f) {
                fwrite(hidden.data(), sizeof(float), T * H, f);
                fclose(f);
            }
            fprintf(stderr, "  DUMP: feat_proj [%d,%d] → %s\n", T, H, path);
        }
    }

    t_proj = ggml_time_us() - t0;
    // ---- 3. Positional conv via ggml_conv_1d_group ----
    t0 = ggml_time_us();
    {
        int G_pos = (int)hp.num_conv_pos_embedding_groups;
        int n_layers = m.n_pos_conv_layers;

        // hidden is [T, H] in C row-major: hidden[t * H + h].
        // ggml tensor [T, H] (ne[0]=T, ne[1]=H) stores data as data[t + h * T],
        // which equals hidden_cf[h * T + t] — channel-first C array.
        // Transpose hidden to channel-first for ggml tensor loading.
        std::vector<float> pos_cf(H * T);
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                pos_cf[h * T + t] = hidden[t * H + h];

        // Dual-backend scheduler: ggml_conv_1d_group is CPU-only (no Metal/CUDA kernel),
        // so the scheduler needs a CPU fallback when m.backend is GPU.
        ggml_backend_t pos_cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(pos_cpu, n_threads);
        int pos_n_be = 1;
        ggml_backend_t pos_bks[2] = {m.backend, nullptr};
        if (!ggml_backend_is_cpu(m.backend)) {
            pos_bks[pos_n_be++] = pos_cpu;
        }
        ggml_backend_sched_t pos_sc = ggml_backend_sched_new(pos_bks, nullptr, pos_n_be, 256, false, false);

        for (int li = 0; li < n_layers; li++) {
            ggml_tensor* w_tensor = (li == 0 && m.pos_conv_w) ? m.pos_conv_w : m.pos_conv_layers[li].w;
            ggml_tensor* b_tensor = (li == 0 && m.pos_conv_b) ? m.pos_conv_b : m.pos_conv_layers[li].b;
            if (!w_tensor) {
                fprintf(stderr, "[wav2vec2] pos_conv layer %d weight missing\n", li);
                break;
            }

            int K_this = (int)w_tensor->ne[0];
            // "Same" padding: pad_left = (K-1)/2, pad_right = (K-1) - pad_left
            int pad_left = (K_this - 1) / 2;
            int pad_right = (K_this - 1) - pad_left;

            size_t gmem = ggml_tensor_overhead() * 30 + ggml_graph_overhead() + 2 * 1024 * 1024;
            struct ggml_init_params gp = {gmem, nullptr, true};
            ggml_context* pctx = ggml_init(gp);

            // Input as ggml [T, H] — data stored channel-first: data[t + h*T]
            ggml_tensor* inp = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, (int64_t)T, (int64_t)H);
            ggml_set_name(inp, "pos_in");
            ggml_set_input(inp);

            // Pre-pad along ne[0]=T for asymmetric "same" padding
            ggml_tensor* inp_padded = ggml_pad_ext(pctx, inp, pad_left, pad_right, 0, 0, 0, 0, 0, 0);

            // Grouped conv: [K, H/G, H] × [T_padded, H] → [T, H]
            ggml_tensor* cur = ggml_conv_1d_group(pctx, w_tensor, inp_padded, 1, 0, 1, G_pos);
            if (ggml_n_dims(cur) > 2)
                cur = ggml_reshape_2d(pctx, cur, cur->ne[0], cur->ne[1]);

            // Bias: output [T, H] in ggml = data[t + h*T].
            // ggml_add broadcasts b[ne[0]] over ne[1]. Here ne[0]=T, b has H elements.
            // Need to transpose to [H, T], add bias, transpose back.
            if (b_tensor) {
                cur = ggml_cont(pctx, ggml_transpose(pctx, cur)); // [H, T]
                cur = ggml_add(pctx, cur, b_tensor);
                cur = ggml_cont(pctx, ggml_transpose(pctx, cur)); // [T, H]
            }

            // Multi-layer: LayerNorm over channels
            if (n_layers > 1) {
                cur = ggml_cont(pctx, ggml_transpose(pctx, cur)); // [H, T]
                cur = ggml_norm(pctx, cur, hp.layer_norm_eps);
                cur = ggml_cont(pctx, ggml_transpose(pctx, cur)); // [T, H]
            }

            cur = ggml_gelu(pctx, cur);

            ggml_set_name(cur, "pos_out");
            ggml_set_output(cur);

            ggml_cgraph* gf_p = ggml_new_graph(pctx);
            ggml_build_forward_expand(gf_p, cur);

            ggml_backend_sched_set_tensor_backend(pos_sc, w_tensor, m.backend);
            if (b_tensor)
                ggml_backend_sched_set_tensor_backend(pos_sc, b_tensor, m.backend);

            ggml_backend_sched_reset(pos_sc);
            bool ok = ggml_backend_sched_alloc_graph(pos_sc, gf_p);
            if (ok) {
                // Load channel-first data into ggml [T, H] tensor
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf_p, "pos_in"), pos_cf.data(), 0,
                                        (size_t)T * H * sizeof(float));
                ok = (ggml_backend_sched_graph_compute(pos_sc, gf_p) == GGML_STATUS_SUCCESS);
            }

            if (ok) {
                ggml_tensor* out_t = ggml_graph_get_tensor(gf_p, "pos_out");
                ggml_backend_tensor_get(out_t, pos_cf.data(), 0, (size_t)T * H * sizeof(float));
            } else {
                fprintf(stderr, "[wav2vec2] pos_conv ggml layer %d failed, manual fallback\n", li);
                // Fallback to manual grouped conv (pos_cf is already channel-first)
                std::vector<float> pw_buf;
                const float* pw;
                size_t pw_n = (size_t)ggml_nelements(w_tensor);
                if (w_tensor->type == GGML_TYPE_F16) {
                    pw_buf.resize(pw_n);
                    const ggml_fp16_t* p16 = (const ggml_fp16_t*)w_tensor->data;
                    for (size_t i2 = 0; i2 < pw_n; i2++)
                        pw_buf[i2] = ggml_fp16_to_fp32(p16[i2]);
                    pw = pw_buf.data();
                } else {
                    pw = (const float*)w_tensor->data;
                }
                const float* pb = b_tensor ? (const float*)b_tensor->data : nullptr;
                std::vector<float> pos_out(H * T, 0.0f);
                grouped_conv1d_same(pos_cf.data(), pw, pb, pos_out.data(), H, H, K_this, G_pos, T);

                if (n_layers > 1) {
                    for (int t2 = 0; t2 < T; t2++) {
                        double sum2 = 0, sq2 = 0;
                        for (int h2 = 0; h2 < H; h2++) {
                            float v2 = pos_out[h2 * T + t2];
                            sum2 += v2;
                            sq2 += (double)v2 * v2;
                        }
                        float mean2 = (float)(sum2 / H), var2 = (float)(sq2 / H) - mean2 * mean2;
                        float inv2 = 1.0f / sqrtf(var2 + hp.layer_norm_eps);
                        for (int h2 = 0; h2 < H; h2++)
                            pos_out[h2 * T + t2] = (pos_out[h2 * T + t2] - mean2) * inv2;
                    }
                }
                for (auto& v : pos_out)
                    v = gelu(v);
                pos_cf = std::move(pos_out);
            }

            ggml_free(pctx);
        }

        ggml_backend_sched_free(pos_sc);
        if (!ggml_backend_is_cpu(m.backend))
            ggml_backend_free(pos_cpu);

        // Add residual: hidden[t*H+h] += pos_cf[h*T+t]
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                hidden[t * H + h] += pos_cf[h * T + t];
    }

    t_pos = ggml_time_us() - t0;
    // ---- 4. Transformer + LN + LM head via ggml graph ----
    t0 = ggml_time_us();
    // hidden is [T, H] row-major: data[t*H + h].
    // ggml [H, T] stores data[h + t*H] = data[t*H + h] — SAME layout!
    // No transpose needed.
    std::vector<float> hidden_ht(hidden.begin(), hidden.end());


    // Build graph
    std::vector<uint8_t> compute_meta;
    ggml_cgraph* gf = wav2vec2_build_transformer_graph(m, T, compute_meta);
    if (!gf)
        return {};

    // Layer-by-layer ggml graph approach: build+compute one layer at a time.
    // This uses ~80 MB per layer (reused) instead of 3+ GB for all layers.
    // Each layer graph uses ggml_graph_compute_with_ctx which correctly
    // references external F16 weight tensors (gallocr/sched corrupt them).
    const int L = (int)hp.num_hidden_layers;
    const int I = (int)hp.intermediate_size;
    const int V = (int)hp.vocab_size;
    const int n_heads = (int)hp.num_attention_heads;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;

    // Per-layer memory: ~80 MB for wav2vec2-large (T=549, H=1024, heads=16)
    size_t layer_mem = (size_t)H * T * 4 * 30 + (size_t)T * T * n_heads * 4 * 2 + (size_t)I * T * 4 +
                       ggml_tensor_overhead() * 200 + ggml_graph_overhead_custom(512, false) + 32 * 1024 * 1024;

    // Try full-graph path with ggml_backend_sched (GPU-ready)
    // Uses the pre-built graph function + explicit weight tensor assignment
    {
        std::vector<uint8_t> compute_meta;
        ggml_cgraph* gf = wav2vec2_build_transformer_graph(m, T, compute_meta);
        if (gf) {
            ggml_backend_cpu_set_n_threads(m.backend, n_threads);
            ggml_backend_t backends[1] = {m.backend};
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);

            // CRITICAL: assign all weight tensors to the model's backend
            // so the scheduler doesn't reallocate them
            auto assign = [&](ggml_tensor* t) {
                if (t)
                    ggml_backend_sched_set_tensor_backend(sched, t, m.backend);
            };
            assign(m.fp_w);
            assign(m.fp_b);
            assign(m.fp_ln_w);
            assign(m.fp_ln_b);
            assign(m.pos_conv_w);
            assign(m.pos_conv_b);
            assign(m.enc_ln_w);
            assign(m.enc_ln_b);
            assign(m.lm_w);
            assign(m.lm_b);
            for (int il2 = 0; il2 < L; il2++) {
                const auto& e2 = m.enc[il2];
                assign(e2.ln1_w);
                assign(e2.ln1_b);
                assign(e2.q_w);
                assign(e2.q_b);
                assign(e2.k_w);
                assign(e2.k_b);
                assign(e2.v_w);
                assign(e2.v_b);
                assign(e2.o_w);
                assign(e2.o_b);
                assign(e2.ln2_w);
                assign(e2.ln2_b);
                assign(e2.fc1_w);
                assign(e2.fc1_b);
                assign(e2.fc2_w);
                assign(e2.fc2_b);
            }

            ggml_backend_sched_reset(sched);
            if (ggml_backend_sched_alloc_graph(sched, gf)) {
                ggml_tensor* inp = ggml_graph_get_tensor(gf, "hidden_in");
                if (inp) {
                    ggml_backend_tensor_set(inp, hidden_ht.data(), 0, H * T * sizeof(float));
                    if (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS) {
                        // Dump after_global_ln if available
                        {
                            const char* dump = getenv("WAV2VEC2_DUMP_DIR");
                            ggml_tensor* gln = ggml_graph_get_tensor(gf, "after_global_ln");
                            if (dump && dump[0] && gln) {
                                int ne = (int)ggml_nelements(gln);
                                std::vector<float> d(ne);
                                ggml_backend_tensor_get(gln, d.data(), 0, ne * sizeof(float));
                                char path[512];
                                snprintf(path, sizeof(path), "%s/after_global_ln.bin", dump);
                                FILE* f = fopen(path, "wb");
                                if (f) {
                                    fwrite(d.data(), sizeof(float), ne, f);
                                    fclose(f);
                                }
                                fprintf(stderr, "  DUMP: after_global_ln [%lld,%lld] → %s\n", (long long)gln->ne[0],
                                        (long long)gln->ne[1], path);
                            }
                        }
                        ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
                        if (out) {
                            int V_out = (int)out->ne[0];
                            int T_out = (int)out->ne[1];
                            std::vector<float> logits_tv(V_out * T_out);
                            ggml_backend_tensor_get(out, logits_tv.data(), 0, logits_tv.size() * sizeof(float));
                            // Dump logits
                            {
                                const char* dump = getenv("WAV2VEC2_DUMP_DIR");
                                if (dump && dump[0]) {
                                    char path[512];
                                    snprintf(path, sizeof(path), "%s/logits.bin", dump);
                                    FILE* f = fopen(path, "wb");
                                    if (f) {
                                        fwrite(logits_tv.data(), sizeof(float), logits_tv.size(), f);
                                        fclose(f);
                                    }
                                    fprintf(stderr, "  DUMP: logits [%d,%d] → %s\n", V_out, T_out, path);
                                }
                            }
                            ggml_backend_sched_free(sched);
                            t_enc = ggml_time_us() - t0;
                            if (bench) {
                                fprintf(stderr, "wav2vec2: =========== performance report ===========\n");
                                fprintf(stderr, "wav2vec2:  audio         %7.2f s\n", n_samples / 16000.0);
                                fprintf(stderr, "wav2vec2:  CNN extract   %7.1f ms\n", t_cnn / 1e3);
                                fprintf(stderr, "wav2vec2:  feat proj     %7.1f ms\n", t_proj / 1e3);
                                fprintf(stderr, "wav2vec2:  pos conv      %7.1f ms\n", t_pos / 1e3);
                                fprintf(stderr, "wav2vec2:  encoder graph %7.1f ms  (%d layers, sched path)\n",
                                        t_enc / 1e3, L);
                                fprintf(stderr, "wav2vec2:  total        %7.1f ms\n",
                                        (t_cnn + t_proj + t_pos + t_enc) / 1e3);
                                fprintf(stderr, "wav2vec2: ================================================\n");
                            }
                            return logits_tv;
                        }
                    }
                }
            }
            ggml_backend_sched_free(sched);
            // Fall through to per-layer path if sched failed
            fprintf(stderr, "[wav2vec2] sched path failed (alloc or compute), using per-layer fallback\n");
        }
    }

    // Fallback: per-layer graphs with compute_with_ctx
    // Pre-allocate layer buffer + work buffer once (reused across all layers)
    std::vector<uint8_t> lbuf(layer_mem);
    std::vector<uint8_t> work_buf;

    for (int il = 0; il < L; il++) {
        const auto& e = m.enc[il];

        ggml_init_params lip = {layer_mem, lbuf.data(), false};
        ggml_context* lctx = ggml_init(lip);
        ggml_cgraph* lgf = ggml_new_graph_custom(lctx, 512, false);

        // Input: copy current hidden state
        ggml_tensor* cur = ggml_new_tensor_2d(lctx, GGML_TYPE_F32, H, T);
        memcpy(cur->data, hidden_ht.data(), H * T * sizeof(float));

        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(lctx, cur, ln_eps);
        x = ggml_mul(lctx, x, e.ln1_w);
        x = ggml_add(lctx, x, e.ln1_b);

        // Q/K/V projections
        ggml_tensor* Q = ggml_add(lctx, ggml_mul_mat(lctx, e.q_w, x), e.q_b);
        ggml_tensor* K = ggml_add(lctx, ggml_mul_mat(lctx, e.k_w, x), e.k_b);
        ggml_tensor* Vt = ggml_add(lctx, ggml_mul_mat(lctx, e.v_w, x), e.v_b);

        // Multi-head reshape + permute
        Q = ggml_reshape_3d(lctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(lctx, K, head_dim, n_heads, T);
        Vt = ggml_reshape_3d(lctx, Vt, head_dim, n_heads, T);
        Q = ggml_cont(lctx, ggml_permute(lctx, Q, 0, 2, 1, 3));
        K = ggml_cont(lctx, ggml_permute(lctx, K, 0, 2, 1, 3));
        Vt = ggml_cont(lctx, ggml_permute(lctx, Vt, 0, 2, 1, 3));

        // Attention
        float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* scores = ggml_mul_mat(lctx, K, Q);
        scores = ggml_soft_max_ext(lctx, scores, nullptr, scale, 0.0f);
        ggml_tensor* V_perm = ggml_cont(lctx, ggml_permute(lctx, Vt, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(lctx, V_perm, scores);
        attn = ggml_cont(lctx, ggml_permute(lctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(lctx, attn, H, T);
        attn = ggml_add(lctx, ggml_mul_mat(lctx, e.o_w, attn), e.o_b);
        cur = ggml_add(lctx, residual, attn);

        // Pre-FFN LayerNorm
        residual = cur;
        x = ggml_norm(lctx, cur, ln_eps);
        x = ggml_mul(lctx, x, e.ln2_w);
        x = ggml_add(lctx, x, e.ln2_b);

        // FFN
        x = ggml_add(lctx, ggml_mul_mat(lctx, e.fc1_w, x), e.fc1_b);
        x = ggml_gelu(lctx, x);
        x = ggml_add(lctx, ggml_mul_mat(lctx, e.fc2_w, x), e.fc2_b);
        cur = ggml_add(lctx, residual, x);

        ggml_set_name(cur, "layer_out");
        ggml_build_forward_expand(lgf, cur);

        // Compute — reuse pre-allocated work buffer
        struct ggml_cplan cplan = ggml_graph_plan(lgf, n_threads, NULL);
        if (cplan.work_size > 0) {
            if (work_buf.size() < cplan.work_size)
                work_buf.resize(cplan.work_size);
            cplan.work_data = work_buf.data();
        }
        ggml_graph_compute(lgf, &cplan);

        // Read output back into hidden_ht for the next layer
        ggml_tensor* lout = ggml_graph_get_tensor(lgf, "layer_out");
        memcpy(hidden_ht.data(), lout->data, H * T * sizeof(float));

        ggml_free(lctx);
    }

    // Global LayerNorm + LM head — use ggml_linear_f32 helper (proven working)
    t_enc = ggml_time_us() - t0;
    // Global LayerNorm + LM head
    t0 = ggml_time_us();
    {
        // hidden_ht is [T, H] layout (same as ggml [H, T] data layout)
        layer_norm(hidden_ht.data(), hidden_ht.data(), (const float*)m.enc_ln_w->data, (const float*)m.enc_ln_b->data,
                   T, H, ln_eps);

        int V_size = (int)hp.vocab_size;
        std::vector<float> logits(T * V_size);
        std::vector<uint8_t> scratch;
        ggml_linear_f32(scratch, m.lm_w, m.lm_b ? (const float*)m.lm_b->data : nullptr, hidden_ht.data(), logits.data(),
                        H, V_size, T, n_threads);
        t_lm = ggml_time_us() - t0;

        if (bench) {
            fprintf(stderr, "wav2vec2: =========== performance report ===========\n");
            fprintf(stderr, "wav2vec2:  audio         %7.2f s\n", n_samples / 16000.0);
            fprintf(stderr, "wav2vec2:  CNN extract   %7.1f ms\n", t_cnn / 1e3);
            fprintf(stderr, "wav2vec2:  feat proj     %7.1f ms\n", t_proj / 1e3);
            fprintf(stderr, "wav2vec2:  pos conv      %7.1f ms\n", t_pos / 1e3);
            fprintf(stderr, "wav2vec2:  transformer   %7.1f ms  (%d layers)\n", t_enc / 1e3, (int)hp.num_hidden_layers);
            fprintf(stderr, "wav2vec2:  LN + LM head  %7.1f ms\n", t_lm / 1e3);
            fprintf(stderr, "wav2vec2:  total        %7.1f ms\n", (t_cnn + t_proj + t_pos + t_enc + t_lm) / 1e3);
            fprintf(stderr, "wav2vec2: ================================================\n");
        }

        return logits;
    }
}

// ===========================================================================
// Manual C++ forward pass (legacy, CPU-only)
// ===========================================================================

std::vector<float> wav2vec2_compute_logits(const wav2vec2_model& m, const float* raw_audio, int n_samples,
                                           int n_threads) {
    // Try graph path (uses compute_with_ctx for correct F16 weight handling)
    auto result = wav2vec2_compute_logits_graph(m, raw_audio, n_samples, n_threads);
    if (!result.empty())
        return result;
    // Fallback to manual path if graph OOM:
    // buffers, and compute_with_ctx needs 3+ GB for the full 24-layer graph
    // (T²×heads attention scores kept alive per layer). The manual path below
    // uses per-op ggml mini-graphs for linear projections (F16 matmul) and
    // manual C++ for attention/normalization — correct and ~2× faster than
    // a monolithic graph since ggml can reuse scratch memory between layers.
    //
    // TODO: Fix by either (a) building layer-by-layer ggml graphs that
    // share a single scratch buffer, or (b) patching gallocr to skip
    // allocation for tensors that already have ->buffer set.

    const auto& hp = m.hparams;

    // ------------------------------------------------------------------
    // 0. Normalize: zero-mean, unit-variance
    // ------------------------------------------------------------------
    std::vector<float> audio(raw_audio, raw_audio + n_samples);
    {
        double sum = 0.0, sq = 0.0;
        for (float v : audio) {
            sum += v;
            sq += (double)v * v;
        }
        float mean = (float)(sum / n_samples);
        float std_ = sqrtf(std::max(0.f, (float)(sq / n_samples) - mean * mean) + 1e-7f);
        for (float& v : audio)
            v = (v - mean) / std_;
    }

    // ------------------------------------------------------------------
    // 1. CNN feature extractor  [1, n_samples] → [C_cnn, T]
    // ------------------------------------------------------------------
    uint32_t L_cnn = (uint32_t)n_samples;
    for (uint32_t i = 0; i < hp.num_feat_extract_layers; i++)
        L_cnn = (L_cnn - hp.conv_kernel[i]) / hp.conv_stride[i] + 1;

    uint32_t L_cur = (uint32_t)n_samples, C_cur = 1;
    std::vector<float> cnn_in(audio.begin(), audio.end()), cnn_out;

    for (uint32_t li = 0; li < hp.num_feat_extract_layers; li++) {
        uint32_t C_out = hp.conv_dim[li];
        uint32_t K = hp.conv_kernel[li];
        uint32_t stride = hp.conv_stride[li];
        uint32_t L_out = (L_cur - K) / stride + 1;

        cnn_out.resize(C_out * L_out);

        // Dequantise conv weight to F32 if needed
        std::vector<float> w_buf;
        const float* wdata;
        if (m.cnn[li].conv_w->type == GGML_TYPE_F16) {
            size_t n = C_out * C_cur * K;
            w_buf.resize(n);
            const ggml_fp16_t* w16 = (const ggml_fp16_t*)m.cnn[li].conv_w->data;
            for (size_t i = 0; i < n; i++)
                w_buf[i] = ggml_fp16_to_fp32(w16[i]);
            wdata = w_buf.data();
        } else {
            wdata = (const float*)m.cnn[li].conv_w->data;
        }
        const float* bdata = m.cnn[li].conv_b ? (const float*)m.cnn[li].conv_b->data : nullptr;
        const float* nw = m.cnn[li].has_norm ? (const float*)m.cnn[li].norm_w->data : nullptr;
        const float* nb = m.cnn[li].has_norm ? (const float*)m.cnn[li].norm_b->data : nullptr;

        conv1d(cnn_in.data(), wdata, bdata, cnn_out.data(), (int)C_cur, (int)C_out, (int)K, (int)stride, (int)L_cur,
               /*left_pad=*/0);

        if (m.cnn[li].has_norm) {
            if (hp.feat_extract_norm_type == 1)
                layer_norm_cf(cnn_out.data(), cnn_out.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
            else
                instance_norm_1d(cnn_out.data(), cnn_out.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
        }
        for (float& v : cnn_out)
            v = gelu(v);

        std::swap(cnn_in, cnn_out);
        C_cur = C_out;
        L_cur = L_out;
    }
    // cnn_in = [C_last=512, T]

    int T = (int)L_cnn;
    int C_cnn = (int)C_cur;

    // Transpose to [T, C_cnn]
    std::vector<float> feat(T * C_cnn);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C_cnn; c++)
            feat[t * C_cnn + c] = cnn_in[c * T + t];

    // ------------------------------------------------------------------
    // 2. Feature projection: LayerNorm(C_cnn) → Linear(C_cnn → H)
    // ------------------------------------------------------------------
    int H = (int)hp.hidden_size;

    layer_norm(feat.data(), feat.data(), (const float*)m.fp_ln_w->data, (const float*)m.fp_ln_b->data, T, C_cnn,
               hp.layer_norm_eps);

    std::vector<float> hidden(T * H);
    std::vector<uint8_t> scratch;

    ggml_linear_f32(scratch, m.fp_w, (const float*)m.fp_b->data, feat.data(), hidden.data(), C_cnn, H, T, n_threads);

    // ------------------------------------------------------------------
    // 3. Positional conv embedding (grouped conv1d, added as residual)
    // ------------------------------------------------------------------
    {
        std::vector<float> hcf(H * T);
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                hcf[h * T + t] = hidden[t * H + h];

        std::vector<float> pos_out(H * T);
        int K_pos = (int)hp.num_conv_pos_embeddings;
        int G_pos = (int)hp.num_conv_pos_embedding_groups;

        std::vector<float> pw_buf;
        const float* pw;
        size_t pos_w_n = (size_t)H * (H / G_pos) * K_pos;
        if (m.pos_conv_w->type == GGML_TYPE_F16) {
            pw_buf.resize(pos_w_n);
            const ggml_fp16_t* p16 = (const ggml_fp16_t*)m.pos_conv_w->data;
            for (size_t i = 0; i < pos_w_n; i++)
                pw_buf[i] = ggml_fp16_to_fp32(p16[i]);
            pw = pw_buf.data();
        } else if (m.pos_conv_w->type == GGML_TYPE_F32) {
            pw = (const float*)m.pos_conv_w->data;
        } else {
            fprintf(stderr, "[wav2vec2] pos_conv.weight has unsupported type %d\n", (int)m.pos_conv_w->type);
            return {};
        }
        const float* pb = (const float*)m.pos_conv_b->data;

        grouped_conv1d_same(hcf.data(), pw, pb, pos_out.data(), H, H, K_pos, G_pos, T);

        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                hidden[t * H + h] += gelu(pos_out[h * T + t]);
    }


    // ------------------------------------------------------------------
    // 4. Transformer encoder layers (pre-norm / stable layer norm)
    // ------------------------------------------------------------------
    int n_heads = (int)hp.num_attention_heads;
    int head_dim = H / n_heads;
    float scale = 1.f / sqrtf((float)head_dim);
    int I = (int)hp.intermediate_size;

    std::vector<float> normed(T * H);
    std::vector<float> Q_buf(T * H), K_buf(T * H), V_buf(T * H);
    std::vector<float> attn_out(T * H);
    std::vector<float> ffn_mid(T * I);
    std::vector<float> ffn_out(T * H);
    std::vector<float> scores(T);

    for (uint32_t li = 0; li < hp.num_hidden_layers; li++) {
        const auto& e = m.enc[li];

        layer_norm(hidden.data(), normed.data(), (const float*)e.ln1_w->data, (const float*)e.ln1_b->data, T, H,
                   hp.layer_norm_eps);

        ggml_linear_f32(scratch, e.q_w, (const float*)e.q_b->data, normed.data(), Q_buf.data(), H, H, T, n_threads);
        ggml_linear_f32(scratch, e.k_w, (const float*)e.k_b->data, normed.data(), K_buf.data(), H, H, T, n_threads);
        ggml_linear_f32(scratch, e.v_w, (const float*)e.v_b->data, normed.data(), V_buf.data(), H, H, T, n_threads);

        std::fill(attn_out.begin(), attn_out.end(), 0.f);
        for (int h = 0; h < n_heads; h++) {
            int off = h * head_dim;
            for (int tq = 0; tq < T; tq++) {
                for (int tk = 0; tk < T; tk++) {
                    float dot = 0.f;
                    const float* q = Q_buf.data() + tq * H + off;
                    const float* k = K_buf.data() + tk * H + off;
                    for (int d = 0; d < head_dim; d++)
                        dot += q[d] * k[d];
                    scores[tk] = dot * scale;
                }
                softmax(scores.data(), T);

                float* ao = attn_out.data() + tq * H + off;
                for (int tv = 0; tv < T; tv++) {
                    const float* v = V_buf.data() + tv * H + off;
                    float s = scores[tv];
                    for (int d = 0; d < head_dim; d++)
                        ao[d] += s * v[d];
                }
            }
        }

        ggml_linear_f32(scratch, e.o_w, (const float*)e.o_b->data, attn_out.data(), normed.data(), H, H, T, n_threads);
        for (int i = 0; i < T * H; i++)
            hidden[i] += normed[i];

        layer_norm(hidden.data(), normed.data(), (const float*)e.ln2_w->data, (const float*)e.ln2_b->data, T, H,
                   hp.layer_norm_eps);

        ggml_linear_f32(scratch, e.fc1_w, (const float*)e.fc1_b->data, normed.data(), ffn_mid.data(), H, I, T,
                        n_threads);
        for (int i = 0; i < T * I; i++)
            ffn_mid[i] = gelu(ffn_mid[i]);

        ggml_linear_f32(scratch, e.fc2_w, (const float*)e.fc2_b->data, ffn_mid.data(), ffn_out.data(), I, H, T,
                        n_threads);
        for (int i = 0; i < T * H; i++)
            hidden[i] += ffn_out[i];
    }

    // ------------------------------------------------------------------
    // 5. Encoder global LayerNorm
    // ------------------------------------------------------------------
    layer_norm(hidden.data(), hidden.data(), (const float*)m.enc_ln_w->data, (const float*)m.enc_ln_b->data, T, H,
               hp.layer_norm_eps);

    // ------------------------------------------------------------------
    // 6. LM head: Linear(H → V) — raw logits (no softmax)
    // ------------------------------------------------------------------
    int V = (int)hp.vocab_size;
    std::vector<float> logits(T * V);
    ggml_linear_f32(scratch, m.lm_w, (const float*)m.lm_b->data, hidden.data(), logits.data(), H, V, T, n_threads);

    return logits;
}

// ===========================================================================
// Greedy CTC decode
// ===========================================================================

std::string wav2vec2_greedy_decode(const wav2vec2_model& m, const float* logits, int T) {
    const auto& hp = m.hparams;
    int V = (int)hp.vocab_size;

    std::string result;
    int prev_id = -1;
    for (int t = 0; t < T; t++) {
        const float* lv = logits + t * V;
        int best_id = (int)(std::max_element(lv, lv + V) - lv);
        if (best_id != prev_id) {
            if (best_id != (int)hp.pad_token_id && best_id < (int)m.vocab.size()) {
                const std::string& tok = m.vocab[best_id];
                if (tok == "|")
                    result += ' ';
                else if (tok != "<unk>" && tok != "<s>" && tok != "</s>")
                    result += tok;
            }
            prev_id = best_id;
        }
    }
    // Trim leading/trailing spaces
    auto lo = result.find_first_not_of(' ');
    auto hi = result.find_last_not_of(' ');
    return (lo == std::string::npos) ? "" : result.substr(lo, hi - lo + 1);
}
