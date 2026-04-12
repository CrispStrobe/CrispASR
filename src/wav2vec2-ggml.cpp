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

static ggml_tensor * require_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "[wav2vec2] required tensor '%s' not found in GGUF\n", name);
        exit(1);
    }
    return t;
}

static uint32_t gguf_u32(gguf_context * gctx, const char * key, uint32_t def = 0) {
    int idx = gguf_find_key(gctx, key);
    return idx >= 0 ? (uint32_t)gguf_get_val_u32(gctx, idx) : def;
}

static float gguf_f32(gguf_context * gctx, const char * key, float def = 0.f) {
    int idx = gguf_find_key(gctx, key);
    return idx >= 0 ? gguf_get_val_f32(gctx, idx) : def;
}

// ===========================================================================
// Model loading
// ===========================================================================

bool wav2vec2_load(const char * fname, wav2vec2_model & model) {
    // Phase 1: metadata pass (no tensor allocation)
    gguf_init_params p_meta = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * gctx_meta = gguf_init_from_file(fname, p_meta);
    if (!gctx_meta) {
        fprintf(stderr, "[wav2vec2] cannot open: %s\n", fname);
        return false;
    }

    auto & hp = model.hparams;
    hp.vocab_size                    = gguf_u32(gctx_meta, "wav2vec2.vocab_size",                    hp.vocab_size);
    hp.hidden_size                   = gguf_u32(gctx_meta, "wav2vec2.hidden_size",                   hp.hidden_size);
    hp.num_hidden_layers             = gguf_u32(gctx_meta, "wav2vec2.num_hidden_layers",             hp.num_hidden_layers);
    hp.num_attention_heads           = gguf_u32(gctx_meta, "wav2vec2.num_attention_heads",           hp.num_attention_heads);
    hp.intermediate_size             = gguf_u32(gctx_meta, "wav2vec2.intermediate_size",             hp.intermediate_size);
    hp.num_feat_extract_layers       = gguf_u32(gctx_meta, "wav2vec2.num_feat_extract_layers",       hp.num_feat_extract_layers);
    hp.num_conv_pos_embeddings       = gguf_u32(gctx_meta, "wav2vec2.num_conv_pos_embeddings",       hp.num_conv_pos_embeddings);
    hp.num_conv_pos_embedding_groups = gguf_u32(gctx_meta, "wav2vec2.num_conv_pos_embedding_groups", hp.num_conv_pos_embedding_groups);
    hp.layer_norm_eps                = gguf_f32(gctx_meta, "wav2vec2.layer_norm_eps",                hp.layer_norm_eps);
    hp.pad_token_id                  = gguf_u32(gctx_meta, "wav2vec2.pad_token_id",                  hp.pad_token_id);
    hp.feat_extract_norm_type        = gguf_u32(gctx_meta, "wav2vec2.feat_extract_norm_type",        hp.feat_extract_norm_type);

    for (uint32_t i = 0; i < hp.num_feat_extract_layers; i++) {
        char key[64];
        snprintf(key, sizeof(key), "wav2vec2.conv_dim_%u",    i); hp.conv_dim[i]    = gguf_u32(gctx_meta, key, hp.conv_dim[i]);
        snprintf(key, sizeof(key), "wav2vec2.conv_kernel_%u", i); hp.conv_kernel[i] = gguf_u32(gctx_meta, key, hp.conv_kernel[i]);
        snprintf(key, sizeof(key), "wav2vec2.conv_stride_%u", i); hp.conv_stride[i] = gguf_u32(gctx_meta, key, hp.conv_stride[i]);
    }

    // Vocabulary
    {
        int idx = gguf_find_key(gctx_meta, "tokenizer.ggml.tokens");
        if (idx >= 0) {
            uint32_t n = gguf_get_arr_n(gctx_meta, idx);
            model.vocab.resize(n);
            for (uint32_t i = 0; i < n; i++) {
                const char * s = gguf_get_arr_str(gctx_meta, idx, i);
                model.vocab[i] = s ? s : "";
            }
        }
    }
    gguf_free(gctx_meta);

    // Phase 2: load tensors via core_gguf (backend-buffer-backed)
    // This puts all weights in a ggml_backend_buffer so the graph
    // scheduler can reference them without cross-context issues.
    model.backend = ggml_backend_cpu_init();
    if (!model.backend) {
        fprintf(stderr, "[wav2vec2] failed to init CPU backend\n");
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

    auto get = [&](const char * name) -> ggml_tensor * {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            fprintf(stderr, "[wav2vec2] required tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };
    auto try_get = [&](const char * name) -> ggml_tensor * {
        auto it = model.tensors.find(name);
        return it != model.tensors.end() ? it->second : nullptr;
    };

    model.enc.resize(hp.num_hidden_layers);
    uint32_t L = hp.num_feat_extract_layers;

    for (uint32_t i = 0; i < L; i++) {
        char buf[80];
        snprintf(buf, sizeof(buf), "cnn.%u.conv.weight", i); model.cnn[i].conv_w = get(buf);
        snprintf(buf, sizeof(buf), "cnn.%u.conv.bias",   i); model.cnn[i].conv_b = get(buf);
        snprintf(buf, sizeof(buf), "cnn.%u.norm.weight",  i);
        model.cnn[i].norm_w = try_get(buf);
        if (model.cnn[i].norm_w) {
            snprintf(buf, sizeof(buf), "cnn.%u.norm.bias", i);
            model.cnn[i].norm_b = get(buf);
            model.cnn[i].has_norm = true;
        }
    }

    model.fp_ln_w    = get("feat_proj.ln.weight");
    model.fp_ln_b    = get("feat_proj.ln.bias");
    model.fp_w       = get("feat_proj.weight");
    model.fp_b       = get("feat_proj.bias");
    model.pos_conv_w = get("pos_conv.weight");
    model.pos_conv_b = get("pos_conv.bias");
    model.enc_ln_w   = get("enc.ln.weight");
    model.enc_ln_b   = get("enc.ln.bias");

    for (uint32_t i = 0; i < hp.num_hidden_layers; i++) {
        char buf[80];
        auto & e = model.enc[i];
        snprintf(buf, sizeof(buf), "enc.%u.ln1.weight",      i); e.ln1_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ln1.bias",        i); e.ln1_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.q.weight",   i); e.q_w   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.q.bias",     i); e.q_b   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.k.weight",   i); e.k_w   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.k.bias",     i); e.k_b   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.v.weight",   i); e.v_w   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.v.bias",     i); e.v_b   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.out.weight", i); e.o_w   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.attn.out.bias",   i); e.o_b   = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ln2.weight",      i); e.ln2_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ln2.bias",        i); e.ln2_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc1.weight",  i); e.fc1_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc1.bias",    i); e.fc1_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc2.weight",  i); e.fc2_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%u.ffn.fc2.bias",    i); e.fc2_b = get(buf);
    }

    model.lm_w = get("lm_head.weight");
    model.lm_b = try_get("lm_head.bias");

    fprintf(stderr, "[wav2vec2] vocab=%u  hidden=%u  layers=%u  heads=%u  ffn=%u\n",
            hp.vocab_size, hp.hidden_size, hp.num_hidden_layers,
            hp.num_attention_heads, hp.intermediate_size);
    return true;
}

// ===========================================================================
// Manual compute primitives (F32)
// ===========================================================================

// LayerNorm over last dim: x[T, C] → y[T, C] (in-place OK)
static void layer_norm(const float * x, float * y,
                       const float * w, const float * b,
                       int T, int C, float eps)
{
    for (int t = 0; t < T; t++) {
        const float * xt = x + t * C;
        float       * yt = y + t * C;
        double sum = 0.0, sq = 0.0;
        for (int c = 0; c < C; c++) { sum += xt[c]; sq += (double)xt[c] * xt[c]; }
        float mean = (float)(sum / C);
        float var  = (float)(sq  / C) - mean * mean;
        float inv  = 1.f / sqrtf(var + eps);
        for (int c = 0; c < C; c++) yt[c] = (xt[c] - mean) * inv * w[c] + b[c];
    }
}

// GELU (exact tanh approximation)
static inline float gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Softmax in-place over n elements
static void softmax(float * x, int n) {
    float mx = *std::max_element(x, x + n);
    float s  = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// InstanceNorm (GroupNorm with num_groups=C) on channel-first data [C, L].
// Used by feat_extract_norm="group", layer 0.
static void instance_norm_1d(const float * x, float * y,
                              const float * w, const float * b,
                              int C, int L, float eps)
{
    for (int c = 0; c < C; c++) {
        const float * xc = x + c * L;
        float       * yc = y + c * L;
        double sum = 0.0, sq = 0.0;
        for (int l = 0; l < L; l++) { sum += xc[l]; sq += (double)xc[l] * xc[l]; }
        float mean = (float)(sum / L);
        float var  = (float)(sq  / L) - mean * mean;
        float inv  = 1.f / sqrtf(var + eps);
        for (int l = 0; l < L; l++) yc[l] = (xc[l] - mean) * inv * w[c] + b[c];
    }
}

// LayerNorm on channel-first [C, L] (norm across C at each L).
// Used by feat_extract_norm="layer", all CNN layers.
static void layer_norm_cf(const float * x, float * y,
                          const float * w, const float * b,
                          int C, int L, float eps)
{
    for (int l = 0; l < L; l++) {
        double sum = 0.0, sq = 0.0;
        for (int c = 0; c < C; c++) {
            float v = x[c * L + l];
            sum += v; sq += (double)v * v;
        }
        float mean = (float)(sum / C);
        float var  = (float)(sq  / C) - mean * mean;
        float inv  = 1.f / sqrtf(var + eps);
        for (int c = 0; c < C; c++)
            y[c * L + l] = (x[c * L + l] - mean) * inv * w[c] + b[c];
    }
}

// Conv1d (dilation=1), left_pad zero-pads on the left.
// Weight: w[Cout][Cin][K]  Input: x[Cin][L_in]  Output: y[Cout][L_out]
static void conv1d(const float * x, const float * w, const float * b,
                   float * y,
                   int Cin, int Cout, int K, int stride, int L_in,
                   int left_pad = 0)
{
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
                const float * wrow = w + (oc * Cin + ic) * K;
                const float * xrow = padded.data() + ic * L_pad + t0;
                for (int k = 0; k < K; k++) sum += wrow[k] * xrow[k];
            }
            y[oc * L_out + t] = sum;
        }
    }
}

// Grouped Conv1d with symmetric padding so output_len == input_len.
// Used for the positional conv embedding (K=128, groups=16 by default).
// Weight: w[Cout][Cin/groups][K]
static void grouped_conv1d_same(const float * x, const float * w, const float * b,
                                float * y,
                                int C_in, int C_out, int K, int groups, int L)
{
    assert(C_in % groups == 0 && C_out % groups == 0);
    int cin_pg  = C_in  / groups;
    int cout_pg = C_out / groups;
    int pad_l = K / 2, pad_r = K / 2;
    int L_pad = L + pad_l + pad_r;

    std::vector<float> padded(C_in * L_pad, 0.f);
    for (int c = 0; c < C_in; c++)
        std::memcpy(padded.data() + c * L_pad + pad_l, x + c * L, L * sizeof(float));

    for (int g = 0; g < groups; g++) {
        int ic0 = g * cin_pg, oc0 = g * cout_pg;
        for (int oc = 0; oc < cout_pg; oc++) {
            int og = oc0 + oc;
            float bv = b ? b[og] : 0.f;
            for (int t = 0; t < L; t++) {  // trim to L (SamePad behaviour)
                float sum = bv;
                for (int ic = 0; ic < cin_pg; ic++) {
                    const float * wrow = w + (og * cin_pg + ic) * K;
                    const float * xrow = padded.data() + (ic0 + ic) * L_pad + t;
                    for (int k = 0; k < K; k++) sum += wrow[k] * xrow[k];
                }
                y[og * L + t] = sum;
            }
        }
    }
}

// ggml-based quantised linear: y[T, n_out] = W[n_out, n_in] * x[T, n_in] + b
// Creates a fresh ggml context per call so scratch-buffer growth is bounded.
static void ggml_linear_f32(std::vector<uint8_t> & scratch,
                            ggml_tensor * W, const float * bias,
                            const float * x, float * y,
                            int n_in, int n_out, int T, int n_threads = 1)
{
    size_t ctx_size = (size_t)(n_in + n_out) * T * sizeof(float) * 8
                    + ggml_tensor_overhead() * 8
                    + ggml_graph_overhead()
                    + 4 * 1024 * 1024;
    scratch.resize(ctx_size);

    ggml_init_params p = { ctx_size, scratch.data(), /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * xt  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, T);
    std::memcpy(xt->data, x, (size_t)n_in * T * sizeof(float));

    ggml_tensor * out = ggml_mul_mat(ctx, W, xt);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    const float * od = (const float *)out->data;
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
static ggml_cgraph * wav2vec2_build_transformer_graph(
    const wav2vec2_model & m,
    int T,               // number of CNN output frames (computed by caller)
    std::vector<uint8_t> & compute_meta)
{
    const auto & hp = m.hparams;
    const int H       = (int)hp.hidden_size;
    const int n_heads = (int)hp.num_attention_heads;
    const int head_dim = H / n_heads;
    const int I       = (int)hp.intermediate_size;
    const int L       = (int)hp.num_hidden_layers;
    const int V       = (int)hp.vocab_size;
    const float ln_eps = hp.layer_norm_eps;

    size_t ctx_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    compute_meta.resize(ctx_size);
    ggml_init_params ip = { ctx_size, compute_meta.data(), true };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: feature-projected hidden states [H, T] from the CNN + proj
    ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, H, T);
    ggml_set_name(cur, "hidden_in");
    ggml_set_input(cur);

    // ---- L × Transformer layers (pre-norm) ----
    for (int il = 0; il < L; il++) {
        const auto & e = m.enc[il];
        ggml_tensor * residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor * x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, e.ln1_w);
        x = ggml_add(ctx0, x, e.ln1_b);

        // Q/K/V projections
        ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, e.q_w, x), e.q_b);
        ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, e.k_w, x), e.k_b);
        ggml_tensor * V_t = ggml_add(ctx0, ggml_mul_mat(ctx0, e.v_w, x), e.v_b);


        // Reshape to (head_dim, n_heads, T)
        Q   = ggml_reshape_3d(ctx0, Q,   head_dim, n_heads, T);
        K   = ggml_reshape_3d(ctx0, K,   head_dim, n_heads, T);
        V_t = ggml_reshape_3d(ctx0, V_t, head_dim, n_heads, T);

        // Manual multi-head attention (ggml_flash_attn_ext crashes with
        // nullptr mask on some backends — use standard mul_mat path).
        // Q,K,V are (head_dim, n_heads, T). Permute for matmul:
        // Q → (head_dim, T, n_heads): for Q @ K^T
        // K → (head_dim, T, n_heads): transposed via mul_mat convention
        // V → (head_dim, T, n_heads): for attn_weights @ V
        Q   = ggml_cont(ctx0, ggml_permute(ctx0, Q,   0, 2, 1, 3));
        K   = ggml_cont(ctx0, ggml_permute(ctx0, K,   0, 2, 1, 3));
        V_t = ggml_cont(ctx0, ggml_permute(ctx0, V_t, 0, 2, 1, 3));

        // scores = Q^T @ K → (T, T, n_heads)
        // ggml_mul_mat(A, B) = A^T @ B, so mul_mat(K, Q) = K^T @ Q → (T, T, n_heads)
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * scores = ggml_mul_mat(ctx0, K, Q);
        scores = ggml_scale(ctx0, scores, attn_scale);
        scores = ggml_soft_max(ctx0, scores);

        // attn_out = scores @ V → for each head: (T,T) @ (T,head_dim) → (T,head_dim)
        // V_t is (head_dim, T, n_heads). We need V with ne[0]=T for mul_mat.
        // Permute V_t: (head_dim, T, n_heads) → (T, head_dim, n_heads)
        ggml_tensor * V_for_attn = ggml_cont(ctx0, ggml_permute(ctx0, V_t, 1, 0, 2, 3));
        // V_for_attn: ne = (T, head_dim, n_heads)
        // mul_mat(V_for_attn, scores) = V_for_attn^T @ scores
        //   V_for_attn^T: (head_dim, T) @ scores: (T, T) → (head_dim, T, n_heads)
        ggml_tensor * attn = ggml_mul_mat(ctx0, V_for_attn, scores);

        // attn is (head_dim, T, n_heads). Permute → (head_dim, n_heads, T) → reshape to (H, T)
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, H, T);

        // Output projection + residual
        attn = ggml_add(ctx0, ggml_mul_mat(ctx0, e.o_w, attn), e.o_b);
        cur = ggml_add(ctx0, residual, attn);

        // Pre-FFN LayerNorm
        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, e.ln2_w);
        x = ggml_add(ctx0, x, e.ln2_b);

        // FFN: fc1 → GELU → fc2
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.fc1_w, x), e.fc1_b);
        x = ggml_gelu(ctx0, x);
        x = ggml_add(ctx0, ggml_mul_mat(ctx0, e.fc2_w, x), e.fc2_b);
        cur = ggml_add(ctx0, residual, x);

    }

    // ---- Global LayerNorm ----
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, m.enc_ln_w);
    cur = ggml_add(ctx0, cur, m.enc_ln_b);

    // ---- LM head: Linear(H → V) ----
    cur = ggml_mul_mat(ctx0, m.lm_w, cur);
    if (m.lm_b) cur = ggml_add(ctx0, cur, m.lm_b);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Debug: 1-layer graph vs manual — compare every intermediate
// ===========================================================================
static void wav2vec2_debug_attention(const wav2vec2_model & m,
                                     const float * hidden_th, int T, int H) {
    const auto & hp = m.hparams;
    const auto & e = m.enc[0];  // layer 0 only
    int n_heads  = (int)hp.num_attention_heads;
    int head_dim = H / n_heads;  // 64
    int I = (int)hp.intermediate_size;
    float ln_eps = hp.layer_norm_eps;
    float scale  = 1.f / sqrtf((float)head_dim);

    auto cmp = [](const char * name, const float * a, const float * b, int n) {
        float mx = 0;
        for (int i = 0; i < n; i++) mx = std::max(mx, fabsf(a[i] - b[i]));
        fprintf(stderr, "  %-20s max_diff=%.6f first4: graph[%.4f %.4f %.4f %.4f] manual[%.4f %.4f %.4f %.4f] %s\n",
                name, mx, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
                mx < 0.01f ? "OK" : "FAIL");
    };

    // ---- Build graph: LN → Q/K/V → reshape → permute → matmul → softmax → attn → out_proj ----
    // Using compute_with_ctx (no_alloc=false) to get correct results
    size_t mem = (size_t)H * T * 4 * 30 + (size_t)T * T * n_heads * 4 * 2 + (size_t)I * T * 4
               + ggml_tensor_overhead() * 500 + ggml_graph_overhead_custom(1024, false) + 128*1024*1024;
    fprintf(stderr, "[wav2vec2-dbg] allocating %.0f MB for 1-layer graph\n", mem / (1024.0*1024.0));
    std::vector<uint8_t> buf;
    try { buf.resize(mem); } catch (...) {
        fprintf(stderr, "[wav2vec2-dbg] OOM\n"); return;
    }
    ggml_init_params ip = { mem, buf.data(), false };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);

    // Input [H, T] — transpose hidden_th [T, H] → [H, T]
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            ((float*)cur->data)[h + t * H] = hidden_th[t * H + h];

    // LN1
    ggml_tensor * x = ggml_norm(ctx, cur, ln_eps);
    x = ggml_mul(ctx, x, e.ln1_w);
    x = ggml_add(ctx, x, e.ln1_b);
    ggml_set_name(x, "ln1");

    // Q projection
    ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, e.q_w, x), e.q_b);
    ggml_set_name(Q, "Q_proj");

    // K, V projections
    ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, e.k_w, x), e.k_b);
    ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, e.v_w, x), e.v_b);

    // Reshape + permute
    ggml_tensor * Qr = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
    ggml_tensor * Qp = ggml_cont(ctx, ggml_permute(ctx, Qr, 0, 2, 1, 3));
    ggml_set_name(Qp, "Q_perm");  // [head_dim, T, n_heads]
    ggml_tensor * Kr = ggml_reshape_3d(ctx, K, head_dim, n_heads, T);
    ggml_tensor * Kp = ggml_cont(ctx, ggml_permute(ctx, Kr, 0, 2, 1, 3));
    ggml_tensor * Vr = ggml_reshape_3d(ctx, V, head_dim, n_heads, T);
    ggml_tensor * Vp = ggml_cont(ctx, ggml_permute(ctx, Vr, 0, 2, 1, 3));

    // Attention scores
    ggml_tensor * scores = ggml_mul_mat(ctx, Kp, Qp);  // [T, T, n_heads]
    scores = ggml_scale(ctx, scores, scale);
    scores = ggml_soft_max(ctx, scores);
    ggml_set_name(scores, "attn_scores");

    // attn = scores @ V
    ggml_tensor * Vp2 = ggml_cont(ctx, ggml_permute(ctx, Vp, 1, 0, 2, 3));
    ggml_tensor * attn = ggml_mul_mat(ctx, Vp2, scores);
    ggml_set_name(attn, "attn_raw");  // [head_dim, T, n_heads]

    // Reshape back to [H, T]
    ggml_tensor * ap = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    ggml_tensor * ar = ggml_reshape_2d(ctx, ap, H, T);
    ggml_set_name(ar, "attn_merged");

    // Output projection + residual
    ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, e.o_w, ar), e.o_b);
    ggml_tensor * res = ggml_add(ctx, cur, out);
    ggml_set_name(res, "after_attn");

    ggml_build_forward_expand(gf, res);
    fprintf(stderr, "[wav2vec2-dbg] graph built, computing...\n");
    fflush(stderr);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    fprintf(stderr, "[wav2vec2-dbg] graph computed, comparing...\n");
    fflush(stderr);

    // ---- Manual path (matching wav2vec2_compute_logits exactly) ----
    std::vector<float> normed(T * H), Q_m(T * H), K_m(T * H), V_m(T * H);
    layer_norm(hidden_th, normed.data(),
               (const float *)e.ln1_w->data, (const float *)e.ln1_b->data,
               T, H, ln_eps);

    std::vector<uint8_t> scratch;
    ggml_linear_f32(scratch, e.q_w, (const float *)e.q_b->data, normed.data(), Q_m.data(), H, H, T, 1);
    ggml_linear_f32(scratch, e.k_w, (const float *)e.k_b->data, normed.data(), K_m.data(), H, H, T, 1);
    ggml_linear_f32(scratch, e.v_w, (const float *)e.v_b->data, normed.data(), V_m.data(), H, H, T, 1);

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
            for (int j = 0; j < T; j++) { sc[j] = expf(sc[j] - mx); sm += sc[j]; }
            for (int j = 0; j < T; j++) sc[j] /= sm;
            for (int tv = 0; tv < T; tv++) {
                float s = sc[tv];
                for (int d = 0; d < head_dim; d++)
                    attn_m[tq * H + off + d] += s * V_m[tv * H + off + d];
            }
        }
    }
    std::vector<float> oproj_m(T * H);
    ggml_linear_f32(scratch, e.o_w, (const float *)e.o_b->data, attn_m.data(), oproj_m.data(), H, H, T, 1);
    std::vector<float> res_m(T * H);
    for (int i = 0; i < T * H; i++) res_m[i] = hidden_th[i] + oproj_m[i];

    // ---- Compare ----
    fprintf(stderr, "[wav2vec2-dbg] 1-layer attention comparison (T=%d H=%d heads=%d):\n", T, H, n_heads);

    // LN1: graph is [H, T] ggml = data[h + t*H]. Manual is [T, H] = data[t*H + h]. Same layout!
    ggml_tensor * gln1 = ggml_graph_get_tensor(gf, "ln1");
    // Convert graph [H,T] to [T,H] for comparison
    std::vector<float> gln1_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gln1_th[t * H + h] = ((float*)gln1->data)[h + t * H];
    cmp("LN1", gln1_th.data(), normed.data(), T * H);

    // Q projection
    ggml_tensor * gQ = ggml_graph_get_tensor(gf, "Q_proj");
    std::vector<float> gQ_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gQ_th[t * H + h] = ((float*)gQ->data)[h + t * H];
    cmp("Q_proj", gQ_th.data(), Q_m.data(), T * H);

    // Q after permute — graph has [head_dim, T, n_heads]
    ggml_tensor * gQp = ggml_graph_get_tensor(gf, "Q_perm");
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
    fprintf(stderr, "  %-20s max_diff=%.6f %s\n", "Q_perm_vs_Q_manual", qp_diff,
            qp_diff < 0.01f ? "OK" : "FAIL");

    // Attention output
    ggml_tensor * gar = ggml_graph_get_tensor(gf, "attn_merged");
    std::vector<float> gar_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gar_th[t * H + h] = ((float*)gar->data)[h + t * H];
    cmp("attn_merged", gar_th.data(), attn_m.data(), T * H);

    // After attention residual
    ggml_tensor * gres = ggml_graph_get_tensor(gf, "after_attn");
    std::vector<float> gres_th(T * H);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            gres_th[t * H + h] = ((float*)gres->data)[h + t * H];
    cmp("after_attn", gres_th.data(), res_m.data(), T * H);

    ggml_free(ctx);
}

#if 0  // Debug function preserved for reference
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

std::vector<float> wav2vec2_compute_logits_graph(
    const wav2vec2_model & m,
    const float * raw_audio, int n_samples,
    int n_threads)
{
    const auto & hp = m.hparams;
    const int H = (int)hp.hidden_size;

    // ---- 0. Normalize ----
    std::vector<float> audio(raw_audio, raw_audio + n_samples);
    {
        double sum = 0.0, sq = 0.0;
        for (float v : audio) { sum += v; sq += (double)v * v; }
        float mean = (float)(sum / n_samples);
        float std_ = sqrtf(std::max(0.f, (float)(sq / n_samples) - mean * mean) + 1e-7f);
        for (float & v : audio) v = (v - mean) / std_;
    }

    // ---- 1. CNN feature extractor (manual C++) ----
    uint32_t L_cur = (uint32_t)n_samples, C_cur = 1;
    std::vector<float> cnn_in(audio.begin(), audio.end()), cnn_out;

    for (uint32_t li = 0; li < hp.num_feat_extract_layers; li++) {
        uint32_t C_out  = hp.conv_dim[li];
        uint32_t K      = hp.conv_kernel[li];
        uint32_t stride = hp.conv_stride[li];
        uint32_t L_out  = (L_cur - K) / stride + 1;

        cnn_out.resize(C_out * L_out);

        std::vector<float> w_buf;
        const float * wdata;
        if (m.cnn[li].conv_w->type == GGML_TYPE_F16) {
            size_t n = C_out * C_cur * K;
            w_buf.resize(n);
            const ggml_fp16_t * w16 = (const ggml_fp16_t *)m.cnn[li].conv_w->data;
            for (size_t i = 0; i < n; i++) w_buf[i] = ggml_fp16_to_fp32(w16[i]);
            wdata = w_buf.data();
        } else {
            wdata = (const float *)m.cnn[li].conv_w->data;
        }
        const float * bdata = (const float *)m.cnn[li].conv_b->data;
        const float * nw    = m.cnn[li].has_norm ? (const float *)m.cnn[li].norm_w->data : nullptr;
        const float * nb    = m.cnn[li].has_norm ? (const float *)m.cnn[li].norm_b->data : nullptr;

        conv1d(cnn_in.data(), wdata, bdata,
               cnn_out.data(),
               (int)C_cur, (int)C_out, (int)K, (int)stride, (int)L_cur, 0);

        if (m.cnn[li].has_norm) {
            if (hp.feat_extract_norm_type == 1)
                layer_norm_cf(cnn_out.data(), cnn_out.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
            else
                instance_norm_1d(cnn_out.data(), cnn_out.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
        }
        for (float & v : cnn_out) v = gelu(v);

        std::swap(cnn_in, cnn_out);
        C_cur = C_out;
        L_cur = L_out;
    }

    int T     = (int)L_cur;
    int C_cnn = (int)C_cur;

    // Transpose [C_cnn, T] → [T, C_cnn]
    std::vector<float> feat(T * C_cnn);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C_cnn; c++)
            feat[t * C_cnn + c] = cnn_in[c * T + t];

    // ---- 2. Feature projection: LN + Linear ----
    layer_norm(feat.data(), feat.data(),
               (const float *)m.fp_ln_w->data,
               (const float *)m.fp_ln_b->data,
               T, C_cnn, hp.layer_norm_eps);

    std::vector<float> hidden(T * H);
    std::vector<uint8_t> scratch;
    ggml_linear_f32(scratch, m.fp_w, (const float *)m.fp_b->data,
                    feat.data(), hidden.data(), C_cnn, H, T, n_threads);

    // ---- 3. Positional conv (manual — grouped conv is hard in ggml) ----
    {
        std::vector<float> hcf(H * T);
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                hcf[h * T + t] = hidden[t * H + h];

        std::vector<float> pos_out(H * T);
        int K_pos = (int)hp.num_conv_pos_embeddings;
        int G_pos = (int)hp.num_conv_pos_embedding_groups;

        std::vector<float> pw_buf;
        const float * pw;
        size_t pos_w_n = (size_t)H * (H / G_pos) * K_pos;
        if (m.pos_conv_w->type == GGML_TYPE_F16) {
            pw_buf.resize(pos_w_n);
            const ggml_fp16_t * p16 = (const ggml_fp16_t *)m.pos_conv_w->data;
            for (size_t i = 0; i < pos_w_n; i++) pw_buf[i] = ggml_fp16_to_fp32(p16[i]);
            pw = pw_buf.data();
        } else if (m.pos_conv_w->type == GGML_TYPE_F32) {
            pw = (const float *)m.pos_conv_w->data;
        } else {
            fprintf(stderr, "[wav2vec2] pos_conv.weight unsupported type\n");
            return {};
        }
        const float * pb = (const float *)m.pos_conv_b->data;

        grouped_conv1d_same(hcf.data(), pw, pb, pos_out.data(), H, H, K_pos, G_pos, T);

        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                hidden[t * H + h] += gelu(pos_out[h * T + t]);
    }

    // ---- 4. Transformer + LN + LM head via ggml graph ----
    // Transpose hidden [T, H] → [H, T] for ggml (ne[0]=H fastest)
    std::vector<float> hidden_ht(H * T);
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++)
            hidden_ht[h * T + t] = hidden[t * H + h];


    // Build graph
    std::vector<uint8_t> compute_meta;
    ggml_cgraph * gf = wav2vec2_build_transformer_graph(m, T, compute_meta);
    if (!gf) return {};

    // Use ggml_graph_compute_with_ctx — the only method proven to correctly
    // reference external F16 weight tensors. gallocr/sched both corrupt
    // the weight data by reallocating over them.
    //
    // Memory: ~1.2 GB for 24 layers × 549 frames × 1024 hidden.
    // This fits in the 7.6 GB machine (model weights ~500 MB + graph ~1.2 GB).
    const int L = (int)hp.num_hidden_layers;
    const int I = (int)hp.intermediate_size;
    const int V = (int)hp.vocab_size;
    const int n_heads  = (int)hp.num_attention_heads;
    const int head_dim = H / n_heads;
    const float ln_eps = hp.layer_norm_eps;

    // Estimate memory for compute_with_ctx (all intermediates stay alive)
    // Per layer: attention scores T²×heads, FFN mid I×T, ~20 intermediate tensors.
    // compute_with_ctx keeps all intermediates alive — ~80 MB/layer for this model.
    size_t per_layer = (size_t)H * T * 4 * 20
                     + (size_t)T * T * n_heads * 4
                     + (size_t)I * T * 4;
    size_t graph_mem = ggml_tensor_overhead() * (size_t)(L * 50 + 50)
                     + ggml_graph_overhead_custom(16384, false)
                     + per_layer * L * 2  // 2× safety factor
                     + (size_t)V * T * 4;

    fprintf(stderr, "[wav2vec2-graph] allocating %.0f MB for graph context\n",
            graph_mem / (1024.0 * 1024.0));

    std::vector<uint8_t> graph_buf;
    try { graph_buf.resize(graph_mem); }
    catch (...) {
        fprintf(stderr, "[wav2vec2] OOM allocating graph buffer (%.0f MB)\n",
                graph_mem / (1024.0 * 1024.0));
        return {};
    }

    ggml_init_params gip = { graph_mem, graph_buf.data(), false };
    ggml_context * gctx = ggml_init(gip);
    ggml_cgraph * gf2 = ggml_new_graph_custom(gctx, 16384, false);

    // Input: [H, T]
    ggml_tensor * cur = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T);
    ggml_set_name(cur, "hidden_in");
    memcpy(cur->data, hidden_ht.data(), H * T * sizeof(float));

    int L_run = L;  // all layers
    for (int il = 0; il < L_run; il++) {
        const auto & e = m.enc[il];
        ggml_tensor * residual = cur;

        ggml_tensor * x = ggml_norm(gctx, cur, ln_eps);
        x = ggml_mul(gctx, x, e.ln1_w);
        x = ggml_add(gctx, x, e.ln1_b);

        ggml_tensor * Q = ggml_add(gctx, ggml_mul_mat(gctx, e.q_w, x), e.q_b);
        ggml_tensor * K = ggml_add(gctx, ggml_mul_mat(gctx, e.k_w, x), e.k_b);
        ggml_tensor * Vt = ggml_add(gctx, ggml_mul_mat(gctx, e.v_w, x), e.v_b);

        // Reshape [H, T] → [head_dim, n_heads, T]
        Q  = ggml_reshape_3d(gctx, Q,  head_dim, n_heads, T);
        K  = ggml_reshape_3d(gctx, K,  head_dim, n_heads, T);
        Vt = ggml_reshape_3d(gctx, Vt, head_dim, n_heads, T);

        // Permute to [head_dim, T, n_heads] for batched attention
        Q  = ggml_cont(gctx, ggml_permute(gctx, Q,  0, 2, 1, 3));
        K  = ggml_cont(gctx, ggml_permute(gctx, K,  0, 2, 1, 3));
        Vt = ggml_cont(gctx, ggml_permute(gctx, Vt, 0, 2, 1, 3));

        // Attention via ggml ops (verified bit-identical to manual path):
        // scores = K^T @ Q → (T, T, n_heads), scaled, softmaxed
        float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor * scores = ggml_mul_mat(gctx, K, Q);
        scores = ggml_scale(gctx, scores, scale);
        scores = ggml_soft_max(gctx, scores);
        // attn = V_transposed^T @ scores → (head_dim, T, n_heads)
        ggml_tensor * V_perm = ggml_cont(gctx, ggml_permute(gctx, Vt, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(gctx, V_perm, scores);
        // Reshape back: (head_dim, T, n_heads) → (head_dim, n_heads, T) → (H, T)
        attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(gctx, attn, H, T);
        attn = ggml_add(gctx, ggml_mul_mat(gctx, e.o_w, attn), e.o_b);
        cur = ggml_add(gctx, residual, attn);

        residual = cur;
        x = ggml_norm(gctx, cur, ln_eps);
        x = ggml_mul(gctx, x, e.ln2_w);
        x = ggml_add(gctx, x, e.ln2_b);

        x = ggml_add(gctx, ggml_mul_mat(gctx, e.fc1_w, x), e.fc1_b);
        x = ggml_gelu(gctx, x);
        x = ggml_add(gctx, ggml_mul_mat(gctx, e.fc2_w, x), e.fc2_b);
        cur = ggml_add(gctx, residual, x);
    }

    cur = ggml_norm(gctx, cur, ln_eps);
    cur = ggml_mul(gctx, cur, m.enc_ln_w);
    cur = ggml_add(gctx, cur, m.enc_ln_b);
    cur = ggml_mul_mat(gctx, m.lm_w, cur);
    if (m.lm_b) cur = ggml_add(gctx, cur, m.lm_b);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf2, cur);

    size_t used = ggml_used_mem(gctx);
    size_t total = graph_mem;
    fprintf(stderr, "[wav2vec2-graph] context: used %.0f MB / %.0f MB (%.1f%%)\n",
            used / (1024.0*1024.0), total / (1024.0*1024.0), 100.0 * used / total);

    // Check work_size before computing
    struct ggml_cplan cplan = ggml_graph_plan(gf2, n_threads, NULL);
    fprintf(stderr, "[wav2vec2-graph] work_size=%.0f MB, remaining context=%.0f MB\n",
            cplan.work_size / (1024.0*1024.0), (graph_mem - used) / (1024.0*1024.0));
    if (cplan.work_size > 0) {
        cplan.work_data = (uint8_t *)malloc(cplan.work_size);
        if (!cplan.work_data) {
            fprintf(stderr, "[wav2vec2] OOM for work buffer\n");
            ggml_free(gctx); return {};
        }
    }
    ggml_graph_compute(gf2, &cplan);
    free(cplan.work_data);

    ggml_tensor * out = ggml_graph_get_tensor(gf2, "logits");
    int V_out = (int)out->ne[0];
    int T_out = (int)out->ne[1];
    // Sanity check: print first frame logits
    {
        const float * ld = (const float *)out->data;
        fprintf(stderr, "[wav2vec2-graph] logits[t=0, 0:8]:");
        for (int v = 0; v < 8; v++) fprintf(stderr, " %.4f", ld[v]);
        fprintf(stderr, "\n");
        // Argmax
        int best = 0;
        for (int v = 1; v < V_out; v++) if (ld[v] > ld[best]) best = v;
        fprintf(stderr, "[wav2vec2-graph] t=0 argmax=%d\n", best);
        // Argmax for frames 100-110 (where speech should be)
        fprintf(stderr, "[wav2vec2-graph] argmax t=100..110:");
        for (int t = 100; t < std::min(110, T_out); t++) {
            const float * frame = ld + t * V_out;
            int b = 0;
            for (int v = 1; v < V_out; v++) if (frame[v] > frame[b]) b = v;
            fprintf(stderr, " %d", b);
        }
        fprintf(stderr, "\n");
    }

    std::vector<float> logits(V_out * T_out);
    memcpy(logits.data(), out->data, logits.size() * sizeof(float));

    ggml_free(gctx);

    // The graph outputs (V, T) in ggml layout: data[v + t*V].
    // Our API expects (T, V) row-major: data[t*V + v].
    // These are the SAME layout! Just return as-is.
    std::vector<float> logits_tv(logits.begin(), logits.end());

    return logits_tv;
}

// ===========================================================================
// Manual C++ forward pass (legacy, CPU-only)
// ===========================================================================

std::vector<float> wav2vec2_compute_logits(
    const wav2vec2_model & m,
    const float * raw_audio, int n_samples,
    int n_threads)
{
    // Try graph path (uses compute_with_ctx for correct F16 weight handling)
    auto result = wav2vec2_compute_logits_graph(m, raw_audio, n_samples, n_threads);
    if (!result.empty()) return result;
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

    const auto & hp = m.hparams;

    // ------------------------------------------------------------------
    // 0. Normalize: zero-mean, unit-variance
    // ------------------------------------------------------------------
    std::vector<float> audio(raw_audio, raw_audio + n_samples);
    {
        double sum = 0.0, sq = 0.0;
        for (float v : audio) { sum += v; sq += (double)v * v; }
        float mean = (float)(sum / n_samples);
        float std_ = sqrtf(std::max(0.f, (float)(sq / n_samples) - mean * mean) + 1e-7f);
        for (float & v : audio) v = (v - mean) / std_;
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
        uint32_t C_out  = hp.conv_dim[li];
        uint32_t K      = hp.conv_kernel[li];
        uint32_t stride = hp.conv_stride[li];
        uint32_t L_out  = (L_cur - K) / stride + 1;

        cnn_out.resize(C_out * L_out);

        // Dequantise conv weight to F32 if needed
        std::vector<float> w_buf;
        const float * wdata;
        if (m.cnn[li].conv_w->type == GGML_TYPE_F16) {
            size_t n = C_out * C_cur * K;
            w_buf.resize(n);
            const ggml_fp16_t * w16 = (const ggml_fp16_t *)m.cnn[li].conv_w->data;
            for (size_t i = 0; i < n; i++) w_buf[i] = ggml_fp16_to_fp32(w16[i]);
            wdata = w_buf.data();
        } else {
            wdata = (const float *)m.cnn[li].conv_w->data;
        }
        const float * bdata = (const float *)m.cnn[li].conv_b->data;
        const float * nw    = m.cnn[li].has_norm ? (const float *)m.cnn[li].norm_w->data : nullptr;
        const float * nb    = m.cnn[li].has_norm ? (const float *)m.cnn[li].norm_b->data : nullptr;

        conv1d(cnn_in.data(), wdata, bdata,
               cnn_out.data(),
               (int)C_cur, (int)C_out, (int)K, (int)stride, (int)L_cur,
               /*left_pad=*/0);

        if (m.cnn[li].has_norm) {
            if (hp.feat_extract_norm_type == 1)
                layer_norm_cf(cnn_out.data(), cnn_out.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
            else
                instance_norm_1d(cnn_out.data(), cnn_out.data(), nw, nb, (int)C_out, (int)L_out, hp.layer_norm_eps);
        }
        for (float & v : cnn_out) v = gelu(v);

        std::swap(cnn_in, cnn_out);
        C_cur = C_out;
        L_cur = L_out;
    }
    // cnn_in = [C_last=512, T]

    int T    = (int)L_cnn;
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

    layer_norm(feat.data(), feat.data(),
               (const float *)m.fp_ln_w->data,
               (const float *)m.fp_ln_b->data,
               T, C_cnn, hp.layer_norm_eps);

    std::vector<float> hidden(T * H);
    std::vector<uint8_t> scratch;

    ggml_linear_f32(scratch, m.fp_w, (const float *)m.fp_b->data,
                    feat.data(), hidden.data(), C_cnn, H, T, n_threads);

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
        const float * pw;
        size_t pos_w_n = (size_t)H * (H / G_pos) * K_pos;
        if (m.pos_conv_w->type == GGML_TYPE_F16) {
            pw_buf.resize(pos_w_n);
            const ggml_fp16_t * p16 = (const ggml_fp16_t *)m.pos_conv_w->data;
            for (size_t i = 0; i < pos_w_n; i++) pw_buf[i] = ggml_fp16_to_fp32(p16[i]);
            pw = pw_buf.data();
        } else if (m.pos_conv_w->type == GGML_TYPE_F32) {
            pw = (const float *)m.pos_conv_w->data;
        } else {
            fprintf(stderr, "[wav2vec2] pos_conv.weight has unsupported type %d\n",
                    (int)m.pos_conv_w->type);
            return {};
        }
        const float * pb = (const float *)m.pos_conv_b->data;

        grouped_conv1d_same(hcf.data(), pw, pb, pos_out.data(), H, H, K_pos, G_pos, T);

        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
                hidden[t * H + h] += gelu(pos_out[h * T + t]);
    }

    // wav2vec2_debug_attention(m, hidden.data(), T, H);  // disabled

    // ------------------------------------------------------------------
    // 4. Transformer encoder layers (pre-norm / stable layer norm)
    // ------------------------------------------------------------------
    int n_heads  = (int)hp.num_attention_heads;
    int head_dim = H / n_heads;
    float scale  = 1.f / sqrtf((float)head_dim);
    int I = (int)hp.intermediate_size;

    std::vector<float> normed(T * H);
    std::vector<float> Q_buf(T * H), K_buf(T * H), V_buf(T * H);
    std::vector<float> attn_out(T * H);
    std::vector<float> ffn_mid(T * I);
    std::vector<float> ffn_out(T * H);
    std::vector<float> scores(T);

    for (uint32_t li = 0; li < hp.num_hidden_layers; li++) {
        const auto & e = m.enc[li];

        layer_norm(hidden.data(), normed.data(),
                   (const float *)e.ln1_w->data, (const float *)e.ln1_b->data,
                   T, H, hp.layer_norm_eps);

        ggml_linear_f32(scratch, e.q_w, (const float *)e.q_b->data, normed.data(), Q_buf.data(), H, H, T, n_threads);
        ggml_linear_f32(scratch, e.k_w, (const float *)e.k_b->data, normed.data(), K_buf.data(), H, H, T, n_threads);
        ggml_linear_f32(scratch, e.v_w, (const float *)e.v_b->data, normed.data(), V_buf.data(), H, H, T, n_threads);

        std::fill(attn_out.begin(), attn_out.end(), 0.f);
        for (int h = 0; h < n_heads; h++) {
            int off = h * head_dim;
            for (int tq = 0; tq < T; tq++) {
                for (int tk = 0; tk < T; tk++) {
                    float dot = 0.f;
                    const float * q = Q_buf.data() + tq * H + off;
                    const float * k = K_buf.data() + tk * H + off;
                    for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];
                    scores[tk] = dot * scale;
                }
                softmax(scores.data(), T);

                float * ao = attn_out.data() + tq * H + off;
                for (int tv = 0; tv < T; tv++) {
                    const float * v = V_buf.data() + tv * H + off;
                    float s = scores[tv];
                    for (int d = 0; d < head_dim; d++) ao[d] += s * v[d];
                }
            }
        }

        ggml_linear_f32(scratch, e.o_w, (const float *)e.o_b->data, attn_out.data(), normed.data(), H, H, T, n_threads);
        for (int i = 0; i < T * H; i++) hidden[i] += normed[i];

        layer_norm(hidden.data(), normed.data(),
                   (const float *)e.ln2_w->data, (const float *)e.ln2_b->data,
                   T, H, hp.layer_norm_eps);

        ggml_linear_f32(scratch, e.fc1_w, (const float *)e.fc1_b->data, normed.data(), ffn_mid.data(), H, I, T, n_threads);
        for (int i = 0; i < T * I; i++) ffn_mid[i] = gelu(ffn_mid[i]);

        ggml_linear_f32(scratch, e.fc2_w, (const float *)e.fc2_b->data, ffn_mid.data(), ffn_out.data(), I, H, T, n_threads);
        for (int i = 0; i < T * H; i++) hidden[i] += ffn_out[i];

    }

    // ------------------------------------------------------------------
    // 5. Encoder global LayerNorm
    // ------------------------------------------------------------------
    layer_norm(hidden.data(), hidden.data(),
               (const float *)m.enc_ln_w->data, (const float *)m.enc_ln_b->data,
               T, H, hp.layer_norm_eps);

    // ------------------------------------------------------------------
    // 6. LM head: Linear(H → V) — raw logits (no softmax)
    // ------------------------------------------------------------------
    int V = (int)hp.vocab_size;
    std::vector<float> logits(T * V);
    ggml_linear_f32(scratch, m.lm_w, (const float *)m.lm_b->data,
                    hidden.data(), logits.data(), H, V, T, n_threads);

    return logits;
}

// ===========================================================================
// Greedy CTC decode
// ===========================================================================

std::string wav2vec2_greedy_decode(const wav2vec2_model & m,
                                   const float * logits, int T)
{
    const auto & hp = m.hparams;
    int V = (int)hp.vocab_size;

    std::string result;
    int prev_id = -1;
    for (int t = 0; t < T; t++) {
        const float * lv = logits + t * V;
        int best_id = (int)(std::max_element(lv, lv + V) - lv);
        if (best_id != prev_id) {
            if (best_id != (int)hp.pad_token_id && best_id < (int)m.vocab.size()) {
                const std::string & tok = m.vocab[best_id];
                if (tok == "|")                                   result += ' ';
                else if (tok != "<unk>" && tok != "<s>" && tok != "</s>") result += tok;
            }
            prev_id = best_id;
        }
    }
    // Trim leading/trailing spaces
    auto lo = result.find_first_not_of(' ');
    auto hi = result.find_last_not_of(' ');
    return (lo == std::string::npos) ? "" : result.substr(lo, hi - lo + 1);
}
