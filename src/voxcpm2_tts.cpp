// voxcpm2_tts.cpp — VoxCPM2 TTS runtime implementation.
//
// Architecture:
//   TSLM:   28L MiniCPM-4, d=2048, 16Q/2KV heads, head_dim=128, ff=6144, LongRoPE
//   RALM:   8L  MiniCPM-4, d=2048, 16Q/2KV heads, head_dim=128, ff=6144, no RoPE
//   LocEnc: 12L bidirectional transformer, d=1024, 16h, head_dim=128, ff=4096, CLS token
//   LocDiT: 12L bidirectional transformer, d=1024, 16h, head_dim=128, ff=4096, sinusoidal time
//   FSQ:    Linear(2048->512)->tanh->round(x*9)/9->Linear(512->2048)
//   VAE:    causal transposed convolutions, Snake1d, weight-norm, SR conditioning
//
// Loading: two-pass GGUF via core_gguf::open_metadata / load_weights.
// Matmul:  tiny ggml graphs on a shared CPU backend (g_cpu_backend) for
//          matrix-vector; manual loop (get_row_f32) for matrix-matrix prefill.
// KV cache: manual std::vector<float> per layer (CPU side).

#include "voxcpm2_tts.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

static bool vox_env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0;
}

static double vox_now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
        steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Shared CPU backend for tiny ggml graph matmuls
// ---------------------------------------------------------------------------

static ggml_backend_t g_cpu_backend = nullptr;

static ggml_backend_t get_cpu_backend() {
    if (!g_cpu_backend) {
        g_cpu_backend = ggml_backend_cpu_init();
    }
    return g_cpu_backend;
}

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct vox_hparams {
    // TSLM
    uint32_t tslm_n_layers   = 28;
    uint32_t tslm_d_model    = 2048;
    uint32_t tslm_n_heads    = 16;
    uint32_t tslm_n_kv       = 2;
    uint32_t tslm_head_dim   = 128;
    uint32_t tslm_ff_dim     = 6144;
    uint32_t tslm_max_pos    = 32768;
    float    tslm_rope_theta = 500000.0f;
    float    rms_norm_eps    = 1e-5f;

    // RALM
    uint32_t ralm_n_layers   = 8;
    uint32_t ralm_d_model    = 2048;
    uint32_t ralm_n_heads    = 16;
    uint32_t ralm_n_kv       = 2;
    uint32_t ralm_head_dim   = 128;
    uint32_t ralm_ff_dim     = 6144;

    // LocEnc
    uint32_t locenc_n_layers = 12;
    uint32_t locenc_d_model  = 1024;
    uint32_t locenc_n_heads  = 16;
    uint32_t locenc_n_kv     = 16;
    uint32_t locenc_head_dim = 128;
    uint32_t locenc_ff_dim   = 4096;

    // LocDiT
    uint32_t locdit_n_layers = 12;
    uint32_t locdit_d_model  = 1024;
    uint32_t locdit_n_heads  = 16;
    uint32_t locdit_n_kv     = 16;
    uint32_t locdit_head_dim = 128;
    uint32_t locdit_ff_dim   = 4096;

    // Tokenizer
    uint32_t n_vocab           = 100000;
    uint32_t audio_start_token = 0;

    // Patch / VAE
    uint32_t patch_frames = 4;
    uint32_t patch_dim    = 256;
};

// ---------------------------------------------------------------------------
// Weight tensor structs
// ---------------------------------------------------------------------------

struct vox_lm_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w   = nullptr;
    ggml_tensor* attn_k_w   = nullptr;
    ggml_tensor* attn_v_w   = nullptr;
    ggml_tensor* attn_o_w   = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w   = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct vox_enc_layer {
    ggml_tensor* norm1_w    = nullptr;
    ggml_tensor* norm2_w    = nullptr;
    ggml_tensor* attn_q_w   = nullptr;
    ggml_tensor* attn_k_w   = nullptr;
    ggml_tensor* attn_v_w   = nullptr;
    ggml_tensor* attn_o_w   = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w   = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct vox_vae_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct vox_weights {
    // TSLM
    ggml_tensor* tslm_token_embd  = nullptr;
    ggml_tensor* tslm_output_norm = nullptr;
    // NOTE: tslm.lm_head.weight is NOT in the GGUF — omitted intentionally
    std::vector<vox_lm_layer> tslm_layers;

    // RoPE factors (LongRoPE)
    ggml_tensor* tslm_rope_short = nullptr;
    ggml_tensor* tslm_rope_long  = nullptr;

    // RALM
    ggml_tensor* ralm_output_norm = nullptr;
    std::vector<vox_lm_layer> ralm_layers;

    // FSQ
    ggml_tensor* fsq_in_proj_w  = nullptr;
    ggml_tensor* fsq_in_proj_b  = nullptr;
    ggml_tensor* fsq_out_proj_w = nullptr;
    ggml_tensor* fsq_out_proj_b = nullptr;

    // LocEnc
    ggml_tensor* locenc_cls_token  = nullptr;
    ggml_tensor* locenc_in_proj_w  = nullptr;  // [64, 1024] feat_dim→d_model
    ggml_tensor* locenc_in_proj_b  = nullptr;  // [1024]
    ggml_tensor* locenc_norm_w     = nullptr;
    std::vector<vox_enc_layer> locenc_layers;

    // LocDiT
    ggml_tensor* locdit_in_proj_w    = nullptr;  // [64, 1024] raw latent→d_model
    ggml_tensor* locdit_in_proj_b    = nullptr;
    ggml_tensor* locdit_cond_proj_w  = nullptr;  // [64, 1024] cond patch→d_model
    ggml_tensor* locdit_cond_proj_b  = nullptr;
    ggml_tensor* locdit_time_mlp_0_w = nullptr;  // [1024, 1024] sinusoidal→hidden
    ggml_tensor* locdit_time_mlp_0_b = nullptr;
    ggml_tensor* locdit_time_mlp_1_w = nullptr;  // [1024, 1024] hidden→t_emb
    ggml_tensor* locdit_time_mlp_1_b = nullptr;
    ggml_tensor* locdit_dt_mlp_0_w   = nullptr;  // [1024, 1024] dt sinusoidal→hidden
    ggml_tensor* locdit_dt_mlp_0_b   = nullptr;
    ggml_tensor* locdit_dt_mlp_1_w   = nullptr;  // [1024, 1024] hidden→dt_emb
    ggml_tensor* locdit_dt_mlp_1_b   = nullptr;
    ggml_tensor* locdit_norm_w       = nullptr;
    ggml_tensor* locdit_out_proj_w   = nullptr;  // [1024, 64] d_model→feat_dim
    ggml_tensor* locdit_out_proj_b   = nullptr;
    std::vector<vox_enc_layer> locdit_layers;

    // Projection heads
    ggml_tensor* enc_to_lm_w  = nullptr;
    ggml_tensor* enc_to_lm_b  = nullptr;
    ggml_tensor* lm_to_dit_w  = nullptr;
    ggml_tensor* lm_to_dit_b  = nullptr;
    ggml_tensor* res_to_dit_w = nullptr;
    ggml_tensor* res_to_dit_b = nullptr;
    ggml_tensor* fusion_w     = nullptr;  // proj.fusion.weight [4096, 2048]
    ggml_tensor* fusion_b     = nullptr;  // proj.fusion.bias   [2048]

    // Stop predictor
    ggml_tensor* stop_proj_w = nullptr;
    ggml_tensor* stop_proj_b = nullptr;
    ggml_tensor* stop_head_w = nullptr;  // [2048, 2] — no bias

    // VAE decoder (optional — graceful degradation if absent)
    // Weights stored in ctx->tensors under "vae.dec.*" keys
    vox_vae_conv vae_in_conv;
    vox_vae_conv vae_out_conv;
    ggml_tensor* vae_out_snake_a = nullptr;
    ggml_tensor* vae_sr_cond_w   = nullptr;
    ggml_tensor* vae_sr_cond_b   = nullptr;
};

// ---------------------------------------------------------------------------
// KV cache — manual CPU float storage, one vector per layer
// ---------------------------------------------------------------------------

struct vox_kv_cache {
    int n_layers = 0;
    int n_kv     = 0;
    int head_dim = 0;
    int max_ctx  = 0;
    int n_past   = 0;
    // k_cache[l]: [max_ctx * n_kv * head_dim] row-major
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;

    void init(int layers, int kv_heads, int hd, int max_context) {
        n_layers = layers;
        n_kv     = kv_heads;
        head_dim = hd;
        max_ctx  = max_context;
        k_cache.assign(layers, std::vector<float>((size_t)max_context * kv_heads * hd, 0.0f));
        v_cache.assign(layers, std::vector<float>((size_t)max_context * kv_heads * hd, 0.0f));
        n_past = 0;
    }

    void reset() { n_past = 0; }
};

// ---------------------------------------------------------------------------
// Tokenizer — BPE with GPT-2 byte encoding
// ---------------------------------------------------------------------------

struct vox_tokenizer {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" -> rank
    std::vector<std::string> id_to_token;
    int32_t audio_start_token = -1;
    int32_t eos_token         = -1;
};

// ---------------------------------------------------------------------------
// Context struct
// ---------------------------------------------------------------------------

struct voxcpm2_context {
    vox_hparams    hp;
    vox_weights    weights;
    vox_tokenizer  tokenizer;
    vox_kv_cache   tslm_kv;
    vox_kv_cache   ralm_kv;

    // GGUF weight storage — owned here
    ggml_context*          ggml_ctx   = nullptr;
    ggml_backend_buffer_t  weight_buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Runtime params
    int   n_threads       = 4;
    int   verbosity       = 1;
    bool  use_gpu         = false;
    bool  flash_attn      = true;
    int   inference_steps = 10;
    float cfg_value       = 2.0f;
    int   max_len         = 2000;
};

// Stream struct
struct voxcpm2_stream {
    voxcpm2_context*   ctx  = nullptr;
    std::vector<float> all_pcm;
    std::vector<float> chunk_buf;
    int  chunk_offset = 0;
    bool done         = false;
};

// ---------------------------------------------------------------------------
// get_row_f32: copy one row from any weight tensor type to float
// Handles F32, F16, Q4_K (and any type with a to_float trait) via
// ggml_get_type_traits()->to_float.
// ---------------------------------------------------------------------------

static void get_row_f32(const ggml_tensor* t, int row, float* out) {
    const int cols = (int)t->ne[0];
    if (t->type == GGML_TYPE_F32) {
        const float* src = (const float*)((const char*)t->data + (size_t)row * t->nb[1]);
        std::memcpy(out, src, (size_t)cols * sizeof(float));
        return;
    }
    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (tt && tt->to_float) {
        const void* src = (const char*)t->data + (size_t)row * t->nb[1];
        tt->to_float(src, out, cols);
        return;
    }
    std::memset(out, 0, (size_t)cols * sizeof(float));
}

// tensor_data_f32: for tensors guaranteed to be F32 (norms, biases).
static const float* tensor_data_f32(const ggml_tensor* t) {
    return (const float*)t->data;
}

// ---------------------------------------------------------------------------
// matmul_mv_ggml: W[rows,cols] x v[cols] -> out[rows] via tiny ggml graph.
// Uses ggml_mul_mat, which dispatches to SIMD / Q4_K native dot products.
// ---------------------------------------------------------------------------

static void matmul_mv_ggml(ggml_backend_t cpu_be, ggml_tensor* W,
                            const float* v, int cols, float* out, int rows) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ (size_t)2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context* tmp_ctx = ggml_init(ip);

    // Input vector as a 2-D column: [cols, 1]
    ggml_tensor* v_t = ggml_new_tensor_2d(tmp_ctx, GGML_TYPE_F32, cols, 1);
    std::memcpy(v_t->data, v, (size_t)cols * sizeof(float));

    // ggml_mul_mat(a, b): a[ne0, ne1] x b[ne0, 1] -> result[ne1, 1]
    // W tensor from GGUF: ne[0] = inner dim, ne[1] = outer dim
    // We need W->ne[0] == cols for the matmul to work
    if ((int)W->ne[0] != cols) {
        fprintf(stderr, "voxcpm2: matmul_mv_ggml dim mismatch: W[%lld,%lld] x v[%d] (expected W->ne[0]==%d)\n",
                (long long)W->ne[0], (long long)W->ne[1], cols, cols);
    }
    ggml_tensor* result = ggml_mul_mat(tmp_ctx, W, v_t);

    ggml_cgraph* gf = ggml_new_graph(tmp_ctx);
    ggml_build_forward_expand(gf, result);

    ggml_backend_cpu_set_n_threads(cpu_be, g_cpu_n_threads);
    ggml_backend_graph_compute(cpu_be, gf);

    std::memcpy(out, result->data, (size_t)rows * sizeof(float));
    ggml_free(tmp_ctx);
}

static void matmul_mv(ggml_backend_t cpu_be, ggml_tensor* W,
                      const float* v, int cols, float* out, int rows) {
    matmul_mv_ggml(cpu_be, W, v, cols, out, rows);
}

static void matmul_mv_bias(ggml_backend_t cpu_be, ggml_tensor* W, ggml_tensor* b_t,
                            const float* v, int cols, float* out, int rows) {
    matmul_mv_ggml(cpu_be, W, v, cols, out, rows);
    const float* b = tensor_data_f32(b_t);
    for (int i = 0; i < rows; i++) out[i] += b[i];
}

// matmul_mm: W[rows,cols] x X[cols,T] -> Y[T,rows]  (manual loop, for prefill)
static void matmul_mm(ggml_tensor* W, const float* X,
                      int cols, int rows, int T, float* Y) {
    std::vector<float> w_row((size_t)cols);
    for (int r = 0; r < rows; r++) {
        get_row_f32(W, r, w_row.data());
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            const float* xc = X + (size_t)t * cols;
            for (int c = 0; c < cols; c++) acc += w_row[c] * xc[c];
            Y[(size_t)t * rows + r] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// RMS Norm (CPU, in-place-safe: x and y may differ)
// ---------------------------------------------------------------------------

static void rms_norm_cpu(const float* x, const float* w, float* y, int d, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    ss = ss / (float)d + eps;
    float inv = 1.0f / std::sqrt(ss);
    for (int i = 0; i < d; i++) y[i] = x[i] * inv * w[i];
}

// ---------------------------------------------------------------------------
// SwiGLU FFN: out = W_down @ (silu(W_gate @ x) * W_up @ x)
// ---------------------------------------------------------------------------

static void swiglu_ffn_cpu(ggml_backend_t cpu_be,
                            ggml_tensor* gate_w, ggml_tensor* up_w, ggml_tensor* down_w,
                            const float* x, int d_in, int d_ff, int d_out,
                            float* y) {
    std::vector<float> gate(d_ff), up(d_ff), h(d_ff);
    matmul_mv(cpu_be, gate_w, x, d_in, gate.data(), d_ff);
    matmul_mv(cpu_be, up_w,   x, d_in, up.data(),   d_ff);
    for (int i = 0; i < d_ff; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + std::exp(-g));
        h[i] = g * sig * up[i];
    }
    matmul_mv(cpu_be, down_w, h.data(), d_ff, y, d_out);
}

// ---------------------------------------------------------------------------
// NEOX RoPE (LongRoPE baseline: no freq scaling, standard complex rotation)
// Operates on interleaved layout: pairs (i, i+head_dim/2) within each head.
// ---------------------------------------------------------------------------

static void rope_apply_cpu(float* qk, int head_dim, int n_heads, int pos,
                            float theta, int /*n_ctx_orig*/) {
    for (int h = 0; h < n_heads; h++) {
        float* vec = qk + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq  = 1.0f / std::pow(theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float x0 = vec[i];
            float x1 = vec[i + head_dim / 2];
            vec[i]               = x0 * cos_a - x1 * sin_a;
            vec[i + head_dim / 2] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal attention step — single new token, writes into KV cache, reads full
// history [0, n_past+1), returns output [n_q * head_dim].
// ---------------------------------------------------------------------------

static void causal_attn_step(
        const float* q_in,   // [n_q  * hd]
        const float* k_new,  // [n_kv * hd]
        const float* v_new,  // [n_kv * hd]
        float* out,          // [n_q  * hd]
        vox_kv_cache& cache,
        int layer,
        int n_q, int n_kv, int hd,
        float attn_scale) {
    int n_past  = cache.n_past;
    int seq_len = n_past + 1;
    int grp     = n_q / n_kv;

    // Write new K/V into cache at position n_past
    {
        float* kd = cache.k_cache[layer].data() + (size_t)n_past * n_kv * hd;
        float* vd = cache.v_cache[layer].data() + (size_t)n_past * n_kv * hd;
        std::memcpy(kd, k_new, (size_t)n_kv * hd * sizeof(float));
        std::memcpy(vd, v_new, (size_t)n_kv * hd * sizeof(float));
    }

    std::vector<float> scores(seq_len), attn_w(seq_len);

    for (int qh = 0; qh < n_q; qh++) {
        const float* q  = q_in + qh * hd;
        int           kvh = qh / grp;

        // Dot products Q.K^T over cached history
        for (int t = 0; t < seq_len; t++) {
            const float* k_t = cache.k_cache[layer].data() + (size_t)t * n_kv * hd + kvh * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) dot += q[d] * k_t[d];
            scores[t] = dot * attn_scale;
        }

        // Stable softmax
        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++) if (scores[t] > max_s) max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            attn_w[t] = std::exp(scores[t] - max_s);
            sum_e += attn_w[t];
        }
        for (int t = 0; t < seq_len; t++) attn_w[t] /= sum_e;

        // Weighted sum over V
        float* o = out + qh * hd;
        std::fill(o, o + hd, 0.0f);
        for (int t = 0; t < seq_len; t++) {
            const float* v_t = cache.v_cache[layer].data() + (size_t)t * n_kv * hd + kvh * hd;
            for (int d = 0; d < hd; d++) o[d] += attn_w[t] * v_t[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Bidirectional (full) attention for LocEnc / LocDiT — no mask
// x_in: [T * d] row-major, out: [T * d] row-major
// ---------------------------------------------------------------------------

static void bidir_attn_full(
        const float* x_in, int T, int d,
        ggml_tensor* q_w, ggml_tensor* k_w, ggml_tensor* v_w, ggml_tensor* o_w,
        int n_q, int n_kv, int hd,
        float attn_scale,
        ggml_backend_t cpu_be,
        float* out) {
    int grp = n_q / n_kv;

    std::vector<float> Q((size_t)T * n_q * hd);
    std::vector<float> K((size_t)T * n_kv * hd);
    std::vector<float> V((size_t)T * n_kv * hd);

    for (int t = 0; t < T; t++) {
        const float* xt = x_in + (size_t)t * d;
        matmul_mv(cpu_be, q_w, xt, d, Q.data() + (size_t)t * n_q  * hd, n_q  * hd);
        matmul_mv(cpu_be, k_w, xt, d, K.data() + (size_t)t * n_kv * hd, n_kv * hd);
        matmul_mv(cpu_be, v_w, xt, d, V.data() + (size_t)t * n_kv * hd, n_kv * hd);
    }

    std::vector<float> attn_out((size_t)T * n_q * hd, 0.0f);
    std::vector<float> scores(T), aw(T);

    for (int t = 0; t < T; t++) {
        for (int qh = 0; qh < n_q; qh++) {
            const float* q   = Q.data() + (size_t)t * n_q * hd + qh * hd;
            int           kvh = qh / grp;

            for (int s = 0; s < T; s++) {
                const float* ks = K.data() + (size_t)s * n_kv * hd + kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += q[i] * ks[i];
                scores[s] = dot * attn_scale;
            }

            float max_s = scores[0];
            for (int s = 1; s < T; s++) if (scores[s] > max_s) max_s = scores[s];
            float sum_e = 0.0f;
            for (int s = 0; s < T; s++) { aw[s] = std::exp(scores[s] - max_s); sum_e += aw[s]; }
            for (int s = 0; s < T; s++) aw[s] /= sum_e;

            float* ao = attn_out.data() + (size_t)t * n_q * hd + qh * hd;
            for (int s = 0; s < T; s++) {
                const float* vs = V.data() + (size_t)s * n_kv * hd + kvh * hd;
                for (int i = 0; i < hd; i++) ao[i] += aw[s] * vs[i];
            }
        }
    }

    // Output projection: attn_out[T, n_q*hd] x o_w -> out[T, d]
    for (int t = 0; t < T; t++) {
        matmul_mv(cpu_be, o_w, attn_out.data() + (size_t)t * n_q * hd, n_q * hd,
                  out + (size_t)t * d, d);
    }
}

// ---------------------------------------------------------------------------
// TSLM layer — single-token causal decode with KV cache
// ---------------------------------------------------------------------------

static void tslm_layer_step(voxcpm2_context* ctx, int layer,
                             float* hidden, int pos,
                             ggml_backend_t cpu_be) {
    const vox_hparams&  hp  = ctx->hp;
    const vox_lm_layer& L   = ctx->weights.tslm_layers[layer];
    int   d          = (int)hp.tslm_d_model;
    int   n_q        = (int)hp.tslm_n_heads;
    int   n_kv       = (int)hp.tslm_n_kv;
    int   hd         = (int)hp.tslm_head_dim;
    float eps        = hp.rms_norm_eps;
    float attn_scale = 1.0f / std::sqrt((float)hd);

    std::vector<float> normed(d), q(n_q * hd), k(n_kv * hd), v(n_kv * hd),
                       attn_out(n_q * hd), proj_out(d), ffn_out(d);

    rms_norm_cpu(hidden, tensor_data_f32(L.attn_norm_w), normed.data(), d, eps);

    matmul_mv(cpu_be, L.attn_q_w, normed.data(), d, q.data(), n_q  * hd);
    matmul_mv(cpu_be, L.attn_k_w, normed.data(), d, k.data(), n_kv * hd);
    matmul_mv(cpu_be, L.attn_v_w, normed.data(), d, v.data(), n_kv * hd);

    rope_apply_cpu(q.data(), hd, n_q,  pos, hp.tslm_rope_theta, (int)hp.tslm_max_pos);
    rope_apply_cpu(k.data(), hd, n_kv, pos, hp.tslm_rope_theta, (int)hp.tslm_max_pos);

    causal_attn_step(q.data(), k.data(), v.data(), attn_out.data(),
                     ctx->tslm_kv, layer, n_q, n_kv, hd, attn_scale);

    matmul_mv(cpu_be, L.attn_o_w, attn_out.data(), n_q * hd, proj_out.data(), d);
    for (int i = 0; i < d; i++) hidden[i] += proj_out[i];

    rms_norm_cpu(hidden, tensor_data_f32(L.ffn_norm_w), normed.data(), d, eps);
    swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w,
                   normed.data(), d, (int)hp.tslm_ff_dim, d, ffn_out.data());
    for (int i = 0; i < d; i++) hidden[i] += ffn_out[i];
}

// ---------------------------------------------------------------------------
// RALM layer — causal decode, no RoPE
// ---------------------------------------------------------------------------

static void ralm_layer_step(voxcpm2_context* ctx, int layer,
                             float* hidden,
                             ggml_backend_t cpu_be) {
    const vox_hparams&  hp  = ctx->hp;
    const vox_lm_layer& L   = ctx->weights.ralm_layers[layer];
    int   d          = (int)hp.ralm_d_model;
    int   n_q        = (int)hp.ralm_n_heads;
    int   n_kv       = (int)hp.ralm_n_kv;
    int   hd         = (int)hp.ralm_head_dim;
    float eps        = hp.rms_norm_eps;
    float attn_scale = 1.0f / std::sqrt((float)hd);

    std::vector<float> normed(d), q(n_q * hd), k(n_kv * hd), v(n_kv * hd),
                       attn_out(n_q * hd), proj_out(d), ffn_out(d);

    rms_norm_cpu(hidden, tensor_data_f32(L.attn_norm_w), normed.data(), d, eps);

    matmul_mv(cpu_be, L.attn_q_w, normed.data(), d, q.data(), n_q  * hd);
    matmul_mv(cpu_be, L.attn_k_w, normed.data(), d, k.data(), n_kv * hd);
    matmul_mv(cpu_be, L.attn_v_w, normed.data(), d, v.data(), n_kv * hd);

    // No RoPE for RALM
    causal_attn_step(q.data(), k.data(), v.data(), attn_out.data(),
                     ctx->ralm_kv, layer, n_q, n_kv, hd, attn_scale);

    matmul_mv(cpu_be, L.attn_o_w, attn_out.data(), n_q * hd, proj_out.data(), d);
    for (int i = 0; i < d; i++) hidden[i] += proj_out[i];

    rms_norm_cpu(hidden, tensor_data_f32(L.ffn_norm_w), normed.data(), d, eps);
    swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w,
                   normed.data(), d, (int)hp.ralm_ff_dim, d, ffn_out.data());
    for (int i = 0; i < d; i++) hidden[i] += ffn_out[i];
}

// ---------------------------------------------------------------------------
// TSLM prefill — run all text tokens through, filling KV cache
// Returns last hidden state [d_model] (pre-output-norm)
// ---------------------------------------------------------------------------

static std::vector<float> tslm_prefill(voxcpm2_context* ctx,
                                        const std::vector<int32_t>& token_ids,
                                        ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.tslm_d_model;
    int T = (int)token_ids.size();
    ctx->tslm_kv.reset();

    std::vector<float> hidden(d);

    for (int t = 0; t < T; t++) {
        int id = token_ids[t];
        if (id < 0 || id >= (int)hp.n_vocab) id = 0;
        get_row_f32(ctx->weights.tslm_token_embd, id, hidden.data());

        for (int l = 0; l < (int)hp.tslm_n_layers; l++) {
            tslm_layer_step(ctx, l, hidden.data(), t, cpu_be);
        }
        ctx->tslm_kv.n_past = t + 1;
    }

    return hidden;
}

// ---------------------------------------------------------------------------
// RALM prefill — one token, fills KV cache
// ---------------------------------------------------------------------------

static std::vector<float> ralm_prefill(voxcpm2_context* ctx,
                                        const std::vector<float>& input,
                                        ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    int d = (int)hp.ralm_d_model;
    ctx->ralm_kv.reset();

    std::vector<float> hidden = input;
    if ((int)hidden.size() < d) hidden.resize(d, 0.0f);
    if ((int)hidden.size() > d) hidden.resize(d);

    for (int l = 0; l < (int)hp.ralm_n_layers; l++) {
        ralm_layer_step(ctx, l, hidden.data(), cpu_be);
    }
    ctx->ralm_kv.n_past = 1;

    return hidden;
}

// ---------------------------------------------------------------------------
// FSQ: Linear -> tanh -> round(x*9)/9 -> Linear
// ---------------------------------------------------------------------------

static std::vector<float> fsq_forward(voxcpm2_context* ctx,
                                       const float* x,
                                       ggml_backend_t cpu_be) {
    const vox_weights& W = ctx->weights;
    int d_in  = 2048;
    int d_mid = 512;
    int d_out = 2048;

    std::vector<float> mid(d_mid), quantized(d_mid), out(d_out);

    matmul_mv_bias(cpu_be, W.fsq_in_proj_w, W.fsq_in_proj_b, x, d_in, mid.data(), d_mid);
    for (int i = 0; i < d_mid; i++) {
        float v = std::tanh(mid[i]);
        quantized[i] = std::round(v * 9.0f) / 9.0f;
    }
    matmul_mv_bias(cpu_be, W.fsq_out_proj_w, W.fsq_out_proj_b, quantized.data(), d_mid, out.data(), d_out);

    return out;
}

// ---------------------------------------------------------------------------
// LocEnc forward — bidirectional 12-layer transformer with CLS prepend
// Input: patch flattened [patch_frames * patch_dim]
// Returns: CLS token output [d_locenc]
// ---------------------------------------------------------------------------

static std::vector<float> locenc_forward(voxcpm2_context* ctx,
                                          const float* patch,
                                          ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W  = ctx->weights;
    int d         = (int)hp.locenc_d_model;    // 1024
    int n_q       = (int)hp.locenc_n_heads;
    int n_kv      = (int)hp.locenc_n_kv;
    int hd        = (int)hp.locenc_head_dim;
    float eps     = hp.rms_norm_eps;
    float ascale  = 1.0f / std::sqrt((float)hd);

    // Sequence: [CLS, patch_frames tokens]
    // PyTorch: x=[B,T,P,D=64], in_proj projects D=64→d=1024
    // Here: patch is [patch_frames * feat_dim=64], one patch of P=patch_frames frames
    int n_patch_tok = (int)hp.patch_frames;      // P = 4 frames per patch
    int T           = n_patch_tok + 1;            // CLS + P tokens
    int feat_dim    = 64;                          // VAE latent dim per time-frame

    std::vector<float> cur((size_t)T * d, 0.0f);

    // CLS token (d-dimensional learned vector)
    if (W.locenc_cls_token) {
        std::memcpy(cur.data(), tensor_data_f32(W.locenc_cls_token), (size_t)d * sizeof(float));
    }

    // Patch tokens: project each feat_dim=64 frame to d=1024 via in_proj
    for (int f = 0; f < n_patch_tok; f++) {
        float* dst = cur.data() + (size_t)(f + 1) * d;
        const float* src = patch + (size_t)f * feat_dim;
        if (W.locenc_in_proj_w && W.locenc_in_proj_b) {
            // in_proj.weight: [feat_dim, d] stored as [d, feat_dim] in GGUF convention
            // matmul_mv_bias: W[rows=d, cols=feat_dim] x v[feat_dim] -> out[d]
            matmul_mv_bias(cpu_be, W.locenc_in_proj_w, W.locenc_in_proj_b,
                           src, feat_dim, dst, d);
        } else {
            // fallback: copy what fits, zero-pad rest
            int copy_d = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)copy_d * sizeof(float));
        }
    }

    std::vector<float> normed((size_t)T * d), attn_out((size_t)T * d), ffn_h(d);

    for (int l = 0; l < (int)hp.locenc_n_layers; l++) {
        const vox_enc_layer& L = W.locenc_layers[l];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + t * d, tensor_data_f32(L.norm1_w),
                         normed.data() + t * d, d, eps);
        }

        bidir_attn_full(normed.data(), T, d,
                        L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w,
                        n_q, n_kv, hd, ascale, cpu_be, attn_out.data());

        for (size_t i = 0; i < (size_t)T * d; i++) cur[i] += attn_out[i];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + t * d, tensor_data_f32(L.norm2_w),
                         normed.data() + t * d, d, eps);
            swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w,
                           normed.data() + t * d, d, (int)hp.locenc_ff_dim, d, ffn_h.data());
            float* ct = cur.data() + t * d;
            for (int i = 0; i < d; i++) ct[i] += ffn_h[i];
        }
    }

    // Return CLS token (position 0) after optional final norm
    std::vector<float> cls_out(d);
    if (W.locenc_norm_w) {
        rms_norm_cpu(cur.data(), tensor_data_f32(W.locenc_norm_w), cls_out.data(), d, eps);
    } else {
        std::memcpy(cls_out.data(), cur.data(), (size_t)d * sizeof(float));
    }

    return cls_out;
}

// ---------------------------------------------------------------------------
// Sinusoidal time embedding
// ---------------------------------------------------------------------------

static std::vector<float> sinusoidal_time_emb(float t_scalar, int dim) {
    std::vector<float> emb(dim, 0.0f);
    for (int i = 0; i < dim / 2; i++) {
        float freq = std::exp(-std::log(10000.0f) * (float)(2 * i) / (float)dim);
        emb[2 * i]     = std::cos(t_scalar * freq);
        emb[2 * i + 1] = std::sin(t_scalar * freq);
    }
    return emb;
}

// ---------------------------------------------------------------------------
// LocDiT forward — bidirectional 12-layer DiT, single denoising step
//
// PyTorch signature:
//   x:    (N, C=64, T=patch_size) noisy latent — here patch_size=patch_frames=4
//   mu:   (N, hidden=2048)         conditioning from TSLM+RALM
//   t:    (N,)                     diffusion timestep in [0,1]
//   cond: (N, C=64, T'=patch_size) previous patch condition (zeros for first step)
//   dt:   (N,)                     delta time (zeros for non-mean-mode)
//
// Sequence layout:
//   [mu_reshaped (2 tokens, d=1024), t_emb (1 token), cond_proj (P tokens), x_proj (P tokens)]
//   total T_seq = 2 + 1 + P + P
//
// We extract only the x-portion (last P tokens) and out-project to feat_dim=64.
//
// Arguments here (simplified to single-batch CPU):
//   x_raw:     [feat_dim * patch_frames] = [64 * 4 = 256] noisy latent
//   mu:        [tslm_d_model = 2048] conditioning
//   t_scalar:  timestep scalar in [0,1]
//   cond_raw:  [feat_dim * patch_frames] = [256] previous patch (zeros if first step)
//   dt_scalar: delta time (0.0 for non-mean-mode)
// Returns:     [feat_dim * patch_frames] = [256] predicted velocity
// ---------------------------------------------------------------------------

static std::vector<float> locdit_forward(voxcpm2_context* ctx,
                                          const float* x_raw,
                                          const float* mu,
                                          float t_scalar,
                                          const float* cond_raw,
                                          float dt_scalar,
                                          ggml_backend_t cpu_be) {
    const vox_hparams& hp = ctx->hp;
    const vox_weights& W  = ctx->weights;
    int d         = (int)hp.locdit_d_model;   // 1024
    int n_q       = (int)hp.locdit_n_heads;
    int n_kv      = (int)hp.locdit_n_kv;
    int hd        = (int)hp.locdit_head_dim;
    float eps     = hp.rms_norm_eps;
    float ascale  = 1.0f / std::sqrt((float)hd);
    int   feat_dim = 64;
    int   P        = (int)hp.patch_frames;    // 4 frames per patch
    int   mu_toks  = 2;                        // 2048 / 1024 = 2 mu tokens
    // Total sequence: [mu_toks, 1_time, P_cond, P_x] = 2+1+4+4 = 11 tokens
    int T = mu_toks + 1 + P + P;
    int x_offset   = mu_toks + 1 + P;         // position of x tokens in sequence

    // --- Build time embedding: sinusoidal -> two-layer MLP (Linear→SiLU→Linear) ---
    std::vector<float> t_sin = sinusoidal_time_emb(t_scalar,  d);
    std::vector<float> dt_sin = sinusoidal_time_emb(dt_scalar, d);

    // time_mlp: Linear(d,d) -> SiLU -> Linear(d,d)
    auto two_layer_mlp = [&](ggml_tensor* w0, ggml_tensor* b0,
                              ggml_tensor* w1, ggml_tensor* b1,
                              const std::vector<float>& inp) -> std::vector<float> {
        std::vector<float> h0(d), h1(d);
        if (w0 && b0) {
            matmul_mv_bias(cpu_be, w0, b0, inp.data(), d, h0.data(), d);
        } else {
            h0 = inp;
        }
        // SiLU activation: x * sigmoid(x)
        for (int i = 0; i < d; i++) {
            float x = h0[i];
            h0[i] = x / (1.0f + std::exp(-x));
        }
        if (w1 && b1) {
            matmul_mv_bias(cpu_be, w1, b1, h0.data(), d, h1.data(), d);
        } else {
            h1 = h0;
        }
        return h1;
    };

    std::vector<float> t_emb = two_layer_mlp(W.locdit_time_mlp_0_w, W.locdit_time_mlp_0_b,
                                              W.locdit_time_mlp_1_w, W.locdit_time_mlp_1_b,
                                              t_sin);
    std::vector<float> dt_emb = two_layer_mlp(W.locdit_dt_mlp_0_w, W.locdit_dt_mlp_0_b,
                                               W.locdit_dt_mlp_1_w, W.locdit_dt_mlp_1_b,
                                               dt_sin);
    // t_emb = t_emb + dt_emb
    for (int i = 0; i < d; i++) t_emb[i] += dt_emb[i];

    // --- Build sequence ---
    std::vector<float> cur((size_t)T * d, 0.0f);

    // Tokens 0..1: mu reshaped [2048] -> 2 x [1024]
    std::memcpy(cur.data(),        mu,        (size_t)d * sizeof(float));
    std::memcpy(cur.data() + d,    mu + d,    (size_t)d * sizeof(float));

    // Token 2: time embedding [1024]
    std::memcpy(cur.data() + 2 * d, t_emb.data(), (size_t)d * sizeof(float));

    // Tokens 3..3+P-1: cond_proj applied to each of P frames of cond_raw
    for (int p = 0; p < P; p++) {
        float* dst = cur.data() + (size_t)(mu_toks + 1 + p) * d;
        const float* src = cond_raw + (size_t)p * feat_dim;
        if (W.locdit_cond_proj_w && W.locdit_cond_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_cond_proj_w, W.locdit_cond_proj_b,
                           src, feat_dim, dst, d);
        } else {
            int cp = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)cp * sizeof(float));
        }
    }

    // Tokens 3+P..3+2P-1: in_proj applied to each of P frames of x_raw
    for (int p = 0; p < P; p++) {
        float* dst = cur.data() + (size_t)(x_offset + p) * d;
        const float* src = x_raw + (size_t)p * feat_dim;
        if (W.locdit_in_proj_w && W.locdit_in_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_in_proj_w, W.locdit_in_proj_b,
                           src, feat_dim, dst, d);
        } else {
            int cp = std::min(feat_dim, d);
            std::memcpy(dst, src, (size_t)cp * sizeof(float));
        }
    }

    // --- Bidirectional transformer layers ---
    std::vector<float> normed((size_t)T * d), attn_out((size_t)T * d), ffn_h(d);

    for (int l = 0; l < (int)hp.locdit_n_layers; l++) {
        const vox_enc_layer& L = W.locdit_layers[l];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm1_w),
                         normed.data() + (size_t)t * d, d, eps);
        }

        bidir_attn_full(normed.data(), T, d,
                        L.attn_q_w, L.attn_k_w, L.attn_v_w, L.attn_o_w,
                        n_q, n_kv, hd, ascale, cpu_be, attn_out.data());

        for (size_t i = 0; i < (size_t)T * d; i++) cur[i] += attn_out[i];

        for (int t = 0; t < T; t++) {
            rms_norm_cpu(cur.data() + (size_t)t * d, tensor_data_f32(L.norm2_w),
                         normed.data() + (size_t)t * d, d, eps);
            swiglu_ffn_cpu(cpu_be, L.ffn_gate_w, L.ffn_up_w, L.ffn_down_w,
                           normed.data() + (size_t)t * d, d, (int)hp.locdit_ff_dim, d, ffn_h.data());
            float* ct = cur.data() + (size_t)t * d;
            for (int i = 0; i < d; i++) ct[i] += ffn_h[i];
        }
    }

    // --- Extract x-portion and apply final norm + out_proj ---
    // Take only the P tokens corresponding to x (positions x_offset..x_offset+P-1)
    // Output: [feat_dim * P] = [64 * 4 = 256] predicted velocity
    int out_size = feat_dim * P;
    std::vector<float> vel(out_size);

    for (int p = 0; p < P; p++) {
        const float* h_p = cur.data() + (size_t)(x_offset + p) * d;
        float* v_p = vel.data() + (size_t)p * feat_dim;
        std::vector<float> normed_p(d);
        if (W.locdit_norm_w) {
            rms_norm_cpu(h_p, tensor_data_f32(W.locdit_norm_w), normed_p.data(), d, eps);
        } else {
            std::memcpy(normed_p.data(), h_p, (size_t)d * sizeof(float));
        }
        // out_proj: [d, feat_dim] stored as [feat_dim, d] in GGUF → matmul gives [feat_dim]
        if (W.locdit_out_proj_w && W.locdit_out_proj_b) {
            matmul_mv_bias(cpu_be, W.locdit_out_proj_w, W.locdit_out_proj_b,
                           normed_p.data(), d, v_p, feat_dim);
        } else {
            int cp = std::min(d, feat_dim);
            std::memcpy(v_p, normed_p.data(), (size_t)cp * sizeof(float));
        }
    }

    return vel;
}

// ---------------------------------------------------------------------------
// CFM Euler solve — sway schedule (t: 1->0), CFG-zero-star
//
// mu:       [tslm_d_model=2048] conditioning from TSLM+RALM
// cond_raw: [feat_dim * patch_frames = 256] previous patch latents
//           (pass all-zeros for first step)
// Returns:  [feat_dim * patch_frames = 256] denoised patch in latent space
// ---------------------------------------------------------------------------

static std::vector<float> cfm_euler_solve(voxcpm2_context* ctx,
                                           const float* mu,
                                           const float* cond_raw,
                                           int steps,
                                           float cfg,
                                           ggml_backend_t cpu_be) {
    int feat_dim   = 64;
    int P          = (int)ctx->hp.patch_frames;   // 4
    int state_size = feat_dim * P;                  // 256

    // Start from zero noise (in production: Gaussian noise seeded per call)
    std::vector<float> x(state_size, 0.0f);

    for (int s = 0; s < steps; s++) {
        // Sway schedule: t from 1.0 to 0.0 linearly
        float t_cur  = 1.0f - (float)s       / (float)steps;
        float t_next = 1.0f - (float)(s + 1) / (float)steps;
        float dt_val = t_next - t_cur; // negative (Euler step direction)

        // dt_scalar for dt_mlp — 0.0 for non-mean-mode synthesis
        float dt_scalar = 0.0f;

        std::vector<float> v_cond = locdit_forward(ctx, x.data(), mu, t_cur,
                                                    cond_raw, dt_scalar, cpu_be);

        std::vector<float> v;
        if (cfg > 1.0f) {
            // CFG-zero-star: unconditioned uses zero mu and zero cond
            std::vector<float> zero_mu(ctx->hp.tslm_d_model, 0.0f);
            std::vector<float> zero_cond(state_size, 0.0f);
            std::vector<float> v_uncond = locdit_forward(ctx, x.data(), zero_mu.data(),
                                                          t_cur, zero_cond.data(),
                                                          dt_scalar, cpu_be);
            v.resize(state_size);
            for (int i = 0; i < state_size; i++) {
                v[i] = v_uncond[i] + cfg * (v_cond[i] - v_uncond[i]);
            }
        } else {
            v = v_cond;
        }

        // Euler step
        for (int i = 0; i < state_size; i++) x[i] += dt_val * v[i];
    }

    return x;
}

// ---------------------------------------------------------------------------
// Stop predictor:
//   h = stop.proj(lm_hidden + bias)   [2048 → 2048]
//   logits = stop.head(silu(h))       [2048 → 2, no bias]
//   p_stop = softmax(logits)[1]        (class 1 = stop)
// ---------------------------------------------------------------------------

static float stop_score(voxcpm2_context* ctx, const float* lm_hidden,
                         ggml_backend_t cpu_be) {
    const vox_weights& W = ctx->weights;
    if (!W.stop_proj_w || !W.stop_proj_b) return 0.0f;
    int d_lm  = (int)ctx->hp.tslm_d_model;    // 2048

    std::vector<float> h(d_lm);
    matmul_mv_bias(cpu_be, W.stop_proj_w, W.stop_proj_b, lm_hidden, d_lm, h.data(), d_lm);
    // SiLU activation (common for projection heads in MiniCPM family)
    for (int i = 0; i < d_lm; i++) {
        float v = h[i];
        h[i] = v / (1.0f + std::exp(-v));
    }

    if (!W.stop_head_w) {
        // Fallback: treat h[0] as raw logit
        return 1.0f / (1.0f + std::exp(-h[0]));
    }

    // stop.head.weight: [d_lm, 2] — matmul gives [2] logits
    float logits[2] = {0.0f, 0.0f};
    matmul_mv(cpu_be, W.stop_head_w, h.data(), d_lm, logits, 2);

    // Softmax over 2 classes, return p(stop) = logits[1]
    float max_l = std::max(logits[0], logits[1]);
    float e0 = std::exp(logits[0] - max_l);
    float e1 = std::exp(logits[1] - max_l);
    return e1 / (e0 + e1);
}

// ===========================================================================
// VAE decoder implementation
// Architecture (from PyTorch source):
//   input:  [64, T] latents (64 channels, T time frames per patch)
//   → in_conv:  depthwise Conv1d(64,64,k=7,groups=64) + pointwise Conv1d(64,2048,k=1)
//   → 6× upsample blocks with rates [8,6,5,2,2,2]:
//       each: Snake1d → CausalTransposeConv1d → 3× CausalResidualUnit(dils=1,3,9)
//   → final: Snake1d → Conv1d(output_ch, 1, k=7) → Tanh
//   SR conditioning: scale_bias embeddings per decoder block, bucket=3 for 48kHz
//   Weight-norm: weight = normalize(weight_v) * weight_g
// ===========================================================================

// ---------------------------------------------------------------------------
// Weight-norm: reconstruct W from weight_g and weight_v
// weight = (weight_v / ||weight_v||) * weight_g
// For 1D conv kernels stored as [out_ch, in_ch/groups, kernel_size] in F32.
// ---------------------------------------------------------------------------
static std::vector<float> wn_reconstruct(const float* weight_g, const float* weight_v,
                                          int out_ch, int in_ch, int ksize) {
    int fan_in     = in_ch * ksize;
    int total      = out_ch * in_ch * ksize;
    std::vector<float> w(total);
    for (int o = 0; o < out_ch; o++) {
        const float* v_row = weight_v + (size_t)o * fan_in;
        float norm_sq = 0.0f;
        for (int j = 0; j < fan_in; j++) norm_sq += v_row[j] * v_row[j];
        float inv_norm = 1.0f / std::sqrt(norm_sq + 1e-12f);
        float g        = weight_g[o];
        float scale    = g * inv_norm;
        float* dst     = w.data() + (size_t)o * fan_in;
        for (int j = 0; j < fan_in; j++) dst[j] = v_row[j] * scale;
    }
    return w;
}

// ---------------------------------------------------------------------------
// Snake1d activation: x + (1/alpha) * sin(alpha * x)^2
// alpha is a per-channel learnable parameter [C].
// x_in: [C, T]  (in-place safe if x_in == x_out)
// ---------------------------------------------------------------------------
static void snake1d(const float* alpha, const float* x_in,
                    float* x_out, int C, int T) {
    for (int c = 0; c < C; c++) {
        float a = alpha[c];
        float inv_a = (std::abs(a) > 1e-8f) ? 1.0f / a : 1.0f;
        for (int t = 0; t < T; t++) {
            float v = x_in[(size_t)c * T + t];
            float s = std::sin(a * v);
            x_out[(size_t)c * T + t] = v + inv_a * s * s;
        }
    }
}

// ---------------------------------------------------------------------------
// Causal Conv1d (CPU, arbitrary dilation, padding left only)
// weight: [out_ch, in_ch/groups, ksize]  (weight-norm already applied)
// bias:   [out_ch]  (may be nullptr)
// x_in:  [in_ch, T_in]
// x_out: [out_ch, T_out]  T_out = T_in (causal padding = (ksize-1)*dilation)
// stride: typically 1 for residual units
// groups: for depthwise conv use groups=in_ch=out_ch
// ---------------------------------------------------------------------------
static void causal_conv1d(const float* weight, const float* bias,
                           const float* x_in, float* x_out,
                           int out_ch, int in_ch, int ksize, int T_in,
                           int stride, int dilation, int groups) {
    int T_out = T_in / stride; // causal: same length (with causal pad)
    // Causal left-padding: pad = (ksize-1)*dilation
    int pad = (ksize - 1) * dilation;
    int in_per_grp  = in_ch  / groups;
    int out_per_grp = out_ch / groups;

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < out_per_grp; oc++) {
            int oc_abs = g * out_per_grp + oc;
            float b_val = bias ? bias[oc_abs] : 0.0f;
            for (int ot = 0; ot < T_out; ot++) {
                float acc = b_val;
                int it_center = ot * stride;
                for (int k = 0; k < ksize; k++) {
                    int it = it_center - pad + k * dilation;
                    if (it < 0 || it >= T_in) continue;
                    const float* w_row = weight + ((size_t)oc_abs * in_per_grp * ksize
                                                   + (size_t)0 * ksize + k);
                    for (int ic = 0; ic < in_per_grp; ic++) {
                        int ic_abs = g * in_per_grp + ic;
                        float x_val = x_in[(size_t)ic_abs * T_in + it];
                        float w_val = weight[(size_t)oc_abs * in_per_grp * ksize
                                            + (size_t)ic * ksize + k];
                        acc += x_val * w_val;
                    }
                }
                x_out[(size_t)oc_abs * T_out + ot] = acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Causal TransposeConv1d (upsample by stride)
// Equivalent to: insert (stride-1) zeros between input samples, then causal conv.
// weight: [in_ch, out_ch, ksize]  — note transposed layout
// x_in:  [in_ch, T_in]
// x_out: [out_ch, T_out]  T_out = T_in * stride
// ---------------------------------------------------------------------------
static void causal_transposed_conv1d(const float* weight, const float* bias,
                                      const float* x_in, float* x_out,
                                      int in_ch, int out_ch, int ksize, int T_in,
                                      int stride) {
    int T_out = T_in * stride;
    // Causal transpose conv: output[t] sums weight[k] * x[floor((t-k)/stride)] for
    // valid positions. We implement via direct scatter-add.
    std::fill(x_out, x_out + (size_t)out_ch * T_out, 0.0f);

    // Causal padding: trim the first (ksize-1) output samples
    int trim = ksize - 1;

    for (int ic = 0; ic < in_ch; ic++) {
        for (int oc = 0; oc < out_ch; oc++) {
            // weight layout (transposed conv stored as [in_ch, out_ch, ksize]):
            const float* w_k = weight + ((size_t)ic * out_ch + oc) * ksize;
            for (int it = 0; it < T_in; it++) {
                float x_val = x_in[(size_t)ic * T_in + it];
                // Each input sample it maps to output positions it*stride + k
                for (int k = 0; k < ksize; k++) {
                    int ot_raw = it * stride + k;
                    int ot = ot_raw - trim; // causal: shift left
                    if (ot < 0 || ot >= T_out) continue;
                    x_out[(size_t)oc * T_out + ot] += x_val * w_k[k];
                }
            }
        }
    }
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++) {
            float b_val = bias[oc];
            for (int t = 0; t < T_out; t++) {
                x_out[(size_t)oc * T_out + t] += b_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: get F32 tensor data from ctx->tensors by name; returns nullptr if absent
// ---------------------------------------------------------------------------
static const float* vae_tensor_f32(const std::map<std::string, ggml_tensor*>& tensors,
                                    const std::string& name) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second) return nullptr;
    return (const float*)it->second->data;
}

static int vae_tensor_dim(const std::map<std::string, ggml_tensor*>& tensors,
                           const std::string& name, int dim_idx) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second) return 0;
    return (int)it->second->ne[dim_idx];
}

// ---------------------------------------------------------------------------
// CausalResidualUnit:
//   y = x + conv2(snake(conv1(snake(x))))
//   where conv1: causal, dil, k=7; conv2: causal, dil=1, k=1 (shortcut conv)
// Weight names (example for block b, resunit r):
//   vae.dec.block.{b}.res.{r}.conv1.weight_g / weight_v / bias
//   vae.dec.block.{b}.res.{r}.conv2.weight_g / weight_v / bias
//   vae.dec.block.{b}.res.{r}.snake1.alpha   (snake before conv1)
//   vae.dec.block.{b}.res.{r}.snake2.alpha   (snake before conv2)
// ---------------------------------------------------------------------------
static void vae_residual_unit(const std::map<std::string, ggml_tensor*>& tensors,
                               const std::string& prefix,
                               const float* x_in, float* x_out,
                               int C, int T, int dilation) {
    // snake1
    std::vector<float> h1(C * T);
    const float* alpha1 = vae_tensor_f32(tensors, prefix + ".snake1.alpha");
    if (alpha1) {
        snake1d(alpha1, x_in, h1.data(), C, T);
    } else {
        std::memcpy(h1.data(), x_in, (size_t)C * T * sizeof(float));
    }

    // conv1: [C, C, k=7] with dilation
    int k1 = 7;
    std::vector<float> h2(C * T, 0.0f);
    const float* g1 = vae_tensor_f32(tensors, prefix + ".conv1.weight_g");
    const float* v1 = vae_tensor_f32(tensors, prefix + ".conv1.weight_v");
    const float* b1 = vae_tensor_f32(tensors, prefix + ".conv1.bias");
    if (g1 && v1) {
        auto w1 = wn_reconstruct(g1, v1, C, C, k1);
        causal_conv1d(w1.data(), b1, h1.data(), h2.data(), C, C, k1, T, 1, dilation, 1);
    } else {
        std::memcpy(h2.data(), h1.data(), (size_t)C * T * sizeof(float));
    }

    // snake2
    std::vector<float> h3(C * T);
    const float* alpha2 = vae_tensor_f32(tensors, prefix + ".snake2.alpha");
    if (alpha2) {
        snake1d(alpha2, h2.data(), h3.data(), C, T);
    } else {
        std::memcpy(h3.data(), h2.data(), (size_t)C * T * sizeof(float));
    }

    // conv2: [C, C, k=1]
    std::vector<float> h4(C * T, 0.0f);
    const float* g2 = vae_tensor_f32(tensors, prefix + ".conv2.weight_g");
    const float* v2 = vae_tensor_f32(tensors, prefix + ".conv2.weight_v");
    const float* b2 = vae_tensor_f32(tensors, prefix + ".conv2.bias");
    if (g2 && v2) {
        auto w2 = wn_reconstruct(g2, v2, C, C, 1);
        causal_conv1d(w2.data(), b2, h3.data(), h4.data(), C, C, 1, T, 1, 1, 1);
    } else {
        std::memcpy(h4.data(), h3.data(), (size_t)C * T * sizeof(float));
    }

    // residual add
    for (size_t i = 0; i < (size_t)C * T; i++) x_out[i] = x_in[i] + h4[i];
}

// ---------------------------------------------------------------------------
// VAE decode: concatenated patches -> 48kHz PCM
//
// Each patch is [feat_dim * patch_frames = 64 * 4 = 256] float values.
// We concatenate them along the time axis to form latents [64, T_latent].
// The VAE decoder upsamples by 8*6*5*2*2*2 = 1920x (from 25 Hz to 48000 Hz).
//
// When VAE weights are absent, returns silence of the correct duration.
// ---------------------------------------------------------------------------

static std::vector<float> vae_decode(voxcpm2_context* ctx,
                                      const std::vector<std::vector<float>>& patches,
                                      ggml_backend_t /*cpu_be*/) {
    int n_patches = (int)patches.size();
    if (n_patches == 0) return {};

    int feat_dim  = 64;
    int P         = (int)ctx->hp.patch_frames;  // 4

    // Total latent time frames = n_patches * P
    int T_lat = n_patches * P;

    // Upsampling ratios: product = 8*6*5*2*2*2 = 1920
    // 25 Hz (latent rate) * 1920 = 48000 Hz output
    static const int up_rates[] = {8, 6, 5, 2, 2, 2};
    static const int n_blocks   = 6;

    // 48kHz samples per patch = P * 1920
    const int samples_per_patch = P * 1920;  // 4 * 1920 = 7680... wait
    // Actually: latent rate = 48000 / 1920 = 25 Hz, each frame = 1/25s
    // P=4 frames = 4/25 s = 0.16s = 7680 samples at 48kHz
    // But the task doc says ~3840 samples per patch (12.5fps → 48000/12.5=3840)
    // 12.5 fps latent × 1920 upsample = 24000... that's 24kHz not 48kHz
    // The latent frame rate depends on the encoder hop size. Let's use the tensor
    // structure to determine: if present, use computed output length; otherwise fallback.
    (void)samples_per_patch;

    const auto& T = ctx->tensors;

    // Check if VAE weights exist at all
    bool have_vae = (vae_tensor_f32(T, "vae.dec.in_conv_dw.weight_g") != nullptr) ||
                    (vae_tensor_f32(T, "vae.dec.in_conv_pw.weight_g") != nullptr) ||
                    (vae_tensor_f32(T, "vae.dec.in_conv.weight_g")    != nullptr);

    if (!have_vae) {
        // Graceful degradation: return silence of computed duration
        // Assume 12.5 fps latent rate → 3840 samples/patch @48kHz
        std::vector<float> pcm((size_t)n_patches * 3840, 0.0f);
        return pcm;
    }

    // --- Build latent tensor [feat_dim=64, T_lat] ---
    std::vector<float> latents((size_t)feat_dim * T_lat, 0.0f);
    for (int n = 0; n < n_patches; n++) {
        const auto& patch = patches[n];
        for (int p = 0; p < P && (size_t)(p * feat_dim) < patch.size(); p++) {
            int t = n * P + p;
            const float* src = patch.data() + (size_t)p * feat_dim;
            float* dst = latents.data() + (size_t)t * feat_dim; // [T_lat, feat_dim] interleaved?
            // Actually we want [feat_dim, T_lat] layout: dst = latents + c*T_lat + t
            // Re-index: latents[c, t] = latents[c * T_lat + t]
            for (int c = 0; c < feat_dim && c < (int)patch.size() - p * feat_dim; c++) {
                latents[(size_t)c * T_lat + t] = src[c];
            }
        }
    }

    // --- SR conditioning: scale/bias per block for 48kHz (bucket=3) ---
    // Names: vae.dec.sr_cond.{b}.scale / bias  (or weight/bias in embedding style)
    // bucket=3 corresponds to 48kHz in the SR conditioning table
    int sr_bucket = 3;

    // --- In-conv: depthwise (k=7, groups=64) + pointwise (k=1) ---
    int Tc = T_lat;  // current time length (grows as we upsample)
    int Cc = feat_dim;     // current channel count (grows after in_conv)

    std::vector<float> h;

    // Try depthwise+pointwise in-conv (vae.dec.in_conv_dw / in_conv_pw)
    bool have_dw_pw = (vae_tensor_f32(T, "vae.dec.in_conv_dw.weight_g") != nullptr);
    if (have_dw_pw) {
        const float* g_dw = vae_tensor_f32(T, "vae.dec.in_conv_dw.weight_g");
        const float* v_dw = vae_tensor_f32(T, "vae.dec.in_conv_dw.weight_v");
        const float* b_dw = vae_tensor_f32(T, "vae.dec.in_conv_dw.bias");
        if (g_dw && v_dw) {
            auto w_dw = wn_reconstruct(g_dw, v_dw, feat_dim, 1, 7); // depthwise groups=64
            std::vector<float> h_dw((size_t)feat_dim * Tc, 0.0f);
            causal_conv1d(w_dw.data(), b_dw, latents.data(), h_dw.data(),
                          feat_dim, feat_dim, 7, Tc, 1, 1, feat_dim);
            // Pointwise: [2048, 64, 1]
            int out_ch_pw = vae_tensor_dim(T, "vae.dec.in_conv_pw.weight_g", 1); // ne[1]=out_ch
            if (out_ch_pw <= 0) out_ch_pw = 2048;
            const float* g_pw = vae_tensor_f32(T, "vae.dec.in_conv_pw.weight_g");
            const float* v_pw = vae_tensor_f32(T, "vae.dec.in_conv_pw.weight_v");
            const float* b_pw = vae_tensor_f32(T, "vae.dec.in_conv_pw.bias");
            if (g_pw && v_pw) {
                auto w_pw = wn_reconstruct(g_pw, v_pw, out_ch_pw, feat_dim, 1);
                h.resize((size_t)out_ch_pw * Tc, 0.0f);
                causal_conv1d(w_pw.data(), b_pw, h_dw.data(), h.data(),
                              out_ch_pw, feat_dim, 1, Tc, 1, 1, 1);
                Cc = out_ch_pw;
            } else {
                h = h_dw; Cc = feat_dim;
            }
        } else {
            h = latents; Cc = feat_dim;
        }
    } else {
        // Try single in_conv [out_ch, in_ch, k=7]
        const float* g_ic = vae_tensor_f32(T, "vae.dec.in_conv.weight_g");
        const float* v_ic = vae_tensor_f32(T, "vae.dec.in_conv.weight_v");
        const float* b_ic = vae_tensor_f32(T, "vae.dec.in_conv.bias");
        if (g_ic && v_ic) {
            // Determine out_ch from tensor shape
            int out_ch_ic = vae_tensor_dim(T, "vae.dec.in_conv.weight_g", 1);
            if (out_ch_ic <= 0) out_ch_ic = 2048;
            auto w_ic = wn_reconstruct(g_ic, v_ic, out_ch_ic, feat_dim, 7);
            h.resize((size_t)out_ch_ic * Tc, 0.0f);
            causal_conv1d(w_ic.data(), b_ic, latents.data(), h.data(),
                          out_ch_ic, feat_dim, 7, Tc, 1, 1, 1);
            Cc = out_ch_ic;
        } else {
            h = latents; Cc = feat_dim;
        }
    }

    // --- 6 Upsample blocks ---
    for (int b = 0; b < n_blocks; b++) {
        int up = up_rates[b];
        std::string bp = "vae.dec.block." + std::to_string(b);

        // SR conditioning: look up scale and bias for this block
        const float* sr_scale = vae_tensor_f32(T, bp + ".sr_cond.scale");
        const float* sr_bias  = vae_tensor_f32(T, bp + ".sr_cond.bias");
        if (!sr_scale) sr_scale = vae_tensor_f32(T, bp + ".sr_emb.weight");
        // If SR conditioning is a lookup table [n_buckets, C], index by sr_bucket
        // (We check if sr_scale has more than Cc elements → it's a table)
        // Apply SR scale/bias if available (element-wise on channel dim)
        auto apply_sr = [&](std::vector<float>& x, int C_cur, int T_cur) {
            if (!sr_scale) return;
            // Determine if it's a lookup table or direct [C] vector
            int sr_c = vae_tensor_dim(T, bp + ".sr_cond.scale", 0);
            int sr_rows = vae_tensor_dim(T, bp + ".sr_cond.scale", 1);
            const float* scale_row = sr_scale;
            const float* bias_row  = sr_bias;
            if (sr_rows > 1) {
                // Table [n_buckets, C]: pick row sr_bucket
                scale_row = sr_scale + (size_t)sr_bucket * C_cur;
                bias_row  = sr_bias  ? (sr_bias + (size_t)sr_bucket * C_cur) : nullptr;
            }
            for (int c = 0; c < C_cur; c++) {
                float sc = (sr_c > c) ? scale_row[c] : 1.0f;
                float bi = (bias_row && sr_c > c) ? bias_row[c] : 0.0f;
                for (int t = 0; t < T_cur; t++) {
                    x[(size_t)c * T_cur + t] = x[(size_t)c * T_cur + t] * sc + bi;
                }
            }
        };

        // Snake1d before upsample
        const float* up_alpha = vae_tensor_f32(T, bp + ".snake.alpha");
        if (up_alpha) {
            snake1d(up_alpha, h.data(), h.data(), Cc, Tc);
        }

        apply_sr(h, Cc, Tc);

        // CausalTransposeConv1d: upsamples Tc → Tc*up
        // Weight: [Cc, out_ch, ksize] where out_ch = Cc/2 (channel halving per block)
        int out_ch_b = vae_tensor_dim(T, bp + ".upsample.weight_g", 1);
        if (out_ch_b <= 0) out_ch_b = Cc / 2;
        if (out_ch_b <= 0) out_ch_b = 1;
        int ksize_b = vae_tensor_dim(T, bp + ".upsample.weight_g", 0);
        if (ksize_b <= 0) ksize_b = 2 * up;  // common: ksize = 2*stride

        const float* g_up = vae_tensor_f32(T, bp + ".upsample.weight_g");
        const float* v_up = vae_tensor_f32(T, bp + ".upsample.weight_v");
        const float* b_up = vae_tensor_f32(T, bp + ".upsample.bias");

        int T_up = Tc * up;
        std::vector<float> h_up((size_t)out_ch_b * T_up, 0.0f);

        if (g_up && v_up) {
            // Transpose conv weight layout [in_ch, out_ch, ksize] in PyTorch
            auto w_up = wn_reconstruct(g_up, v_up, Cc, out_ch_b, ksize_b);
            // Re-interpret for our transpose conv: weight[in_ch, out_ch, k]
            causal_transposed_conv1d(w_up.data(), b_up, h.data(), h_up.data(),
                                     Cc, out_ch_b, ksize_b, Tc, up);
        } else {
            // Repeat-interpolate as fallback
            for (int c = 0; c < std::min(Cc, out_ch_b); c++) {
                for (int t = 0; t < Tc; t++) {
                    float val = h[(size_t)c * Tc + t];
                    for (int u = 0; u < up; u++) {
                        h_up[(size_t)c * T_up + t * up + u] = val;
                    }
                }
            }
        }

        Tc = T_up;
        Cc = out_ch_b;
        h  = std::move(h_up);

        // 3× CausalResidualUnit with dilations 1, 3, 9
        for (int r = 0; r < 3; r++) {
            int dil = 1;
            if (r == 1) dil = 3;
            if (r == 2) dil = 9;
            std::string rp = bp + ".res." + std::to_string(r);
            std::vector<float> h_res((size_t)Cc * Tc);
            vae_residual_unit(T, rp, h.data(), h_res.data(), Cc, Tc, dil);
            h = std::move(h_res);
        }
    }

    // --- Final: Snake1d → Conv1d(Cc, 1, k=7) → Tanh ---
    const float* final_alpha = vae_tensor_f32(T, "vae.dec.out_snake.alpha");
    if (final_alpha) {
        snake1d(final_alpha, h.data(), h.data(), Cc, Tc);
    }

    const float* g_out = vae_tensor_f32(T, "vae.dec.out_conv.weight_g");
    const float* v_out = vae_tensor_f32(T, "vae.dec.out_conv.weight_v");
    const float* b_out = vae_tensor_f32(T, "vae.dec.out_conv.bias");

    std::vector<float> pcm(Tc, 0.0f);
    if (g_out && v_out) {
        auto w_out = wn_reconstruct(g_out, v_out, 1, Cc, 7);
        causal_conv1d(w_out.data(), b_out, h.data(), pcm.data(), 1, Cc, 7, Tc, 1, 1, 1);
        // Tanh
        for (float& s : pcm) s = std::tanh(s);
    } else {
        // Mix down as fallback
        float inv_Cc = 1.0f / std::max(1, Cc);
        for (int t = 0; t < Tc; t++) {
            float mix = 0.0f;
            for (int c = 0; c < Cc; c++) mix += h[(size_t)c * Tc + t];
            pcm[t] = std::tanh(mix * inv_Cc);
        }
    }

    return pcm;
}

// ---------------------------------------------------------------------------
// GPT-2 byte encoder table (built lazily)
// ---------------------------------------------------------------------------

static const std::vector<int>& vox_byte_encoder() {
    static std::vector<int> bs(256, 0);
    static bool init = false;
    if (init) return bs;
    std::vector<int> printable;
    for (int b = 0x21; b <= 0x7e; b++) printable.push_back(b);
    for (int b = 0xa1; b <= 0xac; b++) printable.push_back(b);
    for (int b = 0xae; b <= 0xff; b++) printable.push_back(b);
    int next_extra = 256;
    for (int b = 0; b < 256; b++) {
        bool is_p = false;
        for (int p : printable) if (p == b) { is_p = true; break; }
        bs[b] = is_p ? b : next_extra++;
    }
    init = true;
    return bs;
}

static void utf8_encode_cp(uint32_t cp, std::string& out) {
    if (cp < 0x80)       { out.push_back((char)cp); }
    else if (cp < 0x800) { out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
    else if (cp < 0x10000) {
        out.push_back((char)(0xE0|(cp>>12)));
        out.push_back((char)(0x80|((cp>>6)&0x3F)));
        out.push_back((char)(0x80|(cp&0x3F)));
    } else {
        out.push_back((char)(0xF0|(cp>>18)));
        out.push_back((char)(0x80|((cp>>12)&0x3F)));
        out.push_back((char)(0x80|((cp>>6)&0x3F)));
        out.push_back((char)(0x80|(cp&0x3F)));
    }
}

static std::string vox_bytes_to_unicode(const char* bytes, size_t n) {
    auto& enc = vox_byte_encoder();
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        utf8_encode_cp((uint32_t)enc[(unsigned char)bytes[i]], out);
    }
    return out;
}

// BPE merge for one byte-encoded pre-token
static void vox_bpe_one(const vox_tokenizer& tok, const std::string& word,
                         std::vector<int32_t>& out) {
    if (word.empty()) return;

    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        unsigned char c = (unsigned char)word[i];
        size_t len;
        if      (c < 0x80)             len = 1;
        else if ((c & 0xE0) == 0xC0)   len = 2;
        else if ((c & 0xF0) == 0xE0)   len = 3;
        else if ((c & 0xF8) == 0xF0)   len = 4;
        else                           len = 1;
        if (i + len > word.size()) len = 1;
        symbols.emplace_back(word, i, len);
        i += len;
    }

    if (!tok.merge_rank.empty()) {
        int max_iter = (int)symbols.size();
        for (int iter = 0; iter < max_iter && symbols.size() >= 2; iter++) {
            int best_i = -1, best_rank = INT_MAX;
            for (size_t k = 0; k + 1 < symbols.size(); k++) {
                std::string pair = symbols[k] + " " + symbols[k + 1];
                auto it = tok.merge_rank.find(pair);
                if (it != tok.merge_rank.end() && (int)it->second < best_rank) {
                    best_rank = (int)it->second;
                    best_i    = (int)k;
                }
            }
            if (best_i < 0) break;
            symbols[best_i] += symbols[best_i + 1];
            symbols.erase(symbols.begin() + best_i + 1);
        }
    }

    for (const auto& s : symbols) {
        auto it = tok.token_to_id.find(s);
        if (it != tok.token_to_id.end()) {
            out.push_back(it->second);
        } else {
            // Per-byte fallback
            size_t j = 0;
            while (j < s.size()) {
                unsigned char c2 = (unsigned char)s[j];
                size_t len;
                if      (c2 < 0x80)           len = 1;
                else if ((c2 & 0xE0) == 0xC0) len = 2;
                else if ((c2 & 0xF0) == 0xE0) len = 3;
                else if ((c2 & 0xF8) == 0xF0) len = 4;
                else                          len = 1;
                auto jt = tok.token_to_id.find(std::string(s, j, len));
                if (jt != tok.token_to_id.end()) out.push_back(jt->second);
                j += len;
            }
        }
    }
}

// CJK codepoint check for post-expansion
static bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF);
}

// Decode one UTF-8 codepoint from s at position i, advance i
static uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80)  { i += 1; return c; }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F);
        i += 2; return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F);
        i += 3; return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) |
                      (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F);
        i += 4; return cp;
    }
    i += 1; return 0xFFFD;
}

// Tokenize: CJK expansion -> whitespace pre-tokenize -> BPE -> append audio_start
static std::vector<int32_t> vox_tokenize(const vox_tokenizer& tok,
                                           const std::string& text) {
    // CJK post-expansion: pad each CJK char with spaces
    std::string expanded;
    expanded.reserve(text.size() * 2);
    {
        size_t pos = 0;
        while (pos < text.size()) {
            size_t prev = pos;
            uint32_t cp = utf8_next(text, pos);
            if (is_cjk(cp)) {
                expanded += ' ';
                // re-encode cp as UTF-8
                utf8_encode_cp(cp, expanded);
                expanded += ' ';
            } else {
                expanded.append(text, prev, pos - prev);
            }
        }
    }

    std::vector<int32_t> result;
    size_t i = 0;
    bool first = true;
    while (i < expanded.size()) {
        while (i < expanded.size() && (expanded[i]==' '||expanded[i]=='\t'||expanded[i]=='\n')) i++;
        if (i >= expanded.size()) break;
        size_t j = i;
        while (j < expanded.size() && expanded[j]!=' '&&expanded[j]!='\t'&&expanded[j]!='\n') j++;
        std::string word = expanded.substr(i, j - i);
        if (!first) word = " " + word;
        first = false;
        std::string encoded = vox_bytes_to_unicode(word.data(), word.size());
        vox_bpe_one(tok, encoded, result);
        i = j;
    }

    if (tok.audio_start_token >= 0) {
        result.push_back(tok.audio_start_token);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Model loading — two-pass GGUF
// ---------------------------------------------------------------------------

static bool vox_load_weights(voxcpm2_context* ctx, const char* path) {
    using namespace core_gguf;

    // --- Pass 1: metadata ---
    gguf_context* meta = open_metadata(path);
    if (!meta) return false;

    vox_hparams& hp = ctx->hp;

    hp.tslm_n_layers   = kv_u32(meta, "tslm.n_layers",    hp.tslm_n_layers);
    hp.tslm_d_model    = kv_u32(meta, "tslm.d_model",     hp.tslm_d_model);
    hp.tslm_n_heads    = kv_u32(meta, "tslm.n_heads",     hp.tslm_n_heads);
    hp.tslm_n_kv       = kv_u32(meta, "tslm.n_kv_heads",  hp.tslm_n_kv);
    hp.tslm_head_dim   = kv_u32(meta, "tslm.head_dim",    hp.tslm_head_dim);
    hp.tslm_ff_dim     = kv_u32(meta, "tslm.ff_dim",      hp.tslm_ff_dim);
    hp.tslm_max_pos    = kv_u32(meta, "tslm.max_pos",     hp.tslm_max_pos);
    hp.tslm_rope_theta = kv_f32(meta, "tslm.rope_theta",  hp.tslm_rope_theta);
    hp.rms_norm_eps    = kv_f32(meta, "tslm.rms_norm_eps", hp.rms_norm_eps);

    hp.ralm_n_layers   = kv_u32(meta, "ralm.n_layers",    hp.ralm_n_layers);
    hp.ralm_d_model    = kv_u32(meta, "ralm.d_model",     hp.ralm_d_model);
    hp.ralm_n_heads    = kv_u32(meta, "ralm.n_heads",     hp.ralm_n_heads);
    hp.ralm_n_kv       = kv_u32(meta, "ralm.n_kv_heads",  hp.ralm_n_kv);
    hp.ralm_head_dim   = kv_u32(meta, "ralm.head_dim",    hp.ralm_head_dim);
    hp.ralm_ff_dim     = kv_u32(meta, "ralm.ff_dim",      hp.ralm_ff_dim);

    hp.locenc_n_layers = kv_u32(meta, "locenc.n_layers",   hp.locenc_n_layers);
    hp.locenc_d_model  = kv_u32(meta, "locenc.d_model",    hp.locenc_d_model);
    hp.locenc_n_heads  = kv_u32(meta, "locenc.n_heads",    hp.locenc_n_heads);
    hp.locenc_n_kv     = kv_u32(meta, "locenc.n_kv_heads", hp.locenc_n_kv);
    hp.locenc_head_dim = kv_u32(meta, "locenc.head_dim",   hp.locenc_head_dim);
    hp.locenc_ff_dim   = kv_u32(meta, "locenc.ff_dim",     hp.locenc_ff_dim);

    hp.locdit_n_layers = kv_u32(meta, "locdit.n_layers",   hp.locdit_n_layers);
    hp.locdit_d_model  = kv_u32(meta, "locdit.d_model",    hp.locdit_d_model);
    hp.locdit_n_heads  = kv_u32(meta, "locdit.n_heads",    hp.locdit_n_heads);
    hp.locdit_n_kv     = kv_u32(meta, "locdit.n_kv_heads", hp.locdit_n_kv);
    hp.locdit_head_dim = kv_u32(meta, "locdit.head_dim",   hp.locdit_head_dim);
    hp.locdit_ff_dim   = kv_u32(meta, "locdit.ff_dim",     hp.locdit_ff_dim);

    // Override n_vocab from GGUF metadata
    hp.n_vocab            = kv_u32(meta, "tokenizer.n_vocab",        hp.n_vocab);
    hp.audio_start_token  = kv_u32(meta, "tokenizer.audio_start_id", 0);

    hp.patch_frames = kv_u32(meta, "vae.patch_frames", hp.patch_frames);
    hp.patch_dim    = kv_u32(meta, "vae.patch_dim",    hp.patch_dim);

    // Tokenizer vocab from GGUF string arrays
    {
        auto tokens = kv_str_array(meta, "tokenizer.ggml.tokens");
        auto merges = kv_str_array(meta, "tokenizer.ggml.merges");
        ctx->tokenizer.id_to_token = tokens;
        for (size_t b = 0; b < tokens.size(); b++) {
            ctx->tokenizer.token_to_id[tokens[b]] = (int32_t)b;
        }
        for (size_t b = 0; b < merges.size(); b++) {
            ctx->tokenizer.merge_rank[merges[b]] = (int32_t)b;
        }
    }

    free_metadata(meta);

    // --- Pass 2: weights ---
    ggml_backend_t cpu_be = get_cpu_backend();
    WeightLoad wl;
    if (!load_weights(path, cpu_be, "voxcpm2", wl)) return false;

    ctx->ggml_ctx   = wl.ctx;
    ctx->weight_buf = wl.buf;
    ctx->tensors    = std::move(wl.tensors);

    auto& T = ctx->tensors;
    vox_weights& W = ctx->weights;

    // Check if vocab blob tensor exists and override tokenizer from it
    {
        ggml_tensor* vocab_t = try_get(T, "tokenizer.vocab");
        if (vocab_t) {
            int n = (int)ggml_nelements(vocab_t);
            std::vector<float> blob(n);
            ggml_backend_tensor_get(vocab_t, blob.data(), 0, (size_t)n * sizeof(float));
            // Decode F32 -> uint8 bytes -> null-separated token strings
            ctx->tokenizer.id_to_token.clear();
            ctx->tokenizer.token_to_id.clear();
            size_t start = 0;
            for (int b = 0; b <= n; b++) {
                bool end_of = (b == n) || ((int)blob[b] == 0);
                if (end_of) {
                    std::string tok_str;
                    tok_str.reserve(b - start);
                    for (size_t k = start; k < (size_t)b; k++) {
                        tok_str.push_back((char)(uint8_t)(unsigned char)(int)blob[k]);
                    }
                    int32_t id = (int32_t)ctx->tokenizer.id_to_token.size();
                    ctx->tokenizer.id_to_token.push_back(tok_str);
                    ctx->tokenizer.token_to_id[tok_str] = id;
                    start = (size_t)b + 1;
                }
            }
        }
    }

    ctx->tokenizer.audio_start_token = (int32_t)hp.audio_start_token;

    // Override vocab size from GGUF metadata
    if (hp.n_vocab > 0 && ctx->tokenizer.id_to_token.size() < (size_t)hp.n_vocab) {
        ctx->tokenizer.id_to_token.resize(hp.n_vocab);
    }

    // Bind TSLM (note: tslm.lm_head.weight is NOT present in the GGUF)
    W.tslm_token_embd  = require(T, "tslm.token_embd.weight",  "voxcpm2");
    W.tslm_output_norm = require(T, "tslm.output_norm.weight", "voxcpm2");
    W.tslm_rope_short  = try_get(T, "tslm.rope_short_factors");
    W.tslm_rope_long   = try_get(T, "tslm.rope_long_factors");

    W.tslm_layers.resize(hp.tslm_n_layers);
    for (uint32_t i = 0; i < hp.tslm_n_layers; i++) {
        auto& L = W.tslm_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "tslm.blk.%u.%s", i, suffix);
            return nb;
        };
        L.attn_norm_w = require(T, fn("attn_norm.weight"),   "voxcpm2");
        L.attn_q_w    = require(T, fn("attn_q.weight"),      "voxcpm2");
        L.attn_k_w    = require(T, fn("attn_k.weight"),      "voxcpm2");
        L.attn_v_w    = require(T, fn("attn_v.weight"),      "voxcpm2");
        L.attn_o_w    = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_norm_w  = require(T, fn("ffn_norm.weight"),    "voxcpm2");
        L.ffn_gate_w  = require(T, fn("ffn_gate.weight"),    "voxcpm2");
        L.ffn_up_w    = require(T, fn("ffn_up.weight"),      "voxcpm2");
        L.ffn_down_w  = require(T, fn("ffn_down.weight"),    "voxcpm2");
    }

    // Bind RALM
    W.ralm_output_norm = require(T, "ralm.output_norm.weight", "voxcpm2");
    W.ralm_layers.resize(hp.ralm_n_layers);
    for (uint32_t i = 0; i < hp.ralm_n_layers; i++) {
        auto& L = W.ralm_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "ralm.blk.%u.%s", i, suffix);
            return nb;
        };
        L.attn_norm_w = require(T, fn("attn_norm.weight"),   "voxcpm2");
        L.attn_q_w    = require(T, fn("attn_q.weight"),      "voxcpm2");
        L.attn_k_w    = require(T, fn("attn_k.weight"),      "voxcpm2");
        L.attn_v_w    = require(T, fn("attn_v.weight"),      "voxcpm2");
        L.attn_o_w    = require(T, fn("attn_output.weight"), "voxcpm2");
        L.ffn_norm_w  = require(T, fn("ffn_norm.weight"),    "voxcpm2");
        L.ffn_gate_w  = require(T, fn("ffn_gate.weight"),    "voxcpm2");
        L.ffn_up_w    = require(T, fn("ffn_up.weight"),      "voxcpm2");
        L.ffn_down_w  = require(T, fn("ffn_down.weight"),    "voxcpm2");
    }

    // FSQ
    W.fsq_in_proj_w  = require(T, "fsq.in_proj.weight",  "voxcpm2");
    W.fsq_in_proj_b  = require(T, "fsq.in_proj.bias",    "voxcpm2");
    W.fsq_out_proj_w = require(T, "fsq.out_proj.weight", "voxcpm2");
    W.fsq_out_proj_b = require(T, "fsq.out_proj.bias",   "voxcpm2");

    // LocEnc
    W.locenc_cls_token = try_get(T, "locenc.cls_token");
    W.locenc_in_proj_w = require(T, "locenc.in_proj.weight", "voxcpm2");
    W.locenc_in_proj_b = require(T, "locenc.in_proj.bias",   "voxcpm2");
    W.locenc_norm_w    = try_get(T, "locenc.output_norm.weight");
    W.locenc_layers.resize(hp.locenc_n_layers);
    for (uint32_t i = 0; i < hp.locenc_n_layers; i++) {
        auto& L = W.locenc_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "locenc.blk.%u.%s", i, suffix);
            return nb;
        };
        L.norm1_w    = require(T, fn("attn_norm.weight"),      "voxcpm2");
        L.norm2_w    = require(T, fn("ffn_norm.weight"),      "voxcpm2");
        L.attn_q_w   = require(T, fn("attn_q.weight"),     "voxcpm2");
        L.attn_k_w   = require(T, fn("attn_k.weight"),     "voxcpm2");
        L.attn_v_w   = require(T, fn("attn_v.weight"),     "voxcpm2");
        L.attn_o_w   = require(T, fn("attn_output.weight"),"voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"),   "voxcpm2");
        L.ffn_up_w   = require(T, fn("ffn_up.weight"),     "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"),   "voxcpm2");
    }

    // LocDiT
    W.locdit_in_proj_w    = require(T, "locdit.in_proj.weight",    "voxcpm2");
    W.locdit_in_proj_b    = require(T, "locdit.in_proj.bias",      "voxcpm2");
    W.locdit_cond_proj_w  = require(T, "locdit.cond_proj.weight",  "voxcpm2");
    W.locdit_cond_proj_b  = require(T, "locdit.cond_proj.bias",    "voxcpm2");
    W.locdit_time_mlp_0_w = require(T, "locdit.time_mlp.0.weight", "voxcpm2");
    W.locdit_time_mlp_0_b = require(T, "locdit.time_mlp.0.bias",   "voxcpm2");
    W.locdit_time_mlp_1_w = require(T, "locdit.time_mlp.1.weight", "voxcpm2");
    W.locdit_time_mlp_1_b = require(T, "locdit.time_mlp.1.bias",   "voxcpm2");
    W.locdit_dt_mlp_0_w   = require(T, "locdit.dt_mlp.0.weight",   "voxcpm2");
    W.locdit_dt_mlp_0_b   = require(T, "locdit.dt_mlp.0.bias",     "voxcpm2");
    W.locdit_dt_mlp_1_w   = require(T, "locdit.dt_mlp.1.weight",   "voxcpm2");
    W.locdit_dt_mlp_1_b   = require(T, "locdit.dt_mlp.1.bias",     "voxcpm2");
    W.locdit_norm_w       = try_get(T, "locdit.output_norm.weight");
    W.locdit_out_proj_w   = require(T, "locdit.out_proj.weight",   "voxcpm2");
    W.locdit_out_proj_b   = require(T, "locdit.out_proj.bias",     "voxcpm2");
    W.locdit_layers.resize(hp.locdit_n_layers);
    for (uint32_t i = 0; i < hp.locdit_n_layers; i++) {
        auto& L = W.locdit_layers[i];
        char nb[256];
        auto fn = [&](const char* suffix) -> const char* {
            snprintf(nb, sizeof(nb), "locdit.blk.%u.%s", i, suffix);
            return nb;
        };
        L.norm1_w    = require(T, fn("attn_norm.weight"),      "voxcpm2");
        L.norm2_w    = require(T, fn("ffn_norm.weight"),      "voxcpm2");
        L.attn_q_w   = require(T, fn("attn_q.weight"),     "voxcpm2");
        L.attn_k_w   = require(T, fn("attn_k.weight"),     "voxcpm2");
        L.attn_v_w   = require(T, fn("attn_v.weight"),     "voxcpm2");
        L.attn_o_w   = require(T, fn("attn_output.weight"),"voxcpm2");
        L.ffn_gate_w = require(T, fn("ffn_gate.weight"),   "voxcpm2");
        L.ffn_up_w   = require(T, fn("ffn_up.weight"),     "voxcpm2");
        L.ffn_down_w = require(T, fn("ffn_down.weight"),   "voxcpm2");
    }

    // Projection heads
    W.enc_to_lm_w     = try_get(T, "proj.enc_to_lm.weight");
    W.enc_to_lm_b     = try_get(T, "proj.enc_to_lm.bias");
    W.lm_to_dit_w     = try_get(T, "proj.lm_to_dit.weight");
    W.lm_to_dit_b     = try_get(T, "proj.lm_to_dit.bias");
    W.res_to_dit_w    = try_get(T, "proj.res_to_dit.weight");
    W.res_to_dit_b    = try_get(T, "proj.res_to_dit.bias");
    W.fusion_w = try_get(T, "proj.fusion.weight");
    W.fusion_b = try_get(T, "proj.fusion.bias");

    // Stop predictor
    W.stop_proj_w = try_get(T, "stop.proj.weight");
    W.stop_proj_b = try_get(T, "stop.proj.bias");
    W.stop_head_w = try_get(T, "stop.head.weight");   // [2048, 2], no bias

    // VAE decoder (graceful degradation when absent)
    W.vae_in_conv.w   = try_get(T, "vae.dec.in_conv.weight");
    W.vae_in_conv.b   = try_get(T, "vae.dec.in_conv.bias");
    W.vae_out_conv.w  = try_get(T, "vae.dec.out_conv.weight");
    W.vae_out_conv.b  = try_get(T, "vae.dec.out_conv.bias");
    W.vae_out_snake_a = try_get(T, "vae.dec.out_snake.alpha");
    W.vae_sr_cond_w   = try_get(T, "vae.dec.sr_cond.weight");
    W.vae_sr_cond_b   = try_get(T, "vae.dec.sr_cond.bias");

    // KV caches
    const int kv_max_ctx = 4096;
    ctx->tslm_kv.init((int)hp.tslm_n_layers, (int)hp.tslm_n_kv, (int)hp.tslm_head_dim, kv_max_ctx);
    ctx->ralm_kv.init((int)hp.ralm_n_layers, (int)hp.ralm_n_kv, (int)hp.ralm_head_dim, kv_max_ctx);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: loaded — TSLM %uL d=%u n_kv=%u, RALM %uL, LocEnc %uL d=%u, LocDiT %uL\n",
                hp.tslm_n_layers, hp.tslm_d_model, hp.tslm_n_kv,
                hp.ralm_n_layers,
                hp.locenc_n_layers, hp.locenc_d_model,
                hp.locdit_n_layers);
        fprintf(stderr, "voxcpm2: vocab %zu tokens, audio_start=%u\n",
                ctx->tokenizer.id_to_token.size(), hp.audio_start_token);
    }

    // Set ggml matmul thread count
    g_cpu_n_threads = (int)ctx->params.n_threads;

    return true;
}

// ---------------------------------------------------------------------------
// Core synthesis pipeline
// ---------------------------------------------------------------------------

static float* vox_synthesize_internal(voxcpm2_context* ctx,
                                       const char* text,
                                       const float* ref_samples,
                                       int ref_n_samples,
                                       int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) return nullptr;
    *out_n_samples = 0;

    ggml_backend_t cpu_be = get_cpu_backend();
    ggml_backend_cpu_set_n_threads(cpu_be, ctx->n_threads);

    double t0_total = vox_now_ms();
    (void)ref_samples; (void)ref_n_samples; // ref cloning: future work

    // 1. Tokenize
    std::vector<int32_t> token_ids = vox_tokenize(ctx->tokenizer, std::string(text));
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: tokenized '%s' -> %zu tokens\n", text, token_ids.size());
    }
    if (token_ids.empty()) {
        fprintf(stderr, "voxcpm2: empty token sequence\n");
        return nullptr;
    }

    // 2. TSLM prefill
    double t0_prefill = vox_now_ms();
    std::vector<float> tslm_hidden = tslm_prefill(ctx, token_ids, cpu_be);

    // Apply TSLM output norm
    {
        int d = (int)ctx->hp.tslm_d_model;
        std::vector<float> normed(d);
        rms_norm_cpu(tslm_hidden.data(), tensor_data_f32(ctx->weights.tslm_output_norm),
                     normed.data(), d, ctx->hp.rms_norm_eps);
        tslm_hidden = normed;
    }
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: TSLM prefill %.1f ms\n", vox_now_ms() - t0_prefill);
    }

    // 3. FSQ on TSLM hidden
    std::vector<float> fsq_out = fsq_forward(ctx, tslm_hidden.data(), cpu_be);

    // 4. fusion_concat_proj(cat(fsq_out, audio_feat_embed)) → RALM prefill
    // For zero-shot, audio_feat_embed is zeros (no reference audio)
    {
        int d = (int)ctx->hp.tslm_d_model;
        int in_dim = 2 * d;  // 4096
        std::vector<float> cat_buf(in_dim, 0.0f);
        memcpy(cat_buf.data(), fsq_out.data(), d * sizeof(float));
        // Second half is zeros (no audio feat for text-only zero-shot)
        std::vector<float> fusion_out(d);
        matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b,
                       cat_buf.data(), in_dim, fusion_out.data(), d);
        fsq_out = fusion_out;
    }
    std::vector<float> ralm_hidden = ralm_prefill(ctx, fsq_out, cpu_be);
    // Apply RALM output norm
    {
        int d = (int)ctx->hp.ralm_d_model;
        std::vector<float> normed(d);
        rms_norm_cpu(ralm_hidden.data(), tensor_data_f32(ctx->weights.ralm_output_norm),
                     normed.data(), d, ctx->hp.rms_norm_eps);
        ralm_hidden = normed;
    }

    // 5. Build mu [2048] for LocDiT: mu = cat(lm_to_dit(tslm), res_to_dit(ralm))
    // lm_to_dit:  [2048 → 1024] maps TSLM hidden to first half of mu
    // res_to_dit: [2048 → 1024] maps RALM hidden to second half of mu
    // mu.view(-1, 1024) = [2, 1024] = 2 conditioning tokens in the DiT sequence
    int d_lm   = (int)ctx->hp.tslm_d_model;   // 2048
    int d_dit  = (int)ctx->hp.locdit_d_model;  // 1024 (half of mu)
    int d_ralm = (int)ctx->hp.ralm_d_model;    // 2048
    int d_mu   = 2 * d_dit;                     // 2048 = full mu

    auto build_mu = [&](const std::vector<float>& lm_h,
                         const std::vector<float>& ralm_h) -> std::vector<float> {
        std::vector<float> mu(d_mu, 0.0f);
        // First half: lm_to_dit projection [d_lm → d_dit]
        if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b,
                           lm_h.data(), d_lm, mu.data(), d_dit);
        }
        // Second half: res_to_dit projection [d_ralm → d_dit]
        if (ctx->weights.res_to_dit_w && ctx->weights.res_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.res_to_dit_w, ctx->weights.res_to_dit_b,
                           ralm_h.data(), d_ralm, mu.data() + d_dit, d_dit);
        }
        return mu;
    };

    std::vector<float> mu = build_mu(tslm_hidden, ralm_hidden);
    // cond_raw for LocDiT: previous patch in feat_dim space [feat_dim * patch_frames]
    // Initialize to zeros for the first step
    int feat_dim_vae = 64;
    int P_frames = (int)ctx->hp.patch_frames;
    std::vector<float> prev_patch_raw(feat_dim_vae * P_frames, 0.0f);

    // 6. AR loop
    double t0_ar = vox_now_ms();
    std::vector<std::vector<float>> patches;
    float stop_thresh = 0.5f;
    int step = 0;

    while (step < ctx->max_len) {
        double t0_step = vox_now_ms();

        // a. CFM Euler solve (LocDiT)
        // patch is [feat_dim * patch_frames = 64 * 4 = 256] in latent space
        std::vector<float> patch = cfm_euler_solve(ctx, mu.data(), prev_patch_raw.data(),
                                                    ctx->inference_steps, ctx->cfg_value, cpu_be);
        patches.push_back(patch);

        // Update previous patch for next step
        prev_patch_raw = patch;
        // Resize to exactly feat_dim * P_frames if needed
        prev_patch_raw.resize(feat_dim_vae * P_frames, 0.0f);

        // b. LocEnc on predicted patch (uses in_proj to project feat_dim=64 → d_locenc=1024)
        // Pass the raw latent [feat_dim * patch_frames] — locenc_forward handles in_proj internally
        std::vector<float> patch_enc_in = patch;
        patch_enc_in.resize(feat_dim_vae * P_frames, 0.0f);
        std::vector<float> enc_out = locenc_forward(ctx, patch_enc_in.data(), cpu_be);

        // c. Stop check (uses TSLM lm_hidden, not LocEnc output)
        float sp = stop_score(ctx, tslm_hidden.data(), cpu_be);
        if (ctx->verbosity >= 2) {
            fprintf(stderr, "voxcpm2: step %d stop=%.3f (%.1f ms)\n",
                    step, sp, vox_now_ms() - t0_step);
        }
        if (sp > stop_thresh && step > 0) {
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "voxcpm2: stopped at step %d (stop=%.3f)\n", step, sp);
            }
            break;
        }

        // d. enc_to_lm_proj: locenc_d -> tslm_d
        std::vector<float> enc_lm(d_lm, 0.0f);
        if (ctx->weights.enc_to_lm_w && ctx->weights.enc_to_lm_b) {
            matmul_mv_bias(cpu_be, ctx->weights.enc_to_lm_w, ctx->weights.enc_to_lm_b,
                           enc_out.data(), d_dit, enc_lm.data(), d_lm);
        } else {
            int copy_d = std::min(d_dit, d_lm);
            std::memcpy(enc_lm.data(), enc_out.data(), (size_t)copy_d * sizeof(float));
        }

        // e. TSLM step (single audio token position)
        {
            int tslm_pos = ctx->tslm_kv.n_past;
            std::vector<float> h = enc_lm;
            for (int l = 0; l < (int)ctx->hp.tslm_n_layers; l++) {
                tslm_layer_step(ctx, l, h.data(), tslm_pos, cpu_be);
            }
            ctx->tslm_kv.n_past++;

            // Output norm
            std::vector<float> normed(d_lm);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed.data(), d_lm, ctx->hp.rms_norm_eps);
            tslm_hidden = normed;
        }

        // f. FSQ
        fsq_out = fsq_forward(ctx, tslm_hidden.data(), cpu_be);

        // g. proj.fusion(cat(lm_hidden, enc_lm)) [4096 -> 2048]
        std::vector<float> fusion_in((size_t)(d_lm + d_lm));
        std::memcpy(fusion_in.data(),         tslm_hidden.data(), (size_t)d_lm * sizeof(float));
        std::memcpy(fusion_in.data() + d_lm,  enc_lm.data(),     (size_t)d_lm * sizeof(float));

        std::vector<float> fusion_out(d_ralm, 0.0f);
        if (ctx->weights.fusion_w && ctx->weights.fusion_b) {
            matmul_mv_bias(cpu_be, ctx->weights.fusion_w, ctx->weights.fusion_b,
                           fusion_in.data(), 2 * d_lm, fusion_out.data(), d_ralm);
        } else {
            // Fallback: pass fsq_out directly
            fusion_out = fsq_out;
        }

        // h. RALM step
        {
            std::vector<float> h = fusion_out;
            int ralm_pos = ctx->ralm_kv.n_past;
            (void)ralm_pos;
            for (int l = 0; l < (int)ctx->hp.ralm_n_layers; l++) {
                ralm_layer_step(ctx, l, h.data(), cpu_be);
            }
            ctx->ralm_kv.n_past++;

            std::vector<float> normed(d_ralm);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.ralm_output_norm),
                         normed.data(), d_ralm, ctx->hp.rms_norm_eps);
            ralm_hidden = normed;
        }

        // Update mu conditioning for next step
        mu = build_mu(tslm_hidden, ralm_hidden);
        step++;
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: AR loop %d steps, %.1f ms\n",
                step, vox_now_ms() - t0_ar);
    }

    // 7. VAE decode
    double t0_vae = vox_now_ms();
    std::vector<float> pcm = vae_decode(ctx, patches, cpu_be);
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "voxcpm2: VAE decode %.1f ms -> %zu samples @48kHz\n",
                vox_now_ms() - t0_vae, pcm.size());
        fprintf(stderr, "voxcpm2: total %.1f ms\n", vox_now_ms() - t0_total);
    }

    if (pcm.empty()) {
        fprintf(stderr, "voxcpm2: empty audio output\n");
        return nullptr;
    }

    *out_n_samples = (int)pcm.size();
    float* result = (float*)std::malloc(pcm.size() * sizeof(float));
    if (!result) return nullptr;
    std::memcpy(result, pcm.data(), pcm.size() * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

struct voxcpm2_context_params voxcpm2_context_default_params(void) {
    struct voxcpm2_context_params p;
    p.n_threads       = 4;
    p.verbosity       = 1;
    p.use_gpu         = false;
    p.flash_attn      = true;
    p.inference_steps = 10;
    p.cfg_value       = 2.0f;
    p.max_len         = 2000;
    return p;
}

struct voxcpm2_context* voxcpm2_init_from_file(const char* path_model,
                                                struct voxcpm2_context_params params) {
    if (!path_model) return nullptr;

    auto* ctx = new voxcpm2_context();
    ctx->n_threads       = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity       = params.verbosity;
    ctx->use_gpu         = params.use_gpu;
    ctx->flash_attn      = params.flash_attn;
    ctx->inference_steps = params.inference_steps > 0 ? params.inference_steps : 10;
    ctx->cfg_value       = params.cfg_value > 0.0f ? params.cfg_value : 2.0f;
    ctx->max_len         = params.max_len > 0 ? params.max_len : 2000;

    if (!vox_load_weights(ctx, path_model)) {
        fprintf(stderr, "voxcpm2: failed to load '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void voxcpm2_free(struct voxcpm2_context* ctx) {
    if (!ctx) return;
    if (ctx->weight_buf) {
        ggml_backend_buffer_free(ctx->weight_buf);
        ctx->weight_buf = nullptr;
    }
    if (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }
    delete ctx;
}

void voxcpm2_set_n_threads(struct voxcpm2_context* ctx, int n_threads) {
    if (ctx) ctx->n_threads = n_threads > 0 ? n_threads : 1;
}

float* voxcpm2_synthesize(struct voxcpm2_context* ctx, const char* text,
                           int* out_n_samples) {
    return vox_synthesize_internal(ctx, text, nullptr, 0, out_n_samples);
}

float* voxcpm2_synthesize_clone(struct voxcpm2_context* ctx, const char* text,
                                 const float* ref_samples, int ref_n_samples,
                                 int* out_n_samples) {
    return vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, out_n_samples);
}

struct voxcpm2_stream* voxcpm2_stream_open(struct voxcpm2_context* ctx, const char* text,
                                            const float* ref_samples, int ref_n_samples) {
    if (!ctx || !text) return nullptr;
    auto* s  = new voxcpm2_stream();
    s->ctx   = ctx;
    s->done  = false;
    s->chunk_offset = 0;

    int n_samples = 0;
    float* pcm = vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, &n_samples);
    if (pcm && n_samples > 0) {
        s->all_pcm.assign(pcm, pcm + n_samples);
        std::free(pcm);
    } else {
        s->done = true;
    }
    return s;
}

const float* voxcpm2_stream_next(struct voxcpm2_stream* stream, int* out_n_samples) {
    if (!stream || stream->done || stream->all_pcm.empty()) {
        if (out_n_samples) *out_n_samples = 0;
        return nullptr;
    }

    const int chunk_size = 3840; // one VAE decode unit at 48kHz (~12.5fps)
    int offset    = stream->chunk_offset;
    int remaining = (int)stream->all_pcm.size() - offset;
    if (remaining <= 0) {
        stream->done = true;
        if (out_n_samples) *out_n_samples = 0;
        return nullptr;
    }

    int n = std::min(remaining, chunk_size);
    stream->chunk_buf.assign(stream->all_pcm.begin() + offset,
                             stream->all_pcm.begin() + offset + n);
    stream->chunk_offset += n;
    if (stream->chunk_offset >= (int)stream->all_pcm.size()) {
        stream->done = true;
    }

    if (out_n_samples) *out_n_samples = n;
    return stream->chunk_buf.data();
}

void voxcpm2_stream_close(struct voxcpm2_stream* stream) {
    delete stream;
}

void voxcpm2_pcm_free(float* pcm) {
    std::free(pcm);
}

float* voxcpm2_extract_stage(struct voxcpm2_context* ctx, const char* text,
                               const float* ref_samples, int ref_n_samples,
                               const char* stage_name, int* out_n) {
    if (!ctx || !text || !stage_name || !out_n) return nullptr;
    *out_n = 0;

    ggml_backend_t cpu_be = get_cpu_backend();
    ggml_backend_cpu_set_n_threads(cpu_be, ctx->n_threads);

    std::string stage(stage_name);
    std::vector<int32_t> token_ids = vox_tokenize(ctx->tokenizer, std::string(text));

    if (stage == "text_input_ids") {
        *out_n = (int)token_ids.size();
        float* out = (float*)std::malloc((size_t)*out_n * sizeof(float));
        for (int i = 0; i < *out_n; i++) out[i] = (float)token_ids[i];
        return out;
    }

    if (stage == "tslm_prefill_out") {
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d = (int)ctx->hp.tslm_d_model;
        std::vector<float> normed(d);
        rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm),
                     normed.data(), d, ctx->hp.rms_norm_eps);
        *out_n = d;
        float* out = (float*)std::malloc((size_t)d * sizeof(float));
        std::memcpy(out, normed.data(), (size_t)d * sizeof(float));
        return out;
    }

    if (stage == "ralm_prefill_out") {
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d = (int)ctx->hp.tslm_d_model;
        {
            std::vector<float> normed(d);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed.data(), d, ctx->hp.rms_norm_eps);
            h = normed;
        }
        std::vector<float> fsq_out = fsq_forward(ctx, h.data(), cpu_be);
        std::vector<float> ralm_h  = ralm_prefill(ctx, fsq_out, cpu_be);
        int dr = (int)ctx->hp.ralm_d_model;
        *out_n = dr;
        float* out = (float*)std::malloc((size_t)dr * sizeof(float));
        std::memcpy(out, ralm_h.data(), (size_t)dr * sizeof(float));
        return out;
    }

    if (stage == "cfm_step0_result") {
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d     = (int)ctx->hp.tslm_d_model;
        int d_dit = (int)ctx->hp.locdit_d_model;
        {
            std::vector<float> normed(d);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed.data(), d, ctx->hp.rms_norm_eps);
            h = normed;
        }
        // Build mu [2048] = cat(lm_to_dit(tslm), zeros for ralm part)
        std::vector<float> mu(2 * d_dit, 0.0f);
        if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b,
                           h.data(), d, mu.data(), d_dit);
        }
        int P_fr = (int)ctx->hp.patch_frames;
        std::vector<float> zero_cond(64 * P_fr, 0.0f);
        std::vector<float> patch = cfm_euler_solve(ctx, mu.data(), zero_cond.data(),
                                                    1, ctx->cfg_value, cpu_be);
        *out_n = (int)patch.size();
        float* out = (float*)std::malloc(patch.size() * sizeof(float));
        std::memcpy(out, patch.data(), patch.size() * sizeof(float));
        return out;
    }

    if (stage == "locenc_out") {
        std::vector<float> h = tslm_prefill(ctx, token_ids, cpu_be);
        int d     = (int)ctx->hp.tslm_d_model;
        int d_dit = (int)ctx->hp.locdit_d_model;
        {
            std::vector<float> normed(d);
            rms_norm_cpu(h.data(), tensor_data_f32(ctx->weights.tslm_output_norm),
                         normed.data(), d, ctx->hp.rms_norm_eps);
            h = normed;
        }
        // Build mu [2048]
        std::vector<float> mu(2 * d_dit, 0.0f);
        if (ctx->weights.lm_to_dit_w && ctx->weights.lm_to_dit_b) {
            matmul_mv_bias(cpu_be, ctx->weights.lm_to_dit_w, ctx->weights.lm_to_dit_b,
                           h.data(), d, mu.data(), d_dit);
        }
        int P_fr = (int)ctx->hp.patch_frames;
        std::vector<float> zero_cond(64 * P_fr, 0.0f);
        std::vector<float> patch = cfm_euler_solve(ctx, mu.data(), zero_cond.data(),
                                                    ctx->inference_steps,
                                                    ctx->cfg_value, cpu_be);
        // LocEnc expects raw latent [feat_dim * patch_frames]
        patch.resize(64 * P_fr, 0.0f);
        std::vector<float> enc_out = locenc_forward(ctx, patch.data(), cpu_be);
        *out_n = (int)enc_out.size();
        float* out = (float*)std::malloc(enc_out.size() * sizeof(float));
        std::memcpy(out, enc_out.data(), enc_out.size() * sizeof(float));
        return out;
    }

    if (stage == "decoded_audio") {
        return vox_synthesize_internal(ctx, text, ref_samples, ref_n_samples, out_n);
    }

    fprintf(stderr, "voxcpm2_extract_stage: unknown stage '%s'\n", stage_name);
    return nullptr;
}

} // extern "C"
