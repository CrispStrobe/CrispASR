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

#define CT_CHECK(x) do { if (!(x)) { fprintf(stderr, "CT_CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); abort(); } } while(0)

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float swish(float x)   { return x * sigmoid(x); }

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

    // KV cache for decoder self-attention (pre-allocated)
    // Shape per layer: (n_heads, max_ctx, head_dim) × 2 (K and V)
    std::vector<std::vector<float>> kv_cache_k; // [n_dec_layers][n_heads * max_ctx * head_dim]
    std::vector<std::vector<float>> kv_cache_v;

    // Cross-attention KV cache: computed once per utterance from encoder output.
    // Shape per layer: (T_enc, dec_d_model), row-major.
    std::vector<std::vector<float>> cross_kv_k; // [n_dec_layers][T_enc * dec_d_model]
    std::vector<std::vector<float>> cross_kv_v;

    // ggml backend for compute graph execution (CPU; GPU backends slot in here later)
    ggml_backend_t      ggml_backend = nullptr;
    ggml_gallocr_t      ggml_alloc   = nullptr;

    // Metadata context for graph node descriptors (no_alloc=true; actual buffers via gallocr)
    // Sized generously; only holds ggml_tensor_overhead() * N_NODES bytes.
    std::vector<uint8_t> compute_meta;

    // Cached T_enc from last encode call, needed by decode graph builder.
    int cached_T_enc = 0;
};

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
                               const char * path) {
    struct gguf_init_params gp = { .no_alloc = false, .ctx = nullptr };
    // First pass: read metadata
    struct gguf_context * gguf_ctx = gguf_init_from_file(path, { .no_alloc = true, .ctx = nullptr });
    if (!gguf_ctx) {
        fprintf(stderr, "cohere: failed to open '%s'\n", path);
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
    hp.dec_n_heads  = kv_i(CT_KEY_DEC_N_HEADS);
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
            vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                vocab.id_to_token[i] = gguf_get_arr_str(gguf_ctx, ki, i);
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }
    }

    gguf_free(gguf_ctx);

    // Second pass: load all tensor data (no_alloc=false allocates weight buffers)
    {
        struct ggml_context * weight_ctx = nullptr;
        struct gguf_init_params load_params = { .no_alloc = false, .ctx = &weight_ctx };
        gguf_ctx = gguf_init_from_file(path, load_params);
        if (!gguf_ctx || !weight_ctx) {
            fprintf(stderr, "cohere: failed to load tensors from '%s'\n", path);
            return false;
        }
        model.ctx = weight_ctx;
        for (ggml_tensor * t = ggml_get_first_tensor(weight_ctx); t;
             t = ggml_get_next_tensor(weight_ctx, t)) {
            model.tensors[ggml_get_name(t)] = t;
        }
        gguf_free(gguf_ctx);
    }

    fprintf(stderr, "cohere: loaded %d tensors from '%s'\n", (int)model.tensors.size(), path);

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
        const uint16_t * src = (const uint16_t *)t->data;
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
            mel[m * T + t] = logf(v + log_grd);
        }
    }

    // Per-feature normalization: biased std (matches ONNX: std = sqrt(mean(diff²)))
    for (int m = 0; m < n_mels; m++) {
        float * row = mel.data() + m * T;
        double mean = 0.0, var = 0.0;
        for (int t = 0; t < T; t++) mean += row[t];
        mean /= T;
        for (int t = 0; t < T; t++) { double d = row[t] - mean; var += d*d; }
        float std = sqrtf((float)(var / T + 1e-5));
        for (int t = 0; t < T; t++) row[t] = (row[t] - (float)mean) / std;
    }

    return mel; // shape: [n_mels, T], n_mels-major
}

// ---------------------------------------------------------------------------
// Conv2D forward (naive, float32)
// x:    [in_ch,  H,  W]
// w:    [out_ch, in_ch/groups, kH, kW]   (PyTorch weight layout)
// b:    [out_ch]
// out:  [out_ch, H', W']
// ---------------------------------------------------------------------------

static std::vector<float> ct_conv2d(const float * x, int in_ch, int H, int W,
                                    const float * w, int out_ch, int kH, int kW,
                                    int stride, int pad, int groups,
                                    const float * b = nullptr) {
    int H_out = (H + 2*pad - kH) / stride + 1;
    int W_out = (W + 2*pad - kW) / stride + 1;
    int in_per_group  = in_ch  / groups;
    int out_per_group = out_ch / groups;
    std::vector<float> out(out_ch * H_out * W_out, 0.0f);

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < out_per_group; oc++) {
            int oc_g = g * out_per_group + oc;
            float bias = b ? b[oc_g] : 0.0f;
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float v = bias;
                    for (int ic = 0; ic < in_per_group; ic++) {
                        int ic_g = g * in_per_group + ic;
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float xi = x[ic_g * H * W + ih * W + iw];
                                    // w: [out_ch, in_per_group, kH, kW]
                                    float wi = w[(oc_g * in_per_group + ic) * kH * kW + kh * kW + kw];
                                    v += xi * wi;
                                }
                            }
                        }
                    }
                    out[(oc_g * H_out + oh) * W_out + ow] = v;
                }
            }
        }
    }
    return out;
}

// Depthwise conv1d: x [C, T], w [C, 1, k], b [C], pad_same
static std::vector<float> ct_dw_conv1d(const float * x, int C, int T,
                                        const float * w, int k, const float * b = nullptr) {
    int pad = (k - 1) / 2;
    std::vector<float> out(C * T, 0.0f);
    for (int c = 0; c < C; c++) {
        float bias = b ? b[c] : 0.0f;
        for (int t = 0; t < T; t++) {
            float v = bias;
            for (int ki = 0; ki < k; ki++) {
                int ti = t + ki - pad;
                if (ti >= 0 && ti < T) v += x[c * T + ti] * w[c * k + ki];
            }
            out[c * T + t] = v;
        }
    }
    return out;
}

// Pointwise conv2d (1×1) acting as linear on channels: x [C_in, N], w [C_out, C_in, 1, 1]
static std::vector<float> ct_pw_conv1x1(const float * x, int C_in, int N,
                                         const float * w, int C_out,
                                         const float * b = nullptr) {
    std::vector<float> out(C_out * N, 0.0f);
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            float v = b ? b[co] : 0.0f;
            for (int ci = 0; ci < C_in; ci++) v += w[co * C_in + ci] * x[ci * N + n];
            out[co * N + n] = v;
        }
    }
    return out;
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

// Relative shift: converts (H, T, 2T-1) → (H, T, T)
// For query i, key j: result[h, i, j] = input[h, i, i - j + T - 1]
static std::vector<float> ct_rel_shift(const float * bd, int H, int T) {
    int n2 = 2 * T - 1;
    std::vector<float> out(H * T * T, 0.0f);
    for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) {
                int rel = j - i + T - 1;  // BD[tq=i, tk=j] uses pos enc at tk-tq+T-1
                out[(h * T + i) * T + j] = bd[(h * T + i) * n2 + rel];
            }
    return out;
}

// ---------------------------------------------------------------------------
// Conformer self-attention with relative positional encoding
// x:     (T, d)  — input (row-major T × d)
// out:   (T, d)  — output
// ---------------------------------------------------------------------------

static std::vector<float> ct_rel_pos_mha(
    const float * x, int T, int d,
    int H, int head_dim,
    const float * q_w, const float * q_b,
    const float * k_w, const float * k_b,
    const float * v_w, const float * v_b,
    const float * out_w, const float * out_b,
    const float * pos_w,      // [d, d]
    const float * pos_bias_u, // [H, head_dim]
    const float * pos_bias_v  // [H, head_dim]
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Q, K, V: (T, d) — unscaled; scale applied to (AC+BD) below, matching Python
    auto Q = ct_linear(x, d, T, q_w, d, q_b);
    auto K = ct_linear(x, d, T, k_w, d, k_b);
    auto V = ct_linear(x, d, T, v_w, d, v_b);

    // Relative position encodings and projection
    auto pos_enc = ct_rel_pos_enc(T, d);          // (2T-1, d)
    auto R = ct_linear(pos_enc.data(), d, 2*T-1, pos_w, d); // (2T-1, d)

    // Build Q_u = Q + pos_bias_u and Q_v = Q + pos_bias_v (broadcast bias over T).
    // pos_bias_u/v are (H, head_dim) = (d,), one bias per head dimension.
    std::vector<float> Q_u(T * d), Q_v(T * d);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d; j++) {
            Q_u[t*d + j] = Q[t*d + j] + pos_bias_u[j];
            Q_v[t*d + j] = Q[t*d + j] + pos_bias_v[j];
        }

    // AC[h, T, T] = Q_u[:, h*hd:(h+1)*hd] @ K[:, h*hd:(h+1)*hd]^T
    // BD_raw[h, T, 2T-1] = Q_v[:, h*hd:] @ R[:, h*hd:]^T
    // Use non-contiguous BLAS: slice start = ptr + h*head_dim, leading dim = d.
    const int n2 = 2*T - 1;
    std::vector<float> AC(H * T * T);
    std::vector<float> BD_raw(H * T * n2);
    for (int h = 0; h < H; h++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, head_dim, 1.0f,
                    Q_u.data() + h*head_dim, d,
                    K.data()   + h*head_dim, d,
                    0.0f, AC.data() + h*T*T, T);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, n2, head_dim, 1.0f,
                    Q_v.data() + h*head_dim, d,
                    R.data()   + h*head_dim, d,
                    0.0f, BD_raw.data() + h*T*n2, n2);
    }

    // Relative shift BD_raw → BD
    auto BD = ct_rel_shift(BD_raw.data(), H, T);

    // scores = (AC + BD) * scale, softmax per (head, query)
    std::vector<float> scores(H * T * T);
    for (int i = 0; i < H * T * T; i++) scores[i] = (AC[i] + BD[i]) * scale;
    for (int h = 0; h < H; h++) {
        for (int tq = 0; tq < T; tq++) {
            float * row = scores.data() + (h * T + tq) * T;
            float mx = *std::max_element(row, row + T);
            float sum = 0.0f;
            for (int tk = 0; tk < T; tk++) { row[tk] = expf(row[tk] - mx); sum += row[tk]; }
            for (int tk = 0; tk < T; tk++) row[tk] /= sum;
        }
    }

    // ctx_merged[T, d] = scores[H,T,T] @ V[T,d], written per-head with stride d.
    std::vector<float> ctx_merged(T * d, 0.0f);
    for (int h = 0; h < H; h++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, head_dim, T, 1.0f,
                    scores.data() + h*T*T, T,
                    V.data()      + h*head_dim, d,
                    0.0f,
                    ctx_merged.data() + h*head_dim, d);
    }

    // Output projection
    return ct_linear(ctx_merged.data(), d, T, out_w, d, out_b);
}

// ---------------------------------------------------------------------------
// Conformer convolution module
// x: (T, d), returns (T, d)
// ---------------------------------------------------------------------------

static std::vector<float> ct_conformer_conv(
    const float * x, int T, int d, int k,
    const float * pw1_w, const float * pw1_b,
    const float * dw_w,  const float * dw_b,
    const float * bn_w,  const float * bn_b,
    const float * bn_mean, const float * bn_var,
    const float * pw2_w, const float * pw2_b,
    float bn_eps = 1e-5f
) {
    // pw1: (T, d) → (T, 2d) then GLU → (T, d)
    auto h = ct_linear(x, d, T, pw1_w, 2*d, pw1_b);
    // GLU: h[:, :d] * sigmoid(h[:, d:])
    std::vector<float> glu(T * d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            glu[t * d + i] = h[t * 2*d + i] * sigmoid(h[t * 2*d + d + i]);

    // Reshape to (d, T) for depthwise conv1d
    std::vector<float> transposed(d * T);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            transposed[i * T + t] = glu[t * d + i];

    // Depthwise conv1d: (d, T) → (d, T)  with kernel size k, padding (k-1)/2
    auto dw = ct_dw_conv1d(transposed.data(), d, T, dw_w, k, dw_b);

    // Batch norm (inference mode): y = (x - mean) / sqrt(var + eps) * w + b
    // Stored as separate mean/var/scale/bias. Applied per channel (dim=0 here).
    for (int c = 0; c < d; c++) {
        float inv = bn_w[c] / sqrtf(bn_var[c] + bn_eps);
        float bias = bn_b[c] - bn_mean[c] * inv;
        for (int t = 0; t < T; t++) dw[c * T + t] = dw[c * T + t] * inv + bias;
    }

    // Swish activation
    ct_swish_inplace(dw.data(), d * T);

    // pw2: (d, T) → linear per time-step, returns (d, T)
    auto out_dT = ct_pw_conv1x1(dw.data(), d, T, pw2_w, d, pw2_b);

    // Transpose back to (T, d)
    std::vector<float> out(T * d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            out[t * d + i] = out_dT[i * T + t];
    return out;
}

// ---------------------------------------------------------------------------
// Scratch buffers for encoder layers — pre-allocated once, reused every layer.
// ---------------------------------------------------------------------------

struct EncScratch {
    // layer norm output (reused)
    std::vector<float> x_norm;    // (T, d)
    // FFN
    std::vector<float> ff_h;      // (T, ffn_dim)
    std::vector<float> ff_out;    // (T, d)
    // Attention
    std::vector<float> Q, K, V;   // (T, d) each
    std::vector<float> R;         // (2T-1, d)
    std::vector<float> Q_u, Q_v;  // (T, d) each
    std::vector<float> AC;        // (H, T, T)
    std::vector<float> BD_raw;    // (H, T, 2T-1)
    std::vector<float> BD;        // (H, T, T)
    std::vector<float> scores;    // (H, T, T)
    std::vector<float> ctx_merged;// (T, d)
    std::vector<float> attn_out;  // (T, d)
    std::vector<float> pos_enc;   // (2T-1, d) — precomputed once per encode call
    // Conv module
    std::vector<float> cv_h;      // (T, 2d) — pw1 output
    std::vector<float> cv_glu;    // (T, d)
    std::vector<float> cv_trans;  // (d, T)
    std::vector<float> cv_dw;     // (d, T)
    std::vector<float> cv_out_dT; // (d, T)
    std::vector<float> cv_out;    // (T, d)
};

// ---------------------------------------------------------------------------
// Feed-forward sub-layer (Macaron style, called with scale=0.5)
// x: (T, d), returns (T, d)
// ---------------------------------------------------------------------------

static std::vector<float> ct_ffn(
    const float * x, int T, int d, int ffn_dim, float scale,
    const float * up_w, const float * up_b,
    const float * dn_w, const float * dn_b
) {
    auto h = ct_linear(x, d, T, up_w, ffn_dim, up_b);
    ct_swish_inplace(h.data(), ffn_dim * T);
    auto out = ct_linear(h.data(), ffn_dim, T, dn_w, d, dn_b);
    // scale + residual not applied here (caller does residual)
    for (auto & v : out) v *= scale;
    return out;
}

// Scratch variant: writes ff_h and ff_out into EncScratch.
// Weight tensors converted on-the-fly via ct_tensor_f32 (no 7.8 GB F32 cache).
static void ct_ffn_scratch(
    EncScratch & sc,
    const float * x, int T, int d, int ffn_dim, float scale,
    const ggml_tensor * up_w_t, const float * up_b,
    const ggml_tensor * dn_w_t, const float * dn_b
) {
    ct_linear_into(sc.ff_h, x, d, T, ct_tensor_f32(up_w_t), ffn_dim, up_b);
    ct_swish_inplace(sc.ff_h.data(), ffn_dim * T);
    ct_linear_into(sc.ff_out, sc.ff_h.data(), ffn_dim, T, ct_tensor_f32(dn_w_t), d, dn_b);
    for (auto & v : sc.ff_out) v *= scale;
}

// Scratch variant of ct_conformer_conv: reuses cv_* buffers from EncScratch.
// pw1_w and pw2_w are ggml_tensor* (converted on-the-fly); dw and bn params are float*.
static const std::vector<float> & ct_conformer_conv_scratch(
    EncScratch & sc,
    const float * x, int T, int d, int k,
    const ggml_tensor * pw1_w_t, const float * pw1_b,
    const float * dw_w,  const float * dw_b,
    const float * bn_w,  const float * bn_b,
    const float * bn_mean, const float * bn_var,
    const ggml_tensor * pw2_w_t, const float * pw2_b,
    float bn_eps = 1e-5f
) {
    // pw1: (T, d) → (T, 2d)
    ct_linear_into(sc.cv_h, x, d, T, ct_tensor_f32(pw1_w_t), 2*d, pw1_b);
    // GLU: → (T, d) into sc.cv_glu
    sc.cv_glu.resize((size_t)T * d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            sc.cv_glu[t * d + i] = sc.cv_h[t * 2*d + i] * sigmoid(sc.cv_h[t * 2*d + d + i]);
    // Transpose (T,d) → (d,T)
    sc.cv_trans.resize((size_t)d * T);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            sc.cv_trans[i * T + t] = sc.cv_glu[t * d + i];
    // Depthwise conv1d (d,T) → (d,T): inline from ct_dw_conv1d
    int pad = (k - 1) / 2;
    sc.cv_dw.assign((size_t)d * T, 0.0f);
    for (int c = 0; c < d; c++) {
        float bias = dw_b ? dw_b[c] : 0.0f;
        for (int t = 0; t < T; t++) {
            float v = bias;
            for (int ki = 0; ki < k; ki++) {
                int ti = t + ki - pad;
                if (ti >= 0 && ti < T) v += sc.cv_trans[c * T + ti] * dw_w[c * k + ki];
            }
            sc.cv_dw[c * T + t] = v;
        }
    }
    // Batch norm
    for (int c = 0; c < d; c++) {
        float inv = bn_w[c] / sqrtf(bn_var[c] + bn_eps);
        float bias = bn_b[c] - bn_mean[c] * inv;
        for (int t = 0; t < T; t++) sc.cv_dw[c * T + t] = sc.cv_dw[c * T + t] * inv + bias;
    }
    // Swish
    ct_swish_inplace(sc.cv_dw.data(), d * T);
    // pw2: (d,T) → (d,T) via BLAS: out = pw2_w @ cv_dw, layout (d,T)
    // pw2_w is [d, d, 1, 1], cv_dw is [d, T] row-major → sgemm(d, T, d)
    {
        const float * pw2_w = ct_tensor_f32(pw2_w_t);
        sc.cv_out_dT.resize((size_t)d * T);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    d, T, d, 1.0f,
                    pw2_w, d, sc.cv_dw.data(), T,
                    0.0f, sc.cv_out_dT.data(), T);
        if (pw2_b) {
            for (int co = 0; co < d; co++)
                for (int t = 0; t < T; t++)
                    sc.cv_out_dT[co * T + t] += pw2_b[co];
        }
    }
    // Transpose (d,T) → (T,d) into sc.cv_out
    sc.cv_out.resize((size_t)T * d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            sc.cv_out[t * d + i] = sc.cv_out_dT[i * T + t];
    return sc.cv_out;
}

// Scratch variant of ct_rel_pos_mha: reuses Q/K/V/R/Q_u/Q_v/AC/BD_raw/BD/scores/ctx_merged/attn_out.
// Weight matrices converted on-the-fly via ct_tensor_f32 (no 7.8 GB F32 cache).
// Uses sc.pos_enc (must be pre-filled by caller via ct_rel_pos_enc).
static const std::vector<float> & ct_rel_pos_mha_scratch(
    EncScratch & sc,
    const float * x, int T, int d,
    int H, int head_dim,
    const ggml_tensor * q_w_t, const float * q_b,
    const ggml_tensor * k_w_t, const float * k_b,
    const ggml_tensor * v_w_t, const float * v_b,
    const ggml_tensor * out_w_t, const float * out_b,
    const ggml_tensor * pos_w_t,
    const float * pos_bias_u,
    const float * pos_bias_v
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    const int n2 = 2*T - 1;

    ct_linear_into(sc.Q, x, d, T, ct_tensor_f32(q_w_t), d, q_b);
    ct_linear_into(sc.K, x, d, T, ct_tensor_f32(k_w_t), d, k_b);
    ct_linear_into(sc.V, x, d, T, ct_tensor_f32(v_w_t), d, v_b);

    // R: positional projection (sc.pos_enc already filled by caller)
    ct_linear_into(sc.R, sc.pos_enc.data(), d, n2, ct_tensor_f32(pos_w_t), d);

    // Q_u = Q + pos_bias_u, Q_v = Q + pos_bias_v
    sc.Q_u.resize((size_t)T * d);
    sc.Q_v.resize((size_t)T * d);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d; j++) {
            sc.Q_u[t*d + j] = sc.Q[t*d + j] + pos_bias_u[j];
            sc.Q_v[t*d + j] = sc.Q[t*d + j] + pos_bias_v[j];
        }

    // AC[h, T, T] and BD_raw[h, T, 2T-1]
    sc.AC.resize((size_t)H * T * T);
    sc.BD_raw.resize((size_t)H * T * n2);
    for (int h = 0; h < H; h++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, T, head_dim, 1.0f,
                    sc.Q_u.data() + h*head_dim, d,
                    sc.K.data()   + h*head_dim, d,
                    0.0f, sc.AC.data() + h*T*T, T);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    T, n2, head_dim, 1.0f,
                    sc.Q_v.data() + h*head_dim, d,
                    sc.R.data()   + h*head_dim, d,
                    0.0f, sc.BD_raw.data() + h*T*n2, n2);
    }

    // Relative shift BD_raw → BD
    sc.BD.resize((size_t)H * T * T);
    for (int h = 0; h < H; h++)
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) {
                int rel = j - i + T - 1;
                sc.BD[(h * T + i) * T + j] = sc.BD_raw[(h * T + i) * n2 + rel];
            }

    // scores = (AC + BD) * scale, softmax per (head, query)
    sc.scores.resize((size_t)H * T * T);
    for (int i = 0; i < H * T * T; i++) sc.scores[i] = (sc.AC[i] + sc.BD[i]) * scale;
    for (int h = 0; h < H; h++) {
        for (int tq = 0; tq < T; tq++) {
            float * row = sc.scores.data() + (h * T + tq) * T;
            float mx = *std::max_element(row, row + T);
            float sum = 0.0f;
            for (int tk = 0; tk < T; tk++) { row[tk] = expf(row[tk] - mx); sum += row[tk]; }
            for (int tk = 0; tk < T; tk++) row[tk] /= sum;
        }
    }

    // ctx_merged[T, d] = scores[H,T,T] @ V[T,d], per-head with stride d
    sc.ctx_merged.assign((size_t)T * d, 0.0f);
    for (int h = 0; h < H; h++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, head_dim, T, 1.0f,
                    sc.scores.data() + h*T*T, T,
                    sc.V.data()      + h*head_dim, d,
                    0.0f,
                    sc.ctx_merged.data() + h*head_dim, d);
    }

    // Output projection into sc.attn_out
    ct_linear_into(sc.attn_out, sc.ctx_merged.data(), d, T, ct_tensor_f32(out_w_t), d, out_b);
    return sc.attn_out;
}

// ---------------------------------------------------------------------------
// Full Conformer encoder
// mel: (n_mels, T_mel) row-major
// Returns: (T_enc, dec_d) after subsampling + projection
// ---------------------------------------------------------------------------

static std::vector<float> cohere_encode(const cohere_model & m, const float * mel, int T_mel) {
    const auto & hp = m.hparams;

    // --- Pre-encode: Conv2D subsampling ---
    // Input mel treated as (1, n_mels, T_mel) — single channel 2D signal
    const auto & pre_conv0_w = ct_to_f32_ref(m.pre_conv0_w);
    const auto & pre_conv0_b = ct_to_f32_ref(m.pre_conv0_b);
    const auto & pre_conv2_w = ct_to_f32_ref(m.pre_conv2_w);
    const auto & pre_conv2_b = ct_to_f32_ref(m.pre_conv2_b);
    const auto & pre_conv3_w = ct_to_f32_ref(m.pre_conv3_w);
    const auto & pre_conv3_b = ct_to_f32_ref(m.pre_conv3_b);
    const auto & pre_conv5_w = ct_to_f32_ref(m.pre_conv5_w);
    const auto & pre_conv5_b = ct_to_f32_ref(m.pre_conv5_b);
    const auto & pre_conv6_w = ct_to_f32_ref(m.pre_conv6_w);
    const auto & pre_conv6_b = ct_to_f32_ref(m.pre_conv6_b);
    const auto & pre_out_w = ct_to_f32_ref(m.pre_out_w);
    const auto & pre_out_b = ct_to_f32_ref(m.pre_out_b);

    // Transpose mel (n_mels, T_mel) → (T_mel, n_mels): H=time, W=mel (as in Rust encoder)
    std::vector<float> mel_T(T_mel * hp.n_mels);
    for (int t = 0; t < T_mel; t++)
        for (int m = 0; m < hp.n_mels; m++)
            mel_T[t * hp.n_mels + m] = mel[m * T_mel + t];

    // conv.0: Conv2d(1→256, k=3×3, stride=2, pad=1), groups=1  — H=time, W=mel
    int ch = hp.pre_conv_ch;  // 256
    auto c0 = ct_conv2d(mel_T.data(), 1, T_mel, hp.n_mels, pre_conv0_w.data(), ch, 3, 3, 2, 1, 1, pre_conv0_b.data());
    int H1 = (T_mel      + 2*1 - 3) / 2 + 1;  // ~273
    int W1 = (hp.n_mels  + 2*1 - 3) / 2 + 1;  // 64
    for (int i = 0; i < ch * H1 * W1; i++) c0[i] = c0[i] > 0.0f ? c0[i] : 0.0f;  // ReLU

    // conv.2: depthwise Conv2d(256→256, k=3×3, stride=2, pad=1, groups=256)
    auto c2 = ct_conv2d(c0.data(), ch, H1, W1, pre_conv2_w.data(), ch, 3, 3, 2, 1, ch, pre_conv2_b.data());
    int H2 = (H1 + 2*1 - 3) / 2 + 1;
    int W2 = (W1 + 2*1 - 3) / 2 + 1;

    // conv.3: pointwise 1×1 + ReLU
    auto c3 = ct_conv2d(c2.data(), ch, H2, W2, pre_conv3_w.data(), ch, 1, 1, 1, 0, 1, pre_conv3_b.data());
    for (int i = 0; i < ch * H2 * W2; i++) c3[i] = c3[i] > 0.0f ? c3[i] : 0.0f;  // ReLU

    // conv.5: depthwise Conv2d(256→256, k=3×3, stride=2, pad=1, groups=256)
    auto c5 = ct_conv2d(c3.data(), ch, H2, W2, pre_conv5_w.data(), ch, 3, 3, 2, 1, ch, pre_conv5_b.data());
    int H3 = (H2 + 2*1 - 3) / 2 + 1;
    int W3 = (W2 + 2*1 - 3) / 2 + 1;

    // conv.6: pointwise 1×1 + ReLU
    auto c6 = ct_conv2d(c5.data(), ch, H3, W3, pre_conv6_w.data(), ch, 1, 1, 1, 0, 1, pre_conv6_b.data());
    for (int i = 0; i < ch * H3 * W3; i++) c6[i] = c6[i] > 0.0f ? c6[i] : 0.0f;  // ReLU

    // Flatten (ch, H3, W3) → (H3, ch*W3): T_sub=H3 (time), flat=ch*W3=4096 (features)
    // c6 layout: [c, h, w] = c6[(c*H3 + h)*W3 + w]
    // flat_in layout: [t, feat] = flat_in[t * flat + c*W3 + w], where t=h
    int flat  = ch * W3;  // 256 * 16 = 4096
    int T_sub = H3;       // ~34 (time frames after 3× stride-2)

    // Reshape c6[c, t, w] → flat_in[t, c*W3 + w]
    std::vector<float> flat_in(T_sub * flat);
    for (int t = 0; t < T_sub; t++)
        for (int c = 0; c < ch; c++)
            for (int w = 0; w < W3; w++)
                flat_in[t * flat + c * W3 + w] = c6[(c * H3 + t) * W3 + w];

    // pre_out: (T_sub, 4096) → (T_sub, 1280)
    auto enc_in = ct_linear(flat_in.data(), flat, T_sub, pre_out_w.data(), hp.enc_d_model, pre_out_b.data());
    // enc_in: (T_sub, enc_d_model)  shape: T_sub rows, enc_d_model columns

    int T = T_sub;
    int d = hp.enc_d_model;

    // --- Conformer layers ---
    // Pre-allocate scratch buffers once; reuse across all 48 layers.
    EncScratch sc;
    sc.x_norm.resize((size_t)T * d);
    sc.pos_enc = ct_rel_pos_enc(T, d);  // sinusoidal table is the same every layer

    for (int li = 0; li < hp.enc_n_layers; li++) {
        const auto & l = m.enc_layers[li];

        // FF1: h = x + 0.5 * FF(norm(x))
        // Small norm params cached via ct_to_f32_ref; large weights via ct_tensor_f32 (on-the-fly).
        {
            const auto & nw = ct_to_f32_ref(l.ff1_norm_w);
            const auto & nb = ct_to_f32_ref(l.ff1_norm_b);
            for (int t = 0; t < T; t++)
                ct_layer_norm(sc.x_norm.data() + t*d, enc_in.data() + t*d, d, nw.data(), nb.data());
        }
        ct_ffn_scratch(sc, sc.x_norm.data(), T, d, hp.enc_ffn_dim, 0.5f,
                       l.ff1_up_w, ct_to_f32_ref(l.ff1_up_b).data(),
                       l.ff1_dn_w, ct_to_f32_ref(l.ff1_dn_b).data());
        for (int i = 0; i < T*d; i++) enc_in[i] += sc.ff_out[i];

        // Self-attention
        {
            const auto & nw = ct_to_f32_ref(l.attn_norm_w);
            const auto & nb = ct_to_f32_ref(l.attn_norm_b);
            for (int t = 0; t < T; t++)
                ct_layer_norm(sc.x_norm.data() + t*d, enc_in.data() + t*d, d, nw.data(), nb.data());
        }
        ct_rel_pos_mha_scratch(sc,
            sc.x_norm.data(), T, d, hp.enc_n_heads, hp.enc_head_dim,
            l.attn_q_w,   ct_to_f32_ref(l.attn_q_b).data(),
            l.attn_k_w,   ct_to_f32_ref(l.attn_k_b).data(),
            l.attn_v_w,   ct_to_f32_ref(l.attn_v_b).data(),
            l.attn_out_w, ct_to_f32_ref(l.attn_out_b).data(),
            l.attn_pos_w,
            ct_to_f32_ref(l.attn_pos_bias_u).data(),
            ct_to_f32_ref(l.attn_pos_bias_v).data()
        );
        for (int i = 0; i < T*d; i++) enc_in[i] += sc.attn_out[i];

        // Convolution module
        {
            const auto & nw = ct_to_f32_ref(l.conv_norm_w);
            const auto & nb = ct_to_f32_ref(l.conv_norm_b);
            for (int t = 0; t < T; t++)
                ct_layer_norm(sc.x_norm.data() + t*d, enc_in.data() + t*d, d, nw.data(), nb.data());
        }
        ct_conformer_conv_scratch(sc,
            sc.x_norm.data(), T, d, hp.enc_conv_k,
            l.conv_pw1_w, ct_to_f32_ref(l.conv_pw1_b).data(),
            ct_to_f32_ref(l.conv_dw_w).data(), ct_to_f32_ref(l.conv_dw_b).data(),
            ct_to_f32_ref(l.conv_bn_w).data(), ct_to_f32_ref(l.conv_bn_b).data(),
            ct_to_f32_ref(l.conv_bn_mean).data(), ct_to_f32_ref(l.conv_bn_var).data(),
            l.conv_pw2_w, ct_to_f32_ref(l.conv_pw2_b).data()
        );
        for (int i = 0; i < T*d; i++) enc_in[i] += sc.cv_out[i];

        // FF2
        {
            const auto & nw = ct_to_f32_ref(l.ff2_norm_w);
            const auto & nb = ct_to_f32_ref(l.ff2_norm_b);
            for (int t = 0; t < T; t++)
                ct_layer_norm(sc.x_norm.data() + t*d, enc_in.data() + t*d, d, nw.data(), nb.data());
        }
        ct_ffn_scratch(sc, sc.x_norm.data(), T, d, hp.enc_ffn_dim, 0.5f,
                       l.ff2_up_w, ct_to_f32_ref(l.ff2_up_b).data(),
                       l.ff2_dn_w, ct_to_f32_ref(l.ff2_dn_b).data());
        for (int i = 0; i < T*d; i++) enc_in[i] += sc.ff_out[i];

        // Output norm
        const auto & out_norm_w = ct_to_f32_ref(l.out_norm_w);
        const auto & out_norm_b = ct_to_f32_ref(l.out_norm_b);
        for (int t = 0; t < T; t++)
            ct_layer_norm(enc_in.data() + t*d, enc_in.data() + t*d, d,
                         out_norm_w.data(), out_norm_b.data());

        if ((li + 1) % 8 == 0)
            fprintf(stderr, "cohere: encoder layer %d/%d done\n", li+1, hp.enc_n_layers);
    }

    // Encoder-decoder projection: (T, enc_d) → (T, dec_d)
    const auto & proj_b = ct_to_f32_ref(m.enc_proj_b);
    auto enc_out = ct_linear(enc_in.data(), d, T, ct_tensor_f32(m.enc_proj_w), hp.dec_d_model, proj_b.data());

    return enc_out;  // (T, dec_d)
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
    fprintf(stderr, "cohere: cross KV pre-computed for %d layers, T_enc=%d\n",
            hp.dec_n_layers, T_enc);
}

// ---------------------------------------------------------------------------
// Decoder: one step (auto-regressive)
// enc_out:  (T_enc, dec_d)
// tokens:   [offset .. offset+n_tok-1]
// Returns logits: (n_tok, vocab_size)
// ---------------------------------------------------------------------------

static std::vector<float> cohere_decode_step(
    const cohere_model & m,
    int T_enc,
    const int * tokens, int n_tok, int offset,
    std::vector<std::vector<float>> & kv_k,
    std::vector<std::vector<float>> & kv_v,
    const std::vector<std::vector<float>> & cross_kv_k,
    const std::vector<std::vector<float>> & cross_kv_v
) {
    const auto & hp = m.hparams;
    const int d  = hp.dec_d_model;
    const int H  = hp.dec_n_heads;
    const int hd = hp.dec_head_dim;
    const int ffn = hp.dec_ffn_dim;
    const int max_ctx = hp.dec_max_ctx;

    // Embeddings
    const auto & emb_w = ct_to_f32_ref(m.dec_emb_w);  // [vocab, d]
    const auto & pos_w = ct_to_f32_ref(m.dec_pos_w);  // [max_ctx, d]
    const auto & emb_ln_w = ct_to_f32_ref(m.dec_emb_ln_w);
    const auto & emb_ln_b = ct_to_f32_ref(m.dec_emb_ln_b);

    std::vector<float> h(n_tok * d);
    for (int i = 0; i < n_tok; i++) {
        int tok = tokens[i];
        int pos = offset + i;
        for (int j = 0; j < d; j++)
            h[i*d + j] = emb_w[tok * d + j] + pos_w[pos * d + j];
        ct_layer_norm(h.data() + i*d, h.data() + i*d, d, emb_ln_w.data(), emb_ln_b.data());
    }
    // Decoder layers
    for (int li = 0; li < hp.dec_n_layers; li++) {
        const auto & l = m.dec_layers[li];
        float scale = 1.0f / sqrtf((float)hd);

        const auto & attn_ln_w = ct_to_f32_ref(l.attn_ln_w);
        const auto & attn_ln_b = ct_to_f32_ref(l.attn_ln_b);
        // --- Self-attention with KV cache ---
        std::vector<float> h_norm(n_tok * d);
        for (int i = 0; i < n_tok; i++)
            ct_layer_norm(h_norm.data() + i*d, h.data() + i*d, d, attn_ln_w.data(), attn_ln_b.data());

        const float * q_w_f32 = ct_tensor_f32(l.attn_q_w);
        auto Q = ct_linear(h_norm.data(), d, n_tok, q_w_f32, d, ct_to_f32_ref(l.attn_q_b).data());
        const float * k_w_f32 = ct_tensor_f32(l.attn_k_w);
        auto K = ct_linear(h_norm.data(), d, n_tok, k_w_f32, d, ct_to_f32_ref(l.attn_k_b).data());
        const float * v_w_f32 = ct_tensor_f32(l.attn_v_w);
        auto V = ct_linear(h_norm.data(), d, n_tok, v_w_f32, d, ct_to_f32_ref(l.attn_v_b).data());

        // Write K, V into cache at position [offset, offset+n_tok)
        int cache_size = H * max_ctx * hd;
        auto & ck = kv_k[li];
        auto & cv = kv_v[li];

        for (int i = 0; i < n_tok; i++) {
            int pos = offset + i;
            for (int h_idx = 0; h_idx < H; h_idx++) {
                for (int j = 0; j < hd; j++) {
                    ck[(h_idx * max_ctx + pos) * hd + j] = K[i * d + h_idx * hd + j];
                    cv[(h_idx * max_ctx + pos) * hd + j] = V[i * d + h_idx * hd + j];
                }
            }
        }

        int total_kv = offset + n_tok;

        // Compute attention output
        std::vector<float> sa_out(n_tok * d, 0.0f);
        for (int h_idx = 0; h_idx < H; h_idx++) {
            for (int tq = 0; tq < n_tok; tq++) {
                // Scores (causal: key positions ≤ offset + tq)
                int causal_end = offset + tq + 1;
                std::vector<float> scores(causal_end);
                for (int tk = 0; tk < causal_end; tk++) {
                    float s = 0.0f;
                    for (int j = 0; j < hd; j++)
                        s += Q[tq * d + h_idx * hd + j] * ck[(h_idx * max_ctx + tk) * hd + j];
                    scores[tk] = s * scale;
                }
                float mx = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (auto & s : scores) { s = expf(s - mx); sum += s; }
                for (auto & s : scores) s /= sum;

                for (int j = 0; j < hd; j++) {
                    float v = 0.0f;
                    for (int tk = 0; tk < causal_end; tk++)
                        v += scores[tk] * cv[(h_idx * max_ctx + tk) * hd + j];
                    sa_out[tq * d + h_idx * hd + j] += v;
                }
            }
        }
        // Out projection + residual
        const float * o_w_f32 = ct_tensor_f32(l.attn_o_w);
        auto sa_proj = ct_linear(sa_out.data(), d, n_tok, o_w_f32, d, ct_to_f32_ref(l.attn_o_b).data());
        for (int i = 0; i < n_tok * d; i++) h[i] += sa_proj[i];

        // --- Cross-attention ---
        // CK and CV are pre-computed once per utterance in cross_kv_k/v.
        const auto & cross_ln_w = ct_to_f32_ref(l.cross_ln_w);  const auto & cross_ln_b = ct_to_f32_ref(l.cross_ln_b);
        std::vector<float> h_cross_norm(n_tok * d);
        for (int i = 0; i < n_tok; i++)
            ct_layer_norm(h_cross_norm.data() + i*d, h.data() + i*d, d, cross_ln_w.data(), cross_ln_b.data());

        const float * cq_w_f32 = ct_tensor_f32(l.cross_q_w);
        auto CQ = ct_linear(h_cross_norm.data(), d, n_tok, cq_w_f32, d, ct_to_f32_ref(l.cross_q_b).data());
        const float * CK = cross_kv_k[li].data();
        const float * CV = cross_kv_v[li].data();

        std::vector<float> ca_out(n_tok * d, 0.0f);
        for (int h_idx = 0; h_idx < H; h_idx++) {
            for (int tq = 0; tq < n_tok; tq++) {
                std::vector<float> scores(T_enc);
                for (int te = 0; te < T_enc; te++) {
                    float s = 0.0f;
                    for (int j = 0; j < hd; j++)
                        s += CQ[tq * d + h_idx * hd + j] * CK[te * d + h_idx * hd + j];
                    scores[te] = s * scale;
                }
                float mx = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (auto & s : scores) { s = expf(s - mx); sum += s; }
                for (auto & s : scores) s /= sum;
                for (int j = 0; j < hd; j++) {
                    float v = 0.0f;
                    for (int te = 0; te < T_enc; te++)
                        v += scores[te] * CV[te * d + h_idx * hd + j];
                    ca_out[tq * d + h_idx * hd + j] += v;
                }
            }
        }
        const float * co_w_f32 = ct_tensor_f32(l.cross_o_w);
        auto ca_proj = ct_linear(ca_out.data(), d, n_tok, co_w_f32, d, ct_to_f32_ref(l.cross_o_b).data());
        for (int i = 0; i < n_tok * d; i++) h[i] += ca_proj[i];

        // --- FFN ---
        const auto & ffn_ln_w = ct_to_f32_ref(l.ffn_ln_w);  const auto & ffn_ln_b = ct_to_f32_ref(l.ffn_ln_b);
        std::vector<float> h_ffn_norm(n_tok * d);
        for (int i = 0; i < n_tok; i++)
            ct_layer_norm(h_ffn_norm.data() + i*d, h.data() + i*d, d, ffn_ln_w.data(), ffn_ln_b.data());
        const float * ffn_up_f32 = ct_tensor_f32(l.ffn_up_w);
        auto ffn_h = ct_linear(h_ffn_norm.data(), d, n_tok, ffn_up_f32, ffn, ct_to_f32_ref(l.ffn_up_b).data());
        for (int i = 0; i < n_tok * ffn; i++) ffn_h[i] = ffn_h[i] > 0.0f ? ffn_h[i] : 0.0f;  // ReLU
        const float * ffn_dn_f32 = ct_tensor_f32(l.ffn_dn_w);
        auto ffn_out = ct_linear(ffn_h.data(), ffn, n_tok, ffn_dn_f32, d, ct_to_f32_ref(l.ffn_dn_b).data());
        for (int i = 0; i < n_tok * d; i++) h[i] += ffn_out[i];
    }

    // Final layer norm + head
    const auto & out_ln_w = ct_to_f32_ref(m.dec_out_ln_w);
    const auto & out_ln_b = ct_to_f32_ref(m.dec_out_ln_b);
    for (int i = 0; i < n_tok; i++)
        ct_layer_norm(h.data() + i*d, h.data() + i*d, d, out_ln_w.data(), out_ln_b.data());

    const float * head_w_f32 = ct_tensor_f32(m.dec_head_w);
    auto logits = ct_linear(h.data(), d, n_tok, head_w_f32, hp.vocab_size, ct_to_f32_ref(m.dec_head_b).data());

    // DEBUG: print top-5 logits at last prompt token position
    return logits;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

// Pre-populate the ct_to_f32_ref cache for every model weight tensor.
// Called once at init so all inferences pay zero conversion cost.
static void cohere_model_warm_cache(const cohere_model & m) {
    fprintf(stderr, "cohere: warming F32 weight cache...\n");
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
    fprintf(stderr, "cohere: F32 weight cache ready\n");
}

struct cohere_context_params cohere_context_default_params(void) {
    return { .n_threads = 4, .use_flash = false };
}

struct cohere_context * cohere_init_from_file(const char * path_model,
                                              struct cohere_context_params params) {
    auto * ctx = new cohere_context;
    ctx->params = params;

    if (!cohere_load_model(ctx->model, ctx->vocab, path_model)) {
        delete ctx;
        return nullptr;
    }

    const auto & hp = ctx->model.hparams;
    int kv_n = hp.dec_n_heads * hp.dec_max_ctx * hp.dec_head_dim;
    ctx->kv_cache_k.assign(hp.dec_n_layers, std::vector<float>(kv_n, 0.0f));
    ctx->kv_cache_v.assign(hp.dec_n_layers, std::vector<float>(kv_n, 0.0f));

    // Note: F32 weight cache is lazy (populated on first use).
    // cohere_model_warm_cache() can be called by the application if desired for
    // server-mode workloads that process many utterances (amortises startup cost).

    return ctx;
}

void cohere_free(struct cohere_context * ctx) {
    if (!ctx) return;
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
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

char * cohere_transcribe(struct cohere_context * ctx,
                         const float * samples, int n_samples,
                         const char * lang) {
    const auto & hp  = ctx->model.hparams;
    const auto & voc = ctx->vocab;

    // --- Feature extraction ---
    fprintf(stderr, "cohere: computing mel features for %d samples (%.1fs)...\n",
            n_samples, (float)n_samples / hp.sample_rate);

    const auto & mel_fb = ct_to_f32_ref(ctx->model.fe_mel_fb);
    const auto & window = ct_to_f32_ref(ctx->model.fe_window);

    int T_mel = 0;
    auto mel = cohere_compute_features(hp,
        mel_fb.data() + 0, // skip batch dim → shape (n_mels, n_freqs) stored as [1,128,257]
        window.data(),
        samples, n_samples, T_mel);

    fprintf(stderr, "cohere: mel shape (%d, %d)\n", hp.n_mels, T_mel);

    // --- Encoder ---
    fprintf(stderr, "cohere: running encoder...\n");
    auto enc_out = cohere_encode(ctx->model, mel.data(), T_mel);
    int T_enc = (int)(enc_out.size() / hp.dec_d_model);
    fprintf(stderr, "cohere: encoder output T_enc=%d\n", T_enc);

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

    // Pre-compute cross KV cache once for this utterance
    cohere_precompute_cross_kv(ctx->model, enc_out.data(), T_enc,
                               ctx->cross_kv_k, ctx->cross_kv_v);

    // Reset self-attention KV cache
    for (auto & kv : ctx->kv_cache_k) std::fill(kv.begin(), kv.end(), 0.0f);
    for (auto & kv : ctx->kv_cache_v) std::fill(kv.begin(), kv.end(), 0.0f);

    const int eos_id  = tid("<|endoftext|>");
    const int max_gen = 100; // debug cap; was: hp.dec_max_ctx - (int)prompt.size() - 4

    // --- Run prompt through decoder ---
    auto logits = cohere_decode_step(ctx->model, T_enc,
                                      prompt.data(), (int)prompt.size(), 0,
                                      ctx->kv_cache_k, ctx->kv_cache_v,
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
        logits = cohere_decode_step(ctx->model, T_enc,
                                    &next_tok, 1, offset - 1,
                                    ctx->kv_cache_k, ctx->kv_cache_v,
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
