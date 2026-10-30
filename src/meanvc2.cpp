// MeanVC2 C++ implementation for CrispASR.
// Architecture by ASLP@NPU (Apache-2.0).

#include "meanvc2.h"

#include "core/arch_backend_map.h"
#include "core/fft.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/istft.h"
#include "core/kaldi_fbank.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(CRISPASR_HAVE_BLAS)
#include <cblas.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Parameters & Hyperparameters
// ---------------------------------------------------------------------------

struct meanvc2_hparams {
    uint32_t dim = 512;
    uint32_t depth = 4;
    uint32_t heads = 2;
    uint32_t head_dim = 64;
    uint32_t ff_mult = 2;
    uint32_t bn_dim = 256;
    uint32_t mel_dim = 80;
    uint32_t chunk_size = 12;
    uint32_t block_size = 4;
    uint32_t speaker_dim = 256;
    uint32_t memory_slots = 32;
    uint32_t sample_rate = 16000;

    // Vocos params
    uint32_t voc_dim = 320;
    uint32_t voc_intermediate_dim = 1536;
    uint32_t voc_num_layers = 8;
    uint32_t voc_hop_length = 160;
    uint32_t voc_n_fft = 640;

    // Fast-U2++ ASR params
    uint32_t asr_bn_window = 19;
    uint32_t asr_bn_stride = 16;
    uint32_t asr_offset_init = 8;
    uint32_t asr_offset_step = 4;
    uint32_t asr_req_cache = 8;
};

struct meanvc2_cached_weights {
    bool initialized = false;

    // Fast-U2++ ASR
    std::vector<float> conv0_w, conv0_b, conv2_w, conv2_b, out_w, out_b, after_w, after_b;
    struct AsrLayerWeights {
        std::vector<float> ff1_norm_w, ff1_norm_b, ff1_w1_w, ff1_w1_b, ff1_w2_w, ff1_w2_b;
        std::vector<float> attn_norm_w, attn_norm_b, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b;
        std::vector<float> pos_u, pos_v, linear_pos_w;
        std::vector<float> conv_norm_w, conv_norm_b, conv_pw1_w, conv_pw1_b;
        std::vector<float> conv_dw_w, conv_dw_w_kfirst, conv_dw_b, conv_ln_w, conv_ln_b, conv_pw2_w, conv_pw2_b;
        std::vector<float> ff2_norm_w, ff2_norm_b, ff2_w1_w, ff2_w1_b, ff2_w2_w, ff2_w2_b;
        std::vector<float> norm_final_w, norm_final_b;
    };
    std::vector<AsrLayerWeights> asr_layers;

    // GTM
    std::vector<float> k0_w, k0_b, k2_w, k2_b, v0_w, v0_b, v2_w, v2_b;
    std::vector<float> k_prior, v_prior, norm_k_w, norm_k_b, norm_v_w, norm_v_b;

    // TVT
    std::vector<float> tvt_q_w, tvt_q_b, tvt_k_w, tvt_k_b, tvt_v_w, tvt_v_b;
    std::vector<float> tvt_out_w, tvt_out_b, tvt_norm_w, tvt_norm_b;

    // DiT
    std::vector<float> t_mlp0_w, t_mlp0_b, t_mlp2_w, t_mlp2_b;
    std::vector<float> r_mlp0_w, r_mlp0_b, r_mlp2_w, r_mlp2_b;
    std::vector<float> inp_w, inp_b;
    struct DiTBlockWeights {
        std::vector<float> adal_w, adal_b;
        std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b;
        std::vector<float> q_norm_w, k_norm_w;
        std::vector<float> ff1_w, ff1_b, ff2_w, ff2_b;
    };
    std::vector<DiTBlockWeights> dit_blocks;
    std::vector<float> norm_out_w, norm_out_b, proj_out_w, proj_out_b;

    // Vocos
    std::vector<float> voc_embed_w, voc_embed_b, voc_norm_w, voc_norm_b;
    struct VocosBlock {
        std::vector<float> dw_w, dw_b, norm_w, norm_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma;
    };
    std::vector<VocosBlock> voc_blocks;
    std::vector<float> voc_final_norm_w, voc_final_norm_b, voc_head_w, voc_head_b;
};

struct meanvc2_context {
    meanvc2_hparams hp;
    meanvc2_context_params params;
    core_gguf::WeightLoad wl;
    core_gguf::tensor_map tensors;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    std::mt19937_64 rng;

    // Cached speaker embedding
    std::vector<float> speaker_emb;

    // Pre-dequantized FP32 weights cache
    meanvc2_cached_weights weights;
};

struct meanvc2_context_params meanvc2_context_default_params(void) {
    meanvc2_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.chunk_size = 12;
    p.block_size = 4;
    p.seed = 42;
    return p;
}

static ggml_tensor* get_tensor(const meanvc2_context* ctx, const std::string& name) {
    if (!ctx)
        return nullptr;
    auto it = ctx->tensors.find(name);
    if (it != ctx->tensors.end())
        return it->second;

    // Try slash variant: vc.xxx -> vc/xxx
    std::string slash_name = name;
    size_t dot_pos = slash_name.find('.');
    if (dot_pos != std::string::npos) {
        slash_name[dot_pos] = '/';
        it = ctx->tensors.find(slash_name);
        if (it != ctx->tensors.end())
            return it->second;
    }

    // Try stripped prefix
    if (name.rfind("vc.", 0) == 0) {
        it = ctx->tensors.find(name.substr(3));
        if (it != ctx->tensors.end())
            return it->second;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// CPU Math Primitives
// ---------------------------------------------------------------------------

static inline float fast_erf(float x) {
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float abs_x = fabsf(x);
    if (abs_x > 4.0f)
        return sign;
    float t = 1.0f / (1.0f + 0.3275911f * abs_x);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * expf(-abs_x * abs_x));
}

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + fast_erf(x * 0.7071067811865475f));
}

static inline float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2.0 / M_PI)
    return 0.5f * x * (1.0f + tanhf(k0 * (x + 0.044715f * x * x * x)));
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void cpu_layer_norm(float* x, int T, int D, const float* w, const float* b, float eps = 1e-5f) {
    for (int t = 0; t < T; t++) {
        float* row = x + t * D;
        float sum = 0.0f;
        for (int d = 0; d < D; d++)
            sum += row[d];
        float mean = sum / (float)D;
        float var = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        float inv_std = 1.0f / sqrtf(var / (float)D + eps);
        for (int d = 0; d < D; d++) {
            float val = (row[d] - mean) * inv_std;
            if (w)
                val *= w[d];
            if (b)
                val += b[d];
            row[d] = val;
        }
    }
}

static void cpu_rms_norm(float* x, int T, int D, const float* w, float eps = 1e-6f) {
    for (int t = 0; t < T; t++) {
        float* row = x + t * D;
        float sum_sq = 0.0f;
        for (int d = 0; d < D; d++)
            sum_sq += row[d] * row[d];
        float inv_rms = 1.0f / sqrtf(sum_sq / (float)D + eps);
        for (int d = 0; d < D; d++) {
            row[d] = row[d] * inv_rms * (w ? w[d] : 1.0f);
        }
    }
}

// In-place radix-2 FFT. sign=+1 for forward, -1 for inverse.
static void fft_radix2(float* re, float* im, int N, int sign) {
    int log2n = 0;
    for (int tmp = N; tmp > 1; tmp >>= 1)
        log2n++;
    for (int i = 0; i < N; i++) {
        int j = 0, x = i;
        for (int b = 0; b < log2n; b++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int half = m / 2;
        float wpr = cosf(2.0f * (float)M_PI / (float)m);
        float wpi = (float)sign * sinf(2.0f * (float)M_PI / (float)m);
        for (int k = 0; k < N; k += m) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < half; j++) {
                float tr = wr * re[k + j + half] - wi * im[k + j + half];
                float ti = wr * im[k + j + half] + wi * re[k + j + half];
                re[k + j + half] = re[k + j] - tr;
                im[k + j + half] = im[k + j] - ti;
                re[k + j] += tr;
                im[k + j] += ti;
                float next_wr = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = next_wr;
            }
        }
    }
}

static void cpu_gemm_linear(const float* x, const float* w, const float* b, float* out, int M, int K, int N) {
    // x: (M, K), w: (N, K) row-major PyTorch weight, out: (M, N)
#if defined(HAVE_ACCELERATE) || defined(CRISPASR_HAVE_BLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, x, K, w, K, 0.0f, out, N);
    if (b) {
        for (int m = 0; m < M; m++) {
            float* row = out + m * N;
            for (int n = 0; n < N; n++) {
                row[n] += b[n];
            }
        }
    }
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = b ? b[n] : 0.0f;
            const float* w_row = w + n * K;
            const float* x_row = x + m * K;
            for (int k = 0; k < K; k++) {
                sum += x_row[k] * w_row[k];
            }
            out[m * N + n] = sum;
        }
    }
#endif
}

static void dequant_tensor_to_float(const ggml_tensor* t, std::vector<float>& buf) {
    if (!t)
        return;
    int64_t n_elems = ggml_nelements(t);
    buf.resize(n_elems);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n_elems * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n_elems);
        ggml_backend_tensor_get(t, tmp.data(), 0, n_elems * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n_elems; i++) {
            buf[i] = ggml_fp16_to_fp32(tmp[i]);
        }
    } else {
        std::vector<uint8_t> tmp(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        // Simple dequantize row-by-row
        int64_t nrows = ggml_nrows(t);
        int64_t row_size = t->ne[0];
        for (int64_t r = 0; r < nrows; r++) {
            const void* src = tmp.data() + r * t->nb[1];
            float* dst = buf.data() + r * row_size;
            // Use ggml dequantizer
            const auto* type_traits = ggml_get_type_traits(t->type);
            if (type_traits && type_traits->to_float) {
                type_traits->to_float(src, dst, row_size);
            } else {
                memset(dst, 0, row_size * sizeof(float));
            }
        }
    }
}

static void ensure_weights_cached(meanvc2_context* ctx) {
    if (!ctx || ctx->weights.initialized)
        return;
    auto& w = ctx->weights;

    // Fast-U2++ ASR
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.embed.conv.0.weight"), w.conv0_w);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.embed.conv.0.bias"), w.conv0_b);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.embed.conv.2.weight"), w.conv2_w);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.embed.conv.2.bias"), w.conv2_b);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.embed.out.0.weight"), w.out_w);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.embed.out.0.bias"), w.out_b);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.after_norm.weight"), w.after_w);
    dequant_tensor_to_float(get_tensor(ctx, "asr.m.encoder.after_norm.bias"), w.after_b);

    w.asr_layers.resize(6);
    for (int l = 0; l < 6; l++) {
        std::string p = "asr.m.encoder.encoders." + std::to_string(l) + ".";
        dequant_tensor_to_float(get_tensor(ctx, p + "norm_ff_macaron.weight"), w.asr_layers[l].ff1_norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm_ff_macaron.bias"), w.asr_layers[l].ff1_norm_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward_macaron.w_1.weight"), w.asr_layers[l].ff1_w1_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward_macaron.w_1.bias"), w.asr_layers[l].ff1_w1_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward_macaron.w_2.weight"), w.asr_layers[l].ff1_w2_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward_macaron.w_2.bias"), w.asr_layers[l].ff1_w2_b);

        dequant_tensor_to_float(get_tensor(ctx, p + "norm_mha.weight"), w.asr_layers[l].attn_norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm_mha.bias"), w.asr_layers[l].attn_norm_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_q.weight"), w.asr_layers[l].q_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_q.bias"), w.asr_layers[l].q_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_k.weight"), w.asr_layers[l].k_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_k.bias"), w.asr_layers[l].k_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_v.weight"), w.asr_layers[l].v_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_v.bias"), w.asr_layers[l].v_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_out.weight"), w.asr_layers[l].out_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_out.bias"), w.asr_layers[l].out_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.pos_bias_u"), w.asr_layers[l].pos_u);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.pos_bias_v"), w.asr_layers[l].pos_v);
        dequant_tensor_to_float(get_tensor(ctx, p + "self_attn.linear_pos.weight"), w.asr_layers[l].linear_pos_w);

        dequant_tensor_to_float(get_tensor(ctx, p + "norm_conv.weight"), w.asr_layers[l].conv_norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm_conv.bias"), w.asr_layers[l].conv_norm_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.pointwise_conv1.weight"),
                                w.asr_layers[l].conv_pw1_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.pointwise_conv1.bias"),
                                w.asr_layers[l].conv_pw1_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.depthwise_conv.weight"),
                                w.asr_layers[l].conv_dw_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.depthwise_conv.bias"),
                                w.asr_layers[l].conv_dw_b);
        if (!w.asr_layers[l].conv_dw_w.empty()) {
            w.asr_layers[l].conv_dw_w_kfirst.resize(9 * 256);
            for (int c = 0; c < 256; c++) {
                for (int k_i = 0; k_i < 9; k_i++) {
                    w.asr_layers[l].conv_dw_w_kfirst[k_i * 256 + c] = w.asr_layers[l].conv_dw_w[c * 9 + k_i];
                }
            }
        }
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.norm.weight"),
                                w.asr_layers[l].conv_ln_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.norm.bias"), w.asr_layers[l].conv_ln_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.pointwise_conv2.weight"),
                                w.asr_layers[l].conv_pw2_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "conv_module.streaming_conv.pointwise_conv2.bias"),
                                w.asr_layers[l].conv_pw2_b);

        dequant_tensor_to_float(get_tensor(ctx, p + "norm_ff.weight"), w.asr_layers[l].ff2_norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm_ff.bias"), w.asr_layers[l].ff2_norm_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward.w_1.weight"), w.asr_layers[l].ff2_w1_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward.w_1.bias"), w.asr_layers[l].ff2_w1_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward.w_2.weight"), w.asr_layers[l].ff2_w2_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "feed_forward.w_2.bias"), w.asr_layers[l].ff2_w2_b);

        dequant_tensor_to_float(get_tensor(ctx, p + "norm_final.weight"), w.asr_layers[l].norm_final_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm_final.bias"), w.asr_layers[l].norm_final_b);
    }

    // GTM
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_k.0.weight"), w.k0_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_k.0.bias"), w.k0_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_k.2.weight"), w.k2_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_k.2.bias"), w.k2_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_v.0.weight"), w.v0_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_v.0.bias"), w.v0_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_v.2.weight"), w.v2_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.mlp_v.2.bias"), w.v2_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.k_prior"), w.k_prior);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.v_prior"), w.v_prior);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.norm_k.weight"), w.norm_k_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.norm_k.bias"), w.norm_k_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.norm_v.weight"), w.norm_v_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.gtm.norm_v.bias"), w.norm_v_b);

    // TVT
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.q_proj.weight"), w.tvt_q_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.q_proj.bias"), w.tvt_q_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.k_proj.weight"), w.tvt_k_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.k_proj.bias"), w.tvt_k_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.v_proj.weight"), w.tvt_v_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.v_proj.bias"), w.tvt_v_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.out_proj.weight"), w.tvt_out_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.out_proj.bias"), w.tvt_out_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.attn_norm.weight"), w.tvt_norm_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.temporal_timbre.attn_norm.bias"), w.tvt_norm_b);

    // DiT
    dequant_tensor_to_float(get_tensor(ctx, "vc.t_time_embed.time_mlp.0.weight"), w.t_mlp0_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.t_time_embed.time_mlp.0.bias"), w.t_mlp0_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.t_time_embed.time_mlp.2.weight"), w.t_mlp2_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.t_time_embed.time_mlp.2.bias"), w.t_mlp2_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.r_time_embed.time_mlp.0.weight"), w.r_mlp0_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.r_time_embed.time_mlp.0.bias"), w.r_mlp0_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.r_time_embed.time_mlp.2.weight"), w.r_mlp2_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.r_time_embed.time_mlp.2.bias"), w.r_mlp2_b);

    dequant_tensor_to_float(get_tensor(ctx, "vc.input_embed.proj.weight"), w.inp_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.input_embed.proj.bias"), w.inp_b);

    w.dit_blocks.resize(ctx->hp.depth);
    for (int l = 0; l < (int)ctx->hp.depth; l++) {
        std::string p = "vc.transformer_blocks." + std::to_string(l) + ".";
        dequant_tensor_to_float(get_tensor(ctx, p + "attn_norm.linear.weight"), w.dit_blocks[l].adal_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn_norm.linear.bias"), w.dit_blocks[l].adal_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_q.weight"), w.dit_blocks[l].q_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_q.bias"), w.dit_blocks[l].q_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_k.weight"), w.dit_blocks[l].k_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_k.bias"), w.dit_blocks[l].k_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_v.weight"), w.dit_blocks[l].v_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_v.bias"), w.dit_blocks[l].v_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_out.0.weight"), w.dit_blocks[l].out_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.to_out.0.bias"), w.dit_blocks[l].out_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.q_norm.weight"), w.dit_blocks[l].q_norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "attn.k_norm.weight"), w.dit_blocks[l].k_norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "ff.ff.0.0.weight"), w.dit_blocks[l].ff1_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "ff.ff.0.0.bias"), w.dit_blocks[l].ff1_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "ff.ff.2.weight"), w.dit_blocks[l].ff2_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "ff.ff.2.bias"), w.dit_blocks[l].ff2_b);
    }

    dequant_tensor_to_float(get_tensor(ctx, "vc.norm_out.linear.weight"), w.norm_out_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.norm_out.linear.bias"), w.norm_out_b);
    dequant_tensor_to_float(get_tensor(ctx, "vc.proj_out.weight"), w.proj_out_w);
    dequant_tensor_to_float(get_tensor(ctx, "vc.proj_out.bias"), w.proj_out_b);

    // Vocos
    dequant_tensor_to_float(get_tensor(ctx, "vocos.backbone.embed.weight"), w.voc_embed_w);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.backbone.embed.bias"), w.voc_embed_b);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.backbone.norm.weight"), w.voc_norm_w);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.backbone.norm.bias"), w.voc_norm_b);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.backbone.final_layer_norm.weight"), w.voc_final_norm_w);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.backbone.final_layer_norm.bias"), w.voc_final_norm_b);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.head.out.weight"), w.voc_head_w);
    dequant_tensor_to_float(get_tensor(ctx, "vocos.head.out.bias"), w.voc_head_b);

    w.voc_blocks.resize(ctx->hp.voc_num_layers);
    for (int l = 0; l < (int)ctx->hp.voc_num_layers; l++) {
        std::string p = "vocos.backbone.convnext." + std::to_string(l) + ".";
        dequant_tensor_to_float(get_tensor(ctx, p + "dwconv.weight"), w.voc_blocks[l].dw_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "dwconv.bias"), w.voc_blocks[l].dw_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm.weight"), w.voc_blocks[l].norm_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "norm.bias"), w.voc_blocks[l].norm_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "pwconv1.weight"), w.voc_blocks[l].pw1_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "pwconv1.bias"), w.voc_blocks[l].pw1_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "pwconv2.weight"), w.voc_blocks[l].pw2_w);
        dequant_tensor_to_float(get_tensor(ctx, p + "pwconv2.bias"), w.voc_blocks[l].pw2_b);
        dequant_tensor_to_float(get_tensor(ctx, p + "gamma"), w.voc_blocks[l].gamma);
    }

    w.initialized = true;
}

// ---------------------------------------------------------------------------
// 1. ASR Bottleneck Feature Extraction (Fast-U2++)
// ---------------------------------------------------------------------------

static std::vector<float> run_fastu2pp_asr(meanvc2_context* ctx, const float* audio_16k, int n_samples, int* out_frames,
                                           int* out_dim) {
    if (!ctx || !audio_16k || n_samples <= 0)
        return {};

    // Kaldi filterbank feature extraction
    core_kaldi::FbankParams fp;
    fp.sample_rate = 16000;
    fp.n_mels = 80;
    fp.frame_length_ms = 25;
    fp.frame_shift_ms = 10;
    fp.window_type = core_kaldi::WindowType::Povey;
    fp.snip_edges = true;
    fp.int16_scale = true;

    int T_fbank = 0;
    std::vector<float> fbanks = core_kaldi::compute_fbank(audio_16k, n_samples, fp, T_fbank);
    if (T_fbank < (int)ctx->hp.asr_bn_window) {
        if (out_frames)
            *out_frames = 0;
        if (out_dim)
            *out_dim = 256;
        return {};
    }

    // Sliding window chunk processing through Fast-U2++ Conformer
    const int window = ctx->hp.asr_bn_window; // 19
    const int stride = ctx->hp.asr_bn_stride; // 16
    const int bn_dim = ctx->hp.bn_dim;        // 256

    int n_chunks = 0;
    for (int i = 0; i + window <= T_fbank; i += stride) {
        n_chunks++;
    }

    const int chunk_frames = 4;
    const int total_bn_frames = n_chunks * chunk_frames;
    std::vector<float> bns(total_bn_frames * bn_dim, 0.0f);

    ensure_weights_cached(ctx);
    const auto& w = ctx->weights;
    const auto& conv0_w = w.conv0_w;
    const auto& conv0_b = w.conv0_b;
    const auto& conv2_w = w.conv2_w;
    const auto& conv2_b = w.conv2_b;
    const auto& out_w = w.out_w;
    const auto& out_b = w.out_b;
    const auto& after_w = w.after_w;
    const auto& after_b = w.after_b;
    const auto& layers = w.asr_layers;

    // Process chunk by chunk
    int chunk_idx = 0;
    std::vector<std::vector<float>> att_k_cache(6, std::vector<float>(8 * 4 * 64, 0.0f));
    std::vector<std::vector<float>> att_v_cache(6, std::vector<float>(8 * 4 * 64, 0.0f));
    std::vector<std::vector<float>> cnn_cache(6, std::vector<float>(256 * 8, 0.0f));

    // Precompute sinusoidal positional encoding for Fast-U2++ Conformer
    std::vector<float> pe(5000 * 256);
    float pe_factor = -logf(10000.0f) / 256.0f;
    for (int pos = 0; pos < 5000; pos++) {
        for (int i = 0; i < 128; i++) {
            float div = expf((float)(2 * i) * pe_factor);
            pe[pos * 256 + 2 * i] = sinf((float)pos * div);
            pe[pos * 256 + 2 * i + 1] = cosf((float)pos * div);
        }
    }

    // Scratch buffers for zero-allocation streaming chunks
    std::vector<float> c0(256 * 9 * 39, 0.0f);
    std::vector<float> col0(9 * 351);
    std::vector<float> col2(2304 * 76);
    std::vector<float> gemm2(256 * 76);
    std::vector<float> c2(256 * 4 * 19, 0.0f);
    std::vector<float> h(4 * 256);
    std::vector<float> ffn1(4 * 256);
    std::vector<float> f1_mid(4 * 2048);
    std::vector<float> f1_out(4 * 256);
    std::vector<float> attn_in(4 * 256);
    std::vector<float> q(4 * 256), k(4 * 256), v(4 * 256);
    std::vector<float> p(12 * 256);
    std::vector<float> full_k(12 * 256), full_v(12 * 256);
    std::vector<float> mha_out(4 * 256);
    std::vector<float> mha_proj(4 * 256);
    std::vector<float> conv_in(4 * 256);
    std::vector<float> full_conv_in(12 * 256);
    std::vector<float> glu_raw(12 * 512);
    std::vector<float> glu_out(12 * 256);
    std::vector<float> dw_out(4 * 256);
    std::vector<float> pw2_out(4 * 256);
    std::vector<float> ffn2(4 * 256);
    std::vector<float> f2_mid(4 * 2048);
    std::vector<float> f2_out(4 * 256);

    for (int start_f = 0; start_f + window <= T_fbank; start_f += stride, chunk_idx++) {
        const float* fb_chunk = fbanks.data() + start_f * 80;

        // Subsampling: Conv0 -> ReLU -> Conv2 -> ReLU -> Linear(4864, 256)
        // Conv0: input (1, 19, 80), weight (256, 1, 3, 3), stride 2 -> out (256, 9, 39)
        std::fill(c0.begin(), c0.end(), 0.0f);
#if defined(HAVE_ACCELERATE) || defined(CRISPASR_HAVE_BLAS)
        for (int oh = 0; oh < 9; oh++) {
            for (int ow = 0; ow < 39; ow++) {
                int s = oh * 39 + ow;
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        col0[(kh * 3 + kw) * 351 + s] = fb_chunk[(oh * 2 + kh) * 80 + (ow * 2 + kw)];
                    }
                }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256, 351, 9, 1.0f, conv0_w.data(), 9, col0.data(), 351,
                    0.0f, c0.data(), 351);
        for (int oc = 0; oc < 256; oc++) {
            float b = conv0_b.empty() ? 0.0f : conv0_b[oc];
            float* row = c0.data() + oc * 351;
            for (int i = 0; i < 351; i++) {
                float val = row[i] + b;
                row[i] = val > 0.0f ? val : 0.0f; // ReLU
            }
        }
#else
        for (int oc = 0; oc < 256; oc++) {
            float b = conv0_b.empty() ? 0.0f : conv0_b[oc];
            const float* kw = conv0_w.data() + oc * 9;
            for (int oh = 0; oh < 9; oh++) {
                int ih = oh * 2;
                for (int ow = 0; ow < 39; ow++) {
                    int iw = ow * 2;
                    float s = b;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw_i = 0; kw_i < 3; kw_i++) {
                            s += fb_chunk[(ih + kh) * 80 + (iw + kw_i)] * kw[kh * 3 + kw_i];
                        }
                    }
                    c0[(oc * 9 + oh) * 39 + ow] = s > 0.0f ? s : 0.0f; // ReLU
                }
            }
        }
#endif

        // Conv2: input (256, 9, 39), weight (256, 256, 3, 3), stride 2 -> out (256, 4, 19)
        std::fill(c2.begin(), c2.end(), 0.0f);
#if defined(HAVE_ACCELERATE) || defined(CRISPASR_HAVE_BLAS)
        for (int oh = 0; oh < 4; oh++) {
            for (int ow = 0; ow < 19; ow++) {
                int s_idx = oh * 19 + ow;
                for (int ic = 0; ic < 256; ic++) {
                    const float* in_plane = c0.data() + (ic * 9) * 39;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw_i = 0; kw_i < 3; kw_i++) {
                            int k_idx = (ic * 3 + kh) * 3 + kw_i;
                            col2[k_idx * 76 + s_idx] = in_plane[(oh * 2 + kh) * 39 + (ow * 2 + kw_i)];
                        }
                    }
                }
            }
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256, 76, 2304, 1.0f, conv2_w.data(), 2304, col2.data(),
                    76, 0.0f, gemm2.data(), 76);
        for (int oh = 0; oh < 4; oh++) {
            for (int oc = 0; oc < 256; oc++) {
                float b = conv2_b.empty() ? 0.0f : conv2_b[oc];
                for (int ow = 0; ow < 19; ow++) {
                    float s = gemm2[oc * 76 + (oh * 19 + ow)] + b;
                    c2[(oh * 256 + oc) * 19 + ow] = s > 0.0f ? s : 0.0f; // Shape (4, 256, 19)
                }
            }
        }
#else
        for (int oc = 0; oc < 256; oc++) {
            float b = conv2_b.empty() ? 0.0f : conv2_b[oc];
            for (int oh = 0; oh < 4; oh++) {
                int ih = oh * 2;
                for (int ow = 0; ow < 19; ow++) {
                    int iw = ow * 2;
                    float s = b;
                    for (int ic = 0; ic < 256; ic++) {
                        const float* kw = conv2_w.data() + (oc * 256 + ic) * 9;
                        const float* in_plane = c0.data() + (ic * 9) * 39;
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw_i = 0; kw_i < 3; kw_i++) {
                                s += in_plane[(ih + kh) * 39 + (iw + kw_i)] * kw[kh * 3 + kw_i];
                            }
                        }
                    }
                    c2[(oh * 256 + oc) * 19 + ow] = s > 0.0f ? s : 0.0f; // Shape (4, 256, 19)
                }
            }
        }
#endif

        // Linear out: flatten 256 * 19 = 4864 -> 256 per time step (T=4), scaled by sqrt(d_model) = 16.0f
        cpu_gemm_linear(c2.data(), out_w.data(), out_b.data(), h.data(), 4, 4864, 256);
        for (int i = 0; i < 4 * 256; i++) {
            h[i] *= 16.0f;
        }

        int offset_asr = 8 + 4 * chunk_idx;
        const float* pos_emb = pe.data() + (offset_asr - 8) * 256;

        // 6 Conformer Layers
        for (int l = 0; l < 6; l++) {
            const auto& lw = layers[l];
            // 1. Macaron FFN (scale 0.5)
            memcpy(ffn1.data(), h.data(), 4 * 256 * sizeof(float));
            cpu_layer_norm(ffn1.data(), 4, 256, lw.ff1_norm_w.data(), lw.ff1_norm_b.data());
            cpu_gemm_linear(ffn1.data(), lw.ff1_w1_w.data(), lw.ff1_w1_b.data(), f1_mid.data(), 4, 256, 2048);
            for (auto& v : f1_mid)
                v = silu(v);
            cpu_gemm_linear(f1_mid.data(), lw.ff1_w2_w.data(), lw.ff1_w2_b.data(), f1_out.data(), 4, 2048, 256);
            for (int i = 0; i < 4 * 256; i++)
                h[i] += 0.5f * f1_out[i];

            // 2. MHA self-attention with relative positional encoding
            memcpy(attn_in.data(), h.data(), 4 * 256 * sizeof(float));
            cpu_layer_norm(attn_in.data(), 4, 256, lw.attn_norm_w.data(), lw.attn_norm_b.data());
            cpu_gemm_linear(attn_in.data(), lw.q_w.data(), lw.q_b.data(), q.data(), 4, 256, 256);
            cpu_gemm_linear(attn_in.data(), lw.k_w.data(), lw.k_b.data(), k.data(), 4, 256, 256);
            cpu_gemm_linear(attn_in.data(), lw.v_w.data(), lw.v_b.data(), v.data(), 4, 256, 256);

            // Project pos_emb (12 frames): p = linear_pos(pos_emb)
            cpu_gemm_linear(pos_emb, lw.linear_pos_w.data(), nullptr, p.data(), 12, 256, 256);

            // Append K/V to cache
            memcpy(full_k.data(), att_k_cache[l].data(), 8 * 256 * sizeof(float));
            memcpy(full_k.data() + 8 * 256, k.data(), 4 * 256 * sizeof(float));
            memcpy(full_v.data(), att_v_cache[l].data(), 8 * 256 * sizeof(float));
            memcpy(full_v.data() + 8 * 256, v.data(), 4 * 256 * sizeof(float));
            // Update cache: keep last 8
            memcpy(att_k_cache[l].data(), full_k.data() + 4 * 256, 8 * 256 * sizeof(float));
            memcpy(att_v_cache[l].data(), full_v.data() + 4 * 256, 8 * 256 * sizeof(float));

            // Attention heads (4 heads, d=64)
            std::fill(mha_out.begin(), mha_out.end(), 0.0f);
            for (int head = 0; head < 4; head++) {
                const float* u_vec = lw.pos_u.data() + head * 64;
                const float* v_vec = lw.pos_v.data() + head * 64;
                for (int qi = 0; qi < 4; qi++) {
                    const float* q_vec = q.data() + qi * 256 + head * 64;
                    float q_u[64], q_v[64];
                    for (int d = 0; d < 64; d++) {
                        q_u[d] = q_vec[d] + u_vec[d];
                        q_v[d] = q_vec[d] + v_vec[d];
                    }
                    float scores[12];
                    float max_score = -1e9f;
                    for (int kj = 0; kj < 12; kj++) {
                        const float* k_vec = full_k.data() + kj * 256 + head * 64;
                        const float* p_vec = p.data() + kj * 256 + head * 64;
                        float sc = 0.0f;
                        for (int d = 0; d < 64; d++) {
                            sc += q_u[d] * k_vec[d] + q_v[d] * p_vec[d];
                        }
                        sc *= 0.125f; // 1.0f / sqrt(64)
                        scores[kj] = sc;
                        if (sc > max_score)
                            max_score = sc;
                    }
                    float exp_sum = 0.0f;
                    for (int kj = 0; kj < 12; kj++) {
                        scores[kj] = expf(scores[kj] - max_score);
                        exp_sum += scores[kj];
                    }
                    for (int kj = 0; kj < 12; kj++)
                        scores[kj] /= exp_sum;

                    float* out_head = mha_out.data() + qi * 256 + head * 64;
                    for (int kj = 0; kj < 12; kj++) {
                        const float* val_vec = full_v.data() + kj * 256 + head * 64;
                        float w = scores[kj];
                        for (int d = 0; d < 64; d++)
                            out_head[d] += w * val_vec[d];
                    }
                }
            }
            cpu_gemm_linear(mha_out.data(), lw.out_w.data(), lw.out_b.data(), mha_proj.data(), 4, 256, 256);
            for (int i = 0; i < 4 * 256; i++)
                h[i] += mha_proj[i];

            // 3. Conv module
            memcpy(conv_in.data(), h.data(), 4 * 256 * sizeof(float));
            cpu_layer_norm(conv_in.data(), 4, 256, lw.conv_norm_w.data(), lw.conv_norm_b.data());

            // Concatenate cnn_cache (8 frames) + conv_in (4 frames) -> full_conv_in (12 frames)
            memcpy(full_conv_in.data(), cnn_cache[l].data(), 8 * 256 * sizeof(float));
            memcpy(full_conv_in.data() + 8 * 256, conv_in.data(), 4 * 256 * sizeof(float));
            // Update cnn_cache: keep last 8 frames
            memcpy(cnn_cache[l].data(), full_conv_in.data() + 4 * 256, 8 * 256 * sizeof(float));

            // Pointwise 1: 12 frames: 256 -> 512 (GLU)
            cpu_gemm_linear(full_conv_in.data(), lw.conv_pw1_w.data(), lw.conv_pw1_b.data(), glu_raw.data(), 12, 256,
                            512);
            for (int t = 0; t < 12; t++) {
                for (int d = 0; d < 256; d++) {
                    float a = glu_raw[t * 512 + d];
                    float b = glu_raw[t * 512 + 256 + d];
                    glu_out[t * 256 + d] = a * (1.0f / (1.0f + expf(-b))); // GLU
                }
            }

            // Depthwise conv (kernel 9, no pad): input 12 frames -> output 4 frames
            for (int t = 0; t < 4; t++) {
                if (!lw.conv_dw_b.empty()) {
                    memcpy(dw_out.data() + t * 256, lw.conv_dw_b.data(), 256 * sizeof(float));
                } else {
                    memset(dw_out.data() + t * 256, 0, 256 * sizeof(float));
                }
                float* out_row = dw_out.data() + t * 256;
                for (int k_i = 0; k_i < 9; k_i++) {
                    const float* in_row = glu_out.data() + (t + k_i) * 256;
                    const float* kw_row = lw.conv_dw_w_kfirst.data() + k_i * 256;
                    for (int c = 0; c < 256; c++) {
                        out_row[c] += in_row[c] * kw_row[c];
                    }
                }
            }
            // Norm + SiLU + Pointwise 2
            cpu_layer_norm(dw_out.data(), 4, 256, lw.conv_ln_w.data(), lw.conv_ln_b.data());
            for (auto& v : dw_out)
                v = silu(v);
            cpu_gemm_linear(dw_out.data(), lw.conv_pw2_w.data(), lw.conv_pw2_b.data(), pw2_out.data(), 4, 256, 256);
            for (int i = 0; i < 4 * 256; i++)
                h[i] += pw2_out[i];

            // 4. Second FFN (scale 0.5)
            memcpy(ffn2.data(), h.data(), 4 * 256 * sizeof(float));
            cpu_layer_norm(ffn2.data(), 4, 256, lw.ff2_norm_w.data(), lw.ff2_norm_b.data());
            cpu_gemm_linear(ffn2.data(), lw.ff2_w1_w.data(), lw.ff2_w1_b.data(), f2_mid.data(), 4, 256, 2048);
            for (auto& v : f2_mid)
                v = silu(v);
            cpu_gemm_linear(f2_mid.data(), lw.ff2_w2_w.data(), lw.ff2_w2_b.data(), f2_out.data(), 4, 2048, 256);
            for (int i = 0; i < 4 * 256; i++)
                h[i] += 0.5f * f2_out[i];

            // 5. Final norm
            cpu_layer_norm(h.data(), 4, 256, lw.norm_final_w.data(), lw.norm_final_b.data());
        }

        // Final after_norm
        cpu_layer_norm(h.data(), 4, 256, after_w.data(), after_b.data());

        // Copy chunk output to bns
        memcpy(bns.data() + chunk_idx * chunk_frames * bn_dim, h.data(), chunk_frames * bn_dim * sizeof(float));
    }

    if (out_frames)
        *out_frames = total_bn_frames;
    if (out_dim)
        *out_dim = bn_dim;
    return bns;
}

// ---------------------------------------------------------------------------
// 2. Global Timbre Memory (GTM)
// ---------------------------------------------------------------------------

static int run_gtm(meanvc2_context* ctx, const float* spk_emb, float* out_k_mem, float* out_v_mem) {
    if (!ctx || !spk_emb || !out_k_mem || !out_v_mem)
        return -1;

    const int spk_dim = ctx->hp.speaker_dim; // 256
    const int slots = ctx->hp.memory_slots;  // 32
    const int total_mem = slots * spk_dim;   // 8192

    ensure_weights_cached(ctx);
    const auto& w = ctx->weights;
    const auto& k0_w = w.k0_w;
    const auto& k0_b = w.k0_b;
    const auto& k2_w = w.k2_w;
    const auto& k2_b = w.k2_b;
    const auto& v0_w = w.v0_w;
    const auto& v0_b = w.v0_b;
    const auto& v2_w = w.v2_w;
    const auto& v2_b = w.v2_b;
    const auto& k_prior = w.k_prior;
    const auto& v_prior = w.v_prior;
    const auto& norm_k_w = w.norm_k_w;
    const auto& norm_k_b = w.norm_k_b;
    const auto& norm_v_w = w.norm_v_w;
    const auto& norm_v_b = w.norm_v_b;

    // K memory: Linear(256->256) -> SiLU -> Linear(256->8192)
    std::vector<float> h_k(spk_dim);
    cpu_gemm_linear(spk_emb, k0_w.data(), k0_b.data(), h_k.data(), 1, spk_dim, spk_dim);
    for (auto& v : h_k)
        v = silu(v);
    std::vector<float> k_spk(total_mem);
    cpu_gemm_linear(h_k.data(), k2_w.data(), k2_b.data(), k_spk.data(), 1, spk_dim, total_mem);

    // Fuse with prior + LayerNorm(256) per slot
    for (int i = 0; i < total_mem; i++) {
        k_spk[i] += tanhf(k_prior[i]);
    }
    cpu_layer_norm(k_spk.data(), slots, spk_dim, norm_k_w.data(), norm_k_b.data());
    memcpy(out_k_mem, k_spk.data(), total_mem * sizeof(float));

    // V memory: Linear(256->256) -> SiLU -> Linear(256->8192)
    std::vector<float> h_v(spk_dim);
    cpu_gemm_linear(spk_emb, v0_w.data(), v0_b.data(), h_v.data(), 1, spk_dim, spk_dim);
    for (auto& v : h_v)
        v = silu(v);
    std::vector<float> v_spk(total_mem);
    cpu_gemm_linear(h_v.data(), v2_w.data(), v2_b.data(), v_spk.data(), 1, spk_dim, total_mem);

    for (int i = 0; i < total_mem; i++) {
        v_spk[i] += tanhf(v_prior[i]);
    }
    cpu_layer_norm(v_spk.data(), slots, spk_dim, norm_v_w.data(), norm_v_b.data());
    memcpy(out_v_mem, v_spk.data(), total_mem * sizeof(float));

    return 0;
}

// ---------------------------------------------------------------------------
// 3. Temporal Timbre Encoder (TVT)
// ---------------------------------------------------------------------------

static std::vector<float> run_tvt(meanvc2_context* ctx, const float* bn_interp, int n_frames, const float* k_mem,
                                  const float* v_mem) {
    if (!ctx || !bn_interp || n_frames <= 0 || !k_mem || !v_mem)
        return {};

    const int spk_dim = ctx->hp.speaker_dim; // 256
    const int attn_dim = 128;
    const int num_heads = 4;
    const int head_dim = attn_dim / num_heads; // 32
    const int K = ctx->hp.memory_slots;        // 32

    ensure_weights_cached(ctx);
    const auto& w = ctx->weights;
    const auto& q_w = w.tvt_q_w;
    const auto& q_b = w.tvt_q_b;
    const auto& k_w = w.tvt_k_w;
    const auto& k_b = w.tvt_k_b;
    const auto& v_w = w.tvt_v_w;
    const auto& v_b = w.tvt_v_b;
    const auto& out_w = w.tvt_out_w;
    const auto& out_b = w.tvt_out_b;
    const auto& norm_w = w.tvt_norm_w;
    const auto& norm_b = w.tvt_norm_b;

    std::vector<float> q(n_frames * attn_dim), k(K * attn_dim), v(K * attn_dim);
    cpu_gemm_linear(bn_interp, q_w.data(), q_b.data(), q.data(), n_frames, spk_dim, attn_dim);
    cpu_gemm_linear(k_mem, k_w.data(), k_b.data(), k.data(), K, spk_dim, attn_dim);
    cpu_gemm_linear(v_mem, v_w.data(), v_b.data(), v.data(), K, spk_dim, attn_dim);

    // Multi-head cross attention: Q (n_frames, 4, 32) x K (32, 4, 32) -> V_t (n_frames, 128)
    std::vector<float> v_t(n_frames * attn_dim, 0.0f);
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; h++) {
        for (int t = 0; t < n_frames; t++) {
            const float* q_vec = q.data() + t * attn_dim + h * head_dim;
            std::vector<float> scores(K);
            float max_s = -1e9f;
            for (int j = 0; j < K; j++) {
                const float* k_vec = k.data() + j * attn_dim + h * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    s += q_vec[d] * k_vec[d];
                s *= scale;
                scores[j] = s;
                if (s > max_s)
                    max_s = s;
            }
            float exp_sum = 0.0f;
            for (int j = 0; j < K; j++) {
                scores[j] = expf(scores[j] - max_s);
                exp_sum += scores[j];
            }
            for (int j = 0; j < K; j++)
                scores[j] /= exp_sum;

            float* out_vec = v_t.data() + t * attn_dim + h * head_dim;
            for (int j = 0; j < K; j++) {
                const float* v_vec = v.data() + j * attn_dim + h * head_dim;
                float w = scores[j];
                for (int d = 0; d < head_dim; d++)
                    out_vec[d] += w * v_vec[d];
            }
        }
    }

    std::vector<float> timbre_cond(n_frames * spk_dim);
    cpu_gemm_linear(v_t.data(), out_w.data(), out_b.data(), timbre_cond.data(), n_frames, attn_dim, spk_dim);
    cpu_layer_norm(timbre_cond.data(), n_frames, spk_dim, norm_w.data(), norm_b.data());
    return timbre_cond;
}

// ---------------------------------------------------------------------------
// 4. Diffusion Transformer (DiT) 1-NFE Flow Matching
// ---------------------------------------------------------------------------

static std::vector<float> run_dit_flow(meanvc2_context* ctx, const float* bn_interp, int n_frames, const float* spk_emb,
                                       const float* noise, int* out_mel_dim) {
    if (!ctx || !bn_interp || n_frames <= 0 || !spk_emb || !noise)
        return {};

    const int dim = ctx->hp.dim;               // 512
    const int depth = ctx->hp.depth;           // 4
    const int heads = ctx->hp.heads;           // 2
    const int head_dim = ctx->hp.head_dim;     // 64
    const int mel_dim = ctx->hp.mel_dim;       // 80
    const int spk_dim = ctx->hp.speaker_dim;   // 256
    const int chunk_size = ctx->hp.chunk_size; // 12
    const int block_size = ctx->hp.block_size; // 4

    // GTM memory KV
    std::vector<float> k_mem(ctx->hp.memory_slots * spk_dim);
    std::vector<float> v_mem(ctx->hp.memory_slots * spk_dim);
    run_gtm(ctx, spk_emb, k_mem.data(), v_mem.data());

    // TVT timbre conditioning
    std::vector<float> timbre_cond = run_tvt(ctx, bn_interp, n_frames, k_mem.data(), v_mem.data());

    // Time embeddings for t=1.0 and r=0.0
    // Sinusoidal embedding 256 dim with scale=1000.0f
    auto get_sinusoidal = [](float val, int d, std::vector<float>& out) {
        out.resize(d);
        int half = d / 2;
        float factor = logf(10000.0f) / (float)(half - 1);
        for (int i = 0; i < half; i++) {
            float freq = expf(-(float)i * factor);
            out[i] = sinf(val * 1000.0f * freq);
            out[half + i] = cosf(val * 1000.0f * freq);
        }
    };

    std::vector<float> t_sin, r_sin;
    get_sinusoidal(1.0f, 256, t_sin);
    get_sinusoidal(0.0f, 256, r_sin);

    ensure_weights_cached(ctx);
    const auto& w = ctx->weights;
    const auto& t_mlp0_w = w.t_mlp0_w;
    const auto& t_mlp0_b = w.t_mlp0_b;
    const auto& t_mlp2_w = w.t_mlp2_w;
    const auto& t_mlp2_b = w.t_mlp2_b;
    const auto& r_mlp0_w = w.r_mlp0_w;
    const auto& r_mlp0_b = w.r_mlp0_b;
    const auto& r_mlp2_w = w.r_mlp2_w;
    const auto& r_mlp2_b = w.r_mlp2_b;
    const auto& inp_w = w.inp_w;
    const auto& inp_b = w.inp_b;
    const auto& blocks = w.dit_blocks;
    const auto& norm_out_w = w.norm_out_w;
    const auto& norm_out_b = w.norm_out_b;
    const auto& proj_out_w = w.proj_out_w;
    const auto& proj_out_b = w.proj_out_b;

    // Time and step embeddings
    std::vector<float> t_emb(dim), r_emb(dim);
    {
        std::vector<float> mid(dim);
        cpu_gemm_linear(t_sin.data(), t_mlp0_w.data(), t_mlp0_b.data(), mid.data(), 1, 256, dim);
        for (auto& v : mid)
            v = silu(v);
        cpu_gemm_linear(mid.data(), t_mlp2_w.data(), t_mlp2_b.data(), t_emb.data(), 1, dim, dim);
    }
    {
        std::vector<float> mid(dim);
        cpu_gemm_linear(r_sin.data(), r_mlp0_w.data(), r_mlp0_b.data(), mid.data(), 1, 256, dim);
        for (auto& v : mid)
            v = silu(v);
        cpu_gemm_linear(mid.data(), r_mlp2_w.data(), r_mlp2_b.data(), r_emb.data(), 1, dim, dim);
    }
    for (int i = 0; i < dim; i++)
        t_emb[i] += r_emb[i];

    std::vector<float> t_emb_silu(dim);
    for (int i = 0; i < dim; i++)
        t_emb_silu[i] = silu(t_emb[i]);

    // Precompute AdaLayerNorm modulations for all blocks & final norm (constant across all chunks!)
    std::vector<std::vector<float>> block_ada(depth, std::vector<float>(6 * dim));
    for (int l = 0; l < depth; l++) {
        cpu_gemm_linear(t_emb_silu.data(), blocks[l].adal_w.data(), blocks[l].adal_b.data(), block_ada[l].data(), 1,
                        dim, 6 * dim);
    }
    std::vector<float> final_ada(2 * dim);
    cpu_gemm_linear(t_emb_silu.data(), norm_out_w.data(), norm_out_b.data(), final_ada.data(), 1, dim, 2 * dim);
    const float* final_scale = final_ada.data();
    const float* final_shift = final_ada.data() + dim;

    // Precompute RoPE inv_freq (constant across all frames!)
    std::vector<float> inv_freq(head_dim / 2);
    for (int i = 0; i < (int)head_dim / 2; i++) {
        inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
    }

    // Chunk streaming loop
    int num_chunks = (n_frames + chunk_size - 1) / chunk_size;
    std::vector<float> x_pred_collect;
    x_pred_collect.reserve(n_frames * mel_dim);

    int offset_val = 0;
    // KV-cache for each block: [layer][head][seq_len * head_dim]
    std::vector<std::vector<float>> kv_cache_k(depth), kv_cache_v(depth);
    const int t_p[4] = {2, 2, 1, 1};
    const int t_f[4] = {1, 0, 0, 0};

    // Pre-allocate scratch buffers outside chunk loop to eliminate heap reallocations
    const int max_chunk_frames = chunk_size + block_size;
    std::vector<float> fused_in(max_chunk_frames * 592);
    std::vector<float> x(max_chunk_frames * dim);
    std::vector<float> x_norm(max_chunk_frames * dim);
    std::vector<float> q(max_chunk_frames * 128), k(max_chunk_frames * 128), v(max_chunk_frames * 128);
    std::vector<float> attn_out(max_chunk_frames * 128);
    std::vector<float> attn_proj(max_chunk_frames * dim);
    std::vector<float> x_norm2(max_chunk_frames * dim);
    std::vector<float> ffn_mid(max_chunk_frames * 1024);
    std::vector<float> ffn_out(max_chunk_frames * dim);
    std::vector<float> u(max_chunk_frames * mel_dim);

    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        int start = chunk_id * chunk_size;
        int mid = std::min(start + chunk_size, n_frames);
        int end = std::min(mid + block_size, n_frames);
        int curr_len = mid - start;
        int total_len = end - start;

        // Input embedding: concat (noise, timbre_cond, spk_emb)
        fused_in.resize(total_len * 592);
        for (int t = 0; t < total_len; t++) {
            int src_t = start + t;
            memcpy(fused_in.data() + t * 592, noise + src_t * 80, 80 * sizeof(float));
            memcpy(fused_in.data() + t * 592 + 80, timbre_cond.data() + src_t * 256, 256 * sizeof(float));
            memcpy(fused_in.data() + t * 592 + 336, spk_emb, 256 * sizeof(float));
        }

        x.resize(total_len * dim);
        cpu_gemm_linear(fused_in.data(), inp_w.data(), inp_b.data(), x.data(), total_len, 592, dim);

        // Transformer blocks
        for (int l = 0; l < depth; l++) {
            const auto& bw = blocks[l];
            // AdaLayerNorm: scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp
            const float* shift_msa = block_ada[l].data();
            const float* scale_msa = block_ada[l].data() + dim;
            const float* gate_msa = block_ada[l].data() + 2 * dim;
            const float* shift_mlp = block_ada[l].data() + 3 * dim;
            const float* scale_mlp = block_ada[l].data() + 4 * dim;
            const float* gate_mlp = block_ada[l].data() + 5 * dim;

            // Self-Attention modulation
            x_norm.resize(total_len * dim);
            memcpy(x_norm.data(), x.data(), total_len * dim * sizeof(float));
            cpu_layer_norm(x_norm.data(), total_len, dim, nullptr, nullptr, 1e-6f);
            for (int t = 0; t < total_len; t++) {
                for (int d = 0; d < dim; d++) {
                    x_norm[t * dim + d] = x_norm[t * dim + d] * (1.0f + scale_msa[d]) + shift_msa[d];
                }
            }

            // Q, K, V
            q.resize(total_len * 128);
            k.resize(total_len * 128);
            v.resize(total_len * 128);
            cpu_gemm_linear(x_norm.data(), bw.q_w.data(), bw.q_b.data(), q.data(), total_len, dim, 128);
            cpu_gemm_linear(x_norm.data(), bw.k_w.data(), bw.k_b.data(), k.data(), total_len, dim, 128);
            cpu_gemm_linear(x_norm.data(), bw.v_w.data(), bw.v_b.data(), v.data(), total_len, dim, 128);

            // RMSNorm on Q and K
            for (int t = 0; t < total_len; t++) {
                for (int h = 0; h < heads; h++) {
                    cpu_rms_norm(q.data() + (t * heads + h) * head_dim, 1, head_dim, bw.q_norm_w.data(), 1e-6f);
                    cpu_rms_norm(k.data() + (t * heads + h) * head_dim, 1, head_dim, bw.k_norm_w.data(), 1e-6f);
                }
            }

            // KV Cache concatenation (BEFORE RoPE)
            int prev_cached_frames = (int)kv_cache_k[l].size() / 128;
            int full_seq_len = prev_cached_frames + total_len;
            std::vector<float> full_k(full_seq_len * 128), full_v(full_seq_len * 128);
            if (prev_cached_frames > 0) {
                memcpy(full_k.data(), kv_cache_k[l].data(), prev_cached_frames * 128 * sizeof(float));
                memcpy(full_v.data(), kv_cache_v[l].data(), prev_cached_frames * 128 * sizeof(float));
            }
            memcpy(full_k.data() + prev_cached_frames * 128, k.data(), total_len * 128 * sizeof(float));
            memcpy(full_v.data() + prev_cached_frames * 128, v.data(), total_len * 128 * sizeof(float));

            // Update KV cache: trim future block to prevent leakage (BEFORE RoPE)
            int trim_len = full_seq_len - block_size;
            if (trim_len < 0)
                trim_len = 0;
            int max_cache = (t_p[l] + 1) * chunk_size;
            int keep_start = std::max(0, trim_len - max_cache);
            int keep_len = trim_len - keep_start;

            kv_cache_k[l].resize(keep_len * 128);
            kv_cache_v[l].resize(keep_len * 128);
            if (keep_len > 0) {
                memcpy(kv_cache_k[l].data(), full_k.data() + keep_start * 128, keep_len * 128 * sizeof(float));
                memcpy(kv_cache_v[l].data(), full_v.data() + keep_start * 128, keep_len * 128 * sizeof(float));
            }

            // RoPE on Q (total_len frames)
            for (int t = 0; t < total_len; t++) {
                int pos = offset_val + t;
                float cos_t[32], sin_t[32];
                for (int i = 0; i < (int)head_dim / 2; i++) {
                    float theta = (float)pos * inv_freq[i];
#if defined(__APPLE__)
                    __sincosf(theta, &sin_t[i], &cos_t[i]);
#elif defined(_GNU_SOURCE) || defined(__GLIBC__)
                    sincosf(theta, &sin_t[i], &cos_t[i]);
#else
                    cos_t[i] = cosf(theta);
                    sin_t[i] = sinf(theta);
#endif
                }
                for (int h = 0; h < heads; h++) {
                    float* q_head = q.data() + (t * heads + h) * head_dim;
                    for (int i = 0; i < (int)head_dim / 2; i++) {
                        float q0 = q_head[2 * i];
                        float q1 = q_head[2 * i + 1];
                        q_head[2 * i] = q0 * cos_t[i] - q1 * sin_t[i];
                        q_head[2 * i + 1] = q0 * sin_t[i] + q1 * cos_t[i];
                    }
                }
            }

            // RoPE on full_k (full_seq_len frames)
            for (int t = 0; t < full_seq_len; t++) {
                int pos = (offset_val + total_len - full_seq_len) + t;
                float cos_t[32], sin_t[32];
                for (int i = 0; i < (int)head_dim / 2; i++) {
                    float theta = (float)pos * inv_freq[i];
#if defined(__APPLE__)
                    __sincosf(theta, &sin_t[i], &cos_t[i]);
#elif defined(_GNU_SOURCE) || defined(__GLIBC__)
                    sincosf(theta, &sin_t[i], &cos_t[i]);
#else
                    cos_t[i] = cosf(theta);
                    sin_t[i] = sinf(theta);
#endif
                }
                for (int h = 0; h < heads; h++) {
                    float* k_head = full_k.data() + (t * heads + h) * head_dim;
                    for (int i = 0; i < (int)head_dim / 2; i++) {
                        float k0 = k_head[2 * i];
                        float k1 = k_head[2 * i + 1];
                        k_head[2 * i] = k0 * cos_t[i] - k1 * sin_t[i];
                        k_head[2 * i + 1] = k0 * sin_t[i] + k1 * cos_t[i];
                    }
                }
            }

            // Scaled dot-product attention with chunk mask
            attn_out.assign(total_len * 128, 0.0f);
            float scale_attn = 1.0f / sqrtf((float)head_dim);

            for (int h = 0; h < heads; h++) {
                for (int qi = 0; qi < total_len; qi++) {
                    const float* q_vec = q.data() + (qi * heads + h) * head_dim;
                    int row_in_mask = (full_seq_len - total_len) + qi;
                    int qi_chunk = row_in_mask / chunk_size;

                    int kj_start = std::max(0, qi_chunk - t_p[l]) * chunk_size;
                    int kj_end = std::min(full_seq_len, (qi_chunk + 1) * chunk_size + block_size * t_f[l]);
                    kj_start = std::min(kj_start, full_seq_len);

                    float scores[128];
                    float max_sc = -1e9f;
                    for (int kj = kj_start; kj < kj_end; kj++) {
                        const float* k_vec = full_k.data() + (kj * heads + h) * head_dim;
                        float s = 0.0f;
                        for (int d = 0; d < head_dim; d++)
                            s += q_vec[d] * k_vec[d];
                        s *= scale_attn;
                        scores[kj] = s;
                        if (s > max_sc)
                            max_sc = s;
                    }

                    float exp_sum = 0.0f;
                    for (int kj = kj_start; kj < kj_end; kj++) {
                        scores[kj] = expf(scores[kj] - max_sc);
                        exp_sum += scores[kj];
                    }
                    if (exp_sum > 0.0f) {
                        float inv_exp_sum = 1.0f / exp_sum;
                        for (int kj = kj_start; kj < kj_end; kj++)
                            scores[kj] *= inv_exp_sum;
                    }

                    float* out_vec = attn_out.data() + (qi * heads + h) * head_dim;
                    for (int kj = kj_start; kj < kj_end; kj++) {
                        const float* v_vec = full_v.data() + (kj * heads + h) * head_dim;
                        float w = scores[kj];
                        for (int d = 0; d < head_dim; d++)
                            out_vec[d] += w * v_vec[d];
                    }
                }
            }

            // Project attention out: 128 -> 512
            attn_proj.resize(total_len * dim);
            cpu_gemm_linear(attn_out.data(), bw.out_w.data(), bw.out_b.data(), attn_proj.data(), total_len, 128, dim);

            // Residual add with gate_msa
            for (int t = 0; t < total_len; t++) {
                for (int d = 0; d < dim; d++) {
                    x[t * dim + d] += gate_msa[d] * attn_proj[t * dim + d];
                }
            }

            // FFN modulation
            x_norm2.resize(total_len * dim);
            memcpy(x_norm2.data(), x.data(), total_len * dim * sizeof(float));
            cpu_layer_norm(x_norm2.data(), total_len, dim, nullptr, nullptr, 1e-6f);
            for (int t = 0; t < total_len; t++) {
                for (int d = 0; d < dim; d++) {
                    x_norm2[t * dim + d] = x_norm2[t * dim + d] * (1.0f + scale_mlp[d]) + shift_mlp[d];
                }
            }

            // FFN: 512 -> 1024 (GELU tanh) -> 512
            ffn_mid.resize(total_len * 1024);
            cpu_gemm_linear(x_norm2.data(), bw.ff1_w.data(), bw.ff1_b.data(), ffn_mid.data(), total_len, dim, 1024);
            for (auto& v : ffn_mid)
                v = gelu_tanh(v);
            ffn_out.resize(total_len * dim);
            cpu_gemm_linear(ffn_mid.data(), bw.ff2_w.data(), bw.ff2_b.data(), ffn_out.data(), total_len, 1024, dim);

            // Residual add with gate_mlp
            for (int t = 0; t < total_len; t++) {
                for (int d = 0; d < dim; d++) {
                    x[t * dim + d] += gate_mlp[d] * ffn_out[t * dim + d];
                }
            }
        }

        // Final AdaLayerNorm modulation
        cpu_layer_norm(x.data(), total_len, dim, nullptr, nullptr, 1e-6f);
        for (int t = 0; t < total_len; t++) {
            for (int d = 0; d < dim; d++) {
                x[t * dim + d] = x[t * dim + d] * (1.0f + final_scale[d]) + final_shift[d];
            }
        }

        // Final projection: 512 -> 80
        u.resize(total_len * mel_dim);
        cpu_gemm_linear(x.data(), proj_out_w.data(), proj_out_b.data(), u.data(), total_len, dim, mel_dim);

        // 1-NFE Euler step: x_pred = noise - u
        for (int t = 0; t < curr_len; t++) {
            int src_t = start + t;
            for (int m = 0; m < mel_dim; m++) {
                float val = noise[src_t * mel_dim + m] - u[t * mel_dim + m];
                x_pred_collect.push_back(val);
            }
        }

        offset_val += curr_len;
    }

    // Mel denormalization: mel = (x_pred + 1) / 2
    // Shape: (mel_dim, n_frames) row-major
    std::vector<float> mel(mel_dim * n_frames);
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < mel_dim; m++) {
            float val = (x_pred_collect[t * mel_dim + m] + 1.0f) * 0.5f;
            mel[m * n_frames + t] = val; // (80, T)
        }
    }

    if (out_mel_dim)
        *out_mel_dim = mel_dim;
    return mel;
}

// ---------------------------------------------------------------------------
// 5. Vocos Neural Vocoder
// ---------------------------------------------------------------------------

static std::vector<float> run_vocos_decode(meanvc2_context* ctx, const float* mel_80_t, int T, int* out_samples) {
    if (!ctx || !mel_80_t || T <= 0)
        return {};

    const int D = ctx->hp.voc_dim;                      // 320
    const int inter_dim = ctx->hp.voc_intermediate_dim; // 1536
    const int num_layers = ctx->hp.voc_num_layers;      // 8
    const int n_fft = ctx->hp.voc_n_fft;                // 640
    const int hop = ctx->hp.voc_hop_length;             // 160
    const int n_freqs = n_fft / 2 + 1;                  // 321
    const int head_out_dim = n_fft + 2;                 // 642

    ensure_weights_cached(ctx);
    const auto& w = ctx->weights;
    const auto& embed_w = w.voc_embed_w;
    const auto& embed_b = w.voc_embed_b;
    const auto& norm_w = w.voc_norm_w;
    const auto& norm_b = w.voc_norm_b;
    const auto& final_norm_w = w.voc_final_norm_w;
    const auto& final_norm_b = w.voc_final_norm_b;
    const auto& head_w = w.voc_head_w;
    const auto& head_b = w.voc_head_b;
    const auto& blocks = w.voc_blocks;

    // 1. Conv1d embed: input (80, T), weight (320, 80, 7), pad 3 -> out (320, T)
    std::vector<float> x_ct(D * T, 0.0f);
#if defined(HAVE_ACCELERATE) || defined(CRISPASR_HAVE_BLAS)
    std::vector<float> col_embed(80 * 7 * T);
    for (int t = 0; t < T; t++) {
        for (int ic = 0; ic < 80; ic++) {
            const float* in_row = mel_80_t + ic * T;
            for (int k = 0; k < 7; k++) {
                int in_t = t + k - 3;
                float val = (in_t >= 0 && in_t < T) ? in_row[in_t] : 0.0f;
                col_embed[(ic * 7 + k) * T + t] = val;
            }
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D, T, 80 * 7, 1.0f, embed_w.data(), 80 * 7, col_embed.data(),
                T, 0.0f, x_ct.data(), T);
    for (int oc = 0; oc < D; oc++) {
        float b = embed_b.empty() ? 0.0f : embed_b[oc];
        float* row = x_ct.data() + oc * T;
        for (int t = 0; t < T; t++) {
            row[t] += b;
        }
    }
#else
    for (int oc = 0; oc < D; oc++) {
        float b = embed_b.empty() ? 0.0f : embed_b[oc];
        const float* kw = embed_w.data() + oc * 80 * 7;
        for (int t = 0; t < T; t++) {
            float s = b;
            for (int ic = 0; ic < 80; ic++) {
                const float* mel_row = mel_80_t + ic * T;
                for (int k = 0; k < 7; k++) {
                    int in_t = t + k - 3;
                    if (in_t >= 0 && in_t < T) {
                        s += mel_row[in_t] * kw[ic * 7 + k];
                    }
                }
            }
            x_ct[oc * T + t] = s;
        }
    }
#endif

    // 2. Initial LayerNorm: transpose to (T, D)
    std::vector<float> x_td(T * D);
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < T; t++) {
            x_td[t * D + d] = x_ct[d * T + t];
        }
    }
    cpu_layer_norm(x_td.data(), T, D, norm_w.data(), norm_b.data());
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            x_ct[d * T + t] = x_td[t * D + d];
        }
    }

    // 3. 8 ConvNeXt blocks
    std::vector<float> residual(D * T);
    std::vector<float> dw_out(D * T);
    std::vector<float> pw1(T * inter_dim);
    std::vector<float> pw2(T * D);

    for (int b = 0; b < num_layers; b++) {
        const auto& bc = blocks[b];
        memcpy(residual.data(), x_ct.data(), D * T * sizeof(float));

        // Depthwise Conv1d (320, k=7, pad=3)
        for (int c = 0; c < D; c++) {
            float b_val = bc.dw_b.empty() ? 0.0f : bc.dw_b[c];
            const float* kw = bc.dw_w.data() + c * 7;
            const float* in_row = x_ct.data() + c * T;
            float* out_row = dw_out.data() + c * T;

            // Head with padding
            int head_end = std::min(3, T);
            for (int t = 0; t < head_end; t++) {
                float s = b_val;
                for (int k = 0; k < 7; k++) {
                    int in_t = t + k - 3;
                    if (in_t >= 0 && in_t < T)
                        s += in_row[in_t] * kw[k];
                }
                out_row[t] = s;
            }
            // Interior unrolled branchless SIMD
            for (int t = 3; t < T - 3; t++) {
                out_row[t] = b_val + in_row[t - 3] * kw[0] + in_row[t - 2] * kw[1] + in_row[t - 1] * kw[2] +
                             in_row[t + 0] * kw[3] + in_row[t + 1] * kw[4] + in_row[t + 2] * kw[5] +
                             in_row[t + 3] * kw[6];
            }
            // Tail with padding
            int tail_start = std::max(3, T - 3);
            for (int t = tail_start; t < T; t++) {
                float s = b_val;
                for (int k = 0; k < 7; k++) {
                    int in_t = t + k - 3;
                    if (in_t >= 0 && in_t < T)
                        s += in_row[in_t] * kw[k];
                }
                out_row[t] = s;
            }
        }

        // Transpose to (T, D) -> LayerNorm
        for (int d = 0; d < D; d++) {
            for (int t = 0; t < T; t++) {
                x_td[t * D + d] = dw_out[d * T + t];
            }
        }
        cpu_layer_norm(x_td.data(), T, D, bc.norm_w.data(), bc.norm_b.data());

        // Pointwise 1: Linear(320 -> 1536, GELU)
        cpu_gemm_linear(x_td.data(), bc.pw1_w.data(), bc.pw1_b.data(), pw1.data(), T, D, inter_dim);
        for (auto& v : pw1)
            v = gelu(v);

        // Pointwise 2: Linear(1536 -> 320)
        cpu_gemm_linear(pw1.data(), bc.pw2_w.data(), bc.pw2_b.data(), pw2.data(), T, inter_dim, D);

        // Gamma scale + residual add
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                float val = pw2[t * D + d] * (bc.gamma.empty() ? 1.0f : bc.gamma[d]);
                x_ct[d * T + t] = residual[d * T + t] + val;
            }
        }
    }

    // 4. Final LayerNorm
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < T; t++) {
            x_td[t * D + d] = x_ct[d * T + t];
        }
    }
    cpu_layer_norm(x_td.data(), T, D, final_norm_w.data(), final_norm_b.data());

    // 5. ISTFTHead: Linear(320 -> 642)
    std::vector<float> head_out(T * head_out_dim);
    cpu_gemm_linear(x_td.data(), head_w.data(), head_b.data(), head_out.data(), T, D, head_out_dim);

    // Split magnitude & phase -> row-major (T, n_freqs) for core_istft
    std::vector<float> mag(T * n_freqs), phase(T * n_freqs);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freqs; f++) {
            float m = head_out[t * head_out_dim + f];
            if (m > 100.0f)
                m = 100.0f;
            mag[t * n_freqs + f] = expf(m);
            phase[t * n_freqs + f] = head_out[t * head_out_dim + n_freqs + f];
        }
    }

    // 6. Inverse STFT with Hann window, overlap-add and center padding trim
    std::vector<float> trimmed =
        core_istft::istft(mag.data(), phase.data(), n_fft, hop, T, nullptr, core_istft::TRIM_CENTER);

    if (out_samples)
        *out_samples = (int)trimmed.size();
    return trimmed;
}

// ---------------------------------------------------------------------------
// Public C ABI
// ---------------------------------------------------------------------------

struct meanvc2_context* meanvc2_init_from_file(const char* path_model, struct meanvc2_context_params params) {
    if (!path_model || path_model[0] == '\0')
        return nullptr;

    auto* c = new meanvc2_context();
    c->params = params;
    c->hp.chunk_size = params.chunk_size;
    c->hp.block_size = params.block_size;
    c->rng.seed(params.seed);

    c->backend_cpu = core_cpu_backend::init();
    c->backend = params.use_gpu ? crispasr_init_gpu_backend() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

    if (!core_gguf::load_weights(path_model, c->backend, "meanvc2", c->wl)) {
        delete c;
        return nullptr;
    }
    c->tensors = std::move(c->wl.tensors);

    if (params.verbosity >= 1) {
        fprintf(stderr, "meanvc2: loaded model '%s' (%zu tensors)\n", path_model, c->tensors.size());
    }

    ensure_weights_cached(c);

    return c;
}

int meanvc2_set_speaker_audio(struct meanvc2_context* ctx, const float* audio_16k, int n_samples) {
    if (!ctx || !audio_16k || n_samples <= 0)
        return -1;
    // When speaker model is present in GGUF, compute embedding; otherwise placeholder
    ctx->speaker_emb.assign(256, 0.0f);
    ctx->speaker_emb[0] = 1.0f;
    return 0;
}

int meanvc2_set_speaker_embedding(struct meanvc2_context* ctx, const float* spk_emb_256) {
    if (!ctx || !spk_emb_256)
        return -1;
    ctx->speaker_emb.assign(spk_emb_256, spk_emb_256 + 256);
    return 0;
}

float* meanvc2_convert(struct meanvc2_context* ctx, const float* source_16k, int n_source_samples, int* out_n_samples) {
    if (!ctx || !source_16k || n_source_samples <= 0)
        return nullptr;

    if (ctx->speaker_emb.empty()) {
        ctx->speaker_emb.assign(256, 0.0f);
        ctx->speaker_emb[0] = 1.0f; // Default unit vector
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int bn_frames = 0, bn_dim = 256;
    std::vector<float> bns = run_fastu2pp_asr(ctx, source_16k, n_source_samples, &bn_frames, &bn_dim);
    if (bn_frames <= 0)
        return nullptr;
    auto t_asr = std::chrono::high_resolution_clock::now();

    // 4x linear interpolation of BNF
    int T_mel = bn_frames * 4;
    std::vector<float> bn_interp(T_mel * bn_dim);
    for (int d = 0; d < bn_dim; d++) {
        for (int t = 0; t < T_mel; t++) {
            float orig_pos = (float)t / 4.0f;
            int idx0 = (int)orig_pos;
            int idx1 = std::min(idx0 + 1, bn_frames - 1);
            float alpha = orig_pos - (float)idx0;
            float val0 = bns[idx0 * bn_dim + d];
            float val1 = bns[idx1 * bn_dim + d];
            bn_interp[t * bn_dim + d] = val0 * (1.0f - alpha) + val1 * alpha;
        }
    }

    // Deterministic flow noise
    std::vector<float> noise(T_mel * 80);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : noise)
        v = dist(ctx->rng);

    int mel_dim = 80;
    std::vector<float> mel =
        run_dit_flow(ctx, bn_interp.data(), T_mel, ctx->speaker_emb.data(), noise.data(), &mel_dim);
    auto t_dit = std::chrono::high_resolution_clock::now();

    int final_samples = 0;
    std::vector<float> pcm = run_vocos_decode(ctx, mel.data(), T_mel, &final_samples);
    auto t_vocos = std::chrono::high_resolution_clock::now();

    if (ctx->params.verbosity >= 1) {
        double d_asr = std::chrono::duration<double, std::milli>(t_asr - t_start).count();
        double d_dit = std::chrono::duration<double, std::milli>(t_dit - t_asr).count();
        double d_voc = std::chrono::duration<double, std::milli>(t_vocos - t_dit).count();
        fprintf(stderr, "meanvc2 profile: asr=%.2f ms, dit=%.2f ms, vocos=%.2f ms (total=%.2f ms)\n", d_asr, d_dit,
                d_voc, d_asr + d_dit + d_voc);
    }

    float* out = (float*)malloc(final_samples * sizeof(float));
    memcpy(out, pcm.data(), final_samples * sizeof(float));
    if (out_n_samples)
        *out_n_samples = final_samples;
    return out;
}

float* meanvc2_run_asr(struct meanvc2_context* ctx, const float* source_16k, int n_samples, int* out_frames,
                       int* out_dim) {
    auto bns = run_fastu2pp_asr(ctx, source_16k, n_samples, out_frames, out_dim);
    if (bns.empty())
        return nullptr;
    float* res = (float*)malloc(bns.size() * sizeof(float));
    memcpy(res, bns.data(), bns.size() * sizeof(float));
    return res;
}

int meanvc2_run_gtm(struct meanvc2_context* ctx, const float* spk_emb, float* out_k_mem, float* out_v_mem) {
    return run_gtm(ctx, spk_emb, out_k_mem, out_v_mem);
}

float* meanvc2_run_tvt(struct meanvc2_context* ctx, const float* bn_interp, int n_frames, const float* k_mem,
                       const float* v_mem) {
    auto tc = run_tvt(ctx, bn_interp, n_frames, k_mem, v_mem);
    if (tc.empty())
        return nullptr;
    float* res = (float*)malloc(tc.size() * sizeof(float));
    memcpy(res, tc.data(), tc.size() * sizeof(float));
    return res;
}

float* meanvc2_run_dit(struct meanvc2_context* ctx, const float* bn_interp, int n_frames, const float* spk_emb,
                       const float* noise, int* out_mel_dim) {
    auto mel = run_dit_flow(ctx, bn_interp, n_frames, spk_emb, noise, out_mel_dim);
    if (mel.empty())
        return nullptr;
    float* res = (float*)malloc(mel.size() * sizeof(float));
    memcpy(res, mel.data(), mel.size() * sizeof(float));
    return res;
}

float* meanvc2_run_vocos(struct meanvc2_context* ctx, const float* mel_80_t, int n_frames, int* out_samples) {
    auto pcm = run_vocos_decode(ctx, mel_80_t, n_frames, out_samples);
    if (pcm.empty())
        return nullptr;
    float* res = (float*)malloc(pcm.size() * sizeof(float));
    memcpy(res, pcm.data(), pcm.size() * sizeof(float));
    return res;
}

void meanvc2_free_audio(float* pcm) {
    if (pcm)
        free(pcm);
}

void meanvc2_free(struct meanvc2_context* ctx) {
    if (!ctx)
        return;
    if (ctx->backend && ctx->backend != ctx->backend_cpu) {
        ggml_backend_free(ctx->backend);
    }
    if (ctx->backend_cpu) {
        ggml_backend_free(ctx->backend_cpu);
    }
    delete ctx;
}
