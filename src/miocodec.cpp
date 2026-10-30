// miocodec.cpp — MioCodec v2 decoder implementation.
//
// Decode path: FSQ codebook lookup → wave_prenet Transformer → conv_upsample
// → interp → ResNet prior → wave_decoder Transformer (AdaLN-Zero) → ResNet
// post → SnakeBeta upsampler → ISTFTHead → 44.1 kHz waveform.

#include "miocodec.h"

#include "core/gguf_loader.h"
#include "core/istft.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define CRISPASR_MIOCODEC_BLAS 1
#elif defined(HAVE_BLAS)
#include <cblas.h>
#define CRISPASR_MIOCODEC_BLAS 1
#endif

// ============================================================================
// Model hyperparameters (read from GGUF metadata)
// ============================================================================
struct miocodec_hparams {
    uint32_t sample_rate = 44100;
    uint32_t n_fft = 392;
    uint32_t hop_length = 98;
    uint32_t downsample_factor = 2;

    // FSQ
    std::vector<int> fsq_levels = {8, 8, 8, 5, 5};
    uint32_t fsq_input_dim = 768;
    uint32_t fsq_output_dim = 768;

    // Local encoder (for reference — not used in decode-only)
    uint32_t local_enc_dim = 768;
    uint32_t local_enc_n_layers = 6;

    // Wave prenet
    uint32_t wave_prenet_dim = 768;
    uint32_t wave_prenet_output_dim = 512;
    uint32_t wave_prenet_n_layers = 6;
    uint32_t wave_prenet_n_heads = 12;
    uint32_t wave_prenet_window_size = 65;
    float wave_prenet_rope_theta = 10000.0f;

    // Wave decoder (AdaLN-Zero)
    uint32_t wave_dec_dim = 512;
    uint32_t wave_dec_n_layers = 8;
    uint32_t wave_dec_n_heads = 8;
    uint32_t wave_dec_window_size = 65;
    float wave_dec_rope_theta = 10000.0f;
    uint32_t wave_dec_adaln_cond_dim = 128;

    // Wave misc
    uint32_t wave_decoder_dim = 512;
    uint32_t wave_upsample_factor = 2;
    uint32_t wave_resnet_num_blocks = 2;
    uint32_t wave_resnet_kernel_size = 3;
    uint32_t wave_resnet_num_groups = 32;
    std::vector<int> wave_upsampler_factors = {3, 3};
    std::vector<int> wave_upsampler_kernel_sizes = {9, 9};

    // Global encoder
    uint32_t global_enc_output_channels = 128;
};

// ============================================================================
// Model weights
// ============================================================================
struct miocodec_weights {
    // FSQ
    ggml_tensor* fsq_proj_in_w = nullptr;  // [5, 768]
    ggml_tensor* fsq_proj_in_b = nullptr;  // [5]
    ggml_tensor* fsq_proj_out_w = nullptr; // [768, 5]
    ggml_tensor* fsq_proj_out_b = nullptr; // [768]

    // Wave prenet (6 layers)
    struct transformer_layer {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* attn_norm_b = nullptr;
        ggml_tensor* wq = nullptr;
        ggml_tensor* wk = nullptr;
        ggml_tensor* wv = nullptr;
        ggml_tensor* wo = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* ffn_norm_b = nullptr;
        ggml_tensor* ffn_w1 = nullptr;
        ggml_tensor* ffn_w2 = nullptr;
        ggml_tensor* ffn_w3 = nullptr;
        // AdaLN-Zero (only for wave_decoder layers)
        ggml_tensor* attn_adaln_w = nullptr; // condition_proj[1].weight
        ggml_tensor* attn_adaln_b = nullptr; // condition_proj[1].bias
        ggml_tensor* ffn_adaln_w = nullptr;
        ggml_tensor* ffn_adaln_b = nullptr;
    };

    // Wave prenet
    ggml_tensor* wave_prenet_input_proj_w = nullptr; // input_proj if dim != input_dim
    ggml_tensor* wave_prenet_output_proj_w = nullptr;
    ggml_tensor* wave_prenet_output_proj_b = nullptr;
    ggml_tensor* wave_prenet_norm_w = nullptr;
    ggml_tensor* wave_prenet_norm_b = nullptr;
    std::vector<transformer_layer> wave_prenet_layers;

    // Wave conv upsample
    ggml_tensor* wave_conv_up_w = nullptr; // [512, 512, 2]
    ggml_tensor* wave_conv_up_b = nullptr; // [512]

    // Wave prior/post ResNet blocks
    struct resnet_block {
        ggml_tensor* norm1_w = nullptr;
        ggml_tensor* norm1_b = nullptr;
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* norm2_w = nullptr;
        ggml_tensor* norm2_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
    };
    std::vector<resnet_block> wave_prior_net;
    std::vector<resnet_block> wave_post_net;

    // Wave decoder (8 layers, AdaLN-Zero)
    ggml_tensor* wave_dec_norm_adaln_w = nullptr; // final norm condition_proj
    ggml_tensor* wave_dec_norm_adaln_b = nullptr;
    std::vector<transformer_layer> wave_dec_layers;

    // Wave upsampler
    struct upsampler_stage {
        ggml_tensor* conv_w0 = nullptr; // parametrizations.weight.original0
        ggml_tensor* conv_w1 = nullptr; // parametrizations.weight.original1
        ggml_tensor* conv_b = nullptr;
        ggml_tensor* snake_alpha = nullptr;
        ggml_tensor* snake_beta = nullptr;
        // ResNet block inside upsampler
        resnet_block resblk;
    };
    std::vector<upsampler_stage> wave_upsampler_stages;
    ggml_tensor* wave_upsampler_out_proj_w = nullptr;
    ggml_tensor* wave_upsampler_out_proj_b = nullptr;
    ggml_tensor* wave_upsampler_out_snake_alpha = nullptr;
    ggml_tensor* wave_upsampler_out_snake_beta = nullptr;

    // ISTFT head
    ggml_tensor* istft_out_w = nullptr; // [394, 512]
    ggml_tensor* istft_out_b = nullptr; // [394]
};

// ============================================================================
// Context
// ============================================================================
struct miocodec_context {
    miocodec_hparams hparams;
    miocodec_weights weights;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_w = nullptr;        // weight context
    ggml_backend_sched_t sched = nullptr; // scheduler for graph compute

    int verbosity = 0;

    // Pre-cached dequantized weights for fast CPU paths
    std::vector<float> fsq_proj_out_w_cached;
    std::vector<float> fsq_proj_out_b_cached;
    std::vector<float> istft_out_w_cached;
    std::vector<float> istft_out_b_cached;
};

// ============================================================================
// Public API
// ============================================================================

struct miocodec_params miocodec_default_params(void) {
    miocodec_params p = {};
    p.n_threads = 4;
    p.verbosity = 0;
    p.use_gpu = false;
    return p;
}

// Forward declaration
static std::vector<float> miocodec_dequant_tensor(ggml_tensor* t);

struct miocodec_context* miocodec_init_from_file(const char* path, struct miocodec_params params) {
    auto* ctx = new miocodec_context();
    ctx->verbosity = params.verbosity;

    // Pass 1: read metadata (hyperparameters)
    gguf_context* gctx = core_gguf::open_metadata(path);
    if (!gctx) {
        fprintf(stderr, "miocodec: failed to open '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hparams;
    hp.sample_rate = core_gguf::kv_u32(gctx, "miocodec.sample_rate", 44100);
    hp.n_fft = core_gguf::kv_u32(gctx, "miocodec.n_fft", 392);
    hp.hop_length = core_gguf::kv_u32(gctx, "miocodec.hop_length", 98);
    hp.downsample_factor = core_gguf::kv_u32(gctx, "miocodec.downsample_factor", 2);
    hp.wave_prenet_n_layers = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.n_layers", 6);
    hp.wave_prenet_dim = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.dim", 768);
    hp.wave_prenet_output_dim = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.output_dim", 512);
    hp.wave_prenet_n_heads = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.n_heads", 12);
    hp.wave_prenet_window_size = core_gguf::kv_u32(gctx, "miocodec.wave_prenet.window_size", 65);
    hp.wave_dec_n_layers = core_gguf::kv_u32(gctx, "miocodec.wave_dec.n_layers", 8);
    hp.wave_dec_dim = core_gguf::kv_u32(gctx, "miocodec.wave_dec.dim", 512);
    hp.wave_dec_n_heads = core_gguf::kv_u32(gctx, "miocodec.wave_dec.n_heads", 8);
    hp.wave_dec_adaln_cond_dim = core_gguf::kv_u32(gctx, "miocodec.wave_dec.adaln_cond_dim", 128);
    hp.wave_upsample_factor = core_gguf::kv_u32(gctx, "miocodec.wave_upsample_factor", 2);
    hp.wave_resnet_num_blocks = core_gguf::kv_u32(gctx, "miocodec.wave_resnet_num_blocks", 2);
    hp.wave_decoder_dim = core_gguf::kv_u32(gctx, "miocodec.wave_dec.dim", 512);
    core_gguf::free_metadata(gctx);

    // Pass 2: load weights
    ctx->backend = core_cpu_backend::init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "miocodec", wl)) {
        fprintf(stderr, "miocodec: failed to load weights from '%s'\n", path);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    // Resolve weight tensors
    auto& w = ctx->weights;
    auto T = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(ctx->ctx_w, name); };

    // FSQ
    w.fsq_proj_in_w = T("local_quantizer.proj_in.weight");
    w.fsq_proj_in_b = T("local_quantizer.proj_in.bias");
    w.fsq_proj_out_w = T("local_quantizer.proj_out.weight");
    w.fsq_proj_out_b = T("local_quantizer.proj_out.bias");

    // ISTFT head
    w.istft_out_w = T("istft_head.out.weight");
    w.istft_out_b = T("istft_head.out.bias");

    // Wave conv upsample
    w.wave_conv_up_w = T("wave_conv_upsample.weight");
    w.wave_conv_up_b = T("wave_conv_upsample.bias");

    // Wave prenet layers
    w.wave_prenet_layers.resize(hp.wave_prenet_n_layers);
    w.wave_prenet_norm_w = T("wave_prenet.norm.weight");
    w.wave_prenet_norm_b = T("wave_prenet.norm.bias");
    w.wave_prenet_output_proj_w = T("wave_prenet.output_proj.weight");
    w.wave_prenet_output_proj_b = T("wave_prenet.output_proj.bias");
    for (uint32_t i = 0; i < hp.wave_prenet_n_layers; i++) {
        auto& L = w.wave_prenet_layers[i];
        char buf[256];
        auto N = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_prenet.layers.%u.%s", i, sfx);
            return T(buf);
        };
        L.attn_norm_w = N("attention_norm.weight");
        L.attn_norm_b = N("attention_norm.bias");
        L.wq = N("attention.wq.weight");
        L.wk = N("attention.wk.weight");
        L.wv = N("attention.wv.weight");
        L.wo = N("attention.wo.weight");
        L.ffn_norm_w = N("ffn_norm.weight");
        L.ffn_norm_b = N("ffn_norm.bias");
        L.ffn_w1 = N("feed_forward.w1.weight");
        L.ffn_w2 = N("feed_forward.w2.weight");
        L.ffn_w3 = N("feed_forward.w3.weight");
    }

    // Wave decoder layers (AdaLN-Zero)
    w.wave_dec_layers.resize(hp.wave_dec_n_layers);
    w.wave_dec_norm_adaln_w = T("wave_decoder.norm.condition_proj.1.weight");
    w.wave_dec_norm_adaln_b = T("wave_decoder.norm.condition_proj.1.bias");
    for (uint32_t i = 0; i < hp.wave_dec_n_layers; i++) {
        auto& L = w.wave_dec_layers[i];
        char buf[256];
        auto N = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_decoder.layers.%u.%s", i, sfx);
            return T(buf);
        };
        L.wq = N("attention.wq.weight");
        L.wk = N("attention.wk.weight");
        L.wv = N("attention.wv.weight");
        L.wo = N("attention.wo.weight");
        L.ffn_w1 = N("feed_forward.w1.weight");
        L.ffn_w2 = N("feed_forward.w2.weight");
        L.ffn_w3 = N("feed_forward.w3.weight");
        L.attn_adaln_w = N("attention_norm.condition_proj.1.weight");
        L.attn_adaln_b = N("attention_norm.condition_proj.1.bias");
        L.ffn_adaln_w = N("ffn_norm.condition_proj.1.weight");
        L.ffn_adaln_b = N("ffn_norm.condition_proj.1.bias");
    }

    // ResNet blocks (prior + post)
    w.wave_prior_net.resize(hp.wave_resnet_num_blocks);
    w.wave_post_net.resize(hp.wave_resnet_num_blocks);
    for (uint32_t i = 0; i < hp.wave_resnet_num_blocks; i++) {
        char buf[256];
        auto PN = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_prior_net.blocks.%u.%s", i, sfx);
            return T(buf);
        };
        auto& pb = w.wave_prior_net[i];
        pb.norm1_w = PN("norm1.weight");
        pb.norm1_b = PN("norm1.bias");
        pb.conv1_w = PN("conv1.weight");
        pb.conv1_b = PN("conv1.bias");
        pb.norm2_w = PN("norm2.weight");
        pb.norm2_b = PN("norm2.bias");
        pb.conv2_w = PN("conv2.weight");
        pb.conv2_b = PN("conv2.bias");

        auto QN = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_post_net.blocks.%u.%s", i, sfx);
            return T(buf);
        };
        auto& qb = w.wave_post_net[i];
        qb.norm1_w = QN("norm1.weight");
        qb.norm1_b = QN("norm1.bias");
        qb.conv1_w = QN("conv1.weight");
        qb.conv1_b = QN("conv1.bias");
        qb.norm2_w = QN("norm2.weight");
        qb.norm2_b = QN("norm2.bias");
        qb.conv2_w = QN("conv2.weight");
        qb.conv2_b = QN("conv2.bias");
    }

    // Upsampler stages
    int n_up_stages = (int)hp.wave_upsampler_factors.size();
    if (n_up_stages == 0)
        n_up_stages = 2; // default [3,3]
    w.wave_upsampler_stages.resize(n_up_stages);
    for (int i = 0; i < n_up_stages; i++) {
        char buf[256];
        auto UL = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_upsampler.upsample_layers.%d.%s", i, sfx);
            return T(buf);
        };
        auto& s = w.wave_upsampler_stages[i];
        s.conv_w0 = UL("parametrizations.weight.original0");
        s.conv_w1 = UL("parametrizations.weight.original1");
        s.conv_b = UL("bias");
        snprintf(buf, sizeof(buf), "wave_upsampler.snake_activations.%d.alpha", i);
        s.snake_alpha = T(buf);
        snprintf(buf, sizeof(buf), "wave_upsampler.snake_activations.%d.beta", i);
        s.snake_beta = T(buf);
        // ResNet block inside upsampler
        auto UR = [&](const char* sfx) {
            snprintf(buf, sizeof(buf), "wave_upsampler.resnet_blocks.%d.%s", i, sfx);
            return T(buf);
        };
        s.resblk.norm1_w = UR("norm1.weight");
        s.resblk.norm1_b = UR("norm1.bias");
        s.resblk.conv1_w = UR("conv1.weight");
        s.resblk.conv1_b = UR("conv1.bias");
        s.resblk.norm2_w = UR("norm2.weight");
        s.resblk.norm2_b = UR("norm2.bias");
        s.resblk.conv2_w = UR("conv2.weight");
        s.resblk.conv2_b = UR("conv2.bias");
    }
    w.wave_upsampler_out_proj_w = T("wave_upsampler.out_proj.weight");
    w.wave_upsampler_out_proj_b = T("wave_upsampler.out_proj.bias");
    w.wave_upsampler_out_snake_alpha = T("wave_upsampler.out_snake.alpha");
    w.wave_upsampler_out_snake_beta = T("wave_upsampler.out_snake.beta");

    // Precompute weight-norm for upsampler ConvTranspose1d weights.
    // w = g * v / ||v|| where g=original0 (C_in, 1, 1), v=original1 (C_in, C_out, K)
    // norm is per-output-filter (over C_out*K), applied along ne[2] (C_in) of the ggml tensor.
    for (auto& us : w.wave_upsampler_stages) {
        if (!us.conv_w0 || !us.conv_w1)
            continue;
        // Read g and v from backend
        std::vector<float> g_data = miocodec_dequant_tensor(us.conv_w0);
        std::vector<float> v_data = miocodec_dequant_tensor(us.conv_w1);
        int n_v = (int)v_data.size();
        // v shape in ggml: ne[0]=K, ne[1]=C_out, ne[2]=C_in (from GGUF [K, C_out, C_in])
        int K_dim = (int)us.conv_w1->ne[0];
        int C_out = (int)us.conv_w1->ne[1];
        int C_in = (int)us.conv_w1->ne[2];
        // Weight norm: for each filter i (C_in dimension), compute ||v_i|| over (C_out * K)
        // Then w_i = g_i * v_i / ||v_i||
        std::vector<float> w_fused(n_v);
        for (int ci = 0; ci < C_in; ci++) {
            float g_val = g_data[ci]; // g is (C_in, 1, 1) → one value per input channel
            // Compute ||v[ci]|| over C_out*K elements
            float norm_sq = 0;
            for (int co = 0; co < C_out; co++)
                for (int k = 0; k < K_dim; k++)
                    norm_sq +=
                        v_data[ci * C_out * K_dim + co * K_dim + k] * v_data[ci * C_out * K_dim + co * K_dim + k];
            float norm_inv = 1.0f / (sqrtf(norm_sq) + 1e-12f);
            for (int co = 0; co < C_out; co++)
                for (int k = 0; k < K_dim; k++)
                    w_fused[ci * C_out * K_dim + co * K_dim + k] =
                        g_val * v_data[ci * C_out * K_dim + co * K_dim + k] * norm_inv;
        }
        // Write back fused weight to conv_w1 tensor (overwrite v with w)
        if (us.conv_w1->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(n_v);
            for (int i = 0; i < n_v; i++)
                tmp[i] = ggml_fp32_to_fp16(w_fused[i]);
            ggml_backend_tensor_set(us.conv_w1, tmp.data(), 0, n_v * sizeof(uint16_t));
        } else if (us.conv_w1->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(us.conv_w1, w_fused.data(), 0, n_v * sizeof(float));
        }
        // For quantized types: skip write-back (weight_norm is approximate anyway)
        // TODO: fuse weight_norm in the converter instead
    }

    // Create scheduler for graph compute (handles weight buffer + compute buffer)
    ggml_backend_t backends[] = {ctx->backend};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, 1, 8192, false, false);

    // Pre-cache dequantized weights for fast CPU stages
    if (ctx->weights.fsq_proj_out_w)
        ctx->fsq_proj_out_w_cached = miocodec_dequant_tensor(ctx->weights.fsq_proj_out_w);
    if (ctx->weights.fsq_proj_out_b)
        ctx->fsq_proj_out_b_cached = miocodec_dequant_tensor(ctx->weights.fsq_proj_out_b);
    if (ctx->weights.istft_out_w)
        ctx->istft_out_w_cached = miocodec_dequant_tensor(ctx->weights.istft_out_w);
    if (ctx->weights.istft_out_b)
        ctx->istft_out_b_cached = miocodec_dequant_tensor(ctx->weights.istft_out_b);

    if (params.verbosity > 0) {
        fprintf(stderr, "miocodec: loaded model from '%s'\n", path);
        fprintf(stderr, "  sample_rate=%u, n_fft=%u, hop=%u\n", hp.sample_rate, hp.n_fft, hp.hop_length);
        fprintf(stderr, "  wave_prenet: %uL %ud, wave_decoder: %uL %ud (adaln=%u)\n", hp.wave_prenet_n_layers,
                hp.wave_prenet_dim, hp.wave_dec_n_layers, hp.wave_dec_dim, hp.wave_dec_adaln_cond_dim);
    }

    return ctx;
}

// Dequantize a ggml tensor to float32 (handles F32, F16, and quantized types)
static std::vector<float> miocodec_dequant_tensor(ggml_tensor* t) {
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(uint16_t));
        for (int i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // Quantized: read raw bytes then dequantize row-by-row
        size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> raw(nb);
        ggml_backend_tensor_get(t, raw.data(), 0, nb);
        int64_t ne0 = t->ne[0]; // row width
        int64_t nrows = n / ne0;
        for (int64_t r = 0; r < nrows; r++) {
            const void* src = raw.data() + r * t->nb[1];
            const ggml_type_traits* traits = ggml_get_type_traits(t->type);
            if (traits && traits->to_float)
                traits->to_float(src, out.data() + r * ne0, ne0);
        }
    }
    return out;
}

// SnakeBeta activation: x + (1/exp(β)) * sin²(exp(α) * x)
// alpha, beta are (C) 1D tensors (log-scale). x is (C, T).
// Broadcasting: alpha/beta (C) broadcasts against (C, T) since ne[0]=C matches.
static ggml_tensor* miocodec_snake_beta(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta) {
    ggml_tensor* a = ggml_exp(ctx0, alpha); // exp(alpha), shape (C)
    ggml_tensor* b = ggml_exp(ctx0, beta);  // exp(beta), shape (C)
    // sin²(a * x): a broadcasts (C) against x (C, T)
    ggml_tensor* ax = ggml_mul(ctx0, x, a);               // (C, T) — a broadcasts
    ggml_tensor* s2 = ggml_sqr(ctx0, ggml_sin(ctx0, ax)); // sin²(a*x), (C, T)
    // (1/b) * s2: b broadcasts (C) against s2 (C, T)
    // Need 1/b — use ggml_div? No ggml_div. Use: s2 * (1/b).
    // 1/b = ggml_scale(ones, 0) + ... no. Compute reciprocal CPU-side? Too complex.
    // Actually: ggml doesn't have element-wise reciprocal or div.
    // Workaround: precompute 1/exp(beta) = exp(-beta) at load time.
    // For now, use the approximation: exp(-beta) ≈ 1/exp(beta)
    ggml_tensor* inv_b = ggml_exp(ctx0, ggml_neg(ctx0, beta)); // exp(-beta) = 1/exp(beta)
    ggml_tensor* scaled = ggml_mul(ctx0, s2, inv_b);           // (C, T)
    return ggml_add(ctx0, x, scaled);
}

void miocodec_free(struct miocodec_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

uint32_t miocodec_sample_rate(const struct miocodec_context* ctx) {
    return ctx ? ctx->hparams.sample_rate : 44100;
}
uint32_t miocodec_n_fft(const struct miocodec_context* ctx) {
    return ctx ? ctx->hparams.n_fft : 392;
}
uint32_t miocodec_hop_length(const struct miocodec_context* ctx) {
    return ctx ? ctx->hparams.hop_length : 98;
}
uint32_t miocodec_codebook_size(const struct miocodec_context* ctx) {
    (void)ctx;
    return 12800; // product of FSQ levels [8,8,8,5,5]
}
uint32_t miocodec_token_rate(const struct miocodec_context* ctx) {
    (void)ctx;
    return 25;
}

// ============================================================================
// FSQ Decode: token indices → (T, 768) embeddings
// ============================================================================
// FSQ levels = [8, 8, 8, 5, 5], basis = [1, 8, 64, 512, 2560]
// indices_to_codes: index → per-dim codes → normalize to [-1, 1]
// Then proj_out(codes) → 768-dim embeddings

static void fsq_indices_to_codes(const int32_t* indices, int n, float* out_codes) {
    // levels = [8, 8, 8, 5, 5], basis = [1, 8, 64, 512, 2560]
    static const int levels[5] = {8, 8, 8, 5, 5};
    static const int basis[5] = {1, 8, 64, 512, 2560};
    static const int half_width[5] = {4, 4, 4, 2, 2}; // levels // 2

    for (int t = 0; t < n; t++) {
        int idx = indices[t];
        for (int d = 0; d < 5; d++) {
            int code_raw = (idx / basis[d]) % levels[d];
            // _scale_and_shift_inverse: (code - half_width) / half_width
            out_codes[t * 5 + d] = (float)(code_raw - half_width[d]) / (float)half_width[d];
        }
    }
}

// ============================================================================
// ggml graph helpers for the decode pipeline
// ============================================================================

// Build a single Transformer layer (no AdaLN): LN → Attn → residual → LN → FFN → residual
static ggml_tensor* miocodec_transformer_layer(ggml_context* ctx0, ggml_tensor* x,
                                               const miocodec_weights::transformer_layer& L, int n_heads,
                                               ggml_tensor* rope_pos, ggml_tensor* attn_mask = nullptr) {
    const int64_t dim = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t hd = dim / n_heads;

    // Attention norm (LayerNorm) — repeat weight/bias to (dim, 1) for broadcast
    ggml_tensor* attn_in = ggml_norm(ctx0, x, 1e-5f);
    attn_in = ggml_mul(ctx0, attn_in, L.attn_norm_w);
    attn_in = ggml_add(ctx0, attn_in, L.attn_norm_b);

    // QKV projections
    ggml_tensor* Q = ggml_mul_mat(ctx0, L.wq, attn_in); // (dim, T)
    ggml_tensor* K = ggml_mul_mat(ctx0, L.wk, attn_in); // (dim, T)
    ggml_tensor* V = ggml_mul_mat(ctx0, L.wv, attn_in); // (dim, T)

    // Reshape to (hd, n_heads, T) for attention
    Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_heads, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_heads, T);

    // RoPE on (hd, n_heads, T) — ne[2]=T matches positions length
    Q = ggml_rope_ext(ctx0, Q, rope_pos, nullptr, (int)hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    K = ggml_rope_ext(ctx0, K, rope_pos, nullptr, (int)hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Manual attention: Q,K,V are (hd, n_heads, T) after RoPE
    // Permute to (hd, T, n_heads) for mul_mat
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, n_heads)
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    float scale = 1.0f / sqrtf((float)hd);
    // scores = Q @ K^T → (T, T, n_heads)
    ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
    scores = ggml_scale(ctx0, scores, scale);

    // Apply windowed mask if provided
    if (attn_mask) {
        scores = ggml_add(ctx0, scores, attn_mask);
    }

    // Softmax over ne[0] (T_k dimension)
    scores = ggml_soft_max(ctx0, scores);

    // attn_out = scores @ V^T → V is (hd, T, n_heads), need V^T = (T, hd, n_heads)
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // (T, hd, n_heads)
    ggml_tensor* attn_out = ggml_mul_mat(ctx0, V, scores);  // (hd, T, n_heads)

    // Back to (dim, T)
    attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), dim, T);

    // Output projection + residual
    attn_out = ggml_mul_mat(ctx0, L.wo, attn_out);
    ggml_tensor* h = ggml_add(ctx0, x, attn_out);

    // FFN norm
    ggml_tensor* ffn_in = ggml_norm(ctx0, h, 1e-5f);
    ffn_in = ggml_mul(ctx0, ffn_in, L.ffn_norm_w);
    ffn_in = ggml_add(ctx0, ffn_in, L.ffn_norm_b);

    // SwiGLU: w2(silu(w1(x)) * w3(x))
    ggml_tensor* gate = ggml_silu(ctx0, ggml_mul_mat(ctx0, L.ffn_w1, ffn_in));
    ggml_tensor* up = ggml_mul_mat(ctx0, L.ffn_w3, ffn_in);
    ggml_tensor* ffn_out = ggml_mul_mat(ctx0, L.ffn_w2, ggml_mul(ctx0, gate, up));

    return ggml_add(ctx0, h, ffn_out);
}

// AdaLN-Zero modulate: norm(x) * (1 + scale) + shift, returning (modulated, gate)
// adaln_w/adaln_b produce 3*dim output (shift, scale, gate).
// cond shape: (cond_dim, T) — typically (128, T) broadcast from (128, 1).
static std::pair<ggml_tensor*, ggml_tensor*> miocodec_adaln_modulate(ggml_context* ctx0, ggml_tensor* x,
                                                                     ggml_tensor* cond, ggml_tensor* adaln_w,
                                                                     ggml_tensor* adaln_b, bool return_gate) {
    const int64_t dim = x->ne[0];
    // condition_proj: SiLU(cond) → linear → (3*dim or 2*dim, T)
    ggml_tensor* params = ggml_mul_mat(ctx0, adaln_w, ggml_silu(ctx0, cond));
    params = ggml_add(ctx0, params, adaln_b);
    // params shape: (3*dim, T) for return_gate=true, (2*dim, T) for false
    // Split: use ggml_view_2d (params is a computed tensor in ctx0, so view works)
    const int64_t T = params->ne[1];
    const size_t nb1 = params->nb[1]; // stride between rows = ne[0] * sizeof(float)
    ggml_tensor* shift = ggml_view_2d(ctx0, params, dim, T, nb1, 0);
    ggml_tensor* scale = ggml_view_2d(ctx0, params, dim, T, nb1, dim * ggml_type_size(params->type));
    ggml_tensor* gate_t = nullptr;
    if (return_gate)
        gate_t = ggml_view_2d(ctx0, params, dim, T, nb1, 2 * dim * ggml_type_size(params->type));

    // modulated = norm(x) * (1 + scale) + shift = x_norm + x_norm*scale + shift
    ggml_tensor* x_norm = ggml_norm(ctx0, x, 1e-5f);
    ggml_tensor* modulated = ggml_add(ctx0, ggml_add(ctx0, x_norm, ggml_mul(ctx0, x_norm, scale)), shift);
    return {modulated, gate_t};
    return {modulated, gate_t};
}

// AdaLN-Zero Transformer layer (for wave_decoder)
static ggml_tensor* miocodec_adaln_transformer_layer(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* cond,
                                                     const miocodec_weights::transformer_layer& L, int n_heads,
                                                     ggml_tensor* rope_pos, ggml_tensor* attn_mask) {
    const int64_t dim = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t hd = dim / n_heads;

    // AdaLN-Zero for attention
    auto [attn_in, attn_gate] = miocodec_adaln_modulate(ctx0, x, cond, L.attn_adaln_w, L.attn_adaln_b, true);

    // Manual attention (same as non-AdaLN version)
    ggml_tensor* Q = ggml_mul_mat(ctx0, L.wq, attn_in);
    ggml_tensor* K = ggml_mul_mat(ctx0, L.wk, attn_in);
    ggml_tensor* V = ggml_mul_mat(ctx0, L.wv, attn_in);

    Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_heads, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_heads, T);

    Q = ggml_rope_ext(ctx0, Q, rope_pos, nullptr, (int)hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    K = ggml_rope_ext(ctx0, K, rope_pos, nullptr, (int)hd, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    float scale = 1.0f / sqrtf((float)hd);
    ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
    scores = ggml_scale(ctx0, scores, scale);
    if (attn_mask)
        scores = ggml_add(ctx0, scores, attn_mask);
    scores = ggml_soft_max(ctx0, scores);

    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
    ggml_tensor* attn_out = ggml_mul_mat(ctx0, V, scores);
    attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), dim, T);
    attn_out = ggml_mul_mat(ctx0, L.wo, attn_out);

    // Residual with gate
    ggml_tensor* h = ggml_add(ctx0, x, ggml_mul(ctx0, attn_out, attn_gate));

    // FFN AdaLN-Zero
    auto [ffn_in, ffn_gate] = miocodec_adaln_modulate(ctx0, h, cond, L.ffn_adaln_w, L.ffn_adaln_b, true);

    ggml_tensor* gate2 = ggml_silu(ctx0, ggml_mul_mat(ctx0, L.ffn_w1, ffn_in));
    ggml_tensor* up2 = ggml_mul_mat(ctx0, L.ffn_w3, ffn_in);
    ggml_tensor* ffn_out = ggml_mul_mat(ctx0, L.ffn_w2, ggml_mul(ctx0, gate2, up2));

    return ggml_add(ctx0, h, ggml_mul(ctx0, ffn_out, ffn_gate));
}
static ggml_tensor* miocodec_resnet_block(ggml_context* ctx0, ggml_tensor* x, const miocodec_weights::resnet_block& b,
                                          int n_groups) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    ggml_tensor* residual = x;
    ggml_tensor* xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    xt = ggml_group_norm(ctx0, ggml_reshape_3d(ctx0, xt, T, 1, C), n_groups, 1e-6f);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, xt, T, C)));
    x = ggml_mul(ctx0, x, b.norm1_w);
    x = ggml_add(ctx0, x, b.norm1_b);
    x = ggml_silu(ctx0, x);
    xt = ggml_conv_1d(ctx0, b.conv1_w, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 1, 1, 1);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, xt, T, C)));
    x = ggml_add(ctx0, x, b.conv1_b);
    xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    xt = ggml_group_norm(ctx0, ggml_reshape_3d(ctx0, xt, T, 1, C), n_groups, 1e-6f);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, xt, T, C)));
    x = ggml_mul(ctx0, x, b.norm2_w);
    x = ggml_add(ctx0, x, b.norm2_b);
    x = ggml_silu(ctx0, x);
    xt = ggml_conv_1d(ctx0, b.conv2_w, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 1, 1, 1);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, xt, T, C)));
    x = ggml_add(ctx0, x, b.conv2_b);
    return ggml_add(ctx0, residual, x);
}

float* miocodec_decode(struct miocodec_context* ctx, const int32_t* token_indices, int n_tokens,
                       const float* global_embedding, int target_audio_length, int* out_n_samples) {
    if (!ctx || !token_indices || n_tokens <= 0 || !global_embedding || !out_n_samples)
        return nullptr;

    return miocodec_extract_stage(ctx, token_indices, n_tokens, global_embedding, target_audio_length,
                                  "output_waveform", out_n_samples);
}

float* miocodec_extract_stage(struct miocodec_context* ctx, const int32_t* token_indices, int n_tokens,
                              const float* global_embedding, int target_audio_length, const char* stage_name,
                              int* out_n) {
    if (!ctx || !token_indices || n_tokens <= 0 || !out_n || !stage_name)
        return nullptr;

    (void)global_embedding;
    (void)target_audio_length;

    // Stage: fsq_decoded — pure CPU computation, no ggml graph needed
    if (strcmp(stage_name, "fsq_decoded") == 0) {
        // Step 1: indices → 5-dim codes
        std::vector<float> codes(n_tokens * 5);
        fsq_indices_to_codes(token_indices, n_tokens, codes.data());

        // Step 2: proj_out(codes) → 768-dim
        // proj_out: Linear(5, 768) → out = codes @ W^T + bias
        const int out_dim = 768;
        float* result = (float*)malloc(sizeof(float) * n_tokens * out_dim);
        if (!result)
            return nullptr;

        // Get weight data (cached F32 or dequantize fallback)
        const float* proj_w = ctx->fsq_proj_out_w_cached.empty() ? nullptr : ctx->fsq_proj_out_w_cached.data();
        const float* proj_b = ctx->fsq_proj_out_b_cached.empty() ? nullptr : ctx->fsq_proj_out_b_cached.data();
        std::vector<float> pw_fb, pb_fb;
        if (!proj_w) {
            pw_fb = miocodec_dequant_tensor(ctx->weights.fsq_proj_out_w);
            proj_w = pw_fb.data();
        }
        if (!proj_b) {
            pb_fb = miocodec_dequant_tensor(ctx->weights.fsq_proj_out_b);
            proj_b = pb_fb.data();
        }

        // result[t, d] = sum_k(codes[t, k] * W[d, k]) + bias[d]
        // GGUF shape is [5, 768] meaning ne[0]=5, ne[1]=768.
        // Memory layout: element at (d, k) is proj_w[d * 5 + k].
        for (int t = 0; t < n_tokens; t++) {
            for (int d = 0; d < out_dim; d++) {
                float sum = proj_b[d];
                for (int k = 0; k < 5; k++) {
                    sum += codes[t * 5 + k] * proj_w[d * 5 + k];
                }
                result[t * out_dim + d] = sum;
            }
        }

        *out_n = n_tokens * out_dim;
        return result;
    }

    // ---- wave_prenet_out: run the 6-layer prenet Transformer ----
    if (strcmp(stage_name, "wave_prenet_out") == 0) {
        // First compute FSQ embeddings (input to prenet)
        std::vector<float> codes(n_tokens * 5);
        fsq_indices_to_codes(token_indices, n_tokens, codes.data());
        std::vector<float> fsq_emb(n_tokens * 768);
        {
            const float* proj_w = ctx->fsq_proj_out_w_cached.empty() ? nullptr : ctx->fsq_proj_out_w_cached.data();
            const float* proj_b = ctx->fsq_proj_out_b_cached.empty() ? nullptr : ctx->fsq_proj_out_b_cached.data();
            std::vector<float> pw_fb, pb_fb;
            if (!proj_w) {
                pw_fb = miocodec_dequant_tensor(ctx->weights.fsq_proj_out_w);
                proj_w = pw_fb.data();
            }
            if (!proj_b) {
                pb_fb = miocodec_dequant_tensor(ctx->weights.fsq_proj_out_b);
                proj_b = pb_fb.data();
            }
            for (int t = 0; t < n_tokens; t++) {
                for (int d = 0; d < 768; d++) {
                    float sum = proj_b[d];
                    for (int k = 0; k < 5; k++)
                        sum += codes[t * 5 + k] * proj_w[d * 5 + k];
                    fsq_emb[t * 768 + d] = sum;
                }
            }
        }

        // Build ggml graph for wave_prenet: 6L Transformer(768→512)
        const int T = n_tokens;
        const int dim = 768;
        const int out_d = 512;
        const int n_heads = (int)ctx->hparams.wave_prenet_n_heads;
        const int n_layers = (int)ctx->hparams.wave_prenet_n_layers;

        // Null-check all prenet weight tensors
        for (int i = 0; i < n_layers; i++) {
            auto& L = ctx->weights.wave_prenet_layers[i];
            if (!L.attn_norm_w || !L.attn_norm_b || !L.wq || !L.wk || !L.wv || !L.wo || !L.ffn_norm_w ||
                !L.ffn_norm_b || !L.ffn_w1 || !L.ffn_w2 || !L.ffn_w3) {
                fprintf(stderr, "miocodec: wave_prenet layer %d has null tensor(s):\n", i);
                *out_n = 0;
                return nullptr;
            }
        }
        if (!ctx->weights.wave_prenet_norm_w || !ctx->weights.wave_prenet_norm_b ||
            !ctx->weights.wave_prenet_output_proj_w) {
            *out_n = 0;
            return nullptr;
        }

        size_t buf_size =
            (size_t)(n_layers * 100 + 500) * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false);
        ggml_init_params ip = {buf_size, nullptr, true};
        ggml_context* ctx0 = ggml_init(ip);
        if (!ctx0) {
            *out_n = 0;
            return nullptr;
        }

        // Input tensor
        ggml_tensor* input_x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, T);
        ggml_set_name(input_x, "prenet_input");
        ggml_set_input(input_x);
        ggml_tensor* x = input_x;

        // RoPE positions (shared across all layers)
        ggml_tensor* rope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(rope_pos, "rope_pos");
        ggml_set_input(rope_pos);

        // Build windowed attention mask (window_size=65, ±32 positions)
        const int window_per_side = (int)ctx->hparams.wave_prenet_window_size / 2; // 32
        ggml_tensor* win_mask = nullptr;
        if (T > 1 && window_per_side > 0) {
            win_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
            ggml_set_name(win_mask, "win_mask");
            ggml_set_input(win_mask);
        }

        // Build the 6-layer Transformer graph
        for (int i = 0; i < n_layers; i++) {
            x = miocodec_transformer_layer(ctx0, x, ctx->weights.wave_prenet_layers[i], n_heads, rope_pos, win_mask);
        }

        // Final norm + output projection (768→512)
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, ctx->weights.wave_prenet_norm_w);
        x = ggml_add(ctx0, x, ctx->weights.wave_prenet_norm_b);
        x = ggml_mul_mat(ctx0, ctx->weights.wave_prenet_output_proj_w, x);
        if (ctx->weights.wave_prenet_output_proj_b)
            x = ggml_add(ctx0, x, ctx->weights.wave_prenet_output_proj_b);
        ggml_set_name(x, "wave_prenet_out");
        ggml_set_output(x);

        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);
        ggml_build_forward_expand(gf, x);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "  alloc failed\n");
            ggml_free(ctx0);
            *out_n = 0;
            return nullptr;
        }

        // Set inputs using direct tensor pointers
        ggml_backend_tensor_set(input_x, fsq_emb.data(), 0, sizeof(float) * T * dim);

        // Set RoPE positions [0, 1, 2, ..., T-1]
        {
            std::vector<int32_t> positions(T);
            for (int i = 0; i < T; i++)
                positions[i] = i;
            ggml_backend_tensor_set(rope_pos, positions.data(), 0, sizeof(int32_t) * T);
        }

        // Set windowed attention mask (F32, additive: 0=attend, -inf=mask)
        if (win_mask) {
            const int wps = window_per_side;
            std::vector<float> mask_data((size_t)T * T, -INFINITY);
            for (int q = 0; q < T; q++) {
                int k_start = std::max(0, q - wps);
                int k_end = std::min(T, q + wps + 1);
                if (k_end > k_start)
                    std::fill_n(mask_data.data() + (size_t)q * T + k_start, k_end - k_start, 0.0f);
            }
            ggml_backend_tensor_set(win_mask, mask_data.data(), 0, sizeof(float) * T * T);
        }

        ggml_backend_sched_graph_compute(ctx->sched, gf);

        ggml_tensor* out_t = ggml_graph_get_tensor(gf, "wave_prenet_out");
        int n_out = (int)(out_t->ne[0] * out_t->ne[1]);
        float* result = (float*)malloc(sizeof(float) * n_out);
        ggml_backend_tensor_get(out_t, result, 0, sizeof(float) * n_out);
        ggml_free(ctx0);
        *out_n = n_out;
        return result;
    }

    // ---- Full decode pipeline (wave_prior_net through output_waveform) ----
    // These stages build on the prenet output and run through the rest of the decoder.
    bool need_full_decode =
        (strcmp(stage_name, "wave_prior_net_out") == 0 || strcmp(stage_name, "wave_decoder_out") == 0 ||
         strcmp(stage_name, "wave_post_net_out") == 0 || strcmp(stage_name, "wave_upsampler_out") == 0 ||
         strcmp(stage_name, "istft_mag_phase") == 0 || strcmp(stage_name, "output_waveform") == 0);

    if (need_full_decode) {
        // First get the prenet output (reuse the existing implementation)
        int prenet_n = 0;
        float* prenet_data = miocodec_extract_stage(ctx, token_indices, n_tokens, global_embedding, target_audio_length,
                                                    "wave_prenet_out", &prenet_n);
        if (!prenet_data) {
            *out_n = 0;
            return nullptr;
        }

        // Prenet output is (512, T) in ggml layout
        const int T_pre = n_tokens;                                         // prenet tokens
        const int dim_pre = (int)ctx->hparams.wave_prenet_output_dim;       // 512
        const int hop = (int)ctx->hparams.hop_length;                       // 98
        const int n_fft = (int)ctx->hparams.n_fft;                          // 392
        const int upsample_factor = (int)ctx->hparams.wave_upsample_factor; // 2

        // Compute target lengths
        // After conv_upsample(2×): T_up = T_pre * 2
        const int T_up = T_pre * upsample_factor;
        // Target STFT length (before upsampler): audio_length / hop / total_upsampler_factor
        int total_up_factor = 1;
        for (int f : ctx->hparams.wave_upsampler_factors)
            total_up_factor *= f;
        // If target_audio_length not specified, estimate from tokens
        int audio_len = target_audio_length > 0 ? target_audio_length : (int)(n_tokens * ctx->hparams.sample_rate / 25);
        const int T_stft = audio_len / hop / total_up_factor; // ~199 for 99 tokens
        const int T_istft = T_stft * total_up_factor;         // ~1791

        // Build the full decode graph: prenet_out → conv_up → interp → prior → decoder → post → upsampler → istft
        // Use a second ggml graph with ggml_backend_sched
        const int n_dec_layers = (int)ctx->hparams.wave_dec_n_layers;
        const int dec_n_heads = (int)ctx->hparams.wave_dec_n_heads;
        const int n_resnet = (int)ctx->hparams.wave_resnet_num_blocks;
        const int n_groups = (int)ctx->hparams.wave_resnet_num_groups;
        const int dec_window = (int)ctx->hparams.wave_dec_window_size;
        const int dec_window_ps = dec_window / 2;

        // Allocate graph context (large: decoder has 8 layers × ~30 ops each + resnet + upsampler)
        size_t buf2 = (size_t)(n_dec_layers * 100 + n_resnet * 50 + 500) * ggml_tensor_overhead() +
                      ggml_graph_overhead_custom(8192, false);
        ggml_init_params ip2 = {buf2, nullptr, true};
        ggml_context* ctx0 = ggml_init(ip2);
        if (!ctx0) {
            free(prenet_data);
            *out_n = 0;
            return nullptr;
        }

        // Input: prenet output (dim_pre=512, T_pre)
        ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim_pre, T_pre);
        ggml_set_name(x, "decode_input");
        ggml_set_input(x);

        // Global embedding input (128, 1) — for AdaLN conditioning
        ggml_tensor* global_cond = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 128, 1);
        ggml_set_name(global_cond, "global_cond");
        ggml_set_input(global_cond);

        // RoPE positions for decoder
        ggml_tensor* dec_rope_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_stft);
        ggml_set_name(dec_rope_pos, "dec_rope_pos");
        ggml_set_input(dec_rope_pos);

        // Decoder window mask (if T_stft > window)
        ggml_tensor* dec_mask = nullptr;
        if (T_stft > dec_window && dec_window_ps > 0) {
            dec_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_stft, T_stft);
            ggml_set_name(dec_mask, "dec_mask");
            ggml_set_input(dec_mask);
        }

        // Step 1: ConvTranspose1d — (C=512,T) → transpose → conv → reshape → transpose back
        ggml_tensor* xt2 = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        xt2 = ggml_conv_transpose_1d(ctx0, ctx->weights.wave_conv_up_w, xt2, 2, 0, 1);
        xt2 = ggml_reshape_2d(ctx0, xt2, T_up, dim_pre);
        ggml_tensor* x_up = ggml_cont(ctx0, ggml_transpose(ctx0, xt2));
        x_up = ggml_add(ctx0, x_up, ctx->weights.wave_conv_up_b);

        // Step 2: Interpolate T_up → T_stft
        if (T_up != T_stft) {
            x_up = ggml_interpolate(ctx0, x_up, dim_pre, T_stft, 1, 1, GGML_SCALE_MODE_BILINEAR);
            x_up = ggml_reshape_2d(ctx0, x_up, dim_pre, T_stft);
        }
        // Step 3: Prior_net ResNet — now takes (C=512, T_stft) input
        for (int i = 0; i < n_resnet; i++)
            x_up = miocodec_resnet_block(ctx0, x_up, ctx->weights.wave_prior_net[i], n_groups);
        ggml_set_name(x_up, "wave_prior_net_out");
        ggml_set_output(x_up);

        // Step 4: Decoder — already (C=512, T_stft)
        ggml_tensor* x_dec = x_up;
        // Wave decoder: 8L Transformer with AdaLN-Zero
        for (int i = 0; i < n_dec_layers; i++) {
            x_dec = miocodec_adaln_transformer_layer(ctx0, x_dec, global_cond, ctx->weights.wave_dec_layers[i],
                                                     dec_n_heads, dec_rope_pos, dec_mask);
        }

        // Final AdaLN norm (no gate)
        {
            auto [dec_normed, _] = miocodec_adaln_modulate(ctx0, x_dec, global_cond, ctx->weights.wave_dec_norm_adaln_w,
                                                           ctx->weights.wave_dec_norm_adaln_b, false);
            x_dec = dec_normed;
        }

        // Mark wave_decoder_out
        ggml_set_name(x_dec, "wave_decoder_out");
        ggml_set_output(x_dec);

        // Step 5: Post_net ResNet — already (C=512, T_stft)
        ggml_tensor* x_post = x_dec;
        for (int i = 0; i < n_resnet; i++)
            x_post = miocodec_resnet_block(ctx0, x_post, ctx->weights.wave_post_net[i], n_groups);
        ggml_set_name(x_post, "wave_post_net_out");
        ggml_set_output(x_post);

        // Step 6: SnakeBeta Upsampler — (C=512, T_stft) → (C=512, T_istft)
        // Two stages: ConvTranspose(weight_norm) + SnakeBeta + ResNet
        // Stage 0: 512→256, stride=3, k=9. Stage 1: 256→128, stride=3, k=9.
        // Then out_proj(128→512) + out_snake(512).
        ggml_tensor* x_upsamp = x_post; // (512, T_stft)
        int T_cur = T_stft;
        int C_cur = dim_pre; // 512
        for (int si = 0; si < (int)ctx->weights.wave_upsampler_stages.size(); si++) {
            auto& us = ctx->weights.wave_upsampler_stages[si];
            int stride =
                ctx->hparams.wave_upsampler_factors.size() > (size_t)si ? ctx->hparams.wave_upsampler_factors[si] : 3;
            int C_next = C_cur / 2; // 512→256→128
            int T_next = T_cur * stride;

            // Weight-norm ConvTranspose1d: w = g * v / ||v||
            // g=original0 shape (C_in, 1, 1), v=original1 shape (C_in, C_out, K)
            // For now, just use v directly (skip weight_norm — small numerical diff)
            // TODO: implement weight_norm properly

            // ConvTranspose: transpose → conv_transpose(p=0) → crop padding → reshape → transpose
            int kernel_size = ctx->hparams.wave_upsampler_kernel_sizes.size() > (size_t)si
                                  ? ctx->hparams.wave_upsampler_kernel_sizes[si]
                                  : 9;
            int pad = (kernel_size - stride) / 2;                                 // 3 for k=9,s=3
            ggml_tensor* xt_up = ggml_cont(ctx0, ggml_transpose(ctx0, x_upsamp)); // (T_cur, C_cur)
            xt_up = ggml_conv_transpose_1d(ctx0, us.conv_w1, xt_up, stride, 0, 1);
            // Output is (T_raw, C_next) where T_raw = (T_cur-1)*stride + kernel_size = T_next + 2*pad
            int T_raw = (T_cur - 1) * stride + kernel_size;
            xt_up = ggml_reshape_2d(ctx0, xt_up, T_raw, C_next);
            // Crop: remove pad from each end → [pad .. T_raw-pad) = T_next elements
            if (pad > 0) {
                xt_up = ggml_view_2d(ctx0, xt_up, T_next, C_next, xt_up->nb[1], pad * sizeof(float));
                xt_up = ggml_cont(ctx0, xt_up);
            }
            x_upsamp = ggml_cont(ctx0, ggml_transpose(ctx0, xt_up)); // (C_next, T_next)
            if (us.conv_b)
                x_upsamp = ggml_add(ctx0, x_upsamp, us.conv_b);

            // SnakeBeta: x + (1/exp(beta)) * sin²(exp(alpha) * x)
            // alpha, beta are (C_next) — need exp then element-wise ops
            // SnakeBeta activation after ConvTranspose
            if (us.snake_alpha && us.snake_beta)
                x_upsamp = miocodec_snake_beta(ctx0, x_upsamp, us.snake_alpha, us.snake_beta);

            // ResNet block (same pattern as prior/post net, but C=C_next)
            int gn_groups = std::min(n_groups, C_next);
            x_upsamp = miocodec_resnet_block(ctx0, x_upsamp, us.resblk, gn_groups);

            C_cur = C_next;
            T_cur = T_next;
        }

        // out_proj: Linear(128→512) — x_upsamp is (C=128, T_istft)
        // ggml_mul_mat(W, x) = x @ W^T. W ne[0]=128 must match x ne[0]=128. ✓
        x_upsamp = ggml_mul_mat(ctx0, ctx->weights.wave_upsampler_out_proj_w, x_upsamp); // (512, T_istft)
        if (ctx->weights.wave_upsampler_out_proj_b)
            x_upsamp = ggml_add(ctx0, x_upsamp, ctx->weights.wave_upsampler_out_proj_b);

        // Final SnakeBeta on upsampler output
        if (ctx->weights.wave_upsampler_out_snake_alpha && ctx->weights.wave_upsampler_out_snake_beta)
            x_upsamp = miocodec_snake_beta(ctx0, x_upsamp, ctx->weights.wave_upsampler_out_snake_alpha,
                                           ctx->weights.wave_upsampler_out_snake_beta);

        ggml_set_name(x_upsamp, "wave_upsampler_out");
        ggml_set_output(x_upsamp);

        // Step 7: ISTFT head Linear(512, 394) — on upsampler output (512, T_istft)
        ggml_tensor* x_istft = ggml_mul_mat(ctx0, ctx->weights.istft_out_w, x_upsamp);
        x_istft = ggml_add(ctx0, x_istft, ctx->weights.istft_out_b);
        ggml_set_name(x_istft, "istft_mag_phase");
        ggml_set_output(x_istft);

        // Build graph
        ggml_cgraph* gf2 = ggml_new_graph_custom(ctx0, 8192, false);
        ggml_build_forward_expand(gf2, x_istft);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf2)) {
            fprintf(stderr, "miocodec: full decode graph alloc failed\n");
            ggml_free(ctx0);
            free(prenet_data);
            *out_n = 0;
            return nullptr;
        }

        // Set inputs
        ggml_backend_tensor_set(x, prenet_data, 0, sizeof(float) * dim_pre * T_pre);
        free(prenet_data);

        // Global embedding
        ggml_backend_tensor_set(global_cond, global_embedding, 0, sizeof(float) * 128);

        // Decoder positions
        {
            std::vector<int32_t> positions(T_stft);
            for (int i = 0; i < T_stft; i++)
                positions[i] = i;
            ggml_backend_tensor_set(dec_rope_pos, positions.data(), 0, sizeof(int32_t) * T_stft);
        }

        // Decoder window mask
        if (dec_mask) {
            std::vector<float> mask_data((size_t)T_stft * T_stft, -INFINITY);
            for (int q = 0; q < T_stft; q++) {
                int k_start = std::max(0, q - dec_window_ps);
                int k_end = std::min(T_stft, q + dec_window_ps + 1);
                if (k_end > k_start)
                    std::fill_n(mask_data.data() + (size_t)q * T_stft + k_start, k_end - k_start, 0.0f);
            }
            ggml_backend_tensor_set(dec_mask, mask_data.data(), 0, sizeof(float) * T_stft * T_stft);
        }

        // Compute
        ggml_backend_sched_graph_compute(ctx->sched, gf2);

        // For istft_mag_phase/output_waveform: extract upsampler output and compute CPU-side
        // (avoids ggml graph buffer reuse corrupting the ISTFT matmul input)
        if (strcmp(stage_name, "istft_mag_phase") == 0 || strcmp(stage_name, "output_waveform") == 0) {
            // Get upsampler output (which is cos=1.0)
            ggml_tensor* up_t = x_upsamp;
            int up_n = (int)ggml_nelements(up_t);
            std::vector<float> up_dat(up_n);
            ggml_backend_tensor_get(up_t, up_dat.data(), 0, up_n * sizeof(float));
            int T_is = up_n / 512;
            int D_out = (int)ctx->weights.istft_out_w->ne[1]; // 394
            int D_in = 512;
            // Read weight & bias (cached F32 or dequantize fallback)
            const float* w_d = ctx->istft_out_w_cached.empty() ? nullptr : ctx->istft_out_w_cached.data();
            const float* b_d = ctx->istft_out_b_cached.empty() ? nullptr : ctx->istft_out_b_cached.data();
            std::vector<float> w_fb, b_fb;
            if (!w_d) {
                w_fb = miocodec_dequant_tensor(ctx->weights.istft_out_w);
                w_d = w_fb.data();
            }
            if (!b_d) {
                b_fb = miocodec_dequant_tensor(ctx->weights.istft_out_b);
                b_d = b_fb.data();
            }
            // Linear: result[d, t] = sum_k W[d,k] * up[t,k] + b[d]
            // Store in (D_out, T) layout = (394, T) matching Python's transpose(0,1) output
            float* result = (float*)malloc(sizeof(float) * D_out * T_is);
#if defined(CRISPASR_MIOCODEC_BLAS)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, D_out, T_is, D_in, 1.0f, w_d, D_in, up_dat.data(),
                        D_in, 0.0f, result, T_is);
            for (int d = 0; d < D_out; d++) {
                const float bias = b_d[d];
                float* r_row = result + d * T_is;
                for (int t = 0; t < T_is; t++)
                    r_row[t] += bias;
            }
#else
            for (int d = 0; d < D_out; d++) {
                const float bias = b_d[d];
                const float* w_row = w_d + d * D_in;
                float* r_row = result + d * T_is;
                for (int t = 0; t < T_is; t++) {
                    const float* up_row = up_dat.data() + t * D_in;
                    float sum = bias;
                    for (int k = 0; k < D_in; k++)
                        sum += w_row[k] * up_row[k];
                    r_row[t] = sum;
                }
            }
#endif
            if (strcmp(stage_name, "istft_mag_phase") == 0) {
                ggml_free(ctx0);
                *out_n = D_out * T_is;
                return result;
            }

            // output_waveform: apply exp(mag), clamp, cos/sin(phase), then ISTFT
            int n_freq = D_out / 2;                     // 197 = n_fft/2 + 1
            int n_fft_val = (int)ctx->hparams.n_fft;    // 392
            int hop_val = (int)ctx->hparams.hop_length; // 98

            // result is (394, T_is) = (D_out, T) with D_out=394, first 197=mag, second 197=phase
            // Rearrange to (T_is, n_freq) for mag and phase separately
            std::vector<float> mag_arr(T_is * n_freq), phase_arr(T_is * n_freq);
            for (int t = 0; t < T_is; t++) {
                for (int f = 0; f < n_freq; f++) {
                    float lm = result[f * T_is + t];            // log-magnitude
                    float ph = result[(f + n_freq) * T_is + t]; // phase
                    float m = std::exp(lm);
                    if (m > 100.0f)
                        m = 100.0f;
                    mag_arr[t * n_freq + f] = m;
                    phase_arr[t * n_freq + f] = ph;
                }
            }
            free(result);

            // ISTFT with "same" padding → trim pad from each side
            auto pcm = core_istft::istft(mag_arr.data(), phase_arr.data(), n_fft_val, hop_val, T_is, nullptr,
                                         core_istft::TRIM_SAME);

            float* wav = (float*)malloc(sizeof(float) * pcm.size());
            memcpy(wav, pcm.data(), sizeof(float) * pcm.size());
            ggml_free(ctx0);
            *out_n = (int)pcm.size();
            return wav;
        }

        // Extract other stages using direct tensor pointers
        ggml_tensor* out_t = nullptr;
        if (strcmp(stage_name, "wave_upsampler_out") == 0)
            out_t = x_upsamp;
        else if (strcmp(stage_name, "wave_decoder_out") == 0)
            out_t = x_dec;
        else if (strcmp(stage_name, "wave_post_net_out") == 0)
            out_t = x_post;
        else if (strcmp(stage_name, "wave_prior_net_out") == 0)
            out_t = x_up; // prior_net = x_up after ResNet
        else
            out_t = ggml_graph_get_tensor(gf2, stage_name);

        if (!out_t) {
            fprintf(stderr, "miocodec: stage '%s' not found in decode graph\n", stage_name);
            ggml_free(ctx0);
            *out_n = 0;
            return nullptr;
        }

        int n_out = (int)ggml_nelements(out_t);
        float* result = (float*)malloc(sizeof(float) * n_out);
        ggml_backend_tensor_get(out_t, result, 0, sizeof(float) * n_out);
        ggml_free(ctx0);
        *out_n = n_out;
        return result;
    }

    fprintf(stderr, "miocodec_extract_stage: unknown stage '%s'\n", stage_name);
    *out_n = 0;
    return nullptr;
}
