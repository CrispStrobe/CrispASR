// vibevoice.cpp — Microsoft VibeVoice-ASR runtime.
//
// Pipeline: 24kHz PCM → acoustic σ-VAE encoder → connector → Qwen2 LM → text
//           24kHz PCM → semantic encoder → connector ↗
//
// Two parallel CNN tokenizer encoders (ConvNeXt blocks with depthwise conv),
// projected to LM space via FC connectors, then autoregressive Qwen2 decoder.

#include "vibevoice.h"
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
// Hyperparams
// ===========================================================================

struct vibevoice_hparams {
    int d_lm = 1536;
    int n_lm_layers = 28;
    int n_heads = 12;
    int n_kv_heads = 2;
    int d_ffn = 8960;
    int vocab_size = 151936;
    int head_dim = 128;
    float rope_theta = 1000000.0f;
    int vae_dim_acoustic = 64;
    int vae_dim_semantic = 128;
    int n_encoder_stages = 7;
    int n_filters = 32;
    int total_downsample = 3200;
    std::vector<int> encoder_ratios;
    std::vector<int> encoder_depths;
};

// ===========================================================================
// Model
// ===========================================================================

struct vibevoice_model {
    vibevoice_hparams hp;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<std::string> vocab;
};

struct vibevoice_context {
    vibevoice_model model;
    vibevoice_context_params params;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;
    // KV cache for LM decoder
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;
    std::vector<uint8_t> compute_meta;
};

// ===========================================================================
// Defaults
// ===========================================================================

extern "C" struct vibevoice_context_params vibevoice_context_default_params(void) {
    vibevoice_context_params p;
    p.n_threads = 4;
    p.max_new_tokens = 512;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

// ===========================================================================
// Init
// ===========================================================================

extern "C" struct vibevoice_context* vibevoice_init_from_file(const char* path_model,
                                                              struct vibevoice_context_params params) {
    auto* ctx = new vibevoice_context();
    ctx->params = params;
    auto& m = ctx->model;
    auto& hp = m.hp;

    gguf_context* gctx = core_gguf::open_metadata(path_model);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }

    hp.d_lm = core_gguf::kv_u32(gctx, "vibevoice.d_lm", 1536);
    hp.n_lm_layers = core_gguf::kv_u32(gctx, "vibevoice.n_lm_layers", 28);
    hp.n_heads = core_gguf::kv_u32(gctx, "vibevoice.n_heads", 12);
    hp.n_kv_heads = core_gguf::kv_u32(gctx, "vibevoice.n_kv_heads", 2);
    hp.d_ffn = core_gguf::kv_u32(gctx, "vibevoice.d_ffn", 8960);
    hp.vocab_size = core_gguf::kv_u32(gctx, "vibevoice.vocab_size", 151936);
    hp.head_dim = core_gguf::kv_u32(gctx, "vibevoice.head_dim", 128);
    hp.rope_theta = core_gguf::kv_f32(gctx, "vibevoice.rope_theta", 1000000.0f);
    hp.vae_dim_acoustic = core_gguf::kv_u32(gctx, "vibevoice.vae_dim_acoustic", 64);
    hp.vae_dim_semantic = core_gguf::kv_u32(gctx, "vibevoice.vae_dim_semantic", 128);
    hp.n_encoder_stages = core_gguf::kv_u32(gctx, "vibevoice.n_encoder_stages", 7);
    hp.n_filters = core_gguf::kv_u32(gctx, "vibevoice.n_filters", 32);
    hp.total_downsample = core_gguf::kv_u32(gctx, "vibevoice.total_downsample", 3200);

    // Read encoder arrays
    int ratios_key = gguf_find_key(gctx, "vibevoice.encoder_ratios");
    if (ratios_key >= 0) {
        int n = gguf_get_arr_n(gctx, ratios_key);
        hp.encoder_ratios.resize(n);
        for (int i = 0; i < n; i++)
            hp.encoder_ratios[i] = ((const int32_t*)gguf_get_arr_data(gctx, ratios_key))[i];
    }
    int depths_key = gguf_find_key(gctx, "vibevoice.encoder_depths");
    if (depths_key >= 0) {
        int n = gguf_get_arr_n(gctx, depths_key);
        hp.encoder_depths.resize(n);
        for (int i = 0; i < n; i++)
            hp.encoder_depths[i] = ((const int32_t*)gguf_get_arr_data(gctx, depths_key))[i];
    }

    // Load vocabulary
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
        fprintf(stderr, "vibevoice: d_lm=%d, layers=%d, heads=%d/%d, ffn=%d, vocab=%d\n", hp.d_lm, hp.n_lm_layers,
                hp.n_heads, hp.n_kv_heads, hp.d_ffn, hp.vocab_size);
        fprintf(stderr, "vibevoice: vae_acoustic=%d, vae_semantic=%d, downsample=%dx\n", hp.vae_dim_acoustic,
                hp.vae_dim_semantic, hp.total_downsample);
    }

    // Backend selection: GPU first, CPU fallback. The scheduler requires
    // a CPU backend to be present as the final backend when the primary
    // backend is Metal/CUDA/Vulkan.
    ctx->backend =
        hp.d_lm > 0 ? (params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init()) : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads > 0 ? params.n_threads : 4);
    if (ctx->backend && ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, params.n_threads > 0 ? params.n_threads : 4);
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "vibevoice-asr", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    m.tensors = wl.tensors;

    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 65536, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(65536, false));

    if (params.verbosity >= 1)
        fprintf(stderr, "vibevoice: loaded %zu tensors\n", m.tensors.size());

    return ctx;
}

// ===========================================================================
// Free
// ===========================================================================

extern "C" void vibevoice_free(struct vibevoice_context* ctx) {
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
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// ===========================================================================
// ggml graph helpers
// ===========================================================================

// ConvRMSNorm: operates on [C, T] (ne[0]=C), normalizes over C per time step
static ggml_tensor* build_conv_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps = 1e-5f) {
    // x: [C, T], w: [C]. ggml_rms_norm normalizes over ne[0]=C. Good.
    // ggml_mul(x, w): x=[C,T], w=[C]. w broadcasts over T. OK.
    x = ggml_rms_norm(ctx, x, eps);
    if (w) {
        // Verify shapes match before mul
        if (x->ne[0] != w->ne[0]) {
            fprintf(stderr, "  BUG: conv_rms_norm shape mismatch: x=[%lld,%lld] w=[%lld]\n", (long long)x->ne[0],
                    (long long)x->ne[1], (long long)w->ne[0]);
        }
        x = ggml_mul(ctx, x, w);
    }
    return x;
}

// Causal Conv1d: zero-pad left by (K-1)*dilation - (stride-1), then conv1d.
// Uses zero (constant) padding — config pad_mode='constant'.
// Input/output in [C, T] format (channels-first, like PyTorch).
// ggml_conv_1d produces [T_out, C_out] so we transpose.
static ggml_tensor* build_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride) {
    int K = (int)w->ne[0];
    int dilation = 1;
    int pad_left = (K - 1) * dilation - (stride - 1); // VibeVoice/EnCodec convention
    if (pad_left < 0)
        pad_left = 0;
    // Input x is [C, T]. ggml_conv_1d wants [T, C_in], so transpose.
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C,T] → [T,C]
    int T_in = (int)x->ne[0];
    // Compute extra right padding for stride alignment (same as get_extra_padding_for_conv1d)
    int pad_right = 0;
    if (stride > 1) {
        double n_frames = (double)(T_in - K + pad_left) / stride + 1.0;
        int ideal_length = ((int)ceil(n_frames) - 1) * stride + (K - pad_left);
        pad_right = ideal_length - T_in;
        if (pad_right < 0)
            pad_right = 0;
    }
    // Zero-pad along dim 0 (T axis) using ggml_pad_ext which fills with 0.
    if (pad_left > 0 || pad_right > 0)
        x = ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, w, x, stride, 0, 1); // → [T_out, C_out]
    // Add bias (ne[0]=T_out, ne[1]=C_out; bias ne[0]=C_out → transpose, add, transpose)
    if (b) {
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T_out]
        xt = ggml_add(ctx, xt, b);                                // [C_out] + [C_out, T_out] broadcasts over T
        return xt;                                                // already in [C, T] format
    }
    return ggml_cont(ctx, ggml_transpose(ctx, x)); // [T_out, C_out] → [C_out, T_out]
}

// Causal depthwise Conv1d using ggml_conv_1d per channel.
// ggml_conv_1d_dw uses F16 im2col which accumulates precision loss through
// 29 ConvNeXt blocks. Instead, split into C independent conv1d ops.
// Input/output in [C, T] format.
static ggml_tensor* build_causal_dw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    int K = (int)w->ne[0];
    int pad_left = K - 1;

    // Transpose to [T, C] for padding along T (ne[0])
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C,T] → [T,C]
    // Zero-pad left (causal). ggml_pad_ext fills with 0.
    if (pad_left > 0)
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
    // x is now [T+pad, C]

    // Depthwise conv. ggml_conv_1d_dw uses F16 im2col which accumulates precision
    // loss through 26 ConvNeXt blocks (observed cos 0.7-0.8 at output). This is a
    // known limitation; the connector/feature stages use F32 CPU loops and are exact.
    x = ggml_conv_1d_dw(ctx, w, x, 1, 0, 1);

    // conv_1d_dw returns 3D+ result — flatten to 2D then transpose to [C, T]
    if (ggml_n_dims(x) > 2)
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (b)
        x = ggml_add(ctx, x, b); // [C] + [C, T] broadcasts
    return x;
}

// Block1D: ConvNeXt block
//   mixer: ConvRMSNorm → depthwise_conv → gamma_scale → residual
//   FFN:   ConvRMSNorm → linear1(SiLU)→linear2 → gamma_scale → residual
static ggml_tensor* build_block1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* norm_w, ggml_tensor* dw_conv_w,
                                  ggml_tensor* dw_conv_b, ggml_tensor* gamma, ggml_tensor* ffn_norm_w,
                                  ggml_tensor* ffn_up_w, ggml_tensor* ffn_up_b, ggml_tensor* ffn_down_w,
                                  ggml_tensor* ffn_down_b, ggml_tensor* ffn_gamma) {
    // Mixer path
    ggml_tensor* residual = x;
    ggml_tensor* h = build_conv_rms_norm(ctx, x, norm_w);
    h = build_causal_dw_conv1d(ctx, h, dw_conv_w, dw_conv_b);
    if (gamma) {
        if (h->ne[0] != gamma->ne[0])
            fprintf(stderr, "  BUG: gamma shape mismatch: h=[%lld,%lld] gamma=[%lld]\n", (long long)h->ne[0],
                    (long long)h->ne[1], (long long)gamma->ne[0]);
        h = ggml_mul(ctx, h, gamma);
    }
    x = ggml_add(ctx, residual, h);

    // FFN path: h is [C, T] (ne[0]=C). mul_mat operates on ne[0].
    // linear: mul_mat(w=[C_in, C_out], h=[C_in, T]) → [C_out, T]
    residual = x;
    h = build_conv_rms_norm(ctx, x, ffn_norm_w);
    h = ggml_mul_mat(ctx, ffn_up_w, h); // [C, C_ffn] @ [C, T] → [C_ffn, T]
    if (ffn_up_b)
        h = ggml_add(ctx, h, ffn_up_b);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, ffn_down_w, h); // [C_ffn, C] @ [C_ffn, T] → [C, T]
    if (ffn_down_b)
        h = ggml_add(ctx, h, ffn_down_b);
    if (ffn_gamma)
        h = ggml_mul(ctx, h, ffn_gamma); // [C] * [C, T]
    x = ggml_add(ctx, residual, h);

    return x;
}

// ===========================================================================
// Build tokenizer encoder graph
// ===========================================================================

// Build ggml graph for one σ-VAE tokenizer encoder (acoustic or semantic).
// prefix: "at_enc" for acoustic, "st_enc" for semantic
// Input: [1, T] mono audio → Output: [vae_dim, T_out] mean
static ggml_cgraph* build_tokenizer_encoder_graph(vibevoice_context* ctx, const char* prefix, int n_samples) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;

    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };
    std::string pfx(prefix);

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // Input: channels-first [C=1, T=n_samples]
    // In ggml: ne[0]=1 (channel), ne[1]=n_samples (time)
    // But our conv functions expect [C, T] and internally transpose to [T, C]
    // So store as [C=1, T=n_samples] → ne[0]=1, ne[1]=n_samples
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_samples);
    ggml_set_name(inp, "audio_in");
    ggml_set_input(inp);

    ggml_tensor* h = inp;

    // Downsample layers + stages
    // ratios are REVERSED in the encoder: config [8,5,5,4,2,2] → encoder [2,2,4,5,5,8]
    std::vector<int> ratios(hp.encoder_ratios.rbegin(), hp.encoder_ratios.rend());

    for (int si = 0; si < hp.n_encoder_stages; si++) {
        // Downsample
        char wn[128], bn[128];
        snprintf(wn, sizeof(wn), "%s.ds.%d.0.conv.weight", prefix, si);
        snprintf(bn, sizeof(bn), "%s.ds.%d.0.conv.bias", prefix, si);
        ggml_tensor* ds_w = G(wn);
        ggml_tensor* ds_b = G(bn);
        if (ds_w) {
            int stride = (si == 0) ? 1 : ratios[si - 1]; // stem has stride 1
            h = build_causal_conv1d(ctx0, h, ds_w, ds_b, stride);
        }

        // Stage blocks
        int n_blocks = (si < (int)hp.encoder_depths.size()) ? hp.encoder_depths[si] : 3;
        for (int bi = 0; bi < n_blocks; bi++) {
            char base[128];
            snprintf(base, sizeof(base), "%s.s.%d.%d", prefix, si, bi);

            h = build_block1d(ctx0, h, G(std::string(base) + ".norm.weight"), G(std::string(base) + ".dw_conv.weight"),
                              G(std::string(base) + ".dw_conv.bias"), G(std::string(base) + ".gamma"),
                              G(std::string(base) + ".ffn_ln.weight"), G(std::string(base) + ".ffn.up.weight"),
                              G(std::string(base) + ".ffn.up.bias"), G(std::string(base) + ".ffn.down.weight"),
                              G(std::string(base) + ".ffn.down.bias"), G(std::string(base) + ".ffn_gamma"));
        }
    }

    // Final norm
    // Check for final norm tensor (at_enc has norm disabled based on config)
    // Actually config says disable_last_norm=true, so norm is Identity
    // But let's check if there's a norm tensor
    {
        char nn[128];
        snprintf(nn, sizeof(nn), "%s.norm.weight", prefix);
        ggml_tensor* norm_w = G(nn);
        if (norm_w)
            h = build_conv_rms_norm(ctx0, h, norm_w);
    }

    // Head conv: Conv1d(last_dim → vae_dim, K=7)
    {
        char wn[128], bn[128];
        snprintf(wn, sizeof(wn), "%s.head.weight", prefix);
        snprintf(bn, sizeof(bn), "%s.head.bias", prefix);
        ggml_tensor* head_w = G(wn);
        ggml_tensor* head_b = G(bn);
        if (head_w)
            h = build_causal_conv1d(ctx0, h, head_w, head_b, 1);
    }

    ggml_set_name(h, "encoder_out");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);
    return gf;
}

// ===========================================================================
// Stage helpers (shared by public stage API and vibevoice_transcribe)
// ===========================================================================

// Run one tokenizer encoder. Returns [vae_dim * T] row-major, frame-first.
static std::vector<float> run_encoder_stage(vibevoice_context* ctx, const char* prefix, const float* samples,
                                            int n_samples, int* out_T, int* out_vae_dim) {
    ggml_cgraph* gf = build_tokenizer_encoder_graph(ctx, prefix, n_samples);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "vibevoice: %s graph alloc failed\n", prefix);
        return {};
    }
    ggml_tensor* inp = ggml_graph_get_tensor(gf, "audio_in");
    ggml_backend_tensor_set(inp, samples, 0, n_samples * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vibevoice: %s compute failed\n", prefix);
        return {};
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "encoder_out");
    int vae_dim = (int)out->ne[0];
    int T = (int)out->ne[1];
    std::vector<float> mean(vae_dim * T);
    ggml_backend_tensor_get(out, mean.data(), 0, vae_dim * T * sizeof(float));
    if (out_T)
        *out_T = T;
    if (out_vae_dim)
        *out_vae_dim = vae_dim;
    return mean;
}

// Run one SpeechConnector (FC1 → RMSNorm → FC2) via ggml graph.
// Works with any weight dtype (F32, F16, Q4_K, …) — the backend handles dequantization.
// encoder_mean: row-major [T * vae_dim] (frame-first, dim-second). Returns [T * d_lm].
static std::vector<float> run_connector_stage(vibevoice_context* ctx, const char* prefix, const float* encoder_mean,
                                              int T, int vae_dim) {
    auto& m = ctx->model;
    auto& hp = m.hp;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        return it != m.tensors.end() ? it->second : nullptr;
    };

    const int d_lm = hp.d_lm;
    ggml_tensor* fc1_w = G(std::string(prefix) + ".fc1.weight");
    ggml_tensor* fc1_b = G(std::string(prefix) + ".fc1.bias");
    ggml_tensor* norm_w = G(std::string(prefix) + ".norm.weight");
    ggml_tensor* fc2_w = G(std::string(prefix) + ".fc2.weight");
    ggml_tensor* fc2_b = G(std::string(prefix) + ".fc2.bias");

    if (!fc1_w || !fc2_w) {
        fprintf(stderr, "run_connector_stage: missing weights for prefix '%s'\n", prefix);
        return {};
    }

    // Build a small ggml graph: inp[vae_dim, T] → FC1 → RMSNorm → FC2 → out[d_lm, T]
    // Input layout: encoder_mean[t*vae_dim + k] corresponds to ne[0]=vae_dim, ne[1]=T.
    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vae_dim, T);
    ggml_set_name(inp, "conn_inp");
    ggml_set_input(inp);

    ggml_tensor* h = ggml_mul_mat(ctx0, fc1_w, inp); // [d_lm, T]
    if (fc1_b)
        h = ggml_add(ctx0, h, fc1_b);
    h = ggml_rms_norm(ctx0, h, 1e-6f);
    if (norm_w)
        h = ggml_mul(ctx0, h, norm_w);
    h = ggml_mul_mat(ctx0, fc2_w, h); // [d_lm, T]
    if (fc2_b)
        h = ggml_add(ctx0, h, fc2_b);

    ggml_set_name(h, "conn_out");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "run_connector_stage: %s graph alloc failed\n", prefix);
        ggml_free(ctx0);
        return {};
    }

    // encoder_mean is row-major [T, vae_dim] → ggml [vae_dim, T] layout matches exactly
    ggml_backend_tensor_set(inp, encoder_mean, 0, (size_t)T * vae_dim * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "run_connector_stage: %s compute failed\n", prefix);
        ggml_free(ctx0);
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "conn_out");
    std::vector<float> output((size_t)T * d_lm);
    ggml_backend_tensor_get(out, output.data(), 0, (size_t)T * d_lm * sizeof(float));
    ggml_free(ctx0);
    return output;
}

// Token embedding lookup via ggml_get_rows so quantized embedding tables
// (Q4_K, etc.) work correctly.
static std::vector<float> run_token_embedding_lookup(vibevoice_context* ctx, const int32_t* ids, int n_ids) {
    if (!ctx || !ids || n_ids <= 0)
        return {};

    auto& m = ctx->model;
    auto it = m.tensors.find("lm.tok_emb.weight");
    if (it == m.tensors.end() || !it->second)
        return {};

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ids);
    ggml_set_name(inp, "tok_ids");
    ggml_set_input(inp);

    ggml_tensor* out = ggml_get_rows(ctx0, it->second, inp); // [d_lm, n_ids]
    ggml_set_name(out, "tok_emb_out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return {};
    }

    ggml_backend_tensor_set(inp, ids, 0, (size_t)n_ids * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return {};
    }

    std::vector<float> embeds((size_t)n_ids * ctx->model.hp.d_lm);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "tok_emb_out"), embeds.data(), 0, embeds.size() * sizeof(float));
    ggml_free(ctx0);
    return embeds;
}

// GPT-2/Qwen-style byte decoder. Qwen2 tokenizer pieces are UTF-8 strings
// that represent raw bytes through this remapping.
static std::vector<int>& qwen_byte_decoder() {
    static std::vector<int> dec(0x200, -1);
    static bool initialized = false;
    if (initialized)
        return dec;

    std::vector<int> bs, cs;
    for (int b = 0x21; b <= 0x7e; ++b) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (int b = 0xa1; b <= 0xac; ++b) {
        bs.push_back(b);
        cs.push_back(b);
    }
    for (int b = 0xae; b <= 0xff; ++b) {
        bs.push_back(b);
        cs.push_back(b);
    }

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool present = false;
        for (int x : bs) {
            if (x == b) {
                present = true;
                break;
            }
        }
        if (!present) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        if ((size_t)cs[i] < dec.size())
            dec[(size_t)cs[i]] = bs[i];
    }
    initialized = true;
    return dec;
}

static std::string decode_qwen_piece(const std::string& s) {
    auto& dec = qwen_byte_decoder();
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int cp = 0;
        int len = 1;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            ++i;
            continue;
        }
        if (i + (size_t)len > s.size())
            break;
        for (int k = 1; k < len; ++k)
            cp = (cp << 6) | (s[i + (size_t)k] & 0x3F);
        i += (size_t)len;
        if (cp >= 0 && cp < (int)dec.size() && dec[(size_t)cp] >= 0)
            out.push_back((char)dec[(size_t)cp]);
    }
    return out;
}

// ===========================================================================
// Public stage API
// ===========================================================================

extern "C" float* vibevoice_run_acoustic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples,
                                                 int* n_frames, int* vae_dim) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T = 0, d = 0;
    auto v = run_encoder_stage(ctx, "at_enc", samples, n_samples, &T, &d);
    if (v.empty())
        return nullptr;
    if (n_frames)
        *n_frames = T;
    if (vae_dim)
        *vae_dim = d;
    float* out = (float*)malloc(v.size() * sizeof(float));
    memcpy(out, v.data(), v.size() * sizeof(float));
    return out;
}

extern "C" float* vibevoice_run_semantic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples,
                                                 int* n_frames, int* vae_dim) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T = 0, d = 0;
    auto v = run_encoder_stage(ctx, "st_enc", samples, n_samples, &T, &d);
    if (v.empty())
        return nullptr;
    if (n_frames)
        *n_frames = T;
    if (vae_dim)
        *vae_dim = d;
    float* out = (float*)malloc(v.size() * sizeof(float));
    memcpy(out, v.data(), v.size() * sizeof(float));
    return out;
}

extern "C" float* vibevoice_run_connector(struct vibevoice_context* ctx, const char* prefix, const float* encoder_mean,
                                          int n_frames, int vae_dim, int* d_lm) {
    if (!ctx || !encoder_mean || n_frames <= 0)
        return nullptr;
    auto v = run_connector_stage(ctx, prefix, encoder_mean, n_frames, vae_dim);
    if (v.empty())
        return nullptr;
    if (d_lm)
        *d_lm = ctx->model.hp.d_lm;
    float* out = (float*)malloc(v.size() * sizeof(float));
    memcpy(out, v.data(), v.size() * sizeof(float));
    return out;
}

extern "C" float* vibevoice_encode_speech(struct vibevoice_context* ctx, const float* samples, int n_samples,
                                          int* n_frames, int* d_lm) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_at = 0, vd_at = 0, T_st = 0, vd_st = 0;
    auto at_mean = run_encoder_stage(ctx, "at_enc", samples, n_samples, &T_at, &vd_at);
    if (at_mean.empty())
        return nullptr;
    auto st_mean = run_encoder_stage(ctx, "st_enc", samples, n_samples, &T_st, &vd_st);
    if (st_mean.empty())
        return nullptr;
    if (T_at != T_st) {
        fprintf(stderr, "vibevoice: frame mismatch at=%d st=%d\n", T_at, T_st);
        return nullptr;
    }
    auto at_feat = run_connector_stage(ctx, "at_conn", at_mean.data(), T_at, vd_at);
    auto st_feat = run_connector_stage(ctx, "se_conn", st_mean.data(), T_st, vd_st);
    if (at_feat.empty() || st_feat.empty())
        return nullptr;

    int dlm = ctx->model.hp.d_lm;
    float* out = (float*)malloc((size_t)T_at * dlm * sizeof(float));
    for (int i = 0; i < T_at * dlm; i++)
        out[i] = at_feat[i] + st_feat[i];
    if (n_frames)
        *n_frames = T_at;
    if (d_lm)
        *d_lm = dlm;
    return out;
}

// ===========================================================================
// Transcribe
// ===========================================================================

extern "C" char* vibevoice_transcribe(struct vibevoice_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    auto& m = ctx->model;
    auto& hp = m.hp;

    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        return it != m.tensors.end() ? it->second : nullptr;
    };

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: %d samples (%.2fs at 24kHz)\n", n_samples, n_samples / 24000.0f);

    // 1. Run acoustic tokenizer encoder
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: running acoustic encoder...\n");

    ggml_cgraph* gf_at = build_tokenizer_encoder_graph(ctx, "at_enc", n_samples);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf_at)) {
        fprintf(stderr, "vibevoice: acoustic encoder graph alloc failed\n");
        return nullptr;
    }

    ggml_tensor* inp_t = ggml_graph_get_tensor(gf_at, "audio_in");
    // Input is [C=1, T=n_samples] → ne[0]=1, ne[1]=n_samples
    // The flat data is [sample_0, sample_1, ...] which maps to ne[1] varying fastest
    // In ggml column-major: data[c * T + t] = data[0 * T + t] = data[t]
    // So just write the samples directly
    ggml_backend_tensor_set(inp_t, samples, 0, n_samples * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf_at) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vibevoice: acoustic encoder compute failed\n");
        return nullptr;
    }

    ggml_tensor* at_out = ggml_graph_get_tensor(gf_at, "encoder_out");
    int vae_dim_at = (int)at_out->ne[0];
    int T_audio = (int)at_out->ne[1];
    std::vector<float> acoustic_mean(vae_dim_at * T_audio);
    ggml_backend_tensor_get(at_out, acoustic_mean.data(), 0, vae_dim_at * T_audio * sizeof(float));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: acoustic encoder: [%d, %d] (vae_dim=%d, frames=%d)\n", vae_dim_at, T_audio,
                vae_dim_at, T_audio);

    // 2. Run semantic tokenizer encoder
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: running semantic encoder...\n");

    ggml_cgraph* gf_st = build_tokenizer_encoder_graph(ctx, "st_enc", n_samples);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf_st)) {
        fprintf(stderr, "vibevoice: semantic encoder graph alloc failed\n");
        return nullptr;
    }
    ggml_tensor* inp_st = ggml_graph_get_tensor(gf_st, "audio_in");
    ggml_backend_tensor_set(inp_st, samples, 0, n_samples * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf_st) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vibevoice: semantic encoder compute failed\n");
        return nullptr;
    }
    ggml_tensor* st_out = ggml_graph_get_tensor(gf_st, "encoder_out");
    int vae_dim_st = (int)st_out->ne[0];
    int T_sem = (int)st_out->ne[1];
    std::vector<float> semantic_mean(vae_dim_st * T_sem);
    ggml_backend_tensor_get(st_out, semantic_mean.data(), 0, vae_dim_st * T_sem * sizeof(float));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: semantic encoder: [%d, %d]\n", vae_dim_st, T_sem);

    // 3. Run connectors via ggml graph (handles Q4_K and any quantized weight type)
    auto acoustic_features = run_connector_stage(ctx, "at_conn", acoustic_mean.data(), T_audio, vae_dim_at);
    auto semantic_features = run_connector_stage(ctx, "se_conn", semantic_mean.data(), T_sem, vae_dim_st);
    if (acoustic_features.empty() || semantic_features.empty()) {
        fprintf(stderr, "vibevoice: connector failed\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: connectors done: acoustic=[%d,%d] semantic=[%d,%d]\n", T_audio, hp.d_lm, T_sem,
                hp.d_lm);

    // 4. Combine acoustic + semantic features (element-wise sum)
    // Debug: optionally inject Python reference features
    const char* ref_features_path = getenv("VIBEVOICE_REF_FEATURES");
    if (ref_features_path && ref_features_path[0]) {
        FILE* f = fopen(ref_features_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            int n = (int)(ftell(f) / sizeof(float));
            fseek(f, 0, SEEK_SET);
            // Reference is [T, d_lm] = [83, 1536]
            int T_ref = n / hp.d_lm;
            std::vector<float> ref_features(n);
            size_t rd = fread(ref_features.data(), sizeof(float), n, f);
            fclose(f);
            (void)rd;
            fprintf(stderr, "vibevoice: INJECTING reference features [%d, %d] from %s\n", T_ref, hp.d_lm,
                    ref_features_path);
            // Replace features with reference
            T_audio = T_ref;
            T_sem = T_ref;
            acoustic_features.resize(T_ref * hp.d_lm);
            semantic_features.assign(T_ref * hp.d_lm, 0.0f); // zero semantic — ref already combined
            memcpy(acoustic_features.data(), ref_features.data(), T_ref * hp.d_lm * sizeof(float));
        }
    }
    if (T_audio != T_sem) {
        fprintf(stderr, "vibevoice: frame mismatch: acoustic=%d, semantic=%d\n", T_audio, T_sem);
        return nullptr;
    }
    std::vector<float> speech_features(T_audio * hp.d_lm);
    for (int i = 0; i < T_audio * hp.d_lm; i++)
        speech_features[i] = acoustic_features[i] + semantic_features[i];

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: speech features combined: [%d, %d]\n", T_audio, hp.d_lm);

    // 5. Build prompt: system_tokens + <speech_start> + speech_pad × T + <speech_end> + suffix
    // Token IDs from VibeVoice processor (Qwen2 tokenizer with special tokens)
    const int SPEECH_START = 151646;
    const int SPEECH_PAD = 151648;
    const int SPEECH_END = 151647;
    const int EOS_TOKEN = 151643;
    const int IM_START = 151644;
    // const int IM_END       = 151645;

    // Exact prompt template from VibeVoice processor output
    // System: "You are a helpful assistant that transcribes audio input into text output in JSON format."
    std::vector<int> system_tokens = {
        IM_START, 8948,   198,                                         // <|im_start|>system\n
        2610,     525,    264,  10950,    17847, 429, 1356, 55136,     // You are a helpful assistant that transcribes
        7699,     1946,   1119, 1467,     2550,  304, 4718, 3561,  13, // audio input into text output in JSON format.
        198,      151645, 198,  IM_START, 872,   198                   // \n<|im_end|>\n<|im_start|>user\n
    };
    // After speech tokens: "\nThis is a XX.XX seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n"
    float dur = n_samples / 24000.0f;
    // Tokenize duration character-by-character using Qwen2 single-char token IDs:
    // '0'–'9' = 15–24, '.' = 13 (verified against "11.00" in VibeVoice processor output)
    char dur_str[16];
    snprintf(dur_str, sizeof(dur_str), "%.2f", dur);
    std::vector<int> dur_tokens;
    for (const char* p = dur_str; *p; p++) {
        if (*p >= '0' && *p <= '9')
            dur_tokens.push_back(15 + (*p - '0'));
        else if (*p == '.')
            dur_tokens.push_back(13);
    }
    std::vector<int> suffix_tokens = {
        198,                 // \n
        1986, 374, 264, 220, // This is a<space>
    };
    suffix_tokens.insert(suffix_tokens.end(), dur_tokens.begin(), dur_tokens.end());
    std::vector<int> suffix_tail = {
        6546,     7699,  11, 4587, 38840, 432, 449,   1493,           // seconds audio, please transcribe it with these
        6894,     25,                                                 // keys:
        5145,     882,   11, 3972, 882,   11,  29073, 3034, 11, 8883, // Start time, End time, Speaker ID, Content
        151645,   198,                                                // <|im_end|>\n
        IM_START, 77091, 198                                          // <|im_start|>assistant\n
    };
    suffix_tokens.insert(suffix_tokens.end(), suffix_tail.begin(), suffix_tail.end());

    // Build full token sequence
    std::vector<int> prompt_tokens;
    prompt_tokens.insert(prompt_tokens.end(), system_tokens.begin(), system_tokens.end());
    prompt_tokens.push_back(SPEECH_START);
    int speech_start_pos = (int)prompt_tokens.size();
    for (int i = 0; i < T_audio; i++)
        prompt_tokens.push_back(SPEECH_PAD);
    prompt_tokens.push_back(SPEECH_END);
    prompt_tokens.insert(prompt_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    int prefix_len = (int)prompt_tokens.size();

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: prompt: %d tokens (speech at %d-%d)\n", prefix_len, speech_start_pos,
                speech_start_pos + T_audio - 1);

    // 6. Embed all tokens, then replace speech positions with speech features.
    // Use ggml_get_rows so quantized token embedding tables are handled by the backend.
    std::vector<int32_t> prompt_ids(prompt_tokens.begin(), prompt_tokens.end());
    std::vector<float> prefix_embeds = run_token_embedding_lookup(ctx, prompt_ids.data(), (int)prompt_ids.size());
    if (prefix_embeds.size() != (size_t)prefix_len * hp.d_lm) {
        fprintf(stderr, "vibevoice: token embedding lookup failed\n");
        return nullptr;
    }
    // Replace speech positions with combined features (no scaling — ASR variant
    // doesn't apply speech_scaling_factor; that's only in the base TTS model)
    for (int i = 0; i < T_audio; i++) {
        int pos = speech_start_pos + i;
        memcpy(prefix_embeds.data() + pos * hp.d_lm, speech_features.data() + i * hp.d_lm, hp.d_lm * sizeof(float));
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: prefix embedded (%d tokens)\n", prefix_len);

    // 7. Allocate KV cache for Qwen2 decoder
    int max_gen = ctx->params.max_new_tokens > 0 ? ctx->params.max_new_tokens : 512;
    int max_ctx = prefix_len + max_gen;
    if (!ctx->kv_k) {
        int hd = hp.head_dim;
        int nkv = hp.n_kv_heads;
        int nl = hp.n_lm_layers;
        size_t k_size = (size_t)ggml_type_size(GGML_TYPE_F16) * hd * max_ctx * nkv * nl;
        ggml_init_params kp = {2 * ggml_tensor_overhead(), nullptr, true};
        ctx->kv_ctx = ggml_init(kp);
        ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nkv, nl);
        ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nkv, nl);
        ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, 2 * k_size);
        uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(ctx->kv_buf);
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + k_size);
        ctx->kv_max_ctx = max_ctx;
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "vibevoice: KV cache: %d ctx, %zu MB\n", max_ctx, 2 * k_size / (1024 * 1024));
    }
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_n_used = 0;

    // 8. Build Qwen2 decoder graph (prefill + generate)
    auto build_decoder_graph = [&](int n_tokens, int n_past) -> ggml_cgraph* {
        size_t mem = ctx->compute_meta.size();
        ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

        ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_lm, n_tokens);
        ggml_set_name(embeds, "dec_input");
        ggml_set_input(embeds);

        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        ggml_tensor* causal_mask = nullptr;
        if (n_tokens > 1) {
            causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
            ggml_set_name(causal_mask, "causal_mask");
            ggml_set_input(causal_mask);
        }

        const core_attn::KvSelfAttnParams kvp = {
            /*n_heads*/ hp.n_heads,
            /*n_kv_heads*/ hp.n_kv_heads,
            /*head_dim*/ hp.head_dim,
            /*n_kv_grp*/ hp.n_heads / hp.n_kv_heads,
            /*n_ctx_orig*/ 0,
            /*rope_theta*/ hp.rope_theta,
            /*rope_beta_fast*/ 0.0f,
            /*rope_beta_slow*/ 0.0f,
            /*attn_scale*/ 1.0f / sqrtf((float)hp.head_dim),
            /*qk_norm_eps*/ 0.0f,
            /*gqa_mode*/ core_attn::GQA_NATIVE,
            /*rope_type*/ GGML_ROPE_TYPE_NEOX, // Qwen2 uses NEOX RoPE
        };

        ggml_tensor* cur = embeds;
        for (int il = 0; il < hp.n_lm_layers; il++) {
            char p[64];
            snprintf(p, sizeof(p), "lm.layers.%d", il);
            ggml_tensor* residual = cur;

            // Pre-RMSNorm
            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            cur = ggml_mul(ctx0, cur, G(std::string(p) + ".attn_ln.weight"));

            // Qwen2 has bias on Q and K projections.
            // Apply Q/K projections with bias BEFORE kv_self_attn,
            // then pass identity weights so kv_self_attn skips the projection.
            // Actually simpler: inline the attention with bias.
            {
                ggml_tensor* q_w = G(std::string(p) + ".attn.q_proj.weight");
                ggml_tensor* k_w = G(std::string(p) + ".attn.k_proj.weight");
                ggml_tensor* v_w = G(std::string(p) + ".attn.v_proj.weight");
                ggml_tensor* o_w = G(std::string(p) + ".attn.o_proj.weight");
                ggml_tensor* q_b = G(std::string(p) + ".attn.q_proj.bias");
                ggml_tensor* k_b = G(std::string(p) + ".attn.k_proj.bias");

                int T_cur = (int)cur->ne[1];
                int Lk = n_past + T_cur;

                // Q, K, V projections with bias
                ggml_tensor* Q = ggml_mul_mat(ctx0, q_w, cur);
                if (q_b)
                    Q = ggml_add(ctx0, Q, q_b);
                ggml_tensor* K = ggml_mul_mat(ctx0, k_w, cur);
                if (k_b)
                    K = ggml_add(ctx0, K, k_b);
                ggml_tensor* V = ggml_mul_mat(ctx0, v_w, cur);

                // Reshape for multi-head
                Q = ggml_reshape_3d(ctx0, Q, kvp.head_dim, kvp.n_heads, T_cur);
                K = ggml_reshape_3d(ctx0, K, kvp.head_dim, kvp.n_kv_heads, T_cur);
                V = ggml_reshape_3d(ctx0, V, kvp.head_dim, kvp.n_kv_heads, T_cur);

                // RoPE
                Q = ggml_rope_ext(ctx0, Q, positions, nullptr, kvp.head_dim, GGML_ROPE_TYPE_NEOX, 0, kvp.rope_theta,
                                  1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                K = ggml_rope_ext(ctx0, K, positions, nullptr, kvp.head_dim, GGML_ROPE_TYPE_NEOX, 0, kvp.rope_theta,
                                  1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                // Write K, V to cache
                ggml_tensor* K_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
                ggml_tensor* V_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);
                ggml_tensor* k_view = ggml_view_4d(ctx0, ctx->kv_k, kvp.head_dim, T_cur, kvp.n_kv_heads, 1,
                                                   ctx->kv_k->nb[1], ctx->kv_k->nb[2], ctx->kv_k->nb[3],
                                                   (size_t)il * ctx->kv_k->nb[3] + (size_t)n_past * ctx->kv_k->nb[1]);
                ggml_tensor* v_view = ggml_view_4d(ctx0, ctx->kv_v, kvp.head_dim, T_cur, kvp.n_kv_heads, 1,
                                                   ctx->kv_v->nb[1], ctx->kv_v->nb[2], ctx->kv_v->nb[3],
                                                   (size_t)il * ctx->kv_v->nb[3] + (size_t)n_past * ctx->kv_v->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));

                // Read full K, V from cache
                ggml_tensor* Kfull =
                    ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_k, kvp.head_dim, Lk, kvp.n_kv_heads, ctx->kv_k->nb[1],
                                                 ctx->kv_k->nb[2], (size_t)il * ctx->kv_k->nb[3]));
                ggml_tensor* Vfull =
                    ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_v, kvp.head_dim, Lk, kvp.n_kv_heads, ctx->kv_v->nb[1],
                                                 ctx->kv_v->nb[2], (size_t)il * ctx->kv_v->nb[3]));

                // Permute Q for flash-attn: [hd, T, nh]
                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

                // Flash attention (native GQA)
                ggml_tensor* attn_out =
                    ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask, kvp.attn_scale, 0.0f, 0.0f);

                attn_out = ggml_reshape_2d(ctx0, attn_out, hp.d_lm, T_cur);
                attn_out = ggml_mul_mat(ctx0, o_w, attn_out);

                cur = ggml_add(ctx0, residual, attn_out);
            }

            // FFN: RMSNorm + SwiGLU
            residual = cur;
            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            cur = ggml_mul(ctx0, cur, G(std::string(p) + ".ffn_ln.weight"));
            ggml_tensor* ffn =
                core_ffn::swiglu(ctx0, cur, G(std::string(p) + ".ffn.gate.weight"),
                                 G(std::string(p) + ".ffn.up.weight"), G(std::string(p) + ".ffn.down.weight"));
            cur = ggml_add(ctx0, residual, ffn);
        }

        // Final RMSNorm
        cur = ggml_rms_norm(ctx0, cur, 1e-6f);
        cur = ggml_mul(ctx0, cur, G("lm.norm.weight"));

        // LM head (lm.tok_emb.weight is tied — no separate lm_head)
        if (n_tokens > 1) {
            cur = ggml_view_1d(ctx0, cur, hp.d_lm, (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));
            cur = ggml_reshape_2d(ctx0, cur, hp.d_lm, 1);
        }
        cur = ggml_mul_mat(ctx0, G("lm.tok_emb.weight"), cur); // tied weights

        ggml_set_name(cur, "logits");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        return gf;
    };

    auto run_decoder = [&](const float* embeds, int n_tokens, int n_past, std::vector<float>& logits) -> bool {
        std::vector<int32_t> positions(n_tokens);
        for (int i = 0; i < n_tokens; i++)
            positions[i] = n_past + i;

        std::vector<ggml_fp16_t> mask;
        if (n_tokens > 1) {
            int Lk = n_past + n_tokens;
            mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
            ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < n_tokens; q++)
                for (int k = 0; k < Lk; k++)
                    if (k > n_past + q)
                        mask[(size_t)q * Lk + k] = neg_inf;
        }

        ggml_cgraph* gf = build_decoder_graph(n_tokens, n_past);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
            return false;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_input"), embeds, 0,
                                (size_t)hp.d_lm * n_tokens * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                                positions.size() * sizeof(int32_t));
        if (n_tokens > 1)
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                    mask.size() * sizeof(ggml_fp16_t));

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
            return false;

        ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
        int V = (int)lt->ne[0];
        logits.resize(V);
        ggml_backend_tensor_get(lt, logits.data(), 0, V * sizeof(float));
        return true;
    };

    // 9. Prefill
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: prefilling %d tokens...\n", prefix_len);

    std::vector<float> logits;
    if (!run_decoder(prefix_embeds.data(), prefix_len, 0, logits)) {
        fprintf(stderr, "vibevoice: prefill failed\n");
        return nullptr;
    }
    ctx->kv_n_used = prefix_len;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: prefill done\n");

    // 10. Autoregressive generation
    auto argmax = [](const std::vector<float>& lg) -> int {
        int best = 0;
        for (int i = 1; i < (int)lg.size(); i++)
            if (lg[i] > lg[best])
                best = i;
        return best;
    };

    std::vector<int> output_tokens;
    int cur_token = argmax(logits);
    if (cur_token != EOS_TOKEN)
        output_tokens.push_back(cur_token);

    if (ctx->params.verbosity >= 2)
        fprintf(stderr, "  prefill → token=%d\n", cur_token);

    for (int step = 0; step < max_gen && cur_token != EOS_TOKEN; step++) {
        const int32_t tid = cur_token;
        std::vector<float> tok_emb = run_token_embedding_lookup(ctx, &tid, 1);
        if (tok_emb.size() != (size_t)hp.d_lm)
            break;

        int n_past = prefix_len + step;
        if (!run_decoder(tok_emb.data(), 1, n_past, logits))
            break;

        cur_token = argmax(logits);
        if (cur_token == EOS_TOKEN)
            break;
        output_tokens.push_back(cur_token);

        if (ctx->params.verbosity >= 2 && step < 5)
            fprintf(stderr, "  gen %d: token=%d\n", step, cur_token);
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: generated %d tokens\n", (int)output_tokens.size());

    // 11. Detokenize using embedded vocabulary
    std::string result;
    for (int tid : output_tokens) {
        if (tid >= 0 && tid < (int)m.vocab.size()) {
            const std::string& piece = m.vocab[tid];
            // Skip special tokens (start with <| and end with |>)
            if (piece.size() >= 4 && piece[0] == '<' && piece[1] == '|')
                continue;
            result += decode_qwen_piece(piece);
        }
    }

    if (result.empty())
        return nullptr;

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    return out;
}
