// vibevoice.cpp — Microsoft VibeVoice-ASR runtime.
//
// Pipeline: 24kHz PCM → acoustic σ-VAE encoder → connector → Qwen2 LM → text
//           24kHz PCM → semantic encoder → connector ↗
//
// Two parallel CNN tokenizer encoders (ConvNeXt blocks with depthwise conv),
// projected to LM space via FC connectors, then autoregressive Qwen2 decoder.

// MSVC's <cmath>/<math.h> only exposes M_PI when this is defined BEFORE
// the very first include in the translation unit (any transitive
// #include <math.h> in vibevoice.h or the ggml headers commits the
// macro state — defining it later is a no-op). Linux/macOS get M_PI
// via libc compat; Windows CI fails with C2065 without this.
#define _USE_MATH_DEFINES

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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
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
    int has_decoder = 0;
    int tts_n_layers = 0; // TTS LM layers (0 = ASR-only model)
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
    // KV cache for LM decoder (positive path)
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;
    // KV cache for TTS negative path (CFG)
    ggml_context* kv_neg_ctx = nullptr;
    ggml_backend_buffer_t kv_neg_buf = nullptr;
    ggml_tensor* kv_neg_k = nullptr;
    ggml_tensor* kv_neg_v = nullptr;
    // Voice prompt (pre-computed KV caches for speaker identity)
    struct voice_prompt {
        int lm_seq_len = 0;
        int tts_seq_len = 0;
        int neg_lm_seq_len = 0;
        int neg_tts_seq_len = 0;
        std::map<std::string, ggml_tensor*> tensors;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
    } voice;
    std::vector<uint8_t> compute_meta;
    // Cached prediction head graph (reused across DPM steps)
    ggml_context* pred_graph_ctx = nullptr;
    ggml_cgraph* pred_graph = nullptr;
    int pred_graph_n_frames = 0;
    std::vector<uint8_t> pred_graph_meta;
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
    p.tts_steps = 20;
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
    hp.has_decoder = core_gguf::kv_u32(gctx, "vibevoice.has_decoder", 0);
    hp.tts_n_layers = core_gguf::kv_u32(gctx, "vibevoice.tts_n_layers", 0);

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
    if (!core_gguf::load_weights(path_model, ctx->backend, "vibevoice", wl)) {
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

    if (params.verbosity >= 1) {
        const char* be_name = ggml_backend_name(ctx->backend);
        fprintf(stderr, "vibevoice: loaded %zu tensors (backend: %s)\n", m.tensors.size(), be_name);
    }

    return ctx;
}

// ===========================================================================
// Free
// ===========================================================================

extern "C" void vibevoice_free(struct vibevoice_context* ctx) {
    if (!ctx)
        return;
    if (ctx->pred_graph_ctx)
        ggml_free(ctx->pred_graph_ctx);
    if (ctx->kv_neg_buf)
        ggml_backend_buffer_free(ctx->kv_neg_buf);
    if (ctx->kv_neg_ctx)
        ggml_free(ctx->kv_neg_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->voice.buf)
        ggml_backend_buffer_free(ctx->voice.buf);
    if (ctx->voice.ctx)
        ggml_free(ctx->voice.ctx);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->weight_ctx)
        ggml_free(ctx->weight_ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    // Free the primary backend last — buffers above were allocated against it,
    // and on Metal an unreleased backend leaves the residency set live and
    // trips ggml_metal_rsets_free's assert at process exit.
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
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
//
// Uses ggml_conv_1d (with transpose) — this works on ALL backends including
// Metal/CUDA/Vulkan via im2col. The conv_1d_cf direct kernel is CPU-only
// and saves ~2 transposes per call, but im2col+transpose is fast on GPU.
static ggml_tensor* build_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride) {
    int K = (int)w->ne[0];
    int dilation = 1;
    int pad_left = (K - 1) * dilation - (stride - 1); // VibeVoice/EnCodec convention
    if (pad_left < 0)
        pad_left = 0;
    // x is [C, T], transpose to [T, C] for ggml_conv_1d
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [T, C]
    int T_in = (int)x->ne[0];
    // Compute extra right padding for stride alignment
    int pad_right = 0;
    if (stride > 1) {
        double n_frames = (double)(T_in - K + pad_left) / stride + 1.0;
        int ideal_length = ((int)ceil(n_frames) - 1) * stride + (K - pad_left);
        pad_right = ideal_length - T_in;
        if (pad_right < 0)
            pad_right = 0;
    }
    if (pad_left > 0 || pad_right > 0)
        x = ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
    // conv_1d output: [T_out, C_out] → transpose to [C_out, T_out]
    if (b) {
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T_out]
        xt = ggml_add(ctx, xt, b);
        return xt;
    }
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

// Causal depthwise Conv1d using ggml_conv_1d_dw (im2col, works on ALL backends).
// Input/output in [C, T] format.
// Uses transpose + ggml_conv_1d_dw + transpose to stay GPU-compatible.
static ggml_tensor* build_causal_dw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    int K = (int)w->ne[0];
    int pad_left = K - 1;

    // Transpose [C, T] → [T, C] for ggml_conv_1d_dw
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (pad_left > 0)
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d_dw(ctx, w, x, 1, 0, 1);
    // conv_1d_dw returns 3D+ → flatten then transpose back to [C, T]
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
    //
    // Official VibeVoice σ-VAE FFN (modular_vibevoice_tokenizer.py:591-608):
    //     linear1 → GELU → linear2
    // We previously used SiLU here — auditorily this manifested as "crackly"
    // distortion in the decoded audio: latents matched cos≥0.999 vs the official
    // model but the reconstructed waveform amplitude was ~65% of reference and
    // per-frame audio cos was 0.7-0.9.
    residual = x;
    h = build_conv_rms_norm(ctx, x, ffn_norm_w);
    h = ggml_mul_mat(ctx, ffn_up_w, h); // [C, C_ffn] @ [C, T] → [C_ffn, T]
    if (ffn_up_b)
        h = ggml_add(ctx, h, ffn_up_b);
    h = ggml_gelu(ctx, h);
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

static void vibevoice_dump_i32(const char* dir, const char* name, const int32_t* data, size_t n) {
    if (!dir || !dir[0] || !name || !data || n == 0)
        return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f)
        return;
    fwrite(data, sizeof(int32_t), n, f);
    fclose(f);
}

static void vibevoice_dump_f32(const char* dir, const char* name, const float* data, size_t n) {
    if (!dir || !dir[0] || !name || !data || n == 0)
        return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f)
        return;
    fwrite(data, sizeof(float), n, f);
    fclose(f);
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
    const char* dump_dir = getenv("VIBEVOICE_DUMP_DIR");

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

    // 5. Build prompt matching the current HF VibeVoice ASR processor:
    // <|im_start|>system ... <|im_end|>
    // <|im_start|>user
    // <|object_ref_start|><|box_start|>×N<|object_ref_end|>
    // This is a X.XX seconds audio, please transcribe it with these keys: ...
    // <|im_end|>\n
    // <|im_start|>assistant\n   (add_generation_prompt=True)
    const int AUDIO_BOS = 151646;   // <|object_ref_start|>
    const int AUDIO_TOKEN = 151648; // <|box_start|>
    const int AUDIO_EOS = 151647;   // <|object_ref_end|>
    const int EOS_TOKEN = 151643;
    const int IM_START = 151644;
    const int IM_END = 151645;

    // System prompt from transformers/models/vibevoice_asr/convert_vibevoice_asr_to_hf.py
    std::vector<int> system_tokens = {
        IM_START, 8948,   198,                                         // <|im_start|>system\n
        2610,     525,    264,  10950,    17847, 429, 1356, 55136,     // You are a helpful assistant that transcribes
        7699,     1946,   1119, 1467,     2550,  304, 4718, 3561,  13, // audio input into text output in JSON format.
        198,      IM_END, 198,  IM_START, 872,   198                   // \n<|im_end|>\n<|im_start|>user\n
    };
    // After audio placeholder tokens:
    // "\nThis is a XX.XX seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n"
    float dur = n_samples / 24000.0f;
    // Duration is inserted as plain text before tokenization by the HF
    // processor. For Qwen2.5 digits and '.' map to stable single-char ids.
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
        IM_END,   198,                                                // <|im_end|>\n
        IM_START, 77091, 198 // <|im_start|>assistant\n (add_generation_prompt=True)
    };
    suffix_tokens.insert(suffix_tokens.end(), suffix_tail.begin(), suffix_tail.end());

    // Build full token sequence
    std::vector<int> prompt_tokens;
    prompt_tokens.insert(prompt_tokens.end(), system_tokens.begin(), system_tokens.end());
    prompt_tokens.push_back(AUDIO_BOS);
    int speech_start_pos = (int)prompt_tokens.size();
    for (int i = 0; i < T_audio; i++)
        prompt_tokens.push_back(AUDIO_TOKEN);
    prompt_tokens.push_back(AUDIO_EOS);
    prompt_tokens.insert(prompt_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
    int prefix_len = (int)prompt_tokens.size();

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: prompt: %d tokens (speech at %d-%d)\n", prefix_len, speech_start_pos,
                speech_start_pos + T_audio - 1);

    // 6. Embed all tokens, then replace speech positions with speech features.
    // Use ggml_get_rows so quantized token embedding tables are handled by the backend.
    std::vector<int32_t> prompt_ids(prompt_tokens.begin(), prompt_tokens.end());
    vibevoice_dump_i32(dump_dir, "prompt_ids", prompt_ids.data(), prompt_ids.size());
    vibevoice_dump_f32(dump_dir, "speech_features", speech_features.data(), speech_features.size());
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
    vibevoice_dump_f32(dump_dir, "prefix_embeds", prefix_embeds.data(), prefix_embeds.size());

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
                ggml_tensor* v_b = G(std::string(p) + ".attn.v_proj.bias");

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
                if (v_b)
                    V = ggml_add(ctx0, V, v_b);

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

        // LM head: VibeVoice-ASR-7B has a SEPARATE lm_head.weight (not tied to tok_emb).
        // Fall back to tok_emb if lm_head is absent (older 1.5B converts may tie).
        if (n_tokens > 1) {
            cur = ggml_view_1d(ctx0, cur, hp.d_lm, (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));
            cur = ggml_reshape_2d(ctx0, cur, hp.d_lm, 1);
        }
        ggml_tensor* lm_head_w = G("lm_head.weight");
        if (!lm_head_w)
            lm_head_w = G("lm.tok_emb.weight");
        cur = ggml_mul_mat(ctx0, lm_head_w, cur);

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
    vibevoice_dump_f32(dump_dir, "prefill_logits", logits.data(), logits.size());

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

    // Qwen2.5 chat template's natural assistant-turn stop is <|im_end|> (IM_END).
    // <|endoftext|> (EOS_TOKEN) is end-of-document, not end-of-turn — the model
    // emits IM_END at the end of its response. Stopping only on EOS_TOKEN runs
    // decode to max_gen for every call (cf. gemma4 same-bug LEARNINGS #9).
    auto is_stop = [&](int tok) { return tok == IM_END || tok == EOS_TOKEN; };

    std::vector<int> output_tokens;
    int cur_token = argmax(logits);
    if (!is_stop(cur_token))
        output_tokens.push_back(cur_token);

    if (ctx->params.verbosity >= 2)
        fprintf(stderr, "  prefill → token=%d\n", cur_token);

    for (int step = 0; step < max_gen && !is_stop(cur_token); step++) {
        const int32_t tid = cur_token;
        std::vector<float> tok_emb = run_token_embedding_lookup(ctx, &tid, 1);
        if (tok_emb.size() != (size_t)hp.d_lm)
            break;

        int n_past = prefix_len + step;
        if (!run_decoder(tok_emb.data(), 1, n_past, logits))
            break;
        if (step < 4) {
            char name[64];
            snprintf(name, sizeof(name), "step_%03d_logits", step);
            vibevoice_dump_f32(dump_dir, name, logits.data(), logits.size());
        }

        cur_token = argmax(logits);
        if (is_stop(cur_token))
            break;
        output_tokens.push_back(cur_token);

        if (ctx->params.verbosity >= 2 && step < 5)
            fprintf(stderr, "  gen %d: token=%d\n", step, cur_token);
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: generated %d tokens\n", (int)output_tokens.size());
    if (!output_tokens.empty()) {
        std::vector<int32_t> output_ids(output_tokens.begin(), output_tokens.end());
        vibevoice_dump_i32(dump_dir, "generated_ids", output_ids.data(), output_ids.size());
    }

    // Debug: dump first 20 generated tokens
    if (getenv("VIBEVOICE_DEBUG")) {
        fprintf(stderr, "vibevoice: first 20 tokens: [");
        for (int i = 0; i < std::min((int)output_tokens.size(), 20); i++) {
            int tid = output_tokens[i];
            const char* piece = (tid >= 0 && tid < (int)m.vocab.size()) ? m.vocab[tid].c_str() : "?";
            fprintf(stderr, "%s%d(%s)", i ? ", " : "", tid, piece);
        }
        fprintf(stderr, "]\n");

        // Also dump logits from first decode step for comparison
        fprintf(stderr, "vibevoice: first-step logits top-5: ");
        // Re-run first step just for logging if we kept logits
    }

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

    if (result.empty()) {
        if (getenv("VIBEVOICE_DEBUG")) {
            fprintf(stderr, "vibevoice: result is EMPTY after detokenization (all tokens were special)\n");
        }
        return nullptr;
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    return out;
}

// ===========================================================================
// TTS: Flow Matching + σ-VAE Decoder
// ===========================================================================

// ── Byte-level Qwen2 tokenizer (encode direction) ──────────────────────────

// GPT-2 byte encoder: maps raw byte (0-255) → unicode codepoint used in vocab.
static const std::vector<int>& qwen_byte_encoder() {
    static std::vector<int> enc(256, -1);
    static bool initialized = false;
    if (initialized)
        return enc;

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
        bool found = false;
        for (int x : bs)
            if (x == b) {
                found = true;
                break;
            }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i)
        enc[bs[i]] = cs[i];
    initialized = true;
    return enc;
}

// Greedy longest-match tokenizer: convert text to byte-encoded string,
// then greedily match the longest vocab entry at each position.
// This approximates BPE well for common words (exact match) and falls
// back to single-byte tokens for unknown substrings.
static std::vector<int32_t> tokenize_text_greedy(const vibevoice_model& m, const char* text) {
    // Build vocab lookup (once)
    static std::map<std::string, int> vocab_map;
    static int max_token_len = 0;
    static bool built = false;
    if (!built && !m.vocab.empty()) {
        for (int i = 0; i < (int)m.vocab.size(); i++) {
            if (!m.vocab[i].empty())
                vocab_map[m.vocab[i]] = i;
            if ((int)m.vocab[i].size() > max_token_len)
                max_token_len = (int)m.vocab[i].size();
        }
        built = true;
    }

    // Convert text bytes to GPT-2 byte-encoded string
    const auto& enc = qwen_byte_encoder();
    std::string encoded;
    for (const uint8_t* p = (const uint8_t*)text; *p; p++) {
        int cp = enc[*p];
        if (cp < 0)
            continue;
        if (cp < 0x80) {
            encoded += (char)cp;
        } else if (cp < 0x800) {
            encoded += (char)(0xC0 | (cp >> 6));
            encoded += (char)(0x80 | (cp & 0x3F));
        } else {
            encoded += (char)(0xE0 | (cp >> 12));
            encoded += (char)(0x80 | ((cp >> 6) & 0x3F));
            encoded += (char)(0x80 | (cp & 0x3F));
        }
    }

    // Greedy longest match
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < encoded.size()) {
        int best_len = 0;
        int best_id = -1;
        int try_len = std::min(max_token_len, (int)(encoded.size() - pos));
        for (int len = try_len; len >= 1; len--) {
            auto it = vocab_map.find(encoded.substr(pos, len));
            if (it != vocab_map.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            pos += best_len;
        } else {
            pos++; // skip unknown byte
        }
    }
    return ids;
}

// ── Gaussian noise ──────────────────────────────────────────────────────────

// Mersenne Twister MT19937 — matches PyTorch's torch.manual_seed()
struct mt19937_state {
    uint32_t mt[624];
    int mti;
};

static void mt19937_seed(mt19937_state& s, uint32_t seed) {
    s.mt[0] = seed;
    for (int i = 1; i < 624; i++)
        s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + (uint32_t)i;
    s.mti = 624;
}

static uint32_t mt19937_next(mt19937_state& s) {
    if (s.mti >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (s.mt[i] & 0x80000000u) | (s.mt[(i + 1) % 624] & 0x7FFFFFFFu);
            s.mt[i] = s.mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1)
                s.mt[i] ^= 0x9908B0DFu;
        }
        s.mti = 0;
    }
    uint32_t y = s.mt[s.mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= (y >> 18);
    return y;
}

// Generate standard normal (Gaussian) noise matching PyTorch's CPU
// torch.randn(float32, contiguous, numel >= 16) path.
//
// PyTorch first fills the tensor with uniform floats in [0, 1), using the low
// 24 bits from its MT19937 engine, then transforms every 16-float block:
//   out[0..7]  = sqrt(-2 log(1-u[0..7])) * cos(2*pi*u[8..15])
//   out[8..15] = sqrt(-2 log(1-u[0..7])) * sin(2*pi*u[8..15])
// This exact trajectory matters for VibeVoice TTS: different valid Gaussian
// generators land on different diffusion paths, and some paths create hot
// first-frame latents that decode as audible onset clicks.
static inline float mt_uniform_torch_float(mt19937_state& rng) {
    return (float)(mt19937_next(rng) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

static void torch_normal_fill_16(float* data) {
    for (int j = 0; j < 8; j++) {
        const float u1 = 1.0f - data[j]; // [0, 1) -> (0, 1] for log.
        const float u2 = data[j + 8];
        const float radius = sqrtf(-2.0f * logf(u1));
        const float theta = 2.0f * (float)M_PI * u2;
        data[j] = radius * cosf(theta);
        data[j + 8] = radius * sinf(theta);
    }
}

static void fill_gaussian_noise(float* data, int n, mt19937_state& rng) {
    if (n <= 0)
        return;

    if (n < 16) {
        float tmp[16];
        for (int i = 0; i < 16; i++)
            tmp[i] = mt_uniform_torch_float(rng);
        torch_normal_fill_16(tmp);
        memcpy(data, tmp, (size_t)n * sizeof(float));
        return;
    }

    for (int i = 0; i < n; i++)
        data[i] = mt_uniform_torch_float(rng);

    int i = 0;
    for (; i <= n - 16; i += 16)
        torch_normal_fill_16(data + i);

    if (i < n) {
        // PyTorch recomputes the final overlapping block when numel is not a
        // multiple of 16. VibeVoice's vae_dim is currently 64, but keep the
        // helper faithful for other latent widths.
        float* tail = data + n - 16;
        for (int j = 0; j < 16; j++)
            tail[j] = mt_uniform_torch_float(rng);
        torch_normal_fill_16(tail);
    }
}

// Legacy overload for backward compatibility
static void fill_gaussian_noise(float* data, int n, uint32_t seed) {
    mt19937_state rng;
    mt19937_seed(rng, seed);
    fill_gaussian_noise(data, n, rng);
}

// ── Sinusoidal timestep embedding ───────────────────────────────────────────

static void compute_sinusoidal_embed(float t, float* out, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        float arg = t * freq;
        out[i] = cosf(arg);
        out[half + i] = sinf(arg);
    }
}

// ── DDIM scheduler ──────────────────────────────────────────────────────────

struct ddim_schedule {
    int num_train_steps;
    int num_inference_steps;
    std::vector<float> alphas_cumprod;
    std::vector<int> timesteps;
};

static ddim_schedule make_ddim_schedule(int num_inference_steps) {
    ddim_schedule s;
    s.num_train_steps = 1000;
    s.num_inference_steps = num_inference_steps;

    // Cosine beta schedule with beta clipping (matching diffusers DPMSolverMultistepScheduler).
    // beta_t = min(1 - alpha_bar(t+1)/alpha_bar(t), 0.999)
    // alpha_bar(t) = cos²((t/T + s) / (1+s) * π/2) where s=0.008
    float offset = 0.008f;
    s.alphas_cumprod.resize(s.num_train_steps);
    auto alpha_bar = [&](float t) -> float {
        float frac = (t + offset) / (1.0f + offset);
        float val = cosf(frac * (float)M_PI * 0.5f);
        return val * val;
    };
    float a_prod = 1.0f;
    for (int i = 0; i < s.num_train_steps; i++) {
        float t1 = (float)i / (float)s.num_train_steps;
        float t2 = (float)(i + 1) / (float)s.num_train_steps;
        float beta = 1.0f - alpha_bar(t2) / alpha_bar(t1);
        if (beta > 0.999f)
            beta = 0.999f; // critical: clip beta
        a_prod *= (1.0f - beta);
        s.alphas_cumprod[i] = a_prod;
    }

    // Match diffusers DPMSolverMultistepScheduler.set_timesteps with
    // timestep_spacing="linspace" (the default the official model uses):
    //   ts = linspace(0, num_train_steps - 1, num_inference_steps + 1).round()[::-1][:-1]
    // For 20 steps over 1000 this produces [999, 949, 899, ..., 50] — note the LAST
    // timestep is 50, NOT 0. The final cleanup to t=0 is implicit via the appended
    // sigma=0 (final_sigmas_type="zero"); see dpm_first_order_update for that path.
    s.timesteps.resize(num_inference_steps);
    for (int i = 0; i < num_inference_steps; i++) {
        s.timesteps[i] =
            (int)roundf((float)(s.num_train_steps - 1) * (float)(num_inference_steps - i) / (float)num_inference_steps);
    }

    return s;
}

// Convert v-prediction to x0 prediction (sample prediction)
static void v_to_x0(const float* x_t, const float* v, float* x0, int n, float alpha_t) {
    float sa = sqrtf(alpha_t);
    float s1ma = sqrtf(1.0f - alpha_t);
    for (int i = 0; i < n; i++)
        x0[i] = sa * x_t[i] - s1ma * v[i];
}

// DPM-Solver++ 1st order update.
// Uses diffusers convention: alpha_t = sqrt(alpha_prod_t), sigma_t = sqrt(1 - alpha_prod_t).
static void dpm_first_order(const ddim_schedule& sched, int step_idx, float* x, const float* x0_pred, int n) {
    int t = sched.timesteps[step_idx];
    int t_prev = (step_idx + 1 < sched.num_inference_steps) ? sched.timesteps[step_idx + 1] : -1;
    float sigma_t = sqrtf(1.0f - sched.alphas_cumprod[t]);
    float sigma_prev = (t_prev >= 0) ? sqrtf(1.0f - sched.alphas_cumprod[t_prev]) : 0.0f;
    float alpha_t = sqrtf(sched.alphas_cumprod[t]);
    float alpha_prev = (t_prev >= 0) ? sqrtf(sched.alphas_cumprod[t_prev]) : 1.0f;
    float lambda_t = logf(alpha_t / sigma_t);
    float lambda_prev = (t_prev >= 0) ? logf(alpha_prev / sigma_prev) : 20.0f;
    float h = lambda_prev - lambda_t;
    // DPM-Solver++: x_{s} = (sigma_s/sigma_t)*x_t - alpha_s*(exp(-h)-1)*x0
    for (int i = 0; i < n; i++)
        x[i] = (sigma_prev / sigma_t) * x[i] - alpha_prev * (expf(-h) - 1.0f) * x0_pred[i];
}

// DPM-Solver++ 2nd order midpoint update.
static void dpm_second_order(const ddim_schedule& sched, int step_idx, float* x, const float* x0_cur,
                             const float* x0_prev, int prev_step_idx, int n) {
    int t = sched.timesteps[step_idx];
    int t_prev = (step_idx + 1 < sched.num_inference_steps) ? sched.timesteps[step_idx + 1] : -1;
    int s = sched.timesteps[prev_step_idx];
    float sigma_t = sqrtf(1.0f - sched.alphas_cumprod[t]);
    float sigma_s = sqrtf(1.0f - sched.alphas_cumprod[s]);
    float sigma_prev = (t_prev >= 0) ? sqrtf(1.0f - sched.alphas_cumprod[t_prev]) : 0.0f;
    float alpha_t = sqrtf(sched.alphas_cumprod[t]);
    float alpha_s = sqrtf(sched.alphas_cumprod[s]);
    float alpha_prev = (t_prev >= 0) ? sqrtf(sched.alphas_cumprod[t_prev]) : 1.0f;
    float lambda_t = logf(alpha_t / sigma_t);                                  // current (source)
    float lambda_s = logf(alpha_s / sigma_s);                                  // previous step
    float lambda_prev = (t_prev >= 0) ? logf(alpha_prev / sigma_prev) : 20.0f; // target
    float h = lambda_prev - lambda_t;                                          // h = lambda_target - lambda_current
    float h_0 = lambda_t - lambda_s;                                           // h_0 = lambda_current - lambda_previous
    float r = h_0 / h;                                                         // r0 = h_0 / h (positive)
    float eh = expf(-h) - 1.0f;
    for (int i = 0; i < n; i++) {
        float D0 = x0_cur[i];
        float D1 = (1.0f / r) * (x0_cur[i] - x0_prev[i]);
        // midpoint: x = ratio*x - alpha*(exp(-h)-1)*D0 - 0.5*alpha*(exp(-h)-1)*D1
        x[i] = (sigma_prev / sigma_t) * x[i] - alpha_prev * eh * D0 - 0.5f * alpha_prev * eh * D1;
    }
}

// ── Build prediction head graph ─────────────────────────────────────────────

// One denoising step, batched over n_frames.
// Inputs: noisy [vae_dim, n_frames], t_sinusoidal [256], condition [d_lm, n_frames]
// Output: predicted_eps [vae_dim, n_frames]
// Forward declaration
static ggml_cgraph* build_pred_head_graph_impl(vibevoice_context* ctx, int n_frames, std::vector<uint8_t>& meta_buf);

// True when the active backend is Metal — used to skip optimisations
// that rely on graph reuse across ggml_backend_sched_reset(), which is
// currently broken on the Metal scheduler (see get_pred_head_graph).
//
// Metal devices register with names like "MTL0", "MTL1" — not "Metal".
// Match the "MTL" prefix; CPU is "CPU", CUDA is "CUDA0", Vulkan is
// "Vulkan0", so this is unambiguous.
static bool backend_is_metal(ggml_backend_t b) {
    if (!b)
        return false;
    const char* name = ggml_backend_name(b);
    return name && std::strncmp(name, "MTL", 3) == 0;
}

// Get or build the cached prediction head graph.
// The graph structure is the same for a given n_frames — only input
// tensor data changes. Building once and reusing with
// ggml_backend_sched_reset saves ~30% per synthesis on CPU/CUDA/Vulkan.
//
// Metal exception: after ggml_backend_sched_reset(), reusing the same
// graph fires GGML_ASSERT(src_backend_id != -1) inside
// ggml_backend_sched_split_graph — some tensor view's buffer-id linkage
// doesn't survive the reset on the Metal scheduler specifically.
// Reproducible with the streaming Q4_0 we built locally AND with the
// canonical Q4_K from cstr/VibeVoice-7B-GGUF, so it's the cache + Metal
// scheduler interaction, not the model file. For Metal we fall back to
// rebuild-each-call (matches what build_decoder_graph does) — a few
// percent slower per diffusion sub-step but TTS actually runs.
//
// The proper long-term fix is upstream ggml — make the scheduler's
// view-tensor backend mapping survive sched_reset() — at which point
// this branch can collapse back to always-cache.
static ggml_cgraph* get_pred_head_graph(vibevoice_context* ctx, int n_frames) {
    const bool reuse_ok = !backend_is_metal(ctx->backend);

    if (reuse_ok && ctx->pred_graph && ctx->pred_graph_n_frames == n_frames)
        return ctx->pred_graph;

    if (ctx->pred_graph_ctx) {
        ggml_free(ctx->pred_graph_ctx);
        ctx->pred_graph_ctx = nullptr;
        ctx->pred_graph = nullptr;
    }

    // On Metal: build fresh into the SHARED compute_meta buffer (same as
    // every other builder that survives sched_reset on Metal — run_dec,
    // run_connector_stage, build_decoder_graph). Avoids the
    // src_backend_id=-1 assert that fires when sched_reset+alloc_graph
    // operate on a graph that lives in its own dedicated buffer.
    //
    // On CPU/CUDA the dedicated pred_graph_meta keeps the graph cached
    // across diffusion sub-steps — saves ~25% per synthesis.
    std::vector<uint8_t>* meta = reuse_ok ? &ctx->pred_graph_meta : &ctx->compute_meta;
    if (reuse_ok && meta->empty())
        meta->resize(ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(4096, false));

    ctx->pred_graph = build_pred_head_graph_impl(ctx, n_frames, *meta);
    ctx->pred_graph_n_frames = n_frames;
    return ctx->pred_graph;
}

static ggml_cgraph* build_pred_head_graph_impl(vibevoice_context* ctx, int n_frames, std::vector<uint8_t>& meta_buf) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    int vae_dim = hp.vae_dim_acoustic;
    int d_lm = hp.d_lm;

    size_t mem = meta_buf.size();
    ggml_init_params ip = {mem, meta_buf.data(), true};
    ctx->pred_graph_ctx = ggml_init(ip);
    ggml_context* ctx0 = ctx->pred_graph_ctx;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Rest of the graph building follows below...
    // (The old build_pred_head_graph body, minus the first few lines)

    ggml_tensor* noisy = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vae_dim, n_frames);
    ggml_set_name(noisy, "pred_noisy");
    ggml_set_input(noisy);

    ggml_tensor* t_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_sin, "pred_t_sin");
    ggml_set_input(t_sin);

    ggml_tensor* condition = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_lm, n_frames);
    ggml_set_name(condition, "pred_condition");
    ggml_set_input(condition);

    // Time embedding MLP
    ggml_tensor* t_emb = ggml_mul_mat(ctx0, G("pred.t_emb.0.weight"), t_sin);
    t_emb = ggml_silu(ctx0, t_emb);
    t_emb = ggml_mul_mat(ctx0, G("pred.t_emb.2.weight"), t_emb);

    ggml_tensor* x = ggml_mul_mat(ctx0, G("pred.noisy_proj.weight"), noisy);
    ggml_tensor* cond = ggml_mul_mat(ctx0, G("pred.cond.weight"), condition);
    ggml_tensor* c = ggml_add(ctx0, cond, t_emb);

    for (int i = 0; i < 4; i++) {
        char base[64];
        snprintf(base, sizeof(base), "pred.layers.%d", i);
        ggml_tensor* c_silu = ggml_silu(ctx0, c);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, G(std::string(base) + ".adaln.weight"), c_silu);
        size_t nb1 = adaln_out->nb[1];
        ggml_tensor* shift = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, 0);
        ggml_tensor* scale = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, (size_t)d_lm * sizeof(float));
        ggml_tensor* gate = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, (size_t)2 * d_lm * sizeof(float));
        ggml_tensor* h = ggml_rms_norm(ctx0, x, 1e-5f);
        h = ggml_mul(ctx0, h, G(std::string(base) + ".norm.weight"));
        ggml_tensor* h_scaled = ggml_mul(ctx0, h, scale);
        h = ggml_add(ctx0, h, h_scaled);
        h = ggml_add(ctx0, h, shift);
        h = core_ffn::swiglu(ctx0, h, G(std::string(base) + ".ffn.gate_proj.weight"),
                             G(std::string(base) + ".ffn.up_proj.weight"),
                             G(std::string(base) + ".ffn.down_proj.weight"));
        h = ggml_mul(ctx0, h, gate);
        x = ggml_add(ctx0, x, h);
    }

    {
        ggml_tensor* c_silu_f = ggml_silu(ctx0, c);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, G("pred.final.adaln.weight"), c_silu_f);
        size_t nb1_f = adaln_out->nb[1];
        ggml_tensor* shift = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1_f, 0);
        ggml_tensor* scale = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1_f, (size_t)d_lm * sizeof(float));
        ggml_tensor* h = ggml_rms_norm(ctx0, x, 1e-5f);
        ggml_tensor* h_scaled = ggml_mul(ctx0, h, scale);
        h = ggml_add(ctx0, h, h_scaled);
        h = ggml_add(ctx0, h, shift);
        ggml_tensor* output = ggml_mul_mat(ctx0, G("pred.final.linear.weight"), h);
        ggml_set_name(output, "pred_output");
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
    }

    return gf;
}

// Original function signature preserved for backward compatibility
static ggml_cgraph* build_pred_head_graph(vibevoice_context* ctx, int n_frames) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    int vae_dim = hp.vae_dim_acoustic;
    int d_lm = hp.d_lm;

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* noisy = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vae_dim, n_frames);
    ggml_set_name(noisy, "pred_noisy");
    ggml_set_input(noisy);

    ggml_tensor* t_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_sin, "pred_t_sin");
    ggml_set_input(t_sin);

    ggml_tensor* condition = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_lm, n_frames);
    ggml_set_name(condition, "pred_condition");
    ggml_set_input(condition);

    // Time embedding MLP: sinusoidal[256] → Linear → SiLU → Linear → [d_lm]
    ggml_tensor* t_emb = ggml_mul_mat(ctx0, G("pred.t_emb.0.weight"), t_sin);
    t_emb = ggml_silu(ctx0, t_emb);
    t_emb = ggml_mul_mat(ctx0, G("pred.t_emb.2.weight"), t_emb);

    // Project noisy input: [vae_dim, n_frames] → [d_lm, n_frames]
    ggml_tensor* x = ggml_mul_mat(ctx0, G("pred.noisy_proj.weight"), noisy);

    // Project condition: [d_lm, n_frames] → [d_lm, n_frames]
    ggml_tensor* cond = ggml_mul_mat(ctx0, G("pred.cond.weight"), condition);

    // Combined conditioning: c = cond_proj(condition) + t_embedder(timestep)
    // t_emb is [d_lm] (shared across frames), cond is [d_lm, n_frames]
    // t_emb broadcasts over n_frames dimension
    ggml_tensor* c = ggml_add(ctx0, cond, t_emb);

    // NOTE: x = noisy_proj only. Condition goes into AdaLN via c, NOT added to x.

    // 4 AdaLN + SwiGLU layers
    for (int i = 0; i < 4; i++) {
        char base[64];
        snprintf(base, sizeof(base), "pred.layers.%d", i);

        // AdaLN modulation: SiLU(c) → adaln.w → [3*d_lm, n_frames]
        ggml_tensor* c_silu = ggml_silu(ctx0, c);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, G(std::string(base) + ".adaln.weight"), c_silu);
        // Split along ne[0]: shift=[d_lm, n_frames], scale=[d_lm, n_frames], gate=[d_lm, n_frames]
        size_t nb1 = adaln_out->nb[1]; // stride between frames
        ggml_tensor* shift = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, 0);
        ggml_tensor* scale = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, (size_t)d_lm * sizeof(float));
        ggml_tensor* gate = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1, (size_t)2 * d_lm * sizeof(float));

        // RMSNorm + modulate: h = norm(x) * (1 + scale) + shift
        ggml_tensor* h = ggml_rms_norm(ctx0, x, 1e-6f);
        h = ggml_mul(ctx0, h, G(std::string(base) + ".norm.weight"));
        ggml_tensor* h_scaled = ggml_mul(ctx0, h, scale);
        h = ggml_add(ctx0, h, h_scaled);
        h = ggml_add(ctx0, h, shift);

        // SwiGLU FFN
        h = core_ffn::swiglu(ctx0, h, G(std::string(base) + ".ffn.gate_proj.weight"),
                             G(std::string(base) + ".ffn.up_proj.weight"),
                             G(std::string(base) + ".ffn.down_proj.weight"));

        // Gate (DiT-style, no sigmoid) — per-frame modulation
        h = ggml_mul(ctx0, h, gate);

        x = ggml_add(ctx0, x, h);
    }

    // Final layer: SiLU(c) → AdaLN → project to vae_dim
    {
        ggml_tensor* c_silu_f = ggml_silu(ctx0, c);
        ggml_tensor* adaln_out = ggml_mul_mat(ctx0, G("pred.final.adaln.weight"), c_silu_f);
        size_t nb1_f = adaln_out->nb[1];
        ggml_tensor* shift = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1_f, 0);
        ggml_tensor* scale = ggml_view_2d(ctx0, adaln_out, d_lm, n_frames, nb1_f, (size_t)d_lm * sizeof(float));

        ggml_tensor* h = ggml_rms_norm(ctx0, x, 1e-6f);
        ggml_tensor* h_scaled = ggml_mul(ctx0, h, scale);
        h = ggml_add(ctx0, h, h_scaled);
        h = ggml_add(ctx0, h, shift);

        ggml_tensor* output = ggml_mul_mat(ctx0, G("pred.final.linear.weight"), h);
        ggml_set_name(output, "pred_output");
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
    }

    return gf;
}

// ── Transposed causal Conv1d ────────────────────────────────────────────────

// Causal transposed conv1d for decoder upsampling.
// Input/output in [C, T] format. Upsamples by 'stride'.
// Output trimmed to T_in * stride (removes K - stride from end).
static ggml_tensor* build_transposed_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                            int stride) {
    int T_in = (int)x->ne[1];

    // Transpose to [T, C_in] for ggml conv ops
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    // ggml_conv_transpose_1d: a=[K, C_out, C_in], b=[T, C_in] → [T_out, C_out, 1, 1]
    x = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);

    // Reshape 4D → 2D
    x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);

    // Trim: raw output = (T_in-1)*stride + K, target = T_in*stride
    int K = (int)w->ne[0];
    int T_out_raw = (T_in - 1) * stride + K;
    int T_target = T_in * stride;
    if (T_out_raw > T_target) {
        int C_out = (int)x->ne[1];
        x = ggml_view_2d(ctx, x, T_target, C_out, x->nb[1], 0);
        x = ggml_cont(ctx, x);
    }

    // Transpose back to [C_out, T_out]
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    if (b)
        x = ggml_add(ctx, x, b);

    return x;
}

// ── Build σ-VAE decoder graph ───────────────────────────────────────────────

// Decode acoustic latents [vae_dim, T_frames] → audio [1, T_audio].
static ggml_cgraph* build_vae_decoder_graph(vibevoice_context* ctx, int n_frames) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.vae_dim_acoustic, n_frames);
    ggml_set_name(inp, "dec_latent");
    ggml_set_input(inp);

    ggml_tensor* h = inp;

    // Decoder depths = reversed encoder depths (e.g. [3,3,3,3,3,3,8] → [8,3,3,3,3,3,3])
    std::vector<int> depths(hp.encoder_depths.rbegin(), hp.encoder_depths.rend());
    // Upsample strides = encoder_ratios in original order [8,5,5,4,2,2]
    const auto& ratios = hp.encoder_ratios;

    // 1. Stem conv (us.0): Conv1d(vae_dim → C_max, K=7, stride=1)
    h = build_causal_conv1d(ctx0, h, G("at_dec.us.0.0.conv.weight"), G("at_dec.us.0.0.conv.bias"), 1);

    // 2. Stage 0: ConvNeXt blocks at full channel width
    for (int bi = 0; bi < depths[0]; bi++) {
        char base[128];
        snprintf(base, sizeof(base), "at_dec.s.0.%d", bi);
        h = build_block1d(ctx0, h, G(std::string(base) + ".norm.weight"), G(std::string(base) + ".dw_conv.weight"),
                          G(std::string(base) + ".dw_conv.bias"), G(std::string(base) + ".gamma"),
                          G(std::string(base) + ".ffn_ln.weight"), G(std::string(base) + ".ffn.up.weight"),
                          G(std::string(base) + ".ffn.up.bias"), G(std::string(base) + ".ffn.down.weight"),
                          G(std::string(base) + ".ffn.down.bias"), G(std::string(base) + ".ffn_gamma"));
    }

    // 3. Stages 1-6: transposed conv upsample + ConvNeXt blocks
    int n_upsample_stages = (int)ratios.size(); // 6
    for (int si = 1; si <= n_upsample_stages; si++) {
        char wn[128], bn_str[128];
        snprintf(wn, sizeof(wn), "at_dec.us.%d.0.convtr.weight", si);
        snprintf(bn_str, sizeof(bn_str), "at_dec.us.%d.0.convtr.bias", si);
        int stride = ratios[si - 1];
        h = build_transposed_conv1d(ctx0, h, G(wn), G(bn_str), stride);

        int n_blocks = (si < (int)depths.size()) ? depths[si] : 3;
        for (int bi = 0; bi < n_blocks; bi++) {
            char base[128];
            snprintf(base, sizeof(base), "at_dec.s.%d.%d", si, bi);
            h = build_block1d(ctx0, h, G(std::string(base) + ".norm.weight"), G(std::string(base) + ".dw_conv.weight"),
                              G(std::string(base) + ".dw_conv.bias"), G(std::string(base) + ".gamma"),
                              G(std::string(base) + ".ffn_ln.weight"), G(std::string(base) + ".ffn.up.weight"),
                              G(std::string(base) + ".ffn.up.bias"), G(std::string(base) + ".ffn.down.weight"),
                              G(std::string(base) + ".ffn.down.bias"), G(std::string(base) + ".ffn_gamma"));
        }
    }

    // 4. Head: Conv1d(32 → 1, K=7, stride=1) → mono audio
    h = build_causal_conv1d(ctx0, h, G("at_dec.head.weight"), G("at_dec.head.bias"), 1);

    ggml_set_name(h, "dec_audio");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    return gf;
}

// ── LM hidden state extraction (no KV cache) ───────────────────────────────

// Run text through Qwen2 LM, return hidden states.
// If all_positions=false, returns last token only [d_lm].
// If all_positions=true, returns all tokens [n_tokens * d_lm].
// Single prefill pass with full self-attention (no KV cache needed).
// Run a Qwen2-style transformer prefill (no KV cache, full self-attention) on a stack of
// pre-computed embeddings. Used for the negative-path prefill recipe described in
// microsoft/VibeVoice modeling_vibevoice_streaming_inference.py:
//   neg_base_hidden  = run_qwen2_prefill_no_kv(ctx, embed(IMAGE_PAD), 1, "lm",     n_lm_layers, false)
//   neg_condition    = run_qwen2_prefill_no_kv(ctx, neg_base_hidden + type_text, 1, "tts_lm", tts_n_layers, true)
// Returns hidden states for the LAST token (or all tokens if all_positions=true).
static std::vector<float> run_qwen2_prefill_no_kv(vibevoice_context* ctx, const float* embeds_in, int n_tokens,
                                                  const char* prefix, int n_layers, bool has_final_norm,
                                                  bool all_positions) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* emb_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_lm, n_tokens);
    ggml_set_name(emb_t, "qpf_input");
    ggml_set_input(emb_t);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "qpf_pos");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "qpf_mask");
    ggml_set_input(causal_mask);

    ggml_tensor* cur = emb_t;
    for (int il = 0; il < n_layers; il++) {
        char p[64];
        snprintf(p, sizeof(p), "%s.layers.%d", prefix, il);
        ggml_tensor* residual = cur;
        cur = ggml_rms_norm(ctx0, cur, 1e-6f);
        cur = ggml_mul(ctx0, cur, G(std::string(p) + ".attn_ln.weight"));
        {
            ggml_tensor* Q = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.q_proj.weight"), cur);
            if (auto* qb = G(std::string(p) + ".attn.q_proj.bias"))
                Q = ggml_add(ctx0, Q, qb);
            ggml_tensor* K = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.k_proj.weight"), cur);
            if (auto* kb = G(std::string(p) + ".attn.k_proj.bias"))
                K = ggml_add(ctx0, K, kb);
            ggml_tensor* V = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.v_proj.weight"), cur);
            if (auto* vb = G(std::string(p) + ".attn.v_proj.bias"))
                V = ggml_add(ctx0, V, vb);
            Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, n_tokens);
            K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, n_tokens);
            V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, n_tokens);
            Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f,
                              0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx0, K, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f,
                              0.0f, 1.0f, 0.0f, 0.0f);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
            float scale = 1.0f / sqrtf((float)hp.head_dim);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx0, Q, K, V, causal_mask, scale, 0.0f, 0.0f);
            attn_out = ggml_reshape_2d(ctx0, attn_out, hp.d_lm, n_tokens);
            attn_out = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.o_proj.weight"), attn_out);
            cur = ggml_add(ctx0, residual, attn_out);
        }
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, 1e-6f);
        cur = ggml_mul(ctx0, cur, G(std::string(p) + ".ffn_ln.weight"));
        ggml_tensor* ffn =
            core_ffn::swiglu(ctx0, cur, G(std::string(p) + ".ffn.gate.weight"), G(std::string(p) + ".ffn.up.weight"),
                             G(std::string(p) + ".ffn.down.weight"));
        cur = ggml_add(ctx0, residual, ffn);
    }
    if (has_final_norm) {
        if (auto* nw = G(std::string(prefix) + ".norm.weight")) {
            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            cur = ggml_mul(ctx0, cur, nw);
        }
    }
    if (!all_positions)
        cur = ggml_view_1d(ctx0, cur, hp.d_lm, (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));
    ggml_set_name(cur, "qpf_hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return {};
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qpf_input"), embeds_in, 0,
                            (size_t)hp.d_lm * n_tokens * sizeof(float));
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qpf_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * n_tokens, ggml_fp32_to_fp16(0.0f));
    ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n_tokens; q++)
        for (int k = q + 1; k < n_tokens; k++)
            mask[(size_t)q * n_tokens + k] = neg_inf;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qpf_mask"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return {};
    int out_size = all_positions ? hp.d_lm * n_tokens : hp.d_lm;
    std::vector<float> out(out_size);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "qpf_hidden"), out.data(), 0, (size_t)out_size * sizeof(float));
    return out;
}

static std::vector<float> run_lm_hidden_states(vibevoice_context* ctx, const int32_t* token_ids, int n_tokens,
                                               bool all_positions = false) {
    auto& hp = ctx->model.hp;
    auto& ts = ctx->model.tensors;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    auto embeds = run_token_embedding_lookup(ctx, token_ids, n_tokens);
    if ((int)embeds.size() != hp.d_lm * n_tokens)
        return {};

    size_t mem = ctx->compute_meta.size();
    ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* emb_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_lm, n_tokens);
    ggml_set_name(emb_t, "tts_input");
    ggml_set_input(emb_t);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "tts_pos");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "tts_mask");
    ggml_set_input(causal_mask);

    ggml_tensor* cur = emb_t;

    for (int il = 0; il < hp.n_lm_layers; il++) {
        char p[64];
        snprintf(p, sizeof(p), "lm.layers.%d", il);
        ggml_tensor* residual = cur;

        // Pre-RMSNorm
        cur = ggml_rms_norm(ctx0, cur, 1e-6f);
        cur = ggml_mul(ctx0, cur, G(std::string(p) + ".attn_ln.weight"));

        // Self-attention (full, no KV cache)
        {
            ggml_tensor* Q = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.q_proj.weight"), cur);
            ggml_tensor* q_b = G(std::string(p) + ".attn.q_proj.bias");
            if (q_b)
                Q = ggml_add(ctx0, Q, q_b);

            ggml_tensor* K = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.k_proj.weight"), cur);
            ggml_tensor* k_b = G(std::string(p) + ".attn.k_proj.bias");
            if (k_b)
                K = ggml_add(ctx0, K, k_b);

            ggml_tensor* V = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.v_proj.weight"), cur);
            ggml_tensor* v_b = G(std::string(p) + ".attn.v_proj.bias");
            if (v_b)
                V = ggml_add(ctx0, V, v_b);

            Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, n_tokens);
            K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, n_tokens);
            V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, n_tokens);

            Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f,
                              0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx0, K, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f,
                              0.0f, 1.0f, 0.0f, 0.0f);

            // Permute for flash_attn_ext: [hd, T, n_heads]
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            float scale = 1.0f / sqrtf((float)hp.head_dim);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx0, Q, K, V, causal_mask, scale, 0.0f, 0.0f);
            attn_out = ggml_reshape_2d(ctx0, attn_out, hp.d_lm, n_tokens);
            attn_out = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.o_proj.weight"), attn_out);
            cur = ggml_add(ctx0, residual, attn_out);
        }

        // FFN: RMSNorm + SwiGLU
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, 1e-6f);
        cur = ggml_mul(ctx0, cur, G(std::string(p) + ".ffn_ln.weight"));
        ggml_tensor* ffn =
            core_ffn::swiglu(ctx0, cur, G(std::string(p) + ".ffn.gate.weight"), G(std::string(p) + ".ffn.up.weight"),
                             G(std::string(p) + ".ffn.down.weight"));
        cur = ggml_add(ctx0, residual, ffn);
    }

    // Final RMSNorm (no LM head — we want hidden states)
    // Realtime model's base LM (4 layers) has no final norm
    ggml_tensor* lm_norm_w = G("lm.norm.weight");
    if (lm_norm_w) {
        cur = ggml_rms_norm(ctx0, cur, 1e-6f);
        cur = ggml_mul(ctx0, cur, lm_norm_w);
    }

    // Return last token only, or all tokens
    if (!all_positions)
        cur = ggml_view_1d(ctx0, cur, hp.d_lm, (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));

    ggml_set_name(cur, "tts_hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Run graph
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return {};

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_input"), embeds.data(), 0, embeds.size() * sizeof(float));

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));

    // Causal mask
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * n_tokens, ggml_fp32_to_fp16(0.0f));
    ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n_tokens; q++)
        for (int k = q + 1; k < n_tokens; k++)
            mask[(size_t)q * n_tokens + k] = neg_inf;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_mask"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return {};

    int out_size = all_positions ? hp.d_lm * n_tokens : hp.d_lm;
    std::vector<float> hidden(out_size);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "tts_hidden"), hidden.data(), 0, out_size * sizeof(float));
    return hidden;
}

// ── vibevoice_load_voice ────────────────────────────────────────────────────

extern "C" int vibevoice_load_voice(struct vibevoice_context* ctx, const char* voice_path) {
    if (!ctx || !voice_path)
        return -1;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(voice_path, ctx->backend, "vibevoice-voice", wl)) {
        fprintf(stderr, "vibevoice: failed to load voice prompt '%s'\n", voice_path);
        return -1;
    }

    ctx->voice.ctx = wl.ctx;
    ctx->voice.buf = wl.buf;
    ctx->voice.tensors = wl.tensors;

    // Read metadata
    gguf_context* gctx = core_gguf::open_metadata(voice_path);
    if (gctx) {
        ctx->voice.lm_seq_len = core_gguf::kv_u32(gctx, "voice.lm.seq_len", 0);
        ctx->voice.tts_seq_len = core_gguf::kv_u32(gctx, "voice.tts_lm.seq_len", 0);
        ctx->voice.neg_lm_seq_len = core_gguf::kv_u32(gctx, "voice.neg_lm.seq_len", 0);
        ctx->voice.neg_tts_seq_len = core_gguf::kv_u32(gctx, "voice.neg_tts_lm.seq_len", 0);
        gguf_free(gctx);
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "vibevoice: loaded voice prompt '%s' (%zu tensors, tts_seq=%d)\n", voice_path,
                ctx->voice.tensors.size(), ctx->voice.tts_seq_len);

    return 0;
}

// ── vibevoice_synthesize ────────────────────────────────────────────────────

extern "C" float* vibevoice_synthesize(struct vibevoice_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !text[0])
        return nullptr;

    auto& m = ctx->model;
    auto& hp = m.hp;
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = m.tensors.find(name);
        return it != m.tensors.end() ? it->second : nullptr;
    };

    // Check prerequisites
    if (!hp.has_decoder || !G("at_dec.head.weight")) {
        fprintf(stderr, "vibevoice_synthesize: model lacks decoder tensors (convert with --include-decoder)\n");
        return nullptr;
    }
    if (m.vocab.empty()) {
        fprintf(stderr, "vibevoice_synthesize: model lacks tokenizer (vocab empty)\n");
        return nullptr;
    }
    if (!G("pred.noisy_proj.weight")) {
        fprintf(stderr, "vibevoice_synthesize: model lacks prediction head tensors\n");
        return nullptr;
    }

    int verbosity = ctx->params.verbosity;
    int vae_dim = hp.vae_dim_acoustic;
    int d_lm = hp.d_lm;
    const char* dump_dir = getenv("VIBEVOICE_TTS_DUMP");
    const auto tts_t0 = std::chrono::high_resolution_clock::now();

    // VibeVoice TTS hits Apple's GPU watchdog
    // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) on Metal
    // for both model paths, just at different stages:
    //   - 1.5B/7B base   → LM forward (Q4_K matmul) at frame ~10
    //   - Realtime 0.5B  → VAE decoder for the full latent sequence
    // Both stages submit a single Metal command buffer that holds the
    // GPU long enough for the system to abort the process. ASR works
    // fine on Metal because its hot-path graph is one big prefill, not
    // hundreds of small back-to-back submits, and the VAE decoder is
    // not used for ASR.
    //
    // Real fix needs ggml-metal: faster low-batch Q4_K matmul, or
    // command-buffer splitting in the scheduler. Until then bail with
    // an actionable message; CPU TTS works end-to-end (-ng).
    // Metal bail temporarily disabled for diagnostics — see if pred_head
    // shared-buffer fix unblocks the assert.

    // Detect model type: Realtime (has TTS LM) vs Base (single LM)
    bool is_base_model = (hp.tts_n_layers == 0);

    // 1. Tokenize text (append newline — VibeVoice TTS tokenizer does this)
    std::string text_with_nl = std::string(text) + "\n";
    std::vector<int32_t> text_ids = tokenize_text_greedy(m, text_with_nl.c_str());
    if (text_ids.empty()) {
        fprintf(stderr, "vibevoice TTS: tokenization produced no tokens\n");
        return nullptr;
    }

    if (verbosity >= 1)
        fprintf(stderr, "vibevoice TTS: %d tokens from %d chars (%s model)\n", (int)text_ids.size(), (int)strlen(text),
                is_base_model ? "1.5B/7B base" : "Realtime");

    // ── 1.5B/7B Base Model TTS Path ──
    // Single-LM autoregressive generation: the LM produces text + speech control tokens.
    // When it emits speech_diffusion_id, we run diffusion → audio.
    if (is_base_model) {
        // VibeVoice reuses Qwen2 vision tokens for speech:
        const int SPEECH_START = 151652;     // <|vision_start|>
        const int SPEECH_DIFFUSION = 151654; // <|vision_pad|> — triggers diffusion
        const int SPEECH_END = 151653;       // <|vision_end|>
        const int EOS_TOKEN = 151643;

        // Build TTS prompt:
        // "{system_prompt} Voice input:\n <speech_start> [voice_embeds] <speech_end>
        //  Text input:\n Speaker 1: {text} Speech output:\n <speech_start>"
        std::string sys = " Transform the text provided by various speakers into speech output, utilizing the distinct "
                          "voice of each respective speaker.\n";
        std::string voice_prefix = " Voice input:\n";
        std::string text_prefix = " Text input:\n Speaker 1: ";
        std::string speech_prefix = " Speech output:\n";

        auto sys_ids = tokenize_text_greedy(m, sys.c_str());
        auto vp_ids = tokenize_text_greedy(m, voice_prefix.c_str());
        auto tp_ids = tokenize_text_greedy(m, (text_prefix + text_with_nl).c_str());
        auto sp_ids = tokenize_text_greedy(m, speech_prefix.c_str());

        // If --voice provides a WAV file for the 1.5B model, encode it as voice reference
        // The voice embeds go between <speech_start> and <speech_end> in the voice section
        std::vector<float> voice_embeds;
        int n_voice_frames = 0;

        // Try loading voice audio from the voice GGUF path (reused as audio path for 1.5B)
        // For the 1.5B, --voice points to a reference .wav (not a .gguf KV cache)
        if (ctx->voice.tts_seq_len == 0) {
            // No voice GGUF loaded — check VIBEVOICE_VOICE_AUDIO env for a reference wav
            const char* voice_wav = getenv("VIBEVOICE_VOICE_AUDIO");
            if (voice_wav && voice_wav[0]) {
                // Load 24kHz mono PCM from WAV file
                FILE* fv = fopen(voice_wav, "rb");
                if (fv) {
                    // Skip 44-byte WAV header, read as int16
                    fseek(fv, 44, SEEK_SET);
                    fseek(fv, 0, SEEK_END);
                    long fsize = ftell(fv) - 44;
                    fseek(fv, 44, SEEK_SET);
                    int n_samples_ref = (int)(fsize / 2);
                    std::vector<int16_t> raw16(n_samples_ref);
                    size_t rd = fread(raw16.data(), 2, n_samples_ref, fv);
                    fclose(fv);
                    (void)rd;
                    // Convert to float and resample to 24kHz if needed
                    std::vector<float> ref_pcm(n_samples_ref);
                    for (int i = 0; i < n_samples_ref; i++)
                        ref_pcm[i] = (float)raw16[i] / 32767.0f;

                    // Encode through acoustic + semantic encoders + connectors
                    float* speech_feat =
                        vibevoice_encode_speech(ctx, ref_pcm.data(), n_samples_ref, &n_voice_frames, nullptr);
                    if (speech_feat && n_voice_frames > 0) {
                        voice_embeds.assign(speech_feat, speech_feat + n_voice_frames * d_lm);
                        free(speech_feat);
                        if (verbosity >= 1)
                            fprintf(stderr, "  encoded voice reference: %d frames from %s\n", n_voice_frames,
                                    voice_wav);
                    }
                }
            }
        }

        // Build prompt tokens
        std::vector<int32_t> prompt;
        prompt.insert(prompt.end(), sys_ids.begin(), sys_ids.end());

        int voice_embed_start = -1;
        if (n_voice_frames > 0) {
            // Voice section: " Voice input:\n" + <speech_start> + [placeholders] + <speech_end>
            prompt.insert(prompt.end(), vp_ids.begin(), vp_ids.end());
            prompt.push_back(SPEECH_START);
            voice_embed_start = (int)prompt.size();
            for (int i = 0; i < n_voice_frames; i++)
                prompt.push_back(SPEECH_DIFFUSION); // placeholder tokens for voice frames
            prompt.push_back(SPEECH_END);
        }

        prompt.insert(prompt.end(), tp_ids.begin(), tp_ids.end());
        prompt.insert(prompt.end(), sp_ids.begin(), sp_ids.end());
        prompt.push_back(SPEECH_START);
        int prefix_len = (int)prompt.size();

        if (verbosity >= 1)
            fprintf(stderr, "vibevoice TTS (base): prompt %d tokens (%d voice frames)\n", prefix_len, n_voice_frames);

        // Embed prompt tokens
        auto prefix_embeds = run_token_embedding_lookup(ctx, prompt.data(), prefix_len);
        if ((int)prefix_embeds.size() != prefix_len * d_lm) {
            fprintf(stderr, "vibevoice TTS: embedding failed\n");
            return nullptr;
        }

        // Replace voice placeholder embeddings with actual speech features
        if (voice_embed_start >= 0 && n_voice_frames > 0) {
            for (int i = 0; i < n_voice_frames; i++)
                memcpy(prefix_embeds.data() + (size_t)(voice_embed_start + i) * d_lm,
                       voice_embeds.data() + (size_t)i * d_lm, d_lm * sizeof(float));
        }

        // Allocate KV cache
        int max_gen = ctx->params.max_new_tokens > 0 ? ctx->params.max_new_tokens : 512;
        int max_ctx = prefix_len + max_gen;
        if (!ctx->kv_k || ctx->kv_max_ctx < max_ctx) {
            if (ctx->kv_ctx)
                ggml_free(ctx->kv_ctx);
            if (ctx->kv_buf)
                ggml_backend_buffer_free(ctx->kv_buf);
            int hd = hp.head_dim, nkv = hp.n_kv_heads, nl = hp.n_lm_layers;
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
        }
        ggml_backend_buffer_clear(ctx->kv_buf, 0);

        // Reuse the existing ASR decoder infrastructure
        // build_decoder_graph and run_decoder are defined inside vibevoice_transcribe
        // For the base model TTS, we need the same graph but output hidden states
        // at each step instead of (or in addition to) logits.

        // For simplicity, use the existing transcribe infrastructure:
        // Prefill, then autoregressive generate. On speech_diffusion_id, run diffusion.

        // Prefill
        if (verbosity >= 1)
            fprintf(stderr, "vibevoice TTS (base): prefilling %d tokens...\n", prefix_len);

        std::vector<float> logits;
        // Use the existing build_decoder_graph lambda from vibevoice_transcribe
        // Actually we can't access it since it's inside another function.
        // Let me use run_lm_hidden_states for prefill (no KV cache though).
        // Better: duplicate the decoder graph builder inline.

        // Actually, the simplest approach: call vibevoice_transcribe with a special prompt
        // that triggers TTS generation. But that's hacky.

        // For now: use the prediction head + DPM solver from the Realtime path.
        // The base model's LM produces hidden states. We run the LM autoregressively,
        // and at each step check if the argmax token is SPEECH_DIFFUSION.
        // This requires the full decoder graph with logits output.

        // Build decoder graph (inline, same as vibevoice_transcribe)
        auto build_dec = [&](int n_tokens, int n_past) -> ggml_cgraph* {
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

            ggml_tensor* cur = embeds;
            for (int il = 0; il < hp.n_lm_layers; il++) {
                char p[64];
                snprintf(p, sizeof(p), "lm.layers.%d", il);
                ggml_tensor* residual = cur;
                cur = ggml_rms_norm(ctx0, cur, 1e-6f);
                cur = ggml_mul(ctx0, cur, G(std::string(p) + ".attn_ln.weight"));
                {
                    int T_cur = n_tokens, Lk = n_past + T_cur;
                    ggml_tensor* Q = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.q_proj.weight"), cur);
                    auto* qb = G(std::string(p) + ".attn.q_proj.bias");
                    if (qb)
                        Q = ggml_add(ctx0, Q, qb);
                    ggml_tensor* K = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.k_proj.weight"), cur);
                    auto* kb = G(std::string(p) + ".attn.k_proj.bias");
                    if (kb)
                        K = ggml_add(ctx0, K, kb);
                    ggml_tensor* V = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.v_proj.weight"), cur);
                    auto* vb = G(std::string(p) + ".attn.v_proj.bias");
                    if (vb)
                        V = ggml_add(ctx0, V, vb);
                    Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, T_cur);
                    K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, T_cur);
                    V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, T_cur);
                    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta,
                                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    K = ggml_rope_ext(ctx0, K, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta,
                                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    ggml_tensor* Kp = ggml_permute(ctx0, K, 0, 2, 1, 3);
                    ggml_tensor* Vp = ggml_permute(ctx0, V, 0, 2, 1, 3);
                    auto* kv = ggml_view_4d(ctx0, ctx->kv_k, hp.head_dim, T_cur, hp.n_kv_heads, 1, ctx->kv_k->nb[1],
                                            ctx->kv_k->nb[2], ctx->kv_k->nb[3],
                                            (size_t)il * ctx->kv_k->nb[3] + (size_t)n_past * ctx->kv_k->nb[1]);
                    auto* vv = ggml_view_4d(ctx0, ctx->kv_v, hp.head_dim, T_cur, hp.n_kv_heads, 1, ctx->kv_v->nb[1],
                                            ctx->kv_v->nb[2], ctx->kv_v->nb[3],
                                            (size_t)il * ctx->kv_v->nb[3] + (size_t)n_past * ctx->kv_v->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kp, kv));
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vp, vv));
                    auto* Kf =
                        ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_k, hp.head_dim, Lk, hp.n_kv_heads, ctx->kv_k->nb[1],
                                                     ctx->kv_k->nb[2], (size_t)il * ctx->kv_k->nb[3]));
                    auto* Vf =
                        ggml_cont(ctx0, ggml_view_3d(ctx0, ctx->kv_v, hp.head_dim, Lk, hp.n_kv_heads, ctx->kv_v->nb[1],
                                                     ctx->kv_v->nb[2], (size_t)il * ctx->kv_v->nb[3]));
                    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                    float scale = 1.0f / sqrtf((float)hp.head_dim);
                    auto* attn = ggml_flash_attn_ext(ctx0, Q, Kf, Vf, causal_mask, scale, 0.0f, 0.0f);
                    attn = ggml_reshape_2d(ctx0, attn, hp.d_lm, T_cur);
                    attn = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.o_proj.weight"), attn);
                    cur = ggml_add(ctx0, residual, attn);
                }
                residual = cur;
                cur = ggml_rms_norm(ctx0, cur, 1e-6f);
                cur = ggml_mul(ctx0, cur, G(std::string(p) + ".ffn_ln.weight"));
                auto* ffn =
                    core_ffn::swiglu(ctx0, cur, G(std::string(p) + ".ffn.gate.weight"),
                                     G(std::string(p) + ".ffn.up.weight"), G(std::string(p) + ".ffn.down.weight"));
                cur = ggml_add(ctx0, residual, ffn);
            }
            // Final norm
            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            ggml_tensor* norm_w = G("lm.norm.weight");
            if (norm_w)
                cur = ggml_mul(ctx0, cur, norm_w);

            // Output both hidden states AND logits
            // Hidden: last token, before LM head
            ggml_tensor* hidden = cur;
            if (n_tokens > 1)
                hidden = ggml_view_1d(ctx0, cur, hp.d_lm, (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));
            ggml_set_name(hidden, "hidden_out");
            ggml_set_output(hidden);

            // Logits: last token through LM head
            ggml_tensor* last = hidden;
            if (n_tokens > 1)
                last = ggml_reshape_2d(ctx0, last, hp.d_lm, 1);
            ggml_tensor* lm_head = G("lm_head.weight");
            if (!lm_head)
                lm_head = G("lm.tok_emb.weight");
            ggml_tensor* logits_out = ggml_mul_mat(ctx0, lm_head, last);
            ggml_set_name(logits_out, "logits");
            ggml_set_output(logits_out);

            ggml_build_forward_expand(gf, logits_out);
            return gf;
        };

        auto run_dec = [&](const float* embeds, int n_tokens, int n_past, std::vector<float>& logits_out,
                           std::vector<float>& hidden_out) -> bool {
            std::vector<int32_t> pos(n_tokens);
            for (int i = 0; i < n_tokens; i++)
                pos[i] = n_past + i;
            std::vector<ggml_fp16_t> mask;
            if (n_tokens > 1) {
                int Lk = n_past + n_tokens;
                mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
                ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
                for (int q = 0; q < n_tokens; q++)
                    for (int k = n_past + q + 1; k < Lk; k++)
                        mask[(size_t)q * Lk + k] = neg_inf;
            }
            ggml_cgraph* gf = build_dec(n_tokens, n_past);
            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
                return false;
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_input"), embeds, 0,
                                    (size_t)hp.d_lm * n_tokens * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), pos.data(), 0,
                                    pos.size() * sizeof(int32_t));
            if (n_tokens > 1)
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                        mask.size() * sizeof(ggml_fp16_t));
            if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
                return false;

            auto* lt = ggml_graph_get_tensor(gf, "logits");
            logits_out.resize((int)lt->ne[0]);
            ggml_backend_tensor_get(lt, logits_out.data(), 0, logits_out.size() * sizeof(float));
            hidden_out.resize(hp.d_lm);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "hidden_out"), hidden_out.data(), 0,
                                    hp.d_lm * sizeof(float));
            return true;
        };

        // Prefill
        std::vector<float> logits_v, hidden_v;
        if (!run_dec(prefix_embeds.data(), prefix_len, 0, logits_v, hidden_v)) {
            fprintf(stderr, "vibevoice TTS (base): prefill failed\n");
            return nullptr;
        }
        int n_past = prefix_len;

        // Autoregressive generation
        ddim_schedule sched = make_ddim_schedule(20);
        float scaling_factor = 0.196f, bias_factor = -0.049f;
        {
            auto* sf = G("speech_scaling_factor");
            auto* bf = G("speech_bias_factor");
            if (sf)
                ggml_backend_tensor_get(sf, &scaling_factor, 0, sizeof(float));
            if (bf)
                ggml_backend_tensor_get(bf, &bias_factor, 0, sizeof(float));
        }

        std::vector<float> all_latents;
        mt19937_state rng;
        {
            uint32_t seed = 42;
            if (const char* sv = getenv("VIBEVOICE_TTS_SEED")) {
                seed = (uint32_t)strtoul(sv, nullptr, 0);
            }
            mt19937_seed(rng, seed);
        }
        float cfg_scale = 1.5f; // Lower CFG for base model (no proper negative conditioning)
        int speech_frames = 0;
        int max_speech_frames = std::max(12, (int)(text_ids.size() * 3));

        if (verbosity >= 1)
            fprintf(stderr, "vibevoice TTS (base): generating speech autoregressively...\n");

        for (int step = 0; step < max_gen && speech_frames < max_speech_frames; step++) {
            // Argmax on logits
            int best = 0;
            for (int i = 1; i < (int)logits_v.size(); i++)
                if (logits_v[i] > logits_v[best])
                    best = i;

            if (verbosity >= 2 || (verbosity >= 1 && step < 10)) {
                const char* piece = (best >= 0 && best < (int)m.vocab.size()) ? m.vocab[best].c_str() : "?";
                fprintf(stderr, "  step %d: token=%d (%s)\n", step, best, piece);
            }

            if (best == SPEECH_END || best == EOS_TOKEN) {
                if (verbosity >= 1)
                    fprintf(stderr, "  stop at step %d (token %d), %d speech frames\n", step, best, speech_frames);
                break;
            }

            if (best == SPEECH_DIFFUSION) {
                // Run diffusion: hidden_v is the LM hidden state at this position
                std::vector<float> z(vae_dim);
                std::vector<float> prev_x0(vae_dim, 0.0f);
                fill_gaussian_noise(z.data(), vae_dim, rng);

                // Simple CFG with zero negative condition (no separate neg LM for base model)
                std::vector<float> neg_cond(d_lm, 0.0f);

                for (int si = 0; si < 20; si++) {
                    float t = (float)sched.timesteps[si];
                    std::vector<float> t_sin(256);
                    compute_sinusoidal_embed(t, t_sin.data(), 256);

                    ggml_cgraph* gf = get_pred_head_graph(ctx, 2);
                    ggml_backend_sched_reset(ctx->sched);
                    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
                        break;
                    std::vector<float> z_pair(vae_dim * 2);
                    memcpy(z_pair.data(), z.data(), vae_dim * sizeof(float));
                    memcpy(z_pair.data() + vae_dim, z.data(), vae_dim * sizeof(float));
                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_noisy"), z_pair.data(), 0,
                                            vae_dim * 2 * sizeof(float));
                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_t_sin"), t_sin.data(), 0,
                                            256 * sizeof(float));
                    std::vector<float> cond_pair((size_t)d_lm * 2);
                    memcpy(cond_pair.data(), hidden_v.data(), d_lm * sizeof(float));
                    memcpy(cond_pair.data() + d_lm, neg_cond.data(), d_lm * sizeof(float));
                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_condition"), cond_pair.data(), 0,
                                            (size_t)d_lm * 2 * sizeof(float));
                    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
                        break;
                    std::vector<float> v_both(vae_dim * 2);
                    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "pred_output"), v_both.data(), 0,
                                            vae_dim * 2 * sizeof(float));
                    std::vector<float> v_cfg(vae_dim);
                    for (int i = 0; i < vae_dim; i++)
                        v_cfg[i] = v_both[vae_dim + i] + cfg_scale * (v_both[i] - v_both[vae_dim + i]);

                    int t_cur = sched.timesteps[si];
                    std::vector<float> x0(vae_dim);
                    v_to_x0(z.data(), v_cfg.data(), x0.data(), vae_dim, sched.alphas_cumprod[t_cur]);
                    bool is_last = (si == 19);
                    if (si == 0 || is_last) {
                        dpm_first_order(sched, si, z.data(), x0.data(), vae_dim);
                    } else {
                        dpm_second_order(sched, si, z.data(), x0.data(), prev_x0.data(), si - 1, vae_dim);
                    }
                    prev_x0 = x0;
                }

                all_latents.insert(all_latents.end(), z.begin(), z.end());
                speech_frames++;

                if (verbosity >= 1 && (speech_frames <= 3 || speech_frames % 5 == 0))
                    fprintf(stderr, "  frame %d (step %d)\n", speech_frames, step);

                // Feed speech latent back via acoustic connector
                auto speech_embed = run_connector_stage(ctx, "at_conn", z.data(), 1, vae_dim);
                if (speech_embed.empty())
                    break;

                // LM step with speech embedding
                if (!run_dec(speech_embed.data(), 1, n_past, logits_v, hidden_v))
                    break;
                n_past++;
            } else {
                // Regular text token — embed and feed to LM
                const int32_t tid = best;
                auto tok_emb = run_token_embedding_lookup(ctx, &tid, 1);
                if (tok_emb.size() != (size_t)hp.d_lm)
                    break;
                if (!run_dec(tok_emb.data(), 1, n_past, logits_v, hidden_v))
                    break;
                n_past++;
            }
        }

        if (all_latents.empty()) {
            fprintf(stderr, "vibevoice TTS (base): no speech frames generated\n");
            return nullptr;
        }

        // Scale and decode
        int total_latent = (int)all_latents.size();
        int actual_frames = total_latent / vae_dim;
        std::vector<float> scaled(total_latent);
        for (int i = 0; i < total_latent; i++)
            scaled[i] = all_latents[i] / scaling_factor - bias_factor;

        ggml_cgraph* dec_gf = build_vae_decoder_graph(ctx, actual_frames);
        ggml_backend_sched_reset(ctx->sched);
        // See realtime VAE-decoder call site for why this is forced to CPU
        // on Metal — Apple's GPU watchdog kills the single-buffer compute.
        {
            // Default: route VAE decoder to CPU on Metal only — that's the
            // backend where we've reproduced Apple's interactivity watchdog.
            // CUDA/Vulkan on systems where the GPU also drives the display
            // can hit a similar driver TDR (NVIDIA Windows TDR ~2s by default,
            // Wayland on Linux varies). Override with VIBEVOICE_VAE_BACKEND:
            //   auto (default) → CPU on Metal, GPU elsewhere
            //   cpu            → always CPU (safe choice for desktop NVIDIA)
            //   gpu            → always GPU (servers, datacenter NVIDIA, etc)
            const char* vae_be = getenv("VIBEVOICE_VAE_BACKEND");
            bool force_cpu;
            if (vae_be && std::strcmp(vae_be, "cpu") == 0)
                force_cpu = true;
            else if (vae_be && std::strcmp(vae_be, "gpu") == 0)
                force_cpu = false;
            else
                force_cpu = backend_is_metal(ctx->backend);
            if (force_cpu && ctx->backend_cpu) {
                for (int i = 0; i < ggml_graph_n_nodes(dec_gf); i++) {
                    ggml_backend_sched_set_tensor_backend(ctx->sched, ggml_graph_node(dec_gf, i), ctx->backend_cpu);
                }
            }
        }
        if (!ggml_backend_sched_alloc_graph(ctx->sched, dec_gf))
            return nullptr;
        ggml_backend_tensor_set(ggml_graph_get_tensor(dec_gf, "dec_latent"), scaled.data(), 0,
                                total_latent * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, dec_gf) != GGML_STATUS_SUCCESS)
            return nullptr;

        ggml_tensor* audio_out = ggml_graph_get_tensor(dec_gf, "dec_audio");
        int total_audio = (int)audio_out->ne[0] * (int)audio_out->ne[1];
        std::vector<float> raw_audio((size_t)total_audio);
        ggml_backend_tensor_get(audio_out, raw_audio.data(), 0, (size_t)total_audio * sizeof(float));

        // Trim leading silence
        int trim_start = 0;
        for (int i = 0; i < total_audio; i++) {
            if (fabsf(raw_audio[i]) > 0.005f) {
                trim_start = std::max(0, i - 800);
                break;
            }
        }
        int trimmed_len = total_audio - trim_start;

        if (verbosity >= 1)
            fprintf(stderr, "vibevoice TTS (base): %d frames → %d samples (%.2f sec)\n", actual_frames, trimmed_len,
                    trimmed_len / 24000.0f);

        float* out_buf = (float*)malloc((size_t)trimmed_len * sizeof(float));
        if (!out_buf)
            return nullptr;
        memcpy(out_buf, raw_audio.data() + trim_start, (size_t)trimmed_len * sizeof(float));
        if (out_n_samples)
            *out_n_samples = trimmed_len;
        return out_buf;
    }

    // ── Realtime-0.5B Streaming Model TTS Path (below) ──
    vibevoice_dump_i32(dump_dir, "tts_token_ids", text_ids.data(), text_ids.size());

    // Build TTS prompt: system + user with text + assistant start
    // Same Qwen2 chat template as ASR but with TTS system prompt.
    const int IM_START = 151644;
    const int IM_END = 151645;
    std::vector<int32_t> prompt;
    // <|im_start|>system\nYou are a helpful assistant that generates speech.<|im_end|>\n
    std::vector<int32_t> sys_toks = {IM_START, 8948, 198}; // system\n
    auto sys_text = tokenize_text_greedy(m, "You are a helpful assistant that generates speech from text.");
    prompt.insert(prompt.end(), sys_toks.begin(), sys_toks.end());
    prompt.insert(prompt.end(), sys_text.begin(), sys_text.end());
    prompt.push_back(IM_END);
    prompt.push_back(198); // \n
    // <|im_start|>user\nPlease read the following text aloud: {text}<|im_end|>\n
    // Tokenize the full user message in one pass for correct BPE merges
    prompt.push_back(IM_START);
    prompt.push_back(872); // user
    prompt.push_back(198); // \n
    std::string user_msg = std::string("Please read the following text aloud: ") + text;
    auto user_tokens = tokenize_text_greedy(m, user_msg.c_str());
    prompt.insert(prompt.end(), user_tokens.begin(), user_tokens.end());
    prompt.push_back(IM_END);
    prompt.push_back(198); // \n
    // <|im_start|>assistant\n
    prompt.push_back(IM_START);
    prompt.push_back(77091); // assistant
    prompt.push_back(198);   // \n

    int prefix_len = (int)prompt.size();

    // With voice prompt loaded: skip the template, use only text tokens as input
    // The voice KV cache already contains the system/chat context
    bool has_voice = ctx->voice.tts_seq_len > 0;
    if (has_voice) {
        // Process text in windows of TTS_TEXT_WINDOW_SIZE=5 (matching official pipeline)
        int tts_text_window = 5;
        int first_window = std::min((int)text_ids.size(), tts_text_window);
        prompt.assign(text_ids.begin(), text_ids.begin() + first_window);
        prefix_len = (int)prompt.size();
    }

    // Detect TTS LM presence
    bool has_tts_lm = hp.tts_n_layers > 0 && G("tts_lm.tok_emb.weight");
    const char* lm_prefix = has_tts_lm ? "tts_lm" : "lm";
    int lm_n_layers = has_tts_lm ? hp.tts_n_layers : hp.n_lm_layers;

    if (verbosity >= 1) {
        fprintf(stderr, "vibevoice TTS: prompt %d tokens, using %s (%d layers)\n", prefix_len,
                has_tts_lm ? "TTS LM" : "base LM", lm_n_layers);
    }

    // 2. Embed prompt tokens (use TTS LM's embedding if available)
    std::string emb_key = has_tts_lm ? "tts_lm.tok_emb.weight" : "lm.tok_emb.weight";
    // Use ggml_get_rows with the right embedding table
    auto prefix_embeds = [&]() -> std::vector<float> {
        auto it = m.tensors.find(emb_key);
        if (it == m.tensors.end() || !it->second)
            return run_token_embedding_lookup(ctx, prompt.data(), prefix_len);

        size_t mem = ctx->compute_meta.size();
        ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

        ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prefix_len);
        ggml_set_name(inp, "tts_tok_ids");
        ggml_set_input(inp);
        ggml_tensor* out = ggml_get_rows(ctx0, it->second, inp);
        ggml_set_name(out, "tts_tok_emb");
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
            return {};
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_tok_ids"), prompt.data(), 0,
                                (size_t)prefix_len * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
            return {};
        std::vector<float> embs((size_t)prefix_len * d_lm);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "tts_tok_emb"), embs.data(), 0, embs.size() * sizeof(float));
        return embs;
    }();
    if ((int)prefix_embeds.size() != prefix_len * d_lm) {
        fprintf(stderr, "vibevoice TTS: token embedding failed\n");
        return nullptr;
    }

    // Extract type embeddings (text=1, speech=0) into reusable vectors
    std::vector<float> text_type_emb(d_lm, 0.0f);

    std::vector<float> all_base_hidden; // base LM hidden states for ALL text tokens (voice mode)

    // 2b. If TTS LM: add type embeddings FIRST, then splice base LM hidden
    //     (matching reference: embed → type_embed → splice overwrites type at text positions)
    if (has_tts_lm) {
        // Add type embedding: tts_input_types[1] for text, tts_input_types[0] for speech
        // Use ggml_get_rows to handle F16/Q4_K weight types correctly
        ggml_tensor* type_w = G("tts_types.weight");
        if (type_w) {
            // type_w is [d_lm, 2] in ggml (ne[0]=d_lm, ne[1]=2).
            // Row 1 = text type embedding.
            size_t mem = ctx->compute_meta.size();
            ggml_init_params ip2 = {mem, ctx->compute_meta.data(), true};
            ggml_context* ctx2 = ggml_init(ip2);
            ggml_cgraph* gf2 = ggml_new_graph_custom(ctx2, 256, false);

            int32_t type_id = 1; // text
            ggml_tensor* idx = ggml_new_tensor_1d(ctx2, GGML_TYPE_I32, 1);
            ggml_set_name(idx, "type_idx");
            ggml_set_input(idx);
            ggml_tensor* row = ggml_get_rows(ctx2, type_w, idx);
            ggml_set_name(row, "type_emb");
            ggml_set_output(row);
            ggml_build_forward_expand(gf2, row);

            ggml_backend_sched_reset(ctx->sched);
            if (ggml_backend_sched_alloc_graph(ctx->sched, gf2)) {
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf2, "type_idx"), &type_id, 0, sizeof(int32_t));
                if (ggml_backend_sched_graph_compute(ctx->sched, gf2) == GGML_STATUS_SUCCESS) {
                    ggml_backend_tensor_get(ggml_graph_get_tensor(gf2, "type_emb"), text_type_emb.data(), 0,
                                            d_lm * sizeof(float));
                    for (int i = 0; i < prefix_len; i++)
                        for (int j = 0; j < d_lm; j++)
                            prefix_embeds[(size_t)i * d_lm + j] += text_type_emb[j];
                    if (verbosity >= 2)
                        fprintf(stderr, "  added tts_input_types[text] to %d positions\n", prefix_len);
                }
            }
        }

        // With voice prompt: skip base LM splicing (voice KV already provides context)
        // Without voice: splice base LM hidden states at text positions
        if (!has_voice) {
            auto base_hidden = run_lm_hidden_states(ctx, text_ids.data(), (int)text_ids.size(), true);
            vibevoice_dump_f32(dump_dir, "tts_base_lm_hidden", base_hidden.data(), base_hidden.size());
            int n_text = (int)text_ids.size();
            if ((int)base_hidden.size() == n_text * d_lm) {
                int start_idx = prefix_len - n_text;
                if (start_idx >= 0) {
                    for (int i = 0; i < n_text; i++)
                        memcpy(prefix_embeds.data() + (size_t)(start_idx + i) * d_lm,
                               base_hidden.data() + (size_t)i * d_lm, d_lm * sizeof(float));
                    if (verbosity >= 1)
                        fprintf(stderr, "  spliced %d base LM hidden at tail positions %d-%d (after type embed)\n",
                                n_text, start_idx, start_idx + n_text - 1);
                }
            }
        } else {
            // With voice: run base LM on text tokens WITH voice.lm KV cache.
            // The official pipeline: forward_lm(text_tokens) with voice.lm KV (74 tokens context)
            // We build a temporary KV cache for the 4-layer base LM, pre-fill from voice.lm,
            // then run the base LM transformer with positions starting at voice.lm.seq_len.

            int base_n_layers = hp.n_lm_layers;   // 4 for Realtime model
            int base_vsl = ctx->voice.lm_seq_len; // 74
            int n_text = (int)text_ids.size();    // process ALL text tokens at once

            // Allocate temporary base LM KV cache
            int base_max_ctx = base_vsl + n_text + 16;
            size_t base_k_size =
                (size_t)ggml_type_size(GGML_TYPE_F16) * hp.head_dim * base_max_ctx * hp.n_kv_heads * base_n_layers;
            ggml_init_params bkp = {2 * ggml_tensor_overhead(), nullptr, true};
            ggml_context* base_kv_ctx = ggml_init(bkp);
            ggml_tensor* base_kv_k =
                ggml_new_tensor_4d(base_kv_ctx, GGML_TYPE_F16, hp.head_dim, base_max_ctx, hp.n_kv_heads, base_n_layers);
            ggml_tensor* base_kv_v =
                ggml_new_tensor_4d(base_kv_ctx, GGML_TYPE_F16, hp.head_dim, base_max_ctx, hp.n_kv_heads, base_n_layers);
            ggml_backend_buffer_t base_kv_buf = ggml_backend_alloc_buffer(ctx->backend, 2 * base_k_size);
            uint8_t* base_ptr = (uint8_t*)ggml_backend_buffer_get_base(base_kv_buf);
            ggml_backend_tensor_alloc(base_kv_buf, base_kv_k, base_ptr);
            ggml_backend_tensor_alloc(base_kv_buf, base_kv_v, base_ptr + base_k_size);
            ggml_backend_buffer_clear(base_kv_buf, 0);

            // Pre-fill base LM KV from voice.lm
            size_t el_size = ggml_type_size(GGML_TYPE_F16);
            for (int il = 0; il < base_n_layers; il++) {
                for (int kv_type = 0; kv_type < 2; kv_type++) {
                    char vname[128];
                    snprintf(vname, sizeof(vname), "voice.lm.%d.%s", il, kv_type == 0 ? "k" : "v");
                    auto it = ctx->voice.tensors.find(vname);
                    if (it == ctx->voice.tensors.end())
                        continue;
                    ggml_tensor* src = it->second;
                    ggml_tensor* dst = (kv_type == 0) ? base_kv_k : base_kv_v;
                    size_t head_src_bytes = (size_t)hp.head_dim * base_vsl * el_size;
                    size_t head_dst_stride = (size_t)hp.head_dim * base_max_ctx * el_size;
                    size_t src_bytes = ggml_nbytes(src);
                    std::vector<uint8_t> tmp(src_bytes);
                    ggml_backend_tensor_get(src, tmp.data(), 0, src_bytes);
                    size_t layer_off = (size_t)il * dst->nb[3];
                    for (int ih = 0; ih < hp.n_kv_heads; ih++) {
                        ggml_backend_tensor_set(dst, tmp.data() + (size_t)ih * head_src_bytes,
                                                layer_off + (size_t)ih * head_dst_stride, head_src_bytes);
                    }
                }
            }

            // Run base LM (4 layers) on text tokens with KV cache
            // Build graph using base LM weights ("lm." prefix) with the temp KV cache
            auto base_embeds = run_token_embedding_lookup(ctx, text_ids.data(), n_text);
            if ((int)base_embeds.size() == n_text * d_lm) {
                // Build and run base LM graph with temp KV cache
                size_t mem = ctx->compute_meta.size();
                ggml_init_params ip_b = {mem, ctx->compute_meta.data(), true};
                ggml_context* ctx_b = ggml_init(ip_b);
                ggml_cgraph* gf_b = ggml_new_graph_custom(ctx_b, 65536, false);

                ggml_tensor* emb_t = ggml_new_tensor_2d(ctx_b, GGML_TYPE_F32, d_lm, n_text);
                ggml_set_name(emb_t, "base_in");
                ggml_set_input(emb_t);
                ggml_tensor* positions = ggml_new_tensor_1d(ctx_b, GGML_TYPE_I32, n_text);
                ggml_set_name(positions, "base_pos");
                ggml_set_input(positions);
                ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx_b, GGML_TYPE_F16, base_vsl + n_text, n_text);
                ggml_set_name(causal_mask, "base_mask");
                ggml_set_input(causal_mask);

                ggml_tensor* cur = emb_t;
                for (int il = 0; il < base_n_layers; il++) {
                    char p[64];
                    snprintf(p, sizeof(p), "lm.layers.%d", il);
                    ggml_tensor* residual = cur;
                    cur = ggml_rms_norm(ctx_b, cur, 1e-6f);
                    cur = ggml_mul(ctx_b, cur, G(std::string(p) + ".attn_ln.weight"));
                    {
                        int Lk = base_vsl + n_text;
                        ggml_tensor* Q = ggml_mul_mat(ctx_b, G(std::string(p) + ".attn.q_proj.weight"), cur);
                        auto* qb = G(std::string(p) + ".attn.q_proj.bias");
                        if (qb)
                            Q = ggml_add(ctx_b, Q, qb);
                        ggml_tensor* K = ggml_mul_mat(ctx_b, G(std::string(p) + ".attn.k_proj.weight"), cur);
                        auto* kb = G(std::string(p) + ".attn.k_proj.bias");
                        if (kb)
                            K = ggml_add(ctx_b, K, kb);
                        ggml_tensor* V = ggml_mul_mat(ctx_b, G(std::string(p) + ".attn.v_proj.weight"), cur);
                        auto* vb = G(std::string(p) + ".attn.v_proj.bias");
                        if (vb)
                            V = ggml_add(ctx_b, V, vb);
                        Q = ggml_reshape_3d(ctx_b, Q, hp.head_dim, hp.n_heads, n_text);
                        K = ggml_reshape_3d(ctx_b, K, hp.head_dim, hp.n_kv_heads, n_text);
                        V = ggml_reshape_3d(ctx_b, V, hp.head_dim, hp.n_kv_heads, n_text);
                        Q = ggml_rope_ext(ctx_b, Q, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                          hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                        K = ggml_rope_ext(ctx_b, K, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                                          hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                        // Write K, V to temp base KV cache
                        ggml_tensor* K_perm = ggml_permute(ctx_b, K, 0, 2, 1, 3);
                        ggml_tensor* V_perm = ggml_permute(ctx_b, V, 0, 2, 1, 3);
                        ggml_tensor* k_view = ggml_view_4d(
                            ctx_b, base_kv_k, hp.head_dim, n_text, hp.n_kv_heads, 1, base_kv_k->nb[1], base_kv_k->nb[2],
                            base_kv_k->nb[3], (size_t)il * base_kv_k->nb[3] + (size_t)base_vsl * base_kv_k->nb[1]);
                        ggml_tensor* v_view = ggml_view_4d(
                            ctx_b, base_kv_v, hp.head_dim, n_text, hp.n_kv_heads, 1, base_kv_v->nb[1], base_kv_v->nb[2],
                            base_kv_v->nb[3], (size_t)il * base_kv_v->nb[3] + (size_t)base_vsl * base_kv_v->nb[1]);
                        ggml_build_forward_expand(gf_b, ggml_cpy(ctx_b, K_perm, k_view));
                        ggml_build_forward_expand(gf_b, ggml_cpy(ctx_b, V_perm, v_view));
                        // Read full KV
                        ggml_tensor* Kf = ggml_cont(
                            ctx_b, ggml_view_3d(ctx_b, base_kv_k, hp.head_dim, Lk, hp.n_kv_heads, base_kv_k->nb[1],
                                                base_kv_k->nb[2], (size_t)il * base_kv_k->nb[3]));
                        ggml_tensor* Vf = ggml_cont(
                            ctx_b, ggml_view_3d(ctx_b, base_kv_v, hp.head_dim, Lk, hp.n_kv_heads, base_kv_v->nb[1],
                                                base_kv_v->nb[2], (size_t)il * base_kv_v->nb[3]));
                        Q = ggml_cont(ctx_b, ggml_permute(ctx_b, Q, 0, 2, 1, 3));
                        float scale = 1.0f / sqrtf((float)hp.head_dim);
                        ggml_tensor* attn = ggml_flash_attn_ext(ctx_b, Q, Kf, Vf, causal_mask, scale, 0.0f, 0.0f);
                        attn = ggml_reshape_2d(ctx_b, attn, d_lm, n_text);
                        attn = ggml_mul_mat(ctx_b, G(std::string(p) + ".attn.o_proj.weight"), attn);
                        cur = ggml_add(ctx_b, residual, attn);
                    }
                    residual = cur;
                    cur = ggml_rms_norm(ctx_b, cur, 1e-6f);
                    cur = ggml_mul(ctx_b, cur, G(std::string(p) + ".ffn_ln.weight"));
                    ggml_tensor* ffn =
                        core_ffn::swiglu(ctx_b, cur, G(std::string(p) + ".ffn.gate.weight"),
                                         G(std::string(p) + ".ffn.up.weight"), G(std::string(p) + ".ffn.down.weight"));
                    cur = ggml_add(ctx_b, residual, ffn);
                }
                // No final norm for Realtime base LM (only 4 layers, no lm.norm.weight)
                ggml_set_name(cur, "base_out");
                ggml_set_output(cur);
                ggml_build_forward_expand(gf_b, cur);

                ggml_backend_sched_reset(ctx->sched);
                if (ggml_backend_sched_alloc_graph(ctx->sched, gf_b)) {
                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "base_in"), base_embeds.data(), 0,
                                            base_embeds.size() * sizeof(float));
                    std::vector<int32_t> bpos(n_text);
                    for (int i = 0; i < n_text; i++)
                        bpos[i] = base_vsl + i;
                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "base_pos"), bpos.data(), 0,
                                            bpos.size() * sizeof(int32_t));
                    // Causal mask: can attend to all voice positions + past text positions
                    int Lk = base_vsl + n_text;
                    std::vector<ggml_fp16_t> bmask((size_t)n_text * Lk, ggml_fp32_to_fp16(0.0f));
                    ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
                    for (int q = 0; q < n_text; q++)
                        for (int k = base_vsl + q + 1; k < Lk; k++)
                            bmask[(size_t)q * Lk + k] = neg_inf;
                    ggml_backend_tensor_set(ggml_graph_get_tensor(gf_b, "base_mask"), bmask.data(), 0,
                                            bmask.size() * sizeof(ggml_fp16_t));

                    if (ggml_backend_sched_graph_compute(ctx->sched, gf_b) == GGML_STATUS_SUCCESS) {
                        std::vector<float> base_hidden(n_text * d_lm);
                        ggml_backend_tensor_get(ggml_graph_get_tensor(gf_b, "base_out"), base_hidden.data(), 0,
                                                base_hidden.size() * sizeof(float));
                        // Store ALL base hidden for use by process_text_window
                        all_base_hidden = base_hidden;
                        vibevoice_dump_f32(dump_dir, "tts_base_lm_hidden_voice", base_hidden.data(),
                                           base_hidden.size());
                        if (verbosity >= 1)
                            fprintf(stderr,
                                    "  base LM with voice KV (%d layers, %d ctx): replaced %d text embeddings + type\n",
                                    base_n_layers, base_vsl, n_text);
                    }
                }
            }

            // Clean up temp base LM KV
            ggml_backend_buffer_free(base_kv_buf);
            ggml_free(base_kv_ctx);
        }
    }

    // 3. Allocate KV cache for autoregressive generation
    // Max frames: generous upper bound; EOS classifier will stop early
    int n_frames = std::max(12, (int)(text_ids.size() * 4.0f));
    n_frames = std::min(n_frames, 300);
    int voice_ctx = ctx->voice.tts_seq_len; // 0 if no voice loaded
    int max_ctx = voice_ctx + prefix_len + n_frames + 16;
    int num_steps = ctx->params.tts_steps > 0 ? ctx->params.tts_steps : 20;

    const ggml_type tts_kv_type = GGML_TYPE_F32;
    if (!ctx->kv_k || ctx->kv_max_ctx < max_ctx || ctx->kv_k->type != tts_kv_type) {
        if (ctx->kv_ctx)
            ggml_free(ctx->kv_ctx);
        if (ctx->kv_buf)
            ggml_backend_buffer_free(ctx->kv_buf);
        if (ctx->kv_neg_ctx)
            ggml_free(ctx->kv_neg_ctx);
        if (ctx->kv_neg_buf)
            ggml_backend_buffer_free(ctx->kv_neg_buf);
        int hd = hp.head_dim, nkv = hp.n_kv_heads, nl = lm_n_layers;
        size_t k_size = (size_t)ggml_type_size(tts_kv_type) * hd * max_ctx * nkv * nl;
        // Positive KV cache
        ggml_init_params kp = {2 * ggml_tensor_overhead(), nullptr, true};
        ctx->kv_ctx = ggml_init(kp);
        ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, tts_kv_type, hd, max_ctx, nkv, nl);
        ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, tts_kv_type, hd, max_ctx, nkv, nl);
        ctx->kv_buf = ggml_backend_alloc_buffer(ctx->backend, 2 * k_size);
        uint8_t* base_pos = (uint8_t*)ggml_backend_buffer_get_base(ctx->kv_buf);
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base_pos);
        ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base_pos + k_size);
        // Negative KV cache (for CFG)
        ggml_init_params kp2 = {2 * ggml_tensor_overhead(), nullptr, true};
        ctx->kv_neg_ctx = ggml_init(kp2);
        ctx->kv_neg_k = ggml_new_tensor_4d(ctx->kv_neg_ctx, tts_kv_type, hd, max_ctx, nkv, nl);
        ctx->kv_neg_v = ggml_new_tensor_4d(ctx->kv_neg_ctx, tts_kv_type, hd, max_ctx, nkv, nl);
        ctx->kv_neg_buf = ggml_backend_alloc_buffer(ctx->backend, 2 * k_size);
        uint8_t* base_neg = (uint8_t*)ggml_backend_buffer_get_base(ctx->kv_neg_buf);
        ggml_backend_tensor_alloc(ctx->kv_neg_buf, ctx->kv_neg_k, base_neg);
        ggml_backend_tensor_alloc(ctx->kv_neg_buf, ctx->kv_neg_v, base_neg + k_size);
        ctx->kv_max_ctx = max_ctx;
    }
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ggml_backend_buffer_clear(ctx->kv_neg_buf, 0);

    // Pre-fill KV caches from voice prompt if loaded
    int voice_offset = 0; // position offset for prompt tokens
    if (ctx->voice.tts_seq_len > 0 && has_tts_lm) {
        // Copy voice KV caches into the context KV caches
        // Voice tensors: voice.tts_lm.{layer}.{k,v} -> kv_k/kv_v at positions 0..tts_seq_len
        // Voice GGUF tensors: voice.tts_lm.{layer}.{k,v}
        // Stored from PyTorch [n_kv_heads, seq_len, head_dim] → ggml ne=[head_dim, seq_len, n_kv_heads]
        // KV cache: [head_dim, max_ctx, n_kv_heads, n_layers]
        // The voice tensor layout matches a slice of the KV cache: same ne[0..2], just ne[1] is smaller.
        // We can copy per-layer, writing to offset 0 of each layer.
        int vsl = ctx->voice.tts_seq_len;
        int hd = hp.head_dim;
        int nkv = hp.n_kv_heads;

        auto copy_voice_kv = [&](const char* prefix, ggml_tensor* dst_k, ggml_tensor* dst_v, int seq_len) {
            // src: [hd, seq_len, nkv] contiguous F16
            // dst: [hd, max_ctx, nkv, nl] — max_ctx > seq_len, so copy per-head
            size_t src_el_size = ggml_type_size(GGML_TYPE_F16);
            size_t dst_el_size = ggml_type_size(dst_k->type);
            size_t head_src_elems = (size_t)hd * seq_len;
            size_t head_src_bytes = head_src_elems * src_el_size;
            size_t head_dst_bytes = head_src_elems * dst_el_size;
            size_t head_dst_stride = (size_t)hd * max_ctx * dst_el_size; // nb[2] in dst

            for (int il = 0; il < lm_n_layers; il++) {
                for (int kv_type = 0; kv_type < 2; kv_type++) {
                    char vname[128];
                    snprintf(vname, sizeof(vname), "voice.%s.%d.%s", prefix, il, kv_type == 0 ? "k" : "v");
                    auto it = ctx->voice.tensors.find(vname);
                    if (it == ctx->voice.tensors.end())
                        continue;
                    ggml_tensor* src = it->second;
                    ggml_tensor* dst = (kv_type == 0) ? dst_k : dst_v;

                    // Read entire source tensor
                    size_t src_bytes = ggml_nbytes(src);
                    std::vector<uint8_t> tmp(src_bytes);
                    ggml_backend_tensor_get(src, tmp.data(), 0, src_bytes);

                    // Copy per-head to account for different position strides
                    size_t layer_off = (size_t)il * dst->nb[3];
                    for (int ih = 0; ih < nkv; ih++) {
                        size_t src_head_off = (size_t)ih * head_src_bytes;
                        size_t dst_head_off = layer_off + (size_t)ih * head_dst_stride;
                        if (dst->type == GGML_TYPE_F32) {
                            std::vector<float> tmp_f32(head_src_elems);
                            const ggml_fp16_t* src_h = (const ggml_fp16_t*)(tmp.data() + src_head_off);
                            for (size_t i = 0; i < head_src_elems; i++)
                                tmp_f32[i] = ggml_fp16_to_fp32(src_h[i]);
                            ggml_backend_tensor_set(dst, tmp_f32.data(), dst_head_off, head_dst_bytes);
                        } else {
                            ggml_backend_tensor_set(dst, tmp.data() + src_head_off, dst_head_off, head_src_bytes);
                        }
                    }
                }
            }
        };

        copy_voice_kv("tts_lm", ctx->kv_k, ctx->kv_v, vsl);
        int neg_vsl = ctx->voice.neg_tts_seq_len;
        copy_voice_kv("neg_tts_lm", ctx->kv_neg_k, ctx->kv_neg_v, neg_vsl);
        voice_offset = vsl;
        if (verbosity >= 1)
            fprintf(stderr, "  pre-filled KV caches from voice prompt (%d tokens)\n", vsl);

        // Debug: dump first layer K cache content for verification
        if (dump_dir) {
            // Read first 5 positions of layer 0 from the KV cache
            // KV layout: [hd=64, max_ctx, nkv=2, nl=20], F16
            // Layer 0 starts at offset 0
            // Position t, head h: offset = h * hd * max_ctx * 2 + t * hd * 2 (bytes)
            size_t dump_pos = 5;
            size_t dump_size = (size_t)hd * dump_pos * ggml_type_size(ctx->kv_k->type);
            std::vector<uint8_t> kv_dump(dump_size);
            ggml_backend_tensor_get(ctx->kv_k, kv_dump.data(), 0, dump_size);
            std::vector<float> kv_f32(hd * dump_pos);
            if (ctx->kv_k->type == GGML_TYPE_F32) {
                memcpy(kv_f32.data(), kv_dump.data(), kv_f32.size() * sizeof(float));
            } else {
                for (size_t i = 0; i < kv_f32.size(); i++)
                    kv_f32[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)kv_dump.data())[i]);
            }
            vibevoice_dump_f32(dump_dir, "kv_cache_l0_h0_first5pos", kv_f32.data(), kv_f32.size());
        }
    }

    // Reuse the existing KV-cached decoder graph builder from ASR
    // (build_decoder_graph lambda is inside vibevoice_transcribe; let's inline a simpler version)
    // kv_sel: 0=positive (kv_k/kv_v), 1=negative (kv_neg_k/kv_neg_v)
    // Accumulators for run_lm_step sub-timing
    double lm_build_ms = 0, lm_alloc_ms = 0, lm_compute_ms = 0;
    int lm_step_count = 0;

    auto run_lm_step = [&](const float* embeds, int n_tokens, int n_past, std::vector<float>& hidden_out,
                           int kv_sel = 0) -> bool {
        auto t_build0 = std::chrono::high_resolution_clock::now();
        ggml_tensor* cur_kv_k = (kv_sel == 0) ? ctx->kv_k : ctx->kv_neg_k;
        ggml_tensor* cur_kv_v = (kv_sel == 0) ? ctx->kv_v : ctx->kv_neg_v;
        if (!cur_kv_k || !cur_kv_v)
            return false;
        size_t mem = ctx->compute_meta.size();
        ggml_init_params ip = {mem, ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

        ggml_tensor* emb_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_lm, n_tokens);
        ggml_set_name(emb_t, "tts_step_in");
        ggml_set_input(emb_t);

        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_name(positions, "tts_step_pos");
        ggml_set_input(positions);

        ggml_tensor* causal_mask = nullptr;
        if (n_tokens > 1) {
            causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
            ggml_set_name(causal_mask, "tts_step_mask");
            ggml_set_input(causal_mask);
        }

        ggml_tensor* cur = emb_t;
        for (int il = 0; il < lm_n_layers; il++) {
            char p[64];
            snprintf(p, sizeof(p), "%s.layers.%d", lm_prefix, il);
            ggml_tensor* residual = cur;

            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            cur = ggml_mul(ctx0, cur, G(std::string(p) + ".attn_ln.weight"));

            {
                int T_cur = n_tokens;
                int Lk = n_past + T_cur;

                ggml_tensor* Q = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.q_proj.weight"), cur);
                ggml_tensor* q_b = G(std::string(p) + ".attn.q_proj.bias");
                if (q_b)
                    Q = ggml_add(ctx0, Q, q_b);
                ggml_tensor* K = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.k_proj.weight"), cur);
                ggml_tensor* k_b = G(std::string(p) + ".attn.k_proj.bias");
                if (k_b)
                    K = ggml_add(ctx0, K, k_b);
                ggml_tensor* V = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.v_proj.weight"), cur);
                ggml_tensor* v_b = G(std::string(p) + ".attn.v_proj.bias");
                if (v_b)
                    V = ggml_add(ctx0, V, v_b);

                Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, T_cur);
                K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, T_cur);
                V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, T_cur);

                Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f);
                K = ggml_rope_ext(ctx0, K, positions, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f,
                                  0.0f, 1.0f, 0.0f, 0.0f);

                ggml_tensor* K_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
                ggml_tensor* V_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);
                ggml_tensor* k_view =
                    ggml_view_4d(ctx0, cur_kv_k, hp.head_dim, T_cur, hp.n_kv_heads, 1, cur_kv_k->nb[1], cur_kv_k->nb[2],
                                 cur_kv_k->nb[3], (size_t)il * cur_kv_k->nb[3] + (size_t)n_past * cur_kv_k->nb[1]);
                ggml_tensor* v_view =
                    ggml_view_4d(ctx0, cur_kv_v, hp.head_dim, T_cur, hp.n_kv_heads, 1, cur_kv_v->nb[1], cur_kv_v->nb[2],
                                 cur_kv_v->nb[3], (size_t)il * cur_kv_v->nb[3] + (size_t)n_past * cur_kv_v->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_perm, k_view));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_perm, v_view));

                ggml_tensor* Kfull =
                    ggml_cont(ctx0, ggml_view_3d(ctx0, cur_kv_k, hp.head_dim, Lk, hp.n_kv_heads, cur_kv_k->nb[1],
                                                 cur_kv_k->nb[2], (size_t)il * cur_kv_k->nb[3]));
                ggml_tensor* Vfull =
                    ggml_cont(ctx0, ggml_view_3d(ctx0, cur_kv_v, hp.head_dim, Lk, hp.n_kv_heads, cur_kv_v->nb[1],
                                                 cur_kv_v->nb[2], (size_t)il * cur_kv_v->nb[3]));

                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                float scale = 1.0f / sqrtf((float)hp.head_dim);
                ggml_tensor* attn_out = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask, scale, 0.0f, 0.0f);
                attn_out = ggml_reshape_2d(ctx0, attn_out, hp.d_lm, T_cur);
                attn_out = ggml_mul_mat(ctx0, G(std::string(p) + ".attn.o_proj.weight"), attn_out);
                cur = ggml_add(ctx0, residual, attn_out);
            }

            residual = cur;
            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            cur = ggml_mul(ctx0, cur, G(std::string(p) + ".ffn_ln.weight"));
            ggml_tensor* ffn =
                core_ffn::swiglu(ctx0, cur, G(std::string(p) + ".ffn.gate.weight"),
                                 G(std::string(p) + ".ffn.up.weight"), G(std::string(p) + ".ffn.down.weight"));
            cur = ggml_add(ctx0, residual, ffn);
        }

        // Final norm (hidden states, no LM head)
        // Realtime model's base LM (4 layers) has no final norm — output goes directly to TTS LM
        ggml_tensor* final_norm_w = G(std::string(lm_prefix) + ".norm.weight");
        if (final_norm_w) {
            cur = ggml_rms_norm(ctx0, cur, 1e-6f);
            cur = ggml_mul(ctx0, cur, final_norm_w);
        }

        // Take last token only
        if (n_tokens > 1)
            cur = ggml_view_1d(ctx0, cur, hp.d_lm, (size_t)(n_tokens - 1) * hp.d_lm * sizeof(float));

        ggml_set_name(cur, "tts_hidden_out");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        // Run
        auto t_alloc0 = std::chrono::high_resolution_clock::now();
        lm_build_ms += std::chrono::duration<double, std::milli>(t_alloc0 - t_build0).count();

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
            return false;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_step_in"), embeds, 0,
                                (size_t)hp.d_lm * n_tokens * sizeof(float));

        std::vector<int32_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; i++)
            pos[i] = n_past + i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_step_pos"), pos.data(), 0, pos.size() * sizeof(int32_t));

        if (n_tokens > 1) {
            int Lk = n_past + n_tokens;
            std::vector<ggml_fp16_t> mask((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
            ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < n_tokens; q++)
                for (int k = n_past + q + 1; k < Lk; k++)
                    mask[(size_t)q * Lk + k] = neg_inf;
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tts_step_mask"), mask.data(), 0,
                                    mask.size() * sizeof(ggml_fp16_t));
        }

        auto t_compute0 = std::chrono::high_resolution_clock::now();
        lm_alloc_ms += std::chrono::duration<double, std::milli>(t_compute0 - t_alloc0).count();

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
            return false;

        auto t_compute1 = std::chrono::high_resolution_clock::now();
        lm_compute_ms += std::chrono::duration<double, std::milli>(t_compute1 - t_compute0).count();
        lm_step_count++;

        hidden_out.resize(hp.d_lm);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "tts_hidden_out"), hidden_out.data(), 0,
                                hp.d_lm * sizeof(float));
        return true;
    };

    // 4. Prefill prompt through LM (fills KV cache)
    // 4. Interleaved text/speech window processing (matching official pipeline)
    // TTS_TEXT_WINDOW_SIZE=5, TTS_SPEECH_WINDOW_SIZE=6
    const int TEXT_WINDOW = 5;
    const int SPEECH_WINDOW = 6;
    int text_cursor = 0; // how many text tokens processed so far
    int n_past = voice_offset;
    // Negative KV cache always has the IMAGE_PAD prefill at pos 0 (see prefill block above),
    // regardless of voice mode. Speech frames are appended starting at pos 1.
    int neg_n_past = has_tts_lm ? 1 : 0;
    std::vector<float> hidden;

    // Initial negative condition for CFG. Per the official
    // microsoft/VibeVoice modeling_vibevoice_streaming_inference.py the negative path is
    // PREFILLED ONCE with a single <|image_pad|> token (no past KV) and is NEVER updated
    // by text windows — only by speech frames. We therefore recompute that prefill
    // here (deterministic, identical to what voice.neg_lm/voice.neg_tts_lm cache).
    //
    // Recipe:
    //   1. neg_base_pad_hidden = neg base LM forward(IMAGE_PAD at pos 0, no past KV)
    //   2. neg_prefill_input   = neg_base_pad_hidden + tts_input_types[1]
    //   3. neg_condition       = neg TTS LM forward(neg_prefill_input at pos 0, no past KV).last
    std::vector<float> neg_condition(d_lm, 0.0f);
    if (has_tts_lm) {
        const int32_t IMAGE_PAD = 151655;
        int32_t pad_id = IMAGE_PAD;
        // 1. Run the negative base LM prefill (1 IMAGE_PAD token, no past KV, no final norm).
        //    Realtime base LM has no final norm — pass has_final_norm=false. We do not need
        //    to persist the base LM negative KV (it is never re-attended to in the inner loop).
        auto pad_emb = run_token_embedding_lookup(ctx, &pad_id, 1);
        std::vector<float> neg_base_pad_hidden =
            run_qwen2_prefill_no_kv(ctx, pad_emb.data(), 1, "lm", hp.n_lm_layers, /*has_final_norm=*/false,
                                    /*all_positions=*/true);
        if ((int)neg_base_pad_hidden.size() != d_lm) {
            fprintf(stderr, "vibevoice TTS: neg base prefill failed\n");
            return nullptr;
        }
        // 2. Build TTS LM prefill input. The official forward_tts_lm replaces the input
        //    embedding tail with lm_last_hidden_state, then adds tts_input_types[1] for text.
        //    For a 1-token input the entire embedding is replaced; tts_emb(IMAGE_PAD) is unused.
        std::vector<float> neg_prefill_input(d_lm);
        for (int j = 0; j < d_lm; j++)
            neg_prefill_input[j] = neg_base_pad_hidden[j] + text_type_emb[j];
        // 3. Run TTS LM prefill via run_lm_step at pos 0 with kv_sel=1 — this both returns
        //    the last hidden (= initial neg_condition) AND writes the K, V at pos 0 of the
        //    negative KV cache. In with-voice mode this overwrites the value loaded from
        //    voice.neg_tts_lm.{k,v} with a recomputed-from-weights equivalent (same input
        //    → same output modulo F16 rounding); in no-voice mode this initialises the
        //    cache that subsequent speech frames need to attend to.
        if (!run_lm_step(neg_prefill_input.data(), 1, 0, neg_condition, /*kv_sel=*/1)) {
            fprintf(stderr, "vibevoice TTS: neg TTS LM prefill failed\n");
            return nullptr;
        }
        if (verbosity >= 2 || getenv("VIBEVOICE_TTS_TRACE")) {
            float rms = sqrtf(
                std::inner_product(neg_condition.begin(), neg_condition.end(), neg_condition.begin(), 0.0f) / d_lm);
            fprintf(stderr, "  neg_condition prefill (IMAGE_PAD): rms=%.4f\n", rms);
        }
    }

    // Process a text window: ONLY the positive path is advanced. The negative path is
    // never updated by text windows (matches the official inference loop in
    // modeling_vibevoice_streaming_inference.py).
    auto process_text_window = [&](int cursor, int win_len) -> bool {
        if (win_len <= 0)
            return true;

        // Build input: base_hidden[cursor:cursor+win_len] + type_emb[1]
        std::vector<float> win_embeds((size_t)win_len * d_lm, 0.0f);
        if (!all_base_hidden.empty()) {
            memcpy(win_embeds.data(), all_base_hidden.data() + (size_t)cursor * d_lm,
                   (size_t)win_len * d_lm * sizeof(float));
        }
        for (int i = 0; i < win_len; i++)
            for (int j = 0; j < d_lm; j++)
                win_embeds[(size_t)i * d_lm + j] += text_type_emb[j];

        // Run TTS LM (positive path) ONLY.
        if (!run_lm_step(win_embeds.data(), win_len, n_past, hidden, 0))
            return false;
        n_past += win_len;
        return true;
    };

    // Initial TTS-LM prefill / first text window.
    if (has_voice) {
        // With voice: voice.tts_lm KV is already loaded into the cache (positions 0..vsl_tts-1).
        // We feed the first TEXT_WINDOW tokens of the user text through the TTS LM at the
        // tail (positions vsl_tts..vsl_tts+win-1). all_base_hidden was populated above by the
        // base LM run with voice.lm KV.
        int first_win = std::min((int)text_ids.size(), TEXT_WINDOW);
        if (verbosity >= 1)
            fprintf(stderr, "vibevoice TTS: text window 1: %d tokens, pos %d\n", first_win, n_past);
        if (!process_text_window(0, first_win)) {
            fprintf(stderr, "vibevoice TTS: first text window failed\n");
            return nullptr;
        }
        text_cursor = first_win;
    } else {
        // No voice loaded: prefix_embeds holds the full chat-template prompt (system + user
        // with the entire text + assistant header) with type embeddings already added and
        // base-LM hidden states spliced at the text positions. Prefill the whole thing
        // through the TTS LM at pos 0..prefix_len-1; subsequent iterations only generate
        // speech frames (no more text windows — all text is in this prefix).
        if (verbosity >= 1)
            fprintf(stderr, "vibevoice TTS (no-voice): prefilling chat template (%d tokens) through TTS LM\n",
                    prefix_len);
        if (!run_lm_step(prefix_embeds.data(), prefix_len, 0, hidden, /*kv_sel=*/0)) {
            fprintf(stderr, "vibevoice TTS: no-voice TTS LM prefill failed\n");
            return nullptr;
        }
        n_past = prefix_len;
        text_cursor = (int)text_ids.size(); // skip subsequent text windows
    }
    vibevoice_dump_f32(dump_dir, "tts_prefill_hidden", hidden.data(), hidden.size());
    vibevoice_dump_f32(dump_dir, "tts_neg_condition_frame0", neg_condition.data(), neg_condition.size());

    const auto t_prefill_done = std::chrono::high_resolution_clock::now();
    if (verbosity >= 1)
        fprintf(stderr, "vibevoice TTS: generating frames with text/speech interleaving...\n");

    // 5. Autoregressive frame generation
    ddim_schedule sched = make_ddim_schedule(num_steps);
    float scaling_factor = 0.196f, bias_factor = -0.049f;
    {
        auto* sf = G("speech_scaling_factor");
        auto* bf = G("speech_bias_factor");
        if (sf)
            ggml_backend_tensor_get(sf, &scaling_factor, 0, sizeof(float));
        if (bf)
            ggml_backend_tensor_get(bf, &bias_factor, 0, sizeof(float));
        if (verbosity >= 2 || getenv("VIBEVOICE_TTS_TRACE"))
            fprintf(stderr, "  speech_scaling=%g speech_bias=%g  (sf=%p bf=%p)\n", scaling_factor, bias_factor,
                    (void*)sf, (void*)bf);
    }

    // Extract speech type embedding (type=0) for AR feedback
    std::vector<float> speech_type_emb(d_lm, 0.0f);
    if (has_tts_lm) {
        ggml_tensor* type_w = G("tts_types.weight");
        if (type_w) {
            size_t mem = ctx->compute_meta.size();
            ggml_init_params ip3 = {mem, ctx->compute_meta.data(), true};
            ggml_context* ctx3 = ggml_init(ip3);
            ggml_cgraph* gf3 = ggml_new_graph_custom(ctx3, 256, false);
            int32_t type_id = 0; // speech
            ggml_tensor* idx = ggml_new_tensor_1d(ctx3, GGML_TYPE_I32, 1);
            ggml_set_name(idx, "stype_idx");
            ggml_set_input(idx);
            ggml_tensor* row = ggml_get_rows(ctx3, type_w, idx);
            ggml_set_name(row, "stype_emb");
            ggml_set_output(row);
            ggml_build_forward_expand(gf3, row);
            ggml_backend_sched_reset(ctx->sched);
            if (ggml_backend_sched_alloc_graph(ctx->sched, gf3)) {
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf3, "stype_idx"), &type_id, 0, sizeof(int32_t));
                if (ggml_backend_sched_graph_compute(ctx->sched, gf3) == GGML_STATUS_SUCCESS)
                    ggml_backend_tensor_get(ggml_graph_get_tensor(gf3, "stype_emb"), speech_type_emb.data(), 0,
                                            d_lm * sizeof(float));
            }
        }
    }

    // neg_condition is computed inside process_text_window
    float cfg_scale = 3.0f;

    std::vector<float> all_latents;
    mt19937_state rng;
    {
        uint32_t seed = 42;
        if (const char* sv = getenv("VIBEVOICE_TTS_SEED")) {
            seed = (uint32_t)strtoul(sv, nullptr, 0);
        }
        mt19937_seed(rng, seed);
    }

    std::vector<float> preloaded_noise;
    const char* noise_file = getenv("VIBEVOICE_TTS_NOISE");
    if (noise_file && noise_file[0]) {
        FILE* nf = fopen(noise_file, "rb");
        if (nf) {
            fseek(nf, 0, SEEK_END);
            size_t nb = (size_t)ftell(nf);
            fseek(nf, 0, SEEK_SET);
            preloaded_noise.resize(nb / sizeof(float));
            size_t rd = fread(preloaded_noise.data(), sizeof(float), preloaded_noise.size(), nf);
            fclose(nf);
            (void)rd;
        }
    }

    int total_frames = 0;
    bool finished = false;
    int trace_frame = -1;
    if (const char* tf = getenv("VIBEVOICE_TTS_TRACE_FRAME"))
        trace_frame = atoi(tf);
    double bench_sum_diff = 0, bench_sum_lm = 0;
    int bench_frames = 0;

    while (!finished && total_frames < n_frames) {
        // Generate SPEECH_WINDOW frames
        int frames_this_window = std::min(SPEECH_WINDOW, n_frames - total_frames);

        for (int si = 0; si < frames_this_window; si++) {
            int fi = total_frames + si;
            const bool append_audio_frame = !finished;
            auto t_frame_start = std::chrono::high_resolution_clock::now();
            if (verbosity >= 1 && (fi == 0 || (fi + 1) % 5 == 0 || fi == n_frames - 1))
                fprintf(stderr, "  frame %d/%d...\n", fi + 1, n_frames);

            // a. Run diffusion with Classifier-Free Guidance (CFG) using DPM-Solver++
            std::vector<float> z(vae_dim);
            std::vector<float> prev_x0(vae_dim, 0.0f);
            if (!preloaded_noise.empty() && (size_t)(fi + 1) * vae_dim <= preloaded_noise.size()) {
                memcpy(z.data(), preloaded_noise.data() + (size_t)fi * vae_dim, vae_dim * sizeof(float));
            } else {
                fill_gaussian_noise(z.data(), vae_dim, rng);
            }
            if (fi == 0)
                vibevoice_dump_f32(dump_dir, "tts_noise_frame0", z.data(), z.size());
            // Per-frame conditions/noise dump (VIBEVOICE_TTS_DUMP_PERFRAME=1).
            if (getenv("VIBEVOICE_TTS_DUMP_PERFRAME")) {
                char nm[64];
                snprintf(nm, sizeof(nm), "perframe_pos_cond_f%03d", fi);
                vibevoice_dump_f32(dump_dir, nm, hidden.data(), hidden.size());
                snprintf(nm, sizeof(nm), "perframe_neg_cond_f%03d", fi);
                vibevoice_dump_f32(dump_dir, nm, neg_condition.data(), neg_condition.size());
                snprintf(nm, sizeof(nm), "perframe_noise_f%03d", fi);
                vibevoice_dump_f32(dump_dir, nm, z.data(), z.size());
            }

            for (int step = 0; step < num_steps; step++) {
                float t = (float)sched.timesteps[step];
                std::vector<float> t_sin(256);
                compute_sinusoidal_embed(t, t_sin.data(), 256);

                // Run prediction head with BOTH conditions (batched as 2 frames)
                // Frame 0 = positive condition (text-conditioned)
                // Frame 1 = negative condition (unconditional)
                ggml_cgraph* gf = get_pred_head_graph(ctx, 2);
                ggml_backend_sched_reset(ctx->sched);
                if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
                    return nullptr;

                // Noisy input: same z for both conditions [vae_dim, 2]
                std::vector<float> z_pair(vae_dim * 2);
                memcpy(z_pair.data(), z.data(), vae_dim * sizeof(float));
                memcpy(z_pair.data() + vae_dim, z.data(), vae_dim * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_noisy"), z_pair.data(), 0,
                                        vae_dim * 2 * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_t_sin"), t_sin.data(), 0, 256 * sizeof(float));

                // Conditions: [d_lm, 2] — positive then negative
                std::vector<float> cond_pair((size_t)d_lm * 2);
                memcpy(cond_pair.data(), hidden.data(), d_lm * sizeof(float));
                memcpy(cond_pair.data() + d_lm, neg_condition.data(), d_lm * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pred_condition"), cond_pair.data(), 0,
                                        (size_t)d_lm * 2 * sizeof(float));

                if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
                    return nullptr;

                // Read both predictions [vae_dim, 2]
                std::vector<float> v_both(vae_dim * 2);
                ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "pred_output"), v_both.data(), 0,
                                        vae_dim * 2 * sizeof(float));

                // CFG interpolation: v = uncond + cfg_scale * (cond - uncond)
                std::vector<float> v_cfg(vae_dim);
                for (int i = 0; i < vae_dim; i++) {
                    float v_cond = v_both[i];             // positive
                    float v_uncond = v_both[vae_dim + i]; // negative
                    v_cfg[i] = v_uncond + cfg_scale * (v_cond - v_uncond);
                }
                if (fi == 0 && step == 0)
                    vibevoice_dump_f32(dump_dir, "tts_v_cfg_step0", v_cfg.data(), v_cfg.size());
                // Per-frame step-0 v_cfg dump (VIBEVOICE_TTS_DUMP_PERFRAME=1).
                if (step == 0 && getenv("VIBEVOICE_TTS_DUMP_PERFRAME")) {
                    char nm[64];
                    snprintf(nm, sizeof(nm), "perframe_v_cfg_step0_f%03d", fi);
                    vibevoice_dump_f32(dump_dir, nm, v_cfg.data(), v_cfg.size());
                }
                if (fi == trace_frame) {
                    char nm[96];
                    snprintf(nm, sizeof(nm), "trace_f%03d_v_cfg_s%02d", fi, step);
                    vibevoice_dump_f32(dump_dir, nm, v_cfg.data(), v_cfg.size());
                    snprintf(nm, sizeof(nm), "trace_f%03d_z_before_s%02d", fi, step);
                    vibevoice_dump_f32(dump_dir, nm, z.data(), z.size());
                }

                // Convert v-prediction to x0 prediction
                int t_cur = sched.timesteps[step];
                std::vector<float> x0(vae_dim);
                v_to_x0(z.data(), v_cfg.data(), x0.data(), vae_dim, sched.alphas_cumprod[t_cur]);
                if (fi == trace_frame) {
                    char nm[96];
                    snprintf(nm, sizeof(nm), "trace_f%03d_x0_s%02d", fi, step);
                    vibevoice_dump_f32(dump_dir, nm, x0.data(), x0.size());
                }

                // DPM-Solver++ update
                bool is_last = (step == num_steps - 1);
                bool use_first_order = (step == 0 || is_last); // lower_order_final=true
                if (use_first_order) {
                    dpm_first_order(sched, step, z.data(), x0.data(), vae_dim);
                } else {
                    dpm_second_order(sched, step, z.data(), x0.data(), prev_x0.data(), step - 1, vae_dim);
                }
                if (fi == trace_frame) {
                    char nm[96];
                    snprintf(nm, sizeof(nm), "trace_f%03d_z_after_s%02d", fi, step);
                    vibevoice_dump_f32(dump_dir, nm, z.data(), z.size());
                }
                prev_x0 = x0;
            }

            if (append_audio_frame)
                all_latents.insert(all_latents.end(), z.begin(), z.end());
            if (fi == 0) {
                vibevoice_dump_f32(dump_dir, "tts_latent_frame0", z.data(), z.size());
            }
            if (getenv("VIBEVOICE_TTS_DUMP_PERFRAME")) {
                char nm[64];
                snprintf(nm, sizeof(nm), "perframe_latent_f%03d", fi);
                vibevoice_dump_f32(dump_dir, nm, z.data(), z.size());
            }

            auto t_diff_end = std::chrono::high_resolution_clock::now();

            // b. Feed generated latent back through acoustic connector → LM embedding
            auto speech_embed = run_connector_stage(ctx, "at_conn", z.data(), 1, vae_dim);
            if (speech_embed.empty()) {
                fprintf(stderr, "vibevoice TTS: connector failed at frame %d\n", fi);
                return nullptr;
            }
            if (fi == 0) {
                vibevoice_dump_f32(dump_dir, "tts_acoustic_embed_frame0", speech_embed.data(), speech_embed.size());
            }
            if (getenv("VIBEVOICE_TTS_DUMP_PERFRAME")) {
                char nm[64];
                snprintf(nm, sizeof(nm), "perframe_acoustic_embed_f%03d", fi);
                vibevoice_dump_f32(dump_dir, nm, speech_embed.data(), speech_embed.size());
            }

            // c. Add speech type embedding (type=0) and feed to TTS LM
            if (has_tts_lm) {
                for (int j = 0; j < d_lm; j++)
                    speech_embed[j] += speech_type_emb[j];
            }

            if (fi < n_frames - 1) { // skip last frame's LM step
                // Update positive path
                if (!run_lm_step(speech_embed.data(), 1, n_past, hidden, 0)) {
                    fprintf(stderr, "vibevoice TTS: LM step failed at frame %d\n", fi);
                    return nullptr;
                }
                // Update negative path (same speech embed, different KV cache)
                std::vector<float> neg_hidden_update;
                if (!run_lm_step(speech_embed.data(), 1, neg_n_past, neg_hidden_update, 1)) {
                    fprintf(stderr, "vibevoice TTS: neg LM step failed at frame %d\n", fi);
                    return nullptr;
                }
                neg_condition = neg_hidden_update;
                n_past++;
                neg_n_past++;

                // EOS classifier: official BinaryClassifier is
                // sigmoid(fc2(relu(fc1(hidden)))) > 0.5.
                ggml_tensor* eos_fc1_w = G("tts_eos.fc1.weight");
                ggml_tensor* eos_fc1_b = G("tts_eos.fc1.bias");
                ggml_tensor* eos_fc2_w = G("tts_eos.fc2.weight");
                ggml_tensor* eos_fc2_b = G("tts_eos.fc2.bias");
                if (eos_fc1_w && eos_fc2_w) {
                    // Run EOS classifier via small ggml graph
                    size_t mem2 = ctx->compute_meta.size();
                    ggml_init_params ip_e = {mem2, ctx->compute_meta.data(), true};
                    ggml_context* ctx_e = ggml_init(ip_e);
                    ggml_cgraph* gf_e = ggml_new_graph_custom(ctx_e, 256, false);
                    ggml_tensor* h_in = ggml_new_tensor_1d(ctx_e, GGML_TYPE_F32, d_lm);
                    ggml_set_name(h_in, "eos_in");
                    ggml_set_input(h_in);
                    ggml_tensor* e = ggml_mul_mat(ctx_e, eos_fc1_w, h_in);
                    if (eos_fc1_b)
                        e = ggml_add(ctx_e, e, eos_fc1_b);
                    e = ggml_relu(ctx_e, e);
                    e = ggml_mul_mat(ctx_e, eos_fc2_w, e);
                    if (eos_fc2_b)
                        e = ggml_add(ctx_e, e, eos_fc2_b);
                    ggml_set_name(e, "eos_out");
                    ggml_set_output(e);
                    ggml_build_forward_expand(gf_e, e);
                    ggml_backend_sched_reset(ctx->sched);
                    if (ggml_backend_sched_alloc_graph(ctx->sched, gf_e)) {
                        ggml_backend_tensor_set(ggml_graph_get_tensor(gf_e, "eos_in"), hidden.data(), 0,
                                                d_lm * sizeof(float));
                        if (ggml_backend_sched_graph_compute(ctx->sched, gf_e) == GGML_STATUS_SUCCESS) {
                            float eos_logit = 0;
                            ggml_backend_tensor_get(ggml_graph_get_tensor(gf_e, "eos_out"), &eos_logit, 0,
                                                    sizeof(float));
                            float eos_prob = 1.0f / (1.0f + expf(-eos_logit)); // sigmoid
                            if (getenv("VIBEVOICE_TTS_DUMP_PERFRAME")) {
                                char nm[64];
                                snprintf(nm, sizeof(nm), "perframe_eos_logit_f%03d", fi);
                                vibevoice_dump_f32(dump_dir, nm, &eos_logit, 1);
                                snprintf(nm, sizeof(nm), "perframe_eos_prob_f%03d", fi);
                                vibevoice_dump_f32(dump_dir, nm, &eos_prob, 1);
                            }
                            if (eos_prob > 0.5f) {
                                if (verbosity >= 1)
                                    fprintf(stderr, "  EOS at frame %d (prob=%.3f)\n", fi, eos_prob);
                                finished = true;
                            }
                        }
                    }
                }
            }
            // Per-frame timing accumulation
            if (getenv("VIBEVOICE_BENCH")) {
                auto t_frame_end = std::chrono::high_resolution_clock::now();
                double diff_ms = std::chrono::duration<double, std::milli>(t_diff_end - t_frame_start).count();
                double rest_ms = std::chrono::duration<double, std::milli>(t_frame_end - t_diff_end).count();
                bench_sum_diff += diff_ms;
                bench_sum_lm += rest_ms;
                bench_frames++;
                if (fi == 0 || finished) {
                    fprintf(stderr,
                            "  BENCH[%d frames]: diffusion=%.0fms/frame (%.0fms total), LM+conn+eos=%.0fms/frame "
                            "(%.0fms total)\n",
                            bench_frames, bench_sum_diff / bench_frames, bench_sum_diff, bench_sum_lm / bench_frames,
                            bench_sum_lm);
                }
            }
        } // end speech frame loop

        total_frames += frames_this_window;

        // Process next text window (if any remaining)
        if (text_cursor < (int)text_ids.size() && !finished) {
            int next_win = std::min((int)text_ids.size() - text_cursor, TEXT_WINDOW);
            if (verbosity >= 1)
                fprintf(stderr, "  text window: %d tokens (cursor %d/%d), pos %d\n", next_win, text_cursor,
                        (int)text_ids.size(), n_past);
            if (!process_text_window(text_cursor, next_win)) {
                fprintf(stderr, "vibevoice TTS: text window failed at cursor %d\n", text_cursor);
                return nullptr;
            }
            text_cursor += next_win;
        }
    } // end text/speech interleave loop

    int total_latent = (int)all_latents.size();
    if (verbosity >= 1) {
        float lmin = all_latents[0], lmax = all_latents[0], lsum = 0;
        for (int i = 0; i < total_latent; i++) {
            if (all_latents[i] < lmin)
                lmin = all_latents[i];
            if (all_latents[i] > lmax)
                lmax = all_latents[i];
            lsum += all_latents[i] * all_latents[i];
        }
        fprintf(stderr, "vibevoice TTS: AR generation done (latent: min=%.4f max=%.4f rms=%.4f)\n", lmin, lmax,
                sqrtf(lsum / total_latent));
    }

    // 6. Scale and decode
    const auto t_ar_done = std::chrono::high_resolution_clock::now();
    if (getenv("VIBEVOICE_BENCH") && lm_step_count > 0) {
        fprintf(stderr,
                "  BENCH LM step (%d calls): build=%.0fms (%.1f/call), alloc=%.0fms (%.1f/call), compute=%.0fms "
                "(%.1f/call)\n",
                lm_step_count, lm_build_ms, lm_build_ms / lm_step_count, lm_alloc_ms, lm_alloc_ms / lm_step_count,
                lm_compute_ms, lm_compute_ms / lm_step_count);
    }
    int actual_frames = total_latent / vae_dim;
    std::vector<float> scaled_latent(total_latent);
    for (int i = 0; i < total_latent; i++)
        scaled_latent[i] = all_latents[i] / scaling_factor - bias_factor;

    auto t_vae_build0 = std::chrono::high_resolution_clock::now();
    ggml_cgraph* dec_gf = build_vae_decoder_graph(ctx, actual_frames);
    auto t_vae_alloc0 = std::chrono::high_resolution_clock::now();
    ggml_backend_sched_reset(ctx->sched);

    // On Metal: force the entire VAE decoder graph onto CPU. The decoder
    // is one large σ-VAE conv stack (7 stages, 6 transposed convs, 3200x
    // upsample) that runs as a single Metal command buffer; whatever it
    // takes >5s on M1 trips Apple's interactivity watchdog
    // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity, observed
    // even with n_cb bumped to 4). Encoders, LM, diffusion all stay on
    // Metal — only this one graph runs CPU. Net cost on M1: ~10-15% of
    // TTS time (was ~29% of compute on CPU before recent VAE
    // optimizations); the alternative is the entire process aborting.
    if (backend_is_metal(ctx->backend) && ctx->backend_cpu) {
        for (int i = 0; i < ggml_graph_n_nodes(dec_gf); i++) {
            ggml_backend_sched_set_tensor_backend(ctx->sched, ggml_graph_node(dec_gf, i), ctx->backend_cpu);
        }
    }

    if (!ggml_backend_sched_alloc_graph(ctx->sched, dec_gf)) {
        fprintf(stderr, "vibevoice TTS: decoder graph alloc failed\n");
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(dec_gf, "dec_latent"), scaled_latent.data(), 0,
                            total_latent * sizeof(float));
    auto t_vae_compute0 = std::chrono::high_resolution_clock::now();

    if (ggml_backend_sched_graph_compute(ctx->sched, dec_gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vibevoice TTS: decoder compute failed\n");
        return nullptr;
    }
    auto t_vae_compute1 = std::chrono::high_resolution_clock::now();
    if (getenv("VIBEVOICE_BENCH")) {
        fprintf(stderr, "  BENCH VAE (%d frames→%dx): build=%.0fms, alloc=%.0fms, compute=%.0fms, ops=%d\n",
                actual_frames, actual_frames * 3200,
                std::chrono::duration<double, std::milli>(t_vae_alloc0 - t_vae_build0).count(),
                std::chrono::duration<double, std::milli>(t_vae_compute0 - t_vae_alloc0).count(),
                std::chrono::duration<double, std::milli>(t_vae_compute1 - t_vae_compute0).count(),
                ggml_graph_n_nodes(dec_gf));
    }

    ggml_tensor* audio_out = ggml_graph_get_tensor(dec_gf, "dec_audio");
    int n_ch = (int)audio_out->ne[0];
    int n_audio = (int)audio_out->ne[1];
    int total_audio = n_ch * n_audio;

    const auto tts_t1 = std::chrono::high_resolution_clock::now();
    if (verbosity >= 1) {
        double tts_ms = std::chrono::duration<double, std::milli>(tts_t1 - tts_t0).count();
        double audio_sec = total_audio / 24000.0;
        fprintf(stderr, "vibevoice TTS: output %d samples (%.2f sec at 24kHz) in %.1f ms (%.2fx realtime)\n",
                total_audio, audio_sec, tts_ms, audio_sec / (tts_ms / 1000.0));
    }
    if (getenv("VIBEVOICE_BENCH")) {
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_done - tts_t0).count();
        double ar_ms = std::chrono::duration<double, std::milli>(t_ar_done - t_prefill_done).count();
        double vae_ms = std::chrono::duration<double, std::milli>(tts_t1 - t_ar_done).count();
        double total_ms = std::chrono::duration<double, std::milli>(tts_t1 - tts_t0).count();
        fprintf(stderr, "  BENCH phases: prefill=%.0fms (%.0f%%), AR=%.0fms (%.0f%%), VAE=%.0fms (%.0f%%)\n",
                prefill_ms, 100.0 * prefill_ms / total_ms, ar_ms, 100.0 * ar_ms / total_ms, vae_ms,
                100.0 * vae_ms / total_ms);
    }

    std::vector<float> raw_audio((size_t)total_audio);
    ggml_backend_tensor_get(audio_out, raw_audio.data(), 0, (size_t)total_audio * sizeof(float));

    // ── Start trim: the σ-VAE decoder's first ~100ms is a deterministic
    //               warmup transient from zero-pad propagation through 7
    //               cascaded causal-conv stages. With the GELU FFN fix
    //               (#39) the transient peaks reach ~0.18 — well above any
    //               sensible noise floor — so the previous "find first
    //               sample > 0.005, back up 800" heuristic kept the
    //               transient in the output. Issue #40: that came out as
    //               an audible click at the start.
    //
    //               Always skip at least the warmup samples; THEN apply
    //               the noise-floor heuristic on the remainder so we still
    //               include the attack of the first phoneme.
    const int decoder_warmup_samples = 2400; // 100 ms @ 24 kHz
    const float noise_floor = 0.005f;
    int trim_start = std::min(decoder_warmup_samples, total_audio);
    for (int i = decoder_warmup_samples; i < total_audio; i++) {
        if (fabsf(raw_audio[i]) > noise_floor) {
            trim_start = std::max(decoder_warmup_samples, i - 800); // ~33ms attack margin
            break;
        }
    }

    // ── End trim: the EOS classifier (sigmoid > 0.5) is sensitive to
    //              accumulated F16 drift in the conditions. We sometimes
    //              fire EOS 1-2 frames after the official model would,
    //              and those extra frames contain residual speech instead
    //              of the clean trailing silence the official produces.
    //              Issue #40: that came out as the audio "ending abruptly"
    //              mid-tail. Find the last sample above a slightly
    //              stricter floor and keep ~50ms of silence margin after
    //              it; drop the rest.
    const float tail_floor = 0.01f;
    const int tail_margin_samples = 1200; // 50 ms @ 24 kHz
    int trim_end = total_audio;
    for (int i = total_audio - 1; i > trim_start; i--) {
        if (fabsf(raw_audio[i]) > tail_floor) {
            trim_end = std::min(total_audio, i + tail_margin_samples + 1);
            break;
        }
    }

    int trimmed_len = trim_end - trim_start;
    if (trimmed_len <= 0) {
        // Pathological case: no audio above floor (e.g. very short input).
        // Fall back to the full decoder output minus the warmup, no tail trim.
        trim_start = std::min(decoder_warmup_samples, total_audio);
        trim_end = total_audio;
        trimmed_len = trim_end - trim_start;
    }

    float* out_buf = (float*)malloc((size_t)trimmed_len * sizeof(float));
    if (!out_buf)
        return nullptr;
    memcpy(out_buf, raw_audio.data() + trim_start, (size_t)trimmed_len * sizeof(float));

    // De-click the start. Some valid diffusion noise trajectories produce a
    // hot first speech latent; trimming removes the decoder warmup, but the
    // remaining audio can still begin with a non-zero impulse. A short equal-
    // power fade-in removes the hard discontinuity without audibly changing
    // normal speech attacks.
    const int fade_in_samples = 480; // 20ms @ 24 kHz
    int n_fade_in = std::min(fade_in_samples, trimmed_len);
    for (int i = 0; i < n_fade_in; i++) {
        float t = (float)i / (float)n_fade_in;
        out_buf[i] *= t * t;
    }

    // Linear fade-out over the last `fade_samples` to ensure a clean end.
    // The EOS classifier is sensitive to F16 drift in the conditions and can
    // fire 1-2 frames late vs the F32 reference, so the last decoded frame
    // sometimes still carries speech-level audio (the official model would
    // have stopped already). Fading to zero matches the perceptual quality
    // the user gets from the upstream model where the streaming decoder's
    // tail naturally lands at silence.
    const int fade_samples = 1200; // 50ms @ 24 kHz
    int n_fade = std::min(fade_samples, trimmed_len);
    for (int i = 0; i < n_fade; i++) {
        float t = (float)(n_fade - i) / (float)n_fade; // 1.0 → 0.0
        out_buf[trimmed_len - n_fade + i] *= t;
    }

    if (verbosity >= 1 && (trim_start > 0 || trim_end < total_audio))
        fprintf(stderr,
                "vibevoice TTS: trimmed %d leading + %d trailing samples (kept %d / %d), %d-sample fade-in, "
                "%d-sample fade-out\n",
                trim_start, total_audio - trim_end, trimmed_len, total_audio, n_fade_in, n_fade);

    if (out_n_samples)
        *out_n_samples = trimmed_len;
    return out_buf;
}
