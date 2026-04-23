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
};

struct vibevoice_context {
    vibevoice_model model;
    vibevoice_context_params params;
    ggml_backend_t backend = nullptr;
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

    gguf_free(gctx);

    if (params.verbosity >= 1) {
        fprintf(stderr, "vibevoice: d_lm=%d, layers=%d, heads=%d/%d, ffn=%d, vocab=%d\n", hp.d_lm, hp.n_lm_layers,
                hp.n_heads, hp.n_kv_heads, hp.d_ffn, hp.vocab_size);
        fprintf(stderr, "vibevoice: vae_acoustic=%d, vae_semantic=%d, downsample=%dx\n", hp.vae_dim_acoustic,
                hp.vae_dim_semantic, hp.total_downsample);
    }

    // Load weights
    ctx->backend = hp.d_lm > 0 ? (params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init())
                                : ggml_backend_cpu_init();
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

    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 65536, false, false);
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
    delete ctx;
}

// ===========================================================================
// ggml graph helpers
// ===========================================================================

// ConvRMSNorm: operates on [C, T] (ne[0]=C), normalizes over C per time step
static ggml_tensor* build_conv_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps = 1e-6f) {
    // x: [C, T], w: [C]. ggml_rms_norm normalizes over ne[0]=C. Good.
    // ggml_mul(x, w): x=[C,T], w=[C]. w broadcasts over T. OK.
    x = ggml_rms_norm(ctx, x, eps);
    if (w) {
        // Verify shapes match before mul
        if (x->ne[0] != w->ne[0]) {
            fprintf(stderr, "  BUG: conv_rms_norm shape mismatch: x=[%lld,%lld] w=[%lld]\n",
                    (long long)x->ne[0], (long long)x->ne[1], (long long)w->ne[0]);
        }
        x = ggml_mul(ctx, x, w);
    }
    return x;
}

// Causal Conv1d: left-pad by (K-1), then conv1d with stride.
// Input/output in [C, T] format (channels-first, like PyTorch).
// ggml_conv_1d produces [T_out, C_out] so we transpose.
static ggml_tensor* build_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride) {
    int K = (int)w->ne[0];
    int pad_left = K - 1;
    // Input x is [C, T]. ggml_conv_1d wants [T, C_in], so transpose.
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C,T] → [T,C]
    if (pad_left > 0)
        x = ggml_pad_reflect_1d(ctx, x, pad_left, 0);
    x = ggml_conv_1d(ctx, w, x, stride, 0, 1); // → [T_out, C_out]
    // Add bias (ne[0]=T_out, ne[1]=C_out; bias ne[0]=C_out → transpose, add, transpose)
    if (b) {
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_out, T_out]
        xt = ggml_add(ctx, xt, b); // [C_out] + [C_out, T_out] broadcasts over T
        return xt; // already in [C, T] format
    }
    return ggml_cont(ctx, ggml_transpose(ctx, x)); // [T_out, C_out] → [C_out, T_out]
}

// Causal depthwise Conv1d: left-pad by (K-1), then dw_conv.
// Input/output in [C, T] format.
static ggml_tensor* build_causal_dw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    int K = (int)w->ne[0];
    int pad_left = K - 1;
    // Input x is [C, T]. ggml_conv_1d_dw also needs input in ggml conv format.
    // ggml_conv_1d_dw(a, b, s, p, d): a=kernel [K,1,C], b=input
    // Need to check expected input format for dw conv...
    x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C,T] → [T,C]
    if (pad_left > 0)
        x = ggml_pad_reflect_1d(ctx, x, pad_left, 0);
    x = ggml_conv_1d_dw(ctx, w, x, 1, 0, 1);
    fprintf(stderr, "      dw_conv out: ne=[%lld,%lld,%lld,%lld]\n",
            (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)x->ne[3]);
    // conv_1d_dw returns 3D [1, T_out, C] — need to reshape to 2D [T_out, C] first
    if (ggml_n_dims(x) > 2)
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    if (b) {
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
        fprintf(stderr, "      after transpose: xt=[%lld,%lld], b=[%lld]\n",
                (long long)xt->ne[0], (long long)xt->ne[1], (long long)b->ne[0]);
        xt = ggml_add(ctx, xt, b);
        return xt;
    }
    return ggml_cont(ctx, ggml_transpose(ctx, x));
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
    fprintf(stderr, "    block: x=[%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    ggml_tensor* h = build_conv_rms_norm(ctx, x, norm_w);
    fprintf(stderr, "    after norm: h=[%lld,%lld]\n", (long long)h->ne[0], (long long)h->ne[1]);
    h = build_causal_dw_conv1d(ctx, h, dw_conv_w, dw_conv_b);
    fprintf(stderr, "    after dw_conv: h=[%lld,%lld]\n", (long long)h->ne[0], (long long)h->ne[1]);
    if (gamma) {
        if (h->ne[0] != gamma->ne[0])
            fprintf(stderr, "  BUG: gamma shape mismatch: h=[%lld,%lld] gamma=[%lld]\n",
                    (long long)h->ne[0], (long long)h->ne[1], (long long)gamma->ne[0]);
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
    int n_ds = (int)ratios.size() + 1; // +1 for stem

    for (int si = 0; si < hp.n_encoder_stages; si++) {
        fprintf(stderr, "  stage %d: h=[%lld,%lld]\n", si, (long long)h->ne[0], (long long)h->ne[1]);
        // Downsample
        char wn[128], bn[128];
        snprintf(wn, sizeof(wn), "%s.ds.%d.0.conv.weight", prefix, si);
        snprintf(bn, sizeof(bn), "%s.ds.%d.0.conv.bias", prefix, si);
        ggml_tensor* ds_w = G(wn);
        ggml_tensor* ds_b = G(bn);
        if (ds_w) {
            int stride = (si == 0) ? 1 : ratios[si - 1]; // stem has stride 1
            fprintf(stderr, "  ds.%d: w=[%lld,%lld,%lld] stride=%d\n", si,
                    (long long)ds_w->ne[0], (long long)ds_w->ne[1], (long long)ds_w->ne[2], stride);
            h = build_causal_conv1d(ctx0, h, ds_w, ds_b, stride);
            fprintf(stderr, "  after ds.%d: h=[%lld,%lld]\n", si, (long long)h->ne[0], (long long)h->ne[1]);
        }

        // Stage blocks
        int n_blocks = (si < (int)hp.encoder_depths.size()) ? hp.encoder_depths[si] : 3;
        for (int bi = 0; bi < n_blocks; bi++) {
            char base[128];
            snprintf(base, sizeof(base), "%s.s.%d.%d", prefix, si, bi);

            h = build_block1d(ctx0, h,
                              G(std::string(base) + ".norm.weight"),
                              G(std::string(base) + ".dw_conv.weight"),
                              G(std::string(base) + ".dw_conv.bias"),
                              G(std::string(base) + ".gamma"),
                              G(std::string(base) + ".ffn_ln.weight"),
                              G(std::string(base) + ".ffn.up.weight"),
                              G(std::string(base) + ".ffn.up.bias"),
                              G(std::string(base) + ".ffn.down.weight"),
                              G(std::string(base) + ".ffn.down.bias"),
                              G(std::string(base) + ".ffn_gamma"));
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

    // Dump for reference comparison
    const char* dump_dir = getenv("VIBEVOICE_DUMP_DIR");
    if (dump_dir && dump_dir[0]) {
        char path[512];
        snprintf(path, sizeof(path), "%s/acoustic_mean.bin", dump_dir);
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(acoustic_mean.data(), sizeof(float), vae_dim_at * T_audio, f);
            fclose(f);
            fprintf(stderr, "  DUMP: acoustic_mean [%d,%d] → %s\n", vae_dim_at, T_audio, path);
        }
    }

    // TODO: semantic encoder, connectors, LM decoder
    fprintf(stderr, "vibevoice: acoustic encoder done. LM decoder not yet implemented.\n");
    return nullptr;
}
