// gemma4_e2b.cpp — CrispASR runtime for Google Gemma-4-E2B
//
// Architecture: USM Conformer audio encoder (12L, 1024d, chunked attention,
// LightConv1d, macaron FFN) + Gemma4 LLM decoder (35L, 1536d, GQA 8Q/1KV,
// per-layer embeddings, hybrid sliding/full attention, SwiGLU).
//
// Audio: 128-bin log-mel at 16kHz, 40ms frames, max 30s.
// Tokenizer: BPE (262K vocab) stored in GGUF.

#include "gemma4_e2b.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// ── Hyperparameters ─────────────────────────────────────────────────────────

struct g4e_audio_hp {
    uint32_t hidden_size = 1024;
    uint32_t num_layers = 12;
    uint32_t num_heads = 8;
    uint32_t head_dim = 128; // hidden / heads
    uint32_t conv_kernel_size = 5;
    uint32_t chunk_size = 12;
    uint32_t context_left = 13;
    uint32_t output_proj_dims = 1536;
    float residual_weight = 0.5f;
    float attention_logit_cap = 50.0f;
};

struct g4e_llm_hp {
    uint32_t hidden_size = 1536;
    uint32_t num_layers = 35;
    uint32_t num_heads = 8;
    uint32_t num_kv_heads = 1;
    uint32_t head_dim = 256;
    uint32_t intermediate_size = 6144; // layers 0-14; layers 15+ use 12288
    uint32_t vocab_size = 262144;
    uint32_t max_position_embeddings = 131072;
    uint32_t sliding_window = 512;
    float rope_theta = 10000.0f;
    float final_logit_softcapping = 30.0f;
    float rms_norm_eps = 1e-6f;
};

// ── Model tensors ───────────────────────────────────────────────────────────

struct g4e_audio_layer {
    // Macaron FFN 1
    ggml_tensor* ffn1_pre_ln = nullptr;
    ggml_tensor* ffn1_up_w = nullptr;
    ggml_tensor* ffn1_down_w = nullptr;
    ggml_tensor* ffn1_post_ln = nullptr;

    // Self-attention
    ggml_tensor* attn_pre_ln = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_per_dim_scale = nullptr; // [head_dim]
    ggml_tensor* attn_rel_k_w = nullptr;       // [hidden, hidden] — relative position bias
    ggml_tensor* attn_post_ln = nullptr;

    // LightConv1d
    ggml_tensor* conv_pre_ln = nullptr;
    ggml_tensor* conv_gate_w = nullptr; // [2*hidden, hidden] — GLU gating
    ggml_tensor* conv_dw_w = nullptr;   // [hidden, 1, k] — depthwise conv
    ggml_tensor* conv_ln = nullptr;
    ggml_tensor* conv_out_w = nullptr; // [hidden, hidden]

    // Macaron FFN 2
    ggml_tensor* ffn2_pre_ln = nullptr;
    ggml_tensor* ffn2_up_w = nullptr;
    ggml_tensor* ffn2_down_w = nullptr;
    ggml_tensor* ffn2_post_ln = nullptr;

    // Output norm
    ggml_tensor* out_ln = nullptr;
};

struct g4e_llm_layer {
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* q_proj = nullptr; // [n_heads*head_dim, hidden]
    ggml_tensor* k_proj = nullptr; // [kv_heads*head_dim, hidden]
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* o_proj = nullptr; // [hidden, n_heads*head_dim]
    ggml_tensor* q_norm = nullptr; // [head_dim]
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* post_attn_norm = nullptr;
    ggml_tensor* pre_ffn_norm = nullptr;
    ggml_tensor* gate_proj = nullptr;
    ggml_tensor* up_proj = nullptr;
    ggml_tensor* down_proj = nullptr;
    ggml_tensor* post_ffn_norm = nullptr;
    // Per-layer embeddings (PLE)
    ggml_tensor* ple_gate = nullptr; // [256, hidden]
    ggml_tensor* ple_proj = nullptr; // [hidden, 256]
    ggml_tensor* post_ple_norm = nullptr;
    ggml_tensor* layer_scalar = nullptr; // [1]
};

struct g4e_model {
    g4e_audio_hp audio_hp;
    g4e_llm_hp llm_hp;

    // Weight context
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Audio subsampling
    ggml_tensor* sub_conv0_w = nullptr;      // [128, 1, 3, 3]
    ggml_tensor* sub_norm0_w = nullptr;      // [128]
    ggml_tensor* sub_conv1_w = nullptr;      // [32, 128, 3, 3]
    ggml_tensor* sub_norm1_w = nullptr;      // [32]
    ggml_tensor* sub_input_proj_w = nullptr; // [1024, 1024]

    // Audio conformer layers
    std::vector<g4e_audio_layer> audio_layers;

    // Audio output projection
    ggml_tensor* audio_output_proj_w = nullptr; // [1536, 1024]
    ggml_tensor* audio_output_proj_b = nullptr; // [1536]

    // Audio embedding projection (post-conformer → LLM hidden)
    ggml_tensor* audio_embed_proj_w = nullptr; // [1536, 1536]

    // LLM embeddings
    ggml_tensor* llm_embed_w = nullptr;    // [vocab, hidden]
    ggml_tensor* llm_ple_w = nullptr;      // [vocab, 35*256] — per-layer embeddings
    ggml_tensor* llm_final_norm = nullptr; // [hidden]

    // LLM layers
    std::vector<g4e_llm_layer> llm_layers;

    // Tokenizer
    std::vector<std::string> vocab;
    std::vector<std::string> merges;
};

// ── Context ─────────────────────────────────────────────────────────────────

struct gemma4_e2b_context {
    g4e_model model;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    int n_threads = 4;
    int verbosity = 1;
    float temperature = 0.0f;
};

// ── Conformer encoder graph builder ─────────────────────────────────────────
// Builds a single ggml graph for the full 12-layer USM Conformer encoder.
// Input: mel features [n_mels, T_mel] after Conv2D subsampling → [hidden, T_sub]
// Output: [output_proj_dims, T_sub]

static ggml_tensor* build_macaron_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pre_ln, ggml_tensor* up_w,
                                      ggml_tensor* down_w, ggml_tensor* post_ln, float residual_weight, float eps) {
    // Half-step FFN: x + residual_weight * post_ln(down(silu(up(pre_ln(x)))))
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, pre_ln);
    h = ggml_mul_mat(ctx, up_w, h);
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, down_w, h);
    ggml_tensor* normed = ggml_rms_norm(ctx, h, eps);
    normed = ggml_mul(ctx, normed, post_ln);
    return ggml_add(ctx, x, ggml_scale(ctx, normed, residual_weight));
}

static ggml_tensor* build_light_conv1d(ggml_context* ctx, ggml_tensor* x, const g4e_audio_layer& L, int T, int hidden,
                                       float eps) {
    // LightConv1d: pre_ln → gate_proj (GLU: split → sigmoid gate) → depthwise_conv1d(causal, k=5) →
    // conv_norm → out_proj + residual
    ggml_tensor* residual = x;
    ggml_tensor* h = ggml_rms_norm(ctx, x, eps);
    h = ggml_mul(ctx, h, L.conv_pre_ln);

    // GLU gating: gate_proj produces [2*hidden, T], split into value + gate
    h = ggml_mul_mat(ctx, L.conv_gate_w, h); // [2*hidden, T]
    int half = hidden;
    ggml_tensor* val = ggml_view_2d(ctx, h, half, T, h->nb[1], 0);
    ggml_tensor* gate = ggml_view_2d(ctx, h, half, T, h->nb[1], half * ggml_type_size(h->type));
    h = ggml_mul(ctx, val, ggml_sigmoid(ctx, gate)); // [hidden, T]

    // Causal depthwise conv1d (kernel=5, left-pad=4)
    // Transpose [hidden, T] → [T, hidden] for conv_1d_dw
    ggml_tensor* ht = ggml_cont(ctx, ggml_transpose(ctx, h));
    int k = 5;
    int pad_left = k - 1; // causal: all padding on left
    ht = ggml_pad_ext(ctx, ht, pad_left, 0, 0, 0, 0, 0, 0, 0);
    ht = ggml_conv_1d_dw(ctx, L.conv_dw_w, ht, 1, 0, 1);
    if (ggml_n_dims(ht) > 2)
        ht = ggml_reshape_2d(ctx, ht, ht->ne[0], ht->ne[1]);
    h = ggml_cont(ctx, ggml_transpose(ctx, ht)); // back to [hidden, T]

    // Conv norm + out projection
    h = ggml_rms_norm(ctx, h, eps);
    h = ggml_mul(ctx, h, L.conv_ln);
    h = ggml_mul_mat(ctx, L.conv_out_w, h);

    return ggml_add(ctx, residual, h);
}

// ── Public API ──────────────────────────────────────────────────────────────

extern "C" struct gemma4_e2b_context_params gemma4_e2b_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true, /*temperature=*/0.0f};
}

static uint32_t g4e_gguf_u32(gguf_context* ctx, const char* key, uint32_t def = 0) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
static float g4e_gguf_f32(gguf_context* ctx, const char* key, float def = 0.0f) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

extern "C" struct gemma4_e2b_context* gemma4_e2b_init_from_file(const char* path_model,
                                                                struct gemma4_e2b_context_params params) {
    auto* ctx = new gemma4_e2b_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->temperature = params.temperature;
    auto& m = ctx->model;

    // ── Read GGUF metadata ──────────────────────────────────────────────
    struct gguf_init_params gp = {true, &m.ctx_w};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "gemma4_e2b: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& ahp = m.audio_hp;
    ahp.hidden_size = g4e_gguf_u32(gctx, "gemma4e2b.audio.hidden_size", 1024);
    ahp.num_layers = g4e_gguf_u32(gctx, "gemma4e2b.audio.num_layers", 12);
    ahp.num_heads = g4e_gguf_u32(gctx, "gemma4e2b.audio.num_heads", 8);
    ahp.head_dim = ahp.hidden_size / ahp.num_heads;
    ahp.conv_kernel_size = g4e_gguf_u32(gctx, "gemma4e2b.audio.conv_kernel_size", 5);
    ahp.chunk_size = g4e_gguf_u32(gctx, "gemma4e2b.audio.chunk_size", 12);
    ahp.context_left = g4e_gguf_u32(gctx, "gemma4e2b.audio.context_left", 13);
    ahp.output_proj_dims = g4e_gguf_u32(gctx, "gemma4e2b.audio.output_proj_dims", 1536);
    ahp.residual_weight = g4e_gguf_f32(gctx, "gemma4e2b.audio.residual_weight", 0.5f);
    ahp.attention_logit_cap = g4e_gguf_f32(gctx, "gemma4e2b.audio.attention_logit_cap", 50.0f);

    auto& lhp = m.llm_hp;
    lhp.hidden_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.hidden_size", 1536);
    lhp.num_layers = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_layers", 35);
    lhp.num_heads = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_heads", 8);
    lhp.num_kv_heads = g4e_gguf_u32(gctx, "gemma4e2b.llm.num_kv_heads", 1);
    lhp.head_dim = g4e_gguf_u32(gctx, "gemma4e2b.llm.head_dim", 256);
    lhp.intermediate_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.intermediate_size", 6144);
    lhp.vocab_size = g4e_gguf_u32(gctx, "gemma4e2b.llm.vocab_size", 262144);
    lhp.max_position_embeddings = g4e_gguf_u32(gctx, "gemma4e2b.llm.max_position_embeddings", 131072);
    lhp.sliding_window = g4e_gguf_u32(gctx, "gemma4e2b.llm.sliding_window", 512);
    lhp.rope_theta = g4e_gguf_f32(gctx, "gemma4e2b.llm.rope_theta", 10000.0f);
    lhp.final_logit_softcapping = g4e_gguf_f32(gctx, "gemma4e2b.llm.final_logit_softcapping", 30.0f);
    lhp.rms_norm_eps = g4e_gguf_f32(gctx, "gemma4e2b.llm.rms_norm_eps", 1e-6f);

    // Read tokenizer from GGUF
    int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
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

    if (ahp.hidden_size == 0 || lhp.vocab_size == 0) {
        fprintf(stderr, "gemma4_e2b: invalid model metadata\n");
        delete ctx;
        return nullptr;
    }

    // ── Load weights ────────────────────────────────────────────────────
    // Use CPU for weights — the LLM decoder can use ggml_backend_sched
    // to auto-copy to GPU for batched matmuls, and the audio conformer
    // uses single-graph execution.
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "gemma4_e2b: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    // Load to CPU (same pattern as firered — allows native Q4_K SIMD for decoder)
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend_cpu, "gemma4_e2b", wl)) {
        fprintf(stderr, "gemma4_e2b: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx_w = wl.ctx;
    m.buf_w = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ts.find(name);
        if (it == ts.end()) {
            if (params.verbosity >= 2)
                fprintf(stderr, "gemma4_e2b: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };

    // ── Bind audio tensors ──────────────────────────────────────────────
    m.sub_conv0_w = get("audio.subsample.conv0.weight");
    m.sub_norm0_w = get("audio.subsample.norm0.weight");
    m.sub_conv1_w = get("audio.subsample.conv1.weight");
    m.sub_norm1_w = get("audio.subsample.norm1.weight");
    m.sub_input_proj_w = get("audio.subsample.input_proj.weight");

    m.audio_layers.resize(ahp.num_layers);
    for (uint32_t i = 0; i < ahp.num_layers; i++) {
        char buf[128];
        auto& L = m.audio_layers[i];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "audio.layers.%u.%s", i, suffix);
            return get(buf);
        };
        // Macaron FFN 1
        L.ffn1_pre_ln = g("ffn1.pre_ln.weight");
        L.ffn1_up_w = g("ffn1.up.weight");
        L.ffn1_down_w = g("ffn1.down.weight");
        L.ffn1_post_ln = g("ffn1.post_ln.weight");
        // Self-attention
        L.attn_pre_ln = g("attn_pre_ln.weight");
        L.attn_q_w = g("attn.q.weight");
        L.attn_k_w = g("attn.k.weight");
        L.attn_v_w = g("attn.v.weight");
        L.attn_o_w = g("attn.o.weight");
        L.attn_per_dim_scale = g("attn.per_dim_scale");
        L.attn_rel_k_w = g("attn.rel_k.weight");
        L.attn_post_ln = g("attn_post_ln.weight");
        // LightConv1d
        L.conv_pre_ln = g("conv.pre_ln.weight");
        L.conv_gate_w = g("conv.gate_proj.weight");
        L.conv_dw_w = g("conv.dw_conv.weight");
        L.conv_ln = g("conv.conv_ln.weight");
        L.conv_out_w = g("conv.out_proj.weight");
        // Macaron FFN 2
        L.ffn2_pre_ln = g("ffn2.pre_ln.weight");
        L.ffn2_up_w = g("ffn2.up.weight");
        L.ffn2_down_w = g("ffn2.down.weight");
        L.ffn2_post_ln = g("ffn2.post_ln.weight");
        // Output norm
        L.out_ln = g("out_ln.weight");
    }
    m.audio_output_proj_w = get("audio.output_proj.weight");
    m.audio_output_proj_b = get("audio.output_proj.bias");
    m.audio_embed_proj_w = get("audio.embed_proj.weight");

    // ── Bind LLM tensors ────────────────────────────────────────────────
    m.llm_embed_w = get("llm.embed_tokens.weight");
    m.llm_ple_w = get("llm.embed_tokens_per_layer.weight");
    m.llm_final_norm = get("llm.norm.weight");

    m.llm_layers.resize(lhp.num_layers);
    for (uint32_t i = 0; i < lhp.num_layers; i++) {
        char buf[128];
        auto& L = m.llm_layers[i];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.layers.%u.%s", i, suffix);
            return get(buf);
        };
        L.attn_norm = g("attn_norm.weight");
        L.q_proj = g("attn.q_proj.weight");
        L.k_proj = g("attn.k_proj.weight");
        L.v_proj = g("attn.v_proj.weight");
        L.o_proj = g("attn.o_proj.weight");
        L.q_norm = g("attn.q_norm.weight");
        L.k_norm = g("attn.k_norm.weight");
        L.post_attn_norm = g("post_attn_norm.weight");
        L.pre_ffn_norm = g("pre_ffn_norm.weight");
        L.gate_proj = g("ffn.gate.weight");
        L.up_proj = g("ffn.up.weight");
        L.down_proj = g("ffn.down.weight");
        L.post_ffn_norm = g("post_ffn_norm.weight");
        L.ple_gate = g("ple_gate.weight");
        L.ple_proj = g("ple_proj.weight");
        L.post_ple_norm = g("post_ple_norm.weight");
        L.layer_scalar = g("layer_scalar");
    }

    // Setup scheduler for GPU-accelerated encoder/LLM
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    if (params.verbosity >= 1) {
        fprintf(stderr, "gemma4_e2b: audio %uL×%u, llm %uL×%u, vocab %u\n", ahp.num_layers, ahp.hidden_size,
                lhp.num_layers, lhp.hidden_size, lhp.vocab_size);
    }

    return ctx;
}

extern "C" char* gemma4_e2b_transcribe(struct gemma4_e2b_context* ctx, const float* pcm, int n_samples) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;

    auto& m = ctx->model;
    auto& ahp = m.audio_hp;
    auto& lhp = m.llm_hp;
    bool verbose = ctx->verbosity >= 2 || getenv("GEMMA4_E2B_BENCH");

    if (ctx->verbosity >= 1)
        fprintf(stderr, "gemma4_e2b: %d samples (%.1fs)\n", n_samples, n_samples / 16000.0f);

    // ── Step 1: Mel spectrogram (128-bin, 16kHz, HF/Whisper-style) ──────
    // Uses core_mel::compute — same as qwen3/voxtral (HF cluster:
    // log10 + global clip, n_mels=128, n_fft=400, hop=160, win=400)
    // TODO: bake filterbank into GGUF; for now, compute at runtime
    // core_mel::Params mel_params;
    // mel_params.n_mels = 128;
    // mel_params.n_fft = 400;
    // mel_params.hop_length = 160;  // 10ms at 16kHz
    // mel_params.style = core_mel::Style::HF_WHISPER;
    // auto mel = core_mel::compute(pcm, n_samples, mel_params);
    // int T_mel = mel.T;
    // Max 30s: T_mel = min(T_mel, 3000)

    // ── Step 2: Conv2D subsampling (4x temporal reduction) ──────────────
    // layer0: Conv2D(1→128, k=3, s=2) + RMSNorm → [128, n_mels/2, T_mel/2]
    // layer1: Conv2D(128→32, k=3, s=2) + RMSNorm → [32, n_mels/4, T_mel/4]
    // flatten: [32 * n_mels/4, T_mel/4] = [32*32, T_sub] = [1024, T_sub]
    // input_proj: Linear(1024→1024) → [1024, T_sub]
    // T_sub ≈ T_mel / 4

    // ── Step 3: Conformer encoder (12 layers, macaron architecture) ─────
    // For each layer (residual_weight=0.5):
    //   x = x + 0.5 * FFN1(pre_ln(x))                    // macaron half-step
    //   x = x + self_attn(pre_ln(x))                      // chunked attention + relative pos
    //   x = x + LightConv1d(pre_ln(x))                    // causal depthwise conv + GLU
    //   x = x + 0.5 * FFN2(pre_ln(x))                    // macaron half-step
    //   x = out_norm(x)
    //
    // Self-attention: chunked (chunk_size=12, left_context=13, right_context=0)
    //   Uses per_dim_scale (learned, [head_dim]) + relative_k_proj for position bias
    //   Attention logit cap: 50.0 (tanh capping)
    //
    // LightConv1d: pre_ln → gate_proj (GLU split) → depthwise_conv1d(k=5, causal)
    //   → conv_norm → out_proj
    //
    // FFN: pre_ln → up_proj (SiLU) → down_proj

    // ── Step 4: Output projection + audio embedding ─────────────────────
    // output_proj: Linear(1024→1536, bias) — maps conformer output to LLM dim
    // embed_proj:  Linear(1536→1536)      — audio embedding projection

    // ── Step 5: Gemma4 LLM decode (35 layers) ───────────────────────────
    // Chat template: "<start_of_turn>user\n<audio>Transcribe...<end_of_turn>\n<start_of_turn>model\n"
    // Audio tokens injected at <audio> placeholder position
    //
    // Per-layer: PLE (per-layer embedding lookup + gate + projection) +
    //   hybrid attention (sliding_window=512 most layers, full every 5th) +
    //   pre/post norms + SwiGLU FFN
    //
    // Reuses: core_attn::kv_self_attn (Q/K RMSNorm, GQA 8Q/1KV)
    //         core_ffn::swiglu (gate_proj, up_proj, down_proj)
    //
    // Final: RMSNorm → lm_head → softcap(30.0) → greedy/sample

    // Not yet implemented — waiting for GGUF + differential testing
    if (ctx->verbosity >= 1)
        fprintf(stderr, "gemma4_e2b: forward pass not yet implemented\n");
    return nullptr;
}

extern "C" void gemma4_e2b_free(struct gemma4_e2b_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf_w)
        ggml_backend_buffer_free(ctx->model.buf_w);
    if (ctx->model.ctx_w)
        ggml_free(ctx->model.ctx_w);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" void gemma4_e2b_set_n_threads(struct gemma4_e2b_context* ctx, int n_threads) {
    if (ctx && n_threads > 0) {
        ctx->n_threads = n_threads;
        if (ctx->backend_cpu)
            ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    }
}
