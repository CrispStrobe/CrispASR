// dots_tts.cpp — rednote-hilab/dots.tts TTS runtime.
//
// Implements the dots.tts continuous AR TTS pipeline in ggml:
//   1. Qwen2.5-1.5B LLM backbone (28L, GQA 12Q/2KV, SwiGLU)
//   2. DiT flow-matching head (18L, AdaLN-Zero, Euler ODE)
//   3. PatchEncoder (24L causal transformer, latent→LLM hidden)
//   4. BigVGAN vocoder (6-stage upsample, SnakeBeta, 48 kHz)
//   5. CAM++ speaker encoder (80-mel → 512-d, optional)
//
// Inference loop (per patch of patch_size=4 latent frames):
//   a) LLM processes text + previously encoded patches → hidden states
//   b) Hidden states at audio-span positions condition the DiT
//   c) DiT denoises Gaussian noise → latent patch via Euler ODE
//   d) CFG: v = v_cond + scale × (v_cond - v_uncond)
//   e) PatchEncoder maps latent patch → LLM embedding for next step
//   f) After all patches: BigVGAN decodes latents → 48 kHz PCM
//
// Status (June 2026):
//   Phase A — conversion script + weight loading + LLM forward: DONE
//   Phase B — DiT + flow-matching ODE: DONE
//   Phase C — PatchEncoder: DONE
//   Phase D — BigVGAN vocoder: DONE
//   Phase E — end-to-end synthesis: DONE
//   Phase F — speaker encoder (CAM++): stub (TODO)
//
// Env vars:
//   DOTS_TTS_BENCH=1           — per-stage wall-clock timings
//   CRISPASR_DOTS_TTS_DEBUG=1  — verbose debug prints

#include "dots_tts.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "chatterbox_campplus.h"
#include "core/activation.h"
#include "core/adaln.h"
#include "core/attention.h"
#include "core/audio_resample.h"
#include "core/bpe.h"
#include "core/conv.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/lstm.h"
#include "core/wav_reader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation
// ===========================================================================

static bool dots_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("DOTS_TTS_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

static bool dots_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_DOTS_TTS_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct dots_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit dots_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~dots_bench_stage() {
        if (!dots_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  dots_bench: %-24s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// GGUF helpers
// ===========================================================================

static uint32_t read_u32(gguf_context* meta, const char* key, uint32_t def) {
    int idx = gguf_find_key(meta, key);
    if (idx < 0)
        return def;
    // Handle both UINT32 and INT32 (Python gguf library writes int as INT32)
    enum gguf_type t = gguf_get_kv_type(meta, idx);
    if (t == GGUF_TYPE_UINT32)
        return gguf_get_val_u32(meta, idx);
    if (t == GGUF_TYPE_INT32)
        return (uint32_t)gguf_get_val_i32(meta, idx);
    return def;
}

static float read_f32(gguf_context* meta, const char* key, float def) {
    int idx = gguf_find_key(meta, key);
    return (idx >= 0) ? gguf_get_val_f32(meta, idx) : def;
}

static ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, w);
}

// ===========================================================================
// Model weight structures
// ===========================================================================

// ── LLM (Qwen2.5-1.5B) ──

struct dots_llm_layer {
    ggml_tensor* q_proj = nullptr;
    ggml_tensor* k_proj = nullptr;
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* o_proj = nullptr;
    ggml_tensor* q_proj_b = nullptr; // Qwen2.5 has QKV biases
    ggml_tensor* k_proj_b = nullptr;
    ggml_tensor* v_proj_b = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* gate = nullptr; // SwiGLU gate
    ggml_tensor* up = nullptr;
    ggml_tensor* down = nullptr;
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* ffn_norm = nullptr;
};

struct dots_llm {
    uint32_t n_layers = 28;
    uint32_t hidden_size = 1536;
    uint32_t intermediate_size = 8960;
    uint32_t n_heads = 12;
    uint32_t n_kv_heads = 2;
    uint32_t head_dim = 128;
    uint32_t vocab_size = 151672;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;

    ggml_tensor* tok_emb = nullptr;
    ggml_tensor* final_norm = nullptr;
    ggml_tensor* lm_head = nullptr; // may be tied to tok_emb
    std::vector<dots_llm_layer> layers;
};

// ── PatchEncoder (VAESemanticEncoder) ──

struct dots_penc_layer {
    ggml_tensor* q_proj = nullptr;
    ggml_tensor* k_proj = nullptr;
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* o_proj = nullptr;
    ggml_tensor* o_proj_b = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    // 2-layer MLP (NOT SwiGLU): fc1 (hidden→ffn) + SiLU + fc2 (ffn→hidden)
    ggml_tensor* ffn_up = nullptr;     // fc1 weight
    ggml_tensor* ffn_up_b = nullptr;   // fc1 bias
    ggml_tensor* ffn_down = nullptr;   // fc2 weight
    ggml_tensor* ffn_down_b = nullptr; // fc2 bias
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* ffn_norm = nullptr;
};

struct dots_penc {
    uint32_t n_layers = 24;
    uint32_t hidden_size = 1024;
    uint32_t ffn_hidden = 4096;
    uint32_t n_heads = 16;
    uint32_t head_dim = 64;
    uint32_t input_dim = 128;
    float rope_theta = 10000.0f;

    ggml_tensor* in_proj = nullptr; // Linear(input_dim -> hidden_size)
    ggml_tensor* in_proj_b = nullptr;
    ggml_tensor* out_proj = nullptr; // Linear(hidden_size*out_ds_rate -> out_dim)
    ggml_tensor* out_proj_b = nullptr;
    ggml_tensor* ds_conv_w = nullptr; // downsample Conv1d(in,in,k=2,s=2,causal) weight
    ggml_tensor* ds_conv_b = nullptr;
    ggml_tensor* final_norm = nullptr; // unused: SuperviseEncoder has no final norm
    std::vector<dots_penc_layer> layers;
};

// ── DiT (flow-matching head) ──

struct dots_dit_block {
    ggml_tensor* adaln_w = nullptr; // (hidden, 6*hidden)
    ggml_tensor* adaln_b = nullptr;
    ggml_tensor* q_proj = nullptr;
    ggml_tensor* k_proj = nullptr;
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* o_proj = nullptr;
    ggml_tensor* o_proj_b = nullptr;
    ggml_tensor* q_norm = nullptr;
    ggml_tensor* k_norm = nullptr;
    // 2-layer MLP (NOT SwiGLU): fc1 → SiLU → fc2
    ggml_tensor* ffn_up = nullptr; // fc1
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down = nullptr; // fc2
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* ffn_norm = nullptr;
};

struct dots_dit {
    uint32_t n_layers = 18;
    uint32_t hidden_size = 1024;
    uint32_t ffn_hidden = 4096;
    uint32_t n_heads = 16;
    uint32_t head_dim = 64;
    float rope_theta = 10000.0f;

    // Timestep embedding: sinusoidal → 2-layer MLP
    ggml_tensor* time_mlp_0_w = nullptr;
    ggml_tensor* time_mlp_0_b = nullptr;
    ggml_tensor* time_mlp_1_w = nullptr;
    ggml_tensor* time_mlp_1_b = nullptr;

    // Input/output projections
    ggml_tensor* in_proj_w = nullptr;
    ggml_tensor* in_proj_b = nullptr;
    ggml_tensor* final_norm = nullptr;
    ggml_tensor* final_adaln_w = nullptr;
    ggml_tensor* final_adaln_b = nullptr;
    ggml_tensor* final_proj_w = nullptr;
    ggml_tensor* final_proj_b = nullptr;

    std::vector<dots_dit_block> blocks;
};

// ── Projections (between LLM and DiT/PatchEncoder) ──

struct dots_projections {
    ggml_tensor* hidden_proj_w = nullptr; // LLM hidden → DiT condition
    ggml_tensor* hidden_proj_b = nullptr;
    ggml_tensor* latent_proj_w = nullptr; // latent → DiT input
    ggml_tensor* latent_proj_b = nullptr;
    ggml_tensor* coord_proj_w = nullptr; // noise → DiT coordinate
    ggml_tensor* coord_proj_b = nullptr;
    ggml_tensor* xvec_proj_0_w = nullptr; // speaker emb → DiT condition
    ggml_tensor* xvec_proj_0_b = nullptr;
    ggml_tensor* xvec_proj_1_w = nullptr;
    ggml_tensor* xvec_proj_1_b = nullptr;
    ggml_tensor* eos_proj_0_w = nullptr;
    ggml_tensor* eos_proj_0_b = nullptr;
    ggml_tensor* eos_proj_1_w = nullptr;
    ggml_tensor* eos_proj_1_b = nullptr;
};

// ── BigVGAN vocoder (AudioVAE.inference_from_latents, do_sample=False) ──
// Architecture: post_proj → dec_mi(Linear→SLSTM[4×LSTM+skip]→Linear) → conv_pre
//   (non-causal) → 6× (causal ConvTranspose1d → 3× AMPBlock1 averaged) →
//   alias-free SnakeBeta → conv_post (causal) → clamp[-1,1] → mono 48 kHz PCM.
// AMPBlock1: 3× (alias-free SnakeBeta → dilated conv1 → alias-free SnakeBeta →
//   conv2 → +residual, residual added PER dilation iteration).
// Activation1d (alias-free): upsample 2× (depthwise transposed conv via
//   zero-insert + dw-conv with the checkpoint kaiser-sinc filter) → SnakeBeta →
//   downsample 2× (replicate-pad + strided dw-conv). Validated cos≈1.0 per stage.

struct dots_voc_resblock {
    // AMPBlock1: 3 dilated conv pairs (convs1[0..2] + convs2[0..2])
    // Each pair has a SnakeBeta activation before it (activations[0..5])
    ggml_tensor* act_alpha[6] = {};
    ggml_tensor* act_beta[6] = {};
    // Alias-free resample filters per activation ([1,1,12] shared, broadcast).
    ggml_tensor* act_up_filter[6] = {};
    ggml_tensor* act_down_filter[6] = {};
    ggml_tensor* convs1_w[3] = {};
    ggml_tensor* convs1_b[3] = {};
    ggml_tensor* convs2_w[3] = {};
    ggml_tensor* convs2_b[3] = {};
};

struct dots_vocoder {
    uint32_t sample_rate = 48000;
    uint32_t latent_dim = 128;
    uint32_t initial_ch = 1536;
    uint32_t mi_num_layers = 4;
    uint32_t n_stages = 6;
    std::vector<uint32_t> upsample_rates;  // [10, 6, 4, 2, 2, 2]
    std::vector<uint32_t> upsample_ksizes; // [20, 12, 8, 4, 4, 4]

    // post_proj: Conv1d(128→128, k=1) applied BEFORE dec_mi_layer (pointwise).
    ggml_tensor* post_proj_w = nullptr;
    ggml_tensor* post_proj_b = nullptr;

    // MI layer (decoder side): Linear(128→512) → LSTM(512, 4 layers) → Linear(512→128)
    ggml_tensor* mi_in_w = nullptr; // dec_mi_layer.0
    ggml_tensor* mi_in_b = nullptr;
    ggml_tensor* mi_out_w = nullptr; // dec_mi_layer.2
    ggml_tensor* mi_out_b = nullptr;
    // LSTM (4 layers, hidden=512)
    ggml_tensor* lstm_w_ih[4] = {};
    ggml_tensor* lstm_w_hh[4] = {};
    ggml_tensor* lstm_b_ih[4] = {};
    ggml_tensor* lstm_b_hh[4] = {};

    // conv_pre: Conv1d(128, 1536, k=5)
    ggml_tensor* conv_pre_w = nullptr;
    ggml_tensor* conv_pre_b = nullptr;

    // 6 upsample stages: ConvTranspose1d
    ggml_tensor* ups_w[6] = {};
    ggml_tensor* ups_b[6] = {};

    // 18 resblocks (3 per stage × 6 stages, kernel sizes cycle 3/7/11)
    dots_voc_resblock resblocks[18] = {};

    // Post activation + conv (post activation uses per-channel [24,1,12] filters)
    ggml_tensor* post_alpha = nullptr;
    ggml_tensor* post_beta = nullptr;
    ggml_tensor* post_up_filter = nullptr;
    ggml_tensor* post_down_filter = nullptr;
    ggml_tensor* post_conv_w = nullptr;
    // No post_conv_b (use_bias_at_final=false in config)

    // Weight/compute contexts
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
};

// ── Latent stats (for denormalization) ──

struct dots_latent_stats {
    std::vector<float> mean;
    std::vector<float> std;
};

// ===========================================================================
// KV cache
// ===========================================================================

// KV cache: single 4D tensor (head_dim, max_seq, n_kv_heads, n_layers)
// as required by core_attn::kv_self_attn which indexes layer via nb[3].
struct dots_kv_cache {
    ggml_tensor* k = nullptr; // (hd, max_seq, n_kv, n_layers) F16
    ggml_tensor* v = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int max_seq_len = 0;
};

static bool dots_kv_init(dots_kv_cache& kv, int n_layers, int head_dim, int n_kv_heads, int max_seq,
                         ggml_backend_t backend) {
    kv.max_seq_len = max_seq;

    size_t ctx_size = 2 * ggml_tensor_overhead() + 64;
    ggml_init_params ip = {ctx_size, nullptr, true};
    kv.ctx = ggml_init(ip);
    if (!kv.ctx)
        return false;

    kv.k = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, head_dim, max_seq, n_kv_heads, n_layers);
    kv.v = ggml_new_tensor_4d(kv.ctx, GGML_TYPE_F16, head_dim, max_seq, n_kv_heads, n_layers);
    ggml_set_name(kv.k, "kv_k");
    ggml_set_name(kv.v, "kv_v");

    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buf)
        return false;

    ggml_backend_tensor_memset(kv.k, 0, 0, ggml_nbytes(kv.k));
    ggml_backend_tensor_memset(kv.v, 0, 0, ggml_nbytes(kv.v));
    return true;
}

static void dots_kv_free(dots_kv_cache& kv) {
    if (kv.buf)
        ggml_backend_buffer_free(kv.buf);
    if (kv.ctx)
        ggml_free(kv.ctx);
    kv.buf = nullptr;
    kv.ctx = nullptr;
}

// ===========================================================================
// Context
// ===========================================================================

struct dots_tts_context {
    dots_tts_context_params params;

    // Model components
    dots_llm llm;
    dots_penc penc;
    dots_dit dit;
    dots_projections proj;
    dots_vocoder voc;
    dots_latent_stats latent_stats;

    // Tokenizer (BPE via core_bpe::tokenize_simple)
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    std::vector<std::string> id_to_token;
    int token_audio_gen_span = -1;
    int token_audio_comp_span = -1;
    int token_text_cond_end = -1;

    // Model config
    int latent_dim = 128;
    int patch_size = 4;
    float cfg_droprate = 0.2f;
    int spk_dim = 512;

    // KV caches
    dots_kv_cache llm_kv;
    dots_kv_cache penc_kv;

    // Incremental PatchEncoder streaming state (dots_penc_step): the last latent
    // frame for the stride-2 causal conv window across patches, and the count of
    // encoder tokens already in penc_kv. Reset per utterance.
    std::vector<float> penc_conv_tail; // (input_dim,); empty → zero pad (patch 0)
    int penc_n_cached = 0;

    // Backends
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // Weight contexts
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Compute
    std::vector<uint8_t> buf_compute_meta;

    // RNG
    std::mt19937 rng;

    // Vocoder loaded flag
    bool has_vocoder = false;
    bool has_speaker = false; // CAM++ encoder weights loaded
    bool has_voice = false;   // a reference voice → g_cond computed

    // Speaker embedding (if set via voice prompt)
    std::vector<float> speaker_emb; // 512-d CAM++ x-vector
    std::vector<float> g_cond;      // 1024-d xvec_proj(x-vector·scale) → DiT condition

    // CAM++ speaker encoder (separate GGUF, like the vocoder).
    core_gguf::WeightLoad spk_w;
    cb_campplus_model spk_model;
    chatterbox_campplus::cb_campplus_runtime spk_runtime;
};

// ===========================================================================
// Weight loading
// ===========================================================================

static bool dots_load_core(dots_tts_context* ctx, const char* path, int verbosity) {
    // Two-pass GGUF load
    struct gguf_init_params gip = {true, nullptr};
    gguf_context* meta = gguf_init_from_file(path, gip);
    if (!meta) {
        std::fprintf(stderr, "dots_tts: failed to open %s\n", path);
        return false;
    }

    // Read hyperparameters
    auto& llm = ctx->llm;
    llm.n_layers = read_u32(meta, "dots.llm.n_layers", 28);
    llm.hidden_size = read_u32(meta, "dots.llm.hidden_size", 1536);
    llm.intermediate_size = read_u32(meta, "dots.llm.intermediate_size", 8960);
    llm.n_heads = read_u32(meta, "dots.llm.n_heads", 12);
    llm.n_kv_heads = read_u32(meta, "dots.llm.n_kv_heads", 2);
    llm.vocab_size = read_u32(meta, "dots.llm.vocab_size", 151672);
    llm.rope_theta = read_f32(meta, "dots.llm.rope_theta", 1000000.0f);
    llm.rms_norm_eps = read_f32(meta, "dots.llm.rms_norm_eps", 1e-6f);
    llm.head_dim = llm.hidden_size / llm.n_heads;
    llm.layers.resize(llm.n_layers);

    auto& penc = ctx->penc;
    penc.n_layers = read_u32(meta, "dots.penc.n_layers", 24);
    penc.hidden_size = read_u32(meta, "dots.penc.hidden_size", 1024);
    penc.ffn_hidden = read_u32(meta, "dots.penc.ffn_hidden_size", 4096);
    penc.n_heads = read_u32(meta, "dots.penc.n_heads", 16);
    penc.input_dim = read_u32(meta, "dots.penc.input_dim", 128);
    penc.rope_theta = read_f32(meta, "dots.penc.rope_theta", 10000.0f);
    penc.head_dim = penc.hidden_size / penc.n_heads;
    penc.layers.resize(penc.n_layers);

    auto& dit = ctx->dit;
    dit.n_layers = read_u32(meta, "dots.dit.n_layers", 18);
    dit.hidden_size = read_u32(meta, "dots.dit.hidden_size", 1024);
    dit.ffn_hidden = read_u32(meta, "dots.dit.ffn_hidden_size", 4096);
    dit.n_heads = read_u32(meta, "dots.dit.n_heads", 16);
    dit.rope_theta = read_f32(meta, "dots.dit.rope_theta", 10000.0f);
    dit.head_dim = dit.hidden_size / dit.n_heads;
    dit.blocks.resize(dit.n_layers);

    ctx->latent_dim = (int)read_u32(meta, "dots.latent_dim", 128);
    ctx->patch_size = (int)read_u32(meta, "dots.patch_size", 4);
    ctx->cfg_droprate = read_f32(meta, "dots.cfg_droprate", 0.2f);
    ctx->spk_dim = (int)read_u32(meta, "dots.spk_dim", 512);

    // Special tokens
    ctx->token_audio_gen_span = (int)read_u32(meta, "dots.token.audio_gen_span", -1);
    ctx->token_audio_comp_span = (int)read_u32(meta, "dots.token.audio_comp_span", -1);
    ctx->token_text_cond_end = (int)read_u32(meta, "dots.token.text_cond_end", -1);

    if (verbosity >= 1) {
        std::fprintf(stderr, "dots_tts: LLM %uL h=%u Q=%u KV=%u vocab=%u\n", llm.n_layers, llm.hidden_size, llm.n_heads,
                     llm.n_kv_heads, llm.vocab_size);
        std::fprintf(stderr, "dots_tts: PatchEncoder %uL h=%u heads=%u\n", penc.n_layers, penc.hidden_size,
                     penc.n_heads);
        std::fprintf(stderr, "dots_tts: DiT %uL h=%u heads=%u\n", dit.n_layers, dit.hidden_size, dit.n_heads);
        std::fprintf(stderr, "dots_tts: latent_dim=%d patch_size=%d\n", ctx->latent_dim, ctx->patch_size);
    }

    // Load tokenizer from GGUF (stored as newline-joined strings, not arrays,
    // because the C GGUF reader can't handle 151K-element string arrays).
    {
        int tok_idx = gguf_find_key(meta, "dots.tokenizer.tokens");
        int mrg_idx = gguf_find_key(meta, "dots.tokenizer.merges");
        if (tok_idx >= 0 && mrg_idx >= 0) {
            const char* tok_str = gguf_get_val_str(meta, tok_idx);
            const char* mrg_str = gguf_get_val_str(meta, mrg_idx);
            // Split by newlines
            auto split_nl = [](const char* s) -> std::vector<std::string> {
                std::vector<std::string> out;
                if (!s)
                    return out;
                std::string cur;
                for (const char* p = s; *p; p++) {
                    if (*p == '\n') {
                        out.push_back(cur);
                        cur.clear();
                    } else {
                        cur.push_back(*p);
                    }
                }
                if (!cur.empty())
                    out.push_back(cur);
                return out;
            };
            auto tokens = split_nl(tok_str);
            auto merges = split_nl(mrg_str);
            ctx->id_to_token = tokens;
            for (int i = 0; i < (int)tokens.size(); i++) {
                ctx->token_to_id[tokens[i]] = i;
            }
            for (int i = 0; i < (int)merges.size(); i++) {
                ctx->merge_rank[merges[i]] = i;
            }
            if (verbosity >= 1)
                std::fprintf(stderr, "dots_tts: tokenizer %d tokens, %d merges\n", (int)tokens.size(),
                             (int)merges.size());
        } else {
            std::fprintf(stderr, "dots_tts: WARNING: tokenizer not found in GGUF\n");
        }
    }

    gguf_free(meta);

    // ── Second pass: load weight tensors via core_gguf ──
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "dots_tts", wl)) {
        std::fprintf(stderr, "dots_tts: failed to load weights from %s\n", path);
        return false;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    // ── Map tensors to model structures ──
    auto T = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(ctx->ctx_w, name); };

    // LLM
    llm.tok_emb = T("dots.llm.tok_emb.weight");
    llm.final_norm = T("dots.llm.norm.weight");
    llm.lm_head = T("dots.llm.lm_head.weight");
    // If lm_head is null (tied embeddings), use tok_emb
    if (!llm.lm_head)
        llm.lm_head = llm.tok_emb;

    for (uint32_t i = 0; i < llm.n_layers; i++) {
        auto& L = llm.layers[i];
        char buf[128];
        auto t = [&](const char* suffix) -> ggml_tensor* {
            std::snprintf(buf, sizeof(buf), "dots.llm.layers.%u.%s", i, suffix);
            return T(buf);
        };
        L.q_proj = t("q_proj.weight");
        L.k_proj = t("k_proj.weight");
        L.v_proj = t("v_proj.weight");
        L.o_proj = t("o_proj.weight");
        L.q_proj_b = t("q_proj.bias");
        L.k_proj_b = t("k_proj.bias");
        L.v_proj_b = t("v_proj.bias");
        L.q_norm = t("q_norm.weight");
        L.k_norm = t("k_norm.weight");
        L.gate = t("gate.weight");
        L.up = t("up.weight");
        L.down = t("down.weight");
        L.attn_norm = t("attn_norm.weight");
        L.ffn_norm = t("ffn_norm.weight");
    }

    // PatchEncoder
    penc.in_proj = T("dots.penc.in_proj.weight");
    penc.in_proj_b = T("dots.penc.in_proj.bias");
    penc.out_proj = T("dots.penc.out_proj.weight");
    penc.out_proj_b = T("dots.penc.out_proj.bias");
    penc.ds_conv_w = T("dots.penc.ds_conv.weight");
    penc.ds_conv_b = T("dots.penc.ds_conv.bias");
    penc.final_norm = T("dots.penc.final_norm.weight"); // absent in checkpoint (null)

    for (uint32_t i = 0; i < penc.n_layers; i++) {
        auto& L = penc.layers[i];
        char buf[128];
        auto t = [&](const char* suffix) -> ggml_tensor* {
            std::snprintf(buf, sizeof(buf), "dots.penc.layers.%u.%s", i, suffix);
            return T(buf);
        };
        L.q_proj = t("q_proj.weight");
        L.k_proj = t("k_proj.weight");
        L.v_proj = t("v_proj.weight");
        L.o_proj = t("o_proj.weight");
        L.o_proj_b = t("o_proj.bias");
        L.q_norm = t("q_norm.weight");
        L.k_norm = t("k_norm.weight");
        L.ffn_up = t("ffn_up.weight"); // fc1
        L.ffn_up_b = t("ffn_up.bias");
        L.ffn_down = t("ffn_down.weight"); // fc2
        L.ffn_down_b = t("ffn_down.bias");
        L.attn_norm = t("attn_norm.weight");
        L.ffn_norm = t("ffn_norm.weight");
    }

    // DiT
    dit.time_mlp_0_w = T("dots.dit.time_emb.mlp.0.weight");
    dit.time_mlp_0_b = T("dots.dit.time_emb.mlp.0.bias");
    dit.time_mlp_1_w = T("dots.dit.time_emb.mlp.2.weight");
    dit.time_mlp_1_b = T("dots.dit.time_emb.mlp.2.bias");
    dit.in_proj_w = T("dots.dit.in_proj.weight");
    dit.in_proj_b = T("dots.dit.in_proj.bias");
    dit.final_norm = T("dots.dit.final_norm.weight");
    // Try both name variants (conversion script had ordering bug, fixed now)
    dit.final_adaln_w = T("dots.dit.final_adaln.weight");
    if (!dit.final_adaln_w)
        dit.final_adaln_w = T("dots.dit.output_layer.adaln.weight");
    dit.final_adaln_b = T("dots.dit.final_adaln.bias");
    if (!dit.final_adaln_b)
        dit.final_adaln_b = T("dots.dit.output_layer.adaln.bias");
    dit.final_proj_w = T("dots.dit.final_proj.weight");
    dit.final_proj_b = T("dots.dit.final_proj.bias");

    for (uint32_t i = 0; i < dit.n_layers; i++) {
        auto& B = dit.blocks[i];
        char buf[128];
        auto t = [&](const char* suffix) -> ggml_tensor* {
            std::snprintf(buf, sizeof(buf), "dots.dit.blk.%u.%s", i, suffix);
            return T(buf);
        };
        B.adaln_w = t("adaln.weight");
        B.adaln_b = t("adaln.bias");
        B.q_proj = t("q_proj.weight");
        B.k_proj = t("k_proj.weight");
        B.v_proj = t("v_proj.weight");
        B.o_proj = t("o_proj.weight");
        B.o_proj_b = t("o_proj.bias");
        B.q_norm = t("q_norm.weight");
        B.k_norm = t("k_norm.weight");
        B.ffn_up = t("ffn_up.weight"); // fc1
        B.ffn_up_b = t("ffn_up.bias");
        B.ffn_down = t("ffn_down.weight"); // fc2
        B.ffn_down_b = t("ffn_down.bias");
        B.attn_norm = t("attn_norm.weight");
        B.ffn_norm = t("ffn_norm.weight");
    }

    // Projections
    auto& p = ctx->proj;
    p.hidden_proj_w = T("dots.hidden_proj.weight");
    p.hidden_proj_b = T("dots.hidden_proj.bias");
    p.latent_proj_w = T("dots.latent_proj.weight");
    p.latent_proj_b = T("dots.latent_proj.bias");
    p.coord_proj_w = T("dots.coordinate_proj.weight");
    p.coord_proj_b = T("dots.coordinate_proj.bias");
    // xvec_proj = Sequential(Linear(512→1024), LayerNorm(1024)). The checkpoint
    // stores the LayerNorm at index .1 (gamma/beta, both shape (1024,)), NOT a
    // second Linear at .2 — g_cond = LayerNorm(Linear(xvector·speaker_scale)).
    p.xvec_proj_0_w = T("dots.xvec_proj.0.weight");
    p.xvec_proj_0_b = T("dots.xvec_proj.0.bias");
    p.xvec_proj_1_w = T("dots.xvec_proj.1.weight"); // LayerNorm gamma
    p.xvec_proj_1_b = T("dots.xvec_proj.1.bias");   // LayerNorm beta
    p.eos_proj_0_w = T("dots.eos_proj.0.weight");
    p.eos_proj_0_b = T("dots.eos_proj.0.bias");
    p.eos_proj_1_w = T("dots.eos_proj.2.weight");
    p.eos_proj_1_b = T("dots.eos_proj.2.bias");

    // Latent stats. The converter stores variance under `dots.latent_stats.var`
    // (matching IOHelper, which keeps global_var and uses std = sqrt(var) at
    // denormalize time). Accept a legacy `.std` tensor too. Without this, std
    // stays empty and latents are never denormalized → vocoder garbage.
    {
        ggml_tensor* lm = T("dots.latent_stats.mean");
        ggml_tensor* lv = T("dots.latent_stats.var");
        ggml_tensor* ls = T("dots.latent_stats.std");
        if (lm && (lv || ls)) {
            int n = (int)ggml_nelements(lm);
            ctx->latent_stats.mean.resize(n);
            ctx->latent_stats.std.resize(n);
            ggml_backend_tensor_get(lm, ctx->latent_stats.mean.data(), 0, n * sizeof(float));
            if (ls) {
                ggml_backend_tensor_get(ls, ctx->latent_stats.std.data(), 0, n * sizeof(float));
            } else {
                ggml_backend_tensor_get(lv, ctx->latent_stats.std.data(), 0, n * sizeof(float));
                for (auto& v : ctx->latent_stats.std)
                    v = std::sqrt(v);
            }
            if (verbosity >= 2)
                std::fprintf(stderr, "dots_tts: latent_stats loaded (%d dims, %s)\n", n, ls ? "std" : "var→std");
        } else if (verbosity >= 1) {
            std::fprintf(stderr, "dots_tts: WARNING no latent_stats — latents will NOT be denormalized\n");
        }
    }

    if (verbosity >= 1) {
        size_t buf_size = ggml_backend_buffer_get_size(ctx->buf_w);
        std::fprintf(stderr, "dots_tts: core model loaded (%.1f MiB)\n", (double)buf_size / (1024 * 1024));
    }

    return true;
}

// ===========================================================================
// Sinusoidal timestep embedding
// ===========================================================================

static void dots_timestep_embed(float t, int dim, std::vector<float>& out) {
    // Reference (dit.py TimestepEmbedder.timestep_embedding):
    //   freqs = exp(-log(10000) * arange(half)/half); args = t*freqs
    //   embedding = cat([cos(args), sin(args)])   -- COS first, then SIN.
    out.resize(dim);
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = std::exp(-(float)i / (float)half * std::log(10000.0f));
        out[i] = std::cos(t * freq);
        out[i + half] = std::sin(t * freq);
    }
}

// ===========================================================================
// LLM forward (single step with KV cache)
// ===========================================================================

static void dots_llm_step(dots_tts_context* ctx, const float* input_embeds, int n_tokens, int n_past,
                          float* out_hidden) {
    auto& llm = ctx->llm;
    const int D = (int)llm.hidden_size;
    const int T = n_tokens;

    // Build graph
    size_t n_tensors = llm.n_layers * 80 + 512; // kv_self_attn uses ~40-60 nodes/layer
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_tensors, false);

    // Input
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "llm_input");
    ggml_set_input(x);

    // Positions for RoPE
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal mask
    ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + T, T);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    core_attn::KvSelfAttnParams atp{};
    atp.head_dim = (int)llm.head_dim;
    atp.n_heads = (int)llm.n_heads;
    atp.n_kv_heads = (int)llm.n_kv_heads;
    atp.n_kv_grp = (int)(llm.n_heads / llm.n_kv_heads);
    atp.rope_type = GGML_ROPE_TYPE_NEOX;
    atp.rope_theta = llm.rope_theta;
    atp.n_ctx_orig = 131072;
    // 1/sqrt(head_dim). KvSelfAttnParams has no default for attn_scale, so {}
    // leaves it 0 → flash_attn scale 0 → uniform attention → garbage prefill
    // (a single decode step survives only because the residual stream dominates).
    atp.attn_scale = 1.0f / std::sqrt((float)llm.head_dim);
    atp.qk_norm_eps = llm.rms_norm_eps;
    // Vulkan has no f16→f16 REPEAT pipeline, so the GQA head-expansion
    // (12Q/2KV → repeat_4d) on the F16 KV cache aborts with "Missing op:
    // REPEAT for f16 to f16" (issue #200, RX580/RADV). Casting the cached
    // K/V to F32 before the repeat lowers it to a supported F32 REPEAT.
    // Metal/CPU keep the legacy F16 fast path. Same knob as TADA (#192).
    {
        const char* bname = ggml_backend_name(ctx->backend);
        atp.force_kv_read_f32 = bname && std::strstr(bname, "Vulkan") != nullptr;
    }

    ggml_tensor* cur = x;
    for (uint32_t il = 0; il < llm.n_layers; il++) {
        auto& L = llm.layers[il];
        if (!L.attn_norm || !L.q_proj || !L.k_proj || !L.v_proj || !L.o_proj || !L.gate || !L.up || !L.down ||
            !L.ffn_norm) {
            std::fprintf(stderr, "dots_tts: LLM layer %u has null weight(s)!\n", il);
            std::fprintf(stderr, "  attn_norm=%p q=%p k=%p v=%p o=%p gate=%p up=%p down=%p ffn_norm=%p\n",
                         (void*)L.attn_norm, (void*)L.q_proj, (void*)L.k_proj, (void*)L.v_proj, (void*)L.o_proj,
                         (void*)L.gate, (void*)L.up, (void*)L.down, (void*)L.ffn_norm);
            break;
        }
        ggml_tensor* residual = cur;

        // Pre-attention RMSNorm
        cur = rms_norm(ctx0, cur, L.attn_norm, llm.rms_norm_eps);

        // Self-attention with KV cache
        if (il == 0 && dots_debug_enabled()) {
            std::fprintf(stderr, "  llm L0: cur=(%lld,%lld) q=(%lld,%lld) k=(%lld,%lld) kv=(%lld,%lld,%lld,%lld)\n",
                         (long long)cur->ne[0], (long long)cur->ne[1], (long long)L.q_proj->ne[0],
                         (long long)L.q_proj->ne[1], (long long)L.k_proj->ne[0], (long long)L.k_proj->ne[1],
                         (long long)ctx->llm_kv.k->ne[0], (long long)ctx->llm_kv.k->ne[1],
                         (long long)ctx->llm_kv.k->ne[2], (long long)ctx->llm_kv.k->ne[3]);
        }
        cur = core_attn::kv_self_attn(ctx0, gf, cur, L.q_proj, L.k_proj, L.v_proj, L.o_proj, L.q_norm, L.k_norm,
                                      positions, mask, ctx->llm_kv.k, ctx->llm_kv.v, (int)il, n_past, atp,
                                      /*qkv_w=*/nullptr, /*fixed_kv_len=*/0, /*kv_indices=*/nullptr, L.q_proj_b,
                                      L.k_proj_b, L.v_proj_b);

        cur = ggml_add(ctx0, residual, cur);
        residual = cur;

        // Pre-FFN RMSNorm + SwiGLU
        cur = rms_norm(ctx0, cur, L.ffn_norm, llm.rms_norm_eps);
        cur = core_ffn::swiglu(ctx0, cur, L.gate, L.up, L.down);
        cur = ggml_add(ctx0, residual, cur);
    }

    // Final norm
    cur = rms_norm(ctx0, cur, llm.final_norm, llm.rms_norm_eps);

    ggml_set_name(cur, "llm_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate and compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    // Set inputs
    ggml_backend_tensor_set(x, input_embeds, 0, D * T * sizeof(float));

    // Positions
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++)
        pos[i] = n_past + i;
    ggml_backend_tensor_set(positions, pos.data(), 0, T * sizeof(int32_t));

    // Causal mask (0 for allowed, -inf for masked)
    int Lk = n_past + T;
    std::vector<ggml_fp16_t> mask_data(Lk * T, ggml_fp32_to_fp16(-INFINITY));
    for (int q = 0; q < T; q++) {
        for (int k = 0; k < n_past + q + 1; k++) {
            mask_data[q * Lk + k] = ggml_fp32_to_fp16(0.0f);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(ctx->backend, gf);

    // Read output
    ggml_tensor* out = ggml_graph_get_tensor(gf, "llm_output");
    ggml_backend_tensor_get(out, out_hidden, 0, D * T * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
}

// ===========================================================================
// DiT forward (flow-matching velocity prediction, one timestep)
// ===========================================================================

// g_cond: optional (D,) global conditioning added to the timestep embedding
// to form the AdaLN condition c (reference: c = time_embedder(t) + g_cond).
// attn_mask_add: optional (fm_len*fm_len) additive mask, row-major [query][key]
//   (0 = attend, -INFINITY = block); nullptr = full bidirectional.
// pos_ids_in:   optional (fm_len) RoPE position ids; nullptr = arange(0..T-1).
static void dots_dit_forward(dots_tts_context* ctx, const float* fm_seq, int fm_len, float timestep,
                             const float* g_cond, const float* attn_mask_add, const int32_t* pos_ids_in,
                             float* out_velocity) {
    auto& dit = ctx->dit;
    const int D = (int)dit.hidden_size;
    const int T = fm_len;

    // Null-check critical tensors
    if (!dit.time_mlp_0_w || !dit.in_proj_w || !dit.final_proj_w) {
        std::fprintf(stderr, "dots_tts: DiT has null critical tensors (time_mlp_0=%p in_proj=%p final_proj=%p)\n",
                     (void*)dit.time_mlp_0_w, (void*)dit.in_proj_w, (void*)dit.final_proj_w);
        // Zero output and return
        std::memset(out_velocity, 0, fm_len * (int)dit.hidden_size * sizeof(float));
        return;
    }

    size_t n_tensors = dit.n_layers * 80 + 256; // AdaLN + attn + FFN needs ~50 nodes/block
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_tensors, false);

    // Input sequence
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "dit_input");
    ggml_set_input(x);

    // Timestep embedding: sinusoidal(t_dim) → Linear(t_dim→D) → SiLU → Linear(D→D)
    // t_dim = time_mlp_0_w.ne[0] (the input dimension of the first linear)
    const int t_dim = (int)dit.time_mlp_0_w->ne[0];
    ggml_tensor* t_emb_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, t_dim);
    ggml_set_name(t_emb_in, "t_emb");
    ggml_set_input(t_emb_in);

    ggml_tensor* t_emb = ggml_mul_mat(ctx0, dit.time_mlp_0_w, t_emb_in);
    if (dit.time_mlp_0_b)
        t_emb = ggml_add(ctx0, t_emb, dit.time_mlp_0_b);
    t_emb = ggml_silu(ctx0, t_emb);
    t_emb = ggml_mul_mat(ctx0, dit.time_mlp_1_w, t_emb);
    if (dit.time_mlp_1_b)
        t_emb = ggml_add(ctx0, t_emb, dit.time_mlp_1_b);

    // AdaLN condition c = t_emb (+ g_cond). All block/final modulation uses c.
    ggml_tensor* c = t_emb;
    ggml_tensor* g_cond_in = nullptr;
    if (g_cond) {
        g_cond_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, D);
        ggml_set_name(g_cond_in, "dit_gcond");
        ggml_set_input(g_cond_in);
        c = ggml_add(ctx0, t_emb, g_cond_in);
    }

    const bool dit_dbg = std::getenv("CRISPASR_DOTS_DIT_DEBUG") != nullptr;

    // Input projection
    ggml_tensor* cur = ggml_mul_mat(ctx0, dit.in_proj_w, x);
    if (dit.in_proj_b)
        cur = ggml_add(ctx0, cur, dit.in_proj_b);
    ggml_tensor* dbg_x0 = cur;
    ggml_tensor* dbg_b0 = nullptr;

    // Positions for RoPE (pos_ids_in, or 0..T-1 when null)
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "dit_pos");
    ggml_set_input(positions);

    // Optional additive attention mask (production FM block-causal mask).
    ggml_tensor* attn_mask_t = nullptr;
    if (attn_mask_add) {
        attn_mask_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T); // (Tk, Tq)
        ggml_set_name(attn_mask_t, "dit_mask");
        ggml_set_input(attn_mask_t);
    }

    // DiT blocks with AdaLN-Zero
    for (uint32_t il = 0; il < dit.n_layers; il++) {
        auto& B = dit.blocks[il];
        ggml_tensor* residual = cur;

        // AdaLN modulation (6-way split) from condition c = t_emb + g_cond
        core_adaln::Modulation6 mod = core_adaln::modulate6(ctx0, c, B.adaln_w, B.adaln_b, true);

        // Pre-attention: LayerNorm + modulation
        cur = core_adaln::apply_norm_modulation(ctx0, cur, mod.scale_msa, mod.shift_msa);

        // Self-attention (bidirectional, no causal mask, no KV cache)
        ggml_tensor* Q = ggml_mul_mat(ctx0, B.q_proj, cur);
        ggml_tensor* K = ggml_mul_mat(ctx0, B.k_proj, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, B.v_proj, cur);

        int hd = (int)dit.head_dim;
        int nh = (int)dit.n_heads;
        Q = ggml_reshape_3d(ctx0, Q, hd, nh, T);
        K = ggml_reshape_3d(ctx0, K, hd, nh, T);
        V = ggml_reshape_3d(ctx0, V, hd, nh, T);

        // Q/K norm
        if (B.q_norm) {
            Q = ggml_rms_norm(ctx0, Q, 1e-6f);
            Q = ggml_mul(ctx0, Q, B.q_norm);
        }
        if (B.k_norm) {
            K = ggml_rms_norm(ctx0, K, 1e-6f);
            K = ggml_mul(ctx0, K, B.k_norm);
        }

        // RoPE
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, dit.rope_theta, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, dit.rope_theta, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);

        // Scaled dot-product attention (bidirectional, manual softmax — the
        // validated penc pattern; avoids flash_attn_ext layout/precision pitfalls)
        float scale = 1.0f / std::sqrt((float)hd);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, nh)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3)); // (hd, T, nh)
        ggml_tensor* kq = ggml_mul_mat(ctx0, K, Q);             // (Tk, Tq, nh)
        kq = ggml_soft_max_ext(ctx0, kq, attn_mask_t, scale, 0.0f);
        ggml_tensor* Vp = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // (T, hd, nh)
        ggml_tensor* kqv = ggml_mul_mat(ctx0, Vp, kq);                        // (hd, Tq, nh)
        kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));           // (hd, nh, Tq)
        cur = ggml_reshape_2d(ctx0, kqv, hd * nh, T);

        // Output projection
        cur = ggml_mul_mat(ctx0, B.o_proj, cur);
        if (B.o_proj_b)
            cur = ggml_add(ctx0, cur, B.o_proj_b);

        // Gated residual
        cur = core_adaln::gated_residual(ctx0, residual, cur, mod.gate_msa);
        residual = cur;

        // Pre-FFN: LayerNorm + modulation
        cur = core_adaln::apply_norm_modulation(ctx0, cur, mod.scale_mlp, mod.shift_mlp);

        // FFN (2-layer MLP: fc1 → GELU(tanh) → fc2). DiT uses tanh-approx GELU
        // (dit.py: act_layer=nn.GELU(approximate="tanh")), NOT SiLU.
        cur = ggml_mul_mat(ctx0, B.ffn_up, cur);
        if (B.ffn_up_b)
            cur = ggml_add(ctx0, cur, B.ffn_up_b);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, B.ffn_down, cur);
        if (B.ffn_down_b)
            cur = ggml_add(ctx0, cur, B.ffn_down_b);

        // Gated residual
        cur = core_adaln::gated_residual(ctx0, residual, cur, mod.gate_mlp);
        if (il == 0)
            dbg_b0 = cur;
    }

    // Final AdaLN + projection (condition c = t_emb + g_cond).
    // dots.tts FinalLayer splits `shift, scale = adaLN(c).chunk(2)` (shift FIRST),
    // but the shared modulate2 helper labels chunk0=scale, chunk1=shift (the
    // f5/cosyvoice3 convention). So swap: use .shift (chunk1) as scale and
    // .scale (chunk0) as shift to match dots.tts.
    core_adaln::Modulation2 final_mod = core_adaln::modulate2(ctx0, c, dit.final_adaln_w, dit.final_adaln_b, true);
    cur = core_adaln::apply_norm_modulation(ctx0, cur, /*scale=*/final_mod.shift, /*shift=*/final_mod.scale);
    cur = ggml_mul_mat(ctx0, dit.final_proj_w, cur);
    if (dit.final_proj_b)
        cur = ggml_add(ctx0, cur, dit.final_proj_b);

    ggml_set_name(cur, "dit_output");
    ggml_set_output(cur);
    if (dit_dbg) {
        ggml_set_output(t_emb);
        ggml_set_name(t_emb, "dbg_temb");
        ggml_set_output(c);
        ggml_set_name(c, "dbg_c");
        ggml_set_output(dbg_x0);
        ggml_set_name(dbg_x0, "dbg_x0");
        if (dbg_b0) {
            ggml_set_output(dbg_b0);
            ggml_set_name(dbg_b0, "dbg_b0");
        }
    }
    ggml_build_forward_expand(gf, cur);

    // Allocate and compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    ggml_backend_tensor_set(x, fm_seq, 0, D * T * sizeof(float));

    // Timestep embedding (sinusoidal)
    std::vector<float> t_emb_data;
    dots_timestep_embed(timestep, t_dim, t_emb_data);
    ggml_backend_tensor_set(t_emb_in, t_emb_data.data(), 0, t_dim * sizeof(float));

    if (g_cond_in)
        ggml_backend_tensor_set(g_cond_in, g_cond, 0, D * sizeof(float));

    // Positions
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++)
        pos[i] = pos_ids_in ? pos_ids_in[i] : i;
    ggml_backend_tensor_set(positions, pos.data(), 0, T * sizeof(int32_t));

    if (attn_mask_t)
        ggml_backend_tensor_set(attn_mask_t, attn_mask_add, 0, (size_t)T * T * sizeof(float));

    ggml_backend_graph_compute(ctx->backend, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "dit_output");
    int out_dim = (int)out->ne[0];
    ggml_backend_tensor_get(out, out_velocity, 0, out_dim * T * sizeof(float));

    if (dit_dbg) {
        for (const char* nm : {"dbg_temb", "dbg_c", "dbg_x0", "dbg_b0"}) {
            ggml_tensor* dt = ggml_graph_get_tensor(gf, nm);
            if (!dt)
                continue;
            float v[6];
            ggml_backend_tensor_get(dt, v, 0, sizeof(v));
            std::fprintf(stderr, "[dit-cpp] %s:", nm);
            for (int i = 0; i < 6; i++)
                std::fprintf(stderr, " %+.4f", v[i]);
            std::fprintf(stderr, "\n");
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
}

// ===========================================================================
// PatchEncoder forward (latent patch → LLM embedding)
// ===========================================================================

// Reference: VAESemanticEncoder.decode_patch (semantic_encoder.py).
// One latent patch (in_dim, n_frames) -> one LLM embedding (out_dim).
//   1. causal Conv1d(k=2,s=2) downsample          -> out_ds_rate tokens
//   2. in_proj (Linear + bias)                     -> (hidden, out_ds_rate)
//   3. n_layers x [RMSNorm -> causal MHA(no RoPE,  -> (hidden, out_ds_rate)
//      no qk_norm) -> +res ; RMSNorm -> SiLU MLP -> +res]
//   4. concat the out_ds_rate tokens -> (hidden*out_ds_rate)
//   5. out_proj (Linear + bias)                    -> (out_dim, 1)
//
// MULTI-PATCH via full causal recompute. The reference streams one patch at a
// time with a persisted conv_tail (1 latent frame) + per-layer causal KV cache.
// Because the PatchEncoder uses NO RoPE (position-independent attention) and the
// stride-2 causal downsample window boundaries are identical whether streamed or
// run over the whole sequence at once, running the full causal encoder over ALL
// accumulated latents and grouping the encoder tokens is mathematically IDENTICAL
// to the incremental decode_patch loop. So this takes the entire latent history
// (in_dim, n_frames) [n_frames a multiple of patch_size] and emits one LLM
// embedding (out_dim) per patch -> out_embeds is (n_patches, out_dim). The caller
// (AR loop) uses only the last; the diff harness checks all. The symmetric pad=1
// + crop reproduces the causal left_padding=1 conv exactly (window 0 = [pad,f0]).
static void dots_penc_forward(dots_tts_context* ctx, const float* latents, int n_frames, float* out_embeds) {
    auto& pe = ctx->penc;
    const int in_dim = (int)pe.input_dim;        // 128
    const int H = (int)pe.hidden_size;           // 1024
    const int nh = (int)pe.n_heads;              // 16
    const int hd = (int)pe.head_dim;             // 64
    const int T_in = n_frames;                   // all accumulated latent frames
    const int T = T_in / 2;                      // encoder tokens after stride-2 ds
    const int out_ds_rate = ctx->patch_size / 2; // encoder tokens grouped per LLM embed
    const int n_groups = T / out_ds_rate;        // == n_frames / patch_size patches

    size_t n_tensors = pe.n_layers * 64 + 256;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_tensors, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_dim, T_in); // (in_dim, T_in)
    ggml_set_name(x, "penc_input");
    ggml_set_input(x);

    // Causal attention mask over the T new tokens (F32: 0 keep, -inf mask).
    ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "penc_mask");
    ggml_set_input(mask);

    // ── 1. Downsample conv (k=2, s=2, causal). conv_tail=0 for patch 0, so a
    //        symmetric pad=1 conv reproduces windows [0,f0],[f1,f2],... ; crop.
    ggml_tensor* xT = ggml_cont(ctx0, ggml_transpose(ctx0, x));                        // (T_in, in_dim)
    ggml_tensor* ds = ggml_conv_1d(ctx0, pe.ds_conv_w, xT, /*s*/ 2, /*p*/ 1, /*d*/ 1); // (T_ds, in_dim)
    ds = ggml_cont(ctx0, ggml_view_2d(ctx0, ds, T, in_dim, ds->nb[1], 0));             // first T rows
    ds = ggml_cont(ctx0, ggml_transpose(ctx0, ds));                                    // (in_dim, T)
    if (pe.ds_conv_b)
        ds = ggml_add(ctx0, ds, pe.ds_conv_b);

    // ── 2. in_proj (Linear 128 -> 1024) + bias
    ggml_tensor* cur = ggml_mul_mat(ctx0, pe.in_proj, ds); // (H, T)
    if (pe.in_proj_b)
        cur = ggml_add(ctx0, cur, pe.in_proj_b);

    const float scale = 1.0f / std::sqrt((float)hd);

    // ── 3. Encoder layers
    for (uint32_t il = 0; il < pe.n_layers; il++) {
        auto& L = pe.layers[il];
        if (!L.attn_norm || !L.q_proj || !L.k_proj || !L.v_proj || !L.o_proj || !L.ffn_norm || !L.ffn_up ||
            !L.ffn_down) {
            std::fprintf(stderr, "dots_tts: penc layer %u has null weight(s)!\n", il);
            break;
        }

        // Attention: RMSNorm -> MHA (no RoPE, no qk_norm) + residual
        ggml_tensor* residual = cur;
        ggml_tensor* h = rms_norm(ctx0, cur, L.attn_norm, 1e-6f);
        ggml_tensor* Q = ggml_mul_mat(ctx0, L.q_proj, h); // (H, T)
        ggml_tensor* K = ggml_mul_mat(ctx0, L.k_proj, h);
        ggml_tensor* V = ggml_mul_mat(ctx0, L.v_proj, h);
        Q = ggml_reshape_3d(ctx0, Q, hd, nh, T); // (hd, nh, T)
        K = ggml_reshape_3d(ctx0, K, hd, nh, T);
        V = ggml_reshape_3d(ctx0, V, hd, nh, T);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, nh)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3)); // (hd, T, nh)
        ggml_tensor* kq = ggml_mul_mat(ctx0, K, Q);             // (Tk, Tq, nh)
        kq = ggml_soft_max_ext(ctx0, kq, mask, scale, 0.0f);
        ggml_tensor* Vp = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // (T, hd, nh)
        ggml_tensor* kqv = ggml_mul_mat(ctx0, Vp, kq);                        // (hd, Tq, nh)
        kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));           // (hd, nh, Tq)
        ggml_tensor* attn = ggml_reshape_2d(ctx0, kqv, H, T);                 // (H, T)
        attn = ggml_mul_mat(ctx0, L.o_proj, attn);
        if (L.o_proj_b)
            attn = ggml_add(ctx0, attn, L.o_proj_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN: RMSNorm -> fc1 -> SiLU -> fc2 + residual
        residual = cur;
        h = rms_norm(ctx0, cur, L.ffn_norm, 1e-6f);
        h = ggml_mul_mat(ctx0, L.ffn_up, h);
        if (L.ffn_up_b)
            h = ggml_add(ctx0, h, L.ffn_up_b);
        h = ggml_silu(ctx0, h);
        h = ggml_mul_mat(ctx0, L.ffn_down, h);
        if (L.ffn_down_b)
            h = ggml_add(ctx0, h, L.ffn_down_b);
        cur = ggml_add(ctx0, residual, h);
    }

    // ── 4. group consecutive out_ds_rate tokens: (H, T) -> (H*out_ds_rate,
    //        n_groups). Memory is [tok0(H) | tok1(H) | ...] so a reshape places
    //        consecutive token pairs side-by-side per column (== rearrange
    //        "(s d) h -> s (d h)", d=out_ds_rate).
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, H * out_ds_rate, n_groups);

    // ── 5. out_proj (Linear H*out_ds_rate -> out_dim) + bias
    cur = ggml_mul_mat(ctx0, pe.out_proj, cur); // (out_dim, n_groups)
    if (pe.out_proj_b)
        cur = ggml_add(ctx0, cur, pe.out_proj_b);

    ggml_set_name(cur, "penc_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    ggml_backend_tensor_set(x, latents, 0, (size_t)in_dim * T_in * sizeof(float));

    // Causal mask: query q attends to keys k<=q. Layout (Tk, Tq): elem(k,q).
    std::vector<float> mask_data((size_t)T * T, 0.0f);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++)
            mask_data[(size_t)q * T + k] = (k <= q) ? 0.0f : -INFINITY;
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

    ggml_backend_graph_compute(ctx->backend, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "penc_output");
    ggml_backend_tensor_get(out, out_embeds, 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
}

// Reset the incremental PatchEncoder stream — call once per utterance before the
// AR loop so the first dots_penc_step starts from an empty cache + zero conv pad.
static void dots_penc_reset(dots_tts_context* ctx) {
    ctx->penc_conv_tail.clear();
    ctx->penc_n_cached = 0;
}

// Incremental PatchEncoder: append ONE patch (input_dim × patch_size new latent
// frames) and emit its single LLM embedding (out_dim) using the persistent
// penc_kv cache + conv_tail. Mathematically the streaming form of
// dots_penc_forward (decode_patch with conv_tail + per-layer causal KV), but
// O(N) per patch vs the full recompute's O(N²) — so long utterances stay linear.
//
// The PatchEncoder uses NO RoPE, so core_attn::kv_self_attn is reused with
// all-zero positions: RoPE at position 0 is the identity (no rotation), giving
// exactly the position-independent attention the encoder expects, while the
// proven cache write/read mechanics handle the streaming K/V.
static void dots_penc_step(dots_tts_context* ctx, const float* new_latents, float* out_embed) {
    auto& pe = ctx->penc;
    const int in_dim = (int)pe.input_dim;
    const int H = (int)pe.hidden_size;
    const int nh = (int)pe.n_heads;
    const int hd = (int)pe.head_dim;
    const int patch = ctx->patch_size;           // new latent frames this step
    const int out_ds_rate = ctx->patch_size / 2; // new encoder tokens (== 2)
    const int T = out_ds_rate;
    const int n_past = ctx->penc_n_cached;

    size_t n_tensors = pe.n_layers * 80 + 256;
    ggml_init_params ip = {n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_tensors, false);

    // Conv input: [conv_tail | patch new frames] = (patch+1) frames. The stride-2
    // k=2 (p=0) conv yields `out_ds_rate` tokens covering [tail,f0],[f1,f2],...,
    // exactly matching the full recompute's symmetric-pad windows. For patch 0
    // the tail is zero (== the recompute's left pad).
    const int T_conv = patch + 1;
    ggml_tensor* cin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, in_dim, T_conv); // (in_dim, T_conv)
    ggml_set_name(cin, "penc_step_in");
    ggml_set_input(cin);

    ggml_tensor* cinT = ggml_cont(ctx0, ggml_transpose(ctx0, cin));                      // (T_conv, in_dim)
    ggml_tensor* ds = ggml_conv_1d(ctx0, pe.ds_conv_w, cinT, /*s*/ 2, /*p*/ 0, /*d*/ 1); // (T, in_dim)
    ds = ggml_cont(ctx0, ggml_transpose(ctx0, ds));                                      // (in_dim, T)
    if (pe.ds_conv_b)
        ds = ggml_add(ctx0, ds, pe.ds_conv_b);

    ggml_tensor* cur = ggml_mul_mat(ctx0, pe.in_proj, ds); // (H, T)
    if (pe.in_proj_b)
        cur = ggml_add(ctx0, cur, pe.in_proj_b);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T); // all 0 → no-RoPE
    ggml_set_name(positions, "penc_step_pos");
    ggml_set_input(positions);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + T, T);
    ggml_set_name(mask, "penc_step_mask");
    ggml_set_input(mask);

    core_attn::KvSelfAttnParams atp{};
    atp.head_dim = hd;
    atp.n_heads = nh;
    atp.n_kv_heads = nh; // MHA: grp == 1, no GQA expansion
    atp.n_kv_grp = 1;
    atp.attn_scale = 1.0f / std::sqrt((float)hd);
    atp.rope_theta = 10000.0f; // unused at position 0
    atp.n_ctx_orig = 4096;
    atp.qk_norm_eps = 1e-6f; // unused (no q/k norm)

    for (uint32_t il = 0; il < pe.n_layers; il++) {
        auto& L = pe.layers[il];
        ggml_tensor* residual = cur;
        ggml_tensor* h = rms_norm(ctx0, cur, L.attn_norm, 1e-6f);
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, h, L.q_proj, L.k_proj, L.v_proj, L.o_proj, /*q_norm=*/nullptr, /*k_norm=*/nullptr, positions,
            mask, ctx->penc_kv.k, ctx->penc_kv.v, (int)il, n_past, atp, /*qkv_w=*/nullptr, /*fixed_kv_len=*/0,
            /*kv_indices=*/nullptr, /*q_b=*/nullptr, /*k_b=*/nullptr, /*v_b=*/nullptr, /*o_b=*/L.o_proj_b);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        h = rms_norm(ctx0, cur, L.ffn_norm, 1e-6f);
        h = ggml_mul_mat(ctx0, L.ffn_up, h);
        if (L.ffn_up_b)
            h = ggml_add(ctx0, h, L.ffn_up_b);
        h = ggml_silu(ctx0, h);
        h = ggml_mul_mat(ctx0, L.ffn_down, h);
        if (L.ffn_down_b)
            h = ggml_add(ctx0, h, L.ffn_down_b);
        cur = ggml_add(ctx0, residual, h);
    }

    // Group the out_ds_rate tokens → (H*out_ds_rate, 1) → out_proj → (out_dim, 1).
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_2d(ctx0, cur, H * out_ds_rate, 1);
    cur = ggml_mul_mat(ctx0, pe.out_proj, cur);
    if (pe.out_proj_b)
        cur = ggml_add(ctx0, cur, pe.out_proj_b);
    ggml_set_name(cur, "penc_step_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    std::vector<float> conv_in((size_t)in_dim * T_conv, 0.0f);
    if (!ctx->penc_conv_tail.empty())
        std::memcpy(conv_in.data(), ctx->penc_conv_tail.data(), (size_t)in_dim * sizeof(float));
    std::memcpy(conv_in.data() + in_dim, new_latents, (size_t)in_dim * patch * sizeof(float));
    ggml_backend_tensor_set(cin, conv_in.data(), 0, conv_in.size() * sizeof(float));

    std::vector<int32_t> pos((size_t)T, 0);
    ggml_backend_tensor_set(positions, pos.data(), 0, (size_t)T * sizeof(int32_t));

    int Lk = n_past + T;
    std::vector<ggml_fp16_t> mask_data((size_t)Lk * T, ggml_fp32_to_fp16(-INFINITY));
    for (int q = 0; q < T; q++)
        for (int k = 0; k < n_past + q + 1; k++)
            mask_data[(size_t)q * Lk + k] = ggml_fp32_to_fp16(0.0f);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(ctx->backend, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "penc_step_out");
    ggml_backend_tensor_get(out, out_embed, 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);

    // Advance stream state: cache grew by T tokens; the last new frame becomes the
    // conv_tail for the next patch's first stride-2 window.
    ctx->penc_n_cached += T;
    ctx->penc_conv_tail.assign(new_latents + (size_t)(patch - 1) * in_dim, new_latents + (size_t)patch * in_dim);
}

// ===========================================================================
// Flow-matching ODE solver (Euler method)
// ===========================================================================

// Small Linear (y = x·Wᵀ + b) on the model backend, handling quantized W via a
// tiny one-shot ggml graph. in: (n_rows, in_dim) row-major; returns (n_rows, out_dim).
static std::vector<float> dots_linear(dots_tts_context* ctx, ggml_tensor* w, ggml_tensor* b, const float* in,
                                      int n_rows, int in_dim, int out_dim) {
    size_t n_tensors = 8;
    ggml_init_params ip = {n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* c0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(c0);
    ggml_tensor* x = ggml_new_tensor_2d(c0, GGML_TYPE_F32, in_dim, n_rows);
    ggml_set_input(x);
    ggml_tensor* y = ggml_mul_mat(c0, w, x); // (out_dim, n_rows)
    if (b)
        y = ggml_add(c0, y, b);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(x, in, 0, (size_t)in_dim * n_rows * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);
    std::vector<float> out((size_t)out_dim * n_rows);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(ga);
    ggml_free(c0);
    return out;
}

// Flow-matching ODE driver (core.fm_solver_step + _flow_matching_step_fm).
//   input_seq / cfg_seq: (fm_total_len, dit_dim) row-major. The two CFG branches
//     differ only in the history hidden slots (real hidden_proj vs hidden_proj(0)).
//   latent_start: first of the trailing patch_size noise slots.
//   attn_mask_add: (fm_total_len²) additive [q][k] mask or nullptr (bidirectional).
//   pos_ids:       (fm_total_len) RoPE ids or nullptr (arange).
//   noise:         (patch_size, latent_dim) ODE initial coordinate (caller-seeded).
// Writes the raw (un-denormalized) latent (patch_size, latent_dim) to out_latent.
static void dots_flow_match_core(dots_tts_context* ctx, const float* input_seq, const float* cfg_seq, int fm_total_len,
                                 int latent_start, const float* attn_mask_add, const int32_t* pos_ids,
                                 const float* noise, int num_steps, float cfg_scale, float* out_latent,
                                 const float* g_cond = nullptr) {
    const int dit_dim = (int)ctx->dit.hidden_size;
    const int latent_dim = ctx->latent_dim;
    const int patch_size = fm_total_len - latent_start;
    const int n_latent = patch_size * latent_dim;
    auto& p = ctx->proj;

    std::vector<float> z(noise, noise + n_latent);
    // Mutable per-branch sequences; only the latent slots change each ODE step.
    std::vector<float> seq_c(input_seq, input_seq + (size_t)fm_total_len * dit_dim);
    std::vector<float> seq_u(cfg_seq, cfg_seq + (size_t)fm_total_len * dit_dim);

    const float dt = 1.0f / (float)num_steps;
    std::vector<float> vel_c((size_t)fm_total_len * latent_dim);
    std::vector<float> vel_u((size_t)fm_total_len * latent_dim);

    for (int step = 0; step < num_steps; step++) {
        float t = (float)step * dt;

        // coordinate_proj(z): (patch_size, latent_dim) → (patch_size, dit_dim)
        std::vector<float> z_proj =
            dots_linear(ctx, p.coord_proj_w, p.coord_proj_b, z.data(), patch_size, latent_dim, dit_dim);
        std::memcpy(seq_c.data() + (size_t)latent_start * dit_dim, z_proj.data(),
                    (size_t)patch_size * dit_dim * sizeof(float));
        std::memcpy(seq_u.data() + (size_t)latent_start * dit_dim, z_proj.data(),
                    (size_t)patch_size * dit_dim * sizeof(float));

        // CFG branches as two B=1 DiT forwards. For voice cloning the COND
        // branch carries the speaker g_cond and the UNCOND branch carries zero
        // (reference fm_solver_step: DiT g_cond=[g, 0]). Text-only → both null.
        dots_dit_forward(ctx, seq_c.data(), fm_total_len, t, g_cond, attn_mask_add, pos_ids, vel_c.data());
        dots_dit_forward(ctx, seq_u.data(), fm_total_len, t, /*g_cond=*/nullptr, attn_mask_add, pos_ids, vel_u.data());

        // Euler step on the noise-slot velocities with classifier-free guidance.
        for (int i = 0; i < patch_size; i++) {
            for (int d = 0; d < latent_dim; d++) {
                int src = (latent_start + i) * latent_dim + d;
                float vc = vel_c[src];
                float vu = vel_u[src];
                float v = vc + cfg_scale * (vc - vu);
                z[i * latent_dim + d] += dt * v;
            }
        }
    }

    std::memcpy(out_latent, z.data(), n_latent * sizeof(float));
}

// EOS predictor (core.eos_proj): Linear(llm_dim→llm_dim) → SiLU → Linear(→2).
// Returns the stop probability = softmax(logits)[1] (reference
// _should_stop_after_current_audio: eos_proj(h).softmax(-1)[..., 1] > threshold).
static float dots_eos_prob(dots_tts_context* ctx, const float* llm_hidden) {
    auto& p = ctx->proj;
    const int llm_dim = (int)ctx->llm.hidden_size;
    if (!p.eos_proj_0_w || !p.eos_proj_1_w)
        return 0.0f;
    std::vector<float> h = dots_linear(ctx, p.eos_proj_0_w, p.eos_proj_0_b, llm_hidden, 1, llm_dim, llm_dim);
    for (auto& v : h)
        v = v / (1.0f + std::exp(-v)); // SiLU
    std::vector<float> logits = dots_linear(ctx, p.eos_proj_1_w, p.eos_proj_1_b, h.data(), 1, llm_dim, 2);
    float m = std::max(logits[0], logits[1]);
    float e0 = std::exp(logits[0] - m), e1 = std::exp(logits[1] - m);
    return e1 / (e0 + e1);
}

// ===========================================================================
// BigVGAN vocoder decode (latents → 48 kHz PCM)
// ===========================================================================

// ── Conv1d helper (channel-first: (C, T) in, (C_out, T_out) out) ────────────
// Weight layout: (K, C_in, C_out) in ggml. dots Conv1d: causal => full left-pad
// dilation*(K-1); non-causal => symmetric pad dilation*(K-1)/2 (output T == in T).
static ggml_tensor* dots_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int dilation = 1,
                                bool causal = true) {
    int K = (int)w->ne[0];
    int Cout = (int)w->ne[2];
    int T_in = (int)x->ne[1];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    if (causal) {
        int pad = (K - 1) * dilation; // full left-pad, then crop right
        ggml_tensor* y = ggml_conv_1d(ctx, w, xT, /*stride*/ 1, pad, dilation);
        int Tout = (int)y->ne[0];
        y = ggml_reshape_2d(ctx, y, Tout, Cout);
        if (Tout > T_in) {
            y = ggml_view_2d(ctx, y, T_in, Cout, y->nb[1], 0); // keep first T_in
            y = ggml_cont(ctx, y);
        }
        y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
        if (b)
            y = ggml_add(ctx, y, b);
        return y;
    }
    int pad = dilation * (K - 1) / 2; // symmetric
    ggml_tensor* y = ggml_conv_1d(ctx, w, xT, /*stride*/ 1, pad, dilation);
    int Tout = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, Tout, Cout);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b)
        y = ggml_add(ctx, y, b);
    return y;
}

// ── ConvTranspose1d (upsample) ──────────────────────────────────────────────
// Weight: (K, C_out, C_in). Stride = upsample rate. dots causal ConvTranspose1d
// (k == 2*stride, padding 0): raw output length stride*(T+1); the causal forward
// drops the LAST `stride` samples → keep first stride*T.
static ggml_tensor* dots_convt1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride) {
    int Cout = (int)w->ne[1];
    int T_in = (int)x->ne[1];

    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));          // (T, Cin)
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xT, stride, 0, 1); // (T_raw, Cout)
    int T_raw = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, T_raw, Cout);
    int T_out = stride * T_in; // causal: keep first stride*T (drop last `stride`)
    if (T_raw > T_out) {
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_raw * sizeof(float), 0); // keep first T_out
        y = ggml_cont(ctx, y);
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b)
        y = ggml_add(ctx, y, b);
    return y;
}

// ── Alias-free SnakeBeta activation (BigVGAN v2 Activation1d, causal) ─────────
// x (C,T) channel-first -> (C,T). upsample 2x (depthwise transposed conv with
// the kaiser-sinc filter) -> SnakeBeta -> downsample 2x. The filters are
// symmetric, so the transposed conv == a zero-inserted depthwise conv (no flip),
// and PyTorch conv1d == ggml cross-correlation directly.
//   up:   zero-insert along T -> dw-conv(filter, pad 11) -> keep first 2T, x2
//   down: replicate-pad-left 11 -> dw-conv(filter, stride 2)
static ggml_tensor* dots_alias_free_act(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta,
                                        ggml_tensor* up_filter, ggml_tensor* down_filter) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];
    const int K = (int)up_filter->ne[0]; // 12

    // Per-channel depthwise kernels (K,1,C): broadcast shared [K,1,1] filters.
    auto kern = [&](ggml_tensor* f) -> ggml_tensor* {
        if ((int)f->ne[2] == C)
            return f; // already per-channel ([K,1,C])
        ggml_tensor* tgt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, C);
        return ggml_repeat(ctx, ggml_reshape_3d(ctx, f, K, 1, 1), tgt);
    };
    ggml_tensor* upk = kern(up_filter);
    ggml_tensor* dnk = kern(down_filter);

    // Upsample: zero-insert (C,T)->(C,2T) [x at even t, 0 at odd t].
    ggml_tensor* x3 = ggml_reshape_3d(ctx, x, C, 1, T);
    ggml_tensor* z3 = ggml_scale(ctx, x3, 0.0f);
    ggml_tensor* xc = ggml_concat(ctx, x3, z3, 1);                                      // (C,2,T)
    ggml_tensor* x_up = ggml_reshape_2d(ctx, xc, C, 2 * T);                             // (C,2T)
    ggml_tensor* x_up_t = ggml_cont(ctx, ggml_transpose(ctx, x_up));                    // (2T,C)
    ggml_tensor* up = ggml_conv_1d_dw(ctx, upk, x_up_t, /*s*/ 1, /*p*/ K - 1, /*d*/ 1); // (2T+K-1, C)
    up = ggml_cont(ctx, ggml_view_2d(ctx, up, 2 * T, C, up->nb[1], 0));                 // keep first 2T
    up = ggml_scale(ctx, up, 2.0f);
    ggml_tensor* up_cf = ggml_cont(ctx, ggml_transpose(ctx, up)); // (C,2T)

    // SnakeBeta on the upsampled signal.
    ggml_tensor* s = core_act::snake_beta(ctx, up_cf, alpha, beta); // (C,2T)

    // Downsample: replicate-pad-left (K-1), depthwise conv stride 2.
    ggml_tensor* s_t = ggml_cont(ctx, ggml_transpose(ctx, s));        // (2T,C)
    ggml_tensor* first = ggml_view_2d(ctx, s_t, 1, C, s_t->nb[1], 0); // (1,C) first time-step
    ggml_tensor* pad = ggml_repeat(ctx, first, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K - 1, C)); // (K-1,C)
    ggml_tensor* s_pad = ggml_concat(ctx, pad, s_t, 0);                                           // (2T+K-1, C)
    ggml_tensor* dn = ggml_conv_1d_dw(ctx, dnk, s_pad, /*s*/ 2, /*p*/ 0, /*d*/ 1);                // (T, C)
    return ggml_cont(ctx, ggml_transpose(ctx, dn));                                               // (C,T)
}

// ── AMPBlock1 forward (one resblock) ────────────────────────────────────────
// Structure: 3× (alias-free SnakeBeta → dilated_conv1 → alias-free SnakeBeta → conv2) + residual
static ggml_tensor* dots_resblock_fwd(ggml_context* ctx, ggml_tensor* x, const dots_voc_resblock& rb,
                                      const int dilations[3]) {
    for (int d = 0; d < 3; d++) {
        // AMPBlock1: residual added PER dilation iteration (x = conv_chain(x) + x).
        ggml_tensor* xt = dots_alias_free_act(ctx, x, rb.act_alpha[d * 2], rb.act_beta[d * 2], rb.act_up_filter[d * 2],
                                              rb.act_down_filter[d * 2]);
        xt = dots_conv1d(ctx, xt, rb.convs1_w[d], rb.convs1_b[d], dilations[d]);
        xt = dots_alias_free_act(ctx, xt, rb.act_alpha[d * 2 + 1], rb.act_beta[d * 2 + 1], rb.act_up_filter[d * 2 + 1],
                                 rb.act_down_filter[d * 2 + 1]);
        xt = dots_conv1d(ctx, xt, rb.convs2_w[d], rb.convs2_b[d], 1);
        x = ggml_add(ctx, xt, x);
    }
    return x;
}

// ── BigVGAN vocoder decode ──────────────────────────────────────────────────
static float* dots_vocoder_decode(dots_tts_context* ctx, const float* latents, int n_frames, int* out_n_samples) {
    if (!ctx->has_vocoder) {
        std::fprintf(stderr, "dots_tts: vocoder not loaded\n");
        *out_n_samples = 0;
        return nullptr;
    }

    dots_bench_stage bench("vocoder_graph");
    auto& voc = ctx->voc;

    int total_upsample = 1;
    for (auto r : voc.upsample_rates)
        total_upsample *= (int)r;
    int n_samples = n_frames * total_upsample;
    int latent_dim = (int)voc.latent_dim;

    if (dots_debug_enabled()) {
        std::fprintf(stderr, "dots_tts: vocoder decode %d frames → %d samples (%.1f s at %d Hz)\n", n_frames, n_samples,
                     (float)n_samples / (float)voc.sample_rate, (int)voc.sample_rate);
    }

    // Build ggml graph
    // The graph is large: 6 upsample stages × (conv_transpose + 3 resblocks × 8 ops each)
    // 6 stages × 3 resblocks × (6 snake_beta + 6 conv + 6 add) + MI-LSTM + conv_pre/post
    // MI-LSTM alone creates ~40K nodes (1024 frames × 4 layers × ~10 ops/step)
    size_t n_tensors = 131072;
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        std::fprintf(stderr, "dots_tts: vocoder graph context alloc failed\n");
        *out_n_samples = 0;
        return nullptr;
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_tensors, false);

    // Input: latents (latent_dim, n_frames) — channel-first
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, latent_dim, n_frames);
    ggml_set_name(x, "voc_input");
    ggml_set_input(x);

    // Step 0: post_proj (pointwise Conv1d 128→128) — applied before the MI layer.
    if (voc.post_proj_w)
        x = dots_conv1d(ctx0, x, voc.post_proj_w, voc.post_proj_b, 1, /*causal=*/false);

    // Step 1: dec_mi_layer — Linear(128→512) → SLSTM[4-layer LSTM(512)+skip] → Linear(512→128)
    if (voc.mi_in_w && voc.lstm_w_ih[0]) {
        x = ggml_mul_mat(ctx0, voc.mi_in_w, x);
        if (voc.mi_in_b)
            x = ggml_add(ctx0, x, voc.mi_in_b);
        const int lstm_h = 512;
        ggml_tensor* lstm_in = x; // SLSTM skip residual is the LSTM input
        for (int li = 0; li < 4; li++) {
            if (!voc.lstm_w_ih[li])
                break;
            x = core_lstm::lstm_unidir(ctx0, gf, x, voc.lstm_w_ih[li], voc.lstm_w_hh[li], voc.lstm_b_ih[li],
                                       voc.lstm_b_hh[li], lstm_h, /*reverse=*/false);
        }
        x = ggml_add(ctx0, x, lstm_in); // SLSTM skip: y = LSTM(x) + x
        x = ggml_mul_mat(ctx0, voc.mi_out_w, x);
        if (voc.mi_out_b)
            x = ggml_add(ctx0, x, voc.mi_out_b);
    }

    // Step 2: conv_pre — Conv1d(128→1536, k=5), NON-causal (symmetric pad 2)
    x = dots_conv1d(ctx0, x, voc.conv_pre_w, voc.conv_pre_b, 1, /*causal=*/false);

    // Step 3: 6 upsample stages — causal ConvTranspose1d, then 3 AMPBlock1 averaged.
    const int dilations[3] = {1, 3, 5};
    int n_stages = (int)voc.n_stages;

    for (int si = 0; si < n_stages; si++) {
        int stride = (int)voc.upsample_rates[si];
        x = dots_convt1d(ctx0, x, voc.ups_w[si], voc.ups_b[si], stride);

        int rb_base = si * 3;
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < 3; j++) {
            ggml_tensor* rj = dots_resblock_fwd(ctx0, x, voc.resblocks[rb_base + j], dilations);
            xs = xs ? ggml_add(ctx0, xs, rj) : rj;
        }
        x = ggml_scale(ctx0, xs, 1.0f / 3.0f);
    }

    // Step 4: alias-free SnakeBeta (post) + conv_post (causal, k=7, no bias)
    x = dots_alias_free_act(ctx0, x, voc.post_alpha, voc.post_beta, voc.post_up_filter, voc.post_down_filter);
    x = dots_conv1d(ctx0, x, voc.post_conv_w, nullptr, 1, /*causal=*/true);
    // dots config use_tanh_at_final=false → clamp to [-1, 1] (not tanh).
    x = ggml_clamp(ctx0, x, -1.0f, 1.0f);

    // Output: (1, n_samples) — mono
    ggml_set_name(x, "voc_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // Allocate and compute
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "dots_tts: vocoder graph alloc failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        *out_n_samples = 0;
        return nullptr;
    }

    // Set input — latents in channel-first layout (latent_dim, n_frames)
    // Input data is (n_frames, latent_dim) row-major → transpose needed
    // Actually our latents are stored as flat (n_frames * latent_dim), which in
    // ggml 2D (latent_dim, n_frames) is already the correct memory layout.
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "voc_input"), latents, 0, latent_dim * n_frames * sizeof(float));

    ggml_backend_graph_compute(ctx->backend, gf);

    // Read output
    ggml_tensor* out = ggml_graph_get_tensor(gf, "voc_output");
    int out_len = (int)ggml_nelements(out);
    float* pcm = (float*)std::malloc(out_len * sizeof(float));
    if (!pcm) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        *out_n_samples = 0;
        return nullptr;
    }
    ggml_backend_tensor_get(out, pcm, 0, out_len * sizeof(float));
    *out_n_samples = out_len;

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return pcm;
}

// ===========================================================================
// Token embedding lookup
// ===========================================================================

static void dots_embed_tokens(dots_tts_context* ctx, const int32_t* token_ids, int n_tokens, float* out_embeds) {
    auto& llm = ctx->llm;
    int D = (int)llm.hidden_size;

    // Use ggml_get_rows to handle quantized embedding tensors (Q4_K etc).
    // tok_emb shape: (hidden_size, vocab_size) in ggml convention.
    size_t ctx_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params ip = {ctx_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "tok_ids");
    ggml_set_input(ids);

    // ggml_get_rows: extracts rows from the embedding table → (D, n_tokens) F32
    ggml_tensor* emb = ggml_get_rows(ctx0, llm.tok_emb, ids);
    ggml_set_name(emb, "tok_emb_out");
    ggml_set_output(emb);

    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, emb);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    ggml_backend_tensor_set(ids, token_ids, 0, n_tokens * sizeof(int32_t));
    ggml_backend_graph_compute(ctx->backend, gf);

    ggml_tensor* out = ggml_graph_get_tensor(gf, "tok_emb_out");
    ggml_backend_tensor_get(out, out_embeds, 0, D * n_tokens * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
}

// ===========================================================================
// Tokenize text
// ===========================================================================

static std::vector<int32_t> dots_tokenize(dots_tts_context* ctx, const std::string& text) {
    return core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, text);
}

// ===========================================================================
// Public API
// ===========================================================================

struct dots_tts_context_params dots_tts_context_default_params(void) {
    return {
        /*.n_threads    =*/4,
        /*.verbosity    =*/1,
        /*.use_gpu      =*/false,
        /*.temperature  =*/0.7f,
        /*.seed         =*/42,
        /*.max_patches  =*/256,
        /*.ode_steps    =*/16,
        /*.cfg_scale    =*/3.0f,
        /*.eos_threshold=*/0.8f,
        /*.flash_attn   =*/false,
    };
}

struct dots_tts_context* dots_tts_init_from_file(const char* path_model, struct dots_tts_context_params params) {
    if (!path_model || !*path_model)
        return nullptr;
    auto* ctx = new dots_tts_context();
    ctx->params = params;
    ctx->rng.seed(params.seed > 0 ? params.seed : 42);

    // Initialize backend. Every dots.tts graph runs on a single backend via raw
    // ggml_gallocr (no ggml_backend_sched), so there are no cross-backend copy
    // hazards — GPU is just init_best() with all weights + KV caches resident on
    // it. Opt out with CRISPASR_DOTS_TTS_CPU=1.
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);

    const char* cpu_env = std::getenv("CRISPASR_DOTS_TTS_CPU");
    const bool force_cpu = cpu_env && *cpu_env && *cpu_env != '0';
    if (params.use_gpu && !force_cpu) {
        ctx->backend = ggml_backend_init_best();
        if (!ctx->backend)
            ctx->backend = ctx->backend_cpu;
    } else {
        ctx->backend = ctx->backend_cpu;
    }
    if (params.verbosity >= 1)
        std::fprintf(stderr, "dots_tts: backend = %s\n", ggml_backend_name(ctx->backend));

    // Load core model
    if (!dots_load_core(ctx, path_model, params.verbosity)) {
        delete ctx;
        return nullptr;
    }

    // Initialize LLM KV cache
    int max_llm_seq = 2048; // text + audio spans
    if (!dots_kv_init(ctx->llm_kv, (int)ctx->llm.n_layers, (int)ctx->llm.head_dim, (int)ctx->llm.n_kv_heads,
                      max_llm_seq, ctx->backend)) {
        std::fprintf(stderr, "dots_tts: failed to init LLM KV cache\n");
        delete ctx;
        return nullptr;
    }

    // Initialize PatchEncoder KV cache
    int max_penc_seq = 1024;
    if (!dots_kv_init(ctx->penc_kv, (int)ctx->penc.n_layers, (int)ctx->penc.head_dim, (int)ctx->penc.n_heads,
                      max_penc_seq, ctx->backend)) {
        std::fprintf(stderr, "dots_tts: failed to init PatchEncoder KV cache\n");
        delete ctx;
        return nullptr;
    }

    if (params.verbosity >= 1) {
        std::fprintf(stderr, "dots_tts: ready (LLM KV %d, PEnc KV %d)\n", max_llm_seq, max_penc_seq);
    }

    return ctx;
}

int dots_tts_set_vocoder_path(struct dots_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    struct gguf_init_params gip = {true, nullptr};
    gguf_context* meta = gguf_init_from_file(path, gip);
    if (!meta)
        return -1;

    auto& voc = ctx->voc;
    voc.sample_rate = read_u32(meta, "dots.voc.sample_rate", 48000);
    voc.latent_dim = read_u32(meta, "dots.voc.latent_dim", 128);
    voc.initial_ch = read_u32(meta, "dots.voc.upsample_initial_channel", 1536);
    voc.mi_num_layers = read_u32(meta, "dots.voc.mi_num_layers", 4);

    // Read array hyperparams (GGUF int32 arrays via gguf_get_arr_data)
    auto read_i32_arr = [&](const char* key, std::vector<uint32_t>& out) {
        int idx = gguf_find_key(meta, key);
        if (idx < 0)
            return;
        int n = (int)gguf_get_arr_n(meta, idx);
        const int32_t* data = (const int32_t*)gguf_get_arr_data(meta, idx);
        out.resize(n);
        for (int i = 0; i < n; i++)
            out[i] = (uint32_t)data[i];
    };
    read_i32_arr("dots.voc.upsample_rates", voc.upsample_rates);
    read_i32_arr("dots.voc.upsample_kernel_sizes", voc.upsample_ksizes);
    // resblock kernel sizes are baked into the weight shapes (3, 7, 11)

    gguf_free(meta);

    // Load vocoder weights via core_gguf
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "dots_voc", wl)) {
        return -1;
    }
    voc.ctx_w = wl.ctx;
    voc.buf_w = wl.buf;
    voc.n_stages = (uint32_t)voc.upsample_rates.size();

    // ── Map tensors to vocoder struct ──
    auto VT = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(voc.ctx_w, name); };

    // post_proj (pointwise Conv1d, applied before the MI layer)
    voc.post_proj_w = VT("dots.voc.post_proj.weight");
    voc.post_proj_b = VT("dots.voc.post_proj.bias");

    // MI layer (decoder side)
    voc.mi_in_w = VT("dots.voc.dec_mi_layer.0.weight");
    voc.mi_in_b = VT("dots.voc.dec_mi_layer.0.bias");
    voc.mi_out_w = VT("dots.voc.dec_mi_layer.2.weight");
    voc.mi_out_b = VT("dots.voc.dec_mi_layer.2.bias");
    for (int i = 0; i < 4; i++) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "dots.voc.dec_mi_layer.1.lstm.weight_ih_l%d", i);
        voc.lstm_w_ih[i] = VT(buf);
        std::snprintf(buf, sizeof(buf), "dots.voc.dec_mi_layer.1.lstm.weight_hh_l%d", i);
        voc.lstm_w_hh[i] = VT(buf);
        std::snprintf(buf, sizeof(buf), "dots.voc.dec_mi_layer.1.lstm.bias_ih_l%d", i);
        voc.lstm_b_ih[i] = VT(buf);
        std::snprintf(buf, sizeof(buf), "dots.voc.dec_mi_layer.1.lstm.bias_hh_l%d", i);
        voc.lstm_b_hh[i] = VT(buf);
    }

    // conv_pre
    voc.conv_pre_w = VT("dots.voc.decoder.conv_pre.weight");
    voc.conv_pre_b = VT("dots.voc.decoder.conv_pre.bias");

    // Upsample stages
    for (uint32_t i = 0; i < voc.n_stages && i < 6; i++) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "dots.voc.decoder.ups.%u.0.weight", i);
        voc.ups_w[i] = VT(buf);
        std::snprintf(buf, sizeof(buf), "dots.voc.decoder.ups.%u.0.bias", i);
        voc.ups_b[i] = VT(buf);
    }

    // Resblocks (3 per stage, 18 total)
    for (int rb = 0; rb < 18; rb++) {
        auto& R = voc.resblocks[rb];
        char buf[128];
        // 6 activations (SnakeBeta alpha/beta pairs + alias-free filters)
        for (int a = 0; a < 6; a++) {
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.activations.%d.act.alpha", rb, a);
            R.act_alpha[a] = VT(buf);
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.activations.%d.act.beta", rb, a);
            R.act_beta[a] = VT(buf);
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.activations.%d.upsample.filter", rb, a);
            R.act_up_filter[a] = VT(buf);
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.activations.%d.downsample.lowpass.filter",
                          rb, a);
            R.act_down_filter[a] = VT(buf);
        }
        // 3 conv pairs
        for (int c = 0; c < 3; c++) {
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.convs1.%d.weight", rb, c);
            R.convs1_w[c] = VT(buf);
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.convs1.%d.bias", rb, c);
            R.convs1_b[c] = VT(buf);
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.convs2.%d.weight", rb, c);
            R.convs2_w[c] = VT(buf);
            std::snprintf(buf, sizeof(buf), "dots.voc.decoder.resblocks.%d.convs2.%d.bias", rb, c);
            R.convs2_b[c] = VT(buf);
        }
    }

    // Post activation + conv
    voc.post_alpha = VT("dots.voc.decoder.activation_post.act.alpha");
    voc.post_beta = VT("dots.voc.decoder.activation_post.act.beta");
    voc.post_up_filter = VT("dots.voc.decoder.activation_post.upsample.filter");
    voc.post_down_filter = VT("dots.voc.decoder.activation_post.downsample.lowpass.filter");
    voc.post_conv_w = VT("dots.voc.decoder.conv_post.weight");

    ctx->has_vocoder = true;

    if (ctx->params.verbosity >= 1) {
        size_t buf_size = ggml_backend_buffer_get_size(voc.buf_w);
        std::fprintf(stderr, "dots_tts: vocoder loaded (%.1f MiB, %u Hz, %u stages)\n",
                     (double)buf_size / (1024 * 1024), voc.sample_rate, voc.n_stages);
        // Verify critical tensors
        int missing = 0;
        if (!voc.conv_pre_w) {
            missing++;
            std::fprintf(stderr, "  MISSING: conv_pre_w\n");
        }
        if (!voc.post_conv_w) {
            missing++;
            std::fprintf(stderr, "  MISSING: post_conv_w\n");
        }
        for (uint32_t i = 0; i < voc.n_stages; i++) {
            if (!voc.ups_w[i]) {
                missing++;
                std::fprintf(stderr, "  MISSING: ups_w[%u]\n", i);
            }
        }
        if (missing > 0)
            std::fprintf(stderr, "dots_tts: WARNING: %d vocoder tensors missing\n", missing);
    }
    return 0;
}

// Read a (possibly F16) GGUF tensor into a host f32 vector (row-major flat).
static std::vector<float> dots_read_f32(ggml_tensor* t) {
    if (!t)
        return {};
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t)n);
        ggml_backend_tensor_get(t, tmp.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
    } else {
        return {};
    }
    return out;
}

// Bind the 3D-Speaker CAM++ tensors from a loaded `dots-tts-soar-spk` GGUF
// (`dots.spk.model.*`) into a cb_campplus_model. Same backbone as chatterbox's
// CAMPPlus, only the names differ (`xv.`→`xvector.`, `.nl.bn.`→
// `.nonlinear.batchnorm.`, dense-layer `l1`→`linear1`, `cam.{ll,l1,l2}`→
// `cam_layer.{linear_local,linear1,linear2}`, `nonl{1,2}.bn`→
// `nonlinear{1,2}.batchnorm`, `out_nl`→`out_nonlinear`).
static bool dots_bind_campplus(core_gguf::WeightLoad& w, cb_campplus_model& m) {
    auto* C = w.ctx;
    auto T = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(C, name); };
    char k[160];
    const char* P = "dots.spk.model";

    auto bind_unit = [&](cb_campplus_unit& u, const char* base) {
        std::snprintf(k, sizeof(k), "%s.linear.weight", base), u.lin_w = T(k);
        std::snprintf(k, sizeof(k), "%s.linear.bias", base), u.lin_b = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear.batchnorm.weight", base), u.bn_w = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear.batchnorm.bias", base), u.bn_b = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear.batchnorm.running_mean", base), u.bn_m = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear.batchnorm.running_var", base), u.bn_v = T(k);
    };
    auto bind_resblock = [&](cb_campplus_resblock& b, const char* base) {
        std::snprintf(k, sizeof(k), "%s.conv1.weight", base), b.conv1_w = T(k);
        std::snprintf(k, sizeof(k), "%s.conv1.bias", base), b.conv1_b = T(k);
        std::snprintf(k, sizeof(k), "%s.bn1.weight", base), b.bn1_w = T(k);
        std::snprintf(k, sizeof(k), "%s.bn1.bias", base), b.bn1_b = T(k);
        std::snprintf(k, sizeof(k), "%s.bn1.running_mean", base), b.bn1_m = T(k);
        std::snprintf(k, sizeof(k), "%s.bn1.running_var", base), b.bn1_v = T(k);
        std::snprintf(k, sizeof(k), "%s.conv2.weight", base), b.conv2_w = T(k);
        std::snprintf(k, sizeof(k), "%s.conv2.bias", base), b.conv2_b = T(k);
        std::snprintf(k, sizeof(k), "%s.bn2.weight", base), b.bn2_w = T(k);
        std::snprintf(k, sizeof(k), "%s.bn2.bias", base), b.bn2_b = T(k);
        std::snprintf(k, sizeof(k), "%s.bn2.running_mean", base), b.bn2_m = T(k);
        std::snprintf(k, sizeof(k), "%s.bn2.running_var", base), b.bn2_v = T(k);
        std::snprintf(k, sizeof(k), "%s.shortcut.0.weight", base), b.sc_w = T(k);
        std::snprintf(k, sizeof(k), "%s.shortcut.0.bias", base), b.sc_b = T(k);
        std::snprintf(k, sizeof(k), "%s.shortcut.1.weight", base), b.sc_bn_w = T(k);
        std::snprintf(k, sizeof(k), "%s.shortcut.1.bias", base), b.sc_bn_b = T(k);
        std::snprintf(k, sizeof(k), "%s.shortcut.1.running_mean", base), b.sc_bn_m = T(k);
        std::snprintf(k, sizeof(k), "%s.shortcut.1.running_var", base), b.sc_bn_v = T(k);
    };
    auto bind_dense_layer = [&](cb_campplus_dense_layer& l, const char* base) {
        std::snprintf(k, sizeof(k), "%s.nonlinear1.batchnorm.weight", base), l.nonl1_bn_w = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear1.batchnorm.bias", base), l.nonl1_bn_b = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear1.batchnorm.running_mean", base), l.nonl1_bn_m = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear1.batchnorm.running_var", base), l.nonl1_bn_v = T(k);
        std::snprintf(k, sizeof(k), "%s.linear1.weight", base), l.l1_w = T(k);
        std::snprintf(k, sizeof(k), "%s.linear1.bias", base), l.l1_b = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear2.batchnorm.weight", base), l.nonl2_bn_w = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear2.batchnorm.bias", base), l.nonl2_bn_b = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear2.batchnorm.running_mean", base), l.nonl2_bn_m = T(k);
        std::snprintf(k, sizeof(k), "%s.nonlinear2.batchnorm.running_var", base), l.nonl2_bn_v = T(k);
        std::snprintf(k, sizeof(k), "%s.cam_layer.linear_local.weight", base), l.cam_ll_w = T(k);
        std::snprintf(k, sizeof(k), "%s.cam_layer.linear1.weight", base), l.cam_l1_w = T(k);
        std::snprintf(k, sizeof(k), "%s.cam_layer.linear1.bias", base), l.cam_l1_b = T(k);
        std::snprintf(k, sizeof(k), "%s.cam_layer.linear2.weight", base), l.cam_l2_w = T(k);
        std::snprintf(k, sizeof(k), "%s.cam_layer.linear2.bias", base), l.cam_l2_b = T(k);
    };

    // FCM head.
    auto& head = m.head;
    std::snprintf(k, sizeof(k), "%s.head.conv1.weight", P), head.conv1_w = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn1.weight", P), head.bn1_w = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn1.bias", P), head.bn1_b = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn1.running_mean", P), head.bn1_m = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn1.running_var", P), head.bn1_v = T(k);
    std::snprintf(k, sizeof(k), "%s.head.conv2.weight", P), head.conv2_w = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn2.weight", P), head.bn2_w = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn2.bias", P), head.bn2_b = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn2.running_mean", P), head.bn2_m = T(k);
    std::snprintf(k, sizeof(k), "%s.head.bn2.running_var", P), head.bn2_v = T(k);
    head.layer1.assign(2, cb_campplus_resblock{});
    head.layer2.assign(2, cb_campplus_resblock{});
    for (int i = 0; i < 2; i++) {
        char base[96];
        std::snprintf(base, sizeof(base), "%s.head.layer1.%d", P, i), bind_resblock(head.layer1[i], base);
        std::snprintf(base, sizeof(base), "%s.head.layer2.%d", P, i), bind_resblock(head.layer2[i], base);
    }
    head.layer1[0].stride = 2; // first block of each layer downsamples H by 2
    head.layer2[0].stride = 2;

    // xvector chain.
    char base[96];
    std::snprintf(base, sizeof(base), "%s.xvector.tdnn", P), bind_unit(m.tdnn, base);
    std::snprintf(base, sizeof(base), "%s.xvector.transit1", P), bind_unit(m.transit1, base);
    std::snprintf(base, sizeof(base), "%s.xvector.transit2", P), bind_unit(m.transit2, base);
    std::snprintf(base, sizeof(base), "%s.xvector.transit3", P), bind_unit(m.transit3, base);
    // out_nonlinear is a bare BN (no Linear) at `out_nonlinear.batchnorm.*`.
    m.out_nl.lin_w = nullptr;
    m.out_nl.lin_b = nullptr;
    std::snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.weight", P), m.out_nl.bn_w = T(k);
    std::snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.bias", P), m.out_nl.bn_b = T(k);
    std::snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.running_mean", P), m.out_nl.bn_m = T(k);
    std::snprintf(k, sizeof(k), "%s.xvector.out_nonlinear.batchnorm.running_var", P), m.out_nl.bn_v = T(k);
    // dense: Conv1d(1024→512) + BN(affine=False, running_* only).
    std::snprintf(k, sizeof(k), "%s.xvector.dense.linear.weight", P), m.dense.lin_w = T(k);
    std::snprintf(k, sizeof(k), "%s.xvector.dense.nonlinear.batchnorm.running_mean", P), m.dense.bn_m = T(k);
    std::snprintf(k, sizeof(k), "%s.xvector.dense.nonlinear.batchnorm.running_var", P), m.dense.bn_v = T(k);

    // Dense blocks: 12 / 24 / 16 layers, dilation 1 / 2 / 2, names tdnnd1..N (1-indexed).
    const int nlayers[3] = {12, 24, 16};
    const int dils[3] = {1, 2, 2};
    cb_campplus_dense_block* blocks[3] = {&m.block1, &m.block2, &m.block3};
    for (int bi = 0; bi < 3; bi++) {
        auto& blk = *blocks[bi];
        blk.num_layers = nlayers[bi];
        blk.dilation = dils[bi];
        blk.layers.assign(nlayers[bi], cb_campplus_dense_layer{});
        for (int li = 0; li < nlayers[bi]; li++) {
            std::snprintf(base, sizeof(base), "%s.xvector.block%d.tdnnd%d", P, bi + 1, li + 1);
            bind_dense_layer(blk.layers[li], base);
        }
    }

    if (!m.head.conv1_w || !m.tdnn.lin_w || !m.dense.lin_w || m.block1.layers.empty() || !m.block1.layers[0].l1_w) {
        std::fprintf(stderr, "dots_tts: CAM++ bind failed (missing core tensors)\n");
        return false;
    }
    return true;
}

// Compute g_cond (1024-d DiT global conditioning) from a 16 kHz mono PCM
// reference: x-vector = CAM++(pcm) (512-d, 3D-Speaker stats var floor 1e-2),
// then g_cond = LayerNorm(Linear(x-vector · speaker_scale)). Mirrors
// model.py: `g_cond = core.xvec_proj(speaker_embedding * speaker_scale)`.
static bool dots_compute_g_cond(dots_tts_context* ctx, const float* pcm16k, int n_samples, float speaker_scale) {
    if (!ctx->has_speaker) {
        std::fprintf(stderr, "dots_tts: no speaker encoder loaded (call set_speaker_path first)\n");
        return false;
    }
    auto xvec = chatterbox_campplus::embed_speaker(ctx->spk_model, ctx->spk_runtime, pcm16k, n_samples,
                                                   /*stats_var_floor=*/1e-2f);
    if ((int)xvec.size() != ctx->spk_dim) {
        std::fprintf(stderr, "dots_tts: x-vector size %zu != spk_dim %d\n", xvec.size(), ctx->spk_dim);
        return false;
    }
    ctx->speaker_emb = xvec;

    // xvec_proj.0 = Linear(spk_dim → fm_hidden). Weight ne=(in, out) → flat W[o*in+i].
    auto W0 = dots_read_f32(ctx->proj.xvec_proj_0_w);
    auto b0 = dots_read_f32(ctx->proj.xvec_proj_0_b);
    auto gamma = dots_read_f32(ctx->proj.xvec_proj_1_w);
    auto beta = dots_read_f32(ctx->proj.xvec_proj_1_b);
    const int in_dim = ctx->spk_dim;
    const int out_dim = (int)b0.size();
    if ((int)W0.size() != in_dim * out_dim || (int)gamma.size() != out_dim || (int)beta.size() != out_dim) {
        std::fprintf(stderr, "dots_tts: xvec_proj dims bad (W0=%zu b0=%zu ln=%zu)\n", W0.size(), b0.size(),
                     gamma.size());
        return false;
    }
    std::vector<float> h((size_t)out_dim);
    for (int o = 0; o < out_dim; o++) {
        double acc = (double)b0[(size_t)o];
        const float* wr = W0.data() + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            acc += (double)wr[i] * ((double)xvec[(size_t)i] * (double)speaker_scale);
        h[(size_t)o] = (float)acc;
    }
    // LayerNorm(out_dim), eps 1e-5, biased variance.
    double mean = 0.0;
    for (int o = 0; o < out_dim; o++)
        mean += (double)h[(size_t)o];
    mean /= (double)out_dim;
    double var = 0.0;
    for (int o = 0; o < out_dim; o++) {
        double d = (double)h[(size_t)o] - mean;
        var += d * d;
    }
    var /= (double)out_dim;
    const double inv = 1.0 / std::sqrt(var + 1e-5);
    ctx->g_cond.assign((size_t)out_dim, 0.0f);
    for (int o = 0; o < out_dim; o++)
        ctx->g_cond[(size_t)o] =
            (float)(((double)h[(size_t)o] - mean) * inv * (double)gamma[(size_t)o] + (double)beta[(size_t)o]);
    ctx->has_voice = true;
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "dots_tts: voice g_cond ready (xvec %d-d, g_cond %d-d)\n", in_dim, out_dim);
    return true;
}

int dots_tts_set_speaker_path(struct dots_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    if (!core_gguf::load_weights(path, ctx->backend, "dots-spk", ctx->spk_w)) {
        std::fprintf(stderr, "dots_tts: failed to load speaker encoder %s\n", path);
        ctx->has_speaker = false;
        return -1;
    }
    if (!dots_bind_campplus(ctx->spk_w, ctx->spk_model)) {
        ctx->has_speaker = false;
        return -1;
    }
    ctx->has_speaker = true;
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "dots_tts: CAM++ speaker encoder loaded (%s)\n", path);
    return 0;
}

int dots_tts_set_speaker_pcm(struct dots_tts_context* ctx, const float* pcm_16k, int n_samples) {
    if (!ctx || !pcm_16k || n_samples <= 0)
        return -1;
    // CAM++ crops to 10 s (model.py max_audio_seconds); cap the input here so
    // the deterministic (start=0) crop matches the reference.
    const int cap = 16000 * 10;
    if (n_samples > cap)
        n_samples = cap;
    return dots_compute_g_cond(ctx, pcm_16k, n_samples, /*speaker_scale=*/1.5f) ? 0 : -1;
}

int dots_tts_set_voice_prompt(struct dots_tts_context* ctx, const char* wav_path) {
    if (!ctx || !wav_path)
        return -1;
    if (!ctx->has_speaker) {
        std::fprintf(stderr, "dots_tts: set a speaker encoder GGUF before a voice prompt\n");
        return -1;
    }
    std::vector<float> pcm;
    int sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(wav_path, pcm, sr) || pcm.empty()) {
        std::fprintf(stderr, "dots_tts: failed to read reference WAV %s\n", wav_path);
        return -1;
    }
    if (sr != 16000) {
        pcm = core_audio::resample_polyphase(pcm.data(), (int)pcm.size(), sr, 16000);
    }
    return dots_tts_set_speaker_pcm(ctx, pcm.data(), (int)pcm.size());
}

int dots_tts_set_voice_enabled(struct dots_tts_context* ctx, int on) {
    if (!ctx)
        return -1;
    // Toggle voice conditioning without discarding the computed g_cond, so the
    // CLI can synthesize the neutral spoken AI-disclaimer with the default voice
    // (params.tts_voice cleared) and then re-enable the cloned voice. A no-op
    // when no reference voice was ever computed.
    ctx->has_voice = (on != 0) && !ctx->g_cond.empty();
    return 0;
}

float* dots_tts_synthesize(struct dots_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    dots_bench_stage bench_total("synthesize_total");

    // 1. Build the prefill token sequence matching the reference generation
    //    template (tts_pipeline DEFAULT_TRAIN_TEMPLATE = "[文本]{text}[文本对应语音]{audio}"
    //    → build_generation_schedule): [文本] text [文本对应语音] <|audio_gen_start|>.
    //    The two Chinese prefixes are FIXED strings → hardcode their Qwen2.5 BPE
    //    ids so conditioning is exact regardless of the C++ pretokenizer. Without
    //    this wrapper the LLM conditioning is wrong → garbled audio + no EOS.
    //    (text_cond_end is the INTERLEAVE template; not used here.)
    static const int32_t TTS_TEXT_PREFIX_IDS[] = {58, 108704, 60};                  // "[文本]"
    static const int32_t TTS_AUDIO_PREFIX_IDS[] = {58, 108704, 103124, 105761, 60}; // "[文本对应语音]"
    const int audio_gen_start = ctx->token_audio_gen_span >= 0 ? ctx->token_audio_gen_span - 1 : 151668;

    std::string input_text(text);
    std::vector<int32_t> text_tok = dots_tokenize(ctx, input_text);
    if (text_tok.empty()) {
        std::fprintf(stderr, "dots_tts: tokenization failed\n");
        return nullptr;
    }
    std::vector<int32_t> token_ids;
    for (int32_t t : TTS_TEXT_PREFIX_IDS)
        token_ids.push_back(t);
    token_ids.insert(token_ids.end(), text_tok.begin(), text_tok.end());
    for (int32_t t : TTS_AUDIO_PREFIX_IDS)
        token_ids.push_back(t);
    token_ids.push_back(audio_gen_start);

    int n_text = (int)token_ids.size();
    if (ctx->params.verbosity >= 2) {
        std::fprintf(stderr, "dots_tts: prefill %d tokens:", n_text);
        for (int32_t t : token_ids)
            std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    // 2. Embed text tokens
    int llm_dim = (int)ctx->llm.hidden_size;
    std::vector<float> text_embeds(n_text * llm_dim);
    {
        dots_bench_stage b("embed_tokens");
        dots_embed_tokens(ctx, token_ids.data(), n_text, text_embeds.data());
    }

    // 3. Autoregressive patch generation loop (reference model.py _decode /
    //    _consume_audio_patch). The FM sequence is the growing interleave
    //    [h0, l0(×patch), h1, l1(×patch), ...] where h_i = hidden_proj(llm_hidden)
    //    and l_i = latent_proj(NORMALIZED latent). The penc + vocoder consume the
    //    DENORMALIZED latents. EOS via eos_proj stops generation.
    int max_patches = ctx->params.max_patches > 0 ? ctx->params.max_patches : 256;
    if (const char* e = std::getenv("CRISPASR_DOTS_MAX_PATCHES")) {
        int m = std::atoi(e);
        if (m > 0)
            max_patches = m;
    }
    int latent_dim = ctx->latent_dim;
    int patch_size = ctx->patch_size;
    int dit_dim = (int)ctx->dit.hidden_size;
    int ode_steps = ctx->params.ode_steps > 0 ? ctx->params.ode_steps : 16;
    if (const char* e = std::getenv("CRISPASR_DOTS_ODE_STEPS")) {
        int s = std::atoi(e);
        if (s > 0)
            ode_steps = s;
    }
    float cfg_scale = ctx->params.cfg_scale > 0.0f ? ctx->params.cfg_scale : 3.0f;
    float eos_threshold = ctx->params.eos_threshold > 0.0f ? ctx->params.eos_threshold : 0.8f;
    if (const char* e = std::getenv("CRISPASR_DOTS_EOS_THRESHOLD"))
        eos_threshold = std::atof(e);
    auto& proj = ctx->proj;

    std::vector<float> all_latents; // accumulated DENORMALIZED latents (penc + vocoder)
    std::vector<float> fm_c, fm_u;  // growing FM sequence (cond + cfg), dit_dim units
    int fm_seq_len = 0;             // tokens currently in fm_c/fm_u
    int llm_n_past = 0;

    // Append one hidden chunk: hidden_proj(h) into cond, hidden_proj(0) into cfg.
    std::vector<float> zero_llm(llm_dim, 0.0f);
    auto append_hidden = [&](const float* llm_hidden) {
        std::vector<float> hp =
            dots_linear(ctx, proj.hidden_proj_w, proj.hidden_proj_b, llm_hidden, 1, llm_dim, dit_dim);
        std::vector<float> hn =
            dots_linear(ctx, proj.hidden_proj_w, proj.hidden_proj_b, zero_llm.data(), 1, llm_dim, dit_dim);
        fm_c.insert(fm_c.end(), hp.begin(), hp.end());
        fm_u.insert(fm_u.end(), hn.begin(), hn.end());
        fm_seq_len += 1;
    };
    // Append one latent chunk (NORMALIZED): latent_proj(z) into BOTH branches.
    auto append_latent = [&](const float* z_norm) {
        std::vector<float> lp =
            dots_linear(ctx, proj.latent_proj_w, proj.latent_proj_b, z_norm, patch_size, latent_dim, dit_dim);
        fm_c.insert(fm_c.end(), lp.begin(), lp.end());
        fm_u.insert(fm_u.end(), lp.begin(), lp.end());
        fm_seq_len += patch_size;
    };

    // Prefill: run LLM on text tokens; seed the FM sequence with the last hidden.
    std::vector<float> cur_hidden(llm_dim);
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "dots_tts: prefill %d tokens...\n", n_text);
    {
        dots_bench_stage b("llm_prefill");
        std::vector<float> hidden(n_text * llm_dim);
        dots_llm_step(ctx, text_embeds.data(), n_text, 0, hidden.data());
        llm_n_past = n_text;
        std::memcpy(cur_hidden.data(), hidden.data() + (size_t)(n_text - 1) * llm_dim, llm_dim * sizeof(float));
    }
    append_hidden(cur_hidden.data()); // h0

    dots_penc_reset(ctx); // start the incremental PatchEncoder stream
    const char* penc_verify_env = std::getenv("CRISPASR_DOTS_PENC_VERIFY");
    const bool penc_verify = penc_verify_env && *penc_verify_env && *penc_verify_env != '0';

    int n_patches_done = 0;
    for (int patch_idx = 0; patch_idx < max_patches; patch_idx++) {
        dots_bench_stage b_patch("patch_generate");

        // EOS check (decide stop BEFORE decoding this patch, but still emit it).
        float eos_p = dots_eos_prob(ctx, cur_hidden.data());
        bool stop_after = eos_p > eos_threshold;

        // Flow-matching: trailing patch_size noise slots after the current seq.
        int latent_start = fm_seq_len;
        int fm_total_len = fm_seq_len + patch_size;
        std::vector<float> seq_c(fm_c);
        std::vector<float> seq_u(fm_u);
        seq_c.resize((size_t)fm_total_len * dit_dim, 0.0f);
        seq_u.resize((size_t)fm_total_len * dit_dim, 0.0f);

        // Block-causal FM attn mask + pos ids (model._build_fm_attn_mask/_pos_ids,
        // hidden_patch_size=1, no history bucket padding → latent_start=fm_seq_len).
        std::vector<float> mask((size_t)fm_total_len * fm_total_len, -INFINITY);
        std::vector<int32_t> pos(fm_total_len, 0);
        {
            auto attend = [&](int q, int k) { mask[(size_t)q * fm_total_len + k] = 0.0f; };
            int block_start = fm_seq_len - 1; // hidden_patch_size = 1
            for (int q = 0; q < block_start; q++)
                for (int k = 0; k <= q; k++)
                    attend(q, k);
            for (int q = block_start; q < fm_seq_len; q++) {
                for (int k = 0; k < fm_seq_len; k++)
                    attend(q, k);
                for (int k = latent_start; k < fm_total_len; k++)
                    attend(q, k);
            }
            for (int q = latent_start; q < fm_total_len; q++) {
                for (int k = 0; k < fm_seq_len; k++)
                    attend(q, k);
                for (int k = latent_start; k < fm_total_len; k++)
                    attend(q, k);
            }
            for (int i = 0; i < fm_seq_len; i++)
                pos[i] = i;
            for (int i = 0; i < patch_size; i++)
                pos[latent_start + i] = fm_seq_len + i;
        }

        std::vector<float> noise((size_t)patch_size * latent_dim);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& v : noise)
            v = normal(ctx->rng);

        std::vector<float> z_norm((size_t)patch_size * latent_dim);
        {
            dots_bench_stage b2("flow_match");
            dots_flow_match_core(ctx, seq_c.data(), seq_u.data(), fm_total_len, latent_start, mask.data(), pos.data(),
                                 noise.data(), ode_steps, cfg_scale, z_norm.data(),
                                 ctx->has_voice ? ctx->g_cond.data() : nullptr);
        }

        // Denormalize for penc + vocoder; keep z_norm for FM history.
        std::vector<float> z_denorm(z_norm);
        if (!ctx->latent_stats.std.empty()) {
            for (int i = 0; i < patch_size; i++)
                for (int d = 0; d < latent_dim; d++)
                    z_denorm[i * latent_dim + d] =
                        z_norm[i * latent_dim + d] * ctx->latent_stats.std[d] + ctx->latent_stats.mean[d];
        }
        all_latents.insert(all_latents.end(), z_denorm.begin(), z_denorm.end());

        // Grow FM history with the NORMALIZED latent.
        append_latent(z_norm.data());

        // Incremental PatchEncoder: stream this patch's denorm latents → its one
        // LLM embedding via the persistent KV cache (O(N) per patch). Identical
        // to the full recompute but linear; CRISPASR_DOTS_PENC_VERIFY=1 runs both
        // and prints the cosine so the streaming path can be audited.
        std::vector<float> emb_step((size_t)llm_dim);
        {
            dots_bench_stage b2("penc_step");
            dots_penc_step(ctx, z_denorm.data(), emb_step.data());
        }
        if (penc_verify) {
            int n_total = (int)(all_latents.size() / latent_dim);
            int n_emb = n_total / patch_size;
            std::vector<float> embeds((size_t)n_emb * llm_dim);
            dots_penc_forward(ctx, all_latents.data(), n_total, embeds.data());
            const float* ref = embeds.data() + (size_t)(n_emb - 1) * llm_dim;
            double dot = 0, na = 0, nb = 0, mx = 0;
            for (int i = 0; i < llm_dim; i++) {
                dot += (double)emb_step[i] * ref[i];
                na += (double)emb_step[i] * emb_step[i];
                nb += (double)ref[i] * ref[i];
                mx = std::max(mx, std::fabs((double)emb_step[i] - ref[i]));
            }
            double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
            std::fprintf(stderr, "dots_tts: [penc-verify] patch %d cos=%.6f max|Δ|=%.5f\n", patch_idx, cos, mx);
        }
        const float* llm_embedding = emb_step.data();

        // Feed the penc embedding into the LLM → next conditioning hidden.
        {
            dots_bench_stage b2("llm_step");
            dots_llm_step(ctx, llm_embedding, 1, llm_n_past, cur_hidden.data());
            llm_n_past += 1;
        }

        n_patches_done++;
        if (ctx->params.verbosity >= 1 && (patch_idx == 0 || (patch_idx + 1) % 5 == 0 || stop_after)) {
            std::fprintf(stderr, "dots_tts: patch %d done (eos_p=%.3f%s, fm_len=%d)\n", patch_idx, eos_p,
                         stop_after ? " STOP" : "", fm_seq_len);
            std::fflush(stderr);
        }

        if (stop_after)
            break;

        // Seed the next patch's hidden conditioning (next token is an audio span).
        if (patch_idx + 1 < max_patches)
            append_hidden(cur_hidden.data());
    }

    int n_total_frames = (int)(all_latents.size() / latent_dim);
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "dots_tts: generated %d patches → %d latent frames\n", n_patches_done, n_total_frames);
    if (n_total_frames == 0) {
        std::fprintf(stderr, "dots_tts: no latents generated\n");
        return nullptr;
    }

    // 4. Vocoder decode
    float* pcm = nullptr;
    {
        dots_bench_stage b("vocoder_decode");
        pcm = dots_vocoder_decode(ctx, all_latents.data(), n_total_frames, out_n_samples);
    }

    return pcm;
}

float* dots_tts_generate_latents(struct dots_tts_context* ctx, const char* text, int* out_n) {
    // TODO: implement latent-only generation for debugging
    (void)ctx;
    (void)text;
    *out_n = 0;
    return nullptr;
}

void dots_tts_pcm_free(float* pcm) {
    std::free(pcm);
}

void dots_tts_free(struct dots_tts_context* ctx) {
    if (!ctx)
        return;

    dots_kv_free(ctx->llm_kv);
    dots_kv_free(ctx->penc_kv);

    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);

    if (ctx->voc.buf_w)
        ggml_backend_buffer_free(ctx->voc.buf_w);
    if (ctx->voc.ctx_w)
        ggml_free(ctx->voc.ctx_w);

    if (ctx->spk_w.ctx)
        core_gguf::free_weights(ctx->spk_w);

    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);

    delete ctx;
}

void dots_tts_set_n_threads(struct dots_tts_context* ctx, int n_threads) {
    if (ctx && ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

void dots_tts_set_temperature(struct dots_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void dots_tts_set_seed(struct dots_tts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        ctx->rng.seed(seed > 0 ? seed : 42);
    }
}

// Diff-harness backend selector. The stages default to CPU (deterministic,
// reference-matching). Set CRISPASR_DOTS_DIFF_GPU=1 to run the GPU-resident
// stages (penc/llm/dit/flowmatch/vocoder) on init_best() instead, to confirm
// each stage matches the reference on the GPU backend — not just end-to-end.
// (CAM++ x-vector and the isolated Activation1d are CPU-by-design, so their
// diffs ignore this gate.)
static bool dots_diff_use_gpu() {
    const char* e = std::getenv("CRISPASR_DOTS_DIFF_GPU");
    return e && *e && *e != '0';
}

// ── Diff-harness: validate PatchEncoder decode_patch (patch 0) ──────────────
// Loads a (penc-only is fine) core GGUF + a reference GGUF carrying
// penc_in_patch0 / penc_out_patch0 (tools/dots_penc_reference.py), runs the
// C++ PatchEncoder, and prints cosine / max_abs vs the PyTorch reference.
extern "C" int dots_tts_penc_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    dots_tts_context_params p = dots_tts_context_default_params();
    p.verbosity = verbosity;
    p.use_gpu = dots_diff_use_gpu();
    dots_tts_context* ctx = dots_tts_init_from_file(model_gguf, p);
    if (!ctx) {
        std::fprintf(stderr, "dots_penc_diff: failed to load model %s\n", model_gguf);
        return 2;
    }

    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_penc_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    // Prefer the multi-patch reference (penc_in_all over a few patches +
    // penc_out_patch{0,1,...}) when present; fall back to single-patch.
    ggml_tensor* t_all = ggml_get_tensor(rw.ctx, "penc_in_all");
    ggml_tensor* t_in = t_all ? t_all : ggml_get_tensor(rw.ctx, "penc_in_patch0");
    if (!t_in) {
        std::fprintf(stderr, "dots_penc_diff: reference missing penc_in_all / penc_in_patch0\n");
        dots_tts_free(ctx);
        return 2;
    }

    const int in_dim = (int)t_in->ne[0];   // 128
    const int n_frames = (int)t_in->ne[1]; // patch_size * n_patches
    const int n_patches = n_frames / ctx->patch_size;

    std::vector<float> in_data((size_t)in_dim * n_frames);
    ggml_backend_tensor_get(t_in, in_data.data(), 0, in_data.size() * sizeof(float));

    // Run the full causal recompute -> n_patches embeddings.
    const int out_dim = (int)ctx->llm.hidden_size;
    std::vector<float> got((size_t)out_dim * n_patches, 0.0f);
    dots_penc_forward(ctx, in_data.data(), n_frames, got.data());

    int rc = 0;
    for (int pp = 0; pp < n_patches; pp++) {
        char name[32];
        std::snprintf(name, sizeof(name), "penc_out_patch%d", pp);
        ggml_tensor* t_out = ggml_get_tensor(rw.ctx, name);
        if (!t_out)
            continue;
        const int out_n = (int)ggml_nelements(t_out);
        std::vector<float> ref_out((size_t)out_n);
        ggml_backend_tensor_get(t_out, ref_out.data(), 0, ref_out.size() * sizeof(float));
        const float* g = got.data() + (size_t)pp * out_dim;
        double dot = 0, na = 0, nb = 0, maxabs = 0;
        for (int i = 0; i < out_n; i++) {
            dot += (double)g[i] * ref_out[i];
            na += (double)g[i] * g[i];
            nb += (double)ref_out[i] * ref_out[i];
            double d = std::fabs((double)g[i] - ref_out[i]);
            if (d > maxabs)
                maxabs = d;
        }
        double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        bool pass = cos > 0.999;
        std::printf("dots-tts penc decode_patch[%d]: in=(%d,%d) out_dim=%d  cos=%.6f  max_abs=%.6f  %s\n", pp, in_dim,
                    n_frames, out_n, cos, maxabs, pass ? "PASS" : "FAIL");
        if (verbosity >= 2) {
            std::printf("  got[:6]:");
            for (int i = 0; i < 6 && i < out_n; i++)
                std::printf(" %+.4f", g[i]);
            std::printf("\n  ref[:6]:");
            for (int i = 0; i < 6 && i < out_n; i++)
                std::printf(" %+.4f", ref_out[i]);
            std::printf("\n");
        }
        if (!pass)
            rc = 1;
    }

    core_gguf::free_weights(rw);
    dots_tts_free(ctx);
    return rc;
}

// ── Diff-harness: validate the LLM (Qwen2.5) forward ────────────────────────
// Reference (tools/reference_backends/dots_tts_reference.py _dump_llm) carries
// llm_ids_prefill (N), llm_hidden_prefill (hidden,N), llm_step_embed (hidden,1),
// llm_hidden_step (hidden,1). Validates BOTH the prefill (input_ids, n_past=0)
// and the KV-cached incremental step on an embedding (the penc->LLM feedback
// path). Prints per-stage cosine/max_abs; returns 0 if both PASS.
extern "C" int dots_tts_llm_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    dots_tts_context_params p = dots_tts_context_default_params();
    p.verbosity = verbosity;
    p.use_gpu = dots_diff_use_gpu();
    dots_tts_context* ctx = dots_tts_init_from_file(model_gguf, p);
    if (!ctx) {
        std::fprintf(stderr, "dots_llm_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_llm_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    ggml_tensor* t_ids = ggml_get_tensor(rw.ctx, "llm_ids_prefill");
    ggml_tensor* t_hp = ggml_get_tensor(rw.ctx, "llm_hidden_prefill");
    ggml_tensor* t_se = ggml_get_tensor(rw.ctx, "llm_step_embed");
    ggml_tensor* t_hs = ggml_get_tensor(rw.ctx, "llm_hidden_step");
    if (!t_ids || !t_hp || !t_se || !t_hs) {
        std::fprintf(stderr, "dots_llm_diff: reference missing llm_* tensors\n");
        dots_tts_free(ctx);
        return 2;
    }

    const int D = (int)ctx->llm.hidden_size;
    const int N = (int)ggml_nelements(t_ids);

    auto report = [&](const char* tag, const float* got, const float* ref, int n) -> bool {
        double dot = 0, na = 0, nb = 0, maxabs = 0;
        for (int i = 0; i < n; i++) {
            dot += (double)got[i] * ref[i];
            na += (double)got[i] * got[i];
            nb += (double)ref[i] * ref[i];
            double d = std::fabs((double)got[i] - ref[i]);
            if (d > maxabs)
                maxabs = d;
        }
        double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
        bool pass = cos > 0.999;
        std::printf("dots-tts llm %-16s n=%d  cos=%.6f  max_abs=%.6f  %s\n", tag, n, cos, maxabs,
                    pass ? "PASS" : "FAIL");
        if (verbosity >= 2) {
            std::printf("  got[:6]:");
            for (int i = 0; i < 6 && i < n; i++)
                std::printf(" %+.4f", got[i]);
            std::printf("\n  ref[:6]:");
            for (int i = 0; i < 6 && i < n; i++)
                std::printf(" %+.4f", ref[i]);
            std::printf("\n");
        }
        return pass;
    };

    // ── Prefill: embed the seeded ids, run dots_llm_step(n_past=0). ──
    std::vector<float> idf(N);
    ggml_backend_tensor_get(t_ids, idf.data(), 0, N * sizeof(float));
    std::vector<int32_t> ids(N);
    for (int i = 0; i < N; i++)
        ids[i] = (int32_t)std::lround(idf[i]);
    std::vector<float> emb((size_t)N * D);
    dots_embed_tokens(ctx, ids.data(), N, emb.data());
    std::vector<float> hp((size_t)N * D);
    dots_llm_step(ctx, emb.data(), N, 0, hp.data());

    std::vector<float> ref_hp((size_t)N * D);
    ggml_backend_tensor_get(t_hp, ref_hp.data(), 0, ref_hp.size() * sizeof(float));
    bool ok_pre = report("prefill[all]", hp.data(), ref_hp.data(), N * D);
    bool ok_prelast = report("prefill[last]", hp.data() + (size_t)(N - 1) * D, ref_hp.data() + (size_t)(N - 1) * D, D);

    // ── Incremental step on a seeded embedding (penc->LLM feedback path). ──
    std::vector<float> se(D);
    ggml_backend_tensor_get(t_se, se.data(), 0, D * sizeof(float));
    std::vector<float> hs(D);
    dots_llm_step(ctx, se.data(), 1, N, hs.data());
    std::vector<float> ref_hs(D);
    ggml_backend_tensor_get(t_hs, ref_hs.data(), 0, D * sizeof(float));
    bool ok_step = report("step[embed]", hs.data(), ref_hs.data(), D);

    core_gguf::free_weights(rw);
    dots_tts_free(ctx);
    return (ok_pre && ok_prelast && ok_step) ? 0 : 1;
}

// ── Diff-harness: validate the flow-matching ODE driver ─────────────────────
// Reference carries fm_input_seq (dit_dim,total_len), fm_cfg_seq, fm_mask
// (total_len,total_len additive [q][k]), fm_pos (total_len), fm_noise
// (latent_dim,patch_size), fm_latent_out (latent_dim,patch_size) and fm_meta
// [total_len, latent_start, num_steps, cfg_scale].
extern "C" int dots_tts_flowmatch_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    dots_tts_context_params p = dots_tts_context_default_params();
    p.verbosity = verbosity;
    p.use_gpu = dots_diff_use_gpu();
    dots_tts_context* ctx = dots_tts_init_from_file(model_gguf, p);
    if (!ctx) {
        std::fprintf(stderr, "dots_flowmatch_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_flowmatch_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    auto get = [&](const char* nm) { return ggml_get_tensor(rw.ctx, nm); };
    ggml_tensor* t_in = get("fm_input_seq");
    ggml_tensor* t_cfg = get("fm_cfg_seq");
    ggml_tensor* t_mask = get("fm_mask");
    ggml_tensor* t_pos = get("fm_pos");
    ggml_tensor* t_noise = get("fm_noise");
    ggml_tensor* t_out = get("fm_latent_out");
    ggml_tensor* t_meta = get("fm_meta");
    if (!t_in || !t_cfg || !t_mask || !t_pos || !t_noise || !t_out || !t_meta) {
        std::fprintf(stderr, "dots_flowmatch_diff: reference missing fm_* tensors\n");
        dots_tts_free(ctx);
        return 2;
    }

    float meta[4];
    ggml_backend_tensor_get(t_meta, meta, 0, sizeof(meta));
    const int total_len = (int)meta[0];
    const int latent_start = (int)meta[1];
    const int num_steps = (int)meta[2];
    const float cfg_scale = meta[3];
    const int dit_dim = (int)ctx->dit.hidden_size;
    const int latent_dim = ctx->latent_dim;
    const int patch_size = total_len - latent_start;
    const int out_n = patch_size * latent_dim;

    std::vector<float> in_seq((size_t)total_len * dit_dim), cfg_seq((size_t)total_len * dit_dim);
    std::vector<float> mask((size_t)total_len * total_len), noise((size_t)out_n), ref_out((size_t)out_n);
    std::vector<float> pos_f((size_t)total_len);
    ggml_backend_tensor_get(t_in, in_seq.data(), 0, in_seq.size() * sizeof(float));
    ggml_backend_tensor_get(t_cfg, cfg_seq.data(), 0, cfg_seq.size() * sizeof(float));
    ggml_backend_tensor_get(t_mask, mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_tensor_get(t_pos, pos_f.data(), 0, pos_f.size() * sizeof(float));
    ggml_backend_tensor_get(t_noise, noise.data(), 0, noise.size() * sizeof(float));
    ggml_backend_tensor_get(t_out, ref_out.data(), 0, ref_out.size() * sizeof(float));
    std::vector<int32_t> pos(total_len);
    for (int i = 0; i < total_len; i++)
        pos[i] = (int32_t)pos_f[i];

    std::vector<float> got((size_t)out_n, 0.0f);
    dots_flow_match_core(ctx, in_seq.data(), cfg_seq.data(), total_len, latent_start, mask.data(), pos.data(),
                         noise.data(), num_steps, cfg_scale, got.data());

    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (int i = 0; i < out_n; i++) {
        dot += (double)got[i] * ref_out[i];
        na += (double)got[i] * got[i];
        nb += (double)ref_out[i] * ref_out[i];
        double d = std::fabs((double)got[i] - ref_out[i]);
        if (d > maxabs)
            maxabs = d;
    }
    double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("dots-tts flow-match: total=%d latent_start=%d steps=%d cfg=%.1f out=%d  cos=%.6f  max_abs=%.6f  %s\n",
                total_len, latent_start, num_steps, cfg_scale, out_n, cos, maxabs, cos > 0.999 ? "PASS" : "FAIL");
    if (verbosity >= 2) {
        std::printf("  got[:6]:");
        for (int i = 0; i < 6 && i < out_n; i++)
            std::printf(" %+.4f", got[i]);
        std::printf("\n  ref[:6]:");
        for (int i = 0; i < 6 && i < out_n; i++)
            std::printf(" %+.4f", ref_out[i]);
        std::printf("\n");
    }

    core_gguf::free_weights(rw);
    dots_tts_free(ctx);
    return cos > 0.999 ? 0 : 1;
}

// ── Diff-harness: validate DiT (velocity_field_predictor) one forward ───────
// Reference carries dit_x (D,T), dit_t (scalar), dit_gcond (D), dit_vel
// (latent_dim,T) from tools/reference_backends/dots_tts_reference.py.
extern "C" int dots_tts_dit_diff(const char* model_gguf, const char* ref_gguf, int verbosity) {
    dots_tts_context_params p = dots_tts_context_default_params();
    p.verbosity = verbosity;
    p.use_gpu = dots_diff_use_gpu();
    dots_tts_context* ctx = dots_tts_init_from_file(model_gguf, p);
    if (!ctx) {
        std::fprintf(stderr, "dots_dit_diff: failed to load model %s\n", model_gguf);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_dit_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    ggml_tensor* t_x = ggml_get_tensor(rw.ctx, "dit_x");
    ggml_tensor* t_t = ggml_get_tensor(rw.ctx, "dit_t");
    ggml_tensor* t_g = ggml_get_tensor(rw.ctx, "dit_gcond");
    ggml_tensor* t_vel = ggml_get_tensor(rw.ctx, "dit_vel");
    if (!t_x || !t_t || !t_vel) {
        std::fprintf(stderr, "dots_dit_diff: reference missing dit_x/dit_t/dit_vel\n");
        dots_tts_free(ctx);
        return 2;
    }

    const int D = (int)t_x->ne[0]; // fm hidden (1024)
    const int T = (int)t_x->ne[1]; // sequence length
    const int out_n = (int)ggml_nelements(t_vel);

    std::vector<float> x_data((size_t)D * T);
    ggml_backend_tensor_get(t_x, x_data.data(), 0, x_data.size() * sizeof(float));
    float t_scalar = 0.0f;
    ggml_backend_tensor_get(t_t, &t_scalar, 0, sizeof(float));
    std::vector<float> g_data((size_t)D);
    bool have_g = (t_g != nullptr);
    if (have_g)
        ggml_backend_tensor_get(t_g, g_data.data(), 0, g_data.size() * sizeof(float));
    std::vector<float> ref_vel((size_t)out_n);
    ggml_backend_tensor_get(t_vel, ref_vel.data(), 0, ref_vel.size() * sizeof(float));

    std::vector<float> got((size_t)out_n, 0.0f);
    dots_dit_forward(ctx, x_data.data(), T, t_scalar, have_g ? g_data.data() : nullptr,
                     /*attn_mask=*/nullptr, /*pos_ids=*/nullptr, got.data());

    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (int i = 0; i < out_n; i++) {
        dot += (double)got[i] * ref_vel[i];
        na += (double)got[i] * got[i];
        nb += (double)ref_vel[i] * ref_vel[i];
        double d = std::fabs((double)got[i] - ref_vel[i]);
        if (d > maxabs)
            maxabs = d;
    }
    double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("dots-tts dit forward: D=%d T=%d t=%.3f out=%d  cos=%.6f  max_abs=%.6f  %s\n", D, T, t_scalar, out_n,
                cos, maxabs, cos > 0.999 ? "PASS" : "FAIL");
    if (verbosity >= 2) {
        std::printf("  got[:6]:");
        for (int i = 0; i < 6 && i < out_n; i++)
            std::printf(" %+.4f", got[i]);
        std::printf("\n  ref[:6]:");
        for (int i = 0; i < 6 && i < out_n; i++)
            std::printf(" %+.4f", ref_vel[i]);
        std::printf("\n");
    }

    core_gguf::free_weights(rw);
    dots_tts_free(ctx);
    return cos > 0.999 ? 0 : 1;
}

// ── Diff-harness: validate the BigVGAN vocoder front-end ────────────────────
// Reference (dots-tts-voc) carries voc_latent_in / voc_mi_out / voc_conv_pre /
// voc_audio. model_gguf is the VOCODER GGUF (not the core). Validates the
// non-anti-aliased front-end (post_proj → dec_mi LSTM → conv_pre) stage-by-
// stage; the alias-free resblock path / final audio is reported but not gated.
extern "C" int dots_tts_vocoder_diff(const char* voc_gguf, const char* ref_gguf, int verbosity) {
    auto* ctx = new dots_tts_context();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = dots_diff_use_gpu() ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;
    ctx->params = dots_tts_context_default_params();
    ctx->params.verbosity = verbosity;
    if (dots_tts_set_vocoder_path(ctx, voc_gguf) != 0) {
        std::fprintf(stderr, "dots_voc_diff: failed to load vocoder %s\n", voc_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    auto& voc = ctx->voc;

    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_voc_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    ggml_tensor* t_in = ggml_get_tensor(rw.ctx, "voc_latent_in");
    ggml_tensor* t_post = ggml_get_tensor(rw.ctx, "voc_post_proj");
    ggml_tensor* t_mi = ggml_get_tensor(rw.ctx, "voc_mi_out");
    ggml_tensor* t_cp = ggml_get_tensor(rw.ctx, "voc_conv_pre");
    ggml_tensor* t_audio = ggml_get_tensor(rw.ctx, "voc_audio");
    if (!t_in || !t_mi || !t_cp) {
        std::fprintf(stderr, "dots_voc_diff: reference missing voc_* tensors\n");
        dots_tts_free(ctx);
        return 2;
    }
    const int latent_dim = (int)t_in->ne[0];
    const int n_frames = (int)t_in->ne[1];
    std::vector<float> latent_in((size_t)latent_dim * n_frames);
    ggml_backend_tensor_get(t_in, latent_in.data(), 0, latent_in.size() * sizeof(float));

    auto cosmax = [](const std::vector<float>& a, const std::vector<float>& b, double& cos, double& maxabs) {
        double dot = 0, na = 0, nb = 0;
        maxabs = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            double d = std::fabs((double)a[i] - b[i]);
            if (d > maxabs)
                maxabs = d;
        }
        cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    };

    // ── Front-end graph: post_proj → dec_mi (LSTM) → conv_pre ──
    size_t n_tensors = 4096;
    ggml_init_params ip = {n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_tensors, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, latent_dim, n_frames);
    ggml_set_name(x, "voc_in");
    ggml_set_input(x);

    // post_proj (Conv1d k=1, non-causal — k=1 so padding is moot)
    ggml_tensor* h = dots_conv1d(ctx0, x, voc.post_proj_w, voc.post_proj_b, 1, /*causal=*/false);
    ggml_tensor* post_only = h;
    ggml_set_name(post_only, "post_only");
    ggml_set_output(post_only);
    // dec_mi_layer: Linear(128→512) → SLSTM[4-layer LSTM(512) + skip] → Linear(512→128)
    h = ggml_mul_mat(ctx0, voc.mi_in_w, h);
    if (voc.mi_in_b)
        h = ggml_add(ctx0, h, voc.mi_in_b);
    ggml_tensor* lstm_in = h; // SLSTM skip residual is the LSTM *input*
    for (int li = 0; li < 4; li++) {
        if (!voc.lstm_w_ih[li])
            break;
        h = core_lstm::lstm_unidir(ctx0, gf, h, voc.lstm_w_ih[li], voc.lstm_w_hh[li], voc.lstm_b_ih[li],
                                   voc.lstm_b_hh[li], 512, /*reverse=*/false);
    }
    h = ggml_add(ctx0, h, lstm_in); // SLSTM skip: y = LSTM(x) + x
    h = ggml_mul_mat(ctx0, voc.mi_out_w, h);
    if (voc.mi_out_b)
        h = ggml_add(ctx0, h, voc.mi_out_b);
    ggml_tensor* mi_out = h;
    ggml_set_name(mi_out, "mi_out");
    ggml_set_output(mi_out);
    // conv_pre (non-causal, symmetric pad 2)
    ggml_tensor* cp = dots_conv1d(ctx0, h, voc.conv_pre_w, voc.conv_pre_b, 1, /*causal=*/false);
    ggml_set_name(cp, "conv_pre");
    ggml_set_output(cp);
    // Bisection: stage-0 upsample + resblocks.
    const int dilations[3] = {1, 3, 5};
    ggml_tensor* u0 = dots_convt1d(ctx0, cp, voc.ups_w[0], voc.ups_b[0], (int)voc.upsample_rates[0]);
    ggml_set_name(u0, "ups0");
    ggml_set_output(u0);
    ggml_tensor* xs = nullptr;
    for (int j = 0; j < 3; j++) {
        ggml_tensor* rj = dots_resblock_fwd(ctx0, u0, voc.resblocks[j], dilations);
        xs = xs ? ggml_add(ctx0, xs, rj) : rj;
    }
    ggml_tensor* st0 = ggml_scale(ctx0, xs, 1.0f / 3.0f);
    ggml_set_name(st0, "stage0");
    ggml_set_output(st0);
    ggml_build_forward_expand(gf, cp);
    ggml_build_forward_expand(gf, st0);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_tensor_set(x, latent_in.data(), 0, latent_in.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    int rc = 0;
    if (t_post) {
        ggml_tensor* o = ggml_graph_get_tensor(gf, "post_only");
        std::vector<float> got((size_t)ggml_nelements(o));
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
        std::vector<float> ref((size_t)ggml_nelements(t_post));
        ggml_backend_tensor_get(t_post, ref.data(), 0, ref.size() * sizeof(float));
        double cos, mx;
        cosmax(got, ref, cos, mx);
        std::printf("dots-tts voc post_proj:    n=%zu  cos=%.6f  max_abs=%.6f  %s\n", got.size(), cos, mx,
                    cos > 0.999 ? "PASS" : "FAIL");
    }
    {
        ggml_tensor* o = ggml_graph_get_tensor(gf, "mi_out");
        std::vector<float> got((size_t)ggml_nelements(o));
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
        std::vector<float> ref((size_t)ggml_nelements(t_mi));
        ggml_backend_tensor_get(t_mi, ref.data(), 0, ref.size() * sizeof(float));
        double cos, mx;
        cosmax(got, ref, cos, mx);
        std::printf("dots-tts voc post_proj+mi: n=%zu  cos=%.6f  max_abs=%.6f  %s\n", got.size(), cos, mx,
                    cos > 0.999 ? "PASS" : "FAIL");
        if (cos <= 0.999)
            rc = 1;
    }
    {
        ggml_tensor* o = ggml_graph_get_tensor(gf, "conv_pre");
        std::vector<float> got((size_t)ggml_nelements(o));
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
        std::vector<float> ref((size_t)ggml_nelements(t_cp));
        ggml_backend_tensor_get(t_cp, ref.data(), 0, ref.size() * sizeof(float));
        double cos, mx;
        cosmax(got, ref, cos, mx);
        std::printf("dots-tts voc conv_pre:     n=%zu  cos=%.6f  max_abs=%.6f  %s\n", got.size(), cos, mx,
                    cos > 0.999 ? "PASS" : "FAIL");
        if (cos <= 0.999)
            rc = 1;
    }
    for (const char* nm : {"ups0", "stage0"}) {
        ggml_tensor* o = ggml_graph_get_tensor(gf, nm);
        ggml_tensor* tr = ggml_get_tensor(rw.ctx, std::string("voc_").append(nm).c_str());
        if (!o || !tr)
            continue;
        std::vector<float> got((size_t)ggml_nelements(o)), ref((size_t)ggml_nelements(tr));
        ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
        ggml_backend_tensor_get(tr, ref.data(), 0, ref.size() * sizeof(float));
        double cos, mx;
        cosmax(got, ref, cos, mx);
        std::printf("dots-tts voc %-9s n=%zu (ref=%zu)  cos=%.6f  max_abs=%.6f  %s\n", nm, got.size(), ref.size(), cos,
                    mx, cos > 0.999 ? "PASS" : "FAIL");
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);

    // ── Full decode (audio) — reported only; alias-free path still WIP ──
    if (t_audio) {
        int n_out = 0;
        float* pcm = dots_vocoder_decode(ctx, latent_in.data(), n_frames, &n_out);
        if (pcm) {
            std::vector<float> got(pcm, pcm + n_out);
            std::vector<float> ref((size_t)ggml_nelements(t_audio));
            ggml_backend_tensor_get(t_audio, ref.data(), 0, ref.size() * sizeof(float));
            double cos, mx;
            cosmax(got, ref, cos, mx);
            std::printf("dots-tts voc audio[full]:  got=%d ref=%zu  cos=%.6f  max_abs=%.6f  %s\n", n_out, ref.size(),
                        cos, mx, cos > 0.99 ? "PASS" : "(WIP alias-free)");
            std::free(pcm);
        }
    }

    core_gguf::free_weights(rw);
    dots_tts_free(ctx);
    return rc;
}

// ── Diff-harness: validate the CAM++ speaker path (Phase F) ─────────────────
// `spk_gguf` is the dots-tts-soar-spk encoder; the reference (dots-tts-spk)
// carries spk_pcm_16k (input), spk_xvector (512), spk_g_cond (1024), and the
// four xvec_proj weights (xvec_proj_0_w/0_b, xvec_proj_1_w/1_b) so the diff
// can run the full wav→x-vector→g_cond pipeline without the 4.6 GB core GGUF.
extern "C" int dots_tts_spk_diff(const char* spk_gguf, const char* ref_gguf, int verbosity) {
    auto* ctx = new dots_tts_context();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = ctx->backend_cpu;
    ctx->params = dots_tts_context_default_params();
    ctx->params.verbosity = verbosity;
    ctx->spk_dim = 512;

    if (dots_tts_set_speaker_path(ctx, spk_gguf) != 0) {
        std::fprintf(stderr, "dots_spk_diff: failed to load speaker encoder %s\n", spk_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_spk_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    auto Tref = [&](const char* n) { return ggml_get_tensor(rw.ctx, n); };
    ggml_tensor* t_pcm = Tref("spk_pcm_16k");
    ggml_tensor* t_xv = Tref("spk_xvector");
    ggml_tensor* t_gc = Tref("spk_g_cond");
    // Bind the xvec_proj weights from the reference into the projection slots.
    ctx->proj.xvec_proj_0_w = Tref("xvec_proj_0_w");
    ctx->proj.xvec_proj_0_b = Tref("xvec_proj_0_b");
    ctx->proj.xvec_proj_1_w = Tref("xvec_proj_1_w");
    ctx->proj.xvec_proj_1_b = Tref("xvec_proj_1_b");
    if (!t_pcm || !t_xv || !t_gc || !ctx->proj.xvec_proj_0_w || !ctx->proj.xvec_proj_1_w) {
        std::fprintf(stderr, "dots_spk_diff: reference missing spk_* / xvec_proj_* tensors\n");
        dots_tts_free(ctx);
        return 2;
    }
    const int n_pcm = (int)ggml_nelements(t_pcm);
    std::vector<float> pcm((size_t)n_pcm);
    ggml_backend_tensor_get(t_pcm, pcm.data(), 0, pcm.size() * sizeof(float));

    if (dots_tts_set_speaker_pcm(ctx, pcm.data(), n_pcm) != 0) {
        std::fprintf(stderr, "dots_spk_diff: g_cond computation failed\n");
        dots_tts_free(ctx);
        return 2;
    }

    auto cosmax = [](const std::vector<float>& a, const std::vector<float>& b, double& cos, double& maxabs) {
        double dot = 0, na = 0, nb = 0;
        maxabs = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            double d = std::fabs((double)a[i] - b[i]);
            if (d > maxabs)
                maxabs = d;
        }
        cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    };
    auto ref_vec = [&](ggml_tensor* t) {
        std::vector<float> v((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        return v;
    };

    std::vector<float> xv_ref = ref_vec(t_xv), gc_ref = ref_vec(t_gc);
    double cx, mx, cg, mg;
    cosmax(ctx->speaker_emb, xv_ref, cx, mx);
    cosmax(ctx->g_cond, gc_ref, cg, mg);
    std::printf("dots-tts-spk: x-vector[%zu]  cos=%.6f  max|Δ|=%.5f\n", ctx->speaker_emb.size(), cx, mx);
    std::printf("dots-tts-spk: g_cond  [%zu]  cos=%.6f  max|Δ|=%.5f\n", ctx->g_cond.size(), cg, mg);

    core_gguf::free_weights(rw);
    dots_tts_free(ctx);
    const bool pass = (cx > 0.999 && cg > 0.999);
    std::printf("dots-tts-spk: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ── Diff-harness: validate the isolated alias-free Activation1d ──────────────
// Reference (dots-tts-act) carries act_in / act_out (T,C) + act_alpha/act_beta
// (C,) + act_up_filter/act_down_filter (K,). Validates dots_alias_free_act.
extern "C" int dots_tts_act_diff(const char* ref_gguf, int verbosity) {
    auto* ctx = new dots_tts_context();
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = ctx->backend_cpu;
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "ref", rw)) {
        std::fprintf(stderr, "dots_act_diff: failed to load reference %s\n", ref_gguf);
        dots_tts_free(ctx);
        return 2;
    }
    auto G = [&](const char* n) { return ggml_get_tensor(rw.ctx, n); };
    ggml_tensor* t_in = G("act_in");
    ggml_tensor* t_out = G("act_out");
    ggml_tensor* alpha = G("act_alpha");
    ggml_tensor* beta = G("act_beta");
    ggml_tensor* upf = G("act_up_filter");
    ggml_tensor* dnf = G("act_down_filter");
    if (!t_in || !t_out || !alpha || !beta || !upf || !dnf) {
        std::fprintf(stderr, "dots_act_diff: reference missing act_* tensors\n");
        dots_tts_free(ctx);
        return 2;
    }
    const int C = (int)t_in->ne[0];
    const int T = (int)t_in->ne[1];

    size_t n_tensors = 512;
    ggml_init_params ip = {n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
    ggml_set_name(x, "act_x");
    ggml_set_input(x);
    ggml_tensor* y = dots_alias_free_act(ctx0, x, alpha, beta, upf, dnf);
    ggml_set_name(y, "act_y");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    ggml_gallocr_alloc_graph(ga, gf);
    std::vector<float> in_data((size_t)C * T);
    ggml_backend_tensor_get(t_in, in_data.data(), 0, in_data.size() * sizeof(float));
    ggml_backend_tensor_set(x, in_data.data(), 0, in_data.size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    ggml_tensor* o = ggml_graph_get_tensor(gf, "act_y");
    int n = (int)ggml_nelements(o);
    std::vector<float> got((size_t)n), ref((size_t)ggml_nelements(t_out));
    ggml_backend_tensor_get(o, got.data(), 0, got.size() * sizeof(float));
    ggml_backend_tensor_get(t_out, ref.data(), 0, ref.size() * sizeof(float));
    double dot = 0, na = 0, nb = 0, mx = 0;
    int m = std::min((int)got.size(), (int)ref.size());
    for (int i = 0; i < m; i++) {
        dot += (double)got[i] * ref[i];
        na += (double)got[i] * got[i];
        nb += (double)ref[i] * ref[i];
        double d = std::fabs((double)got[i] - ref[i]);
        if (d > mx)
            mx = d;
    }
    double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    std::printf("dots-tts alias-free act: C=%d T=%d out_n=%d (ref=%d)  cos=%.6f  max_abs=%.6f  %s\n", C, T, n,
                (int)ref.size(), cos, mx, cos > 0.999 ? "PASS" : "FAIL");
    if (verbosity >= 2) {
        std::printf("  got[:6]:");
        for (int i = 0; i < 6 && i < n; i++)
            std::printf(" %+.4f", got[i]);
        std::printf("\n  ref[:6]:");
        for (int i = 0; i < 6 && i < (int)ref.size(); i++)
            std::printf(" %+.4f", ref[i]);
        std::printf("\n");
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx0);
    dots_tts_free(ctx);
    return cos > 0.999 ? 0 : 1;
}
