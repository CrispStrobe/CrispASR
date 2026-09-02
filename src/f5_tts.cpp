// f5_tts.cpp — native ggml runtime for SWivid/F5-TTS.
//
// Architecture (all F32 on CPU, F16 for large weight storage):
//   1. TextEmbedding: char embedding(2546, 512) + sinusoidal pos +
//      4 ConvNeXtV2 blocks (512-d, intermediate 1024)
//   2. InputEmbedding: Linear(712, 1024) concat(x, cond, text) +
//      ConvPositionEmbedding (2× Conv1d k=31 groups=16 + Mish)
//   3. TimestepEmbedding: sinusoidal(256) → Linear(256,1024) → SiLU → Linear(1024,1024)
//   4. DiT: 22 blocks, each:
//      - AdaLN-Zero: SiLU(t) → Linear(1024, 6144) → split 6× (shift/scale/gate for attn+mlp)
//      - Self-attention: Q/K/V proj → RoPE → scaled_dot_product → O proj
//        (16 heads, dim_head=64, bidirectional, no mask for batch=1)
//      - Gated residual: x = x + gate_msa * attn_out
//      - Modulated LayerNorm: ff_norm(x) * (1 + scale_mlp) + shift_mlp
//      - FFN: Linear(1024, 2048) → GELU_tanh → Linear(2048, 1024)
//      - Gated residual: x = x + gate_mlp * ff_out
//   5. AdaLN-Final + Linear(1024, 100) → velocity prediction
//   6. Euler ODE solver: 32 steps with EPSS timesteps + sway + CFG
//   7. Vocos vocoder: Conv1d(100,512,k7) → LN → 8× ConvNeXt(512,1536) →
//      LN → Linear(512, 1026) → split mag+phase → iSTFT → 24 kHz PCM
//
// The implementation uses per-module ggml sub-graphs. Each module builds
// its own graph, computes, and returns CPU-side results.

#include "f5_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/utf8.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)
#include "core/crispasr_env.h"
#include "core/pinyin_g2p.h" // #294: Chinese g2p (jieba-min + pypinyin TONE3)

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
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
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `F5_BENCH=1` for per-stage timings.
// ===========================================================================

static bool f5_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_F5_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Opt-in: batch the CFG cond+uncond DiT passes into one B=2 graph.
static bool f5_batch_cfg_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_F5_BATCH_CFG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Opt-in (#294): feed the DiT matmuls F16 activations instead of F32, by casting
// only the mat-mul RHS (norm/attn/FFN activations); norms/adaln/rope/flash and the
// output stay F32. Intent: the F16 weights make the F32 activation the only wide
// operand, so an F16 RHS halves its bandwidth on the 64×/synth forwards.
//   MEASURED (M1 Metal, F16, jfk ref): output is BYTE-IDENTICAL to the F32-activation
//   path (md5 unchanged) AND ~17% slower (dit_graph 53.3 s vs 45.3 s). The
//   byte-identity proves ggml's F16-weight mul_mat already converts the F32 RHS to
//   F16 internally on Metal — so the explicit cast is redundant work, not a new win.
//   Kept gated (default OFF) ONLY so a CUDA user can A/B it, since a cuBLAS/CUDA
//   kernel may keep F32 activations where this would help. Do NOT default this on
//   without a trustworthy CUDA measurement. (The old "F32->F16 activations is the
//   real lever" note below is wrong for the Metal backend.)
static bool f5_f16_act_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_F5_F16_ACT");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Opt-in (#294): compute the InputEmbedding (input_proj + 2× grouped conv-pos +
// Mish + residual) on the GPU backend instead of host BLAS/scalar. On the default
// (32 steps) that host stage is ~26% of the ODE loop on M1 and a larger share on a
// fast-GPU/slow-CPU box (the #294 reporter's RTX 5060 Ti + Ryzen 2600). Moving it
// to ctx->backend frees the CPU stall between GPU DiT dispatches. Reuses the proven
// grouped-conv-pos graph pattern from cosyvoice3_tts (identical
// Conv1d(dim,dim,k=31,groups=16)+Mish module), adapted to F5's SYMMETRIC pad=15.
// Default OFF pending a CUDA A/B (on M1 the GPU is already the bottleneck, so this
// is expected to regress locally while winning on the reporter's hardware).
static bool f5_embed_gpu_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_F5_EMBED_GPU");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct f5_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit f5_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~f5_bench_stage() {
        if (!f5_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  f5_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ── Hyperparameters ──────────────────────────────────────────────

// Text length for the duration estimate: CODEPOINTS by default, bytes under
// CRISPASR_F5_TEXT_LEN_BYTES=1.
//
// Upstream F5-TTS uses `len(text.encode("utf-8"))` — bytes — on both sides of
// the rate, and so does our reference dumper
// (tools/reference_backends/f5_tts.py:259). Byte/byte cancels when reference
// and target share a script, so upstream only misbehaves when they differ; our
// no-ref-text branch was worse still, pairing a 13 chars/sec constant with a
// byte count, so Devanagari or CJK inflated ~3x and the ODE solve ran for
// minutes (#372).
//
// Codepoints are the better default, so that is the default. The gate exists so
// the exact upstream arithmetic stays reachable: `crispasr-diff f5-tts` compares
// against a reference that is deliberately STILL on bytes, because a reference
// that adopted our change could no longer disagree with us. Same reasoning and
// same shape as CRISPASR_F5_DURATION_CLAMP=0 further down.
static size_t f5_text_len(const char* s) {
    static const bool s_bytes = []() {
        const char* e = crispasr_env::get("CRISPASR_F5_TEXT_LEN_BYTES");
        return e && e[0] == '1';
    }();
    if (!s)
        return 0;
    return s_bytes ? std::strlen(s) : core_utf8::length(s);
}

struct f5_hparams {
    int dim = 1024;
    int depth = 22;
    int heads = 16;
    int dim_head = 64;
    int ff_mult = 2;
    int text_dim = 512;
    int text_num_embeds = 2546;
    int conv_layers = 4;
    int mel_dim = 100;
    int sample_rate = 24000;
    int hop_length = 256;
    int win_length = 1024;
    int n_fft = 1024;
    int ode_steps = 32;
    float cfg_strength = 2.0f;
    float sway_coef = -1.0f;
    int conv_pos_kernel = 31;
    int conv_pos_groups = 16;
    // Vocos
    int voc_dim = 512;
    int voc_intermediate_dim = 1536;
    int voc_num_layers = 8;
    int voc_n_fft = 1024;
    int voc_hop_length = 256;
    // #387 Raon-OpenTTS deltas (default = stock F5/Vocos):
    //   vocoder       "vocos" | "hifigan"  — which decoder to run
    //   mel_spec_type "vocos" | "sbhifigan16k" — mel front-end variant
    //   mel_center    STFT center padding (Vocos true, sbhifigan false)
    // sbhifigan16k also ships its (slaney) filterbank + window as GGUF
    // buffers, so the runtime never rebuilds the slaney basis in C++.
    std::string vocoder = "vocos";
    std::string mel_spec_type = "vocos";
    bool mel_center = true;
};

// ── Weight structure ─────────────────────────────────────────────

struct f5_weights {
    // Text embedding
    ggml_tensor* text_emb_weight = nullptr; // (2546, 512)

    // Text ConvNeXtV2 blocks (4×)
    struct text_block {
        ggml_tensor* dw_weight; // (512, 1, 7) depthwise conv
        ggml_tensor* dw_bias;
        ggml_tensor* norm_weight;
        ggml_tensor* norm_bias;
        ggml_tensor* pw_up_weight; // (1024, 512)
        ggml_tensor* pw_up_bias;
        ggml_tensor* pw_down_weight; // (512, 1024)
        ggml_tensor* pw_down_bias;
        ggml_tensor* grn_gamma; // (1, 1, 1024)
        ggml_tensor* grn_beta;
    };
    std::vector<text_block> text_blocks;

    // Timestep embedding
    ggml_tensor* time_mlp_0_weight; // (1024, 256)
    ggml_tensor* time_mlp_0_bias;
    ggml_tensor* time_mlp_1_weight; // (1024, 1024)
    ggml_tensor* time_mlp_1_bias;

    // Input embedding
    ggml_tensor* input_proj_weight; // (1024, 712)
    ggml_tensor* input_proj_bias;
    ggml_tensor* conv_pos_0_weight; // (1024, 64, 31)
    ggml_tensor* conv_pos_0_bias;
    ggml_tensor* conv_pos_1_weight;
    ggml_tensor* conv_pos_1_bias;

    // DiT blocks (22×)
    struct dit_block {
        // AdaLN
        ggml_tensor* adaln_weight; // (6144, 1024)
        ggml_tensor* adaln_bias;
        // Self-attention
        ggml_tensor* attn_q_weight;
        ggml_tensor* attn_q_bias;
        ggml_tensor* attn_k_weight;
        ggml_tensor* attn_k_bias;
        ggml_tensor* attn_v_weight;
        ggml_tensor* attn_v_bias;
        ggml_tensor* attn_o_weight;
        ggml_tensor* attn_o_bias;
        // FFN
        ggml_tensor* ffn_up_weight; // (2048, 1024)
        ggml_tensor* ffn_up_bias;
        ggml_tensor* ffn_down_weight; // (1024, 2048)
        ggml_tensor* ffn_down_bias;
    };
    std::vector<dit_block> dit_blocks;

    // Final norm + proj
    ggml_tensor* final_adaln_weight; // (2048, 1024)
    ggml_tensor* final_adaln_bias;
    ggml_tensor* final_proj_weight; // (100, 1024)
    ggml_tensor* final_proj_bias;

    // Rotary
    ggml_tensor* rotary_inv_freq; // (32,)

    // Vocos
    ggml_tensor* voc_embed_weight; // (512, 100, 7)
    ggml_tensor* voc_embed_bias;
    ggml_tensor* voc_norm_weight;
    ggml_tensor* voc_norm_bias;
    struct vocos_block {
        ggml_tensor* dw_weight;
        ggml_tensor* dw_bias;
        ggml_tensor* norm_weight;
        ggml_tensor* norm_bias;
        ggml_tensor* pw_up_weight;
        ggml_tensor* pw_up_bias;
        ggml_tensor* pw_down_weight;
        ggml_tensor* pw_down_bias;
        ggml_tensor* layer_scale; // (512,)
    };
    std::vector<vocos_block> voc_blocks;
    ggml_tensor* voc_final_norm_weight;
    ggml_tensor* voc_final_norm_bias;
    ggml_tensor* voc_head_weight; // (1026, 512)
    ggml_tensor* voc_head_bias;
};

// ── Vocab ────────────────────────────────────────────────────────

struct f5_vocab {
    std::vector<std::string> chars;
    std::map<std::string, int> char_to_idx;
};

// ── Text and Vocos weight caches ─────────────────────────────────
// Pre-dequantized F32 copies loaded once at init; avoids per-synthesis
// read_tensor_f32 (and Q4_K dequantization) in compute_text_embed and
// vocos_decode.

struct f5_text_block_cache {
    std::vector<float> dw_w, dw_b;
    std::vector<float> norm_w, norm_b;
    std::vector<float> pw_up_w, pw_up_b;
    std::vector<float> pw_down_w, pw_down_b;
    std::vector<float> grn_g, grn_b;
};

struct f5_voc_block_cache {
    std::vector<float> dw_w, dw_b;
    std::vector<float> norm_w, norm_b;
    std::vector<float> pw_up_w, pw_up_b;
    std::vector<float> pw_down_w, pw_down_b;
    std::vector<float> layer_scale;
};

// ── DiT graph cache ───────────────────────────────────────────────
// All 22 DiT blocks + final AdaLN/proj fused into one ggml graph,
// built once per T and reused across ODE steps (and cond/uncond passes).

struct f5_dit_graph_cache {
    int T_cached = -1;
    int B_cached = -1; // batch (1 = plain, 2 = batched CFG)
    ggml_context* gctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor* hidden_in = nullptr; // (dim, T, B) — input hidden state(s)
    ggml_tensor* t_emb_in = nullptr;  // (dim,)  — timestep embedding (shared over B)
    ggml_tensor* pos_in = nullptr;    // (T,) i32 — constant [0..T-1] (shared over B)
    ggml_tensor* output = nullptr;    // (mel_dim, T, B) — velocity/velocities

    void reset() {
        if (galloc) {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
        if (gctx) {
            ggml_free(gctx);
            gctx = nullptr;
        }
        gf = nullptr;
        hidden_in = t_emb_in = pos_in = output = nullptr;
        T_cached = -1;
        B_cached = -1;
    }
    ~f5_dit_graph_cache() { reset(); }
};

// #294: GPU InputEmbedding graph (opt-in CRISPASR_F5_EMBED_GPU). Built once per
// (T,B), reused across forwards. Input = the concatenated [x|cond'|text] (cat_dim,
// T, B); output = hidden (dim, T, B). input_proj + conv-pos weights already live on
// ctx->backend, so this keeps the whole embed on-device.
struct f5_embed_graph_cache {
    int T_cached = -1;
    int B_cached = -1;
    ggml_context* gctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor* cat_in = nullptr;     // (cat_dim, T, B) — concatenated input
    ggml_tensor* hidden_out = nullptr; // (dim, T, B) — embedded hidden
    void reset() {
        if (galloc) {
            ggml_gallocr_free(galloc);
            galloc = nullptr;
        }
        if (gctx) {
            ggml_free(gctx);
            gctx = nullptr;
        }
        gf = nullptr;
        cat_in = hidden_out = nullptr;
        T_cached = B_cached = -1;
    }
    ~f5_embed_graph_cache() { reset(); }
};

// ── Context ──────────────────────────────────────────────────────

struct f5_tts_context {
    f5_hparams hp;
    f5_weights w;
    f5_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;

    // Runtime params
    int seed;
    int ode_steps;
    float cfg_strength;
    float sway_coef;
    float speed;
    int verbosity;
    int n_threads;
    std::string dump_dir;

    // Reference audio state
    std::vector<float> ref_mel; // (T_ref, mel_dim) row-major
    int ref_mel_T = 0;

    // #387 sbhifigan16k: shipped slaney mel filterbank (n_freqs*n_mels,
    // layout fb[k*n_mels+m]) + Hann window (n_fft), copied verbatim from the
    // GGUF; and the HiFi-GAN vocoder weights as CPU F32 (weight-norm already
    // fused in the converter), keyed by their `voc.*` GGUF name.
    std::vector<float> mel_fb;
    std::vector<float> mel_window;
    std::map<std::string, std::vector<float>> hifigan_w;
    std::string ref_text;

    // Diff harness: inject reference initial noise for reproducibility
    std::vector<float> ref_init_noise;

    // Cached fused DiT graph (rebuilt only when T changes)
    f5_dit_graph_cache dit_cache;
    f5_embed_graph_cache embed_cache; // #294 GPU InputEmbedding (opt-in)

    // Pre-dequantized F32 input-embedding weights.
    // Avoids 64× read_tensor_f32 per synthesis (conv_pos tensors are 7.7 MB each).
    struct {
        std::vector<float> input_proj_w; // (dim, cat_dim)
        std::vector<float> input_proj_b; // (dim,)
        std::vector<float> conv_pos_0_w; // (dim, dim/groups, K)
        std::vector<float> conv_pos_0_b; // (dim,)
        std::vector<float> conv_pos_1_w;
        std::vector<float> conv_pos_1_b;
    } emb_cache;

    // Pre-dequantized text ConvNeXtV2 weights (4 blocks, ~21 MB F32).
    struct {
        std::vector<float> emb; // (text_num_embeds, text_dim)
        std::vector<f5_text_block_cache> blocks;
    } text_cache;

    // Pre-dequantized Vocos vocoder weights (8 blocks, ~54 MB F32).
    struct {
        std::vector<float> embed_w, embed_b;
        std::vector<float> norm_w, norm_b;
        std::vector<f5_voc_block_cache> blocks;
        std::vector<float> final_norm_w, final_norm_b;
        std::vector<float> head_w, head_b;
    } voc_cache;
};

// ── Diff dump helpers ────────────────────────────────────────────

static void dump_stage(const f5_tts_context* ctx, const char* label, const float* data, size_t n) {
    if (ctx->dump_dir.empty())
        return;
    std::string path = ctx->dump_dir + "/" + label + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
}

// ── Mini graph helper ────────────────────────────────────────────

struct f5_mini_graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;

    f5_mini_graph(ggml_backend_sched_t s, size_t ctx_size = 32 * 1024 * 1024) : sched(s) {
        struct ggml_init_params params = {ctx_size, nullptr, true};
        ctx = ggml_init(params);
    }
    ~f5_mini_graph() {
        if (ctx)
            ggml_free(ctx);
    }

    std::vector<float> compute(ggml_tensor* output, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, output);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "f5_tts: graph alloc failed\n");
            return {};
        }
        ggml_backend_sched_graph_compute(sched, gf);
        int n = (int)ggml_nelements(output);
        std::vector<float> result(n);
        ggml_backend_tensor_get(output, result.data(), 0, n * sizeof(float));
        return result;
    }

    bool compute_into(ggml_tensor* output, float* dst, size_t nbytes, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        ggml_build_forward_expand(gf, output);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "f5_tts: graph alloc failed\n");
            return false;
        }
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_tensor_get(output, dst, 0, nbytes);
        return true;
    }

    // Compute graph returning multiple outputs
    bool compute_multi(std::vector<ggml_tensor*>& outputs, int /*n_threads*/) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 32768, false);
        for (auto* o : outputs) {
            ggml_build_forward_expand(gf, o);
        }
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "f5_tts: graph alloc failed\n");
            return false;
        }
        ggml_backend_sched_graph_compute(sched, gf);
        return true;
    }

    void set_input(ggml_tensor* t, const void* data, size_t nbytes) { ggml_backend_tensor_set(t, data, 0, nbytes); }
};

// ── Tensor read helper ───────────────────────────────────────────

static void read_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    const size_t nbytes = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    } else {
        // F16 or quantized (Q4_K, Q8_0, etc.) — dequantize via type traits.
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto to_float = ggml_get_type_traits(t->type)->to_float;
        if (to_float) {
            to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "f5_tts: unsupported tensor type %d for '%s'\n", (int)t->type, t->name);
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
}

// ── sinusoidal position embedding ────────────────────────────────

static void sinusoidal_pos_emb(int dim, int max_len, float* out) {
    // out: (max_len, dim) row-major
    // Matches precompute_freqs_cis: freqs_cos ++ freqs_sin
    int half_dim = dim / 2;
    float theta = 10000.0f;
    for (int t = 0; t < max_len; t++) {
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = (float)t * freq;
            out[t * dim + d] = cosf(angle);            // cos part
            out[t * dim + d + half_dim] = sinf(angle); // sin part
        }
    }
}

// ── Timestep embedding (sinusoidal + MLP) ────────────────────────

static std::vector<float> compute_time_embed(f5_tts_context* ctx, float t_val) {
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;

    // Sinusoidal position embedding for timestep
    int freq_dim = 256;
    int half = freq_dim / 2;
    std::vector<float> sinus(freq_dim);
    float scale = 1000.0f;
    for (int d = 0; d < half; d++) {
        float emb_val = logf(10000.0f) / (float)(half - 1);
        float freq = expf(-(float)d * emb_val);
        float angle = scale * t_val * freq;
        sinus[d] = sinf(angle);
        sinus[d + half] = cosf(angle);
    }

    // MLP: Linear(256→1024) → SiLU → Linear(1024→1024)
    f5_mini_graph mg(ctx->sched);
    ggml_tensor* inp = ggml_new_tensor_1d(mg.ctx, GGML_TYPE_F32, freq_dim);
    ggml_set_name(inp, "time_sinus");
    ggml_set_input(inp);

    // Layer 0: Linear + SiLU
    ggml_tensor* h = ggml_mul_mat(mg.ctx, w.time_mlp_0_weight, inp);
    h = ggml_add(mg.ctx, h, w.time_mlp_0_bias);
    h = ggml_silu(mg.ctx, h);

    // Layer 1: Linear
    h = ggml_mul_mat(mg.ctx, w.time_mlp_1_weight, h);
    h = ggml_add(mg.ctx, h, w.time_mlp_1_bias);
    ggml_set_name(h, "time_emb");
    ggml_set_output(h);

    ggml_cgraph* gf = ggml_new_graph_custom(mg.ctx, 256, false);
    ggml_build_forward_expand(gf, h);
    ggml_backend_sched_reset(mg.sched);
    if (!ggml_backend_sched_alloc_graph(mg.sched, gf))
        return {};
    mg.set_input(inp, sinus.data(), freq_dim * sizeof(float));
    ggml_backend_sched_graph_compute(mg.sched, gf);

    std::vector<float> result(hp.dim);
    ggml_backend_tensor_get(h, result.data(), 0, hp.dim * sizeof(float));
    return result;
}


// ── pinyin conversion ────────────────────────────────────────────
// #294: for text containing Han characters, run the real g2p (jieba-min +
// pypinyin TONE3 + tone-sandhi) so Chinese tokenizes to the pinyin-syllable
// tokens the model was trained on (previously every Han char hit the unknown
// token → no Chinese audio). Pure-ASCII/Latin text keeps the byte-identical
// per-character passthrough below, so the working English path is unchanged.

static std::vector<std::string> convert_to_pinyin(const std::string& text) {
    if (core_pinyin::has_han(text))
        return core_pinyin::convert_char_to_pinyin(text);
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        int len = 1;
        if ((text[i] & 0x80) == 0)
            len = 1;
        else if ((text[i] & 0xE0) == 0xC0)
            len = 2;
        else if ((text[i] & 0xF0) == 0xE0)
            len = 3;
        else if ((text[i] & 0xF8) == 0xF0)
            len = 4;
        if (i + len > text.size())
            break;
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

// Forward declaration (defined after FFT helpers, used here for text encoder ConvNeXt blocks)
static void f5_linear(const float* x, const float* W, const float* bias, float* y, int T, int K, int N);

// ── Text Encoder (embedding + sinusoidal pos + ConvNeXtV2 blocks) ─

static std::vector<float> compute_text_embed(f5_tts_context* ctx, const int32_t* tokens, int n_tokens, int seq_len) {
    const auto& hp = ctx->hp;

    // tokens are in range [-1, vocab_size-1]. We add 1 to make 0 the filler.
    // Then pad/truncate to seq_len.
    std::vector<int32_t> padded(seq_len, 0);
    for (int i = 0; i < std::min(n_tokens, seq_len); i++) {
        padded[i] = tokens[i] + 1; // shift by 1 (filler = 0)
    }

    // Create text mask: 1 where padded == 0 (filler/padding)
    std::vector<float> text_mask(seq_len);
    for (int i = 0; i < seq_len; i++) {
        text_mask[i] = (padded[i] == 0) ? 1.0f : 0.0f;
    }

    // Embedding lookup — use pre-dequantized cache (loaded at init).
    const std::vector<float>& emb_weight = ctx->text_cache.emb;

    std::vector<float> text_emb(seq_len * hp.text_dim, 0.0f);
    for (int t = 0; t < seq_len; t++) {
        int idx = padded[t];
        if (idx >= 0 && idx < hp.text_num_embeds) {
            for (int d = 0; d < hp.text_dim; d++) {
                text_emb[t * hp.text_dim + d] = emb_weight[idx * hp.text_dim + d];
            }
        }
    }

    // Add sinusoidal position embedding (precompute_freqs_cis)
    // Only for positions within valid range
    std::vector<float> freqs(seq_len * hp.text_dim);
    sinusoidal_pos_emb(hp.text_dim, seq_len, freqs.data());
    // Mask positions beyond valid tokens
    for (int t = 0; t < seq_len; t++) {
        if (t < n_tokens) { // valid position
            for (int d = 0; d < hp.text_dim; d++) {
                text_emb[t * hp.text_dim + d] += freqs[t * hp.text_dim + d];
            }
        }
    }

    // Apply text mask (zero out filler positions)
    for (int t = 0; t < seq_len; t++) {
        if (text_mask[t] > 0.5f) {
            for (int d = 0; d < hp.text_dim; d++) {
                text_emb[t * hp.text_dim + d] = 0.0f;
            }
        }
    }

    // ConvNeXtV2 blocks — implemented on CPU for exact semantics.
    // Each block: dwconv(k=7) → LN → pw_up → GELU → GRN → pw_down → residual
    for (int b = 0; b < hp.conv_layers; b++) {
        const auto& bc = ctx->text_cache.blocks[b];
        int D = hp.text_dim;
        int T = seq_len;
        int K = 7, pad = 3;
        int inter_dim = D * 2; // intermediate_dim = text_dim * conv_mult = 512 * 2

        // Use pre-dequantized cached weights.
        const std::vector<float>& dw_w = bc.dw_w;
        const std::vector<float>& dw_b = bc.dw_b;
        const std::vector<float>& norm_w = bc.norm_w;
        const std::vector<float>& norm_b = bc.norm_b;
        const std::vector<float>& pw_up_w = bc.pw_up_w;
        const std::vector<float>& pw_up_b = bc.pw_up_b;
        const std::vector<float>& pw_down_w = bc.pw_down_w;
        const std::vector<float>& pw_down_b = bc.pw_down_b;
        const std::vector<float>& grn_g = bc.grn_g;
        const std::vector<float>& grn_b_v = bc.grn_b;

        // text_emb is (T, D) row-major
        std::vector<float> residual_v = text_emb;

        // 1. Depthwise conv (groups=D, kernel=7, pad=3, dilation=1)
        // For each channel c, convolve text_emb[:,c] with dw_w[c*K..(c+1)*K-1]
        std::vector<float> conv_out(T * D, 0.0f);
        for (int c = 0; c < D; c++) {
            for (int t = 0; t < T; t++) {
                float sum = dw_b[c];
                for (int k = 0; k < K; k++) {
                    int ti = t + k - pad;
                    if (ti >= 0 && ti < T) {
                        sum += text_emb[ti * D + c] * dw_w[c * K + k];
                    }
                }
                conv_out[t * D + c] = sum;
            }
        }

        // 2. LayerNorm (over D dimension, per time step)
        for (int t = 0; t < T; t++) {
            float mean = 0, var = 0;
            for (int d = 0; d < D; d++)
                mean += conv_out[t * D + d];
            mean /= D;
            for (int d = 0; d < D; d++) {
                float diff = conv_out[t * D + d] - mean;
                var += diff * diff;
            }
            var /= D;
            float inv_std = 1.0f / sqrtf(var + 1e-6f);
            for (int d = 0; d < D; d++) {
                conv_out[t * D + d] = ((conv_out[t * D + d] - mean) * inv_std) * norm_w[d] + norm_b[d];
            }
        }

        // 3. Pointwise up: (T, D) × (inter_dim, D)^T → (T, inter_dim)
        std::vector<float> up_out(T * inter_dim);
        f5_linear(conv_out.data(), pw_up_w.data(), pw_up_b.data(), up_out.data(), T, D, inter_dim);

        // 4. GELU (exact: x * 0.5 * (1 + erf(x/sqrt(2))))
        for (auto& v : up_out) {
            v = v * 0.5f * (1.0f + erff(v / sqrtf(2.0f)));
        }

        // 5. GRN: Gx = L2_norm(x, dim=T), Nx = Gx / (mean(Gx, dim=D) + eps)
        // out = gamma * (x * Nx) + beta + x
        // Gx[t, d] = ||x[:, d]||_2 (norm across T for each feature d)
        // Wait, the PyTorch code does: Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        // In (B, T, D) layout, dim=1 is T. So Gx = L2 norm across T → shape (B, 1, D)
        std::vector<float> gx(inter_dim, 0.0f);
        for (int d = 0; d < inter_dim; d++) {
            float sum_sq = 0;
            for (int t = 0; t < T; t++) {
                float v = up_out[t * inter_dim + d];
                sum_sq += v * v;
            }
            gx[d] = sqrtf(sum_sq);
        }
        // Nx = Gx / (mean(Gx, dim=-1) + eps)  → mean over D
        float gx_mean = 0;
        for (int d = 0; d < inter_dim; d++)
            gx_mean += gx[d];
        gx_mean /= inter_dim;
        std::vector<float> nx(inter_dim);
        for (int d = 0; d < inter_dim; d++)
            nx[d] = gx[d] / (gx_mean + 1e-6f);
        // out = gamma * (x * Nx) + beta + x
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < inter_dim; d++) {
                float x_val = up_out[t * inter_dim + d];
                up_out[t * inter_dim + d] = grn_g[d] * (x_val * nx[d]) + grn_b_v[d] + x_val;
            }
        }

        // 6. Pointwise down: (T, inter_dim) × (D, inter_dim)^T → (T, D)
        std::vector<float> down_out(T * D);
        f5_linear(up_out.data(), pw_down_w.data(), pw_down_b.data(), down_out.data(), T, inter_dim, D);

        // 7. Residual
        for (int i = 0; i < T * D; i++) {
            text_emb[i] = residual_v[i] + down_out[i];
        }

        // Apply text mask
        for (int t = 0; t < T; t++) {
            if (text_mask[t] > 0.5f) {
                for (int d = 0; d < D; d++)
                    text_emb[t * D + d] = 0.0f;
            }
        }
    }

    return text_emb;
}

// ── Radix-2 FFT (reused by mel + ISTFT) ─────────────────────────

// In-place radix-2 FFT. sign=+1 for forward, -1 for inverse.
// Arrays re[] and im[] must have length N (power of 2).
// For inverse, caller must divide by N afterwards.
static void fft_radix2(float* re, float* im, int N, int sign) {
    int log2n = 0;
    for (int tmp = N; tmp > 1; tmp >>= 1)
        log2n++;
    // Bit-reversal permutation
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
    // Butterfly passes
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
                float wt = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = wt;
            }
        }
    }
}

// ── HTK mel scale ───────────────────────────────────────────────

static float hz_to_mel(float f) {
    return 2595.0f * log10f(1.0f + f / 700.0f);
}
static float mel_to_hz(float m) {
    return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
}

// ── Mel spectrogram (vocos-type, 24kHz) ──────────────────────────
//
// Matches torchaudio.transforms.MelSpectrogram with:
//   sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256,
//   n_mels=100, power=1, center=True, normalized=False, norm=None
// Followed by: clamp(min=1e-5).log()
//
// Returns (T, n_mels) row-major mel spectrogram.

// sr, and the optional shipped_fb / shipped_window / center args, are #387
// additions. Vocos (stock F5) passes sr=24000, shipped_*=nullptr, center=true
// → identical behaviour. sbhifigan16k passes sr=16000, the shipped slaney
// filterbank + Hann window, and center=false (torchaudio Spectrogram default).
static std::vector<float> compute_mel_spectrogram(const float* pcm_24k, int n_samples, int n_fft, int hop_length,
                                                  int win_length, int n_mels, int& T_out, float sr = 24000.0f,
                                                  const std::vector<float>* shipped_fb = nullptr,
                                                  const std::vector<float>* shipped_window = nullptr,
                                                  bool center = true) {
    int n_freqs = n_fft / 2 + 1; // 513

    // ── Mel filterbank (n_freqs × n_mels): shipped (slaney) or HTK-built ──
    std::vector<float> mel_fb_local;
    const float* mel_fb = nullptr;
    if (shipped_fb && (int)shipped_fb->size() == n_freqs * n_mels) {
        mel_fb = shipped_fb->data(); // slaney, copied verbatim from the GGUF
    } else {
        mel_fb_local.assign(n_freqs * n_mels, 0.0f);
        float f_min = 0.0f, f_max = sr / 2.0f;
        float mel_min = hz_to_mel(f_min);
        float mel_max = hz_to_mel(f_max);
        std::vector<float> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * (float)i / (float)(n_mels + 1));
        for (int m = 0; m < n_mels; m++) {
            float f_left = mel_points[m];
            float f_center = mel_points[m + 1];
            float f_right = mel_points[m + 2];
            for (int k = 0; k < n_freqs; k++) {
                float f = (float)k * sr / (float)n_fft;
                float val = 0.0f;
                if (f >= f_left && f <= f_center && f_center > f_left)
                    val = (f - f_left) / (f_center - f_left);
                else if (f > f_center && f <= f_right && f_right > f_center)
                    val = (f_right - f) / (f_right - f_center);
                mel_fb_local[k * n_mels + m] = val;
            }
        }
        mel_fb = mel_fb_local.data();
    }

    // ── Hann window: shipped (torch periodic) or built ──
    std::vector<float> hann_local;
    const float* hann = nullptr;
    if (shipped_window && (int)shipped_window->size() == win_length) {
        hann = shipped_window->data();
    } else {
        hann_local.assign(win_length, 0.0f);
        for (int i = 0; i < win_length; i++)
            hann_local[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)win_length));
        hann = hann_local.data();
    }

    // ── Padding: center=true reflect-pads n_fft/2 each side (Vocos);
    //    center=false (sbhifigan16k) frames the raw signal, no padding. ──
    const int pad = center ? n_fft / 2 : 0;
    int padded_len = n_samples + 2 * pad;
    std::vector<float> padded(padded_len);
    if (center) {
        for (int i = 0; i < pad; i++)
            padded[i] = pcm_24k[pad - i]; // reflect: index 1,2,3,...,pad
        for (int i = 0; i < n_samples; i++)
            padded[pad + i] = pcm_24k[i];
        for (int i = 0; i < pad; i++)
            padded[pad + n_samples + i] = pcm_24k[n_samples - 2 - i]; // reflect
    } else {
        for (int i = 0; i < n_samples; i++)
            padded[i] = pcm_24k[i];
    }

    // ── STFT frames ──
    int T = (padded_len - n_fft) / hop_length + 1;
    T_out = T;

    // ── Process each frame: window → FFT → magnitude → mel → log ──
    std::vector<float> mel_spec(T * n_mels);
    std::vector<float> frame_re(n_fft), frame_im(n_fft);
    std::vector<float> mag(n_freqs);

    for (int t = 0; t < T; t++) {
        int offset = t * hop_length;

        // Window the frame
        for (int i = 0; i < n_fft; i++) {
            frame_re[i] = padded[offset + i] * hann[i];
            frame_im[i] = 0.0f;
        }

        // Forward FFT (sign = -1 for forward DFT convention e^{-j2pi})
        fft_radix2(frame_re.data(), frame_im.data(), n_fft, -1);

        // Magnitude spectrum (power=1)
        for (int k = 0; k < n_freqs; k++) {
            mag[k] = sqrtf(frame_re[k] * frame_re[k] + frame_im[k] * frame_im[k]);
        }

        // Mel filterbank multiplication: mel[m] = sum_k mag[k] * fb[k, m]
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; k++)
                sum += mag[k] * mel_fb[k * n_mels + m];
            // Clamp and log
            if (sum < 1e-5f)
                sum = 1e-5f;
            mel_spec[t * n_mels + m] = logf(sum);
        }
    }

    return mel_spec;
}

// ── DiT forward (one ODE step) ───────────────────────────────────
//
// Runs the full DiT: input_embed → 22 blocks → final adaln + proj.
// Returns velocity prediction (T, mel_dim).
//
// Bypass Accelerate (validate scalar == GEMM, or run on non-Apple): set
// F5_FORCE_SCALAR=1.
static bool f5_use_scalar() {
#if defined(HAVE_ACCELERATE)
    static const bool force_scalar = crispasr_env::get("CRISPASR_F5_FORCE_SCALAR") != nullptr;
    return force_scalar;
#else
    return true;
#endif
}

// y[T,N] = x[T,K] @ W[N,K]^T + bias[N]  (PyTorch nn.Linear). PLAN §182:
// Accelerate cblas_sgemm replaces the scalar triple loops in the F5 DiT
// (input projection) and Vocos vocoder (pointwise projections) — together the
// bulk of CPU compute. The scalar fallback is the previous behaviour.
static void f5_linear(const float* x, const float* W, const float* bias, float* y, int T, int K, int N) {
#if defined(HAVE_ACCELERATE)
    if (!f5_use_scalar()) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, N, K, 1.0f, x, K, W, K, 0.0f, y, N);
        if (bias) {
            for (int t = 0; t < T; t++) {
                float* row = y + (size_t)t * N;
                for (int o = 0; o < N; o++)
                    row[o] += bias[o];
            }
        }
        return;
    }
#endif
    for (int t = 0; t < T; t++) {
        const float* xr = x + (size_t)t * K;
        float* yr = y + (size_t)t * N;
        for (int o = 0; o < N; o++) {
            const float* wr = W + (size_t)o * K;
            float s = bias ? bias[o] : 0.0f;
            for (int k = 0; k < K; k++)
                s += xr[k] * wr[k];
            yr[o] = s;
        }
    }
}

// Grouped causal-"same" Conv1d on (T, C) row-major data (C inner). weight
// layout [C_out, C_in_per_group, K]; in_ch == out_ch == C; symmetric pad.
// Output (T, C). PLAN §182: per-group im2col + cblas_sgemm replaces the DiT's
// ConvPositionEmbedding scalar loop (~4 GFLOP/pass, the DiT's worst hot spot).
static void f5_grouped_conv1d(const float* in, const float* wt, const float* bias, float* out, int T, int C, int K,
                              int pad, int groups) {
    const int cpg = C / groups; // channels per group (in == out)
#if defined(HAVE_ACCELERATE)
    if (!f5_use_scalar()) {
        std::vector<float> col((size_t)cpg * K * T);
        std::vector<float> Wg((size_t)cpg * cpg * K);
        std::vector<float> og((size_t)cpg * T);
        for (int g = 0; g < groups; g++) {
            const int c0 = g * cpg;
            // col[(ic*K+k), t] = in[(t+k-pad)*C + c0+ic]  (0 if out of bounds)
            for (int ic = 0; ic < cpg; ic++)
                for (int k = 0; k < K; k++) {
                    float* crow = col.data() + (size_t)(ic * K + k) * T;
                    for (int t = 0; t < T; t++) {
                        int ti = t + k - pad;
                        crow[t] = (ti >= 0 && ti < T) ? in[(size_t)ti * C + c0 + ic] : 0.0f;
                    }
                }
            // Wg[oc, ic*K+k] = wt[(c0+oc)*cpg*K + ic*K + k] (contiguous per oc)
            for (int oc = 0; oc < cpg; oc++)
                std::memcpy(Wg.data() + (size_t)oc * cpg * K, wt + (size_t)(c0 + oc) * cpg * K,
                            (size_t)cpg * K * sizeof(float));
            // og(cpg, T) = Wg(cpg, cpg*K) @ col(cpg*K, T)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cpg, T, cpg * K, 1.0f, Wg.data(), cpg * K,
                        col.data(), T, 0.0f, og.data(), T);
            for (int oc = 0; oc < cpg; oc++) {
                float b = bias ? bias[c0 + oc] : 0.0f;
                const float* orow = og.data() + (size_t)oc * T;
                for (int t = 0; t < T; t++)
                    out[(size_t)t * C + c0 + oc] = orow[t] + b;
            }
        }
        return;
    }
#endif
    for (int g = 0; g < groups; g++) {
        const int c0 = g * cpg;
        for (int oc = c0; oc < c0 + cpg; oc++)
            for (int t = 0; t < T; t++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < cpg; ic++)
                    for (int k = 0; k < K; k++) {
                        int ti = t + k - pad;
                        if (ti >= 0 && ti < T)
                            sum += in[(size_t)ti * C + c0 + ic] * wt[(size_t)oc * cpg * K + (size_t)ic * K + k];
                    }
                out[(size_t)t * C + oc] = sum;
            }
    }
}

// ── Fused DiT graph: build (once per T) ──────────────────────────

static bool f5_dit_cache_build(f5_tts_context* ctx, int T, int B) {
    auto& cache = ctx->dit_cache;
    if (cache.T_cached == T && cache.B_cached == B && cache.galloc)
        return true;
    cache.reset();

    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    int dim = hp.dim;
    int n_heads = hp.heads;
    int head_dim = hp.dim_head;

    // 22 blocks × ~42 tensors + final block + inputs: ~950 tensors total.
    // ~4 MB is well above the ~270 KB actually needed.
    struct ggml_init_params p = {4 * 1024 * 1024, nullptr, true};
    cache.gctx = ggml_init(p);
    if (!cache.gctx)
        return false;

    // Graph inputs. hidden_in carries B independent latents ([dim,T,B]); the
    // DiT is identical per batch entry, so t_emb (shared timestep) and pos
    // (shared) broadcast over B. B=1 is the plain path; B=2 is batched CFG.
    cache.hidden_in = ggml_new_tensor_3d(cache.gctx, GGML_TYPE_F32, dim, T, B);
    ggml_set_name(cache.hidden_in, "hidden_in");
    ggml_set_input(cache.hidden_in);

    cache.t_emb_in = ggml_new_tensor_1d(cache.gctx, GGML_TYPE_F32, dim);
    ggml_set_name(cache.t_emb_in, "t_emb");
    ggml_set_input(cache.t_emb_in);

    cache.pos_in = ggml_new_tensor_1d(cache.gctx, GGML_TYPE_I32, T);
    ggml_set_name(cache.pos_in, "pos");
    ggml_set_input(cache.pos_in);

    // Chain all 22 DiT blocks
    // #294: optionally feed the big mat-muls F16 activations (weights are F16
    // already, so the RHS is the only F32 operand). Cast only the mat-mul inputs;
    // everything else (norm/adaln/rope/flash) stays F32, output stays F32.
    const bool f16act = f5_f16_act_enabled();
    auto A = [&](ggml_tensor* t) { return f16act ? ggml_cast(cache.gctx, t, GGML_TYPE_F16) : t; };
    ggml_tensor* x = cache.hidden_in;
    for (int i = 0; i < hp.depth; i++) {
        const auto& blk = w.dit_blocks[i];

        // AdaLN modulation: silu(t_emb) → linear → 6×dim split
        ggml_tensor* emb = ggml_silu(cache.gctx, cache.t_emb_in);
        emb = ggml_mul_mat(cache.gctx, blk.adaln_weight, emb);
        emb = ggml_add(cache.gctx, emb, blk.adaln_bias);

        ggml_tensor* shift_msa = ggml_view_1d(cache.gctx, emb, dim, 0);
        ggml_tensor* scale_msa = ggml_view_1d(cache.gctx, emb, dim, 1 * dim * sizeof(float));
        ggml_tensor* gate_msa = ggml_view_1d(cache.gctx, emb, dim, 2 * dim * sizeof(float));
        ggml_tensor* shift_mlp = ggml_view_1d(cache.gctx, emb, dim, 3 * dim * sizeof(float));
        ggml_tensor* scale_mlp = ggml_view_1d(cache.gctx, emb, dim, 4 * dim * sizeof(float));
        ggml_tensor* gate_mlp = ggml_view_1d(cache.gctx, emb, dim, 5 * dim * sizeof(float));

        // Pre-norm (AdaLN): norm(x) * (1 + scale) + shift
        ggml_tensor* norm_x = ggml_norm(cache.gctx, x, 1e-6f);
        ggml_tensor* scaled = ggml_mul(cache.gctx, norm_x, scale_msa);
        norm_x = ggml_add(cache.gctx, norm_x, scaled);
        norm_x = ggml_add(cache.gctx, norm_x, shift_msa);

        // QKV projections (share one F16 cast of the norm output across q/k/v)
        ggml_tensor* norm_x_m = A(norm_x);
        ggml_tensor* q = ggml_mul_mat(cache.gctx, blk.attn_q_weight, norm_x_m);
        q = ggml_add(cache.gctx, q, blk.attn_q_bias);
        ggml_tensor* k = ggml_mul_mat(cache.gctx, blk.attn_k_weight, norm_x_m);
        k = ggml_add(cache.gctx, k, blk.attn_k_bias);
        ggml_tensor* v = ggml_mul_mat(cache.gctx, blk.attn_v_weight, norm_x_m);
        v = ggml_add(cache.gctx, v, blk.attn_v_bias);

        // Reshape for RoPE: (dim, T) → (head_dim, n_heads, T)
        q = ggml_reshape_4d(cache.gctx, q, head_dim, n_heads, T, B);
        k = ggml_reshape_4d(cache.gctx, k, head_dim, n_heads, T, B);
        v = ggml_reshape_4d(cache.gctx, v, head_dim, n_heads, T, B);

        // RoPE (NORMAL mode, freq_base=10000)
        q = ggml_rope_ext(cache.gctx, q, cache.pos_in, nullptr, head_dim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(cache.gctx, k, cache.pos_in, nullptr, head_dim, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute for flash_attn: (head_dim, n_heads, T) → (head_dim, T, n_heads)
        q = ggml_permute(cache.gctx, q, 0, 2, 1, 3);
        k = ggml_permute(cache.gctx, k, 0, 2, 1, 3);
        v = ggml_permute(cache.gctx, v, 0, 2, 1, 3);

        // Flash attention (bidirectional, no mask)
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* attn_out = ggml_flash_attn_ext(cache.gctx, q, k, v, nullptr, attn_scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_3d(cache.gctx, attn_out, dim, T, B);

        // O-proj + gated residual
        ggml_tensor* attn_proj = ggml_mul_mat(cache.gctx, blk.attn_o_weight, A(attn_out));
        attn_proj = ggml_add(cache.gctx, attn_proj, blk.attn_o_bias);
        ggml_tensor* gated_attn = ggml_mul(cache.gctx, attn_proj, gate_msa);
        ggml_tensor* x_res = ggml_add(cache.gctx, x, gated_attn);

        // FFN pre-norm
        ggml_tensor* ff_norm = ggml_norm(cache.gctx, x_res, 1e-6f);
        ggml_tensor* ff_scaled = ggml_mul(cache.gctx, ff_norm, scale_mlp);
        ff_norm = ggml_add(cache.gctx, ff_norm, ff_scaled);
        ff_norm = ggml_add(cache.gctx, ff_norm, shift_mlp);

        // FFN: up → GELU → down
        ggml_tensor* ff = ggml_mul_mat(cache.gctx, blk.ffn_up_weight, A(ff_norm));
        ff = ggml_add(cache.gctx, ff, blk.ffn_up_bias);
        ff = ggml_gelu(cache.gctx, ff);
        ff = ggml_mul_mat(cache.gctx, blk.ffn_down_weight, A(ff));
        ff = ggml_add(cache.gctx, ff, blk.ffn_down_bias);

        // Gated residual
        x = ggml_add(cache.gctx, x_res, ggml_mul(cache.gctx, ff, gate_mlp));
    }

    // Final AdaLN + projection → velocity (mel_dim, T)
    {
        ggml_tensor* emb = ggml_silu(cache.gctx, cache.t_emb_in);
        emb = ggml_mul_mat(cache.gctx, w.final_adaln_weight, emb);
        emb = ggml_add(cache.gctx, emb, w.final_adaln_bias);

        ggml_tensor* fscale = ggml_view_1d(cache.gctx, emb, dim, 0);
        ggml_tensor* fshift = ggml_view_1d(cache.gctx, emb, dim, dim * sizeof(float));

        ggml_tensor* norm_x = ggml_norm(cache.gctx, x, 1e-6f);
        ggml_tensor* fn_sc = ggml_mul(cache.gctx, norm_x, fscale);
        norm_x = ggml_add(cache.gctx, norm_x, fn_sc);
        norm_x = ggml_add(cache.gctx, norm_x, fshift);

        ggml_tensor* vel = ggml_mul_mat(cache.gctx, w.final_proj_weight, A(norm_x));
        vel = ggml_add(cache.gctx, vel, w.final_proj_bias);
        ggml_set_name(vel, "velocity");
        ggml_set_output(vel);
        cache.output = vel;
    }

    // Build graph
    cache.gf = ggml_new_graph_custom(cache.gctx, 8192, false);
    ggml_build_forward_expand(cache.gf, cache.output);

    // Reserve then allocate memory layout on the compute backend (GPU when
    // use_gpu; falls back to backend_cpu otherwise). The DiT weights already
    // live on ctx->backend, so keeping the graph on the same backend keeps the
    // 64×-per-synthesis forward off the CPU — a single-backend gallocr compute
    // (no sched), matching the §206/§232 direct-gallocr pattern.
    cache.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_reserve(cache.galloc, cache.gf) || !ggml_gallocr_alloc_graph(cache.galloc, cache.gf)) {
        cache.reset();
        return false;
    }

    // Set constant position indices — persist across steps
    std::vector<int32_t> pos_data(T);
    for (int i = 0; i < T; i++)
        pos_data[i] = i;
    ggml_backend_tensor_set(cache.pos_in, pos_data.data(), 0, T * sizeof(int32_t));

    cache.T_cached = T;
    cache.B_cached = B;
    return true;
}

// ── Fused DiT graph: run one step ────────────────────────────────

// Run the DiT for B stacked latents in one graph. `hidden` is [dim,T,B]
// contiguous, output is [mel_dim,T,B] contiguous. B=1 is the plain path;
// B=2 batches the CFG cond+uncond passes into one dispatch chain (the DiT
// is per-forward dispatch-overhead-bound, so halving the dispatch count per
// step is the win). t_emb/pos are shared across B (same timestep/positions).
static std::vector<float> f5_dit_run_batched(f5_tts_context* ctx, const float* hidden, int T, int B,
                                             const float* t_emb_data) {
    auto& cache = ctx->dit_cache;
    if (!f5_dit_cache_build(ctx, T, B))
        return {};

    // Re-assign buffer pointers (fast for same T,B)
    if (!ggml_gallocr_alloc_graph(cache.galloc, cache.gf))
        return {};

    const int dim = ctx->hp.dim;
    ggml_backend_tensor_set(cache.hidden_in, hidden, 0, (size_t)T * dim * B * sizeof(float));
    ggml_backend_tensor_set(cache.t_emb_in, t_emb_data, 0, (size_t)dim * sizeof(float));
    // pos_in must be re-set every step: gallocr re-allocs the graph each call
    // and may alias this input's slot with a prior step's compute intermediate,
    // so a set-once value read back corrupt RoPE positions from step 1 on (the
    // same gallocr input-aliasing bug fixed in omnivoice §245 / voxtral #93).
    {
        std::vector<int32_t> pos_data(T);
        for (int i = 0; i < T; i++)
            pos_data[i] = i;
        ggml_backend_tensor_set(cache.pos_in, pos_data.data(), 0, (size_t)T * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(ctx->backend, cache.gf) != GGML_STATUS_SUCCESS)
        return {};

    const int mel_dim = ctx->hp.mel_dim;
    std::vector<float> velocity((size_t)T * mel_dim * B);
    ggml_backend_tensor_get(cache.output, velocity.data(), 0, velocity.size() * sizeof(float));
    return velocity;
}

// Plain single-latent DiT forward (B=1). Unchanged callers use this.
static std::vector<float> f5_dit_run(f5_tts_context* ctx, const float* hidden, int T, const float* t_emb_data) {
    return f5_dit_run_batched(ctx, hidden, T, 1, t_emb_data);
}

// x:       (T, mel_dim)   — current ODE state
// cond:    (T, mel_dim)   — conditioning (masked ref mel)
// text:    (T, text_dim)  — text embedding
// time_emb: (dim,)        — timestep embedding

// F5_BENCH accumulators: split each DiT forward into host input-embed vs the
// GPU DiT graph, summed across all cond/uncond passes and printed at synth end.
static int64_t g_f5_hostembed_us = 0;
static int64_t g_f5_ditgraph_us = 0;

// ── GPU InputEmbedding (#294, opt-in) ────────────────────────────
// Mish = x * tanh(softplus(x)).
static ggml_tensor* f5_mish_graph(ggml_context* g, ggml_tensor* x) {
    return ggml_mul(g, x, ggml_tanh(g, ggml_softplus(g, x)));
}

// Symmetric ("same") grouped Conv1d for F5's ConvPositionEmbedding.
// h: (C, T) F32 — channel-first. w: (K, C_per_group, C). b: (C,). pad = (K-1)/2 so
// output length == T. Mirrors cosyvoice3_tts::cv3_causal_grouped_conv1d but with
// symmetric padding instead of causal left-pad.
static ggml_tensor* f5_sym_grouped_conv1d_graph(ggml_context* g, ggml_tensor* h, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int C_per_g = (int)w->ne[1];
    const int C = (int)w->ne[2];
    const int T = (int)h->ne[1];
    const int G = C / C_per_g;
    const int pad = (K - 1) / 2;
    ggml_tensor* out = nullptr;
    for (int grp = 0; grp < G; grp++) {
        const size_t c0 = (size_t)grp * C_per_g;
        ggml_tensor* h_g = ggml_view_2d(g, h, C_per_g, T, h->nb[1], c0 * h->nb[0]);
        h_g = ggml_cont(g, ggml_transpose(g, h_g)); // (T, C_per_g)
        ggml_tensor* w_g = ggml_view_3d(g, w, K, C_per_g, C_per_g, w->nb[1], w->nb[2], c0 * w->nb[2]);
        w_g = ggml_cont(g, w_g);
        ggml_tensor* y = ggml_conv_1d(g, w_g, h_g, /*s*/ 1, /*p*/ pad, /*d*/ 1); // (T, C_per_g)
        y = ggml_cont(g, ggml_transpose(g, y));                                  // (C_per_g, T)
        ggml_tensor* b_g = ggml_view_1d(g, b, C_per_g, c0 * b->nb[0]);
        y = ggml_add(g, y, b_g);
        out = out ? ggml_concat(g, out, y, 0) : y;
    }
    return out;
}

// Build the cached GPU embed graph (once per T). B is always 1 here — the CFG arms
// call this separately. cat_in (cat_dim,T,1) → input_proj → 2×(grouped conv + Mish)
// + residual → hidden_out (dim,T,1).
static bool f5_embed_cache_build(f5_tts_context* ctx, int T) {
    auto& ec = ctx->embed_cache;
    if (ec.T_cached == T && ec.B_cached == 1 && ec.galloc)
        return true;
    ec.reset();
    const auto& hp = ctx->hp;
    const auto& w = ctx->w;
    const int cat_dim = hp.mel_dim + hp.mel_dim + hp.text_dim;

    struct ggml_init_params p = {4 * 1024 * 1024, nullptr, true};
    ec.gctx = ggml_init(p);
    if (!ec.gctx)
        return false;

    ec.cat_in = ggml_new_tensor_2d(ec.gctx, GGML_TYPE_F32, cat_dim, T);
    ggml_set_name(ec.cat_in, "cat_in");
    ggml_set_input(ec.cat_in);

    // input_proj: (cat_dim,T) → (dim,T)
    ggml_tensor* h = ggml_mul_mat(ec.gctx, w.input_proj_weight, ec.cat_in);
    h = ggml_add(ec.gctx, h, w.input_proj_bias);
    ggml_tensor* proj_out = h;
    // 2 × (grouped conv-pos + Mish)
    h = f5_mish_graph(ec.gctx, f5_sym_grouped_conv1d_graph(ec.gctx, h, w.conv_pos_0_weight, w.conv_pos_0_bias));
    h = f5_mish_graph(ec.gctx, f5_sym_grouped_conv1d_graph(ec.gctx, h, w.conv_pos_1_weight, w.conv_pos_1_bias));
    h = ggml_add(ec.gctx, h, proj_out); // residual
    ggml_set_name(h, "hidden_out");
    ggml_set_output(h);
    ec.hidden_out = h;

    ec.gf = ggml_new_graph_custom(ec.gctx, 2048, false);
    ggml_build_forward_expand(ec.gf, ec.hidden_out);
    ec.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_reserve(ec.galloc, ec.gf) || !ggml_gallocr_alloc_graph(ec.galloc, ec.gf)) {
        ec.reset();
        return false;
    }
    ec.T_cached = T;
    ec.B_cached = 1;
    return true;
}

// GPU equivalent of f5_compute_hidden (one arm). Builds the cat on host (trivial,
// no FLOP), uploads, runs input_proj + conv-pos on the backend, returns hidden
// [T*dim] in the same layout as the host path.
static std::vector<float> f5_compute_hidden_gpu(f5_tts_context* ctx, const float* x_data, int T, int mel_dim,
                                                const float* cond_data, const float* text_data, int text_dim,
                                                bool drop_audio_cond) {
    const int dim = ctx->hp.dim;
    const int cat_dim = mel_dim + mel_dim + text_dim;
    if (!f5_embed_cache_build(ctx, T))
        return {};
    auto& ec = ctx->embed_cache;
    if (!ggml_gallocr_alloc_graph(ec.galloc, ec.gf))
        return {};
    std::vector<float> cat_input((size_t)T * cat_dim);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < mel_dim; d++)
            cat_input[t * cat_dim + d] = x_data[t * mel_dim + d];
        for (int d = 0; d < mel_dim; d++)
            cat_input[t * cat_dim + mel_dim + d] = drop_audio_cond ? 0.0f : cond_data[t * mel_dim + d];
        for (int d = 0; d < text_dim; d++)
            cat_input[t * cat_dim + mel_dim + mel_dim + d] = text_data[t * text_dim + d];
    }
    ggml_backend_tensor_set(ec.cat_in, cat_input.data(), 0, cat_input.size() * sizeof(float));
    if (ggml_backend_graph_compute(ctx->backend, ec.gf) != GGML_STATUS_SUCCESS)
        return {};
    std::vector<float> hidden((size_t)T * dim);
    ggml_backend_tensor_get(ec.hidden_out, hidden.data(), 0, hidden.size() * sizeof(float));
    return hidden;
}

// Host-side InputEmbedding: cat(x, cond, text) → input_proj → +conv_pos_embed.
// Returns `hidden` [dim,T] (== ggml [dim,T] memory layout). Shared by the plain
// per-pass path and the batched-CFG path (which computes it once per arm).
static std::vector<float> f5_compute_hidden(f5_tts_context* ctx, const float* x_data, int T, int mel_dim,
                                            const float* cond_data, const float* text_data, int text_dim,
                                            bool drop_audio_cond, bool drop_text, int step_idx) {
    (void)drop_text;
    // #294 opt-in: run the whole InputEmbedding on the GPU backend.
    if (f5_embed_gpu_enabled())
        return f5_compute_hidden_gpu(ctx, x_data, T, mel_dim, cond_data, text_data, text_dim, drop_audio_cond);
    const int dim = ctx->hp.dim;

    // Concatenate along feature dim: (T, mel_dim + mel_dim + text_dim) = (T, 712)
    int cat_dim = mel_dim + mel_dim + text_dim;
    std::vector<float> cat_input(T * cat_dim);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < mel_dim; d++)
            cat_input[t * cat_dim + d] = x_data[t * mel_dim + d];
        for (int d = 0; d < mel_dim; d++)
            cat_input[t * cat_dim + mel_dim + d] = drop_audio_cond ? 0.0f : cond_data[t * mel_dim + d];
        for (int d = 0; d < text_dim; d++)
            cat_input[t * cat_dim + mel_dim + mel_dim + d] = text_data[t * text_dim + d];
    }

    if (step_idx == 0 && !drop_audio_cond) {
        dump_stage(ctx, "cat_input", cat_input.data(), cat_input.size());
        dump_stage(ctx, "debug_proj_weight", ctx->emb_cache.input_proj_w.data(), ctx->emb_cache.input_proj_w.size());
    }

    // Linear projection: (T, 712) → (T, 1024) — use pre-cached F32 weights
    std::vector<float> hidden(T * dim, 0.0f);
    f5_linear(cat_input.data(), ctx->emb_cache.input_proj_w.data(), ctx->emb_cache.input_proj_b.data(), hidden.data(),
              T, cat_dim, dim);

    if (step_idx == 0 && !drop_audio_cond) {
        dump_stage(ctx, "input_proj_out", hidden.data(), hidden.size());
    }

    // ConvPositionEmbedding: 2× (Conv1d(dim, dim, k=31, g=16, p=15) + Mish)
    {
        int K = 31, pad_k = 15, groups = 16;

        auto grouped_conv_mish = [&](const std::vector<float>& input, const float* wt, const float* bias) {
            std::vector<float> output(T * dim, 0.0f);
            f5_grouped_conv1d(input.data(), wt, bias, output.data(), T, dim, K, pad_k, groups);
            for (auto& v : output) {
                float sp = logf(1.0f + expf(v));
                v = v * tanhf(sp);
            }
            return output;
        };

        std::vector<float> proj_out = hidden;
        hidden = grouped_conv_mish(hidden, ctx->emb_cache.conv_pos_0_w.data(), ctx->emb_cache.conv_pos_0_b.data());
        hidden = grouped_conv_mish(hidden, ctx->emb_cache.conv_pos_1_w.data(), ctx->emb_cache.conv_pos_1_b.data());

        for (size_t i = 0; i < hidden.size(); i++) {
            hidden[i] += proj_out[i];
        }
    }

    if (step_idx == 0 && !drop_audio_cond) {
        dump_stage(ctx, "input_embed", hidden.data(), hidden.size());
    }
    return hidden;
}

static std::vector<float> dit_forward(f5_tts_context* ctx, const float* x_data, int T, int mel_dim,
                                      const float* cond_data, const float* text_data, int text_dim,
                                      const float* time_emb_data, bool drop_audio_cond, bool drop_text, int step_idx) {
    const int64_t _f5_t_start = ggml_time_us();
    std::vector<float> hidden = f5_compute_hidden(ctx, x_data, T, mel_dim, cond_data, text_data, text_dim,
                                                  drop_audio_cond, drop_text, step_idx);

    // ── 22 DiT blocks + final AdaLN/proj (fused single graph) ──
    {
        const int64_t _f5_t_pre_dit = ggml_time_us();
        g_f5_hostembed_us += _f5_t_pre_dit - _f5_t_start;
        auto velocity = f5_dit_run(ctx, hidden.data(), T, time_emb_data);
        g_f5_ditgraph_us += ggml_time_us() - _f5_t_pre_dit;
        if (velocity.empty())
            return {};
        if (step_idx == 0 && !drop_audio_cond)
            dump_stage(ctx, "dit_output", velocity.data(), velocity.size());
        return velocity;
    }
}

// ── Vocos vocoder ────────────────────────────────────────────────

// ── CPU Conv1d helper (standard or depthwise) ──────────────────────
// input/output: (C, T) row-major. weight: (C_out, C_in_per_group, K).
static void cpu_conv1d(const float* input, int C_in, int T, const float* weight, const float* bias, int C_out, int K,
                       int pad, int groups, float* output) {
    int ch_per_group_in = C_in / groups;
    int ch_per_group_out = C_out / groups;
    for (int g = 0; g < groups; g++) {
        int oc_start = g * ch_per_group_out;
        int ic_start = g * ch_per_group_in;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int oc = oc_start; oc < oc_start + ch_per_group_out; oc++) {
            for (int t = 0; t < T; t++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic_local = 0; ic_local < ch_per_group_in; ic_local++) {
                    int ic = ic_start + ic_local;
                    for (int k = 0; k < K; k++) {
                        int ti = t + k - pad;
                        if (ti >= 0 && ti < T) {
                            int w_idx = oc * ch_per_group_in * K + ic_local * K + k;
                            sum += input[ic * T + ti] * weight[w_idx];
                        }
                    }
                }
                output[oc * T + t] = sum;
            }
        }
    }
}

// Dilated 1-D conv (groups=1), same-padding. #387 HiFi-GAN resblocks.
static void cpu_conv1d_dil(const float* input, int C_in, int T, const float* weight, const float* bias, int C_out,
                           int K, int dilation, float* output) {
    const int pad = dilation * (K - 1) / 2; // "same" for odd K
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oc = 0; oc < C_out; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < C_in; ic++) {
                const float* in = input + (size_t)ic * T;
                const float* w = weight + ((size_t)oc * C_in + ic) * K;
                for (int k = 0; k < K; k++) {
                    int ti = t + k * dilation - pad;
                    if (ti >= 0 && ti < T)
                        sum += in[ti] * w[k];
                }
            }
            output[(size_t)oc * T + t] = sum;
        }
    }
}

static inline void cpu_leaky_relu(std::vector<float>& x, float slope) {
    for (float& v : x)
        v = v > 0.0f ? v : slope * v;
}

// ── CPU LayerNorm over last dim of (T, D) data ─────────────────────
static void cpu_layer_norm(float* data, int T, int D, const float* gamma, const float* beta, float eps) {
    for (int t = 0; t < T; t++) {
        float* row = data + t * D;
        float mean = 0;
        for (int d = 0; d < D; d++)
            mean += row[d];
        mean /= D;
        float var = 0;
        for (int d = 0; d < D; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var /= D;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++) {
            row[d] = ((row[d] - mean) * inv_std) * gamma[d] + beta[d];
        }
    }
}

// ── Transpose (C, T) ↔ (T, C) ──────────────────────────────────────
static void transpose_ct(const float* src, int C, int T, float* dst) {
    // src: (C, T) → dst: (T, C)
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            dst[t * C + c] = src[c * T + t];
}

static void transpose_tc(const float* src, int T, int C, float* dst) {
    // src: (T, C) → dst: (C, T)
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            dst[c * T + t] = src[t * C + c];
}

// ── CPU HiFi-GAN v1 decoder (#387 Raon-OpenTTS sbhifigan16k) ──────────────
// mel_data is (T_mel, mel_dim) row-major (log-mel). Mirrors the reference
// HifiganGenerator: conv_pre → for each of 4 stages { LeakyReLU →
// ConvTranspose1d(rate) → sum_j resblock_j / 3 } → LeakyReLU → conv_post →
// tanh. Weights come from ctx->hifigan_w (weight-norm fused in the converter);
// the arch is the fixed sbhifigan16k config from Raon's vocoder.py.
static std::vector<float> hifigan_decode(f5_tts_context* ctx, const float* mel_data, int T_mel, int mel_dim) {
    const auto& W = ctx->hifigan_w;
    auto has = [&](const std::string& n) { return W.find(n) != W.end(); };
    auto get = [&](const std::string& n) -> const std::vector<float>& { return W.at(n); };

    const int up_rates[4] = {8, 8, 2, 2};
    const int up_kernels[4] = {16, 16, 4, 4};
    const int rb_kernels[3] = {3, 7, 11};
    const int rb_dils[3] = {1, 3, 5};
    const float slope = 0.1f;
    int init_ch = 512;

    const auto voc_t0 = std::chrono::steady_clock::now();

    // mel (T, C) → (C, T) for conv layout.
    std::vector<float> x(mel_dim * T_mel);
    transpose_tc(mel_data, T_mel, mel_dim, x.data());
    int C = mel_dim, T = T_mel;

    // conv_pre: Conv1d(mel_dim, init_ch, k=7, pad=3)
    {
        std::vector<float> out((size_t)init_ch * T, 0.0f);
        cpu_conv1d(x.data(), C, T, get("voc.conv_pre.weight").data(), get("voc.conv_pre.bias").data(), init_ch, 7, 3, 1,
                   out.data());
        x = std::move(out);
        C = init_ch;
    }

    int ch = init_ch;
    for (int s = 0; s < 4; s++) {
        cpu_leaky_relu(x, slope);
        // ConvTranspose1d(ch, ch/2, k=up_kernels[s], stride=up_rates[s],
        // pad=(k-stride)/2). PyTorch CT weight layout is (in_ch, out_ch, k).
        const int out_ch = ch / 2;
        const int stride = up_rates[s], k = up_kernels[s], pad = (k - stride) / 2;
        const int T_out = (T - 1) * stride + k - 2 * pad;
        const std::vector<float>& uw = get("voc.ups." + std::to_string(s) + ".weight");
        const std::vector<float>& ub = get("voc.ups." + std::to_string(s) + ".bias");
        std::vector<float> up((size_t)out_ch * T_out, 0.0f);
        // Gather per output channel (parallel-safe: each oc writes its own row).
        // For each output position `to`, sum over the input taps that scatter
        // into it: to = ti*stride + kk - pad  ⇒  ti = (to + pad - kk)/stride.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int oc = 0; oc < out_ch; oc++) {
            float* orow = up.data() + (size_t)oc * T_out;
            for (int to = 0; to < T_out; to++) {
                float acc = ub[oc];
                for (int kk = 0; kk < k; kk++) {
                    int num = to + pad - kk;
                    if (num < 0 || (num % stride) != 0)
                        continue;
                    int ti = num / stride;
                    if (ti < 0 || ti >= T)
                        continue;
                    for (int ic = 0; ic < ch; ic++) {
                        float v = x[(size_t)ic * T + ti];
                        float w = uw[(((size_t)ic * out_ch) + oc) * k + kk];
                        acc += v * w;
                    }
                }
                orow[to] = acc;
            }
        }
        x = std::move(up);
        ch = out_ch;
        T = T_out;

        // MRF: mean of 3 resblocks over the current (ch, T).
        std::vector<float> acc((size_t)ch * T, 0.0f);
        for (int j = 0; j < 3; j++) {
            const int rb = s * 3 + j;
            std::vector<float> h = x; // resblock input
            for (int d = 0; d < 3; d++) {
                // convs1[d]: dilated, then convs2[d]: dilation 1, each pre-LeakyReLU
                std::vector<float> a = h;
                cpu_leaky_relu(a, slope);
                std::vector<float> b1((size_t)ch * T, 0.0f);
                const std::string p1 = "voc.resblocks." + std::to_string(rb) + ".convs1." + std::to_string(d);
                cpu_conv1d_dil(a.data(), ch, T, get(p1 + ".weight").data(), get(p1 + ".bias").data(), ch, rb_kernels[j],
                               rb_dils[d], b1.data());
                cpu_leaky_relu(b1, slope);
                std::vector<float> b2((size_t)ch * T, 0.0f);
                const std::string p2 = "voc.resblocks." + std::to_string(rb) + ".convs2." + std::to_string(d);
                cpu_conv1d_dil(b1.data(), ch, T, get(p2 + ".weight").data(), get(p2 + ".bias").data(), ch, rb_kernels[j],
                               1, b2.data());
                for (size_t i = 0; i < h.size(); i++)
                    h[i] += b2[i]; // residual
            }
            for (size_t i = 0; i < acc.size(); i++)
                acc[i] += h[i];
        }
        for (size_t i = 0; i < x.size(); i++)
            x[i] = acc[i] / 3.0f;
        (void)has;
    }

    // final LeakyReLU → conv_post: Conv1d(ch, 1, k=7, pad=3) → tanh
    cpu_leaky_relu(x, slope);
    std::vector<float> out((size_t)1 * T, 0.0f);
    cpu_conv1d(x.data(), ch, T, get("voc.conv_post.weight").data(), get("voc.conv_post.bias").data(), 1, 7, 3, 1,
               out.data());
    for (float& v : out)
        v = tanhf(v);

    if (ctx->verbosity >= 1) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - voc_t0).count();
        fprintf(stderr, "f5_tts: hifigan_decode T_mel=%d → %zu samples in %.1f ms\n", T_mel, out.size(), ms);
    }
    return out;
}

static std::vector<float> vocos_decode(f5_tts_context* ctx, const float* mel_data, int T_mel, int mel_dim) {
    const auto& hp = ctx->hp;
    int D = hp.voc_dim; // 512
    int T = T_mel;
    int inter_dim = hp.voc_intermediate_dim; // 1536

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: vocos_decode T=%d mel_dim=%d voc_dim=%d\n", T, mel_dim, D);
    }
    const auto voc_t0 = std::chrono::steady_clock::now();

    // mel_data is (T, 100) row-major. Convert to (100, T) for conv operations.
    std::vector<float> x_ct(mel_dim * T);
    transpose_tc(mel_data, T, mel_dim, x_ct.data());

    const auto& vc = ctx->voc_cache;

    // ── Embed: Conv1d(100, 512, k=7, pad=3) ──
    {
        std::vector<float> out(D * T, 0.0f);
        cpu_conv1d(x_ct.data(), mel_dim, T, vc.embed_w.data(), vc.embed_b.data(), D, 7, 3, 1, out.data());
        x_ct = std::move(out);
    }

    // ── Initial LayerNorm: transpose to (T, D), norm, transpose back ──
    {
        std::vector<float> x_td(T * D);
        transpose_ct(x_ct.data(), D, T, x_td.data());
        cpu_layer_norm(x_td.data(), T, D, vc.norm_w.data(), vc.norm_b.data(), 1e-6f);
        transpose_tc(x_td.data(), T, D, x_ct.data());
    }

    // ── 8× ConvNeXt blocks ──
    for (int b = 0; b < hp.voc_num_layers; b++) {
        const auto& bc = vc.blocks[b];
        std::vector<float> residual = x_ct; // (D, T)

        // 1. Depthwise Conv1d(512, 512, k=7, pad=3, groups=512)
        {
            std::vector<float> out(D * T, 0.0f);
            cpu_conv1d(x_ct.data(), D, T, bc.dw_w.data(), bc.dw_b.data(), D, 7, 3, D, out.data());
            x_ct = std::move(out);
        }

        // 2. Transpose to (T, D), LayerNorm
        std::vector<float> x_td(T * D);
        transpose_ct(x_ct.data(), D, T, x_td.data());
        cpu_layer_norm(x_td.data(), T, D, bc.norm_w.data(), bc.norm_b.data(), 1e-6f);

        // 3. Pointwise up: Linear(512, 1536)
        std::vector<float> up_out(T * inter_dim, 0.0f);
        f5_linear(x_td.data(), bc.pw_up_w.data(), bc.pw_up_b.data(), up_out.data(), T, D, inter_dim);

        // 4. GELU (exact: x * 0.5 * (1 + erf(x/sqrt(2))))
        for (auto& v : up_out) {
            v = v * 0.5f * (1.0f + erff(v / sqrtf(2.0f)));
        }

        // 5. Pointwise down: Linear(1536, 512)
        std::vector<float> down_out(T * D, 0.0f);
        f5_linear(up_out.data(), bc.pw_down_w.data(), bc.pw_down_b.data(), down_out.data(), T, inter_dim, D);

        // 6. Layer scale (gamma)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                down_out[t * D + d] *= bc.layer_scale[d];

        // 7. Transpose back to (D, T) and add residual
        transpose_tc(down_out.data(), T, D, x_ct.data());
        for (size_t i = 0; i < x_ct.size(); i++)
            x_ct[i] += residual[i];
    }

    // ── Final LayerNorm ──
    std::vector<float> x_td(T * D);
    transpose_ct(x_ct.data(), D, T, x_td.data());
    cpu_layer_norm(x_td.data(), T, D, vc.final_norm_w.data(), vc.final_norm_b.data(), 1e-6f);
    // x_td is now (T, D) = backbone output

    // ── ISTFTHead: Linear(512, 1026) → split mag/phase → iSTFT ──
    int n_fft = hp.voc_n_fft;    // 1024
    int hop = hp.voc_hop_length; // 256
    int n_freqs = n_fft / 2 + 1; // 513
    int out_dim = n_fft + 2;     // 1026

    // Linear projection
    std::vector<float> head_out(T * out_dim);
    f5_linear(x_td.data(), vc.head_w.data(), vc.head_b.data(), head_out.data(), T, D, out_dim);

    // Split into magnitude and phase, each (T, 513)
    // head_out is (T, 1026) row-major. First 513 = magnitude, last 513 = phase.
    // mag = exp(mag), clip to max 100
    // S = mag * (cos(phase) + i*sin(phase))  → complex (T, 513)
    // Then iSTFT.

    // Transpose to (n_freqs, T) for ISTFT: S_real and S_imag
    std::vector<float> S_real(n_freqs * T), S_imag(n_freqs * T);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freqs; f++) {
            float mag = expf(head_out[t * out_dim + f]);
            if (mag > 100.0f)
                mag = 100.0f;
            float phase = head_out[t * out_dim + n_freqs + f];
            S_real[f * T + t] = mag * cosf(phase);
            S_imag[f * T + t] = mag * sinf(phase);
        }
    }

    // ── ISTFT (custom "same" padding) ──
    // n_fft=1024, hop=256, win_length=1024
    int pad = (n_fft - hop) / 2; // 384
    int output_size = (T - 1) * hop + n_fft;

    // Hann window
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)n_fft));
    }

    // IRFFT + window + overlap-add
    std::vector<float> audio(output_size, 0.0f);
    std::vector<float> window_env(output_size, 0.0f);

    // For each time frame, compute IRFFT and overlap-add
    std::vector<float> frame_real(n_fft), frame_imag(n_fft);
    for (int t = 0; t < T; t++) {
        // Build full complex spectrum for IRFFT: n_fft complex values
        // We have n_freqs = n_fft/2+1 positive frequency bins.
        // Negative frequencies are conjugate of positive (real signal).
        for (int f = 0; f < n_freqs; f++) {
            frame_real[f] = S_real[f * T + t];
            frame_imag[f] = S_imag[f * T + t];
        }
        // Mirror: for f = n_fft/2+1 .. n_fft-1, X[f] = conj(X[n_fft-f])
        for (int f = n_freqs; f < n_fft; f++) {
            int mirror = n_fft - f;
            frame_real[f] = frame_real[mirror];
            frame_imag[f] = -frame_imag[mirror];
        }

        // IFFT via shared radix-2 (sign=+1 for inverse)
        int N = n_fft;
        // fft_radix2 is in-place, so copy into frame arrays
        fft_radix2(frame_real.data(), frame_imag.data(), N, +1);
        // Divide by N and apply window + overlap-add
        int out_start = t * hop;
        float inv_n = 1.0f / (float)N;
        for (int n = 0; n < N; n++) {
            float sample = frame_real[n] * inv_n; // real part only
            sample *= hann[n];
            audio[out_start + n] += sample;
            window_env[out_start + n] += hann[n] * hann[n];
        }
    }

    // Normalize by window envelope and trim padding
    int trimmed_len = output_size - 2 * pad;
    if (trimmed_len <= 0)
        return {};
    std::vector<float> result(trimmed_len);
    for (int i = 0; i < trimmed_len; i++) {
        float env = window_env[pad + i];
        result[i] = (env > 1e-11f) ? audio[pad + i] / env : 0.0f;
    }

    if (ctx->verbosity >= 1) {
        double voc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - voc_t0).count();
        fprintf(stderr, "f5_tts: vocos_decode produced %d samples in %.1f ms\n", trimmed_len, voc_ms);
    }
    return result;
}

// ── EPSS timestep schedule ───────────────────────────────────────

static std::vector<float> get_epss_timesteps(int n_steps) {
    float dt = 1.0f / 32.0f;
    // Predefined non-uniform timestep schedules
    if (n_steps == 32) {
        // Uniform for 32 steps
        std::vector<float> t(33);
        for (int i = 0; i <= 32; i++)
            t[i] = (float)i / 32.0f;
        return t;
    }
    // Known schedules from F5-TTS
    std::vector<int> steps;
    switch (n_steps) {
    case 5:
        steps = {0, 2, 4, 8, 16, 32};
        break;
    case 6:
        steps = {0, 2, 4, 6, 8, 16, 32};
        break;
    case 7:
        steps = {0, 2, 4, 6, 8, 16, 24, 32};
        break;
    case 10:
        steps = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
        break;
    case 12:
        steps = {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
        break;
    case 16:
        steps = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
        break;
    default: {
        // Linear fallback
        std::vector<float> t(n_steps + 1);
        for (int i = 0; i <= n_steps; i++)
            t[i] = (float)i / (float)n_steps;
        return t;
    }
    }
    std::vector<float> t(steps.size());
    for (size_t i = 0; i < steps.size(); i++)
        t[i] = dt * (float)steps[i];
    return t;
}

// ── ODE Euler solver ─────────────────────────────────────────────

static std::vector<float> euler_solve(f5_tts_context* ctx,
                                      const std::vector<float>& cond,            // (T, mel_dim)
                                      const std::vector<float>& text_emb,        // (T, text_dim)
                                      const std::vector<float>& text_emb_uncond, // (T, text_dim) zeroed
                                      int T, int mel_dim, int text_dim) {
    const auto& hp = ctx->hp;

    // Timestep schedule
    std::vector<float> t_schedule = get_epss_timesteps(ctx->ode_steps);

    // Apply sway sampling coefficient
    if (ctx->sway_coef != 0.0f) {
        for (auto& t : t_schedule) {
            t = t + ctx->sway_coef * (cosf((float)M_PI / 2.0f * t) - 1.0f + t);
        }
    }

    int n_steps = (int)t_schedule.size() - 1;

    // Interval-CFG (opt-in, APPROXIMATE — mirrors OMNIVOICE_CFG_INTERVAL): recompute
    // the uncond DiT forward only every K ODE steps and reuse the cached uncond
    // velocity in between; the cond forward stays fresh every step; the FIRST and
    // LAST step always recompute. This uses a slightly stale uncond, so it CHANGES
    // the output and stays gated OFF by default (K=1 = exact). It uses the separate
    // 2-forward path (which is already the default; the batched F5_BATCH_CFG path
    // fuses cond+uncond in one B=2 graph, nothing to skip — so interval overrides it).
    // Only active when K>1 && CFG is on, so at the default the legacy paths below are
    // byte-for-byte unchanged. Gated CRISPASR_F5_CFG_INTERVAL.
    const int cfg_interval = [] {
        const char* e = std::getenv("CRISPASR_F5_CFG_INTERVAL");
        const int k = e ? atoi(e) : 1;
        return k < 1 ? 1 : k;
    }();
    const bool interval_on = cfg_interval > 1 && ctx->cfg_strength >= 1e-5f;
    std::vector<float> v_uncond_cache; // last computed uncond velocity [T*mel_dim]; reused between recomputes
    if (interval_on && ctx->verbosity >= 1)
        fprintf(stderr, "f5_tts: interval-CFG K=%d (uncond recomputed every %d ODE steps; first+last always)\n",
                cfg_interval, cfg_interval);
    // Footgun guard (#294): interval-CFG reuses a stale uncond velocity between
    // recomputes. That is fine at the default 32 / 16 steps, but stacking it on an
    // aggressive EPSS low-step schedule leaves too few recomputes over a highly
    // non-uniform schedule and the output degrades to noise (MEASURED: 7 steps +
    // K=2 → unintelligible, while 16 + K=2 is clean). Warn — don't block; the
    // caller may have a reason. Threshold picked from the 7-vs-16 A/B.
    if (interval_on && n_steps < 16)
        fprintf(stderr,
                "f5_tts: WARNING interval-CFG (CRISPASR_F5_CFG_INTERVAL=%d) with only %d ODE steps degrades "
                "quality — use one lever or the other (low --tts-steps OR interval CFG), not both.\n",
                cfg_interval, n_steps);

    // DiTReducio-style temporal skipping (opt-in, APPROXIMATE — #294). Consecutive
    // ODE steps produce near-identical DiT velocities, so recompute the full step
    // velocity (BOTH CFG arms) only every K steps + first/last, and reuse the cached
    // velocity in between — skipping the entire DiT forward on reuse steps (unlike
    // interval-CFG, which only skips the uncond arm). ~K× fewer forwards at stride K.
    // Same stale-reuse risk as interval-CFG: fine near the default step count, breaks
    // at aggressive low NFE. Default OFF (K=1 = exact, legacy path byte-identical).
    const int dit_skip = [] {
        const char* e = std::getenv("CRISPASR_F5_DIT_SKIP");
        const int k = e ? atoi(e) : 1;
        return k < 1 ? 1 : k;
    }();
    const bool dit_skip_on = dit_skip > 1;
    std::vector<float> v_step_cache; // last full step velocity [T*mel_dim], reused on skip steps
    if (dit_skip_on && ctx->verbosity >= 1)
        fprintf(stderr, "f5_tts: DiT temporal-skip K=%d (full velocity recomputed every %d steps; first+last always)\n",
                dit_skip, dit_skip);
    if (dit_skip_on && n_steps < 16)
        fprintf(stderr,
                "f5_tts: WARNING DiT temporal-skip (CRISPASR_F5_DIT_SKIP=%d) with only %d ODE steps degrades "
                "quality — pair it with a higher --tts-steps, not an aggressive low count.\n",
                dit_skip, n_steps);

    // Initial noise y0 ~ N(0, 1)
    std::vector<float> x(T * mel_dim);
    if (!ctx->ref_init_noise.empty() && (int)ctx->ref_init_noise.size() == T * mel_dim) {
        // Use injected reference noise for diff validation
        x = ctx->ref_init_noise;
    } else {
        std::mt19937 rng(ctx->seed ? ctx->seed : std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : x)
            v = dist(rng);
    }

    dump_stage(ctx, "ode_step_0", x.data(), x.size());

    // Euler integration
    for (int step = 0; step < n_steps; step++) {
        float t_val = t_schedule[step];
        float dt = t_schedule[step + 1] - t_val;

        // Compute time embedding for this step
        auto time_emb = compute_time_embed(ctx, t_val);
        if (time_emb.empty())
            return {};

        if (step == 0) {
            dump_stage(ctx, "time_embed", time_emb.data(), time_emb.size());
        }

        // DiTReducio temporal skip: on a reuse step, apply the cached full velocity
        // and skip both DiT forwards entirely.
        if (dit_skip_on && step != 0 && step != n_steps - 1 && (step % dit_skip) != 0 && !v_step_cache.empty()) {
            for (size_t i = 0; i < x.size(); i++)
                x[i] += v_step_cache[i] * dt;
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "ode_step_%d", step + 1);
            dump_stage(ctx, lbl, x.data(), x.size());
            continue;
        }

        if (ctx->cfg_strength < 1e-5f) {
            // No CFG: single forward pass
            auto velocity = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb.data(), text_dim,
                                        time_emb.data(), false, false, step);
            if (velocity.empty())
                return {};

            for (size_t i = 0; i < x.size(); i++) {
                x[i] += velocity[i] * dt;
            }
            if (dit_skip_on)
                v_step_cache.assign(velocity.begin(), velocity.end());
        } else {
            // CFG: conditioned + unconditioned forward. The DiT is identical
            // per arm (the cond/uncond difference is baked into `hidden` on the
            // host), so with F5_BATCH_CFG the two arms run as one B=2 DiT graph.
            // MEASURED (M1 Metal, 8 steps): output identical to sequential
            // (corr 1.00000) but NO speedup — slightly slower. The DiT is
            // compute/bandwidth-bound, not dispatch-bound, so B=2 just does 2×
            // work per dispatch. Kept opt-in + off by default; the real lever
            // is DiT compute efficiency (F32→F16 activations). See PLAN §232.
            std::vector<float> v_cond, v_uncond;
            const float* v_unc_ptr = nullptr;
            if (interval_on) {
                // Cond fresh every step; uncond recomputed on the first + last step
                // and every K-th step, otherwise reused from the cache (the approximation).
                v_cond = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb.data(), text_dim, time_emb.data(),
                                     false, false, step);
                if (v_cond.empty())
                    return {};
                const bool recompute_unc =
                    (step == 0) || (step == n_steps - 1) || ((step % cfg_interval) == 0) || v_uncond_cache.empty();
                if (recompute_unc) {
                    v_uncond = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb_uncond.data(), text_dim,
                                           time_emb.data(), true, true, step);
                    if (v_uncond.empty())
                        return {};
                    v_uncond_cache = v_uncond;
                }
                v_unc_ptr = v_uncond_cache.data();
            } else if (f5_batch_cfg_enabled()) {
                auto h_cond = f5_compute_hidden(ctx, x.data(), T, mel_dim, cond.data(), text_emb.data(), text_dim,
                                                false, false, step);
                auto h_uncond = f5_compute_hidden(ctx, x.data(), T, mel_dim, cond.data(), text_emb_uncond.data(),
                                                  text_dim, true, true, step);
                if (h_cond.empty() || h_uncond.empty())
                    return {};
                const int dim = ctx->hp.dim;
                std::vector<float> h_stack((size_t)T * dim * 2);
                std::copy(h_cond.begin(), h_cond.end(), h_stack.begin());
                std::copy(h_uncond.begin(), h_uncond.end(), h_stack.begin() + (size_t)T * dim);
                auto v = f5_dit_run_batched(ctx, h_stack.data(), T, 2, time_emb.data());
                if (v.empty())
                    return {};
                const size_t arm = (size_t)T * mel_dim;
                v_cond.assign(v.begin(), v.begin() + arm);
                v_uncond.assign(v.begin() + arm, v.begin() + 2 * arm);
                v_unc_ptr = v_uncond.data();
            } else {
                v_cond = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb.data(), text_dim, time_emb.data(),
                                     false, false, step);
                v_uncond = dit_forward(ctx, x.data(), T, mel_dim, cond.data(), text_emb_uncond.data(), text_dim,
                                       time_emb.data(), true, true, step);
                if (v_cond.empty() || v_uncond.empty())
                    return {};
                v_unc_ptr = v_uncond.data();
            }

            // CFG: v = v_cond + cfg * (v_cond - v_uncond)
            float cfg = ctx->cfg_strength;
            if (dit_skip_on)
                v_step_cache.resize(x.size());
            for (size_t i = 0; i < x.size(); i++) {
                float v = v_cond[i] + cfg * (v_cond[i] - v_unc_ptr[i]);
                x[i] += v * dt;
                if (dit_skip_on)
                    v_step_cache[i] = v;
            }
        }

        // Dump selected ODE steps
        char label[64];
        snprintf(label, sizeof(label), "ode_step_%d", step + 1);
        dump_stage(ctx, label, x.data(), x.size());

        if (ctx->verbosity >= 2) {
            fprintf(stderr, "  ODE step %d/%d  t=%.4f  dt=%.4f\n", step + 1, n_steps, t_val, dt);
        }
    }

    // Apply conditioning mask: replace ref positions with original cond
    // x = where(cond_mask, cond, x)
    // cond_mask: True for ref positions (where cond != 0)
    for (int t = 0; t < T; t++) {
        bool is_ref = false;
        for (int d = 0; d < mel_dim && !is_ref; d++) {
            if (fabsf(cond[t * mel_dim + d]) > 1e-10f)
                is_ref = true;
        }
        if (is_ref) {
            for (int d = 0; d < mel_dim; d++) {
                x[t * mel_dim + d] = cond[t * mel_dim + d];
            }
        }
    }

    return x;
}

// ── Weight loading ───────────────────────────────────────────────

static bool load_weights(f5_tts_context* ctx, const char* path) {
    auto& hp = ctx->hp;
    auto& w = ctx->w;

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    hp.dim = core_gguf::kv_i32(meta, "f5.dim", hp.dim);
    hp.depth = core_gguf::kv_i32(meta, "f5.depth", hp.depth);
    hp.heads = core_gguf::kv_i32(meta, "f5.heads", hp.heads);
    hp.dim_head = core_gguf::kv_i32(meta, "f5.dim_head", hp.dim_head);
    hp.ff_mult = core_gguf::kv_i32(meta, "f5.ff_mult", hp.ff_mult);
    hp.text_dim = core_gguf::kv_i32(meta, "f5.text_dim", hp.text_dim);
    hp.text_num_embeds = core_gguf::kv_i32(meta, "f5.text_num_embeds", hp.text_num_embeds);
    hp.conv_layers = core_gguf::kv_i32(meta, "f5.conv_layers", hp.conv_layers);
    hp.mel_dim = core_gguf::kv_i32(meta, "f5.mel_dim", hp.mel_dim);
    hp.sample_rate = core_gguf::kv_i32(meta, "f5.sample_rate", hp.sample_rate);
    hp.hop_length = core_gguf::kv_i32(meta, "f5.hop_length", hp.hop_length);
    hp.win_length = core_gguf::kv_i32(meta, "f5.win_length", hp.win_length);
    hp.n_fft = core_gguf::kv_i32(meta, "f5.n_fft", hp.n_fft);
    hp.ode_steps = core_gguf::kv_i32(meta, "f5.ode_steps", hp.ode_steps);
    hp.cfg_strength = core_gguf::kv_f32(meta, "f5.cfg_strength", hp.cfg_strength);
    hp.sway_coef = core_gguf::kv_f32(meta, "f5.sway_sampling_coef", hp.sway_coef);
    hp.voc_dim = core_gguf::kv_i32(meta, "f5.voc_dim", hp.voc_dim);
    hp.voc_intermediate_dim = core_gguf::kv_i32(meta, "f5.voc_intermediate_dim", hp.voc_intermediate_dim);
    hp.voc_num_layers = core_gguf::kv_i32(meta, "f5.voc_num_layers", hp.voc_num_layers);
    hp.voc_n_fft = core_gguf::kv_i32(meta, "f5.voc_n_fft", hp.voc_n_fft);
    hp.voc_hop_length = core_gguf::kv_i32(meta, "f5.voc_hop_length", hp.voc_hop_length);
    // #387 Raon deltas
    hp.vocoder = core_gguf::kv_str(meta, "f5.vocoder", hp.vocoder.c_str());
    hp.mel_spec_type = core_gguf::kv_str(meta, "f5.mel_spec_type", hp.mel_spec_type.c_str());
    hp.mel_center = core_gguf::kv_bool(meta, "f5.mel_center", hp.mel_center);

    // Vocab
    auto vocab_chars = core_gguf::kv_str_array(meta, "f5.vocab");
    for (size_t i = 0; i < vocab_chars.size(); i++) {
        ctx->vocab.chars.push_back(vocab_chars[i]);
        ctx->vocab.char_to_idx[vocab_chars[i]] = (int)i;
    }

    core_gguf::free_metadata(meta);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: dim=%d depth=%d heads=%d ff_mult=%d text_dim=%d mel=%d sr=%d\n", hp.dim, hp.depth,
                hp.heads, hp.ff_mult, hp.text_dim, hp.mel_dim, hp.sample_rate);
        fprintf(stderr, "f5_tts: vocab=%zu voc_dim=%d voc_layers=%d\n", ctx->vocab.chars.size(), hp.voc_dim,
                hp.voc_num_layers);
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "f5_tts", wl))
        return false;
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* { return core_gguf::require(ts, name, "f5_tts"); };
    auto try_get = [&](const char* name) -> ggml_tensor* { return core_gguf::try_get(ts, name); };

    // Text embedding
    w.text_emb_weight = get("f5.text_emb.weight");

    // Text ConvNeXtV2 blocks
    w.text_blocks.resize(hp.conv_layers);
    for (int i = 0; i < hp.conv_layers; i++) {
        char buf[128];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "f5.text_blk.%d.%s", i, suffix);
            return get(buf);
        };
        w.text_blocks[i] = {
            g("dw.weight"),  g("dw.bias"),        g("norm.weight"),  g("norm.bias"), g("pw_up.weight"),
            g("pw_up.bias"), g("pw_down.weight"), g("pw_down.bias"), g("grn_gamma"), g("grn_beta"),
        };
    }

    // Time embedding
    w.time_mlp_0_weight = get("f5.time_mlp_0.weight");
    w.time_mlp_0_bias = get("f5.time_mlp_0.bias");
    w.time_mlp_1_weight = get("f5.time_mlp_1.weight");
    w.time_mlp_1_bias = get("f5.time_mlp_1.bias");

    // Input embedding
    w.input_proj_weight = get("f5.input_proj.weight");
    w.input_proj_bias = get("f5.input_proj.bias");
    w.conv_pos_0_weight = get("f5.conv_pos_0.weight");
    w.conv_pos_0_bias = get("f5.conv_pos_0.bias");
    w.conv_pos_1_weight = get("f5.conv_pos_1.weight");
    w.conv_pos_1_bias = get("f5.conv_pos_1.bias");

    // DiT blocks
    w.dit_blocks.resize(hp.depth);
    for (int i = 0; i < hp.depth; i++) {
        char buf[128];
        auto g = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "f5.blk.%d.%s", i, suffix);
            return get(buf);
        };
        w.dit_blocks[i] = {
            g("adaln.weight"),  g("adaln.bias"),    g("attn_q.weight"),   g("attn_q.bias"),   g("attn_k.weight"),
            g("attn_k.bias"),   g("attn_v.weight"), g("attn_v.bias"),     g("attn_o.weight"), g("attn_o.bias"),
            g("ffn_up.weight"), g("ffn_up.bias"),   g("ffn_down.weight"), g("ffn_down.bias"),
        };
    }

    // Final norm + proj
    w.final_adaln_weight = get("f5.final_adaln.weight");
    w.final_adaln_bias = get("f5.final_adaln.bias");
    w.final_proj_weight = get("f5.final_proj.weight");
    w.final_proj_bias = get("f5.final_proj.bias");

    // Rotary
    w.rotary_inv_freq = get("f5.rotary_inv_freq");

    // Vocos (stock F5) — only when this GGUF actually carries Vocos weights.
    if (hp.vocoder == "vocos") {
        w.voc_embed_weight = get("voc.embed.weight");
        w.voc_embed_bias = get("voc.embed.bias");
        w.voc_norm_weight = get("voc.norm.weight");
        w.voc_norm_bias = get("voc.norm.bias");
        w.voc_blocks.resize(hp.voc_num_layers);
        for (int i = 0; i < hp.voc_num_layers; i++) {
            char buf[128];
            auto g = [&](const char* suffix) -> ggml_tensor* {
                snprintf(buf, sizeof(buf), "voc.blk.%d.%s", i, suffix);
                return get(buf);
            };
            w.voc_blocks[i] = {
                g("dw.weight"),  g("dw.bias"),        g("norm.weight"),  g("norm.bias"),   g("pw_up.weight"),
                g("pw_up.bias"), g("pw_down.weight"), g("pw_down.bias"), g("layer_scale"),
            };
        }
        w.voc_final_norm_weight = get("voc.final_norm.weight");
        w.voc_final_norm_bias = get("voc.final_norm.bias");
        w.voc_head_weight = get("voc.head.weight");
        w.voc_head_bias = get("voc.head.bias");
    } else if (hp.vocoder == "hifigan") {
        // #387: pull every voc.* HiFi-GAN tensor straight into the CPU cache
        // (weight-norm already fused in the converter). The decoder looks them
        // up by name; the arch (rates/kernels) is fixed by sbhifigan16k.
        for (const auto& kv : ts) {
            if (kv.first.rfind("voc.", 0) == 0)
                read_tensor_f32(kv.second, ctx->hifigan_w[kv.first]);
        }
        // Shipped slaney mel filterbank + Hann window.
        read_tensor_f32(get("f5.mel_fb"), ctx->mel_fb);
        read_tensor_f32(get("f5.mel_window"), ctx->mel_window);
    }

    return true;
}

// ── Public API ───────────────────────────────────────────────────

struct f5_tts_params f5_tts_default_params(void) {
    return {
        /* n_threads     */ 4,
        /* verbosity     */ 1,
        /* use_gpu       */ false,
        /* seed          */ 42,
        /* ode_steps     */ 0,    // 0 = use model default
        /* cfg_strength  */ 0.0f, // 0 = use model default
        /* sway_coef     */ 0.0f, // 0 = use model default
        /* speed         */ 1.0f,
    };
}

struct f5_tts_context* f5_tts_init_from_file(const char* path_model, struct f5_tts_params params) {
    auto* ctx = new f5_tts_context();
    ctx->verbosity = params.verbosity;
    ctx->n_threads = params.n_threads;
    ctx->seed = params.seed;
    ctx->speed = params.speed;

    // Initialize backends
    ctx->backend_cpu = core_cpu_backend::init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "f5_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    core_cpu_backend::set_n_threads(ctx->backend_cpu, params.n_threads);

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (params.verbosity >= 1 && ctx->backend != ctx->backend_cpu) {
        fprintf(stderr, "f5_tts: using GPU backend: %s\n", ggml_backend_name(ctx->backend));
    }

    // Load weights
    if (!load_weights(ctx, path_model)) {
        fprintf(stderr, "f5_tts: failed to load model: %s\n", path_model);
        f5_tts_free(ctx);
        return nullptr;
    }

    // Pre-dequantize weights once at load time to avoid per-synthesis reads.
    {
        const auto& w = ctx->w;

        // §184: input-embedding weights (hot: 64× per synthesis)
        auto& ec = ctx->emb_cache;
        read_tensor_f32(w.input_proj_weight, ec.input_proj_w);
        read_tensor_f32(w.input_proj_bias, ec.input_proj_b);
        read_tensor_f32(w.conv_pos_0_weight, ec.conv_pos_0_w);
        read_tensor_f32(w.conv_pos_0_bias, ec.conv_pos_0_b);
        read_tensor_f32(w.conv_pos_1_weight, ec.conv_pos_1_w);
        read_tensor_f32(w.conv_pos_1_bias, ec.conv_pos_1_b);

        // §185: text ConvNeXtV2 weights (1× per synthesis, ~21 MB F32)
        {
            auto& tc = ctx->text_cache;
            read_tensor_f32(w.text_emb_weight, tc.emb);
            tc.blocks.resize(ctx->hp.conv_layers);
            for (int b = 0; b < ctx->hp.conv_layers; b++) {
                const auto& blk = w.text_blocks[b];
                auto& bc = tc.blocks[b];
                read_tensor_f32(blk.dw_weight, bc.dw_w);
                read_tensor_f32(blk.dw_bias, bc.dw_b);
                read_tensor_f32(blk.norm_weight, bc.norm_w);
                read_tensor_f32(blk.norm_bias, bc.norm_b);
                read_tensor_f32(blk.pw_up_weight, bc.pw_up_w);
                read_tensor_f32(blk.pw_up_bias, bc.pw_up_b);
                read_tensor_f32(blk.pw_down_weight, bc.pw_down_w);
                read_tensor_f32(blk.pw_down_bias, bc.pw_down_b);
                read_tensor_f32(blk.grn_gamma, bc.grn_g);
                read_tensor_f32(blk.grn_beta, bc.grn_b);
            }
        }

        // §185: Vocos vocoder weights (1× per synthesis, ~54 MB F32)
        if (ctx->hp.vocoder == "vocos") {
            auto& vc = ctx->voc_cache;
            read_tensor_f32(w.voc_embed_weight, vc.embed_w);
            read_tensor_f32(w.voc_embed_bias, vc.embed_b);
            read_tensor_f32(w.voc_norm_weight, vc.norm_w);
            read_tensor_f32(w.voc_norm_bias, vc.norm_b);
            vc.blocks.resize(ctx->hp.voc_num_layers);
            for (int b = 0; b < ctx->hp.voc_num_layers; b++) {
                const auto& blk = w.voc_blocks[b];
                auto& bc = vc.blocks[b];
                read_tensor_f32(blk.dw_weight, bc.dw_w);
                read_tensor_f32(blk.dw_bias, bc.dw_b);
                read_tensor_f32(blk.norm_weight, bc.norm_w);
                read_tensor_f32(blk.norm_bias, bc.norm_b);
                read_tensor_f32(blk.pw_up_weight, bc.pw_up_w);
                read_tensor_f32(blk.pw_up_bias, bc.pw_up_b);
                read_tensor_f32(blk.pw_down_weight, bc.pw_down_w);
                read_tensor_f32(blk.pw_down_bias, bc.pw_down_b);
                read_tensor_f32(blk.layer_scale, bc.layer_scale);
            }
            read_tensor_f32(w.voc_final_norm_weight, vc.final_norm_w);
            read_tensor_f32(w.voc_final_norm_bias, vc.final_norm_b);
            read_tensor_f32(w.voc_head_weight, vc.head_w);
            read_tensor_f32(w.voc_head_bias, vc.head_b);
        }
    }

    // Create backend scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
        if (!ctx->sched) {
            fprintf(stderr, "f5_tts: failed to create backend scheduler\n");
            f5_tts_free(ctx);
            return nullptr;
        }
    }

    // Apply params (0 = use model default)
    ctx->ode_steps = params.ode_steps > 0 ? params.ode_steps : ctx->hp.ode_steps;
    ctx->cfg_strength = params.cfg_strength > 0.0f ? params.cfg_strength : ctx->hp.cfg_strength;
    ctx->sway_coef = params.sway_coef != 0.0f ? params.sway_coef : ctx->hp.sway_coef;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: loaded %s  ode=%d cfg=%.1f sway=%.1f seed=%d\n", path_model, ctx->ode_steps,
                ctx->cfg_strength, ctx->sway_coef, ctx->seed);
    }

    return ctx;
}

void f5_tts_free(struct f5_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->w_buf)
        core_gguf::release_weight_buffer(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// Port of upstream F5-TTS preprocess_ref_audio_text (audio side): strip silence
// and clip the reference to a max length before it drives the duration estimate.
// Upstream splits the ref on >=1 s silences, keeps the leading speech, and clips
// to 12 s (src/f5_tts/infer/utils_infer.py). We approximate that here: trim
// leading/trailing silence, collapse internal silences longer than ~1 s down to
// ~0.2 s, then clip to `max_sec`. Bounding ref_mel_T to actual speech keeps the
// `ref_T / ref_text_len` speech-rate estimate honest (a ref padded with silence
// otherwise inflates the estimate). Returns the processed PCM.
static std::vector<float> f5_preprocess_ref_audio(const float* pcm, int n, int sr, float max_sec, bool trim_silence) {
    std::vector<float> out(pcm, pcm + n);
    if (trim_silence && n > 0) {
        const int win = std::max(1, sr / 100); // 10 ms analysis window
        const int n_win = n / win;
        if (n_win > 2) {
            std::vector<float> rms(n_win, 0.0f);
            float peak = 1e-9f;
            for (int w = 0; w < n_win; ++w) {
                double s = 0;
                for (int i = 0; i < win; ++i) {
                    float x = pcm[w * win + i];
                    s += (double)x * x;
                }
                rms[w] = (float)std::sqrt(s / win);
                peak = std::max(peak, rms[w]);
            }
            // -50 dBFS full-scale silence gate, but never above 5 % of the ref's
            // own peak so a quiet recording isn't wiped out entirely.
            const float thr = std::min(0.00316f, peak * 0.05f);
            auto silent = [&](int w) { return rms[w] < thr; };
            const int min_sil = std::max(1, sr / win);        // 1 s of windows
            const int keep_pad = std::max(1, (sr / win) / 5); // 0.2 s kept around gaps
            int first = 0;
            while (first < n_win && silent(first))
                ++first;
            int last = n_win - 1;
            while (last >= 0 && silent(last))
                --last;
            if (first <= last) {
                std::vector<float> keep;
                keep.reserve((size_t)n);
                int w = first;
                while (w <= last) {
                    if (!silent(w)) {
                        int s = w;
                        while (w <= last && !silent(w))
                            ++w;
                        keep.insert(keep.end(), pcm + (size_t)s * win, pcm + (size_t)w * win);
                    } else {
                        int s = w;
                        while (w <= last && silent(w))
                            ++w;
                        int len = w - s;                            // internal silent run
                        int kw = (len >= min_sil) ? keep_pad : len; // collapse long gaps only
                        keep.insert(keep.end(), pcm + (size_t)s * win, pcm + (size_t)(s + kw) * win);
                    }
                }
                if (!keep.empty())
                    out.swap(keep);
            }
        }
    }
    if (max_sec > 0.0f) {
        size_t maxn = (size_t)(max_sec * (float)sr);
        if (out.size() > maxn)
            out.resize(maxn);
    }
    return out;
}

int f5_tts_set_reference(struct f5_tts_context* ctx, const float* pcm_24k, int n_samples, const char* ref_text) {
    if (!ctx || !pcm_24k || n_samples <= 0)
        return -1;

    // Reference preprocessing (upstream parity): silence-strip + clip. Gated so
    // it can be turned off for byte-exact comparison against pre-#294 behavior.
    // CRISPASR_F5_REF_MAX_SEC: clip length in seconds (default 12, matches
    //   upstream; 0 disables the clip). CRISPASR_F5_REF_TRIM_SILENCE: 0 disables
    //   the silence strip (default on).
    const char* max_env = crispasr_env::get("CRISPASR_F5_REF_MAX_SEC");
    float ref_max_sec = max_env ? (float)atof(max_env) : 12.0f;
    const char* trim_env = crispasr_env::get("CRISPASR_F5_REF_TRIM_SILENCE");
    bool ref_trim = !(trim_env && std::strcmp(trim_env, "0") == 0);
    std::vector<float> ref_pcm =
        f5_preprocess_ref_audio(pcm_24k, n_samples, ctx->hp.sample_rate, ref_max_sec, ref_trim);
    if (ctx->verbosity >= 1 && (int)ref_pcm.size() != n_samples) {
        fprintf(stderr, "f5_tts: ref preprocess %d -> %zu samples (%.2f -> %.2f s)\n", n_samples, ref_pcm.size(),
                (float)n_samples / (float)ctx->hp.sample_rate, (float)ref_pcm.size() / (float)ctx->hp.sample_rate);
    }
    pcm_24k = ref_pcm.data();
    n_samples = (int)ref_pcm.size();

    // Compute mel spectrogram of reference audio. sbhifigan16k (#387) uses the
    // shipped slaney filterbank + window and STFT center=false at 16 kHz.
    int T_ref;
    const bool sbmel = (ctx->hp.mel_spec_type == "sbhifigan16k");
    ctx->ref_mel = compute_mel_spectrogram(
        pcm_24k, n_samples, ctx->hp.n_fft, ctx->hp.hop_length, ctx->hp.win_length, ctx->hp.mel_dim, T_ref,
        (float)ctx->hp.sample_rate, sbmel ? &ctx->mel_fb : nullptr, sbmel ? &ctx->mel_window : nullptr,
        sbmel ? ctx->hp.mel_center : true);

    // If mel computation not yet implemented, allow setting ref_mel directly
    // via the diff harness (which provides it as a GGUF tensor)
    ctx->ref_mel_T = T_ref;
    ctx->ref_text = ref_text ? ref_text : "";

    // #387 mel-parity probe: dump the computed reference mel (T, mel_dim
    // row-major) as raw f32 and exit, so the sbhifigan slaney/center=false
    // front-end can be diffed against the reference ref_mel without paying
    // for the (minutes-on-CPU) DiT ODE loop that follows.
    if (const char* dp = crispasr_env::get("CRISPASR_F5_DUMP_REFMEL")) {
        FILE* f = fopen(dp, "wb");
        if (f) {
            fwrite(ctx->ref_mel.data(), sizeof(float), ctx->ref_mel.size(), f);
            fclose(f);
            fprintf(stderr, "f5_tts: dumped ref_mel (T=%d, mel=%d) → %s\n", ctx->ref_mel_T, ctx->hp.mel_dim, dp);
        }
    }
    return 0;
}

int f5_tts_synthesize(struct f5_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out || !sample_rate_out)
        return 0;

    const auto& hp = ctx->hp;
    int mel_dim = hp.mel_dim;
    int text_dim = hp.text_dim;

    // #387 vocoder-parity probe: CRISPASR_F5_VOCODE_MEL=<file>[:T] reads a raw
    // f32 mel (T × mel_dim, row-major), runs ONLY the vocoder, and returns the
    // audio — bypassing the DiT so the CPU HiFi-GAN can be diffed against the
    // reference vocoder output without the minutes-long ODE loop.
    if (const char* mp = crispasr_env::get("CRISPASR_F5_VOCODE_MEL")) {
        FILE* f = fopen(mp, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long bytes = ftell(f);
            fseek(f, 0, SEEK_SET);
            int n = (int)(bytes / sizeof(float));
            int T = n / mel_dim;
            std::vector<float> mel(n);
            size_t rd = fread(mel.data(), sizeof(float), n, f);
            fclose(f);
            (void)rd;
            auto audio = (hp.vocoder == "hifigan") ? hifigan_decode(ctx, mel.data(), T, mel_dim)
                                                   : vocos_decode(ctx, mel.data(), T, mel_dim);
            *pcm_out = (float*)malloc(audio.size() * sizeof(float));
            memcpy(*pcm_out, audio.data(), audio.size() * sizeof(float));
            *sample_rate_out = hp.sample_rate;
            fprintf(stderr, "f5_tts: VOCODE_MEL probe T=%d mel=%d → %zu samples\n", T, mel_dim, audio.size());
            return (int)audio.size();
        }
    }

    // ── Text preparation ──
    std::string ref_text = ctx->ref_text;
    auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (!ref_text.empty() && !ends_with(ref_text, ". ") && !ends_with(ref_text, "。")) {
        if (ends_with(ref_text, "."))
            ref_text += " ";
        else
            ref_text += ". ";
    }
    std::string full_text = ref_text + text;

    // Convert to the list_str_to_idx units: for English each element is a single
    // character; for Chinese each element is a full pinyin syllable (e.g.
    // "zhong1") that is a SINGLE vocab entry. #294: map each element DIRECTLY to
    // its vocab id — concatenating the list and re-splitting per UTF-8 byte (the
    // old path) shatters a multi-char pinyin token into individual letters
    // ("zhong1" → z,h,o,n,g,1), which produced garbled Chinese. English is
    // unaffected (its elements are already single characters).
    auto pinyin_chars = convert_to_pinyin(full_text);
    std::vector<int32_t> tokens;
    tokens.reserve(pinyin_chars.size());
    for (const auto& unit : pinyin_chars) {
        auto it = ctx->vocab.char_to_idx.find(unit);
        tokens.push_back(it != ctx->vocab.char_to_idx.end() ? it->second : 0);
    }

    // ── Duration estimation ──
    // The formula estimates speech rate from (ref_T / ref_text_len) mel frames
    // per character, then applies it to gen_text. When ref_text is empty we
    // have no transcript to derive the rate from, so we estimate the ref
    // transcript length from audio duration at ~13 chars/sec (typical English
    // including spaces — calibrated against F5-TTS upstream defaults).
    int ref_T = ctx->ref_mel_T;
    const float mel_fps = (float)ctx->hp.sample_rate / (float)ctx->hp.hop_length; // ~93.75 (24k / hop 256)
    const float fixed_rate = mel_fps / 13.0f; // ~7.2 mel frames/char (English ~13 chars/s)
    int ref_text_len;
    if (ref_text.empty()) {
        float ref_secs = (float)ref_T / mel_fps;
        ref_text_len = std::max(1, (int)(ref_secs * 13.0f));
    } else {
        ref_text_len = (int)f5_text_len(ref_text.c_str());
    }
    int gen_text_len = (int)f5_text_len(text);
    // Per-char speech rate derived from the reference (mel frames per char).
    // #294: the guard here must be ASYMMETRIC. Under-estimating the rate makes
    // `duration` too short and TRUNCATES the generated speech (drops the tail of
    // the sentence); over-estimating only appends trailing silence, which is
    // trimmed. So a too-LOW rate is harmful and a too-HIGH rate is (almost) free.
    //
    // The upstream formula has NO clamp (tools/reference_backends/f5_tts.py:261).
    // The original symmetric clamp added here capped the rate at fixed_rate*2.5,
    // which TRUNCATED slow/expressive references: the reporter's ref implies
    // ~3x fixed_rate, so capping to 2.5x lost the end of the sentence
    // ("sometimes leaves out parts of sentences"). Fix: keep a protective LOWER
    // guard (a bad/too-long ref transcript must not collapse the rate to zero),
    // but only a very loose UPPER guard that catches a garbage near-empty
    // transcript (which would explode the duration) — not genuinely slow speech.
    float rate = (float)ref_T / (float)std::max(1, ref_text_len);
    // The clamp is an ADD-ON over the upstream formula (which has no clamp); gate
    // it so it can be switched off. CRISPASR_F5_DURATION_CLAMP=0 restores the
    // exact upstream `ref_T / ref_text_len * gen_text_len / speed` estimate.
    const char* clamp_env = crispasr_env::get("CRISPASR_F5_DURATION_CLAMP");
    bool duration_clamp = !(clamp_env && std::strcmp(clamp_env, "0") == 0);
    if (duration_clamp)
        rate = std::min(std::max(rate, fixed_rate * 0.75f), fixed_rate * 8.0f);
    int duration = ref_T + (int)(rate * (float)gen_text_len / ctx->speed);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "f5_tts: ref_T=%d duration=%d tokens=%zu text='%s'\n", ref_T, duration, tokens.size(),
                full_text.c_str());
    }

    // ── Text embedding ──
    std::vector<float> text_emb;
    {
        f5_bench_stage _b("text_embed");
        text_emb = compute_text_embed(ctx, tokens.data(), (int)tokens.size(), duration);
    }
    if (text_emb.empty())
        return 0;
    dump_stage(ctx, "text_embed", text_emb.data(), text_emb.size());

    // Unconditional text embedding (all zeros)
    std::vector<float> text_emb_uncond(duration * text_dim, 0.0f);

    // ── Conditioning (ref mel padded to duration) ──
    std::vector<float> cond(duration * mel_dim, 0.0f);
    for (int t = 0; t < ref_T && t < duration; t++) {
        for (int d = 0; d < mel_dim; d++) {
            cond[t * mel_dim + d] = ctx->ref_mel[t * mel_dim + d];
        }
    }
    dump_stage(ctx, "conditioning_input", cond.data(), cond.size());

    // Conditioning mask: where cond != 0 → use cond
    std::vector<float> step_cond(duration * mel_dim, 0.0f);
    for (int t = 0; t < ref_T; t++) {
        for (int d = 0; d < mel_dim; d++) {
            step_cond[t * mel_dim + d] = cond[t * mel_dim + d];
        }
    }

    // ── ODE solve ──
    std::vector<float> generated;
    {
        g_f5_hostembed_us = 0;
        g_f5_ditgraph_us = 0;
        f5_bench_stage _b("ode_solve");
        generated = euler_solve(ctx, step_cond, text_emb, text_emb_uncond, duration, mel_dim, text_dim);
    }
    if (f5_bench_enabled())
        fprintf(stderr, "  f5_bench: %-22s host_embed=%.1f ms  dit_graph=%.1f ms\n", "ode_split",
                g_f5_hostembed_us / 1000.0, g_f5_ditgraph_us / 1000.0);
    if (generated.empty())
        return 0;

    // ── Extract generated portion (after ref_T) ──
    int gen_T = duration - ref_T;
    if (gen_T <= 0)
        return 0;
    std::vector<float> gen_mel(gen_T * mel_dim);
    for (int t = 0; t < gen_T; t++) {
        for (int d = 0; d < mel_dim; d++) {
            gen_mel[t * mel_dim + d] = generated[(ref_T + t) * mel_dim + d];
        }
    }
    dump_stage(ctx, "vocos_input", gen_mel.data(), gen_mel.size());

    // ── Vocoder (Vocos for stock F5; HiFi-GAN for Raon sbhifigan16k, #387) ──
    f5_bench_stage _b_voc("vocoder");
    auto audio = (hp.vocoder == "hifigan") ? hifigan_decode(ctx, gen_mel.data(), gen_T, mel_dim)
                                           : vocos_decode(ctx, gen_mel.data(), gen_T, mel_dim);
    if (audio.empty()) {
        // Fallback: return empty for now, will be filled once vocos is implemented
        *pcm_out = nullptr;
        *sample_rate_out = hp.sample_rate;
        return 0;
    }

    // Copy to malloc'd buffer for caller
    int n_samples = (int)audio.size();
    float* out = (float*)malloc(n_samples * sizeof(float));
    memcpy(out, audio.data(), n_samples * sizeof(float));
    *pcm_out = out;
    *sample_rate_out = hp.sample_rate;
    return n_samples;
}

void f5_tts_set_seed(struct f5_tts_context* ctx, int seed) {
    if (ctx)
        ctx->seed = seed;
}
void f5_tts_set_ode_steps(struct f5_tts_context* ctx, int steps) {
    if (ctx)
        ctx->ode_steps = steps;
}
void f5_tts_set_cfg_strength(struct f5_tts_context* ctx, float s) {
    if (ctx)
        ctx->cfg_strength = s;
}
void f5_tts_set_speed(struct f5_tts_context* ctx, float s) {
    if (ctx)
        ctx->speed = s;
}
int f5_tts_sample_rate(const struct f5_tts_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 24000;
}
int f5_tts_vocab_size(const struct f5_tts_context* ctx) {
    return ctx ? (int)ctx->vocab.chars.size() : 0;
}
void f5_tts_set_dump_dir(struct f5_tts_context* ctx, const char* dir) {
    if (ctx)
        ctx->dump_dir = dir ? dir : "";
}

// Test-only: inject reference initial noise for reproducible diff testing.
void f5_tts_set_init_noise(f5_tts_context* ctx, const float* noise, int n) {
    if (!ctx || !noise || n <= 0)
        return;
    ctx->ref_init_noise.assign(noise, noise + n);
}

// Test-only: inject reference mel directly (bypasses mel computation).
void f5_tts_set_ref_mel(f5_tts_context* ctx, const float* mel, int T, int mel_dim, const char* ref_text) {
    if (!ctx)
        return;
    ctx->ref_mel.assign(mel, mel + T * mel_dim);
    ctx->ref_mel_T = T;
    ctx->ref_text = ref_text ? ref_text : "";
}
