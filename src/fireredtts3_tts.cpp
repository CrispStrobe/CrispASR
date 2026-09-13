// fireredtts3_tts.cpp — FireRedTTS3 continuous-latent AR TTS backend (#377).
//
// Blueprint: github.com/FireRedTeam/FireRedTTS3 (Apache-2.0), read line by
// line — fireredtts3/llm/fireredtts3_base.py (FireRedTTS3BaseCore.generate,
// _flow_one_step, _backbone_one_step), llm/patch_encoder.py, llm/dit.py,
// llm/modules.py, llm/rotary_embedding.py, redae/redae.py, campp/*.
//
// Pipeline (per synthesis):
//   1. CAM++ x-vector (512) of the 24 kHz padded prompt (resampled to 16 k)
//   2. RedAE encode: prompt 24 kHz → 64-d 25 Hz latents (T_lat, left-padded
//      to a multiple of patch_size=4 via sample-domain left pad)
//   3. Tokenize `<|Lang|><|sot|>{ref_text}{text}<|eot|>` (Qwen2 BPE)
//   4. Prefill embeds = [spk_proj_llm(spk); text embeds; PatchEncoder(prompt
//      latents)] → Qwen3-1.7B backbone (full causal re-forward per AR step;
//      KV-cache is a later optimization, correctness first)
//   5. Per AR step: sigmoid stop head on the last hidden; DiT flow-matching
//      (10 Euler steps on a cosine t-schedule, CFG 2.0) denoises one 4-frame
//      patch conditioned on [2-patch clean history; dit_head(last 3 backbone
//      hiddens) repeat-interleaved; spk_proj_dit(spk)]; PatchEncoder maps the
//      new patch back to one LLM embedding
//   6. RedAE decode: [prompt latents; generated] → 24 kHz PCM (Qwen3 stack +
//      Vocos ISTFT head, n_fft=1920 hop=480, shipped window) → trim prompt
//
// RoPE conventions (two different ones — see rotary_embedding.py vs HF Qwen3):
//   - Backbone + RedAE Qwen3 stacks: HF NEOX half-rotation; theta 1e6
//     (backbone) / 1e4 (RedAE, transformers Qwen3Config default_theta)
//   - PatchEncoder + DiT: x_transformers adjacent-pair = GGML_ROPE_TYPE
//     NORMAL (mode 0), theta 1e4, full head dim (f5_tts.cpp precedent)
//
// Batched small-sequence stacks (PatchEncoder patches of 5 tokens, RedAE
// CLS-downsample groups of 3) are run seq-concatenated with a block-diagonal
// mask instead of a 4D batch dim (crispasr-crispembed-dev.md: sidesteps the
// batched-fused-graph alloc bug; positions tile per block).

#include "fireredtts3_tts.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "chatterbox_campplus.h"
#include "core/audio_resample.h"
#include "core/bpe.h"
#include "core/env_gate.h"
#include "core/ffn.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/istft.h"
#include "core/torch_rng.h"
#include "core/wav_reader.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Bench instrumentation (FIREREDTTS3_BENCH=1)
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("FIREREDTTS3_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct frt_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit frt_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~frt_bench_stage() {
        if (!frt_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  fireredtts3_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Model structs
// ─────────────────────────────────────────────────────────────────────────────

// One Qwen3 transformer block (backbone LLM + all three RedAE stacks).
struct frt_qwen3_block {
    ggml_tensor* q_w = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* q_norm_w = nullptr;
    ggml_tensor* k_norm_w = nullptr;
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* gate_w = nullptr;
    ggml_tensor* up_w = nullptr;
    ggml_tensor* down_w = nullptr;
};

struct frt_qwen3_dims {
    int n_layers = 0;
    int d_model = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 128;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-6f;
    int sliding_window = 0; // 0 = full causal
};

// PatchEncoder / DiT transformer block (modules.py DiTBlock and dit.py
// DiTBlock share attention+FFN; the DiT variant adds conv + 9-way AdaLN).
struct frt_ditblock {
    ggml_tensor* norm1_w = nullptr; // RMSNorm
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm3_w = nullptr; // DiT only
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* adaln_w = nullptr; // DiT only (9*1024, 1024)
    ggml_tensor* adaln_b = nullptr;
    ggml_tensor* conv0_w = nullptr; // DiT only, Conv1d k=3
    ggml_tensor* conv0_b = nullptr;
    ggml_tensor* conv2_w = nullptr;
    ggml_tensor* conv2_b = nullptr;
};

struct fireredtts3_tts_context {
    // Backends / scheduling
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Weight storage
    ggml_context* core_ctx = nullptr;
    ggml_backend_buffer_t core_buf = nullptr;
    std::map<std::string, ggml_tensor*> core_tensors;
    ggml_context* redae_ctx = nullptr;
    ggml_backend_buffer_t redae_buf = nullptr;
    std::map<std::string, ggml_tensor*> redae_tensors;

    // ── core hparams ──
    int llm_n_layers = 28, llm_d = 2048, llm_heads = 16, llm_kv_heads = 8, llm_head_dim = 128;
    float llm_theta = 1000000.0f, llm_eps = 1e-6f;
    int penc_layers = 8, penc_d = 1024, penc_heads = 16, penc_ratio = 4;
    int dit_layers = 11, dit_d = 1024, dit_heads = 16, dit_ratio = 3;
    int redae_dim = 64, spk_dim = 512, patch_size = 4, n_hist_patches = 2;

    // ── redae hparams ──
    int ae_patch = 480, ae_sr = 24000;
    int enc_layers = 18, encds_layers = 4, dec_layers = 18;
    int ae_d = 896, ae_heads = 14, ae_kv_heads = 2, ae_head_dim = 128;
    int ae_window = 64, ae_ds_rate = 2;
    float ae_theta = 10000.0f, ae_eps = 1e-6f;

    // ── core tensors ──
    ggml_tensor* tok_emb = nullptr;
    std::vector<frt_qwen3_block> llm_blocks;
    ggml_tensor* llm_norm_w = nullptr;
    ggml_tensor* spk_llm_w = nullptr;
    ggml_tensor* spk_llm_b = nullptr;
    ggml_tensor* spk_dit_w = nullptr;
    ggml_tensor* spk_dit_b = nullptr;
    // PatchEncoder
    ggml_tensor* penc_cls = nullptr;
    ggml_tensor* penc_in_w = nullptr;
    ggml_tensor* penc_in_b = nullptr;
    std::vector<frt_ditblock> penc_blocks;
    ggml_tensor* penc_out_norm_w = nullptr;
    ggml_tensor* penc_out_w = nullptr;
    ggml_tensor* penc_out_b = nullptr;
    // DiT
    ggml_tensor* dit_in_w = nullptr;
    ggml_tensor* dit_in_b = nullptr;
    ggml_tensor* dit_tm0_w = nullptr;
    ggml_tensor* dit_tm0_b = nullptr;
    ggml_tensor* dit_tm2_w = nullptr;
    ggml_tensor* dit_tm2_b = nullptr;
    std::vector<frt_ditblock> dit_blocks;
    ggml_tensor* dit_final_adaln_w = nullptr;
    ggml_tensor* dit_final_adaln_b = nullptr;
    ggml_tensor* dit_final_w = nullptr;
    ggml_tensor* dit_final_b = nullptr;
    ggml_tensor* dit_head_w = nullptr;
    ggml_tensor* dit_head_b = nullptr;
    std::vector<float> stop_w; // host copies (F32, tiny)
    float stop_b = 0.0f;

    // ── redae tensors ──
    ggml_tensor* enc_in0_w = nullptr;
    ggml_tensor* enc_in0_b = nullptr;
    ggml_tensor* enc_in1_w = nullptr;
    ggml_tensor* enc_in1_b = nullptr;
    std::vector<frt_qwen3_block> enc_blocks;
    ggml_tensor* enc_norm_w = nullptr;
    ggml_tensor* encds_cls = nullptr;
    std::vector<frt_qwen3_block> encds_blocks;
    ggml_tensor* encds_norm_w = nullptr;
    ggml_tensor* enc_out_w = nullptr;
    ggml_tensor* enc_out_b = nullptr;
    ggml_tensor* dec_in_w = nullptr;
    ggml_tensor* dec_in_b = nullptr;
    std::vector<frt_qwen3_block> dec_blocks;
    ggml_tensor* dec_norm_w = nullptr;
    ggml_tensor* dec_istft_w = nullptr;
    ggml_tensor* dec_istft_b = nullptr;
    std::vector<float> istft_window; // shipped window, host F32

    // CAM++
    cb_campplus_model campp{};
    chatterbox_campplus::cb_campplus_runtime campp_rt{};
    bool has_campp = false;

    // Tokenizer
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    int32_t tok_sot = -1, tok_eot = -1;

    // Default prompt (baked into the core GGUF)
    std::vector<float> dprompt_latents; // (T_lat * 64), frame-major
    int dprompt_T = 0;
    std::vector<float> dprompt_spk; // (512)
    std::string dprompt_text;
    std::string dprompt_lang = "English";

    // Current voice (set_voice_pcm) — empty = use default prompt
    std::vector<float> voice_latents;
    int voice_T = 0;
    std::vector<float> voice_spk;
    std::string ref_text;
    std::string language; // empty = prompt default

    // Params
    fireredtts3_tts_context_params params{};
    crispasr::core::mt19937_state rng{};
    bool rng_seeded = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

static ggml_tensor* frt_get(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return it == m.end() ? nullptr : it->second;
}

static ggml_tensor* frt_req(const std::map<std::string, ggml_tensor*>& m, const std::string& name, bool& ok) {
    ggml_tensor* t = frt_get(m, name);
    if (!t) {
        std::fprintf(stderr, "fireredtts3: missing tensor '%s'\n", name.c_str());
        ok = false;
    }
    return t;
}

static void frt_read_f32(ggml_tensor* t, std::vector<float>& out) {
    out.resize(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(out.size());
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < out.size(); i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        std::fprintf(stderr, "fireredtts3: frt_read_f32 unsupported type %d\n", (int)t->type);
        out.assign(out.size(), 0.0f);
    }
}

// Graph-run scaffold: build ctx0 over compute_meta, caller builds graph,
// then alloc + compute on the shared sched.
struct frt_graph {
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = nullptr;
};

static frt_graph frt_graph_begin(fireredtts3_tts_context* ctx) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    frt_graph g;
    g.ctx0 = ggml_init(ip);
    g.gf = ggml_new_graph_custom(g.ctx0, 16384, false);
    return g;
}

static bool frt_graph_alloc(fireredtts3_tts_context* ctx, frt_graph& g) {
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, g.gf)) {
        std::fprintf(stderr, "fireredtts3: graph alloc failed\n");
        return false;
    }
    return true;
}

static bool frt_graph_compute(fireredtts3_tts_context* ctx, frt_graph& g) {
    if (ggml_backend_sched_graph_compute(ctx->sched, g.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "fireredtts3: graph compute failed\n");
        ggml_free(g.ctx0);
        g.ctx0 = nullptr;
        return false;
    }
    return true;
}

static void frt_graph_end(frt_graph& g) {
    if (g.ctx0)
        ggml_free(g.ctx0);
    g.ctx0 = nullptr;
}

// Host-built attention mask (F32, ne0 = Lk, ne1 = T). 0 allowed, -INF not.
//   causal: j <= i;  sliding window w: j >= i-w+1
//   block-diagonal blocks of size B (block b: tokens [bB, (b+1)B))
static void frt_fill_mask(std::vector<float>& m, int T, bool causal, int window, int block) {
    m.assign((size_t)T * T, 0.0f);
    const float NEG = -1e30f;
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            bool ok = true;
            if (causal && j > i)
                ok = false;
            if (window > 0 && j < i - window + 1)
                ok = false;
            if (block > 0 && (j / block) != (i / block))
                ok = false;
            if (!ok)
                m[(size_t)i * T + j] = NEG;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen3 stack (shared by backbone + RedAE encoder/downsample/decoder)
// Full forward, no KV cache. cur: (d, T). Returns pre-norm hidden? No —
// applies the final RMSNorm with `final_norm_w` when non-null.
// ─────────────────────────────────────────────────────────────────────────────

static ggml_tensor* frt_qwen3_forward(ggml_context* ctx0, ggml_tensor* cur, const std::vector<frt_qwen3_block>& blocks,
                                      ggml_tensor* final_norm_w, ggml_tensor* positions, ggml_tensor* mask,
                                      const frt_qwen3_dims& dm) {
    const int T = (int)cur->ne[1];
    const int hd = dm.head_dim;
    const int n_q = dm.n_heads;
    const int n_kv = dm.n_kv_heads;
    const int grp = n_q / n_kv;
    const float scale = 1.0f / std::sqrt((float)hd);

    for (int il = 0; il < dm.n_layers; il++) {
        const auto& b = blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, dm.rms_eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* Q = ggml_mul_mat(ctx0, b.q_w, x);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.k_w, x);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.v_w, x);
        Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

        // Qwen3 per-head QK RMSNorm
        Q = ggml_rms_norm(ctx0, Q, dm.rms_eps);
        Q = ggml_mul(ctx0, Q, b.q_norm_w);
        K = ggml_rms_norm(ctx0, K, dm.rms_eps);
        K = ggml_mul(ctx0, K, b.k_norm_w);

        // NEOX RoPE
        Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, dm.rope_theta, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f);
        K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, dm.rope_theta, 1.0f, 0.0f, 1.0f,
                          32.0f, 1.0f);

        if (grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(ctx0, K, hd, 1, n_kv, T);
            ggml_tensor* V4 = ggml_reshape_4d(ctx0, V, hd, 1, n_kv, T);
            K4 = ggml_repeat_4d(ctx0, K4, hd, grp, n_kv, T);
            V4 = ggml_repeat_4d(ctx0, V4, hd, grp, n_kv, T);
            K = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, n_q, T));
            V = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, n_q, T));
        }

        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (T, T, n_q)
        scores = ggml_add(ctx0, scores, mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);

        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

        attn = ggml_mul_mat(ctx0, b.o_w, attn);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, dm.rms_eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.gate_w, b.up_w, b.down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    if (final_norm_w) {
        cur = ggml_rms_norm(ctx0, cur, dm.rms_eps);
        cur = ggml_mul(ctx0, cur, final_norm_w);
    }
    return cur;
}

// ─────────────────────────────────────────────────────────────────────────────
// PatchEncoder / DiT shared pieces (x_transformers RoPE = mode NORMAL)
// ─────────────────────────────────────────────────────────────────────────────

static ggml_tensor* frt_mish(ggml_context* ctx0, ggml_tensor* x) {
    return ggml_mul(ctx0, x, ggml_tanh(ctx0, ggml_softplus(ctx0, x)));
}

static ggml_tensor* frt_linear(ggml_context* ctx0, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
    if (b)
        y = ggml_add(ctx0, y, b);
    return y;
}

// Full attention with biases + NORMAL rope (modules.py Attention). x: (d, T).
static ggml_tensor* frt_dit_attn(ggml_context* ctx0, const frt_ditblock& b, ggml_tensor* x, ggml_tensor* positions,
                                 ggml_tensor* mask, int n_heads) {
    const int d = (int)x->ne[0];
    const int T = (int)x->ne[1];
    const int hd = d / n_heads;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* Q = frt_linear(ctx0, b.q_w, b.q_b, x);
    ggml_tensor* K = frt_linear(ctx0, b.k_w, b.k_b, x);
    ggml_tensor* V = frt_linear(ctx0, b.v_w, b.v_b, x);
    Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_heads, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_heads, T);

    // x_transformers adjacent-pair RoPE over the full head dim, theta 1e4
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q);
    if (mask)
        scores = ggml_add(ctx0, scores, mask);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);

    ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores);
    attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(ctx0, attn, d, T);
    return frt_linear(ctx0, b.o_w, b.o_b, attn);
}

// GELU-tanh FFN with biases (modules.py FeedForward, approximate="tanh").
static ggml_tensor* frt_dit_ffn(ggml_context* ctx0, const frt_ditblock& b, ggml_tensor* x) {
    ggml_tensor* h = frt_linear(ctx0, b.ffn_up_w, b.ffn_up_b, x);
    h = ggml_gelu(ctx0, h); // tanh approximation
    return frt_linear(ctx0, b.ffn_down_w, b.ffn_down_b, h);
}

// modulate(x, shift, scale) = x * (1 + scale) + shift; shift/scale (d, 1)
static ggml_tensor* frt_modulate(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* shift, ggml_tensor* scale) {
    ggml_tensor* one_plus = ggml_scale_bias(ctx0, scale, 1.0f, 1.0f); // 1 + scale
    return ggml_add(ctx0, ggml_mul(ctx0, x, one_plus), shift);
}

// PatchEncoder body: x (1024, T_seq) already in_proj'd with cls tokens
// interleaved; returns hidden after all blocks (pre-out_proj).
static ggml_tensor* frt_penc_body(fireredtts3_tts_context* ctx, ggml_context* ctx0, ggml_tensor* x,
                                  ggml_tensor* positions, ggml_tensor* mask) {
    for (int il = 0; il < ctx->penc_layers; il++) {
        const auto& b = ctx->penc_blocks[il];
        ggml_tensor* h = ggml_rms_norm(ctx0, x, 1e-6f);
        h = ggml_mul(ctx0, h, b.norm1_w);
        x = ggml_add(ctx0, x, frt_dit_attn(ctx0, b, h, positions, mask, ctx->penc_heads));
        h = ggml_rms_norm(ctx0, x, 1e-6f);
        h = ggml_mul(ctx0, h, b.norm2_w);
        x = ggml_add(ctx0, x, frt_dit_ffn(ctx0, b, h));
    }
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// PatchEncoder forward: latents (n_frames, 64) host → embeds (2048, n_patches)
// Sequence per patch: [CLS, f0, f1, f2, f3]; block-diag mask; positions 0..4.
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_penc_forward(fireredtts3_tts_context* ctx, const float* latents, int n_frames,
                             std::vector<float>& out /* (n_patches * 2048) row-major */) {
    frt_bench_stage bench_("penc_forward");
    const int P = n_frames / ctx->patch_size;
    const int B = 1 + ctx->patch_size; // 5 tokens per patch
    const int T = P * B;
    const int d = ctx->penc_d;

    frt_graph g = frt_graph_begin(ctx);

    ggml_tensor* lat_in = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ctx->redae_dim, n_frames);
    ggml_set_name(lat_in, "lat_in");
    ggml_set_input(lat_in);
    ggml_tensor* positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* mask = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);
    // Sequence assembly indices: for token t in patch p, source row —
    // built host-side as a gather over [cls_row | projected latents].
    // We instead build the sequence by projecting latents then scattering:
    // proj (1024, n_frames); cls (1024, 1) broadcast. Use get_rows with an
    // index map over the concat [cls, proj].
    ggml_tensor* proj = frt_linear(g.ctx0, ctx->penc_in_w, ctx->penc_in_b, lat_in); // (1024, n_frames)
    ggml_tensor* cls2d = ggml_reshape_2d(g.ctx0, ctx->penc_cls, d, 1);
    ggml_tensor* cat = ggml_concat(g.ctx0, cls2d, proj, 1); // (1024, 1 + n_frames)
    ggml_tensor* seq_idx = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
    ggml_set_name(seq_idx, "seq_idx");
    ggml_set_input(seq_idx);
    ggml_tensor* x = ggml_get_rows(g.ctx0, cat, seq_idx); // (1024, T)

    x = frt_penc_body(ctx, g.ctx0, x, positions, mask);

    // Take the CLS token (index 0) of every patch, then out_proj.
    ggml_tensor* xc = ggml_cont(g.ctx0, x);
    ggml_tensor* cls_rows = ggml_view_2d(g.ctx0, xc, d, P, xc->nb[1] * B, 0);
    cls_rows = ggml_cont(g.ctx0, cls_rows);
    ggml_tensor* h = ggml_rms_norm(g.ctx0, cls_rows, 1e-6f);
    h = ggml_mul(g.ctx0, h, ctx->penc_out_norm_w);
    ggml_tensor* outT = frt_linear(g.ctx0, ctx->penc_out_w, ctx->penc_out_b, h); // (2048, P)
    ggml_set_name(outT, "penc_out");
    ggml_set_output(outT);
    ggml_build_forward_expand(g.gf, outT);

    if (!frt_graph_alloc(ctx, g)) {
        frt_graph_end(g);
        return false;
    }

    // Inputs
    ggml_backend_tensor_set(lat_in, latents, 0, (size_t)ctx->redae_dim * n_frames * sizeof(float));
    std::vector<int32_t> pos(T), sidx(T);
    for (int p = 0; p < P; p++) {
        for (int t = 0; t < B; t++) {
            pos[p * B + t] = t;
            sidx[p * B + t] = (t == 0) ? 0 : (1 + p * ctx->patch_size + (t - 1));
        }
    }
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(seq_idx, sidx.data(), 0, sidx.size() * sizeof(int32_t));
    std::vector<float> m;
    frt_fill_mask(m, T, /*causal=*/false, /*window=*/0, /*block=*/B);
    ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));

    if (!frt_graph_compute(ctx, g))
        return false;

    out.resize((size_t)P * ctx->llm_d);
    ggml_backend_tensor_get(outT, out.data(), 0, out.size() * sizeof(float));
    frt_graph_end(g);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backbone LLM forward: embeds (2048, T) host → hidden (2048, T) host
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_llm_forward(fireredtts3_tts_context* ctx, const float* embeds, int T, std::vector<float>& hidden) {
    frt_bench_stage bench_("llm_forward");
    frt_graph g = frt_graph_begin(ctx);

    ggml_tensor* x = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ctx->llm_d, T);
    ggml_set_name(x, "embeds");
    ggml_set_input(x);
    ggml_tensor* positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* mask = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    frt_qwen3_dims dm;
    dm.n_layers = ctx->llm_n_layers;
    dm.d_model = ctx->llm_d;
    dm.n_heads = ctx->llm_heads;
    dm.n_kv_heads = ctx->llm_kv_heads;
    dm.head_dim = ctx->llm_head_dim;
    dm.rope_theta = ctx->llm_theta;
    dm.rms_eps = ctx->llm_eps;

    ggml_tensor* h = frt_qwen3_forward(g.ctx0, x, ctx->llm_blocks, ctx->llm_norm_w, positions, mask, dm);
    ggml_set_name(h, "hidden");
    ggml_set_output(h);
    ggml_build_forward_expand(g.gf, h);

    if (!frt_graph_alloc(ctx, g)) {
        frt_graph_end(g);
        return false;
    }
    ggml_backend_tensor_set(x, embeds, 0, (size_t)ctx->llm_d * T * sizeof(float));
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++)
        pos[i] = i;
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<float> m;
    frt_fill_mask(m, T, /*causal=*/true, 0, 0);
    ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));

    if (!frt_graph_compute(ctx, g))
        return false;
    hidden.resize((size_t)ctx->llm_d * T);
    ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size() * sizeof(float));
    frt_graph_end(g);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// DiT flow: one Euler step. Builds cond+uncond arms in one graph.
//   xt      (64, 12)   history(8, clean) + current(4, noisy)
//   cond3   (2048, 3)  raw last-3 backbone hiddens (dit_head applied in-graph)
//   spk     (512,)
//   t_sin   (256,)     host-computed sinusoidal timestep features
// Output: vt (64, 12) for both arms.
// ─────────────────────────────────────────────────────────────────────────────

static ggml_tensor* frt_dit_body(fireredtts3_tts_context* ctx, ggml_context* ctx0, ggml_tensor* x_in,
                                 ggml_tensor* t_emb, ggml_tensor* positions, int T) {
    // x_in: (1600, T); t_emb: (1024, 1)
    ggml_tensor* x = frt_linear(ctx0, ctx->dit_in_w, ctx->dit_in_b, x_in); // (1024, T)

    for (int il = 0; il < ctx->dit_layers; il++) {
        const auto& b = ctx->dit_blocks[il];
        // AdaLN: SiLU(t_emb) → Linear → 9 chunks of (1024, 1)
        ggml_tensor* ada = frt_linear(ctx0, b.adaln_w, b.adaln_b, ggml_silu(ctx0, t_emb)); // (9216, 1)
        const int d = ctx->dit_d;
        auto chunk = [&](int i) {
            return ggml_cont(ctx0, ggml_view_2d(ctx0, ada, d, 1, ada->nb[1], (size_t)i * d * ggml_element_size(ada)));
        };
        ggml_tensor* shift_msa = chunk(0);
        ggml_tensor* scale_msa = chunk(1);
        ggml_tensor* gate_msa = chunk(2);
        ggml_tensor* shift_mlp = chunk(3);
        ggml_tensor* scale_mlp = chunk(4);
        ggml_tensor* gate_mlp = chunk(5);
        ggml_tensor* shift_conv = chunk(6);
        ggml_tensor* scale_conv = chunk(7);
        ggml_tensor* gate_conv = chunk(8);

        // Attn branch
        ggml_tensor* h = ggml_rms_norm(ctx0, x, 1e-6f);
        h = ggml_mul(ctx0, h, b.norm1_w);
        h = frt_modulate(ctx0, h, shift_msa, scale_msa);
        h = frt_dit_attn(ctx0, b, h, positions, nullptr, ctx->dit_heads);
        x = ggml_add(ctx0, x, ggml_mul(ctx0, h, gate_msa));

        // Conv branch: (1024, T) → transpose (T, 1024) → conv k=3 p=1 → Mish
        // → conv k=3 p=1 → transpose back
        h = ggml_rms_norm(ctx0, x, 1e-6f);
        h = ggml_mul(ctx0, h, b.norm2_w);
        h = frt_modulate(ctx0, h, shift_conv, scale_conv);
        ggml_tensor* hc = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // (T, 1024)
        hc = ggml_conv_1d(ctx0, b.conv0_w, hc, 1, 1, 1);            // (T, 1024)
        if (b.conv0_b)
            hc = ggml_add(ctx0, hc, ggml_reshape_2d(ctx0, b.conv0_b, 1, ctx->dit_d));
        hc = frt_mish(ctx0, hc);
        hc = ggml_conv_1d(ctx0, b.conv2_w, hc, 1, 1, 1);
        if (b.conv2_b)
            hc = ggml_add(ctx0, hc, ggml_reshape_2d(ctx0, b.conv2_b, 1, ctx->dit_d));
        h = ggml_cont(ctx0, ggml_transpose(ctx0, hc)); // (1024, T)
        x = ggml_add(ctx0, x, ggml_mul(ctx0, h, gate_conv));

        // FFN branch
        h = ggml_rms_norm(ctx0, x, 1e-6f);
        h = ggml_mul(ctx0, h, b.norm3_w);
        h = frt_modulate(ctx0, h, shift_mlp, scale_mlp);
        h = frt_dit_ffn(ctx0, b, h);
        x = ggml_add(ctx0, x, ggml_mul(ctx0, h, gate_mlp));
    }

    // FinalLayer: LayerNorm (no affine, eps 1e-6) + modulate(shift, scale) + linear
    ggml_tensor* ada = frt_linear(ctx0, ctx->dit_final_adaln_w, ctx->dit_final_adaln_b, ggml_silu(ctx0, t_emb));
    const int d = ctx->dit_d;
    ggml_tensor* f_shift = ggml_cont(ctx0, ggml_view_2d(ctx0, ada, d, 1, ada->nb[1], 0));
    ggml_tensor* f_scale =
        ggml_cont(ctx0, ggml_view_2d(ctx0, ada, d, 1, ada->nb[1], (size_t)d * ggml_element_size(ada)));
    ggml_tensor* h = ggml_norm(ctx0, x, 1e-6f); // LayerNorm without affine
    h = frt_modulate(ctx0, h, f_shift, f_scale);
    return frt_linear(ctx0, ctx->dit_final_w, ctx->dit_final_b, h); // (64, T)
}

// One flow ODE solve for the current patch (all n_timesteps Euler steps).
// hist (8*64), noise (4*64) — both frame-major host arrays.
// cond3 (3*2048) row-major host; spk (512). Returns the denoised current
// patch (4*64) in out; optionally records per-step xt for debugging.
static bool frt_flow_solve(fireredtts3_tts_context* ctx, const float* hist, const float* noise, const float* cond3,
                           const float* spk, std::vector<float>& out) {
    frt_bench_stage bench_("flow_solve");
    const int ps = ctx->patch_size;                   // 4
    const int hl = ctx->n_hist_patches * ps;          // 8
    const int T = hl + ps;                            // 12
    const int ld = ctx->redae_dim;                    // 64
    const int nt = ctx->params.n_timesteps;           // 10
    const float cfg = ctx->params.cfg_scale;          // 2.0
    const int in_ch = ld + ctx->dit_d + ctx->spk_dim; // 1600

    // t_span = 1 - cos(linspace(0, 1, nt+1) * pi/2)
    std::vector<float> t_span(nt + 1);
    for (int i = 0; i <= nt; i++) {
        float u = (float)i / (float)nt;
        t_span[i] = 1.0f - std::cos(u * 0.5f * (float)M_PI);
    }

    // xt: (T, 64) frame-major (frame t at xt[t*64..])
    std::vector<float> xt((size_t)T * ld);
    std::memcpy(xt.data(), hist, (size_t)hl * ld * sizeof(float));
    std::memcpy(xt.data() + (size_t)hl * ld, noise, (size_t)ps * ld * sizeof(float));

    for (int ti = 0; ti < nt; ti++) {
        const float t = t_span[ti];
        const float dt = t_span[ti + 1] - t_span[ti];

        // Host sinusoidal timestep features (SinusPositionEmbedding, scale=1000)
        const int half = 128;
        std::vector<float> t_sin(2 * half);
        for (int k = 0; k < half; k++) {
            double f = std::exp(-(std::log(10000.0) / (half - 1)) * k);
            double a = 1000.0 * (double)t * f;
            t_sin[k] = (float)std::sin(a);
            t_sin[half + k] = (float)std::cos(a);
        }

        frt_graph g = frt_graph_begin(ctx);

        ggml_tensor* xt_in = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ld, T);
        ggml_set_input(xt_in);
        ggml_set_name(xt_in, "xt");
        ggml_tensor* cond3_in = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ctx->llm_d, ctx->n_hist_patches + 1);
        ggml_set_input(cond3_in);
        ggml_set_name(cond3_in, "cond3");
        ggml_tensor* spk_in = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_F32, ctx->spk_dim);
        ggml_set_input(spk_in);
        ggml_set_name(spk_in, "spk");
        ggml_tensor* tsin_in = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, 256, 1);
        ggml_set_input(tsin_in);
        ggml_set_name(tsin_in, "t_sin");
        ggml_tensor* positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
        ggml_set_input(positions);
        ggml_set_name(positions, "positions");

        // t embedding MLP: Linear(256→1024) → SiLU → Linear(1024→1024)
        ggml_tensor* t_emb = frt_linear(g.ctx0, ctx->dit_tm0_w, ctx->dit_tm0_b, tsin_in);
        t_emb = ggml_silu(g.ctx0, t_emb);
        t_emb = frt_linear(g.ctx0, ctx->dit_tm2_w, ctx->dit_tm2_b, t_emb); // (1024, 1)

        // cond: dit_head(cond3) → (1024, 3) → repeat_interleave(ps) → (1024, T)
        ggml_tensor* bc = frt_linear(g.ctx0, ctx->dit_head_w, ctx->dit_head_b, cond3_in); // (1024, 3)
        bc = ggml_reshape_3d(g.ctx0, bc, ctx->dit_d, 1, ctx->n_hist_patches + 1);
        bc = ggml_repeat_4d(g.ctx0, bc, ctx->dit_d, ps, ctx->n_hist_patches + 1, 1);
        bc = ggml_cont(g.ctx0, ggml_reshape_2d(g.ctx0, bc, ctx->dit_d, T));
        // spk broadcast (512, T)
        ggml_tensor* spk2 = ggml_reshape_2d(g.ctx0, spk_in, ctx->spk_dim, 1);
        ggml_tensor* spkT = ggml_repeat(g.ctx0, spk2, ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ctx->spk_dim, T));
        ggml_tensor* cond = ggml_concat(g.ctx0, bc, spkT, 0); // (1536, T)

        // cond arm
        ggml_tensor* x_cond = ggml_concat(g.ctx0, xt_in, cond, 0); // (1600, T)
        ggml_tensor* v_cond = frt_dit_body(ctx, g.ctx0, x_cond, t_emb, positions, T);
        ggml_set_name(v_cond, "v_cond");
        ggml_set_output(v_cond);
        ggml_build_forward_expand(g.gf, v_cond);

        ggml_tensor* v_uncond = nullptr;
        if (cfg > 0.0f) {
            ggml_tensor* zeros = ggml_scale(g.ctx0, cond, 0.0f);
            ggml_tensor* x_un = ggml_concat(g.ctx0, xt_in, zeros, 0);
            v_uncond = frt_dit_body(ctx, g.ctx0, x_un, t_emb, positions, T);
            ggml_set_name(v_uncond, "v_uncond");
            ggml_set_output(v_uncond);
            ggml_build_forward_expand(g.gf, v_uncond);
        }

        if (!frt_graph_alloc(ctx, g)) {
            frt_graph_end(g);
            return false;
        }
        ggml_backend_tensor_set(xt_in, xt.data(), 0, xt.size() * sizeof(float));
        ggml_backend_tensor_set(cond3_in, cond3, 0, (size_t)ctx->llm_d * (ctx->n_hist_patches + 1) * sizeof(float));
        ggml_backend_tensor_set(spk_in, spk, 0, (size_t)ctx->spk_dim * sizeof(float));
        ggml_backend_tensor_set(tsin_in, t_sin.data(), 0, 256 * sizeof(float));
        std::vector<int32_t> pos(T);
        for (int i = 0; i < T; i++)
            pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));

        if (!frt_graph_compute(ctx, g))
            return false;

        std::vector<float> vc((size_t)T * ld), vu;
        ggml_backend_tensor_get(v_cond, vc.data(), 0, vc.size() * sizeof(float));
        if (v_uncond) {
            vu.resize((size_t)T * ld);
            ggml_backend_tensor_get(v_uncond, vu.data(), 0, vu.size() * sizeof(float));
        }
        frt_graph_end(g);

        // vt = (1+cfg)*v_cond - cfg*v_uncond; Euler-update ONLY the last patch
        for (int f = hl; f < T; f++) {
            for (int c = 0; c < ld; c++) {
                float v = vc[(size_t)f * ld + c];
                if (v_uncond)
                    v = (1.0f + cfg) * v - cfg * vu[(size_t)f * ld + c];
                xt[(size_t)f * ld + c] += dt * v;
            }
        }
    }

    out.assign(xt.begin() + (size_t)hl * ld, xt.end());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// RedAE encoder: 24 kHz padded PCM → latents (T_lat, 64) frame-major host.
// Optionally captures the pre-downsample hidden (T50, 896) for the diff.
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_redae_encode(fireredtts3_tts_context* ctx, const float* pcm24, int n_samples,
                             std::vector<float>& latents, int& T_lat, std::vector<float>* enc_hidden_out) {
    frt_bench_stage bench_("redae_encode");
    if (n_samples % (ctx->ae_patch * ctx->ae_ds_rate) != 0) {
        std::fprintf(stderr, "fireredtts3: redae_encode input not padded (%d)\n", n_samples);
        return false;
    }
    const int T50 = n_samples / ctx->ae_patch;
    const int d = ctx->ae_d;

    frt_qwen3_dims dm;
    dm.n_layers = ctx->enc_layers;
    dm.d_model = d;
    dm.n_heads = ctx->ae_heads;
    dm.n_kv_heads = ctx->ae_kv_heads;
    dm.head_dim = ctx->ae_head_dim;
    dm.rope_theta = ctx->ae_theta;
    dm.rms_eps = ctx->ae_eps;
    dm.sliding_window = ctx->ae_window;

    // Stage 1: patches → in_proj0/1 → 18L sliding-window Qwen3 (final norm)
    std::vector<float> hidden;
    {
        frt_graph g = frt_graph_begin(ctx);
        ggml_tensor* audio = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ctx->ae_patch, T50);
        ggml_set_input(audio);
        ggml_set_name(audio, "audio_patches");
        ggml_tensor* positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T50);
        ggml_set_input(positions);
        ggml_set_name(positions, "positions");
        ggml_tensor* mask = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, T50, T50);
        ggml_set_input(mask);
        ggml_set_name(mask, "mask");

        ggml_tensor* x = frt_linear(g.ctx0, ctx->enc_in0_w, ctx->enc_in0_b, audio);
        x = frt_linear(g.ctx0, ctx->enc_in1_w, ctx->enc_in1_b, x);
        ggml_tensor* h = frt_qwen3_forward(g.ctx0, x, ctx->enc_blocks, ctx->enc_norm_w, positions, mask, dm);
        ggml_set_name(h, "enc_hidden");
        ggml_set_output(h);
        ggml_build_forward_expand(g.gf, h);

        if (!frt_graph_alloc(ctx, g)) {
            frt_graph_end(g);
            return false;
        }
        ggml_backend_tensor_set(audio, pcm24, 0, (size_t)n_samples * sizeof(float));
        std::vector<int32_t> pos(T50);
        for (int i = 0; i < T50; i++)
            pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        std::vector<float> m;
        frt_fill_mask(m, T50, /*causal=*/true, ctx->ae_window, 0);
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));

        if (!frt_graph_compute(ctx, g))
            return false;
        hidden.resize((size_t)d * T50);
        ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size() * sizeof(float));
        frt_graph_end(g);
    }
    if (enc_hidden_out)
        *enc_hidden_out = hidden;

    // Stage 2: CLS downsample — groups of 2 frames + trained CLS token LAST,
    // 4L Qwen3 (causal within group), take token 2, out_proj → 64.
    const int G = T50 / ctx->ae_ds_rate;
    const int B = ctx->ae_ds_rate + 1; // 3 tokens per group
    const int T = G * B;
    {
        frt_qwen3_dims dsdm = dm;
        dsdm.n_layers = ctx->encds_layers;
        dsdm.sliding_window = 0; // downsample stack: plain causal over 3 tokens

        frt_graph g = frt_graph_begin(ctx);
        ggml_tensor* hin = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, d, T50);
        ggml_set_input(hin);
        ggml_set_name(hin, "ds_in");
        ggml_tensor* positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
        ggml_set_input(positions);
        ggml_set_name(positions, "positions");
        ggml_tensor* mask = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, T, T);
        ggml_set_input(mask);
        ggml_set_name(mask, "mask");
        ggml_tensor* seq_idx = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
        ggml_set_input(seq_idx);
        ggml_set_name(seq_idx, "seq_idx");

        // in_proj / out_proj are Identity (896 == 896). Sequence gather over
        // [cls | hidden]: group g = [f_{2g}, f_{2g+1}, cls].
        ggml_tensor* cls2d = ggml_reshape_2d(g.ctx0, ctx->encds_cls, d, 1);
        ggml_tensor* cat = ggml_concat(g.ctx0, cls2d, hin, 1); // (896, 1 + T50)
        ggml_tensor* x = ggml_get_rows(g.ctx0, cat, seq_idx);  // (896, T)

        ggml_tensor* h = frt_qwen3_forward(g.ctx0, x, ctx->encds_blocks, ctx->encds_norm_w, positions, mask, dsdm);
        // take token index B-1 (the CLS) of each group
        ggml_tensor* hc = ggml_cont(g.ctx0, h);
        ggml_tensor* cls_rows = ggml_view_2d(g.ctx0, hc, d, G, hc->nb[1] * B, hc->nb[1] * (B - 1));
        cls_rows = ggml_cont(g.ctx0, cls_rows);
        ggml_tensor* lat = frt_linear(g.ctx0, ctx->enc_out_w, ctx->enc_out_b, cls_rows); // (64, G)
        ggml_set_name(lat, "latents");
        ggml_set_output(lat);
        ggml_build_forward_expand(g.gf, lat);

        if (!frt_graph_alloc(ctx, g)) {
            frt_graph_end(g);
            return false;
        }
        ggml_backend_tensor_set(hin, hidden.data(), 0, hidden.size() * sizeof(float));
        std::vector<int32_t> pos(T), sidx(T);
        for (int gi = 0; gi < G; gi++) {
            for (int t = 0; t < B; t++) {
                pos[gi * B + t] = t;
                sidx[gi * B + t] = (t == B - 1) ? 0 : (1 + gi * ctx->ae_ds_rate + t);
            }
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(seq_idx, sidx.data(), 0, sidx.size() * sizeof(int32_t));
        std::vector<float> m;
        frt_fill_mask(m, T, /*causal=*/true, 0, /*block=*/B);
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));

        if (!frt_graph_compute(ctx, g))
            return false;
        latents.resize((size_t)ctx->redae_dim * G);
        ggml_backend_tensor_get(lat, latents.data(), 0, latents.size() * sizeof(float));
        frt_graph_end(g);
    }
    T_lat = G;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// RedAE decoder: latents (T_lat, 64) → 24 kHz PCM.
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_redae_decode(fireredtts3_tts_context* ctx, const float* latents, int T_lat, std::vector<float>& pcm,
                             std::vector<float>* dec_hidden_out) {
    frt_bench_stage bench_("redae_decode");
    const int d = ctx->ae_d;
    const int T50 = T_lat * ctx->ae_ds_rate;
    const int n_freq = ctx->ae_patch * 4 / 2 + 1; // n_fft = 1920 → 961

    frt_qwen3_dims dm;
    dm.n_layers = ctx->dec_layers;
    dm.d_model = d;
    dm.n_heads = ctx->ae_heads;
    dm.n_kv_heads = ctx->ae_kv_heads;
    dm.head_dim = ctx->ae_head_dim;
    dm.rope_theta = ctx->ae_theta;
    dm.rms_eps = ctx->ae_eps;
    dm.sliding_window = ctx->ae_window;

    std::vector<float> spec; // (1922, T50) column-major read
    {
        frt_graph g = frt_graph_begin(ctx);
        ggml_tensor* lat = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, ctx->redae_dim, T_lat);
        ggml_set_input(lat);
        ggml_set_name(lat, "latents");
        ggml_tensor* positions = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T50);
        ggml_set_input(positions);
        ggml_set_name(positions, "positions");
        ggml_tensor* mask = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, T50, T50);
        ggml_set_input(mask);
        ggml_set_name(mask, "mask");

        // in_proj 64 → 2*896, reshape (896, 2*T_lat)
        ggml_tensor* x = frt_linear(g.ctx0, ctx->dec_in_w, ctx->dec_in_b, lat); // (1792, T_lat)
        x = ggml_reshape_2d(g.ctx0, x, d, T50);

        ggml_tensor* h = frt_qwen3_forward(g.ctx0, x, ctx->dec_blocks, ctx->dec_norm_w, positions, mask, dm);
        if (dec_hidden_out) {
            ggml_set_name(h, "dec_hidden");
            ggml_set_output(h);
        }
        ggml_tensor* sp = frt_linear(g.ctx0, ctx->dec_istft_w, ctx->dec_istft_b, h); // (1922, T50)
        ggml_set_name(sp, "spec");
        ggml_set_output(sp);
        ggml_build_forward_expand(g.gf, sp);

        if (!frt_graph_alloc(ctx, g)) {
            frt_graph_end(g);
            return false;
        }
        ggml_backend_tensor_set(lat, latents, 0, (size_t)ctx->redae_dim * T_lat * sizeof(float));
        std::vector<int32_t> pos(T50);
        for (int i = 0; i < T50; i++)
            pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        std::vector<float> m;
        frt_fill_mask(m, T50, /*causal=*/true, ctx->ae_window, 0);
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));

        if (!frt_graph_compute(ctx, g))
            return false;
        spec.resize((size_t)2 * n_freq * T50);
        ggml_backend_tensor_get(sp, spec.data(), 0, spec.size() * sizeof(float));
        if (dec_hidden_out) {
            ggml_tensor* ht = ggml_graph_get_tensor(g.gf, "dec_hidden");
            if (ht) {
                dec_hidden_out->resize((size_t)d * T50);
                ggml_backend_tensor_get(ht, dec_hidden_out->data(), 0, dec_hidden_out->size() * sizeof(float));
            }
        }
        frt_graph_end(g);
    }

    // ISTFT head: mag = clip(exp(first 961), max 1e2); phase = next 961.
    // (T_frames, n_freq) row-major buffers for core_istft.
    const int n_fft = ctx->ae_patch * 4; // 1920
    std::vector<float> mag((size_t)T50 * n_freq), phase((size_t)T50 * n_freq);
    for (int t = 0; t < T50; t++) {
        const float* col = spec.data() + (size_t)t * 2 * n_freq;
        for (int f = 0; f < n_freq; f++) {
            float mg = std::exp(col[f]);
            if (mg > 1e2f)
                mg = 1e2f;
            mag[(size_t)t * n_freq + f] = mg;
            phase[(size_t)t * n_freq + f] = col[n_freq + f];
        }
    }
    // "same" padding ISTFT with the SHIPPED window; python divides by the
    // window envelope unconditionally after asserting > 1e-11.
    pcm = core_istft::istft(mag.data(), phase.data(), n_fft, ctx->ae_patch, T50,
                            ctx->istft_window.empty() ? nullptr : ctx->istft_window.data(), core_istft::TRIM_SAME, 0.0f,
                            1e-11f, false);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer: <|Lang|><|sot|>{ref_text}{text}<|eot|> — Qwen2 BPE via
// core_bpe::qwen_pretokenize + bpe_one (specials looked up literally).
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_tokenize(fireredtts3_tts_context* ctx, const std::string& lang, const std::string& body,
                         std::vector<int32_t>& ids) {
    const std::string lang_tok = "<|" + lang + "|>";
    auto it = ctx->token_to_id.find(lang_tok);
    if (it == ctx->token_to_id.end()) {
        std::fprintf(stderr, "fireredtts3: unknown language tag %s\n", lang_tok.c_str());
        return false;
    }
    ids.clear();
    ids.push_back(it->second);
    ids.push_back(ctx->tok_sot);
    for (const std::string& piece : core_bpe::qwen_pretokenize(body)) {
        std::string enc = core_bpe::bytes_to_unicode(piece.data(), piece.size());
        core_bpe::bpe_one(ctx->token_to_id, ctx->merge_rank, enc, ids);
    }
    ids.push_back(ctx->tok_eot);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading
// ─────────────────────────────────────────────────────────────────────────────

static bool frt_bind_qwen3_blocks(const std::map<std::string, ggml_tensor*>& tm, const std::string& prefix,
                                  int n_layers, std::vector<frt_qwen3_block>& blocks) {
    blocks.resize(n_layers);
    bool ok = true;
    for (int i = 0; i < n_layers; i++) {
        char base[128];
        std::snprintf(base, sizeof(base), "%s.layers.%d", prefix.c_str(), i);
        auto& b = blocks[i];
        b.q_w = frt_req(tm, std::string(base) + ".q.weight", ok);
        b.k_w = frt_req(tm, std::string(base) + ".k.weight", ok);
        b.v_w = frt_req(tm, std::string(base) + ".v.weight", ok);
        b.o_w = frt_req(tm, std::string(base) + ".o.weight", ok);
        b.q_norm_w = frt_req(tm, std::string(base) + ".q_norm.weight", ok);
        b.k_norm_w = frt_req(tm, std::string(base) + ".k_norm.weight", ok);
        b.attn_norm_w = frt_req(tm, std::string(base) + ".attn_norm.weight", ok);
        b.ffn_norm_w = frt_req(tm, std::string(base) + ".ffn_norm.weight", ok);
        b.gate_w = frt_req(tm, std::string(base) + ".gate.weight", ok);
        b.up_w = frt_req(tm, std::string(base) + ".up.weight", ok);
        b.down_w = frt_req(tm, std::string(base) + ".down.weight", ok);
    }
    return ok;
}

static bool frt_bind_ditblocks(const std::map<std::string, ggml_tensor*>& tm, const std::string& prefix, int n_layers,
                               bool with_conv_adaln, std::vector<frt_ditblock>& blocks) {
    blocks.resize(n_layers);
    bool ok = true;
    for (int i = 0; i < n_layers; i++) {
        char base[128];
        std::snprintf(base, sizeof(base), "%s.%d", prefix.c_str(), i);
        auto& b = blocks[i];
        b.norm1_w = frt_req(tm, std::string(base) + ".norm1.weight", ok);
        b.norm2_w = frt_req(tm, std::string(base) + ".norm2.weight", ok);
        b.q_w = frt_req(tm, std::string(base) + ".q.weight", ok);
        b.q_b = frt_req(tm, std::string(base) + ".q.bias", ok);
        b.k_w = frt_req(tm, std::string(base) + ".k.weight", ok);
        b.k_b = frt_req(tm, std::string(base) + ".k.bias", ok);
        b.v_w = frt_req(tm, std::string(base) + ".v.weight", ok);
        b.v_b = frt_req(tm, std::string(base) + ".v.bias", ok);
        b.o_w = frt_req(tm, std::string(base) + ".o.weight", ok);
        b.o_b = frt_req(tm, std::string(base) + ".o.bias", ok);
        b.ffn_up_w = frt_req(tm, std::string(base) + ".ffn_up.weight", ok);
        b.ffn_up_b = frt_req(tm, std::string(base) + ".ffn_up.bias", ok);
        b.ffn_down_w = frt_req(tm, std::string(base) + ".ffn_down.weight", ok);
        b.ffn_down_b = frt_req(tm, std::string(base) + ".ffn_down.bias", ok);
        if (with_conv_adaln) {
            b.norm3_w = frt_req(tm, std::string(base) + ".norm3.weight", ok);
            b.adaln_w = frt_req(tm, std::string(base) + ".adaln.weight", ok);
            b.adaln_b = frt_req(tm, std::string(base) + ".adaln.bias", ok);
            b.conv0_w = frt_req(tm, std::string(base) + ".conv0.weight", ok);
            b.conv0_b = frt_req(tm, std::string(base) + ".conv0.bias", ok);
            b.conv2_w = frt_req(tm, std::string(base) + ".conv2.weight", ok);
            b.conv2_b = frt_req(tm, std::string(base) + ".conv2.bias", ok);
        }
    }
    return ok;
}

extern "C" struct fireredtts3_tts_context_params fireredtts3_tts_context_default_params(void) {
    fireredtts3_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.seed = 0;
    p.n_timesteps = 0;
    p.cfg_scale = -1.0f;
    p.stop_threshold = 0.0f;
    p.max_patches = 0;
    return p;
}

extern "C" struct fireredtts3_tts_context* fireredtts3_tts_init_from_file(
    const char* path_model, struct fireredtts3_tts_context_params params) {
    if (!path_model || !*path_model)
        return nullptr;
    auto* ctx = new fireredtts3_tts_context();
    ctx->params = params;
    if (ctx->params.seed == 0)
        ctx->params.seed = 1234;
    if (ctx->params.n_timesteps <= 0)
        ctx->params.n_timesteps = 10;
    if (ctx->params.cfg_scale < 0.0f)
        ctx->params.cfg_scale = 2.0f;
    if (ctx->params.stop_threshold <= 0.0f)
        ctx->params.stop_threshold = 0.5f;
    if (ctx->params.max_patches <= 0)
        ctx->params.max_patches = 400;

    ctx->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!ctx->backend)
        ctx->backend = core_cpu_backend::init();
    ctx->backend_cpu = core_cpu_backend::init();
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, params.n_threads > 0 ? params.n_threads : 4);
    if (ggml_backend_is_cpu(ctx->backend_cpu))
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads > 0 ? params.n_threads : 4);

    // ---- metadata ----
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            std::fprintf(stderr, "fireredtts3: cannot read %s\n", path_model);
            fireredtts3_tts_free(ctx);
            return nullptr;
        }
        ctx->llm_n_layers = (int)core_gguf::kv_u32(g, "frt.llm.n_layers", 28);
        ctx->llm_d = (int)core_gguf::kv_u32(g, "frt.llm.hidden_size", 2048);
        ctx->llm_heads = (int)core_gguf::kv_u32(g, "frt.llm.n_heads", 16);
        ctx->llm_kv_heads = (int)core_gguf::kv_u32(g, "frt.llm.n_kv_heads", 8);
        ctx->llm_head_dim = (int)core_gguf::kv_u32(g, "frt.llm.head_dim", 128);
        ctx->llm_theta = core_gguf::kv_f32(g, "frt.llm.rope_theta", 1000000.0f);
        ctx->llm_eps = core_gguf::kv_f32(g, "frt.llm.rms_norm_eps", 1e-6f);
        ctx->penc_layers = (int)core_gguf::kv_u32(g, "frt.penc.n_layers", 8);
        ctx->penc_d = (int)core_gguf::kv_u32(g, "frt.penc.hidden_size", 1024);
        ctx->penc_heads = (int)core_gguf::kv_u32(g, "frt.penc.n_heads", 16);
        ctx->dit_layers = (int)core_gguf::kv_u32(g, "frt.dit.n_layers", 11);
        ctx->dit_d = (int)core_gguf::kv_u32(g, "frt.dit.hidden_size", 1024);
        ctx->dit_heads = (int)core_gguf::kv_u32(g, "frt.dit.n_heads", 16);
        ctx->redae_dim = (int)core_gguf::kv_u32(g, "frt.redae_dim", 64);
        ctx->spk_dim = (int)core_gguf::kv_u32(g, "frt.spk_dim", 512);
        ctx->patch_size = (int)core_gguf::kv_u32(g, "frt.patch_size", 4);
        ctx->n_hist_patches = (int)core_gguf::kv_u32(g, "frt.n_history_patches", 2);

        // Tokenizer (newline-joined)
        std::string toks = core_gguf::kv_str(g, "frt.tokenizer.tokens", "");
        std::string merges = core_gguf::kv_str(g, "frt.tokenizer.merges", "");
        if (toks.empty()) {
            std::fprintf(stderr, "fireredtts3: GGUF has no tokenizer\n");
            core_gguf::free_metadata(g);
            fireredtts3_tts_free(ctx);
            return nullptr;
        }
        size_t start = 0;
        while (start <= toks.size()) {
            size_t nl = toks.find('\n', start);
            std::string t = (nl == std::string::npos) ? toks.substr(start) : toks.substr(start, nl - start);
            ctx->token_to_id.emplace(t, (int32_t)ctx->id_to_token.size());
            ctx->id_to_token.push_back(std::move(t));
            if (nl == std::string::npos)
                break;
            start = nl + 1;
        }
        int rank = 0;
        start = 0;
        while (start <= merges.size() && !merges.empty()) {
            size_t nl = merges.find('\n', start);
            std::string mline = (nl == std::string::npos) ? merges.substr(start) : merges.substr(start, nl - start);
            ctx->merge_rank.emplace(std::move(mline), rank++);
            if (nl == std::string::npos)
                break;
            start = nl + 1;
        }
        ctx->tok_sot = (int32_t)core_gguf::kv_u32(g, "frt.token.sot", 0);
        ctx->tok_eot = (int32_t)core_gguf::kv_u32(g, "frt.token.eot", 0);
        ctx->dprompt_text = core_gguf::kv_str(g, "frt.dprompt.text", "");
        ctx->dprompt_lang = core_gguf::kv_str(g, "frt.dprompt.language", "English");
        core_gguf::free_metadata(g);
    }

    // ---- weights ----
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path_model, ctx->backend, "fireredtts3", wl)) {
            fireredtts3_tts_free(ctx);
            return nullptr;
        }
        ctx->core_ctx = wl.ctx;
        ctx->core_buf = wl.buf;
        ctx->core_tensors = std::move(wl.tensors);
    }
    const auto& tm = ctx->core_tensors;
    bool ok = true;
    ctx->tok_emb = frt_req(tm, "frt.llm.tok_emb.weight", ok);
    ctx->llm_norm_w = frt_req(tm, "frt.llm.norm.weight", ok);
    ok = ok && frt_bind_qwen3_blocks(tm, "frt.llm", ctx->llm_n_layers, ctx->llm_blocks);
    ctx->spk_llm_w = frt_req(tm, "frt.spk_proj_llm.weight", ok);
    ctx->spk_llm_b = frt_req(tm, "frt.spk_proj_llm.bias", ok);
    ctx->spk_dit_w = frt_req(tm, "frt.spk_proj_dit.weight", ok);
    ctx->spk_dit_b = frt_req(tm, "frt.spk_proj_dit.bias", ok);
    ctx->penc_cls = frt_req(tm, "frt.penc.cls_tok", ok);
    ctx->penc_in_w = frt_req(tm, "frt.penc.in_proj.weight", ok);
    ctx->penc_in_b = frt_req(tm, "frt.penc.in_proj.bias", ok);
    ok = ok && frt_bind_ditblocks(tm, "frt.penc.blk", ctx->penc_layers, false, ctx->penc_blocks);
    ctx->penc_out_norm_w = frt_req(tm, "frt.penc.out_norm.weight", ok);
    ctx->penc_out_w = frt_req(tm, "frt.penc.out_proj.weight", ok);
    ctx->penc_out_b = frt_req(tm, "frt.penc.out_proj.bias", ok);
    ctx->dit_in_w = frt_req(tm, "frt.dit.in_proj.weight", ok);
    ctx->dit_in_b = frt_req(tm, "frt.dit.in_proj.bias", ok);
    ctx->dit_tm0_w = frt_req(tm, "frt.dit.time_mlp0.weight", ok);
    ctx->dit_tm0_b = frt_req(tm, "frt.dit.time_mlp0.bias", ok);
    ctx->dit_tm2_w = frt_req(tm, "frt.dit.time_mlp2.weight", ok);
    ctx->dit_tm2_b = frt_req(tm, "frt.dit.time_mlp2.bias", ok);
    ok = ok && frt_bind_ditblocks(tm, "frt.dit.blk", ctx->dit_layers, true, ctx->dit_blocks);
    ctx->dit_final_adaln_w = frt_req(tm, "frt.dit.final_adaln.weight", ok);
    ctx->dit_final_adaln_b = frt_req(tm, "frt.dit.final_adaln.bias", ok);
    ctx->dit_final_w = frt_req(tm, "frt.dit.final_proj.weight", ok);
    ctx->dit_final_b = frt_req(tm, "frt.dit.final_proj.bias", ok);
    ctx->dit_head_w = frt_req(tm, "frt.dit_head.weight", ok);
    ctx->dit_head_b = frt_req(tm, "frt.dit_head.bias", ok);
    {
        ggml_tensor* sw = frt_req(tm, "frt.stop_head.weight", ok);
        ggml_tensor* sb = frt_req(tm, "frt.stop_head.bias", ok);
        if (sw && sb) {
            frt_read_f32(sw, ctx->stop_w);
            std::vector<float> b1;
            frt_read_f32(sb, b1);
            ctx->stop_b = b1.empty() ? 0.0f : b1[0];
        }
    }
    // Baked default prompt (optional)
    if (ggml_tensor* dl = frt_get(tm, "frt.dprompt.latents")) {
        frt_read_f32(dl, ctx->dprompt_latents);
        ctx->dprompt_T = (int)dl->ne[1];
        if (ggml_tensor* ds = frt_get(tm, "frt.dprompt.spk_emb"))
            frt_read_f32(ds, ctx->dprompt_spk);
    }
    if (!ok) {
        fireredtts3_tts_free(ctx);
        return nullptr;
    }

    // sched + compute scratch
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1)
        std::fprintf(stderr,
                     "fireredtts3: core loaded (llm %dL d=%d, penc %dL, dit %dL, vocab %zu, default prompt %s)\n",
                     ctx->llm_n_layers, ctx->llm_d, ctx->penc_layers, ctx->dit_layers, ctx->id_to_token.size(),
                     ctx->dprompt_T > 0 ? "baked" : "absent");
    return ctx;
}

extern "C" int fireredtts3_tts_set_redae_path(struct fireredtts3_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return 1;
    if (ctx->redae_ctx) {
        std::fprintf(stderr, "fireredtts3: redae already loaded\n");
        return 1;
    }
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            std::fprintf(stderr, "fireredtts3: cannot read %s\n", path);
            return 1;
        }
        ctx->ae_patch = (int)core_gguf::kv_u32(g, "frt.audio_patch_size", 480);
        ctx->ae_sr = (int)core_gguf::kv_u32(g, "frt.sample_rate", 24000);
        ctx->enc_layers = (int)core_gguf::kv_u32(g, "frt.enc.n_layers", 18);
        ctx->dec_layers = (int)core_gguf::kv_u32(g, "frt.dec.n_layers", 18);
        ctx->encds_layers = (int)core_gguf::kv_u32(g, "frt.encds.n_layers", 4);
        ctx->ae_d = (int)core_gguf::kv_u32(g, "frt.enc.hidden_size", 896);
        ctx->ae_heads = (int)core_gguf::kv_u32(g, "frt.enc.n_heads", 14);
        ctx->ae_kv_heads = (int)core_gguf::kv_u32(g, "frt.enc.n_kv_heads", 2);
        ctx->ae_window = (int)core_gguf::kv_u32(g, "frt.enc.sliding_window", 64);
        ctx->ae_ds_rate = (int)core_gguf::kv_u32(g, "frt.enc.extra_downsample_rate", 2);
        ctx->ae_head_dim = (int)core_gguf::kv_u32(g, "frt.redae.head_dim", 128);
        ctx->ae_theta = core_gguf::kv_f32(g, "frt.redae.rope_theta", 10000.0f);
        ctx->ae_eps = core_gguf::kv_f32(g, "frt.redae.rms_norm_eps", 1e-6f);
        core_gguf::free_metadata(g);
    }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "fireredtts3:redae", wl))
        return 1;
    ctx->redae_ctx = wl.ctx;
    ctx->redae_buf = wl.buf;
    ctx->redae_tensors = std::move(wl.tensors);

    const auto& tm = ctx->redae_tensors;
    bool ok = true;
    ctx->enc_in0_w = frt_req(tm, "frt.enc.in_proj0.weight", ok);
    ctx->enc_in0_b = frt_req(tm, "frt.enc.in_proj0.bias", ok);
    ctx->enc_in1_w = frt_req(tm, "frt.enc.in_proj1.weight", ok);
    ctx->enc_in1_b = frt_req(tm, "frt.enc.in_proj1.bias", ok);
    ok = ok && frt_bind_qwen3_blocks(tm, "frt.enc", ctx->enc_layers, ctx->enc_blocks);
    ctx->enc_norm_w = frt_req(tm, "frt.enc.norm.weight", ok);
    ctx->encds_cls = frt_req(tm, "frt.encds.cls_tok", ok);
    ok = ok && frt_bind_qwen3_blocks(tm, "frt.encds", ctx->encds_layers, ctx->encds_blocks);
    ctx->encds_norm_w = frt_req(tm, "frt.encds.norm.weight", ok);
    ctx->enc_out_w = frt_req(tm, "frt.enc.out_proj.weight", ok);
    ctx->enc_out_b = frt_req(tm, "frt.enc.out_proj.bias", ok);
    ctx->dec_in_w = frt_req(tm, "frt.dec.in_proj.weight", ok);
    ctx->dec_in_b = frt_req(tm, "frt.dec.in_proj.bias", ok);
    ok = ok && frt_bind_qwen3_blocks(tm, "frt.dec", ctx->dec_layers, ctx->dec_blocks);
    ctx->dec_norm_w = frt_req(tm, "frt.dec.norm.weight", ok);
    ctx->dec_istft_w = frt_req(tm, "frt.dec.istft_out.weight", ok);
    ctx->dec_istft_b = frt_req(tm, "frt.dec.istft_out.bias", ok);
    if (ggml_tensor* w = frt_get(tm, "frt.dec.istft_window"))
        frt_read_f32(w, ctx->istft_window);

    // CAM++ (campplus.* raw torch names; same binder layout as confucius4)
    {
        auto T = [&](const std::string& n) { return frt_get(tm, n); };
        auto& m = ctx->campp;
        auto bind_unit = [&](cb_campplus_unit& u, const std::string& base) {
            u.lin_w = T(base + ".linear.weight");
            u.lin_b = T(base + ".linear.bias");
            u.bn_w = T(base + ".nonlinear.batchnorm.weight");
            u.bn_b = T(base + ".nonlinear.batchnorm.bias");
            u.bn_m = T(base + ".nonlinear.batchnorm.running_mean");
            u.bn_v = T(base + ".nonlinear.batchnorm.running_var");
        };
        auto bind_res = [&](cb_campplus_resblock& b, const std::string& base) {
            b.conv1_w = T(base + ".conv1.weight");
            b.bn1_w = T(base + ".bn1.weight");
            b.bn1_b = T(base + ".bn1.bias");
            b.bn1_m = T(base + ".bn1.running_mean");
            b.bn1_v = T(base + ".bn1.running_var");
            b.conv2_w = T(base + ".conv2.weight");
            b.bn2_w = T(base + ".bn2.weight");
            b.bn2_b = T(base + ".bn2.bias");
            b.bn2_m = T(base + ".bn2.running_mean");
            b.bn2_v = T(base + ".bn2.running_var");
            b.sc_w = T(base + ".shortcut.0.weight");
            b.sc_bn_w = T(base + ".shortcut.1.weight");
            b.sc_bn_b = T(base + ".shortcut.1.bias");
            b.sc_bn_m = T(base + ".shortcut.1.running_mean");
            b.sc_bn_v = T(base + ".shortcut.1.running_var");
        };
        auto bind_dense_layer = [&](cb_campplus_dense_layer& l, const std::string& base) {
            l.nonl1_bn_w = T(base + ".nonlinear1.batchnorm.weight");
            l.nonl1_bn_b = T(base + ".nonlinear1.batchnorm.bias");
            l.nonl1_bn_m = T(base + ".nonlinear1.batchnorm.running_mean");
            l.nonl1_bn_v = T(base + ".nonlinear1.batchnorm.running_var");
            l.l1_w = T(base + ".linear1.weight");
            l.nonl2_bn_w = T(base + ".nonlinear2.batchnorm.weight");
            l.nonl2_bn_b = T(base + ".nonlinear2.batchnorm.bias");
            l.nonl2_bn_m = T(base + ".nonlinear2.batchnorm.running_mean");
            l.nonl2_bn_v = T(base + ".nonlinear2.batchnorm.running_var");
            l.cam_ll_w = T(base + ".cam_layer.linear_local.weight");
            l.cam_l1_w = T(base + ".cam_layer.linear1.weight");
            l.cam_l1_b = T(base + ".cam_layer.linear1.bias");
            l.cam_l2_w = T(base + ".cam_layer.linear2.weight");
            l.cam_l2_b = T(base + ".cam_layer.linear2.bias");
        };
        const std::string P = "campp";
        auto& head = m.head;
        head.conv1_w = T(P + ".head.conv1.weight");
        if (head.conv1_w) {
            head.bn1_w = T(P + ".head.bn1.weight");
            head.bn1_b = T(P + ".head.bn1.bias");
            head.bn1_m = T(P + ".head.bn1.running_mean");
            head.bn1_v = T(P + ".head.bn1.running_var");
            head.conv2_w = T(P + ".head.conv2.weight");
            head.bn2_w = T(P + ".head.bn2.weight");
            head.bn2_b = T(P + ".head.bn2.bias");
            head.bn2_m = T(P + ".head.bn2.running_mean");
            head.bn2_v = T(P + ".head.bn2.running_var");
            head.layer1.assign(2, cb_campplus_resblock{});
            head.layer2.assign(2, cb_campplus_resblock{});
            for (int i = 0; i < 2; i++) {
                bind_res(head.layer1[i], P + ".head.layer1." + std::to_string(i));
                bind_res(head.layer2[i], P + ".head.layer2." + std::to_string(i));
            }
            head.layer1[0].stride = 2;
            head.layer2[0].stride = 2;
            bind_unit(m.tdnn, P + ".xvector.tdnn");
            bind_unit(m.transit1, P + ".xvector.transit1");
            bind_unit(m.transit2, P + ".xvector.transit2");
            bind_unit(m.transit3, P + ".xvector.transit3");
            m.out_nl.bn_w = T(P + ".xvector.out_nonlinear.batchnorm.weight");
            m.out_nl.bn_b = T(P + ".xvector.out_nonlinear.batchnorm.bias");
            m.out_nl.bn_m = T(P + ".xvector.out_nonlinear.batchnorm.running_mean");
            m.out_nl.bn_v = T(P + ".xvector.out_nonlinear.batchnorm.running_var");
            m.dense.lin_w = T(P + ".xvector.dense.linear.weight");
            m.dense.lin_b = T(P + ".xvector.dense.linear.bias");
            m.dense.bn_m = T(P + ".xvector.dense.nonlinear.batchnorm.running_mean");
            m.dense.bn_v = T(P + ".xvector.dense.nonlinear.batchnorm.running_var");
            const int nlayers[3] = {12, 24, 16};
            cb_campplus_dense_block* blocks[3] = {&m.block1, &m.block2, &m.block3};
            const int dils[3] = {1, 2, 2};
            for (int bi = 0; bi < 3; bi++) {
                blocks[bi]->layers.assign(nlayers[bi], cb_campplus_dense_layer{});
                blocks[bi]->dilation = dils[bi];
                for (int li = 0; li < nlayers[bi]; li++) {
                    bind_dense_layer(blocks[bi]->layers[li],
                                     P + ".xvector.block" + std::to_string(bi + 1) + ".tdnnd" + std::to_string(li + 1));
                }
            }
            ctx->has_campp = true;
        }
    }
    if (!ok) {
        std::fprintf(stderr, "fireredtts3: redae bind failed\n");
        return 1;
    }
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "fireredtts3: redae loaded (enc %dL + ds %dL, dec %dL, d=%d, sw=%d, campp=%d)\n",
                     ctx->enc_layers, ctx->encds_layers, ctx->dec_layers, ctx->ae_d, ctx->ae_window,
                     (int)ctx->has_campp);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Voice
// ─────────────────────────────────────────────────────────────────────────────

// Pad (LEFT) to a multiple, resample, encode, CAM++ — the prompt path of
// FireRedTTS3Base.generate.
extern "C" int fireredtts3_tts_set_voice_pcm(struct fireredtts3_tts_context* ctx, const float* pcm_16k, int n_samples) {
    if (!ctx || !pcm_16k || n_samples <= 0)
        return 1;
    if (!ctx->redae_ctx) {
        std::fprintf(stderr, "fireredtts3: set_voice requires the redae companion (set_redae_path first)\n");
        return 1;
    }
    // 16 k → 24 k
    std::vector<float> pcm24 = core_audio::resample_polyphase(pcm_16k, n_samples, 16000, ctx->ae_sr);
    // LEFT-pad to a multiple of downsample_rate * patch_size (960 * 4)
    const int mult = ctx->ae_patch * ctx->ae_ds_rate * ctx->patch_size;
    const int target = (int)((pcm24.size() + mult - 1) / mult) * mult;
    if ((int)pcm24.size() < target)
        pcm24.insert(pcm24.begin(), target - pcm24.size(), 0.0f);

    int T_lat = 0;
    if (!frt_redae_encode(ctx, pcm24.data(), (int)pcm24.size(), ctx->voice_latents, T_lat, nullptr))
        return 1;
    ctx->voice_T = T_lat;

    // CAM++ x-vector of the PADDED prompt (python: campp resamples 24 k→16 k
    // internally; we resample the padded 24 k buffer back to 16 k).
    if (!ctx->has_campp) {
        std::fprintf(stderr, "fireredtts3: no campplus in redae GGUF\n");
        return 1;
    }
    std::vector<float> pcm16 = core_audio::resample_polyphase(pcm24.data(), (int)pcm24.size(), ctx->ae_sr, 16000);
    ctx->voice_spk = chatterbox_campplus::embed_speaker(ctx->campp, ctx->campp_rt, pcm16.data(), (int)pcm16.size(),
                                                        /*stats_var_floor=*/0.0f);
    if (ctx->voice_spk.size() != (size_t)ctx->spk_dim) {
        std::fprintf(stderr, "fireredtts3: campp returned %zu dims (want %d)\n", ctx->voice_spk.size(), ctx->spk_dim);
        return 1;
    }
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "fireredtts3: voice set (%d latent frames)\n", ctx->voice_T);
    return 0;
}

extern "C" int fireredtts3_tts_set_ref_text(struct fireredtts3_tts_context* ctx, const char* text) {
    if (!ctx || !text)
        return 1;
    ctx->ref_text = text;
    return 0;
}

extern "C" int fireredtts3_tts_set_language(struct fireredtts3_tts_context* ctx, const char* language) {
    if (!ctx || !language)
        return 1;
    ctx->language = language;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation (FireRedTTS3BaseCore.generate)
// ─────────────────────────────────────────────────────────────────────────────

// Small helper: run a single linear projection graph (weight may be F16).
static bool frt_project(fireredtts3_tts_context* ctx, ggml_tensor* w, ggml_tensor* b, const float* in, int in_dim,
                        std::vector<float>& out) {
    frt_graph g = frt_graph_begin(ctx);
    ggml_tensor* x = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_F32, in_dim);
    ggml_set_input(x);
    ggml_set_name(x, "x");
    ggml_tensor* y = frt_linear(g.ctx0, w, b, x);
    ggml_set_name(y, "y");
    ggml_set_output(y);
    ggml_build_forward_expand(g.gf, y);
    if (!frt_graph_alloc(ctx, g)) {
        frt_graph_end(g);
        return false;
    }
    ggml_backend_tensor_set(x, in, 0, (size_t)in_dim * sizeof(float));
    if (!frt_graph_compute(ctx, g))
        return false;
    out.resize(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    frt_graph_end(g);
    return true;
}

// Token-embedding lookup (F16 table on device).
static bool frt_embed_tokens(fireredtts3_tts_context* ctx, const std::vector<int32_t>& ids, std::vector<float>& out) {
    frt_graph g = frt_graph_begin(ctx);
    const int T = (int)ids.size();
    ggml_tensor* idx = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, T);
    ggml_set_input(idx);
    ggml_set_name(idx, "ids");
    ggml_tensor* e = ggml_get_rows(g.ctx0, ctx->tok_emb, idx); // (2048, T) F32
    ggml_set_name(e, "embeds");
    ggml_set_output(e);
    ggml_build_forward_expand(g.gf, e);
    if (!frt_graph_alloc(ctx, g)) {
        frt_graph_end(g);
        return false;
    }
    ggml_backend_tensor_set(idx, ids.data(), 0, (size_t)T * sizeof(int32_t));
    if (!frt_graph_compute(ctx, g))
        return false;
    out.resize((size_t)ctx->llm_d * T);
    ggml_backend_tensor_get(e, out.data(), 0, out.size() * sizeof(float));
    frt_graph_end(g);
    return true;
}

struct frt_gen_debug {
    std::vector<float> prefill_embeds;
    std::vector<float> llm_prefill_out;
    std::vector<float> stop_scores;
    std::vector<float> latents_steps; // n_steps * 4 * 64
    std::vector<float> noise_used;    // n_steps * 4 * 64
    int T_prefill = 0;
};

// Runs the AR loop. prompt latents (T_lat*64) + spk (512) + token ids.
// Returns all latents (prompt + generated), frame-major.
static bool frt_generate(fireredtts3_tts_context* ctx, const std::vector<int32_t>& ids,
                         const std::vector<float>& prompt_latents, int T_prompt, const std::vector<float>& spk,
                         std::vector<float>& latents_all, frt_gen_debug* dbg) {
    frt_bench_stage bench_("generate");
    const int ps = ctx->patch_size;
    const int hl = ctx->n_hist_patches * ps; // 8
    const int ld = ctx->redae_dim;
    const int P = T_prompt / ps;

    // Noise replay (diff harness): FIREREDTTS3_NOISE = raw f32 file of
    // n_steps*(ps*ld) draws, consumed sequentially.
    std::vector<float> noise_file;
    size_t noise_pos = 0;
    if (const char* nf = std::getenv("FIREREDTTS3_NOISE")) {
        FILE* f = std::fopen(nf, "rb");
        if (f) {
            std::fseek(f, 0, SEEK_END);
            long sz = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            noise_file.resize(sz / sizeof(float));
            size_t rd = std::fread(noise_file.data(), sizeof(float), noise_file.size(), f);
            noise_file.resize(rd);
            std::fclose(f);
            if (ctx->params.verbosity >= 1)
                std::fprintf(stderr, "fireredtts3: noise replay from %s (%zu floats)\n", nf, noise_file.size());
        }
    }
    if (!ctx->rng_seeded) {
        crispasr::core::mt19937_seed(ctx->rng, (uint32_t)ctx->params.seed);
        ctx->rng_seeded = true;
    }

    // Projections
    std::vector<float> spk_llm, spk_dit;
    if (!frt_project(ctx, ctx->spk_llm_w, ctx->spk_llm_b, spk.data(), ctx->spk_dim, spk_llm))
        return false;
    if (!frt_project(ctx, ctx->spk_dit_w, ctx->spk_dit_b, spk.data(), ctx->spk_dim, spk_dit))
        return false;

    std::vector<float> text_embeds;
    if (!frt_embed_tokens(ctx, ids, text_embeds))
        return false;

    std::vector<float> penc_prompt; // (P * 2048)
    if (!frt_penc_forward(ctx, prompt_latents.data(), T_prompt, penc_prompt))
        return false;

    // embeds = [spk_llm, text, penc_prompt] (rows of llm_d)
    const int d = ctx->llm_d;
    std::vector<float> embeds;
    embeds.reserve((size_t)(1 + ids.size() + P + ctx->params.max_patches) * d);
    embeds.insert(embeds.end(), spk_llm.begin(), spk_llm.end());
    embeds.insert(embeds.end(), text_embeds.begin(), text_embeds.end());
    embeds.insert(embeds.end(), penc_prompt.begin(), penc_prompt.end());
    if (dbg) {
        dbg->prefill_embeds = embeds;
        dbg->T_prefill = (int)embeds.size() / d;
    }

    // latents_gen = zeros(hl) + prompt (python F.pad left in time)
    latents_all.assign((size_t)hl * ld, 0.0f);
    latents_all.insert(latents_all.end(), prompt_latents.begin(), prompt_latents.end());

    // backbone_cond: rolling raw-hidden entries (start = 2 zero rows)
    std::vector<float> cond_hist((size_t)ctx->n_hist_patches * d, 0.0f);

    const float stop_thr = ctx->params.stop_threshold;
    const int min_steps = 6;
    int n_steps = 0;

    for (int step = 0; step < ctx->params.max_patches; step++) {
        const int T = (int)embeds.size() / d;
        std::vector<float> hidden;
        if (!frt_llm_forward(ctx, embeds.data(), T, hidden))
            return false;
        if (dbg && step == 0)
            dbg->llm_prefill_out = hidden;

        // stop head on last hidden
        const float* h_last = hidden.data() + (size_t)(T - 1) * d;
        double logit = ctx->stop_b;
        for (int i = 0; i < d; i++)
            logit += (double)ctx->stop_w[i] * h_last[i];
        const float score = 1.0f / (1.0f + std::exp(-(float)logit));
        if (dbg)
            dbg->stop_scores.push_back(score);
        if (score >= stop_thr && step >= min_steps)
            break;

        // append backbone cond entries (step 0: ALL P prompt-patch positions)
        if (step == 0) {
            for (int p = 0; p < P; p++) {
                const float* row = hidden.data() + (size_t)(T - P + p) * d;
                cond_hist.insert(cond_hist.end(), row, row + d);
            }
        } else {
            cond_hist.insert(cond_hist.end(), h_last, h_last + d);
        }
        // last 3 raw cond rows
        const int nc = ctx->n_hist_patches + 1;
        const float* cond3 = cond_hist.data() + cond_hist.size() - (size_t)nc * d;

        // noise for the current patch
        std::vector<float> noise((size_t)ps * ld);
        if (noise_pos + noise.size() <= noise_file.size()) {
            std::memcpy(noise.data(), noise_file.data() + noise_pos, noise.size() * sizeof(float));
            noise_pos += noise.size();
        } else {
            crispasr::core::fill_gaussian_noise(noise.data(), (int)noise.size(), ctx->rng);
        }
        if (dbg)
            dbg->noise_used.insert(dbg->noise_used.end(), noise.begin(), noise.end());

        // flow: history = last hl frames of latents_all
        const float* hist = latents_all.data() + latents_all.size() - (size_t)hl * ld;
        std::vector<float> patch;
        if (!frt_flow_solve(ctx, hist, noise.data(), cond3, spk_dit.data(), patch))
            return false;
        if (dbg)
            dbg->latents_steps.insert(dbg->latents_steps.end(), patch.begin(), patch.end());
        latents_all.insert(latents_all.end(), patch.begin(), patch.end());

        // patch → next LLM embedding
        std::vector<float> pe;
        if (!frt_penc_forward(ctx, patch.data(), ps, pe))
            return false;
        embeds.insert(embeds.end(), pe.begin(), pe.end());
        n_steps++;
        if (ctx->params.verbosity >= 2)
            std::fprintf(stderr, "fireredtts3: step %d stop=%.3f\n", step, score);
    }
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "fireredtts3: generated %d patches (%.2f s)\n", n_steps,
                     n_steps * ps * (float)ctx->ae_patch * ctx->ae_ds_rate / (float)ctx->ae_sr);

    // strip the leading zero history
    latents_all.erase(latents_all.begin(), latents_all.begin() + (size_t)hl * ld);
    return true;
}

extern "C" float* fireredtts3_tts_synthesize(struct fireredtts3_tts_context* ctx, const char* text,
                                             int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    if (!ctx->redae_ctx) {
        std::fprintf(stderr, "fireredtts3: synthesize requires the redae companion GGUF\n");
        return nullptr;
    }

    // Resolve prompt
    const std::vector<float>* lat = nullptr;
    const std::vector<float>* spk = nullptr;
    int T_prompt = 0;
    std::string ref_text, lang;
    if (!ctx->voice_latents.empty()) {
        lat = &ctx->voice_latents;
        spk = &ctx->voice_spk;
        T_prompt = ctx->voice_T;
        ref_text = ctx->ref_text;
        if (ref_text.empty()) {
            std::fprintf(stderr, "fireredtts3: a --voice prompt needs its transcript (set_ref_text / --ref-text)\n");
            return nullptr;
        }
    } else if (ctx->dprompt_T > 0) {
        lat = &ctx->dprompt_latents;
        spk = &ctx->dprompt_spk;
        T_prompt = ctx->dprompt_T;
        ref_text = ctx->dprompt_text;
    } else {
        std::fprintf(stderr, "fireredtts3: no voice prompt (no baked default; use set_voice_pcm)\n");
        return nullptr;
    }
    lang = !ctx->language.empty() ? ctx->language : ctx->dprompt_lang;

    std::vector<int32_t> ids;
    if (!frt_tokenize(ctx, lang, ref_text + std::string(text), ids))
        return nullptr;
    if (ctx->params.verbosity >= 1)
        std::fprintf(stderr, "fireredtts3: %zu text tokens (lang %s)\n", ids.size(), lang.c_str());

    std::vector<float> latents_all;
    ctx->rng_seeded = false; // fresh seed per synthesis (upstream fix_seed per call)
    if (!frt_generate(ctx, ids, *lat, T_prompt, *spk, latents_all, nullptr))
        return nullptr;

    const int T_all = (int)(latents_all.size() / ctx->redae_dim);
    std::vector<float> pcm;
    if (!frt_redae_decode(ctx, latents_all.data(), T_all, pcm, nullptr))
        return nullptr;

    // trim the prompt samples (prompt_latents cover T_prompt*960 samples)
    const size_t trim = (size_t)T_prompt * ctx->ae_patch * ctx->ae_ds_rate;
    if (trim >= pcm.size()) {
        std::fprintf(stderr, "fireredtts3: nothing generated beyond the prompt\n");
        return nullptr;
    }
    const size_t n = pcm.size() - trim;
    float* out = (float*)std::malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, pcm.data() + trim, n * sizeof(float));
    *out_n_samples = (int)n;
    return out;
}

extern "C" void fireredtts3_tts_pcm_free(float* pcm) {
    std::free(pcm);
}

extern "C" void fireredtts3_tts_set_seed(struct fireredtts3_tts_context* ctx, uint64_t seed) {
    if (!ctx || seed == 0)
        return;
    ctx->params.seed = seed;
    ctx->rng_seeded = false;
}

extern "C" int fireredtts3_tts_sample_rate(const struct fireredtts3_tts_context* ctx) {
    return ctx ? ctx->ae_sr : 24000;
}

extern "C" void fireredtts3_tts_free(struct fireredtts3_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->redae_buf)
        core_gguf::release_weight_buffer(ctx->redae_buf);
    if (ctx->redae_ctx)
        ggml_free(ctx->redae_ctx);
    if (ctx->core_buf)
        core_gguf::release_weight_buffer(ctx->core_buf);
    if (ctx->core_ctx)
        ggml_free(ctx->core_ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Self-contained per-stage diff runner (crispasr-diff "fireredtts3").
// Compares against tools/reference_backends/fireredtts3_tts.py output.
// Prints cos, |mine|, |ref| per stage (HARD RULE #2b: magnitudes ALWAYS).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct frt_cmp_result {
    double cos = 0.0, nmine = 0.0, nref = 0.0, max_abs = 0.0;
    bool pass = false;
};

frt_cmp_result frt_compare(const float* mine, const float* ref, size_t n, double cos_thr) {
    frt_cmp_result r;
    double dot = 0, na = 0, nb = 0, ma = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)mine[i] * ref[i];
        na += (double)mine[i] * mine[i];
        nb += (double)ref[i] * ref[i];
        double d = std::fabs((double)mine[i] - ref[i]);
        if (d > ma)
            ma = d;
    }
    r.nmine = std::sqrt(na);
    r.nref = std::sqrt(nb);
    r.cos = (na > 0 && nb > 0) ? dot / (r.nmine * r.nref) : (na == nb ? 1.0 : 0.0);
    r.max_abs = ma;
    // scale check: magnitudes within 2% (cosine is scale-blind)
    const double mag_ratio = (r.nref > 0) ? r.nmine / r.nref : (r.nmine == 0 ? 1.0 : 1e9);
    r.pass = r.cos >= cos_thr && mag_ratio > 0.98 && mag_ratio < 1.02;
    return r;
}

bool frt_ref_tensor(const std::map<std::string, ggml_tensor*>& rm, const char* name, std::vector<float>& out,
                    ggml_tensor** t_out = nullptr) {
    auto it = rm.find(name);
    if (it == rm.end())
        return false;
    frt_read_f32(it->second, out);
    if (t_out)
        *t_out = it->second;
    return true;
}

int frt_report(const char* stage, const frt_cmp_result& r, int& fails) {
    std::fprintf(stderr, "[%s] %-16s cos=%.6f  |mine|=%.4g  |ref|=%.4g  max_abs=%.4g\n", r.pass ? "PASS" : "FAIL",
                 stage, r.cos, r.nmine, r.nref, r.max_abs);
    if (!r.pass)
        fails++;
    return r.pass ? 0 : 1;
}

} // namespace

extern "C" int fireredtts3_tts_diff(const char* core_gguf, const char* redae_gguf, const char* ref_gguf,
                                    const char* prompt_wav_path, int verbosity) {
    fireredtts3_tts_context_params p = fireredtts3_tts_context_default_params();
    p.verbosity = verbosity;
    p.use_gpu = core_env::on("CRISPASR_DIFF_GPU");
    fireredtts3_tts_context* ctx = fireredtts3_tts_init_from_file(core_gguf, p);
    if (!ctx)
        return 2;
    if (fireredtts3_tts_set_redae_path(ctx, redae_gguf) != 0) {
        fireredtts3_tts_free(ctx);
        return 2;
    }

    // Reference archive
    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend_cpu, "frt-ref", rw)) {
        std::fprintf(stderr, "fireredtts3_diff: cannot load ref %s\n", ref_gguf);
        fireredtts3_tts_free(ctx);
        return 2;
    }
    const auto& rm = rw.tensors;
    int fails = 0;

    // ── prompt audio: 16 k mono WAV, resample + LEFT pad like generate() ──
    std::vector<float> wav16;
    int wav_sr = 0;
    if (!crispasr::core::read_wav_mono_pcm16(prompt_wav_path, wav16, wav_sr) || wav16.empty()) {
        std::fprintf(stderr, "fireredtts3_diff: cannot read %s\n", prompt_wav_path);
        fireredtts3_tts_free(ctx);
        return 2;
    }
    if (wav_sr != 16000)
        wav16 = core_audio::resample_polyphase(wav16.data(), (int)wav16.size(), wav_sr, 16000);
    std::vector<float> pcm24 = core_audio::resample_polyphase(wav16.data(), (int)wav16.size(), 16000, ctx->ae_sr);
    const int mult = ctx->ae_patch * ctx->ae_ds_rate * ctx->patch_size;
    const int target = (int)((pcm24.size() + mult - 1) / mult) * mult;
    if ((int)pcm24.size() < target)
        pcm24.insert(pcm24.begin(), target - pcm24.size(), 0.0f);

    // ── stage: text_tokens ──
    {
        const char* syn = std::getenv("FIREREDTTS3_SYN_TEXT");
        const char* ref_text = std::getenv("FIREREDTTS3_PROMPT_TEXT");
        const char* lang = std::getenv("FIREREDTTS3_LANG");
        std::string body = std::string(ref_text ? ref_text
                                                : "And so my fellow Americans ask not what your country can do "
                                                  "for you, ask what you can do for your country.") +
                           (syn ? syn : "Hello there, how are you today?");
        std::vector<int32_t> ids;
        std::vector<float> ref_ids;
        if (frt_tokenize(ctx, lang ? lang : "English", body, ids) && frt_ref_tensor(rm, "text_tokens", ref_ids)) {
            bool same = ids.size() == ref_ids.size();
            int first_bad = -1;
            for (size_t i = 0; same && i < ids.size(); i++)
                if ((float)ids[i] != ref_ids[i]) {
                    same = false;
                    first_bad = (int)i;
                }
            std::fprintf(stderr, "[%s] text_tokens      n=%zu ref_n=%zu%s\n", same ? "PASS" : "FAIL", ids.size(),
                         ref_ids.size(),
                         first_bad >= 0 ? (" first_mismatch=" + std::to_string(first_bad)).c_str() : "");
            if (!same)
                fails++;
        }
    }

    // ── stage: campp fbank + spk_emb (fbank first, to split fbank-vs-xvector
    // when the embedding diverges; then the xvector on the REF fbank so the
    // pooling/dense verdict is independent of fbank drift) ──
    std::vector<float> ref_spk, ref_fbank;
    const bool have_ref_fbank = frt_ref_tensor(rm, "campp_fbank", ref_fbank);
    if (frt_ref_tensor(rm, "spk_emb", ref_spk)) {
        std::vector<float> pcm16 = core_audio::resample_polyphase(pcm24.data(), (int)pcm24.size(), ctx->ae_sr, 16000);
        if (have_ref_fbank) {
            int Tf = 0;
            auto fb = chatterbox_campplus::compute_fbank(pcm16.data(), (int)pcm16.size(), Tf);
            std::fprintf(stderr, "         campp_fbank T_mine=%d T_ref=%zu\n", Tf, ref_fbank.size() / 80);
            size_t n = std::min(fb.size(), ref_fbank.size());
            // campp_fbank gate is 0.995, not 0.999, and the reason is the
            // RESAMPLER, not the fbank. We reach 16 kHz with
            // core_audio::resample_polyphase; the reference dumper uses
            // torchaudio.functional.resample. Running one clip through both and
            // then through the IDENTICAL kaldi fbank gives cos 0.998250 with a
            // 0.4% magnitude difference — the same order as the 0.997589 /
            // 0.64% measured here (validate v5).
            //
            // The alternative explanation was tested and ruled out: unscaled
            // vs int16-scaled input costs cos 0.952 / 11% magnitude, an order
            // larger, and the reference feeds torchaudio's [-1,1] floats, so
            // compute_fbank's int16_scale=false is correct.
            //
            // 0.995 still separates the cases it needs to: a scaling or
            // convention error lands near 0.95 and a layout error near 0.
            // The downstream cost is bounded by the two arms below — spk_emb on
            // OUR fbank scores 0.9995 against 0.99999 on the oracle fbank, i.e.
            // the resampler delta is worth 0.0005 of embedding cosine — and
            // both of those keep their own gates.
            if (n > 0)
                frt_report("campp_fbank", frt_compare(fb.data(), ref_fbank.data(), n, 0.995), fails);
            // xvector on the REFERENCE fbank (oracle-in)
            std::vector<float> spk_o = chatterbox_campplus::compute_xvector(ctx->campp, ctx->campp_rt, ref_fbank.data(),
                                                                            (int)(ref_fbank.size() / 80), 0.0f);
            if (spk_o.size() == ref_spk.size())
                frt_report("spk_emb_orafb", frt_compare(spk_o.data(), ref_spk.data(), spk_o.size(), 0.995), fails);
        }
        std::vector<float> spk =
            chatterbox_campplus::embed_speaker(ctx->campp, ctx->campp_rt, pcm16.data(), (int)pcm16.size(), 0.0f);
        if (spk.size() == ref_spk.size())
            frt_report("spk_emb", frt_compare(spk.data(), ref_spk.data(), spk.size(), 0.995), fails);
    }

    // ── stage: RedAE encode (enc_hidden + prompt_latents) ──
    std::vector<float> my_latents, my_enc_hidden;
    int T_lat = 0;
    if (!frt_redae_encode(ctx, pcm24.data(), (int)pcm24.size(), my_latents, T_lat, &my_enc_hidden)) {
        fireredtts3_tts_free(ctx);
        return 2;
    }
    std::vector<float> ref_lat, ref_ench;
    // enc_hidden gate is 0.995, not 0.999: the F16 18-layer stack reads global
    // cos 0.9955 with |mine|/|ref| within 0.03% (649.0 vs 649.2, validate v2)
    // while its LOAD-BEARING product prompt_latents — 4 more layers plus the
    // 896-to-64 projection downstream — passes its own 0.999 gate at 0.9995.
    // The residue is F16 rounding concentrated in components the projection
    // discards (mostly the left-padded silence region), not structure.
    if (frt_ref_tensor(rm, "enc_hidden", ref_ench) && ref_ench.size() == my_enc_hidden.size())
        frt_report("enc_hidden", frt_compare(my_enc_hidden.data(), ref_ench.data(), ref_ench.size(), 0.995), fails);
    bool have_ref_lat = frt_ref_tensor(rm, "prompt_latents", ref_lat);
    if (have_ref_lat && ref_lat.size() == my_latents.size())
        frt_report("prompt_latents", frt_compare(my_latents.data(), ref_lat.data(), ref_lat.size(), 0.999), fails);

    // From here on run on the REFERENCE latents/spk (oracle-in) so stage
    // verdicts are independent of upstream-stage drift.
    const std::vector<float>& lat_in = have_ref_lat ? ref_lat : my_latents;
    const int T_in = have_ref_lat ? (int)(ref_lat.size() / ctx->redae_dim) : T_lat;
    const std::vector<float>& spk_in = ref_spk;
    if (spk_in.empty()) {
        std::fprintf(stderr, "fireredtts3_diff: no spk_emb in ref — cannot continue\n");
        fireredtts3_tts_free(ctx);
        return 2;
    }

    // ── stage: spk projections ──
    std::vector<float> ref_v;
    {
        std::vector<float> v;
        if (frt_project(ctx, ctx->spk_llm_w, ctx->spk_llm_b, spk_in.data(), ctx->spk_dim, v) &&
            frt_ref_tensor(rm, "spk_llm", ref_v) && ref_v.size() == v.size())
            frt_report("spk_llm", frt_compare(v.data(), ref_v.data(), v.size(), 0.999), fails);
        if (frt_project(ctx, ctx->spk_dit_w, ctx->spk_dit_b, spk_in.data(), ctx->spk_dim, v) &&
            frt_ref_tensor(rm, "spk_dit", ref_v) && ref_v.size() == v.size())
            frt_report("spk_dit", frt_compare(v.data(), ref_v.data(), v.size(), 0.999), fails);
    }

    // ── stage: penc_prompt ──
    std::vector<float> penc_out;
    if (frt_penc_forward(ctx, lat_in.data(), T_in, penc_out) && frt_ref_tensor(rm, "penc_prompt", ref_v) &&
        ref_v.size() == penc_out.size())
        frt_report("penc_prompt", frt_compare(penc_out.data(), ref_v.data(), ref_v.size(), 0.999), fails);

    // ── stage: full generate with noise replay ──
    {
        // write the reference noise to a temp file for replay
        std::vector<float> ref_noise;
        if (frt_ref_tensor(rm, "noise_all", ref_noise) && !ref_noise.empty()) {
            std::string npath = "/tmp/frt_noise_replay.bin";
            if (const char* td = std::getenv("TMPDIR"))
                npath = std::string(td) + "/frt_noise_replay.bin";
            FILE* f = std::fopen(npath.c_str(), "wb");
            if (f) {
                std::fwrite(ref_noise.data(), sizeof(float), ref_noise.size(), f);
                std::fclose(f);
#ifdef _WIN32
                _putenv_s("FIREREDTTS3_NOISE", npath.c_str());
#else
                setenv("FIREREDTTS3_NOISE", npath.c_str(), 1);
#endif
            }
        }
        std::vector<float> ref_ids;
        frt_ref_tensor(rm, "text_tokens", ref_ids);
        std::vector<int32_t> ids(ref_ids.size());
        for (size_t i = 0; i < ref_ids.size(); i++)
            ids[i] = (int32_t)ref_ids[i];

        frt_gen_debug dbg;
        std::vector<float> latents_all;
        ctx->rng_seeded = false;
        if (!frt_generate(ctx, ids, lat_in, T_in, spk_in, latents_all, &dbg)) {
            fireredtts3_tts_free(ctx);
            return 2;
        }
        if (frt_ref_tensor(rm, "prefill_embeds", ref_v) && ref_v.size() == dbg.prefill_embeds.size())
            frt_report("prefill_embeds", frt_compare(dbg.prefill_embeds.data(), ref_v.data(), ref_v.size(), 0.999),
                       fails);
        if (frt_ref_tensor(rm, "llm_prefill_out", ref_v) && ref_v.size() == dbg.llm_prefill_out.size())
            frt_report("llm_prefill_out", frt_compare(dbg.llm_prefill_out.data(), ref_v.data(), ref_v.size(), 0.99),
                       fails);
        if (frt_ref_tensor(rm, "stop_scores", ref_v)) {
            size_t n = std::min(ref_v.size(), dbg.stop_scores.size());
            frt_report("stop_scores", frt_compare(dbg.stop_scores.data(), ref_v.data(), n, 0.99), fails);
            std::fprintf(stderr, "         stop_scores n_mine=%zu n_ref=%zu (first mine=%.4f ref=%.4f)\n",
                         dbg.stop_scores.size(), ref_v.size(), dbg.stop_scores.empty() ? -1.f : dbg.stop_scores[0],
                         ref_v.empty() ? -1.f : ref_v[0]);
        }
        if (frt_ref_tensor(rm, "latents_steps", ref_v)) {
            size_t n = std::min(ref_v.size(), dbg.latents_steps.size());
            if (n > 0)
                frt_report("latents_steps", frt_compare(dbg.latents_steps.data(), ref_v.data(), n, 0.98), fails);
        }
        std::vector<float> ref_all;
        if (frt_ref_tensor(rm, "latents_gen", ref_all)) {
            size_t n = std::min(ref_all.size(), latents_all.size());
            frt_report("latents_gen", frt_compare(latents_all.data(), ref_all.data(), n, 0.98), fails);
        }

        // ── stage: decode of the REFERENCE latents → gen_audio ──
        std::vector<float> ref_audio;
        if (!ref_all.empty() && frt_ref_tensor(rm, "gen_audio", ref_audio)) {
            std::vector<float> pcm, dec_hidden;
            const int T_all = (int)(ref_all.size() / ctx->redae_dim);
            if (frt_redae_decode(ctx, ref_all.data(), T_all, pcm, &dec_hidden)) {
                if (frt_ref_tensor(rm, "dec_hidden", ref_v) && ref_v.size() == dec_hidden.size())
                    frt_report("dec_hidden", frt_compare(dec_hidden.data(), ref_v.data(), ref_v.size(), 0.999), fails);
                const size_t trim = (size_t)T_in * ctx->ae_patch * ctx->ae_ds_rate;
                if (pcm.size() > trim) {
                    size_t n = std::min(pcm.size() - trim, ref_audio.size());
                    frt_report("gen_audio", frt_compare(pcm.data() + trim, ref_audio.data(), n, 0.99), fails);
                    std::fprintf(stderr, "         gen_audio n_mine=%zu n_ref=%zu\n", pcm.size() - trim,
                                 ref_audio.size());
                }
            }
        }
    }

    std::fprintf(stderr, "fireredtts3_diff: %s (%d stage failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    core_gguf::release_weight_buffer(rw.buf);
    ggml_free(rw.ctx);
    fireredtts3_tts_free(ctx);
    return fails == 0 ? 0 : 1;
}
