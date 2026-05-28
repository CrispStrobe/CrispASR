// cosyvoice3_tts.cpp — runtime for FunAudioLLM/Fun-CosyVoice3-0.5B-2512
//
// Phase 2: the CosyVoice3LM forward — Qwen2-0.5B body with two extra
// tensors over a vanilla Qwen2-0.5B-Instruct:
//
//   speech_embd     (6761, 896) — speech-token input embedding
//   speech_lm_head  (6761, 896) — speech-token AR head
//
// The body itself is identical to mimo_asr's Qwen2 step graph minus
// the fused-QKV path (the cosyvoice3 converter emits separate
// q/k/v tensors), so this file mostly leans on `core_attn::kv_self_attn`
// + `core_ffn::swiglu` and only adds the speech-side I/O.
//
// Architecture refs (lifted from PLAN.md §51):
//   d_model       = 896    (hidden_size)
//   n_layers      = 24
//   n_heads       = 14, n_kv_heads = 2  (GQA group = 7)
//   head_dim      = 64     (= d_model / n_heads)
//   ff_dim        = 4864   (intermediate_size)
//   rope_theta    = 1e6, rms_norm_eps = 1e-6
//   text_vocab    = 151936 (Qwen2 tokenizer, gpt2-BPE encoded into GGUF)
//   speech_vocab  = 6761   (head dim; codebook is [0, 6561), the upper
//                            ~200 entries are special / EOS markers)

#include "cosyvoice3_tts.h"

#include "core/attention.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

struct cv3_hp {
    uint32_t n_layers = 24;
    uint32_t d_model = 896;
    uint32_t n_heads = 14;
    uint32_t n_kv_heads = 2;
    uint32_t head_dim = 64;
    uint32_t ff_dim = 4864;
    uint32_t text_vocab = 151936;
    uint32_t max_pos = 32768;
    uint32_t speech_vocab = 6761;
    uint32_t speech_codebook = 6561;
    float rope_theta = 1e6f;
    float rms_norm_eps = 1e-6f;
};

struct cv3_qwen2_block {
    ggml_tensor* attn_norm_w = nullptr; // RMSNorm γ
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct cv3_lm {
    ggml_tensor* token_embd_w = nullptr;     // [d_model, text_vocab]
    ggml_tensor* output_norm_w = nullptr;    // [d_model]
    ggml_tensor* text_output_w = nullptr;    // [d_model, text_vocab] (unused in speech AR)
    ggml_tensor* speech_embd_w = nullptr;    // [d_model, speech_vocab]
    ggml_tensor* speech_lm_head_w = nullptr; // [d_model, speech_vocab]
    std::vector<cv3_qwen2_block> blocks;
};

// ---------------------------------------------------------------------------
// Phase 3 — Flow (DiT-CFM) hparams + tensor binding
// ---------------------------------------------------------------------------

struct cv3_flow_hp {
    uint32_t n_dit_layers = 22;
    uint32_t dit_dim = 1024;
    uint32_t dit_heads = 16;
    uint32_t dit_head_dim = 64;
    uint32_t dit_ff_dim = 2048;
    uint32_t dit_input_dim = 320;
    uint32_t mel_dim = 80;
    uint32_t spk_dim_in = 192;
    uint32_t spk_dim_out = 80;
    uint32_t speech_codebook = 6561;
    uint32_t pre_lookahead_len = 3;
    uint32_t token_mel_ratio = 2;
    uint32_t input_frame_rate = 25;
    uint32_t cfm_n_steps = 10;
    float cfm_inference_cfg_rate = 0.7f;
    float cfm_sigma_min = 1e-6f;
    float rope_theta = 10000.0f;
};

// One DiT block — AdaLN-Zero modulation (6×dim split for γ/β/gate × 2)
// projected from time-embed, followed by MHA (with RoPE inside) and an
// FFN (l1 → SiLU → l2).
struct cv3_dit_block {
    ggml_tensor* adaln_w = nullptr; // [dit_dim, 6*dit_dim]
    ggml_tensor* adaln_b = nullptr; // [6*dit_dim] F32
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* ffn_l1_w = nullptr; // [dit_dim, ff_dim]
    ggml_tensor* ffn_l1_b = nullptr;
    ggml_tensor* ffn_l2_w = nullptr; // [ff_dim, dit_dim]
    ggml_tensor* ffn_l2_b = nullptr;
};

struct cv3_flow {
    bool loaded = false;
    cv3_flow_hp hp;

    // Top-level
    ggml_tensor* input_embd_w = nullptr; // (mel_dim=80, speech_codebook=6561)
    ggml_tensor* pre_la_c1_w = nullptr;  // (K=4, 80, 1024) ggml ne
    ggml_tensor* pre_la_c1_b = nullptr;
    ggml_tensor* pre_la_c2_w = nullptr; // (K=3, 1024, 80)
    ggml_tensor* pre_la_c2_b = nullptr;
    ggml_tensor* spk_affine_w = nullptr; // (spk_dim_in=192, spk_dim_out=80)
    ggml_tensor* spk_affine_b = nullptr;

    // DiT input / time / position / output
    ggml_tensor* dit_in_proj_w = nullptr; // (320, 1024)
    ggml_tensor* dit_in_proj_b = nullptr;
    ggml_tensor* dit_conv_pos_c1_w = nullptr; // grouped conv1d-31 (K, in_per_grp, out)
    ggml_tensor* dit_conv_pos_c1_b = nullptr;
    ggml_tensor* dit_conv_pos_c2_w = nullptr;
    ggml_tensor* dit_conv_pos_c2_b = nullptr;
    ggml_tensor* dit_time_mlp_0_w = nullptr; // (256, 1024)
    ggml_tensor* dit_time_mlp_0_b = nullptr;
    ggml_tensor* dit_time_mlp_2_w = nullptr; // (1024, 1024)
    ggml_tensor* dit_time_mlp_2_b = nullptr;
    ggml_tensor* dit_rope_inv_freq = nullptr; // (head_dim/2,)
    ggml_tensor* dit_norm_out_w = nullptr;
    ggml_tensor* dit_norm_out_b = nullptr;
    ggml_tensor* dit_proj_out_w = nullptr;
    ggml_tensor* dit_proj_out_b = nullptr;

    std::vector<cv3_dit_block> blocks;

    // Flow-side ggml context + buffer (separate from the LM's).
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

} // namespace

struct cosyvoice3_tts_context {
    cosyvoice3_tts_context_params params{};
    int n_threads = 4;
    uint64_t seed = 42;

    cv3_hp hp;
    cv3_lm lm;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_w_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;

    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // Cached T=1 step graph (speech-token in). Built lazily on first
    // step call with fixed_kv_len = kv_max_ctx; subsequent steps reuse
    // the same plan via skip_plan.
    ggml_cgraph* step_t1_gf = nullptr;
    int step_t1_fixed_kv_len = 0;

    // RAS sampler RNG. Seeded once at init from params.seed (or 42);
    // re-seedable via cosyvoice3_tts_set_seed. Advances through every
    // RAS sample so repeated generate() calls don't replay.
    std::mt19937_64 rng{42};

    // Phase 3 — Flow sub-model (DiT-CFM). Populated by
    // cosyvoice3_tts_init_flow_from_file(). Stays empty if only the
    // LM was loaded (`flow.loaded == false`).
    cv3_flow flow{};
};

namespace {

uint32_t cv3_kv_u32(gguf_context* ctx, const char* key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
float cv3_kv_f32(gguf_context* ctx, const char* key, float def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

bool cv3_kv_init(cosyvoice3_tts_context* ctx, int max_ctx) {
    if (ctx->kv_k && ctx->kv_max_ctx >= max_ctx)
        return true;
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }
    const auto& hp = ctx->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.n_kv_heads;
    const int n_lay = (int)hp.n_layers;
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("cosyvoice3_tts");
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "cv3_kv_k");
    ggml_set_name(ctx->kv_v, "cv3_kv_v");
    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "cosyvoice3_tts");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "cosyvoice3_tts: failed to alloc KV buffer (%zu bytes)\n", kbytes + vbytes);
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_max_ctx = max_ctx;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts: kv cache %d MiB k=%s v=%s (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), ggml_type_name(kv_pair.k), ggml_type_name(kv_pair.v), hd, max_ctx,
                n_kv, n_lay);
    }
    return true;
}

// Build the per-step graph (T=1, embeds in, speech-logits out).
//
// Inputs (set externally before compute):
//   "inputs_embeds"   [d_model, T] F32
//   "lm_positions"    [T] I32
//   "lm_causal_mask"  [Lk, T] F16
// Output: "step_logits" [speech_vocab, T] F32 (typically T=1 → 6761)
//
// When fixed_kv_len > 0, positions doubles as kv_indices so the same
// graph topology is reused across steps with different n_past.
ggml_cgraph* cv3_build_lm_graph(cosyvoice3_tts_context* ctx, int n_tokens, int n_past, int fixed_kv_len) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->lm;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int d = (int)hp.d_model;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_input(embeds);
    ggml_set_name(embeds, "inputs_embeds");

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "lm_positions");

    // Always declare the causal mask so the cached-graph reuse path
    // (T=1 + fixed_kv_len > 0) keeps the topology invariant across
    // steps. At T=1 with fixed_kv_len=0 we'd normally skip the mask,
    // but matching mimo_asr_build_step_graph here keeps the patterns
    // consistent.
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
    ggml_set_input(causal_mask);
    ggml_set_name(causal_mask, "lm_causal_mask");

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(n_past + T <= ctx->kv_max_ctx);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* eff_kv_indices = (fixed_kv_len > 0) ? positions : nullptr;

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = m.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, h, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_o_w,
            /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask, ctx->kv_k, ctx->kv_v, (int)il,
            /*n_past*/ n_past, kvp,
            /*qkv_w*/ nullptr, /*fixed_kv_len*/ fixed_kv_len, /*kv_indices*/ eff_kv_indices, b.attn_q_b, b.attn_k_b,
            b.attn_v_b, /*o_b*/ nullptr, /*qkv_b*/ nullptr);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);

    // For T>1 the AR head only needs the last position. The convention
    // matches qwen3_tts.cpp build_graph_talker_kv: slice at T-1 to keep
    // the head matmul small.
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.speech_lm_head_w, cur);
    ggml_set_name(logits, "step_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// Build a one-shot "embedding lookup" graph: ids[N] -> get_rows(table) ->
// [d_model, N] F32. Useful to keep the table on whatever backend the
// weights live on without bouncing them through the CPU.
ggml_cgraph* cv3_build_embed_graph(cosyvoice3_tts_context* ctx, ggml_tensor* table, int n_tokens) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(ids);
    ggml_set_name(ids, "embed_ids");

    ggml_tensor* out = ggml_get_rows(ctx0, table, ids);
    // get_rows on F16 weights produces F32 output, which is what we want
    // to hand back to the caller. ggml_cont to materialise a contiguous
    // buffer the backend can dump straight.
    out = ggml_cont(ctx0, out);
    ggml_set_name(out, "embed_out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_free(ctx0);
    return gf;
}

float* cv3_run_embed(cosyvoice3_tts_context* ctx, ggml_tensor* table, const int32_t* ids, int n_tokens) {
    if (!table || !ids || n_tokens <= 0)
        return nullptr;
    // Building any other graph in `ctx->compute_meta` overwrites the
    // cached step graph's tensor metadata in place, so the next
    // `step_t1_gf` re-use would read garbage. Invalidate the cache.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;
    ggml_cgraph* gf = cv3_build_embed_graph(ctx, table, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: embed alloc_graph failed\n");
        return nullptr;
    }
    ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "embed_ids");
    ggml_backend_tensor_set(ids_t, ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: embed compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "embed_out");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" struct cosyvoice3_tts_context_params cosyvoice3_tts_context_default_params(void) {
    cosyvoice3_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.flash_attn = false;
    p.temperature = 0.0f;
    p.seed = 0; // 0 -> use default 42
    p.max_tokens = 0;
    p.ras_top_k = 25;
    p.ras_top_p = 0.8f;
    p.ras_win_size = 10;
    p.ras_tau_r = 0.1f;
    return p;
}

extern "C" struct cosyvoice3_tts_context* cosyvoice3_tts_init_from_file(const char* path_model,
                                                                        struct cosyvoice3_tts_context_params params) {
    auto* ctx = new cosyvoice3_tts_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->seed = params.seed ? params.seed : 42;
    ctx->rng.seed(ctx->seed);

    // ---- Metadata pass ----
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: failed to read GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.n_layers = cv3_kv_u32(gctx, "cosyvoice3.llm.n_layers", hp.n_layers);
    hp.d_model = cv3_kv_u32(gctx, "cosyvoice3.llm.d_model", hp.d_model);
    hp.n_heads = cv3_kv_u32(gctx, "cosyvoice3.llm.n_heads", hp.n_heads);
    hp.n_kv_heads = cv3_kv_u32(gctx, "cosyvoice3.llm.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = cv3_kv_u32(gctx, "cosyvoice3.llm.head_dim", hp.head_dim);
    hp.ff_dim = cv3_kv_u32(gctx, "cosyvoice3.llm.ff_dim", hp.ff_dim);
    hp.rope_theta = cv3_kv_f32(gctx, "cosyvoice3.llm.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = cv3_kv_f32(gctx, "cosyvoice3.llm.rms_norm_eps", hp.rms_norm_eps);
    hp.text_vocab = cv3_kv_u32(gctx, "cosyvoice3.llm.vocab_size", hp.text_vocab);
    hp.max_pos = cv3_kv_u32(gctx, "cosyvoice3.llm.max_pos", hp.max_pos);
    hp.speech_vocab = cv3_kv_u32(gctx, "cosyvoice3.llm.speech_vocab_size", hp.speech_vocab);
    hp.speech_codebook = cv3_kv_u32(gctx, "cosyvoice3.llm.speech_token_codebook", hp.speech_codebook);
    gguf_free(gctx);

    // ---- Backend init ----
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "cosyvoice3_tts: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    // Phase 2 default: CPU-only. The cosyvoice3 LM is small (~0.5B,
    // 24 layers) so CPU is acceptable for the diff-validation phase.
    // GPU path lands in a later phase once the prefill + step shapes
    // are validated.
    ctx->backend = ctx->backend_cpu;
    if (params.use_gpu && params.verbosity >= 1) {
        fprintf(stderr, "cosyvoice3_tts: --gpu requested but pinned to CPU for Phase 2 (LLM validation)\n");
    }

    // ---- Weight pass ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "cosyvoice3_tts", wl)) {
        fprintf(stderr, "cosyvoice3_tts: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->buf_w_cpu = wl.buf_cpu;
    ctx->tensors = std::move(wl.tensors);

    // ---- Tensor binding ----
    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->tensors, name.c_str(), "cosyvoice3_tts");
    };
    auto try_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::try_get(ctx->tensors, name.c_str());
    };

    auto& m = ctx->lm;
    m.token_embd_w = require_t("token_embd.weight");
    m.output_norm_w = require_t("output_norm.weight");
    m.text_output_w = try_t("output.weight"); // optional; unused in speech AR
    m.speech_embd_w = require_t("cosyvoice3.speech_embd.weight");
    m.speech_lm_head_w = require_t("cosyvoice3.speech_lm_head.weight");
    if (!m.token_embd_w || !m.output_norm_w || !m.speech_embd_w || !m.speech_lm_head_w) {
        fprintf(stderr, "cosyvoice3_tts: missing core LLM tensors\n");
        delete ctx;
        return nullptr;
    }

    m.blocks.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "blk.%u", il);
        auto& b = m.blocks[il];
        std::string p = prefix;
        b.attn_norm_w = require_t(p + ".attn_norm.weight");
        b.attn_q_w = require_t(p + ".attn_q.weight");
        b.attn_q_b = require_t(p + ".attn_q.bias");
        b.attn_k_w = require_t(p + ".attn_k.weight");
        b.attn_k_b = require_t(p + ".attn_k.bias");
        b.attn_v_w = require_t(p + ".attn_v.weight");
        b.attn_v_b = require_t(p + ".attn_v.bias");
        b.attn_o_w = require_t(p + ".attn_output.weight");
        b.ffn_norm_w = require_t(p + ".ffn_norm.weight");
        b.ffn_gate_w = require_t(p + ".ffn_gate.weight");
        b.ffn_up_w = require_t(p + ".ffn_up.weight");
        b.ffn_down_w = require_t(p + ".ffn_down.weight");
        if (!b.attn_norm_w || !b.attn_q_w || !b.attn_q_b || !b.attn_k_w || !b.attn_k_b || !b.attn_v_w || !b.attn_v_b ||
            !b.attn_o_w || !b.ffn_norm_w || !b.ffn_gate_w || !b.ffn_up_w || !b.ffn_down_w) {
            fprintf(stderr, "cosyvoice3_tts: missing tensors in %s.*\n", prefix);
            delete ctx;
            return nullptr;
        }
    }

    // ---- Scheduler + compute_meta ----
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "cosyvoice3_tts: loaded %zu tensors  llm=%uL d=%u h=%u/kv=%u hd=%u ff=%u "
                "text_vocab=%u speech_vocab=%u (codebook=%u)\n",
                ctx->tensors.size(), hp.n_layers, hp.d_model, hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.ff_dim,
                hp.text_vocab, hp.speech_vocab, hp.speech_codebook);
    }
    return ctx;
}

extern "C" void cosyvoice3_tts_free(struct cosyvoice3_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->buf_w_cpu)
        ggml_backend_buffer_free(ctx->buf_w_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->flow.buf_w)
        ggml_backend_buffer_free(ctx->flow.buf_w);
    if (ctx->flow.ctx_w)
        ggml_free(ctx->flow.ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void cosyvoice3_tts_set_n_threads(struct cosyvoice3_tts_context* ctx, int n_threads) {
    if (!ctx)
        return;
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
}

extern "C" void cosyvoice3_tts_set_seed(struct cosyvoice3_tts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->seed = seed ? seed : 42;
    ctx->rng.seed(ctx->seed);
}

extern "C" void cosyvoice3_tts_set_temperature(struct cosyvoice3_tts_context* ctx, float temperature) {
    if (!ctx)
        return;
    ctx->params.temperature = temperature;
}

extern "C" int cosyvoice3_tts_get_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* d_model, uint32_t* n_layers,
                                          uint32_t* n_heads, uint32_t* n_kv_heads, uint32_t* head_dim,
                                          uint32_t* text_vocab, uint32_t* speech_vocab, uint32_t* speech_codebook) {
    if (!ctx)
        return -1;
    const auto& hp = ctx->hp;
    if (d_model)
        *d_model = hp.d_model;
    if (n_layers)
        *n_layers = hp.n_layers;
    if (n_heads)
        *n_heads = hp.n_heads;
    if (n_kv_heads)
        *n_kv_heads = hp.n_kv_heads;
    if (head_dim)
        *head_dim = hp.head_dim;
    if (text_vocab)
        *text_vocab = hp.text_vocab;
    if (speech_vocab)
        *speech_vocab = hp.speech_vocab;
    if (speech_codebook)
        *speech_codebook = hp.speech_codebook;
    return 0;
}

extern "C" void cosyvoice3_tts_reset_kv(struct cosyvoice3_tts_context* ctx) {
    if (!ctx || !ctx->kv_buf)
        return;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;
}

extern "C" float* cosyvoice3_tts_embed_text(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens) {
    if (!ctx || !ids || n_tokens <= 0)
        return nullptr;
    return cv3_run_embed(ctx, ctx->lm.token_embd_w, ids, n_tokens);
}

extern "C" float* cosyvoice3_tts_embed_speech(struct cosyvoice3_tts_context* ctx, const int32_t* ids, int n_tokens) {
    if (!ctx || !ids || n_tokens <= 0)
        return nullptr;
    return cv3_run_embed(ctx, ctx->lm.speech_embd_w, ids, n_tokens);
}

extern "C" float* cosyvoice3_tts_prefill_with_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds,
                                                     int n_tokens, int n_past) {
    if (!ctx || !embeds || n_tokens <= 0 || n_past < 0)
        return nullptr;
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    if (!cv3_kv_init(ctx, std::max(n_past + n_tokens + 1024, 4096)))
        return nullptr;

    // Invalidate any cached step graph — prefill builds in the same
    // shared compute_meta and clobbers the step graph's metadata.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_lm_graph(ctx, n_tokens, n_past, /*fixed_kv_len*/ 0);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: prefill alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_t("inputs_embeds", embeds, (size_t)d * n_tokens * sizeof(float)))
        return nullptr;

    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; i++)
        pos[i] = n_past + i;
    if (!set_t("lm_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    const int Lk = n_past + n_tokens;
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * Lk);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n_tokens; q++)
        for (int k = 0; k < Lk; k++)
            mask[(size_t)q * Lk + k] = (k <= n_past + q) ? z : ninf;
    if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: prefill compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "step_logits");
    if (!logits_t)
        return nullptr;
    const size_t n_log = (size_t)ggml_nelements(logits_t);
    float* out = (float*)malloc(n_log * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(logits_t, out, 0, n_log * sizeof(float));
    return out;
}

extern "C" float* cosyvoice3_tts_step_speech(struct cosyvoice3_tts_context* ctx, int32_t speech_id, int n_past) {
    if (!ctx || speech_id < 0 || n_past < 0)
        return nullptr;
    const auto& hp = ctx->hp;
    const int d = (int)hp.d_model;
    if ((uint32_t)speech_id >= hp.speech_vocab) {
        fprintf(stderr, "cosyvoice3_tts: speech_id %d out of range [0, %u)\n", speech_id, hp.speech_vocab);
        return nullptr;
    }
    if (!cv3_kv_init(ctx, std::max(n_past + 1 + 1024, 4096)))
        return nullptr;
    if (n_past + 1 > ctx->kv_max_ctx) {
        fprintf(stderr, "cosyvoice3_tts: kv overflow (%d+1 > %d)\n", n_past, ctx->kv_max_ctx);
        return nullptr;
    }

    // Step 1: look up speech_embd[speech_id] as a [d_model, 1] F32 buffer.
    float* embed = cv3_run_embed(ctx, ctx->lm.speech_embd_w, &speech_id, 1);
    if (!embed)
        return nullptr;

    // Step 2: run the LM forward with that embedding. Use the cached
    // T=1 graph when possible to amortise the build cost across steps.
    const int fixed_kv = ctx->kv_max_ctx;
    const bool can_skip = (ctx->step_t1_gf != nullptr && ctx->step_t1_fixed_kv_len == fixed_kv);

    ggml_cgraph* gf;
    if (can_skip) {
        gf = ctx->step_t1_gf;
    } else {
        gf = cv3_build_lm_graph(ctx, /*n_tokens*/ 1, /*n_past*/ 0, /*fixed_kv_len*/ fixed_kv);
        if (!gf) {
            free(embed);
            return nullptr;
        }
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "cosyvoice3_tts: step alloc_graph failed\n");
            free(embed);
            return nullptr;
        }
        ctx->step_t1_gf = gf;
        ctx->step_t1_fixed_kv_len = fixed_kv;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_t("inputs_embeds", embed, (size_t)d * sizeof(float))) {
        free(embed);
        return nullptr;
    }
    free(embed);

    int32_t pos = n_past;
    if (!set_t("lm_positions", &pos, sizeof(pos)))
        return nullptr;

    std::vector<ggml_fp16_t> mask((size_t)fixed_kv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int k = 0; k < fixed_kv; k++)
        mask[k] = (k <= n_past) ? z : ninf;
    if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: step compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "step_logits");
    if (!logits_t)
        return nullptr;
    const size_t n_log = (size_t)ggml_nelements(logits_t);
    float* out = (float*)malloc(n_log * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(logits_t, out, 0, n_log * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Sampling — Repetition-Aware Sampling (RAS), ported from upstream
// CosyVoice/cosyvoice/utils/common.py::ras_sampling.
//
// nucleus_sampling: softmax(logits) → stable-sort descending → take
//   while cum_prob < top_p AND count < top_k → multinomial-sample over
//   the kept (non-renormalised) probabilities.
//
// ras_sampling: nucleus sample → if the picked token appears in
//   decoded_history[-win_size:] ≥ win_size·tau_r times, suppress it
//   (logits[id] = -INF) and re-sample via plain softmax-multinomial
//   over the modified logits.
// ---------------------------------------------------------------------------

namespace {

// Stable-softmax: subtract max to avoid overflow.
void softmax_inplace(std::vector<float>& v) {
    float vmax = v.empty() ? 0.0f : v[0];
    for (float x : v)
        if (x > vmax)
            vmax = x;
    double s = 0.0;
    for (auto& x : v) {
        x = std::exp(x - vmax);
        s += x;
    }
    if (s > 0.0) {
        const float inv = (float)(1.0 / s);
        for (auto& x : v)
            x *= inv;
    }
}

// Multinomial sample given an unnormalised positive weight vector. Mirrors
// torch.multinomial(weights, 1, replacement=True): treat weights / sum as
// the categorical distribution, draw one via inverse-CDF. Returns -1 if
// weights are all zero / NaN / negative.
int32_t multinomial_pick(const std::vector<float>& weights, std::mt19937_64& rng) {
    double sum = 0.0;
    for (float w : weights) {
        if (!(w > 0.0f) || std::isnan(w))
            continue;
        sum += w;
    }
    if (!(sum > 0.0))
        return -1;
    std::uniform_real_distribution<double> U(0.0, sum);
    double r = U(rng);
    double acc = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
        if (!(weights[i] > 0.0f) || std::isnan(weights[i]))
            continue;
        acc += weights[i];
        if (r <= acc)
            return (int32_t)i;
    }
    return (int32_t)weights.size() - 1; // floating-point tail
}

// Upstream `nucleus_sampling`: select top tokens until cum_prob >= top_p
// or count >= top_k (whichever comes first), then multinomial-sample
// over the kept (unrenormalised) probabilities.
int32_t nucleus_sample(const float* logits, int n_vocab, float top_p, int top_k, std::mt19937_64& rng) {
    std::vector<float> probs((size_t)n_vocab);
    std::memcpy(probs.data(), logits, (size_t)n_vocab * sizeof(float));
    softmax_inplace(probs);
    // Stable-sort indices by descending prob. std::stable_sort matches
    // PyTorch's `sort(stable=True)` ordering for ties.
    std::vector<int32_t> idx((size_t)n_vocab);
    for (int i = 0; i < n_vocab; i++)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });
    std::vector<float> kept;
    std::vector<int32_t> kept_ids;
    double cum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        // Upstream guard: stop when cum_prob >= top_p OR count >= top_k.
        if (cum >= (double)top_p || (int)kept.size() >= top_k)
            break;
        const float p = probs[idx[i]];
        cum += (double)p;
        kept.push_back(p);
        kept_ids.push_back(idx[i]);
    }
    if (kept.empty())
        return -1;
    int32_t pick = multinomial_pick(kept, rng);
    if (pick < 0)
        return -1;
    return kept_ids[(size_t)pick];
}

} // namespace

extern "C" int32_t cosyvoice3_tts_sample_ras(struct cosyvoice3_tts_context* ctx, const float* logits,
                                             const int32_t* decoded_history, int n_history) {
    if (!ctx || !logits)
        return -1;
    const auto& hp = ctx->hp;
    const int n_vocab = (int)hp.speech_vocab;
    const auto& sp = ctx->params;
    const float top_p = sp.ras_top_p > 0.0f ? sp.ras_top_p : 0.8f;
    const int top_k = sp.ras_top_k > 0 ? sp.ras_top_k : 25;
    const int win_size = sp.ras_win_size > 0 ? sp.ras_win_size : 10;
    const float tau_r = sp.ras_tau_r > 0.0f ? sp.ras_tau_r : 0.1f;

    int32_t pick = nucleus_sample(logits, n_vocab, top_p, top_k, ctx->rng);
    if (pick < 0)
        return -1;

    if (!decoded_history || n_history <= 0)
        return pick;

    // Repetition check over the trailing `win_size` of decoded_history.
    // `pick` is counted in the suffix; if it appears ≥ win_size·tau_r
    // times, suppress and re-sample via plain softmax-multinomial over
    // the FULL distribution (matches upstream `random_sampling`).
    int rep = 0;
    const int start = std::max(0, n_history - win_size);
    for (int i = start; i < n_history; i++) {
        if (decoded_history[i] == pick)
            rep++;
    }
    const float thresh = (float)win_size * tau_r;
    if ((float)rep >= thresh) {
        std::vector<float> mod((size_t)n_vocab);
        std::memcpy(mod.data(), logits, (size_t)n_vocab * sizeof(float));
        mod[(size_t)pick] = -INFINITY;
        softmax_inplace(mod);
        pick = multinomial_pick(mod, ctx->rng);
    }
    return pick;
}

extern "C" int32_t* cosyvoice3_tts_generate_tokens_from_embeds(struct cosyvoice3_tts_context* ctx, const float* embeds,
                                                               int n_tokens, int max_tokens, int stop_token_id,
                                                               int* out_n) {
    if (!ctx || !embeds || n_tokens <= 0 || !out_n)
        return nullptr;
    *out_n = 0;
    const auto& hp = ctx->hp;
    const int speech_vocab = (int)hp.speech_vocab;
    const int max_steps = max_tokens > 0 ? max_tokens : (ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 1500);

    cosyvoice3_tts_reset_kv(ctx);
    float* logits = cosyvoice3_tts_prefill_with_embeds(ctx, embeds, n_tokens, /*n_past*/ 0);
    if (!logits)
        return nullptr;

    std::vector<int32_t> out;
    out.reserve((size_t)max_steps);
    const bool greedy = !(ctx->params.temperature > 0.0f);

    int n_past = n_tokens;
    for (int step = 0; step < max_steps; step++) {
        int32_t pick;
        if (greedy) {
            // Greedy argmax. Restrict to the codebook range so we never
            // emit special-token rows that lie past index speech_codebook.
            int n_pick_range = (int)hp.speech_codebook > 0 ? (int)hp.speech_codebook : speech_vocab;
            float bv = logits[0];
            pick = 0;
            for (int i = 1; i < n_pick_range; i++)
                if (logits[i] > bv) {
                    bv = logits[i];
                    pick = i;
                }
        } else {
            pick = cosyvoice3_tts_sample_ras(ctx, logits, out.empty() ? nullptr : out.data(), (int)out.size());
            if (pick < 0) {
                free(logits);
                *out_n = 0;
                return nullptr;
            }
        }
        free(logits);
        if (stop_token_id >= 0 && pick == stop_token_id)
            break;
        out.push_back(pick);
        if (n_past + 1 > ctx->kv_max_ctx) {
            fprintf(stderr, "cosyvoice3_tts: generate_tokens: kv overflow at step %d\n", step);
            break;
        }
        logits = cosyvoice3_tts_step_speech(ctx, pick, n_past);
        if (!logits) {
            fprintf(stderr, "cosyvoice3_tts: generate_tokens: step %d failed\n", step);
            *out_n = (int)out.size();
            int32_t* dup = (int32_t*)malloc(out.size() * sizeof(int32_t));
            if (!dup)
                return nullptr;
            std::memcpy(dup, out.data(), out.size() * sizeof(int32_t));
            return dup;
        }
        n_past++;
    }
    // The trailing logits buffer is freed inside the loop on success;
    // we only get here after the break, where logits is already freed.

    *out_n = (int)out.size();
    if (out.empty())
        return (int32_t*)malloc(0); // benign 0-length pointer
    int32_t* arr = (int32_t*)malloc(out.size() * sizeof(int32_t));
    if (!arr)
        return nullptr;
    std::memcpy(arr, out.data(), out.size() * sizeof(int32_t));
    return arr;
}

namespace {
// Forward declarations; definitions live further down in this file
// (alongside the per-graph builders they depend on).
float* cv3_extract_flow_dit_stage(cosyvoice3_tts_context* ctx, int block_idx, const float* x, int T, const float* t_emb,
                                  const char* tensor_name);
float* cv3_extract_pre_la_stage(cosyvoice3_tts_context* ctx, const int32_t* ids, int T_tok, const char* tensor_name);
float* cv3_extract_in_pipe_stage(cosyvoice3_tts_context* ctx, const float* pre_la_out, int T_mel, const float* spk_emb,
                                 const float* x_noisy, const float* cond, const char* tensor_name);
float* cv3_extract_dit_full_stage(cosyvoice3_tts_context* ctx, const float* x, int T_mel, const float* t_emb,
                                  const char* tensor_name);
float* cv3_extract_euler_stage(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                               const float* cond, const float* x_init, int n_steps, float cfg_rate,
                               const char* tensor_name);
float* cv3_run_solve_euler(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                           const float* cond, const float* x_init, int n_steps, float cfg_rate, float* dphi_step0_out);
} // namespace

extern "C" float* cosyvoice3_tts_extract_stage(struct cosyvoice3_tts_context* ctx, const char* stage_name,
                                               const int32_t* ids, int n_ids, const float* embeds_in,
                                               int n_embed_tokens, int* out_n) {
    if (!ctx || !stage_name || !out_n)
        return nullptr;
    *out_n = 0;
    const auto& hp = ctx->hp;
    if (strcmp(stage_name, "lm_token_embd") == 0) {
        if (!ids || n_ids <= 0)
            return nullptr;
        float* out = cosyvoice3_tts_embed_text(ctx, ids, n_ids);
        if (!out)
            return nullptr;
        *out_n = n_ids * (int)hp.d_model;
        return out;
    }
    if (strcmp(stage_name, "lm_speech_embd") == 0) {
        if (!ids || n_ids <= 0)
            return nullptr;
        float* out = cosyvoice3_tts_embed_speech(ctx, ids, n_ids);
        if (!out)
            return nullptr;
        *out_n = n_ids * (int)hp.d_model;
        return out;
    }
    if (strcmp(stage_name, "lm_step0_logits") == 0) {
        if (!embeds_in || n_embed_tokens <= 0)
            return nullptr;
        cosyvoice3_tts_reset_kv(ctx);
        float* out = cosyvoice3_tts_prefill_with_embeds(ctx, embeds_in, n_embed_tokens, /*n_past*/ 0);
        if (!out)
            return nullptr;
        *out_n = (int)hp.speech_vocab;
        return out;
    }
    // Flow Phase 3b single-block diff stages:
    //   "flow_dit_blk_<N>_out"     — final block output  [T, dit_dim] F32
    //   "flow_dit_blk_<N>_lnx_a"   — LN(x) before modulation
    //   "flow_dit_blk_<N>_h_a"     — post-modulate, pre-attn
    //   "flow_dit_blk_<N>_attn"    — attention out (pre-residual)
    //   "flow_dit_blk_<N>_xattn"   — x + gate_msa * attn_out
    //   "flow_dit_blk_<N>_ff"      — FFN out (pre-residual)
    //
    // `embeds_in` carries the packed [x | t_emb] block: first
    // n_embed_tokens*dit_dim floats are x [T, dit_dim], remaining dit_dim
    // floats are t_emb (post time_mlp). The caller computes T from the
    // ref archive's tensor shape.
    if (strncmp(stage_name, "flow_dit_blk_", 13) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int d = (int)fh.dit_dim;
        // Parse block index: "flow_dit_blk_<N>_<sfx>". Find the underscore
        // after the digits.
        const char* p = stage_name + 13;
        int block_idx = 0;
        const char* sfx = p;
        while (*sfx >= '0' && *sfx <= '9') {
            block_idx = block_idx * 10 + (*sfx - '0');
            sfx++;
        }
        if (*sfx != '_')
            return nullptr;
        sfx++; // past the underscore separating idx from suffix
        const char* tensor_name = nullptr;
        if (strcmp(sfx, "out") == 0)
            tensor_name = "dit_block_out";
        else if (strcmp(sfx, "lnx_a") == 0)
            tensor_name = "dbg_lnx_a";
        else if (strcmp(sfx, "h_a") == 0)
            tensor_name = "dbg_h_a";
        else if (strcmp(sfx, "attn") == 0)
            tensor_name = "dbg_attn_raw";
        else if (strcmp(sfx, "xattn") == 0)
            tensor_name = "dbg_x_after_attn";
        else if (strcmp(sfx, "ff") == 0)
            tensor_name = "dbg_ff_raw";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_dit_blk stage suffix '%s'\n", sfx);
            return nullptr;
        }
        const int T = n_embed_tokens;
        // embeds_in layout: T*d floats of x, then d floats of t_emb.
        const float* x = embeds_in;
        const float* t_emb = embeds_in + (size_t)T * d;
        float* out = cv3_extract_flow_dit_stage(ctx, block_idx, x, T, t_emb, tensor_name);
        if (!out)
            return nullptr;
        *out_n = T * d;
        return out;
    }
    // Flow Phase 3c pre-lookahead conv stack:
    //   "flow_pre_la_tok_emb"  — input_embedding(speech_ids)  [T_tok, mel_dim]
    //   "flow_pre_la_c1"       — leaky_relu(conv1(...))       [T_tok, 1024]
    //   "flow_pre_la_c2"       — conv2(post-causal-pad)       [T_tok, mel_dim]
    //   "flow_pre_la"          — final pre_la output (with residual)
    //                                                          [T_tok, mel_dim]
    //
    // `ids` carries speech-token IDs (length n_ids = T_tok). The graph
    // does the input_embedding lookup inside, so the diff harness
    // exercises the embedding + conv stack as a single unit.
    if (strncmp(stage_name, "flow_pre_la", 11) == 0) {
        if (!ctx->flow.loaded || !ids || n_ids <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const char* sfx = stage_name + 11;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "pre_la_out";
        else if (strcmp(sfx, "_tok_emb") == 0)
            tensor_name = "pre_la_tok_emb";
        else if (strcmp(sfx, "_c1") == 0)
            tensor_name = "pre_la_c1";
        else if (strcmp(sfx, "_c2") == 0)
            tensor_name = "pre_la_c2";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_pre_la stage suffix '%s'\n", sfx);
            return nullptr;
        }
        float* out = cv3_extract_pre_la_stage(ctx, ids, n_ids, tensor_name);
        if (!out)
            return nullptr;
        // pre_la_tok_emb and pre_la / pre_la_c2 are (T_tok, mel_dim);
        // pre_la_c1 is (T_tok, 1024).
        const int out_dim = (strcmp(tensor_name, "pre_la_c1") == 0) ? 1024 : (int)fh.mel_dim;
        *out_n = n_ids * out_dim;
        return out;
    }
    // Flow Phase 3c InputEmbedding (input pipeline) stages:
    //   "flow_in_pipe_spk"   — F.normalize(spk) -> spk_affine    [spk_dim_out]
    //   "flow_in_pipe_cat"   — cat[x, cond, mu, spks]            [T_mel, 320]
    //   "flow_in_pipe_proj"  — in_proj(cat)                      [T_mel, 1024]
    //   "flow_in_pipe_pos"   — conv_pos_embed(proj)              [T_mel, 1024]
    //   "flow_in_pipe"       — proj + conv_pos_embed(proj)       [T_mel, 1024]
    //
    // `embeds_in` packs (in this order):
    //   pre_la_out          [T_mel, mel_dim]  F32 — already repeat-interleaved
    //                                              upstream by token_mel_ratio,
    //                                              so length is T_mel (= 2*T_tok)
    //   spk_emb_raw         [spk_dim_in]      F32 — pre-normalize, pre-projection
    //   x_noisy             [T_mel, mel_dim]  F32 — CFM solver iterate
    //   cond                [T_mel, mel_dim]  F32 — prompt-prefix conditioning
    //
    // n_embed_tokens carries T_mel. The graph builder normalises spk and
    // broadcasts it over T_mel internally.
    if (strncmp(stage_name, "flow_in_pipe", 12) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int mel = (int)fh.mel_dim;
        const int spk_in = (int)fh.spk_dim_in;
        const int dit_dim = (int)fh.dit_dim;
        const int dit_in_dim = (int)fh.dit_input_dim;
        const int T_mel = n_embed_tokens;
        const char* sfx = stage_name + 12;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "in_pipe_out";
        else if (strcmp(sfx, "_spk") == 0)
            tensor_name = "in_pipe_spk";
        else if (strcmp(sfx, "_cat") == 0)
            tensor_name = "in_pipe_cat";
        else if (strcmp(sfx, "_proj") == 0)
            tensor_name = "in_pipe_proj";
        else if (strcmp(sfx, "_pos") == 0)
            tensor_name = "in_pipe_pos";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_in_pipe stage suffix '%s'\n", sfx);
            return nullptr;
        }
        // embeds_in layout: pre_la (T_mel*mel) | spk_raw (spk_in) | x (T_mel*mel) | cond (T_mel*mel)
        const float* pre_la = embeds_in;
        const float* spk_raw = pre_la + (size_t)T_mel * mel;
        const float* x_noisy = spk_raw + (size_t)spk_in;
        const float* cond = x_noisy + (size_t)T_mel * mel;
        float* out = cv3_extract_in_pipe_stage(ctx, pre_la, T_mel, spk_raw, x_noisy, cond, tensor_name);
        if (!out)
            return nullptr;
        int out_n_local;
        if (strcmp(tensor_name, "in_pipe_spk") == 0)
            out_n_local = (int)fh.spk_dim_out;
        else if (strcmp(tensor_name, "in_pipe_cat") == 0)
            out_n_local = T_mel * dit_in_dim;
        else
            out_n_local = T_mel * dit_dim;
        *out_n = out_n_local;
        return out;
    }
    // Full DiT estimator forward — the entire 22-block stack +
    // AdaLayerNormZero_Final + proj_out. Inputs are post-input-pipeline
    // (validated by the `flow_in_pipe*` stages):
    //
    //   "flow_dit_full_norm"  — output of AdaLN-Final, pre-proj_out
    //                            [T_mel, dit_dim]
    //   "flow_dit_full"       — final mel output (post proj_out)
    //                            [T_mel, mel_dim]
    //
    // `embeds_in` packs [x | t_emb] = T_mel*dit_dim + dit_dim floats.
    // n_embed_tokens = T_mel.
    if (strncmp(stage_name, "flow_dit_full", 13) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int d = (int)fh.dit_dim;
        const int mel = (int)fh.mel_dim;
        const int T_mel = n_embed_tokens;
        const char* sfx = stage_name + 13;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "dit_full_out";
        else if (strcmp(sfx, "_norm") == 0)
            tensor_name = "dit_full_norm";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_dit_full stage suffix '%s'\n", sfx);
            return nullptr;
        }
        const float* x = embeds_in;
        const float* t_emb = embeds_in + (size_t)T_mel * d;
        float* out = cv3_extract_dit_full_stage(ctx, x, T_mel, t_emb, tensor_name);
        if (!out)
            return nullptr;
        const int out_dim = (strcmp(tensor_name, "dit_full_out") == 0) ? mel : d;
        *out_n = T_mel * out_dim;
        return out;
    }
    // Phase 3d-B — CFM Euler ODE driver. Inputs (packed in `embeds_in` in
    // this order):
    //   mu          [T_mel, mel_dim]   F32 — pre_la + repeat_interleave output
    //   spks_proj   [spk_dim_out]      F32 — already through normalize +
    //                                          spk_affine (caller-supplied)
    //   cond        [T_mel, mel_dim]   F32 — prompt-prefix conditioning (zeros
    //                                          for zero-shot)
    //   x_init      [T_mel, mel_dim]   F32 — initial Euler noise (caller-supplied
    //                                          so the diff harness can match
    //                                          upstream's seeded `rand_noise`)
    //
    // n_embed_tokens = T_mel. Stage suffixes:
    //   "flow_euler"            — final mel [T_mel, mel_dim] after 10 Euler steps
    //   "flow_euler_dphi_step0" — dphi_dt at step 0 (after CFG combine), debug
    //
    // `n_steps` and `cfg_rate` are pinned to the upstream defaults (10, 0.7);
    // exposing them on extract_stage would inflate the API for marginal gain.
    if (strncmp(stage_name, "flow_euler", 10) == 0) {
        if (!ctx->flow.loaded || !embeds_in || n_embed_tokens <= 0)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        const int mel = (int)fh.mel_dim;
        const int spk_out = (int)fh.spk_dim_out;
        const int T_mel = n_embed_tokens;
        const char* sfx = stage_name + 10;
        const char* tensor_name = nullptr;
        if (*sfx == 0)
            tensor_name = "euler_out";
        else if (strcmp(sfx, "_dphi_step0") == 0)
            tensor_name = "euler_dphi_step0";
        else {
            fprintf(stderr, "cosyvoice3_tts: unknown flow_euler stage suffix '%s'\n", sfx);
            return nullptr;
        }
        // Unpack: mu | spks_proj | cond | x_init.
        const float* mu = embeds_in;
        const float* spks_proj = mu + (size_t)T_mel * mel;
        const float* cond = spks_proj + (size_t)spk_out;
        const float* x_init = cond + (size_t)T_mel * mel;
        const int n_steps = (int)fh.cfm_n_steps;          // 10
        const float cfg_rate = fh.cfm_inference_cfg_rate; // 0.7
        float* out = cv3_extract_euler_stage(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, tensor_name);
        if (!out)
            return nullptr;
        *out_n = T_mel * mel;
        return out;
    }
    if (strcmp(stage_name, "flow_inventory") == 0) {
        if (!ctx->flow.loaded) {
            fprintf(stderr, "cosyvoice3_tts: flow_inventory requested but flow not loaded\n");
            return nullptr;
        }
        // Return a small sentinel buffer encoding (n_dit_layers, dit_dim,
        // dit_heads, head_dim, ff_dim, input_dim, mel_dim, spk_dim_in,
        // spk_dim_out, n_steps) — useful for the diff harness to verify
        // it sees the flow GGUF.
        float* out = (float*)malloc(10 * sizeof(float));
        if (!out)
            return nullptr;
        const auto& fh = ctx->flow.hp;
        out[0] = (float)fh.n_dit_layers;
        out[1] = (float)fh.dit_dim;
        out[2] = (float)fh.dit_heads;
        out[3] = (float)fh.dit_head_dim;
        out[4] = (float)fh.dit_ff_dim;
        out[5] = (float)fh.dit_input_dim;
        out[6] = (float)fh.mel_dim;
        out[7] = (float)fh.spk_dim_in;
        out[8] = (float)fh.spk_dim_out;
        out[9] = (float)fh.cfm_n_steps;
        *out_n = 10;
        return out;
    }
    fprintf(stderr, "cosyvoice3_tts: unknown stage '%s'\n", stage_name);
    return nullptr;
}

// ---------------------------------------------------------------------------
// Phase 3 — Flow loader + hparam reader
// ---------------------------------------------------------------------------

extern "C" int cosyvoice3_tts_init_flow_from_file(struct cosyvoice3_tts_context* ctx, const char* path) {
    if (!ctx || !path) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: bad args\n");
        return -1;
    }
    if (ctx->flow.loaded) {
        fprintf(stderr, "cosyvoice3_tts: flow already loaded\n");
        return 0;
    }

    // ---- Metadata pass ----
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: failed to read GGUF '%s'\n", path);
        return -1;
    }

    auto& fh = ctx->flow.hp;
    fh.n_dit_layers = cv3_kv_u32(gctx, "cosyvoice3.flow.n_dit_layers", fh.n_dit_layers);
    fh.dit_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_dim", fh.dit_dim);
    fh.dit_heads = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_heads", fh.dit_heads);
    fh.dit_head_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_head_dim", fh.dit_head_dim);
    fh.dit_ff_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_ff_dim", fh.dit_ff_dim);
    fh.dit_input_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.dit_input_dim", fh.dit_input_dim);
    fh.mel_dim = cv3_kv_u32(gctx, "cosyvoice3.flow.mel_dim", fh.mel_dim);
    fh.spk_dim_in = cv3_kv_u32(gctx, "cosyvoice3.flow.spk_dim_in", fh.spk_dim_in);
    fh.spk_dim_out = cv3_kv_u32(gctx, "cosyvoice3.flow.spk_dim_out", fh.spk_dim_out);
    fh.speech_codebook = cv3_kv_u32(gctx, "cosyvoice3.flow.speech_codebook", fh.speech_codebook);
    fh.pre_lookahead_len = cv3_kv_u32(gctx, "cosyvoice3.flow.pre_lookahead_len", fh.pre_lookahead_len);
    fh.token_mel_ratio = cv3_kv_u32(gctx, "cosyvoice3.flow.token_mel_ratio", fh.token_mel_ratio);
    fh.input_frame_rate = cv3_kv_u32(gctx, "cosyvoice3.flow.input_frame_rate", fh.input_frame_rate);
    fh.cfm_n_steps = cv3_kv_u32(gctx, "cosyvoice3.flow.cfm_n_steps", fh.cfm_n_steps);
    fh.cfm_inference_cfg_rate = cv3_kv_f32(gctx, "cosyvoice3.flow.cfm_inference_cfg_rate", fh.cfm_inference_cfg_rate);
    fh.cfm_sigma_min = cv3_kv_f32(gctx, "cosyvoice3.flow.cfm_sigma_min", fh.cfm_sigma_min);
    fh.rope_theta = cv3_kv_f32(gctx, "cosyvoice3.flow.rope_theta", fh.rope_theta);
    gguf_free(gctx);

    // ---- Weight pass ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "cosyvoice3_tts:flow", wl)) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: load_weights failed for '%s'\n", path);
        return -1;
    }
    ctx->flow.ctx_w = wl.ctx;
    ctx->flow.buf_w = wl.buf;
    ctx->flow.tensors = std::move(wl.tensors);

    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->flow.tensors, name.c_str(), "cosyvoice3_tts:flow");
    };

    auto& f = ctx->flow;
    f.input_embd_w = require_t("cosyvoice3.flow.input_embd.w");
    f.pre_la_c1_w = require_t("cosyvoice3.flow.pre_la.conv1.w");
    f.pre_la_c1_b = require_t("cosyvoice3.flow.pre_la.conv1.b");
    f.pre_la_c2_w = require_t("cosyvoice3.flow.pre_la.conv2.w");
    f.pre_la_c2_b = require_t("cosyvoice3.flow.pre_la.conv2.b");
    f.spk_affine_w = require_t("cosyvoice3.flow.spk_affine.w");
    f.spk_affine_b = require_t("cosyvoice3.flow.spk_affine.b");
    f.dit_in_proj_w = require_t("cosyvoice3.flow.dit.in_proj.w");
    f.dit_in_proj_b = require_t("cosyvoice3.flow.dit.in_proj.b");
    f.dit_conv_pos_c1_w = require_t("cosyvoice3.flow.dit.conv_pos.c1.w");
    f.dit_conv_pos_c1_b = require_t("cosyvoice3.flow.dit.conv_pos.c1.b");
    f.dit_conv_pos_c2_w = require_t("cosyvoice3.flow.dit.conv_pos.c2.w");
    f.dit_conv_pos_c2_b = require_t("cosyvoice3.flow.dit.conv_pos.c2.b");
    f.dit_time_mlp_0_w = require_t("cosyvoice3.flow.dit.time_mlp.0.w");
    f.dit_time_mlp_0_b = require_t("cosyvoice3.flow.dit.time_mlp.0.b");
    f.dit_time_mlp_2_w = require_t("cosyvoice3.flow.dit.time_mlp.2.w");
    f.dit_time_mlp_2_b = require_t("cosyvoice3.flow.dit.time_mlp.2.b");
    f.dit_rope_inv_freq = require_t("cosyvoice3.flow.dit.rope_inv_freq");
    f.dit_norm_out_w = require_t("cosyvoice3.flow.dit.norm_out.w");
    f.dit_norm_out_b = require_t("cosyvoice3.flow.dit.norm_out.b");
    f.dit_proj_out_w = require_t("cosyvoice3.flow.dit.proj_out.w");
    f.dit_proj_out_b = require_t("cosyvoice3.flow.dit.proj_out.b");

    f.blocks.resize(fh.n_dit_layers);
    for (uint32_t L = 0; L < fh.n_dit_layers; L++) {
        char prefix[48];
        snprintf(prefix, sizeof(prefix), "cosyvoice3.flow.dit.blk.%u", L);
        auto& b = f.blocks[L];
        std::string p = prefix;
        b.adaln_w = require_t(p + ".adaln.w");
        b.adaln_b = require_t(p + ".adaln.b");
        b.attn_q_w = require_t(p + ".attn.q.w");
        b.attn_q_b = require_t(p + ".attn.q.b");
        b.attn_k_w = require_t(p + ".attn.k.w");
        b.attn_k_b = require_t(p + ".attn.k.b");
        b.attn_v_w = require_t(p + ".attn.v.w");
        b.attn_v_b = require_t(p + ".attn.v.b");
        b.attn_o_w = require_t(p + ".attn.o.w");
        b.attn_o_b = require_t(p + ".attn.o.b");
        b.ffn_l1_w = require_t(p + ".ffn.l1.w");
        b.ffn_l1_b = require_t(p + ".ffn.l1.b");
        b.ffn_l2_w = require_t(p + ".ffn.l2.w");
        b.ffn_l2_b = require_t(p + ".ffn.l2.b");
        if (!b.adaln_w || !b.attn_q_w || !b.ffn_l1_w) {
            fprintf(stderr, "cosyvoice3_tts: init_flow: missing tensors in %s.*\n", prefix);
            return -1;
        }
    }

    if (!f.input_embd_w || !f.dit_in_proj_w || !f.dit_proj_out_w) {
        fprintf(stderr, "cosyvoice3_tts: init_flow: missing top-level flow tensors\n");
        return -1;
    }

    f.loaded = true;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr,
                "cosyvoice3_tts:flow loaded %zu tensors  dit=%uL d=%u h=%u/hd=%u ff=%u "
                "in_dim=%u mel=%u spk=%u/%u codebook=%u cfm_steps=%u cfg=%.2f\n",
                f.tensors.size(), fh.n_dit_layers, fh.dit_dim, fh.dit_heads, fh.dit_head_dim, fh.dit_ff_dim,
                fh.dit_input_dim, fh.mel_dim, fh.spk_dim_in, fh.spk_dim_out, fh.speech_codebook, fh.cfm_n_steps,
                (double)fh.cfm_inference_cfg_rate);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 3b — single DiT block forward (AdaLN-Zero + bidirectional MHA + FFN)
// ---------------------------------------------------------------------------
//
// Upstream ref (cosyvoice/flow/DiT/modules.py, lucidrains-style block):
//
//   AdaLayerNormZero.forward(x, emb):
//     emb = self.linear(self.silu(emb))                              # (B, 6d)
//     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp \
//         = torch.chunk(emb, 6, dim=1)
//     x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
//     return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
//
//   DiTBlock.forward(x, t, mask=None, rope=...):
//     norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
//     x = x + gate_msa[:, None] * self.attn(norm, rope=rope)
//     ff = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
//     x = x + gate_mlp[:, None] * self.ff(ff)
//
// Notes verified against upstream PyTorch source (modules.py l. 230–245
// for AdaLN, l. 500–531 for DiTBlock):
//   - SiLU is applied to t_emb BEFORE the AdaLN linear (not after).
//   - chunk order is (shift, scale, gate) × {msa, mlp}, in that order.
//   - both LayerNorms (AdaLN's `norm` and DiTBlock's `ff_norm`) are
//     `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)` — affine-free
//     (no γ/β weights), NOT RMSNorm. Use `ggml_norm`.
//   - FFN is plain Linear(d→ff)→GELU(approximate="tanh")→Linear(ff→d). No
//     SiLU, no GLU gating. ggml_gelu maps to the tanh approximation.
//   - RoPE comes from `x_transformers.RotaryEmbedding`. Its `freqs =
//     stack((freqs, freqs), -1)` + adjacent-pair `rotate_half` is the
//     interleaved (GPT-J/Llama-classic) RoPE, which is ggml's
//     GGML_ROPE_TYPE_NORMAL (=0), NOT NEOX. theta=10000 from
//     `cosyvoice3.flow.rope_theta`.

namespace {

ggml_cgraph* cv3_build_flow_dit_block_graph(cosyvoice3_tts_context* ctx, int block_idx, int T) {
    const auto& fh = ctx->flow.hp;
    GGML_ASSERT(block_idx >= 0 && (uint32_t)block_idx < fh.n_dit_layers);
    GGML_ASSERT(T > 0);
    const auto& b = ctx->flow.blocks[block_idx];
    const int d = (int)fh.dit_dim;       // 1024
    const int n_h = (int)fh.dit_heads;   // 16
    const int hd = (int)fh.dit_head_dim; // 64
    const float ln_eps = 1e-6f;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float rope_theta = fh.rope_theta;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_input(x);
    ggml_set_name(x, "dit_x_in");

    ggml_tensor* t_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_input(t_emb);
    ggml_set_name(t_emb, "dit_t_emb_in");

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "dit_positions");

    // AdaLN-Zero modulation: linear(silu(t_emb)) → chunk(6) along last dim.
    // adaln.w has ne=(d, 6d), adaln.b has ne=(6d,) F32.
    ggml_tensor* t_silu = ggml_silu(ctx0, t_emb);
    ggml_tensor* mod = ggml_mul_mat(ctx0, b.adaln_w, t_silu); // (6d,)
    mod = ggml_add(ctx0, mod, b.adaln_b);                     // (6d,)
    const size_t fs = sizeof(float);
    auto chunk = [&](int idx) { return ggml_view_1d(ctx0, mod, d, (size_t)(idx * d) * fs); };
    ggml_tensor* shift_msa = chunk(0);
    ggml_tensor* scale_msa = chunk(1);
    ggml_tensor* gate_msa = chunk(2);
    ggml_tensor* shift_mlp = chunk(3);
    ggml_tensor* scale_mlp = chunk(4);
    ggml_tensor* gate_mlp = chunk(5);

    // Pre-attention LayerNorm (affine-free) + modulation.
    //   h = LN(x) * (1 + scale_msa) + shift_msa
    //     = LN(x) + LN(x) * scale_msa + shift_msa
    ggml_tensor* lnx_a = ggml_norm(ctx0, x, ln_eps);
    ggml_set_name(lnx_a, "dbg_lnx_a");
    ggml_set_output(lnx_a);
    ggml_tensor* h_a = ggml_add(ctx0, lnx_a, ggml_mul(ctx0, lnx_a, scale_msa));
    h_a = ggml_add(ctx0, h_a, shift_msa);
    ggml_set_name(h_a, "dbg_h_a");
    ggml_set_output(h_a);

    // Bidirectional MHA with partial RoPE. Upstream `AttnProcessor` calls
    // `apply_rotary_pos_emb` on the pre-reshape Q/K (shape (B, T, n_h*hd))
    // with `rot_dim = head_dim = 64`. The x_transformers helper splits as
    // `t[..., :rot_dim]` + `t[..., rot_dim:]` and only rotates the first
    // 64 channels — which corresponds to *only head 0*. Heads 1..15 carry
    // no positional info. Match by ropeing the full Q/K tensor with
    // `n_dims=hd` so only the first `hd` channels rotate (NORMAL = adjacent
    // pairs, matches x_transformers' interleaved freqs+rotate_half).
    ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h_a), b.attn_q_b); // (d, T)
    ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, h_a), b.attn_k_b);
    ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h_a), b.attn_v_b);
    // ggml_rope_ext requires ne[2] == positions length. Treat the full
    // d-dim as a "single big head" via reshape (d, 1, T), then apply
    // partial RoPE over `hd` elements.
    Q = ggml_reshape_3d(ctx0, Q, d, 1, T);
    K = ggml_reshape_3d(ctx0, K, d, 1, T);
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, /*n_ctx_orig*/ 0, rope_theta, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, /*n_ctx_orig*/ 0, rope_theta, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    // Now reshape into per-head layout for flash-attn.
    Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_h, T);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask*/ nullptr, attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx0, attn, d, T);
    attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_o_w, attn), b.attn_o_b);
    ggml_set_name(attn, "dbg_attn_raw");
    ggml_set_output(attn);

    // Gated residual: x = x + gate_msa * attn_out.
    ggml_tensor* x_after_attn = ggml_add(ctx0, x, ggml_mul(ctx0, attn, gate_msa));
    ggml_set_name(x_after_attn, "dbg_x_after_attn");
    ggml_set_output(x_after_attn);

    // Pre-FFN LayerNorm + modulation.
    ggml_tensor* lnx_f = ggml_norm(ctx0, x_after_attn, ln_eps);
    ggml_tensor* h_f = ggml_add(ctx0, lnx_f, ggml_mul(ctx0, lnx_f, scale_mlp));
    h_f = ggml_add(ctx0, h_f, shift_mlp);

    // FFN: Linear(d→ff) → GELU(tanh) → Linear(ff→d).
    ggml_tensor* ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l1_w, h_f), b.ffn_l1_b);
    ff = ggml_gelu(ctx0, ff);
    ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l2_w, ff), b.ffn_l2_b);
    ggml_set_name(ff, "dbg_ff_raw");
    ggml_set_output(ff);

    // Gated residual: y = x' + gate_mlp * ff_out.
    ggml_tensor* y = ggml_add(ctx0, x_after_attn, ggml_mul(ctx0, ff, gate_mlp));
    ggml_set_name(y, "dit_block_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    ggml_free(ctx0);
    return gf;
}

} // namespace

extern "C" float* cosyvoice3_tts_run_flow_dit_block(struct cosyvoice3_tts_context* ctx, int block_idx, const float* x,
                                                    int T, const float* t_emb) {
    if (!ctx || !ctx->flow.loaded || !x || !t_emb || T <= 0 || block_idx < 0)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    if ((uint32_t)block_idx >= fh.n_dit_layers) {
        fprintf(stderr, "cosyvoice3_tts: block_idx %d out of range [0, %u)\n", block_idx, fh.n_dit_layers);
        return nullptr;
    }
    const int d = (int)fh.dit_dim;

    // Building any graph in ctx->compute_meta clobbers the cached LM
    // step graph's metadata. Invalidate so the next AR step rebuilds.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_flow_dit_block_graph(ctx, block_idx, T);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: dit_block alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    if (!set_t("dit_x_in", x, (size_t)d * T * sizeof(float)))
        return nullptr;
    if (!set_t("dit_t_emb_in", t_emb, (size_t)d * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T);
    for (int i = 0; i < T; i++)
        pos[i] = i;
    if (!set_t("dit_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: dit_block compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "dit_block_out");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

namespace {

// Build + run the per-block graph and extract a specific named tensor.
// Used by `cosyvoice3_tts_extract_stage` for the `flow_dit_blk_<N>_*`
// stages so the diff harness can hit any intermediate (post-LN,
// post-modulate, post-attn, post-residual, post-FFN, block-out) on the
// same graph build.
float* cv3_extract_flow_dit_stage(cosyvoice3_tts_context* ctx, int block_idx, const float* x, int T, const float* t_emb,
                                  const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !x || !t_emb || T <= 0 || block_idx < 0 || !tensor_name)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    if ((uint32_t)block_idx >= fh.n_dit_layers)
        return nullptr;
    const int d = (int)fh.dit_dim;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_flow_dit_block_graph(ctx, block_idx, T);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("dit_x_in", x, (size_t)d * T * sizeof(float)))
        return nullptr;
    if (!set_t("dit_t_emb_in", t_emb, (size_t)d * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T);
    for (int i = 0; i < T; i++)
        pos[i] = i;
    if (!set_t("dit_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in flow_dit_block graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

} // namespace

extern "C" float* cosyvoice3_tts_solve_flow_euler(struct cosyvoice3_tts_context* ctx, const float* mu, int T_mel,
                                                  const float* spks_proj, const float* cond, const float* x_init,
                                                  int n_steps, float cfg_rate) {
    return cv3_run_solve_euler(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, /*dphi_step0_out*/ nullptr);
}

extern "C" int cosyvoice3_tts_get_flow_hparams(struct cosyvoice3_tts_context* ctx, uint32_t* n_dit_layers,
                                               uint32_t* dit_dim, uint32_t* dit_heads, uint32_t* dit_head_dim,
                                               uint32_t* dit_ff_dim, uint32_t* dit_input_dim, uint32_t* mel_dim,
                                               uint32_t* spk_dim_in, uint32_t* spk_dim_out, uint32_t* cfm_n_steps,
                                               float* cfm_cfg_rate) {
    if (!ctx || !ctx->flow.loaded)
        return -1;
    const auto& fh = ctx->flow.hp;
    if (n_dit_layers)
        *n_dit_layers = fh.n_dit_layers;
    if (dit_dim)
        *dit_dim = fh.dit_dim;
    if (dit_heads)
        *dit_heads = fh.dit_heads;
    if (dit_head_dim)
        *dit_head_dim = fh.dit_head_dim;
    if (dit_ff_dim)
        *dit_ff_dim = fh.dit_ff_dim;
    if (dit_input_dim)
        *dit_input_dim = fh.dit_input_dim;
    if (mel_dim)
        *mel_dim = fh.mel_dim;
    if (spk_dim_in)
        *spk_dim_in = fh.spk_dim_in;
    if (spk_dim_out)
        *spk_dim_out = fh.spk_dim_out;
    if (cfm_n_steps)
        *cfm_n_steps = fh.cfm_n_steps;
    if (cfm_cfg_rate)
        *cfm_cfg_rate = fh.cfm_inference_cfg_rate;
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 3c — pre-lookahead conv stack + InputEmbedding (input pipeline)
// ---------------------------------------------------------------------------
//
// Upstream refs:
//   - PreLookaheadLayer (cosyvoice/transformer/upsample_encoder.py l. 66-103,
//     instantiated for cv3 with in=80, channels=1024, pre_lookahead_len=3):
//
//       outputs = inputs.transpose(1, 2)                   # (B, C, T)
//       outputs = F.pad(outputs, (0, pre_lookahead_len))   # right-pad 3 (lookahead)
//       outputs = F.leaky_relu(conv1(outputs))             # Conv1d(80, 1024, k=4)
//       outputs = F.pad(outputs, (K2 - 1, 0))              # left-pad 2 (causal)
//       outputs = conv2(outputs)                            # Conv1d(1024, 80, k=3)
//       outputs = outputs.transpose(1, 2)                  # (B, T, C)
//       return outputs + inputs                            # residual
//
//   - InputEmbedding (cosyvoice/flow/DiT/dit.py l. 76-98):
//
//       to_cat = [x, cond, text_embed, spks_broadcast]    # 4 × mel_dim = 320
//       x = self.proj(torch.cat(to_cat, dim=-1))          # Linear(320, 1024)
//       x = self.conv_pos_embed(x) + x                    # grouped causal conv
//
//   - CausalConvPositionEmbedding (cosyvoice/flow/DiT/modules.py l. 115-144):
//     2 × `nn.Conv1d(dim, dim, k=31, groups=16, padding=0)` + nn.Mish() with
//     each conv preceded by `F.pad(x, (K-1, 0))` (causal). Note: the helper
//     processes (B, C, T) — channel-first — and returns (B, T, C) after the
//     final transpose.

namespace {

// Mish = x * tanh(softplus(x)). ggml has ggml_softplus + ggml_tanh; chatterbox
// also exposes its own Mish helper but it's static to that .cpp. Local copy
// is cheap and keeps phase 3c self-contained.
ggml_tensor* cv3_mish(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* sp = ggml_softplus(ctx, x);
    ggml_tensor* th = ggml_tanh(ctx, sp);
    return ggml_mul(ctx, x, th);
}

// Causal grouped conv1d for conv_pos_embed.
// h: (C, T) F32  — channel-first ggml layout, C inner / T outer.
// w: (K, C_per_group, C) F16  — sub-group weights at offset c0 = g*C_per_group
//                                 along the out dim.
// b: (C,) F32 — per-channel bias.
// Output: (C, T) with the same length T (left-pad K-1 on T, conv pad=0).
ggml_tensor* cv3_causal_grouped_conv1d(ggml_context* ctx, ggml_tensor* h, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int C_per_g = (int)w->ne[1];
    const int C = (int)w->ne[2];
    const int T = (int)h->ne[1];
    const int G = C / C_per_g;
    GGML_ASSERT(h->ne[0] == C);
    GGML_ASSERT(G * C_per_g == C);

    ggml_tensor* out = nullptr;
    for (int g = 0; g < G; g++) {
        const size_t c0 = (size_t)g * C_per_g;
        // Per-group input slice: (C_per_g, T) — then transpose to (T, C_per_g)
        // matching ggml_conv_1d's expected (T, C_in) data layout (see
        // chatterbox_s3gen::causal_conv1d for the established convention).
        ggml_tensor* h_g = ggml_view_2d(ctx, h, C_per_g, T, h->nb[1], c0 * h->nb[0]);
        h_g = ggml_cont(ctx, ggml_transpose(ctx, h_g)); // (T, C_per_g)
        // Per-group weight slice: (K, C_per_g, C_per_g).
        ggml_tensor* w_g = ggml_view_3d(ctx, w, K, C_per_g, C_per_g, w->nb[1], w->nb[2], c0 * w->nb[2]);
        w_g = ggml_cont(ctx, w_g);
        // Left-pad K-1 (causal): ggml_conv_1d with symmetric pad=K-1 produces
        // (T + K - 1, C_per_g); take the first T entries (drops K-1 from
        // RIGHT) — which equals left-pad-only-by-K-1 conv. Same trick as
        // chatterbox_s3gen::causal_conv1d.
        ggml_tensor* y = ggml_conv_1d(ctx, w_g, h_g, /*stride*/ 1, /*pad*/ K - 1, /*dil*/ 1);
        y = ggml_view_2d(ctx, y, T, C_per_g, y->nb[1], 0);
        y = ggml_cont(ctx, y);
        // Transpose back to (C_per_g, T) and add per-group bias slice.
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        ggml_tensor* b_g = ggml_view_1d(ctx, b, C_per_g, c0 * b->nb[0]);
        y = ggml_add(ctx, y, b_g);
        // Concatenate per-group outputs along channel dim (dim 0).
        out = out ? ggml_concat(ctx, out, y, 0) : y;
    }
    return out;
}

// "Right-padded" dense conv1d — pads right by K-1 (lookahead), then conv
// with stride 1 and zero padding. Output length matches input length.
// Layout: x ne=(T, C_in); w ne=(K, C_in, C_out); b ne=(C_out).
ggml_tensor* cv3_lookahead_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int T = (int)x->ne[0];
    // Symmetric pad K-1 produces output of length T + K - 1. Take the
    // RIGHT T entries (drops K-1 from LEFT) — equivalent to right-pad
    // only by K-1.
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, /*stride*/ 1, /*pad*/ K - 1, /*dil*/ 1);
    y = ggml_view_2d(ctx, y, T, (int)y->ne[1], y->nb[1], (size_t)(K - 1) * y->nb[0]);
    y = ggml_cont(ctx, y);
    if (b) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Mirror of chatterbox_s3gen::causal_conv1d (left-pad K-1) — local copy
// to keep phase 3c self-contained.
ggml_tensor* cv3_causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    const int K = (int)w->ne[0];
    const int T = (int)x->ne[0];
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, /*stride*/ 1, /*pad*/ K - 1, /*dil*/ 1);
    // Take the LEFT T entries (drops K-1 from RIGHT) → left-pad-only conv.
    y = ggml_view_2d(ctx, y, T, (int)y->ne[1], y->nb[1], 0);
    y = ggml_cont(ctx, y);
    if (b) {
        ggml_tensor* b2d = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, b2d);
    }
    return y;
}

// Build the pre-lookahead conv stack graph:
//   ids (T_tok) -> input_embedding -> (T_tok, mel_dim)
//                       └> right-pad 3, conv1 (k=4, 80→1024), leaky_relu
//                                                 └> left-pad 2, conv2 (k=3, 1024→80)
//                                                          └> + residual -> (T_tok, mel_dim)
// Named graph outputs (settable via cv3_extract_pre_la_stage):
//   pre_la_tok_emb, pre_la_c1, pre_la_c2, pre_la_out
ggml_cgraph* cv3_build_pre_la_graph(cosyvoice3_tts_context* ctx, int T_tok) {
    const auto& f = ctx->flow;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_tok);
    ggml_set_input(ids_t);
    ggml_set_name(ids_t, "pre_la_ids_in");

    // Embedding lookup. input_embd.w ne=(mel, vocab); get_rows result ne=(mel, T_tok) F32.
    ggml_tensor* tok_emb = ggml_get_rows(ctx0, f.input_embd_w, ids_t);
    tok_emb = ggml_cont(ctx0, tok_emb);
    ggml_set_name(tok_emb, "pre_la_tok_emb");
    ggml_set_output(tok_emb);

    // Upstream wants (T, C) layout for the conv chain (transpose of get_rows).
    ggml_tensor* x_tc = ggml_cont(ctx0, ggml_transpose(ctx0, tok_emb)); // (T_tok, mel)

    // conv1: lookahead (right-pad K-1=3), kernel 4, in=80 out=1024, then LeakyReLU(0.01).
    ggml_tensor* c1 = cv3_lookahead_conv1d(ctx0, x_tc, f.pre_la_c1_w, f.pre_la_c1_b); // (T_tok, 1024)
    c1 = ggml_leaky_relu(ctx0, c1, 0.01f, /*inplace*/ false);
    // Transpose back to (C, T) for the named-output dump (matches upstream's
    // post-conv1 channel-first layout).
    ggml_tensor* c1_out = ggml_cont(ctx0, ggml_transpose(ctx0, c1)); // (1024, T_tok)
    ggml_set_name(c1_out, "pre_la_c1");
    ggml_set_output(c1_out);

    // conv2: causal (left-pad K-1=2), kernel 3, in=1024 out=80.
    ggml_tensor* c2 = cv3_causal_conv1d(ctx0, c1, f.pre_la_c2_w, f.pre_la_c2_b); // (T_tok, mel)
    ggml_tensor* c2_out = ggml_cont(ctx0, ggml_transpose(ctx0, c2));             // (mel, T_tok)
    ggml_set_name(c2_out, "pre_la_c2");
    ggml_set_output(c2_out);

    // Residual: + tok_emb (channel-first).
    ggml_tensor* y = ggml_add(ctx0, c2_out, tok_emb);
    ggml_set_name(y, "pre_la_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    // c1_out and tok_emb are SIDE branches (not on the path to y after the
    // transpose ops materialise them as cont-tensors). Expand them
    // explicitly so they survive into the executed graph.
    ggml_build_forward_expand(gf, tok_emb);
    ggml_build_forward_expand(gf, c1_out);
    ggml_build_forward_expand(gf, c2_out);

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_pre_la_stage(cosyvoice3_tts_context* ctx, const int32_t* ids, int T_tok, const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !ids || T_tok <= 0 || !tensor_name)
        return nullptr;
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_pre_la_graph(ctx, T_tok);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_tensor* ids_t = ggml_graph_get_tensor(gf, "pre_la_ids_in");
    if (!ids_t)
        return nullptr;
    ggml_backend_tensor_set(ids_t, ids, 0, (size_t)T_tok * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: pre_la compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in pre_la graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// Build the InputEmbedding (input pipeline) graph:
//   spks_raw (spk_in)   ──F.normalize──> ─spk_affine─> spks_proj (spk_out=80)
//                                                    └ broadcast to (T_mel, 80)
//   pre_la (T_mel, 80) ┐
//   x_noisy (T_mel, 80)├──cat dim=-1──> (T_mel, 320) ─in_proj(320→1024)─> proj
//   cond    (T_mel, 80)│                                                    │
//   spks_bc (T_mel, 80)┘                                                    │
//                                                                          ├──+──> in_pipe_out
//                                                            conv_pos_embed(proj)
//                                                          (2× grouped causal
//                                                           conv1d-31 + Mish)
//
// Cat order per upstream InputEmbedding.forward:
//   to_cat = [x, cond, text_embed, spks] — x first.
//
// Named outputs (settable via cv3_extract_in_pipe_stage):
//   in_pipe_spk    (spk_dim_out,)
//   in_pipe_cat    (dit_input_dim, T_mel)  — channel-first
//   in_pipe_proj   (dit_dim, T_mel)
//   in_pipe_pos    (dit_dim, T_mel)  — conv_pos_embed(proj) without residual
//   in_pipe_out    (dit_dim, T_mel)  — proj + conv_pos_embed(proj)
ggml_cgraph* cv3_build_in_pipe_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& fh = ctx->flow.hp;
    const auto& f = ctx->flow;
    const int mel = (int)fh.mel_dim;
    const int spk_in = (int)fh.spk_dim_in;
    const int spk_out = (int)fh.spk_dim_out;
    const int dit_in_dim = (int)fh.dit_input_dim;
    GGML_ASSERT(spk_out == mel);
    GGML_ASSERT(dit_in_dim == 4 * mel); // x + cond + mu + spks = 4 × mel_dim

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 2048, false);

    // ---- Inputs ----
    ggml_tensor* pre_la = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(pre_la);
    ggml_set_name(pre_la, "in_pipe_pre_la_in"); // (mel, T_mel)
    ggml_tensor* spk = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, spk_in);
    ggml_set_input(spk);
    ggml_set_name(spk, "in_pipe_spk_in"); // (spk_in,)
    ggml_tensor* x_noisy = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(x_noisy);
    ggml_set_name(x_noisy, "in_pipe_x_in"); // (mel, T_mel)
    ggml_tensor* cond = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(cond);
    ggml_set_name(cond, "in_pipe_cond_in");

    // ---- spk projection: F.normalize(spk, dim=1) -> spk_affine ----
    // F.normalize: x / max(||x||_2, eps), default eps=1e-12.
    ggml_tensor* spk_2d = ggml_reshape_2d(ctx0, spk, spk_in, 1); // (spk_in, 1)
    ggml_tensor* spk_norm = ggml_rms_norm(ctx0, spk_2d, 0.0f);
    // ggml_rms_norm divides by sqrt(mean(x^2) + eps) = sqrt(sum(x^2)/N + eps).
    // F.normalize divides by sqrt(sum(x^2) + eps_l2). Compensate by
    // multiplying with sqrt(N) to convert "RMS" → "L2 norm" denominator.
    spk_norm = ggml_scale(ctx0, spk_norm, 1.0f / std::sqrt((float)spk_in));
    ggml_tensor* spk_proj = ggml_mul_mat(ctx0, f.spk_affine_w, spk_norm); // (spk_out, 1)
    spk_proj = ggml_add(ctx0, spk_proj, f.spk_affine_b);
    // ggml_cont so the named output owns its buffer — set_output on a
    // reshape view of an add result is fragile under sched allocation.
    spk_proj = ggml_cont(ctx0, ggml_reshape_1d(ctx0, spk_proj, spk_out));
    ggml_set_name(spk_proj, "in_pipe_spk");
    ggml_set_output(spk_proj);

    // ---- Broadcast spk over T_mel: (spk_out,) → (spk_out, T_mel) ----
    // Use ggml_repeat with a same-shape target.
    ggml_tensor* spk_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, spk_out, T_mel);
    ggml_tensor* spk_bc = ggml_repeat(ctx0, spk_proj, spk_template);

    // ---- Concat [x, cond, mu(=pre_la), spks] along channel dim ----
    // Each piece is (mel, T_mel) col-major; concat along dim 0 stacks
    // channels → (4*mel = dit_input_dim, T_mel).
    ggml_tensor* cat01 = ggml_concat(ctx0, x_noisy, cond, 0);   // (2*mel, T)
    ggml_tensor* cat012 = ggml_concat(ctx0, cat01, pre_la, 0);  // (3*mel, T)
    ggml_tensor* catted = ggml_concat(ctx0, cat012, spk_bc, 0); // (4*mel = dit_in_dim, T)
    ggml_set_name(catted, "in_pipe_cat");
    ggml_set_output(catted);

    // ---- in_proj: Linear(320, 1024) ----
    ggml_tensor* proj = ggml_mul_mat(ctx0, f.dit_in_proj_w, catted); // (1024, T)
    proj = ggml_add(ctx0, proj, f.dit_in_proj_b);
    ggml_set_name(proj, "in_pipe_proj");
    ggml_set_output(proj);

    // ---- conv_pos_embed: 2× grouped causal conv1d (k=31, groups=16) + Mish ----
    ggml_tensor* pos = cv3_causal_grouped_conv1d(ctx0, proj, f.dit_conv_pos_c1_w, f.dit_conv_pos_c1_b);
    pos = cv3_mish(ctx0, pos);
    pos = cv3_causal_grouped_conv1d(ctx0, pos, f.dit_conv_pos_c2_w, f.dit_conv_pos_c2_b);
    pos = cv3_mish(ctx0, pos);
    ggml_set_name(pos, "in_pipe_pos");
    ggml_set_output(pos);

    // ---- Residual: in_pipe_out = pos + proj ----
    ggml_tensor* y = ggml_add(ctx0, pos, proj);
    ggml_set_name(y, "in_pipe_out");
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    // Side branches not directly on the path to y.
    ggml_build_forward_expand(gf, spk_proj);
    ggml_build_forward_expand(gf, catted);
    ggml_build_forward_expand(gf, proj);
    ggml_build_forward_expand(gf, pos);

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_in_pipe_stage(cosyvoice3_tts_context* ctx, const float* pre_la_out, int T_mel, const float* spk_emb,
                                 const float* x_noisy, const float* cond, const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !pre_la_out || !spk_emb || !x_noisy || !cond || T_mel <= 0 || !tensor_name)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const int spk_in = (int)fh.spk_dim_in;
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_in_pipe_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: in_pipe alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("in_pipe_pre_la_in", pre_la_out, (size_t)mel * T_mel * sizeof(float)))
        return nullptr;
    if (!set_t("in_pipe_spk_in", spk_emb, (size_t)spk_in * sizeof(float)))
        return nullptr;
    if (!set_t("in_pipe_x_in", x_noisy, (size_t)mel * T_mel * sizeof(float)))
        return nullptr;
    if (!set_t("in_pipe_cond_in", cond, (size_t)mel * T_mel * sizeof(float)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: in_pipe compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in in_pipe graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Phase 3d-A — full 22-block DiT estimator forward + norm_out + proj_out
// ---------------------------------------------------------------------------
//
// Composes the per-block forward (phase 3b) into the full 22-layer stack,
// then applies `AdaLayerNormZero_Final` (norm_out: 2-chunk scale/shift) and
// `Linear(1024, 80)` (proj_out). This is exactly the function the CFM Euler
// solver calls inside its 10-step loop.
//
// Upstream ref (`cosyvoice/flow/DiT/dit.py::DiT.forward`, post-input-embed):
//
//     for block in self.transformer_blocks:
//         x = block(x, t, mask=attn_mask.bool(), rope=rope)
//     x = self.norm_out(x, t)                # AdaLayerNormZero_Final
//     output = self.proj_out(x).transpose(1, 2)
//
// AdaLayerNormZero_Final (`modules.py::AdaLayerNormZero_Final`):
//     emb = self.linear(self.silu(emb))      # (B, 2*dim)
//     scale, shift = torch.chunk(emb, 2, dim=1)  # scale FIRST
//     x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
//
// Note: AdaLN-Final's chunk order is (scale, shift) — opposite of the
// per-block AdaLN which is (shift, scale, gate) × {msa, mlp}.

// Inline per-block forward (matches cv3_build_flow_dit_block_graph but
// without the debug `dbg_*` named outputs). Takes pre-computed
// `silu(t_emb)` so we don't recompute the silu 22 times.
ggml_tensor* cv3_dit_block_apply(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* t_silu, ggml_tensor* positions,
                                 const cv3_dit_block& b, int d, int n_h, int hd, float rope_theta) {
    const float ln_eps = 1e-6f;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = (int)x->ne[1];

    ggml_tensor* mod = ggml_mul_mat(ctx0, b.adaln_w, t_silu); // (6d,)
    mod = ggml_add(ctx0, mod, b.adaln_b);
    const size_t fs = sizeof(float);
    auto chunk = [&](int idx) { return ggml_view_1d(ctx0, mod, d, (size_t)(idx * d) * fs); };
    ggml_tensor* shift_msa = chunk(0);
    ggml_tensor* scale_msa = chunk(1);
    ggml_tensor* gate_msa = chunk(2);
    ggml_tensor* shift_mlp = chunk(3);
    ggml_tensor* scale_mlp = chunk(4);
    ggml_tensor* gate_mlp = chunk(5);

    ggml_tensor* lnx_a = ggml_norm(ctx0, x, ln_eps);
    ggml_tensor* h_a = ggml_add(ctx0, lnx_a, ggml_mul(ctx0, lnx_a, scale_msa));
    h_a = ggml_add(ctx0, h_a, shift_msa);

    ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h_a), b.attn_q_b);
    ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, h_a), b.attn_k_b);
    ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h_a), b.attn_v_b);
    Q = ggml_reshape_3d(ctx0, Q, d, 1, T);
    K = ggml_reshape_3d(ctx0, K, d, 1, T);
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_h, T);
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx0, attn, d, T);
    attn = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_o_w, attn), b.attn_o_b);
    ggml_tensor* x_after_attn = ggml_add(ctx0, x, ggml_mul(ctx0, attn, gate_msa));

    ggml_tensor* lnx_f = ggml_norm(ctx0, x_after_attn, ln_eps);
    ggml_tensor* h_f = ggml_add(ctx0, lnx_f, ggml_mul(ctx0, lnx_f, scale_mlp));
    h_f = ggml_add(ctx0, h_f, shift_mlp);
    ggml_tensor* ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l1_w, h_f), b.ffn_l1_b);
    ff = ggml_gelu(ctx0, ff);
    ff = ggml_add(ctx0, ggml_mul_mat(ctx0, b.ffn_l2_w, ff), b.ffn_l2_b);
    return ggml_add(ctx0, x_after_attn, ggml_mul(ctx0, ff, gate_mlp));
}

ggml_cgraph* cv3_build_dit_full_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& fh = ctx->flow.hp;
    const auto& f = ctx->flow;
    const int d = (int)fh.dit_dim;
    const int n_h = (int)fh.dit_heads;
    const int hd = (int)fh.dit_head_dim;
    const int L = (int)fh.n_dit_layers;
    const float rope_theta = fh.rope_theta;
    const float ln_eps = 1e-6f;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    // 22 blocks * ~40 ops + norm_out + proj_out + intermediates ⇒ a few
    // thousand nodes. 8192 leaves ample headroom.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_mel);
    ggml_set_input(x);
    ggml_set_name(x, "dit_full_x_in");
    ggml_tensor* t_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d);
    ggml_set_input(t_emb);
    ggml_set_name(t_emb, "dit_full_t_emb_in");
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_mel);
    ggml_set_input(positions);
    ggml_set_name(positions, "dit_full_positions");

    // Shared silu(t_emb) — used by every block AND by norm_out.
    ggml_tensor* t_silu = ggml_silu(ctx0, t_emb);

    for (int il = 0; il < L; il++) {
        x = cv3_dit_block_apply(ctx0, x, t_silu, positions, f.blocks[il], d, n_h, hd, rope_theta);
    }

    // norm_out: AdaLayerNormZero_Final — chunk(emb, 2) yields (scale, shift)
    // in THIS order (per upstream `AdaLayerNormZero_Final.forward`).
    ggml_tensor* nmod = ggml_mul_mat(ctx0, f.dit_norm_out_w, t_silu); // (2d,)
    nmod = ggml_add(ctx0, nmod, f.dit_norm_out_b);
    const size_t fs = sizeof(float);
    ggml_tensor* nscale = ggml_view_1d(ctx0, nmod, d, 0);
    ggml_tensor* nshift = ggml_view_1d(ctx0, nmod, d, (size_t)d * fs);
    ggml_tensor* lnx = ggml_norm(ctx0, x, ln_eps);
    ggml_tensor* normed = ggml_add(ctx0, lnx, ggml_mul(ctx0, lnx, nscale));
    normed = ggml_add(ctx0, normed, nshift);
    ggml_set_name(normed, "dit_full_norm");
    ggml_set_output(normed);

    // proj_out: Linear(dit_dim, mel_dim).
    ggml_tensor* mel_out = ggml_mul_mat(ctx0, f.dit_proj_out_w, normed); // (mel_dim, T_mel)
    mel_out = ggml_add(ctx0, mel_out, f.dit_proj_out_b);
    ggml_set_name(mel_out, "dit_full_out");
    ggml_set_output(mel_out);
    ggml_build_forward_expand(gf, mel_out);
    ggml_build_forward_expand(gf, normed); // side branch — keep it alive in the executed graph

    ggml_free(ctx0);
    return gf;
}

float* cv3_extract_dit_full_stage(cosyvoice3_tts_context* ctx, const float* x, int T_mel, const float* t_emb,
                                  const char* tensor_name) {
    if (!ctx || !ctx->flow.loaded || !x || !t_emb || T_mel <= 0 || !tensor_name)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    const int d = (int)fh.dit_dim;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_dit_full_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "cosyvoice3_tts: dit_full alloc_graph failed\n");
        return nullptr;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    if (!set_t("dit_full_x_in", x, (size_t)d * T_mel * sizeof(float)))
        return nullptr;
    if (!set_t("dit_full_t_emb_in", t_emb, (size_t)d * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T_mel);
    for (int i = 0; i < T_mel; i++)
        pos[i] = i;
    if (!set_t("dit_full_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: dit_full compute failed\n");
        return nullptr;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, tensor_name);
    if (!out_t) {
        fprintf(stderr, "cosyvoice3_tts: tensor '%s' not in dit_full graph\n", tensor_name);
        return nullptr;
    }
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// ---------------------------------------------------------------------------
// Phase 3d-B — CFM Euler ODE driver
// ---------------------------------------------------------------------------
//
// Upstream refs (cosyvoice/flow/flow_matching.py):
//
//   CausalConditionalCFM.forward(mu, mask, n_timesteps, spks, cond):
//     z = self.rand_noise[:, :, :mu.size(2)] * temperature
//     t_span = linspace(0, 1, n_timesteps + 1)
//     if t_scheduler == 'cosine':
//         t_span = 1 - cos(t_span * pi/2)
//     return solve_euler(z, t_span, mu, mask, spks, cond)
//
//   ConditionalCFM.solve_euler(x, t_span, mu, mask, spks, cond):
//     t, dt = t_span[0], t_span[1] - t_span[0]
//     for step in range(1, len(t_span)):
//         # Classifier-Free Guidance with a batched (2, ...) call:
//         x_in    = stack([x,    x])     # both identical
//         mu_in   = stack([mu,   0])     # uncond branch zeroed
//         spks_in = stack([spks, 0])
//         cond_in = stack([cond, 0])
//         t_in    = [t, t]
//         dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
//         dphi_dt, cfg_dphi_dt = split(dphi_dt, [batch, batch])
//         dphi_dt = (1 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt
//         x = x + dt * dphi_dt
//         t = t + dt
//         dt = t_span[step + 1] - t
//     return x
//
// Our port runs the estimator TWICE per step (cond + uncond, both B=1)
// rather than once with B=2; the numerical result is identical (the
// estimator is deterministic and batch-wise independent) and the
// per-call overhead is small compared to the 22-block forward itself.
//
// For diff parity, we accept `x_init` from the caller — generating
// `torch.manual_seed(0); randn([1,80,15000])[:, :, :T_mel]` bit-exactly
// in C++ would require porting torch's Mersenne-Twister + Box-Muller,
// which is out of scope for the ODE driver. The Python ref harness
// dumps the seeded noise into the GGUF archive and the diff harness
// hands it in via extract_stage.

// Sinusoidal positional embedding for the Euler time step. Mirrors
// `SinusPositionEmbedding.forward(x, scale=1000)` in
// cosyvoice/flow/DiT/modules.py l. 76-83. Returns a freshly allocated
// vector of size `dim` (= time_mlp.0 input = 256).
std::vector<float> cv3_compute_sin_emb(float t, int dim, float scale = 1000.0f) {
    const int half = dim / 2;
    std::vector<float> out((size_t)dim);
    const double decay = std::log(10000.0) / (half - 1);
    for (int i = 0; i < half; i++) {
        const double freq = std::exp(-(double)i * decay);
        const double pos = (double)scale * (double)t * freq;
        out[(size_t)i] = (float)std::sin(pos);          // first half: sin
        out[(size_t)(i + half)] = (float)std::cos(pos); // second half: cos
    }
    return out;
}

// Build the FULL estimator forward graph: time_mlp + InputEmbedding +
// 22 DiT blocks + AdaLN-Final norm_out + proj_out.
//
// Inputs (set externally):
//   "est_x_in"      [mel_dim, T_mel]    F32  current Euler iterate
//   "est_mu_in"     [mel_dim, T_mel]    F32  pre_la output (post repeat-interleave)
//   "est_spks_in"   [spk_dim_out]       F32  ALREADY projected (caller-supplied)
//   "est_cond_in"   [mel_dim, T_mel]    F32  prompt-prefix conditioning
//   "est_sin_emb"   [freq_embed=256]    F32  sinusoidal(t, 256)
//
// Output: "est_dphi" [mel_dim, T_mel]  F32  estimator velocity prediction.
ggml_cgraph* cv3_build_estimator_full_graph(cosyvoice3_tts_context* ctx, int T_mel) {
    const auto& fh = ctx->flow.hp;
    const auto& f = ctx->flow;
    const int mel = (int)fh.mel_dim;
    const int spk_out = (int)fh.spk_dim_out;
    const int dit_in_dim = (int)fh.dit_input_dim; // 320
    const int dit_dim = (int)fh.dit_dim;          // 1024
    const int n_h = (int)fh.dit_heads;
    const int hd = (int)fh.dit_head_dim;
    const int L = (int)fh.n_dit_layers;
    const int freq_embed_dim = 256;
    const float rope_theta = fh.rope_theta;
    const float ln_eps = 1e-6f;
    GGML_ASSERT(spk_out == mel);
    GGML_ASSERT(dit_in_dim == 4 * mel);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ---- Inputs ----
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(x);
    ggml_set_name(x, "est_x_in");
    ggml_tensor* mu = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(mu);
    ggml_set_name(mu, "est_mu_in");
    ggml_tensor* spks_proj_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, spk_out);
    ggml_set_input(spks_proj_in);
    ggml_set_name(spks_proj_in, "est_spks_in");
    ggml_tensor* cond = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mel, T_mel);
    ggml_set_input(cond);
    ggml_set_name(cond, "est_cond_in");
    ggml_tensor* sin_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, freq_embed_dim);
    ggml_set_input(sin_emb);
    ggml_set_name(sin_emb, "est_sin_emb");
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_mel);
    ggml_set_input(positions);
    ggml_set_name(positions, "est_positions");

    // ---- TimestepEmbedding's time_mlp: Linear -> SiLU -> Linear ----
    ggml_tensor* t_emb = ggml_mul_mat(ctx0, f.dit_time_mlp_0_w, sin_emb); // (dit_dim,)
    t_emb = ggml_add(ctx0, t_emb, f.dit_time_mlp_0_b);
    t_emb = ggml_silu(ctx0, t_emb);
    t_emb = ggml_mul_mat(ctx0, f.dit_time_mlp_2_w, t_emb);
    t_emb = ggml_add(ctx0, t_emb, f.dit_time_mlp_2_b);

    // ---- Broadcast spks_proj over T_mel (shape (spk_out, T_mel)) ----
    ggml_tensor* spk_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, spk_out, T_mel);
    ggml_tensor* spk_bc = ggml_repeat(ctx0, spks_proj_in, spk_template);

    // ---- InputEmbedding concat (cat dim = channels): [x, cond, mu, spk_bc] ----
    ggml_tensor* cat01 = ggml_concat(ctx0, x, cond, 0);         // (2*mel, T)
    ggml_tensor* cat012 = ggml_concat(ctx0, cat01, mu, 0);      // (3*mel, T)
    ggml_tensor* catted = ggml_concat(ctx0, cat012, spk_bc, 0); // (4*mel = dit_input_dim, T)

    // ---- in_proj: Linear(320, 1024) ----
    ggml_tensor* proj = ggml_mul_mat(ctx0, f.dit_in_proj_w, catted); // (dit_dim, T)
    proj = ggml_add(ctx0, proj, f.dit_in_proj_b);

    // ---- conv_pos_embed: 2× grouped causal conv1d-31 + Mish, + residual ----
    ggml_tensor* pos_h = cv3_causal_grouped_conv1d(ctx0, proj, f.dit_conv_pos_c1_w, f.dit_conv_pos_c1_b);
    pos_h = cv3_mish(ctx0, pos_h);
    pos_h = cv3_causal_grouped_conv1d(ctx0, pos_h, f.dit_conv_pos_c2_w, f.dit_conv_pos_c2_b);
    pos_h = cv3_mish(ctx0, pos_h);
    ggml_tensor* h = ggml_add(ctx0, pos_h, proj); // (dit_dim, T_mel) — input to first DiT block.

    // ---- Shared silu(t_emb) for all 22 blocks + norm_out ----
    ggml_tensor* t_silu = ggml_silu(ctx0, t_emb);

    // ---- 22-block DiT stack ----
    for (int il = 0; il < L; il++) {
        h = cv3_dit_block_apply(ctx0, h, t_silu, positions, f.blocks[il], dit_dim, n_h, hd, rope_theta);
    }

    // ---- norm_out (AdaLN-Final, chunk order (scale, shift)) ----
    ggml_tensor* nmod = ggml_mul_mat(ctx0, f.dit_norm_out_w, t_silu); // (2*dit_dim,)
    nmod = ggml_add(ctx0, nmod, f.dit_norm_out_b);
    const size_t fs = sizeof(float);
    ggml_tensor* nscale = ggml_view_1d(ctx0, nmod, dit_dim, 0);
    ggml_tensor* nshift = ggml_view_1d(ctx0, nmod, dit_dim, (size_t)dit_dim * fs);
    ggml_tensor* lnx = ggml_norm(ctx0, h, ln_eps);
    ggml_tensor* normed = ggml_add(ctx0, lnx, ggml_mul(ctx0, lnx, nscale));
    normed = ggml_add(ctx0, normed, nshift);

    // ---- proj_out: Linear(dit_dim, mel_dim) ----
    ggml_tensor* dphi = ggml_mul_mat(ctx0, f.dit_proj_out_w, normed); // (mel, T_mel)
    dphi = ggml_add(ctx0, dphi, f.dit_proj_out_b);
    ggml_set_name(dphi, "est_dphi");
    ggml_set_output(dphi);
    ggml_build_forward_expand(gf, dphi);

    ggml_free(ctx0);
    return gf;
}

// Run the full-estimator graph once. Returns a malloc'd float[T_mel*mel_dim]
// buffer (caller frees).
float* cv3_run_estimator_full(cosyvoice3_tts_context* ctx, const float* x, int T_mel, const float* mu,
                              const float* spks_proj, const float* cond, const float* sin_emb_t) {
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const int spk_out = (int)fh.spk_dim_out;
    const int freq_embed_dim = 256;

    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    ggml_cgraph* gf = cv3_build_estimator_full_graph(ctx, T_mel);
    if (!gf)
        return nullptr;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    const size_t mel_bytes = (size_t)mel * T_mel * sizeof(float);
    if (!set_t("est_x_in", x, mel_bytes))
        return nullptr;
    if (!set_t("est_mu_in", mu, mel_bytes))
        return nullptr;
    if (!set_t("est_spks_in", spks_proj, (size_t)spk_out * sizeof(float)))
        return nullptr;
    if (!set_t("est_cond_in", cond, mel_bytes))
        return nullptr;
    if (!set_t("est_sin_emb", sin_emb_t, (size_t)freq_embed_dim * sizeof(float)))
        return nullptr;
    std::vector<int32_t> pos((size_t)T_mel);
    for (int i = 0; i < T_mel; i++)
        pos[i] = i;
    if (!set_t("est_positions", pos.data(), pos.size() * sizeof(int32_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cosyvoice3_tts: estimator_full compute failed\n");
        return nullptr;
    }
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "est_dphi");
    if (!out_t)
        return nullptr;
    const size_t n = (size_t)ggml_nelements(out_t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(out_t, out, 0, n * sizeof(float));
    return out;
}

// 10-step (configurable) cosine-schedule Euler ODE with classifier-free
// guidance. Inputs are caller-provided post-projection — `mu` is post
// `pre_la + repeat_interleave`, `spks_proj` is post `F.normalize +
// spk_affine`. `x_init` is the initial noise (caller matches upstream's
// seeded `rand_noise[:, :, :T_mel]`).
//
// Returns malloc'd float[T_mel*mel_dim] with the final mel iterate.
// If `dphi_step0_out` is non-null, also writes the post-CFG dphi_dt at
// step 1 (the FIRST loop iteration — upstream's `step in range(1, ...)`)
// there; the caller must pre-allocate T_mel*mel_dim floats.
float* cv3_run_solve_euler(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                           const float* cond, const float* x_init, int n_steps, float cfg_rate, float* dphi_step0_out) {
    if (!ctx || !ctx->flow.loaded || !mu || !spks_proj || !cond || !x_init || T_mel <= 0 || n_steps <= 0)
        return nullptr;
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const int spk_out = (int)fh.spk_dim_out;
    const size_t mel_n = (size_t)mel * T_mel;

    // Cosine t-schedule: t_span[k] = 1 - cos(k/N * pi/2), k in [0, N+1].
    std::vector<double> t_span((size_t)(n_steps + 1));
    for (int k = 0; k <= n_steps; k++) {
        const double u = (double)k / (double)n_steps; // linspace(0, 1, N+1)
        t_span[(size_t)k] = 1.0 - std::cos(u * 0.5 * M_PI);
    }

    // x starts as the caller-supplied init noise.
    std::vector<float> x(mel_n);
    std::memcpy(x.data(), x_init, mel_n * sizeof(float));

    // Pre-allocate the zero buffers used for the CFG uncond pass.
    std::vector<float> mu_zero(mel_n, 0.0f);
    std::vector<float> cond_zero(mel_n, 0.0f);
    std::vector<float> spks_zero((size_t)spk_out, 0.0f);

    double t = t_span[0];
    double dt = t_span[1] - t_span[0];
    for (int step = 1; step <= n_steps; step++) {
        std::vector<float> sin_emb = cv3_compute_sin_emb((float)t, 256);

        float* dphi_cond = cv3_run_estimator_full(ctx, x.data(), T_mel, mu, spks_proj, cond, sin_emb.data());
        if (!dphi_cond)
            return nullptr;
        float* dphi_unc = cv3_run_estimator_full(ctx, x.data(), T_mel, mu_zero.data(), spks_zero.data(),
                                                 cond_zero.data(), sin_emb.data());
        if (!dphi_unc) {
            free(dphi_cond);
            return nullptr;
        }

        // CFG combine: dphi = (1 + cfg) * dphi_cond - cfg * dphi_unc.
        const float w_cond = 1.0f + cfg_rate;
        const float w_unc = cfg_rate;
        for (size_t i = 0; i < mel_n; i++) {
            dphi_cond[i] = w_cond * dphi_cond[i] - w_unc * dphi_unc[i];
        }
        free(dphi_unc);

        if (step == 1 && dphi_step0_out) {
            std::memcpy(dphi_step0_out, dphi_cond, mel_n * sizeof(float));
        }

        // Euler step: x = x + dt * dphi.
        const float dtf = (float)dt;
        for (size_t i = 0; i < mel_n; i++) {
            x[i] += dtf * dphi_cond[i];
        }
        free(dphi_cond);

        t = t + dt;
        if (step < n_steps) {
            dt = t_span[(size_t)(step + 1)] - t;
        }
    }

    float* out = (float*)malloc(mel_n * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, x.data(), mel_n * sizeof(float));
    return out;
}

float* cv3_extract_euler_stage(cosyvoice3_tts_context* ctx, const float* mu, int T_mel, const float* spks_proj,
                               const float* cond, const float* x_init, int n_steps, float cfg_rate,
                               const char* tensor_name) {
    const auto& fh = ctx->flow.hp;
    const int mel = (int)fh.mel_dim;
    const size_t mel_n = (size_t)mel * T_mel;

    if (strcmp(tensor_name, "euler_out") == 0) {
        return cv3_run_solve_euler(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, nullptr);
    }
    if (strcmp(tensor_name, "euler_dphi_step0") == 0) {
        std::vector<float> dphi_buf(mel_n);
        float* mel_out =
            cv3_run_solve_euler(ctx, mu, T_mel, spks_proj, cond, x_init, n_steps, cfg_rate, dphi_buf.data());
        if (!mel_out)
            return nullptr;
        free(mel_out);
        float* out = (float*)malloc(mel_n * sizeof(float));
        if (!out)
            return nullptr;
        std::memcpy(out, dphi_buf.data(), mel_n * sizeof(float));
        return out;
    }
    return nullptr;
}

} // namespace
