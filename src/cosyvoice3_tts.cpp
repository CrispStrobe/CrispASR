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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
    ggml_tensor* token_embd_w = nullptr;      // [d_model, text_vocab]
    ggml_tensor* output_norm_w = nullptr;     // [d_model]
    ggml_tensor* text_output_w = nullptr;     // [d_model, text_vocab] (unused in speech AR)
    ggml_tensor* speech_embd_w = nullptr;     // [d_model, speech_vocab]
    ggml_tensor* speech_lm_head_w = nullptr;  // [d_model, speech_vocab]
    std::vector<cv3_qwen2_block> blocks;
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
        fprintf(stderr,
                "cosyvoice3_tts: kv cache %d MiB k=%s v=%s (head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
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
    fprintf(stderr, "cosyvoice3_tts: unknown stage '%s'\n", stage_name);
    return nullptr;
}
