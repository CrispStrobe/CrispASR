// src/tada_tts.cpp — TADA-3B-ML TTS runtime.
//
// HumeAI/tada-3b-ml: Llama-3.2-3B + VibeVoiceDiffusionHead + TADA codec.
// See tada_tts.h for the C ABI.

#include "tada_tts.h"
#include "core/gguf_loader.h"
#include "core/attention.h"
#include "core/ffn.h"
#include "core/bpe.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────── internal types ────────────────────────────

namespace {

struct tada_hp {
    // Llama backbone
    uint32_t n_layers    = 28;
    uint32_t d_model     = 3072;
    uint32_t n_heads     = 24;
    uint32_t n_kv_heads  = 8;
    uint32_t head_dim    = 128;
    uint32_t ff_dim      = 8192;
    uint32_t vocab_size  = 128256;
    uint32_t max_pos     = 131072;
    float    rope_theta  = 500000.0f;
    float    rms_norm_eps = 1e-5f;

    // TADA-specific
    uint32_t acoustic_dim     = 512;
    uint32_t num_time_classes = 256;
    uint32_t num_time_bits    = 8;
    uint32_t time_dim         = 16;
    uint32_t shift_acoustic   = 5;
    uint32_t head_layers      = 6;
    float    head_ffn_ratio   = 4.0f;
    uint32_t fm_hidden        = 3072;  // bottleneck_dim or d_model
    uint32_t fm_latent        = 528;   // acoustic_dim + time_dim
    float    acoustic_mean    = 0.0f;
    float    acoustic_std     = 1.5f;
    bool     has_bottleneck   = false;

    // Derived
    uint32_t fm_ffn_dim() const { return (uint32_t)(fm_hidden * head_ffn_ratio); }
    uint32_t total_dim() const { return acoustic_dim + time_dim; }
};

struct tada_layer {
    ggml_tensor* attn_norm_w;
    ggml_tensor* attn_q_w;
    ggml_tensor* attn_k_w;
    ggml_tensor* attn_v_w;
    ggml_tensor* attn_output_w;
    ggml_tensor* ffn_norm_w;
    ggml_tensor* ffn_gate_w;
    ggml_tensor* ffn_up_w;
    ggml_tensor* ffn_down_w;
};

struct tada_talker {
    ggml_tensor* token_embd_w;
    std::vector<tada_layer> blocks;
    ggml_tensor* output_norm_w;
    ggml_tensor* output_w;  // lm_head

    // TADA-specific embeddings
    ggml_tensor* acoustic_proj_w;      // (acoustic_dim, d_model)
    ggml_tensor* acoustic_proj_b;      // (d_model,) — may be null
    ggml_tensor* time_start_embd_w;    // (num_time_classes, d_model)
    ggml_tensor* time_end_embd_w;      // (num_time_classes, d_model)
    ggml_tensor* acoustic_mask_embd_w; // (2, d_model)
    ggml_tensor* bottleneck_proj_w;    // (d_model, bottleneck_dim) or null
};

struct tada_fm_layer {
    ggml_tensor* ffn_gate_w;
    ggml_tensor* ffn_up_w;
    ggml_tensor* ffn_down_w;
    ggml_tensor* norm_w;
    ggml_tensor* adaln_w;  // SiLU → Linear(cond_dim, 3*embed_dim)
};

struct tada_fm_head {
    ggml_tensor* noisy_proj_w;   // (latent, hidden)
    ggml_tensor* cond_proj_w;    // (hidden, cond_dim)
    ggml_tensor* t_emb_mlp0_w;  // (freq_dim, hidden)
    ggml_tensor* t_emb_mlp1_w;  // (hidden, hidden)
    std::vector<tada_fm_layer> layers;
    ggml_tensor* final_norm_w;   // RMSNorm (no affine — weight may still exist as ones)
    ggml_tensor* final_proj_w;   // (hidden, latent)
    ggml_tensor* final_adaln_w;  // (cond_dim, 2*hidden)
};

struct tada_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

} // anonymous namespace

struct tada_context {
    tada_context_params params;
    tada_hp hp;
    tada_vocab vocab;
    tada_talker talker;
    tada_fm_head fm;

    ggml_backend_t backend     = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w        = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context*          kv_ctx = nullptr;
    ggml_backend_buffer_t  kv_buf = nullptr;
    ggml_tensor*           kv_k   = nullptr;
    ggml_tensor*           kv_v   = nullptr;
    int kv_max_ctx = 0;

    // Codec (lazy)
    std::string codec_path;
    // TODO: codec context pointer once codec runtime is implemented

    uint64_t rng_state = 0;
};

// ──────────────────────── metadata loading ─────────────────────────────

static void load_metadata(tada_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.n_layers    = core_gguf::kv_u32(g, "tada.talker.n_layers",    hp.n_layers);
    hp.d_model     = core_gguf::kv_u32(g, "tada.talker.d_model",     hp.d_model);
    hp.n_heads     = core_gguf::kv_u32(g, "tada.talker.n_heads",     hp.n_heads);
    hp.n_kv_heads  = core_gguf::kv_u32(g, "tada.talker.n_kv_heads",  hp.n_kv_heads);
    hp.head_dim    = core_gguf::kv_u32(g, "tada.talker.head_dim",    hp.head_dim);
    hp.ff_dim      = core_gguf::kv_u32(g, "tada.talker.ff_dim",      hp.ff_dim);
    hp.vocab_size  = core_gguf::kv_u32(g, "tada.talker.vocab_size",  hp.vocab_size);
    hp.max_pos     = core_gguf::kv_u32(g, "tada.talker.max_pos",     hp.max_pos);
    hp.rope_theta  = core_gguf::kv_f32(g, "tada.talker.rope_theta",  hp.rope_theta);
    hp.rms_norm_eps = core_gguf::kv_f32(g, "tada.talker.rms_norm_eps", hp.rms_norm_eps);

    hp.acoustic_dim     = core_gguf::kv_u32(g, "tada.acoustic_dim",     hp.acoustic_dim);
    hp.num_time_classes = core_gguf::kv_u32(g, "tada.num_time_classes",  hp.num_time_classes);
    hp.num_time_bits    = core_gguf::kv_u32(g, "tada.num_time_bits",     hp.num_time_bits);
    hp.time_dim         = core_gguf::kv_u32(g, "tada.time_dim",          hp.time_dim);
    hp.shift_acoustic   = core_gguf::kv_u32(g, "tada.shift_acoustic",    hp.shift_acoustic);
    hp.head_layers      = core_gguf::kv_u32(g, "tada.head_layers",       hp.head_layers);
    hp.head_ffn_ratio   = core_gguf::kv_f32(g, "tada.head_ffn_ratio",    hp.head_ffn_ratio);
    hp.fm_hidden        = core_gguf::kv_u32(g, "tada.fm_hidden",         hp.fm_hidden);
    hp.fm_latent        = core_gguf::kv_u32(g, "tada.fm_latent",         hp.fm_latent);
    hp.acoustic_mean    = core_gguf::kv_f32(g, "tada.acoustic_mean",     hp.acoustic_mean);
    hp.acoustic_std     = core_gguf::kv_f32(g, "tada.acoustic_std",      hp.acoustic_std);

    uint32_t bn = core_gguf::kv_u32(g, "tada.bottleneck_dim", 0);
    hp.has_bottleneck = (bn > 0 && bn != hp.d_model);
}

static void load_vocab(tada_context* c, gguf_context* g) {
    c->vocab.id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    for (size_t i = 0; i < c->vocab.id_to_token.size(); i++) {
        c->vocab.token_to_id[c->vocab.id_to_token[i]] = (int32_t)i;
    }
    // BPE merges
    auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
    for (int i = 0; i < (int)merges.size(); i++) {
        c->vocab.merge_rank[merges[i]] = i;
    }
}

// ──────────────────────── tensor binding ───────────────────────────────

static bool bind_talker(tada_context* c) {
    auto& t = c->talker;
    auto& m = c->tensors;
    const auto& hp = c->hp;

    t.token_embd_w  = core_gguf::require(m, "talker.token_embd.weight", "tada");
    t.output_norm_w = core_gguf::require(m, "talker.output_norm.weight", "tada");
    t.output_w      = core_gguf::try_get(m, "talker.output.weight");
    if (!t.output_w) t.output_w = t.token_embd_w;  // tied embeddings

    // TADA-specific
    t.acoustic_proj_w      = core_gguf::require(m, "tada.acoustic_proj.weight", "tada");
    t.acoustic_proj_b      = core_gguf::try_get(m, "tada.acoustic_proj.bias");
    t.time_start_embd_w    = core_gguf::require(m, "tada.time_start_embd.weight", "tada");
    t.time_end_embd_w      = core_gguf::require(m, "tada.time_end_embd.weight", "tada");
    t.acoustic_mask_embd_w = core_gguf::require(m, "tada.acoustic_mask_embd.weight", "tada");
    t.bottleneck_proj_w    = core_gguf::try_get(m, "tada.bottleneck_proj.weight");

    t.blocks.resize(hp.n_layers);
    char key[256];
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        auto& b = t.blocks[i];
#define BIND(fld, suffix)                                                  \
    snprintf(key, sizeof(key), "talker.blk.%u." suffix ".weight", i);      \
    b.fld = core_gguf::require(m, key, "tada");
        BIND(attn_norm_w,    "attn_norm")
        BIND(attn_q_w,       "attn_q")
        BIND(attn_k_w,       "attn_k")
        BIND(attn_v_w,       "attn_v")
        BIND(attn_output_w,  "attn_output")
        BIND(ffn_norm_w,     "ffn_norm")
        BIND(ffn_gate_w,     "ffn_gate")
        BIND(ffn_up_w,       "ffn_up")
        BIND(ffn_down_w,     "ffn_down")
#undef BIND
    }
    return true;
}

static bool bind_fm_head(tada_context* c) {
    auto& fm = c->fm;
    auto& m = c->tensors;
    const auto& hp = c->hp;

    fm.noisy_proj_w  = core_gguf::require(m, "tada.fm_head.noisy_proj.weight", "tada");
    fm.cond_proj_w   = core_gguf::require(m, "tada.fm_head.cond_proj.weight", "tada");
    fm.t_emb_mlp0_w  = core_gguf::require(m, "tada.fm_head.t_emb_mlp0.weight", "tada");
    fm.t_emb_mlp1_w  = core_gguf::require(m, "tada.fm_head.t_emb_mlp1.weight", "tada");
    fm.final_norm_w  = core_gguf::try_get(m, "tada.fm_head.final_norm.weight");
    fm.final_proj_w  = core_gguf::require(m, "tada.fm_head.final_proj.weight", "tada");
    fm.final_adaln_w = core_gguf::require(m, "tada.fm_head.final_adaln.weight", "tada");

    fm.layers.resize(hp.head_layers);
    char key[256];
    for (uint32_t i = 0; i < hp.head_layers; i++) {
        auto& l = fm.layers[i];
#define BIND_FM(fld, suffix)                                                \
    snprintf(key, sizeof(key), "tada.fm_head.blk.%u." suffix ".weight", i); \
    l.fld = core_gguf::require(m, key, "tada");
        BIND_FM(ffn_gate_w, "ffn_gate")
        BIND_FM(ffn_up_w,   "ffn_up")
        BIND_FM(ffn_down_w, "ffn_down")
        BIND_FM(norm_w,     "norm")
        BIND_FM(adaln_w,    "adaln")
#undef BIND_FM
    }
    return true;
}

// ──────────────────────── KV cache ────────────────────────────────────

static bool kv_init(tada_context* c, int max_ctx) {
    const auto& hp = c->hp;
    const int nl = (int)hp.n_layers;
    const int hd = (int)hp.head_dim;
    const int nkv = (int)hp.n_kv_heads;
    const size_t kv_size = (size_t)nl * hd * nkv * max_ctx;

    ggml_type kv_type = GGML_TYPE_F16;
    ggml_init_params ip = {
        ggml_tensor_overhead() * 2,
        nullptr,
        true
    };
    c->kv_ctx = ggml_init(ip);
    c->kv_k = ggml_new_tensor_1d(c->kv_ctx, kv_type, kv_size);
    c->kv_v = ggml_new_tensor_1d(c->kv_ctx, kv_type, kv_size);
    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf) {
        fprintf(stderr, "tada: failed to allocate KV cache (%d ctx)\n", max_ctx);
        return false;
    }
    c->kv_max_ctx = max_ctx;
    // Zero-init KV cache
    ggml_backend_buffer_clear(c->kv_buf, 0);
    return true;
}

// ──────────────────────── graph builders ──────────────────────────────

// Embed token IDs → d_model float vectors.
static ggml_cgraph* build_graph_embed(tada_context* c, int n_tokens) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, c->talker.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// Build the input embedding for one step:
//   emb = token_embd(id) + acoustic_proj(acoustic) + acoustic_mask_embd(mask)
//         + time_start_embd(t_before) + time_end_embd(t_after)
static ggml_cgraph* build_graph_step_embed(tada_context* c) {
    const int d = (int)c->hp.d_model;
    const int ad = (int)c->hp.acoustic_dim;
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 128, false);

    // Inputs
    ggml_tensor* token_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(token_id, "token_id"); ggml_set_input(token_id);

    ggml_tensor* acoustic = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ad);
    ggml_set_name(acoustic, "acoustic"); ggml_set_input(acoustic);

    ggml_tensor* mask_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(mask_id, "mask_id"); ggml_set_input(mask_id);

    ggml_tensor* t_before = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(t_before, "t_before"); ggml_set_input(t_before);

    ggml_tensor* t_after = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(t_after, "t_after"); ggml_set_input(t_after);

    // token_embd(id)
    ggml_tensor* tok_emb = ggml_get_rows(ctx0, c->talker.token_embd_w, token_id);

    // acoustic_proj(acoustic)
    ggml_tensor* ac_emb = ggml_mul_mat(ctx0, c->talker.acoustic_proj_w, acoustic);
    ac_emb = ggml_reshape_2d(ctx0, ac_emb, d, 1);
    if (c->talker.acoustic_proj_b) {
        ac_emb = ggml_add(ctx0, ac_emb, c->talker.acoustic_proj_b);
    }

    // acoustic_mask_embd(mask)
    ggml_tensor* mask_emb = ggml_get_rows(ctx0, c->talker.acoustic_mask_embd_w, mask_id);

    // time embeddings
    ggml_tensor* ts_emb = ggml_get_rows(ctx0, c->talker.time_start_embd_w, t_before);
    ggml_tensor* te_emb = ggml_get_rows(ctx0, c->talker.time_end_embd_w, t_after);

    // Sum all
    ggml_tensor* out = ggml_add(ctx0, tok_emb, ac_emb);
    out = ggml_add(ctx0, out, mask_emb);
    out = ggml_add(ctx0, out, ts_emb);
    out = ggml_add(ctx0, out, te_emb);
    ggml_set_name(out, "step_embed");
    ggml_build_forward_expand(gf, out);

    ggml_free(ctx0);
    return gf;
}

// Llama forward pass (same as orpheus) with KV cache.
// Input: embeds (d, T), output: hidden_state (d, 1) at last position.
static ggml_cgraph* build_graph_talker_kv(tada_context* c, int n_past, int n_tokens,
                                           bool compute_logits) {
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(c->kv_k && c->kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds"); ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions"); ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask"); ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        n_q, n_kv, hd, n_kv_grp,
        (int)hp.max_pos, theta,
        0.0f, 0.0f, attn_scale, 0.0f,
        core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->talker.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
            nullptr, nullptr, positions,
            (T == 1) ? nullptr : causal_mask,
            c->kv_k, c->kv_v, (int)il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->talker.output_norm_w);

    // Take last position
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // Output hidden state (always needed for FM head)
    ggml_tensor* hidden = ggml_cont(ctx0, cur);
    ggml_set_name(hidden, "hidden_state");
    ggml_build_forward_expand(gf, hidden);

    // Optionally compute logits
    if (compute_logits) {
        ggml_tensor* logits = ggml_mul_mat(ctx0, c->talker.output_w, hidden);
        ggml_set_name(logits, "logits");
        ggml_build_forward_expand(gf, logits);
    }

    ggml_free(ctx0);
    return gf;
}

// VibeVoiceDiffusionHead forward pass.
// Inputs: noisy_z (latent_dim,), timestep (scalar float), condition (hidden_dim,)
// Output: predicted velocity (latent_dim,)
static ggml_cgraph* build_graph_fm_step(tada_context* c) {
    const auto& hp = c->hp;
    const int hid = (int)hp.fm_hidden;
    const int lat = (int)hp.fm_latent;
    (void)hp; // fm_ffn_dim used implicitly via layer weights
    const float eps = hp.rms_norm_eps;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Inputs
    ggml_tensor* noisy_z = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, lat);
    ggml_set_name(noisy_z, "noisy_z"); ggml_set_input(noisy_z);

    ggml_tensor* t_emb_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_emb_sin, "t_emb_sin"); ggml_set_input(t_emb_sin);

    ggml_tensor* cond = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hid);
    ggml_set_name(cond, "fm_cond"); ggml_set_input(cond);

    // x = noisy_images_proj(noisy_z)
    ggml_tensor* x = ggml_mul_mat(ctx0, c->fm.noisy_proj_w, noisy_z);

    // t = t_embedder(timestep) = MLP(sinusoidal_embedding)
    // t_emb_sin already contains the sinusoidal embedding
    ggml_tensor* t = ggml_mul_mat(ctx0, c->fm.t_emb_mlp0_w, t_emb_sin);
    t = ggml_silu(ctx0, t);
    t = ggml_mul_mat(ctx0, c->fm.t_emb_mlp1_w, t);

    // condition = cond_proj(condition)
    ggml_tensor* cond_proj = ggml_mul_mat(ctx0, c->fm.cond_proj_w, cond);

    // c = condition + t
    ggml_tensor* c_emb = ggml_add(ctx0, cond_proj, t);

    // Head layers
    for (uint32_t i = 0; i < hp.head_layers; i++) {
        const auto& l = c->fm.layers[i];

        // adaLN_modulation: silu(c) → Linear → chunk into (shift, scale, gate)
        ggml_tensor* mod = ggml_silu(ctx0, c_emb);
        mod = ggml_mul_mat(ctx0, l.adaln_w, mod);
        // chunk into 3 parts of size hid
        ggml_tensor* shift = ggml_view_1d(ctx0, mod, hid, 0);
        ggml_tensor* scale = ggml_view_1d(ctx0, mod, hid, (size_t)hid * sizeof(float));
        ggml_tensor* gate  = ggml_view_1d(ctx0, mod, hid, (size_t)2 * hid * sizeof(float));

        // x = x + gate * ffn(modulate(norm(x), shift, scale))
        ggml_tensor* h = ggml_rms_norm(ctx0, x, eps);
        h = ggml_mul(ctx0, h, l.norm_w);
        // modulate: h * (1 + scale) + shift
        ggml_tensor* ones = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hid);
        ggml_set_name(ones, "ones");
        // Actually, scale is additive to 1.0, so: h * (1 + scale) + shift
        // = h + h * scale + shift
        h = ggml_add(ctx0, ggml_add(ctx0, h, ggml_mul(ctx0, h, scale)), shift);

        // SwiGLU FFN
        ggml_tensor* ffn_out = core_ffn::swiglu(ctx0, h, l.ffn_gate_w, l.ffn_up_w, l.ffn_down_w);

        // gated residual
        x = ggml_add(ctx0, x, ggml_mul(ctx0, gate, ffn_out));
    }

    // Final layer: adaLN (2-way) → linear
    {
        ggml_tensor* mod = ggml_silu(ctx0, c_emb);
        mod = ggml_mul_mat(ctx0, c->fm.final_adaln_w, mod);
        ggml_tensor* shift = ggml_view_1d(ctx0, mod, hid, 0);
        ggml_tensor* scale = ggml_view_1d(ctx0, mod, hid, (size_t)hid * sizeof(float));

        // norm_final (no affine weights in Python — RMSNorm(elementwise_affine=False))
        ggml_tensor* h = ggml_rms_norm(ctx0, x, eps);
        if (c->fm.final_norm_w) {
            h = ggml_mul(ctx0, h, c->fm.final_norm_w);
        }
        h = ggml_add(ctx0, ggml_add(ctx0, h, ggml_mul(ctx0, h, scale)), shift);
        h = ggml_mul_mat(ctx0, c->fm.final_proj_w, h);
        ggml_set_name(h, "velocity");
        ggml_build_forward_expand(gf, h);
    }

    ggml_free(ctx0);
    return gf;
}

// ──────────────────────── runtime helpers ─────────────────────────────

// Compute sinusoidal timestep embedding (matches Python TimestepEmbedder).
static void sinusoidal_embedding(float t, int dim, float* out) {
    const int half = dim / 2;
    const float log_max = std::log(10000.0f);
    for (int i = 0; i < half; i++) {
        float freq = std::exp(-log_max * (float)i / (float)half);
        float angle = t * freq;
        out[i] = std::cos(angle);
        out[half + i] = std::sin(angle);
    }
    if (dim % 2) out[dim - 1] = 0.0f;
}

// Decode gray code bits → integer time value.
// bits: float array of length num_bits, values ∈ {-1, 1}
static int decode_gray_code(const float* bits, int num_bits) {
    // Step 1: convert {-1, 1} → {0, 1} → gray code integer
    int gray = 0;
    for (int i = 0; i < num_bits; i++) {
        int bit = (bits[i] > 0.0f) ? 1 : 0;
        gray |= (bit << (num_bits - 1 - i));
    }
    // Step 2: gray code → binary integer
    int binary = gray;
    for (int shift = 1; shift < 32; shift <<= 1) {
        binary ^= (binary >> shift);
    }
    return binary;
}

// xorshift64* RNG
static uint64_t rng_next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_normal(uint64_t* state) {
    // Box-Muller
    double u1 = (double)(rng_next(state) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(rng_next(state) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-12) u1 = 1e-12;
    return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2));
}

// Embed tokens → float array (d_model * n_tokens)
static float* embed_tokens(tada_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed(c, n);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), ids, 0,
                            (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n * sizeof(float));
    return r;
}

// Build step embedding: token_embd + acoustic_proj + mask_embd + time_embds
static float* build_step_embedding(tada_context* c, int32_t token_id,
                                    const float* acoustic, int32_t mask_id,
                                    int32_t t_before, int32_t t_after) {
    const int d = (int)c->hp.d_model;
    const int ad = (int)c->hp.acoustic_dim;

    ggml_cgraph* gf = build_graph_step_embed(c);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_id"), &token_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "acoustic"), acoustic, 0, (size_t)ad * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask_id"), &mask_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_before"), &t_before, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_after"), &t_after, 0, sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "step_embed");
    float* r = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * sizeof(float));
    return r;
}

// Run LLM forward pass. Returns hidden_state (d_model,) and optionally logits.
// Both are malloc'd. Caller frees.
struct talker_result {
    float* hidden;  // (d_model,)
    float* logits;  // (vocab,) or nullptr
};

static talker_result run_talker_kv(tada_context* c, const float* embeds,
                                    int n_tokens, int n_past, bool need_logits) {
    talker_result res = {nullptr, nullptr};
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "tada: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return res;
    }
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int vocab = (int)hp.vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
    }

    ggml_cgraph* gf = build_graph_talker_kv(c, n_past, n_tokens, need_logits);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) return res;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) return res;

    ggml_tensor* h = ggml_graph_get_tensor(gf, "hidden_state");
    res.hidden = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(h, res.hidden, 0, (size_t)d * sizeof(float));

    if (need_logits) {
        ggml_tensor* l = ggml_graph_get_tensor(gf, "logits");
        res.logits = (float*)malloc((size_t)vocab * sizeof(float));
        ggml_backend_tensor_get(l, res.logits, 0, (size_t)vocab * sizeof(float));
    }
    return res;
}

// Run one FM step (velocity prediction).
static void run_fm_step(tada_context* c, const float* noisy_z, float timestep,
                         const float* cond, float* velocity_out) {
    const int lat = (int)c->hp.fm_latent;
    const int hid = (int)c->hp.fm_hidden;

    // Prepare sinusoidal embedding of timestep
    float t_emb[256];
    sinusoidal_embedding(timestep, 256, t_emb);

    // Optionally apply bottleneck to condition
    std::vector<float> cond_bn;
    const float* cond_input = cond;
    if (c->hp.has_bottleneck && c->talker.bottleneck_proj_w) {
        // TODO: build bottleneck graph
        // For now, pass cond directly (works when bottleneck_dim == d_model)
        cond_input = cond;
    }

    ggml_cgraph* gf = build_graph_fm_step(c);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "tada: failed to alloc fm_step graph\n");
        return;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0,
                            (size_t)lat * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0,
                            256 * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond"), cond_input, 0,
                            (size_t)hid * sizeof(float));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tada: fm_step compute failed\n");
        return;
    }

    ggml_tensor* vel = ggml_graph_get_tensor(gf, "velocity");
    ggml_backend_tensor_get(vel, velocity_out, 0, (size_t)lat * sizeof(float));
}

// Euler ODE solver for flow matching.
static void fm_euler_solve(tada_context* c, float* speech, const float* cond,
                            int num_steps, float /*cfg_scale*/) {
    const int lat = (int)c->hp.fm_latent;

    // Uniform time schedule [0, 1]
    std::vector<float> t_span(num_steps + 1);
    for (int i = 0; i <= num_steps; i++) {
        t_span[i] = (float)i / (float)num_steps;
    }

    std::vector<float> velocity(lat);
    for (int i = 0; i < num_steps; i++) {
        float dt = t_span[i + 1] - t_span[i];

        // No CFG for now (cfg_scale == 1.0)
        run_fm_step(c, speech, t_span[i], cond, velocity.data());

        // Euler step: speech += dt * velocity
        for (int j = 0; j < lat; j++) {
            speech[j] += dt * velocity[j];
        }
    }
}

// Simple argmax
static int argmax_logits(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

// BPE tokenize text using Llama tokenizer.
static std::vector<int32_t> tokenize(tada_context* c, const std::string& text) {
    return core_bpe::tokenize_simple(c->vocab.token_to_id, c->vocab.merge_rank, text);
}

// ──────────────────────── public API ─────────────────────────────────

extern "C" {

struct tada_context_params tada_context_default_params(void) {
    tada_context_params p = {};
    p.n_threads    = 4;
    p.verbosity    = 1;
    p.use_gpu      = false;
    p.temperature  = 0.0f;
    p.seed         = 42;
    p.max_tokens   = 0;
    p.flash_attn   = false;
    p.num_fm_steps = 0;
    p.acoustic_cfg = 1.0f;
    p.noise_temp   = 0.0f;
    return p;
}

struct tada_context* tada_init_from_file(const char* path_model,
                                          struct tada_context_params params) {
    auto* c = new tada_context();
    c->params = params;
    c->rng_state = params.seed ? params.seed : 42;
    c->compute_meta.resize(16 * 1024 * 1024);

    // ── Pass 1: metadata ──
    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) { delete c; return nullptr; }
    load_metadata(c, meta);
    load_vocab(c, meta);
    core_gguf::free_metadata(meta);

    if (params.verbosity >= 1) {
        const auto& hp = c->hp;
        fprintf(stderr, "tada: %uL %ud %u/%u heads, ff=%u, vocab=%u\n",
                hp.n_layers, hp.d_model, hp.n_heads, hp.n_kv_heads,
                hp.ff_dim, hp.vocab_size);
        fprintf(stderr, "tada: acoustic=%u, time=%u, fm_hidden=%u, fm_latent=%u\n",
                hp.acoustic_dim, hp.time_dim, hp.fm_hidden, hp.fm_latent);
    }

    // ── Backend init ──
    c->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(c->backend, params.n_threads);
    c->backend_cpu = c->backend;

    // ── Pass 2: weights ──
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "tada", wl)) {
        fprintf(stderr, "tada: failed to load weights from %s\n", path_model);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    // ── Bind tensors ──
    if (!bind_talker(c) || !bind_fm_head(c)) {
        fprintf(stderr, "tada: failed to bind tensors\n");
        delete c;
        return nullptr;
    }

    // ── Scheduler ──
    ggml_backend_t backends[] = {c->backend};
    c->sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);

    // ── KV cache ──
    int max_ctx = params.max_tokens > 0 ? params.max_tokens + 256 : 1024;
    if (!kv_init(c, max_ctx)) {
        delete c;
        return nullptr;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "tada: loaded OK, KV cache for %d tokens\n", max_ctx);
    }
    return c;
}

int tada_set_codec_path(struct tada_context* ctx, const char* path) {
    if (!ctx || !path) return -1;
    ctx->codec_path = path;
    // TODO: load codec model
    return 0;
}

void tada_set_seed(struct tada_context* ctx, uint64_t seed) {
    if (ctx) ctx->rng_state = seed ? seed : 42;
}

void tada_set_temperature(struct tada_context* ctx, float temp) {
    if (ctx) ctx->params.temperature = temp;
}

float* tada_synthesize(struct tada_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples) return nullptr;
    *out_n_samples = 0;

    const auto& hp = ctx->hp;
    const int ad = (int)hp.acoustic_dim;
    const int lat = (int)hp.fm_latent;
    const int shift = (int)hp.shift_acoustic;
    const int num_fm_steps = ctx->params.num_fm_steps > 0 ? ctx->params.num_fm_steps : 10;
    const float noise_temp = ctx->params.noise_temp;
    const float cfg_scale = ctx->params.acoustic_cfg;
    const int max_tokens = ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 512;

    // Reset RNG
    ctx->rng_state = ctx->params.seed ? ctx->params.seed : 42;

    // ── Tokenize ──
    std::vector<int32_t> text_ids = tokenize(ctx, std::string(text));
    if (text_ids.empty()) {
        fprintf(stderr, "tada: empty tokenization\n");
        return nullptr;
    }

    // Build full input: BOS + prefix + text + EOS*shift
    int32_t bos = 128000;  // Llama-3 BOS
    int32_t eot = 128009;  // <|eot_id|>
    auto it = ctx->vocab.token_to_id.find("<|begin_of_text|>");
    if (it != ctx->vocab.token_to_id.end()) bos = it->second;
    it = ctx->vocab.token_to_id.find("<|eot_id|>");
    if (it != ctx->vocab.token_to_id.end()) eot = it->second;

    // Prefix: <|start_header_id|>assistant<|end_header_id|>
    std::string prefix = "<|start_header_id|>assistant<|end_header_id|>";
    std::vector<int32_t> prefix_ids = tokenize(ctx, prefix);

    std::vector<int32_t> full_ids;
    full_ids.push_back(bos);
    full_ids.insert(full_ids.end(), prefix_ids.begin(), prefix_ids.end());
    full_ids.insert(full_ids.end(), text_ids.begin(), text_ids.end());
    for (int i = 0; i < shift; i++) full_ids.push_back(eot);

    int num_prompt = (int)full_ids.size();

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: %d prompt tokens, max %d generation tokens\n",
                num_prompt, max_tokens);
    }

    // ── Zero KV cache ──
    ggml_backend_buffer_clear(ctx->kv_buf, 0);

    // ── AR + FM generation loop ──
    std::vector<std::vector<float>> acoustic_features;
    std::vector<int> time_before_list;

    // State
    std::vector<float> cur_acoustic(ad, 0.0f);
    int32_t cur_mask = 0;
    int32_t cur_t_before = 0;
    int32_t cur_t_after = 0;
    int n_past = 0;

    // Extend full_ids as we generate
    std::vector<int32_t> all_ids = full_ids;

    for (int step = 0; step < num_prompt + max_tokens; step++) {
        if (step >= (int)all_ids.size()) break;

        int32_t cur_token = all_ids[step];
        bool need_logits = (step >= num_prompt - 1);

        // Build step embedding
        float* emb = build_step_embedding(ctx, cur_token, cur_acoustic.data(),
                                           cur_mask, cur_t_before, cur_t_after);
        if (!emb) { fprintf(stderr, "tada: embed failed at step %d\n", step); return nullptr; }

        // LLM forward
        talker_result tr = run_talker_kv(ctx, emb, 1, n_past, need_logits);
        free(emb);
        if (!tr.hidden) { fprintf(stderr, "tada: talker failed at step %d\n", step); return nullptr; }
        n_past++;

        // ── Flow matching solver ──
        std::vector<float> speech(lat);
        // Initialize noise
        for (int j = 0; j < lat; j++) {
            speech[j] = rng_normal(&ctx->rng_state) * noise_temp;
        }

        // Solve ODE
        fm_euler_solve(ctx, speech.data(), tr.hidden, num_fm_steps, cfg_scale);

        // Extract time from gray code
        int num_time_bits = (int)hp.num_time_bits;
        int pred_t_before = decode_gray_code(&speech[ad], num_time_bits);
        int pred_t_after = decode_gray_code(&speech[ad + num_time_bits], num_time_bits);

        // Next token prediction (greedy for now)
        if (step >= num_prompt - 1 && tr.logits) {
            int next = argmax_logits(tr.logits, (int)hp.vocab_size);
            if (next == eot) {
                if (ctx->params.verbosity >= 1) {
                    fprintf(stderr, "tada: EOS at step %d\n", step);
                }
                free(tr.hidden);
                free(tr.logits);
                break;
            }
            all_ids.push_back(next);
        }
        free(tr.logits);

        // Update state for next step
        if (step >= shift) {
            // Use FM output as acoustic features
            std::vector<float> feat(speech.begin(), speech.begin() + ad);
            acoustic_features.push_back(feat);
            time_before_list.push_back(pred_t_before);

            cur_acoustic = feat;
            cur_mask = 1;
        } else {
            std::fill(cur_acoustic.begin(), cur_acoustic.end(), 0.0f);
            cur_mask = 0;
        }
        cur_t_before = pred_t_before;
        cur_t_after = pred_t_after;

        free(tr.hidden);
    }

    if (acoustic_features.empty()) {
        fprintf(stderr, "tada: no acoustic features generated\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: %zu acoustic frames, %zu time values\n",
                acoustic_features.size(), time_before_list.size());
    }

    // ── Denormalize and expand ──
    // features * std + mean
    int total_frames = 0;
    for (size_t i = 0; i < acoustic_features.size(); i++) {
        int t = (i == 0) ? 0 : time_before_list[i - 1];
        total_frames += std::max(0, t - 1) + 1;
    }
    // Add trailing frames from last time value
    if (!time_before_list.empty()) {
        total_frames += time_before_list.back();
    }

    // TODO: Run codec decoder on expanded features
    // For now, return empty audio as placeholder
    fprintf(stderr, "tada: codec decoder not yet implemented. "
            "%zu features, %d expanded frames\n",
            acoustic_features.size(), total_frames);

    // Return silence placeholder
    int n_samples = total_frames * 480;  // 480 = upsample factor (4*4*5*6)
    if (n_samples <= 0) n_samples = 24000;  // 1 second of silence
    float* pcm = (float*)calloc(n_samples, sizeof(float));
    *out_n_samples = n_samples;
    return pcm;
}

void tada_pcm_free(float* pcm) {
    free(pcm);
}

void tada_free(struct tada_context* ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->buf_w) ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w) ggml_free(ctx->ctx_w);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

} // extern "C"
