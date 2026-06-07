// src/tada_tts.cpp — TADA-3B-ML TTS runtime.
//
// HumeAI/tada-3b-ml: Llama-3.2-3B + VibeVoiceDiffusionHead + TADA codec.
// See tada_tts.h for the C ABI.

#include "tada_tts.h"
#include "tada_codec.h"
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
    uint32_t n_layers = 28;
    uint32_t d_model = 3072;
    uint32_t n_heads = 24;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 8192;
    uint32_t vocab_size = 128256;
    uint32_t max_pos = 131072;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;

    // TADA-specific
    uint32_t acoustic_dim = 512;
    uint32_t num_time_classes = 256;
    uint32_t num_time_bits = 8;
    uint32_t time_dim = 16;
    uint32_t shift_acoustic = 5;
    uint32_t head_layers = 6;
    float head_ffn_ratio = 4.0f;
    uint32_t fm_hidden = 3072; // bottleneck_dim or d_model
    uint32_t fm_latent = 528;  // acoustic_dim + time_dim
    float acoustic_mean = 0.0f;
    float acoustic_std = 1.5f;
    bool has_bottleneck = false;

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
    ggml_tensor* output_w; // lm_head

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
    ggml_tensor* adaln_w; // SiLU → Linear(cond_dim, 3*embed_dim)
};

struct tada_fm_head {
    ggml_tensor* noisy_proj_w; // (latent, hidden)
    ggml_tensor* cond_proj_w;  // (hidden, cond_dim)
    ggml_tensor* t_emb_mlp0_w; // (freq_dim, hidden)
    ggml_tensor* t_emb_mlp1_w; // (hidden, hidden)
    std::vector<tada_fm_layer> layers;
    ggml_tensor* final_norm_w;  // RMSNorm (no affine — weight may still exist as ones)
    ggml_tensor* final_proj_w;  // (hidden, latent)
    ggml_tensor* final_adaln_w; // (cond_dim, 2*hidden)
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

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache (positive path)
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // KV cache (negative path for CFG)
    ggml_context* kv_neg_ctx = nullptr;
    ggml_backend_buffer_t kv_neg_buf = nullptr;
    ggml_tensor* kv_neg_k = nullptr;
    ggml_tensor* kv_neg_v = nullptr;

    // Codec (lazy-loaded)
    std::string codec_path;
    tada_codec_context* codec_ctx = nullptr;

    // Pre-computed voice prompt (from GGUF)
    std::vector<float> prompt_values;        // (n_prompt, acoustic_dim) flat
    std::vector<int32_t> prompt_masks;       // (n_prompt,) all 1s
    std::vector<int32_t> prompt_time_before; // (n_prompt,) time gaps
    std::vector<int32_t> prompt_time_after;  // (n_prompt,) time gaps
    int n_prompt = 0;

    uint64_t rng_state = 0;
};

// ──────────────────────── metadata loading ─────────────────────────────

static void load_metadata(tada_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.n_layers = core_gguf::kv_u32(g, "tada.talker.n_layers", hp.n_layers);
    hp.d_model = core_gguf::kv_u32(g, "tada.talker.d_model", hp.d_model);
    hp.n_heads = core_gguf::kv_u32(g, "tada.talker.n_heads", hp.n_heads);
    hp.n_kv_heads = core_gguf::kv_u32(g, "tada.talker.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = core_gguf::kv_u32(g, "tada.talker.head_dim", hp.head_dim);
    hp.ff_dim = core_gguf::kv_u32(g, "tada.talker.ff_dim", hp.ff_dim);
    hp.vocab_size = core_gguf::kv_u32(g, "tada.talker.vocab_size", hp.vocab_size);
    hp.max_pos = core_gguf::kv_u32(g, "tada.talker.max_pos", hp.max_pos);
    hp.rope_theta = core_gguf::kv_f32(g, "tada.talker.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = core_gguf::kv_f32(g, "tada.talker.rms_norm_eps", hp.rms_norm_eps);

    hp.acoustic_dim = core_gguf::kv_u32(g, "tada.acoustic_dim", hp.acoustic_dim);
    hp.num_time_classes = core_gguf::kv_u32(g, "tada.num_time_classes", hp.num_time_classes);
    hp.num_time_bits = core_gguf::kv_u32(g, "tada.num_time_bits", hp.num_time_bits);
    hp.time_dim = core_gguf::kv_u32(g, "tada.time_dim", hp.time_dim);
    hp.shift_acoustic = core_gguf::kv_u32(g, "tada.shift_acoustic", hp.shift_acoustic);
    hp.head_layers = core_gguf::kv_u32(g, "tada.head_layers", hp.head_layers);
    hp.head_ffn_ratio = core_gguf::kv_f32(g, "tada.head_ffn_ratio", hp.head_ffn_ratio);
    hp.fm_hidden = core_gguf::kv_u32(g, "tada.fm_hidden", hp.fm_hidden);
    hp.fm_latent = core_gguf::kv_u32(g, "tada.fm_latent", hp.fm_latent);
    hp.acoustic_mean = core_gguf::kv_f32(g, "tada.acoustic_mean", hp.acoustic_mean);
    hp.acoustic_std = core_gguf::kv_f32(g, "tada.acoustic_std", hp.acoustic_std);

    uint32_t bn = core_gguf::kv_u32(g, "tada.bottleneck_dim", 0);
    hp.has_bottleneck = (bn > 0 && bn != hp.d_model);
}

static void load_vocab(tada_context* c, gguf_context* g) {
    c->vocab.id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");

    // Llama-3 tiktoken vocab: 128000 BPE tokens (0='!', 1='"', ...) plus
    // 256 special tokens (128000='<|begin_of_text|>', ...).  Token array
    // indices map 1:1 to embedding weight rows — no offset needed.
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

    t.token_embd_w = core_gguf::require(m, "talker.token_embd.weight", "tada");
    t.output_norm_w = core_gguf::require(m, "talker.output_norm.weight", "tada");
    t.output_w = core_gguf::try_get(m, "talker.output.weight");
    if (!t.output_w)
        t.output_w = t.token_embd_w; // tied embeddings

    // TADA-specific
    t.acoustic_proj_w = core_gguf::require(m, "tada.acoustic_proj.weight", "tada");
    t.acoustic_proj_b = core_gguf::try_get(m, "tada.acoustic_proj.bias");
    t.time_start_embd_w = core_gguf::require(m, "tada.time_start_embd.weight", "tada");
    t.time_end_embd_w = core_gguf::require(m, "tada.time_end_embd.weight", "tada");
    t.acoustic_mask_embd_w = core_gguf::require(m, "tada.acoustic_mask_embd.weight", "tada");
    t.bottleneck_proj_w = core_gguf::try_get(m, "tada.bottleneck_proj.weight");

    t.blocks.resize(hp.n_layers);
    char key[256];
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        auto& b = t.blocks[i];
#define BIND(fld, suffix)                                                                                              \
    snprintf(key, sizeof(key), "talker.blk.%u." suffix ".weight", i);                                                  \
    b.fld = core_gguf::require(m, key, "tada");
        BIND(attn_norm_w, "attn_norm")
        BIND(attn_q_w, "attn_q")
        BIND(attn_k_w, "attn_k")
        BIND(attn_v_w, "attn_v")
        BIND(attn_output_w, "attn_output")
        BIND(ffn_norm_w, "ffn_norm")
        BIND(ffn_gate_w, "ffn_gate")
        BIND(ffn_up_w, "ffn_up")
        BIND(ffn_down_w, "ffn_down")
#undef BIND
    }
    return true;
}

static bool bind_fm_head(tada_context* c) {
    auto& fm = c->fm;
    auto& m = c->tensors;
    const auto& hp = c->hp;

    fm.noisy_proj_w = core_gguf::require(m, "tada.fm_head.noisy_proj.weight", "tada");
    fm.cond_proj_w = core_gguf::require(m, "tada.fm_head.cond_proj.weight", "tada");
    fm.t_emb_mlp0_w = core_gguf::require(m, "tada.fm_head.t_emb_mlp0.weight", "tada");
    fm.t_emb_mlp1_w = core_gguf::require(m, "tada.fm_head.t_emb_mlp1.weight", "tada");
    fm.final_norm_w = core_gguf::try_get(m, "tada.fm_head.final_norm.weight");
    fm.final_proj_w = core_gguf::require(m, "tada.fm_head.final_proj.weight", "tada");
    fm.final_adaln_w = core_gguf::require(m, "tada.fm_head.final_adaln.weight", "tada");

    fm.layers.resize(hp.head_layers);
    char key[256];
    for (uint32_t i = 0; i < hp.head_layers; i++) {
        auto& l = fm.layers[i];
#define BIND_FM(fld, suffix)                                                                                           \
    snprintf(key, sizeof(key), "tada.fm_head.blk.%u." suffix ".weight", i);                                            \
    l.fld = core_gguf::require(m, key, "tada");
        BIND_FM(ffn_gate_w, "ffn_gate")
        BIND_FM(ffn_up_w, "ffn_up")
        BIND_FM(ffn_down_w, "ffn_down")
        BIND_FM(norm_w, "norm")
        BIND_FM(adaln_w, "adaln")
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

    const auto kv_pair = core_attn::kv_dtype_pair_from_env("tada");
    ggml_init_params ip = {ggml_tensor_overhead() * 2, nullptr, true};
    c->kv_ctx = ggml_init(ip);
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, kv_pair.k, hd, max_ctx, nkv, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, kv_pair.v, hd, max_ctx, nkv, nl);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");
    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf) {
        fprintf(stderr, "tada: failed to allocate KV cache (%d ctx)\n", max_ctx);
        return false;
    }
    c->kv_max_ctx = max_ctx;

    // Negative KV cache (for CFG doubled-batch)
    ggml_init_params ip_neg = {ggml_tensor_overhead() * 2, nullptr, true};
    c->kv_neg_ctx = ggml_init(ip_neg);
    c->kv_neg_k = ggml_new_tensor_4d(c->kv_neg_ctx, kv_pair.k, hd, max_ctx, nkv, nl);
    c->kv_neg_v = ggml_new_tensor_4d(c->kv_neg_ctx, kv_pair.v, hd, max_ctx, nkv, nl);
    ggml_set_name(c->kv_neg_k, "kv_neg_k");
    ggml_set_name(c->kv_neg_v, "kv_neg_v");
    c->kv_neg_buf = ggml_backend_alloc_ctx_tensors(c->kv_neg_ctx, c->backend);

    // Zero-init both KV caches
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
    ggml_set_name(token_id, "token_id");
    ggml_set_input(token_id);

    ggml_tensor* acoustic = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, ad);
    ggml_set_name(acoustic, "acoustic");
    ggml_set_input(acoustic);

    ggml_tensor* mask_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(mask_id, "mask_id");
    ggml_set_input(mask_id);

    ggml_tensor* t_before = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(t_before, "t_before");
    ggml_set_input(t_before);

    ggml_tensor* t_after = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(t_after, "t_after");
    ggml_set_input(t_after);

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
static ggml_cgraph* build_graph_talker_kv(tada_context* c, int n_past, int n_tokens, bool compute_logits,
                                          ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
    // Default to positive KV cache
    if (!use_kv_k)
        use_kv_k = c->kv_k;
    if (!use_kv_v)
        use_kv_v = c->kv_v;
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

    GGML_ASSERT(use_kv_k && use_kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        n_q, n_kv, hd, n_kv_grp, (int)hp.max_pos, theta, 0.0f, 0.0f, attn_scale, 0.0f, core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->talker.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                    nullptr, nullptr, positions, (T == 1) ? nullptr : causal_mask,
                                                    use_kv_k, use_kv_v, (int)il, n_past, kvp);
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
    ggml_set_name(noisy_z, "noisy_z");
    ggml_set_input(noisy_z);

    ggml_tensor* t_emb_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_emb_sin, "t_emb_sin");
    ggml_set_input(t_emb_sin);

    ggml_tensor* cond = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hid);
    ggml_set_name(cond, "fm_cond");
    ggml_set_input(cond);

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
        ggml_tensor* gate = ggml_view_1d(ctx0, mod, hid, (size_t)2 * hid * sizeof(float));

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
    if (dim % 2)
        out[dim - 1] = 0.0f;
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
    if (u1 < 1e-12)
        u1 = 1e-12;
    return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2));
}

// Embed tokens → float array (d_model * n_tokens)
static float* embed_tokens(tada_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed(c, n);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n * sizeof(float));
    return r;
}

// Build step embedding: token_embd + acoustic_proj + mask_embd + time_embds
static float* build_step_embedding(tada_context* c, int32_t token_id, const float* acoustic, int32_t mask_id,
                                   int32_t t_before, int32_t t_after) {
    const int d = (int)c->hp.d_model;
    const int ad = (int)c->hp.acoustic_dim;

    ggml_cgraph* gf = build_graph_step_embed(c);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_id"), &token_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "acoustic"), acoustic, 0, (size_t)ad * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask_id"), &mask_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_before"), &t_before, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_after"), &t_after, 0, sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* out = ggml_graph_get_tensor(gf, "step_embed");
    float* r = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * sizeof(float));
    return r;
}

// Run LLM forward pass. Returns hidden_state (d_model,) and optionally logits.
// Both are malloc'd. Caller frees.
struct talker_result {
    float* hidden; // (d_model,)
    float* logits; // (vocab,) or nullptr
};

static talker_result run_talker_kv(tada_context* c, const float* embeds, int n_tokens, int n_past, bool need_logits,
                                   ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
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
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

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

    ggml_cgraph* gf = build_graph_talker_kv(c, n_past, n_tokens, need_logits, use_kv_k, use_kv_v);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return res;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return res;

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
static void run_fm_step(tada_context* c, const float* noisy_z, float timestep, const float* cond, float* velocity_out) {
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

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0, (size_t)lat * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond"), cond_input, 0, (size_t)hid * sizeof(float));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tada: fm_step compute failed\n");
        return;
    }

    ggml_tensor* vel = ggml_graph_get_tensor(gf, "velocity");
    ggml_backend_tensor_get(vel, velocity_out, 0, (size_t)lat * sizeof(float));
}

// Cosine CFG schedule: scale decays from base at t=0 to 1.0 at t=1.
static float scheduled_cfg(float base_scale, float t, const char* schedule) {
    if (base_scale == 1.0f)
        return 1.0f;
    if (strcmp(schedule, "cosine") == 0) {
        return 1.0f + (base_scale - 1.0f) * 0.5f * (1.0f + std::cos((float)M_PI * t));
    }
    if (strcmp(schedule, "linear") == 0) {
        return 1.0f + (base_scale - 1.0f) * (1.0f - t);
    }
    return base_scale; // constant
}

// LogSNR time schedule: uniform in log-SNR space, denser near t=0.
static void build_logsnr_schedule(std::vector<float>& t_span, int num_steps) {
    t_span.resize(num_steps + 1);
    for (int i = 0; i <= num_steps; i++) {
        float log_snr = 5.0f - 10.0f * (float)i / (float)num_steps; // [5, -5]
        t_span[i] = 1.0f / (1.0f + std::exp(log_snr / 2.0f));       // sigmoid(-log_snr/2)
    }
    t_span[0] = 0.0f;
    t_span[num_steps] = 1.0f;
}

// Euler ODE solver for flow matching with CFG.
static void fm_euler_solve(tada_context* c, float* speech, const float* cond, int num_steps, float cfg_scale,
                           const float* neg_cond = nullptr, bool dump_trajectory = false) {
    const int lat = (int)c->hp.fm_latent;
    const int ad = (int)c->hp.acoustic_dim;

    // LogSNR time schedule (matches Python time_schedule="logsnr")
    std::vector<float> t_span;
    build_logsnr_schedule(t_span, num_steps);

    std::vector<float> vel_pos(lat), vel_neg(lat);
    std::vector<float> zero_cond(c->hp.fm_hidden, 0.0f);
    const float* neg = neg_cond ? neg_cond : zero_cond.data();

    for (int i = 0; i < num_steps; i++) {
        float dt = t_span[i + 1] - t_span[i];
        float t_val = t_span[i];

        // Scheduled CFG (cosine decay from cfg_scale at t=0 to 1.0 at t=1)
        float a_cfg = scheduled_cfg(cfg_scale, t_val, "cosine");

        if (a_cfg != 1.0f) {
            // CFG: velocity = v_neg + cfg * (v_pos - v_neg)
            // Separate acoustic and duration CFG (duration_cfg = 1.0)
            run_fm_step(c, speech, t_val, cond, vel_pos.data());
            run_fm_step(c, speech, t_val, neg, vel_neg.data());

            for (int j = 0; j < ad; j++) {
                // Acoustic dims: apply acoustic CFG
                speech[j] += dt * (vel_neg[j] + a_cfg * (vel_pos[j] - vel_neg[j]));
            }
            for (int j = ad; j < lat; j++) {
                // Time dims: duration CFG = 1.0 (no guidance)
                speech[j] += dt * (vel_neg[j] + 1.0f * (vel_pos[j] - vel_neg[j]));
            }

            if (dump_trajectory) {
                float srms = 0, vp = 0, vn = 0;
                for (int j = 0; j < ad; j++) {
                    srms += speech[j] * speech[j];
                    vp += vel_pos[j] * vel_pos[j];
                    vn += vel_neg[j] * vel_neg[j];
                }
                fprintf(stderr,
                        "  euler[%d] t=%.3f dt=%.4f cfg=%.2f vp=%.3f vn=%.3f srms=%.4f s[0:3]=[%.3f,%.3f,%.3f]\n", i,
                        t_val, dt, a_cfg, std::sqrt(vp / ad), std::sqrt(vn / ad), std::sqrt(srms / ad), speech[0],
                        speech[1], speech[2]);
            }
        } else {
            run_fm_step(c, speech, t_val, cond, vel_pos.data());
            for (int j = 0; j < lat; j++) {
                speech[j] += dt * vel_pos[j];
            }
        }
    }
}

// Simple argmax
static int argmax_logits(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
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
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.0f;
    p.seed = 42;
    p.max_tokens = 0;
    p.flash_attn = false;
    p.num_fm_steps = 0;
    p.acoustic_cfg = 1.6f; // match Python InferenceOptions default
    p.noise_temp = 0.9f;   // match Python InferenceOptions default
    return p;
}

struct tada_context* tada_init_from_file(const char* path_model, struct tada_context_params params) {
    auto* c = new tada_context();
    c->params = params;
    c->rng_state = params.seed ? params.seed : 42;
    c->compute_meta.resize(16 * 1024 * 1024);

    // ── Pass 1: metadata ──
    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        delete c;
        return nullptr;
    }
    load_metadata(c, meta);
    load_vocab(c, meta);
    core_gguf::free_metadata(meta);

    if (params.verbosity >= 1) {
        const auto& hp = c->hp;
        fprintf(stderr, "tada: %uL %ud %u/%u heads, ff=%u, vocab=%u\n", hp.n_layers, hp.d_model, hp.n_heads,
                hp.n_kv_heads, hp.ff_dim, hp.vocab_size);
        fprintf(stderr, "tada: acoustic=%u, time=%u, fm_hidden=%u, fm_latent=%u\n", hp.acoustic_dim, hp.time_dim,
                hp.fm_hidden, hp.fm_latent);
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
    if (!ctx || !path)
        return -1;
    ctx->codec_path = path;
    ctx->codec_ctx = tada_codec_init_from_file(path, ctx->params.n_threads);
    if (!ctx->codec_ctx) {
        fprintf(stderr, "tada: failed to load codec from %s\n", path);
        return -1;
    }
    return 0;
}

int tada_load_prompt(struct tada_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    // Load the reference GGUF containing prompt_token_values and prompt_token_positions
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return -1;
    core_gguf::free_metadata(meta);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "tada-prompt", wl)) {
        fprintf(stderr, "tada: failed to load prompt from %s\n", path);
        return -1;
    }

    // Extract prompt_token_values: (n_prompt, acoustic_dim) = ggml ne=[acoustic_dim, n_prompt]
    ggml_tensor* tv = core_gguf::try_get(wl.tensors, "prompt_token_values");
    ggml_tensor* tp = core_gguf::try_get(wl.tensors, "prompt_token_positions");
    if (!tv) {
        fprintf(stderr, "tada: prompt_token_values not found in %s\n", path);
        return -1;
    }

    const int ad = (int)ctx->hp.acoustic_dim;
    const int np = (int)(ggml_nelements(tv) / ad);
    ctx->n_prompt = np;

    // Read token values
    ctx->prompt_values.resize(np * ad);
    ggml_backend_tensor_get(tv, ctx->prompt_values.data(), 0, (size_t)np * ad * sizeof(float));

    // Read positions and compute time gaps
    if (tp) {
        std::vector<float> pos(np);
        ggml_backend_tensor_get(tp, pos.data(), 0, (size_t)np * sizeof(float));

        // Time gaps: time_before[i] = positions[i] - positions[i-1], clamped to [0, num_time_classes-1]
        int max_t = (int)ctx->hp.num_time_classes - 1;
        ctx->prompt_time_before.resize(np + 1, 0);
        ctx->prompt_time_after.resize(np + 1, 0);
        for (int i = 0; i < np; i++) {
            int p_cur = (int)pos[i];
            int p_prev = (i > 0) ? (int)pos[i - 1] : 0;
            int gap = std::min(std::max(p_cur - p_prev, 0), max_t);
            ctx->prompt_time_before[i + 1] = gap; // shifted by 1 (index 0 is padding)
        }
        // time_after[i] = time_before[i+1]
        for (int i = 0; i < np; i++) {
            ctx->prompt_time_after[i] =
                (i + 1 < (int)ctx->prompt_time_before.size()) ? ctx->prompt_time_before[i + 1] : 1;
        }
    }

    // Masks: all 1 for prompt tokens
    ctx->prompt_masks.assign(np, 1);

    // Clean up
    if (wl.buf)
        ggml_backend_buffer_free(wl.buf);
    if (wl.ctx)
        ggml_free(wl.ctx);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: loaded prompt with %d tokens from %s\n", np, path);
    }
    return 0;
}

void tada_set_seed(struct tada_context* ctx, uint64_t seed) {
    if (ctx)
        ctx->rng_state = seed ? seed : 42;
}

void tada_set_temperature(struct tada_context* ctx, float temp) {
    if (ctx)
        ctx->params.temperature = temp;
}

float* tada_synthesize(struct tada_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
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

    // Build full input: BOS + prefix + prompt_text + synth_text + EOT*shift
    // Must use raw token IDs for special tokens (BPE tokenizer doesn't handle <|...|>).
    // Matches Python generate(): prefix = system_header + assistant_header
    auto lookup = [&](const char* name, int32_t fallback) -> int32_t {
        auto it = ctx->vocab.token_to_id.find(name);
        return (it != ctx->vocab.token_to_id.end()) ? it->second : fallback;
    };
    int32_t bos = lookup("<|begin_of_text|>", 128000);
    int32_t eot = lookup("<|eot_id|>", 128009);
    int32_t start_header = lookup("<|start_header_id|>", 128006);
    int32_t end_header = lookup("<|end_header_id|>", 128007);

    // Build prefix: <|start_header_id|>system<|end_header_id|><|eot_id|>
    //               <|start_header_id|>assistant<|end_header_id|>
    // This matches Python's generate() with default system_prompt=None.
    std::vector<int32_t> system_text_ids = tokenize(ctx, std::string("system"));
    std::vector<int32_t> assistant_text_ids = tokenize(ctx, std::string("assistant"));

    std::vector<int32_t> prefix_ids;
    prefix_ids.push_back(start_header);
    prefix_ids.insert(prefix_ids.end(), system_text_ids.begin(), system_text_ids.end());
    prefix_ids.push_back(end_header);
    // empty system prompt (system_prompt or '' in Python) — no text tokens here
    prefix_ids.push_back(eot);
    prefix_ids.push_back(start_header);
    prefix_ids.insert(prefix_ids.end(), assistant_text_ids.begin(), assistant_text_ids.end());
    prefix_ids.push_back(end_header);

    int prefix_len = (int)prefix_ids.size(); // does NOT include BOS

    // Prompt text tokens (transcript of reference audio for voice conditioning)
    std::vector<int32_t> prompt_text_ids;
    const char* prompt_text_env = getenv("TADA_PROMPT_TEXT");
    if (prompt_text_env && ctx->n_prompt > 0) {
        prompt_text_ids = tokenize(ctx, std::string(prompt_text_env));
    }

    // Full sequence: BOS + prefix + prompt_text + synth_text + EOT*shift
    std::vector<int32_t> full_ids;
    full_ids.push_back(bos);
    full_ids.insert(full_ids.end(), prefix_ids.begin(), prefix_ids.end());
    full_ids.insert(full_ids.end(), prompt_text_ids.begin(), prompt_text_ids.end());
    full_ids.insert(full_ids.end(), text_ids.begin(), text_ids.end());
    for (int i = 0; i < shift; i++)
        full_ids.push_back(eot);

    int num_prompt = (int)full_ids.size();

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: %d prompt tokens (prefix_len=%d), max %d generation tokens\n", num_prompt, prefix_len,
                max_tokens);
        fprintf(stderr, "tada: prefix_ids=[");
        for (size_t i = 0; i < prefix_ids.size(); i++)
            fprintf(stderr, "%d%s", prefix_ids[i], i + 1 < prefix_ids.size() ? "," : "");
        fprintf(stderr, "]\n");
        fprintf(stderr, "tada: full_ids(%d)=[", (int)full_ids.size());
        for (int i = 0; i < (int)full_ids.size(); i++)
            fprintf(stderr, "%d%s", full_ids[i], i + 1 < (int)full_ids.size() ? "," : "");
        fprintf(stderr, "]\n");
        fprintf(stderr, "tada: text_ids(%d)=[", (int)text_ids.size());
        for (int i = 0; i < (int)text_ids.size(); i++)
            fprintf(stderr, "%d%s", text_ids[i], i + 1 < (int)text_ids.size() ? "," : "");
        fprintf(stderr, "]\n");
    }

    // ── Zero KV caches ──
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    if (ctx->kv_neg_buf)
        ggml_backend_buffer_clear(ctx->kv_neg_buf, 0);

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
        if (step >= (int)all_ids.size())
            break;

        int32_t cur_token = all_ids[step];
        bool need_logits = (step >= num_prompt - 1);

        // Build step embedding (positive path)
        float* emb = build_step_embedding(ctx, cur_token, cur_acoustic.data(), cur_mask, cur_t_before, cur_t_after);
        if (!emb) {
            fprintf(stderr, "tada: embed failed at step %d\n", step);
            return nullptr;
        }

        // LLM forward (positive — uses real tokens)
        talker_result tr = run_talker_kv(ctx, emb, 1, n_past, need_logits);
        free(emb);
        if (!tr.hidden) {
            fprintf(stderr, "tada: talker failed at step %d\n", step);
            return nullptr;
        }

        // LLM forward (negative — pad token substituted for CFG)
        // Build neg embedding: same acoustic/time, but token replaced with pad
        float* neg_hidden = nullptr;
        if (cfg_scale != 1.0f && ctx->kv_neg_k) {
            int32_t pad_id = 128004; // Llama <|finetune_right_pad_id|>
            auto pad_it = ctx->vocab.token_to_id.find("<|finetune_right_pad_id|>");
            if (pad_it != ctx->vocab.token_to_id.end())
                pad_id = pad_it->second;

            // Keep structural tokens, replace content with pad
            bool is_structural = (cur_token == eot);
            int32_t neg_token = is_structural ? cur_token : pad_id;

            float* neg_emb =
                build_step_embedding(ctx, neg_token, cur_acoustic.data(), cur_mask, cur_t_before, cur_t_after);
            if (neg_emb) {
                talker_result neg_tr = run_talker_kv(ctx, neg_emb, 1, n_past, false, ctx->kv_neg_k, ctx->kv_neg_v);
                free(neg_emb);
                neg_hidden = neg_tr.hidden; // caller frees
            }

            if (step < 3 || (step >= shift && step < shift + 3)) {
                float neg_rms = 0, pos_rms = 0;
                for (int j = 0; j < (int)hp.fm_hidden; j++) {
                    neg_rms += neg_hidden[j] * neg_hidden[j];
                    pos_rms += tr.hidden[j] * tr.hidden[j];
                }
                neg_rms = std::sqrt(neg_rms / hp.fm_hidden);
                pos_rms = std::sqrt(pos_rms / hp.fm_hidden);
                // Cosine similarity
                float dot = 0, na = 0, nb = 0;
                for (int j = 0; j < (int)hp.fm_hidden; j++) {
                    dot += tr.hidden[j] * neg_hidden[j];
                    na += tr.hidden[j] * tr.hidden[j];
                    nb += neg_hidden[j] * neg_hidden[j];
                }
                float cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12f);
                fprintf(stderr, "  step %d: pos_h_rms=%.4f neg_h_rms=%.4f cos_sim=%.4f tok=%d neg_tok=%d\n", step,
                        pos_rms, neg_rms, cos_sim, cur_token, neg_token);
            }
        }
        n_past++;

        // ── Flow matching solver ──
        std::vector<float> speech(lat);
        // Initialize noise
        for (int j = 0; j < lat; j++) {
            speech[j] = rng_normal(&ctx->rng_state) * noise_temp;
        }

        // Solve ODE with proper negative conditioning
        float noise_rms = 0;
        for (int j = 0; j < ad; j++)
            noise_rms += speech[j] * speech[j];
        noise_rms = std::sqrt(noise_rms / ad);

        // Dump Euler trajectory for first 2 GENERATED (non-prompt) features
        int feat_idx_for_dump = step - shift;
        bool is_generated = (feat_idx_for_dump >= ctx->n_prompt);
        bool dump_trajectory = (is_generated && feat_idx_for_dump < ctx->n_prompt + 2 && ctx->params.verbosity >= 1);
        if (dump_trajectory) {
            fprintf(stderr, "  === Euler trajectory step %d ===\n", step);
            fprintf(stderr, "  noise: rms=%.4f [%.4f,%.4f,%.4f,%.4f,%.4f]\n", noise_rms, speech[0], speech[1],
                    speech[2], speech[3], speech[4]);
        }

        fm_euler_solve(ctx, speech.data(), tr.hidden, num_fm_steps, cfg_scale, neg_hidden, dump_trajectory);

        float speech_rms = 0;
        for (int j = 0; j < ad; j++)
            speech_rms += speech[j] * speech[j];
        speech_rms = std::sqrt(speech_rms / ad);

        // Also dump hidden state RMS for conditioning
        float cond_rms = 0;
        for (int j = 0; j < (int)hp.fm_hidden; j++)
            cond_rms += tr.hidden[j] * tr.hidden[j];
        cond_rms = std::sqrt(cond_rms / hp.fm_hidden);

        if (ctx->params.verbosity >= 2 || (step >= shift && step < shift + 3)) {
            fprintf(stderr, "  step %d: noise_rms=%.4f speech_rms=%.4f cond_rms=%.4f\n", step, noise_rms, speech_rms,
                    cond_rms);
        }

        // Free negative hidden state
        if (neg_hidden) {
            free(neg_hidden);
            neg_hidden = nullptr;
        }

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
            int feat_idx = step - shift;
            // Python pads prompt features with prefix_len zeros to align prompt audio
            // with prompt TEXT tokens. prefix_len tokens have zero acoustic features.
            // Prompt audio occupies indices [prefix_len, prefix_len + n_prompt) in the
            // virtual padded array.
            int prompt_feat_idx = feat_idx - prefix_len; // index into actual prompt data
            bool in_prefix = (feat_idx < prefix_len);
            bool in_prompt =
                (!in_prefix && ctx->n_prompt > 0 && prompt_feat_idx >= 0 && prompt_feat_idx < ctx->n_prompt);

            if (in_prefix) {
                // Prefix positions: zero acoustic (matches Python's left-padding)
                std::vector<float> feat(ad, 0.0f);
                acoustic_features.push_back(feat);
                time_before_list.push_back(pred_t_before);
                cur_acoustic = feat;
                cur_mask = 0;
                cur_t_before = pred_t_before;
                cur_t_after = pred_t_after;
            } else if (in_prompt) {
                // Use pre-computed prompt features
                std::vector<float> feat(ctx->prompt_values.begin() + prompt_feat_idx * ad,
                                        ctx->prompt_values.begin() + (prompt_feat_idx + 1) * ad);
                acoustic_features.push_back(feat);
                int tb = (prompt_feat_idx + 1 < (int)ctx->prompt_time_before.size())
                             ? ctx->prompt_time_before[prompt_feat_idx + 1]
                             : pred_t_before;
                time_before_list.push_back(tb);
                cur_acoustic = feat;
                cur_mask = 1;
                cur_t_before = tb;
                cur_t_after = (prompt_feat_idx + 1 < (int)ctx->prompt_time_after.size())
                                  ? ctx->prompt_time_after[prompt_feat_idx + 1]
                                  : pred_t_after;
            } else {
                // Use FM-predicted features
                std::vector<float> feat(speech.begin(), speech.begin() + ad);
                acoustic_features.push_back(feat);
                time_before_list.push_back(pred_t_before);
                cur_acoustic = feat;
                cur_mask = 1;
                cur_t_before = pred_t_before;
                cur_t_after = pred_t_after;
            }
        } else {
            std::fill(cur_acoustic.begin(), cur_acoustic.end(), 0.0f);
            cur_mask = 0;
            cur_t_before = pred_t_before;
            cur_t_after = pred_t_after;
        }

        free(tr.hidden);
    }

    if (acoustic_features.empty()) {
        fprintf(stderr, "tada: no acoustic features generated\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: %zu acoustic frames, %zu time values\n", acoustic_features.size(),
                time_before_list.size());
        // Dump feature stats for debugging
        for (size_t i = 0; i < acoustic_features.size(); i++) {
            float rms = 0;
            for (int d = 0; d < ad; d++)
                rms += acoustic_features[i][d] * acoustic_features[i][d];
            rms = std::sqrt(rms / ad);
            bool is_prompt = (ctx->n_prompt > 0 && (int)i < ctx->n_prompt);
            if (i < 5 || i >= acoustic_features.size() - 3) {
                fprintf(stderr, "  feat[%zu] rms=%.4f %s\n", i, rms, is_prompt ? "(prompt)" : "(generated)");
            }
        }
    }

    // ── Denormalize and expand ──
    // features = features * acoustic_std + acoustic_mean
    float ac_std = hp.acoustic_std;
    float ac_mean = hp.acoustic_mean;

    // Skip prompt + prefix + transition frames to match Python's:
    //   num_prompt_tokens = prefix_len + (n_prompt - transition)   [after pad + trim]
    //   skip = num_prompt_tokens + transition - 1
    //        = prefix_len + n_prompt - 1
    int skip_frames = 0;
    if (ctx->n_prompt > 0) {
        int num_transition_steps = 5;
        skip_frames = prefix_len + ctx->n_prompt - 1;
        if (skip_frames >= (int)acoustic_features.size()) {
            skip_frames = std::max(0, (int)acoustic_features.size() - 1);
        }
        if (ctx->params.verbosity >= 1) {
            fprintf(stderr, "tada: skipping %d prompt+transition frames, decoding %d\n", skip_frames,
                    (int)acoustic_features.size() - skip_frames);
        }
    }

    // Expand with time_before durations (same as model._decode_wav)
    std::vector<float> expanded;
    std::vector<int32_t> token_masks;
    std::vector<int> all_times;
    // time_before for the decode portion starts at skip_frames
    all_times.push_back(0);
    for (int i = skip_frames; i < (int)time_before_list.size(); i++) {
        all_times.push_back(time_before_list[i]);
    }

    // Use only features from skip_frames onwards
    std::vector<std::vector<float>> decode_feats(acoustic_features.begin() + skip_frames, acoustic_features.end());

    for (size_t i = 0; i < decode_feats.size(); i++) {
        // Insert (time - 1) zero frames before this feature
        int n_zeros = std::max(0, all_times[i] - 1);
        for (int z = 0; z < n_zeros; z++) {
            for (int d = 0; d < ad; d++)
                expanded.push_back(0.0f);
            token_masks.push_back(0);
        }
        // Insert the feature (denormalized)
        for (int d = 0; d < ad; d++) {
            expanded.push_back(decode_feats[i][d] * ac_std + ac_mean);
        }
        token_masks.push_back(1);
    }
    // Trailing zeros from last time value
    if (!all_times.empty()) {
        int trail = all_times.back();
        for (int z = 0; z < trail; z++) {
            for (int d = 0; d < ad; d++)
                expanded.push_back(0.0f);
            token_masks.push_back(0);
        }
    }

    int n_expanded = (int)(expanded.size() / ad);
    int n_masks_set = 0;
    for (auto m : token_masks)
        n_masks_set += m;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: time_before values: [");
        for (size_t i = 0; i < all_times.size() && i < 20; i++) {
            fprintf(stderr, "%d%s", all_times[i], i + 1 < all_times.size() ? "," : "");
        }
        fprintf(stderr, "]\n");
        fprintf(stderr, "tada: decode_feats=%zu, n_expanded=%d, token_masks=%d/%d\n", decode_feats.size(), n_expanded,
                n_masks_set, (int)token_masks.size());
        // Dump expanded feature stats
        for (int i = 0; i < n_expanded; i++) {
            if (token_masks[i]) {
                float rms = 0;
                for (int d = 0; d < ad; d++) {
                    float v = expanded[i * ad + d];
                    rms += v * v;
                }
                rms = std::sqrt(rms / ad);
                fprintf(stderr, "  expanded[%d] rms=%.4f (non-zero)\n", i, rms);
            }
        }
        fprintf(stderr, "tada: %zu features → %d expanded frames\n", acoustic_features.size(), n_expanded);
    }

    // ── Codec decode ──
    if (ctx->codec_ctx && n_expanded > 0) {
        int n_samples = 0;
        float* pcm = tada_codec_decode(ctx->codec_ctx, expanded.data(), n_expanded, token_masks.data(), &n_samples);
        if (pcm && n_samples > 0) {
            *out_n_samples = n_samples;
            return pcm;
        }
        fprintf(stderr, "tada: codec decode failed, returning silence\n");
    } else if (!ctx->codec_ctx) {
        fprintf(stderr, "tada: no codec loaded — returning silence\n");
    }

    // Fallback: silence
    int n_samples = n_expanded * 480;
    if (n_samples <= 0)
        n_samples = 24000;
    float* pcm = (float*)calloc(n_samples, sizeof(float));
    *out_n_samples = n_samples;
    return pcm;
}

void tada_pcm_free(float* pcm) {
    free(pcm);
}

void tada_free(struct tada_context* ctx) {
    if (!ctx)
        return;
    if (ctx->codec_ctx)
        tada_codec_free(ctx->codec_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_neg_buf)
        ggml_backend_buffer_free(ctx->kv_neg_buf);
    if (ctx->kv_neg_ctx)
        ggml_free(ctx->kv_neg_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

void tada_test_fm_step(struct tada_context* ctx, const float* noisy_z, const float* t_emb_sin, const float* cond,
                       float* velocity_out) {
    if (!ctx)
        return;
    const int lat = (int)ctx->hp.fm_latent;
    const int hid = (int)ctx->hp.fm_hidden;

    ggml_cgraph* gf = build_graph_fm_step(ctx);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "tada: test_fm_step alloc failed\n");
        return;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0, (size_t)lat * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb_sin, 0, 256 * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond"), cond, 0, (size_t)hid * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "tada: test_fm_step compute failed\n");
        return;
    }

    ggml_tensor* vel = ggml_graph_get_tensor(gf, "velocity");
    ggml_backend_tensor_get(vel, velocity_out, 0, (size_t)lat * sizeof(float));
}

} // extern "C"
