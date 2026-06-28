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
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────── internal types ────────────────────────────

namespace {

// ===========================================================================
// Bench instrumentation — `TADA_BENCH=1` for per-stage timings.
// ===========================================================================

static bool tada_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("TADA_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct tada_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit tada_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~tada_bench_stage() {
        if (!tada_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  tada_bench: %-22s %.2f ms\n", name, ms);
    }
};

struct tada_fm_dump_record {
    int32_t step = 0;
    int32_t feat_idx = 0;
    std::vector<float> speech_in;
    std::vector<float> cond;
    std::vector<float> neg_cond;
    std::vector<float> speech_out;
    std::vector<float> time_bits;
};

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

struct mt19937_state_t {
    uint32_t mt[624];
    int32_t left;
    int32_t pos;
};

struct tada_context {
    tada_context_params params;
    tada_hp hp;
    tada_vocab vocab;
    tada_talker talker;
    tada_fm_head fm;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    bool backend_is_vulkan = false;          // backend is a Vulkan device (#192)
    bool vulkan_native = false;              // CRISPASR_TADA_VULKAN_NATIVE: compute graphs direct (no sched)
    ggml_gallocr_t ar_step_galloc = nullptr; // direct-compute allocator for talker/embed graphs
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
    std::string prompt_text;
    int n_prompt = 0;

    uint64_t rng_state = 0;
    mt19937_state_t mt_rng = {};

    // §176b: Lk-bucketed single-step AR graph cache.
    struct TadaBucket {
        int lk = 0;
        ggml_context* ctx = nullptr;
        std::vector<uint8_t> meta;
        ggml_cgraph* gf = nullptr;
    };
    static constexpr int kBucketN = 4;
    static constexpr int kBucketLks[kBucketN] = {512, 1024, 2048, 4096};
    std::array<TadaBucket, kBucketN> ar_buckets{};
    ggml_backend_sched_t ar_step_sched = nullptr;

    // Fixed-shape FM velocity graph cache.
    ggml_context* fm_step_ctx = nullptr;
    std::vector<uint8_t> fm_step_meta;
    ggml_cgraph* fm_step_gf = nullptr;
    ggml_backend_sched_t fm_step_sched = nullptr;
    ggml_gallocr_t fm_step_galloc = nullptr;

    // B=2 batched CFG FM graph cache (CRISPASR_TADA_FM_B2).
    // Batches pos+neg condition in one forward; halves FM graph compute calls.
    ggml_context* fm_b2_ctx = nullptr;
    std::vector<uint8_t> fm_b2_meta;
    ggml_cgraph* fm_b2_gf = nullptr;
    ggml_backend_sched_t fm_b2_sched = nullptr;
    ggml_gallocr_t fm_b2_galloc = nullptr;
    bool fm_b2_active = false;
    ggml_context* fm_b2_ctx_f16 = nullptr;
    ggml_backend_buffer_t fm_b2_buf_f16 = nullptr;
    tada_fm_head fm_b2_f16{};

    // General batched FM graph cache (arbitrary batch B), used by candidate
    // ranking (num_acoustic_candidates>1): all candidates × CFG are solved in a
    // single forward of batch B per ODE step instead of looping B=2 forwards.
    // Cached graph+sched are reused across ODE steps and AR steps; rebuilt only
    // when B changes. Mirrors Python _solve_flow_matching_ranked's batched solve.
    ggml_context* fm_batch_ctx = nullptr;
    std::vector<uint8_t> fm_batch_meta;
    ggml_cgraph* fm_batch_gf = nullptr;
    ggml_backend_sched_t fm_batch_sched = nullptr;
    ggml_gallocr_t fm_batch_galloc = nullptr;
    int fm_batch_B = 0;
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
                                          ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr,
                                          int fixed_kv_len = 0, ggml_context* arena_ctx = nullptr) {
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
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    GGML_ASSERT(use_kv_k && use_kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = arena_ctx ? arena_ctx : ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1 || fixed_kv_len > 0) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    core_attn::KvSelfAttnParams kvp = {
        n_q, n_kv, hd, n_kv_grp, (int)hp.max_pos, theta, 0.0f, 0.0f, attn_scale, 0.0f, core_attn::GQA_MANUAL_CONT,
    };
    // On the native Vulkan path the F16 KV cache can't be GQA-expanded: Vulkan has
    // no REPEAT f16→f16 pipeline (aborts "Missing op: REPEAT for f16 to f16" on
    // both RADV and MoltenVK; #192). Force the cache read up to F32 so the GQA
    // repeat lowers to a supported F32 REPEAT. Metal/CPU keep the F16 fast path.
    kvp.force_kv_read_f32 = c->vulkan_native;

    ggml_tensor* eff_kv_indices = fixed_kv_len > 0 ? positions : nullptr;

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->talker.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, nullptr, nullptr, positions,
            (T == 1 && !fixed_kv_len) ? nullptr : causal_mask, use_kv_k, use_kv_v, (int)il, n_past, kvp,
            /*qkv_w=*/nullptr, /*fixed_kv_len=*/fixed_kv_len, /*kv_indices=*/eff_kv_indices);
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

    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

// VibeVoiceDiffusionHead forward pass.
// Inputs: noisy_z (latent_dim,), timestep (scalar float), condition (hidden_dim,)
// Output: predicted velocity (latent_dim,)
static ggml_cgraph* build_graph_fm_step(tada_context* c, ggml_context* arena_ctx = nullptr) {
    const auto& hp = c->hp;
    const int hid = (int)hp.fm_hidden;
    const int lat = (int)hp.fm_latent;
    (void)hp; // fm_ffn_dim used implicitly via layer weights
    const float eps = hp.rms_norm_eps;

    ggml_context* ctx0 = arena_ctx;
    if (!ctx0) {
        ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
        ctx0 = ggml_init(ip);
    }
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

    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

// B=2 batched CFG FM graph: runs pos+neg cond in one forward pass.
// Inputs:
//   noisy_z  (lat,)    — same for both paths
//   t_emb_sin (256,)   — same for both paths
//   fm_cond_b2 (hid,2) — cond[:,0]=pos, cond[:,1]=neg stacked column-wise
// Output:
//   velocity_b2 (lat,2) — vel[:,0]=pos_vel, vel[:,1]=neg_vel
static ggml_cgraph* build_graph_fm_step_b2(tada_context* c, ggml_context* arena_ctx = nullptr) {
    const auto& hp = c->hp;
    const tada_fm_head& fm = c->fm_b2_f16.layers.empty() ? c->fm : c->fm_b2_f16;
    const int hid = (int)hp.fm_hidden;
    const int lat = (int)hp.fm_latent;
    const float eps = hp.rms_norm_eps;

    ggml_context* ctx0 = arena_ctx;
    if (!ctx0) {
        ggml_init_params ip = {c->fm_b2_meta.size(), c->fm_b2_meta.data(), true};
        ctx0 = ggml_init(ip);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Shared inputs (same for both CFG paths)
    ggml_tensor* noisy_z = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, lat);
    ggml_set_name(noisy_z, "noisy_z");
    ggml_set_input(noisy_z);

    ggml_tensor* t_emb_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_emb_sin, "t_emb_sin");
    ggml_set_input(t_emb_sin);

    // Batched condition input: (hid, 2) — col 0 = pos, col 1 = neg
    ggml_tensor* cond_b2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hid, 2);
    ggml_set_name(cond_b2, "fm_cond_b2");
    ggml_set_input(cond_b2);

    // Shared branch: x = noisy_proj(noisy_z) → (hid,), then replicate to (hid, 2)
    ggml_tensor* x_shared = ggml_mul_mat(ctx0, fm.noisy_proj_w, noisy_z);
    ggml_tensor* x_b2 = ggml_repeat(ctx0, x_shared, cond_b2);

    // Shared branch: t = t_mlp(t_emb_sin) → (hid,), then replicate to (hid, 2)
    ggml_tensor* t_shared = ggml_mul_mat(ctx0, fm.t_emb_mlp0_w, t_emb_sin);
    t_shared = ggml_silu(ctx0, t_shared);
    t_shared = ggml_mul_mat(ctx0, fm.t_emb_mlp1_w, t_shared);
    ggml_tensor* t_b2 = ggml_repeat(ctx0, t_shared, cond_b2);

    // Batched: cond_proj(cond_b2) → (hid, 2)
    ggml_tensor* c_emb = ggml_add(ctx0, ggml_mul_mat(ctx0, fm.cond_proj_w, cond_b2), t_b2);

    // Head layers: all ops run on (hid, 2) batch
    for (uint32_t i = 0; i < hp.head_layers; i++) {
        const auto& l = fm.layers[i];

        // adaLN: silu(c_emb) @ adaln_w^T → (3*hid, 2), then chunk
        ggml_tensor* mod = ggml_silu(ctx0, c_emb);
        mod = ggml_mul_mat(ctx0, l.adaln_w, mod); // (3*hid, 2)

        // Chunk into shift/scale/gate — each (hid, 2) via 2D strided views.
        // mod is contiguous (3*hid, 2): nb[0]=4, nb[1]=3*hid*4 (non-natural).
        // Vulkan element-wise kernels require contiguous input; wrap with
        // ggml_cont() so each view materialises as a (hid, 2) contiguous tensor.
        size_t col_stride = (size_t)3 * hid * sizeof(float);
        ggml_tensor* shift = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, 2, col_stride, 0));
        ggml_tensor* scale = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, 2, col_stride, (size_t)hid * sizeof(float)));
        ggml_tensor* gate =
            ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, 2, col_stride, (size_t)2 * hid * sizeof(float)));

        // Modulate + FFN
        ggml_tensor* h = ggml_rms_norm(ctx0, x_b2, eps);
        h = ggml_mul(ctx0, h, ggml_repeat(ctx0, l.norm_w, x_b2)); // broadcast norm_w
        h = ggml_add(ctx0, ggml_add(ctx0, h, ggml_mul(ctx0, h, scale)), shift);

        // SwiGLU FFN — gate_w, up_w, down_w operate on (hid, 2)
        ggml_tensor* ffn_out = core_ffn::swiglu(ctx0, h, l.ffn_gate_w, l.ffn_up_w, l.ffn_down_w);

        // Gated residual
        x_b2 = ggml_add(ctx0, x_b2, ggml_mul(ctx0, gate, ffn_out));
    }

    // Final layer: 2-way adaLN → proj
    {
        ggml_tensor* mod = ggml_silu(ctx0, c_emb);
        mod = ggml_mul_mat(ctx0, fm.final_adaln_w, mod); // (2*hid, 2)
        size_t col_stride = (size_t)2 * hid * sizeof(float);
        ggml_tensor* shift = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, 2, col_stride, 0));
        ggml_tensor* scale = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, 2, col_stride, (size_t)hid * sizeof(float)));

        ggml_tensor* h = ggml_rms_norm(ctx0, x_b2, eps);
        if (fm.final_norm_w)
            h = ggml_mul(ctx0, h, ggml_repeat(ctx0, fm.final_norm_w, x_b2));
        h = ggml_add(ctx0, ggml_add(ctx0, h, ggml_mul(ctx0, h, scale)), shift);
        h = ggml_mul_mat(ctx0, fm.final_proj_w, h); // (lat, 2)
        ggml_set_name(h, "velocity_b2");
        ggml_build_forward_expand(gf, h);
    }

    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

static void sinusoidal_embedding(float t, int dim, float* out); // defined below

// General batched FM-head graph: identical math to build_graph_fm_step_b2 but
// with an arbitrary batch dimension B and a per-column noisy state (the b2
// graph shares a single noisy_z across its 2 CFG columns; here every column has
// its own state, so candidate solving and reconstruction scoring can batch
// many independent (noise, cond) pairs into one forward). The timestep is
// shared across the batch (all columns evaluated at the same t).
static ggml_cgraph* build_graph_fm_step_batch(tada_context* c, int B, ggml_context* arena_ctx) {
    const auto& hp = c->hp;
    const tada_fm_head& fm = c->fm_b2_f16.layers.empty() ? c->fm : c->fm_b2_f16;
    const int hid = (int)hp.fm_hidden;
    const int lat = (int)hp.fm_latent;
    const float eps = hp.rms_norm_eps;

    ggml_context* ctx0 = arena_ctx;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Per-column inputs: noisy_z (lat, B), cond (hid, B); shared t (256).
    ggml_tensor* noisy_b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, lat, B);
    ggml_set_name(noisy_b, "noisy_b");
    ggml_set_input(noisy_b);

    ggml_tensor* t_emb_sin = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 256);
    ggml_set_name(t_emb_sin, "t_emb_sin");
    ggml_set_input(t_emb_sin);

    ggml_tensor* cond_b = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hid, B);
    ggml_set_name(cond_b, "cond_b");
    ggml_set_input(cond_b);

    // x = noisy_proj(noisy_b) → (hid, B). No repeat: each column is independent.
    ggml_tensor* x_b = ggml_mul_mat(ctx0, fm.noisy_proj_w, noisy_b);

    // Shared timestep MLP → (hid,), broadcast across the B columns.
    ggml_tensor* t_shared = ggml_mul_mat(ctx0, fm.t_emb_mlp0_w, t_emb_sin);
    t_shared = ggml_silu(ctx0, t_shared);
    t_shared = ggml_mul_mat(ctx0, fm.t_emb_mlp1_w, t_shared);
    ggml_tensor* t_b = ggml_repeat(ctx0, t_shared, cond_b);

    ggml_tensor* c_emb = ggml_add(ctx0, ggml_mul_mat(ctx0, fm.cond_proj_w, cond_b), t_b);

    for (uint32_t i = 0; i < hp.head_layers; i++) {
        const auto& l = fm.layers[i];

        ggml_tensor* mod = ggml_silu(ctx0, c_emb);
        mod = ggml_mul_mat(ctx0, l.adaln_w, mod); // (3*hid, B)

        size_t col_stride = (size_t)3 * hid * sizeof(float);
        ggml_tensor* shift = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, B, col_stride, 0));
        ggml_tensor* scale = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, B, col_stride, (size_t)hid * sizeof(float)));
        ggml_tensor* gate =
            ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, B, col_stride, (size_t)2 * hid * sizeof(float)));

        ggml_tensor* h = ggml_rms_norm(ctx0, x_b, eps);
        h = ggml_mul(ctx0, h, ggml_repeat(ctx0, l.norm_w, x_b));
        h = ggml_add(ctx0, ggml_add(ctx0, h, ggml_mul(ctx0, h, scale)), shift);

        ggml_tensor* ffn_out = core_ffn::swiglu(ctx0, h, l.ffn_gate_w, l.ffn_up_w, l.ffn_down_w);
        x_b = ggml_add(ctx0, x_b, ggml_mul(ctx0, gate, ffn_out));
    }

    {
        ggml_tensor* mod = ggml_silu(ctx0, c_emb);
        mod = ggml_mul_mat(ctx0, fm.final_adaln_w, mod); // (2*hid, B)
        size_t col_stride = (size_t)2 * hid * sizeof(float);
        ggml_tensor* shift = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, B, col_stride, 0));
        ggml_tensor* scale = ggml_cont(ctx0, ggml_view_2d(ctx0, mod, hid, B, col_stride, (size_t)hid * sizeof(float)));

        ggml_tensor* h = ggml_rms_norm(ctx0, x_b, eps);
        if (fm.final_norm_w)
            h = ggml_mul(ctx0, h, ggml_repeat(ctx0, fm.final_norm_w, x_b));
        h = ggml_add(ctx0, ggml_add(ctx0, h, ggml_mul(ctx0, h, scale)), shift);
        h = ggml_mul_mat(ctx0, fm.final_proj_w, h); // (lat, B)
        ggml_set_name(h, "velocity_b");
        ggml_build_forward_expand(gf, h);
    }

    return gf;
}

// Run one batched FM step: B independent (noisy, cond) columns at a shared
// timestep, one forward. Reuses a cached graph+sched keyed by B (rebuilt only
// when B changes), so the per-ODE-step host build cost is paid once per run.
static bool run_fm_step_batch(tada_context* c, const float* noisy_b, float timestep, const float* cond_b,
                              float* vel_b_out, int B) {
    const int lat = (int)c->hp.fm_latent;
    const int hid = (int)c->hp.fm_hidden;

    float t_emb[256];
    sinusoidal_embedding(timestep, 256, t_emb);

    if (c->fm_batch_B != B) {
        // (Re)build the graph + sched for this batch size.
        if (c->fm_batch_ctx) {
            ggml_free(c->fm_batch_ctx);
            c->fm_batch_ctx = nullptr;
        }
        if (c->fm_batch_sched) {
            ggml_backend_sched_free(c->fm_batch_sched);
            c->fm_batch_sched = nullptr;
        }
        c->fm_batch_meta.assign(c->compute_meta.size() * 2, 0);
        ggml_init_params ip = {c->fm_batch_meta.size(), c->fm_batch_meta.data(), true};
        c->fm_batch_ctx = ggml_init(ip);
        if (!c->fm_batch_ctx)
            return false;
        c->fm_batch_gf = build_graph_fm_step_batch(c, B, c->fm_batch_ctx);
        ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
        int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
        c->fm_batch_sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
        if (c->fm_batch_galloc) {
            ggml_gallocr_free(c->fm_batch_galloc);
            c->fm_batch_galloc = nullptr;
        }
        c->fm_batch_B = B;
    }
    ggml_cgraph* gf = c->fm_batch_gf;

    // Vulkan native path: direct compute on the backend (no sched split) — see run_fm_step (#192).
    if (c->vulkan_native) {
        if (!c->fm_batch_galloc)
            c->fm_batch_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->fm_batch_galloc, gf))
            return false;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_b"), noisy_b, 0, (size_t)lat * B * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_b"), cond_b, 0, (size_t)hid * B * sizeof(float));
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS)
            return false;
    } else {
        ggml_backend_sched_t sched = c->fm_batch_sched;
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
            return false;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_b"), noisy_b, 0, (size_t)lat * B * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_b"), cond_b, 0, (size_t)hid * B * sizeof(float));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
            return false;
    }

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "velocity_b"), vel_b_out, 0, (size_t)lat * B * sizeof(float));
    return true;
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

// xorshift64* RNG (kept for fallback; AR loop uses mt19937_randn below)
static uint64_t rng_next(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_normal(uint64_t* state) {
    double u1 = (double)(rng_next(state) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(rng_next(state) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-12)
        u1 = 1e-12;
    return (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2));
}

// ── PyTorch-compatible MT19937 + normal_fill ──────────────────────────────
// Matches torch.randn(N) on CPU/ARM (M1 Mac non-AVX2 path) so that noise
// vectors are bit-identical to the Python reference when seeded with the
// same integer.  Algorithm: normal_fill / normal_fill_16 from
// aten/src/ATen/native/cpu/DistributionTemplates.h.

static void mt19937_init(mt19937_state_t* s, uint32_t seed) {
    s->mt[0] = seed;
    for (int j = 1; j < 624; j++) {
        s->mt[j] = (1812433253u * (s->mt[j - 1] ^ (s->mt[j - 1] >> 30)) + (uint32_t)j) & 0xffffffffu;
    }
    s->left = 1; // PyTorch CPUGeneratorImpl initial value → twist on first draw
    s->pos = 0;
}

static void mt19937_twist(mt19937_state_t* s) {
    static const uint32_t A = 0x9908b0dfu;
    uint32_t* mt = s->mt;
    for (int j = 0; j < 227; j++) {
        uint32_t y = (mt[j] & 0x80000000u) | (mt[j + 1] & 0x7fffffffu);
        mt[j] = mt[j + 397] ^ (y >> 1) ^ ((y & 1u) ? A : 0u);
    }
    for (int j = 227; j < 623; j++) {
        uint32_t y = (mt[j] & 0x80000000u) | (mt[j + 1] & 0x7fffffffu);
        mt[j] = mt[j - 227] ^ (y >> 1) ^ ((y & 1u) ? A : 0u);
    }
    uint32_t y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
    mt[623] = mt[396] ^ (y >> 1) ^ ((y & 1u) ? A : 0u);
    s->left = 624;
    s->pos = 0;
}

static uint32_t mt19937_next(mt19937_state_t* s) {
    s->left--;
    if (s->left == 0)
        mt19937_twist(s);
    uint32_t y = s->mt[s->pos++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

// Generate N float32 N(0,1) samples matching PyTorch's normal_fill<float>:
//   1. Fill array with uniform floats via lower-24-bits / 2^24
//   2. Process 16-element blocks: u1=1-data[j], u2=data[j+8],
//      r=sqrtf(-2*logf(u1)), theta=(2π_double)*u2,
//      data[j]=r*cos(theta), data[j+8]=r*sin(theta)
//   3. Re-draw last 16 if size%16≠0 and repeat
static void mt19937_randn(mt19937_state_t* s, float* data, int N) {
    static const uint32_t MASK24 = (1u << 24) - 1u;
    static const float DIV24 = 1.0f / (float)(1u << 24);
    static const double TWO_PI = 2.0 * M_PI; // double precision, matches c10::pi<double>

    for (int i = 0; i < N; i++)
        data[i] = (float)(mt19937_next(s) & MASK24) * DIV24;

    for (int i = 0; i <= N - 16; i += 16) {
        for (int j = 0; j < 8; j++) {
            float u1 = 1.0f - data[i + j];
            float u2 = data[i + j + 8];
            float r = std::sqrt(-2.0f * std::log(u1));
            double theta = TWO_PI * (double)u2;
            data[i + j] = (float)((double)r * std::cos(theta));
            data[i + j + 8] = (float)((double)r * std::sin(theta));
        }
    }

    if (N % 16 != 0) {
        int tail = N - 16;
        for (int i = 0; i < 16; i++)
            data[tail + i] = (float)(mt19937_next(s) & MASK24) * DIV24;
        for (int j = 0; j < 8; j++) {
            float u1 = 1.0f - data[tail + j];
            float u2 = data[tail + j + 8];
            float r = std::sqrt(-2.0f * std::log(u1));
            double theta = TWO_PI * (double)u2;
            data[tail + j] = (float)((double)r * std::cos(theta));
            data[tail + j + 8] = (float)((double)r * std::sin(theta));
        }
    }
}

// Round-to-nearest-even F32 value to BF16 precision (matches Python bf16 cast).
// Python's torch.bfloat16 truncates the FM noise before scaling:
//   noise = torch.randn(lat).to(bfloat16) * noise_temperature
// Both the initial draw and the scaled result are in BF16, so we must apply the
// same rounding to match the Python reference bit-for-bit.
static float tada_round_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    u += rounding_bias;
    u &= 0xffff0000u;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static void tada_scale_noise_like_python(float* data, int n, float noise_temp) {
    for (int i = 0; i < n; i++) {
        data[i] = tada_round_bf16(tada_round_bf16(data[i]) * noise_temp);
    }
}

// Embed tokens → float array (d_model * n_tokens)
static float* embed_tokens(tada_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed(c, n);
    if (c->vulkan_native) {
        if (!c->ar_step_galloc)
            c->ar_step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->ar_step_galloc, gf))
            return nullptr;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), ids, 0, (size_t)n * sizeof(int32_t));
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS)
            return nullptr;
    } else {
        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf))
            return nullptr;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), ids, 0, (size_t)n * sizeof(int32_t));
        if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
            return nullptr;
    }
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
    if (c->vulkan_native) {
        if (!c->ar_step_galloc)
            c->ar_step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->ar_step_galloc, gf))
            return nullptr;
    } else {
        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf))
            return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "token_id"), &token_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "acoustic"), acoustic, 0, (size_t)ad * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask_id"), &mask_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_before"), &t_before, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_after"), &t_after, 0, sizeof(int32_t));

    if (c->vulkan_native) {
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS)
            return nullptr;
    } else if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "step_embed");
    float* r = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * sizeof(float));
    return r;
}

// §176b: Lk-bucketed single-step AR decode helpers.
static int tada_pick_bucket(tada_context* c, int needed_lk) {
    for (int i = 0; i < tada_context::kBucketN; i++)
        if (tada_context::kBucketLks[i] >= needed_lk && tada_context::kBucketLks[i] <= c->kv_max_ctx)
            return i;
    return -1;
}

static ggml_backend_sched_t tada_step_sched_lazy(tada_context* c) {
    if (c->ar_step_sched)
        return c->ar_step_sched;
    ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
    int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
    c->ar_step_sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    return c->ar_step_sched;
}

static ggml_cgraph* tada_get_or_build_bucket(tada_context* c, int idx) {
    auto& bk = c->ar_buckets[idx];
    if (bk.gf)
        return bk.gf;
    bk.lk = tada_context::kBucketLks[idx];
    bk.meta.assign(c->compute_meta.size(), 0);
    ggml_init_params ip = {bk.meta.size(), bk.meta.data(), true};
    bk.ctx = ggml_init(ip);
    if (!bk.ctx)
        return nullptr;
    bk.gf = build_graph_talker_kv(c, 0, 1, true, nullptr, nullptr, bk.lk, bk.ctx);
    return bk.gf;
}

// Run LLM forward pass. Returns hidden_state (d_model,) and optionally logits.
// Both are malloc'd. Caller frees.
struct talker_result {
    float* hidden; // (d_model,)
    float* logits; // (vocab,) or nullptr
};

static talker_result run_talker_kv_bucket(tada_context* c, const float* embeds, int n_past, bool need_logits) {
    talker_result res = {nullptr, nullptr};
    const int idx = tada_pick_bucket(c, n_past + 1);
    if (idx < 0)
        return res;
    ggml_cgraph* gf = tada_get_or_build_bucket(c, idx);
    if (!gf)
        return res;
    // Vulkan native path: direct compute on the backend (no {backend,CPU} sched).
    // This needs a driver with a REPEAT pipeline for the GQA head expansion (RADV
    // has it; MoltenVK aborts), hence the CRISPASR_TADA_VULKAN_NATIVE gate (#192).
    ggml_backend_sched_t ss = nullptr;
    if (c->vulkan_native) {
        if (!c->ar_step_galloc)
            c->ar_step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->ar_step_galloc, gf))
            return res;
    } else {
        ss = tada_step_sched_lazy(c);
        ggml_backend_sched_reset(ss);
        if (!ggml_backend_sched_alloc_graph(ss, gf))
            return res;
    }
    const int d = (int)c->hp.d_model;
    const int vocab = (int)c->hp.vocab_size;
    const int Lk = c->ar_buckets[idx].lk;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0, (size_t)d * sizeof(float));
    int32_t pos = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), &pos, 0, sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t)Lk);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
    for (int k = 0; k < Lk; k++)
        mask[k] = (k <= n_past) ? z : ni;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                            mask.size() * sizeof(ggml_fp16_t));
    if (c->vulkan_native) {
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS)
            return res;
    } else if (ggml_backend_sched_graph_compute(ss, gf) != GGML_STATUS_SUCCESS) {
        return res;
    }
    res.hidden = (float*)malloc((size_t)d * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "hidden_state"), res.hidden, 0, (size_t)d * sizeof(float));
    if (need_logits) {
        res.logits = (float*)malloc((size_t)vocab * sizeof(float));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), res.logits, 0, (size_t)vocab * sizeof(float));
    }
    return res;
}

static talker_result run_talker_kv(tada_context* c, const float* embeds, int n_tokens, int n_past, bool need_logits,
                                   ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
    // §176b: Lk-bucketed fast path for single-step decode on default KV.
    if (n_tokens == 1 && !use_kv_k && !use_kv_v) {
        talker_result br = run_talker_kv_bucket(c, embeds, n_past, need_logits);
        if (br.hidden)
            return br;
    }
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

    // Vulkan native path: direct compute on the backend (no sched split) — #192.
    if (c->vulkan_native) {
        if (!c->ar_step_galloc)
            c->ar_step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->ar_step_galloc, gf))
            return res;
    } else {
        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf))
            return res;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (c->vulkan_native) {
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS)
            return res;
    } else if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return res;
    }

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

static ggml_backend_sched_t tada_fm_step_sched_lazy(tada_context* c) {
    if (c->fm_step_sched)
        return c->fm_step_sched;
    ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
    int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
    c->fm_step_sched = ggml_backend_sched_new(backends, nullptr, n_be, 4096, false, false);
    return c->fm_step_sched;
}

static ggml_cgraph* tada_get_or_build_fm_step(tada_context* c) {
    if (c->fm_step_gf)
        return c->fm_step_gf;
    c->fm_step_meta.assign(c->compute_meta.size(), 0);
    ggml_init_params ip = {c->fm_step_meta.size(), c->fm_step_meta.data(), true};
    c->fm_step_ctx = ggml_init(ip);
    if (!c->fm_step_ctx)
        return nullptr;
    c->fm_step_gf = build_graph_fm_step(c, c->fm_step_ctx);
    return c->fm_step_gf;
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

    ggml_cgraph* gf = tada_get_or_build_fm_step(c);
    if (!gf) {
        fprintf(stderr, "tada: failed to build fm_step graph\n");
        return;
    }
    // On the Vulkan native path, compute directly on the backend via a cached
    // gallocr, bypassing the {backend,CPU} scheduler whose cross-backend split
    // corrupts the FM head on Vulkan (#192).
    if (c->vulkan_native) {
        if (!c->fm_step_galloc)
            c->fm_step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->fm_step_galloc, gf)) {
            fprintf(stderr, "tada: failed to alloc fm_step graph (direct)\n");
            return;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0, (size_t)lat * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond"), cond_input, 0, (size_t)hid * sizeof(float));
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "tada: fm_step compute failed (direct)\n");
            return;
        }
    } else {
        ggml_backend_sched_t sched = tada_fm_step_sched_lazy(c);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "tada: failed to alloc fm_step graph\n");
            return;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0, (size_t)lat * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond"), cond_input, 0, (size_t)hid * sizeof(float));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "tada: fm_step compute failed\n");
            return;
        }
    }

    ggml_tensor* vel = ggml_graph_get_tensor(gf, "velocity");
    ggml_backend_tensor_get(vel, velocity_out, 0, (size_t)lat * sizeof(float));
}

// Lazily build F16 GPU-resident copies of quantized FM matmul weights for the
// B=2 CFG graph. This mirrors chatterbox T3 / s3gen: batched Metal matmuls
// against quantized weights can take a slower or lower-precision path; F16
// copies route through the stable f16 x f32 mul_mm path.
static bool ensure_fm_b2_f16_weights(tada_context* c) {
    if (!c->fm_b2_f16.layers.empty())
        return true;

    std::vector<ggml_tensor*> qsrc;
    auto want = [&](ggml_tensor* w) {
        if (w && ggml_is_quantized(w->type))
            qsrc.push_back(w);
    };

    want(c->fm.noisy_proj_w);
    want(c->fm.cond_proj_w);
    want(c->fm.t_emb_mlp0_w);
    want(c->fm.t_emb_mlp1_w);
    for (const auto& l : c->fm.layers) {
        want(l.ffn_gate_w);
        want(l.ffn_up_w);
        want(l.ffn_down_w);
        want(l.adaln_w);
    }
    want(c->fm.final_proj_w);
    want(c->fm.final_adaln_w);

    if (qsrc.empty())
        return false;

    const size_t meta = ggml_tensor_overhead() * (qsrc.size() + 8) + 4096;
    ggml_init_params fp = {meta, nullptr, true};
    c->fm_b2_ctx_f16 = ggml_init(fp);
    if (!c->fm_b2_ctx_f16)
        return false;

    std::map<ggml_tensor*, ggml_tensor*> q2f16;
    for (ggml_tensor* s : qsrc) {
        if (q2f16.count(s))
            continue;
        ggml_tensor* d = ggml_new_tensor(c->fm_b2_ctx_f16, GGML_TYPE_F16, ggml_n_dims(s), s->ne);
        if (!d) {
            ggml_free(c->fm_b2_ctx_f16);
            c->fm_b2_ctx_f16 = nullptr;
            return false;
        }
        ggml_set_name(d, ggml_get_name(s));
        q2f16[s] = d;
    }

    c->fm_b2_buf_f16 = ggml_backend_alloc_ctx_tensors(c->fm_b2_ctx_f16, c->backend);
    if (!c->fm_b2_buf_f16) {
        ggml_free(c->fm_b2_ctx_f16);
        c->fm_b2_ctx_f16 = nullptr;
        return false;
    }

    size_t bytes_q = 0, bytes_f16 = 0;
    std::vector<char> raw;
    std::vector<float> f32;
    std::vector<ggml_fp16_t> f16;
    for (const auto& kv : q2f16) {
        ggml_tensor* s = kv.first;
        ggml_tensor* d = kv.second;
        const auto* tt = ggml_get_type_traits(s->type);
        if (!tt || !tt->to_float)
            return false;
        const int64_t n = ggml_nelements(s);
        raw.resize(ggml_nbytes(s));
        ggml_backend_tensor_get(s, raw.data(), 0, ggml_nbytes(s));
        f32.resize((size_t)n);
        tt->to_float(raw.data(), f32.data(), n);
        f16.resize((size_t)n);
        ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
        ggml_backend_tensor_set(d, f16.data(), 0, (size_t)n * sizeof(ggml_fp16_t));
        bytes_q += ggml_nbytes(s);
        bytes_f16 += ggml_nbytes(d);
    }

    auto repl = [&](ggml_tensor* w) -> ggml_tensor* {
        auto it = q2f16.find(w);
        return it != q2f16.end() ? it->second : w;
    };

    c->fm_b2_f16 = c->fm;
    c->fm_b2_f16.noisy_proj_w = repl(c->fm.noisy_proj_w);
    c->fm_b2_f16.cond_proj_w = repl(c->fm.cond_proj_w);
    c->fm_b2_f16.t_emb_mlp0_w = repl(c->fm.t_emb_mlp0_w);
    c->fm_b2_f16.t_emb_mlp1_w = repl(c->fm.t_emb_mlp1_w);
    c->fm_b2_f16.layers.resize(c->fm.layers.size());
    for (size_t i = 0; i < c->fm.layers.size(); i++) {
        tada_fm_layer f = c->fm.layers[i];
        f.ffn_gate_w = repl(f.ffn_gate_w);
        f.ffn_up_w = repl(f.ffn_up_w);
        f.ffn_down_w = repl(f.ffn_down_w);
        f.adaln_w = repl(f.adaln_w);
        c->fm_b2_f16.layers[i] = f;
    }
    c->fm_b2_f16.final_proj_w = repl(c->fm.final_proj_w);
    c->fm_b2_f16.final_adaln_w = repl(c->fm.final_adaln_w);

    if (c->params.verbosity >= 1) {
        fprintf(stderr,
                "tada: FM B=2 on GPU - dequantized %zu FM matmul weights to F16 GPU-resident "
                "(%.0f -> %.0f MiB; correct f16 batched path)\n",
                q2f16.size(), bytes_q / 1048576.0, bytes_f16 / 1048576.0);
    }
    return true;
}

// ── FM B=2 batched CFG helpers ────────────────────────────────────────────

static ggml_backend_sched_t tada_fm_b2_sched_lazy(tada_context* c) {
    if (c->fm_b2_sched)
        return c->fm_b2_sched;
    ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
    int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
    c->fm_b2_sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
    return c->fm_b2_sched;
}

static ggml_cgraph* tada_get_or_build_fm_b2(tada_context* c) {
    if (c->fm_b2_gf)
        return c->fm_b2_gf;
    c->fm_b2_meta.assign(c->compute_meta.size() * 2, 0);
    ggml_init_params ip = {c->fm_b2_meta.size(), c->fm_b2_meta.data(), true};
    c->fm_b2_ctx = ggml_init(ip);
    if (!c->fm_b2_ctx)
        return nullptr;
    c->fm_b2_gf = build_graph_fm_step_b2(c, c->fm_b2_ctx);
    return c->fm_b2_gf;
}

// Run one B=2 FM step: pos+neg cond in one batched forward.
// Returns false on failure (falls back to sequential run_fm_step calls).
static bool run_fm_step_b2(tada_context* c, const float* noisy_z, float timestep, const float* cond_pos,
                           const float* cond_neg, float* vel_pos_out, float* vel_neg_out) {
    const int lat = (int)c->hp.fm_latent;
    const int hid = (int)c->hp.fm_hidden;

    float t_emb[256];
    sinusoidal_embedding(timestep, 256, t_emb);

    ggml_cgraph* gf = tada_get_or_build_fm_b2(c);
    if (!gf)
        return false;

    // Pack pos+neg as column-interleaved (hid, 2) in column-major order:
    // col 0 = cond_pos, col 1 = cond_neg
    std::vector<float> cond_b2((size_t)hid * 2);
    for (int j = 0; j < hid; j++) {
        cond_b2[j] = cond_pos[j];       // col 0
        cond_b2[hid + j] = cond_neg[j]; // col 1
    }

    // Vulkan native path: direct compute on the backend (no sched split) — see run_fm_step (#192).
    if (c->vulkan_native) {
        if (!c->fm_b2_galloc)
            c->fm_b2_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        if (!ggml_gallocr_alloc_graph(c->fm_b2_galloc, gf))
            return false;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0, (size_t)lat * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond_b2"), cond_b2.data(), 0,
                                cond_b2.size() * sizeof(float));
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS)
            return false;
    } else {
        ggml_backend_sched_t sched = tada_fm_b2_sched_lazy(c);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
            return false;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "noisy_z"), noisy_z, 0, (size_t)lat * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb_sin"), t_emb, 0, 256 * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "fm_cond_b2"), cond_b2.data(), 0,
                                cond_b2.size() * sizeof(float));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
            return false;
    }

    ggml_tensor* vel = ggml_graph_get_tensor(gf, "velocity_b2");
    std::vector<float> vel_b2((size_t)lat * 2);
    ggml_backend_tensor_get(vel, vel_b2.data(), 0, vel_b2.size() * sizeof(float));

    // Unpack: col 0 = vel_pos, col 1 = vel_neg
    for (int j = 0; j < lat; j++) {
        vel_pos_out[j] = vel_b2[j];
        vel_neg_out[j] = vel_b2[lat + j];
    }
    return true;
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
            // B=2 path: run pos+neg in one batched forward when available.
            bool used_b2 =
                c->fm_b2_active && run_fm_step_b2(c, speech, t_val, cond, neg, vel_pos.data(), vel_neg.data());
            if (!used_b2) {
                run_fm_step(c, speech, t_val, cond, vel_pos.data());
                run_fm_step(c, speech, t_val, neg, vel_neg.data());
            }

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

// Score an FM candidate by reconstruction loss — mirrors Python
// _score_by_reconstruction (scorer="likelihood", the InferenceOptions default).
// Evaluates the conditional velocity (CFG=1.0) at 3 timesteps along the
// straight-line OT path x_t=(1-t)*noise+t*sample and compares it to the target
// OT velocity (sample-noise) on the acoustic dims only. Returns the negative
// mean MSE (higher = more on-distribution = better candidate). Eval points are
// concentrated in [0.3,0.9] where the model is most discriminative.
static float fm_score_reconstruction(tada_context* c, const float* sample, const float* noise, const float* cond) {
    const int lat = (int)c->hp.fm_latent;
    const int ad = (int)c->hp.acoustic_dim;
    static const float t_pts[3] = {0.3f, 0.6f, 0.9f};
    std::vector<float> x_t(lat), pred_v(lat);
    double total_err = 0.0;
    for (int p = 0; p < 3; p++) {
        const float t = t_pts[p];
        for (int j = 0; j < lat; j++)
            x_t[j] = (1.0f - t) * noise[j] + t * sample[j];
        run_fm_step(c, x_t.data(), t, cond, pred_v.data());
        double err = 0.0;
        for (int j = 0; j < ad; j++) {
            const float d = pred_v[j] - (sample[j] - noise[j]);
            err += (double)d * d;
        }
        total_err += err / ad;
    }
    return (float)(-total_err / 3.0);
}

// Batched candidate ranking — same result as N sequential (fm_euler_solve +
// fm_score_reconstruction) calls, but all N candidates × CFG are solved in one
// batch-2N forward per ODE step and scored in 3 batch-N forwards. This is the
// fast path for num_acoustic_candidates>1: it turns ~13N small FM forwards per
// AR step into 10+3 batched forwards (mirrors Python's batched ranked solve).
// `all_noise` holds N pre-drawn, pre-scaled noise vectors (N*lat). On success
// the best candidate (by reconstruction score) is written to `best_out` (lat).
static bool fm_solve_rank_candidates(tada_context* c, const float* all_noise, int N, const float* cond_pos,
                                     const float* cond_neg, int num_steps, float cfg_scale, float* best_out) {
    const int lat = (int)c->hp.fm_latent;
    const int ad = (int)c->hp.acoustic_dim;
    const int hid = (int)c->hp.fm_hidden;
    if (N < 1)
        return false;

    std::vector<float> zero_cond(hid, 0.0f);
    const float* neg = cond_neg ? cond_neg : zero_cond.data();

    std::vector<float> t_span;
    build_logsnr_schedule(t_span, num_steps);

    // States for all candidates (start from the noise), contiguous N*lat.
    std::vector<float> states((size_t)N * lat);
    std::copy(all_noise, all_noise + (size_t)N * lat, states.begin());

    // Solve: batch = 2N. Columns [0,N)=pos path, [N,2N)=neg path; both use the
    // same per-candidate state, paired with cond_pos / cond_neg respectively.
    const int Bsolve = 2 * N;
    std::vector<float> noisy_b((size_t)lat * Bsolve);
    std::vector<float> cond_b((size_t)hid * Bsolve);
    std::vector<float> vel_b((size_t)lat * Bsolve);
    // cond columns are constant across ODE steps — fill once.
    for (int cnd = 0; cnd < N; cnd++) {
        std::copy(cond_pos, cond_pos + hid, cond_b.begin() + (size_t)cnd * hid);
        std::copy(neg, neg + hid, cond_b.begin() + (size_t)(N + cnd) * hid);
    }

    for (int i = 0; i < num_steps; i++) {
        const float dt = t_span[i + 1] - t_span[i];
        const float t_val = t_span[i];
        const float a_cfg = scheduled_cfg(cfg_scale, t_val, "cosine");

        for (int cnd = 0; cnd < N; cnd++) {
            const float* s = states.data() + (size_t)cnd * lat;
            std::copy(s, s + lat, noisy_b.begin() + (size_t)cnd * lat);       // pos col
            std::copy(s, s + lat, noisy_b.begin() + (size_t)(N + cnd) * lat); // neg col
        }
        if (!run_fm_step_batch(c, noisy_b.data(), t_val, cond_b.data(), vel_b.data(), Bsolve))
            return false;

        for (int cnd = 0; cnd < N; cnd++) {
            float* s = states.data() + (size_t)cnd * lat;
            const float* vp = vel_b.data() + (size_t)cnd * lat;
            const float* vn = vel_b.data() + (size_t)(N + cnd) * lat;
            for (int j = 0; j < ad; j++) // acoustic dims: acoustic CFG
                s[j] += dt * (vn[j] + a_cfg * (vp[j] - vn[j]));
            for (int j = ad; j < lat; j++) // time dims: duration CFG = 1.0
                s[j] += dt * vp[j];
        }
    }

    // Score: for each of 3 eval timesteps, one batch-N forward (cond=cond_pos).
    static const float t_pts[3] = {0.3f, 0.6f, 0.9f};
    std::vector<double> err(N, 0.0);
    std::vector<float> sc_noisy((size_t)lat * N);
    std::vector<float> sc_cond((size_t)hid * N);
    std::vector<float> sc_vel((size_t)lat * N);
    for (int cnd = 0; cnd < N; cnd++)
        std::copy(cond_pos, cond_pos + hid, sc_cond.begin() + (size_t)cnd * hid);
    for (int p = 0; p < 3; p++) {
        const float t = t_pts[p];
        for (int cnd = 0; cnd < N; cnd++) {
            const float* nz = all_noise + (size_t)cnd * lat;
            const float* sm = states.data() + (size_t)cnd * lat;
            float* xt = sc_noisy.data() + (size_t)cnd * lat;
            for (int j = 0; j < lat; j++)
                xt[j] = (1.0f - t) * nz[j] + t * sm[j];
        }
        if (!run_fm_step_batch(c, sc_noisy.data(), t, sc_cond.data(), sc_vel.data(), N))
            return false;
        for (int cnd = 0; cnd < N; cnd++) {
            const float* nz = all_noise + (size_t)cnd * lat;
            const float* sm = states.data() + (size_t)cnd * lat;
            const float* pv = sc_vel.data() + (size_t)cnd * lat;
            double e = 0.0;
            for (int j = 0; j < ad; j++) {
                const float d = pv[j] - (sm[j] - nz[j]);
                e += (double)d * d;
            }
            err[cnd] += e / ad;
        }
    }

    int best = 0;
    double best_err = err[0];
    for (int cnd = 1; cnd < N; cnd++) {
        if (err[cnd] < best_err) {
            best_err = err[cnd];
            best = cnd;
        }
    }
    std::copy(states.data() + (size_t)best * lat, states.data() + (size_t)(best + 1) * lat, best_out);
    return true;
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

// Talker next-token selection, mirroring upstream tada.py `_generate`:
//   suppress pad → [repetition penalty → temperature → top-k → top-p →
//   multinomial] when text_do_sample, else greedy argmax. `logits` is
//   modified in place. `history` is the running id sequence (for rep penalty).
static int tada_sample_token(const tada_context_params& p, float* logits, int n_vocab,
                             const std::vector<int32_t>& history, int pad_id, std::mt19937& rng) {
    if (pad_id >= 0 && pad_id < n_vocab)
        logits[pad_id] = -INFINITY; // upstream: prevent pad from being generated

    const bool do_sample = p.text_do_sample && p.temperature > 0.0f;
    if (!do_sample)
        return argmax_logits(logits, n_vocab);

    // 1. Repetition penalty over ids already in the sequence (once per unique
    //    id; upstream gather/scatter is idempotent across duplicates).
    if (p.text_repetition_penalty != 1.0f) {
        std::unordered_set<int32_t> seen;
        seen.reserve(history.size());
        for (int32_t id : history) {
            if (id < 0 || id >= n_vocab || !seen.insert(id).second)
                continue;
            float s = logits[id];
            logits[id] = (s < 0.0f) ? s * p.text_repetition_penalty : s / p.text_repetition_penalty;
        }
    }

    // 2. Temperature.
    const float inv_t = 1.0f / p.temperature;
    for (int i = 0; i < n_vocab; i++)
        logits[i] *= inv_t;

    // 3. Top-k: keep only the k largest logits.
    if (p.text_top_k > 0 && p.text_top_k < n_vocab) {
        std::vector<float> tmp(logits, logits + n_vocab);
        std::nth_element(tmp.begin(), tmp.begin() + (n_vocab - p.text_top_k), tmp.end());
        const float kth = tmp[n_vocab - p.text_top_k];
        for (int i = 0; i < n_vocab; i++)
            if (logits[i] < kth)
                logits[i] = -INFINITY;
    }

    // 4. Top-p (nucleus): sort desc, remove tokens once the cumulative prob of
    //    the strictly-higher-ranked tokens already reaches top_p (upstream's
    //    shift-right keeps the first token that crosses the threshold).
    if (p.text_top_p > 0.0f && p.text_top_p < 1.0f) {
        std::vector<int> order(n_vocab);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        const float maxl = logits[order[0]];
        double Z = 0.0;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[order[i]] == -INFINITY)
                break;
            Z += std::exp((double)logits[order[i]] - maxl);
        }
        double cum_before = 0.0;
        for (int i = 0; i < n_vocab; i++) {
            const int id = order[i];
            if (logits[id] == -INFINITY)
                break;
            const double pr = std::exp((double)logits[id] - maxl) / Z;
            if (cum_before >= (double)p.text_top_p)
                logits[id] = -INFINITY;
            cum_before += pr;
        }
    }

    // 5. Softmax over survivors + multinomial sample.
    float maxl = -INFINITY;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] > maxl)
            maxl = logits[i];
    double Z = 0.0;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] != -INFINITY)
            Z += std::exp((double)logits[i] - maxl);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng) * Z;
    double acc = 0.0;
    int last = -1;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] == -INFINITY)
            continue;
        last = i;
        acc += std::exp((double)logits[i] - maxl);
        if (acc >= r)
            return i;
    }
    return last >= 0 ? last : argmax_logits(logits, n_vocab);
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
    p.temperature = 0.6f; // upstream text_temperature (used only when text_do_sample)
    p.seed = 42;
    p.max_tokens = 0;
    p.flash_attn = false;
    p.num_fm_steps = 0;
    p.acoustic_cfg = 1.6f;         // match Python InferenceOptions default
    p.noise_temp = 0.9f;           // match Python InferenceOptions default
    p.num_acoustic_candidates = 1; // match Python InferenceOptions default
    // Talker text-decoder sampling. Upstream InferenceOptions defaults are
    // do_sample=True/top_p=0.9/top_k=0/rep_penalty=1.1, but the LIBRARY default
    // stays greedy (text_do_sample=false) so crispasr-diff is deterministic;
    // the CLI adapter and C ABI enable sampling with the upstream values.
    p.text_do_sample = false;
    p.text_top_p = 0.9f;
    p.text_top_k = 0;
    p.text_repetition_penalty = 1.1f;
    return p;
}

struct tada_context* tada_init_from_file(const char* path_model, struct tada_context_params params) {
    auto* c = new tada_context();
    c->params = params;
    c->rng_state = params.seed ? params.seed : 42;
    mt19937_init(&c->mt_rng, (uint32_t)(params.seed ? params.seed : 42));
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
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "tada: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, params.n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend)
        c->backend = c->backend_cpu;

    // On Vulkan, TADA's default scheduler path miscomputes: ggml_backend_sched
    // splits the FM-head and talker graphs across GPU/CPU (because GGML_OP_REPEAT
    // — GQA head expansion + batched FM — has no Vulkan pipeline on some drivers)
    // and corrupts the cross-backend activation copies. The FM solver then
    // produces NaN / exploded values, which poison the autoregressive acoustic
    // feedback so the talker never emits EOS; generation runs to the token cap and
    // the garbage durations expand to tens of thousands of frames (one runaway
    // tried to allocate 113 GB and segfaulted). Issue #192; reproduces on both
    // RADV (POLARIS10) and MoltenVK.
    //
    // Two opt-in escapes from the default CPU fallback:
    //   CRISPASR_TADA_VULKAN_NATIVE=1 — keep Vulkan and compute every TADA graph
    //     directly on the backend (no scheduler, no cross-backend split). This is
    //     the real native-Vulkan fix; the FM head is validated correct this way
    //     (speech_rms 426 -> 1.14 on MoltenVK), but the talker path needs a driver
    //     with a REPEAT pipeline (RADV has it; MoltenVK aborts), so it is gated for
    //     validation on real hardware.
    //   CRISPASR_TADA_ALLOW_VULKAN=1 — keep Vulkan on the (broken) scheduler path,
    //     for debugging only.
    // Metal and CUDA are validated and unaffected by either flag.
    if (c->backend != c->backend_cpu) {
        const char* bname = ggml_backend_name(c->backend);
        const bool is_vulkan = bname && strstr(bname, "Vulkan");
        const char* native = std::getenv("CRISPASR_TADA_VULKAN_NATIVE");
        const char* allow = std::getenv("CRISPASR_TADA_ALLOW_VULKAN");
        const bool want_native = native && native[0] == '1';
        const bool want_allow = allow && allow[0] == '1';
        if (is_vulkan && want_native) {
            c->backend_is_vulkan = true;
            c->vulkan_native = true;
            if (params.verbosity >= 1)
                fprintf(stderr, "tada: Vulkan native path ENABLED (CRISPASR_TADA_VULKAN_NATIVE=1) — "
                                "computing all graphs directly on the backend (#192).\n");
        } else if (is_vulkan && !want_allow) {
            fprintf(stderr,
                    "tada: WARNING: GPU backend '%s' is not yet supported for TADA — its scheduler path "
                    "miscomputes on Vulkan and produces no usable audio (issue #192). Falling back to CPU. "
                    "Use Metal/CUDA for GPU acceleration, set CRISPASR_TADA_VULKAN_NATIVE=1 to try the "
                    "native direct-compute path, or CRISPASR_TADA_ALLOW_VULKAN=1 to force the broken path.\n",
                    bname);
            ggml_backend_free(c->backend);
            c->backend = c->backend_cpu;
        } else if (is_vulkan) {
            c->backend_is_vulkan = true; // ALLOW_VULKAN: broken sched path, debugging only
        }
    }
    if (params.verbosity >= 1) {
        fprintf(stderr, "tada: backend=%s%s\n", ggml_backend_name(c->backend),
                (c->backend == c->backend_cpu) ? " (CPU)" : " + CPU fallback");
    }

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
    ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
    int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
    c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);

    // ── KV cache ──
    int max_ctx = params.max_tokens > 0 ? params.max_tokens + 256 : 1024;
    if (!kv_init(c, max_ctx)) {
        delete c;
        return nullptr;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "tada: loaded OK, KV cache for %d tokens\n", max_ctx);
    }

    // FM B=2 batched CFG: enabled by default. On GPU with quantized FM weights,
    // prefer F16 GPU-resident dequant copies; CRISPASR_TADA_FM_B2=1 forces the
    // native quantized batched path if F16 setup fails.
    {
        const char* env = std::getenv("CRISPASR_TADA_FM_B2");
        bool want = true; // default ON
        bool force_on = false;
        if (env && (env[0] == '0' || env[0] == 'n' || env[0] == 'N'))
            want = false;
        else if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
            want = true;
            force_on = true;
        }

        if (want) {
            // On GPU with quantized FM weights: B=2 batched mul_mat may diverge.
            // Route through F16 copies matching the other Metal workarounds.
            // Scan the whole FM head; some weights (e.g. noisy_proj with 528
            // columns) are not Q4_K-compatible and may stay F16 even in an
            // otherwise quantized FM head.
            bool fm_is_quant = false;
            for (const auto& kv : c->tensors) {
                if (kv.first.find("tada.fm_head.") == 0 && kv.second && ggml_is_quantized(kv.second->type)) {
                    fm_is_quant = true;
                    break;
                }
            }
            if (fm_is_quant && c->backend != c->backend_cpu) {
                if (!ensure_fm_b2_f16_weights(c)) {
                    if (force_on) {
                        if (params.verbosity >= 1)
                            fprintf(stderr, "tada: FM B=2 F16 dequant failed; forcing native GPU+quant batched path\n");
                    } else {
                        if (params.verbosity >= 1)
                            fprintf(stderr, "tada: FM B=2 disabled (GPU+quant weights; F16 dequant failed; set "
                                            "CRISPASR_TADA_FM_B2=1 to force native)\n");
                        want = false;
                    }
                }
            }
        }
        c->fm_b2_active = want;
        if (want && params.verbosity >= 1)
            fprintf(stderr, "tada: FM B=2 batched CFG ENABLED\n");
    }

    return c;
}

int tada_set_codec_path(struct tada_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->codec_path = path;
    const bool share_gpu_backend =
        ctx->params.use_gpu && ctx->backend && ctx->backend_cpu && ctx->backend != ctx->backend_cpu;
    ctx->codec_ctx = share_gpu_backend ? tada_codec_init_from_file_with_backend(path, ctx->params.n_threads,
                                                                                ctx->backend, ctx->backend_cpu)
                                       : tada_codec_init_from_file(path, ctx->params.n_threads);
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
    ctx->prompt_text = core_gguf::kv_str(meta, "crispasr.ref.tada_tts_prompt_text", "");
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
            int p_prev = (i > 0) ? (int)pos[i - 1] : 1;
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
    if (ctx) {
        ctx->rng_state = seed ? seed : 42;
        mt19937_init(&ctx->mt_rng, (uint32_t)(seed ? seed : 42));
    }
}

void tada_set_temperature(struct tada_context* ctx, float temp) {
    if (ctx)
        ctx->params.temperature = temp;
}
void tada_set_num_candidates(struct tada_context* ctx, int n) {
    if (ctx)
        ctx->params.num_acoustic_candidates = n > 0 ? n : 1;
}
void tada_set_do_sample(struct tada_context* ctx, bool enable) {
    if (ctx)
        ctx->params.text_do_sample = enable;
}
void tada_set_top_p(struct tada_context* ctx, float top_p) {
    if (ctx)
        ctx->params.text_top_p = top_p;
}
void tada_set_top_k(struct tada_context* ctx, int top_k) {
    if (ctx)
        ctx->params.text_top_k = top_k > 0 ? top_k : 0;
}
void tada_set_repetition_penalty(struct tada_context* ctx, float penalty) {
    if (ctx)
        ctx->params.text_repetition_penalty = penalty > 0.0f ? penalty : 1.0f;
}
void tada_set_num_fm_steps(struct tada_context* ctx, int steps) {
    // Flow-matching ODE steps (Python num_flow_matching_steps, default 10). More
    // steps = slower but more accurate acoustics; <=0 restores the default.
    if (ctx)
        ctx->params.num_fm_steps = steps > 0 ? steps : 0;
}
void tada_set_acoustic_cfg(struct tada_context* ctx, float cfg) {
    // Acoustic classifier-free-guidance scale (Python acoustic_cfg, default 1.6).
    if (ctx && cfg >= 0.0f)
        ctx->params.acoustic_cfg = cfg;
}
void tada_set_noise_temp(struct tada_context* ctx, float temp) {
    // FM noise temperature (Python noise_temp, default 0.9).
    if (ctx && temp >= 0.0f)
        ctx->params.noise_temp = temp;
}

static std::string tada_normalize_text(std::string text) {
    auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
        if (from.empty())
            return;
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };

    replace_all(text, "; ", ". ");
    replace_all(text, "\"", "");
    replace_all(text, ":", ",");
    replace_all(text, "(", "");
    replace_all(text, ")", "");
    replace_all(text, "--", "-");
    replace_all(text, "-", ", ");
    replace_all(text, ",,", ",");
    replace_all(text, " '", " ");
    replace_all(text, "' ", " ");
    replace_all(text, "  ", " ");

    std::string no_space_before_punct;
    no_space_before_punct.reserve(text.size());
    for (size_t i = 0; i < text.size(); i++) {
        if (std::isspace((unsigned char)text[i]) && i + 1 < text.size() &&
            (text[i + 1] == '.' || text[i + 1] == ',' || text[i + 1] == '?' || text[i + 1] == '!')) {
            continue;
        }
        no_space_before_punct.push_back(text[i]);
    }
    text.swap(no_space_before_punct);

    for (char& ch : text)
        ch = (char)std::tolower((unsigned char)ch);

    bool cap_next = true;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char ch = (unsigned char)text[i];
        if (cap_next && std::isalnum(ch)) {
            text[i] = (char)std::toupper(ch);
            cap_next = false;
        }
        if ((text[i] == '.' || text[i] == '!' || text[i] == '?') && i + 1 < text.size() &&
            std::isspace((unsigned char)text[i + 1])) {
            cap_next = true;
        }
    }
    return text;
}

static void tada_build_input_ids_for_text(tada_context* ctx, const char* text, std::vector<int32_t>& full_ids,
                                          std::vector<int32_t>* text_ids_out = nullptr) {
    std::vector<int32_t> text_ids = tokenize(ctx, tada_normalize_text(std::string(text ? text : "")));
    auto lookup = [&](const char* name, int32_t fallback) -> int32_t {
        auto it = ctx->vocab.token_to_id.find(name);
        return (it != ctx->vocab.token_to_id.end()) ? it->second : fallback;
    };
    int32_t bos = lookup("<|begin_of_text|>", 128000);
    int32_t eot = lookup("<|eot_id|>", 128009);
    int32_t start_header = lookup("<|start_header_id|>", 128006);
    int32_t end_header = lookup("<|end_header_id|>", 128007);

    std::vector<int32_t> system_text_ids = tokenize(ctx, std::string("system"));
    std::vector<int32_t> assistant_text_ids = tokenize(ctx, std::string("assistant"));
    std::vector<int32_t> prefix_ids;
    prefix_ids.push_back(start_header);
    prefix_ids.insert(prefix_ids.end(), system_text_ids.begin(), system_text_ids.end());
    prefix_ids.push_back(end_header);
    prefix_ids.push_back(eot);
    prefix_ids.push_back(start_header);
    prefix_ids.insert(prefix_ids.end(), assistant_text_ids.begin(), assistant_text_ids.end());
    prefix_ids.push_back(end_header);

    full_ids.clear();
    full_ids.push_back(bos);
    full_ids.insert(full_ids.end(), prefix_ids.begin(), prefix_ids.end());
    full_ids.insert(full_ids.end(), text_ids.begin(), text_ids.end());
    for (uint32_t i = 0; i < ctx->hp.shift_acoustic; i++)
        full_ids.push_back(eot);
    if (text_ids_out)
        *text_ids_out = std::move(text_ids);
}

float* tada_extract_stage(struct tada_context* ctx, const char* text, const char* stage, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !stage || !*stage)
        return nullptr;

    std::vector<int32_t> full_ids;
    std::vector<int32_t> text_ids;
    tada_build_input_ids_for_text(ctx, text, full_ids, &text_ids);
    if (strcmp(stage, "text_tokens") == 0) {
        float* out = (float*)malloc(text_ids.size() * sizeof(float));
        if (!out)
            return nullptr;
        for (size_t i = 0; i < text_ids.size(); i++)
            out[i] = (float)text_ids[i];
        if (out_n)
            *out_n = (int)text_ids.size();
        return out;
    }
    if (full_ids.empty())
        return nullptr;

    const int ad = (int)ctx->hp.acoustic_dim;
    const int lat = (int)ctx->hp.fm_latent;
    const int hid = (int)ctx->hp.fm_hidden;
    std::vector<float> zeros(ad, 0.0f);

    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    if (ctx->kv_neg_buf)
        ggml_backend_buffer_clear(ctx->kv_neg_buf, 0);
    ctx->rng_state = ctx->params.seed ? ctx->params.seed : 42;
    mt19937_init(&ctx->mt_rng, (uint32_t)(ctx->params.seed ? ctx->params.seed : 42));

    float* emb = build_step_embedding(ctx, full_ids[0], zeros.data(), 0, 0, 0);
    if (!emb)
        return nullptr;
    if (strcmp(stage, "llm_embed") == 0) {
        if (out_n)
            *out_n = (int)ctx->hp.d_model;
        return emb;
    }

    talker_result tr = run_talker_kv(ctx, emb, 1, 0, false);
    free(emb);
    if (!tr.hidden)
        return nullptr;
    if (strcmp(stage, "llm_hidden_0") == 0) {
        if (out_n)
            *out_n = hid;
        return tr.hidden;
    }

    std::vector<float> speech(lat);
    mt19937_randn(&ctx->mt_rng, speech.data(), lat);
    tada_scale_noise_like_python(speech.data(), lat, ctx->params.noise_temp);
    if (strcmp(stage, "fm_noise_0") == 0) {
        float* out = (float*)malloc((size_t)lat * sizeof(float));
        if (!out) {
            free(tr.hidden);
            return nullptr;
        }
        memcpy(out, speech.data(), (size_t)lat * sizeof(float));
        if (out_n)
            *out_n = lat;
        free(tr.hidden);
        return out;
    }

    std::vector<float> vel(lat);
    if (strcmp(stage, "fm_step_0_0") == 0 || strcmp(stage, "fm_step_0_5") == 0) {
        float t = strcmp(stage, "fm_step_0_5") == 0 ? 0.5f : 0.0f;
        run_fm_step(ctx, speech.data(), t, tr.hidden, vel.data());
        float* out = (float*)malloc((size_t)lat * sizeof(float));
        if (!out) {
            free(tr.hidden);
            return nullptr;
        }
        memcpy(out, vel.data(), (size_t)lat * sizeof(float));
        if (out_n)
            *out_n = lat;
        free(tr.hidden);
        return out;
    }
    if (strcmp(stage, "fm_output_0") == 0) {
        fm_euler_solve(ctx, speech.data(), tr.hidden, ctx->params.num_fm_steps > 0 ? ctx->params.num_fm_steps : 10,
                       ctx->params.acoustic_cfg, nullptr, false);
        float* out = (float*)malloc((size_t)lat * sizeof(float));
        if (!out) {
            free(tr.hidden);
            return nullptr;
        }
        memcpy(out, speech.data(), (size_t)lat * sizeof(float));
        if (out_n)
            *out_n = lat;
        free(tr.hidden);
        return out;
    }

    free(tr.hidden);
    return nullptr;
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

    // Reset RNG (both legacy xorshift and PyTorch-compatible MT19937)
    ctx->rng_state = ctx->params.seed ? ctx->params.seed : 42;
    mt19937_init(&ctx->mt_rng, (uint32_t)(ctx->params.seed ? ctx->params.seed : 42));
    // Separate RNG stream for talker text-token sampling so it stays
    // independent of the FM-noise MT19937 stream (which is re-seeded per gen).
    std::mt19937 sampler_rng((uint32_t)(ctx->params.seed ? ctx->params.seed : 42));

    tada_bench_stage _bs_synth("synthesize");

    // ── Tokenize ──
    std::string norm_text = tada_normalize_text(std::string(text));
    std::vector<int32_t> text_ids = tokenize(ctx, norm_text);
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
        prompt_text_ids = tokenize(ctx, tada_normalize_text(std::string(prompt_text_env)));
    } else if (!ctx->prompt_text.empty() && ctx->n_prompt > 0) {
        prompt_text_ids = tokenize(ctx, tada_normalize_text(ctx->prompt_text));
    }

    // Full sequence: BOS + prefix + [voice_prompt_pads] + prompt_text + synth_text + EOT*shift
    // Python includes n_prompt PAD slots for the voice replay phase. Without a
    // voice-prompt transcript (prompt_text_ids), those slots are all PAD tokens.
    // Omitting them makes num_prompt too small: with 26 voice tokens, the batch
    // prefill consumes all 19 input tokens and the AR generation loop runs 0
    // steps, producing nothing beyond the pre-filled voice-replay frames.
    const int32_t kPadId = 128004; // <|finetune_right_pad_id|>
    const int n_voice_pad = (ctx->n_prompt > 0 && prompt_text_ids.empty()) ? ctx->n_prompt : 0;
    std::vector<int32_t> full_ids;
    full_ids.push_back(bos);
    full_ids.insert(full_ids.end(), prefix_ids.begin(), prefix_ids.end());
    for (int i = 0; i < n_voice_pad; i++)
        full_ids.push_back(kPadId);
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

    // Mirror tada.modules.tada.generate(): prompt acoustic features are
    // left-padded by the chat prefix and then the final transition frames are
    // removed before _generate() sees them.
    const int transition_steps = shift;
    const int prompt_padded_len = prefix_len + ctx->n_prompt;
    const int prompt_used_len = std::max(0, prompt_padded_len - transition_steps);
    // Cap at num_prompt-1 so the main AR loop always runs at least one step with
    // logits. When prompt_used_len is large (long voice prompt), the uncapped
    // formula equals num_prompt and the main loop would run 0 iterations —
    // leaving no room for the generation to continue past the fixed input tokens.
    const int prefill_len = prompt_used_len > 0 ? std::min(num_prompt - 1, transition_steps + prompt_used_len - 1) : 0;

    std::vector<std::vector<float>> acoustic_features;
    std::vector<int> time_before_list;
    std::vector<tada_fm_dump_record> fm_dump_records;
    const bool dump_fm_steps = []() {
        const char* path = std::getenv("TADA_DUMP_FM_STEPS");
        return path && path[0];
    }();

    // State
    std::vector<float> cur_acoustic(ad, 0.0f);
    int32_t cur_mask = 0;
    int32_t cur_t_before = 0;
    int32_t cur_t_after = 0;
    int n_past = 0;

    int32_t pad_id = 128004; // Llama <|finetune_right_pad_id|>
    auto pad_it = ctx->vocab.token_to_id.find("<|finetune_right_pad_id|>");
    if (pad_it != ctx->vocab.token_to_id.end())
        pad_id = pad_it->second;

    // Python generate(use_text_in_prompt=False) masks text content in the
    // prompt-acoustic region on the positive path too. Keep chat structure and
    // header names, but hide prompt transcript content behind the pad token.
    std::vector<int32_t> model_ids = full_ids;
    bool in_header = false;
    for (int i = 0; i < prompt_used_len && i < (int)model_ids.size(); ++i) {
        const int32_t tok = model_ids[i];
        bool keep = false;
        if (tok == start_header) {
            in_header = true;
            keep = true;
        } else if (tok == end_header) {
            in_header = false;
            keep = true;
        } else if (in_header || tok == eot || tok == bos) {
            keep = true;
        }
        if (!keep)
            model_ids[i] = pad_id;
    }

    // Extend model_ids as we generate
    std::vector<int32_t> all_ids = model_ids;

    // ── Batched prefill ──
    // During prefill (steps 0..prefill_len-1) the FM is never needed and
    // pos/neg token embeddings are identical (all tokens are structural or
    // pad-masked content), so we can:
    //   1. Pre-build all prefill embeddings without running Llama,
    //   2. Run a single batched Llama forward (T = prefill_len),
    //   3. memcpy pos KV → neg KV instead of a second Llama pass.
    // This replaces 2×prefill_len separate T=1 graph calls with 1 T=N call.
    // Disable with CRISPASR_TADA_BATCH_PREFILL=0.
    static const bool s_batch_prefill_env = []() {
        const char* e = std::getenv("CRISPASR_TADA_BATCH_PREFILL");
        return !(e && *e == '0');
    }();
    const bool do_batch_prefill = (prefill_len > 1) && s_batch_prefill_env;

    if (do_batch_prefill) {
        tada_bench_stage _bs_pf("prefill_batch");
        const int d = (int)hp.d_model;

        // Pre-build all prefill embeddings and collect acoustic features.
        // State advances through prompt data exactly as the main loop would,
        // but no FM is called (need_fm=false for all steps 0..prefill_len-1).
        std::vector<float> prefill_embs((size_t)d * prefill_len, 0.0f);
        {
            std::vector<float> pf_acoustic(ad, 0.0f);
            int32_t pf_mask = 0, pf_t_before = 0, pf_t_after = 0;

            for (int s = 0; s < prefill_len; s++) {
                float* emb =
                    build_step_embedding(ctx, all_ids[s], pf_acoustic.data(), pf_mask, pf_t_before, pf_t_after);
                if (!emb) {
                    fprintf(stderr, "tada: batched prefill embed failed at step %d\n", s);
                    return nullptr;
                }
                memcpy(prefill_embs.data() + (size_t)s * d, emb, (size_t)d * sizeof(float));
                free(emb);

                // Mirror the state update from the main loop (identical logic,
                // but without pred_t_before from FM — unused in prefill range).
                if (s < shift) {
                    std::fill(pf_acoustic.begin(), pf_acoustic.end(), 0.0f);
                    pf_mask = 0;
                    pf_t_before = 0;
                    pf_t_after = 0;
                } else {
                    int fi = s - shift;
                    bool fi_in_prefix = (fi < prefix_len && fi < prompt_used_len);
                    if (fi_in_prefix) {
                        std::vector<float> feat(ad, 0.0f);
                        acoustic_features.push_back(feat);
                        time_before_list.push_back(0);
                        pf_acoustic = feat;
                        pf_mask = 0;
                        pf_t_before = 0;
                        pf_t_after = 0;
                    } else {
                        int pfidx = fi - prefix_len;
                        bool fi_in_prompt =
                            (fi < prompt_used_len && ctx->n_prompt > 0 && pfidx >= 0 && pfidx < ctx->n_prompt);
                        if (fi_in_prompt) {
                            std::vector<float> feat(ctx->prompt_values.begin() + (size_t)pfidx * ad,
                                                    ctx->prompt_values.begin() + (size_t)(pfidx + 1) * ad);
                            acoustic_features.push_back(feat);
                            bool use_prompted_time =
                                (fi < prompt_used_len - 1) && (pfidx + 1 < (int)ctx->prompt_time_before.size());
                            int tb = use_prompted_time ? ctx->prompt_time_before[pfidx + 1] : 0;
                            int ta = use_prompted_time ? (pfidx + 1 < (int)ctx->prompt_time_after.size()
                                                              ? ctx->prompt_time_after[pfidx + 1]
                                                              : 0)
                                                       : 0;
                            time_before_list.push_back(tb);
                            pf_acoustic = feat;
                            pf_mask = 1;
                            pf_t_before = tb;
                            pf_t_after = ta;
                        }
                    }
                }
            }

            // Hand the final state to the main loop (used by step prefill_len).
            cur_acoustic = pf_acoustic;
            cur_mask = pf_mask;
            cur_t_before = pf_t_before;
            cur_t_after = pf_t_after;
        }

        // Single batched Llama forward for all prefill tokens.
        {
            talker_result pf_tr = run_talker_kv(ctx, prefill_embs.data(), prefill_len, 0, false);
            free(pf_tr.hidden);
            free(pf_tr.logits);
        }
        n_past = prefill_len;

        // Copy pos KV → neg KV: embeddings are identical during prefill
        // (all tokens are structural or pad), so the KV caches match.
        if (ctx->kv_neg_k) {
            ggml_backend_tensor_copy(ctx->kv_k, ctx->kv_neg_k);
            ggml_backend_tensor_copy(ctx->kv_v, ctx->kv_neg_v);
        }

        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "tada: batched prefill %d tokens, neg KV copied\n", prefill_len);
    }

    // ── AR + FM generation loop ──
    // Starts at prefill_len when batched prefill ran, otherwise at 0.
    const int loop_start = do_batch_prefill ? prefill_len : 0;

    // Main AR loop: runs until EOS or max_tokens.
    // After the batched prefill, the loop starts at prefill_len and continues
    // until EOS is predicted. The initial all_ids contains the fixed input
    // tokens; the loop extends it one token at a time as the model generates.
    bool done = false;
    for (int step = loop_start; step < (int)all_ids.size() && !done; step++) {
        if ((int)acoustic_features.size() >= max_tokens)
            break;

        int32_t cur_token = all_ids[step];
        // Compute logits starting at the last fixed-input token (num_prompt-1)
        // so the first generated token is predicted after all inputs are processed.
        // For subsequent steps (step >= num_prompt), we are in the generation phase
        // and logits are always needed (need_logits is always true since step grows).
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
            // Python batches positive and negative prefill with identical
            // masked embeddings. After prefill, negative_step_output keeps
            // chat structural tokens and replaces content with pad.
            bool is_structural = (cur_token == start_header || cur_token == end_header || cur_token == eot);
            int32_t neg_token = (step < prefill_len || is_structural) ? cur_token : pad_id;

            float* neg_emb =
                build_step_embedding(ctx, neg_token, cur_acoustic.data(), cur_mask, cur_t_before, cur_t_after);
            if (neg_emb) {
                talker_result neg_tr = run_talker_kv(ctx, neg_emb, 1, n_past, false, ctx->kv_neg_k, ctx->kv_neg_v);
                free(neg_emb);
                neg_hidden = neg_tr.hidden; // caller frees
            }

            if (ctx->params.verbosity >= 2 && (step < 3 || (step >= shift && step < shift + 3))) {
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
        // Python only runs FM during the AR phase (step >= prefill_len).  During the
        // batch prefill (steps 0..prefill_len-1) no noise is drawn, so consuming RNG
        // here would offset the entire noise sequence relative to Python.  Skip the
        // FM (and the RNG draw) when the FM output will be discarded anyway:
        //   • step < shift: structural tokens, cur_t/cur_ac forced to 0
        //   • in-prefix (feat_idx < prefix_len): cur_t/cur_ac forced to 0
        //   • in-prompt with reference time (feat_idx < prompt_used_len-1): time
        //     comes from the prompt array; FM time/acoustic both discarded
        // FM IS needed at feat_idx == prompt_used_len-1 (time transitions to
        // predicted) and for all fully-generated steps beyond that.
        bool need_fm = false;
        if (step >= shift) {
            int fi = step - shift;
            bool fi_in_prefix = (fi < prefix_len && fi < prompt_used_len);
            if (!fi_in_prefix) {
                int pfidx = fi - prefix_len;
                bool fi_in_prompt = (fi < prompt_used_len && ctx->n_prompt > 0 && pfidx >= 0 && pfidx < ctx->n_prompt);
                bool fi_use_ref_time =
                    fi_in_prompt && (fi < prompt_used_len - 1) && (pfidx + 1 < (int)ctx->prompt_time_before.size());
                need_fm = !fi_use_ref_time;
            }
        }

        std::vector<float> speech(lat, 0.0f);
        bool cand_already_solved = false;
        const int n_cand = ctx->params.num_acoustic_candidates > 1 ? ctx->params.num_acoustic_candidates : 1;
        if (need_fm && n_cand == 1) {
            // Draw noise using PyTorch-compatible MT19937 + normal_fill algorithm
            // so that noise vectors match torch.randn(lat).to(bfloat16) * temp.
            mt19937_randn(&ctx->mt_rng, speech.data(), lat);
            tada_scale_noise_like_python(speech.data(), lat, noise_temp);
        } else if (need_fm) {
            // num_acoustic_candidates>1: draw N noises in one block (matches
            // torch.randn(N*lat)), solve all N candidates × CFG in batched FM
            // forwards, and keep the best by reconstruction score (Python
            // _solve_flow_matching_ranked, scorer="likelihood"). This makes
            // timing/acoustics robust to the noise lottery — a single bad noise
            // sample can otherwise collapse durations.
            std::vector<float> all_noise((size_t)n_cand * lat);
            mt19937_randn(&ctx->mt_rng, all_noise.data(), n_cand * lat);
            tada_scale_noise_like_python(all_noise.data(), n_cand * lat, noise_temp);
            if (fm_solve_rank_candidates(ctx, all_noise.data(), n_cand, tr.hidden, neg_hidden, num_fm_steps, cfg_scale,
                                         speech.data())) {
                cand_already_solved = true;
            } else {
                // Fallback: sequential solve+score if the batched path fails.
                float best_score = -3.4e38f;
                std::vector<float> cand(lat);
                for (int ci = 0; ci < n_cand; ci++) {
                    const float* noise_c = all_noise.data() + (size_t)ci * lat;
                    std::copy(noise_c, noise_c + lat, cand.begin());
                    fm_euler_solve(ctx, cand.data(), tr.hidden, num_fm_steps, cfg_scale, neg_hidden, false);
                    const float sc = fm_score_reconstruction(ctx, cand.data(), noise_c, tr.hidden);
                    if (sc > best_score) {
                        best_score = sc;
                        speech = cand;
                    }
                }
                cand_already_solved = true;
            }
        }

        tada_fm_dump_record fm_rec;
        if (need_fm && dump_fm_steps) {
            fm_rec.step = step;
            fm_rec.feat_idx = step - shift;
            fm_rec.speech_in = speech;
            fm_rec.cond.assign(tr.hidden, tr.hidden + hp.fm_hidden);
            fm_rec.neg_cond.resize(hp.fm_hidden, 0.0f);
            if (neg_hidden)
                fm_rec.neg_cond.assign(neg_hidden, neg_hidden + hp.fm_hidden);
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

        if (need_fm && !cand_already_solved)
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

        if (ctx->params.verbosity >= 2 || (ctx->params.verbosity >= 1 && step >= shift && step < shift + 3)) {
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
        if (need_fm && dump_fm_steps) {
            fm_rec.speech_out = speech;
            fm_rec.time_bits.assign(speech.begin() + ad, speech.begin() + ad + (size_t)hp.time_dim);
            fm_dump_records.push_back(std::move(fm_rec));
        }

        // Next token prediction. Greedy by default (library); the CLI/C-ABI
        // enable upstream sampling (repetition penalty + temperature + top-k/
        // top-p). After the last fixed-input token, push generated tokens to
        // all_ids so the loop extends until EOS.
        if (need_logits && tr.logits) {
            int next = tada_sample_token(ctx->params, tr.logits, (int)hp.vocab_size, all_ids, pad_id, sampler_rng);
            if (next == eot) {
                if (ctx->params.verbosity >= 1)
                    fprintf(stderr, "tada: EOS at step %d\n", step);
                done = true; // stop after this step's acoustic state is updated
            } else {
                all_ids.push_back(next);
            }
        }
        free(tr.logits);

        // Update state for next step
        if (step >= shift) {
            int feat_idx = step - shift;
            // In Python, only prompt_acoustic_features after prefix padding and
            // transition trimming are replayed. Once feat_idx reaches
            // prompt_used_len, the FM prediction becomes the next acoustic state.
            int prompt_feat_idx = feat_idx - prefix_len; // index into actual prompt data
            bool in_prefix = (feat_idx < prefix_len && feat_idx < prompt_used_len);
            bool in_prompt = (!in_prefix && feat_idx < prompt_used_len && ctx->n_prompt > 0 && prompt_feat_idx >= 0 &&
                              prompt_feat_idx < ctx->n_prompt);

            if (in_prefix) {
                // Prefix positions: zero acoustic + zero time (matches Python's left-padding zeros)
                std::vector<float> feat(ad, 0.0f);
                acoustic_features.push_back(feat);
                time_before_list.push_back(0);
                cur_acoustic = feat;
                cur_mask = 0;
                cur_t_before = 0;
                cur_t_after = 0;
            } else if (in_prompt) {
                // Use pre-computed prompt features
                std::vector<float> feat(ctx->prompt_values.begin() + prompt_feat_idx * ad,
                                        ctx->prompt_values.begin() + (prompt_feat_idx + 1) * ad);
                acoustic_features.push_back(feat);
                // Match Python's time-transition boundary exactly:
                //   Python switches from prompted → predicted time when
                //     step - shift_acoustic >= prompt_time_len_before.shape[1] - 1
                //   which in C++ terms is feat_idx >= prompt_used_len - 1.
                // Acoustic stays prompted through feat_idx < prompt_used_len; only time
                // transitions one step earlier.
                bool use_prompted_time =
                    (feat_idx < prompt_used_len - 1) && (prompt_feat_idx + 1 < (int)ctx->prompt_time_before.size());
                int tb = use_prompted_time ? ctx->prompt_time_before[prompt_feat_idx + 1] : pred_t_before;
                int ta = use_prompted_time ? ((prompt_feat_idx + 1 < (int)ctx->prompt_time_after.size())
                                                  ? ctx->prompt_time_after[prompt_feat_idx + 1]
                                                  : pred_t_after)
                                           : pred_t_after;
                time_before_list.push_back(tb);
                cur_acoustic = feat;
                cur_mask = 1;
                cur_t_before = tb;
                cur_t_after = ta;
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
            // step < shift: structural chat-header tokens (system/user/assistant markers).
            // Python's batch prefill zero-initialises time_before for ALL prompt
            // positions and only fills in real values at shift+1..shift+n_t. Keep
            // cur_t_before/cur_t_after = 0 here so every header step gets the same
            // zero time embedding Python would produce.
            std::fill(cur_acoustic.begin(), cur_acoustic.end(), 0.0f);
            cur_mask = 0;
            cur_t_before = 0;
            cur_t_after = 0;
        }

        free(tr.hidden);
    }

    if (acoustic_features.empty()) {
        fprintf(stderr, "tada: no acoustic features generated\n");
        return nullptr;
    }

    if (dump_fm_steps) {
        const char* dump_path = std::getenv("TADA_DUMP_FM_STEPS");
        if (FILE* f = fopen(dump_path, "wb")) {
            uint32_t hdr[4] = {(uint32_t)fm_dump_records.size(), (uint32_t)lat, (uint32_t)hp.fm_hidden,
                               (uint32_t)hp.time_dim};
            fwrite(hdr, sizeof(hdr), 1, f);
            for (size_t i = 0; i < fm_dump_records.size(); i++) {
                float idx[2] = {(float)i, (float)i};
                fwrite(idx, sizeof(float), 2, f);
            }
            auto write_block = [&](const std::vector<float> tada_fm_dump_record::*field, uint32_t width) {
                (void)width;
                for (const auto& r : fm_dump_records) {
                    const std::vector<float>& v = r.*field;
                    fwrite(v.data(), sizeof(float), v.size(), f);
                }
            };
            write_block(&tada_fm_dump_record::speech_in, (uint32_t)lat);
            write_block(&tada_fm_dump_record::cond, (uint32_t)hp.fm_hidden);
            write_block(&tada_fm_dump_record::neg_cond, (uint32_t)hp.fm_hidden);
            write_block(&tada_fm_dump_record::speech_out, (uint32_t)lat);
            write_block(&tada_fm_dump_record::time_bits, (uint32_t)hp.time_dim);
            fclose(f);
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "tada: dumped %zu FM call records to %s\n", fm_dump_records.size(), dump_path);
        } else {
            fprintf(stderr, "tada: WARN: could not open TADA_DUMP_FM_STEPS=%s\n", dump_path);
        }
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

    // Match Python:
    //   encoded = acoustic_features[..., num_prompt_tokens + num_transition_steps - 1:, :]
    // where num_prompt_tokens is prompt_acoustic_features after prefix padding
    // and transition trimming. With no voice prompt this keeps the acoustic
    // frames generated after the short prefix transition instead of decoding a
    // mostly-zero prefix.
    int skip_frames = prompt_used_len + transition_steps - 1;
    if (skip_frames < 0)
        skip_frames = 0;
    if (skip_frames >= (int)acoustic_features.size())
        skip_frames = std::max(0, (int)acoustic_features.size() - 1);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "tada: skipping %d prompt-phase frames, decoding %d (num_prompt=%d shift=%d n_prompt=%d)\n",
                skip_frames, (int)acoustic_features.size() - skip_frames, num_prompt, (int)shift, ctx->n_prompt);
    }

    // Expand with time_before durations (same as model._decode_wav)
    std::vector<float> expanded;
    std::vector<int32_t> token_masks;
    std::vector<int> all_times;
    // time_before for the decode portion starts at skip_frames. Python passes
    // this first duration into _decode_wav(), lets it affect codec attention,
    // then trims the corresponding leading silence from the decoded PCM.
    for (int i = skip_frames; i < (int)time_before_list.size(); i++) {
        all_times.push_back(time_before_list[i]);
    }

    // Use only features from skip_frames onwards
    std::vector<std::vector<float>> decode_feats(acoustic_features.begin() + skip_frames, acoustic_features.end());

    if (const char* dump_acoustic_path = getenv("TADA_DUMP_ACOUSTIC_FEATURES");
        dump_acoustic_path && dump_acoustic_path[0]) {
        if (FILE* f = fopen(dump_acoustic_path, "wb")) {
            uint32_t hdr[2] = {(uint32_t)acoustic_features.size(), (uint32_t)ad};
            fwrite(hdr, sizeof(hdr), 1, f);
            for (const auto& feat : acoustic_features) {
                for (int d = 0; d < ad; d++) {
                    float v = feat[d] * ac_std + ac_mean;
                    fwrite(&v, sizeof(float), 1, f);
                }
            }
            fclose(f);
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "tada: dumped %zu acoustic features to %s\n", acoustic_features.size(),
                        dump_acoustic_path);
        }
    }

    if (const char* dump_time_path = getenv("TADA_DUMP_TIME_BEFORE"); dump_time_path && dump_time_path[0]) {
        if (FILE* f = fopen(dump_time_path, "wb")) {
            std::vector<float> dump_times;
            dump_times.reserve(all_times.size() + (all_times.empty() ? 0 : 1));
            for (int t : all_times)
                dump_times.push_back((float)t);
            if (!all_times.empty())
                dump_times.push_back((float)all_times.back());
            uint32_t n = (uint32_t)dump_times.size();
            fwrite(&n, sizeof(n), 1, f);
            fwrite(dump_times.data(), sizeof(float), dump_times.size(), f);
            fclose(f);
        }
    }

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

    // Safety guard (#192): the codec decodes every expanded frame in a single
    // graph (~1.9 MB/frame), so frame count drives a large allocation. A talker
    // that never emits EOS runs to the generation cap and, combined with corrupt
    // per-token durations (e.g. a broken GPU backend producing garbage logits),
    // expands to tens of thousands of frames — one such runaway tried to
    // allocate 113 GB and segfaulted. A legitimate single-utterance synth never
    // exceeds 512 acoustic tokens × ~32 frames ≈ 16 k frames, so refuse to decode
    // anything well past that and return silence with a clear diagnostic instead
    // of crashing the machine.
    {
        int max_expanded = 16384;
        if (const char* e = getenv("TADA_MAX_EXPANDED_FRAMES"); e && e[0])
            max_expanded = atoi(e);
        if (max_expanded > 0 && n_expanded > max_expanded) {
            fprintf(stderr,
                    "tada: ERROR: expansion runaway — %d frames exceeds cap %d. This usually means "
                    "generation never hit EOS and/or produced corrupt durations (a known symptom of an "
                    "unvalidated GPU backend, e.g. Vulkan). Refusing codec decode to avoid a huge "
                    "allocation; returning silence. Re-run on CPU (--no-gpu), or raise the cap with "
                    "TADA_MAX_EXPANDED_FRAMES.\n",
                    n_expanded, max_expanded);
            *out_n_samples = 24000;
            return (float*)calloc(24000, sizeof(float));
        }
    }

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

    // Optional feature dump for diff harness (TADA_DUMP_FEATURES=/path/to/file).
    // Python side: tools/reference_backends/tada_codec_diff.py --features <path>
    {
        const char* dump_path = getenv("TADA_DUMP_FEATURES");
        if (dump_path && n_expanded > 0) {
            FILE* df = fopen(dump_path, "wb");
            if (df) {
                uint32_t hdr[2] = {(uint32_t)n_expanded, (uint32_t)ad};
                fwrite(hdr, sizeof(hdr), 1, df);
                fwrite(expanded.data(), sizeof(float), (size_t)n_expanded * ad, df);
                fclose(df);
                fprintf(stderr, "tada: dumped %d×%d expanded features → %s\n", n_expanded, ad, dump_path);
            } else {
                fprintf(stderr, "tada: WARN: could not open TADA_DUMP_FEATURES=%s\n", dump_path);
            }
        }
    }

    // ── Codec decode ──
    if (ctx->codec_ctx && n_expanded > 0) {
        tada_bench_stage _bs("codec_decode");
        int n_samples = 0;
        float* pcm = tada_codec_decode(ctx->codec_ctx, expanded.data(), n_expanded, token_masks.data(), &n_samples);
        if (pcm && n_samples > 0) {
            int trim = 0;
            if (!all_times.empty())
                trim = (int)((int64_t)24000 * (int64_t)all_times[0] / 50);
            if (trim > 0 && trim < n_samples) {
                int kept = n_samples - trim;
                float* out = (float*)malloc((size_t)kept * sizeof(float));
                if (out) {
                    memcpy(out, pcm + trim, (size_t)kept * sizeof(float));
                    tada_codec_pcm_free(pcm);
                    *out_n_samples = kept;
                    return out;
                }
            }
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
    if (ctx->fm_b2_sched)
        ggml_backend_sched_free(ctx->fm_b2_sched);
    if (ctx->fm_b2_galloc)
        ggml_gallocr_free(ctx->fm_b2_galloc);
    if (ctx->fm_b2_ctx)
        ggml_free(ctx->fm_b2_ctx);
    if (ctx->fm_b2_buf_f16)
        ggml_backend_buffer_free(ctx->fm_b2_buf_f16);
    if (ctx->fm_b2_ctx_f16)
        ggml_free(ctx->fm_b2_ctx_f16);
    if (ctx->fm_batch_sched)
        ggml_backend_sched_free(ctx->fm_batch_sched);
    if (ctx->fm_batch_galloc)
        ggml_gallocr_free(ctx->fm_batch_galloc);
    if (ctx->fm_batch_ctx)
        ggml_free(ctx->fm_batch_ctx);
    if (ctx->fm_step_sched)
        ggml_backend_sched_free(ctx->fm_step_sched);
    if (ctx->fm_step_galloc)
        ggml_gallocr_free(ctx->fm_step_galloc);
    if (ctx->fm_step_ctx)
        ggml_free(ctx->fm_step_ctx);
    if (ctx->ar_step_sched)
        ggml_backend_sched_free(ctx->ar_step_sched);
    if (ctx->ar_step_galloc)
        ggml_gallocr_free(ctx->ar_step_galloc);
    for (auto& bk : ctx->ar_buckets)
        if (bk.ctx)
            ggml_free(bk.ctx);
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
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
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
