// arktts.cpp — ArkTTS backend (Audio8-TTS-Realistic-0.6b).
//
// Dual-stage transformer (slow AR causal transformer + fast RAS layers)
// with 44.1 kHz Snake-activated vocoder decoder.

#include "arktts.h"

#include "core/activation.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/conv.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/sdpa.h"
#include "gguf.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/ggml_cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct arktts_hparams {
    uint32_t dim = 896;
    uint32_t intermediate_size = 4864;
    uint32_t n_layer = 24;
    uint32_t n_head = 14;
    uint32_t n_local_heads = 2;
    uint32_t head_dim = 64;
    float norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    uint32_t vocab_size = 155776;
    uint32_t num_codebooks = 10;
    uint32_t codebook_size = 4096;
    uint32_t semantic_begin_id = 151678;
    uint32_t semantic_end_id = 155773;
    uint32_t eos_token_id = 151645;

    uint32_t fast_dim = 896;
    uint32_t fast_intermediate_size = 4864;
    uint32_t n_fast_layer = 4;
    uint32_t fast_n_head = 14;
    uint32_t fast_n_local_heads = 2;
    uint32_t fast_head_dim = 64;

    uint32_t sample_rate = 44100;
    uint32_t frame_size = 2048;
    uint32_t hop_length = 512;
};

struct arktts_context {
    arktts_hparams hp;
    arktts_context_params params;
    core_gguf::WeightLoad wl;
    core_gguf::tensor_map tensors;
    ggml_backend_t backend = nullptr;
    std::mt19937_64 rng;
    std::vector<float> ref_audio;
    std::string ref_text;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    std::vector<float> last_slow_normed;

    // KV Cache for Slow Transformer (24 layers, up to max_seq_len tokens)
    int max_seq_len = 1024;
    std::vector<std::vector<float>> slow_kv_k;
    std::vector<std::vector<float>> slow_kv_v;

    // KV Cache for Fast Subnetwork (4 layers, 10 codebooks)
    std::vector<std::vector<float>> fast_kv_k;
    std::vector<std::vector<float>> fast_kv_v;
};

static ggml_tensor* get_tensor(const arktts_context* ctx, const std::string& name) {
    if (!ctx)
        return nullptr;
    auto it = ctx->tensors.find(name);
    return it != ctx->tensors.end() ? it->second : nullptr;
}

static void dequant_row(const ggml_tensor* t, int64_t row, float* dst) {
    if (!t || !dst)
        return;
    int64_t n = t->ne[0];
    const char* src = (const char*)t->data + row * t->nb[1];
    const auto* traits = ggml_get_type_traits(t->type);
    if (traits && traits->to_float) {
        traits->to_float(src, dst, n);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(dst, src, n * sizeof(float));
    }
}

struct arktts_context_params arktts_context_default_params(void) {
    struct arktts_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.7f;
    p.top_p = 0.9f;
    p.top_k = 50;
    p.seed = 42;
    p.max_tokens = 512;
    p.flash_attn = false;
    return p;
}

int arktts_get_sample_rate(const struct arktts_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 44100;
}

void arktts_set_temperature(struct arktts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void arktts_set_seed(struct arktts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        ctx->rng.seed(seed ? seed : 42);
    }
}

int arktts_set_max_new_tokens(struct arktts_context* ctx, int max_new_tokens) {
    if (!ctx)
        return -1;
    ctx->params.max_tokens = max_new_tokens > 0 ? max_new_tokens : 512;
    return 0;
}

int arktts_set_reference(struct arktts_context* ctx, const float* audio_44k, int n_samples,
                         const char* reference_text) {
    if (!ctx)
        return -1;
    if (audio_44k && n_samples > 0) {
        ctx->ref_audio.assign(audio_44k, audio_44k + n_samples);
        ctx->ref_text = reference_text ? reference_text : "";
    } else {
        ctx->ref_audio.clear();
        ctx->ref_text.clear();
    }
    return 0;
}

int arktts_load_tokenizer(struct arktts_context* /*ctx*/, const char* /*tokenizer_json_path*/) {
    return 0;
}

struct arktts_context* arktts_init_from_file(const char* path_model, struct arktts_context_params params) {
    auto c = std::make_unique<arktts_context>();
    c->params = params;
    c->rng.seed(params.seed ? params.seed : 42);

    c->backend = ggml_backend_cpu_init();

    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        return nullptr;
    }

    c->hp.dim = core_gguf::kv_u32(meta, "arktts.dim", c->hp.dim);
    c->hp.intermediate_size = core_gguf::kv_u32(meta, "arktts.intermediate_size", c->hp.intermediate_size);
    c->hp.n_layer = core_gguf::kv_u32(meta, "arktts.n_layer", c->hp.n_layer);
    c->hp.n_head = core_gguf::kv_u32(meta, "arktts.n_head", c->hp.n_head);
    c->hp.n_local_heads = core_gguf::kv_u32(meta, "arktts.n_local_heads", c->hp.n_local_heads);
    c->hp.head_dim = core_gguf::kv_u32(meta, "arktts.head_dim", c->hp.head_dim);
    c->hp.norm_eps = core_gguf::kv_f32(meta, "arktts.norm_eps", c->hp.norm_eps);
    c->hp.rope_theta = core_gguf::kv_f32(meta, "arktts.rope_theta", c->hp.rope_theta);
    c->hp.vocab_size = core_gguf::kv_u32(meta, "arktts.vocab_size", c->hp.vocab_size);
    c->hp.num_codebooks = core_gguf::kv_u32(meta, "arktts.num_codebooks", c->hp.num_codebooks);
    c->hp.codebook_size = core_gguf::kv_u32(meta, "arktts.codebook_size", c->hp.codebook_size);
    c->hp.semantic_begin_id = core_gguf::kv_u32(meta, "arktts.semantic_begin_id", c->hp.semantic_begin_id);
    c->hp.semantic_end_id = core_gguf::kv_u32(meta, "arktts.semantic_end_id", c->hp.semantic_end_id);
    c->hp.sample_rate = core_gguf::kv_u32(meta, "arktts.codec.sample_rate", c->hp.sample_rate);
    c->hp.frame_size = core_gguf::kv_u32(meta, "arktts.codec.frame_size", c->hp.frame_size);
    c->hp.hop_length = core_gguf::kv_u32(meta, "arktts.codec.hop_length", c->hp.hop_length);

    c->id_to_token = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    for (size_t i = 0; i < c->id_to_token.size(); i++) {
        c->token_to_id[c->id_to_token[i]] = (int32_t)i;
    }
    auto merges = core_gguf::kv_str_array(meta, "tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); i++) {
        c->merge_rank[merges[i]] = (int32_t)i;
    }
    core_gguf::free_metadata(meta);

    if (!core_gguf::load_weights(path_model, c->backend, "arktts", c->wl)) {
        return nullptr;
    }
    c->tensors = std::move(c->wl.tensors);

    c->max_seq_len = 1024;
    const int kv_dim = c->hp.n_local_heads * c->hp.head_dim;
    c->slow_kv_k.resize(c->hp.n_layer, std::vector<float>((size_t)c->max_seq_len * kv_dim, 0.0f));
    c->slow_kv_v.resize(c->hp.n_layer, std::vector<float>((size_t)c->max_seq_len * kv_dim, 0.0f));

    const int fast_kv_dim = c->hp.fast_n_local_heads * c->hp.fast_head_dim;
    c->fast_kv_k.resize(c->hp.n_fast_layer, std::vector<float>((size_t)c->hp.num_codebooks * fast_kv_dim, 0.0f));
    c->fast_kv_v.resize(c->hp.n_fast_layer, std::vector<float>((size_t)c->hp.num_codebooks * fast_kv_dim, 0.0f));

    return c.release();
}

float* arktts_run_embed(struct arktts_context* ctx, const int32_t* input_ids, int n_tokens, int* out_dim) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int dim = ctx->hp.dim;
    if (out_dim)
        *out_dim = dim;
    float* embeds = (float*)malloc((size_t)n_tokens * dim * sizeof(float));
    if (!embeds)
        return nullptr;

    ggml_tensor* emb_w = get_tensor(ctx, "embeddings.weight");
    ggml_tensor* cb_w = get_tensor(ctx, "codebook_embeddings.weight");
    if (!emb_w) {
        free(embeds);
        return nullptr;
    }

    std::vector<float> row_buf(dim);
    for (int t = 0; t < n_tokens; t++) {
        int32_t text_id = input_ids[0 * n_tokens + t];
        if (text_id < 0 || text_id >= (int32_t)emb_w->ne[1])
            text_id = 0;
        dequant_row(emb_w, text_id, embeds + (size_t)t * dim);

        if (cb_w && text_id >= (int32_t)ctx->hp.semantic_begin_id && text_id <= (int32_t)ctx->hp.semantic_end_id) {
            for (int c = 0; c < (int)ctx->hp.num_codebooks; c++) {
                int32_t code = input_ids[(c + 1) * n_tokens + t];
                if (code >= 0 && code < (int32_t)ctx->hp.codebook_size) {
                    int32_t cb_idx = code + c * (int32_t)ctx->hp.codebook_size;
                    if (cb_idx < (int32_t)cb_w->ne[1]) {
                        dequant_row(cb_w, cb_idx, row_buf.data());
                        for (int d = 0; d < dim; d++) {
                            embeds[(size_t)t * dim + d] += row_buf[d];
                        }
                    }
                }
            }
        }
    }
    return embeds;
}

float* arktts_run_slow_transformer(struct arktts_context* ctx, const float* embeds, int n_tokens, int* out_vocab) {
    if (!ctx || !embeds || n_tokens <= 0)
        return nullptr;
    const int dim = ctx->hp.dim;
    const int vocab_size = ctx->hp.vocab_size;
    if (out_vocab)
        *out_vocab = vocab_size;

    size_t mem_size = (size_t)256 * 1024 * 1024 + (size_t)n_tokens * 2 * 1024 * 1024;
    ggml_init_params gip = {mem_size, nullptr, false};
    ggml_context* gctx = ggml_init(gip);
    if (!gctx)
        return nullptr;

    ggml_tensor* cur = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, dim, n_tokens);
    memcpy(cur->data, embeds, (size_t)n_tokens * dim * sizeof(float));

    ggml_tensor* pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_tokens);
    int32_t* pos_data = (int32_t*)pos_ids->data;
    for (int i = 0; i < n_tokens; i++) {
        pos_data[i] = i;
    }

    ggml_tensor* mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_tokens, n_tokens);
    float* mask_data = (float*)mask->data;
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_tokens; j++) {
            mask_data[i * n_tokens + j] = (j <= i) ? 0.0f : -1e9f;
        }
    }

    const int n_head = ctx->hp.n_head;
    const int n_local_heads = ctx->hp.n_local_heads;
    const int head_dim = ctx->hp.head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_local_heads * head_dim;
    const int total_dim = q_dim + 2 * kv_dim;
    const size_t esz = sizeof(float);

    std::vector<ggml_tensor*> k_saved(ctx->hp.n_layer);
    std::vector<ggml_tensor*> v_saved(ctx->hp.n_layer);

    for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
        char name[128];
        snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", l);
        ggml_tensor* attn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wqkv.weight", l);
        ggml_tensor* wqkv = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wqkv.bias", l);
        ggml_tensor* wqkv_b = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", l);
        ggml_tensor* wo = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", l);
        ggml_tensor* ffn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", l);
        ggml_tensor* w1 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", l);
        ggml_tensor* w2 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", l);
        ggml_tensor* w3 = get_tensor(ctx, name);

        ggml_tensor* an = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (attn_norm_w)
            an = ggml_mul(gctx, an, attn_norm_w);

        ggml_tensor* qkv = ggml_mul_mat(gctx, wqkv, an);
        if (wqkv_b)
            qkv = ggml_add(gctx, qkv, wqkv_b);

        ggml_tensor* q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, q_dim, n_tokens, total_dim * esz, 0));
        ggml_tensor* k = ggml_cont(gctx, ggml_view_2d(gctx, qkv, kv_dim, n_tokens, total_dim * esz, q_dim * esz));
        ggml_tensor* v =
            ggml_cont(gctx, ggml_view_2d(gctx, qkv, kv_dim, n_tokens, total_dim * esz, (q_dim + kv_dim) * esz));

        q = ggml_reshape_3d(gctx, q, head_dim, n_head, n_tokens);
        k = ggml_reshape_3d(gctx, k, head_dim, n_local_heads, n_tokens);
        v = ggml_reshape_3d(gctx, v, head_dim, n_local_heads, n_tokens);

        q = ggml_rope_ext(gctx, q, pos_ids, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, ctx->hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, pos_ids, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, ctx->hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);

        k_saved[l] = k;
        v_saved[l] = v;

        const int grp = n_head / n_local_heads;
        if (grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(gctx, k, head_dim, 1, n_local_heads, n_tokens);
            ggml_tensor* V4 = ggml_reshape_4d(gctx, v, head_dim, 1, n_local_heads, n_tokens);
            K4 = ggml_repeat_4d(gctx, K4, head_dim, grp, n_local_heads, n_tokens);
            V4 = ggml_repeat_4d(gctx, V4, head_dim, grp, n_local_heads, n_tokens);
            k = ggml_cont(gctx, ggml_reshape_3d(gctx, K4, head_dim, n_head, n_tokens));
            v = ggml_cont(gctx, ggml_reshape_3d(gctx, V4, head_dim, n_head, n_tokens));
        }

        q = ggml_permute(gctx, q, 0, 2, 1, 3);
        k = ggml_permute(gctx, k, 0, 2, 1, 3);
        v = ggml_permute(gctx, v, 0, 2, 1, 3);

        ggml_tensor* attn = core_sdpa::attn(gctx, q, k, v, mask, dim, n_tokens, 1.0f / sqrtf((float)head_dim), false);
        ggml_tensor* attn_out = ggml_mul_mat(gctx, wo, attn);
        cur = ggml_add(gctx, cur, attn_out);

        ggml_tensor* fn = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (ffn_norm_w)
            fn = ggml_mul(gctx, fn, ffn_norm_w);

        ggml_tensor* mlp = core_ffn::swiglu(gctx, fn, w1, w3, w2);
        cur = ggml_add(gctx, cur, mlp);
    }

    ggml_tensor* norm_w = get_tensor(ctx, "norm.weight");
    cur = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
    if (norm_w)
        cur = ggml_mul(gctx, cur, norm_w);

    ggml_tensor* last_hidden = ggml_view_1d(gctx, cur, dim, (size_t)(n_tokens - 1) * dim * sizeof(float));
    ggml_tensor* emb_w = get_tensor(ctx, "embeddings.weight");
    ggml_tensor* logits = ggml_mul_mat(gctx, emb_w, last_hidden);

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, last_hidden);
    for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
        ggml_build_forward_expand(gf, k_saved[l]);
        ggml_build_forward_expand(gf, v_saved[l]);
    }

    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    ctx->last_slow_normed.resize(dim);
    memcpy(ctx->last_slow_normed.data(), last_hidden->data, dim * sizeof(float));

    if ((int)ctx->slow_kv_k.size() < (int)ctx->hp.n_layer) {
        ctx->slow_kv_k.resize(ctx->hp.n_layer);
        ctx->slow_kv_v.resize(ctx->hp.n_layer);
    }
    for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
        if ((int)ctx->slow_kv_k[l].size() < ctx->max_seq_len * kv_dim) {
            ctx->slow_kv_k[l].resize((size_t)ctx->max_seq_len * kv_dim, 0.0f);
            ctx->slow_kv_v[l].resize((size_t)ctx->max_seq_len * kv_dim, 0.0f);
        }
        memcpy(ctx->slow_kv_k[l].data(), k_saved[l]->data, (size_t)n_tokens * kv_dim * sizeof(float));
        memcpy(ctx->slow_kv_v[l].data(), v_saved[l]->data, (size_t)n_tokens * kv_dim * sizeof(float));
    }

    float* out = (float*)malloc(vocab_size * sizeof(float));
    if (out) {
        memcpy(out, logits->data, vocab_size * sizeof(float));
    }
    ggml_free(gctx);
    return out;
}

static float* arktts_step_slow_transformer(struct arktts_context* ctx, const float* single_embed, int pos,
                                           int* out_vocab) {
    if (!ctx || !single_embed || pos < 0 || pos >= ctx->max_seq_len)
        return nullptr;
    const int dim = ctx->hp.dim;
    const int vocab_size = ctx->hp.vocab_size;
    if (out_vocab)
        *out_vocab = vocab_size;

    size_t mem_size = 32 * 1024 * 1024;
    ggml_init_params gip = {mem_size, nullptr, false};
    ggml_context* gctx = ggml_init(gip);
    if (!gctx)
        return nullptr;

    ggml_tensor* cur = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, dim, 1);
    memcpy(cur->data, single_embed, dim * sizeof(float));

    ggml_tensor* pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    *(int32_t*)pos_ids->data = pos;

    const int n_head = ctx->hp.n_head;
    const int n_local_heads = ctx->hp.n_local_heads;
    const int head_dim = ctx->hp.head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_local_heads * head_dim;
    const int total_dim = q_dim + 2 * kv_dim;
    const size_t esz = sizeof(float);
    const int grp = n_head / n_local_heads;

    if (pos >= ctx->max_seq_len) {
        int new_max = std::max(pos + 512, ctx->max_seq_len * 2);
        for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
            ctx->slow_kv_k[l].resize((size_t)new_max * kv_dim, 0.0f);
            ctx->slow_kv_v[l].resize((size_t)new_max * kv_dim, 0.0f);
        }
        ctx->max_seq_len = new_max;
    }

    std::vector<ggml_tensor*> k_step_saved(ctx->hp.n_layer);
    std::vector<ggml_tensor*> v_step_saved(ctx->hp.n_layer);

    for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
        char name[128];
        snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", l);
        ggml_tensor* attn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wqkv.weight", l);
        ggml_tensor* wqkv = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wqkv.bias", l);
        ggml_tensor* wqkv_b = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", l);
        ggml_tensor* wo = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", l);
        ggml_tensor* ffn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", l);
        ggml_tensor* w1 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", l);
        ggml_tensor* w2 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", l);
        ggml_tensor* w3 = get_tensor(ctx, name);

        ggml_tensor* an = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (attn_norm_w)
            an = ggml_mul(gctx, an, attn_norm_w);

        ggml_tensor* qkv = ggml_mul_mat(gctx, wqkv, an);
        if (wqkv_b)
            qkv = ggml_add(gctx, qkv, wqkv_b);

        ggml_tensor* q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, q_dim, 1, total_dim * esz, 0));
        ggml_tensor* k_new = ggml_cont(gctx, ggml_view_2d(gctx, qkv, kv_dim, 1, total_dim * esz, q_dim * esz));
        ggml_tensor* v_new =
            ggml_cont(gctx, ggml_view_2d(gctx, qkv, kv_dim, 1, total_dim * esz, (q_dim + kv_dim) * esz));

        q = ggml_reshape_3d(gctx, q, head_dim, n_head, 1);
        k_new = ggml_reshape_3d(gctx, k_new, head_dim, n_local_heads, 1);
        v_new = ggml_reshape_3d(gctx, v_new, head_dim, n_local_heads, 1);

        q = ggml_rope_ext(gctx, q, pos_ids, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, ctx->hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        k_new = ggml_rope_ext(gctx, k_new, pos_ids, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, ctx->hp.rope_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        k_step_saved[l] = k_new;
        v_step_saved[l] = v_new;

        ggml_tensor* k_all = k_new;
        ggml_tensor* v_all = v_new;
        if (pos > 0) {
            ggml_tensor* k_past = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, head_dim, n_local_heads, pos);
            k_past->data = ctx->slow_kv_k[l].data();
            ggml_tensor* v_past = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, head_dim, n_local_heads, pos);
            v_past->data = ctx->slow_kv_v[l].data();

            k_all = ggml_concat(gctx, k_past, k_new, 2);
            v_all = ggml_concat(gctx, v_past, v_new, 2);
        }

        const int total_len = pos + 1;
        if (grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(gctx, k_all, head_dim, 1, n_local_heads, total_len);
            ggml_tensor* V4 = ggml_reshape_4d(gctx, v_all, head_dim, 1, n_local_heads, total_len);
            K4 = ggml_repeat_4d(gctx, K4, head_dim, grp, n_local_heads, total_len);
            V4 = ggml_repeat_4d(gctx, V4, head_dim, grp, n_local_heads, total_len);
            k_all = ggml_cont(gctx, ggml_reshape_3d(gctx, K4, head_dim, n_head, total_len));
            v_all = ggml_cont(gctx, ggml_reshape_3d(gctx, V4, head_dim, n_head, total_len));
        }

        q = ggml_permute(gctx, q, 0, 2, 1, 3);
        k_all = ggml_permute(gctx, k_all, 0, 2, 1, 3);
        v_all = ggml_permute(gctx, v_all, 0, 2, 1, 3);

        ggml_tensor* attn =
            core_sdpa::attn(gctx, q, k_all, v_all, nullptr, dim, 1, 1.0f / sqrtf((float)head_dim), false);
        ggml_tensor* attn_out = ggml_mul_mat(gctx, wo, attn);
        cur = ggml_add(gctx, cur, attn_out);

        ggml_tensor* fn = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (ffn_norm_w)
            fn = ggml_mul(gctx, fn, ffn_norm_w);

        ggml_tensor* mlp = core_ffn::swiglu(gctx, fn, w1, w3, w2);
        cur = ggml_add(gctx, cur, mlp);
    }

    ggml_tensor* norm_w = get_tensor(ctx, "norm.weight");
    cur = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
    if (norm_w)
        cur = ggml_mul(gctx, cur, norm_w);

    ggml_tensor* last_hidden = cur;
    ggml_tensor* emb_w = get_tensor(ctx, "embeddings.weight");
    ggml_tensor* logits = ggml_mul_mat(gctx, emb_w, last_hidden);

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, last_hidden);
    for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
        ggml_build_forward_expand(gf, k_step_saved[l]);
        ggml_build_forward_expand(gf, v_step_saved[l]);
    }

    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    for (int l = 0; l < (int)ctx->hp.n_layer; l++) {
        memcpy(ctx->slow_kv_k[l].data() + (size_t)pos * kv_dim, k_step_saved[l]->data, kv_dim * sizeof(float));
        memcpy(ctx->slow_kv_v[l].data() + (size_t)pos * kv_dim, v_step_saved[l]->data, kv_dim * sizeof(float));
    }

    ctx->last_slow_normed.resize(dim);
    memcpy(ctx->last_slow_normed.data(), last_hidden->data, dim * sizeof(float));

    float* out = (float*)malloc(vocab_size * sizeof(float));
    if (out) {
        memcpy(out, logits->data, vocab_size * sizeof(float));
    }
    ggml_free(gctx);
    return out;
}

static void arktts_step_fast_subnetwork(struct arktts_context* ctx, const float* token_embed, int pos,
                                        float* out_logits) {
    const int dim = ctx->hp.fast_dim;
    const int codebook_size = ctx->hp.codebook_size;
    const int n_head = ctx->hp.fast_n_head;
    const int n_local_heads = ctx->hp.fast_n_local_heads;
    const int head_dim = ctx->hp.fast_head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_local_heads * head_dim;
    const int total_dim = q_dim + 2 * kv_dim;
    const size_t esz = sizeof(float);
    const int grp = n_head / n_local_heads;

    size_t mem_size = 4 * 1024 * 1024;
    ggml_init_params gip = {mem_size, nullptr, false};
    ggml_context* gctx = ggml_init(gip);
    if (!gctx)
        return;

    ggml_tensor* cur = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, dim, 1);
    memcpy(cur->data, token_embed, dim * sizeof(float));

    ggml_tensor* pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    *(int32_t*)pos_ids->data = pos;

    std::vector<ggml_tensor*> k_step_saved(ctx->hp.n_fast_layer);
    std::vector<ggml_tensor*> v_step_saved(ctx->hp.n_fast_layer);

    for (int l = 0; l < (int)ctx->hp.n_fast_layer; l++) {
        char name[128];
        snprintf(name, sizeof(name), "fast_layers.%d.attention_norm.weight", l);
        ggml_tensor* attn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "fast_layers.%d.attention.wqkv.weight", l);
        ggml_tensor* wqkv = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "fast_layers.%d.attention.wo.weight", l);
        ggml_tensor* wo = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "fast_layers.%d.ffn_norm.weight", l);
        ggml_tensor* ffn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "fast_layers.%d.feed_forward.w1.weight", l);
        ggml_tensor* w1 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "fast_layers.%d.feed_forward.w2.weight", l);
        ggml_tensor* w2 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "fast_layers.%d.feed_forward.w3.weight", l);
        ggml_tensor* w3 = get_tensor(ctx, name);

        ggml_tensor* an = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (attn_norm_w)
            an = ggml_mul(gctx, an, attn_norm_w);

        ggml_tensor* qkv = ggml_mul_mat(gctx, wqkv, an);

        ggml_tensor* q = ggml_cont(gctx, ggml_view_2d(gctx, qkv, q_dim, 1, total_dim * esz, 0));
        ggml_tensor* k_new = ggml_cont(gctx, ggml_view_2d(gctx, qkv, kv_dim, 1, total_dim * esz, q_dim * esz));
        ggml_tensor* v_new =
            ggml_cont(gctx, ggml_view_2d(gctx, qkv, kv_dim, 1, total_dim * esz, (q_dim + kv_dim) * esz));

        q = ggml_reshape_3d(gctx, q, head_dim, n_head, 1);
        k_new = ggml_reshape_3d(gctx, k_new, head_dim, n_local_heads, 1);
        v_new = ggml_reshape_3d(gctx, v_new, head_dim, n_local_heads, 1);

        q = ggml_rope_ext(gctx, q, pos_ids, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, ctx->hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        k_new = ggml_rope_ext(gctx, k_new, pos_ids, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, ctx->hp.rope_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        k_step_saved[l] = k_new;
        v_step_saved[l] = v_new;

        ggml_tensor* k_all = k_new;
        ggml_tensor* v_all = v_new;
        if (pos > 0) {
            ggml_tensor* k_past = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, head_dim, n_local_heads, pos);
            k_past->data = ctx->fast_kv_k[l].data();
            ggml_tensor* v_past = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, head_dim, n_local_heads, pos);
            v_past->data = ctx->fast_kv_v[l].data();

            k_all = ggml_concat(gctx, k_past, k_new, 2);
            v_all = ggml_concat(gctx, v_past, v_new, 2);
        }

        const int total_len = pos + 1;
        if (grp > 1) {
            ggml_tensor* K4 = ggml_reshape_4d(gctx, k_all, head_dim, 1, n_local_heads, total_len);
            ggml_tensor* V4 = ggml_reshape_4d(gctx, v_all, head_dim, 1, n_local_heads, total_len);
            K4 = ggml_repeat_4d(gctx, K4, head_dim, grp, n_local_heads, total_len);
            V4 = ggml_repeat_4d(gctx, V4, head_dim, grp, n_local_heads, total_len);
            k_all = ggml_cont(gctx, ggml_reshape_3d(gctx, K4, head_dim, n_head, total_len));
            v_all = ggml_cont(gctx, ggml_reshape_3d(gctx, V4, head_dim, n_head, total_len));
        }

        q = ggml_permute(gctx, q, 0, 2, 1, 3);
        k_all = ggml_permute(gctx, k_all, 0, 2, 1, 3);
        v_all = ggml_permute(gctx, v_all, 0, 2, 1, 3);

        ggml_tensor* attn =
            core_sdpa::attn(gctx, q, k_all, v_all, nullptr, dim, 1, 1.0f / sqrtf((float)head_dim), false);
        ggml_tensor* attn_out = ggml_mul_mat(gctx, wo, attn);
        cur = ggml_add(gctx, cur, attn_out);

        ggml_tensor* fn = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (ffn_norm_w)
            fn = ggml_mul(gctx, fn, ffn_norm_w);

        ggml_tensor* mlp = core_ffn::swiglu(gctx, fn, w1, w3, w2);
        cur = ggml_add(gctx, cur, mlp);
    }

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_tensor* logits = nullptr;
    if (out_logits) {
        ggml_tensor* fast_norm_w = get_tensor(ctx, "fast_norm.weight");
        cur = ggml_rms_norm(gctx, cur, ctx->hp.norm_eps);
        if (fast_norm_w)
            cur = ggml_mul(gctx, cur, fast_norm_w);
        ggml_tensor* fast_out_w = get_tensor(ctx, "fast_output.weight");
        logits = ggml_mul_mat(gctx, fast_out_w, cur);
        ggml_build_forward_expand(gf, logits);
    }
    for (int l = 0; l < (int)ctx->hp.n_fast_layer; l++) {
        ggml_build_forward_expand(gf, k_step_saved[l]);
        ggml_build_forward_expand(gf, v_step_saved[l]);
    }

    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    for (int l = 0; l < (int)ctx->hp.n_fast_layer; l++) {
        memcpy(ctx->fast_kv_k[l].data() + (size_t)pos * kv_dim, k_step_saved[l]->data, kv_dim * sizeof(float));
        memcpy(ctx->fast_kv_v[l].data() + (size_t)pos * kv_dim, v_step_saved[l]->data, kv_dim * sizeof(float));
    }

    if (out_logits && logits) {
        memcpy(out_logits, logits->data, codebook_size * sizeof(float));
    }
    ggml_free(gctx);
}

int arktts_run_fast_subnetwork(struct arktts_context* ctx, const float* slow_hidden, int32_t semantic_code,
                               int32_t* out_codebooks) {
    if (!ctx || !out_codebooks)
        return -1;
    const int dim = ctx->hp.fast_dim;
    const int n_cb = ctx->hp.num_codebooks;
    const int cb_size = ctx->hp.codebook_size;

    // Codebook 0 is direct semantic offset
    int32_t c0 = semantic_code - (int32_t)ctx->hp.semantic_begin_id;
    if (c0 < 0 || c0 >= cb_size) {
        c0 = std::clamp(c0, 0, cb_size - 1);
    }
    out_codebooks[0] = c0;

    const float* h0 = slow_hidden ? slow_hidden : ctx->last_slow_normed.data();
    if (!h0 || ctx->last_slow_normed.empty())
        return -1;

    ggml_tensor* fast_emb_w = get_tensor(ctx, "fast_embeddings.weight");
    if (!fast_emb_w)
        return -1;

    // Step position 0 with slow_hidden (primes fast KV cache)
    arktts_step_fast_subnetwork(ctx, h0, 0, nullptr);

    // Step positions 1..n_cb-1 with previous codebook embeddings
    std::vector<float> cb_embed(dim);
    std::vector<float> scores(cb_size);
    for (int pos = 1; pos < n_cb; pos++) {
        int32_t prev_code = out_codebooks[pos - 1];
        dequant_row(fast_emb_w, prev_code, cb_embed.data());

        arktts_step_fast_subnetwork(ctx, cb_embed.data(), pos, scores.data());

        int32_t best_c = 0;
        float best_s = scores[0];
        for (int c = 1; c < cb_size; c++) {
            if (scores[c] > best_s) {
                best_s = scores[c];
                best_c = c;
            }
        }
        out_codebooks[pos] = best_c;
    }

    return 0;
}

static inline ggml_tensor* causal_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K,
                                         int dil = 1) {
    const int pad_left = (K - 1) * dil;
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, Cin)
    if (pad_left > 0) {
        xT = ggml_pad_ext(ctx, xT, pad_left, 0, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor* y = ggml_conv_1d(ctx, w, xT, /*stride=*/1, /*padding=*/0, dil); // (T, Cout)
    const int Cout = (int)w->ne[2];
    const int T_out = (int)x->ne[1];
    y = ggml_reshape_2d(ctx, y, T_out, Cout);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

static inline ggml_tensor* causal_conv1d_dw(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int K,
                                            int dil = 1) {
    const int pad_left = (K - 1) * dil;
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C)
    if (pad_left > 0) {
        xT = ggml_pad_ext(ctx, xT, pad_left, 0, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor* y = ggml_conv_1d_dw(ctx, w, xT, /*stride=*/1, /*padding=*/0, dil);
    const int C = (int)w->ne[2];
    const int T_out = (int)x->ne[1];
    y = ggml_reshape_2d(ctx, y, T_out, C);
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C, T)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

static inline ggml_tensor* causal_convt1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                          int stride) {
    const int K = (int)w->ne[0];
    const int crop_right = K - stride;
    return core_convt::convt1d_crop(ctx, x, w, b, stride, /*crop_left=*/0, /*crop_right=*/crop_right);
}

static inline ggml_tensor* arktts_res_unit(ggml_context* ctx, ggml_tensor* x, ggml_tensor* a0, ggml_tensor* c0_w,
                                           ggml_tensor* c0_b, ggml_tensor* a1, ggml_tensor* c1_w, ggml_tensor* c1_b,
                                           int dil) {
    ggml_tensor* y = core_act::snake_alpha(ctx, x, a0);
    y = causal_conv1d(ctx, y, c0_w, c0_b, 7, dil);
    y = core_act::snake_alpha(ctx, y, a1);
    y = causal_conv1d(ctx, y, c1_w, c1_b, 1, 1);
    return ggml_add(ctx, x, y);
}

static inline ggml_tensor* arktts_convnext(ggml_context* ctx, ggml_tensor* x, ggml_tensor* dw_w, ggml_tensor* dw_b,
                                           ggml_tensor* norm_w, ggml_tensor* norm_b, ggml_tensor* pw1_w,
                                           ggml_tensor* pw1_b, ggml_tensor* pw2_w, ggml_tensor* pw2_b,
                                           ggml_tensor* gamma) {
    ggml_tensor* res = x;
    ggml_tensor* y = causal_conv1d_dw(ctx, x, dw_w, dw_b, 7, 1);
    // LayerNorm over channel dimension (ne[0] = 1024)
    y = ggml_norm(ctx, y, 1e-6f);
    if (norm_w)
        y = ggml_mul(ctx, y, norm_w);
    if (norm_b)
        y = ggml_add(ctx, y, norm_b);

    // pw1: 1024 -> 4096
    y = ggml_mul_mat(ctx, pw1_w, y);
    if (pw1_b)
        y = ggml_add(ctx, y, pw1_b);
    y = ggml_gelu(ctx, y);

    // pw2: 4096 -> 1024
    y = ggml_mul_mat(ctx, pw2_w, y);
    if (pw2_b)
        y = ggml_add(ctx, y, pw2_b);

    // gamma scale
    if (gamma)
        y = ggml_mul(ctx, y, gamma);
    return ggml_add(ctx, res, y);
}

static ggml_tensor* build_quant_sum(ggml_context* gctx, arktts_context* ctx, const int32_t* codes, int n_codebooks,
                                    int n_frames) {
    ggml_tensor* quant_sum = nullptr;
    for (int k = 0; k < n_codebooks; k++) {
        ggml_tensor* code_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_frames);
        int32_t* cdata = (int32_t*)code_ids->data;
        for (int t = 0; t < n_frames; t++) {
            cdata[t] = codes[k * n_frames + t];
        }
        ggml_tensor* cb_w = nullptr;
        ggml_tensor* proj_w = nullptr;
        ggml_tensor* proj_b = nullptr;
        if (k == 0) {
            cb_w = get_tensor(ctx, "codec.quantizer.semantic_quantizer.quantizers.0.codebook.weight");
            proj_w = get_tensor(ctx, "codec.quantizer.semantic_quantizer.quantizers.0.out_proj.weight");
            proj_b = get_tensor(ctx, "codec.quantizer.semantic_quantizer.quantizers.0.out_proj.bias");
        } else {
            char name[128];
            snprintf(name, sizeof(name), "codec.quantizer.quantizer.quantizers.%d.codebook.weight", k - 1);
            cb_w = get_tensor(ctx, name);
            snprintf(name, sizeof(name), "codec.quantizer.quantizer.quantizers.%d.out_proj.weight", k - 1);
            proj_w = get_tensor(ctx, name);
            snprintf(name, sizeof(name), "codec.quantizer.quantizer.quantizers.%d.out_proj.bias", k - 1);
            proj_b = get_tensor(ctx, name);
        }
        if (!cb_w || !proj_w)
            continue;
        ggml_tensor* z = ggml_get_rows(gctx, cb_w, code_ids); // [8, n_frames]
        z = ggml_cast(gctx, z, GGML_TYPE_F32);
        ggml_tensor* w2d = ggml_reshape_2d(gctx, proj_w, 8, 1024);
        ggml_tensor* out_k = ggml_mul_mat(gctx, w2d, z); // [1024, n_frames]
        if (proj_b)
            out_k = ggml_add(gctx, out_k, proj_b);
        quant_sum = quant_sum ? ggml_add(gctx, quant_sum, out_k) : out_k;
    }
    return quant_sum;
}

static ggml_tensor* build_post_module(ggml_context* gctx, arktts_context* ctx, ggml_tensor* cur, int n_frames) {
    ggml_tensor* pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_frames);
    int32_t* pos_data = (int32_t*)pos_ids->data;
    for (int i = 0; i < n_frames; i++) {
        pos_data[i] = i;
    }

    ggml_tensor* mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_frames, n_frames);
    float* mask_data = (float*)mask->data;
    for (int i = 0; i < n_frames; i++) {
        for (int j = 0; j < n_frames; j++) {
            mask_data[i * n_frames + j] = (j <= i && (i - j < 128)) ? 0.0f : -1e9f;
        }
    }

    for (int l = 0; l < 8; l++) {
        char name[128];
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.attention_norm.weight", l);
        ggml_tensor* attn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.attention.wqkv.weight", l);
        ggml_tensor* wqkv = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.attention.wo.weight", l);
        ggml_tensor* wo = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.attention_layer_scale.gamma", l);
        ggml_tensor* attn_scale = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.ffn_norm.weight", l);
        ggml_tensor* ffn_norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.feed_forward.w1.weight", l);
        ggml_tensor* w1 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.feed_forward.w2.weight", l);
        ggml_tensor* w2 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.feed_forward.w3.weight", l);
        ggml_tensor* w3 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.post_module.layers.%d.ffn_layer_scale.gamma", l);
        ggml_tensor* ffn_scale = get_tensor(ctx, name);

        ggml_tensor* an = ggml_rms_norm(gctx, cur, 1e-5f);
        if (attn_norm_w)
            an = ggml_mul(gctx, an, attn_norm_w);

        ggml_tensor* qkv = ggml_mul_mat(gctx, wqkv, an);
        ggml_tensor* q = ggml_view_2d(gctx, qkv, 1024, n_frames, qkv->nb[1], 0);
        ggml_tensor* k = ggml_view_2d(gctx, qkv, 512, n_frames, qkv->nb[1], 1024 * sizeof(float));
        ggml_tensor* v = ggml_view_2d(gctx, qkv, 512, n_frames, qkv->nb[1], (1024 + 512) * sizeof(float));

        q = ggml_reshape_3d(gctx, ggml_cont(gctx, q), 64, 16, n_frames);
        k = ggml_reshape_3d(gctx, ggml_cont(gctx, k), 64, 8, n_frames);
        v = ggml_reshape_3d(gctx, ggml_cont(gctx, v), 64, 8, n_frames);

        q = ggml_rope_ext(gctx, q, pos_ids, nullptr, 64, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        k = ggml_rope_ext(gctx, k, pos_ids, nullptr, 64, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);

        ggml_tensor* K4 = ggml_reshape_4d(gctx, k, 64, 1, 8, n_frames);
        ggml_tensor* V4 = ggml_reshape_4d(gctx, v, 64, 1, 8, n_frames);
        K4 = ggml_repeat_4d(gctx, K4, 64, 2, 8, n_frames);
        V4 = ggml_repeat_4d(gctx, V4, 64, 2, 8, n_frames);
        k = ggml_cont(gctx, ggml_reshape_3d(gctx, K4, 64, 16, n_frames));
        v = ggml_cont(gctx, ggml_reshape_3d(gctx, V4, 64, 16, n_frames));

        q = ggml_permute(gctx, q, 0, 2, 1, 3);
        k = ggml_permute(gctx, k, 0, 2, 1, 3);
        v = ggml_permute(gctx, v, 0, 2, 1, 3);

        ggml_tensor* attn = core_sdpa::attn(gctx, q, k, v, mask, 1024, n_frames, 1.0f / sqrtf(64.0f), false);
        ggml_tensor* attn_out = ggml_mul_mat(gctx, wo, attn);
        if (attn_scale)
            attn_out = ggml_mul(gctx, attn_out, attn_scale);
        cur = ggml_add(gctx, cur, attn_out);

        ggml_tensor* fn = ggml_rms_norm(gctx, cur, 1e-5f);
        if (ffn_norm_w)
            fn = ggml_mul(gctx, fn, ffn_norm_w);

        ggml_tensor* mlp = core_ffn::swiglu(gctx, fn, w1, w3, w2);
        if (ffn_scale)
            mlp = ggml_mul(gctx, mlp, ffn_scale);
        cur = ggml_add(gctx, cur, mlp);
    }

    ggml_tensor* post_norm_w = get_tensor(ctx, "codec.quantizer.post_module.norm.weight");
    cur = ggml_rms_norm(gctx, cur, 1e-5f);
    if (post_norm_w)
        cur = ggml_mul(gctx, cur, post_norm_w);
    return cur;
}

static ggml_tensor* build_upsample(ggml_context* gctx, arktts_context* ctx, ggml_tensor* cur, int n_frames) {
    (void)n_frames;
    for (int s = 0; s < 2; s++) {
        char name[128];
        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.0.conv.weight", s);
        ggml_tensor* up_conv_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.0.conv.bias", s);
        ggml_tensor* up_conv_b = get_tensor(ctx, name);

        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.dwconv.conv.weight", s);
        ggml_tensor* dw_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.dwconv.conv.bias", s);
        ggml_tensor* dw_b = get_tensor(ctx, name);

        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.norm.weight", s);
        ggml_tensor* norm_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.norm.bias", s);
        ggml_tensor* norm_b = get_tensor(ctx, name);

        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.pwconv1.weight", s);
        ggml_tensor* pw1_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.pwconv1.bias", s);
        ggml_tensor* pw1_b = get_tensor(ctx, name);

        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.pwconv2.weight", s);
        ggml_tensor* pw2_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.pwconv2.bias", s);
        ggml_tensor* pw2_b = get_tensor(ctx, name);

        snprintf(name, sizeof(name), "codec.quantizer.upsample.%d.1.gamma", s);
        ggml_tensor* gamma = get_tensor(ctx, name);

        cur = causal_convt1d(gctx, cur, up_conv_w, up_conv_b, 2);
        cur = arktts_convnext(gctx, cur, dw_w, dw_b, norm_w, norm_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma);
    }
    return cur;
}

static ggml_tensor* build_decoder(ggml_context* gctx, arktts_context* ctx, ggml_tensor* cur) {
    // 1. Initial causal Conv1d (1024 -> 1536, k=7)
    ggml_tensor* in_conv_w = get_tensor(ctx, "codec.decoder.model.0.conv.weight");
    ggml_tensor* in_conv_b = get_tensor(ctx, "codec.decoder.model.0.conv.bias");
    cur = causal_conv1d(gctx, cur, in_conv_w, in_conv_b, 7, 1);

    // 2. 4 Decoder blocks (strides 8, 8, 4, 2)
    const int strides[4] = {8, 8, 4, 2};
    for (int b = 0; b < 4; b++) {
        int b_idx = b + 1;
        char name[128];
        snprintf(name, sizeof(name), "codec.decoder.model.%d.block.0.alpha", b_idx);
        ggml_tensor* alpha0 = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.decoder.model.%d.block.1.conv.weight", b_idx);
        ggml_tensor* up_w = get_tensor(ctx, name);
        snprintf(name, sizeof(name), "codec.decoder.model.%d.block.1.conv.bias", b_idx);
        ggml_tensor* up_b = get_tensor(ctx, name);

        cur = core_act::snake_alpha(gctx, cur, alpha0);
        cur = causal_convt1d(gctx, cur, up_w, up_b, strides[b]);

        // 3 Residual units (d = 1, 3, 9)
        const int dils[3] = {1, 3, 9};
        for (int r = 0; r < 3; r++) {
            int r_idx = r + 2;
            snprintf(name, sizeof(name), "codec.decoder.model.%d.block.%d.block.0.alpha", b_idx, r_idx);
            ggml_tensor* ra0 = get_tensor(ctx, name);
            snprintf(name, sizeof(name), "codec.decoder.model.%d.block.%d.block.1.conv.weight", b_idx, r_idx);
            ggml_tensor* rc0_w = get_tensor(ctx, name);
            snprintf(name, sizeof(name), "codec.decoder.model.%d.block.%d.block.1.conv.bias", b_idx, r_idx);
            ggml_tensor* rc0_b = get_tensor(ctx, name);

            snprintf(name, sizeof(name), "codec.decoder.model.%d.block.%d.block.2.alpha", b_idx, r_idx);
            ggml_tensor* ra1 = get_tensor(ctx, name);
            snprintf(name, sizeof(name), "codec.decoder.model.%d.block.%d.block.3.conv.weight", b_idx, r_idx);
            ggml_tensor* rc1_w = get_tensor(ctx, name);
            snprintf(name, sizeof(name), "codec.decoder.model.%d.block.%d.block.3.conv.bias", b_idx, r_idx);
            ggml_tensor* rc1_b = get_tensor(ctx, name);

            cur = arktts_res_unit(gctx, cur, ra0, rc0_w, rc0_b, ra1, rc1_w, rc1_b, dils[r]);
        }
    }

    // 3. Final Snake + Conv1d(96, 1, k=7) + Tanh
    ggml_tensor* out_alpha = get_tensor(ctx, "codec.decoder.model.5.alpha");
    ggml_tensor* out_conv_w = get_tensor(ctx, "codec.decoder.model.6.conv.weight");
    ggml_tensor* out_conv_b = get_tensor(ctx, "codec.decoder.model.6.conv.bias");

    cur = core_act::snake_alpha(gctx, cur, out_alpha);
    cur = causal_conv1d(gctx, cur, out_conv_w, out_conv_b, 7, 1);
    cur = ggml_tanh(gctx, cur);

    const int total_samples = (int)cur->ne[1];
    cur = ggml_reshape_1d(gctx, cur, total_samples);
    return cur;
}

float* arktts_run_quant_sum(struct arktts_context* ctx, const int32_t* codes, int n_codebooks, int n_frames) {
    if (!ctx || !codes || n_codebooks <= 0 || n_frames <= 0)
        return nullptr;
    const size_t ctx_size = (size_t)32 * 1024 * 1024;
    struct ggml_init_params params = {ctx_size, nullptr, false};
    ggml_context* gctx = ggml_init(params);
    if (!gctx)
        return nullptr;

    ggml_tensor* quant_sum = build_quant_sum(gctx, ctx, codes, n_codebooks, n_frames);
    if (!quant_sum) {
        ggml_free(gctx);
        return nullptr;
    }

    ggml_tensor* quant_sum_T = ggml_cont(gctx, ggml_transpose(gctx, quant_sum));

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, quant_sum_T);
    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    const int total = 1024 * n_frames;
    float* out = (float*)malloc(total * sizeof(float));
    if (out) {
        memcpy(out, quant_sum_T->data, total * sizeof(float));
    }
    ggml_free(gctx);
    return out;
}

float* arktts_run_post_module(struct arktts_context* ctx, const float* latents, int n_frames) {
    if (!ctx || !latents || n_frames <= 0)
        return nullptr;
    const size_t ctx_size = (size_t)64 * 1024 * 1024;
    struct ggml_init_params params = {ctx_size, nullptr, false};
    ggml_context* gctx = ggml_init(params);
    if (!gctx)
        return nullptr;

    ggml_tensor* cur_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_frames, 1024);
    memcpy(cur_in->data, latents, (size_t)1024 * n_frames * sizeof(float));
    ggml_tensor* cur = ggml_cont(gctx, ggml_transpose(gctx, cur_in));

    ggml_tensor* post_out = build_post_module(gctx, ctx, cur, n_frames);
    ggml_tensor* post_out_T = ggml_cont(gctx, ggml_transpose(gctx, post_out));

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, post_out_T);
    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    const int total = 1024 * n_frames;
    float* out = (float*)malloc(total * sizeof(float));
    if (out) {
        memcpy(out, post_out_T->data, total * sizeof(float));
    }
    ggml_free(gctx);
    return out;
}

float* arktts_run_upsample(struct arktts_context* ctx, const float* post_out, int n_frames) {
    if (!ctx || !post_out || n_frames <= 0)
        return nullptr;
    const size_t ctx_size = (size_t)64 * 1024 * 1024;
    struct ggml_init_params params = {ctx_size, nullptr, false};
    ggml_context* gctx = ggml_init(params);
    if (!gctx)
        return nullptr;

    ggml_tensor* cur_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_frames, 1024);
    memcpy(cur_in->data, post_out, (size_t)1024 * n_frames * sizeof(float));
    ggml_tensor* cur = ggml_cont(gctx, ggml_transpose(gctx, cur_in));

    ggml_tensor* up_out = build_upsample(gctx, ctx, cur, n_frames);
    ggml_tensor* up_out_T = ggml_cont(gctx, ggml_transpose(gctx, up_out));

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, up_out_T);
    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    const int total = 1024 * n_frames * 4;
    float* out = (float*)malloc(total * sizeof(float));
    if (out) {
        memcpy(out, up_out_T->data, total * sizeof(float));
    }
    ggml_free(gctx);
    return out;
}

float* arktts_decode_codes(struct arktts_context* ctx, const int32_t* codes, int n_codebooks, int n_frames,
                           int* out_samples) {
    if (!ctx || !codes || n_codebooks <= 0 || n_frames <= 0)
        return nullptr;

    const size_t ctx_size = (size_t)(1024 + (size_t)n_frames * 64) * 1024 * 1024;
    struct ggml_init_params params = {ctx_size, nullptr, false};
    ggml_context* gctx = ggml_init(params);
    if (!gctx)
        return nullptr;

    ggml_tensor* quant_sum = build_quant_sum(gctx, ctx, codes, n_codebooks, n_frames);
    if (!quant_sum) {
        ggml_free(gctx);
        return nullptr;
    }
    ggml_tensor* post_out = build_post_module(gctx, ctx, quant_sum, n_frames);
    ggml_tensor* up_out = build_upsample(gctx, ctx, post_out, n_frames);
    ggml_tensor* pcm = build_decoder(gctx, ctx, up_out);

    ggml_cgraph* gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, pcm);

    struct ggml_cplan cplan = ggml_graph_plan(gf, ctx->params.n_threads, nullptr);
    std::vector<uint8_t> work_data(cplan.work_size + 16);
    cplan.work_data = work_data.data();
    ggml_graph_compute(gf, &cplan);

    const int total_samples = (int)pcm->ne[0];
    if (out_samples)
        *out_samples = total_samples;

    float* out = (float*)malloc(total_samples * sizeof(float));
    if (out) {
        memcpy(out, pcm->data, total_samples * sizeof(float));
    }
    ggml_free(gctx);
    return out;
}

static std::vector<int32_t> arktts_tokenize(const arktts_context* ctx, const std::string& text) {
    std::vector<int32_t> ids;
    if (!ctx)
        return ids;

    std::string cleaned;
    bool in_space = false;
    for (char ch : text) {
        if (std::isspace((unsigned char)ch)) {
            if (!in_space && !cleaned.empty()) {
                cleaned.push_back(' ');
                in_space = true;
            }
        } else {
            cleaned.push_back(ch);
            in_space = false;
        }
    }
    while (!cleaned.empty() && std::isspace((unsigned char)cleaned.back())) {
        cleaned.pop_back();
    }
    if (cleaned.empty())
        return ids;

    // Format prompt matching ArkttsProcessor:
    // <|im_start|>system\nconvert the provided text to speech<|im_end|>\n
    // <|im_start|>user\n{cleaned}<|im_end|>\n
    // <|im_start|>assistant\n<|voice|>
    ids.push_back(151644); // <|im_start|>
    for (int32_t tok : {8948, 198, 14166, 279, 3897, 1467, 311, 8806}) {
        ids.push_back(tok);
    }
    ids.push_back(151645); // <|im_end|>
    ids.push_back(198);    // \n

    ids.push_back(151644);           // <|im_start|>
    for (int32_t tok : {872, 198}) { // user\n
        ids.push_back(tok);
    }

    if (!ctx->merge_rank.empty()) {
        for (const std::string& piece : core_bpe::qwen_pretokenize(cleaned)) {
            std::string enc = core_bpe::bytes_to_unicode(piece.data(), piece.size());
            core_bpe::bpe_one(ctx->token_to_id, ctx->merge_rank, enc, ids);
        }
    } else {
        for (const std::string& piece : core_bpe::qwen_pretokenize(cleaned)) {
            auto it = ctx->token_to_id.find(piece);
            if (it != ctx->token_to_id.end()) {
                ids.push_back(it->second);
            }
        }
    }

    ids.push_back(151645); // <|im_end|>
    ids.push_back(198);    // \n

    ids.push_back(151644);             // <|im_start|>
    for (int32_t tok : {77091, 198}) { // assistant\n
        ids.push_back(tok);
    }
    ids.push_back(151673); // <|voice|>

    return ids;
}

float* arktts_synthesize(struct arktts_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    std::vector<int32_t> prompt_ids = arktts_tokenize(ctx, text);
    if (prompt_ids.empty())
        return nullptr;

    const int n_prompt = (int)prompt_ids.size();
    const int dim = ctx->hp.dim;
    const int max_tokens = ctx->params.max_tokens > 0 ? ctx->params.max_tokens : 512;

    std::vector<float> embeds;
    embeds.reserve((size_t)(n_prompt + max_tokens) * dim);

    int emb_dim = 0;
    float* p_embeds = arktts_run_embed(ctx, prompt_ids.data(), n_prompt, &emb_dim);
    if (!p_embeds)
        return nullptr;
    embeds.assign(p_embeds, p_embeds + (size_t)n_prompt * dim);
    free(p_embeds);

    ggml_tensor* emb_w = get_tensor(ctx, "embeddings.weight");
    ggml_tensor* cb_w = get_tensor(ctx, "codebook_embeddings.weight");
    if (!emb_w || !cb_w)
        return nullptr;

    std::vector<std::vector<int32_t>> generated_frames;
    std::vector<int32_t> prev_tokens;
    const int32_t sem_beg = (int32_t)ctx->hp.semantic_begin_id;
    const int32_t sem_end = (int32_t)ctx->hp.semantic_end_id;
    const int32_t eos_id = 151645;

    int vocab_size = 0;
    float* logits = arktts_run_slow_transformer(ctx, embeds.data(), n_prompt, &vocab_size);
    if (!logits)
        return nullptr;

    for (int step = 0; step < max_tokens; step++) {
        int32_t best_tok = sem_beg;
        float best_val = -1e30f;

        if (ctx->params.temperature <= 1e-4f) {
            for (int32_t id = sem_beg; id <= sem_end; id++) {
                if (id < vocab_size && logits[id] > best_val) {
                    best_val = logits[id];
                    best_tok = id;
                }
            }
            if (eos_id < vocab_size && logits[eos_id] > best_val) {
                best_val = logits[eos_id];
                best_tok = eos_id;
            }
        } else {
            std::vector<float> cands;
            std::vector<int32_t> cand_ids;
            float max_l = -1e30f;
            for (int32_t id = sem_beg; id <= sem_end; id++) {
                if (id < vocab_size) {
                    float l = logits[id] / ctx->params.temperature;
                    cands.push_back(l);
                    cand_ids.push_back(id);
                    if (l > max_l)
                        max_l = l;
                }
            }
            if (eos_id < vocab_size) {
                float l = logits[eos_id] / ctx->params.temperature;
                cands.push_back(l);
                cand_ids.push_back(eos_id);
                if (l > max_l)
                    max_l = l;
            }

            float sum_exp = 0.0f;
            for (float& val : cands) {
                val = expf(val - max_l);
                sum_exp += val;
            }
            float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
            for (float& val : cands) {
                val *= inv_sum;
            }

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(ctx->rng);
            float cum = 0.0f;
            best_tok = cand_ids.back();
            for (size_t i = 0; i < cands.size(); i++) {
                cum += cands[i];
                if (r <= cum) {
                    best_tok = cand_ids[i];
                    break;
                }
            }
        }
        free(logits);
        logits = nullptr;

        if (best_tok == eos_id) {
            break;
        }

        auto t0_fast = std::chrono::steady_clock::now();
        std::vector<int32_t> frame_cb(10, 0);
        if (arktts_run_fast_subnetwork(ctx, ctx->last_slow_normed.data(), best_tok, frame_cb.data()) != 0) {
            break;
        }
        auto t1_fast = std::chrono::steady_clock::now();
        generated_frames.push_back(frame_cb);
        prev_tokens.push_back(best_tok);

        std::vector<float> step_embed(dim, 0.0f);
        dequant_row(emb_w, best_tok, step_embed.data());

        std::vector<float> cb_row(dim);
        for (int k = 0; k < 10; k++) {
            int64_t cb_idx = (int64_t)frame_cb[k] + (int64_t)k * ctx->hp.codebook_size;
            dequant_row(cb_w, cb_idx, cb_row.data());
            for (int d = 0; d < dim; d++) {
                step_embed[d] += cb_row[d];
            }
        }

        const int cur_pos = n_prompt + step;
        auto t0_slow = std::chrono::steady_clock::now();
        logits = arktts_step_slow_transformer(ctx, step_embed.data(), cur_pos, &vocab_size);
        auto t1_slow = std::chrono::steady_clock::now();

        if (ctx->params.verbosity >= 1 || true) {
            double ms_fast = std::chrono::duration<double, std::milli>(t1_fast - t0_fast).count();
            double ms_slow = std::chrono::duration<double, std::milli>(t1_slow - t0_slow).count();
            fprintf(stderr, "arktts: frame %d: fast=%.1fms, slow=%.1fms (total=%.1fms)\n", step + 1, ms_fast, ms_slow,
                    ms_fast + ms_slow);
        }

        if (!logits) {
            break;
        }
    }
    if (logits) {
        free(logits);
        logits = nullptr;
    }

    if (generated_frames.empty())
        return nullptr;

    const int n_frames = (int)generated_frames.size();
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "arktts: decoding %d acoustic frames to 44.1 kHz audio...\n", n_frames);
    }
    std::vector<int32_t> flat_codes((size_t)10 * n_frames);
    for (int k = 0; k < 10; k++) {
        for (int t = 0; t < n_frames; t++) {
            flat_codes[(size_t)k * n_frames + t] = generated_frames[t][k];
        }
    }

    return arktts_decode_codes(ctx, flat_codes.data(), 10, n_frames, out_n);
}

void arktts_free_audio(float* pcm) {
    if (pcm)
        free(pcm);
}

void arktts_free(struct arktts_context* ctx) {
    if (!ctx)
        return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    delete ctx;
}
