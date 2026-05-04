// t5_translate.cpp — T5 encoder-decoder translation via ggml
//
// Supports T5ForConditionalGeneration models: MADLAD-400, mT5, Flan-T5.
//
// Architecture differences from M2M-100:
//   - RMSNorm (no bias) instead of LayerNorm
//   - Gated-GELU FFN: gate(x) * up(x) → down (3 weight matrices)
//   - Relative position bias (per-head, bucketed) instead of absolute pos emb
//   - No bias in attention Q/K/V/O projections
//   - Explicit d_kv (head dim) not necessarily d_model/n_heads

#include "t5_translate.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ── Hyperparameters ──────────────────────────────────────────────

struct t5_hparams {
    int vocab_size = 256000;
    int d_model = 1024;
    int d_kv = 128; // head dim
    int d_ff = 8192;
    int n_heads = 16;
    int enc_n_layers = 32;
    int dec_n_layers = 32;
    int rel_attn_num_buckets = 32;
    int rel_attn_max_dist = 128;
    float layer_norm_eps = 1e-6f;
    bool tie_word_embeddings = false;
    std::string ff_proj = "gated-gelu"; // "gated-gelu" or "relu"
    int eos_token_id = 2;
    int pad_token_id = 1;
    int dec_start_token_id = 0;
};

// ── Layer tensors ────────────────────────────────────────────────

struct t5_enc_layer {
    ggml_tensor* attn_q = nullptr; // no bias
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* attn_rms = nullptr;
    // FFN: gated-gelu has gate + up + down
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
    ggml_tensor* ffn_rms = nullptr;
};

struct t5_dec_layer {
    // Self-attention
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* attn_rms = nullptr;
    // Cross-attention
    ggml_tensor* cross_q = nullptr;
    ggml_tensor* cross_k = nullptr;
    ggml_tensor* cross_v = nullptr;
    ggml_tensor* cross_o = nullptr;
    ggml_tensor* cross_rms = nullptr;
    // FFN
    ggml_tensor* ffn_gate = nullptr;
    ggml_tensor* ffn_up = nullptr;
    ggml_tensor* ffn_down = nullptr;
    ggml_tensor* ffn_rms = nullptr;
};

// ── Model ────────────────────────────────────────────────────────

struct t5_model {
    t5_hparams hp;
    ggml_tensor* shared_embed = nullptr;
    ggml_tensor* lm_head = nullptr; // separate if !tie_word_embeddings
    ggml_tensor* enc_final_rms = nullptr;
    ggml_tensor* dec_final_rms = nullptr;
    // Relative position bias: only in layer 0 (shared across layers)
    ggml_tensor* enc_rel_bias = nullptr; // (n_heads, rel_attn_num_buckets)
    ggml_tensor* dec_rel_bias = nullptr;
    std::vector<t5_enc_layer> enc_layers;
    std::vector<t5_dec_layer> dec_layers;
};

// ── Tokenizer ────────────────────────────────────────────────────

struct t5_tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
};

// ── Context ──────────────────────────────────────────────────────

struct t5_translate_context {
    t5_translate_context_params params;
    t5_model model;
    t5_tokenizer tokenizer;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Decoder self-attention KV cache
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    int kv_max_ctx = 0;

    // Cross-attention KV cache
    std::vector<ggml_tensor*> cross_kv_k;
    std::vector<ggml_tensor*> cross_kv_v;
    ggml_context* cross_kv_ctx = nullptr;
    ggml_backend_buffer_t cross_kv_buf = nullptr;
    int cross_T_enc = 0;
};

// ── Helpers ──────────────────────────────────────────────────────

static ggml_tensor* T(t5_translate_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return (it != c->tensors.end()) ? it->second : nullptr;
}

static ggml_tensor* TR(t5_translate_context* c, const char* name) {
    auto* t = T(c, name);
    if (!t)
        fprintf(stderr, "t5: required tensor '%s' not found\n", name);
    return t;
}

// ── T5 relative position bias ────────────────────────────────────
// Bucketed relative positions: maps (query_pos, key_pos) → bucket index
// Used only in layer 0 attention, shared across all layers.

// T5 relative-position bucketing — bit-equivalent port of canonical
// HF transformers _relative_position_bucket.  Convention:
//   rel_pos = key_pos - query_pos
//   bidirectional (encoder): bucket [N/2..N-1] = future (rel_pos > 0),
//     bucket [0..N/2-1] = past/self
//   unidirectional (decoder self-attn): only past survives the causal
//     mask; we work with |past distance|
//
// Earlier version had `int n = -rel_pos; ret += (n < 0) ? 0 : num_buckets`
// which inverted the future/past halves vs canonical T5 — fixed.
static int t5_relative_position_bucket(int rel_pos, bool bidirectional, int num_buckets, int max_dist) {
    int ret = 0;
    if (bidirectional) {
        num_buckets /= 2;
        if (rel_pos > 0)
            ret += num_buckets;
        rel_pos = std::abs(rel_pos);
    } else {
        rel_pos = -std::min(rel_pos, 0);
    }
    const int max_exact = num_buckets / 2;
    if (rel_pos < max_exact) {
        ret += rel_pos;
    } else {
        int val = (int)(max_exact + std::log((float)rel_pos / max_exact) / std::log((float)max_dist / max_exact) *
                                        (num_buckets - max_exact));
        val = std::min(val, num_buckets - 1);
        ret += val;
    }
    return ret;
}

static std::vector<float> compute_rel_pos_bias(ggml_tensor* bias_table, int T_q, int T_k, bool bidirectional,
                                               int num_buckets, int max_dist, int n_heads) {
    // bias_table shape: (num_buckets, n_heads) in GGUF = (n_heads, num_buckets) in row-major
    std::vector<float> table(n_heads * num_buckets);
    ggml_backend_tensor_get(bias_table, table.data(), 0, table.size() * sizeof(float));

    // Output: (n_heads, T_q, T_k) for flash_attn_ext mask
    // But flash_attn_ext wants (T_k, T_q) per head in F16...
    // Actually, we'll add the bias as a separate input tensor.
    std::vector<float> bias(n_heads * T_q * T_k, 0.0f);
    for (int q = 0; q < T_q; q++) {
        for (int k = 0; k < T_k; k++) {
            int bucket = t5_relative_position_bucket(k - q, bidirectional, num_buckets, max_dist);
            for (int h = 0; h < n_heads; h++) {
                // table layout: (n_heads, num_buckets) row-major
                float val = table[h * num_buckets + bucket];
                // output layout: (n_heads, T_q, T_k) row-major
                bias[h * T_q * T_k + q * T_k + k] = val;
            }
        }
    }
    return bias;
}

// ── RMS Norm ─────────────────────────────────────────────────────

static ggml_tensor* t5_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// ── Load metadata ────────────────────────────────────────────────

static void load_metadata(t5_translate_context* c, gguf_context* g) {
    auto& hp = c->model.hp;
    auto get_u32 = [&](const char* key, int def) -> int {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? (int)gguf_get_val_u32(g, idx) : def;
    };
    auto get_str = [&](const char* key, const char* def) -> std::string {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? gguf_get_val_str(g, idx) : def;
    };
    hp.vocab_size = get_u32("t5.vocab_size", 256000);
    hp.d_model = get_u32("t5.d_model", 1024);
    hp.d_kv = get_u32("t5.d_kv", 128);
    hp.d_ff = get_u32("t5.d_ff", 8192);
    hp.n_heads = get_u32("t5.n_heads", 16);
    hp.enc_n_layers = get_u32("t5.encoder.n_layers", 32);
    hp.dec_n_layers = get_u32("t5.decoder.n_layers", 32);
    hp.rel_attn_num_buckets = get_u32("t5.relative_attention_num_buckets", 32);
    hp.rel_attn_max_dist = get_u32("t5.relative_attention_max_distance", 128);
    hp.tie_word_embeddings = get_u32("t5.tie_word_embeddings", 0) != 0;
    hp.eos_token_id = get_u32("t5.eos_token_id", 2);
    hp.pad_token_id = get_u32("t5.pad_token_id", 1);
    hp.dec_start_token_id = get_u32("t5.decoder_start_token_id", 0);
    hp.ff_proj = get_str("t5.feed_forward_proj", "gated-gelu");

    float eps_f = 1e-6f;
    {
        int idx = gguf_find_key(g, "t5.layer_norm_epsilon");
        if (idx >= 0)
            eps_f = gguf_get_val_f32(g, idx);
    }
    hp.layer_norm_eps = eps_f;

    // Load tokenizer
    {
        int tidx = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tidx >= 0) {
            int n = gguf_get_arr_n(g, tidx);
            c->tokenizer.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                c->tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
                c->tokenizer.token_to_id[c->tokenizer.id_to_token[i]] = i;
            }
        }
    }
}

// ── Bind tensors ─────────────────────────────────────────────────

// Try two naming conventions: ours and llama.cpp's
static ggml_tensor* T2(t5_translate_context* c, const char* n1, const char* n2) {
    auto* t = T(c, n1);
    return t ? t : T(c, n2);
}

static bool bind_model(t5_translate_context* c) {
    auto& m = c->model;
    const auto& hp = m.hp;

    // Shared embedding: our name or llama.cpp name
    m.shared_embed = T2(c, "shared.embed.weight", "token_embd.weight");
    if (!m.shared_embed) {
        fprintf(stderr, "t5: required tensor 'shared.embed.weight' / 'token_embd.weight' not found\n");
        return false;
    }

    if (!hp.tie_word_embeddings) {
        m.lm_head = T2(c, "lm_head.weight", "output.weight");
        if (!m.lm_head) {
            fprintf(stderr, "t5: required tensor 'lm_head.weight' / 'output.weight' not found\n");
            return false;
        }
    }

    m.enc_final_rms = T2(c, "enc.final_rms.weight", "enc.output_norm.weight");
    m.dec_final_rms = T2(c, "dec.final_rms.weight", "dec.output_norm.weight");
    // Rel bias: ours stores globally, llama.cpp stores in blk.0
    m.enc_rel_bias = T(c, "enc.rel_bias.weight");
    m.dec_rel_bias = T(c, "dec.rel_bias.weight");

    // Encoder layers
    m.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        auto& l = m.enc_layers[i];
        char buf1[128], buf2[128];
        auto w = [&](const char* ours, const char* llama) -> ggml_tensor* {
            snprintf(buf1, sizeof(buf1), "enc.blk.%d.%s", i, ours);
            snprintf(buf2, sizeof(buf2), "enc.blk.%d.%s", i, llama);
            return T2(c, buf1, buf2);
        };
        l.attn_q = w("attn_q.weight", "attn_q.weight");
        l.attn_k = w("attn_k.weight", "attn_k.weight");
        l.attn_v = w("attn_v.weight", "attn_v.weight");
        l.attn_o = w("attn_o.weight", "attn_o.weight");
        l.attn_rms = w("attn_rms.weight", "attn_norm.weight");
        l.ffn_gate = w("ffn_gate.weight", "ffn_gate.weight");
        l.ffn_up = w("ffn_up.weight", "ffn_up.weight");
        l.ffn_down = w("ffn_down.weight", "ffn_down.weight");
        l.ffn_rms = w("ffn_rms.weight", "ffn_norm.weight");
        // llama.cpp stores rel_bias in blk.0
        if (i == 0 && !m.enc_rel_bias) {
            m.enc_rel_bias = w("attn_rel_b.weight", "attn_rel_b.weight");
        }
    }
    if (!m.enc_final_rms) {
        fprintf(stderr, "t5: enc.final_rms / enc.output_norm not found\n");
        return false;
    }

    // Decoder layers
    m.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; i++) {
        auto& l = m.dec_layers[i];
        char buf1[128], buf2[128];
        auto w = [&](const char* ours, const char* llama) -> ggml_tensor* {
            snprintf(buf1, sizeof(buf1), "dec.blk.%d.%s", i, ours);
            snprintf(buf2, sizeof(buf2), "dec.blk.%d.%s", i, llama);
            return T2(c, buf1, buf2);
        };
        l.attn_q = w("attn_q.weight", "attn_q.weight");
        l.attn_k = w("attn_k.weight", "attn_k.weight");
        l.attn_v = w("attn_v.weight", "attn_v.weight");
        l.attn_o = w("attn_o.weight", "attn_o.weight");
        l.attn_rms = w("attn_rms.weight", "attn_norm.weight");
        l.cross_q = w("cross_q.weight", "cross_attn_q.weight");
        l.cross_k = w("cross_k.weight", "cross_attn_k.weight");
        l.cross_v = w("cross_v.weight", "cross_attn_v.weight");
        l.cross_o = w("cross_o.weight", "cross_attn_o.weight");
        l.cross_rms = w("cross_rms.weight", "cross_attn_norm.weight");
        l.ffn_gate = w("ffn_gate.weight", "ffn_gate.weight");
        l.ffn_up = w("ffn_up.weight", "ffn_up.weight");
        l.ffn_down = w("ffn_down.weight", "ffn_down.weight");
        l.ffn_rms = w("ffn_rms.weight", "ffn_norm.weight");
        if (i == 0 && !m.dec_rel_bias) {
            m.dec_rel_bias = w("attn_rel_b.weight", "attn_rel_b.weight");
        }
    }

    return true;
}

// ── Tokenizer ────────────────────────────────────────────────────

static std::vector<int> tokenize_sp(const t5_tokenizer& tok, const std::string& text) {
    std::vector<int> ids;
    std::vector<std::string> words;
    std::string cur;
    for (char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur += ch;
        }
    }
    if (!cur.empty())
        words.push_back(cur);

    for (size_t wi = 0; wi < words.size(); wi++) {
        std::string word = "\xE2\x96\x81" + words[wi];
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int best_id = -1;
            while (end > start) {
                std::string sub = word.substr(start, end - start);
                auto it = tok.token_to_id.find(sub);
                if (it != tok.token_to_id.end()) {
                    best_id = it->second;
                    break;
                }
                end--;
                while (end > start && (word[end] & 0xC0) == 0x80)
                    end--;
            }
            if (best_id < 0) {
                ids.push_back(2);
                break;
            } // unk → eos as fallback
            ids.push_back(best_id);
            start = end;
        }
    }
    ids.push_back(1); // T5 uses </s> = 1 as EOS
    return ids;
}

static std::string detokenize(const t5_tokenizer& tok, const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id < 0 || id >= (int)tok.id_to_token.size())
            continue;
        const std::string& t = tok.id_to_token[id];
        if (t == "<pad>" || t == "</s>" || t == "<unk>")
            continue;
        std::string decoded = t;
        size_t pos = 0;
        while ((pos = decoded.find("\xe2\x96\x81", pos)) != std::string::npos) {
            decoded.replace(pos, 3, " ");
            pos += 1;
        }
        result += decoded;
    }
    if (!result.empty() && result[0] == ' ')
        result = result.substr(1);
    return result;
}

// ── KV cache allocation ──────────────────────────────────────────

static bool alloc_kv_cache(t5_translate_context* c, int max_ctx) {
    const auto& hp = c->model.hp;
    const int hd = hp.d_kv;
    const int nh = hp.n_heads;
    const int nl = hp.dec_n_layers;

    if (c->kv_buf) {
        ggml_backend_buffer_free(c->kv_buf);
        c->kv_buf = nullptr;
    }
    if (c->kv_ctx) {
        ggml_free(c->kv_ctx);
        c->kv_ctx = nullptr;
    }

    size_t ctx_size = ggml_tensor_overhead() * 2 + 64;
    ggml_init_params params = {ctx_size, nullptr, true};
    c->kv_ctx = ggml_init(params);
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");
    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf)
        return false;
    ggml_backend_buffer_clear(c->kv_buf, 0);
    c->kv_max_ctx = max_ctx;
    return true;
}

static bool alloc_cross_kv(t5_translate_context* c, int T_enc) {
    const auto& hp = c->model.hp;
    const int hd = hp.d_kv;
    const int nh = hp.n_heads;
    const int nl = hp.dec_n_layers;

    if (c->cross_kv_buf) {
        ggml_backend_buffer_free(c->cross_kv_buf);
        c->cross_kv_buf = nullptr;
    }
    if (c->cross_kv_ctx) {
        ggml_free(c->cross_kv_ctx);
        c->cross_kv_ctx = nullptr;
    }

    size_t ctx_size = ggml_tensor_overhead() * nl * 2 + 64;
    ggml_init_params params = {ctx_size, nullptr, true};
    c->cross_kv_ctx = ggml_init(params);
    c->cross_kv_k.resize(nl);
    c->cross_kv_v.resize(nl);
    for (int i = 0; i < nl; i++) {
        c->cross_kv_k[i] = ggml_new_tensor_3d(c->cross_kv_ctx, GGML_TYPE_F32, hd, T_enc, nh);
        c->cross_kv_v[i] = ggml_new_tensor_3d(c->cross_kv_ctx, GGML_TYPE_F32, hd, T_enc, nh);
    }
    c->cross_kv_buf = ggml_backend_alloc_ctx_tensors(c->cross_kv_ctx, c->backend);
    if (!c->cross_kv_buf)
        return false;
    c->cross_T_enc = T_enc;
    return true;
}

// ── Encoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_encoder_graph(t5_translate_context* c, int T) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.n_heads;
    const int hd = hp.d_kv;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(inp, "enc_tokens");
    ggml_set_input(inp);

    // Position bucket indices (precomputed on CPU): (T_q, T_k) i32
    ggml_tensor* pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, T, T);
    ggml_set_name(pos_bucket, "enc_pos_bucket");
    ggml_set_input(pos_bucket);

    // Compute position bias in-graph: get_rows from bias table, then reshape+permute
    // bias table: (n_heads, num_buckets) → get_rows with flattened bucket indices
    ggml_tensor* pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, T * T);
    ggml_tensor* pos_bias = ggml_get_rows(ctx0, m.enc_rel_bias, pos_bucket_1d);
    // pos_bias: (n_heads, T*T) → reshape to (n_heads, T_k, T_q) → permute to (T_k, T_q, n_heads)
    pos_bias = ggml_reshape_3d(ctx0, pos_bias, nh, T, T);
    pos_bias = ggml_cont(ctx0, ggml_permute(ctx0, pos_bias, 2, 0, 1, 3)); // (T_k, T_q, nh)

    // Embedding (no positional — T5 uses relative position bias)
    ggml_tensor* cur = ggml_get_rows(ctx0, m.shared_embed, inp);

    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& l = m.enc_layers[il];
        ggml_tensor* residual = cur;

        cur = t5_rms_norm(ctx0, cur, l.attn_rms, hp.layer_norm_eps);

        // Self-attention: manual QK + bias + softmax + V (no flash_attn — bias not supported)
        ggml_tensor* Q = ggml_mul_mat(ctx0, l.attn_q, cur); // (nh*hd, T)
        ggml_tensor* K = ggml_mul_mat(ctx0, l.attn_k, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, l.attn_v, cur);

        // Reshape to (hd, nh, T) then permute to (hd, T, nh)
        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3);
        K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3);
        V = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3);

        // KQ = K^T @ Q → (T_k, T_q, nh) — ggml_mul_mat contracts ne[0]
        ggml_tensor* kq = ggml_mul_mat(ctx0, K, Q); // (T, T, nh)
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        // Add position bias (no causal mask for encoder — bidirectional)
        kq = ggml_add(ctx0, kq, pos_bias);

        // Softmax (scale=1.0 for T5, no causal mask)
        kq = ggml_soft_max(ctx0, kq);

        // V @ softmax(KQ) → (hd, T_q, nh)
        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, V)); // (T, hd, nh)
        ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);              // (hd, T, nh)

        // Permute back and reshape: (hd, nh, T) → (nh*hd, T)
        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, T);

        cur = ggml_mul_mat(ctx0, l.attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // FFN
        residual = cur;
        cur = t5_rms_norm(ctx0, cur, l.ffn_rms, hp.layer_norm_eps);

        ggml_tensor* gate = ggml_gelu(ctx0, ggml_mul_mat(ctx0, l.ffn_gate, cur));
        ggml_tensor* up = ggml_mul_mat(ctx0, l.ffn_up, cur);
        cur = ggml_mul_mat(ctx0, l.ffn_down, ggml_mul(ctx0, gate, up));

        cur = ggml_add(ctx0, cur, residual);
    }

    cur = t5_rms_norm(ctx0, cur, m.enc_final_rms, hp.layer_norm_eps);

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ── Cross-KV computation ─────────────────────────────────────────

static bool compute_cross_kv(t5_translate_context* c, const float* enc_out, int T_enc) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.n_heads;
    const int hd = hp.d_kv;

    if (!alloc_cross_kv(c, T_enc))
        return false;

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& l = m.dec_layers[il];

        ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

        ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        ggml_set_name(enc_inp, "enc_for_cross");
        ggml_set_input(enc_inp);

        ggml_tensor* K = ggml_mul_mat(ctx0, l.cross_k, enc_inp);
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(K, "cross_k");

        ggml_tensor* V = ggml_mul_mat(ctx0, l.cross_v, enc_inp);
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(V, "cross_v");

        ggml_build_forward_expand(gf, K);
        ggml_build_forward_expand(gf, V);

        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
            ggml_free(ctx0);
            return false;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_for_cross"), enc_out, 0,
                                (size_t)D * T_enc * sizeof(float));
        if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx0);
            return false;
        }

        size_t sz = (size_t)hd * T_enc * nh * sizeof(float);
        std::vector<float> buf(hd * T_enc * nh);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "cross_k"), buf.data(), 0, sz);
        ggml_backend_tensor_set(c->cross_kv_k[il], buf.data(), 0, sz);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "cross_v"), buf.data(), 0, sz);
        ggml_backend_tensor_set(c->cross_kv_v[il], buf.data(), 0, sz);
        ggml_free(ctx0);
    }
    return true;
}

// ── Decoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_decoder_graph(t5_translate_context* c, int n_tokens, int offset) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.n_heads;
    const int hd = hp.d_kv;
    const int Lk = offset + n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp, "dec_tokens");
    ggml_set_input(inp);

    // Position bucket indices for self-attention: (Lk, n_tokens) i32
    ggml_tensor* pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, Lk, n_tokens);
    ggml_set_name(pos_bucket, "dec_pos_bucket");
    ggml_set_input(pos_bucket);

    // Causal mask: (Lk, n_tokens) F32
    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Position bias in-graph
    ggml_tensor* pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, (int64_t)Lk * n_tokens);
    ggml_tensor* dec_pos_bias = ggml_get_rows(ctx0, m.dec_rel_bias, pos_bucket_1d);
    dec_pos_bias = ggml_reshape_3d(ctx0, dec_pos_bias, nh, Lk, n_tokens);
    dec_pos_bias = ggml_cont(ctx0, ggml_permute(ctx0, dec_pos_bias, 2, 0, 1, 3)); // (Lk, n_tokens, nh)

    // Embedding
    ggml_tensor* cur = ggml_get_rows(ctx0, m.shared_embed, inp);

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& l = m.dec_layers[il];
        ggml_tensor* residual = cur;

        // ---- Self-attention (manual, with rel-pos bias) ----
        cur = t5_rms_norm(ctx0, cur, l.attn_rms, hp.layer_norm_eps);

        ggml_tensor* Q = ggml_mul_mat(ctx0, l.attn_q, cur);
        ggml_tensor* K = ggml_mul_mat(ctx0, l.attn_k, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, l.attn_v, cur);

        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, n_tokens), 0, 2, 1, 3);
        ggml_tensor* K_new = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, n_tokens), 0, 2, 1, 3);
        ggml_tensor* V_new = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, n_tokens), 0, 2, 1, 3);

        // Write to KV cache
        ggml_tensor* k_view =
            ggml_view_4d(ctx0, c->kv_k, hd, n_tokens, nh, 1, c->kv_k->nb[1], c->kv_k->nb[2], c->kv_k->nb[3],
                         (size_t)il * c->kv_k->nb[3] + (size_t)offset * c->kv_k->nb[1]);
        ggml_tensor* v_view =
            ggml_view_4d(ctx0, c->kv_v, hd, n_tokens, nh, 1, c->kv_v->nb[1], c->kv_v->nb[2], c->kv_v->nb[3],
                         (size_t)il * c->kv_v->nb[3] + (size_t)offset * c->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new, v_view));

        // Read full KV history and cast from F16 to F32
        ggml_tensor* Kfull = ggml_cast(
            ctx0, ggml_view_3d(ctx0, c->kv_k, hd, Lk, nh, c->kv_k->nb[1], c->kv_k->nb[2], (size_t)il * c->kv_k->nb[3]),
            GGML_TYPE_F32);
        ggml_tensor* Vfull = ggml_cast(
            ctx0, ggml_view_3d(ctx0, c->kv_v, hd, Lk, nh, c->kv_v->nb[1], c->kv_v->nb[2], (size_t)il * c->kv_v->nb[3]),
            GGML_TYPE_F32);

        // Manual attention: KQ + pos_bias + causal_mask → softmax → V
        ggml_tensor* kq = ggml_mul_mat(ctx0, Kfull, Q); // (Lk, n_tokens, nh)
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_add(ctx0, kq, dec_pos_bias);
        if (causal_mask)
            kq = ggml_add(ctx0, kq, causal_mask); // broadcast across heads
        kq = ggml_soft_max(ctx0, kq);

        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, Vfull));
        ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq); // (hd, n_tokens, nh)

        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, n_tokens);
        cur = ggml_mul_mat(ctx0, l.attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // ---- Cross-attention (no position bias, no causal mask) ----
        residual = cur;
        cur = t5_rms_norm(ctx0, cur, l.cross_rms, hp.layer_norm_eps);

        ggml_tensor* CQ = ggml_mul_mat(ctx0, l.cross_q, cur);
        CQ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, CQ, hd, nh, n_tokens), 0, 2, 1, 3);

        ggml_tensor* CK = c->cross_kv_k[il];
        ggml_tensor* CV = c->cross_kv_v[il];

        ggml_tensor* ca_kq = ggml_mul_mat(ctx0, CK, CQ);
        ggml_mul_mat_set_prec(ca_kq, GGML_PREC_F32);
        ca_kq = ggml_soft_max(ctx0, ca_kq);

        ggml_tensor* cv_t = ggml_cont(ctx0, ggml_transpose(ctx0, CV));
        ggml_tensor* ca_kqv = ggml_mul_mat(ctx0, cv_t, ca_kq);

        cur = ggml_cont(ctx0, ggml_permute(ctx0, ca_kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, n_tokens);
        cur = ggml_mul_mat(ctx0, l.cross_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // ---- FFN ----
        residual = cur;
        cur = t5_rms_norm(ctx0, cur, l.ffn_rms, hp.layer_norm_eps);

        ggml_tensor* gate = ggml_gelu(ctx0, ggml_mul_mat(ctx0, l.ffn_gate, cur));
        ggml_tensor* up = ggml_mul_mat(ctx0, l.ffn_up, cur);
        cur = ggml_mul_mat(ctx0, l.ffn_down, ggml_mul(ctx0, gate, up));

        cur = ggml_add(ctx0, cur, residual);
    }

    cur = t5_rms_norm(ctx0, cur, m.dec_final_rms, hp.layer_norm_eps);

    if (n_tokens > 1)
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);

    ggml_tensor* head = hp.tie_word_embeddings ? m.shared_embed : m.lm_head;
    cur = ggml_mul_mat(ctx0, head, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ── Run encoder ──────────────────────────────────────────────────

static std::vector<float> run_encoder(t5_translate_context* c, const std::vector<int>& token_ids) {
    const int T = (int)token_ids.size();
    const int D = c->model.hp.d_model;
    const auto& hp = c->model.hp;

    ggml_cgraph* gf = build_encoder_graph(c, T);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return {};

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_tokens"), token_ids.data(), 0, T * sizeof(int32_t));

    // Compute position bucket indices for encoder (bidirectional)
    std::vector<int32_t> buckets(T * T);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++)
            buckets[q * T + k] =
                t5_relative_position_bucket(k - q, true, hp.rel_attn_num_buckets, hp.rel_attn_max_dist);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_pos_bucket"), buckets.data(), 0,
                            buckets.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return {};

    std::vector<float> enc_out(T * D);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "enc_out"), enc_out.data(), 0, enc_out.size() * sizeof(float));
    return enc_out;
}

// ── Run decoder step ─────────────────────────────────────────────

static std::vector<float> run_decoder_step(t5_translate_context* c, const int* tokens, int n_tokens, int offset) {
    const auto& hp = c->model.hp;
    const int Lk = offset + n_tokens;

    ggml_cgraph* gf = build_decoder_graph(c, n_tokens, offset);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf))
        return {};

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_tokens"), tokens, 0, n_tokens * sizeof(int32_t));

    // Compute position bucket indices for decoder (causal / unidirectional)
    std::vector<int32_t> buckets((size_t)n_tokens * Lk);
    for (int q = 0; q < n_tokens; q++)
        for (int k = 0; k < Lk; k++)
            buckets[q * Lk + k] =
                t5_relative_position_bucket(k - (offset + q), false, hp.rel_attn_num_buckets, hp.rel_attn_max_dist);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_pos_bucket"), buckets.data(), 0,
                            buckets.size() * sizeof(int32_t));

    // Causal mask: 0 for visible, -inf for future
    if (n_tokens > 1) {
        std::vector<float> mask((size_t)n_tokens * Lk, 0.0f);
        for (int q = 0; q < n_tokens; q++)
            for (int k = offset + q + 1; k < Lk; k++)
                mask[q * Lk + k] = -INFINITY;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0, mask.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS)
        return {};

    std::vector<float> out(hp.vocab_size);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), out.data(), 0, hp.vocab_size * sizeof(float));
    return out;
}

// ── Public API ───────────────────────────────────────────────────

extern "C" struct t5_translate_context_params t5_translate_context_default_params(void) {
    t5_translate_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct t5_translate_context* t5_translate_init_from_file(const char* path_model,
                                                                    struct t5_translate_context_params params) {
    auto* c = new t5_translate_context();
    c->params = params;

    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    const auto& hp = c->model.hp;
    if (params.verbosity >= 1) {
        fprintf(stderr, "t5: d=%d d_kv=%d enc=%dL dec=%dL heads=%d ff=%d vocab=%d ff_proj=%s\n", hp.d_model, hp.d_kv,
                hp.enc_n_layers, hp.dec_n_layers, hp.n_heads, hp.d_ff, hp.vocab_size, hp.ff_proj.c_str());
    }

    c->backend_cpu = ggml_backend_cpu_init();
    c->backend = c->backend_cpu;

    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path_model, c->backend, "t5", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    if (!bind_model(c)) {
        delete c;
        return nullptr;
    }

    {
        ggml_backend_t backends[] = {c->backend};
        c->sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    return c;
}

extern "C" void t5_translate_free(struct t5_translate_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cross_kv_buf)
        ggml_backend_buffer_free(ctx->cross_kv_buf);
    if (ctx->cross_kv_ctx)
        ggml_free(ctx->cross_kv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" char* t5_translate(struct t5_translate_context* ctx, const char* text, int max_new_tokens) {
    if (!ctx || !text)
        return nullptr;
    if (max_new_tokens <= 0)
        max_new_tokens = 200;

    const auto& hp = ctx->model.hp;

    // Tokenize (MADLAD: text already contains "<2xx> " prefix)
    std::vector<int> enc_ids = tokenize_sp(ctx->tokenizer, text);
    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "t5: input %zu tokens\n", enc_ids.size());
    }

    // Encode
    std::vector<float> enc_out = run_encoder(ctx, enc_ids);
    if (enc_out.empty())
        return nullptr;
    int T_enc = (int)enc_ids.size();
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "t5: encoder done, T_enc=%d\n", T_enc);

    // Cross-KV
    if (!compute_cross_kv(ctx, enc_out.data(), T_enc))
        return nullptr;

    // KV cache
    if (!alloc_kv_cache(ctx, max_new_tokens + 4))
        return nullptr;

    // Greedy decode
    std::vector<int> dec_ids;
    dec_ids.push_back(hp.dec_start_token_id); // <pad> = 0

    std::vector<float> logits = run_decoder_step(ctx, dec_ids.data(), (int)dec_ids.size(), 0);
    if (logits.empty())
        return nullptr;
    int offset = (int)dec_ids.size();

    for (int step = 0; step < max_new_tokens; step++) {
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < hp.vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        if (best_id == hp.eos_token_id)
            break;
        dec_ids.push_back(best_id);

        if (ctx->params.verbosity >= 2) {
            fprintf(stderr, "t5[dec]: step=%d tok=%d '%s'\n", step, best_id,
                    best_id < (int)ctx->tokenizer.id_to_token.size() ? ctx->tokenizer.id_to_token[best_id].c_str()
                                                                     : "?");
        }

        logits = run_decoder_step(ctx, &best_id, 1, offset);
        if (logits.empty())
            return nullptr;
        offset++;
    }

    std::string result = detokenize(ctx->tokenizer, dec_ids);
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "t5: translated → '%s'\n", result.c_str());

    char* out = (char*)malloc(result.size() + 1);
    std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}
