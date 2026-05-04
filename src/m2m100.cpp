// m2m100.cpp — M2M-100 machine translation via ggml
//
// Architecture:
//   Encoder: 12-layer transformer (self-attention + FFN, pre-norm, ReLU)
//   Decoder: 12-layer transformer with cross-attention + KV cache
//   Shared embedding: encoder, decoder, and lm_head share one table
//   Sinusoidal positional embeddings (pre-computed, stored in GGUF)
//
// Supports facebook/m2m100_418M, m2m100_1.2B, and wmt21-dense-24-wide
// (same architecture at different scales).

#include "m2m100.h"
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

struct m2m100_hparams {
    int vocab_size = 128112;
    int d_model = 1024;
    int enc_n_layers = 12;
    int enc_n_heads = 16;
    int enc_ffn_dim = 4096;
    int dec_n_layers = 12;
    int dec_n_heads = 16;
    int dec_ffn_dim = 4096;
    int max_position_emb = 1024;
    bool scale_embedding = true;
    int bos_token_id = 0;
    int eos_token_id = 2;
    int pad_token_id = 1;
    int dec_start_token = 2;
    int head_dim() const { return d_model / enc_n_heads; }
};

// ── Encoder layer tensors ────────────────────────────────────────

struct m2m100_enc_layer {
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
};

// ── Decoder layer tensors ────────────────────────────────────────

struct m2m100_dec_layer {
    // Self-attention
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    // Cross-attention
    ggml_tensor* cross_q_w = nullptr;
    ggml_tensor* cross_q_b = nullptr;
    ggml_tensor* cross_k_w = nullptr;
    ggml_tensor* cross_k_b = nullptr;
    ggml_tensor* cross_v_w = nullptr;
    ggml_tensor* cross_v_b = nullptr;
    ggml_tensor* cross_o_w = nullptr;
    ggml_tensor* cross_o_b = nullptr;
    ggml_tensor* cross_ln_w = nullptr;
    ggml_tensor* cross_ln_b = nullptr;
    // FFN
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
};

// ── Model ────────────────────────────────────────────────────────

struct m2m100_model {
    m2m100_hparams hp;

    // Shared embedding (encoder + decoder + lm_head)
    ggml_tensor* shared_embed = nullptr;

    // Positional embeddings (pre-computed sinusoidal)
    ggml_tensor* enc_pos_emb = nullptr;
    ggml_tensor* dec_pos_emb = nullptr;

    // Encoder
    std::vector<m2m100_enc_layer> enc_layers;
    ggml_tensor* enc_out_ln_w = nullptr;
    ggml_tensor* enc_out_ln_b = nullptr;

    // Decoder
    std::vector<m2m100_dec_layer> dec_layers;
    ggml_tensor* dec_out_ln_w = nullptr;
    ggml_tensor* dec_out_ln_b = nullptr;
    // lm_head shares shared_embed (tied weights)
};

// ── Tokenizer ────────────────────────────────────────────────────

struct m2m100_tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    // Language code → token ID
    std::vector<std::string> lang_codes;
    std::map<std::string, int> lang_to_token_id;
};

// ── Context ──────────────────────────────────────────────────────

struct m2m100_context {
    m2m100_context_params params;
    m2m100_model model;
    m2m100_tokenizer tokenizer;

    // Weight storage
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    // Compute
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for decoder self-attention
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    int kv_max_ctx = 0;

    // Cross-attention KV cache (computed once per encoder run)
    std::vector<ggml_tensor*> cross_kv_k;
    std::vector<ggml_tensor*> cross_kv_v;
    ggml_context* cross_kv_ctx = nullptr;
    ggml_backend_buffer_t cross_kv_buf = nullptr;
    int cross_T_enc = 0;
};

// ── Helpers ──────────────────────────────────────────────────────

static ggml_tensor* T(m2m100_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return (it != c->tensors.end()) ? it->second : nullptr;
}

static ggml_tensor* TR(m2m100_context* c, const char* name) {
    auto* t = T(c, name);
    if (!t) {
        fprintf(stderr, "m2m100: required tensor '%s' not found\n", name);
    }
    return t;
}

// ── Load metadata ────────────────────────────────────────────────

static void load_metadata(m2m100_context* c, gguf_context* g) {
    auto& hp = c->model.hp;
    auto get_u32 = [&](const char* key, int def) -> int {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? (int)gguf_get_val_u32(g, idx) : def;
    };
    hp.vocab_size = get_u32("m2m100.vocab_size", 128112);
    hp.d_model = get_u32("m2m100.d_model", 1024);
    hp.enc_n_layers = get_u32("m2m100.encoder.n_layers", 12);
    hp.enc_n_heads = get_u32("m2m100.encoder.n_heads", 16);
    hp.enc_ffn_dim = get_u32("m2m100.encoder.ffn_dim", 4096);
    hp.dec_n_layers = get_u32("m2m100.decoder.n_layers", 12);
    hp.dec_n_heads = get_u32("m2m100.decoder.n_heads", 16);
    hp.dec_ffn_dim = get_u32("m2m100.decoder.ffn_dim", 4096);
    hp.max_position_emb = get_u32("m2m100.max_position_embeddings", 1024);
    hp.scale_embedding = get_u32("m2m100.scale_embedding", 1) != 0;
    hp.bos_token_id = get_u32("m2m100.bos_token_id", 0);
    hp.eos_token_id = get_u32("m2m100.eos_token_id", 2);
    hp.pad_token_id = get_u32("m2m100.pad_token_id", 1);
    hp.dec_start_token = get_u32("m2m100.decoder_start_token_id", 2);

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

    // Load language codes
    {
        int cidx = gguf_find_key(g, "m2m100.lang_codes");
        int iidx = gguf_find_key(g, "m2m100.lang_token_ids");
        if (cidx >= 0 && iidx >= 0) {
            int n = gguf_get_arr_n(g, cidx);
            c->tokenizer.lang_codes.resize(n);
            for (int i = 0; i < n; i++) {
                std::string code = gguf_get_arr_str(g, cidx, i);
                c->tokenizer.lang_codes[i] = code;
                // Read corresponding token ID
                uint32_t tok_id = ((const uint32_t*)gguf_get_arr_data(g, iidx))[i];
                c->tokenizer.lang_to_token_id[code] = (int)tok_id;
            }
        }
    }
}

// ── Bind tensors ─────────────────────────────────────────────────

static bool bind_model(m2m100_context* c) {
    auto& m = c->model;
    const auto& hp = m.hp;

    m.shared_embed = TR(c, "shared.embed.weight");
    m.enc_pos_emb = TR(c, "enc.pos_emb");
    m.dec_pos_emb = TR(c, "dec.pos_emb");
    if (!m.shared_embed || !m.enc_pos_emb || !m.dec_pos_emb)
        return false;

    // Encoder layers
    m.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        auto& l = m.enc_layers[i];
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%d.%s", i, suffix);
            return T(c, buf);
        };
        l.attn_q_w = w("attn_q.weight");
        l.attn_q_b = w("attn_q.bias");
        l.attn_k_w = w("attn_k.weight");
        l.attn_k_b = w("attn_k.bias");
        l.attn_v_w = w("attn_v.weight");
        l.attn_v_b = w("attn_v.bias");
        l.attn_o_w = w("attn_o.weight");
        l.attn_o_b = w("attn_o.bias");
        l.attn_ln_w = w("attn_ln.weight");
        l.attn_ln_b = w("attn_ln.bias");
        l.ffn_up_w = w("ffn_up.weight");
        l.ffn_up_b = w("ffn_up.bias");
        l.ffn_down_w = w("ffn_down.weight");
        l.ffn_down_b = w("ffn_down.bias");
        l.ffn_ln_w = w("ffn_ln.weight");
        l.ffn_ln_b = w("ffn_ln.bias");
    }
    m.enc_out_ln_w = TR(c, "enc.out_ln.weight");
    m.enc_out_ln_b = TR(c, "enc.out_ln.bias");

    // Decoder layers
    m.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; i++) {
        auto& l = m.dec_layers[i];
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "dec.blk.%d.%s", i, suffix);
            return T(c, buf);
        };
        l.attn_q_w = w("attn_q.weight");
        l.attn_q_b = w("attn_q.bias");
        l.attn_k_w = w("attn_k.weight");
        l.attn_k_b = w("attn_k.bias");
        l.attn_v_w = w("attn_v.weight");
        l.attn_v_b = w("attn_v.bias");
        l.attn_o_w = w("attn_o.weight");
        l.attn_o_b = w("attn_o.bias");
        l.attn_ln_w = w("attn_ln.weight");
        l.attn_ln_b = w("attn_ln.bias");
        l.cross_q_w = w("cross_q.weight");
        l.cross_q_b = w("cross_q.bias");
        l.cross_k_w = w("cross_k.weight");
        l.cross_k_b = w("cross_k.bias");
        l.cross_v_w = w("cross_v.weight");
        l.cross_v_b = w("cross_v.bias");
        l.cross_o_w = w("cross_o.weight");
        l.cross_o_b = w("cross_o.bias");
        l.cross_ln_w = w("cross_ln.weight");
        l.cross_ln_b = w("cross_ln.bias");
        l.ffn_up_w = w("ffn_up.weight");
        l.ffn_up_b = w("ffn_up.bias");
        l.ffn_down_w = w("ffn_down.weight");
        l.ffn_down_b = w("ffn_down.bias");
        l.ffn_ln_w = w("ffn_ln.weight");
        l.ffn_ln_b = w("ffn_ln.bias");
    }
    m.dec_out_ln_w = TR(c, "dec.out_ln.weight");
    m.dec_out_ln_b = TR(c, "dec.out_ln.bias");

    return true;
}

// ── Tokenizer encode ─────────────────────────────────────────────
// Simple greedy longest-match tokenization against the vocab.
// For production use, this should be a proper SentencePiece BPE decoder,
// but longest-match works for basic testing with the M2M-100 vocab.

static std::vector<int> tokenize(const m2m100_tokenizer& tok, const std::string& text, const std::string& src_lang) {
    std::vector<int> ids;

    // Prepend source language token
    auto it = tok.lang_to_token_id.find(src_lang);
    if (it != tok.lang_to_token_id.end()) {
        ids.push_back(it->second);
    }

    // SentencePiece-style greedy longest-match tokenization with ▁ prefix
    // Split on whitespace, prefix each word with ▁, greedy longest match
    std::vector<std::string> words;
    std::string cur;
    for (char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
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
        // Prefix with ▁ (U+2581, 0xE2 0x96 0x81)
        std::string word = "\xE2\x96\x81" + words[wi];
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int best_id = -1;
            while (end > start) {
                std::string sub = word.substr(start, end - start);
                auto it2 = tok.token_to_id.find(sub);
                if (it2 != tok.token_to_id.end()) {
                    best_id = it2->second;
                    break;
                }
                end--;
                // Don't break mid-UTF8
                while (end > start && (word[end] & 0xC0) == 0x80)
                    end--;
            }
            if (best_id < 0) {
                ids.push_back(3); // <unk>
                break;
            }
            ids.push_back(best_id);
            start = end;
        }
    }

    // Append EOS
    ids.push_back(2); // </s>

    return ids;
}

static std::string detokenize(const m2m100_tokenizer& tok, const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id < 0 || id >= (int)tok.id_to_token.size())
            continue;
        const std::string& t = tok.id_to_token[id];
        if (t == "<s>" || t == "</s>" || t == "<pad>" || t == "<unk>")
            continue;
        // Skip language tokens (__XX__)
        if (t.size() >= 5 && t[0] == '_' && t[1] == '_' && t[t.size() - 1] == '_' && t[t.size() - 2] == '_')
            continue;
        // SentencePiece: '▁' → space
        std::string decoded = t;
        size_t pos = 0;
        while ((pos = decoded.find("\xe2\x96\x81", pos)) != std::string::npos) {
            decoded.replace(pos, 3, " ");
            pos += 1;
        }
        result += decoded;
    }
    // Trim leading space
    if (!result.empty() && result[0] == ' ')
        result = result.substr(1);
    return result;
}

// ── KV cache allocation ──────────────────────────────────────────

static bool alloc_kv_cache(m2m100_context* c, int max_ctx) {
    const auto& hp = c->model.hp;
    const int hd = hp.head_dim();
    const int nh = hp.dec_n_heads;
    const int nl = hp.dec_n_layers;

    size_t n_tensors = 2; // k, v
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + 64;
    ggml_init_params params = {ctx_size, nullptr, true};
    c->kv_ctx = ggml_init(params);

    // Shape: (head_dim, max_ctx, n_heads, n_layers)
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");

    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf)
        return false;

    // Zero-init
    ggml_backend_buffer_clear(c->kv_buf, 0);
    c->kv_max_ctx = max_ctx;
    return true;
}

// ── Cross-attention KV cache ─────────────────────────────────────

static bool alloc_cross_kv(m2m100_context* c, int T_enc) {
    const auto& hp = c->model.hp;
    const int hd = hp.head_dim();
    const int nh = hp.dec_n_heads;
    const int nl = hp.dec_n_layers;

    // Free existing
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
        char name[64];
        snprintf(name, sizeof(name), "cross_k_%d", i);
        ggml_set_name(c->cross_kv_k[i], name);
        snprintf(name, sizeof(name), "cross_v_%d", i);
        ggml_set_name(c->cross_kv_v[i], name);
    }

    c->cross_kv_buf = ggml_backend_alloc_ctx_tensors(c->cross_kv_ctx, c->backend);
    if (!c->cross_kv_buf)
        return false;

    c->cross_T_enc = T_enc;
    return true;
}

// ── Encoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_encoder_graph(m2m100_context* c, int T) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.enc_n_heads;
    const int hd = hp.head_dim();
    const float scale = 1.0f / sqrtf((float)hd);
    const float emb_scale = hp.scale_embedding ? sqrtf((float)D) : 1.0f;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input token IDs
    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(inp, "enc_tokens");
    ggml_set_input(inp);

    // Position IDs
    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos, "enc_positions");
    ggml_set_input(pos);

    // Embedding: shared_embed[tokens] * scale + pos_emb[positions]
    ggml_tensor* cur = ggml_get_rows(ctx0, m.shared_embed, inp);
    if (emb_scale != 1.0f) {
        cur = ggml_scale(ctx0, cur, emb_scale);
    }
    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, m.enc_pos_emb, pos));

    // Encoder layers
    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& l = m.enc_layers[il];
        ggml_tensor* residual = cur;

        // Pre-norm for self-attention
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, l.attn_ln_w);
        cur = ggml_add(ctx0, cur, l.attn_ln_b);

        // Self-attention Q, K, V
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_q_w, cur), l.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_k_w, cur), l.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_v_w, cur), l.attn_v_b);

        // Reshape for multi-head: (D, T) → (hd, nh, T) → permute to (hd, T, nh)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3));

        // Flash attention (no causal mask for encoder)
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        // Output projection + residual
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_o_w, attn), l.attn_o_b);
        cur = ggml_add(ctx0, cur, residual);

        // FFN
        residual = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, l.ffn_ln_w);
        cur = ggml_add(ctx0, cur, l.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_up_w, cur), l.ffn_up_b);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_down_w, cur), l.ffn_down_b);

        cur = ggml_add(ctx0, cur, residual);
    }

    // Final encoder LayerNorm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul(ctx0, cur, m.enc_out_ln_w);
    cur = ggml_add(ctx0, cur, m.enc_out_ln_b);

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// ── Compute cross-attention K, V from encoder output ─────────────

static bool compute_cross_kv(m2m100_context* c, const float* enc_out, int T_enc) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.dec_n_heads;
    const int hd = hp.head_dim();

    if (!alloc_cross_kv(c, T_enc))
        return false;

    // Build a ggml graph to compute cross-K and cross-V for all layers.
    // This handles F16/quantized weights natively via ggml_mul_mat.
    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& l = m.dec_layers[il];

        ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

        ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        ggml_set_name(enc_inp, "enc_for_cross");
        ggml_set_input(enc_inp);

        // K = W_k @ enc + b_k → (D, T_enc)
        ggml_tensor* K = ggml_mul_mat(ctx0, l.cross_k_w, enc_inp);
        if (l.cross_k_b)
            K = ggml_add(ctx0, K, l.cross_k_b);
        // Reshape to (hd, nh, T_enc) then permute to (hd, T_enc, nh)
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(K, "cross_k");

        // V = W_v @ enc + b_v
        ggml_tensor* V = ggml_mul_mat(ctx0, l.cross_v_w, enc_inp);
        if (l.cross_v_b)
            V = ggml_add(ctx0, V, l.cross_v_b);
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

        // Copy results to cross_kv tensors
        ggml_tensor* K_out = ggml_graph_get_tensor(gf, "cross_k");
        ggml_tensor* V_out = ggml_graph_get_tensor(gf, "cross_v");
        size_t sz = (size_t)hd * T_enc * nh * sizeof(float);
        std::vector<float> buf(hd * T_enc * nh);
        ggml_backend_tensor_get(K_out, buf.data(), 0, sz);
        ggml_backend_tensor_set(c->cross_kv_k[il], buf.data(), 0, sz);
        ggml_backend_tensor_get(V_out, buf.data(), 0, sz);
        ggml_backend_tensor_set(c->cross_kv_v[il], buf.data(), 0, sz);

        ggml_free(ctx0);
    }

    return true;
}

// ── Decoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_decoder_graph(m2m100_context* c, int n_tokens, int offset) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.dec_n_heads;
    const int hd = hp.head_dim();
    const float attn_scale = 1.0f / sqrtf((float)hd);
    const float emb_scale = hp.scale_embedding ? sqrtf((float)D) : 1.0f;
    const int Lk = offset + n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp, "dec_tokens");
    ggml_set_input(inp);

    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos, "dec_positions");
    ggml_set_input(pos);

    // Causal mask for self-attention (only when n_tokens > 1)
    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Embedding
    ggml_tensor* cur = ggml_get_rows(ctx0, m.shared_embed, inp);
    if (emb_scale != 1.0f) {
        cur = ggml_scale(ctx0, cur, emb_scale);
    }
    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, m.dec_pos_emb, pos));

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& l = m.dec_layers[il];
        ggml_tensor* residual = cur;

        // ---- Self-attention ----
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, l.attn_ln_w);
        cur = ggml_add(ctx0, cur, l.attn_ln_b);

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_q_w, cur), l.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_k_w, cur), l.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_v_w, cur), l.attn_v_b);

        // Reshape + permute to (hd, n_tokens, nh)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, n_tokens), 0, 2, 1, 3));
        ggml_tensor* K_new =
            ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, n_tokens), 0, 2, 1, 3));
        ggml_tensor* V_new =
            ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, n_tokens), 0, 2, 1, 3));

        // Write to KV cache
        ggml_tensor* k_view =
            ggml_view_4d(ctx0, c->kv_k, hd, n_tokens, nh, 1, c->kv_k->nb[1], c->kv_k->nb[2], c->kv_k->nb[3],
                         (size_t)il * c->kv_k->nb[3] + (size_t)offset * c->kv_k->nb[1]);
        ggml_tensor* v_view =
            ggml_view_4d(ctx0, c->kv_v, hd, n_tokens, nh, 1, c->kv_v->nb[1], c->kv_v->nb[2], c->kv_v->nb[3],
                         (size_t)il * c->kv_v->nb[3] + (size_t)offset * c->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new, v_view));

        // Read full KV history
        ggml_tensor* Kfull =
            ggml_view_3d(ctx0, c->kv_k, hd, Lk, nh, c->kv_k->nb[1], c->kv_k->nb[2], (size_t)il * c->kv_k->nb[3]);
        ggml_tensor* Vfull =
            ggml_view_3d(ctx0, c->kv_v, hd, Lk, nh, c->kv_v->nb[1], c->kv_v->nb[2], (size_t)il * c->kv_v->nb[3]);
        Kfull = ggml_cont(ctx0, Kfull);
        Vfull = ggml_cont(ctx0, Vfull);

        // Flash attention with causal mask
        ggml_tensor* sa_out =
            ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (n_tokens == 1) ? nullptr : causal_mask, attn_scale, 0.0f, 0.0f);
        cur = ggml_reshape_2d(ctx0, sa_out, D, n_tokens);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_o_w, cur), l.attn_o_b);
        cur = ggml_add(ctx0, cur, residual);

        // ---- Cross-attention ----
        residual = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, l.cross_ln_w);
        cur = ggml_add(ctx0, cur, l.cross_ln_b);

        ggml_tensor* CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, l.cross_q_w, cur), l.cross_q_b);
        CQ = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, CQ, hd, nh, n_tokens), 0, 2, 1, 3));

        ggml_tensor* CK = c->cross_kv_k[il];
        ggml_tensor* CV = c->cross_kv_v[il];

        ggml_tensor* ca_out = ggml_flash_attn_ext(ctx0, CQ, CK, CV, nullptr, attn_scale, 0.0f, 0.0f);
        cur = ggml_reshape_2d(ctx0, ca_out, D, n_tokens);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.cross_o_w, cur), l.cross_o_b);
        cur = ggml_add(ctx0, cur, residual);

        // ---- FFN ----
        residual = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul(ctx0, cur, l.ffn_ln_w);
        cur = ggml_add(ctx0, cur, l.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_up_w, cur), l.ffn_up_b);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_down_w, cur), l.ffn_down_b);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul(ctx0, cur, m.dec_out_ln_w);
    cur = ggml_add(ctx0, cur, m.dec_out_ln_b);

    // Take last token
    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    }

    // LM head (tied with shared_embed)
    cur = ggml_mul_mat(ctx0, m.shared_embed, cur);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// ── Run encoder ──────────────────────────────────────────────────

static std::vector<float> run_encoder(m2m100_context* c, const std::vector<int>& token_ids) {
    const int T = (int)token_ids.size();
    const int D = c->model.hp.d_model;

    ggml_cgraph* gf = build_encoder_graph(c, T);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "m2m100: failed to alloc encoder graph\n");
        return {};
    }

    // Set input tokens
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_tokens"), token_ids.data(), 0, T * sizeof(int32_t));

    // Set position IDs: offset=2 (M2M-100 padding_idx=1, first real pos=2)
    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; i++)
        positions[i] = i + 2;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_positions"), positions.data(), 0, T * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "m2m100: encoder compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    std::vector<float> enc_out(T * D);
    ggml_backend_tensor_get(out, enc_out.data(), 0, enc_out.size() * sizeof(float));
    return enc_out;
}

// ── Run decoder step ─────────────────────────────────────────────

static std::vector<float> run_decoder_step(m2m100_context* c, const int* tokens, int n_tokens, int offset) {
    const auto& hp = c->model.hp;
    const int vocab = hp.vocab_size;
    const int Lk = offset + n_tokens;

    ggml_cgraph* gf = build_decoder_graph(c, n_tokens, offset);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "m2m100: failed to alloc decoder graph\n");
        return {};
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_tokens"), tokens, 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = offset + i + 2; // offset=2
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_positions"), positions.data(), 0,
                            n_tokens * sizeof(int32_t));

    // Causal mask
    if (n_tokens > 1) {
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = offset + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "m2m100: decoder compute failed\n");
        return {};
    }

    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    std::vector<float> out(vocab);
    ggml_backend_tensor_get(logits, out.data(), 0, vocab * sizeof(float));
    return out;
}

// ── Public API ───────────────────────────────────────────────────

extern "C" struct m2m100_context_params m2m100_context_default_params(void) {
    m2m100_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct m2m100_context* m2m100_init_from_file(const char* path_model, struct m2m100_context_params params) {
    auto* c = new m2m100_context();
    c->params = params;

    // Pass 1: metadata
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
        fprintf(stderr, "m2m100: d=%d enc=%dL dec=%dL heads=%d ffn=%d vocab=%d langs=%d\n", hp.d_model, hp.enc_n_layers,
                hp.dec_n_layers, hp.enc_n_heads, hp.enc_ffn_dim, hp.vocab_size, (int)c->tokenizer.lang_codes.size());
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    c->backend = c->backend_cpu;

    // Pass 2: weights
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path_model, c->backend, "m2m100", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    if (!bind_model(c)) {
        fprintf(stderr, "m2m100: failed to bind tensors\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler
    {
        ggml_backend_t backends[] = {c->backend};
        c->sched = ggml_backend_sched_new(backends, nullptr, 1, 8192, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    return c;
}

extern "C" void m2m100_free(struct m2m100_context* ctx) {
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

extern "C" char* m2m100_translate(struct m2m100_context* ctx, const char* text, const char* src_lang,
                                  const char* tgt_lang, int max_new_tokens) {
    if (!ctx || !text || !src_lang || !tgt_lang)
        return nullptr;
    if (max_new_tokens <= 0)
        max_new_tokens = 200;

    const auto& hp = ctx->model.hp;

    // 1. Tokenize input
    std::vector<int> enc_ids = tokenize(ctx->tokenizer, text, src_lang);
    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "m2m100: input %zu tokens:", enc_ids.size());
        for (int id : enc_ids)
            fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    // 2. Run encoder
    std::vector<float> enc_out = run_encoder(ctx, enc_ids);
    if (enc_out.empty())
        return nullptr;

    int T_enc = (int)enc_ids.size();
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "m2m100: encoder done, T_enc=%d\n", T_enc);
    }

    // 3. Compute cross-attention KV
    if (!compute_cross_kv(ctx, enc_out.data(), T_enc)) {
        fprintf(stderr, "m2m100: failed to compute cross-KV\n");
        return nullptr;
    }

    // 4. Allocate decoder KV cache
    int dec_max = max_new_tokens + 4;
    if (!alloc_kv_cache(ctx, dec_max)) {
        fprintf(stderr, "m2m100: failed to alloc KV cache\n");
        return nullptr;
    }

    // 5. Greedy decode
    // Start with decoder_start_token_id (eos=2) then forced_bos (target lang token)
    auto tgt_it = ctx->tokenizer.lang_to_token_id.find(tgt_lang);
    if (tgt_it == ctx->tokenizer.lang_to_token_id.end()) {
        fprintf(stderr, "m2m100: unknown target language '%s'\n", tgt_lang);
        return nullptr;
    }
    int forced_bos = tgt_it->second;

    std::vector<int> dec_ids;
    dec_ids.push_back(hp.dec_start_token); // </s> = 2
    dec_ids.push_back(forced_bos);         // __de__ etc.

    // First step: prefill with [dec_start, forced_bos]
    std::vector<float> logits = run_decoder_step(ctx, dec_ids.data(), (int)dec_ids.size(), 0);
    if (logits.empty())
        return nullptr;

    int offset = (int)dec_ids.size();

    for (int step = 0; step < max_new_tokens; step++) {
        // Greedy: argmax
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
            fprintf(stderr, "m2m100[dec]: step=%d tok=%d '%s'\n", step, best_id,
                    best_id < (int)ctx->tokenizer.id_to_token.size() ? ctx->tokenizer.id_to_token[best_id].c_str()
                                                                     : "?");
        }

        // Next step: single token
        logits = run_decoder_step(ctx, &best_id, 1, offset);
        if (logits.empty())
            return nullptr;
        offset++;
    }

    // 6. Detokenize
    std::string result = detokenize(ctx->tokenizer, dec_ids);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "m2m100: translated %zu tokens → '%s'\n", dec_ids.size(), result.c_str());
    }

    char* out = (char*)malloc(result.size() + 1);
    std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" int m2m100_n_languages(struct m2m100_context* ctx) {
    return ctx ? (int)ctx->tokenizer.lang_codes.size() : 0;
}

extern "C" const char* m2m100_language(struct m2m100_context* ctx, int index) {
    if (!ctx || index < 0 || index >= (int)ctx->tokenizer.lang_codes.size())
        return nullptr;
    return ctx->tokenizer.lang_codes[index].c_str();
}
