// bert_encoder.cpp — Minimal BERT forward pass for MeloTTS conditioning.
//
// Implements bert-base-uncased inference (layers 0-9 only) using ggml
// graphs. Produces hidden_states[-3] = layer 9 output for each token.

#include "bert_encoder.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ── WordPiece tokenizer ───────────────────────────────────────────

struct bert_tokenizer {
    std::map<std::string, int> vocab;
    int cls_id = 101; // [CLS]
    int sep_id = 102; // [SEP]
    int unk_id = 100; // [UNK]
    int pad_id = 0;   // [PAD]
};

static std::string to_lower_bert(const std::string& s) {
    std::string out;
    for (char c : s)
        out += (char)tolower((unsigned char)c);
    return out;
}

static std::vector<int> bert_tokenize(const bert_tokenizer& tok, const std::string& text) {
    std::vector<int> ids;
    ids.push_back(tok.cls_id);

    // Simple whitespace tokenization + WordPiece
    std::string lower = to_lower_bert(text);
    std::vector<std::string> words;
    std::string cur;
    for (char c : lower) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        words.push_back(cur);

    for (const auto& word : words) {
        // Try to find the whole word first
        auto it = tok.vocab.find(word);
        if (it != tok.vocab.end()) {
            ids.push_back(it->second);
            continue;
        }

        // WordPiece: greedy longest-match from left
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int best_id = tok.unk_id;
            while (end > start) {
                std::string sub = (start == 0) ? word.substr(0, end) : "##" + word.substr(start, end - start);
                auto sit = tok.vocab.find(sub);
                if (sit != tok.vocab.end()) {
                    best_id = sit->second;
                    break;
                }
                end--;
            }
            ids.push_back(best_id);
            if (end == start)
                start++; // skip char if nothing found
            else
                start = end;
        }
    }

    ids.push_back(tok.sep_id);
    return ids;
}

// ── BERT layer weights ────────────────────────────────────────────

struct bert_layer {
    ggml_tensor *q_w, *q_b;
    ggml_tensor *k_w, *k_b;
    ggml_tensor *v_w, *v_b;
    ggml_tensor *o_w, *o_b;
    ggml_tensor *ln1_w, *ln1_b;
    ggml_tensor *ff1_w, *ff1_b;
    ggml_tensor *ff2_w, *ff2_b;
    ggml_tensor *ln2_w, *ln2_b;
};

struct bert_weights {
    ggml_tensor* word_emb;
    ggml_tensor* pos_emb;
    ggml_tensor* type_emb;
    ggml_tensor *emb_ln_w, *emb_ln_b;
    std::vector<bert_layer> layers;
};

struct bert_encoder_context {
    int n_layers;
    int hidden_size;
    int n_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    int max_pos;
    float ln_eps;

    bert_weights w;
    bert_tokenizer tok;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;

    int n_threads;
};

// ── Forward pass ──────────────────────────────────────────────────

static ggml_tensor* bert_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

// GELU approximation matching BERT's implementation
static ggml_tensor* bert_gelu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_gelu(ctx, x);
}

static bool bert_forward(bert_encoder_context* bctx, const std::vector<int>& token_ids,
                         std::vector<float>& out_hidden) {
    int T = (int)token_ids.size();
    int C = bctx->hidden_size;
    int H = bctx->n_heads;
    int D = bctx->head_dim;

    // Build ggml graph for the full BERT forward pass
    size_t ctx_size = 64 * 1024 * 1024; // 64 MB compute buffer
    ggml_init_params params = {ctx_size, nullptr, true};
    ggml_context* ctx = ggml_init(params);
    if (!ctx)
        return false;

    // Input tensors
    ggml_tensor* input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(input_ids, "input_ids");
    ggml_set_input(input_ids);

    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Embeddings: word + position + type(=0)
    // Cast to F32 after get_rows in case embeddings are quantized (Q4_K/Q8_0)
    ggml_tensor* x = ggml_get_rows(ctx, bctx->w.word_emb, input_ids);
    if (x->type != GGML_TYPE_F32)
        x = ggml_cast(ctx, x, GGML_TYPE_F32);
    ggml_tensor* pos = ggml_get_rows(ctx, bctx->w.pos_emb, pos_ids);
    if (pos->type != GGML_TYPE_F32)
        pos = ggml_cast(ctx, pos, GGML_TYPE_F32);
    x = ggml_add(ctx, x, pos);
    // Type embedding is all zeros for single-sentence, so just add type_emb[0]
    // For quantized type_emb, cast the full tensor first then view
    ggml_tensor* type_f32 = bctx->w.type_emb;
    if (type_f32->type != GGML_TYPE_F32)
        type_f32 = ggml_cast(ctx, type_f32, GGML_TYPE_F32);
    ggml_tensor* type0 = ggml_view_1d(ctx, type_f32, C, 0);
    x = ggml_add(ctx, x, type0);
    // LayerNorm
    x = bert_layer_norm(ctx, x, bctx->w.emb_ln_w, bctx->w.emb_ln_b, bctx->ln_eps);

    // Transformer layers
    for (int il = 0; il < bctx->n_layers; il++) {
        const auto& layer = bctx->w.layers[il];

        // Self-attention: Q, K, V projections
        ggml_tensor* Q = ggml_add(ctx, ggml_mul_mat(ctx, layer.q_w, x), layer.q_b);
        ggml_tensor* K = ggml_add(ctx, ggml_mul_mat(ctx, layer.k_w, x), layer.k_b);
        ggml_tensor* V = ggml_add(ctx, ggml_mul_mat(ctx, layer.v_w, x), layer.v_b);

        // Multi-head attention via ggml flash-attn-like pattern
        // Q,K,V are (C, T) after linear. Reshape to (D, H, T) then permute to (D, T, H)
        // For ggml_flash_attn: Q(D, T, H), K(D, T, H), V(D, T, H)
        Q = ggml_reshape_3d(ctx, Q, D, H, T);
        K = ggml_reshape_3d(ctx, K, D, H, T);
        V = ggml_reshape_3d(ctx, V, D, H, T);

        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (D, T, H)
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        // Scaled dot-product: scores = Q^T @ K / sqrt(D), per head
        // ggml_mul_mat(K, Q) with K(D,T,H) and Q(D,T,H) → (T,T,H) [scores per head]
        ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
        KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float)D));
        KQ = ggml_soft_max(ctx, KQ);

        // attn_output = scores @ V → (D, T, H)
        // Transpose V to (T, D, H) for mul_mat: ggml_mul_mat(V_t, KQ)
        ggml_tensor* Vt = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3)); // (T, D, H)
        ggml_tensor* attn = ggml_mul_mat(ctx, Vt, KQ);                      // (D, T, H)

        // Permute back: (D, T, H) → (D, H, T) → reshape to (C, T)
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // (D, H, T)
        attn = ggml_cont(ctx, ggml_reshape_2d(ctx, attn, C, T));

        // Output projection + residual + LayerNorm
        ggml_tensor* o = ggml_add(ctx, ggml_mul_mat(ctx, layer.o_w, attn), layer.o_b);
        x = ggml_add(ctx, x, o);
        x = bert_layer_norm(ctx, x, layer.ln1_w, layer.ln1_b, bctx->ln_eps);

        // FFN: linear → GELU → linear
        ggml_tensor* ff = ggml_add(ctx, ggml_mul_mat(ctx, layer.ff1_w, x), layer.ff1_b);
        ff = bert_gelu(ctx, ff);
        ff = ggml_add(ctx, ggml_mul_mat(ctx, layer.ff2_w, ff), layer.ff2_b);

        // Residual + LayerNorm
        x = ggml_add(ctx, x, ff);
        x = bert_layer_norm(ctx, x, layer.ln2_w, layer.ln2_b, bctx->ln_eps);
    }

    // x is now the hidden states from layer 9 (= hidden_states[-3])
    ggml_set_name(x, "bert_output");
    ggml_set_output(x);

    // Build and compute graph
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(bctx->sched);
    if (!ggml_backend_sched_alloc_graph(bctx->sched, gf)) {
        fprintf(stderr, "bert_encoder: graph alloc failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set inputs
    std::vector<int32_t> ids32(T);
    std::vector<int32_t> pos32(T);
    for (int i = 0; i < T; i++) {
        ids32[i] = token_ids[i];
        pos32[i] = i;
    }
    ggml_backend_tensor_set(input_ids, ids32.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(pos_ids, pos32.data(), 0, T * sizeof(int32_t));

    ggml_backend_sched_graph_compute(bctx->sched, gf);

    // Read output: (T, C) → caller gets flat array
    out_hidden.resize(T * C);
    ggml_backend_tensor_get(x, out_hidden.data(), 0, T * C * sizeof(float));

    ggml_free(ctx);
    return true;
}

// ── Public API ────────────────────────────────────────────────────

extern "C" struct bert_encoder_context* bert_encoder_init(const char* gguf_path, int n_threads) {
    auto* ctx = new bert_encoder_context();
    ctx->n_threads = n_threads;

    // Backend
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ctx->backend = ctx->backend_cpu;

    // Metadata
    gguf_context* meta = core_gguf::open_metadata(gguf_path);
    if (!meta) {
        delete ctx;
        return nullptr;
    }

    ctx->n_layers = (int)core_gguf::kv_u32(meta, "bert.n_layers", 10);
    ctx->hidden_size = (int)core_gguf::kv_u32(meta, "bert.hidden_size", 768);
    ctx->n_heads = (int)core_gguf::kv_u32(meta, "bert.n_heads", 12);
    ctx->head_dim = ctx->hidden_size / ctx->n_heads;
    ctx->intermediate_size = (int)core_gguf::kv_u32(meta, "bert.intermediate_size", 3072);
    ctx->vocab_size = (int)core_gguf::kv_u32(meta, "bert.vocab_size", 30522);
    ctx->max_pos = (int)core_gguf::kv_u32(meta, "bert.max_position_embeddings", 512);
    ctx->ln_eps = core_gguf::kv_f32(meta, "bert.layer_norm_eps", 1e-12f);

    // Tokenizer vocab
    std::string vocab_json = core_gguf::kv_str(meta, "bert.vocab_json", "{}");
    core_gguf::free_metadata(meta);

    // Parse vocab JSON: {"token": id, ...}
    {
        size_t pos = vocab_json.find('{');
        if (pos != std::string::npos) {
            pos++;
            while (pos < vocab_json.size()) {
                while (pos < vocab_json.size() && (vocab_json[pos] == ' ' || vocab_json[pos] == ',' ||
                                                   vocab_json[pos] == '\n' || vocab_json[pos] == '\r'))
                    pos++;
                if (pos >= vocab_json.size() || vocab_json[pos] == '}')
                    break;
                if (vocab_json[pos] != '"')
                    break;
                pos++;
                std::string token;
                while (pos < vocab_json.size() && vocab_json[pos] != '"') {
                    if (vocab_json[pos] == '\\' && pos + 1 < vocab_json.size()) {
                        pos++;
                        if (vocab_json[pos] == '"')
                            token += '"';
                        else if (vocab_json[pos] == '\\')
                            token += '\\';
                        else if (vocab_json[pos] == 'n')
                            token += '\n';
                        else
                            token += vocab_json[pos];
                    } else {
                        token += vocab_json[pos];
                    }
                    pos++;
                }
                if (pos < vocab_json.size())
                    pos++;
                while (pos < vocab_json.size() && (vocab_json[pos] == ':' || vocab_json[pos] == ' '))
                    pos++;
                int id = 0;
                bool neg = false;
                if (pos < vocab_json.size() && vocab_json[pos] == '-') {
                    neg = true;
                    pos++;
                }
                while (pos < vocab_json.size() && vocab_json[pos] >= '0' && vocab_json[pos] <= '9') {
                    id = id * 10 + (vocab_json[pos] - '0');
                    pos++;
                }
                ctx->tok.vocab[token] = neg ? -id : id;
            }
        }
        fprintf(stderr, "bert_encoder: vocab %zu entries\n", ctx->tok.vocab.size());
    }

    // Load weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(gguf_path, ctx->backend, "bert", wl)) {
        delete ctx;
        return nullptr;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;

    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        if (it == wl.tensors.end()) {
            fprintf(stderr, "bert_encoder: missing '%s'\n", name.c_str());
            return nullptr;
        }
        return it->second;
    };

    ctx->w.word_emb = get("embeddings.word_embeddings.weight");
    ctx->w.pos_emb = get("embeddings.position_embeddings.weight");
    ctx->w.type_emb = get("embeddings.token_type_embeddings.weight");
    ctx->w.emb_ln_w = get("embeddings.LayerNorm.weight");
    ctx->w.emb_ln_b = get("embeddings.LayerNorm.bias");

    ctx->w.layers.resize(ctx->n_layers);
    for (int i = 0; i < ctx->n_layers; i++) {
        auto& l = ctx->w.layers[i];
        std::string p = "encoder.layer." + std::to_string(i);
        l.q_w = get(p + ".attention.self.query.weight");
        l.q_b = get(p + ".attention.self.query.bias");
        l.k_w = get(p + ".attention.self.key.weight");
        l.k_b = get(p + ".attention.self.key.bias");
        l.v_w = get(p + ".attention.self.value.weight");
        l.v_b = get(p + ".attention.self.value.bias");
        l.o_w = get(p + ".attention.output.dense.weight");
        l.o_b = get(p + ".attention.output.dense.bias");
        l.ln1_w = get(p + ".attention.output.LayerNorm.weight");
        l.ln1_b = get(p + ".attention.output.LayerNorm.bias");
        l.ff1_w = get(p + ".intermediate.dense.weight");
        l.ff1_b = get(p + ".intermediate.dense.bias");
        l.ff2_w = get(p + ".output.dense.weight");
        l.ff2_b = get(p + ".output.dense.bias");
        l.ln2_w = get(p + ".output.LayerNorm.weight");
        l.ln2_b = get(p + ".output.LayerNorm.bias");
    }

    // Scheduler
    ggml_backend_t backends[] = {ctx->backend};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);

    fprintf(stderr, "bert_encoder: loaded %d layers, hidden=%d, heads=%d\n", ctx->n_layers, ctx->hidden_size,
            ctx->n_heads);
    return ctx;
}

extern "C" void bert_encoder_free(struct bert_encoder_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->w_buf)
        ggml_backend_buffer_free(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" bool bert_encoder_forward(struct bert_encoder_context* ctx, const char* text, float** out_features,
                                     int* out_n_tokens) {
    if (!ctx || !text || !out_features || !out_n_tokens)
        return false;

    std::vector<int> token_ids = bert_tokenize(ctx->tok, text);
    int T = (int)token_ids.size();

    std::vector<float> hidden;
    if (!bert_forward(ctx, token_ids, hidden))
        return false;

    // hidden is (T, C) row-major
    int C = ctx->hidden_size;
    *out_n_tokens = T;
    *out_features = (float*)malloc(T * C * sizeof(float));
    memcpy(*out_features, hidden.data(), T * C * sizeof(float));
    return true;
}

extern "C" int bert_encoder_hidden_size(const struct bert_encoder_context* ctx) {
    return ctx ? ctx->hidden_size : 768;
}

extern "C" int bert_encoder_tokenize(const struct bert_encoder_context* ctx, const char* text, int** out_ids) {
    if (!ctx || !text || !out_ids)
        return 0;
    auto ids = bert_tokenize(ctx->tok, text);
    int n = (int)ids.size();
    *out_ids = (int*)malloc(n * sizeof(int));
    memcpy(*out_ids, ids.data(), n * sizeof(int));
    return n;
}

extern "C" int bert_encoder_word_subtokens(const struct bert_encoder_context* ctx, const char* text,
                                           int** out_subtokens) {
    if (!ctx || !text || !out_subtokens)
        return 0;

    // Tokenize to get the subword tokens
    auto ids = bert_tokenize(ctx->tok, text);
    // ids[0] = [CLS], ids[n-1] = [SEP], content = ids[1..n-2]

    // Reconstruct which content tokens belong to which whitespace word
    // Approach: re-tokenize word by word, count subwords per word
    std::string lower = to_lower_bert(text);
    std::vector<std::string> words;
    std::string cur;
    for (char c : lower) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        words.push_back(cur);

    // For each word, count how many BERT subwords it produces
    std::vector<int> subtokens;
    for (const auto& word : words) {
        // Tokenize this single word
        int n_sub = 0;
        auto it = ctx->tok.vocab.find(word);
        if (it != ctx->tok.vocab.end()) {
            n_sub = 1;
        } else {
            // WordPiece split
            size_t start = 0;
            while (start < word.size()) {
                size_t end = word.size();
                while (end > start) {
                    std::string sub = (start == 0) ? word.substr(0, end) : "##" + word.substr(start, end - start);
                    if (ctx->tok.vocab.count(sub))
                        break;
                    end--;
                }
                n_sub++;
                if (end == start)
                    start++;
                else
                    start = end;
            }
        }
        subtokens.push_back(n_sub);
    }

    int n_words = (int)subtokens.size();
    *out_subtokens = (int*)malloc(n_words * sizeof(int));
    memcpy(*out_subtokens, subtokens.data(), n_words * sizeof(int));
    return n_words;
}
