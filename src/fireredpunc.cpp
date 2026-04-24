// fireredpunc.cpp — FireRedPunc BERT-based punctuation restoration.
//
// Architecture: BERT-base (12L, d=768, 12 heads, d_ffn=3072, GELU)
//               + Linear(768, 5) classifier
//               5 classes: space(0), ，(1), 。(2), ？(3), ！(4)
//
// Input:  unpunctuated text (tokenised with BERT WordPiece vocab)
// Output: text with punctuation inserted

#include "fireredpunc.h"

#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// WordPiece tokenizer (minimal, matching BERT chinese-bert-wwm-ext)
// ---------------------------------------------------------------------------

namespace {

struct WordPieceTokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    int unk_id = 100;              // [UNK]
    int cls_id = 101;              // [CLS]
    int sep_id = 102;              // [SEP]
    int pad_id = 0;                // [PAD]
    bool is_sentencepiece = false; // true for XLM-RoBERTa

    void build_map() {
        token_to_id.clear();
        for (int i = 0; i < (int)id_to_token.size(); i++) {
            token_to_id[id_to_token[i]] = i;
        }
    }

    int lookup(const std::string& tok) const {
        auto it = token_to_id.find(tok);
        return it != token_to_id.end() ? it->second : unk_id;
    }

    // SentencePiece tokenization: split on whitespace, prefix ▁, greedy longest match
    std::vector<int> tokenize_sp(const std::string& text) const {
        std::vector<int> ids;
        // Split on whitespace
        std::vector<std::string> words;
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
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

        for (size_t wi = 0; wi < words.size(); wi++) {
            // Prefix with ▁ (U+2581, 0xE2 0x96 0x81)
            std::string word = "\xE2\x96\x81" + words[wi];
            size_t start = 0;
            while (start < word.size()) {
                size_t end = word.size();
                int best_id = -1;
                while (end > start) {
                    std::string sub = word.substr(start, end - start);
                    auto it = token_to_id.find(sub);
                    if (it != token_to_id.end()) {
                        best_id = it->second;
                        break;
                    }
                    end--;
                    while (end > start && (word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (best_id < 0) {
                    ids.push_back(unk_id);
                    break;
                }
                ids.push_back(best_id);
                start = end;
            }
        }
        return ids;
    }

    // Basic BERT tokenization: lowercase, split on whitespace, WordPiece
    std::vector<int> tokenize(const std::string& text) const {
        if (is_sentencepiece)
            return tokenize_sp(text);
        std::vector<int> ids;

        // Split on whitespace
        std::vector<std::string> words;
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
            } else {
                // Lowercase ASCII
                if (c >= 'A' && c <= 'Z')
                    c = c - 'A' + 'a';
                cur += c;
            }
        }
        if (!cur.empty())
            words.push_back(cur);

        // WordPiece each word
        for (const auto& word : words) {
            // Try to find the word as-is first
            auto it = token_to_id.find(word);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
                continue;
            }

            // WordPiece: greedily match longest subword
            size_t start = 0;
            while (start < word.size()) {
                size_t end = word.size();
                int best_id = -1;
                while (end > start) {
                    std::string sub = (start == 0) ? word.substr(0, end) : ("##" + word.substr(start, end - start));
                    auto sit = token_to_id.find(sub);
                    if (sit != token_to_id.end()) {
                        best_id = sit->second;
                        break;
                    }
                    // Handle multi-byte UTF-8: don't split mid-character
                    end--;
                    while (end > start && (word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (best_id < 0) {
                    // Character not in vocab — use [UNK]
                    ids.push_back(unk_id);
                    break;
                }
                ids.push_back(best_id);
                start = (start == 0) ? (end > start ? end : end) : start + (end - start);
                // Recalculate start for ##-prefixed case
                if (start == 0)
                    start = end;
                else {
                    // Already handled above
                }
            }
        }

        return ids;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Model structure
// ---------------------------------------------------------------------------

struct BertLayer {
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
};

struct fireredpunc_context {
    // Hyperparams
    int d_model = 768;
    int d_ffn = 3072;
    int n_heads = 12;
    int n_layers = 12;
    int vocab_size = 21128;
    int max_pos = 512;
    int n_classes = 5;
    int cls_id = 101;
    int pad_id = 0;

    // Labels
    std::vector<std::string> labels;

    // Tokenizer
    WordPieceTokenizer tokenizer;

    // Weights
    ggml_tensor* tok_emb_w = nullptr;
    ggml_tensor* pos_emb_w = nullptr;
    ggml_tensor* type_emb_w = nullptr;
    ggml_tensor* emb_ln_w = nullptr;
    ggml_tensor* emb_ln_b = nullptr;

    std::vector<BertLayer> layers;

    ggml_tensor* cls_w = nullptr;
    ggml_tensor* cls_b = nullptr;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* w_ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
};

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static bool fireredpunc_load(fireredpunc_context& ctx, const char* path) {
    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    ctx.d_model = (int)core_gguf::kv_u32(meta, "fireredpunc.d_model", 768);
    ctx.d_ffn = (int)core_gguf::kv_u32(meta, "fireredpunc.d_ffn", 3072);
    ctx.n_heads = (int)core_gguf::kv_u32(meta, "fireredpunc.n_heads", 12);
    ctx.n_layers = (int)core_gguf::kv_u32(meta, "fireredpunc.n_layers", 12);
    ctx.vocab_size = (int)core_gguf::kv_u32(meta, "fireredpunc.vocab_size", 21128);
    ctx.max_pos = (int)core_gguf::kv_u32(meta, "fireredpunc.max_pos", 512);
    ctx.n_classes = (int)core_gguf::kv_u32(meta, "fireredpunc.n_classes", 5);
    ctx.cls_id = (int)core_gguf::kv_u32(meta, "fireredpunc.cls_id", 101);
    ctx.pad_id = (int)core_gguf::kv_u32(meta, "fireredpunc.pad_id", 0);

    // Vocab
    ctx.tokenizer.id_to_token = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    ctx.tokenizer.cls_id = ctx.cls_id;
    ctx.tokenizer.pad_id = ctx.pad_id;
    std::string tok_type = core_gguf::kv_str(meta, "fireredpunc.tokenizer_type", "wordpiece");
    ctx.tokenizer.is_sentencepiece = (tok_type == "sentencepiece");
    if (ctx.tokenizer.is_sentencepiece) {
        ctx.tokenizer.unk_id = 3; // XLM-R <unk> = 3
    }
    ctx.tokenizer.build_map();

    // Labels
    ctx.labels = core_gguf::kv_str_array(meta, "fireredpunc.labels");
    if (ctx.labels.empty()) {
        ctx.labels = {" ", "\xef\xbc\x8c", "\xe3\x80\x82", "\xef\xbc\x9f", "\xef\xbc\x81"};
    }

    core_gguf::free_metadata(meta);

    fprintf(stderr, "fireredpunc: %dL, d=%d, ffn=%d, heads=%d, vocab=%d, labels=%d\n", ctx.n_layers, ctx.d_model,
            ctx.d_ffn, ctx.n_heads, ctx.vocab_size, ctx.n_classes);

    // Pass 2: weights
    ctx.backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "fireredpunc", wl))
        return false;
    ctx.w_ctx = wl.ctx;
    ctx.buf = wl.buf;

    auto& T = wl.tensors;
    auto req = [&](const char* n) { return core_gguf::require(T, n, "fireredpunc"); };

    ctx.tok_emb_w = req("emb.tok_emb.weight");
    ctx.pos_emb_w = req("emb.pos_emb.weight");
    ctx.type_emb_w = req("emb.type_emb.weight");
    ctx.emb_ln_w = req("emb.ln.weight");
    ctx.emb_ln_b = req("emb.ln.bias");

    ctx.layers.resize(ctx.n_layers);
    for (int i = 0; i < ctx.n_layers; i++) {
        auto ln = [&](const char* fmt) { return core_gguf::format_layer_name(fmt, i); };
        auto& L = ctx.layers[i];
        L.attn_q_w = req(ln("enc.%d.attn.q.weight").c_str());
        L.attn_q_b = req(ln("enc.%d.attn.q.bias").c_str());
        L.attn_k_w = req(ln("enc.%d.attn.k.weight").c_str());
        L.attn_k_b = req(ln("enc.%d.attn.k.bias").c_str());
        L.attn_v_w = req(ln("enc.%d.attn.v.weight").c_str());
        L.attn_v_b = req(ln("enc.%d.attn.v.bias").c_str());
        L.attn_out_w = req(ln("enc.%d.attn.out.weight").c_str());
        L.attn_out_b = req(ln("enc.%d.attn.out.bias").c_str());
        L.attn_ln_w = req(ln("enc.%d.attn.ln.weight").c_str());
        L.attn_ln_b = req(ln("enc.%d.attn.ln.bias").c_str());
        L.ffn_up_w = req(ln("enc.%d.ffn.up.weight").c_str());
        L.ffn_up_b = req(ln("enc.%d.ffn.up.bias").c_str());
        L.ffn_down_w = req(ln("enc.%d.ffn.down.weight").c_str());
        L.ffn_down_b = req(ln("enc.%d.ffn.down.bias").c_str());
        L.ffn_ln_w = req(ln("enc.%d.ffn.ln.weight").c_str());
        L.ffn_ln_b = req(ln("enc.%d.ffn.ln.bias").c_str());
    }

    ctx.cls_w = req("cls.weight");
    ctx.cls_b = req("cls.bias");

    // Scheduler
    auto* bk = ctx.backend;
    ctx.sched = ggml_backend_sched_new(&bk, nullptr, 1, 8192, false, false);

    return true;
}

// ---------------------------------------------------------------------------
// Graph build: BERT encoder + classifier
// ---------------------------------------------------------------------------

static std::vector<int> fireredpunc_run(fireredpunc_context& ctx, const std::vector<int>& token_ids) {
    const int N = (int)token_ids.size();
    // Prepend CLS + append SEP
    const int seq_len = N + 2;

    // ggml context for compute graph
    size_t mem = ggml_tensor_overhead() * (ctx.n_layers * 40 + 50) + 1024 * 1024;
    struct ggml_init_params gp = {mem, nullptr, true};
    ggml_context* ctx0 = ggml_init(gp);

    // Input: token IDs [seq_len]
    ggml_tensor* inp_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_ids, "inp_ids");
    ggml_set_input(inp_ids);

    // Position IDs [seq_len]
    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Token type IDs [seq_len] — all zeros
    ggml_tensor* type_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(type_ids, "type_ids");
    ggml_set_input(type_ids);

    // Embeddings: tok + pos + type
    ggml_tensor* tok_emb = ggml_get_rows(ctx0, ctx.tok_emb_w, inp_ids);    // [seq_len, d]
    ggml_tensor* pos_emb = ggml_get_rows(ctx0, ctx.pos_emb_w, pos_ids);    // [seq_len, d]
    ggml_tensor* type_emb = ggml_get_rows(ctx0, ctx.type_emb_w, type_ids); // [seq_len, d]

    ggml_tensor* emb = ggml_add(ctx0, ggml_add(ctx0, tok_emb, pos_emb), type_emb);
    emb = ggml_norm(ctx0, emb, 1e-12f);
    emb = ggml_add(ctx0, ggml_mul(ctx0, emb, ctx.emb_ln_w), ctx.emb_ln_b);
    // emb: [d_model, seq_len] (ggml column-major)

    // Dump embedding output for diff-testing
    if (getenv("FIREREDPUNC_DEBUG")) {
        ggml_set_name(emb, "emb_out");
        ggml_set_output(emb);
    }

    const int d = ctx.d_model;
    const int head_dim = d / ctx.n_heads;
    const int nh = ctx.n_heads;

    ggml_tensor* cur = emb;

    for (int i = 0; i < ctx.n_layers; i++) {
        const auto& L = ctx.layers[i];
        ggml_tensor* residual = cur;

        // Self-attention
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_q_w, cur), L.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_k_w, cur), L.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_v_w, cur), L.attn_v_b);

        // Reshape for multi-head attention: [d, seq] -> [head_dim, nh, seq]
        Q = ggml_reshape_3d(ctx0, Q, head_dim, nh, seq_len);
        K = ggml_reshape_3d(ctx0, K, head_dim, nh, seq_len);
        V = ggml_reshape_3d(ctx0, V, head_dim, nh, seq_len);

        // Permute to [head_dim, seq_len, nh] for flash_attn_ext
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Flash attention: handles QK^T, scale, softmax, V matmul
        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* KQV = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        KQV = ggml_reshape_2d(ctx0, KQV, d, seq_len);

        // Output projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_out_w, KQV), L.attn_out_b);

        // Post-norm (residual + LN)
        cur = ggml_add(ctx0, cur, residual);
        cur = ggml_norm(ctx0, cur, 1e-12f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.attn_ln_w), L.attn_ln_b);

        // FFN
        residual = cur;
        ggml_tensor* ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_up_w, cur), L.ffn_up_b);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_down_w, ffn), L.ffn_down_b);

        // Post-norm
        cur = ggml_add(ctx0, ffn, residual);
        cur = ggml_norm(ctx0, cur, 1e-12f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.ffn_ln_w), L.ffn_ln_b);
    }

    // Remove CLS token: cur is [d, seq_len], we want [d, N] starting from position 1
    cur = ggml_view_2d(ctx0, cur, d, N, cur->nb[1], cur->nb[1]); // skip first column

    // Classifier: [n_classes, d] x [d, N] -> [n_classes, N]
    ggml_tensor* logits = ggml_add(ctx0, ggml_mul_mat(ctx0, ctx.cls_w, cur), ctx.cls_b);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);

    // Build & compute graph
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "fireredpunc: failed to allocate graph\n");
        ggml_free(ctx0);
        return {};
    }

    // Set input data
    {
        std::vector<int32_t> ids(seq_len);
        ids[0] = ctx.cls_id;
        for (int i = 0; i < N; i++)
            ids[i + 1] = token_ids[i];
        // SEP token: 102 for BERT, 2 for RoBERTa
        ids[N + 1] = ctx.tokenizer.is_sentencepiece ? 2 : 102;
        ggml_backend_tensor_set(inp_ids, ids.data(), 0, seq_len * sizeof(int32_t));

        std::vector<int32_t> pos(seq_len);
        // RoBERTa: position IDs start at padding_idx+1 (=2 for XLM-R)
        // BERT: position IDs start at 0
        const int pos_offset = ctx.tokenizer.is_sentencepiece ? (ctx.pad_id + 1) : 0;
        for (int i = 0; i < seq_len; i++)
            pos[i] = i + pos_offset;
        ggml_backend_tensor_set(pos_ids, pos.data(), 0, seq_len * sizeof(int32_t));

        std::vector<int32_t> types(seq_len, 0);
        ggml_backend_tensor_set(type_ids, types.data(), 0, seq_len * sizeof(int32_t));
    }

    ggml_backend_sched_graph_compute(ctx.sched, gf);

    // Dump embedding for diff-testing
    if (getenv("FIREREDPUNC_DEBUG")) {
        ggml_tensor* emb_t = ggml_get_tensor(ctx0, "emb_out");
        if (emb_t) {
            // emb is [d, seq_len], read values at position 15 (first "you")
            const int dump_pos = std::min(15, seq_len - 1);
            std::vector<float> emb_vals(d);
            ggml_backend_tensor_get(emb_t, emb_vals.data(), dump_pos * d * sizeof(float), d * sizeof(float));
            fprintf(stderr, "fireredpunc: emb[%d][:8] = [", dump_pos);
            for (int j = 0; j < 8; j++)
                fprintf(stderr, "%s%.6f", j ? ", " : "", emb_vals[j]);
            fprintf(stderr, "]\n");
            float norm = 0;
            for (int j = 0; j < d; j++)
                norm += emb_vals[j] * emb_vals[j];
            fprintf(stderr, "fireredpunc: emb[%d] norm = %.6f\n", dump_pos, sqrtf(norm));
        }
    }

    // Read logits: [n_classes, N]
    std::vector<float> logits_buf(ctx.n_classes * N);
    ggml_backend_tensor_get(logits, logits_buf.data(), 0, logits_buf.size() * sizeof(float));

    // Argmax per token
    std::vector<int> preds(N);
    for (int t = 0; t < N; t++) {
        int best = 0;
        float best_val = logits_buf[t * ctx.n_classes];
        for (int c = 1; c < ctx.n_classes; c++) {
            float v = logits_buf[t * ctx.n_classes + c];
            if (v > best_val) {
                best = c;
                best_val = v;
            }
        }
        preds[t] = best;
    }

    // Debug: print first 5 tokens' logits + IDs for diff-testing
    if (getenv("FIREREDPUNC_DEBUG")) {
        fprintf(stderr, "fireredpunc: %d tokens, %d classes\n", N, ctx.n_classes);
        for (int t = 0; t < N; t++) {
            fprintf(stderr, "  [%d] id=%d pred=%d logits=[", t, token_ids[t], preds[t]);
            for (int c = 0; c < ctx.n_classes; c++)
                fprintf(stderr, "%s%.4f", c ? ", " : "", logits_buf[t * ctx.n_classes + c]);
            fprintf(stderr, "]\n");
        }
    }

    ggml_free(ctx0);
    return preds;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

fireredpunc_context* fireredpunc_init(const char* model_path) {
    auto* ctx = new fireredpunc_context();
    if (!fireredpunc_load(*ctx, model_path)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

char* fireredpunc_process(fireredpunc_context* ctx, const char* text) {
    if (!ctx || !text)
        return nullptr;

    // Tokenize
    std::string input(text);
    std::vector<int> token_ids = ctx->tokenizer.tokenize(input);
    if (token_ids.empty()) {
        char* out = (char*)malloc(strlen(text) + 1);
        strcpy(out, text);
        return out;
    }

    // Chunk if longer than max_pos - 2 (CLS + room)
    const int max_chunk = ctx->max_pos - 2;
    std::vector<int> all_preds;

    for (int offset = 0; offset < (int)token_ids.size(); offset += max_chunk) {
        int end = std::min(offset + max_chunk, (int)token_ids.size());
        std::vector<int> chunk(token_ids.begin() + offset, token_ids.begin() + end);
        std::vector<int> preds = fireredpunc_run(*ctx, chunk);
        all_preds.insert(all_preds.end(), preds.begin(), preds.end());
    }

    // Reconstruct text with punctuation
    // First, get the tokens as strings for WordPiece reassembly
    std::string result;
    std::vector<std::string> words;
    // Split input on whitespace to match tokenizer
    {
        std::string cur;
        for (char c : input) {
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
    }

    // Map predictions back to words. Multiple token predictions per word
    // (from WordPiece splits) — take the prediction of the last subtoken.
    int tok_idx = 0;
    for (size_t w = 0; w < words.size(); w++) {
        if (w > 0)
            result += ' ';
        result += words[w];

        // Count how many subtokens this word consumed
        // Re-tokenize just this word to count
        std::string lword;
        for (char c : words[w]) {
            if (c >= 'A' && c <= 'Z')
                lword += (char)(c - 'A' + 'a');
            else
                lword += c;
        }

        // Find how many tokens this word produced.
        // Must match the tokenizer's splitting exactly.
        int n_subtokens;
        if (ctx->tokenizer.is_sentencepiece) {
            // SentencePiece: prefix with ▁, then greedy longest match
            std::string sp_word = "\xE2\x96\x81" + lword;
            n_subtokens = 0;
            size_t start = 0;
            while (start < sp_word.size()) {
                size_t end = sp_word.size();
                bool found = false;
                while (end > start) {
                    std::string sub = sp_word.substr(start, end - start);
                    if (ctx->tokenizer.token_to_id.count(sub)) {
                        found = true;
                        start = end;
                        n_subtokens++;
                        break;
                    }
                    end--;
                    while (end > start && (sp_word[end] & 0xC0) == 0x80)
                        end--;
                }
                if (!found) {
                    n_subtokens++;
                    break;
                }
            }
        } else {
            // WordPiece: try whole word, then greedy ## splitting
            n_subtokens = 0;
            auto it = ctx->tokenizer.token_to_id.find(lword);
            if (it != ctx->tokenizer.token_to_id.end()) {
                n_subtokens = 1;
            } else {
                size_t start = 0;
                while (start < lword.size()) {
                    size_t end = lword.size();
                    bool found = false;
                    while (end > start) {
                        std::string sub =
                            (start == 0) ? lword.substr(0, end) : ("##" + lword.substr(start, end - start));
                        if (ctx->tokenizer.token_to_id.count(sub)) {
                            found = true;
                            start = end;
                            n_subtokens++;
                            break;
                        }
                        end--;
                        while (end > start && (lword[end] & 0xC0) == 0x80)
                            end--;
                    }
                    if (!found) {
                        n_subtokens++;
                        break;
                    }
                }
            }
        }

        // Take the last subtoken's prediction for this word
        int pred_idx = tok_idx + n_subtokens - 1;
        tok_idx += n_subtokens;

        if (pred_idx < (int)all_preds.size()) {
            int pred = all_preds[pred_idx];
            if (pred > 0 && pred < (int)ctx->labels.size()) {
                result += ctx->labels[pred];
            }
        }
    }

    // Auto-detect Latin script → map Chinese full-width punctuation to ASCII.
    // Count Latin letters vs CJK characters to decide.
    {
        int latin = 0, cjk = 0;
        for (size_t i = 0; i < result.size(); i++) {
            unsigned char c = (unsigned char)result[i];
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
                latin++;
            else if (c >= 0xE0)
                cjk++; // rough: 3-byte+ UTF-8 = CJK/fullwidth
        }
        if (latin > cjk) {
            // Replace full-width punctuation with ASCII equivalents
            std::string mapped;
            mapped.reserve(result.size());
            for (size_t i = 0; i < result.size();) {
                unsigned char b0 = (unsigned char)result[i];
                if (b0 >= 0xE0 && i + 2 < result.size()) {
                    unsigned char b1 = (unsigned char)result[i + 1];
                    unsigned char b2 = (unsigned char)result[i + 2];
                    if (b0 == 0xEF && b1 == 0xBC && b2 == 0x8C) {
                        mapped += ",";
                        i += 3;
                        continue;
                    } // ，
                    if (b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) {
                        mapped += ".";
                        i += 3;
                        continue;
                    } // 。
                    if (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F) {
                        mapped += "?";
                        i += 3;
                        continue;
                    } // ？
                    if (b0 == 0xEF && b1 == 0xBC && b2 == 0x81) {
                        mapped += "!";
                        i += 3;
                        continue;
                    } // ！
                }
                mapped += result[i++];
            }
            result = mapped;
        }
    }

    // Apply rule-based fixes
    // 1. Capitalize after sentence-ending punctuation
    // 2. Capitalize first letter
    // 3. Capitalize "i" when standalone
    std::string fixed;
    bool cap_next = true;
    for (size_t i = 0; i < result.size(); i++) {
        char c = result[i];
        if (cap_next && c >= 'a' && c <= 'z') {
            fixed += (char)(c - 'a' + 'A');
            cap_next = false;
        } else {
            fixed += c;
            // Check for sentence enders (. ? ! and their full-width versions)
            if (c == '.' || c == '?' || c == '!') {
                cap_next = true;
            }
            // Full-width: 。(E3 80 82), ？(EF BC 9F), ！(EF BC 81)
            if (i >= 2) {
                unsigned char b0 = (unsigned char)result[i - 2];
                unsigned char b1 = (unsigned char)result[i - 1];
                unsigned char b2 = (unsigned char)result[i];
                if ((b0 == 0xE3 && b1 == 0x80 && b2 == 0x82) || (b0 == 0xEF && b1 == 0xBC && b2 == 0x9F) ||
                    (b0 == 0xEF && b1 == 0xBC && b2 == 0x81)) {
                    cap_next = true;
                }
            }
        }
    }

    // Capitalize standalone "i" -> "I"
    for (size_t i = 0; i < fixed.size(); i++) {
        if (fixed[i] == 'i') {
            bool start = (i == 0 || fixed[i - 1] == ' ');
            bool end = (i + 1 >= fixed.size() || fixed[i + 1] == ' ' || fixed[i + 1] == ',' || fixed[i + 1] == '.' ||
                        fixed[i + 1] == '?' || fixed[i + 1] == '!');
            // Also check full-width punctuation
            if (start && end) {
                fixed[i] = 'I';
            }
        }
    }

    char* out = (char*)malloc(fixed.size() + 1);
    memcpy(out, fixed.c_str(), fixed.size() + 1);
    return out;
}

void fireredpunc_free(fireredpunc_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}
