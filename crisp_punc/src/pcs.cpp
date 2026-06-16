// pcs.cpp — Punctuation + Capitalization + Segmentation.
//
// Architecture: XLM-RoBERTa-base (12L, d=768, 12 heads, d_ffn=3072, GELU)
//               + 4 classification heads:
//                 post_punc:  Linear(768→256→17)  post-word punctuation
//                 pre_punc:   Linear(768→256→2)   pre-word punctuation (¿)
//                 sbd:        Linear(772→128→2)   sentence boundary detection
//                 truecase:   Linear(769→128→16)  per-character upper/lower
//
// Based on 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase.
// The encoder is identical to the fireredpunc XLM-RoBERTa path.

#include "crisp_punc.h"

#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SentencePiece tokenizer (greedy longest-match, same as fireredpunc SP path)
// ---------------------------------------------------------------------------

namespace {

struct SPTokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    int unk_id = 3;
    int cls_id = 0; // <s>
    int sep_id = 2; // </s>
    int pad_id = 1;

    void build_map() {
        token_to_id.clear();
        for (int i = 0; i < (int)id_to_token.size(); i++)
            token_to_id[id_to_token[i]] = i;
    }

    int lookup(const std::string& tok) const {
        auto it = token_to_id.find(tok);
        return it != token_to_id.end() ? it->second : unk_id;
    }

    std::vector<int> tokenize(const std::string& text) const {
        std::vector<int> ids;
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

        for (const auto& word : words) {
            std::string w = "\xE2\x96\x81" + word; // ▁ prefix
            size_t start = 0;
            while (start < w.size()) {
                size_t end = w.size();
                int best_id = -1;
                while (end > start) {
                    std::string sub = w.substr(start, end - start);
                    auto it = token_to_id.find(sub);
                    if (it != token_to_id.end()) {
                        best_id = it->second;
                        break;
                    }
                    end--;
                    while (end > start && (w[end] & 0xC0) == 0x80)
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

struct pcs_context {
    int d_model = 768;
    int d_ffn = 3072;
    int n_heads = 12;
    int n_layers = 12;
    int vocab_size = 250002;
    int max_pos = 514;
    int n_post_labels = 17;
    int n_pre_labels = 2;
    int pad_id = 1;
    int cls_id = 0;

    std::vector<std::string> post_labels;
    std::vector<std::string> pre_labels;

    SPTokenizer tokenizer;

    // Embeddings
    ggml_tensor* tok_emb_w = nullptr;
    ggml_tensor* pos_emb_w = nullptr;
    ggml_tensor* type_emb_w = nullptr;
    ggml_tensor* emb_ln_w = nullptr;
    ggml_tensor* emb_ln_b = nullptr;

    std::vector<BertLayer> layers;

    // Post-punc head: Linear(768→256) + Linear(256→17)
    ggml_tensor* post_fc1_w = nullptr;
    ggml_tensor* post_fc1_b = nullptr;
    ggml_tensor* post_fc2_w = nullptr;
    ggml_tensor* post_fc2_b = nullptr;

    // Post-punc embedding (17→4) for conditioning SBD
    ggml_tensor* post_emb_w = nullptr;

    // Pre-punc head: Linear(768→256) + Linear(256→2)
    ggml_tensor* pre_fc1_w = nullptr;
    ggml_tensor* pre_fc1_b = nullptr;
    ggml_tensor* pre_fc2_w = nullptr;
    ggml_tensor* pre_fc2_b = nullptr;

    // SBD head: Linear(772→128) + Linear(128→2)
    ggml_tensor* sbd_fc1_w = nullptr;
    ggml_tensor* sbd_fc1_b = nullptr;
    ggml_tensor* sbd_fc2_w = nullptr;
    ggml_tensor* sbd_fc2_b = nullptr;

    // Truecase head: Linear(769→128) + Linear(128→16)
    ggml_tensor* tc_fc1_w = nullptr;
    ggml_tensor* tc_fc1_b = nullptr;
    ggml_tensor* tc_fc2_w = nullptr;
    ggml_tensor* tc_fc2_b = nullptr;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context* w_ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;
};

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

static bool pcs_load(pcs_context& ctx, const char* path) {
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta)
        return false;

    ctx.d_model = (int)core_gguf::kv_u32(meta, "pcs.d_model", 768);
    ctx.d_ffn = (int)core_gguf::kv_u32(meta, "pcs.d_ffn", 3072);
    ctx.n_heads = (int)core_gguf::kv_u32(meta, "pcs.n_heads", 12);
    ctx.n_layers = (int)core_gguf::kv_u32(meta, "pcs.n_layers", 12);
    ctx.vocab_size = (int)core_gguf::kv_u32(meta, "pcs.vocab_size", 250002);
    ctx.max_pos = (int)core_gguf::kv_u32(meta, "pcs.max_pos", 514);
    ctx.n_post_labels = (int)core_gguf::kv_u32(meta, "pcs.n_post_labels", 17);
    ctx.n_pre_labels = (int)core_gguf::kv_u32(meta, "pcs.n_pre_labels", 2);
    ctx.pad_id = (int)core_gguf::kv_u32(meta, "pcs.pad_id", 1);
    ctx.cls_id = (int)core_gguf::kv_u32(meta, "pcs.cls_id", 0);

    ctx.tokenizer.id_to_token = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    ctx.tokenizer.cls_id = ctx.cls_id;
    ctx.tokenizer.pad_id = ctx.pad_id;
    ctx.tokenizer.build_map();

    ctx.post_labels = core_gguf::kv_str_array(meta, "pcs.post_labels");
    ctx.pre_labels = core_gguf::kv_str_array(meta, "pcs.pre_labels");

    core_gguf::free_metadata(meta);

    fprintf(stderr, "pcs: %dL, d=%d, ffn=%d, heads=%d, vocab=%d, post=%d, pre=%d\n", ctx.n_layers, ctx.d_model,
            ctx.d_ffn, ctx.n_heads, ctx.vocab_size, ctx.n_post_labels, ctx.n_pre_labels);

    // Load weights
    ctx.backend = ggml_backend_init_best();
    if (!ctx.backend)
        ctx.backend = ggml_backend_cpu_init();
    ctx.backend_cpu = ggml_backend_cpu_init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "pcs", wl))
        return false;
    ctx.w_ctx = wl.ctx;
    ctx.buf = wl.buf;

    auto& T = wl.tensors;
    auto req = [&](const char* n) { return core_gguf::require(T, n, "pcs"); };
    auto opt = [&](const char* n) -> ggml_tensor* {
        auto it = T.find(n);
        return it != T.end() ? it->second : nullptr;
    };

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

    // Heads
    ctx.post_fc1_w = req("head.post.fc1.weight");
    ctx.post_fc1_b = opt("head.post.fc1.bias");
    ctx.post_fc2_w = req("head.post.fc2.weight");
    ctx.post_fc2_b = opt("head.post.fc2.bias");
    ctx.post_emb_w = opt("head.post_emb.weight");

    ctx.pre_fc1_w = req("head.pre.fc1.weight");
    ctx.pre_fc1_b = opt("head.pre.fc1.bias");
    ctx.pre_fc2_w = req("head.pre.fc2.weight");
    ctx.pre_fc2_b = opt("head.pre.fc2.bias");

    ctx.sbd_fc1_w = req("head.sbd.fc1.weight");
    ctx.sbd_fc1_b = opt("head.sbd.fc1.bias");
    ctx.sbd_fc2_w = req("head.sbd.fc2.weight");
    ctx.sbd_fc2_b = opt("head.sbd.fc2.bias");

    ctx.tc_fc1_w = req("head.tc.fc1.weight");
    ctx.tc_fc1_b = opt("head.tc.fc1.bias");
    ctx.tc_fc2_w = req("head.tc.fc2.weight");
    ctx.tc_fc2_b = opt("head.tc.fc2.bias");

    // Scheduler
    ggml_backend_t backends[2] = {ctx.backend, nullptr};
    int n_backends = 1;
    if (ctx.backend_cpu && ctx.backend_cpu != ctx.backend) {
        backends[n_backends++] = ctx.backend_cpu;
    }
    ctx.sched = ggml_backend_sched_new(backends, nullptr, n_backends, 8192, false, false);

    return true;
}

// ---------------------------------------------------------------------------
// Graph build: encoder + 4 heads
// ---------------------------------------------------------------------------

struct PCSResult {
    std::vector<int> post_preds;              // [N] — post-punctuation class per token
    std::vector<int> pre_preds;               // [N] — pre-punctuation class per token
    std::vector<bool> sbd_preds;              // [N] — sentence boundary per token
    std::vector<std::vector<bool>> cap_preds; // [N][max_chars] — per-char upper/lower
};

static PCSResult pcs_run(pcs_context& ctx, const std::vector<int>& token_ids) {
    const int N = (int)token_ids.size();
    const int seq_len = N + 2; // CLS + tokens + SEP
    const int d = ctx.d_model;
    const int head_dim = d / ctx.n_heads;
    const int nh = ctx.n_heads;

    size_t mem = ggml_tensor_overhead() * (ctx.n_layers * 40 + 100) + 2 * 1024 * 1024;
    struct ggml_init_params gp = {mem, nullptr, true};
    ggml_context* ctx0 = ggml_init(gp);

    // Inputs
    ggml_tensor* inp_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(inp_ids, "inp_ids");
    ggml_set_input(inp_ids);

    ggml_tensor* pos_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    ggml_tensor* type_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(type_ids, "type_ids");
    ggml_set_input(type_ids);

    // Embeddings
    ggml_tensor* tok_emb = ggml_get_rows(ctx0, ctx.tok_emb_w, inp_ids);
    ggml_tensor* pos_emb = ggml_get_rows(ctx0, ctx.pos_emb_w, pos_ids);
    ggml_tensor* type_emb = ggml_get_rows(ctx0, ctx.type_emb_w, type_ids);

    ggml_tensor* emb = ggml_add(ctx0, ggml_add(ctx0, tok_emb, pos_emb), type_emb);
    emb = ggml_norm(ctx0, emb, 1e-12f);
    emb = ggml_add(ctx0, ggml_mul(ctx0, emb, ctx.emb_ln_w), ctx.emb_ln_b);

    // Encoder
    ggml_tensor* cur = emb;
    for (int i = 0; i < ctx.n_layers; i++) {
        const auto& L = ctx.layers[i];
        ggml_tensor* residual = cur;

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_q_w, cur), L.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_k_w, cur), L.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_v_w, cur), L.attn_v_b);

        Q = ggml_reshape_3d(ctx0, Q, head_dim, nh, seq_len);
        K = ggml_reshape_3d(ctx0, K, head_dim, nh, seq_len);
        V = ggml_reshape_3d(ctx0, V, head_dim, nh, seq_len);

        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        const float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* KQV = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        KQV = ggml_reshape_2d(ctx0, KQV, d, seq_len);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_out_w, KQV), L.attn_out_b);
        cur = ggml_add(ctx0, cur, residual);
        cur = ggml_norm(ctx0, cur, 1e-12f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.attn_ln_w), L.attn_ln_b);

        residual = cur;
        ggml_tensor* ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_up_w, cur), L.ffn_up_b);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_down_w, ffn), L.ffn_down_b);

        cur = ggml_add(ctx0, ffn, residual);
        cur = ggml_norm(ctx0, cur, 1e-12f);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.ffn_ln_w), L.ffn_ln_b);
    }

    // Remove CLS: [d, seq_len] → [d, N] starting at position 1
    ggml_tensor* hidden = ggml_view_2d(ctx0, cur, d, N, cur->nb[1], cur->nb[1]);

    // --- Post-punc head: Linear(768→256, ReLU) → Linear(256→17) ---
    ggml_tensor* post_h = ggml_mul_mat(ctx0, ctx.post_fc1_w, hidden);
    if (ctx.post_fc1_b)
        post_h = ggml_add(ctx0, post_h, ctx.post_fc1_b);
    post_h = ggml_relu(ctx0, post_h);
    ggml_tensor* post_logits = ggml_mul_mat(ctx0, ctx.post_fc2_w, post_h);
    if (ctx.post_fc2_b)
        post_logits = ggml_add(ctx0, post_logits, ctx.post_fc2_b);
    ggml_set_name(post_logits, "post_logits");
    ggml_set_output(post_logits);

    // --- Pre-punc head: Linear(768→256, ReLU) → Linear(256→2) ---
    ggml_tensor* pre_h = ggml_mul_mat(ctx0, ctx.pre_fc1_w, hidden);
    if (ctx.pre_fc1_b)
        pre_h = ggml_add(ctx0, pre_h, ctx.pre_fc1_b);
    pre_h = ggml_relu(ctx0, pre_h);
    ggml_tensor* pre_logits = ggml_mul_mat(ctx0, ctx.pre_fc2_w, pre_h);
    if (ctx.pre_fc2_b)
        pre_logits = ggml_add(ctx0, pre_logits, ctx.pre_fc2_b);
    ggml_set_name(pre_logits, "pre_logits");
    ggml_set_output(pre_logits);

    // --- SBD head: cat(hidden, post_emb) → Linear(772→128, ReLU) → Linear(128→2) ---
    // First compute post-punc argmax, then lookup embedding
    // We'll handle SBD and truecase in CPU post-processing for now,
    // since argmax + embedding lookup mid-graph is complex.
    // For the initial implementation, we output post_logits and pre_logits
    // from the graph, then do argmax + SBD + truecase heads on CPU.

    // Actually, let's output the hidden states too so we can do the heads on CPU
    ggml_set_name(hidden, "hidden");
    ggml_set_output(hidden);

    // Build & compute
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, post_logits);
    ggml_build_forward_expand(gf, pre_logits);
    ggml_build_forward_expand(gf, hidden);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
        fprintf(stderr, "pcs: failed to allocate graph\n");
        ggml_free(ctx0);
        return {};
    }

    // Set inputs
    {
        std::vector<int32_t> ids(seq_len);
        ids[0] = ctx.cls_id;
        for (int i = 0; i < N; i++)
            ids[i + 1] = token_ids[i];
        ids[N + 1] = ctx.tokenizer.sep_id;
        ggml_backend_tensor_set(inp_ids, ids.data(), 0, seq_len * sizeof(int32_t));

        std::vector<int32_t> pos(seq_len);
        const int pos_offset = ctx.pad_id + 1; // RoBERTa: positions start at pad_id+1=2
        for (int i = 0; i < seq_len; i++)
            pos[i] = i + pos_offset;
        ggml_backend_tensor_set(pos_ids, pos.data(), 0, seq_len * sizeof(int32_t));

        std::vector<int32_t> types(seq_len, 0);
        ggml_backend_tensor_set(type_ids, types.data(), 0, seq_len * sizeof(int32_t));
    }

    ggml_backend_sched_graph_compute(ctx.sched, gf);

    // Read outputs
    PCSResult result;
    result.post_preds.resize(N);
    result.pre_preds.resize(N);
    result.sbd_preds.resize(N);
    result.cap_preds.resize(N);

    // Post-punc logits: [n_post, N] — argmax per token
    {
        std::vector<float> buf(ctx.n_post_labels * N);
        ggml_backend_tensor_get(post_logits, buf.data(), 0, buf.size() * sizeof(float));
        for (int t = 0; t < N; t++) {
            int best = 0;
            float best_val = buf[t * ctx.n_post_labels];
            for (int c = 1; c < ctx.n_post_labels; c++) {
                float v = buf[t * ctx.n_post_labels + c];
                if (v > best_val) {
                    best = c;
                    best_val = v;
                }
            }
            result.post_preds[t] = best;
        }
    }

    // Pre-punc logits: [n_pre, N] — argmax per token
    {
        std::vector<float> buf(ctx.n_pre_labels * N);
        ggml_backend_tensor_get(pre_logits, buf.data(), 0, buf.size() * sizeof(float));
        for (int t = 0; t < N; t++) {
            result.pre_preds[t] = (buf[t * ctx.n_pre_labels + 1] > buf[t * ctx.n_pre_labels]) ? 1 : 0;
        }
    }

    // Read hidden states for SBD + truecase CPU heads
    std::vector<float> hidden_buf(d * N);
    ggml_backend_tensor_get(hidden, hidden_buf.data(), 0, hidden_buf.size() * sizeof(float));

    // Read post_emb weights for SBD conditioning
    std::vector<float> post_emb_data;
    int post_emb_dim = 0;
    if (ctx.post_emb_w) {
        post_emb_dim = (int)ctx.post_emb_w->ne[0]; // ne[0] = embedding_dim (4)
        int n_emb = (int)ctx.post_emb_w->ne[1];    // ne[1] = n_labels (17)
        post_emb_data.resize(n_emb * post_emb_dim);
        ggml_backend_tensor_get(ctx.post_emb_w, post_emb_data.data(), 0, post_emb_data.size() * sizeof(float));
    }


    // CPU forward for SBD and truecase heads
    // SBD: cat(hidden[768], post_emb[4]) → fc1(772→128, ReLU) → fc2(128→2)
    {
        int sbd_in = d + post_emb_dim;           // 768 + 4 = 772
        int sbd_mid = (int)ctx.sbd_fc1_w->ne[1]; // ne[1]=128 (ne[0]=772=input)

        std::vector<float> sbd_fc1_w_data(sbd_in * sbd_mid);
        ggml_backend_tensor_get(ctx.sbd_fc1_w, sbd_fc1_w_data.data(), 0, sbd_fc1_w_data.size() * sizeof(float));
        std::vector<float> sbd_fc1_b_data(sbd_mid, 0.0f);
        if (ctx.sbd_fc1_b)
            ggml_backend_tensor_get(ctx.sbd_fc1_b, sbd_fc1_b_data.data(), 0, sbd_fc1_b_data.size() * sizeof(float));

        int sbd_out = 2;
        std::vector<float> sbd_fc2_w_data(sbd_mid * sbd_out);
        ggml_backend_tensor_get(ctx.sbd_fc2_w, sbd_fc2_w_data.data(), 0, sbd_fc2_w_data.size() * sizeof(float));
        std::vector<float> sbd_fc2_b_data(sbd_out, 0.0f);
        if (ctx.sbd_fc2_b)
            ggml_backend_tensor_get(ctx.sbd_fc2_b, sbd_fc2_b_data.data(), 0, sbd_fc2_b_data.size() * sizeof(float));

        for (int t = 0; t < N; t++) {
            // Build input: [hidden[t], post_emb[post_pred[t]]]
            std::vector<float> inp(sbd_in);
            for (int j = 0; j < d; j++)
                inp[j] = hidden_buf[t * d + j];
            if (post_emb_dim > 0) {
                int pred = result.post_preds[t];
                for (int j = 0; j < post_emb_dim; j++)
                    inp[d + j] = post_emb_data[pred * post_emb_dim + j];
            }

            // fc1 + ReLU
            std::vector<float> mid(sbd_mid);
            for (int j = 0; j < sbd_mid; j++) {
                float sum = sbd_fc1_b_data[j];
                for (int k = 0; k < sbd_in; k++)
                    sum += inp[k] * sbd_fc1_w_data[j * sbd_in + k];
                mid[j] = sum > 0 ? sum : 0; // ReLU
            }

            // fc2
            float logit0 = sbd_fc2_b_data[0], logit1 = sbd_fc2_b_data[1];
            for (int k = 0; k < sbd_mid; k++) {
                logit0 += mid[k] * sbd_fc2_w_data[0 * sbd_mid + k];
                logit1 += mid[k] * sbd_fc2_w_data[1 * sbd_mid + k];
            }
            result.sbd_preds[t] = logit1 > logit0;
        }
    }

    // Truecase: cat(hidden[768], sbd[1]) → fc1(769→128, ReLU) → fc2(128→16)
    {
        int tc_in = d + 1;                     // 769
        int tc_mid = (int)ctx.tc_fc1_w->ne[1]; // ne[1]=128 (ne[0]=769=input)
        int tc_out = 16;

        std::vector<float> tc_fc1_w_data(tc_in * tc_mid);
        ggml_backend_tensor_get(ctx.tc_fc1_w, tc_fc1_w_data.data(), 0, tc_fc1_w_data.size() * sizeof(float));
        std::vector<float> tc_fc1_b_data(tc_mid, 0.0f);
        if (ctx.tc_fc1_b)
            ggml_backend_tensor_get(ctx.tc_fc1_b, tc_fc1_b_data.data(), 0, tc_fc1_b_data.size() * sizeof(float));

        std::vector<float> tc_fc2_w_data(tc_mid * tc_out);
        ggml_backend_tensor_get(ctx.tc_fc2_w, tc_fc2_w_data.data(), 0, tc_fc2_w_data.size() * sizeof(float));
        std::vector<float> tc_fc2_b_data(tc_out, 0.0f);
        if (ctx.tc_fc2_b)
            ggml_backend_tensor_get(ctx.tc_fc2_b, tc_fc2_b_data.data(), 0, tc_fc2_b_data.size() * sizeof(float));

        for (int t = 0; t < N; t++) {
            std::vector<float> inp(tc_in);
            for (int j = 0; j < d; j++)
                inp[j] = hidden_buf[t * d + j];
            inp[d] = result.sbd_preds[t] ? 1.0f : 0.0f;

            std::vector<float> mid(tc_mid);
            for (int j = 0; j < tc_mid; j++) {
                float sum = tc_fc1_b_data[j];
                for (int k = 0; k < tc_in; k++)
                    sum += inp[k] * tc_fc1_w_data[j * tc_in + k];
                mid[j] = sum > 0 ? sum : 0;
            }

            result.cap_preds[t].resize(tc_out);
            for (int c = 0; c < tc_out; c++) {
                float logit = tc_fc2_b_data[c];
                for (int k = 0; k < tc_mid; k++)
                    logit += mid[k] * tc_fc2_w_data[c * tc_mid + k];
                result.cap_preds[t][c] = logit > 0;
            }
        }
    }

    ggml_free(ctx0);
    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pcs_context* pcs_init(const char* model_path) {
    auto* ctx = new pcs_context();
    if (!pcs_load(*ctx, model_path)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

char* pcs_process(pcs_context* ctx, const char* text) {
    if (!ctx || !text)
        return nullptr;

    std::string input(text);
    std::vector<int> token_ids = ctx->tokenizer.tokenize(input);
    if (token_ids.empty()) {
        char* out = (char*)malloc(strlen(text) + 1);
        strcpy(out, text);
        return out;
    }

    // Chunk if needed
    const int max_chunk = ctx->max_pos - 2;
    PCSResult all_result;

    for (int offset = 0; offset < (int)token_ids.size(); offset += max_chunk) {
        int end = std::min(offset + max_chunk, (int)token_ids.size());
        std::vector<int> chunk(token_ids.begin() + offset, token_ids.begin() + end);
        PCSResult r = pcs_run(*ctx, chunk);
        all_result.post_preds.insert(all_result.post_preds.end(), r.post_preds.begin(), r.post_preds.end());
        all_result.pre_preds.insert(all_result.pre_preds.end(), r.pre_preds.begin(), r.pre_preds.end());
        all_result.sbd_preds.insert(all_result.sbd_preds.end(), r.sbd_preds.begin(), r.sbd_preds.end());
        all_result.cap_preds.insert(all_result.cap_preds.end(), r.cap_preds.begin(), r.cap_preds.end());
    }

    // Reconstruct text: map subtokens back to words, apply punc + case
    std::vector<std::string> words;
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

    // Map subtokens to words and apply truecasing + punctuation
    std::string result;
    int tok_idx = 0;

    for (size_t w = 0; w < words.size(); w++) {
        // Count subtokens for this word
        std::string sp_word = "\xE2\x96\x81" + words[w];
        int n_subtokens = 0;
        {
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
        }

        // Apply pre-punctuation from first subtoken
        if (tok_idx < (int)all_result.pre_preds.size()) {
            int pre = all_result.pre_preds[tok_idx];
            if (pre > 0 && pre < (int)ctx->pre_labels.size()) {
                const std::string& label = ctx->pre_labels[pre];
                if (label != "<NULL>")
                    result += label;
            }
        }

        // Apply truecasing: collect cap predictions across all subtokens
        // Each subtoken has up to 16 chars of case prediction
        std::string cased_word;
        int char_idx = 0; // index into the original word's characters
        for (int st = 0; st < n_subtokens && tok_idx + st < (int)all_result.cap_preds.size(); st++) {
            const auto& caps = all_result.cap_preds[tok_idx + st];
            // The subtoken text (without ▁ prefix for first)
            std::string subtoken_text;
            if (st == 0) {
                // First subtoken includes ▁, skip it
                subtoken_text = words[w]; // We'll process the whole word char by char
            }

            // Apply case predictions character by character
            // caps[c] = true means uppercase at char position c within this subtoken
            int sub_char = 0;
            // Skip ▁ in the first subtoken's prediction
            if (st == 0)
                sub_char = 0; // predictions start after ▁

            // Get the actual text this subtoken represents
            // For simplicity, we process the original word chars sequentially
        }

        // Simpler approach: apply truecasing per character of the word
        // by walking subtokens and their per-char predictions
        cased_word.clear();
        {
            int global_char = 0; // character index within the word
            int sub_start = tok_idx;
            for (int st = 0; st < n_subtokens && sub_start + st < (int)all_result.cap_preds.size(); st++) {
                const auto& caps = all_result.cap_preds[sub_start + st];
                // Determine how many chars this subtoken covers
                // The SP token includes ▁ for the first subtoken of a word
                std::string tok_text;
                if (sub_start + st < (int)token_ids.size()) {
                    int tid = token_ids[sub_start + st];
                    if (tid >= 0 && tid < (int)ctx->tokenizer.id_to_token.size())
                        tok_text = ctx->tokenizer.id_to_token[tid];
                }
                // Remove ▁ prefix
                if (tok_text.size() >= 3 && tok_text[0] == '\xE2' && tok_text[1] == '\x96' && tok_text[2] == '\x81')
                    tok_text = tok_text.substr(3);

                // Map each character of the subtoken to the word
                size_t ti = 0;
                int cap_idx = (st == 0) ? 1 : 0; // skip ▁ position for first subtoken
                while (ti < tok_text.size() && global_char < (int)words[w].size()) {
                    // Get the UTF-8 character length
                    unsigned char c0 = (unsigned char)words[w][global_char];
                    int clen = 1;
                    if (c0 >= 0xC0)
                        clen = 2;
                    if (c0 >= 0xE0)
                        clen = 3;
                    if (c0 >= 0xF0)
                        clen = 4;

                    // Apply case
                    bool should_upper = (cap_idx < (int)caps.size()) ? caps[cap_idx] : false;
                    if (should_upper && clen == 1 && words[w][global_char] >= 'a' && words[w][global_char] <= 'z') {
                        cased_word += (char)(words[w][global_char] - 'a' + 'A');
                    } else if (!should_upper && clen == 1 && words[w][global_char] >= 'A' &&
                               words[w][global_char] <= 'Z') {
                        cased_word += (char)(words[w][global_char] - 'A' + 'a');
                    } else {
                        cased_word += words[w].substr(global_char, clen);
                    }

                    global_char += clen;
                    ti += clen;
                    cap_idx++;
                }
            }
            // Append any remaining characters
            while (global_char < (int)words[w].size()) {
                cased_word += words[w][global_char++];
            }
        }

        if (w > 0)
            result += ' ';
        result += cased_word;

        // Apply post-punctuation from last subtoken
        int last_tok = tok_idx + n_subtokens - 1;
        if (last_tok < (int)all_result.post_preds.size()) {
            int post = all_result.post_preds[last_tok];
            if (post > 0 && post < (int)ctx->post_labels.size()) {
                const std::string& label = ctx->post_labels[post];
                if (label == "<ACRONYM>") {
                    // Insert period after each character — skip for now
                } else if (label != "<NULL>") {
                    result += label;
                }
            }
        }

        tok_idx += n_subtokens;
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

void pcs_free(pcs_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}
