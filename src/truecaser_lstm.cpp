// truecaser_lstm.cpp — BiLSTM character-level truecaser.
//
// Architecture: Embedding(vocab, 50) → BiLSTM(50→150, 2 layers) → Linear(300→2)
// Labels: 0=L (lowercase), 1=U (uppercase) — per character
// Based on mayhewsw/pytorch-truecaser (Apache-2.0).
//
// Binary format (little-endian):
//   "LSTM" magic (4 bytes)
//   uint32 vocab_size, embed_dim, hidden_size, n_layers
//   vocab: vocab_size × (uint16 len + UTF-8 bytes)
//   embedding: float32[vocab_size × embed_dim]
//   for each layer × direction (fwd, rev):
//     weight_ih: float32[4*hidden × input_size]
//     weight_hh: float32[4*hidden × hidden_size]
//     bias_ih: float32[4*hidden]
//     bias_hh: float32[4*hidden]
//   projection: weight float32[2 × 300], bias float32[2]

#include "truecaser_lstm.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct LSTMLayer {
    // weight_ih: [4*H, input], weight_hh: [4*H, H], bias_ih: [4*H], bias_hh: [4*H]
    std::vector<float> wih, whh, bih, bhh;
    int input_size;
    int hidden_size;
};

struct truecaser_lstm_context {
    int vocab_size = 0;
    int embed_dim = 0;
    int hidden_size = 0;
    int n_layers = 0;

    std::map<std::string, int> char_to_id;
    int unk_id = 1;

    std::vector<float> embedding; // [vocab_size × embed_dim]
    std::vector<LSTMLayer> fwd_layers;
    std::vector<LSTMLayer> rev_layers;

    // Projection: Linear(2*hidden, 2)
    std::vector<float> proj_w; // [2, 2*hidden]
    std::vector<float> proj_b; // [2]
};

// ---------------------------------------------------------------------------
// LSTM cell: one step
// ---------------------------------------------------------------------------

static void lstm_step(const LSTMLayer& L, const float* x, const float* h_prev, const float* c_prev, float* h_out,
                      float* c_out) {
    const int H = L.hidden_size;
    const int in_sz = L.input_size;

    // gates = W_ih @ x + b_ih + W_hh @ h + b_hh
    // gates layout: [i, f, g, o] each of size H
    std::vector<float> gates(4 * H);
    for (int i = 0; i < 4 * H; i++) {
        float sum = L.bih[i] + L.bhh[i];
        for (int j = 0; j < in_sz; j++)
            sum += L.wih[i * in_sz + j] * x[j];
        for (int j = 0; j < H; j++)
            sum += L.whh[i * H + j] * h_prev[j];
        gates[i] = sum;
    }

    for (int i = 0; i < H; i++) {
        float ig = 1.0f / (1.0f + expf(-gates[i]));         // input gate
        float fg = 1.0f / (1.0f + expf(-gates[H + i]));     // forget gate
        float gg = tanhf(gates[2 * H + i]);                 // cell gate
        float og = 1.0f / (1.0f + expf(-gates[3 * H + i])); // output gate

        c_out[i] = fg * c_prev[i] + ig * gg;
        h_out[i] = og * tanhf(c_out[i]);
    }
}

// ---------------------------------------------------------------------------
// Forward pass: BiLSTM
// ---------------------------------------------------------------------------

// Run one LSTM direction over a sequence. Returns hidden states [T, H].
static std::vector<float> run_lstm_direction(const LSTMLayer& L, const std::vector<float>& input, int T, bool reverse) {
    const int H = L.hidden_size;
    const int in_sz = L.input_size;

    std::vector<float> output(T * H);
    std::vector<float> h(H, 0.0f), c(H, 0.0f);
    std::vector<float> h_new(H), c_new(H);

    for (int step = 0; step < T; step++) {
        int t = reverse ? (T - 1 - step) : step;
        lstm_step(L, &input[t * in_sz], h.data(), c.data(), h_new.data(), c_new.data());
        std::copy(h_new.begin(), h_new.end(), h.begin());
        std::copy(c_new.begin(), c_new.end(), c.begin());
        std::copy(h.begin(), h.end(), &output[t * H]);
    }

    return output;
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

truecaser_lstm_context* truecaser_lstm_init(const char* model_path) {
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "truecaser_lstm: cannot open '%s'\n", model_path);
        return nullptr;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "LSTM", 4) != 0) {
        fprintf(stderr, "truecaser_lstm: bad magic in '%s'\n", model_path);
        fclose(f);
        return nullptr;
    }

    auto* ctx = new truecaser_lstm_context();

    uint32_t vs, ed, hs, nl;
    fread(&vs, 4, 1, f);
    fread(&ed, 4, 1, f);
    fread(&hs, 4, 1, f);
    fread(&nl, 4, 1, f);
    ctx->vocab_size = vs;
    ctx->embed_dim = ed;
    ctx->hidden_size = hs;
    ctx->n_layers = nl;

    // Vocabulary
    for (uint32_t i = 0; i < vs; i++) {
        uint16_t klen;
        fread(&klen, 2, 1, f);
        std::string key(klen, '\0');
        fread(&key[0], 1, klen, f);
        ctx->char_to_id[key] = (int)i;
    }

    // Embedding
    ctx->embedding.resize(vs * ed);
    fread(ctx->embedding.data(), sizeof(float), vs * ed, f);

    // BiLSTM layers
    ctx->fwd_layers.resize(nl);
    ctx->rev_layers.resize(nl);
    for (int layer = 0; layer < (int)nl; layer++) {
        int in_sz = (layer == 0) ? (int)ed : (int)(2 * hs);

        // Forward direction
        auto& fwd = ctx->fwd_layers[layer];
        fwd.input_size = in_sz;
        fwd.hidden_size = hs;
        fwd.wih.resize(4 * hs * in_sz);
        fwd.whh.resize(4 * hs * hs);
        fwd.bih.resize(4 * hs);
        fwd.bhh.resize(4 * hs);
        fread(fwd.wih.data(), sizeof(float), fwd.wih.size(), f);
        fread(fwd.whh.data(), sizeof(float), fwd.whh.size(), f);
        fread(fwd.bih.data(), sizeof(float), fwd.bih.size(), f);
        fread(fwd.bhh.data(), sizeof(float), fwd.bhh.size(), f);

        // Reverse direction
        auto& rev = ctx->rev_layers[layer];
        rev.input_size = in_sz;
        rev.hidden_size = hs;
        rev.wih.resize(4 * hs * in_sz);
        rev.whh.resize(4 * hs * hs);
        rev.bih.resize(4 * hs);
        rev.bhh.resize(4 * hs);
        fread(rev.wih.data(), sizeof(float), rev.wih.size(), f);
        fread(rev.whh.data(), sizeof(float), rev.whh.size(), f);
        fread(rev.bih.data(), sizeof(float), rev.bih.size(), f);
        fread(rev.bhh.data(), sizeof(float), rev.bhh.size(), f);
    }

    // Projection
    int proj_in = 2 * hs;
    ctx->proj_w.resize(2 * proj_in);
    ctx->proj_b.resize(2);
    fread(ctx->proj_w.data(), sizeof(float), ctx->proj_w.size(), f);
    fread(ctx->proj_b.data(), sizeof(float), ctx->proj_b.size(), f);

    fclose(f);
    fprintf(stderr, "truecaser_lstm: loaded %d-char vocab, %d layers, hidden=%d from '%s'\n", ctx->vocab_size,
            ctx->n_layers, ctx->hidden_size, model_path);
    return ctx;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

char* truecaser_lstm_process(truecaser_lstm_context* ctx, const char* text) {
    if (!ctx || !text)
        return nullptr;

    std::string input(text);
    // Lowercase the input for character lookup
    std::string lower;
    lower.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        char c = input[i];
        if (c >= 'A' && c <= 'Z')
            lower += (char)(c - 'A' + 'a');
        else
            lower += c;
    }

    // Convert characters to IDs (one ID per byte for simplicity — matches the
    // Python tokenizer which splits on individual characters)
    const int T = (int)lower.size();
    if (T == 0) {
        char* out = (char*)malloc(1);
        out[0] = '\0';
        return out;
    }

    std::vector<int> ids(T);
    for (int i = 0; i < T; i++) {
        std::string ch(1, lower[i]);
        auto it = ctx->char_to_id.find(ch);
        ids[i] = (it != ctx->char_to_id.end()) ? it->second : ctx->unk_id;
    }

    // Embedding lookup: [T, embed_dim]
    const int E = ctx->embed_dim;
    const int H = ctx->hidden_size;
    std::vector<float> cur(T * E);
    for (int t = 0; t < T; t++) {
        int id = ids[t];
        for (int j = 0; j < E; j++)
            cur[t * E + j] = ctx->embedding[id * E + j];
    }

    // BiLSTM layers
    for (int layer = 0; layer < ctx->n_layers; layer++) {
        int in_sz = (layer == 0) ? E : 2 * H;

        std::vector<float> fwd_out = run_lstm_direction(ctx->fwd_layers[layer], cur, T, false);
        std::vector<float> rev_out = run_lstm_direction(ctx->rev_layers[layer], cur, T, true);

        // Concatenate: [T, 2*H]
        cur.resize(T * 2 * H);
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < H; j++) {
                cur[t * 2 * H + j] = fwd_out[t * H + j];
                cur[t * 2 * H + H + j] = rev_out[t * H + j];
            }
        }
    }

    // Projection: Linear(2*H, 2) → logits per character
    const int proj_in = 2 * H;
    std::vector<int> preds(T);
    for (int t = 0; t < T; t++) {
        float logit_L = ctx->proj_b[0];
        float logit_U = ctx->proj_b[1];
        for (int j = 0; j < proj_in; j++) {
            logit_L += cur[t * proj_in + j] * ctx->proj_w[0 * proj_in + j];
            logit_U += cur[t * proj_in + j] * ctx->proj_w[1 * proj_in + j];
        }
        preds[t] = (logit_U > logit_L) ? 1 : 0;
    }

    // Apply predictions: uppercase characters where pred=U(1)
    std::string result = lower;
    for (int i = 0; i < T; i++) {
        if (preds[i] == 1 && result[i] >= 'a' && result[i] <= 'z') {
            result[i] = (char)(result[i] - 'a' + 'A');
        }
    }

    if (getenv("TRUECASER_DEBUG")) {
        fprintf(stderr, "truecaser_lstm: %d chars, preds:", T);
        for (int i = 0; i < T; i++)
            fprintf(stderr, "%c", preds[i] ? 'U' : 'l');
        fprintf(stderr, "\n");
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

void truecaser_lstm_free(truecaser_lstm_context* ctx) {
    delete ctx;
}
