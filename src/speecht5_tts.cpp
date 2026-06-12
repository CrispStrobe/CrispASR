// speecht5_tts.cpp -- SpeechT5 TTS backend for CrispASR.
//
// Microsoft SpeechT5 (MIT license): ~80M param text-to-speech model
// producing 80-bin mel spectrograms autoregressively, refined by a
// 5-layer conv post-net, then vocoded to 16 kHz PCM via HiFi-GAN.
//
// Forward pass:
//   1. Text encoder: Embedding(81, 768) + ScaledPosEnc +
//      LayerNorm + 12-layer transformer with relative position bias
//      (each layer: self-attn + post-LN + FFN + post-LN)
//   2. Speech decoder (AR loop):
//      - Prenet: 2x (Linear + ReLU) + final_layer + ScaledPosEnc
//        + speaker_embeds_layer(cat(hidden, spk_emb)) + ReLU
//      - 6-layer decoder: self-attn + cross-attn + FFN (all post-LN)
//      - feat_out: Linear(768 -> 160) -> reshape to (reduction=2, 80)
//      - prob_out: Linear(768 -> 2) -> sigmoid -> stop token
//   3. Post-net: 5-layer Conv1d(k=5) + BN + Tanh stack
//   4. HiFi-GAN vocoder: core_hifigan::forward()
//
// Implementation uses per-module ggml sub-graphs (mini_graph pattern
// from piper_tts.cpp). The encoder runs once; the decoder runs in a
// loop generating one reduction frame per step.

#include "speecht5_tts.h"

#include "core/gguf_loader.h"
#include "core/hifigan.h"

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

// ── Helpers ────────────────────────────────────────────────────────

namespace {

static ggml_tensor* W(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

// Read F32 data from a tensor (handles F16 -> F32 dequant).
static void read_tensor_f32(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    const size_t nbytes = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    } else {
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto to_float = ggml_get_type_traits(t->type)->to_float;
        if (to_float) {
            to_float(raw.data(), out.data(), n);
        }
    }
}

// ── Mini graph: build + compute + read ────────────────────────────

struct mini_graph {
    ggml_context* ctx = nullptr;
    ggml_backend_sched_t sched = nullptr;

    mini_graph(ggml_backend_sched_t s, size_t ctx_size = 32 * 1024 * 1024) : sched(s) {
        struct ggml_init_params params = {ctx_size, nullptr, true};
        ctx = ggml_init(params);
    }
    ~mini_graph() {
        if (ctx)
            ggml_free(ctx);
    }

    void set_input(ggml_tensor* t, const void* data, size_t nbytes) { ggml_backend_tensor_set(t, data, 0, nbytes); }
};

// ── Hyperparameters ───────────────────────────────────────────────

struct speecht5_hp {
    int hidden_size = 768;
    int num_mel_bins = 80;
    int encoder_layers = 12;
    int decoder_layers = 6;
    int encoder_attention_heads = 12;
    int decoder_attention_heads = 12;
    int encoder_ffn_dim = 3072;
    int decoder_ffn_dim = 3072;
    int vocab_size = 81;
    int reduction_factor = 2;
    int prenet_layers = 2;
    int prenet_units = 256;
    int postnet_layers = 5;
    int postnet_units = 256;
    int postnet_kernel = 5;
    int speaker_dim = 512;
    int max_text_positions = 450;
    int max_speech_positions = 4000;
    int encoder_max_relative_position = 160;
    float layer_norm_eps = 1e-5f;

    int head_dim() const { return hidden_size / encoder_attention_heads; }
};

// ── Simple char-level tokenizer ───────────────────────────────────
// SpeechT5 uses a sentencepiece char-level tokenizer. For simplicity,
// we store the vocab from GGUF and do character-level lookup.

struct speecht5_tokenizer {
    std::vector<std::string> vocab;
    std::map<std::string, int> token_to_id;
    int pad_id = 1; // SpeechT5 pad_token_id = 1

    bool load(const std::vector<std::string>& v) {
        vocab = v;
        token_to_id.clear();
        for (int i = 0; i < (int)v.size(); i++) {
            token_to_id[v[i]] = i;
        }
        return !vocab.empty();
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids;
        // SpeechT5 uses a SentencePiece char-level tokenizer that prepends ▁
        // at the start of text and replaces spaces with ▁.
        const std::string sp_marker = "\xe2\x96\x81"; // "▁"

        // Prepend ▁ at start (SentencePiece convention)
        auto sp_it = token_to_id.find(sp_marker);
        if (sp_it != token_to_id.end()) {
            ids.push_back(sp_it->second);
        }

        size_t i = 0;
        while (i < text.size()) {
            if (text[i] == ' ') {
                // Spaces become ▁
                if (sp_it != token_to_id.end()) {
                    ids.push_back(sp_it->second);
                }
                i++;
                continue;
            }
            // Try longest match first (for multi-byte UTF-8 chars)
            bool found = false;
            for (int len = 4; len >= 1; len--) {
                if (i + len > text.size())
                    continue;
                std::string sub = text.substr(i, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    ids.push_back(it->second);
                    i += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                i++; // skip unknown characters
            }
        }
        // Append EOS (</s> = id 2 for SpeechT5)
        auto eos_it = token_to_id.find("</s>");
        if (eos_it != token_to_id.end()) {
            ids.push_back(eos_it->second);
        }
        return ids;
    }
};

} // namespace

// ── Context ────────────────────────────────────────────────────────

struct speecht5_tts_context {
    speecht5_hp hp;
    core_hifigan::hparams voc_hp;
    speecht5_tokenizer tokenizer;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    core_gguf::WeightLoad wl;
    std::map<std::string, ggml_tensor*>& tensors() { return wl.tensors; }

    int n_threads = 4;
    int verbosity = 1;
    float threshold = 0.5f;
    int max_len = 0;

    // Pre-permuted HiFi-GAN upsample weights for decomposed col2im path
    std::vector<ggml_tensor*> ups_w_perm;
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;

    // Speaker embedding (512-dim x-vector)
    std::vector<float> speaker_emb;

    // Decoder self-attention KV cache.
    // For each decoder layer, we store accumulated K and V vectors.
    // Layout: [layer][step * hidden_size ... (step+1) * hidden_size]
    // At step N, each layer has N entries of hidden_size floats.
    struct decoder_kv_cache {
        std::vector<std::vector<float>> k; // [n_layers][steps * hidden_size]
        std::vector<std::vector<float>> v; // [n_layers][steps * hidden_size]
        int n_steps = 0;

        void reset(int n_layers) {
            k.assign(n_layers, std::vector<float>());
            v.assign(n_layers, std::vector<float>());
            n_steps = 0;
        }

        void append(int layer, const std::vector<float>& k_vec, const std::vector<float>& v_vec) {
            k[layer].insert(k[layer].end(), k_vec.begin(), k_vec.end());
            v[layer].insert(v[layer].end(), v_vec.begin(), v_vec.end());
        }
    } self_kv_cache;

    ~speecht5_tts_context() {
        core_gguf::free_weights(wl);
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

// ── Sinusoidal positional encoding ────────────────────────────────
// Pre-compute the (max_len, dim) sinusoidal PE table.

static std::vector<float> make_sinusoidal_pe(int max_len, int dim) {
    std::vector<float> pe(max_len * dim, 0.0f);
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < dim; i += 2) {
            float div_term = expf(-(float)i * logf(10000.0f) / (float)dim);
            pe[pos * dim + i] = sinf((float)pos * div_term);
            if (i + 1 < dim) {
                pe[pos * dim + i + 1] = cosf((float)pos * div_term);
            }
        }
    }
    return pe;
}

// ── Text encoder ──────────────────────────────────────────────────

static std::vector<float> run_encoder(speecht5_tts_context* ctx, const std::vector<int32_t>& token_ids, int* out_T) {
    const auto& hp = ctx->hp;
    const auto& ts = ctx->tensors();
    const int T = (int)token_ids.size();
    *out_T = T;

    mini_graph mg(ctx->sched);
    auto* gc = mg.ctx;

    // Input token IDs
    ggml_tensor* inp_ids = ggml_new_tensor_1d(gc, GGML_TYPE_I32, T);
    ggml_set_name(inp_ids, "inp_ids");
    ggml_set_input(inp_ids);

    // Text embedding: lookup
    ggml_tensor* embed_w = W(ts, "enc.embed.weight");
    ggml_tensor* x = ggml_get_rows(gc, embed_w, inp_ids); // (T, hidden_size)

    // Scaled positional encoding: x = x + alpha * PE[:T]
    // PE is computed on CPU, passed as input tensor
    ggml_tensor* pe_input = ggml_new_tensor_2d(gc, GGML_TYPE_F32, hp.hidden_size, T);
    ggml_set_name(pe_input, "enc_pe");
    ggml_set_input(pe_input);

    ggml_tensor* alpha = W(ts, "enc.pos_alpha");
    if (alpha) {
        // Alpha is a scalar — read its value and use ggml_scale
        float alpha_val = 1.0f;
        if (alpha->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(alpha, &alpha_val, 0, sizeof(float));
        } else {
            // F16 scalar
            ggml_fp16_t alpha_f16;
            ggml_backend_tensor_get(alpha, &alpha_f16, 0, sizeof(ggml_fp16_t));
            alpha_val = ggml_fp16_to_fp32(alpha_f16);
        }
        ggml_tensor* scaled_pe = ggml_scale(gc, pe_input, alpha_val);
        x = ggml_add(gc, x, scaled_pe);
    } else {
        x = ggml_add(gc, x, pe_input);
    }

    // Encoder: LayerNorm + dropout(skip at inference) + layers
    ggml_tensor* enc_ln_w = W(ts, "enc.ln.weight");
    ggml_tensor* enc_ln_b = W(ts, "enc.ln.bias");
    if (enc_ln_w) {
        x = ggml_norm(gc, x, hp.layer_norm_eps);
        x = ggml_mul(gc, x, enc_ln_w);
        if (enc_ln_b)
            x = ggml_add(gc, x, enc_ln_b);
    }

    // Relative position bias: pe_k embedding lookup
    // pos_diff[i,j] = i - j, clamped to [-max_rel, max_rel-1], offset by max_rel
    // Result shape: (T, T, head_dim)
    ggml_tensor* rel_pos_w = W(ts, "enc.rel_pos.weight"); // (2*max_rel, head_dim)
    ggml_tensor* rel_pos_ids = ggml_new_tensor_2d(gc, GGML_TYPE_I32, T, T);
    ggml_set_name(rel_pos_ids, "rel_pos_ids");
    ggml_set_input(rel_pos_ids);

    // Position bias: lookup -> (T, T, head_dim)
    ggml_tensor* position_bias = nullptr;
    if (rel_pos_w) {
        // Reshape rel_pos_ids to 1D for get_rows, then reshape back
        ggml_tensor* flat_ids = ggml_reshape_1d(gc, rel_pos_ids, T * T);
        ggml_tensor* flat_bias = ggml_get_rows(gc, rel_pos_w, flat_ids);     // (T*T, head_dim)
        position_bias = ggml_reshape_3d(gc, flat_bias, hp.head_dim(), T, T); // (head_dim, T_key, T_query)
    }

    // Encoder layers
    for (int i = 0; i < hp.encoder_layers; i++) {
        std::string pfx = "enc.layer." + std::to_string(i);

        // Self-attention with relative position bias
        ggml_tensor* residual = x;

        // Q, K, V projections
        ggml_tensor* q_w = W(ts, pfx + ".attn.q.weight");
        ggml_tensor* q_b = W(ts, pfx + ".attn.q.bias");
        ggml_tensor* k_w = W(ts, pfx + ".attn.k.weight");
        ggml_tensor* k_b = W(ts, pfx + ".attn.k.bias");
        ggml_tensor* v_w = W(ts, pfx + ".attn.v.weight");
        ggml_tensor* v_b = W(ts, pfx + ".attn.v.bias");
        ggml_tensor* o_w = W(ts, pfx + ".attn.o.weight");
        ggml_tensor* o_b = W(ts, pfx + ".attn.o.bias");

        // x is (hidden_size, T) in ggml layout
        ggml_tensor* q = ggml_mul_mat(gc, q_w, x);
        if (q_b)
            q = ggml_add(gc, q, q_b);
        // Scale Q: q *= 1/sqrt(head_dim)
        q = ggml_scale(gc, q, 1.0f / sqrtf((float)hp.head_dim()));

        ggml_tensor* k = ggml_mul_mat(gc, k_w, x);
        if (k_b)
            k = ggml_add(gc, k, k_b);

        ggml_tensor* v = ggml_mul_mat(gc, v_w, x);
        if (v_b)
            v = ggml_add(gc, v, v_b);

        // Reshape to multi-head: (hidden_size, T) -> (head_dim, n_heads, T)
        int n_heads = hp.encoder_attention_heads;
        int hd = hp.head_dim();

        q = ggml_reshape_3d(gc, q, hd, n_heads, T);
        k = ggml_reshape_3d(gc, k, hd, n_heads, T);
        v = ggml_reshape_3d(gc, v, hd, n_heads, T);

        // Attention: softmax(Q @ K^T / sqrt(d) + bias) @ V
        // attn_weights: (T, T, n_heads) via permuted bmm
        // Use ggml_flash_attn_ext or manual path
        // For simplicity, use manual path with relative bias support:

        // Q: (hd, n_heads, T) -> permute to (hd, T, n_heads)
        ggml_tensor* q_perm = ggml_cont(gc, ggml_permute(gc, q, 0, 2, 1, 3)); // (hd, T, n_heads)
        ggml_tensor* k_perm = ggml_cont(gc, ggml_permute(gc, k, 0, 2, 1, 3)); // (hd, T, n_heads)
        ggml_tensor* v_perm = ggml_cont(gc, ggml_permute(gc, v, 0, 2, 1, 3)); // (hd, T, n_heads)

        // attn_weights = Q @ K^T: for each head, (T, hd) @ (hd, T) = (T, T)
        // In ggml: mul_mat(K, Q) with K as (hd, T, n_heads) and Q as (hd, T, n_heads)
        // gives (T, T, n_heads)
        ggml_tensor* attn_w = ggml_mul_mat(gc, k_perm, q_perm); // (T, T, n_heads)

        // Add relative position bias (content-dependent, Q-based)
        // HF SpeechT5: rel_bias = matmul(Q, position_bias.T) per batch of query positions
        // In PyTorch: Q is (T, n_heads, hd), position_bias is (T, T, hd)
        //   matmul((T, n_heads, hd), (T, hd, T)) -> (T, n_heads, T)
        //
        // In ggml layout:
        //   q (pre-permute) is (hd, n_heads, T)
        //   position_bias is (hd, T_key, T_query) after our permute above
        //   mul_mat(position_bias, q) contracts on ne[0]=hd for both
        //   Result: (T_key, n_heads, T_query) — need permute to (T_key, T_query, n_heads)
        if (position_bias) {
            ggml_tensor* rel_bias = ggml_mul_mat(gc, position_bias, q); // (T, n_heads, T)
            // Permute to (T, T, n_heads) to match attn_w layout
            rel_bias = ggml_cont(gc, ggml_permute(gc, rel_bias, 0, 2, 1, 3));
            attn_w = ggml_add(gc, attn_w, rel_bias);
        }

        // Softmax
        attn_w = ggml_soft_max(gc, attn_w);

        // attn_output = attn_w @ V
        // attn_w: (T, T, n_heads) — scores per head
        // v_perm: (hd, T, n_heads) — values per head
        // We need: output[d, i, h] = sum_j attn_w[j, i, h] * v[d, j, h]
        // In ggml mul_mat contracts over ne[0], so transpose v to (T, hd, n_heads)
        ggml_tensor* v_t = ggml_cont(gc, ggml_permute(gc, v_perm, 1, 0, 2, 3)); // (T, hd, n_heads)
        ggml_tensor* attn_out = ggml_mul_mat(gc, v_t, attn_w);                  // (hd, T, n_heads)
        // Reshape to (hidden_size, T)
        // attn_out is (hd, T, n_heads). Need to permute to (hd, n_heads, T) before
        // reshape to (hidden=hd*n_heads, T), so heads are concatenated correctly.
        attn_out = ggml_cont(gc, ggml_permute(gc, attn_out, 0, 2, 1, 3)); // (hd, n_heads, T)
        attn_out = ggml_reshape_2d(gc, attn_out, hp.hidden_size, T);

        // Output projection
        ggml_tensor* attn_proj = ggml_mul_mat(gc, o_w, attn_out);
        if (o_b)
            attn_proj = ggml_add(gc, attn_proj, o_b);

        // Residual + LayerNorm
        x = ggml_add(gc, residual, attn_proj);
        ggml_tensor* ln1_w = W(ts, pfx + ".ln1.weight");
        ggml_tensor* ln1_b = W(ts, pfx + ".ln1.bias");
        if (ln1_w) {
            x = ggml_norm(gc, x, hp.layer_norm_eps);
            x = ggml_mul(gc, x, ln1_w);
            if (ln1_b)
                x = ggml_add(gc, x, ln1_b);
        }

        // FFN: GELU(x @ W_up) @ W_down
        ggml_tensor* ffn_up_w = W(ts, pfx + ".ffn.up.weight");
        ggml_tensor* ffn_up_b = W(ts, pfx + ".ffn.up.bias");
        ggml_tensor* ffn_down_w = W(ts, pfx + ".ffn.down.weight");
        ggml_tensor* ffn_down_b = W(ts, pfx + ".ffn.down.bias");

        ggml_tensor* ffn = ggml_mul_mat(gc, ffn_up_w, x);
        if (ffn_up_b)
            ffn = ggml_add(gc, ffn, ffn_up_b);
        ffn = ggml_gelu(gc, ffn);
        ffn = ggml_mul_mat(gc, ffn_down_w, ffn);
        if (ffn_down_b)
            ffn = ggml_add(gc, ffn, ffn_down_b);

        // Residual + LayerNorm
        x = ggml_add(gc, x, ffn);
        ggml_tensor* ln2_w = W(ts, pfx + ".ln2.weight");
        ggml_tensor* ln2_b = W(ts, pfx + ".ln2.bias");
        if (ln2_w) {
            x = ggml_norm(gc, x, hp.layer_norm_eps);
            x = ggml_mul(gc, x, ln2_w);
            if (ln2_b)
                x = ggml_add(gc, x, ln2_b);
        }
    }

    // Compute encoder output
    ggml_set_name(x, "encoder_out");

    // Set up graph inputs and compute
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 32768, false);
    ggml_build_forward_expand(gf, x);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "speecht5: encoder graph alloc failed\n");
        return {};
    }

    // Set input data
    mg.set_input(inp_ids, token_ids.data(), T * sizeof(int32_t));

    // Set PE data
    auto pe_data = make_sinusoidal_pe(T, hp.hidden_size);
    // Transpose PE from (T, hidden) to (hidden, T) for ggml
    std::vector<float> pe_transposed(T * hp.hidden_size);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < hp.hidden_size; d++) {
            pe_transposed[d + t * hp.hidden_size] = pe_data[t * hp.hidden_size + d];
        }
    }
    mg.set_input(pe_input, pe_transposed.data(), T * hp.hidden_size * sizeof(float));

    // Set relative position IDs
    int max_rel = hp.encoder_max_relative_position;
    std::vector<int32_t> rel_ids(T * T);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            int diff = i - j;
            if (diff < -max_rel)
                diff = -max_rel;
            if (diff >= max_rel)
                diff = max_rel - 1;
            rel_ids[i * T + j] = diff + max_rel;
        }
    }
    // Transpose for ggml (column-major)
    std::vector<int32_t> rel_ids_t(T * T);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            rel_ids_t[j + i * T] = rel_ids[i * T + j];
        }
    }
    // Only set rel_pos_ids if it's actually used in the graph (position bias is TODO)
    if (rel_pos_ids->buffer) {
        mg.set_input(rel_pos_ids, rel_ids_t.data(), T * T * sizeof(int32_t));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Read encoder output
    int n = (int)ggml_nelements(x);
    std::vector<float> result(n);
    ggml_backend_tensor_get(x, result.data(), 0, n * sizeof(float));
    return result;
}

// ── Decoder step (one AR step) ────────────────────────────────────
// Runs decoder prenet + 6 decoder layers + feat_out + prob_out.
// Returns mel frame (reduction_factor * num_mel_bins) and stop prob.

struct decoder_step_result {
    std::vector<float> mel_frame; // (reduction_factor * num_mel_bins)
    float stop_prob = 0.0f;
};

static decoder_step_result run_decoder_step(speecht5_tts_context* ctx,
                                            const std::vector<float>& encoder_out, // (hidden_size * T_enc)
                                            int T_enc,
                                            const std::vector<float>& prev_mel_frame, // (num_mel_bins) - last mel frame
                                            int dec_step // current decoder step (0-indexed)
) {
    const auto& hp = ctx->hp;
    const auto& ts = ctx->tensors();
    decoder_step_result result;
    result.mel_frame.resize(hp.reduction_factor * hp.num_mel_bins, 0.0f);

    mini_graph mg(ctx->sched);
    auto* gc = mg.ctx;

    // Inputs
    ggml_tensor* enc_hidden = ggml_new_tensor_2d(gc, GGML_TYPE_F32, hp.hidden_size, T_enc);
    ggml_set_name(enc_hidden, "enc_hidden");
    ggml_set_input(enc_hidden);

    ggml_tensor* mel_input = ggml_new_tensor_1d(gc, GGML_TYPE_F32, hp.num_mel_bins);
    ggml_set_name(mel_input, "mel_input");
    ggml_set_input(mel_input);

    ggml_tensor* spk_input = ggml_new_tensor_1d(gc, GGML_TYPE_F32, hp.speaker_dim);
    ggml_set_name(spk_input, "spk_input");
    ggml_set_input(spk_input);

    ggml_tensor* dec_pe = ggml_new_tensor_1d(gc, GGML_TYPE_F32, hp.hidden_size);
    ggml_set_name(dec_pe, "dec_pe");
    ggml_set_input(dec_pe);

    // ── Prenet ──
    // 2x Linear + ReLU (no dropout at inference)
    ggml_tensor* x = ggml_reshape_2d(gc, mel_input, hp.num_mel_bins, 1); // (mel, 1)
    for (int j = 0; j < hp.prenet_layers; j++) {
        std::string pn = "dec.prenet." + std::to_string(j);
        ggml_tensor* pw = W(ts, pn + ".weight");
        ggml_tensor* pb = W(ts, pn + ".bias");
        if (pw) {
            x = ggml_mul_mat(gc, pw, x);
            if (pb)
                x = ggml_add(gc, x, pb);
            x = ggml_relu(gc, x);
        }
    }

    // Final layer: Linear(prenet_units -> hidden_size)
    ggml_tensor* final_w = W(ts, "dec.prenet.final.weight");
    ggml_tensor* final_b = W(ts, "dec.prenet.final.bias");
    if (final_w) {
        x = ggml_mul_mat(gc, final_w, x);
        if (final_b)
            x = ggml_add(gc, x, final_b);
    }

    // Add scaled positional encoding for current step
    ggml_tensor* dec_alpha = W(ts, "dec.pos_alpha");
    ggml_tensor* pe_1d = ggml_reshape_2d(gc, dec_pe, hp.hidden_size, 1);
    if (dec_alpha) {
        float alpha_val = 1.0f;
        ggml_backend_tensor_get(dec_alpha, &alpha_val, 0, sizeof(float));
        pe_1d = ggml_scale(gc, pe_1d, alpha_val);
    }
    x = ggml_add(gc, x, pe_1d);

    // Speaker embedding: normalize, concat, project
    ggml_tensor* spk_2d = ggml_reshape_2d(gc, spk_input, hp.speaker_dim, 1);
    // L2 normalize speaker embedding
    // Speaker embedding is pre-normalized on the host side (L2 norm)
    // before being passed as input, matching the HF implementation.

    // Concatenate: (hidden_size + speaker_dim, 1)
    ggml_tensor* cat = ggml_concat(gc, x, spk_2d, 0); // concat on dim 0

    // Speaker projection: Linear(hidden+spk, hidden) + ReLU
    ggml_tensor* spk_w = W(ts, "dec.spk_proj.weight");
    ggml_tensor* spk_b = W(ts, "dec.spk_proj.bias");
    if (spk_w) {
        x = ggml_mul_mat(gc, spk_w, cat);
        if (spk_b)
            x = ggml_add(gc, x, spk_b);
        x = ggml_relu(gc, x);
    }

    // x is now (hidden_size, 1) -- the decoder input for this step

    // ── Decoder layers with KV-cached self-attention ──
    // At step N, self-attention attends to all N+1 positions (including current).
    // We pass the full KV cache as input tensors and append current K/V after compute.
    const int T_kv = dec_step + 1; // total positions including current step

    // Create input tensors for past KV (one per layer)
    // These hold the full K/V including the current step (we'll compute current K/V
    // in the graph and concatenate with past via an input tensor).
    // Actually, simpler approach: pass past K/V as inputs, compute current K/V in graph,
    // concatenate them, then do attention. But ggml_concat requires both tensors in the graph.
    // Simplest correct approach: pass ALL K/V (past + current) as pre-computed input tensors.
    // We compute current K/V on the host after graph execution of a "projection-only" pass,
    // OR we do the full attention with all KV as inputs.
    //
    // Strategy: We'll compute Q/K/V projections inside the graph for the current step,
    // but for past K/V we pass them as input tensors. Then we concatenate.
    // However this requires dynamic graph sizes. Instead, use the simplest approach:
    // 1. Each layer's self-attn projects current x -> Q, K_cur, V_cur (in graph)
    // 2. Past K/V are passed as 2D input tensors (hidden_size, dec_step)
    // 3. Full K = concat(past_K, K_cur), Full V = concat(past_V, V_cur)
    // 4. Attend Q (1 position) to Full K/V (T_kv positions)
    // 5. After compute, read K_cur and V_cur from the graph to append to the cache.
    //
    // But reading intermediate tensors after compute is tricky with mini_graph.
    // Even simpler: compute K_cur and V_cur on the HOST (just matrix multiply),
    // append to cache BEFORE building the graph, then pass the full cache as input.
    // The K/V projection is just: K = W_k @ x + b_k, V = W_v @ x + b_v
    // We can do this with a tiny sub-graph or manually.
    //
    // Actually the cleanest approach for correctness:
    // Pre-compute K and V for each layer on the host before building the main graph.
    // We need the hidden state 'x' at each layer's self-attention input, but that depends
    // on the previous layer's output. So we can't pre-compute all layers' K/V at once.
    //
    // Final approach: build the graph with past KV as inputs AND current K/V computed
    // in-graph, concatenated. After compute, we read back the current K/V values
    // from named output tensors.

    struct layer_kv_tensors {
        ggml_tensor* past_k_input; // (hidden_size, dec_step) or null if step 0
        ggml_tensor* past_v_input;
        ggml_tensor* cur_k_out; // (hidden_size, 1) — named for readback
        ggml_tensor* cur_v_out;
    };
    std::vector<layer_kv_tensors> layer_kv(hp.decoder_layers);

    for (int i = 0; i < hp.decoder_layers; i++) {
        std::string pfx = "dec.layer." + std::to_string(i);

        // Self-attention with KV cache
        ggml_tensor* residual = x;
        ggml_tensor* sq_w = W(ts, pfx + ".self_attn.q.weight");
        ggml_tensor* sq_b = W(ts, pfx + ".self_attn.q.bias");
        ggml_tensor* sk_w = W(ts, pfx + ".self_attn.k.weight");
        ggml_tensor* sk_b = W(ts, pfx + ".self_attn.k.bias");
        ggml_tensor* sv_w = W(ts, pfx + ".self_attn.v.weight");
        ggml_tensor* sv_b = W(ts, pfx + ".self_attn.v.bias");
        ggml_tensor* so_w = W(ts, pfx + ".self_attn.o.weight");
        ggml_tensor* so_b = W(ts, pfx + ".self_attn.o.bias");

        // Project current position -> Q, K_cur, V_cur
        ggml_tensor* sq = ggml_mul_mat(gc, sq_w, x); // (hidden_size, 1)
        if (sq_b)
            sq = ggml_add(gc, sq, sq_b);
        sq = ggml_scale(gc, sq, 1.0f / sqrtf((float)hp.head_dim()));

        ggml_tensor* sk_cur = ggml_mul_mat(gc, sk_w, x); // (hidden_size, 1)
        if (sk_b)
            sk_cur = ggml_add(gc, sk_cur, sk_b);

        ggml_tensor* sv_cur = ggml_mul_mat(gc, sv_w, x); // (hidden_size, 1)
        if (sv_b)
            sv_cur = ggml_add(gc, sv_cur, sv_b);

        // Create copies for readback (originals may have buffers reused by later ops)
        ggml_tensor* sk_cur_copy = ggml_dup(gc, sk_cur);
        ggml_tensor* sv_cur_copy = ggml_dup(gc, sv_cur);
        std::string k_name = "sa_k_" + std::to_string(i);
        std::string v_name = "sa_v_" + std::to_string(i);
        ggml_set_name(sk_cur_copy, k_name.c_str());
        ggml_set_name(sv_cur_copy, v_name.c_str());

        // Build full K and V by concatenating past cache with current
        ggml_tensor* full_k;
        ggml_tensor* full_v;

        if (dec_step > 0) {
            // Past K/V input tensors: (hidden_size, dec_step)
            ggml_tensor* past_k = ggml_new_tensor_2d(gc, GGML_TYPE_F32, hp.hidden_size, dec_step);
            std::string pk_name = "past_k_" + std::to_string(i);
            ggml_set_name(past_k, pk_name.c_str());
            ggml_set_input(past_k);

            ggml_tensor* past_v = ggml_new_tensor_2d(gc, GGML_TYPE_F32, hp.hidden_size, dec_step);
            std::string pv_name = "past_v_" + std::to_string(i);
            ggml_set_name(past_v, pv_name.c_str());
            ggml_set_input(past_v);

            layer_kv[i].past_k_input = past_k;
            layer_kv[i].past_v_input = past_v;

            // Concat: past (hidden_size, dec_step) + cur (hidden_size, 1) -> (hidden_size, T_kv)
            full_k = ggml_concat(gc, past_k, sk_cur, 1); // concat along dim 1 (sequence)
            full_v = ggml_concat(gc, past_v, sv_cur, 1);
        } else {
            // First step: K/V is just current
            full_k = sk_cur;
            full_v = sv_cur;
            layer_kv[i].past_k_input = nullptr;
            layer_kv[i].past_v_input = nullptr;
        }

        layer_kv[i].cur_k_out = sk_cur_copy;
        layer_kv[i].cur_v_out = sv_cur_copy;

        // Multi-head attention: Q (1 pos) attending to full K/V (T_kv pos)
        int n_heads = hp.decoder_attention_heads;
        int hd = hp.head_dim();

        // Q: (hidden_size, 1) -> (hd, n_heads, 1)
        ggml_tensor* q_mh = ggml_reshape_3d(gc, sq, hd, n_heads, 1);
        // K: (hidden_size, T_kv) -> (hd, n_heads, T_kv)
        ggml_tensor* k_mh = ggml_reshape_3d(gc, full_k, hd, n_heads, T_kv);
        // V: (hidden_size, T_kv) -> (hd, n_heads, T_kv)
        ggml_tensor* v_mh = ggml_reshape_3d(gc, full_v, hd, n_heads, T_kv);

        // Permute for batched matmul: (hd, seq, n_heads)
        q_mh = ggml_cont(gc, ggml_permute(gc, q_mh, 0, 2, 1, 3)); // (hd, 1, n_heads)
        k_mh = ggml_cont(gc, ggml_permute(gc, k_mh, 0, 2, 1, 3)); // (hd, T_kv, n_heads)
        v_mh = ggml_cont(gc, ggml_permute(gc, v_mh, 0, 2, 1, 3)); // (hd, T_kv, n_heads)

        // attn_w = Q @ K^T: mul_mat(K, Q) -> (T_kv, 1, n_heads)
        ggml_tensor* attn_w = ggml_mul_mat(gc, k_mh, q_mh); // (T_kv, 1, n_heads)
        attn_w = ggml_soft_max(gc, attn_w);

        // Output: attn_w @ V
        // V transposed: (T_kv, hd, n_heads)
        ggml_tensor* v_t = ggml_cont(gc, ggml_permute(gc, v_mh, 1, 0, 2, 3)); // (T_kv, hd, n_heads)
        ggml_tensor* self_out = ggml_mul_mat(gc, v_t, attn_w);                // (hd, 1, n_heads)

        // Reshape to (hidden_size, 1)
        self_out = ggml_reshape_2d(gc, ggml_cont(gc, self_out), hp.hidden_size, 1);

        // Output projection
        self_out = ggml_mul_mat(gc, so_w, self_out);
        if (so_b)
            self_out = ggml_add(gc, self_out, so_b);

        // Residual + LN
        x = ggml_add(gc, residual, self_out);
        ggml_tensor* ln_self_w = W(ts, pfx + ".ln_self.weight");
        ggml_tensor* ln_self_b = W(ts, pfx + ".ln_self.bias");
        if (ln_self_w) {
            x = ggml_norm(gc, x, hp.layer_norm_eps);
            x = ggml_mul(gc, x, ln_self_w);
            if (ln_self_b)
                x = ggml_add(gc, x, ln_self_b);
        }

        // Cross-attention: Q from decoder, K/V from encoder
        residual = x;
        ggml_tensor* cq_w = W(ts, pfx + ".cross_attn.q.weight");
        ggml_tensor* cq_b = W(ts, pfx + ".cross_attn.q.bias");
        ggml_tensor* ck_w = W(ts, pfx + ".cross_attn.k.weight");
        ggml_tensor* ck_b = W(ts, pfx + ".cross_attn.k.bias");
        ggml_tensor* cv_w = W(ts, pfx + ".cross_attn.v.weight");
        ggml_tensor* cv_b = W(ts, pfx + ".cross_attn.v.bias");
        ggml_tensor* co_w = W(ts, pfx + ".cross_attn.o.weight");
        ggml_tensor* co_b = W(ts, pfx + ".cross_attn.o.bias");

        // Q: (hidden_size, 1) projected
        ggml_tensor* cq = ggml_mul_mat(gc, cq_w, x);
        if (cq_b)
            cq = ggml_add(gc, cq, cq_b);
        cq = ggml_scale(gc, cq, 1.0f / sqrtf((float)hp.head_dim()));

        // K, V: from encoder hidden (hidden_size, T_enc)
        ggml_tensor* ck = ggml_mul_mat(gc, ck_w, enc_hidden);
        if (ck_b)
            ck = ggml_add(gc, ck, ck_b);

        ggml_tensor* cv = ggml_mul_mat(gc, cv_w, enc_hidden);
        if (cv_b)
            cv = ggml_add(gc, cv, cv_b);

        // Multi-head cross-attention
        // (n_heads and hd already declared above in self-attention)

        // Q: (hidden_size, 1) -> (hd, n_heads, 1)
        ggml_tensor* cq_mh = ggml_reshape_3d(gc, cq, hd, n_heads, 1);
        // K: (hidden_size, T_enc) -> (hd, n_heads, T_enc)
        ggml_tensor* ck_mh = ggml_reshape_3d(gc, ck, hd, n_heads, T_enc);
        // V: (hidden_size, T_enc) -> (hd, n_heads, T_enc)
        ggml_tensor* cv_mh = ggml_reshape_3d(gc, cv, hd, n_heads, T_enc);

        // For each head: attn_w = Q @ K^T -> (1, T_enc)
        // ggml: mul_mat(K, Q) with K as (hd, T_enc, n_heads), Q as (hd, 1, n_heads)
        // But we need to permute for the batched matmul.
        cq_mh = ggml_cont(gc, ggml_permute(gc, cq_mh, 0, 2, 1, 3)); // (hd, 1, n_heads)
        ck_mh = ggml_cont(gc, ggml_permute(gc, ck_mh, 0, 2, 1, 3)); // (hd, T_enc, n_heads)
        cv_mh = ggml_cont(gc, ggml_permute(gc, cv_mh, 0, 2, 1, 3)); // (hd, T_enc, n_heads)

        ggml_tensor* cross_attn_w = ggml_mul_mat(gc, ck_mh, cq_mh); // (T_enc, 1, n_heads)
        cross_attn_w = ggml_soft_max(gc, cross_attn_w);

        // Output: cross_attn_w @ V
        // cross_attn_w: (T_enc, 1, n_heads), cv_mh: (hd, T_enc, n_heads)
        // Transpose V to (T_enc, hd, n_heads), then mul_mat contracts on T_enc
        ggml_tensor* cv_t = ggml_cont(gc, ggml_permute(gc, cv_mh, 1, 0, 2, 3)); // (T_enc, hd, n_heads)
        ggml_tensor* cross_out = ggml_mul_mat(gc, cv_t, cross_attn_w);          // (hd, 1, n_heads)

        // Permute (hd, 1, n_heads) -> (hd, n_heads, 1) then reshape to (hidden, 1)
        cross_out = ggml_cont(gc, ggml_permute(gc, cross_out, 0, 2, 1, 3));
        cross_out = ggml_reshape_2d(gc, cross_out, hp.hidden_size, 1);

        // Output projection
        cross_out = ggml_mul_mat(gc, co_w, cross_out);
        if (co_b)
            cross_out = ggml_add(gc, cross_out, co_b);

        // Residual + LN
        x = ggml_add(gc, residual, cross_out);
        ggml_tensor* ln_cross_w = W(ts, pfx + ".ln_cross.weight");
        ggml_tensor* ln_cross_b = W(ts, pfx + ".ln_cross.bias");
        if (ln_cross_w) {
            x = ggml_norm(gc, x, hp.layer_norm_eps);
            x = ggml_mul(gc, x, ln_cross_w);
            if (ln_cross_b)
                x = ggml_add(gc, x, ln_cross_b);
        }

        // FFN
        ggml_tensor* ffn_up_w = W(ts, pfx + ".ffn.up.weight");
        ggml_tensor* ffn_up_b = W(ts, pfx + ".ffn.up.bias");
        ggml_tensor* ffn_down_w = W(ts, pfx + ".ffn.down.weight");
        ggml_tensor* ffn_down_b = W(ts, pfx + ".ffn.down.bias");

        ggml_tensor* ffn = ggml_mul_mat(gc, ffn_up_w, x);
        if (ffn_up_b)
            ffn = ggml_add(gc, ffn, ffn_up_b);
        ffn = ggml_gelu(gc, ffn);
        ffn = ggml_mul_mat(gc, ffn_down_w, ffn);
        if (ffn_down_b)
            ffn = ggml_add(gc, ffn, ffn_down_b);

        // Residual + LN
        x = ggml_add(gc, x, ffn);
        ggml_tensor* ln_final_w = W(ts, pfx + ".ln_final.weight");
        ggml_tensor* ln_final_b = W(ts, pfx + ".ln_final.bias");
        if (ln_final_w) {
            x = ggml_norm(gc, x, hp.layer_norm_eps);
            x = ggml_mul(gc, x, ln_final_w);
            if (ln_final_b)
                x = ggml_add(gc, x, ln_final_b);
        }
    }

    // dec_hidden = x — used by feat_out/prob_out below

    // ── Output heads ──
    // feat_out: Linear(hidden_size -> reduction_factor * num_mel_bins)
    ggml_tensor* feat_w = W(ts, "dec.postnet.feat_out.weight");
    ggml_tensor* feat_b = W(ts, "dec.postnet.feat_out.bias");
    ggml_tensor* mel_out = ggml_mul_mat(gc, feat_w, x);
    if (feat_b)
        mel_out = ggml_add(gc, mel_out, feat_b);

    // prob_out: Linear(hidden_size -> reduction_factor)
    ggml_tensor* prob_w = W(ts, "dec.postnet.prob_out.weight");
    ggml_tensor* prob_b = W(ts, "dec.postnet.prob_out.bias");
    ggml_tensor* prob_out = ggml_mul_mat(gc, prob_w, x);
    if (prob_b)
        prob_out = ggml_add(gc, prob_out, prob_b);

    ggml_set_name(mel_out, "mel_out");
    ggml_set_name(prob_out, "prob_out");

    // Compute
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 32768, false);
    ggml_build_forward_expand(gf, mel_out);
    ggml_build_forward_expand(gf, prob_out);
    // Also expand K/V output copy tensors so they get computed and preserved
    for (int i = 0; i < hp.decoder_layers; i++) {
        ggml_build_forward_expand(gf, layer_kv[i].cur_k_out);
        ggml_build_forward_expand(gf, layer_kv[i].cur_v_out);
    }
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "speecht5: decoder graph alloc failed\n");
        return result;
    }

    // Set inputs
    mg.set_input(enc_hidden, encoder_out.data(), T_enc * hp.hidden_size * sizeof(float));
    mg.set_input(mel_input, prev_mel_frame.data(), hp.num_mel_bins * sizeof(float));

    // Normalized speaker embedding
    std::vector<float> spk_norm_data(hp.speaker_dim, 0.0f);
    if (!ctx->speaker_emb.empty()) {
        float l2 = 0.0f;
        for (int d = 0; d < hp.speaker_dim; d++) {
            l2 += ctx->speaker_emb[d] * ctx->speaker_emb[d];
        }
        l2 = sqrtf(l2 + 1e-12f);
        for (int d = 0; d < hp.speaker_dim; d++) {
            spk_norm_data[d] = ctx->speaker_emb[d] / l2;
        }
    }
    mg.set_input(spk_input, spk_norm_data.data(), hp.speaker_dim * sizeof(float));

    // Decoder PE for current step
    auto dec_pe_full = make_sinusoidal_pe(dec_step + 2, hp.hidden_size);
    // Use step dec_step (0-indexed)
    std::vector<float> dec_pe_step(hp.hidden_size);
    for (int d = 0; d < hp.hidden_size; d++) {
        dec_pe_step[d] = dec_pe_full[dec_step * hp.hidden_size + d];
    }
    mg.set_input(dec_pe, dec_pe_step.data(), hp.hidden_size * sizeof(float));

    // Set past KV cache inputs
    for (int i = 0; i < hp.decoder_layers; i++) {
        if (dec_step > 0 && layer_kv[i].past_k_input) {
            mg.set_input(layer_kv[i].past_k_input, ctx->self_kv_cache.k[i].data(),
                         dec_step * hp.hidden_size * sizeof(float));
            mg.set_input(layer_kv[i].past_v_input, ctx->self_kv_cache.v[i].data(),
                         dec_step * hp.hidden_size * sizeof(float));
        }
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // ── SPEECHT5_DUMP_DIR: per-step intermediate dumps ──
    {
        static const char* dump_dir = getenv("SPEECHT5_DUMP_DIR");
        if (dump_dir) {
            auto dump_f32 = [&](const char* tag, const float* data, size_t n) {
                std::string path = std::string(dump_dir) + "/step" + std::to_string(dec_step) + "_" + tag + ".f32";
                FILE* f = fopen(path.c_str(), "wb");
                if (f) {
                    fwrite(data, sizeof(float), n, f);
                    fclose(f);
                }
            };
            // Dump mel_out
            {
                int mel_n = hp.reduction_factor * hp.num_mel_bins;
                std::vector<float> buf(mel_n);
                ggml_backend_tensor_get(mel_out, buf.data(), 0, mel_n * sizeof(float));
                dump_f32("mel", buf.data(), mel_n);
            }
            // Dump self-attn K/V for first few steps (layer 0 only)
            if (dec_step <= 2) {
                std::vector<float> k0(hp.hidden_size);
                ggml_backend_tensor_get(layer_kv[0].cur_k_out, k0.data(), 0, hp.hidden_size * sizeof(float));
                dump_f32("self_k_L0", k0.data(), hp.hidden_size);
            }
            if (dec_step == 0) {
                fprintf(stderr, "speecht5: SPEECHT5_DUMP_DIR active, dumping to %s\n", dump_dir);
            }
        }
    }

    // Read back current K/V and append to cache
    for (int i = 0; i < hp.decoder_layers; i++) {
        std::vector<float> k_cur(hp.hidden_size);
        std::vector<float> v_cur(hp.hidden_size);
        ggml_backend_tensor_get(layer_kv[i].cur_k_out, k_cur.data(), 0, hp.hidden_size * sizeof(float));
        ggml_backend_tensor_get(layer_kv[i].cur_v_out, v_cur.data(), 0, hp.hidden_size * sizeof(float));
        ctx->self_kv_cache.append(i, k_cur, v_cur);
    }
    ctx->self_kv_cache.n_steps++;

    // Read outputs
    int mel_n = hp.reduction_factor * hp.num_mel_bins;
    ggml_backend_tensor_get(mel_out, result.mel_frame.data(), 0, mel_n * sizeof(float));

    std::vector<float> prob_data(hp.reduction_factor);
    ggml_backend_tensor_get(prob_out, prob_data.data(), 0, hp.reduction_factor * sizeof(float));

    // Sigmoid and sum for stop probability
    float prob_sum = 0.0f;
    for (int r = 0; r < hp.reduction_factor; r++) {
        prob_sum += 1.0f / (1.0f + expf(-prob_data[r]));
    }
    result.stop_prob = prob_sum;

    return result;
}

// ── Post-net (5-layer Conv1d + BN + Tanh) ─────────────────────────

static std::vector<float> run_postnet(speecht5_tts_context* ctx,
                                      const std::vector<float>& mel_spectrogram, // (T_mel * num_mel_bins), row-major
                                      int T_mel) {
    const auto& hp = ctx->hp;
    const auto& ts = ctx->tensors();

    mini_graph mg(ctx->sched);
    auto* gc = mg.ctx;

    // Input: (T_mel, num_mel_bins) -- ggml conv_1d expects (T, C_in)
    ggml_tensor* x = ggml_new_tensor_2d(gc, GGML_TYPE_F32, T_mel, hp.num_mel_bins);
    ggml_set_name(x, "postnet_in");
    ggml_set_input(x);

    // No transpose needed: ggml conv_1d takes (T, C_in) directly
    ggml_tensor* h = x;

    // Store deferred BN inputs (set after graph alloc)
    struct bn_deferred {
        ggml_tensor* scale_t;
        ggml_tensor* shift_t;
        std::vector<float> scale_data;
        std::vector<float> shift_data;
    };
    std::vector<bn_deferred> bn_inputs;

    // 5-layer conv1d + batch_norm + tanh
    for (int i = 0; i < hp.postnet_layers; i++) {
        std::string pfx = "dec.postnet.conv." + std::to_string(i);
        ggml_tensor* conv_w = W(ts, pfx + ".weight");
        ggml_tensor* bn_w = W(ts, pfx + ".bn.weight");
        ggml_tensor* bn_b = W(ts, pfx + ".bn.bias");
        ggml_tensor* bn_mean = W(ts, pfx + ".bn.mean");
        ggml_tensor* bn_var = W(ts, pfx + ".bn.var");

        if (!conv_w)
            continue;

        // Conv1d with padding = (kernel-1)/2
        int pad = (hp.postnet_kernel - 1) / 2;
        h = ggml_conv_1d(gc, conv_w, h, 1, pad, 1);

        // Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        // Pre-compute inv_std = weight / sqrt(var + eps) and bias_adj = bias - mean * inv_std
        // on the host to avoid ggml_new_f32 in no_alloc context.
        if (bn_mean && bn_var && bn_w && bn_b) {
            int C = (int)bn_mean->ne[0];
            // Read BN params from weight tensors
            std::vector<float> mean_v(C), var_v(C), weight_v(C), bias_v(C);
            ggml_backend_tensor_get(bn_mean, mean_v.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(bn_var, var_v.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(bn_w, weight_v.data(), 0, C * sizeof(float));
            ggml_backend_tensor_get(bn_b, bias_v.data(), 0, C * sizeof(float));

            // Compute fused BN params: scale = w / sqrt(v + eps), shift = b - mean * scale
            std::vector<float> scale_v(C), shift_v(C);
            for (int c = 0; c < C; c++) {
                float inv_std = 1.0f / sqrtf(var_v[c] + hp.layer_norm_eps);
                scale_v[c] = weight_v[c] * inv_std;
                shift_v[c] = bias_v[c] - mean_v[c] * scale_v[c];
            }

            // Create input tensors for fused BN
            ggml_tensor* bn_scale_t = ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, C);
            ggml_set_name(bn_scale_t, (std::string("bn_scale_") + std::to_string(i)).c_str());
            ggml_set_input(bn_scale_t);
            ggml_tensor* bn_shift_t = ggml_new_tensor_2d(gc, GGML_TYPE_F32, 1, C);
            ggml_set_name(bn_shift_t, (std::string("bn_shift_") + std::to_string(i)).c_str());
            ggml_set_input(bn_shift_t);

            // h = h * scale + shift (affine transform = fused batch norm)
            h = ggml_mul(gc, h, bn_scale_t);
            h = ggml_add(gc, h, bn_shift_t);

            // Defer data setting until after graph alloc
            bn_inputs.push_back({bn_scale_t, bn_shift_t, std::move(scale_v), std::move(shift_v)});
        }

        // Tanh on all layers except the last
        if (i < hp.postnet_layers - 1) {
            h = ggml_tanh(gc, h);
        }
    }

    // Add residual (postnet is residual) — h and x are both (T_mel, num_mel_bins)
    h = ggml_add(gc, h, x);

    ggml_set_name(h, "postnet_out");

    ggml_cgraph* gf = ggml_new_graph_custom(gc, 32768, false);
    ggml_build_forward_expand(gf, h);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "speecht5: postnet graph alloc failed\n");
        return {};
    }

    // Set input: mel spectrogram -- transpose from row-major (T_mel, num_mel_bins)
    // to ggml column-major layout where ne[0]=T_mel is fastest
    {
        std::vector<float> transposed(T_mel * hp.num_mel_bins);
        for (int t = 0; t < T_mel; t++) {
            for (int c = 0; c < hp.num_mel_bins; c++) {
                transposed[t + c * T_mel] = mel_spectrogram[t * hp.num_mel_bins + c];
            }
        }
        mg.set_input(x, transposed.data(), T_mel * hp.num_mel_bins * sizeof(float));
    }

    // Set deferred BN inputs
    for (auto& bn : bn_inputs) {
        mg.set_input(bn.scale_t, bn.scale_data.data(), bn.scale_data.size() * sizeof(float));
        mg.set_input(bn.shift_t, bn.shift_data.data(), bn.shift_data.size() * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    // Read output and transpose back from ggml column-major to row-major (T_mel, num_mel_bins)
    int n = (int)ggml_nelements(h);
    std::vector<float> raw(n);
    ggml_backend_tensor_get(h, raw.data(), 0, n * sizeof(float));
    std::vector<float> result(n);
    for (int t = 0; t < T_mel; t++) {
        for (int c = 0; c < hp.num_mel_bins; c++) {
            result[t * hp.num_mel_bins + c] = raw[t + c * T_mel];
        }
    }
    return result;
}

// ── HiFi-GAN vocoder ──────────────────────────────────────────────

static std::vector<float> run_vocoder(speecht5_tts_context* ctx,
                                      const std::vector<float>& mel_spectrogram, // (T_mel * num_mel_bins), row-major
                                      int T_mel) {
    const auto& ts = ctx->tensors();
    const auto& vhp = ctx->voc_hp;

    mini_graph mg(ctx->sched, 64 * 1024 * 1024);
    auto* gc = mg.ctx;

    // Input mel: (T_mel, num_mel_bins) — ggml conv_1d expects (T, C_in) directly
    ggml_tensor* mel_in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, T_mel, vhp.model_in_dim);
    ggml_set_name(mel_in, "voc_mel");
    ggml_set_input(mel_in);

    // Run HiFi-GAN — input is (T, C_in) = (T_mel, mel_dim)
    ggml_tensor* waveform = core_hifigan::forward(gc, mel_in, ts, "voc", vhp, ctx->ups_w_perm);

    ggml_set_name(waveform, "waveform");

    // Build and compute
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 65536, false);
    ggml_build_forward_expand(gf, waveform);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "speecht5: vocoder graph alloc failed\n");
        return {};
    }

    // Transpose mel from row-major (T_mel, mel_dim) to ggml column-major (ne[0]=T fastest)
    {
        int C = vhp.model_in_dim;
        std::vector<float> transposed(T_mel * C);
        for (int t = 0; t < T_mel; t++) {
            for (int c = 0; c < C; c++) {
                transposed[t + c * T_mel] = mel_spectrogram[t * C + c];
            }
        }
        mg.set_input(mel_in, transposed.data(), T_mel * C * sizeof(float));
    }

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    int n = (int)ggml_nelements(waveform);
    std::vector<float> result(n);
    ggml_backend_tensor_get(waveform, result.data(), 0, n * sizeof(float));
    return result;
}

// ── Public API ─────────────────────────────────────────────────────

struct speecht5_tts_params speecht5_tts_default_params(void) {
    struct speecht5_tts_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.threshold = 0.5f;
    p.max_len = 0;
    p.seed = 0;
    return p;
}

struct speecht5_tts_context* speecht5_tts_init(const char* path, struct speecht5_tts_params params) {
    auto* ctx = new speecht5_tts_context();
    ctx->n_threads = params.n_threads;
    ctx->verbosity = params.verbosity;
    ctx->threshold = params.threshold;
    ctx->max_len = params.max_len;

    // Backend — prefer GPU (CUDA/Metal/Vulkan) when available + requested
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : nullptr;
    if (!ctx->backend) {
        ctx->backend = ggml_backend_cpu_init();
    }
    if (!ctx->backend) {
        fprintf(stderr, "speecht5: failed to init any backend\n");
        delete ctx;
        return nullptr;
    }
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);
    }
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, params.n_threads);
    }

    // Create backend scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
            backends[n_be++] = ctx->backend_cpu;
        }
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 65536, false, false);
        if (!ctx->sched) {
            fprintf(stderr, "speecht5: failed to create backend scheduler\n");
            delete ctx;
            return nullptr;
        }
    }

    // Load GGUF metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        fprintf(stderr, "speecht5: failed to open GGUF '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.hidden_size = (int)core_gguf::kv_u32(meta, "speecht5.hidden_size", hp.hidden_size);
    hp.num_mel_bins = (int)core_gguf::kv_u32(meta, "speecht5.num_mel_bins", hp.num_mel_bins);
    hp.encoder_layers = (int)core_gguf::kv_u32(meta, "speecht5.encoder_layers", hp.encoder_layers);
    hp.decoder_layers = (int)core_gguf::kv_u32(meta, "speecht5.decoder_layers", hp.decoder_layers);
    hp.encoder_attention_heads =
        (int)core_gguf::kv_u32(meta, "speecht5.encoder_attention_heads", hp.encoder_attention_heads);
    hp.decoder_attention_heads =
        (int)core_gguf::kv_u32(meta, "speecht5.decoder_attention_heads", hp.decoder_attention_heads);
    hp.encoder_ffn_dim = (int)core_gguf::kv_u32(meta, "speecht5.encoder_ffn_dim", hp.encoder_ffn_dim);
    hp.decoder_ffn_dim = (int)core_gguf::kv_u32(meta, "speecht5.decoder_ffn_dim", hp.decoder_ffn_dim);
    hp.vocab_size = (int)core_gguf::kv_u32(meta, "speecht5.vocab_size", hp.vocab_size);
    hp.reduction_factor = (int)core_gguf::kv_u32(meta, "speecht5.reduction_factor", hp.reduction_factor);
    hp.prenet_layers = (int)core_gguf::kv_u32(meta, "speecht5.speech_decoder_prenet_layers", hp.prenet_layers);
    hp.prenet_units = (int)core_gguf::kv_u32(meta, "speecht5.speech_decoder_prenet_units", hp.prenet_units);
    hp.postnet_layers = (int)core_gguf::kv_u32(meta, "speecht5.speech_decoder_postnet_layers", hp.postnet_layers);
    hp.postnet_units = (int)core_gguf::kv_u32(meta, "speecht5.speech_decoder_postnet_units", hp.postnet_units);
    hp.postnet_kernel = (int)core_gguf::kv_u32(meta, "speecht5.speech_decoder_postnet_kernel", hp.postnet_kernel);
    hp.speaker_dim = (int)core_gguf::kv_u32(meta, "speecht5.speaker_embedding_dim", hp.speaker_dim);
    hp.max_text_positions = (int)core_gguf::kv_u32(meta, "speecht5.max_text_positions", hp.max_text_positions);
    hp.max_speech_positions = (int)core_gguf::kv_u32(meta, "speecht5.max_speech_positions", hp.max_speech_positions);
    hp.encoder_max_relative_position =
        (int)core_gguf::kv_u32(meta, "speecht5.encoder_max_relative_position", hp.encoder_max_relative_position);
    hp.layer_norm_eps = core_gguf::kv_f32(meta, "speecht5.layer_norm_eps", hp.layer_norm_eps);

    // Vocoder hyperparams
    auto& vhp = ctx->voc_hp;
    vhp.model_in_dim = (int)core_gguf::kv_u32(meta, "speecht5.vocoder.model_in_dim", vhp.model_in_dim);
    vhp.upsample_initial_ch =
        (int)core_gguf::kv_u32(meta, "speecht5.vocoder.upsample_initial_channel", vhp.upsample_initial_ch);
    vhp.leaky_relu_slope = core_gguf::kv_f32(meta, "speecht5.vocoder.leaky_relu_slope", vhp.leaky_relu_slope);
    vhp.normalize_before = core_gguf::kv_bool(meta, "speecht5.vocoder.normalize_before", vhp.normalize_before);

    // Read arrays from GGUF for vocoder
    // For now, use defaults since GGUF array reading is more complex
    vhp.upsample_rates = {4, 4, 4, 4};
    vhp.upsample_kernel_sizes = {8, 8, 8, 8};
    vhp.resblock_kernel_sizes = {3, 7, 11};
    vhp.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};

    // Load vocab
    auto vocab_arr = core_gguf::kv_str_array(meta, "speecht5.vocab");
    if (!vocab_arr.empty()) {
        ctx->tokenizer.load(vocab_arr);
        if (params.verbosity > 0) {
            fprintf(stderr, "speecht5: loaded vocab with %d tokens\n", (int)vocab_arr.size());
        }
    }

    core_gguf::free_metadata(meta);

    // Load weights
    if (!core_gguf::load_weights(path, ctx->backend, "speecht5", ctx->wl)) {
        fprintf(stderr, "speecht5: failed to load weights from '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    // Permute HiFi-GAN upsample ConvTranspose1d weights
    {
        const int n = ctx->voc_hp.num_upsamples();
        std::vector<ggml_tensor*> srcs(n);
        std::vector<ggml_tensor**> dsts(n);
        ctx->ups_w_perm.resize(n, nullptr);
        auto& ts = ctx->tensors();
        for (int i = 0; i < n; i++) {
            std::string wname = "voc.ups." + std::to_string(i) + ".weight";
            auto it = ts.find(wname);
            srcs[i] = (it != ts.end()) ? it->second : nullptr;
            dsts[i] = &ctx->ups_w_perm[i];
        }
        core_convt::permute_convt1d_weights_batch(srcs.data(), dsts.data(), n, ctx->backend, &ctx->ctx_perm,
                                                  &ctx->buf_perm);
    }

    if (params.verbosity > 0) {
        fprintf(stderr, "speecht5: backend=%s\n", ggml_backend_name(ctx->backend));
        fprintf(stderr, "speecht5: loaded model — hidden=%d mel=%d enc=%d dec=%d vocab=%d\n", hp.hidden_size,
                hp.num_mel_bins, hp.encoder_layers, hp.decoder_layers, hp.vocab_size);
        fprintf(stderr, "speecht5: vocoder — init_ch=%d rates=[%d,%d,%d,%d] kernels=[%d,%d,%d]\n",
                vhp.upsample_initial_ch, vhp.upsample_rates[0], vhp.upsample_rates[1], vhp.upsample_rates[2],
                vhp.upsample_rates[3], vhp.resblock_kernel_sizes[0], vhp.resblock_kernel_sizes[1],
                vhp.resblock_kernel_sizes[2]);
    }

    if (ctx->max_len <= 0) {
        ctx->max_len = hp.max_speech_positions / hp.reduction_factor;
    }

    return ctx;
}

int speecht5_tts_set_speaker(struct speecht5_tts_context* ctx, const float* xvector, int dim) {
    if (!ctx || !xvector || dim <= 0)
        return -1;
    ctx->speaker_emb.assign(xvector, xvector + dim);
    // Pad or truncate to speaker_dim
    ctx->speaker_emb.resize(ctx->hp.speaker_dim, 0.0f);
    return 0;
}

float* speecht5_tts_synthesize(struct speecht5_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (ctx->speaker_emb.empty()) {
        fprintf(stderr, "speecht5: no speaker embedding set. Call speecht5_tts_set_speaker() first.\n");
        return nullptr;
    }

    // Tokenize
    std::vector<int32_t> token_ids = ctx->tokenizer.encode(text);
    if (token_ids.empty()) {
        fprintf(stderr, "speecht5: failed to tokenize text\n");
        return nullptr;
    }
    if (ctx->verbosity > 0) {
        fprintf(stderr, "speecht5: tokenized %d chars -> %d tokens\n", (int)strlen(text), (int)token_ids.size());
    }

    // Run encoder
    int T_enc = 0;
    auto encoder_out = run_encoder(ctx, token_ids, &T_enc);
    if (encoder_out.empty()) {
        fprintf(stderr, "speecht5: encoder failed\n");
        return nullptr;
    }
    if (ctx->verbosity > 0) {
        fprintf(stderr, "speecht5: encoder output: %d tokens x %d dim\n", T_enc, ctx->hp.hidden_size);
    }

    // AR decoding loop
    const auto& hp = ctx->hp;
    int max_steps = ctx->max_len;
    int min_steps = (int)(T_enc * 0.0f / hp.reduction_factor); // minlenratio=0
    std::vector<float> all_mel_frames;                         // accumulated mel frames

    // Reset KV cache for this synthesis
    ctx->self_kv_cache.reset(hp.decoder_layers);

    // Start with zeros
    std::vector<float> prev_mel(hp.num_mel_bins, 0.0f);

    for (int step = 0; step < max_steps; step++) {
        auto result = run_decoder_step(ctx, encoder_out, T_enc, prev_mel, step);

        // Append mel frame
        all_mel_frames.insert(all_mel_frames.end(), result.mel_frame.begin(), result.mel_frame.end());

        // Update prev_mel with last mel frame of this step
        // (last reduction_factor-th frame)
        for (int d = 0; d < hp.num_mel_bins; d++) {
            prev_mel[d] = result.mel_frame[(hp.reduction_factor - 1) * hp.num_mel_bins + d];
        }

        // Check stop condition
        if (step >= min_steps && result.stop_prob >= ctx->threshold) {
            if (ctx->verbosity > 0) {
                fprintf(stderr, "speecht5: stopped at step %d (prob=%.3f)\n", step + 1, result.stop_prob);
            }
            break;
        }

        if (ctx->verbosity > 1 && (step + 1) % 50 == 0) {
            fprintf(stderr, "speecht5: decoder step %d, stop_prob=%.3f\n", step + 1, result.stop_prob);
        }
    }

    // Total mel frames
    int T_mel = (int)all_mel_frames.size() / hp.num_mel_bins;
    if (T_mel <= 0) {
        fprintf(stderr, "speecht5: no mel frames generated\n");
        return nullptr;
    }
    if (ctx->verbosity > 0) {
        fprintf(stderr, "speecht5: generated %d mel frames\n", T_mel);
    }

    // Run post-net
    auto mel_refined = run_postnet(ctx, all_mel_frames, T_mel);
    if (mel_refined.empty()) {
        fprintf(stderr, "speecht5: postnet failed, using unrefined mel\n");
        mel_refined = all_mel_frames;
    }

    // Run vocoder
    auto waveform = run_vocoder(ctx, mel_refined, T_mel);
    if (waveform.empty()) {
        fprintf(stderr, "speecht5: vocoder failed\n");
        return nullptr;
    }

    if (ctx->verbosity > 0) {
        fprintf(stderr, "speecht5: generated %d audio samples (%.2fs at 16kHz)\n", (int)waveform.size(),
                (float)waveform.size() / 16000.0f);
    }

    // Return
    *out_n_samples = (int)waveform.size();
    float* pcm = (float*)malloc(waveform.size() * sizeof(float));
    if (!pcm)
        return nullptr;
    memcpy(pcm, waveform.data(), waveform.size() * sizeof(float));
    return pcm;
}

void speecht5_tts_pcm_free(float* pcm) {
    free(pcm);
}

void speecht5_tts_free(struct speecht5_tts_context* ctx) {
    if (ctx) {
        if (ctx->buf_perm)
            ggml_backend_buffer_free(ctx->buf_perm);
        if (ctx->ctx_perm)
            ggml_free(ctx->ctx_perm);
    }
    delete ctx;
}
