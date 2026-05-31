// fastpitch_tts.cpp -- NVIDIA FastPitch TTS native ggml runtime.
//
// Non-autoregressive TTS: text -> encode -> predict duration/pitch ->
// expand -> decode mel -> HiFi-GAN -> PCM.  Single forward pass, no AR loop.
//
// Section 133 in the CrispASR backend lineup.

#include "fastpitch_tts.h"

#include "core/align.h"
#include "core/hifigan.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ── Hyperparameters ──────────────────────────────────────────────────

struct fastpitch_hparams {
    int n_mel_channels = 80;
    int n_speakers = 5;
    int symbols_embedding_dim = 384;
    int max_token_duration = 75;

    // Encoder
    int enc_n_layers = 6;
    int enc_n_heads = 1;
    int enc_d_head = 64;
    int enc_d_inner = 1024;

    // Decoder
    int dec_n_layers = 6;
    int dec_n_heads = 1;
    int dec_d_head = 64;
    int dec_d_inner = 1024;

    // Duration predictor
    int dur_n_layers = 2;
    int dur_filter_size = 256;
    int dur_kernel_size = 3;

    // Pitch predictor
    int pitch_n_layers = 2;
    int pitch_filter_size = 256;
    int pitch_kernel_size = 3;

    // Pitch embedding
    int pitch_embedding_kernel_size = 3;

    // Audio
    int sample_rate = 22050;

    // Vocoder
    core_hifigan::hparams voc_hp;
};

// ── mini_graph: short-lived ggml context + backend allocator ─────────

struct mini_graph {
    ggml_backend_t backend;
    ggml_context* ctx;
    ggml_gallocr_t alloc;

    mini_graph(ggml_backend_t be, size_t mem = 256 * 1024 * 1024) : backend(be) {
        struct ggml_init_params gp = {};
        gp.mem_size = mem;
        gp.mem_buffer = nullptr;
        gp.no_alloc = true;
        ctx = ggml_init(gp);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    }
    ~mini_graph() {
        if (alloc)
            ggml_gallocr_free(alloc);
        if (ctx)
            ggml_free(ctx);
    }
};

// ── Context ──────────────────────────────────────────────────────────

struct fastpitch_tts_context {
    fastpitch_tts_params params;
    fastpitch_hparams hp;

    // ggml model
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context* ctx_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Tokenizer: simple char-level mapping for German text.
    // NeMo FastPitch uses a phoneme tokenizer (IPA via G2P).
    // For the GGUF runtime we store the phoneme_id_map in metadata or
    // fall back to a basic character-level tokenizer.
    std::map<std::string, int> char_to_id;
    int pad_id = 0;
    int n_vocab = 0;
};

// ── GGUF loading ─────────────────────────────────────────────────────

static uint32_t gguf_get_u32(const gguf_context* g, const char* key, uint32_t def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0)
        return def;
    return (uint32_t)gguf_get_val_u32(g, idx);
}

static std::vector<int> gguf_get_int_array(const gguf_context* g, const char* key) {
    std::vector<int> result;
    int idx = gguf_find_key(g, key);
    if (idx < 0)
        return result;
    const int n = (int)gguf_get_arr_n(g, idx);
    for (int i = 0; i < n; i++) {
        result.push_back((int)((const int32_t*)gguf_get_arr_data(g, idx))[i]);
    }
    return result;
}

static fastpitch_tts_context* load_model(const char* path, fastpitch_tts_params params) {
    struct gguf_init_params gip = {};
    gip.no_alloc = false;
    gip.ctx = nullptr;

    struct gguf_context* gguf = gguf_init_from_file(path, gip);
    if (!gguf) {
        fprintf(stderr, "fastpitch: failed to open GGUF: %s\n", path);
        return nullptr;
    }

    auto* ctx = new fastpitch_tts_context();
    ctx->params = params;

    // Read hyperparameters
    auto& hp = ctx->hp;
    hp.n_mel_channels = (int)gguf_get_u32(gguf, "fastpitch.n_mel_channels", 80);
    hp.n_speakers = (int)gguf_get_u32(gguf, "fastpitch.n_speakers", 5);
    hp.symbols_embedding_dim = (int)gguf_get_u32(gguf, "fastpitch.symbols_embedding_dim", 384);
    hp.max_token_duration = (int)gguf_get_u32(gguf, "fastpitch.max_token_duration", 75);

    hp.enc_n_layers = (int)gguf_get_u32(gguf, "fastpitch.enc_n_layers", 6);
    hp.enc_n_heads = (int)gguf_get_u32(gguf, "fastpitch.enc_n_heads", 1);
    hp.enc_d_head = (int)gguf_get_u32(gguf, "fastpitch.enc_d_head", 64);
    hp.enc_d_inner = (int)gguf_get_u32(gguf, "fastpitch.enc_d_inner", 1024);

    hp.dec_n_layers = (int)gguf_get_u32(gguf, "fastpitch.dec_n_layers", 6);
    hp.dec_n_heads = (int)gguf_get_u32(gguf, "fastpitch.dec_n_heads", 1);
    hp.dec_d_head = (int)gguf_get_u32(gguf, "fastpitch.dec_d_head", 64);
    hp.dec_d_inner = (int)gguf_get_u32(gguf, "fastpitch.dec_d_inner", 1024);

    hp.dur_n_layers = (int)gguf_get_u32(gguf, "fastpitch.dur_n_layers", 2);
    hp.dur_filter_size = (int)gguf_get_u32(gguf, "fastpitch.dur_filter_size", 256);
    hp.dur_kernel_size = (int)gguf_get_u32(gguf, "fastpitch.dur_kernel_size", 3);

    hp.pitch_n_layers = (int)gguf_get_u32(gguf, "fastpitch.pitch_n_layers", 2);
    hp.pitch_filter_size = (int)gguf_get_u32(gguf, "fastpitch.pitch_filter_size", 256);
    hp.pitch_kernel_size = (int)gguf_get_u32(gguf, "fastpitch.pitch_kernel_size", 3);

    hp.pitch_embedding_kernel_size = (int)gguf_get_u32(gguf, "fastpitch.pitch_embedding_kernel_size", 3);
    hp.sample_rate = (int)gguf_get_u32(gguf, "fastpitch.sample_rate", 22050);

    // Vocoder hparams
    hp.voc_hp.model_in_dim = (int)gguf_get_u32(gguf, "fastpitch.voc_model_in_dim", 80);
    hp.voc_hp.upsample_initial_ch = (int)gguf_get_u32(gguf, "fastpitch.voc_upsample_initial_ch", 512);
    hp.voc_hp.leaky_relu_slope = 0.1f;
    hp.voc_hp.normalize_before = false; // FastPitch's HiFi-GAN typically has no normalization

    auto rates = gguf_get_int_array(gguf, "fastpitch.voc_upsample_rates");
    auto kernels = gguf_get_int_array(gguf, "fastpitch.voc_upsample_kernel_sizes");
    auto rb_ks = gguf_get_int_array(gguf, "fastpitch.voc_resblock_kernel_sizes");
    auto rb_dil = gguf_get_int_array(gguf, "fastpitch.voc_resblock_dilations");
    int n_dil = (int)gguf_get_u32(gguf, "fastpitch.voc_n_dilations", 3);

    if (!rates.empty())
        hp.voc_hp.upsample_rates = rates;
    else
        hp.voc_hp.upsample_rates = {8, 8, 2, 2};

    if (!kernels.empty())
        hp.voc_hp.upsample_kernel_sizes = kernels;
    else
        hp.voc_hp.upsample_kernel_sizes = {16, 16, 4, 4};

    if (!rb_ks.empty())
        hp.voc_hp.resblock_kernel_sizes = rb_ks;
    else
        hp.voc_hp.resblock_kernel_sizes = {3, 7, 11};

    // Parse flat dilation array into per-kernel dilation vectors
    hp.voc_hp.resblock_dilation_sizes.clear();
    if (!rb_dil.empty() && n_dil > 0) {
        int n_rb_kernels = (int)hp.voc_hp.resblock_kernel_sizes.size();
        for (int j = 0; j < n_rb_kernels; j++) {
            std::vector<int> dils;
            for (int d = 0; d < n_dil && (j * n_dil + d) < (int)rb_dil.size(); d++) {
                dils.push_back(rb_dil[j * n_dil + d]);
            }
            hp.voc_hp.resblock_dilation_sizes.push_back(dils);
        }
    } else {
        hp.voc_hp.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    }

    gguf_free(gguf);

    // ── Load tensors into ggml context ──

    struct gguf_init_params gip2 = {};
    gip2.no_alloc = false;
    gip2.ctx = &ctx->ctx_w;

    struct gguf_context* gguf2 = gguf_init_from_file(path, gip2);
    if (!gguf2) {
        delete ctx;
        return nullptr;
    }

    // Collect tensor map
    const int n_tensors = gguf_get_n_tensors(gguf2);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf2, i);
        ggml_tensor* t = ggml_get_tensor(ctx->ctx_w, name);
        if (t) {
            ctx->tensors[name] = t;
        }
    }

    gguf_free(gguf2);

    // CPU backend
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "fastpitch: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);

    // Infer vocab size from embedding tensor
    auto it = ctx->tensors.find("enc.emb.weight");
    if (it != ctx->tensors.end()) {
        // ggml layout: (d_model, n_vocab)
        ctx->n_vocab = (int)it->second->ne[1];
    }

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "fastpitch: loaded %d tensors, %d vocab, %d speakers, "
                "d_model=%d, enc=%d layers, dec=%d layers, mel=%d, sr=%d\n",
                n_tensors, ctx->n_vocab, hp.n_speakers, hp.symbols_embedding_dim, hp.enc_n_layers, hp.dec_n_layers,
                hp.n_mel_channels, hp.sample_rate);
    }

    return ctx;
}

// ── Tokenizer ────────────────────────────────────────────────────────
//
// NeMo FastPitch uses a phoneme tokenizer. For initial runtime we do
// simple character-level tokenization: each character maps to its Unicode
// codepoint index, clamped to vocab size. A proper G2P + phoneme map
// should be loaded from GGUF metadata in production.

static std::vector<int> tokenize_text(const fastpitch_tts_context* ctx, const char* text) {
    std::vector<int> ids;
    if (!text)
        return ids;

    // Use char_to_id map if populated, otherwise raw codepoints
    if (!ctx->char_to_id.empty()) {
        const char* p = text;
        while (*p) {
            // Try multi-char match (up to 4 bytes for UTF-8)
            bool found = false;
            for (int len = 4; len >= 1; len--) {
                if (p + len > text + strlen(text))
                    continue;
                std::string candidate(p, len);
                auto it = ctx->char_to_id.find(candidate);
                if (it != ctx->char_to_id.end()) {
                    ids.push_back(it->second);
                    p += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Unknown character -> skip or use pad
                p++;
            }
        }
    } else {
        // Fallback: raw ASCII codepoints clamped to vocab
        int vocab = ctx->n_vocab > 0 ? ctx->n_vocab : 256;
        for (const char* p = text; *p; p++) {
            int c = (unsigned char)*p;
            if (c < vocab) {
                ids.push_back(c);
            }
        }
    }

    return ids;
}

// ── Graph helpers ────────────────────────────────────────────────────

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

// Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta,
                               float eps = 1e-5f) {
    x = ggml_norm(ctx, x, eps);
    if (gamma)
        x = ggml_mul(ctx, x, gamma);
    if (beta)
        x = ggml_add(ctx, x, beta);
    return x;
}

// Conv1d: weight (K, Cin, Cout in ggml), bias (Cout,), input (D, T)
static ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride = 1,
                           int pad = 0, int dilation = 1) {
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (b) {
        ggml_tensor* bias = ggml_reshape_2d(ctx, b, (int)b->ne[0], 1);
        y = ggml_add(ctx, y, bias);
    }
    return y;
}

// ── Encoder forward pass ─────────────────────────────────────────────
//
// FFTransformerEncoder: embedding + positional encoding + N transformer layers
// Each layer: LayerNorm -> MultiHeadAttn -> residual -> LayerNorm -> ConvFFN -> residual
//
// Input:  token IDs as 1D integer tensor
// Output: (D, T) encoded features

static ggml_tensor* build_encoder_graph(ggml_context* gctx, const fastpitch_tts_context* ctx,
                                        ggml_tensor* token_ids,  // (T,) I32
                                        ggml_tensor* speaker_emb // (D,) or nullptr
) {
    const auto& hp = ctx->hp;
    const auto& ts = ctx->tensors;
    const int D = hp.symbols_embedding_dim;

    // Token embedding: lookup
    ggml_tensor* emb_w = T(ts, "enc.emb.weight");
    ggml_tensor* x = ggml_get_rows(gctx, emb_w, token_ids);
    // x is (D, T) after get_rows with transposed embedding

    // Positional embedding: add learned sinusoidal positions
    ggml_tensor* pos_emb = T(ts, "enc.pos_emb");
    if (pos_emb) {
        // pos_emb is typically (1, D, max_len). We need (D, T) slice.
        int T_len = (int)x->ne[1];
        // Create a view of the first T positions
        ggml_tensor* pos_slice = ggml_view_2d(gctx, pos_emb, D, T_len, pos_emb->nb[1], 0);
        x = ggml_add(gctx, x, pos_slice);
    }

    // Add speaker conditioning (if multi-speaker, "add" mode)
    ggml_tensor* cond_proj = T(ts, "enc.cond_input.add_proj.weight");
    if (speaker_emb && cond_proj) {
        // Project speaker embedding to D dims
        ggml_tensor* spk = ggml_mul_mat(gctx, cond_proj, speaker_emb);
        ggml_tensor* cond_b = T(ts, "enc.cond_input.add_proj.bias");
        if (cond_b)
            spk = ggml_add(gctx, spk, cond_b);
        // Broadcast add: spk is (D,1), x is (D,T)
        spk = ggml_reshape_2d(gctx, spk, D, 1);
        x = ggml_add(gctx, x, spk);
    }

    // Transformer layers
    for (int i = 0; i < hp.enc_n_layers; i++) {
        std::string pfx = "enc.layer." + std::to_string(i);

        // Pre-norm for attention
        ggml_tensor* ln1_w = T(ts, pfx + ".attn_norm.weight");
        ggml_tensor* ln1_b = T(ts, pfx + ".attn_norm.bias");
        ggml_tensor* normed = layer_norm(gctx, x, ln1_w, ln1_b);

        // Multi-head attention (bidirectional, no causal mask)
        ggml_tensor* qkv_w = T(ts, pfx + ".attn.qkv.weight");
        ggml_tensor* qkv_b = T(ts, pfx + ".attn.qkv.bias");

        // QKV projection: (3*n_heads*d_head, D) @ (D, T) -> (3*H*d, T)
        ggml_tensor* qkv = ggml_mul_mat(gctx, qkv_w, normed);
        if (qkv_b)
            qkv = ggml_add(gctx, qkv, qkv_b);

        int n_heads = hp.enc_n_heads;
        int d_head = hp.enc_d_head;
        int d_qkv = 3 * n_heads * d_head;
        int T_len = (int)x->ne[1];

        // Split Q, K, V: each (n_heads * d_head, T)
        ggml_tensor* Q = ggml_view_2d(gctx, qkv, n_heads * d_head, T_len, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(gctx, qkv, n_heads * d_head, T_len, qkv->nb[1],
                                      (size_t)(n_heads * d_head) * ggml_type_size(qkv->type));
        ggml_tensor* V = ggml_view_2d(gctx, qkv, n_heads * d_head, T_len, qkv->nb[1],
                                      (size_t)(2 * n_heads * d_head) * ggml_type_size(qkv->type));

        Q = ggml_cont(gctx, Q);
        K = ggml_cont(gctx, K);
        V = ggml_cont(gctx, V);

        // Reshape for multi-head: (d_head, T, n_heads)
        Q = ggml_reshape_3d(gctx, Q, d_head, T_len, n_heads);
        K = ggml_reshape_3d(gctx, K, d_head, T_len, n_heads);
        V = ggml_reshape_3d(gctx, V, d_head, T_len, n_heads);

        // Flash attention (bidirectional = no mask)
        // ggml_flash_attn_ext expects Q(d,T_q,n_heads), K(d,T_k,n_heads), V(d,T_k,n_heads)
        float scale = 1.0f / sqrtf((float)d_head);
        ggml_tensor* attn = ggml_flash_attn_ext(gctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);

        // Reshape back: (d_head, T, n_heads) -> (n_heads*d_head, T)
        attn = ggml_reshape_2d(gctx, attn, n_heads * d_head, T_len);

        // Output projection
        ggml_tensor* o_w = T(ts, pfx + ".attn.out.weight");
        ggml_tensor* o_b = T(ts, pfx + ".attn.out.bias");
        ggml_tensor* attn_out = ggml_mul_mat(gctx, o_w, attn);
        if (o_b)
            attn_out = ggml_add(gctx, attn_out, o_b);

        // Residual
        x = ggml_add(gctx, x, attn_out);

        // Pre-norm for FFN
        ggml_tensor* ln2_w = T(ts, pfx + ".ffn_norm.weight");
        ggml_tensor* ln2_b = T(ts, pfx + ".ffn_norm.bias");
        normed = layer_norm(gctx, x, ln2_w, ln2_b);

        // PositionwiseConvFF: Conv1d(D, d_inner, k1) -> ReLU -> Conv1d(d_inner, D, k2)
        ggml_tensor* ffn_c1_w = T(ts, pfx + ".ffn.conv1.weight");
        ggml_tensor* ffn_c1_b = T(ts, pfx + ".ffn.conv1.bias");
        ggml_tensor* ffn_c2_w = T(ts, pfx + ".ffn.conv2.weight");
        ggml_tensor* ffn_c2_b = T(ts, pfx + ".ffn.conv2.bias");

        // Determine padding from kernel size
        int k1 = ffn_c1_w ? (int)ffn_c1_w->ne[0] : 3;
        int k2 = ffn_c2_w ? (int)ffn_c2_w->ne[0] : 3;
        int pad1 = (k1 - 1) / 2;
        int pad2 = (k2 - 1) / 2;

        ggml_tensor* ff = conv1d(gctx, normed, ffn_c1_w, ffn_c1_b, 1, pad1, 1);
        ff = ggml_relu(gctx, ff);
        ff = conv1d(gctx, ff, ffn_c2_w, ffn_c2_b, 1, pad2, 1);

        // Residual
        x = ggml_add(gctx, x, ff);
    }

    return x; // (D, T)
}

// ── TemporalPredictor (duration/pitch predictor) ─────────────────────
//
// N conv layers: Conv1d -> ReLU -> LayerNorm -> (repeat)
// Final: Linear projection -> 1 value per timestep

static ggml_tensor* build_temporal_predictor(ggml_context* gctx, const fastpitch_tts_context* ctx,
                                             ggml_tensor* x,            // (D, T) encoder output
                                             const std::string& prefix, // "dur_pred" or "pitch_pred"
                                             int n_layers, int filter_size, int kernel_size,
                                             ggml_tensor* speaker_emb // optional conditioning
) {
    const auto& ts = ctx->tensors;

    // Optional ConditionalInput: project speaker embedding and add
    ggml_tensor* cond_proj_w = T(ts, prefix + ".cond_input.add_proj.weight");
    if (speaker_emb && cond_proj_w) {
        ggml_tensor* spk = ggml_mul_mat(gctx, cond_proj_w, speaker_emb);
        ggml_tensor* cond_b = T(ts, prefix + ".cond_input.add_proj.bias");
        if (cond_b)
            spk = ggml_add(gctx, spk, cond_b);
        int D = (int)x->ne[0];
        spk = ggml_reshape_2d(gctx, spk, D, 1);
        x = ggml_add(gctx, x, spk);
    }

    // Conv layers
    int pad = (kernel_size - 1) / 2;
    for (int i = 0; i < n_layers; i++) {
        std::string cpfx = prefix + ".conv." + std::to_string(i);
        std::string npfx = prefix + ".norm." + std::to_string(i);

        ggml_tensor* cw = T(ts, cpfx + ".weight");
        ggml_tensor* cb = T(ts, cpfx + ".bias");

        x = conv1d(gctx, x, cw, cb, 1, pad, 1);
        x = ggml_relu(gctx, x);

        // LayerNorm after conv (NeMo's ConditionalLayerNorm)
        ggml_tensor* ln_w = T(ts, npfx + ".weight");
        ggml_tensor* ln_b = T(ts, npfx + ".bias");
        if (ln_w) {
            // Conv output is (C, T). LayerNorm is over C dimension.
            // ggml_norm normalizes over ne[0] which is C — correct for (C, T).
            x = layer_norm(gctx, x, ln_w, ln_b);
        }
    }

    // Final linear projection to scalar per timestep
    // fc.weight: (1, filter_size), fc.bias: (1,)
    // But the output should be (1, T), so we do matmul
    ggml_tensor* fc_w = T(ts, prefix + ".fc.weight");
    ggml_tensor* fc_b = T(ts, prefix + ".fc.bias");

    // Transpose x to (T, filter_size) for matmul, then back
    ggml_tensor* out = ggml_mul_mat(gctx, fc_w, x);
    if (fc_b)
        out = ggml_add(gctx, out, fc_b);

    return out; // (1, T) -- one value per timestep
}

// ── Decoder forward pass ─────────────────────────────────────────────
//
// Same structure as encoder but without token embedding.
// Input: length-regulated features (D, T_frames)
// Output: (D, T_frames)

static ggml_tensor* build_decoder_graph(ggml_context* gctx, const fastpitch_tts_context* ctx,
                                        ggml_tensor* x,          // (D, T_frames)
                                        ggml_tensor* speaker_emb // optional
) {
    const auto& hp = ctx->hp;
    const auto& ts = ctx->tensors;
    const int D = hp.symbols_embedding_dim;

    // Add positional embedding
    ggml_tensor* pos_emb = T(ts, "dec.pos_emb");
    if (pos_emb) {
        int T_len = (int)x->ne[1];
        ggml_tensor* pos_slice = ggml_view_2d(gctx, pos_emb, D, T_len, pos_emb->nb[1], 0);
        x = ggml_add(gctx, x, pos_slice);
    }

    // Speaker conditioning
    ggml_tensor* cond_proj = T(ts, "dec.cond_input.add_proj.weight");
    if (speaker_emb && cond_proj) {
        ggml_tensor* spk = ggml_mul_mat(gctx, cond_proj, speaker_emb);
        ggml_tensor* cond_b = T(ts, "dec.cond_input.add_proj.bias");
        if (cond_b)
            spk = ggml_add(gctx, spk, cond_b);
        spk = ggml_reshape_2d(gctx, spk, D, 1);
        x = ggml_add(gctx, x, spk);
    }

    // Transformer layers (same structure as encoder)
    for (int i = 0; i < hp.dec_n_layers; i++) {
        std::string pfx = "dec.layer." + std::to_string(i);

        ggml_tensor* ln1_w = T(ts, pfx + ".attn_norm.weight");
        ggml_tensor* ln1_b = T(ts, pfx + ".attn_norm.bias");
        ggml_tensor* normed = layer_norm(gctx, x, ln1_w, ln1_b);

        ggml_tensor* qkv_w = T(ts, pfx + ".attn.qkv.weight");
        ggml_tensor* qkv_b = T(ts, pfx + ".attn.qkv.bias");

        ggml_tensor* qkv = ggml_mul_mat(gctx, qkv_w, normed);
        if (qkv_b)
            qkv = ggml_add(gctx, qkv, qkv_b);

        int n_heads = hp.dec_n_heads;
        int d_head = hp.dec_d_head;
        int T_len = (int)x->ne[1];

        ggml_tensor* Q = ggml_view_2d(gctx, qkv, n_heads * d_head, T_len, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(gctx, qkv, n_heads * d_head, T_len, qkv->nb[1],
                                      (size_t)(n_heads * d_head) * ggml_type_size(qkv->type));
        ggml_tensor* V = ggml_view_2d(gctx, qkv, n_heads * d_head, T_len, qkv->nb[1],
                                      (size_t)(2 * n_heads * d_head) * ggml_type_size(qkv->type));

        Q = ggml_cont(gctx, Q);
        K = ggml_cont(gctx, K);
        V = ggml_cont(gctx, V);

        Q = ggml_reshape_3d(gctx, Q, d_head, T_len, n_heads);
        K = ggml_reshape_3d(gctx, K, d_head, T_len, n_heads);
        V = ggml_reshape_3d(gctx, V, d_head, T_len, n_heads);

        float scale = 1.0f / sqrtf((float)d_head);
        ggml_tensor* attn = ggml_flash_attn_ext(gctx, Q, K, V, nullptr, scale, 0.0f, 0.0f);

        attn = ggml_reshape_2d(gctx, attn, n_heads * d_head, T_len);

        ggml_tensor* o_w = T(ts, pfx + ".attn.out.weight");
        ggml_tensor* o_b = T(ts, pfx + ".attn.out.bias");
        ggml_tensor* attn_out = ggml_mul_mat(gctx, o_w, attn);
        if (o_b)
            attn_out = ggml_add(gctx, attn_out, o_b);

        x = ggml_add(gctx, x, attn_out);

        ggml_tensor* ln2_w = T(ts, pfx + ".ffn_norm.weight");
        ggml_tensor* ln2_b = T(ts, pfx + ".ffn_norm.bias");
        normed = layer_norm(gctx, x, ln2_w, ln2_b);

        ggml_tensor* ffn_c1_w = T(ts, pfx + ".ffn.conv1.weight");
        ggml_tensor* ffn_c1_b = T(ts, pfx + ".ffn.conv1.bias");
        ggml_tensor* ffn_c2_w = T(ts, pfx + ".ffn.conv2.weight");
        ggml_tensor* ffn_c2_b = T(ts, pfx + ".ffn.conv2.bias");

        int k1 = ffn_c1_w ? (int)ffn_c1_w->ne[0] : 3;
        int k2 = ffn_c2_w ? (int)ffn_c2_w->ne[0] : 3;
        int pad1 = (k1 - 1) / 2;
        int pad2 = (k2 - 1) / 2;

        ggml_tensor* ff = conv1d(gctx, normed, ffn_c1_w, ffn_c1_b, 1, pad1, 1);
        ff = ggml_relu(gctx, ff);
        ff = conv1d(gctx, ff, ffn_c2_w, ffn_c2_b, 1, pad2, 1);

        x = ggml_add(gctx, x, ff);
    }

    return x;
}

// ── Full synthesis pipeline ──────────────────────────────────────────

static int synthesize_internal(fastpitch_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    const auto& hp = ctx->hp;
    const int D = hp.symbols_embedding_dim;

    // ── Step 1: Tokenize ──
    std::vector<int> token_ids = tokenize_text(ctx, text);
    if (token_ids.empty()) {
        fprintf(stderr, "fastpitch: empty token sequence for input text\n");
        return 0;
    }
    int T_text = (int)token_ids.size();

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "fastpitch: %d tokens from '%s'\n", T_text, text);
    }

    // ── Step 2: Build and run encoder + predictors graph ──
    {
        mini_graph mg(ctx->backend_cpu);
        auto* gc = mg.ctx;

        // Input: token IDs
        ggml_tensor* ids = ggml_new_tensor_1d(gc, GGML_TYPE_I32, T_text);
        ggml_set_name(ids, "token_ids");
        ggml_set_input(ids);

        // Speaker embedding (if multi-speaker)
        ggml_tensor* spk_emb = nullptr;
        ggml_tensor* spk_emb_input = nullptr;
        if (hp.n_speakers > 1) {
            ggml_tensor* spk_table = T(ctx->tensors, "speaker_emb.weight");
            if (spk_table) {
                // Speaker ID as 1-element I32 tensor
                spk_emb_input = ggml_new_tensor_1d(gc, GGML_TYPE_I32, 1);
                ggml_set_name(spk_emb_input, "speaker_id");
                ggml_set_input(spk_emb_input);
                spk_emb = ggml_get_rows(gc, spk_table, spk_emb_input);
                // spk_emb: (D, 1) -- one row from embedding table
            }
        }

        // Encoder
        ggml_tensor* enc_out = build_encoder_graph(gc, ctx, ids, spk_emb);

        // Duration predictor
        ggml_tensor* dur_pred = build_temporal_predictor(gc, ctx, enc_out, "dur_pred", hp.dur_n_layers,
                                                         hp.dur_filter_size, hp.dur_kernel_size, spk_emb);
        ggml_set_name(dur_pred, "dur_pred");
        ggml_set_output(dur_pred);

        // Pitch predictor
        ggml_tensor* pitch_pred = build_temporal_predictor(gc, ctx, enc_out, "pitch_pred", hp.pitch_n_layers,
                                                           hp.pitch_filter_size, hp.pitch_kernel_size, spk_emb);
        ggml_set_name(pitch_pred, "pitch_pred");
        ggml_set_output(pitch_pred);

        // Also output encoder features for length regulation
        ggml_tensor* enc_copy = ggml_cont(gc, enc_out);
        ggml_set_name(enc_copy, "enc_out");
        ggml_set_output(enc_copy);

        // Build graph
        ggml_cgraph* gf = ggml_new_graph_custom(gc, 16384, false);
        ggml_build_forward_expand(gf, dur_pred);
        ggml_build_forward_expand(gf, pitch_pred);
        ggml_build_forward_expand(gf, enc_copy);

        if (!ggml_gallocr_alloc_graph(mg.alloc, gf)) {
            fprintf(stderr, "fastpitch: encoder graph alloc failed\n");
            return 0;
        }

        // Set inputs
        ggml_backend_tensor_set(ids, token_ids.data(), 0, T_text * sizeof(int32_t));

        if (spk_emb_input) {
            int32_t sid = ctx->params.speaker_id;
            if (sid >= hp.n_speakers)
                sid = 0;
            ggml_backend_tensor_set(spk_emb_input, &sid, 0, sizeof(int32_t));
        }

        // Compute
        ggml_backend_graph_compute(ctx->backend_cpu, gf);

        // Read outputs
        std::vector<float> dur_data(T_text);
        std::vector<float> pitch_data(T_text);
        std::vector<float> enc_data((size_t)D * T_text);

        // dur_pred shape: (1, T_text) -- log durations
        ggml_backend_tensor_get(dur_pred, dur_data.data(), 0, T_text * sizeof(float));
        ggml_backend_tensor_get(pitch_pred, pitch_data.data(), 0, T_text * sizeof(float));
        ggml_backend_tensor_get(enc_copy, enc_data.data(), 0, (size_t)D * T_text * sizeof(float));

        // ── Step 3: Process durations and pitch (CPU) ──

        // Convert log durations to integer durations:
        // dur = clamp(round(exp(log_dur) - 1), 0, max_token_duration)
        std::vector<int> durations(T_text);
        float pace = ctx->params.pace > 0.0f ? ctx->params.pace : 1.0f;
        for (int i = 0; i < T_text; i++) {
            float d = expf(dur_data[i]) - 1.0f;
            d = d / pace; // adjust for speech rate
            int di = (int)roundf(d);
            if (di < 0)
                di = 0;
            if (di > hp.max_token_duration)
                di = hp.max_token_duration;
            durations[i] = di;
        }

        // Apply pitch shift
        if (ctx->params.pitch_shift != 0.0f) {
            for (int i = 0; i < T_text; i++) {
                pitch_data[i] += ctx->params.pitch_shift;
            }
        }

        // ── Step 4: Length regulation (repeat_interleave) ──

        int T_frames = 0;
        float* expanded = core_align::repeat_interleave(enc_data.data(), D, T_text, durations.data(), &T_frames);

        if (!expanded || T_frames <= 0) {
            fprintf(stderr, "fastpitch: length regulation produced 0 frames\n");
            if (expanded)
                free(expanded);
            return 0;
        }

        if (ctx->params.verbosity >= 2) {
            fprintf(stderr, "fastpitch: %d tokens -> %d frames (%.1fx expansion)\n", T_text, T_frames,
                    (float)T_frames / T_text);
        }

        // ── Step 5: Add pitch embedding to expanded features ──

        // Build pitch embedding graph: pitch values -> Conv1d -> add to features
        // First, expand pitch to frame-level using the same durations
        std::vector<float> pitch_frames(T_frames);
        {
            int j = 0;
            for (int i = 0; i < T_text; i++) {
                for (int k = 0; k < durations[i] && j < T_frames; k++) {
                    pitch_frames[j++] = pitch_data[i];
                }
            }
        }

        // Pitch embedding via Conv1d
        ggml_tensor* pitch_emb_w = T(ctx->tensors, "pitch_emb.weight");
        ggml_tensor* pitch_emb_b = T(ctx->tensors, "pitch_emb.bias");

        if (pitch_emb_w) {
            mini_graph mg_pitch(ctx->backend_cpu, 64 * 1024 * 1024);
            auto* gc2 = mg_pitch.ctx;

            // Input: pitch values (1, T_frames)
            ggml_tensor* pitch_in = ggml_new_tensor_2d(gc2, GGML_TYPE_F32, 1, T_frames);
            ggml_set_name(pitch_in, "pitch_in");
            ggml_set_input(pitch_in);

            int pk = hp.pitch_embedding_kernel_size;
            int ppad = (pk - 1) / 2;
            ggml_tensor* pemb = conv1d(gc2, pitch_in, pitch_emb_w, pitch_emb_b, 1, ppad, 1);
            ggml_set_name(pemb, "pitch_emb");
            ggml_set_output(pemb);

            ggml_cgraph* gf2 = ggml_new_graph_custom(gc2, 256, false);
            ggml_build_forward_expand(gf2, pemb);

            if (ggml_gallocr_alloc_graph(mg_pitch.alloc, gf2)) {
                ggml_backend_tensor_set(pitch_in, pitch_frames.data(), 0, T_frames * sizeof(float));
                ggml_backend_graph_compute(ctx->backend_cpu, gf2);

                // Add pitch embedding to expanded features
                std::vector<float> pemb_data((size_t)D * T_frames);
                ggml_backend_tensor_get(pemb, pemb_data.data(), 0, (size_t)D * T_frames * sizeof(float));

                for (int i = 0; i < D * T_frames; i++) {
                    expanded[i] += pemb_data[i];
                }
            }
        }

        // ── Step 6: Decoder ──

        std::vector<float> dec_out_data;
        {
            mini_graph mg_dec(ctx->backend_cpu);
            auto* gc3 = mg_dec.ctx;

            ggml_tensor* dec_in = ggml_new_tensor_2d(gc3, GGML_TYPE_F32, D, T_frames);
            ggml_set_name(dec_in, "dec_in");
            ggml_set_input(dec_in);

            // Speaker embedding for decoder conditioning
            ggml_tensor* dec_spk = nullptr;
            ggml_tensor* dec_spk_input = nullptr;
            if (hp.n_speakers > 1 && T(ctx->tensors, "speaker_emb.weight")) {
                dec_spk_input = ggml_new_tensor_1d(gc3, GGML_TYPE_I32, 1);
                ggml_set_name(dec_spk_input, "dec_speaker_id");
                ggml_set_input(dec_spk_input);
                dec_spk = ggml_get_rows(gc3, T(ctx->tensors, "speaker_emb.weight"), dec_spk_input);
            }

            ggml_tensor* dec_out = build_decoder_graph(gc3, ctx, dec_in, dec_spk);

            // Output projection: (n_mel, D) @ (D, T_frames) -> (n_mel, T_frames)
            ggml_tensor* proj_w = T(ctx->tensors, "proj.weight");
            ggml_tensor* proj_b = T(ctx->tensors, "proj.bias");
            ggml_tensor* mel = ggml_mul_mat(gc3, proj_w, dec_out);
            if (proj_b)
                mel = ggml_add(gc3, mel, proj_b);

            ggml_set_name(mel, "mel_output");
            ggml_set_output(mel);

            ggml_cgraph* gf3 = ggml_new_graph_custom(gc3, 16384, false);
            ggml_build_forward_expand(gf3, mel);

            if (!ggml_gallocr_alloc_graph(mg_dec.alloc, gf3)) {
                fprintf(stderr, "fastpitch: decoder graph alloc failed\n");
                free(expanded);
                return 0;
            }

            ggml_backend_tensor_set(dec_in, expanded, 0, (size_t)D * T_frames * sizeof(float));

            if (dec_spk_input) {
                int32_t sid = ctx->params.speaker_id;
                if (sid >= hp.n_speakers)
                    sid = 0;
                ggml_backend_tensor_set(dec_spk_input, &sid, 0, sizeof(int32_t));
            }

            ggml_backend_graph_compute(ctx->backend_cpu, gf3);

            // Read mel output: (n_mel, T_frames)
            dec_out_data.resize((size_t)hp.n_mel_channels * T_frames);
            ggml_backend_tensor_get(mel, dec_out_data.data(), 0, dec_out_data.size() * sizeof(float));
        }

        free(expanded);

        // ── Step 7: HiFi-GAN vocoder ──

        int T_mel = T_frames;
        {
            mini_graph mg_voc(ctx->backend_cpu);
            auto* gc4 = mg_voc.ctx;

            // Input mel: (n_mel, T_mel)
            ggml_tensor* mel_in = ggml_new_tensor_2d(gc4, GGML_TYPE_F32, hp.n_mel_channels, T_mel);
            ggml_set_name(mel_in, "voc_mel_in");
            ggml_set_input(mel_in);

            // Run shared HiFi-GAN forward
            ggml_tensor* audio = core_hifigan::forward(gc4, mel_in, ctx->tensors, "voc", hp.voc_hp);

            ggml_set_name(audio, "audio_out");
            ggml_set_output(audio);

            ggml_cgraph* gf4 = ggml_new_graph_custom(gc4, 16384, false);
            ggml_build_forward_expand(gf4, audio);

            if (!ggml_gallocr_alloc_graph(mg_voc.alloc, gf4)) {
                fprintf(stderr, "fastpitch: vocoder graph alloc failed\n");
                return 0;
            }

            ggml_backend_tensor_set(mel_in, dec_out_data.data(), 0, dec_out_data.size() * sizeof(float));

            ggml_backend_graph_compute(ctx->backend_cpu, gf4);

            // Read audio output
            int T_audio = (int)audio->ne[0];
            // In case output is (1, T_audio):
            if (audio->ne[1] > 1 && audio->ne[0] == 1) {
                T_audio = (int)audio->ne[1];
            } else {
                T_audio = (int)(ggml_nelements(audio));
            }

            float* pcm = (float*)malloc((size_t)T_audio * sizeof(float));
            if (!pcm)
                return 0;

            ggml_backend_tensor_get(audio, pcm, 0, (size_t)T_audio * sizeof(float));

            *pcm_out = pcm;
            if (sample_rate_out)
                *sample_rate_out = hp.sample_rate;

            if (ctx->params.verbosity >= 1) {
                float dur_s = (float)T_audio / hp.sample_rate;
                fprintf(stderr, "fastpitch: synthesized %d samples (%.2fs @ %d Hz)\n", T_audio, dur_s, hp.sample_rate);
            }

            return T_audio;
        }
    }

    return 0; // unreachable
}

// ── Public C API ─────────────────────────────────────────────────────

struct fastpitch_tts_params fastpitch_tts_default_params(void) {
    struct fastpitch_tts_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.speaker_id = 0;
    p.pace = 1.0f;
    p.pitch_shift = 0.0f;
    return p;
}

struct fastpitch_tts_context* fastpitch_tts_init_from_file(const char* path_model, struct fastpitch_tts_params params) {
    if (!path_model)
        return nullptr;
    return load_model(path_model, params);
}

void fastpitch_tts_free(struct fastpitch_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    // buf_w is freed by ggml_free(ctx_w) when ctx_w owns it
    delete ctx;
}

int fastpitch_tts_synthesize(struct fastpitch_tts_context* ctx, const char* text, float** pcm_out,
                             int* sample_rate_out) {
    if (!ctx || !text || !pcm_out)
        return 0;
    *pcm_out = nullptr;
    return synthesize_internal(ctx, text, pcm_out, sample_rate_out);
}

void fastpitch_tts_set_speaker(struct fastpitch_tts_context* ctx, int speaker_id) {
    if (ctx)
        ctx->params.speaker_id = speaker_id;
}

void fastpitch_tts_set_pace(struct fastpitch_tts_context* ctx, float pace) {
    if (ctx)
        ctx->params.pace = pace;
}

void fastpitch_tts_set_pitch_shift(struct fastpitch_tts_context* ctx, float shift) {
    if (ctx)
        ctx->params.pitch_shift = shift;
}

int fastpitch_tts_sample_rate(const struct fastpitch_tts_context* ctx) {
    return ctx ? ctx->hp.sample_rate : 22050;
}

int fastpitch_tts_n_speakers(const struct fastpitch_tts_context* ctx) {
    return ctx ? ctx->hp.n_speakers : 1;
}
