// fastpitch_tts.cpp -- NVIDIA FastPitch TTS native ggml runtime.
//
// Non-autoregressive TTS: text -> encode -> predict duration/pitch ->
// expand -> decode mel -> HiFi-GAN -> PCM.  Single forward pass, no AR loop.
//
// Section 133 in the CrispASR backend lineup.

#include "fastpitch_tts.h"

#include "core/align.h"
#include "core/gguf_loader.h"
#include "core/hifigan.h"

#include "ggml-alloc.h"
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

// ── Dump helpers (env-gated: FASTPITCH_DUMP_DIR) ────────────────────

static void dump_f32(const char* dir, const char* name, const float* data, size_t n) {
    if (!dir)
        return;
    std::string path = std::string(dir) + "/" + name + ".f32";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return;
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "fastpitch: dumped %s (%zu floats)\n", path.c_str(), n);
}

static void dump_i32(const char* dir, const char* name, const int32_t* data, size_t n) {
    if (!dir)
        return;
    std::string path = std::string(dir) + "/" + name + ".i32";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return;
    fwrite(data, sizeof(int32_t), n, f);
    fclose(f);
    fprintf(stderr, "fastpitch: dumped %s (%zu ints)\n", path.c_str(), n);
}

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
    int enc_d_inner = 1536;

    // Decoder
    int dec_n_layers = 6;
    int dec_n_heads = 1;
    int dec_d_head = 64;
    int dec_d_inner = 1536;

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

// ── mini_graph: short-lived ggml context for graph building ──────────

struct mini_graph {
    ggml_context* ctx;

    mini_graph(size_t mem = 256 * 1024 * 1024) {
        struct ggml_init_params gp = {};
        gp.mem_size = mem;
        gp.mem_buffer = nullptr;
        gp.no_alloc = true;
        ctx = ggml_init(gp);
    }
    ~mini_graph() {
        if (ctx)
            ggml_free(ctx);
    }
};

// ── Context ──────────────────────────────────────────────────────────

struct fastpitch_tts_context {
    fastpitch_tts_params params;
    fastpitch_hparams hp;

    // ggml model + backend
    ggml_backend_t backend = nullptr;     // primary (GPU or CPU)
    ggml_backend_t backend_cpu = nullptr; // always CPU (for n_threads, fallback)
    ggml_backend_buffer_t buf_w = nullptr;
    struct ggml_context* ctx_w = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Tokenizer: ARPABET phoneme-based (NeMo EnglishPhonemesTokenizer).
    // Vocabulary is loaded from GGUF metadata (semicolon-separated string).
    // Tokens include: space, consonants, stressed vowels, lowercase chars,
    // punctuation, special tokens (pad, blank, oov).
    std::map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    int pad_id = 0;
    int blank_id = -1;
    int space_id = 0;
    int n_vocab = 0;

    // Pitch normalization (from training data stats)
    float pitch_mean = 0.0f;
    float pitch_std = 1.0f;
};

// ── GGUF loading ─────────────────────────────────────────────────────

static uint32_t gguf_get_u32(const gguf_context* g, const char* key, uint32_t def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0)
        return def;
    return (uint32_t)gguf_get_val_u32(g, idx);
}

static float gguf_get_f32(const gguf_context* g, const char* key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0)
        return def;
    return gguf_get_val_f32(g, idx);
}

static std::string gguf_get_str(const gguf_context* g, const char* key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0)
        return "";
    return gguf_get_val_str(g, idx);
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

    // Pitch normalization stats
    ctx->pitch_mean = gguf_get_f32(gguf, "fastpitch.pitch_mean", 0.0f);
    ctx->pitch_std = gguf_get_f32(gguf, "fastpitch.pitch_std", 1.0f);

    // Load tokenizer vocabulary (semicolon-separated string)
    {
        std::string vocab_str = gguf_get_str(gguf, "fastpitch.tokenizer_vocab");
        if (!vocab_str.empty()) {
            size_t pos = 0;
            int id = 0;
            while (pos < vocab_str.size()) {
                size_t next = vocab_str.find(';', pos);
                if (next == std::string::npos)
                    next = vocab_str.size();
                std::string token = vocab_str.substr(pos, next - pos);
                ctx->id_to_token.push_back(token);
                ctx->token_to_id[token] = id;
                id++;
                pos = next + 1;
            }
            ctx->n_vocab = id;
            // Find special token IDs
            auto it_pad = ctx->token_to_id.find("<pad>");
            if (it_pad != ctx->token_to_id.end())
                ctx->pad_id = it_pad->second;
            auto it_blank = ctx->token_to_id.find("<blank>");
            if (it_blank != ctx->token_to_id.end())
                ctx->blank_id = it_blank->second;
            auto it_space = ctx->token_to_id.find(" ");
            if (it_space != ctx->token_to_id.end())
                ctx->space_id = it_space->second;
        }
    }

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

    // ── Init backends ──

    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "fastpitch: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (params.verbosity >= 1 && ctx->backend != ctx->backend_cpu) {
        fprintf(stderr, "fastpitch: using GPU backend: %s\n", ggml_backend_name(ctx->backend));
    }

    // ── Load tensors via core_gguf (allocates on the active backend) ──

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "fastpitch", wl)) {
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    // ── Create backend scheduler ──
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;

        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be,
                                            /*graph_size=*/16384,
                                            /*parallel=*/false, /*op_offload=*/false);
        if (!ctx->sched) {
            fprintf(stderr, "fastpitch: failed to create backend scheduler\n");
            delete ctx;
            return nullptr;
        }
    }

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
                (int)ctx->tensors.size(), ctx->n_vocab, hp.n_speakers, hp.symbols_embedding_dim, hp.enc_n_layers,
                hp.dec_n_layers, hp.n_mel_channels, hp.sample_rate);
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

    // Character-level tokenization using the NeMo ARPABET vocabulary.
    // For now this handles chars + punctuation; G2P phoneme conversion
    // is not yet implemented (would require CMUDict lookup).
    if (!ctx->token_to_id.empty()) {
        // Pad with space at beginning and end (NeMo pad_with_space=True)
        ids.push_back(ctx->space_id);

        // Convert to lowercase and tokenize character by character
        bool last_was_space = true;
        for (const char* p = text; *p; p++) {
            char c = *p;
            // Lowercase
            if (c >= 'A' && c <= 'Z')
                c = c - 'A' + 'a';

            std::string token(1, c);
            auto it = ctx->token_to_id.find(token);
            if (it != ctx->token_to_id.end()) {
                // Avoid consecutive spaces
                if (c == ' ') {
                    if (!last_was_space) {
                        ids.push_back(it->second);
                        last_was_space = true;
                    }
                } else {
                    ids.push_back(it->second);
                    last_was_space = false;
                }
            }
            // Unknown chars are skipped (NeMo behavior)
        }

        // Pad with space at end
        if (!ids.empty() && ids.back() != ctx->space_id) {
            ids.push_back(ctx->space_id);
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

// Conv1d: weight (K, Cin, Cout in ggml), bias (Cout,)
// ggml_conv_1d expects input as (T, Cin) and weight as (K, Cin, Cout).
// Output is 3D: (OL, OC, N). We squeeze to 2D (OL, OC) and add bias.
static ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride = 1,
                           int pad = 0, int dilation = 1) {
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    // ggml_conv_1d returns 3D (OL, OC, N). Squeeze batch dim to get (OL, OC).
    y = ggml_reshape_2d(ctx, y, (int)y->ne[0], (int)y->ne[1]);
    if (b) {
        ggml_tensor* bias = ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]);
        y = ggml_add(ctx, y, bias);
    }
    return y;
}

// Transpose helper: (A, B) -> (B, A) with ggml_cont to ensure contiguous
static ggml_tensor* transpose_2d(ggml_context* ctx, ggml_tensor* x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
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
    // x is (D, T) after get_rows: ne[0]=D, ne[1]=T

    // Positional embedding: pre-computed sinusoidal positions from inv_freq
    ggml_tensor* pos_emb = T(ts, "enc.pos_emb");
    if (pos_emb) {
        int T_len = (int)x->ne[1];
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

    // Transformer layers (POST-norm: LayerNorm after residual, matching NeMo default pre_lnorm=False)
    for (int i = 0; i < hp.enc_n_layers; i++) {
        std::string pfx = "enc.layer." + std::to_string(i);
        // Multi-head attention on raw input (post-norm = no norm before attention)
        ggml_tensor* qkv_w = T(ts, pfx + ".attn.qkv.weight");
        ggml_tensor* qkv_b = T(ts, pfx + ".attn.qkv.bias");

        ggml_tensor* qkv = ggml_mul_mat(gctx, qkv_w, x);
        if (qkv_b)
            qkv = ggml_add(gctx, qkv, qkv_b);

        int n_heads = hp.enc_n_heads;
        int d_head = hp.enc_d_head;
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

        // Post-norm: LayerNorm(residual + attn_out)
        x = ggml_add(gctx, x, attn_out);
        ggml_tensor* ln1_w = T(ts, pfx + ".attn_norm.weight");
        ggml_tensor* ln1_b = T(ts, pfx + ".attn_norm.bias");
        x = layer_norm(gctx, x, ln1_w, ln1_b);

        // PositionwiseConvFF on post-normed features (no norm before FFN in post-norm)
        // NeMo transposes to (B, D, T) for Conv1d, then back.
        // ggml_conv_1d expects input (T, Cin), so transpose (D, T) -> (T, D).
        ggml_tensor* ffn_c1_w = T(ts, pfx + ".ffn.conv1.weight");
        ggml_tensor* ffn_c1_b = T(ts, pfx + ".ffn.conv1.bias");
        ggml_tensor* ffn_c2_w = T(ts, pfx + ".ffn.conv2.weight");
        ggml_tensor* ffn_c2_b = T(ts, pfx + ".ffn.conv2.bias");

        int k1 = ffn_c1_w ? (int)ffn_c1_w->ne[0] : 3;
        int k2 = ffn_c2_w ? (int)ffn_c2_w->ne[0] : 3;
        int pad1 = (k1 - 1) / 2;
        int pad2 = (k2 - 1) / 2;

        ggml_tensor* xt = transpose_2d(gctx, x); // (D, T) -> (T, D) for conv
        ggml_tensor* ff = conv1d(gctx, xt, ffn_c1_w, ffn_c1_b, 1, pad1, 1);
        ff = ggml_relu(gctx, ff);
        ff = conv1d(gctx, ff, ffn_c2_w, ffn_c2_b, 1, pad2, 1);
        ff = transpose_2d(gctx, ff); // (T, D) -> (D, T) back

        // Post-norm: LayerNorm(residual + ffn_out)
        x = ggml_add(gctx, x, ff);
        ggml_tensor* ln2_w = T(ts, pfx + ".ffn_norm.weight");
        ggml_tensor* ln2_b = T(ts, pfx + ".ffn_norm.bias");
        x = layer_norm(gctx, x, ln2_w, ln2_b);
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

    // Conv layers: input x is (D, T), conv1d needs (T, D)
    int pad = (kernel_size - 1) / 2;
    // Transpose to (T, D) for conv1d
    x = transpose_2d(gctx, x);
    for (int i = 0; i < n_layers; i++) {
        std::string cpfx = prefix + ".conv." + std::to_string(i);
        std::string npfx = prefix + ".norm." + std::to_string(i);

        ggml_tensor* cw = T(ts, cpfx + ".weight");
        ggml_tensor* cb = T(ts, cpfx + ".bias");

        x = conv1d(gctx, x, cw, cb, 1, pad, 1);
        x = ggml_relu(gctx, x);

        // LayerNorm after conv: output of conv1d is (T, C), norm over C (ne[1])
        // Need to transpose to (C, T), norm, transpose back
        ggml_tensor* ln_w = T(ts, npfx + ".weight");
        ggml_tensor* ln_b = T(ts, npfx + ".bias");
        if (ln_w) {
            x = transpose_2d(gctx, x); // (T, C) -> (C, T)
            x = layer_norm(gctx, x, ln_w, ln_b);
            x = transpose_2d(gctx, x); // (C, T) -> (T, C)
        }
    }
    // Back to (C, T) for linear projection
    x = transpose_2d(gctx, x); // (T, C) -> (C, T)

    // Final linear projection: fc.weight (1, filter_size), fc.bias (1,)
    ggml_tensor* fc_w = T(ts, prefix + ".fc.weight");
    ggml_tensor* fc_b = T(ts, prefix + ".fc.bias");

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

    // Transformer layers (POST-norm, matching NeMo default pre_lnorm=False)
    for (int i = 0; i < hp.dec_n_layers; i++) {
        std::string pfx = "dec.layer." + std::to_string(i);

        // Multi-head attention on raw input (post-norm)
        ggml_tensor* qkv_w = T(ts, pfx + ".attn.qkv.weight");
        ggml_tensor* qkv_b = T(ts, pfx + ".attn.qkv.bias");

        ggml_tensor* qkv = ggml_mul_mat(gctx, qkv_w, x);
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

        // Post-norm: LayerNorm(residual + attn_out)
        x = ggml_add(gctx, x, attn_out);
        ggml_tensor* ln1_w = T(ts, pfx + ".attn_norm.weight");
        ggml_tensor* ln1_b = T(ts, pfx + ".attn_norm.bias");
        x = layer_norm(gctx, x, ln1_w, ln1_b);

        // PositionwiseConvFF (post-norm: no norm before FFN)
        // Transpose for conv1d: (D, T) -> (T, D)
        ggml_tensor* ffn_c1_w = T(ts, pfx + ".ffn.conv1.weight");
        ggml_tensor* ffn_c1_b = T(ts, pfx + ".ffn.conv1.bias");
        ggml_tensor* ffn_c2_w = T(ts, pfx + ".ffn.conv2.weight");
        ggml_tensor* ffn_c2_b = T(ts, pfx + ".ffn.conv2.bias");

        int k1 = ffn_c1_w ? (int)ffn_c1_w->ne[0] : 3;
        int k2 = ffn_c2_w ? (int)ffn_c2_w->ne[0] : 3;
        int pad1 = (k1 - 1) / 2;
        int pad2 = (k2 - 1) / 2;

        ggml_tensor* xt = transpose_2d(gctx, x);
        ggml_tensor* ff = conv1d(gctx, xt, ffn_c1_w, ffn_c1_b, 1, pad1, 1);
        ff = ggml_relu(gctx, ff);
        ff = conv1d(gctx, ff, ffn_c2_w, ffn_c2_b, 1, pad2, 1);
        ff = transpose_2d(gctx, ff); // back to (D, T)

        // Post-norm: LayerNorm(residual + ffn_out)
        x = ggml_add(gctx, x, ff);
        ggml_tensor* ln2_w = T(ts, pfx + ".ffn_norm.weight");
        ggml_tensor* ln2_b = T(ts, pfx + ".ffn_norm.bias");
        x = layer_norm(gctx, x, ln2_w, ln2_b);
    }

    return x;
}

// ── Full synthesis pipeline ──────────────────────────────────────────

static int synthesize_internal(fastpitch_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    const auto& hp = ctx->hp;
    const int D = hp.symbols_embedding_dim;
    const char* dump_dir = getenv("FASTPITCH_DUMP_DIR");

    // ── Step 1: Tokenize ──
    std::vector<int> token_ids;

    // Allow teacher-forcing tokens from file (for diff testing)
    const char* force_tokens_path = getenv("FASTPITCH_FORCE_TOKENS");
    if (force_tokens_path) {
        FILE* f = fopen(force_tokens_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            int n = (int)(sz / sizeof(int32_t));
            token_ids.resize(n);
            fread(token_ids.data(), sizeof(int32_t), n, f);
            fclose(f);
            fprintf(stderr, "fastpitch: forced %d tokens from %s\n", n, force_tokens_path);
        }
    }

    if (token_ids.empty()) {
        token_ids = tokenize_text(ctx, text);
    }

    if (token_ids.empty()) {
        fprintf(stderr, "fastpitch: empty token sequence for input text\n");
        return 0;
    }
    int T_text = (int)token_ids.size();

    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "fastpitch: %d tokens from '%s'\n", T_text, text);
    }

    dump_i32(dump_dir, "cpp_tokens", (const int32_t*)token_ids.data(), T_text);

    // ── Step 2: Build and run encoder + predictors graph ──
    {
        mini_graph mg;
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

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
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
        ggml_backend_sched_graph_compute(ctx->sched, gf);

        // Read outputs
        std::vector<float> dur_data(T_text);
        std::vector<float> pitch_data(T_text);
        std::vector<float> enc_data((size_t)D * T_text);

        // dur_pred shape: (1, T_text) -- log durations
        ggml_backend_tensor_get(dur_pred, dur_data.data(), 0, T_text * sizeof(float));
        ggml_backend_tensor_get(pitch_pred, pitch_data.data(), 0, T_text * sizeof(float));
        ggml_backend_tensor_get(enc_copy, enc_data.data(), 0, (size_t)D * T_text * sizeof(float));

        dump_f32(dump_dir, "cpp_enc_out", enc_data.data(), (size_t)D * T_text);
        dump_f32(dump_dir, "cpp_dur_pred", dur_data.data(), T_text);
        dump_f32(dump_dir, "cpp_pitch_pred", pitch_data.data(), T_text);

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

        {
            std::vector<int32_t> dur_i32(durations.begin(), durations.end());
            dump_i32(dump_dir, "cpp_durations", dur_i32.data(), T_text);
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
            mini_graph mg_pitch(64 * 1024 * 1024);
            auto* gc2 = mg_pitch.ctx;

            // Input: pitch values — conv1d expects (T, Cin) = (T_frames, 1)
            ggml_tensor* pitch_in = ggml_new_tensor_2d(gc2, GGML_TYPE_F32, T_frames, 1);
            ggml_set_name(pitch_in, "pitch_in");
            ggml_set_input(pitch_in);

            int pk = hp.pitch_embedding_kernel_size;
            int ppad = (pk - 1) / 2;
            ggml_tensor* pemb = conv1d(gc2, pitch_in, pitch_emb_w, pitch_emb_b, 1, ppad, 1);
            // pemb is (T_frames, D) from conv1d, transpose to (D, T_frames) for add
            pemb = transpose_2d(gc2, pemb);
            ggml_set_name(pemb, "pitch_emb");
            ggml_set_output(pemb);

            ggml_cgraph* gf2 = ggml_new_graph_custom(gc2, 256, false);
            ggml_build_forward_expand(gf2, pemb);

            ggml_backend_sched_reset(ctx->sched);
            if (ggml_backend_sched_alloc_graph(ctx->sched, gf2)) {
                ggml_backend_tensor_set(pitch_in, pitch_frames.data(), 0, T_frames * sizeof(float));
                ggml_backend_sched_graph_compute(ctx->sched, gf2);

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
            mini_graph mg_dec;
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

            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf3)) {
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

        dump_f32(dump_dir, "cpp_mel", dec_out_data.data(), dec_out_data.size());
        free(expanded);

        // ── Step 7: HiFi-GAN vocoder ──

        int T_mel = T_frames;
        {
            mini_graph mg_voc;
            auto* gc4 = mg_voc.ctx;

            // Input mel for HiFi-GAN vocoder.
            // Decoder output is ggml (ne[0]=n_mel, ne[1]=T_mel), stored as n_mel
            // contiguous values per timestep in the flat buffer.
            // HiFi-GAN (core_hifigan::forward) expects ne[0]=T_mel, ne[1]=n_mel
            // where ne[0] is the spatial dimension for conv1d.
            // We must CPU-transpose the data to match.
            ggml_tensor* mel_in = ggml_new_tensor_2d(gc4, GGML_TYPE_F32, T_mel, hp.n_mel_channels);
            ggml_set_name(mel_in, "voc_mel_in");
            ggml_set_input(mel_in);

            // Run shared HiFi-GAN forward
            ggml_tensor* audio = core_hifigan::forward(gc4, mel_in, ctx->tensors, "voc", hp.voc_hp);

            ggml_set_name(audio, "audio_out");
            ggml_set_output(audio);

            ggml_cgraph* gf4 = ggml_new_graph_custom(gc4, 16384, false);
            ggml_build_forward_expand(gf4, audio);

            ggml_backend_sched_reset(ctx->sched);
            if (!ggml_backend_sched_alloc_graph(ctx->sched, gf4)) {
                fprintf(stderr, "fastpitch: vocoder graph alloc failed\n");
                return 0;
            }

            // Transpose mel from ggml (n_mel, T) to vocoder (T, n_mel) layout.
            // Source: data[t * n_mel + c] (n_mel contiguous per timestep)
            // Target: data[c * T_mel + t] (T_mel contiguous per channel for ne[0]=T_mel)
            {
                std::vector<float> mel_voc((size_t)hp.n_mel_channels * T_mel);
                for (int t = 0; t < T_mel; t++) {
                    for (int c = 0; c < hp.n_mel_channels; c++) {
                        mel_voc[(size_t)c * T_mel + t] = dec_out_data[(size_t)t * hp.n_mel_channels + c];
                    }
                }
                ggml_backend_tensor_set(mel_in, mel_voc.data(), 0, mel_voc.size() * sizeof(float));
            }

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

            dump_f32(dump_dir, "cpp_audio", pcm, T_audio);

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
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
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
