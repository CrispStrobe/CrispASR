// kyutai_stt.cpp — Kyutai STT runtime (stt-1b-en_fr, stt-2.6b-en).
//
// Architecture: Mimi audio codec encoder (SEANet CNN + 8L transformer + downsample + RVQ)
//             + 16L causal transformer LM (2048d, RoPE, SwiGLU, RMSNorm)
//
// Audio flow (batch, non-streaming):
//   24 kHz PCM → SEANet encoder (stride 960 → 25 Hz)
//             → 8-layer transformer (dim=512, 8 heads, RoPE, gating FFN)
//             → stride-2 downsample conv → 12.5 Hz
//             → RVQ encode (32 codebooks, 2048 entries, dim=256)
//             → LM: sum 32 audio embeddings per frame, autoregressive decode
//             → SentencePiece vocab lookup → text
//
// Reference: moshi.cpp (MIT, github.com/kyutai-labs/moshi)

#include "kyutai_stt.h"

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
#include <string>
#include <vector>

// Temperature-aware token selection: argmax when temp<=0, softmax sampling otherwise.
// When `out_prob` is non-null, also returns the softmax probability of the picked token.
static int sample_token(const float* logits, int vocab, float temperature, float* out_prob = nullptr) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[best])
                best = i;
        if (out_prob) {
            float maxv = logits[best];
            float s = 0.f;
            for (int i = 0; i < vocab; i++)
                s += expf(logits[i] - maxv);
            *out_prob = 1.0f / s;
        }
        return best;
    }
    // Softmax with temperature
    float maxv = logits[0];
    for (int i = 1; i < vocab; i++)
        if (logits[i] > maxv)
            maxv = logits[i];
    float sum = 0;
    std::vector<float> probs(vocab);
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf((logits[i] - maxv) / temperature);
        sum += probs[i];
    }
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab; i++)
        probs[i] *= inv_sum;
    float r = ((float)rand() / (float)RAND_MAX);
    float acc = 0;
    int picked = vocab - 1;
    for (int i = 0; i < vocab; i++) {
        acc += probs[i];
        if (acc >= r) {
            picked = i;
            break;
        }
    }
    if (out_prob)
        *out_prob = probs[picked];
    return picked;
}

// ===========================================================================
// Model structures
// ===========================================================================

struct kyutai_hparams {
    // LM
    int dim = 2048;
    int num_heads = 16;
    int num_layers = 16;
    int text_card = 8000;
    int card = 2048; // audio codebook size
    int n_q = 32;    // number of audio codebooks
    int context = 750;
    float max_period = 100000.0f;
    float hidden_scale = 4.125f;
    int existing_text_padding_id = 3;
    float audio_delay_seconds = 0.5f;

    // Mimi encoder
    int mimi_dim = 512;
    int mimi_num_heads = 8;
    int mimi_num_layers = 8;
    int mimi_context = 250;
    int codebook_dim = 256;
    int n_q_semantic = 1;
    int n_q_acoustic = 31;
    int sample_rate = 24000;
    float frame_rate = 12.5f;

    // Derived
    int head_dim = 128;     // dim / num_heads
    int mimi_head_dim = 64; // mimi_dim / mimi_num_heads
    int ffn_hidden = 0;     // computed from hidden_scale
};

// --- SEANet encoder layers (conv1d + resnet blocks) ---
// SEANet has this structure (kernel, stride, channels):
//   model.0:  Conv1d(1 → 64,  k=7, s=1)
//   model.1:  ResnetBlock(64)   [block.1: Conv(64→32, k=3), block.3: Conv(32→64, k=1)]
//   model.3:  Conv1d(64 → 128,  k=8, s=4)
//   model.4:  ResnetBlock(128)  [block.1: Conv(128→64, k=3), block.3: Conv(64→128, k=1)]
//   model.6:  Conv1d(128 → 256, k=10, s=5)
//   model.7:  ResnetBlock(256)  [block.1: Conv(256→128, k=3), block.3: Conv(128→256, k=1)]
//   model.9:  Conv1d(256 → 512, k=12, s=6)
//   model.10: ResnetBlock(512)  [block.1: Conv(512→256, k=3), block.3: Conv(256→512, k=1)]
//   model.12: Conv1d(512 → 1024, k=16, s=8)
//   model.14: Conv1d(1024 → 512, k=3, s=1)  ← "final" conv

struct seanet_conv {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;
};

struct seanet_resblock {
    seanet_conv shortcut; // block.1 conv
    seanet_conv expand;   // block.3 conv
};

struct seanet_encoder {
    // 5 strided convolutions + 4 resnet blocks + 1 final conv = 10 sections
    seanet_conv conv_init;       // model.0
    seanet_resblock resblock[4]; // model.1, 4, 7, 10
    seanet_conv conv_stride[4];  // model.3, 6, 9, 12
    seanet_conv conv_final;      // model.14
};

// --- Mimi encoder transformer layer ---
struct mimi_enc_layer {
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* attn_in_w = nullptr; // combined QKV
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* layer_scale_1 = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* linear1_w = nullptr; // FFN up (dim→4*dim)
    ggml_tensor* linear2_w = nullptr; // FFN down (4*dim→dim)
    ggml_tensor* layer_scale_2 = nullptr;
};

// --- RVQ codebook ---
struct rvq_codebook {
    ggml_tensor* embedding = nullptr; // [num_codes, codebook_dim]
};

struct rvq_group {
    seanet_conv input_proj;  // Conv1d: mimi_dim → codebook_dim
    seanet_conv output_proj; // Conv1d: codebook_dim → mimi_dim
    std::vector<rvq_codebook> codebooks;
};

// --- LM transformer layer ---
struct lm_layer {
    ggml_tensor* norm1_alpha = nullptr;  // RMSNorm
    ggml_tensor* attn_in_w = nullptr;    // combined QKV [dim, 3*dim]
    ggml_tensor* attn_out_w = nullptr;   // [dim, dim]
    ggml_tensor* norm2_alpha = nullptr;  // RMSNorm
    ggml_tensor* gating_in_w = nullptr;  // SwiGLU: [dim, 2*ffn_hidden]
    ggml_tensor* gating_out_w = nullptr; // [ffn_hidden, dim]
};

struct kyutai_model {
    kyutai_hparams hp;

    // Mimi encoder
    seanet_encoder seanet;
    std::vector<mimi_enc_layer> mimi_layers;
    seanet_conv downsample; // stride-2 conv

    // RVQ
    rvq_group rvq_first; // 1 semantic codebook
    rvq_group rvq_rest;  // 31 acoustic codebooks

    // LM
    std::vector<ggml_tensor*> audio_emb; // n_q embedding tables [card+1, dim]
    ggml_tensor* text_emb = nullptr;     // [text_card+1, dim]
    std::vector<lm_layer> lm_layers;
    ggml_tensor* out_norm_alpha = nullptr; // RMSNorm
    ggml_tensor* text_linear_w = nullptr;  // [dim, text_card]

    // Tokenizer (SentencePiece vocab baked into GGUF)
    std::vector<std::string> vocab;

    // Weight memory
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct kyutai_stt_context {
    kyutai_stt_context_params params;
    kyutai_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for LM (autoregressive decoding)
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // [head_dim, max_ctx, n_heads, n_layers]
    ggml_tensor* kv_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Implementation
// ===========================================================================

extern "C" struct kyutai_stt_context_params kyutai_stt_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true, /*temperature=*/0.0f};
}

// --- Helpers ---

static void load_seanet_conv(const std::map<std::string, ggml_tensor*>& ts, const char* prefix, seanet_conv& conv) {
    std::string w_name = std::string(prefix) + ".weight";
    std::string b_name = std::string(prefix) + ".bias";
    auto it_w = ts.find(w_name);
    auto it_b = ts.find(b_name);
    conv.w = (it_w != ts.end()) ? it_w->second : nullptr;
    conv.b = (it_b != ts.end()) ? it_b->second : nullptr;
}

// Causal (left-padded) Conv1d, matching moshi StreamingConv1d.
// Padding: prepend (kernel_size - stride) zeros to the LEFT of the input.
// ggml_conv_1d: kernel a = [K, IC, OC], data b = [T, IC]
// output shape in ggml ne: [OL, OC, 1] (3D with batch=1)
static ggml_tensor* conv1d_fwd(ggml_context* ctx, const seanet_conv& conv, ggml_tensor* x, int stride) {
    int kernel_size = (int)conv.w->ne[0];
    int pad_left = kernel_size - stride;
    if (pad_left > 0) {
        // Causal (left-only) padding with ggml_pad_ext: lp0=pad_left on dim 0 (time)
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
    }
    // No padding in conv itself
    ggml_tensor* out = ggml_conv_1d(ctx, conv.w, x, stride, 0, 1);
    // [OL, OC, 1] → [OL, OC]
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
    if (conv.b) {
        ggml_tensor* b = ggml_reshape_2d(ctx, conv.b, 1, ggml_nelements(conv.b));
        out = ggml_add(ctx, out, b);
    }
    return out; // [T_out, channels_out] in ggml ne terms
}

// ELU activation: elu(x) = x if x > 0, else alpha*(exp(x)-1), alpha=1.0
static ggml_tensor* elu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_elu(ctx, x);
}

// ResnetBlock: shortcut(elu(x)) → expand → add(x)
static ggml_tensor* resblock_fwd(ggml_context* ctx, const seanet_resblock& rb, ggml_tensor* x) {
    ggml_tensor* h = elu(ctx, x);
    // block.1: Conv(c_in → c_in/2, k=3, s=1)
    h = conv1d_fwd(ctx, rb.shortcut, h, 1);
    h = elu(ctx, h);
    // block.3: Conv(c_in/2 → c_in, k=1, s=1, pad=0)
    h = conv1d_fwd(ctx, rb.expand, h, 1);
    // skip connection
    return ggml_add(ctx, x, h);
}

// ---- RMSNorm with alpha (Kyutai uses learned scale 'alpha') ----
// alpha shape is [dim, 1, 1] in the GGUF. We need [dim].
static ggml_tensor* rms_norm_alpha(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, float eps = 1e-5f) {
    ggml_tensor* n = ggml_rms_norm(ctx, x, eps);
    // alpha is stored as [dim, 1, 1], flatten to [dim]
    ggml_tensor* a = ggml_reshape_1d(ctx, alpha, alpha->ne[0]);
    return ggml_mul(ctx, n, a);
}

// ===========================================================================
// Model loading
// ===========================================================================

extern "C" struct kyutai_stt_context* kyutai_stt_init_from_file(const char* path_model,
                                                                struct kyutai_stt_context_params params) {
    auto* sctx = new kyutai_stt_context();
    sctx->params = params;
    sctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    sctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!sctx->backend)
        sctx->backend = ggml_backend_cpu_init();
    sctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(sctx->backend_cpu, sctx->n_threads);
    if (ggml_backend_is_cpu(sctx->backend))
        ggml_backend_cpu_set_n_threads(sctx->backend, sctx->n_threads);

    auto& m = sctx->model;
    auto& hp = m.hp;

    // ---- pass 1: read hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "kyutai_stt: failed to open '%s'\n", path_model);
            delete sctx;
            return nullptr;
        }
        // LM hparams
        hp.dim = core_gguf::kv_u32(gctx, "kyutai.dim", hp.dim);
        hp.num_heads = core_gguf::kv_u32(gctx, "kyutai.num_heads", hp.num_heads);
        hp.num_layers = core_gguf::kv_u32(gctx, "kyutai.num_layers", hp.num_layers);
        hp.text_card = core_gguf::kv_u32(gctx, "kyutai.text_card", hp.text_card);
        hp.card = core_gguf::kv_u32(gctx, "kyutai.card", hp.card);
        hp.n_q = core_gguf::kv_u32(gctx, "kyutai.n_q", hp.n_q);
        hp.context = core_gguf::kv_u32(gctx, "kyutai.context", hp.context);
        hp.max_period = core_gguf::kv_f32(gctx, "kyutai.max_period", hp.max_period);
        hp.hidden_scale = core_gguf::kv_f32(gctx, "kyutai.hidden_scale", hp.hidden_scale);
        hp.existing_text_padding_id =
            core_gguf::kv_u32(gctx, "kyutai.existing_text_padding_id", hp.existing_text_padding_id);
        hp.audio_delay_seconds = core_gguf::kv_f32(gctx, "kyutai.stt.audio_delay_seconds", hp.audio_delay_seconds);

        // Mimi hparams
        hp.mimi_dim = core_gguf::kv_u32(gctx, "kyutai.mimi.encoder_dim", hp.mimi_dim);
        hp.mimi_num_heads = core_gguf::kv_u32(gctx, "kyutai.mimi.encoder_num_heads", hp.mimi_num_heads);
        hp.mimi_num_layers = core_gguf::kv_u32(gctx, "kyutai.mimi.encoder_num_layers", hp.mimi_num_layers);
        hp.mimi_context = core_gguf::kv_u32(gctx, "kyutai.mimi.encoder_context", hp.mimi_context);
        hp.codebook_dim = core_gguf::kv_u32(gctx, "kyutai.mimi.codebook_dim", hp.codebook_dim);
        hp.n_q_semantic = core_gguf::kv_u32(gctx, "kyutai.mimi.n_q_semantic", hp.n_q_semantic);
        hp.n_q_acoustic = core_gguf::kv_u32(gctx, "kyutai.mimi.n_q_acoustic", hp.n_q_acoustic);
        hp.sample_rate = core_gguf::kv_u32(gctx, "kyutai.mimi.sample_rate", hp.sample_rate);
        hp.frame_rate = core_gguf::kv_f32(gctx, "kyutai.mimi.frame_rate", hp.frame_rate);

        // Derived
        hp.head_dim = hp.dim / hp.num_heads;
        hp.mimi_head_dim = hp.mimi_dim / hp.mimi_num_heads;
        // SwiGLU FFN hidden dim: dim * hidden_scale * 2/3 rounded
        // But linear_in is [dim, 2*ffn_hidden], linear_out is [ffn_hidden, dim]
        // From weights: linear_in shape [2048, 11264], so 11264 = 2*5632
        hp.ffn_hidden = (int)(hp.dim * hp.hidden_scale * 2.0 / 3.0);
        // Round to nearest multiple of 256 for efficiency
        hp.ffn_hidden = ((hp.ffn_hidden + 255) / 256) * 256;
        // Actually, just derive from the weight shape later. Use 5632 as default.
        if (hp.dim == 2048)
            hp.ffn_hidden = 5632;

        // Tokenizer
        m.vocab.resize(hp.text_card + 1);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < (int)m.vocab.size(); i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, sctx->backend, "kyutai_stt", wl)) {
        fprintf(stderr, "kyutai_stt: failed to load weights from '%s'\n", path_model);
        delete sctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ts.find(name);
        if (it == ts.end()) {
            fprintf(stderr, "kyutai_stt: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };
    auto try_get = [&](const char* name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    // --- SEANet encoder (shortened tensor names from converter) ---
    // model.0: Conv1d(1→64, k=7, s=1)
    load_seanet_conv(ts, "mimi.encoder.model.0.conv2", m.seanet.conv_init);
    // model.1: ResnetBlock(64)
    load_seanet_conv(ts, "mimi.encoder.model.1.blk1", m.seanet.resblock[0].shortcut);
    load_seanet_conv(ts, "mimi.encoder.model.1.blk3", m.seanet.resblock[0].expand);
    // model.3: Conv1d(64→128, k=8, s=4)
    load_seanet_conv(ts, "mimi.encoder.model.3.conv2", m.seanet.conv_stride[0]);
    // model.4: ResnetBlock(128)
    load_seanet_conv(ts, "mimi.encoder.model.4.blk1", m.seanet.resblock[1].shortcut);
    load_seanet_conv(ts, "mimi.encoder.model.4.blk3", m.seanet.resblock[1].expand);
    // model.6: Conv1d(128→256, k=10, s=5)
    load_seanet_conv(ts, "mimi.encoder.model.6.conv2", m.seanet.conv_stride[1]);
    // model.7: ResnetBlock(256)
    load_seanet_conv(ts, "mimi.encoder.model.7.blk1", m.seanet.resblock[2].shortcut);
    load_seanet_conv(ts, "mimi.encoder.model.7.blk3", m.seanet.resblock[2].expand);
    // model.9: Conv1d(256→512, k=12, s=6)
    load_seanet_conv(ts, "mimi.encoder.model.9.conv2", m.seanet.conv_stride[2]);
    // model.10: ResnetBlock(512)
    load_seanet_conv(ts, "mimi.encoder.model.10.blk1", m.seanet.resblock[3].shortcut);
    load_seanet_conv(ts, "mimi.encoder.model.10.blk3", m.seanet.resblock[3].expand);
    // model.12: Conv1d(512→1024, k=16, s=8)
    load_seanet_conv(ts, "mimi.encoder.model.12.conv2", m.seanet.conv_stride[3]);
    // model.14: Conv1d(1024→512, k=3, s=1)
    load_seanet_conv(ts, "mimi.encoder.model.14.conv2", m.seanet.conv_final);

    // --- Mimi encoder transformer (shortened names: enc_tfm.) ---
    m.mimi_layers.resize(hp.mimi_num_layers);
    for (int i = 0; i < hp.mimi_num_layers; i++) {
        char buf[128];
        auto& L = m.mimi_layers[i];
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.norm1.weight", i);
        L.norm1_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.norm1.bias", i);
        L.norm1_b = try_get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.attn.qkv_w", i);
        L.attn_in_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.attn.out_w", i);
        L.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.ls1", i);
        L.layer_scale_1 = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.norm2.weight", i);
        L.norm2_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.norm2.bias", i);
        L.norm2_b = try_get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.ffn_up_w", i);
        L.linear1_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.ffn_down_w", i);
        L.linear2_w = get(buf);
        snprintf(buf, sizeof(buf), "mimi.enc_tfm.layers.%d.ls2", i);
        L.layer_scale_2 = get(buf);
    }

    // --- Downsampler (shortened: conv3. = conv.conv.conv.) ---
    load_seanet_conv(ts, "mimi.downsample.conv3", m.downsample);

    // --- RVQ ---
    // rvq_first (1 semantic codebook)
    load_seanet_conv(ts, "mimi.quantizer.rvq_first.input_proj", m.rvq_first.input_proj);
    load_seanet_conv(ts, "mimi.quantizer.rvq_first.output_proj", m.rvq_first.output_proj);
    m.rvq_first.codebooks.resize(hp.n_q_semantic);
    for (int i = 0; i < hp.n_q_semantic; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "mimi.quantizer.rvq_first.vq.layers.%d._codebook.embedding", i);
        m.rvq_first.codebooks[i].embedding = get(buf);
    }

    // rvq_rest (31 acoustic codebooks)
    load_seanet_conv(ts, "mimi.quantizer.rvq_rest.input_proj", m.rvq_rest.input_proj);
    load_seanet_conv(ts, "mimi.quantizer.rvq_rest.output_proj", m.rvq_rest.output_proj);
    m.rvq_rest.codebooks.resize(hp.n_q_acoustic);
    for (int i = 0; i < hp.n_q_acoustic; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "mimi.quantizer.rvq_rest.vq.layers.%d._codebook.embedding", i);
        m.rvq_rest.codebooks[i].embedding = get(buf);
    }

    // --- LM ---
    m.audio_emb.resize(hp.n_q);
    for (int i = 0; i < hp.n_q; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "lm.emb.%d.weight", i);
        m.audio_emb[i] = get(buf);
    }
    m.text_emb = get("lm.text_emb.weight");
    m.out_norm_alpha = get("lm.out_norm.alpha");
    m.text_linear_w = get("lm.text_linear.weight");

    m.lm_layers.resize(hp.num_layers);
    for (int i = 0; i < hp.num_layers; i++) {
        char buf[128];
        auto& L = m.lm_layers[i];
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.norm1.alpha", i);
        L.norm1_alpha = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.attn.qkv_w", i);
        L.attn_in_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.attn.out_w", i);
        L.attn_out_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.norm2.alpha", i);
        L.norm2_alpha = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.gating_in_w", i);
        L.gating_in_w = get(buf);
        snprintf(buf, sizeof(buf), "lm.transformer.layers.%d.gating_out_w", i);
        L.gating_out_w = get(buf);
    }

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {sctx->backend, nullptr};
    if (sctx->backend_cpu && sctx->backend_cpu != sctx->backend) {
        backends[n_be++] = sctx->backend_cpu;
    }
    sctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    sctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "kyutai_stt: loaded %d mimi layers + %d LM layers, vocab %d, n_q=%d\n", hp.mimi_num_layers,
                hp.num_layers, hp.text_card, hp.n_q);
        fprintf(stderr, "kyutai_stt: dim=%d, heads=%d, head_dim=%d, ffn_hidden=%d\n", hp.dim, hp.num_heads, hp.head_dim,
                hp.ffn_hidden);
    }

    return sctx;
}

extern "C" void kyutai_stt_free(struct kyutai_stt_context* ctx) {
    if (!ctx)
        return;
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// KV cache for LM autoregressive decoding
// ===========================================================================

static bool kv_cache_init(kyutai_stt_context* ctx, int max_ctx) {
    auto& hp = ctx->model.hp;
    // [head_dim, max_ctx, n_heads, n_layers] for both K and V
    // KV cache dimensions

    // Allocate context with no_alloc=true, then manual buffer
    struct ggml_init_params gp = {
        /*.mem_size   =*/ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ctx->kv_ctx = ggml_init(gp);
    if (!ctx->kv_ctx)
        return false;

    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hp.head_dim, max_ctx, hp.num_heads, hp.num_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, hp.head_dim, max_ctx, hp.num_heads, hp.num_layers);

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }

    // Zero-fill
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    return true;
}

// ===========================================================================
// Mimi encoder graph
// ===========================================================================

// Run SEANet encoder: raw PCM → [mimi_dim, T_enc]
static ggml_tensor* build_seanet_encoder(ggml_context* ctx, const seanet_encoder& se, ggml_tensor* pcm) {
    // pcm is [n_samples] → reshape to [n_samples, 1] for conv1d (data b = [T, IC])
    ggml_tensor* x = ggml_reshape_2d(ctx, pcm, ggml_nelements(pcm), 1);

    // SEANet stride schedule: [4, 5, 6, 8], kernels: [8, 10, 12, 16]
    static const int strides[] = {4, 5, 6, 8};
    static const int kernels[] = {8, 10, 12, 16};

    // model.0: Conv1d(1→64, k=7, s=1, pad=3)
    x = conv1d_fwd(ctx, se.conv_init, x, 1);

    // 4 stages: resblock → strided conv
    for (int i = 0; i < 4; i++) {
        x = resblock_fwd(ctx, se.resblock[i], x);
        x = elu(ctx, x);
        x = conv1d_fwd(ctx, se.conv_stride[i], x, strides[i]);
    }

    // model.14: Conv1d(1024→512, k=3, s=1, pad=1)
    x = elu(ctx, x);
    x = conv1d_fwd(ctx, se.conv_final, x, 1);

    // Transpose from conv layout [T, channels] to transformer layout [channels, T]
    x = ggml_cont(ctx, ggml_transpose(ctx, x));

    return x; // [mimi_dim=512, T_enc]
}

// Run Mimi encoder transformer (8 layers): [mimi_dim, T] → [mimi_dim, T]
static ggml_tensor* build_mimi_transformer(ggml_context* ctx, const std::vector<mimi_enc_layer>& layers, ggml_tensor* x,
                                           int n_heads, int head_dim) {
    int T = (int)x->ne[1];
    int dim = (int)x->ne[0]; // 512

    // Build position indices for RoPE
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    // Set positions 0..T-1 (will be filled at compute time via backend)
    // Actually, we need to set these. Use ggml_arange.
    positions = ggml_arange(ctx, 0.0f, (float)T, 1.0f);
    positions = ggml_cast(ctx, positions, GGML_TYPE_I32);

    for (size_t li = 0; li < layers.size(); li++) {
        const auto& L = layers[li];
        ggml_tensor* residual = x;

        // Pre-norm (LayerNorm with weight+bias)
        ggml_tensor* h = ggml_norm(ctx, x, 1e-5f);
        h = ggml_mul(ctx, h, L.norm1_w);
        if (L.norm1_b)
            h = ggml_add(ctx, h, L.norm1_b);

        // Self-attention with combined QKV projection
        // in_proj_weight: [dim, 3*dim]
        ggml_tensor* qkv = ggml_mul_mat(ctx, L.attn_in_w, h);
        // Split into Q, K, V — each [dim, T]
        ggml_tensor* Q = ggml_view_2d(ctx, qkv, dim, T, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(ctx, qkv, dim, T, qkv->nb[1], dim * ggml_type_size(qkv->type));
        ggml_tensor* V = ggml_view_2d(ctx, qkv, dim, T, qkv->nb[1], 2 * dim * ggml_type_size(qkv->type));

        // Reshape for multi-head: [head_dim, n_heads, T]
        // ggml_rope_ext expects a->ne[2] == positions->ne[0] (T)
        Q = ggml_reshape_3d(ctx, ggml_cont(ctx, Q), head_dim, n_heads, T);
        K = ggml_reshape_3d(ctx, ggml_cont(ctx, K), head_dim, n_heads, T);
        V = ggml_reshape_3d(ctx, ggml_cont(ctx, V), head_dim, n_heads, T);

        // RoPE (causal, max_period=10000 for encoder transformer)
        Q = ggml_rope_ext(ctx, Q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr, head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);

        // Permute to [head_dim, T, n_heads] for flash attention
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // [hd, T, nh]
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

        // Flash attention — non-causal for encoder (no mask)
        ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
        // attn is [head_dim, T, n_heads] → reshape to [dim, T]
        attn = ggml_reshape_2d(ctx, attn, dim, T);

        // Output projection
        attn = ggml_mul_mat(ctx, L.attn_out_w, attn);

        // Layer scale 1
        if (L.layer_scale_1) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.layer_scale_1, dim, 1);
            attn = ggml_mul(ctx, attn, ls);
        }

        x = ggml_add(ctx, residual, attn);

        // FFN
        residual = x;
        h = ggml_norm(ctx, x, 1e-5f);
        h = ggml_mul(ctx, h, L.norm2_w);
        if (L.norm2_b)
            h = ggml_add(ctx, h, L.norm2_b);

        // Gating FFN: linear1(dim → 4*dim) → GELU → linear2(4*dim → dim)
        // Actually moshi uses a simple linear1+gelu+linear2 FFN, NOT gating/SwiGLU
        h = ggml_mul_mat(ctx, L.linear1_w, h);
        h = ggml_gelu(ctx, h);
        h = ggml_mul_mat(ctx, L.linear2_w, h);

        // Layer scale 2
        if (L.layer_scale_2) {
            ggml_tensor* ls = ggml_reshape_2d(ctx, L.layer_scale_2, dim, 1);
            h = ggml_mul(ctx, h, ls);
        }

        x = ggml_add(ctx, residual, h);
    }

    return x; // [mimi_dim, T]
}

// Run RVQ encode on a single RVQ group.
// x: [mimi_dim, T] → integer codes [n_codebooks, T]
// This requires iterative compute (each codebook depends on residual from previous).
// Returns the codes as a flat int32 array.
static void rvq_encode_group(kyutai_stt_context* sctx, ggml_tensor* x_projected, const rvq_group& rvq, int n_codebooks,
                             std::vector<std::vector<int32_t>>& out_codes, int T) {
    auto& hp = sctx->model.hp;

    // We need to do residual quantization iteratively:
    // For each codebook:
    //   1. Find nearest codebook entry for the residual
    //   2. Subtract the quantized value from the residual
    //
    // This requires building and computing a graph for each codebook step.
    // The input x_projected is [codebook_dim, T] after the input_proj.

    // x_projected has ggml shape [T, codebook_dim]
    // Memory layout: ne[0]=T values per channel, cdim channels
    // So data[d * T + t] = frame t, dimension d
    int cdim = hp.codebook_dim;
    std::vector<float> raw(cdim * T);
    ggml_backend_tensor_get(x_projected, raw.data(), 0, cdim * T * sizeof(float));

    // Rearrange to frame-major: residual[t * cdim + d]
    std::vector<float> residual(cdim * T);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < cdim; d++) {
            residual[t * cdim + d] = raw[d * T + t];
        }
    }

    for (int q = 0; q < n_codebooks; q++) {
        // Codebook embedding: ggml shape [codebook_dim, num_codes]
        // ne[0]=codebook_dim, ne[1]=num_codes
        int cb_dim = (int)rvq.codebooks[q].embedding->ne[0];
        int num_codes = (int)rvq.codebooks[q].embedding->ne[1];

        std::vector<float> codebook_data(num_codes * cb_dim);
        ggml_backend_tensor_get(rvq.codebooks[q].embedding, codebook_data.data(), 0,
                                num_codes * cb_dim * sizeof(float));
        // codebook memory: codebook_data[code * cb_dim + d]

        std::vector<int32_t> codes(T);

        // For each frame, find nearest codebook entry
        for (int t = 0; t < T; t++) {
            float best_dist = INFINITY;
            int best_idx = 0;
            for (int c = 0; c < num_codes; c++) {
                float dist = 0;
                for (int d = 0; d < cb_dim; d++) {
                    float diff = residual[t * cdim + d] - codebook_data[c * cb_dim + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = c;
                }
            }
            codes[t] = best_idx;

            // Subtract quantized from residual
            for (int d = 0; d < cb_dim; d++) {
                residual[t * cdim + d] -= codebook_data[best_idx * cb_dim + d];
            }
        }

        out_codes.push_back(std::move(codes));
    }
}

// ===========================================================================
// Full Mimi encode pipeline
// ===========================================================================

// Encode PCM (24 kHz) → audio codes [n_q, T_frames]
static bool mimi_encode(kyutai_stt_context* sctx, const float* pcm_24k, int n_samples,
                        std::vector<std::vector<int32_t>>& codes, int& T_frames) {
    auto& m = sctx->model;
    auto& hp = m.hp;

    // Build graph for SEANet + transformer + downsample
    struct ggml_init_params gp = {
        /*.mem_size   =*/sctx->compute_meta.size(),
        /*.mem_buffer =*/sctx->compute_meta.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input PCM as a 1D tensor
    ggml_tensor* pcm = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_samples);
    ggml_set_name(pcm, "pcm_input");
    ggml_set_input(pcm);

    // SEANet encoder
    ggml_tensor* enc = build_seanet_encoder(ctx0, m.seanet, pcm);
    ggml_set_name(enc, "seanet_out");

    // enc is now [mimi_dim, T_enc] (transposed for transformer)
    // Encoder transformer expects [dim, T]
    enc = build_mimi_transformer(ctx0, m.mimi_layers, enc, hp.mimi_num_heads, hp.mimi_head_dim);
    ggml_set_name(enc, "enc_transformer_out");

    // Transpose back to [T, channels] for conv1d
    enc = ggml_cont(ctx0, ggml_transpose(ctx0, enc)); // [T_enc, mimi_dim]

    // Downsample: Conv1d(512→512, k=4, s=2)
    // Padding: (kernel_size - stride) / 2 = (4-2)/2 = 1
    enc = conv1d_fwd(ctx0, m.downsample, enc, 2);
    ggml_set_name(enc, "downsampled");
    // enc is [T_frames, mimi_dim]

    // RVQ input projection (Conv1d, kernel=1, stride=1)
    ggml_tensor* proj_first = conv1d_fwd(ctx0, m.rvq_first.input_proj, enc, 1);
    ggml_set_name(proj_first, "rvq_first_proj");
    // proj_first is [T_frames, codebook_dim]

    ggml_tensor* proj_rest = conv1d_fwd(ctx0, m.rvq_rest.input_proj, enc, 1);
    ggml_set_name(proj_rest, "rvq_rest_proj");
    // proj_rest is [T_frames, codebook_dim]

    // Mark outputs
    ggml_set_output(proj_first);
    ggml_set_output(proj_rest);

    ggml_build_forward_expand(gf, proj_first);
    ggml_build_forward_expand(gf, proj_rest);

    // Allocate and compute
    ggml_backend_sched_reset(sctx->sched);
    if (!ggml_backend_sched_alloc_graph(sctx->sched, gf)) {
        fprintf(stderr, "kyutai_stt: failed to alloc mimi encoder graph\n");
        ggml_free(ctx0);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(pcm, pcm_24k, 0, n_samples * sizeof(float));

    // Set position indices for transformer (ggml_arange handles this internally)

    if (ggml_backend_sched_graph_compute(sctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kyutai_stt: mimi encoder compute failed\n");
        ggml_free(ctx0);
        return false;
    }

    // Get T_frames from output shape: [T, codebook_dim]
    T_frames = (int)proj_first->ne[0];
    int cdim = (int)proj_first->ne[1];

    if (sctx->params.verbosity >= 2) {
        fprintf(stderr, "kyutai_stt: mimi encode: %d samples → %d frames (cdim=%d)\n", n_samples, T_frames, cdim);
    }

    // RVQ encode on CPU (iterative residual quantization)
    codes.clear();
    rvq_encode_group(sctx, proj_first, m.rvq_first, hp.n_q_semantic, codes, T_frames);
    rvq_encode_group(sctx, proj_rest, m.rvq_rest, hp.n_q_acoustic, codes, T_frames);

    ggml_free(ctx0);
    return true;
}

// ===========================================================================
// LM decoder (autoregressive)
// ===========================================================================

// Build LM forward for a single time step (autoregressive).
// input_emb: [dim, 1] — sum of all audio embeddings + text embedding for this step.
// n_past: number of past tokens in KV cache.
// Returns logits [text_card].
static ggml_tensor* build_lm_step(ggml_context* ctx, ggml_cgraph* gf, kyutai_stt_context* sctx, ggml_tensor* input_emb,
                                  int n_past) {
    auto& m = sctx->model;
    auto& hp = m.hp;

    ggml_tensor* x = input_emb; // [dim, 1]

    // Position tensor
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(positions, "lm_pos");
    ggml_set_input(positions);

    // Causal mask: not needed for single-token steps with KV cache

    for (int li = 0; li < hp.num_layers; li++) {
        const auto& L = m.lm_layers[li];
        ggml_tensor* residual = x;

        // Pre-norm (RMSNorm with alpha)
        ggml_tensor* h = rms_norm_alpha(ctx, x, L.norm1_alpha);

        // Self-attention with KV cache
        // Combined QKV: in_proj_weight is [dim, 3*dim]
        ggml_tensor* qkv = ggml_mul_mat(ctx, L.attn_in_w, h);
        // Split Q, K, V
        int dim = hp.dim;
        ggml_tensor* Q = ggml_view_2d(ctx, qkv, dim, 1, qkv->nb[1], 0);
        ggml_tensor* K_cur = ggml_view_2d(ctx, qkv, dim, 1, qkv->nb[1], dim * ggml_type_size(qkv->type));
        ggml_tensor* V_cur = ggml_view_2d(ctx, qkv, dim, 1, qkv->nb[1], 2 * dim * ggml_type_size(qkv->type));

        // Reshape for multi-head: [head_dim, n_heads, 1]
        int hd = hp.head_dim;
        int nh = hp.num_heads;
        Q = ggml_reshape_3d(ctx, ggml_cont(ctx, Q), hd, nh, 1);
        K_cur = ggml_reshape_3d(ctx, ggml_cont(ctx, K_cur), hd, nh, 1);
        V_cur = ggml_reshape_3d(ctx, ggml_cont(ctx, V_cur), hd, nh, 1);

        // RoPE
        Q = ggml_rope_ext(ctx, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, hp.max_period, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        K_cur = ggml_rope_ext(ctx, K_cur, positions, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, hp.max_period, 1.0f, 0.0f,
                              1.0f, 0.0f, 0.0f);

        // Permute to (hd, T, nh) for cache write
        ggml_tensor* K_perm = ggml_permute(ctx, K_cur, 0, 2, 1, 3); // [hd, 1, nh]
        ggml_tensor* V_perm = ggml_permute(ctx, V_cur, 0, 2, 1, 3);

        // Write K, V into cache at position n_past
        // KV cache shape: [head_dim, max_ctx, n_heads, n_layers]
        {
            ggml_tensor* k_dst =
                ggml_view_4d(ctx, sctx->kv_k, hd, 1, nh, 1, sctx->kv_k->nb[1], sctx->kv_k->nb[2], sctx->kv_k->nb[3],
                             (size_t)li * sctx->kv_k->nb[3] + (size_t)n_past * sctx->kv_k->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, K_perm, k_dst));

            ggml_tensor* v_dst =
                ggml_view_4d(ctx, sctx->kv_v, hd, 1, nh, 1, sctx->kv_v->nb[1], sctx->kv_v->nb[2], sctx->kv_v->nb[3],
                             (size_t)li * sctx->kv_v->nb[3] + (size_t)n_past * sctx->kv_v->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, V_perm, v_dst));
        }

        // Read full K, V from cache [head_dim, n_past+1, n_heads]
        int kv_len = n_past + 1;
        ggml_tensor* K_full = ggml_cont(ctx, ggml_view_3d(ctx, sctx->kv_k, hd, kv_len, nh, sctx->kv_k->nb[1],
                                                          sctx->kv_k->nb[2], (size_t)li * sctx->kv_k->nb[3]));
        ggml_tensor* V_full = ggml_cont(ctx, ggml_view_3d(ctx, sctx->kv_v, hd, kv_len, nh, sctx->kv_v->nb[1],
                                                          sctx->kv_v->nb[2], (size_t)li * sctx->kv_v->nb[3]));

        // Permute Q to (hd, 1, nh) for flash-attn
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));

        // Flash attention (causal — but with single query, mask not needed)
        ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K_full, V_full, nullptr, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        // attn: [hd, 1, nh] → [dim, 1]
        attn = ggml_reshape_2d(ctx, attn, dim, 1);

        // Output projection
        attn = ggml_mul_mat(ctx, L.attn_out_w, attn);
        x = ggml_add(ctx, residual, attn);

        // FFN
        residual = x;
        h = rms_norm_alpha(ctx, x, L.norm2_alpha);

        // SwiGLU: gating_in is [dim, 2*ffn_hidden], split into gate + up
        ggml_tensor* gate_up = ggml_mul_mat(ctx, L.gating_in_w, h);
        int ffn_h = (int)(L.gating_in_w->ne[1]) / 2;
        ggml_tensor* gate = ggml_view_2d(ctx, gate_up, ffn_h, 1, gate_up->nb[1], 0);
        ggml_tensor* up = ggml_view_2d(ctx, gate_up, ffn_h, 1, gate_up->nb[1], ffn_h * ggml_type_size(gate_up->type));
        gate = ggml_cont(ctx, gate);
        up = ggml_cont(ctx, up);
        ggml_tensor* mlp = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        mlp = ggml_mul_mat(ctx, L.gating_out_w, mlp);

        x = ggml_add(ctx, residual, mlp);
    }

    // Final RMSNorm + text projection
    x = rms_norm_alpha(ctx, x, m.out_norm_alpha);
    ggml_tensor* logits = ggml_mul_mat(ctx, m.text_linear_w, x);
    // logits: [text_card, 1]
    logits = ggml_reshape_1d(ctx, logits, hp.text_card);

    return logits;
}

// ===========================================================================
// Resample 16 kHz → 24 kHz (simple linear interpolation)
// ===========================================================================

static void resample_16k_to_24k(const float* in, int n_in, std::vector<float>& out) {
    // Ratio: 24000/16000 = 1.5
    int n_out = (int)((double)n_in * 24000.0 / 16000.0);
    out.resize(n_out);
    double ratio = (double)(n_in - 1) / (double)(n_out - 1);
    for (int i = 0; i < n_out; i++) {
        double pos = i * ratio;
        int idx = (int)pos;
        double frac = pos - idx;
        if (idx + 1 < n_in) {
            out[i] = (float)(in[idx] * (1.0 - frac) + in[idx + 1] * frac);
        } else {
            out[i] = in[n_in - 1];
        }
    }
}

// ===========================================================================
// High-level transcribe
// ===========================================================================

// Internal: transcribe and (optionally) capture per-token ids + softmax probs.
// The decoded text fragment per emitted text token (excluding pad/special) is
// appended to `result`; ids/probs are appended in lock-step when out vectors
// are non-null.
static char* kyutai_stt_transcribe_impl(struct kyutai_stt_context* ctx, const float* samples, int n_samples,
                                        std::vector<int32_t>* out_token_ids, std::vector<float>* out_token_probs,
                                        std::vector<int32_t>* out_frame_indices = nullptr) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    auto& hp = ctx->model.hp;
    auto& m = ctx->model;

    // Step 1: Resample 16 kHz → 24 kHz
    std::vector<float> pcm_24k;
    resample_16k_to_24k(samples, n_samples, pcm_24k);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kyutai_stt: resampled %d → %d samples (16k → 24k)\n", n_samples, (int)pcm_24k.size());
    }

    // Step 2: Mimi encode → audio codes
    std::vector<std::vector<int32_t>> codes;
    int T_frames = 0;
    if (!mimi_encode(ctx, pcm_24k.data(), (int)pcm_24k.size(), codes, T_frames)) {
        fprintf(stderr, "kyutai_stt: mimi encode failed\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kyutai_stt: mimi encoded %d frames, %d codebooks\n", T_frames, (int)codes.size());
    }

    // Step 3: Initialize KV cache for LM
    // Audio delay in frames
    int delay_frames = (int)(hp.audio_delay_seconds * hp.frame_rate);
    int max_ctx = T_frames + delay_frames + 64; // some extra for text tokens
    if (!kv_cache_init(ctx, max_ctx)) {
        fprintf(stderr, "kyutai_stt: KV cache init failed\n");
        return nullptr;
    }

    // Step 4: Autoregressive LM decoding
    // The LM consumes audio codes and generates text tokens.
    // At each step: sum all n_q audio embeddings + text embedding → transformer → logits
    // Start with padding text token.
    std::string result;
    // Initial token: text_card (8000) NOT padding_id (3) — this is the start-of-sequence token
    int text_token = hp.text_card;
    int n_past = 0;

    for (int t = 0; t < T_frames; t++) {
        // Build input embedding: sum of n_q audio embeddings + text embedding
        // This needs a small graph each step.

        struct ggml_init_params gp = {
            /*.mem_size   =*/ctx->compute_meta.size(),
            /*.mem_buffer =*/ctx->compute_meta.data(),
            /*.no_alloc   =*/true,
        };
        ggml_context* ctx0 = ggml_init(gp);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

        // Create input token indices
        ggml_tensor* text_tok = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(text_tok, "text_token");
        ggml_set_input(text_tok);

        // Audio token indices (one per codebook)
        std::vector<ggml_tensor*> audio_toks(hp.n_q);
        for (int q = 0; q < hp.n_q; q++) {
            char name[32];
            snprintf(name, sizeof(name), "audio_tok_%d", q);
            audio_toks[q] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            ggml_set_name(audio_toks[q], name);
            ggml_set_input(audio_toks[q]);
        }

        // Look up and sum embeddings
        ggml_tensor* emb = ggml_get_rows(ctx0, m.text_emb, text_tok);
        for (int q = 0; q < hp.n_q; q++) {
            ggml_tensor* a_emb = ggml_get_rows(ctx0, m.audio_emb[q], audio_toks[q]);
            emb = ggml_add(ctx0, emb, a_emb);
        }
        // emb: [dim, 1]

        // LM forward
        ggml_tensor* logits = build_lm_step(ctx0, gf, ctx, emb, n_past);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);

        ggml_build_forward_expand(gf, logits);

        // Allocate and compute
        ggml_backend_sched_reset(ctx->sched);

        // Tell scheduler which backend owns the KV cache tensors
        if (ctx->kv_k)
            ggml_backend_sched_set_tensor_backend(ctx->sched, ctx->kv_k, ctx->backend);
        if (ctx->kv_v)
            ggml_backend_sched_set_tensor_backend(ctx->sched, ctx->kv_v, ctx->backend);

        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "kyutai_stt: failed to alloc LM graph at step %d\n", t);
            ggml_free(ctx0);
            break;
        }

        // Set input data
        int32_t text_id = text_token;
        ggml_backend_tensor_set(text_tok, &text_id, 0, sizeof(int32_t));

        for (int q = 0; q < hp.n_q; q++) {
            // First step uses initial audio token (card=2048), rest use actual codes
            int32_t audio_id = (t == 0) ? hp.card : codes[q][t];
            ggml_backend_tensor_set(audio_toks[q], &audio_id, 0, sizeof(int32_t));
        }

        // Set position
        int32_t pos = n_past;
        ggml_tensor* pos_tensor = ggml_get_tensor(ctx0, "lm_pos");
        if (pos_tensor) {
            ggml_backend_tensor_set(pos_tensor, &pos, 0, sizeof(int32_t));
        }

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "kyutai_stt: LM compute failed at step %d\n", t);
            ggml_free(ctx0);
            break;
        }

        // Read logits and pick (greedy or sampled per ctx->params.temperature)
        std::vector<float> logits_data(hp.text_card);
        ggml_backend_tensor_get(logits, logits_data.data(), 0, hp.text_card * sizeof(float));

        float picked_prob = 0.0f;
        text_token = sample_token(logits_data.data(), hp.text_card, ctx->params.temperature, &picked_prob);
        n_past++;

        if (ctx->params.verbosity >= 2 && (t < 5 || t % 20 == 0)) {
            fprintf(stderr, "  [frame %d] text_token=%d logit=%.4f\n", t, text_token, logits_data[text_token]);
        }

        // Decode token to text
        if (text_token != 0 && text_token != hp.existing_text_padding_id && text_token < (int)m.vocab.size()) {
            std::string piece = m.vocab[text_token];
            // SentencePiece: ▁ (U+2581, 3 bytes: E2 96 81) = space
            std::string decoded;
            for (size_t ci = 0; ci < piece.size(); ci++) {
                if ((unsigned char)piece[ci] == 0xE2 && ci + 2 < piece.size() && (unsigned char)piece[ci + 1] == 0x96 &&
                    (unsigned char)piece[ci + 2] == 0x81) {
                    decoded += ' ';
                    ci += 2;
                } else {
                    decoded += piece[ci];
                }
            }
            result += decoded;
            if (out_token_ids && out_token_probs) {
                out_token_ids->push_back(text_token);
                out_token_probs->push_back(picked_prob);
            }
            if (out_frame_indices) {
                out_frame_indices->push_back(t);
            }

            if (ctx->params.verbosity >= 2) {
                fprintf(stderr, "  [%d] token=%d piece='%s'\n", t, text_token, decoded.c_str());
            }
        }

        ggml_free(ctx0);
    }

    // Clean up KV cache
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }

    // Return result
    if (result.empty())
        return nullptr;
    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    return out;
}

extern "C" char* kyutai_stt_transcribe(struct kyutai_stt_context* ctx, const float* samples, int n_samples) {
    return kyutai_stt_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr);
}

extern "C" struct kyutai_stt_result* kyutai_stt_transcribe_with_probs(struct kyutai_stt_context* ctx,
                                                                      const float* samples, int n_samples) {
    std::vector<int32_t> ids;
    std::vector<float> probs;
    char* text = kyutai_stt_transcribe_impl(ctx, samples, n_samples, &ids, &probs);
    if (!text)
        return nullptr;
    auto* r = (kyutai_stt_result*)calloc(1, sizeof(kyutai_stt_result));
    r->text = text;
    r->n_tokens = (int)ids.size();
    if (r->n_tokens > 0) {
        r->token_ids = (int*)malloc(sizeof(int) * (size_t)r->n_tokens);
        r->token_probs = (float*)malloc(sizeof(float) * (size_t)r->n_tokens);
        for (int i = 0; i < r->n_tokens; i++) {
            r->token_ids[i] = ids[i];
            r->token_probs[i] = probs[i];
        }
    }
    return r;
}

extern "C" void kyutai_stt_set_seed(struct kyutai_stt_context* ctx, unsigned int seed) {
    (void)ctx;
    if (seed != 0)
        srand(seed);
}

extern "C" void kyutai_stt_result_free(struct kyutai_stt_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->token_ids);
    free(r->token_probs);
    free(r);
}

extern "C" const char* kyutai_stt_token_text(struct kyutai_stt_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->model.vocab.size())
        return nullptr;
    return ctx->model.vocab[id].c_str();
}

// ---------------------------------------------------------------------------
// PLAN #61c — per-token + word-level timing.
//
// Each emitted text token is bound to the LM frame index that produced
// it. We subtract the training-time `audio_delay_seconds` lookahead
// (typically 0.5 s = 6.25 frames) to recover the audio time the token
// actually corresponds to. Frame duration is 8 cs (12.5 Hz Mimi rate).
// ---------------------------------------------------------------------------

namespace {
// Decode one SentencePiece vocab piece into its visible text + a flag
// for whether it starts a new word (▁ U+2581 prefix). The piece may be
// longer than the buffer — caller already truncates.
struct DecodedPiece {
    std::string text;
    bool starts_word;
};

DecodedPiece decode_piece(const std::string& piece) {
    DecodedPiece out{"", false};
    if (piece.size() >= 3 && (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81) {
        out.starts_word = true;
        out.text.assign(piece.begin() + 3, piece.end());
    } else {
        out.text = piece;
    }
    return out;
}

void copy_text_truncating(char* dst, size_t cap, const std::string& src) {
    const size_t n = std::min(src.size(), cap - 1);
    memcpy(dst, src.data(), n);
    dst[n] = '\0';
}
} // namespace

extern "C" struct kyutai_stt_result_ex* kyutai_stt_transcribe_ex(struct kyutai_stt_context* ctx, const float* samples,
                                                                 int n_samples, int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    std::vector<int32_t> ids;
    std::vector<float> probs;
    std::vector<int32_t> frame_indices;
    char* text = kyutai_stt_transcribe_impl(ctx, samples, n_samples, &ids, &probs, &frame_indices);
    if (!text)
        return nullptr;

    auto& hp = ctx->model.hp;
    const int delay_frames = (int)(hp.audio_delay_seconds * hp.frame_rate);
    // Frame duration in centiseconds: 12.5 Hz → 8 cs/frame. Round to
    // nearest cs from the float frame_rate to handle non-12.5 rates.
    const double cs_per_frame = 100.0 / hp.frame_rate;

    auto* r = (kyutai_stt_result_ex*)calloc(1, sizeof(kyutai_stt_result_ex));
    if (!r) {
        free(text);
        return nullptr;
    }
    r->text = text;
    r->n_tokens = (int)ids.size();

    if (r->n_tokens > 0) {
        r->tokens = (kyutai_stt_token_data*)calloc((size_t)r->n_tokens, sizeof(kyutai_stt_token_data));
        if (!r->tokens) {
            kyutai_stt_result_ex_free(r);
            return nullptr;
        }

        // Word grouping: open a new word at every ▁-prefixed piece.
        std::vector<kyutai_stt_word_data> words;
        std::string current_text;
        int64_t current_t0 = 0;
        int64_t current_t1 = 0;
        double current_p_sum = 0.0;
        int current_n = 0;

        auto flush_word = [&]() {
            if (current_n == 0)
                return;
            kyutai_stt_word_data w{};
            copy_text_truncating(w.text, sizeof(w.text), current_text);
            w.t0 = current_t0;
            w.t1 = current_t1;
            w.p = (float)(current_p_sum / current_n);
            words.push_back(w);
            current_text.clear();
            current_p_sum = 0.0;
            current_n = 0;
        };

        for (int i = 0; i < r->n_tokens; i++) {
            const int id = ids[i];
            const int frame_lm = frame_indices[i];
            const int frame_audio = std::max(0, frame_lm - delay_frames);
            const int64_t t0 = t_offset_cs + (int64_t)std::llround(frame_audio * cs_per_frame);
            const int64_t t1 = t_offset_cs + (int64_t)std::llround((frame_audio + 1) * cs_per_frame);

            // Per-token entry
            kyutai_stt_token_data& td = r->tokens[i];
            td.id = id;
            td.t0 = t0;
            td.t1 = t1;
            td.p = probs[i];

            std::string piece;
            if (id >= 0 && id < (int)ctx->model.vocab.size())
                piece = ctx->model.vocab[id];
            DecodedPiece dp = decode_piece(piece);
            // Token text: prefix with a space if this token starts a new word
            // (mirrors parakeet's per-token `text` field which also includes the
            // leading space for word-starting subwords).
            std::string td_text = dp.starts_word ? (" " + dp.text) : dp.text;
            copy_text_truncating(td.text, sizeof(td.text), td_text);

            // Word grouping
            if (dp.starts_word)
                flush_word();
            if (current_n == 0)
                current_t0 = t0;
            current_t1 = t1;
            current_text += dp.text;
            current_p_sum += probs[i];
            current_n++;
        }
        flush_word();

        if (!words.empty()) {
            r->n_words = (int)words.size();
            r->words = (kyutai_stt_word_data*)calloc(words.size(), sizeof(kyutai_stt_word_data));
            if (!r->words) {
                kyutai_stt_result_ex_free(r);
                return nullptr;
            }
            for (size_t i = 0; i < words.size(); i++)
                r->words[i] = words[i];
        }
    }

    return r;
}

extern "C" void kyutai_stt_result_ex_free(struct kyutai_stt_result_ex* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}
