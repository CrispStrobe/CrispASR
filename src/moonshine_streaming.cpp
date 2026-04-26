// moonshine_streaming.cpp — CrispASR runtime for Moonshine Streaming models
//
// Architecture: raw-waveform audio frontend → sliding-window transformer encoder
// → autoregressive transformer decoder with cross-attention.
//
// Key differences from regular moonshine:
//  - Audio frontend: raw waveform frames (no mel), CMVN, asinh compression,
//    Linear+SiLU, two CausalConv1d with stride-2
//  - Encoder: sliding-window attention (per-layer windows), unit-offset LayerNorm
//    (baked into weights at convert time), no positional embeddings
//  - Decoder: SiLU-gated MLP, learned positional embedding for cross-attention
//  - Encoder/decoder may have different hidden sizes (small/medium)

#include "moonshine_streaming.h"
#include "moonshine-tokenizer.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Hyperparameters ─────────────────────────────────────────────────────────

struct ms_hparams {
    uint32_t enc_hidden = 0;
    uint32_t dec_hidden = 0;
    uint32_t enc_n_layers = 0;
    uint32_t dec_n_layers = 0;
    uint32_t enc_n_heads = 0;
    uint32_t dec_n_heads = 0;
    uint32_t enc_kv_heads = 0;
    uint32_t dec_kv_heads = 0;
    uint32_t enc_intermediate = 0;
    uint32_t dec_intermediate = 0;
    uint32_t vocab_size = 0;
    uint32_t bos_token_id = 1;
    uint32_t eos_token_id = 2;
    uint32_t max_positions = 4096;
    uint32_t enc_head_dim = 0;
    uint32_t dec_head_dim = 0;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 0.8f;
    uint32_t sample_rate = 16000;
    float frame_ms = 5.0f;
    uint32_t frame_size = 80; // samples per frame (frame_ms * sample_rate / 1000)
    // Per-layer sliding window: [left, right] for each encoder layer
    std::vector<std::pair<uint32_t, uint32_t>> sliding_windows;
};

// ── Model tensors ───────────────────────────────────────────────────────────

struct ms_enc_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_fc1_w = nullptr;
    ggml_tensor* ffn_fc1_b = nullptr;
    ggml_tensor* ffn_fc2_w = nullptr;
    ggml_tensor* ffn_fc2_b = nullptr;
};

struct ms_dec_layer {
    // Self-attention
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    // Cross-attention
    ggml_tensor* cross_attn_norm_w = nullptr;
    ggml_tensor* cross_attn_q_w = nullptr;
    ggml_tensor* cross_attn_k_w = nullptr;
    ggml_tensor* cross_attn_v_w = nullptr;
    ggml_tensor* cross_attn_o_w = nullptr;
    // FFN
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_fc1_w = nullptr; // [2*intermediate, hidden] for SiLU-gated
    ggml_tensor* ffn_fc1_b = nullptr;
    ggml_tensor* ffn_fc2_w = nullptr;
    ggml_tensor* ffn_fc2_b = nullptr;
};

struct ms_model {
    ms_hparams hp;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Audio frontend
    ggml_tensor* embedder_log_k = nullptr;
    ggml_tensor* embedder_linear_w = nullptr;
    ggml_tensor* embedder_conv1_w = nullptr;
    ggml_tensor* embedder_conv1_b = nullptr;
    ggml_tensor* embedder_conv2_w = nullptr;
    ggml_tensor* embedder_conv2_b = nullptr;

    // Encoder
    std::vector<ms_enc_layer> enc;
    ggml_tensor* enc_output_norm_w = nullptr;

    // Decoder
    ggml_tensor* dec_embed_w = nullptr;    // [vocab, dec_hidden]
    ggml_tensor* dec_pos_emb_w = nullptr;  // [max_pos, enc_hidden] — added to encoder output before cross-attn
    ggml_tensor* dec_enc_proj_w = nullptr; // [dec_hidden, enc_hidden] — optional, when enc != dec hidden
    std::vector<ms_dec_layer> dec;
    ggml_tensor* dec_output_norm_w = nullptr;
    ggml_tensor* dec_output_w = nullptr; // [vocab, dec_hidden]
};

// ── KV cache ────────────────────────────────────────────────────────────────

struct ms_kv_cache {
    std::vector<float> k; // [head_dim * n_kv_heads * max_len]
    std::vector<float> v;
    int cur_len = 0;
    int max_len = 0;
    int head_dim = 0;
    int n_kv_heads = 0;

    void alloc(int hd, int nkv, int maxl) {
        head_dim = hd;
        n_kv_heads = nkv;
        max_len = maxl;
        cur_len = 0;
        k.assign((size_t)hd * nkv * maxl, 0.0f);
        v.assign((size_t)hd * nkv * maxl, 0.0f);
    }
};

// ── Context ─────────────────────────────────────────────────────────────────

struct moonshine_streaming_context {
    ms_model model;
    moonshine_tokenizer tokenizer;
    ggml_backend_t backend = nullptr;

    // Per-layer KV caches for decoder
    std::vector<ms_kv_cache> kv_self;
    std::vector<ms_kv_cache> kv_cross;

    int n_threads = 4;
    int verbosity = 1;
    float temperature = 0.0f;
};

// ── GGUF helpers ────────────────────────────────────────────────────────────

static uint32_t gguf_get_u32(gguf_context* ctx, const char* key, uint32_t def = 0) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}

static float gguf_get_f32(gguf_context* ctx, const char* key, float def = 0.0f) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

static std::string dir_of(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

// ── Public API ──────────────────────────────────────────────────────────────

extern "C" struct moonshine_streaming_context_params moonshine_streaming_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/false, /*temperature=*/0.0f};
}

extern "C" struct moonshine_streaming_context* moonshine_streaming_init_from_file(
    const char* path_model, struct moonshine_streaming_context_params params) {
    auto* ctx = new moonshine_streaming_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->temperature = params.temperature;

    auto& m = ctx->model;
    auto& hp = m.hp;

    // ── Pass 1: read GGUF metadata ──────────────────────────────────────
    struct gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&m.ctx_w};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "moonshine_streaming: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    hp.enc_hidden = gguf_get_u32(gctx, "moonshine_streaming.encoder.embedding_length");
    hp.dec_hidden = gguf_get_u32(gctx, "moonshine_streaming.decoder.embedding_length");
    hp.enc_n_layers = gguf_get_u32(gctx, "moonshine_streaming.encoder.block_count");
    hp.dec_n_layers = gguf_get_u32(gctx, "moonshine_streaming.decoder.block_count");
    hp.enc_n_heads = gguf_get_u32(gctx, "moonshine_streaming.encoder.attention.head_count");
    hp.dec_n_heads = gguf_get_u32(gctx, "moonshine_streaming.decoder.attention.head_count");
    hp.enc_kv_heads = gguf_get_u32(gctx, "moonshine_streaming.encoder.attention.head_count_kv", hp.enc_n_heads);
    hp.dec_kv_heads = gguf_get_u32(gctx, "moonshine_streaming.decoder.attention.head_count_kv", hp.dec_n_heads);
    hp.enc_intermediate = gguf_get_u32(gctx, "moonshine_streaming.encoder.feed_forward_length");
    hp.dec_intermediate = gguf_get_u32(gctx, "moonshine_streaming.decoder.feed_forward_length");
    hp.vocab_size = gguf_get_u32(gctx, "moonshine_streaming.vocab_size");
    hp.bos_token_id = gguf_get_u32(gctx, "moonshine_streaming.bos_token_id", 1);
    hp.eos_token_id = gguf_get_u32(gctx, "moonshine_streaming.eos_token_id", 2);
    hp.max_positions = gguf_get_u32(gctx, "moonshine_streaming.max_position_embeddings", 4096);
    hp.rope_theta = gguf_get_f32(gctx, "moonshine_streaming.rope.freq_base", 10000.0f);
    hp.partial_rotary_factor = gguf_get_f32(gctx, "moonshine_streaming.decoder.partial_rotary_factor", 0.8f);
    hp.sample_rate = gguf_get_u32(gctx, "moonshine_streaming.audio.sample_rate", 16000);
    hp.frame_ms = gguf_get_f32(gctx, "moonshine_streaming.audio.frame_ms", 5.0f);
    hp.frame_size = (uint32_t)(hp.frame_ms * hp.sample_rate / 1000.0f);

    hp.enc_head_dim = hp.enc_hidden / hp.enc_n_heads;
    hp.dec_head_dim = hp.dec_hidden / hp.dec_n_heads;

    // Read per-layer sliding windows
    hp.sliding_windows.resize(hp.enc_n_layers);
    for (uint32_t i = 0; i < hp.enc_n_layers; i++) {
        char key[128];
        snprintf(key, sizeof(key), "moonshine_streaming.encoder.layers.%u.window_left", i);
        hp.sliding_windows[i].first = gguf_get_u32(gctx, key, 16);
        snprintf(key, sizeof(key), "moonshine_streaming.encoder.layers.%u.window_right", i);
        hp.sliding_windows[i].second = gguf_get_u32(gctx, key, 4);
    }

    gguf_free(gctx);

    if (hp.enc_hidden == 0 || hp.dec_hidden == 0 || hp.vocab_size == 0) {
        fprintf(stderr, "moonshine_streaming: invalid model metadata\n");
        delete ctx;
        return nullptr;
    }

    // ── Allocate weight buffer ─────────────────────────────────��────────
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "moonshine_streaming: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    m.buf_w = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_w, ggml_backend_cpu_buffer_type());
    if (!m.buf_w) {
        fprintf(stderr, "moonshine_streaming: failed to allocate tensor buffer\n");
        delete ctx;
        return nullptr;
    }

    // Load tensor data from GGUF
    {
        FILE* f = fopen(path_model, "rb");
        if (!f) {
            fprintf(stderr, "moonshine_streaming: failed to open '%s' for reading\n", path_model);
            delete ctx;
            return nullptr;
        }
        struct gguf_init_params gp2 = {/*.no_alloc=*/false, /*.ctx=*/nullptr};
        gguf_context* gctx2 = gguf_init_from_file(path_model, gp2);
        // No ctx_w this time — just need tensor data offsets

        // Copy tensor data by name
        for (struct ggml_tensor* t = ggml_get_first_tensor(m.ctx_w); t; t = ggml_get_next_tensor(m.ctx_w, t)) {
            const char* name = ggml_get_name(t);
            int tidx = gguf_find_tensor(gctx2, name);
            if (tidx < 0) {
                fprintf(stderr, "moonshine_streaming: tensor '%s' not found in GGUF\n", name);
                continue;
            }
            size_t offset = gguf_get_data_offset(gctx2) + gguf_get_tensor_offset(gctx2, tidx);
            fseek(f, (long)offset, SEEK_SET);
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> buf(nbytes);
            if (fread(buf.data(), 1, nbytes, f) != nbytes) {
                fprintf(stderr, "moonshine_streaming: short read for tensor '%s'\n", name);
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        }
        fclose(f);
        gguf_free(gctx2);
    }

    // ── Bind tensor pointers ────────────────────────────────────────────
    auto get = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(m.ctx_w, name); };

    // Audio frontend
    m.embedder_log_k = get("encoder.embedder.log_k");
    m.embedder_linear_w = get("encoder.embedder.linear.weight");
    m.embedder_conv1_w = get("encoder.embedder.conv1.weight");
    m.embedder_conv1_b = get("encoder.embedder.conv1.bias");
    m.embedder_conv2_w = get("encoder.embedder.conv2.weight");
    m.embedder_conv2_b = get("encoder.embedder.conv2.bias");

    // Encoder
    m.enc.resize(hp.enc_n_layers);
    for (uint32_t i = 0; i < hp.enc_n_layers; i++) {
        char buf[128];
        auto& L = m.enc[i];
        snprintf(buf, sizeof(buf), "encoder.layers.%u.attn_norm.weight", i);
        L.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.attn.q.weight", i);
        L.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.attn.k.weight", i);
        L.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.attn.v.weight", i);
        L.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.attn.o.weight", i);
        L.attn_o_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.ffn_norm.weight", i);
        L.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.ffn.fc1.weight", i);
        L.ffn_fc1_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.ffn.fc1.bias", i);
        L.ffn_fc1_b = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.ffn.fc2.weight", i);
        L.ffn_fc2_w = get(buf);
        snprintf(buf, sizeof(buf), "encoder.layers.%u.ffn.fc2.bias", i);
        L.ffn_fc2_b = get(buf);
    }
    m.enc_output_norm_w = get("encoder.output_norm.weight");

    // Decoder
    m.dec_embed_w = get("decoder.embed_tokens.weight");
    m.dec_pos_emb_w = get("decoder.pos_emb.weight");
    m.dec_enc_proj_w = get("decoder.enc_proj.weight"); // nullptr for tiny (same hidden)
    m.dec.resize(hp.dec_n_layers);
    for (uint32_t i = 0; i < hp.dec_n_layers; i++) {
        char buf[128];
        auto& L = m.dec[i];
        snprintf(buf, sizeof(buf), "decoder.layers.%u.attn_norm.weight", i);
        L.attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.attn.q.weight", i);
        L.attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.attn.k.weight", i);
        L.attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.attn.v.weight", i);
        L.attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.attn.o.weight", i);
        L.attn_o_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.cross_attn_norm.weight", i);
        L.cross_attn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.cross_attn.q.weight", i);
        L.cross_attn_q_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.cross_attn.k.weight", i);
        L.cross_attn_k_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.cross_attn.v.weight", i);
        L.cross_attn_v_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.cross_attn.o.weight", i);
        L.cross_attn_o_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.ffn_norm.weight", i);
        L.ffn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.ffn.fc1.weight", i);
        L.ffn_fc1_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.ffn.fc1.bias", i);
        L.ffn_fc1_b = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.ffn.fc2.weight", i);
        L.ffn_fc2_w = get(buf);
        snprintf(buf, sizeof(buf), "decoder.layers.%u.ffn.fc2.bias", i);
        L.ffn_fc2_b = get(buf);
    }
    m.dec_output_norm_w = get("decoder.output_norm.weight");
    m.dec_output_w = get("decoder.output.weight");

    // ── Load tokenizer ─────────────────────────────────────��────────────
    std::string tok_path = dir_of(path_model) + "/tokenizer.bin";
    if (!ctx->tokenizer.load(tok_path.c_str())) {
        fprintf(stderr, "moonshine_streaming: failed to load tokenizer from '%s'\n", tok_path.c_str());
        delete ctx;
        return nullptr;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "moonshine_streaming: enc=%uL×%u dec=%uL×%u vocab=%u (enc_h=%u dec_h=%u)\n", hp.enc_n_layers,
                hp.enc_hidden, hp.dec_n_layers, hp.dec_hidden, hp.vocab_size, hp.enc_hidden, hp.dec_hidden);
    }

    return ctx;
}

// ── Audio frontend ──────────────────────────────────────────────────────────
// Raw waveform → 80-sample frames → CMVN → asinh(exp(log_k)*x) →
// Linear(80→hidden)+SiLU → CausalConv1d(hidden→2*hidden,k=5,s=2)+SiLU →
// CausalConv1d(2*hidden→hidden,k=5,s=2) → [hidden, T_enc]

static void audio_frontend_cpu(const float* pcm, int n_samples, const ms_model& m, std::vector<float>& out,
                               int& T_out) {
    const auto& hp = m.hp;
    int frame_size = (int)hp.frame_size; // 80
    int n_frames = n_samples / frame_size;
    if (n_frames < 1) {
        T_out = 0;
        return;
    }

    int enc_h = (int)hp.enc_hidden;

    // Read log_k scalar
    float log_k = 0.0f;
    ggml_backend_tensor_get(m.embedder_log_k, &log_k, 0, sizeof(float));
    float k_val = expf(log_k);

    // Step 1: Frame + CMVN + asinh compression
    // CMVN: per-frame center + RMS normalize
    std::vector<float> frames(n_frames * frame_size);
    for (int t = 0; t < n_frames; t++) {
        const float* src = pcm + t * frame_size;
        float mean = 0.0f;
        for (int i = 0; i < frame_size; i++)
            mean += src[i];
        mean /= frame_size;
        float var = 0.0f;
        for (int i = 0; i < frame_size; i++) {
            float d = src[i] - mean;
            var += d * d;
        }
        float rms = sqrtf(var / frame_size + 1e-8f);
        for (int i = 0; i < frame_size; i++) {
            float x = (src[i] - mean) / rms;
            // asinh(k * x) = log(k*x + sqrt((k*x)^2 + 1))
            float kx = k_val * x;
            frames[t * frame_size + i] = logf(kx + sqrtf(kx * kx + 1.0f));
        }
    }

    // Step 2: Linear(80 → enc_hidden) + SiLU
    // linear_w: [enc_hidden, 80] in ggml = [80, enc_hidden] row-major
    std::vector<float> linear_w(enc_h * frame_size);
    ggml_backend_tensor_get(m.embedder_linear_w, linear_w.data(), 0, enc_h * frame_size * sizeof(float));

    std::vector<float> linear_out(n_frames * enc_h);
    for (int t = 0; t < n_frames; t++) {
        for (int o = 0; o < enc_h; o++) {
            float sum = 0.0f;
            for (int i = 0; i < frame_size; i++) {
                sum += linear_w[o * frame_size + i] * frames[t * frame_size + i];
            }
            // SiLU: x * sigmoid(x)
            float s = sum / (1.0f + expf(-sum));
            linear_out[t * enc_h + o] = s;
        }
    }

    // Step 3: CausalConv1d(enc_hidden → 2*enc_hidden, k=5, s=2) + SiLU
    int conv1_oc = 2 * enc_h;
    int conv1_ic = enc_h;
    int conv1_k = 5;
    int conv1_s = 2;
    int conv1_pad = conv1_k - 1; // causal: left-pad only

    std::vector<float> conv1_w(conv1_oc * conv1_ic * conv1_k);
    std::vector<float> conv1_b(conv1_oc);
    ggml_backend_tensor_get(m.embedder_conv1_w, conv1_w.data(), 0, conv1_w.size() * sizeof(float));
    ggml_backend_tensor_get(m.embedder_conv1_b, conv1_b.data(), 0, conv1_b.size() * sizeof(float));

    // Padded input: [conv1_pad + n_frames, conv1_ic]
    int padded_len1 = conv1_pad + n_frames;
    std::vector<float> padded1(padded_len1 * conv1_ic, 0.0f);
    memcpy(padded1.data() + conv1_pad * conv1_ic, linear_out.data(), n_frames * conv1_ic * sizeof(float));

    int T_conv1 = (padded_len1 - conv1_k) / conv1_s + 1;
    std::vector<float> conv1_out(T_conv1 * conv1_oc);
    for (int t = 0; t < T_conv1; t++) {
        int t_in = t * conv1_s;
        for (int oc = 0; oc < conv1_oc; oc++) {
            float sum = conv1_b[oc];
            for (int ic = 0; ic < conv1_ic; ic++) {
                for (int kk = 0; kk < conv1_k; kk++) {
                    // PyTorch conv1d weight: [OC, IC, K]
                    sum += conv1_w[oc * conv1_ic * conv1_k + ic * conv1_k + kk] * padded1[(t_in + kk) * conv1_ic + ic];
                }
            }
            float s = sum / (1.0f + expf(-sum)); // SiLU
            conv1_out[t * conv1_oc + oc] = s;
        }
    }

    // Step 4: CausalConv1d(2*enc_hidden → enc_hidden, k=5, s=2) — no activation
    int conv2_oc = enc_h;
    int conv2_ic = conv1_oc;
    int conv2_k = 5;
    int conv2_s = 2;
    int conv2_pad = conv2_k - 1;

    std::vector<float> conv2_w(conv2_oc * conv2_ic * conv2_k);
    std::vector<float> conv2_b(conv2_oc);
    ggml_backend_tensor_get(m.embedder_conv2_w, conv2_w.data(), 0, conv2_w.size() * sizeof(float));
    ggml_backend_tensor_get(m.embedder_conv2_b, conv2_b.data(), 0, conv2_b.size() * sizeof(float));

    int padded_len2 = conv2_pad + T_conv1;
    std::vector<float> padded2(padded_len2 * conv2_ic, 0.0f);
    memcpy(padded2.data() + conv2_pad * conv2_ic, conv1_out.data(), T_conv1 * conv2_ic * sizeof(float));

    T_out = (padded_len2 - conv2_k) / conv2_s + 1;
    out.resize(T_out * conv2_oc);
    for (int t = 0; t < T_out; t++) {
        int t_in = t * conv2_s;
        for (int oc = 0; oc < conv2_oc; oc++) {
            float sum = conv2_b[oc];
            for (int ic = 0; ic < conv2_ic; ic++) {
                for (int kk = 0; kk < conv2_k; kk++) {
                    sum += conv2_w[oc * conv2_ic * conv2_k + ic * conv2_k + kk] * padded2[(t_in + kk) * conv2_ic + ic];
                }
            }
            out[t * conv2_oc + oc] = sum; // no activation on conv2
        }
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────
// Sliding-window transformer encoder. Uses ggml graphs with
// ggml_graph_compute_with_ctx for CPU execution (model is small).

// TODO: implement full encoder with sliding-window attention
// For now, placeholder that builds one graph per layer
static void encode(moonshine_streaming_context* ctx, const float* frontend_out, int T_enc,
                   std::vector<float>& enc_output) {
    auto& m = ctx->model;
    auto& hp = m.hp;
    int d = (int)hp.enc_hidden;
    int n_heads = (int)hp.enc_n_heads;
    int head_dim = (int)hp.enc_head_dim;
    int kv_heads = (int)hp.enc_kv_heads;
    float ln_eps = 1e-5f;

    // Working buffer: x [d, T] in ggml = [T, d] row-major
    std::vector<float> x(T_enc * d);
    memcpy(x.data(), frontend_out, T_enc * d * sizeof(float));

    for (uint32_t li = 0; li < hp.enc_n_layers; li++) {
        auto& L = m.enc[li];
        auto [win_left, win_right] = hp.sliding_windows[li];

        // Build ggml graph for this layer
        size_t mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(4096, false);
        struct ggml_init_params gp = {mem, nullptr, true};
        ggml_context* ctx0 = ggml_init(gp);
        if (!ctx0)
            break;
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

        // Input: [d, T]
        ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_enc);
        ggml_set_name(inp, "x_in");
        ggml_set_input(inp);

        // Pre-norm (unit-offset already baked into weight)
        ggml_tensor* xn = ggml_norm(ctx0, inp, ln_eps);
        xn = ggml_mul(ctx0, xn, L.attn_norm_w);

        // Q/K/V projections
        ggml_tensor* Q = ggml_mul_mat(ctx0, L.attn_q_w, xn); // [n_heads*head_dim, T]
        ggml_tensor* K = ggml_mul_mat(ctx0, L.attn_k_w, xn); // [kv_heads*head_dim, T]
        ggml_tensor* V = ggml_mul_mat(ctx0, L.attn_v_w, xn); // [kv_heads*head_dim, T]

        // Reshape for flash_attn: [head_dim, T, n_heads]
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3); // [head_dim, T, n_heads]
        Q = ggml_cont(ctx0, Q);
        K = ggml_reshape_3d(ctx0, K, head_dim, kv_heads, T_enc);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        K = ggml_cont(ctx0, K);
        V = ggml_reshape_3d(ctx0, V, head_dim, kv_heads, T_enc);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        V = ggml_cont(ctx0, V);

        // Sliding-window attention mask
        ggml_tensor* mask = nullptr;
        if (win_left < (uint32_t)T_enc || win_right < (uint32_t)T_enc) {
            // Build mask: [T, T] where allowed positions = 0, blocked = -inf
            mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_enc, T_enc);
            ggml_set_name(mask, "sw_mask");
            ggml_set_input(mask);
        }

        float scale = 1.0f / sqrtf((float)head_dim);
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, mask, scale, 0.0f, 0.0f);

        // Reshape back: [head_dim, T, n_heads] → [d, T]
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3); // [head_dim, n_heads, T]
        attn = ggml_cont(ctx0, attn);
        attn = ggml_reshape_2d(ctx0, attn, d, T_enc);

        // Output projection + residual
        ggml_tensor* out = ggml_mul_mat(ctx0, L.attn_o_w, attn);
        out = ggml_add(ctx0, inp, out);

        // FFN: pre-norm → fc1+GELU → fc2 + residual
        ggml_tensor* fn = ggml_norm(ctx0, out, ln_eps);
        fn = ggml_mul(ctx0, fn, L.ffn_norm_w);
        fn = ggml_mul_mat(ctx0, L.ffn_fc1_w, fn);
        if (L.ffn_fc1_b)
            fn = ggml_add(ctx0, fn, L.ffn_fc1_b);
        fn = ggml_gelu(ctx0, fn);
        fn = ggml_mul_mat(ctx0, L.ffn_fc2_w, fn);
        if (L.ffn_fc2_b)
            fn = ggml_add(ctx0, fn, L.ffn_fc2_b);
        out = ggml_add(ctx0, out, fn);

        ggml_set_name(out, "layer_out");
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        // Compute
        ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

        // Set input data
        ggml_backend_tensor_set(inp, x.data(), 0, T_enc * d * sizeof(float));

        // Build sliding window mask if needed
        if (mask) {
            std::vector<float> mask_data(T_enc * T_enc);
            for (int tq = 0; tq < T_enc; tq++) {
                for (int tk = 0; tk < T_enc; tk++) {
                    // Allow if tk >= tq - win_left AND tk <= tq + win_right
                    bool allowed = (tk >= tq - (int)win_left) && (tk <= tq + (int)win_right);
                    mask_data[tq * T_enc + tk] = allowed ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
        }

        // Re-compute with actual data
        ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

        // Read output
        ggml_tensor* out_t = ggml_graph_get_tensor(gf, "layer_out");
        ggml_backend_tensor_get(out_t, x.data(), 0, T_enc * d * sizeof(float));
        ggml_free(ctx0);
    }

    // Final layer norm
    {
        std::vector<float> norm_w(d);
        ggml_backend_tensor_get(m.enc_output_norm_w, norm_w.data(), 0, d * sizeof(float));
        for (int t = 0; t < T_enc; t++) {
            float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < d; i++)
                mean += x[t * d + i];
            mean /= d;
            for (int i = 0; i < d; i++) {
                float v = x[t * d + i] - mean;
                var += v * v;
            }
            float inv_std = 1.0f / sqrtf(var / d + 1e-5f);
            for (int i = 0; i < d; i++)
                x[t * d + i] = (x[t * d + i] - mean) * inv_std * norm_w[i];
        }
    }

    enc_output = std::move(x);
}

// ── Decoder ─────────────────────────────────────────────────────────────────
// Autoregressive transformer decoder with cross-attention.
// TODO: implement full decoder with KV cache
// For now, placeholder showing the structure

extern "C" char* moonshine_streaming_transcribe(struct moonshine_streaming_context* ctx, const float* pcm,
                                                int n_samples) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;

    auto& m = ctx->model;

    // Step 1: Audio frontend
    std::vector<float> frontend_out;
    int T_enc = 0;
    audio_frontend_cpu(pcm, n_samples, m, frontend_out, T_enc);
    if (T_enc <= 0)
        return nullptr;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "moonshine_streaming: %d samples → %d encoder frames\n", n_samples, T_enc);
    }

    // Step 2: Encoder
    std::vector<float> enc_output;
    encode(ctx, frontend_out.data(), T_enc, enc_output);

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "  enc_out[0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", enc_output[0], enc_output[1],
                enc_output[2], enc_output[3], enc_output[4], enc_output[5], enc_output[6], enc_output[7]);
    }

    // Step 3: Decoder (greedy, autoregressive)
    // TODO: implement full decoder with cross-attention KV cache
    // For now, return placeholder
    std::string result = "[moonshine-streaming decoder not yet implemented]";

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" void moonshine_streaming_free(struct moonshine_streaming_context* ctx) {
    if (!ctx)
        return;
    if (ctx->model.buf_w)
        ggml_backend_buffer_free(ctx->model.buf_w);
    if (ctx->model.ctx_w)
        ggml_free(ctx->model.ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" void moonshine_streaming_set_n_threads(struct moonshine_streaming_context* ctx, int n_threads) {
    if (ctx && n_threads > 0) {
        ctx->n_threads = n_threads;
        if (ctx->backend)
            ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
}
