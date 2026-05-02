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

    // head_dim is explicit in config, NOT derived from hidden/heads
    // (small: enc_hidden=620, heads=8, head_dim=64 — 620/8=77.5 would be WRONG)
    hp.enc_head_dim =
        gguf_get_u32(gctx, "moonshine_streaming.encoder.attention.head_dim", hp.enc_hidden / hp.enc_n_heads);
    hp.dec_head_dim =
        gguf_get_u32(gctx, "moonshine_streaming.decoder.attention.head_dim", hp.dec_hidden / hp.dec_n_heads);

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

// Build the full encoder as a single ggml graph (gallocr pattern from moonshine.cpp).
// Build encoder as single ggml graph with gallocr (same pattern as moonshine.cpp).
static int run_encoder(moonshine_streaming_context* ctx, const float* frontend_out, int T_enc,
                       std::vector<float>& enc_output) {
    auto& m = ctx->model;
    auto& hp = m.hp;
    int d = (int)hp.enc_hidden;
    int n_heads = (int)hp.enc_n_heads;
    int head_dim = (int)hp.enc_head_dim;
    int kv_heads = (int)hp.enc_kv_heads;
    float ln_eps = 1e-5f;
    bool verbose = ctx->verbosity >= 2 || getenv("MOONSHINE_STREAMING_BENCH");

    const size_t n_tensors = hp.enc_n_layers * 30 + 50;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(16384, false);
    struct ggml_init_params gp = {mem_size, nullptr, true};
    ggml_context* ctx0 = ggml_init(gp);
    if (!ctx0)
        return -1;

    // Input
    ggml_tensor* cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_enc);
    ggml_set_name(cur, "enc_input");
    ggml_set_input(cur);

    // Per-layer masks
    std::vector<ggml_tensor*> masks(hp.enc_n_layers, nullptr);
    for (uint32_t li = 0; li < hp.enc_n_layers; li++) {
        auto [wl, wr] = hp.sliding_windows[li];
        if (wl < (uint32_t)T_enc || wr < (uint32_t)T_enc) {
            char name[32];
            snprintf(name, sizeof(name), "mask_%u", li);
            masks[li] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_enc, T_enc);
            ggml_set_name(masks[li], name);
            ggml_set_input(masks[li]);
        }
    }

    // Transformer layers
    float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t li = 0; li < hp.enc_n_layers; li++) {
        auto& L = m.enc[li];
        ggml_tensor* residual = cur;

        // Pre-norm (unit-offset baked at convert time)
        cur = ggml_norm(ctx0, cur, ln_eps);
        cur = ggml_mul(ctx0, cur, L.attn_norm_w);

        // Q/K/V
        ggml_tensor* Q = ggml_mul_mat(ctx0, L.attn_q_w, cur);
        ggml_tensor* K = ggml_mul_mat(ctx0, L.attn_k_w, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, L.attn_v_w, cur);

        // Reshape for flash_attn: [head_dim, T, heads]
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc), 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, head_dim, kv_heads, T_enc), 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, kv_heads, T_enc), 0, 2, 1, 3));

        // Sliding-window attention
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, masks[li], scale, 0.0f, 0.0f);

        // Result is [head_dim, n_heads, T] — reshape to [n_heads*head_dim, T]
        attn = ggml_reshape_2d(ctx0, attn, n_heads * head_dim, T_enc);
        cur = ggml_add(ctx0, residual, ggml_mul_mat(ctx0, L.attn_o_w, attn));

        // FFN: pre-norm + fc1+GELU + fc2 + residual
        residual = cur;
        ggml_tensor* fn = ggml_mul(ctx0, ggml_norm(ctx0, cur, ln_eps), L.ffn_norm_w);
        fn = ggml_mul_mat(ctx0, L.ffn_fc1_w, fn);
        if (L.ffn_fc1_b)
            fn = ggml_add(ctx0, fn, L.ffn_fc1_b);
        fn = ggml_gelu_erf(ctx0, fn);
        fn = ggml_mul_mat(ctx0, L.ffn_fc2_w, fn);
        if (L.ffn_fc2_b)
            fn = ggml_add(ctx0, fn, L.ffn_fc2_b);
        cur = ggml_add(ctx0, fn, residual);
    }

    // Final norm
    cur = ggml_mul(ctx0, ggml_norm(ctx0, cur, ln_eps), m.enc_output_norm_w);
    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, cur);

    // Check weight integrity before gallocr
    if (verbose) {
        float w0[4];
        ggml_backend_tensor_get(m.enc[0].attn_q_w, w0, 0, 4 * sizeof(float));
        fprintf(stderr, "  pre-gallocr: enc.0.attn.q[0..3] = [%.6f,%.6f,%.6f,%.6f]\n", w0[0], w0[1], w0[2], w0[3]);
    }

    // Allocate + set inputs
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) {
        fprintf(stderr, "moonshine_streaming: encoder alloc failed\n");
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    // Check weight integrity after gallocr
    if (verbose) {
        float w0[4];
        ggml_backend_tensor_get(m.enc[0].attn_q_w, w0, 0, 4 * sizeof(float));
        fprintf(stderr, "  post-gallocr: enc.0.attn.q[0..3] = [%.6f,%.6f,%.6f,%.6f]\n", w0[0], w0[1], w0[2], w0[3]);
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_input"), frontend_out, 0, (size_t)T_enc * d * sizeof(float));

    // Set masks
    for (uint32_t li = 0; li < hp.enc_n_layers; li++) {
        if (!masks[li])
            continue;
        auto [wl, wr] = hp.sliding_windows[li];
        size_t mask_sz = (size_t)T_enc * T_enc;
        std::vector<ggml_fp16_t> mask_data(mask_sz);
        for (int tq = 0; tq < T_enc; tq++)
            for (int tk = 0; tk < T_enc; tk++) {
                bool ok = (tk >= tq - (int)wl) && (tk <= tq + (int)wr);
                mask_data[(size_t)tq * T_enc + tk] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
            }
        char name[32];
        snprintf(name, sizeof(name), "mask_%u", li);
        ggml_tensor* mt = ggml_graph_get_tensor(gf, name);
        if (mt)
            ggml_backend_tensor_set(mt, mask_data.data(), 0, mask_sz * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moonshine_streaming: encoder compute failed\n");
        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
        return -1;
    }

    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "encoder_output");
    enc_output.resize((size_t)d * T_enc);
    ggml_backend_tensor_get(out_t, enc_output.data(), 0, (size_t)d * T_enc * sizeof(float));

    if (verbose) {
        fprintf(stderr, "moonshine_streaming: encoder %u layers done\n", hp.enc_n_layers);
        fprintf(stderr, "  enc_out[0,:8] = [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", enc_output[0], enc_output[1],
                enc_output[2], enc_output[3], enc_output[4], enc_output[5], enc_output[6], enc_output[7]);
    }

    ggml_gallocr_free(gallocr);
    ggml_free(ctx0);
    return 0;
}

// ── Decoder ─────────────────────────────────────────────────────────────────
// Autoregressive transformer decoder with cross-attention.
// TODO: implement full decoder with KV cache
// For now, placeholder showing the structure

// Internal: optional per-token capture. The legacy
// `moonshine_streaming_transcribe` calls this with nullptr out-vectors.
static char* moonshine_streaming_transcribe_impl(struct moonshine_streaming_context* ctx, const float* pcm,
                                                 int n_samples, std::vector<int32_t>* out_token_ids,
                                                 std::vector<float>* out_token_probs) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;

    const bool capture_probs = (out_token_ids && out_token_probs);
    if (capture_probs) {
        out_token_ids->clear();
        out_token_probs->clear();
    }

    auto& m = ctx->model;

    // Step 1: Audio frontend
    std::vector<float> frontend_out;
    int T_enc = 0;
    audio_frontend_cpu(pcm, n_samples, m, frontend_out, T_enc);
    if (T_enc <= 0)
        return nullptr;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "moonshine_streaming: %d samples → %d encoder frames\n", n_samples, T_enc);
        if (T_enc > 0 && ctx->verbosity >= 2) {
            int d = (int)m.hp.enc_hidden;
            fprintf(stderr, "  frontend[0,:8] = [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", frontend_out[0],
                    frontend_out[1], frontend_out[2], frontend_out[3], frontend_out[4], frontend_out[5],
                    frontend_out[6], frontend_out[7]);
        }
    }

    // Step 2: Encoder
    std::vector<float> enc_output;
    if (run_encoder(ctx, frontend_out.data(), T_enc, enc_output) != 0) {
        return nullptr;
    }

    if (ctx->verbosity >= 2) {
        fprintf(stderr, "  enc_out[0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", enc_output[0], enc_output[1],
                enc_output[2], enc_output[3], enc_output[4], enc_output[5], enc_output[6], enc_output[7]);
    }

    // Step 3: Decoder — adapted from moonshine.cpp's decoder pattern.
    // Uses ggml tensor KV caches with gallocr per step.
    auto& hp = m.hp;
    int enc_h = (int)hp.enc_hidden;
    int dec_h = (int)hp.dec_hidden;
    int dec_layers = (int)hp.dec_n_layers;
    int dec_heads = (int)hp.dec_n_heads;
    int dec_kv_heads = (int)hp.dec_kv_heads;
    int dec_head_dim = (int)hp.dec_head_dim;
    int dec_inter = (int)hp.dec_intermediate;
    int vocab = (int)hp.vocab_size;
    int rotary_dim = (int)(dec_head_dim * hp.partial_rotary_factor);
    float rope_theta = hp.rope_theta;
    float ln_eps = 1e-5f;
    float scale = 1.0f / sqrtf((float)dec_head_dim);
    int max_tokens = std::min((int)ceil(n_samples / 16000.0 * 6.5), 194);

    // ── Allocate KV caches as ggml tensors ──────────────────────────────
    int kv_max = max_tokens + 2;
    size_t kv_tensors = dec_layers * 4 + 4; // self K/V + cross K/V per layer
    size_t kv_mem = ggml_tensor_overhead() * kv_tensors + 256;
    struct ggml_init_params kv_gp = {kv_mem, nullptr, true};
    ggml_context* kv_ctx = ggml_init(kv_gp);
    if (!kv_ctx)
        return nullptr;

    struct kv_layer {
        ggml_tensor* self_k;
        ggml_tensor* self_v;
        ggml_tensor* cross_k;
        ggml_tensor* cross_v;
    };
    std::vector<kv_layer> kv(dec_layers);
    for (int i = 0; i < dec_layers; i++) {
        kv[i].self_k = ggml_new_tensor_3d(kv_ctx, GGML_TYPE_F32, dec_head_dim, kv_max, dec_kv_heads);
        kv[i].self_v = ggml_new_tensor_3d(kv_ctx, GGML_TYPE_F32, dec_head_dim, kv_max, dec_kv_heads);
        kv[i].cross_k = ggml_new_tensor_3d(kv_ctx, GGML_TYPE_F32, dec_head_dim, T_enc, dec_kv_heads);
        kv[i].cross_v = ggml_new_tensor_3d(kv_ctx, GGML_TYPE_F32, dec_head_dim, T_enc, dec_kv_heads);
    }
    ggml_backend_buffer_t kv_buf = ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx, ggml_backend_cpu_buffer_type());
    if (!kv_buf) {
        ggml_free(kv_ctx);
        return nullptr;
    }
    ggml_backend_buffer_clear(kv_buf, 0);

    // ── Precompute cross-attention K/V ──────────────────────────────────
    // For moonshine-streaming: add learned pos_emb to encoder output before K/V projection
    {
        size_t xkv_n = dec_layers * 10 + 20;
        size_t xkv_mem = ggml_tensor_overhead() * xkv_n + ggml_graph_overhead();
        struct ggml_init_params xkv_gp = {xkv_mem, nullptr, true};
        ggml_context* xctx = ggml_init(xkv_gp);
        if (!xctx) {
            ggml_backend_buffer_free(kv_buf);
            ggml_free(kv_ctx);
            return nullptr;
        }

        // Encoder output [enc_h, T_enc]
        ggml_tensor* enc_inp = ggml_new_tensor_2d(xctx, GGML_TYPE_F32, enc_h, T_enc);
        ggml_set_name(enc_inp, "xkv_enc");
        ggml_set_input(enc_inp);

        ggml_cgraph* xgf = ggml_new_graph(xctx);
        std::vector<ggml_tensor*> k_outs(dec_layers), v_outs(dec_layers);

        // Add learned positional embedding to encoder output before projection
        ggml_tensor* enc_pos = enc_inp;
        if (m.dec_pos_emb_w) {
            // pos_emb is [max_pos, enc_h] — slice to [T_enc, enc_h] = [enc_h, T_enc] in ggml
            ggml_tensor* pos_ids = ggml_new_tensor_1d(xctx, GGML_TYPE_I32, T_enc);
            ggml_set_name(pos_ids, "pos_ids");
            ggml_set_input(pos_ids);
            ggml_tensor* pos_emb = ggml_get_rows(xctx, m.dec_pos_emb_w, pos_ids);
            enc_pos = ggml_add(xctx, enc_inp, pos_emb);
        }

        // Optional encoder→decoder projection
        if (m.dec_enc_proj_w) {
            enc_pos = ggml_mul_mat(xctx, m.dec_enc_proj_w, enc_pos);
        }

        for (int i = 0; i < dec_layers; i++) {
            auto& L = m.dec[i];
            ggml_tensor* K = ggml_mul_mat(xctx, L.cross_attn_k_w, enc_pos);
            K = ggml_reshape_3d(xctx, K, dec_head_dim, dec_kv_heads, T_enc);
            K = ggml_cont(xctx, ggml_permute(xctx, K, 0, 2, 1, 3));
            char nk[32];
            snprintf(nk, sizeof(nk), "xk_%d", i);
            ggml_set_name(K, nk);
            ggml_set_output(K);
            k_outs[i] = K;
            ggml_build_forward_expand(xgf, K);

            ggml_tensor* V = ggml_mul_mat(xctx, L.cross_attn_v_w, enc_pos);
            V = ggml_reshape_3d(xctx, V, dec_head_dim, dec_kv_heads, T_enc);
            V = ggml_cont(xctx, ggml_permute(xctx, V, 0, 2, 1, 3));
            char nv[32];
            snprintf(nv, sizeof(nv), "xv_%d", i);
            ggml_set_name(V, nv);
            ggml_set_output(V);
            v_outs[i] = V;
            ggml_build_forward_expand(xgf, V);
        }

        ggml_gallocr_t xalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(xalloc, xgf)) {
            ggml_gallocr_free(xalloc);
            ggml_free(xctx);
            ggml_backend_buffer_free(kv_buf);
            ggml_free(kv_ctx);
            return nullptr;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(xgf, "xkv_enc"), enc_output.data(), 0,
                                (size_t)enc_h * T_enc * sizeof(float));
        if (m.dec_pos_emb_w) {
            std::vector<int32_t> pos_data(T_enc);
            for (int i = 0; i < T_enc; i++)
                pos_data[i] = i;
            ggml_tensor* pt = ggml_graph_get_tensor(xgf, "pos_ids");
            if (pt)
                ggml_backend_tensor_set(pt, pos_data.data(), 0, T_enc * sizeof(int32_t));
        }

        ggml_backend_graph_compute(ctx->backend, xgf);

        size_t kv_bytes = (size_t)dec_head_dim * T_enc * dec_kv_heads * sizeof(float);
        for (int i = 0; i < dec_layers; i++) {
            ggml_backend_tensor_get(k_outs[i], kv[i].cross_k->data, 0, kv_bytes);
            ggml_backend_tensor_get(v_outs[i], kv[i].cross_v->data, 0, kv_bytes);
        }
        ggml_gallocr_free(xalloc);
        ggml_free(xctx);
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "moonshine_streaming: cross-KV precomputed (%d layers)\n", dec_layers);

    // ── Greedy decode loop ──────────────────────────────────────────────
    ggml_gallocr_t dec_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    int32_t cur_token = (int32_t)hp.bos_token_id;
    std::vector<int32_t> tokens;
    int kv_pos = 0;

    for (int step = 0; step < max_tokens; step++) {
        size_t dn = dec_layers * 60 + 50;
        size_t dmem = ggml_tensor_overhead() * dn + ggml_graph_overhead();
        struct ggml_init_params dgp = {dmem, nullptr, true};
        ggml_context* dctx = ggml_init(dgp);
        if (!dctx)
            break;

        ggml_tensor* tok_inp = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, 1);
        ggml_set_name(tok_inp, "tok");
        ggml_set_input(tok_inp);
        ggml_tensor* pos_inp = ggml_new_tensor_1d(dctx, GGML_TYPE_I32, 1);
        ggml_set_name(pos_inp, "pos");
        ggml_set_input(pos_inp);

        ggml_cgraph* dgf = ggml_new_graph(dctx);

        // Token embedding + cast to F32
        ggml_tensor* cur = ggml_get_rows(dctx, m.dec_embed_w, tok_inp);
        if (cur->type != GGML_TYPE_F32)
            cur = ggml_cast(dctx, cur, GGML_TYPE_F32);

        for (int li = 0; li < dec_layers; li++) {
            auto& L = m.dec[li];

            // === Self-attention ===
            ggml_tensor* res = cur;
            cur = ggml_mul(dctx, ggml_norm(dctx, cur, ln_eps), L.attn_norm_w);

            ggml_tensor* Q = ggml_mul_mat(dctx, L.attn_q_w, cur);
            ggml_tensor* Kn = ggml_mul_mat(dctx, L.attn_k_w, cur);
            ggml_tensor* Vn = ggml_mul_mat(dctx, L.attn_v_w, cur);

            Q = ggml_reshape_3d(dctx, Q, dec_head_dim, dec_heads, 1);
            Kn = ggml_reshape_3d(dctx, Kn, dec_head_dim, dec_kv_heads, 1);
            Vn = ggml_reshape_3d(dctx, Vn, dec_head_dim, dec_kv_heads, 1);

            // Partial RoPE
            Q = ggml_rope_ext(dctx, Q, pos_inp, nullptr, rotary_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kn = ggml_rope_ext(dctx, Kn, pos_inp, nullptr, rotary_dim, 0, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Write to self-KV cache
            Kn = ggml_permute(dctx, Kn, 0, 2, 1, 3);
            Vn = ggml_permute(dctx, Vn, 0, 2, 1, 3);
            ggml_tensor* ks = ggml_view_3d(dctx, kv[li].self_k, dec_head_dim, 1, dec_kv_heads, kv[li].self_k->nb[1],
                                           kv[li].self_k->nb[2], kv_pos * kv[li].self_k->nb[1]);
            ggml_tensor* vs = ggml_view_3d(dctx, kv[li].self_v, dec_head_dim, 1, dec_kv_heads, kv[li].self_v->nb[1],
                                           kv[li].self_v->nb[2], kv_pos * kv[li].self_v->nb[1]);
            ggml_build_forward_expand(dgf, ggml_cpy(dctx, Kn, ks));
            ggml_build_forward_expand(dgf, ggml_cpy(dctx, Vn, vs));

            int kvl = kv_pos + 1;
            ggml_tensor* Kc = ggml_view_3d(dctx, kv[li].self_k, dec_head_dim, kvl, dec_kv_heads, kv[li].self_k->nb[1],
                                           kv[li].self_k->nb[2], 0);
            ggml_tensor* Vc = ggml_view_3d(dctx, kv[li].self_v, dec_head_dim, kvl, dec_kv_heads, kv[li].self_v->nb[1],
                                           kv[li].self_v->nb[2], 0);
            Q = ggml_permute(dctx, Q, 0, 2, 1, 3);
            ggml_tensor* attn = ggml_flash_attn_ext(dctx, Q, Kc, Vc, nullptr, scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(dctx, attn, dec_heads * dec_head_dim, 1);
            cur = ggml_add(dctx, ggml_mul_mat(dctx, L.attn_o_w, attn), res);

            // === Cross-attention ===
            res = cur;
            cur = ggml_mul(dctx, ggml_norm(dctx, cur, ln_eps), L.cross_attn_norm_w);
            ggml_tensor* Qx = ggml_mul_mat(dctx, L.cross_attn_q_w, cur);
            Qx = ggml_reshape_3d(dctx, Qx, dec_head_dim, dec_heads, 1);
            Qx = ggml_permute(dctx, Qx, 0, 2, 1, 3);
            ggml_tensor* Kcx = ggml_view_3d(dctx, kv[li].cross_k, dec_head_dim, T_enc, dec_kv_heads,
                                            kv[li].cross_k->nb[1], kv[li].cross_k->nb[2], 0);
            ggml_tensor* Vcx = ggml_view_3d(dctx, kv[li].cross_v, dec_head_dim, T_enc, dec_kv_heads,
                                            kv[li].cross_v->nb[1], kv[li].cross_v->nb[2], 0);
            ggml_tensor* xa = ggml_flash_attn_ext(dctx, Qx, Kcx, Vcx, nullptr, scale, 0.0f, 0.0f);
            xa = ggml_reshape_2d(dctx, xa, dec_heads * dec_head_dim, 1);
            cur = ggml_add(dctx, ggml_mul_mat(dctx, L.cross_attn_o_w, xa), res);

            // === SiLU-gated FFN ===
            res = cur;
            cur = ggml_mul(dctx, ggml_norm(dctx, cur, ln_eps), L.ffn_norm_w);
            ggml_tensor* fc1 = ggml_mul_mat(dctx, L.ffn_fc1_w, cur);
            if (L.ffn_fc1_b)
                fc1 = ggml_add(dctx, fc1, L.ffn_fc1_b);
            // Split: first half = value, second half = gate
            ggml_tensor* val = ggml_view_2d(dctx, fc1, dec_inter, 1, fc1->nb[1], 0);
            ggml_tensor* gate = ggml_view_2d(dctx, fc1, dec_inter, 1, fc1->nb[1], dec_inter * sizeof(float));
            cur = ggml_mul(dctx, ggml_silu(dctx, gate), val);
            cur = ggml_mul_mat(dctx, L.ffn_fc2_w, cur);
            if (L.ffn_fc2_b)
                cur = ggml_add(dctx, cur, L.ffn_fc2_b);
            cur = ggml_add(dctx, cur, res);
        }

        // Final norm + logits
        cur = ggml_mul(dctx, ggml_norm(dctx, cur, ln_eps), m.dec_output_norm_w);
        ggml_tensor* logits = ggml_mul_mat(dctx, m.dec_output_w, cur);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(dgf, logits);

        if (!ggml_gallocr_alloc_graph(dec_alloc, dgf)) {
            ggml_free(dctx);
            break;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(dgf, "tok"), &cur_token, 0, sizeof(int32_t));
        int32_t pos_val = kv_pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(dgf, "pos"), &pos_val, 0, sizeof(int32_t));

        if (ggml_backend_graph_compute(ctx->backend, dgf) != GGML_STATUS_SUCCESS) {
            ggml_free(dctx);
            break;
        }

        std::vector<float> logits_data(vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(dgf, "logits"), logits_data.data(), 0, vocab * sizeof(float));
        ggml_free(dctx);

        // Greedy argmax + optional softmax probability of the picked token.
        int best = 0;
        float bv = logits_data[0];
        for (int i = 1; i < vocab; i++)
            if (logits_data[i] > bv) {
                bv = logits_data[i];
                best = i;
            }

        if (best == (int)hp.eos_token_id)
            break;
        tokens.push_back(best);
        if (capture_probs) {
            float s = 0.f;
            for (int i = 0; i < vocab; i++)
                s += expf(logits_data[i] - bv);
            out_token_ids->push_back(best);
            out_token_probs->push_back((s > 0.f) ? (1.0f / s) : 0.0f);
        }
        cur_token = best;
        kv_pos++;

        if (ctx->verbosity >= 2 && step < 3) {
            fprintf(stderr, "  dec step %d: token=%d\n", step, best);
        }
    }
    ggml_gallocr_free(dec_alloc);

    if (ctx->verbosity >= 1)
        fprintf(stderr, "moonshine_streaming: decoder produced %d tokens\n", (int)tokens.size());

    // ── Detokenize ──────────────────────────────────────────────────────
    std::string result = ctx->tokenizer.tokens_to_text(tokens);

    ggml_backend_buffer_free(kv_buf);
    ggml_free(kv_ctx);

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

extern "C" char* moonshine_streaming_transcribe(struct moonshine_streaming_context* ctx, const float* pcm,
                                                int n_samples) {
    return moonshine_streaming_transcribe_impl(ctx, pcm, n_samples, nullptr, nullptr);
}

extern "C" struct moonshine_streaming_result* moonshine_streaming_transcribe_with_probs(
    struct moonshine_streaming_context* ctx, const float* pcm, int n_samples) {
    std::vector<int32_t> ids;
    std::vector<float> probs;
    char* text = moonshine_streaming_transcribe_impl(ctx, pcm, n_samples, &ids, &probs);
    if (!text)
        return nullptr;
    auto* r = (moonshine_streaming_result*)calloc(1, sizeof(moonshine_streaming_result));
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

extern "C" void moonshine_streaming_result_free(struct moonshine_streaming_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->token_ids);
    free(r->token_probs);
    free(r);
}

extern "C" const char* moonshine_streaming_token_text(struct moonshine_streaming_context* ctx, int id) {
    if (!ctx)
        return "";
    // moonshine-streaming uses the same shared moonshine tokenizer as moonshine.
    // Single-id detokenize via token_to_piece (returns empty for special tokens).
    static thread_local std::string scratch;
    scratch = ctx->tokenizer.token_to_piece(id);
    return scratch.c_str();
}

// ---------------------------------------------------------------------------
// Streaming API (PLAN #62c follow-on) — chunked-batch over a rolling window.
// Same shape as kyutai_stt_stream_*; see src/kyutai_stt.cpp for the
// design rationale.
// ---------------------------------------------------------------------------

struct moonshine_streaming_stream {
    moonshine_streaming_context* ctx; // not owned
    int step_samples_16k;
    int length_samples_16k;

    std::vector<float> rolling;
    int64_t total_fed_samples;
    int samples_since_last_decode;

    std::string out_text;
    double out_t0_s;
    double out_t1_s;
    bool has_output;
    int64_t decode_counter;
};

static int moonshine_streaming_stream_run_decode(moonshine_streaming_stream* s) {
    if (s->rolling.empty())
        return 0;
    char* text = moonshine_streaming_transcribe(s->ctx, s->rolling.data(), (int)s->rolling.size());
    s->out_text = text ? text : "";
    free(text);
    const int64_t window_start_samples = s->total_fed_samples - (int64_t)s->rolling.size();
    s->out_t0_s      = (double)window_start_samples / 16000.0;
    s->out_t1_s      = (double)s->total_fed_samples / 16000.0;
    s->has_output    = true;
    s->decode_counter += 1;
    return 0;
}

extern "C" struct moonshine_streaming_stream* moonshine_streaming_stream_open(struct moonshine_streaming_context* ctx,
                                                                              int step_ms, int length_ms) {
    if (!ctx)
        return nullptr;
    auto* s = new moonshine_streaming_stream();
    s->ctx                       = ctx;
    s->step_samples_16k          = (step_ms > 0 ? step_ms : 3000) * 16;
    s->length_samples_16k        = (length_ms > 0 ? length_ms : 10000) * 16;
    s->total_fed_samples         = 0;
    s->samples_since_last_decode = 0;
    s->out_t0_s                  = 0.0;
    s->out_t1_s                  = 0.0;
    s->has_output                = false;
    s->decode_counter            = 0;
    return s;
}

extern "C" int moonshine_streaming_stream_feed(struct moonshine_streaming_stream* s, const float* pcm, int n_samples) {
    if (!s || !pcm || n_samples <= 0)
        return -1;
    s->rolling.insert(s->rolling.end(), pcm, pcm + n_samples);
    if ((int)s->rolling.size() > s->length_samples_16k) {
        const int drop = (int)s->rolling.size() - s->length_samples_16k;
        s->rolling.erase(s->rolling.begin(), s->rolling.begin() + drop);
    }
    s->total_fed_samples += n_samples;
    s->samples_since_last_decode += n_samples;
    if (s->samples_since_last_decode < s->step_samples_16k) {
        return 0;
    }
    s->samples_since_last_decode = 0;
    if (moonshine_streaming_stream_run_decode(s) != 0)
        return -2;
    return 1;
}

extern "C" int moonshine_streaming_stream_get_text(struct moonshine_streaming_stream* s, char* out, int cap,
                                                   double* t0_s, double* t1_s, int64_t* counter) {
    if (!s || !out || cap <= 0)
        return -1;
    if (!s->has_output) {
        out[0] = '\0';
        if (t0_s)
            *t0_s = 0.0;
        if (t1_s)
            *t1_s = 0.0;
        if (counter)
            *counter = 0;
        return 0;
    }
    const size_t n = std::min((size_t)(cap - 1), s->out_text.size());
    memcpy(out, s->out_text.data(), n);
    out[n] = '\0';
    if (t0_s)
        *t0_s = s->out_t0_s;
    if (t1_s)
        *t1_s = s->out_t1_s;
    if (counter)
        *counter = s->decode_counter;
    return (int)n;
}

extern "C" int moonshine_streaming_stream_flush(struct moonshine_streaming_stream* s) {
    if (!s)
        return -1;
    if (s->rolling.empty())
        return 0;
    s->samples_since_last_decode = 0;
    return moonshine_streaming_stream_run_decode(s) == 0 ? 1 : -2;
}

extern "C" void moonshine_streaming_stream_close(struct moonshine_streaming_stream* s) {
    if (!s)
        return;
    delete s;
}
