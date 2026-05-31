// dia_tts.cpp — Nari Labs Dia 1.6B TTS runtime.
//
// Implements the Dia encoder-decoder transformer + DAC codec decode
// pipeline entirely in ggml. The architecture follows a
// text-encoder / audio-decoder pattern with CFG (Classifier-Free
// Guidance) at each decoder step.
//
// Key implementation details:
//
// 1. The encoder processes byte-level text (vocab 256) through a
//    12-layer Llama-style transformer. [S1]/[S2] tags are replaced
//    with bytes 0x01/0x02. The encoder always processes the full
//    max_encoder_context_length with an attention mask that separates
//    the conditional (text) and unconditional (padding) sequences.
//
// 2. The decoder uses GQA (16 query heads, 4 KV heads) with
//    cross-attention to the encoder output. KV caching is used for
//    both self-attention and cross-attention.
//
// 3. Multi-codebook generation uses a delay pattern [0,8,9,10,11,12,13,14,15].
//    At each step, the decoder produces logits for all 9 codebook channels.
//    After EOS on channel 0, generation continues for max_delay (15) more
//    steps to flush the delayed channels.
//
// 4. CFG is applied at each step: the batch dimension is always 2
//    (conditional + unconditional). Final logits = uncond + cfg_scale * (cond - uncond).
//
// 5. DAC codec decodes the generated codes to 44.1 kHz PCM, reusing
//    the shared core_dac implementation from dac_decoder.h.

#include "dia_tts.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include "core/dac_decoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// Model structures
// -----------------------------------------------------------------------

struct dia_encoder_layer {
    ggml_tensor* q_proj = nullptr; // (head_dim*n_heads, enc_hidden)
    ggml_tensor* k_proj = nullptr;
    ggml_tensor* v_proj = nullptr;
    ggml_tensor* o_proj = nullptr;      // (enc_hidden, head_dim*n_heads)
    ggml_tensor* pre_sa_norm = nullptr; // (enc_hidden,)
    ggml_tensor* post_sa_norm = nullptr;
    ggml_tensor* gate = nullptr; // (intermediate, enc_hidden)
    ggml_tensor* up = nullptr;
    ggml_tensor* wo = nullptr; // (enc_hidden, intermediate)
};

struct dia_decoder_layer {
    // Self-attention
    ggml_tensor* self_q_proj = nullptr; // (dec_hidden, dec_hidden)
    ggml_tensor* self_k_proj = nullptr; // (kv_dim, dec_hidden)
    ggml_tensor* self_v_proj = nullptr;
    ggml_tensor* self_o_proj = nullptr;
    ggml_tensor* pre_sa_norm = nullptr; // (dec_hidden,)

    // Cross-attention
    ggml_tensor* cross_q_proj = nullptr; // (dec_hidden, dec_hidden)
    ggml_tensor* cross_k_proj = nullptr; // (dec_hidden, enc_hidden)
    ggml_tensor* cross_v_proj = nullptr;
    ggml_tensor* cross_o_proj = nullptr;
    ggml_tensor* pre_ca_norm = nullptr;

    // MLP
    ggml_tensor* gate = nullptr;
    ggml_tensor* up = nullptr;
    ggml_tensor* wo_mlp = nullptr;
    ggml_tensor* pre_mlp_norm = nullptr;
};

struct dia_encoder {
    ggml_tensor* embedding = nullptr; // (enc_hidden, vocab_size)
    ggml_tensor* norm = nullptr;      // (enc_hidden,)
    std::vector<dia_encoder_layer> layers;
};

struct dia_decoder {
    ggml_tensor* norm = nullptr;          // (dec_hidden,)
    std::vector<ggml_tensor*> embeddings; // per-codebook (dec_hidden, vocab_size)
    std::vector<ggml_tensor*> heads;      // per-codebook (vocab_size, dec_hidden)
    std::vector<dia_decoder_layer> layers;
};

struct dia_model {
    // Architecture parameters
    uint32_t n_output_heads = 9;
    uint32_t n_encoder_layers = 12;
    uint32_t n_decoder_layers = 18;
    uint32_t encoder_hidden_size = 1024;
    uint32_t decoder_hidden_size = 2048;
    uint32_t encoder_attn_heads = 16;
    uint32_t decoder_attn_heads = 16;
    uint32_t decoder_kv_heads = 4;
    uint32_t head_dim = 128;
    uint32_t encoder_intermediate = 4096;
    uint32_t decoder_intermediate = 8192;
    uint32_t encoder_vocab_size = 256;
    uint32_t output_vocab_size = 1028;
    uint32_t audio_vocab_size = 1024;
    uint32_t max_encoder_context = 1024;
    uint32_t max_generation_size = 3072;
    uint32_t eos_token_id = 1024;
    uint32_t pad_token_id = 1025;
    uint32_t bos_token_id = 1026;
    uint32_t max_delay = 15;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;

    std::vector<uint32_t> delay_pattern = {0, 8, 9, 10, 11, 12, 13, 14, 15};

    dia_encoder encoder;
    dia_decoder decoder;
    core_dac::DacWeights dac;

    bool has_dac = false;

    // ggml context that owns the weight tensors
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
};

// -----------------------------------------------------------------------
// KV cache
// -----------------------------------------------------------------------

struct dia_kv_cache {
    // Self-attention KV cache (per decoder layer)
    std::vector<ggml_tensor*> k_l; // (head_dim * n_heads, max_gen * 2)
    std::vector<ggml_tensor*> v_l;

    // Cross-attention KV cache (per decoder layer, computed once)
    std::vector<ggml_tensor*> cross_k_l;
    std::vector<ggml_tensor*> cross_v_l;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    bool cross_cached = false;
};

// -----------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------

struct dia_tts_context {
    dia_tts_context_params params;
    dia_model model;
    dia_kv_cache kv;

    // Backends
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend = nullptr; // GPU or CPU
    ggml_backend_sched_t sched = nullptr;

    // Compute
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_buffer_t buf_output = nullptr;
    float* logits = nullptr;

    // Generation state
    uint32_t current_position = 0;
    int delay_steps = -1;   // set to max_delay when EOS seen on ch0
    size_t prompt_size = 0; // actual text length (non-padded)
    std::vector<uint32_t> output_tokens;
    std::vector<uint32_t> current_audio_tokens; // 9 tokens for current step

    // RNG
    std::mt19937 rng;
};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static uint32_t read_u32(gguf_context* meta, const char* key, uint32_t def) {
    int idx = gguf_find_key(meta, key);
    return (idx >= 0) ? gguf_get_val_u32(meta, idx) : def;
}

static float read_f32(gguf_context* meta, const char* key, float def) {
    int idx = gguf_find_key(meta, key);
    return (idx >= 0) ? gguf_get_val_f32(meta, idx) : def;
}

// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
static ggml_tensor* dia_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// -----------------------------------------------------------------------
// Tokenizer (byte-level with [S1]/[S2] replacement)
// -----------------------------------------------------------------------

static std::vector<uint32_t> dia_tokenize(const std::string& text, uint32_t max_len) {
    // Ensure text starts with a speaker tag
    std::string processed = text;
    {
        // Trim leading whitespace
        size_t start = processed.find_first_not_of(" \t\n\r");
        if (start != std::string::npos) {
            processed = processed.substr(start);
        }
    }
    if (processed.substr(0, 4) != "[S1]" && processed.substr(0, 4) != "[S2]") {
        processed = "[S1] " + processed;
    }
    // Ensure trailing period
    if (!processed.empty() && processed.back() != '.') {
        processed += ".";
    }

    // Replace [S1] -> 0x01, [S2] -> 0x02
    std::string out;
    out.reserve(processed.size());
    for (size_t i = 0; i < processed.size();) {
        if (i + 3 < processed.size() && processed[i] == '[' && processed[i + 1] == 'S' &&
            (processed[i + 2] == '1' || processed[i + 2] == '2') && processed[i + 3] == ']') {
            out.push_back(processed[i + 2] == '1' ? '\x01' : '\x02');
            i += 4;
        } else {
            out.push_back(processed[i]);
            i++;
        }
    }

    std::vector<uint32_t> tokens;
    tokens.reserve(max_len);
    for (size_t i = 0; i < out.size() && tokens.size() < max_len; i++) {
        tokens.push_back(static_cast<uint32_t>(static_cast<uint8_t>(out[i])));
    }
    return tokens;
}

// -----------------------------------------------------------------------
// Sampling
// -----------------------------------------------------------------------

static uint32_t dia_sample_token(const float* logits, uint32_t vocab_size, float temperature, float top_p, int top_k,
                                 std::mt19937& rng) {
    if (temperature <= 0.0f) {
        // Greedy
        return (uint32_t)(std::max_element(logits, logits + vocab_size) - logits);
    }

    // Apply temperature
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(logits, logits + vocab_size);
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = (logits[i] - max_logit) / temperature;
    }

    // Softmax
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(probs[i]);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // Top-k filter
    if (top_k > 0 && top_k < (int)vocab_size) {
        std::vector<std::pair<float, uint32_t>> sorted_probs(vocab_size);
        for (uint32_t i = 0; i < vocab_size; i++) {
            sorted_probs[i] = {probs[i], i};
        }
        std::partial_sort(sorted_probs.begin(), sorted_probs.begin() + top_k, sorted_probs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        float threshold = sorted_probs[top_k - 1].first;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            }
        }
        // Re-normalize
        sum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++)
            sum += probs[i];
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < vocab_size; i++)
                probs[i] /= sum;
        }
    }

    // Top-p (nucleus) filter
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, uint32_t>> sorted_probs(vocab_size);
        for (uint32_t i = 0; i < vocab_size; i++) {
            sorted_probs[i] = {probs[i], i};
        }
        std::sort(sorted_probs.begin(), sorted_probs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        float cumsum = 0.0f;
        for (auto& [p, idx] : sorted_probs) {
            cumsum += p;
            if (cumsum > top_p) {
                p = 0.0f;
            }
        }
        for (auto& [p, idx] : sorted_probs) {
            probs[idx] = p;
        }
        // Re-normalize
        sum = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++)
            sum += probs[i];
        if (sum > 0.0f) {
            for (uint32_t i = 0; i < vocab_size; i++)
                probs[i] /= sum;
        }
    }

    // Sample
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

// -----------------------------------------------------------------------
// Weight loading
// -----------------------------------------------------------------------

static void dia_load_metadata(dia_model& m, gguf_context* meta) {
    m.head_dim = read_u32(meta, "dia.attn_head_size", m.head_dim);
    m.eos_token_id = read_u32(meta, "dia.eos_token_id", m.eos_token_id);
    m.bos_token_id = read_u32(meta, "dia.bos_token_id", m.bos_token_id);
    m.pad_token_id = read_u32(meta, "dia.pad_token_id", m.pad_token_id);
    m.max_delay = read_u32(meta, "dia.max_delay", m.max_delay);
    m.rope_theta = read_f32(meta, "dia.rope_theta", m.rope_theta);
    m.rms_norm_eps = read_f32(meta, "dia.rms_norm_eps", m.rms_norm_eps);

    m.max_encoder_context = read_u32(meta, "dia.encoder.max_context_length", m.max_encoder_context);
    m.encoder_attn_heads = read_u32(meta, "dia.encoder.attn_heads", m.encoder_attn_heads);
    m.n_encoder_layers = read_u32(meta, "dia.encoder.layers", m.n_encoder_layers);

    m.decoder_hidden_size = read_u32(meta, "dia.decoder.hidden_size", m.decoder_hidden_size);
    m.n_decoder_layers = read_u32(meta, "dia.decoder.layers", m.n_decoder_layers);
    m.n_output_heads = read_u32(meta, "dia.decoder.output_heads", m.n_output_heads);
    m.decoder_attn_heads = read_u32(meta, "dia.decoder.attn_heads", m.decoder_attn_heads);
    m.decoder_kv_heads = read_u32(meta, "dia.decoder.query_heads", m.decoder_kv_heads);
    m.output_vocab_size = read_u32(meta, "dia.decoder.output_vocab_size", m.output_vocab_size);
    m.audio_vocab_size = read_u32(meta, "dia.decoder.audio_vocab_size", m.audio_vocab_size);
    m.max_generation_size = read_u32(meta, "dia.decoder.max_generation_size", m.max_generation_size);
}

static bool starts_with(const std::string& s, const char* prefix) {
    return s.compare(0, strlen(prefix), prefix) == 0;
}

static std::vector<std::string> split_dot(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == '.') {
            if (i > start)
                parts.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return parts;
}

static void dia_assign_weight(dia_model& m, const std::string& name, ggml_tensor* tensor) {
    auto parts = split_dot(name);
    if (parts.size() < 3) {
        fprintf(stderr, "dia: unknown tensor '%s'\n", name.c_str());
        return;
    }

    // DAC weights (audio_encoder.*)
    if (parts[0] == "audio_encoder") {
        m.has_dac = true;
        // TODO: map DAC weights to m.dac
        return;
    }

    if (parts[0] != "dia") {
        fprintf(stderr, "dia: unknown tensor prefix '%s'\n", name.c_str());
        return;
    }

    if (parts[1] == "encoder") {
        if (parts[2] == "embedding") {
            m.encoder.embedding = tensor;
        } else if (parts[2] == "norm") {
            m.encoder.norm = tensor;
        } else if (parts[2] == "layers" && parts.size() >= 5) {
            int idx = std::stoi(parts[3]);
            if (idx >= (int)m.encoder.layers.size()) {
                fprintf(stderr, "dia: encoder layer %d out of range\n", idx);
                return;
            }
            auto& layer = m.encoder.layers[idx];
            const auto& part = parts[4];
            if (part == "q_proj")
                layer.q_proj = tensor;
            else if (part == "k_proj")
                layer.k_proj = tensor;
            else if (part == "v_proj")
                layer.v_proj = tensor;
            else if (part == "o_proj")
                layer.o_proj = tensor;
            else if (part == "pre_sa_norm")
                layer.pre_sa_norm = tensor;
            else if (part == "post_sa_norm")
                layer.post_sa_norm = tensor;
            else if (part == "gate")
                layer.gate = tensor;
            else if (part == "up")
                layer.up = tensor;
            else if (part == "wo")
                layer.wo = tensor;
            else
                fprintf(stderr, "dia: unknown encoder layer part '%s'\n", part.c_str());
        }
    } else if (parts[1] == "decoder") {
        if (parts[2] == "norm") {
            m.decoder.norm = tensor;
        } else if (parts[2] == "embeddings" && parts.size() >= 4) {
            int idx = std::stoi(parts[3]);
            if (idx < (int)m.decoder.embeddings.size()) {
                m.decoder.embeddings[idx] = tensor;
            }
        } else if (parts[2] == "heads" && parts.size() >= 4) {
            int idx = std::stoi(parts[3]);
            if (idx < (int)m.decoder.heads.size()) {
                m.decoder.heads[idx] = tensor;
            }
        } else if (parts[2] == "layers" && parts.size() >= 5) {
            int idx = std::stoi(parts[3]);
            if (idx >= (int)m.decoder.layers.size()) {
                fprintf(stderr, "dia: decoder layer %d out of range\n", idx);
                return;
            }
            auto& layer = m.decoder.layers[idx];
            const auto& part = parts[4];
            if (part == "self_q_proj")
                layer.self_q_proj = tensor;
            else if (part == "self_k_proj")
                layer.self_k_proj = tensor;
            else if (part == "self_v_proj")
                layer.self_v_proj = tensor;
            else if (part == "self_o_proj")
                layer.self_o_proj = tensor;
            else if (part == "pre_sa_norm")
                layer.pre_sa_norm = tensor;
            else if (part == "cross_q_proj")
                layer.cross_q_proj = tensor;
            else if (part == "cross_k_proj")
                layer.cross_k_proj = tensor;
            else if (part == "cross_v_proj")
                layer.cross_v_proj = tensor;
            else if (part == "cross_o_proj")
                layer.cross_o_proj = tensor;
            else if (part == "pre_ca_norm")
                layer.pre_ca_norm = tensor;
            else if (part == "gate")
                layer.gate = tensor;
            else if (part == "up")
                layer.up = tensor;
            else if (part == "wo")
                layer.wo_mlp = tensor;
            else if (part == "pre_mlp_norm")
                layer.pre_mlp_norm = tensor;
            else
                fprintf(stderr, "dia: unknown decoder layer part '%s'\n", part.c_str());
        }
    }
}

// -----------------------------------------------------------------------
// KV cache initialization
// -----------------------------------------------------------------------

static bool dia_kv_cache_init(dia_kv_cache& cache, const dia_model& m) {
    const int n_layers = (int)m.n_decoder_layers;
    const int64_t attn_size = (int64_t)m.head_dim * m.decoder_attn_heads;

    size_t n_tensors = 4 * n_layers;
    ggml_init_params params = {
        n_tensors * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    cache.ctx = ggml_init(params);
    if (!cache.ctx)
        return false;

    cache.k_l.resize(n_layers);
    cache.v_l.resize(n_layers);
    cache.cross_k_l.resize(n_layers);
    cache.cross_v_l.resize(n_layers);

    for (int i = 0; i < n_layers; i++) {
        // Self-attention: (attn_size, max_gen) * 2 for conditional + unconditional
        cache.k_l[i] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, attn_size * m.max_generation_size * 2);
        cache.v_l[i] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, attn_size * m.max_generation_size * 2);

        // Cross-attention: (attn_size, max_enc_ctx) * 2
        cache.cross_k_l[i] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, attn_size * m.max_encoder_context * 2);
        cache.cross_v_l[i] = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F32, attn_size * m.max_encoder_context * 2);

        ggml_format_name(cache.k_l[i], "cache_k_l%d", i);
        ggml_format_name(cache.v_l[i], "cache_v_l%d", i);
        ggml_format_name(cache.cross_k_l[i], "cache_cross_k_l%d", i);
        ggml_format_name(cache.cross_v_l[i], "cache_cross_v_l%d", i);
    }

    cache.buf = ggml_backend_alloc_ctx_tensors_from_buft(cache.ctx, ggml_backend_cpu_buffer_type());
    if (!cache.buf)
        return false;
    ggml_backend_buffer_clear(cache.buf, 0);

    return true;
}

// -----------------------------------------------------------------------
// Graph building — Encoder
// -----------------------------------------------------------------------

// Forward declaration for RoPE mode
// Dia uses NeoX-style RoPE (mode=2 in ggml_rope)
static const int DIA_ROPE_MODE = 2;

static ggml_tensor* build_dia_encoder(ggml_context* ctx, dia_model& m,
                                      ggml_tensor* inp_tokens, // (max_enc_ctx * 2,) I32
                                      ggml_tensor* positions,  // (max_enc_ctx,) I32
                                      ggml_tensor* attn_mask   // (max_enc_ctx, max_enc_ctx) F32
) {
    const int T = (int)m.max_encoder_context;
    const int B = 2; // conditional + unconditional

    // Embedding lookup: (enc_hidden, T, B)
    ggml_tensor* cur =
        ggml_reshape_3d(ctx, ggml_get_rows(ctx, m.encoder.embedding, inp_tokens), m.encoder_hidden_size, T, B);

    for (auto& layer : m.encoder.layers) {
        ggml_tensor* residual = cur;

        // Pre self-attention norm
        cur = dia_rms_norm(ctx, cur, layer.pre_sa_norm, m.rms_norm_eps);

        // Self-attention
        {
            ggml_tensor* Q = ggml_mul_mat(ctx, layer.q_proj, cur);
            ggml_tensor* K = ggml_mul_mat(ctx, layer.k_proj, cur);
            ggml_tensor* V = ggml_mul_mat(ctx, layer.v_proj, cur);

            // Reshape to (head_dim, n_heads, T, B) and apply RoPE
            Q = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, Q, m.head_dim, m.encoder_attn_heads, T, B)),
                          positions, m.head_dim, DIA_ROPE_MODE);
            K = ggml_rope(ctx, ggml_cont(ctx, ggml_reshape_4d(ctx, K, m.head_dim, m.encoder_attn_heads, T, B)),
                          positions, m.head_dim, DIA_ROPE_MODE);

            // Attention: Q K^T / sqrt(d)
            ggml_tensor* q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
            ggml_tensor* k = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
            ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
            kq = ggml_soft_max_ext(ctx, kq, attn_mask, 1.0f, 0.0f);

            // V transpose and multiply
            ggml_tensor* v = ggml_cont_4d(ctx, ggml_transpose(ctx, V), T, m.head_dim, m.encoder_attn_heads, B);
            ggml_tensor* kqv = ggml_mul_mat(ctx, kq, v);
            ggml_tensor* kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);

            // Encoder attention projects to decoder_hidden_size, then O projects back
            cur = ggml_cont_3d(ctx, kqv_merged, m.decoder_hidden_size, T, B);
            cur = ggml_mul_mat(ctx, layer.o_proj, cur);
        }

        cur = ggml_add(ctx, cur, residual);
        ggml_tensor* residual_mlp = cur;

        // Pre MLP norm
        cur = dia_rms_norm(ctx, cur, layer.post_sa_norm, m.rms_norm_eps);

        // MLP: SiLU(gate(x)) * up(x), then down
        {
            cur = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, layer.gate, cur)), ggml_mul_mat(ctx, layer.up, cur));
            cur = ggml_mul_mat(ctx, layer.wo, cur);
        }

        cur = ggml_add(ctx, cur, residual_mlp);
    }

    cur = dia_rms_norm(ctx, cur, m.encoder.norm, m.rms_norm_eps);
    return cur;
}

// -----------------------------------------------------------------------
// Graph building — Decoder (single step)
// -----------------------------------------------------------------------

static ggml_tensor* build_dia_decoder_embedding(ggml_context* ctx, dia_model& m,
                                                ggml_tensor* audio_tokens // (n_output_heads * 2,) I32
) {
    const int B = 2;
    ggml_tensor* emb = nullptr;

    for (int i = 0; i < (int)m.n_output_heads; i++) {
        // Stride view: pick tokens for this codebook (interleaved cond+uncond)
        ggml_tensor* view = ggml_view_1d(ctx, audio_tokens, B, i * ggml_element_size(audio_tokens));
        view->nb[0] = m.n_output_heads * ggml_element_size(audio_tokens);

        ggml_tensor* e = ggml_get_rows(ctx, m.decoder.embeddings[i], view);
        emb = (i == 0) ? e : ggml_add(ctx, emb, e);
    }
    return emb;
}

// -----------------------------------------------------------------------
// CFG scale: logits = uncond + cfg_scale * (cond - uncond)
// -----------------------------------------------------------------------

// We apply CFG after getting per-head logits from the decoder.
// The decoder output has batch dim 2; index 0 = conditional, 1 = unconditional.
// This is done in post-processing after graph compute.

// -----------------------------------------------------------------------
// Delay pattern logic
// -----------------------------------------------------------------------

static bool dia_check_stopping(dia_tts_context& ctx) {
    auto& m = ctx.model;
    auto& tokens = ctx.current_audio_tokens;

    if (ctx.delay_steps == -1 &&
        (tokens[0] == m.eos_token_id || ctx.current_position >= m.max_generation_size - m.max_delay)) {
        ctx.delay_steps = (int)m.max_delay;
    }

    if (ctx.delay_steps > 0) {
        int step_after_eos = (int)m.max_delay - ctx.delay_steps;
        for (int i = 0; i < (int)m.delay_pattern.size(); i++) {
            if (step_after_eos == (int)m.delay_pattern[i]) {
                tokens[i] = m.eos_token_id;
            } else if (step_after_eos > (int)m.delay_pattern[i]) {
                tokens[i] = m.pad_token_id;
            }
        }
        ctx.delay_steps -= 1;
    }
    return ctx.delay_steps == 0;
}

// Revert delay pattern to recover aligned codes
static void dia_revert_delay(const std::vector<uint32_t>& raw_tokens, std::vector<uint32_t>& filtered,
                             const dia_model& m) {
    size_t n_steps = raw_tokens.size() / m.n_output_heads;
    if (n_steps <= m.max_delay)
        return;

    size_t valid_steps = n_steps - m.max_delay;
    filtered.reserve(valid_steps * m.n_output_heads);

    for (size_t t = 0; t < valid_steps; t++) {
        bool skip = false;
        for (int c = 0; c < (int)m.n_output_heads; c++) {
            size_t src_t = t + m.delay_pattern[c];
            size_t idx = src_t * m.n_output_heads + c;
            if (idx >= raw_tokens.size() || raw_tokens[idx] >= m.audio_vocab_size) {
                skip = true;
                break;
            }
        }
        if (!skip) {
            for (int c = 0; c < (int)m.n_output_heads; c++) {
                size_t src_t = t + m.delay_pattern[c];
                size_t idx = src_t * m.n_output_heads + c;
                filtered.push_back(raw_tokens[idx]);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Public C API
// -----------------------------------------------------------------------

struct dia_tts_context_params dia_tts_context_default_params(void) {
    dia_tts_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 1.2f;
    p.cfg_scale = 3.0f;
    p.top_p = 0.95f;
    p.top_k = 45;
    p.seed = 0;
    p.max_tokens = 0;
    p.flash_attn = false;
    return p;
}

struct dia_tts_context* dia_tts_init_from_file(const char* path_model, struct dia_tts_context_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new (std::nothrow) dia_tts_context();
    if (!ctx)
        return nullptr;
    ctx->params = params;

    // Initialize RNG
    if (params.seed != 0) {
        ctx->rng.seed(params.seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }

    // Open GGUF
    gguf_init_params gguf_params = {true, nullptr};
    gguf_context* meta = gguf_init_from_file(path_model, gguf_params);
    if (!meta) {
        fprintf(stderr, "dia_tts: failed to open GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    // Load metadata
    dia_load_metadata(ctx->model, meta);
    auto& m = ctx->model;

    if (params.verbosity >= 1) {
        fprintf(stderr, "dia_tts: encoder %u layers, %u heads, %u hidden\n", m.n_encoder_layers, m.encoder_attn_heads,
                m.encoder_hidden_size);
        fprintf(stderr, "dia_tts: decoder %u layers, %u heads (%u kv), %u hidden\n", m.n_decoder_layers,
                m.decoder_attn_heads, m.decoder_kv_heads, m.decoder_hidden_size);
        fprintf(stderr, "dia_tts: %u codebooks, max_gen %u, max_delay %u\n", m.n_output_heads, m.max_generation_size,
                m.max_delay);
    }

    // Allocate layer structures
    m.encoder.layers.resize(m.n_encoder_layers);
    m.decoder.layers.resize(m.n_decoder_layers);
    m.decoder.embeddings.resize(m.n_output_heads, nullptr);
    m.decoder.heads.resize(m.n_output_heads, nullptr);

    // Count tensors and create ggml context for weights
    int n_tensors = gguf_get_n_tensors(meta);
    ggml_init_params weight_params = {
        (size_t)(n_tensors + 1) * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context* ctx_data = ggml_init(weight_params);
    if (!ctx_data) {
        fprintf(stderr, "dia_tts: failed to create weight context\n");
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    // Load weights from GGUF with data
    gguf_init_params gguf_data_params = {false, &ctx_data};
    gguf_context* meta_data = gguf_init_from_file(path_model, gguf_data_params);
    if (!meta_data) {
        fprintf(stderr, "dia_tts: failed to load GGUF data\n");
        ggml_free(ctx_data);
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    // Assign weights
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(meta_data, i);
        ggml_tensor* tensor = ggml_get_tensor(ctx_data, name);
        if (tensor) {
            dia_assign_weight(m, name, tensor);
        } else if (params.verbosity >= 2) {
            fprintf(stderr, "dia_tts: tensor '%s' not found in context\n", name);
        }
    }

    m.ctx_w = ctx_data;

    // Initialize backend
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = ctx->backend_cpu; // CPU-only for now

    // Initialize KV cache
    if (!dia_kv_cache_init(ctx->kv, m)) {
        fprintf(stderr, "dia_tts: failed to init KV cache\n");
        gguf_free(meta_data);
        gguf_free(meta);
        delete ctx;
        return nullptr;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "dia_tts: model loaded from '%s'\n", path_model);
    }

    gguf_free(meta_data);
    gguf_free(meta);
    return ctx;
}

int dia_tts_set_codec_path(struct dia_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    // TODO: load DAC codec from separate GGUF
    // For now, the DAC must be embedded in the main GGUF
    (void)path;
    fprintf(stderr, "dia_tts: separate DAC codec loading not yet implemented\n");
    return -1;
}

float* dia_tts_synthesize(struct dia_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    auto& m = ctx->model;
    auto& p = ctx->params;

    // Tokenize text
    auto tokens = dia_tokenize(text, m.max_encoder_context);
    if (tokens.empty()) {
        fprintf(stderr, "dia_tts: empty text after tokenization\n");
        return nullptr;
    }

    ctx->prompt_size = tokens.size();

    if (p.verbosity >= 1) {
        fprintf(stderr, "dia_tts: text length = %zu bytes, generating...\n", tokens.size());
    }

    if (tokens.size() <= 100 && p.verbosity >= 1) {
        fprintf(stderr, "dia_tts: WARNING: prompts shorter than 100 bytes produce inconsistent results\n");
    }

    // Reset generation state
    ctx->current_position = 0;
    ctx->delay_steps = -1;
    ctx->output_tokens.clear();

    // Initialize current audio tokens with BOS
    ctx->current_audio_tokens.resize(m.n_output_heads);
    for (uint32_t i = 0; i < m.n_output_heads; i++) {
        ctx->current_audio_tokens[i] = m.bos_token_id;
    }

    uint32_t max_gen = (p.max_tokens > (int)m.max_delay) ? (uint32_t)p.max_tokens : m.max_generation_size;

    // TODO: implement ggml graph-based encoder + decoder loop
    // For now this is a skeleton that outlines the generation flow:
    //
    // 1. Run encoder on tokenized text (batch=2: cond + uncond)
    // 2. Cache cross-attention K/V from encoder output
    // 3. Loop: decoder step -> sample -> check stopping
    //    a. Build decoder input embeddings from current audio tokens
    //    b. Run decoder (self-attn with KV cache + cross-attn)
    //    c. Apply CFG: logits = uncond + cfg_scale * (cond - uncond)
    //    d. Sample per-codebook tokens
    //    e. Append to output_tokens
    //    f. Check delay-pattern stopping condition
    // 4. Revert delay pattern
    // 5. Decode via DAC codec

    fprintf(stderr, "dia_tts: encoder + decoder graph execution not yet implemented\n");
    fprintf(stderr, "dia_tts: runtime skeleton ready; implement graph building next\n");

    return nullptr;
}

void dia_tts_pcm_free(float* pcm) {
    free(pcm);
}

void dia_tts_free(struct dia_tts_context* ctx) {
    if (!ctx)
        return;

    if (ctx->kv.buf) {
        ggml_backend_buffer_free(ctx->kv.buf);
    }
    if (ctx->kv.ctx) {
        ggml_free(ctx->kv.ctx);
    }
    if (ctx->buf_output) {
        ggml_backend_buffer_free(ctx->buf_output);
    }
    if (ctx->model.ctx_w) {
        ggml_free(ctx->model.ctx_w);
    }
    if (ctx->sched) {
        ggml_backend_sched_free(ctx->sched);
    }
    if (ctx->backend_cpu) {
        ggml_backend_free(ctx->backend_cpu);
    }
    delete ctx;
}

void dia_tts_set_n_threads(struct dia_tts_context* ctx, int n_threads) {
    if (ctx)
        ctx->params.n_threads = n_threads;
}

void dia_tts_set_temperature(struct dia_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void dia_tts_set_cfg_scale(struct dia_tts_context* ctx, float cfg_scale) {
    if (ctx)
        ctx->params.cfg_scale = cfg_scale;
}

void dia_tts_set_seed(struct dia_tts_context* ctx, uint64_t seed) {
    if (ctx) {
        ctx->params.seed = seed;
        if (seed != 0) {
            ctx->rng.seed(seed);
        } else {
            std::random_device rd;
            ctx->rng.seed(rd());
        }
    }
}
