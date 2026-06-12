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
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/activation.h"
#include "core/conv.h"
#include "core/dac_decoder.h"
#include "core/gguf_loader.h"

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

    // DAC codec weight context (separate GGUF)
    ggml_context* ctx_dac = nullptr;
    ggml_backend_buffer_t buf_dac = nullptr;
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;
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
    // Ensure trailing period (only if text doesn't end with punctuation)
    if (!processed.empty() && processed.back() != '.' && processed.back() != '?' && processed.back() != '!') {
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
static const int DIA_ROPE_MODE = 2; // first-half/second-half pairing (ggml NeoX mode)

static ggml_tensor* build_dia_encoder(ggml_context* ctx, dia_model& m,
                                      ggml_tensor* inp_tokens, // (max_enc_ctx * 2,) I32
                                      ggml_tensor* positions,  // (max_enc_ctx,) I32
                                      ggml_tensor* attn_mask   // (max_enc_ctx, max_enc_ctx) F32
) {
    const int B = 2;                            // conditional + unconditional
    const int T = (int)(inp_tokens->ne[0] / B); // actual sequence length

    // Embedding lookup: (enc_hidden, T, B)
    ggml_tensor* cur =
        ggml_reshape_3d(ctx, ggml_get_rows(ctx, m.encoder.embedding, inp_tokens), m.encoder_hidden_size, T, B);

    int layer_idx = 0;
    for (auto& layer : m.encoder.layers) {
        if (!layer.q_proj || !layer.k_proj || !layer.v_proj || !layer.o_proj) {
            fprintf(stderr, "dia: encoder layer %d has NULL attention weight!\n", layer_idx);
            return nullptr;
        }
        if (!layer.gate || !layer.up || !layer.wo) {
            fprintf(stderr, "dia: encoder layer %d has NULL MLP weight!\n", layer_idx);
            return nullptr;
        }
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

            // V: (2048, T, B) -> reshape to (head_dim, n_heads, T, B)
            ggml_tensor* v_4d = ggml_reshape_4d(ctx, V, m.head_dim, m.encoder_attn_heads, T, B);
            // Permute to (head_dim, T, n_heads, B) matching Q/K layout
            ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_4d, 0, 2, 1, 3));
            // Transpose to (T, head_dim, n_heads, B) for mul_mat contraction on T_key
            ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
            // attn_out = V^T @ softmax_weights: contracts on T_key dim
            // kq=(T_key, T_query, n_heads, B), v_t=(T_key, hd, n_heads, B)
            ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq); // (hd, T_query, n_heads, B)
            // Permute to (hd, n_heads, T, B) for correct head concatenation
            ggml_tensor* kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));

            // Reshape to (hd * n_heads = 2048, T, B), then output projection
            cur = ggml_reshape_3d(ctx, kqv_merged, m.head_dim * m.encoder_attn_heads, T, B);
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
        layer_idx++;
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

// -----------------------------------------------------------------------
// DAC codec loading
// -----------------------------------------------------------------------

static bool dia_load_dac_weights(dia_model& m, const std::map<std::string, ggml_tensor*>& t, int verbosity) {
    const char* tag = "dia_dac";
    const int n_cb = m.dac.config.n_codebooks;

    // Quantizer weights
    m.dac.quantizers.resize(n_cb);
    for (int k = 0; k < n_cb; k++) {
        auto& q = m.dac.quantizers[k];
        char name[128];
        std::snprintf(name, sizeof(name), "quantizer.quantizers.%d.codebook.weight", k);
        q.codebook = core_gguf::require(t, name, tag);

        std::snprintf(name, sizeof(name), "quantizer.quantizers.%d.out_proj.weight", k);
        q.out_proj_w = core_gguf::require(t, name, tag);

        std::snprintf(name, sizeof(name), "quantizer.quantizers.%d.out_proj.bias", k);
        q.out_proj_b = core_gguf::require(t, name, tag);

        if (!q.codebook || !q.out_proj_w || !q.out_proj_b) {
            fprintf(stderr, "dia_dac: missing quantizer %d weights\n", k);
            return false;
        }
    }

    // Decoder input conv: decoder.conv1.weight (7, 1024, 1536), decoder.conv1.bias (1536)
    m.dac.in_conv_w = core_gguf::require(t, "decoder.conv1.weight", tag);
    m.dac.in_conv_b = core_gguf::require(t, "decoder.conv1.bias", tag);
    if (!m.dac.in_conv_w || !m.dac.in_conv_b) {
        return false;
    }

    // Decoder blocks (4 blocks)
    for (int b = 0; b < 4; b++) {
        auto& blk = m.dac.blocks[b];
        char name[128];

        // Block snake alpha: decoder.block.B.snake1.alpha
        std::snprintf(name, sizeof(name), "decoder.block.%d.snake1.alpha", b);
        blk.snake_alpha = core_gguf::require(t, name, tag);

        // ConvTranspose1d: decoder.block.B.conv_t1.weight/bias
        std::snprintf(name, sizeof(name), "decoder.block.%d.conv_t1.weight", b);
        blk.up_w = core_gguf::require(t, name, tag);
        std::snprintf(name, sizeof(name), "decoder.block.%d.conv_t1.bias", b);
        blk.up_b = core_gguf::require(t, name, tag);

        if (!blk.snake_alpha || !blk.up_w || !blk.up_b) {
            return false;
        }

        // 3 ResidualUnits
        for (int r = 0; r < 3; r++) {
            auto& ru = blk.res[r];

            std::snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.snake1.alpha", b, r + 1);
            ru.alpha0 = core_gguf::require(t, name, tag);
            std::snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv1.weight", b, r + 1);
            ru.conv0_w = core_gguf::require(t, name, tag);
            std::snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv1.bias", b, r + 1);
            ru.conv0_b = core_gguf::require(t, name, tag);

            std::snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.snake2.alpha", b, r + 1);
            ru.alpha1 = core_gguf::require(t, name, tag);
            std::snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv2.weight", b, r + 1);
            ru.conv1_w = core_gguf::require(t, name, tag);
            std::snprintf(name, sizeof(name), "decoder.block.%d.res_unit%d.conv2.bias", b, r + 1);
            ru.conv1_b = core_gguf::require(t, name, tag);

            if (!ru.alpha0 || !ru.conv0_w || !ru.conv0_b || !ru.alpha1 || !ru.conv1_w || !ru.conv1_b) {
                fprintf(stderr, "dia_dac: missing res_unit%d in block %d\n", r + 1, b);
                return false;
            }
        }
    }

    // Output snake + conv: decoder.snake1.alpha, decoder.conv2.weight/bias
    m.dac.out_snake_alpha = core_gguf::require(t, "decoder.snake1.alpha", tag);
    m.dac.out_conv_w = core_gguf::require(t, "decoder.conv2.weight", tag);
    m.dac.out_conv_b = core_gguf::require(t, "decoder.conv2.bias", tag);
    if (!m.dac.out_snake_alpha || !m.dac.out_conv_w || !m.dac.out_conv_b) {
        return false;
    }

    if (verbosity >= 1) {
        fprintf(stderr, "dia_dac: loaded %d codebooks, 4 decoder blocks\n", n_cb);
    }
    return true;
}

int dia_tts_set_codec_path(struct dia_tts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    auto& m = ctx->model;
    int verbosity = ctx->params.verbosity;

    if (m.has_dac) {
        fprintf(stderr, "dia_tts: DAC codec already loaded\n");
        return 0;
    }

    // Load DAC GGUF using core_gguf loader
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = ctx->backend ? ctx->backend : ctx->backend_cpu;
    if (!core_gguf::load_weights(path, backend, "dia_dac", wl)) {
        fprintf(stderr, "dia_tts: failed to load DAC codec from '%s'\n", path);
        return -1;
    }

    // Map tensors to DacWeights struct
    if (!dia_load_dac_weights(m, wl.tensors, verbosity)) {
        fprintf(stderr, "dia_tts: failed to map DAC codec tensors\n");
        if (wl.buf)
            ggml_backend_buffer_free(wl.buf);
        if (wl.ctx)
            ggml_free(wl.ctx);
        return -1;
    }

    m.ctx_dac = wl.ctx;
    m.buf_dac = wl.buf;
    m.has_dac = true;

    // Permute ConvTranspose1d weights for decomposed path
    {
        const int n = 4; // DAC has 4 decoder blocks
        ggml_tensor* srcs[4] = {m.dac.blocks[0].up_w, m.dac.blocks[1].up_w, m.dac.blocks[2].up_w, m.dac.blocks[3].up_w};
        ggml_tensor** dsts[4] = {&m.dac.blocks[0].up_w_perm, &m.dac.blocks[1].up_w_perm, &m.dac.blocks[2].up_w_perm,
                                 &m.dac.blocks[3].up_w_perm};
        core_convt::permute_convt1d_weights_batch(srcs, dsts, n, backend, &m.ctx_perm, &m.buf_perm);
    }

    if (verbosity >= 1) {
        fprintf(stderr, "dia_tts: DAC codec loaded from '%s'\n", path);
    }
    return 0;
}

// -----------------------------------------------------------------------
// DAC decode graph helpers
// -----------------------------------------------------------------------

// Standard Conv1d k=K, groups=1, dilation=d.
// Weight ne=[K, Cin, Cout]. Input (Cin, T) -> output (Cout, T).
static ggml_tensor* dac_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int pad, int dil) {
    const int Cout = (int)w->ne[2];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));          // (T, Cin)
    ggml_tensor* y = ggml_conv_1d(ctx, w, xT, /*stride*/ 1, pad, dil); // (T_out, Cout, 1)
    const int T_out = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, T_out, Cout);   // (T_out, Cout)
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

// DAC ResidualUnit: Snake1d -> Conv1d(k=7, dil=d) -> Snake1d -> Conv1d(k=1) + residual
static ggml_tensor* dac_residual_unit(ggml_context* ctx, ggml_tensor* x, const core_dac::DacResUnit& ru, int dil) {
    ggml_tensor* y = core_act::snake_alpha(ctx, x, ru.alpha0);
    y = dac_conv1d(ctx, y, ru.conv0_w, ru.conv0_b, /*pad*/ 3 * dil, /*dil*/ dil);
    y = core_act::snake_alpha(ctx, y, ru.alpha1);
    y = dac_conv1d(ctx, y, ru.conv1_w, ru.conv1_b, /*pad*/ 0, /*dil*/ 1);
    return ggml_add(ctx, x, y);
}

// DAC DecoderBlock: Snake1d -> ConvTranspose1d(stride=s, k=2s, p=s/2) -> 3 ResidualUnits
static ggml_tensor* dac_decoder_block(ggml_context* ctx, ggml_tensor* x, const core_dac::DacDecoderBlock& blk,
                                      int stride) {
    x = core_act::snake_alpha(ctx, x, blk.snake_alpha);
    if (blk.up_w_perm) {
        const int K = (int)blk.up_w->ne[0];
        x = core_convt::convt1d_decomp(ctx, x, blk.up_w_perm, blk.up_b, stride, K, stride / 2, stride / 2);
    } else {
        x = core_convt::convt1d_crop(ctx, x, blk.up_w, blk.up_b, stride,
                                     /*crop_left*/ stride / 2, /*crop_right*/ stride / 2);
    }
    static const int dilations[3] = {1, 3, 9};
    for (int r = 0; r < 3; r++) {
        x = dac_residual_unit(ctx, x, blk.res[r], dilations[r]);
    }
    return x;
}

// Build DAC decode graph: codes (n_codebooks, n_frames) -> PCM (n_frames * 512)
static ggml_cgraph* dac_build_decode_graph(const core_dac::DacWeights& dac, ggml_context* ctx0, int n_frames) {
    const auto& cfg = dac.config;
    const int n_cb = cfg.n_codebooks;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: per-codebook code tensors (1D, i32)
    std::vector<ggml_tensor*> codes_in(n_cb);
    for (int k = 0; k < n_cb; k++) {
        char name[32];
        std::snprintf(name, sizeof(name), "dac_codes_%d", k);
        codes_in[k] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
        ggml_set_name(codes_in[k], name);
        ggml_set_input(codes_in[k]);
    }

    // RVQ dequantize: for each codebook, lookup + linear project, then sum
    ggml_tensor* z_q = nullptr;
    for (int k = 0; k < n_cb; k++) {
        const auto& q = dac.quantizers[k];
        // codebook: (codebook_dim=8, codebook_size=1024) — ggml_get_rows indexes dim1
        ggml_tensor* z = ggml_get_rows(ctx0, q.codebook, codes_in[k]); // (8, n_frames)
        z = ggml_cast(ctx0, z, GGML_TYPE_F32);

        // out_proj: weight (1, 8, 1024) — reshape to (8, 1024) for mul_mat
        ggml_tensor* proj_w = ggml_reshape_2d(ctx0, q.out_proj_w, 8, 1024);
        proj_w = ggml_cast(ctx0, proj_w, GGML_TYPE_F32);
        z = ggml_mul_mat(ctx0, proj_w, z); // (1024, n_frames)
        if (q.out_proj_b) {
            z = ggml_add(ctx0, z, q.out_proj_b);
        }

        if (k == 0) {
            z_q = z;
        } else {
            z_q = ggml_add(ctx0, z_q, z);
        }
    }
    z_q = ggml_cont(ctx0, z_q); // (1024, n_frames)

    // Decoder input conv: Conv1d(1024, 1536, k=7, p=3)
    ggml_tensor* h = dac_conv1d(ctx0, z_q, dac.in_conv_w, dac.in_conv_b, /*pad*/ 3, /*dil*/ 1);

    // 4 decoder blocks with strides [8, 8, 4, 2]
    for (int b = 0; b < cfg.n_decoder_blocks; b++) {
        h = dac_decoder_block(ctx0, h, dac.blocks[b], cfg.upsampling_ratios[b]);
        h = ggml_cont(ctx0, h);
    }

    // Output: Snake1d -> Conv1d(96, 1, k=7, p=3) -> tanh
    h = core_act::snake_alpha(ctx0, h, dac.out_snake_alpha);
    h = dac_conv1d(ctx0, h, dac.out_conv_w, dac.out_conv_b, /*pad*/ 3, /*dil*/ 1);
    h = ggml_tanh(ctx0, h);

    // Reshape to 1D PCM
    const int T_pcm = (int)h->ne[1]; // (1, T_pcm)
    h = ggml_reshape_1d(ctx0, h, (int64_t)T_pcm);
    h = ggml_cont(ctx0, h);
    ggml_set_name(h, "dac_pcm");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    return gf;
}

// Run DAC decode on integer codes and return PCM float buffer.
// filtered_codes: interleaved [frame0_cb0, frame0_cb1, ..., frame0_cb8, frame1_cb0, ...]
// n_frames: number of code frames
// Returns malloc'd float buffer with n_frames * hop_length samples, or nullptr on error.
static float* dac_decode(dia_tts_context* ctx, const std::vector<uint32_t>& filtered_codes, size_t n_frames,
                         int* out_n_samples) {
    auto& m = ctx->model;
    const auto& cfg = m.dac.config;
    const int n_cb = cfg.n_codebooks;
    const int verbosity = ctx->params.verbosity;

    // Build per-codebook code arrays (transposing from interleaved layout)
    std::vector<std::vector<int32_t>> codes_per_cb(n_cb);
    for (int k = 0; k < n_cb; k++) {
        codes_per_cb[k].resize(n_frames);
    }
    for (size_t f = 0; f < n_frames; f++) {
        for (int k = 0; k < n_cb; k++) {
            codes_per_cb[k][f] = (int32_t)filtered_codes[f * n_cb + k];
        }
    }

    // Allocate compute context
    const size_t buf_size =
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8 + ggml_graph_overhead_custom(16384, false);
    std::vector<uint8_t> compute_meta(buf_size);
    ggml_init_params ip = {compute_meta.size(), compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        fprintf(stderr, "dia_dac: failed to init compute context\n");
        return nullptr;
    }

    ggml_cgraph* gf = dac_build_decode_graph(m.dac, ctx0, (int)n_frames);

    // Allocate graph buffers
    ggml_backend_t backend = ctx->backend ? ctx->backend : ctx->backend_cpu;
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        ggml_free(ctx0);
        return nullptr;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "dia_dac: failed to allocate decode graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Set input codes
    for (int k = 0; k < n_cb; k++) {
        char name[32];
        std::snprintf(name, sizeof(name), "dac_codes_%d", k);
        ggml_tensor* inp = ggml_graph_get_tensor(gf, name);
        if (!inp) {
            fprintf(stderr, "dia_dac: input tensor '%s' not in graph\n", name);
            ggml_gallocr_free(galloc);
            ggml_free(ctx0);
            return nullptr;
        }
        ggml_backend_tensor_set(inp, codes_per_cb[k].data(), 0, n_frames * sizeof(int32_t));
    }

    if (verbosity >= 1) {
        fprintf(stderr, "dia_dac: decoding %zu frames -> %zu samples...\n", n_frames, n_frames * cfg.hop_length);
    }

    // Compute
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "dia_dac: graph compute failed (status=%d)\n", (int)st);
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Extract PCM output
    ggml_tensor* pcm_out = ggml_graph_get_tensor(gf, "dac_pcm");
    if (!pcm_out) {
        fprintf(stderr, "dia_dac: pcm output tensor not in graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    const int n_samples = (int)ggml_nelements(pcm_out);
    float* pcm = (float*)malloc(n_samples * sizeof(float));
    if (!pcm) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_backend_tensor_get(pcm_out, pcm, 0, n_samples * sizeof(float));
    *out_n_samples = n_samples;

    if (verbosity >= 1) {
        fprintf(stderr, "dia_dac: decoded %d PCM samples (%.2f sec at %d Hz)\n", n_samples,
                (float)n_samples / cfg.sample_rate, cfg.sample_rate);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return pcm;
}

float* dia_tts_synthesize(struct dia_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    auto& m = ctx->model;
    auto& p = ctx->params;

    // DIA_DECODE_CODES=path: isolate the DAC. Load post-revert codebook (T*9 int32,
    // interleaved [frame*9+ch]) and decode it directly, bypassing generation.
    if (const char* cp = getenv("DIA_DECODE_CODES")) {
        std::vector<uint32_t> codes;
        FILE* f = fopen(cp, "rb");
        if (f) {
            int32_t v;
            while (fread(&v, sizeof(int32_t), 1, f) == 1)
                codes.push_back((uint32_t)v);
            fclose(f);
        }
        size_t n_frames = codes.size() / m.n_output_heads;
        fprintf(stderr, "DIA_DECODE_CODES: %zu frames\n", n_frames);
        if (!m.has_dac || n_frames == 0)
            return nullptr;
        int ns = 0;
        float* pcm = dac_decode(ctx, codes, n_frames, &ns);
        *out_n_samples = ns;
        return pcm;
    }

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
    // TEMP: limit for CPU testing (full 3072 steps impractical on this CPU).
    // Override with DIA_MAX_STEPS for longer prompts on faster backends.
    uint32_t step_cap = 200;
    if (const char* ms = getenv("DIA_MAX_STEPS"))
        step_cap = (uint32_t)atoi(ms);
    if (max_gen > step_cap)
        max_gen = step_cap;

    // --- diff/debug hooks (env-gated; output paths come from the env value, never hardcoded) ---
    // DIA_GREEDY=1         : temperature=0 (argmax) for deterministic diffing
    // DIA_DUMP_TOKENS=1    : print emitted 9-channel tokens per step to stderr
    // DIA_FORCE_TOKENS=f   : teacher-force per-step input from file f (N*9 int32 raw); caps max_gen=N
    // DIA_DUMP_STEPLOGITS=f: append per-step [uncond(9*V) | cond(9*V)] f32 to file f
    // DIA_DUMP_DIR=d       : write step-0 stage dumps (encoder/cross/ca/final/logits) under dir d
    const bool dia_greedy = getenv("DIA_GREEDY") != nullptr;
    const bool dia_dump_tokens = getenv("DIA_DUMP_TOKENS") != nullptr;
    const char* dia_dump_dir = getenv("DIA_DUMP_DIR");
    auto dia_dpath = [&](const char* name) { return std::string(dia_dump_dir ? dia_dump_dir : ".") + "/" + name; };
    std::vector<int32_t> dia_forced;
    if (const char* fp = getenv("DIA_FORCE_TOKENS")) {
        FILE* f = fopen(fp, "rb");
        if (f) {
            int32_t v;
            while (fread(&v, sizeof(int32_t), 1, f) == 1)
                dia_forced.push_back(v);
            fclose(f);
        }
        fprintf(stderr, "DIA_FORCE_TOKENS: loaded %zu ints (%zu steps)\n", dia_forced.size(),
                dia_forced.size() / m.n_output_heads);
    }
    const bool dia_force = !dia_forced.empty();
    const char* dia_steplogits_path = getenv("DIA_DUMP_STEPLOGITS");
    if (dia_force)
        max_gen = (uint32_t)(dia_forced.size() / m.n_output_heads);
    if (dia_steplogits_path)
        remove(dia_steplogits_path); // truncate; we append per step

    // Delayed-domain token buffer (mirrors Python dec_output.generated_tokens).
    // The model is trained in the delay domain: channel c's INPUT is held at BOS
    // until step delay[c], then consumes previously-generated tokens. Without this,
    // free-run generation feeds all channels real tokens immediately (out-of-dist).
    //   gen[t][c] = BOS for t <= delay[c] (prefill), else filled as generation proceeds.
    const int dia_max_delay = (int)m.max_delay;
    const int gen_len = (int)max_gen + dia_max_delay + 2;
    std::vector<std::vector<int32_t>> gen(gen_len, std::vector<int32_t>(m.n_output_heads, -1));
    for (int t = 0; t <= dia_max_delay && t < gen_len; t++)
        for (uint32_t c = 0; c < m.n_output_heads; c++)
            if (t <= (int)m.delay_pattern[c])
                gen[t][c] = (int32_t)m.bos_token_id;
    int bos_countdown = dia_max_delay; // apply_mask region (start-of-sequence delay)

    // ===================================================================
    // 1. Run encoder (batch=2: conditional + unconditional)
    // ===================================================================
    // Use actual text length instead of max_encoder_context to avoid
    // attention dilution from padding (mask handling is complex with CFG batch).
    const int T_enc = (int)ctx->prompt_size;
    const int B = 2;
    const int enc_hidden = (int)m.encoder_hidden_size;
    const int dec_hidden = (int)m.decoder_hidden_size;
    const int kv_dim = (int)(m.decoder_kv_heads * m.head_dim);         // self-attn: 4*128=512
    const int cross_kv_dim = (int)(m.decoder_attn_heads * m.head_dim); // cross-attn: 16*128=2048 (MHA, not GQA)
    const int n_heads = (int)m.decoder_attn_heads;
    const int n_kv_heads = (int)m.decoder_kv_heads;
    const int head_dim = (int)m.head_dim;

    // Prepare encoder input: [uncond_tokens(zeros) | cond_tokens]
    // Dia CFG convention: batch 0 = unconditional, batch 1 = conditional
    std::vector<int32_t> enc_input(T_enc * B, 0);
    for (size_t i = 0; i < tokens.size() && i < (size_t)T_enc; i++) {
        enc_input[T_enc + i] = (int32_t)tokens[i]; // conditional = batch 1 (offset T_enc)
    }
    // batch 0 (first T_enc) = all zeros (unconditional)

    // Positions: [0, 1, 2, ..., T_enc-1]
    std::vector<int32_t> positions(T_enc);
    for (int i = 0; i < T_enc; i++)
        positions[i] = i;

    // Attention mask: (T_enc, T_enc, 1, B) — 0 = attend, -inf = block.
    // Since T_enc = prompt_size (no padding), all tokens are real in the cond batch
    // and all zeros in the uncond batch. No cross-group masking needed.
    std::vector<float> enc_mask(T_enc * T_enc * 1 * B, 0.0f);

    // Encoder graph
    std::vector<float> encoder_output;
    {
        size_t ctx_size = 256 * 1024 * 1024; // 256 MB for graph metadata
        ggml_init_params gp = {ctx_size, nullptr, true};
        ggml_context* ctx0 = ggml_init(gp);
        if (!ctx0) {
            fprintf(stderr, "dia_tts: failed to allocate encoder graph context\n");
            return nullptr;
        }

        ggml_tensor* inp_tokens_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_enc * B);
        ggml_set_name(inp_tokens_t, "enc_tokens");
        ggml_set_input(inp_tokens_t);

        ggml_tensor* positions_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_enc);
        ggml_set_name(positions_t, "enc_positions");
        ggml_set_input(positions_t);

        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_enc, T_enc, 1, B);
        ggml_set_name(mask_t, "enc_mask");
        ggml_set_input(mask_t);

        ggml_tensor* enc_out = build_dia_encoder(ctx0, m, inp_tokens_t, positions_t, mask_t);
        ggml_set_name(enc_out, "enc_output");
        ggml_set_output(enc_out);

        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);
        ggml_build_forward_expand(gf, enc_out);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            fprintf(stderr, "dia_tts: encoder graph alloc failed\n");
            if (alloc)
                ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return nullptr;
        }

        // Set inputs
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_tokens"), enc_input.data(), 0,
                                T_enc * B * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_positions"), positions.data(), 0,
                                T_enc * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_mask"), enc_mask.data(), 0,
                                enc_mask.size() * sizeof(float));

        ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "dia_tts: encoder compute failed (status=%d)\n", (int)st);
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return nullptr;
        }

        // Read encoder output: (enc_hidden, T_enc, B) = (1024, 1024, 2)
        ggml_tensor* enc_result = ggml_graph_get_tensor(gf, "enc_output");
        encoder_output.resize(enc_hidden * T_enc * B);
        ggml_backend_tensor_get(enc_result, encoder_output.data(), 0, encoder_output.size() * sizeof(float));

        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
    }

    if (p.verbosity >= 1) {
        fprintf(stderr, "dia_tts: encoder done, output shape (%d, %d, %d)\n", enc_hidden, T_enc, B);
        // Print first few values of conditional encoder output for diff-testing
        // Conditional = batch index 1, offset = enc_hidden * T_enc * 1
        size_t cond_offset = (size_t)enc_hidden * T_enc;
        fprintf(stderr, "dia_tts: enc cond pos0 first 4: %.6f %.6f %.6f %.6f\n", encoder_output[cond_offset + 0],
                encoder_output[cond_offset + 1], encoder_output[cond_offset + 2], encoder_output[cond_offset + 3]);
        // Compute norm of first position
        float norm0 = 0;
        for (int d = 0; d < enc_hidden; d++)
            norm0 += encoder_output[cond_offset + d] * encoder_output[cond_offset + d];
        fprintf(stderr, "dia_tts: enc cond pos0 norm: %.4f (ref: 2.2434)\n", sqrtf(norm0));
        if (dia_dump_dir) {
            float un0 = 0;
            for (int d = 0; d < enc_hidden; d++)
                un0 += encoder_output[d] * encoder_output[d]; // uncond = batch 0, pos 0
            fprintf(stderr, "dia_tts: enc uncond pos0 norm: %.4f\n", sqrtf(un0));
        }
    }

    // ===================================================================
    // 2. Precompute cross-attention K/V for all decoder layers
    // ===================================================================
    // cross_k[layer]: (cross_kv_dim=2048, T_enc, B) — projected from encoder output
    // cross_v[layer]: (cross_kv_dim=2048, T_enc, B)
    std::vector<std::vector<float>> cross_k(m.n_decoder_layers);
    std::vector<std::vector<float>> cross_v(m.n_decoder_layers);

    {
        size_t ctx_size = 64 * 1024 * 1024;
        ggml_init_params gp = {ctx_size, nullptr, true};
        ggml_context* ctx0 = ggml_init(gp);
        if (!ctx0) {
            fprintf(stderr, "dia_tts: failed to allocate cross-attn context\n");
            return nullptr;
        }

        // Encoder output as input tensor
        ggml_tensor* enc_in = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, enc_hidden, T_enc, B);
        ggml_set_name(enc_in, "enc_in");
        ggml_set_input(enc_in);

        // Project cross K/V for each layer
        std::vector<ggml_tensor*> outputs;
        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            auto& layer = m.decoder.layers[l];
            ggml_tensor* ck = ggml_mul_mat(ctx0, layer.cross_k_proj, enc_in);
            ggml_tensor* cv = ggml_mul_mat(ctx0, layer.cross_v_proj, enc_in);
            std::string kname = "cross_k_" + std::to_string(l);
            std::string vname = "cross_v_" + std::to_string(l);
            ggml_set_name(ck, kname.c_str());
            ggml_set_name(cv, vname.c_str());
            ggml_set_output(ck);
            ggml_set_output(cv);
            outputs.push_back(ck);
            outputs.push_back(cv);
        }

        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);
        for (auto* o : outputs)
            ggml_build_forward_expand(gf, o);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            fprintf(stderr, "dia_tts: cross-attn graph alloc failed\n");
            if (alloc)
                ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return nullptr;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_in"), encoder_output.data(), 0,
                                encoder_output.size() * sizeof(float));

        ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "dia_tts: cross-attn compute failed\n");
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return nullptr;
        }

        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            std::string kname = "cross_k_" + std::to_string(l);
            std::string vname = "cross_v_" + std::to_string(l);
            cross_k[l].resize(cross_kv_dim * T_enc * B);
            cross_v[l].resize(cross_kv_dim * T_enc * B);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, kname.c_str()), cross_k[l].data(), 0,
                                    cross_k[l].size() * sizeof(float));
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, vname.c_str()), cross_v[l].data(), 0,
                                    cross_v[l].size() * sizeof(float));
        }

        if (dia_dump_dir) {
            // cross_k[0]/cross_v[0] layout (cross_kv_dim, T_enc, B)
            FILE* fk = fopen(dia_dpath("cpp_cross_k0.f32").c_str(), "wb");
            FILE* fv = fopen(dia_dpath("cpp_cross_v0.f32").c_str(), "wb");
            if (fk) {
                fwrite(cross_k[0].data(), sizeof(float), cross_k[0].size(), fk);
                fclose(fk);
            }
            if (fv) {
                fwrite(cross_v[0].data(), sizeof(float), cross_v[0].size(), fv);
                fclose(fv);
            }
            fprintf(stderr, "DIA_DUMP: cpp_cross_{k,v}0.f32 (cross_kv_dim=%d T_enc=%d B=%d)\n", cross_kv_dim, T_enc, B);
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
    }

    if (p.verbosity >= 1) {
        fprintf(stderr, "dia_tts: cross-attention K/V cached for %u layers\n", m.n_decoder_layers);
    }

    // ===================================================================
    // 3. Decoder AR loop
    // ===================================================================
    // Self-attention KV cache: stored per-layer in CPU vectors
    // Shape per layer: k[step * kv_dim * B], v[step * kv_dim * B]
    std::vector<std::vector<float>> self_k(m.n_decoder_layers);
    std::vector<std::vector<float>> self_v(m.n_decoder_layers);

    for (uint32_t step = 0; step < max_gen; step++) {
        ctx->current_position = step;
        if (step == 0 && p.verbosity >= 1)
            fprintf(stderr, "dia_tts: starting decoder loop (max_gen=%u, T_past=%d)\n", max_gen, (int)step);

        // Step input tokens. Free-run: read the delayed-domain buffer (channels held at
        // BOS until their delay elapses). Teacher-forcing: use the reference sequence.
        if (dia_force) {
            for (uint32_t h = 0; h < m.n_output_heads; h++)
                ctx->current_audio_tokens[h] = (uint32_t)dia_forced[(size_t)step * m.n_output_heads + h];
        } else {
            for (uint32_t h = 0; h < m.n_output_heads; h++) {
                int32_t g = gen[step][h];
                ctx->current_audio_tokens[h] = (g >= 0) ? (uint32_t)g : m.bos_token_id;
            }
        }

        // Build decoder step graph
        size_t ctx_size = 128 * 1024 * 1024;
        ggml_init_params gp = {ctx_size, nullptr, true};
        ggml_context* ctx0 = ggml_init(gp);
        if (!ctx0) {
            fprintf(stderr, "dia_tts: failed to allocate decoder step context at step %u\n", step);
            return nullptr;
        }

        const int T_past = (int)step; // number of past positions in KV cache
        const int T_cur = 1;          // single position

        // --- Input: audio tokens for current step (n_output_heads * B interleaved) ---
        ggml_tensor* audio_tokens_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, m.n_output_heads * B);
        ggml_set_name(audio_tokens_t, "audio_tokens");
        ggml_set_input(audio_tokens_t);

        // --- Decoder embedding: sum of codebook embeddings -> (dec_hidden, 1, B) ---
        ggml_tensor* cur = build_dia_decoder_embedding(ctx0, m, audio_tokens_t);
        // cur shape: (dec_hidden, B) from get_rows -> reshape to (dec_hidden, 1, B)
        cur = ggml_reshape_3d(ctx0, cur, dec_hidden, T_cur, B);

        // --- Self-attention KV past inputs ---
        std::vector<ggml_tensor*> past_k_inputs(m.n_decoder_layers);
        std::vector<ggml_tensor*> past_v_inputs(m.n_decoder_layers);
        // Cross-attention K/V inputs (full encoder length)
        std::vector<ggml_tensor*> cross_k_inputs(m.n_decoder_layers);
        std::vector<ggml_tensor*> cross_v_inputs(m.n_decoder_layers);

        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            if (T_past > 0) {
                past_k_inputs[l] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, kv_dim, T_past, B);
                past_v_inputs[l] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, kv_dim, T_past, B);
                std::string kn = "past_k_" + std::to_string(l);
                std::string vn = "past_v_" + std::to_string(l);
                ggml_set_name(past_k_inputs[l], kn.c_str());
                ggml_set_name(past_v_inputs[l], vn.c_str());
                ggml_set_input(past_k_inputs[l]);
                ggml_set_input(past_v_inputs[l]);
            }
            cross_k_inputs[l] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, cross_kv_dim, T_enc, B);
            cross_v_inputs[l] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, cross_kv_dim, T_enc, B);
            std::string ckn = "cross_k_in_" + std::to_string(l);
            std::string cvn = "cross_v_in_" + std::to_string(l);
            ggml_set_name(cross_k_inputs[l], ckn.c_str());
            ggml_set_name(cross_v_inputs[l], cvn.c_str());
            ggml_set_input(cross_k_inputs[l]);
            ggml_set_input(cross_v_inputs[l]);
        }

        // --- Position for RoPE ---
        ggml_tensor* dec_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_cur);
        ggml_set_name(dec_pos, "dec_pos");
        ggml_set_input(dec_pos);

        // --- New K/V outputs for self-attention (to append to cache) ---
        std::vector<ggml_tensor*> new_k_outputs(m.n_decoder_layers);
        std::vector<ggml_tensor*> new_v_outputs(m.n_decoder_layers);

        // --- Decoder layers ---
        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            auto& layer = m.decoder.layers[l];
            ggml_tensor* residual = cur;

            // Pre self-attention norm
            cur = dia_rms_norm(ctx0, cur, layer.pre_sa_norm, m.rms_norm_eps);

            // Self-attention
            {
                // Q/K/V projections for current position
                ggml_tensor* Q = ggml_mul_mat(ctx0, layer.self_q_proj, cur);     // (dec_hidden, 1, B)
                ggml_tensor* K_cur = ggml_mul_mat(ctx0, layer.self_k_proj, cur); // (kv_dim, 1, B)
                ggml_tensor* V_cur = ggml_mul_mat(ctx0, layer.self_v_proj, cur); // (kv_dim, 1, B)

                // Reshape Q to (head_dim, n_heads, 1, B) and apply RoPE
                Q = ggml_reshape_4d(ctx0, Q, head_dim, n_heads, T_cur, B);
                Q = ggml_rope(ctx0, Q, dec_pos, head_dim, DIA_ROPE_MODE);

                // Reshape K to (head_dim, n_kv_heads, 1, B) and apply RoPE
                K_cur = ggml_reshape_4d(ctx0, K_cur, head_dim, n_kv_heads, T_cur, B);
                K_cur = ggml_rope(ctx0, K_cur, dec_pos, head_dim, DIA_ROPE_MODE);

                // Store new K/V for cache (flatten back)
                ggml_tensor* k_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, K_cur), kv_dim, T_cur, B);
                ggml_tensor* v_out = V_cur; // already (kv_dim, 1, B)
                std::string kon = "new_k_" + std::to_string(l);
                std::string von = "new_v_" + std::to_string(l);
                ggml_set_name(k_out, kon.c_str());
                ggml_set_output(k_out);
                ggml_set_name(v_out, von.c_str());
                ggml_set_output(v_out);
                new_k_outputs[l] = k_out;
                new_v_outputs[l] = v_out;

                // Concatenate past K/V with current for attention
                // Full K: (head_dim, n_kv_heads, T_past+1, B)
                ggml_tensor* K_full;
                ggml_tensor* V_full;
                if (T_past > 0) {
                    // past_k: (kv_dim, T_past, B) -> reshape to (head_dim, n_kv_heads, T_past, B)
                    ggml_tensor* past_k_4d = ggml_reshape_4d(ctx0, past_k_inputs[l], head_dim, n_kv_heads, T_past, B);
                    // Apply RoPE to past K with past positions - NO, past K already has RoPE applied
                    // Actually we need to store post-RoPE K in cache. K_cur already has RoPE.
                    // Concat on T dimension
                    K_full =
                        ggml_concat(ctx0, past_k_4d,
                                    ggml_reshape_4d(ctx0, ggml_cont(ctx0, K_cur), head_dim, n_kv_heads, T_cur, B), 2);
                    ggml_tensor* past_v_4d = ggml_reshape_4d(ctx0, past_v_inputs[l], head_dim, n_kv_heads, T_past, B);
                    ggml_tensor* v_cur_4d = ggml_reshape_4d(ctx0, V_cur, head_dim, n_kv_heads, T_cur, B);
                    V_full = ggml_concat(ctx0, past_v_4d, v_cur_4d, 2);
                } else {
                    K_full = ggml_reshape_4d(ctx0, ggml_cont(ctx0, K_cur), head_dim, n_kv_heads, T_cur, B);
                    V_full = ggml_reshape_4d(ctx0, V_cur, head_dim, n_kv_heads, T_cur, B);
                }

                // GQA: repeat_interleave K/V heads to match Q heads
                // n_heads=16, n_kv_heads=4 -> each KV head maps to 4 consecutive Q heads
                // Pattern: insert unit dim before n_kv, repeat, flatten
                // (hd, n_kv, T, B) -> (hd, 1, n_kv, T*B) -> repeat -> (hd, n_rep, n_kv, T*B) -> (hd, n_heads, T, B)
                int n_rep = n_heads / n_kv_heads;
                if (n_rep > 1) {
                    int T_full = T_past + T_cur;
                    K_full = ggml_reshape_4d(ctx0, K_full, head_dim, 1, n_kv_heads, T_full * B);
                    K_full = ggml_repeat_4d(ctx0, K_full, head_dim, n_rep, n_kv_heads, T_full * B);
                    K_full = ggml_cont(ctx0, ggml_reshape_4d(ctx0, K_full, head_dim, n_heads, T_full, B));
                    V_full = ggml_reshape_4d(ctx0, V_full, head_dim, 1, n_kv_heads, T_full * B);
                    V_full = ggml_repeat_4d(ctx0, V_full, head_dim, n_rep, n_kv_heads, T_full * B);
                    V_full = ggml_cont(ctx0, ggml_reshape_4d(ctx0, V_full, head_dim, n_heads, T_full, B));
                }

                // Attention: Q @ K^T / sqrt(d)
                // Q: (head_dim, n_heads, 1, B)
                // K: (head_dim, n_heads, T_full, B)
                // Permute Q to (head_dim, 1, n_heads, B) and K to (head_dim, T_full, n_heads, B)
                ggml_tensor* q_perm = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                ggml_tensor* k_perm = ggml_cont(ctx0, ggml_permute(ctx0, K_full, 0, 2, 1, 3));
                // kq: (T_full, 1, n_heads, B) — dot product contracts on head_dim
                ggml_tensor* kq = ggml_mul_mat(ctx0, k_perm, q_perm);
                // Scale by 1/sqrt(head_dim)
                // Dia uses scale=1.0 (no 1/sqrt(d) scaling) — matching Python SelfAttention/CrossAttention
                // Softmax (no mask needed for causal since we only have past + current)
                kq = ggml_soft_max(ctx0, kq);

                // V @ attn_weights
                // V: (head_dim, n_heads, T_full, B) -> permute to (head_dim, T_full, n_heads, B)
                ggml_tensor* v_perm = ggml_cont(ctx0, ggml_permute(ctx0, V_full, 0, 2, 1, 3));
                // v_perm transposed: (T_full, head_dim, n_heads, B)
                ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_perm));
                // kqv: mul_mat(v_t, kq) contracts on T_full -> (head_dim, 1, n_heads, B)
                ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);
                // Permute to (head_dim, n_heads, 1, B) then reshape to (dec_hidden, 1, B)
                kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
                cur = ggml_reshape_3d(ctx0, kqv, dec_hidden, T_cur, B);
                cur = ggml_mul_mat(ctx0, layer.self_o_proj, cur);
            }

            cur = ggml_add(ctx0, cur, residual);
            residual = cur;

            // Pre cross-attention norm
            cur = dia_rms_norm(ctx0, cur, layer.pre_ca_norm, m.rms_norm_eps);

            // Cross-attention
            {
                // Q from decoder hidden state
                ggml_tensor* Q = ggml_mul_mat(ctx0, layer.cross_q_proj, cur); // (dec_hidden, 1, B)
                Q = ggml_reshape_4d(ctx0, Q, head_dim, n_heads, T_cur, B);
                // No RoPE on cross-attention Q (cross-attn uses absolute position from encoder)

                // K/V from precomputed cross-attention cache
                // cross_k: (cross_kv_dim=2048, T_enc, B) -> (hd, n_heads, T_enc, B)
                // MHA (16q, 16kv) — no GQA repeat needed
                ggml_tensor* K_cross = ggml_reshape_4d(ctx0, cross_k_inputs[l], head_dim, n_heads, T_enc, B);
                ggml_tensor* V_cross = ggml_reshape_4d(ctx0, cross_v_inputs[l], head_dim, n_heads, T_enc, B);

                // Attention
                ggml_tensor* q_perm = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                ggml_tensor* k_perm = ggml_cont(ctx0, ggml_permute(ctx0, K_cross, 0, 2, 1, 3));
                ggml_tensor* kq = ggml_mul_mat(ctx0, k_perm, q_perm);
                // Dia uses scale=1.0 (no 1/sqrt(d) scaling) — matching Python SelfAttention/CrossAttention
                kq = ggml_soft_max(ctx0, kq);

                ggml_tensor* v_perm = ggml_cont(ctx0, ggml_permute(ctx0, V_cross, 0, 2, 1, 3));
                ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_perm));
                ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);
                kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
                cur = ggml_reshape_3d(ctx0, kqv, dec_hidden, T_cur, B);
                cur = ggml_mul_mat(ctx0, layer.cross_o_proj, cur);
                if (step == 0 && l == 0 && dia_dump_dir) {
                    ggml_set_name(cur, "ca0_out");
                    ggml_set_output(cur);
                }
            }

            cur = ggml_add(ctx0, cur, residual);
            residual = cur;

            // Pre MLP norm
            cur = dia_rms_norm(ctx0, cur, layer.pre_mlp_norm, m.rms_norm_eps);

            // MLP: SiLU(gate(x)) * up(x), then down
            {
                cur = ggml_mul(ctx0, ggml_silu(ctx0, ggml_mul_mat(ctx0, layer.gate, cur)),
                               ggml_mul_mat(ctx0, layer.up, cur));
                cur = ggml_mul_mat(ctx0, layer.wo_mlp, cur);
            }

            cur = ggml_add(ctx0, cur, residual);
        }

        // Final norm
        cur = dia_rms_norm(ctx0, cur, m.decoder.norm, m.rms_norm_eps);
        // cur: (dec_hidden, 1, B)
        if (step == 0 && dia_dump_dir) {
            ggml_set_name(cur, "dia_final_hidden");
            ggml_set_output(cur);
        }

        // Project to logits for each codebook head
        // We need all 9 heads' logits. Output shape: (output_vocab_size * n_output_heads, 1, B)
        std::vector<ggml_tensor*> head_logits(m.n_output_heads);
        for (int h = 0; h < (int)m.n_output_heads; h++) {
            head_logits[h] = ggml_mul_mat(ctx0, m.decoder.heads[h], cur); // (output_vocab_size, 1, B)
            std::string hn = "logits_" + std::to_string(h);
            ggml_set_name(head_logits[h], hn.c_str());
            ggml_set_output(head_logits[h]);
        }

        // Build and compute graph
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);
        for (int h = 0; h < (int)m.n_output_heads; h++) {
            ggml_build_forward_expand(gf, head_logits[h]);
        }
        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            ggml_build_forward_expand(gf, new_k_outputs[l]);
            ggml_build_forward_expand(gf, new_v_outputs[l]);
        }

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            fprintf(stderr, "dia_tts: decoder step %u graph alloc failed\n", step);
            if (alloc)
                ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return nullptr;
        }

        // Set inputs
        // Audio tokens: interleaved [head0_cond, head0_uncond, head1_cond, head1_uncond, ...]
        // Actually from build_dia_decoder_embedding: stride view picks every n_output_heads-th element
        // So layout is: [cond_h0, cond_h1, ..., cond_h8, uncond_h0, ..., uncond_h8]
        std::vector<int32_t> audio_input(m.n_output_heads * B);
        for (int h = 0; h < (int)m.n_output_heads; h++) {
            audio_input[h] = (int32_t)ctx->current_audio_tokens[h];                    // conditional
            audio_input[m.n_output_heads + h] = (int32_t)ctx->current_audio_tokens[h]; // unconditional (same tokens)
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "audio_tokens"), audio_input.data(), 0,
                                audio_input.size() * sizeof(int32_t));

        // Position
        int32_t pos_val = (int32_t)step;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_pos"), &pos_val, 0, sizeof(int32_t));

        // Past self-attention K/V.
        // The cache vectors accumulate per step in [step][batch] order:
        //   self_k[l] = [s0b0, s0b1, s1b0, s1b1, ...]  (each block kv_dim floats)
        // but the past_k_* input tensor is (kv_dim, T_past, B), i.e. ggml [batch][step]:
        //   [b0: t0,t1,..,t_{T-1}][b1: t0,t1,...].
        // These coincide only at T_past==1, so without reordering the cache the
        // self-attention silently corrupts from the 3rd decode step on. Reorder here.
        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            if (T_past > 0) {
                std::string kn = "past_k_" + std::to_string(l);
                std::string vn = "past_v_" + std::to_string(l);
                std::vector<float> pk((size_t)kv_dim * T_past * B), pv((size_t)kv_dim * T_past * B);
                for (int t = 0; t < T_past; t++) {
                    for (int b = 0; b < B; b++) {
                        const size_t src = ((size_t)t * B + b) * kv_dim;      // [step][batch]
                        const size_t dst = ((size_t)b * T_past + t) * kv_dim; // (kv_dim,T_past,B)
                        std::memcpy(&pk[dst], &self_k[l][src], kv_dim * sizeof(float));
                        std::memcpy(&pv[dst], &self_v[l][src], kv_dim * sizeof(float));
                    }
                }
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, kn.c_str()), pk.data(), 0, pk.size() * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, vn.c_str()), pv.data(), 0, pv.size() * sizeof(float));
            }
            // Cross-attention K/V
            std::string ckn = "cross_k_in_" + std::to_string(l);
            std::string cvn = "cross_v_in_" + std::to_string(l);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, ckn.c_str()), cross_k[l].data(), 0,
                                    cross_k[l].size() * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, cvn.c_str()), cross_v[l].data(), 0,
                                    cross_v[l].size() * sizeof(float));
        }

        // Compute
        ggml_status st = ggml_backend_graph_compute(ctx->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "dia_tts: decoder step %u compute failed\n", step);
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return nullptr;
        }

        if (step == 0 && dia_dump_dir) {
            ggml_tensor* fh = ggml_graph_get_tensor(gf, "dia_final_hidden");
            if (fh) {
                std::vector<float> hbuf(dec_hidden * B);
                ggml_backend_tensor_get(fh, hbuf.data(), 0, hbuf.size() * sizeof(float));
                FILE* f = fopen(dia_dpath("cpp_final_hidden.f32").c_str(), "wb");
                if (f) {
                    fwrite(hbuf.data(), sizeof(float), hbuf.size(), f);
                    fclose(f);
                }
            }
            ggml_tensor* ca = ggml_graph_get_tensor(gf, "ca0_out");
            if (ca) {
                std::vector<float> cbuf(dec_hidden * B);
                ggml_backend_tensor_get(ca, cbuf.data(), 0, cbuf.size() * sizeof(float));
                FILE* f = fopen(dia_dpath("cpp_ca0_out.f32").c_str(), "wb");
                if (f) {
                    fwrite(cbuf.data(), sizeof(float), cbuf.size(), f);
                    fclose(f);
                }
            }
        }

        // Read new K/V and append to self-attention cache
        for (int l = 0; l < (int)m.n_decoder_layers; l++) {
            std::string kon = "new_k_" + std::to_string(l);
            std::string von = "new_v_" + std::to_string(l);
            std::vector<float> k_new(kv_dim * T_cur * B);
            std::vector<float> v_new(kv_dim * T_cur * B);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, kon.c_str()), k_new.data(), 0,
                                    k_new.size() * sizeof(float));
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, von.c_str()), v_new.data(), 0,
                                    v_new.size() * sizeof(float));
            self_k[l].insert(self_k[l].end(), k_new.begin(), k_new.end());
            self_v[l].insert(self_v[l].end(), v_new.begin(), v_new.end());
        }

        // Read logits and apply CFG, then sample
        if (step == 0 && p.verbosity >= 1) {
            fprintf(stderr, "dia_tts: decoder step 0 logits dump (for diff-testing):\n");
        }
        // Capture step-0 cond+uncond logits (9 x V) for DIA_DUMP_DIR, and per-step for STEPLOGITS.
        const bool dia_dump_logits = (step == 0) && (dia_dump_dir != nullptr);
        const bool dia_cap = dia_dump_logits || (dia_steplogits_path != nullptr);
        std::vector<float> dump_cond, dump_uncond;
        if (dia_cap) {
            dump_cond.resize((size_t)m.n_output_heads * m.output_vocab_size);
            dump_uncond.resize((size_t)m.n_output_heads * m.output_vocab_size);
        }
        for (int h = 0; h < (int)m.n_output_heads; h++) {
            std::string hn = "logits_" + std::to_string(h);
            // Logits shape: (output_vocab_size, 1, B) = (1028, 1, 2)
            // Batch 0 = unconditional, Batch 1 = conditional (matches encoder input order)
            std::vector<float> logits_raw(m.output_vocab_size * B);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, hn.c_str()), logits_raw.data(), 0,
                                    logits_raw.size() * sizeof(float));

            // CFG filtering (matching Python _decoder_step lines 442-445):
            // 1. Compute CFG logits for top-k selection
            // 2. Apply mask to CONDITIONAL logits (not CFG) for sampling
            float* uncond = logits_raw.data();                     // batch 0
            float* cond = logits_raw.data() + m.output_vocab_size; // batch 1
            if (dia_cap) {
                std::memcpy(&dump_cond[(size_t)h * m.output_vocab_size], cond, m.output_vocab_size * sizeof(float));
                std::memcpy(&dump_uncond[(size_t)h * m.output_vocab_size], uncond, m.output_vocab_size * sizeof(float));
            }
            // CFG combine (nari-labs/Dia-1.6B OLD checkpoint: official _decoder_step does
            //   logits = cond + cfg_scale*(cond - uncond)
            // and samples from THESE combined logits — NOT from cond masked to the CFG top-k
            // (that masked scheme is the newer -0626 code and does not match this checkpoint).
            std::vector<float> final_logits(m.output_vocab_size);
            for (uint32_t i = 0; i < m.output_vocab_size; i++)
                final_logits[i] = cond[i] + p.cfg_scale * (cond[i] - uncond[i]);

            if (step == 0 && h == 0 && p.verbosity >= 1) {
                fprintf(stderr, "  cond h0 first4: %.4f %.4f %.4f %.4f argmax=%d\n", cond[0], cond[1], cond[2], cond[3],
                        (int)(std::max_element(cond, cond + m.output_vocab_size) - cond));
                fprintf(stderr, "  Python ref:     -2.8282 -1.6599 -4.0806 -0.2944 argmax=568\n");
            }

            // Dia logit masking (from Python _decoder_step):
            // 1. Mask logits > audio_vocab_size (EOS/PAD/BOS region) to -inf for valid audio range
            for (uint32_t i = m.audio_vocab_size + 1; i < m.output_vocab_size; i++) {
                final_logits[i] = -INFINITY;
            }
            // 2. For channels 1-8: also mask EOS itself (only channel 0 can signal EOS)
            if (h > 0) {
                for (uint32_t i = m.audio_vocab_size; i < m.output_vocab_size; i++) {
                    final_logits[i] = -INFINITY;
                }
            }

            // Sample token for this codebook
            uint32_t token = dia_sample_token(final_logits.data(), m.output_vocab_size,
                                              dia_greedy ? 0.0f : p.temperature, p.top_p, p.top_k, ctx->rng);
            ctx->current_audio_tokens[h] = token;
        }

        if (dia_dump_tokens) {
            fprintf(stderr, "DIA_TOK step %u:", step);
            for (int h = 0; h < (int)m.n_output_heads; h++)
                fprintf(stderr, " %u", ctx->current_audio_tokens[h]);
            fprintf(stderr, "\n");
        }

        if (dia_dump_logits) {
            FILE* fc = fopen(dia_dpath("cpp_step0_cond.f32").c_str(), "wb");
            FILE* fu = fopen(dia_dpath("cpp_step0_uncond.f32").c_str(), "wb");
            if (fc) {
                fwrite(dump_cond.data(), sizeof(float), dump_cond.size(), fc);
                fclose(fc);
            }
            if (fu) {
                fwrite(dump_uncond.data(), sizeof(float), dump_uncond.size(), fu);
                fclose(fu);
            }
            fprintf(stderr, "DIA_DUMP_DIR: wrote cpp_step0_{cond,uncond}.f32 (%zu floats each)\n", dump_cond.size());
        }
        if (dia_steplogits_path) {
            // append per step: [uncond(9*V) | cond(9*V)] to match ref (N,2,9,V)
            FILE* fs = fopen(dia_steplogits_path, "ab");
            if (fs) {
                fwrite(dump_uncond.data(), sizeof(float), dump_uncond.size(), fs);
                fwrite(dump_cond.data(), sizeof(float), dump_cond.size(), fs);
                fclose(fs);
            }
        }

        ggml_gallocr_free(alloc);
        ggml_free(ctx0);

        // EOS/delay override on the sampled tokens (end-of-sequence delay) + stop check.
        bool stop = dia_check_stopping(*ctx);

        if (dia_force) {
            // teacher-forcing: emit sampled directly (output unused for diffing)
            for (uint32_t h = 0; h < m.n_output_heads; h++)
                ctx->output_tokens.push_back(ctx->current_audio_tokens[h]);
        } else {
            // Write sampled (post-override) into the delayed buffer at step+1, with the
            // start-of-sequence BOS mask (only fill positions still unwritten), mirroring
            // Python update_one(pred, step+1, apply_mask=bos_countdown>0). Then emit gen[step+1].
            const bool apply_mask = (bos_countdown > 0);
            if (step + 1 < (uint32_t)gen_len) {
                for (uint32_t c = 0; c < m.n_output_heads; c++)
                    if (!apply_mask || gen[step + 1][c] == -1)
                        gen[step + 1][c] = (int32_t)ctx->current_audio_tokens[c];
                for (uint32_t c = 0; c < m.n_output_heads; c++)
                    ctx->output_tokens.push_back((uint32_t)gen[step + 1][c]);
            }
            if (bos_countdown > 0)
                bos_countdown--;
        }

        if (stop) {
            if (p.verbosity >= 1)
                fprintf(stderr, "dia_tts: stopped at step %u\n", step + 1);
            break;
        }

        if (p.verbosity >= 2 && (step + 1) % 100 == 0) {
            fprintf(stderr, "dia_tts: decoder step %u/%u\n", step + 1, max_gen);
        }
    }

    if (p.verbosity >= 1) {
        fprintf(stderr, "dia_tts: generated %zu raw tokens (%zu steps)\n", ctx->output_tokens.size(),
                ctx->output_tokens.size() / m.n_output_heads);
    }

    // ===================================================================
    // 4. Revert delay pattern
    // ===================================================================
    std::vector<uint32_t> filtered_codes;
    dia_revert_delay(ctx->output_tokens, filtered_codes, m);

    if (filtered_codes.empty()) {
        fprintf(stderr, "dia_tts: no valid codes after delay reversal\n");
        return nullptr;
    }

    size_t n_frames = filtered_codes.size() / m.n_output_heads;
    if (p.verbosity >= 1) {
        fprintf(stderr, "dia_tts: %zu valid code frames after delay reversal\n", n_frames);
    }

    // ===================================================================
    // 5. Decode via DAC codec
    // ===================================================================
    if (!m.has_dac) {
        fprintf(stderr, "dia_tts: DAC codec not loaded, cannot decode to audio\n");
        fprintf(stderr, "dia_tts: returning raw codes (set codec path first)\n");
        return nullptr;
    }

    // DAC decode: codes (9, n_frames) -> PCM at 44.1 kHz
    int n_samples = 0;
    float* pcm = dac_decode(ctx, filtered_codes, n_frames, &n_samples);
    if (!pcm || n_samples <= 0) {
        fprintf(stderr, "dia_tts: DAC decode failed\n");
        if (pcm)
            free(pcm);
        return nullptr;
    }

    *out_n_samples = n_samples;
    return pcm;
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
    if (ctx->model.buf_perm) {
        ggml_backend_buffer_free(ctx->model.buf_perm);
    }
    if (ctx->model.ctx_perm) {
        ggml_free(ctx->model.ctx_perm);
    }
    if (ctx->model.buf_dac) {
        ggml_backend_buffer_free(ctx->model.buf_dac);
    }
    if (ctx->model.ctx_dac) {
        ggml_free(ctx->model.ctx_dac);
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
