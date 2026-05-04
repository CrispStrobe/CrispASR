// chatterbox.cpp — ResembleAI/chatterbox TTS backend.
//
// Phase 3 of PLAN #57 — the Chatterbox pipeline:
//   1. T3 (520M Llama AR) — text → speech tokens at 25 Hz
//   2. S3Gen (CFM flow matching) — speech tokens → mel spectrogram
//   3. HiFTGenerator — mel → 24 kHz waveform
//
// This file implements:
//   - T3 model loading, graph building, AR decode with CFG
//   - Precomputed conditioning from conds.pt (built-in voice)
//   - Character tokenizer for text input
//   - Stub hooks for S3Gen + vocoder (separate file later)

#include "chatterbox.h"
#include "chatterbox_s3gen.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/attention.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ── Hyperparameters ──────────────────────────────────────────────

struct cb_t3_hp {
    std::string arch = "chatterbox"; // "chatterbox" (Llama) or "chatterbox_turbo"/"kartoffelbox" (GPT-2)
    uint32_t n_layers = 30;
    uint32_t hidden_size = 1024;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 16;
    uint32_t head_dim = 64;
    uint32_t intermediate_size = 4096;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 500000.0f;
    float rope_factor = 8.0f;
    float rope_high_freq_factor = 4.0f;
    float rope_low_freq_factor = 1.0f;
    uint32_t rope_original_max_pos = 8192;

    uint32_t text_vocab_size = 704;
    uint32_t speech_vocab_size = 8194;
    uint32_t text_pos_emb_size = 2050;
    uint32_t speech_pos_emb_size = 4100;

    uint32_t start_text_token = 255;
    uint32_t stop_text_token = 0;
    uint32_t start_speech_token = 6561;
    uint32_t stop_speech_token = 6562;
    uint32_t speech_cond_prompt_len = 150;
    uint32_t speaker_embed_size = 256;
    uint32_t perceiver_n_queries = 32;

    // Kartoffelbox GPT-2 specific
    uint32_t wpe_max_positions = 8196;
};

// ── Tensor structs ───────────────────────────────────────────────

struct cb_t3_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct cb_t3_gpt2_block {
    ggml_tensor* attn_norm_w = nullptr;   // ln_1.weight
    ggml_tensor* attn_norm_b = nullptr;   // ln_1.bias
    ggml_tensor* attn_qkv_w = nullptr;    // c_attn.weight (1024 -> 3072)
    ggml_tensor* attn_qkv_b = nullptr;    // c_attn.bias
    ggml_tensor* attn_output_w = nullptr; // c_proj.weight
    ggml_tensor* attn_output_b = nullptr; // c_proj.bias
    ggml_tensor* ffn_norm_w = nullptr;    // ln_2.weight
    ggml_tensor* ffn_norm_b = nullptr;    // ln_2.bias
    ggml_tensor* ffn_fc_w = nullptr;      // c_fc.weight (1024 -> 4096)
    ggml_tensor* ffn_fc_b = nullptr;      // c_fc.bias
    ggml_tensor* ffn_proj_w = nullptr;    // c_proj.weight (4096 -> 1024)
    ggml_tensor* ffn_proj_b = nullptr;    // c_proj.bias
};

struct cb_t3_model {
    // Custom embeddings
    ggml_tensor* text_emb_w = nullptr;       // (text_vocab, hidden)
    ggml_tensor* speech_emb_w = nullptr;     // (speech_vocab, hidden)
    ggml_tensor* text_pos_emb_w = nullptr;   // (text_pos_size, hidden)
    ggml_tensor* speech_pos_emb_w = nullptr; // (speech_pos_size, hidden)

    // Transformer blocks (Llama path)
    std::vector<cb_t3_layer> blocks;
    ggml_tensor* output_norm_w = nullptr;

    // GPT-2 blocks (Kartoffelbox path)
    std::vector<cb_t3_gpt2_block> gpt2_blocks;
    ggml_tensor* output_norm_b = nullptr; // ln_f.bias (GPT-2 only)
    ggml_tensor* wpe_w = nullptr;         // wpe.weight (GPT-2 only)

    // Heads
    ggml_tensor* speech_head_w = nullptr; // (speech_vocab, hidden)
    ggml_tensor* speech_head_b = nullptr; // (speech_vocab) — GPT-2 only
    ggml_tensor* text_head_w = nullptr;   // (text_vocab, hidden)

    // Conditioning encoder
    ggml_tensor* cond_spkr_w = nullptr;    // (hidden, spk_embed_size)
    ggml_tensor* cond_spkr_b = nullptr;    // (hidden)
    ggml_tensor* cond_emotion_w = nullptr; // (hidden, 1)

    // Perceiver
    ggml_tensor* perceiver_query = nullptr;  // (1, n_queries, hidden)
    ggml_tensor* perceiver_norm_w = nullptr; // (hidden)
    ggml_tensor* perceiver_norm_b = nullptr; // (hidden)
    ggml_tensor* perceiver_q_w = nullptr;    // (hidden, hidden)
    ggml_tensor* perceiver_q_b = nullptr;
    ggml_tensor* perceiver_k_w = nullptr; // (hidden, hidden)
    ggml_tensor* perceiver_k_b = nullptr;
    ggml_tensor* perceiver_v_w = nullptr; // (hidden, hidden)
    ggml_tensor* perceiver_v_b = nullptr;
    ggml_tensor* perceiver_out_w = nullptr; // (hidden, hidden)
    ggml_tensor* perceiver_out_b = nullptr;
};

struct cb_ve_model {
    // 3-layer LSTM
    ggml_tensor* lstm_ih_w[3] = {}; // weight_ih_l{i}: (4*hidden, input)
    ggml_tensor* lstm_hh_w[3] = {}; // weight_hh_l{i}: (4*hidden, hidden)
    ggml_tensor* lstm_ih_b[3] = {}; // bias_ih_l{i}: (4*hidden)
    ggml_tensor* lstm_hh_b[3] = {}; // bias_hh_l{i}: (4*hidden)
    ggml_tensor* proj_w = nullptr;  // (embed, hidden)
    ggml_tensor* proj_b = nullptr;  // (embed)
};

// Precomputed conditioning from conds.pt
struct cb_precomputed_conds {
    bool loaded = false;

    // T3 conditioning
    ggml_tensor* speaker_emb = nullptr;          // (1, 256)
    ggml_tensor* speech_prompt_tokens = nullptr; // (1, 150)
    float emotion_adv = 0.5f;

    // S3Gen conditioning
    ggml_tensor* gen_prompt_token = nullptr; // (1, N)
    uint32_t gen_prompt_token_len = 0;
    ggml_tensor* gen_prompt_feat = nullptr; // (1, T, 80)
    ggml_tensor* gen_embedding = nullptr;   // (1, 192)
};

// Character tokenizer for Chatterbox text input
struct cb_tokenizer {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<std::string> id_to_token;
    // GPT-2 BPE (Kartoffelbox only)
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
    bool has_bpe = false;
};

// ── Punctuation normalization (from chatterbox/tts.py) ──────────

static std::string punc_norm(const std::string& text) {
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }
    std::string s = text;

    // Capitalise first letter
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') {
        s[0] = s[0] - 'a' + 'A';
    }

    // Replace uncommon punctuation
    auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(s, "...", ", ");
    replace_all(s, ":", ",");
    replace_all(s, " - ", ", ");
    replace_all(s, ";", ", ");
    replace_all(s, " ,", ",");

    // Trim trailing spaces
    while (!s.empty() && s.back() == ' ')
        s.pop_back();

    // Add period if no sentence ender
    if (!s.empty()) {
        char last = s.back();
        if (last != '.' && last != '!' && last != '?' && last != '-' && last != ',') {
            s += '.';
        }
    }
    return s;
}

// ── Text tokenization ───────────────────────────────────────────

static std::vector<int32_t> tokenize_text(const cb_tokenizer& tok, const std::string& text) {
    // Chatterbox uses a character-level tokenizer. Each character maps
    // to a token ID via the vocabulary.
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());
    for (size_t i = 0; i < text.size(); i++) {
        std::string ch(1, text[i]);
        auto it = tok.token_to_id.find(ch);
        if (it != tok.token_to_id.end()) {
            tokens.push_back(it->second);
        } else {
            // Unknown char — skip or use space
            auto sp = tok.token_to_id.find(" ");
            if (sp != tok.token_to_id.end()) {
                tokens.push_back(sp->second);
            }
        }
    }
    return tokens;
}

// GPT-2 BPE tokenization (Kartoffelbox path)
static std::vector<int32_t> tokenize_text_bpe(const cb_tokenizer& tok, const std::string& text) {
    if (!tok.has_bpe) {
        // Fallback to character-level if no merges loaded
        return tokenize_text(tok, text);
    }
    const auto& be = core_bpe::byte_encoder();
    std::vector<int32_t> result;

    // GPT-2 pre-tokenizer: split on whitespace boundaries (simplified)
    // Each word gets a leading Ġ (U+0120) if preceded by space
    std::string buf;
    for (size_t i = 0; i < text.size(); i++) {
        if (i > 0 && text[i] != ' ' && text[i - 1] == ' ') {
            // Encode accumulated word
            if (!buf.empty()) {
                std::string encoded;
                for (uint8_t b : buf) {
                    int cp = be[b];
                    if (cp < 128)
                        encoded += (char)cp;
                    else {
                        // UTF-8 encode the codepoint
                        if (cp < 0x80)
                            encoded += (char)cp;
                        else if (cp < 0x800) {
                            encoded += (char)(0xC0 | (cp >> 6));
                            encoded += (char)(0x80 | (cp & 0x3F));
                        } else {
                            encoded += (char)(0xE0 | (cp >> 12));
                            encoded += (char)(0x80 | ((cp >> 6) & 0x3F));
                            encoded += (char)(0x80 | (cp & 0x3F));
                        }
                    }
                }
                core_bpe::bpe_one(tok.token_to_id, tok.merge_rank, encoded, result);
                buf.clear();
            }
            // Start new word with Ġ prefix (byte 0x20 = space maps to Ġ = U+0120)
        }
        if (text[i] != ' ' || i == 0 || text[i - 1] != ' ') {
            buf += text[i];
        }
    }
    // Encode remaining
    if (!buf.empty()) {
        std::string encoded;
        for (uint8_t b : buf) {
            int cp = be[b];
            if (cp < 128)
                encoded += (char)cp;
            else {
                if (cp < 0x800) {
                    encoded += (char)(0xC0 | (cp >> 6));
                    encoded += (char)(0x80 | (cp & 0x3F));
                } else {
                    encoded += (char)(0xE0 | (cp >> 12));
                    encoded += (char)(0x80 | ((cp >> 6) & 0x3F));
                    encoded += (char)(0x80 | (cp & 0x3F));
                }
            }
        }
        core_bpe::bpe_one(tok.token_to_id, tok.merge_rank, encoded, result);
    }
    return result;
}

// ── Sampler ──────────────────────────────────────────────────────

static uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
}

static float rand_uniform(uint64_t& rng) {
    return (float)(xorshift64star(rng) >> 11) / (float)(1ULL << 53);
}

static int32_t sample_token(const float* logits, int vocab_size, float temperature, float min_p, float top_p,
                            float rep_penalty, const std::vector<int32_t>& prev_tokens, uint64_t& rng) {
    std::vector<float> probs(vocab_size);

    // Apply repetition penalty
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = logits[i];
    }
    if (rep_penalty != 1.0f) {
        for (int32_t tok : prev_tokens) {
            if (tok >= 0 && tok < vocab_size) {
                if (probs[tok] > 0)
                    probs[tok] /= rep_penalty;
                else
                    probs[tok] *= rep_penalty;
            }
        }
    }

    // Temperature
    if (temperature <= 0.0f) {
        // Greedy
        return (int32_t)(std::max_element(probs.begin(), probs.end()) - probs.begin());
    }
    if (temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= temperature;
        }
    }

    // Softmax
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(probs[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // Min-p filtering
    if (min_p > 0.0f) {
        float max_prob = *std::max_element(probs.begin(), probs.end());
        float threshold = max_prob * min_p;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold)
                probs[i] = 0.0f;
        }
    }

    // Top-p filtering
    if (top_p < 1.0f) {
        // Sort indices by probability descending
        std::vector<int> indices(vocab_size);
        for (int i = 0; i < vocab_size; i++)
            indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return probs[a] > probs[b]; });
        float cumsum = 0.0f;
        for (int idx : indices) {
            cumsum += probs[idx];
            if (cumsum > top_p) {
                probs[idx] = 0.0f;
            }
        }
    }

    // Re-normalize
    sum = 0.0f;
    for (int i = 0; i < vocab_size; i++)
        sum += probs[i];
    if (sum <= 0.0f) {
        return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
    }

    // Multinomial sampling
    float r = rand_uniform(rng) * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= r)
            return i;
    }
    return vocab_size - 1;
}

// ── Bind T3 tensors ─────────────────────────────────────────────

static bool bind_t3(chatterbox_context* c);
static bool bind_t3_gpt2(chatterbox_context* c);
static bool bind_ve(chatterbox_context* c);
static void load_metadata(chatterbox_context* c, gguf_context* g);

} // namespace

// ── Context structure ───────────────────────────────────────────

struct chatterbox_context {
    chatterbox_context_params params{};
    int n_threads = 4;

    cb_t3_hp hp;
    cb_t3_model t3;
    cb_ve_model ve;
    cb_tokenizer tokenizer;
    cb_precomputed_conds conds;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for T3 (lazy-allocated) — conditioned pass
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // Second KV cache for CFG unconditioned pass
    ggml_context* kv_cfg_ctx = nullptr;
    ggml_backend_buffer_t kv_cfg_buf = nullptr;
    ggml_tensor* kv_k_cfg = nullptr;
    ggml_tensor* kv_v_cfg = nullptr;

    // S3Gen context (lazy-loaded from set_s3gen_path)
    std::string s3gen_path;
    chatterbox_s3gen_context* s3gen_ctx = nullptr;

    // RNG
    uint64_t rng_state = 0xdeadbeefcafebabeULL;

    ~chatterbox_context() {
        if (s3gen_ctx)
            chatterbox_s3gen_free(s3gen_ctx);
        if (sched)
            ggml_backend_sched_free(sched);
        if (kv_cfg_buf)
            ggml_backend_buffer_free(kv_cfg_buf);
        if (kv_cfg_ctx)
            ggml_free(kv_cfg_ctx);
        if (kv_buf)
            ggml_backend_buffer_free(kv_buf);
        if (kv_ctx)
            ggml_free(kv_ctx);
        if (ctx_w)
            ggml_free(ctx_w);
        if (buf_w)
            ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
    }
};

namespace {

// ── Metadata loading ────────────────────────────────────────────

static void load_metadata(chatterbox_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.arch = core_gguf::kv_str(g, "chatterbox.t3.arch", "chatterbox");
    hp.n_layers = core_gguf::kv_u32(g, "chatterbox.t3.n_layers", hp.n_layers);
    hp.hidden_size = core_gguf::kv_u32(g, "chatterbox.t3.hidden_size", hp.hidden_size);
    hp.n_heads = core_gguf::kv_u32(g, "chatterbox.t3.n_heads", hp.n_heads);
    hp.n_kv_heads = core_gguf::kv_u32(g, "chatterbox.t3.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = core_gguf::kv_u32(g, "chatterbox.t3.head_dim", hp.head_dim);
    hp.intermediate_size = core_gguf::kv_u32(g, "chatterbox.t3.intermediate_size", hp.intermediate_size);
    hp.rms_norm_eps = core_gguf::kv_f32(g, "chatterbox.t3.rms_norm_eps", hp.rms_norm_eps);
    hp.rope_theta = core_gguf::kv_f32(g, "chatterbox.t3.rope_theta", hp.rope_theta);
    hp.rope_factor = core_gguf::kv_f32(g, "chatterbox.t3.rope_factor", hp.rope_factor);
    hp.rope_high_freq_factor = core_gguf::kv_f32(g, "chatterbox.t3.rope_high_freq_factor", hp.rope_high_freq_factor);
    hp.rope_low_freq_factor = core_gguf::kv_f32(g, "chatterbox.t3.rope_low_freq_factor", hp.rope_low_freq_factor);
    hp.rope_original_max_pos = core_gguf::kv_u32(g, "chatterbox.t3.rope_original_max_pos", hp.rope_original_max_pos);

    hp.text_vocab_size = core_gguf::kv_u32(g, "chatterbox.t3.text_vocab_size", hp.text_vocab_size);
    hp.speech_vocab_size = core_gguf::kv_u32(g, "chatterbox.t3.speech_vocab_size", hp.speech_vocab_size);
    hp.text_pos_emb_size = core_gguf::kv_u32(g, "chatterbox.t3.text_pos_emb_size", hp.text_pos_emb_size);
    hp.speech_pos_emb_size = core_gguf::kv_u32(g, "chatterbox.t3.speech_pos_emb_size", hp.speech_pos_emb_size);

    hp.start_text_token = core_gguf::kv_u32(g, "chatterbox.t3.start_text_token", hp.start_text_token);
    hp.stop_text_token = core_gguf::kv_u32(g, "chatterbox.t3.stop_text_token", hp.stop_text_token);
    hp.start_speech_token = core_gguf::kv_u32(g, "chatterbox.t3.start_speech_token", hp.start_speech_token);
    hp.stop_speech_token = core_gguf::kv_u32(g, "chatterbox.t3.stop_speech_token", hp.stop_speech_token);
    hp.speech_cond_prompt_len = core_gguf::kv_u32(g, "chatterbox.t3.speech_cond_prompt_len", hp.speech_cond_prompt_len);
    hp.speaker_embed_size = core_gguf::kv_u32(g, "chatterbox.t3.speaker_embed_size", hp.speaker_embed_size);
    hp.perceiver_n_queries = core_gguf::kv_u32(g, "chatterbox.t3.perceiver_n_queries", hp.perceiver_n_queries);
    hp.wpe_max_positions = core_gguf::kv_u32(g, "chatterbox.t3.wpe_max_positions", hp.wpe_max_positions);

    // Precomputed conds
    c->conds.emotion_adv = core_gguf::kv_f32(g, "chatterbox.conds.emotion_adv", c->conds.emotion_adv);
    c->conds.gen_prompt_token_len = core_gguf::kv_u32(g, "chatterbox.conds.gen_prompt_token_len", 0);

    // Text tokenizer vocab — try GPT-2 BPE first, then character-level
    auto bpe_tokens = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!bpe_tokens.empty()) {
        // GPT-2 BPE tokenizer (Kartoffelbox)
        c->tokenizer.id_to_token = std::move(bpe_tokens);
        c->tokenizer.token_to_id.reserve(c->tokenizer.id_to_token.size());
        for (int i = 0; i < (int)c->tokenizer.id_to_token.size(); i++) {
            c->tokenizer.token_to_id[c->tokenizer.id_to_token[i]] = i;
        }
        auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++) {
            c->tokenizer.merge_rank[merges[i]] = i;
        }
        c->tokenizer.has_bpe = !merges.empty();
        if (c->params.verbosity >= 1 && c->tokenizer.has_bpe) {
            fprintf(stderr, "chatterbox: GPT-2 BPE tokenizer: %zu tokens, %zu merges\n",
                    c->tokenizer.id_to_token.size(), c->tokenizer.merge_rank.size());
        }
    } else {
        // Character-level tokenizer (base Chatterbox)
        auto tok_array = core_gguf::kv_str_array(g, "chatterbox.t3.text_tokens");
        if (!tok_array.empty()) {
            c->tokenizer.id_to_token = std::move(tok_array);
            c->tokenizer.token_to_id.reserve(c->tokenizer.id_to_token.size());
            for (int i = 0; i < (int)c->tokenizer.id_to_token.size(); i++) {
                c->tokenizer.token_to_id[c->tokenizer.id_to_token[i]] = i;
            }
        }
    }
}

// ── Bind T3 model tensors ───────────────────────────────────────

static bool bind_t3(chatterbox_context* c) {
    auto& m = c->t3;
    auto& ts = c->tensors;
    const char* tag = "chatterbox";

    m.text_emb_w = core_gguf::require(ts, "t3.text_emb.weight", tag);
    m.speech_emb_w = core_gguf::require(ts, "t3.speech_emb.weight", tag);
    m.text_pos_emb_w = core_gguf::require(ts, "t3.text_pos_emb.weight", tag);
    m.speech_pos_emb_w = core_gguf::require(ts, "t3.speech_pos_emb.weight", tag);
    m.output_norm_w = core_gguf::require(ts, "t3.output_norm.weight", tag);
    m.speech_head_w = core_gguf::require(ts, "t3.speech_head.weight", tag);
    m.text_head_w = core_gguf::try_get(ts, "t3.text_head.weight");

    // Conditioning
    m.cond_spkr_w = core_gguf::require(ts, "t3.cond.spkr_enc.weight", tag);
    m.cond_spkr_b = core_gguf::try_get(ts, "t3.cond.spkr_enc.bias");
    m.cond_emotion_w = core_gguf::try_get(ts, "t3.cond.emotion_adv.weight");

    // Perceiver
    m.perceiver_query = core_gguf::try_get(ts, "t3.cond.perceiver.pre_attention_query");
    m.perceiver_norm_w = core_gguf::try_get(ts, "t3.cond.perceiver.attn.norm.weight");
    m.perceiver_norm_b = core_gguf::try_get(ts, "t3.cond.perceiver.attn.norm.bias");
    m.perceiver_q_w = core_gguf::try_get(ts, "t3.cond.perceiver.attn.to_q.weight");
    m.perceiver_q_b = core_gguf::try_get(ts, "t3.cond.perceiver.attn.to_q.bias");
    m.perceiver_k_w = core_gguf::try_get(ts, "t3.cond.perceiver.attn.to_k.weight");
    m.perceiver_k_b = core_gguf::try_get(ts, "t3.cond.perceiver.attn.to_k.bias");
    m.perceiver_v_w = core_gguf::try_get(ts, "t3.cond.perceiver.attn.to_v.weight");
    m.perceiver_v_b = core_gguf::try_get(ts, "t3.cond.perceiver.attn.to_v.bias");
    m.perceiver_out_w = core_gguf::try_get(ts, "t3.cond.perceiver.attn.proj_out.weight");
    m.perceiver_out_b = core_gguf::try_get(ts, "t3.cond.perceiver.attn.proj_out.bias");

    if (!m.text_emb_w || !m.speech_emb_w || !m.text_pos_emb_w || !m.speech_pos_emb_w || !m.output_norm_w ||
        !m.speech_head_w) {
        return false;
    }

    // Transformer blocks
    m.blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = m.blocks[i];
        char key[96];
#define BIND(fld, sub)                                                                                                 \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "t3.blk.%u." sub ".weight", i);                                                \
        b.fld = core_gguf::require(ts, key, tag);                                                                      \
    } while (0)
        BIND(attn_norm_w, "attn_norm");
        BIND(attn_q_w, "attn_q");
        BIND(attn_k_w, "attn_k");
        BIND(attn_v_w, "attn_v");
        BIND(attn_output_w, "attn_output");
        BIND(ffn_norm_w, "ffn_norm");
        BIND(ffn_gate_w, "ffn_gate");
        BIND(ffn_up_w, "ffn_up");
        BIND(ffn_down_w, "ffn_down");
#undef BIND
        if (!b.attn_norm_w || !b.attn_q_w || !b.attn_k_w || !b.attn_v_w || !b.attn_output_w || !b.ffn_norm_w ||
            !b.ffn_gate_w || !b.ffn_up_w || !b.ffn_down_w) {
            fprintf(stderr, "chatterbox: missing tensor in T3 layer %u\n", i);
            return false;
        }
    }

    // Precomputed conds
    c->conds.speaker_emb = core_gguf::try_get(ts, "conds.t3.speaker_emb");
    c->conds.speech_prompt_tokens = core_gguf::try_get(ts, "conds.t3.speech_prompt_tokens");
    c->conds.gen_prompt_token = core_gguf::try_get(ts, "conds.gen.prompt_token");
    c->conds.gen_prompt_feat = core_gguf::try_get(ts, "conds.gen.prompt_feat");
    c->conds.gen_embedding = core_gguf::try_get(ts, "conds.gen.embedding");
    c->conds.loaded = (c->conds.speaker_emb != nullptr);

    return true;
}

// ── Bind GPT-2 T3 tensors (Kartoffelbox) ────────────────────────

static bool bind_t3_gpt2(chatterbox_context* c) {
    auto& m = c->t3;
    auto& ts = c->tensors;
    const char* tag = "kartoffelbox";

    m.text_emb_w = core_gguf::require(ts, "t3.text_emb.weight", tag);
    m.speech_emb_w = core_gguf::require(ts, "t3.speech_emb.weight", tag);
    m.wpe_w = core_gguf::require(ts, "t3.wpe.weight", tag);
    m.output_norm_w = core_gguf::require(ts, "t3.output_norm.weight", tag);
    m.output_norm_b = core_gguf::require(ts, "t3.output_norm.bias", tag);
    m.speech_head_w = core_gguf::require(ts, "t3.speech_head.weight", tag);
    m.speech_head_b = core_gguf::try_get(ts, "t3.speech_head.bias");
    m.text_head_w = core_gguf::try_get(ts, "t3.text_head.weight");

    // Conditioning
    m.cond_spkr_w = core_gguf::try_get(ts, "t3.cond.spkr_enc.weight");
    m.cond_spkr_b = core_gguf::try_get(ts, "t3.cond.spkr_enc.bias");

    if (!m.text_emb_w || !m.speech_emb_w || !m.wpe_w || !m.output_norm_w || !m.output_norm_b || !m.speech_head_w) {
        return false;
    }

    // GPT-2 transformer blocks
    m.gpt2_blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = m.gpt2_blocks[i];
        char key[96];
#define BIND_GPT2(fld, sub)                                                                                            \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "t3.blk.%u." sub, i);                                                          \
        b.fld = core_gguf::require(ts, key, tag);                                                                      \
    } while (0)
        BIND_GPT2(attn_norm_w, "attn_norm.weight");
        BIND_GPT2(attn_norm_b, "attn_norm.bias");
        BIND_GPT2(attn_qkv_w, "attn_qkv.weight");
        BIND_GPT2(attn_qkv_b, "attn_qkv.bias");
        BIND_GPT2(attn_output_w, "attn_output.weight");
        BIND_GPT2(attn_output_b, "attn_output.bias");
        BIND_GPT2(ffn_norm_w, "ffn_norm.weight");
        BIND_GPT2(ffn_norm_b, "ffn_norm.bias");
        BIND_GPT2(ffn_fc_w, "ffn_fc.weight");
        BIND_GPT2(ffn_fc_b, "ffn_fc.bias");
        BIND_GPT2(ffn_proj_w, "ffn_proj.weight");
        BIND_GPT2(ffn_proj_b, "ffn_proj.bias");
#undef BIND_GPT2
        if (!b.attn_norm_w || !b.attn_norm_b || !b.attn_qkv_w || !b.attn_qkv_b || !b.attn_output_w ||
            !b.attn_output_b || !b.ffn_norm_w || !b.ffn_norm_b || !b.ffn_fc_w || !b.ffn_fc_b || !b.ffn_proj_w ||
            !b.ffn_proj_b) {
            fprintf(stderr, "kartoffelbox: missing tensor in GPT-2 layer %u\n", i);
            return false;
        }
    }

    // Precomputed conds (optional for Kartoffelbox)
    c->conds.speaker_emb = core_gguf::try_get(ts, "conds.t3.speaker_emb");
    c->conds.speech_prompt_tokens = core_gguf::try_get(ts, "conds.t3.speech_prompt_tokens");
    c->conds.gen_prompt_token = core_gguf::try_get(ts, "conds.gen.prompt_token");
    c->conds.gen_prompt_feat = core_gguf::try_get(ts, "conds.gen.prompt_feat");
    c->conds.gen_embedding = core_gguf::try_get(ts, "conds.gen.embedding");
    c->conds.loaded = (c->conds.speaker_emb != nullptr);

    return true;
}

// ── Bind VE tensors ─────────────────────────────────────────────

static bool bind_ve(chatterbox_context* c) {
    auto& ve = c->ve;
    auto& ts = c->tensors;

    for (int i = 0; i < 3; i++) {
        char key[64];
        std::snprintf(key, sizeof(key), "ve.lstm.weight_ih_l%d", i);
        ve.lstm_ih_w[i] = core_gguf::try_get(ts, key);
        std::snprintf(key, sizeof(key), "ve.lstm.weight_hh_l%d", i);
        ve.lstm_hh_w[i] = core_gguf::try_get(ts, key);
        std::snprintf(key, sizeof(key), "ve.lstm.bias_ih_l%d", i);
        ve.lstm_ih_b[i] = core_gguf::try_get(ts, key);
        std::snprintf(key, sizeof(key), "ve.lstm.bias_hh_l%d", i);
        ve.lstm_hh_b[i] = core_gguf::try_get(ts, key);
    }
    ve.proj_w = core_gguf::try_get(ts, "ve.proj.weight");
    ve.proj_b = core_gguf::try_get(ts, "ve.proj.bias");
    return true; // VE is optional
}

// ── KV cache allocation ─────────────────────────────────────────

static bool kv_alloc(chatterbox_context* c, int max_ctx) {
    if (c->kv_ctx && max_ctx <= c->kv_max_ctx)
        return true;

    // Free existing
    if (c->kv_buf)
        ggml_backend_buffer_free(c->kv_buf);
    if (c->kv_ctx)
        ggml_free(c->kv_ctx);
    c->kv_buf = nullptr;
    c->kv_ctx = nullptr;

    const auto& hp = c->hp;
    const int hd = hp.head_dim;
    const int n_kv = hp.n_kv_heads;
    const int nl = hp.n_layers;
    c->kv_max_ctx = max_ctx;

    // KV shape: (head_dim, max_ctx, n_kv_heads, n_layers)
    struct ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    c->kv_ctx = ggml_init(ip);
    if (!c->kv_ctx)
        return false;

    // PLAN #60e + #69e: per-half KV dtype. CRISPASR_KV_QUANT sets both,
    // CRISPASR_KV_QUANT_{K,V} override per half. Default f16/f16.
    // Chatterbox uses core_attn::kv_self_attn for the cache write/read,
    // so quant types are safe (the helper switches to ggml_set_rows for
    // quant writes and ggml_cast(F32) for quant reads).
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("chatterbox");
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);

    // PLAN #69b: optional KV-on-CPU spill for VRAM-tight users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(c->backend, c->backend_cpu, "chatterbox");
    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, kv_backend);
    if (!c->kv_buf) {
        fprintf(stderr, "chatterbox: failed to allocate KV cache\n");
        ggml_free(c->kv_ctx);
        c->kv_ctx = nullptr;
        return false;
    }

    size_t kb = ggml_nbytes(c->kv_k);
    size_t vb = ggml_nbytes(c->kv_v);

    // Also allocate CFG unconditioned KV cache (same K/V split as the
    // primary cache — they're attended in lockstep).
    if (c->kv_cfg_buf)
        ggml_backend_buffer_free(c->kv_cfg_buf);
    if (c->kv_cfg_ctx)
        ggml_free(c->kv_cfg_ctx);
    struct ggml_init_params ip2 = {2 * ggml_tensor_overhead(), nullptr, true};
    c->kv_cfg_ctx = ggml_init(ip2);
    if (c->kv_cfg_ctx) {
        c->kv_k_cfg = ggml_new_tensor_4d(c->kv_cfg_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
        c->kv_v_cfg = ggml_new_tensor_4d(c->kv_cfg_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
        c->kv_cfg_buf = ggml_backend_alloc_ctx_tensors(c->kv_cfg_ctx, kv_backend);
    }

    if (c->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: kv cache %d MiB k=%s v=%s (on %s, hd=%d max=%d n_kv=%d nl=%d) + CFG\n",
                (int)((kb + vb) * 2 / 1048576), ggml_type_name(kv_pair.k), ggml_type_name(kv_pair.v),
                kv_backend == c->backend_cpu ? "cpu" : "gpu", hd, max_ctx, n_kv, nl);
    }
    return true;
}

// ── T3 graph building ───────────────────────────────────────────

// Llama-520M transformer: inputs_embeds (D, T) → speech logits (speech_vocab,)
// Uses core_attn::kv_self_attn for each layer, matching orpheus pattern.
static ggml_cgraph* build_graph_t3_kv(chatterbox_context* c, int n_past, int n_tokens, ggml_tensor* use_kv_k = nullptr,
                                      ggml_tensor* use_kv_v = nullptr) {
    // Use provided KV tensors or default to c->kv_k/kv_v
    if (!use_kv_k)
        use_kv_k = c->kv_k;
    if (!use_kv_v)
        use_kv_v = c->kv_v;
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_size;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(c->kv_k && c->kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.rope_original_max_pos,
        /*rope_theta*/ hp.rope_theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->t3.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions,
                                    (T == 1) ? nullptr : causal_mask, use_kv_k, use_kv_v, (int)il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->t3.output_norm_w);
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, c->t3.speech_head_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the T3 transformer on pre-built embeddings. Returns logits (speech_vocab,).
static float* run_t3_kv(chatterbox_context* c, const float* embeds, int n_tokens, int n_past,
                        ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "chatterbox: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return nullptr;
    }
    const int D = (int)c->hp.hidden_size;
    const int vocab = (int)c->hp.speech_vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
    }

    ggml_cgraph* gf = build_graph_t3_kv(c, n_past, n_tokens, use_kv_k, use_kv_v);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "chatterbox: failed to alloc T3 graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)D * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "chatterbox: T3 compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// Build the conditioning + text + speech_start embedding on CPU.
// Returns concatenated embeddings (D, cond_len + text_len + 1).
static std::vector<float> build_prefill_embeds(chatterbox_context* c, const std::vector<int32_t>& text_tokens) {
    const int D = (int)c->hp.hidden_size;
    const auto& m = c->t3;

    // We'll compute conditioning + text + speech_start embeddings on CPU
    // by reading weight tensors directly.

    // 1. Speaker embedding projection: spkr_enc(speaker_emb) → (D,)
    // For precomputed conds, speaker_emb is (1, 256)
    std::vector<float> spkr_emb(c->hp.speaker_embed_size);
    ggml_backend_tensor_get(c->conds.speaker_emb, spkr_emb.data(), 0, spkr_emb.size() * sizeof(float));

    // Project: W (D, 256) × emb (256,) + bias (D,)
    std::vector<float> spkr_proj(D, 0.0f);
    {
        std::vector<float> w(D * c->hp.speaker_embed_size);
        ggml_backend_tensor_get(m.cond_spkr_w, w.data(), 0, w.size() * sizeof(float));
        for (int i = 0; i < D; i++) {
            float sum = 0.0f;
            for (int j = 0; j < (int)c->hp.speaker_embed_size; j++) {
                sum += w[i * c->hp.speaker_embed_size + j] * spkr_emb[j];
            }
            spkr_proj[i] = sum;
        }
        if (m.cond_spkr_b) {
            std::vector<float> bias(D);
            ggml_backend_tensor_get(m.cond_spkr_b, bias.data(), 0, D * sizeof(float));
            for (int i = 0; i < D; i++)
                spkr_proj[i] += bias[i];
        }
    }

    // 2. Perceiver: cross-attend from 32 learned queries to 150 speech prompt embeddings
    // Output: 32 conditioning tokens of dimension D
    std::vector<float> perceiver_out; // (32 * D) if perceiver is available
    int perceiver_len = 0;
    if (c->conds.speech_prompt_tokens && m.perceiver_query && m.perceiver_q_w && m.perceiver_k_w && m.perceiver_v_w) {
        int n_prompt = (int)c->conds.speech_prompt_tokens->ne[0];
        int n_q = (int)c->hp.perceiver_n_queries; // 32
        int n_heads = 4;
        int hd = D / n_heads; // 256

        // Read embedding tables
        std::vector<float> speech_emb_tab(c->hp.speech_vocab_size * D);
        ggml_backend_tensor_get(m.speech_emb_w, speech_emb_tab.data(), 0, speech_emb_tab.size() * sizeof(float));
        std::vector<float> speech_pos_tab(c->hp.speech_pos_emb_size * D);
        ggml_backend_tensor_get(m.speech_pos_emb_w, speech_pos_tab.data(), 0, speech_pos_tab.size() * sizeof(float));

        // Read prompt token IDs
        std::vector<int32_t> prompt_ids(n_prompt);
        ggml_backend_tensor_get(c->conds.speech_prompt_tokens, prompt_ids.data(), 0, n_prompt * sizeof(int32_t));

        // Embed prompt: speech_emb(tok) + speech_pos_emb(pos)
        std::vector<float> prompt_emb(n_prompt * D, 0.0f);
        for (int i = 0; i < n_prompt; i++) {
            int tok = prompt_ids[i];
            if (tok < 0 || tok >= (int)c->hp.speech_vocab_size)
                tok = 0;
            for (int j = 0; j < D; j++) {
                prompt_emb[i * D + j] = speech_emb_tab[tok * D + j] + speech_pos_tab[i * D + j];
            }
        }

        // Read perceiver weights
        std::vector<float> query(n_q * D);
        ggml_backend_tensor_get(m.perceiver_query, query.data(), 0, n_q * D * sizeof(float));

        // Helper: read Linear weight (out_dim, in_dim) and optional bias (out_dim)
        auto read_linear = [&](ggml_tensor* w_t, ggml_tensor* b_t, int out_d, int in_d) {
            std::vector<float> w(out_d * in_d);
            std::vector<float> b(out_d, 0.0f);
            ggml_backend_tensor_get(w_t, w.data(), 0, w.size() * sizeof(float));
            if (b_t)
                ggml_backend_tensor_get(b_t, b.data(), 0, b.size() * sizeof(float));
            return std::make_pair(w, b);
        };

        // Note: avoid `auto [qw, qb] = ...` structured bindings here.
        // The `mha` lambda below uses `[&]` default capture and references
        // these names; clang under C++17 (baseline for ubuntu-22-clang
        // matrix builds) rejects with "reference to local binding 'qw'
        // declared in enclosing function". Fixed in C++20 (P1091) but
        // we're on C++17. Plain locals + const refs work everywhere.
        auto q_pair = read_linear(m.perceiver_q_w, m.perceiver_q_b, D, D);
        auto k_pair = read_linear(m.perceiver_k_w, m.perceiver_k_b, D, D);
        auto v_pair = read_linear(m.perceiver_v_w, m.perceiver_v_b, D, D);
        auto o_pair = read_linear(m.perceiver_out_w, m.perceiver_out_b, D, D);
        const auto& qw = q_pair.first;
        const auto& qb = q_pair.second;
        const auto& kw = k_pair.first;
        const auto& kb = k_pair.second;
        const auto& vw = v_pair.first;
        const auto& vb = v_pair.second;
        const auto& ow = o_pair.first;
        const auto& ob = o_pair.second;

        // Read LayerNorm
        std::vector<float> norm_w(D, 1.0f), norm_b(D, 0.0f);
        if (m.perceiver_norm_w)
            ggml_backend_tensor_get(m.perceiver_norm_w, norm_w.data(), 0, D * sizeof(float));
        if (m.perceiver_norm_b)
            ggml_backend_tensor_get(m.perceiver_norm_b, norm_b.data(), 0, D * sizeof(float));

        // LayerNorm helper (eps=1e-5)
        auto layer_norm = [&](const float* in, float* out, int len) {
            float mean = 0.0f;
            for (int i = 0; i < len; i++)
                mean += in[i];
            mean /= len;
            float var = 0.0f;
            for (int i = 0; i < len; i++) {
                float d = in[i] - mean;
                var += d * d;
            }
            var /= len;
            float inv_std = 1.0f / std::sqrt(var + 1e-5f);
            for (int i = 0; i < len; i++) {
                out[i] = (in[i] - mean) * inv_std * norm_w[i] + norm_b[i];
            }
        };

        // Matrix multiply: out[M,N] = W[M,K] × in[K,N] + bias[M]
        auto matmul_bias = [](const float* W, const float* in, const float* bias, float* out, int M, int K, int N) {
            for (int n = 0; n < N; n++) {
                for (int m = 0; m < M; m++) {
                    float sum = bias ? bias[m] : 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += W[m * K + k] * in[n * K + k];
                    }
                    out[n * M + m] = sum;
                }
            }
        };

        // Multi-head attention: Q(n_q, D) × K(n_kv, D)^T → softmax → × V(n_kv, D) → O
        auto mha = [&](const float* Q_in, int n_q_len, const float* KV_in, int n_kv_len, float* out_buf) {
            // Project Q, K, V
            std::vector<float> Q_proj(n_q_len * D), K_proj(n_kv_len * D), V_proj(n_kv_len * D);
            std::vector<float> Q_norm(n_q_len * D), KV_norm(n_kv_len * D);

            // LayerNorm both inputs
            for (int i = 0; i < n_q_len; i++)
                layer_norm(&Q_in[i * D], &Q_norm[i * D], D);
            for (int i = 0; i < n_kv_len; i++)
                layer_norm(&KV_in[i * D], &KV_norm[i * D], D);

            matmul_bias(qw.data(), Q_norm.data(), qb.data(), Q_proj.data(), D, D, n_q_len);
            matmul_bias(kw.data(), KV_norm.data(), kb.data(), K_proj.data(), D, D, n_kv_len);
            matmul_bias(vw.data(), KV_norm.data(), vb.data(), V_proj.data(), D, D, n_kv_len);

            float scale = 1.0f / std::sqrt((float)hd);

            // Per-head attention
            std::vector<float> attn_out(n_q_len * D, 0.0f);
            for (int h = 0; h < n_heads; h++) {
                // QK^T
                for (int qi = 0; qi < n_q_len; qi++) {
                    // Softmax numerator
                    std::vector<float> scores(n_kv_len);
                    float max_s = -1e30f;
                    for (int ki = 0; ki < n_kv_len; ki++) {
                        float dot = 0.0f;
                        for (int d = 0; d < hd; d++) {
                            dot += Q_proj[qi * D + h * hd + d] * K_proj[ki * D + h * hd + d];
                        }
                        scores[ki] = dot * scale;
                        if (scores[ki] > max_s)
                            max_s = scores[ki];
                    }
                    // Softmax
                    float sum_exp = 0.0f;
                    for (int ki = 0; ki < n_kv_len; ki++) {
                        scores[ki] = std::exp(scores[ki] - max_s);
                        sum_exp += scores[ki];
                    }
                    for (int ki = 0; ki < n_kv_len; ki++)
                        scores[ki] /= sum_exp;
                    // Attention × V
                    for (int d = 0; d < hd; d++) {
                        float val = 0.0f;
                        for (int ki = 0; ki < n_kv_len; ki++) {
                            val += scores[ki] * V_proj[ki * D + h * hd + d];
                        }
                        attn_out[qi * D + h * hd + d] = val;
                    }
                }
            }

            // Output projection + residual
            std::vector<float> proj(n_q_len * D);
            matmul_bias(ow.data(), attn_out.data(), ob.data(), proj.data(), D, D, n_q_len);
            for (int i = 0; i < n_q_len * D; i++) {
                out_buf[i] = Q_in[i] + proj[i];
            }
        };

        // Pass 1: cross-attention (query attends to prompt_emb)
        std::vector<float> cross_out(n_q * D);
        mha(query.data(), n_q, prompt_emb.data(), n_prompt, cross_out.data());

        // Pass 2: self-attention (cross_out attends to itself)
        perceiver_out.resize(n_q * D);
        mha(cross_out.data(), n_q, cross_out.data(), n_q, perceiver_out.data());

        perceiver_len = n_q;
        if (c->params.verbosity >= 2) {
            fprintf(stderr, "chatterbox: perceiver → %d conditioning tokens\n", perceiver_len);
        }
    }

    int cond_len = 1 + perceiver_len; // spkr(1) + perceiver(32)

    // 3. Emotion adversarial: emotion_adv_fc(emotion_scalar) → (D,)
    std::vector<float> emotion_proj(D, 0.0f);
    if (m.cond_emotion_w) {
        std::vector<float> w(D);
        ggml_backend_tensor_get(m.cond_emotion_w, w.data(), 0, D * sizeof(float));
        for (int i = 0; i < D; i++) {
            emotion_proj[i] = w[i] * c->conds.emotion_adv;
        }
        cond_len++;
    }

    // 4. Text embeddings: text_emb(token) + text_pos_emb(pos)
    int text_len = (int)text_tokens.size();

    // 5. Speech start embedding: speech_emb(start_token) + speech_pos_emb(0)
    int speech_start_len = 1;

    int total_len = cond_len + text_len + speech_start_len;
    std::vector<float> embeds(total_len * D, 0.0f);

    // Place conditioning: [spkr(1), perceiver(32), emotion(1)]
    int pos = 0;
    std::memcpy(&embeds[pos * D], spkr_proj.data(), D * sizeof(float));
    pos++;
    // Perceiver output
    if (perceiver_len > 0) {
        std::memcpy(&embeds[pos * D], perceiver_out.data(), perceiver_len * D * sizeof(float));
        pos += perceiver_len;
    }
    if (m.cond_emotion_w) {
        std::memcpy(&embeds[pos * D], emotion_proj.data(), D * sizeof(float));
        pos++;
    }

    // Place text embeddings: text_emb + text_pos_emb
    {
        std::vector<float> text_emb_table(c->hp.text_vocab_size * D);
        ggml_backend_tensor_get(m.text_emb_w, text_emb_table.data(), 0, text_emb_table.size() * sizeof(float));
        std::vector<float> text_pos_table(c->hp.text_pos_emb_size * D);
        ggml_backend_tensor_get(m.text_pos_emb_w, text_pos_table.data(), 0, text_pos_table.size() * sizeof(float));

        for (int i = 0; i < text_len; i++) {
            int tok = text_tokens[i];
            if (tok < 0 || tok >= (int)c->hp.text_vocab_size)
                tok = 0;
            for (int j = 0; j < D; j++) {
                embeds[(pos + i) * D + j] = text_emb_table[tok * D + j] + text_pos_table[i * D + j];
            }
        }
        pos += text_len;
    }

    // Place speech start embedding: speech_emb(start_token) + speech_pos_emb(0)
    {
        std::vector<float> speech_emb_table(c->hp.speech_vocab_size * D);
        ggml_backend_tensor_get(m.speech_emb_w, speech_emb_table.data(), 0, speech_emb_table.size() * sizeof(float));
        std::vector<float> speech_pos_table(c->hp.speech_pos_emb_size * D);
        ggml_backend_tensor_get(m.speech_pos_emb_w, speech_pos_table.data(), 0,
                                speech_pos_table.size() * sizeof(float));

        int start_tok = (int)c->hp.start_speech_token;
        for (int j = 0; j < D; j++) {
            embeds[pos * D + j] = speech_emb_table[start_tok * D + j] + speech_pos_table[0 * D + j];
        }
    }

    return embeds;
}

// Build embedding for a single speech token at a given position.
static std::vector<float> build_speech_token_embed(chatterbox_context* c, int32_t token_id,
                                                   int speech_pos // position index for speech_pos_emb
) {
    const int D = (int)c->hp.hidden_size;
    std::vector<float> embed(D);

    // speech_emb(token) + speech_pos_emb(pos)
    std::vector<float> tok_emb(D);
    ggml_backend_tensor_get(c->t3.speech_emb_w, tok_emb.data(), (size_t)token_id * D * sizeof(float),
                            D * sizeof(float));
    std::vector<float> pos_emb(D);
    ggml_backend_tensor_get(c->t3.speech_pos_emb_w, pos_emb.data(), (size_t)speech_pos * D * sizeof(float),
                            D * sizeof(float));
    for (int j = 0; j < D; j++) {
        embed[j] = tok_emb[j] + pos_emb[j];
    }
    return embed;
}

// ── GPT-2 T3 graph building (Kartoffelbox) ─────────────────────

// GPT-2 transformer: inputs_embeds (D, T) → speech logits (speech_vocab,)
// Uses KV cache for autoregressive decoding. Position comes from WPE lookup (no RoPE).
static ggml_cgraph* build_graph_t3_gpt2_kv(chatterbox_context* c, int n_past, int n_tokens,
                                           ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
    if (!use_kv_k)
        use_kv_k = c->kv_k;
    if (!use_kv_v)
        use_kv_v = c->kv_v;
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_size;
    const int n_h = (int)hp.n_heads;
    const int hd = (int)hp.head_dim;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const float ln_eps = 1e-5f;
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(c->kv_k && c->kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->t3.gpt2_blocks[il];
        ggml_tensor* residual = cur;

        // Pre-attention LayerNorm
        ggml_tensor* x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);
        x = ggml_add(ctx0, x, b.attn_norm_b);

        // Fused QKV: x @ c_attn_w + c_attn_b → (3*D, T)
        ggml_tensor* qkv = ggml_mul_mat(ctx0, b.attn_qkv_w, x);
        qkv = ggml_add(ctx0, qkv, b.attn_qkv_b);

        // Split into Q, K, V — each (D, T)
        const size_t ts = ggml_type_size(qkv->type);
        ggml_tensor* Q = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor* K = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], D * ts);
        ggml_tensor* V = ggml_view_2d(ctx0, qkv, D, T, qkv->nb[1], 2 * D * ts);
        if (T > 1) {
            Q = ggml_cont(ctx0, Q);
            K = ggml_cont(ctx0, K);
            V = ggml_cont(ctx0, V);
        }

        // Reshape to (hd, n_h, T)
        Q = ggml_reshape_3d(ctx0, Q, hd, n_h, T);
        K = ggml_reshape_3d(ctx0, K, hd, n_h, T);
        V = ggml_reshape_3d(ctx0, V, hd, n_h, T);

        // No RoPE for GPT-2 — positions encoded via WPE

        // Permute new K/V to (hd, T, n_h) for cache write
        ggml_tensor* K_new_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
        ggml_tensor* V_new_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

        // Write into KV cache at [n_past, n_past+T)
        const int n_kv = n_h; // MHA — n_kv_heads == n_heads
        ggml_tensor* k_view =
            ggml_view_4d(ctx0, use_kv_k, hd, T, n_kv, 1, use_kv_k->nb[1], use_kv_k->nb[2], use_kv_k->nb[3],
                         (size_t)il * use_kv_k->nb[3] + (size_t)n_past * use_kv_k->nb[1]);
        ggml_tensor* v_view =
            ggml_view_4d(ctx0, use_kv_v, hd, T, n_kv, 1, use_kv_v->nb[1], use_kv_v->nb[2], use_kv_v->nb[3],
                         (size_t)il * use_kv_v->nb[3] + (size_t)n_past * use_kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));

        // Read full K/V history
        ggml_tensor* k_layer_view =
            ggml_view_3d(ctx0, use_kv_k, hd, Lk, n_kv, use_kv_k->nb[1], use_kv_k->nb[2], (size_t)il * use_kv_k->nb[3]);
        ggml_tensor* v_layer_view =
            ggml_view_3d(ctx0, use_kv_v, hd, Lk, n_kv, use_kv_v->nb[1], use_kv_v->nb[2], (size_t)il * use_kv_v->nb[3]);
        ggml_tensor* Kfull = ggml_cont(ctx0, k_layer_view);
        ggml_tensor* Vfull = ggml_cont(ctx0, v_layer_view);

        // Permute Q to (hd, T, n_h)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

        // Flash attention
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (T == 1) ? nullptr : causal_mask, attn_scale,
                                                /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        // Output projection + residual
        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        attn = ggml_add(ctx0, attn, b.attn_output_b);
        cur = ggml_add(ctx0, residual, attn);

        // FFN
        residual = cur;
        x = ggml_norm(ctx0, cur, ln_eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        x = ggml_add(ctx0, x, b.ffn_norm_b);

        // GELU FFN: c_fc → gelu → c_proj
        ggml_tensor* mlp = ggml_mul_mat(ctx0, b.ffn_fc_w, x);
        mlp = ggml_add(ctx0, mlp, b.ffn_fc_b);
        mlp = ggml_gelu(ctx0, mlp);
        mlp = ggml_mul_mat(ctx0, b.ffn_proj_w, mlp);
        mlp = ggml_add(ctx0, mlp, b.ffn_proj_b);

        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, ln_eps);
    cur = ggml_mul(ctx0, cur, c->t3.output_norm_w);
    cur = ggml_add(ctx0, cur, c->t3.output_norm_b);

    // Take last token for prefill
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }

    // Speech head
    cur = ggml_mul_mat(ctx0, c->t3.speech_head_w, cur);
    if (c->t3.speech_head_b) {
        cur = ggml_add(ctx0, cur, c->t3.speech_head_b);
    }
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the GPT-2 T3 transformer. Returns logits (speech_vocab,).
static float* run_t3_gpt2_kv(chatterbox_context* c, const float* embeds, int n_tokens, int n_past,
                             ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "kartoffelbox: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return nullptr;
    }
    const int D = (int)c->hp.hidden_size;
    const int vocab = (int)c->hp.speech_vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        mask.assign((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = n_past + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
    }

    ggml_cgraph* gf = build_graph_t3_gpt2_kv(c, n_past, n_tokens, use_kv_k, use_kv_v);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kartoffelbox: failed to alloc T3 GPT-2 graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)D * n_tokens * sizeof(float));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kartoffelbox: T3 GPT-2 compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// Build the conditioning + text + speech_start embedding for Kartoffelbox (GPT-2).
// WPE is added to token embeddings. No perceiver, no emotion.
static std::vector<float> build_prefill_embeds_gpt2(chatterbox_context* c, const std::vector<int32_t>& text_tokens) {
    const int D = (int)c->hp.hidden_size;
    const auto& m = c->t3;

    // Read WPE table
    int wpe_size = (int)c->hp.wpe_max_positions;
    std::vector<float> wpe_table((size_t)wpe_size * D);
    ggml_backend_tensor_get(m.wpe_w, wpe_table.data(), 0, wpe_table.size() * sizeof(float));

    // 1. Conditioning: speaker_emb projection + speech prompt token embeddings
    // Python: cond_enc(t3_cond) returns [spkr_proj, speech_emb(cond_tokens)]
    // For GPT-2 (Turbo): no perceiver, no text/speech pos embeddings, no emotion
    int cond_len = 0;
    std::vector<float> spkr_proj(D, 0.0f);
    std::vector<float> cond_speech_embs; // speech prompt conditioning embeddings

    if (c->conds.loaded && m.cond_spkr_w) {
        // Speaker embedding projection → 1 token
        std::vector<float> spkr_emb(c->hp.speaker_embed_size);
        ggml_backend_tensor_get(c->conds.speaker_emb, spkr_emb.data(), 0, spkr_emb.size() * sizeof(float));

        std::vector<float> w(D * c->hp.speaker_embed_size);
        ggml_backend_tensor_get(m.cond_spkr_w, w.data(), 0, w.size() * sizeof(float));
        for (int i = 0; i < D; i++) {
            float sum = 0.0f;
            for (int j = 0; j < (int)c->hp.speaker_embed_size; j++) {
                sum += w[i * c->hp.speaker_embed_size + j] * spkr_emb[j];
            }
            spkr_proj[i] = sum;
        }
        if (m.cond_spkr_b) {
            std::vector<float> bias(D);
            ggml_backend_tensor_get(m.cond_spkr_b, bias.data(), 0, D * sizeof(float));
            for (int i = 0; i < D; i++)
                spkr_proj[i] += bias[i];
        }
        cond_len = 1;

        // Speech prompt conditioning tokens → N embeddings (no pos emb for GPT-2)
        if (c->conds.speech_prompt_tokens) {
            int n_prompt = (int)c->conds.speech_prompt_tokens->ne[0];
            std::vector<int32_t> prompt_toks(n_prompt);
            ggml_backend_tensor_get(c->conds.speech_prompt_tokens, prompt_toks.data(), 0, n_prompt * sizeof(int32_t));

            std::vector<float> speech_emb_table(c->hp.speech_vocab_size * D);
            ggml_backend_tensor_get(m.speech_emb_w, speech_emb_table.data(), 0,
                                    speech_emb_table.size() * sizeof(float));

            cond_speech_embs.resize((size_t)n_prompt * D);
            for (int i = 0; i < n_prompt; i++) {
                int tok = std::max(0, std::min((int)c->hp.speech_vocab_size - 1, (int)prompt_toks[i]));
                for (int j = 0; j < D; j++) {
                    cond_speech_embs[i * D + j] = speech_emb_table[tok * D + j];
                }
            }
            cond_len += n_prompt;
        }
    }

    int text_len = (int)text_tokens.size();
    int speech_start_len = 1;
    int total_len = cond_len + text_len + speech_start_len;

    std::vector<float> embeds((size_t)total_len * D, 0.0f);

    // Read embedding tables
    std::vector<float> text_emb_table(c->hp.text_vocab_size * D);
    ggml_backend_tensor_get(m.text_emb_w, text_emb_table.data(), 0, text_emb_table.size() * sizeof(float));
    std::vector<float> speech_emb_table(c->hp.speech_vocab_size * D);
    ggml_backend_tensor_get(m.speech_emb_w, speech_emb_table.data(), 0, speech_emb_table.size() * sizeof(float));

    int pos = 0;
    int wpe_pos = 0;

    // Place conditioning: [speaker_proj, cond_speech_embs...]
    if (cond_len > 0) {
        // Speaker projection at position 0
        for (int j = 0; j < D; j++) {
            embeds[pos * D + j] = spkr_proj[j] + wpe_table[wpe_pos * D + j];
        }
        pos++;
        wpe_pos++;

        // Speech conditioning tokens (if any)
        int n_cond_speech = (int)(cond_speech_embs.size() / D);
        for (int i = 0; i < n_cond_speech; i++) {
            for (int j = 0; j < D; j++) {
                embeds[(pos + i) * D + j] = cond_speech_embs[i * D + j] + wpe_table[(wpe_pos + i) * D + j];
            }
        }
        pos += n_cond_speech;
        wpe_pos += n_cond_speech;
    }

    // Place text embeddings: text_emb(tok) + wpe(pos)
    for (int i = 0; i < text_len; i++) {
        int tok = text_tokens[i];
        if (tok < 0 || tok >= (int)c->hp.text_vocab_size)
            tok = 0;
        for (int j = 0; j < D; j++) {
            embeds[(pos + i) * D + j] = text_emb_table[tok * D + j] + wpe_table[(wpe_pos + i) * D + j];
        }
    }
    pos += text_len;
    wpe_pos += text_len;

    // Place speech start embedding: speech_emb(start_token) + wpe(pos)
    {
        int start_tok = (int)c->hp.start_speech_token;
        if (start_tok >= (int)c->hp.speech_vocab_size)
            start_tok = 0;
        for (int j = 0; j < D; j++) {
            embeds[pos * D + j] = speech_emb_table[start_tok * D + j] + wpe_table[wpe_pos * D + j];
        }
    }

    return embeds;
}

// Build embedding for a single speech token with WPE (GPT-2 Kartoffelbox).
static std::vector<float> build_speech_token_embed_gpt2(chatterbox_context* c, int32_t token_id, int abs_pos) {
    const int D = (int)c->hp.hidden_size;
    std::vector<float> embed(D);

    std::vector<float> tok_emb(D);
    ggml_backend_tensor_get(c->t3.speech_emb_w, tok_emb.data(), (size_t)token_id * D * sizeof(float),
                            D * sizeof(float));
    std::vector<float> pos_emb(D);
    ggml_backend_tensor_get(c->t3.wpe_w, pos_emb.data(), (size_t)abs_pos * D * sizeof(float), D * sizeof(float));
    for (int j = 0; j < D; j++) {
        embed[j] = tok_emb[j] + pos_emb[j];
    }
    return embed;
}

} // namespace

// ── Public C ABI ────────────────────────────────────────────────

extern "C" struct chatterbox_context_params chatterbox_context_default_params(void) {
    chatterbox_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.8f;
    p.cfg_weight = 0.5f;
    p.exaggeration = 0.5f;
    p.repetition_penalty = 1.2f;
    p.min_p = 0.05f;
    p.top_p = 1.0f;
    p.max_speech_tokens = 1000;
    p.cfm_steps = 10;
    return p;
}

extern "C" struct chatterbox_context* chatterbox_init_from_file(const char* path_model,
                                                                struct chatterbox_context_params params) {
    auto* c = new chatterbox_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    const bool is_gpt2 = (c->hp.arch == "chatterbox_turbo" || c->hp.arch == "kartoffelbox");

    if (params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: arch=%s T3 %uL d=%u h=%u hd=%u ff=%u text_vocab=%u speech_vocab=%u\n",
                c->hp.arch.c_str(), c->hp.n_layers, c->hp.hidden_size, c->hp.n_heads, c->hp.head_dim,
                c->hp.intermediate_size, c->hp.text_vocab_size, c->hp.speech_vocab_size);
        if (is_gpt2) {
            fprintf(stderr, "chatterbox: GPT-2 wpe_max=%u  tokenizer=%zu tokens\n", c->hp.wpe_max_positions,
                    c->tokenizer.id_to_token.size());
        } else {
            fprintf(stderr, "chatterbox: rope_theta=%.0f  tokenizer=%zu tokens  conds_emotion=%.2f\n",
                    (double)c->hp.rope_theta, c->tokenizer.id_to_token.size(), c->conds.emotion_adv);
        }
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "chatterbox: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    c->backend = c->backend_cpu;

    // Pass 2: weights
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path_model, c->backend, "chatterbox", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    // Bind tensors
    if (is_gpt2) {
        if (!bind_t3_gpt2(c)) {
            fprintf(stderr, "kartoffelbox: failed to bind GPT-2 T3 tensors\n");
            delete c;
            return nullptr;
        }
    } else {
        if (!bind_t3(c)) {
            fprintf(stderr, "chatterbox: failed to bind T3 tensors\n");
            delete c;
            return nullptr;
        }
    }
    bind_ve(c); // optional

    if (params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: precomputed conds %s\n",
                c->conds.loaded ? "loaded" : "NOT loaded (voice cloning required)");
    }

    // Compute scheduler
    {
        ggml_backend_t backends[] = {c->backend};
        c->sched = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    return c;
}

extern "C" int chatterbox_set_s3gen_path(struct chatterbox_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->s3gen_path = path;

    // Free existing
    if (ctx->s3gen_ctx) {
        chatterbox_s3gen_free(ctx->s3gen_ctx);
        ctx->s3gen_ctx = nullptr;
    }

    ctx->s3gen_ctx = chatterbox_s3gen_init_from_file(path, ctx->n_threads, ctx->params.verbosity);
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: failed to load S3Gen from %s\n", path);
        return -1;
    }
    return 0;
}

extern "C" int32_t* chatterbox_synthesize_tokens(struct chatterbox_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    const bool is_gpt2 = (ctx->hp.arch == "chatterbox_turbo" || ctx->hp.arch == "kartoffelbox");

    if (!is_gpt2 && !ctx->conds.loaded) {
        fprintf(stderr, "chatterbox: no conditioning loaded. Call chatterbox_set_voice_from_wav first.\n");
        return nullptr;
    }

    // 1. Normalize and tokenize text
    std::string norm_text = punc_norm(text);
    std::vector<int32_t> text_tokens;
    if (is_gpt2 && ctx->tokenizer.has_bpe) {
        text_tokens = tokenize_text_bpe(ctx->tokenizer, norm_text);
    } else {
        text_tokens = tokenize_text(ctx->tokenizer, norm_text);
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: text \"%s\" → %zu %s tokens\n", norm_text.c_str(), text_tokens.size(),
                (is_gpt2 && ctx->tokenizer.has_bpe) ? "BPE" : "char");
    }

    // 2. Add start/stop text tokens
    text_tokens.insert(text_tokens.begin(), (int32_t)ctx->hp.start_text_token);
    text_tokens.push_back((int32_t)ctx->hp.stop_text_token);

    // 3. Allocate KV cache
    const int max_speech = ctx->params.max_speech_tokens;
    int max_ctx = (int)text_tokens.size() + max_speech + 64; // generous padding
    if (!kv_alloc(ctx, max_ctx)) {
        fprintf(stderr, "chatterbox: failed to allocate KV cache\n");
        return nullptr;
    }

    // 4. Build prefill embeddings on CPU
    std::vector<float> prefill_embeds;
    if (is_gpt2) {
        prefill_embeds = build_prefill_embeds_gpt2(ctx, text_tokens);
    } else {
        prefill_embeds = build_prefill_embeds(ctx, text_tokens);
    }
    int prefill_len = (int)(prefill_embeds.size() / ctx->hp.hidden_size);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: prefill %d tokens (max_speech=%d)\n", prefill_len, max_speech);
    }

    // 5. CFG setup: build unconditioned prefill if cfg_weight > 0
    const float cfg_w = ctx->params.cfg_weight;
    const bool use_cfg = (!is_gpt2 && cfg_w > 0.0f && ctx->kv_k_cfg);
    std::vector<float> uncond_embeds;
    if (use_cfg) {
        // Unconditioned: zero out text embeddings, keep cond + speech_start
        uncond_embeds = prefill_embeds; // copy
        const int D = ctx->hp.hidden_size;
        int text_start = prefill_len - (int)text_tokens.size() - 1; // cond_len
        int text_end = prefill_len - 1;                             // before speech_start
        // Zero out text region
        for (int i = text_start; i < text_end; i++) {
            std::memset(&uncond_embeds[i * D], 0, D * sizeof(float));
        }
    }

    // 6. Prefill: run the full prefix through the transformer
    int n_past = 0;
    float* logits = nullptr;
    if (is_gpt2) {
        logits = run_t3_gpt2_kv(ctx, prefill_embeds.data(), prefill_len, n_past);
    } else {
        logits = run_t3_kv(ctx, prefill_embeds.data(), prefill_len, n_past);
    }
    if (!logits) {
        fprintf(stderr, "chatterbox: prefill failed\n");
        return nullptr;
    }
    // Also prefill the unconditioned path (Llama CFG only)
    float* logits_uncond = nullptr;
    int n_past_cfg = 0;
    if (use_cfg) {
        logits_uncond = run_t3_kv(ctx, uncond_embeds.data(), prefill_len, n_past_cfg, ctx->kv_k_cfg, ctx->kv_v_cfg);
        n_past_cfg += prefill_len;
    }
    n_past += prefill_len;

    // 7. AR decode loop with CFG
    std::vector<int32_t> speech_tokens;
    speech_tokens.reserve(max_speech);
    int speech_pos = 1;

    for (int step = 0; step < max_speech; step++) {
        // Blend logits with CFG: logits = cond + cfg * (cond - uncond)
        const int V = (int)ctx->hp.speech_vocab_size;
        std::vector<float> blended(V);
        if (use_cfg && logits_uncond) {
            for (int i = 0; i < V; i++) {
                blended[i] = logits[i] + cfg_w * (logits[i] - logits_uncond[i]);
            }
        } else {
            std::memcpy(blended.data(), logits, V * sizeof(float));
        }

        // Sample next token
        int32_t tok = sample_token(blended.data(), V, ctx->params.temperature, ctx->params.min_p, ctx->params.top_p,
                                   ctx->params.repetition_penalty, speech_tokens, ctx->rng_state);
        free(logits);
        logits = nullptr;
        if (logits_uncond) {
            free(logits_uncond);
            logits_uncond = nullptr;
        }

        if (ctx->params.verbosity >= 2 && step < 16) {
            fprintf(stderr, "chatterbox[ar]: step=%d tok=%d\n", step, tok);
        }

        // Check for EOS
        if (tok == (int32_t)ctx->hp.stop_speech_token) {
            if (ctx->params.verbosity >= 1) {
                fprintf(stderr, "chatterbox: EOS at step %d\n", step);
            }
            break;
        }

        speech_tokens.push_back(tok);

        // Build embedding for this token
        std::vector<float> tok_embed;
        if (is_gpt2) {
            // For GPT-2: absolute position = prefill_len + speech_pos - 1
            tok_embed = build_speech_token_embed_gpt2(ctx, tok, prefill_len + speech_pos - 1);
        } else {
            tok_embed = build_speech_token_embed(ctx, tok, speech_pos);
        }
        speech_pos++;

        // Conditioned forward step
        if (is_gpt2) {
            logits = run_t3_gpt2_kv(ctx, tok_embed.data(), 1, n_past);
        } else {
            logits = run_t3_kv(ctx, tok_embed.data(), 1, n_past);
        }
        if (!logits) {
            fprintf(stderr, "chatterbox: decode step %d failed\n", step);
            break;
        }
        n_past++;

        // Unconditioned forward step for CFG (Llama only)
        if (use_cfg) {
            logits_uncond = run_t3_kv(ctx, tok_embed.data(), 1, n_past_cfg, ctx->kv_k_cfg, ctx->kv_v_cfg);
            n_past_cfg++;
        }
    }
    if (logits)
        free(logits);
    if (logits_uncond)
        free(logits_uncond);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: AR emitted %zu speech tokens\n", speech_tokens.size());
    }

    // Filter to valid range
    const int max_valid_tok = is_gpt2 ? (int)ctx->hp.speech_vocab_size - 2 : 6561;
    std::vector<int32_t> valid;
    valid.reserve(speech_tokens.size());
    for (int32_t t : speech_tokens) {
        if (t >= 0 && t < max_valid_tok)
            valid.push_back(t);
    }

    if (valid.empty()) {
        return nullptr;
    }

    int32_t* out = (int32_t*)malloc(valid.size() * sizeof(int32_t));
    std::memcpy(out, valid.data(), valid.size() * sizeof(int32_t));
    *out_n = (int)valid.size();
    return out;
}

// Internal: run T3 + S3Gen to get mel, return channel-first (80, T_mel)
static std::vector<float> synthesize_mel_internal(chatterbox_context* ctx, const char* text, int* out_T_mel) {
    *out_T_mel = 0;
    if (!ctx->s3gen_ctx)
        return {};

    int n_tokens = 0;
    int32_t* speech_tokens = chatterbox_synthesize_tokens(ctx, text, &n_tokens);
    if (!speech_tokens || n_tokens == 0) {
        if (speech_tokens)
            chatterbox_tokens_free(speech_tokens);
        return {};
    }

    // Get precomputed conditioning
    std::vector<int32_t> pt_buf;
    std::vector<float> pf_buf, se_buf;
    const int32_t* prompt_tokens = nullptr;
    int n_prompt = 0;
    const float* prompt_feat = nullptr;
    int prompt_feat_len = 0;
    const float* spk_emb = nullptr;

    if (ctx->conds.gen_prompt_token) {
        n_prompt = (int)ctx->conds.gen_prompt_token->ne[0];
        pt_buf.resize(n_prompt);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_token, pt_buf.data(), 0, n_prompt * sizeof(int32_t));
        prompt_tokens = pt_buf.data();
    }
    if (ctx->conds.gen_prompt_feat) {
        prompt_feat_len = (int)ctx->conds.gen_prompt_feat->ne[1];
        pf_buf.resize(prompt_feat_len * 80);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_feat, pf_buf.data(), 0, pf_buf.size() * sizeof(float));
        prompt_feat = pf_buf.data();
    }
    if (ctx->conds.gen_embedding) {
        se_buf.resize(192);
        ggml_backend_tensor_get(ctx->conds.gen_embedding, se_buf.data(), 0, 192 * sizeof(float));
        spk_emb = se_buf.data();
    }

    // Run S3Gen to get mel (this calls the encoder + CFM denoiser)
    // We need to refactor s3gen to return mel instead of PCM, but for now
    // we'll use the existing synthesize and ignore the PCM, re-running encoder+CFM.
    // TODO: refactor to avoid double computation
    int n_samples = 0;
    float* pcm = chatterbox_s3gen_synthesize(ctx->s3gen_ctx, speech_tokens, n_tokens, prompt_tokens, n_prompt,
                                             prompt_feat, prompt_feat_len, spk_emb, ctx->params.cfm_steps, &n_samples);

    chatterbox_tokens_free(speech_tokens);
    if (pcm)
        chatterbox_s3gen_pcm_free(pcm);

    // For now, return empty — the proper implementation needs S3Gen
    // to expose the mel before vocoding.
    return {};
}

extern "C" float* chatterbox_synthesize_mel(struct chatterbox_context* ctx, const char* text, int* out_T_mel) {
    if (!ctx || !text || !out_T_mel)
        return nullptr;
    *out_T_mel = 0;
    // TODO: implement properly by having S3Gen return mel
    fprintf(stderr, "chatterbox: synthesize_mel not yet fully implemented\n");
    return nullptr;
}

extern "C" float* chatterbox_synthesize(struct chatterbox_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: S3Gen not loaded. Call chatterbox_set_s3gen_path first.\n");
        return nullptr;
    }

    // Step 1: T3 → speech tokens
    int n_tokens = 0;
    int32_t* speech_tokens = chatterbox_synthesize_tokens(ctx, text, &n_tokens);
    if (!speech_tokens || n_tokens == 0) {
        fprintf(stderr, "chatterbox: T3 produced no speech tokens\n");
        if (speech_tokens)
            chatterbox_tokens_free(speech_tokens);
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: T3 → %d speech tokens, running S3Gen...\n", n_tokens);
    }

    // Step 2+3: S3Gen → mel → waveform
    // Get precomputed conditioning tensors
    const int32_t* prompt_tokens = nullptr;
    int n_prompt = 0;
    const float* prompt_feat = nullptr;
    int prompt_feat_len = 0;
    const float* spk_emb = nullptr;

    std::vector<int32_t> pt_buf;
    std::vector<float> pf_buf;
    std::vector<float> se_buf;

    if (ctx->conds.gen_prompt_token) {
        n_prompt = (int)ctx->conds.gen_prompt_token->ne[0];
        pt_buf.resize(n_prompt);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_token, pt_buf.data(), 0, n_prompt * sizeof(int32_t));
        prompt_tokens = pt_buf.data();
    }
    if (ctx->conds.gen_prompt_feat) {
        prompt_feat_len = (int)ctx->conds.gen_prompt_feat->ne[1]; // (1, T, 80)
        pf_buf.resize(prompt_feat_len * 80);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_feat, pf_buf.data(), 0, pf_buf.size() * sizeof(float));
        prompt_feat = pf_buf.data();
    }
    if (ctx->conds.gen_embedding) {
        se_buf.resize(192);
        ggml_backend_tensor_get(ctx->conds.gen_embedding, se_buf.data(), 0, 192 * sizeof(float));
        spk_emb = se_buf.data();
    }

    float* pcm =
        chatterbox_s3gen_synthesize(ctx->s3gen_ctx, speech_tokens, n_tokens, prompt_tokens, n_prompt, prompt_feat,
                                    prompt_feat_len, spk_emb, ctx->params.cfm_steps, out_n_samples);

    chatterbox_tokens_free(speech_tokens);
    return pcm;
}

extern "C" float* chatterbox_synthesize_from_tokens(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                                    int n_speech_tokens, int* out_n_samples) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: S3Gen not loaded.\n");
        return nullptr;
    }
    // Extract conds (same code as chatterbox_synthesize)
    const int32_t* prompt_tokens = nullptr;
    int n_prompt = 0;
    const float* prompt_feat = nullptr;
    int prompt_feat_len = 0;
    const float* spk_emb = nullptr;
    std::vector<int32_t> pt_buf;
    std::vector<float> pf_buf;
    std::vector<float> se_buf;
    if (ctx->conds.gen_prompt_token) {
        n_prompt = (int)ctx->conds.gen_prompt_token->ne[0];
        pt_buf.resize(n_prompt);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_token, pt_buf.data(), 0, n_prompt * sizeof(int32_t));
        prompt_tokens = pt_buf.data();
    }
    if (ctx->conds.gen_prompt_feat) {
        prompt_feat_len = (int)ctx->conds.gen_prompt_feat->ne[1];
        pf_buf.resize(prompt_feat_len * 80);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_feat, pf_buf.data(), 0, pf_buf.size() * sizeof(float));
        prompt_feat = pf_buf.data();
    }
    if (ctx->conds.gen_embedding) {
        se_buf.resize(192);
        ggml_backend_tensor_get(ctx->conds.gen_embedding, se_buf.data(), 0, 192 * sizeof(float));
        spk_emb = se_buf.data();
    }
    return chatterbox_s3gen_synthesize(ctx->s3gen_ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt,
                                       prompt_feat, prompt_feat_len, spk_emb, ctx->params.cfm_steps, out_n_samples);
}

extern "C" int chatterbox_set_voice_from_wav(struct chatterbox_context* ctx, const char* wav_path) {
    (void)ctx;
    (void)wav_path;
    fprintf(stderr, "chatterbox: voice cloning from WAV not yet implemented\n");
    return -1;
}

extern "C" void chatterbox_set_exaggeration(struct chatterbox_context* ctx, float exaggeration) {
    if (ctx)
        ctx->conds.emotion_adv = exaggeration;
}

extern "C" void chatterbox_set_cfg_weight(struct chatterbox_context* ctx, float cfg_weight) {
    if (ctx)
        ctx->params.cfg_weight = cfg_weight;
}

extern "C" void chatterbox_set_cfm_steps(struct chatterbox_context* ctx, int steps) {
    if (ctx)
        ctx->params.cfm_steps = (steps > 0 && steps <= 100) ? steps : 10;
}

extern "C" void chatterbox_tokens_free(int32_t* tokens) {
    free(tokens);
}

extern "C" void chatterbox_pcm_free(float* pcm) {
    free(pcm);
}

extern "C" void chatterbox_free(struct chatterbox_context* ctx) {
    delete ctx;
}

extern "C" void chatterbox_set_n_threads(struct chatterbox_context* ctx, int n_threads) {
    if (ctx)
        ctx->n_threads = n_threads > 0 ? n_threads : 4;
}
