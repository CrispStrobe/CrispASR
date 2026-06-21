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

#define _USE_MATH_DEFINES
#include "chatterbox.h"
#include "chatterbox_s3gen.h"
#include "chatterbox_ve.h"
#include "chatterbox_text_prep.h"
#include "core/attention.h"
#include "core/audio_resample.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ===========================================================================
// Bench instrumentation — `CHATTERBOX_BENCH=1` for per-stage timings.
// ===========================================================================

static bool chatterbox_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CHATTERBOX_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct chatterbox_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit chatterbox_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~chatterbox_bench_stage() {
        if (!chatterbox_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  chatterbox_bench: %-22s %.2f ms\n", name, ms);
    }
};

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

// `cb_ve_model` lives in chatterbox_ve.h so chatterbox_ve.cpp can take it
// by const-ref without exposing internal helpers across the public ABI.

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
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
    bool has_bpe = false;
    bool bpe_byte_level = false;
    bool bpe_space_token = false;
};

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

    // GPT-2 pre-tokenizer: split into "(optional leading space) + (non-space
    // run)" chunks matching the ` ?\p{L}+` arm of the GPT-2 regex. byte_encoder
    // maps byte 0x20 to U+0120 ("Ġ"), so a chunk like " world" encodes as
    // "Ġworld" — the trained vocab entry. Issue #94 follow-up: the previous
    // implementation appended the trailing space to the *previous* word
    // (yielding "helloĠ" + bare "world") which produced unseen BPE pieces
    // and audibly broken synthesis for less-common words ("hello chatterbox
    // turbo" came out as "henay…"). See tools/tok_test.py for the diff.
    auto encode_chunk = [&](const char* begin, const char* end) {
        if (begin == end)
            return;
        std::string encoded;
        for (const char* p = begin; p < end; ++p) {
            const int cp = be[(uint8_t)*p];
            if (cp < 0x80) {
                encoded += (char)cp;
            } else if (cp < 0x800) {
                encoded += (char)(0xC0 | (cp >> 6));
                encoded += (char)(0x80 | (cp & 0x3F));
            } else {
                encoded += (char)(0xE0 | (cp >> 12));
                encoded += (char)(0x80 | ((cp >> 6) & 0x3F));
                encoded += (char)(0x80 | (cp & 0x3F));
            }
        }
        core_bpe::bpe_one(tok.token_to_id, tok.merge_rank, encoded, result);
    };

    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        if (text[i] == ' ') {
            // Consume one leading space if it's followed by a non-space char
            // (the " ?\p{L}+" arm). Multiple consecutive spaces are emitted as
            // one whitespace chunk (the "\s+" arm), so the BPE merger can
            // recover any multi-Ġ vocab entry the trained tokenizer used.
            if (i + 1 < text.size() && text[i + 1] != ' ') {
                ++i;
            } else {
                while (i < text.size() && text[i] == ' ')
                    ++i;
                encode_chunk(text.data() + start, text.data() + i);
                continue;
            }
        }
        while (i < text.size() && text[i] != ' ')
            ++i;
        encode_chunk(text.data() + start, text.data() + i);
    }
    return result;
}

// Plain HF BPE tokenization used by base Chatterbox.
//
// The python `EnTokenizer.encode` (resemble-ai/chatterbox) does
// `txt.replace(' ', '[SPACE]')` BEFORE calling the underlying HF
// tokenizer. So spaces DO produce `[SPACE]` tokens in the output, and
// the BPE merges per non-space chunk. We replicate that: replace
// ' ' → "[SPACE]" eagerly, then split into pieces at every
// "[SPACE]" marker (each marker emits its own token directly via
// vocab lookup) plus at punctuation/alnum boundaries (so "there,"
// becomes "there" + "," — python's HF tokenizer also splits this way
// via the merges table since 'there,' is not in vocab).
static std::vector<int32_t> tokenize_text_hf_bpe(const cb_tokenizer& tok, const std::string& text) {
    if (!tok.has_bpe) {
        return tokenize_text(tok, text);
    }

    std::vector<int32_t> result;
    std::string encoded = text;
    if (tok.bpe_space_token) {
        std::string replaced;
        replaced.reserve(encoded.size() * 2);
        for (char ch : encoded) {
            if (ch == ' ') {
                replaced += "[SPACE]";
            } else {
                replaced.push_back(ch);
            }
        }
        encoded.swap(replaced);
    }

    auto flush_pretok = [&](const std::string& piece) {
        if (piece.empty()) {
            return;
        }
        auto it = tok.token_to_id.find(piece);
        if (it != tok.token_to_id.end()) {
            result.push_back(it->second);
            return;
        }
        core_bpe::bpe_one(tok.token_to_id, tok.merge_rank, piece, result);
    };

    std::string cur;
    for (size_t i = 0; i < encoded.size();) {
        if (tok.bpe_space_token && encoded.compare(i, 7, "[SPACE]") == 0) {
            flush_pretok(cur);
            cur.clear();
            flush_pretok("[SPACE]");
            i += 7;
            continue;
        }

        const unsigned char ch = (unsigned char)encoded[i];
        const bool is_alnum =
            (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
        const bool cur_alnum =
            !cur.empty() && (((unsigned char)cur.back() >= '0' && (unsigned char)cur.back() <= '9') ||
                             ((unsigned char)cur.back() >= 'A' && (unsigned char)cur.back() <= 'Z') ||
                             ((unsigned char)cur.back() >= 'a' && (unsigned char)cur.back() <= 'z') ||
                             (unsigned char)cur.back() == '_');
        if (!cur.empty() && is_alnum != cur_alnum) {
            flush_pretok(cur);
            cur.clear();
        }
        cur.push_back((char)ch);
        ++i;
    }
    flush_pretok(cur);
    return result;
}

// ── Sampler ──────────────────────────────────────────────────────

struct mt19937_state {
    uint32_t mt[624];
    int left = 1;
    int next = 0;
};

static void mt19937_seed(mt19937_state& s, uint32_t seed) {
    s.mt[0] = seed;
    for (int i = 1; i < 624; i++) {
        s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + (uint32_t)i;
    }
    s.left = 1;
    s.next = 0;
}

static inline uint32_t mt19937_mix_bits(uint32_t u, uint32_t v) {
    return (u & 0x80000000u) | (v & 0x7fffffffu);
}

static inline uint32_t mt19937_twist(uint32_t u, uint32_t v) {
    return (mt19937_mix_bits(u, v) >> 1) ^ ((v & 1u) ? 0x9908b0dfu : 0u);
}

static void mt19937_next_state(mt19937_state& s) {
    uint32_t* p = s.mt;
    s.left = 624;
    s.next = 0;
    for (int j = 624 - 397 + 1; --j; ++p) {
        *p = p[397] ^ mt19937_twist(p[0], p[1]);
    }
    for (int j = 397; --j; ++p) {
        *p = p[397 - 624] ^ mt19937_twist(p[0], p[1]);
    }
    *p = p[397 - 624] ^ mt19937_twist(p[0], s.mt[0]);
}

static uint32_t mt19937_next(mt19937_state& s) {
    if (--s.left == 0) {
        mt19937_next_state(s);
    }
    uint32_t y = s.mt[s.next++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= (y >> 18);
    return y;
}

static double rand_uniform_torch(mt19937_state& rng) {
    return (double)(mt19937_next(rng) & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

static double rand_uniform_torch_u32(mt19937_state& rng) {
    return (double)mt19937_next(rng) * (1.0 / 4294967296.0);
}

static int32_t sample_token(const float* logits, int vocab_size, float temperature, int top_k, float min_p, float top_p,
                            float rep_penalty, const std::vector<int32_t>& prev_tokens, mt19937_state& rng) {
    // Issue #94: two distinct LogitsProcessorList orderings exist in the
    // chatterbox python reference, and they're not interchangeable:
    //
    //   base inference (t3.py:309-378):  rep_penalty → temperature → min_p → top_p
    //   inference_turbo (t3.py:415-490): temperature → top_k → top_p → rep_penalty
    //
    // Selecting on `top_k > 0` keeps base bit-equivalent to the prior C++
    // (top_k defaults to 0) while giving turbo the right ordering. The
    // turbo path skips min_p entirely (turbo defaults set min_p=0); the
    // base path skips top_k (defaults to 0). Both paths share the final
    // softmax → torch.multinomial implementation.
    std::vector<float> probs(vocab_size);
    const float kNegInf = -std::numeric_limits<float>::infinity();
    const bool use_hf_turbo_order = (top_k > 0);

    if (!use_hf_turbo_order) {
        // --- Base order: rep_penalty → temperature → min_p → top_p ----
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = logits[i];
        }
        if (rep_penalty != 1.0f) {
            std::vector<uint8_t> seen((size_t)vocab_size, 0);
            for (int32_t tok : prev_tokens) {
                if (tok >= 0 && tok < vocab_size) {
                    if (seen[(size_t)tok]) {
                        continue;
                    }
                    seen[(size_t)tok] = 1;
                    if (probs[tok] > 0)
                        probs[tok] /= rep_penalty;
                    else
                        probs[tok] *= rep_penalty;
                }
            }
        }

        if (temperature <= 0.0f) {
            return (int32_t)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        }
        if (temperature != 1.0f) {
            for (int i = 0; i < vocab_size; i++) {
                probs[i] /= temperature;
            }
        }

        if (min_p > 0.0f) {
            float max_val = *std::max_element(probs.begin(), probs.end());
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                probs[i] = std::exp(probs[i] - max_val);
                sum += probs[i];
            }
            for (int i = 0; i < vocab_size; i++) {
                probs[i] /= sum;
            }

            float max_prob = *std::max_element(probs.begin(), probs.end());
            float threshold = max_prob * min_p;
            std::vector<uint8_t> to_remove((size_t)vocab_size, 0);
            for (int i = 0; i < vocab_size; i++) {
                if (probs[i] < threshold) {
                    to_remove[(size_t)i] = 1;
                }
            }
            auto best_it = std::max_element(probs.begin(), probs.end());
            if (best_it != probs.end()) {
                to_remove[(size_t)(best_it - probs.begin())] = 0;
            }
            for (int i = 0; i < vocab_size; i++) {
                if (to_remove[(size_t)i]) {
                    probs[i] = 0.0f;
                }
            }
        }

        if (top_p < 1.0f) {
            std::vector<int> indices(vocab_size);
            for (int i = 0; i < vocab_size; i++)
                indices[i] = i;
            std::sort(indices.begin(), indices.end(), [&](int a, int b) { return probs[a] < probs[b]; });
            std::vector<uint8_t> to_remove((size_t)vocab_size, 0);
            float cumsum = 0.0f;
            for (int idx : indices) {
                cumsum += probs[idx];
                if (cumsum <= (1.0f - top_p)) {
                    to_remove[(size_t)idx] = 1;
                }
            }
            if (!indices.empty()) {
                to_remove[(size_t)indices.back()] = 0;
            }
            for (int i = 0; i < vocab_size; i++) {
                if (to_remove[(size_t)i]) {
                    probs[i] = 0.0f;
                }
            }
        }

        float sum = 0.0f;
        bool already_probs = (min_p > 0.0f);
        if (!already_probs) {
            float max_val = *std::max_element(probs.begin(), probs.end());
            for (int i = 0; i < vocab_size; i++) {
                probs[i] = std::exp(probs[i] - max_val);
            }
        }
        for (int i = 0; i < vocab_size; i++) {
            sum += probs[i];
        }
        if (sum <= 0.0f) {
            return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
        }
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }
    } else {
        // --- HF turbo order: temperature → top_k → top_p → rep_penalty ---
        // Operates on logits (not softmaxed probs) until the final softmax.
        // Matches HF LogitsProcessorList composition in T3.inference_turbo
        // (chatterbox/models/t3/t3.py:415-490): rep_penalty is applied
        // AFTER top_k/top_p, so masked tokens (-inf) stay masked even if
        // they were previously seen.
        std::vector<float> work(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            work[i] = logits[i];
        }

        if (temperature <= 0.0f) {
            return (int32_t)(std::max_element(work.begin(), work.end()) - work.begin());
        }
        if (temperature != 1.0f) {
            for (int i = 0; i < vocab_size; i++) {
                work[i] /= temperature;
            }
        }

        if (top_k < vocab_size) {
            // HF TopKLogitsWarper: indices_to_remove = scores < topk(scores, k).values[-1]
            std::vector<int> idx(vocab_size);
            for (int i = 0; i < vocab_size; i++)
                idx[i] = i;
            std::nth_element(idx.begin(), idx.begin() + (top_k - 1), idx.end(),
                             [&](int a, int b) { return work[a] > work[b]; });
            const float kth_value = work[idx[top_k - 1]];
            for (int i = 0; i < vocab_size; i++) {
                if (work[i] < kth_value) {
                    work[i] = kNegInf;
                }
            }
        }

        if (top_p < 1.0f) {
            // HF TopPLogitsWarper sorts logits ascending, softmaxes the
            // sorted vector, and drops indices whose cumulative softmax
            // probability is <= (1 - top_p). min_tokens_to_keep=1 means
            // the top-1 token is always retained.
            float max_val = kNegInf;
            for (int i = 0; i < vocab_size; i++) {
                if (work[i] > max_val) {
                    max_val = work[i];
                }
            }
            std::vector<float> sm(vocab_size);
            float sum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sm[i] = std::exp(work[i] - max_val);
                sum += sm[i];
            }
            if (sum > 0.0f) {
                for (int i = 0; i < vocab_size; i++) {
                    sm[i] /= sum;
                }
                std::vector<int> indices(vocab_size);
                for (int i = 0; i < vocab_size; i++)
                    indices[i] = i;
                std::sort(indices.begin(), indices.end(), [&](int a, int b) { return sm[a] < sm[b]; });
                float cumsum = 0.0f;
                for (size_t k = 0; k + 1 < indices.size(); k++) {
                    cumsum += sm[indices[k]];
                    if (cumsum <= (1.0f - top_p)) {
                        work[indices[k]] = kNegInf;
                    }
                }
            }
        }

        if (rep_penalty != 1.0f) {
            std::vector<uint8_t> seen((size_t)vocab_size, 0);
            for (int32_t tok : prev_tokens) {
                if (tok >= 0 && tok < vocab_size) {
                    if (seen[(size_t)tok]) {
                        continue;
                    }
                    seen[(size_t)tok] = 1;
                    if (work[tok] == kNegInf) {
                        continue;
                    }
                    if (work[tok] > 0)
                        work[tok] /= rep_penalty;
                    else
                        work[tok] *= rep_penalty;
                }
            }
        }

        // Final softmax. Tokens masked to -inf become 0 cleanly via exp().
        float max_val = kNegInf;
        for (int i = 0; i < vocab_size; i++) {
            if (work[i] > max_val) {
                max_val = work[i];
            }
        }
        if (max_val == kNegInf) {
            return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
        }
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = std::exp(work[i] - max_val);
            sum += probs[i];
        }
        if (sum <= 0.0f) {
            return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
        }
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= sum;
        }

        // Suppress unused-warning for min_p on this branch (turbo defaults
        // set min_p=0, so this is intentional).
        (void)min_p;
    }

    // Faithful CPU torch.multinomial(probs, num_samples=1) port:
    //   1. cum_dist[j] = sum_{i<=j} probs[i] in float32 (matches scalar_t in
    //      aten/src/ATen/native/cpu/MultinomialKernel.cpp).
    //   2. cum_dist[j] /= cum_dist[V-1]; cum_dist[V-1] = 1.0f (the upstream
    //      kernel snaps the last bucket to exactly 1 to defend against
    //      float-precision shortfall on the right edge of the CDF).
    //   3. uniform_real_distribution<double>(0,1)(gen) is two MT19937 32-bit
    //      draws combined as random64 = (r1 << 32) | r2, masked to 53 bits,
    //      divided by 2^53 (aten/src/ATen/core/DistributionsHelper.h).
    //   4. lower_bound: find smallest j with cum_dist[j] >= u.
    //
    // Replaces an earlier Gumbel-max implementation. Gumbel-max samples the
    // same distribution but consumes V uniforms per token and accumulates a
    // tiny pointwise bias from float32 -log1p(-u) at the right tail; on long
    // chatterbox prompts the cumulative effect is mis-pronounced syllables
    // (issue #76: "titan at dawn" → "titanette dawn"). The lower_bound path
    // matches torch.multinomial bit-for-bit when the MT19937 state agrees,
    // and statistically when it doesn't.
    {
        std::vector<float> cum_dist((size_t)vocab_size);
        float run = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            run += probs[i];
            cum_dist[(size_t)i] = run;
        }
        if (run <= 0.0f) {
            // Pathological: all probs zero after filtering. Fall back to
            // argmax of the original logits.
            return (int32_t)(std::max_element(logits, logits + vocab_size) - logits);
        }
        for (int i = 0; i < vocab_size; i++) {
            cum_dist[(size_t)i] /= run;
        }
        cum_dist[(size_t)(vocab_size - 1)] = 1.0f;

        const uint32_t r1 = mt19937_next(rng);
        const uint32_t r2 = mt19937_next(rng);
        const uint64_t v = ((uint64_t)r1 << 32) | (uint64_t)r2;
        const uint64_t mask53 = ((uint64_t)1 << 53) - 1;
        const double divisor = 1.0 / (double)((uint64_t)1 << 53);
        const double u = (double)(v & mask53) * divisor;

        int left = 0;
        int right = vocab_size;
        while (right - left > 0) {
            const int mid = left + (right - left) / 2;
            if ((double)cum_dist[(size_t)mid] < u) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left >= vocab_size) {
            left = vocab_size - 1;
        }
        return (int32_t)left;
    }
}

// ── Bind T3 tensors ─────────────────────────────────────────────

static bool bind_t3(chatterbox_context* c);
static bool bind_t3_gpt2(chatterbox_context* c);
static bool bind_ve(chatterbox_context* c);
static void load_metadata(chatterbox_context* c, gguf_context* g);
static std::vector<float> build_llama3_rope_freq_factors(const cb_t3_hp& hp);

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
    std::string language; // multilingual: "[de]", "[fr]", etc. Empty = English

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Voice clone — separate WeightLoad for runtime-loaded conds bundle.
    // Replaces conds.* pointers in `conds` while leaving the original
    // baked-in default voice tensors in `ctx_w`/`buf_w` untouched (they
    // become unreferenced but free with the model context). Lifecycle:
    // freed in dtor. nullptr when no voice GGUF has been loaded.
    ggml_context* voice_ctx_w = nullptr;
    ggml_backend_buffer_t voice_buf_w = nullptr;
    std::map<std::string, ggml_tensor*> voice_tensors;

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

    // §186 Lk-bucketed T3 AR step graphs (PLAN §186). Pre-built graphs at
    // fixed Lk sizes eliminate per-step sched reset + graph rebuild + alloc
    // for T=1 cond-pass decode steps. Gated: only when no debug dump env vars
    // are active and use_kv_k == nullptr (cond pass). kv_max_ctx-bounded.
    struct T3Bucket {
        int lk = 0;
        ggml_context* ctx = nullptr;
        std::vector<uint8_t> meta;
        ggml_cgraph* gf = nullptr;
    };
    static constexpr int kT3NBuckets = 4;
    static constexpr int kT3BucketLks[kT3NBuckets] = {512, 1024, 2048, 4096};
    std::array<T3Bucket, kT3NBuckets> t3_buckets{};
    ggml_backend_sched_t t3_step_sched = nullptr;
    int t3_active_bucket = -1;
    // §212: optional parallel bucket cache for the CFG *unconditioned* pass,
    // bound to kv_k_cfg/kv_v_cfg with its own scheduler so cond + uncond can
    // both stay allocated and alternate without re-allocating every token. The
    // unconditioned pass otherwise skips the bucket fast path (use_kv_k set) and
    // rebuilds + re-allocates its graph each step. Same graph topology as the
    // cond bucket — only the KV-cache binding differs — so it is bit-identical
    // (token-parity verified). Opt IN with CRISPASR_CHATTERBOX_T3_CFG_BUCKET=1;
    // default OFF because graph-rebuild is negligible on this compute-bound
    // decode (cf. §208) and the dual-scheduler path showed no win on contended
    // M1. Kept gated for quiet-machine evaluation + regression bisection.
    std::array<T3Bucket, kT3NBuckets> t3_buckets_cfg{};
    ggml_backend_sched_t t3_step_sched_cfg = nullptr;
    int t3_active_bucket_cfg = -1;

    // §214: dedicated scheduler for the batched classifier-free-guidance (B=2)
    // T3 decode step (CRISPASR_CHATTERBOX_T3_CFG_B2). Kept separate from c->sched
    // and the bucket schedulers so the B=2 graph can rebuild + re-alloc per step
    // without disturbing the cond/uncond bucket allocations.
    ggml_backend_sched_t t3_sched_b2 = nullptr;

    // §188: CPU-side F32 copies of embedding tables.
    // Populated once at init; eliminates tensor_get_f32 on every decode step.
    std::vector<float> speech_emb_cache;     // (speech_vocab_size × hidden)
    std::vector<float> speech_pos_emb_cache; // (speech_pos_emb_size × hidden)
    std::vector<float> text_emb_cache;       // (text_vocab_size × hidden)
    std::vector<float> text_pos_emb_cache;   // (text_pos_emb_size × hidden)
    std::vector<float> wpe_cache;            // (wpe_max_positions × hidden) GPT-2 path

    // Per-call performance counters (populated when CHATTERBOX_BENCH=1)
    struct {
        int64_t t_tokenize_us = 0;
        int64_t t_prefill_build_us = 0;
        int64_t t_prefill_us = 0;
        int64_t t_decode_us = 0;
        int n_text_tokens = 0;
        int n_prefill_tokens = 0;
        int n_speech_tokens = 0;
    } last_perf;

    // S3Gen context (lazy-loaded from set_s3gen_path)
    std::string s3gen_path;
    chatterbox_s3gen_context* s3gen_ctx = nullptr;

    // RNG
    mt19937_state rng_state{};
    uint32_t rng_seed = 0;
    std::vector<float> rope_freq_factors;

    ~chatterbox_context() {
        if (s3gen_ctx)
            chatterbox_s3gen_free(s3gen_ctx);
        if (t3_step_sched)
            ggml_backend_sched_free(t3_step_sched);
        if (t3_step_sched_cfg)
            ggml_backend_sched_free(t3_step_sched_cfg);
        if (t3_sched_b2)
            ggml_backend_sched_free(t3_sched_b2);
        for (auto& bk : t3_buckets)
            if (bk.ctx)
                ggml_free(bk.ctx);
        for (auto& bk : t3_buckets_cfg)
            if (bk.ctx)
                ggml_free(bk.ctx);
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
        if (voice_ctx_w)
            ggml_free(voice_ctx_w);
        if (voice_buf_w)
            ggml_backend_buffer_free(voice_buf_w);
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

static void prepend_language_token(chatterbox_context* ctx, std::vector<int32_t>& text_tokens) {
    if (!ctx || ctx->language.empty() || text_tokens.empty()) {
        return;
    }
    const auto it = ctx->tokenizer.token_to_id.find(ctx->language);
    if (it == ctx->tokenizer.token_to_id.end()) {
        return;
    }
    text_tokens.insert(text_tokens.begin(), it->second);
}

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
    c->rope_freq_factors = build_llama3_rope_freq_factors(hp);

    // Precomputed conds
    c->conds.emotion_adv = core_gguf::kv_f32(g, "chatterbox.conds.emotion_adv", c->conds.emotion_adv);
    c->conds.gen_prompt_token_len = core_gguf::kv_u32(g, "chatterbox.conds.gen_prompt_token_len", 0);

    // Text tokenizer vocab — try GPT-2 BPE first, then character-level
    auto bpe_tokens = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!bpe_tokens.empty()) {
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
        c->tokenizer.bpe_space_token = c->tokenizer.token_to_id.find("[SPACE]") != c->tokenizer.token_to_id.end();
        c->tokenizer.bpe_byte_level = !c->tokenizer.bpe_space_token;
        if (c->params.verbosity >= 1 && c->tokenizer.has_bpe) {
            fprintf(stderr, "chatterbox: %s tokenizer: %zu tokens, %zu merges\n",
                    c->tokenizer.bpe_byte_level ? "GPT-2 BPE" : "HF BPE", c->tokenizer.id_to_token.size(),
                    c->tokenizer.merge_rank.size());
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

static std::vector<float> build_llama3_rope_freq_factors(const cb_t3_hp& hp) {
    if (hp.rope_factor <= 1.0f || hp.rope_original_max_pos == 0 || hp.head_dim == 0 || (hp.head_dim % 2) != 0) {
        return {};
    }

    const int n_pairs = (int)hp.head_dim / 2;
    std::vector<float> factors((size_t)n_pairs, 1.0f);

    const float old_context_len = (float)hp.rope_original_max_pos;
    const float low_freq_wavelen = old_context_len / hp.rope_low_freq_factor;
    const float high_freq_wavelen = old_context_len / hp.rope_high_freq_factor;
    const float inv_factor = 1.0f / hp.rope_factor;

    for (int i = 0; i < n_pairs; ++i) {
        const float inv_freq = std::pow(hp.rope_theta, -(2.0f * i) / (float)hp.head_dim);
        const float wavelen = 2.0f * (float)M_PI / inv_freq;
        float new_inv_freq = inv_freq;
        if (wavelen > low_freq_wavelen) {
            new_inv_freq = inv_freq * inv_factor;
        } else if (wavelen >= high_freq_wavelen) {
            const float smooth = (old_context_len / wavelen - hp.rope_low_freq_factor) /
                                 (hp.rope_high_freq_factor - hp.rope_low_freq_factor);
            new_inv_freq = ((1.0f - smooth) * inv_freq * inv_factor) + (smooth * inv_freq);
        }
        factors[(size_t)i] = inv_freq / new_inv_freq;
    }

    return factors;
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

// Read a tensor's data as float32, dequantizing on the fly if needed.
// build_prefill_embeds reads weight tensors (spkr_proj, perceiver QKV, etc.)
// directly on CPU. Those tensors may be F16 or Q8_0 in a quantized GGUF —
// raw ggml_backend_tensor_get with sizeof(float) would then exceed ggml_nbytes
// and trip the OOB assertion in ggml-backend.cpp.
static void tensor_get_f32(ggml_tensor* t, float* out, size_t n) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out, (int64_t)n);
    } else {
        // Quantized type — read raw bytes then dequantize via ggml CPU traits.
        size_t raw = ggml_nbytes(t);
        std::vector<uint8_t> tmp(raw);
        ggml_backend_tensor_get(t, tmp.data(), 0, raw);
        const ggml_type_traits* tt = ggml_get_type_traits(t->type);
        GGML_ASSERT(tt && tt->to_float && "unsupported weight type in build_prefill_embeds");
        tt->to_float(tmp.data(), out, (int64_t)n);
    }
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
// fixed_kv_len > 0: pin Lk to a constant (bucket mode). Graph topology is
// then invariant across calls with different n_past; kv_indices=positions
// redirects K/V writes to the correct cache row at runtime. arena_ctx
// provides the ggml_context for the long-lived bucket graph (caller owns it
// and must NOT pass it to ggml_free; omit for per-call dynamic graphs).
static ggml_cgraph* build_graph_t3_kv(chatterbox_context* c, int n_past, int n_tokens, ggml_tensor* use_kv_k = nullptr,
                                      ggml_tensor* use_kv_v = nullptr, int fixed_kv_len = 0,
                                      ggml_context* arena_ctx = nullptr) {
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
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    GGML_ASSERT(c->kv_k && c->kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = arena_ctx ? arena_ctx : ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* rope_freq_factors = nullptr;
    if (!c->rope_freq_factors.empty()) {
        rope_freq_factors = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t)c->rope_freq_factors.size());
        ggml_set_name(rope_freq_factors, "rope_freq_factors");
        ggml_set_input(rope_freq_factors);
    }
    // Bucket mode always needs a mask: un-written tail slots must be -inf.
    ggml_tensor* causal_mask = nullptr;
    if (T > 1 || fixed_kv_len > 0) {
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
        /*rope_type*/ GGML_ROPE_TYPE_NEOX,
        /*n_rot*/ 0,
        /*v_rms_norm*/ false,
        /*rope_freq_factors*/ rope_freq_factors,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->t3.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // PLAN #83 diag: name layer-N intermediates so the per-op CPU/GPU bisect
        // dump in run_t3_kv can fetch and compare them. Layer index is
        // controllable via CRISPASR_CHATTERBOX_DUMP_LAYER (default 0).
        const char* dbg_layer_env = std::getenv("CRISPASR_CHATTERBOX_DUMP_LAYER");
        const int dbg_layer = dbg_layer_env ? (int)std::strtol(dbg_layer_env, nullptr, 10) : 0;
        if ((int)il == dbg_layer) {
            ggml_set_name(x, "L0_norm_out");
            ggml_set_output(x);
            ggml_build_forward_expand(gf, x);

            // Independent diagnostic K projection, rope, and pre-cpy view —
            // computed only when one of the dump knobs is set.  Avoids an
            // unconditional extra matmul in the production graph.
            if (std::getenv("CRISPASR_CHATTERBOX_DUMP_NORM_AT") || std::getenv("CRISPASR_CHATTERBOX_DUMP_KPROJ_AT") ||
                std::getenv("CRISPASR_CHATTERBOX_DUMP_KROPE_AT") || std::getenv("CRISPASR_CHATTERBOX_DUMP_WK")) {
                // Dequantize the layer-0 K weight to F32 so the dump can
                // compare CPU vs GPU dequant directly (without going through
                // a matmul). Helps separate dequant precision from matmul
                // precision.
                ggml_tensor* W_K_dbg = ggml_cast(ctx0, b.attn_k_w, GGML_TYPE_F32);
                ggml_set_name(W_K_dbg, "L0_W_K_f32");
                ggml_set_output(W_K_dbg);
                ggml_build_forward_expand(gf, W_K_dbg);

                ggml_tensor* K_dbg = ggml_mul_mat(ctx0, b.attn_k_w, x);
                ggml_mul_mat_set_prec(K_dbg, GGML_PREC_F32);
                ggml_set_name(K_dbg, "L0_K_proj");
                ggml_set_output(K_dbg);
                ggml_build_forward_expand(gf, K_dbg);

                ggml_tensor* K_dbg_3d = ggml_reshape_3d(ctx0, K_dbg, hd, n_kv, T);
                ggml_tensor* K_rope_dbg =
                    ggml_rope_ext(ctx0, K_dbg_3d, positions, kvp.rope_freq_factors, /*n_rot*/ hd, kvp.rope_type,
                                  kvp.n_ctx_orig, kvp.rope_theta, /*freq_scale*/ 1.0f,
                                  /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f, kvp.rope_beta_fast, kvp.rope_beta_slow);
                ggml_set_name(K_rope_dbg, "L0_K_rope");
                ggml_set_output(K_rope_dbg);
                ggml_build_forward_expand(gf, K_rope_dbg);

                // Q projection diag — same shape & path as K_dbg.
                ggml_tensor* Q_dbg = ggml_mul_mat(ctx0, b.attn_q_w, x);
                ggml_mul_mat_set_prec(Q_dbg, GGML_PREC_F32);
                ggml_set_name(Q_dbg, "L0_Q_proj");
                ggml_set_output(Q_dbg);
                ggml_build_forward_expand(gf, Q_dbg);

                // V projection diag.
                ggml_tensor* V_dbg = ggml_mul_mat(ctx0, b.attn_v_w, x);
                ggml_mul_mat_set_prec(V_dbg, GGML_PREC_F32);
                ggml_set_name(V_dbg, "L0_V_proj");
                ggml_set_output(V_dbg);
                ggml_build_forward_expand(gf, V_dbg);
            }
        }

        ggml_tensor* eff_kv_indices = fixed_kv_len > 0 ? positions : nullptr;
        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask,
                                                    use_kv_k, use_kv_v, (int)il, n_past, kvp,
                                                    /*qkv_w=*/nullptr, /*fixed_kv_len=*/fixed_kv_len,
                                                    /*kv_indices=*/eff_kv_indices);
        if (std::getenv("CRISPASR_CHATTERBOX_DUMP_ATTN_AT")) {
            const char* lyr_env = std::getenv("CRISPASR_CHATTERBOX_DUMP_LAYER");
            const int dbg_layer = lyr_env ? (int)std::strtol(lyr_env, nullptr, 10) : 0;
            if ((int)il == dbg_layer) {
                ggml_set_name(attn, "DBG_attn_out");
                ggml_set_output(attn);
                ggml_build_forward_expand(gf, attn);
            }
        }
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        if (std::getenv("CRISPASR_CHATTERBOX_DUMP_FFN_AT")) {
            const char* lyr_env = std::getenv("CRISPASR_CHATTERBOX_DUMP_LAYER");
            const int dbg_layer = lyr_env ? (int)std::strtol(lyr_env, nullptr, 10) : 0;
            if ((int)il == dbg_layer) {
                ggml_set_name(mlp, "DBG_ffn_out");
                ggml_set_output(mlp);
                ggml_build_forward_expand(gf, mlp);
            }
        }
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

    // PLAN #83: tag every mul_mat AND flash_attn_ext in the T3 graph with
    // GGML_PREC_F32. ggml-metal-ops.cpp's mul_mat dispatch picks the bespoke
    // kernel_mul_mv_q4_K_q8_K (mirrors CPU's Q8_K-input dot product
    // bit-identical) for Q4_K weights, and falls through to
    // kernel_mul_mv_ext (F32-precise dot) for other weight types. flash_attn
    // is routed to the CPU backend via the supports_op gate when PREC_F32 is
    // set on it (Metal's FA kernel uses simdgroup_half8x8 internally for
    // Q×K^T regardless of K type, leaking ~1e-4 drift even with F32 KV).
    // Other backends ignore PREC_F32 by design.
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(t, GGML_PREC_F32);
        } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32);
        }
    }

    if (!arena_ctx)
        ggml_free(ctx0);
    return gf;
}

// §214: batched classifier-free-guidance (B=2) T3 decode step.
//
// Runs the conditioned and unconditioned CFG passes as ONE batch-2 forward so
// each weight is read once for both passes — halving weight-bandwidth + dispatch
// on the single-token AR decode, which is the largest synthesis stage (the
// reference impl gianni-cor/chatterbox.cpp measured −42 % T3 from this change).
//
// Reuses the two existing KV caches (kv_k/kv_v cond, kv_k_cfg/kv_v_cfg uncond):
// the heavy Q/K/V/O/FFN/head GEMMs are batched over ne[2]=2 (ggml_mul_mat
// broadcasts the 2D weight over the batch dim); the cheap per-token attention
// (cache write/read + flash) is SPLIT per batch — the two passes use different
// caches and must not attend across each other. Both batches share the same
// token embedding and the same position (lockstep n_past == n_past_cfg in the
// decode loop), so the inputs are trivially replicated.
//
// Output `logits` is (speech_vocab, 1, 2): batch 0 = cond, batch 1 = uncond.
//
// This MUST stay token-parity-identical to the legacy sequential cond+uncond
// path. The graph faithfully replicates core_attn::kv_self_attn's chatterbox
// configuration (NEOX RoPE over the full head_dim, GQA_MANUAL_CONT expansion,
// F16-cache ggml_cpy write / quant-cache ggml_set_rows write, dequant-on-read)
// and tags every mul_mat + flash_attn_ext with GGML_PREC_F32 (load-bearing for
// the T3 multinomial sampler — see build_graph_t3_kv ~§83).
static ggml_cgraph* build_graph_t3_kv_b2(chatterbox_context* c, int n_past) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_size;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int Lk = n_past + 1;
    const int n_ctx_orig = (int)hp.rope_original_max_pos;
    const float rope_theta = hp.rope_theta;

    GGML_ASSERT(c->kv_k && c->kv_v && c->kv_k_cfg && c->kv_v_cfg && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Shared inputs: same embed for both passes (replicated along batch ne[2]),
    // same single position. Decode-only (T=1) so no causal mask is needed.
    ggml_tensor* embeds = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, D, 1, 2);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    ggml_tensor* rope_freq_factors = nullptr;
    if (!c->rope_freq_factors.empty()) {
        rope_freq_factors = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t)c->rope_freq_factors.size());
        ggml_set_name(rope_freq_factors, "rope_freq_factors");
        ggml_set_input(rope_freq_factors);
    }

    ggml_tensor* kv_ks[2] = {c->kv_k, c->kv_k_cfg};
    ggml_tensor* kv_vs[2] = {c->kv_v, c->kv_v_cfg};
    const bool quant_kv = ggml_is_quantized(c->kv_k->type);
    static const bool s_kv_read_f32 = []() {
        const char* s = std::getenv("CRISPASR_KV_READ_F32");
        return s && *s && std::strcmp(s, "0") != 0;
    }();
    const bool need_dequant = quant_kv || (s_kv_read_f32 && c->kv_k->type != GGML_TYPE_F32);

    ggml_tensor* cur = embeds; // (D, 1, 2)
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->t3.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w); // (D, 1, 2)

        // Batched Q/K/V projections — each weight read once for both passes.
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x); // (n_q*hd, 1, 2)
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, x); // (n_kv*hd, 1, 2)
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, x);
        Q = ggml_reshape_4d(ctx0, Q, hd, n_q, 1, 2);
        K = ggml_reshape_4d(ctx0, K, hd, n_kv, 1, 2);
        V = ggml_reshape_4d(ctx0, V, hd, n_kv, 1, 2);

        // RoPE (NEOX, full head_dim). positions (len 1 == ne[2]) is applied
        // independently to each batch at ne[3]. Matches kv_self_attn exactly:
        // freq_scale 1, ext_factor 0, attn_factor 1, beta_fast/slow 0.
        Q = ggml_rope_ext(ctx0, Q, positions, rope_freq_factors, hd, GGML_ROPE_TYPE_NEOX, n_ctx_orig, rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, positions, rope_freq_factors, hd, GGML_ROPE_TYPE_NEOX, n_ctx_orig, rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor* attn_b[2];
        for (int bi = 0; bi < 2; bi++) {
            ggml_tensor* kv_k = kv_ks[bi];
            ggml_tensor* kv_v = kv_vs[bi];
            // Per-batch slices (each batch is a contiguous block at ne[3]=bi).
            ggml_tensor* Qb = ggml_view_3d(ctx0, Q, hd, n_q, 1, Q->nb[1], Q->nb[2], (size_t)bi * Q->nb[3]);
            ggml_tensor* Kb = ggml_view_3d(ctx0, K, hd, n_kv, 1, K->nb[1], K->nb[2], (size_t)bi * K->nb[3]);
            ggml_tensor* Vb = ggml_view_3d(ctx0, V, hd, n_kv, 1, V->nb[1], V->nb[2], (size_t)bi * V->nb[3]);

            // Permute new K/V to (hd, 1, n_kv) for the cache write.
            ggml_tensor* K_new_perm = ggml_permute(ctx0, Kb, 0, 2, 1, 3);
            ggml_tensor* V_new_perm = ggml_permute(ctx0, Vb, 0, 2, 1, 3);

            // Write into the per-batch cache at [n_past, n_past+1). Quant caches
            // scatter via ggml_set_rows (positions == row ids); F16 caches use a
            // strided-view ggml_cpy. Mirrors core_attn::kv_self_attn.
            if (quant_kv) {
                ggml_tensor* k_layer =
                    ggml_view_3d(ctx0, kv_k, hd, kv_k->ne[1], n_kv, kv_k->nb[1], kv_k->nb[2], (size_t)il * kv_k->nb[3]);
                ggml_tensor* v_layer =
                    ggml_view_3d(ctx0, kv_v, hd, kv_v->ne[1], n_kv, kv_v->nb[1], kv_v->nb[2], (size_t)il * kv_v->nb[3]);
                ggml_build_forward_expand(gf, ggml_set_rows(ctx0, k_layer, K_new_perm, positions));
                ggml_build_forward_expand(gf, ggml_set_rows(ctx0, v_layer, V_new_perm, positions));
            } else {
                ggml_tensor* k_view = ggml_view_4d(ctx0, kv_k, hd, 1, n_kv, 1, kv_k->nb[1], kv_k->nb[2], kv_k->nb[3],
                                                   (size_t)il * kv_k->nb[3] + (size_t)n_past * kv_k->nb[1]);
                ggml_tensor* v_view = ggml_view_4d(ctx0, kv_v, hd, 1, n_kv, 1, kv_v->nb[1], kv_v->nb[2], kv_v->nb[3],
                                                   (size_t)il * kv_v->nb[3] + (size_t)n_past * kv_v->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));
            }

            // Read the full [0, Lk) history back from the cache.
            ggml_tensor* k_layer_view =
                ggml_view_3d(ctx0, kv_k, hd, Lk, n_kv, kv_k->nb[1], kv_k->nb[2], (size_t)il * kv_k->nb[3]);
            ggml_tensor* v_layer_view =
                ggml_view_3d(ctx0, kv_v, hd, Lk, n_kv, kv_v->nb[1], kv_v->nb[2], (size_t)il * kv_v->nb[3]);
            ggml_tensor* Kfull =
                need_dequant ? ggml_cast(ctx0, k_layer_view, GGML_TYPE_F32) : ggml_cont(ctx0, k_layer_view);
            ggml_tensor* Vfull =
                need_dequant ? ggml_cast(ctx0, v_layer_view, GGML_TYPE_F32) : ggml_cont(ctx0, v_layer_view);

            // GQA expansion (no-op for chatterbox default n_kv == n_q, grp == 1).
            if (grp > 1) {
                ggml_tensor* K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
                ggml_tensor* V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
                K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, grp, n_kv);
                V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, grp, n_kv);
                Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, Lk, n_q));
                Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, Lk, n_q));
            }

            ggml_tensor* Qp = ggml_cont(ctx0, ggml_permute(ctx0, Qb, 0, 2, 1, 3)); // (hd, 1, n_q)
            ggml_tensor* a = ggml_flash_attn_ext(ctx0, Qp, Kfull, Vfull, /*mask*/ nullptr, attn_scale,
                                                 /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
            attn_b[bi] = ggml_reshape_2d(ctx0, a, hd * n_q, 1);
        }

        // Re-batch the two attention outputs and project once.
        ggml_tensor* attn_cat = ggml_concat(ctx0, attn_b[0], attn_b[1], 2); // (hd*n_q, 1, 2)
        ggml_tensor* o = ggml_mul_mat(ctx0, b.attn_output_w, attn_cat);     // (D, 1, 2)
        cur = ggml_add(ctx0, residual, o);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w); // (D, 1, 2)
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->t3.output_norm_w);
    cur = ggml_mul_mat(ctx0, c->t3.speech_head_w, cur); // (speech_vocab, 1, 2)
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    // PLAN #83: GGML_PREC_F32 on every mul_mat + flash_attn_ext (sampler parity).
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(t, GGML_PREC_F32);
        } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32);
        }
    }

    ggml_free(ctx0); // metadata lives in c->compute_meta; gf stays valid (cf. build_graph_t3_kv)
    return gf;
}

// §214: run one batched-CFG B=2 decode step. Fills out_cond/out_uncond (each
// `speech_vocab` floats) with the cond/uncond logit rows. Returns false on
// failure. n_past must equal the (lockstep) cond/uncond cache length.
static bool run_t3_kv_b2(chatterbox_context* c, const float* embeds, int n_past, float* out_cond, float* out_uncond) {
    if (n_past + 1 > c->kv_max_ctx) {
        fprintf(stderr, "chatterbox: kv overflow b2 (%d+1 > %d)\n", n_past, c->kv_max_ctx);
        return false;
    }
    const int D = (int)c->hp.hidden_size;
    const int vocab = (int)c->hp.speech_vocab_size;

    if (!c->t3_sched_b2) {
        ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
        int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
        c->t3_sched_b2 = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        if (!c->t3_sched_b2)
            return false;
    }

    ggml_cgraph* gf = build_graph_t3_kv_b2(c, n_past);
    if (!gf)
        return false;

    ggml_backend_sched_reset(c->t3_sched_b2);
    if (!ggml_backend_sched_alloc_graph(c->t3_sched_b2, gf)) {
        fprintf(stderr, "chatterbox: failed to alloc T3 b2 graph\n");
        return false;
    }

    // Same embedding for both passes (batch 0 = cond, batch 1 = uncond).
    ggml_tensor* emb_t = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(emb_t, embeds, 0, (size_t)D * sizeof(float));
    ggml_backend_tensor_set(emb_t, embeds, (size_t)D * sizeof(float), (size_t)D * sizeof(float));
    int32_t pos = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), &pos, 0, sizeof(int32_t));
    if (!c->rope_freq_factors.empty())
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rope_freq_factors"), c->rope_freq_factors.data(), 0,
                                c->rope_freq_factors.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(c->t3_sched_b2, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "chatterbox: T3 b2 compute failed\n");
        return false;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits"); // (vocab, 1, 2)
    ggml_backend_tensor_get(out, out_cond, 0, (size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, out_uncond, (size_t)vocab * sizeof(float), (size_t)vocab * sizeof(float));
    return true;
}

// §186: returns true if any debug-dump env vars are set (those change graph topology).
static bool t3_debug_active() {
    return std::getenv("CRISPASR_CHATTERBOX_DUMP_LOGITS_AT") || std::getenv("CRISPASR_CHATTERBOX_DUMP_NORM_AT") ||
           std::getenv("CRISPASR_CHATTERBOX_DUMP_KPROJ_AT") || std::getenv("CRISPASR_CHATTERBOX_DUMP_KROPE_AT") ||
           std::getenv("CRISPASR_CHATTERBOX_DUMP_QPROJ_AT") || std::getenv("CRISPASR_CHATTERBOX_DUMP_VPROJ_AT") ||
           std::getenv("CRISPASR_CHATTERBOX_DUMP_ATTN_AT") || std::getenv("CRISPASR_CHATTERBOX_DUMP_FFN_AT") ||
           std::getenv("CRISPASR_CHATTERBOX_DUMP_WK") || std::getenv("CRISPASR_CHATTERBOX_DUMP_KV_AT") ||
           std::getenv("CRISPASR_CHATTERBOX_NAIVE_ATTN");
}

// §186: pick smallest bucket whose Lk >= needed_lk and fits in the KV cache.
static int t3_pick_bucket(chatterbox_context* c, int needed_lk) {
    for (int i = 0; i < chatterbox_context::kT3NBuckets; i++) {
        const int bk_lk = chatterbox_context::kT3BucketLks[i];
        if (bk_lk >= needed_lk && bk_lk <= c->kv_max_ctx)
            return i;
    }
    return -1;
}

// §186: lazily create the dedicated step scheduler (separate from c->sched so
// bucket allocations survive sched resets during synthesis).
static ggml_backend_sched_t t3_step_sched_lazy(chatterbox_context* c, bool cfg = false) {
    ggml_backend_sched_t& s = cfg ? c->t3_step_sched_cfg : c->t3_step_sched;
    if (s)
        return s;
    ggml_backend_t backends[2] = {c->backend, c->backend_cpu};
    int n_be = (c->backend && c->backend != c->backend_cpu) ? 2 : 1;
    s = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    return s;
}

// §186: get or build the pre-allocated graph for bucket[idx].
static ggml_cgraph* t3_get_or_build_bucket(chatterbox_context* c, int idx, bool cfg = false) {
    auto& bk = cfg ? c->t3_buckets_cfg[idx] : c->t3_buckets[idx];
    if (bk.gf)
        return bk.gf;
    bk.lk = chatterbox_context::kT3BucketLks[idx];
    bk.meta.assign(c->compute_meta.size(), 0);
    ggml_init_params ip = {bk.meta.size(), bk.meta.data(), true};
    bk.ctx = ggml_init(ip);
    if (!bk.ctx) {
        fprintf(stderr, "chatterbox: t3_bucket%s[%d] arena init failed\n", cfg ? "_cfg" : "", idx);
        return nullptr;
    }
    // §212: cfg bucket binds the unconditioned KV cache; otherwise default (cond).
    bk.gf = build_graph_t3_kv(c, /*n_past=*/0, /*n_tokens=*/1,
                              /*use_kv_k=*/cfg ? c->kv_k_cfg : nullptr,
                              /*use_kv_v=*/cfg ? c->kv_v_cfg : nullptr,
                              /*fixed_kv_len=*/bk.lk, /*arena_ctx=*/bk.ctx);
    if (!bk.gf) {
        ggml_free(bk.ctx);
        bk.ctx = nullptr;
        return nullptr;
    }
    return bk.gf;
}

// §186: fast Lk-bucketed single-step T3 decode (no debug dumps). §212: `cfg`
// selects the unconditioned-pass bucket cache (bound to kv_k_cfg/kv_v_cfg, own
// scheduler) so the CFG uncond pass is graph-cached just like the cond pass
// instead of rebuilding its graph every token.
static float* run_t3_kv_bucket(chatterbox_context* c, const float* embeds, int n_past, bool cfg = false) {
    const int idx = t3_pick_bucket(c, n_past + 1);
    if (idx < 0)
        return nullptr;

    ggml_cgraph* gf = t3_get_or_build_bucket(c, idx, cfg);
    if (!gf)
        return nullptr;

    const auto& bk = cfg ? c->t3_buckets_cfg[idx] : c->t3_buckets[idx];
    const int D = (int)c->hp.hidden_size;
    const int Lk = bk.lk;
    const int vocab = (int)c->hp.speech_vocab_size;

    ggml_backend_sched_t step_sched = t3_step_sched_lazy(c, cfg);
    if (!step_sched)
        return nullptr;

    int& active = cfg ? c->t3_active_bucket_cfg : c->t3_active_bucket;
    if (active != idx) {
        ggml_backend_sched_reset(step_sched);
        if (!ggml_backend_sched_alloc_graph(step_sched, gf)) {
            fprintf(stderr, "chatterbox: t3_bucket%s[%d] alloc failed\n", cfg ? "_cfg" : "", idx);
            return nullptr;
        }
        active = idx;
    }

    int32_t pos = n_past;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), &pos, 0, sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0, (size_t)D * sizeof(float));
    if (!c->rope_freq_factors.empty())
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rope_freq_factors"), c->rope_freq_factors.data(), 0,
                                c->rope_freq_factors.size() * sizeof(float));

    // Fill causal mask: positions [0..n_past] visible, [n_past+1..Lk-1] = -inf.
    {
        std::vector<ggml_fp16_t> mask(Lk, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int k = n_past + 1; k < Lk; k++)
            mask[k] = neg_inf;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                (size_t)Lk * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(step_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "chatterbox: t3_bucket compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// Run the T3 transformer on pre-built embeddings. Returns logits (speech_vocab,).
static float* run_t3_kv(chatterbox_context* c, const float* embeds, int n_tokens, int n_past,
                        ggml_tensor* use_kv_k = nullptr, ggml_tensor* use_kv_v = nullptr) {
    // §186: Lk-bucketed fast path for T=1 cond-pass decode (no debug dumps, no custom KV).
    if (n_tokens == 1 && !use_kv_k && !t3_debug_active()) {
        if (float* r = run_t3_kv_bucket(c, embeds, n_past))
            return r;
        // Fall through if no bucket fits (n_past+1 > max bucket or > kv_max_ctx).
    }
    // §212: optional Lk-bucketed fast path for the T=1 CFG *unconditioned* pass
    // (use_kv_k == kv_k_cfg), so it is graph-cached instead of rebuilt every
    // token. Bit-identical graph (token-parity verified), just bound to the cfg
    // KV cache. Opt IN with CRISPASR_CHATTERBOX_T3_CFG_BUCKET=1 (default OFF —
    // no proven win on contended M1; kept for quiet-machine evaluation).
    if (n_tokens == 1 && use_kv_k == c->kv_k_cfg && c->kv_k_cfg && !t3_debug_active()) {
        static const bool s_cfg_bucket = []() {
            const char* s = std::getenv("CRISPASR_CHATTERBOX_T3_CFG_BUCKET");
            return s && (s[0] == '1' || s[0] == 'y' || s[0] == 'Y');
        }();
        if (s_cfg_bucket) {
            if (float* r = run_t3_kv_bucket(c, embeds, n_past, /*cfg=*/true))
                return r;
        }
    }

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
    if (!c->rope_freq_factors.empty()) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rope_freq_factors"), c->rope_freq_factors.data(), 0,
                                c->rope_freq_factors.size() * sizeof(float));
    }
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

    // Optional logit dump: print first 8 logit values for the cond pass
    // at the requested n_past. Comparing CPU vs GPU under
    // CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=<n_past> localises drift in
    // the prefill (n_past after prefill) vs accumulating over decode.
    if (const char* e = std::getenv("CRISPASR_CHATTERBOX_DUMP_LOGITS_AT"); e && *e) {
        const int dump_n_past = (int)std::strtol(e, nullptr, 10);
        if (n_past + n_tokens > dump_n_past && n_past <= dump_n_past && (use_kv_k == nullptr || use_kv_k == c->kv_k)) {
            fprintf(stderr, "[LGT] n_past=%d:", dump_n_past);
            for (int i = 0; i < std::min(8, vocab); i++)
                fprintf(stderr, " %.4f", r[i]);
            fprintf(stderr, "\n");
        }
    }

    // PLAN #83 diag: dump layer-0 intermediate tensors named in build_graph_t3_kv
    // (L0_norm_out, L0_K_proj, L0_K_rope) for the requested token position.
    // Run on CPU and on GPU at the same n_past, diff per-element, find which op
    // contributes the dominant drift.
    auto dump_intermediate = [&](const char* env_name, const char* tensor_name, int row_dim_size) {
        const char* e = std::getenv(env_name);
        if (!e || !*e)
            return;
        const int row_id = (int)std::strtol(e, nullptr, 10);
        if (n_past + n_tokens <= row_id || n_past > row_id)
            return;
        if (use_kv_k != nullptr && use_kv_k != c->kv_k)
            return;
        ggml_tensor* t = ggml_graph_get_tensor(gf, tensor_name);
        if (!t)
            return;
        const int local_row = row_id - n_past;
        const size_t row_bytes = (size_t)row_dim_size * sizeof(float);
        std::vector<float> buf(row_dim_size);
        // L0_norm_out and L0_K_proj are 2D (D, T); row stride = D*sizeof(F32).
        // L0_K_rope is 3D (hd, n_kv, T) — use t->nb[2] for the time stride.
        const size_t off_bytes = (ggml_n_dims(t) == 3 ? (size_t)local_row * t->nb[2] : (size_t)local_row * t->nb[1]);
        if (off_bytes + row_bytes > ggml_nbytes(t))
            return;
        ggml_backend_tensor_get(t, buf.data(), off_bytes, row_bytes);
        fprintf(stderr, "[%s] t=%d:", tensor_name, row_id);
        for (int i = 0; i < std::min(8, row_dim_size); i++)
            fprintf(stderr, " %.4f", buf[i]);
        fprintf(stderr, "\n");
    };
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_NORM_AT", "L0_norm_out", (int)c->hp.hidden_size);
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_KPROJ_AT", "L0_K_proj", (int)c->hp.hidden_size);
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_QPROJ_AT", "L0_Q_proj", (int)c->hp.hidden_size);
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_VPROJ_AT", "L0_V_proj", (int)c->hp.hidden_size);
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_KROPE_AT", "L0_K_rope", (int)c->hp.head_dim);
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_ATTN_AT", "DBG_attn_out", (int)c->hp.hidden_size);
    dump_intermediate("CRISPASR_CHATTERBOX_DUMP_FFN_AT", "DBG_ffn_out", (int)c->hp.hidden_size);

    // Special: dump the dequantised K weight tensor (row 0).  Stride through
    // the row in chunks to expose any per-block drift.
    if (const char* e = std::getenv("CRISPASR_CHATTERBOX_DUMP_WK"); e && *e) {
        if (use_kv_k == nullptr || use_kv_k == c->kv_k) {
            ggml_tensor* w = ggml_graph_get_tensor(gf, "L0_W_K_f32");
            if (w) {
                const int n_dump = std::min(256, (int)w->ne[0]);
                std::vector<float> wbuf(n_dump);
                ggml_backend_tensor_get(w, wbuf.data(), 0, n_dump * sizeof(float));
                for (int chunk = 0; chunk < n_dump; chunk += 8) {
                    fprintf(stderr, "[L0_W_K_f32] row=0 cols=%d..%d:", chunk, chunk + 7);
                    for (int i = 0; i < 8 && chunk + i < n_dump; i++)
                        fprintf(stderr, " %.6f", wbuf[chunk + i]);
                    fprintf(stderr, "\n");
                }
            }
        }
    }

    // Optional KV cache snapshot for Metal-vs-CPU divergence debugging.
    // CRISPASR_CHATTERBOX_DUMP_KV_AT=<n_past> dumps the layer-0 K cache
    // contents for the row at the specified n_past offset to stderr,
    // ONLY for the cond pass (use_kv_k == c->kv_k or null). Allows
    // comparing on/off-GPU writes byte-by-byte at a single step.
    if (const char* e = std::getenv("CRISPASR_CHATTERBOX_DUMP_KV_AT"); e && *e) {
        const int dump_n_past = (int)std::strtol(e, nullptr, 10);
        const char* lyr_env = std::getenv("CRISPASR_CHATTERBOX_DUMP_KV_LAYER");
        const int dump_layer = lyr_env ? (int)std::strtol(lyr_env, nullptr, 10) : 0;
        if (n_past + n_tokens > dump_n_past && n_past <= dump_n_past && (use_kv_k == nullptr || use_kv_k == c->kv_k)) {
            const int hd_l = (int)c->hp.head_dim;
            ggml_tensor* kv_t = c->kv_k;
            const size_t row_bytes = (size_t)hd_l * ggml_type_size(kv_t->type);
            const size_t off_bytes = (size_t)dump_layer * kv_t->nb[3] + (size_t)dump_n_past * kv_t->nb[1];
            std::vector<uint8_t> raw(row_bytes);
            ggml_backend_tensor_get(kv_t, raw.data(), off_bytes, row_bytes);
            fprintf(stderr, "[KV] L=%d h=0 t=%d type=%s hd=%d:", dump_layer, dump_n_past, ggml_type_name(kv_t->type),
                    hd_l);
            if (kv_t->type == GGML_TYPE_F16) {
                for (int i = 0; i < std::min(8, hd_l); i++) {
                    fprintf(stderr, " %.4f", ggml_fp16_to_fp32(((ggml_fp16_t*)raw.data())[i]));
                }
            } else if (kv_t->type == GGML_TYPE_F32) {
                // memcpy avoids the unsigned-char* → float* type-pun cast
                // that trips cppcheck's invalidPointerCast portability
                // warning. The buffer really is F32 in this branch.
                for (int i = 0; i < std::min(8, hd_l); i++) {
                    float v;
                    std::memcpy(&v, raw.data() + i * sizeof(float), sizeof(float));
                    fprintf(stderr, " %.4f", v);
                }
            }
            fprintf(stderr, "\n");
        }
    }
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
        tensor_get_f32(m.cond_spkr_w, w.data(), w.size());
        for (int i = 0; i < D; i++) {
            float sum = 0.0f;
            for (int j = 0; j < (int)c->hp.speaker_embed_size; j++) {
                sum += w[i * c->hp.speaker_embed_size + j] * spkr_emb[j];
            }
            spkr_proj[i] = sum;
        }
        if (m.cond_spkr_b) {
            std::vector<float> bias(D);
            tensor_get_f32(m.cond_spkr_b, bias.data(), D);
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

        const auto& speech_emb_tab = c->speech_emb_cache;
        const auto& speech_pos_tab = c->speech_pos_emb_cache;

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
        tensor_get_f32(m.perceiver_query, query.data(), query.size());

        // Helper: read Linear weight (out_dim, in_dim) and optional bias (out_dim)
        auto read_linear = [&](ggml_tensor* w_t, ggml_tensor* b_t, int out_d, int in_d) {
            std::vector<float> w(out_d * in_d);
            std::vector<float> b(out_d, 0.0f);
            tensor_get_f32(w_t, w.data(), w.size());
            if (b_t)
                tensor_get_f32(b_t, b.data(), b.size());
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
            tensor_get_f32(m.perceiver_norm_w, norm_w.data(), D);
        if (m.perceiver_norm_b)
            tensor_get_f32(m.perceiver_norm_b, norm_b.data(), D);

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
        tensor_get_f32(m.cond_emotion_w, w.data(), D);
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
        const auto& text_emb_table = c->text_emb_cache;
        const auto& text_pos_table = c->text_pos_emb_cache;
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
        const auto& speech_emb_table = c->speech_emb_cache;
        const auto& speech_pos_table = c->speech_pos_emb_cache;

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
    const auto& emb_tab = c->speech_emb_cache;
    const auto& pos_tab = c->speech_pos_emb_cache;
    int tid = (token_id >= 0 && token_id < (int)c->hp.speech_vocab_size) ? token_id : 0;
    int pid = (speech_pos >= 0 && speech_pos < (int)c->hp.speech_pos_emb_size) ? speech_pos : 0;
    for (int j = 0; j < D; j++)
        embed[j] = emb_tab[tid * D + j] + pos_tab[pid * D + j];
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

    // Issue #94 follow-up: per-layer hidden-state dump for the GPT-2 graph.
    // Set CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS=1 to mark each
    // post-attn-residual and post-FFN-residual as graph outputs. The runner
    // side reads them by name and writes raw float32 to
    // /tmp/cb_gpt2_step_<n_past>_LNN_post_{attn,ffn}.bin. Lets us bisect
    // which transformer layer first diverges from the HuggingFace reference
    // at the T == 1 decoding step.
    //
    // Note: the input tensor (`inputs_embeds`) is *not* dumped — set_output
    // on a set_input tensor is ignored by the scheduler in practice and the
    // memory gets repurposed after layer 0 reads it. Verify the input via a
    // stderr print of `tok_embed.data()` first10 in the AR-loop call site if
    // you need to compare it against speech_emb(tok) + wpe(pos).
    const bool dump_layers = (std::getenv("CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS") != nullptr);

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

        // Attention. CRISPASR_CHATTERBOX_NAIVE_ATTN=1 swaps ggml_flash_attn_ext
        // for an explicit softmax(QK^T)V path. Useful for isolating flash_attn
        // accumulator-order differences from other bugs. Layout follows
        // src/qwen3_asr.cpp:895-924: scores = mul_mat(K, Q); soft_max_ext
        // (fused scale + mask + softmax); V is permuted (Lk, hd, n_h) for the
        // attn = mul_mat(V', scores) step that contracts over Lk. The result
        // is (hd, T, n_h) which must be permuted (0, 2, 1, 3) to (hd, n_h, T)
        // before reshape_2d(D, T) so the head dim packs correctly for the WO
        // projection — flash_attn_ext outputs (hd, n_h, T) natively per its
        // ggml.h docs, so skipping the permute here gave wrong outputs in an
        // earlier attempt.
        ggml_tensor* attn;
        if (std::getenv("CRISPASR_CHATTERBOX_NAIVE_ATTN")) {
            ggml_tensor* scores = ggml_mul_mat(ctx0, Kfull, Q);
            scores = ggml_soft_max_ext(ctx0, scores, (T > 1) ? causal_mask : nullptr, attn_scale, 0.0f);
            ggml_tensor* Vp = ggml_cont(ctx0, ggml_permute(ctx0, Vfull, 1, 0, 2, 3));
            attn = ggml_mul_mat(ctx0, Vp, scores);
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(ctx0, attn, D, T);
        } else {
            attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (T == 1) ? nullptr : causal_mask, attn_scale,
                                       /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, D, T);
        }

        // Output projection + residual
        attn = ggml_mul_mat(ctx0, b.attn_output_w, attn);
        attn = ggml_add(ctx0, attn, b.attn_output_b);
        cur = ggml_add(ctx0, residual, attn);
        if (dump_layers) {
            char nm[32];
            std::snprintf(nm, sizeof nm, "L%02u_post_attn", il);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }

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
        if (dump_layers) {
            char nm[32];
            std::snprintf(nm, sizeof nm, "L%02u_post_ffn", il);
            ggml_set_name(cur, nm);
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }
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

    // PLAN #83: tag every mul_mat AND flash_attn_ext with GGML_PREC_F32.
    // (Same as the Llama path above.)
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_MUL_MAT) {
            ggml_mul_mat_set_prec(t, GGML_PREC_F32);
        } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_flash_attn_ext_set_prec(t, GGML_PREC_F32);
        }
    }

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
    // Issue #94 follow-up: optional per-layer hidden-state dump for bisecting
    // T3 step >= 1 divergence vs the HF reference. The graph builder names
    // L_in_emb, LNN_post_attn, LNN_post_ffn. Files are written to
    // /tmp/cb_gpt2_step_<n_past>_<name>.bin as raw little-endian float32 of
    // size D*T (T is usually 1 on the AR-step call). The runner doesn't know
    // whether the graph builder added these names (env-gated), so we tolerate
    // misses.
    if (std::getenv("CRISPASR_CHATTERBOX_DUMP_GPT2_LAYERS")) {
        auto dump = [&](const char* nm) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
            if (!t) {
                return;
            }
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> buf(nbytes);
            ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
            char path[256];
            std::snprintf(path, sizeof path, "/tmp/cb_gpt2_step_%d_%s.bin", n_past, nm);
            FILE* fp = std::fopen(path, "wb");
            if (fp) {
                std::fwrite(buf.data(), 1, nbytes, fp);
                std::fclose(fp);
            }
        };
        for (uint32_t il = 0; il < c->hp.n_layers; il++) {
            char nm1[32], nm2[32];
            std::snprintf(nm1, sizeof nm1, "L%02u_post_attn", il);
            std::snprintf(nm2, sizeof nm2, "L%02u_post_ffn", il);
            dump(nm1);
            dump(nm2);
        }
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

    const auto& wpe_table = c->wpe_cache;

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
        tensor_get_f32(m.cond_spkr_w, w.data(), w.size());
        for (int i = 0; i < D; i++) {
            float sum = 0.0f;
            for (int j = 0; j < (int)c->hp.speaker_embed_size; j++) {
                sum += w[i * c->hp.speaker_embed_size + j] * spkr_emb[j];
            }
            spkr_proj[i] = sum;
        }
        if (m.cond_spkr_b) {
            std::vector<float> bias(D);
            tensor_get_f32(m.cond_spkr_b, bias.data(), D);
            for (int i = 0; i < D; i++)
                spkr_proj[i] += bias[i];
        }
        cond_len = 1;

        // Speech prompt conditioning tokens → N embeddings (no pos emb for GPT-2)
        if (c->conds.speech_prompt_tokens) {
            int n_prompt = (int)c->conds.speech_prompt_tokens->ne[0];
            std::vector<int32_t> prompt_toks(n_prompt);
            ggml_backend_tensor_get(c->conds.speech_prompt_tokens, prompt_toks.data(), 0, n_prompt * sizeof(int32_t));

            const auto& speech_emb_table = c->speech_emb_cache;
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

    const auto& text_emb_table = c->text_emb_cache;
    const auto& speech_emb_table = c->speech_emb_cache;

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
    const auto& emb_tab = c->speech_emb_cache;
    const auto& wpe_tab = c->wpe_cache;
    int tid = (token_id >= 0 && token_id < (int)c->hp.speech_vocab_size) ? token_id : 0;
    int pid = (abs_pos >= 0 && abs_pos < (int)c->hp.wpe_max_positions) ? abs_pos : 0;
    for (int j = 0; j < D; j++)
        embed[j] = emb_tab[tid * D + j] + wpe_tab[pid * D + j];
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
    p.top_k = 0;
    p.max_speech_tokens = 1000;
    // S3Gen flow-matching Euler steps. 0 = auto, resolved per-model in s3gen:
    // standard models → 6, meanflow/turbo distilled models → 2. (Standard:
    // upstream/reference uses 10; 6 is perceptually identical — log-mel corr
    // 0.981 vs 10, identical ASR roundtrip — at ~46% less CFM time. Meanflow:
    // the distilled 2-step schedule; §207's 10→6 standard change previously
    // broke the meanflow auto-downgrade, leaving turbo at 6 steps.) Override
    // with --tts-steps N; the crispasr-diff harness pins 10 to match the
    // reference dump, the "reference-exact" setting.
    p.cfm_steps = 0;
    // PLAN #89: flash_attn defaults to true (lost in commit ff5536ae;
    // restored 2026-05-11). The compute-graph wiring per backend
    // lands in PLAN #86; until then this is plumbing-only on
    // chatterbox.
    p.flash_attn = true;
    return p;
}

extern "C" struct chatterbox_context* chatterbox_init_from_file(const char* path_model,
                                                                struct chatterbox_context_params params) {
    auto* c = new chatterbox_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    {
        const char* env = std::getenv("CRISPASR_CHATTERBOX_T3_SEED");
        if (env && env[0]) {
            c->rng_seed = (uint32_t)strtoul(env, nullptr, 10);
        } else {
            c->rng_seed = 0;
        }
        mt19937_seed(c->rng_state, c->rng_seed);
    }

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

    if (!c->tokenizer.id_to_token.empty() && c->tokenizer.id_to_token.size() != c->hp.text_vocab_size) {
        fprintf(stderr,
                "chatterbox: tokenizer/model vocab mismatch: tokenizer has %zu tokens, "
                "T3 text_vocab_size=%u. Re-convert with the tokenizer paired to this T3 checkpoint.\n",
                c->tokenizer.id_to_token.size(), c->hp.text_vocab_size);
        delete c;
        return nullptr;
    }

    // Issue #94 follow-up: chatterbox-turbo's Python tts_turbo.generate()
    // explicitly disables CFG (cfg_weight=0.0 default, plus a log line
    // "CFG, min_p and exaggeration are not supported by Turbo version"
    // when set). It also runs a different sampler stack from base
    // chatterbox: HF inference_turbo wires `temperature → top_k=1000 →
    // top_p=0.95 → repetition_penalty` (tts_turbo.py:248-260,
    // chatterbox/models/t3/t3.py:415-490). top_k=1000 is what filters
    // the long-tail garbage that an isolated top_p=0.95 lets through —
    // without it, the multinomial pick at step 1 lands on S3GEN_SIL
    // (4299) on most prompts, producing the "IN-and-Hello…" /
    // "HI Low World Test" prefix artifact users were hearing.
    if (is_gpt2) {
        c->params.cfg_weight = 0.0f;
        c->params.min_p = 0.0f;
        c->params.top_p = 0.95f;
        c->params.top_k = 1000;
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: arch=%s T3 %uL d=%u h=%u hd=%u ff=%u text_vocab=%u speech_vocab=%u\n",
                c->hp.arch.c_str(), c->hp.n_layers, c->hp.hidden_size, c->hp.n_heads, c->hp.head_dim,
                c->hp.intermediate_size, c->hp.text_vocab_size, c->hp.speech_vocab_size);
        fprintf(stderr, "chatterbox: T3 sampler seed=%u\n", c->rng_seed);
        if (is_gpt2) {
            fprintf(stderr, "chatterbox: GPT-2 wpe_max=%u  tokenizer=%zu tokens\n", c->hp.wpe_max_positions,
                    c->tokenizer.id_to_token.size());
        } else {
            fprintf(stderr, "chatterbox: rope_theta=%.0f  tokenizer=%zu tokens  conds_emotion=%.2f\n",
                    (double)c->hp.rope_theta, c->tokenizer.id_to_token.size(), c->conds.emotion_adv);
        }
    }

    // Backend split. The chatterbox graph has two halves: the T3 AR
    // transformer (30-layer Llama, multinomial speech-token sampler) and
    // S3Gen (Conformer encoder + CFM denoiser + HiFT vocoder). Verified
    // 2026-05-18 (handover-prompts/chatterbox-gpu-bug-is-s3gen.md): with
    // F16 T3 weights, T3 on GPU produces an AR speech-token sequence
    // BIT-IDENTICAL to T3 on CPU, and the round-4 Q4_K×Q8_K kernel +
    // PREC_F32 tagging in T3 keep Q4_K T3 within tolerance too. The
    // user-audible "GPU produces garbled audio" regression is owned
    // entirely by the three S3Gen sub-graphs — likely some Conv1d /
    // norm precision plumbing that hasn't been audited yet. Default is
    // therefore T3 GPU + S3Gen CPU: correct output, ~most of the GPU
    // speedup (T3 AR loop is the slow stage).
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "chatterbox: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    // Wire the CPU backend thread count. Without this, ggml_backend_cpu_init
    // leaves it at GGML_DEFAULT_N_THREADS (4), so the compute-bound T3 AR
    // decode ran at 4 threads on every machine regardless of -t. Use up to 8
    // cores by default (output is bit-identical — ggml matmul splits output
    // rows per-thread, no cross-thread reduction, so the result is
    // thread-count-independent). Never drop below the caller's -t. Override
    // with CRISPASR_CHATTERBOX_THREADS=<n>.
    {
        int cb_threads;
        if (const char* e = std::getenv("CRISPASR_CHATTERBOX_THREADS")) {
            cb_threads = std::max(1, atoi(e));
        } else {
            const int hw = (int)std::thread::hardware_concurrency();
            cb_threads = std::max(c->n_threads, std::min(8, hw > 0 ? hw : 4));
        }
        c->n_threads = cb_threads;
        ggml_backend_cpu_set_n_threads(c->backend_cpu, cb_threads);
        if (params.verbosity >= 1)
            fprintf(stderr, "chatterbox: CPU backend threads=%d\n", cb_threads);
    }
    // Env knobs (all override the default):
    //   CRISPASR_CHATTERBOX_FORCE_GPU=1       — both T3 and S3Gen on GPU
    //                                           (legacy; S3Gen output is
    //                                           garbled, kept for diag).
    //   CRISPASR_CHATTERBOX_T3_CPU_S3GEN_GPU=1 — flip the split (S3Gen on
    //                                           GPU only, T3 on CPU).
    //   CRISPASR_CHATTERBOX_S3GEN_CPU=1       — force S3Gen to CPU even
    //                                           under FORCE_GPU=1.
    //   CRISPASR_CHATTERBOX_FULL_CPU=1        — old default; both halves
    //                                           on CPU.
    // cppcheck-suppress duplicateAssignExpression
    bool t3_use_gpu = params.use_gpu;
    bool s3gen_use_gpu = params.use_gpu; // NOLINT — intentionally same init, diverges below
    if (params.use_gpu) {
        auto env_set = [](const char* name) {
            const char* v = std::getenv(name);
            return v && *v && std::strcmp(v, "0") != 0;
        };
        const bool force_gpu = env_set("CRISPASR_CHATTERBOX_FORCE_GPU");
        const bool split_t3_cpu = env_set("CRISPASR_CHATTERBOX_T3_CPU_S3GEN_GPU");
        const bool s3gen_cpu_override = env_set("CRISPASR_CHATTERBOX_S3GEN_CPU");
        const bool full_cpu = env_set("CRISPASR_CHATTERBOX_FULL_CPU");
        const bool t3_gpu_override = env_set("CRISPASR_CHATTERBOX_T3_GPU");

        if (full_cpu) {
            fprintf(stderr, "chatterbox: full CPU (CRISPASR_CHATTERBOX_FULL_CPU=1).\n");
            t3_use_gpu = false;
            s3gen_use_gpu = false;
        } else if (force_gpu) {
            // PLAN #83 R9 #5 (2026-05-24): the "S3Gen GPU path is broken"
            // warning that used to live here is stale. Bug A (sched src
            // mutation) and Bug B (sched parallel-sync) are both fixed; GPU
            // residency now hits s3gen_mel cos_min=0.999976 on M1 Metal
            // (matches the prior workaround baseline). Production CPU
            // residency is the default; this branch is for users who
            // explicitly opt into GPU.
            fprintf(stderr, "chatterbox: T3+s3gen forced to GPU (CRISPASR_CHATTERBOX_FORCE_GPU=1).\n");
            // Note: T3 GPU on Metal is slower than CPU (kernel-launch overhead).
            if (s3gen_cpu_override) {
                fprintf(stderr,
                        "chatterbox: s3gen forced to CPU (CRISPASR_CHATTERBOX_S3GEN_CPU=1) — T3 stays on GPU.\n");
                s3gen_use_gpu = false;
            }
        } else if (split_t3_cpu) {
            fprintf(stderr, "chatterbox: T3 → CPU, s3gen → GPU (CRISPASR_CHATTERBOX_T3_CPU_S3GEN_GPU=1).\n");
            t3_use_gpu = false;
        } else {
            // Default split: T3 stays on CPU on Metal (kernel-launch overhead
            // across 30L × N_speech sequential AR steps is slower than CPU on
            // Apple Silicon). S3Gen GPU is now the default: the compound
            // Metal-precision drift in UNet1D CFM (PLAN #83) is fixed by
            // GGML_PREC_F32 mul_mat hints (mul_mat_hp()) + parallel=true sched.
            // Opt back into T3-GPU on Metal with CRISPASR_CHATTERBOX_T3_GPU=1.
            // S3Gen CPU opt-out: CRISPASR_CHATTERBOX_S3GEN_CPU=1.
#if defined(GGML_USE_METAL)
            const bool t3_gpu_default = false;
            const char* t3_default_reason =
                "Metal kernel-launch overhead × AR steps (override: CRISPASR_CHATTERBOX_T3_GPU=1)";
#elif defined(GGML_USE_VULKAN) && !defined(GGML_USE_CUDA)
            // Vulkan-only build: the §186 Lk-bucketed T3 step graph is allocated
            // once and reused across AR steps; on Vulkan that reuse segfaults in
            // ggml_vk_tensor_subbuffer (null buffer) at the rms_norm of step ~2
            // (issue #170). S3Gen GPU on Vulkan is fine, so keep T3 on CPU and
            // s3gen on GPU. Opt into the (crashing) T3-GPU path with
            // CRISPASR_CHATTERBOX_T3_GPU=1.
            const bool t3_gpu_default = false;
            const char* t3_default_reason =
                "Vulkan T3 step-graph segfault (issue #170; override: CRISPASR_CHATTERBOX_T3_GPU=1)";
#else
            const bool t3_gpu_default = true;
            const char* t3_default_reason = "non-Metal GPU";
#endif
            t3_use_gpu = t3_gpu_override ? true : t3_gpu_default;
            s3gen_use_gpu = !s3gen_cpu_override;
            fprintf(stderr,
                    "chatterbox: T3 → %s, s3gen → %s (%s). "
                    "Overrides: CRISPASR_CHATTERBOX_FORCE_GPU=1, CRISPASR_CHATTERBOX_FULL_CPU=1, "
                    "CRISPASR_CHATTERBOX_T3_GPU=1, CRISPASR_CHATTERBOX_S3GEN_CPU=1.\n",
                    t3_use_gpu ? "GPU" : "CPU", s3gen_use_gpu ? "GPU" : "CPU",
                    s3gen_use_gpu ? "UNet GGML_PREC_F32" : t3_default_reason);
        }
    }
    // Issue #94: flush so consumers see progress on slow-disk loads — the
    // 658 MB chatterbox-turbo T3 file can take 30-60 s on slow/external
    // disks, and a silent gap between this message and "precomputed conds
    // loaded" reads as a hang.
    std::fflush(stderr);
    // c->params.use_gpu controls the s3gen sub-context backend (set later
    // via chatterbox_set_s3gen_path). c->backend is the T3 backend.
    c->params.use_gpu = s3gen_use_gpu;
    bool effective_use_gpu = t3_use_gpu;
    c->backend = effective_use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        if (params.verbosity >= 1 && effective_use_gpu) {
            fprintf(stderr, "chatterbox: GPU backend unavailable, falling back to CPU\n");
        }
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights
    if (params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: loading T3 weights from %s\n", path_model);
        std::fflush(stderr);
    }
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
    if (params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: T3 loaded %zu tensors\n", c->tensors.size());
        std::fflush(stderr);
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
        std::fflush(stderr);
    }

    // Compute scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend != c->backend_cpu)
            backends[n_be++] = c->backend_cpu;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    // §188: cache embedding tables once; eliminates per-step tensor_get_f32.
    {
        const int D = (int)c->hp.hidden_size;
        const auto& m = c->t3;
        if (m.speech_emb_w) {
            c->speech_emb_cache.resize((size_t)c->hp.speech_vocab_size * D);
            tensor_get_f32(m.speech_emb_w, c->speech_emb_cache.data(), c->speech_emb_cache.size());
        }
        if (m.speech_pos_emb_w) {
            c->speech_pos_emb_cache.resize((size_t)c->hp.speech_pos_emb_size * D);
            tensor_get_f32(m.speech_pos_emb_w, c->speech_pos_emb_cache.data(), c->speech_pos_emb_cache.size());
        }
        if (m.text_emb_w) {
            c->text_emb_cache.resize((size_t)c->hp.text_vocab_size * D);
            tensor_get_f32(m.text_emb_w, c->text_emb_cache.data(), c->text_emb_cache.size());
        }
        if (m.text_pos_emb_w) {
            c->text_pos_emb_cache.resize((size_t)c->hp.text_pos_emb_size * D);
            tensor_get_f32(m.text_pos_emb_w, c->text_pos_emb_cache.data(), c->text_pos_emb_cache.size());
        }
        if (m.wpe_w) {
            c->wpe_cache.resize((size_t)c->hp.wpe_max_positions * D);
            tensor_get_f32(m.wpe_w, c->wpe_cache.data(), c->wpe_cache.size());
        }
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

    ctx->s3gen_ctx = chatterbox_s3gen_init_from_file(path, ctx->n_threads, ctx->params.verbosity, ctx->params.use_gpu);
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: failed to load S3Gen from %s\n", path);
        return -1;
    }
    return 0;
}

extern "C" int32_t* chatterbox_dump_text_tokens(struct chatterbox_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    // Mirror the tokenization block of chatterbox_synthesize_tokens exactly:
    // normalize (NFKD on the multilingual path, #170) → BPE tokenize →
    // prepend [lang]. Deterministic — no AR sampling.
    std::string norm_text = chatterbox_text_prep::normalize(text, !ctx->language.empty());
    std::vector<int32_t> text_tokens;
    if (ctx->tokenizer.has_bpe) {
        text_tokens = ctx->tokenizer.bpe_byte_level ? tokenize_text_bpe(ctx->tokenizer, norm_text)
                                                    : tokenize_text_hf_bpe(ctx->tokenizer, norm_text);
    } else {
        text_tokens = tokenize_text(ctx->tokenizer, norm_text);
    }
    prepend_language_token(ctx, text_tokens);

    const int n = (int)text_tokens.size();
    int32_t* out = (int32_t*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (!out)
        return nullptr;
    for (int i = 0; i < n; ++i)
        out[i] = text_tokens[(size_t)i];
    *out_n = n;
    return out;
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
    int64_t t_tok0 = ggml_time_us();
    std::string norm_text = chatterbox_text_prep::normalize(text, !ctx->language.empty());

    std::vector<int32_t> text_tokens;
    if (ctx->tokenizer.has_bpe) {
        text_tokens = ctx->tokenizer.bpe_byte_level ? tokenize_text_bpe(ctx->tokenizer, norm_text)
                                                    : tokenize_text_hf_bpe(ctx->tokenizer, norm_text);
    } else {
        text_tokens = tokenize_text(ctx->tokenizer, norm_text);
    }
    prepend_language_token(ctx, text_tokens);
    ctx->last_perf.t_tokenize_us = ggml_time_us() - t_tok0;
    ctx->last_perf.n_text_tokens = (int)text_tokens.size();

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: text \"%s\" → %zu %s tokens\n", norm_text.c_str(), text_tokens.size(),
                ctx->tokenizer.has_bpe ? "BPE" : "char");
    }

    // 2. Add start/stop text tokens. Base chatterbox wraps text with
    // [SOT, ..., EOT] via _ensure_BOT_EOT in its inference path
    // (chatterbox/models/t3/t3.py:255). The turbo path (`is_gpt2 == true`
    // → ChatterboxTurboTTS.inference_turbo, t3.py:415) does NOT call
    // _ensure_BOT_EOT — it feeds the bare tokenizer output to the
    // GPT-2 backbone. Adding SOT/EOT for turbo shifts every WPE
    // position by 2 and was the dominant audio-quality defect in #94.
    if (!is_gpt2) {
        text_tokens.insert(text_tokens.begin(), (int32_t)ctx->hp.start_text_token);
        text_tokens.push_back((int32_t)ctx->hp.stop_text_token);
    }

    if (ctx->params.verbosity >= 2 || std::getenv("CHATTERBOX_DEBUG")) {
        fprintf(stderr, "chatterbox: text_tokens(%zu) %s = [", text_tokens.size(),
                is_gpt2 ? "(no SOT/EOT for turbo)" : "[SOT,...,EOT]");
        for (size_t i = 0; i < text_tokens.size(); ++i) {
            fprintf(stderr, "%d%s", (int)text_tokens[i], i + 1 == text_tokens.size() ? "" : ", ");
        }
        fprintf(stderr, "]\n");
    }

    // 3. Allocate KV cache
    const int max_speech = ctx->params.max_speech_tokens;
    int max_ctx = (int)text_tokens.size() + max_speech + 64; // generous padding
    if (!kv_alloc(ctx, max_ctx)) {
        fprintf(stderr, "chatterbox: failed to allocate KV cache\n");
        return nullptr;
    }

    // 4. Build prefill embeddings on CPU
    int64_t t_preb0 = ggml_time_us();
    std::vector<float> prefill_embeds;
    if (is_gpt2) {
        prefill_embeds = build_prefill_embeds_gpt2(ctx, text_tokens);
    } else {
        prefill_embeds = build_prefill_embeds(ctx, text_tokens);
        // Python's T3.inference() appends a second speech-start embed (pos=0) on top of
        // prepare_input_embeds output before running the transformer, so the prefill ends
        // with [...|speech_start@pos0|BOS@pos0]. The duplicate is identical to the last
        // token but shifts the last RoPE position from N-1 to N, which is load-bearing
        // for the first-step logits.
        const int D = (int)ctx->hp.hidden_size;
        std::vector<float> bos_extra(prefill_embeds.end() - D, prefill_embeds.end());
        prefill_embeds.insert(prefill_embeds.end(), bos_extra.begin(), bos_extra.end());
    }
    int prefill_len = (int)(prefill_embeds.size() / ctx->hp.hidden_size);
    ctx->last_perf.t_prefill_build_us = ggml_time_us() - t_preb0;
    ctx->last_perf.n_prefill_tokens = prefill_len;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: prefill %d tokens (max_speech=%d)\n", prefill_len, max_speech);
    }

    // 5. CFG setup: build unconditioned prefill if cfg_weight > 0
    const float cfg_w = ctx->params.cfg_weight;
    const bool use_cfg = (!is_gpt2 && cfg_w > 0.0f && ctx->kv_k_cfg);
    // §214: opt into the batched-CFG B=2 decode (cond+uncond as one B=2 forward).
    // Default OFF (legacy sequential cond+uncond) until token-parity + a real
    // quiet-machine speedup are demonstrated. Requires the CFG uncond cache.
    bool use_b2 = use_cfg && []() {
        const char* s = std::getenv("CRISPASR_CHATTERBOX_T3_CFG_B2");
        return s && (s[0] == '1' || s[0] == 'y' || s[0] == 'Y');
    }();
    // B2 is bit-identical to the legacy sequential cond+uncond path on CPU (all
    // weight quants) and on GPU with F16 weights, but DEGENERATES on GPU with
    // quantized T3 weights: Metal's batched (ne[2]=2) quantized mat-vec does not
    // hit the PREC_F32 mul_mv_q*_K exact-dot kernel that the single-token legacy
    // decode uses, so the batched quant matmul drifts and the speech-token loop
    // collapses into repetition. (Same class as the §211 batched-quant-CFM
    // garbage.) Restrict B2 to the known-safe configs. T3 defaults to CPU on
    // Metal/Vulkan, so this only bites the explicit T3_GPU + quant + B2 combo.
    if (use_b2 && ctx->backend != ctx->backend_cpu && !ctx->t3.blocks.empty() &&
        ggml_is_quantized(ctx->t3.blocks[0].attn_q_w->type)) {
        use_b2 = false;
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "chatterbox: T3 CFG B2 disabled — GPU + quantized T3 weights "
                            "(Metal batched quant matmul unsupported); use F16 T3 or run T3 on CPU.\n");
    }
    if (use_b2 && ctx->params.verbosity >= 1)
        fprintf(stderr, "chatterbox: T3 CFG B2 (batched cond+uncond decode) ENABLED\n");
    std::vector<float> uncond_embeds;
    if (use_cfg) {
        // Unconditioned CFG path in Python zeros only text token embeddings
        // before adding learned text positions, so the text span becomes
        // text_pos_emb(pos) rather than all zeros.
        uncond_embeds = prefill_embeds; // copy
        const int D = ctx->hp.hidden_size;
        int text_start = prefill_len - (int)text_tokens.size() - 2; // cond_len
        int text_end = prefill_len - 2;                             // before speech tokens
        const auto& text_pos_table = ctx->text_pos_emb_cache;
        for (int i = text_start; i < text_end; i++) {
            const int pos_idx = i - text_start;
            std::memcpy(&uncond_embeds[(size_t)i * D], &text_pos_table[(size_t)pos_idx * D], (size_t)D * sizeof(float));
        }
    }

    // 6. Prefill: run the full prefix through the transformer
    int n_past = 0;
    float* logits = nullptr;
    int64_t t_pre0 = ggml_time_us();
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
    ctx->last_perf.t_prefill_us = ggml_time_us() - t_pre0;

    // 7. AR decode loop with CFG
    std::vector<int32_t> speech_tokens;
    speech_tokens.reserve(max_speech);
    int speech_pos = 1;

    int64_t t_dec0 = ggml_time_us();
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
        // Python's HF inference_turbo passes input_ids = generated_speech_tokens
        // (NO BOS) to RepetitionPenaltyLogitsProcessor from step 1 onward; only
        // at step 0 is input_ids = speech_start_token = [BOS] (t3.py:450 vs
        // t3.py:471). Match that exactly here: at step 0 token_hist = [BOS]
        // (penalizing the BOS logit, which is normally far below speech
        // tokens anyway); from step 1 onward token_hist = generated tokens.
        std::vector<int32_t> token_hist;
        if (speech_tokens.empty()) {
            token_hist.push_back((int32_t)ctx->hp.start_speech_token);
        } else {
            token_hist.reserve(speech_tokens.size());
            token_hist.insert(token_hist.end(), speech_tokens.begin(), speech_tokens.end());
        }
        // Sample next token. CRISPASR_CHATTERBOX_TEMP overrides the
        // configured temperature for divergence-debugging experiments
        // (temperature=0 forces greedy argmax — eliminates multinomial
        // sensitivity to small logit drifts so we can isolate kernel
        // numerical issues from sampler boundary flips).
        float temp_eff = ctx->params.temperature;
        if (const char* e = std::getenv("CRISPASR_CHATTERBOX_TEMP"); e && *e) {
            temp_eff = std::strtof(e, nullptr);
        }
        int32_t tok = sample_token(blended.data(), V, temp_eff, ctx->params.top_k, ctx->params.min_p, ctx->params.top_p,
                                   ctx->params.repetition_penalty, token_hist, ctx->rng_state);
        free(logits);
        logits = nullptr;
        if (logits_uncond) {
            free(logits_uncond);
            logits_uncond = nullptr;
        }

        if (ctx->params.verbosity >= 1 && step < 32) {
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

        if (use_b2) {
            // §214: run cond + uncond as one batch-2 forward (n_past == n_past_cfg
            // by construction). One weight read serves both passes.
            logits = (float*)malloc((size_t)V * sizeof(float));
            logits_uncond = (float*)malloc((size_t)V * sizeof(float));
            if (!run_t3_kv_b2(ctx, tok_embed.data(), n_past, logits, logits_uncond)) {
                fprintf(stderr, "chatterbox: decode step %d (b2) failed\n", step);
                free(logits);
                logits = nullptr;
                free(logits_uncond);
                logits_uncond = nullptr;
                break;
            }
            n_past++;
            n_past_cfg++;
        } else {
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
    }
    if (logits)
        free(logits);
    if (logits_uncond)
        free(logits_uncond);
    ctx->last_perf.t_decode_us = ggml_time_us() - t_dec0;
    ctx->last_perf.n_speech_tokens = (int)speech_tokens.size();

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

    // Issue #94 follow-up: chatterbox-turbo's Python implementation appends
    // 3 S3GEN_SIL silence tokens after the AR output before handing tokens
    // to s3gen (chatterbox/tts_turbo.py:286-287). Without this padding the
    // generated mel cuts off abruptly — the last-frame silence is what
    // chatterbox-turbo's meanflow s3gen was trained on. S3GEN_SIL=4299
    // comes from chatterbox/models/s3gen/const.py. Base chatterbox doesn't
    // do this (chatterbox/tts.py path).
    if (is_gpt2) {
        constexpr int32_t S3GEN_SIL = 4299;
        valid.push_back(S3GEN_SIL);
        valid.push_back(S3GEN_SIL);
        valid.push_back(S3GEN_SIL);
    }

    if (ctx->params.verbosity >= 2 || std::getenv("CHATTERBOX_DEBUG")) {
        fprintf(stderr, "chatterbox: speech_tokens(%zu) first=[", valid.size());
        for (size_t i = 0; i < valid.size() && i < 12; ++i)
            fprintf(stderr, "%d%s", (int)valid[i], i + 1 == valid.size() || i == 11 ? "" : ",");
        fprintf(stderr, "] last=[");
        size_t tail_start = valid.size() > 6 ? valid.size() - 6 : 0;
        for (size_t i = tail_start; i < valid.size(); ++i)
            fprintf(stderr, "%d%s", (int)valid[i], i + 1 == valid.size() ? "" : ",");
        fprintf(stderr, "]\n");
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

    float* mel_cf =
        chatterbox_s3gen_synthesize_mel(ctx->s3gen_ctx, speech_tokens, n_tokens, prompt_tokens, n_prompt, prompt_feat,
                                        prompt_feat_len, spk_emb, ctx->params.cfm_steps, out_T_mel);
    chatterbox_tokens_free(speech_tokens);
    if (!mel_cf || *out_T_mel <= 0) {
        if (mel_cf)
            chatterbox_s3gen_pcm_free(mel_cf);
        return {};
    }
    std::vector<float> mel(mel_cf, mel_cf + (size_t)80 * (size_t)(*out_T_mel));
    chatterbox_s3gen_pcm_free(mel_cf);
    return mel;
}

extern "C" float* chatterbox_synthesize_mel(struct chatterbox_context* ctx, const char* text, int* out_T_mel) {
    if (!ctx || !text || !out_T_mel)
        return nullptr;
    *out_T_mel = 0;
    std::vector<float> mel = synthesize_mel_internal(ctx, text, out_T_mel);
    if (mel.empty() || *out_T_mel <= 0)
        return nullptr;
    float* out = (float*)malloc(mel.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, mel.data(), mel.size() * sizeof(float));
    return out;
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
    int64_t t_t3_0 = ggml_time_us();
    int n_tokens = 0;
    int32_t* speech_tokens;
    {
        chatterbox_bench_stage _b("t3_ar_decode");
        speech_tokens = chatterbox_synthesize_tokens(ctx, text, &n_tokens);
    }
    int64_t t_t3_us = ggml_time_us() - t_t3_0;
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

    if (!prompt_tokens && !spk_emb) {
        const bool is_gpt2 = (ctx->hp.arch == "chatterbox_turbo" || ctx->hp.arch == "kartoffelbox");
        if (is_gpt2) {
            fprintf(stderr, "chatterbox-turbo: WARNING — no voice conditioning loaded. "
                            "Audio will be unconditioned (noisy/quiet). Use --voice ref.wav or "
                            "ensure the T3 GGUF includes baked conds.pt tensors.\n");
        }
    }

    int64_t t_s3_0 = ggml_time_us();
    float* pcm;
    {
        chatterbox_bench_stage _b("s3gen_vocoder");
        pcm = chatterbox_s3gen_synthesize(ctx->s3gen_ctx, speech_tokens, n_tokens, prompt_tokens, n_prompt, prompt_feat,
                                          prompt_feat_len, spk_emb, ctx->params.cfm_steps, out_n_samples);
    }
    int64_t t_s3gen_us = ggml_time_us() - t_s3_0;

    chatterbox_tokens_free(speech_tokens);

    if (pcm && *out_n_samples > 0) {
        const char* bench = std::getenv("CHATTERBOX_BENCH");
        if (bench && bench[0]) {
            const auto& tp = ctx->last_perf;
            chatterbox_s3gen_perf sp{};
            chatterbox_s3gen_get_perf(ctx->s3gen_ctx, &sp);
            const double audio_s = (double)*out_n_samples / 24000.0;
            const double total_s = (t_t3_us + t_s3gen_us) / 1e6;
            fprintf(stderr, "chatterbox: =========== perf report ===========\n");
            fprintf(stderr, "chatterbox:  audio duration   %7.2f s  (%d samples @ 24 kHz)\n", audio_s, *out_n_samples);
            fprintf(stderr, "chatterbox:  text tokens      %7d\n", tp.n_text_tokens);
            fprintf(stderr, "chatterbox:  prefill tokens   %7d\n", tp.n_prefill_tokens);
            fprintf(stderr, "chatterbox:  speech tokens    %7d\n", tp.n_speech_tokens);
            fprintf(stderr, "chatterbox:  mel frames       %7d  (%.2f s @ 80 Hz)\n", sp.T_mel, sp.T_mel / 80.0f);
            fprintf(stderr, "chatterbox: ------- T3 (AR decode) -------------\n");
            fprintf(stderr, "chatterbox:   tokenize        %7.1f ms\n", tp.t_tokenize_us / 1e3);
            fprintf(stderr, "chatterbox:   prefill build   %7.1f ms\n", tp.t_prefill_build_us / 1e3);
            fprintf(stderr, "chatterbox:   prefill compute %7.1f ms\n", tp.t_prefill_us / 1e3);
            fprintf(stderr, "chatterbox:   decode loop     %7.1f ms  (%d tokens, %.2f ms/tok)\n", tp.t_decode_us / 1e3,
                    tp.n_speech_tokens, tp.n_speech_tokens > 0 ? tp.t_decode_us / 1e3 / tp.n_speech_tokens : 0.0);
            fprintf(stderr, "chatterbox:   T3 total        %7.1f ms\n", t_t3_us / 1e3);
            fprintf(stderr, "chatterbox: ------- S3Gen (CFM + vocoder) ------\n");
            fprintf(stderr, "chatterbox:   encoder         %7.1f ms\n", sp.t_encoder_us / 1e3);
            fprintf(stderr, "chatterbox:   CFM euler       %7.1f ms  (%d steps, %.1f ms/step)\n", sp.t_cfm_us / 1e3,
                    sp.n_cfm_steps, sp.n_cfm_steps > 0 ? sp.t_cfm_us / 1e3 / sp.n_cfm_steps : 0.0);
            fprintf(stderr, "chatterbox:   HiFT vocoder    %7.1f ms\n", sp.t_vocoder_us / 1e3);
            fprintf(stderr, "chatterbox:   S3Gen total     %7.1f ms\n", t_s3gen_us / 1e3);
            fprintf(stderr, "chatterbox: ------- totals ---------------------------\n");
            fprintf(stderr, "chatterbox:   wall time       %7.1f ms\n", total_s * 1e3);
            fprintf(stderr, "chatterbox:   RTF             %7.3f  (%.1fx real-time)\n", total_s / audio_s,
                    audio_s / total_s);
        }
    }

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

extern "C" float* chatterbox_synthesize_mel_from_tokens(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                                        int n_speech_tokens, int* out_T_mel) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_T_mel)
        return nullptr;
    *out_T_mel = 0;
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: S3Gen not loaded.\n");
        return nullptr;
    }
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
    return chatterbox_s3gen_synthesize_mel(ctx->s3gen_ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt,
                                           prompt_feat, prompt_feat_len, spk_emb, ctx->params.cfm_steps, out_T_mel);
}

extern "C" float* chatterbox_synthesize_mel_from_tokens_with_noise(struct chatterbox_context* ctx,
                                                                   const int32_t* speech_tokens, int n_speech_tokens,
                                                                   const float* init_noise_cf, int init_noise_T_total,
                                                                   int* out_T_mel) {
    if (!ctx || !speech_tokens || n_speech_tokens <= 0 || !out_T_mel || !init_noise_cf || init_noise_T_total <= 0)
        return nullptr;
    *out_T_mel = 0;
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: S3Gen not loaded.\n");
        return nullptr;
    }
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
    return chatterbox_s3gen_synthesize_mel_with_noise(
        ctx->s3gen_ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt, prompt_feat, prompt_feat_len, spk_emb,
        ctx->params.cfm_steps, init_noise_cf, init_noise_T_total, out_T_mel);
}

extern "C" float* chatterbox_vocode_mel(struct chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                        int* out_n_samples) {
    return chatterbox_vocode_mel_with_source_stft(ctx, mel_cf, T_mel, nullptr, 0, out_n_samples);
}

extern "C" float* chatterbox_vocode_mel_with_source_stft(struct chatterbox_context* ctx, const float* mel_cf, int T_mel,
                                                         const float* source_stft_cf, int T_src, int* out_n_samples) {
    if (!ctx || !mel_cf || T_mel <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: S3Gen not loaded.\n");
        return nullptr;
    }
    return chatterbox_s3gen_vocode_with_source_stft(ctx->s3gen_ctx, mel_cf, T_mel, source_stft_cf, T_src,
                                                    out_n_samples);
}

extern "C" float* chatterbox_vocode_mel_dump_with_source_stft(struct chatterbox_context* ctx, const float* mel_cf,
                                                              int T_mel, const float* source_stft_cf, int T_src,
                                                              int* out_n_samples, const char** stage_names,
                                                              float** stage_data, int* stage_sizes, int n_stages) {
    if (!ctx || !mel_cf || T_mel <= 0 || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;
    if (!ctx->s3gen_ctx) {
        fprintf(stderr, "chatterbox: S3Gen not loaded.\n");
        return nullptr;
    }
    return chatterbox_s3gen_vocode_dump_with_source_stft(ctx->s3gen_ctx, mel_cf, T_mel, source_stft_cf, T_src,
                                                         out_n_samples, stage_names, stage_data, stage_sizes, n_stages);
}

extern "C" float* chatterbox_hift_from_conv_post(const float* stft_cf, int T_stft, int T_mel, int* out_n_samples) {
    return chatterbox_s3gen_hift_from_conv_post(stft_cf, T_stft, T_mel, out_n_samples);
}

// Internal: replace the precomputed-conds tensor pointers from a freshly-loaded
// voice GGUF (baked by tools/bake-chatterbox-voice-from-wav.py). Tensor names
// match the ones the converter writes for the built-in voice
// (see chatterbox.cpp:846-851 and convert-chatterbox-to-gguf.py:521-548):
//   conds.t3.speaker_emb           f32  (1, 256)
//   conds.t3.speech_prompt_tokens  i32  (T_prompt,)
//   conds.gen.prompt_token         i32  (T_speech_tokens,)
//   conds.gen.prompt_feat          f32  (T_mel, 80)
//   conds.gen.embedding            f32  (1, 192)
// Plus optional metadata:
//   chatterbox.conds.emotion_adv          f32
//   chatterbox.conds.gen_prompt_token_len u32
//
// Side-effect: frees any previous voice GGUF buffer; original baked-in
// conds tensors in ctx_w stay allocated but become unreferenced (they
// are freed when the model context is destroyed). Returns 0 on success.
static int chatterbox_load_voice_gguf(chatterbox_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    // Drop any previously-loaded voice GGUF before we read the new one,
    // so a failed load doesn't leak the old buffer.
    if (ctx->voice_ctx_w) {
        ggml_free(ctx->voice_ctx_w);
        ctx->voice_ctx_w = nullptr;
    }
    if (ctx->voice_buf_w) {
        ggml_backend_buffer_free(ctx->voice_buf_w);
        ctx->voice_buf_w = nullptr;
    }
    ctx->voice_tensors.clear();

    // Optional scalar metadata (emotion_adv, gen_prompt_token_len) lives in
    // the GGUF kv table — read it via the metadata-only pass first.
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            fprintf(stderr, "chatterbox: voice GGUF '%s' could not be opened\n", path);
            return -1;
        }
        ctx->conds.emotion_adv = core_gguf::kv_f32(g, "chatterbox.conds.emotion_adv", ctx->conds.emotion_adv);
        ctx->conds.gen_prompt_token_len =
            core_gguf::kv_u32(g, "chatterbox.conds.gen_prompt_token_len", ctx->conds.gen_prompt_token_len);
        core_gguf::free_metadata(g);
    }

    // Load tensors into a fresh ctx + backend buffer. We share the model's
    // primary `backend` so conds tensors live alongside the model weights
    // (avoiding cross-backend tensor_get rounds during synthesise).
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path, ctx->backend, "chatterbox-voice", wl)) {
            fprintf(stderr, "chatterbox: voice GGUF '%s' failed to load tensors\n", path);
            return -1;
        }
        ctx->voice_ctx_w = wl.ctx;
        ctx->voice_buf_w = wl.buf;
        ctx->voice_tensors = std::move(wl.tensors);
    }

    // Re-bind conds.* to the voice's tensors. Anything missing in the voice
    // GGUF stays at its previous value — useful for partial bundles, but
    // the canonical baker always writes the full set.
    ggml_tensor* t = nullptr;
    if ((t = core_gguf::try_get(ctx->voice_tensors, "conds.t3.speaker_emb")))
        ctx->conds.speaker_emb = t;
    if ((t = core_gguf::try_get(ctx->voice_tensors, "conds.t3.speech_prompt_tokens")))
        ctx->conds.speech_prompt_tokens = t;
    if ((t = core_gguf::try_get(ctx->voice_tensors, "conds.gen.prompt_token"))) {
        ctx->conds.gen_prompt_token = t;
        // The metadata-derived length wins when present; otherwise infer.
        if (ctx->conds.gen_prompt_token_len == 0) {
            ctx->conds.gen_prompt_token_len = (uint32_t)t->ne[0];
        }
    }
    if ((t = core_gguf::try_get(ctx->voice_tensors, "conds.gen.prompt_feat")))
        ctx->conds.gen_prompt_feat = t;
    if ((t = core_gguf::try_get(ctx->voice_tensors, "conds.gen.embedding")))
        ctx->conds.gen_embedding = t;

    ctx->conds.loaded = (ctx->conds.speaker_emb != nullptr);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "chatterbox: voice loaded from '%s' (%zu tensors, emotion_adv=%.2f)\n", path,
                ctx->voice_tensors.size(), (double)ctx->conds.emotion_adv);
    }
    return ctx->conds.loaded ? 0 : -1;
}

// Public entry point. Routes by file extension: ``.gguf`` paths load a
// pre-baked voice bundle; everything else is treated as a reference WAV
// and (currently) returns -1 with a pointer at the baker script. Native
// WAV → cond extraction (VE LSTM, CAMPPlus TDNN, S3Tokenizer) is a
// separate refactor — see PLAN entry on chatterbox voice cloning.
// Decode a tiny WAV file into 16 kHz mono float32 PCM. Supports PCM16 and
// IEEE-float WAV files at exactly 16 kHz; mono passes through, stereo gets
// averaged. Anything else returns false with a clear error.
//
// The chatterbox runtime reaches for this only on the .wav branch of
// chatterbox_decode_wav_mono — decodes a small WAV (PCM16 / IEEE-float)
// at exactly `expected_sr` (16000 OR 24000 supported in this caller's
// flow). Stereo is averaged to mono. Anything else returns false with a
// clear error and the caller can either point the user at ffmpeg or
// fall back to the python baker.
//
// 24 kHz input enables the atomic native voice-clone path: the runtime
// resamples to 16 kHz (via `core_audio::resample_polyphase`, kaiser-
// windowed sinc) for VE / S3Tokenizer / CAMPPlus, and uses the original
// 24 kHz directly for the prompt mel. 16 kHz input keeps the older
// partial-clone behaviour (T3-side conds only).
static bool chatterbox_decode_wav_mono(const char* path, int expected_sr, std::vector<float>& out_pcm,
                                       std::string& err) {
    out_pcm.clear();
    err.clear();
    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        err = std::string("could not open: ") + path;
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    long fsize = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (fsize <= 12) {
        std::fclose(fp);
        err = "file too small to be WAV";
        return false;
    }
    std::vector<uint8_t> buf((size_t)fsize);
    const size_t rd = std::fread(buf.data(), 1, buf.size(), fp);
    std::fclose(fp);
    if (rd != buf.size()) {
        err = "fread truncated";
        return false;
    }
    auto rd_u16 = [&](size_t off) -> uint16_t { return (uint16_t)buf[off] | ((uint16_t)buf[off + 1] << 8); };
    auto rd_u32 = [&](size_t off) -> uint32_t {
        return (uint32_t)buf[off] | ((uint32_t)buf[off + 1] << 8) | ((uint32_t)buf[off + 2] << 16) |
               ((uint32_t)buf[off + 3] << 24);
    };
    if (std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return false;
    }
    uint32_t audio_format = 0, channels = 0, sr = 0, bits_per_sample = 0;
    size_t data_offset = 0;
    uint32_t data_size = 0;
    size_t off = 12;
    while (off + 8 <= buf.size()) {
        const uint32_t chunk_sz = rd_u32(off + 4);
        const size_t chunk_data = off + 8;
        if (chunk_data + (size_t)chunk_sz > buf.size())
            break;
        if (std::memcmp(buf.data() + off, "fmt ", 4) == 0 && chunk_sz >= 16) {
            audio_format = rd_u16(chunk_data);
            channels = rd_u16(chunk_data + 2);
            sr = rd_u32(chunk_data + 4);
            bits_per_sample = rd_u16(chunk_data + 14);
        } else if (std::memcmp(buf.data() + off, "data", 4) == 0) {
            data_offset = chunk_data;
            data_size = chunk_sz;
        }
        off = chunk_data + chunk_sz + (chunk_sz & 1u);
    }
    if (data_offset == 0) {
        err = "no data chunk";
        return false;
    }
    if ((int)sr != expected_sr) {
        char tmp[160];
        std::snprintf(tmp, sizeof(tmp), "sample rate %u not supported (need %d); pre-convert or use the python baker",
                      sr, expected_sr);
        err = tmp;
        return false;
    }
    if (channels != 1 && channels != 2) {
        err = "channel count not supported (need 1 or 2)";
        return false;
    }
    const bool is_pcm = (audio_format == 1);
    const bool is_float = (audio_format == 3);
    if (!is_pcm && !is_float) {
        err = "WAV format must be PCM (1) or IEEE-float (3)";
        return false;
    }
    if (is_pcm && bits_per_sample != 16) {
        err = "PCM bits per sample must be 16";
        return false;
    }
    if (is_float && bits_per_sample != 32) {
        err = "float WAV bits per sample must be 32";
        return false;
    }

    const uint32_t bytes_per_sample = bits_per_sample / 8;
    const uint64_t total_samples = (uint64_t)data_size / bytes_per_sample;
    if (total_samples == 0) {
        err = "empty data chunk";
        return false;
    }
    const uint64_t n_frames = total_samples / channels;
    out_pcm.assign((size_t)n_frames, 0.0f);
    const uint8_t* src = buf.data() + data_offset;
    if (is_pcm) {
        for (uint64_t i = 0; i < n_frames; i++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; c++) {
                const size_t b = (size_t)(i * channels + c) * 2u;
                const int16_t s = (int16_t)((uint16_t)src[b] | ((uint16_t)src[b + 1] << 8));
                sum += (float)s / 32768.0f;
            }
            out_pcm[(size_t)i] = sum / (float)channels;
        }
    } else {
        // is_float, 32-bit IEEE-float
        for (uint64_t i = 0; i < n_frames; i++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < channels; c++) {
                float v;
                std::memcpy(&v, src + (size_t)(i * channels + c) * 4u, sizeof(float));
                sum += v;
            }
            out_pcm[(size_t)i] = sum / (float)channels;
        }
    }
    return true;
}

// Replace ctx->conds.* tensors from a freshly-computed native bundle.
// `emb` is the 256-d speaker embedding (Module 2 / VE), `prompt_tokens`
// and `speech_prompt_tokens` are the int32 token streams from
// S3Tokenizer V2 (Module 3, full audio + first 6 s respectively).
// The remaining conds (`gen_prompt_feat` 24 kHz mel, `gen_embedding`
// 192-d CAMPPlus x-vector) keep pointing at the default voice's baked-in
// tensors until module 4 lands, so S3Gen will still render with the
// default voice's timbre — but T3 sees the new speaker_emb AND speech
// prompt tokens, and S3Gen sees the new prompt_token, so the rendered
// prosody is the cloned speaker's even if the timbre lags.
static int chatterbox_install_native_voice(chatterbox_context* ctx, const float emb[256], const int32_t* prompt_tokens,
                                           int n_prompt_tokens, const int32_t* speech_prompt_tokens,
                                           int n_speech_prompt_tokens, const float* prompt_feat, int T_prompt_feat,
                                           const float* gen_embedding) {
    if (!ctx || !emb)
        return -1;

    // Drop any previously loaded voice GGUF / native bundle.
    if (ctx->voice_ctx_w) {
        ggml_free(ctx->voice_ctx_w);
        ctx->voice_ctx_w = nullptr;
    }
    if (ctx->voice_buf_w) {
        ggml_backend_buffer_free(ctx->voice_buf_w);
        ctx->voice_buf_w = nullptr;
    }
    ctx->voice_tensors.clear();

    // Up to 5 tensors: speaker_emb + speech_prompt_tokens + prompt_token
    // + prompt_feat + gen.embedding. The atomic install only fires when
    // all five are available — partial fills are still supported for
    // earlier-module-only flows (M2-only, M2+M3-only).
    ggml_init_params ip = {ggml_tensor_overhead() * 12, nullptr, true};
    ggml_context* vctx = ggml_init(ip);
    if (!vctx) {
        fprintf(stderr, "chatterbox: ggml_init failed for native voice\n");
        return -1;
    }

    ggml_tensor* spkr = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, 256, 1);
    ggml_set_name(spkr, "conds.t3.speaker_emb");

    ggml_tensor* prompt_t = nullptr;
    if (prompt_tokens && n_prompt_tokens > 0) {
        prompt_t = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, n_prompt_tokens);
        ggml_set_name(prompt_t, "conds.gen.prompt_token");
    }
    ggml_tensor* speech_prompt_t = nullptr;
    if (speech_prompt_tokens && n_speech_prompt_tokens > 0) {
        speech_prompt_t = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, n_speech_prompt_tokens);
        ggml_set_name(speech_prompt_t, "conds.t3.speech_prompt_tokens");
    }
    // prompt_feat: (1, T, 80) row-major in the converter / baker; in
    // ggml ne=(80, T, 1) (ne[0]=80 fastest, ne[1]=T, ne[2]=1).
    ggml_tensor* prompt_feat_t = nullptr;
    if (prompt_feat && T_prompt_feat > 0) {
        prompt_feat_t = ggml_new_tensor_3d(vctx, GGML_TYPE_F32, 80, T_prompt_feat, 1);
        ggml_set_name(prompt_feat_t, "conds.gen.prompt_feat");
    }
    // gen.embedding: (1, 192) — ggml ne=(192, 1).
    ggml_tensor* gen_emb_t = nullptr;
    if (gen_embedding) {
        gen_emb_t = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, 192, 1);
        ggml_set_name(gen_emb_t, "conds.gen.embedding");
    }

    ggml_backend_buffer_t vbuf = ggml_backend_alloc_ctx_tensors(vctx, ctx->backend);
    if (!vbuf) {
        fprintf(stderr, "chatterbox: failed to alloc native voice buffer\n");
        ggml_free(vctx);
        return -1;
    }
    ggml_backend_tensor_set(spkr, emb, 0, 256 * sizeof(float));
    if (prompt_t)
        ggml_backend_tensor_set(prompt_t, prompt_tokens, 0, (size_t)n_prompt_tokens * sizeof(int32_t));
    if (speech_prompt_t)
        ggml_backend_tensor_set(speech_prompt_t, speech_prompt_tokens, 0,
                                (size_t)n_speech_prompt_tokens * sizeof(int32_t));
    if (prompt_feat_t)
        ggml_backend_tensor_set(prompt_feat_t, prompt_feat, 0, (size_t)T_prompt_feat * 80 * sizeof(float));
    if (gen_emb_t)
        ggml_backend_tensor_set(gen_emb_t, gen_embedding, 0, 192 * sizeof(float));

    ctx->voice_ctx_w = vctx;
    ctx->voice_buf_w = vbuf;
    ctx->voice_tensors["conds.t3.speaker_emb"] = spkr;
    ctx->conds.speaker_emb = spkr;
    if (prompt_t) {
        ctx->voice_tensors["conds.gen.prompt_token"] = prompt_t;
        ctx->conds.gen_prompt_token = prompt_t;
        ctx->conds.gen_prompt_token_len = (uint32_t)n_prompt_tokens;
    }
    if (speech_prompt_t) {
        ctx->voice_tensors["conds.t3.speech_prompt_tokens"] = speech_prompt_t;
        ctx->conds.speech_prompt_tokens = speech_prompt_t;
    }
    if (prompt_feat_t) {
        ctx->voice_tensors["conds.gen.prompt_feat"] = prompt_feat_t;
        ctx->conds.gen_prompt_feat = prompt_feat_t;
    }
    if (gen_emb_t) {
        ctx->voice_tensors["conds.gen.embedding"] = gen_emb_t;
        ctx->conds.gen_embedding = gen_emb_t;
    }
    ctx->conds.loaded = true;
    return 0;
}

extern "C" int chatterbox_set_voice_from_wav(struct chatterbox_context* ctx, const char* wav_path) {
    if (!ctx || !wav_path)
        return -1;

    auto ends_with = [](const char* s, const char* suffix) {
        const size_t ls = std::strlen(s);
        const size_t lx = std::strlen(suffix);
        if (ls < lx)
            return false;
        return std::strcmp(s + ls - lx, suffix) == 0;
    };

    if (ends_with(wav_path, ".gguf") || ends_with(wav_path, ".GGUF")) {
        return chatterbox_load_voice_gguf(ctx, wav_path);
    }

    // Native WAV cloning. The path forks on the input sample rate:
    //   - 24 kHz mono WAV: full atomic clone — all 5 conds are derived
    //     from the same reference audio and installed together. The 16 kHz
    //     versions used by VE / S3Tokenizer / CAMPPlus come from a
    //     core_audio polyphase resampler (kaiser-windowed sinc, β=8.6).
    //   - 16 kHz mono WAV: partial M2+M3 clone (T3-side conds only) —
    //     gen.{prompt_token, prompt_feat, embedding} stay at default
    //     voice values. Updating gen.prompt_token alone without the
    //     matching prompt_feat / embedding feeds S3Gen's flow matcher
    //     inconsistent conditioning and silences the output (verified).
    std::vector<float> pcm_24k;
    std::vector<float> pcm_16k_owner; // owns the 16 k buffer when it's
                                      // resampled from 24 kHz
    const float* pcm_16k = nullptr;
    int n_16k = 0;
    int n_24k = 0;
    bool atomic_path = false;

    {
        std::vector<float> pcm_native;
        std::string err24;
        // Try 24 kHz first — atomic path needs it.
        if (chatterbox_decode_wav_mono(wav_path, 24000, pcm_native, err24)) {
            pcm_24k = std::move(pcm_native);
            n_24k = (int)pcm_24k.size();
            // Resample 24 → 16 kHz for VE / S3Tokenizer / CAMPPlus.
            pcm_16k_owner = core_audio::resample_polyphase(pcm_24k.data(), n_24k, 24000, 16000);
            pcm_16k = pcm_16k_owner.data();
            n_16k = (int)pcm_16k_owner.size();
            atomic_path = true;
        } else {
            std::string err16;
            std::vector<float> pcm_16k_native;
            if (!chatterbox_decode_wav_mono(wav_path, 16000, pcm_16k_native, err16)) {
                fprintf(
                    stderr,
                    "chatterbox: native WAV cloning failed.\n"
                    "  Tried 24 kHz: %s\n  Tried 16 kHz: %s\n"
                    "  Re-encode the reference (`ffmpeg -i %s -ar 24000 -ac 1 ref.wav`) — 24 kHz mono PCM16/F32 "
                    "enables full atomic cloning, 16 kHz keeps the partial M2+M3 path. Or fall back to the python "
                    "baker (`python models/bake-chatterbox-voice-from-wav.py --input %s --output my_voice.gguf`).\n",
                    err24.c_str(), err16.c_str(), wav_path, wav_path);
                return -1;
            }
            pcm_16k_owner = std::move(pcm_16k_native);
            pcm_16k = pcm_16k_owner.data();
            n_16k = (int)pcm_16k_owner.size();
        }
    }

    // VE — 256-d speaker embedding from the 16 kHz audio.
    float emb[256];
    if (!chatterbox_ve::compute_speaker_emb(ctx->ve, ctx->sched, ctx->compute_meta, pcm_16k, n_16k, emb)) {
        fprintf(stderr, "chatterbox: VoiceEncoder forward failed for '%s'\n", wav_path);
        return -1;
    }

    // S3Tokenizer V2 — first 6 s capped at 150 tokens for the T3-side
    // `conds.t3.speech_prompt_tokens`.
    std::vector<int32_t> speech_prompt_tokens;
    std::vector<int32_t> prompt_tokens;
    if (ctx->s3gen_ctx) {
        const int n6 = std::min(n_16k, 6 * 16000);
        int n_tok6 = 0;
        int32_t* tk6 = chatterbox_s3gen_tokenize_pcm(ctx->s3gen_ctx, pcm_16k, n6, /*max_tokens*/ 150, &n_tok6);
        if (tk6 && n_tok6 > 0) {
            speech_prompt_tokens.assign(tk6, tk6 + n_tok6);
            std::free(tk6);
        }
        // gen.prompt_token: full audio, no max_len. ONLY install when we
        // also have the matching prompt_feat + embedding (atomic path).
        if (atomic_path) {
            int n_tok_full = 0;
            int32_t* tkf = chatterbox_s3gen_tokenize_pcm(ctx->s3gen_ctx, pcm_16k, n_16k, /*max_tokens*/ 0, &n_tok_full);
            if (tkf && n_tok_full > 0) {
                prompt_tokens.assign(tkf, tkf + n_tok_full);
                std::free(tkf);
            }
        }
    }

    // CAMPPlus + 24 kHz prompt mel — only on the atomic (24 kHz input) path.
    // Both go through the s3gen sub-context's public C ABI hooks (the
    // campplus weights live there). Returned buffers are malloc'd; copy
    // into stable std::vectors and free.
    std::vector<float> gen_emb;     // 192-d
    std::vector<float> prompt_feat; // (T_mel, 80)
    int T_prompt_feat = 0;
    if (atomic_path && ctx->s3gen_ctx) {
        float* xv = chatterbox_s3gen_dump_campplus_xvector(ctx->s3gen_ctx, pcm_16k, n_16k);
        if (!xv) {
            fprintf(stderr, "chatterbox: CAMPPlus xvector failed\n");
            return -1;
        }
        gen_emb.assign(xv, xv + 192);
        std::free(xv);

        constexpr int kDecCondLen = 10 * 24000;
        float* pf =
            chatterbox_s3gen_dump_prompt_feat_24k(ctx->s3gen_ctx, pcm_24k.data(), n_24k, kDecCondLen, &T_prompt_feat);
        if (!pf || T_prompt_feat <= 0) {
            if (pf)
                std::free(pf);
            fprintf(stderr, "chatterbox: 24 kHz prompt mel failed\n");
            return -1;
        }
        prompt_feat.assign(pf, pf + (size_t)T_prompt_feat * 80);
        std::free(pf);

        // Match `embed_ref`'s sanity clamp: gen.prompt_feat covers
        // 2 * gen.prompt_token frames; trim tokens if the mel is shorter.
        if (!prompt_tokens.empty() && (int)prompt_tokens.size() > T_prompt_feat / 2) {
            prompt_tokens.resize((size_t)(T_prompt_feat / 2));
        }
    }

    int rc = chatterbox_install_native_voice(
        ctx, emb, atomic_path ? prompt_tokens.data() : nullptr, atomic_path ? (int)prompt_tokens.size() : 0,
        speech_prompt_tokens.data(), (int)speech_prompt_tokens.size(), atomic_path ? prompt_feat.data() : nullptr,
        atomic_path ? T_prompt_feat : 0, atomic_path ? gen_emb.data() : nullptr);
    if (rc != 0)
        return rc;

    if (ctx->params.verbosity >= 1) {
        if (atomic_path) {
            fprintf(stderr,
                    "chatterbox: atomic native WAV clone (%s, %d samples @ 24 kHz, %d @ 16 kHz) — all 5 conds "
                    "(speaker_emb, %zu speech_prompt_tokens, %zu prompt_token, prompt_feat (T_mel=%d), "
                    "gen.embedding (192-d)) installed.\n",
                    wav_path, n_24k, n_16k, speech_prompt_tokens.size(), prompt_tokens.size(), T_prompt_feat);
        } else {
            fprintf(stderr,
                    "chatterbox: partial native WAV clone (%s, %d samples @ 16 kHz) — T3-side conds cloned "
                    "(speaker_emb, %zu speech_prompt_tokens). S3Gen prompt (gen.{prompt_token, prompt_feat, "
                    "embedding}) still defaults; re-encode at 24 kHz mono for full atomic cloning.\n",
                    wav_path, n_16k, speech_prompt_tokens.size());
        }
    }
    return 0;
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

// Runtime sampling-knob setters. The chatterbox AR loop reads
// `ctx->params.X` on every sample (see sample_token call site in
// chatterbox_synthesize_codes), so post-init mutation is safe.
// Each setter clamps to a sensible range and silently no-ops on
// nonsense input rather than throwing — the C ABI surface stays
// crash-free for thin wrappers like the Dart binding.
//
// Restored 2026-05-11 — the original bodies (commit 95e2fdf7) were
// accidentally dropped by the unrelated indextts ECAPA fix in
// commit ff7bcc50 while the header / crispasr_c_api.cpp still
// referenced them. Same content; lint sweep re-runs cleanly.
extern "C" void chatterbox_set_temperature(struct chatterbox_context* ctx, float temperature) {
    if (!ctx) {
        return;
    }
    // 0.0 = greedy (argmax). Allow up to 4.0; beyond that the softmax
    // is essentially uniform and you're just sampling noise.
    if (temperature < 0.0f) {
        temperature = 0.0f;
    }
    if (temperature > 4.0f) {
        temperature = 4.0f;
    }
    ctx->params.temperature = temperature;
}

extern "C" void chatterbox_set_top_p(struct chatterbox_context* ctx, float top_p) {
    if (!ctx) {
        return;
    }
    if (top_p < 0.0f) {
        top_p = 0.0f;
    }
    if (top_p > 1.0f) {
        top_p = 1.0f;
    }
    ctx->params.top_p = top_p;
}

extern "C" void chatterbox_set_min_p(struct chatterbox_context* ctx, float min_p) {
    if (!ctx) {
        return;
    }
    if (min_p < 0.0f) {
        min_p = 0.0f;
    }
    if (min_p > 1.0f) {
        min_p = 1.0f;
    }
    ctx->params.min_p = min_p;
}

extern "C" void chatterbox_set_top_k(struct chatterbox_context* ctx, int top_k) {
    if (!ctx) {
        return;
    }
    if (top_k < 0) {
        top_k = 0;
    }
    ctx->params.top_k = top_k;
}

extern "C" void chatterbox_set_repetition_penalty(struct chatterbox_context* ctx, float r) {
    if (!ctx) {
        return;
    }
    // 1.0 = no penalty (Pytorch default). Below 1.0 *encourages* repeats.
    if (r < 0.5f) {
        r = 0.5f;
    }
    if (r > 2.0f) {
        r = 2.0f;
    }
    ctx->params.repetition_penalty = r;
}

extern "C" void chatterbox_set_max_speech_tokens(struct chatterbox_context* ctx, int n) {
    if (!ctx) {
        return;
    }
    // Default is 1000; allow up to 4000 (= ~80 s of speech tokens at
    // 50 Hz). Anything below 32 is unusable.
    if (n < 32) {
        n = 32;
    }
    if (n > 4000) {
        n = 4000;
    }
    ctx->params.max_speech_tokens = n;
}

extern "C" void chatterbox_set_seed(struct chatterbox_context* ctx, uint32_t seed) {
    if (!ctx)
        return;
    ctx->rng_seed = seed;
    mt19937_seed(ctx->rng_state, seed);
    if (ctx->s3gen_ctx)
        chatterbox_s3gen_set_seed(ctx->s3gen_ctx, seed);
}

extern "C" void chatterbox_set_language(struct chatterbox_context* ctx, const char* lang) {
    if (!ctx)
        return;
    if (!lang || !*lang) {
        ctx->language.clear();
        return;
    }
    // Validate: check the tokenizer vocab for [lang] token
    std::string tag = std::string("[") + lang + "]";
    if (ctx->tokenizer.token_to_id.find(tag) == ctx->tokenizer.token_to_id.end()) {
        fprintf(stderr,
                "chatterbox: warning: language token '%s' not in vocab — "
                "generation may not work as expected\n",
                tag.c_str());
    }
    ctx->language = tag;
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

// (chatterbox_set_temperature / set_top_p / set_min_p /
//  set_repetition_penalty / set_max_speech_tokens are defined above
//  at lines ~3258-3313 with proper bound-checking — `r < 0.5f` clamp
//  on repetition_penalty, `n > 0 ? n : 1000` fallback on
//  max_speech_tokens, etc. The simpler one-liner copies that lived
//  here briefly in 934313cc were duplicate definitions causing GCC
//  redefinition errors on every Linux/macOS/Windows/iOS/Android CI
//  job — removed.)

// Diff/debug: VE pipeline stages — see chatterbox_ve.h for the spec.
//
// These are deliberately public C-ABI: `examples/cli/crispasr_diff_main.cpp`
// calls them on the same 16 kHz mono float32 buffer the reference dumper
// fed to `model.ve.embeds_from_wavs`. Each returns a malloc'd buffer the
// caller frees with plain `free()` (mirrors `chatterbox_dump_t3_prefill_emb`).
extern "C" float* chatterbox_dump_ve_mel(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                         int* out_T) {
    if (!ctx || !pcm_16k || n_samples <= 0 || !out_T)
        return nullptr;
    int T = 0;
    auto mel = chatterbox_ve::compute_mel(pcm_16k, n_samples, T);
    if (mel.empty() || T <= 0) {
        *out_T = 0;
        return nullptr;
    }
    float* r = (float*)std::malloc(mel.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, mel.data(), mel.size() * sizeof(float));
    *out_T = T;
    return r;
}

extern "C" float* chatterbox_dump_ve_partial_emb(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                                 int* out_n_partials) {
    if (!ctx || !pcm_16k || n_samples <= 0 || !out_n_partials)
        return nullptr;
    int T = 0;
    auto mel = chatterbox_ve::compute_mel(pcm_16k, n_samples, T);
    if (mel.empty() || T <= 0) {
        *out_n_partials = 0;
        return nullptr;
    }
    int n_partials = 0;
    auto embs =
        chatterbox_ve::compute_partial_embeds(ctx->ve, ctx->sched, ctx->compute_meta, mel.data(), T, n_partials);
    if (embs.empty() || n_partials <= 0) {
        *out_n_partials = 0;
        return nullptr;
    }
    float* r = (float*)std::malloc(embs.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, embs.data(), embs.size() * sizeof(float));
    *out_n_partials = n_partials;
    return r;
}

extern "C" float* chatterbox_dump_ve_speaker_emb(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples) {
    if (!ctx || !pcm_16k || n_samples <= 0)
        return nullptr;
    float* r = (float*)std::malloc(256 * sizeof(float));
    if (!r)
        return nullptr;
    if (!chatterbox_ve::compute_speaker_emb(ctx->ve, ctx->sched, ctx->compute_meta, pcm_16k, n_samples, r)) {
        std::free(r);
        return nullptr;
    }
    return r;
}

// S3Tokenizer V2 (module 3) — forwarders to the s3gen sub-context.
extern "C" float* chatterbox_dump_s3tok_log_mel(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                                int* out_T) {
    if (!ctx || !ctx->s3gen_ctx)
        return nullptr;
    return chatterbox_s3gen_dump_s3tok_log_mel(ctx->s3gen_ctx, pcm_16k, n_samples, out_T);
}

extern "C" float* chatterbox_dump_s3tok_proj_down(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                                  int max_tokens, int* out_T_tok) {
    if (!ctx || !ctx->s3gen_ctx)
        return nullptr;
    return chatterbox_s3gen_dump_s3tok_proj_down(ctx->s3gen_ctx, pcm_16k, n_samples, max_tokens, out_T_tok);
}

extern "C" float* chatterbox_dump_s3tok_tokens(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                               int max_tokens, int* out_T_tok) {
    if (!ctx || !ctx->s3gen_ctx)
        return nullptr;
    return chatterbox_s3gen_dump_s3tok_tokens(ctx->s3gen_ctx, pcm_16k, n_samples, max_tokens, out_T_tok);
}

extern "C" float* chatterbox_dump_campplus_fbank(struct chatterbox_context* ctx, const float* pcm_16k, int n_samples,
                                                 int* out_T) {
    if (!ctx || !ctx->s3gen_ctx)
        return nullptr;
    return chatterbox_s3gen_dump_campplus_fbank(ctx->s3gen_ctx, pcm_16k, n_samples, out_T);
}

extern "C" float* chatterbox_dump_campplus_xvector(struct chatterbox_context* ctx, const float* pcm_16k,
                                                   int n_samples) {
    if (!ctx || !ctx->s3gen_ctx)
        return nullptr;
    return chatterbox_s3gen_dump_campplus_xvector(ctx->s3gen_ctx, pcm_16k, n_samples);
}

extern "C" float* chatterbox_dump_prompt_feat_24k(struct chatterbox_context* ctx, const float* pcm_24k, int n_samples,
                                                  int max_samples, int* out_T_mel) {
    if (!ctx || !ctx->s3gen_ctx)
        return nullptr;
    return chatterbox_s3gen_dump_prompt_feat_24k(ctx->s3gen_ctx, pcm_24k, n_samples, max_samples, out_T_mel);
}

extern "C" float* chatterbox_dump_s3gen_encoder_out(struct chatterbox_context* ctx, const int32_t* speech_tokens,
                                                    int n_speech_tokens, int* out_T_mel) {
    if (out_T_mel)
        *out_T_mel = 0;
    if (!ctx || !ctx->s3gen_ctx || !speech_tokens || n_speech_tokens <= 0)
        return nullptr;
    std::vector<int32_t> pt_buf;
    const int32_t* prompt_tokens = nullptr;
    int n_prompt = 0;
    if (ctx->conds.gen_prompt_token) {
        n_prompt = (int)ctx->conds.gen_prompt_token->ne[0];
        pt_buf.resize(n_prompt);
        ggml_backend_tensor_get(ctx->conds.gen_prompt_token, pt_buf.data(), 0, n_prompt * sizeof(int32_t));
        prompt_tokens = pt_buf.data();
    }
    return chatterbox_s3gen_dump_encoder_out(ctx->s3gen_ctx, speech_tokens, n_speech_tokens, prompt_tokens, n_prompt,
                                             out_T_mel);
}

extern "C" float* chatterbox_dump_t3_prefill_emb(struct chatterbox_context* ctx, const char* text, int* out_T,
                                                 int* out_D, int* out_cond_T) {
    if (!ctx || !text || !out_T || !out_D || !out_cond_T)
        return nullptr;
    if (ctx->hp.arch == "chatterbox_turbo" || ctx->hp.arch == "kartoffelbox")
        return nullptr; // GPT-2 path not supported here
    if (!ctx->conds.loaded)
        return nullptr;

    std::string norm_text = chatterbox_text_prep::normalize(text, !ctx->language.empty());
    std::vector<int32_t> text_tokens;
    if (ctx->tokenizer.has_bpe) {
        text_tokens = ctx->tokenizer.bpe_byte_level ? tokenize_text_bpe(ctx->tokenizer, norm_text)
                                                    : tokenize_text_hf_bpe(ctx->tokenizer, norm_text);
    } else {
        text_tokens = tokenize_text(ctx->tokenizer, norm_text);
    }
    prepend_language_token(ctx, text_tokens);
    text_tokens.insert(text_tokens.begin(), (int32_t)ctx->hp.start_text_token);
    text_tokens.push_back((int32_t)ctx->hp.stop_text_token);

    std::vector<float> emb = build_prefill_embeds(ctx, text_tokens);
    if (emb.empty())
        return nullptr;

    const int D = (int)ctx->hp.hidden_size;
    const int T = (int)(emb.size() / D);
    // cond_len = T - text_len - 1 (speech_start)
    *out_T = T;
    *out_D = D;
    *out_cond_T = T - (int)text_tokens.size() - 1;

    float* r = (float*)malloc(emb.size() * sizeof(float));
    if (r)
        std::memcpy(r, emb.data(), emb.size() * sizeof(float));
    return r;
}

extern "C" int chatterbox_dump_t3_next_logits(struct chatterbox_context* ctx, const char* text,
                                              const int32_t* prefix_tokens, int n_prefix, float** out_logits_cond,
                                              float** out_logits_uncond, float** out_logits_blended, int* out_V) {
    if (!ctx || !text || !out_V) {
        return -1;
    }
    if (out_logits_cond) {
        *out_logits_cond = nullptr;
    }
    if (out_logits_uncond) {
        *out_logits_uncond = nullptr;
    }
    if (out_logits_blended) {
        *out_logits_blended = nullptr;
    }

    if (ctx->hp.arch == "chatterbox_turbo" || ctx->hp.arch == "kartoffelbox") {
        return -2;
    }
    if (!ctx->conds.loaded) {
        return -3;
    }

    std::string norm_text = chatterbox_text_prep::normalize(text, !ctx->language.empty());
    std::vector<int32_t> text_tokens;
    if (ctx->tokenizer.has_bpe) {
        text_tokens = ctx->tokenizer.bpe_byte_level ? tokenize_text_bpe(ctx->tokenizer, norm_text)
                                                    : tokenize_text_hf_bpe(ctx->tokenizer, norm_text);
    } else {
        text_tokens = tokenize_text(ctx->tokenizer, norm_text);
    }
    prepend_language_token(ctx, text_tokens);
    text_tokens.insert(text_tokens.begin(), (int32_t)ctx->hp.start_text_token);
    text_tokens.push_back((int32_t)ctx->hp.stop_text_token);

    std::vector<float> prefill_embeds = build_prefill_embeds(ctx, text_tokens);
    if (prefill_embeds.empty()) {
        return -4;
    }

    const int D = (int)ctx->hp.hidden_size;
    std::vector<float> bos_extra(prefill_embeds.end() - D, prefill_embeds.end());
    prefill_embeds.insert(prefill_embeds.end(), bos_extra.begin(), bos_extra.end());
    const int prefill_len = (int)(prefill_embeds.size() / D);

    const int max_speech = ctx->params.max_speech_tokens > 0 ? ctx->params.max_speech_tokens : 1000;
    const int max_ctx = (int)text_tokens.size() + max_speech + 64;
    if (!kv_alloc(ctx, max_ctx)) {
        return -7;
    }

    const float cfg_w = ctx->params.cfg_weight;
    const bool use_cfg = (cfg_w > 0.0f && ctx->kv_k_cfg);
    std::vector<float> uncond_embeds;
    if (use_cfg) {
        uncond_embeds = prefill_embeds;
        const int text_start = prefill_len - (int)text_tokens.size() - 2;
        const int text_end = prefill_len - 2;
        const auto& text_pos_table = ctx->text_pos_emb_cache;
        for (int i = text_start; i < text_end; ++i) {
            const int pos_idx = i - text_start;
            std::memcpy(&uncond_embeds[(size_t)i * D], &text_pos_table[(size_t)pos_idx * D], (size_t)D * sizeof(float));
        }
    }

    int n_past = 0;
    float* logits_cond = run_t3_kv(ctx, prefill_embeds.data(), prefill_len, n_past);
    if (!logits_cond) {
        return -5;
    }

    float* logits_uncond = nullptr;
    int n_past_cfg = 0;
    if (use_cfg) {
        logits_uncond = run_t3_kv(ctx, uncond_embeds.data(), prefill_len, n_past_cfg, ctx->kv_k_cfg, ctx->kv_v_cfg);
        if (!logits_uncond) {
            free(logits_cond);
            return -6;
        }
    }

    n_past += prefill_len;
    n_past_cfg += prefill_len;

    for (int i = 0; i < n_prefix; ++i) {
        const int32_t tok = prefix_tokens ? prefix_tokens[i] : 0;
        std::vector<float> tok_embed = build_speech_token_embed(ctx, tok, i + 1);

        float* next_cond = run_t3_kv(ctx, tok_embed.data(), 1, n_past);
        if (!next_cond) {
            free(logits_cond);
            free(logits_uncond);
            return -8;
        }
        free(logits_cond);
        logits_cond = next_cond;
        n_past++;

        if (use_cfg) {
            float* next_uncond = run_t3_kv(ctx, tok_embed.data(), 1, n_past_cfg, ctx->kv_k_cfg, ctx->kv_v_cfg);
            if (!next_uncond) {
                free(logits_cond);
                free(logits_uncond);
                return -9;
            }
            free(logits_uncond);
            logits_uncond = next_uncond;
            n_past_cfg++;
        }
    }

    const int V = (int)ctx->hp.speech_vocab_size;
    *out_V = V;

    auto dup_logits = [&](const float* src) -> float* {
        if (!src) {
            return nullptr;
        }
        float* dst = (float*)malloc((size_t)V * sizeof(float));
        if (dst) {
            std::memcpy(dst, src, (size_t)V * sizeof(float));
        }
        return dst;
    };

    if (out_logits_cond) {
        *out_logits_cond = dup_logits(logits_cond);
    }
    if (out_logits_uncond) {
        *out_logits_uncond = dup_logits(logits_uncond);
    }
    if (out_logits_blended) {
        float* blended = (float*)malloc((size_t)V * sizeof(float));
        if (blended) {
            for (int i = 0; i < V; ++i) {
                blended[i] = (use_cfg && logits_uncond) ? (logits_cond[i] + cfg_w * (logits_cond[i] - logits_uncond[i]))
                                                        : logits_cond[i];
            }
        }
        *out_logits_blended = blended;
    }

    free(logits_cond);
    free(logits_uncond);
    return 0;
}

extern "C" int chatterbox_dump_t3_step0_logits(struct chatterbox_context* ctx, const char* text,
                                               float** out_logits_cond, float** out_logits_uncond,
                                               float** out_logits_blended, int* out_V) {
    return chatterbox_dump_t3_next_logits(ctx, text, nullptr, 0, out_logits_cond, out_logits_uncond, out_logits_blended,
                                          out_V);
}
