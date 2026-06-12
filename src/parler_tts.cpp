// parler_tts.cpp -- Parler TTS runtime (T5 encoder + MusicGen decoder + DAC).
//
// Architecture (parler-tts-mini-v1.1):
//
//   T5 ENCODER (flan-t5-large, encoder-only):
//     24 layers, d_model=1024, d_kv=64, n_heads=16, d_ff=2816
//     Gated-GELU FFN (gate + up -> down), RMS norm, relative position bias
//     Encodes voice description -> (T_desc, 1024) hidden states
//
//   DECODER (MusicGen-style causal transformer):
//     24 layers, hidden=1024, n_heads=16, ffn_dim=4096
//     Self-attention + cross-attention on T5 encoder output
//     LayerNorm with bias (not RMS), sinusoidal positional embeddings
//     GELU activation in FFN
//     9 codebooks, vocab_size=1088 (1024 audio + pad(1024) + bos(1025) + extras)
//     Delay pattern: codebook k is delayed by k steps
//
//   DAC 44 kHz CODEC:
//     9 codebooks x 1024 entries, upsampling 512x -> 44.1 kHz
//     Reuses src/core/dac_decoder.h
//
// Flow:
//   1. T5 encode description -> enc_hidden (cached, run once per voice)
//   2. Tokenize text prompt -> prompt_ids
//   3. Embed prompt_ids via embed_prompts + positional embedding
//   4. AR decode: for each step, cross-attend to enc_hidden, self-attend
//      to past, produce 9 codebook logits, sample, apply delay pattern
//   5. Un-delay the codebook tokens
//   6. DAC decode tokens -> 44.1 kHz PCM

#include "parler_tts.h"
#include "core/dac_decoder.h"
#include "core/gguf_loader.h"
#include "core/sentencepiece.h"

#include "ggml-alloc.h"
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
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// ── Configuration ───────────────────────────────────────────────────

struct parler_t5_config {
    int d_model = 1024;
    int d_kv = 64; // per-head dim
    int d_ff = 2816;
    int n_heads = 16;
    int n_layers = 24;
    int vocab_size = 32128;
    int rel_attn_num_buckets = 32;
    int rel_attn_max_dist = 128;
    float layer_norm_eps = 1e-6f;
    bool is_gated_gelu = true;
};

struct parler_decoder_config {
    int hidden_size = 1024;
    int num_layers = 24;
    int num_heads = 16;
    int num_kv_heads = 16;
    int num_cross_kv_heads = 16;
    int ffn_dim = 4096;
    int vocab_size = 1088;
    int num_codebooks = 9;
    int max_position_embeddings = 4096;
    int bos_token_id = 1025;
    int eos_token_id = 1024;
    int pad_token_id = 1024;
    int max_generation = 2580;
    bool use_fused_lm_heads = true;
    bool rope_embeddings = false;
    float layer_norm_eps = 1e-5f;
};

struct parler_dac_config {
    int n_codebooks = 9;
    int codebook_size = 1024;
    int codebook_dim = 8;
    int hidden_size = 1024;
    int sample_rate = 44100;
    int hop_length = 512;
};

// ── T5 encoder layer tensors ────────────────────────────────────────

struct parler_t5_layer {
    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_o = nullptr;
    ggml_tensor* attn_rms = nullptr; // RMS norm weight
    ggml_tensor* ffn_gate = nullptr; // gated-gelu: wi_0
    ggml_tensor* ffn_up = nullptr;   // gated-gelu: wi_1
    ggml_tensor* ffn_down = nullptr; // gated-gelu: wo
    ggml_tensor* ffn_rms = nullptr;  // RMS norm weight
};

// ── Decoder layer tensors ───────────────────────────────────────────

struct parler_dec_layer {
    // Self-attention
    ggml_tensor* self_attn_q = nullptr;
    ggml_tensor* self_attn_k = nullptr;
    ggml_tensor* self_attn_v = nullptr;
    ggml_tensor* self_attn_o = nullptr;
    ggml_tensor* self_attn_norm_w = nullptr;
    ggml_tensor* self_attn_norm_b = nullptr;

    // Cross-attention
    ggml_tensor* cross_attn_q = nullptr;
    ggml_tensor* cross_attn_k = nullptr;
    ggml_tensor* cross_attn_v = nullptr;
    ggml_tensor* cross_attn_o = nullptr;
    ggml_tensor* cross_attn_norm_w = nullptr;
    ggml_tensor* cross_attn_norm_b = nullptr;

    // Cached cross-attention KV (computed once from T5 encoder output)
    ggml_tensor* cross_k_cache = nullptr;
    ggml_tensor* cross_v_cache = nullptr;

    // FFN
    ggml_tensor* fc1 = nullptr;
    ggml_tensor* fc2 = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
};

// ── Model ───────────────────────────────────────────────────────────

struct parler_model {
    parler_t5_config t5_cfg;
    parler_decoder_config dec_cfg;
    parler_dac_config dac_cfg;

    // T5 encoder
    ggml_tensor* t5_embed = nullptr;
    ggml_tensor* t5_rel_bias = nullptr;
    ggml_tensor* t5_final_rms = nullptr;
    std::vector<parler_t5_layer> t5_layers;

    // Decoder
    std::vector<ggml_tensor*> dec_embeds; // [num_codebooks] per-codebook embeddings
    ggml_tensor* dec_embed_prompts = nullptr;
    ggml_tensor* dec_pos_embed = nullptr; // sinusoidal positional embedding table
    ggml_tensor* dec_final_norm_w = nullptr;
    ggml_tensor* dec_final_norm_b = nullptr;
    std::vector<parler_dec_layer> dec_layers;
    std::vector<ggml_tensor*> lm_heads; // [num_codebooks] (hidden -> vocab_size)

    // DAC
    core_dac::DacWeights dac;
};

// ── Tokenizer ───────────────────────────────────────────────────────

struct parler_tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    std::unordered_map<std::string, int32_t> spm_vocab; // for core_spm::tokenize
    std::vector<float> scores;
    int bos_id = 1;
    int eos_id = 2; // LLaMA default
    int unk_id = 0;
};

// ── Context ─────────────────────────────────────────────────────────

struct parler_tts_context {
    parler_tts_context_params params;
    parler_model model;

    // Prompt tokenizer (sentencepiece: BPE or unigram)
    parler_tokenizer prompt_tokenizer;
    bool tokenizer_is_bpe = false; // true = use BPE merge, false = Viterbi unigram

    // Description tokenizer (T5 sentencepiece)
    parler_tokenizer desc_tokenizer;

    // GGML state
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // T5 encoder output (cached after set_description)
    std::vector<float> enc_hidden; // (T_desc, d_model) flat
    int enc_T = 0;                 // number of tokens in cached encoding
    bool enc_cached = false;

    // Cached cross-attention KV per decoder layer
    // These are computed once from enc_hidden and stored in model.dec_layers[i].cross_k/v_cache

    // Decoder self-attention KV cache
    // Flat buffers: (n_layers, max_seq, hidden_size) for k and v
    std::vector<float> kv_k;
    std::vector<float> kv_v;

    // RNG
    std::mt19937 rng;
};

// ── T5 relative position bias ───────────────────────────────────────

static int t5_relative_position_bucket(int rel_pos, bool bidirectional, int num_buckets, int max_dist) {
    int ret = 0;
    if (bidirectional) {
        num_buckets /= 2;
        if (rel_pos > 0)
            ret += num_buckets;
        rel_pos = std::abs(rel_pos);
    } else {
        rel_pos = -std::min(rel_pos, 0);
    }
    const int max_exact = num_buckets / 2;
    if (rel_pos < max_exact) {
        ret += rel_pos;
    } else {
        int val = (int)(max_exact + std::log((float)rel_pos / max_exact) / std::log((float)max_dist / max_exact) *
                                        (num_buckets - max_exact));
        val = std::min(val, num_buckets - 1);
        ret += val;
    }
    return ret;
}

// ── SentencePiece unigram tokenizer ─────────────────────────────────

static std::vector<int> tokenize_unigram(const parler_tokenizer& tok, const std::string& text) {
    // Viterbi-based unigram tokenizer with scores
    if (tok.id_to_token.empty())
        return {};

    // Prepend space for sentencepiece convention
    std::string input = " " + text;
    const int n = (int)input.size();

    // Build token -> id lookup for fast matching
    // Simple forward longest-match with fallback to unknown
    std::vector<int> ids;

    if (!tok.scores.empty()) {
        // Viterbi segmentation
        const float NEG_INF = -1e30f;
        std::vector<float> best_score(n + 1, NEG_INF);
        std::vector<int> best_len(n + 1, 0);
        best_score[0] = 0.0f;

        for (int i = 0; i < n; i++) {
            if (best_score[i] == NEG_INF)
                continue;
            // Try all possible token lengths starting at position i
            for (const auto& [token, id] : tok.token_to_id) {
                int tlen = (int)token.size();
                if (i + tlen > n)
                    continue;
                if (input.compare(i, tlen, token) == 0) {
                    float score = best_score[i] + (id < (int)tok.scores.size() ? tok.scores[id] : 0.0f);
                    if (score > best_score[i + tlen]) {
                        best_score[i + tlen] = score;
                        best_len[i + tlen] = tlen;
                    }
                }
            }
            // Fallback: single byte as unknown
            if (best_score[i + 1] == NEG_INF) {
                best_score[i + 1] = best_score[i] - 100.0f;
                best_len[i + 1] = 1;
            }
        }

        // Backtrack
        std::vector<int> lengths;
        int pos = n;
        while (pos > 0) {
            lengths.push_back(best_len[pos]);
            pos -= best_len[pos];
        }
        std::reverse(lengths.begin(), lengths.end());

        pos = 0;
        for (int len : lengths) {
            std::string piece = input.substr(pos, len);
            auto it = tok.token_to_id.find(piece);
            if (it != tok.token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(tok.unk_id);
            }
            pos += len;
        }
    } else {
        // Greedy longest-match fallback
        int pos = 0;
        while (pos < n) {
            int best_len_found = 0;
            int best_id = tok.unk_id;
            for (const auto& [token, id] : tok.token_to_id) {
                int tlen = (int)token.size();
                if (tlen > best_len_found && pos + tlen <= n && input.compare(pos, tlen, token) == 0) {
                    best_len_found = tlen;
                    best_id = id;
                }
            }
            if (best_len_found == 0) {
                ids.push_back(tok.unk_id);
                pos++;
            } else {
                ids.push_back(best_id);
                pos += best_len_found;
            }
        }
    }

    // Append EOS
    ids.push_back(tok.eos_id);
    return ids;
}

// ── Load from GGUF ──────────────────────────────────────────────────

static void load_metadata(parler_tts_context* c, gguf_context* g) {
    auto& m = c->model;
    auto get_u32 = [&](const char* key, int def) -> int {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? (int)gguf_get_val_u32(g, idx) : def;
    };
    auto get_str = [&](const char* key, const char* def) -> std::string {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? gguf_get_val_str(g, idx) : def;
    };
    auto get_bool = [&](const char* key, bool def) -> bool {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? gguf_get_val_bool(g, idx) : def;
    };

    // T5 encoder config
    m.t5_cfg.d_model = get_u32("parler.t5enc.d_model", 1024);
    m.t5_cfg.d_kv = get_u32("parler.t5enc.d_kv", 64);
    m.t5_cfg.d_ff = get_u32("parler.t5enc.d_ff", 2816);
    m.t5_cfg.n_heads = get_u32("parler.t5enc.n_heads", 16);
    m.t5_cfg.n_layers = get_u32("parler.t5enc.n_layers", 24);
    m.t5_cfg.vocab_size = get_u32("parler.t5enc.vocab_size", 32128);
    m.t5_cfg.rel_attn_num_buckets = get_u32("parler.t5enc.rel_attn_num_buckets", 32);
    m.t5_cfg.rel_attn_max_dist = get_u32("parler.t5enc.rel_attn_max_distance", 128);
    std::string ff_proj = get_str("parler.t5enc.feed_forward_proj", "gated-gelu");
    m.t5_cfg.is_gated_gelu = (ff_proj.find("gated") != std::string::npos);

    // Decoder config
    m.dec_cfg.hidden_size = get_u32("parler.decoder.hidden_size", 1024);
    m.dec_cfg.num_layers = get_u32("parler.decoder.num_layers", 24);
    m.dec_cfg.num_heads = get_u32("parler.decoder.num_heads", 16);
    m.dec_cfg.num_kv_heads = get_u32("parler.decoder.num_kv_heads", 16);
    m.dec_cfg.num_cross_kv_heads = get_u32("parler.decoder.num_cross_kv_heads", 16);
    m.dec_cfg.ffn_dim = get_u32("parler.decoder.ffn_dim", 4096);
    m.dec_cfg.vocab_size = get_u32("parler.decoder.vocab_size", 1088);
    m.dec_cfg.num_codebooks = get_u32("parler.decoder.num_codebooks", 9);
    m.dec_cfg.max_position_embeddings = get_u32("parler.decoder.max_position_embeddings", 4096);
    m.dec_cfg.bos_token_id = get_u32("parler.decoder.bos_token_id", 1025);
    m.dec_cfg.eos_token_id = get_u32("parler.decoder.eos_token_id", 1024);
    m.dec_cfg.pad_token_id = get_u32("parler.decoder.pad_token_id", 1024);
    m.dec_cfg.max_generation = get_u32("parler.decoder.max_generation", 2580);
    m.dec_cfg.use_fused_lm_heads = get_bool("parler.decoder.use_fused_lm_heads", true);
    m.dec_cfg.rope_embeddings = get_bool("parler.decoder.rope_embeddings", false);

    // DAC config
    m.dac_cfg.n_codebooks = get_u32("parler.dac.n_codebooks", 9);
    m.dac_cfg.codebook_size = get_u32("parler.dac.codebook_size", 1024);
    m.dac_cfg.codebook_dim = get_u32("parler.dac.codebook_dim", 8);
    m.dac_cfg.hidden_size = get_u32("parler.dac.hidden_size", 1024);
    m.dac_cfg.sample_rate = get_u32("parler.dac.sample_rate", 44100);
    m.dac_cfg.hop_length = get_u32("parler.dac.hop_length", 512);

    // Load prompt tokenizer (also used for description — same LLaMA tokenizer)
    {
        int tidx = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tidx >= 0) {
            int n = gguf_get_arr_n(g, tidx);
            c->prompt_tokenizer.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                c->prompt_tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
                c->prompt_tokenizer.token_to_id[c->prompt_tokenizer.id_to_token[i]] = i;
                c->prompt_tokenizer.spm_vocab[c->prompt_tokenizer.id_to_token[i]] = i;
            }
        }
        int sidx = gguf_find_key(g, "tokenizer.ggml.scores");
        if (sidx >= 0) {
            int n = gguf_get_arr_n(g, sidx);
            c->prompt_tokenizer.scores.resize(n);
            const float* sp = (const float*)gguf_get_arr_data(g, sidx);
            for (int i = 0; i < n; i++)
                c->prompt_tokenizer.scores[i] = sp[i];
        }
        // LLaMA sentencepiece defaults
        c->prompt_tokenizer.bos_id = 1;
        c->prompt_tokenizer.eos_id = 2;
        c->prompt_tokenizer.unk_id = 0;

        // Check if the tokenizer is BPE (sentencepiece model_type=2)
        c->tokenizer_is_bpe = get_bool("parler.tokenizer.is_bpe", false);
    }

    // Load description tokenizer (T5)
    {
        int tidx = gguf_find_key(g, "parler.desc_tokenizer.tokens");
        if (tidx >= 0) {
            int n = gguf_get_arr_n(g, tidx);
            c->desc_tokenizer.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                c->desc_tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
                c->desc_tokenizer.token_to_id[c->desc_tokenizer.id_to_token[i]] = i;
            }
        }
        int sidx = gguf_find_key(g, "parler.desc_tokenizer.scores");
        if (sidx >= 0) {
            int n = gguf_get_arr_n(g, sidx);
            c->desc_tokenizer.scores.resize(n);
            const float* sp = (const float*)gguf_get_arr_data(g, sidx);
            for (int i = 0; i < n; i++)
                c->desc_tokenizer.scores[i] = sp[i];
        }
        // T5 EOS is typically 1 for flan-t5
        c->desc_tokenizer.eos_id = 1;
        c->desc_tokenizer.unk_id = 2;
    }
}

static bool bind_tensors(parler_tts_context* c) {
    auto& m = c->model;
    auto T = [&](const char* name) -> ggml_tensor* {
        auto it = c->tensors.find(name);
        return (it != c->tensors.end()) ? it->second : nullptr;
    };

    // T5 encoder
    m.t5_embed = T("t5enc.embed.weight");
    m.t5_rel_bias = T("t5enc.rel_bias.weight");
    m.t5_final_rms = T("t5enc.final_rms.weight");

    m.t5_layers.resize(m.t5_cfg.n_layers);
    for (int i = 0; i < m.t5_cfg.n_layers; i++) {
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "t5enc.blk.%d.%s", i, suffix);
            return T(buf);
        };
        auto& l = m.t5_layers[i];
        l.attn_q = w("attn_q.weight");
        l.attn_k = w("attn_k.weight");
        l.attn_v = w("attn_v.weight");
        l.attn_o = w("attn_o.weight");
        l.attn_rms = w("attn_rms.weight");
        l.ffn_gate = w("ffn_gate.weight");
        l.ffn_up = w("ffn_up.weight");
        l.ffn_down = w("ffn_down.weight");
        l.ffn_rms = w("ffn_rms.weight");
    }

    // Decoder
    m.dec_embeds.resize(m.dec_cfg.num_codebooks);
    m.lm_heads.resize(m.dec_cfg.num_codebooks);
    for (int i = 0; i < m.dec_cfg.num_codebooks; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dec.embed.%d.weight", i);
        m.dec_embeds[i] = T(buf);
        snprintf(buf, sizeof(buf), "dec.lm_head.%d.weight", i);
        m.lm_heads[i] = T(buf);
    }
    m.dec_embed_prompts = T("dec.embed_prompts.weight");
    m.dec_pos_embed = T("dec.pos_embed.weight");
    m.dec_final_norm_w = T("dec.final_norm.weight");
    m.dec_final_norm_b = T("dec.final_norm.bias");

    m.dec_layers.resize(m.dec_cfg.num_layers);
    for (int i = 0; i < m.dec_cfg.num_layers; i++) {
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "dec.blk.%d.%s", i, suffix);
            return T(buf);
        };
        auto& l = m.dec_layers[i];
        l.self_attn_q = w("self_attn_q.weight");
        l.self_attn_k = w("self_attn_k.weight");
        l.self_attn_v = w("self_attn_v.weight");
        l.self_attn_o = w("self_attn_o.weight");
        l.self_attn_norm_w = w("self_attn_norm.weight");
        l.self_attn_norm_b = w("self_attn_norm.bias");
        l.cross_attn_q = w("cross_attn_q.weight");
        l.cross_attn_k = w("cross_attn_k.weight");
        l.cross_attn_v = w("cross_attn_v.weight");
        l.cross_attn_o = w("cross_attn_o.weight");
        l.cross_attn_norm_w = w("cross_attn_norm.weight");
        l.cross_attn_norm_b = w("cross_attn_norm.bias");
        l.fc1 = w("fc1.weight");
        l.fc2 = w("fc2.weight");
        l.ffn_norm_w = w("ffn_norm.weight");
        l.ffn_norm_b = w("ffn_norm.bias");
    }

    // DAC weights
    m.dac.quantizers.resize(m.dac_cfg.n_codebooks);
    for (int k = 0; k < m.dac_cfg.n_codebooks; k++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dac.quant.%d.weight", k);
        m.dac.quantizers[k].codebook = T(buf);
        snprintf(buf, sizeof(buf), "dac.quant_proj.%d.weight", k);
        m.dac.quantizers[k].out_proj_w = T(buf);
        snprintf(buf, sizeof(buf), "dac.quant_proj.%d.bias", k);
        m.dac.quantizers[k].out_proj_b = T(buf);
    }
    m.dac.in_conv_w = T("dac.dec.in_conv.weight");
    m.dac.in_conv_b = T("dac.dec.in_conv.bias");
    m.dac.out_snake_alpha = T("dac.dec.out_snake.alpha");
    m.dac.out_conv_w = T("dac.dec.out_conv.weight");
    m.dac.out_conv_b = T("dac.dec.out_conv.bias");
    for (int b = 0; b < 4; b++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dac.dec.blk.%d.0.alpha", b);
        m.dac.blocks[b].snake_alpha = T(buf);
        snprintf(buf, sizeof(buf), "dac.dec.blk.%d.1.weight", b);
        m.dac.blocks[b].up_w = T(buf);
        snprintf(buf, sizeof(buf), "dac.dec.blk.%d.1.bias", b);
        m.dac.blocks[b].up_b = T(buf);
        for (int r = 0; r < 3; r++) {
            auto& ru = m.dac.blocks[b].res[r];
            snprintf(buf, sizeof(buf), "dac.dec.blk.%d.%d.alpha0", b, r + 2);
            ru.alpha0 = T(buf);
            snprintf(buf, sizeof(buf), "dac.dec.blk.%d.%d.conv0.weight", b, r + 2);
            ru.conv0_w = T(buf);
            snprintf(buf, sizeof(buf), "dac.dec.blk.%d.%d.conv0.bias", b, r + 2);
            ru.conv0_b = T(buf);
            snprintf(buf, sizeof(buf), "dac.dec.blk.%d.%d.alpha1", b, r + 2);
            ru.alpha1 = T(buf);
            snprintf(buf, sizeof(buf), "dac.dec.blk.%d.%d.conv1.weight", b, r + 2);
            ru.conv1_w = T(buf);
            snprintf(buf, sizeof(buf), "dac.dec.blk.%d.%d.conv1.bias", b, r + 2);
            ru.conv1_b = T(buf);
        }
    }

    return true;
}

// ── Delay pattern ───────────────────────────────────────────────────
//
// MusicGen delay pattern for K codebooks:
//   Codebook k is delayed by k steps.
//   At generation step t, we predict codebook k's token for time (t - k).
//
// Example with 4 codebooks, T=5 steps:
//   Step 0: [BOS,  BOS,  BOS,  BOS ]
//   Step 1: [c0_0, BOS,  BOS,  BOS ]
//   Step 2: [c0_1, c1_0, BOS,  BOS ]
//   Step 3: [c0_2, c1_1, c2_0, BOS ]
//   Step 4: [c0_3, c1_2, c2_1, c3_0]
//   ...
//
// After generation, un-delay to get aligned codebooks:
//   Time 0: [c0_0, c1_0, c2_0, c3_0]
//   Time 1: [c0_1, c1_1, c2_1, c3_1]
//   ...

static std::vector<int32_t> apply_delay_pattern_undelay(const std::vector<int32_t>& delayed_codes, int num_codebooks,
                                                        int bos_token_id, int eos_token_id) {
    // delayed_codes: (num_steps * num_codebooks)
    // Each step has num_codebooks tokens, where codebook k's token at step t
    // corresponds to audio time (t - k).
    int num_steps = (int)delayed_codes.size() / num_codebooks;
    if (num_steps <= num_codebooks)
        return {};

    int T_audio = num_steps - num_codebooks + 1;
    std::vector<int32_t> aligned(T_audio * num_codebooks, eos_token_id);

    for (int t = 0; t < T_audio; t++) {
        for (int k = 0; k < num_codebooks; k++) {
            int step = t + k;
            if (step < num_steps) {
                int32_t tok = delayed_codes[step * num_codebooks + k];
                if (tok != bos_token_id && tok != eos_token_id) {
                    aligned[t * num_codebooks + k] = tok;
                }
            }
        }
    }
    return aligned;
}

// ── Sampling ────────────────────────────────────────────────────────

static int32_t sample_token(const float* logits, int vocab_size, float temperature, std::mt19937& rng, int top_k = 0) {
    // Optional top-k: keep only the top_k highest logits
    std::vector<float> filtered(logits, logits + vocab_size);
    if (top_k > 0 && top_k < vocab_size) {
        std::vector<float> sorted_logits(filtered.begin(), filtered.end());
        std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + top_k, sorted_logits.end(),
                          std::greater<float>());
        float threshold = sorted_logits[top_k - 1];
        for (int i = 0; i < vocab_size; i++) {
            if (filtered[i] < threshold)
                filtered[i] = -1e30f;
        }
    }

    if (temperature <= 0.0f || temperature < 1e-6f) {
        // Greedy
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (filtered[i] > filtered[best])
                best = i;
        }
        return (int32_t)best;
    }

    // Temperature-scaled softmax + multinomial sampling
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(filtered.begin(), filtered.end());
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp((filtered[i] - max_logit) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++)
        probs[i] /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return (int32_t)dist(rng);
}

// ── Public API ──────────────────────────────────────────────────────

struct parler_tts_context_params parler_tts_context_default_params(void) {
    return {
        /*.n_threads       =*/4,
        /*.verbosity       =*/1,
        /*.use_gpu         =*/false,
        /*.temperature     =*/1.0f,
        /*.seed            =*/0,
        /*.max_audio_tokens=*/0,
        /*.top_k           =*/0,
        /*.flash_attn      =*/false,
    };
}

struct parler_tts_context* parler_tts_init_from_file(const char* path_model, struct parler_tts_context_params params) {
    auto* ctx = new parler_tts_context{};
    ctx->params = params;

    // Seed RNG
    if (params.seed != 0) {
        ctx->rng.seed((unsigned)params.seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }

    // Load GGUF
    if (params.verbosity >= 1)
        fprintf(stderr, "parler_tts: loading '%s'\n", path_model);

    // Pass 1: metadata (hyperparameters + vocab)
    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        fprintf(stderr, "parler_tts: failed to open '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    load_metadata(ctx, meta);
    core_gguf::free_metadata(meta);

    // Pass 2: allocate backend buffer and load weights
    ctx->backend_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, params.n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, params.n_threads);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "parler_tts", wl)) {
        fprintf(stderr, "parler_tts: failed to load weights from '%s'\n", path_model);
        if (ctx->backend && ctx->backend != ctx->backend_cpu)
            ggml_backend_free(ctx->backend);
        ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->tensors = std::move(wl.tensors);

    if (params.verbosity >= 1)
        fprintf(stderr, "parler_tts: loaded %zu tensors\n", ctx->tensors.size());

    bind_tensors(ctx);

    // Permute DAC ConvTranspose1d weights for decomposed path
    {
        const int n = 4; // DAC always has 4 decoder blocks
        std::vector<ggml_tensor*> srcs(n);
        std::vector<ggml_tensor**> dsts(n);
        for (int i = 0; i < n; i++) {
            srcs[i] = ctx->model.dac.blocks[i].up_w;
            dsts[i] = &ctx->model.dac.blocks[i].up_w_perm;
        }
        core_convt::permute_convt1d_weights_batch(srcs.data(), dsts.data(), n, ctx->backend, &ctx->ctx_perm,
                                                  &ctx->buf_perm);
    }

    if (params.verbosity >= 1) {
        auto& dc = ctx->model.dec_cfg;
        auto& tc = ctx->model.t5_cfg;
        fprintf(stderr, "parler_tts: T5 enc  d=%d h=%d L=%d d_kv=%d d_ff=%d\n", tc.d_model, tc.n_heads, tc.n_layers,
                tc.d_kv, tc.d_ff);
        fprintf(stderr, "parler_tts: decoder d=%d h=%d L=%d ffn=%d cb=%d vocab=%d\n", dc.hidden_size, dc.num_heads,
                dc.num_layers, dc.ffn_dim, dc.num_codebooks, dc.vocab_size);
        fprintf(stderr, "parler_tts: DAC     cb=%d cbsz=%d sr=%d hop=%d\n", ctx->model.dac_cfg.n_codebooks,
                ctx->model.dac_cfg.codebook_size, ctx->model.dac_cfg.sample_rate, ctx->model.dac_cfg.hop_length);
    }

    // Create backend scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 65536, false, false);
    }

    return ctx;
}

// ── Cast helper for F16 weights ────────────────────────────────────
static ggml_tensor* cast_f32(ggml_context* ctx, ggml_tensor* t) {
    return (t && t->type != GGML_TYPE_F32) ? ggml_cast(ctx, t, GGML_TYPE_F32) : t;
}

// ── T5 RMS norm helper ──────────────────────────────────────────────

static ggml_tensor* parler_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, cast_f32(ctx, weight));
}

// ── T5 encoder graph builder ────────────────────────────────────────

static ggml_cgraph* build_t5_encoder_graph(parler_tts_context* c, int T) {
    const auto& m = c->model;
    const auto& hp = m.t5_cfg;
    const int nh = hp.n_heads;
    const int hd = hp.d_kv;

    size_t meta_sz = ggml_tensor_overhead() * 16384 + ggml_graph_overhead();
    c->compute_meta.resize(meta_sz);
    ggml_init_params ip = {meta_sz, c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(inp, "enc_tokens");
    ggml_set_input(inp);

    // Position bucket indices (precomputed on CPU): (T, T) i32
    ggml_tensor* pos_bucket = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, T, T);
    ggml_set_name(pos_bucket, "enc_pos_bucket");
    ggml_set_input(pos_bucket);

    // Compute position bias in-graph: get_rows from bias table
    // bias table: (n_heads, num_buckets) -> get_rows with flattened bucket indices
    ggml_tensor* pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, T * T);
    ggml_tensor* pos_bias = ggml_get_rows(ctx0, m.t5_rel_bias, pos_bucket_1d);
    // pos_bias: (n_heads, T*T) -> reshape to (n_heads, T, T) -> permute to (T, T, nh)
    pos_bias = ggml_reshape_3d(ctx0, pos_bias, nh, T, T);
    pos_bias = ggml_cont(ctx0, ggml_permute(ctx0, pos_bias, 2, 0, 1, 3)); // (T_k, T_q, nh)

    // Embedding lookup (T5 uses no positional embedding - relies on relative bias)
    ggml_tensor* cur = ggml_get_rows(ctx0, m.t5_embed, inp);

    for (int il = 0; il < hp.n_layers; il++) {
        const auto& l = m.t5_layers[il];
        ggml_tensor* residual = cur;

        // Pre-attention RMS norm
        cur = parler_rms_norm(ctx0, cur, l.attn_rms, hp.layer_norm_eps);

        // Self-attention: manual Q/K/V + position bias (T5 has no attention scaling)
        ggml_tensor* Q = ggml_mul_mat(ctx0, l.attn_q, cur); // (nh*hd, T)
        ggml_tensor* K = ggml_mul_mat(ctx0, l.attn_k, cur);
        ggml_tensor* V = ggml_mul_mat(ctx0, l.attn_v, cur);

        // Reshape to (hd, nh, T) then permute to (hd, T, nh)
        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3);
        K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3);
        V = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3);

        // KQ = K^T @ Q -> (T_k, T_q, nh) — ggml_mul_mat contracts ne[0]
        ggml_tensor* kq = ggml_mul_mat(ctx0, K, Q); // (T, T, nh)
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        // Add position bias (bidirectional, no causal mask)
        kq = ggml_add(ctx0, kq, pos_bias);

        // Softmax (scale=1.0 for T5)
        kq = ggml_soft_max(ctx0, kq);

        // V @ softmax(KQ) -> (hd, T_q, nh)
        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, V)); // (T, hd, nh)
        ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);              // (hd, T, nh)

        // Permute back: (hd, nh, T) -> (nh*hd, T)
        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, nh * hd, T);

        cur = ggml_mul_mat(ctx0, l.attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // FFN
        residual = cur;
        cur = parler_rms_norm(ctx0, cur, l.ffn_rms, hp.layer_norm_eps);

        // Gated-GELU: gate = GELU(W_gate @ x), up = W_up @ x, out = W_down @ (gate * up)
        ggml_tensor* gate = ggml_gelu(ctx0, ggml_mul_mat(ctx0, l.ffn_gate, cur));
        ggml_tensor* up = ggml_mul_mat(ctx0, l.ffn_up, cur);
        cur = ggml_mul_mat(ctx0, l.ffn_down, ggml_mul(ctx0, gate, up));

        cur = ggml_add(ctx0, cur, residual);
    }

    // Final RMS norm
    cur = parler_rms_norm(ctx0, cur, m.t5_final_rms, hp.layer_norm_eps);

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

int parler_tts_set_description(struct parler_tts_context* ctx, const char* description) {
    if (!ctx || !description)
        return -1;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: encoding description: '%s'\n", description);

    // Tokenize the description.
    // The model uses a LLaMA BPE tokenizer (sentencepiece .model). The GGUF
    // currently stores the BPE vocab from tokenizer.json which our unigram Viterbi
    // tokenizer can't handle correctly. Override via PARLER_DESC_IDS env var
    // (comma-separated ints) for diff-testing.
    std::vector<int> desc_ids;
    const char* override_ids = getenv("PARLER_DESC_IDS");
    if (override_ids && override_ids[0]) {
        // Parse comma-separated IDs for diff-testing
        std::string s(override_ids);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t next = s.find(',', pos);
            if (next == std::string::npos)
                next = s.size();
            desc_ids.push_back(std::stoi(s.substr(pos, next - pos)));
            pos = next + 1;
        }
        fprintf(stderr, "parler_tts: using PARLER_DESC_IDS override (%zu tokens)\n", desc_ids.size());
    } else if (!ctx->prompt_tokenizer.spm_vocab.empty() && !ctx->prompt_tokenizer.scores.empty()) {
        // Use sentencepiece tokenizer (BPE or Viterbi unigram, same tokenizer for desc + prompt)
        fprintf(stderr, "parler_tts: using core_spm %s tokenizer (vocab=%zu, scores=%zu)\n",
                ctx->tokenizer_is_bpe ? "BPE" : "unigram", ctx->prompt_tokenizer.spm_vocab.size(),
                ctx->prompt_tokenizer.scores.size());
        core_spm::Config cfg;
        cfg.unk_id = ctx->prompt_tokenizer.unk_id;
        std::vector<int32_t> ids32;
        if (ctx->tokenizer_is_bpe) {
            ids32 = core_spm::tokenize_bpe(description, ctx->prompt_tokenizer.spm_vocab, ctx->prompt_tokenizer.scores,
                                           cfg, true);
        } else {
            ids32 = core_spm::tokenize(description, ctx->prompt_tokenizer.spm_vocab, ctx->prompt_tokenizer.scores, cfg,
                                       true);
        }
        // Prepend BOS
        desc_ids.push_back(ctx->prompt_tokenizer.bos_id);
        for (int32_t id : ids32)
            desc_ids.push_back((int)id);
    } else {
        fprintf(stderr, "parler_tts: fallback tokenizer (spm_vocab=%zu, scores=%zu)\n",
                ctx->prompt_tokenizer.spm_vocab.size(), ctx->prompt_tokenizer.scores.size());
        desc_ids = tokenize_unigram(ctx->desc_tokenizer, description);
        if (desc_ids.empty()) {
            desc_ids = tokenize_unigram(ctx->prompt_tokenizer, description);
        }
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "parler_tts: description tokens (%zu): ", desc_ids.size());
        for (size_t i = 0; i < std::min(desc_ids.size(), (size_t)30); i++)
            fprintf(stderr, "%d ", desc_ids[i]);
        if (desc_ids.size() > 30)
            fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    const auto& hp = ctx->model.t5_cfg;
    const int D = hp.d_model;
    const int T = (int)desc_ids.size();

    // Build and run T5 encoder graph
    ggml_cgraph* gf = build_t5_encoder_graph(ctx, T);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parler_tts: T5 encoder graph alloc failed\n");
        return -1;
    }

    // Set input: token IDs
    std::vector<int32_t> tokens_i32(desc_ids.begin(), desc_ids.end());
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_tokens"), tokens_i32.data(), 0, T * sizeof(int32_t));

    // Compute position bucket indices (bidirectional)
    std::vector<int32_t> buckets(T * T);
    for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++)
            buckets[q * T + k] =
                t5_relative_position_bucket(k - q, true, hp.rel_attn_num_buckets, hp.rel_attn_max_dist);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_pos_bucket"), buckets.data(), 0,
                            buckets.size() * sizeof(int32_t));

    // Compute
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parler_tts: T5 encoder compute failed\n");
        return -1;
    }

    // Read encoder output
    ctx->enc_T = T;
    ctx->enc_hidden.resize(T * D);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "enc_out"), ctx->enc_hidden.data(), 0,
                            ctx->enc_hidden.size() * sizeof(float));
    ctx->enc_cached = true;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "parler_tts: T5 encoder output cached (%d tokens, %d dim)\n", T, D);
        // Print first few values for diff-testing
        fprintf(stderr, "parler_tts: enc_out[0,:5] = ");
        for (int i = 0; i < std::min(5, D); i++)
            fprintf(stderr, "%.6f ", ctx->enc_hidden[i]);
        fprintf(stderr, "\n");
    }

    // Optionally dump encoder output for diff-testing
    const char* dump_path = getenv("PARLER_DUMP_ENC");
    if (dump_path && dump_path[0]) {
        FILE* f = fopen(dump_path, "wb");
        if (f) {
            fwrite(ctx->enc_hidden.data(), sizeof(float), ctx->enc_hidden.size(), f);
            fclose(f);
            fprintf(stderr, "parler_tts: dumped encoder output to %s\n", dump_path);
        }
    }

    return 0;
}

// ── DAC decoder helpers ─────────────────────────────────────────────

// Conv1d: x is (C_in, T), w is (K, C_in, C_out) per ggml convention, b is (C_out,)
// Returns (C_out, T_out) with same-padding for stride=1.
static ggml_tensor* dac_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride, int pad,
                               int dilation) {
    const int Cout = (int)w->ne[2];
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    ggml_tensor* y = ggml_conv_1d(ctx, w, xt, stride, pad, dilation);
    if (b) {
        ggml_tensor* b_f32 = cast_f32(ctx, b);
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b_f32, 1, Cout, 1);
        y = ggml_add(ctx, y, b3);
    }
    int T_out = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, T_out, Cout);
    return ggml_cont(ctx, ggml_transpose(ctx, y)); // (C_out, T_out)
}

// ConvTranspose1d: x is (C_in, T), w is (K, C_out, C_in) per ggml convention
// Applies padding by cropping output. Returns (C_out, T_out).
static ggml_tensor* dac_conv_transpose_1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* w_perm,
                                          ggml_tensor* b, int stride, int pad) {
    if (w_perm) {
        const int K = (int)w->ne[0];
        ggml_tensor* y = core_convt::convt1d_decomp(ctx, x, w_perm, nullptr, stride, K, pad, pad);
        if (b)
            y = ggml_add(ctx, y, cast_f32(ctx, b));
        return y;
    }
    const int Cout = (int)w->ne[1];
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // (T, C_in)
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xt, stride, 0, 1);
    int T_unpad = (int)y->ne[0];
    int T_out = T_unpad - 2 * pad;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout);
    if (pad > 0) {
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_unpad * sizeof(float), (size_t)pad * sizeof(float));
        y = ggml_cont(ctx, y);
    }
    ggml_tensor* yt = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C_out, T_out)
    if (b)
        yt = ggml_add(ctx, yt, cast_f32(ctx, b));
    return yt;
}

// ResidualUnit: snake -> conv(k=7,d) -> snake -> conv(k=1) -> add residual
static ggml_tensor* dac_res_unit(ggml_context* ctx, ggml_tensor* x, const core_dac::DacResUnit& ru, int dilation) {
    ggml_tensor* y = core_dac::snake(ctx, x, ru.alpha0);
    int pad0 = dilation * (7 - 1) / 2;
    y = dac_conv1d(ctx, y, ru.conv0_w, ru.conv0_b, 1, pad0, dilation);
    y = core_dac::snake(ctx, y, ru.alpha1);
    y = dac_conv1d(ctx, y, ru.conv1_w, ru.conv1_b, 1, 0, 1);
    return ggml_add(ctx, x, y);
}

float* parler_tts_synthesize(struct parler_tts_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;
    *out_n_samples = 0;

    if (!ctx->enc_cached) {
        fprintf(stderr, "parler_tts: call parler_tts_set_description() first\n");
        return nullptr;
    }

    // Get audio codes
    int n_codes = 0;
    int32_t* codes = parler_tts_synthesize_codes(ctx, text, &n_codes);
    if (!codes || n_codes <= 0)
        return nullptr;

    const auto& dac = ctx->model.dac;
    const auto& dac_cfg = dac.config;
    const int num_codebooks = dac_cfg.n_codebooks;
    const int T_audio = n_codes / num_codebooks;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: DAC decode %d frames x %d codebooks\n", T_audio, num_codebooks);

    // Build DAC decode graph
    // Step 1: RVQ dequantize - lookup embeddings and project
    // Step 2: Decoder conv stack

    size_t meta_sz = ggml_tensor_overhead() * 32768 + ggml_graph_overhead();
    ctx->compute_meta.resize(meta_sz);
    ggml_init_params ip = {meta_sz, ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input: codebook indices, one tensor per codebook (T_audio,) i32
    std::vector<ggml_tensor*> code_inputs(num_codebooks);
    for (int k = 0; k < num_codebooks; k++) {
        code_inputs[k] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_audio);
        char buf[32];
        snprintf(buf, sizeof(buf), "codes_%d", k);
        ggml_set_name(code_inputs[k], buf);
        ggml_set_input(code_inputs[k]);
    }

    // RVQ dequantize: for each codebook, lookup -> out_proj -> sum
    ggml_tensor* z_q = nullptr;
    for (int k = 0; k < num_codebooks; k++) {
        if (!dac.quantizers[k].codebook || !dac.quantizers[k].out_proj_w)
            continue;
        // Embedding lookup: (codebook_dim, T_audio)
        ggml_tensor* emb = ggml_get_rows(ctx0, dac.quantizers[k].codebook, code_inputs[k]);
        // out_proj: (codebook_dim, T_audio) -> (hidden_size, T_audio) via matmul
        ggml_tensor* proj = ggml_mul_mat(ctx0, dac.quantizers[k].out_proj_w, emb);
        if (dac.quantizers[k].out_proj_b) {
            proj = ggml_add(ctx0, proj, cast_f32(ctx0, dac.quantizers[k].out_proj_b));
        }
        if (z_q == nullptr) {
            z_q = proj;
        } else {
            z_q = ggml_add(ctx0, z_q, proj);
        }
    }

    if (!z_q) {
        fprintf(stderr, "parler_tts: DAC quantizers not loaded\n");
        ggml_free(ctx0);
        parler_tts_codes_free(codes);
        return nullptr;
    }

    // z_q is (hidden_size=1024, T_audio)
    // Decoder conv stack
    ggml_tensor* x = z_q;

    // Input conv: Conv1d(1024, 1536, k=7, p=3)
    x = dac_conv1d(ctx0, x, dac.in_conv_w, dac.in_conv_b, 1, 3, 1);

    // 4 decoder blocks
    for (int b = 0; b < dac_cfg.n_decoder_blocks; b++) {
        const auto& blk = dac.blocks[b];
        int stride = dac_cfg.upsampling_ratios[b];

        // Snake activation
        x = core_dac::snake(ctx0, x, blk.snake_alpha);

        // ConvTranspose1d (upsample): padding = stride/2
        x = dac_conv_transpose_1d(ctx0, x, blk.up_w, blk.up_w_perm, blk.up_b, stride, stride / 2);

        // 3 residual units with dilations 1, 3, 9
        for (int r = 0; r < 3; r++) {
            x = dac_res_unit(ctx0, x, blk.res[r], dac_cfg.residual_dilations[r]);
        }
    }

    // Final: Snake -> Conv1d(96, 1, k=7, p=3) -> Tanh
    x = core_dac::snake(ctx0, x, dac.out_snake_alpha);
    x = dac_conv1d(ctx0, x, dac.out_conv_w, dac.out_conv_b, 1, 3, 1);
    x = ggml_tanh(ctx0, x);

    // x is (1, T_out) - squeeze to 1D
    ggml_tensor* output = ggml_reshape_1d(ctx0, x, ggml_nelements(x));
    ggml_set_name(output, "pcm_out");
    ggml_build_forward_expand(gf, output);
    ggml_free(ctx0);

    // Allocate and compute
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parler_tts: DAC graph alloc failed\n");
        parler_tts_codes_free(codes);
        return nullptr;
    }

    // Set codebook inputs
    for (int k = 0; k < num_codebooks; k++) {
        std::vector<int32_t> cb_codes(T_audio);
        for (int t = 0; t < T_audio; t++) {
            int32_t c = codes[t * num_codebooks + k];
            // Clip to valid codebook range (0..codebook_size-1)
            cb_codes[t] = (c >= 0 && c < dac_cfg.codebook_size) ? c : 0;
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "codes_%d", k);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, buf), cb_codes.data(), 0, T_audio * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parler_tts: DAC compute failed\n");
        parler_tts_codes_free(codes);
        return nullptr;
    }

    // Read output PCM
    int n_samples = (int)ggml_nelements(ggml_graph_get_tensor(gf, "pcm_out"));
    float* pcm = (float*)malloc(n_samples * sizeof(float));
    if (!pcm) {
        parler_tts_codes_free(codes);
        return nullptr;
    }
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "pcm_out"), pcm, 0, n_samples * sizeof(float));
    *out_n_samples = n_samples;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: DAC decode -> %d samples (%.2fs @ %d Hz)\n", n_samples,
                (float)n_samples / ctx->model.dac_cfg.sample_rate, ctx->model.dac_cfg.sample_rate);

    parler_tts_codes_free(codes);
    return pcm;
}

// ── Decoder: build one-step graph ───────────────────────────────────
//
// The decoder runs one step at a time. Each step:
//   input_embed = sum(codebook_embeds[k] for each active codebook) + pos_embed[step]
//   For prefill: input_embed = prompt_embed + pos_embed[step]
//   Then: 24 layers of (LayerNorm -> self-attn -> residual -> LayerNorm -> cross-attn -> residual -> LayerNorm -> FFN -> residual)
//   Output: hidden state -> 9 LM heads -> 9 logit vectors
//
// KV cache is stored in flat CPU vectors and fed as external inputs each step.

// Build a decoder step graph.
// T_dec: number of new tokens to process (prefill: n_prompt+1, incremental: 1)
// past_len: number of tokens already in KV cache (0 for prefill)
// T_enc: encoder sequence length
static ggml_cgraph* build_decoder_step_graph(parler_tts_context* c, int T_dec, int past_len, int T_enc) {
    const auto& m = c->model;
    const auto& dc = m.dec_cfg;
    const int D = dc.hidden_size;
    const int nh = dc.num_heads;
    const int hd = D / nh; // per-head dim
    const int n_layers = dc.num_layers;
    const int n_cb = dc.num_codebooks;
    const int kv_len = past_len + T_dec; // total KV length after this step

    size_t meta_sz = ggml_tensor_overhead() * 32768 + ggml_graph_overhead();
    c->compute_meta.resize(meta_sz);
    ggml_init_params ip = {meta_sz, c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input: pre-computed hidden states (D, T_dec)
    ggml_tensor* inp_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_dec);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);

    ggml_tensor* cur = inp_hidden;

    // Causal mask for self-attention during prefill (T_dec > 1)
    // For incremental decode (T_dec=1), no mask needed (single query attends to all KV)
    ggml_tensor* causal_mask = nullptr;
    if (T_dec > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kv_len, T_dec);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    char name_buf[64];

    for (int il = 0; il < n_layers; il++) {
        const auto& l = m.dec_layers[il];

        // Self-attention KV cache inputs: (D, past_len) -- past KV
        ggml_tensor* past_k = nullptr;
        ggml_tensor* past_v = nullptr;
        if (past_len > 0) {
            past_k = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, past_len);
            snprintf(name_buf, sizeof(name_buf), "self_k_%d", il);
            ggml_set_name(past_k, name_buf);
            ggml_set_input(past_k);

            past_v = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, past_len);
            snprintf(name_buf, sizeof(name_buf), "self_v_%d", il);
            ggml_set_name(past_v, name_buf);
            ggml_set_input(past_v);
        }

        // Cross-attention KV (precomputed from encoder)
        ggml_tensor* cross_k = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        snprintf(name_buf, sizeof(name_buf), "cross_k_%d", il);
        ggml_set_name(cross_k, name_buf);
        ggml_set_input(cross_k);

        ggml_tensor* cross_v = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        snprintf(name_buf, sizeof(name_buf), "cross_v_%d", il);
        ggml_set_name(cross_v, name_buf);
        ggml_set_input(cross_v);

        ggml_tensor* residual = cur;

        // --- Self-attention ---
        cur = ggml_norm(ctx0, cur, dc.layer_norm_eps);
        cur = ggml_mul(ctx0, cur, l.self_attn_norm_w);
        cur = ggml_add(ctx0, cur, l.self_attn_norm_b);

        // Q/K/V projections for new tokens
        ggml_tensor* Q = ggml_mul_mat(ctx0, l.self_attn_q, cur);     // (D, T_dec)
        ggml_tensor* K_new = ggml_mul_mat(ctx0, l.self_attn_k, cur); // (D, T_dec)
        ggml_tensor* V_new = ggml_mul_mat(ctx0, l.self_attn_v, cur); // (D, T_dec)

        // Output new K/V for cache update — mark as output to prevent
        // the scheduler from reusing their memory before we read them post-compute
        snprintf(name_buf, sizeof(name_buf), "new_k_%d", il);
        ggml_set_name(K_new, name_buf);
        ggml_set_output(K_new);
        ggml_build_forward_expand(gf, K_new);
        snprintf(name_buf, sizeof(name_buf), "new_v_%d", il);
        ggml_set_name(V_new, name_buf);
        ggml_set_output(V_new);
        ggml_build_forward_expand(gf, V_new);

        // Full KV: concatenate past + new
        ggml_tensor* K_full = (past_len > 0) ? ggml_concat(ctx0, past_k, K_new, 1) : K_new;
        ggml_tensor* V_full = (past_len > 0) ? ggml_concat(ctx0, past_v, V_new, 1) : V_new;

        // Reshape to multi-head: (hd, nh, seq) -> permute to (hd, seq, nh)
        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T_dec), 0, 2, 1, 3);
        ggml_tensor* K_mh = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_full, hd, nh, kv_len), 0, 2, 1, 3);
        ggml_tensor* V_mh = ggml_permute(ctx0, ggml_reshape_3d(ctx0, V_full, hd, nh, kv_len), 0, 2, 1, 3);

        // QK^T: (kv_len, T_dec, nh)
        ggml_tensor* kq = ggml_mul_mat(ctx0, K_mh, Q);
        kq = ggml_scale(ctx0, kq, 1.0f / sqrtf((float)hd));

        // Apply causal mask if prefilling (T_dec > 1)
        if (causal_mask) {
            // mask shape: (kv_len, T_dec) — broadcast across heads
            kq = ggml_add(ctx0, kq, causal_mask);
        }

        kq = ggml_soft_max(ctx0, kq);

        // V @ softmax(QK^T) -> (hd, T_dec, nh)
        ggml_tensor* v_t = ggml_cont(ctx0, ggml_transpose(ctx0, V_mh));
        ggml_tensor* kqv = ggml_mul_mat(ctx0, v_t, kq);

        // Merge heads: (hd, nh, T_dec) -> (D, T_dec)
        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, D, T_dec);

        cur = ggml_mul_mat(ctx0, l.self_attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // --- Cross-attention ---
        residual = cur;
        cur = ggml_norm(ctx0, cur, dc.layer_norm_eps);
        cur = ggml_mul(ctx0, cur, l.cross_attn_norm_w);
        cur = ggml_add(ctx0, cur, l.cross_attn_norm_b);

        Q = ggml_mul_mat(ctx0, l.cross_attn_q, cur); // (D, T_dec)
        Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T_dec), 0, 2, 1, 3);

        ggml_tensor* CK = ggml_permute(ctx0, ggml_reshape_3d(ctx0, cross_k, hd, nh, T_enc), 0, 2, 1, 3);
        ggml_tensor* CV = ggml_permute(ctx0, ggml_reshape_3d(ctx0, cross_v, hd, nh, T_enc), 0, 2, 1, 3);

        kq = ggml_mul_mat(ctx0, CK, Q); // (T_enc, T_dec, nh)
        kq = ggml_scale(ctx0, kq, 1.0f / sqrtf((float)hd));
        kq = ggml_soft_max(ctx0, kq);

        ggml_tensor* cv_t = ggml_cont(ctx0, ggml_transpose(ctx0, CV));
        kqv = ggml_mul_mat(ctx0, cv_t, kq);

        cur = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, D, T_dec);

        cur = ggml_mul_mat(ctx0, l.cross_attn_o, cur);
        cur = ggml_add(ctx0, cur, residual);

        // --- FFN ---
        residual = cur;
        cur = ggml_norm(ctx0, cur, dc.layer_norm_eps);
        cur = ggml_mul(ctx0, cur, l.ffn_norm_w);
        cur = ggml_add(ctx0, cur, l.ffn_norm_b);

        cur = ggml_gelu(ctx0, ggml_mul_mat(ctx0, l.fc1, cur));
        cur = ggml_mul_mat(ctx0, l.fc2, cur);
        cur = ggml_add(ctx0, cur, residual);
    }

    // Final LayerNorm
    cur = ggml_norm(ctx0, cur, dc.layer_norm_eps);
    cur = ggml_mul(ctx0, cur, m.dec_final_norm_w);
    cur = ggml_add(ctx0, cur, m.dec_final_norm_b);

    // For logits, only take the LAST position (for AR generation)
    if (T_dec > 1) {
        // Extract last column: view at offset (T_dec-1)*D
        cur = ggml_view_1d(ctx0, cur, D, (size_t)(T_dec - 1) * D * sizeof(float));
        cur = ggml_reshape_2d(ctx0, cur, D, 1);
    }

    // LM heads: one per codebook
    for (int k = 0; k < n_cb; k++) {
        ggml_tensor* logits_k = ggml_mul_mat(ctx0, m.lm_heads[k], cur); // (vocab_size, 1)
        snprintf(name_buf, sizeof(name_buf), "logits_%d", k);
        ggml_set_name(logits_k, name_buf);
        ggml_build_forward_expand(gf, logits_k);
    }

    ggml_free(ctx0);
    return gf;
}

int32_t* parler_tts_synthesize_codes(struct parler_tts_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text || !out_n)
        return nullptr;
    *out_n = 0;

    if (!ctx->enc_cached) {
        fprintf(stderr, "parler_tts: call parler_tts_set_description() first\n");
        return nullptr;
    }

    // Tokenize the text prompt (override via PARLER_PROMPT_IDS for diff-testing)
    std::vector<int> prompt_ids;
    const char* prompt_override = getenv("PARLER_PROMPT_IDS");
    if (prompt_override && prompt_override[0]) {
        std::string s(prompt_override);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t next = s.find(',', pos);
            if (next == std::string::npos)
                next = s.size();
            prompt_ids.push_back(std::stoi(s.substr(pos, next - pos)));
            pos = next + 1;
        }
        fprintf(stderr, "parler_tts: using PARLER_PROMPT_IDS override (%zu tokens)\n", prompt_ids.size());
    } else if (!ctx->prompt_tokenizer.spm_vocab.empty() && !ctx->prompt_tokenizer.scores.empty()) {
        core_spm::Config cfg;
        cfg.unk_id = ctx->prompt_tokenizer.unk_id;
        std::vector<int32_t> ids32;
        if (ctx->tokenizer_is_bpe) {
            ids32 =
                core_spm::tokenize_bpe(text, ctx->prompt_tokenizer.spm_vocab, ctx->prompt_tokenizer.scores, cfg, true);
        } else {
            ids32 = core_spm::tokenize(text, ctx->prompt_tokenizer.spm_vocab, ctx->prompt_tokenizer.scores, cfg, true);
        }
        prompt_ids.push_back(ctx->prompt_tokenizer.bos_id);
        for (int32_t id : ids32)
            prompt_ids.push_back((int)id);
    } else {
        prompt_ids = tokenize_unigram(ctx->prompt_tokenizer, text);
    }
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "parler_tts: prompt tokens (%zu): ", prompt_ids.size());
        for (size_t i = 0; i < std::min(prompt_ids.size(), (size_t)15); i++)
            fprintf(stderr, "%d ", prompt_ids[i]);
        if (prompt_ids.size() > 15)
            fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }

    const auto& m = ctx->model;
    const auto& dc = m.dec_cfg;
    const int D = dc.hidden_size;
    const int num_codebooks = dc.num_codebooks;
    // Default max_gen: use user param, otherwise model default (2580 ≈ 30s audio)
    int max_gen = ctx->params.max_audio_tokens > 0 ? ctx->params.max_audio_tokens : dc.max_generation;
    const int T_enc = ctx->enc_T;
    const int n_layers = dc.num_layers;
    const float temperature = ctx->params.temperature;

    // Precompute cross-attention KV projections (one-time per description)
    // cross_k[layer] = K_proj @ enc_hidden, cross_v[layer] = V_proj @ enc_hidden
    // Each is (D, T_enc) stored flat
    std::vector<std::vector<float>> cross_k_cache(n_layers);
    std::vector<std::vector<float>> cross_v_cache(n_layers);

    {
        // Build a small graph for cross KV projection
        size_t meta_sz = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
        ctx->compute_meta.resize(meta_sz);
        ggml_init_params ip = {meta_sz, ctx->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

        ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        ggml_set_name(enc_inp, "enc_for_cross");
        ggml_set_input(enc_inp);

        for (int il = 0; il < n_layers; il++) {
            const auto& l = m.dec_layers[il];
            char buf[64];

            ggml_tensor* ck = ggml_mul_mat(ctx0, l.cross_attn_k, enc_inp); // (D, T_enc)
            snprintf(buf, sizeof(buf), "xk_%d", il);
            ggml_set_name(ck, buf);
            ggml_build_forward_expand(gf, ck);

            ggml_tensor* cv = ggml_mul_mat(ctx0, l.cross_attn_v, enc_inp); // (D, T_enc)
            snprintf(buf, sizeof(buf), "xv_%d", il);
            ggml_set_name(cv, buf);
            ggml_build_forward_expand(gf, cv);
        }

        ggml_free(ctx0);

        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "parler_tts: cross-KV alloc failed\n");
            return nullptr;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_for_cross"), ctx->enc_hidden.data(), 0,
                                (size_t)D * T_enc * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "parler_tts: cross-KV compute failed\n");
            return nullptr;
        }

        for (int il = 0; il < n_layers; il++) {
            char buf[64];
            cross_k_cache[il].resize(D * T_enc);
            cross_v_cache[il].resize(D * T_enc);
            snprintf(buf, sizeof(buf), "xk_%d", il);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, buf), cross_k_cache[il].data(), 0,
                                    (size_t)D * T_enc * sizeof(float));
            snprintf(buf, sizeof(buf), "xv_%d", il);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, buf), cross_v_cache[il].data(), 0,
                                    (size_t)D * T_enc * sizeof(float));
        }
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: cross-attention KV cached (%d layers x %d enc tokens)\n", n_layers, T_enc);

    // KV cache is allocated below in the generation state section

    // Read positional embedding table (handles F32, F16, and quantized types)
    std::vector<float> pos_embed_table;
    if (m.dec_pos_embed) {
        int pos_dim = (int)m.dec_pos_embed->ne[0]; // D
        int max_pos = (int)m.dec_pos_embed->ne[1]; // max_position_embeddings
        pos_embed_table.resize((size_t)pos_dim * max_pos);
        if (m.dec_pos_embed->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(m.dec_pos_embed, pos_embed_table.data(), 0, pos_embed_table.size() * sizeof(float));
        } else {
            // Dequantize row by row
            size_t row_bytes = ggml_row_size(m.dec_pos_embed->type, pos_dim);
            std::vector<uint8_t> row_buf(row_bytes);
            const struct ggml_type_traits* traits = ggml_get_type_traits(m.dec_pos_embed->type);
            for (int r = 0; r < max_pos; r++) {
                ggml_backend_tensor_get(m.dec_pos_embed, row_buf.data(), (size_t)r * row_bytes, row_bytes);
                if (m.dec_pos_embed->type == GGML_TYPE_F16) {
                    const ggml_fp16_t* fp16 = (const ggml_fp16_t*)row_buf.data();
                    for (int d = 0; d < pos_dim; d++)
                        pos_embed_table[r * pos_dim + d] = ggml_fp16_to_fp32(fp16[d]);
                } else if (traits && traits->to_float) {
                    traits->to_float((const char*)row_buf.data(), &pos_embed_table[r * pos_dim], pos_dim);
                }
            }
        }
    }

    // Helper: read one embedding row via ggml_backend_tensor_get (works with backend buffers)
    // Handles F32, F16, and quantized types (Q8_0, Q4_K, etc.)
    auto read_embed_row = [&](ggml_tensor* emb_table, int token_id, float* out, int dim) {
        if (!emb_table || token_id < 0 || token_id >= (int)emb_table->ne[1])
            return;
        size_t row_bytes = ggml_row_size(emb_table->type, dim);
        std::vector<uint8_t> row_buf(row_bytes);
        ggml_backend_tensor_get(emb_table, row_buf.data(), (size_t)token_id * row_bytes, row_bytes);
        if (emb_table->type == GGML_TYPE_F32) {
            memcpy(out, row_buf.data(), dim * sizeof(float));
        } else if (emb_table->type == GGML_TYPE_F16) {
            const ggml_fp16_t* fp16_src = (const ggml_fp16_t*)row_buf.data();
            for (int i = 0; i < dim; i++)
                out[i] = ggml_fp16_to_fp32(fp16_src[i]);
        } else {
            // Quantized type: dequantize the row
            const struct ggml_type_traits* traits = ggml_get_type_traits(emb_table->type);
            if (traits && traits->to_float) {
                traits->to_float((const char*)row_buf.data(), out, dim);
            } else {
                // Fallback: zero
                memset(out, 0, dim * sizeof(float));
            }
        }
    };

    // Generation state
    std::vector<int32_t> delayed_codes;
    const int n_prompt = (int)prompt_ids.size();
    std::vector<bool> eos_reached(num_codebooks, false);

    // Helper to set graph inputs
    auto safe_set = [&](ggml_cgraph* g, const char* name, const void* data, size_t sz) -> bool {
        ggml_tensor* t = ggml_graph_get_tensor(g, name);
        if (!t) {
            fprintf(stderr, "parler_tts: tensor '%s' not found\n", name);
            return false;
        }
        if (!t->buffer) {
            fprintf(stderr, "parler_tts: tensor '%s' no buffer\n", name);
            return false;
        }
        ggml_backend_tensor_set(t, data, 0, sz);
        return true;
    };

    // Total KV cache capacity = n_prompt + 1 (BOS) + max_gen
    const int kv_capacity = n_prompt + 1 + max_gen;
    ctx->kv_k.resize((size_t)n_layers * kv_capacity * D, 0.0f);
    ctx->kv_v.resize((size_t)n_layers * kv_capacity * D, 0.0f);

    // ── Prefill: process all prompt tokens + first BOS codebook token together ──
    // Build input: [prompt_embed_0, prompt_embed_1, ..., prompt_embed_{n-1}, bos_sum]
    // with positional embedding added
    const int prefill_len = n_prompt + 1; // prompt tokens + 1 BOS codebook token
    std::vector<float> prefill_embed(prefill_len * D, 0.0f);

    // Prompt tokens
    const bool parler_debug = (getenv("PARLER_DEBUG") && getenv("PARLER_DEBUG")[0] == '1');
    for (int i = 0; i < n_prompt; i++) {
        read_embed_row(m.dec_embed_prompts, prompt_ids[i], &prefill_embed[i * D], D);
    }
    // BOS codebook sum: sum(embed_k(BOS) for k=0..8)
    for (int k = 0; k < num_codebooks; k++) {
        std::vector<float> bos_emb(D, 0.0f);
        read_embed_row(m.dec_embeds[k], dc.bos_token_id, bos_emb.data(), D);
        for (int d = 0; d < D; d++)
            prefill_embed[n_prompt * D + d] += bos_emb[d];
    }
    // Add positional embeddings to all positions
    for (int i = 0; i < prefill_len; i++) {
        if ((size_t)(i + 1) * D <= pos_embed_table.size()) {
            for (int d = 0; d < D; d++)
                prefill_embed[i * D + d] += pos_embed_table[(size_t)i * D + d];
        }
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: prefill %d tokens (%d prompt + 1 BOS)...\n", prefill_len, n_prompt);

    // Build causal mask for prefill: (kv_len=prefill_len, T_dec=prefill_len)
    // mask[q][k] = 0.0 if k <= q, else -inf
    std::vector<float> causal_mask_data(prefill_len * prefill_len, 0.0f);
    for (int q = 0; q < prefill_len; q++)
        for (int k = 0; k < prefill_len; k++)
            if (k > q)
                causal_mask_data[q * prefill_len + k] = -1e9f;

    {
        ggml_cgraph* gf = build_decoder_step_graph(ctx, prefill_len, 0, T_enc);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "parler_tts: prefill alloc failed\n");
            return nullptr;
        }

        safe_set(gf, "inp_hidden", prefill_embed.data(), (size_t)prefill_len * D * sizeof(float));
        safe_set(gf, "causal_mask", causal_mask_data.data(), causal_mask_data.size() * sizeof(float));

        for (int il = 0; il < n_layers; il++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "cross_k_%d", il);
            safe_set(gf, buf, cross_k_cache[il].data(), (size_t)D * T_enc * sizeof(float));
            snprintf(buf, sizeof(buf), "cross_v_%d", il);
            safe_set(gf, buf, cross_v_cache[il].data(), (size_t)D * T_enc * sizeof(float));
        }

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "parler_tts: prefill compute failed\n");
            return nullptr;
        }

        // (PARLER_DEBUG=1 intermediates available via PARLER_DUMP_ENC env var)

        // Read KV cache from prefill (T_dec tokens per layer)
        for (int il = 0; il < n_layers; il++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "new_k_%d", il);
            float* k_dst = ctx->kv_k.data() + (size_t)il * kv_capacity * D;
            ggml_tensor* kt = ggml_graph_get_tensor(gf, buf);
            ggml_backend_tensor_get(kt, k_dst, 0, (size_t)prefill_len * D * sizeof(float));
            snprintf(buf, sizeof(buf), "new_v_%d", il);
            float* v_dst = ctx->kv_v.data() + (size_t)il * kv_capacity * D;
            ggml_tensor* vt = ggml_graph_get_tensor(gf, buf);
            ggml_backend_tensor_get(vt, v_dst, 0, (size_t)prefill_len * D * sizeof(float));

            if (parler_debug && (il == 0 || il == 23)) {
                fprintf(stderr, "parler_tts: KV cache layer %d K[:5]=%.4f %.4f %.4f %.4f %.4f data=%p\n", il, k_dst[0],
                        k_dst[1], k_dst[2], k_dst[3], k_dst[4], (void*)kt->data);
            }
        }

        // Read first-step logits and sample (from the prefill's last position)
        for (int k = 0; k < num_codebooks; k++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "logits_%d", k);
            std::vector<float> logits(dc.vocab_size);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, buf), logits.data(), 0, dc.vocab_size * sizeof(float));

            if (parler_debug && k == 0) {
                fprintf(stderr, "parler_tts: prefill logits[0,:5] = ");
                for (int j = 0; j < std::min(5, dc.vocab_size); j++)
                    fprintf(stderr, "%.4f ", logits[j]);
                int am = 0;
                for (int j = 1; j < dc.vocab_size; j++)
                    if (logits[j] > logits[am])
                        am = j;
                fprintf(stderr, " argmax=%d\n", am);
            }

            // First gen step: codebook 0 starts producing, others are BOS (delay)
            if (k > 0) {
                delayed_codes.push_back(dc.bos_token_id);
            } else {
                int32_t tok = sample_token(logits.data(), dc.vocab_size, temperature, ctx->rng, ctx->params.top_k);
                delayed_codes.push_back(tok);
                if (tok == dc.eos_token_id)
                    eos_reached[0] = true;
            }
        }

        // Dump prefill tokens
        if (parler_debug) {
            fprintf(stderr, "parler_tts: step 0 tokens:");
            int last_idx = (int)delayed_codes.size() - num_codebooks;
            for (int k = 0; k < num_codebooks; k++)
                fprintf(stderr, " %d", delayed_codes[last_idx + k]);
            fprintf(stderr, "\n");
        }
    }

    int past_len = prefill_len; // KV cache now has prefill_len entries

    // ── Incremental AR decode loop ──
    for (int gen_step = 1; gen_step < max_gen; gen_step++) {
        // Input embedding: sum of codebook embeddings for previous step's tokens + pos embed
        std::vector<float> inp_embed(D, 0.0f);

        // Sum ALL codebook embeddings for previous step (including BOS/EOS — matches upstream)
        int prev_idx = (int)delayed_codes.size() / num_codebooks - 1;
        for (int k = 0; k < num_codebooks; k++) {
            int32_t tok = delayed_codes[prev_idx * num_codebooks + k];
            std::vector<float> emb(D, 0.0f);
            read_embed_row(m.dec_embeds[k], tok, emb.data(), D);
            for (int d = 0; d < D; d++)
                inp_embed[d] += emb[d];
        }

        // Add positional embedding at position (past_len)
        int pos = past_len; // = n_prompt + 1 + gen_step
        if ((size_t)(pos + 1) * D <= pos_embed_table.size()) {
            for (int d = 0; d < D; d++)
                inp_embed[d] += pos_embed_table[(size_t)pos * D + d];
        }

        if (parler_debug && gen_step <= 2) {
            fprintf(stderr, "parler_tts: step %d inp_embed[:5] = %.6f %.6f %.6f %.6f %.6f pos=%d\n", gen_step,
                    inp_embed[0], inp_embed[1], inp_embed[2], inp_embed[3], inp_embed[4], pos);
        }

        // Build incremental step graph
        ggml_cgraph* gf = build_decoder_step_graph(ctx, 1, past_len, T_enc);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "parler_tts: gen step %d alloc failed\n", gen_step);
            break;
        }

        safe_set(gf, "inp_hidden", inp_embed.data(), D * sizeof(float));

        for (int il = 0; il < n_layers; il++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "self_k_%d", il);
            safe_set(gf, buf, ctx->kv_k.data() + (size_t)il * kv_capacity * D, (size_t)past_len * D * sizeof(float));
            snprintf(buf, sizeof(buf), "self_v_%d", il);
            safe_set(gf, buf, ctx->kv_v.data() + (size_t)il * kv_capacity * D, (size_t)past_len * D * sizeof(float));
            snprintf(buf, sizeof(buf), "cross_k_%d", il);
            safe_set(gf, buf, cross_k_cache[il].data(), (size_t)D * T_enc * sizeof(float));
            snprintf(buf, sizeof(buf), "cross_v_%d", il);
            safe_set(gf, buf, cross_v_cache[il].data(), (size_t)D * T_enc * sizeof(float));
        }

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "parler_tts: gen step %d compute failed\n", gen_step);
            break;
        }

        // Read new K/V and append to cache
        for (int il = 0; il < n_layers; il++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "new_k_%d", il);
            float* k_dst = ctx->kv_k.data() + (size_t)il * kv_capacity * D + (size_t)past_len * D;
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, buf), k_dst, 0, D * sizeof(float));
            snprintf(buf, sizeof(buf), "new_v_%d", il);
            float* v_dst = ctx->kv_v.data() + (size_t)il * kv_capacity * D + (size_t)past_len * D;
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, buf), v_dst, 0, D * sizeof(float));
        }
        past_len++;

        // Read logits and sample for each codebook
        std::vector<int32_t> step_tokens(num_codebooks);
        bool all_eos = true;

        for (int k = 0; k < num_codebooks; k++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "logits_%d", k);
            std::vector<float> logits(dc.vocab_size);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, buf), logits.data(), 0, dc.vocab_size * sizeof(float));

            // Dump step logits for diff-testing
            if (parler_debug && gen_step < 3 && k == 0) {
                fprintf(stderr, "parler_tts: step %d logits[0,:5] = %.4f %.4f %.4f %.4f %.4f argmax=%d\n", gen_step,
                        logits[0], logits[1], logits[2], logits[3], logits[4],
                        (int)(std::max_element(logits.begin(), logits.end()) - logits.begin()));
            }

            // Delay pattern: codebook k starts producing after k steps
            if (gen_step < k) {
                step_tokens[k] = dc.bos_token_id;
                all_eos = false;
            } else if (eos_reached[k]) {
                step_tokens[k] = dc.eos_token_id;
            } else {
                int32_t tok = sample_token(logits.data(), dc.vocab_size, temperature, ctx->rng, ctx->params.top_k);
                if (tok == dc.eos_token_id) {
                    eos_reached[k] = true;
                    step_tokens[k] = dc.eos_token_id;
                } else {
                    step_tokens[k] = tok;
                    all_eos = false;
                }
            }
        }

        delayed_codes.insert(delayed_codes.end(), step_tokens.begin(), step_tokens.end());

        if (parler_debug && gen_step < 20) {
            fprintf(stderr, "parler_tts: step %d tokens:", gen_step);
            for (int k = 0; k < num_codebooks; k++)
                fprintf(stderr, " %d", step_tokens[k]);
            fprintf(stderr, "\n");
        }

        if (all_eos) {
            if (ctx->params.verbosity >= 1)
                fprintf(stderr, "parler_tts: all codebooks hit EOS at gen step %d\n", gen_step);
            break;
        }
    }
    fprintf(stderr, "parler_tts: decoder loop done. delayed_codes size=%zu\n", delayed_codes.size());
    if (delayed_codes.empty()) {
        fprintf(stderr, "parler_tts: no audio codes generated\n");
        return nullptr;
    }

    // Un-delay
    auto aligned = apply_delay_pattern_undelay(delayed_codes, num_codebooks, dc.bos_token_id, dc.eos_token_id);

    if (aligned.empty()) {
        fprintf(stderr, "parler_tts: no valid audio codes after un-delay\n");
        return nullptr;
    }

    int n = (int)aligned.size();
    int32_t* result = (int32_t*)malloc(n * sizeof(int32_t));
    memcpy(result, aligned.data(), n * sizeof(int32_t));
    *out_n = n;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "parler_tts: generated %d aligned codes (%d frames x %d codebooks)\n", n, n / num_codebooks,
                num_codebooks);

    return result;
}

void parler_tts_codes_free(int32_t* codes) {
    free(codes);
}

void parler_tts_pcm_free(float* pcm) {
    free(pcm);
}

void parler_tts_free(struct parler_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_perm)
        ggml_backend_buffer_free(ctx->buf_perm);
    if (ctx->ctx_perm)
        ggml_free(ctx->ctx_perm);
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

void parler_tts_set_n_threads(struct parler_tts_context* ctx, int n_threads) {
    if (ctx && ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

void parler_tts_set_temperature(struct parler_tts_context* ctx, float temperature) {
    if (ctx)
        ctx->params.temperature = temperature;
}

void parler_tts_set_seed(struct parler_tts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->params.seed = seed;
    if (seed != 0) {
        ctx->rng.seed((unsigned)seed);
    } else {
        std::random_device rd;
        ctx->rng.seed(rd());
    }
}
