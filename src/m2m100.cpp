// m2m100.cpp — M2M-100 machine translation via ggml
//
// Architecture:
//   Encoder: 12-layer transformer (self-attention + FFN, pre-norm, ReLU)
//   Decoder: 12-layer transformer with cross-attention + KV cache
//   Shared embedding: encoder, decoder, and lm_head share one table
//   Sinusoidal positional embeddings (pre-computed, stored in GGUF)
//
// Supports facebook/m2m100_418M, m2m100_1.2B, and wmt21-dense-24-wide
// (same architecture at different scales).
//
// Also runs MarianMT / Opus-MT (GGUF architecture "marian",
// models/convert-marian-to-gguf.py). Same skeleton, six differences, all read
// from the GGUF and all off for an m2m100 file:
//   post-norm layers and no output LayerNorm    (hp.pre_norm, out_ln absent)
//   activation from config.json (swish)         (hp.activation)
//   positions start at row 0, not row 2         (hp.pos_offset)
//   a bias on the logits                        (final_logits_bias)
//   decoder starts from <pad> alone, no language tokens; <pad> is never
//   generated                                   (hp.suppress_ids)
//   tokenizer: SentencePiece unigram for the pieces, vocab.json for the ids
//                                               (core/marian_tokenizer.h)

#include "m2m100.h"
#include "core/attention.h"
#include "core/beam_decode.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (§232 m2m100 GPU path)
#if defined(GGML_USE_METAL)
#include "ggml-metal.h" // core_cpu_backend::is_metal(§232 CUDA/Vulkan-default gate)
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include "core/sentencepiece.h"
#include "core/marian_tokenizer.h"
#include "core/crispasr_env.h"

#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/ggml_cpu_backend.h"

// ===========================================================================
// Bench instrumentation — `M2M100_BENCH=1` for per-stage timings.
// ===========================================================================

static bool m2m100_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_M2M100_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

// Per-step split of the decoder, accumulated over one translate call.
struct m2m100_step_bench {
    double build_ms = 0, alloc_ms = 0, compute_ms = 0, read_ms = 0;
    int steps = 0;
};
static m2m100_step_bench g_m2m100_step_bench;
static double m2m100_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct m2m100_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit m2m100_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~m2m100_bench_stage() {
        if (!m2m100_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  m2m100_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ── Hyperparameters ──────────────────────────────────────────────

struct m2m100_hparams {
    int vocab_size = 128112;
    int d_model = 1024;
    int enc_n_layers = 12;
    int enc_n_heads = 16;
    int enc_ffn_dim = 4096;
    int dec_n_layers = 12;
    int dec_n_heads = 16;
    int dec_ffn_dim = 4096;
    int max_position_emb = 1024;
    bool scale_embedding = true;
    int bos_token_id = 0;
    int eos_token_id = 2;
    // generation_config.json early_stopping (HF beam semantics only): m2m100
    // 418M/1.2B and wmt21 x-en declare true, wmt21 en-x does not (false).
    // GGUFs predating the key: the 2048-wide wmt21 we ship is en-x -> false.
    bool early_stopping = true;
    int pad_token_id = 1;
    int dec_start_token = 2;
    // ── Marian (every default below is M2M-100's) ──
    enum Activation { ACT_RELU = 0, ACT_SILU = 1, ACT_GELU = 2 };
    bool marian = false;       // general.architecture == "marian"
    bool pre_norm = true;      // config.json normalize_before
    int activation = ACT_RELU; // config.json activation_function
    int pos_offset = 2;        // row of position 0 in the pos_emb tables
    int unk_token_id = 3;
    std::vector<int> suppress_ids; // generation_config bad_words_ids (single tokens)
    int gen_num_beams = 5;
    int gen_max_length = 0;  // 0 = not declared
    std::string source_lang; // tokenizer_config.json; "" when the checkpoint has none
    std::string target_lang;
    int head_dim() const { return d_model / enc_n_heads; }
};

// ── Encoder layer tensors ────────────────────────────────────────

struct m2m100_enc_layer {
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
};

// ── Decoder layer tensors ────────────────────────────────────────

struct m2m100_dec_layer {
    // Self-attention
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* attn_ln_w = nullptr;
    ggml_tensor* attn_ln_b = nullptr;
    // Cross-attention
    ggml_tensor* cross_q_w = nullptr;
    ggml_tensor* cross_q_b = nullptr;
    ggml_tensor* cross_k_w = nullptr;
    ggml_tensor* cross_k_b = nullptr;
    ggml_tensor* cross_v_w = nullptr;
    ggml_tensor* cross_v_b = nullptr;
    ggml_tensor* cross_o_w = nullptr;
    ggml_tensor* cross_o_b = nullptr;
    ggml_tensor* cross_ln_w = nullptr;
    ggml_tensor* cross_ln_b = nullptr;
    // FFN
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_up_b = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
    ggml_tensor* ffn_down_b = nullptr;
    ggml_tensor* ffn_ln_w = nullptr;
    ggml_tensor* ffn_ln_b = nullptr;
};

// ── Model ────────────────────────────────────────────────────────

struct m2m100_model {
    m2m100_hparams hp;

    // Shared embedding (encoder + decoder + lm_head)
    ggml_tensor* shared_embed = nullptr;

    // Positional embeddings (pre-computed sinusoidal)
    ggml_tensor* enc_pos_emb = nullptr;
    ggml_tensor* dec_pos_emb = nullptr;

    // Encoder
    std::vector<m2m100_enc_layer> enc_layers;
    ggml_tensor* enc_out_ln_w = nullptr;
    ggml_tensor* enc_out_ln_b = nullptr;

    // Decoder
    std::vector<m2m100_dec_layer> dec_layers;
    ggml_tensor* dec_out_ln_w = nullptr;
    ggml_tensor* dec_out_ln_b = nullptr;
    // lm_head shares shared_embed (tied weights)
    ggml_tensor* final_logits_bias = nullptr; // Marian only
};

// ── Tokenizer ────────────────────────────────────────────────────

struct m2m100_tokenizer {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;
    // BPE merge scores (= SPM piece scores) + hash map, for core_spm::tokenize_bpe.
    // token_to_id_u / id_to_token include ~390 SPM "intermediate" pieces (ids
    // >= embed_vocab_size) that are absent from the embedding vocab but needed as
    // intermediate BPE merges; a final token with id >= embed_vocab_size maps to
    // <unk> (matches HF's convert_token_to_id).
    std::vector<float> scores;
    std::unordered_map<std::string, int32_t> token_to_id_u;
    int embed_vocab_size = 0; // ids < this have embedding rows
    // Language code → token ID
    std::vector<std::string> lang_codes;
    std::map<std::string, int> lang_to_token_id;
};

// ── Context ──────────────────────────────────────────────────────

struct m2m100_context {
    m2m100_context_params params;
    m2m100_model model;
    m2m100_tokenizer tokenizer;
    core_marian_tok::Tokenizer marian_tok; // used instead of `tokenizer` when hp.marian
    bool warned_lang = false;

    // Weight storage
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Backend
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    // Compute
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for decoder self-attention
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    int kv_max_ctx = 0;

    // Cross-attention KV cache (computed once per encoder run)
    std::vector<ggml_tensor*> cross_kv_k;
    std::vector<ggml_tensor*> cross_kv_v;
    ggml_context* cross_kv_ctx = nullptr;
    ggml_backend_buffer_t cross_kv_buf = nullptr;
    int cross_T_enc = 0;
    int beam_size = m2m100_default_beam_size();
};

// ── Helpers ──────────────────────────────────────────────────────

static ggml_tensor* T(m2m100_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return (it != c->tensors.end()) ? it->second : nullptr;
}

static ggml_tensor* TR(m2m100_context* c, const char* name) {
    auto* t = T(c, name);
    if (!t) {
        fprintf(stderr, "%s: required tensor '%s' not found\n", c->model.hp.marian ? "marian" : "m2m100", name);
    }
    return t;
}

// ── Load metadata ────────────────────────────────────────────────

static bool load_marian_metadata(m2m100_context* c, gguf_context* g);

static bool load_metadata(m2m100_context* c, gguf_context* g) {
    auto& hp = c->model.hp;
    {
        const int aidx = gguf_find_key(g, "general.architecture");
        if (aidx >= 0 && gguf_get_kv_type(g, aidx) == GGUF_TYPE_STRING &&
            std::strcmp(gguf_get_val_str(g, aidx), "marian") == 0)
            return load_marian_metadata(c, g);
    }
    auto get_u32 = [&](const char* key, int def) -> int {
        int idx = gguf_find_key(g, key);
        return (idx >= 0) ? (int)gguf_get_val_u32(g, idx) : def;
    };
    hp.vocab_size = get_u32("m2m100.vocab_size", 128112);
    hp.d_model = get_u32("m2m100.d_model", 1024);
    hp.early_stopping = get_u32("m2m100.gen.early_stopping", hp.d_model == 2048 ? 0 : 1) != 0;
    hp.enc_n_layers = get_u32("m2m100.encoder.n_layers", 12);
    hp.enc_n_heads = get_u32("m2m100.encoder.n_heads", 16);
    hp.enc_ffn_dim = get_u32("m2m100.encoder.ffn_dim", 4096);
    hp.dec_n_layers = get_u32("m2m100.decoder.n_layers", 12);
    hp.dec_n_heads = get_u32("m2m100.decoder.n_heads", 16);
    hp.dec_ffn_dim = get_u32("m2m100.decoder.ffn_dim", 4096);
    hp.max_position_emb = get_u32("m2m100.max_position_embeddings", 1024);
    hp.scale_embedding = get_u32("m2m100.scale_embedding", 1) != 0;
    hp.bos_token_id = get_u32("m2m100.bos_token_id", 0);
    hp.eos_token_id = get_u32("m2m100.eos_token_id", 2);
    hp.pad_token_id = get_u32("m2m100.pad_token_id", 1);
    hp.dec_start_token = get_u32("m2m100.decoder_start_token_id", 2);

    // Load tokenizer
    {
        int tidx = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tidx >= 0) {
            int n = gguf_get_arr_n(g, tidx);
            c->tokenizer.id_to_token.resize(n);
            c->tokenizer.token_to_id_u.reserve(n * 2);
            for (int i = 0; i < n; i++) {
                c->tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
                c->tokenizer.token_to_id[c->tokenizer.id_to_token[i]] = i;
                c->tokenizer.token_to_id_u[c->tokenizer.id_to_token[i]] = i;
            }
        }
        c->tokenizer.embed_vocab_size = hp.vocab_size; // ids >= this are BPE-intermediate only
        // BPE merge scores (= SPM piece scores). M2M-100's SP model is BPE, not
        // Unigram, so tokenization follows merge order — greedy longest-match
        // mis-splits words and degrades translations.
        int sidx = gguf_find_key(g, "tokenizer.ggml.scores");
        if (sidx >= 0) {
            int n = gguf_get_arr_n(g, sidx);
            c->tokenizer.scores.resize(n);
            const float* sp = (const float*)gguf_get_arr_data(g, sidx);
            for (int i = 0; i < n; i++)
                c->tokenizer.scores[i] = sp[i];
        }
    }

    // Load language codes
    {
        int cidx = gguf_find_key(g, "m2m100.lang_codes");
        int iidx = gguf_find_key(g, "m2m100.lang_token_ids");
        if (cidx >= 0 && iidx >= 0) {
            int n = gguf_get_arr_n(g, cidx);
            c->tokenizer.lang_codes.resize(n);
            for (int i = 0; i < n; i++) {
                std::string code = gguf_get_arr_str(g, cidx, i);
                c->tokenizer.lang_codes[i] = code;
                // Read corresponding token ID
                uint32_t tok_id = ((const uint32_t*)gguf_get_arr_data(g, iidx))[i];
                c->tokenizer.lang_to_token_id[code] = (int)tok_id;
            }
        }
    }
    return true;
}

// ── Load metadata: Marian ────────────────────────────────────────
// Every key is required. A Marian GGUF with a tokenizer table missing would
// still load and translate — into text segmented by some other rule — so a
// missing or mistyped key fails the load instead.

static bool load_marian_spm(gguf_context* g, const char* side, core_marian_tok::Normalizer& norm,
                            core_marian_tok::Unigram& uni) {
    auto key = [&](const char* leaf) { return std::string("tokenizer.marian.") + side + "." + leaf; };
    auto fail = [&](const char* leaf, const char* why) {
        fprintf(stderr, "marian: GGUF key '%s' %s\n", key(leaf).c_str(), why);
        return false;
    };
    const int pidx = gguf_find_key(g, key("pieces").c_str());
    const int sidx = gguf_find_key(g, key("scores").c_str());
    const int tidx = gguf_find_key(g, key("types").c_str());
    if (pidx < 0 || gguf_get_kv_type(g, pidx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, pidx) != GGUF_TYPE_STRING)
        return fail("pieces", "is missing or not a string array");
    if (sidx < 0 || gguf_get_kv_type(g, sidx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, sidx) != GGUF_TYPE_FLOAT32)
        return fail("scores", "is missing or not a float32 array");
    if (tidx < 0 || gguf_get_kv_type(g, tidx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, tidx) != GGUF_TYPE_INT32)
        return fail("types", "is missing or not an int32 array");
    const size_t n = gguf_get_arr_n(g, pidx);
    if (n == 0 || gguf_get_arr_n(g, sidx) != n || gguf_get_arr_n(g, tidx) != n)
        return fail("pieces", "/ scores / types differ in length");
    std::vector<std::string> pieces(n);
    for (size_t i = 0; i < n; i++)
        pieces[i] = gguf_get_arr_str(g, pidx, i);
    const float* sp = (const float*)gguf_get_arr_data(g, sidx);
    const int32_t* tp = (const int32_t*)gguf_get_arr_data(g, tidx);
    uni.load(pieces, std::vector<float>(sp, sp + n), std::vector<int32_t>(tp, tp + n));

    // The charsmap is absent only for an "identity" normalizer; the converter
    // records which it was, so absence cannot pass for it by accident.
    const int nidx = gguf_find_key(g, key("normalizer").c_str());
    if (nidx < 0 || gguf_get_kv_type(g, nidx) != GGUF_TYPE_STRING)
        return fail("normalizer", "is missing");
    const std::string norm_name = gguf_get_val_str(g, nidx);
    const int cidx = gguf_find_key(g, key("charsmap").c_str());
    if (cidx >= 0) {
        if (gguf_get_kv_type(g, cidx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, cidx) != GGUF_TYPE_UINT8)
            return fail("charsmap", "is not a uint8 array");
        if (!norm.load_charsmap((const uint8_t*)gguf_get_arr_data(g, cidx), gguf_get_arr_n(g, cidx)))
            return fail("charsmap", "is not a SentencePiece precompiled charsmap");
    } else if (norm_name != "identity") {
        return fail("charsmap", "is missing although the normalizer is not 'identity'");
    }
    struct {
        const char* leaf;
        bool* dst;
    } flags[] = {{"add_dummy_prefix", &norm.add_dummy_prefix},
                 {"remove_extra_whitespaces", &norm.remove_extra_whitespaces},
                 {"escape_whitespaces", &norm.escape_whitespaces}};
    for (auto& f : flags) {
        const int idx = gguf_find_key(g, key(f.leaf).c_str());
        if (idx < 0 || gguf_get_kv_type(g, idx) != GGUF_TYPE_BOOL)
            return fail(f.leaf, "is missing or not a bool");
        *f.dst = gguf_get_val_bool(g, idx);
    }
    return true;
}

static bool load_marian_metadata(m2m100_context* c, gguf_context* g) {
    auto& hp = c->model.hp;
    bool ok = true;
    auto req_u32 = [&](const char* key) -> int {
        const int idx = gguf_find_key(g, key);
        if (idx < 0 || gguf_get_kv_type(g, idx) != GGUF_TYPE_UINT32) {
            fprintf(stderr, "marian: GGUF key '%s' is missing\n", key);
            ok = false;
            return 0;
        }
        return (int)gguf_get_val_u32(g, idx);
    };
    auto get_str = [&](const char* key) -> std::string {
        const int idx = gguf_find_key(g, key);
        return (idx >= 0 && gguf_get_kv_type(g, idx) == GGUF_TYPE_STRING) ? gguf_get_val_str(g, idx) : "";
    };
    hp.marian = true;
    hp.pos_offset = 0;
    hp.bos_token_id = -1; // Marian has no <s>
    hp.vocab_size = req_u32("marian.vocab_size");
    hp.d_model = req_u32("marian.d_model");
    hp.enc_n_layers = req_u32("marian.encoder.n_layers");
    hp.enc_n_heads = req_u32("marian.encoder.n_heads");
    hp.enc_ffn_dim = req_u32("marian.encoder.ffn_dim");
    hp.dec_n_layers = req_u32("marian.decoder.n_layers");
    hp.dec_n_heads = req_u32("marian.decoder.n_heads");
    hp.dec_ffn_dim = req_u32("marian.decoder.ffn_dim");
    hp.max_position_emb = req_u32("marian.max_position_embeddings");
    hp.scale_embedding = req_u32("marian.scale_embedding") != 0;
    hp.pre_norm = req_u32("marian.normalize_before") != 0;
    hp.eos_token_id = req_u32("marian.eos_token_id");
    hp.pad_token_id = req_u32("marian.pad_token_id");
    hp.unk_token_id = req_u32("marian.unk_token_id");
    hp.dec_start_token = req_u32("marian.decoder_start_token_id");
    hp.gen_num_beams = req_u32("marian.gen.num_beams");
    hp.gen_max_length = req_u32("marian.gen.max_length");
    hp.early_stopping = req_u32("marian.gen.early_stopping") != 0;
    hp.source_lang = get_str("marian.source_lang");
    hp.target_lang = get_str("marian.target_lang");
    if (!ok)
        return false;
    if (hp.d_model <= 0 || hp.enc_n_heads <= 0 || hp.enc_n_heads != hp.dec_n_heads ||
        hp.d_model % hp.enc_n_heads != 0 || hp.vocab_size <= 0 || hp.max_position_emb <= 0 || hp.enc_n_layers <= 0 ||
        hp.dec_n_layers <= 0) {
        fprintf(stderr, "marian: implausible hyperparameters in the GGUF\n");
        return false;
    }

    const std::string act = get_str("marian.activation_function");
    if (act == "swish" || act == "silu") {
        hp.activation = m2m100_hparams::ACT_SILU;
    } else if (act == "relu") {
        hp.activation = m2m100_hparams::ACT_RELU;
    } else if (act == "gelu") {
        hp.activation = m2m100_hparams::ACT_GELU;
    } else {
        fprintf(stderr, "marian: activation_function '%s' is not implemented\n", act.c_str());
        return false;
    }

    {
        const int idx = gguf_find_key(g, "marian.suppress_token_ids");
        if (idx >= 0) {
            if (gguf_get_kv_type(g, idx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, idx) != GGUF_TYPE_INT32) {
                fprintf(stderr, "marian: GGUF key 'marian.suppress_token_ids' is not an int32 array\n");
                return false;
            }
            const int32_t* d = (const int32_t*)gguf_get_arr_data(g, idx);
            for (size_t i = 0; i < gguf_get_arr_n(g, idx); i++)
                if (d[i] >= 0 && d[i] < hp.vocab_size)
                    hp.suppress_ids.push_back(d[i]);
        }
    }

    // vocab.json: the id space of the embedding table
    const int tidx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tidx < 0 || gguf_get_kv_type(g, tidx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, tidx) != GGUF_TYPE_STRING ||
        (int)gguf_get_arr_n(g, tidx) != hp.vocab_size) {
        fprintf(stderr, "marian: 'tokenizer.ggml.tokens' is missing or does not have vocab_size (%d) entries\n",
                hp.vocab_size);
        return false;
    }
    auto& mt = c->marian_tok;
    c->tokenizer.id_to_token.resize(hp.vocab_size);
    mt.vocab.reserve((size_t)hp.vocab_size * 2);
    for (int i = 0; i < hp.vocab_size; i++) {
        c->tokenizer.id_to_token[i] = gguf_get_arr_str(g, tidx, i);
        mt.vocab.emplace(c->tokenizer.id_to_token[i], i);
    }
    mt.unk_id = hp.unk_token_id;
    mt.eos_id = hp.eos_token_id;
    if (hp.eos_token_id >= hp.vocab_size || hp.pad_token_id >= hp.vocab_size || hp.unk_token_id >= hp.vocab_size ||
        hp.dec_start_token >= hp.vocab_size) {
        fprintf(stderr, "marian: a special token id is outside the vocabulary\n");
        return false;
    }
    if (!load_marian_spm(g, "source", mt.norm, mt.src))
        return false;

    for (const std::string& l : {hp.source_lang, hp.target_lang})
        if (!l.empty())
            c->tokenizer.lang_codes.push_back(l);
    return true;
}

// ── Bind tensors ─────────────────────────────────────────────────

static bool bind_model(m2m100_context* c) {
    auto& m = c->model;
    const auto& hp = m.hp;

    m.shared_embed = TR(c, "shared.embed.weight");
    m.enc_pos_emb = TR(c, "enc.pos_emb");
    m.dec_pos_emb = TR(c, "dec.pos_emb");
    if (!m.shared_embed || !m.enc_pos_emb || !m.dec_pos_emb)
        return false;
    if (hp.marian) {
        m.final_logits_bias = TR(c, "final_logits_bias");
        if (!m.final_logits_bias)
            return false;
    }

    // Encoder layers
    m.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        auto& l = m.enc_layers[i];
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%d.%s", i, suffix);
            return T(c, buf);
        };
        l.attn_q_w = w("attn_q.weight");
        l.attn_q_b = w("attn_q.bias");
        l.attn_k_w = w("attn_k.weight");
        l.attn_k_b = w("attn_k.bias");
        l.attn_v_w = w("attn_v.weight");
        l.attn_v_b = w("attn_v.bias");
        l.attn_o_w = w("attn_o.weight");
        l.attn_o_b = w("attn_o.bias");
        l.attn_ln_w = w("attn_ln.weight");
        l.attn_ln_b = w("attn_ln.bias");
        l.ffn_up_w = w("ffn_up.weight");
        l.ffn_up_b = w("ffn_up.bias");
        l.ffn_down_w = w("ffn_down.weight");
        l.ffn_down_b = w("ffn_down.bias");
        l.ffn_ln_w = w("ffn_ln.weight");
        l.ffn_ln_b = w("ffn_ln.bias");
    }
    // Marian has no output LayerNorm on either stack.
    m.enc_out_ln_w = hp.marian ? nullptr : TR(c, "enc.out_ln.weight");
    m.enc_out_ln_b = hp.marian ? nullptr : TR(c, "enc.out_ln.bias");

    // Decoder layers
    m.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; i++) {
        auto& l = m.dec_layers[i];
        char buf[128];
        auto w = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "dec.blk.%d.%s", i, suffix);
            return T(c, buf);
        };
        l.attn_q_w = w("attn_q.weight");
        l.attn_q_b = w("attn_q.bias");
        l.attn_k_w = w("attn_k.weight");
        l.attn_k_b = w("attn_k.bias");
        l.attn_v_w = w("attn_v.weight");
        l.attn_v_b = w("attn_v.bias");
        l.attn_o_w = w("attn_o.weight");
        l.attn_o_b = w("attn_o.bias");
        l.attn_ln_w = w("attn_ln.weight");
        l.attn_ln_b = w("attn_ln.bias");
        l.cross_q_w = w("cross_q.weight");
        l.cross_q_b = w("cross_q.bias");
        l.cross_k_w = w("cross_k.weight");
        l.cross_k_b = w("cross_k.bias");
        l.cross_v_w = w("cross_v.weight");
        l.cross_v_b = w("cross_v.bias");
        l.cross_o_w = w("cross_o.weight");
        l.cross_o_b = w("cross_o.bias");
        l.cross_ln_w = w("cross_ln.weight");
        l.cross_ln_b = w("cross_ln.bias");
        l.ffn_up_w = w("ffn_up.weight");
        l.ffn_up_b = w("ffn_up.bias");
        l.ffn_down_w = w("ffn_down.weight");
        l.ffn_down_b = w("ffn_down.bias");
        l.ffn_ln_w = w("ffn_ln.weight");
        l.ffn_ln_b = w("ffn_ln.bias");
    }
    m.dec_out_ln_w = hp.marian ? nullptr : TR(c, "dec.out_ln.weight");
    m.dec_out_ln_b = hp.marian ? nullptr : TR(c, "dec.out_ln.bias");

    if (hp.marian) {
        // The M2M-100 path binds its layer tensors unchecked and has for a
        // long time; a new converter does not get that benefit of the doubt.
        for (const auto& stack : {std::make_pair("enc", hp.enc_n_layers), std::make_pair("dec", hp.dec_n_layers)}) {
            const bool dec = stack.first[0] == 'd';
            for (int i = 0; i < stack.second; i++) {
                for (const char* part : {"attn_q", "attn_k", "attn_v", "attn_o", "attn_ln", "cross_q", "cross_k",
                                         "cross_v", "cross_o", "cross_ln", "ffn_up", "ffn_down", "ffn_ln"}) {
                    if (!dec && std::strncmp(part, "cross", 5) == 0)
                        continue;
                    for (const char* wb : {"weight", "bias"}) {
                        char name[96];
                        snprintf(name, sizeof(name), "%s.blk.%d.%s.%s", stack.first, i, part, wb);
                        if (!TR(c, name))
                            return false;
                    }
                }
            }
        }
        if ((int)m.enc_pos_emb->ne[1] < hp.max_position_emb || (int)m.dec_pos_emb->ne[1] < hp.max_position_emb ||
            (int)m.enc_pos_emb->ne[0] != hp.d_model || (int)m.shared_embed->ne[1] != hp.vocab_size ||
            (int)m.shared_embed->ne[0] != hp.d_model || (int)m.final_logits_bias->ne[0] != hp.vocab_size) {
            fprintf(stderr, "marian: tensor shapes do not match the GGUF hyperparameters\n");
            return false;
        }
    }

    return true;
}

// ── Tokenizer encode ─────────────────────────────────────────────
// M2M-100 uses a SentencePiece BPE model: tokenization follows merge order
// (highest score = lowest rank, merged first), via core_spm::tokenize_bpe over the
// GGUF's tokenizer.ggml.scores. Falls back to greedy longest-match if scores absent.

static std::vector<int> tokenize(const m2m100_tokenizer& tok, const std::string& text, const std::string& src_lang) {
    std::vector<int> ids;

    // Prepend source language token
    auto it = tok.lang_to_token_id.find(src_lang);
    if (it != tok.lang_to_token_id.end()) {
        ids.push_back(it->second);
    }

    if (!tok.scores.empty() && !tok.token_to_id_u.empty()) {
        core_spm::Config cfg;
        cfg.unk_id = 3; // <unk>
        auto bpe = core_spm::tokenize_bpe(text, tok.token_to_id_u, tok.scores, cfg, /*prepend_space=*/true);
        for (int id : bpe) {
            // BPE-intermediate pieces (id >= embed_vocab_size) have no embedding
            // row and should never be final — if one leaks out, map it to <unk>
            // (matches HF M2M100Tokenizer.convert_token_to_id).
            ids.push_back((tok.embed_vocab_size > 0 && id >= tok.embed_vocab_size) ? 3 : id);
        }
        ids.push_back(2); // </s>
        return ids;
    }

    // Fallback: SentencePiece-style greedy longest-match tokenization with ▁ prefix
    // Split on whitespace, prefix each word with ▁, greedy longest match
    std::vector<std::string> words;
    std::string cur;
    for (char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur += ch;
        }
    }
    if (!cur.empty())
        words.push_back(cur);

    for (size_t wi = 0; wi < words.size(); wi++) {
        // Prefix with ▁ (U+2581, 0xE2 0x96 0x81)
        std::string word = "\xE2\x96\x81" + words[wi];
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int best_id = -1;
            while (end > start) {
                std::string sub = word.substr(start, end - start);
                auto it2 = tok.token_to_id.find(sub);
                if (it2 != tok.token_to_id.end()) {
                    best_id = it2->second;
                    break;
                }
                end--;
                // Don't break mid-UTF8
                while (end > start && (word[end] & 0xC0) == 0x80)
                    end--;
            }
            if (best_id < 0) {
                ids.push_back(3); // <unk>
                break;
            }
            ids.push_back(best_id);
            start = end;
        }
    }

    // Append EOS
    ids.push_back(2); // </s>

    return ids;
}

static std::string detokenize(const m2m100_tokenizer& tok, const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        if (id < 0 || id >= (int)tok.id_to_token.size())
            continue;
        const std::string& t = tok.id_to_token[id];
        if (t == "<s>" || t == "</s>" || t == "<pad>" || t == "<unk>")
            continue;
        // Skip language tokens (__XX__)
        if (t.size() >= 5 && t[0] == '_' && t[1] == '_' && t[t.size() - 1] == '_' && t[t.size() - 2] == '_')
            continue;
        // SentencePiece: '▁' → space
        std::string decoded = t;
        size_t pos = 0;
        while ((pos = decoded.find("\xe2\x96\x81", pos)) != std::string::npos) {
            decoded.replace(pos, 3, " ");
            pos += 1;
        }
        result += decoded;
    }
    // Trim leading space
    if (!result.empty() && result[0] == ' ')
        result = result.substr(1);
    return result;
}

// ── Marian tokenizer glue ────────────────────────────────────────

// Multi-target Marian checkpoints select the output language with a leading
// `>>xxx<<` token. A single-pair model has no such token and this is a no-op.
static std::string marian_with_target_code(const m2m100_context* c, const std::string& text,
                                           const std::string& tgt_lang) {
    if (tgt_lang.empty() || core_marian_tok::Tokenizer::language_code_len(text) > 0)
        return text;
    const std::string code = ">>" + tgt_lang + "<<";
    if (c->marian_tok.vocab.find(code) == c->marian_tok.vocab.end())
        return text;
    return code + " " + text;
}

static std::string marian_detokenize(const m2m100_context* c, const std::vector<int>& ids) {
    const auto& hp = c->model.hp;
    std::vector<std::string> pieces;
    pieces.reserve(ids.size());
    for (int id : ids) {
        if (id < 0 || id >= (int)c->tokenizer.id_to_token.size())
            continue;
        if (id == hp.eos_token_id || id == hp.pad_token_id || id == hp.unk_token_id)
            continue;
        pieces.push_back(c->tokenizer.id_to_token[id]);
    }
    return core_marian_tok::join_pieces(pieces);
}

// A single-pair model translates one direction whatever it is asked for; say
// so once instead of answering a de→en request with an en→de model silently.
static void marian_check_languages(m2m100_context* c, const char* src_lang, const char* tgt_lang) {
    if (c->warned_lang)
        return;
    const auto& hp = c->model.hp;
    auto differs = [](const std::string& model, const char* asked) {
        return model.size() == 2 && asked && std::strlen(asked) == 2 && model != asked;
    };
    if (differs(hp.source_lang, src_lang) || differs(hp.target_lang, tgt_lang)) {
        c->warned_lang = true;
        fprintf(stderr, "marian: this model translates %s→%s; the requested %s→%s is ignored\n", hp.source_lang.c_str(),
                hp.target_lang.c_str(), src_lang, tgt_lang);
    }
}

// ── Layer pieces shared by both graphs ───────────────────────────

static ggml_tensor* layer_norm(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    x = ggml_norm(ctx0, x, 1e-5f);
    x = ggml_mul(ctx0, x, w);
    return ggml_add(ctx0, x, b);
}

static ggml_tensor* ffn_activation(ggml_context* ctx0, ggml_tensor* x, int activation) {
    switch (activation) {
    case m2m100_hparams::ACT_SILU:
        return ggml_silu(ctx0, x);
    case m2m100_hparams::ACT_GELU:
        return ggml_gelu_erf(ctx0, x);
    default:
        return ggml_relu(ctx0, x);
    }
}

// ── KV cache allocation ──────────────────────────────────────────

static bool alloc_kv_cache(m2m100_context* c, int max_ctx) {
    const auto& hp = c->model.hp;
    const int hd = hp.head_dim();
    const int nh = hp.dec_n_heads;
    const int nl = hp.dec_n_layers;

    // One translate call used to allocate a cache and never free the previous
    // one (only the last was released, in m2m100_free) — ~10 MB leaked per
    // sentence on the 418M, which a live session calls for every sentence and
    // every draft. Reuse the buffer when it is big enough; cleared, as a fresh
    // one was.
    if (c->kv_buf && c->kv_max_ctx >= max_ctx) {
        ggml_backend_buffer_clear(c->kv_buf, 0);
        return true;
    }
    if (c->kv_buf) {
        ggml_backend_buffer_free(c->kv_buf);
        c->kv_buf = nullptr;
    }
    if (c->kv_ctx) {
        ggml_free(c->kv_ctx);
        c->kv_ctx = nullptr;
    }
    c->kv_k = c->kv_v = nullptr;
    c->kv_max_ctx = 0;

    size_t n_tensors = 2; // k, v
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + 64;
    ggml_init_params params = {ctx_size, nullptr, true};
    c->kv_ctx = ggml_init(params);

    // Shape: (head_dim, max_ctx, n_heads, n_layers)
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, nh, nl);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");

    c->kv_buf = ggml_backend_alloc_ctx_tensors(c->kv_ctx, c->backend);
    if (!c->kv_buf)
        return false;

    // Zero-init
    ggml_backend_buffer_clear(c->kv_buf, 0);
    c->kv_max_ctx = max_ctx;
    return true;
}

// ── Cross-attention KV cache ─────────────────────────────────────

static bool alloc_cross_kv(m2m100_context* c, int T_enc) {
    const auto& hp = c->model.hp;
    const int hd = hp.head_dim();
    const int nh = hp.dec_n_heads;
    const int nl = hp.dec_n_layers;

    // Free existing
    if (c->cross_kv_buf) {
        ggml_backend_buffer_free(c->cross_kv_buf);
        c->cross_kv_buf = nullptr;
    }
    if (c->cross_kv_ctx) {
        ggml_free(c->cross_kv_ctx);
        c->cross_kv_ctx = nullptr;
    }

    size_t ctx_size = ggml_tensor_overhead() * nl * 2 + 64;
    ggml_init_params params = {ctx_size, nullptr, true};
    c->cross_kv_ctx = ggml_init(params);

    c->cross_kv_k.resize(nl);
    c->cross_kv_v.resize(nl);
    for (int i = 0; i < nl; i++) {
        c->cross_kv_k[i] = ggml_new_tensor_3d(c->cross_kv_ctx, GGML_TYPE_F16, hd, T_enc, nh);
        c->cross_kv_v[i] = ggml_new_tensor_3d(c->cross_kv_ctx, GGML_TYPE_F16, hd, T_enc, nh);
        char name[64];
        snprintf(name, sizeof(name), "cross_k_%d", i);
        ggml_set_name(c->cross_kv_k[i], name);
        snprintf(name, sizeof(name), "cross_v_%d", i);
        ggml_set_name(c->cross_kv_v[i], name);
    }

    c->cross_kv_buf = ggml_backend_alloc_ctx_tensors(c->cross_kv_ctx, c->backend);
    if (!c->cross_kv_buf)
        return false;

    c->cross_T_enc = T_enc;
    return true;
}

// ── Encoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_encoder_graph(m2m100_context* c, int T) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.enc_n_heads;
    const int hd = hp.head_dim();
    const float scale = 1.0f / sqrtf((float)hd);
    const float emb_scale = hp.scale_embedding ? sqrtf((float)D) : 1.0f;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input token IDs
    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(inp, "enc_tokens");
    ggml_set_input(inp);

    // Position IDs
    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(pos, "enc_positions");
    ggml_set_input(pos);

    // Embedding: shared_embed[tokens] * scale + pos_emb[positions]
    ggml_tensor* cur = ggml_get_rows(ctx0, m.shared_embed, inp);
    if (emb_scale != 1.0f) {
        cur = ggml_scale(ctx0, cur, emb_scale);
    }
    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, m.enc_pos_emb, pos));

    // Encoder layers
    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& l = m.enc_layers[il];
        ggml_tensor* residual = cur;

        // Pre-norm for self-attention (M2M-100); Marian normalizes after the
        // residual instead.
        if (hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.attn_ln_w, l.attn_ln_b);

        // Self-attention Q, K, V
        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_q_w, cur), l.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_k_w, cur), l.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_v_w, cur), l.attn_v_b);

        // Reshape for multi-head: (D, T) → (hd, nh, T) → permute to (hd, T, nh)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3));

        // Flash attention (no causal mask for encoder)
        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, D, T);

        // Output projection + residual
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_o_w, attn), l.attn_o_b);
        cur = ggml_add(ctx0, cur, residual);
        if (!hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.attn_ln_w, l.attn_ln_b);

        // FFN
        residual = cur;
        if (hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.ffn_ln_w, l.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_up_w, cur), l.ffn_up_b);
        cur = ffn_activation(ctx0, cur, hp.activation);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_down_w, cur), l.ffn_down_b);

        cur = ggml_add(ctx0, cur, residual);
        if (!hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.ffn_ln_w, l.ffn_ln_b);
    }

    // Final encoder LayerNorm (absent in Marian)
    if (m.enc_out_ln_w)
        cur = layer_norm(ctx0, cur, m.enc_out_ln_w, m.enc_out_ln_b);

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// ── Compute cross-attention K, V from encoder output ─────────────

static bool compute_cross_kv(m2m100_context* c, const float* enc_out, int T_enc) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.dec_n_heads;
    const int hd = hp.head_dim();

    if (!alloc_cross_kv(c, T_enc))
        return false;

    // Build a ggml graph to compute cross-K and cross-V for all layers.
    // This handles F16/quantized weights natively via ggml_mul_mat.
    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& l = m.dec_layers[il];

        ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
        ggml_context* ctx0 = ggml_init(ip);
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

        ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, T_enc);
        ggml_set_name(enc_inp, "enc_for_cross");
        ggml_set_input(enc_inp);

        // K = W_k @ enc + b_k → (D, T_enc)
        ggml_tensor* K = ggml_mul_mat(ctx0, l.cross_k_w, enc_inp);
        if (l.cross_k_b)
            K = ggml_add(ctx0, K, l.cross_k_b);
        // Reshape to (hd, nh, T_enc) then permute to (hd, T_enc, nh)
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(K, "cross_k");

        // V = W_v @ enc + b_v
        ggml_tensor* V = ggml_mul_mat(ctx0, l.cross_v_w, enc_inp);
        if (l.cross_v_b)
            V = ggml_add(ctx0, V, l.cross_v_b);
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T_enc), 0, 2, 1, 3));
        ggml_set_name(V, "cross_v");

        ggml_build_forward_expand(gf, K);
        ggml_build_forward_expand(gf, V);

        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
            ggml_free(ctx0);
            return false;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_for_cross"), enc_out, 0,
                                (size_t)D * T_enc * sizeof(float));

        if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx0);
            return false;
        }

        // Copy results to cross_kv tensors
        ggml_tensor* K_out = ggml_graph_get_tensor(gf, "cross_k");
        ggml_tensor* V_out = ggml_graph_get_tensor(gf, "cross_v");
        const size_t n_elem = (size_t)hd * T_enc * nh;
        std::vector<float> buf(n_elem);
        std::vector<ggml_fp16_t> buf16(n_elem);
        ggml_backend_tensor_get(K_out, buf.data(), 0, n_elem * sizeof(float));
        ggml_fp32_to_fp16_row(buf.data(), buf16.data(), (int)n_elem);
        ggml_backend_tensor_set(c->cross_kv_k[il], buf16.data(), 0, n_elem * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(V_out, buf.data(), 0, n_elem * sizeof(float));
        ggml_fp32_to_fp16_row(buf.data(), buf16.data(), (int)n_elem);
        ggml_backend_tensor_set(c->cross_kv_v[il], buf16.data(), 0, n_elem * sizeof(ggml_fp16_t));

        ggml_free(ctx0);
    }

    return true;
}

// ── Decoder graph ────────────────────────────────────────────────

static ggml_cgraph* build_decoder_graph(m2m100_context* c, int n_tokens, int offset) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int D = hp.d_model;
    const int nh = hp.dec_n_heads;
    const int hd = hp.head_dim();
    const float attn_scale = 1.0f / sqrtf((float)hd);
    const float emb_scale = hp.scale_embedding ? sqrtf((float)D) : 1.0f;
    const int Lk = offset + n_tokens;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp, "dec_tokens");
    ggml_set_input(inp);

    ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(pos, "dec_positions");
    ggml_set_input(pos);

    // Causal mask for self-attention (only when n_tokens > 1)
    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    // Embedding
    ggml_tensor* cur = ggml_get_rows(ctx0, m.shared_embed, inp);
    if (emb_scale != 1.0f) {
        cur = ggml_scale(ctx0, cur, emb_scale);
    }
    cur = ggml_add(ctx0, cur, ggml_get_rows(ctx0, m.dec_pos_emb, pos));

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& l = m.dec_layers[il];
        ggml_tensor* residual = cur;

        // ---- Self-attention ----
        if (hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.attn_ln_w, l.attn_ln_b);

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_q_w, cur), l.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_k_w, cur), l.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_v_w, cur), l.attn_v_b);

        // Reshape + permute to (hd, n_tokens, nh)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, n_tokens), 0, 2, 1, 3));
        ggml_tensor* K_new =
            ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, n_tokens), 0, 2, 1, 3));
        ggml_tensor* V_new =
            ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, n_tokens), 0, 2, 1, 3));

        // Write to KV cache
        ggml_tensor* k_view =
            ggml_view_4d(ctx0, c->kv_k, hd, n_tokens, nh, 1, c->kv_k->nb[1], c->kv_k->nb[2], c->kv_k->nb[3],
                         (size_t)il * c->kv_k->nb[3] + (size_t)offset * c->kv_k->nb[1]);
        ggml_tensor* v_view =
            ggml_view_4d(ctx0, c->kv_v, hd, n_tokens, nh, 1, c->kv_v->nb[1], c->kv_v->nb[2], c->kv_v->nb[3],
                         (size_t)il * c->kv_v->nb[3] + (size_t)offset * c->kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new, v_view));

        // Read full KV history
        ggml_tensor* Kfull =
            ggml_view_3d(ctx0, c->kv_k, hd, Lk, nh, c->kv_k->nb[1], c->kv_k->nb[2], (size_t)il * c->kv_k->nb[3]);
        ggml_tensor* Vfull =
            ggml_view_3d(ctx0, c->kv_v, hd, Lk, nh, c->kv_v->nb[1], c->kv_v->nb[2], (size_t)il * c->kv_v->nb[3]);
        Kfull = ggml_cont(ctx0, Kfull);
        Vfull = ggml_cont(ctx0, Vfull);

        // Flash attention with causal mask
        ggml_tensor* sa_out =
            ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, (n_tokens == 1) ? nullptr : causal_mask, attn_scale, 0.0f, 0.0f);
        cur = ggml_reshape_2d(ctx0, sa_out, D, n_tokens);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.attn_o_w, cur), l.attn_o_b);
        cur = ggml_add(ctx0, cur, residual);
        if (!hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.attn_ln_w, l.attn_ln_b);

        // ---- Cross-attention ----
        residual = cur;
        if (hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.cross_ln_w, l.cross_ln_b);

        ggml_tensor* CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, l.cross_q_w, cur), l.cross_q_b);
        CQ = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, CQ, hd, nh, n_tokens), 0, 2, 1, 3));

        ggml_tensor* CK = c->cross_kv_k[il];
        ggml_tensor* CV = c->cross_kv_v[il];

        ggml_tensor* ca_out = ggml_flash_attn_ext(ctx0, CQ, CK, CV, nullptr, attn_scale, 0.0f, 0.0f);
        cur = ggml_reshape_2d(ctx0, ca_out, D, n_tokens);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.cross_o_w, cur), l.cross_o_b);
        cur = ggml_add(ctx0, cur, residual);
        if (!hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.cross_ln_w, l.cross_ln_b);

        // ---- FFN ----
        residual = cur;
        if (hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.ffn_ln_w, l.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_up_w, cur), l.ffn_up_b);
        cur = ffn_activation(ctx0, cur, hp.activation);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, l.ffn_down_w, cur), l.ffn_down_b);
        cur = ggml_add(ctx0, cur, residual);
        if (!hp.pre_norm)
            cur = layer_norm(ctx0, cur, l.ffn_ln_w, l.ffn_ln_b);
    }

    // Final LayerNorm (absent in Marian)
    if (m.dec_out_ln_w)
        cur = layer_norm(ctx0, cur, m.dec_out_ln_w, m.dec_out_ln_b);

    // Take last token
    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx0, cur, D, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    }

    // LM head (tied with shared_embed)
    cur = ggml_mul_mat(ctx0, m.shared_embed, cur);
    if (m.final_logits_bias)
        cur = ggml_add(ctx0, cur, m.final_logits_bias);

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}

// ── Run encoder ──────────────────────────────────────────────────

static std::vector<float> run_encoder(m2m100_context* c, const std::vector<int>& token_ids) {
    const int T = (int)token_ids.size();
    const int D = c->model.hp.d_model;

    ggml_cgraph* gf = build_encoder_graph(c, T);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "m2m100: failed to alloc encoder graph\n");
        return {};
    }

    // Set input tokens
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_tokens"), token_ids.data(), 0, T * sizeof(int32_t));

    // Set position IDs: offset=2 for M2M-100 (padding_idx=1, first real pos=2),
    // 0 for Marian.
    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; i++)
        positions[i] = i + c->model.hp.pos_offset;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "enc_positions"), positions.data(), 0, T * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "m2m100: encoder compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    std::vector<float> enc_out(T * D);
    ggml_backend_tensor_get(out, enc_out.data(), 0, enc_out.size() * sizeof(float));
    return enc_out;
}

// ── Run decoder step ─────────────────────────────────────────────

static std::vector<float> run_decoder_step(m2m100_context* c, const int* tokens, int n_tokens, int offset) {
    const auto& hp = c->model.hp;
    const int vocab = hp.vocab_size;
    const int Lk = offset + n_tokens;

    const bool bench = m2m100_bench_enabled();
    double tb0 = bench ? m2m100_now_ms() : 0;
    ggml_cgraph* gf = build_decoder_graph(c, n_tokens, offset);
    double tb1 = bench ? m2m100_now_ms() : 0;
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "m2m100: failed to alloc decoder graph\n");
        return {};
    }
    double tb2 = bench ? m2m100_now_ms() : 0;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_tokens"), tokens, 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = offset + i + hp.pos_offset; // M2M-100: 2, Marian: 0
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_positions"), positions.data(), 0,
                            n_tokens * sizeof(int32_t));

    // Causal mask
    if (n_tokens > 1) {
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens, ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = offset + q + 1; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = neg_inf;
            }
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "m2m100: decoder compute failed\n");
        return {};
    }
    double tb3 = bench ? m2m100_now_ms() : 0;

    ggml_tensor* logits = ggml_graph_get_tensor(gf, "logits");
    std::vector<float> out(vocab);
    ggml_backend_tensor_get(logits, out.data(), 0, vocab * sizeof(float));
    if (bench) {
        auto& b = g_m2m100_step_bench;
        b.build_ms += tb1 - tb0;
        b.alloc_ms += tb2 - tb1;
        b.compute_ms += tb3 - tb2;
        b.read_ms += m2m100_now_ms() - tb3;
        b.steps++;
    }
    // generation_config bad_words_ids: never generated, in greedy and in beam
    // (Marian: <pad>, which is also the decoder start token).
    for (int id : hp.suppress_ids)
        out[id] = -std::numeric_limits<float>::infinity();
    return out;
}

// ── Public API ───────────────────────────────────────────────────

extern "C" struct m2m100_context_params m2m100_context_default_params(void) {
    m2m100_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true; // §232: GPU allowed by default; the is_metal gate + env decide actual use
    return p;
}

extern "C" int m2m100_default_beam_size(void) {
    // facebook/m2m100_418M and both WMT21 dense directions declare
    // num_beams=5. Greedy is an explicit opt-in for this model family (#439).
    return 5;
}

extern "C" struct m2m100_context* m2m100_init_from_file(const char* path_model, struct m2m100_context_params params) {
    auto* c = new m2m100_context();
    c->params = params;

    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            delete c;
            return nullptr;
        }
        const bool meta_ok = load_metadata(c, g);
        core_gguf::free_metadata(g);
        if (!meta_ok) {
            delete c;
            return nullptr;
        }
    }

    const auto& hp = c->model.hp;
    if (hp.marian) {
        // generation_config.json num_beams (4 for Opus-MT); M2M-100's 5 otherwise.
        c->beam_size = hp.gen_num_beams > 1 ? hp.gen_num_beams : 1;
        if (params.verbosity >= 1) {
            fprintf(stderr, "marian: %s→%s d=%d enc=%dL dec=%dL heads=%d ffn=%d vocab=%d\n",
                    hp.source_lang.empty() ? "?" : hp.source_lang.c_str(),
                    hp.target_lang.empty() ? "?" : hp.target_lang.c_str(), hp.d_model, hp.enc_n_layers, hp.dec_n_layers,
                    hp.enc_n_heads, hp.enc_ffn_dim, hp.vocab_size);
        }
    } else if (params.verbosity >= 1) {
        fprintf(stderr, "m2m100: d=%d enc=%dL dec=%dL heads=%d ffn=%d vocab=%d langs=%d\n", hp.d_model, hp.enc_n_layers,
                hp.dec_n_layers, hp.enc_n_heads, hp.enc_ffn_dim, hp.vocab_size, (int)c->tokenizer.lang_codes.size());
    }

    // Backend selection (§232). m2m100 loads weights + KV onto c->backend via
    // core_gguf::load_weights + ggml_backend_alloc_ctx_tensors, so picking a GPU
    // backend is the whole change.
    //   * CRISPASR_M2M100_GPU=1 forces GPU on ANY backend; =0 forces CPU.
    //   * default: GPU on CUDA/Vulkan (Kaggle P100 A/B: identical en->de
    //     output, 1.24x wall against a slow OpenBLAS baseline).
    //   * Metal, m2m100: GPU. An earlier M1 reading called it neutral; in
    //     interleaved pairs at load 4-5 (2026-10-06, 418M q8_0, warm, tokens
    //     identical) the GPU is ~20% faster alone (median 134-140 vs 158-186
    //     ms, 3/3) and ~9% faster inside the live pipeline, where it shares
    //     the GPU with the recogniser (summed 9.67 vs 10.59 s over 4 pairs,
    //     3/4 in its favour). Under CPU contention the gap widens.
    //   * Metal, Marian / Opus-MT: CPU. Alone the GPU is slightly ahead, but
    //     in the live pipeline it is ~2x SLOWER (median 98-118 vs 38-69 ms,
    //     3/3): a decoder step is ~4 ms of CPU work, less than the wait
    //     behind the recogniser's GPU work.
    c->backend_cpu = core_cpu_backend::init();
    // Honour the caller's thread count. It used to be parsed, stored and never
    // applied, so `-t` did nothing for m2m100 / marian.
    if (params.n_threads > 0)
        core_cpu_backend::set_n_threads(c->backend_cpu, params.n_threads);
    const char* gpu_env = std::getenv("CRISPASR_M2M100_GPU");
    const bool force_gpu = gpu_env && std::atoi(gpu_env) != 0;
    const bool force_cpu = gpu_env && std::atoi(gpu_env) == 0;
    c->backend = c->backend_cpu;
    if (!force_cpu && (force_gpu || params.use_gpu)) {
        ggml_backend_t gpu = crispasr_init_gpu_backend();
        if (gpu) {
            bool is_metal = false;
#if defined(GGML_USE_METAL)
            is_metal = core_cpu_backend::is_metal(gpu);
#endif
            if (!is_metal || force_gpu || !hp.marian) {
                c->backend = gpu;
                if (params.verbosity >= 1)
                    fprintf(stderr, "m2m100: GPU backend enabled (%s)\n", ggml_backend_name(c->backend));
            } else {
                ggml_backend_free(gpu);
                if (params.verbosity >= 1)
                    fprintf(stderr,
                            "%s: runs on the CPU on Metal by default (measured faster in the live "
                            "pipeline); set CRISPASR_M2M100_GPU=1 to force the GPU\n",
                            hp.marian ? "marian" : "m2m100");
            }
        }
    }

    // Pass 2: weights
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path_model, c->backend, "m2m100", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    if (!bind_model(c)) {
        fprintf(stderr, "m2m100: failed to bind tensors\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler
    {
        ggml_backend_t backends[] = {c->backend, c->backend_cpu};
        int n_be = (c->backend != c->backend_cpu) ? 2 : 1;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    return c;
}

extern "C" void m2m100_free(struct m2m100_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cross_kv_buf)
        ggml_backend_buffer_free(ctx->cross_kv_buf);
    if (ctx->cross_kv_ctx)
        ggml_free(ctx->cross_kv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void m2m100_set_beam_size(struct m2m100_context* ctx, int beam_size) {
    if (!ctx)
        return;
    ctx->beam_size = beam_size > 1 ? beam_size : 1;
}

extern "C" int m2m100_model_beam_size(struct m2m100_context* ctx) {
    if (!ctx)
        return m2m100_default_beam_size();
    const auto& hp = ctx->model.hp;
    return hp.marian ? (hp.gen_num_beams > 1 ? hp.gen_num_beams : 1) : m2m100_default_beam_size();
}

extern "C" int m2m100_is_marian(struct m2m100_context* ctx) {
    return ctx && ctx->model.hp.marian ? 1 : 0;
}

extern "C" int m2m100_tokenize(struct m2m100_context* ctx, const char* text, const char* src_lang, const char* tgt_lang,
                               int32_t* out_ids, int capacity) {
    if (!ctx || !text)
        return -1;
    std::vector<int> ids;
    if (ctx->model.hp.marian) {
        for (int32_t id :
             ctx->marian_tok.encode(marian_with_target_code(ctx, text, tgt_lang ? tgt_lang : ""), /*add_eos=*/true))
            ids.push_back((int)id);
    } else {
        ids = tokenize(ctx->tokenizer, text, src_lang ? src_lang : "");
    }
    for (int i = 0; i < (int)ids.size() && i < capacity && out_ids; i++)
        out_ids[i] = ids[i];
    return (int)ids.size();
}

extern "C" char* m2m100_translate(struct m2m100_context* ctx, const char* text, const char* src_lang,
                                  const char* tgt_lang, int max_new_tokens) {
    if (!ctx || !text || !src_lang || !tgt_lang)
        return nullptr;

    const auto& hp = ctx->model.hp;
    const char* tag = hp.marian ? "marian" : "m2m100";
    // Rows of the position tables that hold real positions.
    const int max_positions = (int)ctx->model.dec_pos_emb->ne[1] - hp.pos_offset;
    if (max_new_tokens <= 0) {
        // The checkpoint's own bound: max_length counts the start token.
        max_new_tokens = (hp.marian && hp.gen_max_length > 1) ? hp.gen_max_length - 1 : 200;
    }
    if (hp.marian && max_new_tokens > max_positions - 1)
        max_new_tokens = max_positions - 1; // one position is the start token

    m2m100_bench_stage _bs_total("translate_total");

    // 1. Tokenize input
    std::vector<int> enc_ids;
    if (hp.marian) {
        marian_check_languages(ctx, src_lang, tgt_lang);
        for (int32_t id : ctx->marian_tok.encode(marian_with_target_code(ctx, text, tgt_lang), /*add_eos=*/true))
            enc_ids.push_back((int)id);
    } else {
        enc_ids = tokenize(ctx->tokenizer, text, src_lang);
    }
    if (ctx->params.verbosity >= 2) {
        fprintf(stderr, "%s: input %zu tokens:", tag, enc_ids.size());
        for (int id : enc_ids)
            fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    // A longer input would index past the position table. (The M2M-100 table
    // has 1024 positions and never had this check.)
    if ((int)enc_ids.size() > (int)ctx->model.enc_pos_emb->ne[1] - hp.pos_offset) {
        fprintf(stderr, "%s: input is %zu tokens, the model has %d positions — split the text\n", tag, enc_ids.size(),
                (int)ctx->model.enc_pos_emb->ne[1] - hp.pos_offset);
        return nullptr;
    }

    // 2. Run encoder
    std::vector<float> enc_out;
    {
        m2m100_bench_stage _bs_enc("encoder");
        enc_out = run_encoder(ctx, enc_ids);
    }
    if (enc_out.empty())
        return nullptr;

    int T_enc = (int)enc_ids.size();
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "%s: encoder done, T_enc=%d\n", tag, T_enc);
    }

    // 3. Compute cross-attention KV
    if (!compute_cross_kv(ctx, enc_out.data(), T_enc)) {
        fprintf(stderr, "m2m100: failed to compute cross-KV\n");
        return nullptr;
    }

    // 4. Allocate decoder KV cache
    int dec_max = max_new_tokens + 4;
    if (!alloc_kv_cache(ctx, dec_max)) {
        fprintf(stderr, "m2m100: failed to alloc KV cache\n");
        return nullptr;
    }

    // 5. Greedy decode
    // Start with decoder_start_token_id (eos=2) then forced_bos (target lang token).
    // Marian: decoder_start_token_id (<pad>) alone — no language token.
    std::vector<int> dec_ids;
    dec_ids.push_back(hp.dec_start_token); // </s> = 2
    if (!hp.marian) {
        auto tgt_it = ctx->tokenizer.lang_to_token_id.find(tgt_lang);
        if (tgt_it == ctx->tokenizer.lang_to_token_id.end()) {
            fprintf(stderr, "m2m100: unknown target language '%s'\n", tgt_lang);
            return nullptr;
        }
        dec_ids.push_back(tgt_it->second); // __de__ etc.
    }

    m2m100_bench_stage _bs_dec("decode");

    // First step: prefill with [dec_start, forced_bos]
    std::vector<float> logits = run_decoder_step(ctx, dec_ids.data(), (int)dec_ids.size(), 0);
    if (logits.empty())
        return nullptr;

    const int prompt_len = (int)dec_ids.size();

    if (ctx->beam_size > 1) {
        // Beam search on per-beam snapshots of the self-attention cache: one
        // single-token forward per beam per step, O(beam x T). The cross-
        // attention K/V are shared by every beam and never snapshotted.
        // (Replaying each beam's whole suffix per step, as this did before,
        // is O(beam x T^2/2) forwards — and for a text-to-text model the
        // decode IS the cost. CRISPASR_M2M100_BEAM_REPLAY=1 restores it.)
        const char* env_replay = std::getenv("CRISPASR_M2M100_BEAM_REPLAY");
        const bool beam_replay = env_replay && *env_replay && *env_replay != '0';
        auto replay = [](m2m100_context* c, const int32_t* toks, int n, int pl) -> float* {
            auto lg = run_decoder_step(c, (const int*)toks, n, pl);
            if (lg.empty())
                return nullptr;
            float* out = (float*)std::malloc(lg.size() * sizeof(float));
            std::memcpy(out, lg.data(), lg.size() * sizeof(float));
            return out;
        };
        if (beam_replay) {
            const long long worst =
                (long long)ctx->beam_size * (long long)max_new_tokens * (long long)max_new_tokens / 2;
            if (worst > 20000) {
                std::fprintf(stderr,
                             "%s: beam %d with up to %d tokens is worst-case ~%lld decoder forwards "
                             "(replay mode). Cap with --translate-max-tokens.\n",
                             tag, ctx->beam_size, max_new_tokens, worst);
            }
        }
        core_beam_decode::Config bcfg;
        bcfg.max_new_tokens = max_new_tokens;
        bcfg.eos_id = hp.eos_token_id;
        bcfg.vocab_size = hp.vocab_size;
        bcfg.beam_size = ctx->beam_size;
        bcfg.prompt_len = prompt_len;
        // generation_config.json: length_penalty 1.0 (default), early_stopping per checkpoint
        bcfg.early_stopping =
            hp.early_stopping ? core_beam_decode::EarlyStopping::True : core_beam_decode::EarlyStopping::False;
        // the forced target-language BOS: HF generates it, we prompt with it.
        // Marian has none.
        bcfg.length_offset = hp.marian ? 0 : 1;
        core_beam_decode::Result br;
        if (beam_replay) {
            br = core_beam_decode::run_with_probs(ctx, logits.data(), replay, bcfg);
        } else {
            core_attn::kv_snapshot_pool kv_pool(ctx->kv_k, ctx->kv_v);
            auto save_fn = [&kv_pool](m2m100_context*) -> core_attn::kv_snapshot* { return kv_pool.save(); };
            auto restore_fn = [&kv_pool](m2m100_context*, core_attn::kv_snapshot* sn) { kv_pool.restore(sn); };
            auto snap_free_fn = [&kv_pool](core_attn::kv_snapshot* sn) { kv_pool.release(sn); };
            auto step_fn = [](m2m100_context* c, int32_t tok, int n_past) -> float* {
                const int t = (int)tok;
                auto lg = run_decoder_step(c, &t, 1, n_past);
                if (lg.empty())
                    return nullptr;
                float* out = (float*)std::malloc(lg.size() * sizeof(float));
                std::memcpy(out, lg.data(), lg.size() * sizeof(float));
                return out;
            };
            br = core_beam_decode::run_with_probs_branched(ctx, logits.data(), save_fn, restore_fn, snap_free_fn,
                                                           step_fn, bcfg);
        }
        for (int32_t t : br.tokens) {
            if (t == hp.eos_token_id)
                break;
            dec_ids.push_back((int)t);
        }
    } else {
        int offset = prompt_len;
        for (int step = 0; step < max_new_tokens; step++) {
            // Greedy: argmax — NaN-robust (see lfm2_audio note). Seed from -inf,
            // skip non-finite, abort if the whole row is non-finite.
            int best_id = -1;
            float best_val = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < hp.vocab_size; i++) {
                if (std::isfinite(logits[i]) && logits[i] > best_val) {
                    best_val = logits[i];
                    best_id = i;
                }
            }
            if (best_id < 0) {
                fprintf(stderr, "%s: non-finite logits — aborting decode\n", tag);
                break;
            }

            if (best_id == hp.eos_token_id)
                break;

            dec_ids.push_back(best_id);

            if (ctx->params.verbosity >= 2) {
                fprintf(stderr, "%s[dec]: step=%d tok=%d '%s'\n", tag, step, best_id,
                        best_id < (int)ctx->tokenizer.id_to_token.size() ? ctx->tokenizer.id_to_token[best_id].c_str()
                                                                         : "?");
            }

            // Next step: single token
            logits = run_decoder_step(ctx, &best_id, 1, offset);
            if (logits.empty())
                return nullptr;
            offset++;
        }
    }

    if (m2m100_bench_enabled()) {
        auto& b = g_m2m100_step_bench;
        std::fprintf(stderr,
                     "  m2m100_bench: decoder steps %d: build %.2f  alloc %.2f  compute %.2f  read %.2f ms "
                     "(per step %.2f / %.2f / %.2f / %.2f)\n",
                     b.steps, b.build_ms, b.alloc_ms, b.compute_ms, b.read_ms, b.build_ms / std::max(1, b.steps),
                     b.alloc_ms / std::max(1, b.steps), b.compute_ms / std::max(1, b.steps),
                     b.read_ms / std::max(1, b.steps));
        b = m2m100_step_bench();
    }

    // 6. Detokenize
    std::string result = hp.marian ? marian_detokenize(ctx, dec_ids) : detokenize(ctx->tokenizer, dec_ids);
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "%s: translated %zu tokens → '%s'\n", tag, dec_ids.size(), result.c_str());
    }

    char* out = (char*)malloc(result.size() + 1);
    std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" int m2m100_n_languages(struct m2m100_context* ctx) {
    return ctx ? (int)ctx->tokenizer.lang_codes.size() : 0;
}

extern "C" const char* m2m100_language(struct m2m100_context* ctx, int index) {
    if (!ctx || index < 0 || index >= (int)ctx->tokenizer.lang_codes.size())
        return nullptr;
    return ctx->tokenizer.lang_codes[index].c_str();
}
