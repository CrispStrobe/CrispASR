// outetts.cpp -- OuteTTS-0.3-1B TTS backend.
//
// OLMo 1B (Llama-compatible architecture: RMSNorm, SwiGLU, MHA, RoPE)
// finetuned to emit interleaved text + audio tokens.  Audio tokens
// <|0|>..<|4099|> index into a WavTokenizer single codebook (4096 entries,
// 75 tok/s, 24 kHz).
//
// V2 prompt format:
//   <|im_start|>\n
//   <|text_start|>word1<|space|>word2<|text_end|>\n
//   <|audio_start|>\n
//   [model generates audio tokens interleaved with word text]
//   <|audio_end|>
//
// Implementation follows the orpheus.cpp pattern (same Llama-style LLM
// forward); the only delta is the prompt format and the codec token
// extraction logic (single codebook, no 7-slot de-interleave).

#include "outetts.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "outetts_wavtok.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
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

struct outetts_hp {
    uint32_t n_layers = 16;
    uint32_t d_model = 2048;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 16; // MHA
    uint32_t head_dim = 128;
    uint32_t ff_dim = 8192;
    uint32_t vocab_size = 57344;
    uint32_t max_pos = 4096;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;

    // OuteTTS V2 special tokens (discovered from GGUF metadata)
    uint32_t im_start_token = 0;
    uint32_t im_end_token = 0;
    uint32_t text_start_token = 0;
    uint32_t text_end_token = 0;
    uint32_t audio_start_token = 0;
    uint32_t audio_end_token = 0;
    uint32_t space_token = 0;

    // Audio code token block
    uint32_t audio_token_offset = 0; // token id of <|0|>
    uint32_t audio_token_count = 4100;

    // WavTokenizer info
    uint32_t wavtok_codebook_size = 4096;
    uint32_t wavtok_sample_rate = 24000;
};

struct outetts_layer {
    // OLMo uses parameter-free LayerNorm -- no norm weights stored.
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct outetts_talker {
    ggml_tensor* token_embd_w = nullptr;
    std::vector<outetts_layer> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr;
};

struct outetts_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

} // namespace

// Speaker profile word: word text + duration + audio codes
struct outetts_speaker_word {
    std::string word;
    float duration = 0.0f;
    std::vector<int32_t> codes;
};

struct outetts_speaker {
    std::string text;
    std::vector<outetts_speaker_word> words;
};

struct outetts_context {
    outetts_context_params params{};
    int n_threads = 4;

    outetts_hp hp;
    outetts_vocab vocab;
    outetts_talker talker;

    // Speaker profile for voice cloning (optional)
    outetts_speaker speaker;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_backend_buffer_t buf_w_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // WavTokenizer decoder
    std::string wavtok_codec_path;
    wavtok_decoder_ctx* wavtok_dec = nullptr;

    // RNG state (xorshift64*)
    uint64_t rng_state = 0xdeadbeefcafebabeULL;

    ~outetts_context() {
        if (wavtok_dec) {
            wavtok_decoder_free(wavtok_dec);
        }
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (kv_buf) {
            ggml_backend_buffer_free(kv_buf);
        }
        if (kv_ctx) {
            ggml_free(kv_ctx);
        }
        if (ctx_w) {
            ggml_free(ctx_w);
        }
        if (buf_w) {
            ggml_backend_buffer_free(buf_w);
        }
        if (buf_w_cpu) {
            ggml_backend_buffer_free(buf_w_cpu);
        }
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace {

static bool bind_talker(outetts_context* c) {
    auto& t = c->talker;
    auto& tensors = c->tensors;

    t.token_embd_w = core_gguf::require(tensors, "talker.token_embd.weight", "outetts");
    // OLMo may not have a final layer norm (output_norm is optional)
    t.output_norm_w = core_gguf::try_get(tensors, "talker.output_norm.weight");
    // Tied embeddings: try separate output, fall back to token_embd
    t.output_w = core_gguf::try_get(tensors, "talker.output.weight");
    if (!t.output_w) {
        t.output_w = t.token_embd_w;
    }

    if (!t.token_embd_w) {
        return false;
    }

    t.blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = t.blocks[i];
        char key[64];
#define FMT(fld, sub)                                                                                                  \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "talker.blk.%u." sub ".weight", i);                                            \
        b.fld = core_gguf::require(tensors, key, "outetts");                                                           \
    } while (0)
        FMT(attn_q_w, "attn_q");
        FMT(attn_k_w, "attn_k");
        FMT(attn_v_w, "attn_v");
        FMT(attn_output_w, "attn_output");
        FMT(ffn_gate_w, "ffn_gate");
        FMT(ffn_up_w, "ffn_up");
        FMT(ffn_down_w, "ffn_down");
#undef FMT
        if (!b.attn_q_w || !b.attn_k_w || !b.attn_v_w || !b.attn_output_w || !b.ffn_gate_w || !b.ffn_up_w ||
            !b.ffn_down_w) {
            fprintf(stderr, "outetts: missing tensor in layer %u\n", i);
            return false;
        }
    }
    return true;
}

static void load_metadata(outetts_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.n_layers = core_gguf::kv_u32(g, "outetts.talker.n_layers", hp.n_layers);
    hp.d_model = core_gguf::kv_u32(g, "outetts.talker.d_model", hp.d_model);
    hp.n_heads = core_gguf::kv_u32(g, "outetts.talker.n_heads", hp.n_heads);
    hp.n_kv_heads = core_gguf::kv_u32(g, "outetts.talker.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = core_gguf::kv_u32(g, "outetts.talker.head_dim", hp.head_dim);
    hp.ff_dim = core_gguf::kv_u32(g, "outetts.talker.ff_dim", hp.ff_dim);
    hp.vocab_size = core_gguf::kv_u32(g, "outetts.talker.vocab_size", hp.vocab_size);
    hp.max_pos = core_gguf::kv_u32(g, "outetts.talker.max_pos", hp.max_pos);
    hp.rope_theta = core_gguf::kv_f32(g, "outetts.talker.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = core_gguf::kv_f32(g, "outetts.talker.rms_norm_eps", hp.rms_norm_eps);

    hp.im_start_token = core_gguf::kv_u32(g, "outetts.im_start_token", hp.im_start_token);
    hp.im_end_token = core_gguf::kv_u32(g, "outetts.im_end_token", hp.im_end_token);
    hp.text_start_token = core_gguf::kv_u32(g, "outetts.text_start_token", hp.text_start_token);
    hp.text_end_token = core_gguf::kv_u32(g, "outetts.text_end_token", hp.text_end_token);
    hp.audio_start_token = core_gguf::kv_u32(g, "outetts.audio_start_token", hp.audio_start_token);
    hp.audio_end_token = core_gguf::kv_u32(g, "outetts.audio_end_token", hp.audio_end_token);
    hp.space_token = core_gguf::kv_u32(g, "outetts.space_token", hp.space_token);

    hp.audio_token_offset = core_gguf::kv_u32(g, "outetts.audio_token_offset", hp.audio_token_offset);
    hp.audio_token_count = core_gguf::kv_u32(g, "outetts.audio_token_count", hp.audio_token_count);

    hp.wavtok_codebook_size = core_gguf::kv_u32(g, "outetts.wavtok.codebook_size", hp.wavtok_codebook_size);
    hp.wavtok_sample_rate = core_gguf::kv_u32(g, "outetts.wavtok.sample_rate", hp.wavtok_sample_rate);

    // Tokenizer
    auto tok = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!tok.empty()) {
        c->vocab.id_to_token = std::move(tok);
        c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
        for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++) {
            c->vocab.token_to_id[c->vocab.id_to_token[i]] = i;
        }
    }
    auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); i++) {
        c->vocab.merge_rank[merges[i]] = (int32_t)i;
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

static bool kv_alloc(outetts_context* c, int max_ctx) {
    if (c->kv_k) {
        return true;
    }
    const auto& hp = c->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.n_kv_heads;
    const int nl = (int)hp.n_layers;
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    c->kv_ctx = ggml_init(kp);
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("outetts");
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");
    const size_t kb = ggml_nbytes(c->kv_k), vb = ggml_nbytes(c->kv_v);
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(c->backend, c->backend_cpu, "outetts");
    c->kv_buf = ggml_backend_alloc_buffer(kv_backend, kb + vb);
    if (!c->kv_buf) {
        fprintf(stderr, "outetts: kv alloc failed\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(c->kv_buf);
    ggml_backend_tensor_alloc(c->kv_buf, c->kv_k, base);
    ggml_backend_tensor_alloc(c->kv_buf, c->kv_v, base + kb);
    c->kv_max_ctx = max_ctx;
    if (c->params.verbosity >= 1) {
        fprintf(stderr, "outetts: kv cache %d MiB (hd=%d max=%d n_kv=%d nl=%d)\n", (int)((kb + vb) / 1048576), hd,
                max_ctx, n_kv, nl);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

static ggml_cgraph* build_graph_embed(outetts_context* c, int n_tokens) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, c->talker.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// OLMo block stack (Llama-compatible: RMSNorm -> attn -> RMSNorm -> SwiGLU).
static ggml_cgraph* build_graph_talker_kv(outetts_context* c, int n_past, int n_tokens) {
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int n_q = (int)hp.n_heads;
    const int n_kv = (int)hp.n_kv_heads;
    const int hd = (int)hp.head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.rms_norm_eps;
    const float theta = hp.rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = n_past + T;

    GGML_ASSERT(c->kv_k && c->kv_v && Lk <= c->kv_max_ctx);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
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
        /*n_ctx_orig*/ (int)hp.max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ 0.0f,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->talker.blocks[il];
        ggml_tensor* residual = cur;

        // Pre-norm: OLMo uses parameter-free LayerNorm (no weight/bias)
        ggml_tensor* x = ggml_norm(ctx0, cur, eps);

        // Self-attention with KV cache
        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                    nullptr, nullptr, positions, (T == 1) ? nullptr : causal_mask,
                                                    c->kv_k, c->kv_v, (int)il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        // FFN
        residual = cur;
        x = ggml_norm(ctx0, cur, eps);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final norm + lm_head
    // OLMo uses parameter-free LayerNorm (no weight/bias) everywhere,
    // including the final norm before the lm_head.
    cur = ggml_norm(ctx0, cur, eps);
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, c->talker.output_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ---------------------------------------------------------------------------
// Compute helpers
// ---------------------------------------------------------------------------

static float* embed_tokens(outetts_context* c, const int32_t* ids, int n) {
    const int d = (int)c->hp.d_model;
    ggml_cgraph* gf = build_graph_embed(c, n);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), ids, 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* r = (float*)malloc((size_t)d * n * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)d * n * sizeof(float));
    return r;
}

static float* run_talker_kv(outetts_context* c, const float* embeds, int n_tokens, int n_past) {
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "outetts: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
        return nullptr;
    }
    const auto& hp = c->hp;
    const int d = (int)hp.d_model;
    const int vocab = (int)hp.vocab_size;
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        positions[i] = n_past + i;
    }

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

    ggml_cgraph* gf = build_graph_talker_kv(c, n_past, n_tokens);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "outetts: failed to alloc talker_kv graph\n");
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "outetts: talker_kv compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

static int argmax_logits(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}

static int sample_logits(const float* logits, int n, float temperature, int top_k, uint64_t* rng_state) {
    if (temperature <= 0.0f) {
        return argmax_logits(logits, n);
    }
    if (top_k <= 0 || top_k > n) {
        top_k = n;
    }
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) {
        idx[i] = i;
    }
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    float max_l = logits[idx[0]];
    std::vector<float> probs(top_k);
    double sum = 0.0;
    for (int i = 0; i < top_k; i++) {
        double p = std::exp((double)(logits[idx[i]] - max_l) / (double)temperature);
        probs[i] = (float)p;
        sum += p;
    }
    if (sum <= 0.0) {
        return idx[0];
    }
    for (int i = 0; i < top_k; i++) {
        probs[i] = (float)(probs[i] / sum);
    }
    uint64_t x = *rng_state ? *rng_state : 0xdeadbeefcafebabeULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    double r = (double)((x * 0x2545f4914f6cdd1dULL) >> 11) / (double)(1ULL << 53);
    double cum = 0.0;
    for (int i = 0; i < top_k; i++) {
        cum += probs[i];
        if (r < cum) {
            return idx[i];
        }
    }
    return idx[top_k - 1];
}

// ---------------------------------------------------------------------------
// Prompt construction (V2 format)
// ---------------------------------------------------------------------------

// Split text into words, handling basic punctuation.
static std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string cur;
    for (char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\n') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur += ch;
        }
    }
    if (!cur.empty()) {
        words.push_back(cur);
    }
    return words;
}

static std::string to_lower(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        out += (ch >= 'A' && ch <= 'Z') ? (char)(ch + 32) : ch;
    }
    return out;
}

// Tokenize a word using BPE (helper for prompt construction)
static void append_bpe_word(std::vector<int32_t>& out, const outetts_vocab& v, const std::string& word) {
    std::vector<int32_t> toks = core_bpe::tokenize_simple(v.token_to_id, v.merge_rank, word);
    out.insert(out.end(), toks.begin(), toks.end());
}

// Look up a duration token like <|t_0.53|>
static int32_t find_time_token(const outetts_vocab& v, float duration) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "<|t_%.2f|>", duration);
    auto it = v.token_to_id.find(buf);
    return it != v.token_to_id.end() ? it->second : -1;
}

// Build OuteTTS V2 prompt.
//
// Without speaker:
//   <|im_start|>\n<|text_start|>word1<|space|>word2<|text_end|>\n<|audio_start|>\n
//
// With speaker (voice cloning):
//   <|im_start|>\n<|text_start|>spk_w1<|space|>spk_w2<|space|>new_w1<|space|>new_w2<|text_end|>\n
//   <|audio_start|>\n
//   spk_w1<|t_0.53|><|123|><|456|>...<|space|>\n
//   spk_w2<|t_0.27|><|789|>...<|space|>\n
//
static std::vector<int32_t> build_prompt_ids(outetts_context* c, const std::string& text) {
    const auto& hp = c->hp;
    const auto& v = c->vocab;
    const bool has_speaker = !c->speaker.words.empty();

    std::string text_lower = to_lower(text);

    std::vector<int32_t> out;
    out.reserve(has_speaker ? 4096 : 256);

    auto tok_id = [&](const std::string& name) -> int32_t {
        auto it = v.token_to_id.find(name);
        return it != v.token_to_id.end() ? it->second : -1;
    };

    // Newline token: GPT-NeoX byte-level BPE encodes '\n' as 'Ċ' (U+010A)
    int32_t nl_id = tok_id("\xc4\x8a");
    if (nl_id < 0)
        nl_id = tok_id("\n");
    if (nl_id < 0)
        nl_id = tok_id("\\n");

    auto push_nl = [&]() {
        if (nl_id >= 0)
            out.push_back(nl_id);
    };

    // <|im_start|>\n
    out.push_back((int32_t)hp.im_start_token);
    push_nl();

    // <|text_start|>
    out.push_back((int32_t)hp.text_start_token);

    // Speaker words (if speaker profile loaded)
    if (has_speaker) {
        for (size_t i = 0; i < c->speaker.words.size(); i++) {
            if (i > 0)
                out.push_back((int32_t)hp.space_token);
            append_bpe_word(out, v, c->speaker.words[i].word);
        }
        out.push_back((int32_t)hp.space_token);
    }

    // New text words
    std::vector<std::string> words = split_words(text_lower);
    for (size_t i = 0; i < words.size(); i++) {
        if (i > 0)
            out.push_back((int32_t)hp.space_token);
        append_bpe_word(out, v, words[i]);
    }

    // <|text_end|>\n
    out.push_back((int32_t)hp.text_end_token);
    push_nl();

    // <|audio_start|>\n
    out.push_back((int32_t)hp.audio_start_token);
    push_nl();

    // Speaker audio context: word + duration + codes for each speaker word
    if (has_speaker) {
        const int32_t ato = (int32_t)hp.audio_token_offset;
        for (size_t i = 0; i < c->speaker.words.size(); i++) {
            const auto& sw = c->speaker.words[i];

            // word text (BPE tokenized)
            append_bpe_word(out, v, sw.word);

            // <|t_0.53|> duration marker
            int32_t t_tok = find_time_token(v, sw.duration);
            if (t_tok >= 0) {
                out.push_back(t_tok);
            }

            // Audio code tokens: <|123|>, <|456|>, ...
            for (int32_t code : sw.codes) {
                // code is codebook index 0-4095 → token is audio_token_offset + code
                out.push_back(ato + code);
            }

            // <|space|>\n between speaker words
            out.push_back((int32_t)hp.space_token);
            push_nl();
        }
    }

    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Public C ABI
// ---------------------------------------------------------------------------

extern "C" struct outetts_context_params outetts_context_default_params(void) {
    outetts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.4f; // upstream default
    p.seed = 0;
    p.max_audio_tokens = 0;
    p.flash_attn = true;
    return p;
}

extern "C" struct outetts_context* outetts_init_from_file(const char* path_model,
                                                          struct outetts_context_params params) {
    auto* c = new outetts_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    c->rng_state = params.seed != 0 ? params.seed : 0xdeadbeefcafebabeULL;

    // Pass 1: hparams + vocab
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "outetts: talker=%uL d=%u h=%u/%u hd=%u ff=%u vocab=%u\n"
                "outetts: rope_theta=%.0f  audio_offset=%u count=%u\n",
                c->hp.n_layers, c->hp.d_model, c->hp.n_heads, c->hp.n_kv_heads, c->hp.head_dim, c->hp.ff_dim,
                c->hp.vocab_size, (double)c->hp.rope_theta, c->hp.audio_token_offset, c->hp.audio_token_count);
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "outetts: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "outetts", wl)) {
        fprintf(stderr, "outetts: failed to load weights from '%s'\n", path_model);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->buf_w_cpu = wl.buf_cpu;
    c->tensors = std::move(wl.tensors);

    if (!bind_talker(c)) {
        fprintf(stderr, "outetts: tensor binding failed\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend) {
            backends[n_be++] = c->backend_cpu;
        }
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    return c;
}

extern "C" void outetts_free(struct outetts_context* ctx) {
    delete ctx;
}

extern "C" void outetts_set_n_threads(struct outetts_context* ctx, int n_threads) {
    if (!ctx)
        return;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
}

extern "C" void outetts_set_temperature(struct outetts_context* ctx, float temperature) {
    if (!ctx)
        return;
    ctx->params.temperature = std::max(0.0f, std::min(4.0f, temperature));
}

extern "C" void outetts_set_seed(struct outetts_context* ctx, uint64_t seed) {
    if (!ctx)
        return;
    ctx->params.seed = seed;
    ctx->rng_state = seed != 0 ? seed : 0xdeadbeefcafebabeULL;
}

// Minimal JSON parser for speaker profile:
// {"text": "...", "words": [{"word": "...", "duration": 0.53, "codes": [123, 456, ...]}, ...]}
static bool parse_speaker_json(const std::string& json, outetts_speaker& spk) {
    spk.text.clear();
    spk.words.clear();

    // Helper: skip whitespace
    auto skip_ws = [&](size_t& p) {
        while (p < json.size() && (json[p] == ' ' || json[p] == '\n' || json[p] == '\r' || json[p] == '\t'))
            p++;
    };

    // Helper: parse a JSON string starting at pos (which should point to opening ")
    auto parse_str = [&](size_t& p) -> std::string {
        if (p >= json.size() || json[p] != '"')
            return "";
        p++; // skip opening "
        std::string s;
        while (p < json.size() && json[p] != '"') {
            if (json[p] == '\\' && p + 1 < json.size()) {
                p++;
                if (json[p] == '"')
                    s += '"';
                else if (json[p] == '\\')
                    s += '\\';
                else if (json[p] == 'n')
                    s += '\n';
                else
                    s += json[p];
            } else {
                s += json[p];
            }
            p++;
        }
        if (p < json.size())
            p++; // skip closing "
        return s;
    };

    // Helper: parse a number (int or float)
    auto parse_num = [&](size_t& p) -> double {
        skip_ws(p);
        size_t start = p;
        while (p < json.size() && (json[p] == '-' || json[p] == '.' || (json[p] >= '0' && json[p] <= '9')))
            p++;
        return std::atof(json.substr(start, p - start).c_str());
    };

    // Find top-level object
    size_t pos = json.find('{');
    if (pos == std::string::npos)
        return false;
    pos++;

    while (pos < json.size()) {
        skip_ws(pos);
        if (pos >= json.size() || json[pos] == '}')
            break;
        if (json[pos] == ',') {
            pos++;
            continue;
        }

        std::string key = parse_str(pos);
        skip_ws(pos);
        if (pos < json.size() && json[pos] == ':')
            pos++;
        skip_ws(pos);

        if (key == "text") {
            spk.text = parse_str(pos);
        } else if (key == "words") {
            // Parse array of word objects
            if (pos >= json.size() || json[pos] != '[')
                return false;
            pos++; // skip [
            while (pos < json.size()) {
                skip_ws(pos);
                if (json[pos] == ']') {
                    pos++;
                    break;
                }
                if (json[pos] == ',') {
                    pos++;
                    continue;
                }
                if (json[pos] != '{')
                    return false;
                pos++; // skip {

                outetts_speaker_word w;
                while (pos < json.size()) {
                    skip_ws(pos);
                    if (json[pos] == '}') {
                        pos++;
                        break;
                    }
                    if (json[pos] == ',') {
                        pos++;
                        continue;
                    }

                    std::string wkey = parse_str(pos);
                    skip_ws(pos);
                    if (pos < json.size() && json[pos] == ':')
                        pos++;
                    skip_ws(pos);

                    if (wkey == "word") {
                        w.word = parse_str(pos);
                    } else if (wkey == "duration") {
                        w.duration = (float)parse_num(pos);
                    } else if (wkey == "codes") {
                        if (pos >= json.size() || json[pos] != '[')
                            return false;
                        pos++; // skip [
                        while (pos < json.size()) {
                            skip_ws(pos);
                            if (json[pos] == ']') {
                                pos++;
                                break;
                            }
                            if (json[pos] == ',') {
                                pos++;
                                continue;
                            }
                            w.codes.push_back((int32_t)parse_num(pos));
                        }
                    } else {
                        // Skip unknown value
                        if (json[pos] == '"')
                            parse_str(pos);
                        else if (json[pos] == '[') {
                            int depth = 1;
                            pos++;
                            while (pos < json.size() && depth > 0) {
                                if (json[pos] == '[')
                                    depth++;
                                if (json[pos] == ']')
                                    depth--;
                                pos++;
                            }
                        } else
                            parse_num(pos);
                    }
                }
                spk.words.push_back(std::move(w));
            }
        } else {
            // Skip unknown top-level value
            if (json[pos] == '"')
                parse_str(pos);
            else if (json[pos] == '{') {
                int depth = 1;
                pos++;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '{')
                        depth++;
                    if (json[pos] == '}')
                        depth--;
                    pos++;
                }
            } else if (json[pos] == '[') {
                int depth = 1;
                pos++;
                while (pos < json.size() && depth > 0) {
                    if (json[pos] == '[')
                        depth++;
                    if (json[pos] == ']')
                        depth--;
                    pos++;
                }
            } else
                parse_num(pos);
        }
    }
    return !spk.words.empty();
}

extern "C" int outetts_load_speaker(struct outetts_context* ctx, const char* json_path) {
    if (!ctx || !json_path)
        return -1;

    FILE* f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "outetts: cannot open speaker JSON '%s'\n", json_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json(fsize, '\0');
    size_t nread = fread(&json[0], 1, fsize, f);
    fclose(f);
    if ((long)nread != fsize) {
        fprintf(stderr, "outetts: failed to read speaker JSON '%s'\n", json_path);
        return -1;
    }

    if (!parse_speaker_json(json, ctx->speaker)) {
        fprintf(stderr, "outetts: failed to parse speaker JSON '%s'\n", json_path);
        return -1;
    }

    int total_codes = 0;
    for (const auto& w : ctx->speaker.words)
        total_codes += (int)w.codes.size();

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "outetts: loaded speaker profile: %d words, %d codes, text='%s'\n",
                (int)ctx->speaker.words.size(), total_codes, ctx->speaker.text.c_str());
    }
    return 0;
}

extern "C" int outetts_set_codec_path(struct outetts_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->wavtok_codec_path = path;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "outetts: codec path set to '%s'\n", path);
    }
    return 0;
}

extern "C" int32_t* outetts_synthesize_codes(struct outetts_context* ctx, const char* text, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !text)
        return nullptr;
    const auto& hp = ctx->hp;

    std::vector<int32_t> prompt_ids = build_prompt_ids(ctx, text);
    if (prompt_ids.empty())
        return nullptr;

    const int max_audio = ctx->params.max_audio_tokens > 0 ? ctx->params.max_audio_tokens : 4096;
    const int kv_need = (int)prompt_ids.size() + max_audio + 8;
    if (!kv_alloc(ctx, kv_need))
        return nullptr;
    int n_past = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "outetts: prompt %d tokens (max_audio=%d)\n", (int)prompt_ids.size(), max_audio);
    }

    // Prefill
    float* prompt_embeds = embed_tokens(ctx, prompt_ids.data(), (int)prompt_ids.size());
    if (!prompt_embeds)
        return nullptr;
    float* logits = run_talker_kv(ctx, prompt_embeds, (int)prompt_ids.size(), n_past);
    free(prompt_embeds);
    if (!logits)
        return nullptr;
    n_past += (int)prompt_ids.size();

    // AR decode loop
    std::vector<int32_t> audio_codes;
    audio_codes.reserve(max_audio);
    const int top_k = 40;           // upstream default
    const float rep_penalty = 1.1f; // upstream default repetition_penalty
    const bool debug = ctx->params.verbosity >= 2;
    const int32_t ato = (int32_t)hp.audio_token_offset;
    const int32_t atc = (int32_t)hp.audio_token_count;
    int n_non_audio = 0;
    std::vector<int32_t> all_generated; // for repetition penalty

    for (int step = 0; step < max_audio; step++) {
        // Apply repetition penalty: divide logits of previously generated tokens
        for (int32_t prev : all_generated) {
            if (prev >= 0 && prev < (int32_t)hp.vocab_size) {
                if (logits[prev] > 0.0f) {
                    logits[prev] /= rep_penalty;
                } else {
                    logits[prev] *= rep_penalty;
                }
            }
        }
        int tok = sample_logits(logits, (int)hp.vocab_size, ctx->params.temperature, top_k, &ctx->rng_state);
        free(logits);
        logits = nullptr;
        all_generated.push_back((int32_t)tok);

        if (debug && step < 20) {
            const char* tok_name =
                (tok >= 0 && tok < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[tok].c_str() : "?";
            fprintf(stderr, "outetts[ar]: step=%d tok=%d (%s)\n", step, tok, tok_name);
        }

        // Stop on <|audio_end|>
        if (tok == (int)hp.audio_end_token) {
            break;
        }

        // Collect audio codes
        if (tok >= ato && tok < ato + atc) {
            int32_t code = tok - ato;
            // Clamp to valid codebook range
            if (code >= (int32_t)hp.wavtok_codebook_size) {
                code = hp.wavtok_codebook_size - 1;
            }
            audio_codes.push_back(code);
            n_non_audio = 0; // reset
        } else {
            n_non_audio++;
            // If too many consecutive non-audio tokens, model may have
            // fallen out of the audio generation mode
            if (n_non_audio > 10) {
                if (ctx->params.verbosity >= 1) {
                    fprintf(stderr, "outetts: >10 non-audio tokens, stopping AR\n");
                }
                break;
            }
        }

        // Decode step
        int32_t single = (int32_t)tok;
        float* step_emb = embed_tokens(ctx, &single, 1);
        if (!step_emb)
            break;
        logits = run_talker_kv(ctx, step_emb, 1, n_past);
        free(step_emb);
        if (!logits)
            break;
        n_past++;
    }
    if (logits)
        free(logits);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "outetts: AR emitted %d audio codes (~%.1f s at 75 tok/s)\n", (int)audio_codes.size(),
                (float)audio_codes.size() / 75.0f);
    }

    if (audio_codes.empty())
        return nullptr;

    int32_t* result = (int32_t*)malloc(audio_codes.size() * sizeof(int32_t));
    if (!result)
        return nullptr;
    std::memcpy(result, audio_codes.data(), audio_codes.size() * sizeof(int32_t));
    if (out_n)
        *out_n = (int)audio_codes.size();
    return result;
}

extern "C" float* outetts_synthesize(struct outetts_context* ctx, const char* text, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !text)
        return nullptr;

    if (ctx->wavtok_codec_path.empty()) {
        fprintf(stderr, "outetts: no WavTokenizer codec path set -- call outetts_set_codec_path first\n");
        return nullptr;
    }
    if (!ctx->wavtok_dec) {
        wavtok_decoder_params dp = wavtok_decoder_default_params();
        dp.n_threads = ctx->n_threads;
        dp.verbosity = ctx->params.verbosity;
        dp.use_gpu = ctx->params.use_gpu ? 1 : 0;
        ctx->wavtok_dec = wavtok_decoder_init_from_file(ctx->wavtok_codec_path.c_str(), dp);
        if (!ctx->wavtok_dec) {
            fprintf(stderr, "outetts: failed to load WavTokenizer from '%s'\n", ctx->wavtok_codec_path.c_str());
            return nullptr;
        }
    }

    int n_codes = 0;
    int32_t* codes = outetts_synthesize_codes(ctx, text, &n_codes);
    if (!codes || n_codes <= 0)
        return nullptr;

    int n_pcm = 0;
    float* pcm = wavtok_decoder_decode(ctx->wavtok_dec, codes, n_codes, &n_pcm);
    free(codes);

    if (!pcm || n_pcm <= 0)
        return nullptr;
    if (out_n_samples)
        *out_n_samples = n_pcm;
    return pcm;
}

extern "C" void outetts_codes_free(int32_t* codes) {
    std::free(codes);
}

extern "C" void outetts_pcm_free(float* pcm) {
    std::free(pcm);
}
