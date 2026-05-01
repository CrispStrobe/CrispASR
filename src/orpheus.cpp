// orpheus.cpp — Orpheus-3B (Llama-3.2-3B + SNAC 24 kHz) TTS backend.
//
// Slice (a) of PLAN #57 Phase 2 landed the foundation (GGUF loader,
// hparams, fixed-speaker table, C ABI). Slice (b) shipped the SNAC
// C++ decoder (`orpheus_snac.{h,cpp}`).
//
// This file completes slice (c): the talker AR forward. The Orpheus
// talker is a Llama-3.2-3B-Instruct that was finetuned to emit a
// stream of <custom_token_N> LM tokens; every 7 tokens form one SNAC
// super-frame. We:
//
//   1. Tokenize "{name}: {text}" with the Llama-3 BPE merges that the
//      converter stamped into the GGUF.
//   2. Wrap with the Orpheus prompt header/footer:
//        [<|audio_start|>=128259]
//          tokens of "{name}: {text}"
//        [<|eot_id|>=128009, <|audio_eot|>=128260,
//         <|audio_eom|>=128261, <|audio_end|>=128257]
//      (verbatim from canopyai/Orpheus-TTS:engine_class.py).
//   3. Prefill, then AR-decode on a persistent KV cache until we sample
//      <|audio_end|> (128257) again or hit max_audio_tokens.
//   4. Filter to <custom_token_N> ids only, de-interleave per the 7-slot
//      layout (slot 0→c0, slots 1+4→c1, slots 2,3,5,6→c2), and drive
//      the SNAC C++ decoder for 24 kHz mono PCM.

#include "orpheus.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/attention.h"
#include "orpheus_snac.h"

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

struct orpheus_hp {
    // Llama-3.2-3B-Instruct talker
    uint32_t n_layers = 28;
    uint32_t d_model = 3072;
    uint32_t n_heads = 24;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 8192;
    uint32_t vocab_size = 156938; // base 128256 + 7*4096 custom + handful of markers
    uint32_t max_pos = 131072;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;

    // Audio wrapper / codec slot tokens (literal IDs from
    // orpheus_tts_pypi/orpheus_tts/{engine_class.py,decoder.py}).
    uint32_t audio_start = 128259;
    uint32_t audio_pre_end = 128009;
    uint32_t audio_end_a = 128260;
    uint32_t audio_end_b = 128261;
    uint32_t audio_end = 128257;
    uint32_t custom_token_offset = 128266; // <custom_token_0> id
    uint32_t custom_token_count = 7 * 4096;

    // SNAC slot layout — 7 LM tokens / super-frame, codes per book = [1,2,4]
    uint32_t super_frame_slots = 7;
    uint32_t cb_count = 3;
    uint32_t cb_size = 4096;

    // Variant ("base" | "fixed_speaker"). Drives whether --voice is
    // required at synthesis time.
    std::string tts_model_type = "fixed_speaker";

    // Baked speakers — used as the literal `name: text` prompt prefix.
    std::vector<std::string> spk_names;
};

struct orpheus_layer {
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

struct orpheus_talker {
    ggml_tensor* token_embd_w = nullptr; // (d_model, vocab_size)
    std::vector<orpheus_layer> blocks;
    ggml_tensor* output_norm_w = nullptr;
    ggml_tensor* output_w = nullptr; // lm_head (d_model, vocab_size)
};

struct orpheus_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" → rank
};

} // namespace

struct orpheus_context {
    orpheus_context_params params{};
    int n_threads = 4;

    orpheus_hp hp;
    orpheus_vocab vocab;
    orpheus_talker talker;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler — created once, reused across embed / talker_kv calls.
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache (lazy-allocated on first run_talker_kv call).
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;

    // SNAC codec (lazy-loaded on first orpheus_synthesize call).
    std::string snac_codec_path;
    snac_decoder_ctx* snac_dec = nullptr;

    // Currently selected fixed speaker (literal `name:` prompt prefix).
    int active_speaker = -1;

    // Sampler RNG state (xorshift64*).
    uint64_t rng_state = 0xdeadbeefcafebabeULL;

    ~orpheus_context() {
        if (snac_dec) {
            snac_decoder_free(snac_dec);
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
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace {

static std::string lower(const std::string& s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return r;
}

// Bind the talker tensors out of the freshly-loaded weight map.
static bool bind_talker(orpheus_context* c) {
    auto& t = c->talker;
    auto& tensors = c->tensors;

    t.token_embd_w = core_gguf::require(tensors, "talker.token_embd.weight", "orpheus");
    t.output_norm_w = core_gguf::require(tensors, "talker.output_norm.weight", "orpheus");
    // lm_head — Llama-3.2-3B does NOT tie input + output embeddings, so
    // we expect a separate talker.output.weight tensor.
    t.output_w = core_gguf::try_get(tensors, "talker.output.weight");
    if (!t.output_w) {
        // Some converters tie the embeddings instead — fall back to
        // token_embd.weight for the lm_head.
        t.output_w = t.token_embd_w;
    }

    if (!t.token_embd_w || !t.output_norm_w) {
        return false;
    }

    t.blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = t.blocks[i];
        char key[64];
#define FMT(fld, sub)                                                                                                  \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "talker.blk.%u." sub ".weight", i);                                            \
        b.fld = core_gguf::require(tensors, key, "orpheus");                                                           \
    } while (0)
        FMT(attn_norm_w, "attn_norm");
        FMT(attn_q_w, "attn_q");
        FMT(attn_k_w, "attn_k");
        FMT(attn_v_w, "attn_v");
        FMT(attn_output_w, "attn_output");
        FMT(ffn_norm_w, "ffn_norm");
        FMT(ffn_gate_w, "ffn_gate");
        FMT(ffn_up_w, "ffn_up");
        FMT(ffn_down_w, "ffn_down");
#undef FMT
        if (!b.attn_norm_w || !b.attn_q_w || !b.attn_k_w || !b.attn_v_w || !b.attn_output_w || !b.ffn_norm_w ||
            !b.ffn_gate_w || !b.ffn_up_w || !b.ffn_down_w) {
            fprintf(stderr, "orpheus: missing tensor in layer %u\n", i);
            return false;
        }
    }
    return true;
}

static void load_metadata(orpheus_context* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.n_layers = core_gguf::kv_u32(g, "orpheus.talker.n_layers", hp.n_layers);
    hp.d_model = core_gguf::kv_u32(g, "orpheus.talker.d_model", hp.d_model);
    hp.n_heads = core_gguf::kv_u32(g, "orpheus.talker.n_heads", hp.n_heads);
    hp.n_kv_heads = core_gguf::kv_u32(g, "orpheus.talker.n_kv_heads", hp.n_kv_heads);
    hp.head_dim = core_gguf::kv_u32(g, "orpheus.talker.head_dim", hp.head_dim);
    hp.ff_dim = core_gguf::kv_u32(g, "orpheus.talker.ff_dim", hp.ff_dim);
    hp.vocab_size = core_gguf::kv_u32(g, "orpheus.talker.vocab_size", hp.vocab_size);
    hp.max_pos = core_gguf::kv_u32(g, "orpheus.talker.max_pos", hp.max_pos);
    hp.rope_theta = core_gguf::kv_f32(g, "orpheus.talker.rope_theta", hp.rope_theta);
    hp.rms_norm_eps = core_gguf::kv_f32(g, "orpheus.talker.rms_norm_eps", hp.rms_norm_eps);

    hp.audio_start = core_gguf::kv_u32(g, "orpheus.audio_start_token", hp.audio_start);
    hp.audio_pre_end = core_gguf::kv_u32(g, "orpheus.audio_pre_end_token", hp.audio_pre_end);
    hp.audio_end_a = core_gguf::kv_u32(g, "orpheus.audio_end_a_token", hp.audio_end_a);
    hp.audio_end_b = core_gguf::kv_u32(g, "orpheus.audio_end_b_token", hp.audio_end_b);
    hp.audio_end = core_gguf::kv_u32(g, "orpheus.audio_end_token", hp.audio_end);
    hp.custom_token_offset = core_gguf::kv_u32(g, "orpheus.custom_token_offset", hp.custom_token_offset);
    hp.custom_token_count = core_gguf::kv_u32(g, "orpheus.custom_token_count", hp.custom_token_count);

    hp.super_frame_slots = core_gguf::kv_u32(g, "orpheus.snac.super_frame_slots", hp.super_frame_slots);
    hp.cb_count = core_gguf::kv_u32(g, "orpheus.snac.codebook_count", hp.cb_count);
    hp.cb_size = core_gguf::kv_u32(g, "orpheus.snac.codebook_size", hp.cb_size);

    hp.tts_model_type = core_gguf::kv_str(g, "orpheus.tts_model_type", hp.tts_model_type.c_str());
    hp.spk_names = core_gguf::kv_str_array(g, "orpheus.spk_names");

    // Tokenizer (Llama-3.2 BPE in tokenizer.ggml.tokens / .merges).
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

} // namespace

// ---------------------------------------------------------------------------
// Public C ABI
// ---------------------------------------------------------------------------

extern "C" struct orpheus_context_params orpheus_context_default_params(void) {
    orpheus_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.0f;
    p.max_audio_tokens = 0;
    return p;
}

extern "C" struct orpheus_context* orpheus_init_from_file(const char* path_model,
                                                          struct orpheus_context_params params) {
    auto* c = new orpheus_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Pass 1: hparams + vocab.
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
                "orpheus: variant=%s  talker=%uL d=%u h=%u/%u hd=%u ff=%u vocab=%u\n"
                "orpheus: rope_theta=%.0f  custom_token_offset=%u count=%u  speakers=%zu\n",
                c->hp.tts_model_type.c_str(), c->hp.n_layers, c->hp.d_model, c->hp.n_heads, c->hp.n_kv_heads,
                c->hp.head_dim, c->hp.ff_dim, c->hp.vocab_size, (double)c->hp.rope_theta, c->hp.custom_token_offset,
                c->hp.custom_token_count, c->hp.spk_names.size());
    }

    // Backend selection.
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "orpheus: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "orpheus", wl)) {
        fprintf(stderr, "orpheus: failed to load weights from '%s'\n", path_model);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_talker(c)) {
        fprintf(stderr, "orpheus: tensor binding failed\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler — sized for the worst-case talker prefill graph.
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

    // Default to first baked speaker for fixed_speaker variants.
    if (c->hp.tts_model_type == "fixed_speaker" && !c->hp.spk_names.empty()) {
        c->active_speaker = 0;
    }
    return c;
}

extern "C" void orpheus_free(struct orpheus_context* ctx) {
    delete ctx;
}

extern "C" void orpheus_set_n_threads(struct orpheus_context* ctx, int n_threads) {
    if (!ctx) {
        return;
    }
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
}

extern "C" int orpheus_set_codec_path(struct orpheus_context* ctx, const char* path) {
    if (!ctx || !path) {
        return -1;
    }
    ctx->snac_codec_path = path;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: codec path set to '%s' (SNAC C++ decoder lands in slice b)\n", path);
    }
    return 0;
}

extern "C" int orpheus_n_speakers(struct orpheus_context* ctx) {
    if (!ctx) {
        return 0;
    }
    return (int)ctx->hp.spk_names.size();
}

extern "C" const char* orpheus_get_speaker_name(struct orpheus_context* ctx, int i) {
    if (!ctx || i < 0 || (size_t)i >= ctx->hp.spk_names.size()) {
        return nullptr;
    }
    return ctx->hp.spk_names[i].c_str();
}

extern "C" int orpheus_set_speaker_by_name(struct orpheus_context* ctx, const char* name) {
    if (!ctx || !name) {
        return -1;
    }
    const std::string target = lower(name);
    for (size_t i = 0; i < ctx->hp.spk_names.size(); i++) {
        if (lower(ctx->hp.spk_names[i]) == target) {
            ctx->active_speaker = (int)i;
            return 0;
        }
    }
    return -2;
}

extern "C" int orpheus_is_fixed_speaker(struct orpheus_context* ctx) {
    return (ctx && ctx->hp.tts_model_type == "fixed_speaker") ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Talker forward (slice c)
// ---------------------------------------------------------------------------

namespace {

static bool kv_alloc(orpheus_context* c, int max_ctx) {
    if (c->kv_k) {
        return true;
    }
    const auto& hp = c->hp;
    const int hd = (int)hp.head_dim;
    const int n_kv = (int)hp.n_kv_heads;
    const int nl = (int)hp.n_layers;
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    c->kv_ctx = ggml_init(kp);
    c->kv_k = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    c->kv_v = ggml_new_tensor_4d(c->kv_ctx, GGML_TYPE_F16, hd, max_ctx, n_kv, nl);
    ggml_set_name(c->kv_k, "kv_k");
    ggml_set_name(c->kv_v, "kv_v");
    const size_t kb = ggml_nbytes(c->kv_k), vb = ggml_nbytes(c->kv_v);
    c->kv_buf = ggml_backend_alloc_buffer(c->backend, kb + vb);
    if (!c->kv_buf) {
        fprintf(stderr, "orpheus: kv alloc failed\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(c->kv_buf);
    ggml_backend_tensor_alloc(c->kv_buf, c->kv_k, base);
    ggml_backend_tensor_alloc(c->kv_buf, c->kv_v, base + kb);
    c->kv_max_ctx = max_ctx;
    if (c->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: kv cache %d MiB (hd=%d max=%d n_kv=%d nl=%d)\n", (int)((kb + vb) / 1048576), hd,
                max_ctx, n_kv, nl);
    }
    return true;
}

// Token embedding lookup graph: input_ids (T,) → embeds (d_model, T).
static ggml_cgraph* build_graph_embed(orpheus_context* c, int n_tokens) {
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

// Llama-3.2 talker block stack with KV-cached self-attention + SwiGLU.
//   inputs_embeds (d, T)    → logits (vocab,)
//   positions     (T,)      I32 absolute positions
//   causal_mask   (Lk, T)   F16 (only when T > 1)
static ggml_cgraph* build_graph_talker_kv(orpheus_context* c, int n_past, int n_tokens) {
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
        /*qk_norm_eps*/ 0.0f, // Llama-3 has no Q/K norm
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        const auto& b = c->talker.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w,
                                                    /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions,
                                                    (T == 1) ? nullptr : causal_mask, c->kv_k, c->kv_v, (int)il, n_past,
                                                    kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, c->talker.output_norm_w);
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, c->talker.output_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

static float* embed_tokens(orpheus_context* c, const int32_t* ids, int n) {
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

// Returns logits at the last position (vocab,), malloc'd. Caller frees.
static float* run_talker_kv(orpheus_context* c, const float* embeds, int n_tokens, int n_past) {
    if (n_past + n_tokens > c->kv_max_ctx) {
        fprintf(stderr, "orpheus: kv overflow (%d+%d > %d)\n", n_past, n_tokens, c->kv_max_ctx);
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
        fprintf(stderr, "orpheus: failed to alloc talker_kv graph\n");
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
        fprintf(stderr, "orpheus: talker_kv compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    float* r = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(out, r, 0, (size_t)vocab * sizeof(float));
    return r;
}

// Greedy argmax over the vocab.
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

// Top-k + temperature sampler. xorshift64* RNG via *rng_state. Falls back
// to argmax when temperature == 0.
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
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
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

// Build the Orpheus talker prompt:
//   [audio_start=128259, BOS=128000, ...tokenize("{name}: {text}")...,
//    eot_id=128009, audio_eot=128260, audio_eom=128261, audio_end=128257]
// Verbatim from canopyai/Orpheus-TTS:engine_class.py:_format_prompt — the
// HF tokenizer is called with `add_special_tokens=True` (default), which
// inserts the Llama-3 BOS (128000) at the start of `prompt_tokens.input_ids`,
// then engine_class.py prepends `start_token` and appends `end_tokens`.
static std::vector<int32_t> build_prompt_ids(orpheus_context* c, const std::string& text) {
    const auto& hp = c->hp;
    std::string adapted;
    if (hp.tts_model_type == "fixed_speaker" && c->active_speaker >= 0 &&
        c->active_speaker < (int)hp.spk_names.size()) {
        adapted = hp.spk_names[c->active_speaker] + ": " + text;
    } else {
        adapted = text;
    }
    std::vector<int32_t> body = core_bpe::tokenize_simple(c->vocab.token_to_id, c->vocab.merge_rank, adapted);
    std::vector<int32_t> out;
    out.reserve(body.size() + 6);
    out.push_back((int32_t)hp.audio_start);
    out.push_back(128000); // <|begin_of_text|> — Llama-3 BOS, added by HF tokenizer
    out.insert(out.end(), body.begin(), body.end());
    out.push_back((int32_t)hp.audio_pre_end);
    out.push_back((int32_t)hp.audio_end_a);
    out.push_back((int32_t)hp.audio_end_b);
    out.push_back((int32_t)hp.audio_end);
    return out;
}

} // namespace

extern "C" int32_t* orpheus_synthesize_codes(struct orpheus_context* ctx, const char* text, int* out_n) {
    if (out_n) {
        *out_n = 0;
    }
    if (!ctx || !text) {
        return nullptr;
    }
    const auto& hp = ctx->hp;

    std::vector<int32_t> prompt_ids = build_prompt_ids(ctx, text);
    if (prompt_ids.empty()) {
        return nullptr;
    }

    const int max_audio = ctx->params.max_audio_tokens > 0 ? ctx->params.max_audio_tokens : 8192;
    const int kv_need = (int)prompt_ids.size() + max_audio + 8;
    if (!kv_alloc(ctx, kv_need)) {
        return nullptr;
    }
    // Reset the cache by treating n_past=0 — the K/V buffer doesn't need to
    // be zeroed because the causal mask plus kv_self_attn's own bookkeeping
    // restrict reads to [0, n_past+T).
    int n_past = 0;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: prompt %d tokens (max_audio=%d)\n", (int)prompt_ids.size(), max_audio);
    }

    // Prefill.
    float* prompt_embeds = embed_tokens(ctx, prompt_ids.data(), (int)prompt_ids.size());
    if (!prompt_embeds) {
        return nullptr;
    }
    float* logits = run_talker_kv(ctx, prompt_embeds, (int)prompt_ids.size(), n_past);
    free(prompt_embeds);
    if (!logits) {
        return nullptr;
    }
    n_past += (int)prompt_ids.size();

    // AR loop. Sample a token from the last logits, append, decode-step,
    // until we see audio_end (128257) outside the codec block, or until
    // we hit max_audio_tokens. We do NOT stop on audio_end_b/audio_pre_end:
    // for the unsloth/canopylabs ID layout, 128261/128009 are either
    // Llama-3 specials or text_N<10 reserved markers that the reference
    // `tokens_decoder` filters silently rather than terminating on.
    std::vector<int32_t> custom_ids;
    custom_ids.reserve(max_audio);
    const int top_k = 50; // engine_class.py default sampling shape
    const bool debug = ctx->params.verbosity >= 2 || std::getenv("ORPHEUS_DEBUG") != nullptr;
    int n_dropped = 0;
    int first_tok = -1;
    int last_tok = -1;
    for (int step = 0; step < max_audio; step++) {
        int tok = sample_logits(logits, (int)hp.vocab_size, ctx->params.temperature, top_k, &ctx->rng_state);
        free(logits);
        logits = nullptr;
        if (first_tok < 0) {
            first_tok = tok;
        }
        last_tok = tok;

        if (debug && step < 16) {
            fprintf(stderr, "orpheus[ar]: step=%d tok=%d (custom=%d cb_relative=%d)\n", step, tok,
                    (int)(tok >= (int)hp.custom_token_offset &&
                          tok < (int)hp.custom_token_offset + (int)hp.custom_token_count),
                    tok - (int)hp.custom_token_offset);
        }

        if (tok == (int)hp.audio_end) {
            break;
        }
        if (tok >= (int)hp.custom_token_offset && tok < (int)hp.custom_token_offset + (int)hp.custom_token_count) {
            custom_ids.push_back(tok);
        } else {
            n_dropped++;
            if (n_dropped > 4) {
                // The model fell out of the codec block — definitely done.
                break;
            }
        }

        // Decode-step.
        int32_t single = (int32_t)tok;
        float* step_emb = embed_tokens(ctx, &single, 1);
        if (!step_emb) {
            break;
        }
        logits = run_talker_kv(ctx, step_emb, 1, n_past);
        free(step_emb);
        if (!logits) {
            break;
        }
        n_past++;
    }
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: AR first_tok=%d last_tok=%d non_custom_dropped=%d\n", first_tok, last_tok, n_dropped);
    }
    if (logits) {
        free(logits);
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: AR emitted %d codec tokens (%d super-frames)\n", (int)custom_ids.size(),
                (int)custom_ids.size() / (int)hp.super_frame_slots);
    }

    // Trim to a whole number of super-frames so the de-interleave is clean.
    const int per_sf = (int)hp.super_frame_slots; // 7
    int n_keep = (int)custom_ids.size() - ((int)custom_ids.size() % per_sf);
    if (n_keep <= 0) {
        return nullptr;
    }
    int32_t* out = (int32_t*)malloc((size_t)n_keep * sizeof(int32_t));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, custom_ids.data(), (size_t)n_keep * sizeof(int32_t));
    if (out_n) {
        *out_n = n_keep;
    }
    return out;
}

extern "C" float* orpheus_synthesize(struct orpheus_context* ctx, const char* text, int* out_n_samples) {
    if (out_n_samples) {
        *out_n_samples = 0;
    }
    if (!ctx || !text) {
        return nullptr;
    }
    if (ctx->snac_codec_path.empty()) {
        fprintf(stderr, "orpheus: no SNAC codec path set — call orpheus_set_codec_path first\n");
        return nullptr;
    }
    if (!ctx->snac_dec) {
        snac_decoder_params sp = snac_decoder_default_params();
        sp.n_threads = ctx->n_threads;
        sp.verbosity = ctx->params.verbosity;
        sp.use_gpu = ctx->params.use_gpu;
        ctx->snac_dec = snac_decoder_init_from_file(ctx->snac_codec_path.c_str(), sp);
        if (!ctx->snac_dec) {
            fprintf(stderr, "orpheus: failed to load SNAC codec from '%s'\n", ctx->snac_codec_path.c_str());
            return nullptr;
        }
    }

    int n_codes = 0;
    int32_t* codes = orpheus_synthesize_codes(ctx, text, &n_codes);
    if (!codes || n_codes <= 0) {
        return nullptr;
    }

    const auto& hp = ctx->hp;
    const int per_sf = (int)hp.super_frame_slots; // 7
    const int n_sf = n_codes / per_sf;

    // De-interleave per the canonical 7-slot super-frame layout
    // (orpheus_tts/decoder.py:convert_to_audio).
    //   slot 0       → c0
    //   slots 1, 4   → c1
    //   slots 2,3,5,6→ c2
    // For each LM token id, recover the codebook entry:
    //   text_N  = lm_id - custom_token_offset       ( = "N" from "<custom_token_N>" )
    //   cb      = text_N - 10 - (slot_idx * 4096)
    // The "-10" skips the first 10 reserved entries of the custom token block.
    std::vector<int32_t> c0, c1, c2;
    c0.reserve(n_sf);
    c1.reserve(2 * n_sf);
    c2.reserve(4 * n_sf);
    bool clamped = false;
    for (int j = 0; j < n_sf; j++) {
        for (int s = 0; s < per_sf; s++) {
            int32_t lm_id = codes[j * per_sf + s];
            int32_t text_n = lm_id - (int32_t)hp.custom_token_offset;
            int32_t cb = text_n - 10 - (s * (int32_t)hp.cb_size);
            if (cb < 0 || cb >= (int32_t)hp.cb_size) {
                // Clamp out-of-range entries to a safe value so the decoder
                // doesn't index OOB. This shouldn't fire for a healthy model.
                cb = 0;
                clamped = true;
            }
            switch (s) {
                case 0: c0.push_back(cb); break;
                case 1: c1.push_back(cb); break;
                case 2: c2.push_back(cb); break;
                case 3: c2.push_back(cb); break;
                case 4: c1.push_back(cb); break;
                case 5: c2.push_back(cb); break;
                case 6: c2.push_back(cb); break;
            }
        }
    }
    free(codes);

    if (clamped && ctx->params.verbosity >= 1) {
        fprintf(stderr, "orpheus: WARN — some codebook entries out of range; clamped to 0\n");
    }

    int n_pcm = 0;
    float* pcm = snac_decoder_decode(ctx->snac_dec, c0.data(), (int)c0.size(), c1.data(), (int)c1.size(), c2.data(),
                                     (int)c2.size(), &n_pcm);
    if (!pcm || n_pcm <= 0) {
        return nullptr;
    }
    if (out_n_samples) {
        *out_n_samples = n_pcm;
    }
    return pcm;
}

extern "C" void orpheus_codes_free(int32_t* codes) {
    std::free(codes);
}

extern "C" void orpheus_pcm_free(float* pcm) {
    std::free(pcm);
}
