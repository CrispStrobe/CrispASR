// breeze_tts_2.cpp -- Breeze TTS 2 runtime (issue #412).
//
// See breeze_tts_2.h for the architecture summary and the licence position.
// This file implements the three transformer stacks; the codec is the
// already-shipped qwen3-tts tokenizer, driven through qwen3_tts.h.
//
// THE THREE CONFIG DECOYS. Reading config.json at face value builds the wrong
// model three separate times, and each wrong build runs and produces
// plausible-sounding audio:
//
//   1. `use_bidirectional_attention: false` is a MISNOMER. The registered
//      implementation is the checkpoint's own t5gemma2_compat.py, which sets
//      is_causal = False and passes causal=False into flash-attn. The text
//      encoder is BIDIRECTIONAL, and its sliding window is SYMMETRIC —
//      [i-255, i+256], not the usual left-only.
//   2. Top-level `rope_theta: 500000` and the llama3 `rope_scaling` beside it
//      are DEAD for the backbone. breeze_backbone_factory.py builds Qwen3
//      layers from the NESTED backbone_config: theta 1e6, scaling null.
//   3. Top-level `rms_norm_eps: 1e-5` is likewise dead for the backbone, which
//      takes 1e-6 from that same nested block. The 1e-5 is real for the DEPTH
//      DECODER only.
//
// All three are read from GGUF KV keys the converter writes explicitly, so
// they cannot be re-derived incorrectly here — but a reader who "fixes" a
// constant against config.json will reintroduce them.

#include "breeze_tts_2.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/crispasr_env.h"
#include "core/ffn.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "qwen3_tts.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ===========================================================================
// Bench instrumentation — CRISPASR_BREEZE_BENCH=1.
// ===========================================================================

static bool breeze_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = crispasr_env::get("CRISPASR_BREEZE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct breeze_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit breeze_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~breeze_bench_stage() {
        if (!breeze_bench_enabled())
            return;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "  breeze_bench: %-24s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct breeze_hparams {
    // ---- text encoder (T5Gemma2) ----
    uint32_t te_n_layers = 26;
    uint32_t te_d_model = 1152;
    uint32_t te_n_heads = 4;
    uint32_t te_n_kv_heads = 1;
    uint32_t te_head_dim = 256;
    uint32_t te_ff_dim = 6912;
    uint32_t te_vocab_size = 262158;
    uint32_t te_max_pos = 8192;
    float te_rms_norm_eps = 1e-6f;
    float te_embed_scale = 33.941125f; // sqrt(1152)
    float te_attn_scale = 0.0625f;     // query_pre_attn_scalar ** -0.5 = 256^-0.5
    uint32_t te_eoi_token_index = 256000;
    uint32_t te_sliding_window = 512;
    // SYMMETRIC window, not causal: q attends k in [q-(left-1), q+(right-1)].
    uint32_t te_win_left = 256;  // dist = q-k in [0, 255]
    uint32_t te_win_right = 257; // dist in [-256, -1]
    float te_rope_theta_sliding = 10000.0f;
    float te_rope_theta_full = 1000000.0f;
    float te_rope_factor_full = 8.0f; // "linear": inv_freq /= factor
    // 1 = full_attention, 0 = sliding_attention, indexed by layer.
    std::vector<int> te_layer_full;
    // True when the converter has already folded Gemma's `1 + w` into the
    // stored norm weights (see load_metadata).
    bool te_norm_offset_baked = false;
    bool te_norm_unit_offset = true;

    // ---- backbone (Qwen3) — NESTED config only ----
    uint32_t bb_n_layers = 28;
    uint32_t bb_d_model = 2048;
    uint32_t bb_n_heads = 16;
    uint32_t bb_n_kv_heads = 8;
    uint32_t bb_head_dim = 128;
    uint32_t bb_ff_dim = 6144;
    uint32_t bb_max_pos = 40960;
    float bb_rope_theta = 1000000.0f; // NOT the top-level 500000
    float bb_rms_norm_eps = 1e-6f;    // NOT the top-level 1e-5

    // ---- depth decoder ----
    uint32_t dd_n_layers = 12;
    uint32_t dd_d_model = 1024;
    uint32_t dd_n_heads = 8;
    uint32_t dd_n_kv_heads = 2;
    uint32_t dd_head_dim = 128;
    uint32_t dd_ff_dim = 8192;
    uint32_t dd_max_pos = 33;
    uint32_t dd_vocab_size = 2051;
    uint32_t dd_backbone_hidden = 2048;
    float dd_rope_theta = 500000.0f;
    float dd_rms_norm_eps = 1e-5f; // the top-level value IS correct here
    float dd_rope_scaling_factor = 32.0f;
    float dd_rope_low_freq_factor = 0.001953125f;
    float dd_rope_high_freq_factor = 0.0078125f;
    uint32_t dd_rope_orig_max_pos = 16;

    // ---- audio / vocab ----
    uint32_t num_codebooks = 16;
    uint32_t audio_vocab_size = 2051;
    uint32_t hidden_size = 2048;
    uint32_t audio_token_id = 262144;     // <|AUDIO|>
    uint32_t audio_eos_token_id = 262145; // <|audio_eos|>
    uint32_t codebook_pad_token_id = 2050;
    uint32_t codebook_eos_token_id = 0;
    uint32_t backbone_eos_token_id = 2051; // the extra lm_head class
    uint32_t lm_head_out = 2052;
    uint32_t bos_token_id = 2;
    uint32_t eos_token_id = 1;
    uint32_t pad_token_id = 0;
    // Codec ids in [lo, hi) are RESERVED and must be masked out of the
    // sampler. Sampling into them yields codes the codec cannot decode.
    uint32_t reserved_lo = 2048;
    uint32_t reserved_hi = 2051;
    uint32_t ins_bos_token_id = 262156;
    uint32_t ins_eos_token_id = 262157;
    std::vector<int32_t> speaker_token_ids;
    std::vector<std::string> speaker_token_names;

    // ---- sampling defaults ----
    float s_temperature = 0.9f;
    float s_depth_temperature = 0.9f;
    uint32_t s_top_k = 50;
    float s_top_p = 1.0f;
    uint32_t s_max_new_tokens = 750;
    float s_repetition_penalty = 1.1f;
    uint32_t s_max_seq_len = 2048;
};

// ===========================================================================
// Model structures
// ===========================================================================

// Gemma-shaped block: a norm on BOTH sides of each sublayer.
struct te_layer {
    ggml_tensor* attn_pre_norm_w = nullptr;
    ggml_tensor* attn_post_norm_w = nullptr;
    ggml_tensor* ffn_pre_norm_w = nullptr;
    ggml_tensor* ffn_post_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

// Llama/Qwen3-shaped block (backbone and depth decoder). q/k norms are
// present on the backbone (Qwen3) and ABSENT on the depth decoder — that
// asymmetry is real, not an oversight: BreezeAttention builds no norms.
struct llm_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct breeze_model {
    breeze_hparams hp;

    // text encoder
    ggml_tensor* te_token_embd_w = nullptr;
    ggml_tensor* te_eoi_embd = nullptr;
    ggml_tensor* te_output_norm_w = nullptr;
    ggml_tensor* te_proj_w = nullptr;
    std::vector<te_layer> te_layers;

    // backbone
    // ONE physical tensor, bound twice: config.tie_codebooks_embeddings ties
    // backbone_model.embed_tokens.embed_audio_tokens to
    // depth_decoder.model.embed_tokens, and the converter stores it once.
    ggml_tensor* audio_embd_w = nullptr;
    ggml_tensor* bb_output_norm_w = nullptr;
    ggml_tensor* bb_codebook0_head_w = nullptr;
    std::vector<llm_layer> bb_layers;

    // depth decoder
    ggml_tensor* dd_projection_w = nullptr;
    ggml_tensor* dd_output_norm_w = nullptr;
    std::vector<ggml_tensor*> dd_cb_heads; // 15 of them, pre-transposed
    std::vector<llm_layer> dd_layers;

    // tokenizer
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
};

} // namespace

// ===========================================================================
// Context
// ===========================================================================

struct breeze_tts_2_context {
    breeze_tts_2_context_params params{};
    int n_threads = 4;

    breeze_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Text-encoder KV scratch. The encoder is not autoregressive — every
    // segment is one full pass at n_past = 0 — so this is a scratch buffer that
    // core_attn::kv_self_attn happens to write through, not a cache with a
    // history. Sized to the longest segment we accept.
    ggml_context* te_kv_ctx = nullptr;
    ggml_backend_buffer_t te_kv_buf = nullptr;
    ggml_tensor* te_kv_k = nullptr;
    ggml_tensor* te_kv_v = nullptr;
    int te_kv_max_ctx = 0;

    // Backbone KV cache. The trailing dim is n_layers * n_branch, so CFG
    // branches get independent histories without touching core_attn: it takes
    // `il` as the trailing index, and il = layer * n_branch + branch is just
    // arithmetic at the call site.
    ggml_context* bb_kv_ctx = nullptr;
    ggml_backend_buffer_t bb_kv_buf = nullptr;
    ggml_tensor* bb_kv_k = nullptr;
    ggml_tensor* bb_kv_v = nullptr;
    int bb_kv_max_ctx = 0;
    int n_branch = 1;

    // Depth-decoder KV cache: reset every frame, and per branch for the same
    // reason (the depth combine is also per-branch, per codebook).
    ggml_context* dd_kv_ctx = nullptr;
    ggml_backend_buffer_t dd_kv_buf = nullptr;
    ggml_tensor* dd_kv_k = nullptr;
    ggml_tensor* dd_kv_v = nullptr;

    // llama3 RoPE frequency factors for the depth decoder, precomputed.
    std::vector<float> dd_rope_freq_factors;
    // Constant `factor` vector implementing the text encoder's
    // rope_type="linear" on its full-attention layers (see build_te_graph).
    std::vector<float> te_rope_linear_factors;
    // eoi_embedding, read to the host once so the substitution can be
    // expressed as an additive correction (see build_te_graph).
    std::vector<float> te_eoi_host;

    // Codec companion (qwen3-tts-tokenizer-12hz). Optional: without it the
    // runtime still produces codes, which is all the diff harness needs.
    qwen3_tts_context* codec = nullptr;

    uint64_t rng_state = 0xbadc0ffee0ddf00dULL;
    int max_new_tokens = 0;

    // Diagnostics: codes from the most recent synthesis.
    std::vector<std::vector<int32_t>> last_codes;

    ~breeze_tts_2_context() {
        if (sched)
            ggml_backend_sched_free(sched);
        if (te_kv_buf)
            ggml_backend_buffer_free(te_kv_buf);
        if (te_kv_ctx)
            ggml_free(te_kv_ctx);
        if (bb_kv_buf)
            ggml_backend_buffer_free(bb_kv_buf);
        if (bb_kv_ctx)
            ggml_free(bb_kv_ctx);
        if (dd_kv_buf)
            ggml_backend_buffer_free(dd_kv_buf);
        if (dd_kv_ctx)
            ggml_free(dd_kv_ctx);
        if (codec)
            qwen3_tts_free(codec);
        if (model.buf_w)
            core_gguf::release_weight_buffer(model.buf_w);
        if (model.ctx_w)
            ggml_free(model.ctx_w);
        if (backend_cpu && backend_cpu != backend)
            ggml_backend_free(backend_cpu);
        if (backend)
            ggml_backend_free(backend);
    }
};

namespace {

// ===========================================================================
// Metadata
// ===========================================================================

static std::vector<int> kv_i32_array(gguf_context* g, const char* key, int n_expected) {
    std::vector<int> out;
    const int k = gguf_find_key(g, key);
    if (k < 0)
        return out;
    const int n = gguf_get_arr_n(g, k);
    const auto* d = (const int32_t*)gguf_get_arr_data(g, k);
    if (!d)
        return out;
    const int take = n_expected > 0 ? std::min(n, n_expected) : n;
    out.resize((size_t)take);
    for (int i = 0; i < take; i++)
        out[(size_t)i] = (int)d[i];
    return out;
}

static void load_metadata(breeze_tts_2_context* c, gguf_context* g) {
    auto& hp = c->model.hp;

#define U32(field, key) hp.field = core_gguf::kv_u32(g, key, hp.field)
#define F32(field, key) hp.field = core_gguf::kv_f32(g, key, hp.field)

    // ---- text encoder ----
    U32(te_n_layers, "breeze.te.n_layers");
    U32(te_d_model, "breeze.te.d_model");
    U32(te_n_heads, "breeze.te.n_heads");
    U32(te_n_kv_heads, "breeze.te.n_kv_heads");
    U32(te_head_dim, "breeze.te.head_dim");
    U32(te_ff_dim, "breeze.te.ff_dim");
    U32(te_vocab_size, "breeze.te.vocab_size");
    U32(te_max_pos, "breeze.te.max_pos");
    F32(te_rms_norm_eps, "breeze.te.rms_norm_eps");
    F32(te_embed_scale, "breeze.te.embed_scale");
    F32(te_attn_scale, "breeze.te.attn_scale");
    U32(te_eoi_token_index, "breeze.te.eoi_token_index");
    U32(te_sliding_window, "breeze.te.sliding_window");
    U32(te_win_left, "breeze.te.sliding_window_left");
    U32(te_win_right, "breeze.te.sliding_window_right");
    F32(te_rope_theta_sliding, "breeze.te.rope_theta_sliding");
    F32(te_rope_theta_full, "breeze.te.rope_theta_full");
    F32(te_rope_factor_full, "breeze.te.rope_factor_full");
    hp.te_norm_unit_offset = core_gguf::kv_bool(g, "breeze.te.norm_unit_offset", hp.te_norm_unit_offset);
    hp.te_norm_offset_baked = core_gguf::kv_bool(g, "breeze.te.norm_weights_pre_offset", hp.te_norm_offset_baked);

    hp.te_layer_full = kv_i32_array(g, "breeze.te.layer_types", (int)hp.te_n_layers);
    if (hp.te_layer_full.size() != (size_t)hp.te_n_layers) {
        // Without the real pattern every layer would silently become sliding,
        // which is a plausible-sounding encoder with the wrong long-range
        // context on 4 of 26 layers. Fail loudly instead.
        fprintf(stderr,
                "breeze_tts_2: breeze.te.layer_types has %zu entries, expected %u — "
                "the sliding/full pattern is not recoverable from anything else in the file\n",
                hp.te_layer_full.size(), hp.te_n_layers);
        hp.te_layer_full.assign((size_t)hp.te_n_layers, 0);
    }

    // ---- backbone ----
    U32(bb_n_layers, "breeze.bb.n_layers");
    U32(bb_d_model, "breeze.bb.d_model");
    U32(bb_n_heads, "breeze.bb.n_heads");
    U32(bb_n_kv_heads, "breeze.bb.n_kv_heads");
    U32(bb_head_dim, "breeze.bb.head_dim");
    U32(bb_ff_dim, "breeze.bb.ff_dim");
    U32(bb_max_pos, "breeze.bb.max_pos");
    F32(bb_rope_theta, "breeze.bb.rope_theta");
    F32(bb_rms_norm_eps, "breeze.bb.rms_norm_eps");

    // ---- depth decoder ----
    U32(dd_n_layers, "breeze.dd.n_layers");
    U32(dd_d_model, "breeze.dd.d_model");
    U32(dd_n_heads, "breeze.dd.n_heads");
    U32(dd_n_kv_heads, "breeze.dd.n_kv_heads");
    U32(dd_head_dim, "breeze.dd.head_dim");
    U32(dd_ff_dim, "breeze.dd.ff_dim");
    U32(dd_max_pos, "breeze.dd.max_pos");
    U32(dd_vocab_size, "breeze.dd.vocab_size");
    U32(dd_backbone_hidden, "breeze.dd.backbone_hidden");
    F32(dd_rope_theta, "breeze.dd.rope_theta");
    F32(dd_rms_norm_eps, "breeze.dd.rms_norm_eps");
    F32(dd_rope_scaling_factor, "breeze.dd.rope_scaling_factor");
    F32(dd_rope_low_freq_factor, "breeze.dd.rope_low_freq_factor");
    F32(dd_rope_high_freq_factor, "breeze.dd.rope_high_freq_factor");
    U32(dd_rope_orig_max_pos, "breeze.dd.rope_orig_max_pos");

    // ---- audio / vocab ----
    U32(num_codebooks, "breeze.num_codebooks");
    U32(audio_vocab_size, "breeze.audio_vocab_size");
    U32(hidden_size, "breeze.hidden_size");
    U32(audio_token_id, "breeze.audio_token_id");
    U32(audio_eos_token_id, "breeze.audio_eos_token_id");
    U32(codebook_pad_token_id, "breeze.codebook_pad_token_id");
    U32(codebook_eos_token_id, "breeze.codebook_eos_token_id");
    U32(backbone_eos_token_id, "breeze.backbone_eos_token_id");
    U32(lm_head_out, "breeze.lm_head_out");
    U32(bos_token_id, "breeze.bos_token_id");
    U32(eos_token_id, "breeze.eos_token_id");
    U32(pad_token_id, "breeze.pad_token_id");
    U32(reserved_lo, "breeze.reserved_codec_id_lo");
    U32(reserved_hi, "breeze.reserved_codec_id_hi");
    U32(ins_bos_token_id, "breeze.ins_bos_token_id");
    U32(ins_eos_token_id, "breeze.ins_eos_token_id");

    // ---- sampling ----
    F32(s_temperature, "breeze.sampling.temperature");
    F32(s_depth_temperature, "breeze.sampling.depth_temperature");
    U32(s_top_k, "breeze.sampling.top_k");
    F32(s_top_p, "breeze.sampling.top_p");
    U32(s_max_new_tokens, "breeze.sampling.max_new_tokens");
    F32(s_repetition_penalty, "breeze.sampling.repetition_penalty");
    U32(s_max_seq_len, "breeze.sampling.max_seq_len");

#undef U32
#undef F32

    {
        const std::vector<int> spk = kv_i32_array(g, "breeze.speaker_token_ids", 0);
        hp.speaker_token_ids.assign(spk.begin(), spk.end());
        hp.speaker_token_names = core_gguf::kv_str_array(g, "breeze.speaker_token_names");
    }

    // ---- tokenizer ----
    auto tok = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if (!tok.empty()) {
        c->model.id_to_token = std::move(tok);
        c->model.token_to_id.reserve(c->model.id_to_token.size());
        for (int i = 0; i < (int)c->model.id_to_token.size(); i++)
            c->model.token_to_id[c->model.id_to_token[(size_t)i]] = i;
    }
    auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
    for (size_t i = 0; i < merges.size(); i++)
        c->model.merge_rank[merges[i]] = (int32_t)i;
}

// ===========================================================================
// Weight binding
// ===========================================================================

static bool bind_weights(breeze_tts_2_context* c) {
    auto& m = c->model;
    auto& hp = m.hp;
    auto& ts = c->tensors;
    const char* tag = "breeze_tts_2";

    // ---- text encoder ----
    m.te_token_embd_w = core_gguf::require(ts, "te.token_embd.weight", tag);
    m.te_output_norm_w = core_gguf::require(ts, "te.output_norm.weight", tag);
    m.te_proj_w = core_gguf::require(ts, "te_proj.weight", tag);
    {
        auto it = ts.find("te.eoi_embd");
        m.te_eoi_embd = (it != ts.end()) ? it->second : nullptr;
    }

    m.te_layers.resize(hp.te_n_layers);
    for (uint32_t i = 0; i < hp.te_n_layers; i++) {
        auto& b = m.te_layers[i];
        char key[96];
#define TE_BIND(fld, sub)                                                                                              \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "te.blk.%u." sub ".weight", i);                                                \
        b.fld = core_gguf::require(ts, key, tag);                                                                      \
    } while (0)
        TE_BIND(attn_pre_norm_w, "attn_pre_norm");
        TE_BIND(attn_post_norm_w, "attn_post_norm");
        TE_BIND(ffn_pre_norm_w, "ffn_pre_norm");
        TE_BIND(ffn_post_norm_w, "ffn_post_norm");
        TE_BIND(attn_q_w, "attn_q");
        TE_BIND(attn_k_w, "attn_k");
        TE_BIND(attn_v_w, "attn_v");
        TE_BIND(attn_output_w, "attn_output");
        TE_BIND(attn_q_norm_w, "attn_q_norm");
        TE_BIND(attn_k_norm_w, "attn_k_norm");
        TE_BIND(ffn_gate_w, "ffn_gate");
        TE_BIND(ffn_up_w, "ffn_up");
        TE_BIND(ffn_down_w, "ffn_down");
#undef TE_BIND
        if (!b.attn_q_w || !b.ffn_down_w)
            return false;
    }

    // ---- backbone ----
    // ONE tensor, two bindings. The checkpoint does not contain
    // backbone_model.embed_tokens.embed_audio_tokens at all — it is tied to
    // depth_decoder.model.embed_tokens — and a converter that looked for the
    // backbone name, found nothing and carried on would produce a GGUF that
    // loads and then emits silence.
    m.audio_embd_w = core_gguf::require(ts, "backbone.audio_embd.weight", tag);
    m.bb_output_norm_w = core_gguf::require(ts, "backbone.output_norm.weight", tag);
    m.bb_codebook0_head_w = core_gguf::require(ts, "backbone.codebook0_head.weight", tag);
    if (!m.audio_embd_w || !m.bb_codebook0_head_w)
        return false;

    m.bb_layers.resize(hp.bb_n_layers);
    for (uint32_t i = 0; i < hp.bb_n_layers; i++) {
        auto& b = m.bb_layers[i];
        char key[96];
#define BB_BIND(fld, sub)                                                                                              \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "backbone.blk.%u." sub ".weight", i);                                          \
        b.fld = core_gguf::require(ts, key, tag);                                                                      \
    } while (0)
        BB_BIND(attn_norm_w, "attn_norm");
        BB_BIND(ffn_norm_w, "ffn_norm");
        BB_BIND(attn_q_w, "attn_q");
        BB_BIND(attn_k_w, "attn_k");
        BB_BIND(attn_v_w, "attn_v");
        BB_BIND(attn_output_w, "attn_output");
        BB_BIND(attn_q_norm_w, "attn_q_norm"); // Qwen3: present
        BB_BIND(attn_k_norm_w, "attn_k_norm");
        BB_BIND(ffn_gate_w, "ffn_gate");
        BB_BIND(ffn_up_w, "ffn_up");
        BB_BIND(ffn_down_w, "ffn_down");
#undef BB_BIND
        if (!b.attn_q_w || !b.ffn_down_w)
            return false;
    }

    // ---- depth decoder ----
    m.dd_projection_w = core_gguf::require(ts, "depth.projection.weight", tag);
    m.dd_output_norm_w = core_gguf::require(ts, "depth.output_norm.weight", tag);
    if (!m.dd_projection_w)
        return false;

    m.dd_layers.resize(hp.dd_n_layers);
    for (uint32_t i = 0; i < hp.dd_n_layers; i++) {
        auto& b = m.dd_layers[i];
        char key[96];
#define DD_BIND(fld, sub)                                                                                              \
    do {                                                                                                               \
        std::snprintf(key, sizeof(key), "depth.blk.%u." sub ".weight", i);                                             \
        b.fld = core_gguf::require(ts, key, tag);                                                                      \
    } while (0)
        DD_BIND(attn_norm_w, "attn_norm");
        DD_BIND(ffn_norm_w, "ffn_norm");
        DD_BIND(attn_q_w, "attn_q");
        DD_BIND(attn_k_w, "attn_k");
        DD_BIND(attn_v_w, "attn_v");
        DD_BIND(attn_output_w, "attn_output");
        DD_BIND(ffn_gate_w, "ffn_gate");
        DD_BIND(ffn_up_w, "ffn_up");
        DD_BIND(ffn_down_w, "ffn_down");
#undef DD_BIND
        // q_norm/k_norm stay null here ON PURPOSE — BreezeAttention builds
        // none. Binding them "just in case" would find nothing and pass a
        // null through, but stating it is cheaper than re-deriving it.
        b.attn_q_norm_w = nullptr;
        b.attn_k_norm_w = nullptr;
        if (!b.attn_q_w || !b.ffn_down_w)
            return false;
    }

    // 15 per-codebook heads, already transposed to [d_model, vocab] by the
    // converter so each is a plain mul_mat. CSM keeps the 3-D tensor and pays
    // a cont(transpose(slice)) on every depth step; there is no reason to.
    const int n_heads = (int)hp.num_codebooks - 1;
    m.dd_cb_heads.assign((size_t)n_heads, nullptr);
    for (int i = 0; i < n_heads; i++) {
        char key[64];
        std::snprintf(key, sizeof(key), "depth.cb_head.%d.weight", i);
        m.dd_cb_heads[(size_t)i] = core_gguf::require(ts, key, tag);
        if (!m.dd_cb_heads[(size_t)i])
            return false;
    }

    return m.te_token_embd_w && m.te_proj_w && m.bb_output_norm_w && m.dd_output_norm_w && m.te_output_norm_w;
}

// ===========================================================================
// RoPE helpers
// ===========================================================================

// llama3 scaling, as transformers computes it, expressed as the per-pair
// divisor ggml's `freq_factors` wants (ggml does theta/ff, so ff is
// inv_freq_original / inv_freq_scaled).
//
// The depth decoder applies this with original_max_position_embeddings = 16
// over a 33-slot sequence. That is unusual enough to look like a typo and is
// not one: nearly every frequency lands in the interpolation regime, and
// getting the ramp slightly wrong changes the codes for codebooks 8-15 while
// leaving 0-7 plausible — which sounds like a codec artefact, not a RoPE bug.
static std::vector<float> build_llama3_freq_factors(float theta, int head_dim, float factor, float low_freq_factor,
                                                    float high_freq_factor, int orig_max_pos) {
    if (factor <= 1.0f || orig_max_pos <= 0 || head_dim <= 0 || (head_dim % 2) != 0)
        return {};
    const int n_pairs = head_dim / 2;
    std::vector<float> f((size_t)n_pairs, 1.0f);
    const float old_ctx = (float)orig_max_pos;
    const float low_wavelen = old_ctx / low_freq_factor;
    const float high_wavelen = old_ctx / high_freq_factor;
    const float inv_factor = 1.0f / factor;
    for (int i = 0; i < n_pairs; i++) {
        const float inv_freq = std::pow(theta, -(2.0f * (float)i) / (float)head_dim);
        const float wavelen = 2.0f * (float)M_PI / inv_freq;
        float scaled = inv_freq;
        if (wavelen > low_wavelen) {
            scaled = inv_freq * inv_factor;
        } else if (wavelen >= high_wavelen) {
            const float smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            scaled = ((1.0f - smooth) * inv_freq * inv_factor) + (smooth * inv_freq);
        }
        f[(size_t)i] = inv_freq / scaled;
    }
    return f;
}

// The text encoder's full-attention layers use rope_type="linear", which in
// the reference means exactly `inv_freq /= factor` with attention_scaling 1.0
// — NOT llama3 and NOT yarn. ggml divides theta by freq_factors per pair, so a
// CONSTANT vector of `factor` reproduces it bit-for-bit, and no change to
// core_attn::kv_self_attn (which hardcodes freq_scale = 1) is needed.
static std::vector<float> build_linear_freq_factors(int head_dim, float factor) {
    if (head_dim <= 0 || (head_dim % 2) != 0 || factor == 1.0f)
        return {};
    return std::vector<float>((size_t)(head_dim / 2), factor);
}

// ===========================================================================
// KV caches
// ===========================================================================

static bool alloc_kv(breeze_tts_2_context* c, ggml_context** kctx, ggml_backend_buffer_t* kbuf, ggml_tensor** kk,
                     ggml_tensor** kv, int head_dim, int max_ctx, int n_kv_heads, int n_slots, const char* tag) {
    ggml_init_params gp = {ggml_tensor_overhead() * 2, nullptr, true};
    *kctx = ggml_init(gp);
    if (!*kctx)
        return false;
    const auto pair = core_attn::kv_dtype_pair_from_env(tag);
    *kk = ggml_new_tensor_4d(*kctx, pair.k, head_dim, max_ctx, n_kv_heads, n_slots);
    *kv = ggml_new_tensor_4d(*kctx, pair.v, head_dim, max_ctx, n_kv_heads, n_slots);
    ggml_backend_t be = core_attn::kv_backend_from_env(c->backend, c->backend_cpu, tag);
    *kbuf = ggml_backend_alloc_ctx_tensors(*kctx, be);
    if (!*kbuf) {
        ggml_free(*kctx);
        *kctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(*kbuf, 0);
    return true;
}

// ===========================================================================
// Tokenizer + prompt assembly
// ===========================================================================
//
// What the reference actually does (breeze_infer/templates.py::_prepare_one):
//   * every TEXT segment is tokenized with add_special_tokens=True, decoded
//     back to a string and re-encoded as part of the whole prompt — the net
//     effect of that round trip is a Gemma <bos> at the head of EVERY text
//     segment, including ones that sit mid-prompt;
//   * an AUDIO segment renders as "<|AUDIO|>" repeated once per codec frame,
//     followed by "<|audio_eos|>";
//   * text_ids_mask is true exactly over the text segments, and text_ids_len
//     records one length per text segment.
//
// The segment boundaries are load-bearing twice over: the text encoder runs
// each segment as its OWN sequence (concatenating them changes every hidden
// state), and the reference asserts that mask and lengths agree — an
// assertion this runtime does not get for free, so the prompt dump entry
// point exists to check it against the fixture instead.

// Tokenize a rendered segment, emitting the special-token ids directly rather
// than letting the BPE merge loop see their literal spelling. `[S0]` and
// `<ins_bos>` are single added tokens upstream; a whitespace-split
// pre-tokenizer would glue them to the following word and silently produce a
// different prompt.
static void tokenize_with_specials(const breeze_model& m, const std::string& text, std::vector<int32_t>& out) {
    struct Special {
        std::string tag;
        int32_t id;
    };
    std::vector<Special> specials;
    const auto& hp = m.hp;
    for (size_t i = 0; i < hp.speaker_token_names.size() && i < hp.speaker_token_ids.size(); i++)
        specials.push_back({hp.speaker_token_names[i], hp.speaker_token_ids[i]});
    specials.push_back({"<ins_bos>", (int32_t)hp.ins_bos_token_id});
    specials.push_back({"<ins_eos>", (int32_t)hp.ins_eos_token_id});
    specials.push_back({"<|AUDIO|>", (int32_t)hp.audio_token_id});
    specials.push_back({"<|audio_eos|>", (int32_t)hp.audio_eos_token_id});

    size_t pos = 0;
    std::string pending;
    auto flush = [&]() {
        if (pending.empty())
            return;
        // SentencePiece BPE, not GPT-2 byte-level: Gemma's normalizer turns
        // each space into U+2581 and merges across the whole string. Measured
        // against the reference fixture, tokenize_simple() turned 39 true
        // tokens into ~69 and made the prompt unmatchable.
        auto ids = core_bpe::tokenize_spm_bpe(m.token_to_id, m.merge_rank, pending);
        out.insert(out.end(), ids.begin(), ids.end());
        pending.clear();
    };
    while (pos < text.size()) {
        bool matched = false;
        for (const auto& sp : specials) {
            if (!sp.tag.empty() && text.compare(pos, sp.tag.size(), sp.tag) == 0) {
                flush();
                out.push_back(sp.id);
                pos += sp.tag.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            pending.push_back(text[pos]);
            pos++;
        }
    }
    flush();
}

struct PromptSegment {
    bool is_text = true;
    std::vector<int32_t> ids;
};

// Build the segment list for one CFG branch.
//   ref_frames > 0     -> insert the reference-audio segment
//   instruction != ""  -> wrap it in <ins_bos>/<ins_eos> before the text
// This covers all four upstream shapes: plain (_tts_plain_segments),
// instruction (_tts_instruction_segments), clone (_ref_clone_tata_segments)
// and edit (_ref_edit_tata_segments).
static std::vector<PromptSegment> build_segments(const breeze_model& m, const std::string& speaker,
                                                 const std::string& text, const std::string& ref_text, int ref_frames,
                                                 const std::string& instruction) {
    const auto& hp = m.hp;
    std::vector<PromptSegment> segs;
    const std::string prefix = speaker.empty() ? std::string() : ("[" + speaker + "]");

    auto text_segment = [&](const std::string& body) {
        PromptSegment s;
        s.is_text = true;
        // The <bos> that the add_special_tokens=True round trip leaves at the
        // head of every text segment.
        s.ids.push_back((int32_t)hp.bos_token_id);
        tokenize_with_specials(m, body, s.ids);
        segs.push_back(std::move(s));
    };

    if (ref_frames > 0) {
        text_segment(prefix + ref_text);
        PromptSegment a;
        a.is_text = false;
        a.ids.assign((size_t)ref_frames, (int32_t)hp.audio_token_id);
        a.ids.push_back((int32_t)hp.audio_eos_token_id);
        segs.push_back(std::move(a));
    }
    if (!instruction.empty())
        text_segment(prefix + "<ins_bos>" + instruction + "<ins_eos>" + text);
    else
        text_segment(prefix + text);
    return segs;
}

// ===========================================================================
// Graph: text encoder (T5Gemma2)
// ===========================================================================
//
// The combination this repo did not already have: BIDIRECTIONAL *and*
// windowed, symmetrically. Every other sliding-window path here is a causal
// decoder (left-only) and every other bidirectional path is unwindowed. The
// two traps that follow from that:
//
//   * full_attention layers (5, 11, 17, 23) take NO mask at all. Passing a
//     causal mask there produces a perfectly plausible encoder output with
//     the wrong tail context, and the damage only surfaces as prosody drift
//     many stages later.
//   * sliding layers span [i-255, i+256] — 256 left INCLUDING self, 256
//     right. Not 512 either side, not 512 left.
//
// The Gemma `1 + w` norm form is handled at conversion time (the converter
// folds the +1 into the stored weights and records
// breeze.te.norm_weights_pre_offset), so every norm here is the plain
// rms_norm * w that core_attn::kv_self_attn also assumes for q_norm/k_norm.
// If a GGUF ever says the fold did NOT happen, we add it back explicitly
// rather than silently computing a different model.

static ggml_tensor* te_norm(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* w, float eps, bool add_unit) {
    ggml_tensor* n = ggml_rms_norm(ctx0, x, eps);
    if (add_unit) {
        // out = n * (1 + w) == n + n*w. ggml has no scalar-add-to-tensor that
        // broadcasts a weight vector, so spell it out.
        return ggml_add(ctx0, n, ggml_mul(ctx0, n, w));
    }
    return ggml_mul(ctx0, n, w);
}

struct te_graph_io {
    ggml_cgraph* gf = nullptr;
    bool has_eoi = false;
};

// Builds the encoder for ONE segment of length T. Segments are encoded
// independently upstream; batching them changes every hidden state, so this
// takes one and the caller loops.
static te_graph_io build_te_graph(breeze_tts_2_context* c, int T, bool dump_layers, bool has_eoi) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    te_graph_io io;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(ids, "te_ids");
    ggml_set_input(ids);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "te_positions");
    ggml_set_input(positions);

    // Symmetric-window mask, shared by every sliding layer. Built once.
    ggml_tensor* slide_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
    ggml_set_name(slide_mask, "te_slide_mask");
    ggml_set_input(slide_mask);

    ggml_tensor* rope_linear = nullptr;
    if (!c->te_rope_linear_factors.empty()) {
        rope_linear = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t)c->te_rope_linear_factors.size());
        ggml_set_name(rope_linear, "te_rope_linear");
        ggml_set_input(rope_linear);
    }

    // ---- embedding ----
    ggml_tensor* cur = ggml_get_rows(ctx0, m.te_token_embd_w, ids);
    cur = ggml_scale(ctx0, cur, hp.te_embed_scale);

    // eoi_token_index substitution: where id == eoi, the embedding is REPLACED
    // by eoi_embedding — and notably NOT scaled by embed_scale, because the
    // reference's torch.where selects it after the scale was applied to the
    // looked-up rows only. Expressed as (keep * looked_up) + delta so it stays
    // a pure graph op; both inputs are no-ops (keep=1, delta=0) when the
    // segment contains no eoi token, and the ops are skipped entirely then.
    if (has_eoi) {
        ggml_tensor* keep = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, T);
        ggml_set_name(keep, "te_eoi_keep");
        ggml_set_input(keep);
        ggml_tensor* delta = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t)hp.te_d_model, T);
        ggml_set_name(delta, "te_eoi_delta");
        ggml_set_input(delta);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, keep), delta);
    }

    if (dump_layers) {
        ggml_tensor* e = ggml_cont(ctx0, cur);
        ggml_set_name(e, "te_layer0");
        ggml_set_output(e);
        ggml_build_forward_expand(gf, e);
    }

    const int grp = (int)(hp.te_n_heads / hp.te_n_kv_heads);

    core_attn::KvSelfAttnParams kvp_slide{};
    kvp_slide.n_heads = (int)hp.te_n_heads;
    kvp_slide.n_kv_heads = (int)hp.te_n_kv_heads;
    kvp_slide.head_dim = (int)hp.te_head_dim;
    kvp_slide.n_kv_grp = grp;
    kvp_slide.n_ctx_orig = (int)hp.te_max_pos;
    kvp_slide.rope_theta = hp.te_rope_theta_sliding;
    kvp_slide.rope_beta_fast = 0.0f;
    kvp_slide.rope_beta_slow = 0.0f;
    // query_pre_attn_scalar ** -0.5, NOT head_dim ** -0.5. They coincide at
    // 256 in this checkpoint; do not collapse them.
    kvp_slide.attn_scale = hp.te_attn_scale;
    kvp_slide.qk_norm_eps = hp.te_rms_norm_eps;
    kvp_slide.gqa_mode = core_attn::GQA_MANUAL_CONT;

    core_attn::KvSelfAttnParams kvp_full = kvp_slide;
    kvp_full.rope_theta = hp.te_rope_theta_full;
    kvp_full.rope_freq_factors = rope_linear; // "linear": inv_freq /= factor

    for (uint32_t il = 0; il < hp.te_n_layers; il++) {
        const auto& blk = m.te_layers[il];
        const bool is_full = (il < hp.te_layer_full.size()) && hp.te_layer_full[il] != 0;

        ggml_tensor* residual = cur;
        ggml_tensor* x = te_norm(ctx0, cur, blk.attn_pre_norm_w, hp.te_rms_norm_eps, false);
        ggml_tensor* attn = core_attn::kv_self_attn(ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w,
                                                    blk.attn_output_w, blk.attn_q_norm_w, blk.attn_k_norm_w, positions,
                                                    /*mask*/ is_full ? nullptr : slide_mask, c->te_kv_k, c->te_kv_v,
                                                    (int)il, /*n_past*/ 0, is_full ? kvp_full : kvp_slide);
        // post_self_attn_layernorm applies to the ATTENTION OUTPUT, before the
        // residual add — the Gemma sandwich, not a post-residual norm.
        attn = te_norm(ctx0, attn, blk.attn_post_norm_w, hp.te_rms_norm_eps, false);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = te_norm(ctx0, cur, blk.ffn_pre_norm_w, hp.te_rms_norm_eps, false);
        // gelu_pytorch_tanh + gate/up/down == GeGLU. ggml_gelu is the tanh
        // approximation, which is what gelu_pytorch_tanh means.
        ggml_tensor* mlp = core_ffn::geglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        mlp = te_norm(ctx0, mlp, blk.ffn_post_norm_w, hp.te_rms_norm_eps, false);
        cur = ggml_add(ctx0, residual, mlp);

        if (dump_layers) {
            ggml_tensor* h = ggml_cont(ctx0, cur);
            char nm[32];
            std::snprintf(nm, sizeof(nm), "te_layer%u", il + 1);
            ggml_set_name(h, nm);
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);
        }
    }

    cur = te_norm(ctx0, cur, m.te_output_norm_w, hp.te_rms_norm_eps, false);
    ggml_set_name(cur, "te_hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    io.gf = gf;
    io.has_eoi = has_eoi;
    return io;
}

// The symmetric window, as an additive F16 mask. left/right come from the
// GGUF, computed by the converter as (sw+1)/2 and sw/2+1 — reproduced rather
// than re-derived so a different sliding_window stays correct.
static std::vector<ggml_fp16_t> build_sliding_mask(int T, int win_left, int win_right) {
    std::vector<ggml_fp16_t> mask((size_t)T * T, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; q++) {
        for (int k = 0; k < T; k++) {
            const int dist = q - k;
            const bool local = (dist >= 0 && dist < win_left) || (dist < 0 && -dist < win_right);
            if (!local)
                mask[(size_t)q * T + k] = neg;
        }
    }
    return mask;
}

// ===========================================================================
// Graph: summed-codebook frame embedding
// ===========================================================================
//
// CSM's scheme, kept verbatim: audio_tokens_offsets = arange(16) * 2051, so
// codebook c's code k is row c*2051 + k of one shared table, and a frame is
// the SUM of its 16 rows. audio_embed_size == hidden_size == 2048, so there is
// no audio_embeds_projector to apply (breeze.py:786-791).
//
// Done in-graph rather than by reading the table to the host: the table is
// [2048, 32816] and a host copy is 268 MB of f32 that would be touched once
// per frame.
static ggml_cgraph* build_frame_embed_graph(breeze_tts_2_context* c, int n_frames) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int n_cb = (int)hp.num_codebooks;
    const int d = (int)hp.hidden_size;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 2048, false);

    // Offsets are baked into the ids by the caller (code + cb*vocab), so this
    // is one gather of n_cb * n_frames rows.
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_cb * n_frames);
    ggml_set_name(ids, "fe_ids");
    ggml_set_input(ids);

    ggml_tensor* rows = ggml_get_rows(ctx0, m.audio_embd_w, ids); // [d, n_cb*n_frames]
    rows = ggml_reshape_3d(ctx0, rows, d, n_cb, n_frames);        // [d, n_cb, n_frames]
    // sum over the codebook axis: ggml_sum_rows reduces ne0, so bring the
    // codebook axis there first.
    rows = ggml_cont(ctx0, ggml_permute(ctx0, rows, 1, 0, 2, 3)); // [n_cb, d, n_frames]
    ggml_tensor* summed = ggml_sum_rows(ctx0, rows);              // [1, d, n_frames]
    summed = ggml_reshape_2d(ctx0, summed, d, n_frames);          // [d, n_frames]
    ggml_set_name(summed, "fe_out");
    ggml_set_output(summed);
    ggml_build_forward_expand(gf, summed);

    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Graph: backbone (Qwen3)
// ===========================================================================
//
// Every constant here comes from the NESTED backbone_config. theta is 1e6 and
// eps is 1e-6; the top-level 500000 / 1e-5 are decoys that the backbone
// factory never reads.
//
// `branch` selects the KV-cache slice. The cache's trailing dim is
// n_layers * n_branch and core_attn takes `il` as that index, so CFG costs
// arithmetic here and nothing in core.
static ggml_cgraph* build_backbone_graph(breeze_tts_2_context* c, int n_past, int T, int branch, bool dump_layers) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int d = (int)hp.bb_d_model;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    const int Lk = n_past + T;
    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "bb_embeds");
    ggml_set_input(embeds);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "bb_positions");
    ggml_set_input(positions);
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "bb_mask");
        ggml_set_input(causal_mask);
    }

    core_attn::KvSelfAttnParams kvp{};
    kvp.n_heads = (int)hp.bb_n_heads;
    kvp.n_kv_heads = (int)hp.bb_n_kv_heads;
    kvp.head_dim = (int)hp.bb_head_dim;
    kvp.n_kv_grp = (int)(hp.bb_n_heads / hp.bb_n_kv_heads);
    kvp.n_ctx_orig = (int)hp.bb_max_pos;
    kvp.rope_theta = hp.bb_rope_theta;
    kvp.rope_beta_fast = 0.0f;
    kvp.rope_beta_slow = 0.0f;
    kvp.attn_scale = 1.0f / std::sqrt((float)hp.bb_head_dim);
    kvp.qk_norm_eps = hp.bb_rms_norm_eps;
    kvp.gqa_mode = core_attn::GQA_MANUAL_CONT;

    ggml_tensor* cur = embeds;
    for (uint32_t il = 0; il < hp.bb_n_layers; il++) {
        const auto& blk = m.bb_layers[il];
        const int cache_il = (int)il * c->n_branch + branch;

        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps), blk.attn_norm_w);
        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_output_w,
                                    blk.attn_q_norm_w, blk.attn_k_norm_w, positions, (T == 1) ? nullptr : causal_mask,
                                    c->bb_kv_k, c->bb_kv_v, cache_il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps), blk.ffn_norm_w);
        cur = ggml_add(ctx0, residual, core_ffn::swiglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w));

        if (dump_layers) {
            // Last position only — that is the one the depth decoder is
            // conditioned on, and the only one the fixture records.
            ggml_tensor* h = cur;
            if (T > 1)
                h = ggml_view_2d(ctx0, h, d, 1, h->nb[1], (size_t)(T - 1) * h->nb[1]);
            h = ggml_cont(ctx0, h);
            char nm[32];
            std::snprintf(nm, sizeof(nm), "bb_layer%u", il);
            ggml_set_name(h, nm);
            ggml_set_output(h);
            ggml_build_forward_expand(gf, h);
        }
    }

    cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.bb_rms_norm_eps), m.bb_output_norm_w);
    if (T > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    cur = ggml_cont(ctx0, cur);
    ggml_set_name(cur, "bb_hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // lm_head is 2052 wide: 2051 codes plus the backbone EOS class at index
    // 2051. Sampling must both honour that extra class and mask ids in
    // [2048, 2051), which are reserved and undecodable.
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.bb_codebook0_head_w, cur);
    ggml_set_name(logits, "bb_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Graph: depth decoder
// ===========================================================================
//
// Per frame: position 0 carries the backbone's last hidden state UNPROJECTED
// (backbone_hidden_size == audio_embed_size, so breeze.py's
// backbone_hidden_state_projector is None), positions 1..k carry the codes
// chosen so far, each embedded at offset (position - 1) * vocab_size. Then
// ONE shared inputs_embeds_projector takes all of it 2048 -> 1024.
//
// The head for position k is cb_head[k - 1]. Off-by-one here shifts every
// codebook by one and still produces audio.
static ggml_cgraph* build_depth_graph(breeze_tts_2_context* c, int T, int n_past, int codebook_idx, int branch) {
    const auto& m = c->model;
    const auto& hp = m.hp;
    const int d = (int)hp.dd_d_model;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input arrives in BACKBONE dim (2048) and is projected inside the graph,
    // exactly as the reference does — projecting on the host would be a second
    // implementation of the same matmul, in a different precision.
    ggml_tensor* embeds_bb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t)hp.dd_backbone_hidden, T);
    ggml_set_name(embeds_bb, "dd_embeds_bb");
    ggml_set_input(embeds_bb);
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "dd_positions");
    ggml_set_input(positions);
    ggml_tensor* rope_ff = nullptr;
    if (!c->dd_rope_freq_factors.empty()) {
        rope_ff = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (int64_t)c->dd_rope_freq_factors.size());
        ggml_set_name(rope_ff, "dd_rope_ff");
        ggml_set_input(rope_ff);
    }
    const int Lk = n_past + T;
    ggml_tensor* causal_mask = nullptr;
    if (T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "dd_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = ggml_mul_mat(ctx0, m.dd_projection_w, embeds_bb); // 2048 -> 1024

    core_attn::KvSelfAttnParams kvp{};
    kvp.n_heads = (int)hp.dd_n_heads;
    kvp.n_kv_heads = (int)hp.dd_n_kv_heads;
    kvp.head_dim = (int)hp.dd_head_dim;
    kvp.n_kv_grp = (int)(hp.dd_n_heads / hp.dd_n_kv_heads);
    kvp.n_ctx_orig = (int)hp.dd_rope_orig_max_pos;
    kvp.rope_theta = hp.dd_rope_theta;
    kvp.rope_beta_fast = 0.0f;
    kvp.rope_beta_slow = 0.0f;
    kvp.attn_scale = 1.0f / std::sqrt((float)hp.dd_head_dim);
    kvp.qk_norm_eps = 0.0f; // no q/k norm on this stack
    kvp.gqa_mode = core_attn::GQA_MANUAL_CONT;
    kvp.rope_freq_factors = rope_ff;

    for (uint32_t il = 0; il < hp.dd_n_layers; il++) {
        const auto& blk = m.dd_layers[il];
        const int cache_il = (int)il * c->n_branch + branch;

        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.dd_rms_norm_eps), blk.attn_norm_w);
        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_output_w,
                                    /*q_norm*/ nullptr, /*k_norm*/ nullptr, positions, (T == 1) ? nullptr : causal_mask,
                                    c->dd_kv_k, c->dd_kv_v, cache_il, n_past, kvp);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.dd_rms_norm_eps), blk.ffn_norm_w);
        cur = ggml_add(ctx0, residual, core_ffn::swiglu(ctx0, x, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w));
    }

    cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hp.dd_rms_norm_eps), m.dd_output_norm_w);
    if (T > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    cur = ggml_cont(ctx0, cur);

    // Pre-transposed to [d_model, vocab] by the converter, so this is a plain
    // mul_mat. CSM slices a 3-D tensor and pays cont(transpose()) every step.
    ggml_tensor* head = m.dd_cb_heads[(size_t)(codebook_idx - 1)];
    ggml_tensor* logits = ggml_mul_mat(ctx0, head, cur);
    ggml_set_name(logits, "dd_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// Graph execution helper
// ===========================================================================

struct GraphInput {
    const char* name;
    const void* data;
    size_t bytes;
};

static bool run_graph(breeze_tts_2_context* c, ggml_cgraph* gf, const std::vector<GraphInput>& inputs,
                      const char* what) {
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "breeze_tts_2: failed to allocate %s graph\n", what);
        return false;
    }
    for (const auto& in : inputs) {
        if (!in.data || in.bytes == 0)
            continue;
        ggml_tensor* t = ggml_graph_get_tensor(gf, in.name);
        if (!t) {
            fprintf(stderr, "breeze_tts_2: %s graph has no input '%s'\n", what, in.name);
            return false;
        }
        ggml_backend_tensor_set(t, in.data, 0, in.bytes);
    }
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "breeze_tts_2: %s compute failed\n", what);
        return false;
    }
    return true;
}

static bool fetch(ggml_cgraph* gf, const char* name, float* out, size_t n) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name);
    if (!t || !out)
        return false;
    ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
    return true;
}

// ===========================================================================
// Sampling
// ===========================================================================

static uint64_t xorshift64(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

// Mask the reserved codec ids out of a logit vector, in place.
//
// Ids in [codec_config.codebook_size, vocab_size) — 2048..2050 here — exist in
// the head's output space and are NOT decodable codes. The reference
// suppresses them explicitly; a greedy pick that lands on 2050 yields a frame
// the codec cannot render, which reads as a codec bug.
static void mask_reserved(float* logits, int n, const breeze_hparams& hp) {
    const int lo = (int)hp.reserved_lo;
    const int hi = std::min((int)hp.reserved_hi, n);
    for (int i = lo; i < hi; i++)
        logits[i] = -INFINITY;
}

static void apply_repetition_penalty(float* logits, int n, const std::vector<int32_t>& history, float penalty) {
    if (penalty == 1.0f || history.empty())
        return;
    for (int32_t tok : history) {
        if (tok < 0 || tok >= n)
            continue;
        // The HF convention: divide positives, multiply negatives.
        logits[tok] = logits[tok] > 0.0f ? logits[tok] / penalty : logits[tok] * penalty;
    }
}

static int32_t sample_logits(float* logits, int n, float temperature, int topk, float topp, uint64_t& rng) {
    if (temperature <= 0.0f) {
        int best = 0;
        float bv = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        }
        return best;
    }
    std::vector<std::pair<float, int>> v;
    v.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        if (std::isfinite(logits[i]))
            v.emplace_back(logits[i] / temperature, i);
    }
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (topk > 0 && (int)v.size() > topk)
        v.resize((size_t)topk);

    const float mx = v[0].first;
    double sum = 0.0;
    for (auto& p : v) {
        p.first = std::exp(p.first - mx);
        sum += p.first;
    }
    if (sum <= 0.0)
        return v[0].second;
    // top-p: keep the smallest prefix whose mass reaches topp. topp == 1.0
    // (the shipped default) keeps everything, so this is inert by default.
    if (topp > 0.0f && topp < 1.0f) {
        double acc = 0.0;
        size_t keep = 0;
        for (; keep < v.size(); keep++) {
            acc += v[keep].first / sum;
            if (acc >= (double)topp) {
                keep++;
                break;
            }
        }
        v.resize(std::max<size_t>(keep, 1));
        sum = 0.0;
        for (auto& p : v)
            sum += p.first;
    }
    const double r = (double)(xorshift64(rng) >> 11) / (double)(1ULL << 53) * sum;
    double acc = 0.0;
    for (const auto& p : v) {
        acc += p.first;
        if (acc >= r)
            return p.second;
    }
    return v.back().second;
}

// ===========================================================================
// Text encoder driver
// ===========================================================================

// Encode one segment and return its [T, d_model] hidden state, row-major.
static bool encode_segment(breeze_tts_2_context* c, const int32_t* ids, int T, std::vector<float>& out_hidden,
                           std::vector<std::vector<float>>* out_layers) {
    const auto& hp = c->model.hp;
    if (T <= 0 || T > c->te_kv_max_ctx) {
        fprintf(stderr, "breeze_tts_2: text segment of %d tokens exceeds the encoder scratch (%d)\n", T,
                c->te_kv_max_ctx);
        return false;
    }
    const int d = (int)hp.te_d_model;

    bool has_eoi = false;
    for (int i = 0; i < T; i++)
        if (ids[i] == (int32_t)hp.te_eoi_token_index)
            has_eoi = true;

    std::vector<int32_t> positions((size_t)T);
    for (int i = 0; i < T; i++)
        positions[(size_t)i] = i;
    auto mask = build_sliding_mask(T, (int)hp.te_win_left, (int)hp.te_win_right);

    std::vector<float> keep, delta;
    if (has_eoi) {
        keep.assign((size_t)T, 1.0f);
        delta.assign((size_t)T * d, 0.0f);
        for (int i = 0; i < T; i++) {
            if (ids[i] != (int32_t)hp.te_eoi_token_index)
                continue;
            keep[(size_t)i] = 0.0f;
            if ((int)c->te_eoi_host.size() == d)
                std::memcpy(&delta[(size_t)i * d], c->te_eoi_host.data(), (size_t)d * sizeof(float));
        }
    }

    const bool dump_layers = out_layers != nullptr;
    te_graph_io io = build_te_graph(c, T, dump_layers, has_eoi);

    std::vector<GraphInput> in = {
        {"te_ids", ids, (size_t)T * sizeof(int32_t)},
        {"te_positions", positions.data(), positions.size() * sizeof(int32_t)},
        {"te_slide_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)},
    };
    if (!c->te_rope_linear_factors.empty())
        in.push_back(
            {"te_rope_linear", c->te_rope_linear_factors.data(), c->te_rope_linear_factors.size() * sizeof(float)});
    if (has_eoi) {
        in.push_back({"te_eoi_keep", keep.data(), keep.size() * sizeof(float)});
        in.push_back({"te_eoi_delta", delta.data(), delta.size() * sizeof(float)});
    }
    if (!run_graph(c, io.gf, in, "text encoder"))
        return false;

    out_hidden.assign((size_t)T * d, 0.0f);
    if (!fetch(io.gf, "te_hidden", out_hidden.data(), out_hidden.size()))
        return false;

    if (out_layers) {
        out_layers->assign((size_t)hp.te_n_layers + 1, {});
        for (uint32_t j = 0; j <= hp.te_n_layers; j++) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "te_layer%u", j);
            std::vector<float> buf((size_t)T * d, 0.0f);
            if (fetch(io.gf, nm, buf.data(), buf.size()))
                (*out_layers)[j] = std::move(buf);
        }
    }
    return true;
}

// text_encoder_proj: [n, 1152] -> [n, 2048]. text_encoder_proj_type is
// "linear" and text_encoder_layer_projs is absent from the checkpoint, so the
// DimFusion path is inert — do not build it.
static bool project_text(breeze_tts_2_context* c, const float* hidden, int n, std::vector<float>& out) {
    const auto& hp = c->model.hp;
    const int din = (int)hp.te_d_model;
    const int dout = (int)hp.hidden_size;

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 512, false);
    ggml_tensor* h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, din, n);
    ggml_set_name(h, "tp_in");
    ggml_set_input(h);
    ggml_tensor* o = ggml_mul_mat(ctx0, c->model.te_proj_w, h);
    ggml_set_name(o, "tp_out");
    ggml_set_output(o);
    ggml_build_forward_expand(gf, o);
    ggml_free(ctx0);

    if (!run_graph(c, gf, {{"tp_in", hidden, (size_t)n * din * sizeof(float)}}, "text projection"))
        return false;
    out.assign((size_t)n * dout, 0.0f);
    return fetch(gf, "tp_out", out.data(), out.size());
}

// Summed-codebook embeddings for n frames of codes ([n, 16] row-major).
static bool embed_frames(breeze_tts_2_context* c, const int32_t* codes, int n_frames, std::vector<float>& out) {
    const auto& hp = c->model.hp;
    const int n_cb = (int)hp.num_codebooks;
    const int vocab = (int)hp.audio_vocab_size;
    std::vector<int32_t> ids((size_t)n_cb * n_frames);
    for (int f = 0; f < n_frames; f++)
        for (int cb = 0; cb < n_cb; cb++)
            ids[(size_t)f * n_cb + cb] = codes[(size_t)f * n_cb + cb] + cb * vocab; // audio_tokens_offsets

    ggml_cgraph* gf = build_frame_embed_graph(c, n_frames);
    if (!run_graph(c, gf, {{"fe_ids", ids.data(), ids.size() * sizeof(int32_t)}}, "frame embedding"))
        return false;
    out.assign((size_t)n_frames * hp.hidden_size, 0.0f);
    return fetch(gf, "fe_out", out.data(), out.size());
}

// Raw rows of the shared audio embedding table. The DEPTH decoder embeds ONE
// codebook per step (embed_tokens(input_ids + offset)), not a sum of sixteen —
// same table, different access pattern, and conflating them is silent.
static bool embed_rows(breeze_tts_2_context* c, const int32_t* ids, int n, std::vector<float>& out) {
    const int d = (int)c->model.hp.hidden_size;
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);
    ggml_tensor* t_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(t_ids, "er_ids");
    ggml_set_input(t_ids);
    ggml_tensor* rows = ggml_get_rows(ctx0, c->model.audio_embd_w, t_ids);
    ggml_set_name(rows, "er_out");
    ggml_set_output(rows);
    ggml_build_forward_expand(gf, rows);
    ggml_free(ctx0);

    if (!run_graph(c, gf, {{"er_ids", ids, (size_t)n * sizeof(int32_t)}}, "embedding rows"))
        return false;
    out.assign((size_t)n * d, 0.0f);
    return fetch(gf, "er_out", out.data(), out.size());
}

// ===========================================================================
// Prompt -> prefill embeddings
// ===========================================================================
//
// inputs_embeds starts as ZEROS and is filled in two passes: the projected
// text-encoder rows go to the text positions, and the summed reference-audio
// codebook embeddings go to the <|AUDIO|> positions. Get this wrong and
// nothing complains — the model runs and the audio is garbage — which is why
// backbone_inputs_embeds is a fixture stage of its own.

struct AssembledPrompt {
    std::vector<int32_t> ids;       // [L]
    std::vector<uint8_t> text_mask; // [L], 1 where a TEXT token sits
    std::vector<int32_t> seg_lens;  // one entry per TEXT segment
    std::vector<float> embeds;      // [L, hidden_size]
    int L = 0;
};

static bool assemble_prompt(breeze_tts_2_context* c, const std::vector<PromptSegment>& segs, const int32_t* ref_codes,
                            int ref_frames, AssembledPrompt& out) {
    const auto& hp = c->model.hp;
    const int d = (int)hp.hidden_size;
    const int n_cb = (int)hp.num_codebooks;

    out.ids.clear();
    out.text_mask.clear();
    out.seg_lens.clear();
    for (const auto& s : segs) {
        out.ids.insert(out.ids.end(), s.ids.begin(), s.ids.end());
        out.text_mask.insert(out.text_mask.end(), s.ids.size(), s.is_text ? 1 : 0);
        if (s.is_text)
            out.seg_lens.push_back((int32_t)s.ids.size());
    }
    out.L = (int)out.ids.size();
    if (out.L == 0)
        return false;

    // --- text: each segment encoded on its own, then one projection ---
    std::vector<float> all_hidden;
    all_hidden.reserve((size_t)out.L * hp.te_d_model);
    for (const auto& s : segs) {
        if (!s.is_text)
            continue;
        std::vector<float> h;
        if (!encode_segment(c, s.ids.data(), (int)s.ids.size(), h, nullptr))
            return false;
        all_hidden.insert(all_hidden.end(), h.begin(), h.end());
    }
    const int n_text = (int)(all_hidden.size() / hp.te_d_model);
    std::vector<float> proj;
    if (n_text > 0 && !project_text(c, all_hidden.data(), n_text, proj))
        return false;

    out.embeds.assign((size_t)out.L * d, 0.0f);
    int ti = 0;
    for (int i = 0; i < out.L; i++) {
        if (!out.text_mask[(size_t)i])
            continue;
        std::memcpy(&out.embeds[(size_t)i * d], &proj[(size_t)ti * d], (size_t)d * sizeof(float));
        ti++;
    }

    // --- reference audio: summed codebook embeddings at the <|AUDIO|> slots ---
    if (ref_frames > 0 && ref_codes) {
        std::vector<float> aud;
        if (!embed_frames(c, ref_codes, ref_frames, aud))
            return false;
        int ai = 0;
        for (int i = 0; i < out.L && ai < ref_frames; i++) {
            if (out.ids[(size_t)i] != (int32_t)hp.audio_token_id)
                continue;
            std::memcpy(&out.embeds[(size_t)i * d], &aud[(size_t)ai * d], (size_t)d * sizeof(float));
            ai++;
        }
    }

    // --- <|audio_eos|>: the embedding of an all-codebook_eos frame ---
    // Not a learned token: breeze.py embeds a frame of sixteen
    // codebook_eos_token_id (0) values and writes THAT at every audio-eos
    // position.
    bool need_eos = false;
    for (int i = 0; i < out.L; i++)
        if (out.ids[(size_t)i] == (int32_t)hp.audio_eos_token_id)
            need_eos = true;
    if (need_eos) {
        std::vector<int32_t> eos_frame((size_t)n_cb, (int32_t)hp.codebook_eos_token_id);
        std::vector<float> eos_emb;
        if (!embed_frames(c, eos_frame.data(), 1, eos_emb))
            return false;
        for (int i = 0; i < out.L; i++) {
            if (out.ids[(size_t)i] == (int32_t)hp.audio_eos_token_id)
                std::memcpy(&out.embeds[(size_t)i * d], eos_emb.data(), (size_t)d * sizeof(float));
        }
    }
    return true;
}

static std::vector<ggml_fp16_t> build_causal_mask(int T, int Lk, int n_past) {
    std::vector<ggml_fp16_t> mask((size_t)Lk * T, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t neg = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; q++)
        for (int k = n_past + q + 1; k < Lk; k++)
            mask[(size_t)q * Lk + k] = neg;
    return mask;
}

// ===========================================================================
// One frame of the depth decoder
// ===========================================================================
//
// 15 sequential steps, each conditioned on the previous pick. `branch` selects
// the KV slice; the CFG combine, when it is wired, happens on the logits of
// each step across branches, which is why this returns per-step logits.
static bool depth_decode_frame(breeze_tts_2_context* c, const float* bb_hidden, int32_t cb0, float temperature,
                               std::vector<int32_t>& codes, std::vector<std::vector<float>>* out_logits, int branch) {
    const auto& hp = c->model.hp;
    const int n_cb = (int)hp.num_codebooks;
    const int vocab = (int)hp.dd_vocab_size;
    const int dbb = (int)hp.dd_backbone_hidden;

    ggml_backend_buffer_clear(c->dd_kv_buf, 0);
    codes.assign((size_t)n_cb, 0);
    codes[0] = cb0;
    if (out_logits)
        out_logits->assign((size_t)n_cb - 1, {});

    for (int cbi = 1; cbi < n_cb; cbi++) {
        std::vector<float> embeds;
        std::vector<int32_t> positions;
        int T, n_past;

        if (cbi == 1) {
            // First call carries BOTH position 0 (the backbone hidden state,
            // written in UNPROJECTED — backbone_hidden_state_projector is None
            // because 2048 == 2048) and position 1 (codebook 0 at offset 0).
            T = 2;
            n_past = 0;
            positions = {0, 1};
            embeds.assign((size_t)dbb * 2, 0.0f);
            std::memcpy(&embeds[0], bb_hidden, (size_t)dbb * sizeof(float));
            const int32_t id = codes[0]; // offset = clamp(1-1,0)*vocab = 0
            std::vector<float> row;
            if (!embed_rows(c, &id, 1, row))
                return false;
            std::memcpy(&embeds[(size_t)dbb], row.data(), (size_t)dbb * sizeof(float));
        } else {
            T = 1;
            n_past = cbi;
            positions = {cbi};
            const int32_t id = codes[(size_t)cbi - 1] + (cbi - 1) * (int32_t)hp.audio_vocab_size;
            std::vector<float> row;
            if (!embed_rows(c, &id, 1, row))
                return false;
            embeds = std::move(row);
        }

        ggml_cgraph* gf = build_depth_graph(c, T, n_past, cbi, branch);
        std::vector<GraphInput> in = {
            {"dd_embeds_bb", embeds.data(), embeds.size() * sizeof(float)},
            {"dd_positions", positions.data(), positions.size() * sizeof(int32_t)},
        };
        std::vector<ggml_fp16_t> mask;
        if (T > 1) {
            mask = build_causal_mask(T, n_past + T, n_past);
            in.push_back({"dd_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)});
        }
        if (!c->dd_rope_freq_factors.empty())
            in.push_back(
                {"dd_rope_ff", c->dd_rope_freq_factors.data(), c->dd_rope_freq_factors.size() * sizeof(float)});
        if (!run_graph(c, gf, in, "depth decoder"))
            return false;

        std::vector<float> logits((size_t)vocab, 0.0f);
        if (!fetch(gf, "dd_logits", logits.data(), logits.size()))
            return false;
        // Dump the RAW logits, then mask a COPY for sampling. The reserved-id
        // mask writes -inf into [2048, 2051), and the reference dumps its
        // logits before its own suppression — so masking in place made |cpp|
        // inf and every cosine NaN, which reported as a catastrophic failure
        // of a stage that was merely being measured wrong.
        if (out_logits)
            (*out_logits)[(size_t)cbi - 1] = logits;
        std::vector<float> sampling_logits = logits;
        mask_reserved(sampling_logits.data(), vocab, hp);
        codes[(size_t)cbi] =
            sample_logits(sampling_logits.data(), vocab, temperature, (int)hp.s_top_k, hp.s_top_p, c->rng_state);
    }
    return true;
}

// ===========================================================================
// Autoregressive generation
// ===========================================================================

struct GenOptions {
    float temperature = 0.9f;
    float depth_temperature = 0.9f;
    int max_frames = 750;
    bool greedy = false;
    // 1.0 == off. Kept SEPARATE from `greedy` rather than derived from it,
    // because they answer different questions and the diff harness needs both
    // set deliberately. See the call sites.
    float repetition_penalty = 1.0f;
};

static bool generate_codes(breeze_tts_2_context* c, const AssembledPrompt& prompt, const GenOptions& opt,
                           std::vector<std::vector<int32_t>>& out_frames) {
    const auto& hp = c->model.hp;
    const int d = (int)hp.bb_d_model;
    const int n_logits = (int)hp.lm_head_out; // 2052: 2051 codes + the EOS class
    const int n_cb = (int)hp.num_codebooks;

    const float bb_temp = opt.greedy ? 0.0f : opt.temperature;
    const float dd_temp = opt.greedy ? 0.0f : opt.depth_temperature;

    ggml_backend_buffer_clear(c->bb_kv_buf, 0);
    out_frames.clear();

    if (prompt.L > c->bb_kv_max_ctx) {
        fprintf(stderr, "breeze_tts_2: prompt of %d tokens exceeds the backbone cache (%d)\n", prompt.L,
                c->bb_kv_max_ctx);
        return false;
    }

    // ---- prefill ----
    std::vector<float> bb_hidden((size_t)d, 0.0f);
    std::vector<float> bb_logits((size_t)n_logits, 0.0f);
    {
        breeze_bench_stage _b("backbone_prefill");
        std::vector<int32_t> positions((size_t)prompt.L);
        for (int i = 0; i < prompt.L; i++)
            positions[(size_t)i] = i;
        auto mask = build_causal_mask(prompt.L, prompt.L, 0);
        ggml_cgraph* gf = build_backbone_graph(c, 0, prompt.L, /*branch*/ 0, /*dump_layers*/ false);
        if (!run_graph(c, gf,
                       {{"bb_embeds", prompt.embeds.data(), prompt.embeds.size() * sizeof(float)},
                        {"bb_positions", positions.data(), positions.size() * sizeof(int32_t)},
                        {"bb_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)}},
                       "backbone prefill"))
            return false;
        if (!fetch(gf, "bb_hidden", bb_hidden.data(), bb_hidden.size()))
            return false;
        if (!fetch(gf, "bb_logits", bb_logits.data(), bb_logits.size()))
            return false;
    }

    int n_past = prompt.L;
    std::vector<int32_t> cb0_history;

    for (int f = 0; f < opt.max_frames; f++) {
        mask_reserved(bb_logits.data(), n_logits, hp);
        apply_repetition_penalty(bb_logits.data(), n_logits, cb0_history, opt.repetition_penalty);
        const int32_t cb0 =
            sample_logits(bb_logits.data(), n_logits, bb_temp, (int)hp.s_top_k, hp.s_top_p, c->rng_state);
        if (cb0 == (int32_t)hp.backbone_eos_token_id)
            break;
        cb0_history.push_back(cb0);

        std::vector<int32_t> frame;
        {
            breeze_bench_stage _b("depth_frame");
            if (!depth_decode_frame(c, bb_hidden.data(), cb0, dd_temp, frame, nullptr, /*branch*/ 0))
                return false;
        }
        // A frame of all pad is the EOS frame; it must not reach the codec.
        bool all_pad = true;
        for (int i = 0; i < n_cb; i++)
            if (frame[(size_t)i] != (int32_t)hp.codebook_pad_token_id)
                all_pad = false;
        if (all_pad)
            break;
        out_frames.push_back(frame);

        if (n_past + 1 > c->bb_kv_max_ctx)
            break;

        // Feed the frame back and step the backbone once.
        std::vector<float> emb;
        if (!embed_frames(c, frame.data(), 1, emb))
            return false;
        const int32_t pos = n_past;
        ggml_cgraph* gf = build_backbone_graph(c, n_past, 1, /*branch*/ 0, /*dump_layers*/ false);
        if (!run_graph(c, gf,
                       {{"bb_embeds", emb.data(), emb.size() * sizeof(float)}, {"bb_positions", &pos, sizeof(int32_t)}},
                       "backbone step"))
            return false;
        if (!fetch(gf, "bb_hidden", bb_hidden.data(), bb_hidden.size()))
            return false;
        if (!fetch(gf, "bb_logits", bb_logits.data(), bb_logits.size()))
            return false;
        n_past++;
    }
    return true;
}

// Flatten [n_frames][16] to the row-major [T*16] the codec wants.
static std::vector<int32_t> flatten(const std::vector<std::vector<int32_t>>& frames) {
    std::vector<int32_t> out;
    for (const auto& f : frames)
        out.insert(out.end(), f.begin(), f.end());
    return out;
}

static float* decode_to_pcm(breeze_tts_2_context* c, const std::vector<std::vector<int32_t>>& frames,
                            int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (frames.empty())
        return nullptr;
    if (!c->codec) {
        fprintf(stderr,
                "breeze_tts_2: no codec loaded — %zu frames of codes were generated but cannot be rendered. "
                "Pass the qwen3-tts-tokenizer-12hz GGUF via --codec-model.\n",
                frames.size());
        return nullptr;
    }
    auto flat = flatten(frames);
    breeze_bench_stage _b("codec_decode");
    return qwen3_tts_decode_codes(c->codec, flat.data(), (int)flat.size(), out_n_samples);
}

} // namespace

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct breeze_tts_2_context_params breeze_tts_2_context_default_params(void) {
    struct breeze_tts_2_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = 0.9f;
    p.depth_temperature = 0.9f;
    p.topk = 50;
    p.topp = 1.0f;
    p.repetition_penalty = 1.1f;
    p.seed = 0;
    p.max_new_tokens = 0;
    p.codec_path = nullptr;
    return p;
}

extern "C" struct breeze_tts_2_context* breeze_tts_2_init_from_file(const char* path_model,
                                                                    struct breeze_tts_2_context_params params) {
    auto* c = new breeze_tts_2_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    if (params.seed != 0)
        c->rng_state = params.seed;

    c->backend = params.use_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
    if (!c->backend)
        c->backend = core_cpu_backend::init();
    c->backend_cpu = core_cpu_backend::init();
    core_cpu_backend::set_n_threads(c->backend_cpu, c->n_threads);
    if (core_cpu_backend::is_cpu(c->backend))
        core_cpu_backend::set_n_threads(c->backend, c->n_threads);

    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            fprintf(stderr, "breeze_tts_2: failed to open '%s'\n", path_model);
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "breeze_tts_2", wl)) {
        fprintf(stderr, "breeze_tts_2: failed to load weights from '%s'\n", path_model);
        delete c;
        return nullptr;
    }
    c->model.ctx_w = wl.ctx;
    c->model.buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_weights(c)) {
        fprintf(stderr, "breeze_tts_2: failed to bind weights\n");
        delete c;
        return nullptr;
    }

    auto& hp = c->model.hp;

    // The converter is expected to have folded Gemma's `1 + w` into the stored
    // text-encoder norm weights, which is what lets core_attn::kv_self_attn's
    // plain `rms_norm * w` q/k-norm be correct here. Say so if it did not,
    // rather than computing a different model in silence.
    if (hp.te_norm_unit_offset && !hp.te_norm_offset_baked) {
        fprintf(stderr, "breeze_tts_2: this GGUF says the text encoder uses Gemma's (1 + w) norm form but "
                        "does NOT have the +1 folded into its weights (breeze.te.norm_weights_pre_offset is "
                        "unset). Re-convert with a current models/convert-breeze-tts-2-to-gguf.py; the q/k "
                        "norms inside core attention cannot apply the offset.\n");
        delete c;
        return nullptr;
    }

    c->sched = nullptr;
    {
        int n_be = 1;
        ggml_backend_t backends[2] = {c->backend, nullptr};
        if (c->backend_cpu && c->backend_cpu != c->backend)
            backends[n_be++] = c->backend_cpu;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    c->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // Precomputed RoPE tables.
    c->dd_rope_freq_factors = build_llama3_freq_factors(hp.dd_rope_theta, (int)hp.dd_head_dim,
                                                        hp.dd_rope_scaling_factor, hp.dd_rope_low_freq_factor,
                                                        hp.dd_rope_high_freq_factor, (int)hp.dd_rope_orig_max_pos);
    c->te_rope_linear_factors = build_linear_freq_factors((int)hp.te_head_dim, hp.te_rope_factor_full);

    if (c->model.te_eoi_embd) {
        c->te_eoi_host.assign((size_t)hp.te_d_model, 0.0f);
        ggml_backend_tensor_get(c->model.te_eoi_embd, c->te_eoi_host.data(), 0, c->te_eoi_host.size() * sizeof(float));
    }

    c->n_branch = 1; // see breeze_tts_2_capabilities()
    c->max_new_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : (int)hp.s_max_new_tokens;

    const int te_ctx = std::min<int>((int)hp.te_max_pos, 2048);
    if (!alloc_kv(c, &c->te_kv_ctx, &c->te_kv_buf, &c->te_kv_k, &c->te_kv_v, (int)hp.te_head_dim, te_ctx,
                  (int)hp.te_n_kv_heads, (int)hp.te_n_layers, "breeze_te")) {
        fprintf(stderr, "breeze_tts_2: failed to allocate the text-encoder scratch\n");
        delete c;
        return nullptr;
    }
    c->te_kv_max_ctx = te_ctx;

    int bb_ctx = (int)hp.s_max_seq_len;
    if (params.max_new_tokens > 0 && params.max_new_tokens + 1024 > bb_ctx)
        bb_ctx = params.max_new_tokens + 1024;
    if (!alloc_kv(c, &c->bb_kv_ctx, &c->bb_kv_buf, &c->bb_kv_k, &c->bb_kv_v, (int)hp.bb_head_dim, bb_ctx,
                  (int)hp.bb_n_kv_heads, (int)hp.bb_n_layers * c->n_branch, "breeze_bb")) {
        fprintf(stderr, "breeze_tts_2: failed to allocate the backbone KV cache\n");
        delete c;
        return nullptr;
    }
    c->bb_kv_max_ctx = bb_ctx;

    if (!alloc_kv(c, &c->dd_kv_ctx, &c->dd_kv_buf, &c->dd_kv_k, &c->dd_kv_v, (int)hp.dd_head_dim,
                  (int)hp.num_codebooks + 1, (int)hp.dd_n_kv_heads, (int)hp.dd_n_layers * c->n_branch, "breeze_dd")) {
        fprintf(stderr, "breeze_tts_2: failed to allocate the depth-decoder KV cache\n");
        delete c;
        return nullptr;
    }

    if (params.codec_path && *params.codec_path) {
        auto cp = qwen3_tts_context_default_params();
        cp.n_threads = c->n_threads;
        cp.use_gpu = params.use_gpu;
        cp.verbosity = params.verbosity;
        c->codec = qwen3_tts_init_codec_only(params.codec_path, cp);
        if (!c->codec)
            fprintf(stderr, "breeze_tts_2: could not load the codec from '%s' — codes only\n", params.codec_path);
    }

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "breeze_tts_2: text encoder %uL/%ud (%d full layers) + backbone %uL/%ud + depth %uL/%ud, "
                "%u codebooks @ %u\n",
                hp.te_n_layers, hp.te_d_model, (int)std::count(hp.te_layer_full.begin(), hp.te_layer_full.end(), 1),
                hp.bb_n_layers, hp.bb_d_model, hp.dd_n_layers, hp.dd_d_model, hp.num_codebooks, hp.audio_vocab_size);
    }
    return c;
}

extern "C" void breeze_tts_2_free(struct breeze_tts_2_context* ctx) {
    delete ctx;
}

extern "C" uint32_t breeze_tts_2_capabilities(const struct breeze_tts_2_context* ctx) {
    (void)ctx;
    // ⚠ SINGLE BRANCH ONLY — and that is a scope statement, not a stub to be
    // read past. The KV caches are already allocated with an n_branch slice
    // dimension and both graph builders take a branch index, so the multi-
    // branch structure is present; what is NOT implemented is the per-branch
    // prompt assembly and the logits combine
    //     logits = uncond + s_ref*(ref - uncond) + s_ins*(ins - uncond)
    // at every backbone step and again per codebook inside the depth decoder.
    //
    // So this build reaches Voice Clone and plain TTS, and does NOT reach
    // Voice Design or Voice Direction. Those callers get a refusal from
    // breeze_tts_2_synthesize_guided rather than a single-branch result that
    // would sound plausible and ignore the instruction entirely.
    return 1u << BREEZE_CFG_NONE;
}

extern "C" void breeze_tts_2_pcm_free(float* pcm) {
    qwen3_tts_pcm_free(pcm);
}

extern "C" void breeze_tts_2_set_temperature(struct breeze_tts_2_context* ctx, float t) {
    if (ctx)
        ctx->params.temperature = t;
}
extern "C" void breeze_tts_2_set_topk(struct breeze_tts_2_context* ctx, int k) {
    if (ctx)
        ctx->params.topk = k;
}
extern "C" void breeze_tts_2_set_seed(struct breeze_tts_2_context* ctx, uint64_t seed) {
    if (ctx && seed != 0)
        ctx->rng_state = seed;
}
extern "C" void breeze_tts_2_set_n_threads(struct breeze_tts_2_context* ctx, int n) {
    if (!ctx || n <= 0)
        return;
    ctx->n_threads = n;
    core_cpu_backend::set_n_threads(ctx->backend_cpu, n);
    if (core_cpu_backend::is_cpu(ctx->backend))
        core_cpu_backend::set_n_threads(ctx->backend, n);
}
extern "C" void breeze_tts_2_set_max_new_tokens(struct breeze_tts_2_context* ctx, int n) {
    if (ctx && n > 0)
        ctx->max_new_tokens = n;
}

namespace {

// Shared body for every synthesis entry point.
static float* synth_impl(breeze_tts_2_context* c, const char* text, const char* instruction, const float* ref_pcm,
                         int ref_n_samples, const char* ref_text, int* out_n_samples, bool greedy, int frame_cap,
                         std::vector<std::vector<int32_t>>* out_frames, const int32_t* pre_ref_codes = nullptr,
                         int pre_ref_frames = 0) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!c || !text || !*text)
        return nullptr;

    // Reference audio -> codec codes. Uses the codec ENCODER, which is the same
    // qwen3-tts context the decoder comes from.
    std::vector<int32_t> ref_codes;
    int ref_frames = 0;
    if (pre_ref_codes && pre_ref_frames > 0) {
        // Caller already has codes (the diff harness, feeding the oracle's).
        // Skips the codec encoder AND the resampler, so neither can be blamed
        // for a downstream mismatch.
        ref_codes.assign(pre_ref_codes, pre_ref_codes + (size_t)pre_ref_frames * c->model.hp.num_codebooks);
        ref_frames = pre_ref_frames;
    } else if (ref_pcm && ref_n_samples > 0) {
        if (!c->codec) {
            fprintf(stderr, "breeze_tts_2: a reference clip was given but no codec is loaded — the reference "
                            "cannot be encoded. Pass the qwen3-tts-tokenizer-12hz GGUF via --codec-model.\n");
            return nullptr;
        }
        int32_t* codes = nullptr;
        int n_frames = 0;
        if (qwen3_tts_encode_pcm_to_codes(c->codec, ref_pcm, ref_n_samples, &codes, &n_frames) != 0 || !codes) {
            fprintf(stderr, "breeze_tts_2: reference audio encode failed\n");
            return nullptr;
        }
        ref_codes.assign(codes, codes + (size_t)n_frames * c->model.hp.num_codebooks);
        qwen3_tts_codes_free(codes);
        ref_frames = n_frames;
    }

    auto segs =
        build_segments(c->model, "S0", text, ref_text ? ref_text : "", ref_frames, instruction ? instruction : "");
    AssembledPrompt prompt;
    {
        breeze_bench_stage _b("prompt_assembly");
        if (!assemble_prompt(c, segs, ref_codes.empty() ? nullptr : ref_codes.data(), ref_frames, prompt))
            return nullptr;
    }

    GenOptions opt;
    opt.temperature = c->params.temperature;
    opt.depth_temperature = c->params.depth_temperature;
    opt.greedy = greedy;
    opt.max_frames = frame_cap > 0 ? frame_cap : c->max_new_tokens;
    // Repetition penalty is NOT in generation_config.json — infer.py and
    // api.py pass 1.1 at call time, so it is part of the shipped SYNTHESIS
    // recipe and not part of the model's defaults. The Kaggle reference oracle
    // calls generate() without it, which means the fixture's `codes` are
    // penalty-free; applying 1.1 in the greedy path would therefore diverge
    // from the fixture by construction and send the first bisect chasing a
    // difference that is ours, not the model's.
    opt.repetition_penalty = greedy ? 1.0f : c->model.hp.s_repetition_penalty;

    std::vector<std::vector<int32_t>> frames;
    if (!generate_codes(c, prompt, opt, frames))
        return nullptr;
    c->last_codes = frames;
    if (out_frames)
        *out_frames = frames;
    return decode_to_pcm(c, frames, out_n_samples);
}

} // namespace

extern "C" float* breeze_tts_2_synthesize(struct breeze_tts_2_context* ctx, const char* text, int* out_n_samples) {
    return synth_impl(ctx, text, nullptr, nullptr, 0, nullptr, out_n_samples, /*greedy*/ false, 0, nullptr);
}

extern "C" float* breeze_tts_2_synthesize_with_reference(struct breeze_tts_2_context* ctx, const char* text,
                                                         const float* ref_pcm, int ref_n_samples, const char* ref_text,
                                                         int* out_n_samples) {
    return synth_impl(ctx, text, nullptr, ref_pcm, ref_n_samples, ref_text, out_n_samples, false, 0, nullptr);
}

extern "C" float* breeze_tts_2_synthesize_guided(struct breeze_tts_2_context* ctx, const char* text,
                                                 const char* instruction, const float* ref_pcm, int ref_n_samples,
                                                 const char* ref_text, float cfg_scale_ref, float cfg_scale_ins,
                                                 int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx)
        return nullptr;
    const bool wants_ins = instruction && *instruction;
    const bool wants_ref = ref_pcm && ref_n_samples > 0;
    const breeze_cfg_mode mode =
        wants_ins ? (wants_ref ? BREEZE_CFG_BOTH : BREEZE_CFG_INS) : (wants_ref ? BREEZE_CFG_REF : BREEZE_CFG_NONE);
    const uint32_t caps = breeze_tts_2_capabilities(ctx);
    if ((caps & (1u << mode)) == 0) {
        // Refuse rather than downgrade. A single-branch run of a Voice Design
        // request produces fluent speech that ignores the instruction, and
        // nothing about the output says so.
        fprintf(stderr,
                "breeze_tts_2: this build does not implement classifier-free guidance (%s needs %d prompt "
                "branches). Voice Clone and plain TTS work; Voice Design and Voice Direction do not. "
                "Refusing rather than returning an unguided result that would sound fine and ignore the "
                "instruction.\n",
                mode == BREEZE_CFG_BOTH ? "Voice Direction" : (mode == BREEZE_CFG_INS ? "Voice Design" : "guidance"),
                mode == BREEZE_CFG_BOTH ? 3 : 2);
        return nullptr;
    }
    (void)cfg_scale_ref;
    (void)cfg_scale_ins;
    return synth_impl(ctx, text, instruction, ref_pcm, ref_n_samples, ref_text, out_n_samples, false, 0, nullptr);
}

// ===========================================================================
// Diff-harness entry points
// ===========================================================================

extern "C" int breeze_tts_2_run_text_encoder_dump(struct breeze_tts_2_context* ctx, const int32_t* seg_ids, int seg_len,
                                                  float* out_hidden, float** out_layers, int n_layer_slots) {
    if (!ctx || !seg_ids || seg_len <= 0)
        return -1;
    const int d = (int)ctx->model.hp.te_d_model;
    std::vector<float> hidden;
    std::vector<std::vector<float>> layers;
    if (!encode_segment(ctx, seg_ids, seg_len, hidden, out_layers ? &layers : nullptr))
        return -1;
    if (out_hidden)
        std::memcpy(out_hidden, hidden.data(), hidden.size() * sizeof(float));
    if (out_layers) {
        for (int j = 0; j < n_layer_slots && j < (int)layers.size(); j++) {
            if (out_layers[j] && !layers[(size_t)j].empty())
                std::memcpy(out_layers[j], layers[(size_t)j].data(), (size_t)seg_len * d * sizeof(float));
        }
    }
    return seg_len;
}

extern "C" int breeze_tts_2_run_text_proj_dump(struct breeze_tts_2_context* ctx, const float* hidden, int n,
                                               float* out_proj) {
    if (!ctx || !hidden || n <= 0 || !out_proj)
        return -1;
    std::vector<float> proj;
    if (!project_text(ctx, hidden, n, proj))
        return -1;
    std::memcpy(out_proj, proj.data(), proj.size() * sizeof(float));
    return n;
}

extern "C" int breeze_tts_2_run_backbone_dump(struct breeze_tts_2_context* ctx, const float* embeds, int L,
                                              float* out_hidden, float* out_logits, float** out_layers,
                                              int n_layer_slots) {
    if (!ctx || !embeds || L <= 0)
        return -1;
    const auto& hp = ctx->model.hp;
    if (L > ctx->bb_kv_max_ctx)
        return -1;
    ggml_backend_buffer_clear(ctx->bb_kv_buf, 0);

    std::vector<int32_t> positions((size_t)L);
    for (int i = 0; i < L; i++)
        positions[(size_t)i] = i;
    auto mask = build_causal_mask(L, L, 0);
    ggml_cgraph* gf = build_backbone_graph(ctx, 0, L, 0, out_layers != nullptr);
    if (!run_graph(ctx, gf,
                   {{"bb_embeds", embeds, (size_t)L * hp.bb_d_model * sizeof(float)},
                    {"bb_positions", positions.data(), positions.size() * sizeof(int32_t)},
                    {"bb_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)}},
                   "backbone dump"))
        return -1;
    if (out_hidden && !fetch(gf, "bb_hidden", out_hidden, hp.bb_d_model))
        return -1;
    if (out_logits && !fetch(gf, "bb_logits", out_logits, hp.lm_head_out))
        return -1;
    if (out_layers) {
        for (int j = 0; j < n_layer_slots && j < (int)hp.bb_n_layers; j++) {
            if (!out_layers[j])
                continue;
            char nm[32];
            std::snprintf(nm, sizeof(nm), "bb_layer%d", j);
            fetch(gf, nm, out_layers[j], hp.bb_d_model);
        }
    }
    return L;
}

extern "C" int breeze_tts_2_run_depth_dump(struct breeze_tts_2_context* ctx, const float* backbone_hidden, int32_t cb0,
                                           float** out_logits_cb, int32_t* out_codes) {
    if (!ctx || !backbone_hidden)
        return -1;
    const auto& hp = ctx->model.hp;
    std::vector<int32_t> codes;
    std::vector<std::vector<float>> logits;
    // Greedy: the acceptance criterion for this stage is argmax equality per
    // codebook, which a sampled pick cannot be held to.
    if (!depth_decode_frame(ctx, backbone_hidden, cb0, /*temperature*/ 0.0f, codes, &logits, 0))
        return -1;
    if (out_codes)
        std::memcpy(out_codes, codes.data(), codes.size() * sizeof(int32_t));
    if (out_logits_cb) {
        for (size_t i = 0; i < logits.size(); i++) {
            if (out_logits_cb[i] && !logits[i].empty())
                std::memcpy(out_logits_cb[i], logits[i].data(), logits[i].size() * sizeof(float));
        }
    }
    return 0;
}

extern "C" int breeze_tts_2_run_generate_codes(struct breeze_tts_2_context* ctx, const char* text, const float* ref_pcm,
                                               int ref_n_samples, const char* ref_text, int32_t* out_codes,
                                               int max_frames_cap) {
    if (!ctx || !text || !out_codes || max_frames_cap <= 0)
        return -1;
    std::vector<std::vector<int32_t>> frames;
    int n_samples = 0;
    float* pcm = synth_impl(ctx, text, nullptr, ref_pcm, ref_n_samples, ref_text, &n_samples,
                            /*greedy*/ true, max_frames_cap, &frames);
    if (pcm)
        qwen3_tts_pcm_free(pcm);
    // No codec is not a failure here: the fixture stage is the CODES.
    if (frames.empty())
        return -1;
    const int n_cb = (int)ctx->model.hp.num_codebooks;
    const int n = std::min((int)frames.size(), max_frames_cap);
    for (int f = 0; f < n; f++)
        std::memcpy(out_codes + (size_t)f * n_cb, frames[(size_t)f].data(), (size_t)n_cb * sizeof(int32_t));
    return n;
}

extern "C" int breeze_tts_2_run_prefill_embeds_dump(struct breeze_tts_2_context* ctx, const int32_t* ids,
                                                    const int32_t* text_mask, int L, const int32_t* ref_codes,
                                                    int ref_frames, float* out_embeds) {
    if (!ctx || !ids || !text_mask || L <= 0 || !out_embeds)
        return -1;
    // Rebuild the segment list from the ORACLE's ids and mask rather than from
    // our own tokenizer: a run of equal mask values IS a segment, which is
    // exactly the invariant templates.py maintains (text_ids_len carries one
    // entry per text segment, and the mask is false across the audio span).
    std::vector<PromptSegment> segs;
    int i = 0;
    while (i < L) {
        const bool is_text = text_mask[i] != 0;
        PromptSegment seg;
        seg.is_text = is_text;
        while (i < L && (text_mask[i] != 0) == is_text) {
            seg.ids.push_back(ids[i]);
            i++;
        }
        segs.push_back(std::move(seg));
    }
    AssembledPrompt prompt;
    if (!assemble_prompt(ctx, segs, ref_frames > 0 ? ref_codes : nullptr, ref_frames, prompt))
        return -1;
    if (prompt.L != L) {
        fprintf(stderr, "breeze_tts_2: reassembled prompt is %d tokens, fixture says %d\n", prompt.L, L);
        return -1;
    }
    std::memcpy(out_embeds, prompt.embeds.data(), prompt.embeds.size() * sizeof(float));
    return prompt.L;
}

extern "C" int breeze_tts_2_run_generate_codes_ref(struct breeze_tts_2_context* ctx, const char* text,
                                                   const char* ref_text, const int32_t* ref_codes, int ref_frames,
                                                   int32_t* out_codes, int max_frames_cap) {
    if (!ctx || !text || !out_codes || max_frames_cap <= 0)
        return -1;
    std::vector<std::vector<int32_t>> frames;
    int n_samples = 0;
    float* pcm = synth_impl(ctx, text, nullptr, nullptr, 0, ref_text, &n_samples, /*greedy*/ true, max_frames_cap,
                            &frames, ref_codes, ref_frames);
    if (pcm)
        qwen3_tts_pcm_free(pcm);
    if (frames.empty())
        return -1;
    const int n_cb = (int)ctx->model.hp.num_codebooks;
    const int n = std::min((int)frames.size(), max_frames_cap);
    for (int f = 0; f < n; f++)
        std::memcpy(out_codes + (size_t)f * n_cb, frames[(size_t)f].data(), (size_t)n_cb * sizeof(int32_t));
    return n;
}

extern "C" int breeze_tts_2_run_prompt_dump(struct breeze_tts_2_context* ctx, const char* text, const char* ref_text,
                                            int n_ref_audio_frames, const char* instruction, int32_t* out_ids,
                                            int32_t* out_text_mask, int32_t* out_seg_lens, int max_L, int max_segs,
                                            int* out_n_segs) {
    if (!ctx || !text)
        return -1;
    auto segs = build_segments(ctx->model, "S0", text, ref_text ? ref_text : "", n_ref_audio_frames,
                               instruction ? instruction : "");
    int L = 0;
    int n_segs = 0;
    for (const auto& s : segs) {
        for (int32_t id : s.ids) {
            if (L < max_L) {
                if (out_ids)
                    out_ids[L] = id;
                if (out_text_mask)
                    out_text_mask[L] = s.is_text ? 1 : 0;
            }
            L++;
        }
        if (s.is_text) {
            if (n_segs < max_segs && out_seg_lens)
                out_seg_lens[n_segs] = (int32_t)s.ids.size();
            n_segs++;
        }
    }
    if (out_n_segs)
        *out_n_segs = n_segs;
    return L;
}
