// mimo_asr.cpp — runtime for XiaomiMiMo/MiMo-V2.5-ASR
//
// SCAFFOLD ONLY (April 2026): the model loads, the tensors are
// populated from the GGUF, and the transcribe entry point returns
// an explicit "not implemented" string so callers can integrate
// against the C ABI today and the encoder/LLM forward passes can
// land incrementally without breaking the build.
//
// Architecture (from the converter docstring + config.json):
//
//   audio side:
//     - 8-channel RVQ codes (12.5 fps after group_size=4 → 25 Hz
//       in the .nemo's frame definition; see `audio.frame_rate`)
//     - speech_embeddings: per-channel embedding tables (codebook
//       sized) → group_proj that fuses the 8 channels into a single
//       1024-d sequence
//     - 6-layer input_local_transformer (1024d, 64h × 16d head,
//       SiLU MLP, RoPE) processes that sequence
//     - hidden_proj projects to LM hidden size
//
//   LLM side:
//     - 36-layer Qwen2 (3584d, 32 heads, 8 KV heads, head_dim 112,
//       SiLU SwiGLU, RoPE theta 640000, RMSNorm)
//     - tied embedding for embed_tokens / lm_head
//     - codebook_head: per-channel head used during ASR-direction
//       inference is the LLM's own lm_head; the per-codebook heads
//       are present in the checkpoint but only used in TTS-direction
//       generation. We can ignore them for ASR.
//
// Known follow-ups (PLAN #51):
//   - Audio: load the mimo-tokenizer GGUF (cstr/mimo-tokenizer-GGUF)
//     and run its encoder over PCM → 8-channel RVQ code stream.
//   - Wire the codes through speech_embeddings → input_local_transformer
//     → hidden_proj → LLM.embed augmentation as the prefill prompt.
//   - LLM forward: standard Qwen2 — fully covered by
//     core_attn::kv_self_attn + core_ffn::swiglu, same call site as
//     qwen3-asr / gemma4-e2b.

#include "mimo_asr.h"

#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "mimo_tokenizer.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct mimo_asr_hp {
    // LLM (Qwen2 — model.layers.{0..35}). Defaults match config.json
    // for XiaomiMiMo/MiMo-V2.5-ASR but the GGUF metadata is the source
    // of truth and overrides them in mimo_asr_init_from_file.
    uint32_t llm_hidden = 4096;
    uint32_t llm_layers = 36;
    uint32_t llm_heads = 32;
    uint32_t llm_kv_heads = 8;
    uint32_t llm_head_dim = 128; // hidden / heads
    uint32_t llm_intermediate = 11008;
    uint32_t llm_vocab = 151680;
    uint32_t llm_max_pos = 8192;
    float llm_rope_theta = 640000.0f;
    float llm_rms_eps = 1e-6f;

    // Audio (input_local_transformer — audio.blk.{0..5}).
    uint32_t audio_channels = 8;
    uint32_t audio_group_size = 4;
    uint32_t audio_layers = 6;
    uint32_t audio_dim = 1024;
    uint32_t audio_heads = 64;
    uint32_t audio_head_dim = 16;
    uint32_t audio_intermediate = 4096;
    uint32_t audio_out_hidden = 4096; // == llm_hidden after group_proj

    // Per-channel speech vocab + zero-emb idx, parsed from the
    // hyphen-separated KVs `mimo_asr.audio.speech_vocab.{0..7}` and
    // `mimo_asr.audio.speech_zeroemb.{0..7}`.
    std::array<uint32_t, 8> speech_vocab{1025, 1025, 129, 129, 129, 129, 129, 129};
    std::array<uint32_t, 8> speech_zeroemb_idx{1024, 1024, 128, 128, 128, 128, 128, 128};
};

// One Qwen2 transformer block. Same layout for the 6L audio
// input_local_transformer and the 36L Qwen2 LM — only the dimensions
// differ. Qwen2 has Q/K/V projection biases but no o-projection bias,
// and no per-head Q/K RMSNorm (those are Qwen3-only).
struct mimo_asr_qwen2_block {
    ggml_tensor* attn_norm_w = nullptr; // RMSNorm γ, no bias
    // Either separate Q/K/V (legacy GGUFs) or a fused `attn.qkv` pair
    // (post-PLAN #60d). When `attn_qkv_w` is non-null, the LM forward
    // takes the fused single-matmul path through `core_attn::kv_self_attn`
    // and the per-projection q/k/v_w/_b are nullptr. The 6L audio
    // input_local_transformer always uses the separate path (the converter
    // only fuses LM layers, and the audio bidirectional graph reads
    // attn_q_w/k_w/v_w directly).
    ggml_tensor* attn_qkv_w = nullptr; // fused [d, q_dim + 2*kv_dim], or nullptr
    ggml_tensor* attn_qkv_b = nullptr; // fused [q_dim + 2*kv_dim] F32, or nullptr
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;   // no bias
    ggml_tensor* ffn_norm_w = nullptr; // RMSNorm γ, no bias
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

// 6-layer audio input transformer + speech codebook embeddings +
// group-fusion projection. `hidden_proj_w` (TTS-direction) is loaded
// lazily and stays nullptr for ASR.
struct mimo_asr_audio_tower {
    std::vector<mimo_asr_qwen2_block> blocks; // 6
    ggml_tensor* norm_w = nullptr;            // audio.norm.weight (RMSNorm γ)
    std::array<ggml_tensor*, 8> speech_emb{}; // audio.emb.{0..7}.weight
    ggml_tensor* group_proj_w = nullptr;      // audio.group_proj.weight, [out_hidden, group_size*input_dim]
};

// 36-layer Qwen2 LM. lm_head is NOT tied to embeddings.
struct mimo_asr_llm {
    ggml_tensor* embed_w = nullptr;           // llm.embed.weight, [hidden, vocab]
    std::vector<mimo_asr_qwen2_block> blocks; // 36
    ggml_tensor* final_norm_w = nullptr;      // llm.final_norm.weight
    ggml_tensor* lm_head_w = nullptr;         // llm.lm_head.weight, [hidden, vocab]
};

struct mimo_asr_model {
    mimo_asr_audio_tower audio;
    mimo_asr_llm llm;
};

} // namespace

struct mimo_asr_context {
    mimo_asr_context_params params{};
    int n_threads = 4;

    mimo_asr_hp hp;
    mimo_asr_model model;

    // Backends + weights
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    // PLAN #69a: optional second buffer for layers spilled to CPU.
    ggml_backend_buffer_t buf_w_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;

    // KV cache for the 36L Qwen2 LM (F16, ne = (head_dim, max_ctx,
    // n_kv_heads, n_layers)). Lazy-initialised on first prefill call.
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    // 51b' cached T=1 step graph. Built lazily on first decode step of a
    // transcribe call with fixed_kv_len = kv_max_ctx so the topology is
    // invariant across n_past; subsequent steps reuse the same plan via
    // skip_plan (no reset/alloc), updating only input tensor values.
    // Reset to nullptr at the start of every transcribe (kv_max_ctx may
    // change) and after extract_stage runs (which rebuilds the prefill
    // graph in the shared compute_meta).
    ggml_cgraph* step_t1_gf = nullptr;
    int step_t1_fixed_kv_len = 0;

    // Tokeniser GGUF path — set via mimo_asr_set_tokenizer_path before
    // the first transcribe call. Empty until set. The tokenizer context
    // is instantiated lazily on first transcribe (heavy: 569 tensors).
    std::string tokenizer_path;
    mimo_tokenizer_context* tokenizer = nullptr;

    std::vector<std::string> vocab;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;

    // Special-token ids resolved at init from the vocab (or fallbacks for
    // the stale Q4_K with a truncated vocab — step 9 fixes this).
    int32_t id_empty = 151667;    // <|empty|>
    int32_t id_im_start = 151644; // <|im_start|>
    int32_t id_im_end = 151645;   // <|im_end|>
    int32_t id_eos = 151643;      // <|endoftext|>
    int32_t id_sosp = 151651;     // <|sosp|>
    int32_t id_eosp = 151652;     // <|eosp|>
    int32_t id_eot = 151653;      // <|eot|>
    int32_t id_eostm = 151654;    // <|eostm|>
};

static uint32_t mimo_kv_u32(gguf_context* ctx, const char* key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_u32(ctx, id) : def;
}
static float mimo_kv_f32(gguf_context* ctx, const char* key, float def) {
    int64_t id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}

extern "C" struct mimo_asr_context_params mimo_asr_context_default_params(void) {
    mimo_asr_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.0f;
    return p;
}

extern "C" struct mimo_asr_context* mimo_asr_init_from_file(const char* path_model,
                                                            struct mimo_asr_context_params params) {
    auto* ctx = new mimo_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Read GGUF metadata
    ggml_context* gctx_dummy = nullptr;
    gguf_init_params gp = {/*no_alloc=*/true, &gctx_dummy};
    gguf_context* gctx = gguf_init_from_file(path_model, gp);
    if (!gctx) {
        fprintf(stderr, "mimo_asr: failed to read GGUF '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.llm_hidden = mimo_kv_u32(gctx, "mimo_asr.llm.hidden_size", hp.llm_hidden);
    hp.llm_layers = mimo_kv_u32(gctx, "mimo_asr.llm.num_layers", hp.llm_layers);
    hp.llm_heads = mimo_kv_u32(gctx, "mimo_asr.llm.num_heads", hp.llm_heads);
    hp.llm_kv_heads = mimo_kv_u32(gctx, "mimo_asr.llm.num_kv_heads", hp.llm_kv_heads);
    hp.llm_intermediate = mimo_kv_u32(gctx, "mimo_asr.llm.intermediate_size", hp.llm_intermediate);
    hp.llm_vocab = mimo_kv_u32(gctx, "mimo_asr.llm.vocab_size", hp.llm_vocab);
    hp.llm_max_pos = mimo_kv_u32(gctx, "mimo_asr.llm.max_position_embeddings", hp.llm_max_pos);
    hp.llm_rope_theta = mimo_kv_f32(gctx, "mimo_asr.llm.rope_theta", hp.llm_rope_theta);
    hp.llm_rms_eps = mimo_kv_f32(gctx, "mimo_asr.llm.rms_norm_eps", hp.llm_rms_eps);

    hp.audio_channels = mimo_kv_u32(gctx, "mimo_asr.audio.channels", hp.audio_channels);
    hp.audio_group_size = mimo_kv_u32(gctx, "mimo_asr.audio.group_size", hp.audio_group_size);
    hp.audio_layers = mimo_kv_u32(gctx, "mimo_asr.audio.input_layers", hp.audio_layers);
    hp.audio_dim = mimo_kv_u32(gctx, "mimo_asr.audio.input_dim", hp.audio_dim);
    hp.audio_heads = mimo_kv_u32(gctx, "mimo_asr.audio.input_heads", hp.audio_heads);
    hp.audio_head_dim = mimo_kv_u32(gctx, "mimo_asr.audio.input_head_dim", hp.audio_head_dim);
    hp.audio_intermediate = mimo_kv_u32(gctx, "mimo_asr.audio.input_intermediate", hp.audio_intermediate);

    // Vocab + BPE merges + token_to_id reverse map. Step 9 reconverts the
    // GGUF with `tokenizer.ggml.merges` populated; the encode side
    // (`tokenize_text`) falls back to a per-byte path when merges are
    // empty, so the loader never errors on the stale-vocab GGUF — but
    // transcribe can't produce sane prompts without merges either.
    int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        ctx->vocab.resize(n);
        ctx->token_to_id.reserve((size_t)n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s) {
                ctx->vocab[i] = s;
                ctx->token_to_id.emplace(s, (int32_t)i);
            }
        }
    }
    int merges_key = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (merges_key >= 0) {
        int n = gguf_get_arr_n(gctx, merges_key);
        ctx->merge_rank.reserve((size_t)n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, merges_key, i);
            if (s)
                ctx->merge_rank.emplace(s, (int32_t)i);
        }
    }
    gguf_free(gctx);

    // Resolve special-token ids from the vocab. Defaults match MiMo's
    // stable special-token block (151643..151667); the lookup wins when
    // the GGUF carries the full vocab (post step 9).
    auto find_id = [&](const char* name, int32_t fallback) {
        auto it = ctx->token_to_id.find(name);
        return it != ctx->token_to_id.end() ? it->second : fallback;
    };
    ctx->id_empty = find_id("<|empty|>", ctx->id_empty);
    ctx->id_im_start = find_id("<|im_start|>", ctx->id_im_start);
    ctx->id_im_end = find_id("<|im_end|>", ctx->id_im_end);
    ctx->id_eos = find_id("<|endoftext|>", ctx->id_eos);
    ctx->id_sosp = find_id("<|sosp|>", ctx->id_sosp);
    ctx->id_eosp = find_id("<|eosp|>", ctx->id_eosp);
    ctx->id_eot = find_id("<|eot|>", ctx->id_eot);
    ctx->id_eostm = find_id("<|eostm|>", ctx->id_eostm);

    if (params.verbosity >= 1) {
        fprintf(stderr,
                "mimo_asr: vocab=%zu merges=%zu specials: empty=%d im_start=%d im_end=%d "
                "eos=%d sosp=%d eosp=%d\n",
                ctx->vocab.size(), ctx->merge_rank.size(), ctx->id_empty, ctx->id_im_start, ctx->id_im_end, ctx->id_eos,
                ctx->id_sosp, ctx->id_eosp);
    }

    // Backends
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend_cpu) {
        fprintf(stderr, "mimo_asr: failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ctx->backend_cpu;
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    // PLAN #72: load weights onto the user-picked backend (GPU when
    // use_gpu=true). Was hardcoded to backend_cpu under the old
    // assumption that Q4_K CPU SIMD beat the Metal/CUDA path. Today's
    // GPU Q4_K kernels are mature and dominate CPU on big-enough
    // models — mimo_asr at 1.4B params is comfortably big enough.
    // PLAN #69a: when CRISPASR_N_GPU_LAYERS is set and < total LLM
    // layers, route model.layers.<il>.* with il >= N onto the CPU backend.
    core_gguf::WeightLoad wl;
    int n_gpu_layers_env = -1;
    if (const char* s = std::getenv("CRISPASR_N_GPU_LAYERS")) {
        n_gpu_layers_env = std::atoi(s);
    }
    const int total_layers = (int)hp.llm_layers;
    const bool do_split = ctx->backend_cpu && ctx->backend_cpu != ctx->backend && n_gpu_layers_env >= 0 &&
                          n_gpu_layers_env < total_layers;
    if (do_split) {
        core_gguf::LayerSplitConfig cfg{"model.layers.", n_gpu_layers_env};
        if (!core_gguf::load_weights_split(path_model, ctx->backend, ctx->backend_cpu,
                                           core_gguf::is_gpu_tensor_with_prefix, &cfg, "mimo_asr", wl)) {
            fprintf(stderr, "mimo_asr: split load failed from '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
        fprintf(stderr, "mimo_asr: layer offload: gpu=[0,%d), cpu=[%d,%d) (CRISPASR_N_GPU_LAYERS=%d)\n",
                n_gpu_layers_env, n_gpu_layers_env, total_layers, n_gpu_layers_env);
    } else {
        if (!core_gguf::load_weights(path_model, ctx->backend, "mimo_asr", wl)) {
            fprintf(stderr, "mimo_asr: failed to load weights from '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;
    ctx->buf_w_cpu = wl.buf_cpu;
    ctx->tensors = std::move(wl.tensors);

    // ---- Bind tensors into typed structs ------------------------------
    // The 36L Qwen2 LM lives under `model.layers.{0..35}.*` (the converter
    // rename rule for `model.layers.` is missing on purpose — see TODO #51
    // gotcha 2). The 6L `input_local_transformer` audio path is at
    // `audio.blk.{0..5}.*` + `audio.emb.{0..7}` + `audio.group_proj` +
    // `audio.norm`. We deliberately ignore `llm.blk.*`, `llm.codebook_head.*`,
    // `llm.norm.*`, and `audio.hidden_proj` — those are TTS-direction
    // tensors unused by ASR.
    auto require_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ctx->tensors, name.c_str(), "mimo_asr");
    };
    auto try_t = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::try_get(ctx->tensors, name.c_str());
    };

    auto bind_qwen2_block = [&](mimo_asr_qwen2_block& b, const std::string& prefix) -> bool {
        b.attn_norm_w = require_t(prefix + ".attn_norm.weight");
        // PLAN #60d: prefer the fused `attn.qkv.{weight,bias}` pair when
        // present; fall back to separate Q/K/V tensors so legacy GGUFs
        // (and the audio.blk.* path, which the converter never fuses) keep
        // working unchanged.
        b.attn_qkv_w = try_t(prefix + ".attn.qkv.weight");
        b.attn_qkv_b = try_t(prefix + ".attn.qkv.bias");
        if (!b.attn_qkv_w) {
            b.attn_q_w = require_t(prefix + ".attn.q.weight");
            b.attn_q_b = require_t(prefix + ".attn.q.bias");
            b.attn_k_w = require_t(prefix + ".attn.k.weight");
            b.attn_k_b = require_t(prefix + ".attn.k.bias");
            b.attn_v_w = require_t(prefix + ".attn.v.weight");
            b.attn_v_b = require_t(prefix + ".attn.v.bias");
        }
        b.attn_o_w = require_t(prefix + ".attn.o.weight");
        b.ffn_norm_w = require_t(prefix + ".ffn_norm.weight");
        b.ffn_gate_w = require_t(prefix + ".ffn.gate.weight");
        b.ffn_up_w = require_t(prefix + ".ffn.up.weight");
        b.ffn_down_w = require_t(prefix + ".ffn.down.weight");
        const bool qkv_ok = b.attn_qkv_w
                                ? (b.attn_qkv_b != nullptr)
                                : (b.attn_q_w && b.attn_q_b && b.attn_k_w && b.attn_k_b && b.attn_v_w && b.attn_v_b);
        return b.attn_norm_w && qkv_ok && b.attn_o_w && b.ffn_norm_w && b.ffn_gate_w && b.ffn_up_w && b.ffn_down_w;
    };

    auto& a = ctx->model.audio;
    a.blocks.resize(hp.audio_layers);
    for (uint32_t l = 0; l < hp.audio_layers; l++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "audio.blk.%u", l);
        if (!bind_qwen2_block(a.blocks[l], buf)) {
            fprintf(stderr, "mimo_asr: missing tensors in %s.*\n", buf);
            delete ctx;
            return nullptr;
        }
    }
    a.norm_w = require_t("audio.norm.weight");
    a.group_proj_w = require_t("audio.group_proj.weight");
    for (uint32_t i = 0; i < hp.audio_channels; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "audio.emb.%u.weight", i);
        a.speech_emb[i] = require_t(buf);
        if (!a.speech_emb[i]) {
            fprintf(stderr, "mimo_asr: missing %s\n", buf);
            delete ctx;
            return nullptr;
        }
    }
    if (!a.norm_w || !a.group_proj_w) {
        delete ctx;
        return nullptr;
    }

    auto& l = ctx->model.llm;
    l.embed_w = require_t("llm.embed.weight");
    l.final_norm_w = require_t("llm.final_norm.weight");
    l.lm_head_w = require_t("llm.lm_head.weight");
    if (!l.embed_w || !l.final_norm_w || !l.lm_head_w) {
        delete ctx;
        return nullptr;
    }
    l.blocks.resize(hp.llm_layers);
    for (uint32_t li = 0; li < hp.llm_layers; li++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "model.layers.%u", li);
        if (!bind_qwen2_block(l.blocks[li], buf)) {
            fprintf(stderr, "mimo_asr: missing tensors in %s.*\n", buf);
            delete ctx;
            return nullptr;
        }
    }
    (void)try_t; // hidden_proj is intentionally unused for ASR

    // ---- Scheduler + compute_meta -------------------------------------
    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "mimo_asr: loaded %zu tensors  llm=%uL/%u  audio=%uL/%u (×%u channels)\n", ctx->tensors.size(),
                hp.llm_layers, hp.llm_hidden, hp.audio_layers, hp.audio_dim, hp.audio_channels);
    }
    return ctx;
}

// ===========================================================================
// LM forward — 36L Qwen2 with biased Q/K/V (KV cache via core_attn).
// Audio path — 6L bidirectional Qwen2 input_local_transformer + speech
// embedding sum + group_proj fusion.
// Stage extraction API — runs prefill on a caller-supplied [9, T_total]
// input_ids tensor and returns one of:
//   prefill_audio_features      [T_groups, llm_hidden]    F32
//   prefill_inputs_embeds       [T_groups, llm_hidden]    F32
//   prefill_last_hidden         [llm_hidden]              F32
//   prefill_text_logits_step0   [vocab]                   F32
// ===========================================================================

static bool mimo_asr_kv_init(mimo_asr_context* ctx, int max_ctx) {
    if (ctx->kv_k && ctx->kv_max_ctx >= max_ctx)
        return true;
    if (ctx->kv_buf) {
        ggml_backend_buffer_free(ctx->kv_buf);
        ctx->kv_buf = nullptr;
    }
    if (ctx->kv_ctx) {
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
    }
    const auto& hp = ctx->hp;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_kv_heads;
    const int n_lay = (int)hp.llm_layers;
    // PLAN #60e + #69e: per-half KV dtype.
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("mimo_asr");
    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "mimo_kv_k");
    ggml_set_name(ctx->kv_v, "mimo_kv_v");
    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    // PLAN #69b: optional KV-on-CPU spill.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "mimo_asr");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        fprintf(stderr, "mimo_asr: failed to alloc KV buffer (%zu bytes)\n", kbytes + vbytes);
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    // Zero the cache so any inadvertently-read slot is bit-stable across runs.
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;
    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "mimo_asr: kv cache %d MiB k=%s v=%s (on %s, head_dim=%d max_ctx=%d n_kv=%d n_layers=%d)\n",
                (int)((kbytes + vbytes) / 1048576), ggml_type_name(kv_pair.k), ggml_type_name(kv_pair.v),
                kv_backend == ctx->backend_cpu ? "cpu" : "gpu", hd, max_ctx, n_kv, n_lay);
    }
    return true;
}

namespace {

// Build the input_local_transformer + speech_group_downcast graph. Returns
// `inputs_embeds [llm_hidden, T_groups]`.
//
// Inputs (set externally before compute):
//   speech_codes_c[8] : [T_groups*group_size] I32 — per-channel codes
//   speech_emb_mask   : [audio_dim, T_groups*group_size] F32
//                       1 where (text==empty AND code!=zeroemb) else 0
//                       (channel-independent because the speech_active mask
//                        propagates uniformly; channel-specific zeroemb is
//                        baked in by replacing those code IDs with index 0
//                        of an embedding table — but get_rows reads what's
//                        there.  Simplest: pass per-channel masks and AND
//                        with the speech_active mask.)
//
// We pass per-channel masks separately and AND in graph form.
struct AudioGraphIO {
    ggml_tensor* speech_codes_c[8] = {};       // [T_groups*4] I32
    ggml_tensor* combined_mask_c[8] = {};      // [audio_dim, T_groups*4] F32
    ggml_tensor* speech_active_mask = nullptr; // [llm_hidden, T_groups] F32 (broadcast for group_proj output gating)
    ggml_tensor* text_input_ids = nullptr;     // [T_groups] I32 (only meaningful text, not -100s)
    ggml_tensor* text_zero_mask = nullptr;     // [llm_hidden, T_groups] F32 (1 where text_input != empty else 0)
    int T_groups = 0;
    int group_size = 0;
};

static ggml_tensor* build_input_local_block(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* x,
                                            const mimo_asr_qwen2_block& b, const mimo_asr_hp& hp, int n_groups,
                                            int group_size, ggml_tensor* positions) {
    (void)gf; // accepted for symmetry with helpers that need it for set_rows
    // x : [audio_dim, group_size, n_groups]
    // `positions` is a shared 1D I32 tensor of length group_size, set by
    // the caller to [0, 1, ..., group_size-1].

    const int d = (int)hp.audio_dim;
    const int hd = (int)hp.audio_head_dim;
    const int n_h = (int)hp.audio_heads;
    const int gs = group_size;
    const int ng = n_groups;
    const float eps = hp.llm_rms_eps;
    const float scale = 1.0f / std::sqrt((float)hd);

    ggml_tensor* residual = x;

    // Pre-attn norm.
    ggml_tensor* h = ggml_rms_norm(ctx0, x, eps);
    h = ggml_mul(ctx0, h, b.attn_norm_w);

    // Q/K/V projections (with biases). h is [d, gs, ng] → flatten to
    // [d, gs*ng] for the matmul, then reshape result back.
    ggml_tensor* h2 = ggml_cont(ctx0, ggml_reshape_2d(ctx0, h, d, gs * ng));
    ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_q_w, h2), b.attn_q_b);
    ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_k_w, h2), b.attn_k_b);
    ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, b.attn_v_w, h2), b.attn_v_b);

    Q = ggml_reshape_4d(ctx0, Q, hd, n_h, gs, ng);
    K = ggml_reshape_4d(ctx0, K, hd, n_h, gs, ng);
    V = ggml_reshape_4d(ctx0, V, hd, n_h, gs, ng);

    // RoPE expects (hd, n_h, T, B) — exactly the post-reshape layout, with
    // T=gs at ne[2] matching positions->ne[0]=gs. Apply rotation BEFORE
    // permuting to flash_attn_ext's layout.
    Q = ggml_rope_ext(ctx0, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, hp.llm_rope_theta,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx0, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, (int)hp.llm_max_pos, hp.llm_rope_theta,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Permute (hd, n_h, gs, ng) → (hd, gs, n_h, ng) so flash_attn_ext sees
    // T=gs at ne[1] (its expected layout for n_tokens).
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

    // ggml_flash_attn_ext expects (hd, T, n_h, B). We have exactly that.
    // mask=nullptr → bidirectional, attends within each (n_h, B) slice
    // across T positions. The "B" dim here is ng (per-group batch).
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, /*mask*/ nullptr, scale, 0.0f, 0.0f);
    // attn: (hd, n_h, T=gs, B=ng) ?  Actually flash_attn_ext output is
    // (hd, n_h, T, B) per ggml convention. Reshape to [d, gs*ng].
    attn = ggml_reshape_2d(ctx0, attn, hd * n_h, gs * ng);

    // Output projection (no bias). Reshape back to [d, gs, ng] so the
    // residual add and downstream FFN see the same 3D layout as the input.
    attn = ggml_mul_mat(ctx0, b.attn_o_w, attn);
    attn = ggml_reshape_3d(ctx0, attn, d, gs, ng);
    ggml_tensor* y = ggml_add(ctx0, residual, attn);

    // FFN
    residual = y;
    h = ggml_rms_norm(ctx0, y, eps);
    h = ggml_mul(ctx0, h, b.ffn_norm_w);
    ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
    return ggml_add(ctx0, residual, mlp);
}

// Build the full prefill graph: speech embedding sum → input_local_transformer
// → group_proj → text fusion → 36L Qwen2 LM → final norm + lm_head.
//
// Inputs (must be set on the context before compute):
//   speech_codes_c[i]      [T_groups*group_size] I32
//   combined_mask_c[i]     [audio_dim, T_groups*group_size] F32
//   text_input_ids         [T_groups] I32
//   text_zero_mask         [llm_hidden, T_groups] F32 (1 where text!=empty)
//   speech_active_mask     [llm_hidden, T_groups] F32 (1 where text==empty)
//   ilt_positions          [group_size] I32 = 0..gs-1
//   positions              [T_groups] I32 = 0..T_groups-1
//   causal_mask            [Lk=T_groups, T_groups] F16
//
// Outputs (extracted by name after compute):
//   prefill_audio_features       [llm_hidden, T_groups] F32 (post group_proj)
//   prefill_inputs_embeds        [llm_hidden, T_groups] F32 (LM input)
//   prefill_last_hidden          [llm_hidden, 1]        F32 (post final norm)
//   prefill_text_logits_step0    [vocab, 1]             F32 (lm_head out)
//
// `diag_captures` controls whether the four diag stages above are kept
// resident as graph outputs. The diff harness (mimo_asr_extract_stage)
// passes true to read them back; the production transcribe path passes
// false so the scheduler can reuse those buffers earlier — same compute,
// less memory pressure on the allocator. `prefill_text_logits_step0` is
// the consumed output and is always kept. (~5% win + cleaner alloc per
// PLAN #51 perf wave.)
static ggml_cgraph* mimo_asr_build_prefill_graph(mimo_asr_context* ctx, int T_groups, int n_past, bool diag_captures) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int d = (int)hp.llm_hidden;
    const int ad = (int)hp.audio_dim;
    const int gs = (int)hp.audio_group_size;
    const int n_q = (int)hp.llm_heads;
    const int n_kv = (int)hp.llm_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = T_groups;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ---- Audio path: per-channel embedding lookup + sum, masked ----
    ggml_tensor* speech_sum = nullptr;
    for (int i = 0; i < (int)hp.audio_channels; i++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "speech_codes_%d", i);
        ggml_tensor* codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T * gs);
        ggml_set_input(codes);
        ggml_set_name(codes, nm);

        snprintf(nm, sizeof(nm), "combined_mask_%d", i);
        ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ad, T * gs);
        ggml_set_input(mask);
        ggml_set_name(mask, nm);

        ggml_tensor* e = ggml_get_rows(ctx0, m.audio.speech_emb[i], codes); // [ad, T*gs]
        e = ggml_mul(ctx0, e, mask);                                        // mask zero rows
        speech_sum = speech_sum ? ggml_add(ctx0, speech_sum, e) : e;
    }
    // speech_sum: [ad, T*gs]

    // Reshape to [ad, gs, T] for per-group transformer. Force contiguous
    // because subsequent matmul + reshape paths assume row-major contig.
    ggml_tensor* x = ggml_cont(ctx0, ggml_reshape_3d(ctx0, speech_sum, ad, gs, T));

    // Shared per-group RoPE positions tensor [gs] = 0..gs-1.
    ggml_tensor* ilt_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, gs);
    ggml_set_input(ilt_positions);
    ggml_set_name(ilt_positions, "ilt_positions");

    // 6L input_local_transformer (full bidirectional, per-group).
    for (uint32_t il = 0; il < hp.audio_layers; il++) {
        x = build_input_local_block(ctx0, gf, x, m.audio.blocks[il], hp, T, gs, ilt_positions);
    }
    // Final audio.norm RMSNorm (matches upstream
    // input_local_transformer.norm of the Qwen2Model trunk).
    x = ggml_rms_norm(ctx0, x, eps);
    x = ggml_mul(ctx0, x, m.audio.norm_w);
    // x: [ad, gs, T]

    // Reshape to [ad*gs, T] for group_proj. ggml_reshape_2d collapses
    // contiguous dims; since gs is the "middle" dim and T is outer, we
    // need a contig copy first. The current x has ne=[ad, gs, T]; flatten
    // to [ad*gs, T].
    x = ggml_cont(ctx0, x);
    x = ggml_reshape_2d(ctx0, x, ad * gs, T);

    // group_proj: weight is [ad*gs, llm_hidden]; ggml_mul_mat(W, x) does
    // y = W^T @ x with W stored as [in_dim, out_dim]. Result: [d, T].
    x = ggml_mul_mat(ctx0, m.audio.group_proj_w, x);
    // x: [d, T]

    // Mask non-speech positions (text==empty positions are speech, so
    // multiply by `speech_active_mask` which is 1 there). Equivalent to
    // upstream: speech_grouped *= is_speech mask (broadcast on hidden).
    ggml_tensor* speech_active_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_input(speech_active_mask);
    ggml_set_name(speech_active_mask, "speech_active_mask");
    x = ggml_mul(ctx0, x, speech_active_mask);

    // ---- Capture: prefill_audio_features (the speech_grouped before fusion) ----
    // ggml_set_output is mandatory on every named capture tensor: without
    // it, the backend scheduler treats the cont/add nodes as ordinary
    // intermediates and may reuse their buffers when allocating later
    // ops (e.g. inputs_embeds), so the values read back via
    // ggml_graph_get_tensor end up clobbered. Symptom: the per-stage
    // cosine looks fine in isolation but inputs_embeds collapses to
    // ~0 against the ref. Gated on diag_captures: the production path
    // skips the cont entirely (it's a pure clone of x).
    if (diag_captures) {
        ggml_tensor* audio_features = ggml_cont(ctx0, x);
        ggml_set_name(audio_features, "prefill_audio_features");
        ggml_set_output(audio_features);
        ggml_build_forward_expand(gf, audio_features);
    }

    // ---- Text embedding lookup + zero mask + fusion ----
    ggml_tensor* text_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(text_ids);
    ggml_set_name(text_ids, "text_input_ids");

    ggml_tensor* text_embeds = ggml_get_rows(ctx0, m.llm.embed_w, text_ids); // [d, T]

    ggml_tensor* text_zero_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_input(text_zero_mask);
    ggml_set_name(text_zero_mask, "text_zero_mask");
    text_embeds = ggml_mul(ctx0, text_embeds, text_zero_mask);

    // Capture post-mask text_embeds in isolation, so we can bisect
    // text-vs-audio-vs-fusion when inputs_embeds drifts.
    if (diag_captures) {
        ggml_tensor* dbg_text_embeds = ggml_cont(ctx0, text_embeds);
        ggml_set_name(dbg_text_embeds, "prefill_text_embeds");
        ggml_set_output(dbg_text_embeds); // see prefill_audio_features comment
        ggml_build_forward_expand(gf, dbg_text_embeds);
    }

    ggml_tensor* inputs_embeds = ggml_add(ctx0, text_embeds, x); // [d, T]
    if (diag_captures) {
        ggml_set_name(inputs_embeds, "prefill_inputs_embeds");
        ggml_set_output(inputs_embeds);
        // Mark as build-target so it survives optimisation.
        ggml_build_forward_expand(gf, inputs_embeds);
    }

    // ---- 36L Qwen2 LM (KV-cached prefill / step decode) ----
    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(n_past + T <= ctx->kv_max_ctx);

    // For step decode (T==1, n_past>0), positions are filled by the host
    // with [n_past..n_past+T-1] and the causal mask covers the full Lk =
    // n_past + T columns vs T queries (allowing attention to all prior KV
    // entries). Caller fills the host-side buffers in run_lm_step.
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "lm_positions");

    ggml_tensor* causal_mask = nullptr;
    const int Lk = n_past + T;
    if (T > 1 || n_past > 0) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_input(causal_mask);
        ggml_set_name(causal_mask, "lm_causal_mask");
    }

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    ggml_tensor* cur = inputs_embeds;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.attn_norm_w);

        // KV-cached self-attn with Qwen2 Q/K/V biases (no o-bias). When
        // the GGUF stores fused `attn.qkv.{weight,bias}` (PLAN #60d),
        // pass them through; the per-projection Q/K/V tensors are nullptr
        // in that case and the helper takes the single-matmul path.
        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, h, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_o_w,
            /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask, ctx->kv_k, ctx->kv_v, (int)il,
            /*n_past*/ n_past, kvp,
            /*qkv_w*/ b.attn_qkv_w, /*fixed_kv_len*/ 0, /*kv_indices*/ nullptr, b.attn_q_b, b.attn_k_b, b.attn_v_b,
            /*o_b*/ nullptr, /*qkv_b*/ b.attn_qkv_b);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final norm (post-trunk).
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.final_norm_w);

    // Slice to last position only.
    ggml_tensor* last_hidden = cur;
    if (T > 1) {
        last_hidden = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    last_hidden = ggml_cont(ctx0, last_hidden);
    if (diag_captures) {
        ggml_set_name(last_hidden, "prefill_last_hidden");
        ggml_set_output(last_hidden);
        ggml_build_forward_expand(gf, last_hidden);
    }

    // lm_head on last position. Always kept — this is the consumed output.
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.llm.lm_head_w, last_hidden);
    ggml_set_name(logits, "prefill_text_logits_step0");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// 51b — Step decode graph (T=1, n_past>0). Skips the audio path entirely:
// during decode, every step's text row is the new text token (replicated
// gs times), the audio rows are speech_zeroemb_idx pads, the
// speech_active_mask is zero everywhere and the text_zero_mask is one,
// so the audio branch contributes a literal zero to the LM input. We
// elide it and feed embed_w[next] straight into the LM trunk.
//
// `fixed_kv_len`/`kv_indices` mirror the qwen3-tts O15 path:
//   fixed_kv_len > 0  → pin Lk so the topology is invariant across n_past
//   kv_indices != nullptr → scatter-write K/V via ggml_set_rows so the
//                            destination slot is a runtime input rather
//                            than baked into the graph as a byte offset.
// Pass them together to enable the cached-graph reuse path; pass both
// zero/null for the single-call form.
//
// Inputs (set externally before compute):
//   text_input_ids   [1] I32        — the new text token
//   lm_positions    [1] I32        — [n_past] (also reused as kv_indices)
//   lm_causal_mask  [Lk, 1] F16    — 0 for k <= n_past, -INF beyond
//
// Output (read by name):
//   step_logits     [vocab, 1] F32 — lm_head out
static ggml_cgraph* mimo_asr_build_step_graph(mimo_asr_context* ctx, int n_past, int fixed_kv_len) {
    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int n_q = (int)hp.llm_heads;
    const int n_kv = (int)hp.llm_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = 1;
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Embed lookup straight from llm.embed_w (no audio fusion).
    ggml_tensor* text_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(text_ids);
    ggml_set_name(text_ids, "text_input_ids");
    ggml_tensor* inputs_embeds = ggml_get_rows(ctx0, m.llm.embed_w, text_ids); // [d, 1]

    // Positions for RoPE; doubles as kv_indices when fixed_kv_len > 0.
    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(positions);
    ggml_set_name(positions, "lm_positions");

    // Causal mask is always declared at T=1 in the cached path so the
    // topology stays invariant (matches qwen3_tts.cpp O15 pattern).
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
    ggml_set_input(causal_mask);
    ggml_set_name(causal_mask, "lm_causal_mask");

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(n_past + T <= ctx->kv_max_ctx);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 0.0f,
        /*rope_beta_slow*/ 0.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    // When fixed_kv_len is set, hand `positions` to kv_self_attn as the
    // kv_indices tensor — it scatters K/V via ggml_set_rows keyed by the
    // runtime n_past, so the cached graph stays correct across steps.
    ggml_tensor* eff_kv_indices = (fixed_kv_len > 0) ? positions : nullptr;

    ggml_tensor* cur = inputs_embeds;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.attn_norm_w);

        ggml_tensor* attn = core_attn::kv_self_attn(
            ctx0, gf, h, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_o_w,
            /*q_norm_w*/ nullptr, /*k_norm_w*/ nullptr, positions, causal_mask, ctx->kv_k, ctx->kv_v, (int)il,
            /*n_past*/ n_past, kvp,
            /*qkv_w*/ b.attn_qkv_w, /*fixed_kv_len*/ fixed_kv_len, /*kv_indices*/ eff_kv_indices, b.attn_q_b,
            b.attn_k_b, b.attn_v_b, /*o_b*/ nullptr, /*qkv_b*/ b.attn_qkv_b);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, eps);
        h = ggml_mul(ctx0, h, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.final_norm_w);
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.llm.lm_head_w, cur); // [vocab, 1]
    ggml_set_name(logits, "step_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_free(ctx0);
    return gf;
}

// Helper: precompute the per-channel masks + text masks from raw input_ids
// in [9, T_total] format. Fills the supplied vectors.
struct PrefillInputs {
    int T_total = 0;
    int T_groups = 0;
    int group_size = 0;
    std::vector<int32_t> text_input_ids;              // [T_groups]
    std::vector<float> text_zero_mask;                // [d, T_groups], 1 where text != empty
    std::vector<float> speech_active_mask;            // [d, T_groups], 1 where text == empty
    std::array<std::vector<int32_t>, 8> speech_codes; // [T_total] per channel
    std::array<std::vector<float>, 8> combined_masks; // [audio_dim, T_total] per channel
};

static PrefillInputs make_prefill_inputs(const mimo_asr_hp& hp, const int32_t* input_ids_9xT, int T_total,
                                         uint32_t empty_token_id) {
    PrefillInputs out;
    const int gs = (int)hp.audio_group_size;
    const int d = (int)hp.llm_hidden;
    const int ad = (int)hp.audio_dim;
    out.T_total = T_total;
    out.group_size = gs;
    out.T_groups = T_total / gs;
    const int Tg = out.T_groups;

    out.text_input_ids.resize(Tg);
    out.text_zero_mask.assign((size_t)d * Tg, 0.0f);
    out.speech_active_mask.assign((size_t)d * Tg, 0.0f);

    for (int g = 0; g < Tg; g++) {
        // input_ids[0, g*gs] is the meaningful text token at this group.
        int32_t tok = input_ids_9xT[0 * T_total + g * gs];
        out.text_input_ids[g] = tok;
        bool is_speech = ((uint32_t)tok == empty_token_id);
        float t_mask = is_speech ? 0.0f : 1.0f;
        float s_mask = is_speech ? 1.0f : 0.0f;
        for (int j = 0; j < d; j++) {
            out.text_zero_mask[(size_t)g * d + j] = t_mask;
            out.speech_active_mask[(size_t)g * d + j] = s_mask;
        }
    }

    // Per-channel codes + masks.
    for (int c = 0; c < (int)hp.audio_channels; c++) {
        out.speech_codes[c].resize(T_total);
        out.combined_masks[c].assign((size_t)ad * T_total, 0.0f);
        const uint32_t zeroemb = hp.speech_zeroemb_idx[c];
        for (int t = 0; t < T_total; t++) {
            int32_t code = input_ids_9xT[(1 + c) * T_total + t];
            // The lookup table has `speech_vocab[c]` rows; the zeroemb_idx
            // is a valid in-range index (it's the padding index of the
            // embedding). To stay in-bounds we keep the code as-is, and
            // the per-element mask zeros out the contribution.
            out.speech_codes[c][t] = code;
            // is_real_audio = (code != zeroemb) AND (text at this group is empty)
            int g = t / gs;
            bool text_is_empty = ((uint32_t)out.text_input_ids[g] == empty_token_id);
            bool code_is_real = ((uint32_t)code != zeroemb);
            float m = (text_is_empty && code_is_real) ? 1.0f : 0.0f;
            for (int j = 0; j < ad; j++) {
                out.combined_masks[c][(size_t)t * ad + j] = m;
            }
        }
    }
    return out;
}

} // namespace

// ===========================================================================
// Stage extraction API — runs prefill on a caller-supplied input_ids tensor
// and returns a freshly-malloc'd float buffer with the named stage.
// ===========================================================================
extern "C" float* mimo_asr_extract_stage(struct mimo_asr_context* ctx, const int32_t* input_ids_9xT, int T_total,
                                         const char* stage, int* n_out) {
    if (!ctx || !input_ids_9xT || T_total <= 0 || !stage)
        return nullptr;
    const auto& hp = ctx->hp;
    if (T_total % (int)hp.audio_group_size != 0) {
        fprintf(stderr, "mimo_asr_extract_stage: T_total=%d must be a multiple of group_size=%u\n", T_total,
                hp.audio_group_size);
        return nullptr;
    }
    const int Tg = T_total / (int)hp.audio_group_size;
    if (!mimo_asr_kv_init(ctx, std::max(Tg + 256, 4096)))
        return nullptr;

    // Look up empty-token id from vocab. The on-disk Q4_K shipped with a
    // truncated vocab (151643 entries) where `<|empty|>` is missing — we
    // fall back to its known id (151667) so prefill works against the
    // stale GGUF. Once step 9 re-converts with the padded vocab, the
    // search hits and the fallback path is dead code.
    uint32_t empty_id = 151667;
    for (size_t i = 0; i < ctx->vocab.size(); i++) {
        if (ctx->vocab[i] == "<|empty|>") {
            empty_id = (uint32_t)i;
            break;
        }
    }

    PrefillInputs pi = make_prefill_inputs(hp, input_ids_9xT, T_total, empty_id);

    ggml_cgraph* gf = mimo_asr_build_prefill_graph(ctx, Tg, /*n_past*/ 0, /*diag_captures*/ true);
    // Building a fresh prefill graph in compute_meta invalidates any cached
    // step graph from a prior transcribe — drop the pointer so the next
    // decode loop rebuilds rather than reading clobbered metadata.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "mimo_asr_extract_stage: failed to alloc graph\n");
        return nullptr;
    }

    // Set inputs.
    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t) {
            fprintf(stderr, "mimo_asr_extract_stage: missing graph input '%s'\n", nm);
            return false;
        }
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    // Per-channel inputs.
    for (int c = 0; c < (int)hp.audio_channels; c++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "speech_codes_%d", c);
        if (!set_t(nm, pi.speech_codes[c].data(), pi.speech_codes[c].size() * sizeof(int32_t)))
            return nullptr;
        snprintf(nm, sizeof(nm), "combined_mask_%d", c);
        if (!set_t(nm, pi.combined_masks[c].data(), pi.combined_masks[c].size() * sizeof(float)))
            return nullptr;
    }

    // Speech active mask + text zero mask.
    if (!set_t("speech_active_mask", pi.speech_active_mask.data(), pi.speech_active_mask.size() * sizeof(float)))
        return nullptr;
    if (!set_t("text_zero_mask", pi.text_zero_mask.data(), pi.text_zero_mask.size() * sizeof(float)))
        return nullptr;
    if (!set_t("text_input_ids", pi.text_input_ids.data(), pi.text_input_ids.size() * sizeof(int32_t)))
        return nullptr;

    // ilt_positions = 0..gs-1 (shared by all 6 input_local_transformer layers).
    {
        std::vector<int32_t> p(hp.audio_group_size);
        for (uint32_t i = 0; i < hp.audio_group_size; i++)
            p[i] = (int32_t)i;
        if (!set_t("ilt_positions", p.data(), p.size() * sizeof(int32_t)))
            return nullptr;
    }

    // LM positions = 0..Tg-1.
    {
        std::vector<int32_t> p(Tg);
        for (int i = 0; i < Tg; i++)
            p[i] = i;
        if (!set_t("lm_positions", p.data(), p.size() * sizeof(int32_t)))
            return nullptr;
    }

    // Causal mask (only when Tg > 1).
    if (Tg > 1) {
        std::vector<ggml_fp16_t> mask((size_t)Tg * Tg);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < Tg; q++) {
            for (int k = 0; k < Tg; k++) {
                mask[(size_t)q * Tg + k] = (k <= q) ? z : ninf;
            }
        }
        if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
            return nullptr;
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "mimo_asr_extract_stage: graph compute failed\n");
        return nullptr;
    }
    ctx->kv_n_used = Tg;

    // Pull the requested stage tensor.
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, stage);
    if (!out_t) {
        fprintf(stderr, "mimo_asr_extract_stage: unknown stage '%s'\n", stage);
        return nullptr;
    }
    const size_t total = (size_t)ggml_nelements(out_t);
    float* result = (float*)malloc(total * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(out_t, result, 0, total * sizeof(float));
    if (n_out)
        *n_out = (int)total;
    return result;
}

extern "C" int mimo_asr_get_hparams(struct mimo_asr_context* ctx, uint32_t* llm_hidden, uint32_t* llm_vocab,
                                    uint32_t* audio_dim, uint32_t* audio_channels, uint32_t* audio_group_size) {
    if (!ctx)
        return -1;
    if (llm_hidden)
        *llm_hidden = ctx->hp.llm_hidden;
    if (llm_vocab)
        *llm_vocab = ctx->hp.llm_vocab;
    if (audio_dim)
        *audio_dim = ctx->hp.audio_dim;
    if (audio_channels)
        *audio_channels = ctx->hp.audio_channels;
    if (audio_group_size)
        *audio_group_size = ctx->hp.audio_group_size;
    return 0;
}

extern "C" int mimo_asr_set_tokenizer_path(struct mimo_asr_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->tokenizer_path = path;
    return 0;
}

// ===========================================================================
// PLAN #51 step 8 — full transcribe pipeline.
//
// Mirrors `MimoAudio.asr_sft` from ref/mimo/.../mimo_audio.py:
//   - audio tokenize via mimo_tokenizer (8-channel codes, 25 fps).
//   - assemble asr_sft prompt: text + audio + template + assistant header,
//     each as a [9, T_seg] block following process_speechdata.InputSegment.
//   - run prefill on the LM, greedy-argmax for next text token, then loop
//     on a [9, group_size] step graph with advancing n_past.
//   - stop on <|im_end|> / eos; skip <|empty|>/<|eot|>/<|eostm|> in output.
// ===========================================================================

// BPE encode a text fragment. Mirrors `qwen3_asr_tokenize`: split on
// recognised <|...|> special tokens (emit their ids directly), then
// whitespace-pre-split + bytes_to_unicode + bpe_one for the rest. Falls
// back to the per-byte path when merge_rank is empty (stale GGUF).
static std::vector<int32_t> mimo_asr_tokenize_text(const mimo_asr_context* ctx, const std::string& text) {
    std::vector<int32_t> result;
    const auto& t2i = ctx->token_to_id;
    const auto& mr = ctx->merge_rank;
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '<' && i + 1 < text.size() && text[i + 1] == '|') {
            size_t end = text.find("|>", i + 2);
            if (end != std::string::npos) {
                std::string special = text.substr(i, end + 2 - i);
                auto it = t2i.find(special);
                if (it != t2i.end()) {
                    result.push_back(it->second);
                    i = end + 2;
                    continue;
                }
            }
        }
        size_t j = i;
        if (text[j] == '<' && j + 1 < text.size() && text[j + 1] == '|')
            j++;
        while (j < text.size()) {
            if (text[j] == '<' && j + 1 < text.size() && text[j + 1] == '|') {
                size_t end = text.find("|>", j + 2);
                if (end != std::string::npos && t2i.find(text.substr(j, end + 2 - j)) != t2i.end())
                    break;
            }
            j++;
        }
        std::string chunk = text.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;

        // GPT-2-style pre-split: each pre-token is "[optional single ws]
        // + run of non-whitespace". Whitespace runs are emitted as their
        // own pre-tokens so newlines round-trip correctly.
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n')
                k++;
            if (k == start)
                k++;
            std::string pre = chunk.substr(start, k - start);
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(t2i, mr, encoded, result);
        }
    }
    return result;
}

// `insert_between` mirrors process_speechdata.InputSegment.insert_between:
// take a [1, L] tensor and produce a [1, L*gs] tensor where original
// values land at stride gs and `fill` (typically -100) fills the rest.
static std::vector<int32_t> mimo_asr_insert_between(const std::vector<int32_t>& src, int gs, int32_t fill) {
    std::vector<int32_t> out((size_t)src.size() * gs, fill);
    for (size_t i = 0; i < src.size(); i++)
        out[i * gs] = src[i];
    return out;
}

// Build a [9, T_seg] block for a TEXT segment (no audio). T_seg = len(tokens)*gs.
// Row 0 = tokens at stride gs, -100 fillers elsewhere; rows 1..8 = the
// per-channel speech_zeroemb_idx for every position. Matches the
// no-audio branch of InputSegment.to_input_id.
static std::vector<int32_t> mimo_asr_build_text_segment(const mimo_asr_context* ctx,
                                                        const std::vector<int32_t>& tokens) {
    const auto& hp = ctx->hp;
    const int gs = (int)hp.audio_group_size;
    const int channels = (int)hp.audio_channels;
    const int T_seg = (int)tokens.size() * gs;
    std::vector<int32_t> block((size_t)(channels + 1) * T_seg);
    auto row0 = mimo_asr_insert_between(tokens, gs, /*fill*/ -100);
    std::memcpy(block.data(), row0.data(), (size_t)T_seg * sizeof(int32_t));
    for (int c = 0; c < channels; c++) {
        int32_t pad = (int32_t)hp.speech_zeroemb_idx[c];
        for (int t = 0; t < T_seg; t++)
            block[(size_t)(1 + c) * T_seg + t] = pad;
    }
    return block;
}

// Build a [9, T_seg] block for an AUDIO segment. T_seg = (n_frames + 2*gs).
// Layout matches the audio branch of InputSegment.to_input_id with
// add_sosp_eosp=True:
//   - row 0: insert_between([sosp, empty*n_groups, eosp], gs, -100)
//   - rows 1..8: [speech_zeroemb_pad(gs) | audio_codes(n_frames) | speech_zeroemb_pad(gs)]
// `codes` is laid out [n_frames * channels] row-major (time-first), as
// returned by mimo_tokenizer_encode_pcm16k.
static std::vector<int32_t> mimo_asr_build_audio_segment(const mimo_asr_context* ctx, const int32_t* codes,
                                                         int n_frames) {
    const auto& hp = ctx->hp;
    const int gs = (int)hp.audio_group_size;
    const int channels = (int)hp.audio_channels;
    const int n_groups = n_frames / gs; // must be exact
    const int T_seg = n_frames + 2 * gs;
    std::vector<int32_t> block((size_t)(channels + 1) * T_seg);

    // Row 0
    std::vector<int32_t> text_pre;
    text_pre.reserve((size_t)(n_groups + 2));
    text_pre.push_back(ctx->id_sosp);
    for (int g = 0; g < n_groups; g++)
        text_pre.push_back(ctx->id_empty);
    text_pre.push_back(ctx->id_eosp);
    auto row0 = mimo_asr_insert_between(text_pre, gs, /*fill*/ -100);
    std::memcpy(block.data(), row0.data(), (size_t)T_seg * sizeof(int32_t));

    // Rows 1..8
    for (int c = 0; c < channels; c++) {
        int32_t pad = (int32_t)hp.speech_zeroemb_idx[c];
        int32_t* dst = block.data() + (size_t)(1 + c) * T_seg;
        for (int t = 0; t < gs; t++)
            dst[t] = pad;
        for (int t = 0; t < n_frames; t++)
            dst[gs + t] = codes[(size_t)t * channels + c];
        for (int t = 0; t < gs; t++)
            dst[gs + n_frames + t] = pad;
    }
    return block;
}

// Concatenate a list of [9, T_i] blocks into a single [9, sum(T_i)] tensor.
static std::vector<int32_t> mimo_asr_concat_segments(int channels_plus_text,
                                                     const std::vector<std::vector<int32_t>>& segments,
                                                     std::vector<int>& seg_lens_out) {
    int T_total = 0;
    seg_lens_out.clear();
    seg_lens_out.reserve(segments.size());
    for (const auto& s : segments) {
        int t = (int)(s.size() / channels_plus_text);
        seg_lens_out.push_back(t);
        T_total += t;
    }
    std::vector<int32_t> out((size_t)channels_plus_text * T_total);
    int off = 0;
    for (const auto& s : segments) {
        int t = (int)(s.size() / channels_plus_text);
        for (int r = 0; r < channels_plus_text; r++) {
            std::memcpy(out.data() + (size_t)r * T_total + off, s.data() + (size_t)r * t, (size_t)t * sizeof(int32_t));
        }
        off += t;
    }
    return out;
}

// 51b/51b' — Step decode runner. Builds the lightweight T=1 step graph
// (no audio path) on first call, then reuses the cached plan via
// skip_plan on every subsequent step within the same transcribe call.
// `next_token` is the most recent argmax; `n_past_groups` is the number
// of LM groups already in the KV cache.
//
// Cached-graph contract (matches qwen3-tts O15):
//   • Lk pinned to ctx->kv_max_ctx so the graph topology is invariant
//     across n_past — any change in n_past is a tensor *value* change,
//     not a graph-topology change.
//   • kv_indices is the same `lm_positions` tensor so the K/V scatter
//     destination is a runtime input rather than a static byte offset.
//   • Causal mask is shaped (Lk, 1) and filled so columns
//     [0..n_past] are zero (visible) and the rest are -INF — masking
//     the never-written tail slots so they can't leak NaN/garbage.
//   • The KV buffer is zero-cleared at mimo_asr_kv_init (already in
//     place at line ~459) — required to avoid CUDA / partial-CPU
//     noise from uninit'd KV pages (cf. qwen3-tts commit 7298dd5).
static float* mimo_asr_run_lm_step(mimo_asr_context* ctx, int32_t next_token, int n_past_groups) {
    const auto& hp = ctx->hp;
    const int vocab = (int)hp.llm_vocab;
    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    if (n_past_groups + 1 > ctx->kv_max_ctx) {
        fprintf(stderr, "mimo_asr_run_lm_step: kv overflow (%d+1 > %d)\n", n_past_groups, ctx->kv_max_ctx);
        return nullptr;
    }

    const int fixed_kv = ctx->kv_max_ctx;
    const bool can_skip = (ctx->step_t1_gf != nullptr && ctx->step_t1_fixed_kv_len == fixed_kv);

    ggml_cgraph* gf;
    if (can_skip) {
        gf = ctx->step_t1_gf;
    } else {
        // n_past=0 here is intentional — only kv_indices (= lm_positions)
        // controls the cache write slot when fixed_kv_len > 0, and Lk is
        // already pinned to fixed_kv. The `n_past` parameter is unused
        // by core_attn::kv_self_attn in that mode (the static-offset
        // write path is suppressed by kv_indices != nullptr).
        gf = mimo_asr_build_step_graph(ctx, /*n_past*/ 0, /*fixed_kv_len*/ fixed_kv);
        if (!gf)
            return nullptr;
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "mimo_asr_run_lm_step: alloc_graph failed\n");
            return nullptr;
        }
        ctx->step_t1_gf = gf;
        ctx->step_t1_fixed_kv_len = fixed_kv;
    }

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };

    int32_t tok = next_token;
    int32_t pos = n_past_groups;
    if (!set_t("text_input_ids", &tok, sizeof(tok)))
        return nullptr;
    if (!set_t("lm_positions", &pos, sizeof(pos)))
        return nullptr;

    // Causal mask shape (fixed_kv, 1): visible up to and including
    // position n_past_groups, -INF beyond.
    std::vector<ggml_fp16_t> mask((size_t)fixed_kv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int k = 0; k < fixed_kv; k++)
        mask[k] = (k <= n_past_groups) ? z : ninf;
    if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
        return nullptr;

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "mimo_asr_run_lm_step: compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "step_logits");
    if (!logits_t)
        return nullptr;
    float* out = (float*)malloc((size_t)vocab * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(logits_t, out, 0, (size_t)vocab * sizeof(float));
    return out;
}

// Run the prefill/step graph and return malloc'd logits for the *last*
// position. T_total must be a multiple of group_size. n_past is the
// number of LM groups already in the KV cache. Returns nullptr on failure.
static float* mimo_asr_run_lm(mimo_asr_context* ctx, const int32_t* input_ids_9xT, int T_total, int n_past) {
    const auto& hp = ctx->hp;
    const int gs = (int)hp.audio_group_size;
    if (T_total % gs != 0)
        return nullptr;
    const int Tg = T_total / gs;

    PrefillInputs pi = make_prefill_inputs(hp, input_ids_9xT, T_total, (uint32_t)ctx->id_empty);

    // Production path: skip diag captures (~5% win + cleaner allocator,
    // PLAN #51 perf wave). Honour MIMO_ASR_DIAG=1 to keep the diag tensors
    // resident when debugging a transcribe-time regression directly.
    const bool diag_env = std::getenv("MIMO_ASR_DIAG") != nullptr;
    ggml_cgraph* gf = mimo_asr_build_prefill_graph(ctx, Tg, n_past, /*diag_captures*/ diag_env);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    auto set_t = [&](const char* nm, const void* data, size_t bytes) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
        if (!t)
            return false;
        ggml_backend_tensor_set(t, data, 0, bytes);
        return true;
    };
    for (int c = 0; c < (int)hp.audio_channels; c++) {
        char nm[32];
        snprintf(nm, sizeof(nm), "speech_codes_%d", c);
        if (!set_t(nm, pi.speech_codes[c].data(), pi.speech_codes[c].size() * sizeof(int32_t)))
            return nullptr;
        snprintf(nm, sizeof(nm), "combined_mask_%d", c);
        if (!set_t(nm, pi.combined_masks[c].data(), pi.combined_masks[c].size() * sizeof(float)))
            return nullptr;
    }
    if (!set_t("speech_active_mask", pi.speech_active_mask.data(), pi.speech_active_mask.size() * sizeof(float)))
        return nullptr;
    if (!set_t("text_zero_mask", pi.text_zero_mask.data(), pi.text_zero_mask.size() * sizeof(float)))
        return nullptr;
    if (!set_t("text_input_ids", pi.text_input_ids.data(), pi.text_input_ids.size() * sizeof(int32_t)))
        return nullptr;
    {
        std::vector<int32_t> p((size_t)gs);
        for (int i = 0; i < gs; i++)
            p[i] = i;
        if (!set_t("ilt_positions", p.data(), p.size() * sizeof(int32_t)))
            return nullptr;
    }
    {
        std::vector<int32_t> p((size_t)Tg);
        for (int i = 0; i < Tg; i++)
            p[i] = n_past + i;
        if (!set_t("lm_positions", p.data(), p.size() * sizeof(int32_t)))
            return nullptr;
    }
    if (Tg > 1 || n_past > 0) {
        const int Lk = n_past + Tg;
        std::vector<ggml_fp16_t> mask((size_t)Tg * Lk);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < Tg; q++)
            for (int k = 0; k < Lk; k++)
                mask[(size_t)q * Lk + k] = (k <= n_past + q) ? z : ninf;
        if (!set_t("lm_causal_mask", mask.data(), mask.size() * sizeof(ggml_fp16_t)))
            return nullptr;
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "prefill_text_logits_step0");
    if (!logits_t)
        return nullptr;
    const size_t nlog = (size_t)ggml_nelements(logits_t);
    float* out = (float*)malloc(nlog * sizeof(float));
    if (!out)
        return nullptr;
    ggml_backend_tensor_get(logits_t, out, 0, nlog * sizeof(float));
    return out;
}

// Internal: shared implementation for `mimo_asr_transcribe` and
// `mimo_asr_transcribe_with_probs`. When `out_token_ids` and
// `out_token_probs` are non-null, both are populated in lock-step with
// the emitted tokens (one entry per greedy step, including the EOS-trimmed
// trailing token). Returns a malloc'd char* matching the visible transcript.
static char* mimo_asr_transcribe_impl(struct mimo_asr_context* ctx, const float* pcm, int n_samples,
                                      std::vector<int32_t>* out_token_ids, std::vector<float>* out_token_probs) {
    if (!ctx || !pcm || n_samples <= 0)
        return nullptr;

    // 1. Lazy-init the audio tokenizer.
    if (!ctx->tokenizer) {
        if (ctx->tokenizer_path.empty()) {
            fprintf(stderr, "mimo_asr_transcribe: tokenizer path not set; call mimo_asr_set_tokenizer_path() first\n");
            return nullptr;
        }
        auto tp = mimo_tokenizer_context_default_params();
        tp.n_threads = ctx->n_threads;
        tp.use_gpu = ctx->params.use_gpu;
        tp.verbosity = ctx->params.verbosity;
        ctx->tokenizer = mimo_tokenizer_init_from_file(ctx->tokenizer_path.c_str(), tp);
        if (!ctx->tokenizer) {
            fprintf(stderr, "mimo_asr_transcribe: failed to load tokenizer at '%s'\n", ctx->tokenizer_path.c_str());
            return nullptr;
        }
    }

    // 2. Encode PCM to 8-channel codes.
    int n_frames = 0;
    int32_t* codes = mimo_tokenizer_encode_pcm16k(ctx->tokenizer, pcm, n_samples, &n_frames);
    if (!codes || n_frames <= 0) {
        fprintf(stderr, "mimo_asr_transcribe: audio tokenization failed (n_frames=%d)\n", n_frames);
        free(codes);
        return nullptr;
    }
    const int gs = (int)ctx->hp.audio_group_size;
    const int channels = (int)ctx->hp.audio_channels;
    if (n_frames % gs != 0) {
        // Trim to a multiple of group_size — upstream pads, but trimming
        // a few frames at the end is safe for ASR (drops <0.16s).
        n_frames -= n_frames % gs;
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "mimo_asr_transcribe: audio %d samples -> %d code frames\n", n_samples, n_frames);

    // 3. Build the asr_sft prompt segments. Pinned to template[0] for
    // determinism (matches the diff harness reference dump).
    std::vector<std::vector<int32_t>> segments;
    auto add_text = [&](const std::string& s) {
        auto toks = mimo_asr_tokenize_text(ctx, s);
        segments.push_back(mimo_asr_build_text_segment(ctx, toks));
    };
    add_text("<|im_start|>user\n");
    segments.push_back(mimo_asr_build_audio_segment(ctx, codes, n_frames));
    free(codes);
    add_text("Please transcribe this audio file"); // asr_en_templates[0]
    add_text("<|im_end|>\n");
    add_text("<|im_start|>assistant\n");
    add_text("<think>\n\n</think>\n<english>");

    std::vector<int> seg_lens;
    auto input_ids = mimo_asr_concat_segments(channels + 1, segments, seg_lens);
    const int T_total = (int)(input_ids.size() / (channels + 1));
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "mimo_asr_transcribe: prompt T_total=%d (T_groups=%d)\n", T_total, T_total / gs);

    // 4. KV cache budget: prompt groups + max_new_tokens (one new group
    // per generated text token).
    const int max_new = 256;
    if (!mimo_asr_kv_init(ctx, T_total / gs + max_new + 16))
        return nullptr;

    // The cached step graph from a prior transcribe is invalid: kv_max_ctx
    // may have changed, and even when it didn't, mimo_asr_run_lm below
    // rebuilds the prefill graph in the shared compute_meta and clobbers
    // the old step graph's metadata. Drop the pointer so the first decode
    // step rebuilds.
    ctx->step_t1_gf = nullptr;
    ctx->step_t1_fixed_kv_len = 0;

    // 5. Prefill.
    const bool bench = std::getenv("MIMO_ASR_BENCH") != nullptr;
    auto now_ms = []() {
        using namespace std::chrono;
        return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
    };
    const double t_prefill0 = bench ? now_ms() : 0.0;
    float* logits = mimo_asr_run_lm(ctx, input_ids.data(), T_total, /*n_past*/ 0);
    if (!logits) {
        fprintf(stderr, "mimo_asr_transcribe: prefill failed\n");
        return nullptr;
    }
    const double t_prefill1 = bench ? now_ms() : 0.0;
    int vocab = (int)ctx->hp.llm_vocab;
    // Returns the picked id; when `out_p` is non-null also writes the
    // numerically stable softmax probability of that pick.
    auto pick = [&](const float* L, float* out_p) {
        int best = 0;
        float bv = L[0];
        for (int i = 1; i < vocab; i++)
            if (L[i] > bv) {
                bv = L[i];
                best = i;
            }
        if (out_p) {
            float s = 0.f;
            for (int i = 0; i < vocab; i++)
                s += expf(L[i] - bv);
            *out_p = (s > 0.f) ? (1.0f / s) : 0.0f;
        }
        return best;
    };

    int n_past_groups = T_total / gs;
    std::vector<int32_t> generated;
    std::vector<float> generated_probs;
    generated.reserve((size_t)max_new);
    // Only collect per-step softmax when the caller asked for it. For mimo's
    // 151k vocab the per-step softmax adds ~150k expf calls (~ms-scale) on
    // top of the LM forward, and the legacy argmax-only path must stay
    // bit-identical and fast.
    const bool capture_probs = (out_token_ids && out_token_probs);
    if (capture_probs)
        generated_probs.reserve((size_t)max_new);
    float prob = 0.0f;
    int next = pick(logits, capture_probs ? &prob : nullptr);
    free(logits);
    generated.push_back(next);
    if (capture_probs)
        generated_probs.push_back(prob);

    // 6. Greedy decode loop. Each step uses the lightweight T=1 step
    //    graph (51b/51b'): no audio path, fixed Lk for cached-graph
    //    reuse across steps. The full [9, gs] step would compute zero
    //    in the audio branch (speech_active_mask=0, all audio inputs
    //    zero-mask out) so we skip it entirely.
    const double t_decode0 = bench ? now_ms() : 0.0;
    int decode_steps = 0;
    for (int step = 1; step < max_new; step++) {
        if (next == ctx->id_im_end || next == ctx->id_eos)
            break;
        float* L = mimo_asr_run_lm_step(ctx, next, n_past_groups);
        if (!L)
            return nullptr;
        next = pick(L, capture_probs ? &prob : nullptr);
        free(L);
        n_past_groups++;
        generated.push_back(next);
        if (capture_probs)
            generated_probs.push_back(prob);
        decode_steps++;
    }
    const double t_decode1 = bench ? now_ms() : 0.0;
    if (bench) {
        const double prefill_ms = t_prefill1 - t_prefill0;
        const double decode_ms = t_decode1 - t_decode0;
        const double per_step_ms = decode_steps > 0 ? decode_ms / (double)decode_steps : 0.0;
        fprintf(stderr,
                "mimo_asr_bench: prefill=%.1f ms (Tg=%d)  decode=%.1f ms over %d steps "
                "(%.2f ms/step)  total_lm=%.1f ms\n",
                prefill_ms, T_total / gs, decode_ms, decode_steps, per_step_ms, prefill_ms + decode_ms);
    }

    // 7. Detokenize. The upstream drops the last token (typically
    //    <|im_end|>) and string-replaces <|empty|>/<|eot|>/<|eostm|>
    //    out of the visible transcript.
    if (!generated.empty() && (generated.back() == ctx->id_im_end || generated.back() == ctx->id_eos)) {
        generated.pop_back();
        if (capture_probs && !generated_probs.empty())
            generated_probs.pop_back();
    }
    if (capture_probs) {
        *out_token_ids = generated;
        *out_token_probs = generated_probs;
    }

    std::string detok = core_bpe::detokenize(ctx->vocab, generated.data(), generated.size());

    auto strip = [&](const char* needle) {
        std::string n = needle;
        size_t p;
        while ((p = detok.find(n)) != std::string::npos)
            detok.erase(p, n.size());
    };
    strip("<|empty|>");
    strip("<|eot|>");
    strip("<|eostm|>");
    strip("<english>");
    strip("<chinese>");

    // Trim leading whitespace
    size_t lead = 0;
    while (lead < detok.size() && (detok[lead] == ' ' || detok[lead] == '\t' || detok[lead] == '\n'))
        lead++;
    detok.erase(0, lead);

    char* out = (char*)malloc(detok.size() + 1);
    if (out)
        std::memcpy(out, detok.c_str(), detok.size() + 1);
    return out;
}

extern "C" void mimo_asr_free(struct mimo_asr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->tokenizer)
        mimo_tokenizer_free(ctx->tokenizer);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->buf_w_cpu)
        ggml_backend_buffer_free(ctx->buf_w_cpu);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" char* mimo_asr_transcribe(struct mimo_asr_context* ctx, const float* pcm, int n_samples) {
    return mimo_asr_transcribe_impl(ctx, pcm, n_samples, nullptr, nullptr);
}

extern "C" struct mimo_asr_result* mimo_asr_transcribe_with_probs(struct mimo_asr_context* ctx, const float* pcm,
                                                                  int n_samples) {
    std::vector<int32_t> ids;
    std::vector<float> probs;
    char* text = mimo_asr_transcribe_impl(ctx, pcm, n_samples, &ids, &probs);
    if (!text)
        return nullptr;
    auto* r = (mimo_asr_result*)calloc(1, sizeof(mimo_asr_result));
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

extern "C" void mimo_asr_result_free(struct mimo_asr_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->token_ids);
    free(r->token_probs);
    free(r);
}

extern "C" const char* mimo_asr_token_text(struct mimo_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.size())
        return "";
    return ctx->vocab[id].c_str();
}

extern "C" void mimo_asr_set_n_threads(struct mimo_asr_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}
