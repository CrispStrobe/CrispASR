// funasr.cpp — FunAudioLLM/Fun-ASR-Nano-2512 ggml runtime.
//
// See funasr.h for the public contract and architecture overview. The
// stage names exposed via funasr_extract_stage match the reference
// dumper at tools/reference_backends/funasr.py exactly.

#include "funasr.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/attention.h"
#include "core/beam_decode.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/kaldi_fbank.h"
#include "core/lfr.h"
#include "core/sanm.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation
//
// Enable per-stage timings with `FUNASR_BENCH=1`. Mirrors the granite /
// voxcpm2 pattern — RAII scope objects that log their elapsed time on
// destruction. Cost when disabled is one cached env-var read per stage
// (~1 ns); safe to leave compiled in.
//
// Reported stages (per crispasr_transcribe call):
//   fbank+lfr             frontend (kaldi-fbank + LFR stacking)
//   enc+ada_compute       SANM encoder + adaptor graph compute
//   prompt_tokenize       Qwen3 BPE encode for prefix/suffix
//   embed                 token_embd lookup + splice
//   kv_init               KV cache allocation (first call only)
//   llm_prefill           single prefill pass (T_prompt tokens)
//   llm_decode_total      AR decode loop wall time
//   decode_step_avg       average per-step time across all decoded tokens
// ===========================================================================

static bool funasr_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("FUNASR_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct funasr_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit funasr_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~funasr_bench_stage() {
        if (!funasr_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  funasr_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct funasr_hparams {
    // Frontend
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;
    uint32_t frame_length_ms = 25;
    uint32_t frame_shift_ms = 10;
    uint32_t lfr_m = 7;
    uint32_t lfr_n = 6;

    // Encoder (SenseVoiceEncoderSmall)
    uint32_t input_size = 560; // n_mels * lfr_m
    uint32_t d_model = 512;
    uint32_t n_heads = 4;
    uint32_t head_dim = 128;
    uint32_t ffn_dim = 2048;
    uint32_t n_blocks_base = 50;
    uint32_t n_blocks_tp = 20;
    uint32_t sanm_kernel = 11;
    float enc_ln_eps = 1e-5f;

    // Audio adaptor (Transformer with downsample_rate=1)
    uint32_t ada_d_in = 512;
    uint32_t ada_ffn = 2048;
    uint32_t ada_d_out = 1024; // = LLM hidden
    uint32_t ada_n_layers = 2;
    // Upstream funasr.models.llm_asr.adaptor.Transformer instantiates
    // MultiHeadedAttention with `kwargs.get("attention_heads", 8)` and the
    // config.yaml for Fun-ASR-Nano-2512 does NOT pass attention_heads, so
    // the adaptor's 2 transformer blocks run at 8 heads (head_dim = 128).
    // The converter wrote 16 into funasr.ada_n_heads by mistake; ignore
    // that KV here and use the upstream default.
    uint32_t ada_n_heads = 8;
    uint32_t ada_head_dim = 128;  // ada_d_out / ada_n_heads
    uint32_t ada_ffn_inner = 256; // llm_dim // 4 from funasr.models.llm_asr.adaptor.Transformer
    float ada_ln_eps = 1e-12f;    // funasr.models.transformer.layer_norm.LayerNorm default

    // LLM (Qwen3-0.6B)
    uint32_t llm_n_layers = 28;
    uint32_t llm_d_model = 1024;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 3072;
    float llm_rope_theta = 1.0e6f;
    float llm_rms_eps = 1e-6f;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos = 40960;

    // Special tokens — read from GGUF KVs (with Qwen3 defaults as backup).
    uint32_t eos_token_id = 151645;
    uint32_t pad_token_id = 151643;

    // ChatML / FunASR control tokens — Qwen3 added_tokens, looked up by string at load.
    uint32_t im_start_id = 151644;
    uint32_t im_end_id = 151645;

    // Adaptor token-budget reduction matches the model.use_low_frame_rate
    // training flag. Fun-ASR-Nano-2512 sets it to true (23/183 frames on
    // JFK); Fun-ASR-MLT-Nano-2512 omits it and the upstream default is
    // false (all 183 frames used). Read from the GGUF KV at load time;
    // fall back to true for GGUFs that predate the KV.
    bool use_low_frame_rate = true;
};

// ===========================================================================
// Per-block tensor containers
// ===========================================================================

struct funasr_enc_block {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;
    ggml_tensor *attn_out_w = nullptr, *attn_out_b = nullptr;
    ggml_tensor* attn_fsmn_w = nullptr;
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_l2_w = nullptr, *ffn_l2_b = nullptr;
};

struct funasr_adaptor_block {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *out_w = nullptr, *out_b = nullptr;
    ggml_tensor *ffn_l1_w = nullptr, *ffn_l1_b = nullptr;
    ggml_tensor *ffn_l2_w = nullptr, *ffn_l2_b = nullptr;
};

struct funasr_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_qkv_w = nullptr; // fused Q+K+V for single matmul (§136)
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct funasr_encoder {
    std::vector<funasr_enc_block> blocks; // 70 = base + tp
    ggml_tensor *after_norm_w = nullptr, *after_norm_b = nullptr;
    ggml_tensor *tp_norm_w = nullptr, *tp_norm_b = nullptr;
};

struct funasr_adaptor {
    ggml_tensor *linear1_w = nullptr, *linear1_b = nullptr;
    ggml_tensor *linear2_w = nullptr, *linear2_b = nullptr;
    std::vector<funasr_adaptor_block> blocks;
};

struct funasr_llm {
    ggml_tensor* token_embd_w = nullptr;
    ggml_tensor* output_w = nullptr;
    ggml_tensor* output_norm_w = nullptr;
    std::vector<funasr_llm_block> blocks;
};

struct funasr_model {
    funasr_hparams hparams;
    funasr_encoder enc;
    funasr_adaptor ada;
    funasr_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal PE for the encoder, computed once at load. depth = input_size = 560,
    // length = some max T. row-major (max_T, depth).
    std::vector<float> enc_pe;
    int enc_pe_max_T = 0;
};

struct funasr_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct funasr_context {
    funasr_context_params params;
    funasr_model model;
    funasr_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    // Fused QKV weights (§136). Created at init to merge the 3 separate
    // Q/K/V projections into one matmul so ggml_backend_sched can't split
    // them across backends (the root cause of the CUDA !-loop).
    ggml_context* fused_ctx = nullptr;
    ggml_backend_buffer_t fused_buf = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache for the LLM body — same layout as qwen3_asr.
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int beam_size = 1;

    // Cached per-step LLM decode graph (PLAN funasr-perf #1). Built once
    // at first decode call (via funasr_ensure_step_graph) using
    // `kv_indices` runtime input + `fixed_kv_len = kv_max_ctx`, so the
    // topology stays constant across calls regardless of n_past. Each
    // decode step just re-writes the four input tensors (inputs_embeds,
    // positions, kv_indices, causal_mask) and re-runs the graph — skipping
    // the per-step `ggml_init` + tensor pool + sched_alloc rebuild that
    // dominates non-cached decode at ~10 ms/tok on M1.
    //
    // Bypasses ggml_backend_sched (which re-plans per call and conflicts
    // with cached graphs holding views into pre-allocated KV buffers);
    // uses a dedicated `ggml_gallocr_t` reserved once, plus
    // `ggml_backend_graph_compute(ctx->backend, gf)` per step. Mirrors
    // the voxcpm2 TSLM step-bucket pattern (HISTORY 2026-05-19).
    std::vector<uint8_t> step_compute_meta;
    ggml_context* step_ctx0 = nullptr;
    ggml_cgraph* step_graph = nullptr;
    ggml_gallocr_t step_galloc = nullptr;

    int n_threads = 4;

    // Per-session knobs. Set at init from env vars / params; consumed by
    // the graph builders below.
    //   enc_flash_attn — fold the SANM/adaptor attention into one
    //     ggml_flash_attn_ext kernel (Metal/CUDA). Default ON; disable
    //     with FUNASR_NO_FA=1 for diffing against a pre-FA reference
    //     or to dodge a hypothetical backend bug.
    //   step_graph_cache — reuse the per-step LLM decode graph across
    //     calls. Default OFF on this workload — see comment in
    //     funasr_ensure_step_graph for the perf-regression analysis.
    //     Enable opportunistically with FUNASR_STEP_CACHE=1 for
    //     experimentation; the right long-term fix is bucketed Lk
    //     graphs (PLAN funasr-perf #1).
    bool enc_flash_attn = true;
    bool step_graph_cache = false;

    // Stage-capture state — set by funasr_extract_stage to request a
    // specific intermediate tensor; consumed by the encoder graph builder
    // (which wires a named ggml_dup snap at the requested stage so the
    // sched keeps it alive after compute).
    std::string requested_stage;
};

// ===========================================================================
// Loader
// ===========================================================================

static ggml_tensor* try_get(funasr_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(funasr_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "funasr");
}

static void compute_encoder_pe(funasr_model& m, int max_T) {
    // SinusoidalPositionEncoder.encode(positions=1..T, depth=input_size, dtype=f32)
    //   log_inc = log(10000) / (depth/2 - 1)
    //   inv_t   = exp(arange(depth/2) * (-log_inc))
    //   pe[t, :half]   = sin((t+1) * inv_t)            (positions start at 1)
    //   pe[t, half:]   = cos((t+1) * inv_t)
    const int D = (int)m.hparams.input_size;
    const int half = D / 2;
    const float log_inc = std::log(10000.0f) / (float)(half - 1);
    std::vector<float> inv_t((size_t)half);
    for (int i = 0; i < half; i++)
        inv_t[(size_t)i] = std::exp(-log_inc * (float)i);
    m.enc_pe.assign((size_t)max_T * (size_t)D, 0.0f);
    for (int t = 0; t < max_T; t++) {
        const float pos = (float)(t + 1); // positions = arange(1, T+1)
        float* row = m.enc_pe.data() + (size_t)t * (size_t)D;
        for (int i = 0; i < half; i++) {
            const float a = pos * inv_t[(size_t)i];
            row[i] = std::sin(a);
            row[half + i] = std::cos(a);
        }
    }
    m.enc_pe_max_T = max_T;
}

static bool funasr_load_model(funasr_model& model, funasr_vocab& vocab, const char* path, ggml_backend_t backend,
                              ggml_backend_t cpu_backend) {
    // Pass 1: hparams + vocab
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;
        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "funasr.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "funasr.n_mels", hp.n_mels);
        hp.frame_length_ms = core_gguf::kv_u32(gctx, "funasr.frame_length_ms", hp.frame_length_ms);
        hp.frame_shift_ms = core_gguf::kv_u32(gctx, "funasr.frame_shift_ms", hp.frame_shift_ms);
        hp.lfr_m = core_gguf::kv_u32(gctx, "funasr.lfr_m", hp.lfr_m);
        hp.lfr_n = core_gguf::kv_u32(gctx, "funasr.lfr_n", hp.lfr_n);
        hp.d_model = core_gguf::kv_u32(gctx, "funasr.d_model", hp.d_model);
        hp.n_heads = core_gguf::kv_u32(gctx, "funasr.n_heads", hp.n_heads);
        hp.ffn_dim = core_gguf::kv_u32(gctx, "funasr.ffn_dim", hp.ffn_dim);
        hp.n_blocks_base = core_gguf::kv_u32(gctx, "funasr.n_blocks_base", hp.n_blocks_base);
        hp.n_blocks_tp = core_gguf::kv_u32(gctx, "funasr.n_blocks_tp", hp.n_blocks_tp);
        hp.sanm_kernel = core_gguf::kv_u32(gctx, "funasr.sanm_kernel", hp.sanm_kernel);
        hp.input_size = hp.n_mels * hp.lfr_m;
        hp.head_dim = hp.d_model / hp.n_heads;

        hp.ada_d_in = core_gguf::kv_u32(gctx, "funasr.ada_d_in", hp.ada_d_in);
        hp.ada_ffn = core_gguf::kv_u32(gctx, "funasr.ada_ffn", hp.ada_ffn);
        hp.ada_d_out = core_gguf::kv_u32(gctx, "funasr.ada_d_out", hp.ada_d_out);
        hp.ada_n_layers = core_gguf::kv_u32(gctx, "funasr.ada_n_layers", hp.ada_n_layers);
        // The converter now writes the correct ada_n_heads=8; old GGUFs may
        // still carry 16 (the original converter bug). Ignore the KV and
        // keep the struct default of 8 until all published GGUFs are rebuilt.
        hp.ada_ffn_inner = core_gguf::kv_u32(gctx, "funasr.ada_ffn_inner", hp.ada_ffn_inner);
        hp.ada_head_dim = hp.ada_d_out / hp.ada_n_heads;
        hp.use_low_frame_rate = core_gguf::kv_bool(gctx, "funasr.use_low_frame_rate", hp.use_low_frame_rate);

        hp.llm_n_layers = core_gguf::kv_u32(gctx, "funasr.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(gctx, "funasr.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "funasr.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "funasr.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "funasr.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "funasr.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "funasr.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "funasr.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "funasr.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "funasr.llm.max_pos", hp.llm_max_pos);

        hp.eos_token_id = core_gguf::kv_u32(gctx, "funasr.eos_token_id", hp.eos_token_id);
        hp.pad_token_id = core_gguf::kv_u32(gctx, "funasr.pad_token_id", hp.pad_token_id);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[(size_t)i]] = i;
            }
        }
        // Qwen3 specials — written into the GGUF token list as readable
        // names by the converter, since they're in tokenizer.json's
        // added_tokens array. Walk the table once to find <|im_start|>
        // / <|im_end|> so we don't hardcode the IDs.
        auto find_id = [&](const char* s) {
            auto it = vocab.token_to_id.find(s);
            return it == vocab.token_to_id.end() ? -1 : (int)it->second;
        };
        const int ims = find_id("<|im_start|>");
        const int ime = find_id("<|im_end|>");
        if (ims >= 0)
            hp.im_start_id = (uint32_t)ims;
        if (ime >= 0)
            hp.im_end_id = (uint32_t)ime;

        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++)
            vocab.merge_rank[merges[(size_t)i]] = i;

        core_gguf::free_metadata(gctx);
    }

    // Pass 2: tensor data.
    // Issue #125 root cause: on GPUs without DP4A (P100 sm_60), quantized
    // MUL_MAT falls through to cuBLAS F16 GEMM whose F16 accumulator
    // overflows at the FFN down projection (swiglu values reach ~15K,
    // dot-product partial sums over 3072 elements exceed F16 max 65504).
    // Fixed in ggml-cuda.cu: block F16 cuBLAS path when src1 is F32.
    //
    // FUNASR_LLM_CPU=1 forces the old weight-split workaround (enc GPU,
    // LLM CPU) for testing or as a safety net.
    core_gguf::WeightLoad wl;
    const bool force_llm_cpu = []() {
        const char* s = std::getenv("FUNASR_LLM_CPU");
        return s && *s && *s != '0';
    }();
    if (!ggml_backend_is_cpu(backend) && cpu_backend && force_llm_cpu) {
        auto is_gpu = [](const char* name, void*) -> bool { return std::strncmp(name, "funasr.", 7) == 0; };
        if (!core_gguf::load_weights_split(path, backend, cpu_backend, is_gpu, nullptr, "funasr", wl))
            return false;
    } else {
        if (!core_gguf::load_weights(path, backend, "funasr", wl))
            return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    const auto& hp = model.hparams;
    const int n_enc = (int)(hp.n_blocks_base + hp.n_blocks_tp);
    model.enc.blocks.resize((size_t)n_enc);
    for (int i = 0; i < n_enc; i++) {
        char buf[128];
        auto& b = model.enc.blocks[(size_t)i];
        auto get = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "funasr.enc.blk.%d.%s", i, suf);
            return require(model, buf);
        };
        b.norm1_w = get("norm1.w");
        b.norm1_b = get("norm1.b");
        b.norm2_w = get("norm2.w");
        b.norm2_b = get("norm2.b");
        b.attn_qkv_w = get("attn.qkv.w");
        b.attn_qkv_b = get("attn.qkv.b");
        b.attn_out_w = get("attn.out.w");
        b.attn_out_b = get("attn.out.b");
        b.attn_fsmn_w = get("attn.fsmn.w");
        b.ffn_l1_w = get("ffn.l1.w");
        b.ffn_l1_b = get("ffn.l1.b");
        b.ffn_l2_w = get("ffn.l2.w");
        b.ffn_l2_b = get("ffn.l2.b");
    }
    model.enc.after_norm_w = require(model, "funasr.enc.after_norm.w");
    model.enc.after_norm_b = require(model, "funasr.enc.after_norm.b");
    model.enc.tp_norm_w = require(model, "funasr.enc.tp_norm.w");
    model.enc.tp_norm_b = require(model, "funasr.enc.tp_norm.b");

    // Adaptor
    model.ada.linear1_w = require(model, "funasr.adaptor.linear1.w");
    model.ada.linear1_b = require(model, "funasr.adaptor.linear1.b");
    model.ada.linear2_w = require(model, "funasr.adaptor.linear2.w");
    model.ada.linear2_b = require(model, "funasr.adaptor.linear2.b");
    model.ada.blocks.resize((size_t)hp.ada_n_layers);
    for (uint32_t i = 0; i < hp.ada_n_layers; i++) {
        char buf[128];
        auto& b = model.ada.blocks[(size_t)i];
        auto get = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "funasr.adaptor.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.norm1_w = get("norm1.w");
        b.norm1_b = get("norm1.b");
        b.norm2_w = get("norm2.w");
        b.norm2_b = get("norm2.b");
        b.q_w = get("attn.q.w");
        b.q_b = get("attn.q.b");
        b.k_w = get("attn.k.w");
        b.k_b = get("attn.k.b");
        b.v_w = get("attn.v.w");
        b.v_b = get("attn.v.b");
        b.out_w = get("attn.out.w");
        b.out_b = get("attn.out.b");
        b.ffn_l1_w = get("ffn.l1.w");
        b.ffn_l1_b = get("ffn.l1.b");
        b.ffn_l2_w = get("ffn.l2.w");
        b.ffn_l2_b = get("ffn.l2.b");
    }

    // LLM (Qwen3-0.6B, llama.cpp tensor naming)
    model.llm.token_embd_w = require(model, "token_embd.weight");
    model.llm.output_w = require(model, "output.weight");
    model.llm.output_norm_w = require(model, "output_norm.weight");
    model.llm.blocks.resize((size_t)hp.llm_n_layers);
    for (uint32_t i = 0; i < hp.llm_n_layers; i++) {
        char buf[128];
        auto& b = model.llm.blocks[(size_t)i];
        auto get = [&](const char* suf) {
            std::snprintf(buf, sizeof(buf), "blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn_q.weight");
        b.attn_k_w = get("attn_k.weight");
        b.attn_v_w = get("attn_v.weight");
        b.attn_output_w = get("attn_output.weight");
        b.attn_q_norm_w = get("attn_q_norm.weight");
        b.attn_k_norm_w = get("attn_k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn_gate.weight");
        b.ffn_up_w = get("ffn_up.weight");
        b.ffn_down_w = get("ffn_down.weight");
    }

    // Precompute encoder PE for up to 8192 LFR frames (~ 8 minutes).
    compute_encoder_pe(model, 8192);
    return true;
}

// ===========================================================================
// Frontend — kaldi-fbank + LFR. Matches funasr.frontends.wav_frontend.WavFrontend
// invoked with window="hamming", upsacle_samples=True, dither=0.0, snip_edges=True.
// ===========================================================================

static std::vector<float> funasr_compute_features(funasr_context* ctx, const float* pcm, int n_samples, int& T_lfr_out,
                                                  int& D_lfr_out) {
    const auto& hp = ctx->model.hparams;
    core_kaldi::FbankParams fp;
    fp.sample_rate = (int)hp.sample_rate;
    fp.n_mels = (int)hp.n_mels;
    fp.frame_length_ms = (int)hp.frame_length_ms;
    fp.frame_shift_ms = (int)hp.frame_shift_ms;
    fp.int16_scale = true;
    fp.window_type = core_kaldi::WindowType::Hamming;

    int T = 0;
    std::vector<float> mel = core_kaldi::compute_fbank(pcm, n_samples, fp, T);
    if (T == 0)
        return {};

    int T_lfr = 0;
    std::vector<float> lfr = core_lfr::stack(mel.data(), T, (int)hp.n_mels, (int)hp.lfr_m, (int)hp.lfr_n, T_lfr);
    T_lfr_out = T_lfr;
    D_lfr_out = (int)hp.input_size;
    return lfr;
}

// ===========================================================================
// Encoder + adaptor compute graph
//
// Built as a single graph from (T_lfr, 560) features → (T_lfr, 1024) adaptor
// output. The stage-capture mechanism wires a named ggml_dup snap inside
// the loop so the diff harness can pull intermediate activations back out.
// ===========================================================================

static ggml_tensor* maybe_snap(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* t, const std::string& want,
                               const char* name) {
    // Snap = a named contiguous copy that the scheduler can't fold away.
    // Always emit (cheap on Metal/CPU), so the diff harness gets the same
    // graph as production.
    ggml_tensor* s = ggml_dup(ctx0, t);
    ggml_set_name(s, name);
    ggml_build_forward_expand(gf, s);
    (void)want;
    return t;
}

static ggml_cgraph* funasr_build_graph_features(funasr_context* ctx, int T_lfr) {
    const auto& hp = ctx->model.hparams;
    const int D_in = (int)hp.input_size;
    const int D = (int)hp.d_model;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // ---- inputs ----
    // mel_in: (D_in, T_lfr) F32 — written by the caller from the LFR output.
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_in, T_lfr);
    ggml_set_name(mel_in, "mel_features");
    ggml_set_input(mel_in);

    // pe_in: (D_in, T_lfr) F32 — written from precomputed model.enc_pe.
    ggml_tensor* pe_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D_in, T_lfr);
    ggml_set_name(pe_in, "enc_pe");
    ggml_set_input(pe_in);

    // Mel + sqrt(d_model) scale + PE.
    ggml_tensor* cur = ggml_scale(ctx0, mel_in, std::sqrt((float)D));
    cur = ggml_add(ctx0, cur, pe_in);

    // ---- Encoder blocks 0..69 ----
    const int n_base = (int)hp.n_blocks_base;
    const int n_tp = (int)hp.n_blocks_tp;
    const int K = (int)hp.sanm_kernel;
    const int n_heads = (int)hp.n_heads;
    const int hd = (int)hp.head_dim;

    for (int i = 0; i < n_base + n_tp; i++) {
        const auto& src = ctx->model.enc.blocks[(size_t)i];
        core_sanm::BlockWeights w;
        w.norm1_w = src.norm1_w;
        w.norm1_b = src.norm1_b;
        w.norm2_w = src.norm2_w;
        w.norm2_b = src.norm2_b;
        w.attn_qkv_w = src.attn_qkv_w;
        w.attn_qkv_b = src.attn_qkv_b;
        w.attn_out_w = src.attn_out_w;
        w.attn_out_b = src.attn_out_b;
        w.attn_fsmn_w = src.attn_fsmn_w;
        w.ffn_l1_w = src.ffn_l1_w;
        w.ffn_l1_b = src.ffn_l1_b;
        w.ffn_l2_w = src.ffn_l2_w;
        w.ffn_l2_b = src.ffn_l2_b;

        core_sanm::BlockParams p;
        // Block 0 has in_size=560, all others in_size=512.
        p.in_size = (i == 0) ? D_in : D;
        p.size = D;
        p.n_heads = n_heads;
        p.head_dim = hd;
        p.kernel = K;
        p.ln_eps = hp.enc_ln_eps;
        p.flash_attn = ctx->enc_flash_attn;

        const bool attn_residual = (p.in_size == p.size);
        cur = core_sanm::build_block(ctx0, cur, T_lfr, w, p, attn_residual);

        // Stage snap per layer (named "encoder_layer_K").
        char nm[32];
        std::snprintf(nm, sizeof(nm), "encoder_layer_%d", i);
        cur = maybe_snap(ctx0, gf, cur, ctx->requested_stage, nm);

        // After the last base block, apply after_norm — matches
        // SenseVoiceEncoderSmall.forward.
        if (i == n_base - 1) {
            cur = ggml_norm_affine(ctx0, cur, ctx->model.enc.after_norm_w, ctx->model.enc.after_norm_b, hp.enc_ln_eps);
            cur = maybe_snap(ctx0, gf, cur, ctx->requested_stage, "encoder_main_out");
        }
    }
    // Final tp_norm.
    cur = ggml_norm_affine(ctx0, cur, ctx->model.enc.tp_norm_w, ctx->model.enc.tp_norm_b, hp.enc_ln_eps);
    cur = maybe_snap(ctx0, gf, cur, ctx->requested_stage, "encoder_output");

    // ---- audio_adaptor prelude: linear1 + ReLU + linear2 ----
    const int D_ada = (int)hp.ada_d_out;
    auto mm_bias = [&](ggml_tensor* W, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, W, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };
    cur = mm_bias(ctx->model.ada.linear1_w, cur, ctx->model.ada.linear1_b);
    cur = ggml_relu(ctx0, cur);
    cur = mm_bias(ctx->model.ada.linear2_w, cur, ctx->model.ada.linear2_b);
    // cur is now (D_ada=1024, T_lfr)

    // ---- 2 Transformer blocks (separate Q/K/V, FFN inner=256, LN eps=1e-12) ----
    for (uint32_t li = 0; li < hp.ada_n_layers; li++) {
        const auto& b = ctx->model.ada.blocks[(size_t)li];
        const int ah = (int)hp.ada_n_heads;
        const int ahd = (int)hp.ada_head_dim;
        const float ascale = 1.0f / std::sqrt((float)ahd);

        // pre-norm self-attention
        ggml_tensor* residual = cur;
        ggml_tensor* x = ggml_norm_affine(ctx0, cur, b.norm1_w, b.norm1_b, hp.ada_ln_eps);

        ggml_tensor* Q = mm_bias(b.q_w, x, b.q_b);
        ggml_tensor* K_ = mm_bias(b.k_w, x, b.k_b);
        ggml_tensor* V = mm_bias(b.v_w, x, b.v_b);

        Q = ggml_reshape_3d(ctx0, Q, ahd, ah, T_lfr);
        K_ = ggml_reshape_3d(ctx0, K_, ahd, ah, T_lfr);
        V = ggml_reshape_3d(ctx0, V, ahd, ah, T_lfr);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K_ = ggml_cont(ctx0, ggml_permute(ctx0, K_, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* attn;
        if (ctx->enc_flash_attn) {
            // Same fused kernel as the SANM block — bidirectional encoder
            // attention, no mask, scale = 1/sqrt(head_dim).
            attn = ggml_flash_attn_ext(ctx0, Q, K_, V, /*mask*/ nullptr, ascale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(ctx0, attn, D_ada, T_lfr);
        } else {
            ggml_tensor* scores = ggml_mul_mat(ctx0, K_, Q);
            scores = ggml_soft_max_ext(ctx0, scores, nullptr, ascale, 0.0f);
            ggml_tensor* V_p = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
            attn = ggml_mul_mat(ctx0, V_p, scores);
            attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(ctx0, attn, D_ada, T_lfr);
        }
        attn = mm_bias(b.out_w, attn, b.out_b);
        cur = ggml_add(ctx0, residual, attn);

        // pre-norm FFN
        ggml_tensor* res2 = cur;
        x = ggml_norm_affine(ctx0, cur, b.norm2_w, b.norm2_b, hp.ada_ln_eps);
        x = mm_bias(b.ffn_l1_w, x, b.ffn_l1_b);
        x = ggml_relu(ctx0, x);
        x = mm_bias(b.ffn_l2_w, x, b.ffn_l2_b);
        cur = ggml_add(ctx0, res2, x);

        char nm[32];
        std::snprintf(nm, sizeof(nm), "audio_adaptor_layer_%u", li);
        cur = maybe_snap(ctx0, gf, cur, ctx->requested_stage, nm);
    }

    ggml_set_name(cur, "audio_adaptor_output");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the encoder+adaptor graph and return (T_lfr, ada_d_out) F32 row-major.
// Stage snaps are pulled out by funasr_extract_stage from the same graph.
static std::vector<float> funasr_run_encoder_adaptor(funasr_context* ctx, const std::vector<float>& lfr, int T_lfr,
                                                     int D_lfr, std::vector<float>* stage_out, const char* stage_name) {
    const auto& hp = ctx->model.hparams;
    const int D_ada = (int)hp.ada_d_out;
    if (T_lfr <= 0)
        return {};
    if (T_lfr > ctx->model.enc_pe_max_T) {
        // Lazy-extend PE if a very long clip was supplied.
        compute_encoder_pe(ctx->model, T_lfr + 256);
    }

    ggml_cgraph* gf = funasr_build_graph_features(ctx, T_lfr);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        std::fprintf(stderr, "funasr: failed to alloc encoder graph\n");
        return {};
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_features");
    ggml_backend_tensor_set(mel_in, lfr.data(), 0, (size_t)D_lfr * (size_t)T_lfr * sizeof(float));
    ggml_tensor* pe_in = ggml_graph_get_tensor(gf, "enc_pe");
    ggml_backend_tensor_set(pe_in, ctx->model.enc_pe.data(), 0, (size_t)D_lfr * (size_t)T_lfr * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "funasr: encoder graph compute failed\n");
        return {};
    }

    // ---- Per-stage tensor dump (FUNASR_DUMP_STAGES=1) ----
    // Prints min/max/mean/L2/first-8 for key encoder stages so CPU-vs-CUDA
    // divergence can be localised from Kaggle logs. Default OFF.
    {
        static int dump_flag = -1;
        if (dump_flag < 0) {
            const char* e = std::getenv("FUNASR_DUMP_STAGES");
            dump_flag = (e && *e && *e != '0') ? 1 : 0;
        }
        if (dump_flag) {
            // Stages to dump — every 10th encoder layer + boundaries + adaptor.
            const char* names[] = {"encoder_layer_0",     "encoder_layer_9",       "encoder_layer_19",
                                   "encoder_layer_29",    "encoder_layer_39",      "encoder_layer_49",
                                   "encoder_layer_59",    "encoder_layer_69",      "encoder_main_out",
                                   "encoder_output",      "audio_adaptor_layer_0", "audio_adaptor_layer_1",
                                   "audio_adaptor_output"};
            std::fprintf(stderr, "funasr_dump: ---- encoder/adaptor stage stats (T_lfr=%d) ----\n", T_lfr);
            for (const char* nm : names) {
                ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
                if (!t) {
                    std::fprintf(stderr, "funasr_dump: %-28s [not found]\n", nm);
                    continue;
                }
                const size_t n = ggml_nelements(t);
                std::vector<float> buf(n, 0.0f);
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
                float mn = buf[0], mx = buf[0], sm = 0.0f, sq = 0.0f;
                int n_nan = 0, n_inf = 0;
                for (size_t i = 0; i < n; i++) {
                    float v = buf[i];
                    if (std::isnan(v)) {
                        n_nan++;
                        continue;
                    }
                    if (std::isinf(v)) {
                        n_inf++;
                        continue;
                    }
                    if (v < mn)
                        mn = v;
                    if (v > mx)
                        mx = v;
                    sm += v;
                    sq += v * v;
                }
                float mean = (n > 0) ? sm / (float)n : 0.0f;
                float l2 = std::sqrt(sq);
                std::fprintf(stderr,
                             "funasr_dump: %-28s n=%-8zu min=%12.6f max=%12.6f mean=%12.6f L2=%12.4f nan=%d inf=%d "
                             "first8=[",
                             nm, n, mn, mx, mean, l2, n_nan, n_inf);
                for (size_t i = 0; i < 8 && i < n; i++)
                    std::fprintf(stderr, "%s%.6f", i ? "," : "", buf[i]);
                std::fprintf(stderr, "]\n");
            }
            std::fprintf(stderr, "funasr_dump: ---- end ----\n");
        }
    }

    if (stage_out && stage_name && std::strcmp(stage_name, "mel_features") == 0) {
        stage_out->assign(lfr.begin(), lfr.begin() + (ptrdiff_t)D_lfr * T_lfr);
    } else if (stage_out && stage_name) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, stage_name);
        if (t) {
            const size_t n = ggml_nelements(t);
            stage_out->assign((size_t)n, 0.0f);
            ggml_backend_tensor_get(t, stage_out->data(), 0, n * sizeof(float));
        }
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "audio_adaptor_output");
    const size_t total = (size_t)D_ada * (size_t)T_lfr;
    std::vector<float> result((size_t)total, 0.0f);
    ggml_backend_tensor_get(out, result.data(), 0, total * sizeof(float));
    return result;
}

// ===========================================================================
// Prompt builder
//
// The upstream prompt (FunASRNano.inference + data_load_speech +
// generate_chatml) is:
//
//   <|im_start|>system
//   You are a helpful assistant.<|im_end|>
//   <|im_start|>user
//   {get_prompt(hotwords=[], language=None, itn=True)}<|startofspeech|>!!<|endofspeech|><|im_end|>
//   <|im_start|>assistant
//
// where get_prompt(...) returns "语音转写：" (no hotwords, no language
// specifier, with ITN). The <|startofspeech|>...<|endofspeech|> chunk is
// STRIPPED before tokenization (regex split keeps the marker as its own
// segment but data_load_speech replaces the markers with zero placeholder
// tokens — see fun_asr_nano/model.py:371-413). The final token sequence
// is:
//
//   tokenize("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
//            "<|im_start|>user\n语音转写：")
//   + [0] * fake_token_len
//   + tokenize("<|im_end|>\n<|im_start|>assistant\n")
//
// fake_token_len comes from use_low_frame_rate=True at training time —
// three Conv1d(k=3, s=2, p=1) downsamples + a final //2+1, applied to
// T_lfr. The adaptor itself produces T_lfr frames (downsample_rate=1);
// only the first fake_token_len are spliced into the prompt.
// ===========================================================================

static int compute_fake_token_len(int T_lfr, bool use_low_frame_rate) {
    if (!use_low_frame_rate)
        return T_lfr;
    int o = T_lfr;
    o = (o - 1) / 2 + 1;
    o = (o - 1) / 2 + 1;
    o = (o - 1) / 2 + 1;
    return o;
}
// ===========================================================================
// BPE tokenize (GPT-2 byte-level, Qwen3-compatible). Adapted from
// qwen3_asr.cpp::qwen3_asr_tokenize but kept local — we don't link in
// qwen3_asr's context (it requires audio.* tensors we don't ship).
// ===========================================================================

static std::vector<int32_t> funasr_bpe_encode(const funasr_vocab& v, const std::string& s) {
    std::vector<int32_t> result;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '|') {
            size_t end = s.find("|>", i + 2);
            if (end != std::string::npos) {
                std::string special = s.substr(i, end + 2 - i);
                auto it = v.token_to_id.find(special);
                if (it != v.token_to_id.end()) {
                    result.push_back(it->second);
                    i = end + 2;
                    continue;
                }
            }
        }
        size_t j = i;
        if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|')
            j++;
        while (j < s.size()) {
            if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|') {
                size_t end = s.find("|>", j + 2);
                if (end != std::string::npos) {
                    std::string special = s.substr(j, end + 2 - j);
                    if (v.token_to_id.find(special) != v.token_to_id.end())
                        break;
                }
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty())
            continue;
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n')
                k++;
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n') {
                k++;
            }
            if (k == start)
                k++;
            std::string pre(chunk, start, k - start);
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }
    return result;
}

// Decode token IDs back to UTF-8. The GGUF stores Qwen3's vocab in
// byte-encoded form (the GPT-2 byte→unicode roundtrip), so we reverse
// that mapping on each codepoint.
static std::string funasr_decode_token(const funasr_vocab& v, int id) {
    if (id < 0 || id >= (int)v.id_to_token.size())
        return "";
    const std::string& s = v.id_to_token[(size_t)id];
    // Build the reverse byte_decoder lazily.
    static thread_local std::unordered_map<int, unsigned char> dec;
    if (dec.empty()) {
        const auto& enc = core_bpe::byte_encoder();
        for (int b = 0; b < 256; b++)
            dec[enc[b]] = (unsigned char)b;
    }

    std::string out;
    out.reserve(s.size());
    // Decode UTF-8 codepoints in s and look each one up in dec.
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        unsigned char c = (unsigned char)s[i];
        int adv = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((c & 0x1F) << 6) | ((unsigned char)s[i + 1] & 0x3F);
            adv = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((c & 0x0F) << 12) | (((unsigned char)s[i + 1] & 0x3F) << 6) | ((unsigned char)s[i + 2] & 0x3F);
            adv = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((c & 0x07) << 18) | (((unsigned char)s[i + 1] & 0x3F) << 12) |
                 (((unsigned char)s[i + 2] & 0x3F) << 6) | ((unsigned char)s[i + 3] & 0x3F);
            adv = 4;
        }
        auto it = dec.find((int)cp);
        if (it != dec.end())
            out.push_back((char)it->second);
        // Specials and non-roundtripped codepoints (unlikely for valid Qwen3 tokens) are dropped.
        i += adv;
    }
    return out;
}

// ===========================================================================
// LLM forward graph (KV-cached).
//
// Two flavours, both implemented by `funasr_build_graph_llm_kv_impl`:
//
//   Per-call path (cached_step=false). Used for the prefill pass and any
//   T>1 forward. n_past + T is baked into the graph (Lk = n_past+T) and
//   the K/V writes use ggml_cpy with static byte offsets. The graph is
//   built from the shared compute_meta pool and freed when the caller
//   exits (per legacy behaviour).
//
//   Cached-step path (cached_step=true). Used for the AR decode loop
//   (T = 1). Built once, reused for every step. Lk is fixed at
//   kv_max_ctx so the topology stays constant; K/V writes use
//   ggml_set_rows keyed by the runtime `kv_indices` tensor (which we
//   alias to `positions` — by construction it carries [n_past] for
//   T=1 decode). The causal mask is a runtime input of shape
//   (kv_max_ctx, 1) F16 — mask[k] = 0 if k <= n_past else -inf,
//   refreshed on every step.
//
// The persistent ggml_context used by the cached path stays alive for
// the session's lifetime (freed in funasr_free); the graph object
// + every tensor pointer it holds stay valid across calls.
// ===========================================================================

static ggml_cgraph* funasr_build_graph_llm_kv_impl(funasr_context* ctx, int n_past, int n_tokens, bool cached_step,
                                                   ggml_context* dedicated_ctx0 = nullptr) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_q = (int)hp.llm_n_heads;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv_grp = n_q / n_kv;
    const float eps = hp.llm_rms_eps;
    const float theta = hp.llm_rope_theta;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = n_tokens;
    const int Lk = cached_step ? ctx->kv_max_ctx : (n_past + T);

    GGML_ASSERT(ctx->kv_k && ctx->kv_v);
    GGML_ASSERT(Lk <= ctx->kv_max_ctx);

    ggml_context* ctx0;
    if (cached_step) {
        GGML_ASSERT(dedicated_ctx0 != nullptr);
        ctx0 = dedicated_ctx0;
    } else {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
        ctx0 = ggml_init(ip);
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Cached-step path always needs a runtime causal_mask of shape
    // (kv_max_ctx, 1) — masks the unwritten cache slots to -inf. The
    // per-call path only needs the mask when T > 1 (prefill).
    ggml_tensor* causal_mask = nullptr;
    if (cached_step || T > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    ggml_tensor* cur = embeds;

    const core_attn::KvSelfAttnParams kvp = {
        /*n_heads*/ n_q,
        /*n_kv_heads*/ n_kv,
        /*head_dim*/ hd,
        /*n_kv_grp*/ n_kv_grp,
        /*n_ctx_orig*/ (int)hp.llm_max_pos,
        /*rope_theta*/ theta,
        /*rope_beta_fast*/ 32.0f,
        /*rope_beta_slow*/ 1.0f,
        /*attn_scale*/ attn_scale,
        /*qk_norm_eps*/ eps,
        /*gqa_mode*/ core_attn::GQA_MANUAL_CONT,
    };

    // For the cached step path: tell kv_self_attn to write via
    // ggml_set_rows keyed by `positions` (the runtime n_past tensor)
    // and to read the full kv_max_ctx window — which keeps the graph
    // topology constant across calls. Per-call path leaves both null
    // for the legacy static-offset write.
    ggml_tensor* const kv_indices = cached_step ? positions : nullptr;
    const int fixed_kv_len = cached_step ? ctx->kv_max_ctx : 0;

    // FUNASR_LLM_LAYERS=N limits the number of LLM layers for debugging.
    // Useful with FUNASR_NAN_CHECK=1 to isolate the failing layer faster.
    uint32_t n_layers_eff = hp.llm_n_layers;
    {
        static int layers_override = -1;
        if (layers_override < 0) {
            const char* e = std::getenv("FUNASR_LLM_LAYERS");
            layers_override = (e && *e) ? std::atoi(e) : 0;
        }
        if (layers_override > 0 && (uint32_t)layers_override < n_layers_eff) {
            n_layers_eff = (uint32_t)layers_override;
        }
    }

    for (uint32_t il = 0; il < n_layers_eff; il++) {
        const auto& b = m.llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.attn_norm_w);

        // Use fused QKV when available (§136 — prevents sched cross-backend
        // split that causes all-NaN on CUDA). Falls back to separate Q/K/V
        // when fusion wasn't possible (type mismatch, alloc failure).
        ggml_tensor* attn =
            core_attn::kv_self_attn(ctx0, gf, x, b.attn_q_w, b.attn_k_w, b.attn_v_w, b.attn_output_w, b.attn_q_norm_w,
                                    b.attn_k_norm_w, positions, causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, kvp,
                                    /*qkv_w*/ b.attn_qkv_w, fixed_kv_len, kv_indices);
        cur = ggml_add(ctx0, residual, attn);

        residual = cur;
        x = ggml_rms_norm(ctx0, cur, eps);
        x = ggml_mul(ctx0, x, b.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, x, b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);

        // LLM layer snap — gated on FUNASR_DUMP_STAGES to avoid bloating
        // the graph in production (28 extra dup nodes).
        {
            static int dump_flag = -1;
            if (dump_flag < 0) {
                const char* e = std::getenv("FUNASR_DUMP_STAGES");
                dump_flag = (e && *e && *e != '0') ? 1 : 0;
            }
            if (dump_flag) {
                char nm[32];
                std::snprintf(nm, sizeof(nm), "llm_layer_%u", il);
                ggml_tensor* s = ggml_dup(ctx0, cur);
                ggml_set_name(s, nm);
                ggml_build_forward_expand(gf, s);
            }
        }
    }

    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, m.llm.output_norm_w);

    // Snap the pre-lm_head hidden state for the dump.
    {
        static int dump_flag = -1;
        if (dump_flag < 0) {
            const char* e = std::getenv("FUNASR_DUMP_STAGES");
            dump_flag = (e && *e && *e != '0') ? 1 : 0;
        }
        if (dump_flag) {
            ggml_tensor* s = ggml_dup(ctx0, cur);
            ggml_set_name(s, "llm_pre_lmhead");
            ggml_build_forward_expand(gf, s);
        }
    }

    // Last-token-only lm_head — decode loop only needs next-token logits.
    if (T > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(T - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);

    if (!cached_step) {
        // Per-call path: tensor pool can be freed; the graph's tensors
        // were copied into the sched's allocator on alloc_graph.
        ggml_free(ctx0);
    }
    return gf;
}

static ggml_cgraph* funasr_build_graph_llm_kv(funasr_context* ctx, int n_past, int n_tokens) {
    return funasr_build_graph_llm_kv_impl(ctx, n_past, n_tokens, /*cached_step*/ false);
}

// Lazily build the cached step graph the first time the AR decode loop
// runs. kv_max_ctx must be set (i.e. funasr_kv_init has already
// allocated the KV buffers).
//
// Default OFF — see why: a fixed Lk in the cached graph forces every
// per-step attention to compute over the full kv_max_ctx window, while
// the per-call path only attends to (n_past+1) keys. On samples/jfk.wav
// the cached path measured 135 ms/tok vs the per-call's 42 ms/tok — the
// ~10 ms graph-build savings get eaten by ~100 ms of extra attention
// work. The infrastructure here (kv_indices + fixed_kv_len + dedicated
// gallocr) is correct (77/77 PASS, byte-identical text) and ready for
// PLAN funasr-perf #1 (bucketed Lk graphs, voxcpm2-TSLM pattern) which
// would size the bucket to actual prompt + decode usage and recover
// the graph-build savings without the attention-window penalty.
static void funasr_ensure_step_graph(funasr_context* ctx) {
    if (ctx->step_graph || !ctx->step_graph_cache)
        return;
    if (!ctx->kv_k || ctx->kv_max_ctx <= 0)
        return;

    // 16384-node budget mirrors the per-call path; in practice T=1 +
    // 28-layer Qwen3-0.6B uses only ~1000 nodes, but keeping the
    // budget identical avoids surprises if someone bumps the layer
    // count later.
    ctx->step_compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    ggml_init_params ip = {ctx->step_compute_meta.size(), ctx->step_compute_meta.data(), true};
    ctx->step_ctx0 = ggml_init(ip);
    if (!ctx->step_ctx0) {
        std::fprintf(stderr, "funasr: failed to init step graph context\n");
        return;
    }
    ctx->step_graph = funasr_build_graph_llm_kv_impl(ctx, /*n_past*/ 0, /*n_tokens*/ 1,
                                                     /*cached_step*/ true, ctx->step_ctx0);
    if (!ctx->step_graph) {
        ggml_free(ctx->step_ctx0);
        ctx->step_ctx0 = nullptr;
        return;
    }
    // Reserve the gallocr — allocates the intermediate-activation buffer
    // on the default Metal/CUDA backend buffer type. Per-call cost is
    // just `ggml_gallocr_alloc_graph` (which re-bumps the arena) + tensor
    // sets + compute.
    ctx->step_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ctx->step_galloc || !ggml_gallocr_reserve(ctx->step_galloc, ctx->step_graph)) {
        std::fprintf(stderr, "funasr: failed to reserve step gallocr\n");
        if (ctx->step_galloc) {
            ggml_gallocr_free(ctx->step_galloc);
            ctx->step_galloc = nullptr;
        }
        ggml_free(ctx->step_ctx0);
        ctx->step_ctx0 = nullptr;
        ctx->step_graph = nullptr;
        ctx->step_graph_cache = false;
    }
}

// ===========================================================================
// Embedding lookup graph — input_ids → (d, T) inputs_embeds via token_embd.
// ===========================================================================

static ggml_cgraph* funasr_build_graph_embed(funasr_context* ctx, int n_tokens) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);
    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// KV-cache init / reset
// ===========================================================================

static bool funasr_kv_init(funasr_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
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
    // Invalidate the cached step graph — its tensor pointers reference
    // the about-to-be-freed kv_k/kv_v. Next ensure_step_graph rebuilds
    // against the new cache.
    if (ctx->step_galloc) {
        ggml_gallocr_free(ctx->step_galloc);
        ctx->step_galloc = nullptr;
    }
    if (ctx->step_ctx0) {
        ggml_free(ctx->step_ctx0);
        ctx->step_ctx0 = nullptr;
    }
    ctx->step_graph = nullptr;
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int n_lay = (int)hp.llm_n_layers;

    ggml_init_params kp = {ggml_tensor_overhead() * 4 + 1024, nullptr, true};
    ctx->kv_ctx = ggml_init(kp);
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("funasr");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, n_lay);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, n_lay);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");
    const size_t kbytes = ggml_nbytes(ctx->kv_k);
    const size_t vbytes = ggml_nbytes(ctx->kv_v);
    // When LLM weights are split to CPU (issue #125), KV cache must also
    // be on CPU so the sched routes the entire LLM to CPU.
    ggml_backend_t kv_backend = (ctx->model.buf_cpu)
                                    ? ctx->backend_cpu
                                    : core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "funasr");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, kbytes + vbytes);
    if (!ctx->kv_buf) {
        std::fprintf(stderr, "funasr: failed to allocate kv buffer\n");
        return false;
    }
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + kbytes);
    // Zero-fill the KV cache. On CUDA, ggml_backend_alloc_buffer does not
    // zero memory (cudaMalloc). If the graph scheduler reads a KV slot
    // before the corresponding ggml_cpy writes it (aliasing-based race in
    // the same compute graph), uninitialized NaN/Inf values propagate
    // through flash_attn and poison every downstream tensor. Zeroing is
    // cheap (~1 MB for the funasr Qwen2-0.6B KV) and prevents this.
    {
        std::vector<uint8_t> zeros(std::max(kbytes, vbytes), 0);
        ggml_backend_tensor_set(ctx->kv_k, zeros.data(), 0, kbytes);
        ggml_backend_tensor_set(ctx->kv_v, zeros.data(), 0, vbytes);
    }
    ctx->kv_max_ctx = max_ctx;
    return true;
}

// ===========================================================================
// Per-node NaN/Inf checker — FUNASR_NAN_CHECK=1
//
// Uses the sched eval_callback to inspect every graph node right after it
// runs. Prints the FIRST node whose output contains NaN or Inf, then
// turns itself off so the run completes without flooding the log. The
// callback forces per-node synchronization (slow) — debug-only.
// ===========================================================================

struct funasr_nan_check_state {
    bool found = false;
    int node_idx = 0;
};

static bool funasr_nan_check_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* st = (funasr_nan_check_state*)user_data;
    if (st->found)
        return false; // already found the culprit, stop asking

    if (ask) {
        // We want the data for every non-view node that has a float type.
        if (t->view_src != nullptr)
            return false;
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)
            return false;
        return true; // yes, read this node back
    }

    // ask == false → data is ready, inspect it.
    const size_t n = ggml_nelements(t);
    if (n == 0) {
        st->node_idx++;
        return true;
    }

    std::vector<float> buf(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else {
        // F16 — read raw then convert
        std::vector<ggml_fp16_t> h(n);
        ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            buf[i] = ggml_fp16_to_fp32(h[i]);
    }

    int n_nan = 0, n_inf = 0;
    float mx = -1e30f, mn = 1e30f;
    for (size_t i = 0; i < n; i++) {
        float v = buf[i];
        if (std::isnan(v)) {
            n_nan++;
            continue;
        }
        if (std::isinf(v)) {
            n_inf++;
            continue;
        }
        if (v > mx)
            mx = v;
        if (v < mn)
            mn = v;
    }

    // Print every checked node (concise) so we can trace the full sequence.
    std::fprintf(
        stderr,
        "funasr_nan_check: node#%-3d op=%-12s name=%-30s ne=[%lld,%lld,%lld,%lld] min=%.4g max=%.4g nan=%d inf=%d\n",
        st->node_idx, ggml_op_name(t->op), t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
        (long long)t->ne[3], (double)mn, (double)mx, n_nan, n_inf);

    if (n_nan > 0 || n_inf > 0) {
        std::fprintf(stderr, "funasr_nan_check: *** FIRST BAD NODE ABOVE ***\n");
        // Dump source tensor stats by reading them back
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (!t->src[j])
                continue;
            auto* s = t->src[j];
            size_t sn = ggml_nelements(s);
            std::vector<float> sb(sn);
            bool readable = (s->type == GGML_TYPE_F32 || s->type == GGML_TYPE_F16) && s->data;
            if (readable) {
                if (s->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(s, sb.data(), 0, sn * sizeof(float));
                } else {
                    std::vector<ggml_fp16_t> sh(sn);
                    ggml_backend_tensor_get(s, sh.data(), 0, sn * sizeof(ggml_fp16_t));
                    for (size_t i = 0; i < sn; i++)
                        sb[i] = ggml_fp16_to_fp32(sh[i]);
                }
                int snan = 0, sinf = 0;
                float smx = -1e30f, smn = 1e30f;
                for (size_t i = 0; i < sn; i++) {
                    float v = sb[i];
                    if (std::isnan(v)) {
                        snan++;
                        continue;
                    }
                    if (std::isinf(v)) {
                        sinf++;
                        continue;
                    }
                    if (v > smx)
                        smx = v;
                    if (v < smn)
                        smn = v;
                }
                std::fprintf(stderr,
                             "  src[%d]: op=%-12s name=%-30s type=%-4s ne=[%lld,%lld,%lld,%lld] "
                             "min=%.6g max=%.6g nan=%d inf=%d\n",
                             j, ggml_op_name(s->op), s->name, ggml_type_name(s->type), (long long)s->ne[0],
                             (long long)s->ne[1], (long long)s->ne[2], (long long)s->ne[3], (double)smn, (double)smx,
                             snan, sinf);
            } else {
                std::fprintf(stderr,
                             "  src[%d]: op=%-12s name=%-30s type=%-4s ne=[%lld,%lld,%lld,%lld] [not readable]\n", j,
                             ggml_op_name(s->op), s->name, ggml_type_name(s->type), (long long)s->ne[0],
                             (long long)s->ne[1], (long long)s->ne[2], (long long)s->ne[3]);
            }
        }
        st->found = true;
        return false; // stop processing further
    }

    st->node_idx++;
    return true; // continue
}

// ===========================================================================
// Run the LLM once. n_past=0 + n_tokens=T_prompt = prefill; subsequent
// calls with n_tokens=1 are the per-step decode.
// ===========================================================================

static std::vector<float> funasr_run_llm_step(funasr_context* ctx, const float* inputs_embeds, int n_tokens,
                                              int n_past) {
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;

    // Decide which graph to drive. Single-token decode steps use the
    // cached step graph (built once, reused) when step_graph_cache is on;
    // anything else (prefill, opt-out) takes the per-call path.
    const bool use_cached = ctx->step_graph_cache && n_tokens == 1;
    if (use_cached) {
        funasr_ensure_step_graph(ctx);
        if (!ctx->step_graph) {
            // Build failed — fall back to per-call. Logged inside ensure.
            // (Setting use_cached=false would shadow the const; just goto
            // the per-call branch below by clearing the cache once.)
            ctx->step_graph_cache = false;
        }
    }

    if (use_cached && ctx->step_graph && ctx->step_galloc) {
        // ---- Cached-step path (T=1, Lk = kv_max_ctx, kv_indices=positions). ----
        // Bypass the sched (which re-plans per call and conflicts with
        // graphs holding views into pre-allocated KV buffers); drive
        // the gallocr + backend directly, voxcpm2-TSLM style.
        const int Lk = ctx->kv_max_ctx;
        const int32_t pos = (int32_t)n_past;

        std::vector<ggml_fp16_t> mask((size_t)Lk);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int k = 0; k < Lk; k++)
            mask[(size_t)k] = (k <= n_past) ? zero_h : ninf_h;

        ggml_cgraph* gf = ctx->step_graph;
        if (!ggml_gallocr_alloc_graph(ctx->step_galloc, gf)) {
            std::fprintf(stderr, "funasr: failed to alloc cached step graph\n");
            return {};
        }
        ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
        ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * sizeof(float));
        ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
        ggml_backend_tensor_set(pos_in, &pos, 0, sizeof(int32_t));
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

        if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "funasr: cached step graph compute failed\n");
            return {};
        }
        ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
        std::vector<float> result((size_t)vocab, 0.0f);
        ggml_backend_tensor_get(out, result.data(), 0, (size_t)vocab * sizeof(float));
        return result;
    }

    // ---- Per-call path (prefill or step-cache opt-out). ----
    const int Lk = n_past + n_tokens;

    std::vector<int32_t> positions((size_t)n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[(size_t)i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        mask.assign((size_t)Lk * (size_t)n_tokens, zero_h);
        for (int q = 0; q < n_tokens; q++)
            for (int k = n_past + q + 1; k < Lk; k++)
                mask[(size_t)q * Lk + k] = ninf_h;
    }

    ggml_cgraph* gf = funasr_build_graph_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        std::fprintf(stderr, "funasr: failed to alloc llm graph\n");
        return {};
    }
    ggml_tensor* embeds_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(embeds_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(pos_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    // Install per-node NaN checker when FUNASR_NAN_CHECK=1 (prefill only).
    funasr_nan_check_state nan_state;
    {
        static int nan_check_flag = -1;
        if (nan_check_flag < 0) {
            const char* e = std::getenv("FUNASR_NAN_CHECK");
            nan_check_flag = (e && *e && *e != '0') ? 1 : 0;
        }
        if (nan_check_flag && n_tokens > 1) {
            std::fprintf(stderr, "funasr: NaN checker enabled — per-node sync (slow)\n");
            ggml_backend_sched_set_eval_callback(ctx->sched, funasr_nan_check_cb, &nan_state);
        }
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "funasr: llm graph compute failed\n");
        ggml_backend_sched_set_eval_callback(ctx->sched, nullptr, nullptr);
        return {};
    }

    // Clear callback after compute.
    ggml_backend_sched_set_eval_callback(ctx->sched, nullptr, nullptr);
    if (nan_state.found) {
        std::fprintf(stderr,
                     "funasr: NaN/Inf detected at node #%d (see above). "
                     "Continuing to produce dump for comparison.\n",
                     nan_state.node_idx);
    }

    // ---- LLM layer dump (FUNASR_DUMP_STAGES=1, prefill only) ----
    {
        static int dump_flag = -1;
        if (dump_flag < 0) {
            const char* e = std::getenv("FUNASR_DUMP_STAGES");
            dump_flag = (e && *e && *e != '0') ? 1 : 0;
        }
        if (dump_flag && n_tokens > 1) {
            std::fprintf(stderr, "funasr_dump: ---- LLM layer stats (n_tokens=%d, n_past=%d, n_layers=%u) ----\n",
                         n_tokens, n_past, ctx->model.hparams.llm_n_layers);
            for (uint32_t li = 0; li < ctx->model.hparams.llm_n_layers; li++) {
                char nmbuf[32];
                std::snprintf(nmbuf, sizeof(nmbuf), "llm_layer_%u", li);
                const char* nm = nmbuf;
                ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
                if (!t) {
                    std::fprintf(stderr, "funasr_dump: %-28s [not found]\n", nm);
                    continue;
                }
                const size_t n = ggml_nelements(t);
                std::vector<float> buf(n, 0.0f);
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
                float mn = buf[0], mx = buf[0], sm = 0.0f;
                int n_nan = 0, n_inf = 0;
                for (size_t i = 0; i < n; i++) {
                    float v = buf[i];
                    if (std::isnan(v)) {
                        n_nan++;
                        continue;
                    }
                    if (std::isinf(v)) {
                        n_inf++;
                        continue;
                    }
                    if (v < mn)
                        mn = v;
                    if (v > mx)
                        mx = v;
                    sm += v;
                }
                float mean = (n > 0) ? sm / (float)n : 0.0f;
                std::fprintf(stderr,
                             "funasr_dump: %-28s n=%-8zu min=%12.6f max=%12.6f mean=%12.6f nan=%d inf=%d first8=[", nm,
                             n, mn, mx, mean, n_nan, n_inf);
                for (size_t i = 0; i < 8 && i < n; i++)
                    std::fprintf(stderr, "%s%.6f", i ? "," : "", buf[i]);
                std::fprintf(stderr, "]\n");
            }
            // Also dump the pre-lm_head hidden state.
            {
                const char* nm = "llm_pre_lmhead";
                ggml_tensor* t = ggml_graph_get_tensor(gf, nm);
                if (t) {
                    const size_t n = ggml_nelements(t);
                    std::vector<float> buf(n, 0.0f);
                    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
                    float mn = buf[0], mx = buf[0], sm = 0.0f;
                    int n_nan = 0, n_inf = 0;
                    for (size_t i = 0; i < n; i++) {
                        float v = buf[i];
                        if (std::isnan(v)) {
                            n_nan++;
                        } else if (std::isinf(v)) {
                            n_inf++;
                        } else {
                            if (v < mn)
                                mn = v;
                            if (v > mx)
                                mx = v;
                            sm += v;
                        }
                    }
                    std::fprintf(stderr, "funasr_dump: %-28s n=%-8zu min=%12.6f max=%12.6f mean=%12.6f nan=%d inf=%d\n",
                                 nm, n, mn, mx, (n > 0) ? sm / (float)n : 0.0f, n_nan, n_inf);
                }
            }
            std::fprintf(stderr, "funasr_dump: ---- end LLM ----\n");
        }
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    std::vector<float> result((size_t)vocab, 0.0f);
    ggml_backend_tensor_get(out, result.data(), 0, (size_t)vocab * sizeof(float));
    return result;
}

// Get the inputs_embeds for an arbitrary token sequence via the model's
// token_embd table (shared with output.weight thanks to Qwen3 tied embeddings).
static std::vector<float> funasr_embed_tokens(funasr_context* ctx, const std::vector<int32_t>& ids) {
    const int n = (int)ids.size();
    const int d = (int)ctx->model.hparams.llm_d_model;
    ggml_cgraph* gf = funasr_build_graph_embed(ctx, n);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        std::fprintf(stderr, "funasr: failed to alloc embed graph\n");
        return {};
    }
    ggml_tensor* ids_in = ggml_graph_get_tensor(gf, "input_ids");
    ggml_backend_tensor_set(ids_in, ids.data(), 0, (size_t)n * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "funasr: embed graph compute failed\n");
        return {};
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    std::vector<float> result((size_t)d * (size_t)n, 0.0f);
    ggml_backend_tensor_get(out, result.data(), 0, (size_t)d * (size_t)n * sizeof(float));
    return result;
}

// ===========================================================================
// High-level pipeline: build prompt, splice audio, run AR decode → text.
// ===========================================================================

static const char* PROMPT_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                   "\xE8\xAF\xAD\xE9\x9F\xB3\xE8\xBD\xAC\xE5\x86\x99\xEF\xBC\x9A"; // "语音转写："
static const char* PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";

static std::string funasr_transcribe_impl(funasr_context* ctx, const float* pcm, int n_samples,
                                          std::vector<float>* stage_out, const char* stage_name,
                                          std::vector<int32_t>* out_ids = nullptr,
                                          std::vector<float>* out_probs = nullptr) {
    const auto& hp = ctx->model.hparams;

    int T_lfr = 0, D_lfr = 0;
    std::vector<float> lfr;
    {
        funasr_bench_stage s("fbank+lfr");
        lfr = funasr_compute_features(ctx, pcm, n_samples, T_lfr, D_lfr);
    }
    if (T_lfr <= 0)
        return "";
    if (stage_out && stage_name && std::strcmp(stage_name, "mel_features") == 0) {
        stage_out->assign(lfr.begin(), lfr.begin() + (ptrdiff_t)D_lfr * T_lfr);
        return ""; // mel-only request — short-circuit
    }

    std::vector<float> adaptor_out;
    {
        funasr_bench_stage s("enc+ada_compute");
        adaptor_out = funasr_run_encoder_adaptor(ctx, lfr, T_lfr, D_lfr, stage_out, stage_name);
    }
    if (adaptor_out.empty())
        return "";
    if (stage_out && stage_name &&
        (std::strncmp(stage_name, "encoder_", 8) == 0 || std::strncmp(stage_name, "audio_adaptor_", 14) == 0)) {
        return "";
    }

    // Build the prompt.
    std::vector<int32_t> prefix_ids;
    std::vector<int32_t> suffix_ids;
    {
        funasr_bench_stage s("prompt_tokenize");
        prefix_ids = funasr_bpe_encode(ctx->vocab, PROMPT_PREFIX);
        suffix_ids = funasr_bpe_encode(ctx->vocab, PROMPT_SUFFIX);
    }
    const int fake_token_len = compute_fake_token_len(T_lfr, hp.use_low_frame_rate);
    const int fbank_beg = (int)prefix_ids.size();
    const int total_prompt = (int)prefix_ids.size() + fake_token_len + (int)suffix_ids.size();
    if (std::getenv("CRISPASR_VERBOSE")) {
        std::fprintf(
            stderr, "funasr: T_lfr=%d use_low_frame_rate=%d fake_token_len=%d (prefix=%zu suffix=%zu prompt=%d)\n",
            T_lfr, (int)hp.use_low_frame_rate, fake_token_len, prefix_ids.size(), suffix_ids.size(), total_prompt);
    }
    std::vector<int32_t> ids((size_t)total_prompt, 0);
    std::copy(prefix_ids.begin(), prefix_ids.end(), ids.begin());
    std::copy(suffix_ids.begin(), suffix_ids.end(), ids.begin() + fbank_beg + fake_token_len);

    // Embed the prompt and splice in the (truncated) adaptor output.
    std::vector<float> inputs_embeds;
    {
        funasr_bench_stage s("embed");
        inputs_embeds = funasr_embed_tokens(ctx, ids);
        if (inputs_embeds.empty())
            return "";
        const int d = (int)hp.llm_d_model;
        const int D_ada = (int)hp.ada_d_out;
        GGML_ASSERT(d == D_ada);
        // adaptor_out is (D_ada, T_lfr) row-major (column-major in PyTorch —
        // here adaptor_out[t*D_ada + i] is row t, col i). Splice is by row.
        for (int t = 0; t < fake_token_len && t < T_lfr; t++) {
            std::memcpy(inputs_embeds.data() + (size_t)(fbank_beg + t) * d, adaptor_out.data() + (size_t)t * D_ada,
                        (size_t)d * sizeof(float));
        }
    }
    const int d = (int)hp.llm_d_model;

    // ---- Dump spliced embeddings (FUNASR_DUMP_STAGES=1) ----
    {
        static int dump_flag = -1;
        if (dump_flag < 0) {
            const char* e = std::getenv("FUNASR_DUMP_STAGES");
            dump_flag = (e && *e && *e != '0') ? 1 : 0;
        }
        if (dump_flag) {
            // Stats over the audio portion of the spliced embeddings.
            const size_t audio_start = (size_t)fbank_beg * (size_t)d;
            const size_t audio_len = (size_t)std::min(fake_token_len, T_lfr) * (size_t)d;
            float mn = inputs_embeds[audio_start], mx = mn, sm = 0.0f, sq = 0.0f;
            int n_nan = 0, n_inf = 0;
            for (size_t i = 0; i < audio_len; i++) {
                float v = inputs_embeds[audio_start + i];
                if (std::isnan(v)) {
                    n_nan++;
                    continue;
                }
                if (std::isinf(v)) {
                    n_inf++;
                    continue;
                }
                if (v < mn)
                    mn = v;
                if (v > mx)
                    mx = v;
                sm += v;
                sq += v * v;
            }
            float mean = audio_len > 0 ? sm / (float)audio_len : 0.0f;
            float l2 = std::sqrt(sq);
            std::fprintf(stderr,
                         "funasr_dump: %-28s n=%-8zu min=%12.6f max=%12.6f mean=%12.6f L2=%12.4f nan=%d inf=%d "
                         "first8=[",
                         "spliced_audio_embeds", audio_len, mn, mx, mean, l2, n_nan, n_inf);
            for (size_t i = 0; i < 8 && i < audio_len; i++)
                std::fprintf(stderr, "%s%.6f", i ? "," : "", inputs_embeds[audio_start + i]);
            std::fprintf(stderr, "]\n");

            // Also dump the first prefill logits after they're computed.
        }
    }

    // KV cache sized for prompt + up to max_new_tokens.
    const int max_new_tokens = 512;
    {
        funasr_bench_stage s("kv_init");
        // Tight kv_max_ctx — only allocate what this session actually needs.
        // The cached step graph (when enabled) attends to the full Lk window
        // every step, so over-allocation directly costs decode time.
        if (!funasr_kv_init(ctx, total_prompt + max_new_tokens + 16)) {
            std::fprintf(stderr, "funasr: kv_init failed\n");
            return "";
        }
    }

    // Prefill — feed the entire prompt at n_past=0; we get only the last
    // token's logits back from the slice in build_graph_llm_kv.
    std::vector<float> logits;
    {
        funasr_bench_stage s("llm_prefill");
        logits = funasr_run_llm_step(ctx, inputs_embeds.data(), total_prompt, 0);
    }
    if (logits.empty())
        return "";

    // ---- Dump prefill logits (FUNASR_DUMP_STAGES=1) ----
    {
        static int dump_flag = -1;
        if (dump_flag < 0) {
            const char* e = std::getenv("FUNASR_DUMP_STAGES");
            dump_flag = (e && *e && *e != '0') ? 1 : 0;
        }
        if (dump_flag) {
            float mn = logits[0], mx = mn, sm = 0.0f;
            int n_nan = 0, n_inf = 0;
            for (float v : logits) {
                if (std::isnan(v)) {
                    n_nan++;
                    continue;
                }
                if (std::isinf(v)) {
                    n_inf++;
                    continue;
                }
                if (v < mn)
                    mn = v;
                if (v > mx)
                    mx = v;
                sm += v;
            }
            int top = 0;
            for (int i = 1; i < (int)logits.size(); i++)
                if (logits[i] > logits[top])
                    top = i;
            std::fprintf(stderr,
                         "funasr_dump: %-28s n=%-8zu min=%12.6f max=%12.6f mean=%12.6f argmax=%-6d nan=%d inf=%d\n",
                         "prefill_logits", logits.size(), mn, mx, sm / (float)logits.size(), top, n_nan, n_inf);
        }
    }

    auto argmax = [](const std::vector<float>& v) {
        int best = 0;
        float bv = v[0];
        for (int i = 1; i < (int)v.size(); i++) {
            if (v[i] > bv) {
                bv = v[i];
                best = i;
            }
        }
        return best;
    };
    auto softmax_of = [](const std::vector<float>& v, int idx) -> float {
        float mx = *std::max_element(v.begin(), v.end());
        float sum = 0;
        for (float x : v)
            sum += std::exp(x - mx);
        return std::exp(v[idx] - mx) / sum;
    };
    const float temperature = ctx->params.temperature;
    auto pick_next = [&](std::vector<float>& v) -> int {
        if (temperature > 0.0f) {
            // Temperature sampling
            float mx = *std::max_element(v.begin(), v.end());
            std::vector<float> probs(v.size());
            float sum = 0;
            for (size_t i = 0; i < v.size(); i++) {
                probs[i] = std::exp((v[i] - mx) / temperature);
                sum += probs[i];
            }
            for (auto& p : probs)
                p /= sum;
            static thread_local std::mt19937 rng(std::random_device{}());
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            return dist(rng);
        }
        return argmax(v);
    };
    std::vector<int32_t> generated;
    std::vector<float> generated_probs;
    auto decode_t0 = std::chrono::steady_clock::now();

    if (ctx->beam_size > 1) {
        // Beam search via replay-from-prefix. Each beam step embeds
        // the full suffix token-by-token and forwards through the LLM.
        auto replay = [](funasr_context* c, const int32_t* toks, int n, int prompt_len) -> float* {
            const int d = (int)c->model.hparams.llm_d_model;
            // Embed all suffix tokens and concatenate
            std::vector<float> embeds((size_t)n * d);
            for (int i = 0; i < n; i++) {
                auto e = funasr_embed_tokens(c, {toks[i]});
                if (e.empty())
                    return nullptr;
                std::memcpy(embeds.data() + (size_t)i * d, e.data(), d * sizeof(float));
            }
            auto lg = funasr_run_llm_step(c, embeds.data(), n, prompt_len);
            if (lg.empty())
                return nullptr;
            float* out = (float*)std::malloc(lg.size() * sizeof(float));
            std::memcpy(out, lg.data(), lg.size() * sizeof(float));
            return out;
        };
        core_beam_decode::Config bcfg;
        bcfg.max_new_tokens = max_new_tokens;
        bcfg.eos_id = (int)hp.eos_token_id;
        bcfg.vocab_size = (int)hp.llm_vocab_size;
        bcfg.beam_size = ctx->beam_size;
        bcfg.prompt_len = total_prompt;
        auto br = core_beam_decode::run_with_probs(ctx, logits.data(), replay, bcfg);
        for (size_t i = 0; i < br.tokens.size(); i++) {
            if (br.tokens[i] == (int32_t)hp.eos_token_id)
                break;
            generated.push_back(br.tokens[i]);
            generated_probs.push_back(br.probs[i]);
        }
    } else {
        int next_id = pick_next(logits);
        float next_prob = softmax_of(logits, next_id);
        int n_past = total_prompt;
        int prev_id = -1;
        int repeat_run = 0;
        for (int step = 0; step < max_new_tokens && next_id != (int)hp.eos_token_id; step++) {
            if (next_id == prev_id) {
                if (++repeat_run >= 20) {
                    std::fprintf(stderr,
                                 "funasr: greedy decode degenerated (token %d repeated >%d times). "
                                 "Aborting at step %d. This usually means the audio adaptor / encoder "
                                 "produced unusable embeddings — re-run with CRISPASR_VERBOSE=1 to "
                                 "inspect frames_spliced.\n",
                                 next_id, repeat_run, step);
                    break;
                }
            } else {
                repeat_run = 0;
                prev_id = next_id;
            }
            generated.push_back(next_id);
            generated_probs.push_back(next_prob);
            std::vector<float> step_embed = funasr_embed_tokens(ctx, {next_id});
            if (step_embed.empty())
                break;
            logits = funasr_run_llm_step(ctx, step_embed.data(), 1, n_past);
            if (logits.empty())
                break;
            n_past += 1;
            next_id = pick_next(logits);
            next_prob = softmax_of(logits, next_id);
        }
    }
    if (funasr_bench_enabled()) {
        auto decode_t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(decode_t1 - decode_t0).count();
        const int n_steps = (int)generated.size();
        std::fprintf(stderr, "  funasr_bench: %-22s %.2f ms  (%d tokens, %.2f ms/tok)\n", "llm_decode_total", ms,
                     n_steps, n_steps > 0 ? ms / n_steps : 0.0);
    }
    (void)d;

    // Pass back raw token IDs and probabilities if requested.
    if (out_ids)
        *out_ids = generated;
    if (out_probs)
        *out_probs = generated_probs;

    // Detokenize, skipping any special tokens that survived.
    std::string out;
    for (int id : generated) {
        const std::string& tok = ctx->vocab.id_to_token[(size_t)id];
        // Skip ChatML / special markers — Qwen3's tokenizer.batch_decode does
        // this with skip_special_tokens=True by default.
        if (tok.size() >= 2 && tok[0] == '<' && tok[1] == '|')
            continue;
        out += funasr_decode_token(ctx->vocab, id);
    }
    return out;
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" funasr_context_params funasr_context_default_params(void) {
    funasr_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.temperature = 0.0f;
    return p;
}

extern "C" funasr_context* funasr_init_from_file(const char* path, funasr_context_params params) {
    funasr_context* ctx = new funasr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!funasr_load_model(ctx->model, ctx->vocab, path, ctx->backend, ctx->backend_cpu)) {
        delete ctx;
        return nullptr;
    }

    // ---- Fuse Q/K/V into a single QKV weight per LLM layer (§136). ----
    // The ggml_backend_sched with [CUDA,CPU] misroutes the 3 separate
    // mul_mats across backends → Inf → NaN on CUDA (issue #125). A single
    // fused matmul forces the sched to keep it on one backend.
    {
        auto& blocks = ctx->model.llm.blocks;
        bool can_fuse = !blocks.empty() && blocks[0].attn_q_w && blocks[0].attn_k_w && blocks[0].attn_v_w;
        if (can_fuse) {
            const ggml_type t0 = blocks[0].attn_q_w->type;
            for (auto& b : blocks) {
                if (!b.attn_q_w || !b.attn_k_w || !b.attn_v_w || b.attn_q_w->type != t0 || b.attn_k_w->type != t0 ||
                    b.attn_v_w->type != t0 || b.attn_q_w->ne[0] != b.attn_k_w->ne[0] ||
                    b.attn_q_w->ne[0] != b.attn_v_w->ne[0]) {
                    can_fuse = false;
                    break;
                }
            }
        }
        if (can_fuse) {
            const ggml_type t0 = blocks[0].attn_q_w->type;
            int q_out = (int)blocks[0].attn_q_w->ne[1];
            int k_out = (int)blocks[0].attn_k_w->ne[1];
            int hidden = (int)blocks[0].attn_q_w->ne[0];
            int qkv_out = q_out + 2 * k_out;
            size_t fused_mem = ggml_tensor_overhead() * blocks.size() + 256;
            ggml_init_params fgp = {fused_mem, nullptr, true};
            ctx->fused_ctx = ggml_init(fgp);
            if (ctx->fused_ctx) {
                for (auto& b : blocks)
                    b.attn_qkv_w = ggml_new_tensor_2d(ctx->fused_ctx, t0, hidden, qkv_out);
                ctx->fused_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                    ctx->fused_ctx, ggml_backend_get_default_buffer_type(ctx->backend));
                if (ctx->fused_buf) {
                    for (auto& b : blocks) {
                        size_t qb = ggml_nbytes(b.attn_q_w), kb = ggml_nbytes(b.attn_k_w);
                        std::vector<uint8_t> tmp(qb + 2 * kb);
                        ggml_backend_tensor_get(b.attn_q_w, tmp.data(), 0, qb);
                        ggml_backend_tensor_get(b.attn_k_w, tmp.data() + qb, 0, kb);
                        ggml_backend_tensor_get(b.attn_v_w, tmp.data() + qb + kb, 0, kb);
                        ggml_backend_tensor_set(b.attn_qkv_w, tmp.data(), 0, tmp.size());
                    }
                    if (params.verbosity >= 1)
                        std::fprintf(stderr, "funasr: fused QKV for %zu LLM layers (%d+%d+%d→%d, type=%s)\n",
                                     blocks.size(), q_out, k_out, k_out, qkv_out, ggml_type_name(t0));
                } else {
                    ggml_free(ctx->fused_ctx);
                    ctx->fused_ctx = nullptr;
                    for (auto& b : blocks)
                        b.attn_qkv_w = nullptr;
                }
            }
        }
    }

    {
        int n_be = 0;
        ggml_backend_t backends[2];
        backends[n_be++] = ctx->backend;
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
            backends[n_be++] = ctx->backend_cpu;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    }
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    // FA opt-out: keep the historical mul_mat + soft_max_ext path for
    // regression diffing. The default ON path matches the parakeet /
    // qwen3-asr default; FUNASR_NO_FA=1 reverts to the original kernel
    // sequence.
    if (const char* s = std::getenv("FUNASR_NO_FA")) {
        if (*s && *s != '0')
            ctx->enc_flash_attn = false;
    }
    // Step-graph cache opt-in. The default-off path runs ~3× faster on
    // typical ASR workloads — the cached graph attends to a fixed
    // (kv_max_ctx-wide) window every step, which costs more than it
    // saves vs the per-call path's growing-Lk attention. The cache
    // becomes a win only with bucketed Lk graphs (voxcpm2 TSLM pattern,
    // PLAN funasr-perf #1) — when that lands we'll flip the default.
    if (const char* s = std::getenv("FUNASR_STEP_CACHE")) {
        if (*s && *s != '0')
            ctx->step_graph_cache = true;
    }

    if (params.verbosity >= 1) {
        std::fprintf(stderr,
                     "funasr: loaded %s  (enc %u blocks + tp %u blocks, adaptor %u, llm %u, vocab %u, fa=%s, "
                     "step_cache=%s)\n",
                     path, ctx->model.hparams.n_blocks_base, ctx->model.hparams.n_blocks_tp,
                     ctx->model.hparams.ada_n_layers, ctx->model.hparams.llm_n_layers,
                     (uint32_t)ctx->vocab.id_to_token.size(), ctx->enc_flash_attn ? "on" : "off",
                     ctx->step_graph_cache ? "on" : "off");
    }
    return ctx;
}

extern "C" void funasr_set_beam_size(funasr_context* ctx, int beam_size) {
    if (!ctx)
        return;
    ctx->beam_size = beam_size > 1 ? beam_size : 1;
}

extern "C" void funasr_free(funasr_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->step_galloc)
        ggml_gallocr_free(ctx->step_galloc);
    if (ctx->step_ctx0)
        ggml_free(ctx->step_ctx0);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->fused_buf)
        ggml_backend_buffer_free(ctx->fused_buf);
    if (ctx->fused_ctx)
        ggml_free(ctx->fused_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" char* funasr_transcribe(funasr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::string s = funasr_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr);
    char* out = (char*)std::malloc(s.size() + 1);
    if (!out)
        return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

extern "C" float* funasr_extract_stage(funasr_context* ctx, const float* samples, int n_samples, const char* stage_name,
                                       int* n_out) {
    if (n_out)
        *n_out = 0;
    if (!ctx || !samples || n_samples <= 0 || !stage_name)
        return nullptr;
    ctx->requested_stage = stage_name;

    if (std::strcmp(stage_name, "generated_text") == 0) {
        std::string txt = funasr_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr);
        char* buf = (char*)std::malloc(txt.size() + 1);
        if (!buf)
            return nullptr;
        std::memcpy(buf, txt.data(), txt.size());
        buf[txt.size()] = '\0';
        if (n_out)
            *n_out = (int)txt.size();
        // cppcheck-suppress invalidPointerCast
        return (float*)buf; // caller casts back to char* (generated_text path)
    }

    std::vector<float> staged;
    (void)funasr_transcribe_impl(ctx, samples, n_samples, &staged, stage_name);
    if (staged.empty())
        return nullptr;
    float* out = (float*)std::malloc(staged.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, staged.data(), staged.size() * sizeof(float));
    if (n_out)
        *n_out = (int)staged.size();
    return out;
}

extern "C" funasr_result* funasr_transcribe_with_probs(funasr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    std::vector<int32_t> ids;
    std::vector<float> probs;
    std::string s = funasr_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr, &ids, &probs);

    auto* r = (funasr_result*)std::malloc(sizeof(funasr_result));
    if (!r)
        return nullptr;
    r->text = (char*)std::malloc(s.size() + 1);
    if (r->text) {
        std::memcpy(r->text, s.data(), s.size());
        r->text[s.size()] = '\0';
    }
    r->n_tokens = (int)ids.size();
    r->token_ids = (int32_t*)std::malloc(ids.size() * sizeof(int32_t));
    r->token_probs = (float*)std::malloc(probs.size() * sizeof(float));
    if (r->token_ids)
        std::memcpy(r->token_ids, ids.data(), ids.size() * sizeof(int32_t));
    if (r->token_probs)
        std::memcpy(r->token_probs, probs.data(), probs.size() * sizeof(float));
    return r;
}

extern "C" void funasr_result_free(funasr_result* r) {
    if (!r)
        return;
    std::free(r->text);
    std::free(r->token_ids);
    std::free(r->token_probs);
    std::free(r);
}

extern "C" const char* funasr_token_text(funasr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    static thread_local std::string buf;
    buf = funasr_decode_token(ctx->vocab, id);
    return buf.c_str();
}
