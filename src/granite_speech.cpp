// granite_speech.cpp — ibm-granite/granite-4.0-1b-speech ggml runtime
//
// Three-module speech-LLM:
//   1. 16-layer Conformer encoder (Macaron FFN, depthwise conv, rel pos emb)
//   2. 2-layer BLIP-2 Q-Former projector (3 learned query tokens → 3 LLM tokens)
//   3. 40-layer Granite 1B LLM (GQA 16/4, μP multipliers, RoPE)

#include "granite_speech.h"
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "core/ffn.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/conformer_ibm.h"
#include "core/cpu_ops.h"
#include "core/fft.h"
#include "core/granite_llm.h"
#include "core/mel.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <climits>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation
// ===========================================================================
//
// Enable per-stage timing with `GRANITE_BENCH=1`. The cost when disabled is
// one cached env-var read per stage (≈1 ns) — safe to leave compiled in.
// Useful for A/B-ing the CPU-loop encoder vs the GRANITE_ENCODER_GRAPH=1
// path and for spotting which stage dominates after future changes.

static bool granite_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("GRANITE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct granite_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit granite_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~granite_bench_stage() {
        if (!granite_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "  bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct granite_speech_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;

    // Encoder
    uint32_t enc_n_layers = 16;
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_heads = 8;
    uint32_t enc_head_dim = 128;
    uint32_t enc_input_dim = 160;
    uint32_t enc_conv_kernel = 15;
    uint32_t enc_ff_dim = 4096;
    uint32_t enc_context_size = 200; // block-local attention window
    uint32_t enc_max_pos_emb = 512;  // Shaw RPE table half-width

    // Projector (Q-Former)
    uint32_t proj_n_layers = 2;
    uint32_t proj_d_model = 1024;
    uint32_t proj_n_heads = 16;
    uint32_t proj_ff_dim = 4096;
    // For granite-speech-4.1-2b-plus: encoder feeds the projector a
    // concatenation of (cat_layers + final_layer). Default = same as
    // proj_d_model (no concat). Plus sets it to 2 * proj_d_model = 2048.
    uint32_t proj_encoder_hidden_size = 1024;
    // Comma-separated 0-indexed encoder layer indices to concatenate
    // with the final layer output before the projector. Empty = no
    // concat (base / 4.0 / 4.1 base behaviour). Plus = "3".
    std::string proj_cat_layers = "";

    // LLM
    uint32_t llm_n_layers = 40;
    uint32_t llm_d_model = 2048;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 4;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 4096;
    float llm_rope_theta = 10000.0f;
    float llm_rms_eps = 1e-5f;
    uint32_t llm_vocab_size = 100353;

    // μP multipliers
    float embedding_multiplier = 12.0f;
    float attention_multiplier = 0.0078125f; // 1/128
    float residual_multiplier = 0.22f;
    float logits_scaling = 8.0f;

    uint32_t downsample_rate = 5;
    uint32_t window_size = 15;
    uint32_t audio_token_index = 100352;

    // Control-token ids used by the CLI adapter (the C++ decode loop needs
    // them to stop on EOS and to splice audio features at the correct
    // position). Default to the granite-4.0-1b values so older GGUFs
    // without these keys keep working unchanged.
    uint32_t eos_token_id = 100257;
    uint32_t bos_token_id = 100257;
};

// ===========================================================================
// Model tensors
// ===========================================================================

struct granite_enc_block {
    // Attention
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_kv_w = nullptr; // combined K+V: (2*head_dim*n_heads, d_model)
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* attn_rel_pos_w = nullptr; // (max_pos*2+1, head_dim)

    // Conv module
    ggml_tensor* conv_up_w = nullptr;
    ggml_tensor* conv_up_b = nullptr;
    ggml_tensor* conv_dw_w = nullptr; // depthwise: (2*d_model, 1, kernel)
    ggml_tensor* conv_bn_w = nullptr;
    ggml_tensor* conv_bn_b = nullptr;
    ggml_tensor* conv_bn_mean = nullptr;
    ggml_tensor* conv_bn_var = nullptr;
    ggml_tensor* conv_down_w = nullptr;
    ggml_tensor* conv_down_b = nullptr;
    ggml_tensor* conv_norm_w = nullptr;
    ggml_tensor* conv_norm_b = nullptr;

    // FFN1 (Macaron pre)
    ggml_tensor* ff1_norm_w = nullptr;
    ggml_tensor* ff1_norm_b = nullptr;
    ggml_tensor* ff1_up_w = nullptr;
    ggml_tensor* ff1_up_b = nullptr;
    ggml_tensor* ff1_down_w = nullptr;
    ggml_tensor* ff1_down_b = nullptr;

    // FFN2 (Macaron post)
    ggml_tensor* ff2_norm_w = nullptr;
    ggml_tensor* ff2_norm_b = nullptr;
    ggml_tensor* ff2_up_w = nullptr;
    ggml_tensor* ff2_up_b = nullptr;
    ggml_tensor* ff2_down_w = nullptr;
    ggml_tensor* ff2_down_b = nullptr;

    // Post-norm
    ggml_tensor* post_norm_w = nullptr;
    ggml_tensor* post_norm_b = nullptr;
};

struct granite_proj_block {
    // Self-attention
    ggml_tensor *sa_q_w = nullptr, *sa_q_b = nullptr;
    ggml_tensor *sa_k_w = nullptr, *sa_k_b = nullptr;
    ggml_tensor *sa_v_w = nullptr, *sa_v_b = nullptr;
    ggml_tensor *sa_out_w = nullptr, *sa_out_b = nullptr;
    ggml_tensor *sa_norm_w = nullptr, *sa_norm_b = nullptr;

    // Cross-attention
    ggml_tensor *ca_q_w = nullptr, *ca_q_b = nullptr;
    ggml_tensor *ca_k_w = nullptr, *ca_k_b = nullptr;
    ggml_tensor *ca_v_w = nullptr, *ca_v_b = nullptr;
    ggml_tensor *ca_out_w = nullptr, *ca_out_b = nullptr;
    ggml_tensor *ca_norm_w = nullptr, *ca_norm_b = nullptr;

    // FFN
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
};

struct granite_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct granite_speech_model {
    granite_speech_hparams hparams;

    struct {
        ggml_tensor* input_w = nullptr;
        ggml_tensor* input_b = nullptr;
        ggml_tensor* mel_filters = nullptr;
        ggml_tensor* mel_window = nullptr;
        // Mid-CTC residual (applied after layer 8)
        ggml_tensor* ctc_out_w = nullptr; // (1024, 348)
        ggml_tensor* ctc_out_b = nullptr; // (348,)
        ggml_tensor* ctc_mid_w = nullptr; // (348, 1024)
        ggml_tensor* ctc_mid_b = nullptr; // (1024,)
        std::vector<granite_enc_block> blocks;
    } encoder;

    struct {
        ggml_tensor* query = nullptr; // (1, n_query, d_model)
        ggml_tensor* ln_w = nullptr;
        ggml_tensor* ln_b = nullptr;
        ggml_tensor* linear_w = nullptr; // (llm_d, proj_d)
        ggml_tensor* linear_b = nullptr;
        std::vector<granite_proj_block> blocks;
    } projector;

    struct {
        ggml_tensor* token_embd_w = nullptr;
        ggml_tensor* output_norm_w = nullptr;
        ggml_tensor* output_w = nullptr; // separate lm_head (not tied)
        std::vector<granite_llm_block> blocks;
    } llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a: optional second buffer for layers spilled to CPU.
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct granite_speech_context {
    granite_speech_context_params params;
    granite_speech_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;

    int n_threads = 4;

    // Precomputed relative position embedding lookup, per encoder block.
    // Layout: rpe_per_layer[il][c * C * hd + r * hd + d] = rel_pos_emb(attention_dists[c][r])[d]
    // granite-speech-4.1-2b stores DIFFERENT attn_rel_pos.weight per encoder
    // block — they are NOT tied across layers, despite older comments in the
    // tree. The graph encoder used to reuse layer 0's table for all 16 blocks
    // and silently dropped half the JFK transcript (PLAN #16).
    std::vector<std::vector<float>> rpe_per_layer;

    // For granite-speech-4.1-2b-plus: 0-indexed encoder layer indices whose
    // hidden states are concatenated (along the feature dim) with the final
    // encoder output before the projector. Empty for base / 4.0 / 4.1 base.
    // Plus = {3}. Convention: index N means "the output of encoder block N"
    // (after `il == N` in the per-layer loop).
    std::vector<int> proj_cat_layers_parsed;

    // Tokenizer (GPT-2-style byte-level BPE).
    //
    // id_to_token holds the byte-encoded unicode form of each vocab
    // entry, used for detokenization (granite_speech_decode_tokens).
    //
    // token_to_id is the reverse map, populated alongside id_to_token
    // at load time so granite_speech_tokenize() can do constant-time
    // lookups.
    //
    // merge_rank is loaded from `tokenizer.ggml.merges` if the GGUF
    // includes it (newer converter writes it; older GGUFs don't).
    // Empty merge_rank means tokenize() can only handle text that
    // already maps to a single vocab entry — fine for the hardcoded
    // prompt fragments, but not for arbitrary user text.
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank; // "left right" -> rank
};

// ===========================================================================
// GGUF loader helpers
// ===========================================================================

// Loader helpers moved to src/core/gguf_loader.

// ===========================================================================
// Model loading
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static bool granite_speech_load_model(granite_speech_model& model, const char* path, ggml_backend_t backend,
                                      ggml_backend_t backend_cpu) {
    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g)
            return false;
        auto& hp = model.hparams;

        hp.enc_n_layers = core_gguf::kv_u32(g, "granite_speech.enc.n_layers", hp.enc_n_layers);
        hp.enc_d_model = core_gguf::kv_u32(g, "granite_speech.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(g, "granite_speech.enc.n_heads", hp.enc_n_heads);
        hp.enc_head_dim = core_gguf::kv_u32(g, "granite_speech.enc.head_dim", hp.enc_head_dim);
        hp.enc_input_dim = core_gguf::kv_u32(g, "granite_speech.enc.input_dim", hp.enc_input_dim);
        hp.enc_conv_kernel = core_gguf::kv_u32(g, "granite_speech.enc.conv_kernel", hp.enc_conv_kernel);
        hp.enc_ff_dim = core_gguf::kv_u32(g, "granite_speech.enc.ff_dim", hp.enc_ff_dim);
        hp.enc_context_size = core_gguf::kv_u32(g, "granite_speech.enc.context_size", hp.enc_context_size);
        hp.enc_max_pos_emb = core_gguf::kv_u32(g, "granite_speech.enc.max_pos_emb", hp.enc_max_pos_emb);

        hp.proj_n_layers = core_gguf::kv_u32(g, "granite_speech.proj.n_layers", hp.proj_n_layers);
        hp.proj_d_model = core_gguf::kv_u32(g, "granite_speech.proj.d_model", hp.proj_d_model);
        hp.proj_n_heads = core_gguf::kv_u32(g, "granite_speech.proj.n_heads", hp.proj_n_heads);
        hp.proj_ff_dim = core_gguf::kv_u32(g, "granite_speech.proj.ff_dim", hp.proj_ff_dim);
        // Default proj_encoder_hidden_size to proj_d_model so non-plus
        // GGUFs keep the old single-layer-output behaviour.
        hp.proj_encoder_hidden_size = core_gguf::kv_u32(g, "granite_speech.proj.encoder_hidden_size", hp.proj_d_model);
        hp.proj_cat_layers = core_gguf::kv_str(g, "granite_speech.proj.cat_layers", hp.proj_cat_layers.c_str());

        hp.llm_n_layers = core_gguf::kv_u32(g, "granite_speech.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(g, "granite_speech.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(g, "granite_speech.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(g, "granite_speech.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(g, "granite_speech.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(g, "granite_speech.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(g, "granite_speech.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(g, "granite_speech.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(g, "granite_speech.llm.vocab_size", hp.llm_vocab_size);

        hp.embedding_multiplier =
            core_gguf::kv_f32(g, "granite_speech.llm.embedding_multiplier", hp.embedding_multiplier);
        hp.attention_multiplier =
            core_gguf::kv_f32(g, "granite_speech.llm.attention_multiplier", hp.attention_multiplier);
        hp.residual_multiplier = core_gguf::kv_f32(g, "granite_speech.llm.residual_multiplier", hp.residual_multiplier);
        hp.logits_scaling = core_gguf::kv_f32(g, "granite_speech.llm.logits_scaling", hp.logits_scaling);

        hp.downsample_rate = core_gguf::kv_u32(g, "granite_speech.downsample_rate", hp.downsample_rate);
        hp.window_size = core_gguf::kv_u32(g, "granite_speech.window_size", hp.window_size);
        hp.audio_token_index = core_gguf::kv_u32(g, "granite_speech.audio_token_index", hp.audio_token_index);
        hp.eos_token_id = core_gguf::kv_u32(g, "granite_speech.llm.eos_token_id", hp.eos_token_id);
        hp.bos_token_id = core_gguf::kv_u32(g, "granite_speech.llm.bos_token_id", hp.bos_token_id);

        core_gguf::free_metadata(g);
    }

    // Pass 2: tensor data via shared helper.
    // PLAN #69a: when CRISPASR_N_GPU_LAYERS is set and < total layers,
    // route LLM layers [N..total) onto the CPU backend. Encoder blocks
    // (`enc.blk.*`) stay on GPU — the shared predicate only matches
    // `blk.<N>.*` (the LLM naming), treating encoder tensors as
    // non-layered.
    core_gguf::WeightLoad wl;
    int n_gpu_layers_env = -1;
    if (const char* s = std::getenv("CRISPASR_N_GPU_LAYERS")) {
        n_gpu_layers_env = std::atoi(s);
    }
    const int total_layers = (int)model.hparams.llm_n_layers;
    const bool do_split =
        backend_cpu && backend_cpu != backend && n_gpu_layers_env >= 0 && n_gpu_layers_env < total_layers;
    if (do_split) {
        int threshold = n_gpu_layers_env;
        if (!core_gguf::load_weights_split(path, backend, backend_cpu, core_gguf::is_gpu_tensor_blk, &threshold,
                                           "granite_speech", wl)) {
            return false;
        }
        fprintf(stderr, "granite_speech: layer offload: gpu=[0,%d), cpu=[%d,%d) (CRISPASR_N_GPU_LAYERS=%d)\n",
                n_gpu_layers_env, n_gpu_layers_env, total_layers, n_gpu_layers_env);
    } else {
        if (!core_gguf::load_weights(path, backend, "granite_speech", wl)) {
            return false;
        }
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.buf_cpu = wl.buf_cpu;
    model.tensors = std::move(wl.tensors);

    // Bind tensors
    auto get = [&](const std::string& n) -> ggml_tensor* {
        auto it = model.tensors.find(n);
        return it != model.tensors.end() ? it->second : nullptr;
    };
    auto require = [&](const std::string& n) -> ggml_tensor* {
        auto* t = get(n);
        if (!t)
            fprintf(stderr, "granite_speech: missing '%s'\n", n.c_str());
        return t;
    };

    // Encoder
    auto& e = model.encoder;
    e.input_w = require("enc.input.weight");
    e.input_b = require("enc.input.bias");
    e.mel_filters = get("audio.mel_filters");
    e.mel_window = get("audio.mel_window");
    e.ctc_out_w = get("enc.ctc_out.weight");
    e.ctc_out_b = get("enc.ctc_out.bias");
    e.ctc_mid_w = get("enc.ctc_mid.weight");
    e.ctc_mid_b = get("enc.ctc_mid.bias");

    e.blocks.resize(model.hparams.enc_n_layers);
    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto p = "enc.blk." + std::to_string(il) + ".";
        auto& b = e.blocks[il];
        b.attn_norm_w = get(p + "attn_norm.weight");
        b.attn_norm_b = get(p + "attn_norm.bias");
        b.attn_q_w = require(p + "attn_q.weight");
        b.attn_kv_w = require(p + "attn_kv.weight");
        b.attn_out_w = require(p + "attn_out.weight");
        b.attn_out_b = get(p + "attn_out.bias");
        b.attn_rel_pos_w = get(p + "attn_rel_pos.weight");

        b.conv_up_w = get(p + "conv_up.weight");
        b.conv_up_b = get(p + "conv_up.bias");
        b.conv_dw_w = get(p + "conv_dw.weight");
        b.conv_bn_w = get(p + "conv_bn.weight");
        b.conv_bn_b = get(p + "conv_bn.bias");
        b.conv_bn_mean = get(p + "conv_bn.running_mean");
        b.conv_bn_var = get(p + "conv_bn.running_var");
        b.conv_down_w = get(p + "conv_down.weight");
        b.conv_down_b = get(p + "conv_down.bias");
        b.conv_norm_w = get(p + "conv_norm.weight");
        b.conv_norm_b = get(p + "conv_norm.bias");

        b.ff1_norm_w = get(p + "ff1_norm.weight");
        b.ff1_norm_b = get(p + "ff1_norm.bias");
        b.ff1_up_w = require(p + "ff1_up.weight");
        b.ff1_up_b = get(p + "ff1_up.bias");
        b.ff1_down_w = require(p + "ff1_down.weight");
        b.ff1_down_b = get(p + "ff1_down.bias");

        b.ff2_norm_w = get(p + "ff2_norm.weight");
        b.ff2_norm_b = get(p + "ff2_norm.bias");
        b.ff2_up_w = require(p + "ff2_up.weight");
        b.ff2_up_b = get(p + "ff2_up.bias");
        b.ff2_down_w = require(p + "ff2_down.weight");
        b.ff2_down_b = get(p + "ff2_down.bias");

        b.post_norm_w = get(p + "post_norm.weight");
        b.post_norm_b = get(p + "post_norm.bias");
    }

    // Projector
    auto& pr = model.projector;
    pr.query = require("proj.query");
    pr.ln_w = get("proj.ln.weight");
    pr.ln_b = get("proj.ln.bias");
    pr.linear_w = require("proj.linear.weight");
    pr.linear_b = get("proj.linear.bias");

    pr.blocks.resize(model.hparams.proj_n_layers);
    for (uint32_t il = 0; il < model.hparams.proj_n_layers; il++) {
        auto p = "proj.blk." + std::to_string(il) + ".";
        auto& b = pr.blocks[il];
        b.sa_q_w = require(p + "sa_query.weight");
        b.sa_q_b = get(p + "sa_query.bias");
        b.sa_k_w = require(p + "sa_key.weight");
        b.sa_k_b = get(p + "sa_key.bias");
        b.sa_v_w = require(p + "sa_value.weight");
        b.sa_v_b = get(p + "sa_value.bias");
        b.sa_out_w = require(p + "sa_out.weight");
        b.sa_out_b = get(p + "sa_out.bias");
        b.sa_norm_w = get(p + "sa_norm.weight");
        b.sa_norm_b = get(p + "sa_norm.bias");

        b.ca_q_w = require(p + "ca_query.weight");
        b.ca_q_b = get(p + "ca_query.bias");
        b.ca_k_w = require(p + "ca_key.weight");
        b.ca_k_b = get(p + "ca_key.bias");
        b.ca_v_w = require(p + "ca_value.weight");
        b.ca_v_b = get(p + "ca_value.bias");
        b.ca_out_w = require(p + "ca_out.weight");
        b.ca_out_b = get(p + "ca_out.bias");
        b.ca_norm_w = get(p + "ca_norm.weight");
        b.ca_norm_b = get(p + "ca_norm.bias");

        b.ffn_up_w = require(p + "ffn_up.weight");
        b.ffn_up_b = get(p + "ffn_up.bias");
        b.ffn_down_w = require(p + "ffn_down.weight");
        b.ffn_down_b = get(p + "ffn_down.bias");
        b.ffn_norm_w = get(p + "ffn_norm.weight");
        b.ffn_norm_b = get(p + "ffn_norm.bias");
    }

    // LLM
    auto& l = model.llm;
    l.token_embd_w = require("token_embd.weight");
    l.output_norm_w = require("output_norm.weight");
    l.output_w = require("output.weight");

    l.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        auto p = "blk." + std::to_string(il) + ".";
        auto& b = l.blocks[il];
        b.attn_norm_w = require(p + "attn_norm.weight");
        b.attn_q_w = require(p + "attn_q.weight");
        b.attn_k_w = require(p + "attn_k.weight");
        b.attn_v_w = require(p + "attn_v.weight");
        b.attn_out_w = require(p + "attn_output.weight");
        b.ffn_norm_w = require(p + "ffn_norm.weight");
        b.ffn_gate_w = require(p + "ffn_gate.weight");
        b.ffn_up_w = require(p + "ffn_up.weight");
        b.ffn_down_w = require(p + "ffn_down.weight");
    }

    return true;
}

// ===========================================================================
// Public API (stubs — to be implemented)
// ===========================================================================

extern "C" struct granite_speech_context_params granite_speech_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true};
}

// ---- Tokenizer encode side (delegates to src/core/bpe.h) ----

// Pre-split text at Granite-style special-token markers (<|foo|>) so
// they resolve to their single vocab id instead of being byte-encoded
// and BPE-merged. Non-special segments go through tokenize_simple as
// before, preserving the bit-identical path for granite-4.0 prompts
// that don't contain any <|...|> markers.
//
// Markers are picked up from token_to_id: we look for any vocab entry
// matching "<|...|>" and sort them by length descending so overlapping
// markers match the longest form first.
extern "C" int32_t* granite_speech_tokenize(struct granite_speech_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text) {
        if (out_n)
            *out_n = 0;
        return nullptr;
    }
    const std::string input(text);

    // Build the marker list once (lazily, cached on the context would be
    // cleaner, but the cost is negligible — O(vocab) string compares).
    static thread_local std::vector<std::pair<std::string, int32_t>> specials;
    static thread_local const granite_speech_context* specials_for = nullptr;
    if (specials_for != ctx) {
        specials.clear();
        for (const auto& kv : ctx->token_to_id) {
            const std::string& s = kv.first;
            if (s.size() >= 4 && s.front() == '<' && s.back() == '>' && s[1] == '|' && s[s.size() - 2] == '|') {
                specials.emplace_back(s, kv.second);
            }
        }
        std::sort(specials.begin(), specials.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        specials_for = ctx;
    }

    std::vector<int32_t> out_ids;
    size_t i = 0;
    size_t plain_start = 0;

    auto flush_plain = [&](size_t end) {
        if (end <= plain_start)
            return;
        std::string seg = input.substr(plain_start, end - plain_start);
        auto ids = core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, seg);
        out_ids.insert(out_ids.end(), ids.begin(), ids.end());
    };

    while (i < input.size()) {
        // Try to match a special token starting at input[i].
        bool matched = false;
        if (input[i] == '<' && i + 1 < input.size() && input[i + 1] == '|') {
            for (const auto& p : specials) {
                const std::string& s = p.first;
                if (i + s.size() <= input.size() && std::memcmp(input.data() + i, s.data(), s.size()) == 0) {
                    flush_plain(i);
                    out_ids.push_back(p.second);
                    i += s.size();
                    plain_start = i;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched)
            i++;
    }
    flush_plain(i);

    int* out_arr = (int*)malloc(out_ids.size() * sizeof(int));
    if (!out_arr) {
        if (out_n)
            *out_n = 0;
        return nullptr;
    }
    std::memcpy(out_arr, out_ids.data(), out_ids.size() * sizeof(int));
    if (out_n)
        *out_n = (int)out_ids.size();
    return out_arr;
}

extern "C" struct granite_speech_context* granite_speech_init_from_file(const char* path,
                                                                        struct granite_speech_context_params params) {
    auto* ctx = new granite_speech_context();
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

    if (!granite_speech_load_model(ctx->model, path, ctx->backend, ctx->backend_cpu)) {
        delete ctx;
        return nullptr;
    }

    // Load tokenizer vocab + (optional) BPE merges. The vocab is needed
    // for both decode (id->string) and encode (string->id); the merges
    // are only required for tokenizing arbitrary text. When the merges
    // table is missing (older GGUF written before the encoder existed)
    // granite_speech_tokenize() falls back to single-token vocab
    // lookups, which is enough for the kPrefix/kSuffix integer-ID
    // path used by the default transcribe prompt but not for runtime
    // user text like a translate instruction.
    {
        gguf_init_params mp = {true, nullptr};
        gguf_context* g = gguf_init_from_file(path, mp);
        if (g) {
            int ki = gguf_find_key(g, "tokenizer.ggml.tokens");
            if (ki >= 0) {
                int n = gguf_get_arr_n(g, ki);
                ctx->id_to_token.resize(n);
                ctx->token_to_id.reserve((size_t)n);
                for (int i = 0; i < n; i++) {
                    std::string tok = gguf_get_arr_str(g, ki, i);
                    ctx->id_to_token[i] = tok;
                    ctx->token_to_id.emplace(std::move(tok), i);
                }
                if (params.verbosity >= 1)
                    fprintf(stderr, "granite_speech: loaded %d vocab tokens\n", n);
            }
            int mi = gguf_find_key(g, "tokenizer.ggml.merges");
            if (mi >= 0) {
                int n = gguf_get_arr_n(g, mi);
                ctx->merge_rank.reserve((size_t)n);
                for (int i = 0; i < n; i++) {
                    ctx->merge_rank[gguf_get_arr_str(g, mi, i)] = i;
                }
                if (params.verbosity >= 1)
                    fprintf(stderr, "granite_speech: loaded %d BPE merges\n", n);
            }
            gguf_free(g);
        }
    }

    // Parse cat_layers ("3,7,11" or empty) into vector<int> for the
    // PLUS variant's encoder layer concatenation feature.
    {
        const std::string& s = ctx->model.hparams.proj_cat_layers;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos)
                j = s.size();
            std::string tok = s.substr(i, j - i);
            // strip whitespace
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
                tok.pop_back();
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
                tok.erase(0, 1);
            if (!tok.empty()) {
                try {
                    ctx->proj_cat_layers_parsed.push_back(std::stoi(tok));
                } catch (...) {
                    fprintf(stderr, "granite_speech: ignoring unparseable cat_layers entry '%s'\n", tok.c_str());
                }
            }
            i = j + 1;
        }
        if (!ctx->proj_cat_layers_parsed.empty() && params.verbosity >= 1) {
            fprintf(stderr, "granite_speech: PLUS variant — concatenating encoder layers [");
            for (size_t k = 0; k < ctx->proj_cat_layers_parsed.size(); k++)
                fprintf(stderr, "%s%d", k > 0 ? "," : "", ctx->proj_cat_layers_parsed[k]);
            fprintf(stderr, "] + final → projector input width = %u\n", ctx->model.hparams.proj_encoder_hidden_size);
        }
    }

    // Fold batch norm into scale+shift tensors (load-time, once)
    // BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    //       = x * scale + shift
    // where scale = gamma/sqrt(var+eps), shift = beta - mean*scale
    {
        const float eps = 1e-5f;
        const int inner = 2 * (int)ctx->model.hparams.enc_d_model; // conv expansion = 2 * d_model
        int folded = 0;
        for (uint32_t il = 0; il < ctx->model.hparams.enc_n_layers; il++) {
            auto& b = ctx->model.encoder.blocks[il];
            if (!b.conv_bn_w || !b.conv_bn_b || !b.conv_bn_mean || !b.conv_bn_var)
                continue;

            std::vector<float> gamma(inner), beta(inner), mean(inner), var(inner);
            ggml_backend_tensor_get(b.conv_bn_w, gamma.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_get(b.conv_bn_b, beta.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_get(b.conv_bn_mean, mean.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_get(b.conv_bn_var, var.data(), 0, inner * sizeof(float));

            std::vector<float> scale(inner), shift(inner);
            for (int c = 0; c < inner; c++) {
                scale[c] = gamma[c] / std::sqrt(var[c] + eps);
                shift[c] = beta[c] - mean[c] * scale[c];
            }

            // Write precomputed scale/shift back into bn_w/bn_b tensors
            ggml_backend_tensor_set(b.conv_bn_w, scale.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_set(b.conv_bn_b, shift.data(), 0, inner * sizeof(float));
            folded++;
        }
        if (params.verbosity >= 1)
            fprintf(stderr, "granite_speech: BN folded for %d encoder layers\n", folded);
    }

    // Precompute Shaw RPE lookup table per layer (built via
    // core_conformer_ibm::build_shaw_rpe_lookup; shared with granite_nle).
    // granite-speech-4.1-2b stores DIFFERENT attn_rel_pos.weight per encoder
    // block — they are NOT tied across layers. The earlier "tied" assumption
    // silently used layer 0's RPE for all 16 layers in the graph encoder
    // path, which is what regressed JFK on `GRANITE_ENCODER_GRAPH=1` (PLAN #16).
    {
        const int C = (int)ctx->model.hparams.enc_context_size;
        const int max_pos = (int)ctx->model.hparams.enc_max_pos_emb;
        const int hd = (int)ctx->model.hparams.enc_head_dim;
        const int n_layers = (int)ctx->model.hparams.enc_n_layers;
        ctx->rpe_per_layer.assign(n_layers, {});
        int built = 0;
        for (int il = 0; il < n_layers; il++) {
            ggml_tensor* rpe_w = ctx->model.encoder.blocks[il].attn_rel_pos_w;
            if (!rpe_w)
                continue;
            if (!core_conformer_ibm::build_shaw_rpe_lookup(rpe_w, C, hd, max_pos, ctx->rpe_per_layer[il])) {
                fprintf(stderr, "granite_speech: unsupported RPE type %s at layer %d — skipping\n",
                        ggml_type_name(rpe_w->type), il);
                ctx->rpe_per_layer[il].clear();
            } else {
                built++;
            }
        }
        if (params.verbosity >= 1)
            fprintf(stderr, "granite_speech: RPE lookups precomputed (%d × %d × %d, %d/%d layers)\n", C, C, hd, built,
                    n_layers);
    }

    // Create scheduler
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
        const auto& hp = ctx->model.hparams;
        fprintf(stderr, "granite_speech: loaded %s (enc %u layers, proj %u layers, llm %u layers, vocab %u)\n", path,
                hp.enc_n_layers, hp.proj_n_layers, hp.llm_n_layers, hp.llm_vocab_size);
    }
    return ctx;
}

extern "C" void granite_speech_free(struct granite_speech_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.buf_cpu)
        ggml_backend_buffer_free(ctx->model.buf_cpu);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Mel spectrogram (80 bins, n_fft=512, hop=160, per-utterance max norm).
// FFT lives in core/fft.h (shared with granite_nle).
// ===========================================================================

extern "C" float* granite_speech_compute_mel(struct granite_speech_context* ctx, const float* samples, int n_samples,
                                             int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    granite_bench_stage _b("compute_mel");
    const int n_fft = 512, win_length = 400, hop = 160, n_mels = 80, n_freqs = n_fft / 2 + 1;

    // Load mel filters from GGUF (shape: [n_freqs, n_mels], HF layout).
    std::vector<float> filt((size_t)n_freqs * n_mels);
    if (!ctx->model.encoder.mel_filters)
        return nullptr;
    ggml_backend_tensor_get(ctx->model.encoder.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // Hann window: win_length=400 samples, synthesized here (granite
    // doesn't ship a window tensor in the GGUF like the other models).
    // core_mel::compute() handles the center-pad from win_length to n_fft
    // internally, so we only construct the win_length-sized version.
    std::vector<float> hann((size_t)win_length);
    for (int i = 0; i < win_length; i++) {
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / win_length));
    }

    // HF / Whisper cluster parameters, NOT dropping the last STFT frame
    // (torchaudio's MelSpectrogram keeps it; granite's PyTorch reference
    // does the same). Output layout is TimeMels so the per-frame stacking
    // below can use plain std::memcpy on consecutive rows.
    //
    // Granite's original normalization was `v / 4.0 + 1.0`, which is
    // mathematically identical to core_mel's GlobalClipMax
    // `(v + 4.0) / 4.0`. No new knob needed.
    // HF / Whisper cluster parameters, NOT dropping the last STFT frame
    // (torchaudio's MelSpectrogram keeps it; granite's PyTorch reference
    // does the same). stacked_frames=2 folds consecutive pairs of 80-mel
    // frames into a single 160-column row; trailing odd frames are
    // dropped automatically by core_mel::compute(). The encoder receives
    // (T_stacked, 160) directly — no post-processing.
    //
    // Granite's original normalization was `v / 4.0 + 1.0`, which is
    // mathematically identical to core_mel's GlobalClipMax
    // `(v + 4.0) / 4.0`. No new knob needed.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win_length;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Log10;
    p.log_guard = core_mel::LogGuard::MaxClip;
    p.norm = core_mel::Normalization::GlobalClipMax;
    p.layout = core_mel::Layout::TimeMels;
    p.fb_layout = core_mel::FbLayout::FreqsMels;
    p.matmul = core_mel::MatmulPrecision::Double;
    p.log_eps = 1e-10f;
    p.center_pad = true;
    p.drop_last_frame = false;
    p.stacked_frames = 2;

    int T_stacked = 0;
    auto stacked = core_mel::compute(samples, n_samples, hann.data(), win_length, filt.data(), n_freqs,
                                     core_fft::fft_radix2_wrapper, p, T_stacked);

    if (stacked.empty())
        return nullptr;

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (size_t i = 0; i < stacked.size(); i++) {
            if (stacked[i] < mn)
                mn = stacked[i];
            if (stacked[i] > mx)
                mx = stacked[i];
            sum += stacked[i];
        }
        fprintf(stderr, "  mel: (%d, 160) min=%.4f max=%.4f mean=%.6f\n", T_stacked, mn, mx, sum / stacked.size());
    }

    if (out_n_mels)
        *out_n_mels = 160;
    if (out_T_mel)
        *out_T_mel = T_stacked;

    float* result = (float*)malloc(stacked.size() * sizeof(float));
    std::memcpy(result, stacked.data(), stacked.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Encoder, Projector, LLM — stubs (to be implemented in next step)
// ===========================================================================

// ===========================================================================
// Depthwise conv helper (CPU, applied between ggml graph segments)
// Each channel c has its own K-tap filter: out[c,t] = sum_k w[k,c] * in[c,t+k-pad]
// ===========================================================================

static void depthwise_conv_1d_cpu(float* out, const float* in, const float* weight, int channels, int T,
                                  int kernel_size, int pad) {
    for (int c = 0; c < channels; c++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int ti = t + k - pad;
                if (ti >= 0 && ti < T) {
                    // weight layout from GGUF: ne[0]=K, ne[1]=1, ne[2]=C
                    // Data: weight[k + 0*K + c*1*K] = weight[k + c*K]
                    // But GGUF ne ordering: element at [k, 0, c] = data[c * 1 * K + 0 * K + k] = data[c*K + k]
                    sum += weight[(size_t)c * kernel_size + k] * in[(size_t)c * T + ti];
                }
            }
            out[(size_t)c * T + t] = sum;
        }
    }
}

// Block-local Shaw RPE attention lives in core/conformer_ibm.h
// (shared with granite_nle).

// ===========================================================================
// CPU-based linear layer helper (builds tiny ggml graph per matmul)
// ===========================================================================

// Apply: out = W @ x + b, where x is (d_in, T), W is (d_in, d_out), out is (d_out, T)
// ctx is kept in the signature for symmetry with the graph-building
// helpers in this file; the body pulls W/bias via ggml_backend_tensor_get
// which doesn't need the context.
static void cpu_linear(granite_speech_context* /*ctx*/, float* out, const float* x, ggml_tensor* W, ggml_tensor* bias,
                       int d_in, int d_out, int T) {
    // Simple CPU matmul (no ggml graph overhead for small ops)
    // W in ggml: ne[0]=d_in, ne[1]=d_out. W[i,j] = data[j * d_in + i]
    // ggml_mul_mat(W, x) computes W^T @ x
    // For F32 weights, do it directly:
    std::vector<float> w_f32((size_t)d_in * d_out);
    ggml_backend_tensor_get(W, w_f32.data(), 0, w_f32.size() * sizeof(float));

    std::vector<float> b_f32;
    if (bias) {
        b_f32.resize(d_out);
        ggml_backend_tensor_get(bias, b_f32.data(), 0, d_out * sizeof(float));
    }

    // out[o, t] = sum_i W[i, o] * x[i, t] + bias[o]
    // = sum_i w_f32[o * d_in + i] * x[i + t * d_in] + b[o]
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < d_out; o++) {
            float sum = 0.0f;
            for (int i = 0; i < d_in; i++)
                sum += w_f32[(size_t)o * d_in + i] * x[(size_t)i + (size_t)t * d_in];
            if (bias)
                sum += b_f32[o];
            out[(size_t)o + (size_t)t * d_out] = sum;
        }
    }
}

// ===========================================================================
// Conformer encoder — CPU-based forward (not ggml graph)
//
// The encoder uses block-local attention (context_size=200) with Shaw
// relative position embeddings. This is hard to express as a single ggml
// graph, so we implement the encoder as a hybrid: ggml graphs for the
// per-block matmuls, and CPU loops for the block chunking and relative
// position logic.
//
// For simplicity in V1, we use a simplified global attention encoder
// (same as before) and note that block-local attention + rel pos +
// depthwise conv are needed for accuracy. These will be added when
// we have ground truth to compare against.
// ===========================================================================

// NOTE: This function is the legacy "global attention" encoder path
// retained as a reference implementation. The live granite path uses the
// hybrid CPU-loop encoder at granite_speech_run_encoder() below with
// Shaw-style block-local attention. This builder is currently not called
// from any caller; it stays in tree as a simpler baseline we can diff
// against if the CPU path ever drifts. The unused-local casts below
// keep the file warning-free without deleting the reference.
static ggml_cgraph* granite_build_encoder(granite_speech_context* ctx, int T, bool with_rpe) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.enc_d_model;       // 1024
    const int n_heads = (int)hp.enc_n_heads; // 8
    const int hd = (int)hp.enc_head_dim;     // 128
    const int ff = (int)hp.enc_ff_dim;       // 4096
    const int n_layers = (int)hp.enc_n_layers;
    const int input_dim = (int)hp.enc_input_dim; // 160
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    (void)d;
    (void)ff; // reserved for future use (see header comment)

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: (T, input_dim=160) stacked mel frames
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, input_dim, T);
    ggml_set_name(inp, "enc_input");
    ggml_set_input(inp);

    // Input linear: (160 → 1024)
    ggml_tensor* cur = ggml_mul_mat(ctx0, m.encoder.input_w, inp);
    if (m.encoder.input_b)
        cur = ggml_add(ctx0, cur, m.encoder.input_b);

    // PLUS variant: collect post-norm hidden states from configured cat_layers.
    // HF convention: index N = output of encoder block N-1. Index 0 means the
    // input embedding (this point in the graph). The captured tensors stay
    // alive because they're sources of the final ggml_concat below — fanout
    // off the residual stream, no extra compute.
    const auto& cat_idx = ctx->proj_cat_layers_parsed;
    const bool do_cat_concat = !cat_idx.empty();
    std::vector<ggml_tensor*> cat_taps(cat_idx.size(), nullptr);
    for (size_t k = 0; k < cat_idx.size(); k++) {
        if (cat_idx[k] == 0)
            cat_taps[k] = cur;
    }

    // Block-local attention with Shaw relative position embeddings
    const int ctx_size = (int)hp.enc_context_size;
    const int n_blocks_attn = (T + ctx_size - 1) / ctx_size;
    const int T_padded = n_blocks_attn * ctx_size;
    (void)T_padded; // reserved — the block path would clip-pad here

    // Block-diagonal mask: (T, T) F16. Only used by the approximate
    // (with_rpe=false) flash-attn path; the per-block subgraph path
    // doesn't need it because block boundaries are explicit.
    ggml_tensor* block_mask = nullptr;
    if (!with_rpe) {
        block_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        ggml_set_name(block_mask, "block_mask");
        ggml_set_input(block_mask);
    }

    // RPE lookup as input: stacked across layers.
    // Shape: (ctx_size * head_dim, ctx_size, n_layers) F32.
    // Memory layout per layer: rpe[k=(r*hd + d), c]; element flat-offset =
    // (r*hd + d) + (ctx_size*hd) * c. That matches the CPU vector layout
    // cpu_rpe[c * ctx_size * hd + r * hd + d] so each layer's upload is a
    // direct memcpy at offset il * (ctx_size * hd * ctx_size).
    // Reshaping to ne=[hd, ctx_size, ctx_size, n_layers] gives axes
    // [d, r, c, layer]; per-layer slicing uses ggml_view_3d on the layer dim.
    ggml_tensor* rpe_tensor = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ctx_size * hd, ctx_size, with_rpe ? n_layers : 1);
    ggml_set_name(rpe_tensor, "rpe_lookup");
    ggml_set_input(rpe_tensor);

    ggml_tensor* rpe_4d = with_rpe ? ggml_reshape_4d(ctx0, rpe_tensor, hd, ctx_size, ctx_size, n_layers) : nullptr;

    // 16 × Conformer blocks
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.encoder.blocks[il];

        // --- FFN1 (Macaron half-step) ---
        {
            ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.ff1_norm_w)
                x = ggml_mul(ctx0, x, b.ff1_norm_w);
            if (b.ff1_norm_b)
                x = ggml_add(ctx0, x, b.ff1_norm_b);
            x = ggml_mul_mat(ctx0, b.ff1_up_w, x);
            if (b.ff1_up_b)
                x = ggml_add(ctx0, x, b.ff1_up_b);
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, b.ff1_down_w, x);
            if (b.ff1_down_b)
                x = ggml_add(ctx0, x, b.ff1_down_b);
            cur = ggml_add(ctx0, cur, ggml_scale(ctx0, x, 0.5f)); // half-step residual
        }

        // --- MHSA (Block-local attention, context_size=200, Shaw rel pos) ---
        {
            ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.attn_norm_w)
                x = ggml_mul(ctx0, x, b.attn_norm_w);
            if (b.attn_norm_b)
                x = ggml_add(ctx0, x, b.attn_norm_b);

            // Q: (d → d), KV: (d → 2d) combined
            ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, x);   // (d, T)
            ggml_tensor* KV = ggml_mul_mat(ctx0, b.attn_kv_w, x); // (2*d, T)

            // Split KV
            int kv_dim = n_heads * hd;
            ggml_tensor* K = ggml_cont(ctx0, ggml_view_2d(ctx0, KV, kv_dim, T, KV->nb[1], 0));
            ggml_tensor* V =
                ggml_cont(ctx0, ggml_view_2d(ctx0, KV, kv_dim, T, KV->nb[1], kv_dim * ggml_type_size(KV->type)));

            // Reshape to (hd, nh, T) then permute to (hd, T, nh)
            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, T);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, T);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, T);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            ggml_tensor* attn = nullptr;
            if (with_rpe) {
                // Per-block manual attention with Shaw RPE bias. Mirrors the
                // CPU path in core_conformer_ibm::shaw_block_attention_cpu —
                // bit-identical math, but emitted as ggml ops so the whole
                // 16-layer encoder stays a single graph dispatch. For each
                // block we compute scores = Q·K^T + Q·RPE, softmax, then ·V.
                //
                // This layer's RPE = a (hd, ctx_size, ctx_size) view into the
                // stacked rpe_4d on the layer (4th) axis.
                ggml_tensor* rpe_3d_il = ggml_view_3d(ctx0, rpe_4d, hd, ctx_size, ctx_size, rpe_4d->nb[1],
                                                      rpe_4d->nb[2], (size_t)il * rpe_4d->nb[3]);

                std::vector<ggml_tensor*> block_outs;
                block_outs.reserve(n_blocks_attn);
                for (int blk = 0; blk < n_blocks_attn; blk++) {
                    const int blk_start = blk * ctx_size;
                    const int blk_len = std::min(ctx_size, T - blk_start);

                    // Slice Q/K/V over the time axis [blk_start, blk_start+blk_len).
                    ggml_tensor* Q_blk = ggml_cont(ctx0, ggml_view_3d(ctx0, Q, hd, blk_len, n_heads, Q->nb[1], Q->nb[2],
                                                                      (size_t)blk_start * Q->nb[1]));
                    ggml_tensor* K_blk = ggml_cont(ctx0, ggml_view_3d(ctx0, K, hd, blk_len, n_heads, K->nb[1], K->nb[2],
                                                                      (size_t)blk_start * K->nb[1]));
                    ggml_tensor* V_blk = ggml_cont(ctx0, ggml_view_3d(ctx0, V, hd, blk_len, n_heads, V->nb[1], V->nb[2],
                                                                      (size_t)blk_start * V->nb[1]));

                    // QK^T: (blk_len_r=K, blk_len_c=Q, nh)
                    ggml_tensor* scores = ggml_mul_mat(ctx0, K_blk, Q_blk);

                    // Q·RPE: vectorised across heads via 4D broadcast matmul.
                    //   A = rpe_blk_4d : (hd, blk_len_r, blk_len_c, 1)
                    //   B = Q_blk_4d   : (hd, 1,         blk_len_c, n_heads)
                    //   out            : (blk_len_r, 1, blk_len_c, n_heads)
                    // Broadcasting in batch dim 0 (1 → n_heads).
                    ggml_tensor* rpe_blk = ggml_cont(ctx0, ggml_view_3d(ctx0, rpe_3d_il, hd, blk_len, blk_len,
                                                                        rpe_3d_il->nb[1], rpe_3d_il->nb[2], 0));
                    ggml_tensor* rpe_blk_4d = ggml_reshape_4d(ctx0, rpe_blk, hd, blk_len, blk_len, 1);
                    ggml_tensor* Q_blk_4d = ggml_reshape_4d(ctx0, Q_blk, hd, 1, blk_len, n_heads);
                    ggml_tensor* pos_bias_4d = ggml_mul_mat(ctx0, rpe_blk_4d, Q_blk_4d);
                    ggml_tensor* pos_bias = ggml_reshape_3d(ctx0, pos_bias_4d, blk_len, blk_len, n_heads);

                    scores = ggml_add(ctx0, scores, pos_bias);
                    scores = ggml_soft_max_ext(ctx0, scores, /*mask*/ nullptr, attn_scale, 0.0f);

                    // attn_blk = V_blk^T @ scores → (hd, blk_len_c, n_heads).
                    // ggml_mul_mat contracts ne[0]; we permute V_blk to put
                    // blk_len_r on ne[0] so the contraction is over r.
                    ggml_tensor* V_blk_T = ggml_cont(ctx0, ggml_permute(ctx0, V_blk, 1, 0, 2, 3));
                    ggml_tensor* out_blk = ggml_mul_mat(ctx0, V_blk_T, scores);
                    block_outs.push_back(out_blk);
                }

                // Concatenate block outputs along the time axis (ne[1]) →
                // [hd, T, n_heads]. Then permute axes 1 and 2 to land at
                // [hd, n_heads, T], which matches `ggml_flash_attn_ext`'s
                // output convention so the downstream reshape + attn_out_w
                // matmul is bit-identical to the no-RPE path.
                attn = block_outs[0];
                for (size_t i = 1; i < block_outs.size(); i++)
                    attn = ggml_concat(ctx0, attn, block_outs[i], 1);
                attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
                attn = ggml_reshape_2d(ctx0, attn, n_heads * hd, T);
            } else {
                // Approximate (no RPE) flash-attn path with block-diagonal mask.
                // Kept available via GRANITE_ENCODER_GRAPH=1 (without
                // GRANITE_ENCODER_GRAPH_RPE) for A/B comparison.
                attn = ggml_flash_attn_ext(ctx0, Q, K, V, block_mask, attn_scale, 0.0f, 0.0f);
                attn = ggml_reshape_2d(ctx0, attn, n_heads * hd, T);
            }

            // Output projection
            attn = ggml_mul_mat(ctx0, b.attn_out_w, attn);
            if (b.attn_out_b)
                attn = ggml_add(ctx0, attn, b.attn_out_b);
            cur = ggml_add(ctx0, cur, attn);
        }

        // --- Conv module ---
        // LayerNorm → pointwise up → GLU → depthwise conv → BN → SiLU → pointwise down
        {
            // LayerNorm at the start (the HF ConformerConvModule.norm)
            ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.conv_norm_w)
                x = ggml_mul(ctx0, x, b.conv_norm_w);
            if (b.conv_norm_b)
                x = ggml_add(ctx0, x, b.conv_norm_b);
            if (b.conv_up_w) {
                // conv_up_w ne=(1, 1024, 4096) — 1×1 conv, reshape to (1024, 4096) for matmul
                int in_ch = (int)b.conv_up_w->ne[1];
                int out_ch = (int)b.conv_up_w->ne[2];
                x = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, b.conv_up_w, in_ch, out_ch), x);
                if (b.conv_up_b)
                    x = ggml_add(ctx0, x, b.conv_up_b);
            }

            // GLU: split into two halves, first half * sigmoid(second half)
            int half_dim = (int)x->ne[0] / 2;
            ggml_tensor* x1 = ggml_cont(ctx0, ggml_view_2d(ctx0, x, half_dim, T, x->nb[1], 0));
            ggml_tensor* x2 =
                ggml_cont(ctx0, ggml_view_2d(ctx0, x, half_dim, T, x->nb[1], half_dim * ggml_type_size(x->type)));
            x = ggml_mul(ctx0, x1, ggml_sigmoid(ctx0, x2));

            // Depthwise conv (kernel=15, groups=2048, pad=7) + batch norm + SiLU
            // conv_dw_w ne=(15, 1, 2048) — each of 2048 channels has its own 15-tap filter
            // ggml doesn't support grouped conv, so we use ggml_conv_1d per-channel
            // by treating x as (T, 2048) and convolving with (15, 1, 2048) with pad=7
            //
            // For a depthwise conv: output[c, t] = sum_k weight[k, 0, c] * input[c, t+k-pad]
            // This is equivalent to a standard conv with C_in=1 applied independently per channel.
            // ggml_conv_1d(kernel=(15, 1, 2048), input=(T, 2048)) should work IF ggml
            // treats ne[1]=1 as the input channel dim.
            //
            // Actually, ggml_conv_1d does: output = kernel^T @ im2col(input)
            // With kernel (K=15, C_in=1, C_out=2048) and input (T, C_in=2048):
            //   This is a 15-tap conv with 2048 input channels and 2048 output channels,
            //   NOT depthwise. It would produce cross-channel mixing.
            //
            // For true depthwise, we'd need to loop over channels or use a diagonal weight.
            // WORKAROUND: skip the depthwise conv and just apply batch_norm + SiLU.
            // The batch norm at least preserves the per-channel statistics.
            // Batch norm (applied to x of shape (inner_dim=2048, T)):
            // BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
            //       = x * scale + shift
            // where scale = gamma / sqrt(var + eps), shift = beta - gamma * mean / sqrt(var + eps)
            // These 1D (2048,) tensors broadcast over T via ggml_mul/ggml_add
            // Depthwise conv (kernel=15, groups=inner_dim, pad=7)
            // Uses ggml_conv_2d_dw_direct (same pattern as Parakeet encoder)
            // Re-enabled for testing — using same pattern as Parakeet
            if (b.conv_dw_w) {
                int K = (int)b.conv_dw_w->ne[0]; // 15
                int inner = half_dim;            // 2048
                int dw_pad = K / 2;              // 7
                ggml_tensor* dw_w = ggml_cast(ctx0, b.conv_dw_w, GGML_TYPE_F32);
                ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w, K, 1, 1, inner);
                ggml_tensor* x_t = ggml_cont(ctx0, ggml_transpose(ctx0, x));
                x_t = ggml_reshape_4d(ctx0, x_t, T, 1, inner, 1);
                x_t = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, x_t, 1, 1, dw_pad, 0, 1, 1);
                x = ggml_cont(ctx0, ggml_permute(ctx0, x_t, 1, 2, 0, 3));
                x = ggml_reshape_2d(ctx0, x, inner, T);
            }

            // BN (precomputed at load time): bn_w = scale, bn_b = shift
            if (b.conv_bn_w && b.conv_bn_b) {
                x = ggml_mul(ctx0, x, b.conv_bn_w);
                x = ggml_add(ctx0, x, b.conv_bn_b);
            }
            x = ggml_silu(ctx0, x);

            // Pointwise down: conv_down_w ne=(1, 2048, 1024)
            if (b.conv_down_w) {
                int in_ch = (int)b.conv_down_w->ne[1];
                int out_ch = (int)b.conv_down_w->ne[2];
                x = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, b.conv_down_w, in_ch, out_ch), x);
                if (b.conv_down_b)
                    x = ggml_add(ctx0, x, b.conv_down_b);
            }

            cur = ggml_add(ctx0, cur, x);
        }

        // --- FFN2 (Macaron half-step) ---
        {
            ggml_tensor* x = ggml_norm(ctx0, cur, 1e-5f);
            if (b.ff2_norm_w)
                x = ggml_mul(ctx0, x, b.ff2_norm_w);
            if (b.ff2_norm_b)
                x = ggml_add(ctx0, x, b.ff2_norm_b);
            x = ggml_mul_mat(ctx0, b.ff2_up_w, x);
            if (b.ff2_up_b)
                x = ggml_add(ctx0, x, b.ff2_up_b);
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, b.ff2_down_w, x);
            if (b.ff2_down_b)
                x = ggml_add(ctx0, x, b.ff2_down_b);
            cur = ggml_add(ctx0, cur, ggml_scale(ctx0, x, 0.5f));
        }

        // --- Post LayerNorm ---
        {
            cur = ggml_norm(ctx0, cur, 1e-5f);
            if (b.post_norm_w)
                cur = ggml_mul(ctx0, cur, b.post_norm_w);
            if (b.post_norm_b)
                cur = ggml_add(ctx0, cur, b.post_norm_b);
        }

        // PLUS: snapshot the post-norm hidden state for any cat_layer index
        // that maps to this block. HF index il+1 = output of block il.
        // Captured BEFORE the mid-CTC residual so it matches HF's
        // output_hidden_states tuple (CPU loop does the same).
        for (size_t k = 0; k < cat_idx.size(); k++) {
            if (cat_idx[k] == il + 1)
                cat_taps[k] = cur;
        }

        // Mid-CTC residual at layer 8 (after 8th layer = index 7)
        if (il == n_layers / 2 - 1 && m.encoder.ctc_out_w && m.encoder.ctc_mid_w) {
            // out: (d → ctc_dim) → softmax → out_mid: (ctc_dim → d) → add to hidden
            ggml_tensor* mid = ggml_mul_mat(ctx0, m.encoder.ctc_out_w, cur);
            if (m.encoder.ctc_out_b)
                mid = ggml_add(ctx0, mid, m.encoder.ctc_out_b);
            mid = ggml_soft_max(ctx0, mid); // softmax over last dim (348)
            mid = ggml_mul_mat(ctx0, m.encoder.ctc_mid_w, mid);
            if (m.encoder.ctc_mid_b)
                mid = ggml_add(ctx0, mid, m.encoder.ctc_mid_b);
            cur = ggml_add(ctx0, cur, mid);
        }
    }

    // PLUS: concat captured cat_layer hiddens + final encoder output along
    // the feature dim (ne[0]). Layout per frame: [cat_0, cat_1, ..., final],
    // each (d) wide → total (n_cat + 1) * d. Matches the projector's
    // expected input width and the CPU loop's wide buffer layout.
    if (do_cat_concat) {
        ggml_tensor* wide = nullptr;
        for (size_t k = 0; k < cat_taps.size(); k++) {
            ggml_tensor* t = cat_taps[k] ? cat_taps[k] : cur; // safety: shouldn't happen
            wide = wide ? ggml_concat(ctx0, wide, t, 0) : t;
        }
        wide = ggml_concat(ctx0, wide, cur, 0);
        cur = wide;
    }

    ggml_set_name(cur, "enc_output");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Run the full encoder as a single ggml compute graph (replaces the manual
// per-op CPU loop path). Currently uses flash_attn_ext with a block-diagonal
// mask but omits Shaw relative position embeddings — this produces slightly
// different output from the CPU path. The main benefit is that the graph
// automatically uses ggml's optimised matmul/conv kernels and can run on
// GPU via the scheduler.
static float* granite_run_encoder_graph(granite_speech_context* ctx, const float* mel, int n_mels, int T, int d,
                                        int* out_dim) {
    (void)d;
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.enc_head_dim; // 128
    const int ctx_size = (int)hp.enc_context_size;

    // Default: per-block manual-attention path with Shaw RPE bias
    // (bit-identical to the CPU loop). Falls back to flash_attn_ext without
    // RPE only when rpe_per_layer is missing entries — that path is
    // numerically approximate and is kept just so the encoder still runs on
    // models with an unsupported RPE weight type.
    const int n_layers = (int)ctx->model.hparams.enc_n_layers;
    bool with_rpe = (int)ctx->rpe_per_layer.size() == n_layers;
    for (int il = 0; il < n_layers && with_rpe; il++) {
        if (ctx->rpe_per_layer[il].empty())
            with_rpe = false;
    }
    if (!with_rpe)
        fprintf(stderr, "granite_encoder_graph: rpe_per_layer missing entries — using approximate no-RPE path\n");

    ggml_cgraph* gf = granite_build_encoder(ctx, T, with_rpe);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "granite_encoder_graph: alloc failed\n");
        return nullptr;
    }

    // Fill input mel
    ggml_tensor* inp = ggml_graph_get_tensor(gf, "enc_input");
    ggml_backend_tensor_set(inp, mel, 0, (size_t)n_mels * T * sizeof(float));

    // Fill block-diagonal attention mask: only present in the no-RPE path.
    // Within each ctx_size block = 0 (attend), across blocks = -inf (masked).
    ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "block_mask");
    if (mask_t) {
        std::vector<ggml_fp16_t> mask_data((size_t)T * T);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int r = 0; r < T; r++)
            for (int c = 0; c < T; c++)
                mask_data[(size_t)r * T + c] = (r / ctx_size == c / ctx_size) ? zero : neg_inf;
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    // RPE lookup. With with_rpe=true, upload the per-layer precomputed CPU
    // tables stacked along the third axis (one (ctx_size*hd, ctx_size) slab
    // per layer). Each layer's flat layout matches the CPU vector
    // cpu_rpe[c * ctx_size * hd + r * hd + d] so the per-layer copy is a
    // direct memcpy at offset il * per_layer.
    // Without with_rpe, the tensor still exists in the graph (declared as
    // input) but is unused — fill with zeros to satisfy the scheduler.
    ggml_tensor* rpe_t = ggml_graph_get_tensor(gf, "rpe_lookup");
    if (rpe_t) {
        const size_t per_layer = (size_t)ctx_size * hd * ctx_size;
        if (with_rpe) {
            std::vector<float> stacked(per_layer * n_layers, 0.0f);
            for (int il = 0; il < n_layers; il++) {
                if ((int)ctx->rpe_per_layer.size() > il && ctx->rpe_per_layer[il].size() == per_layer) {
                    std::memcpy(stacked.data() + (size_t)il * per_layer, ctx->rpe_per_layer[il].data(),
                                per_layer * sizeof(float));
                }
            }
            ggml_backend_tensor_set(rpe_t, stacked.data(), 0, stacked.size() * sizeof(float));
        } else {
            std::vector<float> rpe_zeros(per_layer, 0.0f);
            ggml_backend_tensor_set(rpe_t, rpe_zeros.data(), 0, rpe_zeros.size() * sizeof(float));
        }
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "granite_encoder_graph: compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_output");
    if (!out)
        return nullptr;
    // ne[0] is the feature dim — `d` for the base graph, `(n_cat + 1) * d`
    // for the PLUS variant after the in-graph concat.
    const int actual_dim = (int)out->ne[0];
    if (out_dim)
        *out_dim = actual_dim;
    size_t total = (size_t)T * actual_dim;
    float* result = (float*)malloc(total * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, total * sizeof(float));
    return result;
}

// Single-matmul dispatcher (out = W @ x [+ bias]) lives in
// core/cpu_ops.h::matmul. Local wrapper that adapts the call to take
// `ctx` so existing call sites stay readable.
static inline bool run_matmul(granite_speech_context* ctx, float* out, const float* x, int d_in, int T, ggml_tensor* W,
                              ggml_tensor* bias, int d_out) {
    return core_cpu::matmul(ctx->compute_meta, ctx->sched, out, x, d_in, T, W, bias, d_out);
}

// Run two matmuls with the same input in a single graph dispatch. ggml
// schedules both branches in parallel, saving one scheduler reset / Metal
// command-buffer round-trip per encoder layer (~16 dispatches saved per
// encoder forward at T=550). Used for fusing the encoder attention's
// Q (d -> d) and KV (d -> 2d) projections, which share the same `normed`
// input.
static bool run_matmul_pair(granite_speech_context* ctx, float* out_a, ggml_tensor* W_a, int d_out_a, float* out_b,
                            ggml_tensor* W_b, int d_out_b, const float* x, int d_in, int T) {
    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);

    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_in, T);
    ggml_set_name(inp, "mm_pair_in");
    ggml_set_input(inp);

    ggml_tensor* r_a = ggml_mul_mat(ctx0, W_a, inp);
    ggml_set_name(r_a, "mm_pair_out_a");
    ggml_set_output(r_a);

    ggml_tensor* r_b = ggml_mul_mat(ctx0, W_b, inp);
    ggml_set_name(r_b, "mm_pair_out_b");
    ggml_set_output(r_b);

    ggml_build_forward_expand(gf, r_a);
    ggml_build_forward_expand(gf, r_b);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return false;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mm_pair_in"), x, 0, (size_t)d_in * T * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return false;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "mm_pair_out_a"), out_a, 0, (size_t)d_out_a * T * sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "mm_pair_out_b"), out_b, 0, (size_t)d_out_b * T * sizeof(float));
    return true;
}

// FFN, fused norm+Q/KV matmul pair, and conv module helpers all live in
// core/conformer_ibm.h (shared with granite_nle).

// CPU LayerNorm lives in core/cpu_ops.h::layernorm.

extern "C" float* granite_speech_run_encoder(struct granite_speech_context* ctx, const float* mel, int n_mels,
                                             int T_mel, int* out_N, int* out_dim) {
    if (!ctx || !mel || n_mels != 160)
        return nullptr;
    granite_bench_stage _b("run_encoder");
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model; // 1024

    // Default path: run the full 16-layer encoder as a single ggml graph
    // (PLAN #16). The graph encoder includes Shaw RPE per-layer and is
    // bit-near-identical to the per-op CPU loop while being ~2.3× faster
    // end-to-end on M1 + Q4_K. PLUS variant (cat_layers) is supported via
    // an in-graph concat of the captured per-layer post-norm hiddens with
    // the final encoder output. Set GRANITE_DISABLE_ENCODER_GRAPH=1 to
    // fall back to the CPU loop (slower but kept around for debugging).
    bool use_graph = true;
    if (const char* e = std::getenv("GRANITE_DISABLE_ENCODER_GRAPH"))
        if (e[0] != '0' && e[0] != '\0')
            use_graph = false;
    if (use_graph) {
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "  encoder: using ggml graph path%s\n",
                    ctx->proj_cat_layers_parsed.empty() ? "" : " (PLUS in-graph concat)");
        int graph_dim = d;
        float* result = granite_run_encoder_graph(ctx, mel, n_mels, T_mel, d, &graph_dim);
        if (result) {
            if (out_N)
                *out_N = T_mel;
            if (out_dim)
                *out_dim = graph_dim;
            return result;
        }
        fprintf(stderr, "  encoder: graph path failed, falling back to CPU loops\n");
    }

    const int n_heads = (int)hp.enc_n_heads;   // 8
    const int hd = (int)hp.enc_head_dim;       // 128
    const int n_layers = (int)hp.enc_n_layers; // 16
    const int ctx_size = (int)hp.enc_context_size;
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = T_mel;
    const int remainder = T % ctx_size;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  encoder: per-layer processing T=%d d=%d layers=%d ctx=%d\n", T, d, n_layers, ctx_size);

    // For the PLUS variant, capture intermediate hidden states at the
    // configured cat_layers indices. After the last layer we concatenate
    // them along the feature dim with the final encoder output.
    //
    // INDEXING CONVENTION: cat_layers indices follow HuggingFace's
    // `output_hidden_states` tuple — index 0 is the *input embedding*
    // (after input_linear, before any encoder block), index N is the
    // output of encoder block N-1. So `cat_layers: [3]` captures after
    // the 3rd encoder block, which is `il == 2` in our 0-indexed loop.
    const bool do_cat_concat = !ctx->proj_cat_layers_parsed.empty();
    std::vector<std::vector<float>> cat_layer_outputs;
    if (do_cat_concat)
        cat_layer_outputs.resize(ctx->proj_cat_layers_parsed.size());

    // Input linear: mel (160, T) → hidden (d, T)
    std::vector<float> hidden((size_t)d * T);
    run_matmul(ctx, hidden.data(), mel, n_mels, T, ctx->model.encoder.input_w, ctx->model.encoder.input_b, d);

    // PLUS: snapshot the input embedding for any cat_layer index 0.
    if (do_cat_concat) {
        for (size_t k = 0; k < ctx->proj_cat_layers_parsed.size(); k++) {
            if (ctx->proj_cat_layers_parsed[k] == 0) {
                cat_layer_outputs[k].assign(hidden.begin(), hidden.end());
            }
        }
    }

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30, mx = -1e30, s = 0;
        for (size_t i = 0; i < (size_t)d * T; i++) {
            if (hidden[i] < mn)
                mn = hidden[i];
            if (hidden[i] > mx)
                mx = hidden[i];
            s += hidden[i];
        }
        fprintf(stderr, "  input_linear: min=%.4f max=%.4f mean=%.6f first_4=[%.4f,%.4f,%.4f,%.4f]\n", mn, mx,
                s / (d * T), hidden[0], hidden[1], hidden[2], hidden[3]);
    }

    // Buffers
    std::vector<float> ffn_out((size_t)d * T);
    std::vector<float> Q((size_t)d * T), KV((size_t)d * 2 * T);
    std::vector<float> attn_out((size_t)d * T);
    std::vector<float> conv_out((size_t)d * T);

    // Use the per-layer Shaw RPE lookups precomputed at load time
    // (ctx->rpe_per_layer, built via core_conformer_ibm::build_shaw_rpe_lookup).
    const auto& rpe_per_layer = ctx->rpe_per_layer;

    for (int il = 0; il < n_layers; il++) {
        const auto& b = ctx->model.encoder.blocks[il];

        // --- FFN1 (Macaron half-step) ---
        core_conformer_ibm::run_ffn(ctx->compute_meta, ctx->sched, ffn_out.data(), hidden.data(), d, T, b.ff1_norm_w,
                                    b.ff1_norm_b, b.ff1_up_w, b.ff1_up_b, b.ff1_down_w, b.ff1_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += 0.5f * ffn_out[i];

        if (ctx->params.verbosity >= 2 && il == 0)
            fprintf(stderr, "  L0 after FFN1: [%.4f,%.4f,%.4f,%.4f]\n", hidden[0], hidden[1], hidden[2], hidden[3]);

        // --- Attention: norm + Q/KV projections (fused graph) → Shaw attention on CPU ---
        // The norm + Q + KV go in a single graph dispatch: the layernorm
        // runs on the same backend as the matmuls (Metal-accelerated when
        // available) and we skip the CPU normalisation pass.
        core_conformer_ibm::run_norm_matmul_pair(ctx->compute_meta, ctx->sched, Q.data(), b.attn_q_w, d, KV.data(),
                                                 b.attn_kv_w, d * 2, hidden.data(), d, T, b.attn_norm_w, b.attn_norm_b,
                                                 1e-5f);

        // Split KV: KV layout is (2*d, T) — ne[0]=2*d per frame
        // First d values = K, next d values = V
        std::vector<float> K((size_t)d * T), V((size_t)d * T);
        for (int t = 0; t < T; t++) {
            std::memcpy(K.data() + (size_t)t * d, KV.data() + (size_t)t * 2 * d, d * sizeof(float));
            std::memcpy(V.data() + (size_t)t * d, KV.data() + (size_t)t * 2 * d + d, d * sizeof(float));
        }

        // Shaw block attention on CPU
        core_conformer_ibm::shaw_block_attention_cpu(attn_out.data(), Q.data(), K.data(), V.data(),
                                                     rpe_per_layer[il].empty() ? nullptr : rpe_per_layer[il].data(), T,
                                                     n_heads, hd, ctx_size, attn_scale, remainder);

        if (ctx->params.verbosity >= 2 && il == 0) {
            fprintf(stderr, "  L0 after attn[0][:4] = %.4f %.4f %.4f %.4f\n", attn_out[0], attn_out[1], attn_out[2],
                    attn_out[3]);
        }

        // Output projection + residual
        std::vector<float> proj_out((size_t)d * T);
        run_matmul(ctx, proj_out.data(), attn_out.data(), d, T, b.attn_out_w, b.attn_out_b, d);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += proj_out[i];

        if (ctx->params.verbosity >= 2 && il == 0)
            fprintf(stderr, "  L0 after attn: [%.4f,%.4f,%.4f,%.4f]\n", hidden[0], hidden[1], hidden[2], hidden[3]);

        // --- Conv module ---
        core_conformer_ibm::run_conv_module(ctx->compute_meta, ctx->sched, conv_out.data(), hidden.data(), d, T,
                                            b.conv_norm_w, b.conv_norm_b, b.conv_up_w, b.conv_up_b, b.conv_dw_w,
                                            b.conv_bn_w, b.conv_bn_b, b.conv_down_w, b.conv_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += conv_out[i];

        if (ctx->params.verbosity >= 2 && il == 0)
            fprintf(stderr, "  L0 after conv: [%.4f,%.4f,%.4f,%.4f]\n", hidden[0], hidden[1], hidden[2], hidden[3]);

        // --- FFN2 (Macaron half-step) ---
        core_conformer_ibm::run_ffn(ctx->compute_meta, ctx->sched, ffn_out.data(), hidden.data(), d, T, b.ff2_norm_w,
                                    b.ff2_norm_b, b.ff2_up_w, b.ff2_up_b, b.ff2_down_w, b.ff2_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += 0.5f * ffn_out[i];

        // --- Post LayerNorm ---
        {
            std::vector<float> nw(d), nb(d);
            if (b.post_norm_w)
                ggml_backend_tensor_get(b.post_norm_w, nw.data(), 0, d * sizeof(float));
            if (b.post_norm_b)
                ggml_backend_tensor_get(b.post_norm_b, nb.data(), 0, d * sizeof(float));
            core_cpu::layernorm(hidden.data(), hidden.data(), b.post_norm_w ? nw.data() : nullptr,
                                b.post_norm_b ? nb.data() : nullptr, d, T, 1e-5f);
        }

        // PLUS: snapshot the post-norm hidden state at any cat_layer index.
        // HF convention: `output_hidden_states[il + 1]` is the output of
        // encoder block `il`. So a config index of N corresponds to our
        // `il == N - 1`. Done before mid-CTC residual so the snapshot
        // matches what HF stores in its hidden_states tuple.
        if (do_cat_concat) {
            for (size_t k = 0; k < ctx->proj_cat_layers_parsed.size(); k++) {
                if (ctx->proj_cat_layers_parsed[k] == il + 1) {
                    cat_layer_outputs[k].assign(hidden.begin(), hidden.end());
                }
            }
        }

        // --- Mid-CTC residual at layer 8 ---
        if (il == n_layers / 2 - 1 && ctx->model.encoder.ctc_out_w && ctx->model.encoder.ctc_mid_w) {
            const int ctc_dim = (int)ctx->model.encoder.ctc_out_w->ne[1]; // output_dim from tensor shape
            std::vector<float> mid_out((size_t)ctc_dim * T), mid_back((size_t)d * T);
            run_matmul(ctx, mid_out.data(), hidden.data(), d, T, ctx->model.encoder.ctc_out_w,
                       ctx->model.encoder.ctc_out_b, ctc_dim);
            // Softmax per frame
            for (int t = 0; t < T; t++) {
                float* row = mid_out.data() + t * ctc_dim;
                float mx = -1e30f;
                for (int i = 0; i < ctc_dim; i++)
                    if (row[i] > mx)
                        mx = row[i];
                float sum = 0;
                for (int i = 0; i < ctc_dim; i++) {
                    row[i] = std::exp(row[i] - mx);
                    sum += row[i];
                }
                for (int i = 0; i < ctc_dim; i++)
                    row[i] /= sum;
            }
            run_matmul(ctx, mid_back.data(), mid_out.data(), ctc_dim, T, ctx->model.encoder.ctc_mid_w,
                       ctx->model.encoder.ctc_mid_b, d);
            for (size_t i = 0; i < (size_t)d * T; i++)
                hidden[i] += mid_back[i];
        }

        if (ctx->params.verbosity >= 2) {
            if (il == 0 || il == 3 || il == 7 || il == n_layers - 1) {
                float mn = 1e30, mx = -1e30, s = 0;
                for (size_t i = 0; i < (size_t)d * T; i++) {
                    if (hidden[i] < mn)
                        mn = hidden[i];
                    if (hidden[i] > mx)
                        mx = hidden[i];
                    s += hidden[i];
                }
                fprintf(stderr, "  layer %d/%d: min=%.4f max=%.4f mean=%.6f first_4=[%.4f,%.4f,%.4f,%.4f]\n", il + 1,
                        n_layers, mn, mx, s / (d * T), hidden[0], hidden[1], hidden[2], hidden[3]);
            }
        }
    }

    // PLUS variant: concat captured cat_layers + final into a wider tensor.
    // Layout per frame: [cat_layer_0[d], cat_layer_1[d], ..., final[d]] →
    // total feature dim = (n_cat + 1) * d. Frame-major so the projector's
    // existing per-frame input stride works unchanged.
    if (do_cat_concat) {
        const int n_cat = (int)cat_layer_outputs.size();
        const int wide_d = (n_cat + 1) * d;
        const int expect_d = (int)ctx->model.hparams.proj_encoder_hidden_size;
        if (wide_d != expect_d) {
            fprintf(stderr,
                    "granite_speech: cat_layers width mismatch — cat=%d * d=%d + final=%d → %d, but "
                    "proj.encoder_hidden_size = %d\n",
                    n_cat, d, d, wide_d, expect_d);
        }
        std::vector<float> wide((size_t)wide_d * T);
        for (int t = 0; t < T; t++) {
            float* dst = wide.data() + (size_t)t * wide_d;
            for (int k = 0; k < n_cat; k++) {
                if (cat_layer_outputs[k].empty())
                    continue;
                std::memcpy(dst + (size_t)k * d, cat_layer_outputs[k].data() + (size_t)t * d,
                            (size_t)d * sizeof(float));
            }
            std::memcpy(dst + (size_t)n_cat * d, hidden.data() + (size_t)t * d, (size_t)d * sizeof(float));
        }
        size_t wide_total = (size_t)T * wide_d;
        float* result = (float*)malloc(wide_total * sizeof(float));
        std::memcpy(result, wide.data(), wide_total * sizeof(float));
        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "  encoder: PLUS concat output (%d, %d)\n", T_mel, wide_d);
        if (out_N)
            *out_N = T_mel;
        if (out_dim)
            *out_dim = wide_d;
        return result;
    }

    size_t total = (size_t)T * d;
    float* result = (float*)malloc(total * sizeof(float));
    std::memcpy(result, hidden.data(), total * sizeof(float));

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, sum = 0;
        for (size_t i = 0; i < total; i++) {
            if (result[i] < mn)
                mn = result[i];
            if (result[i] > mx)
                mx = result[i];
            sum += result[i];
        }
        fprintf(stderr, "  encoder output: (%d, %d) min=%.4f max=%.4f mean=%.6f\n", T_mel, d, mn, mx, sum / total);
        fprintf(stderr, "  encoder[0][:4] = %.4f %.4f %.4f %.4f\n", result[0], result[1], result[2], result[3]);
    }

    if (out_N)
        *out_N = T_mel;
    if (out_dim)
        *out_dim = d;
    return result;
}

// ===========================================================================
// Q-Former projector (2 layers: self-attn + cross-attn + FFN per layer)
// ===========================================================================

static ggml_cgraph* granite_build_projector(granite_speech_context* ctx, int enc_len) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.proj_d_model;                 // 1024 (queries / K-V output dim)
    const int enc_d = (int)hp.proj_encoder_hidden_size; // 1024 (base) / 2048 (plus, cat-layers)
    const int n_heads = (int)hp.proj_n_heads;           // 16
    const int hd = d / n_heads;                         // 64
    const int n_layers = (int)hp.proj_n_layers;
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    // Query tokens: (1, n_query=3, 1024)
    int n_query = (int)m.projector.query->ne[1]; // 3

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Encoder output as input
    ggml_tensor* enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, enc_d, enc_len);
    ggml_set_name(enc, "proj_enc_input");
    ggml_set_input(enc);

    // Learned query tokens: extract from (1, n_query, d) → (d, n_query)
    ggml_tensor* query = ggml_reshape_2d(ctx0, m.projector.query, d, n_query);

    // Apply input LayerNorm to query embeddings (NOT encoder features)
    // This matches HF BLIP-2 QFormer: self.layernorm(query_embeds)
    ggml_tensor* cur = query; // (d, n_query)
    if (m.projector.ln_w) {
        cur = ggml_norm(ctx0, cur, 1e-12f);
        cur = ggml_mul(ctx0, cur, m.projector.ln_w);
        if (m.projector.ln_b)
            cur = ggml_add(ctx0, cur, m.projector.ln_b);
    }

    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.projector.blocks[il];

        // --- Self-attention among query tokens ---
        {
            ggml_tensor* Q = ggml_mul_mat(ctx0, b.sa_q_w, cur);
            if (b.sa_q_b)
                Q = ggml_add(ctx0, Q, b.sa_q_b);
            ggml_tensor* K = ggml_mul_mat(ctx0, b.sa_k_w, cur);
            if (b.sa_k_b)
                K = ggml_add(ctx0, K, b.sa_k_b);
            ggml_tensor* V = ggml_mul_mat(ctx0, b.sa_v_w, cur);
            if (b.sa_v_b)
                V = ggml_add(ctx0, V, b.sa_v_b);

            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, n_query);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, n_query);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, n_query);
            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            V = ggml_permute(ctx0, V, 0, 2, 1, 3);

            ggml_tensor* sa = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
            sa = ggml_reshape_2d(ctx0, sa, d, n_query);
            sa = ggml_mul_mat(ctx0, b.sa_out_w, sa);
            if (b.sa_out_b)
                sa = ggml_add(ctx0, sa, b.sa_out_b);

            cur = ggml_add(ctx0, cur, sa);
            // LayerNorm
            if (b.sa_norm_w) {
                cur = ggml_norm(ctx0, cur, 1e-12f);
                cur = ggml_mul(ctx0, cur, b.sa_norm_w);
                if (b.sa_norm_b)
                    cur = ggml_add(ctx0, cur, b.sa_norm_b);
            }
        }

        // --- Cross-attention: queries attend to encoder output ---
        {
            ggml_tensor* Q = ggml_mul_mat(ctx0, b.ca_q_w, cur);
            if (b.ca_q_b)
                Q = ggml_add(ctx0, Q, b.ca_q_b);
            ggml_tensor* K = ggml_mul_mat(ctx0, b.ca_k_w, enc);
            if (b.ca_k_b)
                K = ggml_add(ctx0, K, b.ca_k_b);
            ggml_tensor* V = ggml_mul_mat(ctx0, b.ca_v_w, enc);
            if (b.ca_v_b)
                V = ggml_add(ctx0, V, b.ca_v_b);

            Q = ggml_reshape_3d(ctx0, Q, hd, n_heads, n_query);
            K = ggml_reshape_3d(ctx0, K, hd, n_heads, enc_len);
            V = ggml_reshape_3d(ctx0, V, hd, n_heads, enc_len);
            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            V = ggml_permute(ctx0, V, 0, 2, 1, 3);

            ggml_tensor* ca = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, attn_scale, 0.0f, 0.0f);
            ca = ggml_reshape_2d(ctx0, ca, d, n_query);
            ca = ggml_mul_mat(ctx0, b.ca_out_w, ca);
            if (b.ca_out_b)
                ca = ggml_add(ctx0, ca, b.ca_out_b);

            cur = ggml_add(ctx0, cur, ca);
            if (b.ca_norm_w) {
                cur = ggml_norm(ctx0, cur, 1e-12f);
                cur = ggml_mul(ctx0, cur, b.ca_norm_w);
                if (b.ca_norm_b)
                    cur = ggml_add(ctx0, cur, b.ca_norm_b);
            }
        }

        // --- FFN ---
        {
            ggml_tensor* x = ggml_mul_mat(ctx0, b.ffn_up_w, cur);
            if (b.ffn_up_b)
                x = ggml_add(ctx0, x, b.ffn_up_b);
            x = ggml_gelu_erf(ctx0, x);
            x = ggml_mul_mat(ctx0, b.ffn_down_w, x);
            if (b.ffn_down_b)
                x = ggml_add(ctx0, x, b.ffn_down_b);

            cur = ggml_add(ctx0, cur, x);
            if (b.ffn_norm_w) {
                cur = ggml_norm(ctx0, cur, 1e-12f);
                cur = ggml_mul(ctx0, cur, b.ffn_norm_w);
                if (b.ffn_norm_b)
                    cur = ggml_add(ctx0, cur, b.ffn_norm_b);
            }
        }
    }

    // Final linear projection: (1024 → 2048)
    cur = ggml_mul_mat(ctx0, m.projector.linear_w, cur);
    if (m.projector.linear_b)
        cur = ggml_add(ctx0, cur, m.projector.linear_b);

    ggml_set_name(cur, "proj_output");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float* granite_speech_run_projector(struct granite_speech_context* ctx, const float* enc_out, int enc_len,
                                               int enc_dim, int* out_N, int* out_dim) {
    // Plus variant supplies a wider (concat) encoder output. Accept either
    // proj_d_model (base / 4.1) or proj_encoder_hidden_size (plus = 2048).
    if (!ctx || !enc_out)
        return nullptr;
    const int expect_d = (int)ctx->model.hparams.proj_encoder_hidden_size;
    if (enc_dim != expect_d) {
        fprintf(stderr, "granite_speech: projector expected enc_dim=%d, got %d\n", expect_d, enc_dim);
        return nullptr;
    }
    granite_bench_stage _b("run_projector");
    const int d = enc_dim;                                          // 1024 (base) or 2048 (plus)
    const int llm_d = (int)ctx->model.hparams.llm_d_model;          // 2048
    const int window_size = (int)ctx->model.hparams.window_size;    // 15
    const int downsample = (int)ctx->model.hparams.downsample_rate; // 5
    const int n_query = window_size / downsample;                   // 3
    const int nblocks = (enc_len + window_size - 1) / window_size;  // ceil
    const int total_tokens = nblocks * n_query;

    if (ctx->params.verbosity >= 2)
        fprintf(stderr, "  projector: enc_len=%d window=%d nblocks=%d n_query=%d total_tokens=%d\n", enc_len,
                window_size, nblocks, n_query, total_tokens);

    // Pad encoder output to multiple of window_size
    int padded_len = nblocks * window_size;
    std::vector<float> padded((size_t)padded_len * d, 0.0f);
    std::memcpy(padded.data(), enc_out, (size_t)enc_len * d * sizeof(float));

    // Run Q-Former per window
    std::vector<float> all_proj((size_t)total_tokens * llm_d);

    for (int blk = 0; blk < nblocks; blk++) {
        const float* window_data = padded.data() + (size_t)blk * window_size * d;

        // Build Q-Former graph for this window
        ggml_cgraph* gf = granite_build_projector(ctx, window_size);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
            return nullptr;

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "proj_enc_input"), window_data, 0,
                                (size_t)window_size * d * sizeof(float));

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
            return nullptr;

        ggml_tensor* out = ggml_graph_get_tensor(gf, "proj_output");
        ggml_backend_tensor_get(out, all_proj.data() + (size_t)blk * n_query * llm_d, 0,
                                (size_t)n_query * llm_d * sizeof(float));
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  projector: %d windows × %d queries = %d audio tokens (dim=%d)\n", nblocks, n_query,
                total_tokens, llm_d);

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, s = 0;
        for (size_t i = 0; i < all_proj.size(); i++) {
            if (all_proj[i] < mn)
                mn = all_proj[i];
            if (all_proj[i] > mx)
                mx = all_proj[i];
            s += all_proj[i];
        }
        fprintf(stderr, "  projector out: min=%.6f max=%.6f mean=%.6f first_4=[%.6f,%.6f,%.6f,%.6f]\n", mn, mx,
                s / all_proj.size(), all_proj[0], all_proj[1], all_proj[2], all_proj[3]);
    }

    float* result = (float*)malloc(all_proj.size() * sizeof(float));
    std::memcpy(result, all_proj.data(), all_proj.size() * sizeof(float));
    if (out_N)
        *out_N = total_tokens;
    if (out_dim)
        *out_dim = llm_d;
    return result;
}

extern "C" bool granite_speech_kv_init(struct granite_speech_context* ctx, int max_ctx) {
    if (!ctx || max_ctx <= 0)
        return false;
    // Idempotent: same per-chunk re-init pattern as voxtral4b (issue
    // #54). Without this guard, the per-transcribe call in
    // crispasr_backend_granite leaks the entire KV backend buffer
    // every chunk.
    if (ctx->kv_k)
        return true;
    const auto& hp = ctx->model.hparams;
    const int hd = (int)hp.llm_head_dim;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int nl = (int)hp.llm_n_layers;

    ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(ip);
    // PLAN #60e + #69e: per-half KV dtype. CRISPASR_KV_QUANT sets both,
    // CRISPASR_KV_QUANT_{K,V} override per half. Default f16/f16.
    const auto kv_pair = core_attn::kv_dtype_pair_from_env("granite_speech");
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hd, max_ctx, n_kv, nl);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hd, max_ctx, n_kv, nl);
    // Size the backend buffer from actual ggml_nbytes (was hardcoded F16
    // byte count, which over-allocated for quant types).
    const size_t k_size = ggml_nbytes(ctx->kv_k);
    const size_t v_size = ggml_nbytes(ctx->kv_v);
    // PLAN #69b: optional KV-on-CPU spill for long-context / tight-VRAM users.
    ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->backend, ctx->backend_cpu, "granite_speech");
    ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, k_size + v_size);
    char* base = (char*)ggml_backend_buffer_get_base(ctx->kv_buf);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_k, base);
    ggml_backend_tensor_alloc(ctx->kv_buf, ctx->kv_v, base + k_size);

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "granite_speech: kv cache %.0f MiB k=%s v=%s (on %s)\n", (k_size + v_size) / 1048576.0,
                ggml_type_name(kv_pair.k), ggml_type_name(kv_pair.v), kv_backend == ctx->backend_cpu ? "cpu" : "gpu");
    return true;
}

extern "C" void granite_speech_kv_reset(struct granite_speech_context* ctx) {
    if (ctx && ctx->kv_buf)
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
}

// ===========================================================================
// Granite LLM graph (40 layers, GQA 16/4, μP multipliers)
// ===========================================================================

static ggml_cgraph* granite_build_llm_kv(granite_speech_context* ctx, int n_past, int n_tokens) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model; // 2048
    const int n_layers = (int)hp.llm_n_layers;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds, "inputs_embeds");
    ggml_set_input(embeds);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_past + n_tokens, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    core_granite_llm::Hparams llm_hp = {};
    llm_hp.n_layers = n_layers;
    llm_hp.d_model = d;
    llm_hp.n_heads = (int)hp.llm_n_heads;
    llm_hp.n_kv_heads = (int)hp.llm_n_kv_heads;
    llm_hp.head_dim = (int)hp.llm_head_dim;
    llm_hp.rms_eps = hp.llm_rms_eps;
    llm_hp.rope_theta = hp.llm_rope_theta;
    llm_hp.embedding_multiplier = hp.embedding_multiplier;
    llm_hp.attention_multiplier = hp.attention_multiplier;
    llm_hp.residual_multiplier = hp.residual_multiplier;

    std::vector<core_granite_llm::LayerWeights> blocks(n_layers);
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.llm.blocks[il];
        blocks[il] = {b.attn_norm_w, b.attn_q_w,   b.attn_k_w, b.attn_v_w,  b.attn_out_w,
                      b.ffn_norm_w,  b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w};
    }

    ggml_tensor* cur = core_granite_llm::build_decoder(ctx0, gf, embeds, positions, causal_mask, ctx->kv_k, ctx->kv_v,
                                                       n_past, blocks, m.llm.output_norm_w, llm_hp, /*is_causal*/ true);

    // LM head (separate, not tied) with μP logits scaling
    if (n_tokens > 1) {
        cur = ggml_view_1d(ctx0, cur, d, (size_t)(n_tokens - 1) * d * sizeof(float));
        cur = ggml_reshape_2d(ctx0, cur, d, 1);
    }
    cur = ggml_mul_mat(ctx0, m.llm.output_w, cur);
    cur = ggml_scale(ctx0, cur, 1.0f / hp.logits_scaling); // μP logits scaling

    ggml_set_name(cur, "logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" float* granite_speech_run_llm_kv(struct granite_speech_context* ctx, const float* inputs_embeds,
                                            int n_tokens, int n_past, int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;

    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;

    std::vector<ggml_fp16_t> mask;
    if (n_tokens > 1) {
        int Lk = n_past + n_tokens;
        mask.resize((size_t)n_tokens * Lk, ggml_fp32_to_fp16(0.0f));
        ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++)
            for (int k = 0; k < Lk; k++)
                if (k > n_past + q)
                    mask[(size_t)q * Lk + k] = neg_inf;
    }

    ggml_cgraph* gf = granite_build_llm_kv(ctx, n_past, n_tokens);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inputs_embeds"), inputs_embeds, 0,
                            (size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));
    if (n_tokens > 1)
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0,
                                mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    // Debug: dump per-layer values during prefill
    if (ctx->params.verbosity >= 2 && n_tokens > 1) {
        auto dump = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t)
                return;
            float buf[8];
            ggml_backend_tensor_get(t, buf, 0, 4 * sizeof(float));
            float buf2[4];
            size_t last_off = (size_t)(n_tokens - 1) * d * sizeof(float);
            ggml_backend_tensor_get(t, buf2, last_off, 4 * sizeof(float));
            fprintf(stderr, "  %s: [0,:4]=[%.4f,%.4f,%.4f,%.4f] [-1,:4]=[%.4f,%.4f,%.4f,%.4f]\n", name, buf[0], buf[1],
                    buf[2], buf[3], buf2[0], buf2[1], buf2[2], buf2[3]);
        };
        dump("emb_scaled");

        // Q/K before/after RoPE for layer 0
        // Q shape: (hd, n_q, n_tokens) = (128, 16, 89) in ggml
        // To read head 0, pos 0: offset 0, 4 floats
        // To read head 0, pos 88: offset = 88 * hd * n_q * sizeof(float)? No...
        // ggml layout: ne[0]=hd=128, ne[1]=n_q=16, ne[2]=n_tokens=89
        // element [d, head, tok] = data[tok * n_q * hd + head * hd + d]
        // head 0, tok 0: offset=0
        // head 0, tok 88: offset = 88 * 16 * 128 * sizeof(float)
        {
            const int hd_loc = (int)hp.llm_head_dim;
            const int n_q_loc = (int)hp.llm_n_heads;
            auto dump_qk = [&](const char* name) {
                ggml_tensor* t = ggml_graph_get_tensor(gf, name);
                if (!t)
                    return;
                float buf0[4], buf88[4];
                // head0, pos0
                ggml_backend_tensor_get(t, buf0, 0, 4 * sizeof(float));
                // head0, pos 88
                size_t off88 = (size_t)(n_tokens - 1) * n_q_loc * hd_loc * sizeof(float);
                ggml_backend_tensor_get(t, buf88, off88, 4 * sizeof(float));
                fprintf(stderr, "  %s [h0,p0,:4]=[%.4f,%.4f,%.4f,%.4f] [h0,p88,:4]=[%.4f,%.4f,%.4f,%.4f]\n", name,
                        buf0[0], buf0[1], buf0[2], buf0[3], buf88[0], buf88[1], buf88[2], buf88[3]);
            };
            dump_qk("L0_Q_pre");
            dump_qk("L0_Q_post");
            dump_qk("L0_K_pre");
            dump_qk("L0_K_post");
        }

        dump("layer_0");
        dump("layer_1");
        dump("layer_19");
        dump("layer_38");
        dump("layer_39");
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    return result;
}

extern "C" float* granite_speech_embed_tokens(struct granite_speech_context* ctx, const int32_t* input_ids,
                                              int n_tokens) {
    if (!ctx || !input_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_d_model;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 64, false);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "input_ids");
    ggml_set_input(ids);
    ggml_tensor* out = ggml_get_rows(ctx0, ctx->model.llm.token_embd_w, ids);

    // NOTE: embedding_multiplier (μP) is NOT applied here — it's applied in
    // granite_build_llm_kv to ALL inputs (text + audio) uniformly, matching HF/mlx.

    ggml_set_name(out, "embeds");
    ggml_build_forward_expand(gf, out);
    ggml_free(ctx0);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input_ids"), input_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;
    ggml_tensor* emb = ggml_graph_get_tensor(gf, "embeds");
    float* result = (float*)malloc((size_t)n_tokens * d * sizeof(float));
    ggml_backend_tensor_get(emb, result, 0, (size_t)n_tokens * d * sizeof(float));
    return result;
}

extern "C" const char* granite_speech_token_text(struct granite_speech_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->id_to_token.size())
        return "";
    return ctx->id_to_token[id].c_str();
}

extern "C" char* granite_speech_decode_tokens(struct granite_speech_context* ctx, const int32_t* ids, int n_ids) {
    if (!ctx || !ids || n_ids <= 0)
        return nullptr;
    std::string result = core_bpe::detokenize(ctx->id_to_token, ids, (size_t)n_ids);
    char* out = (char*)malloc(result.size() + 1);
    std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" char* granite_speech_transcribe(struct granite_speech_context*, const float*, int) {
    fprintf(stderr, "granite_speech: full transcribe not yet implemented\n");
    return nullptr;
}

// ---------------------------------------------------------------------------
// Control-token accessors
//
// Surface the audio placeholder / EOS / vocab size from hparams so the CLI
// adapter (crispasr_backend_granite.cpp) can stop hardcoding granite-4.0-1b
// specific constants. granite-3.x variants use a different tokenizer with
// their own ids — querying at runtime lets a single backend host all four
// model revisions with no per-version branching.
// ---------------------------------------------------------------------------

extern "C" int granite_speech_audio_token_id(struct granite_speech_context* ctx) {
    return ctx ? (int)ctx->model.hparams.audio_token_index : -1;
}

extern "C" int granite_speech_eos_token_id(struct granite_speech_context* ctx) {
    return ctx ? (int)ctx->model.hparams.eos_token_id : -1;
}

extern "C" int granite_speech_vocab_size(struct granite_speech_context* ctx) {
    return ctx ? (int)ctx->model.hparams.llm_vocab_size : -1;
}
