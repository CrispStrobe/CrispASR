// granite_nle.cpp — runtime for ibm-granite/granite-speech-4.1-2b-nar.
//
// SCAFFOLD: tensor table + GGUF loader + public-API stubs. The forward
// pass (encoder, projector, LLM editing) lives behind public APIs that
// currently return nullptr / "not implemented" so callers can wire the
// CLI plumbing while the math gets filled in.
//
// Implementation roadmap (in order):
//   1. compute_mel — copy from granite_speech.cpp (same 80-bin log-mel
//      pipeline). Diff against `tools/dump_reference.py` "nle" backend.
//   2. run_encoder — same Conformer block as granite_speech, plus
//      self-conditioning at layer 8 and BPE auxiliary head with
//      posterior-weighted-pool window=4.
//   3. run_projector — windowed Q-Former (block_size=15, downsample=5,
//      32-head SDPA, mlp_ratio=2). 4-encoder-layer concat + 4 LayerNorms.
//   4. run_llm_editing — single non-causal LLM pass over flat
//      [audio_embs, text_with_slots]. All self_attn layers run with
//      `is_causal=False` (in upstream this is patched in __init__).
//   5. transcribe — orchestrates the whole pipeline plus the editing
//      slot decode (argmax + unique_consecutive + drop EOS).
//
// All scaffolding values default to the granite-speech-4.1-2b-nar
// config; the loader reads any value the GGUF supplies (with old
// values as fallback for legacy GGUFs that may exist later).

#include "granite_nle.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "core/bpe.h"
#include "core/conformer_ibm.h"
#include "core/cpu_ops.h"
#include "core/ctc.h"
#include "core/fft.h"
#include "core/granite_llm.h"
#include "core/qformer.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct granite_nle_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 80;

    // Encoder (16-layer Macaron Conformer, same shape as granite_speech base
    // but with extra heads).
    uint32_t enc_n_layers = 16;
    uint32_t enc_d_model = 1024;
    uint32_t enc_n_heads = 8;
    uint32_t enc_head_dim = 128;
    uint32_t enc_input_dim = 160;
    uint32_t enc_conv_kernel = 15;
    uint32_t enc_ff_dim = 4096;
    uint32_t enc_context_size = 200;
    uint32_t enc_max_pos_emb = 512;
    uint32_t enc_ctc_vocab = 348;    // char-level CTC
    uint32_t enc_bpe_vocab = 100353; // aux BPE CTC
    uint32_t enc_bpe_pooling_window = 4;
    uint32_t enc_self_conditioning_layer = 8; // 1-indexed in upstream

    // Projector (windowed Q-Former with 4-encoder-layer concat input).
    uint32_t proj_num_encoder_layers = 4;
    uint32_t proj_encoder_dim = 1024;
    uint32_t proj_hidden_size = 2048;
    uint32_t proj_llm_dim = 2048;
    uint32_t proj_n_layers = 2;
    uint32_t proj_n_heads = 32;
    uint32_t proj_mlp_ratio = 2;
    uint32_t proj_block_size = 15;
    uint32_t proj_downsample_rate = 5;
    float proj_layernorm_eps = 1e-6f;
    bool proj_attn_bias = true;
    bool proj_mlp_bias = true;
    // Comma-separated list of encoder layer indices to feed into projector;
    // negative values index from the end (e.g. -1 = last layer). Default
    // matches NAR upstream config.
    std::string proj_encoder_layer_indices = "4,8,12,-1";

    // LLM (Granite 4.0-1B used as NAR refiner).
    uint32_t llm_n_layers = 40;
    uint32_t llm_d_model = 2048;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 4;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 4096;
    float llm_rope_theta = 10000.0f;
    float llm_rms_eps = 1e-5f;
    uint32_t llm_vocab_size = 100352; // 100353 in base; 100352 in NAR
    float embedding_multiplier = 12.0f;
    float attention_multiplier = 0.0078125f;
    float residual_multiplier = 0.22f;
    float logits_scaling = 8.0f;
    uint32_t eos_token_id = 100257;
    uint32_t bos_token_id = 100257;
    uint32_t pad_token_id = 100256;
    bool tie_word_embeddings = true; // NAR has no separate lm_head.weight

    // Top-level NAR flag.
    bool scale_projected_embeddings = true;
};

// ===========================================================================
// Model tensors
// ===========================================================================

struct granite_nle_enc_block {
    // FFN1 (Macaron half-step)
    ggml_tensor* ff1_norm_w = nullptr;
    ggml_tensor* ff1_norm_b = nullptr;
    ggml_tensor* ff1_up_w = nullptr;
    ggml_tensor* ff1_up_b = nullptr;
    ggml_tensor* ff1_down_w = nullptr;
    ggml_tensor* ff1_down_b = nullptr;

    // Attention (block-local, Shaw RPE)
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_kv_w = nullptr;
    ggml_tensor* attn_out_w = nullptr;
    ggml_tensor* attn_out_b = nullptr;
    ggml_tensor* attn_rel_pos_w = nullptr;

    // Conv module
    ggml_tensor* conv_up_w = nullptr;
    ggml_tensor* conv_up_b = nullptr;
    ggml_tensor* conv_dw_w = nullptr;
    ggml_tensor* conv_bn_w = nullptr;
    ggml_tensor* conv_bn_b = nullptr;
    ggml_tensor* conv_bn_mean = nullptr;
    ggml_tensor* conv_bn_var = nullptr;
    ggml_tensor* conv_down_w = nullptr;
    ggml_tensor* conv_down_b = nullptr;
    ggml_tensor* conv_norm_w = nullptr;
    ggml_tensor* conv_norm_b = nullptr;

    // FFN2
    ggml_tensor* ff2_norm_w = nullptr;
    ggml_tensor* ff2_norm_b = nullptr;
    ggml_tensor* ff2_up_w = nullptr;
    ggml_tensor* ff2_up_b = nullptr;
    ggml_tensor* ff2_down_w = nullptr;
    ggml_tensor* ff2_down_b = nullptr;

    // Post-block LayerNorm
    ggml_tensor* post_norm_w = nullptr;
    ggml_tensor* post_norm_b = nullptr;
};

struct granite_nle_proj_block {
    // 2-layer simplified Q-Former (cross-attention + MLP).
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_q_b = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_k_b = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_v_b = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_o_b = nullptr;
    ggml_tensor* mlp_norm_w = nullptr;
    ggml_tensor* mlp_norm_b = nullptr;
    ggml_tensor* mlp_fc1_w = nullptr;
    ggml_tensor* mlp_fc1_b = nullptr;
    ggml_tensor* mlp_fc2_w = nullptr;
    ggml_tensor* mlp_fc2_b = nullptr;
};

struct granite_nle_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct granite_nle_model {
    granite_nle_hparams hparams;

    // Encoder
    struct {
        ggml_tensor* input_w = nullptr;
        ggml_tensor* input_b = nullptr;
        ggml_tensor* mel_filters = nullptr; // optional: written by converter
        std::vector<granite_nle_enc_block> blocks;
        ggml_tensor* ctc_out_w = nullptr; // 348-vocab CTC
        ggml_tensor* ctc_out_b = nullptr;
        ggml_tensor* ctc_mid_w = nullptr; // 348 → 1024 self-conditioning
        ggml_tensor* ctc_mid_b = nullptr;
        ggml_tensor* bpe_out_w = nullptr; // 100353-vocab aux head
        ggml_tensor* bpe_out_b = nullptr;
    } encoder;

    // Projector
    struct {
        // Per-encoder-layer LayerNorms (one per encoder_layer_indices entry).
        std::vector<ggml_tensor*> layer_norm_w;
        std::vector<ggml_tensor*> layer_norm_b;
        ggml_tensor* layer_proj_w = nullptr;
        ggml_tensor* layer_proj_b = nullptr;
        ggml_tensor* query = nullptr;      // (1, block_size/downsample, hidden_size)
        ggml_tensor* window_pos = nullptr; // (1, block_size, hidden_size)
        std::vector<granite_nle_proj_block> blocks;
        ggml_tensor* out_norm_w = nullptr;
        ggml_tensor* out_norm_b = nullptr;
        ggml_tensor* out_linear_w = nullptr; // (hidden_size → llm_dim)
        ggml_tensor* out_linear_b = nullptr;
    } projector;

    // LLM (Granite 4.0-1B; tied embeddings — no separate lm_head)
    struct {
        ggml_tensor* token_embd_w = nullptr;
        std::vector<granite_nle_llm_block> blocks;
        ggml_tensor* norm_w = nullptr;
        // lm_head reuses token_embd_w when tie_word_embeddings is true.
    } llm;

    // Storage
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

// ===========================================================================
// Context (public-facing opaque type)
// ===========================================================================

struct granite_nle_context {
    granite_nle_context_params params;
    int n_threads = 4;

    granite_nle_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Precomputed Shaw RPE lookup, per encoder layer:
    //   rpe_per_layer[il][(c*ctx_size + r) * head_dim + d] =
    //     rel_pos_emb_table[clamp(c-r, -ctx_size, ctx_size) + max_pos_emb][d]
    // Built once at load time (RPE weights don't change at inference).
    std::vector<std::vector<float>> rpe_per_layer;

    // Parsed + normalised encoder_layer_indices (HF tuple convention):
    //   index 0 = post-input_linear hidden state
    //   index N (1..n_layers) = output of conformer block N (after the
    //     mid-CTC residual at idx==self_conditioning_layer if applicable)
    //   negative values are normalised to (n_layers + 1 + idx) at load time
    std::vector<int> enc_layer_indices_parsed;

    // CTC tokenizer (char-level, 348 vocab). Built from the
    // `granite_nle.ctc.vocab` newline-joined string at load time.
    std::vector<std::string> ctc_id_to_str;

    // LLM tokenizer (Granite 4.0 BPE). Loaded from
    // `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` (same names the
    // base granite_speech converter uses).
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int> merge_rank;

    // Last encoder run cached for accessor APIs.
    std::vector<float> last_ctc_logits;
    int last_ctc_T = 0;
    std::vector<float> last_bpe_logits;
    int last_bpe_T = 0;
};

// ===========================================================================
// Bench instrumentation (mirrors granite_speech.cpp)
// ===========================================================================

static bool granite_nle_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("GRANITE_NLE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct granite_nle_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit granite_nle_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~granite_nle_bench_stage() {
        if (!granite_nle_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "  bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// FFT lives in core/fft.h (shared with granite_speech).
// GGUF loading
// ===========================================================================

static bool granite_nle_load_model(granite_nle_model& model, const char* path, ggml_backend_t backend) {
    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g)
            return false;
        auto& hp = model.hparams;

        hp.enc_n_layers = core_gguf::kv_u32(g, "granite_nle.enc.n_layers", hp.enc_n_layers);
        hp.enc_d_model = core_gguf::kv_u32(g, "granite_nle.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(g, "granite_nle.enc.n_heads", hp.enc_n_heads);
        hp.enc_head_dim = core_gguf::kv_u32(g, "granite_nle.enc.head_dim", hp.enc_head_dim);
        hp.enc_input_dim = core_gguf::kv_u32(g, "granite_nle.enc.input_dim", hp.enc_input_dim);
        hp.enc_conv_kernel = core_gguf::kv_u32(g, "granite_nle.enc.conv_kernel", hp.enc_conv_kernel);
        hp.enc_ff_dim = core_gguf::kv_u32(g, "granite_nle.enc.ff_dim", hp.enc_ff_dim);
        hp.enc_context_size = core_gguf::kv_u32(g, "granite_nle.enc.context_size", hp.enc_context_size);
        hp.enc_max_pos_emb = core_gguf::kv_u32(g, "granite_nle.enc.max_pos_emb", hp.enc_max_pos_emb);
        hp.enc_ctc_vocab = core_gguf::kv_u32(g, "granite_nle.enc.ctc_vocab", hp.enc_ctc_vocab);
        hp.enc_bpe_vocab = core_gguf::kv_u32(g, "granite_nle.enc.bpe_vocab", hp.enc_bpe_vocab);
        hp.enc_bpe_pooling_window =
            core_gguf::kv_u32(g, "granite_nle.enc.bpe_pooling_window", hp.enc_bpe_pooling_window);
        hp.enc_self_conditioning_layer =
            core_gguf::kv_u32(g, "granite_nle.enc.self_conditioning_layer", hp.enc_self_conditioning_layer);

        hp.proj_num_encoder_layers =
            core_gguf::kv_u32(g, "granite_nle.proj.num_encoder_layers", hp.proj_num_encoder_layers);
        hp.proj_encoder_dim = core_gguf::kv_u32(g, "granite_nle.proj.encoder_dim", hp.proj_encoder_dim);
        hp.proj_hidden_size = core_gguf::kv_u32(g, "granite_nle.proj.hidden_size", hp.proj_hidden_size);
        hp.proj_llm_dim = core_gguf::kv_u32(g, "granite_nle.proj.llm_dim", hp.proj_llm_dim);
        hp.proj_n_layers = core_gguf::kv_u32(g, "granite_nle.proj.n_layers", hp.proj_n_layers);
        hp.proj_n_heads = core_gguf::kv_u32(g, "granite_nle.proj.n_heads", hp.proj_n_heads);
        hp.proj_mlp_ratio = core_gguf::kv_u32(g, "granite_nle.proj.mlp_ratio", hp.proj_mlp_ratio);
        hp.proj_block_size = core_gguf::kv_u32(g, "granite_nle.proj.block_size", hp.proj_block_size);
        hp.proj_downsample_rate = core_gguf::kv_u32(g, "granite_nle.proj.downsample_rate", hp.proj_downsample_rate);
        hp.proj_layernorm_eps = core_gguf::kv_f32(g, "granite_nle.proj.layernorm_eps", hp.proj_layernorm_eps);
        hp.proj_attn_bias = core_gguf::kv_bool(g, "granite_nle.proj.attn_bias", hp.proj_attn_bias);
        hp.proj_mlp_bias = core_gguf::kv_bool(g, "granite_nle.proj.mlp_bias", hp.proj_mlp_bias);
        hp.proj_encoder_layer_indices =
            core_gguf::kv_str(g, "granite_nle.proj.encoder_layer_indices", hp.proj_encoder_layer_indices.c_str());

        hp.llm_n_layers = core_gguf::kv_u32(g, "granite_nle.llm.n_layers", hp.llm_n_layers);
        hp.llm_d_model = core_gguf::kv_u32(g, "granite_nle.llm.d_model", hp.llm_d_model);
        hp.llm_n_heads = core_gguf::kv_u32(g, "granite_nle.llm.n_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(g, "granite_nle.llm.n_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(g, "granite_nle.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(g, "granite_nle.llm.ff_dim", hp.llm_ff_dim);
        hp.llm_rope_theta = core_gguf::kv_f32(g, "granite_nle.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(g, "granite_nle.llm.rms_norm_eps", hp.llm_rms_eps);
        hp.llm_vocab_size = core_gguf::kv_u32(g, "granite_nle.llm.vocab_size", hp.llm_vocab_size);
        hp.embedding_multiplier = core_gguf::kv_f32(g, "granite_nle.llm.embedding_multiplier", hp.embedding_multiplier);
        hp.attention_multiplier = core_gguf::kv_f32(g, "granite_nle.llm.attention_multiplier", hp.attention_multiplier);
        hp.residual_multiplier = core_gguf::kv_f32(g, "granite_nle.llm.residual_multiplier", hp.residual_multiplier);
        hp.logits_scaling = core_gguf::kv_f32(g, "granite_nle.llm.logits_scaling", hp.logits_scaling);
        hp.eos_token_id = core_gguf::kv_u32(g, "granite_nle.llm.eos_token_id", hp.eos_token_id);
        hp.bos_token_id = core_gguf::kv_u32(g, "granite_nle.llm.bos_token_id", hp.bos_token_id);
        hp.pad_token_id = core_gguf::kv_u32(g, "granite_nle.llm.pad_token_id", hp.pad_token_id);
        hp.tie_word_embeddings = core_gguf::kv_bool(g, "granite_nle.llm.tie_word_embeddings", hp.tie_word_embeddings);
        hp.scale_projected_embeddings =
            core_gguf::kv_bool(g, "granite_nle.scale_projected_embeddings", hp.scale_projected_embeddings);

        core_gguf::free_metadata(g);
    }

    // Pass 2: tensor data via shared helper.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "granite_nle", wl)) {
        fprintf(stderr, "granite_nle: failed to load weights from %s\n", path);
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;

    // Helper: lookup-or-null
    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it == wl.tensors.end() ? nullptr : it->second;
    };

    // Encoder roots
    model.encoder.input_w = get("enc.input.weight");
    model.encoder.input_b = get("enc.input.bias");
    model.encoder.mel_filters = get("audio.mel_filters");
    model.encoder.ctc_out_w = get("enc.ctc_out.weight");
    model.encoder.ctc_out_b = get("enc.ctc_out.bias");
    model.encoder.ctc_mid_w = get("enc.ctc_mid.weight");
    model.encoder.ctc_mid_b = get("enc.ctc_mid.bias");
    model.encoder.bpe_out_w = get("enc.bpe_out.weight");
    model.encoder.bpe_out_b = get("enc.bpe_out.bias");

    // Encoder blocks
    model.encoder.blocks.resize(model.hparams.enc_n_layers);
    for (uint32_t il = 0; il < model.hparams.enc_n_layers; il++) {
        auto& b = model.encoder.blocks[il];
        char p[128];
        snprintf(p, sizeof(p), "enc.blk.%u.", il);
        std::string s = p;

        b.ff1_norm_w = get(s + "ff1_norm.weight");
        b.ff1_norm_b = get(s + "ff1_norm.bias");
        b.ff1_up_w = get(s + "ff1_up.weight");
        b.ff1_up_b = get(s + "ff1_up.bias");
        b.ff1_down_w = get(s + "ff1_down.weight");
        b.ff1_down_b = get(s + "ff1_down.bias");

        b.attn_norm_w = get(s + "attn_norm.weight");
        b.attn_norm_b = get(s + "attn_norm.bias");
        b.attn_q_w = get(s + "attn_q.weight");
        b.attn_kv_w = get(s + "attn_kv.weight");
        b.attn_out_w = get(s + "attn_out.weight");
        b.attn_out_b = get(s + "attn_out.bias");
        b.attn_rel_pos_w = get(s + "attn_rel_pos.weight");

        b.conv_up_w = get(s + "conv_up.weight");
        b.conv_up_b = get(s + "conv_up.bias");
        b.conv_dw_w = get(s + "conv_dw.weight");
        b.conv_bn_w = get(s + "conv_bn.weight");
        b.conv_bn_b = get(s + "conv_bn.bias");
        b.conv_bn_mean = get(s + "conv_bn.running_mean");
        b.conv_bn_var = get(s + "conv_bn.running_var");
        b.conv_down_w = get(s + "conv_down.weight");
        b.conv_down_b = get(s + "conv_down.bias");
        b.conv_norm_w = get(s + "conv_norm.weight");
        b.conv_norm_b = get(s + "conv_norm.bias");

        b.ff2_norm_w = get(s + "ff2_norm.weight");
        b.ff2_norm_b = get(s + "ff2_norm.bias");
        b.ff2_up_w = get(s + "ff2_up.weight");
        b.ff2_up_b = get(s + "ff2_up.bias");
        b.ff2_down_w = get(s + "ff2_down.weight");
        b.ff2_down_b = get(s + "ff2_down.bias");

        b.post_norm_w = get(s + "post_norm.weight");
        b.post_norm_b = get(s + "post_norm.bias");
    }

    // Projector
    const uint32_t n_enc_layers_for_proj = model.hparams.proj_num_encoder_layers;
    model.projector.layer_norm_w.resize(n_enc_layers_for_proj);
    model.projector.layer_norm_b.resize(n_enc_layers_for_proj);
    for (uint32_t i = 0; i < n_enc_layers_for_proj; i++) {
        char p[128];
        snprintf(p, sizeof(p), "proj.layer_norm.%u.", i);
        model.projector.layer_norm_w[i] = get(std::string(p) + "weight");
        model.projector.layer_norm_b[i] = get(std::string(p) + "bias");
    }
    model.projector.layer_proj_w = get("proj.layer_proj.weight");
    model.projector.layer_proj_b = get("proj.layer_proj.bias");
    model.projector.query = get("proj.query");
    model.projector.window_pos = get("proj.window_positions");
    model.projector.out_norm_w = get("proj.out_norm.weight");
    model.projector.out_norm_b = get("proj.out_norm.bias");
    model.projector.out_linear_w = get("proj.out_linear.weight");
    model.projector.out_linear_b = get("proj.out_linear.bias");

    model.projector.blocks.resize(model.hparams.proj_n_layers);
    for (uint32_t il = 0; il < model.hparams.proj_n_layers; il++) {
        auto& b = model.projector.blocks[il];
        char p[128];
        snprintf(p, sizeof(p), "proj.blk.%u.", il);
        std::string s = p;
        b.attn_norm_w = get(s + "attn_norm.weight");
        b.attn_norm_b = get(s + "attn_norm.bias");
        b.attn_q_w = get(s + "attn_q.weight");
        b.attn_q_b = get(s + "attn_q.bias");
        b.attn_k_w = get(s + "attn_k.weight");
        b.attn_k_b = get(s + "attn_k.bias");
        b.attn_v_w = get(s + "attn_v.weight");
        b.attn_v_b = get(s + "attn_v.bias");
        b.attn_o_w = get(s + "attn_o.weight");
        b.attn_o_b = get(s + "attn_o.bias");
        b.mlp_norm_w = get(s + "mlp_norm.weight");
        b.mlp_norm_b = get(s + "mlp_norm.bias");
        b.mlp_fc1_w = get(s + "mlp_fc1.weight");
        b.mlp_fc1_b = get(s + "mlp_fc1.bias");
        b.mlp_fc2_w = get(s + "mlp_fc2.weight");
        b.mlp_fc2_b = get(s + "mlp_fc2.bias");
    }

    // LLM
    model.llm.token_embd_w = get("llm.token_embd.weight");
    model.llm.norm_w = get("llm.norm.weight");
    model.llm.blocks.resize(model.hparams.llm_n_layers);
    for (uint32_t il = 0; il < model.hparams.llm_n_layers; il++) {
        auto& b = model.llm.blocks[il];
        char p[128];
        snprintf(p, sizeof(p), "llm.blk.%u.", il);
        std::string s = p;
        b.attn_norm_w = get(s + "attn_norm.weight");
        b.attn_q_w = get(s + "attn_q.weight");
        b.attn_k_w = get(s + "attn_k.weight");
        b.attn_v_w = get(s + "attn_v.weight");
        b.attn_o_w = get(s + "attn_o.weight");
        b.ffn_norm_w = get(s + "ffn_norm.weight");
        b.ffn_gate_w = get(s + "ffn_gate.weight");
        b.ffn_up_w = get(s + "ffn_up.weight");
        b.ffn_down_w = get(s + "ffn_down.weight");
    }

    return true;
}

// ===========================================================================
// Public C API — load + free
// ===========================================================================

extern "C" struct granite_nle_context_params granite_nle_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true};
}

extern "C" struct granite_nle_context* granite_nle_init_from_file(const char* path,
                                                                  struct granite_nle_context_params params) {
    auto* ctx = new granite_nle_context();
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

    if (!granite_nle_load_model(ctx->model, path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    // Load CTC + LLM tokenizer tables. Both are written by the converter
    // under the standard `granite_nle.ctc.*` and `tokenizer.ggml.*` keys.
    {
        gguf_init_params mp = {true, nullptr};
        gguf_context* g = gguf_init_from_file(path, mp);
        if (g) {
            // CTC vocab — newline-joined string, indices align with logits.
            int kv = gguf_find_key(g, "granite_nle.ctc.vocab");
            if (kv >= 0) {
                std::string s = gguf_get_val_str(g, kv);
                size_t i = 0;
                while (i <= s.size()) {
                    size_t j = s.find('\n', i);
                    if (j == std::string::npos)
                        j = s.size();
                    ctx->ctc_id_to_str.emplace_back(s.substr(i, j - i));
                    if (j == s.size())
                        break;
                    i = j + 1;
                }
                if (params.verbosity >= 1)
                    fprintf(stderr, "granite_nle: loaded %zu CTC vocab entries\n", ctx->ctc_id_to_str.size());
            }

            // LLM tokenizer — same names as base granite_speech for consistency.
            int kt = gguf_find_key(g, "tokenizer.ggml.tokens");
            if (kt >= 0) {
                int n = gguf_get_arr_n(g, kt);
                ctx->id_to_token.resize(n);
                ctx->token_to_id.reserve((size_t)n);
                for (int i = 0; i < n; i++) {
                    std::string tok = gguf_get_arr_str(g, kt, i);
                    ctx->id_to_token[i] = tok;
                    ctx->token_to_id.emplace(std::move(tok), i);
                }
            }
            int km = gguf_find_key(g, "tokenizer.ggml.merges");
            if (km >= 0) {
                int n = gguf_get_arr_n(g, km);
                for (int i = 0; i < n; i++) {
                    ctx->merge_rank[gguf_get_arr_str(g, km, i)] = i;
                }
            }
            gguf_free(g);
        }
    }

    // Parse encoder_layer_indices ("4,8,12,-1") into normalised tuple
    // indices.  HF semantics:
    //   all_hidden_states[0]   = post-input_linear hidden state
    //   all_hidden_states[N]   = output of conformer block N (1-indexed),
    //                            including any self-conditioning residual
    //                            if N == self_conditioning_layer
    //   negative idx           = idx + (n_layers + 1)
    {
        const std::string& s = ctx->model.hparams.proj_encoder_layer_indices;
        const int n_layers = (int)ctx->model.hparams.enc_n_layers;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos)
                j = s.size();
            std::string tok = s.substr(i, j - i);
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
                tok.pop_back();
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
                tok.erase(0, 1);
            if (!tok.empty()) {
                try {
                    int v = std::stoi(tok);
                    if (v < 0)
                        v += n_layers + 1;
                    ctx->enc_layer_indices_parsed.push_back(v);
                } catch (...) {
                    fprintf(stderr, "granite_nle: ignoring unparseable encoder_layer_indices entry '%s'\n",
                            tok.c_str());
                }
            }
            i = j + 1;
        }
        if (params.verbosity >= 1) {
            fprintf(stderr, "granite_nle: encoder_layer_indices=[");
            for (size_t k = 0; k < ctx->enc_layer_indices_parsed.size(); k++)
                fprintf(stderr, "%s%d", k > 0 ? "," : "", ctx->enc_layer_indices_parsed[k]);
            fprintf(stderr, "] (normalised; n_layers=%d)\n", n_layers);
        }
    }

    // Fold batch norm into scale+shift tensors (load-time, once)
    //   y = gamma * (x - mean) / sqrt(var + eps) + beta
    //     = x * scale + shift
    // where scale = gamma / sqrt(var + eps), shift = beta - mean*scale.
    {
        const float eps = 1e-5f;
        const int inner = 2 * (int)ctx->model.hparams.enc_d_model;
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
            ggml_backend_tensor_set(b.conv_bn_w, scale.data(), 0, inner * sizeof(float));
            ggml_backend_tensor_set(b.conv_bn_b, shift.data(), 0, inner * sizeof(float));
            folded++;
        }
        if (params.verbosity >= 1)
            fprintf(stderr, "granite_nle: BN folded for %d encoder layers\n", folded);
    }

    // Precompute per-layer Shaw RPE lookup table (built via
    // core_conformer_ibm::build_shaw_rpe_lookup; shared with granite_speech).
    {
        const int C = (int)ctx->model.hparams.enc_context_size;
        const int max_pos = (int)ctx->model.hparams.enc_max_pos_emb;
        const int hd = (int)ctx->model.hparams.enc_head_dim;
        const int n_layers = (int)ctx->model.hparams.enc_n_layers;

        ctx->rpe_per_layer.assign(n_layers, {});
        for (int il = 0; il < n_layers; il++) {
            ggml_tensor* rpe_w = ctx->model.encoder.blocks[il].attn_rel_pos_w;
            if (!rpe_w)
                continue;
            if (!core_conformer_ibm::build_shaw_rpe_lookup(rpe_w, C, hd, max_pos,
                                                           ctx->rpe_per_layer[il])) {
                fprintf(stderr, "granite_nle: unsupported RPE type %s at layer %d — skipping\n",
                        ggml_type_name(rpe_w->type), il);
                ctx->rpe_per_layer[il].clear();
            }
        }
        if (params.verbosity >= 1)
            fprintf(stderr, "granite_nle: RPE lookup precomputed (%d layers × %d × %d × %d)\n", n_layers, C, C, hd);
    }

    // Create backend scheduler
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
        fprintf(stderr, "granite_nle: loaded %s (enc %u layers, proj %u layers, llm %u layers, vocab %u)\n", path,
                hp.enc_n_layers, hp.proj_n_layers, hp.llm_n_layers, hp.llm_vocab_size);
    }

    return ctx;
}

extern "C" void granite_nle_free(struct granite_nle_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Public C API — pipeline stubs
// ===========================================================================
//
// All forward-pass functions below are STUBS pending implementation.
// They print a one-line warning and return nullptr / 0 so callers can
// build and link against the API while the math is filled in.

extern "C" float* granite_nle_compute_mel(struct granite_nle_context* ctx, const float* samples, int n_samples,
                                          int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    granite_nle_bench_stage _b("compute_mel");
    const int n_fft = 512, win_length = 400, hop = 160, n_mels = 80, n_freqs = n_fft / 2 + 1;

    // Mel filter bank: ne[0]=n_freqs (257), ne[1]=n_mels (80) — HF FreqsMels
    // layout. Written into the GGUF by the converter.
    if (!ctx->model.encoder.mel_filters)
        return nullptr;
    std::vector<float> filt((size_t)n_freqs * n_mels);
    ggml_backend_tensor_get(ctx->model.encoder.mel_filters, filt.data(), 0, filt.size() * sizeof(float));

    // Hann window over win_length=400 samples; core_mel::compute() handles
    // the centre-pad to n_fft=512.
    std::vector<float> hann((size_t)win_length);
    for (int i = 0; i < win_length; i++)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / win_length));

    // Same parameters as base granite_speech: HF / Whisper cluster with
    // log10 + GlobalClipMax normalisation and stacked_frames=2 to fold
    // each consecutive pair of 80-mel rows into a 160-column row.
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
            mn = std::min(mn, stacked[i]);
            mx = std::max(mx, stacked[i]);
            sum += stacked[i];
        }
        fprintf(stderr, "  granite_nle mel: (%d, 160) min=%.4f max=%.4f mean=%.6f\n", T_stacked, mn, mx,
                sum / stacked.size());
    }

    if (out_n_mels)
        *out_n_mels = 160;
    if (out_T_mel)
        *out_T_mel = T_stacked;

    float* result = (float*)malloc(stacked.size() * sizeof(float));
    if (!result)
        return nullptr;
    std::memcpy(result, stacked.data(), stacked.size() * sizeof(float));
    return result;
}

// ===========================================================================
// Encoder helpers (graph-dispatch + CPU primitives) — ported from the base
// granite_speech runtime, which proved their correctness against PyTorch
// at cos_min ≥ 0.999 on JFK across base / 4.1 / PLUS variants. The shapes,
// fused-norm-then-matmul trick, and Shaw block attention all carry over.
// ===========================================================================

// Single-matmul dispatcher (out = W @ x [+ bias]) lives in
// core/cpu_ops.h::matmul. Local wrapper that adapts the call to take
// `ctx` so existing call sites stay readable.
static inline bool nle_run_matmul(granite_nle_context* ctx, float* out, const float* x, int d_in, int T, ggml_tensor* W,
                                  ggml_tensor* bias, int d_out) {
    return core_cpu::matmul(ctx->compute_meta, ctx->sched, out, x, d_in, T, W, bias, d_out);
}

// FFN, fused norm+Q/KV matmul pair, conv module, and Shaw block attention
// helpers all live in core/conformer_ibm.h (shared with granite_speech).
// CPU LayerNorm lives in core/cpu_ops.h::layernorm.

extern "C" float* granite_nle_run_encoder(struct granite_nle_context* ctx, const float* mel, int n_mels, int T_mel,
                                          int* out_T, int* out_dim) {
    if (!ctx || !mel || n_mels != (int)ctx->model.hparams.enc_input_dim)
        return nullptr;
    granite_nle_bench_stage _b("run_encoder");

    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;
    const int n_heads = (int)hp.enc_n_heads;
    const int hd = (int)hp.enc_head_dim;
    const int n_layers = (int)hp.enc_n_layers;
    const int ctx_size = (int)hp.enc_context_size;
    const int self_cond_layer_1based = (int)hp.enc_self_conditioning_layer; // upstream is 1-indexed
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    const int T = T_mel;
    const int remainder = T % ctx_size;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  encoder: per-layer T=%d d=%d layers=%d ctx=%d self_cond=%d\n", T, d, n_layers, ctx_size,
                self_cond_layer_1based);

    // Snapshots for the encoder_layer_indices output. snapshots[k] is the
    // hidden-state copy at the requested HF tuple index. Index 0 is taken
    // after input_linear; index N (1..n_layers) is taken after the post-
    // norm AND any self-conditioning residual at layer N.
    const auto& want = ctx->enc_layer_indices_parsed;
    std::vector<std::vector<float>> snapshots(want.size());

    // Captured at the self-conditioning layer for the BPE auxiliary head:
    // softmax(mid_logits)[:, 0] is the per-frame blank probability, which
    // becomes the importance weight (1 - blank_prob) for posterior_weighted_pool.
    std::vector<float> blank_prob_mid;

    // Input linear: mel (160, T) → hidden (d, T)
    std::vector<float> hidden((size_t)d * T);
    nle_run_matmul(ctx, hidden.data(), mel, n_mels, T, ctx->model.encoder.input_w, ctx->model.encoder.input_b, d);

    for (size_t k = 0; k < want.size(); k++) {
        if (want[k] == 0)
            snapshots[k].assign(hidden.begin(), hidden.end());
    }

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30, mx = -1e30, s = 0;
        for (size_t i = 0; i < (size_t)d * T; i++) {
            mn = std::min(mn, hidden[i]);
            mx = std::max(mx, hidden[i]);
            s += hidden[i];
        }
        fprintf(stderr, "  input_linear: min=%.4f max=%.4f mean=%.6f\n", mn, mx, s / (d * T));
    }

    // Buffers reused across layers.
    std::vector<float> ffn_out((size_t)d * T);
    std::vector<float> Q((size_t)d * T), KV((size_t)d * 2 * T);
    std::vector<float> attn_out((size_t)d * T);
    std::vector<float> conv_out((size_t)d * T);
    std::vector<float> proj_out((size_t)d * T);

    for (int il = 0; il < n_layers; il++) {
        const auto& b = ctx->model.encoder.blocks[il];

        // FFN1 (Macaron half-step)
        core_conformer_ibm::run_ffn(ctx->compute_meta, ctx->sched,
                                    ffn_out.data(), hidden.data(), d, T,
                                    b.ff1_norm_w, b.ff1_norm_b, b.ff1_up_w, b.ff1_up_b,
                                    b.ff1_down_w, b.ff1_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += 0.5f * ffn_out[i];

        // Attention: norm + Q/KV (fused graph) → Shaw block attention on CPU
        core_conformer_ibm::run_norm_matmul_pair(ctx->compute_meta, ctx->sched,
                                                 Q.data(), b.attn_q_w, d,
                                                 KV.data(), b.attn_kv_w, d * 2,
                                                 hidden.data(), d, T,
                                                 b.attn_norm_w, b.attn_norm_b, 1e-5f);

        std::vector<float> K((size_t)d * T), V((size_t)d * T);
        for (int t = 0; t < T; t++) {
            std::memcpy(K.data() + (size_t)t * d, KV.data() + (size_t)t * 2 * d, d * sizeof(float));
            std::memcpy(V.data() + (size_t)t * d, KV.data() + (size_t)t * 2 * d + d, d * sizeof(float));
        }

        core_conformer_ibm::shaw_block_attention_cpu(
            attn_out.data(), Q.data(), K.data(), V.data(),
            ctx->rpe_per_layer[il].empty() ? nullptr : ctx->rpe_per_layer[il].data(),
            T, n_heads, hd, ctx_size, attn_scale, remainder);

        nle_run_matmul(ctx, proj_out.data(), attn_out.data(), d, T, b.attn_out_w, b.attn_out_b, d);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += proj_out[i];

        // Conv module
        core_conformer_ibm::run_conv_module(ctx->compute_meta, ctx->sched,
                                            conv_out.data(), hidden.data(), d, T,
                                            b.conv_norm_w, b.conv_norm_b,
                                            b.conv_up_w, b.conv_up_b,
                                            b.conv_dw_w,
                                            b.conv_bn_w, b.conv_bn_b,
                                            b.conv_down_w, b.conv_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += conv_out[i];

        // FFN2 (Macaron half-step)
        core_conformer_ibm::run_ffn(ctx->compute_meta, ctx->sched,
                                    ffn_out.data(), hidden.data(), d, T,
                                    b.ff2_norm_w, b.ff2_norm_b, b.ff2_up_w, b.ff2_up_b,
                                    b.ff2_down_w, b.ff2_down_b);
        for (size_t i = 0; i < (size_t)d * T; i++)
            hidden[i] += 0.5f * ffn_out[i];

        // Post LayerNorm
        {
            std::vector<float> nw(d), nb(d);
            if (b.post_norm_w)
                ggml_backend_tensor_get(b.post_norm_w, nw.data(), 0, d * sizeof(float));
            if (b.post_norm_b)
                ggml_backend_tensor_get(b.post_norm_b, nb.data(), 0, d * sizeof(float));
            core_cpu::layernorm(hidden.data(), hidden.data(), b.post_norm_w ? nw.data() : nullptr,
                                b.post_norm_b ? nb.data() : nullptr, d, T, 1e-5f);
        }

        // Self-conditioning residual: HF runs `out(hidden) → softmax → out_mid()`
        // on the layer with idx == self_conditioning_layer (1-indexed).
        // Done BEFORE the snapshot append so the snapshot at this index
        // includes the self-conditioning residual (matches HF's
        // all_hidden_states ordering — append happens after the residual).
        if (il + 1 == self_cond_layer_1based && ctx->model.encoder.ctc_out_w && ctx->model.encoder.ctc_mid_w) {
            const int ctc_dim = (int)ctx->model.encoder.ctc_out_w->ne[1];
            std::vector<float> mid_out((size_t)ctc_dim * T), mid_back((size_t)d * T);
            nle_run_matmul(ctx, mid_out.data(), hidden.data(), d, T, ctx->model.encoder.ctc_out_w,
                           ctx->model.encoder.ctc_out_b, ctc_dim);
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
            nle_run_matmul(ctx, mid_back.data(), mid_out.data(), ctc_dim, T, ctx->model.encoder.ctc_mid_w,
                           ctx->model.encoder.ctc_mid_b, d);
            for (size_t i = 0; i < (size_t)d * T; i++)
                hidden[i] += mid_back[i];

            // Capture per-frame blank prob for the posterior-weighted pool
            // used by the BPE auxiliary head after the final layer.
            blank_prob_mid.assign(T, 0.0f);
            for (int t = 0; t < T; t++)
                blank_prob_mid[t] = mid_out[(size_t)t * ctc_dim + 0];
        }

        // Snapshot at HF tuple index il+1.
        for (size_t k = 0; k < want.size(); k++) {
            if (want[k] == il + 1)
                snapshots[k].assign(hidden.begin(), hidden.end());
        }

        if (ctx->params.verbosity >= 2 && (il == 0 || il == 3 || il == 7 || il == n_layers - 1)) {
            float mn = 1e30, mx = -1e30, s = 0;
            for (size_t i = 0; i < (size_t)d * T; i++) {
                mn = std::min(mn, hidden[i]);
                mx = std::max(mx, hidden[i]);
                s += hidden[i];
            }
            fprintf(stderr, "  layer %d/%d: min=%.4f max=%.4f mean=%.6f\n", il + 1, n_layers, mn, mx, s / (d * T));
        }
    }

    // Final CTC head — cache logits for the BPE editing path.
    if (ctx->model.encoder.ctc_out_w) {
        const int ctc_dim = (int)hp.enc_ctc_vocab;
        ctx->last_ctc_logits.assign((size_t)ctc_dim * T, 0.0f);
        nle_run_matmul(ctx, ctx->last_ctc_logits.data(), hidden.data(), d, T, ctx->model.encoder.ctc_out_w,
                       ctx->model.encoder.ctc_out_b, ctc_dim);
        ctx->last_ctc_T = T;
    }

    // BPE auxiliary head: posterior-weighted pool of the final hidden_states
    // with importance = 1 - blank_prob_mid (computed at the self-conditioning
    // layer), then linear → 100353-vocab logits. See modeling_ctc.py
    // posterior_weighted_pool. Pad T to a multiple of pool_window with zeros;
    // pooled rows = ceil(T / pool_window).
    ctx->last_bpe_logits.clear();
    ctx->last_bpe_T = 0;
    if (ctx->model.encoder.bpe_out_w && !blank_prob_mid.empty()) {
        const int pool_window = (int)hp.enc_bpe_pooling_window;
        const int bpe_dim = (int)hp.enc_bpe_vocab;
        const int num_windows = core_ctc::num_windows_for(T, pool_window);

        std::vector<float> importance((size_t)T);
        for (int t = 0; t < T; t++)
            importance[t] = 1.0f - blank_prob_mid[t];

        std::vector<float> pooled((size_t)num_windows * d);
        core_ctc::posterior_weighted_pool(hidden.data(), importance.data(),
                                          T, d, pool_window, pooled.data());

        ctx->last_bpe_logits.assign((size_t)bpe_dim * num_windows, 0.0f);
        nle_run_matmul(ctx, ctx->last_bpe_logits.data(), pooled.data(), d, num_windows, ctx->model.encoder.bpe_out_w,
                       ctx->model.encoder.bpe_out_b, bpe_dim);
        ctx->last_bpe_T = num_windows;
    }

    // Build the final concatenated encoder output.
    //   per-frame layout: [snapshot_0[d], snapshot_1[d], ..., snapshot_K-1[d]]
    //   total feature dim = K * d, where K = encoder_layer_indices.size()
    // If encoder_layer_indices is empty, fall back to last hidden state.
    if (snapshots.empty()) {
        size_t total = (size_t)T * d;
        float* result = (float*)malloc(total * sizeof(float));
        if (!result)
            return nullptr;
        std::memcpy(result, hidden.data(), total * sizeof(float));
        if (out_T)
            *out_T = T;
        if (out_dim)
            *out_dim = d;
        return result;
    }

    const int K = (int)snapshots.size();
    const int wide_d = K * d;
    std::vector<float> wide((size_t)wide_d * T);
    for (int t = 0; t < T; t++) {
        float* dst = wide.data() + (size_t)t * wide_d;
        for (int k = 0; k < K; k++) {
            if (snapshots[k].empty()) {
                std::memset(dst + (size_t)k * d, 0, (size_t)d * sizeof(float));
            } else {
                std::memcpy(dst + (size_t)k * d, snapshots[k].data() + (size_t)t * d, (size_t)d * sizeof(float));
            }
        }
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  encoder: concat output (T=%d, %d × %d = %d)\n", T, K, d, wide_d);

    size_t total = (size_t)wide_d * T;
    float* result = (float*)malloc(total * sizeof(float));
    if (!result)
        return nullptr;
    std::memcpy(result, wide.data(), total * sizeof(float));
    if (out_T)
        *out_T = T;
    if (out_dim)
        *out_dim = wide_d;
    return result;
}

extern "C" const float* granite_nle_last_ctc_logits(struct granite_nle_context* ctx, int* out_T, int* out_vocab) {
    if (!ctx || ctx->last_ctc_logits.empty())
        return nullptr;
    if (out_T)
        *out_T = ctx->last_ctc_T;
    if (out_vocab)
        *out_vocab = (int)ctx->model.hparams.enc_ctc_vocab;
    return ctx->last_ctc_logits.data();
}

extern "C" const float* granite_nle_last_bpe_logits(struct granite_nle_context* ctx, int* out_T, int* out_vocab) {
    if (!ctx || ctx->last_bpe_logits.empty())
        return nullptr;
    if (out_T)
        *out_T = ctx->last_bpe_T;
    if (out_vocab)
        *out_vocab = (int)ctx->model.hparams.enc_bpe_vocab;
    return ctx->last_bpe_logits.data();
}

// ===========================================================================
// Projector — windowed simplified Q-Former
// ===========================================================================
//
// Two-pass implementation:
//   pass A — full sequence:
//             per-encoder-layer LayerNorm + concat → layer_proj → GELU
//             input  (T, K*D) F32       (K=4, D=1024 → 4096-wide)
//             output (T, hidden_size)  F32 (hidden_size=2048)
//   pass B — per-block (block_size=15 frames) graph, repeated nblocks times:
//             query_embeds = query_template + mean_pool(block, downsample=5)
//             enc_kv       = block + window_positions
//             for each of 2 Q-Former layers:
//                cur = q + cross_attn(attn_norm(q), enc_kv)
//                cur = cur + mlp(mlp_norm(cur))   (fc1 → SiLU → fc2)
//             out = out_linear(out_norm(cur))     (hidden_size → llm_dim)
//
// All compute lives inside ggml graphs so the GPU backend can pick them up
// (the small per-block dispatch is the same pattern used by the base
// granite_speech projector and is fast enough in practice).

// Per-window simplified Q-Former forward + pass-A LayerNorm/GELU helper now
// live in core/qformer.h. The wrappers below build the core_qformer structs
// from this TU's projector weights and delegate.

static bool nle_proj_layer_proj(granite_nle_context* ctx, std::vector<float>& out, const float* enc, int T, int wide_d,
                                int hidden) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int K = (int)hp.proj_num_encoder_layers;
    const int D = (int)hp.proj_encoder_dim;
    if (K * D != wide_d) {
        fprintf(stderr, "granite_nle: proj input dim %d != K*D=%d\n", wide_d, K * D);
        return false;
    }
    return core_qformer::run_layer_proj(ctx->compute_meta, ctx->sched, out, enc, T, wide_d, hidden, K, D,
                                        hp.proj_layernorm_eps, m.projector.layer_norm_w, m.projector.layer_norm_b,
                                        m.projector.layer_proj_w, m.projector.layer_proj_b);
}

static ggml_cgraph* nle_proj_build_block(granite_nle_context* ctx) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_layers = (int)hp.proj_n_layers;

    core_qformer::Hparams qhp = {};
    qhp.hidden = (int)hp.proj_hidden_size;
    qhp.llm_d = (int)hp.proj_llm_dim;
    qhp.n_heads = (int)hp.proj_n_heads;
    qhp.n_layers = n_layers;
    qhp.block_size = (int)hp.proj_block_size;
    qhp.downsample = (int)hp.proj_downsample_rate;
    qhp.eps = hp.proj_layernorm_eps;

    std::vector<core_qformer::BlockWeights> blocks(n_layers);
    for (int il = 0; il < n_layers; il++) {
        const auto& b = m.projector.blocks[il];
        blocks[il] = {b.attn_norm_w, b.attn_norm_b, b.attn_q_w,   b.attn_q_b,   b.attn_k_w,   b.attn_k_b,
                      b.attn_v_w,    b.attn_v_b,    b.attn_o_w,   b.attn_o_b,   b.mlp_norm_w, b.mlp_norm_b,
                      b.mlp_fc1_w,   b.mlp_fc1_b,   b.mlp_fc2_w,  b.mlp_fc2_b};
    }

    core_qformer::OuterWeights outer = {m.projector.query,       m.projector.window_pos,  m.projector.out_norm_w,
                                        m.projector.out_norm_b,  m.projector.out_linear_w, m.projector.out_linear_b};

    return core_qformer::build_block(ctx->compute_meta, blocks, outer, qhp);
}

extern "C" float* granite_nle_run_projector(struct granite_nle_context* ctx, const float* enc_concat, int T, int dim,
                                            int* out_T, int* out_dim) {
    if (!ctx || !enc_concat || T <= 0 || dim <= 0)
        return nullptr;
    granite_nle_bench_stage _b("run_projector");

    const auto& hp = ctx->model.hparams;
    const int hidden = (int)hp.proj_hidden_size;
    const int llm_d = (int)hp.proj_llm_dim;
    const int block_size = (int)hp.proj_block_size;
    const int down = (int)hp.proj_downsample_rate;
    const int q_len = block_size / down;

    if (!ctx->model.projector.layer_proj_w || !ctx->model.projector.query || !ctx->model.projector.window_pos ||
        !ctx->model.projector.out_linear_w) {
        fprintf(stderr, "granite_nle: projector tensors not loaded\n");
        return nullptr;
    }

    // pass A: layer-norm-per-layer + concat + layer_proj + GELU
    std::vector<float> after_proj;
    if (!nle_proj_layer_proj(ctx, after_proj, enc_concat, T, dim, hidden)) {
        fprintf(stderr, "granite_nle: projector pass A failed\n");
        return nullptr;
    }

    // Pad to multiple of block_size on the time axis (zeros at the tail).
    const int nblocks = (T + block_size - 1) / block_size;
    const int T_pad = nblocks * block_size;
    std::vector<float> padded((size_t)T_pad * hidden, 0.0f);
    std::memcpy(padded.data(), after_proj.data(), (size_t)T * hidden * sizeof(float));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  projector: T=%d nblocks=%d q_len=%d hidden=%d → %d × %d\n", T, nblocks, q_len, hidden,
                nblocks * q_len, llm_d);

    // pass B: one Q-Former graph per block. Same shape every block, so we
    // could batch via ne[3]; per-block keeps it simple and is fast enough.
    const int total = nblocks * q_len;
    std::vector<float> all_out((size_t)total * llm_d, 0.0f);

    for (int blk = 0; blk < nblocks; blk++) {
        const float* blk_data = padded.data() + (size_t)blk * block_size * hidden;
        ggml_cgraph* gf = nle_proj_build_block(ctx);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
            return nullptr;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "qformer_blk_in"), blk_data, 0,
                                (size_t)hidden * block_size * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
            return nullptr;
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "qformer_blk_out"),
                                all_out.data() + (size_t)blk * q_len * llm_d, 0,
                                (size_t)q_len * llm_d * sizeof(float));
    }

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, s = 0;
        for (size_t i = 0; i < all_out.size(); i++) {
            if (all_out[i] < mn)
                mn = all_out[i];
            if (all_out[i] > mx)
                mx = all_out[i];
            s += all_out[i];
        }
        fprintf(stderr, "  projector out: (%d, %d) min=%.6f max=%.6f mean=%.6f\n", total, llm_d, mn, mx,
                s / all_out.size());
    }

    float* result = (float*)malloc(all_out.size() * sizeof(float));
    if (!result)
        return nullptr;
    std::memcpy(result, all_out.data(), all_out.size() * sizeof(float));
    if (out_T)
        *out_T = total;
    if (out_dim)
        *out_dim = llm_d;
    return result;
}

// ===========================================================================
// Non-causal LLM editing forward (Granite 4.0-1B, 40 layers, GQA 16/4)
// ===========================================================================
//
// Input layout: [audio_embs (D, n_audio), text_embs (D, n_text)] concatenated
// along time. The text_embs come from `embed_tokens(text_ids_with_slots)`.
// Audio is pre-divided by `embedding_multiplier` so the LLM's uniform
// embedding scale recovers the original projector output, while text gets
// the standard µP scale-up.
//
// Every layer runs flash-attn-ext WITHOUT a causal mask — this is the
// distinguishing feature of the NAR pipeline.
//
// The graph slices the last n_text positions before the LM head so the
// output is exactly (vocab, n_text). LM head is tied to embed_tokens
// (`tie_word_embeddings = True` in the NAR config), so we matmul against
// the same `token_embd_w` tensor used for the lookup.
//
// The 40-block backbone (RMSNorm + attn + RMSNorm + SwiGLU + residuals +
// final RMSNorm) lives in core/granite_llm.h, shared with granite_speech;
// we hand it a pre-built (audio | text_embs) tensor and an is_causal=false
// flag to select the non-causal flash-attn path.

static ggml_cgraph* nle_build_llm(granite_nle_context* ctx, int n_audio, int n_text) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int d = (int)hp.llm_d_model;
    const int n_layers = (int)hp.llm_n_layers;
    const int N = n_audio + n_text;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* audio_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_audio);
    ggml_set_name(audio_in, "audio_embs");
    ggml_set_input(audio_in);

    ggml_tensor* text_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_text);
    ggml_set_name(text_ids, "text_ids");
    ggml_set_input(text_ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // The caller passes audio_embs already in LLM-ready form: the upstream
    // `_build_flat_llm_inputs` divides projector output by
    // embedding_multiplier BEFORE concatenation so the uniform downstream
    // `inputs_embeds * embedding_multiplier` recovers the original
    // projector output for audio while still scaling text embeds by 12×.
    // Mirror that contract — caller does the divide; we do not divide
    // again here.
    //
    // Embed text_ids via tied token_embd_w (output forced to F32), then
    // concat [audio | text] along time before handing it to the shared
    // 40-block backbone.
    ggml_tensor* text_embs = ggml_get_rows(ctx0, m.llm.token_embd_w, text_ids);
    if (text_embs->type != GGML_TYPE_F32)
        text_embs = ggml_cast(ctx0, text_embs, GGML_TYPE_F32);
    ggml_tensor* inputs_embeds = ggml_concat(ctx0, audio_in, text_embs, 1);

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
        blocks[il] = {b.attn_norm_w, b.attn_q_w,    b.attn_k_w, b.attn_v_w,    b.attn_o_w,
                      b.ffn_norm_w,  b.ffn_gate_w, b.ffn_up_w, b.ffn_down_w};
    }

    ggml_tensor* cur = core_granite_llm::build_decoder(ctx0, gf, inputs_embeds, positions,
                                                       /*causal_mask*/ nullptr,
                                                       /*kv_k*/ nullptr, /*kv_v*/ nullptr, /*n_past*/ 0,
                                                       blocks, m.llm.norm_w, llm_hp,
                                                       /*is_causal*/ false);

    // Slice the text portion: positions [n_audio, N) → (d, n_text).
    ggml_tensor* text_h = ggml_view_2d(ctx0, cur, d, n_text, cur->nb[1], (size_t)n_audio * cur->nb[1]);
    text_h = ggml_cont(ctx0, text_h);

    // LM head, tied to token_embd_w. Output is (vocab, n_text).
    //
    // Note: NO `1/logits_scaling` divide here. The upstream NLE forward
    // (modeling_nle.py L242) calls `self.llm.lm_head(...)` directly,
    // which is just an nn.Linear — the µP `/logits_scaling` divide that
    // `GraniteForCausalLM.forward` would normally apply is bypassed.
    // Argmax (and therefore the slot-decoded transcript) is identical
    // either way; the raw logit values must match the bypassed path so
    // diff comparisons stay tight.
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.llm.token_embd_w, text_h);

    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);
    ggml_free(ctx0);
    return gf;
}

extern "C" float* granite_nle_run_llm_editing(struct granite_nle_context* ctx, const float* audio_embs, int n_audio,
                                              const int32_t* text_ids, int n_text, int* out_n, int* out_vocab) {
    if (!ctx || !audio_embs || !text_ids || n_audio <= 0 || n_text <= 0)
        return nullptr;
    granite_nle_bench_stage _b("run_llm_editing");

    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_d_model;
    const int vocab = (int)hp.llm_vocab_size;
    const int N = n_audio + n_text;

    if (!ctx->model.llm.token_embd_w || !ctx->model.llm.norm_w) {
        fprintf(stderr, "granite_nle: LLM weights not loaded\n");
        return nullptr;
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  llm_editing: n_audio=%d n_text=%d N=%d → (vocab=%d, %d)\n", n_audio, n_text, N, vocab,
                n_text);

    std::vector<int32_t> positions(N);
    for (int i = 0; i < N; i++)
        positions[i] = i;

    ggml_cgraph* gf = nle_build_llm(ctx, n_audio, n_text);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf))
        return nullptr;

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "audio_embs"), audio_embs, 0,
                            (size_t)d * n_audio * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "text_ids"), text_ids, 0, (size_t)n_text * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "positions"), positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS)
        return nullptr;

    float* result = (float*)malloc((size_t)vocab * n_text * sizeof(float));
    if (!result)
        return nullptr;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), result, 0, (size_t)vocab * n_text * sizeof(float));

    if (ctx->params.verbosity >= 2) {
        float mn = 1e30f, mx = -1e30f, s = 0;
        for (int i = 0; i < n_text * vocab; i++) {
            if (result[i] < mn)
                mn = result[i];
            if (result[i] > mx)
                mx = result[i];
            s += result[i];
        }
        fprintf(stderr, "  editing logits: (%d, %d) min=%.4f max=%.4f mean=%.6f\n", n_text, vocab, mn, mx,
                s / (n_text * vocab));
    }

    if (const char* p = std::getenv("GRANITE_NLE_EDIT_DUMP")) {
        FILE* fp = std::fopen(p, "wb");
        if (fp) {
            std::fwrite(result, sizeof(float), (size_t)vocab * n_text, fp);
            std::fclose(fp);
            fprintf(stderr, "  edit logits dumped to %s\n", p);
        }
    }

    if (out_n)
        *out_n = n_text;
    if (out_vocab)
        *out_vocab = vocab;
    return result;
}

extern "C" char* granite_nle_transcribe(struct granite_nle_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    granite_nle_bench_stage _b("transcribe");

    const auto& hp = ctx->model.hparams;
    const int down = (int)hp.proj_downsample_rate;
    const int eos_id = (int)hp.eos_token_id;

    // 1. Mel
    int n_mels = 0, T_mel = 0;
    float* mel = granite_nle_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    if (!mel) {
        fprintf(stderr, "granite_nle: compute_mel failed\n");
        return nullptr;
    }

    // 2. Encoder (also fills last_bpe_logits)
    int enc_T = 0, enc_dim = 0;
    float* enc_out = granite_nle_run_encoder(ctx, mel, n_mels, T_mel, &enc_T, &enc_dim);
    free(mel);
    if (!enc_out) {
        fprintf(stderr, "granite_nle: encoder failed\n");
        return nullptr;
    }

    // 3+4. BPE-CTC greedy decode → LLM token IDs.
    // Upstream _decode_bpe_ctc_greedy: argmax → unique_consecutive → drop
    // blanks (id 0) → shift to LLM token IDs (id - 1). Lifted to
    // core_ctc::greedy_decode_with_blank.
    std::string ctc_text;
    {
        const std::vector<int32_t> ids = core_ctc::greedy_decode_with_blank(
            ctx->last_bpe_logits.data(), ctx->last_bpe_T, (int)hp.enc_bpe_vocab,
            /*blank_id=*/0, /*shift=*/-1);
        if (!ids.empty())
            ctc_text = core_bpe::detokenize(ctx->id_to_token, ids.data(), ids.size());
        else
            ctc_text = " ";
        // Match upstream: strip + lowercase + fall back to " " if empty.
        size_t a = ctc_text.find_first_not_of(" \t\n\r");
        size_t b2 = ctc_text.find_last_not_of(" \t\n\r");
        if (a == std::string::npos)
            ctc_text = " ";
        else
            ctc_text = ctc_text.substr(a, b2 - a + 1);
        for (char& c : ctc_text)
            if (c >= 'A' && c <= 'Z')
                c = (char)(c + ('a' - 'A'));
        if (ctc_text.empty())
            ctc_text = " ";
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "  transcribe: ctc_text=%.200s\n", ctc_text.c_str());

    // 6. Re-tokenize the CTC text with the LLM tokenizer.
    auto llm_ids = core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, ctc_text);

    // 7. add_insertion_slots: total_len = max(2*n+1, 8); positions
    //    [eos, t0, eos, t1, ..., t_{n-1}, eos, ..., eos].
    const int n = (int)llm_ids.size();
    const int total_len = std::max(2 * n + 1, 8);
    std::vector<int32_t> text_ids((size_t)total_len, (int32_t)eos_id);
    for (int i = 0; i < n; i++)
        text_ids[(size_t)(2 * i + 1)] = llm_ids[(size_t)i];

    // 8. Projector → divide by embedding_multiplier, slice to n_audio_kept.
    int proj_T = 0, proj_dim = 0;
    float* proj_out = granite_nle_run_projector(ctx, enc_out, enc_T, enc_dim, &proj_T, &proj_dim);
    free(enc_out);
    if (!proj_out) {
        fprintf(stderr, "granite_nle: projector failed\n");
        return nullptr;
    }
    const int n_audio_kept = enc_T / down;
    const int llm_d = (int)hp.llm_d_model;
    if (n_audio_kept > proj_T) {
        free(proj_out);
        fprintf(stderr, "granite_nle: n_audio_kept=%d > proj_T=%d\n", n_audio_kept, proj_T);
        return nullptr;
    }
    std::vector<float> audio_embs((size_t)n_audio_kept * llm_d);
    const float scale = 1.0f / hp.embedding_multiplier;
    for (size_t i = 0; i < audio_embs.size(); i++)
        audio_embs[i] = proj_out[i] * scale;
    free(proj_out);

    // 9. LLM editing forward.
    int edit_n = 0, edit_V = 0;
    float* edit_logits =
        granite_nle_run_llm_editing(ctx, audio_embs.data(), n_audio_kept, text_ids.data(), total_len, &edit_n, &edit_V);
    if (!edit_logits) {
        fprintf(stderr, "granite_nle: llm_editing failed\n");
        return nullptr;
    }

    // 10. Slot decode: per-row argmax → unique_consecutive → drop EOS → detokenize.
    std::vector<int32_t> slot_argmax((size_t)edit_n);
    for (int t = 0; t < edit_n; t++) {
        const float* row = edit_logits + (size_t)t * edit_V;
        int best = 0;
        float bestv = row[0];
        for (int v = 1; v < edit_V; v++) {
            if (row[v] > bestv) {
                bestv = row[v];
                best = v;
            }
        }
        slot_argmax[t] = best;
    }
    free(edit_logits);

    std::vector<int32_t> uniq;
    for (size_t i = 0; i < slot_argmax.size(); i++) {
        if (i == 0 || slot_argmax[i] != slot_argmax[i - 1])
            uniq.push_back(slot_argmax[i]);
    }
    std::vector<int32_t> kept;
    kept.reserve(uniq.size());
    for (int32_t id : uniq)
        if (id != eos_id)
            kept.push_back(id);

    std::string text;
    if (!kept.empty())
        text = core_bpe::detokenize(ctx->id_to_token, kept.data(), kept.size());

    char* result = (char*)malloc(text.size() + 1);
    if (!result)
        return nullptr;
    std::memcpy(result, text.data(), text.size());
    result[text.size()] = '\0';
    return result;
}

extern "C" int32_t* granite_nle_tokenize(struct granite_nle_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text) {
        if (out_n)
            *out_n = 0;
        return nullptr;
    }
    auto ids = core_bpe::tokenize_simple(ctx->token_to_id, ctx->merge_rank, std::string(text));
    int32_t* arr = (int32_t*)malloc(ids.size() * sizeof(int32_t));
    if (!arr) {
        if (out_n)
            *out_n = 0;
        return nullptr;
    }
    std::memcpy(arr, ids.data(), ids.size() * sizeof(int32_t));
    if (out_n)
        *out_n = (int)ids.size();
    return arr;
}

extern "C" char* granite_nle_detokenize(struct granite_nle_context* ctx, const int32_t* ids, int n) {
    if (!ctx || !ids || n <= 0)
        return nullptr;
    std::string out = core_bpe::detokenize(ctx->id_to_token, ids, (size_t)n);
    char* r = (char*)malloc(out.size() + 1);
    if (!r)
        return nullptr;
    std::memcpy(r, out.data(), out.size());
    r[out.size()] = '\0';
    return r;
}

extern "C" char* granite_nle_ctc_decode(struct granite_nle_context* ctx, const int32_t* ids, int n) {
    if (!ctx || !ids || n <= 0)
        return nullptr;
    std::string out;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= (int32_t)ctx->ctc_id_to_str.size())
            continue;
        out += ctx->ctc_id_to_str[(size_t)id];
    }
    char* r = (char*)malloc(out.size() + 1);
    if (!r)
        return nullptr;
    std::memcpy(r, out.data(), out.size());
    r[out.size()] = '\0';
    return r;
}

extern "C" int granite_nle_eos_token_id(struct granite_nle_context* ctx) {
    return ctx ? (int)ctx->model.hparams.eos_token_id : -1;
}
