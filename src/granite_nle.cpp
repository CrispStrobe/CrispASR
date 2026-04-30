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
// FFT helpers (radix-2 Cooley-Tukey, recursive, in-place). Identical to
// granite_speech's FFT — promoted to a shared interface in a future cleanup.
// ===========================================================================

static void granite_nle_fft(float* in, int N, float* out) {
    if (N <= 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    if (N % 2 != 0) {
        for (int k = 0; k < N; k++) {
            double re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                double a = -2.0 * M_PI * k * n / N;
                re += in[n] * cos(a);
                im += in[n] * sin(a);
            }
            out[2 * k] = (float)re;
            out[2 * k + 1] = (float)im;
        }
        return;
    }
    int half = N / 2;
    std::vector<float> even(half), odd(half);
    for (int i = 0; i < half; i++) {
        even[i] = in[2 * i];
        odd[i] = in[2 * i + 1];
    }
    std::vector<float> E(2 * half), O(2 * half);
    granite_nle_fft(even.data(), half, E.data());
    granite_nle_fft(odd.data(), half, O.data());
    for (int k = 0; k < half; k++) {
        double a = -2.0 * M_PI * k / N;
        float wr = (float)cos(a), wi = (float)sin(a);
        float tre = wr * O[2 * k] - wi * O[2 * k + 1];
        float tim = wr * O[2 * k + 1] + wi * O[2 * k];
        out[2 * k] = E[2 * k] + tre;
        out[2 * k + 1] = E[2 * k + 1] + tim;
        out[2 * (k + half)] = E[2 * k] - tre;
        out[2 * (k + half) + 1] = E[2 * k + 1] - tim;
    }
}

static void granite_nle_fft_wrapper(const float* in, int N, float* out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size() < 4 * N)
        scratch_in.assign((size_t)4 * N, 0.0f);
    if ((int)scratch_out.size() < 8 * N)
        scratch_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    granite_nle_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}

// ===========================================================================
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
                                     granite_nle_fft_wrapper, p, T_stacked);
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

extern "C" float* granite_nle_run_encoder(struct granite_nle_context* ctx, const float* /*mel*/, int /*n_mels*/,
                                          int /*T_mel*/, int* out_T, int* out_dim) {
    if (!ctx)
        return nullptr;
    fprintf(stderr, "granite_nle: run_encoder not yet implemented\n");
    if (out_T)
        *out_T = 0;
    if (out_dim)
        *out_dim = 0;
    return nullptr;
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

extern "C" float* granite_nle_run_projector(struct granite_nle_context* ctx, const float* /*enc_concat*/, int /*T*/,
                                            int /*dim*/, int* out_T, int* out_dim) {
    if (!ctx)
        return nullptr;
    fprintf(stderr, "granite_nle: run_projector not yet implemented\n");
    if (out_T)
        *out_T = 0;
    if (out_dim)
        *out_dim = 0;
    return nullptr;
}

extern "C" float* granite_nle_run_llm_editing(struct granite_nle_context* ctx, const float* /*audio_embs*/,
                                              int /*n_audio*/, const int32_t* /*text_ids*/, int /*n_text*/, int* out_n,
                                              int* out_vocab) {
    if (!ctx)
        return nullptr;
    fprintf(stderr, "granite_nle: run_llm_editing not yet implemented\n");
    if (out_n)
        *out_n = 0;
    if (out_vocab)
        *out_vocab = 0;
    return nullptr;
}

extern "C" char* granite_nle_transcribe(struct granite_nle_context* ctx, const float* /*samples*/, int /*n_samples*/) {
    if (!ctx)
        return nullptr;
    fprintf(stderr, "granite_nle: transcribe not yet implemented\n");
    return nullptr;
}

extern "C" int32_t* granite_nle_tokenize(struct granite_nle_context* ctx, const char* text, int* out_n) {
    if (!ctx || !text) {
        if (out_n)
            *out_n = 0;
        return nullptr;
    }
    // Stub: defer to core_bpe once wired up. For now return empty so the
    // CLI can detect "unsupported" and skip.
    if (out_n)
        *out_n = 0;
    return nullptr;
}

extern "C" char* granite_nle_detokenize(struct granite_nle_context* ctx, const int32_t* ids, int n) {
    if (!ctx || !ids || n <= 0)
        return nullptr;
    std::string out;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= (int32_t)ctx->id_to_token.size())
            continue;
        out += ctx->id_to_token[(size_t)id];
    }
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
