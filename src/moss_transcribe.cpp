// moss_transcribe.cpp — MOSS-Transcribe-preview-2B ggml runtime
//
// Stock Qwen3-Omni-MoE audio encoder (conv2d stem + 32 pre-LN layers with
// windowed attention + proj head) + Gated-MLP adapter + Qwen3-1.7B LLM with
// audio-token injection. See moss_transcribe.h for the architecture summary.
//
// Adapted from the sibling moss_audio.cpp (MOSS-Audio-4B): DeepStack removed;
// encoder head is conv_out(no bias)+ln_post+proj1+gelu+proj2; the encoder uses
// block-diagonal windowed attention (window_aftercnn = 8×13 cnn frames) over
// the full concatenated sequence rather than independent per-chunk attention;
// LM retargeted to Qwen3-1.7B; lm_head tied to the token embedding table.

#include "moss_transcribe.h"

#include "core/beam_decode.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Core helpers
#include "core/gguf_loader.h"
#include "core/ffn.h"
#include "core/attention.h"
#include "core/mel.h"
#include "core/bpe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Bench instrumentation — `MOSS_TRANSCRIBE_BENCH=1` for per-stage timings.
// ===========================================================================

static bool moss_transcribe_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("MOSS_TRANSCRIBE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct moss_transcribe_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit moss_transcribe_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~moss_transcribe_bench_stage() {
        if (!moss_transcribe_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  moss_transcribe_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct moss_transcribe_hparams {
    // Mel front-end (README MelConfig: 128 bins, n_fft 400, hop 160)
    uint32_t n_mels = 128;
    uint32_t n_fft = 400;
    uint32_t hop_length = 160;
    uint32_t sample_rate = 16000;

    // Audio encoder (qwen3_omni_moe_audio_encoder)
    uint32_t enc_layers = 32;
    uint32_t enc_d_model = 1280;
    uint32_t enc_n_heads = 20;
    uint32_t enc_head_dim = 64; // 1280 / 20
    uint32_t enc_ffn_dim = 5120;
    uint32_t enc_ds_hidden = 480; // downsample_hidden_size (Conv2d channels)
    uint32_t enc_max_pos = 1500;
    uint32_t enc_output_dim = 2048; // proj2 output (adapter input)
    uint32_t n_window = 50;         // chunk = n_window*2 = 100 mel frames
    uint32_t n_window_infer = 800;  // attention window groups n_window_infer/(2*n_window) chunks
    float enc_ln_eps = 1e-5f;

    // Adapter
    uint32_t adapter_hidden = 8192;

    // LLM (Qwen3-1.7B)
    uint32_t llm_hidden = 2048;
    uint32_t llm_layers = 28;
    uint32_t llm_n_heads = 16;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim = 128;
    uint32_t llm_ff_dim = 6144;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos = 40960;
    float llm_rope_theta = 1000000.0f;
    float llm_rms_eps = 1e-6f;

    // Special tokens
    uint32_t bos_token_id = 151643;
    uint32_t eos_token_id = 151645;   // <|im_end|>
    uint32_t start_token_id = 151644; // <|im_start|>
    uint32_t audio_start_id = 151669; // <|audio_bos|>
    uint32_t audio_end_id = 151670;   // <|audio_eos|>
    uint32_t audio_placeholder_id = 0;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct moss_transcribe_enc_block {
    // Pre-LN self-attention (Whisper-style; q/k/v/out all have bias)
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr, *attn_k_b = nullptr;
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_o_w = nullptr, *attn_o_b = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_fc1_w = nullptr, *ffn_fc1_b = nullptr;
    ggml_tensor *ffn_fc2_w = nullptr, *ffn_fc2_b = nullptr;
};

struct moss_transcribe_encoder {
    // Conv stem: 3 × Conv2d(stride=2)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;
    // conv_out: Linear(ds_hidden*16 = 7680, d_model = 1280), NO bias
    ggml_tensor* conv_out_w = nullptr;
    // ln_post (post-layer LayerNorm)
    ggml_tensor *ln_post_w = nullptr, *ln_post_b = nullptr;
    // proj1 (d→d) + proj2 (d→output_dim), both with bias
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr;
    ggml_tensor *proj2_w = nullptr, *proj2_b = nullptr;
    // Transformer layers
    std::vector<moss_transcribe_enc_block> blocks;
};

// GatedMLP (SwiGLU): gate_proj + up_proj + down_proj
struct moss_transcribe_gated_mlp {
    ggml_tensor* gate_w = nullptr;
    ggml_tensor* up_w = nullptr;
    ggml_tensor* down_w = nullptr;
};

struct moss_transcribe_llm_block {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_o_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct moss_transcribe_llm {
    ggml_tensor* embed_w = nullptr;
    std::vector<moss_transcribe_llm_block> blocks;
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* lm_head_w = nullptr; // tied → == embed_w
};

struct moss_transcribe_model {
    moss_transcribe_hparams hparams;
    moss_transcribe_encoder enc;
    moss_transcribe_gated_mlp adapter;
    moss_transcribe_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal position embeddings (max_pos × d_model), precomputed
    std::vector<float> audio_pe;
};

struct moss_transcribe_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct moss_transcribe_context {
    moss_transcribe_context_params params;
    moss_transcribe_model model;
    moss_transcribe_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    // KV cache for LLM
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr;
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_used = 0;

    // Cached conv-stem graph (fixed 100-frame chunk).
    ggml_cgraph* cached_conv_gf = nullptr;
    ggml_context* cached_conv_ctx = nullptr;
    std::vector<uint8_t> cached_conv_meta;

    int n_threads = 4;
    std::string model_path;
    int beam_size = 1;
};

// ===========================================================================
// Helpers
// ===========================================================================

static ggml_tensor* try_get(moss_transcribe_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}
static ggml_tensor* require(moss_transcribe_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "moss_transcribe");
}

// Conv2d stride-2 downsampling output length (kernel 3, pad 1).
static int conv_out_len(int L) {
    return (L - 1) / 2 + 1;
}

// HF processing_Moss._get_feat_extract_output_lengths: mel frames → audio tokens.
static int feat_output_len(int L) {
    int leave = L % 100;
    int feat = (leave - 1) / 2 + 1;
    return ((feat - 1) / 2 + 1 - 1) / 2 + 1 + (L / 100) * 13;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool moss_transcribe_load_model(moss_transcribe_model& model, moss_transcribe_vocab& vocab, const char* path,
                                       ggml_backend_t backend) {
    // ---- pass 1: metadata + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.n_mels = core_gguf::kv_u32(gctx, "moss_transcribe.enc.num_mel_bins", hp.n_mels);
        hp.enc_layers = core_gguf::kv_u32(gctx, "moss_transcribe.enc.encoder_layers", hp.enc_layers);
        hp.enc_d_model = core_gguf::kv_u32(gctx, "moss_transcribe.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "moss_transcribe.enc.encoder_attention_heads", hp.enc_n_heads);
        hp.enc_ffn_dim = core_gguf::kv_u32(gctx, "moss_transcribe.enc.encoder_ffn_dim", hp.enc_ffn_dim);
        hp.enc_ds_hidden = core_gguf::kv_u32(gctx, "moss_transcribe.enc.downsample_hidden_size", hp.enc_ds_hidden);
        hp.enc_max_pos = core_gguf::kv_u32(gctx, "moss_transcribe.enc.max_source_positions", hp.enc_max_pos);
        hp.enc_output_dim = core_gguf::kv_u32(gctx, "moss_transcribe.enc.output_dim", hp.enc_output_dim);
        hp.n_window = core_gguf::kv_u32(gctx, "moss_transcribe.enc.n_window", hp.n_window);
        hp.n_window_infer = core_gguf::kv_u32(gctx, "moss_transcribe.enc.n_window_infer", hp.n_window_infer);
        hp.enc_ln_eps = core_gguf::kv_f32(gctx, "moss_transcribe.enc.layer_norm_eps", hp.enc_ln_eps);
        hp.enc_head_dim = hp.enc_d_model / hp.enc_n_heads;

        hp.adapter_hidden = core_gguf::kv_u32(gctx, "moss_transcribe.adapter.hidden_size", hp.adapter_hidden);

        hp.llm_hidden = core_gguf::kv_u32(gctx, "moss_transcribe.llm.hidden_size", hp.llm_hidden);
        hp.llm_layers = core_gguf::kv_u32(gctx, "moss_transcribe.llm.num_layers", hp.llm_layers);
        hp.llm_n_heads = core_gguf::kv_u32(gctx, "moss_transcribe.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "moss_transcribe.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim = core_gguf::kv_u32(gctx, "moss_transcribe.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim = core_gguf::kv_u32(gctx, "moss_transcribe.llm.intermediate_size", hp.llm_ff_dim);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "moss_transcribe.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos = core_gguf::kv_u32(gctx, "moss_transcribe.llm.max_position_embeddings", hp.llm_max_pos);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "moss_transcribe.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps = core_gguf::kv_f32(gctx, "moss_transcribe.llm.rms_norm_eps", hp.llm_rms_eps);

        hp.bos_token_id = core_gguf::kv_u32(gctx, "moss_transcribe.bos_token_id", hp.bos_token_id);
        hp.eos_token_id = core_gguf::kv_u32(gctx, "moss_transcribe.eos_token_id", hp.eos_token_id);
        hp.start_token_id = core_gguf::kv_u32(gctx, "moss_transcribe.start_token_id", hp.start_token_id);
        hp.audio_start_id = core_gguf::kv_u32(gctx, "moss_transcribe.audio_start_id", hp.audio_start_id);
        hp.audio_end_id = core_gguf::kv_u32(gctx, "moss_transcribe.audio_end_id", hp.audio_end_id);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++)
                vocab.token_to_id[vocab.id_to_token[i]] = i;
        }

        struct SpecialTok {
            int id;
            const char* text;
        };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"}, {151644, "<|im_start|>"},  {151645, "<|im_end|>"},
            {151669, "<|audio_bos|>"}, {151670, "<|audio_eos|>"},
        };
        for (const auto& sp : specials) {
            if (sp.id < (int)vocab.id_to_token.size()) {
                auto old_it = vocab.token_to_id.find(vocab.id_to_token[sp.id]);
                if (old_it != vocab.token_to_id.end() && old_it->second == sp.id)
                    vocab.token_to_id.erase(old_it);
                vocab.id_to_token[sp.id] = sp.text;
                vocab.token_to_id[sp.text] = sp.id;
            }
        }

        auto merges = core_gguf::kv_str_array(gctx, "tokenizer.ggml.merges");
        for (int i = 0; i < (int)merges.size(); i++)
            vocab.merge_rank[merges[i]] = i;

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "moss_transcribe", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind tensors ----
    auto& enc = model.enc;
    enc.conv1_w = require(model, "enc.conv1.weight");
    enc.conv1_b = require(model, "enc.conv1.bias");
    enc.conv2_w = require(model, "enc.conv2.weight");
    enc.conv2_b = require(model, "enc.conv2.bias");
    enc.conv3_w = require(model, "enc.conv3.weight");
    enc.conv3_b = require(model, "enc.conv3.bias");
    enc.conv_out_w = require(model, "enc.conv_out.weight");
    enc.ln_post_w = require(model, "enc.ln_post.weight");
    enc.ln_post_b = require(model, "enc.ln_post.bias");
    enc.proj1_w = require(model, "enc.proj1.weight");
    enc.proj1_b = require(model, "enc.proj1.bias");
    enc.proj2_w = require(model, "enc.proj2.weight");
    enc.proj2_b = require(model, "enc.proj2.bias");

    enc.blocks.resize(model.hparams.enc_layers);
    for (uint32_t i = 0; i < model.hparams.enc_layers; i++) {
        char buf[128];
        auto& b = enc.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w = get("attn.q.weight");
        b.attn_q_b = get("attn.q.bias");
        b.attn_k_w = get("attn.k.weight");
        b.attn_k_b = get("attn.k.bias");
        b.attn_v_w = get("attn.v.weight");
        b.attn_v_b = get("attn.v.bias");
        b.attn_o_w = get("attn.o.weight");
        b.attn_o_b = get("attn.o.bias");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_norm_b = get("ffn_norm.bias");
        b.ffn_fc1_w = get("ffn.fc1.weight");
        b.ffn_fc1_b = get("ffn.fc1.bias");
        b.ffn_fc2_w = get("ffn.fc2.weight");
        b.ffn_fc2_b = get("ffn.fc2.bias");
    }

    // Adapter
    model.adapter.gate_w = require(model, "adapter.gate.weight");
    model.adapter.up_w = require(model, "adapter.up.weight");
    model.adapter.down_w = require(model, "adapter.down.weight");

    // LLM
    auto& llm = model.llm;
    llm.embed_w = require(model, "llm.embed.weight");
    llm.final_norm_w = require(model, "llm.final_norm.weight");
    // Tied lm_head (no separate tensor in the checkpoint).
    llm.lm_head_w = try_get(model, "llm.lm_head.weight");
    if (!llm.lm_head_w)
        llm.lm_head_w = llm.embed_w;

    llm.blocks.resize(model.hparams.llm_layers);
    for (uint32_t i = 0; i < model.hparams.llm_layers; i++) {
        char buf[128];
        auto& b = llm.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_q_w = get("attn.q.weight");
        b.attn_k_w = get("attn.k.weight");
        b.attn_v_w = get("attn.v.weight");
        b.attn_o_w = get("attn.o.weight");
        b.attn_q_norm_w = get("attn.q_norm.weight");
        b.attn_k_norm_w = get("attn.k_norm.weight");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_gate_w = get("ffn.gate.weight");
        b.ffn_up_w = get("ffn.up.weight");
        b.ffn_down_w = get("ffn.down.weight");
    }

    // ---- precompute sinusoidal position embeddings ----
    {
        const int C = (int)model.hparams.enc_d_model;
        const int L = (int)model.hparams.enc_max_pos;
        const int half = C / 2;
        const float log_inc = std::log(10000.0f) / (float)(half - 1);
        std::vector<float> inv_t(half);
        for (int i = 0; i < half; i++)
            inv_t[i] = std::exp(-log_inc * (float)i);
        model.audio_pe.assign((size_t)L * C, 0.0f);
        for (int p = 0; p < L; p++) {
            float* row = model.audio_pe.data() + (size_t)p * C;
            for (int i = 0; i < half; i++) {
                float angle = (float)p * inv_t[i];
                row[i] = std::sin(angle);
                row[half + i] = std::cos(angle);
            }
        }
    }

    const auto& hp = model.hparams;
    fprintf(stderr,
            "moss_transcribe: loaded %u enc layers (d=%u, out=%u), adapter (hidden=%u), "
            "%u LLM layers (d=%u), vocab=%u\n",
            hp.enc_layers, hp.enc_d_model, hp.enc_output_dim, hp.adapter_hidden, hp.llm_layers, hp.llm_hidden,
            hp.llm_vocab_size);
    return true;
}

// ===========================================================================
// FFT (radix-2 with DFT fallback) — for the Whisper-style mel
// ===========================================================================

static void moss_transcribe_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

static void moss_transcribe_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        moss_transcribe_dft(in, N, out);
        return;
    }
    std::vector<float> even_in(half_N), odd_in(half_N);
    for (int i = 0; i < half_N; i++) {
        even_in[i] = in[2 * i];
        odd_in[i] = in[2 * i + 1];
    }
    std::vector<float> even_out(2 * half_N), odd_out(2 * half_N);
    moss_transcribe_fft(even_in.data(), half_N, even_out.data());
    moss_transcribe_fft(odd_in.data(), half_N, odd_out.data());
    for (int k = 0; k < half_N; k++) {
        float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        float cos_a = std::cos(ang), sin_a = std::sin(ang);
        float tre = odd_out[2 * k] * cos_a - odd_out[2 * k + 1] * sin_a;
        float tim = odd_out[2 * k] * sin_a + odd_out[2 * k + 1] * cos_a;
        out[2 * k] = even_out[2 * k] + tre;
        out[2 * k + 1] = even_out[2 * k + 1] + tim;
        out[2 * (k + half_N)] = even_out[2 * k] - tre;
        out[2 * (k + half_N) + 1] = even_out[2 * k + 1] - tim;
    }
}

// ===========================================================================
// Mel spectrogram (Whisper-style, 128-bin). No 30 s padding — MOSS-Transcribe
// runs the feature extractor on the raw waveform (feature_lens = raw frames).
// ===========================================================================

extern "C" float* moss_transcribe_compute_mel(struct moss_transcribe_context* ctx, const float* samples, int n_samples,
                                              int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels_val = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;

    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft));

    std::vector<float> mel_filters =
        core_mel::build_slaney_fb((int)hp.sample_rate, n_fft, n_mels_val, 0.0f, -1.0f, core_mel::FbLayout::MelsFreqs);

    core_mel::FftR2C fft_fn = [](const float* in, int N, float* out) {
        moss_transcribe_fft(const_cast<float*>(in), N, out);
    };
    core_mel::Params mel_params;
    mel_params.n_fft = n_fft;
    mel_params.hop_length = hop;
    mel_params.win_length = n_fft;
    mel_params.n_mels = n_mels_val;
    mel_params.log_base = core_mel::LogBase::Log10;
    mel_params.spec_kind = core_mel::SpecKind::Power;
    mel_params.norm = core_mel::Normalization::GlobalClipMax;
    mel_params.layout = core_mel::Layout::MelsTime;
    mel_params.log_guard = core_mel::LogGuard::MaxClip;
    mel_params.log_eps = 1e-10f;
    mel_params.center_pad = true;
    mel_params.center_pad_reflect = true;

    int T_mel_actual = 0;
    std::vector<float> mel_out = core_mel::compute(samples, n_samples, hann.data(), n_fft, mel_filters.data(), n_freqs,
                                                   fft_fn, mel_params, T_mel_actual);

    // WhisperFeatureExtractor drops the trailing STFT frame (stft[..., :-1]),
    // yielding exactly n_samples/hop frames. core_mel's center padding produces
    // one extra; truncate to match the Python reference (else the audio-token
    // count and per-bin layout drift by one frame).
    const int T_target = n_samples / hop;
    if (T_mel_actual > T_target && T_target > 0) {
        std::vector<float> trunc((size_t)n_mels_val * T_target);
        for (int f = 0; f < n_mels_val; f++)
            memcpy(trunc.data() + (size_t)f * T_target, mel_out.data() + (size_t)f * T_mel_actual,
                   (size_t)T_target * sizeof(float));
        mel_out = std::move(trunc);
        T_mel_actual = T_target;
    }

    float* result = (float*)malloc(mel_out.size() * sizeof(float));
    memcpy(result, mel_out.data(), mel_out.size() * sizeof(float));
    if (const char* dp = std::getenv("MOSS_TRANSCRIBE_MEL_DUMP")) {
        FILE* f = fopen(dp, "wb");
        if (f) {
            fwrite(result, sizeof(float), mel_out.size(), f);
            fclose(f);
            fprintf(stderr, "moss_transcribe: dumped mel (%d,%d) to %s\n", n_mels_val, T_mel_actual, dp);
        }
    }
    if (out_n_mels)
        *out_n_mels = n_mels_val;
    if (out_T_mel)
        *out_T_mel = T_mel_actual;
    return result;
}

// ===========================================================================
// Audio encoder — conv stem graph (per 100-frame chunk)
// ===========================================================================

// Builds the conv front-end for one fixed CHUNK_T=100 mel-frame chunk:
//   mel(128,100) → conv1/2/3(gelu) → conv_out(no bias) → (d_model, T_down=13).
// Output tensor "conv_stem_out". Sinusoidal pos embed is added on the host
// side after extraction (per-chunk reset 0..T_down-1).
static ggml_cgraph* moss_transcribe_build_conv_graph(moss_transcribe_context* ctx, int chunk_T,
                                                     ggml_context* arena_ctx) {
    const auto& hp = ctx->model.hparams;
    const auto& enc = ctx->model.enc;
    const int n_mels = (int)hp.n_mels;

    ggml_context* ctx0 = arena_ctx;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // mel input as ggml ne=(n_mels, chunk_T) — freq fastest (see moss_audio note).
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, chunk_T);
    ggml_set_name(mel_in, "mel_input");
    ggml_set_input(mel_in);
    ggml_tensor* x = ggml_reshape_4d(ctx0, mel_in, n_mels, chunk_T, 1, 1);

    // Kernel stored (KW,KH,IC,OC); with data ne[0]=freq we need (KH,KW,IC,OC).
    auto kt = [&](ggml_tensor* w) -> ggml_tensor* { return ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3)); };
    auto bias4 = [&](ggml_tensor* b) -> ggml_tensor* { return ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1); };

    x = ggml_conv_2d(ctx0, kt(enc.conv1_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, bias4(enc.conv1_b));
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_conv_2d(ctx0, kt(enc.conv2_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, bias4(enc.conv2_b));
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_conv_2d(ctx0, kt(enc.conv3_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, bias4(enc.conv3_b));
    x = ggml_gelu_erf(ctx0, x);

    // After conv3: ggml ne=(F_down, T_down, C=480, 1). PyTorch flattens
    // (b,c,f,t)->(b,t,c*f) with f fastest → merge ne[0]=F_down and ne[2]=C.
    int F_down = conv_out_len(conv_out_len(conv_out_len(n_mels)));
    int T_down = conv_out_len(conv_out_len(conv_out_len(chunk_T)));
    int C_out = (int)hp.enc_ds_hidden;
    int conv_in = F_down * C_out;
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // (F, C, T, 1)
    x = ggml_reshape_2d(ctx0, x, conv_in, T_down);

    // conv_out: Linear(conv_in, d_model), NO bias.
    x = ggml_mul_mat(ctx0, enc.conv_out_w, x); // (d, T_down)

    ggml_set_name(x, "conv_stem_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// ===========================================================================
// Audio encoder — transformer graph (32 layers + ln_post + proj1/2) over the
// full concatenated sequence with a block-diagonal windowed attention mask.
// ===========================================================================

static ggml_cgraph* moss_transcribe_build_encoder_xf_graph(moss_transcribe_context* ctx, int T_enc) {
    const auto& hp = ctx->model.hparams;
    const auto& enc = ctx->model.enc;
    const int d = (int)hp.enc_d_model;
    const int n_heads = (int)hp.enc_n_heads;
    const int head_dim = (int)hp.enc_head_dim;
    const int out_dim = (int)hp.enc_output_dim;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: hidden states (d, T_enc), conv_out + pos already applied on host.
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_enc);
    ggml_set_name(x, "xf_input");
    ggml_set_input(x);

    // Block-diagonal windowed attention mask (F16 for flash_attn_ext).
    ggml_tensor* attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_enc, T_enc);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    const float attn_scale = 1.0f / std::sqrt((float)head_dim);

    for (uint32_t il = 0; il < hp.enc_layers; il++) {
        const auto& blk = enc.blocks[il];
        ggml_tensor* residual = x;

        // Pre-LN self-attention
        ggml_tensor* h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.attn_norm_w), blk.attn_norm_b);

        ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_q_w, h), blk.attn_q_b);
        ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_k_w, h), blk.attn_k_b);
        ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_v_w, h), blk.attn_v_b);

        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_enc);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_enc);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_enc);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, n_h)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, K, V, attn_mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, d, T_enc);

        ggml_tensor* attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.attn_o_w, attn), blk.attn_o_b);
        x = ggml_add(ctx0, residual, attn_out);

        // Pre-LN FFN
        residual = x;
        h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.ffn_norm_w), blk.ffn_norm_b);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffn_fc1_w, h), blk.ffn_fc1_b);
        h = ggml_gelu_erf(ctx0, h);
        h = ggml_add(ctx0, ggml_mul_mat(ctx0, blk.ffn_fc2_w, h), blk.ffn_fc2_b);
        x = ggml_add(ctx0, residual, h);

        if (il == 0) {
            ggml_tensor* l0 = ggml_cont(ctx0, x);
            ggml_set_name(l0, "enc_layer_0");
            ggml_set_output(l0);
            ggml_build_forward_expand(gf, l0);
        }
    }

    // ln_post
    x = ggml_norm(ctx0, x, hp.enc_ln_eps);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, enc.ln_post_w), enc.ln_post_b);

    // proj1 → gelu → proj2 (→ output_dim)
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, enc.proj1_w, x), enc.proj1_b);
    x = ggml_gelu_erf(ctx0, x);
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, enc.proj2_w, x), enc.proj2_b);
    (void)out_dim;

    ggml_set_name(x, "encoder_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// ===========================================================================
// Audio encoder — execution
// ===========================================================================

extern "C" float* moss_transcribe_run_encoder(struct moss_transcribe_context* ctx, const float* mel, int n_mels,
                                              int T_mel, int* out_T_enc, int* out_d) {
    if (!ctx || !mel)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;
    const int out_dim = (int)hp.enc_output_dim;
    const int chunk_T = (int)hp.n_window * 2;                                   // 100
    const int T_chunk_down = conv_out_len(conv_out_len(conv_out_len(chunk_T))); // 13

    // ---- 1. Chunk + conv stem → concatenated hidden (d, T_enc) ----
    int num_chunks = (T_mel + chunk_T - 1) / chunk_T;
    if (num_chunks < 1)
        num_chunks = 1;
    std::vector<int> chunk_len(num_chunks, chunk_T);
    int tail = T_mel % chunk_T;
    if (tail > 0)
        chunk_len[num_chunks - 1] = tail;

    std::vector<int> valid_len(num_chunks);
    int T_enc = 0;
    for (int c = 0; c < num_chunks; c++) {
        valid_len[c] = feat_output_len(chunk_len[c]);
        if (valid_len[c] > T_chunk_down)
            valid_len[c] = T_chunk_down;
        T_enc += valid_len[c];
    }

    std::vector<float> hidden((size_t)d * T_enc, 0.0f); // (d, T_enc) ne-order (d fast)

    // Build/cache conv-stem graph (fixed chunk_T).
    if (!ctx->cached_conv_gf) {
        ctx->cached_conv_meta.assign(ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), 0);
        ggml_init_params aip = {ctx->cached_conv_meta.size(), ctx->cached_conv_meta.data(), true};
        ctx->cached_conv_ctx = ggml_init(aip);
        ctx->cached_conv_gf = moss_transcribe_build_conv_graph(ctx, chunk_T, ctx->cached_conv_ctx);
    }

    int out_off = 0;
    for (int c = 0; c < num_chunks; c++) {
        std::vector<float> chunk_mel((size_t)n_mels * chunk_T, 0.0f);
        int t_start = c * chunk_T;
        int t_len = chunk_len[c];
        for (int t = 0; t < t_len; t++)
            for (int f = 0; f < n_mels; f++)
                chunk_mel[(size_t)f + n_mels * (size_t)t] = mel[(size_t)f * T_mel + t_start + t];

        ggml_cgraph* gf = ctx->cached_conv_gf;
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "moss_transcribe: conv graph alloc failed (chunk %d)\n", c);
            return nullptr;
        }
        ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_input");
        ggml_backend_tensor_set(mel_in, chunk_mel.data(), 0, (size_t)n_mels * chunk_T * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "moss_transcribe: conv graph compute failed (chunk %d)\n", c);
            return nullptr;
        }
        ggml_tensor* cs = ggml_graph_get_tensor(gf, "conv_stem_out"); // (d, T_chunk_down)
        std::vector<float> chunk_out((size_t)d * T_chunk_down);
        ggml_backend_tensor_get(cs, chunk_out.data(), 0, chunk_out.size() * sizeof(float));

        // Keep first valid_len[c] frames; add per-chunk sinusoidal pos (0..valid-1).
        int valid = valid_len[c];
        for (int t = 0; t < valid; t++) {
            const float* pe = ctx->model.audio_pe.data() + (size_t)t * d;
            float* dst = hidden.data() + (size_t)(out_off + t) * d;
            const float* src = chunk_out.data() + (size_t)t * d;
            for (int j = 0; j < d; j++)
                dst[j] = src[j] + pe[j];
        }
        out_off += valid;
    }

    // ---- 2. Transformer over full sequence with block-diagonal windows ----
    // window_aftercnn = T_chunk_down * (n_window_infer / (2*n_window)).
    const int win = T_chunk_down * ((int)hp.n_window_infer / chunk_T);
    ggml_cgraph* gf = moss_transcribe_build_encoder_xf_graph(ctx, T_enc);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_transcribe: encoder xf alloc failed\n");
        return nullptr;
    }
    ggml_tensor* xin = ggml_graph_get_tensor(gf, "xf_input");
    ggml_backend_tensor_set(xin, hidden.data(), 0, (size_t)d * T_enc * sizeof(float));

    // Block-diagonal mask: window blocks of `win` frames, bidirectional within.
    {
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "attn_mask");
        std::vector<ggml_fp16_t> mask((size_t)T_enc * T_enc);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-INFINITY);
        std::vector<int> block_of(T_enc);
        for (int i = 0; i < T_enc; i++)
            block_of[i] = (win > 0) ? (i / win) : 0;
        for (int q = 0; q < T_enc; q++)
            for (int k = 0; k < T_enc; k++)
                mask[(size_t)q * T_enc + k] = (block_of[q] == block_of[k]) ? zero_h : ninf_h;
        ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_transcribe: encoder xf compute failed\n");
        return nullptr;
    }
    ggml_tensor* eo = ggml_graph_get_tensor(gf, "encoder_output"); // (out_dim, T_enc)
    float* result = (float*)malloc((size_t)out_dim * T_enc * sizeof(float));
    ggml_backend_tensor_get(eo, result, 0, (size_t)out_dim * T_enc * sizeof(float));
    if (const char* dp = std::getenv("MOSS_TRANSCRIBE_L0_DUMP")) {
        ggml_tensor* l0 = ggml_graph_get_tensor(gf, "enc_layer_0");
        if (l0) {
            std::vector<float> b((size_t)d * T_enc);
            ggml_backend_tensor_get(l0, b.data(), 0, b.size() * sizeof(float));
            FILE* f = fopen(dp, "wb");
            if (f) {
                fwrite(b.data(), sizeof(float), b.size(), f);
                fclose(f);
                fprintf(stderr, "moss_transcribe: dumped enc_layer_0 (%d,%d)\n", d, T_enc);
            }
        }
    }
    if (const char* dp = std::getenv("MOSS_TRANSCRIBE_ENC_DUMP")) {
        FILE* f = fopen(dp, "wb");
        if (f) {
            fwrite(result, sizeof(float), (size_t)out_dim * T_enc, f);
            fclose(f);
            fprintf(stderr, "moss_transcribe: dumped encoder_output (%d,%d) win=%d\n", out_dim, T_enc, win);
        }
    }

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d)
        *out_d = out_dim;
    return result;
}

// ===========================================================================
// Audio adapter (SwiGLU GatedMLP: output_dim → adapter_hidden → llm_hidden)
// ===========================================================================

static ggml_cgraph* moss_transcribe_build_adapter_graph(moss_transcribe_context* ctx, int T_enc) {
    const auto& hp = ctx->model.hparams;
    const int d_enc = (int)hp.enc_output_dim;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, T_enc);
    ggml_set_name(enc_out, "enc_output_in");
    ggml_set_input(enc_out);

    ggml_tensor* adapted =
        core_ffn::swiglu(ctx0, enc_out, ctx->model.adapter.gate_w, ctx->model.adapter.up_w, ctx->model.adapter.down_w);
    ggml_set_name(adapted, "adapter_output");
    ggml_set_output(adapted);
    ggml_build_forward_expand(gf, adapted);
    return gf;
}

extern "C" float* moss_transcribe_run_adapter(struct moss_transcribe_context* ctx, const float* encoder_out, int T_enc,
                                              int d_enc, int* out_T, int* out_d) {
    if (!ctx || !encoder_out)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_hidden;

    ggml_cgraph* gf = moss_transcribe_build_adapter_graph(ctx, T_enc);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_transcribe: adapter alloc failed\n");
        return nullptr;
    }
    ggml_tensor* in = ggml_graph_get_tensor(gf, "enc_output_in");
    ggml_backend_tensor_set(in, encoder_out, 0, (size_t)d_enc * T_enc * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_transcribe: adapter compute failed\n");
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "adapter_output");
    if (!out)
        return nullptr;
    if (out_T)
        *out_T = T_enc;
    if (out_d)
        *out_d = d_llm;
    float* result = (float*)malloc((size_t)d_llm * T_enc * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)d_llm * T_enc * sizeof(float));
    return result;
}

// ===========================================================================
// LLM graph with KV cache (Qwen3)
// ===========================================================================

static ggml_cgraph* moss_transcribe_build_llm_kv_graph(moss_transcribe_context* ctx, int n_tokens, int n_past,
                                                       bool last_token_only) {
    const auto& hp = ctx->model.hparams;
    const auto& llm = ctx->model.llm;
    const int d = (int)hp.llm_hidden;
    const int n_heads = (int)hp.llm_n_heads;
    const int n_kv_heads = (int)hp.llm_n_kv_heads;
    const int head_dim = (int)hp.llm_head_dim;
    const int Lk = n_past + n_tokens;

    struct ggml_init_params gparams = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* embeds_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(embeds_in, "inputs_embeds");
    ggml_set_input(embeds_in);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor* causal_mask = nullptr;
    if (n_tokens > 1) {
        causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, n_tokens);
        ggml_set_name(causal_mask, "causal_mask");
        ggml_set_input(causal_mask);
    }

    core_attn::KvSelfAttnParams attn_p = {};
    attn_p.n_heads = n_heads;
    attn_p.n_kv_heads = n_kv_heads;
    attn_p.head_dim = head_dim;
    attn_p.n_kv_grp = n_heads / n_kv_heads;
    attn_p.n_ctx_orig = 0;
    attn_p.rope_theta = hp.llm_rope_theta;
    attn_p.rope_beta_fast = 32.0f;
    attn_p.rope_beta_slow = 1.0f;
    attn_p.attn_scale = 1.0f / std::sqrt((float)head_dim);
    attn_p.qk_norm_eps = hp.llm_rms_eps;
    attn_p.gqa_mode = core_attn::GQA_MANUAL_CONT;
    attn_p.rope_type = GGML_ROPE_TYPE_NEOX;

    ggml_tensor* cur = embeds_in;
    for (uint32_t il = 0; il < hp.llm_layers; il++) {
        const auto& blk = llm.blocks[il];
        ggml_tensor* residual = cur;

        ggml_tensor* h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.attn_norm_w);

        ggml_tensor* attn_out = core_attn::kv_self_attn(ctx0, gf, h, blk.attn_q_w, blk.attn_k_w, blk.attn_v_w,
                                                        blk.attn_o_w, blk.attn_q_norm_w, blk.attn_k_norm_w, positions,
                                                        causal_mask, ctx->kv_k, ctx->kv_v, (int)il, n_past, attn_p);
        cur = ggml_add(ctx0, residual, attn_out);

        residual = cur;
        h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
    cur = ggml_mul(ctx0, cur, llm.final_norm_w);

    if (last_token_only && n_tokens > 1)
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    cur = ggml_mul_mat(ctx0, llm.lm_head_w, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ===========================================================================
// KV cache management
// ===========================================================================

extern "C" bool moss_transcribe_kv_init(struct moss_transcribe_context* ctx, int max_ctx) {
    if (!ctx)
        return false;
    const auto& hp = ctx->model.hparams;
    const int n_layers = (int)hp.llm_layers;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;

    ggml_type kv_type = core_attn::kv_dtype_from_env("moss_transcribe");

    struct ggml_init_params kv_params = {2 * ggml_tensor_overhead(), nullptr, true};
    ctx->kv_ctx = ggml_init(kv_params);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        fprintf(stderr, "moss_transcribe: kv alloc failed for max_ctx=%d\n", max_ctx);
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    return true;
}

extern "C" void moss_transcribe_kv_reset(struct moss_transcribe_context* ctx) {
    if (ctx && ctx->kv_buf) {
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        ctx->kv_n_used = 0;
    }
}

// ===========================================================================
// Tokenizer (GPT-2 byte-level BPE via core_bpe)
// ===========================================================================

extern "C" int moss_transcribe_tokenize(struct moss_transcribe_context* ctx, const char* text, int32_t* out_tokens,
                                        int max_tokens) {
    if (!ctx || !text || !out_tokens || max_tokens <= 0)
        return 0;
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;
    const std::string s = text;
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
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n')
                k++;
            if (k == start)
                k++;
            std::string pre(chunk, start, k - start);
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }
    int n = std::min((int)result.size(), max_tokens);
    std::memcpy(out_tokens, result.data(), (size_t)n * sizeof(int32_t));
    return n;
}

extern "C" const char* moss_transcribe_token_text(struct moss_transcribe_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return nullptr;
    return ctx->vocab.id_to_token[token_id].c_str();
}

// ===========================================================================
// Embed tokens
// ===========================================================================

extern "C" float* moss_transcribe_embed_tokens(struct moss_transcribe_context* ctx, const int32_t* token_ids,
                                               int n_tokens) {
    if (!ctx || !token_ids || n_tokens <= 0)
        return nullptr;
    const int d = (int)ctx->model.hparams.llm_hidden;

    if (n_tokens == 1 && ctx->model.llm.embed_w) {
        const ggml_tensor* w = ctx->model.llm.embed_w;
        const size_t row_bytes = ggml_row_size(w->type, d);
        float* result = (float*)malloc((size_t)d * sizeof(float));
        if (!result)
            return nullptr;
        std::vector<uint8_t> raw(row_bytes);
        ggml_backend_tensor_get(w, raw.data(), (size_t)token_ids[0] * row_bytes, row_bytes);
        if (w->type == GGML_TYPE_F32)
            std::memcpy(result, raw.data(), (size_t)d * sizeof(float));
        else
            ggml_get_type_traits(w->type)->to_float(raw.data(), result, d);
        return result;
    }

    struct ggml_init_params gp = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);
    ggml_tensor* emb = ggml_get_rows(ctx0, ctx->model.llm.embed_w, ids);
    ggml_set_name(emb, "embeds");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_set(ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_tensor* out = ggml_graph_get_tensor(gf, "embeds");
    float* result = (float*)malloc((size_t)d * n_tokens * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)d * n_tokens * sizeof(float));
    ggml_free(ctx0);
    return result;
}

// ===========================================================================
// Run LLM with KV cache (prefill or single-step)
// ===========================================================================

extern "C" float* moss_transcribe_run_llm_kv(struct moss_transcribe_context* ctx, const float* inputs_embeds,
                                             int n_tokens, int n_past, int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_hidden;
    const int vocab = (int)hp.llm_vocab_size;
    const int Lk = n_past + n_tokens;

    ggml_cgraph* gf = moss_transcribe_build_llm_kv_graph(ctx, n_tokens, n_past, /*last_token_only=*/true);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_transcribe: llm alloc failed\n");
        return nullptr;
    }
    ggml_tensor* emb_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(emb_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++)
        positions[i] = n_past + i;
    ggml_backend_tensor_set(pos_in, positions.data(), 0, (size_t)n_tokens * sizeof(int32_t));

    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            int abs_q = n_past + q;
            for (int k = 0; k < Lk; k++)
                mask[(size_t)q * Lk + k] = (k <= abs_q) ? zero_h : neginf_h;
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_transcribe: llm compute failed\n");
        return nullptr;
    }
    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t)
        return nullptr;
    if (out_n_tokens)
        *out_n_tokens = 1;
    if (out_vocab_size)
        *out_vocab_size = vocab;
    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    ctx->kv_n_used = Lk;
    return result;
}

// ===========================================================================
// Prompt build — chat_template_default.py (the canonical MossProcessor
// template; NOT the bare legacy layout). Framing:
//   <|im_start|>user\n <|audio_start|> [placeholder*N] <|audio_end|>
//   <|im_end|>\n <|im_start|>assistant\n  → generate → <|im_end|>
// Token ids 872="user", 77091="assistant", 198="\n" are the template's
// constant_text_token text_ids (Qwen3 vocab). Without the user/assistant
// framing the model never gets the transcribe cue and emits garbage.
// ===========================================================================

extern "C" int moss_transcribe_build_prompt(struct moss_transcribe_context* ctx, int n_audio, int32_t* out_ids,
                                            int max_ids) {
    if (!ctx || !out_ids || n_audio < 0)
        return -1;
    const auto& hp = ctx->model.hparams;
    if (n_audio + 10 > max_ids)
        return -1;
    const int32_t TOK_USER = 872, TOK_ASSISTANT = 77091, TOK_NL = 198;
    int n = 0;
    out_ids[n++] = (int32_t)hp.start_token_id; // <|im_start|>
    out_ids[n++] = TOK_USER;                   // user
    out_ids[n++] = TOK_NL;                     // \n
    out_ids[n++] = (int32_t)hp.audio_start_id; // <|audio_start|>
    for (int i = 0; i < n_audio; i++)
        out_ids[n++] = (int32_t)hp.audio_placeholder_id;
    out_ids[n++] = (int32_t)hp.audio_end_id;   // <|audio_end|>
    out_ids[n++] = (int32_t)hp.eos_token_id;   // <|im_end|>
    out_ids[n++] = TOK_NL;                     // \n
    out_ids[n++] = (int32_t)hp.start_token_id; // <|im_start|>
    out_ids[n++] = TOK_ASSISTANT;              // assistant
    out_ids[n++] = TOK_NL;                     // \n
    return n;
}

// ===========================================================================
// High-level transcribe
// ===========================================================================

static char* moss_transcribe_impl(struct moss_transcribe_context* ctx, const float* samples, int n_samples,
                                  moss_transcribe_token_cb on_tok, void* userdata) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_hidden;
    moss_transcribe_bench_stage _b_total("total");

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_transcribe: %d samples (%.1f s)\n", n_samples, (float)n_samples / (float)hp.sample_rate);

    // 1. Mel
    int n_mels = 0, T_mel = 0;
    float* mel = nullptr;
    {
        moss_transcribe_bench_stage _b("mel");
        mel = moss_transcribe_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    }
    if (!mel) {
        fprintf(stderr, "moss_transcribe: mel failed\n");
        return nullptr;
    }

    // 2. Encoder
    int T_enc = 0, enc_d = 0;
    float* encoder_out = nullptr;
    {
        moss_transcribe_bench_stage _b("encoder");
        encoder_out = moss_transcribe_run_encoder(ctx, mel, n_mels, T_mel, &T_enc, &enc_d);
    }
    free(mel);
    if (!encoder_out) {
        fprintf(stderr, "moss_transcribe: encoder failed\n");
        return nullptr;
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_transcribe: encoder %d frames × %d dims\n", T_enc, enc_d);

    // 3. Adapter
    int adapt_T = 0, adapt_d = 0;
    float* audio_embeds = nullptr;
    {
        moss_transcribe_bench_stage _b("adapter");
        audio_embeds = moss_transcribe_run_adapter(ctx, encoder_out, T_enc, enc_d, &adapt_T, &adapt_d);
    }
    free(encoder_out);
    if (!audio_embeds)
        return nullptr;

    // 4. Build prompt + embed text tokens
    std::vector<int32_t> prompt_ids((size_t)T_enc + 16);
    int n_prompt = moss_transcribe_build_prompt(ctx, T_enc, prompt_ids.data(), (int)prompt_ids.size());
    if (n_prompt < 0) {
        free(audio_embeds);
        return nullptr;
    }
    prompt_ids.resize(n_prompt);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "moss_transcribe: %d mel frames, %d enc tokens, %d prompt tokens\n", T_mel, T_enc, n_prompt);
        fprintf(stderr, "moss_transcribe: prompt first6:");
        for (int i = 0; i < std::min(6, n_prompt); i++)
            fprintf(stderr, " %d", prompt_ids[i]);
        fprintf(stderr, " ... last3:");
        for (int i = std::max(0, n_prompt - 3); i < n_prompt; i++)
            fprintf(stderr, " %d", prompt_ids[i]);
        fprintf(stderr, "\n");
    }

    float* text_embeds = moss_transcribe_embed_tokens(ctx, prompt_ids.data(), n_prompt);
    if (!text_embeds) {
        free(audio_embeds);
        return nullptr;
    }

    // 5. Scatter audio embeddings at placeholder positions
    {
        int audio_idx = 0;
        for (int pos = 0; pos < n_prompt; pos++) {
            if (prompt_ids[pos] == (int32_t)hp.audio_placeholder_id && audio_idx < T_enc) {
                memcpy(text_embeds + (size_t)pos * d_llm, audio_embeds + (size_t)audio_idx * d_llm,
                       (size_t)d_llm * sizeof(float));
                audio_idx++;
            }
        }
    }
    free(audio_embeds);

    // 6. KV cache + prefill
    int max_ctx = n_prompt + 512;
    if (ctx->kv_k) {
        if (ctx->kv_max_ctx < max_ctx) {
            if (ctx->kv_buf)
                ggml_backend_buffer_free(ctx->kv_buf);
            if (ctx->kv_ctx)
                ggml_free(ctx->kv_ctx);
            ctx->kv_buf = nullptr;
            ctx->kv_ctx = nullptr;
            ctx->kv_k = nullptr;
            ctx->kv_v = nullptr;
            moss_transcribe_kv_init(ctx, max_ctx);
        } else {
            moss_transcribe_kv_reset(ctx);
        }
    } else {
        moss_transcribe_kv_init(ctx, max_ctx);
    }

    int vocab = 0;
    float* logits = moss_transcribe_run_llm_kv(ctx, text_embeds, n_prompt, 0, nullptr, &vocab);
    free(text_embeds);
    if (!logits)
        return nullptr;

    // 7. Decode
    std::vector<int32_t> generated;
    const int max_new = 512;
    if (ctx->beam_size > 1) {
        auto replay = [&vocab](moss_transcribe_context* c, const int32_t* toks, int n, int prompt_len) -> float* {
            float* emb = moss_transcribe_embed_tokens(c, toks, n);
            if (!emb)
                return nullptr;
            int dummy = 0;
            float* lg = moss_transcribe_run_llm_kv(c, emb, n, prompt_len, &dummy, &vocab);
            std::free(emb);
            return lg;
        };
        core_beam_decode::Config cfg;
        cfg.max_new_tokens = max_new;
        cfg.eos_id = (int)hp.eos_token_id;
        cfg.vocab_size = vocab;
        cfg.beam_size = ctx->beam_size;
        cfg.prompt_len = n_prompt;
        auto br = core_beam_decode::run_with_probs(ctx, logits, replay, cfg);
        generated = std::move(br.tokens);
        logits = nullptr;
        if (!generated.empty() && generated.back() == (int)hp.eos_token_id)
            generated.pop_back();
    } else {
        for (int step = 0; step < max_new; step++) {
            int best_id = 0;
            float best_val = logits[0];
            for (int i = 1; i < vocab; i++)
                if (logits[i] > best_val) {
                    best_val = logits[i];
                    best_id = i;
                }
            float tok_prob = 0.0f;
            if (on_tok && best_id != (int)hp.eos_token_id) {
                float s = 0.0f;
                for (int i = 0; i < vocab; i++)
                    s += expf(logits[i] - best_val);
                tok_prob = (s > 0.0f) ? (1.0f / s) : 0.0f;
            }
            free(logits);
            logits = nullptr;
            if (ctx->params.verbosity >= 1 && step < 5)
                fprintf(stderr, "moss_transcribe: step %d argmax=%d (%.4f)\n", step, best_id, best_val);
            if (best_id == (int)hp.eos_token_id)
                break;
            generated.push_back(best_id);
            if (on_tok)
                on_tok(best_id, tok_prob, userdata);
            float* next_emb = moss_transcribe_embed_tokens(ctx, &best_id, 1);
            if (!next_emb)
                break;
            int dummy = 0;
            logits = moss_transcribe_run_llm_kv(ctx, next_emb, 1, n_prompt + (int)generated.size() - 1, &dummy, &vocab);
            free(next_emb);
            if (!logits)
                break;
        }
        if (logits)
            free(logits);
    }

    // 8. Detokenize
    std::string result;
    for (int id : generated) {
        const char* t = moss_transcribe_token_text(ctx, id);
        if (t)
            result += core_bpe::token_bytes_to_utf8(t);
    }
    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_transcribe: %zu tokens: \"%s\"\n", generated.size(), result.substr(0, 120).c_str());

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" char* moss_transcribe_transcribe(struct moss_transcribe_context* ctx, const float* samples, int n_samples) {
    return moss_transcribe_impl(ctx, samples, n_samples, nullptr, nullptr);
}

extern "C" void moss_transcribe_transcribe_cb(struct moss_transcribe_context* ctx, const float* samples, int n_samples,
                                              moss_transcribe_token_cb cb, void* userdata) {
    char* s = moss_transcribe_impl(ctx, samples, n_samples, cb, userdata);
    free(s);
}

// ===========================================================================
// Init / Free
// ===========================================================================

extern "C" struct moss_transcribe_context_params moss_transcribe_context_default_params(void) {
    moss_transcribe_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.flash_attn = false;
    return p;
}

extern "C" struct moss_transcribe_context* moss_transcribe_init_from_file(
    const char* path_model, struct moss_transcribe_context_params params) {
    auto* ctx = new moss_transcribe_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads;
    ctx->model_path = path_model;

    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!moss_transcribe_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "moss_transcribe: failed to load model from %s\n", path_model);
        moss_transcribe_free(ctx);
        return nullptr;
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
    return ctx;
}

extern "C" void moss_transcribe_free(struct moss_transcribe_context* ctx) {
    if (!ctx)
        return;
    if (ctx->cached_conv_ctx)
        ggml_free(ctx->cached_conv_ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
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

extern "C" void moss_transcribe_set_beam_size(struct moss_transcribe_context* ctx, int beam_size) {
    if (ctx)
        ctx->beam_size = beam_size > 0 ? beam_size : 1;
}
