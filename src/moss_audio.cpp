// moss_audio.cpp — MOSS-Audio-4B-Instruct ggml runtime
//
// Architecture: 32-layer Whisper-style audio encoder with DeepStack
// 3-tap cross-layer injection + 36-layer Qwen3 LLM.
// See moss_audio.h for the full architecture description.

#include "moss_audio.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Hyperparameters
// ===========================================================================

struct moss_audio_hparams {
    // Audio encoder
    uint32_t n_mels        = 128;
    uint32_t n_fft         = 400;
    uint32_t hop_length    = 160;
    uint32_t sample_rate   = 16000;
    uint32_t enc_layers    = 32;
    uint32_t enc_d_model   = 1280;
    uint32_t enc_n_heads   = 20;
    uint32_t enc_head_dim  = 64;   // 1280 / 20
    uint32_t enc_ffn_dim   = 5120;
    uint32_t enc_ds_hidden = 480;  // downsample_hidden_size (Conv2d channels)
    uint32_t enc_max_pos   = 1500;
    uint32_t enc_output_dim = 1280;
    uint32_t enc_attn_window = 100;
    float    enc_ln_eps    = 1e-5f;

    // DeepStack
    uint32_t ds_num_taps        = 3;
    uint32_t ds_tap_layers[3]   = {8, 16, 24};
    uint32_t ds_num_inject      = 3;

    // Adapter
    uint32_t adapter_hidden = 8192;

    // LLM (Qwen3)
    uint32_t llm_layers     = 36;
    uint32_t llm_hidden     = 2560;
    uint32_t llm_n_heads    = 32;
    uint32_t llm_n_kv_heads = 8;
    uint32_t llm_head_dim   = 128;
    uint32_t llm_ff_dim     = 9728;
    uint32_t llm_vocab_size = 151936;
    uint32_t llm_max_pos    = 40960;
    float    llm_rope_theta = 1000000.0f;
    float    llm_rms_eps    = 1e-6f;

    // Special tokens
    uint32_t bos_token_id     = 151643;
    uint32_t eos_token_id     = 151645;
    uint32_t audio_token_id   = 151654;
    uint32_t audio_start_id   = 151669;
    uint32_t audio_end_id     = 151670;
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

struct moss_audio_enc_block {
    // Pre-LN self-attention (Whisper-style)
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    ggml_tensor *attn_q_w = nullptr, *attn_q_b = nullptr;
    ggml_tensor *attn_k_w = nullptr;  // no bias
    ggml_tensor *attn_v_w = nullptr, *attn_v_b = nullptr;
    ggml_tensor *attn_o_w = nullptr, *attn_o_b = nullptr;
    // Pre-LN FFN (GELU)
    ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    ggml_tensor *ffn_fc1_w = nullptr, *ffn_fc1_b = nullptr;
    ggml_tensor *ffn_fc2_w = nullptr, *ffn_fc2_b = nullptr;
};

struct moss_audio_encoder {
    // Conv stem: 3 × Conv2d(stride=2)
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *conv3_w = nullptr, *conv3_b = nullptr;
    // stem_proj: Linear(ds_hidden * 16 = 7680, d_model = 1280)
    ggml_tensor *stem_proj_w = nullptr, *stem_proj_b = nullptr;
    // Final layer norm
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
    // Transformer layers
    std::vector<moss_audio_enc_block> blocks;
};

// GatedMLP: gate_proj + up_proj + down_proj (SiLU gating)
struct moss_audio_gated_mlp {
    ggml_tensor *gate_w = nullptr;
    ggml_tensor *up_w   = nullptr;
    ggml_tensor *down_w = nullptr;
};

struct moss_audio_llm_block {
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

struct moss_audio_llm {
    ggml_tensor* embed_w = nullptr;
    std::vector<moss_audio_llm_block> blocks;
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* lm_head_w = nullptr;
};

struct moss_audio_model {
    moss_audio_hparams hparams;
    moss_audio_encoder enc;
    moss_audio_gated_mlp adapter;
    moss_audio_gated_mlp deepstack[3];
    moss_audio_llm llm;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Sinusoidal position embeddings (max_pos × d_model), precomputed
    std::vector<float> audio_pe;
};

struct moss_audio_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

struct moss_audio_context {
    moss_audio_context_params params;
    moss_audio_model model;
    moss_audio_vocab vocab;

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

    int n_threads = 4;

    // Sampling
    uint32_t seed = 0;
    std::mt19937 rng;

    std::string model_path;
};

// ===========================================================================
// Helpers
// ===========================================================================

#include "core/gguf_loader.h"

static ggml_tensor* try_get(moss_audio_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(moss_audio_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "moss_audio");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool moss_audio_load_model(moss_audio_model& model, moss_audio_vocab& vocab,
                                  const char* path, ggml_backend_t backend) {
    // ---- pass 1: metadata + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx) return false;

        auto& hp = model.hparams;
        hp.n_mels      = core_gguf::kv_u32(gctx, "moss_audio.enc.num_mel_bins", hp.n_mels);
        hp.enc_layers   = core_gguf::kv_u32(gctx, "moss_audio.enc.encoder_layers", hp.enc_layers);
        hp.enc_d_model  = core_gguf::kv_u32(gctx, "moss_audio.enc.d_model", hp.enc_d_model);
        hp.enc_n_heads  = core_gguf::kv_u32(gctx, "moss_audio.enc.encoder_attention_heads", hp.enc_n_heads);
        hp.enc_ffn_dim  = core_gguf::kv_u32(gctx, "moss_audio.enc.encoder_ffn_dim", hp.enc_ffn_dim);
        hp.enc_ds_hidden = core_gguf::kv_u32(gctx, "moss_audio.enc.downsample_hidden_size", hp.enc_ds_hidden);
        hp.enc_max_pos  = core_gguf::kv_u32(gctx, "moss_audio.enc.max_source_positions", hp.enc_max_pos);
        hp.enc_output_dim = core_gguf::kv_u32(gctx, "moss_audio.enc.output_dim", hp.enc_output_dim);
        hp.enc_attn_window = core_gguf::kv_u32(gctx, "moss_audio.enc.encoder_attention_window_size", hp.enc_attn_window);
        hp.enc_ln_eps   = core_gguf::kv_f32(gctx, "moss_audio.enc.layer_norm_eps", hp.enc_ln_eps);
        hp.enc_head_dim = hp.enc_d_model / hp.enc_n_heads;

        hp.ds_num_taps  = core_gguf::kv_u32(gctx, "moss_audio.deepstack.num_taps", hp.ds_num_taps);
        for (uint32_t i = 0; i < hp.ds_num_taps && i < 3; i++) {
            char key[64];
            snprintf(key, sizeof(key), "moss_audio.deepstack.tap.%u", i);
            hp.ds_tap_layers[i] = core_gguf::kv_u32(gctx, key, hp.ds_tap_layers[i]);
        }
        hp.ds_num_inject = core_gguf::kv_u32(gctx, "moss_audio.deepstack.num_inject_layers", hp.ds_num_inject);

        hp.adapter_hidden = core_gguf::kv_u32(gctx, "moss_audio.adapter.hidden_size", hp.adapter_hidden);

        hp.llm_hidden    = core_gguf::kv_u32(gctx, "moss_audio.llm.hidden_size", hp.llm_hidden);
        hp.llm_layers    = core_gguf::kv_u32(gctx, "moss_audio.llm.num_layers", hp.llm_layers);
        hp.llm_n_heads   = core_gguf::kv_u32(gctx, "moss_audio.llm.num_heads", hp.llm_n_heads);
        hp.llm_n_kv_heads = core_gguf::kv_u32(gctx, "moss_audio.llm.num_kv_heads", hp.llm_n_kv_heads);
        hp.llm_head_dim  = core_gguf::kv_u32(gctx, "moss_audio.llm.head_dim", hp.llm_head_dim);
        hp.llm_ff_dim    = core_gguf::kv_u32(gctx, "moss_audio.llm.intermediate_size", hp.llm_ff_dim);
        hp.llm_vocab_size = core_gguf::kv_u32(gctx, "moss_audio.llm.vocab_size", hp.llm_vocab_size);
        hp.llm_max_pos   = core_gguf::kv_u32(gctx, "moss_audio.llm.max_position_embeddings", hp.llm_max_pos);
        hp.llm_rope_theta = core_gguf::kv_f32(gctx, "moss_audio.llm.rope_theta", hp.llm_rope_theta);
        hp.llm_rms_eps   = core_gguf::kv_f32(gctx, "moss_audio.llm.rms_norm_eps", hp.llm_rms_eps);

        hp.bos_token_id   = core_gguf::kv_u32(gctx, "moss_audio.bos_token_id", hp.bos_token_id);
        hp.eos_token_id   = core_gguf::kv_u32(gctx, "moss_audio.eos_token_id", hp.eos_token_id);
        hp.audio_token_id = core_gguf::kv_u32(gctx, "moss_audio.audio_token_id", hp.audio_token_id);
        hp.audio_start_id = core_gguf::kv_u32(gctx, "moss_audio.audio_start_id", hp.audio_start_id);
        hp.audio_end_id   = core_gguf::kv_u32(gctx, "moss_audio.audio_end_id", hp.audio_end_id);

        // Vocab
        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            vocab.token_to_id.reserve(vocab.id_to_token.size());
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Patch special tokens
        struct SpecialTok { int id; const char* text; };
        static const SpecialTok specials[] = {
            {151643, "<|endoftext|>"}, {151644, "<|im_start|>"}, {151645, "<|im_end|>"},
            {151654, "<|AUDIO|>"},     {151669, "<|audio_bos|>"}, {151670, "<|audio_eos|>"},
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
    if (!core_gguf::load_weights(path, backend, "moss_audio", wl))
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
    enc.stem_proj_w = require(model, "enc.stem_proj.weight");
    enc.stem_proj_b = require(model, "enc.stem_proj.bias");
    enc.norm_w = require(model, "enc.norm.weight");
    enc.norm_b = require(model, "enc.norm.bias");

    enc.blocks.resize(model.hparams.enc_layers);
    for (uint32_t i = 0; i < model.hparams.enc_layers; i++) {
        char buf[128];
        auto& b = enc.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        auto try_ = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "enc.blk.%u.%s", i, suf);
            return try_get(model, buf);
        };
        b.attn_norm_w = get("attn_norm.weight");
        b.attn_norm_b = get("attn_norm.bias");
        b.attn_q_w = get("attn.q.weight");
        b.attn_q_b = try_("attn.q.bias");
        b.attn_k_w = get("attn.k.weight");
        // k_proj has no bias in MOSS-Audio encoder
        b.attn_v_w = get("attn.v.weight");
        b.attn_v_b = try_("attn.v.bias");
        b.attn_o_w = get("attn.o.weight");
        b.attn_o_b = try_("attn.o.bias");
        b.ffn_norm_w = get("ffn_norm.weight");
        b.ffn_norm_b = get("ffn_norm.bias");
        b.ffn_fc1_w = get("ffn.fc1.weight");
        b.ffn_fc1_b = get("ffn.fc1.bias");
        b.ffn_fc2_w = get("ffn.fc2.weight");
        b.ffn_fc2_b = get("ffn.fc2.bias");
    }

    // Adapter
    model.adapter.gate_w = require(model, "adapter.gate.weight");
    model.adapter.up_w   = require(model, "adapter.up.weight");
    model.adapter.down_w = require(model, "adapter.down.weight");

    // DeepStack mergers
    for (uint32_t i = 0; i < model.hparams.ds_num_taps && i < 3; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "deepstack.%u.gate.weight", i);
        model.deepstack[i].gate_w = require(model, buf);
        snprintf(buf, sizeof(buf), "deepstack.%u.up.weight", i);
        model.deepstack[i].up_w = require(model, buf);
        snprintf(buf, sizeof(buf), "deepstack.%u.down.weight", i);
        model.deepstack[i].down_w = require(model, buf);
    }

    // LLM
    auto& llm = model.llm;
    llm.embed_w = require(model, "llm.embed.weight");
    llm.final_norm_w = require(model, "llm.final_norm.weight");
    llm.lm_head_w = require(model, "llm.lm_head.weight");

    llm.blocks.resize(model.hparams.llm_layers);
    for (uint32_t i = 0; i < model.hparams.llm_layers; i++) {
        char buf[128];
        auto& b = llm.blocks[i];
        auto get = [&](const char* suf) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, suf);
            return require(model, buf);
        };
        b.attn_norm_w  = get("attn_norm.weight");
        b.attn_q_w     = get("attn.q.weight");
        b.attn_k_w     = get("attn.k.weight");
        b.attn_v_w     = get("attn.v.weight");
        b.attn_o_w     = get("attn.o.weight");
        b.attn_q_norm_w = get("attn.q_norm.weight");
        b.attn_k_norm_w = get("attn.k_norm.weight");
        b.ffn_norm_w   = get("ffn_norm.weight");
        b.ffn_gate_w   = get("ffn.gate.weight");
        b.ffn_up_w     = get("ffn.up.weight");
        b.ffn_down_w   = get("ffn.down.weight");
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
    fprintf(stderr, "moss_audio: loaded %u enc layers (d=%u), adapter (hidden=%u), "
            "%u deepstack taps, %u LLM layers (d=%u), vocab=%u\n",
            hp.enc_layers, hp.enc_d_model, hp.adapter_hidden,
            hp.ds_num_taps, hp.llm_layers, hp.llm_hidden, hp.llm_vocab_size);

    return true;
}

// ===========================================================================
// FFT (same as qwen3_asr — needed for Whisper-style mel)
// ===========================================================================

static void moss_audio_dft(const float* in, int N, float* out) {
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

static void moss_audio_fft(float* in, int N, float* out) {
    if (N == 1) { out[0] = in[0]; out[1] = 0.0f; return; }
    int half_N = N / 2;
    if (N - half_N * 2 == 1) { moss_audio_dft(in, N, out); return; }
    std::vector<float> even_in(half_N), odd_in(half_N);
    for (int i = 0; i < half_N; i++) { even_in[i] = in[2 * i]; odd_in[i] = in[2 * i + 1]; }
    std::vector<float> even_out(2 * half_N), odd_out(2 * half_N);
    moss_audio_fft(even_in.data(), half_N, even_out.data());
    moss_audio_fft(odd_in.data(), half_N, odd_out.data());
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
// Mel spectrogram (Whisper-style, 128-bin)
// ===========================================================================

extern "C" float* moss_audio_compute_mel(struct moss_audio_context* ctx,
                                         const float* samples, int n_samples,
                                         int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_mels_val = (int)hp.n_mels;
    const int n_freqs = n_fft / 2 + 1;

    // Hann window
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; i++)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft));

    // Mel filterbank — Whisper uses slaney mel scale with slaney area norm
    // (librosa default: htk=False, norm='slaney'). core_mel::build_slaney_fb
    // matches this exactly.
    std::vector<float> mel_filters = core_mel::build_slaney_fb(
        (int)hp.sample_rate, n_fft, n_mels_val, 0.0f, -1.0f,
        core_mel::FbLayout::MelsFreqs);

    // STFT + mel + log via core_mel
    core_mel::FftR2C fft_fn = [](const float* in, int N, float* out) {
        moss_audio_fft(const_cast<float*>(in), N, out);
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
    mel_params.center_pad_reflect = true; // WhisperFeatureExtractor uses reflect padding

    int T_mel_actual = 0;
    std::vector<float> mel_out = core_mel::compute(
        samples, n_samples, hann.data(), n_fft,
        mel_filters.data(), n_freqs, fft_fn, mel_params, T_mel_actual);

    // Pad to 3000 frames (30s Whisper convention) — WhisperFeatureExtractor
    // always pads to nb_max_frames=3000. The encoder chunks this into 400-frame
    // pieces, so the padding adds extra zero-content chunks that produce
    // near-zero encoder output (masked by the padding mask). Without this
    // padding, the number of encoder tokens differs from the Python reference.
    const int T_padded = 3000;
    if (T_mel_actual < T_padded) {
        std::vector<float> padded((size_t)n_mels_val * T_padded, 0.0f);
        // mel_out is (n_mels, T_mel_actual) row-major
        for (int f = 0; f < n_mels_val; f++) {
            memcpy(padded.data() + (size_t)f * T_padded,
                   mel_out.data() + (size_t)f * T_mel_actual,
                   (size_t)T_mel_actual * sizeof(float));
        }
        mel_out = std::move(padded);
        T_mel_actual = T_padded;
    }

    float* result = (float*)malloc(mel_out.size() * sizeof(float));
    memcpy(result, mel_out.data(), mel_out.size() * sizeof(float));
    if (out_n_mels) *out_n_mels = n_mels_val;
    if (out_T_mel) *out_T_mel = T_mel_actual;
    return result;
}

// ===========================================================================
// Audio encoder graph
// ===========================================================================

// Conv2d stride-2 downsampling: input (n_mels, T) → output shape after 3 convs
static int conv_out_len(int L) { return (L - 1) / 2 + 1; }

static ggml_cgraph* moss_audio_build_encoder_graph(
    moss_audio_context* ctx, int T_mel,
    // Output: encoder hidden states + deepstack taps
    bool capture_deepstack)
{
    const auto& hp = ctx->model.hparams;
    const auto& enc = ctx->model.enc;
    const int n_mels = (int)hp.n_mels;
    const int d = (int)hp.enc_d_model;
    const int n_heads = (int)hp.enc_n_heads;
    const int head_dim = (int)hp.enc_head_dim;
    (void)hp.enc_ffn_dim; // used by encoder layers

    // Downsampled temporal dimension after 3× stride-2 convs
    int T1 = conv_out_len(T_mel);
    int T2 = conv_out_len(T1);
    int T_down = conv_out_len(T2);

    // Estimate graph size
     // 512 MB compute buffer estimate
    struct ggml_init_params gparams = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: mel spectrogram. Data is (n_mels, T_mel) row-major F32.
    // PyTorch conv2d: input (N=1, IC=1, H=n_mels, W=T_mel).
    // ggml conv2d: input ne=(IW, IH, IC, N) where IW=ne[0] is stride-s0 dim.
    //
    // The mel data layout: mel[f*T + t] = mel value at freq f, time t.
    // ggml tensor ne=(T_mel, n_mels): element at (t, f) = mel[f*T + t]. ✓
    // BUT: ggml_conv_2d im2col treats ne[0] as the FIRST spatial dim
    // and ne[1] as the SECOND. The PyTorch kernel is (OC, IC, KH=freq, KW=time).
    // In ggml ne: (KW=time, KH=freq, IC, OC). So:
    //   - s0/p0/d0 apply to ne[0] = T_mel (time) via KW
    //   - s1/p1/d1 apply to ne[1] = n_mels (freq) via KH
    // This is correct.
    //
    // Alternative test: swap H and W to see if that fixes the values.
    // If mel is stored as ne=(n_mels, T_mel), then:
    //   - ne[0] = n_mels → s0 applies to freq
    //   - ne[1] = T_mel → s1 applies to time
    // This SWAPS the conv kernel directions. Only correct if the kernel
    // is symmetric (it's 3×3 so it could be close but not identical).
    // Input mel: ggml ne=(n_mels, T_mel, 1, 1).
    // This means ne[0]=n_mels (freq) varies fastest.
    // ggml conv2d applies kernel ne[0]=KW to data ne[0]=n_mels.
    // PyTorch kernel: (OC, IC, KH, KW) stored as ggml ne=(KW, KH, IC, OC).
    // PyTorch conv: KH applies to H=freq, KW applies to W=time.
    // So to match: data ne[0] must be the KW direction.
    // If KW=time and data ne[0]=n_mels=freq, the kernel is transposed.
    // FIX: store data as ne=(n_mels, T) so KW (ne[0]) applies to freq.
    // Then transpose the kernel to swap KH/KW...
    // OR: keep kernel as-is and swap the input so ne[0]=freq, ne[1]=time.
    // With the kernel's KW applying to freq (ne[0]) and KH to time (ne[1]):
    //   effectively: KH_eff=KW_orig (time→freq), KW_eff=KH_orig (freq→time)
    //   This is a transposition of the kernel's spatial dims.
    // Since the Python kernel has KH=freq,KW=time, and ggml applies
    // KW(ne[0]) to ne[0], we need data ne[0]=time to match.
    // So ne=(T_mel, n_mels) is correct... unless ggml im2col transposes.
    //
    // Let's try ne=(n_mels, T_mel) to test the hypothesis:
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel_in, "mel_input");
    ggml_set_input(mel_in);
    ggml_tensor* x = ggml_reshape_4d(ctx0, mel_in, n_mels, T_mel, 1, 1);

    // Transpose conv kernels: GGUF stores (KW, KH, IC, OC) but with
    // data ne[0]=freq we need (KH, KW, IC, OC) so KH (freq kernel)
    // aligns with data ne[0].
    auto conv_transpose_kernel = [&](ggml_tensor* w) -> ggml_tensor* {
        return ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3));
    };

    // Conv1
    x = ggml_conv_2d(ctx0, conv_transpose_kernel(enc.conv1_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, enc.conv1_b, 1, 1, enc.conv1_b->ne[0], 1));
    x = ggml_gelu_erf(ctx0, x);

    // Conv2
    x = ggml_conv_2d(ctx0, conv_transpose_kernel(enc.conv2_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, enc.conv2_b, 1, 1, enc.conv2_b->ne[0], 1));
    x = ggml_gelu_erf(ctx0, x);

    // Conv3
    x = ggml_conv_2d(ctx0, conv_transpose_kernel(enc.conv3_w), x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, enc.conv3_b, 1, 1, enc.conv3_b->ne[0], 1));
    x = ggml_gelu_erf(ctx0, x);

    // After conv3: ggml ne = (T_down, F_down, C_out=480, B=1)
    //   = PyTorch shape (B=1, C=480, F=F_down, T=T_down)
    //
    // Python: x.permute(0, 3, 1, 2).contiguous().flatten(2)
    //   (B, C, F, T) → (B, T, C, F) → (B, T, C*F=7680)
    //
    // ggml_permute semantics: output dim i reads from input dim ax_i.
    // Source ne = (T, F, C, B). Target ne = (F, C, T, B) so flatten
    // gives F*C with F fastest — matches PyTorch's c*F+f indexing.
    //   permute(1, 2, 0, 3): out[0]=in[1](F), out[1]=in[2](C), out[2]=in[0](T)
    // After conv3 with ne=(n_mels, T) input:
    //   ne = (F_down=16, T_down=50, C=480, B=1)
    // PyTorch: (B, C, T_down, F_down) → permute(0,2,1,3) → (B, F, C, T)...
    // Actually: PyTorch conv output is (B, C, OH, OW). With input (B, IC, H=freq, W=time):
    //   OH = freq_down, OW = time_down. So PyTorch output = (1, 480, F_down, T_down).
    //
    // Python: x.permute(0, 3, 1, 2) = (B, T_down, C, F_down) → flatten(2) = (B, T_down, C*F_down)
    //
    // Our ggml: ne = (F_down=16, T_down=50, C=480, B=1)
    // Want: ne = (C*F_down=7680, T_down=50, B=1)
    // Need to merge ne[0]=F_down and ne[2]=C with F_down fastest.
    // Permute to (F_down, C, T_down, B): permute(x, 0, 2, 1, 3)
    // Then cont+reshape to (F_down*C, T_down, B)
    int F_down = conv_out_len(conv_out_len(conv_out_len(n_mels)));
    int C_out = (int)hp.enc_ds_hidden;
    int stem_in = F_down * C_out;
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // (F, C, T, B)
    x = ggml_reshape_2d(ctx0, x, stem_in, T_down);

    // stem_proj: Linear(7680, 1280)
    x = ggml_mul_mat(ctx0, enc.stem_proj_w, x);
    x = ggml_add(ctx0, x, enc.stem_proj_b);

    // Capture stem_proj output for diff validation
    ggml_tensor* stem_out = ggml_cont(ctx0, x);
    ggml_set_name(stem_out, "enc_post_stem_proj");
    ggml_set_output(stem_out);

    // Positional embedding (sinusoidal, precomputed)
    ggml_tensor* pe_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_down);
    ggml_set_name(pe_in, "pos_embed");
    ggml_set_input(pe_in);
    x = ggml_add(ctx0, x, pe_in);

    // Padding mask: (T_down, T_down) F16. For key positions that are padded,
    // the mask is -inf; for valid positions, 0. Used by flash_attn_ext.
    // Bidirectional (encoder), so all valid positions attend to all valid positions.
    ggml_tensor* attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_down, T_down);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // 32 WhisperEncoderLayers
    ggml_tensor* ds_taps[3] = {nullptr, nullptr, nullptr};

    for (uint32_t il = 0; il < hp.enc_layers; il++) {
        const auto& blk = enc.blocks[il];
        ggml_tensor* residual = x;

        // Pre-LN self-attention
        ggml_tensor* h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.attn_norm_w), blk.attn_norm_b);

        // Q, K, V projections
        ggml_tensor* Q = ggml_mul_mat(ctx0, blk.attn_q_w, h);
        if (blk.attn_q_b) Q = ggml_add(ctx0, Q, blk.attn_q_b);
        ggml_tensor* K = ggml_mul_mat(ctx0, blk.attn_k_w, h);
        ggml_tensor* V = ggml_mul_mat(ctx0, blk.attn_v_w, h);
        if (blk.attn_v_b) V = ggml_add(ctx0, V, blk.attn_v_b);

        // Reshape to (head_dim, n_heads, T), then permute to (hd, T, n_h)
        Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, T_down);
        K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, T_down);
        V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T_down);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3)); // (hd, T, n_h)
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

        // Manual self-attention (matching qwen3_asr encoder pattern):
        // scores = Q @ K^T, shape (T, T, n_heads)
        float attn_scale = 1.0f / std::sqrt((float)head_dim);
        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (T, T, n_h)
        scores = ggml_add(ctx0, scores, attn_mask);
        scores = ggml_soft_max_ext(ctx0, scores, nullptr, attn_scale, 0.0f);

        // attn = scores @ V: permute V to (T, hd, n_h) for dot over T
        ggml_tensor* V2 = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3));
        ggml_tensor* attn = ggml_mul_mat(ctx0, V2, scores); // (hd, T, n_h)

        // Reshape: (hd, T, n_h) → (hd, n_h, T) → (d, T)
        attn = ggml_cont(ctx0, ggml_permute(ctx0, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(ctx0, attn, d, T_down);

        // Output projection
        ggml_tensor* attn_out = ggml_mul_mat(ctx0, blk.attn_o_w, attn);
        if (blk.attn_o_b) attn_out = ggml_add(ctx0, attn_out, blk.attn_o_b);

        x = ggml_add(ctx0, residual, attn_out);

        // Pre-LN FFN
        residual = x;
        h = ggml_norm(ctx0, x, hp.enc_ln_eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, blk.ffn_norm_w), blk.ffn_norm_b);
        h = ggml_mul_mat(ctx0, blk.ffn_fc1_w, h);
        h = ggml_add(ctx0, h, blk.ffn_fc1_b);
        h = ggml_gelu_erf(ctx0, h);
        h = ggml_mul_mat(ctx0, blk.ffn_fc2_w, h);
        h = ggml_add(ctx0, h, blk.ffn_fc2_b);

        x = ggml_add(ctx0, residual, h);

        // DeepStack tap capture
        if (capture_deepstack) {
            for (uint32_t t = 0; t < hp.ds_num_taps; t++) {
                if (il == hp.ds_tap_layers[t]) {
                    ds_taps[t] = ggml_cont(ctx0, x);
                    char name[32];
                    snprintf(name, sizeof(name), "ds_tap_%u", t);
                    ggml_set_name(ds_taps[t], name);
                    ggml_set_output(ds_taps[t]);
                }
            }
        }
    }

    // Final layer norm
    x = ggml_norm(ctx0, x, hp.enc_ln_eps);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, enc.norm_w), enc.norm_b);

    // out_proj is Identity for 4B (output_dim == d_model)

    ggml_set_name(x, "encoder_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, stem_out);

    // Also build forward for deepstack taps
    for (int t = 0; t < 3; t++) {
        if (ds_taps[t]) ggml_build_forward_expand(gf, ds_taps[t]);
    }

    return gf;
}

// ===========================================================================
// Adapter + DeepStack projection graph
// ===========================================================================

static ggml_cgraph* moss_audio_build_adapter_graph(
    moss_audio_context* ctx, int T_enc)
{
    const auto& hp = ctx->model.hparams;
    const int d_enc = (int)hp.enc_d_model;

    struct ggml_init_params gparams = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Input: encoder output (T_enc, d_enc)
    ggml_tensor* enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, T_enc);
    ggml_set_name(enc_out, "enc_output_in");
    ggml_set_input(enc_out);

    // Audio adapter: GatedMLP (SwiGLU)
    ggml_tensor* adapted = core_ffn::swiglu(ctx0, enc_out,
        ctx->model.adapter.gate_w, ctx->model.adapter.up_w, ctx->model.adapter.down_w);
    ggml_set_name(adapted, "adapter_output");
    ggml_set_output(adapted);
    ggml_build_forward_expand(gf, adapted);

    // DeepStack projections
    for (uint32_t i = 0; i < hp.ds_num_taps && i < 3; i++) {
        ggml_tensor* ds_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_enc, T_enc);
        char name[32];
        snprintf(name, sizeof(name), "ds_tap_%u_in", i);
        ggml_set_name(ds_in, name);
        ggml_set_input(ds_in);

        ggml_tensor* ds_proj = core_ffn::swiglu(ctx0, ds_in,
            ctx->model.deepstack[i].gate_w, ctx->model.deepstack[i].up_w,
            ctx->model.deepstack[i].down_w);
        snprintf(name, sizeof(name), "ds_proj_%u", i);
        ggml_set_name(ds_proj, name);
        ggml_set_output(ds_proj);
        ggml_build_forward_expand(gf, ds_proj);
    }

    return gf;
}

// ===========================================================================
// Encoder + Adapter execution
// ===========================================================================

extern "C" float* moss_audio_run_encoder(struct moss_audio_context* ctx,
                                          const float* mel, int n_mels, int T_mel,
                                          int* out_T_enc, int* out_d,
                                          float** ds_tap_0, float** ds_tap_1, float** ds_tap_2) {
    if (!ctx || !mel) return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.enc_d_model;
    const bool want_ds = (ds_tap_0 || ds_tap_1 || ds_tap_2);

    // ---- Chunked encoder processing ----
    // The Python reference splits the mel into chunks of chunk_frames=400,
    // pads each to chunk_frames, runs conv+transformer independently per
    // chunk (no cross-chunk attention), then selects valid output tokens.
    const int chunk_frames = 400; // n_window(200) * 2

    // Compute chunk boundaries
    int num_chunks = (T_mel + chunk_frames - 1) / chunk_frames;
    std::vector<int> chunk_lengths(num_chunks, chunk_frames);
    int tail = T_mel % chunk_frames;
    if (tail > 0) chunk_lengths[num_chunks - 1] = tail;
    // If tail == 0, last chunk is full (chunk_frames)

    // For each chunk, compute valid output length after 3× stride-2 conv
    std::vector<int> valid_lens(num_chunks);
    int total_valid = 0;
    for (int c = 0; c < num_chunks; c++) {
        valid_lens[c] = conv_out_len(conv_out_len(conv_out_len(chunk_lengths[c])));
        total_valid += valid_lens[c];
    }

    // Padded chunk conv output length (all chunks padded to chunk_frames)
    const int T_chunk_down = conv_out_len(conv_out_len(conv_out_len(chunk_frames)));

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_audio: encoder chunking: %d chunks of %d frames, "
                "total_valid=%d tokens (T_chunk_down=%d)\n",
                num_chunks, chunk_frames, total_valid, T_chunk_down);

    // Allocate output buffers
    float* result = (float*)malloc((size_t)d * total_valid * sizeof(float));
    float* ds_results[3] = {nullptr, nullptr, nullptr};
    if (want_ds) {
        if (ds_tap_0) ds_results[0] = (float*)malloc((size_t)d * total_valid * sizeof(float));
        if (ds_tap_1) ds_results[1] = (float*)malloc((size_t)d * total_valid * sizeof(float));
        if (ds_tap_2) ds_results[2] = (float*)malloc((size_t)d * total_valid * sizeof(float));
    }

    int out_offset = 0; // write position in output buffers

    for (int c = 0; c < num_chunks; c++) {
        // Prepare padded mel chunk for ggml ne=(T=chunk_frames, n_mels).
        // ggml ne[0]=T varies fastest: data[t + chunk_frames * f].
        // Input mel is (n_mels, T_mel) row-major: mel[f * T_mel + t].
        std::vector<float> chunk_mel((size_t)n_mels * chunk_frames, 0.0f);
        int t_start = c * chunk_frames;
        int t_len = chunk_lengths[c];
        // Pack mel with freq (n_mels) as ne[0] (fastest) and time as ne[1].
        // This transposes from (n_mels, T) row-major to (T, n_mels) ne-order
        // = ggml ne=(n_mels, T). ggml conv2d applies KW (kernel ne[0]) along
        // data ne[0]=n_mels and KH (ne[1]) along data ne[1]=T.
        for (int t = 0; t < t_len; t++) {
            for (int f = 0; f < n_mels; f++) {
                chunk_mel[(size_t)f + n_mels * (size_t)t] =
                    mel[(size_t)f * T_mel + t_start + t];
            }
        }

        // Build and run encoder graph for this chunk
        ggml_cgraph* gf = moss_audio_build_encoder_graph(ctx, chunk_frames, want_ds);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            fprintf(stderr, "moss_audio: encoder graph alloc failed (chunk %d)\n", c);
            free(result);
            for (int t = 0; t < 3; t++) free(ds_results[t]);
            return nullptr;
        }

        // Set mel input
        ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel_input");
        ggml_backend_tensor_set(mel_in, chunk_mel.data(), 0,
                                (size_t)n_mels * chunk_frames * sizeof(float));
        if (c == 0 && ctx->params.verbosity >= 1) {
            // Verify: chunk_mel[0] = mel(f=0,t=0), chunk_mel[T] = mel(f=1,t=0)
            fprintf(stderr, "moss_audio: mel_pack check: [0]=%f [T=%d]=%f [1]=%f\n",
                    chunk_mel[0], chunk_frames, chunk_mel[chunk_frames], chunk_mel[1]);
        }

        // Set padding mask: bidirectional, but mask out padded positions
        // Valid positions [0, valid_lens[c]) can attend to each other;
        // padded positions [valid_lens[c], T_chunk_down) are masked with -inf.
        ggml_tensor* mask_t = ggml_graph_get_tensor(gf, "attn_mask");
        if (mask_t) {
            int valid = valid_lens[c];
            std::vector<float> mask_data((size_t)T_chunk_down * T_chunk_down);
            for (int q = 0; q < T_chunk_down; q++) {
                for (int k = 0; k < T_chunk_down; k++) {
                    // Mask: valid queries attend to valid keys only.
                    // Padded queries attend to ALL keys (avoid NaN softmax;
                    // their output is discarded anyway).
                    mask_data[(size_t)q * T_chunk_down + k] =
                        (q >= valid) ? 0.0f : (k < valid ? 0.0f : -INFINITY);
                }
            }
            ggml_backend_tensor_set(mask_t, mask_data.data(), 0,
                                    mask_data.size() * sizeof(float));
        }

        // Set positional embedding for this chunk
        ggml_tensor* pe_in = ggml_graph_get_tensor(gf, "pos_embed");
        if (pe_in) {
            const size_t pe_bytes = (size_t)d * T_chunk_down * sizeof(float);
            if ((size_t)T_chunk_down * d <= ctx->model.audio_pe.size()) {
                ggml_backend_tensor_set(pe_in, ctx->model.audio_pe.data(), 0, pe_bytes);
            } else {
                std::vector<float> pe_buf((size_t)d * T_chunk_down, 0.0f);
                size_t copy = std::min(ctx->model.audio_pe.size(), pe_buf.size());
                memcpy(pe_buf.data(), ctx->model.audio_pe.data(), copy * sizeof(float));
                ggml_backend_tensor_set(pe_in, pe_buf.data(), 0, pe_bytes);
            }
        }

        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "moss_audio: encoder graph compute failed (chunk %d)\n", c);
            free(result);
            for (int t = 0; t < 3; t++) free(ds_results[t]);
            return nullptr;
        }

        // Diagnostic: dump stem_proj for chunk 0
        if (c == 0 && ctx->params.verbosity >= 1) {
            ggml_tensor* stem_t = ggml_graph_get_tensor(gf, "enc_post_stem_proj");
            if (stem_t) {
                std::vector<float> stem_buf(ggml_nelements(stem_t));
                ggml_backend_tensor_get(stem_t, stem_buf.data(), 0, stem_buf.size() * sizeof(float));
                fprintf(stderr, "moss_audio: stem_proj[0] first8=[%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f] absmax=%.2f\n",
                        stem_buf[0], stem_buf[1], stem_buf[2], stem_buf[3],
                        stem_buf[4], stem_buf[5], stem_buf[6], stem_buf[7],
                        *std::max_element(stem_buf.begin(), stem_buf.end(),
                            [](float a, float b){ return fabsf(a) < fabsf(b); }));
            }
        }

        // Extract encoder output — only valid_lens[c] tokens (not padded ones)
        ggml_tensor* enc_out = ggml_graph_get_tensor(gf, "encoder_output");
        if (!enc_out) {
            fprintf(stderr, "moss_audio: missing encoder_output (chunk %d)\n", c);
            free(result);
            for (int t = 0; t < 3; t++) free(ds_results[t]);
            return nullptr;
        }

        int valid = valid_lens[c];
        // enc_out shape: ne=(d, T_chunk_down). Copy only first `valid` rows.
        // Row t starts at offset t * d (ggml ne[0]=d is the fast dim).
        ggml_backend_tensor_get(enc_out, result + (size_t)out_offset * d,
                                0, (size_t)valid * d * sizeof(float));

        // Extract deepstack taps (same valid subset)
        if (want_ds) {
            for (int t = 0; t < 3; t++) {
                if (!ds_results[t]) continue;
                char name[32];
                snprintf(name, sizeof(name), "ds_tap_%d", t);
                ggml_tensor* tap = ggml_graph_get_tensor(gf, name);
                if (tap) {
                    ggml_backend_tensor_get(tap, ds_results[t] + (size_t)out_offset * d,
                                            0, (size_t)valid * d * sizeof(float));
                }
            }
        }

        out_offset += valid;
    }

    if (out_T_enc) *out_T_enc = total_valid;
    if (out_d) *out_d = d;
    if (ds_tap_0) *ds_tap_0 = ds_results[0];
    if (ds_tap_1) *ds_tap_1 = ds_results[1];
    if (ds_tap_2) *ds_tap_2 = ds_results[2];

    return result;
}

extern "C" float* moss_audio_run_adapter(struct moss_audio_context* ctx,
                                          const float* encoder_out, int T_enc, int d_enc,
                                          int* out_T, int* out_d) {
    if (!ctx || !encoder_out) return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_hidden;

    ggml_cgraph* gf = moss_audio_build_adapter_graph(ctx, T_enc);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_audio: adapter graph alloc failed\n");
        return nullptr;
    }

    ggml_tensor* in = ggml_graph_get_tensor(gf, "enc_output_in");
    ggml_backend_tensor_set(in, encoder_out, 0, (size_t)d_enc * T_enc * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_audio: adapter graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "adapter_output");
    if (!out) return nullptr;

    if (out_T) *out_T = T_enc;
    if (out_d) *out_d = d_llm;

    float* result = (float*)malloc((size_t)d_llm * T_enc * sizeof(float));
    ggml_backend_tensor_get(out, result, 0, (size_t)d_llm * T_enc * sizeof(float));
    return result;
}

// Run deepstack mergers on captured taps. Returns 3 buffers of (T_enc, d_llm).
static void moss_audio_run_deepstack_mergers(
    moss_audio_context* ctx,
    float* ds_taps[3], int T_enc, int d_enc,
    float* ds_projs[3])
{
    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_hidden;

    // Reuse the adapter graph builder — it already has deepstack projection
    ggml_cgraph* gf = moss_audio_build_adapter_graph(ctx, T_enc);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_audio: deepstack merger alloc failed\n");
        return;
    }

    // Set the deepstack tap inputs
    for (uint32_t i = 0; i < hp.ds_num_taps && i < 3; i++) {
        if (!ds_taps[i]) continue;
        char name[32];
        snprintf(name, sizeof(name), "ds_tap_%u_in", i);
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (t) {
            ggml_backend_tensor_set(t, ds_taps[i], 0, (size_t)d_enc * T_enc * sizeof(float));
        }
    }

    // Also need to set encoder output (for the adapter part)
    // Use zeros — we only care about the deepstack projection outputs
    {
        std::vector<float> zeros((size_t)d_enc * T_enc, 0.0f);
        ggml_tensor* enc_in = ggml_graph_get_tensor(gf, "enc_output_in");
        if (enc_in)
            ggml_backend_tensor_set(enc_in, zeros.data(), 0, zeros.size() * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_audio: deepstack merger compute failed\n");
        return;
    }

    // Extract projected deepstack outputs
    for (uint32_t i = 0; i < hp.ds_num_taps && i < 3; i++) {
        char name[32];
        snprintf(name, sizeof(name), "ds_proj_%u", i);
        ggml_tensor* out = ggml_graph_get_tensor(gf, name);
        if (out) {
            ds_projs[i] = (float*)malloc((size_t)d_llm * T_enc * sizeof(float));
            ggml_backend_tensor_get(out, ds_projs[i], 0, (size_t)d_llm * T_enc * sizeof(float));
        } else {
            ds_projs[i] = nullptr;
        }
    }
}

// ===========================================================================
// LLM graph with KV cache (Qwen3)
// ===========================================================================

static ggml_cgraph* moss_audio_build_llm_kv_graph(
    moss_audio_context* ctx, int n_tokens, int n_past,
    bool last_token_only,
    // DeepStack injection: if non-null, add these as residuals at LM layers 0..2
    // for positions where audio_mask[pos] is true
    int n_audio_tokens,  // length of deepstack projections
    bool has_deepstack)
{
    const auto& hp = ctx->model.hparams;
    const auto& llm = ctx->model.llm;
    const int d = (int)hp.llm_hidden;
    const int n_heads = (int)hp.llm_n_heads;
    const int n_kv_heads = (int)hp.llm_n_kv_heads;
    const int head_dim = (int)hp.llm_head_dim;
    const int Lk = n_past + n_tokens;

    struct ggml_init_params gparams = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
    ggml_context* ctx0 = ggml_init(gparams);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Inputs
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

    // DeepStack injection: ds_full_0/1/2 tensors are pre-scattered on the
    // CPU side (zero at non-audio positions, deepstack projections at audio
    // positions). They are created inside the layer loop when has_deepstack
    // is true. No separate audio_mask input needed.
    (void)n_audio_tokens; // audio token count used by caller to build ds_full

    // KV self-attention params (Qwen3 style)
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

        // Pre-attn RMSNorm
        ggml_tensor* h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.attn_norm_w);

        // Self-attention with KV cache
        ggml_tensor* attn_out = core_attn::kv_self_attn(
            ctx0, gf, h,
            blk.attn_q_w, blk.attn_k_w, blk.attn_v_w, blk.attn_o_w,
            blk.attn_q_norm_w, blk.attn_k_norm_w,
            positions, causal_mask,
            ctx->kv_k, ctx->kv_v,
            (int)il, n_past, attn_p);

        cur = ggml_add(ctx0, residual, attn_out);

        // DeepStack injection: after LM layers 0, 1, 2, add the
        // corresponding projected encoder tap as a residual at audio
        // positions. The Python reference uses forward hooks that run
        // AFTER each layer's forward pass.
        //
        // Implementation: ds_full_N is a (d, n_tokens) tensor that is
        // zero everywhere except at audio positions where it holds the
        // deepstack projection. It's pre-built on the CPU side and
        // passed as a graph input. We simply add it to cur.
        if (has_deepstack && il < hp.ds_num_inject) {
            // ds_full_N contains the scattered deepstack projection
            // (zero at non-audio positions, ds_proj values at audio positions)
            char name[32];
            snprintf(name, sizeof(name), "ds_full_%u", il);
            ggml_tensor* ds_full = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, n_tokens);
            ggml_set_name(ds_full, name);
            ggml_set_input(ds_full);
            cur = ggml_add(ctx0, cur, ds_full);
        }

        // Pre-FFN RMSNorm + SwiGLU FFN
        residual = cur;
        h = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
        h = ggml_mul(ctx0, h, blk.ffn_norm_w);
        ggml_tensor* mlp = core_ffn::swiglu(ctx0, h, blk.ffn_gate_w, blk.ffn_up_w, blk.ffn_down_w);
        cur = ggml_add(ctx0, residual, mlp);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, hp.llm_rms_eps);
    cur = ggml_mul(ctx0, cur, llm.final_norm_w);

    // LM head — last token only for generation
    if (last_token_only && n_tokens > 1) {
        cur = ggml_view_2d(ctx0, cur, d, 1, cur->nb[1], (size_t)(n_tokens - 1) * cur->nb[1]);
    }
    cur = ggml_mul_mat(ctx0, llm.lm_head_w, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ===========================================================================
// KV cache management
// ===========================================================================

extern "C" bool moss_audio_kv_init(struct moss_audio_context* ctx, int max_ctx) {
    if (!ctx) return false;
    const auto& hp = ctx->model.hparams;
    const int n_layers = (int)hp.llm_layers;
    const int n_kv = (int)hp.llm_n_kv_heads;
    const int hd = (int)hp.llm_head_dim;

    ggml_type kv_type = core_attn::kv_dtype_from_env("moss_audio");

    struct ggml_init_params kv_params = { 2 * ggml_tensor_overhead(), nullptr, true };
    ctx->kv_ctx = ggml_init(kv_params);
    ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_type, hd, max_ctx, n_kv, n_layers);
    ggml_set_name(ctx->kv_k, "kv_k");
    ggml_set_name(ctx->kv_v, "kv_v");

    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) {
        fprintf(stderr, "moss_audio: kv alloc failed for max_ctx=%d\n", max_ctx);
        ggml_free(ctx->kv_ctx);
        ctx->kv_ctx = nullptr;
        return false;
    }
    ctx->kv_max_ctx = max_ctx;
    ctx->kv_n_used = 0;

    // Zero-init
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    return true;
}

extern "C" void moss_audio_kv_reset(struct moss_audio_context* ctx) {
    if (ctx && ctx->kv_buf) {
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        ctx->kv_n_used = 0;
    }
}

// ===========================================================================
// LLM prefill with DeepStack injection
// ===========================================================================

// Internal: run LLM prefill with per-layer deepstack injection.
// ds_full[0..2] are pre-scattered (d, n_tokens) F32 tensors (zero at
// non-audio positions, deepstack proj at audio positions). May be empty.
// Returns logits for the last token.
static float* moss_audio_run_llm_prefill_with_deepstack(
    moss_audio_context* ctx, const float* inputs_embeds, int n_tokens,
    const std::vector<float> ds_full[3], int n_audio_tokens,
    int* out_vocab_size)
{
    if (!ctx || !inputs_embeds || n_tokens <= 0 || !ctx->kv_k) return nullptr;
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_hidden;
    const int vocab = (int)hp.llm_vocab_size;
    const int n_past = 0; // prefill always starts at 0
    const int Lk = n_tokens;
    const bool has_ds = !ds_full[0].empty();

    ggml_cgraph* gf = moss_audio_build_llm_kv_graph(ctx, n_tokens, n_past,
                                                      /*last_token_only=*/true,
                                                      n_audio_tokens, has_ds);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_audio: llm prefill alloc failed\n");
        return nullptr;
    }

    // Set inputs_embeds
    ggml_tensor* emb_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(emb_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));

    // Positions
    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = i;
    ggml_backend_tensor_set(pos_in, positions.data(), 0, (size_t)n_tokens * sizeof(int32_t));

    // Causal mask
    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            for (int k = 0; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = (k <= q) ? zero_h : neginf_h;
            }
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    // Set DeepStack pre-scattered tensors
    if (has_ds) {
        for (uint32_t t = 0; t < hp.ds_num_inject && t < 3; t++) {
            if (ds_full[t].empty()) continue;
            char name[32];
            snprintf(name, sizeof(name), "ds_full_%u", t);
            ggml_tensor* ds_t = ggml_graph_get_tensor(gf, name);
            if (ds_t) {
                ggml_backend_tensor_set(ds_t, ds_full[t].data(), 0,
                                        ds_full[t].size() * sizeof(float));
            }
        }
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_audio: llm prefill compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t) return nullptr;

    if (out_vocab_size) *out_vocab_size = vocab;
    ctx->kv_n_used = n_tokens;

    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    return result;
}

// ===========================================================================
// Tokenizer (GPT-2 byte-level BPE via core_bpe, same as qwen3_asr)
// ===========================================================================

#include "core/bpe.h"

extern "C" int moss_audio_tokenize(struct moss_audio_context* ctx, const char* text,
                                   int32_t* out_tokens, int max_tokens) {
    if (!ctx || !text || !out_tokens || max_tokens <= 0) return 0;
    const auto& v = ctx->vocab;
    std::vector<int32_t> result;

    const std::string s = text;
    size_t i = 0;
    while (i < s.size()) {
        // 1. Special-token check: <|...|>
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

        // 2. Plain text segment up to next special token
        size_t j = i;
        if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|') j++;
        while (j < s.size()) {
            if (s[j] == '<' && j + 1 < s.size() && s[j + 1] == '|') {
                size_t end = s.find("|>", j + 2);
                if (end != std::string::npos) {
                    std::string special = s.substr(j, end + 2 - j);
                    if (v.token_to_id.find(special) != v.token_to_id.end()) break;
                }
            }
            j++;
        }
        std::string chunk = s.substr(i, j - i);
        i = j;
        if (chunk.empty()) continue;

        // 3. Pre-split on whitespace boundaries (GPT-2 style)
        size_t k = 0;
        while (k < chunk.size()) {
            size_t start = k;
            if (chunk[k] == ' ' || chunk[k] == '\t' || chunk[k] == '\n') k++;
            while (k < chunk.size() && chunk[k] != ' ' && chunk[k] != '\t' && chunk[k] != '\n') k++;
            if (k == start) k++;
            std::string pre(chunk, start, k - start);

            // 4. Byte-encode via GPT-2 table and BPE-merge
            std::string encoded = core_bpe::bytes_to_unicode(pre.data(), pre.size());
            core_bpe::bpe_one(v.token_to_id, v.merge_rank, encoded, result);
        }
    }

    int n = std::min((int)result.size(), max_tokens);
    std::memcpy(out_tokens, result.data(), (size_t)n * sizeof(int32_t));
    return n;
}

extern "C" const char* moss_audio_token_text(struct moss_audio_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return nullptr;
    return ctx->vocab.id_to_token[token_id].c_str();
}

// ===========================================================================
// Embed tokens
// ===========================================================================

extern "C" float* moss_audio_embed_tokens(struct moss_audio_context* ctx,
                                           const int32_t* token_ids, int n_tokens) {
    if (!ctx || !token_ids || n_tokens <= 0) return nullptr;
    const int d = (int)ctx->model.hparams.llm_hidden;

    // Build a tiny graph: embed_tokens lookup
    struct ggml_init_params gp = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
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
    if (ggml_backend_sched_alloc_graph(ctx->sched, gf) != true) {
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
// Run LLM with KV cache
// ===========================================================================

extern "C" float* moss_audio_run_llm_kv(struct moss_audio_context* ctx,
                                         const float* inputs_embeds, int n_tokens, int n_past,
                                         int* out_n_tokens, int* out_vocab_size) {
    if (!ctx || !inputs_embeds || n_tokens <= 0) return nullptr;
    if (!ctx->kv_k) {
        fprintf(stderr, "moss_audio: kv cache not initialized\n");
        return nullptr;
    }
    const auto& hp = ctx->model.hparams;
    const int d = (int)hp.llm_hidden;
    const int vocab = (int)hp.llm_vocab_size;
    const int Lk = n_past + n_tokens;

    ggml_cgraph* gf = moss_audio_build_llm_kv_graph(ctx, n_tokens, n_past,
                                                      /*last_token_only=*/true,
                                                      0, false);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "moss_audio: llm graph alloc failed\n");
        return nullptr;
    }

    // Set inputs
    ggml_tensor* emb_in = ggml_graph_get_tensor(gf, "inputs_embeds");
    ggml_backend_tensor_set(emb_in, inputs_embeds, 0, (size_t)d * n_tokens * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) positions[i] = n_past + i;
    ggml_backend_tensor_set(pos_in, positions.data(), 0, (size_t)n_tokens * sizeof(int32_t));

    if (n_tokens > 1) {
        ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal_mask");
        std::vector<ggml_fp16_t> mask((size_t)Lk * n_tokens);
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; q++) {
            int abs_q = n_past + q;
            for (int k = 0; k < Lk; k++) {
                mask[(size_t)q * Lk + k] = (k <= abs_q) ? zero_h : neginf_h;
            }
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "moss_audio: llm graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* logits_t = ggml_graph_get_tensor(gf, "logits");
    if (!logits_t) return nullptr;

    if (out_n_tokens) *out_n_tokens = 1; // last_token_only
    if (out_vocab_size) *out_vocab_size = vocab;

    float* result = (float*)malloc((size_t)vocab * sizeof(float));
    ggml_backend_tensor_get(logits_t, result, 0, (size_t)vocab * sizeof(float));
    ctx->kv_n_used = Lk;
    return result;
}

// ===========================================================================
// High-level process/transcribe
// ===========================================================================

// Build the chat-template prompt with audio tokens
static std::vector<int32_t> moss_audio_build_prompt(
    moss_audio_context* ctx, const char* text_prompt, int n_audio_tokens)
{
    const auto& hp = ctx->model.hparams;
    std::vector<int32_t> ids;

    // System prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    auto push_text = [&](const char* t) {
        int32_t buf[4096];
        int n = moss_audio_tokenize(ctx, t, buf, 4096);
        for (int i = 0; i < n; i++) ids.push_back(buf[i]);
    };

    // <|im_start|>
    ids.push_back(151644);
    push_text("system\nYou are a helpful assistant.");
    // <|im_end|>\n
    ids.push_back((int32_t)hp.eos_token_id);
    push_text("\n");

    // <|im_start|>user\n
    ids.push_back(151644);
    push_text("user\n");

    // <|audio_bos|> <|AUDIO|>×N with time markers <|audio_eos|>
    // Time markers: every 2 seconds (25 audio tokens at 12.5 Hz),
    // insert the second count as digit token IDs (0→15, ..., 9→24).
    ids.push_back((int32_t)hp.audio_start_id);
    {
        const int tokens_per_marker = 25; // 12.5 Hz × 2 seconds
        const int marker_interval = 2;    // seconds
        int audio_consumed = 0;
        float total_duration = (float)n_audio_tokens / 12.5f;
        int num_full_seconds = (int)total_duration;

        for (int sec = marker_interval; sec <= num_full_seconds; sec += marker_interval) {
            int marker_pos = (sec / marker_interval) * tokens_per_marker;
            int segment_len = marker_pos - audio_consumed;
            for (int i = 0; i < segment_len && audio_consumed < n_audio_tokens; i++) {
                ids.push_back((int32_t)hp.audio_token_id);
                audio_consumed++;
            }
            // Insert digit tokens for the second count
            // Digit token IDs: '0'→15, '1'→16, ..., '9'→24
            std::string sec_str = std::to_string(sec);
            for (char c : sec_str) {
                ids.push_back((int32_t)(15 + (c - '0')));
            }
        }
        // Remaining audio tokens after last marker
        while (audio_consumed < n_audio_tokens) {
            ids.push_back((int32_t)hp.audio_token_id);
            audio_consumed++;
        }
    }
    ids.push_back((int32_t)hp.audio_end_id);

    push_text("\n");
    push_text(text_prompt);
    // <|im_end|>\n
    ids.push_back((int32_t)hp.eos_token_id);
    push_text("\n");

    // <|im_start|>assistant\n
    ids.push_back(151644);
    push_text("assistant\n");

    return ids;
}

extern "C" char* moss_audio_process(struct moss_audio_context* ctx, const float* samples,
                                     int n_samples, const char* prompt) {
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    if (!prompt) prompt = "Transcribe this audio.";

    const auto& hp = ctx->model.hparams;
    const int d_llm = (int)hp.llm_hidden;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_audio: processing %d samples (%.1f sec), prompt=\"%s\"\n",
                n_samples, (float)n_samples / (float)hp.sample_rate, prompt);

    // 1. Mel spectrogram (or load from file for debugging)
    int n_mels = 0, T_mel = 0;
    float* mel = nullptr;
    const char* mel_override = std::getenv("MOSS_AUDIO_MEL_FILE");
    if (mel_override) {
        // Load pre-computed mel from raw F32 file (n_mels × T row-major)
        FILE* mf = fopen(mel_override, "rb");
        if (mf) {
            fseek(mf, 0, SEEK_END);
            size_t sz = (size_t)ftell(mf);
            fseek(mf, 0, SEEK_SET);
            n_mels = 128;
            T_mel = (int)(sz / sizeof(float) / n_mels);
            mel = (float*)malloc(sz);
            fread(mel, 1, sz, mf);
            fclose(mf);
            fprintf(stderr, "moss_audio: loaded mel from %s (%d × %d)\n", mel_override, n_mels, T_mel);
        }
    }
    if (!mel) {
        mel = moss_audio_compute_mel(ctx, samples, n_samples, &n_mels, &T_mel);
    }
    if (!mel) { fprintf(stderr, "moss_audio: mel failed\n"); return nullptr; }

    // 2. Run audio encoder (with DeepStack tap capture)
    int T_enc = 0, enc_d = 0;
    float *ds_tap_0 = nullptr, *ds_tap_1 = nullptr, *ds_tap_2 = nullptr;
    float* encoder_out = moss_audio_run_encoder(ctx, mel, n_mels, T_mel,
                                                 &T_enc, &enc_d,
                                                 &ds_tap_0, &ds_tap_1, &ds_tap_2);
    free(mel);
    if (!encoder_out) { fprintf(stderr, "moss_audio: encoder failed\n"); return nullptr; }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "moss_audio: encoder done: %d frames × %d dims\n", T_enc, enc_d);
        // Print first few values of encoder output for diagnostic
        float sum = 0, absmax = 0;
        for (int i = 0; i < T_enc * enc_d; i++) {
            sum += encoder_out[i];
            float a = fabsf(encoder_out[i]);
            if (a > absmax) absmax = a;
        }
        fprintf(stderr, "moss_audio: encoder out: mean=%.6f absmax=%.6f first=[%.4f %.4f %.4f %.4f]\n",
                sum / (T_enc * enc_d), absmax,
                encoder_out[0], encoder_out[1], encoder_out[2], encoder_out[3]);
    }

    // 3. Run audio adapter (encoder output → LLM space)
    // DIAGNOSTIC: test adapter with small random input (simulates "correct" encoder)
    if (ctx->params.verbosity >= 2) {
        std::vector<float> small_in((size_t)T_enc * enc_d);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto& v : small_in) v = dist(rng);
        int zt = 0, zd = 0;
        float* small_out = moss_audio_run_adapter(ctx, small_in.data(), T_enc, enc_d, &zt, &zd);
        if (small_out) {
            float zm = 0, za = 0;
            for (int i = 0; i < zt * zd; i++) { zm += small_out[i]; float a = fabsf(small_out[i]); if (a > za) za = a; }
            fprintf(stderr, "moss_audio: adapter(N(0,0.1)): mean=%.6f absmax=%.6f\n", zm/(zt*zd), za);
            free(small_out);
        }
    }
    int adapt_T = 0, adapt_d = 0;
    float* audio_embeds = moss_audio_run_adapter(ctx, encoder_out, T_enc, enc_d,
                                                  &adapt_T, &adapt_d);
    if (!audio_embeds) {
        free(encoder_out);
        free(ds_tap_0); free(ds_tap_1); free(ds_tap_2);
        return nullptr;
    }

    if (ctx->params.verbosity >= 1) {
        float sum = 0, absmax = 0;
        for (int i = 0; i < T_enc * adapt_d; i++) {
            sum += audio_embeds[i];
            float a = fabsf(audio_embeds[i]);
            if (a > absmax) absmax = a;
        }
        fprintf(stderr, "moss_audio: adapter out: mean=%.6f absmax=%.6f\n",
                sum / (T_enc * adapt_d), absmax);
    }

    // 4. Run DeepStack mergers on captured taps
    float* ds_taps_arr[3] = { ds_tap_0, ds_tap_1, ds_tap_2 };
    float* ds_projs[3] = { nullptr, nullptr, nullptr };
    moss_audio_run_deepstack_mergers(ctx, ds_taps_arr, T_enc, enc_d, ds_projs);
    free(encoder_out);
    free(ds_tap_0); free(ds_tap_1); free(ds_tap_2);

    // 5. Build prompt token IDs and embed text tokens
    auto prompt_ids = moss_audio_build_prompt(ctx, prompt, T_enc);
    int n_prompt = (int)prompt_ids.size();

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "moss_audio: %d mel frames, %d enc tokens, %d prompt tokens\n",
                T_mel, T_enc, n_prompt);
        // Dump first and last non-audio tokens for debugging
        int n_audio = 0;
        for (int i = 0; i < n_prompt; i++)
            if (prompt_ids[i] == (int32_t)hp.audio_token_id) n_audio++;
        fprintf(stderr, "moss_audio: prompt: %d audio + %d text/special tokens\n",
                n_audio, n_prompt - n_audio);
        // Print first 15 and last 10 token IDs
        fprintf(stderr, "moss_audio: prompt first15: ");
        for (int i = 0; i < std::min(15, n_prompt); i++)
            fprintf(stderr, "%d ", prompt_ids[i]);
        fprintf(stderr, "\nmoss_audio: prompt last10: ");
        for (int i = std::max(0, n_prompt - 10); i < n_prompt; i++)
            fprintf(stderr, "%d ", prompt_ids[i]);
        fprintf(stderr, "\n");
    }

    float* text_embeds = moss_audio_embed_tokens(ctx, prompt_ids.data(), n_prompt);
    if (!text_embeds) {
        free(audio_embeds);
        for (int i = 0; i < 3; i++) free(ds_projs[i]);
        return nullptr;
    }

    // 6. Scatter audio embeddings into text embeddings at audio_token positions
    //    (masked_scatter in PyTorch: replace embed[audio_mask] with audio_embeds)
    {
        int audio_idx = 0;
        for (int pos = 0; pos < n_prompt; pos++) {
            if (prompt_ids[pos] == (int32_t)hp.audio_token_id) {
                if (audio_idx < T_enc) {
                    // Replace text embedding at this position with audio embedding
                    memcpy(text_embeds + (size_t)pos * d_llm,
                           audio_embeds + (size_t)audio_idx * d_llm,
                           (size_t)d_llm * sizeof(float));
                    audio_idx++;
                }
            }
        }
        if (ctx->params.verbosity >= 2)
            fprintf(stderr, "moss_audio: scattered %d audio embeds into prompt\n", audio_idx);
    }
    free(audio_embeds);

    if (ctx->params.verbosity >= 1) {
        // Check text_embeds stats after scatter
        float sum = 0, absmax = 0;
        int n_audio = 0;
        for (int pos = 0; pos < n_prompt; pos++) {
            if (prompt_ids[pos] == (int32_t)hp.audio_token_id) n_audio++;
            for (int j = 0; j < d_llm; j++) {
                float v = text_embeds[(size_t)pos * d_llm + j];
                sum += v;
                float a = fabsf(v);
                if (a > absmax) absmax = a;
            }
        }
        fprintf(stderr, "moss_audio: embeds after scatter: mean=%.6f absmax=%.6f n_audio=%d/%d\n",
                sum / ((size_t)n_prompt * d_llm), absmax, n_audio, n_prompt);
    }

    // 7. Build pre-scattered DeepStack tensors
    // DIAGNOSTIC: try without deepstack to isolate the issue
    bool skip_deepstack = false;
    //    For each of the 3 injection layers, we build a (d_llm, n_prompt)
    //    tensor that is zero everywhere except at audio-token positions,
    //    where it contains the projected deepstack embeddings. The LLM
    //    graph adds ds_full_N to the hidden state after layer N.
    std::vector<float> ds_full[3];
    for (uint32_t t = 0; t < hp.ds_num_inject && t < 3 && !skip_deepstack; t++) {
        if (!ds_projs[t]) continue;
        ds_full[t].assign((size_t)d_llm * n_prompt, 0.0f);
        int audio_idx = 0;
        for (int pos = 0; pos < n_prompt; pos++) {
            if (prompt_ids[pos] == (int32_t)hp.audio_token_id && audio_idx < T_enc) {
                memcpy(ds_full[t].data() + (size_t)pos * d_llm,
                       ds_projs[t] + (size_t)audio_idx * d_llm,
                       (size_t)d_llm * sizeof(float));
                audio_idx++;
            }
        }
    }
    for (int i = 0; i < 3; i++) free(ds_projs[i]);

    // 8. Initialize KV cache and run LLM prefill + decode
    int max_ctx = n_prompt + 512;
    if (ctx->kv_k) {
        // Reuse existing cache if large enough, otherwise reinit
        if (ctx->kv_max_ctx < max_ctx) {
            if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
            if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
            ctx->kv_buf = nullptr; ctx->kv_ctx = nullptr;
            ctx->kv_k = nullptr; ctx->kv_v = nullptr;
            moss_audio_kv_init(ctx, max_ctx);
        } else {
            moss_audio_kv_reset(ctx);
        }
    } else {
        moss_audio_kv_init(ctx, max_ctx);
    }

    // Prefill: run all prompt tokens through the LLM with DeepStack injection
    int vocab = 0;
    float* logits = moss_audio_run_llm_prefill_with_deepstack(
        ctx, text_embeds, n_prompt, ds_full, T_enc, &vocab);
    free(text_embeds);

    if (!logits) return nullptr;

    // 9. Greedy decode loop
    std::vector<int32_t> generated;
    int max_new = 512;
    for (int step = 0; step < max_new; step++) {
        // Argmax
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best_id = i; }
        }
        free(logits);
        logits = nullptr;

        if (ctx->params.verbosity >= 1 && step < 5)
            fprintf(stderr, "moss_audio: step %d argmax=%d (%.4f) token='%s'\n",
                    step, best_id, best_val,
                    (best_id >= 0 && best_id < (int)ctx->vocab.id_to_token.size())
                        ? ctx->vocab.id_to_token[best_id].c_str() : "?");
        if (best_id == (int)hp.eos_token_id) break;
        generated.push_back(best_id);

        // Embed next token and run one decode step
        float* next_emb = moss_audio_embed_tokens(ctx, &best_id, 1);
        if (!next_emb) break;

        int dummy_n = 0;
        logits = moss_audio_run_llm_kv(ctx, next_emb, 1,
                                        n_prompt + (int)generated.size() - 1,
                                        &dummy_n, &vocab);
        free(next_emb);
        if (!logits) break;
    }
    if (logits) free(logits);

    // 10. Decode tokens to text
    std::string result;
    for (int id : generated) {
        const char* t = moss_audio_token_text(ctx, id);
        if (t) result += t;
    }

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "moss_audio: generated %zu tokens: \"%s\"\n",
                generated.size(), result.substr(0, 100).c_str());

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

extern "C" char* moss_audio_transcribe(struct moss_audio_context* ctx,
                                        const float* samples, int n_samples) {
    return moss_audio_process(ctx, samples, n_samples, "Transcribe this audio.");
}

// ===========================================================================
// Init / Free
// ===========================================================================

extern "C" struct moss_audio_context_params moss_audio_context_default_params(void) {
    moss_audio_context_params p;
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.flash_attn = false;
    return p;
}

extern "C" struct moss_audio_context* moss_audio_init_from_file(
    const char* path_model, struct moss_audio_context_params params)
{
    auto* ctx = new moss_audio_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads;
    ctx->model_path = path_model;

    // Backend selection
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Load model
    if (!moss_audio_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "moss_audio: failed to load model from %s\n", path_model);
        moss_audio_free(ctx);
        return nullptr;
    }

    // Scheduler
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

extern "C" void moss_audio_free(struct moss_audio_context* ctx) {
    if (!ctx) return;
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf) ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" void moss_audio_set_seed(struct moss_audio_context* ctx, uint32_t seed) {
    if (ctx) {
        ctx->seed = seed;
        ctx->rng.seed(seed);
    }
}
