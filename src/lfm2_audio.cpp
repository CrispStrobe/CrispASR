// lfm2_audio.cpp — LiquidAI LFM2.5-Audio ggml runtime
//
// Architecture:
//   Mel:         128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann, NeMo-style)
//   Encoder:     17× FastConformer (d_model=512, 8H, rel_pos, dw_striding 8×)
//   Adapter:     LayerNorm(512) → Linear(512→2048) → GELU → Linear(2048→2048)
//   LFM2:        16L hybrid backbone (10 conv + 6 GQA attn), hidden=2048
//   Depthformer: 6L (dim=1024), 8 codebooks → Mimi audio tokens
//
// DRY: reuses core/gguf_loader.h, core/mel.h, core/fastconformer.h, core/attention.h

#include "lfm2_audio.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/fastconformer.h"
#include "core/gguf_loader.h"
#include "core/mel.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/fft.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct lfm2_audio_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t codebooks = 8;
    uint32_t audio_vocab_size = 2049;

    // FastConformer encoder
    uint32_t enc_n_layers = 17;
    uint32_t enc_d_model = 512;
    uint32_t enc_n_heads = 8;
    uint32_t enc_ff_expansion = 4;
    uint32_t enc_conv_kernel = 9;
    uint32_t enc_subsampling_factor = 8;
    uint32_t enc_subsampling_channels = 256;

    // LFM2 backbone
    uint32_t lfm_hidden_size = 2048;
    uint32_t lfm_n_layers = 16;
    uint32_t lfm_n_heads = 32;
    uint32_t lfm_n_kv_heads = 8;
    uint32_t lfm_head_dim = 64;
    uint32_t lfm_ff_dim = 8192;
    uint32_t lfm_conv_kernel = 3;
    float lfm_rope_theta = 1000000.0f;
    std::string lfm_layer_types; // "ccaccaccacacacac"

    // Depthformer
    uint32_t depth_n_layers = 6;
    uint32_t depth_dim = 1024;
    uint32_t depth_tie = 1;
    uint32_t text_vocab_size = 65536;
    uint32_t interleaved_n_text = 6;
    uint32_t interleaved_n_audio = 9;
};

// ===========================================================================
// LFM2 backbone per-layer weights
// ===========================================================================

struct lfm2_layer_weights {
    ggml_tensor* operator_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor *ff_w1 = nullptr, *ff_w2 = nullptr, *ff_w3 = nullptr;

    bool is_attention = false;

    // Conv layers
    ggml_tensor* conv_conv_w = nullptr;     // [hidden, 1, kernel]
    ggml_tensor* conv_in_proj_w = nullptr;  // [3*hidden, hidden]
    ggml_tensor* conv_out_proj_w = nullptr; // [hidden, hidden]

    // Attention layers
    ggml_tensor* attn_q_proj_w = nullptr;
    ggml_tensor* attn_k_proj_w = nullptr;
    ggml_tensor* attn_v_proj_w = nullptr;
    ggml_tensor* attn_out_proj_w = nullptr;
    ggml_tensor* attn_q_ln_w = nullptr;
    ggml_tensor* attn_k_ln_w = nullptr;
};

// ===========================================================================
// Model
// ===========================================================================

struct lfm2_audio_model {
    lfm2_audio_hparams hparams;

    // FastConformer encoder (reused from core/fastconformer.h)
    core_conformer::PreEncodeWeights pre_enc;
    std::vector<core_conformer::BlockWeights> enc_blocks;

    // Audio adapter MLP
    ggml_tensor *adapter_norm_w = nullptr, *adapter_norm_b = nullptr;
    ggml_tensor *adapter_lin0_w = nullptr, *adapter_lin0_b = nullptr;
    ggml_tensor *adapter_lin1_w = nullptr, *adapter_lin1_b = nullptr;

    // LFM2 backbone
    ggml_tensor* lfm_embed_tokens_w = nullptr;
    ggml_tensor* lfm_embedding_norm_w = nullptr;
    std::vector<lfm2_layer_weights> lfm_layers;

    // Mel preprocessor (from GGUF — librosa slaney mel filterbank + Hann window)
    ggml_tensor* mel_fb = nullptr;     // (n_mels, n_freqs)
    ggml_tensor* mel_window = nullptr; // (win_length,)

    // Audio embedding (for audio output tokens — shared across codebooks)
    ggml_tensor* audio_embd_embedding_w = nullptr;
    ggml_tensor* audio_embd_embedding_norm_w = nullptr;
    ggml_tensor* audio_embd_to_logits_w = nullptr;

    // Depthformer (generates 8-codebook Mimi tokens from backbone hidden state)
    ggml_tensor *depth_linear_w = nullptr, *depth_linear_b = nullptr; // (hidden → codebooks*depth_dim)

    struct DepthLayerWeights {
        ggml_tensor* operator_norm_w = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* attn_qkv_proj_w = nullptr; // fused Q+K+V
        ggml_tensor* attn_out_proj_w = nullptr;
        ggml_tensor* attn_q_ln_w = nullptr;
        ggml_tensor* attn_k_ln_w = nullptr;
        ggml_tensor *ff_w1 = nullptr, *ff_w2 = nullptr, *ff_w3 = nullptr;
    };
    std::vector<DepthLayerWeights> depth_layers;

    struct DepthCodebook {
        ggml_tensor* embedding_w = nullptr;      // (depth_dim, audio_vocab_size)
        ggml_tensor* embedding_norm_w = nullptr; // (depth_dim,)
        ggml_tensor* to_logits_w = nullptr;      // (depth_dim, audio_vocab_size)
    };
    std::vector<DepthCodebook> depth_codebooks;

    // core_gguf resources
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Vocabulary
    std::vector<std::string> vocab;
};

// ===========================================================================
// Context
// ===========================================================================

struct lfm2_audio_context {
    lfm2_audio_model model;
    int n_threads = 4;
    int verbosity = 1;
    bool use_gpu = false;
    ggml_backend_t backend = nullptr;

    // Compute buffers: large for prefill, small for decode steps
    std::vector<uint8_t> compute_meta; // 2 GB for prefill (T >> 1)
    std::vector<uint8_t> decode_meta;  // 64 MB for decode (T=1)

    // Staged callback (set by run_lfm_staged)
    lfm2_audio_stage_cb lfm_stage_cb = nullptr;
    void* lfm_stage_ud = nullptr;

    // ---- KV cache for attention layers ----
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_k = nullptr; // (head_dim, max_ctx, n_kv_heads, n_attn_layers)
    ggml_tensor* kv_v = nullptr;
    int kv_max_ctx = 0;
    int kv_n_past = 0; // tokens already in cache

    // ---- Conv state cache for conv layers ----
    // Each conv layer caches the last (kernel-1)=2 Bx vectors.
    // Layout: [n_conv_layers][hidden * (kernel-1)] flat CPU float.
    std::vector<std::vector<float>> conv_states;
    bool conv_states_valid = false;

    void reset_kv() {
        kv_n_past = 0;
        if (kv_buf)
            ggml_backend_buffer_clear(kv_buf, 0);
        conv_states_valid = false;
        for (auto& s : conv_states)
            std::fill(s.begin(), s.end(), 0.0f);
    }

    // ---- Detokenizer (codes → PCM, lazy-loaded from companion GGUF) ----
    struct DetokModel {
        bool loaded = false;
        bool tried = false; // avoid repeated load attempts
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        std::map<std::string, ggml_tensor*> tensors;

        // FusedEmbedding: (hidden=512, 8*2048=16384)
        ggml_tensor* emb_w = nullptr;
        // 8 LFM2 layers
        struct Layer {
            bool is_attention = false;
            ggml_tensor* operator_norm_w = nullptr;
            ggml_tensor* ffn_norm_w = nullptr;
            ggml_tensor *ff_w1 = nullptr, *ff_w2 = nullptr, *ff_w3 = nullptr;
            // conv
            ggml_tensor *conv_conv_w = nullptr, *conv_in_proj_w = nullptr, *conv_out_proj_w = nullptr;
            // attn
            ggml_tensor *attn_q_proj_w = nullptr, *attn_k_proj_w = nullptr, *attn_v_proj_w = nullptr;
            ggml_tensor *attn_out_proj_w = nullptr, *attn_q_ln_w = nullptr, *attn_k_ln_w = nullptr;
        };
        std::vector<Layer> layers;
        ggml_tensor* embedding_norm_w = nullptr;
        ggml_tensor* output_w = nullptr; // (512→1282)
        ggml_tensor* output_b = nullptr;
        // hparams
        int hidden = 512;
        int n_layers = 8;
        int n_heads = 16;
        int n_kv_heads = 8;
        int head_dim = 32;
        int sliding_window = 30;
        int output_size = 1282;
        int conv_kernel = 3;
        float rope_theta = 1000000.0f;
        std::string layer_types; // "ccacacacc" etc
    } detok;
    std::string model_path; // to derive detokenizer path
};

// ===========================================================================
// Tensor lookup helpers (DRY: delegates to core_gguf)
// ===========================================================================

static ggml_tensor* R(lfm2_audio_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "lfm2-audio");
}

static ggml_tensor* G(lfm2_audio_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

// ===========================================================================
// FFT (shared via core/fft.h where available, else inline Radix-2 DIT)
// ===========================================================================

static void lfm2_fft_r2c(const float* in, int N, float* out) {
    core_fft::fft_radix2_wrapper(in, N, out);
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool lfm2_audio_load(lfm2_audio_model& model, const char* path, ggml_backend_t backend, int verbosity) {
    // ---- Pass 1: metadata ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "lfm2audio.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "lfm2audio.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "lfm2audio.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "lfm2audio.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "lfm2audio.hop_length", hp.hop_length);
        hp.codebooks = core_gguf::kv_u32(gctx, "lfm2audio.codebooks", hp.codebooks);
        hp.audio_vocab_size = core_gguf::kv_u32(gctx, "lfm2audio.audio_vocab_size", hp.audio_vocab_size);
        hp.enc_n_layers = core_gguf::kv_u32(gctx, "lfm2audio.enc_n_layers", hp.enc_n_layers);
        hp.enc_d_model = core_gguf::kv_u32(gctx, "lfm2audio.enc_d_model", hp.enc_d_model);
        hp.enc_n_heads = core_gguf::kv_u32(gctx, "lfm2audio.enc_n_heads", hp.enc_n_heads);
        hp.enc_conv_kernel = core_gguf::kv_u32(gctx, "lfm2audio.enc_conv_kernel", hp.enc_conv_kernel);
        hp.enc_subsampling_factor =
            core_gguf::kv_u32(gctx, "lfm2audio.enc_subsampling_factor", hp.enc_subsampling_factor);
        hp.enc_subsampling_channels =
            core_gguf::kv_u32(gctx, "lfm2audio.enc_subsampling_channels", hp.enc_subsampling_channels);
        hp.lfm_hidden_size = core_gguf::kv_u32(gctx, "lfm2audio.lfm_hidden_size", hp.lfm_hidden_size);
        hp.lfm_n_layers = core_gguf::kv_u32(gctx, "lfm2audio.lfm_n_layers", hp.lfm_n_layers);
        hp.lfm_n_heads = core_gguf::kv_u32(gctx, "lfm2audio.lfm_n_heads", hp.lfm_n_heads);
        hp.lfm_n_kv_heads = core_gguf::kv_u32(gctx, "lfm2audio.lfm_n_kv_heads", hp.lfm_n_kv_heads);
        hp.lfm_head_dim = core_gguf::kv_u32(gctx, "lfm2audio.lfm_head_dim", hp.lfm_head_dim);
        hp.lfm_ff_dim = core_gguf::kv_u32(gctx, "lfm2audio.lfm_ff_dim", hp.lfm_ff_dim);
        hp.lfm_conv_kernel = core_gguf::kv_u32(gctx, "lfm2audio.lfm_conv_kernel", hp.lfm_conv_kernel);
        hp.lfm_rope_theta = core_gguf::kv_f32(gctx, "lfm2audio.lfm_rope_theta", hp.lfm_rope_theta);
        hp.lfm_layer_types = core_gguf::kv_str(gctx, "lfm2audio.lfm_layer_types", "ccaccaccacacacac");
        hp.depth_n_layers = core_gguf::kv_u32(gctx, "lfm2audio.depth_n_layers", hp.depth_n_layers);
        hp.depth_dim = core_gguf::kv_u32(gctx, "lfm2audio.depth_dim", hp.depth_dim);
        hp.text_vocab_size = core_gguf::kv_u32(gctx, "lfm2audio.text_vocab_size", hp.text_vocab_size);

        model.vocab = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");

        core_gguf::free_metadata(gctx);

        if (verbosity >= 1) {
            fprintf(stderr,
                    "lfm2-audio: enc=%uL×%u, lfm=%uL×%u (%s), "
                    "depth=%uL×%u, codebooks=%u, vocab=%zu\n",
                    hp.enc_n_layers, hp.enc_d_model, hp.lfm_n_layers, hp.lfm_hidden_size, hp.lfm_layer_types.c_str(),
                    hp.depth_n_layers, hp.depth_dim, hp.codebooks, model.vocab.size());
        }
    }

    // ---- Pass 2: weight data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "lfm2-audio", wl))
        return false;
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- Bind encoder tensors (core_conformer pattern) ----
    auto& hp = model.hparams;

    model.pre_enc.conv0_w = R(model, "encoder.pre.conv.0.weight");
    model.pre_enc.conv0_b = R(model, "encoder.pre.conv.0.bias");
    model.pre_enc.conv2_w = R(model, "encoder.pre.conv.2.weight");
    model.pre_enc.conv2_b = R(model, "encoder.pre.conv.2.bias");
    model.pre_enc.conv3_w = R(model, "encoder.pre.conv.3.weight");
    model.pre_enc.conv3_b = R(model, "encoder.pre.conv.3.bias");
    model.pre_enc.conv5_w = R(model, "encoder.pre.conv.5.weight");
    model.pre_enc.conv5_b = R(model, "encoder.pre.conv.5.bias");
    model.pre_enc.conv6_w = R(model, "encoder.pre.conv.6.weight");
    model.pre_enc.conv6_b = R(model, "encoder.pre.conv.6.bias");
    model.pre_enc.out_w = R(model, "encoder.pre.out.weight");
    model.pre_enc.out_b = R(model, "encoder.pre.out.bias");

    model.enc_blocks.resize(hp.enc_n_layers);
    for (uint32_t i = 0; i < hp.enc_n_layers; i++) {
        auto& b = model.enc_blocks[i];
        char buf[128];
        auto T = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suffix);
            return G(model, buf);
        };
        b.norm_ff1_w = T("norm_ff1.weight");
        b.norm_ff1_b = T("norm_ff1.bias");
        b.ff1_l1_w = T("ff1.linear1.weight");
        b.ff1_l1_b = T("ff1.linear1.bias");
        b.ff1_l2_w = T("ff1.linear2.weight");
        b.ff1_l2_b = T("ff1.linear2.bias");
        b.norm_attn_w = T("norm_attn.weight");
        b.norm_attn_b = T("norm_attn.bias");
        b.attn_q_w = T("attn.q.weight");
        b.attn_q_b = T("attn.q.bias");
        b.attn_k_w = T("attn.k.weight");
        b.attn_k_b = T("attn.k.bias");
        b.attn_v_w = T("attn.v.weight");
        b.attn_v_b = T("attn.v.bias");
        b.attn_out_w = T("attn.out.weight");
        b.attn_out_b = T("attn.out.bias");
        b.attn_pos_w = T("attn.pos.weight");
        b.pos_bias_u = T("attn.pos_bias_u");
        b.pos_bias_v = T("attn.pos_bias_v");
        b.norm_conv_w = T("norm_conv.weight");
        b.norm_conv_b = T("norm_conv.bias");
        b.conv_pw1_w = T("conv.pw1.weight");
        b.conv_pw1_b = T("conv.pw1.bias");
        b.conv_dw_w = T("conv.dw.weight");
        b.conv_dw_b = T("conv.dw.bias");
        b.conv_pw2_w = T("conv.pw2.weight");
        b.conv_pw2_b = T("conv.pw2.bias");
        b.norm_ff2_w = T("norm_ff2.weight");
        b.norm_ff2_b = T("norm_ff2.bias");
        b.ff2_l1_w = T("ff2.linear1.weight");
        b.ff2_l1_b = T("ff2.linear1.bias");
        b.ff2_l2_w = T("ff2.linear2.weight");
        b.ff2_l2_b = T("ff2.linear2.bias");
        b.norm_out_w = T("norm_out.weight");
        b.norm_out_b = T("norm_out.bias");
    }

    // ---- Bind adapter tensors ----
    model.adapter_norm_w = R(model, "adapter.norm.weight");
    model.adapter_norm_b = R(model, "adapter.norm.bias");
    model.adapter_lin0_w = R(model, "adapter.linear0.weight");
    model.adapter_lin0_b = R(model, "adapter.linear0.bias");
    model.adapter_lin1_w = R(model, "adapter.linear1.weight");
    model.adapter_lin1_b = R(model, "adapter.linear1.bias");

    // ---- Bind LFM2 backbone tensors ----
    model.lfm_embed_tokens_w = R(model, "lfm.embed_tokens.weight");
    model.lfm_embedding_norm_w = R(model, "lfm.embedding_norm.weight");
    model.lfm_layers.resize(hp.lfm_n_layers);
    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        auto& l = model.lfm_layers[i];
        char buf[128];
        auto T = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "lfm.layers.%u.%s", i, suffix);
            return G(model, buf);
        };
        l.operator_norm_w = T("operator_norm.weight");
        l.ffn_norm_w = T("ffn_norm.weight");
        l.ff_w1 = T("ff.w1.weight");
        l.ff_w2 = T("ff.w2.weight");
        l.ff_w3 = T("ff.w3.weight");

        l.is_attention = (i < hp.lfm_layer_types.size() && hp.lfm_layer_types[i] == 'a');
        if (l.is_attention) {
            l.attn_q_proj_w = T("attn.q_proj.weight");
            l.attn_k_proj_w = T("attn.k_proj.weight");
            l.attn_v_proj_w = T("attn.v_proj.weight");
            l.attn_out_proj_w = T("attn.out_proj.weight");
            l.attn_q_ln_w = T("attn.q_layernorm.weight");
            l.attn_k_ln_w = T("attn.k_layernorm.weight");
        } else {
            l.conv_conv_w = T("conv.conv.weight");
            l.conv_in_proj_w = T("conv.in_proj.weight");
            l.conv_out_proj_w = T("conv.out_proj.weight");
        }
    }

    // ---- Bind audio embedding ----
    model.audio_embd_embedding_w = G(model, "audio_embd.embedding.weight");
    model.audio_embd_embedding_norm_w = G(model, "audio_embd.embedding_norm.weight");
    model.audio_embd_to_logits_w = G(model, "audio_embd.to_logits.weight");

    // ---- Depthformer weights ----
    model.depth_linear_w = G(model, "depth.linear.weight");
    model.depth_linear_b = G(model, "depth.linear.bias");
    model.depth_layers.resize(hp.depth_n_layers);
    for (uint32_t i = 0; i < hp.depth_n_layers; i++) {
        auto& dl = model.depth_layers[i];
        char buf[128];
        auto T = [&](const char* suffix) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "depth.layers.%u.%s", i, suffix);
            return G(model, buf);
        };
        dl.operator_norm_w = T("operator_norm.weight");
        dl.ffn_norm_w = T("ffn_norm.weight");
        dl.attn_qkv_proj_w = T("attn.qkv_proj.weight");
        dl.attn_out_proj_w = T("attn.out_proj.weight");
        dl.attn_q_ln_w = T("attn.q_layernorm.weight");
        dl.attn_k_ln_w = T("attn.k_layernorm.weight");
        dl.ff_w1 = T("ff.w1.weight");
        dl.ff_w2 = T("ff.w2.weight");
        dl.ff_w3 = T("ff.w3.weight");
    }
    model.depth_codebooks.resize(hp.codebooks);
    for (uint32_t c = 0; c < hp.codebooks; c++) {
        auto& cb = model.depth_codebooks[c];
        char buf[128];
        snprintf(buf, sizeof(buf), "depth.codebook.%u.embedding.weight", c);
        cb.embedding_w = G(model, buf);
        snprintf(buf, sizeof(buf), "depth.codebook.%u.embedding_norm.weight", c);
        cb.embedding_norm_w = G(model, buf);
        snprintf(buf, sizeof(buf), "depth.codebook.%u.to_logits.weight", c);
        cb.to_logits_w = G(model, buf);
    }

    // ---- Mel preprocessor (stored by converter from librosa slaney fb) ----
    model.mel_fb = G(model, "preprocessor.fb");
    model.mel_window = G(model, "preprocessor.window");

    return true;
}

// ===========================================================================
// Mel spectrogram (NeMo-style, delegates to core_mel::compute)
// ===========================================================================

static std::vector<float> lfm2_compute_mel_impl(lfm2_audio_context* ctx, const float* samples, int n_samples,
                                                int& T_out) {
    const auto& hp = ctx->model.hparams;
    const auto& model = ctx->model;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!model.mel_fb || !model.mel_window) {
        fprintf(stderr, "lfm2-audio: missing preprocessor.fb / window in GGUF\n");
        return {};
    }

    // Read Hann window from GGUF
    std::vector<float> window((size_t)win);
    ggml_backend_tensor_get(model.mel_window, window.data(), 0, win * sizeof(float));

    // Read mel filterbank from GGUF — stored as (n_mels, n_freqs) by the
    // converter (librosa convention). core_mel::compute expects the same
    // layout: float32[n_mels * n_freqs] row-major.
    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true;
    p.preemph = 0.97f; // NeMo default

    return core_mel::compute(samples, n_samples, window.data(), win, mel_fb.data(), n_freqs, lfm2_fft_r2c, p, T_out);
}

// ===========================================================================
// Public API
// ===========================================================================

lfm2_audio_context_params lfm2_audio_context_default_params(void) {
    lfm2_audio_context_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

lfm2_audio_context* lfm2_audio_init_from_file(const char* path_model, lfm2_audio_context_params params) {
    auto* ctx = new lfm2_audio_context();
    ctx->n_threads = params.n_threads;
    ctx->verbosity = params.verbosity;
    ctx->use_gpu = params.use_gpu;
    ctx->backend = ggml_backend_cpu_init();
    ctx->compute_meta.resize(256ULL * 1024 * 1024); // 2 GB scratch for prefill
    ctx->decode_meta.resize(64ULL * 1024 * 1024);   // 64 MB scratch for T=1 decode

    ctx->model_path = path_model;
    if (!lfm2_audio_load(ctx->model, path_model, ctx->backend, params.verbosity)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Initialize KV cache for attention layers
    {
        auto& hp = ctx->model.hparams;
        // Count attention vs conv layers
        int n_attn = 0, n_conv = 0;
        for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
            if (ctx->model.lfm_layers[i].is_attention)
                n_attn++;
            else
                n_conv++;
        }
        const int max_ctx = 2048; // max sequence length
        struct ggml_init_params gp = {ggml_tensor_overhead() * 2, nullptr, true};
        ctx->kv_ctx = ggml_init(gp);
        if (ctx->kv_ctx) {
            ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, (int)hp.lfm_head_dim, max_ctx,
                                           (int)hp.lfm_n_kv_heads, n_attn);
            ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, GGML_TYPE_F16, (int)hp.lfm_head_dim, max_ctx,
                                           (int)hp.lfm_n_kv_heads, n_attn);
            ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
            if (ctx->kv_buf) {
                ggml_backend_buffer_clear(ctx->kv_buf, 0);
                ctx->kv_max_ctx = max_ctx;
            }
        }
        // Initialize conv state cache
        const int conv_state_len = (int)(hp.lfm_conv_kernel - 1); // 2
        ctx->conv_states.resize(n_conv);
        for (auto& s : ctx->conv_states)
            s.resize((size_t)hp.lfm_hidden_size * conv_state_len, 0.0f);
    }

    return ctx;
}

void lfm2_audio_free(lfm2_audio_context* ctx) {
    if (!ctx)
        return;
    if (ctx->detok.buf)
        ggml_backend_buffer_free(ctx->detok.buf);
    if (ctx->detok.ctx)
        ggml_free(ctx->detok.ctx);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int lfm2_audio_n_mels(lfm2_audio_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 0;
}

int lfm2_audio_sample_rate(lfm2_audio_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 0;
}

int lfm2_audio_test_load(lfm2_audio_context* ctx) {
    if (!ctx)
        return -1;
    auto& hp = ctx->model.hparams;
    fprintf(stderr, "lfm2-audio test_load OK: enc=%uL lfm=%uL(%s) depth=%uL vocab=%zu\n", hp.enc_n_layers,
            hp.lfm_n_layers, hp.lfm_layer_types.c_str(), hp.depth_n_layers, ctx->model.vocab.size());
    return 0;
}

// ===========================================================================
// Stage: mel spectrogram
// ===========================================================================

float* lfm2_audio_compute_mel(lfm2_audio_context* ctx, const float* samples, int n_samples, int* out_T_mel,
                              int* out_n_mels) {
    if (!ctx)
        return nullptr;
    int T_mel = 0;
    auto mel = lfm2_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;

    float* out = (float*)malloc(mel.size() * sizeof(float));
    if (out)
        memcpy(out, mel.data(), mel.size() * sizeof(float));
    if (out_T_mel)
        *out_T_mel = T_mel;
    if (out_n_mels)
        *out_n_mels = (int)ctx->model.hparams.n_mels;
    return out;
}

// ===========================================================================
// Stage: FastConformer encoder (delegates to core_conformer)
// ===========================================================================

float* lfm2_audio_run_encoder(lfm2_audio_context* ctx, const float* mel, int T_mel, int n_mels, int* out_T_enc,
                              int* out_d_model) {
    if (!ctx || !mel)
        return nullptr;
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int d = (int)hp.enc_d_model;

    // Allocate compute context with gallocr (no_alloc)
    const size_t n_tens = 4096;
    ggml_init_params ip = {n_tens * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false), nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return nullptr;

    // Mel input: (n_mels, T_mel)
    ggml_tensor* mel_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel_t, "mel_in");
    ggml_set_input(mel_t);

    // Pre-encode (dw_striding 8×)
    int T_enc = 0;
    ggml_tensor* enc =
        core_conformer::build_pre_encode(ctx0, mel_t, model.pre_enc, (int)hp.enc_subsampling_channels, &T_enc);

    // Sinusoidal rel-pos table
    auto pos_vec = core_conformer::make_pos_enc(d, T_enc);
    ggml_tensor* pos_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T_enc - 1);
    ggml_set_name(pos_t, "pos_enc");
    ggml_set_input(pos_t);

    // Encoder blocks
    core_conformer::BlockParams bp = {d, (int)hp.enc_n_heads, d / (int)hp.enc_n_heads, (int)hp.enc_conv_kernel, 1e-5f};
    for (uint32_t i = 0; i < hp.enc_n_layers; i++)
        enc = core_conformer::build_block(ctx0, enc, pos_t, T_enc, model.enc_blocks[i], bp);

    ggml_tensor* out = ggml_dup(ctx0, enc);
    ggml_set_name(out, "encoder_output");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, out);

    // Allocate via gallocr
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Set inputs
    ggml_backend_tensor_set(mel_t, mel, 0, sizeof(float) * n_mels * T_mel);
    ggml_backend_tensor_set(pos_t, pos_vec.data(), 0, sizeof(float) * d * (2 * T_enc - 1));

    ggml_backend_graph_compute(ctx->backend, gf);

    float* result = (float*)malloc(sizeof(float) * T_enc * d);
    if (result)
        ggml_backend_tensor_get(out, result, 0, sizeof(float) * T_enc * d);

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

// Forward declarations for LFM2 backbone helpers (defined below)
static ggml_tensor* lfm2_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps);
static ggml_tensor* lfm2_swiglu_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* w2,
                                    ggml_tensor* w3);
static ggml_tensor* lfm2_short_conv(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w, int hidden, int T);
static ggml_tensor* lfm2_gqa_attention(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                       ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                       int T);
static ggml_tensor* lfm2_build_layer(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                     ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                     int T, float norm_eps);

// ===========================================================================
// Transcribe: full ASR pipeline
//   1. mel → encoder → adapter → audio embeddings
//   2. Embed fixed prompt tokens → text embeddings
//   3. Assemble: prefix_text + audio + suffix_text
//   4. LFM backbone prefill
//   5. Auto-regressive text generation (greedy, no KV cache for now)
// ===========================================================================

// Fixed token IDs for the ASR chat template. These are pre-tokenized
// using the LFM2.5 BPE tokenizer and hardcoded to avoid needing a full
// BPE encoder at runtime. The decode side just uses the vocab lookup.
static const int32_t kTokenStartOfText = 1;
static const int32_t kTokenImStart = 6;
static const int32_t kTokenImEnd = 7;
static const int32_t kTokenNewline = 708;
static const int32_t kTokenAudioStart = 128;
static const int32_t kTokenTextEnd = 130;

// <|startoftext|><|im_start|>system\nPerform ASR in japanese.<|im_end|>\n<|im_start|>user\n
static const std::vector<int32_t> kPrefixJa = {1,     6,    24131, 708, 8173, 1199, 11866, 559, 797,
                                               41035, 3391, 523,   7,   708,  6,    6423,  708};
// <|startoftext|><|im_start|>system\nPerform ASR in english.<|im_end|>\n<|im_start|>user\n
static const std::vector<int32_t> kPrefixEn = {1,   6,     24131, 708, 8173, 1199, 11866, 559,
                                               797, 48103, 523,   7,   708,  6,    6423,  708};
// <|im_end|>\n
static const std::vector<int32_t> kSuffix = {7, 708};

// GPT-2 byte decoder: reverse of bytes_to_unicode()
// Maps unicode codepoints back to raw bytes.
static const std::vector<int>& byte_decoder() {
    static std::vector<int> bd(512, -1);
    static bool init = false;
    if (init)
        return bd;
    const auto& be = core_bpe::byte_encoder();
    for (int b = 0; b < 256; b++)
        bd[be[b]] = b;
    init = true;
    return bd;
}

// Decode a BPE token to raw UTF-8 bytes using the GPT-2 byte_decoder
static std::string decode_token(const lfm2_audio_model& model, int32_t id) {
    if (id < 0 || id >= (int32_t)model.vocab.size())
        return "";
    const std::string& piece = model.vocab[id];
    // Skip special tokens (they start with < and end with >)
    if (!piece.empty() && piece[0] == '<' && piece.back() == '>')
        return "";
    const auto& bd = byte_decoder();
    std::string result;
    // Iterate over UTF-8 codepoints in the piece
    size_t i = 0;
    while (i < piece.size()) {
        uint32_t cp = 0;
        int len = 1;
        uint8_t c = (uint8_t)piece[i];
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        }
        for (int j = 1; j < len && i + j < piece.size(); j++)
            cp = (cp << 6) | ((uint8_t)piece[i + j] & 0x3F);
        i += len;
        // Map codepoint back to byte via the GPT-2 byte_decoder
        if (cp < (uint32_t)bd.size() && bd[cp] >= 0) {
            result += (char)(uint8_t)bd[cp];
        } else {
            // Codepoint not in the byte_decoder — pass through as UTF-8
            if (cp < 0x80) {
                result += (char)cp;
            } else if (cp < 0x800) {
                result += (char)(0xC0 | (cp >> 6));
                result += (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                result += (char)(0xE0 | (cp >> 12));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
            }
        }
    }
    return result;
}

// Build a ggml graph that runs one LFM backbone forward pass on the full
// input sequence and returns logits over the text vocabulary at position T-1.
// This is the non-cached prefill+decode step.
static std::vector<float> lfm2_run_backbone_logits(lfm2_audio_context* ctx, const float* embeddings, int T,
                                                   int hidden) {
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int n_heads = (int)hp.lfm_n_heads;
    const int n_kv = (int)hp.lfm_n_kv_heads;
    const int hd = (int)hp.lfm_head_dim;
    const float norm_eps = 1e-5f;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T);
    memcpy(x->data, embeddings, sizeof(float) * T * hidden);

    // Run all LFM2 layers
    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        x = lfm2_build_layer(ctx0, x, model.lfm_layers[i], nullptr, hidden, n_heads, n_kv, hd, T, norm_eps);
    }

    // Final RMSNorm
    x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, norm_eps);

    // Extract last position: (hidden, T) → view column T-1 → (hidden,)
    ggml_tensor* last = ggml_view_1d(ctx0, x, hidden, (int64_t)(T - 1) * hidden * sizeof(float));

    // Text logits: embed_tokens^T @ last_hidden → (vocab_size,)
    ggml_tensor* logits = ggml_mul_mat(ctx0, model.lfm_embed_tokens_w, last);

    ggml_tensor* out = ggml_dup(ctx0, logits);
    ggml_set_name(out, "logits");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

    int vocab_size = (int)out->ne[0];
    std::vector<float> result(vocab_size);
    memcpy(result.data(), out->data, sizeof(float) * vocab_size);

    ggml_free(ctx0);
    return result;
}

// Embed a sequence of token IDs using lfm.embed_tokens
static std::vector<float> lfm2_embed_tokens(lfm2_audio_context* ctx, const std::vector<int32_t>& tokens) {
    auto& model = ctx->model;
    const int hidden = (int)model.hparams.lfm_hidden_size;
    const int n_tok = (int)tokens.size();

    // embed_tokens weight: (hidden, vocab_size) in ggml → row i = embedding for token i
    std::vector<float> emb(n_tok * hidden);
    for (int i = 0; i < n_tok; i++) {
        int32_t id = tokens[i];
        // Read row `id` from the embedding matrix
        ggml_backend_tensor_get(model.lfm_embed_tokens_w, emb.data() + i * hidden,
                                (size_t)id * hidden * ggml_type_size(model.lfm_embed_tokens_w->type) /
                                    ggml_blck_size(model.lfm_embed_tokens_w->type),
                                hidden * sizeof(float));
    }
    return emb;
}

char* lfm2_audio_transcribe(lfm2_audio_context* ctx, const float* samples, int n_samples, const char* prompt,
                            int max_tokens) {
    if (!ctx || !samples)
        return nullptr;
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int hidden = (int)hp.lfm_hidden_size;

    if (max_tokens <= 0)
        max_tokens = 512;

    // Step 1: mel → encoder → adapter
    int T_mel = 0;
    auto mel = lfm2_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;

    int T_enc = 0, d_model = 0;
    float* enc = lfm2_audio_run_encoder(ctx, mel.data(), T_mel, (int)hp.n_mels, &T_enc, &d_model);
    if (!enc)
        return nullptr;

    int adapter_hidden = 0;
    float* adapted = lfm2_audio_run_adapter(ctx, enc, T_enc, d_model, &adapter_hidden);
    free(enc);
    if (!adapted)
        return nullptr;

    // Step 2: Embed prefix and suffix text tokens
    // Select prompt based on user hint or auto-detect from GGUF metadata.
    // The prompt parameter can be "ja", "japanese", "en", "english", or a
    // full system prompt string (not yet supported for arbitrary prompts).
    const auto* prefix_ids_ptr = &kPrefixJa; // default: Japanese
    if (prompt) {
        std::string p(prompt);
        if (p == "en" || p == "english" || p.find("english") != std::string::npos)
            prefix_ids_ptr = &kPrefixEn;
    }
    const auto& prefix_ids = *prefix_ids_ptr;

    // For the embed_tokens lookup, we need F32 data. The weight is likely F16.
    // Use a simple CPU dequant via ggml_backend_tensor_get_f32 pattern.
    // Actually, ggml_backend_tensor_get gives raw bytes; for F16 we need to convert.
    // Simpler: build a tiny ggml graph that does the embedding lookup.
    auto embed_text = [&](const std::vector<int32_t>& ids) -> std::vector<float> {
        const int n = (int)ids.size();
        const size_t mem = 16 * 1024 * 1024;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context* c = ggml_init(ip);
        if (!c)
            return {};

        ggml_tensor* id_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        memcpy(id_t->data, ids.data(), n * sizeof(int32_t));

        ggml_tensor* emb = ggml_get_rows(c, model.lfm_embed_tokens_w, id_t);
        ggml_tensor* out = ggml_dup(c, emb);

        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(c, gf, 1);

        std::vector<float> result(n * hidden);
        memcpy(result.data(), out->data, sizeof(float) * n * hidden);
        ggml_free(c);
        return result;
    };

    auto prefix_emb = embed_text(prefix_ids);
    auto suffix_emb = embed_text(kSuffix);

    // Step 3: Assemble full input sequence
    // Layout: [prefix_text_emb | audio_adapter_emb | suffix_text_emb]
    const int T_prefix = (int)prefix_ids.size();
    const int T_audio = T_enc;
    const int T_suffix = (int)kSuffix.size();
    const int T_context = T_prefix + T_audio + T_suffix;

    std::vector<float> context_emb(T_context * hidden);
    memcpy(context_emb.data(), prefix_emb.data(), sizeof(float) * T_prefix * hidden);
    memcpy(context_emb.data() + T_prefix * hidden, adapted, sizeof(float) * T_audio * hidden);
    memcpy(context_emb.data() + (T_prefix + T_audio) * hidden, suffix_emb.data(), sizeof(float) * T_suffix * hidden);
    free(adapted);

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: prefill T=%d (prefix=%d audio=%d suffix=%d)\n", T_context, T_prefix, T_audio,
                T_suffix);

    // Step 4: Auto-regressive greedy decode WITH KV cache.
    //
    // Phase A (prefill): run the full context through the backbone once,
    //   populating the KV cache for all attention layers and conv state
    //   caches for all conv layers. Extract logits at the last position.
    //
    // Phase B (decode): for each new token, run a single token through
    //   the backbone using cached K/V + conv states. Only 1 forward pass
    //   per token instead of full-sequence recompute.
    //
    // The core_attn::kv_self_attn helper handles the KV cache write/read.
    // Conv state is managed manually (small: 2048*2 floats per layer).

    ctx->reset_kv();
    std::string transcript;
    std::vector<int32_t> generated_ids;

    // Helper: build graph, run backbone on T_in tokens starting at n_past,
    // return logits from the last position.
    auto run_step = [&](const float* embeddings, int T_in) -> std::vector<float> {
        const int n_past = ctx->kv_n_past;
        const float norm_eps = 1e-5f;

        // Use gallocr: build graph with no_alloc, then allocate exactly what's needed.
        // This avoids the 2 GB fixed buffer and reduces memory pressure dramatically.
        const size_t n_tensors_est = 2048;
        ggml_init_params ip = {n_tensors_est * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false),
                               nullptr, true};
        ggml_context* ctx0 = ggml_init(ip);
        if (!ctx0)
            return {};

        ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T_in);
        ggml_set_name(x, "inp_emb");
        ggml_set_input(x);

        // Positions
        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_in);
        ggml_set_name(positions, "inp_pos");
        ggml_set_input(positions);

        // Causal mask for prefill (T_in > 1); nullptr for single-token decode
        ggml_tensor* causal_mask = nullptr;
        if (T_in > 1) {
            const int Lk = n_past + T_in;
            causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, Lk, T_in);
            ggml_set_name(causal_mask, "inp_mask");
            ggml_set_input(causal_mask);
        }

        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

        int attn_idx = 0, conv_idx = 0;
        const auto& hp = model.hparams;
        const int n_heads = (int)hp.lfm_n_heads;
        const int n_kv = (int)hp.lfm_n_kv_heads;
        const int hd = (int)hp.lfm_head_dim;

        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            auto& w = model.lfm_layers[il];
            ggml_tensor* residual = x;

            // RMSNorm
            ggml_tensor* h = lfm2_rms_norm(ctx0, x, w.operator_norm_w, norm_eps);

            if (w.is_attention) {
                // --- GQA attention with KV cache ---
                core_attn::KvSelfAttnParams kvp = {};
                kvp.head_dim = hd;
                kvp.n_heads = n_heads;
                kvp.n_kv_heads = n_kv;
                kvp.n_kv_grp = n_heads / n_kv;
                kvp.n_ctx_orig = 0;
                kvp.rope_type = GGML_ROPE_TYPE_NEOX;
                kvp.rope_theta = hp.lfm_rope_theta;
                kvp.rope_beta_fast = 0.0f;
                kvp.rope_beta_slow = 0.0f;
                kvp.attn_scale = 1.0f / sqrtf((float)hd);
                kvp.qk_norm_eps = norm_eps;
                kvp.gqa_mode = core_attn::GQA_NATIVE;

                h = core_attn::kv_self_attn(ctx0, gf, h, w.attn_q_proj_w, w.attn_k_proj_w, w.attn_v_proj_w,
                                            w.attn_out_proj_w, w.attn_q_ln_w, w.attn_k_ln_w, positions, causal_mask,
                                            ctx->kv_k, ctx->kv_v, attn_idx, n_past, kvp);
                attn_idx++;
            } else {
                // --- ShortConv with conv state cache ---
                const int K = (int)hp.lfm_conv_kernel; // 3

                if (T_in > 1) {
                    // PREFILL: run validated conv + snapshot last K-1 Bx columns
                    // Compute Bx in a parallel branch (doesn't affect the main conv path)
                    ggml_tensor* bcx_snap = ggml_mul_mat(ctx0, w.conv_in_proj_w, h);
                    ggml_tensor* B_snap = ggml_view_2d(ctx0, bcx_snap, hidden, T_in, bcx_snap->nb[1], 0);
                    ggml_tensor* x_snap =
                        ggml_view_2d(ctx0, bcx_snap, hidden, T_in, bcx_snap->nb[1], 2 * hidden * sizeof(float));
                    ggml_tensor* Bx_snap = ggml_mul(ctx0, ggml_cont(ctx0, B_snap), ggml_cont(ctx0, x_snap));
                    // Take last K-1 columns
                    int tail_start = T_in - (K - 1);
                    if (tail_start < 0)
                        tail_start = 0;
                    int tail_len = T_in - tail_start;
                    ggml_tensor* bx_tail = ggml_view_2d(ctx0, Bx_snap, hidden, tail_len, hidden * sizeof(float),
                                                        (int64_t)tail_start * hidden * sizeof(float));
                    ggml_tensor* snap = ggml_dup(ctx0, bx_tail);
                    char sname[32];
                    snprintf(sname, sizeof(sname), "cs_%d", conv_idx);
                    ggml_set_name(snap, sname);
                    ggml_build_forward_expand(gf, snap);

                    // Main path: validated conv function
                    h = lfm2_short_conv(ctx0, h, w, hidden, T_in);
                } else {
                    // DECODE (T=1): feed [cached_state | h] through the full
                    // ShortConv, treating it as T=K input. The conv's causal
                    // padding produces K outputs; we take the LAST one.
                    //
                    // This reuses the validated lfm2_short_conv path exactly,
                    // so the conv math is guaranteed correct.

                    // 1. in_proj the full K-token input
                    // Build (hidden, K) input: [cached_h_col0, cached_h_col1, h_new]
                    // BUT we don't have the cached pre-in_proj h — we have cached Bx.
                    // The ShortConv computes: BCx=in_proj(h), B*x=Bx, conv(Bx), C*conv, out_proj.
                    // We need to feed the cached Bx directly into the conv.
                    //
                    // So: compute in_proj for the new token only, get Bx_new.
                    // Then assemble [cached_Bx | Bx_new], run conv, get output.

                    ggml_tensor* bcx = ggml_mul_mat(ctx0, w.conv_in_proj_w, h);
                    ggml_tensor* B_part = ggml_view_1d(ctx0, bcx, hidden, 0);
                    ggml_tensor* C_part = ggml_view_1d(ctx0, bcx, hidden, hidden * sizeof(float));
                    ggml_tensor* x_inner = ggml_view_1d(ctx0, bcx, hidden, 2 * hidden * sizeof(float));
                    ggml_tensor* Bx_new = ggml_mul(ctx0, ggml_cont(ctx0, B_part), ggml_cont(ctx0, x_inner));

                    // 2. Assemble full Bx sequence: [cached | new] = (hidden, K)
                    ggml_tensor* cached = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, K - 1);
                    {
                        char cn[32];
                        snprintf(cn, sizeof(cn), "conv_cache_%d", conv_idx);
                        ggml_set_name(cached, cn);
                        ggml_set_input(cached);
                    }
                    ggml_tensor* Bx_col = ggml_reshape_2d(ctx0, Bx_new, hidden, 1);
                    ggml_tensor* Bx_full = ggml_concat(ctx0, cached, Bx_col, 1); // (hidden, K)

                    // 3. Run depthwise conv on the K-length Bx sequence
                    // Same as lfm2_short_conv's conv section but with T=K
                    ggml_tensor* conv_w_f32 = ggml_cast(ctx0, w.conv_conv_w, GGML_TYPE_F32);
                    ggml_tensor* conv_w_4d = ggml_reshape_4d(ctx0, conv_w_f32, K, 1, 1, hidden);
                    ggml_tensor* Bx_t = ggml_cont(ctx0, ggml_transpose(ctx0, Bx_full)); // (K, hidden)
                    ggml_tensor* Bx_4d = ggml_reshape_4d(ctx0, Bx_t, K, 1, hidden, 1);
                    ggml_tensor* conv_raw = ggml_conv_2d_dw_direct(ctx0, conv_w_4d, Bx_4d, 1, 1, K - 1, 0, 1, 1);
                    // conv_raw shape: (K + K-1, 1, hidden, 1) = (2K-1, 1, hidden, 1)
                    // Permute to (hidden, 2K-1)
                    conv_raw = ggml_cont(ctx0, ggml_permute(ctx0, conv_raw, 1, 2, 0, 3));
                    int T_out = (int)conv_raw->ne[1];
                    conv_raw = ggml_reshape_2d(ctx0, conv_raw, hidden, T_out);
                    // Take the LAST column (position K-1 = the one corresponding
                    // to the new token with full causal context)
                    ggml_tensor* conv_out = ggml_view_2d(ctx0, conv_raw, hidden, 1, hidden * sizeof(float),
                                                         (int64_t)(K - 1) * hidden * sizeof(float));
                    conv_out = ggml_cont(ctx0, conv_out);

                    // 4. y = C * conv_out, then out_proj
                    ggml_tensor* C_col = ggml_reshape_2d(ctx0, ggml_cont(ctx0, C_part), hidden, 1);
                    ggml_tensor* y = ggml_mul(ctx0, C_col, conv_out);
                    h = ggml_mul_mat(ctx0, w.conv_out_proj_w, y);

                    // 5. Snapshot Bx_new for state update
                    ggml_tensor* snap = ggml_dup(ctx0, Bx_new);
                    char sname[32];
                    snprintf(sname, sizeof(sname), "cs_%d", conv_idx);
                    ggml_set_name(snap, sname);
                    ggml_build_forward_expand(gf, snap);
                }
                conv_idx++;
            }

            x = ggml_add(ctx0, residual, h);

            // FFN
            residual = x;
            h = lfm2_rms_norm(ctx0, x, w.ffn_norm_w, norm_eps);
            h = lfm2_swiglu_ffn(ctx0, h, w.ff_w1, w.ff_w2, w.ff_w3);
            x = ggml_add(ctx0, residual, h);
        }

        // Final RMSNorm
        x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, norm_eps);

        // Logits at last position
        ggml_tensor* last = ggml_view_1d(ctx0, x, hidden, (int64_t)(T_in - 1) * hidden * sizeof(float));
        ggml_tensor* logits = ggml_mul_mat(ctx0, model.lfm_embed_tokens_w, last);
        ggml_tensor* out = ggml_dup(ctx0, logits);
        ggml_set_name(out, "logits");

        ggml_build_forward_expand(gf, out);

        // Allocate graph via gallocr — only allocates what's needed
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            ggml_gallocr_free(galloc);
            ggml_free(ctx0);
            return {};
        }

        // Set inputs via backend API (since no_alloc=true, ->data is on the backend)
        ggml_backend_tensor_set(x, embeddings, 0, sizeof(float) * T_in * hidden);
        {
            std::vector<int32_t> pos_data(T_in);
            for (int i = 0; i < T_in; i++)
                pos_data[i] = n_past + i;
            ggml_backend_tensor_set(positions, pos_data.data(), 0, T_in * sizeof(int32_t));
        }
        if (causal_mask) {
            const int Lk = n_past + T_in;
            std::vector<ggml_fp16_t> mask_data(Lk * T_in);
            ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), neginf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < T_in; q++)
                for (int k = 0; k < Lk; k++)
                    mask_data[q * Lk + k] = (k <= n_past + q) ? zero : neginf;
            ggml_backend_tensor_set(causal_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }
        // Set conv state inputs
        if (T_in == 1) {
            const int K = (int)model.hparams.lfm_conv_kernel;
            int ci = 0;
            for (uint32_t il = 0; il < model.hparams.lfm_n_layers; il++) {
                if (model.lfm_layers[il].is_attention)
                    continue;
                char cn[32];
                snprintf(cn, sizeof(cn), "conv_cache_%d", ci);
                ggml_tensor* ct = ggml_graph_get_tensor(gf, cn);
                if (ct)
                    ggml_backend_tensor_set(ct, ctx->conv_states[ci].data(), 0, sizeof(float) * hidden * (K - 1));
                ci++;
            }
        }

        ggml_backend_graph_compute(ctx->backend, gf);

        int vocab_size = (int)out->ne[0];
        std::vector<float> result(vocab_size);
        ggml_backend_tensor_get(out, result.data(), 0, sizeof(float) * vocab_size);

        // Update conv state caches from graph snapshots
        {
            const int K = (int)hp.lfm_conv_kernel;
            int ci = 0;
            for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
                if (model.lfm_layers[il].is_attention)
                    continue;
                char sname[32];
                snprintf(sname, sizeof(sname), "cs_%d", ci);
                ggml_tensor* snap = ggml_graph_get_tensor(gf, sname);
                if (snap) {
                    auto& state = ctx->conv_states[ci];
                    if (T_in > 1) {
                        int snap_cols = (int)snap->ne[1];
                        std::fill(state.begin(), state.end(), 0.0f);
                        int offset = (K - 1) - snap_cols;
                        if (offset < 0)
                            offset = 0;
                        ggml_backend_tensor_get(snap, state.data() + offset * hidden, 0,
                                                sizeof(float) * snap_cols * hidden);
                    } else {
                        memmove(state.data(), state.data() + hidden, sizeof(float) * hidden * ((K - 1) - 1));
                        ggml_backend_tensor_get(snap, state.data() + hidden * ((K - 1) - 1), 0, sizeof(float) * hidden);
                    }
                }
                ci++;
            }
        }

        ctx->kv_n_past += T_in;
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return result;
    };

    // Phase A: Prefill — run full context through backbone
    auto logits = run_step(context_emb.data(), T_context);
    if (!logits.empty() && ctx->verbosity >= 1) {
        int top = 0;
        for (int i = 1; i < (int)logits.size(); i++)
            if (logits[i] > logits[top])
                top = i;
        fprintf(stderr, "lfm2-audio: prefill top token=%d logit=%.3f\n", top, logits[top]);
    }
    if (logits.empty()) {
        (void)kTokenStartOfText;
        (void)kTokenImStart;
        (void)kTokenNewline;
        (void)kTokenAudioStart;
        (void)kTokenTextEnd;
        if (ctx->verbosity >= 1)
            fprintf(stderr, "lfm2-audio: prefill failed\n");
        return nullptr;
    }

    for (int step = 0; step < max_tokens; step++) {
        // Greedy argmax
        int best_id = 0;
        float best_val = logits[0];
        for (int i = 1; i < (int)logits.size(); i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }

        // Stop on <|im_end|> or <|endoftext|>
        if (best_id == kTokenImEnd || best_id == 2)
            break;

        generated_ids.push_back(best_id);

        // Decode token to text
        std::string piece = decode_token(model, best_id);
        transcript += piece;

        if (ctx->verbosity >= 2)
            fprintf(stderr, "  [%d] token=%d piece=%s\n", step, best_id, piece.c_str());

        // Phase B: Decode — embed the new token and run single-token step
        auto new_emb = embed_text({best_id});
        if (new_emb.empty())
            break;
        logits = run_step(new_emb.data(), 1);
        if (logits.empty())
            break;
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: generated %zu tokens\n", generated_ids.size());

    // Return malloc'd copy
    char* result = (char*)malloc(transcript.size() + 1);
    if (result) {
        memcpy(result, transcript.c_str(), transcript.size());
        result[transcript.size()] = '\0';
    }
    return result;
}

// ===========================================================================
// Stage: audio adapter MLP
//   LayerNorm(d_model) → Linear(d_model→hidden) → GELU → Linear(hidden→hidden)
// ===========================================================================

float* lfm2_audio_run_adapter(lfm2_audio_context* ctx, const float* encoder_out, int T_enc, int d_model,
                              int* out_hidden_size) {
    if (!ctx || !encoder_out)
        return nullptr;
    auto& model = ctx->model;
    const int hidden = (int)model.hparams.lfm_hidden_size;

    ggml_init_params ip = {256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return nullptr;

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, T_enc);
    ggml_set_name(x, "adapter_in");
    ggml_set_input(x);

    // LayerNorm
    x = ggml_norm(ctx0, x, 1e-5f);
    x = ggml_add(ctx0, ggml_mul(ctx0, x, model.adapter_norm_w), model.adapter_norm_b);

    // Linear0: (d_model → hidden) + GELU
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, model.adapter_lin0_w, x), model.adapter_lin0_b);
    x = ggml_gelu(ctx0, x);

    // Linear1: (hidden → hidden)
    x = ggml_add(ctx0, ggml_mul_mat(ctx0, model.adapter_lin1_w, x), model.adapter_lin1_b);

    ggml_tensor* out = ggml_dup(ctx0, x);
    ggml_set_name(out, "adapter_output");

    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx0);
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "adapter_in"), encoder_out, 0, sizeof(float) * T_enc * d_model);
    ggml_backend_graph_compute(ctx->backend, gf);

    float* result = (float*)malloc(sizeof(float) * T_enc * hidden);
    if (result)
        ggml_backend_tensor_get(out, result, 0, sizeof(float) * T_enc * hidden);

    if (out_hidden_size)
        *out_hidden_size = hidden;
    ggml_gallocr_free(galloc);
    ggml_free(ctx0);
    return result;
}

// ===========================================================================
// LFM2 backbone graph builder helpers
// ===========================================================================

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
static ggml_tensor* lfm2_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// SwiGLU FFN: w2(silu(w1(x)) * w3(x))
static ggml_tensor* lfm2_swiglu_ffn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w1, ggml_tensor* w2,
                                    ggml_tensor* w3) {
    ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx, w1, x));
    ggml_tensor* up = ggml_mul_mat(ctx, w3, x);
    return ggml_mul_mat(ctx, w2, ggml_mul(ctx, gate, up));
}

// LFM2 ShortConv layer (non-cached prefill path):
//   BCx = in_proj(x)  → split into B, C, x_inner (each hidden-sized)
//   Bx = B * x_inner
//   conv_out = causal_conv1d(Bx, kernel=3)
//   y = C * conv_out
//   out = out_proj(y)
static ggml_tensor* lfm2_short_conv(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w, int hidden, int T) {
    // x: (hidden, T) in ggml layout (ne[0]=hidden, ne[1]=T)

    // in_proj: (hidden → 3*hidden)
    ggml_tensor* bcx = ggml_mul_mat(ctx, w.conv_in_proj_w, x); // (3*hidden, T)

    // Split into B, C, x_inner along dimension 0
    ggml_tensor* B_part = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], 0);
    ggml_tensor* C_part = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], hidden * sizeof(float));
    ggml_tensor* x_inner = ggml_view_2d(ctx, bcx, hidden, T, bcx->nb[1], 2 * hidden * sizeof(float));

    // Bx = B * x_inner (element-wise)
    ggml_tensor* Bx = ggml_mul(ctx, ggml_cont(ctx, B_part), ggml_cont(ctx, x_inner));

    // Causal conv1d with kernel=3: pad left by 2, conv, take first T frames
    // Bx is (hidden, T). For depthwise conv1d:
    //   - transpose to (T, hidden) = (T, 1, hidden, 1) for conv_2d_dw_direct
    //   - kernel: (K, 1, 1, hidden) where K=3
    //   - causal: pad_left = K-1 = 2
    const int K = 3; // conv_L_cache = 3 = kernel size
    ggml_tensor* conv_w = ggml_cast(ctx, w.conv_conv_w, GGML_TYPE_F32);
    // conv_w stored as (hidden, 1, K) in GGUF. Reshape to (K, 1, 1, hidden)
    ggml_tensor* conv_w_4d = ggml_reshape_4d(ctx, conv_w, K, 1, 1, hidden);

    // Bx: (hidden, T) → transpose → (T, hidden) → reshape to (T, 1, hidden, 1)
    ggml_tensor* Bx_t = ggml_cont(ctx, ggml_transpose(ctx, Bx));
    ggml_tensor* Bx_4d = ggml_reshape_4d(ctx, Bx_t, T, 1, hidden, 1);

    // Depthwise conv with causal padding (pad_left=K-1=2, pad_right=0)
    ggml_tensor* conv_out = ggml_conv_2d_dw_direct(ctx, conv_w_4d, Bx_4d, 1, 1, K - 1, 0, 1, 1);
    // conv_out: (T+K-1, 1, hidden, 1) → take first T frames
    // Actually conv_2d_dw_direct with pad_w=K-1 outputs (T+K-1, ...).
    // We need only the first T frames (causal: no future info).
    // Permute to (hidden, T) layout: permute(1,2,0,3) then reshape
    conv_out = ggml_cont(ctx, ggml_permute(ctx, conv_out, 1, 2, 0, 3));
    int T_conv = (int)conv_out->ne[1]; // may be T or T+K-1
    if (T_conv > T) {
        // Slice to first T frames
        conv_out =
            ggml_view_2d(ctx, ggml_reshape_2d(ctx, conv_out, hidden, T_conv), hidden, T, hidden * sizeof(float), 0);
    } else {
        conv_out = ggml_reshape_2d(ctx, conv_out, hidden, T);
    }

    // y = C * conv_out (element-wise)
    ggml_tensor* y = ggml_mul(ctx, ggml_cont(ctx, C_part), ggml_cont(ctx, conv_out));

    // out_proj: (hidden → hidden)
    return ggml_mul_mat(ctx, w.conv_out_proj_w, y);
}

// LFM2 GQA attention layer (non-cached prefill path):
//   Q = q_layernorm(q_proj(x))
//   K = k_layernorm(k_proj(x))
//   V = v_proj(x)
//   Apply RoPE to Q, K
//   Causal self-attention with GQA
//   out = out_proj(attn_output)
static ggml_tensor* lfm2_gqa_attention(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                       ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                       int T) {
    // x: (hidden, T)

    // Q, K, V projections
    ggml_tensor* Q = ggml_mul_mat(ctx, w.attn_q_proj_w, x); // (hidden, T) = (n_heads*hd, T)
    ggml_tensor* K = ggml_mul_mat(ctx, w.attn_k_proj_w, x); // (n_kv*hd, T)
    ggml_tensor* V = ggml_mul_mat(ctx, w.attn_v_proj_w, x); // (n_kv*hd, T)

    // Reshape to (head_dim, n_heads, T) for Q, (head_dim, n_kv, T) for K/V
    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, T);
    K = ggml_reshape_3d(ctx, K, head_dim, n_kv_heads, T);
    V = ggml_reshape_3d(ctx, V, head_dim, n_kv_heads, T);

    // Per-head QK layernorm (RMSNorm on head_dim axis = ne[0])
    Q = ggml_rms_norm(ctx, Q, 1e-5f);
    Q = ggml_mul(ctx, Q, w.attn_q_ln_w); // broadcasts (head_dim,) over (head_dim, n_heads, T)
    K = ggml_rms_norm(ctx, K, 1e-5f);
    K = ggml_mul(ctx, K, w.attn_k_ln_w);

    // Create positions tensor [0, 1, 2, ..., T-1]
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    {
        int32_t* pos = (int32_t*)positions->data;
        for (int i = 0; i < T; i++)
            pos[i] = i;
    }

    // RoPE — LFM2 uses the standard HF rotate_half pattern.
    // ggml GGML_ROPE_TYPE_NEOX matches: split into halves, rotate, recombine.
    Q = ggml_rope_ext(ctx, Q, positions, /*freq_override=*/nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, positions, /*freq_override=*/nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f,
                      0.0f, 1.0f, 0.0f, 0.0f);

    // Permute for flash_attn_ext: Q (head_dim, T, n_heads), K (head_dim, T, n_kv), V same
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // (hd, n_heads, T) → (hd, T, n_heads)
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Causal self-attention with explicit causal mask
    const float scale = 1.0f / sqrtf((float)head_dim);
    // Build causal mask: (T, T) F16 with 0 for allowed and -inf for forbidden
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    {
        ggml_fp16_t* m = (ggml_fp16_t*)mask->data;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                m[i * T + j] = (j <= i) ? zero : neginf;
    }
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    // attn: (head_dim, T, n_heads) → reshape to (hidden, T)
    attn = ggml_reshape_2d(ctx, attn, hidden, T);

    // Output projection
    return ggml_mul_mat(ctx, w.attn_out_proj_w, attn);
}

// Build one LFM2 decoder layer: RMSNorm → operator → residual → RMSNorm → SwiGLU → residual
static ggml_tensor* lfm2_build_layer(ggml_context* ctx, ggml_tensor* x, const lfm2_layer_weights& w,
                                     ggml_tensor* rope_freqs, int hidden, int n_heads, int n_kv_heads, int head_dim,
                                     int T, float norm_eps) {
    ggml_tensor* residual = x;

    // RMSNorm → operator
    ggml_tensor* h = lfm2_rms_norm(ctx, x, w.operator_norm_w, norm_eps);

    if (w.is_attention) {
        h = lfm2_gqa_attention(ctx, h, w, rope_freqs, hidden, n_heads, n_kv_heads, head_dim, T);
    } else {
        h = lfm2_short_conv(ctx, h, w, hidden, T);
    }

    // Residual
    x = ggml_add(ctx, residual, h);

    // RMSNorm → SwiGLU FFN → residual
    residual = x;
    h = lfm2_rms_norm(ctx, x, w.ffn_norm_w, norm_eps);
    h = lfm2_swiglu_ffn(ctx, h, w.ff_w1, w.ff_w2, w.ff_w3);
    x = ggml_add(ctx, residual, h);

    return x;
}

// ===========================================================================
// Stage: full prefill (encoder → adapter → assemble → LFM2 backbone)
// ===========================================================================

// Run the full LFM2 prefill pipeline on raw audio. This mirrors what
// the Python _prefill + lfm.forward does:
//   1. mel → encoder → adapter → audio embeddings
//   2. Tokenize system prompt + chat template → text embeddings
//   3. Assemble: text_emb[..] + audio_emb[..] + text_emb[..] (by modality_flag)
//   4. Run LFM2 backbone on the assembled sequence
//   5. Apply embedding_norm
//
// For now, the simplified version just runs encoder → adapter → LFM on audio
// embeddings alone (no text tokens). This is enough for diff validation of
// the adapter and backbone layers. Full chat-template assembly comes in
// the transcribe implementation.
float* lfm2_audio_run_lfm(lfm2_audio_context* ctx, const float* samples, int n_samples, int* out_T,
                          int* out_hidden_size) {
    if (!ctx || !samples)
        return nullptr;
    if (ctx->verbosity >= 2)
        fprintf(stderr, "lfm2-audio: run_lfm n_samples=%d\n", n_samples);
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int hidden = (int)hp.lfm_hidden_size;
    const int n_heads = (int)hp.lfm_n_heads;
    const int n_kv = (int)hp.lfm_n_kv_heads;
    const int hd = (int)hp.lfm_head_dim;
    const float norm_eps = 1e-5f;

    // Step 1: mel → encoder → adapter
    int T_mel = 0, n_mels = 0;
    auto mel = lfm2_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    n_mels = (int)hp.n_mels;

    // Run encoder
    int T_enc = 0, d_model = 0;
    float* enc = lfm2_audio_run_encoder(ctx, mel.data(), T_mel, n_mels, &T_enc, &d_model);
    if (!enc)
        return nullptr;

    // Run adapter
    int adapter_hidden = 0;
    float* adapted = lfm2_audio_run_adapter(ctx, enc, T_enc, d_model, &adapter_hidden);
    free(enc);
    if (!adapted)
        return nullptr;

    // Step 2: Build full graph for LFM2 backbone on the adapted audio embeddings
    // (simplified: no text tokens for now — just audio)
    const int T = T_enc; // just audio frames for now

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) {
        free(adapted);
        return nullptr;
    }

    // Input: adapted audio embeddings (T, hidden) → (hidden, T) in ggml
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T);
    memcpy(x->data, adapted, sizeof(float) * T * hidden);
    free(adapted);

    // Verify critical LFM weights are loaded
    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        auto& l = model.lfm_layers[i];
        if (!l.operator_norm_w || !l.ffn_norm_w || !l.ff_w1 || !l.ff_w2 || !l.ff_w3) {
            fprintf(stderr, "lfm2-audio: missing weights in LFM layer %u\n", i);
            ggml_free(ctx0);
            return nullptr;
        }
        if (l.is_attention && (!l.attn_q_proj_w || !l.attn_k_proj_w || !l.attn_v_proj_w)) {
            fprintf(stderr, "lfm2-audio: missing attention weights in LFM layer %u\n", i);
            ggml_free(ctx0);
            return nullptr;
        }
        if (!l.is_attention && (!l.conv_in_proj_w || !l.conv_conv_w || !l.conv_out_proj_w)) {
            fprintf(stderr, "lfm2-audio: missing conv weights in LFM layer %u\n", i);
            ggml_free(ctx0);
            return nullptr;
        }
    }


    // Run all 16 LFM2 layers, with optional per-layer snapshots
    std::vector<ggml_tensor*> layer_snaps(hp.lfm_n_layers, nullptr);
    bool do_snaps = (std::getenv("LFM2_SNAP_LAYERS") != nullptr);

    for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
        x = lfm2_build_layer(ctx0, x, model.lfm_layers[i], /*rope_freqs=*/nullptr, hidden, n_heads, n_kv, hd, T,
                             norm_eps);
        if (do_snaps) {
            layer_snaps[i] = ggml_dup(ctx0, x);
            char name[32];
            snprintf(name, sizeof(name), "lfm_ao_layer_%u", i);
            ggml_set_name(layer_snaps[i], name);
        }
    }

    // Final embedding norm (RMSNorm)
    x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, norm_eps);

    ggml_tensor* out = ggml_dup(ctx0, x);
    ggml_set_name(out, "lfm_output");

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);
    for (auto* snap : layer_snaps)
        if (snap)
            ggml_build_forward_expand(gf, snap);
    ggml_build_forward_expand(gf, out);

    ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);

    float* result = (float*)malloc(sizeof(float) * T * hidden);
    if (result)
        memcpy(result, out->data, sizeof(float) * T * hidden);

    // Invoke staged callback for per-layer snapshots
    if (do_snaps && ctx->lfm_stage_cb) {
        for (uint32_t i = 0; i < hp.lfm_n_layers; i++) {
            if (layer_snaps[i]) {
                char name[32];
                snprintf(name, sizeof(name), "lfm_ao_layer_%u", i);
                ctx->lfm_stage_cb(name, (const float*)layer_snaps[i]->data, T, hidden, ctx->lfm_stage_ud);
            }
        }
    }

    if (out_T)
        *out_T = T;
    if (out_hidden_size)
        *out_hidden_size = hidden;
    ggml_free(ctx0);
    return result;
}

int lfm2_audio_run_lfm_staged(lfm2_audio_context* ctx, const float* samples, int n_samples, lfm2_audio_stage_cb cb,
                              void* userdata) {
    if (!ctx)
        return -1;
    ctx->lfm_stage_cb = cb;
    ctx->lfm_stage_ud = userdata;

    // Force layer snaps on
    setenv("LFM2_SNAP_LAYERS", "1", 1);
    int T = 0, hidden = 0;
    float* result = lfm2_audio_run_lfm(ctx, samples, n_samples, &T, &hidden);
    unsetenv("LFM2_SNAP_LAYERS");

    ctx->lfm_stage_cb = nullptr;
    ctx->lfm_stage_ud = nullptr;

    free(result);
    return result ? 0 : -1;
}

// ===========================================================================
// Depthformer: generate 8-codebook Mimi codes from a backbone hidden vector
//
// For each audio frame, the depthformer:
//   1. depth_linear(hidden) → (codebooks, depth_dim) = (8, 1024)
//   2. For codebook i=0..7:
//      a. input = depth_linear_out[i] + embedding(prev_token) (0 for first)
//      b. Run 6-layer transformer (with KV cache within this frame)
//      c. Logits = to_logits(embedding_norm(output))
//      d. Greedy argmax → code[i]
//      e. Embed code[i] for next codebook
//   3. Return 8 codes. If code[0] == 2048 (EOAudio), stop generating.
// ===========================================================================

static std::vector<int32_t> lfm2_depthformer_sample_frame(lfm2_audio_context* ctx, const float* backbone_hidden) {
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int codebooks = (int)hp.codebooks;
    const int depth_dim = (int)hp.depth_dim;
    const int hidden = (int)hp.lfm_hidden_size;
    const int depth_n_layers = (int)hp.depth_n_layers;
    const float norm_eps = 1e-5f;
    const int audio_vocab = (int)hp.audio_vocab_size;

    // depth_linear: (hidden) → (codebooks * depth_dim)
    // Do this on CPU for simplicity (it's one matmul)
    const size_t mem_sz = 128 * 1024 * 1024; // 128 MB
    std::vector<uint8_t> buf(mem_sz);
    ggml_init_params ip = {mem_sz, buf.data(), false};
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0)
        return {};

    // Input hidden vector
    ggml_tensor* h_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden);
    memcpy(h_in->data, backbone_hidden, sizeof(float) * hidden);

    // depth_linear
    ggml_tensor* projected = ggml_add(ctx0, ggml_mul_mat(ctx0, model.depth_linear_w, h_in), model.depth_linear_b);
    // projected: (codebooks * depth_dim,) = (8192,)

    ggml_tensor* proj_out = ggml_dup(ctx0, projected);
    ggml_set_name(proj_out, "depth_proj");

    ggml_cgraph* gf0 = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf0, proj_out);
    ggml_graph_compute_with_ctx(ctx0, gf0, ctx->n_threads);

    // Extract projected values
    std::vector<float> proj_data(codebooks * depth_dim);
    memcpy(proj_data.data(), proj_out->data, sizeof(float) * codebooks * depth_dim);
    ggml_free(ctx0);

    // Now run the depthformer autoregressively over 8 codebooks
    std::vector<int32_t> codes(codebooks);
    std::vector<float> prev_emb(depth_dim, 0.0f); // zero for first codebook

    for (int c = 0; c < codebooks; c++) {
        // Input: depth_linear_out[c] + prev_emb
        std::vector<float> input(depth_dim);
        for (int j = 0; j < depth_dim; j++)
            input[j] = proj_data[c * depth_dim + j] + prev_emb[j];

        // Run 6-layer transformer on this single token
        // (no KV cache across codebooks — each frame is independent,
        //  but within a frame the 8 codebook steps share a cache)
        const size_t step_mem = 8 * 1024 * 1024;
        std::vector<uint8_t> step_buf(step_mem);
        ggml_init_params sip = {step_mem, step_buf.data(), false};
        ggml_context* sc = ggml_init(sip);
        if (!sc)
            break;

        ggml_tensor* x = ggml_new_tensor_2d(sc, GGML_TYPE_F32, depth_dim, c + 1);
        // Fill positions 0..c-1 with previously computed inputs, position c with current
        // Actually for efficiency, build the full (depth_dim, c+1) sequence
        // For now, just process position c with positions 0..c-1 via no-cache:
        // This is O(codebooks^2) per frame but codebooks=8, so 8*6 = 48 layer calls total.
        // Fast enough.

        // Simpler: build (depth_dim, 1) input for just this codebook position
        // and use KV cache. But building a KV cache for 6 layers × 8 steps is complex.
        //
        // Simplest correct approach: run all c+1 tokens through the transformer
        // each time (no cache). Total ops: sum(c+1 for c=0..7) = 36 token-layer pairs × 6 layers = 216.
        // At depth_dim=1024, this is tiny.

        // Build full sequence of c+1 inputs
        // prev inputs stored in a running buffer
        static thread_local std::vector<float> depth_sequence;
        if (c == 0)
            depth_sequence.clear();
        depth_sequence.insert(depth_sequence.end(), input.begin(), input.end());

        ggml_tensor* seq = ggml_new_tensor_2d(sc, GGML_TYPE_F32, depth_dim, c + 1);
        memcpy(seq->data, depth_sequence.data(), sizeof(float) * depth_dim * (c + 1));

        // Positions for RoPE
        ggml_tensor* positions = ggml_new_tensor_1d(sc, GGML_TYPE_I32, c + 1);
        {
            int32_t* pos = (int32_t*)positions->data;
            for (int i = 0; i <= c; i++)
                pos[i] = i;
        }

        // Causal mask (only needed if c > 0)
        ggml_tensor* mask = nullptr;
        if (c > 0) {
            mask = ggml_new_tensor_2d(sc, GGML_TYPE_F16, c + 1, c + 1);
            ggml_fp16_t* m = (ggml_fp16_t*)mask->data;
            ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            ggml_fp16_t neginf = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q <= c; q++)
                for (int k = 0; k <= c; k++)
                    m[q * (c + 1) + k] = (k <= q) ? zero : neginf;
        }

        // 6-layer depthformer transformer
        // The depthformer uses MHA with fused QKV (not GQA — head_dim=32, 32 heads, 8 KV heads)
        // Actually: qkv_proj is (1024→1536) = Q(1024) + K(256) + V(256)
        // So n_heads=32, n_kv_heads=8, head_dim=32
        const int d_n_heads = 32;
        const int d_n_kv = 8;
        const int d_hd = depth_dim / d_n_heads; // 32

        ggml_tensor* cur = seq;
        for (int il = 0; il < depth_n_layers; il++) {
            auto& dl = model.depth_layers[il];
            ggml_tensor* residual = cur;

            // RMSNorm → MHA
            ggml_tensor* h = lfm2_rms_norm(sc, cur, dl.operator_norm_w, norm_eps);

            // Fused QKV
            ggml_tensor* qkv = ggml_mul_mat(sc, dl.attn_qkv_proj_w, h);
            const int q_dim = d_n_heads * d_hd; // 1024
            const int kv_dim = d_n_kv * d_hd;   // 256
            ggml_tensor* Q = ggml_view_2d(sc, qkv, q_dim, c + 1, qkv->nb[1], 0);
            ggml_tensor* K = ggml_view_2d(sc, qkv, kv_dim, c + 1, qkv->nb[1], q_dim * sizeof(float));
            ggml_tensor* V = ggml_view_2d(sc, qkv, kv_dim, c + 1, qkv->nb[1], (q_dim + kv_dim) * sizeof(float));
            if (c > 0) {
                Q = ggml_cont(sc, Q);
                K = ggml_cont(sc, K);
                V = ggml_cont(sc, V);
            }

            Q = ggml_reshape_3d(sc, Q, d_hd, d_n_heads, c + 1);
            K = ggml_reshape_3d(sc, K, d_hd, d_n_kv, c + 1);
            V = ggml_reshape_3d(sc, V, d_hd, d_n_kv, c + 1);

            // QK layernorm
            Q = ggml_mul(sc, ggml_rms_norm(sc, Q, norm_eps), dl.attn_q_ln_w);
            K = ggml_mul(sc, ggml_rms_norm(sc, K, norm_eps), dl.attn_k_ln_w);

            // RoPE
            Q = ggml_rope_ext(sc, Q, positions, nullptr, d_hd, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f);
            K = ggml_rope_ext(sc, K, positions, nullptr, d_hd, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f);

            // Permute for flash_attn
            Q = ggml_cont(sc, ggml_permute(sc, Q, 0, 2, 1, 3));
            K = ggml_cont(sc, ggml_permute(sc, K, 0, 2, 1, 3));
            V = ggml_cont(sc, ggml_permute(sc, V, 0, 2, 1, 3));

            float scale = 1.0f / sqrtf((float)d_hd);
            ggml_tensor* attn = ggml_flash_attn_ext(sc, Q, K, V, mask, scale, 0.0f, 0.0f);
            attn = ggml_reshape_2d(sc, attn, depth_dim, c + 1);
            attn = ggml_mul_mat(sc, dl.attn_out_proj_w, attn);

            cur = ggml_add(sc, residual, attn);

            // FFN
            residual = cur;
            h = lfm2_rms_norm(sc, cur, dl.ffn_norm_w, norm_eps);
            h = lfm2_swiglu_ffn(sc, h, dl.ff_w1, dl.ff_w2, dl.ff_w3);
            cur = ggml_add(sc, residual, h);
        }

        // Extract last position's output
        ggml_tensor* last = ggml_view_1d(sc, cur, depth_dim, (int64_t)c * depth_dim * sizeof(float));

        // Logits: embedding_norm(last) → to_logits
        auto& cb = model.depth_codebooks[c];
        ggml_tensor* normed = ggml_mul(sc, ggml_rms_norm(sc, last, norm_eps), cb.embedding_norm_w);
        ggml_tensor* logits = ggml_mul_mat(sc, cb.to_logits_w, normed);
        ggml_tensor* logits_out = ggml_dup(sc, logits);
        ggml_set_name(logits_out, "depth_logits");

        // Also compute embedding of the chosen token for next codebook
        // (we'll do this after graph eval by looking up the argmax)

        ggml_cgraph* gf = ggml_new_graph_custom(sc, 4096, false);
        ggml_build_forward_expand(gf, logits_out);
        ggml_graph_compute_with_ctx(sc, gf, ctx->n_threads);

        // Greedy argmax
        const float* ldata = (const float*)logits_out->data;
        int best = 0;
        for (int i = 1; i < audio_vocab; i++)
            if (ldata[i] > ldata[best])
                best = i;
        codes[c] = best;

        // Embed the token for next codebook input
        // cb.embedding_w has shape (depth_dim, audio_vocab_size)
        // Row `best` = embedding vector for token `best`
        {
            // Read embedding row via a tiny graph
            ggml_tensor* id_t = ggml_new_tensor_1d(sc, GGML_TYPE_I32, 1);
            *(int32_t*)id_t->data = best;
            ggml_tensor* emb = ggml_get_rows(sc, cb.embedding_w, id_t);
            ggml_tensor* emb_out = ggml_dup(sc, emb);
            ggml_cgraph* gf2 = ggml_new_graph(sc);
            ggml_build_forward_expand(gf2, emb_out);
            ggml_graph_compute_with_ctx(sc, gf2, 1);
            memcpy(prev_emb.data(), emb_out->data, sizeof(float) * depth_dim);
        }

        ggml_free(sc);
    }

    return codes;
}

// ===========================================================================
// Audio detokenizer (ISTFT-based): codes → PCM at 24 kHz
//
// Uses the separate detokenizer GGUF:
//   FusedEmbedding(codes) → upsample 6× → 8L LFM2 (512d) → linear → ISTFT
//
// For now, use a simplified CPU-side ISTFT since the detokenizer model
// needs separate loading. Instead, we output the raw Mimi codes and let
// the caller decode them externally.
// ===========================================================================

// ISTFT-based audio decoder: Mimi codes → PCM at 24 kHz.
// Uses the detokenizer model weights if loaded, otherwise returns nullptr.
#include "core/istft.h"

// Helper: embed audio codes via audio_embedding and sum across codebooks.
// Returns (hidden,) float vector = sum_c audio_embd[code_c + c * audio_vocab].
static std::vector<float> lfm2_embed_audio_codes(lfm2_audio_context* ctx, const std::vector<int32_t>& codes) {
    auto& model = ctx->model;
    const int hidden = (int)model.hparams.lfm_hidden_size;
    const int codebooks = (int)model.hparams.codebooks;
    const int audio_vocab = (int)model.hparams.audio_vocab_size;

    const size_t mem = 16 * 1024 * 1024;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context* c = ggml_init(ip);
    if (!c)
        return {};

    ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, codebooks);
    {
        int32_t* id = (int32_t*)ids->data;
        for (int i = 0; i < codebooks; i++)
            id[i] = codes[i] + i * audio_vocab;
    }
    ggml_tensor* emb = ggml_get_rows(c, model.audio_embd_embedding_w, ids);
    // emb: (hidden, codebooks). Sum over codebooks.
    ggml_tensor* summed = ggml_sum_rows(c, ggml_cont(c, ggml_transpose(c, emb)));
    summed = ggml_reshape_1d(c, summed, hidden);
    ggml_tensor* out = ggml_dup(c, summed);

    ggml_cgraph* gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(c, gf, 1);

    std::vector<float> result(hidden);
    memcpy(result.data(), out->data, sizeof(float) * hidden);
    ggml_free(c);
    return result;
}

// Lazy-load the detokenizer companion GGUF.
static bool lfm2_load_detokenizer(lfm2_audio_context* ctx) {
    auto& d = ctx->detok;
    if (d.loaded)
        return true;
    if (d.tried)
        return false;
    d.tried = true;
    std::string path = ctx->model_path;
    auto pos = path.rfind(".gguf");
    if (pos == std::string::npos)
        return false;
    std::string dp = path.substr(0, pos) + "-detokenizer.gguf";
    FILE* f = fopen(dp.c_str(), "rb");
    if (!f) {
        if (ctx->verbosity >= 1)
            fprintf(stderr, "lfm2-audio: detokenizer not found at %s\n", dp.c_str());
        return false;
    }
    fclose(f);
    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: loading detokenizer from %s\n", dp.c_str());
    gguf_context* gc = core_gguf::open_metadata(dp.c_str());
    if (!gc)
        return false;
    d.hidden = (int)core_gguf::kv_u32(gc, "lfm2audio.detok_hidden_size", 512);
    d.n_layers = (int)core_gguf::kv_u32(gc, "lfm2audio.detok_n_layers", 8);
    d.n_heads = (int)core_gguf::kv_u32(gc, "lfm2audio.detok_n_heads", 16);
    d.n_kv_heads = (int)core_gguf::kv_u32(gc, "lfm2audio.detok_n_kv_heads", 8);
    d.head_dim = d.hidden / d.n_heads;
    d.sliding_window = (int)core_gguf::kv_u32(gc, "lfm2audio.detok_sliding_window", 30);
    d.output_size = (int)core_gguf::kv_u32(gc, "lfm2audio.detok_output_size", 1282);
    d.layer_types = core_gguf::kv_str(gc, "lfm2audio.detok_layer_types", "ccacacacc");
    core_gguf::free_metadata(gc);
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(dp.c_str(), ctx->backend, "lfm2-detok", wl))
        return false;
    d.ctx = wl.ctx;
    d.buf = wl.buf;
    d.tensors = std::move(wl.tensors);
    auto G = [&](const char* n) { return core_gguf::try_get(d.tensors, n); };
    d.emb_w = G("detok.emb.weight");
    d.embedding_norm_w = G("detok.lfm.embedding_norm.weight");
    d.output_w = G("detok.output.weight");
    d.output_b = G("detok.output.bias");
    d.layers.resize(d.n_layers);
    for (int i = 0; i < d.n_layers; i++) {
        auto& l = d.layers[i];
        char b[128];
        auto T = [&](const char* s) {
            snprintf(b, sizeof(b), "detok.lfm.layers.%d.%s", i, s);
            return G(b);
        };
        l.operator_norm_w = T("operator_norm.weight");
        l.ffn_norm_w = T("ffn_norm.weight");
        l.ff_w1 = T("feed_forward.w1.weight");
        l.ff_w2 = T("feed_forward.w2.weight");
        l.ff_w3 = T("feed_forward.w3.weight");
        l.is_attention = (i < (int)d.layer_types.size() && d.layer_types[i] == 'a');
        if (l.is_attention) {
            l.attn_q_proj_w = T("self_attn.q_proj.weight");
            l.attn_k_proj_w = T("self_attn.k_proj.weight");
            l.attn_v_proj_w = T("self_attn.v_proj.weight");
            l.attn_out_proj_w = T("self_attn.out_proj.weight");
            l.attn_q_ln_w = T("self_attn.q_layernorm.weight");
            l.attn_k_ln_w = T("self_attn.k_layernorm.weight");
        } else {
            l.conv_conv_w = T("conv.conv.weight");
            l.conv_in_proj_w = T("conv.in_proj.weight");
            l.conv_out_proj_w = T("conv.out_proj.weight");
        }
    }
    d.loaded = (d.emb_w && d.output_w);
    if (d.loaded && ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: detokenizer loaded (%dL, %dd)\n", d.n_layers, d.hidden);
    return d.loaded;
}

static float* lfm2_detokenize(lfm2_audio_context* ctx, const std::vector<std::vector<int32_t>>& codes,
                              int* out_n_samples) {
    if (!lfm2_load_detokenizer(ctx)) {
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }
    auto& d = ctx->detok;
    const int nf = (int)codes.size(), cb = (int)codes[0].size(), h = d.hidden, Tu = nf * 6, nq = d.output_size / 2;

    // 1. FusedEmbedding: codes → mean embedding on CPU
    std::vector<float> emb(nf * h, 0.0f);
    for (int t = 0; t < nf; t++) {
        const size_t em = 16 * 1024 * 1024;
        std::vector<uint8_t> eb(em);
        ggml_context* ec = ggml_init({em, eb.data(), false});
        if (!ec)
            break;
        ggml_tensor* ids = ggml_new_tensor_1d(ec, GGML_TYPE_I32, cb);
        for (int c = 0; c < cb; c++)
            ((int32_t*)ids->data)[c] = codes[t][c] + c * 2048;
        ggml_tensor* rows = ggml_get_rows(ec, d.emb_w, ids);
        ggml_tensor* o = ggml_dup(ec, rows);
        ggml_cgraph* g = ggml_new_graph(ec);
        ggml_build_forward_expand(g, o);
        ggml_graph_compute_with_ctx(ec, g, 1);
        const float* od = (const float*)o->data;
        for (int j = 0; j < h; j++) {
            float s = 0;
            for (int c = 0; c < cb; c++)
                s += od[c * h + j];
            emb[t * h + j] = s / cb;
        }
        ggml_free(ec);
    }

    // 2. Upsample 6× nearest-exact
    std::vector<float> up(Tu * h);
    for (int t = 0; t < nf; t++)
        for (int r = 0; r < 6; r++)
            memcpy(up.data() + (t * 6 + r) * h, emb.data() + t * h, sizeof(float) * h);

    // 3. Run 8L LFM2 backbone with sliding window mask
    const size_t mem = 512ULL * 1024 * 1024;
    std::vector<uint8_t> buf(mem);
    ggml_context* c0 = ggml_init({mem, buf.data(), false});
    if (!c0) {
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }
    ggml_tensor* x = ggml_new_tensor_2d(c0, GGML_TYPE_F32, h, Tu);
    memcpy(x->data, up.data(), sizeof(float) * Tu * h);
    ggml_tensor* mask = ggml_new_tensor_2d(c0, GGML_TYPE_F16, Tu, Tu);
    {
        ggml_fp16_t *m = (ggml_fp16_t*)mask->data, z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < Tu; q++)
            for (int k = 0; k < Tu; k++)
                m[q * Tu + k] = ((k <= q) && (q - k) < d.sliding_window) ? z : ni;
    }
    ggml_tensor* pos = ggml_new_tensor_1d(c0, GGML_TYPE_I32, Tu);
    for (int i = 0; i < Tu; i++)
        ((int32_t*)pos->data)[i] = i;
    ggml_cgraph* gf = ggml_new_graph_custom(c0, 16384, false);
    for (int il = 0; il < d.n_layers; il++) {
        auto& l = d.layers[il];
        ggml_tensor* res = x;
        ggml_tensor* hh = lfm2_rms_norm(c0, x, l.operator_norm_w, 1e-5f);
        if (l.is_attention) {
            ggml_tensor *Q = ggml_mul_mat(c0, l.attn_q_proj_w, hh), *K = ggml_mul_mat(c0, l.attn_k_proj_w, hh),
                        *V = ggml_mul_mat(c0, l.attn_v_proj_w, hh);
            Q = ggml_reshape_3d(c0, Q, d.head_dim, d.n_heads, Tu);
            K = ggml_reshape_3d(c0, K, d.head_dim, d.n_kv_heads, Tu);
            V = ggml_reshape_3d(c0, V, d.head_dim, d.n_kv_heads, Tu);
            if (l.attn_q_ln_w) {
                Q = ggml_mul(c0, ggml_rms_norm(c0, Q, 1e-5f), l.attn_q_ln_w);
                K = ggml_mul(c0, ggml_rms_norm(c0, K, 1e-5f), l.attn_k_ln_w);
            }
            Q = ggml_rope_ext(c0, Q, pos, nullptr, d.head_dim, GGML_ROPE_TYPE_NEOX, 0, d.rope_theta, 1, 0, 1, 0, 0);
            K = ggml_rope_ext(c0, K, pos, nullptr, d.head_dim, GGML_ROPE_TYPE_NEOX, 0, d.rope_theta, 1, 0, 1, 0, 0);
            Q = ggml_cont(c0, ggml_permute(c0, Q, 0, 2, 1, 3));
            K = ggml_cont(c0, ggml_permute(c0, K, 0, 2, 1, 3));
            V = ggml_cont(c0, ggml_permute(c0, V, 0, 2, 1, 3));
            ggml_tensor* a = ggml_flash_attn_ext(c0, Q, K, V, mask, 1.0f / sqrtf((float)d.head_dim), 0, 0);
            hh = ggml_mul_mat(c0, l.attn_out_proj_w, ggml_reshape_2d(c0, a, h, Tu));
        } else {
            ggml_tensor* bc = ggml_mul_mat(c0, l.conv_in_proj_w, hh);
            ggml_tensor *B = ggml_view_2d(c0, bc, h, Tu, bc->nb[1], 0),
                        *C = ggml_view_2d(c0, bc, h, Tu, bc->nb[1], h * sizeof(float));
            ggml_tensor* xi = ggml_view_2d(c0, bc, h, Tu, bc->nb[1], 2 * h * sizeof(float));
            ggml_tensor* Bx = ggml_mul(c0, ggml_cont(c0, B), ggml_cont(c0, xi));
            int K = d.conv_kernel;
            ggml_tensor* cw = ggml_cast(c0, l.conv_conv_w, GGML_TYPE_F32);
            ggml_tensor* cw4 = ggml_reshape_4d(c0, cw, K, 1, 1, h);
            ggml_tensor* Bt = ggml_cont(c0, ggml_transpose(c0, Bx));
            ggml_tensor* B4 = ggml_reshape_4d(c0, Bt, Tu, 1, h, 1);
            ggml_tensor* cr = ggml_conv_2d_dw_direct(c0, cw4, B4, 1, 1, K - 1, 0, 1, 1);
            cr = ggml_cont(c0, ggml_permute(c0, cr, 1, 2, 0, 3));
            int Tc = (int)cr->ne[1];
            if (Tc > Tu)
                cr = ggml_view_2d(c0, ggml_reshape_2d(c0, cr, h, Tc), h, Tu, h * sizeof(float), 0);
            else
                cr = ggml_reshape_2d(c0, cr, h, Tu);
            ggml_tensor* y = ggml_mul(c0, ggml_cont(c0, C), ggml_cont(c0, cr));
            hh = ggml_mul_mat(c0, l.conv_out_proj_w, y);
        }
        x = ggml_add(c0, res, hh);
        res = x;
        hh = lfm2_rms_norm(c0, x, l.ffn_norm_w, 1e-5f);
        hh = lfm2_swiglu_ffn(c0, hh, l.ff_w1, l.ff_w2, l.ff_w3);
        x = ggml_add(c0, res, hh);
    }
    x = lfm2_rms_norm(c0, x, d.embedding_norm_w, 1e-5f);
    ggml_tensor* lin = ggml_mul_mat(c0, d.output_w, x);
    if (d.output_b)
        lin = ggml_add(c0, lin, d.output_b);
    ggml_tensor* out = ggml_dup(c0, lin);
    ggml_set_name(out, "do");
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(c0, gf, ctx->n_threads);

    // 4. Split log_mag + angle → ISTFT → PCM
    const float* od = (const float*)out->data;
    std::vector<float> mag(Tu * nq), phase(Tu * nq);
    for (int t = 0; t < Tu; t++)
        for (int f = 0; f < nq; f++) {
            mag[t * nq + f] = expf(od[t * d.output_size + f]);
            phase[t * nq + f] = od[t * d.output_size + nq + f];
        }
    ggml_free(c0);
    auto pcm_v = core_istft::istft(mag.data(), phase.data(), 1280, 320, Tu, nullptr, core_istft::TRIM_SAME);
    for (float& s : pcm_v)
        s = std::max(-1.0f, std::min(1.0f, s));
    float* pcm = (float*)malloc(pcm_v.size() * sizeof(float));
    memcpy(pcm, pcm_v.data(), pcm_v.size() * sizeof(float));
    if (out_n_samples)
        *out_n_samples = (int)pcm_v.size();
    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: detokenizer produced %zu samples (%.1fs at 24kHz)\n", pcm_v.size(),
                pcm_v.size() / 24000.0f);
    return pcm;
}

float* lfm2_audio_synthesize(lfm2_audio_context* ctx, const char* text, const char* language, int* out_n_samples) {
    if (!ctx || !text)
        return nullptr;
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int hidden = (int)hp.lfm_hidden_size;
    const int codebooks = (int)hp.codebooks;
    const int n_text = (int)hp.interleaved_n_text;   // 6
    const int n_audio = (int)hp.interleaved_n_audio; // 9

    if (!model.depth_linear_w || model.depth_codebooks.empty()) {
        fprintf(stderr, "lfm2-audio: depthformer weights not loaded\n");
        return nullptr;
    }

    // Step 1: Build prompt — TTS uses "Perform TTS in {language}."
    // Pre-tokenized for Japanese and English
    // <|startoftext|><|im_start|>system\nPerform TTS in japanese.<|im_end|>\n<|im_start|>user\n
    // "Perform" = [8173, 1199], " T" = 837, "TS" = 10255, " in" = 797,
    // " japan" = 41035, "ese" = 3391, "." = 523
    static const std::vector<int32_t> kTTSPrefixJa = {1,     6,    24131, 708, 8173, 1199, 837,  10255, 797,
                                                      41035, 3391, 523,   7,   708,  6,    6423, 708};
    // <|startoftext|><|im_start|>system\nPerform TTS in english.<|im_end|>\n<|im_start|>user\n
    static const std::vector<int32_t> kTTSPrefixEn = {1,   6,     24131, 708, 8173, 1199, 837,  10255,
                                                      797, 48103, 523,   7,   708,  6,    6423, 708};

    const auto* prefix_ptr = &kTTSPrefixJa;
    if (language && (std::string(language) == "en" || std::string(language) == "english"))
        prefix_ptr = &kTTSPrefixEn;

    // Tokenize the input text using BPE
    // For now, just use the core_bpe encoder
    // TODO: proper BPE encoding. For now, encode character-by-character as a fallback.
    // Actually, for TTS the text needs to be tokenized. Let me use the vocab lookup.

    // Simple approach: for Japanese text, each character maps to a token.
    // For a proper implementation we need BPE merges. For now, use a hack:
    // search the vocab for the input text as a whole, then fall back to
    // per-character lookup.
    std::vector<int32_t> text_tokens;
    {
        // Try to find each character/subword in the vocab
        std::string remaining(text);
        while (!remaining.empty()) {
            bool found = false;
            // Try longest match first (up to 8 bytes)
            for (int len = std::min((int)remaining.size(), 16); len >= 1; len--) {
                std::string sub = remaining.substr(0, len);
                // Convert to GPT-2 byte encoding
                std::string encoded;
                const auto& be = core_bpe::byte_encoder();
                for (unsigned char ch : sub) {
                    int cp = be[ch];
                    if (cp < 128) {
                        encoded += (char)cp;
                    } else {
                        // Encode as UTF-8
                        if (cp < 0x800) {
                            encoded += (char)(0xC0 | (cp >> 6));
                            encoded += (char)(0x80 | (cp & 0x3F));
                        } else {
                            encoded += (char)(0xE0 | (cp >> 12));
                            encoded += (char)(0x80 | ((cp >> 6) & 0x3F));
                            encoded += (char)(0x80 | (cp & 0x3F));
                        }
                    }
                }
                // Search vocab
                for (int i = 0; i < (int)model.vocab.size(); i++) {
                    if (model.vocab[i] == encoded) {
                        text_tokens.push_back(i);
                        remaining = remaining.substr(len);
                        found = true;
                        break;
                    }
                }
                if (found)
                    break;
            }
            if (!found) {
                // Skip unknown byte
                remaining = remaining.substr(1);
            }
        }
    }

    if (text_tokens.empty()) {
        fprintf(stderr, "lfm2-audio: could not tokenize input text\n");
        return nullptr;
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: TTS text tokenized to %zu tokens\n", text_tokens.size());

    // Build full input: prefix + text + <|im_end|>\n
    std::vector<int32_t> all_tokens;
    all_tokens.insert(all_tokens.end(), prefix_ptr->begin(), prefix_ptr->end());
    all_tokens.insert(all_tokens.end(), text_tokens.begin(), text_tokens.end());
    all_tokens.push_back(kTokenImEnd);
    all_tokens.push_back(kTokenNewline);

    // Embed all tokens
    auto embed_text = [&](const std::vector<int32_t>& ids) -> std::vector<float> {
        const int n = (int)ids.size();
        const size_t mem = 16 * 1024 * 1024;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context* c = ggml_init(ip);
        if (!c)
            return {};
        ggml_tensor* id_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        memcpy(id_t->data, ids.data(), n * sizeof(int32_t));
        ggml_tensor* emb = ggml_get_rows(c, model.lfm_embed_tokens_w, id_t);
        ggml_tensor* out = ggml_dup(c, emb);
        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(c, gf, 1);
        std::vector<float> result(n * hidden);
        memcpy(result.data(), out->data, sizeof(float) * n * hidden);
        ggml_free(c);
        return result;
    };

    auto text_emb = embed_text(all_tokens);
    if (text_emb.empty())
        return nullptr;

    // Step 2: Interleaved generation using cached backbone.
    //
    // The model alternates between text (n_text=6 tokens) and audio
    // (n_audio=9 frames). We reuse the KV+conv cached backbone from
    // the ASR transcribe path. Audio frames go through the depthformer
    // to produce 8-codebook Mimi codes.
    // Reuse the SAME cached backbone from the ASR transcribe path.
    ctx->reset_kv();

    // Prefill: embed all text tokens and run through the cached backbone
    auto prefill_result = [&]() -> std::pair<std::vector<float>, std::vector<float>> {
        const int T = (int)all_tokens.size();
        // Reuse the run_step from transcribe — same KV+conv cache pattern.
        // We duplicate the code here to also return the hidden state.
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
        ggml_context* ctx0 = ggml_init(ip);
        if (!ctx0)
            return {};

        ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T);
        memcpy(x->data, text_emb.data(), sizeof(float) * T * hidden);
        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        for (int i = 0; i < T; i++)
            ((int32_t*)positions->data)[i] = i;
        ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        {
            ggml_fp16_t *m = (ggml_fp16_t*)mask->data, z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < T; q++)
                for (int k = 0; k < T; k++)
                    m[q * T + k] = (k <= q) ? z : ni;
        }

        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);
        int ai = 0, ci = 0;
        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            auto& w = model.lfm_layers[il];
            ggml_tensor* res = x;
            ggml_tensor* h = lfm2_rms_norm(ctx0, x, w.operator_norm_w, 1e-5f);
            if (w.is_attention) {
                core_attn::KvSelfAttnParams kvp = {};
                kvp.head_dim = (int)hp.lfm_head_dim;
                kvp.n_heads = (int)hp.lfm_n_heads;
                kvp.n_kv_heads = (int)hp.lfm_n_kv_heads;
                kvp.n_kv_grp = kvp.n_heads / kvp.n_kv_heads;
                kvp.rope_type = GGML_ROPE_TYPE_NEOX;
                kvp.rope_theta = hp.lfm_rope_theta;
                kvp.attn_scale = 1.0f / sqrtf((float)hp.lfm_head_dim);
                kvp.qk_norm_eps = 1e-5f;
                kvp.gqa_mode = core_attn::GQA_NATIVE;
                h = core_attn::kv_self_attn(ctx0, gf, h, w.attn_q_proj_w, w.attn_k_proj_w, w.attn_v_proj_w,
                                            w.attn_out_proj_w, w.attn_q_ln_w, w.attn_k_ln_w, positions, mask, ctx->kv_k,
                                            ctx->kv_v, ai, 0, kvp);
                ai++;
            } else {
                const int K = (int)hp.lfm_conv_kernel;
                ggml_tensor* bc = ggml_mul_mat(ctx0, w.conv_in_proj_w, h);
                ggml_tensor* Bs = ggml_view_2d(ctx0, bc, hidden, T, bc->nb[1], 0);
                ggml_tensor* xs = ggml_view_2d(ctx0, bc, hidden, T, bc->nb[1], 2 * hidden * sizeof(float));
                ggml_tensor* Bx = ggml_mul(ctx0, ggml_cont(ctx0, Bs), ggml_cont(ctx0, xs));
                int tail = T - (K - 1);
                if (tail < 0)
                    tail = 0;
                int tl = T - tail;
                ggml_tensor* snap = ggml_dup(ctx0, ggml_view_2d(ctx0, Bx, hidden, tl, hidden * sizeof(float),
                                                                (int64_t)tail * hidden * sizeof(float)));
                char sn[16];
                snprintf(sn, sizeof(sn), "cs_%d", ci);
                ggml_set_name(snap, sn);
                ggml_build_forward_expand(gf, snap);
                h = lfm2_short_conv(ctx0, h, w, hidden, T);
                ci++;
            }
            x = ggml_add(ctx0, res, h);
            res = x;
            h = lfm2_rms_norm(ctx0, x, w.ffn_norm_w, 1e-5f);
            h = lfm2_swiglu_ffn(ctx0, h, w.ff_w1, w.ff_w2, w.ff_w3);
            x = ggml_add(ctx0, res, h);
        }
        x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, 1e-5f);
        ggml_tensor* last = ggml_view_1d(ctx0, x, hidden, (int64_t)(T - 1) * hidden * sizeof(float));
        ggml_tensor* lg = ggml_dup(ctx0, ggml_mul_mat(ctx0, model.lfm_embed_tokens_w, last));
        ggml_tensor* hd = ggml_dup(ctx0, last);
        ggml_set_name(lg, "lg");
        ggml_set_name(hd, "hd");
        ggml_build_forward_expand(gf, lg);
        ggml_build_forward_expand(gf, hd);
        ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);
        // Save conv states
        ci = 0;
        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            if (model.lfm_layers[il].is_attention)
                continue;
            char sn[16];
            snprintf(sn, sizeof(sn), "cs_%d", ci);
            ggml_tensor* s = ggml_graph_get_tensor(gf, sn);
            if (s) {
                int sc = (int)s->ne[1];
                auto& st = ctx->conv_states[ci];
                std::fill(st.begin(), st.end(), 0.0f);
                int off = (int)hp.lfm_conv_kernel - 1 - sc;
                if (off < 0)
                    off = 0;
                memcpy(st.data() + off * hidden, s->data, sizeof(float) * sc * hidden);
            }
            ci++;
        }
        ctx->kv_n_past = T;
        std::vector<float> rlg((int)lg->ne[0]);
        memcpy(rlg.data(), lg->data, sizeof(float) * rlg.size());
        std::vector<float> rhd(hidden);
        memcpy(rhd.data(), hd->data, sizeof(float) * hidden);
        ggml_free(ctx0);
        return {rlg, rhd};
    };

    auto [logits, last_hidden] = prefill_result();
    if (logits.empty())
        return nullptr;

    auto argmax = [](const std::vector<float>& v) {
        int b = 0;
        for (int i = 1; i < (int)v.size(); i++)
            if (v[i] > v[b])
                b = i;
        return b;
    };
    int cur_token = argmax(logits);
    std::vector<float> cur_hidden = last_hidden;

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: TTS prefill top=%d\n", cur_token);

    // Interleaved generation: alternate text (n_text=6) and audio (n_audio=9)
    enum { MOD_TEXT, MOD_AUDIO };
    int cur_mod = MOD_TEXT, mod_left = n_text;
    bool text_done = false;
    std::vector<std::vector<int32_t>> all_codes;

    // Single-token backbone step with KV+conv cache (same as ASR decode)
    auto step1 = [&](const float* emb) -> std::pair<std::vector<float>, std::vector<float>> {
        ggml_init_params ip = {ctx->decode_meta.size(), ctx->decode_meta.data(), false};
        ggml_context* c = ggml_init(ip);
        if (!c)
            return {};
        ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, hidden, 1);
        memcpy(x->data, emb, sizeof(float) * hidden);
        ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        *(int32_t*)pos->data = ctx->kv_n_past;
        ggml_cgraph* gf = ggml_new_graph_custom(c, 65536, false);
        int ai = 0, ci = 0;
        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            auto& w = model.lfm_layers[il];
            ggml_tensor* res = x;
            ggml_tensor* h = lfm2_rms_norm(c, x, w.operator_norm_w, 1e-5f);
            if (w.is_attention) {
                core_attn::KvSelfAttnParams kvp = {};
                kvp.head_dim = (int)hp.lfm_head_dim;
                kvp.n_heads = (int)hp.lfm_n_heads;
                kvp.n_kv_heads = (int)hp.lfm_n_kv_heads;
                kvp.n_kv_grp = kvp.n_heads / kvp.n_kv_heads;
                kvp.rope_type = GGML_ROPE_TYPE_NEOX;
                kvp.rope_theta = hp.lfm_rope_theta;
                kvp.attn_scale = 1.0f / sqrtf((float)hp.lfm_head_dim);
                kvp.qk_norm_eps = 1e-5f;
                kvp.gqa_mode = core_attn::GQA_NATIVE;
                h = core_attn::kv_self_attn(c, gf, h, w.attn_q_proj_w, w.attn_k_proj_w, w.attn_v_proj_w,
                                            w.attn_out_proj_w, w.attn_q_ln_w, w.attn_k_ln_w, pos, nullptr, ctx->kv_k,
                                            ctx->kv_v, ai, ctx->kv_n_past, kvp);
                ai++;
            } else {
                const int K = (int)hp.lfm_conv_kernel;
                ggml_tensor* bc = ggml_mul_mat(c, w.conv_in_proj_w, h);
                ggml_tensor* Bp = ggml_view_1d(c, bc, hidden, 0);
                ggml_tensor* Cp = ggml_view_1d(c, bc, hidden, hidden * sizeof(float));
                ggml_tensor* xp = ggml_view_1d(c, bc, hidden, 2 * hidden * sizeof(float));
                ggml_tensor* Bx = ggml_mul(c, ggml_cont(c, Bp), ggml_cont(c, xp));
                ggml_tensor* cached = ggml_new_tensor_2d(c, GGML_TYPE_F32, hidden, K - 1);
                memcpy(cached->data, ctx->conv_states[ci].data(), sizeof(float) * hidden * (K - 1));
                ggml_tensor* Bxf = ggml_concat(c, cached, ggml_reshape_2d(c, Bx, hidden, 1), 1);
                ggml_tensor* cw = ggml_cast(c, w.conv_conv_w, GGML_TYPE_F32);
                ggml_tensor* cw4 = ggml_reshape_4d(c, cw, K, 1, 1, hidden);
                ggml_tensor* Bt = ggml_cont(c, ggml_transpose(c, Bxf));
                ggml_tensor* B4 = ggml_reshape_4d(c, Bt, K, 1, hidden, 1);
                ggml_tensor* cr = ggml_conv_2d_dw_direct(c, cw4, B4, 1, 1, K - 1, 0, 1, 1);
                cr = ggml_cont(c, ggml_permute(c, cr, 1, 2, 0, 3));
                cr = ggml_reshape_2d(c, cr, hidden, (int)cr->ne[1]);
                ggml_tensor* co = ggml_cont(c, ggml_view_2d(c, cr, hidden, 1, hidden * sizeof(float),
                                                            (int64_t)(K - 1) * hidden * sizeof(float)));
                ggml_tensor* y = ggml_mul(c, ggml_reshape_2d(c, ggml_cont(c, Cp), hidden, 1), co);
                h = ggml_mul_mat(c, w.conv_out_proj_w, y);
                ggml_tensor* snap = ggml_dup(c, Bx);
                char sn[16];
                snprintf(sn, sizeof(sn), "cs_%d", ci);
                ggml_set_name(snap, sn);
                ggml_build_forward_expand(gf, snap);
                ci++;
            }
            x = ggml_add(c, res, h);
            res = x;
            h = lfm2_rms_norm(c, x, w.ffn_norm_w, 1e-5f);
            h = lfm2_swiglu_ffn(c, h, w.ff_w1, w.ff_w2, w.ff_w3);
            x = ggml_add(c, res, h);
        }
        x = lfm2_rms_norm(c, x, model.lfm_embedding_norm_w, 1e-5f);
        ggml_tensor* lg = ggml_dup(c, ggml_mul_mat(c, model.lfm_embed_tokens_w, x));
        ggml_tensor* hd = ggml_dup(c, x);
        ggml_set_name(lg, "lg");
        ggml_set_name(hd, "hd");
        ggml_build_forward_expand(gf, lg);
        ggml_build_forward_expand(gf, hd);
        ggml_graph_compute_with_ctx(c, gf, ctx->n_threads);
        // Update conv states
        ci = 0;
        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            if (model.lfm_layers[il].is_attention)
                continue;
            char sn[16];
            snprintf(sn, sizeof(sn), "cs_%d", ci);
            ggml_tensor* s = ggml_graph_get_tensor(gf, sn);
            if (s) {
                auto& st = ctx->conv_states[ci];
                memmove(st.data(), st.data() + hidden, sizeof(float) * hidden * ((int)hp.lfm_conv_kernel - 2));
                memcpy(st.data() + hidden * ((int)hp.lfm_conv_kernel - 2), s->data, sizeof(float) * hidden);
            }
            ci++;
        }
        ctx->kv_n_past++;
        std::vector<float> rlg((int)lg->ne[0]);
        memcpy(rlg.data(), lg->data, sizeof(float) * rlg.size());
        std::vector<float> rhd(hidden);
        memcpy(rhd.data(), hd->data, sizeof(float) * hidden);
        ggml_free(c);
        return {rlg, rhd};
    };

    for (int step = 0; step < 1000; step++) {
        mod_left--;
        if (cur_mod == MOD_TEXT) {
            if (cur_token == kTokenImEnd || cur_token == 2)
                break;
            if (cur_token == kTokenTextEnd)
                text_done = true;
            if (mod_left <= 0 || text_done) {
                cur_mod = MOD_AUDIO;
                mod_left = n_audio;
            }
            auto tok_emb = embed_text({cur_token});
            if (tok_emb.empty())
                break;
            auto [lg, hd] = step1(tok_emb.data());
            if (lg.empty())
                break;
            cur_token = argmax(lg);
            cur_hidden = hd;
        } else {
            auto codes = lfm2_depthformer_sample_frame(ctx, cur_hidden.data());
            if (codes.empty())
                break;
            if (codes[0] == 2048)
                break; // EOAudio
            all_codes.push_back(codes);
            if (ctx->verbosity >= 2)
                fprintf(stderr, "  audio[%zu]: [%d,%d,%d,%d,...]\n", all_codes.size(), codes[0], codes[1], codes[2],
                        codes[3]);
            if (mod_left <= 0 && !text_done) {
                cur_mod = MOD_TEXT;
                mod_left = n_text;
            }
            auto ae = lfm2_embed_audio_codes(ctx, codes);
            if (ae.empty())
                break;
            auto [lg, hd] = step1(ae.data());
            if (lg.empty())
                break;
            cur_token = argmax(lg);
            cur_hidden = hd;
        }
    }

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: TTS generated %zu audio frames\n", all_codes.size());

    if (all_codes.empty()) {
        if (out_n_samples)
            *out_n_samples = 0;
        return nullptr;
    }

    // Decode codes → PCM via detokenizer
    float* pcm = lfm2_detokenize(ctx, all_codes, out_n_samples);
    if (!pcm) {
        if (ctx->verbosity >= 1)
            fprintf(stderr, "lfm2-audio: detokenizer not loaded, returning silence (%zu frames = %.1fs)\n",
                    all_codes.size(), all_codes.size() / 12.5f);
        const int total = (int)all_codes.size() * 1920;
        pcm = (float*)calloc(total, sizeof(float));
        if (out_n_samples)
            *out_n_samples = total;
    }
    return pcm;
}

// ===========================================================================
// Speech-to-speech: audio in → conformer → adapter → interleaved generation
// → depthformer → detokenizer → PCM out.
//
// The prefill includes: system_prompt + audio_embeddings + end_turn + assistant_start.
// The generation loop is the same interleaved mode as TTS but with audio context.
// ===========================================================================

float* lfm2_audio_speech_to_speech(lfm2_audio_context* ctx, const float* in_samples, int n_in_samples,
                                   const char* language, char** out_text, int* out_n_samples) {
    if (!ctx || !in_samples)
        return nullptr;
    auto& model = ctx->model;
    auto& hp = model.hparams;
    const int hidden = (int)hp.lfm_hidden_size;
    const int codebooks = (int)hp.codebooks;
    const int n_text_budget = (int)hp.interleaved_n_text;
    const int n_audio_budget = (int)hp.interleaved_n_audio;

    // Step 1: Encode input audio → adapter embeddings
    int T_mel = 0;
    auto mel = lfm2_compute_mel_impl(ctx, in_samples, n_in_samples, T_mel);
    if (mel.empty())
        return nullptr;

    int T_enc = 0, d_model = 0;
    float* enc = lfm2_audio_run_encoder(ctx, mel.data(), T_mel, (int)hp.n_mels, &T_enc, &d_model);
    if (!enc)
        return nullptr;

    int adapter_hidden = 0;
    float* adapted = lfm2_audio_run_adapter(ctx, enc, T_enc, d_model, &adapter_hidden);
    free(enc);
    if (!adapted)
        return nullptr;

    // Step 2: Build prompt sequence:
    // <|startoftext|><|im_start|>system\nRespond with interleaved text and audio.<|im_end|>\n
    // <|im_start|>user\n [audio_embeddings] <|im_end|>\n
    // <|im_start|>assistant\n
    // Pre-tokenized system prompt for speech-to-speech
    static const std::vector<int32_t> kS2SPrefix = {1,    6,   24131, 708, 3104, 4168, 916, 1251, 799, 17927,
                                                    3304, 810, 14052, 523, 7,    708,  6,   6423, 708};
    static const std::vector<int32_t> kS2SSuffix = {7, 708, 6, 64015, 708}; // <|im_end|>\n<|im_start|>assistant\n

    // Embed text tokens
    auto embed_text_fn = [&](const std::vector<int32_t>& ids) -> std::vector<float> {
        const int n = (int)ids.size();
        const size_t mem = 16 * 1024 * 1024;
        std::vector<uint8_t> buf(mem);
        ggml_context* c = ggml_init({mem, buf.data(), false});
        if (!c)
            return {};
        ggml_tensor* id_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        memcpy(id_t->data, ids.data(), n * sizeof(int32_t));
        ggml_tensor* emb = ggml_get_rows(c, model.lfm_embed_tokens_w, id_t);
        ggml_tensor* out = ggml_dup(c, emb);
        ggml_cgraph* gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(c, gf, 1);
        std::vector<float> result(n * hidden);
        memcpy(result.data(), out->data, sizeof(float) * n * hidden);
        ggml_free(c);
        return result;
    };

    auto prefix_emb = embed_text_fn(kS2SPrefix);
    auto suffix_emb = embed_text_fn(kS2SSuffix);
    if (prefix_emb.empty() || suffix_emb.empty()) {
        free(adapted);
        return nullptr;
    }

    // Assemble: prefix + audio + suffix
    const int T_prefix = (int)kS2SPrefix.size();
    const int T_audio = T_enc;
    const int T_suffix = (int)kS2SSuffix.size();
    const int T_total = T_prefix + T_audio + T_suffix;

    std::vector<float> context_emb(T_total * hidden);
    memcpy(context_emb.data(), prefix_emb.data(), sizeof(float) * T_prefix * hidden);
    memcpy(context_emb.data() + T_prefix * hidden, adapted, sizeof(float) * T_audio * hidden);
    memcpy(context_emb.data() + (T_prefix + T_audio) * hidden, suffix_emb.data(), sizeof(float) * T_suffix * hidden);
    free(adapted);

    if (ctx->verbosity >= 1)
        fprintf(stderr, "lfm2-audio: S2S prefill T=%d (prefix=%d audio=%d suffix=%d)\n", T_total, T_prefix, T_audio,
                T_suffix);

    // Step 3: Prefill cached backbone (same as TTS)
    ctx->reset_kv();
    {
        ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), false};
        ggml_context* ctx0 = ggml_init(ip);
        if (!ctx0)
            return nullptr;
        ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, T_total);
        memcpy(x->data, context_emb.data(), sizeof(float) * T_total * hidden);
        ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T_total);
        for (int i = 0; i < T_total; i++)
            ((int32_t*)positions->data)[i] = i;
        ggml_tensor* mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T_total, T_total);
        {
            ggml_fp16_t *m = (ggml_fp16_t*)mask->data, z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
            for (int q = 0; q < T_total; q++)
                for (int k = 0; k < T_total; k++)
                    m[q * T_total + k] = (k <= q) ? z : ni;
        }
        ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);
        int ai = 0, ci = 0;
        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            auto& w = model.lfm_layers[il];
            ggml_tensor* res = x;
            ggml_tensor* h = lfm2_rms_norm(ctx0, x, w.operator_norm_w, 1e-5f);
            if (w.is_attention) {
                core_attn::KvSelfAttnParams kvp = {};
                kvp.head_dim = (int)hp.lfm_head_dim;
                kvp.n_heads = (int)hp.lfm_n_heads;
                kvp.n_kv_heads = (int)hp.lfm_n_kv_heads;
                kvp.n_kv_grp = kvp.n_heads / kvp.n_kv_heads;
                kvp.rope_type = GGML_ROPE_TYPE_NEOX;
                kvp.rope_theta = hp.lfm_rope_theta;
                kvp.attn_scale = 1.0f / sqrtf((float)hp.lfm_head_dim);
                kvp.qk_norm_eps = 1e-5f;
                kvp.gqa_mode = core_attn::GQA_NATIVE;
                h = core_attn::kv_self_attn(ctx0, gf, h, w.attn_q_proj_w, w.attn_k_proj_w, w.attn_v_proj_w,
                                            w.attn_out_proj_w, w.attn_q_ln_w, w.attn_k_ln_w, positions, mask, ctx->kv_k,
                                            ctx->kv_v, ai, 0, kvp);
                ai++;
            } else {
                const int K = (int)hp.lfm_conv_kernel;
                ggml_tensor* bc = ggml_mul_mat(ctx0, w.conv_in_proj_w, h);
                ggml_tensor* Bs = ggml_view_2d(ctx0, bc, hidden, T_total, bc->nb[1], 0);
                ggml_tensor* xs = ggml_view_2d(ctx0, bc, hidden, T_total, bc->nb[1], 2 * hidden * sizeof(float));
                ggml_tensor* Bx = ggml_mul(ctx0, ggml_cont(ctx0, Bs), ggml_cont(ctx0, xs));
                int tail = T_total - (K - 1);
                if (tail < 0)
                    tail = 0;
                int tl = T_total - tail;
                ggml_tensor* snap = ggml_dup(ctx0, ggml_view_2d(ctx0, Bx, hidden, tl, hidden * sizeof(float),
                                                                (int64_t)tail * hidden * sizeof(float)));
                char sn[16];
                snprintf(sn, sizeof(sn), "cs_%d", ci);
                ggml_set_name(snap, sn);
                ggml_build_forward_expand(gf, snap);
                h = lfm2_short_conv(ctx0, h, w, hidden, T_total);
                ci++;
            }
            x = ggml_add(ctx0, res, h);
            res = x;
            h = lfm2_rms_norm(ctx0, x, w.ffn_norm_w, 1e-5f);
            h = lfm2_swiglu_ffn(ctx0, h, w.ff_w1, w.ff_w2, w.ff_w3);
            x = ggml_add(ctx0, res, h);
        }
        x = lfm2_rms_norm(ctx0, x, model.lfm_embedding_norm_w, 1e-5f);
        ggml_tensor* last = ggml_view_1d(ctx0, x, hidden, (int64_t)(T_total - 1) * hidden * sizeof(float));
        ggml_tensor* lg = ggml_dup(ctx0, ggml_mul_mat(ctx0, model.lfm_embed_tokens_w, last));
        ggml_tensor* hd = ggml_dup(ctx0, last);
        ggml_set_name(lg, "lg");
        ggml_set_name(hd, "hd");
        ggml_build_forward_expand(gf, lg);
        ggml_build_forward_expand(gf, hd);
        ggml_graph_compute_with_ctx(ctx0, gf, ctx->n_threads);
        // Save conv states
        ci = 0;
        for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
            if (model.lfm_layers[il].is_attention)
                continue;
            char sn[16];
            snprintf(sn, sizeof(sn), "cs_%d", ci);
            ggml_tensor* s = ggml_graph_get_tensor(gf, sn);
            if (s) {
                int sc = (int)s->ne[1];
                auto& st = ctx->conv_states[ci];
                std::fill(st.begin(), st.end(), 0.0f);
                int off = (int)hp.lfm_conv_kernel - 1 - sc;
                if (off < 0)
                    off = 0;
                memcpy(st.data() + off * hidden, s->data, sizeof(float) * sc * hidden);
            }
            ci++;
        }
        ctx->kv_n_past = T_total;

        // Get initial logits + hidden
        std::vector<float> logits_v((int)lg->ne[0]);
        memcpy(logits_v.data(), lg->data, sizeof(float) * logits_v.size());
        std::vector<float> hidden_v(hidden);
        memcpy(hidden_v.data(), hd->data, sizeof(float) * hidden);
        ggml_free(ctx0);

        // Step 4: Interleaved decode (same loop as TTS synthesize)
        auto argmax = [](const std::vector<float>& v) {
            int b = 0;
            for (int i = 1; i < (int)v.size(); i++)
                if (v[i] > v[b])
                    b = i;
            return b;
        };
        int cur_token = argmax(logits_v);
        std::vector<float> cur_hidden = hidden_v;
        enum { MOD_TEXT, MOD_AUDIO };
        int cur_mod = MOD_TEXT, mod_left = n_text_budget;
        bool text_done = false;
        std::vector<std::vector<int32_t>> all_codes;
        std::string transcript;

        // Reuse the same backbone step1 lambda from synthesize — duplicate the pattern
        auto step1 = [&](const float* emb) -> std::pair<std::vector<float>, std::vector<float>> {
            ggml_init_params ip = {ctx->decode_meta.size(), ctx->decode_meta.data(), false};
            ggml_context* c = ggml_init(ip);
            if (!c)
                return {};
            ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, hidden, 1);
            memcpy(x->data, emb, sizeof(float) * hidden);
            ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
            *(int32_t*)pos->data = ctx->kv_n_past;
            ggml_cgraph* gf = ggml_new_graph_custom(c, 65536, false);
            int ai = 0, ci = 0;
            for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
                auto& w = model.lfm_layers[il];
                ggml_tensor* res = x;
                ggml_tensor* h = lfm2_rms_norm(c, x, w.operator_norm_w, 1e-5f);
                if (w.is_attention) {
                    core_attn::KvSelfAttnParams kvp = {};
                    kvp.head_dim = (int)hp.lfm_head_dim;
                    kvp.n_heads = (int)hp.lfm_n_heads;
                    kvp.n_kv_heads = (int)hp.lfm_n_kv_heads;
                    kvp.n_kv_grp = kvp.n_heads / kvp.n_kv_heads;
                    kvp.rope_type = GGML_ROPE_TYPE_NEOX;
                    kvp.rope_theta = hp.lfm_rope_theta;
                    kvp.attn_scale = 1.0f / sqrtf((float)hp.lfm_head_dim);
                    kvp.qk_norm_eps = 1e-5f;
                    kvp.gqa_mode = core_attn::GQA_NATIVE;
                    h = core_attn::kv_self_attn(c, gf, h, w.attn_q_proj_w, w.attn_k_proj_w, w.attn_v_proj_w,
                                                w.attn_out_proj_w, w.attn_q_ln_w, w.attn_k_ln_w, pos, nullptr,
                                                ctx->kv_k, ctx->kv_v, ai, ctx->kv_n_past, kvp);
                    ai++;
                } else {
                    const int K = (int)hp.lfm_conv_kernel;
                    ggml_tensor* bc = ggml_mul_mat(c, w.conv_in_proj_w, h);
                    ggml_tensor* Bp = ggml_view_1d(c, bc, hidden, 0);
                    ggml_tensor* Cp = ggml_view_1d(c, bc, hidden, hidden * sizeof(float));
                    ggml_tensor* xp = ggml_view_1d(c, bc, hidden, 2 * hidden * sizeof(float));
                    ggml_tensor* Bx = ggml_mul(c, ggml_cont(c, Bp), ggml_cont(c, xp));
                    ggml_tensor* cached = ggml_new_tensor_2d(c, GGML_TYPE_F32, hidden, K - 1);
                    memcpy(cached->data, ctx->conv_states[ci].data(), sizeof(float) * hidden * (K - 1));
                    ggml_tensor* Bxf = ggml_concat(c, cached, ggml_reshape_2d(c, Bx, hidden, 1), 1);
                    ggml_tensor* cw = ggml_cast(c, w.conv_conv_w, GGML_TYPE_F32);
                    ggml_tensor* cw4 = ggml_reshape_4d(c, cw, K, 1, 1, hidden);
                    ggml_tensor* Bt = ggml_cont(c, ggml_transpose(c, Bxf));
                    ggml_tensor* B4 = ggml_reshape_4d(c, Bt, K, 1, hidden, 1);
                    ggml_tensor* cr = ggml_conv_2d_dw_direct(c, cw4, B4, 1, 1, K - 1, 0, 1, 1);
                    cr = ggml_cont(c, ggml_permute(c, cr, 1, 2, 0, 3));
                    cr = ggml_reshape_2d(c, cr, hidden, (int)cr->ne[1]);
                    ggml_tensor* co = ggml_cont(c, ggml_view_2d(c, cr, hidden, 1, hidden * sizeof(float),
                                                                (int64_t)(K - 1) * hidden * sizeof(float)));
                    ggml_tensor* y = ggml_mul(c, ggml_reshape_2d(c, ggml_cont(c, Cp), hidden, 1), co);
                    h = ggml_mul_mat(c, w.conv_out_proj_w, y);
                    ggml_tensor* snap = ggml_dup(c, Bx);
                    char sn[16];
                    snprintf(sn, sizeof(sn), "cs_%d", ci);
                    ggml_set_name(snap, sn);
                    ggml_build_forward_expand(gf, snap);
                    ci++;
                }
                x = ggml_add(c, res, h);
                res = x;
                h = lfm2_rms_norm(c, x, w.ffn_norm_w, 1e-5f);
                h = lfm2_swiglu_ffn(c, h, w.ff_w1, w.ff_w2, w.ff_w3);
                x = ggml_add(c, res, h);
            }
            x = lfm2_rms_norm(c, x, model.lfm_embedding_norm_w, 1e-5f);
            ggml_tensor* lg = ggml_dup(c, ggml_mul_mat(c, model.lfm_embed_tokens_w, x));
            ggml_tensor* hd = ggml_dup(c, x);
            ggml_set_name(lg, "lg");
            ggml_set_name(hd, "hd");
            ggml_build_forward_expand(gf, lg);
            ggml_build_forward_expand(gf, hd);
            ggml_graph_compute_with_ctx(c, gf, ctx->n_threads);
            ci = 0;
            for (uint32_t il = 0; il < hp.lfm_n_layers; il++) {
                if (model.lfm_layers[il].is_attention)
                    continue;
                char sn[16];
                snprintf(sn, sizeof(sn), "cs_%d", ci);
                ggml_tensor* s = ggml_graph_get_tensor(gf, sn);
                if (s) {
                    auto& st = ctx->conv_states[ci];
                    memmove(st.data(), st.data() + hidden, sizeof(float) * hidden * ((int)hp.lfm_conv_kernel - 2));
                    memcpy(st.data() + hidden * ((int)hp.lfm_conv_kernel - 2), s->data, sizeof(float) * hidden);
                }
                ci++;
            }
            ctx->kv_n_past++;
            std::vector<float> rlg((int)lg->ne[0]);
            memcpy(rlg.data(), lg->data, sizeof(float) * rlg.size());
            std::vector<float> rhd(hidden);
            memcpy(rhd.data(), hd->data, sizeof(float) * hidden);
            ggml_free(c);
            return {rlg, rhd};
        };

        for (int step = 0; step < 1000; step++) {
            mod_left--;
            if (cur_mod == MOD_TEXT) {
                if (cur_token == kTokenImEnd || cur_token == 2)
                    break;
                if (cur_token == kTokenTextEnd)
                    text_done = true;
                if (mod_left <= 0 || text_done) {
                    cur_mod = MOD_AUDIO;
                    mod_left = n_audio_budget;
                }
                // Collect text for transcript
                std::string piece = decode_token(model, cur_token);
                transcript += piece;
                auto te = embed_text_fn({cur_token});
                if (te.empty())
                    break;
                auto [lg, hd] = step1(te.data());
                if (lg.empty())
                    break;
                cur_token = argmax(lg);
                cur_hidden = hd;
            } else {
                auto codes = lfm2_depthformer_sample_frame(ctx, cur_hidden.data());
                if (codes.empty())
                    break;
                if (codes[0] == 2048)
                    break;
                all_codes.push_back(codes);
                if (mod_left <= 0 && !text_done) {
                    cur_mod = MOD_TEXT;
                    mod_left = n_text_budget;
                }
                auto ae = lfm2_embed_audio_codes(ctx, codes);
                if (ae.empty())
                    break;
                auto [lg, hd] = step1(ae.data());
                if (lg.empty())
                    break;
                cur_token = argmax(lg);
                cur_hidden = hd;
            }
        }

        if (ctx->verbosity >= 1)
            fprintf(stderr, "lfm2-audio: S2S generated %zu audio frames, transcript: %s\n", all_codes.size(),
                    transcript.c_str());

        // Return transcript if requested
        if (out_text && !transcript.empty()) {
            *out_text = (char*)malloc(transcript.size() + 1);
            memcpy(*out_text, transcript.c_str(), transcript.size());
            (*out_text)[transcript.size()] = '\0';
        }

        // Decode audio codes → PCM
        if (all_codes.empty()) {
            if (out_n_samples)
                *out_n_samples = 0;
            return nullptr;
        }
        float* pcm = lfm2_detokenize(ctx, all_codes, out_n_samples);
        if (!pcm) {
            const int total = (int)all_codes.size() * 1920;
            pcm = (float*)calloc(total, sizeof(float));
            if (out_n_samples)
                *out_n_samples = total;
        }
        return pcm;
    }
}
