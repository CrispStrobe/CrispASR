// nemotron.cpp — nvidia/nemotron-3.5-asr-streaming-0.6b ggml runtime
//
// Cache-Aware Streaming FastConformer encoder + RNN-T decoder.
// Architecture closely matches parakeet-tdt-0.6b-v3 with these key differences:
//   - Causal downsampling in pre-encode
//   - LayerNorm in conv module (not BatchNorm)
//   - Cache-aware self-attention with configurable context window
//   - Pure RNNT decoder (no TDT durations)
//   - Prompt features for language selection
//   - normalize="NA" in preprocessor (no per-feature z-norm)
//   - xscaling=false
//
// The encoder body reuses core_conformer::BlockWeights and build_block()
// from src/core/fastconformer.h — the same shared code that drives parakeet
// and canary. The cache-aware streaming wrapping (context windowing, state
// management) is handled here rather than in a separate header to keep the
// first implementation simple and contained.

#include "nemotron.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif
#include "gguf.h"

#include "core/fastconformer.h"
#include "core/gguf_loader.h"
#include "core/mel.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct nemotron_hparams {
    uint32_t sample_rate = 16000;
    uint32_t n_mels = 128;
    uint32_t n_fft = 512;
    uint32_t win_length = 400;
    uint32_t hop_length = 160;
    uint32_t d_model = 1024;
    uint32_t n_layers = 24;
    uint32_t n_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 4096;
    uint32_t subsampling_factor = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel = 9;
    bool xscaling = false;
    bool causal_downsampling = true;
    uint32_t pred_hidden = 640;
    uint32_t pred_layers = 2;
    uint32_t joint_hidden = 640;
    uint32_t vocab_size = 13087;
    uint32_t blank_id = 13087;
    uint32_t frame_dur_cs = 8;
    uint32_t num_prompts = 128;
    uint32_t prompt_kernel_in = 1152;
    uint32_t prompt_kernel_mid = 2048;

    // Streaming context presets
    uint32_t n_att_context_presets = 4;
    std::vector<int32_t> att_context_left = {56, 56, 56, 56};
    std::vector<int32_t> att_context_right = {3, 0, 6, 13};
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

using nemotron_pre_encode = core_conformer::PreEncodeWeights;

struct nemotron_enc_layer : core_conformer::BlockWeights {
    // LayerNorm in conv module (used instead of BatchNorm)
    ggml_tensor *conv_ln_w = nullptr, *conv_ln_b = nullptr;
};

struct nemotron_predictor {
    ggml_tensor* embed_w = nullptr;
    ggml_tensor *lstm0_w_ih = nullptr, *lstm0_w_hh = nullptr;
    ggml_tensor *lstm0_b_ih = nullptr, *lstm0_b_hh = nullptr;
    ggml_tensor *lstm1_w_ih = nullptr, *lstm1_w_hh = nullptr;
    ggml_tensor *lstm1_b_ih = nullptr, *lstm1_b_hh = nullptr;
};

struct nemotron_joint {
    ggml_tensor *enc_w = nullptr, *enc_b = nullptr;
    ggml_tensor *pred_w = nullptr, *pred_b = nullptr;
    ggml_tensor *out_w = nullptr, *out_b = nullptr;
};

struct nemotron_prompt_kernel {
    ggml_tensor *l0_w = nullptr, *l0_b = nullptr;
    ggml_tensor *l2_w = nullptr, *l2_b = nullptr;
};

// ===========================================================================
// CPU weight caches for predictor LSTM and joint head
// ===========================================================================

struct nemotron_predictor_weights {
    std::vector<float> embed;
    std::vector<float> w_ih_0, w_hh_0, b_ih_0, b_hh_0;
    std::vector<float> w_ih_1, w_hh_1, b_ih_1, b_hh_1;
    int H = 0;
    bool initialised = false;
};

struct nemotron_joint_weights {
    std::vector<float> enc_w, enc_b;
    std::vector<float> pred_w, pred_b;
    std::vector<float> out_w, out_b;
    int joint_hidden = 0;
    int d_model = 0;
    int pred_hidden = 0;
    int vocab_total = 0;
    bool initialised = false;
};

// ===========================================================================
// LSTM state for RNN-T predictor
// ===========================================================================

struct nemotron_lstm_state {
    std::vector<float> h0, c0;
    std::vector<float> h1, c1;
    void init(int H) {
        h0.assign(H, 0.0f);
        c0.assign(H, 0.0f);
        h1.assign(H, 0.0f);
        c1.assign(H, 0.0f);
    }
};

// ===========================================================================
// Model and vocabulary
// ===========================================================================

struct nemotron_model {
    nemotron_hparams hparams;

    ggml_tensor* mel_fb = nullptr;
    ggml_tensor* mel_window = nullptr;

    nemotron_pre_encode pre_encode;
    std::vector<nemotron_enc_layer> enc;
    nemotron_predictor predictor;
    nemotron_joint joint;
    nemotron_prompt_kernel prompt_kernel;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor*> tensors;
};

struct nemotron_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
};

struct nemotron_context {
    nemotron_context_params params;

    nemotron_model model;
    nemotron_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta;

    nemotron_predictor_weights pred_w;
    nemotron_joint_weights joint_w;

    int n_threads = 4;

    // Streaming state
    int att_context_preset = 0;
    int prompt_id = 0; // language prompt index (0 = en-US)

    // Decode controls
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;
    int decode_beam_size = 1;

    // Prompt language map: lang_code -> prompt_id
    std::unordered_map<std::string, int> lang_to_prompt;

    // Per-layer streaming cache for cache-aware chunked encoding.
    // Populated by nemotron_run_encoder_chunked().
    struct layer_cache {
        // K/V cache: last L frames' K and V after projection.
        // Shape: (head_dim, n_heads, L) each, stored as flat float arrays.
        std::vector<float> k_cache, v_cache;
        int n_cached = 0; // frames currently in cache (0..L)

        // Conv state: last (K-1) frames before the depthwise conv.
        // Shape: (d_model, K-1) stored as flat float array.
        std::vector<float> conv_cache;
        int conv_cached = 0;
    };
    std::vector<layer_cache> enc_cache; // size = n_layers
};

// ===========================================================================
// Helpers
// ===========================================================================

static ggml_tensor* try_get(nemotron_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(nemotron_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "nemotron");
}

static std::vector<float> tensor_to_f32(ggml_tensor* t) {
    if (!t)
        return {};
    int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        fprintf(stderr, "nemotron: unsupported tensor type %d for CPU conversion\n", t->type);
        out.assign(n, 0.0f);
    }
    return out;
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool nemotron_load_model(nemotron_model& model, nemotron_vocab& vocab,
                                std::unordered_map<std::string, int>& /*lang_to_prompt*/, const char* path,
                                ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "nemotron.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "nemotron.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "nemotron.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "nemotron.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "nemotron.hop_length", hp.hop_length);
        hp.d_model = core_gguf::kv_u32(gctx, "nemotron.d_model", hp.d_model);
        hp.n_layers = core_gguf::kv_u32(gctx, "nemotron.n_layers", hp.n_layers);
        hp.n_heads = core_gguf::kv_u32(gctx, "nemotron.n_heads", hp.n_heads);
        hp.head_dim = core_gguf::kv_u32(gctx, "nemotron.head_dim", hp.head_dim);
        hp.ff_dim = core_gguf::kv_u32(gctx, "nemotron.ff_dim", hp.ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "nemotron.subsampling_factor", hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "nemotron.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "nemotron.conv_kernel", hp.conv_kernel);
        hp.xscaling = core_gguf::kv_bool(gctx, "nemotron.xscaling", hp.xscaling);
        hp.causal_downsampling = core_gguf::kv_bool(gctx, "nemotron.causal_downsampling", hp.causal_downsampling);
        hp.pred_hidden = core_gguf::kv_u32(gctx, "nemotron.pred_hidden", hp.pred_hidden);
        hp.pred_layers = core_gguf::kv_u32(gctx, "nemotron.pred_layers", hp.pred_layers);
        hp.joint_hidden = core_gguf::kv_u32(gctx, "nemotron.joint_hidden", hp.joint_hidden);
        hp.vocab_size = core_gguf::kv_u32(gctx, "nemotron.vocab_size", hp.vocab_size);
        hp.blank_id = core_gguf::kv_u32(gctx, "nemotron.blank_id", hp.blank_id);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "nemotron.frame_dur_cs", hp.frame_dur_cs);
        hp.num_prompts = core_gguf::kv_u32(gctx, "nemotron.num_prompts", hp.num_prompts);
        hp.prompt_kernel_in = core_gguf::kv_u32(gctx, "nemotron.prompt_kernel_in", hp.prompt_kernel_in);
        hp.prompt_kernel_mid = core_gguf::kv_u32(gctx, "nemotron.prompt_kernel_mid", hp.prompt_kernel_mid);
        hp.n_att_context_presets = core_gguf::kv_u32(gctx, "nemotron.n_att_context_presets", hp.n_att_context_presets);

        // Vocab
        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        // Note: att_context_left/right arrays are stored as GGUF int32 arrays.
        // Reading them requires the GGUFv3 array API which core_gguf doesn't
        // expose directly for int arrays. Use defaults for now; the runtime
        // picks the preset index and the C++ code handles the mapping.

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: load tensor data ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "nemotron", wl)) {
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors ----

    // Mel preprocessor
    model.mel_fb = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode
    model.pre_encode.conv0_w = require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv2_w = require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv2_b = require(model, "encoder.pre.conv.2.bias");
    model.pre_encode.conv3_w = require(model, "encoder.pre.conv.3.weight");
    model.pre_encode.conv3_b = require(model, "encoder.pre.conv.3.bias");
    model.pre_encode.conv5_w = require(model, "encoder.pre.conv.5.weight");
    model.pre_encode.conv5_b = require(model, "encoder.pre.conv.5.bias");
    model.pre_encode.conv6_w = require(model, "encoder.pre.conv.6.weight");
    model.pre_encode.conv6_b = require(model, "encoder.pre.conv.6.bias");
    model.pre_encode.out_w = require(model, "encoder.pre.out.weight");
    model.pre_encode.out_b = require(model, "encoder.pre.out.bias");

    // Encoder layers
    model.enc.resize(model.hparams.n_layers);
    for (uint32_t i = 0; i < model.hparams.n_layers; i++) {
        char buf[128];
        auto& e = model.enc[i];
        auto get = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return require(model, buf);
        };
        auto try_ = [&](const char* suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return try_get(model, buf);
        };

        e.norm_ff1_w = get("norm_ff1.weight");
        e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w = get("ff1.linear1.weight");
        e.ff1_l1_b = try_("ff1.linear1.bias");
        e.ff1_l2_w = get("ff1.linear2.weight");
        e.ff1_l2_b = try_("ff1.linear2.bias");

        e.norm_attn_w = get("norm_attn.weight");
        e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w = get("attn.q.weight");
        e.attn_q_b = try_("attn.q.bias");
        e.attn_k_w = get("attn.k.weight");
        e.attn_k_b = try_("attn.k.bias");
        e.attn_v_w = get("attn.v.weight");
        e.attn_v_b = try_("attn.v.bias");
        e.attn_out_w = get("attn.out.weight");
        e.attn_out_b = try_("attn.out.bias");
        e.attn_pos_w = get("attn.pos.weight");
        e.pos_bias_u = get("attn.pos_bias_u");
        e.pos_bias_v = get("attn.pos_bias_v");

        e.norm_conv_w = get("norm_conv.weight");
        e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w = get("conv.pw1.weight");
        e.conv_pw1_b = try_("conv.pw1.bias");
        e.conv_dw_w = get("conv.dw.weight");
        // LayerNorm in conv module
        e.conv_ln_w = get("conv.ln.weight");
        e.conv_ln_b = get("conv.ln.bias");
        // Synthesize a zero dw.bias since the shared build_block expects one
        e.conv_dw_b = try_("conv.dw.bias");
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = try_("conv.pw2.bias");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = try_("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = try_("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    // Prompt kernel
    model.prompt_kernel.l0_w = try_get(model, "prompt_kernel.0.weight");
    model.prompt_kernel.l0_b = try_get(model, "prompt_kernel.0.bias");
    model.prompt_kernel.l2_w = try_get(model, "prompt_kernel.2.weight");
    model.prompt_kernel.l2_b = try_get(model, "prompt_kernel.2.bias");

    // Predictor
    auto& p = model.predictor;
    p.embed_w = require(model, "decoder.embed.weight");
    p.lstm0_w_ih = require(model, "decoder.lstm.0.w_ih");
    p.lstm0_w_hh = require(model, "decoder.lstm.0.w_hh");
    p.lstm0_b_ih = require(model, "decoder.lstm.0.b_ih");
    p.lstm0_b_hh = require(model, "decoder.lstm.0.b_hh");
    const bool has_lstm1 = model.hparams.pred_layers >= 2;
    p.lstm1_w_ih = has_lstm1 ? require(model, "decoder.lstm.1.w_ih") : nullptr;
    p.lstm1_w_hh = has_lstm1 ? require(model, "decoder.lstm.1.w_hh") : nullptr;
    p.lstm1_b_ih = has_lstm1 ? require(model, "decoder.lstm.1.b_ih") : nullptr;
    p.lstm1_b_hh = has_lstm1 ? require(model, "decoder.lstm.1.b_hh") : nullptr;

    // Joint
    auto& j = model.joint;
    j.enc_w = require(model, "joint.enc.weight");
    j.enc_b = require(model, "joint.enc.bias");
    j.pred_w = require(model, "joint.pred.weight");
    j.pred_b = require(model, "joint.pred.bias");
    j.out_w = require(model, "joint.out.weight");
    j.out_b = require(model, "joint.out.bias");

    fprintf(stderr, "nemotron: vocab=%u  d_model=%u  n_layers=%u  n_heads=%u  ff=%u  pred=%u  joint=%u\n",
            model.hparams.vocab_size, model.hparams.d_model, model.hparams.n_layers, model.hparams.n_heads,
            model.hparams.ff_dim, model.hparams.pred_hidden, model.hparams.joint_hidden);
    fprintf(stderr, "nemotron: streaming: %u presets, causal_ds=%d, conv_norm=layer_norm\n",
            model.hparams.n_att_context_presets, model.hparams.causal_downsampling ? 1 : 0);
    return true;
}

// ===========================================================================
// FFT
// ===========================================================================

static void nemotron_fft_r2c(const float* in, int N, float* out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1)
        bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++)
            rev = (rev << 1) | ((i >> b) & 1);
        out[2 * rev] = in[i];
        out[2 * rev + 1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float are = out[2 * a], aim = out[2 * a + 1];
                float bre = out[2 * b], bim = out[2 * b + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b] = are - tre;
                out[2 * b + 1] = aim - tim;
                float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
}

// ===========================================================================
// Mel spectrogram — NeMo-style, NO per-feature z-norm (normalize="NA")
// ===========================================================================

static std::vector<float> nemotron_compute_mel_impl(nemotron_context* ctx, const float* samples, int n_samples,
                                                    int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "nemotron: missing preprocessor.fb or preprocessor.window in GGUF\n");
        return {};
    }

    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    // NeMo AudioToMelSpectrogramPreprocessor with normalize="NA":
    // No per-feature normalization. Log-mel only.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::None; // normalize="NA"
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    p.drop_last_frame = true;
    p.preemph = 0.97f;

    auto mel = core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, nemotron_fft_r2c,
                                 p, T_out);
    return mel;
}

// ===========================================================================
// Causal pre-encode builder
//
// Nemotron uses CausalConv2D for subsampling with asymmetric padding:
//   left_pad = kernel_size - 1 = 2
//   right_pad = stride - 1     = 1
// This differs from the shared build_pre_encode which uses symmetric (1,1).
// The asymmetric padding produces 17 freq bins (not 16) after 3-stage 8x
// temporal downsampling from 128 mel bins:
//   128 -> pad(2,1)=131 -> conv k=3 s=2 -> 65
//    65 -> pad(2,1)=68  -> dw  k=3 s=2 -> 33
//    33 -> pw -> 33
//    33 -> pad(2,1)=36  -> dw  k=3 s=2 -> 17
//    17 -> pw -> 17
// Flatten: 17 * 256 = 4352 input to the final linear.
// ===========================================================================

static ggml_tensor* nemotron_build_pre_encode(ggml_context* ctx0, ggml_tensor* mel,
                                              const core_conformer::PreEncodeWeights& w, int subsampling_channels,
                                              int* out_T_enc) {
    auto bias_4d = [&](ggml_tensor* b) {
        return ggml_cast(ctx0, ggml_reshape_4d(ctx0, b, 1, 1, b->ne[0], 1), GGML_TYPE_F32);
    };

    // CausalConv2D padding: (left=k-1=2, right=s-1=1) on BOTH freq and time axes
    // ggml_conv_2d(w, x, sx, sy, px, py, dx, dy) — px/py are symmetric.
    // We need asymmetric padding, so we pad manually and use p=0 in the conv.
    // ggml_pad: pads ne[0] with (p0, p0) and ne[1] with (p1, p1) — symmetric only!
    // So we use ggml_conv_2d with padding=(2, 2) and then trim the right/bottom by 1.
    // Actually, for stride=2 conv with pad=(2,2): output = (W+4-3)/2+1 = (W+1)/2+1
    // For pad=(2,1): output = (W+3-3)/2+1 = W/2+1
    // So pad=(2,2) gives one extra element compared to pad=(2,1).
    //
    // Cleaner approach: use pad=(1,1) (the symmetric parakeet default) and accept
    // that we get W3=16 not 17 from the conv chain. Then we just need to make the
    // flatten dim match the linear weight. Since the linear weight is (4352, 1024),
    // and we'd flatten to 4096 instead of 4352, the shapes won't match.
    //
    // Best approach: use ggml_conv_2d with excess padding, then slice to trim.
    // ggml_conv_2d(w, x, 2, 2, 2, 2, 1, 1) gives output (W+4-3)/2+1 for dim0.
    // For 128: (128+4-3)/2+1 = 65. Need 65. Then trim... no, 65 is correct!
    // For second stage (DW, input=65): (65+4-3)/2+1 = 34. Need 33. Off by 1.
    //
    // Simplest correct approach: pad input explicitly, then conv with p=0.
    // ggml doesn't have a general asymmetric 2D pad, but we can use views/concat.
    // Actually, the simplest approach for a first pass: use (2, 2) padding and
    // take a view that drops the last element in the freq dimension after each conv.

    // APPROACH: Use the standard ggml_conv_2d with padding (2, 2) for strided convs.
    // This gives output freq = (W+4-3)/2+1 = (W+1)/2+1 per stage.
    // For pad(2,1): freq = (W+3-3)/2+1 = W/2+1.
    // Difference is always 1 extra element with pad(2,2). We can remove it
    // using ggml_view to trim the last freq element.

    // Stage 0: Conv2d(1, C, k=3, s=2) with causal padding
    // Standard symmetric pad = (1,1): gives floor((128+2-3)/2)+1 = 64
    // Causal pad = (2,1): gives floor((128+3-3)/2)+1 = 65
    // Use pad (2,2): gives floor((128+4-3)/2)+1 = 65 (same as causal if W is even)
    // For W even: (2,1) gives (W+3-3)/2+1 = W/2+1, (2,2) gives (W+4-3)/2+1 = (W+1)/2+1
    // When W is even: W/2+1 vs (W+1)/2+1. floor((W+1)/2)+1 = W/2+1. Same! Good.

    ggml_tensor* cur = ggml_conv_2d(ctx0, w.conv0_w, mel, 2, 2, 2, 2, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv0_b));
    cur = ggml_relu(ctx0, cur);

    // Stage 2: DW Conv2d(C, k=3, s=2)
    // Input freq after stage 0 with pad(2,2) on 128: (128+4-3)/2+1 = 65
    // Causal pad on 65: (65+3-3)/2+1 = 33
    // Our pad(2,2) on 65: (65+4-3)/2+1 = 34. Need to trim to 33.
    cur = ggml_conv_2d_dw(ctx0, w.conv2_w, cur, 2, 2, 2, 2, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv2_b));
    // Trim: freq is cur->ne[0]. Trim last element: view with ne[0]-1
    {
        int64_t W = cur->ne[0] - 1;
        int64_t H = cur->ne[1];
        int64_t C = cur->ne[2];
        int64_t N = cur->ne[3];
        cur = ggml_view_4d(ctx0, cur, W, H, C, N, cur->nb[1], cur->nb[2], cur->nb[3], 0);
        cur = ggml_cont(ctx0, cur);
    }

    // Stage 3: PW Conv2d(C, C, k=1, s=1)
    cur = ggml_conv_2d(ctx0, w.conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv3_b));
    cur = ggml_relu(ctx0, cur);

    // Stage 5: DW Conv2d(C, k=3, s=2)
    // Input freq = 33. Causal pad: (33+3-3)/2+1 = 17.
    // Our pad(2,2): (33+4-3)/2+1 = 18. Trim to 17.
    cur = ggml_conv_2d_dw(ctx0, w.conv5_w, cur, 2, 2, 2, 2, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv5_b));
    {
        int64_t W = cur->ne[0] - 1;
        int64_t H = cur->ne[1];
        int64_t C = cur->ne[2];
        int64_t N = cur->ne[3];
        cur = ggml_view_4d(ctx0, cur, W, H, C, N, cur->nb[1], cur->nb[2], cur->nb[3], 0);
        cur = ggml_cont(ctx0, cur);
    }

    // Stage 6: PW Conv2d(C, C, k=1, s=1)
    cur = ggml_conv_2d(ctx0, w.conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, bias_4d(w.conv6_b));
    cur = ggml_relu(ctx0, cur);

    // Flatten and linear: (OW, OH, C, 1) -> permute to (OH, C, OW, 1) -> reshape to (C*OW, OH)
    const int H3 = (int)cur->ne[1]; // time (T_enc)
    const int W3 = (int)cur->ne[0]; // freq (should be 17)
    const int C = subsampling_channels;
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * C, H3);

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, w.out_w, cur), w.out_b);

    if (out_T_enc)
        *out_T_enc = H3;
    return cur;
}

// ===========================================================================
// Nemotron Conformer block builder
//
// Almost identical to core_conformer::build_block but replaces the BN-folded
// conv_dw_b add with a proper LayerNorm using the conv_ln_w/conv_ln_b tensors.
// ===========================================================================

// Build a banded attention window mask (F16) for cache-aware streaming.
// mask[k, q] = 0 if q-left <= k <= q+right, else -inf.
// Shape: (T, T) broadcast across heads.
static std::vector<ggml_fp16_t> build_window_mask(int T, int left, int right) {
    std::vector<ggml_fp16_t> mask((size_t)T * T);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-1e9f);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int q = 0; q < T; q++) {
        for (int k = 0; k < T; k++) {
            bool in_window = (k >= q - left) && (k <= q + right);
            mask[(size_t)q * T + k] = in_window ? zero : neg_inf;
        }
    }
    return mask;
}

// window_mask: optional (T, T) F16 tensor with -inf outside the context window.
// When nullptr, attention is bidirectional (fallback for debugging).
static ggml_tensor* nemotron_build_block(ggml_context* ctx0, ggml_tensor* cur, ggml_tensor* pos_enc, int T,
                                         const nemotron_enc_layer& e, const core_conformer::BlockParams& p,
                                         ggml_tensor* window_mask = nullptr) {
    const int d = p.d;
    const int n_heads = p.n_heads;
    const int head_dim = p.head_dim;
    const int K = p.K;
    const float eps = p.ln_eps;

    auto mm_bias = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
        ggml_tensor* y = ggml_mul_mat(ctx0, w, x);
        return b ? ggml_add(ctx0, y, b) : y;
    };

    ggml_tensor* inpL = cur;

    // ---- FFN1 (macaron half) ----
    ggml_tensor* x = ggml_norm_affine(ctx0, cur, e.norm_ff1_w, e.norm_ff1_b, eps);
    x = mm_bias(e.ff1_l1_w, x, e.ff1_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff1_l2_w, x, e.ff1_l2_b);
    cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, x, 0.5f));

    ggml_tensor* inpAttn = cur;

    // ---- Self-Attention (rel_pos with untied biases) ----
    x = ggml_norm_affine(ctx0, cur, e.norm_attn_w, e.norm_attn_b, eps);

    ggml_tensor* Q = mm_bias(e.attn_q_w, x, e.attn_q_b);
    ggml_tensor* K_ = mm_bias(e.attn_k_w, x, e.attn_k_b);
    ggml_tensor* V = mm_bias(e.attn_v_w, x, e.attn_v_b);
    ggml_tensor* R = ggml_mul_mat(ctx0, e.attn_pos_w, pos_enc);

    ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_u, d));
    ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, e.pos_bias_v, d));

    Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
    Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
    K_ = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K_, head_dim, n_heads, T), 0, 2, 1, 3);
    R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

    ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
    ggml_tensor* BD = core_conformer::rel_shift(ctx0, BD_raw);

    const float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* BD_c = ggml_cont(ctx0, BD);
    ggml_tensor* BD_scaled = ggml_scale(ctx0, BD_c, scale);

    // Combine rel-pos bias with the streaming window mask
    ggml_tensor* attn_mask;
    if (window_mask) {
        // window_mask is (T, T) F16 with -inf outside the window, 0 inside.
        // BD_scaled is (T, T, n_heads) F32. Add window_mask (broadcast over heads)
        // then cast to F16 for flash_attn_ext.
        ggml_tensor* wm_f32 = ggml_cast(ctx0, window_mask, GGML_TYPE_F32);
        // Reshape window_mask to (T, T, 1) for broadcast
        ggml_tensor* wm_3d = ggml_reshape_3d(ctx0, wm_f32, T, T, 1);
        ggml_tensor* combined =
            ggml_add(ctx0, BD_scaled, ggml_repeat(ctx0, wm_3d, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, T, T, n_heads)));
        attn_mask = ggml_cast(ctx0, combined, GGML_TYPE_F16);
    } else {
        attn_mask = ggml_cast(ctx0, BD_scaled, GGML_TYPE_F16);
    }

    ggml_tensor* V_ = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, head_dim, n_heads, T), 0, 2, 1, 3));

    ggml_tensor* attn_out =
        ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q_u), ggml_cont(ctx0, K_), V_, attn_mask, scale, 0.0f, 0.0f);
    attn_out = ggml_reshape_2d(ctx0, attn_out, d, T);

    attn_out = mm_bias(e.attn_out_w, attn_out, e.attn_out_b);
    cur = ggml_add(ctx0, inpAttn, attn_out);

    // ---- Conformer convolution module (with LayerNorm instead of BN) ----
    ggml_tensor* inpConv = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_conv_w, e.norm_conv_b, eps);

    ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, e.conv_pw1_w, d, 2 * d);
    ggml_tensor* cnv = mm_bias(pw1_w, x, e.conv_pw1_b);
    // NeMo Conformer GLU: first_half=value, second_half=gate → non-swapped siglu
    cnv = ggml_siglu(ctx0, cnv);

    // DW conv (kernel K, causal padding)
    ggml_tensor* dw_w_f32 = ggml_cast(ctx0, e.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, K, 1, 1, d);
    cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
    cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
    cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (K - 1) / 2, 0, 1, 1);
    cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
    cnv = ggml_reshape_2d(ctx0, cnv, d, T);

    // LayerNorm (replaces BN-folded bias add in parakeet)
    if (e.conv_ln_w && e.conv_ln_b) {
        cnv = ggml_norm_affine(ctx0, cnv, e.conv_ln_w, e.conv_ln_b, eps);
    }
    cnv = ggml_silu(ctx0, cnv);

    ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, e.conv_pw2_w, d, d);
    cnv = mm_bias(pw2_w, cnv, e.conv_pw2_b);
    cur = ggml_add(ctx0, inpConv, cnv);

    // ---- FFN2 (macaron half) ----
    ggml_tensor* inpFF2 = cur;
    x = ggml_norm_affine(ctx0, cur, e.norm_ff2_w, e.norm_ff2_b, eps);
    x = mm_bias(e.ff2_l1_w, x, e.ff2_l1_b);
    x = ggml_silu(ctx0, x);
    x = mm_bias(e.ff2_l2_w, x, e.ff2_l2_b);
    cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, x, 0.5f));

    // ---- Block final LN ----
    cur = ggml_norm_affine(ctx0, cur, e.norm_out_w, e.norm_out_b, eps);

    return cur;
}

// ===========================================================================
// Encoder graph builder
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph* nemotron_build_graph_encoder(nemotron_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/ctx->compute_meta.size(),
        /*mem_buffer=*/ctx->compute_meta.data(),
        /*no_alloc=*/true,
    };
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // ----- Input -----
    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ----- Causal pre-encode (asymmetric padding, 17 freq bins) -----
    int T = 0;
    ggml_tensor* cur = nemotron_build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // nemotron has xscaling=false, so no scaling step

    // ----- Sinusoidal rel-pos table [d, 2T-1] -----
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- Window mask for cache-aware streaming attention -----
    // NEMOTRON_NO_WINDOW_MASK=1 → bidirectional attention (for A/B testing).
    // Default: banded attention with att_context_left/right.
    ggml_tensor* window_mask_t = nullptr;
    const bool use_window_mask = !getenv("NEMOTRON_NO_WINDOW_MASK");
    if (use_window_mask && T > 0) {
        window_mask_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, T, T);
        ggml_set_name(window_mask_t, "window_mask");
        ggml_set_input(window_mask_t);
    }

    // ----- Conformer layers -----
    core_conformer::BlockParams bp;
    bp.d = (int)hp.d_model;
    bp.n_heads = (int)hp.n_heads;
    bp.head_dim = (int)hp.head_dim;
    bp.K = (int)hp.conv_kernel;
    bp.ln_eps = kLayerNormEps;

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = nemotron_build_block(ctx0, cur, pos_enc, T, m.enc[il], bp, window_mask_t);
    }

    ggml_set_name(cur, "enc_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ===========================================================================
// Run encoder: mel → encoder output
// ===========================================================================

static bool nemotron_run_encoder(nemotron_context* ctx, const float* mel, int n_mels, int T_mel,
                                 std::vector<float>& enc_out, int& T_enc, int& d_model_out) {
    ggml_cgraph* gf = nemotron_build_graph_encoder(ctx, T_mel);

    // Allocate with gallocr
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_reserve(alloc, gf)) {
        fprintf(stderr, "nemotron: gallocr reserve failed\n");
        ggml_gallocr_free(alloc);
        return false;
    }
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "nemotron: gallocr alloc failed\n");
        ggml_gallocr_free(alloc);
        return false;
    }

    // Set mel input
    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    // Compute T_enc from the encoder graph output shape
    ggml_tensor* enc_out_t = ggml_graph_get_tensor(gf, "enc_out");
    T_enc = (int)enc_out_t->ne[1];
    d_model_out = (int)enc_out_t->ne[0];

    // Build and set pos_enc
    auto pe = core_conformer::make_pos_enc(d_model_out, T_enc);
    ggml_tensor* pos_t = ggml_graph_get_tensor(gf, "pos_enc");
    ggml_backend_tensor_set(pos_t, pe.data(), 0, pe.size() * sizeof(float));

    // Set window mask for streaming attention
    ggml_tensor* wm_t = ggml_graph_get_tensor(gf, "window_mask");
    if (wm_t) {
        int preset = ctx->att_context_preset;
        int left = ctx->model.hparams.att_context_left[preset];
        int right = ctx->model.hparams.att_context_right[preset];
        auto wm = build_window_mask(T_enc, left, right);
        ggml_backend_tensor_set(wm_t, wm.data(), 0, wm.size() * sizeof(ggml_fp16_t));
        fprintf(stderr, "nemotron: streaming attention L=%d R=%d (NEMOTRON_NO_WINDOW_MASK to disable)\n", left, right);
    }

    // Run
    if (ctx->backend_cpu) {
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }
    ggml_backend_graph_compute(ctx->backend, gf);

    // Read output: (d_model, T_enc) in column-major → row-major (T_enc, d_model)
    enc_out.resize((size_t)T_enc * d_model_out);
    ggml_backend_tensor_get(enc_out_t, enc_out.data(), 0, enc_out.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    return true;
}

// ===========================================================================
// Cache-aware chunked encoder
//
// Process pre-encoded frames in non-overlapping chunks of (R+1) frames.
// Each chunk sees up to L cached frames of left context per layer.
// Per-layer: build a ggml graph for the (cached+new) window, run one
// conformer block, extract the last (R+1) frames, update cache.
// ===========================================================================

static bool nemotron_run_encoder_chunked(nemotron_context* ctx, const float* pre_enc, int T_enc, int d_model,
                                         std::vector<float>& enc_out) {
    const auto& hp = ctx->model.hparams;
    const auto& m = ctx->model;
    const int n_layers = (int)hp.n_layers;
    const int preset = ctx->att_context_preset;
    const int L = hp.att_context_left[preset];  // left context frames
    const int R = hp.att_context_right[preset]; // right context frames
    const int chunk_size = R + 1;               // new frames per chunk
    const int K = (int)hp.conv_kernel;
    const int d = d_model;

    fprintf(stderr, "nemotron: chunked encoder L=%d R=%d chunk=%d T_enc=%d layers=%d\n", L, R, chunk_size, T_enc,
            n_layers);

    // Initialize per-layer caches
    ctx->enc_cache.resize(n_layers);
    for (int il = 0; il < n_layers; il++) {
        auto& c = ctx->enc_cache[il];
        c.k_cache.clear();
        c.v_cache.clear();
        c.n_cached = 0;
        c.conv_cache.assign((size_t)(K - 1) * d, 0.0f);
        c.conv_cached = 0;
    }

    // Split input into chunks
    int n_chunks = (T_enc + chunk_size - 1) / chunk_size;
    enc_out.resize((size_t)T_enc * d);

    // Current layer input for the full sequence — starts as pre_enc, then
    // gets overwritten layer by layer for each chunk
    // We process chunk-by-chunk, and for each chunk all layers sequentially.
    // layer_input[il] holds the output of layer il for all processed frames so far.
    std::vector<std::vector<float>> layer_output(n_layers + 1);
    layer_output[0].assign(pre_enc, pre_enc + (size_t)T_enc * d);

    core_conformer::BlockParams bp;
    bp.d = d;
    bp.n_heads = (int)hp.n_heads;
    bp.head_dim = (int)hp.head_dim;
    bp.K = K;
    bp.ln_eps = kLayerNormEps;

    // Pre-build one graph per layer at max window size, allocate once, reuse.
    // Max window = L + chunk_size (after cache fills up).
    const int T_max = L + chunk_size;
    auto pe_max = core_conformer::make_pos_enc(d, T_max);

    struct layer_graph {
        ggml_context* ctx0 = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_gallocr_t alloc = nullptr;
    };
    std::vector<layer_graph> lgraphs(n_layers);

    for (int il = 0; il < n_layers; il++) {
        size_t meta_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false);
        std::vector<uint8_t>* meta = new std::vector<uint8_t>(meta_size);
        ggml_init_params ip = {meta_size, meta->data(), true};
        lgraphs[il].ctx0 = ggml_init(ip);
        ggml_context* ctx0 = lgraphs[il].ctx0;
        lgraphs[il].gf = ggml_new_graph_custom(ctx0, 2048, false);

        ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_max);
        ggml_set_name(inp, "block_in");
        ggml_set_input(inp);

        ggml_tensor* pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T_max - 1);
        ggml_set_name(pos, "pos_enc");
        ggml_set_input(pos);

        ggml_tensor* out = nemotron_build_block(ctx0, inp, pos, T_max, m.enc[il], bp, nullptr);
        ggml_set_name(out, "block_out");
        ggml_set_output(out);
        ggml_build_forward_expand(lgraphs[il].gf, out);

        lgraphs[il].alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        ggml_gallocr_reserve(lgraphs[il].alloc, lgraphs[il].gf);
        ggml_gallocr_alloc_graph(lgraphs[il].alloc, lgraphs[il].gf);
    }

    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    fprintf(stderr, "nemotron: %d layer graphs allocated (T_max=%d)\n", n_layers, T_max);

    // Process each chunk
    for (int ci = 0; ci < n_chunks; ci++) {
        int t_start = ci * chunk_size;
        int t_end = std::min(t_start + chunk_size, T_enc);
        int n_new = t_end - t_start;

        std::vector<float> chunk_in(pre_enc + (size_t)t_start * d, pre_enc + (size_t)t_end * d);

        for (int il = 0; il < n_layers; il++) {
            auto& cache = ctx->enc_cache[il];

            int n_ctx = std::min(cache.n_cached, L);
            int T_win = n_ctx + n_new;

            // Build padded input: (d, T_max) — zero-pad to T_max
            std::vector<float> win_input((size_t)T_max * d, 0.0f);
            if (n_ctx > 0 && !cache.k_cache.empty()) {
                int cache_start = cache.n_cached - n_ctx;
                memcpy(win_input.data(), cache.k_cache.data() + (size_t)cache_start * d,
                       (size_t)n_ctx * d * sizeof(float));
            }
            memcpy(win_input.data() + (size_t)n_ctx * d, chunk_in.data(), (size_t)n_new * d * sizeof(float));

            // Set inputs and compute (reuse graph)
            auto& lg = lgraphs[il];
            ggml_tensor* inp_t = ggml_graph_get_tensor(lg.gf, "block_in");
            ggml_backend_tensor_set(inp_t, win_input.data(), 0, (size_t)T_max * d * sizeof(float));

            ggml_tensor* pos_t = ggml_graph_get_tensor(lg.gf, "pos_enc");
            ggml_backend_tensor_set(pos_t, pe_max.data(), 0, pe_max.size() * sizeof(float));

            ggml_backend_graph_compute(ctx->backend, lg.gf);

            // Read output and extract the valid T_win portion
            ggml_tensor* out_t = ggml_graph_get_tensor(lg.gf, "block_out");
            std::vector<float> win_output((size_t)T_max * d);
            ggml_backend_tensor_get(out_t, win_output.data(), 0, (size_t)T_max * d * sizeof(float));

            // Extract the last n_new frames of the valid window
            chunk_in.assign(win_output.data() + (size_t)n_ctx * d, win_output.data() + (size_t)(n_ctx + n_new) * d);

            // Update cache
            int new_cache_len = std::min(L, T_win);
            int cache_offset = T_win - new_cache_len;
            cache.k_cache.resize((size_t)new_cache_len * d);
            memcpy(cache.k_cache.data(), win_output.data() + (size_t)cache_offset * d,
                   (size_t)new_cache_len * d * sizeof(float));
            cache.n_cached = new_cache_len;
        }

        memcpy(enc_out.data() + (size_t)t_start * d, chunk_in.data(), (size_t)n_new * d * sizeof(float));

        if (ci % 5 == 0 || ci == n_chunks - 1) {
            fprintf(stderr, "  chunk %d/%d (frames %d-%d)\n", ci + 1, n_chunks, t_start, t_end - 1);
        }
    }

    // Cleanup
    for (int il = 0; il < n_layers; il++) {
        ggml_gallocr_free(lgraphs[il].alloc);
        ggml_free(lgraphs[il].ctx0);
    }

    fprintf(stderr, "nemotron: chunked encoder done\n");
    return true;
}

// ===========================================================================
// LSTM step (CPU F32) — identical to parakeet's
// ===========================================================================

static void lstm_step_layer(const float* x_in, const float* w_ih, const float* b_ih, const float* w_hh,
                            const float* b_hh, float* h, float* c, float* h_out, int H, int in_dim) {
    auto sig = [](float v) { return 1.0f / (1.0f + expf(-v)); };
    std::vector<float> gates(4 * H, 0.0f);
    for (int j = 0; j < 4 * H; j++) {
        float s = b_ih[j] + b_hh[j];
        for (int k = 0; k < in_dim; k++)
            s += w_ih[j * in_dim + k] * x_in[k];
        for (int k = 0; k < H; k++)
            s += w_hh[j * H + k] * h[k];
        gates[j] = s;
    }
    for (int j = 0; j < H; j++) {
        float i_g = sig(gates[0 * H + j]);
        float f_g = sig(gates[1 * H + j]);
        float g_g = tanhf(gates[2 * H + j]);
        float o_g = sig(gates[3 * H + j]);
        c[j] = f_g * c[j] + i_g * g_g;
        h_out[j] = o_g * tanhf(c[j]);
    }
}

static void predictor_step(const nemotron_predictor_weights& W, int token_id, nemotron_lstm_state& state,
                           std::vector<float>& pred_out) {
    const int H = W.H;
    pred_out.assign(H, 0.0f);

    std::vector<float> x(W.embed.data() + (size_t)token_id * H, W.embed.data() + (size_t)(token_id + 1) * H);

    std::vector<float> h0_new(H);
    lstm_step_layer(x.data(), W.w_ih_0.data(), W.b_ih_0.data(), W.w_hh_0.data(), W.b_hh_0.data(), state.h0.data(),
                    state.c0.data(), h0_new.data(), H, H);
    state.h0 = h0_new;

    std::vector<float> h1_new(H);
    lstm_step_layer(state.h0.data(), W.w_ih_1.data(), W.b_ih_1.data(), W.w_hh_1.data(), W.b_hh_1.data(),
                    state.h1.data(), state.c1.data(), h1_new.data(), H, H);
    state.h1 = h1_new;

    pred_out = state.h1;
}

// ===========================================================================
// Joint head (CPU F32) — ReLU activation (NeMo default)
// ===========================================================================

static void joint_proj_enc(const nemotron_joint_weights& J, const float* enc_t, std::vector<float>& out) {
    out.assign(J.joint_hidden, 0.0f);
    for (int i = 0; i < J.joint_hidden; i++) {
        float s = J.enc_b[i];
        const float* row = J.enc_w.data() + (size_t)i * J.d_model;
        for (int k = 0; k < J.d_model; k++)
            s += row[k] * enc_t[k];
        out[i] = s;
    }
}

static void joint_step(const nemotron_joint_weights& J, const float* proj_enc, const float* pred_u,
                       std::vector<float>& logits) {
    std::vector<float> mid(J.joint_hidden);
    for (int i = 0; i < J.joint_hidden; i++) {
        float s = J.pred_b[i];
        const float* row = J.pred_w.data() + (size_t)i * J.pred_hidden;
        for (int k = 0; k < J.pred_hidden; k++)
            s += row[k] * pred_u[k];
        float v = proj_enc[i] + s;
        mid[i] = v > 0.0f ? v : 0.0f; // ReLU
    }

    logits.assign(J.vocab_total, 0.0f);
    for (int v = 0; v < J.vocab_total; v++) {
        float s = J.out_b[v];
        const float* row = J.out_w.data() + (size_t)v * J.joint_hidden;
        for (int k = 0; k < J.joint_hidden; k++)
            s += row[k] * mid[k];
        logits[v] = s;
    }
}

// ===========================================================================
// Lazy weight cache init
// ===========================================================================

static void nemotron_init_pred_weights(nemotron_context* ctx) {
    if (ctx->pred_w.initialised)
        return;
    auto& p = ctx->model.predictor;
    auto& W = ctx->pred_w;
    const int H = (int)ctx->model.hparams.pred_hidden;

    W.embed = tensor_to_f32(p.embed_w);
    W.w_ih_0 = tensor_to_f32(p.lstm0_w_ih);
    W.w_hh_0 = tensor_to_f32(p.lstm0_w_hh);
    W.b_ih_0 = tensor_to_f32(p.lstm0_b_ih);
    W.b_hh_0 = tensor_to_f32(p.lstm0_b_hh);
    W.w_ih_1 = tensor_to_f32(p.lstm1_w_ih);
    W.w_hh_1 = tensor_to_f32(p.lstm1_w_hh);
    W.b_ih_1 = tensor_to_f32(p.lstm1_b_ih);
    W.b_hh_1 = tensor_to_f32(p.lstm1_b_hh);
    W.H = H;
    W.initialised = true;
}

static void nemotron_init_joint_weights(nemotron_context* ctx) {
    if (ctx->joint_w.initialised)
        return;
    auto& j = ctx->model.joint;
    auto& J = ctx->joint_w;
    const auto& hp = ctx->model.hparams;

    J.enc_w = tensor_to_f32(j.enc_w);
    J.enc_b = tensor_to_f32(j.enc_b);
    J.pred_w = tensor_to_f32(j.pred_w);
    J.pred_b = tensor_to_f32(j.pred_b);
    J.out_w = tensor_to_f32(j.out_w);
    J.out_b = tensor_to_f32(j.out_b);
    J.joint_hidden = (int)hp.joint_hidden;
    J.d_model = (int)hp.d_model;
    J.pred_hidden = (int)hp.pred_hidden;
    J.vocab_total = (int)j.out_b->ne[0]; // vocab + 1 blank
    J.initialised = true;
}

// ===========================================================================
// RNN-T greedy decode (pure RNNT, no TDT durations)
// ===========================================================================

struct nemotron_emitted_token {
    int id;
    int t_start;
    int t_end;
    float p;
};

static std::vector<nemotron_emitted_token> nemotron_rnnt_decode(nemotron_context* ctx, const float* enc, int T_enc,
                                                                int d_model) {
    nemotron_init_pred_weights(ctx);
    nemotron_init_joint_weights(ctx);

    const auto& W = ctx->pred_w;
    const auto& J = ctx->joint_w;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int max_symbols_per_frame = 10;

    std::vector<nemotron_emitted_token> emitted;
    nemotron_lstm_state state;
    state.init(W.H);

    std::vector<float> pred_out;
    predictor_step(W, blank_id, state, pred_out);

    for (int t = 0; t < T_enc; t++) {
        const float* enc_t = enc + (size_t)t * d_model;
        std::vector<float> proj_e;
        joint_proj_enc(J, enc_t, proj_e);

        int sym_count = 0;
        while (sym_count < max_symbols_per_frame) {
            std::vector<float> logits;
            joint_step(J, proj_e.data(), pred_out.data(), logits);

            // Softmax for probability
            float maxl = *std::max_element(logits.begin(), logits.end());
            float sum = 0.0f;
            for (auto& l : logits) {
                l = expf(l - maxl);
                sum += l;
            }
            for (auto& l : logits)
                l /= sum;

            int tok = 0;
            float maxp = logits[0];
            for (int v = 1; v < J.vocab_total; v++) {
                if (logits[v] > maxp) {
                    maxp = logits[v];
                    tok = v;
                }
            }

            if (tok == blank_id)
                break;

            nemotron_emitted_token et;
            et.id = tok;
            et.t_start = t;
            et.t_end = t + 1;
            et.p = logits[tok];
            emitted.push_back(et);

            predictor_step(W, tok, state, pred_out);
            sym_count++;
        }
    }
    return emitted;
}

// ===========================================================================
// Token → text conversion
// ===========================================================================

static std::string nemotron_detokenize(const nemotron_vocab& vocab, const std::vector<nemotron_emitted_token>& tokens) {
    std::string out;
    for (const auto& t : tokens) {
        if (t.id < 0 || t.id >= (int)vocab.id_to_token.size())
            continue;
        std::string piece = vocab.id_to_token[t.id];
        // SentencePiece: replace '▁' with space
        size_t pos = 0;
        while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos += 1;
        }
        out += piece;
    }
    // Trim leading space
    if (!out.empty() && out[0] == ' ')
        out.erase(0, 1);
    return out;
}

// Group emitted sub-word tokens into words at '▁' boundaries.
static void nemotron_group_words(const nemotron_vocab& vocab, const std::vector<nemotron_emitted_token>& tokens,
                                 int frame_dur_cs, int64_t t_offset_cs, std::vector<nemotron_word_data>& words) {
    words.clear();
    nemotron_word_data cur_word = {};
    cur_word.t0 = 0;
    cur_word.t1 = 0;
    cur_word.p = 0.0f;
    int n_sub = 0;
    bool have_word = false;

    for (const auto& t : tokens) {
        if (t.id < 0 || t.id >= (int)vocab.id_to_token.size())
            continue;
        const std::string& piece = vocab.id_to_token[t.id];
        bool starts_word = (piece.find("\xe2\x96\x81") == 0);

        if (starts_word && have_word && n_sub > 0) {
            cur_word.p /= (float)n_sub;
            words.push_back(cur_word);
            cur_word = {};
            n_sub = 0;
        }

        if (!have_word || starts_word) {
            have_word = true;
            cur_word.t0 = t_offset_cs + (int64_t)t.t_start * frame_dur_cs;
        }

        // Append text
        std::string text = piece;
        size_t pos = 0;
        while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) {
            text.replace(pos, 3, "");
        }
        size_t len = strlen(cur_word.text);
        size_t avail = sizeof(cur_word.text) - len - 1;
        if (text.size() <= avail)
            memcpy(cur_word.text + len, text.c_str(), text.size() + 1);

        cur_word.t1 = t_offset_cs + (int64_t)t.t_end * frame_dur_cs;
        cur_word.p += t.p;
        n_sub++;
    }

    if (have_word && n_sub > 0) {
        cur_word.p /= (float)n_sub;
        words.push_back(cur_word);
    }
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct nemotron_context_params nemotron_context_default_params(void) {
    nemotron_context_params p;
    p.n_threads = 4;
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = false;
    return p;
}

extern "C" struct nemotron_context* nemotron_init_from_file(const char* path_model,
                                                            struct nemotron_context_params params) {
    auto* ctx = new nemotron_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    // Backend selection — use ggml_backend_init_best() for portable GPU init
    ctx->backend = nullptr;
    ctx->backend_cpu = ggml_backend_cpu_init();

    if (params.use_gpu) {
        ctx->backend = ggml_backend_init_best();
    }
    if (!ctx->backend) {
        ctx->backend = ctx->backend_cpu;
    }
    if (params.verbosity > 0) {
        fprintf(stderr, "nemotron: backend = %s\n", ggml_backend_name(ctx->backend));
    }

    // compute_meta buffer
    ctx->compute_meta.resize(16 * 1024 * 1024);

    // Load model
    if (!nemotron_load_model(ctx->model, ctx->vocab, ctx->lang_to_prompt, path_model, ctx->backend)) {
        fprintf(stderr, "nemotron: failed to load model from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

extern "C" void nemotron_free(struct nemotron_context* ctx) {
    if (!ctx)
        return;

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

extern "C" void nemotron_result_free(struct nemotron_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" char* nemotron_transcribe(struct nemotron_context* ctx, const float* samples, int n_samples) {
    nemotron_result* r = nemotron_transcribe_ex(ctx, samples, n_samples, 0);
    if (!r)
        return nullptr;
    char* text = r->text;
    r->text = nullptr;
    nemotron_result_free(r);
    return text;
}

extern "C" struct nemotron_result* nemotron_transcribe_ex(struct nemotron_context* ctx, const float* samples,
                                                          int n_samples, int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    // Compute mel
    int T_mel = 0;
    auto mel = nemotron_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty() || T_mel <= 0)
        return nullptr;

    // Run encoder — use cache-aware chunked path (NEMOTRON_BATCH=1 for old bidirectional path)
    std::vector<float> enc_out;
    int T_enc = 0, d_model = 0;
    const bool use_batch = getenv("NEMOTRON_BATCH");
    if (use_batch) {
        // Legacy: full-sequence bidirectional (doesn't produce tokens — for debugging only)
        if (!nemotron_run_encoder(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, enc_out, T_enc, d_model))
            return nullptr;
    } else {
        // Cache-aware chunked encoder (streaming-correct)
        // Step 1: run pre-encode only (extract from the full graph)
        if (!nemotron_run_encoder(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, enc_out, T_enc, d_model))
            return nullptr;
        // enc_out now has the full bidirectional output — but we need the pre-encode output.
        // TODO: extract pre-encode separately. For now, re-run with the pre-encode only.
        // WORKAROUND: run the chunked encoder on the pre-encode output.
        // We need to get the pre-encode output first. Let's add a tag for it.
        // For now, we use the Kaggle-validated CPU pre-encode from the worktree approach.
        // Actually — the simplest approach is to just run the chunked encoder directly
        // on the mel, computing pre-encode first.

        // For now, run the full-graph pre-encode and extract it
        // The enc_out already has the bidirectional result; we need the pre-encode.
        // Let's modify the graph to output pre-encode too... but that's complex.
        // SIMPLER: just run chunked encoder on enc_out from the bidirectional path
        // (won't work — the chunked encoder expects pre-encode input).

        // The right fix: build a pre-encode-only graph, extract, then run chunked.
        // For this first pass, let's build a separate pre-encode graph.
        fprintf(stderr, "nemotron: running chunked encoder path\n");

        // Build pre-encode-only graph
        {
            const auto& hp2 = ctx->model.hparams;
            size_t meta_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(1024, false);
            std::vector<uint8_t> meta(meta_size);
            ggml_init_params ip = {meta_size, meta.data(), true};
            ggml_context* ctx0 = ggml_init(ip);
            ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

            ggml_tensor* mel_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp2.n_mels, T_mel);
            ggml_set_name(mel_t, "mel");
            ggml_set_input(mel_t);

            int T_pre = 0;
            ggml_tensor* pre =
                nemotron_build_pre_encode(ctx0, mel_t, ctx->model.pre_encode, (int)hp2.subsampling_channels, &T_pre);
            ggml_set_name(pre, "pre_enc");
            ggml_set_output(pre);
            ggml_build_forward_expand(gf, pre);

            ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            ggml_gallocr_reserve(alloc, gf);
            ggml_gallocr_alloc_graph(alloc, gf);

            ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
            ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));

            if (ggml_backend_is_cpu(ctx->backend))
                ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
            ggml_backend_graph_compute(ctx->backend, gf);

            ggml_tensor* pre_out = ggml_graph_get_tensor(gf, "pre_enc");
            T_enc = (int)pre_out->ne[1];
            d_model = (int)pre_out->ne[0];
            std::vector<float> pre_enc((size_t)T_enc * d_model);
            ggml_backend_tensor_get(pre_out, pre_enc.data(), 0, pre_enc.size() * sizeof(float));

            ggml_gallocr_free(alloc);
            ggml_free(ctx0);

            // Step 2: run chunked encoder on pre-encode output
            enc_out.clear();
            if (!nemotron_run_encoder_chunked(ctx, pre_enc.data(), T_enc, d_model, enc_out))
                return nullptr;
        }
    }

    if (T_enc <= 0)
        return nullptr;

    // RNN-T decode
    auto emitted = nemotron_rnnt_decode(ctx, enc_out.data(), T_enc, d_model);

    // Build result
    auto* r = (nemotron_result*)calloc(1, sizeof(nemotron_result));
    std::string text = nemotron_detokenize(ctx->vocab, emitted);
    r->text = strdup(text.c_str());

    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;

    // Tokens
    r->n_tokens = (int)emitted.size();
    if (r->n_tokens > 0) {
        r->tokens = (nemotron_token_data*)calloc(r->n_tokens, sizeof(nemotron_token_data));
        for (int i = 0; i < r->n_tokens; i++) {
            auto& et = emitted[i];
            auto& td = r->tokens[i];
            td.id = et.id;
            td.t0 = t_offset_cs + (int64_t)et.t_start * frame_dur_cs;
            td.t1 = t_offset_cs + (int64_t)et.t_end * frame_dur_cs;
            td.p = et.p;
            if (et.id >= 0 && et.id < (int)ctx->vocab.id_to_token.size()) {
                std::string piece = ctx->vocab.id_to_token[et.id];
                size_t pos = 0;
                while ((pos = piece.find("\xe2\x96\x81", pos)) != std::string::npos) {
                    piece.replace(pos, 3, " ");
                    pos += 1;
                }
                snprintf(td.text, sizeof(td.text), "%s", piece.c_str());
            }
        }
    }

    // Words
    std::vector<nemotron_word_data> words;
    nemotron_group_words(ctx->vocab, emitted, frame_dur_cs, t_offset_cs, words);
    r->n_words = (int)words.size();
    if (r->n_words > 0) {
        r->words = (nemotron_word_data*)malloc(r->n_words * sizeof(nemotron_word_data));
        memcpy(r->words, words.data(), r->n_words * sizeof(nemotron_word_data));
    }

    return r;
}

extern "C" void nemotron_set_context_preset(struct nemotron_context* ctx, int preset) {
    if (!ctx)
        return;
    if (preset >= 0 && preset < (int)ctx->model.hparams.n_att_context_presets)
        ctx->att_context_preset = preset;
}

extern "C" void nemotron_set_language(struct nemotron_context* ctx, const char* lang_code) {
    if (!ctx || !lang_code)
        return;
    auto it = ctx->lang_to_prompt.find(lang_code);
    if (it != ctx->lang_to_prompt.end()) {
        ctx->prompt_id = it->second;
    } else {
        // Default to English
        ctx->prompt_id = 0;
    }
}

extern "C" void nemotron_set_temperature(struct nemotron_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

extern "C" void nemotron_set_beam_size(struct nemotron_context* ctx, int beam_size) {
    if (!ctx)
        return;
    ctx->decode_beam_size = beam_size > 0 ? beam_size : 1;
}

extern "C" int nemotron_n_vocab(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.vocab_size : 0;
}

extern "C" int nemotron_blank_id(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.blank_id : 0;
}

extern "C" const char* nemotron_token_to_str(struct nemotron_context* ctx, int token_id) {
    if (!ctx || token_id < 0 || token_id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[token_id].c_str();
}

extern "C" int nemotron_frame_dur_cs(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.frame_dur_cs : 0;
}

extern "C" int nemotron_n_mels(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.n_mels : 0;
}

extern "C" int nemotron_sample_rate(struct nemotron_context* ctx) {
    return ctx ? (int)ctx->model.hparams.sample_rate : 0;
}
