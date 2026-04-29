// parakeet.cpp — nvidia/parakeet-tdt-0.6b-v3 ggml runtime
//
// First iteration: loader + public C API skeleton.
// Encoder forward (FastConformer), predictor LSTM, joint head, and the
// TDT greedy decode loop will land in subsequent commits.
//
// Architecture summary (see parakeet-todo.md for the full plan):
//   Mel:       128 mels @ 16 kHz, n_fft=512, win=400, hop=160 (Hann window)
//   Encoder:   24× FastConformer block, d_model=1024, 8 heads, head_dim=128,
//              ff_dim=4096, conv kernel=9, 8× temporal subsampling via dw_striding
//   Predictor: embed(8193, 640) + 2-layer LSTM(640, 640)
//   Joint:     enc(1024→640) + pred(640→640) → tanh → linear(640 → 8198)
//              8198 = 8192 vocab + 1 blank + 5 TDT durations {0,1,2,3,4}

#include "parakeet.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "core/fastconformer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ===========================================================================
// CPU weight caches for the predictor LSTM and joint head
//
// Both are populated lazily (parakeet_init_pred_weights / _joint_weights)
// from the model's GGUF tensors and used by the manual F32 stepping in
// predictor_step / joint_step. Stored on the parakeet_context by value.
// ===========================================================================

struct parakeet_predictor_weights {
    std::vector<float> embed; // [vocab+1, H]
    std::vector<float> w_ih_0, w_hh_0, b_ih_0, b_hh_0;
    std::vector<float> w_ih_1, w_hh_1, b_ih_1, b_hh_1;
    int H = 0;
    bool initialised = false;
};

struct parakeet_joint_weights {
    std::vector<float> enc_w, enc_b;   // (joint_hidden, d_model), (joint_hidden,)
    std::vector<float> pred_w, pred_b; // (joint_hidden, pred_hidden), (joint_hidden,)
    std::vector<float> out_w, out_b;   // (vocab_total, joint_hidden), (vocab_total,)
    int joint_hidden = 0;
    int d_model = 0;
    int pred_hidden = 0;
    int vocab_total = 0;
    bool initialised = false;
};

// ===========================================================================
// Hyper-parameters
// ===========================================================================

struct parakeet_hparams {
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
    bool xscaling = true; // RelPositionalEncoding scales encoder input by sqrt(d_model)
    uint32_t pred_hidden = 640;
    uint32_t pred_layers = 2;
    uint32_t joint_hidden = 640;
    uint32_t vocab_size = 8192;
    uint32_t blank_id = 8192;
    uint32_t n_tdt_durations = 5;
    uint32_t frame_dur_cs = 8; // 80 ms per encoder frame
    std::vector<int32_t> tdt_durations = {0, 1, 2, 3, 4};
};

// ===========================================================================
// Per-layer tensor containers
// ===========================================================================

// Pre-encode weights: exactly the shared FastConformer layout.
using parakeet_pre_encode = core_conformer::PreEncodeWeights;

// Per-layer tensor container: inherits the shared Conformer block weights
// and adds the BN tensors used only at load time (BN folding).
struct parakeet_enc_layer : core_conformer::BlockWeights {
    // Raw BN tensors (consumed by parakeet_fold_batchnorm and then unused).
    ggml_tensor *conv_bn_w = nullptr, *conv_bn_b = nullptr;
    ggml_tensor *conv_bn_rm = nullptr, *conv_bn_rv = nullptr;
};

struct parakeet_predictor {
    ggml_tensor* embed_w = nullptr; // (vocab+1, pred_hidden)
    // 2-layer LSTM (PyTorch convention: w_ih [4H, in], w_hh [4H, H], b_ih [4H], b_hh [4H])
    ggml_tensor *lstm0_w_ih = nullptr, *lstm0_w_hh = nullptr;
    ggml_tensor *lstm0_b_ih = nullptr, *lstm0_b_hh = nullptr;
    ggml_tensor *lstm1_w_ih = nullptr, *lstm1_w_hh = nullptr;
    ggml_tensor *lstm1_b_ih = nullptr, *lstm1_b_hh = nullptr;
};

struct parakeet_joint {
    ggml_tensor *enc_w = nullptr, *enc_b = nullptr;   // (joint_hidden, d_model)
    ggml_tensor *pred_w = nullptr, *pred_b = nullptr; // (joint_hidden, pred_hidden)
    ggml_tensor *out_w = nullptr, *out_b = nullptr;   // (vocab+1+n_dur, joint_hidden)
};

// ===========================================================================
// Model and vocabulary
// ===========================================================================

struct parakeet_model {
    parakeet_hparams hparams;

    // Mel preprocessor weights (baked into the .nemo checkpoint)
    ggml_tensor* mel_fb = nullptr;     // (1, n_mels, n_fft/2+1)
    ggml_tensor* mel_window = nullptr; // (win_length,)

    parakeet_pre_encode pre_encode;
    std::vector<parakeet_enc_layer> enc;
    parakeet_predictor predictor;
    parakeet_joint joint;

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor*> tensors;
};

struct parakeet_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
};

struct parakeet_context {
    parakeet_context_params params;

    parakeet_model model;
    parakeet_vocab vocab;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> compute_meta; // metadata buffer for graph allocation

    // CPU-side weight caches for the predictor LSTM and joint head.
    // Lazy-initialised on first transcribe call.
    parakeet_predictor_weights pred_w;
    parakeet_joint_weights joint_w;

    int n_threads = 4;

    // Decode-time sampling controls. Default temperature == 0 keeps the
    // bit-identical pure-argmax path; > 0 switches the TDT decoder over
    // to numerically-stable softmax sampling on the vocab+blank logits.
    // Set via parakeet_set_temperature(); the field is sticky on the ctx.
    float decode_temperature = 0.0f;
    uint64_t decode_seed = 0;
};

// ---------------------------------------------------------------------------
// Transformer-XL relative-position shift.
// Same trick as cohere.cpp: a single ggml_view_3d that walks the BD matrix
// with stride (nb[1] - nb[0]) along the time axis, dropping the upper
// triangle so each row picks up the rel-pos score r_{j-i}. Zero-cost view.
// ---------------------------------------------------------------------------
// rel_shift moved to core_conformer::rel_shift in src/core/fastconformer.h.

// ===========================================================================
// Loader helpers — thin wrappers around core_gguf:: to preserve the
// existing call-site syntax (try_get(model, name) / require(model, name))
// while the underlying work lives in src/core/gguf_loader.
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* try_get(parakeet_model& m, const char* name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor* require(parakeet_model& m, const char* name) {
    return core_gguf::require(m.tensors, name, "parakeet");
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool parakeet_load_model(parakeet_model& model, parakeet_vocab& vocab, const char* path,
                                ggml_backend_t backend) {
    // ---- pass 1: read hparams + vocab via metadata-only context ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path);
        if (!gctx)
            return false;

        auto& hp = model.hparams;
        hp.sample_rate = core_gguf::kv_u32(gctx, "parakeet.sample_rate", hp.sample_rate);
        hp.n_mels = core_gguf::kv_u32(gctx, "parakeet.n_mels", hp.n_mels);
        hp.n_fft = core_gguf::kv_u32(gctx, "parakeet.n_fft", hp.n_fft);
        hp.win_length = core_gguf::kv_u32(gctx, "parakeet.win_length", hp.win_length);
        hp.hop_length = core_gguf::kv_u32(gctx, "parakeet.hop_length", hp.hop_length);
        hp.d_model = core_gguf::kv_u32(gctx, "parakeet.d_model", hp.d_model);
        hp.n_layers = core_gguf::kv_u32(gctx, "parakeet.n_layers", hp.n_layers);
        hp.n_heads = core_gguf::kv_u32(gctx, "parakeet.n_heads", hp.n_heads);
        hp.head_dim = core_gguf::kv_u32(gctx, "parakeet.head_dim", hp.head_dim);
        hp.ff_dim = core_gguf::kv_u32(gctx, "parakeet.ff_dim", hp.ff_dim);
        hp.subsampling_factor = core_gguf::kv_u32(gctx, "parakeet.subsampling_factor", hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "parakeet.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel = core_gguf::kv_u32(gctx, "parakeet.conv_kernel", hp.conv_kernel);
        hp.xscaling = core_gguf::kv_bool(gctx, "parakeet.xscaling", hp.xscaling);
        hp.pred_hidden = core_gguf::kv_u32(gctx, "parakeet.pred_hidden", hp.pred_hidden);
        hp.pred_layers = core_gguf::kv_u32(gctx, "parakeet.pred_layers", hp.pred_layers);
        hp.joint_hidden = core_gguf::kv_u32(gctx, "parakeet.joint_hidden", hp.joint_hidden);
        hp.vocab_size = core_gguf::kv_u32(gctx, "parakeet.vocab_size", hp.vocab_size);
        hp.blank_id = core_gguf::kv_u32(gctx, "parakeet.blank_id", hp.blank_id);
        hp.n_tdt_durations = core_gguf::kv_u32(gctx, "parakeet.n_tdt_durations", hp.n_tdt_durations);
        hp.frame_dur_cs = core_gguf::kv_u32(gctx, "parakeet.frame_dur_cs", hp.frame_dur_cs);

        // Vocab: tokenizer.ggml.tokens is a string array.
        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        core_gguf::free_metadata(gctx);
    }

    // ---- pass 2: load tensor data via the shared helper ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "parakeet", wl)) {
        return false;
    }
    model.ctx = wl.ctx;
    model.buf = wl.buf;
    model.tensors = std::move(wl.tensors);

    // ---- bind named tensors into the per-layer structs ----

    // Mel preprocessor (optional — may be absent if recomputed at runtime)
    model.mel_fb = try_get(model, "preprocessor.fb");
    model.mel_window = try_get(model, "preprocessor.window");

    // Pre-encode (subsampling)
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
        e.conv_dw_b = get("conv.dw.bias"); // synthetic — populated by BN fold
        e.conv_pw2_w = get("conv.pw2.weight");
        e.conv_pw2_b = try_("conv.pw2.bias");
        e.conv_bn_w = get("conv.bn.weight");
        e.conv_bn_b = get("conv.bn.bias");
        e.conv_bn_rm = get("conv.bn.running_mean");
        e.conv_bn_rv = get("conv.bn.running_var");

        e.norm_ff2_w = get("norm_ff2.weight");
        e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w = get("ff2.linear1.weight");
        e.ff2_l1_b = try_("ff2.linear1.bias");
        e.ff2_l2_w = get("ff2.linear2.weight");
        e.ff2_l2_b = try_("ff2.linear2.bias");

        e.norm_out_w = get("norm_out.weight");
        e.norm_out_b = get("norm_out.bias");
    }

    // Predictor
    auto& p = model.predictor;
    p.embed_w = require(model, "decoder.embed.weight");
    p.lstm0_w_ih = require(model, "decoder.lstm.0.w_ih");
    p.lstm0_w_hh = require(model, "decoder.lstm.0.w_hh");
    p.lstm0_b_ih = require(model, "decoder.lstm.0.b_ih");
    p.lstm0_b_hh = require(model, "decoder.lstm.0.b_hh");
    p.lstm1_w_ih = require(model, "decoder.lstm.1.w_ih");
    p.lstm1_w_hh = require(model, "decoder.lstm.1.w_hh");
    p.lstm1_b_ih = require(model, "decoder.lstm.1.b_ih");
    p.lstm1_b_hh = require(model, "decoder.lstm.1.b_hh");

    // Joint
    auto& j = model.joint;
    j.enc_w = require(model, "joint.enc.weight");
    j.enc_b = require(model, "joint.enc.bias");
    j.pred_w = require(model, "joint.pred.weight");
    j.pred_b = require(model, "joint.pred.bias");
    j.out_w = require(model, "joint.out.weight");
    j.out_b = require(model, "joint.out.bias");

    fprintf(stderr, "parakeet: vocab=%u  d_model=%u  n_layers=%u  n_heads=%u  ff=%u  pred=%u  joint=%u\n",
            model.hparams.vocab_size, model.hparams.d_model, model.hparams.n_layers, model.hparams.n_heads,
            model.hparams.ff_dim, model.hparams.pred_hidden, model.hparams.joint_hidden);
    return true;
}

// ===========================================================================
// FFT (iterative Cooley-Tukey, real-input, N must be a power of 2)
// ===========================================================================

static void parakeet_fft_r2c(const float* in, int N, float* out) {
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
// NeMo-style mel spectrogram
//
// AudioToMelSpectrogramPreprocessor defaults for parakeet-tdt-0.6b-v3:
//   sample_rate=16000  features=128  n_fft=512
//   window_size=0.025 (= 400 samples)  window_stride=0.010 (= 160 samples)
//   window=hann  log=True  normalize=per_feature  dither=1e-5
//   mag_power=2.0  log_zero_guard_value=2^-24 ≈ 5.96e-8
//
// No pre-emphasis (unlike Cohere). Window and mel filterbank are loaded
// directly from the GGUF preprocessor.* tensors.
//
// The heavy lifting lives in src/core/mel.cpp now — this function just
// pulls the GGUF-stored window and filterbank, then delegates. The FFT
// function pointer keeps parakeet's own Cooley-Tukey implementation so
// numerical output is bit-exact with the pre-refactor version.
//
// Returns mel as a flat row-major [T, n_mels] (the layout the parakeet
// encoder expects — ne[0]=n_mels fastest means each frame's 128 mels are
// contiguous in memory).
// ===========================================================================

#include "core/mel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static std::vector<float> parakeet_compute_mel_impl(parakeet_context* ctx, const float* samples, int n_samples,
                                                    int& T_out) {
    const auto& hp = ctx->model.hparams;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int win = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) {
        fprintf(stderr, "parakeet: missing preprocessor.fb or preprocessor.window in GGUF\n");
        return {};
    }

    // Pull window and filterbank from the GGUF.
    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    // Configure the shared helper for the NeMo cluster:
    //   ln + per-mel z-score, (T, n_mels) output, center-padded input,
    //   log_eps = 2^-24 (NeMo log_zero_guard_value), 0.97 pre-emphasis
    //   (NeMo AudioToMelSpectrogramPreprocessor default — applied at
    //   inference, missing it caused JA token deletions: issue #37).
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
    p.preemph = 0.97f;

    auto mel = core_mel::compute(samples, n_samples, window_raw.data(), win, mel_fb.data(), n_freqs, parakeet_fft_r2c,
                                 p, T_out);
    return mel;
}

// ===========================================================================
// BatchNorm folding (load-time, once)
//
// Inference-time BN: y = (x - mean) / sqrt(var + eps) * gamma + beta
//                     = x * s + (beta - mean * s)   where s = gamma/sqrt(var+eps)
//
// Since mean / var are fixed after training, we fold s into the depthwise conv
// weights and absorb the bias shift into the synthetic conv_dw_b tensor (which
// the converter pre-allocated as zeros). After this the encoder graph drops
// the BN block entirely.
// ===========================================================================

static void parakeet_fold_batchnorm(parakeet_model& model) {
    const int d = (int)model.hparams.d_model;
    const int K = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.n_layers; il++) {
        auto& e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b || !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv) {
            fprintf(stderr, "parakeet: BN fold: missing tensor on layer %u\n", il);
            return;
        }

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w, bn_w.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b, bn_b.data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        // Fold s into conv_dw_w (F16, ggml shape [K, 1, d]).
        {
            std::vector<float> w_f32((size_t)K * d);
            // dw_w is stored as F16; read+convert via ggml helpers.
            std::vector<ggml_fp16_t> w_f16((size_t)K * d);
            ggml_backend_tensor_get(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < w_f16.size(); i++)
                w_f32[i] = ggml_fp16_to_fp32(w_f16[i]);
            for (int c = 0; c < d; c++)
                for (int ki = 0; ki < K; ki++)
                    w_f32[ki + c * K] *= s[c];
            for (size_t i = 0; i < w_f16.size(); i++)
                w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
            ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        }

        // Fold into conv_dw_b: b[c] = (existing_b[c] - mean[c]) * s[c] + bn_b[c]
        // Read existing bias (may be non-zero for models with explicit dw bias)
        std::vector<float> dw_b(d, 0.0f);
        ggml_backend_tensor_get(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
        for (int c = 0; c < d; c++)
            dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_b[c];
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }

    fprintf(stderr, "parakeet: BN folded into conv_dw weights for %u layers\n", model.hparams.n_layers);
}

// ===========================================================================
// Encoder graph builder
//
// Input:  mel [n_mels, T_mel]
// Output: enc_out [d_model, T_enc]   where T_enc = T_mel / subsampling_factor
// ===========================================================================

static const float kLayerNormEps = 1e-5f;
static const float kBatchNormEps = 1e-5f;

static ggml_cgraph* parakeet_build_graph_encoder(parakeet_context* ctx, int T_mel) {
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

    // ----- Pre-encode (dw_striding subsampling 8×) -----
    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // ----- xscaling: NeMo's RelPositionalEncoding multiplies the encoder
    // input by sqrt(d_model) before the conformer layers when the model's
    // `encoder.xscaling: true` (default for parakeet-tdt-0.6b-v3 and -ja).
    // Without it, every layer's input is 32× too small, the rel-pos
    // sinusoid is the same scale, and the model produces near-random
    // activations downstream.
    if (hp.xscaling) {
        const float xscale = sqrtf((float)hp.d_model);
        cur = ggml_scale(ctx0, cur, xscale);
    }

    // ----- Sinusoidal rel-pos table [d, 2T-1] -----
    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- 24× FastConformer block -----
    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads, (int)hp.head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// make_pos_enc moved to core_conformer::make_pos_enc in src/core/fastconformer.h.

// Run the encoder once. Returns enc_out as a flat row-major [T_enc, d_model].
// Caller computes T_enc as ceil(T_mel / subsampling_factor) (approximately —
// the actual value depends on the conv arithmetic and is reported back).
static std::vector<float> parakeet_encode_mel(parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                              int* out_T_enc) {
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "parakeet: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return {};
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    ggml_cgraph* gf = parakeet_build_graph_encoder(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parakeet: failed to alloc encoder graph\n");
        return {};
    }

    // Set inputs
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2; // pos_enc has 2T-1 columns; recover T
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    // Compute
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parakeet: encoder graph compute failed\n");
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "enc_out");
    if (!out) {
        fprintf(stderr, "parakeet: missing enc_out tensor\n");
        return {};
    }
    const int d = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    if (out_T_enc)
        *out_T_enc = Te;

    std::vector<float> result((size_t)d * Te);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

// ===========================================================================
// LSTM predictor (manual F32 step on the CPU)
//
// We don't go through ggml here — the predictor runs once per *emitted token*
// (not per encoder frame), the input is a single 640-vector, and the work
// per step is two small (640 → 4*640) GEMMs. A direct loop is simpler than
// building a per-step graph and the performance ceiling is identical.
//
// PyTorch LSTM weight layout:
//   weight_ih [4H, in_dim]   gates packed as [i, f, g, o]
//   weight_hh [4H, H]
//   bias_ih   [4H]
//   bias_hh   [4H]
// Forward (per layer):
//   gates = weight_ih @ x + bias_ih + weight_hh @ h + bias_hh
//   i = sigmoid(gates[0..H])
//   f = sigmoid(gates[H..2H])
//   g = tanh   (gates[2H..3H])
//   o = sigmoid(gates[3H..4H])
//   c' = f * c + i * g
//   h' = o * tanh(c')
// ===========================================================================

struct parakeet_lstm_state {
    std::vector<float> h0, c0; // layer 0
    std::vector<float> h1, c1; // layer 1
};

static void lstm_init_state(parakeet_lstm_state& s, int H) {
    s.h0.assign(H, 0.0f);
    s.c0.assign(H, 0.0f);
    s.h1.assign(H, 0.0f);
    s.c1.assign(H, 0.0f);
}

// Read an F16/F32 ggml tensor into a flat F32 std::vector for CPU stepping.
static std::vector<float> tensor_to_f32(ggml_tensor* t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        // Quantised types: dequantise via ggml-cpu helper
        const struct ggml_type_traits* tr = ggml_get_type_traits(t->type);
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        tr->to_float(raw.data(), out.data(), n);
    }
    return out;
}

static void lstm_step_layer(const float* x, // [in_dim]
                            const float* w_ih, const float* b_ih, const float* w_hh, const float* b_hh, float* h,
                            float* c,     // [H]   in/out
                            float* h_out, // [H]   out
                            int in_dim, int H) {
    const int H4 = 4 * H;
    std::vector<float> gates(H4);

    // gates = b_ih + b_hh + w_ih @ x + w_hh @ h
    for (int i = 0; i < H4; i++)
        gates[i] = b_ih[i] + b_hh[i];

    for (int i = 0; i < H4; i++) {
        const float* row = w_ih + (size_t)i * in_dim;
        float s = 0.0f;
        for (int k = 0; k < in_dim; k++)
            s += row[k] * x[k];
        gates[i] += s;
    }
    for (int i = 0; i < H4; i++) {
        const float* row = w_hh + (size_t)i * H;
        float s = 0.0f;
        for (int k = 0; k < H; k++)
            s += row[k] * h[k];
        gates[i] += s;
    }

    auto sig = [](float x) { return 1.0f / (1.0f + expf(-x)); };

    for (int j = 0; j < H; j++) {
        float i_g = sig(gates[0 * H + j]);
        float f_g = sig(gates[1 * H + j]);
        float g_g = tanhf(gates[2 * H + j]);
        float o_g = sig(gates[3 * H + j]);
        c[j] = f_g * c[j] + i_g * g_g;
        h_out[j] = o_g * tanhf(c[j]);
    }
}

// Run one predictor step:  input token id  →  pred_out [H]
static void predictor_step(const parakeet_predictor_weights& W, int token_id, parakeet_lstm_state& state,
                           std::vector<float>& pred_out) {
    const int H = W.H;
    pred_out.assign(H, 0.0f);

    // Embed token
    std::vector<float> x(W.embed.data() + (size_t)token_id * H, W.embed.data() + (size_t)(token_id + 1) * H);

    // Layer 0
    std::vector<float> h0_new(H);
    lstm_step_layer(x.data(), W.w_ih_0.data(), W.b_ih_0.data(), W.w_hh_0.data(), W.b_hh_0.data(), state.h0.data(),
                    state.c0.data(), h0_new.data(), H, H);
    state.h0 = h0_new;

    // Layer 1 — input is layer 0's hidden
    std::vector<float> h1_new(H);
    lstm_step_layer(state.h0.data(), W.w_ih_1.data(), W.b_ih_1.data(), W.w_hh_1.data(), W.b_hh_1.data(),
                    state.h1.data(), state.c1.data(), h1_new.data(), H, H);
    state.h1 = h1_new;

    pred_out = state.h1;
}

// ===========================================================================
// Joint head (CPU, F32) — runs once per (encoder_frame, predictor_state)
//
//   joint_in_e = enc_w @ enc[t]  + enc_b           [joint_hidden]
//   joint_in_p = pred_w @ pred_u + pred_b          [joint_hidden]
//   logits     = out_w @ tanh(joint_in_e + joint_in_p) + out_b
// Output: [vocab+1+n_dur] (8198 for parakeet-tdt-0.6b-v3)
// ===========================================================================

// Pre-compute proj_e once per encoder frame so we don't redo it inside the
// inner predictor loop.
static void joint_proj_enc(const parakeet_joint_weights& J, const float* enc_t, std::vector<float>& out) {
    out.assign(J.joint_hidden, 0.0f);
    for (int i = 0; i < J.joint_hidden; i++) {
        float s = J.enc_b[i];
        const float* row = J.enc_w.data() + (size_t)i * J.d_model;
        for (int k = 0; k < J.d_model; k++)
            s += row[k] * enc_t[k];
        out[i] = s;
    }
}

static void joint_step(const parakeet_joint_weights& J,
                       const float* proj_enc, // [joint_hidden]
                       const float* pred_u,   // [pred_hidden]
                       std::vector<float>& logits) {
    std::vector<float> mid(J.joint_hidden);
    for (int i = 0; i < J.joint_hidden; i++) {
        float s = J.pred_b[i];
        const float* row = J.pred_w.data() + (size_t)i * J.pred_hidden;
        for (int k = 0; k < J.pred_hidden; k++)
            s += row[k] * pred_u[k];
        // NeMo RNNTJoint uses ReLU (not tanh) — see jointnet.activation in
        // model_config.yaml.
        float v = proj_enc[i] + s;
        mid[i] = v > 0.0f ? v : 0.0f;
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
// Lazy weight cache initialisation (predictor + joint, F32 on CPU)
// ===========================================================================

static void parakeet_init_pred_weights(parakeet_context* ctx) {
    if (ctx->pred_w.initialised)
        return;
    auto& p = ctx->model.predictor;
    auto& W = ctx->pred_w;
    const int H = (int)ctx->model.hparams.pred_hidden;
    const int blank_id = (int)ctx->model.hparams.blank_id;

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

    // Sanity checks against the actual tensor shapes. ggml stores
    // PyTorch (rows, cols) as ne[0]=cols, ne[1]=rows.
    const int64_t embed_cols = p.embed_w->ne[0];       // == hidden
    const int64_t embed_rows = p.embed_w->ne[1];       // == vocab+1
    const int64_t lstm0_ih_cols = p.lstm0_w_ih->ne[0]; // == in_dim
    const int64_t lstm0_ih_rows = p.lstm0_w_ih->ne[1]; // == 4*H
    if (embed_cols != H) {
        fprintf(stderr,
                "parakeet: WARN embed.weight cols=%lld but pred_hidden=%d "
                "(GGUF hparam may be wrong)\n",
                (long long)embed_cols, H);
    }
    if (lstm0_ih_cols != H || lstm0_ih_rows != 4 * H) {
        fprintf(stderr,
                "parakeet: WARN lstm.0.w_ih shape=%lldx%lld but expected "
                "%dx%d (4*H, H)\n",
                (long long)lstm0_ih_rows, (long long)lstm0_ih_cols, 4 * H, H);
    }
    if (embed_rows < blank_id + 1) {
        fprintf(stderr, "parakeet: WARN embed.weight rows=%lld but blank_id+1=%d\n", (long long)embed_rows,
                blank_id + 1);
    }

    // NeMo's RNNT decoder uses Embedding(..., padding_idx=blank_id), so
    // the blank row should be all-zeros after training. If non-zero,
    // either the converter dropped padding_idx semantics or the
    // checkpoint was built with a different convention — the SOS step
    // (predictor at blank input) would produce wrong output.
    if (getenv("PARAKEET_DEBUG") && blank_id < (int)embed_rows) {
        const float* row = W.embed.data() + (size_t)blank_id * H;
        double s = 0;
        float maxv = 0;
        for (int k = 0; k < H; k++) {
            s += row[k] * row[k];
            if (fabsf(row[k]) > maxv)
                maxv = fabsf(row[k]);
        }
        fprintf(stderr,
                "parakeet: embed[blank=%d]  L2=%.4f  max|.|=%.4f  "
                "(NeMo padding_idx → expect ~0)\n",
                blank_id, sqrt(s), (double)maxv);
    }
}

static void parakeet_init_joint_weights(parakeet_context* ctx) {
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
    J.vocab_total = (int)j.out_b->ne[0]; // 8198 = 8192 vocab + 1 blank + 5 dur
    J.initialised = true;
}

// ===========================================================================
// TDT greedy decode
//
// State at each step: (t, u, predictor_state, last_token).
//   t = current encoder frame index (0 .. T_enc-1)
//   u = predictor step index (used for the predictor's autoregressive state)
//
// At each step, run the joint head on (enc[t], pred_state) and split the
// 8198-class output into (vocab_logits[8193], duration_logits[5]).
//
// Greedy:
//   token_id = argmax(vocab_logits)            (8192 = blank)
//   dur_skip = argmax(duration_logits)         (in {0, 1, 2, 3, 4})
//
// If token_id == blank: do not emit. Advance t by max(1, dur_skip).
// Else: emit (token_id, t, t + dur_skip). Advance the predictor by feeding
//       token_id, advance u++, advance t by max(1, dur_skip).
//
// Word timestamps come for free: each emitted token spans frames
// [t, t + dur_skip), and frame_dur = 80 ms in this model.
//
// Stop when t >= T_enc.
// ===========================================================================

struct parakeet_emitted_token {
    int id;
    int t_start; // encoder frame at emission
    int t_end;   // emission + duration
    float p;     // softmax probability of the emitted token [0, 1]
};

static std::vector<parakeet_emitted_token> parakeet_tdt_decode(parakeet_context* ctx, const float* enc, int T_enc,
                                                               int d_model) {
    parakeet_init_pred_weights(ctx);
    parakeet_init_joint_weights(ctx);

    const auto& hp = ctx->model.hparams;
    const int blank_id = (int)hp.blank_id;     // 8192
    const int n_vocab_blk = blank_id + 1;      // 8193 (vocab + blank)
    const int n_dur = (int)hp.n_tdt_durations; // 5
    const int max_per_step = 10;               // safety: cap predictor advances per t

    auto& W = ctx->pred_w;
    auto& J = ctx->joint_w;
    if (J.vocab_total != n_vocab_blk + n_dur) {
        fprintf(stderr, "parakeet: joint vocab_total mismatch (%d vs expected %d)\n", J.vocab_total,
                n_vocab_blk + n_dur);
    }

    std::vector<parakeet_emitted_token> emitted;
    emitted.reserve(256);

    parakeet_lstm_state state;
    lstm_init_state(state, W.H);

    // SOS / first input is the blank token (NeMo convention)
    std::vector<float> pred_out;
    predictor_step(W, blank_id, state, pred_out);
    if (getenv("PARAKEET_DEBUG"))
        fprintf(
            stderr, "parakeet: pred_out[blank]: mean=%.4f std=%.4f [0..3]=%.4f %.4f %.4f %.4f\n",
            [&] {
                double s = 0;
                for (auto v : pred_out)
                    s += v;
                return s / pred_out.size();
            }(),
            [&] {
                double s = 0, m = 0;
                for (auto v : pred_out) {
                    m += v;
                    s += v * v;
                }
                m /= pred_out.size();
                return sqrt(s / pred_out.size() - m * m);
            }(),
            (double)pred_out[0], (double)pred_out[1], (double)pred_out[2], (double)pred_out[3]);

    std::vector<float> proj_e(J.joint_hidden);
    std::vector<float> logits(J.vocab_total);

    // Sampling state — only touched when ctx->decode_temperature > 0.
    // We initialize unconditionally because seeding a mt19937_64 is
    // cheap, and keeping it outside the inner loop preserves the same
    // RNG sequence for the whole utterance.
    const bool sampling = ctx->decode_temperature > 0.0f;
    std::mt19937_64 rng(ctx->decode_seed != 0 ? ctx->decode_seed : (uint64_t)std::random_device{}());

    int t = 0;
    int total_steps = 0;
    while (t < T_enc) {
        joint_proj_enc(J, enc + (size_t)t * d_model, proj_e);

        int n_inner = 0;
        while (n_inner < max_per_step) {
            joint_step(J, proj_e.data(), pred_out.data(), logits);

            if (getenv("PARAKEET_DEBUG") && total_steps < 5) {
                // Show first few logits
                int best = 0;
                float best_v = logits[0];
                for (int v = 1; v < n_vocab_blk; v++)
                    if (logits[v] > best_v) {
                        best_v = logits[v];
                        best = v;
                    }
                fprintf(stderr, "parakeet: t=%d step=%d best_tok=%d (%.3f) blank_logit=%.3f\n", t, total_steps, best,
                        best_v, logits[blank_id]);
            }
            total_steps++;

            // Argmax (default) or temperature sample over the vocab+blank
            // logits. The duration logits are picked separately by argmax
            // either way — they're trained as a 5-way classifier and
            // sampling them would just inject random latency without any
            // quality benefit.
            int tok = 0;
            float tok_lp = logits[0];
            for (int v = 1; v < n_vocab_blk; v++) {
                if (logits[v] > tok_lp) {
                    tok_lp = logits[v];
                    tok = v;
                }
            }
            if (sampling) {
                // Numerically-stable softmax over the n_vocab_blk
                // half of the joint logits, then inverse-CDF sample.
                const float inv_t = 1.0f / ctx->decode_temperature;
                std::vector<double> pr((size_t)n_vocab_blk);
                double sum = 0.0;
                for (int v = 0; v < n_vocab_blk; v++) {
                    const double e = std::exp((double)((logits[v] - tok_lp) * inv_t));
                    pr[(size_t)v] = e;
                    sum += e;
                }
                if (sum > 0.0) {
                    std::uniform_real_distribution<double> unif(0.0, sum);
                    const double rr = unif(rng);
                    double acc = 0.0;
                    for (int v = 0; v < n_vocab_blk; v++) {
                        acc += pr[(size_t)v];
                        if (rr <= acc) {
                            tok = v;
                            break;
                        }
                    }
                    tok_lp = logits[tok];
                }
            }

            // Argmax over the duration logits (last n_dur entries)
            int dur_id = 0;
            float dur_lp = logits[n_vocab_blk];
            for (int d = 1; d < n_dur; d++) {
                if (logits[n_vocab_blk + d] > dur_lp) {
                    dur_lp = logits[n_vocab_blk + d];
                    dur_id = d;
                }
            }
            int dur_skip = (int)hp.tdt_durations[dur_id]; // 0..4

            if (tok == blank_id) {
                // Blank → never emit, always advance t by at least 1 frame.
                t += std::max(1, dur_skip);
                break;
            }

            // Softmax probability of the picked token, scoped to the
            // vocab+blank half of the joint logits (the duration logits
            // are a separate softmax). Numerically stable: subtract the
            // per-row max before exponentiating.
            float tok_p = 1.0f;
            {
                double sum = 0.0;
                for (int v = 0; v < n_vocab_blk; v++) {
                    sum += std::exp((double)(logits[v] - tok_lp));
                }
                tok_p = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
            }

            // Real token: emit and advance the predictor
            int t_end = std::min(T_enc, t + std::max(0, dur_skip));
            emitted.push_back({tok, t, t_end, tok_p});
            predictor_step(W, tok, state, pred_out);

            // Diagnostic: dump predictor stats for the first few emissions
            // so we can compare with NeMo step-by-step (not just SOS).
            if (getenv("PARAKEET_DEBUG") && (int)emitted.size() <= 5) {
                double s = 0, m = 0;
                float minv = pred_out[0], maxv = pred_out[0];
                for (auto v : pred_out) {
                    m += v;
                    s += v * v;
                    if (v < minv)
                        minv = v;
                    if (v > maxv)
                        maxv = v;
                }
                m /= pred_out.size();
                double std_ = sqrt(s / pred_out.size() - m * m);
                fprintf(stderr,
                        "parakeet: emit#%zu tok=%d dur=%d  pred_out: mean=%.4f std=%.4f "
                        "min=%.3f max=%.3f [0..3]=%.3f %.3f %.3f %.3f\n",
                        emitted.size(), tok, dur_skip, m, std_, (double)minv, (double)maxv, (double)pred_out[0],
                        (double)pred_out[1], (double)pred_out[2], (double)pred_out[3]);
            }

            // Advance encoder frame by the predicted duration (≥ 0). If 0,
            // we stay on this frame for another inner step.
            if (dur_skip > 0) {
                t += dur_skip;
                break;
            }
            n_inner++;
        }

        if (n_inner >= max_per_step) {
            // Force a one-frame advance to guarantee progress.
            t++;
        }
    }

    return emitted;
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t pick_backend() {
    ggml_backend_t b = ggml_backend_init_best();
    return b ? b : ggml_backend_cpu_init();
}

static ggml_backend_t pick_backend(bool use_gpu) {
    return use_gpu ? pick_backend() : ggml_backend_cpu_init();
}

// ===========================================================================
// Public C API
// ===========================================================================

extern "C" struct parakeet_context_params parakeet_context_default_params(void) {
    parakeet_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.use_flash = false;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" struct parakeet_context* parakeet_init_from_file(const char* path_model,
                                                            struct parakeet_context_params params) {
    auto* ctx = new parakeet_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = pick_backend(params.use_gpu);
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ctx->backend_cpu;

    if (!parakeet_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        fprintf(stderr, "parakeet: failed to load '%s'\n", path_model);
        parakeet_free(ctx);
        return nullptr;
    }

    parakeet_fold_batchnorm(ctx->model);
    return ctx;
}

extern "C" void parakeet_free(struct parakeet_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

// Internal C++ entry point for tests — declared in parakeet.h via a different
// linkage section to avoid polluting the public C API.
extern std::vector<float> parakeet_encode_mel(parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                              int* out_T_enc);

// ---- Stage-level entry points for crispasr-diff ----

extern "C" void parakeet_set_temperature(struct parakeet_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

extern "C" float* parakeet_compute_mel(struct parakeet_context* ctx, const float* samples, int n_samples,
                                       int* out_n_mels, int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    int T_mel = 0;
    auto mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    const int n_mels = (int)ctx->model.hparams.n_mels;
    if (out_n_mels)
        *out_n_mels = n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;
    float* r = (float*)malloc(mel.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, mel.data(), mel.size() * sizeof(float));
    return r;
}

extern "C" float* parakeet_run_encoder(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                       int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    int T_enc = 0;
    auto enc = parakeet_encode_mel(ctx, mel, n_mels, T_mel, &T_enc);
    if (enc.empty())
        return nullptr;
    const int d = (int)ctx->model.hparams.d_model;
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;
    float* r = (float*)malloc(enc.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, enc.data(), enc.size() * sizeof(float));
    return r;
}

// Build encoder graph with per-layer outputs tagged for read-back. Each
// stage is named "dump_pre_encode" or "dump_layer_K" and marked with
// ggml_set_output() so the scheduler keeps its buffer live after compute.
// Mirrors parakeet_build_graph_encoder; kept separate so the production
// path (single output) doesn't pay for graph-output bookkeeping.
static ggml_cgraph* parakeet_build_graph_encoder_dump(parakeet_context* ctx, int T_mel) {
    const auto& m = ctx->model;
    const auto& hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {ctx->compute_meta.size(), ctx->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    int T = 0;
    ggml_tensor* cur = core_conformer::build_pre_encode(ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // Tag pre-encode output BEFORE the xscaling. NeMo's pre_encode forward
    // also returns the post-projection output (the multiply by sqrt(d) is
    // an attribute of the rel-pos encoder, not the subsampling module).
    {
        ggml_tensor* tag = ggml_cont(ctx0, cur);
        ggml_set_name(tag, "dump_pre_encode");
        ggml_set_output(tag);
        ggml_build_forward_expand(gf, tag);
    }

    if (hp.xscaling) {
        const float xscale = sqrtf((float)hp.d_model);
        cur = ggml_scale(ctx0, cur, xscale);
    }

    ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads, (int)hp.head_dim, (int)hp.conv_kernel, kLayerNormEps,
    };
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
        char nm[64];
        snprintf(nm, sizeof(nm), "dump_layer_%u", il);
        ggml_tensor* tag = ggml_cont(ctx0, cur);
        ggml_set_name(tag, nm);
        ggml_set_output(tag);
        ggml_build_forward_expand(gf, tag);
        cur = tag; // chain next layer off the tagged tensor (numerics fix)
    }

    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

extern "C" int parakeet_run_encoder_dump(struct parakeet_context* ctx, const float* mel, int n_mels, int T_mel,
                                         float** out, int out_count, int* out_T_enc, int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0 || !out)
        return 1;
    if (n_mels != (int)ctx->model.hparams.n_mels) {
        fprintf(stderr, "parakeet: mel feature mismatch (%d vs %d)\n", n_mels, (int)ctx->model.hparams.n_mels);
        return 2;
    }

    if (!ctx->sched) {
        ggml_backend_t backends[2] = {ctx->backend, ctx->backend_cpu};
        int n_be = (ctx->backend != ctx->backend_cpu) ? 2 : 1;
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 8192, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    }

    ggml_cgraph* gf = parakeet_build_graph_encoder_dump(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "parakeet: dump: failed to alloc encoder graph\n");
        return 3;
    }

    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    ggml_tensor* pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "parakeet: dump: encoder graph compute failed\n");
        return 4;
    }

    const int d = (int)ctx->model.hparams.d_model;
    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;

    auto read_to = [&](const char* name, float* dst) {
        ggml_tensor* t = ggml_graph_get_tensor(gf, name);
        if (!t) {
            fprintf(stderr, "parakeet: dump: missing %s tensor\n", name);
            return false;
        }
        ggml_backend_tensor_get(t, dst, 0, (size_t)d * T_enc * sizeof(float));
        return true;
    };

    if (out_count >= 1 && out[0])
        read_to("dump_pre_encode", out[0]);
    const int n_layers = (int)ctx->model.hparams.n_layers;
    for (int il = 0; il < n_layers && (il + 1) < out_count; il++) {
        if (!out[il + 1])
            continue;
        char nm[64];
        snprintf(nm, sizeof(nm), "dump_layer_%d", il);
        read_to(nm, out[il + 1]);
    }
    return 0;
}

extern "C" int parakeet_test_encoder(struct parakeet_context* ctx, int T_mel) {
    int n_mels = (int)ctx->model.hparams.n_mels;
    std::vector<float> mel((size_t)n_mels * T_mel, 0.0f);
    int T_enc = 0;
    auto out = parakeet_encode_mel(ctx, mel.data(), n_mels, T_mel, &T_enc);
    if (out.empty())
        return -1;
    if (getenv("PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: encoder OK — T_mel=%d → T_enc=%d  d=%d  out[0..3]=%g %g %g %g\n", T_mel, T_enc,
                (int)ctx->model.hparams.d_model, (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    return T_enc;
}

extern "C" int parakeet_test_audio(struct parakeet_context* ctx, const float* samples, int n_samples) {
    int T_mel = 0;
    auto mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return -1;

    if (getenv("PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: mel OK — n_samples=%d (%.2fs)  T_mel=%d  n_mels=%d  mel[0..3]=%g %g %g %g\n",
                n_samples, (double)n_samples / ctx->model.hparams.sample_rate, T_mel, (int)ctx->model.hparams.n_mels,
                (double)mel[0], (double)mel[1], (double)mel[2], (double)mel[3]);

    int T_enc = 0;
    auto enc_out = parakeet_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    if (enc_out.empty())
        return -1;

    // Print a few summary stats over the encoder output
    double sum = 0, sq = 0;
    float mn = enc_out[0], mx = enc_out[0];
    for (float v : enc_out) {
        sum += v;
        sq += (double)v * v;
        if (v < mn)
            mn = v;
        if (v > mx)
            mx = v;
    }
    double mean = sum / enc_out.size();
    double var = sq / enc_out.size() - mean * mean;
    if (getenv("PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: encoder OK — T_enc=%d  d=%d  mean=%.4f  std=%.4f  min=%.3f  max=%.3f\n", T_enc,
                (int)ctx->model.hparams.d_model, mean, sqrt(var), (double)mn, (double)mx);
    return T_enc;
}

extern "C" void parakeet_result_free(struct parakeet_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r->words);
    free(r);
}

extern "C" int parakeet_n_vocab(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.vocab_size;
}
extern "C" int parakeet_blank_id(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.blank_id;
}
extern "C" int parakeet_frame_dur_cs(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.frame_dur_cs;
}
extern "C" int parakeet_n_mels(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.n_mels;
}
extern "C" int parakeet_sample_rate(struct parakeet_context* ctx) {
    return (int)ctx->model.hparams.sample_rate;
}

extern "C" const char* parakeet_token_to_str(struct parakeet_context* ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "";
    return ctx->vocab.id_to_token[id].c_str();
}

// ===========================================================================
// Public transcribe entry points
// ===========================================================================

// Convert a SentencePiece vocab string to user-visible text:
//   '▁foo'  → ' foo'    (word-start prefix)
//   '<unk>' → ''        (filtered)
//   anything else → as-is
static std::string spiece_to_text(const std::string& piece) {
    if (piece.empty())
        return "";
    if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>')
        return "";
    // Replace leading U+2581 (▁ = 0xE2 0x96 0x81) with a space
    if (piece.size() >= 3 && (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81) {
        return std::string(" ") + piece.substr(3);
    }
    return piece;
}

extern "C" struct parakeet_result* parakeet_transcribe_ex(struct parakeet_context* ctx, const float* samples,
                                                          int n_samples, int64_t t_offset_cs) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    // 1. Mel
    int T_mel = 0;
    auto mel = parakeet_compute_mel_impl(ctx, samples, n_samples, T_mel);
    if (mel.empty())
        return nullptr;
    if (getenv("PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: mel OK (%d frames)\n", T_mel);

    // 2. Encoder
    int T_enc = 0;
    auto enc = parakeet_encode_mel(ctx, mel.data(), (int)ctx->model.hparams.n_mels, T_mel, &T_enc);
    if (enc.empty())
        return nullptr;
    if (getenv("PARAKEET_DEBUG")) {
        fprintf(stderr, "parakeet: encoder OK (%d frames)\n", T_enc);
        int d = (int)ctx->model.hparams.d_model;
        double s = 0, sq = 0;
        for (int i = 0; i < T_enc * d; i++) {
            s += enc[i];
            sq += (double)enc[i] * enc[i];
        }
        double m = s / (T_enc * d), v = sq / (T_enc * d) - m * m;
        fprintf(stderr, "parakeet: enc mean=%.4f std=%.4f enc[0,:4]=%.4f %.4f %.4f %.4f\n", m, sqrt(v), (double)enc[0],
                (double)enc[1], (double)enc[2], (double)enc[3]);
    }

    // 3. TDT greedy decode
    auto emitted = parakeet_tdt_decode(ctx, enc.data(), T_enc, (int)ctx->model.hparams.d_model);
    if (getenv("PARAKEET_DEBUG"))
        fprintf(stderr, "parakeet: decode OK (%d tokens)\n", (int)emitted.size());

    // 4. Build result
    auto* r = (parakeet_result*)calloc(1, sizeof(parakeet_result));
    r->n_tokens = (int)emitted.size();
    r->tokens = (parakeet_token_data*)calloc(r->n_tokens > 0 ? r->n_tokens : 1, sizeof(parakeet_token_data));
    std::string text;
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;
    for (int i = 0; i < r->n_tokens; i++) {
        const auto& e = emitted[i];
        const std::string& piece =
            (e.id >= 0 && e.id < (int)ctx->vocab.id_to_token.size()) ? ctx->vocab.id_to_token[e.id] : std::string("");
        std::string vis = spiece_to_text(piece);

        parakeet_token_data& td = r->tokens[i];
        td.id = e.id;
        td.t0 = t_offset_cs + (int64_t)e.t_start * frame_dur_cs;
        td.t1 = t_offset_cs + (int64_t)e.t_end * frame_dur_cs;
        td.p = e.p;
        size_t n = std::min(vis.size(), sizeof(td.text) - 1);
        memcpy(td.text, vis.data(), n);
        td.text[n] = '\0';
        text += vis;
    }
    // strip leading space
    if (!text.empty() && text[0] == ' ')
        text = text.substr(1);
    r->text = strdup(text.c_str());

    // ----- Group sub-word tokens into words -----
    //
    // Latin SentencePiece convention: a token starting with U+2581 (▁ → ' ')
    // begins a new word. Punctuation tokens attach to the previous word.
    //
    // Japanese parakeet (and other no-space tokenizers) emit no leading-space
    // markers because written Japanese has no inter-word spaces. In that
    // mode every non-punctuation token is its own "word" — sufficient
    // granularity for word-level SRT (issue #37).
    {
        std::vector<parakeet_word_data> words;
        words.reserve(r->n_tokens);

        // Detect tokenizer style: count tokens that look like real
        // word-starts (leading space + at least one more char). A
        // standalone " " token (BOS-ish) doesn't count.
        int n_space_word_starts = 0;
        for (int i = 0; i < r->n_tokens; i++) {
            const char* t = r->tokens[i].text;
            if (t[0] == ' ' && t[1] != '\0')
                n_space_word_starts++;
        }
        const bool space_prefix_style = (n_space_word_starts >= 2);

        // Pure-punctuation detector. Recognises ASCII punct plus the
        // common CJK punctuation (。、？！「」『』・,) so JA tokens
        // like "、" attach to the previous word instead of forming their
        // own subtitle entry.
        auto is_punct_only = [](const char* s) {
            if (!s || !*s)
                return false;
            const unsigned char* p = (const unsigned char*)s;
            while (*p) {
                unsigned char c = *p;
                if (c < 0x80) {
                    if (!(c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' || c == '\'' ||
                          c == '"' || c == '(' || c == ')' || c == '-'))
                        return false;
                    p++;
                } else if (c == 0xE3 && p[1] == 0x80 && p[2] >= 0x80 && p[2] <= 0xBF) {
                    // U+3000–U+303F: CJK Symbols and Punctuation (。、「」『』 etc.)
                    p += 3;
                } else if (c == 0xE3 && p[1] == 0x83 && p[2] == 0xBB) {
                    // U+30FB ・ (katakana middle dot)
                    p += 3;
                } else if (c == 0xEF && p[1] == 0xBC && ((p[2] >= 0x81 && p[2] <= 0x8F) || p[2] == 0x9F)) {
                    // U+FF01–U+FF0F (full-width !"#$%&'()*+,-./) and U+FF1F (？)
                    p += 3;
                } else {
                    return false;
                }
            }
            return true;
        };

        // True end-of-sentence punct (for the gap-insertion pass below).
        auto ends_with_sentence_punct = [](const char* s) {
            size_t len = s ? strlen(s) : 0;
            if (len == 0)
                return false;
            unsigned char last = (unsigned char)s[len - 1];
            if (last == '.' || last == '!' || last == '?')
                return true;
            if (len >= 3) {
                const unsigned char* tail = (const unsigned char*)(s + len - 3);
                // U+3002 。 = E3 80 82, U+FF01 ！ = EF BC 81, U+FF1F ？ = EF BC 9F
                if (tail[0] == 0xE3 && tail[1] == 0x80 && tail[2] == 0x82)
                    return true;
                if (tail[0] == 0xEF && tail[1] == 0xBC && (tail[2] == 0x81 || tail[2] == 0x9F))
                    return true;
            }
            return false;
        };

        parakeet_word_data cur = {};
        bool have_cur = false;

        for (int i = 0; i < r->n_tokens; i++) {
            const auto& td = r->tokens[i];
            if (!td.text[0])
                continue;
            // Skip standalone space tokens (e.g. an initial " " BOS marker).
            // Without this they leak through as empty word entries in
            // no-space mode.
            if (td.text[0] == ' ' && td.text[1] == '\0')
                continue;

            const bool has_leading_space = (td.text[0] == ' ');
            const bool is_punct = is_punct_only(td.text);

            // A token starts a new word if either:
            //   - it has a Latin-style leading-space marker, or
            //   - the segment is no-space style (e.g. JA) and the token
            //     is not pure punctuation (so 、 attaches to prev word).
            const bool is_new_word = !is_punct && (has_leading_space || !space_prefix_style);

            if (is_new_word && have_cur) {
                words.push_back(cur);
                cur = {};
                have_cur = false;
            }

            if (!have_cur) {
                cur.t0 = td.t0;
                have_cur = true;
            }
            cur.t1 = td.t1;

            // Append, dropping the leading space.
            const char* src = td.text + (has_leading_space ? 1 : 0);
            size_t cur_len = strlen(cur.text);
            size_t cap = sizeof(cur.text) - cur_len - 1;
            size_t add = strlen(src);
            if (add > cap)
                add = cap;
            memcpy(cur.text + cur_len, src, add);
            cur.text[cur_len + add] = '\0';
        }
        if (have_cur)
            words.push_back(cur);

        // Post-process: insert minimum gaps after sentence-ending punctuation.
        // The TDT decoder often produces contiguous timestamps even across
        // sentence boundaries (e.g. "code." t1==6.400, "In" t0==6.400).
        // When a word ends with .!?。！？ and the next word starts at the
        // exact same frame, shrink the punctuated word's t1 by one frame
        // duration to create a visible gap in subtitles.
        for (size_t wi = 0; wi + 1 < words.size(); wi++) {
            if (!ends_with_sentence_punct(words[wi].text))
                continue;
            if (words[wi].t1 >= words[wi + 1].t0 && words[wi].t1 > words[wi].t0) {
                int64_t shrunk = words[wi].t1 - frame_dur_cs;
                if (shrunk > words[wi].t0)
                    words[wi].t1 = shrunk;
            }
        }

        r->n_words = (int)words.size();
        r->words = (parakeet_word_data*)calloc(r->n_words > 0 ? r->n_words : 1, sizeof(parakeet_word_data));
        for (int i = 0; i < r->n_words; i++)
            r->words[i] = words[i];
    }

    return r;
}

extern "C" char* parakeet_transcribe(struct parakeet_context* ctx, const float* samples, int n_samples) {
    parakeet_result* r = parakeet_transcribe_ex(ctx, samples, n_samples, 0);
    if (!r)
        return nullptr;
    char* out = strdup(r->text ? r->text : "");
    parakeet_result_free(r);
    return out;
}
