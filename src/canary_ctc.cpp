// canary_ctc.cpp — slim runtime for canary's auxiliary CTC alignment model.
//
// The encoder forward is a direct port of parakeet's FastConformer encoder
// (the architecture is identical: 24 layers, no biases on q/k/v/out/ff,
// rel-pos attention with Transformer-XL untied biases). On top of the
// encoder we add a single CTC linear head: ctc.weight @ enc_out → logits.
//
// We use this for forced word alignment via subword CTC Viterbi (see
// canary_ctc_align_words below). It can also do greedy CTC decode for
// sanity checking.

#include "canary_ctc.h"

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
#include <cmath>
#include <cstdio>
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
// Hyperparameters (mirror canary_ctc.* keys in the GGUF)
// ===========================================================================

struct canary_ctc_hparams {
    uint32_t sample_rate          = 16000;
    uint32_t n_mels               = 128;
    uint32_t n_fft                = 512;
    uint32_t win_length           = 400;
    uint32_t hop_length           = 160;
    uint32_t d_model              = 1024;
    uint32_t n_layers             = 24;
    uint32_t n_heads              = 8;
    uint32_t head_dim             = 128;
    uint32_t ff_dim               = 4096;
    uint32_t subsampling_factor   = 8;
    uint32_t subsampling_channels = 256;
    uint32_t conv_kernel          = 9;
    uint32_t vocab_size           = 16384;   // 16384 SentencePiece pieces
    uint32_t blank_id             = 16384;   // last index = blank
    uint32_t frame_dur_cs         = 8;
    // NeMo ConformerEncoder "xscaling": when true, the encoder input is
    // multiplied by sqrt(d_model) AFTER the conv subsampling / pre_encode
    // projection and BEFORE the first Conformer block. This scales the
    // residual stream (the Macaron FFN's residual connection carries the
    // scaled value forward) and is a trained part of several NeMo
    // releases — e.g. the stt_en_fastconformer_ctc_large standalones
    // have xscaling=true, while canary-1b-v2 / its aligner have
    // xscaling=false. Default false preserves canary_ctc's existing
    // bit-identical path; the stt converter sets this to 1 in the GGUF.
    uint32_t xscaling             = 0;
};

// ===========================================================================
// Per-layer tensor containers (mirror parakeet_enc_layer — no biases)
// ===========================================================================

// Pre-encode weights: exactly the shared FastConformer layout.
using cc_pre_encode = core_conformer::PreEncodeWeights;

// Per-layer tensor container: inherits the shared Conformer block weights
// (canary_ctc is bias-less like parakeet) and adds the BN tensors used only
// at load time (BN folding).
struct cc_enc_layer : core_conformer::BlockWeights {
    ggml_tensor * conv_bn_w  = nullptr, * conv_bn_b  = nullptr;
    ggml_tensor * conv_bn_rm = nullptr, * conv_bn_rv = nullptr;
};

struct cc_model {
    canary_ctc_hparams hparams;

    ggml_tensor * mel_fb     = nullptr;
    ggml_tensor * mel_window = nullptr;

    cc_pre_encode               pre_encode;
    std::vector<cc_enc_layer>   enc;

    // CTC head: linear (d_model → vocab_total) where vocab_total = vocab_size + 1 (blank)
    ggml_tensor * ctc_w = nullptr;
    ggml_tensor * ctc_b = nullptr;

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::map<std::string, ggml_tensor *> tensors;
};

struct canary_ctc_vocab {
    std::vector<std::string>            id_to_token;
    std::unordered_map<std::string,int> token_to_id;
};

struct canary_ctc_context {
    canary_ctc_context_params params;

    cc_model         model;
    canary_ctc_vocab vocab;

    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    std::vector<uint8_t> compute_meta;

    int n_threads = 4;
};

// ===========================================================================
// Loader helpers
// ===========================================================================

#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor * cc_try_get(cc_model & m, const char * name) {
    return core_gguf::try_get(m.tensors, name);
}

static ggml_tensor * cc_require(cc_model & m, const char * name) {
    return core_gguf::require(m.tensors, name, "canary_ctc");
}

// ===========================================================================
// FFT (iterative Cooley-Tukey, real-input)
// ===========================================================================

static void cc_fft_r2c(const float * in, int N, float * out) {
    int bits = 0;
    for (int n = N; n > 1; n >>= 1) bits++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++) rev = (rev << 1) | ((i >> b) & 1);
        out[2*rev]   = in[i];
        out[2*rev+1] = 0.0f;
    }
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float are = out[2*a], aim = out[2*a+1];
                float bre = out[2*b], bim = out[2*b+1];
                float tre = ure*bre - uim*bim, tim = ure*bim + uim*bre;
                out[2*a]   = are + tre; out[2*a+1] = aim + tim;
                out[2*b]   = are - tre; out[2*b+1] = aim - tim;
                float new_ure = ure*wre - uim*wim;
                uim = ure*wim + uim*wre;
                ure = new_ure;
            }
        }
    }
}

// ===========================================================================
// NeMo-style mel spectrogram (shared with parakeet / canary / cohere)
// ===========================================================================

#include "core/mel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static std::vector<float> cc_compute_mel(canary_ctc_context * ctx,
                                         const float * samples, int n_samples,
                                         int & T_out)
{
    const auto & hp = ctx->model.hparams;
    const int n_fft   = (int)hp.n_fft;
    const int hop     = (int)hp.hop_length;
    const int win     = (int)hp.win_length;
    const int n_freqs = n_fft / 2 + 1;
    const int n_mels  = (int)hp.n_mels;

    if (!ctx->model.mel_fb || !ctx->model.mel_window) return {};

    std::vector<float> window_raw((size_t)win);
    ggml_backend_tensor_get(ctx->model.mel_window, window_raw.data(), 0, win * sizeof(float));

    std::vector<float> mel_fb((size_t)n_mels * n_freqs);
    ggml_backend_tensor_get(ctx->model.mel_fb, mel_fb.data(), 0, mel_fb.size() * sizeof(float));

    core_mel::Params p;
    p.n_fft      = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels     = n_mels;
    p.log_base   = core_mel::LogBase::Ln;
    p.norm       = core_mel::Normalization::PerFeatureZ;
    p.layout     = core_mel::Layout::TimeMels;
    p.log_eps    = (float)(1.0 / (1 << 24));
    p.center_pad = true;

    return core_mel::compute(
        samples, n_samples,
        window_raw.data(), win,
        mel_fb.data(), n_freqs,
        cc_fft_r2c,
        p,
        T_out);
}

// rel_shift and make_pos_enc moved to core_conformer in src/core/fastconformer.h.

// ===========================================================================
// Encoder graph + CTC head — direct port of parakeet's encoder, plus a final
// ggml_mul_mat with ctc.weight to produce per-frame logits.
// ===========================================================================

static const float kLayerNormEps = 1e-5f;

static ggml_cgraph * cc_build_graph(canary_ctc_context * ctx, int T_mel) {
    const auto & m  = ctx->model;
    const auto & hp = m.hparams;
    const int n_mels = (int)hp.n_mels;

    ggml_init_params ip = {
        /*mem_size=*/   ctx->compute_meta.size(),
        /*mem_buffer=*/ ctx->compute_meta.data(),
        /*no_alloc=*/   true,
    };
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // ----- Pre-encode (dw_striding 8×) -----
    int T = 0;
    ggml_tensor * cur = core_conformer::build_pre_encode(
        ctx0, mel, m.pre_encode, (int)hp.subsampling_channels, &T);

    // ----- Optional xscaling (NeMo standalone FastConformer-CTC) -----
    // When xscaling is set in the GGUF, multiply the pre-encoder output
    // by sqrt(d_model) before feeding it to the first Conformer block.
    // This matches the NeMo `self.xscaling` path that ships with
    // stt_*_fastconformer_ctc_* releases. canary_ctc aligner has this
    // flag cleared, so the op is elided and the bit-identical legacy
    // path is preserved.
    if (hp.xscaling) {
        const float xscale = std::sqrt((float)hp.d_model);
        cur = ggml_scale(ctx0, cur, xscale);
    }

    // ----- Sinusoidal rel-pos table -----
    ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int)hp.d_model, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // ----- N × FastConformer block (biases optional — nullptr bias
    // fields skip the ggml_add inside core_conformer::build_block) -----
    core_conformer::BlockParams bp = {
        (int)hp.d_model, (int)hp.n_heads, (int)hp.head_dim,
        (int)hp.conv_kernel, kLayerNormEps,
    };
    for (uint32_t il = 0; il < hp.n_layers; il++) {
        cur = core_conformer::build_block(ctx0, cur, pos_enc, T, m.enc[il], bp);
    }

    // CTC head: linear (d_model → vocab_total)
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.ctc_w, cur), m.ctc_b);
    ggml_set_name(cur, "ctc_logits");
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// ===========================================================================
// BatchNorm folding (load-time, once)
// ===========================================================================

static void cc_fold_batchnorm(cc_model & model) {
    const int d   = (int)model.hparams.d_model;
    const int K   = (int)model.hparams.conv_kernel;
    const float eps = 1e-5f;

    for (uint32_t il = 0; il < model.hparams.n_layers; il++) {
        auto & e = model.enc[il];
        if (!e.conv_dw_w || !e.conv_dw_b ||
            !e.conv_bn_w || !e.conv_bn_b || !e.conv_bn_rm || !e.conv_bn_rv) continue;

        std::vector<float> bn_mean(d), bn_var(d), bn_w(d), bn_b(d);
        ggml_backend_tensor_get(e.conv_bn_rm, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_rv, bn_var .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_w,  bn_w   .data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(e.conv_bn_b,  bn_b   .data(), 0, d * sizeof(float));

        std::vector<float> s(d);
        for (int c = 0; c < d; c++) s[c] = bn_w[c] / sqrtf(bn_var[c] + eps);

        std::vector<ggml_fp16_t> w_f16((size_t)K * d);
        ggml_backend_tensor_get(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));
        std::vector<float> w_f32((size_t)K * d);
        for (size_t i = 0; i < w_f16.size(); i++) w_f32[i] = ggml_fp16_to_fp32(w_f16[i]);
        for (int c = 0; c < d; c++)
            for (int ki = 0; ki < K; ki++)
                w_f32[ki + c * K] *= s[c];
        for (size_t i = 0; i < w_f16.size(); i++) w_f16[i] = ggml_fp32_to_fp16(w_f32[i]);
        ggml_backend_tensor_set(e.conv_dw_w, w_f16.data(), 0, w_f16.size() * sizeof(ggml_fp16_t));

        // Read the original depthwise-conv bias (the pretrained value)
        // BEFORE overwriting it — NeMo's FastConformer Conformer conv
        // module learns a bias on the depthwise conv, so the pre-BN
        // output is `conv(x) + orig_dw_b`. Folding BN into the conv
        // gives:
        //   y = scale * (conv(x) + orig_dw_b) + shift
        //     = scale * conv(x) + (scale * orig_dw_b + shift)
        //
        // canary_ctc aligner has synthetic zero orig_dw_b so the term
        // vanishes, but for stt_en_fastconformer_ctc_large (and any
        // other NeMo FC-CTC release with biased conv.depthwise_conv)
        // dropping the orig_dw_b contribution silently zeros a learned
        // offset and collapses the encoder output across time.
        std::vector<float> orig_dw_b(d);
        ggml_backend_tensor_get(e.conv_dw_b, orig_dw_b.data(), 0, d * sizeof(float));

        std::vector<float> dw_b(d);
        for (int c = 0; c < d; c++) {
            const float shift = -bn_mean[c] * s[c] + bn_b[c];
            dw_b[c] = s[c] * orig_dw_b[c] + shift;
        }
        ggml_backend_tensor_set(e.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
    }
    fprintf(stderr, "canary_ctc: BN folded into conv_dw weights for %u layers\n",
            model.hparams.n_layers);
}

// ===========================================================================
// Model loading
// ===========================================================================

static bool cc_load_model(cc_model & model, canary_ctc_vocab & vocab,
                          const char * path, ggml_backend_t backend)
{
    {
        gguf_context * gctx = core_gguf::open_metadata(path);
        if (!gctx) return false;

        auto & hp = model.hparams;
        hp.sample_rate          = core_gguf::kv_u32(gctx, "canary_ctc.sample_rate",          hp.sample_rate);
        hp.n_mels               = core_gguf::kv_u32(gctx, "canary_ctc.n_mels",               hp.n_mels);
        hp.n_fft                = core_gguf::kv_u32(gctx, "canary_ctc.n_fft",                hp.n_fft);
        hp.win_length           = core_gguf::kv_u32(gctx, "canary_ctc.win_length",           hp.win_length);
        hp.hop_length           = core_gguf::kv_u32(gctx, "canary_ctc.hop_length",           hp.hop_length);
        hp.d_model              = core_gguf::kv_u32(gctx, "canary_ctc.d_model",              hp.d_model);
        hp.n_layers             = core_gguf::kv_u32(gctx, "canary_ctc.n_layers",             hp.n_layers);
        hp.n_heads              = core_gguf::kv_u32(gctx, "canary_ctc.n_heads",              hp.n_heads);
        hp.head_dim             = core_gguf::kv_u32(gctx, "canary_ctc.head_dim",             hp.head_dim);
        hp.ff_dim               = core_gguf::kv_u32(gctx, "canary_ctc.ff_dim",               hp.ff_dim);
        hp.subsampling_factor   = core_gguf::kv_u32(gctx, "canary_ctc.subsampling_factor",   hp.subsampling_factor);
        hp.subsampling_channels = core_gguf::kv_u32(gctx, "canary_ctc.subsampling_channels", hp.subsampling_channels);
        hp.conv_kernel          = core_gguf::kv_u32(gctx, "canary_ctc.conv_kernel",          hp.conv_kernel);
        hp.vocab_size           = core_gguf::kv_u32(gctx, "canary_ctc.vocab_size",           hp.vocab_size);
        hp.blank_id             = core_gguf::kv_u32(gctx, "canary_ctc.blank_id",             hp.blank_id);
        hp.frame_dur_cs         = core_gguf::kv_u32(gctx, "canary_ctc.frame_dur_cs",         hp.frame_dur_cs);
        hp.xscaling             = core_gguf::kv_u32(gctx, "canary_ctc.xscaling",             hp.xscaling);

        auto tokens = core_gguf::kv_str_array(gctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
        }

        core_gguf::free_metadata(gctx);
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "canary_ctc", wl)) {
        return false;
    }
    model.ctx     = wl.ctx;
    model.buf     = wl.buf;
    model.tensors = std::move(wl.tensors);

    // Bind tensors
    model.mel_fb     = cc_try_get(model, "preprocessor.fb");
    model.mel_window = cc_try_get(model, "preprocessor.window");

    model.pre_encode.conv0_w = cc_require(model, "encoder.pre.conv.0.weight");
    model.pre_encode.conv0_b = cc_require(model, "encoder.pre.conv.0.bias");
    model.pre_encode.conv2_w = cc_require(model, "encoder.pre.conv.2.weight");
    model.pre_encode.conv2_b = cc_require(model, "encoder.pre.conv.2.bias");
    model.pre_encode.conv3_w = cc_require(model, "encoder.pre.conv.3.weight");
    model.pre_encode.conv3_b = cc_require(model, "encoder.pre.conv.3.bias");
    model.pre_encode.conv5_w = cc_require(model, "encoder.pre.conv.5.weight");
    model.pre_encode.conv5_b = cc_require(model, "encoder.pre.conv.5.bias");
    model.pre_encode.conv6_w = cc_require(model, "encoder.pre.conv.6.weight");
    model.pre_encode.conv6_b = cc_require(model, "encoder.pre.conv.6.bias");
    model.pre_encode.out_w   = cc_require(model, "encoder.pre.out.weight");
    model.pre_encode.out_b   = cc_require(model, "encoder.pre.out.bias");

    model.enc.resize(model.hparams.n_layers);
    for (uint32_t i = 0; i < model.hparams.n_layers; i++) {
        char buf[128];
        auto & e = model.enc[i];
        auto get = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return cc_require(model, buf);
        };
        auto get_opt = [&](const char * suf) {
            snprintf(buf, sizeof(buf), "encoder.layers.%u.%s", i, suf);
            return cc_try_get(model, buf);
        };

        e.norm_ff1_w  = get("norm_ff1.weight"); e.norm_ff1_b = get("norm_ff1.bias");
        e.ff1_l1_w    = get("ff1.linear1.weight");
        e.ff1_l1_b    = get_opt("ff1.linear1.bias");  // optional — present for canary/stt, absent for canary_ctc aligner
        e.ff1_l2_w    = get("ff1.linear2.weight");
        e.ff1_l2_b    = get_opt("ff1.linear2.bias");
        e.norm_attn_w = get("norm_attn.weight"); e.norm_attn_b = get("norm_attn.bias");
        e.attn_q_w    = get("attn.q.weight");
        e.attn_q_b    = get_opt("attn.q.bias");
        e.attn_k_w    = get("attn.k.weight");
        e.attn_k_b    = get_opt("attn.k.bias");
        e.attn_v_w    = get("attn.v.weight");
        e.attn_v_b    = get_opt("attn.v.bias");
        e.attn_out_w  = get("attn.out.weight");
        e.attn_out_b  = get_opt("attn.out.bias");
        e.attn_pos_w  = get("attn.pos.weight");
        e.pos_bias_u  = get("attn.pos_bias_u");
        e.pos_bias_v  = get("attn.pos_bias_v");
        e.norm_conv_w = get("norm_conv.weight"); e.norm_conv_b = get("norm_conv.bias");
        e.conv_pw1_w  = get("conv.pw1.weight");
        e.conv_pw1_b  = get_opt("conv.pw1.bias");
        e.conv_dw_w   = get("conv.dw.weight");
        e.conv_dw_b   = get("conv.dw.bias");
        e.conv_pw2_w  = get("conv.pw2.weight");
        e.conv_pw2_b  = get_opt("conv.pw2.bias");
        e.conv_bn_w   = get("conv.bn.weight"); e.conv_bn_b = get("conv.bn.bias");
        e.conv_bn_rm  = get("conv.bn.running_mean");
        e.conv_bn_rv  = get("conv.bn.running_var");
        e.norm_ff2_w  = get("norm_ff2.weight"); e.norm_ff2_b = get("norm_ff2.bias");
        e.ff2_l1_w    = get("ff2.linear1.weight");
        e.ff2_l1_b    = get_opt("ff2.linear1.bias");
        e.ff2_l2_w    = get("ff2.linear2.weight");
        e.ff2_l2_b    = get_opt("ff2.linear2.bias");
        e.norm_out_w  = get("norm_out.weight"); e.norm_out_b = get("norm_out.bias");
    }

    model.ctc_w = cc_require(model, "ctc.weight");
    model.ctc_b = cc_require(model, "ctc.bias");

    fprintf(stderr,
        "canary_ctc: vocab=%u  d_model=%u  n_layers=%u  n_heads=%u  ff=%u\n",
        model.hparams.vocab_size, model.hparams.d_model,
        model.hparams.n_layers, model.hparams.n_heads, model.hparams.ff_dim);
    return true;
}

// ===========================================================================
// Backend selection
// ===========================================================================

static ggml_backend_t cc_pick_backend() {
    ggml_backend_t b = ggml_backend_init_best();
    return b ? b : ggml_backend_cpu_init();
}

// ===========================================================================
// Public C API
// ===========================================================================

extern "C" struct canary_ctc_context_params canary_ctc_context_default_params(void) {
    canary_ctc_context_params p = {};
    p.n_threads = std::min(4, (int)std::thread::hardware_concurrency());
    p.verbosity = 1;
    return p;
}

extern "C" struct canary_ctc_context * canary_ctc_init_from_file(
    const char * path_model, struct canary_ctc_context_params params)
{
    auto * ctx = new canary_ctc_context();
    ctx->params    = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    ctx->backend = cc_pick_backend();
    // Always have a CPU fallback backend for ops the primary doesn't support.
    if (ggml_backend_is_cpu(ctx->backend)) {
        ctx->backend_cpu = ctx->backend;
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    } else {
        ctx->backend_cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    }

    if (!cc_load_model(ctx->model, ctx->vocab, path_model, ctx->backend)) {
        canary_ctc_free(ctx);
        return nullptr;
    }
    cc_fold_batchnorm(ctx->model);
    return ctx;
}

extern "C" void canary_ctc_free(struct canary_ctc_context * ctx) {
    if (!ctx) return;
    if (ctx->sched)             ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf)         ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)         ggml_free(ctx->model.ctx);
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend)
        ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend)           ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" int canary_ctc_n_vocab(struct canary_ctc_context * ctx)      { return (int)ctx->model.hparams.vocab_size; }
extern "C" int canary_ctc_blank_id(struct canary_ctc_context * ctx)     { return (int)ctx->model.hparams.blank_id; }
extern "C" int canary_ctc_frame_dur_cs(struct canary_ctc_context * ctx) { return (int)ctx->model.hparams.frame_dur_cs; }
extern "C" int canary_ctc_n_mels(struct canary_ctc_context * ctx)       { return (int)ctx->model.hparams.n_mels; }
extern "C" int canary_ctc_sample_rate(struct canary_ctc_context * ctx)  { return (int)ctx->model.hparams.sample_rate; }

extern "C" float * canary_ctc_compute_mel_debug(
    struct canary_ctc_context * ctx,
    const float * samples, int n_samples,
    int * out_n_mels, int * out_T_mel)
{
    int T_mel = 0;
    auto mel = cc_compute_mel(ctx, samples, n_samples, T_mel);
    if (mel.empty()) return nullptr;
    const int n_mels = (int)ctx->model.hparams.n_mels;
    float * out = (float *)std::malloc(mel.size() * sizeof(float));
    if (!out) return nullptr;
    std::memcpy(out, mel.data(), mel.size() * sizeof(float));
    if (out_n_mels) *out_n_mels = n_mels;
    if (out_T_mel)  *out_T_mel  = T_mel;
    return out;
}

// Debug: feed a pre-computed mel spectrogram straight into the encoder
// graph, bypassing cc_compute_mel. Used for diffing our mel pipeline
// against a NeMo reference without changing the encoder graph.
// `mel` is (n_mels, T_mel) row-major = TimeMels layout.
extern "C" int canary_ctc_compute_logits_from_mel_debug(
    struct canary_ctc_context * ctx,
    const float * mel, int T_mel,
    float ** out_logits, int * out_T_enc, int * out_vocab_total)
{
    if (!ctx->sched) {
        int n_backends = 1;
        ggml_backend_t backends[2] = { ctx->backend, nullptr };
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
            backends[1] = ctx->backend_cpu;
            n_backends = 2;
        }
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_backends, 16384, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(
            ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }
    ggml_cgraph * gf = cc_build_graph(ctx, T_mel);
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return -2;

    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel, 0,
                            (size_t)ctx->model.hparams.n_mels * T_mel * sizeof(float));

    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return -3;

    ggml_tensor * out = ggml_graph_get_tensor(gf, "ctc_logits");
    if (!out) return -4;
    const int V  = (int)out->ne[0];
    const int Te = (int)out->ne[1];
    *out_T_enc       = Te;
    *out_vocab_total = V;
    *out_logits      = (float *)malloc((size_t)V * Te * sizeof(float));
    ggml_backend_tensor_get(out, *out_logits, 0, (size_t)V * Te * sizeof(float));
    return 0;
}

extern "C" int canary_ctc_compute_logits(struct canary_ctc_context * ctx,
                                         const float * samples, int n_samples,
                                         float ** out_logits,
                                         int   * out_T_enc,
                                         int   * out_vocab_total)
{
    int T_mel = 0;
    auto mel = cc_compute_mel(ctx, samples, n_samples, T_mel);
    if (mel.empty()) return -1;

    if (!ctx->sched) {
        int n_backends = 1;
        ggml_backend_t backends[2] = { ctx->backend, nullptr };
        if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
            backends[1] = ctx->backend_cpu;
            n_backends = 2;
        }
        ctx->sched = ggml_backend_sched_new(backends, nullptr, n_backends, 16384, false, false);
    }
    if (ctx->compute_meta.empty()) {
        ctx->compute_meta.resize(
            ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));
    }

    ggml_cgraph * gf = cc_build_graph(ctx, T_mel);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) return -2;

    ggml_tensor * mel_in = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_in, mel.data(), 0,
                            (size_t)ctx->model.hparams.n_mels * T_mel * sizeof(float));

    ggml_tensor * pos_in = ggml_graph_get_tensor(gf, "pos_enc");
    int T_enc = (int)pos_in->ne[1];
    T_enc = (T_enc + 1) / 2;
    auto pe = core_conformer::make_pos_enc((int)ctx->model.hparams.d_model, T_enc);
    ggml_backend_tensor_set(pos_in, pe.data(), 0, pe.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) return -3;

    ggml_tensor * out = ggml_graph_get_tensor(gf, "ctc_logits");
    if (!out) return -4;

    const int V  = (int)out->ne[0];
    const int Te = (int)out->ne[1];

    *out_T_enc       = Te;
    *out_vocab_total = V;
    *out_logits      = (float *)malloc((size_t)V * Te * sizeof(float));
    ggml_backend_tensor_get(out, *out_logits, 0, (size_t)V * Te * sizeof(float));
    return 0;
}

// ---------------------------------------------------------------------------
// Greedy CTC decode (sanity check)
// ---------------------------------------------------------------------------

extern "C" char * canary_ctc_greedy_decode(struct canary_ctc_context * ctx,
                                           const float * logits, int T_enc, int V)
{
    const int blank = (int)ctx->model.hparams.blank_id;
    std::string result;
    int prev = -1;
    for (int t = 0; t < T_enc; t++) {
        const float * lv = logits + (size_t)t * V;
        int best = (int)(std::max_element(lv, lv + V) - lv);
        if (best != prev) {
            if (best != blank && best < (int)ctx->vocab.id_to_token.size()) {
                const std::string & tok = ctx->vocab.id_to_token[best];
                if (tok.size() >= 3 &&
                    (unsigned char)tok[0] == 0xE2 &&
                    (unsigned char)tok[1] == 0x96 &&
                    (unsigned char)tok[2] == 0x81) {
                    result += ' '; result += tok.substr(3);
                } else if (tok.size() >= 2 && tok[0] == '<' && tok.back() == '>') {
                    // skip special tokens
                } else {
                    result += tok;
                }
            }
            prev = best;
        }
    }
    auto lo = result.find_first_not_of(' ');
    auto hi = result.find_last_not_of(' ');
    std::string trimmed = (lo == std::string::npos) ? "" : result.substr(lo, hi - lo + 1);
    return strdup(trimmed.c_str());
}

// ---------------------------------------------------------------------------
// Subword tokenisation: SentencePiece-style greedy longest-prefix match
// ---------------------------------------------------------------------------

// Build a per-word token sequence by greedy longest-prefix matching against
// the vocab. SentencePiece convention: words are prefixed with U+2581 (▁).
// `out` receives token IDs in order; returns true on success.
static bool tokenise_word(const std::string & word,
                          const canary_ctc_vocab & vocab,
                          std::vector<int> & out)
{
    // The vocab tokens look like ["▁the", "▁of", "ing", "ed", "<unk>", ...]
    // For a word "Americans" we want something like ["▁Americ", "ans"] or
    // similar — start with the ▁-prefixed form and then continue with
    // continuations.
    const std::string boundary = "\xE2\x96\x81"; // U+2581 ▁
    std::string remaining = boundary + word;

    // Lowercase doesn't really matter for SentencePiece (it preserves case),
    // but we keep the original.
    while (!remaining.empty()) {
        // Find the longest prefix of `remaining` that's in the vocab
        int best_len = 0;
        int best_id  = -1;
        // Limit to ~32 byte prefixes (SentencePiece pieces are typically short)
        const int max_prefix = std::min<int>(remaining.size(), 32);
        for (int len = max_prefix; len >= 1; len--) {
            std::string prefix = remaining.substr(0, len);
            auto it = vocab.token_to_id.find(prefix);
            if (it != vocab.token_to_id.end()) {
                best_len = len;
                best_id  = it->second;
                break;
            }
        }
        if (best_id < 0) {
            // No match — fall back to a single byte. If even single bytes
            // aren't in the vocab, give up on this word.
            // (This is rare for a SentencePiece BPE on Latin/Cyrillic text.)
            std::string one(1, remaining[0]);
            auto it = vocab.token_to_id.find(one);
            if (it == vocab.token_to_id.end()) {
                // Try <unk>
                it = vocab.token_to_id.find("<unk>");
                if (it == vocab.token_to_id.end()) return false;
            }
            out.push_back(it->second);
            remaining = remaining.substr(1);
        } else {
            out.push_back(best_id);
            remaining = remaining.substr(best_len);
        }
    }
    return !out.empty();
}

// ---------------------------------------------------------------------------
// CTC forced alignment via Viterbi DP
//
// Same algorithm as src/align.cpp's ctc_forced_align but works on subword
// token IDs instead of characters. The DP runs over a CTC-expanded label
// sequence: [blank, t0, blank, t1, ..., blank, t_{N-1}, blank].
// ---------------------------------------------------------------------------

static int argmax_int(const std::vector<float> & v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

extern "C" int canary_ctc_align_words(struct canary_ctc_context * ctx,
                                      const float * logits,
                                      int           T_enc,
                                      int           V,
                                      const char ** words,
                                      int           n_words,
                                      struct canary_ctc_word * out_words)
{
    if (T_enc <= 0 || V <= 0 || n_words <= 0) return -1;
    const int blank_id = (int)ctx->model.hparams.blank_id;
    const int frame_dur_cs = (int)ctx->model.hparams.frame_dur_cs;

    // ----- 1. Tokenise each word and build per-word token ranges in `chars` -----
    struct word_range { int cs, ce; };
    std::vector<int>        chars;
    std::vector<word_range> wranges(n_words);

    for (int wi = 0; wi < n_words; wi++) {
        int cs = (int)chars.size();
        std::vector<int> tok_ids;
        bool ok = words[wi] ? tokenise_word(words[wi], ctx->vocab, tok_ids) : false;
        if (!ok || tok_ids.empty()) {
            wranges[wi] = {-1, -1};
            // Still copy the word text into the output for the caller
        } else {
            for (int t : tok_ids) chars.push_back(t);
            wranges[wi] = {cs, (int)chars.size() - 1};
        }
        // Pre-fill output text
        size_t L = std::min(strlen(words[wi] ? words[wi] : ""), sizeof(out_words[wi].text) - 1);
        memcpy(out_words[wi].text, words[wi] ? words[wi] : "", L);
        out_words[wi].text[L] = '\0';
        out_words[wi].t0 = 0;
        out_words[wi].t1 = 0;
    }

    int N = (int)chars.size();
    if (N == 0) return -2;

    // ----- 2. CTC-expanded label sequence with blanks between every label -----
    int S = 2 * N + 1;
    std::vector<int> seq(S);
    for (int j = 0; j < S; j++)
        seq[j] = (j % 2 == 0) ? blank_id : chars[j / 2];

    // ----- 3. Log-softmax over the CTC logits per frame -----
    std::vector<float> lp((size_t)T_enc * V);
    for (int t = 0; t < T_enc; t++) {
        const float * src = logits + (size_t)t * V;
        float * dst = lp.data() + (size_t)t * V;
        float mx = *std::max_element(src, src + V);
        float s = 0.f;
        for (int i = 0; i < V; i++) { dst[i] = src[i] - mx; s += expf(dst[i]); }
        float ls = logf(std::max(s, 1e-30f));
        for (int i = 0; i < V; i++) dst[i] -= ls;
    }

    // ----- 4. Viterbi DP -----
    const float NEG_INF = -1e30f;
    std::vector<float> alpha(S, NEG_INF), alpha_next(S);
    std::vector<std::vector<int8_t>> back(T_enc, std::vector<int8_t>(S, 0));

    // t = 0
    {
        const float * lp0 = lp.data();
        alpha[0] = lp0[seq[0]];
        if (S > 1) alpha[1] = lp0[seq[1]];
    }

    for (int t = 1; t < T_enc; t++) {
        const float * lpt = lp.data() + (size_t)t * V;
        std::fill(alpha_next.begin(), alpha_next.end(), NEG_INF);

        for (int j = 0; j < S; j++) {
            int tok = seq[j];
            float best = alpha[j]; int8_t bsrc = 0;
            if (j >= 1 && alpha[j-1] > best) { best = alpha[j-1]; bsrc = 1; }
            if (j >= 2 && tok != blank_id && seq[j] != seq[j-2] && alpha[j-2] > best) {
                best = alpha[j-2]; bsrc = 2;
            }
            if (best > NEG_INF) {
                alpha_next[j] = best + lpt[tok];
                back[t][j] = bsrc;
            }
        }
        alpha.swap(alpha_next);
    }

    // ----- 5. Traceback -----
    std::vector<int> path(T_enc);
    int j_cur = (alpha[S-1] >= alpha[S-2]) ? S-1 : S-2;
    for (int t = T_enc-1; t >= 0; t--) {
        path[t] = j_cur;
        if (t > 0) {
            switch (back[t][j_cur]) {
                case 0: break;
                case 1: j_cur--; break;
                case 2: j_cur -= 2; break;
            }
        }
    }

    // ----- 6. Map path → per-word t0/t1 -----
    for (int wi = 0; wi < n_words; wi++) {
        if (wranges[wi].cs < 0) continue;
        int es0 = 2 * wranges[wi].cs + 1;
        int es1 = 2 * wranges[wi].ce + 1;
        int t0_frame = -1, t1_frame = -1;
        for (int t = 0; t < T_enc; t++) {
            int j = path[t];
            if (j >= es0 && j <= es1) {
                if (t0_frame < 0) t0_frame = t;
                t1_frame = t;
            }
        }
        if (t0_frame >= 0) {
            out_words[wi].t0 = (int64_t)t0_frame      * frame_dur_cs;
            out_words[wi].t1 = (int64_t)(t1_frame + 1) * frame_dur_cs;
        }
    }
    return 0;
}
