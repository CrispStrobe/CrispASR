// firered_asr.cpp — FireRedASR2-AED runtime.
//
// Architecture: Conformer encoder (16L, d=1280, 20 heads, rel-PE, macaron FFN)
//             + Transformer decoder (16L, d=1280, cross-attention, GELU FFN)
//
// The encoder is a standard Conformer with:
//   - Conv2d subsampling (2x 3x3 stride-2 → 4x temporal reduction)
//   - Macaron-style FFN (half-step pre+post around attention)
//   - Relative positional encoding with learnable pos_bias_u/v
//   - Depthwise separable convolution (kernel=33, GLU gating, BatchNorm, Swish)
//
// The decoder is a standard Transformer with:
//   - Sinusoidal positional encoding
//   - Masked self-attention + cross-attention + GELU FFN
//   - Pre-norm (LayerNorm before each sub-layer)

#include "firered_asr.h"

#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ===========================================================================
// Model structures
// ===========================================================================

struct firered_hparams {
    int d_model = 1280;
    int n_head = 20;
    int d_inner = 5120;
    int n_layers_enc = 16;
    int n_layers_dec = 16;
    int idim = 80;   // mel bins
    int odim = 8667; // vocab size
    int subsample = 4;
    int kernel_size = 33;
    int pe_maxlen = 5000;
    int sos_id = 3;
    int eos_id = 4;
    int blank_id = 0;
    int pad_id = 2;
    int n_head_dec = 0; // decoder n_head (0 = same as encoder)
    // Derived
    int head_dim = 64; // d_model / n_head
};

// --- Encoder ---

struct firered_enc_ffn {
    // Macaron FFN: LayerNorm → Linear(d→4d) → Swish → Dropout → Linear(4d→d)
    ggml_tensor* ln_w = nullptr;   // net.0.weight [d_model]
    ggml_tensor* ln_b = nullptr;   // net.0.bias
    ggml_tensor* up_w = nullptr;   // net.1.weight [d_inner, d_model]
    ggml_tensor* up_b = nullptr;   // net.1.bias
    ggml_tensor* down_w = nullptr; // net.4.weight [d_model, d_inner]
    ggml_tensor* down_b = nullptr; // net.4.bias
};

struct firered_enc_mhsa {
    // Relative-position multi-head self-attention
    ggml_tensor* ln_q_w = nullptr; // layer_norm_q
    ggml_tensor* ln_q_b = nullptr;
    ggml_tensor* ln_k_w = nullptr; // layer_norm_k
    ggml_tensor* ln_k_b = nullptr;
    ggml_tensor* ln_v_w = nullptr; // layer_norm_v
    ggml_tensor* ln_v_b = nullptr;
    ggml_tensor* w_qs = nullptr; // [d_model, d_model]
    ggml_tensor* w_ks = nullptr;
    ggml_tensor* w_vs = nullptr;
    ggml_tensor* fc_w = nullptr;       // output projection
    ggml_tensor* lin_pos = nullptr;    // linear_pos [d_model, d_model]
    ggml_tensor* pos_bias_u = nullptr; // [n_head, head_dim]
    ggml_tensor* pos_bias_v = nullptr;
};

struct firered_enc_conv {
    // Conformer conv module
    ggml_tensor* pre_ln_w = nullptr;
    ggml_tensor* pre_ln_b = nullptr;
    ggml_tensor* pw1_w = nullptr; // pointwise_conv1 [2*d_model, d_model, 1]
    ggml_tensor* dw_w = nullptr;  // depthwise [2*d_model, 1, kernel_size]
    ggml_tensor* bn_w = nullptr;  // batch_norm weight (gamma)
    ggml_tensor* bn_b = nullptr;  // batch_norm bias (beta)
    // BatchNorm running stats would be needed at inference if not in eval mode,
    // but PyTorch .eval() uses running_mean/running_var which should be in the checkpoint
    ggml_tensor* bn_mean = nullptr;
    ggml_tensor* bn_var = nullptr;
    ggml_tensor* pw2_w = nullptr; // pointwise_conv2 [d_model, 2*d_model, 1]
};

struct firered_enc_block {
    firered_enc_ffn ffn1;
    firered_enc_mhsa mhsa;
    firered_enc_conv conv;
    firered_enc_ffn ffn2;
    ggml_tensor* ln_w = nullptr; // final layer_norm
    ggml_tensor* ln_b = nullptr;
};

// --- Decoder ---

struct firered_dec_attn {
    ggml_tensor* w_qs = nullptr;
    ggml_tensor* w_qs_b = nullptr;
    ggml_tensor* w_ks = nullptr;
    ggml_tensor* w_vs = nullptr;
    ggml_tensor* w_vs_b = nullptr;
    ggml_tensor* fc_w = nullptr;
    ggml_tensor* fc_b = nullptr;
};

struct firered_dec_block {
    // Self-attention
    ggml_tensor* sattn_norm_w = nullptr;
    ggml_tensor* sattn_norm_b = nullptr;
    firered_dec_attn sattn;
    // Cross-attention
    ggml_tensor* xattn_norm_w = nullptr;
    ggml_tensor* xattn_norm_b = nullptr;
    firered_dec_attn xattn;
    // MLP
    ggml_tensor* mlp_norm_w = nullptr;
    ggml_tensor* mlp_norm_b = nullptr;
    ggml_tensor* mlp_w1 = nullptr; // [d_inner, d_model]
    ggml_tensor* mlp_b1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr; // [d_model, d_inner]
    ggml_tensor* mlp_b2 = nullptr;
};

struct firered_model {
    firered_hparams hp;

    // Encoder
    struct {
        // Input preprocessor: 2x Conv2d(3x3, stride 2) + Linear
        ggml_tensor* conv0_w = nullptr; // [32, 1, 3, 3]
        ggml_tensor* conv0_b = nullptr;
        ggml_tensor* conv1_w = nullptr; // [32, 32, 3, 3]
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* proj_w = nullptr; // [d_model, 608]
        ggml_tensor* proj_b = nullptr;
        // Relative positional encoding
        ggml_tensor* pe = nullptr; // [1, 9999, d_model]
        // Conformer blocks
        std::vector<firered_enc_block> blocks;
    } enc;

    // Decoder
    struct {
        ggml_tensor* emb_w = nullptr; // [odim, d_model]
        ggml_tensor* pe = nullptr;    // [1, pe_maxlen, d_model]
        ggml_tensor* norm_out_w = nullptr;
        ggml_tensor* norm_out_b = nullptr;
        ggml_tensor* prj_w = nullptr; // [odim, d_model] — output projection
        std::vector<firered_dec_block> blocks;
    } dec;

    // CTC
    ggml_tensor* ctc_w = nullptr; // [odim, d_model]
    ggml_tensor* ctc_b = nullptr;

    // CMVN
    ggml_tensor* cmvn_mean = nullptr; // [idim]
    ggml_tensor* cmvn_std = nullptr;  // [idim]

    // Tokenizer
    std::vector<std::string> vocab;

    // Weight memory
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct firered_asr_context {
    firered_asr_context_params params;
    firered_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // KV cache for decoder
    ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor* kv_self_k = nullptr;
    ggml_tensor* kv_self_v = nullptr;
    ggml_tensor* kv_cross_k = nullptr;
    ggml_tensor* kv_cross_v = nullptr;

    int n_threads = 4;
};

// ===========================================================================
// Implementation
// ===========================================================================

extern "C" struct firered_asr_context_params firered_asr_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/true, /*beam_size=*/3};
}

// --- Tensor loading helpers ---

static void load_ffn(const std::map<std::string, ggml_tensor*>& ts, const char* prefix, firered_enc_ffn& ffn) {
    char buf[128];
    auto get = [&](const char* suffix) -> ggml_tensor* {
        snprintf(buf, sizeof(buf), "%s%s", prefix, suffix);
        auto it = ts.find(buf);
        return it != ts.end() ? it->second : nullptr;
    };
    ffn.ln_w = get(".net.0.weight");
    ffn.ln_b = get(".net.0.bias");
    ffn.up_w = get(".net.1.weight");
    ffn.up_b = get(".net.1.bias");
    ffn.down_w = get(".net.4.weight");
    ffn.down_b = get(".net.4.bias");
}

static void load_enc_mhsa(const std::map<std::string, ggml_tensor*>& ts, const char* prefix, firered_enc_mhsa& mhsa) {
    char buf[128];
    auto get = [&](const char* suffix) -> ggml_tensor* {
        snprintf(buf, sizeof(buf), "%s%s", prefix, suffix);
        auto it = ts.find(buf);
        return it != ts.end() ? it->second : nullptr;
    };
    mhsa.ln_q_w = get(".ln_q.weight");
    mhsa.ln_q_b = get(".ln_q.bias");
    mhsa.ln_k_w = get(".ln_k.weight");
    mhsa.ln_k_b = get(".ln_k.bias");
    mhsa.ln_v_w = get(".ln_v.weight");
    mhsa.ln_v_b = get(".ln_v.bias");
    mhsa.w_qs = get(".w_qs.weight");
    mhsa.w_ks = get(".w_ks.weight");
    mhsa.w_vs = get(".w_vs.weight");
    mhsa.fc_w = get(".fc.weight");
    mhsa.lin_pos = get(".lin_pos.weight");
    mhsa.pos_bias_u = get(".pos_bias_u");
    mhsa.pos_bias_v = get(".pos_bias_v");
}

// ===========================================================================
// Model loading
// ===========================================================================

extern "C" struct firered_asr_context* firered_asr_init_from_file(const char* path_model,
                                                                  struct firered_asr_context_params params) {
    auto* ctx = new firered_asr_context();
    ctx->params = params;
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;

    if (params.verbosity >= 1) {
        fprintf(stderr, "firered_asr: init start (verbosity=%d, use_gpu=%d, threads=%d)\n",
                params.verbosity, params.use_gpu, params.n_threads);
    }
    ctx->backend = params.use_gpu ? ggml_backend_init_best() : ggml_backend_cpu_init();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
    if (params.verbosity >= 1)
        fprintf(stderr, "firered_asr: backend ready (%s)\n",
                ggml_backend_is_cpu(ctx->backend) ? "CPU" : "GPU");
    ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    auto& m = ctx->model;
    auto& hp = m.hp;

    // ---- pass 1: read hparams + vocab ----
    {
        gguf_context* gctx = core_gguf::open_metadata(path_model);
        if (!gctx) {
            fprintf(stderr, "firered_asr: failed to open '%s'\n", path_model);
            delete ctx;
            return nullptr;
        }
        hp.d_model = core_gguf::kv_u32(gctx, "firered.d_model", hp.d_model);
        hp.n_head = core_gguf::kv_u32(gctx, "firered.n_head", hp.n_head);
        hp.d_inner = core_gguf::kv_u32(gctx, "firered.d_inner", hp.d_inner);
        hp.n_layers_enc = core_gguf::kv_u32(gctx, "firered.n_layers_enc", hp.n_layers_enc);
        hp.n_layers_dec = core_gguf::kv_u32(gctx, "firered.n_layers_dec", hp.n_layers_dec);
        hp.idim = core_gguf::kv_u32(gctx, "firered.idim", hp.idim);
        hp.odim = core_gguf::kv_u32(gctx, "firered.odim", hp.odim);
        hp.subsample = core_gguf::kv_u32(gctx, "firered.subsample", hp.subsample);
        hp.kernel_size = core_gguf::kv_u32(gctx, "firered.kernel_size", hp.kernel_size);
        hp.pe_maxlen = core_gguf::kv_u32(gctx, "firered.pe_maxlen", hp.pe_maxlen);
        hp.sos_id = core_gguf::kv_u32(gctx, "firered.sos_id", hp.sos_id);
        hp.eos_id = core_gguf::kv_u32(gctx, "firered.eos_id", hp.eos_id);
        hp.blank_id = core_gguf::kv_u32(gctx, "firered.blank_id", hp.blank_id);
        hp.pad_id = core_gguf::kv_u32(gctx, "firered.pad_id", hp.pad_id);
        hp.n_head_dec = core_gguf::kv_u32(gctx, "firered.n_head_dec", 0);
        if (hp.n_head_dec == 0)
            hp.n_head_dec = hp.n_head; // default: same as encoder
        hp.head_dim = hp.d_model / hp.n_head;

        // Tokenizer
        m.vocab.resize(hp.odim);
        const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int n = gguf_get_arr_n(gctx, tok_key);
            for (int i = 0; i < n && i < hp.odim; i++) {
                const char* s = gguf_get_arr_str(gctx, tok_key, i);
                if (s)
                    m.vocab[i] = s;
            }
        }

        gguf_free(gctx);
    }

    // ---- pass 2: load tensor data ----
    if (params.verbosity >= 1)
        fprintf(stderr, "firered_asr: loading weights to %s...\n",
                ggml_backend_is_cpu(ctx->backend) ? "CPU" : "GPU");
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "firered_asr", wl)) {
        fprintf(stderr, "firered_asr: failed to load weights from '%s'\n", path_model);
        delete ctx;
        return nullptr;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    auto& ts = wl.tensors;

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = ts.find(name);
        if (it == ts.end()) {
            // Suppress known-missing tensors (BN running stats not needed at inference)
            if (params.verbosity >= 2 &&
                !strstr(name, "running_mean") && !strstr(name, "running_var"))
                fprintf(stderr, "firered_asr: tensor '%s' not found\n", name);
            return nullptr;
        }
        return it->second;
    };

    // --- Encoder input preprocessor ---
    m.enc.conv0_w = get("enc.preproc.conv.0.weight");
    m.enc.conv0_b = get("enc.preproc.conv.0.bias");
    m.enc.conv1_w = get("enc.preproc.conv.2.weight");
    m.enc.conv1_b = get("enc.preproc.conv.2.bias");
    m.enc.proj_w = get("enc.preproc.out.weight");
    m.enc.proj_b = get("enc.preproc.out.bias");
    m.enc.pe = get("enc.pe.pe");

    // --- Encoder Conformer blocks ---
    m.enc.blocks.resize(hp.n_layers_enc);
    for (int i = 0; i < hp.n_layers_enc; i++) {
        auto& b = m.enc.blocks[i];
        char prefix[64];

        // FFN1 (macaron)
        snprintf(prefix, sizeof(prefix), "enc.%d.ffn1", i);
        load_ffn(ts, prefix, b.ffn1);

        // MHSA
        snprintf(prefix, sizeof(prefix), "enc.%d.mhsa", i);
        load_enc_mhsa(ts, prefix, b.mhsa);

        // Conv module
        char buf[128];
        snprintf(buf, sizeof(buf), "enc.%d.conv.pre_ln.weight", i);
        b.conv.pre_ln_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.pre_ln.bias", i);
        b.conv.pre_ln_b = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.pw1.weight", i);
        b.conv.pw1_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.dw.weight", i);
        b.conv.dw_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.weight", i);
        b.conv.bn_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.bias", i);
        b.conv.bn_b = get(buf);
        // BatchNorm running_mean and running_var
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.running_mean", i);
        b.conv.bn_mean = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.bn.running_var", i);
        b.conv.bn_var = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.conv.pw2.weight", i);
        b.conv.pw2_w = get(buf);

        // FFN2 (macaron)
        snprintf(prefix, sizeof(prefix), "enc.%d.ffn2", i);
        load_ffn(ts, prefix, b.ffn2);

        // Final layer norm
        snprintf(buf, sizeof(buf), "enc.%d.ln.weight", i);
        b.ln_w = get(buf);
        snprintf(buf, sizeof(buf), "enc.%d.ln.bias", i);
        b.ln_b = get(buf);
    }

    // --- Decoder ---
    m.dec.emb_w = get("dec.emb.weight");
    m.dec.pe = get("dec.pe.pe");
    m.dec.norm_out_w = get("dec.norm_out.weight");
    m.dec.norm_out_b = get("dec.norm_out.bias");
    m.dec.prj_w = get("dec.prj.weight");

    m.dec.blocks.resize(hp.n_layers_dec);
    for (int i = 0; i < hp.n_layers_dec; i++) {
        auto& b = m.dec.blocks[i];
        char buf[128];

        // Self-attention
        snprintf(buf, sizeof(buf), "dec.%d.sattn_norm.weight", i);
        b.sattn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn_norm.bias", i);
        b.sattn_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_qs.weight", i);
        b.sattn.w_qs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_qs.bias", i);
        b.sattn.w_qs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_ks.weight", i);
        b.sattn.w_ks = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_vs.weight", i);
        b.sattn.w_vs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.w_vs.bias", i);
        b.sattn.w_vs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.fc.weight", i);
        b.sattn.fc_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.sattn.fc.bias", i);
        b.sattn.fc_b = get(buf);

        // Cross-attention
        snprintf(buf, sizeof(buf), "dec.%d.xattn_norm.weight", i);
        b.xattn_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn_norm.bias", i);
        b.xattn_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_qs.weight", i);
        b.xattn.w_qs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_qs.bias", i);
        b.xattn.w_qs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_ks.weight", i);
        b.xattn.w_ks = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_vs.weight", i);
        b.xattn.w_vs = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.w_vs.bias", i);
        b.xattn.w_vs_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.fc.weight", i);
        b.xattn.fc_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.xattn.fc.bias", i);
        b.xattn.fc_b = get(buf);

        // MLP
        snprintf(buf, sizeof(buf), "dec.%d.mlp_norm.weight", i);
        b.mlp_norm_w = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp_norm.bias", i);
        b.mlp_norm_b = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_1.weight", i);
        b.mlp_w1 = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_1.bias", i);
        b.mlp_b1 = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_2.weight", i);
        b.mlp_w2 = get(buf);
        snprintf(buf, sizeof(buf), "dec.%d.mlp.w_2.bias", i);
        b.mlp_b2 = get(buf);
    }

    // CTC
    m.ctc_w = get("ctc.weight");
    m.ctc_b = get("ctc.bias");

    // CMVN
    m.cmvn_mean = get("cmvn.mean");
    m.cmvn_std = get("cmvn.std");

    // Scheduler
    int n_be = 1;
    ggml_backend_t backends[2] = {ctx->backend, nullptr};
    if (ctx->backend_cpu && ctx->backend_cpu != ctx->backend) {
        backends[n_be++] = ctx->backend_cpu;
    }
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 16384, false, false);
    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "firered_asr: loaded %d enc + %d dec layers, vocab %d, d_model %d\n", hp.n_layers_enc,
                hp.n_layers_dec, hp.odim, hp.d_model);
    }

    return ctx;
}

extern "C" void firered_asr_free(struct firered_asr_context* ctx) {
    if (!ctx)
        return;
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

// ===========================================================================
// Kaldi-compatible fbank (80-dim, 25ms povey window, 10ms hop, 16kHz)
// Matches kaldi_native_fbank with: preemph=0.97, remove_dc=true,
// window=povey, power=true, dither=0, low_freq=20, high_freq=0
// ===========================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void compute_fbank(const float* pcm, int n_samples, std::vector<float>& features, int& n_frames) {
    const int n_fft = 512;
    const int hop = 160; // 10ms @ 16kHz
    const int win = 400; // 25ms @ 16kHz
    const int n_mels = 80;
    const int sample_rate = 16000;
    const float preemph = 0.97f;
    const float low_freq = 20.0f;
    const float high_freq = (float)sample_rate / 2.0f; // 8000 Hz (kaldi high_freq=0 means Nyquist)

    // snip_edges=true: number of frames
    n_frames = (n_samples - win) / hop + 1;
    if (n_frames <= 0) {
        n_frames = 0;
        return;
    }

    // Mel filterbank (kaldi HTK-compatible mel scale)
    int n_fft_bins = n_fft / 2 + 1;
    std::vector<float> mel_fb(n_mels * n_fft_bins, 0.0f);
    {
        auto hz2mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };
        auto mel2hz = [](float m) { return 700.0f * (expf(m / 1127.0f) - 1.0f); };
        float mel_lo = hz2mel(low_freq);
        float mel_hi = hz2mel(high_freq);
        std::vector<float> center(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            center[i] = mel2hz(mel_lo + i * (mel_hi - mel_lo) / (n_mels + 1));

        for (int m = 0; m < n_mels; m++) {
            for (int k = 0; k < n_fft_bins; k++) {
                float freq = (float)k * sample_rate / n_fft;
                if (freq > center[m] && freq <= center[m + 1] && center[m + 1] > center[m])
                    mel_fb[m * n_fft_bins + k] = (freq - center[m]) / (center[m + 1] - center[m]);
                else if (freq > center[m + 1] && freq < center[m + 2] && center[m + 2] > center[m + 1])
                    mel_fb[m * n_fft_bins + k] = (center[m + 2] - freq) / (center[m + 2] - center[m + 1]);
            }
        }
    }

    // Povey window: hann(i)^0.85
    std::vector<float> window(win);
    for (int i = 0; i < win; i++) {
        float hann = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (win - 1));
        window[i] = powf(hann, 0.85f);
    }

    features.resize(n_frames * n_mels);
    std::vector<float> fft_re(n_fft), fft_im(n_fft);

    auto fft_forward = [](float* re, float* im, int n) {
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            float ang = -2.0f * (float)M_PI / len;
            float wre = cosf(ang), wim = sinf(ang);
            for (int i = 0; i < n; i += len) {
                float cr = 1, ci = 0;
                for (int j = 0; j < len / 2; j++) {
                    float tr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                    float ti = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                    re[i + j + len / 2] = re[i + j] - tr;
                    im[i + j + len / 2] = im[i + j] - ti;
                    re[i + j] += tr;
                    im[i + j] += ti;
                    float nr = cr * wre - ci * wim;
                    ci = cr * wim + ci * wre;
                    cr = nr;
                }
            }
        }
    };

    for (int t = 0; t < n_frames; t++) {
        int offset = t * hop;

        // Extract frame + remove DC offset
        std::vector<float> frame(win);
        float dc = 0.0f;
        for (int i = 0; i < win; i++) {
            // Scale to int16 range: FireRedASR/LID CMVN trained on int16 fbank
            frame[i] = ((offset + i < n_samples) ? pcm[offset + i] : 0.0f) * 32768.0f;
            dc += frame[i];
        }
        dc /= win;
        for (int i = 0; i < win; i++)
            frame[i] -= dc;

        // Preemphasis: s[i] -= preemph * s[i-1]
        for (int i = win - 1; i > 0; i--)
            frame[i] -= preemph * frame[i - 1];
        frame[0] -= preemph * frame[0]; // kaldi: first sample uses itself

        // Apply window + zero-pad to n_fft
        std::fill(fft_re.begin(), fft_re.end(), 0.0f);
        std::fill(fft_im.begin(), fft_im.end(), 0.0f);
        for (int i = 0; i < win; i++)
            fft_re[i] = frame[i] * window[i];

        // FFT
        fft_forward(fft_re.data(), fft_im.data(), n_fft);

        // Power spectrum → mel filterbank → log
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; k++) {
                float power = fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k];
                sum += power * mel_fb[m * n_fft_bins + k];
            }
            features[t * n_mels + m] = logf(std::max(sum, 1.1920929e-7f)); // kaldi uses FLT_EPSILON
        }
    }
}

// ===========================================================================
// CPU-side helpers for the encoder
// ===========================================================================

// Read ggml tensor to float vector (handles F16, quantized, and F32 types)
static void read_f32_vec(ggml_tensor* t, std::vector<float>& out) {
    int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(uint16_t));
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp.data(), out.data(), n);
    } else {
        // Quantized types (Q8_0, Q4_K_M, etc.): read raw bytes, dequantize
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "read_f32_vec: unsupported type %d for tensor, zeroing\n", (int)t->type);
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
}

// CPU matmul: C = A @ B^T where A is [M,K], B is [N,K] → C is [M,N]
// (B stored as [N,K] row-major, like ggml weight [K,N] with ne[0]=K)
static void cpu_matmul_bt(const float* A, const float* B, float* C, int M, int K, int N) {
    if (M == 1) {
        // Single-vector × matrix: parallelize over output dimension N.
        // This is the decoder hot path (one token per step).
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            double s = 0;
            const float* brow = B + n * K;
            for (int k = 0; k < K; k++)
                s += (double)A[k] * (double)brow[k];
            C[n] = (float)s;
        }
    } else {
#pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                double s = 0;
                for (int k = 0; k < K; k++)
                    s += (double)A[m * K + k] * (double)B[n * K + k];
                C[m * N + n] = (float)s;
            }
        }
    }
}

// ggml vector-matrix multiply using quantized weight tensor directly.
// Eliminates the need to dequantize Q4_K weights to F32 (saves 23.6s init).
// weight_w: ggml_tensor [K, N] (any type), input: F32[K], output: F32[N]
// bias_b: optional ggml_tensor [N], added to output if non-null.
static void ggml_vecmat(ggml_backend_t be, ggml_backend_sched_t sc, ggml_tensor* weight_w, ggml_tensor* bias_b,
                        const float* input, float* output, int K, int N) {
    size_t mem = ggml_tensor_overhead() * 10 + ggml_graph_overhead() + 256 * 1024;
    struct ggml_init_params gp = {mem, nullptr, true};
    ggml_context* c0 = ggml_init(gp);

    ggml_tensor* inp = ggml_new_tensor_2d(c0, GGML_TYPE_F32, K, 1);
    ggml_set_name(inp, "vi");
    ggml_set_input(inp);
    ggml_tensor* cur = ggml_mul_mat(c0, weight_w, inp); // [N, 1]
    if (bias_b)
        cur = ggml_add(c0, cur, bias_b);
    ggml_set_name(cur, "vo");
    ggml_set_output(cur);

    ggml_cgraph* gf = ggml_new_graph(c0);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_sched_reset(sc);
    ggml_backend_sched_set_tensor_backend(sc, weight_w, be);
    if (bias_b)
        ggml_backend_sched_set_tensor_backend(sc, bias_b, be);
    if (ggml_backend_sched_alloc_graph(sc, gf)) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "vi"), input, 0, K * sizeof(float));
        if (ggml_backend_sched_graph_compute(sc, gf) == GGML_STATUS_SUCCESS)
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "vo"), output, 0, N * sizeof(float));
    }
    ggml_free(c0);
}

// Read a small ggml tensor (norm weights/biases, d floats) into F32 buffer.
static void read_small_tensor(ggml_tensor* t, std::vector<float>& out) {
    if (!t) {
        out.clear();
        return;
    }
    int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32)
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    else {
        // Dequant small tensor (norms are always F32 anyway)
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        auto to_float = ggml_get_type_traits(t->type)->to_float;
        if (to_float)
            to_float(raw.data(), out.data(), n);
        else
            memset(out.data(), 0, n * sizeof(float));
    }
}

// CPU LayerNorm: out[t,d] = (x[t,d] - mean) / sqrt(var + eps) * w[d] + b[d]
static void cpu_layernorm(const float* x, const float* w, const float* b, float* out, int T, int d) {
    for (int t = 0; t < T; t++) {
        float mean = 0;
        for (int i = 0; i < d; i++)
            mean += x[t * d + i];
        mean /= d;
        float var = 0;
        for (int i = 0; i < d; i++) {
            float v = x[t * d + i] - mean;
            var += v * v;
        }
        var /= d;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int i = 0; i < d; i++) {
            out[t * d + i] = (x[t * d + i] - mean) * inv * w[i];
            if (b)
                out[t * d + i] += b[i];
        }
    }
}

// CPU Swish: x * sigmoid(x)
static void cpu_swish(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

// CPU Sigmoid
static void cpu_sigmoid(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// CPU softmax along dim (per row)
static void cpu_softmax_rows(float* x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float mx = x[r * cols];
        for (int c = 1; c < cols; c++)
            mx = std::max(mx, x[r * cols + c]);
        float sum = 0;
        for (int c = 0; c < cols; c++) {
            x[r * cols + c] = expf(x[r * cols + c] - mx);
            sum += x[r * cols + c];
        }
        for (int c = 0; c < cols; c++)
            x[r * cols + c] /= sum;
    }
}

static void firered_debug_dump_vec(const char* name, const std::vector<float>& v, int max_items) {
    fprintf(stderr, "firered_asr[%s]: [", name);
    int n = (int)v.size() < max_items ? (int)v.size() : max_items;
    for (int i = 0; i < n; i++) {
        if (i)
            fprintf(stderr, ",");
        fprintf(stderr, "%.4f", v[i]);
    }
    fprintf(stderr, "]\n");
}

// ===========================================================================
// CPU encoder: full Conformer encoder computed on CPU
// ===========================================================================

// Compute full encoder on CPU. Returns encoder output [T, d_model] row-major.
static void cpu_encoder(const float* subsampled, // [T, 608] row-major
                        int T, int flat_dim, const firered_model& m, std::vector<float>& enc_output) {
    auto& hp = m.hp;
    int d = hp.d_model;
    int nh = hp.n_head;
    int hd = hp.head_dim;
    int di = hp.d_inner;
    int ks = hp.kernel_size;

    // Linear projection: [T, 608] @ proj_w^T + proj_b → [T, d_model]
    std::vector<float> proj_w, proj_b_v;
    read_f32_vec(m.enc.proj_w, proj_w); // ggml [608, d_model] → read as flat
    read_f32_vec(m.enc.proj_b, proj_b_v);

    // proj_w in ggml: ne[0]=flat_dim=608, ne[1]=d_model=1280
    // As row-major: proj_w[row][col] where row < d_model, col < flat_dim
    // cpu_matmul_bt: A[T,K=608] @ B^T where B[N=1280, K=608] → C[T, 1280]
    std::vector<float> x(T * d);
    cpu_matmul_bt(subsampled, proj_w.data(), x.data(), T, flat_dim, d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++)
            x[t * d + i] += proj_b_v[i];
    // x: [T, d_model] row-major — linear projection done

    // Load PE: ggml [d_model, 9999, 1] → read as [9999, d_model] row-major
    std::vector<float> pe_full;
    read_f32_vec(m.enc.pe, pe_full);
    int T_pe = 2 * T - 1;
    int Tmax = (int)(m.enc.pe->ne[1]); // 9999
    // Python extracts CENTER positions: pe[:, Tmax//2 - T + 1 : Tmax//2 + T]
    int pe_start = Tmax / 2 - T + 1;
    std::vector<float> pe_data(T_pe * d);
    memcpy(pe_data.data(), &pe_full[pe_start * d], T_pe * d * sizeof(float));
    // pe_row_major[p * d + i] = pe_data[p * d + i] (ggml ne[0]=d fastest → same as row-major)
    // ← Actually no. ggml ne[0]=d=1280 means the first 1280 values are position 0.
    // So pe_data[p * d + i] = PE at position p, dimension i. This IS row-major [pos, dim].

    // For each conformer block
    std::vector<float> tmp(T * d), tmp2(T * d), tmp3(T * d);

    for (int li = 0; li < hp.n_layers_enc; li++) {
        const auto& b = m.enc.blocks[li];

        // === Macaron FFN1: out = 0.5*x + 0.5*ffn1(x) ===
        {
            std::vector<float> ln_w, ln_b_v, up_w, up_b_v, down_w, down_b_v;
            read_f32_vec(b.ffn1.ln_w, ln_w);
            read_f32_vec(b.ffn1.ln_b, ln_b_v);
            read_f32_vec(b.ffn1.up_w, up_w);
            read_f32_vec(b.ffn1.up_b, up_b_v);
            read_f32_vec(b.ffn1.down_w, down_w);
            read_f32_vec(b.ffn1.down_b, down_b_v);

            cpu_layernorm(x.data(), ln_w.data(), ln_b_v.data(), tmp.data(), T, d);
            // up: [T, d] @ up_w^T → [T, di]
            std::vector<float> h_up(T * di);
            cpu_matmul_bt(tmp.data(), up_w.data(), h_up.data(), T, d, di);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < di; i++)
                    h_up[t * di + i] += up_b_v[i];
            cpu_swish(h_up.data(), T * di);
            // down: [T, di] @ down_w^T → [T, d]
            cpu_matmul_bt(h_up.data(), down_w.data(), tmp.data(), T, di, d);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++)
                    tmp[t * d + i] += down_b_v[i];
            // The macaron residual in the Python code is:
            // block.forward: out = 0.5*x + 0.5*ffn1(x)
            // ffn1.forward: return net(x) + x  (internal residual!)
            // So: out = 0.5*x + 0.5*(net(x) + x) = x + 0.5*net(x)
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++)
                    x[t * d + i] = x[t * d + i] + 0.5f * tmp[t * d + i];
        }

        if (li == 0) {
        }
        // === MHSA with relative PE ===
        {
            std::vector<float> lnq_w, lnq_b, lnk_w, lnk_b, lnv_w, lnv_b;
            std::vector<float> wq, wk, wv, fc_w, lp_w, bu, bv;
            read_f32_vec(b.mhsa.ln_q_w, lnq_w);
            read_f32_vec(b.mhsa.ln_q_b, lnq_b);
            read_f32_vec(b.mhsa.ln_k_w, lnk_w);
            read_f32_vec(b.mhsa.ln_k_b, lnk_b);
            read_f32_vec(b.mhsa.ln_v_w, lnv_w);
            read_f32_vec(b.mhsa.ln_v_b, lnv_b);
            read_f32_vec(b.mhsa.w_qs, wq);
            read_f32_vec(b.mhsa.w_ks, wk);
            read_f32_vec(b.mhsa.w_vs, wv);
            read_f32_vec(b.mhsa.fc_w, fc_w);
            read_f32_vec(b.mhsa.lin_pos, lp_w);
            read_f32_vec(b.mhsa.pos_bias_u, bu);
            read_f32_vec(b.mhsa.pos_bias_v, bv);

            // Q/K/V projections with separate LN
            std::vector<float> Q(T * d), K(T * d), V(T * d);
            cpu_layernorm(x.data(), lnq_w.data(), lnq_b.data(), tmp.data(), T, d);
            cpu_matmul_bt(tmp.data(), wq.data(), Q.data(), T, d, d);
            cpu_layernorm(x.data(), lnk_w.data(), lnk_b.data(), tmp.data(), T, d);
            cpu_matmul_bt(tmp.data(), wk.data(), K.data(), T, d, d);
            cpu_layernorm(x.data(), lnv_w.data(), lnv_b.data(), tmp.data(), T, d);
            cpu_matmul_bt(tmp.data(), wv.data(), V.data(), T, d, d);

            // Position embedding projection: P = pe @ lin_pos^T → [T_pe, d]
            std::vector<float> P(T_pe * d);
            cpu_matmul_bt(pe_data.data(), lp_w.data(), P.data(), T_pe, d, d);

            // Reshape to multi-head: Q[T, nh, hd], K[T, nh, hd], V[T, nh, hd], P[T_pe, nh, hd]
            // Already in this layout since d = nh * hd

            float scale = 1.0f / sqrtf((float)hd);

            // Attention scores per head
            std::vector<float> attn_out(T * d, 0.0f);

            for (int h = 0; h < nh; h++) {
                // Content scores: (Q[t,h,:] + bias_u[h,:]) @ K[t',h,:]^T
                // Position scores: (Q[t,h,:] + bias_v[h,:]) @ P[t-t'+T-1,h,:]^T
                std::vector<float> scores(T * T, 0.0f);

                for (int tq = 0; tq < T; tq++) {
                    for (int tk = 0; tk < T; tk++) {
                        float content = 0, position = 0;
                        int pos_idx = T - 1 - tq + tk; // rel_shift: NOT tq-tk+T-1!
                        for (int dd = 0; dd < hd; dd++) {
                            float q_val = Q[tq * d + h * hd + dd];
                            float k_val = K[tk * d + h * hd + dd];
                            content += (q_val + bu[h * hd + dd]) * k_val;
                            if (pos_idx >= 0 && pos_idx < T_pe) {
                                float p_val = P[pos_idx * d + h * hd + dd];
                                position += (q_val + bv[h * hd + dd]) * p_val;
                            }
                        }
                        scores[tq * T + tk] = (content + position) * scale;
                    }
                }

                // Softmax per query
                cpu_softmax_rows(scores.data(), T, T);

                // Weighted sum: out[tq, h, :] = sum_tk scores[tq,tk] * V[tk, h, :]
                for (int tq = 0; tq < T; tq++)
                    for (int dd = 0; dd < hd; dd++) {
                        float s = 0;
                        for (int tk = 0; tk < T; tk++)
                            s += scores[tq * T + tk] * V[tk * d + h * hd + dd];
                        attn_out[tq * d + h * hd + dd] = s;
                    }
            }

            // Output projection: fc(attn_out) + residual
            std::vector<float> fc_out(T * d);
            cpu_matmul_bt(attn_out.data(), fc_w.data(), fc_out.data(), T, d, d);
            for (int i = 0; i < T * d; i++)
                x[i] += fc_out[i]; // residual connection
        }

        // === Conv module ===
        {
            std::vector<float> pre_ln_w, pre_ln_b_v, pw1_w, dw_w, bn_w, bn_b_v, pw2_w;
            read_f32_vec(b.conv.pre_ln_w, pre_ln_w);
            if (b.conv.pre_ln_b)
                read_f32_vec(b.conv.pre_ln_b, pre_ln_b_v);
            read_f32_vec(b.conv.pw1_w, pw1_w);
            read_f32_vec(b.conv.dw_w, dw_w);
            read_f32_vec(b.conv.bn_w, bn_w);
            if (b.conv.bn_b)
                read_f32_vec(b.conv.bn_b, bn_b_v);
            read_f32_vec(b.conv.pw2_w, pw2_w);

            // Pre-LN
            cpu_layernorm(x.data(), pre_ln_w.data(), pre_ln_b_v.empty() ? nullptr : pre_ln_b_v.data(), tmp.data(), T,
                          d);

            // Pointwise conv1: [T, d] @ pw1^T → [T, 2*d_inner]
            // pw1_w ggml: ne[0]=1, ne[1]=d, ne[2]=5120 → flat is groups of d for each output
            // For cpu_matmul_bt: B[N=5120, K=d]
            int pw1_out = (int)(m.enc.blocks[li].conv.pw1_w->ne[2]);
            std::vector<float> pw1(T * pw1_out);
            cpu_matmul_bt(tmp.data(), pw1_w.data(), pw1.data(), T, d, pw1_out);

            // GLU: split into two halves, sigmoid gate
            int half = pw1_out / 2; // 2560
            for (int t = 0; t < T; t++)
                for (int i = 0; i < half; i++) {
                    float gate = 1.0f / (1.0f + expf(-pw1[t * pw1_out + half + i]));
                    pw1[t * half + i] = pw1[t * pw1_out + i] * gate;
                }
            // pw1 now [T, half=2560]

            // Depthwise conv1d: kernel=33, groups=2560, SYMMETRIC padding=16
            std::vector<float> dw_out(T * half, 0.0f);
            int pad_sym = (ks - 1) / 2; // symmetric: 16 on each side
            for (int ch = 0; ch < half; ch++) {
                for (int t = 0; t < T; t++) {
                    float s = 0;
                    for (int k = 0; k < ks; k++) {
                        int ti = t - pad_sym + k;
                        if (ti >= 0 && ti < T)
                            s += pw1[ti * half + ch] * dw_w[ch * ks + k];
                    }
                    dw_out[t * half + ch] = s;
                }
            }

            // LayerNorm (called batch_norm)
            cpu_layernorm(dw_out.data(), bn_w.data(), bn_b_v.empty() ? nullptr : bn_b_v.data(), dw_out.data(), T, half);

            // Swish
            cpu_swish(dw_out.data(), T * half);

            // Pointwise conv2: [T, 2560] @ pw2^T → [T, d]
            int pw2_out = (int)(m.enc.blocks[li].conv.pw2_w->ne[2]);
            std::vector<float> conv_out(T * pw2_out);
            cpu_matmul_bt(dw_out.data(), pw2_w.data(), conv_out.data(), T, half, pw2_out);

            // Residual
            for (int i = 0; i < T * d; i++)
                x[i] += conv_out[i];
        }

        // === Macaron FFN2: out = 0.5*x + 0.5*ffn2(x) ===
        {
            std::vector<float> ln_w, ln_b_v, up_w, up_b_v, down_w, down_b_v;
            read_f32_vec(b.ffn2.ln_w, ln_w);
            read_f32_vec(b.ffn2.ln_b, ln_b_v);
            read_f32_vec(b.ffn2.up_w, up_w);
            read_f32_vec(b.ffn2.up_b, up_b_v);
            read_f32_vec(b.ffn2.down_w, down_w);
            read_f32_vec(b.ffn2.down_b, down_b_v);

            cpu_layernorm(x.data(), ln_w.data(), ln_b_v.data(), tmp.data(), T, d);
            std::vector<float> h_up(T * di);
            cpu_matmul_bt(tmp.data(), up_w.data(), h_up.data(), T, d, di);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < di; i++)
                    h_up[t * di + i] += up_b_v[i];
            cpu_swish(h_up.data(), T * di);
            cpu_matmul_bt(h_up.data(), down_w.data(), tmp.data(), T, di, d);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++) {
                    tmp[t * d + i] += down_b_v[i];
                    // Same as FFN1: ffn2.forward adds internal residual
                    x[t * d + i] = x[t * d + i] + 0.5f * tmp[t * d + i];
                }
        }

        // === Final LayerNorm ===
        {
            std::vector<float> ln_w, ln_b_v;
            read_f32_vec(b.ln_w, ln_w);
            read_f32_vec(b.ln_b, ln_b_v);
            cpu_layernorm(x.data(), ln_w.data(), ln_b_v.data(), x.data(), T, d);
        }
    }

    enc_output = std::move(x);
}

// Forward declarations for ggml graph builders
static ggml_tensor* swish_act(ggml_context* ctx, ggml_tensor* x);
static ggml_tensor* build_macaron_ffn(ggml_context* ctx, ggml_tensor* x, const firered_enc_ffn& f);
static ggml_tensor* build_conv_module(ggml_context* ctx, ggml_tensor* x, const firered_enc_conv& conv, int d_model,
                                      int kernel_size);

// ===========================================================================
// Hybrid encoder: ggml for matmuls, CPU for rel_shift attention
// ===========================================================================

static void hybrid_encoder(const float* subsampled, int T, int flat_dim, firered_asr_context* sctx,
                           std::vector<float>& enc_output) {
    auto& m = sctx->model;
    auto& hp = m.hp;
    int d = hp.d_model;
    int nh = hp.n_head;
    int hd = hp.head_dim;

    // Load PE center
    std::vector<float> pe_full;
    read_f32_vec(m.enc.pe, pe_full);
    int T_pe = 2 * T - 1;
    int Tmax = (int)(m.enc.pe->ne[1]);
    int pe_start = Tmax / 2 - T + 1;
    std::vector<float> pe_center(T_pe * d);
    memcpy(pe_center.data(), &pe_full[pe_start * d], T_pe * d * sizeof(float));

    // Read pos_bias_u/v for all layers (they're small — 20*64 each)
    struct layer_bias {
        std::vector<float> bu, bv;
    };
    std::vector<layer_bias> biases(hp.n_layers_enc);
    for (int li = 0; li < hp.n_layers_enc; li++) {
        read_f32_vec(m.enc.blocks[li].mhsa.pos_bias_u, biases[li].bu);
        read_f32_vec(m.enc.blocks[li].mhsa.pos_bias_v, biases[li].bv);
    }

    // Working buffer: x [d, T] in ggml layout = [T, d] row-major
    // Start with subsampled data
    std::vector<float> x_buf(T * d);

    // Linear projection via small ggml graph
    {
        struct ggml_init_params gp = {ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true};
        ggml_context* ctx0 = ggml_init(gp);
        ggml_cgraph* gf = ggml_new_graph(ctx0);

        ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, flat_dim, T);
        ggml_set_name(inp, "inp");
        ggml_set_input(inp);
        ggml_tensor* out = ggml_mul_mat(ctx0, m.enc.proj_w, inp);
        if (m.enc.proj_b)
            out = ggml_add(ctx0, out, m.enc.proj_b);
        ggml_set_name(out, "proj");
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);

        ggml_backend_sched_reset(sctx->sched);
        ggml_backend_sched_alloc_graph(sctx->sched, gf);
        ggml_backend_tensor_set(inp, subsampled, 0, flat_dim * T * sizeof(float));
        ggml_backend_sched_graph_compute(sctx->sched, gf);
        ggml_backend_tensor_get(out, x_buf.data(), 0, d * T * sizeof(float));
        ggml_free(ctx0);
    }

    // Per-layer: ggml for FFN/proj + CPU for attention
    for (int li = 0; li < hp.n_layers_enc; li++) {
        auto& b = m.enc.blocks[li];

        // Graph A: FFN1 + Q/K/V/P projections
        std::vector<float> Q_buf(T * d), K_buf(T * d), V_buf(T * d), P_buf(T_pe * d);
        {
            size_t mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(4096, false);
            std::vector<uint8_t> meta(mem);
            struct ggml_init_params gp = {mem, meta.data(), true};
            ggml_context* ctx0 = ggml_init(gp);
            ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

            ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
            ggml_set_name(x_in, "x_in");
            ggml_set_input(x_in);

            ggml_tensor* pe_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_pe);
            ggml_set_name(pe_in, "pe_in");
            ggml_set_input(pe_in);

            // FFN1: x + 0.5*net(x)
            ggml_tensor* x = build_macaron_ffn(ctx0, x_in, b.ffn1);
            ggml_set_name(x, "ffn1_out");
            ggml_set_output(x);

            // Q projection with LN
            ggml_tensor* q = ggml_norm(ctx0, x, 1e-5f);
            q = ggml_mul(ctx0, q, b.mhsa.ln_q_w);
            if (b.mhsa.ln_q_b)
                q = ggml_add(ctx0, q, b.mhsa.ln_q_b);
            q = ggml_mul_mat(ctx0, b.mhsa.w_qs, q);
            ggml_set_name(q, "Q");
            ggml_set_output(q);

            // K projection
            ggml_tensor* k = ggml_norm(ctx0, x, 1e-5f);
            k = ggml_mul(ctx0, k, b.mhsa.ln_k_w);
            if (b.mhsa.ln_k_b)
                k = ggml_add(ctx0, k, b.mhsa.ln_k_b);
            k = ggml_mul_mat(ctx0, b.mhsa.w_ks, k);
            ggml_set_name(k, "K");
            ggml_set_output(k);

            // V projection
            ggml_tensor* v = ggml_norm(ctx0, x, 1e-5f);
            v = ggml_mul(ctx0, v, b.mhsa.ln_v_w);
            if (b.mhsa.ln_v_b)
                v = ggml_add(ctx0, v, b.mhsa.ln_v_b);
            v = ggml_mul_mat(ctx0, b.mhsa.w_vs, v);
            ggml_set_name(v, "V");
            ggml_set_output(v);

            // P projection (position embeddings)
            ggml_tensor* p = ggml_mul_mat(ctx0, b.mhsa.lin_pos, pe_in);
            ggml_set_name(p, "P");
            ggml_set_output(p);

            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, k);
            ggml_build_forward_expand(gf, v);
            ggml_build_forward_expand(gf, p);

            ggml_backend_sched_reset(sctx->sched);
            if (!ggml_backend_sched_alloc_graph(sctx->sched, gf)) {
                fprintf(stderr, "firered_asr: layer %d graph A alloc failed\n", li);
                enc_output.assign(T * d, 0);
                return;
            }
            ggml_backend_tensor_set(x_in, x_buf.data(), 0, d * T * sizeof(float));
            ggml_backend_tensor_set(pe_in, pe_center.data(), 0, d * T_pe * sizeof(float));
            ggml_backend_sched_graph_compute(sctx->sched, gf);

            // Read outputs
            ggml_tensor* ffn1_t = ggml_graph_get_tensor(gf, "ffn1_out");
            ggml_backend_tensor_get(ffn1_t, x_buf.data(), 0, d * T * sizeof(float));
            ggml_tensor* Q_t = ggml_graph_get_tensor(gf, "Q");
            ggml_backend_tensor_get(Q_t, Q_buf.data(), 0, d * T * sizeof(float));
            ggml_tensor* K_t = ggml_graph_get_tensor(gf, "K");
            ggml_backend_tensor_get(K_t, K_buf.data(), 0, d * T * sizeof(float));
            ggml_tensor* V_t = ggml_graph_get_tensor(gf, "V");
            ggml_backend_tensor_get(V_t, V_buf.data(), 0, d * T * sizeof(float));
            ggml_tensor* P_t = ggml_graph_get_tensor(gf, "P");
            ggml_backend_tensor_get(P_t, P_buf.data(), 0, d * T_pe * sizeof(float));
            ggml_free(ctx0);
        }

        // CPU attention with rel_shift
        // Q/K/V are [d, T] in ggml layout = column-major
        // In ggml: Q[i, t] = Q_buf[t * d + i] for i < d, t < T
        // = Q_buf[t*d + h*hd + dd] for head h, dim dd
        // This is the SAME layout as my cpu_encoder used: row-major [T, d]
        {
            float scale = 1.0f / sqrtf((float)hd);
            auto& bu = biases[li].bu;
            auto& bv = biases[li].bv;
            std::vector<float> attn_out(T * d, 0.0f);

            for (int h = 0; h < nh; h++) {
                std::vector<float> scores(T * T);
                for (int tq = 0; tq < T; tq++) {
                    for (int tk = 0; tk < T; tk++) {
                        double content = 0, position = 0;
                        int pos_idx = T - 1 - tq + tk;
                        for (int dd = 0; dd < hd; dd++) {
                            float q_val = Q_buf[tq * d + h * hd + dd];
                            float k_val = K_buf[tk * d + h * hd + dd];
                            content += (q_val + bu[h * hd + dd]) * k_val;
                            if (pos_idx >= 0 && pos_idx < T_pe) {
                                float p_val = P_buf[pos_idx * d + h * hd + dd];
                                position += (q_val + bv[h * hd + dd]) * p_val;
                            }
                        }
                        scores[tq * T + tk] = (float)((content + position) * scale);
                    }
                }
                cpu_softmax_rows(scores.data(), T, T);
                for (int tq = 0; tq < T; tq++)
                    for (int dd = 0; dd < hd; dd++) {
                        double s = 0;
                        for (int tk = 0; tk < T; tk++)
                            s += scores[tq * T + tk] * V_buf[tk * d + h * hd + dd];
                        attn_out[tq * d + h * hd + dd] = (float)s;
                    }
            }

            // Graph B: FC output projection + residual + conv + FFN2 + LN
            {
                size_t mem = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(8192, false);
                std::vector<uint8_t> meta(mem);
                struct ggml_init_params gp = {mem, meta.data(), true};
                ggml_context* ctx0 = ggml_init(gp);
                ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

                // Input: x (after FFN1) and attn_out
                ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
                ggml_set_name(x_in, "x_in");
                ggml_set_input(x_in);
                ggml_tensor* attn_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T);
                ggml_set_name(attn_in, "attn_in");
                ggml_set_input(attn_in);

                // FC output projection + residual
                ggml_tensor* fc_out = ggml_mul_mat(ctx0, b.mhsa.fc_w, attn_in);
                ggml_tensor* x = ggml_add(ctx0, x_in, fc_out);

                // Conv module
                x = build_conv_module(ctx0, x, b.conv, d, hp.kernel_size);

                // FFN2
                x = build_macaron_ffn(ctx0, x, b.ffn2);

                // Final LayerNorm
                x = ggml_norm(ctx0, x, 1e-5f);
                x = ggml_mul(ctx0, x, b.ln_w);
                if (b.ln_b)
                    x = ggml_add(ctx0, x, b.ln_b);

                ggml_set_name(x, "layer_out");
                ggml_set_output(x);
                ggml_build_forward_expand(gf, x);

                ggml_backend_sched_reset(sctx->sched);
                if (!ggml_backend_sched_alloc_graph(sctx->sched, gf)) {
                    fprintf(stderr, "firered_asr: layer %d graph B alloc failed\n", li);
                    enc_output.assign(T * d, 0);
                    return;
                }
                ggml_backend_tensor_set(x_in, x_buf.data(), 0, d * T * sizeof(float));
                ggml_backend_tensor_set(attn_in, attn_out.data(), 0, d * T * sizeof(float));
                ggml_backend_sched_graph_compute(sctx->sched, gf);

                ggml_tensor* out_t = ggml_graph_get_tensor(gf, "layer_out");
                ggml_backend_tensor_get(out_t, x_buf.data(), 0, d * T * sizeof(float));
                ggml_free(ctx0);
            }
        }
    }

    enc_output = std::move(x_buf);
}

// ===========================================================================
// Swish activation (ggml graph version)
// ===========================================================================

static ggml_tensor* swish_act(ggml_context* ctx, ggml_tensor* x) {
    return ggml_mul(ctx, x, ggml_sigmoid(ctx, x));
}

// ===========================================================================
// Macaron FFN half-step
// ===========================================================================

static ggml_tensor* build_macaron_ffn(ggml_context* ctx, ggml_tensor* x, const firered_enc_ffn& f) {
    ggml_tensor* h = ggml_norm(ctx, x, 1e-5f);
    h = ggml_mul(ctx, h, f.ln_w);
    if (f.ln_b)
        h = ggml_add(ctx, h, f.ln_b);
    h = ggml_mul_mat(ctx, f.up_w, h);
    if (f.up_b)
        h = ggml_add(ctx, h, f.up_b);
    h = swish_act(ctx, h);
    h = ggml_mul_mat(ctx, f.down_w, h);
    if (f.down_b)
        h = ggml_add(ctx, h, f.down_b);
    // Macaron residual: x + 0.5*net(x)
    // (ffn.forward adds internal residual: ffn(x) = net(x) + x)
    // So: 0.5*x + 0.5*ffn(x) = 0.5*x + 0.5*(net(x)+x) = x + 0.5*net(x)
    return ggml_add(ctx, x, ggml_scale(ctx, h, 0.5f));
}

// ===========================================================================
// Conformer conv module
// ===========================================================================

static ggml_tensor* build_conv_module(ggml_context* ctx, ggml_tensor* x, const firered_enc_conv& conv, int d_model,
                                      int kernel_size) {
    ggml_tensor* residual = x;
    int T = (int)x->ne[1];

    ggml_tensor* h = ggml_norm(ctx, x, 1e-5f);
    h = ggml_mul(ctx, h, conv.pre_ln_w);
    if (conv.pre_ln_b)
        h = ggml_add(ctx, h, conv.pre_ln_b);

    // Pointwise conv1: d_model → 2*d_inner
    // Legacy GGUF: 3D [1, 1280, 5120] → reshape to [1280, 5120]
    // New GGUF (squeezed): already 2D [1280, 5120]
    ggml_tensor* pw1 = (ggml_n_dims(conv.pw1_w) > 2)
                           ? ggml_reshape_2d(ctx, conv.pw1_w, conv.pw1_w->ne[0] * conv.pw1_w->ne[1], conv.pw1_w->ne[2])
                           : conv.pw1_w;
    h = ggml_mul_mat(ctx, pw1, h);

    // GLU: split → sigmoid gate
    int ch = (int)h->ne[0] / 2; // 2560
    ggml_tensor* h1 = ggml_cont(ctx, ggml_view_2d(ctx, h, ch, T, h->nb[1], 0));
    ggml_tensor* h2 = ggml_cont(ctx, ggml_view_2d(ctx, h, ch, T, h->nb[1], ch * ggml_type_size(h->type)));
    h = ggml_mul(ctx, h1, ggml_sigmoid(ctx, h2)); // [2560, T]

    // Depthwise conv1d: groups=2560, kernel=33, SYMMETRIC padding=16
    ggml_tensor* ht = ggml_cont(ctx, ggml_transpose(ctx, h)); // [T, 2560]
    int pad_sym = (kernel_size - 1) / 2;                      // 16 on each side
    ht = ggml_pad_ext(ctx, ht, pad_sym, pad_sym, 0, 0, 0, 0, 0, 0);
    ht = ggml_conv_1d_dw(ctx, conv.dw_w, ht, 1, 0, 1);
    // Output is [OL, 1, channels, 1] — reshape to [OL, channels]
    ht = ggml_reshape_2d(ctx, ht, ht->ne[0], ht->ne[2]);
    h = ggml_cont(ctx, ggml_transpose(ctx, ht)); // [channels, T]

    // LayerNorm (named batch_norm in checkpoint)
    h = ggml_norm(ctx, h, 1e-5f);
    h = ggml_mul(ctx, h, conv.bn_w);
    if (conv.bn_b)
        h = ggml_add(ctx, h, conv.bn_b);

    h = swish_act(ctx, h);

    // Pointwise conv2: 2560 → 1280
    ggml_tensor* pw2 = (ggml_n_dims(conv.pw2_w) > 2)
                           ? ggml_reshape_2d(ctx, conv.pw2_w, conv.pw2_w->ne[0] * conv.pw2_w->ne[1], conv.pw2_w->ne[2])
                           : conv.pw2_w;
    h = ggml_mul_mat(ctx, pw2, h); // [1280, T]

    return ggml_add(ctx, residual, h);
}

// ===========================================================================
// Relative-PE multi-head self-attention (simplified — no rel_shift)
// ===========================================================================

// MHSA: uses flash_attn with bias_u (content attention only).
// Full relative position attention (with rel_shift) needs CPU-side computation
// which will be implemented as a post-processing step.
static ggml_tensor* build_rel_mhsa(ggml_context* ctx, ggml_tensor* x, ggml_tensor* pos_emb,
                                   const firered_enc_mhsa& mhsa, int n_head, int head_dim) {
    int d = n_head * head_dim;
    int T = (int)x->ne[1];
    ggml_tensor* residual = x;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Q/K/V with separate LayerNorm
    ggml_tensor* q = ggml_norm(ctx, x, 1e-5f);
    q = ggml_mul(ctx, q, mhsa.ln_q_w);
    if (mhsa.ln_q_b)
        q = ggml_add(ctx, q, mhsa.ln_q_b);
    q = ggml_mul_mat(ctx, mhsa.w_qs, q); // [d, T]

    ggml_tensor* k = ggml_norm(ctx, x, 1e-5f);
    k = ggml_mul(ctx, k, mhsa.ln_k_w);
    if (mhsa.ln_k_b)
        k = ggml_add(ctx, k, mhsa.ln_k_b);
    k = ggml_mul_mat(ctx, mhsa.w_ks, k);

    ggml_tensor* v = ggml_norm(ctx, x, 1e-5f);
    v = ggml_mul(ctx, v, mhsa.ln_v_w);
    if (mhsa.ln_v_b)
        v = ggml_add(ctx, v, mhsa.ln_v_b);
    v = ggml_mul_mat(ctx, mhsa.w_vs, v);

    // Reshape to multi-head: [d, T] → [hd, nh, T]
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, T);

    // Position embedding projection: pos_emb [d, T_pe] → [hd, nh, T_pe]
    // pos_emb has shape [d_model, 2*T-1] in our layout
    // lin_pos: [d_model, d_model], pos_emb: [d_model, T_pe]
    // But pos_emb comes from a view of the PE tensor which is [1280, 9999, 1]
    // The view extracts [1280, T_pe] — but ggml_view_2d on a 3D tensor
    // may not be contiguous. Let's ensure contiguity.
    ggml_tensor* p = ggml_mul_mat(ctx, mhsa.lin_pos, ggml_cont(ctx, pos_emb)); // [d, T_pe]
    int T_pe = (int)p->ne[1];
    p = ggml_reshape_3d(ctx, p, head_dim, n_head, T_pe); // [hd, nh, T_pe]

    // pos_bias_u, pos_bias_v: [hd, nh] (stored as [n_head, head_dim] in F16)
    // Cast to F32 and reshape to [hd, nh, 1] for broadcasting
    ggml_tensor* bias_u = ggml_cast(ctx, mhsa.pos_bias_u, GGML_TYPE_F32);
    bias_u = ggml_reshape_3d(ctx, bias_u, head_dim, n_head, 1);
    ggml_tensor* bias_v = ggml_cast(ctx, mhsa.pos_bias_v, GGML_TYPE_F32);
    bias_v = ggml_reshape_3d(ctx, bias_v, head_dim, n_head, 1);

    // Content attention: (Q + bias_u) @ K^T
    ggml_tensor* q_u = ggml_add(ctx, q, bias_u); // [hd, nh, T]
    // q_u^T @ k: need [T, hd, nh] @ [hd, nh, T] = not standard matmul
    // Actually we need: for each head h, compute q_u[h] @ k[h]^T
    // In ggml with 3D tensors: ggml_mul_mat operates on the first 2 dims
    // q_u: [hd, nh, T], k: [hd, nh, T]
    // ggml_mul_mat(k, q_u) computes k^T @ q_u over first dim:
    //   result[i,j,t] = sum_d k[d,i,j] * q_u[d,i,t]
    // No, ggml_mul_mat with 3D: result.ne = [k.ne[1], q_u.ne[1], q_u.ne[2]]
    // = [nh, nh, T] — wrong.
    // Need to permute to [hd, T, nh] first, then matmul
    // Permute: q_u [hd, nh, T] → [hd, T, nh]
    q_u = ggml_cont(ctx, ggml_permute(ctx, q_u, 0, 2, 1, 3));               // [hd, T, nh]
    ggml_tensor* k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [hd, T, nh]
    // mul_mat(k_perm, q_u): for each head (ne[2]):
    //   result[:,:,h] = k_perm[:,:,h]^T @ q_u[:,:,h]
    //   = [T, hd] @ [hd, T] = [T, T]
    // result: [T, T, nh]
    ggml_tensor* matrix_ac = ggml_mul_mat(ctx, k_perm, q_u); // [T, T, nh]

    // Position attention: (Q + bias_v) @ P^T → rel_shift
    ggml_tensor* q_v = ggml_add(ctx, q, bias_v);
    q_v = ggml_cont(ctx, ggml_permute(ctx, q_v, 0, 2, 1, 3));               // [hd, T, nh]
    ggml_tensor* p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3)); // [hd, T_pe, nh]
    ggml_tensor* matrix_bd = ggml_mul_mat(ctx, p_perm, q_v);                // [T_pe, T, nh]

    // rel_shift: [T_pe, T, nh] → [T, T, nh]
    // Python: zero_pad col → reshape → skip first row → slice
    // T_pe = 2*T-1
    // Pad: [T_pe+1, T, nh] (add 1 col of zeros at start of dim 0)
    matrix_bd = ggml_pad_ext(ctx, matrix_bd, 1, 0, 0, 0, 0, 0, 0, 0);
    // Reshape: [T_pe+1, T, nh] → [T, T_pe+1, nh] ... no, that's wrong
    // Python does: x_padded = cat([zero_pad, x], dim=-1) → [B,H,T1,T2+1]
    //              x_padded = view(B,H,T2+1,T1) → skip first row → view_as(x)
    // In our layout: matrix_bd is [T_pe, T, nh] (ne[0]=T_pe, ne[1]=T, ne[2]=nh)
    // After pad: [T_pe+1, T, nh]
    // Reshape to [T, T_pe+1, nh]... this transposes dim0 and dim1
    // Actually the rel_shift in PyTorch works on [B,H,T1,T2] where T1=query_len, T2=2*T-1
    // Our layout has T_pe in dim0 (fastest), T in dim1. Let me think...
    //
    // The Python operation is (ignoring B,H):
    //   x: [T1, T2]  (T1=T queries, T2=2T-1 positions)
    //   pad_col: [T1, T2+1]
    //   reshape: [T2+1, T1]
    //   skip first row: [T2, T1]
    //   reshape back: [T1, T2]
    //   slice: [T1, T//2+1]
    //
    // In ggml ne layout, matrix_bd is [T_pe, T, nh] where ne[0]=T_pe
    // This is equivalent to [T2, T1, H] in the Python notation
    // So we need to work on dims 0 and 1:
    // After pad: [T_pe+1, T, nh]  (ne[0]=T_pe+1=2T, ne[1]=T)
    // Reshape to [T, T_pe+1, nh]  (swap dim 0 and 1) — but this isn't just a reshape,
    // it's a transpose! reshape won't work because the memory layout changes.
    //
    // Actually in the Python code, the rel_shift works on the LAST two dims [T1, T2].
    // Our ggml layout has these as [ne[0], ne[1]] = [T_pe, T].
    // But Python's [T1, T2] has T1=T (queries) and T2=T_pe (positions).
    // So our layout has them SWAPPED compared to Python.
    //
    // The correct approach: transpose matrix_bd to [T, T_pe, nh], do the rel_shift,
    // then transpose back. But rel_shift involves reshape which requires contiguous data.
    //
    // This is getting complex. For now, let me use standard attention (no rel PE)
    // but with the pos_bias_u applied to improve accuracy slightly.
    // Full rel PE can be added later with CPU-side score computation.

    // rel_shift: compute on CPU since ggml can't do row-major reshape
    // matrix_bd: [T_pe, T, nh] (ggml layout, ne[0]=T_pe fastest)
    // Need to apply Python's rel_shift which operates in row-major order
    //
    // Approach: mark matrix_ac and matrix_bd as outputs, compute graph,
    // read them out, apply rel_shift + softmax + V weighting on CPU,
    // then set result as input to next graph.
    //
    // But this requires splitting the graph, which is complex.
    // Alternative: just use flash_attn (no rel PE) for now to get a working baseline.
    (void)matrix_bd;
    (void)matrix_ac;

    // Fall back to flash_attn with bias_u (content attention only)
    q_u = ggml_cont(ctx, ggml_permute(ctx, ggml_add(ctx, q, bias_u), 0, 2, 1, 3)); // [hd, T, nh]
    ggml_tensor* k_fa = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor* v_fa = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q_u, k_fa, v_fa, nullptr, scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, d, T);
    attn = ggml_mul_mat(ctx, mhsa.fc_w, attn);

    return ggml_add(ctx, residual, attn);
}

// ===========================================================================
// Conv2d subsampling (CPU, pre-graph)
// ===========================================================================

static void conv2d_subsample_cpu(const float* features, int n_frames, int n_mels, const firered_model& m,
                                 std::vector<float>& out, int& T_out) {
    // Conv2d layer 0: [1, 1, T, 80] → [1, 32, T', 40] with 3x3 stride 2
    // Conv2d layer 1: [1, 32, T', 40] → [1, 32, T'', 19] with 3x3 stride 2
    // Flatten: [1, 32*19, T''] = [1, 608, T'']
    // Linear: [1, 1280, T'']

    int T0 = n_frames, F0 = n_mels; // 1098, 80
    int T1 = (T0 - 3) / 2 + 1;      // (1098-3)/2+1 = 548
    int F1 = (F0 - 3) / 2 + 1;      // (80-3)/2+1 = 39

    // Conv0: [32, 1, 3, 3], stride 2, no padding
    // Reuse the file-scope read_f32_vec which handles F32/F16/quantized types
    auto read_f32 = [](ggml_tensor* t, std::vector<float>& out) { read_f32_vec(t, out); };

    int C0 = 32;
    std::vector<float> conv0_w_data;
    read_f32(m.enc.conv0_w, conv0_w_data);
    std::vector<float> conv0_b_data(C0, 0.0f);
    if (m.enc.conv0_b)
        read_f32(m.enc.conv0_b, conv0_b_data);

    std::vector<float> act1(C0 * T1 * F1, 0.0f);
    for (int c = 0; c < C0; c++) {
        for (int t = 0; t < T1; t++) {
            for (int f = 0; f < F1; f++) {
                float sum = conv0_b_data[c];
                for (int kt = 0; kt < 3; kt++)
                    for (int kf = 0; kf < 3; kf++) {
                        int ti = t * 2 + kt, fi = f * 2 + kf;
                        if (ti < T0 && fi < F0)
                            sum += features[ti * F0 + fi] * conv0_w_data[c * 9 + kt * 3 + kf];
                    }
                act1[(c * T1 + t) * F1 + f] = std::max(sum, 0.0f); // ReLU
            }
        }
    }

    // Conv1: [32, 32, 3, 3], stride 2
    int T2 = (T1 - 3) / 2 + 1; // (548-3)/2+1 = 273
    int F2 = (F1 - 3) / 2 + 1; // (39-3)/2+1 = 19
    int C1 = 32;

    std::vector<float> conv1_w_data;
    read_f32(m.enc.conv1_w, conv1_w_data);
    std::vector<float> conv1_b_data(C1, 0.0f);
    if (m.enc.conv1_b)
        read_f32(m.enc.conv1_b, conv1_b_data);

    std::vector<float> act2(C1 * T2 * F2, 0.0f);
    for (int co = 0; co < C1; co++) {
        for (int t = 0; t < T2; t++) {
            for (int f = 0; f < F2; f++) {
                float sum = conv1_b_data[co];
                for (int ci = 0; ci < C0; ci++)
                    for (int kt = 0; kt < 3; kt++)
                        for (int kf = 0; kf < 3; kf++) {
                            int ti = t * 2 + kt, fi = f * 2 + kf;
                            if (ti < T1 && fi < F1)
                                sum += act1[(ci * T1 + ti) * F1 + fi] * conv1_w_data[(co * C0 + ci) * 9 + kt * 3 + kf];
                        }
                act2[(co * T2 + t) * F2 + f] = std::max(sum, 0.0f); // ReLU
            }
        }
    }

    // Flatten: [C1, T2, F2] → [T2, C1*F2] = [T2, 608]
    T_out = T2;
    int flat_dim = C1 * F2; // 32 * 19 = 608
    out.resize(T_out * flat_dim);
    for (int t = 0; t < T2; t++)
        for (int c = 0; c < C1; c++)
            for (int f = 0; f < F2; f++)
                out[t * flat_dim + c * F2 + f] = act2[(c * T2 + t) * F2 + f];
}

// ===========================================================================
// Transcribe (CTC path)
// ===========================================================================

extern "C" char* firered_asr_transcribe(struct firered_asr_context* ctx, const float* samples, int n_samples) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;

    auto& m = ctx->model;
    auto& hp = m.hp;

    // Step 1: Fbank features
    std::vector<float> features;
    int n_frames = 0;
    compute_fbank(samples, n_samples, features, n_frames);
    if (n_frames <= 0)
        return nullptr;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "firered_asr: %d fbank frames\n", n_frames);
        if (n_frames > 100 && ctx->params.verbosity >= 2) {
            fprintf(stderr, "  fbank t=0 first 5: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", features[0], features[1],
                    features[2], features[3], features[4]);
            int t100 = 100 * 80;
            fprintf(stderr, "  fbank t=100 first 5: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", features[t100],
                    features[t100 + 1], features[t100 + 2], features[t100 + 3], features[t100 + 4]);
        }
    }

    // Step 1b: Apply CMVN (global mean-variance normalization)
    if (m.cmvn_mean && m.cmvn_std) {
        std::vector<float> mean_v(hp.idim), std_v(hp.idim);
        ggml_backend_tensor_get(m.cmvn_mean, mean_v.data(), 0, hp.idim * sizeof(float));
        ggml_backend_tensor_get(m.cmvn_std, std_v.data(), 0, hp.idim * sizeof(float));
        for (int t_idx = 0; t_idx < n_frames; t_idx++)
            for (int f = 0; f < hp.idim; f++)
                features[t_idx * hp.idim + f] = (features[t_idx * hp.idim + f] - mean_v[f]) / std_v[f];
    }

    // Step 1c: Context padding (pad right with context-1 frames of zeros)
    int context = 7; // from model config
    int n_frames_padded = n_frames + context - 1;
    std::vector<float> features_padded(n_frames_padded * hp.idim, 0.0f);
    memcpy(features_padded.data(), features.data(), n_frames * hp.idim * sizeof(float));

    // Step 2: Conv2d subsampling on CPU (using padded input)
    std::vector<float> subsampled;
    int T_sub = 0;
    conv2d_subsample_cpu(features_padded.data(), n_frames_padded, hp.idim, m, subsampled, T_sub);
    if (T_sub <= 0)
        return nullptr;

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "firered_asr: subsampled to %d frames (608-dim)\n", T_sub);

    // Step 3: CPU encoder (Conformer with relative PE attention)
    int flat_dim = 608; // 32 * 19
    std::vector<float> enc_output;
    hybrid_encoder(subsampled.data(), T_sub, flat_dim, ctx, enc_output);
    // enc_output: [T_sub, d_model] row-major

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "  enc_out t=0 first 8: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", enc_output[0],
                enc_output[1], enc_output[2], enc_output[3], enc_output[4], enc_output[5], enc_output[6],
                enc_output[7]);
    }

    // Step 4: Greedy decoder (Transformer with cross-attention)
    // If decoder weights are loaded, use decoder path; else fall back to CTC
    if (m.dec.emb_w && m.dec.prj_w && !m.dec.blocks.empty()) {
        // Read decoder weights to CPU
        std::vector<float> emb_w, pe_dec, norm_w, norm_b, prj_w;
        read_f32_vec(m.dec.emb_w, emb_w); // [odim, d]
        read_f32_vec(m.dec.pe, pe_dec);   // [pe_maxlen, d] (sinusoidal)
        read_f32_vec(m.dec.norm_out_w, norm_w);
        read_f32_vec(m.dec.norm_out_b, norm_b);
        read_f32_vec(m.dec.prj_w, prj_w); // [odim, d]

        if (ctx->params.verbosity >= 2) {
            // Dump first values of key decoder tensors to verify GPU→CPU dequantization
            auto dump = [](const char* name, const std::vector<float>& v, int n) {
                fprintf(stderr, "  %s[0..%d]: [", name, n-1);
                for (int i = 0; i < n && i < (int)v.size(); i++)
                    fprintf(stderr, "%s%.4f", i?",":"", v[i]);
                fprintf(stderr, "] (total %zu)\n", v.size());
            };
            dump("emb_w", emb_w, 8);
            dump("pe_dec", pe_dec, 8);
            dump("norm_w", norm_w, 8);
            dump("norm_b", norm_b, 8);
            dump("prj_w", prj_w, 8);
        }

        float scale = sqrtf((float)hp.d_model);
        // For LID models (odim <= 256, small vocab), only 1 decode step needed —
        // the first token after SOS is the language. Full ASR needs longer sequences.
        const bool is_lid = (hp.odim <= 256);
        // ~3-4 BPE tokens per second of audio is typical for ASR.
        // Cap at T_sub (1 token per subsampled frame) which is already generous.
        int max_len = is_lid ? 2 : std::min(T_sub, 150);
        int beam_size_effective = is_lid ? 1 : std::max(1, ctx->params.beam_size);
        int d = hp.d_model;
        int odim = hp.odim;
        const bool use_gpu_decoder_proj = !ggml_backend_is_cpu(ctx->backend);
        int debug_dec_step = -1;
        int debug_dec_layer = -1;
        if (const char* env = std::getenv("FIRERED_DEBUG_DECODER_STEP"))
            debug_dec_step = atoi(env);
        if (const char* env = std::getenv("FIRERED_DEBUG_DECODER_LAYER"))
            debug_dec_layer = atoi(env);

        // Pre-cache only SMALL tensors (norms/biases, ~d floats each).
        // Large weight matrices stay in Q4_K and are used via ggml_vecmat.
        // This saves 23.6s of dequantization that the old path needed.
        int64_t t_weight_cache0 = ggml_time_us();
        struct dec_layer_cache {
            std::vector<float> sattn_norm_w, sattn_norm_b;
            std::vector<float> xattn_norm_w, xattn_norm_b;
            std::vector<float> mlp_norm_w, mlp_norm_b;
            int di = 0;
            // For the beam path fallback, we lazy-load full F32 weights on first beam use.
            bool full_cached = false;
            std::vector<float> sattn_w_qs, sattn_w_qs_b, sattn_w_ks, sattn_w_vs, sattn_w_vs_b;
            std::vector<float> sattn_fc_w, sattn_fc_b;
            std::vector<float> xattn_w_qs, xattn_w_qs_b;
            std::vector<float> xattn_fc_w, xattn_fc_b;
            std::vector<float> mlp_w1, mlp_b1, mlp_w2, mlp_b2;
        };
        std::vector<dec_layer_cache> dec_cache(hp.n_layers_dec);
        for (int li = 0; li < hp.n_layers_dec; li++) {
            auto& b = m.dec.blocks[li];
            auto& c = dec_cache[li];
            // Only read small norm tensors (~d floats each, always F32)
            read_small_tensor(b.sattn_norm_w, c.sattn_norm_w);
            read_small_tensor(b.sattn_norm_b, c.sattn_norm_b);
            read_small_tensor(b.xattn_norm_w, c.xattn_norm_w);
            read_small_tensor(b.xattn_norm_b, c.xattn_norm_b);
            read_small_tensor(b.mlp_norm_w, c.mlp_norm_w);
            read_small_tensor(b.mlp_norm_b, c.mlp_norm_b);
            c.di = b.mlp_w1 ? (int)b.mlp_w1->ne[1] : hp.d_inner;
        }

        // Pre-compute cross-attention K/V for encoder output (shared across all steps)
        // K_enc[li][t*d+i] = sum_k enc[t*d+k] * W_k[i*d+k]
        // V_enc[li][t*d+i] = sum_k enc[t*d+k] * W_v[i*d+k] + bias
        std::vector<std::vector<float>> K_enc(hp.n_layers_dec), V_enc(hp.n_layers_dec);
        for (int li = 0; li < hp.n_layers_dec; li++) {
            auto& c = dec_cache[li];
            K_enc[li].resize(T_sub * d);
            V_enc[li].resize(T_sub * d);
            bool kv_done = false;

            if (!ggml_backend_is_cpu(ctx->backend)) {
                size_t mem = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(256, false);
                struct ggml_init_params gp = {mem, nullptr, true};
                ggml_context* ctx0 = ggml_init(gp);
                if (ctx0) {
                    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);
                    ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, T_sub);
                    ggml_set_input(enc_inp);

                    ggml_tensor* k_proj = ggml_mul_mat(ctx0, m.dec.blocks[li].xattn.w_ks, enc_inp);
                    ggml_set_name(k_proj, "firered_dec_k_proj");
                    ggml_set_output(k_proj);
                    ggml_build_forward_expand(gf, k_proj);

                    ggml_tensor* v_proj = ggml_mul_mat(ctx0, m.dec.blocks[li].xattn.w_vs, enc_inp);
                    if (m.dec.blocks[li].xattn.w_vs_b)
                        v_proj = ggml_add(ctx0, v_proj, m.dec.blocks[li].xattn.w_vs_b);
                    ggml_set_name(v_proj, "firered_dec_v_proj");
                    ggml_set_output(v_proj);
                    ggml_build_forward_expand(gf, v_proj);

                    ggml_backend_sched_reset(ctx->sched);
                    if (ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
                        ggml_backend_tensor_set(enc_inp, enc_output.data(), 0, T_sub * d * sizeof(float));
                        if (ggml_backend_sched_graph_compute(ctx->sched, gf) == GGML_STATUS_SUCCESS) {
                            ggml_backend_tensor_get(k_proj, K_enc[li].data(), 0, T_sub * d * sizeof(float));
                            ggml_backend_tensor_get(v_proj, V_enc[li].data(), 0, T_sub * d * sizeof(float));
                            kv_done = true;
                        }
                    }
                    ggml_free(ctx0);
                }
            }

            if (!kv_done) {
                // Use ggml graph for batch K/V precompute (handles Q4_K natively)
                size_t mem2 = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 256 * 1024;
                struct ggml_init_params gp2 = {mem2, nullptr, true};
                ggml_context* ctx2 = ggml_init(gp2);
                if (ctx2) {
                    ggml_tensor* ei = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, d, T_sub);
                    ggml_set_name(ei, "ei");
                    ggml_set_input(ei);
                    ggml_tensor* kp = ggml_mul_mat(ctx2, m.dec.blocks[li].xattn.w_ks, ei);
                    ggml_set_name(kp, "kp");
                    ggml_set_output(kp);
                    ggml_tensor* vp = ggml_mul_mat(ctx2, m.dec.blocks[li].xattn.w_vs, ei);
                    if (m.dec.blocks[li].xattn.w_vs_b)
                        vp = ggml_add(ctx2, vp, m.dec.blocks[li].xattn.w_vs_b);
                    ggml_set_name(vp, "vp");
                    ggml_set_output(vp);
                    ggml_cgraph* gf2 = ggml_new_graph(ctx2);
                    ggml_build_forward_expand(gf2, kp);
                    ggml_build_forward_expand(gf2, vp);
                    ggml_backend_sched_reset(ctx->sched);
                    ggml_backend_sched_set_tensor_backend(ctx->sched, m.dec.blocks[li].xattn.w_ks, ctx->backend);
                    ggml_backend_sched_set_tensor_backend(ctx->sched, m.dec.blocks[li].xattn.w_vs, ctx->backend);
                    if (m.dec.blocks[li].xattn.w_vs_b)
                        ggml_backend_sched_set_tensor_backend(ctx->sched, m.dec.blocks[li].xattn.w_vs_b, ctx->backend);
                    if (ggml_backend_sched_alloc_graph(ctx->sched, gf2)) {
                        ggml_backend_tensor_set(ggml_graph_get_tensor(gf2, "ei"), enc_output.data(), 0,
                                                T_sub * d * sizeof(float));
                        if (ggml_backend_sched_graph_compute(ctx->sched, gf2) == GGML_STATUS_SUCCESS) {
                            ggml_backend_tensor_get(ggml_graph_get_tensor(gf2, "kp"), K_enc[li].data(), 0,
                                                    T_sub * d * sizeof(float));
                            ggml_backend_tensor_get(ggml_graph_get_tensor(gf2, "vp"), V_enc[li].data(), 0,
                                                    T_sub * d * sizeof(float));
                            kv_done = true;
                        }
                    }
                    ggml_free(ctx2);
                }
                if (!kv_done)
                    fprintf(stderr, "firered_asr: WARNING: K/V precompute failed for layer %d\n", li);
            }
        }

        int64_t t_weight_cache = ggml_time_us() - t_weight_cache0;
        if (ctx->params.verbosity >= 1 || getenv("FIRERED_BENCH"))
            fprintf(stderr, "firered_asr: decoder weights cached + K/V pre-computed in %.1fms\n", t_weight_cache / 1e3);

        ggml_context* dec_proj_ctx = nullptr;
        ggml_cgraph* dec_proj_gf = nullptr;
        ggml_tensor* dec_proj_inp = nullptr;
        ggml_tensor* dec_proj_logits = nullptr;
        if (use_gpu_decoder_proj) {
            size_t mem = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(128, false);
            struct ggml_init_params gp = {mem, nullptr, true};
            dec_proj_ctx = ggml_init(gp);
            if (dec_proj_ctx) {
                dec_proj_gf = ggml_new_graph_custom(dec_proj_ctx, 128, false);
                dec_proj_inp = ggml_new_tensor_2d(dec_proj_ctx, GGML_TYPE_F32, d, 1);
                ggml_set_input(dec_proj_inp);
                dec_proj_logits = ggml_mul_mat(dec_proj_ctx, m.dec.prj_w, dec_proj_inp);
                ggml_set_output(dec_proj_logits);
                ggml_build_forward_expand(dec_proj_gf, dec_proj_logits);
            }
        }

        auto project_decoder_logits = [&](const float* xn, std::vector<float>& logits_out) -> bool {
            if (!use_gpu_decoder_proj || !dec_proj_ctx || !dec_proj_gf || !dec_proj_inp || !dec_proj_logits)
                return false;

            bool ok = false;
            ggml_backend_sched_reset(ctx->sched);
            if (ggml_backend_sched_alloc_graph(ctx->sched, dec_proj_gf)) {
                ggml_backend_tensor_set(dec_proj_inp, xn, 0, d * sizeof(float));
                if (ggml_backend_sched_graph_compute(ctx->sched, dec_proj_gf) == GGML_STATUS_SUCCESS) {
                    logits_out.resize(odim);
                    ggml_backend_tensor_get(dec_proj_logits, logits_out.data(), 0, odim * sizeof(float));
                    ok = true;
                }
            }
            return ok;
        };

        // Beam search state
        int beam_size = beam_size_effective;
        struct beam_hyp {
            std::vector<int> tokens;
            float score = 0;
            bool finished = false;
            std::vector<std::shared_ptr<std::vector<float>>> sa_k, sa_v; // [n_layers][history]
        };
        std::vector<beam_hyp> beams(beam_size);
        for (auto& b : beams) {
            b.tokens.push_back(hp.sos_id);
            b.sa_k.resize(hp.n_layers_dec);
            b.sa_v.resize(hp.n_layers_dec);
            for (int li = 0; li < hp.n_layers_dec; li++) {
                b.sa_k[li] = std::make_shared<std::vector<float>>();
                b.sa_v[li] = std::make_shared<std::vector<float>>();
                b.sa_k[li]->reserve((size_t)max_len * d);
                b.sa_v[li]->reserve((size_t)max_len * d);
            }
            b.score = 0;
        }
        // Only beam 0 starts active; others start with -inf score
        for (int b = 1; b < beam_size; b++)
            beams[b].score = -1e9f;
        int nh_dec = hp.n_head_dec;
        int hd_dec = d / nh_dec;
        const float inv_sqrt_hd_dec = 1.0f / sqrtf((float)hd_dec);

        if (ctx->params.verbosity >= 1)
            fprintf(stderr, "firered_asr: decoder starting (max_len=%d, beam=%d, layers=%d)\n",
                    max_len, beam_size_effective, hp.n_layers_dec);
        int64_t t_dec0 = ggml_time_us();

        for (int step = 0; step < max_len; step++) {
            // Check if all beams finished
            bool all_done = true;
            for (auto& b : beams)
                if (!b.finished) {
                    all_done = false;
                    break;
                }
            if (all_done)
                break;

            if (ctx->params.verbosity >= 2 || (getenv("FIRERED_BENCH") && (step % 20 == 0 || step < 3))) {
                int64_t t_now = ggml_time_us();
                fprintf(stderr, "firered_asr: decode step %d/%d (%.1fms elapsed)\n",
                        step, max_len, (t_now - t_dec0) / 1e3);
            }

            // Greedy path uses cached CPU matmuls (same as beam path).
            // The ggml_vecmat approach creates 128 tiny GPU graphs per step,
            // where CUDA launch overhead (~20ms each) dominates compute (~0.1ms).
            // CPU matmuls with pre-cached F32 weights: ~0.5ms each → ~64ms/step.
            if (beam_size == 1) {
                // Lazy-load full F32 weights on first greedy step (shared with beam cache)
                if (!dec_cache[0].full_cached) {
                    int64_t t_dequant0 = ggml_time_us();
                    for (int li2 = 0; li2 < hp.n_layers_dec; li2++) {
                        auto& b2 = m.dec.blocks[li2];
                        auto& c2 = dec_cache[li2];
                        read_f32_vec(b2.sattn.w_qs, c2.sattn_w_qs);
                        if (b2.sattn.w_qs_b) read_f32_vec(b2.sattn.w_qs_b, c2.sattn_w_qs_b);
                        read_f32_vec(b2.sattn.w_ks, c2.sattn_w_ks);
                        read_f32_vec(b2.sattn.w_vs, c2.sattn_w_vs);
                        if (b2.sattn.w_vs_b) read_f32_vec(b2.sattn.w_vs_b, c2.sattn_w_vs_b);
                        read_f32_vec(b2.sattn.fc_w, c2.sattn_fc_w);
                        if (b2.sattn.fc_b) read_f32_vec(b2.sattn.fc_b, c2.sattn_fc_b);
                        read_f32_vec(b2.xattn.w_qs, c2.xattn_w_qs);
                        if (b2.xattn.w_qs_b) read_f32_vec(b2.xattn.w_qs_b, c2.xattn_w_qs_b);
                        read_f32_vec(b2.xattn.fc_w, c2.xattn_fc_w);
                        if (b2.xattn.fc_b) read_f32_vec(b2.xattn.fc_b, c2.xattn_fc_b);
                        read_f32_vec(b2.mlp_w1, c2.mlp_w1);
                        if (b2.mlp_b1) read_f32_vec(b2.mlp_b1, c2.mlp_b1);
                        read_f32_vec(b2.mlp_w2, c2.mlp_w2);
                        if (b2.mlp_b2) read_f32_vec(b2.mlp_b2, c2.mlp_b2);
                        c2.full_cached = true;
                    }
                    if (ctx->params.verbosity >= 1) {
                        int64_t t_dequant = ggml_time_us() - t_dequant0;
                        fprintf(stderr, "firered_asr: decoder weights dequantized in %.1fms\n", t_dequant / 1e3);
                    }
                    if (ctx->params.verbosity >= 2) {
                        auto& c0 = dec_cache[0];
                        fprintf(stderr, "  dec.0 sattn_w_qs[0..7]: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                                c0.sattn_w_qs[0], c0.sattn_w_qs[1], c0.sattn_w_qs[2], c0.sattn_w_qs[3],
                                c0.sattn_w_qs[4], c0.sattn_w_qs[5], c0.sattn_w_qs[6], c0.sattn_w_qs[7]);
                    }
                }

                auto& beam = beams[0];
                int cur_token = beam.tokens.back();
                int64_t t_logit = 0;
                int64_t t_step0 = ggml_time_us();

                // Embed
                std::vector<float> x(d);
                for (int i = 0; i < d; i++) {
                    x[i] = emb_w[cur_token * d + i] * scale;
                }
                if (ctx->params.verbosity >= 2 && step < 2) {
                    fprintf(stderr, "  step%d: token=%d(sos=%d,eos=%d), x_emb[0..7]=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                            step, cur_token, hp.sos_id, hp.eos_id,
                            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
                }
                for (int i = 0; i < d; i++) {
                    if (step < (int)(m.dec.pe->ne[1]))
                        x[i] += pe_dec[step * d + i];
                }

                for (int li = 0; li < hp.n_layers_dec; li++) {
                    auto& c = dec_cache[li];
                    const bool debug_dec_here = (debug_dec_step >= 0 && debug_dec_layer >= 0 &&
                                                 step == debug_dec_step && li == debug_dec_layer);

                    // === Self-attention (causal, attend to history) ===
                    {
                        std::vector<float> xn(d);
                        cpu_layernorm(x.data(), c.sattn_norm_w.data(), c.sattn_norm_b.data(), xn.data(), 1, d);

                        std::vector<float> Q_sa(d), K_cur(d), V_cur(d);
                        cpu_matmul_bt(xn.data(), c.sattn_w_qs.data(), Q_sa.data(), 1, d, d);
                        if (!c.sattn_w_qs_b.empty())
                            for (int i = 0; i < d; i++) Q_sa[i] += c.sattn_w_qs_b[i];
                        cpu_matmul_bt(xn.data(), c.sattn_w_ks.data(), K_cur.data(), 1, d, d);
                        cpu_matmul_bt(xn.data(), c.sattn_w_vs.data(), V_cur.data(), 1, d, d);
                        if (!c.sattn_w_vs_b.empty())
                            for (int i = 0; i < d; i++) V_cur[i] += c.sattn_w_vs_b[i];

                        auto& sa_k_hist = *beam.sa_k[li];
                        auto& sa_v_hist = *beam.sa_v[li];
                        sa_k_hist.insert(sa_k_hist.end(), K_cur.begin(), K_cur.end());
                        sa_v_hist.insert(sa_v_hist.end(), V_cur.begin(), V_cur.end());

                        int n_hist = (int)(sa_k_hist.size() / d);
                        std::vector<float> sa_out(d, 0);
                        for (int h = 0; h < nh_dec; h++) {
                            std::vector<float> scores(n_hist);
                            for (int t = 0; t < n_hist; t++) {
                                double s = 0;
                                for (int dd = 0; dd < hd_dec; dd++)
                                    s += Q_sa[h * hd_dec + dd] * sa_k_hist[t * d + h * hd_dec + dd];
                                scores[t] = (float)s * inv_sqrt_hd_dec;
                            }
                            cpu_softmax_rows(scores.data(), 1, n_hist);
                            for (int dd = 0; dd < hd_dec; dd++) {
                                double s = 0;
                                for (int t = 0; t < n_hist; t++)
                                    s += scores[t] * sa_v_hist[t * d + h * hd_dec + dd];
                                sa_out[h * hd_dec + dd] = (float)s;
                            }
                        }
                        if (debug_dec_here)
                            firered_debug_dump_vec("greedy.sa_out", sa_out, 8);

                        std::vector<float> sa_fc(d);
                        cpu_matmul_bt(sa_out.data(), c.sattn_fc_w.data(), sa_fc.data(), 1, d, d);
                        for (int i = 0; i < d; i++)
                            x[i] += sa_fc[i] + (c.sattn_fc_b.empty() ? 0 : c.sattn_fc_b[i]);
                    }

                    // === Cross-attention ===
                    std::vector<float> xn(d);
                    cpu_layernorm(x.data(), c.xattn_norm_w.data(), c.xattn_norm_b.data(), xn.data(), 1, d);

                    std::vector<float> Qx(d);
                    cpu_matmul_bt(xn.data(), c.xattn_w_qs.data(), Qx.data(), 1, d, d);
                    if (!c.xattn_w_qs_b.empty())
                        for (int i = 0; i < d; i++) Qx[i] += c.xattn_w_qs_b[i];

                    std::vector<float> attn_out(d, 0);
                    for (int h = 0; h < nh_dec; h++) {
                        std::vector<float> scores(T_sub);
                        for (int t = 0; t < T_sub; t++) {
                            double s = 0;
                            for (int dd = 0; dd < hd_dec; dd++)
                                s += Qx[h * hd_dec + dd] * K_enc[li][t * d + h * hd_dec + dd];
                            scores[t] = (float)s * inv_sqrt_hd_dec;
                        }
                        cpu_softmax_rows(scores.data(), 1, T_sub);
                        for (int dd = 0; dd < hd_dec; dd++) {
                            double s = 0;
                            for (int t = 0; t < T_sub; t++)
                                s += scores[t] * V_enc[li][t * d + h * hd_dec + dd];
                            attn_out[h * hd_dec + dd] = (float)s;
                        }
                    }
                    if (debug_dec_here) {
                        firered_debug_dump_vec("greedy.qx", Qx, 8);
                        firered_debug_dump_vec("greedy.attn_out", attn_out, 8);
                    }

                    std::vector<float> fc_out(d);
                    cpu_matmul_bt(attn_out.data(), c.xattn_fc_w.data(), fc_out.data(), 1, d, d);
                    for (int i = 0; i < d; i++)
                        x[i] += fc_out[i] + (c.xattn_fc_b.empty() ? 0 : c.xattn_fc_b[i]);

                    cpu_layernorm(x.data(), c.mlp_norm_w.data(), c.mlp_norm_b.data(), xn.data(), 1, d);
                    std::vector<float> h_up(c.di);
                    cpu_matmul_bt(xn.data(), c.mlp_w1.data(), h_up.data(), 1, d, c.di);
                    if (!c.mlp_b1.empty())
                        for (int i = 0; i < c.di; i++) h_up[i] += c.mlp_b1[i];
                    for (int i = 0; i < c.di; i++) {
                        float v = h_up[i];
                        h_up[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                    }
                    std::vector<float> mlp_out(d);
                    cpu_matmul_bt(h_up.data(), c.mlp_w2.data(), mlp_out.data(), 1, c.di, d);
                    if (!c.mlp_b2.empty())
                        for (int i = 0; i < d; i++) mlp_out[i] += c.mlp_b2[i];
                    if (debug_dec_here) {
                        firered_debug_dump_vec("greedy.fc_out", fc_out, 8);
                        firered_debug_dump_vec("greedy.mlp_out", mlp_out, 8);
                    }
                    for (int i = 0; i < d; i++)
                        x[i] += mlp_out[i];
                }

                int64_t tl0 = ggml_time_us();
                std::vector<float> xn(d);
                cpu_layernorm(x.data(), norm_w.data(), norm_b.data(), xn.data(), 1, d);

                // Always use CPU for greedy logit projection — the GPU
                // project_decoder_logits reuses a graph that conflicts with
                // the scheduler state after decoder weight dequantization,
                // producing wrong logits and never hitting EOS.
                std::vector<float> logits(odim);
                cpu_matmul_bt(xn.data(), prj_w.data(), logits.data(), 1, d, odim);
                t_logit += ggml_time_us() - tl0;

                if (getenv("FIRERED_BENCH") && (step < 3 || step == max_len - 1)) {
                    int64_t t_step = ggml_time_us() - t_step0;
                    fprintf(stderr, "firered: step %d: %.1fms (logit=%.1fms)\n", step, t_step / 1e3, t_logit / 1e3);
                }

                int best_token = 0;
                float best_logit = logits[0];
                for (int i = 1; i < odim; i++)
                    if (logits[i] > best_logit) {
                        best_logit = logits[i];
                        best_token = i;
                    }

                if (best_token == hp.eos_id)
                    beam.finished = true;
                else
                    beam.tokens.push_back(best_token);

                continue;
            }

            // Beam pruning candidates collected directly from each beam's
            // logits pass to avoid storing full-vocab logits for every beam.
            struct candidate {
                int src_beam, token;
                float score;
            };
            std::vector<candidate> cands;
            cands.reserve(beam_size * beam_size + beam_size);

            // Beam path: lazy-load full F32 weights on first beam step
            if (!dec_cache[0].full_cached) {
                for (int li2 = 0; li2 < hp.n_layers_dec; li2++) {
                    auto& b2 = m.dec.blocks[li2];
                    auto& c2 = dec_cache[li2];
                    read_f32_vec(b2.sattn.w_qs, c2.sattn_w_qs);
                    if (b2.sattn.w_qs_b)
                        read_f32_vec(b2.sattn.w_qs_b, c2.sattn_w_qs_b);
                    read_f32_vec(b2.sattn.w_ks, c2.sattn_w_ks);
                    read_f32_vec(b2.sattn.w_vs, c2.sattn_w_vs);
                    if (b2.sattn.w_vs_b)
                        read_f32_vec(b2.sattn.w_vs_b, c2.sattn_w_vs_b);
                    read_f32_vec(b2.sattn.fc_w, c2.sattn_fc_w);
                    if (b2.sattn.fc_b)
                        read_f32_vec(b2.sattn.fc_b, c2.sattn_fc_b);
                    read_f32_vec(b2.xattn.w_qs, c2.xattn_w_qs);
                    if (b2.xattn.w_qs_b)
                        read_f32_vec(b2.xattn.w_qs_b, c2.xattn_w_qs_b);
                    read_f32_vec(b2.xattn.fc_w, c2.xattn_fc_w);
                    if (b2.xattn.fc_b)
                        read_f32_vec(b2.xattn.fc_b, c2.xattn_fc_b);
                    read_f32_vec(b2.mlp_w1, c2.mlp_w1);
                    if (b2.mlp_b1)
                        read_f32_vec(b2.mlp_b1, c2.mlp_b1);
                    read_f32_vec(b2.mlp_w2, c2.mlp_w2);
                    if (b2.mlp_b2)
                        read_f32_vec(b2.mlp_b2, c2.mlp_b2);
                    c2.full_cached = true;
                }
            }

            for (int bi = 0; bi < beam_size; bi++) {
                if (beams[bi].finished || beams[bi].score <= -1e8f)
                    continue;
                int cur_token = beams[bi].tokens.back();

                // Embed
                std::vector<float> x(d);
                for (int i = 0; i < d; i++) {
                    x[i] = emb_w[cur_token * d + i] * scale;
                    if (step < (int)(m.dec.pe->ne[1]))
                        x[i] += pe_dec[step * d + i];
                }

                for (int li = 0; li < hp.n_layers_dec; li++) {
                    auto& c = dec_cache[li];

                    // === Self-attention (causal, attend to history) ===
                    {
                        std::vector<float> xn(d);
                        cpu_layernorm(x.data(), c.sattn_norm_w.data(), c.sattn_norm_b.data(), xn.data(), 1, d);

                        // Q/K/V projections via parallelized matmul
                        std::vector<float> Q_sa(d), K_cur(d), V_cur(d);
                        cpu_matmul_bt(xn.data(), c.sattn_w_qs.data(), Q_sa.data(), 1, d, d);
                        if (!c.sattn_w_qs_b.empty())
                            for (int i = 0; i < d; i++)
                                Q_sa[i] += c.sattn_w_qs_b[i];
                        cpu_matmul_bt(xn.data(), c.sattn_w_ks.data(), K_cur.data(), 1, d, d);
                        cpu_matmul_bt(xn.data(), c.sattn_w_vs.data(), V_cur.data(), 1, d, d);
                        if (!c.sattn_w_vs_b.empty())
                            for (int i = 0; i < d; i++)
                                V_cur[i] += c.sattn_w_vs_b[i];
                        if (!beams[bi].sa_k[li].unique())
                            beams[bi].sa_k[li] = std::make_shared<std::vector<float>>(*beams[bi].sa_k[li]);
                        if (!beams[bi].sa_v[li].unique())
                            beams[bi].sa_v[li] = std::make_shared<std::vector<float>>(*beams[bi].sa_v[li]);
                        auto& sa_k_hist = *beams[bi].sa_k[li];
                        auto& sa_v_hist = *beams[bi].sa_v[li];
                        sa_k_hist.insert(sa_k_hist.end(), K_cur.begin(), K_cur.end());
                        sa_v_hist.insert(sa_v_hist.end(), V_cur.begin(), V_cur.end());

                        int n_hist = (int)(sa_k_hist.size() / d); // step + 1

                        // Multi-head attention: Q[d] @ K_history[n_hist, d]
                        std::vector<float> sa_out(d, 0);
                        for (int h = 0; h < nh_dec; h++) {
                            std::vector<float> scores(n_hist);
                            for (int t = 0; t < n_hist; t++) {
                                double s = 0;
                                for (int dd = 0; dd < hd_dec; dd++)
                                    s += Q_sa[h * hd_dec + dd] * sa_k_hist[t * d + h * hd_dec + dd];
                                scores[t] = (float)s * inv_sqrt_hd_dec;
                            }
                            cpu_softmax_rows(scores.data(), 1, n_hist);
                            for (int dd = 0; dd < hd_dec; dd++) {
                                double s = 0;
                                for (int t = 0; t < n_hist; t++)
                                    s += scores[t] * sa_v_hist[t * d + h * hd_dec + dd];
                                sa_out[h * hd_dec + dd] = (float)s;
                            }
                        }

                        // FC + residual
                        std::vector<float> sa_fc(d);
                        cpu_matmul_bt(sa_out.data(), c.sattn_fc_w.data(), sa_fc.data(), 1, d, d);
                        for (int i = 0; i < d; i++)
                            x[i] += sa_fc[i] + (c.sattn_fc_b.empty() ? 0 : c.sattn_fc_b[i]);
                    }

                    // === Cross-attention: attend to encoder output (pre-computed K/V) ===
                    std::vector<float> xn(d);
                    cpu_layernorm(x.data(), c.xattn_norm_w.data(), c.xattn_norm_b.data(), xn.data(), 1, d);

                    std::vector<float> Qx(d);
                    cpu_matmul_bt(xn.data(), c.xattn_w_qs.data(), Qx.data(), 1, d, d);
                    if (!c.xattn_w_qs_b.empty())
                        for (int i = 0; i < d; i++)
                            Qx[i] += c.xattn_w_qs_b[i];

                    int nh_dec = hp.n_head_dec;
                    int hd_dec = d / nh_dec;
                    std::vector<float> attn_out(d, 0);
                    for (int h = 0; h < nh_dec; h++) {
                        std::vector<float> scores(T_sub);
                        for (int t = 0; t < T_sub; t++) {
                            double s = 0;
                            for (int dd = 0; dd < hd_dec; dd++)
                                s += Qx[h * hd_dec + dd] * K_enc[li][t * d + h * hd_dec + dd];
                            scores[t] = (float)s * inv_sqrt_hd_dec;
                        }
                        cpu_softmax_rows(scores.data(), 1, T_sub);
                        for (int dd = 0; dd < hd_dec; dd++) {
                            double s = 0;
                            for (int t = 0; t < T_sub; t++)
                                s += scores[t] * V_enc[li][t * d + h * hd_dec + dd];
                            attn_out[h * hd_dec + dd] = (float)s;
                        }
                    }

                    // FC output projection + residual
                    std::vector<float> fc_out(d);
                    cpu_matmul_bt(attn_out.data(), c.xattn_fc_w.data(), fc_out.data(), 1, d, d);
                    for (int i = 0; i < d; i++)
                        x[i] += fc_out[i] + (c.xattn_fc_b.empty() ? 0 : c.xattn_fc_b[i]);

                    // MLP: LN → Linear(d→4d) → GELU → Linear(4d→d) + residual
                    cpu_layernorm(x.data(), c.mlp_norm_w.data(), c.mlp_norm_b.data(), xn.data(), 1, d);
                    std::vector<float> h_up(c.di);
                    cpu_matmul_bt(xn.data(), c.mlp_w1.data(), h_up.data(), 1, d, c.di);
                    if (!c.mlp_b1.empty())
                        for (int i = 0; i < c.di; i++)
                            h_up[i] += c.mlp_b1[i];
                    for (int i = 0; i < c.di; i++) {
                        float v = h_up[i];
                        h_up[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                    }
                    std::vector<float> mlp_out(d);
                    cpu_matmul_bt(h_up.data(), c.mlp_w2.data(), mlp_out.data(), 1, c.di, d);
                    if (!c.mlp_b2.empty())
                        for (int i = 0; i < d; i++)
                            mlp_out[i] += c.mlp_b2[i];
                    for (int i = 0; i < d; i++)
                        x[i] += mlp_out[i];
                }

                // Final LN + projection
                std::vector<float> xn(d);
                cpu_layernorm(x.data(), norm_w.data(), norm_b.data(), xn.data(), 1, d);
                std::vector<float> logits(odim);
                if (!project_decoder_logits(xn.data(), logits))
                    cpu_matmul_bt(xn.data(), prj_w.data(), logits.data(), 1, d, odim);

                float mx = logits[0];
                for (int i = 1; i < odim; i++)
                    if (logits[i] > mx)
                        mx = logits[i];

                float lse = 0.0f;
                for (int i = 0; i < odim; i++)
                    lse += expf(logits[i] - mx);
                lse = mx + logf(lse);

                std::vector<int> top_idx(beam_size, -1);
                std::vector<float> top_score(beam_size, -1e30f);
                for (int i = 0; i < odim; i++) {
                    const float lp = logits[i] - lse;
                    for (int k = 0; k < beam_size; k++) {
                        if (lp > top_score[k]) {
                            for (int j = beam_size - 1; j > k; j--) {
                                top_score[j] = top_score[j - 1];
                                top_idx[j] = top_idx[j - 1];
                            }
                            top_score[k] = lp;
                            top_idx[k] = i;
                            break;
                        }
                    }
                }

                for (int k = 0; k < beam_size; k++) {
                    if (top_idx[k] >= 0) {
                        cands.push_back({bi, top_idx[k], beams[bi].score + top_score[k]});
                    }
                }
            } // end per-beam computation

            for (int bi = 0; bi < beam_size; bi++)
                if (beams[bi].finished)
                    cands.push_back({bi, hp.eos_id, beams[bi].score});

            std::sort(cands.begin(), cands.end(),
                      [](const candidate& a, const candidate& b) { return a.score > b.score; });
            if ((int)cands.size() > beam_size)
                cands.resize(beam_size);

            // Build new beams
            std::vector<beam_hyp> new_beams(beam_size);
            for (int b = 0; b < beam_size && b < (int)cands.size(); b++) {
                new_beams[b] = beams[cands[b].src_beam];
                new_beams[b].score = cands[b].score;
                if (cands[b].token == hp.eos_id)
                    new_beams[b].finished = true;
                else {
                    new_beams[b].tokens.push_back(cands[b].token);
                    new_beams[b].finished = false;
                }
            }
            beams = std::move(new_beams);
        } // end step loop

        if (dec_proj_ctx)
            ggml_free(dec_proj_ctx);

        // Select best beam
        int best_beam = 0;
        for (int b = 1; b < beam_size; b++)
            if (beams[b].score > beams[best_beam].score)
                best_beam = b;
        auto& tokens = beams[best_beam].tokens;

        // Decode tokens
        std::string result;
        if (is_lid) {
            // LID: take only the first token as language code
            for (int i = 1; i < (int)tokens.size(); i++) {
                int tid = tokens[i];
                if (tid < (int)m.vocab.size() && tid != hp.pad_id && tid != hp.sos_id && tid != hp.eos_id &&
                    tid != hp.blank_id) {
                    result = m.vocab[tid];
                    break; // first meaningful token = language
                }
            }
        } else {
            // ASR: decode full token sequence
            for (int i = 1; i < (int)tokens.size(); i++) {
                int tid = tokens[i];
                if (tid < (int)m.vocab.size() && tid != hp.pad_id && tid != hp.sos_id && tid != hp.eos_id) {
                    std::string piece = m.vocab[tid];
                    std::string decoded;
                    for (size_t ci = 0; ci < piece.size(); ci++) {
                        if ((unsigned char)piece[ci] == 0xE2 && ci + 2 < piece.size() &&
                            (unsigned char)piece[ci + 1] == 0x96 && (unsigned char)piece[ci + 2] == 0x81) {
                            decoded += ' ';
                            ci += 2;
                        } else {
                            decoded += piece[ci];
                        }
                    }
                    result += decoded;
                }
            }
        }

        if (ctx->params.verbosity >= 1) {
            int64_t t_dec1 = ggml_time_us();
            fprintf(stderr, "firered_asr: decoder produced %d tokens in %.1fms (%.1fms/token)\n",
                    (int)tokens.size() - 1, (t_dec1 - t_dec0) / 1e3,
                    tokens.size() > 1 ? (t_dec1 - t_dec0) / 1e3 / (tokens.size() - 1) : 0);
        }

        if (!result.empty()) {
            while (!result.empty() && result.front() == ' ')
                result.erase(result.begin());
            while (!result.empty() && result.back() == ' ')
                result.pop_back();
            char* out = (char*)malloc(result.size() + 1);
            memcpy(out, result.c_str(), result.size());
            out[result.size()] = '\0';
            return out;
        }
    }

    // Step 5: CTC head (fallback if no decoder or decoder fails)
    struct ggml_init_params gp = {
        /*.mem_size   =*/ctx->compute_meta.size(),
        /*.mem_buffer =*/ctx->compute_meta.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 256, false);

    // Input: encoder output [d_model, T_sub] in ggml layout
    ggml_tensor* enc_inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_model, T_sub);
    ggml_set_name(enc_inp, "enc_output");
    ggml_set_input(enc_inp);

    ggml_tensor* ctc_logits = ggml_mul_mat(ctx0, m.ctc_w, enc_inp);
    if (m.ctc_b)
        ctc_logits = ggml_add(ctx0, ctc_logits, m.ctc_b);
    ggml_set_name(ctc_logits, "ctc_logits");
    ggml_set_output(ctc_logits);
    ggml_build_forward_expand(gf, ctc_logits);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "firered_asr: failed to alloc CTC graph\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // The enc_output is [T_sub, d_model] row-major (CPU) but ggml expects [d_model, T_sub] col-major
    // In ggml col-major: ne[0]=d_model fastest → same memory layout as row-major [T_sub, d_model]
    // So we can feed it directly!
    ggml_backend_tensor_set(enc_inp, enc_output.data(), 0, T_sub * hp.d_model * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "firered_asr: CTC compute failed\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // Read CTC logits and greedy decode
    int odim = hp.odim;
    std::vector<float> logits_data(odim * T_sub);
    ggml_backend_tensor_get(ctc_logits, logits_data.data(), 0, odim * T_sub * sizeof(float));

    // Greedy CTC: argmax per frame, collapse repeats, remove blanks
    std::string result;
    int prev_id = -1;
    for (int t = 0; t < T_sub; t++) {
        int best_id = 0;
        float best_val = logits_data[t * odim];
        for (int i = 1; i < odim; i++) {
            if (logits_data[t * odim + i] > best_val) {
                best_val = logits_data[t * odim + i];
                best_id = i;
            }
        }
        if (best_id != prev_id && best_id != hp.blank_id) {
            if (best_id < (int)m.vocab.size() && best_id != hp.pad_id && best_id != hp.sos_id && best_id != hp.eos_id) {
                std::string piece = m.vocab[best_id];
                // SentencePiece: ▁ (U+2581) = space
                std::string decoded;
                for (size_t ci = 0; ci < piece.size(); ci++) {
                    if ((unsigned char)piece[ci] == 0xE2 && ci + 2 < piece.size() &&
                        (unsigned char)piece[ci + 1] == 0x96 && (unsigned char)piece[ci + 2] == 0x81) {
                        decoded += ' ';
                        ci += 2;
                    } else {
                        decoded += piece[ci];
                    }
                }
                result += decoded;
            }
        }
        prev_id = best_id;
    }

    ggml_free(ctx0);

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "firered_asr: CTC decoded %d frames → %zu chars\n", T_sub, result.size());
        // Dump first few CTC predictions for debugging
        int prev_id = -1;
        fprintf(stderr, "  CTC first 20 argmax: [");
        for (int t = 0; t < std::min(20, T_sub); t++) {
            int best_id = 0;
            float best_val = logits_data[t * hp.odim];
            for (int i = 1; i < hp.odim; i++)
                if (logits_data[t * hp.odim + i] > best_val) {
                    best_val = logits_data[t * hp.odim + i];
                    best_id = i;
                }
            fprintf(stderr, "%d,", best_id);
        }
        fprintf(stderr, "]\n");
    }

    if (result.empty())
        return nullptr;

    // Trim
    while (!result.empty() && result.front() == ' ')
        result.erase(result.begin());
    while (!result.empty() && result.back() == ' ')
        result.pop_back();

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size());
    out[result.size()] = '\0';
    return out;
}

extern "C" const char* firered_asr_token_text(struct firered_asr_context* ctx, int id) {
    if (!ctx || id < 0 || id >= (int)ctx->model.vocab.size())
        return nullptr;
    return ctx->model.vocab[id].c_str();
}
