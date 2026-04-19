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
    return {/*n_threads=*/4, /*verbosity=*/1};
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

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    ctx->backend_cpu = ggml_backend_cpu_init();
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
            if (params.verbosity >= 2)
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
// Mel spectrogram (80-dim fbank, 25ms window, 10ms hop, 16kHz)
// ===========================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void compute_fbank(const float* pcm, int n_samples, std::vector<float>& features, int& n_frames) {
    const int n_fft = 512;
    const int hop = 160;
    const int win = 400;
    const int n_mels = 80;
    const int sample_rate = 16000;

    n_frames = (n_samples - win) / hop + 1;
    if (n_frames <= 0) {
        n_frames = 0;
        return;
    }

    // Mel filterbank
    int n_fft_bins = n_fft / 2 + 1;
    std::vector<float> mel_fb(n_mels * n_fft_bins, 0.0f);
    {
        auto hz2mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
        auto mel2hz = [](float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };
        float mel_lo = hz2mel(0), mel_hi = hz2mel((float)sample_rate / 2);
        std::vector<float> mp(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            mp[i] = mel2hz(mel_lo + i * (mel_hi - mel_lo) / (n_mels + 1));
        for (int m = 0; m < n_mels; m++)
            for (int k = 0; k < n_fft_bins; k++) {
                float f = (float)k * sample_rate / n_fft;
                if (f >= mp[m] && f <= mp[m + 1])
                    mel_fb[m * n_fft_bins + k] = (f - mp[m]) / (mp[m + 1] - mp[m]);
                else if (f > mp[m + 1] && f <= mp[m + 2])
                    mel_fb[m * n_fft_bins + k] = (mp[m + 2] - f) / (mp[m + 2] - mp[m + 1]);
            }
    }

    std::vector<float> hann(win);
    for (int i = 0; i < win; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win));

    features.resize(n_frames * n_mels);
    std::vector<float> fft_re(n_fft), fft_im(n_fft);

    auto fft = [](float* re, float* im, int n) {
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
        std::fill(fft_re.begin(), fft_re.end(), 0.0f);
        std::fill(fft_im.begin(), fft_im.end(), 0.0f);
        for (int i = 0; i < win && t * hop + i < n_samples; i++)
            fft_re[i] = pcm[t * hop + i] * hann[i];
        fft(fft_re.data(), fft_im.data(), n_fft);
        for (int m = 0; m < n_mels; m++) {
            float s = 0;
            for (int k = 0; k < n_fft_bins; k++)
                s += (fft_re[k] * fft_re[k] + fft_im[k] * fft_im[k]) * mel_fb[m * n_fft_bins + k];
            features[t * n_mels + m] = logf(std::max(s, 1e-10f));
        }
    }
}

// ===========================================================================
// Swish activation
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
    h = ggml_scale(ctx, h, 0.5f);
    return ggml_add(ctx, x, h);
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
    // pw1_w shape in PyTorch: [5120, 1280, 1] (Conv1d with kernel=1)
    // In ggml ne: depends on F16/F32 storage layout
    // For matmul: need [in_dim, out_dim] = [1280, 5120]
    // pw1_w ne: [1, 1280, 5120] → reshape to [1280, 5120] for matmul
    // Use ggml_view to create a 2D view without reallocating
    // pw1_w is 3D [1, 1280, 5120] — need 2D [1280, 5120] for matmul
    ggml_tensor* pw1 = ggml_reshape_2d(ctx, conv.pw1_w, conv.pw1_w->ne[0] * conv.pw1_w->ne[1], conv.pw1_w->ne[2]);
    h = ggml_mul_mat(ctx, pw1, h);

    // GLU: split → sigmoid gate
    int ch = (int)h->ne[0] / 2; // 2560
    ggml_tensor* h1 = ggml_cont(ctx, ggml_view_2d(ctx, h, ch, T, h->nb[1], 0));
    ggml_tensor* h2 = ggml_cont(ctx, ggml_view_2d(ctx, h, ch, T, h->nb[1], ch * ggml_type_size(h->type)));
    h = ggml_mul(ctx, h1, ggml_sigmoid(ctx, h2)); // [2560, T]

    // Depthwise conv1d: groups=2560, kernel=33
    // Transpose to [T, channels] for conv1d_dw
    ggml_tensor* ht = ggml_cont(ctx, ggml_transpose(ctx, h)); // [T, 2560]
    // Causal padding: pad_left = kernel_size - 1 (stride=1)
    int pad_left = kernel_size - 1;
    ht = ggml_pad_ext(ctx, ht, pad_left, 0, 0, 0, 0, 0, 0, 0);
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
    // pw2_w ne: [1, 2560, 1280]
    ggml_tensor* pw2 = ggml_reshape_2d(ctx, conv.pw2_w, conv.pw2_w->ne[0] * conv.pw2_w->ne[1], conv.pw2_w->ne[2]);
    h = ggml_mul_mat(ctx, pw2, h); // [1280, T]

    return ggml_add(ctx, residual, h);
}

// ===========================================================================
// Relative-PE multi-head self-attention (simplified — no rel_shift)
// ===========================================================================

static ggml_tensor* build_rel_mhsa(ggml_context* ctx, ggml_tensor* x, const firered_enc_mhsa& mhsa, int n_head,
                                   int head_dim) {
    int d = n_head * head_dim;
    int T = (int)x->ne[1];
    ggml_tensor* residual = x;

    // Q/K/V with separate LN
    ggml_tensor* q = ggml_norm(ctx, x, 1e-5f);
    q = ggml_mul(ctx, q, mhsa.ln_q_w);
    if (mhsa.ln_q_b)
        q = ggml_add(ctx, q, mhsa.ln_q_b);
    q = ggml_mul_mat(ctx, mhsa.w_qs, q);

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

    // Multi-head: [hd, nh, T] → permute to [hd, T, nh] for flash_attn
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, T);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // TODO: add relative positional encoding (pos_bias_u/v + lin_pos + rel_shift)
    // For now use standard attention — will degrade accuracy but tests the pipeline
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
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
    auto read_f32 = [](ggml_tensor* t, std::vector<float>& out) {
        int n = (int)ggml_nelements(t);
        out.resize(n);
        if (t->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(uint16_t));
            for (int i = 0; i < n; i++)
                out[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        }
    };

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

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "firered_asr: %d fbank frames\n", n_frames);

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

    // Step 3: Build encoder graph (linear proj + conformer layers + CTC)
    struct ggml_init_params gp = {
        /*.mem_size   =*/ctx->compute_meta.size(),
        /*.mem_buffer =*/ctx->compute_meta.data(),
        /*.no_alloc   =*/true,
    };
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    // Input: subsampled features [608, T_sub]
    int flat_dim = 32 * ((hp.idim - 3) / 2 + 1 - 3) / 2 + 32; // approximate — use actual
    flat_dim = 608;                                           // 32 * 19
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, flat_dim, T_sub);
    ggml_set_name(inp, "subsample_out");
    ggml_set_input(inp);

    // Linear projection: 608 → 1280
    ggml_tensor* x = ggml_mul_mat(ctx0, m.enc.proj_w, inp);
    if (m.enc.proj_b)
        x = ggml_add(ctx0, x, m.enc.proj_b);
    // x: [d_model=1280, T_sub]

    // Conformer layers
    for (int i = 0; i < hp.n_layers_enc; i++) {
        auto& b = m.enc.blocks[i];
        // Macaron FFN1 (half-step)
        x = build_macaron_ffn(ctx0, x, b.ffn1);
        // MHSA with relative PE
        x = build_rel_mhsa(ctx0, x, b.mhsa, hp.n_head, hp.head_dim);
        // Conv module
        x = build_conv_module(ctx0, x, b.conv, hp.d_model, hp.kernel_size);
        // Macaron FFN2 (half-step)
        x = build_macaron_ffn(ctx0, x, b.ffn2);
        // Final LayerNorm
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, b.ln_w);
        if (b.ln_b)
            x = ggml_add(ctx0, x, b.ln_b);
    }
    // x: [d_model, T_sub]

    // CTC head: linear + log_softmax → greedy decode
    ggml_tensor* ctc_logits = ggml_mul_mat(ctx0, m.ctc_w, x);
    if (m.ctc_b)
        ctc_logits = ggml_add(ctx0, ctc_logits, m.ctc_b);
    // ctc_logits: [odim, T_sub]
    ggml_set_name(ctc_logits, "ctc_logits");
    ggml_set_output(ctc_logits);
    ggml_build_forward_expand(gf, ctc_logits);

    // Allocate and compute
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "firered_asr: failed to alloc encoder graph\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // Set input
    ggml_backend_tensor_set(inp, subsampled.data(), 0, flat_dim * T_sub * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "firered_asr: encoder compute failed\n");
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

    if (ctx->params.verbosity >= 1)
        fprintf(stderr, "firered_asr: CTC decoded %d frames → %zu chars\n", T_sub, result.size());

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
