// openvoice2.cpp — OpenVoice V2 Tone Color Converter runtime.
//
// Voice conversion pipeline:
//   1. STFT(src_audio) → src_spec (513 × T_src)
//   2. STFT(ref_audio) → ref_spec (513 × T_ref)
//   3. ref_enc(ref_spec) → target_se (256-d speaker embedding)
//   4. enc_q(src_spec, g=0) → z (192 × T_src)  [zero_g mode]
//   5. flow_forward(z, g=src_se) → z_p  [to prior space]
//   6. flow_reverse(z_p, g=target_se) → z_hat  [from prior space with target voice]
//   7. hifigan_decode(z_hat, g=0) → audio  [zero_g mode]
//
// The flow is the ONLY component that uses speaker conditioning (zero_g=true).
// This means enc_q and dec ignore g, and voice conversion happens entirely
// through the normalizing flow's speaker-conditioned affine transforms.

#include "openvoice2.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/conv.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Helpers ──────────────────────────────────────────────────────────

static void read_f32(const ggml_tensor* t, std::vector<float>& out) {
    int64_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        fprintf(stderr, "openvoice2: unsupported tensor type %d for read_f32\n", t->type);
        std::fill(out.begin(), out.end(), 0.0f);
    }
}

// Local conv1d_cf helper (channels-first conv1d via ggml)
static ggml_tensor* conv1d_cf(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride = 1,
                              int pad = 0, int dilation = 1) {
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* y = ggml_conv_1d(ctx, w, xT, stride, pad, dilation);
    y = ggml_cont(ctx, ggml_transpose(ctx, y));
    if (b)
        y = ggml_add(ctx, y, b);
    return y;
}

// ── Hyperparameters ──────────────────────────────────────────────────

struct openvoice2_hparams {
    int inter_channels = 192;
    int hidden_channels = 192;
    int filter_channels = 768;
    int gin_channels = 256;
    int sample_rate = 22050;
    int hop_length = 256;
    int win_length = 1024;
    int filter_length = 1024;
    int spec_channels = 513;
    int n_mels = 80;
    bool zero_g = true;

    int n_wn_layers_enc_q = 16;
    int n_flow_blocks = 4;
    int n_wn_layers_flow = 4;
    int n_upsample_stages = 4;
    int n_resblocks = 12;
    int n_ref_convs = 6;

    std::vector<int> upsample_rates;
    std::vector<int> upsample_kernel_sizes;
    std::vector<int> resblock_kernel_sizes;
    std::vector<int> resblock_dilation_sizes;
};

// ── Weight structures ────────────────────────────────────────────────

struct ov2_wn_layer {
    ggml_tensor* in_w; // dilated conv weight
    ggml_tensor* in_b; // dilated conv bias
    ggml_tensor* res_skip_w;
    ggml_tensor* res_skip_b;
};

struct ov2_wn_block {
    ggml_tensor* cond_w; // speaker conditioning conv (gin → 2*hidden)
    ggml_tensor* cond_b;
    std::vector<ov2_wn_layer> layers;
};

struct ov2_flow_block {
    ggml_tensor* pre_w; // Conv1d (half → hidden, k=1)
    ggml_tensor* pre_b;
    ggml_tensor* post_w; // Conv1d (hidden → half, k=1)
    ggml_tensor* post_b;
    ov2_wn_block wn; // inner WaveNet
};

struct ov2_ref_enc {
    // 6 Conv2d layers (weight-norm fused)
    struct conv2d_layer {
        ggml_tensor* w;
        ggml_tensor* b;
    };
    std::vector<conv2d_layer> convs;
    // GRU
    ggml_tensor* gru_w_ih;
    ggml_tensor* gru_w_hh;
    ggml_tensor* gru_b_ih;
    ggml_tensor* gru_b_hh;
    // LayerNorm
    ggml_tensor* ln_w;
    ggml_tensor* ln_b;
    // Output projection
    ggml_tensor* proj_w;
    ggml_tensor* proj_b;
};

struct ov2_enc_q {
    ggml_tensor* pre_w; // Conv1d (spec_channels → hidden, k=1)
    ggml_tensor* pre_b;
    ggml_tensor* proj_w; // Conv1d (hidden → 2*inter, k=1)
    ggml_tensor* proj_b;
    ov2_wn_block wn; // 16-layer WaveNet
};

struct ov2_hifigan {
    ggml_tensor* conv_pre_w;
    ggml_tensor* conv_pre_b;
    struct upsample_stage {
        ggml_tensor* w;
        ggml_tensor* b;
        ggml_tensor* w_perm = nullptr;
    };
    std::vector<upsample_stage> ups;
    struct resblock {
        ggml_tensor* convs1[3]; // 3 dilated convs
        ggml_tensor* convs1_b[3];
        ggml_tensor* convs2[3]; // 3 post convs
        ggml_tensor* convs2_b[3];
    };
    std::vector<resblock> resblocks;
    ggml_tensor* conv_post_w;
    ggml_tensor* conv_post_b;
    ggml_tensor* cond_w; // speaker conditioning (gin → upsample_initial_channel)
    ggml_tensor* cond_b;
};

struct ov2_base_speaker {
    std::string name;
    ggml_tensor* embedding; // (gin_channels,) F32
};

struct ov2_weights {
    ov2_ref_enc ref_enc;
    ov2_enc_q enc_q;
    std::vector<ov2_flow_block> flow_blocks;
    ov2_hifigan dec;
    std::vector<ov2_base_speaker> base_speakers;
};

// ── Context ──────────────────────────────────────────────────────────

struct openvoice2_context {
    openvoice2_hparams hp;
    ov2_weights w;

    ggml_backend_t backend;
    ggml_backend_t backend_cpu;
    ggml_backend_sched_t sched;

    // Owned by core_gguf::WeightLoad
    ggml_context* w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;

    ggml_context* ctx_perm = nullptr;
    ggml_backend_buffer_t buf_perm = nullptr;

    int verbosity;
    float tau;
    std::mt19937 rng;
    std::string dump_dir;
};

// ── Diff dump helpers ────────────────────────────────────────────────

static void dump_stage(const openvoice2_context* ctx, const char* label, const float* data, size_t n) {
    if (ctx->dump_dir.empty())
        return;
    std::string path = ctx->dump_dir + "/" + label + ".bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data, sizeof(float), n, f);
        fclose(f);
    }
}

// ── GGUF loading ─────────────────────────────────────────────────────

static ggml_tensor* find_tensor(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return it != m.end() ? it->second : nullptr;
}

static ggml_tensor* require_tensor(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto* t = find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "openvoice2: missing tensor '%s'\n", name.c_str());
    }
    return t;
}

static bool load_wn_block(const std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix, int n_layers,
                          ov2_wn_block& out) {
    out.cond_w = find_tensor(tensors, prefix + ".cond_layer.weight");
    out.cond_b = find_tensor(tensors, prefix + ".cond_layer.bias");
    out.layers.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        std::string lp = prefix + ".in_layers." + std::to_string(i);
        out.layers[i].in_w = require_tensor(tensors, lp + ".weight");
        out.layers[i].in_b = require_tensor(tensors, lp + ".bias");
        std::string rp = prefix + ".res_skip_layers." + std::to_string(i);
        out.layers[i].res_skip_w = require_tensor(tensors, rp + ".weight");
        out.layers[i].res_skip_b = require_tensor(tensors, rp + ".bias");
        if (!out.layers[i].in_w || !out.layers[i].res_skip_w)
            return false;
    }
    return true;
}

extern "C" struct openvoice2_context_params openvoice2_context_default_params(void) {
    return {/*n_threads=*/4, /*verbosity=*/1, /*use_gpu=*/false, /*tau=*/0.0f};
}

extern "C" struct openvoice2_context* openvoice2_init_from_file(const char* path,
                                                                struct openvoice2_context_params params) {
    auto* ctx = new openvoice2_context();
    ctx->verbosity = params.verbosity;
    ctx->tau = params.tau;
    {
        const char* e = std::getenv("OV2_TAU");
        if (e)
            ctx->tau = (float)std::atof(e);
    }
    ctx->rng.seed(42);

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path);
    if (!meta) {
        fprintf(stderr, "openvoice2: failed to load '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    auto& hp = ctx->hp;
    hp.inter_channels = core_gguf::kv_u32(meta, "openvoice2.inter_channels", 192);
    hp.hidden_channels = core_gguf::kv_u32(meta, "openvoice2.hidden_channels", 192);
    hp.filter_channels = core_gguf::kv_u32(meta, "openvoice2.filter_channels", 768);
    hp.gin_channels = core_gguf::kv_u32(meta, "openvoice2.gin_channels", 256);
    hp.sample_rate = core_gguf::kv_u32(meta, "openvoice2.sample_rate", 22050);
    hp.hop_length = core_gguf::kv_u32(meta, "openvoice2.hop_length", 256);
    hp.win_length = core_gguf::kv_u32(meta, "openvoice2.win_length", 1024);
    hp.filter_length = core_gguf::kv_u32(meta, "openvoice2.filter_length", 1024);
    hp.spec_channels = core_gguf::kv_u32(meta, "openvoice2.spec_channels", 513);
    hp.n_mels = core_gguf::kv_u32(meta, "openvoice2.n_mels", 80);
    hp.n_wn_layers_enc_q = core_gguf::kv_u32(meta, "openvoice2.n_wn_layers_enc_q", 16);
    hp.n_flow_blocks = core_gguf::kv_u32(meta, "openvoice2.n_flow_blocks", 4);
    hp.n_wn_layers_flow = core_gguf::kv_u32(meta, "openvoice2.n_wn_layers_flow", 4);
    hp.n_upsample_stages = core_gguf::kv_u32(meta, "openvoice2.n_upsample_stages", 4);
    hp.n_resblocks = core_gguf::kv_u32(meta, "openvoice2.n_resblocks", 12);
    hp.n_ref_convs = core_gguf::kv_u32(meta, "openvoice2.n_ref_convs", 6);

    // Read arrays from metadata
    {
        int idx = gguf_find_key(meta, "openvoice2.upsample_rates");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(meta, idx);
            hp.upsample_rates.resize(n);
            for (int i = 0; i < n; i++)
                hp.upsample_rates[i] = (int)((const int32_t*)gguf_get_arr_data(meta, idx))[i];
        }
    }
    {
        int idx = gguf_find_key(meta, "openvoice2.upsample_kernel_sizes");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(meta, idx);
            hp.upsample_kernel_sizes.resize(n);
            for (int i = 0; i < n; i++)
                hp.upsample_kernel_sizes[i] = (int)((const int32_t*)gguf_get_arr_data(meta, idx))[i];
        }
    }
    {
        int idx = gguf_find_key(meta, "openvoice2.resblock_kernel_sizes");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(meta, idx);
            hp.resblock_kernel_sizes.resize(n);
            for (int i = 0; i < n; i++)
                hp.resblock_kernel_sizes[i] = (int)((const int32_t*)gguf_get_arr_data(meta, idx))[i];
        }
    }
    {
        int idx = gguf_find_key(meta, "openvoice2.resblock_dilation_sizes");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(meta, idx);
            hp.resblock_dilation_sizes.resize(n);
            for (int i = 0; i < n; i++)
                hp.resblock_dilation_sizes[i] = (int)((const int32_t*)gguf_get_arr_data(meta, idx))[i];
        }
    }

    // zero_g flag
    {
        int idx = gguf_find_key(meta, "openvoice2.zero_g");
        hp.zero_g = (idx >= 0) ? gguf_get_val_bool(meta, idx) : true;
    }

    gguf_free(meta);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "openvoice2: loaded — inter=%d hidden=%d gin=%d sr=%d\n", hp.inter_channels, hp.hidden_channels,
                hp.gin_channels, hp.sample_rate);
        fprintf(stderr, "openvoice2: enc_q %d WN layers, flow %d blocks x %d WN layers\n", hp.n_wn_layers_enc_q,
                hp.n_flow_blocks, hp.n_wn_layers_flow);
        fprintf(stderr, "openvoice2: dec %d upsample, %d resblocks, ref_enc %d convs\n", hp.n_upsample_stages,
                hp.n_resblocks, hp.n_ref_convs);
    }

    // Pass 2: load weights via core_gguf
    ctx->backend_cpu = ggml_backend_cpu_init();
    ctx->backend = ctx->backend_cpu;

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "openvoice2", wl)) {
        fprintf(stderr, "openvoice2: failed to load weights\n");
        ggml_backend_free(ctx->backend_cpu);
        delete ctx;
        return nullptr;
    }
    ctx->w_ctx = wl.ctx;
    ctx->w_buf = wl.buf;
    auto& tensors = wl.tensors;

    // Base speaker embeddings
    for (auto& [name, tensor] : tensors) {
        if (name.rfind("base_speaker.", 0) == 0) {
            std::string spk_name = name.substr(strlen("base_speaker."));
            ctx->w.base_speakers.push_back({spk_name, tensor});
        }
    }
    if (ctx->verbosity >= 1)
        fprintf(stderr, "openvoice2: loaded %zu tensors, %zu base speakers\n", tensors.size(),
                ctx->w.base_speakers.size());

    // Set up scheduler
    ggml_backend_t backends[2];
    int n_be = 0;
    backends[n_be++] = ctx->backend;
    if (ctx->backend_cpu != ctx->backend)
        backends[n_be++] = ctx->backend_cpu;
    ctx->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);

    // ── Map tensors to weight structures ──

    // ref_enc
    auto& re = ctx->w.ref_enc;
    re.convs.resize(hp.n_ref_convs);
    for (int i = 0; i < hp.n_ref_convs; i++) {
        std::string p = "ref_enc.convs." + std::to_string(i);
        re.convs[i].w = require_tensor(tensors, p + ".weight");
        re.convs[i].b = require_tensor(tensors, p + ".bias");
    }
    re.gru_w_ih = require_tensor(tensors, "ref_enc.gru.weight_ih_l0");
    re.gru_w_hh = require_tensor(tensors, "ref_enc.gru.weight_hh_l0");
    re.gru_b_ih = require_tensor(tensors, "ref_enc.gru.bias_ih_l0");
    re.gru_b_hh = require_tensor(tensors, "ref_enc.gru.bias_hh_l0");
    re.ln_w = find_tensor(tensors, "ref_enc.layernorm.weight");
    re.ln_b = find_tensor(tensors, "ref_enc.layernorm.bias");
    re.proj_w = require_tensor(tensors, "ref_enc.proj.weight");
    re.proj_b = require_tensor(tensors, "ref_enc.proj.bias");

    // enc_q
    auto& eq = ctx->w.enc_q;
    eq.pre_w = require_tensor(tensors, "enc_q.pre.weight");
    eq.pre_b = require_tensor(tensors, "enc_q.pre.bias");
    eq.proj_w = require_tensor(tensors, "enc_q.proj.weight");
    eq.proj_b = require_tensor(tensors, "enc_q.proj.bias");
    load_wn_block(tensors, "enc_q.enc", hp.n_wn_layers_enc_q, eq.wn);

    // flow blocks
    ctx->w.flow_blocks.resize(hp.n_flow_blocks);
    for (int i = 0; i < hp.n_flow_blocks; i++) {
        int idx = i * 2; // even indices (odd are Flip layers)
        std::string p = "flow.flows." + std::to_string(idx);
        auto& fb = ctx->w.flow_blocks[i];
        fb.pre_w = require_tensor(tensors, p + ".pre.weight");
        fb.pre_b = require_tensor(tensors, p + ".pre.bias");
        fb.post_w = require_tensor(tensors, p + ".post.weight");
        fb.post_b = require_tensor(tensors, p + ".post.bias");
        load_wn_block(tensors, p + ".enc", hp.n_wn_layers_flow, fb.wn);
    }

    // HiFi-GAN decoder
    auto& dec = ctx->w.dec;
    dec.conv_pre_w = require_tensor(tensors, "dec.conv_pre.weight");
    dec.conv_pre_b = require_tensor(tensors, "dec.conv_pre.bias");
    dec.conv_post_w = require_tensor(tensors, "dec.conv_post.weight");
    dec.conv_post_b = find_tensor(tensors, "dec.conv_post.bias"); // optional
    dec.cond_w = find_tensor(tensors, "dec.cond.weight");
    dec.cond_b = find_tensor(tensors, "dec.cond.bias");

    dec.ups.resize(hp.n_upsample_stages);
    for (int i = 0; i < hp.n_upsample_stages; i++) {
        std::string p = "dec.ups." + std::to_string(i);
        dec.ups[i].w = require_tensor(tensors, p + ".weight");
        dec.ups[i].b = require_tensor(tensors, p + ".bias");
    }

    // Permute ConvTranspose1d weights for decomposed path
    {
        const int n = hp.n_upsample_stages;
        std::vector<ggml_tensor*> srcs(n);
        std::vector<ggml_tensor**> dsts(n);
        for (int i = 0; i < n; i++) {
            srcs[i] = dec.ups[i].w;
            dsts[i] = &dec.ups[i].w_perm;
        }
        core_convt::permute_convt1d_weights_batch(srcs.data(), dsts.data(), n, ctx->backend, &ctx->ctx_perm,
                                                  &ctx->buf_perm);
    }

    dec.resblocks.resize(hp.n_resblocks);
    for (int i = 0; i < hp.n_resblocks; i++) {
        std::string p = "dec.resblocks." + std::to_string(i);
        for (int j = 0; j < 3; j++) {
            std::string c1 = p + ".convs1." + std::to_string(j);
            std::string c2 = p + ".convs2." + std::to_string(j);
            dec.resblocks[i].convs1[j] = require_tensor(tensors, c1 + ".weight");
            dec.resblocks[i].convs1_b[j] = require_tensor(tensors, c1 + ".bias");
            dec.resblocks[i].convs2[j] = require_tensor(tensors, c2 + ".weight");
            dec.resblocks[i].convs2_b[j] = require_tensor(tensors, c2 + ".bias");
        }
    }

    return ctx;
}

// ── STFT ─────────────────────────────────────────────────────────────

static void hann_window(int N, std::vector<float>& win) {
    win.resize(N);
    for (int i = 0; i < N; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / N));
}

static void stft_magnitude(const float* pcm, int n_samples, int fft_size, int hop, int win_len,
                           std::vector<float>& spec, int& T_out) {
    std::vector<float> win;
    hann_window(win_len, win);

    // Reflect-pad input (matches PyTorch's F.pad(..., mode='reflect'))
    int pad = (fft_size - hop) / 2;
    int padded_len = n_samples + 2 * pad;
    std::vector<float> padded(padded_len);
    for (int i = 0; i < pad; i++)
        padded[i] = pcm[std::min(pad - i, n_samples - 1)];
    for (int i = 0; i < n_samples; i++)
        padded[pad + i] = pcm[i];
    for (int i = 0; i < pad; i++)
        padded[pad + n_samples + i] = pcm[std::max(n_samples - 2 - i, 0)];

    int n_fft_bins = fft_size / 2 + 1;
    T_out = (padded_len - win_len) / hop + 1;
    if (T_out < 1)
        T_out = 1;
    spec.resize(n_fft_bins * T_out, 0.0f);

    for (int t = 0; t < T_out; t++) {
        int start = t * hop;
        for (int k = 0; k < n_fft_bins; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < win_len; n++) {
                int idx = start + n;
                if (idx < padded_len) {
                    float x = padded[idx] * win[n];
                    float angle = -2.0f * (float)M_PI * k * n / fft_size;
                    re += x * cosf(angle);
                    im += x * sinf(angle);
                }
            }
            // Match upstream: sqrt(re^2 + im^2 + 1e-6)
            spec[t * n_fft_bins + k] = sqrtf(re * re + im * im + 1e-6f);
        }
    }
}

// ── WaveNet forward ──────────────────────────────────────────────────
// Gated dilated convolution with speaker conditioning.

static void wavenet_forward(const ov2_wn_block& wn, int n_layers, int hidden, int T,
                            const std::vector<float>& x_in,   // (hidden, T)
                            const std::vector<float>& g_cond, // (2*hidden*n_layers, 1) or empty
                            std::vector<float>& out) {        // (hidden, T)
    // x_in layout: [t * hidden + c]
    std::vector<float> h = x_in; // working copy
    std::vector<float> skip_sum(hidden * T, 0.0f);

    for (int il = 0; il < n_layers; il++) {
        const auto& layer = wn.layers[il];

        // Read weights
        std::vector<float> w_in, b_in;
        read_f32(layer.in_w, w_in);
        read_f32(layer.in_b, b_in);

        int out_ch = (int)layer.in_b->ne[0]; // 2 * hidden (for gating)
        int k_size = (int)layer.in_w->ne[0];
        int dilation = 1; // OpenVoice2 uses dilation=1 for all WN layers

        // Dilated conv1d: (hidden, T) -> (2*hidden, T)
        int pad = (k_size - 1) * dilation / 2;
        std::vector<float> conv_out(out_ch * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < out_ch; oc++) {
                float sum = b_in[oc];
                for (int ki = 0; ki < k_size; ki++) {
                    int ti = t + (ki - pad) * dilation;
                    if (ti >= 0 && ti < T) {
                        for (int ic = 0; ic < hidden; ic++) {
                            sum += h[ti * hidden + ic] * w_in[ki + ic * k_size + oc * k_size * hidden];
                        }
                    }
                }
                conv_out[t * out_ch + oc] = sum;
            }
        }

        // Add speaker conditioning if available
        if (!g_cond.empty() && wn.cond_w) {
            // g_cond is pre-computed: cond_layer(g) -> (2*hidden*n_layers, 1)
            // Slice for this layer: offset = il * 2 * hidden
            int offset = il * out_ch;
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < out_ch; c++) {
                    conv_out[t * out_ch + c] += g_cond[offset + c];
                }
            }
        }

        // Gated activation: tanh(first_half) * sigmoid(second_half)
        std::vector<float> gated(hidden * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < hidden; c++) {
                float t_val = tanhf(conv_out[t * out_ch + c]);
                float s_val = 1.0f / (1.0f + expf(-conv_out[t * out_ch + hidden + c]));
                gated[t * hidden + c] = t_val * s_val;
            }
        }

        // Res + skip projection
        std::vector<float> w_rs, b_rs;
        read_f32(layer.res_skip_w, w_rs);
        read_f32(layer.res_skip_b, b_rs);
        int rs_out = (int)layer.res_skip_b->ne[0];

        std::vector<float> rs(rs_out * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < rs_out; oc++) {
                float sum = b_rs[oc];
                for (int ic = 0; ic < hidden; ic++)
                    sum += gated[t * hidden + ic] * w_rs[ic + oc * hidden]; // k=1 conv
                rs[t * rs_out + oc] = sum;
            }
        }

        if (rs_out == 2 * hidden) {
            // Split: first half = residual, second half = skip
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < hidden; c++) {
                    h[t * hidden + c] += rs[t * rs_out + c];
                    skip_sum[t * hidden + c] += rs[t * rs_out + hidden + c];
                }
            }
        } else {
            // Last layer: all goes to skip
            for (int t = 0; t < T; t++)
                for (int c = 0; c < rs_out && c < hidden; c++)
                    skip_sum[t * hidden + c] += rs[t * rs_out + c];
        }
    }

    out = skip_sum;
}

// ── Speaker conditioning projection ──────────────────────────────────

static void compute_g_cond(const ov2_wn_block& wn, const std::vector<float>& g_vec, int gin, int n_layers, int hidden,
                           std::vector<float>& g_cond) {
    if (!wn.cond_w || g_vec.empty()) {
        g_cond.clear();
        return;
    }
    // cond_layer is Conv1d(gin, 2*hidden*n_layers, k=1)
    int out_ch = 2 * hidden * n_layers;
    g_cond.resize(out_ch, 0.0f);

    std::vector<float> w_cond, b_cond;
    read_f32(wn.cond_w, w_cond);
    read_f32(wn.cond_b, b_cond);

    for (int oc = 0; oc < out_ch; oc++) {
        float sum = b_cond[oc];
        for (int ic = 0; ic < gin; ic++)
            sum += g_vec[ic] * w_cond[ic + oc * gin];
        g_cond[oc] = sum;
    }
}

// ── Reference encoder ────────────────────────────────────────────────
// 6 Conv2d + GRU + proj → 256-d speaker embedding

static bool ref_enc_forward(openvoice2_context* ctx, const std::vector<float>& spec, int T_spec,
                            std::vector<float>& out_emb) {
    const auto& hp = ctx->hp;
    const auto& re = ctx->w.ref_enc;

    int n_bins = hp.spec_channels; // 513

    // LayerNorm on input spectrogram
    std::vector<float> x = spec; // (T_spec, n_bins) row-major
    if (re.ln_w && re.ln_b) {
        std::vector<float> ln_w, ln_b;
        read_f32(re.ln_w, ln_w);
        read_f32(re.ln_b, ln_b);
        for (int t = 0; t < T_spec; t++) {
            float mean = 0, var = 0;
            for (int c = 0; c < n_bins; c++)
                mean += x[t * n_bins + c];
            mean /= n_bins;
            for (int c = 0; c < n_bins; c++) {
                float d = x[t * n_bins + c] - mean;
                var += d * d;
            }
            var /= n_bins;
            float inv_std = 1.0f / sqrtf(var + 1e-5f);
            for (int c = 0; c < n_bins; c++) {
                x[t * n_bins + c] = (x[t * n_bins + c] - mean) * inv_std * ln_w[c] + ln_b[c];
            }
        }
    }

    dump_stage(ctx, "ref_enc_layernorm", x.data(), x.size());

    // Reshape to (batch=1, C=1, H=T_spec, W=n_bins) for Conv2d.
    // Python: x.transpose(1,2) → (1,T,513) → unsqueeze(1) → (1,1,T,513)
    // So H=T_spec, W=n_bins — NOT transposed from (T,bins) layout.
    int H = T_spec, W = n_bins, C_in = 1;
    std::vector<float> feat(C_in * H * W);
    // x is already (T, bins) — copy directly as (H=T, W=bins)
    memcpy(feat.data(), x.data(), H * W * sizeof(float));

    static const int filters[] = {1, 32, 32, 64, 64, 128, 128};

    for (int li = 0; li < hp.n_ref_convs; li++) {
        int C_out = filters[li + 1];
        C_in = filters[li];

        std::vector<float> w_conv, b_conv;
        read_f32(re.convs[li].w, w_conv);
        read_f32(re.convs[li].b, b_conv);

        // Conv2d: k=3x3, stride=2x2, padding=1x1
        int H_out = (H + 2 * 1 - 3) / 2 + 1;
        int W_out = (W + 2 * 1 - 3) / 2 + 1;

        std::vector<float> out(C_out * H_out * W_out, 0.0f);

        for (int oc = 0; oc < C_out; oc++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    float sum = b_conv[oc];
                    for (int ic = 0; ic < C_in; ic++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = oh * 2 - 1 + kh;
                                int iw = ow * 2 - 1 + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    sum += feat[(ic * H + ih) * W + iw] * w_conv[((oc * C_in + ic) * 3 + kh) * 3 + kw];
                                }
                            }
                        }
                    }
                    // ReLU
                    out[(oc * H_out + oh) * W_out + ow] = sum > 0 ? sum : 0;
                }
            }
        }

        feat = out;
        H = H_out;
        W = W_out;
        C_in = C_out;

        // Per-layer dump for diff debugging
        {
            char label[64];
            snprintf(label, sizeof(label), "ref_enc_conv%d", li);
            dump_stage(ctx, label, feat.data(), feat.size());
            if (ctx->verbosity >= 2) {
                float mn = *std::min_element(feat.begin(), feat.end());
                float mx = *std::max_element(feat.begin(), feat.end());
                double sm = 0;
                for (auto v : feat)
                    sm += v;
                fprintf(stderr, "openvoice2: %s (%d×%d×%d) mean=%.6f min=%.4f max=%.4f\n", label, C_in, H, W,
                        sm / feat.size(), mn, mx);
            }
        }
    }

    // Reshape: (C_out=128, H, W) → (H, C_out * W) for GRU
    // GRU processes along the H (frequency) axis, with input_size = C_out * W
    int gru_input_size = C_in * W; // 128 * W
    int gru_hidden = 128;

    // Read GRU weights
    std::vector<float> w_ih, w_hh, b_ih, b_hh;
    read_f32(re.gru_w_ih, w_ih);
    read_f32(re.gru_w_hh, w_hh);
    read_f32(re.gru_b_ih, b_ih);
    read_f32(re.gru_b_hh, b_hh);

    // GRU forward: process H time steps (frequency axis)
    std::vector<float> h_state(gru_hidden, 0.0f);
    for (int t = 0; t < H; t++) {
        // Build input vector: flatten (C_in, W) at frequency step t
        std::vector<float> x_t(gru_input_size);
        for (int c = 0; c < C_in; c++)
            for (int w_idx = 0; w_idx < W; w_idx++)
                x_t[c * W + w_idx] = feat[(c * H + t) * W + w_idx];

        // GRU gates: r, z, n
        // gates_ih = W_ih @ x_t + b_ih   (3*hidden)
        // gates_hh = W_hh @ h + b_hh     (3*hidden)
        std::vector<float> g_ih(3 * gru_hidden, 0.0f);
        std::vector<float> g_hh(3 * gru_hidden, 0.0f);

        for (int o = 0; o < 3 * gru_hidden; o++) {
            float s1 = b_ih[o], s2 = b_hh[o];
            for (int i = 0; i < gru_input_size; i++)
                s1 += x_t[i] * w_ih[o * gru_input_size + i];
            for (int i = 0; i < gru_hidden; i++)
                s2 += h_state[i] * w_hh[o * gru_hidden + i];
            g_ih[o] = s1;
            g_hh[o] = s2;
        }

        // r = sigmoid(g_ih[0:H] + g_hh[0:H])
        // z = sigmoid(g_ih[H:2H] + g_hh[H:2H])
        // n = tanh(g_ih[2H:3H] + r * g_hh[2H:3H])
        // h = (1-z)*n + z*h_prev
        for (int i = 0; i < gru_hidden; i++) {
            float r = 1.0f / (1.0f + expf(-(g_ih[i] + g_hh[i])));
            float z = 1.0f / (1.0f + expf(-(g_ih[gru_hidden + i] + g_hh[gru_hidden + i])));
            float n = tanhf(g_ih[2 * gru_hidden + i] + r * g_hh[2 * gru_hidden + i]);
            h_state[i] = (1.0f - z) * n + z * h_state[i];
        }
    }

    dump_stage(ctx, "ref_enc_gru_out", h_state.data(), h_state.size());

    // Linear projection: (128) → (256)
    std::vector<float> w_proj, b_proj;
    read_f32(re.proj_w, w_proj);
    read_f32(re.proj_b, b_proj);

    out_emb.resize(hp.gin_channels);
    for (int o = 0; o < hp.gin_channels; o++) {
        float sum = b_proj[o];
        for (int i = 0; i < gru_hidden; i++)
            sum += h_state[i] * w_proj[o * gru_hidden + i];
        out_emb[o] = sum;
    }

    return true;
}

// ── Posterior encoder (enc_q) ────────────────────────────────────────
// spec → WaveNet → mean + logvar → sample z

static void enc_q_forward(openvoice2_context* ctx, const std::vector<float>& spec, int T,
                          const std::vector<float>& g_vec, std::vector<float>& z_out) {
    const auto& hp = ctx->hp;
    const auto& eq = ctx->w.enc_q;
    int hidden = hp.hidden_channels;
    int inter = hp.inter_channels;
    int spec_ch = hp.spec_channels;

    // Pre conv: (spec_channels, T) → (hidden, T), k=1
    std::vector<float> w_pre, b_pre;
    read_f32(eq.pre_w, w_pre);
    read_f32(eq.pre_b, b_pre);

    std::vector<float> h(hidden * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int oc = 0; oc < hidden; oc++) {
            float sum = b_pre[oc];
            for (int ic = 0; ic < spec_ch; ic++)
                sum += spec[t * spec_ch + ic] * w_pre[ic + oc * spec_ch];
            h[t * hidden + oc] = sum;
        }
    }

    // Compute speaker conditioning for WaveNet
    std::vector<float> g_cond;
    if (hp.zero_g) {
        // zero_g mode: enc_q ignores speaker conditioning
        g_cond.clear();
    } else {
        compute_g_cond(eq.wn, g_vec, hp.gin_channels, hp.n_wn_layers_enc_q, hidden, g_cond);
    }

    // WaveNet
    std::vector<float> wn_out;
    wavenet_forward(eq.wn, hp.n_wn_layers_enc_q, hidden, T, h, g_cond, wn_out);

    // Proj: (hidden, T) → (2*inter, T), k=1
    std::vector<float> w_proj, b_proj;
    read_f32(eq.proj_w, w_proj);
    read_f32(eq.proj_b, b_proj);

    int proj_out = 2 * inter;
    std::vector<float> stats(proj_out * T, 0.0f);
    for (int t = 0; t < T; t++) {
        for (int oc = 0; oc < proj_out; oc++) {
            float sum = b_proj[oc];
            for (int ic = 0; ic < hidden; ic++)
                sum += wn_out[t * hidden + ic] * w_proj[ic + oc * hidden];
            stats[t * proj_out + oc] = sum;
        }
    }

    // Split into mean + logvar, sample z = mean + exp(logvar) * noise * tau
    z_out.resize(inter * T);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < inter; c++) {
            float mean = stats[t * proj_out + c];
            float logvar = stats[t * proj_out + inter + c];
            float noise = dist(ctx->rng);
            z_out[t * inter + c] = mean + expf(logvar) * noise * ctx->tau;
        }
    }
}

// ── WaveNet-based coupling flow ──────────────────────────────────────
// Forward: z → z_p (to prior space)
// Reverse: z_p → z (from prior space)

static void flow_wavenet(openvoice2_context* ctx, std::vector<float>& z, int T, const std::vector<float>& g_vec,
                         bool reverse) {
    const auto& hp = ctx->hp;
    int C = hp.inter_channels;
    int half = C / 2;
    int hidden = hp.hidden_channels;
    int n_blocks = hp.n_flow_blocks;

    int start = reverse ? (n_blocks - 1) : 0;
    int end = reverse ? -1 : n_blocks;
    int step = reverse ? -1 : 1;

    for (int fi = start; fi != end; fi += step) {
        const auto& fb = ctx->w.flow_blocks[fi];

        // Flip channels
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C / 2; c++)
                std::swap(z[t * C + c], z[t * C + C - 1 - c]);

        // Split into z0, z1
        std::vector<float> z0(half * T), z1(half * T);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z0[t * half + c] = z[t * C + c];
                z1[t * half + c] = z[t * C + half + c];
            }
        }

        // Pre conv: (half → hidden, k=1)
        std::vector<float> w_pre, b_pre;
        read_f32(fb.pre_w, w_pre);
        read_f32(fb.pre_b, b_pre);

        std::vector<float> h(hidden * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < hidden; oc++) {
                float sum = b_pre[oc];
                for (int ic = 0; ic < half; ic++)
                    sum += z0[t * half + ic] * w_pre[ic + oc * half];
                h[t * hidden + oc] = sum;
            }
        }

        // Speaker conditioning for this flow block's WaveNet
        std::vector<float> g_cond;
        compute_g_cond(fb.wn, g_vec, hp.gin_channels, hp.n_wn_layers_flow, hidden, g_cond);

        // WaveNet
        std::vector<float> wn_out;
        wavenet_forward(fb.wn, hp.n_wn_layers_flow, hidden, T, h, g_cond, wn_out);

        // Post conv: (hidden → half, k=1)
        std::vector<float> w_post, b_post;
        read_f32(fb.post_w, w_post);
        read_f32(fb.post_b, b_post);

        std::vector<float> m(half * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int oc = 0; oc < half; oc++) {
                float sum = b_post[oc];
                for (int ic = 0; ic < hidden; ic++)
                    sum += wn_out[t * hidden + ic] * w_post[ic + oc * hidden];
                m[t * half + oc] = sum;
            }
        }

        // Affine transform (mean_only)
        if (reverse) {
            // Inverse: z1 = z1 - m
            for (int t = 0; t < T; t++)
                for (int c = 0; c < half; c++)
                    z1[t * half + c] -= m[t * half + c];
        } else {
            // Forward: z1 = z1 + m
            for (int t = 0; t < T; t++)
                for (int c = 0; c < half; c++)
                    z1[t * half + c] += m[t * half + c];
        }

        // Recombine
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < half; c++) {
                z[t * C + c] = z0[t * half + c];
                z[t * C + half + c] = z1[t * half + c];
            }
        }
    }
}

// ── Resample helper ──────────────────────────────────────────────────

static void resample_linear(const float* in, int n_in, int sr_in, int sr_out, std::vector<float>& out) {
    if (sr_in == sr_out) {
        out.assign(in, in + n_in);
        return;
    }
    int n_out = (int)((int64_t)n_in * sr_out / sr_in);
    out.resize(n_out);
    for (int i = 0; i < n_out; i++) {
        float pos = (float)i * sr_in / sr_out;
        int idx = (int)pos;
        float frac = pos - idx;
        float s0 = (idx < n_in) ? in[idx] : 0.0f;
        float s1 = (idx + 1 < n_in) ? in[idx + 1] : s0;
        out[i] = s0 + frac * (s1 - s0);
    }
}

// ── HiFi-GAN decode (ggml graph, same pattern as melotts.cpp) ────────

static bool hifigan_decode_cpu(openvoice2_context* ctx, const std::vector<float>& z, const std::vector<float>& g_vec,
                               int T_latent, std::vector<float>& pcm_out) {
    const auto& hp = ctx->hp;
    const auto& dec = ctx->w.dec;
    int C_in = hp.inter_channels;
    int gin = hp.gin_channels;

    // Allocate ggml context for the graph
    ggml_init_params gp = {64 * 1024 * 1024, nullptr, true};
    ggml_context* gc = ggml_init(gp);
    if (!gc)
        return false;

    // Input tensor
    ggml_tensor* x_input = ggml_new_tensor_2d(gc, GGML_TYPE_F32, C_in, T_latent);
    ggml_set_name(x_input, "dec_input");
    ggml_set_input(x_input);

    // Speaker conditioning (zero for zero_g mode)
    ggml_tensor* g_input = nullptr;
    if (dec.cond_w && !g_vec.empty()) {
        g_input = ggml_new_tensor_1d(gc, GGML_TYPE_F32, gin);
        ggml_set_name(g_input, "dec_g");
        ggml_set_input(g_input);
    }

    // conv_pre: (inter_channels, T) → (upsample_initial_ch=512, T)
    ggml_tensor* x = conv1d_cf(gc, x_input, dec.conv_pre_w, dec.conv_pre_b, 1, 3, 1);

    // Speaker conditioning: x += cond(g)
    if (g_input && dec.cond_w) {
        ggml_tensor* g_2d = ggml_reshape_2d(gc, g_input, gin, 1);
        ggml_tensor* g_proj = conv1d_cf(gc, g_2d, dec.cond_w, dec.cond_b);
        x = ggml_add(gc, x, g_proj);
    }

    // Upsample stages
    int n_rk = (int)hp.resblock_kernel_sizes.size();
    int rb_idx = 0;

    for (int us = 0; us < hp.n_upsample_stages; us++) {
        x = ggml_leaky_relu(gc, x, 0.1f, false);

        int stride = hp.upsample_rates[us];
        int kernel = hp.upsample_kernel_sizes[us];
        int crop_each = (kernel - stride) / 2;

        if (dec.ups[us].w_perm) {
            x = core_convt::convt1d_decomp(gc, x, dec.ups[us].w_perm, dec.ups[us].b, stride, kernel, crop_each,
                                           crop_each);
        } else {
            x = core_convt::convt1d_crop(gc, x, dec.ups[us].w, dec.ups[us].b, stride, crop_each, crop_each);
        }

        // MRF: average of resblocks
        ggml_tensor* sum_rb = nullptr;

        for (int ri = 0; ri < n_rk; ri++) {
            const auto& rb = dec.resblocks[rb_idx + ri];
            int rk = hp.resblock_kernel_sizes[ri];

            ggml_tensor* y = x;
            // 3 dilated conv pairs per ResBlock1
            for (int di = 0; di < 3; di++) {
                int d = hp.resblock_dilation_sizes[ri * 3 + di];
                int p = (rk * d - d) / 2;

                ggml_tensor* yt = ggml_leaky_relu(gc, y, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.convs1[di], rb.convs1_b[di], 1, p, d);
                yt = ggml_leaky_relu(gc, yt, 0.1f, false);
                yt = conv1d_cf(gc, yt, rb.convs2[di], rb.convs2_b[di], 1, (rk - 1) / 2, 1);
                y = ggml_add(gc, y, yt);
            }

            if (sum_rb == nullptr)
                sum_rb = y;
            else
                sum_rb = ggml_add(gc, sum_rb, y);
        }

        x = ggml_scale(gc, sum_rb, 1.0f / (float)n_rk);
        rb_idx += n_rk;
    }

    // Final: LeakyReLU → conv_post → tanh
    x = ggml_leaky_relu(gc, x, 0.1f, false);
    x = conv1d_cf(gc, x, dec.conv_post_w, dec.conv_post_b, 1, 3, 1);
    x = ggml_tanh(gc, x);

    // Build and compute graph
    ggml_cgraph* gf = ggml_new_graph_custom(gc, 32768, false);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "openvoice2: HiFi-GAN graph alloc failed\n");
        ggml_free(gc);
        return false;
    }

    ggml_backend_tensor_set(x_input, z.data(), 0, z.size() * sizeof(float));
    if (g_input && !g_vec.empty())
        ggml_backend_tensor_set(g_input, g_vec.data(), 0, gin * sizeof(float));

    ggml_backend_sched_graph_compute(ctx->sched, gf);

    int T_audio = (int)ggml_nelements(x);
    pcm_out.resize(T_audio);
    ggml_backend_tensor_get(x, pcm_out.data(), 0, T_audio * sizeof(float));

    // Debug: PCM stats
    {
        float mn = *std::min_element(pcm_out.begin(), pcm_out.end());
        float mx = *std::max_element(pcm_out.begin(), pcm_out.end());
        double sum = 0;
        for (auto v : pcm_out)
            sum += v;
        fprintf(stderr, "openvoice2: HiFi-GAN output %d samples, min=%.6f max=%.6f mean=%.6f\n", T_audio, mn, mx,
                (float)(sum / T_audio));
    }

    ggml_free(gc);
    return true;
}

// ── Main voice conversion API ────────────────────────────────────────

extern "C" bool openvoice2_convert(struct openvoice2_context* ctx, const float* src_pcm, int n_src, int src_sr,
                                   const float* ref_pcm, int n_ref, int ref_sr, float** out_pcm, int* n_out) {
    if (!ctx || !src_pcm || !ref_pcm || !out_pcm || !n_out)
        return false;

    const auto& hp = ctx->hp;
    int target_sr = hp.sample_rate; // 22050

    // Resample to target sample rate
    std::vector<float> src_22k, ref_22k;
    resample_linear(src_pcm, n_src, src_sr, target_sr, src_22k);
    resample_linear(ref_pcm, n_ref, ref_sr, target_sr, ref_22k);

    if (ctx->verbosity >= 1)
        fprintf(stderr, "openvoice2: src %d→%d samples, ref %d→%d samples\n", n_src, (int)src_22k.size(), n_ref,
                (int)ref_22k.size());

    // 1. STFT of source and reference
    int T_src, T_ref;
    std::vector<float> src_spec, ref_spec;
    stft_magnitude(src_22k.data(), (int)src_22k.size(), hp.filter_length, hp.hop_length, hp.win_length, src_spec,
                   T_src);
    stft_magnitude(ref_22k.data(), (int)ref_22k.size(), hp.filter_length, hp.hop_length, hp.win_length, ref_spec,
                   T_ref);

    if (ctx->verbosity >= 1)
        fprintf(stderr, "openvoice2: STFT — src T=%d, ref T=%d (%d bins)\n", T_src, T_ref, hp.spec_channels);

    // 2. Extract target speaker embedding from reference
    std::vector<float> target_se;
    if (!ref_enc_forward(ctx, ref_spec, T_ref, target_se)) {
        fprintf(stderr, "openvoice2: ref_enc failed\n");
        return false;
    }

    if (ctx->verbosity >= 1) {
        float se_min = *std::min_element(target_se.begin(), target_se.end());
        float se_max = *std::max_element(target_se.begin(), target_se.end());
        float se_mean = 0;
        for (auto v : target_se)
            se_mean += v;
        se_mean /= target_se.size();
        fprintf(stderr, "openvoice2: target_se (256-d) min=%.4f max=%.4f mean=%.6f\n", se_min, se_max, se_mean);
    }

    // Helper for vector stats
    auto vec_stats = [](const char* label, const std::vector<float>& v) {
        float mn = *std::min_element(v.begin(), v.end());
        float mx = *std::max_element(v.begin(), v.end());
        double sum = 0, sum2 = 0;
        for (auto x : v) {
            sum += x;
            sum2 += x * (double)x;
        }
        float mean = (float)(sum / v.size());
        float std_dev = (float)std::sqrt(sum2 / v.size() - mean * (double)mean);
        fprintf(stderr, "openvoice2: %s n=%zu min=%.4f max=%.4f mean=%.6f std=%.6f\n", label, v.size(), mn, mx, mean,
                std_dev);
    };

    // 3. Posterior encoder: src_spec → z (with g=0 for zero_g)
    std::vector<float> g_zero(hp.gin_channels, 0.0f);
    std::vector<float> z;
    enc_q_forward(ctx, src_spec, T_src, g_zero, z);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "openvoice2: enc_q → z (%d × %d)\n", hp.inter_channels, T_src);
        vec_stats("enc_q_z", z);
    }

    // 4. Source speaker embedding — use pre-saved base speaker if available
    //    (upstream OpenVoice2 uses base_speakers/ses/en-default.pth, not ref_enc).
    std::vector<float> src_se;
    if (!ctx->w.base_speakers.empty()) {
        // Default to first base speaker (en-au alphabetically, or en-default if present)
        const ov2_base_speaker* best = &ctx->w.base_speakers[0];
        for (const auto& bs : ctx->w.base_speakers) {
            if (bs.name == "en-default") {
                best = &bs;
                break;
            }
        }
        read_f32(best->embedding, src_se);
        if (ctx->verbosity >= 1)
            fprintf(stderr, "openvoice2: using base speaker '%s' as source SE\n", best->name.c_str());
    } else {
        // Fallback: extract from source audio (less accurate for synthetic input)
        ref_enc_forward(ctx, src_spec, T_src, src_se);
        if (ctx->verbosity >= 1)
            fprintf(stderr, "openvoice2: extracted source SE from audio (no base speakers in GGUF)\n");
    }

    // 5. Flow forward: z → z_p (normalize with source voice)
    flow_wavenet(ctx, z, T_src, src_se, /*reverse=*/false);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "openvoice2: flow forward (source → prior)\n");
        vec_stats("z_after_flow_fwd", z);
    }

    // 6. Flow reverse: z_p → z_hat (denormalize with target voice)
    flow_wavenet(ctx, z, T_src, target_se, /*reverse=*/true);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "openvoice2: flow reverse (prior → target)\n");
        vec_stats("z_after_flow_rev", z);
    }

    // 7. HiFi-GAN decode: z_hat → audio (with g=0 for zero_g)
    std::vector<float> pcm;
    if (!hifigan_decode_cpu(ctx, z, g_zero, T_src, pcm)) {
        fprintf(stderr, "openvoice2: hifigan decode failed\n");
        return false;
    }

    // Peak-normalize output to fill ±0.95 range.
    // The voice converter produces weak output (±0.3) on synthetic MeloTTS input;
    // normalization makes it audible without amplifying noise beyond tanh clipping.
    {
        const char* no_norm = std::getenv("OV2_NO_NORMALIZE");
        if (!no_norm || std::strcmp(no_norm, "1") != 0) {
            float peak = 0.0f;
            for (auto v : pcm) {
                float a = v < 0 ? -v : v;
                if (a > peak)
                    peak = a;
            }
            if (peak > 0.01f) {
                float gain = 0.95f / peak;
                for (auto& v : pcm)
                    v *= gain;
                if (ctx->verbosity >= 1)
                    fprintf(stderr, "openvoice2: peak-normalized (gain=%.2f, peak was %.4f)\n", gain, peak);
            }
        }
    }

    *n_out = (int)pcm.size();
    *out_pcm = (float*)malloc(pcm.size() * sizeof(float));
    memcpy(*out_pcm, pcm.data(), pcm.size() * sizeof(float));

    if (ctx->verbosity >= 1)
        fprintf(stderr, "openvoice2: output %d samples @ %d Hz (%.1f s)\n", *n_out, hp.sample_rate,
                (float)*n_out / hp.sample_rate);

    return true;
}

extern "C" bool openvoice2_extract_speaker_embedding(struct openvoice2_context* ctx, const float* ref_pcm, int n_ref,
                                                     int ref_sr, float* out_embedding) {
    if (!ctx || !ref_pcm || !out_embedding)
        return false;

    std::vector<float> ref_22k;
    resample_linear(ref_pcm, n_ref, ref_sr, ctx->hp.sample_rate, ref_22k);

    int T_ref;
    std::vector<float> ref_spec;
    stft_magnitude(ref_22k.data(), (int)ref_22k.size(), ctx->hp.filter_length, ctx->hp.hop_length, ctx->hp.win_length,
                   ref_spec, T_ref);

    std::vector<float> emb;
    if (!ref_enc_forward(ctx, ref_spec, T_ref, emb))
        return false;

    memcpy(out_embedding, emb.data(), 256 * sizeof(float));
    return true;
}

extern "C" void openvoice2_free(struct openvoice2_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->buf_perm)
        ggml_backend_buffer_free(ctx->buf_perm);
    if (ctx->ctx_perm)
        ggml_free(ctx->ctx_perm);
    if (ctx->w_buf)
        ggml_backend_buffer_free(ctx->w_buf);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}

extern "C" void openvoice2_set_dump_dir(struct openvoice2_context* ctx, const char* dir) {
    if (ctx)
        ctx->dump_dir = dir ? dir : "";
}
