// src/htdemucs.cpp — HTDemucs source separation runtime.
//
// Phase 1: GGUF loader + STFT/iSTFT + weight loading.
// Phase 2: encoder + decoder U-Net.
// Phase 3: CrossTransformer.
// Phase 4: Full forward + overlap-add chunking.

#include "htdemucs.h"
#include "htdemucs_ggml_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"
#include "core/fft.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend

#if defined(HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h> // cblas_sgemm — CrossTransformer is 86% of runtime
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include "core/ggml_cpu_backend.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Debug gating
// ---------------------------------------------------------------------------
static bool htdemucs_debug() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_DEBUG");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

// ---------------------------------------------------------------------------
// Phase profiler — CRISPASR_HTDEMUCS_PROFILE=1 prints a per-phase wall-time
// breakdown of one forward pass. Zero cost when off (the accumulate is guarded
// and the clock read only happens inside the guard).
// ---------------------------------------------------------------------------
static bool htdemucs_profile() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_PROFILE");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

static double htd_now_ms();
static bool htdemucs_use_ggml();
static bool htdemucs_use_fused();
static bool htdemucs_fused_ggml(struct htdemucs_context* ctx, std::vector<float>& x_buf, int& x_C, int& x_Fq, int x_T,
                                std::vector<float>& xt_buf, int& xt_C, int xt_T,
                                const std::vector<float>& freq_emb_bcast);
static bool htdemucs_dec_freq_ggml(struct htdemucs_context* ctx, const struct htdemucs_dec_layer& dec,
                                   std::vector<float>& x_buf, const std::vector<float>& skip_buf, int& x_C, int& x_Fq,
                                   int x_T, int stride, int pad, bool is_last, std::vector<float>* pre_out, int* pre_C,
                                   int* pre_Fq);
static bool htdemucs_enc_freq_ggml(struct htdemucs_context* ctx, const struct htdemucs_enc_layer& enc,
                                   std::vector<float>& x_buf, int& x_C, int& x_Fq, int x_T, int stride, int pad,
                                   const std::vector<float>* inject, int inject_C, int idx);
static bool htdemucs_transformer_ggml(struct htdemucs_context* ctx, std::vector<float>& x_buf, int x_seq,
                                      std::vector<float>& xt_buf, int xt_seq, int dim, int n_heads, int n_layers,
                                      int classic_parity);

namespace {
struct htd_prof {
    std::map<std::string, double> ms;
    std::vector<std::string> order;
    void add(const char* k, double v) {
        auto it = ms.find(k);
        if (it == ms.end()) {
            ms[k] = v;
            order.push_back(k);
        } else {
            it->second += v;
        }
    }
    double t_start = 0.0;
    void report() const {
        double tot = 0;
        for (auto& kv : ms)
            tot += kv.second;
        const double wall = t_start > 0.0 ? htd_now_ms() - t_start : 0.0;
        fprintf(stderr, "\n=== htdemucs phase profile (wall %.2f s; [nested] rows double-count) ===\n", wall / 1000.0);
        for (const auto& k : order) {
            const double v = ms.at(k);
            fprintf(stderr, "  %-22s %8.1f ms  %5.1f%%\n", k.c_str(), v, wall > 0 ? 100.0 * v / wall : 0.0);
        }
        double nested = 0;
        for (const auto& k : order)
            if (k.find("[nested") != std::string::npos)
                nested += ms.at(k);
        const double covered = tot - nested;
        if (wall > covered)
            fprintf(stderr, "  %-22s %8.1f ms  %5.1f%%  (uninstrumented)\n", "[other]", wall - covered,
                    100.0 * (wall - covered) / wall);
    }
};
} // namespace

static double htd_now_ms() {
    return (double)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
               .count() /
           1000.0;
}

namespace {
// RAII phase timer. Constructed with a null prof when profiling is off.
struct htd_scope {
    htd_prof* p;
    const char* n;
    double t0;
    htd_scope(htd_prof* prof, const char* name) : p(prof), n(name), t0(p ? htd_now_ms() : 0.0) {}
    ~htd_scope() {
        if (p)
            p->add(n, htd_now_ms() - t0);
    }
};
} // namespace

#define HTD_PROF(prof, name) htd_scope _htd_scope_guard(htdemucs_profile() ? &(prof) : nullptr, name)

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------
struct htdemucs_hparams {
    int audio_channels = 2;
    int channels = 48;
    int nfft = 4096;
    int depth = 4;
    int bottom_channels = 512;
    int samplerate = 44100;
    float segment = 7.8f;
    bool cac = true;
    int kernel_size = 8;
    int stride = 4;
    int context = 1;
    int n_sources = 4;
    int dconv_depth = 2;
    int dconv_compress = 8;
    bool has_rewrite = true;
    bool has_freq_emb = true;
    float freq_emb_scale = 0.2f;

    // Transformer
    int t_layers = 5;
    int t_heads = 8;
    float t_max_period = 10000.0f;
    float t_weight_pos_embed = 1.0f;
    int t_classic_parity = 0; // 0 = cross-first-is-odd, 1 = cross-first-is-even

    int hop_length() const { return nfft / 4; }
    int training_length() const { return (int)(segment * samplerate); }
};

// ---------------------------------------------------------------------------
// Per-layer weight structs
// ---------------------------------------------------------------------------

// DConv: 2 residual sub-layers, each: Conv1d→GroupNorm→GELU→Conv1d→GroupNorm→GLU→LayerScale
struct htdemucs_dconv_sublayer {
    ggml_tensor* conv1_w = nullptr; // dilated conv weight
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* norm1_w = nullptr; // GroupNorm(1)
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* conv2_w = nullptr; // 1x1 → 2*channels
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* norm2_w = nullptr; // GroupNorm(1)
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* scale = nullptr; // LayerScale
};

struct htdemucs_dconv {
    std::vector<htdemucs_dconv_sublayer> layers; // dconv_depth layers
};

// Encoder layer (freq branch = Conv2d, time branch = Conv1d)
struct htdemucs_enc_layer {
    ggml_tensor* conv_w = nullptr; // main conv
    ggml_tensor* conv_b = nullptr;
    ggml_tensor* norm1_w = nullptr; // GroupNorm after conv (if not empty)
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* rewrite_w = nullptr; // 1x1 rewrite → 2*ch (GLU)
    ggml_tensor* rewrite_b = nullptr;
    ggml_tensor* norm2_w = nullptr; // GroupNorm after rewrite
    ggml_tensor* norm2_b = nullptr;
    htdemucs_dconv dconv;
    bool empty = false; // last freq layer before merge
    bool freq = true;   // true = freq (Conv2d), false = time (Conv1d)
    bool has_norm = false;
    int kernel_size = 8;
    int stride_val = 4;
    int pad = 2;
};

// Decoder layer
struct htdemucs_dec_layer {
    ggml_tensor* conv_tr_w = nullptr; // ConvTranspose
    ggml_tensor* conv_tr_b = nullptr;
    ggml_tensor* norm2_w = nullptr; // GroupNorm after ConvTranspose
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* rewrite_w = nullptr;
    ggml_tensor* rewrite_b = nullptr;
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    htdemucs_dconv dconv;
    bool empty = false;
    bool freq = true;
    bool last = false;
    int kernel_size = 8;
    int stride_val = 4;
    int pad = 2;
};

// CrossTransformer self-attention layer
struct htdemucs_self_attn_layer {
    ggml_tensor* in_proj_w = nullptr; // (3*d, d)
    ggml_tensor* in_proj_b = nullptr;
    ggml_tensor* out_proj_w = nullptr; // (d, d)
    ggml_tensor* out_proj_b = nullptr;
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* linear1_w = nullptr; // FFN
    ggml_tensor* linear1_b = nullptr;
    ggml_tensor* linear2_w = nullptr;
    ggml_tensor* linear2_b = nullptr;
    ggml_tensor* gamma1_scale = nullptr; // LayerScale
    ggml_tensor* gamma2_scale = nullptr;
    ggml_tensor* norm_out_w = nullptr; // optional norm_out (GroupNorm(1))
    ggml_tensor* norm_out_b = nullptr;
};

// CrossTransformer cross-attention layer
struct htdemucs_cross_attn_layer {
    ggml_tensor* cross_attn_in_proj_w = nullptr;
    ggml_tensor* cross_attn_in_proj_b = nullptr;
    ggml_tensor* cross_attn_out_proj_w = nullptr;
    ggml_tensor* cross_attn_out_proj_b = nullptr;
    ggml_tensor* norm1_w = nullptr;
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
    ggml_tensor* norm3_w = nullptr;
    ggml_tensor* norm3_b = nullptr;
    ggml_tensor* linear1_w = nullptr;
    ggml_tensor* linear1_b = nullptr;
    ggml_tensor* linear2_w = nullptr;
    ggml_tensor* linear2_b = nullptr;
    ggml_tensor* gamma1_scale = nullptr;
    ggml_tensor* gamma2_scale = nullptr;
    ggml_tensor* norm_out_w = nullptr;
    ggml_tensor* norm_out_b = nullptr;
};

struct htdemucs_transformer_layer {
    bool is_cross = false; // false = self-attn, true = cross-attn
    htdemucs_self_attn_layer self_attn;
    htdemucs_cross_attn_layer cross_attn;
};

// Full model weights
struct htdemucs_model {
    htdemucs_hparams hparams;

    // Encoder: depth freq layers + (depth-1) time layers
    std::vector<htdemucs_enc_layer> encoder;  // freq branch
    std::vector<htdemucs_enc_layer> tencoder; // time branch

    // Decoder: depth freq layers + (depth-1) time layers
    std::vector<htdemucs_dec_layer> decoder;
    std::vector<htdemucs_dec_layer> tdecoder;

    // Frequency embedding
    ggml_tensor* freq_emb_w = nullptr; // (n_freqs, channels)

    // Channel up/downsamplers around transformer
    ggml_tensor* channel_up_w = nullptr; // (bottom_ch, transformer_ch, 1)
    ggml_tensor* channel_up_b = nullptr;
    ggml_tensor* channel_down_w = nullptr;
    ggml_tensor* channel_down_b = nullptr;
    ggml_tensor* channel_up_t_w = nullptr;
    ggml_tensor* channel_up_t_b = nullptr;
    ggml_tensor* channel_down_t_w = nullptr;
    ggml_tensor* channel_down_t_b = nullptr;

    // CrossTransformer
    ggml_tensor* norm_in_w = nullptr; // LayerNorm for spec input
    ggml_tensor* norm_in_b = nullptr;
    ggml_tensor* norm_in_t_w = nullptr; // LayerNorm for time input
    ggml_tensor* norm_in_t_b = nullptr;
    std::vector<htdemucs_transformer_layer> ct_layers;   // spec layers
    std::vector<htdemucs_transformer_layer> ct_layers_t; // time layers

    // Source names
    std::vector<std::string> source_names;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct htdemucs_context {
    htdemucs_model model;
    htdemucs_params params;

    // ggml state
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    ggml_context* ctx_w = nullptr;

    // Precomputed Hann window (nfft)
    std::vector<float> hann_window;

    bool is_gpu = false;

    // Per-stage intermediate capture for the parity diff harness. Off in the
    // normal path (separate() never touches the map), so this costs nothing
    // outside htdemucs_diff().
    bool capture_stages = false;
    std::map<std::string, std::vector<float>> captures;
};

// Record a stage intermediate under `name`. All captures are stored in the
// PyTorch reference layout so the diff is a straight elementwise compare —
// see htdemucs_diff() for the layout contract.
static void htd_capture(htdemucs_context* ctx, const char* name, const float* data, size_t n) {
    if (!ctx || !ctx->capture_stages)
        return;
    ctx->captures[name].assign(data, data + n);
}

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

static bool load_hparams(htdemucs_hparams& hp, gguf_context* meta) {
    hp.audio_channels = core_gguf::kv_u32(meta, "htdemucs.audio_channels", 2);
    hp.channels = core_gguf::kv_u32(meta, "htdemucs.channels", 48);
    hp.nfft = core_gguf::kv_u32(meta, "htdemucs.nfft", 4096);
    hp.depth = core_gguf::kv_u32(meta, "htdemucs.depth", 4);
    hp.bottom_channels = core_gguf::kv_u32(meta, "htdemucs.bottom_channels", 512);
    hp.samplerate = core_gguf::kv_u32(meta, "htdemucs.samplerate", 44100);
    hp.segment = core_gguf::kv_f32(meta, "htdemucs.segment", 7.8f);
    hp.cac = core_gguf::kv_u32(meta, "htdemucs.cac", 1) != 0;
    hp.kernel_size = core_gguf::kv_u32(meta, "htdemucs.kernel_size", 8);
    hp.stride = core_gguf::kv_u32(meta, "htdemucs.stride", 4);
    hp.context = core_gguf::kv_u32(meta, "htdemucs.context", 1);
    hp.n_sources = core_gguf::kv_u32(meta, "htdemucs.n_sources", 4);
    hp.dconv_depth = core_gguf::kv_u32(meta, "htdemucs.dconv_depth", 2);
    hp.dconv_compress = core_gguf::kv_u32(meta, "htdemucs.dconv_compress", 8);
    hp.has_rewrite = core_gguf::kv_u32(meta, "htdemucs.has_rewrite", 1) != 0;
    hp.has_freq_emb = core_gguf::kv_u32(meta, "htdemucs.has_freq_emb", 1) != 0;
    hp.freq_emb_scale = core_gguf::kv_f32(meta, "htdemucs.freq_emb_scale", 0.2f);
    hp.t_layers = core_gguf::kv_u32(meta, "htdemucs.t_layers", 5);
    hp.t_heads = core_gguf::kv_u32(meta, "htdemucs.t_heads", 8);
    hp.t_max_period = core_gguf::kv_f32(meta, "htdemucs.t_max_period", 10000.0f);
    hp.t_weight_pos_embed = core_gguf::kv_f32(meta, "htdemucs.t_weight_pos_embed", 1.0f);
    hp.t_classic_parity = core_gguf::kv_u32(meta, "htdemucs.t_classic_parity", 0);
    return true;
}

// ---------------------------------------------------------------------------
// Weight binding (Phase 2)
// ---------------------------------------------------------------------------

static void bind_dconv(htdemucs_dconv& dc, const core_gguf::tensor_map& t, const std::string& prefix, int depth) {
    dc.layers.resize(depth);
    for (int d = 0; d < depth; d++) {
        auto& sl = dc.layers[d];
        std::string p = prefix + "." + std::to_string(d);
        // Sequential: [0]=dilated_conv, [1]=groupnorm, [2]=GELU, [3]=1x1_conv, [4]=groupnorm, [5]=GLU, [6]=LayerScale
        sl.conv1_w = core_gguf::try_get(t, (p + ".0.weight").c_str());
        sl.conv1_b = core_gguf::try_get(t, (p + ".0.bias").c_str());
        sl.norm1_w = core_gguf::try_get(t, (p + ".1.weight").c_str());
        sl.norm1_b = core_gguf::try_get(t, (p + ".1.bias").c_str());
        sl.conv2_w = core_gguf::try_get(t, (p + ".3.weight").c_str());
        sl.conv2_b = core_gguf::try_get(t, (p + ".3.bias").c_str());
        sl.norm2_w = core_gguf::try_get(t, (p + ".4.weight").c_str());
        sl.norm2_b = core_gguf::try_get(t, (p + ".4.bias").c_str());
        sl.scale = core_gguf::try_get(t, (p + ".6.scale").c_str());
    }
}

static void bind_enc_layer(htdemucs_enc_layer& el, const core_gguf::tensor_map& t, const std::string& prefix) {
    el.conv_w = core_gguf::try_get(t, (prefix + ".conv.weight").c_str());
    el.conv_b = core_gguf::try_get(t, (prefix + ".conv.bias").c_str());
    el.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    el.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    el.rewrite_w = core_gguf::try_get(t, (prefix + ".rewrite.weight").c_str());
    el.rewrite_b = core_gguf::try_get(t, (prefix + ".rewrite.bias").c_str());
    el.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    el.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    el.has_norm = el.norm1_w != nullptr;
    el.empty = (el.norm1_w == nullptr && el.rewrite_w == nullptr);
}

static void bind_dec_layer(htdemucs_dec_layer& dl, const core_gguf::tensor_map& t, const std::string& prefix) {
    dl.conv_tr_w = core_gguf::try_get(t, (prefix + ".conv_tr.weight").c_str());
    dl.conv_tr_b = core_gguf::try_get(t, (prefix + ".conv_tr.bias").c_str());
    dl.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    dl.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    dl.rewrite_w = core_gguf::try_get(t, (prefix + ".rewrite.weight").c_str());
    dl.rewrite_b = core_gguf::try_get(t, (prefix + ".rewrite.bias").c_str());
    dl.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    dl.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    dl.empty = (dl.rewrite_w == nullptr);
}

static void bind_self_attn_layer(htdemucs_self_attn_layer& sa, const core_gguf::tensor_map& t,
                                 const std::string& prefix) {
    sa.in_proj_w = core_gguf::try_get(t, (prefix + ".self_attn.in_proj_weight").c_str());
    sa.in_proj_b = core_gguf::try_get(t, (prefix + ".self_attn.in_proj_bias").c_str());
    sa.out_proj_w = core_gguf::try_get(t, (prefix + ".self_attn.out_proj.weight").c_str());
    sa.out_proj_b = core_gguf::try_get(t, (prefix + ".self_attn.out_proj.bias").c_str());
    sa.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    sa.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    sa.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    sa.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    sa.linear1_w = core_gguf::try_get(t, (prefix + ".linear1.weight").c_str());
    sa.linear1_b = core_gguf::try_get(t, (prefix + ".linear1.bias").c_str());
    sa.linear2_w = core_gguf::try_get(t, (prefix + ".linear2.weight").c_str());
    sa.linear2_b = core_gguf::try_get(t, (prefix + ".linear2.bias").c_str());
    sa.gamma1_scale = core_gguf::try_get(t, (prefix + ".gamma_1.scale").c_str());
    sa.gamma2_scale = core_gguf::try_get(t, (prefix + ".gamma_2.scale").c_str());
    sa.norm_out_w = core_gguf::try_get(t, (prefix + ".norm_out.weight").c_str());
    sa.norm_out_b = core_gguf::try_get(t, (prefix + ".norm_out.bias").c_str());
}

static void bind_cross_attn_layer(htdemucs_cross_attn_layer& ca, const core_gguf::tensor_map& t,
                                  const std::string& prefix) {
    ca.cross_attn_in_proj_w = core_gguf::try_get(t, (prefix + ".cross_attn.in_proj_weight").c_str());
    ca.cross_attn_in_proj_b = core_gguf::try_get(t, (prefix + ".cross_attn.in_proj_bias").c_str());
    ca.cross_attn_out_proj_w = core_gguf::try_get(t, (prefix + ".cross_attn.out_proj.weight").c_str());
    ca.cross_attn_out_proj_b = core_gguf::try_get(t, (prefix + ".cross_attn.out_proj.bias").c_str());
    ca.norm1_w = core_gguf::try_get(t, (prefix + ".norm1.weight").c_str());
    ca.norm1_b = core_gguf::try_get(t, (prefix + ".norm1.bias").c_str());
    ca.norm2_w = core_gguf::try_get(t, (prefix + ".norm2.weight").c_str());
    ca.norm2_b = core_gguf::try_get(t, (prefix + ".norm2.bias").c_str());
    ca.norm3_w = core_gguf::try_get(t, (prefix + ".norm3.weight").c_str());
    ca.norm3_b = core_gguf::try_get(t, (prefix + ".norm3.bias").c_str());
    ca.linear1_w = core_gguf::try_get(t, (prefix + ".linear1.weight").c_str());
    ca.linear1_b = core_gguf::try_get(t, (prefix + ".linear1.bias").c_str());
    ca.linear2_w = core_gguf::try_get(t, (prefix + ".linear2.weight").c_str());
    ca.linear2_b = core_gguf::try_get(t, (prefix + ".linear2.bias").c_str());
    ca.gamma1_scale = core_gguf::try_get(t, (prefix + ".gamma_1.scale").c_str());
    ca.gamma2_scale = core_gguf::try_get(t, (prefix + ".gamma_2.scale").c_str());
    ca.norm_out_w = core_gguf::try_get(t, (prefix + ".norm_out.weight").c_str());
    ca.norm_out_b = core_gguf::try_get(t, (prefix + ".norm_out.bias").c_str());
}

static bool bind_weights(htdemucs_model& m, const core_gguf::tensor_map& t) {
    auto& hp = m.hparams;
    int bound = 0;

    // Encoder (freq branch)
    m.encoder.resize(hp.depth);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "encoder." + std::to_string(i);
        bind_enc_layer(m.encoder[i], t, p);
        if (m.encoder[i].conv_w)
            bound++;
        // DConv
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.encoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        // Determine freq vs time, kernel/stride/pad from weight shape
        m.encoder[i].freq = (m.encoder[i].conv_w && ggml_n_dims(m.encoder[i].conv_w) == 4);
    }

    // Encoder (time branch) — as many layers as freq>1 layers.
    // For depth=4, nfft=4096: freqs = 2048→512→128→32→8→1 but depth=4 so:
    // layer 0: freq=True (2048→512), layer 1: freq=True (512→128),
    // layer 2: freq=True (128→32), layer 3: freq=True (32→8, last_freq=True, ker=32→8)
    // Wait, let me re-check: nfft/2=2048, stride=4:
    // layer 0: freqs=2048, freq=True. 2048>kernel_size(8). freqs=2048/4=512
    // layer 1: freqs=512, freq=True. 512>8. freqs=512/4=128
    // layer 2: freqs=128, freq=True. 128>8. freqs=128/4=32
    // layer 3: freqs=32, freq=True. 32>8. freqs=32/4=8
    // But depth=4, so we stop. The time encoders exist for layers where freq=True,
    // which is all 4 layers. But the LAST freq layer that makes freqs<=kernel_size
    // has empty=True for the time encoder.
    // Let me just count from the GGUF:
    m.tencoder.resize(0);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "tencoder." + std::to_string(i);
        auto* w = core_gguf::try_get(t, (p + ".conv.weight").c_str());
        if (!w)
            break;
        m.tencoder.resize(i + 1);
        bind_enc_layer(m.tencoder[i], t, p);
        m.tencoder[i].freq = false; // time branch always Conv1d
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.tencoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        bound++;
    }

    // Decoder (freq branch)
    m.decoder.resize(hp.depth);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "decoder." + std::to_string(i);
        bind_dec_layer(m.decoder[i], t, p);
        m.decoder[i].freq = (m.decoder[i].conv_tr_w && ggml_n_dims(m.decoder[i].conv_tr_w) == 4);
        m.decoder[i].last = (i == hp.depth - 1);
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.decoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        if (m.decoder[i].conv_tr_w)
            bound++;
    }

    // Decoder (time branch)
    m.tdecoder.resize(0);
    for (int i = 0; i < hp.depth; i++) {
        std::string p = "tdecoder." + std::to_string(i);
        auto* w = core_gguf::try_get(t, (p + ".conv_tr.weight").c_str());
        if (!w)
            break;
        m.tdecoder.resize(i + 1);
        bind_dec_layer(m.tdecoder[i], t, p);
        m.tdecoder[i].freq = false;
        m.tdecoder[i].last = (i == 0); // tdecoder is reversed
        if (core_gguf::try_get(t, (p + ".dconv.layers.0.0.weight").c_str())) {
            bind_dconv(m.tdecoder[i].dconv, t, p + ".dconv.layers", hp.dconv_depth);
        }
        bound++;
    }

    // Frequency embedding
    m.freq_emb_w = core_gguf::try_get(t, "freq_emb.embedding.weight");

    // Channel up/downsamplers
    m.channel_up_w = core_gguf::try_get(t, "channel_upsampler.weight");
    m.channel_up_b = core_gguf::try_get(t, "channel_upsampler.bias");
    m.channel_down_w = core_gguf::try_get(t, "channel_downsampler.weight");
    m.channel_down_b = core_gguf::try_get(t, "channel_downsampler.bias");
    m.channel_up_t_w = core_gguf::try_get(t, "channel_upsampler_t.weight");
    m.channel_up_t_b = core_gguf::try_get(t, "channel_upsampler_t.bias");
    m.channel_down_t_w = core_gguf::try_get(t, "channel_downsampler_t.weight");
    m.channel_down_t_b = core_gguf::try_get(t, "channel_downsampler_t.bias");

    // CrossTransformer norm_in
    m.norm_in_w = core_gguf::try_get(t, "crosstransformer.norm_in.weight");
    m.norm_in_b = core_gguf::try_get(t, "crosstransformer.norm_in.bias");
    m.norm_in_t_w = core_gguf::try_get(t, "crosstransformer.norm_in_t.weight");
    m.norm_in_t_b = core_gguf::try_get(t, "crosstransformer.norm_in_t.bias");

    // CrossTransformer layers (spec + time)
    m.ct_layers.resize(hp.t_layers);
    m.ct_layers_t.resize(hp.t_layers);
    for (int i = 0; i < hp.t_layers; i++) {
        bool is_cross = (i % 2 != hp.t_classic_parity);
        // Spec layers
        m.ct_layers[i].is_cross = is_cross;
        std::string sp = "crosstransformer.layers." + std::to_string(i);
        if (is_cross) {
            bind_cross_attn_layer(m.ct_layers[i].cross_attn, t, sp);
        } else {
            bind_self_attn_layer(m.ct_layers[i].self_attn, t, sp);
        }
        // Time layers
        m.ct_layers_t[i].is_cross = is_cross;
        std::string tp_str = "crosstransformer.layers_t." + std::to_string(i);
        if (is_cross) {
            bind_cross_attn_layer(m.ct_layers_t[i].cross_attn, t, tp_str);
        } else {
            bind_self_attn_layer(m.ct_layers_t[i].self_attn, t, tp_str);
        }
    }

    fprintf(stderr,
            "htdemucs: bound %d enc/dec layers, %d tenc, %d tdec, "
            "%d transformer layers\n",
            bound, (int)m.tencoder.size(), (int)m.tdecoder.size(), hp.t_layers);
    return true;
}

static htdemucs_context* htdemucs_init_impl(const char* model_path, htdemucs_params params) {
    auto ctx = new htdemucs_context();
    ctx->params = params;

    // Pass 1: read metadata / hparams
    gguf_context* meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "htdemucs: failed to open %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    if (!load_hparams(ctx->model.hparams, meta)) {
        fprintf(stderr, "htdemucs: failed to read hparams from %s\n", model_path);
        core_gguf::free_metadata(meta);
        delete ctx;
        return nullptr;
    }
    core_gguf::free_metadata(meta);

    auto& hp = ctx->model.hparams;
    fprintf(stderr,
            "htdemucs: loaded hparams: depth=%d, channels=%d, nfft=%d, "
            "samplerate=%d, segment=%.1f, t_layers=%d\n",
            hp.depth, hp.channels, hp.nfft, hp.samplerate, hp.segment, hp.t_layers);

    // Pass 2: load weights.
    //
    // Backend selection now honours params.use_gpu (it was hardcoded to CPU, so
    // use_gpu and n_threads were dead fields). crispasr_init_gpu_backend() tries
    // the compiled backends in priority order (CUDA > Metal > Vulkan) and falls
    // back to CPU. Gated by CRISPASR_HTDEMUCS_GGML because only the graph path
    // benefits — the CPU/BLAS path wants a CPU backend for its weight reads.
    // GPU only makes sense with the graph path: under the CPU/BLAS path the
    // weights would live on the GPU and every scalar/BLAS kernel would pay a
    // device->host read. CRISPASR_HTDEMUCS_GPU=1 requests it without having to
    // thread a flag through all three surfaces (CLI, session C-ABI, server).
    const char* gpu_env = getenv("CRISPASR_HTDEMUCS_GPU");
    // Prefer the GPU backend whenever the build has one — the fused graph
    // path is dramatically faster on CUDA (measured RTF ≈ 0.3 vs ≈ 7.4 on
    // an RTX 3090 for a 30 s clip; the CPU default left htdemucs on CPU
    // even on CUDA builds). CRISPASR_HTDEMUCS_GPU=0 opts out (forces CPU);
    // =1 forces GPU. crispasr_init_gpu_backend() falls back to CPU on
    // GPU-less hosts, so the default is safe everywhere.
    bool want_gpu = true;
    if (gpu_env && gpu_env[0])
        want_gpu = atoi(gpu_env) != 0;
    if (params.use_gpu)
        want_gpu = true;
    // The ggml graph path is the gate to the GPU backend: want_gpu WITHOUT
    // it still runs on CPU (measured — 0% GPU util on an RTX 3090 with
    // CRISPASR_HTDEMUCS_GPU=1 alone). For GPU setups the graph path only
    // pays off with FUSED (single graph, no host<->device roundtrips; the
    // per-layer graphs measured slower than CPU+Accelerate). So: GPU users
    // set CRISPASR_HTDEMUCS_GGML=1 + CRISPASR_HTDEMUCS_FUSED=1; want_gpu
    // alone stays on the CPU/BLAS path (no silent behavior change for
    // CPU-only hosts, which keep their BLAS path).
    const bool want_graph = htdemucs_use_ggml() || (want_gpu && htdemucs_use_fused());
    if (want_graph) {
        ctx->backend = want_gpu ? crispasr_init_gpu_backend() : core_cpu_backend::init();
        if (!ctx->backend)
            ctx->backend = core_cpu_backend::init();
    } else {
        ctx->backend = core_cpu_backend::init();
    }
    ctx->is_gpu = !core_cpu_backend::is_cpu(ctx->backend);
    fprintf(stderr, "htdemucs: backend = %s\n", ggml_backend_name(ctx->backend));
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "htdemucs", wl)) {
        fprintf(stderr, "htdemucs: failed to load weights from %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    ctx->ctx_w = wl.ctx;
    ctx->buf_w = wl.buf;

    fprintf(stderr, "htdemucs: loaded %zu tensors\n", wl.tensors.size());

    if (!bind_weights(ctx->model, wl.tensors)) {
        fprintf(stderr, "htdemucs: failed to bind weights\n");
        delete ctx;
        return nullptr;
    }

    // Source names
    ctx->model.source_names = {"drums", "bass", "other", "vocals"};

    // Precompute Hann window (periodic, matching torch.hann_window)
    ctx->hann_window.resize(hp.nfft);
    for (int i = 0; i < hp.nfft; i++) {
        ctx->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / hp.nfft));
    }

    return ctx;
}

// ---------------------------------------------------------------------------
// CPU STFT (complex output)
// ---------------------------------------------------------------------------

// Forward STFT with center=True, normalized=True, Hann window.
// Input: pcm[n_channels][n_samples]
// Output: complex spectrogram as separate real/imag arrays.
//   real[n_channels][n_freqs][n_frames], imag[same]
//   where n_freqs = nfft/2 + 1.
struct stft_result {
    int n_channels;
    int n_freqs;
    int n_frames;
    std::vector<float> real; // [n_channels * n_freqs * n_frames]
    std::vector<float> imag;
};

static stft_result compute_stft(const float* pcm_channels, int n_channels, int n_samples, int nfft, int hop,
                                const float* window) {
    stft_result r;
    r.n_channels = n_channels;
    r.n_freqs = nfft / 2 + 1;

    // center=True: pad nfft/2 on each side
    int pad = nfft / 2;
    int padded_len = n_samples + 2 * pad;

    // Reflect padding
    std::vector<float> padded(padded_len);

    r.n_frames = (padded_len - nfft) / hop + 1;
    size_t spec_size = (size_t)n_channels * r.n_freqs * r.n_frames;
    r.real.resize(spec_size, 0.0f);
    r.imag.resize(spec_size, 0.0f);

    // Normalization factor for normalized=True
    float norm_factor = 1.0f / sqrtf((float)nfft);

    // Temp buffers for FFT (need power-of-2 for radix-2)
    // nfft=4096 is already power of 2
    std::vector<float> fft_re(nfft), fft_im(nfft);

    for (int ch = 0; ch < n_channels; ch++) {
        const float* src = pcm_channels + (size_t)ch * n_samples;

        // Reflect pad
        for (int i = 0; i < padded_len; i++) {
            int idx = i - pad;
            if (idx < 0)
                idx = -idx; // reflect left
            if (idx >= n_samples)
                idx = 2 * n_samples - 2 - idx; // reflect right
            idx = std::max(0, std::min(n_samples - 1, idx));
            padded[i] = src[idx];
        }

        for (int f = 0; f < r.n_frames; f++) {
            int start = f * hop;
            // Apply window and copy to FFT buffer
            for (int i = 0; i < nfft; i++) {
                fft_re[i] = padded[start + i] * window[i];
                fft_im[i] = 0.0f;
            }
            // In-place radix-2 FFT
            core_fft::fft_radix2_inplace(fft_re.data(), fft_im.data(), nfft);

            // Store half-spectrum (n_freqs = nfft/2+1), normalized
            size_t base = (size_t)ch * r.n_freqs * r.n_frames + (size_t)f;
            for (int k = 0; k < r.n_freqs; k++) {
                r.real[base + (size_t)k * r.n_frames] = fft_re[k] * norm_factor;
                r.imag[base + (size_t)k * r.n_frames] = fft_im[k] * norm_factor;
            }
        }
    }
    return r;
}

// Inverse STFT
static void compute_istft(const float* real, const float* imag, int n_channels, int n_freqs, int n_frames, int nfft,
                          int hop, const float* window, int output_length, int start_offset, float* out) {
    // torch.stft(normalized=True) DIVIDES the forward transform by sqrt(nfft)
    // (see compute_stft), so the inverse must MULTIPLY by it. Dividing again
    // here scaled every separated stem down by 1/nfft = 1/4096.
    float norm_factor = sqrtf((float)nfft);

    std::vector<float> fft_re(nfft), fft_im(nfft);
    std::vector<float> win_sum;

    for (int ch = 0; ch < n_channels; ch++) {
        int out_len = (n_frames - 1) * hop + nfft;
        std::vector<float> signal(out_len, 0.0f);
        win_sum.assign(out_len, 0.0f);

        size_t ch_base = (size_t)ch * n_freqs * n_frames;

        for (int f = 0; f < n_frames; f++) {
            // Reconstruct full spectrum from half
            for (int k = 0; k < n_freqs; k++) {
                fft_re[k] = real[ch_base + (size_t)k * n_frames + f] * norm_factor;
                fft_im[k] = imag[ch_base + (size_t)k * n_frames + f] * norm_factor;
            }
            // Mirror for negative frequencies
            for (int k = n_freqs; k < nfft; k++) {
                fft_re[k] = fft_re[nfft - k];
                fft_im[k] = -fft_im[nfft - k];
            }

            // Inverse FFT = conjugate + forward FFT + conjugate + scale
            for (int k = 0; k < nfft; k++)
                fft_im[k] = -fft_im[k];
            core_fft::fft_radix2_inplace(fft_re.data(), fft_im.data(), nfft);
            float scale = 1.0f / (float)nfft;
            for (int k = 0; k < nfft; k++) {
                fft_re[k] *= scale;
                fft_im[k] = -fft_im[k] * scale; // unused but correct
            }

            // Overlap-add with window
            int start = f * hop;
            for (int i = 0; i < nfft; i++) {
                signal[start + i] += fft_re[i] * window[i];
                win_sum[start + i] += window[i] * window[i];
            }
        }

        // Normalize by window sum
        for (int i = 0; i < out_len; i++) {
            if (win_sum[i] > 1e-8f)
                signal[i] /= win_sum[i];
        }

        // Copy out starting at the caller-supplied offset. This absorbs both the
        // center=True trim (nfft/2) and _spec()'s reflect pre-padding.
        float* dst = out + (size_t)ch * output_length;
        for (int i = 0; i < output_length; i++) {
            const int j = start_offset + i;
            dst[i] = (j >= 0 && j < out_len) ? signal[j] : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

htdemucs_params htdemucs_default_params(void) {
    htdemucs_params p;
    memset(&p, 0, sizeof(p));
    p.n_threads = 0;
    p.use_gpu = false;
    p.gpu_device = 0;
    return p;
}

htdemucs_context* htdemucs_init_from_file(const char* model_path, htdemucs_params params) {
    return htdemucs_init_impl(model_path, params);
}

void htdemucs_free(htdemucs_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_w)
        core_gguf::release_weight_buffer(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ---------------------------------------------------------------------------
// Tensor read helper — reads any ggml tensor as F32 regardless of storage type
// ---------------------------------------------------------------------------
// Returns false (and leaves `out` zeroed) only for a type ggml itself cannot
// dequantize. Callers MUST treat that as fatal — see the note below.
static bool read_tensor_f32_checked(ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.assign((size_t)n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t)n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++)
            out[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
        return true;
    }
    // Any QUANTIZED type: dequantize through ggml's own type traits rather than
    // enumerating formats here. This is what made htdemucs-q4_k.gguf produce
    // garbage — Q4_K (type 12) fell through to the old zero-fill fallback, so
    // all 40 CrossTransformer attention/FFN weight tensors loaded as SILENT
    // ZEROS and separation output was ~100x amplified noise. A warning is not
    // enough for a missing weight; see the caller, which now aborts the load.
    const ggml_type_traits* tr = ggml_get_type_traits(t->type);
    if (tr && tr->to_float) {
        std::vector<uint8_t> raw((size_t)ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        tr->to_float(raw.data(), out.data(), n);
        return true;
    }
    fprintf(stderr, "htdemucs: ERROR: tensor '%s' has type %d which ggml cannot dequantize\n", t->name, (int)t->type);
    return false;
}

// Back-compat wrapper for the many existing call sites. A tensor that cannot be
// read is a corrupt/incompatible model, so this is loud and fatal rather than
// quietly returning zeros — the previous silent-zero behaviour shipped a broken
// default quantisation for months without anyone noticing.
static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    std::vector<float> out;
    if (!read_tensor_f32_checked(t, out)) {
        fprintf(stderr,
                "htdemucs: FATAL: refusing to run with an unreadable weight tensor ('%s'). "
                "Re-quantize the model or use the F16 build.\n",
                t->name);
        std::abort();
    }
    return out;
}

// ---------------------------------------------------------------------------
// GEMM for the CrossTransformer hot path.
//
// The transformer is ~86% of a forward pass (self-attn 52%, cross-attn 33%,
// measured with CRISPASR_HTDEMUCS_PROFILE=1), and every matmul in it was a
// scalar triple loop whose innermost stride was seq_len floats — ~10 KB, so a
// cache miss per multiply-add. Routing them through cblas_sgemm fixes both the
// FLOP rate and the access pattern.
//
// Gated by CRISPASR_HTDEMUCS_BLAS (default ON where Accelerate is available).
// Set it to 0 to fall back to the scalar path — that is the A/B lever and the
// regression-bisection mechanism; do not remove it.
// ---------------------------------------------------------------------------
static bool htdemucs_use_blas() {
#if defined(HAVE_ACCELERATE)
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_BLAS");
        v = (e && atoi(e) == 0) ? 0 : 1; // default ON
    }
    return v != 0;
#else
    return false;
#endif
}

// C[M,N] = A[M,K] * B[K,N], all row-major. Falls back to a cache-blocked scalar
// loop when Accelerate is unavailable or the gate is off.
static void htd_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(HAVE_ACCELERATE)
    if (htdemucs_use_blas()) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        return;
    }
#endif
    std::fill(C, C + (size_t)M * N, 0.0f);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            const float a = A[(size_t)i * K + k];
            if (a == 0.0f)
                continue;
            const float* brow = B + (size_t)k * N;
            float* crow = C + (size_t)i * N;
            for (int j = 0; j < N; j++)
                crow[j] += a * brow[j];
        }
}

// ---------------------------------------------------------------------------
// Weight cache — read_tensor_f32() copies (and for F16, converts) a whole
// tensor on every call, and the DConv stacks call it from INSIDE their
// per-frequency-band loop: 9 reads x 680 band invocations per encoder layer is
// ~6k redundant dequant+copy passes over the same few weights.
//
// Cache by tensor pointer. Weights are immutable after load, so this is safe
// and the cache lives for the process. Gated CRISPASR_HTDEMUCS_WCACHE
// (default ON) so the old behaviour stays A/B-able.
// ---------------------------------------------------------------------------
// GGML graph port for the CrossTransformer (CRISPASR_HTDEMUCS_GGML).
// Default OFF pending a full A/B: per the dev-guide inverse-default rule a
// verified-but-not-yet-faster path stays opt-in and the old path stays default.
static bool htdemucs_use_ggml() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_GGML");
        v = (e && atoi(e) != 0) ? 1 : 0; // default OFF
    }
    return v != 0;
}

// FUSED: run encoder + transformer + decoder as ONE graph so activations never
// leave the device (CRISPASR_HTDEMUCS_FUSED, default OFF, requires _GGML=1).
// The per-layer graphs pay a host<->device roundtrip per layer, which measured
// SLOWER than CPU+Accelerate for the encoder despite the transformer being
// 3.3-6x faster.
static bool htdemucs_use_fused() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_FUSED");
        v = (e && atoi(e) != 0) ? 1 : 0; // default OFF
    }
    return v != 0;
}

// FASTCONV: batched im2col + gemm for the CPU convs (default ON). Set
// CRISPASR_HTDEMUCS_FASTCONV=0 for the original per-frame scalar path.
static bool htdemucs_fastconv() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_FASTCONV");
        v = (e && atoi(e) == 0) ? 0 : 1; // default ON
    }
    return v != 0;
}

static bool htdemucs_use_wcache() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("CRISPASR_HTDEMUCS_WCACHE");
        v = (e && atoi(e) == 0) ? 0 : 1; // default ON
    }
    return v != 0;
}

// Returns a reference to the cached F32 copy of `t`. The caller must NOT
// mutate it (all current uses are read-only weight reads).
static size_t g_wcache_bytes = 0; // instrumentation only (CRISPASR_HTDEMUCS_MEMSTATS=1)

static const std::vector<float>& cached_tensor_f32(ggml_tensor* t) {
    static std::map<const ggml_tensor*, std::vector<float>> cache;
    if (!htdemucs_use_wcache()) {
        // Gate OFF: re-read every time (the original behaviour). Still stored so
        // the returned reference stays valid for the caller.
        cache[t] = read_tensor_f32(t);
        return cache[t];
    }
    auto it = cache.find(t);
    if (it != cache.end())
        return it->second;
    const std::vector<float>& v = cache.emplace(t, read_tensor_f32(t)).first->second;
    g_wcache_bytes += v.size() * sizeof(float);
    if (std::getenv("CRISPASR_HTDEMUCS_MEMSTATS"))
        fprintf(stderr, "htdemucs: wcache += %8.2f MB (total %7.2f MB) %s\n", v.size() * 4.0 / 1048576.0,
                g_wcache_bytes / 1048576.0, t->name);
    return v;
}


// ---------------------------------------------------------------------------
// CPU Conv2d for freq encoder (avoids ggml im2col OOM on 8 GB VPS).
// Conv2d with kernel [OC, IC, K, 1], stride [S, 1], pad [P, 0]:
// Only convolves on the freq (H) axis; time (W) axis passes through.
// ---------------------------------------------------------------------------
// Input:  x[T × Fq × IC], layout x[t + fq*T + ic*T*Fq]
// Weight: w ggml tensor ne=(1, K, IC, OC)
// Bias:   b ggml tensor ne=(OC,) or nullptr
// Output: out[T × Fq_out × OC], layout out[t + fo*T + oc*T*Fq_out]
//   where Fq_out = (Fq + 2*P - K) / S + 1
static std::vector<float> cpu_conv2d_freq(const std::vector<float>& x, int T, int Fq, int IC, ggml_tensor* w_tensor,
                                          ggml_tensor* b_tensor, int stride, int pad, int& out_Fq) {
    if (htdemucs_debug())
        fprintf(stderr, "htdemucs: cpu_conv2d_freq T=%d Fq=%d IC=%d w_ne=(%d,%d,%d,%d) type=%d\n", T, Fq, IC,
                (int)w_tensor->ne[0], (int)w_tensor->ne[1], (int)w_tensor->ne[2], (int)w_tensor->ne[3], w_tensor->type);
    const std::vector<float>& w = cached_tensor_f32(w_tensor);
    int K = (int)w_tensor->ne[1];
    int OC = (int)w_tensor->ne[3];
    out_Fq = (Fq + 2 * pad - K) / stride + 1;
    if (htdemucs_debug())
        fprintf(stderr, "htdemucs: cpu_conv2d_freq K=%d OC=%d out_Fq=%d out_size=%zu\n", K, OC, out_Fq,
                (size_t)T * out_Fq * OC);

    // FASTCONV path (CRISPASR_HTDEMUCS_FASTCONV, default ON): batched im2col +
    // ONE gemm. The original per-time-frame path below rebuilt patches and ran a
    // dot-product loop 336x per layer; enc.conv2d was 21.5% of the forward and is
    // ~0.2% here. The old path is KEPT and reachable with FASTCONV=0 — it is the
    // A/B and regression-bisection lever.
    const int patch_cols = IC * K;
    if (!htdemucs_fastconv()) {
        std::vector<float> out((size_t)T * out_Fq * OC, 0.0f);
        std::vector<float> patches((size_t)out_Fq * patch_cols);
        for (int t = 0; t < T; t++) {
            for (int fo = 0; fo < out_Fq; fo++)
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < K; kh++) {
                        const int fi = fo * stride + kh - pad;
                        patches[fo * patch_cols + ic * K + kh] =
                            (fi >= 0 && fi < Fq) ? x[t + (size_t)fi * T + (size_t)ic * T * Fq] : 0.0f;
                    }
            for (int fo = 0; fo < out_Fq; fo++) {
                const float* pp = patches.data() + fo * patch_cols;
                for (int oc = 0; oc < OC; oc++) {
                    const float* wr = w.data() + (size_t)oc * patch_cols;
                    float sum = 0;
                    for (int c = 0; c < patch_cols; c++)
                        sum += pp[c] * wr[c];
                    out[t + (size_t)fo * T + (size_t)oc * T * out_Fq] = sum;
                }
            }
        }
        if (b_tensor) {
            const std::vector<float>& b = cached_tensor_f32(b_tensor);
            for (int oc = 0; oc < OC; oc++)
                for (size_t sp = 0; sp < (size_t)T * out_Fq; sp++)
                    out[sp + (size_t)oc * T * out_Fq] += b[oc];
        }
        return out;
    }
    const size_t n_pos = (size_t)out_Fq * T;
    std::vector<float> out(n_pos * OC, 0.0f);

    // patches[(ic*K + kh), (fo*T + t)] — K-major rows, position-major columns,
    // which is exactly the B operand layout sgemm wants.
    std::vector<float> patches((size_t)patch_cols * n_pos);
    for (int ic = 0; ic < IC; ic++)
        for (int kh = 0; kh < K; kh++) {
            float* prow = patches.data() + (size_t)(ic * K + kh) * n_pos;
            for (int fo = 0; fo < out_Fq; fo++) {
                const int fi = fo * stride + kh - pad;
                float* pdst = prow + (size_t)fo * T;
                if (fi >= 0 && fi < Fq) {
                    const float* xsrc = x.data() + (size_t)fi * T + (size_t)ic * T * Fq;
                    memcpy(pdst, xsrc, (size_t)T * sizeof(float));
                } else {
                    memset(pdst, 0, (size_t)T * sizeof(float));
                }
            }
        }

    // out_tmp[oc, (fo*T + t)] = sum_c w[oc, c] * patches[c, (fo*T + t)]
    htd_gemm(OC, (int)n_pos, patch_cols, w.data(), patches.data(), out.data());

    if (b_tensor) {
        const std::vector<float>& b = cached_tensor_f32(b_tensor);
        for (int oc = 0; oc < OC; oc++) {
            float* orow = out.data() + (size_t)oc * n_pos;
            const float bv = b[oc];
            for (size_t i = 0; i < n_pos; i++)
                orow[i] += bv;
        }
    }
    return out;
}

// CPU 1x1 Conv2d = matmul. w ne=(1,1,IC,OC), spatial = T*Fq.
static std::vector<float> cpu_conv2d_1x1(const std::vector<float>& x, int spatial, int IC, ggml_tensor* w_tensor,
                                         ggml_tensor* b_tensor, int& out_C) {
    // Conv2d weights are ne=(KW, KH, IC, OC). A Conv1d weight is ne=(K, IC, OC)
    // with ne[3]==1, so passing one here would silently yield out_C = 1 and an
    // empty result — that is how the tdecoder rewrite used to null-deref.
    GGML_ASSERT(w_tensor->ne[0] == 1 && w_tensor->ne[1] == 1 &&
                "cpu_conv2d_1x1 requires a 1x1 Conv2d kernel; use cpu_conv1d_time for Conv1d");
    const std::vector<float>& w = cached_tensor_f32(w_tensor);
    out_C = (int)w_tensor->ne[3];
    std::vector<float> out((size_t)spatial * out_C, 0.0f);
    // A 1x1 conv is a pure channel matmul: (out_C, IC) x (IC, spatial).
    // Gated CRISPASR_HTDEMUCS_FASTCONV (default ON); the scalar path is kept.
    if (htdemucs_fastconv()) {
        htd_gemm(out_C, spatial, IC, w.data(), x.data(), out.data());
    } else {
        for (int oc = 0; oc < out_C; oc++)
            for (int sp = 0; sp < spatial; sp++) {
                float sum = 0;
                for (int ic = 0; ic < IC; ic++)
                    sum += w[(size_t)oc * IC + ic] * x[sp + (size_t)ic * spatial];
                out[sp + (size_t)oc * spatial] = sum;
            }
    }
    if (b_tensor) {
        const std::vector<float>& b = cached_tensor_f32(b_tensor);
        for (int oc = 0; oc < out_C; oc++)
            for (int s = 0; s < spatial; s++)
                out[s + (size_t)oc * spatial] += b[oc];
    }
    return out;
}

// CPU GELU (tanh approximation)
static void cpu_gelu_inplace(std::vector<float>& x) {
    for (size_t i = 0; i < x.size(); i++) {
        float v = x[i];
        x[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

// CPU Conv1d over a channel-major (C, T) buffer with symmetric zero padding.
// Weight ne = (K, IC, OC), row-major w[oc][ic][k]. Returns (OC, T_out) with
// T_out = T + 2*pad - K + 1 (stride 1, as used by the decoder rewrites).
static std::vector<float> cpu_conv1d_time(const std::vector<float>& x, int T, int IC, ggml_tensor* w_tensor,
                                          ggml_tensor* b_tensor, int pad, int& out_C, int& out_T) {
    const int K = (int)w_tensor->ne[0];
    out_C = (int)w_tensor->ne[2];
    out_T = T + 2 * pad - K + 1;
    const std::vector<float>& w = cached_tensor_f32(w_tensor);
    std::vector<float> out((size_t)out_T * out_C, 0.0f);
    if (htdemucs_fastconv()) {
        // im2col + one GEMM: (out_C, IC*K) x (IC*K, out_T). The scalar form below
        // was 45% of the forward once everything else was optimised — this is the
        // tdecoder rewrite, a K=3 conv over ~344k time steps at the last layer.
        std::vector<float> patches((size_t)IC * K * out_T, 0.0f);
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++) {
                float* prow = patches.data() + (size_t)(ic * K + k) * out_T;
                const float* xsrc = x.data() + (size_t)ic * T;
                const int lo = std::max(0, pad - k);
                const int hi = std::min(out_T, T + pad - k);
                for (int t_out = lo; t_out < hi; t_out++)
                    prow[t_out] = xsrc[t_out + k - pad];
            }
        htd_gemm(out_C, out_T, IC * K, w.data(), patches.data(), out.data());
    } else {
        for (int oc = 0; oc < out_C; oc++)
            for (int t_out = 0; t_out < out_T; t_out++) {
                float sum = 0;
                for (int ic = 0; ic < IC; ic++)
                    for (int k = 0; k < K; k++) {
                        const int t_in = t_out + k - pad;
                        if (t_in < 0 || t_in >= T)
                            continue;
                        sum += x[t_in + (size_t)ic * T] * w[((size_t)oc * IC + ic) * K + k];
                    }
                out[t_out + (size_t)oc * out_T] = sum;
            }
    }
    if (b_tensor) {
        const std::vector<float>& b = cached_tensor_f32(b_tensor);
        for (int oc = 0; oc < out_C; oc++)
            for (int t = 0; t < out_T; t++)
                out[t + (size_t)oc * out_T] += b[oc];
    }
    return out;
}

// CPU GroupNorm with num_groups=1 over a channel-major (C, T) buffer, followed
// by the per-channel affine. num_groups=1 means the single group spans ALL
// channels, so mean/var are taken jointly over every (c, t) element — not
// per-channel. Matches torch.nn.GroupNorm(1, C, eps=1e-5) as used by the
// DConv sublayers (demucs.demucs.DConv).
static void cpu_group_norm1_inplace(float* x, int C, int T, const float* w, const float* b, float eps = 1e-5f) {
    const size_t n = (size_t)C * T;
    if (n == 0)
        return;
    double mean = 0.0;
    for (size_t i = 0; i < n; i++)
        mean += x[i];
    mean /= (double)n;
    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= (double)n; // biased, as torch normalization layers use
    const float inv = 1.0f / std::sqrt((float)var + eps);
    const float mf = (float)mean;
    for (int c = 0; c < C; c++) {
        const float wc = w ? w[c] : 1.0f;
        const float bc = b ? b[c] : 0.0f;
        float* row = x + (size_t)c * T;
        for (int t = 0; t < T; t++)
            row[t] = (row[t] - mf) * inv * wc + bc;
    }
}

static void cpu_group_norm1_inplace(std::vector<float>& x, int C, int T, const float* w, const float* b,
                                    float eps = 1e-5f) {
    cpu_group_norm1_inplace(x.data(), C, T, w, b, eps);
}

// A K=1 ggml_conv_1d is a pure channel matmul; routing it through
// ggml_conv_1d materialises an im2col that is just a copy of the input (the
// dev-guide "K=1 conv is a channel matmul" rule). These four channel
// up/down-samplers measured 41% of the forward once everything else was
// optimised. Operates in place on a (C, spatial) buffer.
static void htd_k1_conv(std::vector<float>& buf, int& C_io, int spatial, ggml_tensor* w, ggml_tensor* b) {
    const int oc_n = (int)w->ne[2]; // Conv1d weight ne = (K=1, IC, OC)
    std::vector<float> out((size_t)oc_n * spatial, 0.0f);
    htd_gemm(oc_n, spatial, C_io, cached_tensor_f32(w).data(), buf.data(), out.data());
    if (b) {
        const std::vector<float>& bb = cached_tensor_f32(b);
        for (int o = 0; o < oc_n; o++) {
            float* r = out.data() + (size_t)o * spatial;
            for (int i = 0; i < spatial; i++)
                r[i] += bb[o];
        }
    }
    buf = std::move(out);
    C_io = oc_n;
}

// CPU DConv residual stack over a channel-major (C, T) buffer, in place.
//
// Mirrors demucs.demucs.DConv: for each sublayer d,
//   h = Conv1d(C -> hidden, K, dilation=2^d, padding=dilation*(K/2))(x)
//   h = GELU(GroupNorm(1, hidden)(h))
//   h = Conv1d(hidden -> 2C, K=1)(h)
//   h = GLU(GroupNorm(1, 2C)(h))
//   x = x + LayerScale * h
//
// Used by BOTH the frequency branch (once per frequency band, matching
// Python's `y.permute(0,2,1,3).reshape(-1, C, T)`) and the time branch.
// GroupNorm(num_groups=1) over one band of a (rows, n_cols) buffer whose columns
// for that band are the contiguous span [col0, col0+T). Mean/var are taken over
// all rows AND those T columns jointly, matching torch GroupNorm(1, rows) applied
// per batch element. Affine is per-row.
static void cpu_group_norm1_strided(float* buf, int rows, int n_cols, int col0, int T, const float* w, const float* b,
                                    float eps = 1e-5f) {
    const size_t n = (size_t)rows * T;
    if (n == 0)
        return;
    double mean = 0.0;
    for (int r = 0; r < rows; r++) {
        const float* p = buf + (size_t)r * n_cols + col0;
        for (int t = 0; t < T; t++)
            mean += p[t];
    }
    mean /= (double)n;
    double var = 0.0;
    for (int r = 0; r < rows; r++) {
        const float* p = buf + (size_t)r * n_cols + col0;
        for (int t = 0; t < T; t++) {
            const double d = p[t] - mean;
            var += d * d;
        }
    }
    var /= (double)n;
    const float inv = 1.0f / std::sqrt((float)var + eps);
    const float mf = (float)mean;
    for (int r = 0; r < rows; r++) {
        float* p = buf + (size_t)r * n_cols + col0;
        const float wr = w ? w[r] : 1.0f;
        const float br = b ? b[r] : 0.0f;
        for (int t = 0; t < T; t++)
            p[t] = (p[t] - mf) * inv * wr + br;
    }
}

// Batched DConv over ALL frequency bands at once, operating directly on the
// (C, Fq, T) buffer (index t + fq*T + c*T*Fq) with no per-band copy.
//
// The per-band version ran the whole stack 512/128/32/8 times per encoder layer
// — 680 invocations of two tiny convs each. Here conv1 and conv2 become ONE GEMM
// apiece across every band, since both are linear with band-independent weights:
//
//   conv1: (hidden, C*K)  x (C*K, n_bands*T)   [im2col over dilated taps]
//   conv2: (2C,     hidden) x (hidden, n_bands*T)
//
// GroupNorm stays per-band (num_groups=1 normalises each band separately);
// GELU/GLU are elementwise; LayerScale is per-channel and shared across bands.
static void cpu_dconv_bands(std::vector<float>& x, int n_bands, int C, int T, const htdemucs_dconv& dc) {
    const size_t n_cols = (size_t)n_bands * T;
    for (size_t d = 0; d < dc.layers.size(); d++) {
        const auto& sl = dc.layers[d];
        if (!sl.conv1_w || !sl.conv2_w)
            continue;
        const int dilation = 1 << (int)d;
        const int K = (int)sl.conv1_w->ne[0];
        const int hidden = (int)sl.conv1_w->ne[2];
        const int patch_rows = C * K;

        // im2col: patches[(ic*K + k), (fq*T + t_out)], read straight out of the
        // (C, Fq, T) buffer. Out-of-range taps are the conv's zero padding.
        std::vector<float> patches((size_t)patch_rows * n_cols, 0.0f);
        for (int ic = 0; ic < C; ic++)
            for (int k = 0; k < K; k++) {
                float* prow = patches.data() + (size_t)(ic * K + k) * n_cols;
                const int shift = (k - K / 2) * dilation;
                for (int fq = 0; fq < n_bands; fq++) {
                    const float* xsrc = x.data() + (size_t)fq * T + (size_t)ic * T * n_bands;
                    float* pdst = prow + (size_t)fq * T;
                    const int lo = std::max(0, -shift);
                    const int hi = std::min(T, T - shift);
                    for (int t = lo; t < hi; t++)
                        pdst[t] = xsrc[t + shift];
                }
            }

        std::vector<float> h((size_t)hidden * n_cols, 0.0f);
        htd_gemm(hidden, (int)n_cols, patch_rows, cached_tensor_f32(sl.conv1_w).data(), patches.data(), h.data());
        if (sl.conv1_b) {
            const std::vector<float>& b1 = cached_tensor_f32(sl.conv1_b);
            for (int hc = 0; hc < hidden; hc++) {
                float* hr = h.data() + (size_t)hc * n_cols;
                for (size_t i = 0; i < n_cols; i++)
                    hr[i] += b1[hc];
            }
        }
        if (sl.norm1_w) {
            const std::vector<float>& n1w = cached_tensor_f32(sl.norm1_w);
            const float* n1b = sl.norm1_b ? cached_tensor_f32(sl.norm1_b).data() : nullptr;
            for (int fq = 0; fq < n_bands; fq++)
                cpu_group_norm1_strided(h.data(), hidden, (int)n_cols, fq * T, T, n1w.data(), n1b);
        }
        cpu_gelu_inplace(h);

        const int out2C = (int)sl.conv2_w->ne[2];
        std::vector<float> h2((size_t)out2C * n_cols, 0.0f);
        htd_gemm(out2C, (int)n_cols, hidden, cached_tensor_f32(sl.conv2_w).data(), h.data(), h2.data());
        if (sl.conv2_b) {
            const std::vector<float>& b2 = cached_tensor_f32(sl.conv2_b);
            for (int oc = 0; oc < out2C; oc++) {
                float* hr = h2.data() + (size_t)oc * n_cols;
                for (size_t i = 0; i < n_cols; i++)
                    hr[i] += b2[oc];
            }
        }
        if (sl.norm2_w) {
            const std::vector<float>& n2w = cached_tensor_f32(sl.norm2_w);
            const float* n2b = sl.norm2_b ? cached_tensor_f32(sl.norm2_b).data() : nullptr;
            for (int fq = 0; fq < n_bands; fq++)
                cpu_group_norm1_strided(h2.data(), out2C, (int)n_cols, fq * T, T, n2w.data(), n2b);
        }

        // GLU + LayerScale + residual, straight back into the (C, Fq, T) buffer.
        const int half = out2C / 2;
        const std::vector<float>* sc = sl.scale ? &cached_tensor_f32(sl.scale) : nullptr;
        for (int c = 0; c < half; c++) {
            const float* a = h2.data() + (size_t)c * n_cols;
            const float* g = h2.data() + (size_t)(half + c) * n_cols;
            const float s = sc ? (*sc)[c] : 1.0f;
            for (int fq = 0; fq < n_bands; fq++) {
                float* xdst = x.data() + (size_t)fq * T + (size_t)c * T * n_bands;
                const size_t col = (size_t)fq * T;
                for (int t = 0; t < T; t++)
                    xdst[t] += s * (a[col + t] / (1.0f + expf(-g[col + t])));
            }
        }
    }
}

static void cpu_dconv_inplace(std::vector<float>& x, int C, int T, const htdemucs_dconv& dc) {
    for (size_t d = 0; d < dc.layers.size(); d++) {
        const auto& sl = dc.layers[d];
        if (!sl.conv1_w || !sl.conv2_w)
            continue;
        const int dilation = 1 << (int)d;
        const int K = (int)sl.conv1_w->ne[0];
        const int hidden = (int)sl.conv1_w->ne[2];

        // Dilated Conv1d(C -> hidden) with padding = dilation * (K/2).
        const std::vector<float>& w1 = cached_tensor_f32(sl.conv1_w);
        std::vector<float> h((size_t)T * hidden, 0.0f);
        for (int t_out = 0; t_out < T; t_out++)
            for (int hc = 0; hc < hidden; hc++) {
                float sum = 0;
                for (int ic = 0; ic < C; ic++)
                    for (int k = 0; k < K; k++) {
                        const int t_in = t_out + (k - K / 2) * dilation;
                        if (t_in < 0 || t_in >= T)
                            continue;
                        sum += x[t_in + (size_t)ic * T] * w1[(size_t)hc * C * K + ic * K + k];
                    }
                h[t_out + (size_t)hc * T] = sum;
            }
        if (sl.conv1_b) {
            const std::vector<float>& b1 = cached_tensor_f32(sl.conv1_b);
            for (int hc = 0; hc < hidden; hc++)
                for (int t = 0; t < T; t++)
                    h[t + (size_t)hc * T] += b1[hc];
        }

        if (sl.norm1_w) {
            const std::vector<float>& n1w = cached_tensor_f32(sl.norm1_w);
            cpu_group_norm1_inplace(h, hidden, T, n1w.data(),
                                    sl.norm1_b ? cached_tensor_f32(sl.norm1_b).data() : nullptr);
        }
        cpu_gelu_inplace(h);

        // Conv1d(hidden -> 2C, K=1) == a per-timestep matmul.
        const std::vector<float>& w2 = cached_tensor_f32(sl.conv2_w);
        const int out2C = (int)sl.conv2_w->ne[2];
        std::vector<float> h2((size_t)T * out2C, 0.0f);
        for (int oc = 0; oc < out2C; oc++)
            for (int t = 0; t < T; t++) {
                float sum = 0;
                for (int hc = 0; hc < hidden; hc++)
                    sum += w2[(size_t)oc * hidden + hc] * h[t + (size_t)hc * T];
                h2[t + (size_t)oc * T] = sum;
            }
        if (sl.conv2_b) {
            const std::vector<float>& b2 = cached_tensor_f32(sl.conv2_b);
            for (int oc = 0; oc < out2C; oc++)
                for (int t = 0; t < T; t++)
                    h2[t + (size_t)oc * T] += b2[oc];
        }

        if (sl.norm2_w) {
            const std::vector<float>& n2w = cached_tensor_f32(sl.norm2_w);
            cpu_group_norm1_inplace(h2, out2C, T, n2w.data(),
                                    sl.norm2_b ? cached_tensor_f32(sl.norm2_b).data() : nullptr);
        }

        // GLU on the channel dim -> C channels, then LayerScale + residual.
        const int half = out2C / 2;
        std::vector<float> gl((size_t)T * half);
        for (int c = 0; c < half; c++)
            for (int t = 0; t < T; t++) {
                const float a = h2[t + (size_t)c * T];
                const float b = h2[t + (size_t)(half + c) * T];
                gl[t + (size_t)c * T] = a / (1.0f + expf(-b));
            }
        const std::vector<float>* sc = sl.scale ? &cached_tensor_f32(sl.scale) : nullptr;
        for (int c = 0; c < C; c++) {
            const float s = sc ? (*sc)[c] : 1.0f;
            for (int t = 0; t < T; t++)
                x[t + (size_t)c * T] += s * gl[t + (size_t)c * T];
        }
    }
}

// CPU GLU: split channel dim, a * sigmoid(b)
// x layout: (spatial, 2*C). Returns (spatial, C).
static std::vector<float> cpu_glu(const std::vector<float>& x, int spatial, int double_C) {
    int half = double_C / 2;
    std::vector<float> out((size_t)spatial * half);
    for (int c = 0; c < half; c++)
        for (int s = 0; s < spatial; s++) {
            float a = x[s + (size_t)c * spatial];
            float b = x[s + (size_t)(half + c) * spatial];
            out[s + (size_t)c * spatial] = a / (1.0f + expf(-b));
        }
    return out;
}

// ---------------------------------------------------------------------------
// ggml graph helpers for the encoder/decoder/transformer
// ---------------------------------------------------------------------------

// GroupNorm + affine: y = weight * group_norm(x) + bias
// GroupNorm(num_groups=1) + per-channel affine for a 2D (T, C) tensor.
// Reshapes to (T, 1, C, 1) so ggml_group_norm's single group spans all C
// channels (normalizing jointly over T*C, matching torch GroupNorm(1, C)), then
// applies the affine with the weight reshaped to (1, 1, C, 1) so it broadcasts
// along the channel axis rather than against ne[0].
//
// Unused for htdemucs itself (norm_starts=4 with depth=4 leaves every encoder
// norm as Identity), but kept correct so a model with norms enabled works.
static ggml_tensor* ggml_group_norm_affine_2d(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                              float eps) {
    const int T = (int)x->ne[0], C = (int)x->ne[1];
    ggml_tensor* y = ggml_reshape_4d(g, x, T, 1, C, 1);
    y = ggml_group_norm(g, y, 1, eps);
    if (w)
        y = ggml_mul(g, y, ggml_reshape_4d(g, w, 1, 1, (int)w->ne[0], 1));
    if (b)
        y = ggml_add(g, y, ggml_reshape_4d(g, b, 1, 1, (int)b->ne[0], 1));
    return ggml_reshape_2d(g, y, T, C);
}

static ggml_tensor* ggml_group_norm_affine(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
                                           int n_groups, float eps) {
    ggml_tensor* y = ggml_group_norm(g, x, n_groups, eps);
    if (w)
        y = ggml_mul(g, y, w);
    if (b)
        y = ggml_add(g, y, b);
    return y;
}

// LayerNorm (= GroupNorm with groups=1, applied on last dim)
static ggml_tensor* ggml_layer_norm_affine(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* y = ggml_norm(g, x, eps);
    if (w)
        y = ggml_mul(g, y, w);
    if (b)
        y = ggml_add(g, y, b);
    return y;
}

// DConv residual block: x = x + LayerScale * GLU(Norm(Conv1d(GELU(Norm(DilConv(x))))))
// Applied on time axis for both freq and time branches (freq branch reshapes to apply DConv per-freq-band).
static ggml_tensor* apply_dconv(ggml_context* g, ggml_tensor* x, const htdemucs_dconv& dc) {
    for (size_t d = 0; d < dc.layers.size(); d++) {
        auto& sl = dc.layers[d];
        if (!sl.conv1_w)
            continue;
        // Residual: x = x + scale * GLU(norm2(conv2(GELU(norm1(conv1(x))))))
        int dilation = 1 << (int)d;
        int K = (int)sl.conv1_w->ne[0];
        int pad = dilation * (K / 2);

        ggml_tensor* h = ggml_conv_1d(g, sl.conv1_w, x, 1, pad, dilation);
        if (sl.conv1_b)
            h = ggml_add(g, h, sl.conv1_b);
        if (sl.norm1_w)
            h = ggml_group_norm_affine(g, h, sl.norm1_w, sl.norm1_b, 1, 1e-5f);
        h = ggml_gelu(g, h);
        // 1x1 conv → 2*channels
        h = ggml_conv_1d(g, sl.conv2_w, h, 1, 0, 1);
        if (sl.conv2_b)
            h = ggml_add(g, h, sl.conv2_b);
        if (sl.norm2_w)
            h = ggml_group_norm_affine(g, h, sl.norm2_w, sl.norm2_b, 1, 1e-5f);
        // GLU on channel dim (dim=1 for (C, T), but ggml is (T, 2*C) → split on ne[0] axis)
        // Actually ggml tensors from conv_1d come out as (T_out, C_out).
        // GLU splits the channel dim in half.
        // For ggml: conv_1d output is (T, 2*C). GLU along dim 0 (the channel dim in ggml).
        // ggml doesn't have GLU directly. GLU(x) = x[:C] * sigmoid(x[C:])
        int C_half = (int)h->ne[0] / 2;
        int T_out = (int)h->ne[1];
        ggml_tensor* a = ggml_view_2d(g, h, C_half, T_out, h->nb[1], 0);
        ggml_tensor* b_gate = ggml_view_2d(g, h, C_half, T_out, h->nb[1], C_half * ggml_element_size(h));
        h = ggml_mul(g, ggml_cont(g, a), ggml_sigmoid(g, ggml_cont(g, b_gate)));
        // LayerScale
        if (sl.scale)
            h = ggml_mul(g, h, sl.scale);
        x = ggml_add(g, x, h);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Full forward pass (Phase 2b — in progress)
// ---------------------------------------------------------------------------

// The full forward is built as a ggml graph. This is the core of the
// separation engine. It mirrors htdemucs.py:HTDemucs.forward() exactly.
//
// Input:  stereo PCM at 44100 Hz (2 channels, n_samples per channel).
// Output: 4 × stereo PCM (drums, bass, other, vocals).

// Whole-buffer forward: one graph over the entire input. Peak memory and time
// both grow with length (time QUADRATICALLY -- the CrossTransformer attends
// over all time frames), so htdemucs_separate below only calls this directly
// for inputs at or under one segment. Measured on M1 before segmentation:
// 2 s 1.13 GB / 10 s 1.44 GB / 30 s 3.39 GB / 60 s 7.62 GB, and 30 s -> 60 s
// took 51.9 s -> 197 s for 2x the audio.
static htdemucs_result* htdemucs_separate_full(htdemucs_context* ctx, const float* pcm_stereo, int n_samples) {
    if (!ctx || !pcm_stereo || n_samples <= 0)
        return nullptr;

    auto& hp = ctx->model.hparams;
    auto& m = ctx->model;
    htd_prof prof;
    if (htdemucs_profile())
        prof.t_start = htd_now_ms();

    // Step 1: deinterleave stereo to channel-major
    int training_length = hp.training_length();
    int work_length = std::max(n_samples, training_length);
    std::vector<float> pcm_ch(2 * work_length, 0.0f);
    for (int i = 0; i < n_samples; i++) {
        pcm_ch[i] = pcm_stereo[2 * i];                   // L
        pcm_ch[work_length + i] = pcm_stereo[2 * i + 1]; // R
    }

    // Step 2: HTDemucs-specific STFT (matches _spec() in htdemucs.py)
    //
    // The Python _spec() does:
    //   le = ceil(T / hop_length)
    //   pad = hop_length // 2 * 3   (= 1536 for hop=1024)
    //   x = pad1d(x, (pad, pad + le*hop - T), mode="reflect")
    //   z = spectro(x, nfft, hop)   (center=True, normalized=True)
    //   z = z[..., :-1, :]          (drop last freq bin: 2049 → 2048)
    //   z = z[..., 2:2+le]          (trim to le frames)
    //
    // spectro's center=True adds nfft/2 reflect pad on each side internally.
    // So total padding before the raw STFT is:
    //   left:  pad + nfft/2 = 1536 + 2048 = 3584
    //   right: (pad + le*hop - T) + nfft/2

    int nfft = hp.nfft;
    int hop = hp.hop_length();
    int le = (int)ceil((double)work_length / hop);

    // Apply _spec() pre-padding (reflect) to each channel
    int pre_pad_left = hop / 2 * 3; // 1536
    int pre_pad_right = pre_pad_left + le * hop - work_length;
    int pre_padded_len = work_length + pre_pad_left + pre_pad_right;
    std::vector<float> pre_padded(2 * pre_padded_len);

    for (int ch = 0; ch < 2; ch++) {
        const float* src = pcm_ch.data() + (size_t)ch * work_length;
        float* dst = pre_padded.data() + (size_t)ch * pre_padded_len;
        for (int i = 0; i < pre_padded_len; i++) {
            int idx = i - pre_pad_left;
            // Reflect padding (matches pad1d with mode="reflect")
            if (idx < 0)
                idx = -idx;
            if (idx >= work_length)
                idx = 2 * work_length - 2 - idx;
            idx = std::max(0, std::min(work_length - 1, idx));
            dst[i] = src[idx];
        }
    }

    // Now run the raw STFT with center=True on the pre-padded signal
    stft_result spec;
    {
        HTD_PROF(prof, "stft");
        spec = compute_stft(pre_padded.data(), 2, pre_padded_len, nfft, hop, ctx->hann_window.data());
    }

    // z[..., :-1, :] — drop the last frequency bin (2049 → 2048)
    int Fq = nfft / 2; // 2048 (was nfft/2+1 = 2049)

    // z[..., 2:2+le] — take frames 2..2+le (the _spec frame slicing)
    int T = le;
    int frame_offset = 2;

    // Build CaC magnitude: (B, C*2, Fq, T) from complex spectrogram.
    // For B=1, C=2: we get 4 channels — [real_L, real_R, imag_L, imag_R]
    // Actually Python does: view_as_real(z).permute(0,1,4,2,3).reshape(B, C*2, Fr, T)
    // where view_as_real gives (B, C, Fr, T, 2), permute → (B, C, 2, Fr, T),
    // reshape → (B, C*2, Fr, T) = (1, 4, 2048, le)
    // So channel order is: [L_real, L_imag, R_real, R_imag]
    int n_cac_ch = hp.audio_channels * 2; // 4
    std::vector<float> cac_mag(n_cac_ch * Fq * T, 0.0f);

    for (int ch = 0; ch < 2; ch++) {
        for (int fq = 0; fq < Fq; fq++) {
            for (int t = 0; t < T; t++) {
                size_t src_idx =
                    (size_t)ch * spec.n_freqs * spec.n_frames + (size_t)fq * spec.n_frames + (frame_offset + t);
                float re_val = spec.real[src_idx];
                float im_val = spec.imag[src_idx];
                // CaC channel order: [ch*2] = real, [ch*2+1] = imag
                cac_mag[((size_t)(ch * 2) * Fq + fq) * T + t] = re_val;
                cac_mag[((size_t)(ch * 2 + 1) * Fq + fq) * T + t] = im_val;
            }
        }
    }

    if (htdemucs_debug()) {
        fprintf(stderr, "htdemucs: STFT spec %d×%d, CaC %d×%d×%d\n", spec.n_freqs, spec.n_frames, n_cac_ch, Fq, T);
    }

    fprintf(stderr, "htdemucs: STFT → %d freqs × %d frames, CaC → %d channels\n", Fq, T, hp.cac ? 4 : 2);
    // Step 4: Normalize spec branch (mean/std over all dims)
    float spec_mean = 0.0f, spec_var = 0.0f;
    size_t spec_n = (size_t)n_cac_ch * Fq * T;
    for (size_t i = 0; i < spec_n; i++)
        spec_mean += cac_mag[i];
    spec_mean /= (float)spec_n;
    for (size_t i = 0; i < spec_n; i++) {
        float d = cac_mag[i] - spec_mean;
        spec_var += d * d;
    }
    float spec_std = sqrtf(spec_var / (float)spec_n);
    for (size_t i = 0; i < spec_n; i++) {
        cac_mag[i] = (cac_mag[i] - spec_mean) / (1e-5f + spec_std);
    }
    // cac_mag is already (C, Fq, T) with t fastest — the reference layout.
    htd_capture(ctx, "spec_input", cac_mag.data(), spec_n);

    // Step 5: Normalize time branch
    float time_mean = 0.0f, time_var = 0.0f;
    size_t time_n = (size_t)2 * work_length;
    for (size_t i = 0; i < time_n; i++)
        time_mean += pcm_ch[i];
    time_mean /= (float)time_n;
    for (size_t i = 0; i < time_n; i++) {
        float d = pcm_ch[i] - time_mean;
        time_var += d * d;
    }
    float time_std = sqrtf(time_var / (float)time_n);
    for (size_t i = 0; i < time_n; i++) {
        pcm_ch[i] = (pcm_ch[i] - time_mean) / (1e-5f + time_std);
    }
    // pcm_ch is (2, work_length), channel-major — the reference layout.
    htd_capture(ctx, "time_input", pcm_ch.data(), time_n);

    if (htdemucs_debug()) {
        fprintf(stderr, "htdemucs: spec_norm mean=%.6f std=%.6f, time_norm mean=%.6f std=%.6f\n", spec_mean, spec_std,
                time_mean, time_std);
    }

    // Step 6: Encoder forward (dimension tracking for now, ggml graphs next)
    //
    // Freq branch: cac_mag[n_cac_ch × Fq × T], Time branch: pcm_ch[2 × work_length]
    // Each encoder layer changes dims; we track them for decoder symmetry.

    std::vector<float> x_buf(cac_mag);
    int x_C = n_cac_ch, x_Fq = Fq, x_T = T;
    std::vector<float> xt_buf(pcm_ch.begin(), pcm_ch.begin() + 2 * work_length);
    int xt_C = hp.audio_channels, xt_T = work_length;

    // Skip connection storage for decoder
    struct saved_activation {
        std::vector<float> data;
        int C, Fq, T;
    };
    std::vector<saved_activation> saved_freq; // freq branch skips
    std::vector<saved_activation> saved_time; // time branch skips
    std::vector<int> lengths_freq;            // saved input T for decoder
    std::vector<int> lengths_time;

    int freqs_cur = nfft / 2;
    bool encoder_ok = true;

    // FUSED path: the whole frequency chain runs as one graph after the time
    // encoder, so keep the untouched spec_input and pre-build the freq_emb
    // broadcast (1, Fq, C) with the 10 * freq_emb_scale factor applied.
    const bool fused = htdemucs_use_ggml() && htdemucs_use_fused();
    std::vector<float> fused_spec_in;
    std::vector<float> fused_freq_emb;
    if (fused) {
        fused_spec_in = x_buf;
        if (m.freq_emb_w) {
            const int emb_C = (int)m.freq_emb_w->ne[0];
            const int emb_F = (int)m.freq_emb_w->ne[1];
            const std::vector<float>& ew = cached_tensor_f32(m.freq_emb_w);
            const float ts = 10.0f * hp.freq_emb_scale;
            fused_freq_emb.assign((size_t)emb_F * emb_C, 0.0f);
            for (int f = 0; f < emb_F; f++)
                for (int c = 0; c < emb_C; c++)
                    fused_freq_emb[(size_t)c * emb_F + f] = ew[(size_t)f * emb_C + c] * ts;
        }
    }

    for (int idx = 0; idx < hp.depth && encoder_ok; idx++) {
        auto& enc = m.encoder[idx];
        bool freq = (freqs_cur > 1);
        int stri = freq ? hp.stride : 2;
        int ker = freq ? hp.kernel_size : 4;
        int pad_val = ker / 4;
        bool last_freq = false;
        (void)last_freq;

        if (freq && freqs_cur <= hp.kernel_size) {
            ker = freqs_cur;
            pad_val = 0;
            last_freq = true;
        }

        if (htdemucs_debug()) {
            fprintf(stderr, "htdemucs: enc[%d] x=(%d,%d,%d) xt=(%d,%d) freq=%d freqs=%d ker=%d stri=%d pad=%d\n", idx,
                    x_C, x_Fq, x_T, xt_C, xt_T, freq ? 1 : 0, freqs_cur, ker, stri, pad_val);
        }

        // Python records the time length BEFORE the tencoder runs
        //   (`lengths_t.append(xt.shape[-1])` precedes `xt = tenc(xt)`),
        // and the decoder pops it to crop each ConvTranspose1d back to its
        // layer's INPUT length. Recording the OUTPUT length instead shifted the
        // whole stack by one level, so the last tdecoder produced 85995 samples
        // instead of 343980 — which silently failed the
        // `xt_T >= n_samples` guard below and dropped the time branch from
        // every stem.
        if (idx < (int)m.tencoder.size())
            lengths_time.push_back(xt_T);

        // --- Time branch encoder (runs before freq branch) ---
        std::vector<float> inject_buf; // injection from time→freq at merge point
        bool has_inject = false;
        if (idx < (int)m.tencoder.size() && m.tencoder[idx].conv_w && !getenv("CRISPASR_HTDEMUCS_SKIP_TIME")) {
            HTD_PROF(prof, "tenc[nested total]");
            auto& tenc = m.tencoder[idx];

            // Pad xt so length is divisible by stride
            int xt_pad = 0;
            if (xt_T % hp.stride != 0) {
                xt_pad = hp.stride - (xt_T % hp.stride);
            }
            int xt_padded_T = xt_T + xt_pad;

            // The time encoder is split into two ggml graphs so the DConv stack
            // can run on the CPU between them, matching HEncLayer.forward:
            //   conv -> GELU(norm1) -> dconv -> GLU(norm2(rewrite))
            // ggml's (T, C) 2D layout is flat `t + c*T`, i.e. exactly the
            // channel-major (C, T) convention xt_buf uses, so no transpose is
            // needed when handing buffers between the graphs and the CPU code.

            // --- Graph 1: Conv1d + bias + GELU (norm1 is Identity for htdemucs) ---
            {
                HTD_PROF(prof, "tenc.conv(ggml)");
                size_t t_ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
                ggml_init_params tgp = {t_ctx_size, nullptr, true};
                ggml_context* tg = ggml_init(tgp);

                ggml_tensor* xt_in = ggml_new_tensor_2d(tg, GGML_TYPE_F32, xt_padded_T, xt_C);
                ggml_set_name(xt_in, "tenc_in");
                ggml_set_input(xt_in);

                int t_ker = hp.kernel_size;
                int t_stri = hp.stride;
                int t_pad = t_ker / 4;
                ggml_tensor* ty = ggml_conv_1d(tg, tenc.conv_w, xt_in, t_stri, t_pad, 1);
                if (tenc.conv_b) {
                    ggml_tensor* tb = ggml_reshape_2d(tg, tenc.conv_b, 1, (int)tenc.conv_b->ne[0]);
                    ty = ggml_add(tg, ty, tb);
                }
                if (!tenc.empty) {
                    if (tenc.norm1_w)
                        ty = ggml_group_norm_affine_2d(tg, ty, tenc.norm1_w, tenc.norm1_b, 1e-5f);
                    ty = ggml_gelu(tg, ty);
                }

                ggml_set_name(ty, "tenc_pre_dconv");
                ggml_set_output(ty);
                ggml_cgraph* tgf = ggml_new_graph(tg);
                ggml_build_forward_expand(tgf, ty);
                ggml_gallocr_t talloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
                if (ggml_gallocr_alloc_graph(talloc, tgf)) {
                    std::vector<float> xt_padded((size_t)xt_C * xt_padded_T, 0.0f);
                    for (int c = 0; c < xt_C; c++)
                        memcpy(xt_padded.data() + (size_t)c * xt_padded_T, xt_buf.data() + (size_t)c * xt_T,
                               (size_t)xt_T * sizeof(float));
                    ggml_backend_tensor_set(xt_in, xt_padded.data(), 0, xt_padded.size() * sizeof(float));
                    ggml_backend_graph_compute(ctx->backend, tgf);

                    xt_T = (int)ty->ne[0];
                    xt_C = (int)ty->ne[1];
                    xt_buf = read_tensor_f32(ty);
                }
                ggml_gallocr_free(talloc);
                ggml_free(tg);
            }

            // --- CPU DConv (was missing entirely: the time branch skipped it) ---
            if (!tenc.empty && !tenc.dconv.layers.empty()) {
                HTD_PROF(prof, "tenc.dconv");
                // The time branch is a single band, so there is nothing to batch,
                // but cpu_dconv_bands still gets the im2col+GEMM treatment for the
                // two convs. Its (C, Fq, T) indexing collapses to (C, T) at
                // n_bands = 1, which is exactly xt_buf's layout.
                if (htdemucs_fastconv())
                    cpu_dconv_bands(xt_buf, 1, xt_C, xt_T, tenc.dconv);
                else
                    cpu_dconv_inplace(xt_buf, xt_C, xt_T, tenc.dconv);
            }

            // --- Graph 2: rewrite Conv1d(K=1) + norm2 + GLU ---
            if (!tenc.empty && tenc.rewrite_w) {
                HTD_PROF(prof, "tenc.rewrite(ggml)");
                size_t t_ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
                ggml_init_params tgp = {t_ctx_size, nullptr, true};
                ggml_context* tg = ggml_init(tgp);

                ggml_tensor* rw_in = ggml_new_tensor_2d(tg, GGML_TYPE_F32, xt_T, xt_C);
                ggml_set_name(rw_in, "tenc_rewrite_in");
                ggml_set_input(rw_in);

                ggml_tensor* trw = ggml_conv_1d(tg, tenc.rewrite_w, rw_in, 1, 0, 1);
                if (tenc.rewrite_b) {
                    ggml_tensor* trb = ggml_reshape_2d(tg, tenc.rewrite_b, 1, (int)tenc.rewrite_b->ne[0]);
                    trw = ggml_add(tg, trw, trb);
                }
                if (tenc.norm2_w)
                    trw = ggml_group_norm_affine_2d(tg, trw, tenc.norm2_w, tenc.norm2_b, 1e-5f);

                // GLU over the channel dim (ne[1]).
                int trw_T = (int)trw->ne[0];
                int trw_half = (int)trw->ne[1] / 2;
                ggml_tensor* ta = ggml_view_2d(tg, trw, trw_T, trw_half, trw->nb[1], 0);
                ggml_tensor* tb = ggml_view_2d(tg, trw, trw_T, trw_half, trw->nb[1], (size_t)trw_half * trw->nb[1]);
                ggml_tensor* ty = ggml_mul(tg, ggml_cont(tg, ta), ggml_sigmoid(tg, ggml_cont(tg, tb)));

                ggml_set_name(ty, "tenc_out");
                ggml_set_output(ty);
                ggml_cgraph* tgf = ggml_new_graph(tg);
                ggml_build_forward_expand(tgf, ty);
                ggml_gallocr_t talloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
                if (ggml_gallocr_alloc_graph(talloc, tgf)) {
                    ggml_backend_tensor_set(rw_in, xt_buf.data(), 0, xt_buf.size() * sizeof(float));
                    ggml_backend_graph_compute(ctx->backend, tgf);
                    xt_T = (int)ty->ne[0];
                    xt_C = (int)ty->ne[1];
                    xt_buf = read_tensor_f32(ty);
                }
                ggml_gallocr_free(talloc);
                ggml_free(tg);
            }

            if (tenc.empty) {
                // Merge point: inject time output into freq encoder
                inject_buf = xt_buf;
                has_inject = true;
            }

            // xt_buf is (C, T) channel-major — the reference layout.
            htd_capture(ctx, ("enc_time_" + std::to_string(idx)).c_str(), xt_buf.data(), xt_buf.size());

            if (htdemucs_debug()) {
                fprintf(stderr, "htdemucs: tenc[%d] output (%d, %d) empty=%d\n", idx, xt_C, xt_T, tenc.empty ? 1 : 0);
            }
        }

        // --- Per-layer ggml graph for freq encoder ---
        if (htdemucs_debug()) {
            // Print RSS before graph alloc
            FILE* sf = fopen("/proc/self/status", "r");
            if (sf) {
                char buf[256];
                while (fgets(buf, sizeof(buf), sf))
                    if (strncmp(buf, "VmRSS:", 6) == 0 || strncmp(buf, "VmPeak:", 7) == 0)
                        fprintf(stderr, "htdemucs: %s", buf);
                fclose(sf);
            }
        }
        if (!enc.conv_w) {
            encoder_ok = false;
            break;
        }
        bool enc_done = fused; // fused runs the whole freq chain later
        if (!fused && htdemucs_use_ggml() && enc.conv_w) {
            HTD_PROF(prof, "enc.ggml_graph");
            enc_done = htdemucs_enc_freq_ggml(ctx, enc, x_buf, x_C, x_Fq, x_T, stri, pad_val,
                                              has_inject ? &inject_buf : nullptr, has_inject ? x_C : 0, idx);
        }
        if (!enc_done) {
            // CPU Conv2d (avoids ggml im2col OOM on 8 GB VPS)
            int new_Fq = 0;
            {
                HTD_PROF(prof, "enc.conv2d");
                x_buf = cpu_conv2d_freq(x_buf, x_T, x_Fq, x_C, enc.conv_w, enc.conv_b, stri, pad_val, new_Fq);
            }
            x_C = (int)enc.conv_w->ne[3]; // OC
            x_Fq = new_Fq;
            // x_T unchanged

            // Dummy ggml context (for the non-Conv2d ops that follow — will be removed later)
            // Actually, let's do everything CPU-side now.
            (void)ctx; // suppress unused warning for the ggml path below
            // Inject time branch (at merge point only)
            if (has_inject) {
                for (int oc = 0; oc < x_C; oc++)
                    for (int fo = 0; fo < x_Fq; fo++)
                        for (int t = 0; t < x_T; t++)
                            x_buf[t + (size_t)fo * x_T + (size_t)oc * x_T * x_Fq] += inject_buf[t + (size_t)oc * x_T];
            }

            if (idx == 0)
                htd_capture(ctx, "enc0_conv", x_buf.data(), x_buf.size());

            if (!enc.empty) {
                // GELU (no GroupNorm for htdemucs — norm_starts=4, depth=4)
                cpu_gelu_inplace(x_buf);
                if (idx == 0)
                    htd_capture(ctx, "enc0_gelu", x_buf.data(), x_buf.size());
                // DConv: per-freq-band dilated conv residual
                // Python: y.permute(0,2,1,3).reshape(-1,C,T) → DConv → reshape back
                // = for each freq band: run DConv on (C, T) slice
                if (!enc.dconv.layers.empty() && htdemucs_fastconv()) {
                    // Batched across all frequency bands (one GEMM per conv).
                    HTD_PROF(prof, "enc.dconv");
                    cpu_dconv_bands(x_buf, x_Fq, x_C, x_T, enc.dconv);
                } else if (!enc.dconv.layers.empty()) {
                    HTD_PROF(prof, "enc.dconv");
                    for (int fq = 0; fq < x_Fq; fq++) {
                        // Extract the (C, T) slice for this frequency band,
                        // run the DConv stack on it, write it back.
                        std::vector<float> slice((size_t)x_T * x_C);
                        for (int c = 0; c < x_C; c++)
                            for (int t = 0; t < x_T; t++)
                                slice[t + (size_t)c * x_T] = x_buf[t + (size_t)fq * x_T + (size_t)c * x_T * x_Fq];

                        cpu_dconv_inplace(slice, x_C, x_T, enc.dconv);

                        for (int c = 0; c < x_C; c++)
                            for (int t = 0; t < x_T; t++)
                                x_buf[t + (size_t)fq * x_T + (size_t)c * x_T * x_Fq] = slice[t + (size_t)c * x_T];
                    }
                }
                if (idx == 0)
                    htd_capture(ctx, "enc0_dconv", x_buf.data(), x_buf.size());

                // Rewrite: 1x1 Conv2d → GLU
                if (enc.rewrite_w) {
                    HTD_PROF(prof, "enc.rewrite");
                    int rw_OC = 0;
                    auto rw_out = cpu_conv2d_1x1(x_buf, x_T * x_Fq, x_C, enc.rewrite_w, enc.rewrite_b, rw_OC);
                    x_buf = cpu_glu(rw_out, x_T * x_Fq, rw_OC);
                    x_C = rw_OC / 2;
                }
                if (idx == 0)
                    htd_capture(ctx, "enc0_rewrite", x_buf.data(), x_buf.size());
            }

            if (htdemucs_debug()) {
                fprintf(stderr, "htdemucs: enc[%d] output (%d, %d, %d) = %zu floats\n", idx, x_C, x_Fq, x_T,
                        x_buf.size());
            }
        }

        // Freq embedding (after layer 0 only)
        // Python: emb = freq_emb(arange(Fq)).t()[None,:,:,None].expand_as(x)
        //         x = x + freq_emb_scale * emb
        // Embedding: (n_freqs, C) → lookup frs 0..Fq-1 → (Fq, C) → t → (C, Fq)
        // Broadcast over T: x[t,fq,c] += scale * emb_w[fq, c]
        if (!fused && idx == 0 && m.freq_emb_w) {
            // nn.Embedding weight is (num_embeddings, embedding_dim) row-major
            // = ggml ne(embedding_dim, num_embeddings), so ne[0] is the
            // CHANNEL count and ne[1] is the frequency count — not the
            // other way round.
            int emb_C = (int)m.freq_emb_w->ne[0];       // embedding_dim  (48 channels)
            int emb_n_freqs = (int)m.freq_emb_w->ne[1]; // num_embeddings (512 freqs)
            // The embedding has emb_scale built into the weights (ScaledEmbedding)
            // but freq_emb_scale is an additional multiplier.
            std::vector<float> emb_data(emb_n_freqs * emb_C);
            {
                auto _rd = read_tensor_f32(m.freq_emb_w);
                memcpy(emb_data.data(), _rd.data(), std::min(emb_data.size(), _rd.size()) * sizeof(float));
            }
            float scale = hp.freq_emb_scale;
            // ScaledEmbedding weight is already scaled by `self.scale` (=10) in __init__,
            // but at forward time it multiplies by `self.scale` again. The GGUF stores
            // the raw (unscaled) weight. So effective embedding = weight * scale_emb.
            // In the converter we stored the raw weight. The ScaledEmbedding.forward() does
            // `self.embedding(x) * self.scale`. And then `x + freq_emb_scale * emb`.
            // For the SMOOTH variant: weights are cumsum → normalized. But stored as-is in GGUF.
            // The ScaledEmbedding stores weight/=scale in __init__, then forward *= scale.
            // So GGUF weight = nn.Embedding.weight.data / scale (after smooth processing).
            // Forward: output = GGUF_weight * scale_emb_init (=10) * freq_emb_scale (=0.2)
            float total_scale = 10.0f * scale; // scale_emb(10) * freq_emb_scale(0.2) = 2.0
            // x_buf layout: (T, Fq, C) flattened. Add emb[fq][c] * total_scale.
            int n_freq_to_use = std::min(x_Fq, emb_n_freqs);
            for (int fq = 0; fq < n_freq_to_use; fq++) {
                for (int c = 0; c < x_C && c < emb_C; c++) {
                    // Row-major (n_freqs, emb_C): stride by emb_C, not n_freqs.
                    float e = emb_data[(size_t)fq * emb_C + c] * total_scale;
                    for (int t = 0; t < x_T; t++) {
                        x_buf[t + (size_t)fq * x_T + (size_t)c * x_T * x_Fq] += e;
                    }
                }
            }
            if (htdemucs_debug()) {
                fprintf(stderr, "htdemucs: freq_emb added (%d freqs, %d ch, scale=%.2f)\n", n_freq_to_use, emb_C,
                        total_scale);
            }
        }


        if (!fused) // fused computes the freq encoder inside its graph
            htd_capture(ctx, ("enc_freq_" + std::to_string(idx)).c_str(), x_buf.data(), x_buf.size());

        // Save skip connections (fused keeps them as in-graph tensors)
        lengths_freq.push_back(x_T);
        if (!fused)
            saved_freq.push_back({x_buf, x_C, x_Fq, x_T});
        // Time branch: save non-empty tencoder outputs
        if (idx < (int)m.tencoder.size() && !m.tencoder[idx].empty) {
            saved_time.push_back({xt_buf, xt_C, 1, xt_T}); // length recorded above
        }

        // Update freq tracking
        if (freq) {
            freqs_cur = (freqs_cur <= hp.kernel_size) ? 1 : freqs_cur / hp.stride;
        }
    }

    fprintf(stderr, "htdemucs: encoder %s, output (%d, %d, %d)\n", encoder_ok ? "OK" : "FAILED", x_C, x_Fq, x_T);

    bool fused_done = false;
    if (fused) {
        HTD_PROF(prof, "fused.graph");
        x_buf = std::move(fused_spec_in);
        x_C = n_cac_ch;
        x_Fq = Fq;
        fused_done = htdemucs_fused_ggml(ctx, x_buf, x_C, x_Fq, x_T, xt_buf, xt_C, xt_T, fused_freq_emb);
        if (!fused_done)
            fprintf(stderr, "htdemucs: FUSED graph unavailable for this model — cannot continue\n");
    }

    if (!fused_done) {
        // On the fused path these buffers already hold the FINAL decoder output,
        // so capturing here would report a wrong value rather than an absent one.
        // The fused graph is validated on the output_* stages instead; per-stage
        // capture would need set_output taps inside the fused graph.
        htd_capture(ctx, "pre_transformer_z", x_buf.data(), x_buf.size());
        htd_capture(ctx, "pre_transformer_xt", xt_buf.data(), xt_buf.size());
    }

    // Step 7: CrossTransformer
    //
    // At the bottleneck: x = (x_T, x_Fq, x_C, 1) = (336, 8, 384, 1)
    //                    xt = (xt_T, xt_C) = (1344, 384)
    //
    // Python flow:
    //   1. channel_upsampler:   x (C=384) → x (C=512) via 1x1 Conv
    //   2. channel_upsampler_t: xt (C=384) → xt (C=512)
    //   3. Flatten+permute spec: (B,C,Fr,T) → (B, T*Fr, C)
    //   4. Add 2D sin pos emb + LayerNorm
    //   5. Permute time: (B,C,T) → (B, T, C), add 1D sin pos emb + LayerNorm
    //   6. 5 transformer layers (alternating self/cross attn)
    //   7. channel_downsampler:   x (C=512) → x (C=384)
    //   8. channel_downsampler_t: xt (C=512) → xt (C=384)
    //
    // For this first implementation, skip the transformer and just run
    // the channel up/downsamplers (identity if bottom_channels == transformer_channels).
    // The transformer layers will be added in Phase 3.
    //
    // Actually, let me implement the channel up/down at minimum, since they
    // are simple 1x1 convolutions and affect the output.

    if (!fused_done && m.channel_up_w && m.channel_down_w) {
        HTD_PROF(prof, "transformer[nested total]");
        // Channel upsample: 1x1 Conv on flattened freq branch
        // x: (x_T, x_Fq, x_C, 1) → flatten to (x_T*x_Fq, x_C) → conv → (x_T*x_Fq, bottom_ch)
        // Then reshape back to (x_T, x_Fq, bottom_ch, 1)
        int flat_len = x_T * x_Fq;
        int bot_ch = hp.bottom_channels; // 512

        // Freq branch: flatten spatial, conv1d upsample
        if (htdemucs_fastconv()) {
            HTD_PROF(prof, "chan.up/down(k1)");
            htd_k1_conv(x_buf, x_C, flat_len, m.channel_up_w, m.channel_up_b);
        } else {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* flat_in = ggml_new_tensor_2d(cg, GGML_TYPE_F32, flat_len, x_C);
            ggml_set_input(flat_in);
            ggml_tensor* up = ggml_conv_1d(cg, m.channel_up_w, flat_in, 1, 0, 1);
            if (m.channel_up_b) {
                ggml_tensor* ub = ggml_reshape_2d(cg, m.channel_up_b, 1, bot_ch);
                up = ggml_add(cg, up, ub);
            }
            ggml_set_output(up);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, up);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                // Reshape x_buf from (T, Fq, C) to (T*Fq, C) for the 1x1 conv
                // x_buf layout is already (T, Fq, C) flattened, which when viewed as
                // (T*Fq, C) is contiguous in the T*Fq dimension = correct for conv
                ggml_backend_tensor_set(flat_in, x_buf.data(), 0, x_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int up_len = (int)up->ne[0];
                int up_C = (int)up->ne[1];
                x_buf.resize((size_t)up_len * up_C);
                {
                    auto _rd = read_tensor_f32(up);
                    memcpy(x_buf.data(), _rd.data(), std::min(x_buf.size(), _rd.size()) * sizeof(float));
                }
                // Reshape back conceptually: (T*Fq, bot_ch) → (T, Fq, bot_ch)
                x_C = up_C;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        // Time branch: conv1d upsample
        if (htdemucs_fastconv()) {
            HTD_PROF(prof, "chan.up/down(k1)");
            htd_k1_conv(xt_buf, xt_C, xt_T, m.channel_up_t_w, m.channel_up_t_b);
        } else {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* xt_in2 = ggml_new_tensor_2d(cg, GGML_TYPE_F32, xt_T, xt_C);
            ggml_set_input(xt_in2);
            ggml_tensor* up_t = ggml_conv_1d(cg, m.channel_up_t_w, xt_in2, 1, 0, 1);
            if (m.channel_up_t_b) {
                ggml_tensor* utb = ggml_reshape_2d(cg, m.channel_up_t_b, 1, bot_ch);
                up_t = ggml_add(cg, up_t, utb);
            }
            ggml_set_output(up_t);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, up_t);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                ggml_backend_tensor_set(xt_in2, xt_buf.data(), 0, xt_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int up_T = (int)up_t->ne[0];
                int up_C = (int)up_t->ne[1];
                xt_buf.resize((size_t)up_T * up_C);
                {
                    auto _rd = read_tensor_f32(up_t);
                    memcpy(xt_buf.data(), _rd.data(), std::min(xt_buf.size(), _rd.size()) * sizeof(float));
                }
                xt_C = up_C;
                xt_T = up_T;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        fprintf(stderr, "htdemucs: channel upsample → freq (%d,%d,%d), time (%d,%d)\n", x_C, x_Fq, x_T, xt_C, xt_T);

        // Phase 3: CrossTransformer (5 layers alternating self/cross attention)
        //
        // At this point:
        //   x_buf: (T*Fq, C) = (2688, 512) — freq branch (flattened spatial)
        //   xt_buf: (xt_T, C) = (1344, 512) — time branch
        //
        // The Python transformer uses batch_first=True, so shapes are (B, seq, dim).
        // For ggml mul_mat, we need (dim, seq) — transpose from our (seq, dim) layout.
        // We'll transpose in-place before the transformer and transpose back after.

        int x_seq = x_T * x_Fq;       // 2688
        int dim = x_C;                // 512
        int xt_seq = xt_T;            // 1344
        int n_heads = hp.t_heads;     // 8
        int head_dim = dim / n_heads; // 64

        // NOTE: no transpose here. ggml_conv_1d's output has ne[0]=seq (the FAST
        // axis), so the channel upsampler already left x_buf/xt_buf in C memory as
        // [dim][seq] (index d*seq + s) — exactly what the attention code below
        // indexes. The previous code read that as (seq, dim) and "transposed" it,
        // scrambling both branches before every transformer layer.
        //
        // The freq sequence index is our native spatial order s = fr*T1 + t1
        // (Python flattens as (t1 fr) instead). Token order is irrelevant here:
        // self-attention is permutation-equivariant, cross-attention sums over the
        // whole key/value set, and FFN/LayerNorm are per-token — so as long as the
        // position embedding is applied to the correct (t1, fr) pair and the
        // decoder reads back the same order, the result is identical.

        // LayerNorm on both branches (norm_in / norm_in_t)
        // x: (dim, seq) — normalize over dim (ne[0])
        auto cpu_layernorm = [](float* data, int dim, int seq, const float* w, const float* b) {
            for (int s = 0; s < seq; s++) {
                // data layout: data[d * seq + s] for (dim, seq) C layout.
                // Normalize over dim for each seq position.
                float sum = 0;
                for (int d = 0; d < dim; d++)
                    sum += data[(size_t)d * seq + s];
                float mean = sum / dim;
                float var = 0;
                for (int d = 0; d < dim; d++) {
                    float v = data[(size_t)d * seq + s] - mean;
                    var += v * v;
                }
                float inv_std = 1.0f / sqrtf(var / dim + 1e-5f);
                for (int d = 0; d < dim; d++) {
                    size_t idx = (size_t)d * seq + s;
                    data[idx] = (data[idx] - mean) * inv_std * (w ? w[d] : 1.0f) + (b ? b[d] : 0.0f);
                }
            }
        };

        if (m.norm_in_w) {
            std::vector<float> nw(dim), nb(dim);
            {
                auto _rd = read_tensor_f32(m.norm_in_w);
                memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
            }
            if (m.norm_in_b) {
                auto _rd = read_tensor_f32(m.norm_in_b);
                memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
            }
            cpu_layernorm(x_buf.data(), dim, x_seq, nw.data(), m.norm_in_b ? nb.data() : nullptr);
        }
        if (m.norm_in_t_w) {
            std::vector<float> nw(dim), nb(dim);
            {
                auto _rd = read_tensor_f32(m.norm_in_t_w);
                memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
            }
            if (m.norm_in_t_b) {
                auto _rd = read_tensor_f32(m.norm_in_t_b);
                memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
            }
            cpu_layernorm(xt_buf.data(), dim, xt_seq, nw.data(), m.norm_in_t_b ? nb.data() : nullptr);
        }

        // Reference order is norm_in FIRST, then `x = x + weight_pos_embed * pos_emb`
        // (CrossTransformerEncoder.forward). The previous code added the position
        // embedding before the LayerNorm, which renormalised it away.
        // 2D sinusoidal position embedding for freq branch
        // Python: create_2d_sin_embedding(C, Fr, T1, max_period=10000)
        // pe[0:C/2:2, :, :] = sin(pos_w * div_term)  (width=T1 positions)
        // pe[1:C/2:2, :, :] = cos(pos_w * div_term)
        // pe[C/2::2, :, :]  = sin(pos_h * div_term)  (height=Fr positions)
        // pe[C/2+1::2, :, :] = cos(pos_h * div_term)
        // Then rearranged: (1, C, Fr, T1) → (1, T1*Fr, C) = (1, x_seq, dim)
        // x_buf is (dim, x_seq) after transpose. Add pe in same layout.
        {
            int Fr = x_Fq, T1 = x_T;
            int half_d = dim / 2;
            float max_period = hp.t_max_period;
            // Precompute div_term for half_d/2 entries
            std::vector<float> div_term(half_d / 2);
            for (int i = 0; i < half_d / 2; i++)
                div_term[i] = expf(-(float)(2 * i) * logf(max_period) / (float)half_d);

            for (int t = 0; t < T1; t++) {
                for (int fr = 0; fr < Fr; fr++) {
                    // Our spatial order is fr-major (see the note above), not
                    // Python's (t1 fr). phase_w still comes from t, phase_h from fr.
                    int s = fr * T1 + t;
                    for (int i = 0; i < half_d / 2; i++) {
                        float phase_w = (float)t * div_term[i];
                        float phase_h = (float)fr * div_term[i];
                        // Width dims: pe[2*i] = sin, pe[2*i+1] = cos
                        x_buf[(size_t)(2 * i) * x_seq + s] += hp.t_weight_pos_embed * sinf(phase_w);
                        x_buf[(size_t)(2 * i + 1) * x_seq + s] += hp.t_weight_pos_embed * cosf(phase_w);
                        // Height dims: pe[half_d + 2*i] = sin, pe[half_d + 2*i+1] = cos
                        x_buf[(size_t)(half_d + 2 * i) * x_seq + s] += hp.t_weight_pos_embed * sinf(phase_h);
                        x_buf[(size_t)(half_d + 2 * i + 1) * x_seq + s] += hp.t_weight_pos_embed * cosf(phase_h);
                    }
                }
            }
        }
        // 1D sinusoidal position embedding for time branch
        // Python: create_sin_embedding(T2, C, shift=0, max_period=10000)
        // pos[t] / (max_period^(d / (C/2-1))), then [cos, sin] concat
        {
            int half_d = dim / 2;
            float max_period = hp.t_max_period;
            for (int s = 0; s < xt_seq; s++) {
                for (int d = 0; d < half_d; d++) {
                    float phase = (float)s / powf(max_period, (float)d / (float)(half_d - 1));
                    xt_buf[(size_t)d * xt_seq + s] += hp.t_weight_pos_embed * cosf(phase);
                    xt_buf[(size_t)(half_d + d) * xt_seq + s] += hp.t_weight_pos_embed * sinf(phase);
                }
            }
        }


        htd_capture(ctx, "ct_in_z", x_buf.data(), x_buf.size());
        // (norm_in + positional embedding happen just above)
        htd_capture(ctx, "ct_in_xt", xt_buf.data(), xt_buf.size());

        // GGML graph port (opt-in). Runs every layer of both branches in one
        // graph on ctx->backend; falls through to the CPU/BLAS path below if the
        // graph cannot be built or allocated.
        bool ct_done = false;
        if (htdemucs_use_ggml()) {
            HTD_PROF(prof, "ct.ggml_graph");
            ct_done = htdemucs_transformer_ggml(ctx, x_buf, x_seq, xt_buf, xt_seq, dim, n_heads, hp.t_layers,
                                                hp.t_classic_parity);
        }

        // 5 transformer layers
        for (int li = 0; li < hp.t_layers && !ct_done; li++) {
            auto& layer_s = m.ct_layers[li];   // spec branch layer
            auto& layer_t = m.ct_layers_t[li]; // time branch layer

            // Helper: CPU multi-head attention
            // data layout: (dim, seq) in C memory. Weight layout: ggml tensor data.
            // Modifies x_data in-place.
            auto cpu_self_attn_layer = [&](float* x_data, int seq_len, const htdemucs_self_attn_layer& sa) {
                if (!sa.in_proj_w)
                    return;
                // norm_first=True: x = x + gamma1(SA(norm1(x))); x = x + gamma2(FFN(norm2(x)))

                // 1. norm1(x) → tmp
                std::vector<float> tmp(dim * seq_len);
                memcpy(tmp.data(), x_data, tmp.size() * sizeof(float));
                {
                    std::vector<float> nw(dim), nb(dim);
                    {
                        auto _rd = read_tensor_f32(sa.norm1_w);
                        memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                    }
                    if (sa.norm1_b) {
                        auto _rd = read_tensor_f32(sa.norm1_b);
                        memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                    }
                    cpu_layernorm(tmp.data(), dim, seq_len, nw.data(), sa.norm1_b ? nb.data() : nullptr);
                }

                // 2. QKV projection: in_proj_weight (3*dim, dim), in_proj_bias (3*dim)
                // qkv = tmp @ in_proj_weight^T + in_proj_bias
                // tmp is (dim, seq), weight is (dim, 3*dim) in ggml ne → transposed: (3*dim, dim)
                int out_dim = 3 * dim;
                std::vector<float> w_data = read_tensor_f32(sa.in_proj_w);
                // w_data ggml layout: ne[0]=dim, ne[1]=3*dim → w[out_d][in_d] = w_data[in_d * out_dim + out_d]
                // Wait, ggml stores row-major with ne[0] as fast. So w_data[i] accesses
                // element at (i % ne[0], i / ne[0]) = (in_d, out_d). So w[out_d][in_d] = w_data[out_d * dim + in_d].
                // No wait — for a 2D tensor (ne[0], ne[1]), element [i0, i1] = data[i1 * ne[0] + i0].
                // So w[i0, i1] = w_data[i1 * dim + i0] where i0 ∈ [0,dim), i1 ∈ [0,3*dim).
                // matmul: qkv[o, s] = sum_d w[d, o] * tmp[d, s] for d in [0, dim)
                // = sum_d w_data[o * dim + d] * tmp[d * seq_len + s]

                // qkv[o,s] = sum_d W[o,d] * tmp[d,s]  ->  (3dim,dim) x (dim,seq)
                std::vector<float> qkv((size_t)out_dim * seq_len, 0.0f);
                htd_gemm(out_dim, seq_len, dim, w_data.data(), tmp.data(), qkv.data());
                // Add bias
                if (sa.in_proj_b) {
                    std::vector<float> bias(out_dim);
                    {
                        auto _rd = read_tensor_f32(sa.in_proj_b);
                        memcpy(bias.data(), _rd.data(), std::min(bias.size(), _rd.size()) * sizeof(float));
                    }
                    for (int o = 0; o < out_dim; o++)
                        for (int s = 0; s < seq_len; s++)
                            qkv[(size_t)o * seq_len + s] += bias[o];
                }

                // 3. Split QKV into Q, K, V (each dim × seq_len)
                float* Q = qkv.data();
                float* K = qkv.data() + (size_t)dim * seq_len;
                float* V = qkv.data() + (size_t)2 * dim * seq_len;

                // 4. Multi-head attention: for each head, compute attn scores and weighted V
                float scale = 1.0f / sqrtf((float)head_dim);
                std::vector<float> attn_out(dim * seq_len, 0.0f);

                std::vector<float> scores((size_t)seq_len * seq_len);
                for (int h = 0; h < n_heads; h++) {
                    int hoff = h * head_dim;
                    // Q/K/V are (dim, seq) row-major, so a head slice is a
                    // (head_dim, seq) block with row stride seq_len. scores =
                    // Q_h^T * K_h -> (seq, seq); sgemm reads the slices in place
                    // via lda/ldb, no packing needed.
#if defined(HAVE_ACCELERATE)
                    if (htdemucs_use_blas()) {
                        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, seq_len, seq_len, head_dim, scale,
                                    Q + (size_t)hoff * seq_len, seq_len, K + (size_t)hoff * seq_len, seq_len, 0.0f,
                                    scores.data(), seq_len);
                    } else
#endif
                    {
                        for (int s1 = 0; s1 < seq_len; s1++)
                            for (int s2 = 0; s2 < seq_len; s2++) {
                                float dot = 0;
                                for (int hd = 0; hd < head_dim; hd++)
                                    dot +=
                                        Q[(size_t)(hoff + hd) * seq_len + s1] * K[(size_t)(hoff + hd) * seq_len + s2];
                                scores[(size_t)s1 * seq_len + s2] = dot * scale;
                            }
                    }
                    for (int s1 = 0; s1 < seq_len; s1++) {
                        // Softmax over s2 for each s1
                        float max_s = -1e30f;
                        for (int s2 = 0; s2 < seq_len; s2++)
                            max_s = std::max(max_s, scores[(size_t)s1 * seq_len + s2]);
                        float sum_exp = 0;
                        for (int s2 = 0; s2 < seq_len; s2++) {
                            scores[(size_t)s1 * seq_len + s2] = expf(scores[(size_t)s1 * seq_len + s2] - max_s);
                            sum_exp += scores[(size_t)s1 * seq_len + s2];
                        }
                        for (int s2 = 0; s2 < seq_len; s2++)
                            scores[(size_t)s1 * seq_len + s2] /= sum_exp;
                    }
                    // attn_out_h[hd, s1] = sum_s2 V_h[hd, s2] * scores[s1, s2]
                    //   = V_h (head_dim, seq) * scores^T (seq, seq)
#if defined(HAVE_ACCELERATE)
                    if (htdemucs_use_blas()) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, head_dim, seq_len, seq_len, 1.0f,
                                    V + (size_t)hoff * seq_len, seq_len, scores.data(), seq_len, 0.0f,
                                    attn_out.data() + (size_t)hoff * seq_len, seq_len);
                    } else
#endif
                    {
                        for (int s1 = 0; s1 < seq_len; s1++)
                            for (int hd = 0; hd < head_dim; hd++) {
                                float sum = 0;
                                for (int s2 = 0; s2 < seq_len; s2++)
                                    sum += scores[(size_t)s1 * seq_len + s2] * V[(size_t)(hoff + hd) * seq_len + s2];
                                attn_out[(size_t)(hoff + hd) * seq_len + s1] = sum;
                            }
                    }
                }

                // 5. Output projection: out_proj_weight (dim, dim)
                {
                    std::vector<float> ow = read_tensor_f32(sa.out_proj_w);
                    std::vector<float> proj((size_t)dim * seq_len, 0.0f);
                    htd_gemm(dim, seq_len, dim, ow.data(), attn_out.data(), proj.data());
                    if (sa.out_proj_b) {
                        std::vector<float> ob(dim);
                        {
                            auto _rd = read_tensor_f32(sa.out_proj_b);
                            memcpy(ob.data(), _rd.data(), std::min(ob.size(), _rd.size()) * sizeof(float));
                        }
                        for (int o = 0; o < dim; o++)
                            for (int s = 0; s < seq_len; s++)
                                proj[(size_t)o * seq_len + s] += ob[o];
                    }
                    attn_out = std::move(proj);
                }

                // 6. LayerScale gamma1 + residual
                if (sa.gamma1_scale) {
                    std::vector<float> gs(dim);
                    {
                        auto _rd = read_tensor_f32(sa.gamma1_scale);
                        memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                    }
                    for (int d = 0; d < dim; d++)
                        for (int s = 0; s < seq_len; s++)
                            attn_out[(size_t)d * seq_len + s] *= gs[d];
                }
                for (size_t i = 0; i < (size_t)dim * seq_len; i++)
                    x_data[i] += attn_out[i];

                // 7. FFN: x = x + gamma2(FFN(norm2(x)))
                memcpy(tmp.data(), x_data, tmp.size() * sizeof(float));
                {
                    std::vector<float> nw(dim), nb(dim);
                    {
                        auto _rd = read_tensor_f32(sa.norm2_w);
                        memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                    }
                    if (sa.norm2_b) {
                        auto _rd = read_tensor_f32(sa.norm2_b);
                        memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                    }
                    cpu_layernorm(tmp.data(), dim, seq_len, nw.data(), sa.norm2_b ? nb.data() : nullptr);
                }
                // linear1: (dim → hidden), GELU, linear2: (hidden → dim)
                int hidden = (int)sa.linear1_w->ne[1]; // ne = (dim, hidden)
                {
                    std::vector<float> w1 = read_tensor_f32(sa.linear1_w);
                    std::vector<float> b1(hidden);
                    if (sa.linear1_b) {
                        auto _rd = read_tensor_f32(sa.linear1_b);
                        memcpy(b1.data(), _rd.data(), std::min(b1.size(), _rd.size()) * sizeof(float));
                    }
                    std::vector<float> h((size_t)hidden * seq_len, 0.0f);
                    htd_gemm(hidden, seq_len, dim, w1.data(), tmp.data(), h.data());
                    if (sa.linear1_b)
                        for (int o = 0; o < hidden; o++)
                            for (int s = 0; s < seq_len; s++)
                                h[(size_t)o * seq_len + s] += b1[o];
                    // GELU
                    for (size_t i = 0; i < h.size(); i++) {
                        float v = h[i];
                        h[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                    }
                    // linear2
                    std::vector<float> w2 = read_tensor_f32(sa.linear2_w);
                    std::vector<float> b2(dim);
                    if (sa.linear2_b) {
                        auto _rd = read_tensor_f32(sa.linear2_b);
                        memcpy(b2.data(), _rd.data(), std::min(b2.size(), _rd.size()) * sizeof(float));
                    }
                    std::vector<float> ffn_out((size_t)dim * seq_len, 0.0f);
                    htd_gemm(dim, seq_len, hidden, w2.data(), h.data(), ffn_out.data());
                    if (sa.linear2_b)
                        for (int o = 0; o < dim; o++)
                            for (int s = 0; s < seq_len; s++)
                                ffn_out[(size_t)o * seq_len + s] += b2[o];
                    // gamma2 + residual
                    if (sa.gamma2_scale) {
                        std::vector<float> gs(dim);
                        {
                            auto _rd = read_tensor_f32(sa.gamma2_scale);
                            memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                        }
                        for (int d = 0; d < dim; d++)
                            for (int s = 0; s < seq_len; s++)
                                ffn_out[(size_t)d * seq_len + s] *= gs[d];
                    }
                    for (size_t i = 0; i < (size_t)dim * seq_len; i++)
                        x_data[i] += ffn_out[i];
                }

                // norm_out (if present)
                if (sa.norm_out_w) {
                    std::vector<float> nw(dim), nb(dim);
                    {
                        auto _rd = read_tensor_f32(sa.norm_out_w);
                        memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                    }
                    if (sa.norm_out_b) {
                        auto _rd = read_tensor_f32(sa.norm_out_b);
                        memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                    }
                    // norm_out is MyGroupNorm(num_groups=1) — ONE mean/std over all
                    // channels AND all tokens jointly — not a per-token LayerNorm.
                    // Using cpu_layernorm here left ~1% error in every layer, which
                    // layer 4 then amplified enormously (its pre-norm activations are
                    // outlier-dominated, norm ~85k -> ~78).
                    cpu_group_norm1_inplace(x_data, dim, seq_len, nw.data(), sa.norm_out_b ? nb.data() : nullptr);
                }
            };

            if (!layer_s.is_cross) {
                // Self-attention: each branch independently
                HTD_PROF(prof, "ct.self_attn");
                cpu_self_attn_layer(x_buf.data(), x_seq, layer_s.self_attn);
                cpu_self_attn_layer(xt_buf.data(), xt_seq, layer_t.self_attn);
            } else {
                // Cross-attention: spec attends to time, time attends to spec.
                // Spec layer: Q=norm1(x), K=V=norm2(xt) → CA → gamma1 + residual → FFN
                // Time layer: Q=norm1(xt), K=V=norm2(old_x) → CA → gamma1 + residual → FFN
                auto cpu_cross_attn_layer = [&](float* q_data, int q_seq, const float* k_data, int k_seq,
                                                const htdemucs_cross_attn_layer& ca) {
                    if (!ca.cross_attn_in_proj_w)
                        return;

                    // 1. norm1(q) → q_normed, norm2(k) → k_normed
                    std::vector<float> q_normed(dim * q_seq), k_normed(dim * k_seq);
                    memcpy(q_normed.data(), q_data, q_normed.size() * sizeof(float));
                    memcpy(k_normed.data(), k_data, k_normed.size() * sizeof(float));
                    {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm1_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm1_b) {
                            auto _rd = read_tensor_f32(ca.norm1_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(q_normed.data(), dim, q_seq, nw.data(), ca.norm1_b ? nb.data() : nullptr);
                    }
                    {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm2_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm2_b) {
                            auto _rd = read_tensor_f32(ca.norm2_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(k_normed.data(), dim, k_seq, nw.data(), ca.norm2_b ? nb.data() : nullptr);
                    }

                    // 2. QKV projection from in_proj_weight (3*dim, dim)
                    // Q from q_normed, K and V from k_normed
                    int out3 = 3 * dim;
                    std::vector<float> w_data = read_tensor_f32(ca.cross_attn_in_proj_w);
                    std::vector<float> bias(out3, 0.0f);
                    if (ca.cross_attn_in_proj_b) {
                        auto _rd = read_tensor_f32(ca.cross_attn_in_proj_b);
                        memcpy(bias.data(), _rd.data(), std::min(bias.size(), _rd.size()) * sizeof(float));
                    }

                    // Q = W_q @ q_normed + b_q  (first dim rows of weight)
                    std::vector<float> Q((size_t)dim * q_seq, 0.0f);
                    htd_gemm(dim, q_seq, dim, w_data.data(), q_normed.data(), Q.data());
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < q_seq; s++)
                            Q[(size_t)o * q_seq + s] += bias[o];
                    // K = W_k @ k_normed + b_k  (second dim rows)
                    std::vector<float> K((size_t)dim * k_seq, 0.0f);
                    htd_gemm(dim, k_seq, dim, w_data.data() + (size_t)dim * dim, k_normed.data(), K.data());
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < k_seq; s++)
                            K[(size_t)o * k_seq + s] += bias[dim + o];
                    // V = W_v @ k_normed + b_v  (third dim rows)
                    std::vector<float> V((size_t)dim * k_seq, 0.0f);
                    htd_gemm(dim, k_seq, dim, w_data.data() + (size_t)2 * dim * dim, k_normed.data(), V.data());
                    for (int o = 0; o < dim; o++)
                        for (int s = 0; s < k_seq; s++)
                            V[(size_t)o * k_seq + s] += bias[2 * dim + o];

                    // 3. Multi-head cross-attention
                    float scale = 1.0f / sqrtf((float)head_dim);
                    std::vector<float> attn_out(dim * q_seq, 0.0f);
                    for (int h = 0; h < n_heads; h++) {
                        int hoff = h * head_dim;
                        // scores[q_s, k_s] = Q_h^T * K_h for this head
                        std::vector<float> scores((size_t)q_seq * k_seq);
#if defined(HAVE_ACCELERATE)
                        if (htdemucs_use_blas()) {
                            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, q_seq, k_seq, head_dim, scale,
                                        Q.data() + (size_t)hoff * q_seq, q_seq, K.data() + (size_t)hoff * k_seq, k_seq,
                                        0.0f, scores.data(), k_seq);
                        } else
#endif
                        {
                            for (int qs = 0; qs < q_seq; qs++)
                                for (int ks = 0; ks < k_seq; ks++) {
                                    float dot = 0;
                                    for (int hd = 0; hd < head_dim; hd++)
                                        dot +=
                                            Q[(size_t)(hoff + hd) * q_seq + qs] * K[(size_t)(hoff + hd) * k_seq + ks];
                                    scores[(size_t)qs * k_seq + ks] = dot * scale;
                                }
                        }
                        for (int qs = 0; qs < q_seq; qs++) {
                            // softmax over ks
                            float mx = -1e30f;
                            for (int ks = 0; ks < k_seq; ks++)
                                mx = std::max(mx, scores[(size_t)qs * k_seq + ks]);
                            float se = 0;
                            for (int ks = 0; ks < k_seq; ks++) {
                                scores[(size_t)qs * k_seq + ks] = expf(scores[(size_t)qs * k_seq + ks] - mx);
                                se += scores[(size_t)qs * k_seq + ks];
                            }
                            for (int ks = 0; ks < k_seq; ks++)
                                scores[(size_t)qs * k_seq + ks] /= se;
                        }
                        // weighted sum of V: V_h (head_dim, k_seq) * scores^T (k_seq, q_seq)
#if defined(HAVE_ACCELERATE)
                        if (htdemucs_use_blas()) {
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, head_dim, q_seq, k_seq, 1.0f,
                                        V.data() + (size_t)hoff * k_seq, k_seq, scores.data(), k_seq, 0.0f,
                                        attn_out.data() + (size_t)hoff * q_seq, q_seq);
                        } else
#endif
                        {
                            for (int qs = 0; qs < q_seq; qs++)
                                for (int hd = 0; hd < head_dim; hd++) {
                                    float sum = 0;
                                    for (int ks = 0; ks < k_seq; ks++)
                                        sum += scores[(size_t)qs * k_seq + ks] * V[(size_t)(hoff + hd) * k_seq + ks];
                                    attn_out[(size_t)(hoff + hd) * q_seq + qs] = sum;
                                }
                        }
                    }

                    // 4. Output projection
                    {
                        std::vector<float> ow = read_tensor_f32(ca.cross_attn_out_proj_w);
                        std::vector<float> proj((size_t)dim * q_seq, 0.0f);
                        htd_gemm(dim, q_seq, dim, ow.data(), attn_out.data(), proj.data());
                        if (ca.cross_attn_out_proj_b) {
                            std::vector<float> ob(dim);
                            {
                                auto _rd = read_tensor_f32(ca.cross_attn_out_proj_b);
                                memcpy(ob.data(), _rd.data(), std::min(ob.size(), _rd.size()) * sizeof(float));
                            }
                            for (int o = 0; o < dim; o++)
                                for (int s = 0; s < q_seq; s++)
                                    proj[(size_t)o * q_seq + s] += ob[o];
                        }
                        attn_out = std::move(proj);
                    }

                    // 5. gamma1 + residual
                    if (ca.gamma1_scale) {
                        std::vector<float> gs(dim);
                        {
                            auto _rd = read_tensor_f32(ca.gamma1_scale);
                            memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                        }
                        for (int d = 0; d < dim; d++)
                            for (int s = 0; s < q_seq; s++)
                                attn_out[(size_t)d * q_seq + s] *= gs[d];
                    }
                    for (size_t i = 0; i < (size_t)dim * q_seq; i++)
                        q_data[i] += attn_out[i];

                    // 6. FFN: norm3 → linear1 → GELU → linear2 → gamma2 + residual
                    std::vector<float> tmp(dim * q_seq);
                    memcpy(tmp.data(), q_data, tmp.size() * sizeof(float));
                    {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm3_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm3_b) {
                            auto _rd = read_tensor_f32(ca.norm3_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        cpu_layernorm(tmp.data(), dim, q_seq, nw.data(), ca.norm3_b ? nb.data() : nullptr);
                    }
                    int hidden = (int)ca.linear1_w->ne[1];
                    {
                        std::vector<float> w1 = read_tensor_f32(ca.linear1_w);
                        std::vector<float> b1(hidden);
                        if (ca.linear1_b) {
                            auto _rd = read_tensor_f32(ca.linear1_b);
                            memcpy(b1.data(), _rd.data(), std::min(b1.size(), _rd.size()) * sizeof(float));
                        }
                        std::vector<float> hbuf((size_t)hidden * q_seq, 0.0f);

                        htd_gemm(hidden, q_seq, dim, w1.data(), tmp.data(), hbuf.data());

                        if (ca.linear1_b)

                            for (int o = 0; o < hidden; o++)

                                for (int s = 0; s < q_seq; s++)

                                    hbuf[(size_t)o * q_seq + s] += b1[o];
                        for (size_t i = 0; i < hbuf.size(); i++) {
                            float v = hbuf[i];
                            hbuf[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                        }
                        std::vector<float> w2 = read_tensor_f32(ca.linear2_w);
                        std::vector<float> b2(dim);
                        if (ca.linear2_b) {
                            auto _rd = read_tensor_f32(ca.linear2_b);
                            memcpy(b2.data(), _rd.data(), std::min(b2.size(), _rd.size()) * sizeof(float));
                        }
                        std::vector<float> ffn_out((size_t)dim * q_seq, 0.0f);

                        htd_gemm(dim, q_seq, hidden, w2.data(), hbuf.data(), ffn_out.data());

                        if (ca.linear2_b)

                            for (int o = 0; o < dim; o++)

                                for (int s = 0; s < q_seq; s++)

                                    ffn_out[(size_t)o * q_seq + s] += b2[o];
                        if (ca.gamma2_scale) {
                            std::vector<float> gs(dim);
                            {
                                auto _rd = read_tensor_f32(ca.gamma2_scale);
                                memcpy(gs.data(), _rd.data(), std::min(gs.size(), _rd.size()) * sizeof(float));
                            }
                            for (int d = 0; d < dim; d++)
                                for (int s = 0; s < q_seq; s++)
                                    ffn_out[(size_t)d * q_seq + s] *= gs[d];
                        }
                        for (size_t i = 0; i < (size_t)dim * q_seq; i++)
                            q_data[i] += ffn_out[i];
                    }

                    // norm_out
                    if (ca.norm_out_w) {
                        std::vector<float> nw(dim), nb(dim);
                        {
                            auto _rd = read_tensor_f32(ca.norm_out_w);
                            memcpy(nw.data(), _rd.data(), std::min(nw.size(), _rd.size()) * sizeof(float));
                        }
                        if (ca.norm_out_b) {
                            auto _rd = read_tensor_f32(ca.norm_out_b);
                            memcpy(nb.data(), _rd.data(), std::min(nb.size(), _rd.size()) * sizeof(float));
                        }
                        // norm_out is GroupNorm(1) over all channels+tokens (see above).
                        cpu_group_norm1_inplace(q_data, dim, q_seq, nw.data(), ca.norm_out_b ? nb.data() : nullptr);
                    }
                };

                // Cross-attention: spec layer attends to time, time layer attends to spec
                // Save old_x for the time layer's cross-attention (time attends to old spec)
                std::vector<float> old_x(x_buf);
                HTD_PROF(prof, "ct.cross_attn");
                cpu_cross_attn_layer(x_buf.data(), x_seq, xt_buf.data(), xt_seq, layer_s.cross_attn);
                cpu_cross_attn_layer(xt_buf.data(), xt_seq, old_x.data(), x_seq, layer_t.cross_attn);
            }

            htd_capture(ctx, ("ct_l" + std::to_string(li) + "_z").c_str(), x_buf.data(), x_buf.size());
            htd_capture(ctx, ("ct_l" + std::to_string(li) + "_xt").c_str(), xt_buf.data(), xt_buf.size());
        }

        if (htdemucs_debug()) {
            float mx = 0, mxt = 0;
            for (size_t i = 0; i < x_buf.size(); i++) {
                float a = fabsf(x_buf[i]);
                if (a > mx)
                    mx = a;
            }
            for (size_t i = 0; i < xt_buf.size(); i++) {
                float a = fabsf(xt_buf[i]);
                if (a > mxt)
                    mxt = a;
            }
            fprintf(stderr, "htdemucs: post-transformer freq max=%.2f, time max=%.2f\n", mx, mxt);
        }

        // No transpose back either — see the note above; the buffers stay
        // [dim][seq], which is what the channel downsampler expects.

        // Channel downsample: 1x1 Conv back to transformer_channels
        // Freq branch
        if (htdemucs_fastconv()) {
            HTD_PROF(prof, "chan.up/down(k1)");
            htd_k1_conv(x_buf, x_C, x_T * x_Fq, m.channel_down_w, m.channel_down_b);
        } else {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* flat_in = ggml_new_tensor_2d(cg, GGML_TYPE_F32, x_T * x_Fq, x_C);
            ggml_set_input(flat_in);
            ggml_tensor* dn = ggml_conv_1d(cg, m.channel_down_w, flat_in, 1, 0, 1);
            if (m.channel_down_b) {
                int dn_C = (int)m.channel_down_w->ne[2];
                ggml_tensor* db = ggml_reshape_2d(cg, m.channel_down_b, 1, dn_C);
                dn = ggml_add(cg, dn, db);
            }
            ggml_set_output(dn);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, dn);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                ggml_backend_tensor_set(flat_in, x_buf.data(), 0, x_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int dn_len = (int)dn->ne[0];
                int dn_C = (int)dn->ne[1];
                x_buf.resize((size_t)dn_len * dn_C);
                {
                    auto _rd = read_tensor_f32(dn);
                    memcpy(x_buf.data(), _rd.data(), std::min(x_buf.size(), _rd.size()) * sizeof(float));
                }
                x_C = dn_C;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        // Time branch downsample
        if (htdemucs_fastconv()) {
            HTD_PROF(prof, "chan.up/down(k1)");
            htd_k1_conv(xt_buf, xt_C, xt_T, m.channel_down_t_w, m.channel_down_t_b);
        } else {
            size_t g_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
            ggml_init_params gp = {g_size, nullptr, true};
            ggml_context* cg = ggml_init(gp);

            ggml_tensor* xt_in2 = ggml_new_tensor_2d(cg, GGML_TYPE_F32, xt_T, xt_C);
            ggml_set_input(xt_in2);
            ggml_tensor* dn_t = ggml_conv_1d(cg, m.channel_down_t_w, xt_in2, 1, 0, 1);
            if (m.channel_down_t_b) {
                int dn_C = (int)m.channel_down_t_w->ne[2];
                ggml_tensor* dtb = ggml_reshape_2d(cg, m.channel_down_t_b, 1, dn_C);
                dn_t = ggml_add(cg, dn_t, dtb);
            }
            ggml_set_output(dn_t);

            ggml_cgraph* gf = ggml_new_graph(cg);
            ggml_build_forward_expand(gf, dn_t);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (ggml_gallocr_alloc_graph(al, gf)) {
                ggml_backend_tensor_set(xt_in2, xt_buf.data(), 0, xt_buf.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, gf);
                int dn_T = (int)dn_t->ne[0];
                int dn_C = (int)dn_t->ne[1];
                xt_buf.resize((size_t)dn_T * dn_C);
                {
                    auto _rd = read_tensor_f32(dn_t);
                    memcpy(xt_buf.data(), _rd.data(), std::min(xt_buf.size(), _rd.size()) * sizeof(float));
                }
                xt_C = dn_C;
                xt_T = dn_T;
            }
            ggml_gallocr_free(al);
            ggml_free(cg);
        }

        fprintf(stderr, "htdemucs: channel downsample → freq (%d,%d,%d), time (%d,%d)\n", x_C, x_Fq, x_T, xt_C, xt_T);
    } else {
        fprintf(stderr, "htdemucs: no channel up/downsamplers (bottom_channels=0?)\n");
    }

    if (!fused_done) {
        htd_capture(ctx, "post_transformer_z", x_buf.data(), x_buf.size());
        htd_capture(ctx, "post_transformer_xt", xt_buf.data(), xt_buf.size());
    }

    // Step 8: Decoder (reverse of encoder, with skip connections)
    // Decoder index 0 is the innermost (smallest spatial dims), matching
    // encoder index depth-1. The freq dims grow back: 8→32→128→512→2048.
    int dec_strides[] = {4, 4, 4, 4}; // same stride pattern as encoder (all freq layers)
    for (int idx = 0; idx < hp.depth && encoder_ok; idx++) {
        auto& dec = m.decoder[idx];
        if (!dec.conv_tr_w)
            break;

        // Stride for this decoder layer (reverse of encoder)
        int stri = dec_strides[idx];                 // TODO: derive properly from encoder
        int pad_val = (int)dec.conv_tr_w->ne[1] / 4; // K/4

        // Pop skip from end (LIFO order). Fused keeps skips as in-graph tensors,
        // so saved_freq is empty on that path.
        saved_activation skip_f;
        if (!saved_freq.empty()) {
            skip_f = saved_freq.back();
            saved_freq.pop_back();
        }

        // Full in-graph decoder layer (opt-in). Produces both the layer output
        // and `pre`, which the time decoder consumes.
        std::vector<float> pre_buf_g;
        int pre_C_g = 0, pre_Fq_g = 0, pre_T_g = x_T;
        // Shared by both paths; the time decoder reads these below.
        std::vector<float> pre_buf;
        int pre_T = x_T, pre_Fq = 0, pre_C = 0;
        bool dec_done = fused_done; // fused already produced the decoder output
        if (!fused_done && htdemucs_use_ggml() && dec.conv_tr_w) {
            HTD_PROF(prof, "dec.ggml_graph");
            dec_done = htdemucs_dec_freq_ggml(ctx, dec, x_buf, skip_f.data, x_C, x_Fq, x_T, stri, pad_val,
                                              idx == hp.depth - 1, &pre_buf_g, &pre_C_g, &pre_Fq_g);
        }
        if (dec_done && !fused_done) {
            htd_capture(ctx, ("dec_freq_" + std::to_string(idx)).c_str(), x_buf.data(), x_buf.size());
        }

        if (!dec_done) {
            // Build decoder layer graph
            size_t d_ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
            ggml_init_params dgp = {d_ctx_size, nullptr, true};
            ggml_context* dg = ggml_init(dgp);

            bool is_freq = dec.freq;
            bool is_last = (idx == hp.depth - 1);

            // Input tensor: (T, Fq, C, 1) for freq, (T, C) for time
            ggml_tensor* dx_in = nullptr;
            ggml_tensor* skip_in = nullptr;

            if (is_freq) {
                dx_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, x_T, x_Fq, x_C, 1);
                skip_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, skip_f.T, skip_f.Fq, skip_f.C, 1);
            } else {
                // TODO: handle non-freq decoder layers
                dx_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, x_T, x_Fq, x_C, 1);
                skip_in = ggml_new_tensor_4d(dg, GGML_TYPE_F32, skip_f.T, skip_f.Fq, skip_f.C, 1);
            }
            ggml_set_name(dx_in, "dec_in");
            ggml_set_input(dx_in);
            ggml_set_name(skip_in, "dec_skip");
            ggml_set_input(skip_in);

            ggml_tensor* dy = dx_in;

            if (!dec.empty) {
                // x = x + skip
                dy = ggml_add(dg, dy, skip_in);

                // Rewrite: Conv2d(C→2*C, [1+2*ctx, 1+2*ctx], stride=1, pad=[ctx,ctx]) → GroupNorm → GLU
                if (dec.rewrite_w) {
                    int ctx_pad = hp.context; // context=1 for decoder
                    ggml_tensor* drw = ggml_conv_2d(dg, dec.rewrite_w, dy, 1, 1, ctx_pad, ctx_pad, 1, 1);
                    if (dec.rewrite_b) {
                        ggml_tensor* drwb = ggml_reshape_4d(dg, dec.rewrite_b, 1, 1, (int)dec.rewrite_b->ne[0], 1);
                        drw = ggml_add(dg, drw, drwb);
                    }
                    if (dec.norm1_w) {
                        drw = ggml_group_norm(dg, drw, 4, 1e-5f);
                        ggml_tensor* w4d = ggml_reshape_4d(dg, dec.norm1_w, 1, 1, (int)dec.norm1_w->ne[0], 1);
                        drw = ggml_mul(dg, drw, w4d);
                        if (dec.norm1_b) {
                            ggml_tensor* b4d = ggml_reshape_4d(dg, dec.norm1_b, 1, 1, (int)dec.norm1_b->ne[0], 1);
                            drw = ggml_add(dg, drw, b4d);
                        }
                    }
                    // GLU
                    int drw_C = (int)drw->ne[2], drw_half = drw_C / 2;
                    int drw_Fq = (int)drw->ne[1], drw_T = (int)drw->ne[0];
                    size_t drw_ch_stride = drw->nb[2];
                    ggml_tensor* da =
                        ggml_view_4d(dg, drw, drw_T, drw_Fq, drw_half, 1, drw->nb[1], drw->nb[2], drw->nb[3], 0);
                    ggml_tensor* db = ggml_view_4d(dg, drw, drw_T, drw_Fq, drw_half, 1, drw->nb[1], drw->nb[2],
                                                   drw->nb[3], (size_t)drw_half * drw_ch_stride);
                    dy = ggml_mul(dg, ggml_cont(dg, da), ggml_sigmoid(dg, ggml_cont(dg, db)));
                }
            } else {
                // Empty decoder: dy stays as input (no skip add, no rewrite)
            }

            // ConvTranspose2d on freq axis (kernel [K, 1], stride [S, 1]).
            // ggml lacks conv_transpose_2d with asymmetric stride, so we do this
            // CPU-side: run the ggml graph up to here, read back dy, apply
            // ConvTranspose1d on the freq axis per-time-frame, write new dy.
            //
            // This is the approach taken by the first-pass parity build.
            // Phase 5 (GPU perf) will fuse this into a single graph.

            // First: finalize the pre-convtranspose ggml graph
            ggml_set_name(dy, "dec_pre_ct");
            ggml_set_output(dy);
            ggml_cgraph* dgf = ggml_new_graph(dg);
            ggml_build_forward_expand(dgf, dy);
            ggml_gallocr_t dalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
            if (!ggml_gallocr_alloc_graph(dalloc, dgf)) {
                fprintf(stderr, "htdemucs: dec[%d] pre-CT graph alloc failed\n", idx);
                ggml_gallocr_free(dalloc);
                ggml_free(dg);
                break;
            }
            {
                HTD_PROF(prof, "dec.rewrite(ggml)");
                ggml_backend_tensor_set(dx_in, x_buf.data(), 0, x_buf.size() * sizeof(float));
                ggml_backend_tensor_set(skip_in, skip_f.data.data(), 0, skip_f.data.size() * sizeof(float));
                ggml_backend_graph_compute(ctx->backend, dgf);
            }

            // Read pre-ConvTranspose result
            pre_T = (int)dy->ne[0];
            pre_Fq = (int)dy->ne[1];
            pre_C = (int)dy->ne[2];
            size_t pre_n = (size_t)pre_T * pre_Fq * pre_C;
            pre_buf.assign(pre_n, 0.0f);
            {
                auto _rd = read_tensor_f32(dy);
                memcpy(pre_buf.data(), _rd.data(), std::min(pre_buf.size(), _rd.size()) * sizeof(float));
            }
            ggml_gallocr_free(dalloc);
            ggml_free(dg);

            // DConv, which the decoder was skipping entirely. HDecLayer.forward is
            //   x+skip -> GLU(norm1(rewrite)) -> dconv -> conv_tr
            // and every decoder layer has a real DConv. Same per-frequency-band
            // treatment as the encoder: pre_buf is (C, Fq, T) with t fastest.
            if (!dec.empty && !dec.dconv.layers.empty() && htdemucs_fastconv()) {
                HTD_PROF(prof, "dec.dconv");
                cpu_dconv_bands(pre_buf, pre_Fq, pre_C, pre_T, dec.dconv);
            } else if (!dec.empty && !dec.dconv.layers.empty()) {
                HTD_PROF(prof, "dec.dconv");
                std::vector<float> slice((size_t)pre_T * pre_C);
                for (int fq = 0; fq < pre_Fq; fq++) {
                    for (int c = 0; c < pre_C; c++)
                        for (int t = 0; t < pre_T; t++)
                            slice[t + (size_t)c * pre_T] = pre_buf[t + (size_t)fq * pre_T + (size_t)c * pre_T * pre_Fq];

                    cpu_dconv_inplace(slice, pre_C, pre_T, dec.dconv);

                    for (int c = 0; c < pre_C; c++)
                        for (int t = 0; t < pre_T; t++)
                            pre_buf[t + (size_t)fq * pre_T + (size_t)c * pre_T * pre_Fq] = slice[t + (size_t)c * pre_T];
                }
            }

            if (htdemucs_debug()) {
                int nc = 0;
                float mx = 0;
                for (size_t i = 0; i < pre_n; i++) {
                    if (std::isnan(pre_buf[i]))
                        nc++;
                    float av = fabsf(pre_buf[i]);
                    if (av > mx)
                        mx = av;
                }
                fprintf(stderr, "htdemucs: dec[%d] pre-CT (%d,%d,%d) max=%.2f nan=%d\n", idx, pre_C, pre_Fq, pre_T, mx,
                        nc);
            }

            // CPU ConvTranspose2d with kernel [K,1], stride [S,1]
            // Weight layout in GGUF: ne = (1, K, OC, IC) for ConvTranspose2d
            // PyTorch ConvTranspose2d(IC, OC, [K,1], [S,1]) weight shape: (IC, OC, K, 1)
            // In ggml ne order: (KW=1, KH=K, ne2=OC, ne3=IC)
            {
                HTD_PROF(prof, "dec.convtranspose");
                int ct_K = (int)dec.conv_tr_w->ne[1]; // KH
                int ct_OC = (int)dec.conv_tr_w->ne[2];
                int ct_IC = (int)dec.conv_tr_w->ne[3];
                int ct_S = stri;
                int ct_pad = pad_val;

                // Output freq size: Fq_out = (Fq_in - 1) * S + K - 2*pad
                // For htdemucs: pad = K/4 (symmetric), but ConvTranspose2d has no padding param.
                // Actually PyTorch ConvTranspose2d uses: Fq_out = (Fq_in - 1) * S - 2*P + K
                // where P = pad_val from the encoder. The decoder uses the same kernel_size and
                // the forward does: z = conv_tr(y), then crops: z[..., pad:-pad, :]
                // So the ConvTranspose output is (Fq_in-1)*S + K, then cropped by pad on each side.
                int ct_Fq_raw = (pre_Fq - 1) * ct_S + ct_K;
                int ct_Fq_out = ct_Fq_raw - 2 * ct_pad;

                // Read weight data (may be F16)
                // read_tensor_f32 already handles the F16 -> F32 conversion; cache it
                // so the whole kernel is not re-converted on every decoder call.
                const std::vector<float>& w_data = cached_tensor_f32(dec.conv_tr_w);

                // Output buffer: (pre_T, ct_Fq_out, ct_OC) = (T, Fq_out, OC)
                size_t ct_out_n = (size_t)pre_T * ct_Fq_out * ct_OC;
                std::vector<float> ct_out(ct_out_n, 0.0f);

                // ConvTranspose1d on freq axis: for each time frame t:
                // y[oc, fq_out, t] = sum_ic sum_kh x[ic, fq_in, t] * w[ic, oc, kh]
                // where fq_out = fq_in * S + kh - pad
                // Weight indexing: w_data[ic * OC * K + oc * K + kh] (ggml ne order: (1, K, OC, IC))
                // → w[ic][oc][kh] = w_data[ic * ct_OC * ct_K + oc * ct_K + kh]
                if (htdemucs_fastconv()) {
                    // ConvTranspose as ONE GEMM + a strided scatter-add. The scalar
                    // form above is a 5-deep loop with a scattered innermost WRITE;
                    // it measured 68% of the whole forward. Reformulated:
                    //
                    //   tmp[(oc*K + kh), (fq_in*T + t)] = sum_ic W[ic][oc][kh] * x[ic, fq_in, t]
                    //
                    // The x operand is already (IC, Fq_in*T) in memory — pre_buf is
                    // indexed t + fq_in*T + ic*T*Fq_in — so no packing is needed, and
                    // the IC reduction (the only contraction) goes through sgemm.
                    // The remaining scatter walks contiguous T-runs.
                    std::vector<float> wt((size_t)ct_OC * ct_K * ct_IC);
                    for (int ic = 0; ic < ct_IC; ic++)
                        for (int oc = 0; oc < ct_OC; oc++)
                            for (int kh = 0; kh < ct_K; kh++)
                                wt[(size_t)(oc * ct_K + kh) * ct_IC + ic] =
                                    w_data[(size_t)ic * ct_OC * ct_K + oc * ct_K + kh];

                    const size_t ncol = (size_t)pre_Fq * pre_T;
                    std::vector<float> tmp((size_t)ct_OC * ct_K * ncol, 0.0f);
                    htd_gemm(ct_OC * ct_K, (int)ncol, ct_IC, wt.data(), pre_buf.data(), tmp.data());

                    for (int oc = 0; oc < ct_OC; oc++)
                        for (int kh = 0; kh < ct_K; kh++) {
                            const float* src = tmp.data() + (size_t)(oc * ct_K + kh) * ncol;
                            float* dstc = ct_out.data() + (size_t)oc * pre_T * ct_Fq_out;
                            for (int fq_in = 0; fq_in < pre_Fq; fq_in++) {
                                const int fq_out = fq_in * ct_S + kh - ct_pad;
                                if (fq_out < 0 || fq_out >= ct_Fq_out)
                                    continue;
                                const float* srow = src + (size_t)fq_in * pre_T;
                                float* drow = dstc + (size_t)fq_out * pre_T;
                                for (int t = 0; t < pre_T; t++)
                                    drow[t] += srow[t];
                            }
                        }
                } else {
                    for (int t = 0; t < pre_T; t++) {
                        for (int ic = 0; ic < ct_IC; ic++) {
                            for (int fq_in = 0; fq_in < pre_Fq; fq_in++) {
                                float x_val = pre_buf[t + (size_t)fq_in * pre_T + (size_t)ic * pre_T * pre_Fq];
                                for (int kh = 0; kh < ct_K; kh++) {
                                    int fq_out = fq_in * ct_S + kh - ct_pad;
                                    if (fq_out < 0 || fq_out >= ct_Fq_out)
                                        continue;
                                    for (int oc = 0; oc < ct_OC; oc++) {
                                        float w_val = w_data[(size_t)ic * ct_OC * ct_K + oc * ct_K + kh];
                                        ct_out[t + (size_t)fq_out * pre_T + (size_t)oc * pre_T * ct_Fq_out] +=
                                            x_val * w_val;
                                    }
                                }
                            }
                        }
                    }
                }

                // Add bias
                if (dec.conv_tr_b) {
                    std::vector<float> bias(ct_OC);
                    {
                        auto _rd = read_tensor_f32(dec.conv_tr_b);
                        memcpy(bias.data(), _rd.data(), std::min(bias.size(), _rd.size()) * sizeof(float));
                    }
                    for (int oc = 0; oc < ct_OC; oc++) {
                        for (int fq = 0; fq < ct_Fq_out; fq++) {
                            for (int t = 0; t < pre_T; t++) {
                                ct_out[t + (size_t)fq * pre_T + (size_t)oc * pre_T * ct_Fq_out] += bias[oc];
                            }
                        }
                    }
                }

                x_buf = std::move(ct_out);
                x_C = ct_OC;
                x_Fq = ct_Fq_out;
                x_T = pre_T;

                if (htdemucs_debug()) {
                    fprintf(stderr, "htdemucs: dec[%d] ConvTranspose K=%d S=%d pad=%d: (%d,%d,%d) → (%d,%d,%d)\n", idx,
                            ct_K, ct_S, ct_pad, ct_IC, pre_Fq, pre_T, ct_OC, ct_Fq_out, pre_T);
                }
            }

            // GroupNorm after ConvTranspose (CPU-side)
            if (dec.norm2_w) {
                // CPU GroupNorm: normalize per group of channels over (Fq × T) spatial dims
                int ng = 4;
                std::vector<float> norm_w(x_C), norm_b(x_C);
                {
                    auto _rd = read_tensor_f32(dec.norm2_w);
                    memcpy(norm_w.data(), _rd.data(), std::min(norm_w.size(), _rd.size()) * sizeof(float));
                }
                if (dec.norm2_b) {
                    auto _rd = read_tensor_f32(dec.norm2_b);
                    memcpy(norm_b.data(), _rd.data(), std::min(norm_b.size(), _rd.size()) * sizeof(float));
                }
                int ch_per_group = x_C / ng;
                size_t spatial = (size_t)x_Fq * x_T;
                for (int grp = 0; grp < ng; grp++) {
                    int c_start = grp * ch_per_group;
                    int c_end = c_start + ch_per_group;
                    // Compute mean/var over channels in this group and all spatial positions
                    double sum = 0, sum2 = 0;
                    size_t count = (size_t)(c_end - c_start) * spatial;
                    for (int c = c_start; c < c_end; c++)
                        for (size_t s = 0; s < spatial; s++) {
                            float v = x_buf[s + (size_t)c * spatial];
                            sum += v;
                            sum2 += (double)v * v;
                        }
                    float mean_g = (float)(sum / count);
                    float var_g = (float)(sum2 / count - (double)mean_g * mean_g);
                    float inv_std = 1.0f / sqrtf(var_g + 1e-5f);
                    for (int c = c_start; c < c_end; c++)
                        for (size_t s = 0; s < spatial; s++) {
                            size_t i = s + (size_t)c * spatial;
                            x_buf[i] = (x_buf[i] - mean_g) * inv_std * norm_w[c] + (dec.norm2_b ? norm_b[c] : 0.0f);
                        }
                }
            }

            // Crop freq padding from ConvTranspose output
            // Python: if self.freq and self.pad: z = z[..., self.pad:-self.pad, :]
            // Already handled in the ConvTranspose code above (ct_Fq_out = ct_Fq_raw - 2*pad)

            // GELU (not on last layer)
            if (!is_last) {
                for (size_t i = 0; i < x_buf.size(); i++) {
                    float v = x_buf[i];
                    // GELU(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                    x_buf[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                }
            }
        }

        // `pre` for the time decoder comes from whichever path ran.
        if (dec_done) {
            pre_buf = std::move(pre_buf_g);
            pre_C = pre_C_g;
            pre_Fq = pre_Fq_g;
            pre_T = pre_T_g;
        }

        // Time branch decoder
        int tdec_offset = hp.depth - (int)m.tdecoder.size();
        if (idx >= tdec_offset && (idx - tdec_offset) < (int)m.tdecoder.size()) {
            auto& tdec = m.tdecoder[idx - tdec_offset];
            if (tdec.conv_tr_w) {
                if (tdec.empty) {
                    // Empty tdecoder: take pre_buf (freq pre-ConvTranspose output),
                    // squeeze freq dim (Fq must be 1), run ConvTranspose1d
                    // pre_buf: (pre_T, pre_Fq, pre_C) where pre_Fq=1 (innermost layer)
                    // Squeeze → (pre_T, pre_C) = time signal
                    xt_buf.resize((size_t)pre_T * pre_C);
                    for (int c = 0; c < pre_C; c++)
                        for (int t = 0; t < pre_T; t++)
                            xt_buf[t + (size_t)c * pre_T] = pre_buf[t + (size_t)c * pre_T * pre_Fq];
                    xt_C = pre_C;
                    xt_T = pre_T;
                } else {
                    // Non-empty: xt = xt + skip_t, then rewrite + GLU
                    if (!saved_time.empty()) {
                        auto skip_t = saved_time.back();
                        saved_time.pop_back();
                        // xt + skip
                        for (int c = 0; c < xt_C; c++)
                            for (int t = 0; t < xt_T; t++)
                                xt_buf[t + (size_t)c * xt_T] += skip_t.data[t + (size_t)c * skip_t.T];
                    }
                    // Rewrite + GLU. This is a Conv1d(chin, 2*chin, 1+2*context)
                    // with padding=context — NOT a 1x1 conv (context=1 => K=3).
                    if (tdec.rewrite_w) {
                        HTD_PROF(prof, "tdec.rewrite");
                        int rw_OC = 0, rw_T = 0;
                        auto rw_out = cpu_conv1d_time(xt_buf, xt_T, xt_C, tdec.rewrite_w, tdec.rewrite_b, hp.context,
                                                      rw_OC, rw_T);
                        xt_buf = cpu_glu(rw_out, rw_T, rw_OC);
                        xt_C = rw_OC / 2;
                        xt_T = rw_T;
                    }
                }

                // DConv, also missing on the time decoder (see the freq branch).
                if (!tdec.empty && !tdec.dconv.layers.empty()) {
                    if (htdemucs_fastconv())
                        cpu_dconv_bands(xt_buf, 1, xt_C, xt_T, tdec.dconv);
                    else
                        cpu_dconv_inplace(xt_buf, xt_C, xt_T, tdec.dconv);
                }

                // ConvTranspose1d on time axis
                HTD_PROF(prof, "tdec.convtranspose");
                int ct_K = (int)tdec.conv_tr_w->ne[0]; // kernel size
                int ct_OC = (int)tdec.conv_tr_w->ne[1];
                int ct_IC = (int)tdec.conv_tr_w->ne[2];
                int ct_pad = ct_K / 4;
                int t_out_raw = (xt_T - 1) * stri + ct_K;
                int t_length = !lengths_time.empty() ? lengths_time.back() : (t_out_raw - 2 * ct_pad);
                if (!lengths_time.empty())
                    lengths_time.pop_back();

                const std::vector<float>& ct_w = cached_tensor_f32(tdec.conv_tr_w);
                // ConvTranspose1d: out[oc, t_out] = sum_{ic, k} x[ic, t_in] * w[k, oc, ic]
                // where t_out = t_in * stride + k
                std::vector<float> ct_out((size_t)ct_OC * t_out_raw, 0.0f);
                if (htdemucs_fastconv()) {
                    // Same GEMM + scatter-add shape as the freq ConvTranspose:
                    //   tmp[(oc*K + k), t_in] = sum_ic W[ic][oc][k] * x[ic, t_in]
                    // xt_buf is already (IC, xt_T), so it is the B operand as-is;
                    // the scatter then adds each (oc, k) row at stride `stri`.
                    std::vector<float> wt((size_t)ct_OC * ct_K * ct_IC);
                    for (int ic = 0; ic < ct_IC; ic++)
                        for (int oc = 0; oc < ct_OC; oc++)
                            for (int k = 0; k < ct_K; k++)
                                wt[(size_t)(oc * ct_K + k) * ct_IC + ic] =
                                    ct_w[(size_t)ic * ct_OC * ct_K + oc * ct_K + k];

                    std::vector<float> tmp((size_t)ct_OC * ct_K * xt_T, 0.0f);
                    htd_gemm(ct_OC * ct_K, xt_T, ct_IC, wt.data(), xt_buf.data(), tmp.data());

                    for (int oc = 0; oc < ct_OC; oc++)
                        for (int k = 0; k < ct_K; k++) {
                            const float* srow = tmp.data() + (size_t)(oc * ct_K + k) * xt_T;
                            float* drow = ct_out.data() + (size_t)oc * t_out_raw;
                            for (int t_in = 0; t_in < xt_T; t_in++) {
                                const int t_o = t_in * stri + k;
                                if (t_o >= t_out_raw)
                                    break;
                                drow[t_o] += srow[t_in];
                            }
                        }
                } else {
                    for (int ic = 0; ic < ct_IC; ic++)
                        for (int t_in = 0; t_in < xt_T; t_in++) {
                            float x_val = xt_buf[t_in + (size_t)ic * xt_T];
                            for (int k = 0; k < ct_K; k++) {
                                int t_o = t_in * stri + k;
                                if (t_o >= t_out_raw)
                                    continue;
                                for (int oc = 0; oc < ct_OC; oc++)
                                    ct_out[t_o + (size_t)oc * t_out_raw] +=
                                        x_val * ct_w[(size_t)ic * ct_OC * ct_K + oc * ct_K + k];
                            }
                        }
                }
                if (tdec.conv_tr_b) {
                    auto b = read_tensor_f32(tdec.conv_tr_b);
                    for (int oc = 0; oc < ct_OC; oc++)
                        for (int t = 0; t < t_out_raw; t++)
                            ct_out[t + (size_t)oc * t_out_raw] += b[oc];
                }
                // Crop padding: out[pad : pad + length]
                xt_T = std::min(t_length, t_out_raw - 2 * ct_pad);
                xt_C = ct_OC;
                std::vector<float> cropped(xt_C * xt_T);
                for (int oc = 0; oc < xt_C; oc++)
                    for (int t = 0; t < xt_T; t++)
                        cropped[t + (size_t)oc * xt_T] = ct_out[(ct_pad + t) + (size_t)oc * t_out_raw];
                xt_buf = std::move(cropped);

                // GroupNorm (skip — norm_starts=4) + GELU (not on last layer)
                bool tdec_last = (idx == hp.depth - 1);
                if (!tdec_last) {
                    for (size_t i = 0; i < xt_buf.size(); i++) {
                        float v = xt_buf[i];
                        xt_buf[i] = v * 0.5f * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                    }
                }

                if (htdemucs_debug()) {
                    fprintf(stderr, "htdemucs: tdec[%d] output (%d, %d)\n", idx - tdec_offset, xt_C, xt_T);
                }
            }
        }

        if (htdemucs_debug()) {
            int nc = 0;
            for (size_t i = 0; i < x_buf.size() && nc < 3; i++)
                if (std::isnan(x_buf[i]) || std::isinf(x_buf[i]))
                    nc++;
            fprintf(stderr, "htdemucs: dec[%d] output (%d, %d, %d)%s\n", idx, x_C, x_Fq, x_T,
                    nc > 0 ? " *** HAS NaN ***" : "");
        }

        if (!fused_done) // fused captures nothing per-layer; it is validated on output_*
            htd_capture(ctx, ("dec_freq_" + std::to_string(idx)).c_str(), x_buf.data(), x_buf.size());
    }

    // NaN check after decoder
    {
        int nan_count = 0;
        for (size_t i = 0; i < x_buf.size() && nan_count < 5; i++) {
            if (std::isnan(x_buf[i]) || std::isinf(x_buf[i]))
                nan_count++;
        }
        if (nan_count > 0)
            fprintf(stderr, "htdemucs: WARNING: %d NaN/Inf in decoder output\n", nan_count);
    }
    fprintf(stderr, "htdemucs: decoder done, output (%d, %d, %d)\n", x_C, x_Fq, x_T);

    // Step 9: CaC unmask → iSTFT → denormalize → sum branches → output
    //
    // Decoder output x_buf: (T, Fq, C, 1) where C = n_sources * audio_channels * 2
    // (CaC = 2× for real/imag). For 4 sources × 2 channels × 2 (real/imag) = 16 channels.
    // Reshape: (B, S, C*2//S, Fq, T) → for each source: (2, Fq, T) complex spec
    // Then iSTFT each source independently.

    int S = hp.n_sources;       // 4
    int ac = hp.audio_channels; // 2
    int ch_per_source = ac * 2; // 4 (2 audio channels × 2 for CaC real/imag)

    // Denormalize: x = x * std + mean
    for (size_t i = 0; i < x_buf.size(); i++) {
        x_buf[i] = x_buf[i] * spec_std + spec_mean;
    }

    // CaC unmask: decoder output → per-source complex spectrogram → iSTFT
    // x_buf layout: (x_T, x_Fq, x_C) where x_C = S * ch_per_source = 16
    // For source s, audio channel c: real = x_buf[..., s*ch_per_source + c*2]
    //                                 imag = x_buf[..., s*ch_per_source + c*2+1]
    auto r = new htdemucs_result();
    r->n_sources = S;
    r->n_channels = ac;
    r->n_samples = n_samples;
    r->sample_rate = hp.samplerate;
    r->sources = new float*[S];
    r->source_names = new const char*[S];

    for (int s = 0; s < S; s++) {
        r->source_names[s] = m.source_names[s].c_str();

        HTD_PROF(prof, "cac_unmask");
        // Extract per-source complex spectrogram (ac channels × Fq × T)
        // Need to add back the dropped freq bin (Fq → Fq+1) with zeros
        int n_freqs_full = Fq + 1; // restore to nfft/2+1
        // _ispec pads 2 zero frames on each side: F.pad(z, (2, 2)). They are not
        // merely trimmed later — they change the window-sum normalisation in the
        // first/last nfft samples, which is exactly the region we crop into.
        const int istft_frame_pad = 2;
        const int n_frames_pad = x_T + 2 * istft_frame_pad;
        std::vector<float> src_real((size_t)ac * n_freqs_full * n_frames_pad, 0.0f);
        std::vector<float> src_imag((size_t)ac * n_freqs_full * n_frames_pad, 0.0f);

        for (int c = 0; c < ac; c++) {
            for (int fq = 0; fq < Fq; fq++) {
                for (int t = 0; t < x_T; t++) {
                    // x_buf index: t + fq * x_T + ch * x_T * x_Fq
                    int ch_re = s * ch_per_source + c * 2;
                    int ch_im = s * ch_per_source + c * 2 + 1;
                    float re = x_buf[t + (size_t)fq * x_T + (size_t)ch_re * x_T * x_Fq];
                    float im = x_buf[t + (size_t)fq * x_T + (size_t)ch_im * x_T * x_Fq];
                    // Output spec layout for iSTFT: (ch, freqs, frames), written
                    // at frame offset istft_frame_pad.
                    const size_t di =
                        (size_t)c * n_freqs_full * n_frames_pad + (size_t)fq * n_frames_pad + (istft_frame_pad + t);
                    src_real[di] = re;
                    src_imag[di] = im;
                }
            }
        }

        // iSTFT mirroring _ispec: the freq bin and the 2+2 frames are restored
        // above; here we drop the center=True pad (nfft/2) and then _spec()'s
        // reflect pre-padding (hop/2*3), which is _ispec's x[pad : pad+length].
        const int istft_start = nfft / 2 + pre_pad_left;
        std::vector<float> src_pcm((size_t)ac * work_length, 0.0f);
        {
            HTD_PROF(prof, "istft");
            compute_istft(src_real.data(), src_imag.data(), ac, n_freqs_full, n_frames_pad, nfft, hop,
                          ctx->hann_window.data(), work_length, istft_start, src_pcm.data());
        }

        // Capture the spectrogram branch alone, before the time branch is added.
        if (ctx->capture_stages) {
            std::vector<float> cm((size_t)ac * n_samples);
            for (int c = 0; c < ac; c++)
                for (int i = 0; i < n_samples; i++)
                    cm[(size_t)c * n_samples + i] = src_pcm[(size_t)c * work_length + i];
            htd_capture(ctx, ("spec_" + m.source_names[s]).c_str(), cm.data(), cm.size());
        }

        // Denormalize time branch and add to freq branch
        // xt_buf: (xt_T, xt_C) where xt_C = S * ac = 4 * 2 = 8
        // Reshape to per-source: xt[s][c][t] = xt_buf[t + (s*ac+c)*xt_T]
        // Denormalize: xt = xt * time_std + time_mean
        if (xt_C >= S * ac && xt_T >= n_samples) {
            int src_ch = s * ac;
            for (int c = 0; c < ac; c++) {
                for (int i = 0; i < n_samples; i++) {
                    float tv = xt_buf[i + (size_t)(src_ch + c) * xt_T];
                    tv = tv * time_std + time_mean;
                    src_pcm[(size_t)c * work_length + i] += tv;
                }
            }
        } else if (htdemucs_debug()) {
            fprintf(stderr, "htdemucs: TIME BRANCH SKIPPED (xt_C=%d need>=%d, xt_T=%d need>=%d)\n", xt_C, S * ac, xt_T,
                    n_samples);
        }

        // Capture the time branch alone (denormalized), for the same reason.
        if (ctx->capture_stages && xt_C >= S * ac && xt_T >= n_samples) {
            std::vector<float> cm((size_t)ac * n_samples);
            for (int c = 0; c < ac; c++)
                for (int i = 0; i < n_samples; i++)
                    cm[(size_t)c * n_samples + i] = xt_buf[i + (size_t)(s * ac + c) * xt_T] * time_std + time_mean;
            htd_capture(ctx, ("time_" + m.source_names[s]).c_str(), cm.data(), cm.size());
        }

        // Capture channel-major (ac, n_samples) to match the reference layout,
        // before the interleave below.
        if (ctx->capture_stages) {
            std::vector<float> chan_major((size_t)ac * n_samples);
            for (int c = 0; c < ac; c++)
                for (int i = 0; i < n_samples; i++)
                    chan_major[(size_t)c * n_samples + i] = src_pcm[(size_t)c * work_length + i];
            htd_capture(ctx, ("output_" + m.source_names[s]).c_str(), chan_major.data(), chan_major.size());
        }

        // Trim to original length and interleave stereo
        r->sources[s] = new float[(size_t)ac * n_samples];
        for (int i = 0; i < n_samples; i++) {
            for (int c = 0; c < ac; c++) {
                r->sources[s][i * ac + c] = src_pcm[(size_t)c * work_length + i];
            }
        }
    }

    fprintf(stderr, "htdemucs: separated %d samples → %d sources (%d ch @ %d Hz)\n", n_samples, S, ac, hp.samplerate);

    if (htdemucs_profile())
        prof.report();

    return r;
}

// Segmented forward with weighted overlap-add.
//
// Ported from upstream Demucs `apply_model(..., split=True)` in demucs/apply.py
// (MIT), whose defaults are overlap=0.25 and transition_power=1.0:
//
//   segment_length = int(samplerate * model.segment)      // 7.8 s * 44100
//   stride         = int((1 - overlap) * segment_length)
//   weight         = concat(arange(1, L/2+1), arange(L - L/2, 0, -1))
//   weight         = (weight / weight.max()) ** transition_power
//   out[off : off+L]        += weight[:n] * chunk_out
//   sum_weight[off : off+L] += weight[:n]
//   out /= sum_weight
//
// i.e. a triangle peaking mid-segment, normalised by the accumulated weight so
// every output sample is a proper weighted average regardless of how many
// segments covered it. (0xShug0/audio.cpp, Apache-2.0, arrives at the same
// shape independently -- make_triangular_overlap_window + overlap_add.)
//
// WHY: the whole-buffer path is O(T^2) in time and grows ~0.12 GB per second of
// audio, so a 3.5-minute song needs ~26 GB and simply OOMs. Segmenting bounds
// peak memory to one segment's working set and makes time linear in length.
//
// Short inputs (<= one segment) still take the whole-buffer path, so their
// output is bit-identical to before this change -- there is no regression
// surface for the case that already worked. CRISPASR_HTDEMUCS_NO_SEGMENT=1
// forces the old behaviour everywhere for A/B.
htdemucs_result* htdemucs_separate(htdemucs_context* ctx, const float* pcm_stereo, int n_samples) {
    if (!ctx || !pcm_stereo || n_samples <= 0)
        return nullptr;

    const auto& hp = ctx->model.hparams;
    const int seg_len = hp.training_length();
    const bool no_seg = std::getenv("CRISPASR_HTDEMUCS_NO_SEGMENT") != nullptr;

    if (no_seg || n_samples <= seg_len || seg_len <= 0)
        return htdemucs_separate_full(ctx, pcm_stereo, n_samples);

    const float overlap = 0.25f;
    int stride = (int)((1.0f - overlap) * (float)seg_len);
    if (stride <= 0)
        stride = seg_len;

    // Triangular weight, peaking mid-segment. Built exactly as upstream:
    // ascending 1..L/2 then descending (L - L/2)..1, then scaled by the max.
    std::vector<float> weight((size_t)seg_len);
    {
        const int half = seg_len / 2;
        for (int i = 0; i < half; i++)
            weight[(size_t)i] = (float)(i + 1);
        for (int i = half; i < seg_len; i++)
            weight[(size_t)i] = (float)(seg_len - i);
        float wmax = 0.0f;
        for (float w : weight)
            wmax = std::max(wmax, w);
        if (wmax > 0.0f)
            for (float& w : weight)
                w /= wmax;
    }

    int n_sources = 0, n_ch = 0, sr = hp.samplerate;
    std::vector<std::vector<float>> acc; // per source, interleaved
    std::vector<float> sum_weight((size_t)n_samples, 0.0f);
    std::vector<const char*> names;
    std::vector<float> chunk((size_t)seg_len * 2, 0.0f);

    for (int off = 0; off < n_samples; off += stride) {
        // Zero-pad the tail chunk to a full segment, as TensorChunk.padded does.
        const int valid = std::min(seg_len, n_samples - off);
        std::fill(chunk.begin(), chunk.end(), 0.0f);
        std::memcpy(chunk.data(), pcm_stereo + (size_t)off * 2, (size_t)valid * 2 * sizeof(float));

        htdemucs_result* part = htdemucs_separate_full(ctx, chunk.data(), seg_len);
        if (!part) {
            for (auto& a : acc)
                a.clear();
            return nullptr;
        }
        if (acc.empty()) {
            n_sources = part->n_sources;
            n_ch = part->n_channels;
            sr = part->sample_rate;
            acc.assign((size_t)n_sources, std::vector<float>((size_t)n_samples * n_ch, 0.0f));
            names.assign(part->source_names, part->source_names + n_sources);
        }
        const int n = std::min(valid, part->n_samples);
        for (int s = 0; s < n_sources; s++) {
            const float* src = part->sources[s];
            float* dst = acc[(size_t)s].data();
            for (int i = 0; i < n; i++) {
                const float w = weight[(size_t)i];
                for (int c = 0; c < n_ch; c++)
                    dst[(size_t)(off + i) * n_ch + c] += w * src[(size_t)i * n_ch + c];
            }
        }
        for (int i = 0; i < n; i++)
            sum_weight[(size_t)(off + i)] += weight[(size_t)i];

        htdemucs_result_free(part);
    }

    if (acc.empty())
        return nullptr;

    // Normalise. sum_weight is positive everywhere the loop covered; guard
    // anyway so a pathological stride can never divide by zero.
    for (int s = 0; s < n_sources; s++) {
        float* dst = acc[(size_t)s].data();
        for (int i = 0; i < n_samples; i++) {
            const float w = sum_weight[(size_t)i];
            if (w <= 0.0f)
                continue;
            for (int c = 0; c < n_ch; c++)
                dst[(size_t)i * n_ch + c] /= w;
        }
    }

    auto* r = new htdemucs_result();
    r->n_sources = n_sources;
    r->n_channels = n_ch;
    r->n_samples = n_samples;
    r->sample_rate = sr;
    r->sources = new float*[n_sources];
    r->source_names = new const char*[n_sources];
    for (int s = 0; s < n_sources; s++) {
        const size_t nfl = (size_t)n_samples * n_ch;
        r->sources[s] = new float[nfl];
        std::memcpy(r->sources[s], acc[(size_t)s].data(), nfl * sizeof(float));
        r->source_names[s] = names[(size_t)s];
    }
    fprintf(stderr, "htdemucs: segmented %d samples into %d chunks of %d (stride %d)\n", n_samples,
            (n_samples + stride - 1) / stride, seg_len, stride);
    return r;
}

void htdemucs_result_free(htdemucs_result* r) {
    if (!r)
        return;
    for (int s = 0; s < r->n_sources; s++)
        delete[] r->sources[s];
    delete[] r->sources;
    delete[] r->source_names;
    delete r;
}

int htdemucs_sample_rate(const htdemucs_context* ctx) {
    return ctx ? ctx->model.hparams.samplerate : 44100;
}

int htdemucs_n_sources(const htdemucs_context* ctx) {
    return ctx ? ctx->model.hparams.n_sources : 4;
}

const char* htdemucs_source_name(const htdemucs_context* ctx, int idx) {
    if (!ctx || idx < 0 || idx >= (int)ctx->model.source_names.size())
        return nullptr;
    return ctx->model.source_names[idx].c_str();
}

// ---------------------------------------------------------------------------
// Frequency encoder layer as a ggml graph (CRISPASR_HTDEMUCS_GGML=1)
//
// The CPU buffer layout (t + fq*T + c*T*Fq) is ALREADY ggml's ne = (T, Fq, C),
// so nothing has to be transposed here — unlike the transformer, this maps
// straight onto ggml_conv_2d with W = time and H = frequency.
//
//   conv        kernel (1, K, IC, OC), s1 = stride, p1 = pad   [freq axis only]
//   dconv conv1 kernel (K, 1, C, hidden), d0 = dilation        [time axis only]
//   dconv conv2 kernel (1, 1, hidden, 2C)                      [1x1 = matmul]
//   rewrite     kernel (1, 1, C, 2C)
//
// KH = 1 on the DConv is what makes each frequency band independent, which is
// exactly Python's `y.permute(0,2,1,3).reshape(-1, C, T)` per-band treatment.
// ---------------------------------------------------------------------------

// GroupNorm(num_groups=1) applied PER frequency band. x is (T, Fq, C); permute
// to (T, C, Fq) so the bands sit on ne[2], then ask for one group per band —
// each group then reduces over ne[0]*ne[1] = T*C, which is the per-band
// semantics. Affine is per-channel.
static ggml_tensor* g_dconv_groupnorm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int n_bands,
                                      float eps) {
    ggml_tensor* p = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3)); // (T, C, Fq)
    p = ggml_group_norm(g, p, n_bands, eps);
    // The affine weights here are the issue-#398 offenders: the F16 GGUF
    // stores `*.dconv.layers.N.4.weight` as F16 (see htdemucs_ggml_util.h).
    if (w)
        p = ggml_mul(g, p, ggml_reshape_3d(g, htd_bcast_f32(g, w), 1, (int)w->ne[0], 1));
    if (b)
        p = ggml_add(g, p, ggml_reshape_3d(g, htd_bcast_f32(g, b), 1, (int)b->ne[0], 1));
    return ggml_cont(g, ggml_permute(g, p, 0, 2, 1, 3)); // back to (T, Fq, C)
}

// GLU over the channel axis (ne[2]) of a (T, Fq, 2C) tensor -> (T, Fq, C).
static ggml_tensor* g_glu_c(ggml_context* g, ggml_tensor* x) {
    const int T = (int)x->ne[0], F = (int)x->ne[1], half = (int)x->ne[2] / 2;
    ggml_tensor* a = ggml_view_3d(g, x, T, F, half, x->nb[1], x->nb[2], 0);
    ggml_tensor* b = ggml_view_3d(g, x, T, F, half, x->nb[1], x->nb[2], (size_t)half * x->nb[2]);
    return ggml_mul(g, ggml_cont(g, a), ggml_sigmoid(g, ggml_cont(g, b)));
}

// The DConv residual stack on a (T, Fq, C) tensor.
static ggml_tensor* g_dconv(ggml_context* g, ggml_tensor* x, const htdemucs_dconv& dc, int n_bands) {
    for (size_t d = 0; d < dc.layers.size(); d++) {
        const auto& sl = dc.layers[d];
        if (!sl.conv1_w || !sl.conv2_w)
            continue;
        const int dilation = 1 << (int)d;
        const int K = (int)sl.conv1_w->ne[0];
        const int C = (int)sl.conv1_w->ne[1];
        const int hidden = (int)sl.conv1_w->ne[2];

        // Conv1d(C -> hidden, K, dilation) along time only: KH = 1.
        ggml_tensor* w1 = ggml_reshape_4d(g, sl.conv1_w, K, 1, C, hidden);
        ggml_tensor* h = ggml_conv_2d(g, w1, x, 1, 1, dilation * (K / 2), 0, dilation, 1);
        if (sl.conv1_b)
            h = ggml_add(g, h, ggml_reshape_3d(g, htd_bcast_f32(g, sl.conv1_b), 1, 1, hidden));
        if (sl.norm1_w)
            h = g_dconv_groupnorm(g, h, sl.norm1_w, sl.norm1_b, n_bands, 1e-5f);
        h = ggml_gelu(g, h);

        const int out2C = (int)sl.conv2_w->ne[2];
        ggml_tensor* w2 = ggml_reshape_4d(g, sl.conv2_w, 1, 1, hidden, out2C);
        ggml_tensor* h2 = ggml_conv_2d(g, w2, h, 1, 1, 0, 0, 1, 1);
        if (sl.conv2_b)
            h2 = ggml_add(g, h2, ggml_reshape_3d(g, htd_bcast_f32(g, sl.conv2_b), 1, 1, out2C));
        if (sl.norm2_w)
            h2 = g_dconv_groupnorm(g, h2, sl.norm2_w, sl.norm2_b, n_bands, 1e-5f);

        ggml_tensor* y = g_glu_c(g, h2);
        if (sl.scale)
            y = ggml_mul(g, y, ggml_reshape_3d(g, htd_bcast_f32(g, sl.scale), 1, 1, (int)sl.scale->ne[0]));
        x = ggml_add(g, x, y);
    }
    return x;
}

// One frequency-encoder layer, entirely in-graph. x_buf is the CPU (C, Fq, T)
// buffer and is updated in place; x_C / x_Fq are updated to the new dims.
// Returns false if the graph could not be built or allocated.
static bool htdemucs_enc_freq_ggml(htdemucs_context* ctx, const htdemucs_enc_layer& enc, std::vector<float>& x_buf,
                                   int& x_C, int& x_Fq, int x_T, int stride, int pad, const std::vector<float>* inject,
                                   int inject_C, int idx) {
    const size_t n_nodes = 2048;
    const size_t ctx_size = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params gp = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(gp);
    if (!g)
        return false;

    ggml_tensor* X = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, x_Fq, x_C);
    ggml_set_name(X, "enc_in");
    ggml_set_input(X);

    // Conv2d over the frequency axis only: kernel (KW=1, KH=K, IC, OC).
    ggml_tensor* x = ggml_conv_2d(g, enc.conv_w, X, 1, stride, 0, pad, 1, 1);
    const int OC = (int)enc.conv_w->ne[3];
    if (enc.conv_b)
        x = ggml_add(g, x, ggml_reshape_3d(g, htd_bcast_f32(g, enc.conv_b), 1, 1, OC));

    ggml_tensor* INJ = nullptr;
    if (inject) {
        // Time-branch injection broadcasts over the frequency axis.
        INJ = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, 1, inject_C);
        ggml_set_name(INJ, "enc_inject");
        ggml_set_input(INJ);
        x = ggml_add(g, x, INJ);
    }

    const int out_Fq = (int)x->ne[1];
    // Layer-0 sub-stages for the diff harness. set_output is required or gallocr
    // recycles the buffers before we read them.
    const bool cap0 = ctx->capture_stages && idx == 0;
    std::vector<std::pair<const char*, ggml_tensor*>> subs;
    if (cap0) {
        ggml_set_output(x);
        subs.emplace_back("enc0_conv", x);
    }
    if (!enc.empty) {
        x = ggml_gelu(g, x);
        if (cap0) {
            ggml_set_output(x);
            subs.emplace_back("enc0_gelu", x);
        }
        if (!enc.dconv.layers.empty())
            x = g_dconv(g, x, enc.dconv, out_Fq);
        if (cap0) {
            ggml_set_output(x);
            subs.emplace_back("enc0_dconv", x);
        }
        if (enc.rewrite_w) {
            ggml_tensor* rw = ggml_conv_2d(g, enc.rewrite_w, x, 1, 1, 0, 0, 1, 1);
            if (enc.rewrite_b)
                rw = ggml_add(g, rw,
                              ggml_reshape_3d(g, htd_bcast_f32(g, enc.rewrite_b), 1, 1, (int)enc.rewrite_w->ne[3]));
            x = g_glu_c(g, rw);
        }
        if (cap0) {
            ggml_set_output(x);
            subs.emplace_back("enc0_rewrite", x);
        }
    }

    ggml_set_name(x, "enc_out");
    ggml_set_output(x);
    ggml_cgraph* gf = ggml_new_graph_custom(g, n_nodes, false);
    ggml_build_forward_expand(gf, x);
    for (auto& kv : subs)
        ggml_build_forward_expand(gf, kv.second);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "htdemucs: enc graph alloc failed — falling back to CPU\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }
    ggml_backend_tensor_set(X, x_buf.data(), 0, x_buf.size() * sizeof(float));
    if (INJ && inject)
        ggml_backend_tensor_set(INJ, inject->data(), 0, inject->size() * sizeof(float));
    ggml_backend_graph_compute(ctx->backend, gf);

    for (auto& kv : subs) {
        std::vector<float> tmp(ggml_nelements(kv.second));
        ggml_backend_tensor_get(kv.second, tmp.data(), 0, tmp.size() * sizeof(float));
        htd_capture(ctx, kv.first, tmp.data(), tmp.size());
    }

    x_C = (int)x->ne[2];
    x_Fq = (int)x->ne[1];
    x_buf.assign((size_t)x_C * x_Fq * x_T, 0.0f);
    ggml_backend_tensor_get(x, x_buf.data(), 0, x_buf.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// Frequency decoder layer as a ggml graph (CRISPASR_HTDEMUCS_GGML=1)
//
// The one op ggml cannot express directly is the ConvTranspose2d: it only has
// ggml_conv_transpose_2d_p0 (a single stride, zero padding), while this model
// needs kernel [K,1] with stride [S,1] on the frequency axis only.
//
// Decomposed instead into K independent 1x1 convolutions plus a strided
// scatter-add, which is the same identity the CPU GEMM+scatter path uses:
//
//   out_raw[t, fq_in*S + kh, oc] += sum_ic W[ic][oc][kh] * x[t, fq_in, ic]
//
// For a FIXED kh that inner sum is exactly a 1x1 conv over the channel axis,
// so it is ggml_conv_2d with a (1,1,IC,OC) kernel; the placement is a
// ggml_acc into a view with freq stride S. Scattering into the UNCROPPED
// output keeps every offset non-negative (the crop that Python writes as
// z[..., pad:-pad, :] is applied afterwards as a view), which is what makes
// the offsets expressible at all.
//
// The K per-tap kernels are sliced host-side and uploaded as small graph
// inputs — far less error-prone than in-graph permutes, and negligible
// (IC*OC floats each, uploaded once per layer).
// ---------------------------------------------------------------------------
static bool htdemucs_dec_freq_ggml(htdemucs_context* ctx, const htdemucs_dec_layer& dec, std::vector<float>& x_buf,
                                   const std::vector<float>& skip_buf, int& x_C, int& x_Fq, int x_T, int stride,
                                   int pad, bool is_last, std::vector<float>* pre_out, int* pre_C, int* pre_Fq) {
    const size_t n_nodes = 4096;
    const size_t ctx_size = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params gp = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(gp);
    if (!g)
        return false;

    const int in_C = x_C, in_Fq = x_Fq;
    ggml_tensor* X = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, in_Fq, in_C);
    ggml_tensor* SK = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, in_Fq, in_C);
    ggml_set_input(X);
    ggml_set_input(SK);

    ggml_tensor* y = X;
    if (!dec.empty) {
        y = ggml_add(g, y, SK);
        if (dec.rewrite_w) {
            // 3x3 Conv2d with context padding on the freq axis.
            ggml_tensor* rw = ggml_conv_2d(g, dec.rewrite_w, y, 1, 1, 1, 1, 1, 1);
            if (dec.rewrite_b)
                rw = ggml_add(g, rw,
                              ggml_reshape_3d(g, htd_bcast_f32(g, dec.rewrite_b), 1, 1, (int)dec.rewrite_w->ne[3]));
            y = g_glu_c(g, rw);
        }
        if (!dec.dconv.layers.empty())
            y = g_dconv(g, y, dec.dconv, in_Fq);
    }
    // `pre` is what the tdecoder consumes; capture it before the transpose.
    ggml_tensor* PRE = y;
    ggml_set_output(PRE);

    const int ct_K = (int)dec.conv_tr_w->ne[1];
    const int ct_OC = (int)dec.conv_tr_w->ne[2];
    const int ct_IC = (int)dec.conv_tr_w->ne[3];
    const int fq_raw = (in_Fq - 1) * stride + ct_K;

    ggml_tensor* ZERO = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, fq_raw, ct_OC);
    ggml_set_input(ZERO);
    ggml_tensor* acc = ZERO;

    std::vector<ggml_tensor*> Wk(ct_K);
    for (int kh = 0; kh < ct_K; kh++) {
        Wk[kh] = ggml_new_tensor_4d(g, GGML_TYPE_F32, 1, 1, ct_IC, ct_OC);
        ggml_set_input(Wk[kh]);
        ggml_tensor* part = ggml_conv_2d(g, Wk[kh], y, 1, 1, 0, 0, 1, 1); // (T, in_Fq, OC)
        acc = ggml_acc(g, acc, part, (size_t)stride * x_T * sizeof(float), (size_t)x_T * fq_raw * sizeof(float),
                       (size_t)x_T * fq_raw * ct_OC * sizeof(float), (size_t)kh * x_T * sizeof(float));
    }
    if (dec.conv_tr_b)
        acc = ggml_add(g, acc, ggml_reshape_3d(g, htd_bcast_f32(g, dec.conv_tr_b), 1, 1, ct_OC));

    // Crop `pad` frequency rows from each side: z[..., pad:-pad, :].
    const int fq_out = fq_raw - 2 * pad;
    ggml_tensor* out = ggml_cont(
        g, ggml_view_3d(g, acc, x_T, fq_out, ct_OC, acc->nb[1], acc->nb[2], (size_t)pad * x_T * sizeof(float)));
    if (!is_last)
        out = ggml_gelu(g, out);
    ggml_set_output(out);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_nodes, false);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, PRE);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "htdemucs: dec graph alloc failed — falling back to CPU\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }

    ggml_backend_tensor_set(X, x_buf.data(), 0, x_buf.size() * sizeof(float));
    ggml_backend_tensor_set(SK, skip_buf.data(), 0,
                            std::min(skip_buf.size(), (size_t)in_C * in_Fq * x_T) * sizeof(float));
    {
        std::vector<float> zeros((size_t)x_T * fq_raw * ct_OC, 0.0f);
        ggml_backend_tensor_set(ZERO, zeros.data(), 0, zeros.size() * sizeof(float));
    }
    {
        // conv_tr weight is ne = (1, K, OC, IC), i.e. w[ic][oc][kh]; each tap
        // becomes a (1,1,IC,OC) kernel laid out as w1x1[oc][ic].
        const std::vector<float>& w = cached_tensor_f32(dec.conv_tr_w);
        std::vector<float> slice((size_t)ct_IC * ct_OC);
        for (int kh = 0; kh < ct_K; kh++) {
            for (int oc = 0; oc < ct_OC; oc++)
                for (int ic = 0; ic < ct_IC; ic++)
                    slice[(size_t)oc * ct_IC + ic] = w[(size_t)ic * ct_OC * ct_K + oc * ct_K + kh];
            ggml_backend_tensor_set(Wk[kh], slice.data(), 0, slice.size() * sizeof(float));
        }
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    if (pre_out) {
        pre_out->assign(ggml_nelements(PRE), 0.0f);
        ggml_backend_tensor_get(PRE, pre_out->data(), 0, pre_out->size() * sizeof(float));
        *pre_C = (int)PRE->ne[2];
        *pre_Fq = (int)PRE->ne[1];
    }
    x_C = (int)out->ne[2];
    x_Fq = (int)out->ne[1];
    x_buf.assign((size_t)x_C * x_Fq * x_T, 0.0f);
    ggml_backend_tensor_get(out, x_buf.data(), 0, x_buf.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// CrossTransformer as a ggml graph (CRISPASR_HTDEMUCS_GGML=1)
//
// The CPU/BLAS path stays the default; this is the opt-in graph port that lets
// the transformer run on Metal/CUDA/Vulkan instead of Accelerate.
//
// LAYOUT. The CPU buffers are (dim-slow, seq-fast): index d*seq + s. ggml
// contracts mul_mat on ne[0], so the graph wants ne = (dim, seq), i.e. memory
// index s*dim + d — the transpose of the CPU layout. We transpose once on
// upload and once on download (1.4M floats, negligible next to the layers).
//
// Weight tensors are already ggml tensors on ctx->backend from load_weights, so
// nothing is re-uploaded per call.
// ---------------------------------------------------------------------------

// LayerNorm over ne[0] (the channel axis) with per-channel affine.
static ggml_tensor* g_layernorm(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    ggml_tensor* y = ggml_norm(g, x, eps);
    if (w)
        y = ggml_mul(g, y, ggml_reshape_2d(g, htd_bcast_f32(g, w), (int)w->ne[0], 1));
    if (b)
        y = ggml_add(g, y, ggml_reshape_2d(g, htd_bcast_f32(g, b), (int)b->ne[0], 1));
    return y;
}

// norm_out: GroupNorm(num_groups=1) over ALL channels AND tokens jointly, then
// a per-channel affine. Flattening to 1D and asking for one group makes
// ggml_group_norm reduce over the whole tensor, which is the semantics we
// need (see the norm_out bug in the parity work).
static ggml_tensor* g_groupnorm1(ggml_context* g, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int dim, int seq,
                                 float eps) {
    ggml_tensor* flat = ggml_reshape_4d(g, x, dim * seq, 1, 1, 1);
    flat = ggml_group_norm(g, flat, 1, eps);
    ggml_tensor* y = ggml_reshape_2d(g, flat, dim, seq);
    if (w)
        y = ggml_mul(g, y, ggml_reshape_2d(g, htd_bcast_f32(g, w), dim, 1));
    if (b)
        y = ggml_add(g, y, ggml_reshape_2d(g, htd_bcast_f32(g, b), dim, 1));
    return y;
}

// Multi-head attention. Q is (dim, q_seq); K, V are (dim, k_seq). Returns
// (dim, q_seq). mask = nullptr is correct here: this model attends over the
// full sequence, there is no causal or padding mask.
static ggml_tensor* g_mha(ggml_context* g, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V, int dim, int n_heads,
                          int q_seq, int k_seq) {
    const int hd = dim / n_heads;
    const float scale = 1.0f / sqrtf((float)hd);

    ggml_tensor* q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, Q, hd, n_heads, q_seq), 0, 2, 1, 3));
    ggml_tensor* k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, K, hd, n_heads, k_seq), 0, 2, 1, 3));
    ggml_tensor* scores = ggml_mul_mat(g, k, q); // (k_seq, q_seq, nh)
    scores = ggml_soft_max_ext(g, scores, nullptr, scale, 0.0f);
    ggml_tensor* v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, V, hd, n_heads, k_seq), 1, 2, 0, 3));
    ggml_tensor* out = ggml_mul_mat(g, v, scores); // (hd, q_seq, nh)
    return ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)), dim, q_seq);
}

static ggml_tensor* g_linear(ggml_context* g, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(g, w, x);
    if (b)
        y = ggml_add(g, y, ggml_reshape_2d(g, htd_bcast_f32(g, b), (int)b->ne[0], 1));
    return y;
}

// x = x + gamma * y   (LayerScale is per-channel)
static ggml_tensor* g_layerscale_add(ggml_context* g, ggml_tensor* x, ggml_tensor* y, ggml_tensor* gamma, int dim) {
    if (gamma)
        y = ggml_mul(g, y, ggml_reshape_2d(g, htd_bcast_f32(g, gamma), dim, 1));
    return ggml_add(g, x, y);
}

// One norm_first self-attention layer:
//   x = x + g1*SA(ln1(x)); x = x + g2*FFN(ln2(x)); x = norm_out(x)
static ggml_tensor* g_self_attn_layer(ggml_context* g, ggml_tensor* x, const htdemucs_self_attn_layer& sa, int dim,
                                      int n_heads, int seq) {
    ggml_tensor* cur = g_layernorm(g, x, sa.norm1_w, sa.norm1_b, 1e-5f);
    ggml_tensor* qkv = g_linear(g, sa.in_proj_w, sa.in_proj_b, cur); // (3*dim, seq)
    const size_t off = (size_t)dim * ggml_element_size(qkv);
    ggml_tensor* Q = ggml_view_2d(g, qkv, dim, seq, qkv->nb[1], 0);
    ggml_tensor* K = ggml_view_2d(g, qkv, dim, seq, qkv->nb[1], off);
    ggml_tensor* V = ggml_view_2d(g, qkv, dim, seq, qkv->nb[1], 2 * off);
    ggml_tensor* att = g_mha(g, ggml_cont(g, Q), ggml_cont(g, K), ggml_cont(g, V), dim, n_heads, seq, seq);
    att = g_linear(g, sa.out_proj_w, sa.out_proj_b, att);
    x = g_layerscale_add(g, x, att, sa.gamma1_scale, dim);

    cur = g_layernorm(g, x, sa.norm2_w, sa.norm2_b, 1e-5f);
    cur = ggml_gelu(g, g_linear(g, sa.linear1_w, sa.linear1_b, cur));
    cur = g_linear(g, sa.linear2_w, sa.linear2_b, cur);
    x = g_layerscale_add(g, x, cur, sa.gamma2_scale, dim);

    if (sa.norm_out_w)
        x = g_groupnorm1(g, x, sa.norm_out_w, sa.norm_out_b, dim, seq, 1e-5f);
    return x;
}

// One cross-attention layer: Q from `x` via norm1, K/V from `other` via norm2,
// FFN gated by norm3.
static ggml_tensor* g_cross_attn_layer(ggml_context* g, ggml_tensor* x, ggml_tensor* other,
                                       const htdemucs_cross_attn_layer& ca, int dim, int n_heads, int q_seq,
                                       int k_seq) {
    ggml_tensor* qn = g_layernorm(g, x, ca.norm1_w, ca.norm1_b, 1e-5f);
    ggml_tensor* kn = g_layernorm(g, other, ca.norm2_w, ca.norm2_b, 1e-5f);

    ggml_tensor* W = ca.cross_attn_in_proj_w; // (dim, 3*dim)
    ggml_tensor* Wq = ggml_view_2d(g, W, dim, dim, W->nb[1], 0);
    ggml_tensor* Wk = ggml_view_2d(g, W, dim, dim, W->nb[1], (size_t)dim * W->nb[1]);
    ggml_tensor* Wv = ggml_view_2d(g, W, dim, dim, W->nb[1], (size_t)2 * dim * W->nb[1]);

    ggml_tensor* Q = ggml_mul_mat(g, ggml_cont(g, Wq), qn);
    ggml_tensor* K = ggml_mul_mat(g, ggml_cont(g, Wk), kn);
    ggml_tensor* V = ggml_mul_mat(g, ggml_cont(g, Wv), kn);
    if (ca.cross_attn_in_proj_b) {
        ggml_tensor* b = ca.cross_attn_in_proj_b;
        const size_t es = ggml_element_size(b);
        Q = ggml_add(g, Q, ggml_reshape_2d(g, htd_bcast_f32(g, ggml_cont(g, ggml_view_1d(g, b, dim, 0))), dim, 1));
        K = ggml_add(
            g, K,
            ggml_reshape_2d(g, htd_bcast_f32(g, ggml_cont(g, ggml_view_1d(g, b, dim, (size_t)dim * es))), dim, 1));
        V = ggml_add(
            g, V,
            ggml_reshape_2d(g, htd_bcast_f32(g, ggml_cont(g, ggml_view_1d(g, b, dim, (size_t)2 * dim * es))), dim, 1));
    }

    ggml_tensor* att = g_mha(g, Q, K, V, dim, n_heads, q_seq, k_seq);
    att = g_linear(g, ca.cross_attn_out_proj_w, ca.cross_attn_out_proj_b, att);
    x = g_layerscale_add(g, x, att, ca.gamma1_scale, dim);

    ggml_tensor* cur = g_layernorm(g, x, ca.norm3_w, ca.norm3_b, 1e-5f);
    cur = ggml_gelu(g, g_linear(g, ca.linear1_w, ca.linear1_b, cur));
    cur = g_linear(g, ca.linear2_w, ca.linear2_b, cur);
    x = g_layerscale_add(g, x, cur, ca.gamma2_scale, dim);

    if (ca.norm_out_w)
        x = g_groupnorm1(g, x, ca.norm_out_w, ca.norm_out_b, dim, q_seq, 1e-5f);
    return x;
}

// Runs all t_layers on both branches in ONE graph. x_buf/xt_buf are the CPU
// (dim-slow, seq-fast) buffers and are updated in place. Returns false if the
// graph could not be built or allocated, so the caller can fall back.
static bool htdemucs_transformer_ggml(htdemucs_context* ctx, std::vector<float>& x_buf, int x_seq,
                                      std::vector<float>& xt_buf, int xt_seq, int dim, int n_heads, int n_layers,
                                      int classic_parity) {
    auto& m = ctx->model;
    const size_t n_nodes = 8192;
    const size_t ctx_size = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params gp = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(gp);
    if (!g)
        return false;

    ggml_tensor* X = ggml_new_tensor_2d(g, GGML_TYPE_F32, dim, x_seq);
    ggml_tensor* XT = ggml_new_tensor_2d(g, GGML_TYPE_F32, dim, xt_seq);
    ggml_set_name(X, "ct_x");
    ggml_set_name(XT, "ct_xt");
    ggml_set_input(X);
    ggml_set_input(XT);

    ggml_tensor* x = X;
    ggml_tensor* xt = XT;
    // Per-layer outputs for the diff harness. They MUST be set_output when we
    // intend to read them back, or gallocr will have recycled their buffers by
    // the time we look (the classic "snapshot reads another layer's data" trap).
    std::vector<ggml_tensor*> lay_x, lay_xt;
    const bool cap = ctx->capture_stages;
    for (int li = 0; li < n_layers; li++) {
        const auto& ls = m.ct_layers[li];
        const auto& lt = m.ct_layers_t[li];
        if ((li % 2) == classic_parity) {
            x = g_self_attn_layer(g, x, ls.self_attn, dim, n_heads, x_seq);
            xt = g_self_attn_layer(g, xt, lt.self_attn, dim, n_heads, xt_seq);
        } else {
            // old_x feeds the time branch — capture BEFORE x is overwritten.
            ggml_tensor* old_x = x;
            x = g_cross_attn_layer(g, x, xt, ls.cross_attn, dim, n_heads, x_seq, xt_seq);
            xt = g_cross_attn_layer(g, xt, old_x, lt.cross_attn, dim, n_heads, xt_seq, x_seq);
        }
        if (cap) {
            ggml_set_output(x);
            ggml_set_output(xt);
            lay_x.push_back(x);
            lay_xt.push_back(xt);
        }
    }
    ggml_set_name(x, "ct_x_out");
    ggml_set_name(xt, "ct_xt_out");
    ggml_set_output(x);
    ggml_set_output(xt);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_nodes, false);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, xt);
    for (auto* t : lay_x)
        ggml_build_forward_expand(gf, t);
    for (auto* t : lay_xt)
        ggml_build_forward_expand(gf, t);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "htdemucs: transformer graph alloc failed — falling back to CPU path\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }

    // Upload transposed: CPU (d*seq + s) -> ggml (s*dim + d).
    auto upload = [&](ggml_tensor* t, const std::vector<float>& src, int seq) {
        std::vector<float> tmp((size_t)dim * seq);
        for (int d = 0; d < dim; d++)
            for (int s = 0; s < seq; s++)
                tmp[(size_t)s * dim + d] = src[(size_t)d * seq + s];
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(float));
    };
    upload(X, x_buf, x_seq);
    upload(XT, xt_buf, xt_seq);

    ggml_backend_graph_compute(ctx->backend, gf);

    auto download = [&](ggml_tensor* t, std::vector<float>& dst, int seq) {
        std::vector<float> tmp((size_t)dim * seq);
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * sizeof(float));
        dst.assign((size_t)dim * seq, 0.0f);
        for (int d = 0; d < dim; d++)
            for (int s = 0; s < seq; s++)
                dst[(size_t)d * seq + s] = tmp[(size_t)s * dim + d];
    };
    if (cap) {
        auto grab = [&](ggml_tensor* t, int seq, const std::string& name) {
            std::vector<float> tmp((size_t)dim * seq), out((size_t)dim * seq);
            ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * sizeof(float));
            for (int d = 0; d < dim; d++)
                for (int sI = 0; sI < seq; sI++)
                    out[(size_t)d * seq + sI] = tmp[(size_t)sI * dim + d];
            htd_capture(ctx, name.c_str(), out.data(), out.size());
        };
        for (size_t i = 0; i < lay_x.size(); i++) {
            grab(lay_x[i], x_seq, "ct_l" + std::to_string(i) + "_z");
            grab(lay_xt[i], xt_seq, "ct_l" + std::to_string(i) + "_xt");
        }
    }

    download(x, x_buf, x_seq);
    download(xt, xt_buf, xt_seq);

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return true;
}

static bool htdemucs_fused_run(htdemucs_context* ctx, ggml_context* g, ggml_cgraph* gf, ggml_gallocr_t alloc,
                               ggml_tensor* X, ggml_tensor* XT, ggml_tensor* out, ggml_tensor* xt_out,
                               std::vector<float>& x_buf, int& x_C, int& x_Fq, int x_T, std::vector<float>& xt_buf,
                               int& xt_C, int xt_T, const std::vector<float>& freq_emb_bcast);

// Fills the fused graph's inputs (looked up by name, since they are created
// deep inside the builder), computes, and reads the two outputs back.
static bool htdemucs_fused_run(htdemucs_context* ctx, ggml_context* g, ggml_cgraph* gf, ggml_gallocr_t alloc,
                               ggml_tensor* X, ggml_tensor* XT, ggml_tensor* out, ggml_tensor* xt_out,
                               std::vector<float>& x_buf, int& x_C, int& x_Fq, int x_T, std::vector<float>& xt_buf,
                               int& xt_C, int xt_T, const std::vector<float>& freq_emb_bcast) {
    auto& m = ctx->model;

    ggml_backend_tensor_set(X, x_buf.data(), 0, x_buf.size() * sizeof(float));
    ggml_backend_tensor_set(XT, xt_buf.data(), 0, xt_buf.size() * sizeof(float));

    if (ggml_tensor* e = ggml_get_tensor(g, "fused_freq_emb"))
        ggml_backend_tensor_set(e, freq_emb_bcast.data(), 0,
                                std::min(freq_emb_bcast.size(), (size_t)ggml_nelements(e)) * sizeof(float));

    // Position embeddings: same formulas as the CPU path, prepared host-side
    // into the (dim, seq) layout the graph uses.
    if (ggml_tensor* pz = ggml_get_tensor(g, "fused_pos_z")) {
        const int dim = (int)pz->ne[0], seq = (int)pz->ne[1];
        const int half = dim / 2, Fr = seq / x_T, T1 = x_T;
        std::vector<float> pe((size_t)dim * seq, 0.0f);
        std::vector<float> dt(half / 2);
        for (int i = 0; i < half / 2; i++)
            dt[i] = expf(-(float)(2 * i) * logf(ctx->model.hparams.t_max_period) / (float)half);
        for (int t = 0; t < T1; t++)
            for (int fr = 0; fr < Fr; fr++) {
                const int sidx = fr * T1 + t; // native fr-major order
                for (int i = 0; i < half / 2; i++) {
                    const float pw = (float)t * dt[i], ph = (float)fr * dt[i];
                    float* col = pe.data() + (size_t)sidx * dim;
                    col[2 * i] = ctx->model.hparams.t_weight_pos_embed * sinf(pw);
                    col[2 * i + 1] = ctx->model.hparams.t_weight_pos_embed * cosf(pw);
                    col[half + 2 * i] = ctx->model.hparams.t_weight_pos_embed * sinf(ph);
                    col[half + 2 * i + 1] = ctx->model.hparams.t_weight_pos_embed * cosf(ph);
                }
            }
        ggml_backend_tensor_set(pz, pe.data(), 0, pe.size() * sizeof(float));
    }
    if (ggml_tensor* pt = ggml_get_tensor(g, "fused_pos_xt")) {
        const int dim = (int)pt->ne[0], seq = (int)pt->ne[1], half = dim / 2;
        std::vector<float> pe((size_t)dim * seq, 0.0f);
        for (int sI = 0; sI < seq; sI++)
            for (int d = 0; d < half; d++) {
                const float ph = (float)sI / powf(ctx->model.hparams.t_max_period, (float)d / (float)(half - 1));
                pe[(size_t)sI * dim + d] = ctx->model.hparams.t_weight_pos_embed * cosf(ph);
                pe[(size_t)sI * dim + half + d] = ctx->model.hparams.t_weight_pos_embed * sinf(ph);
            }
        ggml_backend_tensor_set(pt, pe.data(), 0, pe.size() * sizeof(float));
    }

    // Per-layer ConvTranspose zero accumulators and 1x1 tap kernels.
    for (int idx = 0; idx < ctx->model.hparams.depth; idx++) {
        if (ggml_tensor* z = ggml_get_tensor(g, ("fused_zero_" + std::to_string(idx)).c_str())) {
            std::vector<float> zeros(ggml_nelements(z), 0.0f);
            ggml_backend_tensor_set(z, zeros.data(), 0, zeros.size() * sizeof(float));
        }
        const auto& dec = m.decoder[idx];
        if (!dec.conv_tr_w)
            continue;
        const int ct_K = (int)dec.conv_tr_w->ne[1];
        const int ct_OC = (int)dec.conv_tr_w->ne[2];
        const int ct_IC = (int)dec.conv_tr_w->ne[3];
        const std::vector<float>& w = cached_tensor_f32(dec.conv_tr_w);
        std::vector<float> slice((size_t)ct_IC * ct_OC);
        for (int kh = 0; kh < ct_K; kh++) {
            ggml_tensor* wk =
                ggml_get_tensor(g, ("fused_wk_" + std::to_string(idx) + "_" + std::to_string(kh)).c_str());
            if (!wk)
                continue;
            for (int oc = 0; oc < ct_OC; oc++)
                for (int ic = 0; ic < ct_IC; ic++)
                    slice[(size_t)oc * ct_IC + ic] = w[(size_t)ic * ct_OC * ct_K + oc * ct_K + kh];
            ggml_backend_tensor_set(wk, slice.data(), 0, slice.size() * sizeof(float));
        }
    }

    ggml_backend_graph_compute(ctx->backend, gf);

    xt_C = (int)xt_out->ne[0];
    xt_buf.assign(ggml_nelements(xt_out), 0.0f);
    {
        // graph holds (dim, seq); the CPU time decoder wants (C, T).
        std::vector<float> tmp(xt_buf.size());
        ggml_backend_tensor_get(xt_out, tmp.data(), 0, tmp.size() * sizeof(float));
        for (int d = 0; d < xt_C; d++)
            for (int t = 0; t < xt_T; t++)
                xt_buf[(size_t)d * xt_T + t] = tmp[(size_t)t * xt_C + d];
    }

    x_C = (int)out->ne[2];
    x_Fq = (int)out->ne[1];
    x_buf.assign(ggml_nelements(out), 0.0f);
    ggml_backend_tensor_get(out, x_buf.data(), 0, x_buf.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return true;
}

// ---------------------------------------------------------------------------
// FUSED forward: encoder + CrossTransformer + decoder in ONE ggml graph
// (CRISPASR_HTDEMUCS_GGML=1, requires CRISPASR_HTDEMUCS_FUSED=1)
//
// The per-layer graphs were measured SLOWER than CPU+Accelerate for the
// encoder (1.2-2.1x) despite the transformer being 3.3-6x faster, because each
// layer paid a graph build + gallocr alloc and a full host<->device roundtrip
// of its activations — encoder layer 0 alone is 8.2M floats = 33 MB each way.
// Fusing removes ~16 roundtrips, leaving:
//
//   upload   spec_input, xt (pre-transformer), freq_emb table
//   download post-transformer xt, final decoder output
//
// Skip connections stay as in-graph tensors instead of being downloaded and
// re-uploaded, which is the whole point.
//
// `pre` (the tdecoder's alternative input) is NOT produced here: it is only
// consumed when a tdecoder layer is `empty`, and htdemucs has none. Models
// that do have one fall back to the per-layer path.
// ---------------------------------------------------------------------------
static bool htdemucs_fused_ggml(htdemucs_context* ctx, std::vector<float>& x_buf, int& x_C, int& x_Fq, int x_T,
                                std::vector<float>& xt_buf, int& xt_C, int xt_T,
                                const std::vector<float>& freq_emb_bcast) {
    auto& hp = ctx->model.hparams;
    auto& m = ctx->model;

    for (const auto& td : m.tdecoder)
        if (td.empty)
            return false; // `pre` would be needed; use the per-layer path

    const size_t n_nodes = 16384;
    const size_t ctx_size = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
    ggml_init_params gp = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(gp);
    if (!g)
        return false;

    ggml_tensor* X = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, x_Fq, x_C);
    ggml_set_name(X, "fused_spec_in");
    ggml_set_input(X);
    ggml_tensor* XT = ggml_new_tensor_2d(g, GGML_TYPE_F32, xt_T, xt_C);
    ggml_set_name(XT, "fused_xt_in");
    ggml_set_input(XT);

    ggml_tensor* x = X;

    // ---------------- encoder ----------------
    std::vector<ggml_tensor*> skips;
    int freqs_cur = hp.nfft / 2;
    for (int idx = 0; idx < hp.depth; idx++) {
        const auto& enc = m.encoder[idx];
        if (!enc.conv_w) {
            ggml_free(g);
            return false;
        }
        const bool freq = (freqs_cur > 1);
        int stri = freq ? hp.stride : 2;
        int ker = freq ? hp.kernel_size : 4;
        int pad_val = ker / 4;
        if (freq && freqs_cur <= hp.kernel_size) {
            ker = freqs_cur;
            pad_val = 0;
        }

        x = ggml_conv_2d(g, enc.conv_w, x, 1, stri, 0, pad_val, 1, 1);
        if (enc.conv_b)
            x = ggml_add(g, x, ggml_reshape_3d(g, htd_bcast_f32(g, enc.conv_b), 1, 1, (int)enc.conv_w->ne[3]));
        const int nb = (int)x->ne[1];
        if (!enc.empty) {
            x = ggml_gelu(g, x);
            if (!enc.dconv.layers.empty())
                x = g_dconv(g, x, enc.dconv, nb);
            if (enc.rewrite_w) {
                ggml_tensor* rw = ggml_conv_2d(g, enc.rewrite_w, x, 1, 1, 0, 0, 1, 1);
                if (enc.rewrite_b)
                    rw = ggml_add(g, rw,
                                  ggml_reshape_3d(g, htd_bcast_f32(g, enc.rewrite_b), 1, 1, (int)enc.rewrite_w->ne[3]));
                x = g_glu_c(g, rw);
            }
        }
        if (idx == 0 && m.freq_emb_w && !freq_emb_bcast.empty()) {
            // (1, Fq, C) broadcast over the time axis; prepared host-side with
            // the 10 * freq_emb_scale factor already applied.
            ggml_tensor* EMB = ggml_new_tensor_3d(g, GGML_TYPE_F32, 1, (int)x->ne[1], (int)x->ne[2]);
            ggml_set_name(EMB, "fused_freq_emb");
            ggml_set_input(EMB);
            x = ggml_add(g, x, EMB);
        }
        skips.push_back(x);
        if (freq)
            freqs_cur = (freqs_cur <= hp.kernel_size) ? 1 : freqs_cur / hp.stride;
    }

    const int enc_C = (int)x->ne[2], enc_Fq = (int)x->ne[1];
    const int x_seq = x_T * enc_Fq;

    // ---------------- channel up + transformer + channel down ----------------
    // (T, Fq, C) -> (T*Fq, C) is a pure reshape; the transformer wants
    // ne = (dim, seq), so transpose once here and once back.
    ggml_tensor* xf = ggml_reshape_2d(g, x, x_seq, enc_C);
    // xt_buf is channel-major (C, T), i.e. ggml ne = (T, C); the transformer
    // and the channel samplers contract on ne[0], so it must be (C, T).
    ggml_tensor* xtc = ggml_cont(g, ggml_transpose(g, XT));
    if (m.channel_up_w && m.channel_down_w) {
        xf = ggml_cont(g, ggml_transpose(g, xf)); // (C, seq)
        xf = g_linear(g, ggml_reshape_2d(g, m.channel_up_w, (int)m.channel_up_w->ne[1], (int)m.channel_up_w->ne[2]),
                      m.channel_up_b, xf);
        xtc = g_linear(g,
                       ggml_reshape_2d(g, m.channel_up_t_w, (int)m.channel_up_t_w->ne[1], (int)m.channel_up_t_w->ne[2]),
                       m.channel_up_t_b, xtc);

        const int dim = (int)xf->ne[0];
        const int n_heads = hp.t_heads;
        // norm_in then + weight_pos_embed * pos_emb (order matters).
        // Position embeddings are uploaded rather than generated in-graph.
        ggml_tensor* PZ = ggml_new_tensor_2d(g, GGML_TYPE_F32, dim, x_seq);
        ggml_tensor* PT = ggml_new_tensor_2d(g, GGML_TYPE_F32, dim, xt_T);
        ggml_set_name(PZ, "fused_pos_z");
        ggml_set_name(PT, "fused_pos_xt");
        ggml_set_input(PZ);
        ggml_set_input(PT);
        xf = ggml_add(g, g_layernorm(g, xf, m.norm_in_w, m.norm_in_b, 1e-5f), PZ);
        xtc = ggml_add(g, g_layernorm(g, xtc, m.norm_in_t_w, m.norm_in_t_b, 1e-5f), PT);

        for (int li = 0; li < hp.t_layers; li++) {
            const auto& ls = m.ct_layers[li];
            const auto& lt = m.ct_layers_t[li];
            if ((li % 2) == hp.t_classic_parity) {
                xf = g_self_attn_layer(g, xf, ls.self_attn, dim, n_heads, x_seq);
                xtc = g_self_attn_layer(g, xtc, lt.self_attn, dim, n_heads, xt_T);
            } else {
                ggml_tensor* old_x = xf;
                xf = g_cross_attn_layer(g, xf, xtc, ls.cross_attn, dim, n_heads, x_seq, xt_T);
                xtc = g_cross_attn_layer(g, xtc, old_x, lt.cross_attn, dim, n_heads, xt_T, x_seq);
            }
        }
        xf = g_linear(g,
                      ggml_reshape_2d(g, m.channel_down_w, (int)m.channel_down_w->ne[1], (int)m.channel_down_w->ne[2]),
                      m.channel_down_b, xf);
        xtc = g_linear(
            g, ggml_reshape_2d(g, m.channel_down_t_w, (int)m.channel_down_t_w->ne[1], (int)m.channel_down_t_w->ne[2]),
            m.channel_down_t_b, xtc);
        xf = ggml_cont(g, ggml_transpose(g, xf)); // back to (seq, C)
    }
    ggml_set_name(xtc, "fused_xt_out");
    ggml_set_output(xtc);
    x = ggml_reshape_3d(g, xf, x_T, enc_Fq, (int)(ggml_nelements(xf) / ((size_t)x_T * enc_Fq)));

    // ---------------- decoder ----------------
    for (int idx = 0; idx < hp.depth; idx++) {
        const auto& dec = m.decoder[idx];
        if (!dec.conv_tr_w) {
            ggml_free(g);
            return false;
        }
        ggml_tensor* skip = skips[hp.depth - 1 - idx];
        const int in_Fq = (int)x->ne[1];
        const int stride = hp.stride;
        const int pad = (int)dec.conv_tr_w->ne[1] / 4;

        ggml_tensor* y = x;
        if (!dec.empty) {
            y = ggml_add(g, y, skip);
            if (dec.rewrite_w) {
                ggml_tensor* rw = ggml_conv_2d(g, dec.rewrite_w, y, 1, 1, hp.context, hp.context, 1, 1);
                if (dec.rewrite_b)
                    rw = ggml_add(g, rw,
                                  ggml_reshape_3d(g, htd_bcast_f32(g, dec.rewrite_b), 1, 1, (int)dec.rewrite_w->ne[3]));
                y = g_glu_c(g, rw);
            }
            if (!dec.dconv.layers.empty())
                y = g_dconv(g, y, dec.dconv, in_Fq);
        }

        const int ct_K = (int)dec.conv_tr_w->ne[1];
        const int ct_OC = (int)dec.conv_tr_w->ne[2];
        const int ct_IC = (int)dec.conv_tr_w->ne[3];
        const int fq_raw = (in_Fq - 1) * stride + ct_K;

        ggml_tensor* ZERO = ggml_new_tensor_3d(g, GGML_TYPE_F32, x_T, fq_raw, ct_OC);
        ggml_set_name(ZERO, ("fused_zero_" + std::to_string(idx)).c_str());
        ggml_set_input(ZERO);
        ggml_tensor* acc = ZERO;
        for (int kh = 0; kh < ct_K; kh++) {
            ggml_tensor* Wk = ggml_new_tensor_4d(g, GGML_TYPE_F32, 1, 1, ct_IC, ct_OC);
            ggml_set_name(Wk, ("fused_wk_" + std::to_string(idx) + "_" + std::to_string(kh)).c_str());
            ggml_set_input(Wk);
            ggml_tensor* part = ggml_conv_2d(g, Wk, y, 1, 1, 0, 0, 1, 1);
            acc = ggml_acc(g, acc, part, (size_t)stride * x_T * sizeof(float), (size_t)x_T * fq_raw * sizeof(float),
                           (size_t)x_T * fq_raw * ct_OC * sizeof(float), (size_t)kh * x_T * sizeof(float));
        }
        if (dec.conv_tr_b)
            acc = ggml_add(g, acc, ggml_reshape_3d(g, htd_bcast_f32(g, dec.conv_tr_b), 1, 1, ct_OC));
        x = ggml_cont(g, ggml_view_3d(g, acc, x_T, fq_raw - 2 * pad, ct_OC, acc->nb[1], acc->nb[2],
                                      (size_t)pad * x_T * sizeof(float)));
        if (idx != hp.depth - 1)
            x = ggml_gelu(g, x);
    }

    ggml_set_name(x, "fused_out");
    ggml_set_output(x);

    ggml_cgraph* gf = ggml_new_graph_custom(g, n_nodes, false);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, xtc);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "htdemucs: FUSED graph alloc failed — falling back to per-layer path\n");
        ggml_gallocr_free(alloc);
        ggml_free(g);
        return false;
    }
    return htdemucs_fused_run(ctx, g, gf, alloc, X, XT, x, xtc, x_buf, x_C, x_Fq, x_T, xt_buf, xt_C, xt_T,
                              freq_emb_bcast);
}

// ---------------------------------------------------------------------------
// Parity diff harness (tools/reference_backends/htdemucs.py is the reference)
// ---------------------------------------------------------------------------

namespace {

double htd_cosine(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0 || nb == 0)
        return (na == 0 && nb == 0) ? 1.0 : 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double htd_max_abs_diff(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

double htd_l2_norm(const float* a, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++)
        s += (double)a[i] * a[i];
    return std::sqrt(s);
}

bool htd_ref_get(core_gguf::WeightLoad& rw, const char* name, std::vector<float>& out, int64_t& nelem) {
    auto it = rw.tensors.find(name);
    if (it == rw.tensors.end() || !it->second)
        return false;
    ggml_tensor* t = it->second;
    nelem = ggml_nelements(t);
    out.resize((size_t)nelem);
    ggml_backend_tensor_get(t, out.data(), 0, (size_t)nelem * sizeof(float));
    return true;
}

} // namespace

int htdemucs_diff(const char* model_gguf, const char* ref_gguf, const char* audio_wav, int verbosity) {
    (void)audio_wav; // The waveform is replayed FROM the reference (input-aligned).

    htdemucs_context* ctx = htdemucs_init_from_file(model_gguf, htdemucs_default_params());
    if (!ctx) {
        fprintf(stderr, "htdemucs_diff: failed to load model %s\n", model_gguf);
        return 2;
    }

    core_gguf::WeightLoad rw;
    if (!core_gguf::load_weights(ref_gguf, ctx->backend, "htdemucs_ref", rw)) {
        fprintf(stderr, "htdemucs_diff: failed to load reference %s\n", ref_gguf);
        htdemucs_free(ctx);
        return 2;
    }

    // Replay the exact 44.1 kHz stereo waveform the reference ran on, so a
    // resampler mismatch can never be mistaken for a model parity failure.
    std::vector<float> in_wav;
    int64_t in_n = 0;
    if (!htd_ref_get(rw, "input_wav", in_wav, in_n)) {
        fprintf(stderr, "htdemucs_diff: reference has no input_wav stage — re-dump with the updated dumper\n");
        core_gguf::free_weights(rw);
        htdemucs_free(ctx);
        return 2;
    }
    const int ac = ctx->model.hparams.audio_channels;
    const int n_samples = (int)(in_n / ac);
    std::vector<float> interleaved((size_t)n_samples * ac);
    for (int c = 0; c < ac; c++)
        for (int i = 0; i < n_samples; i++)
            interleaved[(size_t)i * ac + c] = in_wav[(size_t)c * n_samples + i];

    fprintf(stderr, "htdemucs_diff: replaying reference input (%d ch × %d samples)\n", ac, n_samples);

    ctx->capture_stages = true;
    htdemucs_result* res = htdemucs_separate(ctx, interleaved.data(), n_samples);
    if (!res) {
        fprintf(stderr, "htdemucs_diff: separate() failed\n");
        core_gguf::free_weights(rw);
        htdemucs_free(ctx);
        return 2;
    }

    // F32 gate per the dev guide. Stages are compared in reference order so the
    // FIRST failure is the earliest divergence = the bug.
    const double COS_MIN = 0.999;
    int n_fail = 0, n_missing = 0, n_run = 0;
    // std::string, not const char*: the stage names below are built from
    // temporaries like ("enc_freq_" + std::to_string(i)).c_str(), so storing the
    // pointer would dangle and mis-report the first divergence.
    std::string first_fail;

    auto report = [&](const char* stage) {
        std::vector<float> ref;
        int64_t rn = 0;
        if (!htd_ref_get(rw, stage, ref, rn))
            return; // stage not in this dump
        auto it = ctx->captures.find(stage);
        if (it == ctx->captures.end()) {
            fprintf(stderr, "  %-20s MISSING (no C++ capture)\n", stage);
            n_missing++;
            return;
        }
        const std::vector<float>& mine = it->second;
        const int64_t n = (int64_t)std::min(mine.size(), (size_t)rn);
        const double cos = htd_cosine(mine.data(), ref.data(), n);
        const double mad = htd_max_abs_diff(mine.data(), ref.data(), n);
        const bool same_size = mine.size() == (size_t)rn;
        const bool ok = cos >= COS_MIN && same_size;
        n_run++;
        if (!ok) {
            n_fail++;
            if (first_fail.empty())
                first_fail = stage;
        }
        // Always print both magnitudes: a 10-30x outlier on either side means
        // "same name, wrong data" (a harness bug), not a runtime bug.
        fprintf(stderr, "  %-20s %s cos=%.6f max_abs=%.3e  mine=%zu ref=%lld  |mine|=%.4f |ref|=%.4f%s\n", stage,
                ok ? "PASS" : "FAIL", cos, mad, mine.size(), (long long)rn, htd_l2_norm(mine.data(), n),
                htd_l2_norm(ref.data(), n), same_size ? "" : "  *** SHAPE MISMATCH ***");
        (void)verbosity;
    };

    fprintf(stderr, "\n=== htdemucs per-stage parity (cos_min >= %.3f) ===\n", COS_MIN);
    report("spec_input");
    report("time_input");
    // Encoder layer 0 bisection: conv -> gelu -> dconv -> rewrite -> freq_emb.
    report("enc0_conv");
    report("enc0_gelu");
    report("enc0_dconv");
    report("enc0_rewrite");
    for (int i = 0; i < 4; i++)
        report(("enc_freq_" + std::to_string(i)).c_str());
    for (int i = 0; i < 3; i++)
        report(("enc_time_" + std::to_string(i)).c_str());
    report("pre_transformer_z");
    report("pre_transformer_xt");
    // CrossTransformer bisection.
    report("ct_in_z");
    report("ct_in_xt");
    for (int i = 0; i < 5; i++) {
        report(("ct_l" + std::to_string(i) + "_z").c_str());
        report(("ct_l" + std::to_string(i) + "_xt").c_str());
    }
    report("post_transformer_z");
    report("post_transformer_xt");
    for (int i = 0; i < 4; i++)
        report(("dec_freq_" + std::to_string(i)).c_str());
    for (const char* s : {"drums", "bass", "other", "vocals"}) {
        report(("spec_" + std::string(s)).c_str());
        report(("time_" + std::string(s)).c_str());
        report(("output_" + std::string(s)).c_str());
    }

    fprintf(stderr, "\n%d/%d stages passed", n_run - n_fail, n_run);
    if (n_missing)
        fprintf(stderr, ", %d missing", n_missing);
    if (!first_fail.empty())
        fprintf(stderr, " — FIRST DIVERGENCE: %s", first_fail.c_str());
    fprintf(stderr, "\n");

    htdemucs_result_free(res);
    // The reference weights live in a buffer on ctx->backend, so they MUST be
    // released before the backend. Leaking them is invisible on CPU but trips
    // Metal's live-resource assert in the device destructor at exit.
    core_gguf::free_weights(rw);
    htdemucs_free(ctx);
    return n_fail > 0 ? 1 : 0;
}
