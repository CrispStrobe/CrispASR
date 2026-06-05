// outetts_wavtok.cpp -- WavTokenizer decoder (Vocos backbone + iSTFT head).
//
// Architecture:
//   codes (int32, T) -> codebook lookup (4096x512) -> (512, T)
//   Conv1d(512, 768, k=7, pad=3) -> pos_net (ResNet+SelfAttn+GroupNorm)
//   -> AdaNorm (bandwidth_id=0 fixed) -> 12x ConvNeXtBlock(768, 2304)
//   -> final LayerNorm -> Linear(768, 1282) -> split mag/phase
//   -> iSTFT(n_fft=1280, hop=320) -> 24kHz mono PCM

#include "outetts_wavtok.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
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

namespace {

struct wavtok_hp {
    uint32_t input_channels = 512;
    uint32_t backbone_dim = 768;
    uint32_t intermediate_dim = 2304;
    uint32_t n_layers = 12;
    uint32_t adanorm_num = 4;
    uint32_t n_fft = 1280;
    uint32_t hop_length = 320;
    uint32_t sample_rate = 24000;
    uint32_t codebook_size = 4096;
    uint32_t codebook_dim = 512;
};

// ResNet block in pos_net (indices 0,1,3,4)
struct wavtok_resnet_block {
    ggml_tensor* conv1_w = nullptr; // (768, 768, 3)
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr; // (768, 768, 3)
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* norm1_w = nullptr; // (768,)
    ggml_tensor* norm1_b = nullptr;
    ggml_tensor* norm2_w = nullptr;
    ggml_tensor* norm2_b = nullptr;
};

// Self-attention block in pos_net (index 2)
struct wavtok_attn_block {
    ggml_tensor* q_w = nullptr; // (768, 768, 1)
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* proj_out_w = nullptr;
    ggml_tensor* proj_out_b = nullptr;
    ggml_tensor* norm_w = nullptr;
    ggml_tensor* norm_b = nullptr;
};

// ConvNeXt block
struct wavtok_convnext_block {
    ggml_tensor* dw_conv_w = nullptr; // (768, 1, 7)
    ggml_tensor* dw_conv_b = nullptr;
    ggml_tensor* adanorm_scale_w = nullptr; // (4, 768) -- use row 0 at inference
    ggml_tensor* adanorm_shift_w = nullptr; // (4, 768)
    ggml_tensor* pw_up_w = nullptr;         // (2304, 768)
    ggml_tensor* pw_up_b = nullptr;
    ggml_tensor* pw_down_w = nullptr; // (768, 2304)
    ggml_tensor* pw_down_b = nullptr;
    ggml_tensor* grn_gamma = nullptr; // (768,)
};

struct wavtok_model {
    ggml_tensor* codebook_w = nullptr; // (4096, 512)
    ggml_tensor* conv_pre_w = nullptr; // (768, 512, 7)
    ggml_tensor* conv_pre_b = nullptr;

    // Backbone-level AdaNorm (applied before ConvNeXt blocks)
    ggml_tensor* backbone_norm_scale_w = nullptr; // (4, 768)
    ggml_tensor* backbone_norm_shift_w = nullptr;

    // Position network: 4 ResNet blocks + 1 self-attn + 1 GroupNorm
    wavtok_resnet_block pos_resnet[4];  // indices 0,1,3,4
    wavtok_attn_block pos_attn;         // index 2
    ggml_tensor* pos_gnorm_w = nullptr; // (768,)
    ggml_tensor* pos_gnorm_b = nullptr;

    // ConvNeXt blocks
    std::vector<wavtok_convnext_block> blocks;

    // Final norm
    ggml_tensor* final_norm_w = nullptr;
    ggml_tensor* final_norm_b = nullptr;

    // iSTFT head
    ggml_tensor* istft_head_w = nullptr; // (1282, 768)
    ggml_tensor* istft_head_b = nullptr;
    ggml_tensor* istft_window = nullptr; // (1280,)
};

} // namespace

struct wavtok_decoder_ctx {
    wavtok_decoder_params params{};
    wavtok_hp hp;
    wavtok_model model;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    ~wavtok_decoder_ctx() {
        if (ctx_w)
            ggml_free(ctx_w);
        if (buf_w)
            ggml_backend_buffer_free(buf_w);
        if (backend && backend != backend_cpu)
            ggml_backend_free(backend);
        if (backend_cpu)
            ggml_backend_free(backend_cpu);
    }
};

namespace {

static bool bind_model(wavtok_decoder_ctx* c) {
    auto& m = c->model;
    auto& t = c->tensors;

    m.codebook_w = core_gguf::require(t, "wavtok.codebook.weight", "wavtok");
    m.conv_pre_w = core_gguf::require(t, "wavtok.conv_pre.weight", "wavtok");
    m.conv_pre_b = core_gguf::require(t, "wavtok.conv_pre.bias", "wavtok");

    m.backbone_norm_scale_w = core_gguf::try_get(t, "wavtok.backbone_norm.scale.weight");
    m.backbone_norm_shift_w = core_gguf::try_get(t, "wavtok.backbone_norm.shift.weight");

    // Position network: ResNet blocks at 0,1,3,4 and self-attn at 2
    int resnet_idx = 0;
    int posnet_indices[] = {0, 1, 3, 4};
    for (int pi : posnet_indices) {
        auto& rb = m.pos_resnet[resnet_idx++];
        char key[80];
#define POS(fld, sub)                                                                                                  \
    std::snprintf(key, sizeof(key), "wavtok.pos_net.%d." sub, pi);                                                     \
    rb.fld = core_gguf::require(t, key, "wavtok")
        POS(conv1_w, "conv1.weight");
        POS(conv1_b, "conv1.bias");
        POS(conv2_w, "conv2.weight");
        POS(conv2_b, "conv2.bias");
        POS(norm1_w, "norm1.weight");
        POS(norm1_b, "norm1.bias");
        POS(norm2_w, "norm2.weight");
        POS(norm2_b, "norm2.bias");
#undef POS
    }

    // Self-attention at pos_net.2
    auto& pa = m.pos_attn;
    pa.q_w = core_gguf::require(t, "wavtok.pos_net.2.q.weight", "wavtok");
    pa.q_b = core_gguf::require(t, "wavtok.pos_net.2.q.bias", "wavtok");
    pa.k_w = core_gguf::require(t, "wavtok.pos_net.2.k.weight", "wavtok");
    pa.k_b = core_gguf::require(t, "wavtok.pos_net.2.k.bias", "wavtok");
    pa.v_w = core_gguf::require(t, "wavtok.pos_net.2.v.weight", "wavtok");
    pa.v_b = core_gguf::require(t, "wavtok.pos_net.2.v.bias", "wavtok");
    pa.proj_out_w = core_gguf::require(t, "wavtok.pos_net.2.proj_out.weight", "wavtok");
    pa.proj_out_b = core_gguf::require(t, "wavtok.pos_net.2.proj_out.bias", "wavtok");
    pa.norm_w = core_gguf::require(t, "wavtok.pos_net.2.norm.weight", "wavtok");
    pa.norm_b = core_gguf::require(t, "wavtok.pos_net.2.norm.bias", "wavtok");

    // GroupNorm at pos_net.5
    m.pos_gnorm_w = core_gguf::require(t, "wavtok.pos_net.5.weight", "wavtok");
    m.pos_gnorm_b = core_gguf::require(t, "wavtok.pos_net.5.bias", "wavtok");

    // ConvNeXt blocks
    m.blocks.resize(c->hp.n_layers);
    for (uint32_t i = 0; i < c->hp.n_layers; i++) {
        auto& b = m.blocks[i];
        char key[80];
#define BLK(fld, sub)                                                                                                  \
    std::snprintf(key, sizeof(key), "wavtok.block.%u." sub, i);                                                        \
    b.fld = core_gguf::require(t, key, "wavtok")
        BLK(dw_conv_w, "dw_conv.weight");
        BLK(dw_conv_b, "dw_conv.bias");
        BLK(adanorm_scale_w, "adanorm_scale.weight");
        BLK(adanorm_shift_w, "adanorm_shift.weight");
        BLK(pw_up_w, "pw_up.weight");
        BLK(pw_up_b, "pw_up.bias");
        BLK(pw_down_w, "pw_down.weight");
        BLK(pw_down_b, "pw_down.bias");
        BLK(grn_gamma, "grn_gamma");
#undef BLK
    }

    m.final_norm_w = core_gguf::require(t, "wavtok.final_norm.weight", "wavtok");
    m.final_norm_b = core_gguf::require(t, "wavtok.final_norm.bias", "wavtok");

    m.istft_head_w = core_gguf::require(t, "wavtok.istft_head.weight", "wavtok");
    m.istft_head_b = core_gguf::require(t, "wavtok.istft_head.bias", "wavtok");
    m.istft_window = core_gguf::try_get(t, "wavtok.istft_window");

    return true;
}

static void load_metadata(wavtok_decoder_ctx* c, gguf_context* g) {
    auto& hp = c->hp;
    hp.input_channels = core_gguf::kv_u32(g, "wavtok.input_channels", hp.input_channels);
    hp.backbone_dim = core_gguf::kv_u32(g, "wavtok.backbone_dim", hp.backbone_dim);
    hp.intermediate_dim = core_gguf::kv_u32(g, "wavtok.intermediate_dim", hp.intermediate_dim);
    hp.n_layers = core_gguf::kv_u32(g, "wavtok.n_layers", hp.n_layers);
    hp.adanorm_num = core_gguf::kv_u32(g, "wavtok.adanorm_num_embeddings", hp.adanorm_num);
    hp.n_fft = core_gguf::kv_u32(g, "wavtok.n_fft", hp.n_fft);
    hp.hop_length = core_gguf::kv_u32(g, "wavtok.hop_length", hp.hop_length);
    hp.sample_rate = core_gguf::kv_u32(g, "wavtok.sample_rate", hp.sample_rate);
    hp.codebook_size = core_gguf::kv_u32(g, "wavtok.codebook_size", hp.codebook_size);
    hp.codebook_dim = core_gguf::kv_u32(g, "wavtok.codebook_dim", hp.codebook_dim);
}

// ---------------------------------------------------------------------------
// iSTFT (CPU post-processing)
// ---------------------------------------------------------------------------

// Direct inverse RFFT: given n_freq = N/2+1 complex bins (conjugate-symmetric
// for a real signal), compute the N-point real IFFT. Works for any N, not
// just powers of 2.
//
// x[n] = (1/N) * ( X[0].re
//                 + X[N/2].re * (-1)^n    [if N even]
//                 + 2 * sum_{k=1}^{N/2-1} Re(X[k] * exp(j*2*pi*k*n/N)) )
static void irfft_direct(const float* spec_re, const float* spec_im, int N, float* out) {
    const int half = N / 2;
    const double inv_N = 1.0 / N;

    // Precompute twiddle factors for k=1..half-1
    // (could optimize with Goertzel or trig recurrence, but N=1280 is small)
    for (int n = 0; n < N; n++) {
        double val = spec_re[0]; // DC (always real)
        if (N % 2 == 0) {
            // Nyquist bin: X[N/2] is real for real signals
            val += spec_re[half] * ((n & 1) ? -1.0 : 1.0);
        }
        double angle_base = 2.0 * M_PI * n * inv_N;
        for (int k = 1; k < half; k++) {
            double angle = angle_base * k;
            // Re(X[k] * exp(j*angle)) = Re_k * cos - Im_k * sin
            val += 2.0 * (spec_re[k] * std::cos(angle) - spec_im[k] * std::sin(angle));
        }
        out[n] = (float)(val * inv_N);
    }
}

// iSTFT: (n_freq, T_frames) magnitude + phase -> PCM samples.
// n_freq = n_fft/2 + 1 = 641
static std::vector<float> istft_cpu(const float* mag, const float* phase, int n_freq, int T_frames, int n_fft, int hop,
                                    const float* window) {
    const int out_len = (T_frames - 1) * hop + n_fft;

    std::vector<float> output(out_len, 0.0f);
    std::vector<float> win_sum(out_len, 0.0f);
    std::vector<float> frame_re(n_freq);
    std::vector<float> frame_im(n_freq);
    std::vector<float> frame_out(n_fft);

    // Build window if not provided
    std::vector<float> hann;
    if (!window) {
        hann.resize(n_fft);
        for (int i = 0; i < n_fft; i++) {
            hann[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / n_fft));
        }
        window = hann.data();
    }

    for (int t = 0; t < T_frames; t++) {
        // Build complex spectrum from magnitude + phase
        for (int f = 0; f < n_freq; f++) {
            float m = mag[t * n_freq + f];
            float p = phase[t * n_freq + f];
            frame_re[f] = m * std::cos(p);
            frame_im[f] = m * std::sin(p);
        }

        // Inverse RFFT (exact N-point, no zero-padding)
        irfft_direct(frame_re.data(), frame_im.data(), n_fft, frame_out.data());

        // Overlap-add with window
        int offset = t * hop;
        for (int i = 0; i < n_fft && (offset + i) < out_len; i++) {
            float w = window[i];
            output[offset + i] += frame_out[i] * w;
            win_sum[offset + i] += w * w;
        }
    }

    // Normalize by window sum
    for (int i = 0; i < out_len; i++) {
        if (win_sum[i] > 1e-8f) {
            output[i] /= win_sum[i];
        }
    }

    // WavTokenizer uses padding="same": trim (win_length - hop_length) / 2 from each end
    // (not center=True which would trim n_fft/2)
    int trim = (n_fft - hop) / 2;
    if ((int)output.size() > 2 * trim) {
        output.erase(output.begin(), output.begin() + trim);
        output.resize(output.size() - trim);
    }

    // Clamp
    for (auto& s : output) {
        s = std::max(-1.0f, std::min(1.0f, s));
    }

    return output;
}

// ---------------------------------------------------------------------------
// Backbone forward (ggml graph build + compute)
// ---------------------------------------------------------------------------

// Ensure tensor is F32 for binary ops (ggml add/mul don't support f32+f16).
static ggml_tensor* ensure_f32(ggml_context* ctx, ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32) {
        return ggml_cast(ctx, t, GGML_TYPE_F32);
    }
    return t;
}

// Standard Conv1d: x (C_in, T), w (K, C_in, C_out) -> (C_out, T_out)
// Pattern from orpheus_snac.cpp:conv1d_k (battle-tested).
static ggml_tensor* conv1d_k(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int pad) {
    // Transpose [C_in, T] -> [T, C_in]
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* y = ggml_conv_1d(ctx, w, xt, 1, pad, 1);
    // conv_1d returns 3D (OL, OC, N) -> flatten and transpose to (C_out, T_out)
    if (ggml_n_dims(y) > 2) {
        y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1] * y->ne[2]);
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (C_out, T_out)
    if (b) {
        y = ggml_add(ctx, y, ensure_f32(ctx, b));
    }
    return y;
}

// Depthwise Conv1d: x (C, T), w (K, 1, C) -> (C, T)
// Pattern from vibevoice.cpp:392 (battle-tested).
static ggml_tensor* dw_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int pad) {
    // Transpose [C, T] -> [T, C]
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    x = ggml_conv_1d_dw(ctx, w, x, 1, pad, 1);
    // conv_1d_dw returns 3D+ -> flatten then transpose back to [C, T]
    if (ggml_n_dims(x) > 2) {
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    }
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (b) {
        x = ggml_add(ctx, x, ensure_f32(ctx, b));
    }
    return x;
}

// LayerNorm along channel dim: x (C, T) -> (C, T)
// ggml_norm normalizes along ne[0] which is C — exactly what LayerNorm(C) does.
static ggml_tensor* layer_norm_channels(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w) {
        x = ggml_mul(ctx, x, ensure_f32(ctx, w));
    }
    if (b) {
        x = ggml_add(ctx, x, ensure_f32(ctx, b));
    }
    return x;
}

// GroupNorm along channel dim: x (C, T) -> (C, T)
// ggml_group_norm normalizes along ne[0] in groups.
// For 32 groups with C=768: each group has 768/32 = 24 channels.
static ggml_tensor* group_norm_channels(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int n_groups,
                                        float eps) {
    x = ggml_group_norm(ctx, x, n_groups, eps);
    if (w)
        x = ggml_mul(ctx, x, ensure_f32(ctx, w));
    if (b)
        x = ggml_add(ctx, x, ensure_f32(ctx, b));
    return x;
}

// ResNet block (pos_net): GroupNorm(32) -> GELU -> Conv1d -> GroupNorm(32) -> GELU -> Conv1d -> residual
// Matches the VocosBackbone ResnetBlock (NOT HiFi-GAN ResBlock1).
// NOTE: upstream WavTokenizer uses GELU, not SiLU. Previously this
// used ggml_silu which subtly degraded audio quality.
static ggml_tensor* resnet_block(ggml_context* ctx, ggml_tensor* x, const wavtok_resnet_block& rb) {
    ggml_tensor* residual = x;
    x = group_norm_channels(ctx, x, rb.norm1_w, rb.norm1_b, 32, 1e-6f);
    x = ggml_gelu(ctx, x);
    x = conv1d_k(ctx, x, rb.conv1_w, rb.conv1_b, 1);
    x = group_norm_channels(ctx, x, rb.norm2_w, rb.norm2_b, 32, 1e-6f);
    x = ggml_gelu(ctx, x);
    // dropout(0.1) skipped at inference
    x = conv1d_k(ctx, x, rb.conv2_w, rb.conv2_b, 1);
    return ggml_add(ctx, x, residual);
}

// Self-attention (1x1 conv Q/K/V): x (C, T) -> (C, T)
// Norm uses GroupNorm(32) like ResnetBlock.
static ggml_tensor* self_attn_block(ggml_context* ctx, ggml_tensor* x, const wavtok_attn_block& ab) {
    ggml_tensor* residual = x;
    x = group_norm_channels(ctx, x, ab.norm_w, ab.norm_b, 32, 1e-6f);

    // Q, K, V via 1x1 conv (equivalent to linear along channels)
    ggml_tensor* q = conv1d_k(ctx, x, ab.q_w, ab.q_b, 0);
    ggml_tensor* k = conv1d_k(ctx, x, ab.k_w, ab.k_b, 0);
    ggml_tensor* v = conv1d_k(ctx, x, ab.v_w, ab.v_b, 0);

    // Scaled dot-product attention: q,k,v are (C, T)
    int C = (int)q->ne[0];
    float scale = 1.0f / std::sqrt((float)C);
    // mul_mat(k, q) with k (C, T) and q (C, T) gives (T, T)
    ggml_tensor* attn = ggml_mul_mat(ctx, k, q);
    attn = ggml_scale(ctx, attn, scale);
    attn = ggml_soft_max(ctx, attn);
    // attn (T, T) @ v^T (T, C) -> (C, T) ... use mul_mat(v, attn)
    // Actually: mul_mat transposes its first arg, so mul_mat(v, attn) with
    // v (C, T) and attn (T, T) -> (C, T). That's correct.
    ggml_tensor* out = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), attn);

    // Output projection
    out = conv1d_k(ctx, out, ab.proj_out_w, ab.proj_out_b, 0);
    return ggml_add(ctx, out, residual);
}

// AdaNorm: normalize x, then scale/shift using bandwidth_id=0 row
// scale_w (C, 4), shift_w (C, 4) in ggml layout (ne[0]=C, ne[1]=4) -> use row 0
static ggml_tensor* ada_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale_w, ggml_tensor* shift_w) {
    int C = (int)x->ne[0];
    // Extract row 0 (bandwidth_id=0): view of ne[0]=C elements at offset 0
    // Cast to F32 since the scale/shift weights may be F16.
    ggml_tensor* scale = ggml_view_1d(ctx, scale_w, C, 0);
    ggml_tensor* shift = ggml_view_1d(ctx, shift_w, C, 0);
    if (scale->type != GGML_TYPE_F32) {
        scale = ggml_cast(ctx, scale, GGML_TYPE_F32);
    }
    if (shift->type != GGML_TYPE_F32) {
        shift = ggml_cast(ctx, shift, GGML_TYPE_F32);
    }

    // ggml_norm normalizes along ne[0]=C — correct for LayerNorm(C)
    x = ggml_norm(ctx, x, 1e-6f);
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, shift);
    return x;
}

// Helper: name a tensor as a stage dump point. After compute, named tensors
// can be read back and dumped to disk.
static ggml_tensor* mark_stage(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* x, const char* name) {
    ggml_tensor* out = ggml_cpy(ctx, x, ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, x->ne));
    ggml_set_name(out, name);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    return x; // return original x so we don't break the data flow
}

// Dump a tensor to a binary file (flat F32).
// ggml stores ne[0] contiguously, so a 2D (ne[0]=C, ne[1]=T) tensor is laid
// out as [T][C] in C row-major. The Python reference uses shape (1, C, T) with
// T contiguous within each channel → [C][T] in row-major. We transpose 2D
// tensors on dump so comparisons are element-wise correct.
static void dump_tensor(ggml_backend_t /*backend*/, ggml_cgraph* gf, const char* name, const char* dump_dir) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name);
    if (!t) {
        fprintf(stderr, "wavtok dump: tensor '%s' not found\n", name);
        return;
    }
    size_t n = ggml_nelements(t);
    std::vector<float> raw(n);
    ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(float));

    // Transpose 2D: ggml (ne0, ne1) stored as [ne1][ne0] → flip to [ne0][ne1]
    std::vector<float> data(n);
    int dims = ggml_n_dims(t);
    if (dims == 2) {
        int ne0 = (int)t->ne[0];
        int ne1 = (int)t->ne[1];
        for (int i1 = 0; i1 < ne1; i1++) {
            for (int i0 = 0; i0 < ne0; i0++) {
                data[i0 * ne1 + i1] = raw[i1 * ne0 + i0];
            }
        }
    } else {
        data = raw;
    }

    char path[512];
    std::snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "wavtok dump: cannot open '%s'\n", path);
        return;
    }
    fwrite(data.data(), sizeof(float), n, f);
    fclose(f);
    float mn = *std::min_element(data.begin(), data.end());
    float mx = *std::max_element(data.begin(), data.end());
    fprintf(stderr, "  dump %s: %zu elems [%d dims: %lld x %lld], range [%.6f, %.6f]\n", name, n, dims,
            (long long)t->ne[0], (long long)t->ne[1], mn, mx);
}

// ConvNeXt block: dw_conv -> adanorm -> pw_up -> gelu -> grn -> pw_down -> residual
// All tensors kept in (C, T) layout throughout.
static ggml_tensor* convnext_block(ggml_context* ctx, ggml_tensor* x, const wavtok_convnext_block& b) {
    ggml_tensor* residual = x;

    // Depthwise conv: (C, T) -> (C, T)
    x = dw_conv1d(ctx, x, b.dw_conv_w, b.dw_conv_b, 3);

    // AdaNorm (using bandwidth_id=0): (C, T) -> (C, T)
    x = ada_norm(ctx, x, b.adanorm_scale_w, b.adanorm_shift_w);

    // Pointwise up: (C, T) -> (intermediate, T)
    // ggml_mul_mat(a, b): a ne[0]=C, ne[1]=intermediate; b ne[0]=C, ne[1]=T
    // result ne[0]=intermediate, ne[1]=T
    x = ggml_mul_mat(ctx, b.pw_up_w, x);
    if (b.pw_up_b) {
        x = ggml_add(ctx, x, ensure_f32(ctx, b.pw_up_b));
    }
    x = ggml_gelu(ctx, x);

    // Pointwise down: (intermediate, T) -> (C, T)
    x = ggml_mul_mat(ctx, b.pw_down_w, x);
    if (b.pw_down_b) {
        x = ggml_add(ctx, x, ensure_f32(ctx, b.pw_down_b));
    }

    // Gamma: per-channel residual gate (Vocos convention). gamma is (C,).
    if (b.grn_gamma) {
        x = ggml_mul(ctx, x, ensure_f32(ctx, b.grn_gamma));
    }

    return ggml_add(ctx, x, residual);
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" struct wavtok_decoder_params wavtok_decoder_default_params(void) {
    wavtok_decoder_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = 0;
    p.dump_dir = nullptr;
    return p;
}

extern "C" struct wavtok_decoder_ctx* wavtok_decoder_init_from_file(const char* path,
                                                                    struct wavtok_decoder_params params) {
    auto* c = new wavtok_decoder_ctx();
    c->params = params;

    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            delete c;
            return nullptr;
        }
        load_metadata(c, g);
        core_gguf::free_metadata(g);
    }

    if (params.verbosity >= 1) {
        fprintf(stderr, "wavtok: dim=%u layers=%u n_fft=%u hop=%u codebook=%ux%u\n", c->hp.backbone_dim, c->hp.n_layers,
                c->hp.n_fft, c->hp.hop_length, c->hp.codebook_size, c->hp.codebook_dim);
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, params.n_threads);
    c->backend = c->backend_cpu; // decoder is small, CPU is fine

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "wavtokenizer", wl)) {
        fprintf(stderr, "wavtok: failed to load weights from '%s'\n", path);
        delete c;
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!bind_model(c)) {
        fprintf(stderr, "wavtok: tensor binding failed\n");
        delete c;
        return nullptr;
    }

    return c;
}

extern "C" float* wavtok_decoder_decode(struct wavtok_decoder_ctx* ctx, const int32_t* codes, int n_codes,
                                        int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !codes || n_codes <= 0)
        return nullptr;

    const auto& hp = ctx->hp;
    const auto& m = ctx->model;
    const int n_fft = (int)hp.n_fft;
    const int hop = (int)hp.hop_length;
    const int n_freq = n_fft / 2 + 1; // 641

    // Env var overrides for diff testing
    const char* env_dump = std::getenv("WAVTOK_DUMP_DIR");
    const char* dump_dir_eff = ctx->params.dump_dir ? ctx->params.dump_dir : env_dump;
    const bool dumping = dump_dir_eff != nullptr;

    // Allow overriding input codes for diff testing (e.g. "0,1,2,3,4,5,6,7,8,9")
    std::vector<int32_t> fixed_codes;
    const char* env_codes = std::getenv("WAVTOK_FIXED_CODES");
    if (env_codes && env_codes[0]) {
        std::string sc(env_codes);
        size_t pos = 0;
        while (pos < sc.size()) {
            size_t comma = sc.find(',', pos);
            if (comma == std::string::npos)
                comma = sc.size();
            fixed_codes.push_back(std::atoi(sc.substr(pos, comma - pos).c_str()));
            pos = comma + 1;
        }
        codes = fixed_codes.data();
        n_codes = (int)fixed_codes.size();
        fprintf(stderr, "wavtok: FIXED_CODES override: %d codes\n", n_codes);
    }

    const int T = n_codes;

    // Build and compute the backbone graph
    const size_t meta_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params ip = {meta_size, meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // Input: codes (T,) I32
    ggml_tensor* codes_in = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(codes_in, "codes_in");
    ggml_set_input(codes_in);

    // Codebook lookup: (codebook_dim, T)
    ggml_tensor* x = ggml_get_rows(ctx0, m.codebook_w, codes_in);
    // x is (codebook_dim, T) = (512, T)

    // Conv_pre: (512, T) -> (768, T)
    x = ggml_cont(ctx0, ggml_cast(ctx0, x, GGML_TYPE_F32));
    if (dumping)
        mark_stage(ctx0, gf, x, "01_codebook");
    x = conv1d_k(ctx0, x, m.conv_pre_w, m.conv_pre_b, 3);
    if (dumping)
        mark_stage(ctx0, gf, x, "02_conv_pre");

    // Position network: ResNet(0) -> ResNet(1) -> SelfAttn(2) -> ResNet(3) -> ResNet(4) -> GroupNorm(5)
    // Note: pos_net comes BEFORE AdaNorm (matching VocosBackbone.forward order)
    x = resnet_block(ctx0, x, m.pos_resnet[0]);
    if (dumping)
        mark_stage(ctx0, gf, x, "04_posnet_resnet0");
    x = resnet_block(ctx0, x, m.pos_resnet[1]);
    if (dumping)
        mark_stage(ctx0, gf, x, "04_posnet_resnet1");
    x = self_attn_block(ctx0, x, m.pos_attn);
    if (dumping)
        mark_stage(ctx0, gf, x, "04_posnet_selfattn2");
    x = resnet_block(ctx0, x, m.pos_resnet[2]);
    if (dumping)
        mark_stage(ctx0, gf, x, "04_posnet_resnet3");
    x = resnet_block(ctx0, x, m.pos_resnet[3]);
    if (dumping)
        mark_stage(ctx0, gf, x, "04_posnet_resnet4");

    // GroupNorm at pos_net.5 (num_groups=32, eps=1e-6)
    x = group_norm_channels(ctx0, x, m.pos_gnorm_w, m.pos_gnorm_b, 32, 1e-6f);
    if (dumping)
        mark_stage(ctx0, gf, x, "05_posnet_gnorm");

    // Backbone-level AdaNorm (AFTER pos_net, before ConvNeXt blocks)
    if (m.backbone_norm_scale_w && m.backbone_norm_shift_w) {
        x = ada_norm(ctx0, x, m.backbone_norm_scale_w, m.backbone_norm_shift_w);
    }
    if (dumping)
        mark_stage(ctx0, gf, x, "03_backbone_adanorm");

    // 12x ConvNeXt blocks
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        x = convnext_block(ctx0, x, m.blocks[i]);
        if (dumping && (i == 0 || i == 11)) {
            char sname[32];
            std::snprintf(sname, sizeof(sname), "06_convnext_%u", i);
            mark_stage(ctx0, gf, x, sname);
        }
    }

    // Final LayerNorm
    x = layer_norm_channels(ctx0, x, m.final_norm_w, m.final_norm_b, 1e-6f);
    if (dumping)
        mark_stage(ctx0, gf, x, "07_final_norm");

    // iSTFT head: Linear(768, 1282)
    // x is (C=768, T). mul_mat(head_w, x) with head_w ne[0]=768, ne[1]=1282
    // gives (1282, T).
    x = ggml_mul_mat(ctx0, m.istft_head_w, x);
    if (m.istft_head_b) {
        x = ggml_add(ctx0, x, ensure_f32(ctx0, m.istft_head_b));
    }

    ggml_set_name(x, "stft_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    // Allocate and compute
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "wavtok: graph alloc failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "codes_in"), codes, 0, T * sizeof(int32_t));

    if (ggml_backend_graph_compute(ctx->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "wavtok: graph compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return nullptr;
    }

    // Dump stages if requested
    if (dumping) {
        const char* dd = dump_dir_eff;
        dump_tensor(ctx->backend, gf, "01_codebook", dd);
        dump_tensor(ctx->backend, gf, "02_conv_pre", dd);
        dump_tensor(ctx->backend, gf, "03_backbone_adanorm", dd);
        dump_tensor(ctx->backend, gf, "04_posnet_resnet0", dd);
        dump_tensor(ctx->backend, gf, "04_posnet_resnet1", dd);
        dump_tensor(ctx->backend, gf, "04_posnet_selfattn2", dd);
        dump_tensor(ctx->backend, gf, "04_posnet_resnet3", dd);
        dump_tensor(ctx->backend, gf, "04_posnet_resnet4", dd);
        dump_tensor(ctx->backend, gf, "05_posnet_gnorm", dd);
        dump_tensor(ctx->backend, gf, "06_convnext_0", dd);
        dump_tensor(ctx->backend, gf, "06_convnext_11", dd);
        dump_tensor(ctx->backend, gf, "07_final_norm", dd);
        // stft_out renamed to match Python's 08_istft_head
        dump_tensor(ctx->backend, gf, "stft_out", dd);
        // Also save as 08_istft_head for compare script
        {
            ggml_tensor* st = ggml_graph_get_tensor(gf, "stft_out");
            if (st) {
                int ne0 = (int)st->ne[0], ne1 = (int)st->ne[1];
                size_t n = (size_t)ne0 * ne1;
                std::vector<float> raw(n), data(n);
                ggml_backend_tensor_get(st, raw.data(), 0, n * sizeof(float));
                // Transpose to (T, 1282) to match Python's (10, 1282)
                for (int i1 = 0; i1 < ne1; i1++)
                    for (int i0 = 0; i0 < ne0; i0++)
                        data[i1 * ne0 + i0] = raw[i1 * ne0 + i0];
                // stft_out ggml is already (1282, T) with ne[0]=1282 contiguous
                // So raw is [T][1282] in row-major which IS (T, 1282) — no transpose needed
                char path[512];
                std::snprintf(path, sizeof(path), "%s/08_istft_head.bin", dd);
                FILE* f = fopen(path, "wb");
                if (f) {
                    fwrite(raw.data(), sizeof(float), n, f);
                    fclose(f);
                }
            }
        }
    }

    // Read back STFT output: (1282, T)
    ggml_tensor* stft_out = ggml_graph_get_tensor(gf, "stft_out");
    const int stft_dim = (int)stft_out->ne[0]; // 1282
    const int T_out = (int)stft_out->ne[1];
    std::vector<float> stft_data(stft_dim * T_out);
    ggml_backend_tensor_get(stft_out, stft_data.data(), 0, stft_data.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);

    // Split into magnitude and phase
    // Output layout: first n_freq values are magnitude, next n_freq are phase
    // (1282 = 2 * 641)
    std::vector<float> mag(n_freq * T_out);
    std::vector<float> phase(n_freq * T_out);
    for (int t = 0; t < T_out; t++) {
        for (int f = 0; f < n_freq; f++) {
            // Vocos ISTFTHead: first half is magnitude (exp for positive), second half is phase angle
            float m_raw = stft_data[t * stft_dim + f];
            float p_raw = stft_data[t * stft_dim + n_freq + f];
            mag[t * n_freq + f] =
                std::min(std::exp(m_raw), 1e2f); // log-magnitude -> magnitude, clipped (ISTFTHead safeguard)
            phase[t * n_freq + f] = p_raw;       // direct phase angle
        }
    }

    // Read window if available
    std::vector<float> window;
    if (ctx->model.istft_window) {
        window.resize(n_fft);
        ggml_backend_tensor_get(ctx->model.istft_window, window.data(), 0, n_fft * sizeof(float));
    }

    // iSTFT
    std::vector<float> pcm =
        istft_cpu(mag.data(), phase.data(), n_freq, T_out, n_fft, hop, window.empty() ? nullptr : window.data());

    // Dump mag, phase, audio if requested
    if (dumping) {
        auto dump_vec = [&](const char* fname, const float* d, size_t n) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%s.bin", dump_dir_eff, fname);
            FILE* f = fopen(path, "wb");
            if (f) {
                fwrite(d, sizeof(float), n, f);
                fclose(f);
            }
            float mn = *std::min_element(d, d + n);
            float mx = *std::max_element(d, d + n);
            fprintf(stderr, "  dump %s: %zu elems, range [%.6f, %.6f]\n", fname, n, mn, mx);
        };
        dump_vec("09_mag", mag.data(), mag.size());
        dump_vec("09_phase", phase.data(), phase.size());
        dump_vec("10_audio", pcm.data(), pcm.size());
    }

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "wavtok: decoded %d codes -> %d samples (%.2f s at %u Hz)\n", n_codes, (int)pcm.size(),
                (float)pcm.size() / hp.sample_rate, hp.sample_rate);
    }

    float* out = (float*)malloc(pcm.size() * sizeof(float));
    if (!out)
        return nullptr;
    std::memcpy(out, pcm.data(), pcm.size() * sizeof(float));
    if (out_n_samples)
        *out_n_samples = (int)pcm.size();
    return out;
}

extern "C" void wavtok_decoder_free(struct wavtok_decoder_ctx* ctx) {
    delete ctx;
}
