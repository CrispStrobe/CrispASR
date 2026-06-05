// silero_lid.cpp — native ggml runtime for Silero Language Classifier 95.
//
// Architecture: 8 stage pairs (conv + transformer) + attention pooling + 2 classifiers.
// Input: raw 16 kHz mono PCM audio. Output: 95-language log-probabilities.
//
// The model processes raw audio (not mel features) through a MobileNet-style
// depthwise-separable conv encoder, interleaved with small transformer blocks
// that mix information across the time axis. A learned attention-weighted pool
// collapses the time dimension, and two linear classifiers emit per-language
// and per-language-group log-probabilities.

#include "silero_lid.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "core/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Model structures
// ===========================================================================

struct lid_conv_block {
    ggml_tensor *dw_w = nullptr, *dw_b = nullptr;     // (C, 1, K)
    ggml_tensor *pw_w = nullptr, *pw_b = nullptr;     // (Cout, Cin, 1)
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr; // optional residual proj
};

struct lid_tx_block {
    // Named biases + norms
    ggml_tensor* qkv_b = nullptr; // (3*dim,)
    ggml_tensor* out_b = nullptr; // (dim,)
    ggml_tensor* ff1_b = nullptr; // (dim,)
    ggml_tensor* ff2_b = nullptr; // (dim,)
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    // Numeric weights
    ggml_tensor* qkv_w = nullptr; // (dim, 3*dim)
    ggml_tensor* out_w = nullptr; // (dim, dim)
    ggml_tensor* ff1_w = nullptr; // (dim, dim)
    ggml_tensor* ff2_w = nullptr; // (dim, dim)
    // 1×1 conv projection at stage boundary
    ggml_tensor *conv1x1_w = nullptr, *conv1x1_b = nullptr;
};

struct lid_stage {
    int dim;
    std::vector<lid_conv_block> conv_blocks; // 12 per stage
    lid_tx_block tx;
};

struct lid_model {
    // Front-end: learned Conv1d(1→322, kernel=320, stride=160)
    ggml_tensor* frontend_w = nullptr; // (322, 1, 320)
    int frontend_stride = 160;
    int frontend_kernel = 320;
    int frontend_channels = 322;
    int n_downsample_stages = 4; // stride-2 after each of the first 4 stages

    std::vector<lid_stage> stages;                      // 8
    ggml_tensor* adaptive_norm_filter = nullptr;        // (1, 1, 17)
    ggml_tensor* pool_weight = nullptr;                 // (192,)
    ggml_tensor *lang_w = nullptr, *lang_b = nullptr;   // (95, 192)
    ggml_tensor *group_w = nullptr, *group_b = nullptr; // (58, 192)

    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

struct silero_lid_context {
    lid_model model;
    std::vector<std::string> lang_strs;  // 95 entries
    std::vector<std::string> group_strs; // 58 entries
    int n_threads = 4;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

// ===========================================================================
// Loader
// ===========================================================================

static ggml_tensor* lid_get(lid_model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    return it != m.tensors.end() ? it->second : nullptr;
}

static bool lid_load(lid_model& m, const char* path, ggml_backend_t backend) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "silero_lid", wl))
        return false;
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.tensors = std::move(wl.tensors);

    // Bind top-level
    m.frontend_w = lid_get(m, "lid.frontend.weight");
    m.adaptive_norm_filter = lid_get(m, "lid.adaptive_norm.filter");
    m.pool_weight = lid_get(m, "lid.pool.weight");
    m.lang_w = lid_get(m, "lid.lang.weight");
    m.lang_b = lid_get(m, "lid.lang.bias");
    m.group_w = lid_get(m, "lid.group.weight");
    m.group_b = lid_get(m, "lid.group.bias");

    // 8 stages
    int dims[] = {128, 128, 128, 128, 192, 192, 192, 192};
    m.stages.resize(8);
    for (int si = 0; si < 8; si++) {
        auto& st = m.stages[si];
        st.dim = dims[si];
        st.conv_blocks.resize(12);
        for (int bi = 0; bi < 12; bi++) {
            char buf[128];
            auto& cb = st.conv_blocks[bi];
            snprintf(buf, sizeof(buf), "lid.conv.%d.%d.dw_conv.weight", si, bi);
            cb.dw_w = lid_get(m, buf);
            snprintf(buf, sizeof(buf), "lid.conv.%d.%d.dw_conv.bias", si, bi);
            cb.dw_b = lid_get(m, buf);
            snprintf(buf, sizeof(buf), "lid.conv.%d.%d.pw_conv.weight", si, bi);
            cb.pw_w = lid_get(m, buf);
            snprintf(buf, sizeof(buf), "lid.conv.%d.%d.pw_conv.bias", si, bi);
            cb.pw_b = lid_get(m, buf);
            snprintf(buf, sizeof(buf), "lid.conv.%d.%d.proj.weight", si, bi);
            cb.proj_w = lid_get(m, buf);
            snprintf(buf, sizeof(buf), "lid.conv.%d.%d.proj.bias", si, bi);
            cb.proj_b = lid_get(m, buf);
        }
        auto& tx = st.tx;
        char buf[128];
        snprintf(buf, sizeof(buf), "lid.%d.tx.qkv.weight", si);
        tx.qkv_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.qkv.bias", si);
        tx.qkv_b = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.out.weight", si);
        tx.out_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.out.bias", si);
        tx.out_b = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.ff1.weight", si);
        tx.ff1_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.ff1.bias", si);
        tx.ff1_b = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.ff2.weight", si);
        tx.ff2_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.ff2.bias", si);
        tx.ff2_b = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.norm1.weight", si);
        tx.norm1_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.norm1.bias", si);
        tx.norm1_b = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.norm2.weight", si);
        tx.norm2_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.norm2.bias", si);
        tx.norm2_b = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.conv1x1.weight", si);
        tx.conv1x1_w = lid_get(m, buf);
        snprintf(buf, sizeof(buf), "lid.%d.tx.conv1x1.bias", si);
        tx.conv1x1_b = lid_get(m, buf);
    }

    if (!m.lang_w || !m.pool_weight) {
        fprintf(stderr, "silero_lid: missing critical tensors\n");
        return false;
    }
    return true;
}

// ===========================================================================
// Forward pass (manual F32, CPU-only for the small 17 MB model)
// ===========================================================================

// Depthwise-separable conv1d with residual:
//   dw_conv(k=5) + bias → ReLU → pw_conv(k=1) + bias + residual_input → ReLU
// The ONNX graph (confirmed by tracing Conv_73→Relu_74→Conv_75→Add_76→Relu_77)
// adds the ORIGINAL block input as a skip connection after the pointwise conv.
// When C_in != C_out, the residual is projected via the block's `proj` weight.
static void dw_sep_conv1d(const float* in, int C_in, int T_in, const float* dw_w, const float* dw_b, int K,
                          const float* pw_w, const float* pw_b, int C_out, float* out, bool add_residual = true) {
    int pad = K / 2;
    // Depthwise: [C_in, 1, K] with same padding
    std::vector<float> dw_out(C_in * T_in, 0.f);
    for (int c = 0; c < C_in; c++) {
        for (int t = 0; t < T_in; t++) {
            float sum = dw_b[c];
            for (int k = 0; k < K; k++) {
                int ti = t + k - pad;
                if (ti >= 0 && ti < T_in)
                    sum += dw_w[c * K + k] * in[c * T_in + ti];
            }
            dw_out[c * T_in + t] = std::max(0.f, sum); // ReLU
        }
    }
    // Pointwise: [C_out, C_in, 1] + optional residual + ReLU
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < T_in; t++) {
            float sum = pw_b[co];
            for (int ci = 0; ci < C_in; ci++)
                sum += pw_w[co * C_in + ci] * dw_out[ci * T_in + t];
            if (add_residual && C_in == C_out && co < C_in) {
                sum += in[co * T_in + t];
            }
            // ReLU only when we're doing the normal residual path.
            // For blocks with proj, the caller adds proj + ReLU.
            out[co * T_in + t] = add_residual ? std::max(0.f, sum) : sum;
        }
    }
}

// Layer norm over the channel dimension (C fastest, T slow) → [C, T]
static void layer_norm_ct(float* data, int C, int T, const float* w, const float* b, float eps = 1e-5f) {
    for (int t = 0; t < T; t++) {
        float sum = 0, sq = 0;
        for (int c = 0; c < C; c++) {
            float v = data[c * T + t];
            sum += v;
            sq += v * v;
        }
        float mean = sum / C;
        float var = sq / C - mean * mean;
        float inv_std = 1.f / sqrtf(var + eps);
        for (int c = 0; c < C; c++) {
            data[c * T + t] = (data[c * T + t] - mean) * inv_std * w[c] + b[c];
        }
    }
}

// Simple self-attention: Q/K/V from combined QKV, output projection, residual
static void self_attention(float* x, int D, int T, const float* qkv_w, const float* qkv_b, const float* out_w,
                           const float* out_b) {
    // x is [D, T] in channel-first layout.
    // Transpose to [T, D] for matmul.
    std::vector<float> xt(T * D);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            xt[t * D + d] = x[d * T + t];

    // QKV = xt @ qkv_w + qkv_b  → [T, 3D]
    // ONNX MatMul: output = input @ weight, weight shape (D, 3D).
    // GGUF stores (D, 3D) as ne=(3D, D) → data[d * 3D + j] = weight[d, j].
    std::vector<float> qkv(T * 3 * D);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < 3 * D; j++) {
            float sum = qkv_b[j];
            for (int d = 0; d < D; d++)
                sum += xt[t * D + d] * qkv_w[d * 3 * D + j];
            qkv[t * 3 * D + j] = sum;
        }
    }

    // ONNX slice order is K [0:D], Q [D:2D], V [2D:3D] (confirmed by
    // intermediate dump: tensor 746=K, 749=Q, 752=V).
    auto K = qkv.data();
    auto Q = qkv.data() + D;
    auto V = qkv.data() + 2 * D;
    int stride_qkv = 3 * D;

    // Multi-head attention: n_heads=2, head_dim=64 for D=128.
    // ONNX Div_196 scale = 8.0 = sqrt(64) → 1/scale = 1/8 = 0.125.
    // (Not 1/sqrt(D=128) = 0.088 which is what single-head would use.)
    const int n_heads = 2;
    const int head_dim = D / n_heads;           // 64
    float scale = 1.f / sqrtf((float)head_dim); // 1/8 = 0.125

    std::vector<float> attn(T * D, 0.f);

    for (int h = 0; h < n_heads; h++) {
        const int h_off = h * head_dim;
        std::vector<float> scores(T * T);
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += Q[i * stride_qkv + h_off + d] * K[j * stride_qkv + h_off + d];
                scores[i * T + j] = dot * scale;
            }
            float mx = *std::max_element(scores.data() + i * T, scores.data() + (i + 1) * T);
            float sm = 0;
            for (int j = 0; j < T; j++) {
                scores[i * T + j] = expf(scores[i * T + j] - mx);
                sm += scores[i * T + j];
            }
            for (int j = 0; j < T; j++)
                scores[i * T + j] /= sm;
        }
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++)
                for (int d = 0; d < head_dim; d++)
                    attn[i * D + h_off + d] += scores[i * T + j] * V[j * stride_qkv + h_off + d];
    }

    // Output projection: attn @ out_w + out_b → [T, D]
    // out_w stored as (D, D) → data[dd * D + d] = weight[dd, d]
    std::vector<float> proj(T * D);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            float sum = out_b[d];
            for (int dd = 0; dd < D; dd++)
                sum += attn[t * D + dd] * out_w[dd * D + d];
            proj[t * D + d] = sum;
        }
    }

    // Residual add back to x (transpose proj [T,D] → [D,T])
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            x[d * T + t] += proj[t * D + d];
}

// Simple FFN: linear1 → relu → linear2, residual
static void ffn_residual(float* x, int D, int T, const float* ff1_w, const float* ff1_b, const float* ff2_w,
                         const float* ff2_b) {
    // x is [D, T]. Transpose to [T, D], run FFN, add back.
    std::vector<float> xt(T * D), mid(T * D), out(T * D);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            xt[t * D + d] = x[d * T + t];

    // linear1: x @ ff1_w + ff1_b. ff1_w stored as (D, D) → data[dd * D + d]
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
            float sum = ff1_b[d];
            for (int dd = 0; dd < D; dd++)
                sum += xt[t * D + dd] * ff1_w[dd * D + d];
            mid[t * D + d] = std::max(0.f, sum); // ReLU
        }

    // linear2
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
            float sum = ff2_b[d];
            for (int dd = 0; dd < D; dd++)
                sum += mid[t * D + dd] * ff2_w[dd * D + d];
            out[t * D + d] = sum;
        }

    // Residual
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            x[d * T + t] += out[t * D + d];
}


// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct silero_lid_context* silero_lid_init(const char* gguf_path, int n_threads) {
    auto* ctx = new silero_lid_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend)
        ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!lid_load(ctx->model, gguf_path, ctx->backend)) {
        delete ctx;
        return nullptr;
    }

    // Load language strings from GGUF metadata
    {
        gguf_init_params mp = {true, nullptr};
        gguf_context* g = gguf_init_from_file(gguf_path, mp);
        if (g) {
            int ki = gguf_find_key(g, "silero_lid.lang_strs");
            if (ki >= 0) {
                int n = gguf_get_arr_n(g, ki);
                ctx->lang_strs.resize(n);
                for (int i = 0; i < n; i++)
                    ctx->lang_strs[i] = gguf_get_arr_str(g, ki, i);
            }
            ki = gguf_find_key(g, "silero_lid.group_strs");
            if (ki >= 0) {
                int n = gguf_get_arr_n(g, ki);
                ctx->group_strs.resize(n);
                for (int i = 0; i < n; i++)
                    ctx->group_strs[i] = gguf_get_arr_str(g, ki, i);
            }
            gguf_free(g);
        }
    }

    fprintf(stderr, "silero_lid: loaded %zu lang, %zu groups, %zu stages\n", ctx->lang_strs.size(),
            ctx->group_strs.size(), ctx->model.stages.size());
    return ctx;
}

extern "C" void silero_lid_free(struct silero_lid_context* ctx) {
    if (!ctx)
        return;
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

extern "C" int silero_lid_n_langs(struct silero_lid_context* ctx) {
    return ctx ? (int)ctx->lang_strs.size() : 0;
}

extern "C" const char* silero_lid_detect(struct silero_lid_context* ctx, const float* samples, int n_samples,
                                         float* out_confidence) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& m = ctx->model;

    // ---- Front-end: learned Conv1d(1→322, k=320, stride=160) ----
    // This is the model's "learned STFT" that frames raw audio into
    // ~100 Hz feature frames. Without this step the model would run
    // attention on T=176000 which is O(T^2) infeasible.
    int T;
    int C;
    std::vector<float> cur;
    if (m.frontend_w) {
        const float* fw = (const float*)m.frontend_w->data;
        int K_fe = m.frontend_kernel;   // 320
        int S_fe = m.frontend_stride;   // 160
        int C_fe = m.frontend_channels; // 322

        // Zero-pad the input with S_fe (160) samples on each side.
        // ONNX Pad_24 uses mode=constant (zeros) with pad vector
        // [0,0,0,160, 0,0,0,160] on the [1,1,1,N] tensor.
        int pad_lr = S_fe; // 160 on each side
        int N_padded = pad_lr + n_samples + pad_lr;
        std::vector<float> padded(N_padded, 0.f);
        std::memcpy(padded.data() + pad_lr, samples, n_samples * sizeof(float));

        T = (N_padded - K_fe) / S_fe + 1;
        C = C_fe;
        cur.resize(C_fe * T);
        // Conv1d: out[co, t] = sum_k(fw[co, 0, k] * padded[t*stride + k])
        for (int co = 0; co < C_fe; co++) {
            for (int t = 0; t < T; t++) {
                float sum = 0.f;
                for (int k = 0; k < K_fe; k++)
                    sum += fw[co * K_fe + k] * padded[t * S_fe + k];
                cur[co * T + t] = sum;
            }
        }
    } else {
        // Fallback: no front-end, use raw samples (will be slow)
        T = n_samples;
        C = 1;
        cur.assign(samples, samples + n_samples);
    }

    // ---- Magnitude + log + adaptive normalization ----
    //
    // The 322-channel front-end output is a COMPLEX representation:
    // channels 0..160 = real part, channels 161..321 = imaginary part.
    // (322 = 2 × 161, matching an n_fft/2+1 = 161 spectral bins.)
    //
    // ONNX data flow (confirmed by graph trace):
    //   1. Split into real/imag, each (161, T)
    //   2. magnitude = sqrt(real² + imag²)      → (161, T)
    //   3. log_mag = log(magnitude * scale + eps)
    //   4. per_frame_mean = mean(log_mag, axis=channels) → (1, T)
    //   5. smoothed = conv1d(pad(per_frame_mean), filter_17) → (1, T)
    //   6. global_mean = mean(smoothed, axis=time) → scalar
    //   7. normalized = log_mag - global_mean     → (161, T)
    //   8. Feed normalized to the encoder (161 channels)
    {
        const int C_half = C / 2; // 161
        // Step 1-2: compute magnitude from real + imag channels
        std::vector<float> mag(C_half * T);
        for (int c = 0; c < C_half; c++) {
            for (int t = 0; t < T; t++) {
                float re = cur[c * T + t];
                float im = cur[(c + C_half) * T + t];
                mag[c * T + t] = sqrtf(re * re + im * im);
            }
        }

        // Step 3: log(scale * magnitude + offset)
        // ONNX Constant_49 = 1048576.0 = 2^20 (scale), Constant_51 = 1.0 (offset).
        const float log_scale = 1048576.0f;
        const float log_offset = 1.0f;
        std::vector<float> log_mag(C_half * T);
        for (int i = 0; i < C_half * T; i++) {
            log_mag[i] = logf(log_scale * mag[i] + log_offset);
        }

        // Step 4: per-frame mean over channels
        std::vector<float> frame_mean(T, 0.f);
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C_half; c++)
                frame_mean[t] += log_mag[c * T + t];
            frame_mean[t] /= C_half;
        }

        // Step 5: smooth with adaptive normalization filter (17-tap)
        // ONNX uses reflection padding with pad=8 on each side of the time axis.
        if (m.adaptive_norm_filter) {
            const float* filt = (const float*)m.adaptive_norm_filter->data;
            int K_filt = 17, pad_f = K_filt / 2; // pad_f = 8

            // Reflection-pad frame_mean: [pad_f + T + pad_f]
            std::vector<float> padded_fm(pad_f + T + pad_f);
            // Left reflection: padded[i] = frame_mean[pad_f - i] for i=0..pad_f-1
            for (int i = 0; i < pad_f; i++)
                padded_fm[i] = frame_mean[pad_f - i];
            // Center: copy frame_mean
            std::memcpy(padded_fm.data() + pad_f, frame_mean.data(), T * sizeof(float));
            // Right reflection: padded[pad_f+T+i] = frame_mean[T-2-i]
            for (int i = 0; i < pad_f; i++)
                padded_fm[pad_f + T + i] = frame_mean[T - 2 - i];

            // Conv1d with the padded signal (no further padding needed)
            std::vector<float> smooth(T, 0.f);
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < K_filt; k++)
                    smooth[t] += filt[k] * padded_fm[t + k];
            }

            // Step 6: global mean of smoothed
            float global_mean = 0;
            for (int t = 0; t < T; t++)
                global_mean += smooth[t];
            global_mean /= T;

            // Step 7: normalize = log_mag - global_mean
            for (int i = 0; i < C_half * T; i++) {
                log_mag[i] -= global_mean;
            }
        }

        // Replace cur with the 161-channel log-magnitude features
        cur = std::move(log_mag);
        C = C_half; // 161
    }

    for (int si = 0; si < (int)m.stages.size(); si++) {
        const auto& st = m.stages[si];

        // ---- 12 conv blocks ----
        for (int bi = 0; bi < (int)st.conv_blocks.size(); bi++) {
            const auto& cb = st.conv_blocks[bi];
            if (!cb.dw_w || !cb.pw_w)
                continue;
            int C_out = (int)cb.pw_w->ne[2]; // ne for (Cout, Cin, 1) → ne[2]=Cout
            // Actually: pw_w is stored as (Cout, Cin, 1) in numpy → ne=(1, Cin, Cout) in ggml
            // For F32 tensor: ne[0]=1, ne[1]=Cin, ne[2]=Cout
            // So C_out = ne[2] in ggml... but core_gguf stores tensors as-is from numpy.
            // Let me just read the shape from the raw dims.
            C_out = (int)cb.pw_w->ne[0]; // ggml ne[0] = numpy shape[-1] reversed...
            // Actually for a 3D tensor: numpy (Cout, Cin, 1) → ggml ne=(1, Cin, Cout)
            // So ne[2] = Cout. Let me check: pw_w for stage 0, block 0 should be (161, 161, 1).
            // In ggml ne-order: ne[0]=1, ne[1]=161, ne[2]=161. So C_out = ne[2].
            C_out = (int)cb.pw_w->ne[2];
            int C_in = C;

            std::vector<float> out(C_out * T);
            const float* dw_w_f = (const float*)cb.dw_w->data;
            const float* dw_b_f = (const float*)cb.dw_b->data;
            const float* pw_w_f = (const float*)cb.pw_w->data;
            const float* pw_b_f = (const float*)cb.pw_b->data;

            // For blocks with proj: skip the residual+ReLU in dw_sep_conv1d
            // (they'll be applied after the proj add below)
            bool has_proj = (cb.proj_w != nullptr);
            dw_sep_conv1d(cur.data(), C_in, T, dw_w_f, dw_b_f, 5, pw_w_f, pw_b_f, C_out, out.data(),
                          /*add_residual=*/!has_proj);


            // For the LAST block of each stage (which changes channel count),
            // ONNX applies a separate 1×1 conv (proj) on the ORIGINAL BLOCK
            // INPUT and ADDS it to the pw_conv output as the residual:
            //   output = ReLU(pw_conv(ReLU(dw_conv(input))) + proj(input))
            // This is different from regular blocks where the residual is
            // just the identity: output = ReLU(pw_conv(...) + input).
            if (cb.proj_w) {
                int C_proj = (int)cb.proj_w->ne[2];
                const float* pj_w = (const float*)cb.proj_w->data;
                const float* pj_b = cb.proj_b ? (const float*)cb.proj_b->data : nullptr;
                // proj applies to cur (block input), NOT out (conv output)!
                // proj: (C_proj, C_in, 1) where C_in = original channel count
                std::vector<float> proj_res(C_proj * T);
                for (int co = 0; co < C_proj; co++) {
                    for (int t = 0; t < T; t++) {
                        float sum = pj_b ? pj_b[co] : 0.f;
                        for (int ci = 0; ci < C; ci++)
                            sum += pj_w[co * C + ci] * cur[ci * T + t];
                        proj_res[co * T + t] = sum;
                    }
                }
                // Add proj(input) to pw_conv output + ReLU
                for (int co = 0; co < C_proj; co++) {
                    for (int t = 0; t < T; t++) {
                        out[co * T + t] = std::max(0.f, out[co * T + t] + proj_res[co * T + t]);
                    }
                }
                C_out = C_proj;
            }

            cur = std::move(out);
            C = C_out;
        }

        // ---- Transformer block (runs BEFORE stride-2 downsample) ----
        // ONNX order: conv stage → transformer → transpose → stride-2 → next stage
        const auto& tx = st.tx;
        int D = C;

        if (tx.qkv_w) {
            // POST-norm transformer (ONNX confirmed: no LN before QKV)
            // 1. Self-attention (adds residual internally)
            self_attention(cur.data(), D, T, (const float*)tx.qkv_w->data, (const float*)tx.qkv_b->data,
                           (const float*)tx.out_w->data, (const float*)tx.out_b->data);
            // 2. Post-attention LayerNorm
            if (tx.norm1_w)
                layer_norm_ct(cur.data(), D, T, (const float*)tx.norm1_w->data, (const float*)tx.norm1_b->data);
            // 3. FFN (adds residual internally)
            ffn_residual(cur.data(), D, T, (const float*)tx.ff1_w->data, (const float*)tx.ff1_b->data,
                         (const float*)tx.ff2_w->data, (const float*)tx.ff2_b->data);
            // 4. Post-FFN LayerNorm
            if (tx.norm2_w)
                layer_norm_ct(cur.data(), D, T, (const float*)tx.norm2_w->data, (const float*)tx.norm2_b->data);
        }

        // ---- Stride-2 temporal downsampling (AFTER transformer, first 4 stages) ----
        // ONNX order: conv → transformer → transpose → stride-2 Conv → ReLU → next
        if (tx.conv1x1_w && si < m.n_downsample_stages) {
            int C_out = (int)tx.conv1x1_w->ne[2];
            // Conv1d output size: (T - kernel + 2*pad) / stride + 1
            // For kernel=1, pad=0, stride=2: (T-1)/2 + 1
            int T_out = (T - 1) / 2 + 1;
            std::vector<float> ds(C_out * T_out);
            const float* cw = (const float*)tx.conv1x1_w->data;
            const float* cb = tx.conv1x1_b ? (const float*)tx.conv1x1_b->data : nullptr;
            for (int co = 0; co < C_out; co++) {
                for (int t = 0; t < T_out; t++) {
                    float sum = cb ? cb[co] : 0.f;
                    int t_in = t * 2;
                    for (int ci = 0; ci < C; ci++)
                        sum += cw[co * C + ci] * cur[ci * T + t_in];
                    ds[co * T_out + t] = sum;
                }
            }
            for (float& v : ds)
                v = std::max(0.f, v);
            cur = std::move(ds);
            C = C_out;
            T = T_out;
        } else if (tx.conv1x1_w) {
            // Stride-1 projection (192-dim stages 4-7)
            // ONNX: Conv → ReLU (confirmed for all stages, e.g. Conv_1401 → Relu_1402)
            int C_out = (int)tx.conv1x1_w->ne[2];
            std::vector<float> proj(C_out * T);
            const float* cw = (const float*)tx.conv1x1_w->data;
            const float* cb = tx.conv1x1_b ? (const float*)tx.conv1x1_b->data : nullptr;
            for (int co = 0; co < C_out; co++) {
                for (int t = 0; t < T; t++) {
                    float sum = cb ? cb[co] : 0.f;
                    for (int ci = 0; ci < C; ci++)
                        sum += cw[co * C + ci] * cur[ci * T + t];
                    proj[co * T + t] = sum;
                }
            }
            for (float& v : proj)
                v = std::max(0.f, v); // ReLU
            cur = std::move(proj);
            C = C_out;
        }
    }

    // ---- Attention-weighted pooling over time ----
    // pool_weight is (D=192,). Compute per-frame score, softmax, weighted sum.
    int D = C;
    std::vector<float> frame_scores(T);
    const float* pw = (const float*)m.pool_weight->data;
    for (int t = 0; t < T; t++) {
        float dot = 0;
        for (int d = 0; d < D; d++)
            dot += pw[d] * cur[d * T + t];
        frame_scores[t] = tanhf(dot); // ONNX: Tanh_1405 before Softmax
    }
    // Softmax
    float mx = *std::max_element(frame_scores.begin(), frame_scores.end());
    float sum = 0;
    for (int t = 0; t < T; t++) {
        frame_scores[t] = expf(frame_scores[t] - mx);
        sum += frame_scores[t];
    }
    for (int t = 0; t < T; t++)
        frame_scores[t] /= sum;

    // Weighted sum → [D]
    std::vector<float> pooled(D, 0.f);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            pooled[d] += frame_scores[t] * cur[d * T + t];

    // ---- Language classifier: linear(D → 95) ----
    int n_langs = (int)ctx->lang_strs.size();
    std::vector<float> logits(n_langs);
    const float* lw = (const float*)m.lang_w->data;
    const float* lb = (const float*)m.lang_b->data;
    for (int i = 0; i < n_langs; i++) {
        float s = lb[i];
        for (int d = 0; d < D; d++)
            s += lw[i * D + d] * pooled[d];
        logits[i] = s;
    }

    // Argmax
    int best = 0;
    for (int i = 1; i < n_langs; i++)
        if (logits[i] > logits[best])
            best = i;

    if (out_confidence)
        *out_confidence = logits[best];

    if (best < (int)ctx->lang_strs.size())
        return ctx->lang_strs[best].c_str();
    return nullptr;
}
