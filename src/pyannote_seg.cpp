// pyannote_seg.cpp — native runtime for pyannote-segmentation-3.0.
//
// Architecture: SincNet front-end → 3× MaxPool → 4× bidirectional LSTM
//               → 3× Linear → LogSoftmax.
// The model is tiny (6 MB) and runs entirely on CPU in manual F32.

#include "pyannote_seg.h"
#include "core/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
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

// ===========================================================================
// Model
// ===========================================================================

struct pyannote_lstm {
    // ONNX convention: W (2, 4H, C_in), R (2, 4H, H), B (2, 8H)
    // Direction 0 = forward, 1 = backward.
    ggml_tensor * W = nullptr;  // weight_ih: (2, 4*hidden, input_size)
    ggml_tensor * R = nullptr;  // weight_hh: (2, 4*hidden, hidden_size)
    ggml_tensor * B = nullptr;  // bias:      (2, 8*hidden)
    int input_size  = 0;
    int hidden_size = 128;
};

struct pyannote_model {
    // SincNet front-end
    ggml_tensor * sinc_filters = nullptr;  // (80, 1, 251) learned sinc
    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;  // (60, 80, 5)
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;  // (60, 60, 5)
    ggml_tensor * norm0_w = nullptr, * norm0_b = nullptr;  // InstanceNorm (80)
    ggml_tensor * norm1_w = nullptr, * norm1_b = nullptr;  // (60)
    ggml_tensor * norm2_w = nullptr, * norm2_b = nullptr;  // (60)

    // 4 bidirectional LSTM layers
    pyannote_lstm lstm[4];

    // Linear classifiers
    ggml_tensor * linear0_b = nullptr;  // (128,)
    ggml_tensor * linear1_b = nullptr;  // (128,)
    ggml_tensor * linear2_b = nullptr;  // (7,) — actually the last classifier's bias
    ggml_tensor * matmul0_w = nullptr;  // (256, 128)
    ggml_tensor * matmul1_w = nullptr;  // (128, 128)
    ggml_tensor * matmul2_w = nullptr;  // (128, 7)

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
};

struct pyannote_seg_context {
    pyannote_model model;
    int n_threads = 4;
};

// ===========================================================================
// Loader
// ===========================================================================

static bool pyannote_load(pyannote_model & m, const char * path) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, backend, "pyannote_seg", wl)) {
        ggml_backend_free(backend);
        return false;
    }
    m.ctx = wl.ctx;
    m.buf = wl.buf;
    m.tensors = std::move(wl.tensors);
    ggml_backend_free(backend);

    auto get = [&](const char * name) -> ggml_tensor * {
        auto it = m.tensors.find(name);
        return it != m.tensors.end() ? it->second : nullptr;
    };

    m.sinc_filters = get("pyannote..sincnet.conv1d.0.Concat_2_output_0");
    m.conv1_w = get("pyannote.sincnet.conv1d.1.weight");
    m.conv1_b = get("pyannote.sincnet.conv1d.1.bias");
    m.conv2_w = get("pyannote.sincnet.conv1d.2.weight");
    m.conv2_b = get("pyannote.sincnet.conv1d.2.bias");
    m.norm0_w = get("pyannote.sincnet.norm1d.0.weight");
    m.norm0_b = get("pyannote.sincnet.norm1d.0.bias");
    m.norm1_w = get("pyannote.sincnet.norm1d.1.weight");
    m.norm1_b = get("pyannote.sincnet.norm1d.1.bias");
    m.norm2_w = get("pyannote.sincnet.norm1d.2.weight");
    m.norm2_b = get("pyannote.sincnet.norm1d.2.bias");

    // LSTM layers
    int lstm_input_sizes[] = {60, 256, 256, 256};
    const char * lstm_names[] = {
        "pyannote.onnx::LSTM_783", "pyannote.onnx::LSTM_784", "pyannote.onnx::LSTM_785",
        "pyannote.onnx::LSTM_826", "pyannote.onnx::LSTM_827", "pyannote.onnx::LSTM_828",
        "pyannote.onnx::LSTM_869", "pyannote.onnx::LSTM_870", "pyannote.onnx::LSTM_871",
        "pyannote.onnx::LSTM_912", "pyannote.onnx::LSTM_913", "pyannote.onnx::LSTM_914",
    };
    for (int i = 0; i < 4; i++) {
        m.lstm[i].B = get(lstm_names[i * 3 + 0]);
        m.lstm[i].W = get(lstm_names[i * 3 + 1]);
        m.lstm[i].R = get(lstm_names[i * 3 + 2]);
        m.lstm[i].input_size = lstm_input_sizes[i];
        m.lstm[i].hidden_size = 128;
    }

    m.linear0_b = get("pyannote.linear.0.bias");
    m.linear1_b = get("pyannote.linear.1.bias");
    m.linear2_b = get("pyannote.ortshared_1_1_7_0_token_109"); // (7,) last bias
    m.matmul0_w = get("pyannote.onnx::MatMul_915");  // (256, 128)
    m.matmul1_w = get("pyannote.onnx::MatMul_916");  // (128, 128)
    m.matmul2_w = get("pyannote.onnx::MatMul_917");  // (128, 7)

    if (!m.sinc_filters || !m.lstm[0].W || !m.matmul0_w) {
        fprintf(stderr, "pyannote_seg: missing critical tensors\n");
        return false;
    }
    return true;
}

// ===========================================================================
// LSTM forward (bidirectional)
// ===========================================================================

// Run one bidirectional LSTM layer. Input: (T, C_in) row-major.
// Output: (T, 2*hidden) row-major (forward + backward concatenated).
static void bilstm_forward(
    const float * input, int T, int C_in,
    const pyannote_lstm & lstm,
    float * output)
{
    const int H = lstm.hidden_size;
    const int dirs = 2;

    for (int dir = 0; dir < dirs; dir++) {
        // Extract per-direction weights
        // W: (2, 4H, C_in) → dir-slice: W[dir] = (4H, C_in)
        const float * W_dir = (const float *)lstm.W->data + (size_t)dir * 4 * H * C_in;
        const float * R_dir = (const float *)lstm.R->data + (size_t)dir * 4 * H * H;
        // B: (2, 8H) = (2, 4H input_bias + 4H hidden_bias)
        const float * Bi_dir = (const float *)lstm.B->data + (size_t)dir * 8 * H;
        const float * Bh_dir = Bi_dir + 4 * H;

        std::vector<float> h(H, 0.f), c(H, 0.f);
        std::vector<float> gates(4 * H);

        auto sigmoid = [](float x) { return 1.f / (1.f + expf(-x)); };

        for (int step = 0; step < T; step++) {
            int t = (dir == 0) ? step : (T - 1 - step);
            const float * x = input + (size_t)t * C_in;

            // gates = W @ x + R @ h + bias
            for (int g = 0; g < 4 * H; g++) {
                float sum = Bi_dir[g] + Bh_dir[g];
                for (int j = 0; j < C_in; j++)
                    sum += W_dir[g * C_in + j] * x[j];
                for (int j = 0; j < H; j++)
                    sum += R_dir[g * H + j] * h[j];
                gates[g] = sum;
            }

            // i=sigmoid, f=sigmoid, g=tanh, o=sigmoid
            for (int j = 0; j < H; j++) {
                float i_g = sigmoid(gates[0 * H + j]);
                float f_g = sigmoid(gates[1 * H + j]);
                float g_g = tanhf(gates[2 * H + j]);
                float o_g = sigmoid(gates[3 * H + j]);
                c[j] = f_g * c[j] + i_g * g_g;
                h[j] = o_g * tanhf(c[j]);
            }

            // Write output: forward → first H, backward → second H
            float * out_t = output + (size_t)t * 2 * H;
            for (int j = 0; j < H; j++)
                out_t[dir * H + j] = h[j];
        }
    }
}

// ===========================================================================
// Public API
// ===========================================================================

extern "C" struct pyannote_seg_context * pyannote_seg_init(
    const char * gguf_path, int n_threads)
{
    auto * ctx = new pyannote_seg_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    if (!pyannote_load(ctx->model, gguf_path)) {
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "pyannote_seg: loaded (%d LSTM layers, hidden=%d)\n",
            4, ctx->model.lstm[0].hidden_size);
    return ctx;
}

extern "C" void pyannote_seg_free(struct pyannote_seg_context * ctx) {
    if (!ctx) return;
    if (ctx->model.buf) ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->model.ctx) ggml_free(ctx->model.ctx);
    delete ctx;
}

extern "C" float * pyannote_seg_run(
    struct pyannote_seg_context * ctx,
    const float * samples, int n_samples,
    int * out_T)
{
    if (!ctx || !samples || n_samples <= 0) return nullptr;
    const auto & m = ctx->model;

    // ---- SincNet front-end ----
    // Conv1d(1→80, kernel=251, stride=10) using the learned sinc filters
    const int K0 = 251, S0 = 10;
    const int C0 = 80;
    int T = (n_samples - K0) / S0 + 1;
    const float * sinc = (const float *)m.sinc_filters->data;

    std::vector<float> feat(C0 * T);
    for (int co = 0; co < C0; co++) {
        for (int t = 0; t < T; t++) {
            float sum = 0;
            for (int k = 0; k < K0; k++)
                sum += sinc[co * K0 + k] * samples[t * S0 + k];
            feat[co * T + t] = std::abs(sum);  // SincNet uses |conv|
        }
    }

    // InstanceNorm + LeakyReLU after sinc
    // TODO: proper InstanceNorm; for now just normalize per-channel
    if (m.norm0_w) {
        const float * nw = (const float *)m.norm0_w->data;
        const float * nb = (const float *)m.norm0_b->data;
        for (int c = 0; c < C0; c++) {
            float mean = 0, var = 0;
            for (int t = 0; t < T; t++) mean += feat[c * T + t];
            mean /= T;
            for (int t = 0; t < T; t++) {
                float d = feat[c * T + t] - mean;
                var += d * d;
            }
            var /= T;
            float inv = 1.f / sqrtf(var + 1e-5f);
            for (int t = 0; t < T; t++)
                feat[c * T + t] = nw[c] * (feat[c * T + t] - mean) * inv + nb[c];
        }
    }
    // LeakyReLU
    for (float & v : feat) v = v >= 0 ? v : 0.01f * v;

    // MaxPool1d(kernel=3, stride=3)
    int T1 = T / 3;
    std::vector<float> pool1(C0 * T1);
    for (int c = 0; c < C0; c++)
        for (int t = 0; t < T1; t++) {
            float mx = feat[c * T + t * 3];
            for (int k = 1; k < 3; k++) mx = std::max(mx, feat[c * T + t * 3 + k]);
            pool1[c * T1 + t] = mx;
        }

    // Conv1d(80→60, k=5, pad=2) + InstanceNorm + LeakyReLU + MaxPool(3)
    auto do_conv_norm_pool = [&](const std::vector<float> & in, int C_in, int T_in,
                                  ggml_tensor * cw, ggml_tensor * cb,
                                  ggml_tensor * nw, ggml_tensor * nb,
                                  int C_out, int K, int pool_k,
                                  std::vector<float> & out, int & T_out) {
        int pad = K / 2;
        std::vector<float> conv_out(C_out * T_in);
        const float * w = (const float *)cw->data;
        const float * b = (const float *)cb->data;
        for (int co = 0; co < C_out; co++) {
            for (int t = 0; t < T_in; t++) {
                float sum = b[co];
                for (int ci = 0; ci < C_in; ci++)
                    for (int k = 0; k < K; k++) {
                        int ti = t + k - pad;
                        if (ti >= 0 && ti < T_in)
                            sum += w[(co * C_in + ci) * K + k] * in[ci * T_in + ti];
                    }
                conv_out[co * T_in + t] = sum;
            }
        }
        // InstanceNorm
        if (nw && nb) {
            const float * nwp = (const float *)nw->data;
            const float * nbp = (const float *)nb->data;
            for (int c = 0; c < C_out; c++) {
                float mean = 0, var = 0;
                for (int t = 0; t < T_in; t++) mean += conv_out[c * T_in + t];
                mean /= T_in;
                for (int t = 0; t < T_in; t++) {
                    float d = conv_out[c * T_in + t] - mean;
                    var += d * d;
                }
                var /= T_in;
                float inv = 1.f / sqrtf(var + 1e-5f);
                for (int t = 0; t < T_in; t++)
                    conv_out[c * T_in + t] = nwp[c] * (conv_out[c * T_in + t] - mean) * inv + nbp[c];
            }
        }
        // LeakyReLU
        for (float & v : conv_out) v = v >= 0 ? v : 0.01f * v;
        // MaxPool
        T_out = T_in / pool_k;
        out.resize(C_out * T_out);
        for (int c = 0; c < C_out; c++)
            for (int t = 0; t < T_out; t++) {
                float mx = conv_out[c * T_in + t * pool_k];
                for (int k = 1; k < pool_k; k++)
                    mx = std::max(mx, conv_out[c * T_in + t * pool_k + k]);
                out[c * T_out + t] = mx;
            }
    };

    std::vector<float> stage1_out;
    int T2;
    do_conv_norm_pool(pool1, C0, T1, m.conv1_w, m.conv1_b, m.norm1_w, m.norm1_b, 60, 5, 3, stage1_out, T2);

    std::vector<float> stage2_out;
    int T3;
    do_conv_norm_pool(stage1_out, 60, T2, m.conv2_w, m.conv2_b, m.norm2_w, m.norm2_b, 60, 5, 3, stage2_out, T3);

    // Transpose to (T, C=60) for LSTM input
    std::vector<float> lstm_in(T3 * 60);
    for (int t = 0; t < T3; t++)
        for (int c = 0; c < 60; c++)
            lstm_in[t * 60 + c] = stage2_out[c * T3 + t];

    // ---- 4× bidirectional LSTM ----
    int T_lstm = T3;
    std::vector<float> lstm_out;
    for (int li = 0; li < 4; li++) {
        int C_in = (li == 0) ? 60 : 256;
        lstm_out.resize(T_lstm * 256);
        bilstm_forward(lstm_in.data(), T_lstm, C_in, m.lstm[li], lstm_out.data());
        lstm_in = lstm_out;
    }

    // ---- 3× Linear + LeakyReLU + LogSoftmax ----
    // Linear 0: (256→128) + LeakyReLU
    auto linear = [](const float * in, int T, int C_in,
                     const float * w, const float * b, int C_out,
                     float * out, bool leaky_relu = true) {
        for (int t = 0; t < T; t++) {
            for (int co = 0; co < C_out; co++) {
                float sum = b ? b[co] : 0.f;
                for (int ci = 0; ci < C_in; ci++)
                    sum += w[ci * C_out + co] * in[t * C_in + ci];
                out[t * C_out + co] = leaky_relu ? (sum >= 0 ? sum : 0.01f * sum) : sum;
            }
        }
    };

    std::vector<float> lin0(T_lstm * 128);
    linear(lstm_out.data(), T_lstm, 256,
           (const float *)m.matmul0_w->data,
           m.linear0_b ? (const float *)m.linear0_b->data : nullptr,
           128, lin0.data());

    std::vector<float> lin1(T_lstm * 128);
    linear(lin0.data(), T_lstm, 128,
           (const float *)m.matmul1_w->data,
           m.linear1_b ? (const float *)m.linear1_b->data : nullptr,
           128, lin1.data());

    std::vector<float> lin2(T_lstm * 7);
    linear(lin1.data(), T_lstm, 128,
           (const float *)m.matmul2_w->data,
           m.linear2_b ? (const float *)m.linear2_b->data : nullptr,
           7, lin2.data(), false);  // no LeakyReLU before LogSoftmax

    // LogSoftmax over the 7 classes per frame
    for (int t = 0; t < T_lstm; t++) {
        float * lv = lin2.data() + t * 7;
        float mx = *std::max_element(lv, lv + 7);
        float sum = 0;
        for (int i = 0; i < 7; i++) { lv[i] = expf(lv[i] - mx); sum += lv[i]; }
        for (int i = 0; i < 7; i++) lv[i] = logf(lv[i] / sum);
    }

    if (out_T) *out_T = T_lstm;
    float * result = (float *)malloc(T_lstm * 7 * sizeof(float));
    if (result) memcpy(result, lin2.data(), T_lstm * 7 * sizeof(float));
    return result;
}
