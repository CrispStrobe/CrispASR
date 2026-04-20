// ecapa_lid.cpp — ECAPA-TDNN LID runtime.
//
// All-CPU implementation. The model is small enough (~21M params) that
// ggml graph overhead isn't worth it — plain loops with simple helpers
// are faster for this model size.

#include "ecapa_lid.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===========================================================================
// Helpers
// ===========================================================================

static void read_f32(ggml_tensor* t, std::vector<float>& out) {
    if (!t) {
        out.clear();
        return;
    }
    int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(uint16_t));
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp.data(), out.data(), n);
    } else {
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float)
            traits->to_float(raw.data(), out.data(), n);
        else
            std::fill(out.begin(), out.end(), 0.0f);
    }
}

// Conv1d: out[co][t] = sum_ci sum_k w[co][ci][k] * in[ci][t*stride + k*dilation] + bias[co]
// in: [C_in, T], out: [C_out, T_out], w: [C_out, C_in, K]
static void conv1d(const float* in, int C_in, int T_in, const float* w, const float* bias, int C_out, int K,
                   int stride, int dilation, float* out, int& T_out) {
    T_out = (T_in - dilation * (K - 1) - 1) / stride + 1;
    if (T_out <= 0)
        return;
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < T_out; t++) {
            double s = bias ? bias[co] : 0;
            for (int ci = 0; ci < C_in; ci++) {
                for (int k = 0; k < K; k++) {
                    int ti = t * stride + k * dilation;
                    if (ti >= 0 && ti < T_in)
                        s += (double)w[(co * C_in + ci) * K + k] * (double)in[ci * T_in + ti];
                }
            }
            out[co * T_out + t] = (float)s;
        }
    }
}

// BatchNorm1d: out = (in - running_mean) / sqrt(running_var + eps) * weight + bias
static void batchnorm1d(float* data, int C, int T, const float* weight, const float* bias, const float* mean,
                        const float* var, float eps = 1e-5f) {
    for (int c = 0; c < C; c++) {
        float scale = weight[c] / sqrtf(var[c] + eps);
        float shift = bias[c] - mean[c] * scale;
        for (int t = 0; t < T; t++)
            data[c * T + t] = data[c * T + t] * scale + shift;
    }
}

static void relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0)
            data[i] = 0;
}

// Pad input symmetrically: [C, T] → [C, T + pad_left + pad_right]
static void pad_1d(const float* in, int C, int T, int pad_left, int pad_right, float* out) {
    int T_new = T + pad_left + pad_right;
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < T_new; t++) {
            int ti = t - pad_left;
            out[c * T_new + t] = (ti >= 0 && ti < T) ? in[c * T + ti] : 0.0f;
        }
    }
}

// ===========================================================================
// Model structures
// ===========================================================================

struct bn_params {
    std::vector<float> weight, bias, mean, var;
};

struct conv_bn {
    std::vector<float> conv_w, conv_b;
    bn_params bn;
    int C_out = 0, C_in = 0, K = 0;
};

struct se_block {
    conv_bn conv1; // [128, 1024, 1] — squeeze
    conv_bn conv2; // [1024, 128, 1] — excite
};

struct res2net_sub {
    std::vector<float> conv_w, conv_b;
    bn_params bn;
};

struct se_res2net_block {
    conv_bn tdnn1;                  // [1024, 1024, 1]
    std::vector<res2net_sub> subs;  // 7 sub-bands (scale-1)
    se_block se;
    conv_bn tdnn2;                  // [1024, 1024, 1]
};

struct ecapa_model {
    // Block 0: initial TDNN
    conv_bn block0;           // Conv1d(60, 1024, 5)

    // Blocks 1-3: SE-Res2Net
    std::vector<se_res2net_block> se_blocks;

    // MFA
    conv_bn mfa;              // Conv1d(3072, 3072, 1)

    // ASP (attentive statistical pooling)
    conv_bn asp_tdnn;         // Conv1d(9216, 128, 1) — attention input
    std::vector<float> asp_conv_w, asp_conv_b; // [3072, 128, 1] — attention output

    // ASP BN
    bn_params asp_bn;         // BN(6144)

    // FC: embedding
    std::vector<float> fc_w, fc_b; // Conv1d(6144, 256, 1)

    // Classifier
    bn_params cls_bn;
    std::vector<float> cls_w1, cls_b1; // Linear(256, 512)
    bn_params cls_bn1;
    std::vector<float> cls_w2, cls_b2; // Linear(512, 107)

    // Labels
    std::vector<std::string> labels;
    int n_mels = 60;
    int n_classes = 107;
};

struct ecapa_lid_context {
    ecapa_model model;
    int n_threads = 4;
    std::string last_result;
};

// ===========================================================================
// Fbank (60-dim, Hamming window, matching SpeechBrain defaults)
// ===========================================================================

static void compute_fbank60(const float* pcm, int n_samples, std::vector<float>& features, int& n_frames) {
    const int sr = 16000, n_mels = 60;
    const int win_len = 400, hop = 160, n_fft = 512;
    const float low_freq = 0.0f, high_freq = (float)sr / 2;

    n_frames = (n_samples - win_len) / hop + 1;
    if (n_frames <= 0) {
        n_frames = 0;
        return;
    }

    int bins = n_fft / 2 + 1;
    std::vector<float> mel_fb(n_mels * bins, 0.0f);
    {
        auto hz2mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
        auto mel2hz = [](float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };
        float ml = hz2mel(low_freq), mh = hz2mel(high_freq);
        std::vector<float> c(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            c[i] = mel2hz(ml + i * (mh - ml) / (n_mels + 1));
        for (int m = 0; m < n_mels; m++)
            for (int k = 0; k < bins; k++) {
                float f = (float)k * sr / n_fft;
                if (f > c[m] && f <= c[m + 1] && c[m + 1] > c[m])
                    mel_fb[m * bins + k] = (f - c[m]) / (c[m + 1] - c[m]);
                else if (f > c[m + 1] && f < c[m + 2] && c[m + 2] > c[m + 1])
                    mel_fb[m * bins + k] = (c[m + 2] - f) / (c[m + 2] - c[m + 1]);
            }
    }

    // Hamming window
    std::vector<float> window(win_len);
    for (int i = 0; i < win_len; i++)
        window[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (win_len - 1));

    features.resize(n_frames * n_mels);
    std::vector<float> fre(n_fft), fim(n_fft);
    auto fft = [](float* r, float* im, int n) {
        for (int i = 1, j = 0; i < n; i++) {
            int b = n >> 1;
            for (; j & b; b >>= 1)
                j ^= b;
            j ^= b;
            if (i < j) {
                std::swap(r[i], r[j]);
                std::swap(im[i], im[j]);
            }
        }
        for (int l = 2; l <= n; l <<= 1) {
            float a = -2.f * (float)M_PI / l, wr = cosf(a), wi = sinf(a);
            for (int i = 0; i < n; i += l) {
                float cr = 1, ci = 0;
                for (int j = 0; j < l / 2; j++) {
                    float t = r[i + j + l / 2] * cr - im[i + j + l / 2] * ci,
                          u = r[i + j + l / 2] * ci + im[i + j + l / 2] * cr;
                    r[i + j + l / 2] = r[i + j] - t;
                    im[i + j + l / 2] = im[i + j] - u;
                    r[i + j] += t;
                    im[i + j] += u;
                    float nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
    };

    for (int t = 0; t < n_frames; t++) {
        int off = t * hop;
        std::fill(fre.begin(), fre.end(), 0);
        std::fill(fim.begin(), fim.end(), 0);
        for (int i = 0; i < win_len; i++) {
            float v = (off + i < n_samples) ? pcm[off + i] : 0.0f;
            fre[i] = v * window[i];
        }
        fft(fre.data(), fim.data(), n_fft);
        for (int m = 0; m < n_mels; m++) {
            float s = 0;
            for (int k = 0; k < bins; k++)
                s += (fre[k] * fre[k] + fim[k] * fim[k]) * mel_fb[m * bins + k];
            features[t * n_mels + m] = logf(std::max(s, 1e-10f));
        }
    }
}

// ===========================================================================
// Load helpers
// ===========================================================================

static void load_conv_bn(const std::map<std::string, ggml_tensor*>& ts, const std::string& prefix, conv_bn& cb) {
    auto get = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };
    read_f32(get(prefix + ".conv.weight"), cb.conv_w);
    read_f32(get(prefix + ".conv.bias"), cb.conv_b);
    read_f32(get(prefix + ".bn.weight"), cb.bn.weight);
    read_f32(get(prefix + ".bn.bias"), cb.bn.bias);
    read_f32(get(prefix + ".bn.running_mean"), cb.bn.mean);
    read_f32(get(prefix + ".bn.running_var"), cb.bn.var);

    // Infer dimensions from conv weight shape
    ggml_tensor* w = get(prefix + ".conv.weight");
    if (w) {
        // ggml shape: ne[0] is fastest. For 3D [K, C_in, C_out] or 2D [C_in, C_out]
        if (ggml_n_dims(w) == 3) {
            cb.K = (int)w->ne[0];
            cb.C_in = (int)w->ne[1];
            cb.C_out = (int)w->ne[2];
        } else {
            cb.K = 1;
            cb.C_in = (int)w->ne[0];
            cb.C_out = (int)w->ne[1];
        }
    }
}

// ===========================================================================
// Init
// ===========================================================================

extern "C" struct ecapa_lid_context* ecapa_lid_init(const char* model_path, int n_threads) {
    auto* ctx = new ecapa_lid_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    auto& m = ctx->model;

    // Read hyperparams
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }
    m.n_mels = core_gguf::kv_u32(gctx, "ecapa.n_mels", 60);
    m.n_classes = core_gguf::kv_u32(gctx, "ecapa.n_classes", 107);

    // Labels
    const int tok_key = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tok_key >= 0) {
        int n = gguf_get_arr_n(gctx, tok_key);
        m.labels.resize(n);
        for (int i = 0; i < n; i++) {
            const char* s = gguf_get_arr_str(gctx, tok_key, i);
            if (s)
                m.labels[i] = s;
        }
    }
    gguf_free(gctx);

    fprintf(stderr, "ecapa_lid: %d mels, %d classes, %zu labels\n", m.n_mels, m.n_classes, m.labels.size());

    // Load weights
    ggml_backend_t backend = ggml_backend_init_best();
    if (!backend) {
        delete ctx;
        return nullptr;
    }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "ecapa-tdnn-lid", wl)) {
        ggml_backend_free(backend);
        delete ctx;
        return nullptr;
    }

    auto& ts = wl.tensors;

    // Block 0
    load_conv_bn(ts, "emb.blocks.0", m.block0);

    // SE-Res2Net blocks 1-3
    m.se_blocks.resize(3);
    for (int i = 0; i < 3; i++) {
        auto& b = m.se_blocks[i];
        std::string bp = "emb.blocks." + std::to_string(i + 1);

        load_conv_bn(ts, bp + ".tdnn1", b.tdnn1);
        load_conv_bn(ts, bp + ".tdnn2", b.tdnn2);

        // Res2Net sub-bands (7 = scale-1)
        b.subs.resize(7);
        for (int j = 0; j < 7; j++) {
            std::string sp = bp + ".res2net_block.blocks." + std::to_string(j);
            auto get = [&](const std::string& name) -> ggml_tensor* {
                auto it = ts.find(name);
                return it != ts.end() ? it->second : nullptr;
            };
            read_f32(get(sp + ".conv.weight"), b.subs[j].conv_w);
            read_f32(get(sp + ".conv.bias"), b.subs[j].conv_b);
            read_f32(get(sp + ".bn.weight"), b.subs[j].bn.weight);
            read_f32(get(sp + ".bn.bias"), b.subs[j].bn.bias);
            read_f32(get(sp + ".bn.running_mean"), b.subs[j].bn.mean);
            read_f32(get(sp + ".bn.running_var"), b.subs[j].bn.var);
        }

        // SE block
        load_conv_bn(ts, bp + ".se_block.conv1", b.se.conv1);
        load_conv_bn(ts, bp + ".se_block.conv2", b.se.conv2);
    }

    // MFA
    load_conv_bn(ts, "emb.mfa", m.mfa);

    // ASP
    load_conv_bn(ts, "emb.asp.tdnn", m.asp_tdnn);
    {
        auto it_w = ts.find("emb.asp.conv.weight");
        auto it_b = ts.find("emb.asp.conv.bias");
        if (it_w != ts.end())
            read_f32(it_w->second, m.asp_conv_w);
        if (it_b != ts.end())
            read_f32(it_b->second, m.asp_conv_b);
    }

    // ASP BN
    {
        auto get = [&](const std::string& name) -> ggml_tensor* {
            auto it = ts.find(name);
            return it != ts.end() ? it->second : nullptr;
        };
        read_f32(get("emb.asp_bn.norm.weight"), m.asp_bn.weight);
        read_f32(get("emb.asp_bn.norm.bias"), m.asp_bn.bias);
        read_f32(get("emb.asp_bn.norm.running_mean"), m.asp_bn.mean);
        read_f32(get("emb.asp_bn.norm.running_var"), m.asp_bn.var);
    }

    // FC
    {
        auto it_w = ts.find("emb.fc.conv.weight");
        auto it_b = ts.find("emb.fc.conv.bias");
        if (it_w != ts.end())
            read_f32(it_w->second, m.fc_w);
        if (it_b != ts.end())
            read_f32(it_b->second, m.fc_b);
    }

    // Classifier
    {
        auto get = [&](const std::string& name) -> ggml_tensor* {
            auto it = ts.find(name);
            return it != ts.end() ? it->second : nullptr;
        };
        read_f32(get("cls.bn.weight"), m.cls_bn.weight);
        read_f32(get("cls.bn.bias"), m.cls_bn.bias);
        read_f32(get("cls.bn.running_mean"), m.cls_bn.mean);
        read_f32(get("cls.bn.running_var"), m.cls_bn.var);
        read_f32(get("cls.DNN.block_0.linear.weight"), m.cls_w1);
        read_f32(get("cls.DNN.block_0.linear.bias"), m.cls_b1);
        read_f32(get("cls.DNN.block_0.bn.weight"), m.cls_bn1.weight);
        read_f32(get("cls.DNN.block_0.bn.bias"), m.cls_bn1.bias);
        read_f32(get("cls.DNN.block_0.bn.running_mean"), m.cls_bn1.mean);
        read_f32(get("cls.DNN.block_0.bn.running_var"), m.cls_bn1.var);
        read_f32(get("cls.out.w.weight"), m.cls_w2);
        read_f32(get("cls.out.w.bias"), m.cls_b2);
    }

    // Clean up — keep data in CPU vectors, free ggml context
    ggml_free(wl.ctx);
    ggml_backend_buffer_free(wl.buf);
    ggml_backend_free(backend);

    return ctx;
}

extern "C" void ecapa_lid_free(struct ecapa_lid_context* ctx) {
    delete ctx;
}

// ===========================================================================
// Forward pass
// ===========================================================================

extern "C" const char* ecapa_lid_detect(struct ecapa_lid_context* ctx, const float* samples, int n_samples,
                                        float* confidence) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    auto& m = ctx->model;

    // Cap input at 5 seconds — LID doesn't need more, and the MFA conv is O(C²×T)
    constexpr int kMaxSamples = 16000 * 5;
    if (n_samples > kMaxSamples)
        n_samples = kMaxSamples;

    // 1. Fbank
    std::vector<float> fbank;
    int T = 0;
    compute_fbank60(samples, n_samples, fbank, T);
    fprintf(stderr, "ecapa_lid: fbank T=%d, n_mels=%d\n", T, m.n_mels);
    fflush(stderr);
    if (T <= 0)
        return nullptr;

    // Transpose to [C=60, T] for conv1d processing
    std::vector<float> x(m.n_mels * T);
    for (int c = 0; c < m.n_mels; c++)
        for (int t = 0; t < T; t++)
            x[c * T + t] = fbank[t * m.n_mels + c];

    // 2. Sentence-level mean normalization
    for (int c = 0; c < m.n_mels; c++) {
        float mean = 0;
        for (int t = 0; t < T; t++)
            mean += x[c * T + t];
        mean /= T;
        for (int t = 0; t < T; t++)
            x[c * T + t] -= mean;
    }

    // 3. Block 0: Conv1d(60→1024, k=5, d=1) + BN + ReLU
    int pad0 = (m.block0.K - 1) / 2; // symmetric padding for k=5 → pad=2
    std::vector<float> x_pad(m.n_mels * (T + 2 * pad0));
    pad_1d(x.data(), m.n_mels, T, pad0, pad0, x_pad.data());
    int T0 = 0;
    std::vector<float> h0(1024 * T); // output channels=1024
    fprintf(stderr, "ecapa_lid: block0 K=%d C_in=%d C_out=%d conv_w.size=%zu bn.weight=%zu\n", m.block0.K,
            m.block0.C_in, m.block0.C_out, m.block0.conv_w.size(), m.block0.bn.weight.size());
    fflush(stderr);
    if (m.block0.conv_w.empty()) {
        fprintf(stderr, "ecapa_lid: ERROR — block0 weights not loaded!\n");
        return nullptr;
    }
    conv1d(x_pad.data(), m.n_mels, T + 2 * pad0, m.block0.conv_w.data(), m.block0.conv_b.data(), 1024,
           m.block0.K, 1, 1, h0.data(), T0);
    fprintf(stderr, "ecapa_lid: block0 done, T0=%d\n", T0);
    batchnorm1d(h0.data(), 1024, T0, m.block0.bn.weight.data(), m.block0.bn.bias.data(), m.block0.bn.mean.data(),
                m.block0.bn.var.data());
    relu_inplace(h0.data(), 1024 * T0);

    // 4. SE-Res2Net blocks 1-3
    int C = 1024, scale = 8, sub_c = C / scale; // sub_c = 128
    std::vector<std::vector<float>> block_outputs; // for MFA concatenation
    std::vector<float> cur = h0;
    int T_cur = T0;

    for (int bi = 0; bi < 3; bi++) {
        auto& blk = m.se_blocks[bi];
        int dilation = bi + 2; // dilations: 2, 3, 4

        // tdnn1: pointwise conv (1024→1024, k=1) + BN + ReLU
        std::vector<float> h_tdnn1(C * T_cur);
        int T_tmp = 0;
        conv1d(cur.data(), C, T_cur, blk.tdnn1.conv_w.data(), blk.tdnn1.conv_b.data(), C, 1, 1, 1, h_tdnn1.data(),
               T_tmp);
        batchnorm1d(h_tdnn1.data(), C, T_cur, blk.tdnn1.bn.weight.data(), blk.tdnn1.bn.bias.data(),
                    blk.tdnn1.bn.mean.data(), blk.tdnn1.bn.var.data());
        relu_inplace(h_tdnn1.data(), C * T_cur);

        // Res2Net: split into 8 sub-bands of 128 channels
        // Sub-band 0: pass through
        // Sub-bands 1-7: conv1d(128→128, k=3, d=dilation) + BN + ReLU, with residual from prev
        std::vector<float> res2_out(C * T_cur, 0);
        // Copy sub-band 0 directly
        for (int t = 0; t < T_cur; t++)
            for (int c = 0; c < sub_c; c++)
                res2_out[c * T_cur + t] = h_tdnn1[c * T_cur + t];

        std::vector<float> prev_sub(sub_c * T_cur, 0);
        for (int si = 0; si < 7; si++) { // sub-bands 1-7
            int c_off = (si + 1) * sub_c;
            std::vector<float> sub_in(sub_c * T_cur);
            for (int t = 0; t < T_cur; t++)
                for (int c = 0; c < sub_c; c++)
                    sub_in[c * T_cur + t] = h_tdnn1[(c_off + c) * T_cur + t] + prev_sub[c * T_cur + t];

            // Conv1d(128→128, k=3, dilation=dilation) with causal-style padding
            int pad_r2 = dilation; // padding = dilation for k=3
            std::vector<float> sub_padded(sub_c * (T_cur + 2 * pad_r2));
            pad_1d(sub_in.data(), sub_c, T_cur, pad_r2, pad_r2, sub_padded.data());
            int T_r2 = 0;
            std::vector<float> sub_out(sub_c * T_cur);
            conv1d(sub_padded.data(), sub_c, T_cur + 2 * pad_r2, blk.subs[si].conv_w.data(),
                   blk.subs[si].conv_b.data(), sub_c, 3, 1, dilation, sub_out.data(), T_r2);

            // BN + ReLU
            if (T_r2 == T_cur) {
                batchnorm1d(sub_out.data(), sub_c, T_r2, blk.subs[si].bn.weight.data(), blk.subs[si].bn.bias.data(),
                            blk.subs[si].bn.mean.data(), blk.subs[si].bn.var.data());
                relu_inplace(sub_out.data(), sub_c * T_r2);

                // Copy to output and save as prev
                for (int t = 0; t < T_cur; t++)
                    for (int c = 0; c < sub_c; c++) {
                        res2_out[(c_off + c) * T_cur + t] = sub_out[c * T_cur + t];
                        prev_sub[c * T_cur + t] = sub_out[c * T_cur + t];
                    }
            }
        }

        // SE: squeeze-excitation
        // Global average pool → conv1(1024→128) → ReLU → conv2(128→1024) → sigmoid → scale
        std::vector<float> se_pool(C, 0);
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T_cur; t++)
                se_pool[c] += res2_out[c * T_cur + t];
            se_pool[c] /= T_cur;
        }

        // SE conv1: 1024→128 (stored as squeezed 2D)
        std::vector<float> se1(128, 0);
        for (int i = 0; i < 128; i++) {
            double s = blk.se.conv1.conv_b.empty() ? 0 : blk.se.conv1.conv_b[i];
            for (int k = 0; k < C; k++)
                s += se_pool[k] * blk.se.conv1.conv_w[i * C + k];
            se1[i] = std::max((float)s, 0.0f); // ReLU
        }

        // SE conv2: 128→1024
        std::vector<float> se2(C, 0);
        for (int i = 0; i < C; i++) {
            double s = blk.se.conv2.conv_b.empty() ? 0 : blk.se.conv2.conv_b[i];
            for (int k = 0; k < 128; k++)
                s += se1[k] * blk.se.conv2.conv_w[i * 128 + k];
            se2[i] = 1.0f / (1.0f + expf(-(float)s)); // sigmoid
        }

        // Scale
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T_cur; t++)
                res2_out[c * T_cur + t] *= se2[c];

        // tdnn2: pointwise conv (1024→1024, k=1) + BN + ReLU
        std::vector<float> h_tdnn2(C * T_cur);
        conv1d(res2_out.data(), C, T_cur, blk.tdnn2.conv_w.data(), blk.tdnn2.conv_b.data(), C, 1, 1, 1,
               h_tdnn2.data(), T_tmp);
        batchnorm1d(h_tdnn2.data(), C, T_cur, blk.tdnn2.bn.weight.data(), blk.tdnn2.bn.bias.data(),
                    blk.tdnn2.bn.mean.data(), blk.tdnn2.bn.var.data());
        relu_inplace(h_tdnn2.data(), C * T_cur);

        // Residual
        for (int i = 0; i < C * T_cur; i++)
            h_tdnn2[i] += cur[i];

        cur = h_tdnn2;
        block_outputs.push_back(cur); // save for MFA
    }

    // 5. MFA: concatenate block outputs [3 × 1024, T] → [3072, T]
    int C_mfa = 3072;
    std::vector<float> mfa_in(C_mfa * T_cur);
    for (int bi = 0; bi < 3; bi++)
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T_cur; t++)
                mfa_in[(bi * C + c) * T_cur + t] = block_outputs[bi][c * T_cur + t];

    // MFA conv(3072→3072, k=1) + BN + ReLU
    std::vector<float> mfa_out(C_mfa * T_cur);
    int T_mfa = 0;
    conv1d(mfa_in.data(), C_mfa, T_cur, m.mfa.conv_w.data(), m.mfa.conv_b.data(), C_mfa, 1, 1, 1, mfa_out.data(),
           T_mfa);
    batchnorm1d(mfa_out.data(), C_mfa, T_cur, m.mfa.bn.weight.data(), m.mfa.bn.bias.data(), m.mfa.bn.mean.data(),
                m.mfa.bn.var.data());
    relu_inplace(mfa_out.data(), C_mfa * T_cur);

    // 6. ASP: attentive statistical pooling
    // Concatenate mfa_out [3072, T] with mfa_out [3072, T] × 3 (for context) → actually just [3072, T]
    // The ASP first maps via TDNN: [3072*3=9216, T]→[128, T] ... wait, the input is 9216?
    // Actually: ASP input is concat of mfa_out repeated 3 times (global context)
    // Let me check: asp.tdnn.conv.weight is [128, 9216, 1]
    // 9216 = 3072 * 3 — global mean appended, or 3072 * T collapsed...
    // Actually for ASP: input = cat(h, mean.expand, std.expand) where mean/std are per-channel
    // h=[3072,T], mean=[3072] expanded to [3072,T], std=[3072] expanded to [3072,T]
    // → input = [9216, T]
    std::vector<float> h_mean(C_mfa, 0), h_std(C_mfa, 0);
    for (int c = 0; c < C_mfa; c++) {
        float sum = 0, sum2 = 0;
        for (int t = 0; t < T_cur; t++) {
            float v = mfa_out[c * T_cur + t];
            sum += v;
            sum2 += v * v;
        }
        h_mean[c] = sum / T_cur;
        float var = sum2 / T_cur - h_mean[c] * h_mean[c];
        h_std[c] = sqrtf(std::max(var, 1e-10f));
    }

    std::vector<float> asp_in(9216 * T_cur);
    for (int t = 0; t < T_cur; t++) {
        for (int c = 0; c < C_mfa; c++) {
            asp_in[(c) * T_cur + t] = mfa_out[c * T_cur + t];
            asp_in[(C_mfa + c) * T_cur + t] = h_mean[c];
            asp_in[(2 * C_mfa + c) * T_cur + t] = h_std[c];
        }
    }

    // ASP TDNN: [9216→128, k=1] + BN + tanh
    std::vector<float> asp_h(128 * T_cur);
    int T_asp = 0;
    conv1d(asp_in.data(), 9216, T_cur, m.asp_tdnn.conv_w.data(), m.asp_tdnn.conv_b.data(), 128, 1, 1, 1,
           asp_h.data(), T_asp);
    batchnorm1d(asp_h.data(), 128, T_cur, m.asp_tdnn.bn.weight.data(), m.asp_tdnn.bn.bias.data(),
                m.asp_tdnn.bn.mean.data(), m.asp_tdnn.bn.var.data());
    // tanh
    for (int i = 0; i < 128 * T_cur; i++)
        asp_h[i] = tanhf(asp_h[i]);

    // ASP attention: [128→3072, k=1] → softmax over time
    std::vector<float> attn(C_mfa * T_cur);
    conv1d(asp_h.data(), 128, T_cur, m.asp_conv_w.data(), m.asp_conv_b.data(), C_mfa, 1, 1, 1, attn.data(), T_asp);

    // Softmax over time dimension for each channel
    for (int c = 0; c < C_mfa; c++) {
        float mx = attn[c * T_cur];
        for (int t = 1; t < T_cur; t++)
            if (attn[c * T_cur + t] > mx)
                mx = attn[c * T_cur + t];
        float sum = 0;
        for (int t = 0; t < T_cur; t++) {
            attn[c * T_cur + t] = expf(attn[c * T_cur + t] - mx);
            sum += attn[c * T_cur + t];
        }
        for (int t = 0; t < T_cur; t++)
            attn[c * T_cur + t] /= sum;
    }

    // Weighted mean and std
    std::vector<float> w_mean(C_mfa, 0), w_std(C_mfa, 0);
    for (int c = 0; c < C_mfa; c++) {
        float wm = 0, wm2 = 0;
        for (int t = 0; t < T_cur; t++) {
            float a = attn[c * T_cur + t];
            float v = mfa_out[c * T_cur + t];
            wm += a * v;
            wm2 += a * v * v;
        }
        w_mean[c] = wm;
        float var = wm2 - wm * wm;
        w_std[c] = sqrtf(std::max(var, 1e-10f));
    }

    // Concatenate [w_mean, w_std] → [6144]
    std::vector<float> pool(6144);
    for (int c = 0; c < C_mfa; c++) {
        pool[c] = w_mean[c];
        pool[C_mfa + c] = w_std[c];
    }

    // 7. ASP BN + FC(6144→256)
    // BN on [6144, 1]
    for (int c = 0; c < 6144; c++)
        pool[c] = (pool[c] - m.asp_bn.mean[c]) / sqrtf(m.asp_bn.var[c] + 1e-5f) * m.asp_bn.weight[c] +
                  m.asp_bn.bias[c];

    // FC: linear 6144→256
    std::vector<float> emb(256, 0);
    for (int i = 0; i < 256; i++) {
        double s = m.fc_b.empty() ? 0 : m.fc_b[i];
        for (int k = 0; k < 6144; k++)
            s += pool[k] * m.fc_w[i * 6144 + k];
        emb[i] = (float)s;
    }

    // 8. Classifier: BN(256) → Linear(256→512) + BN + LeakyReLU → Linear(512→107)
    // BN(256)
    for (int c = 0; c < 256; c++)
        emb[c] = (emb[c] - m.cls_bn.mean[c]) / sqrtf(m.cls_bn.var[c] + 1e-5f) * m.cls_bn.weight[c] +
                 m.cls_bn.bias[c];

    // Linear(256→512)
    std::vector<float> h1(512, 0);
    for (int i = 0; i < 512; i++) {
        double s = m.cls_b1[i];
        for (int k = 0; k < 256; k++)
            s += emb[k] * m.cls_w1[i * 256 + k];
        h1[i] = (float)s;
    }

    // BN(512)
    for (int c = 0; c < 512; c++)
        h1[c] = (h1[c] - m.cls_bn1.mean[c]) / sqrtf(m.cls_bn1.var[c] + 1e-5f) * m.cls_bn1.weight[c] +
                m.cls_bn1.bias[c];

    // LeakyReLU(0.01)
    for (int i = 0; i < 512; i++)
        h1[i] = h1[i] > 0 ? h1[i] : 0.01f * h1[i];

    // Linear(512→107)
    std::vector<float> logits(m.n_classes, 0);
    for (int i = 0; i < m.n_classes; i++) {
        double s = m.cls_b2[i];
        for (int k = 0; k < 512; k++)
            s += h1[k] * m.cls_w2[i * 512 + k];
        logits[i] = (float)s;
    }

    // Softmax + argmax
    float mx = logits[0];
    for (int i = 1; i < m.n_classes; i++)
        if (logits[i] > mx)
            mx = logits[i];
    float sum = 0;
    for (int i = 0; i < m.n_classes; i++) {
        logits[i] = expf(logits[i] - mx);
        sum += logits[i];
    }
    int best = 0;
    float best_conf = logits[0] / sum;
    for (int i = 1; i < m.n_classes; i++) {
        float p = logits[i] / sum;
        if (p > best_conf) {
            best_conf = p;
            best = i;
        }
    }

    if (confidence)
        *confidence = best_conf;

    if (best >= 0 && best < (int)m.labels.size()) {
        ctx->last_result = m.labels[best];
        return ctx->last_result.c_str();
    }

    return nullptr;
}
