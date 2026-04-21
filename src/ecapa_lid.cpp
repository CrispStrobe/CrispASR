// ecapa_lid.cpp — ECAPA-TDNN LID runtime (ggml graph).
//
// Builds one ggml graph for the entire forward pass. All conv1d/BN/matmul
// operations use ggml ops for BLAS-accelerated computation.

#include "ecapa_lid.h"
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

// ===========================================================================
// Model — stores ggml tensor pointers into the weight buffer
// ===========================================================================

struct ecapa_model {
    int n_mels = 60, n_classes = 107, n_fft_orig = 400, fbank_bins = 201;
    int cls_type = 0;      // 0=DNN (VoxLingua107), 1=cosine (CommonLanguage)
    int lin_neurons = 256; // FC output dim (256 for VoxLingua107, 192 for CommonLanguage)
    std::vector<float> mel_fb_embedded;
    std::vector<std::string> labels;
    std::map<std::string, ggml_tensor*> tensors; // all weight tensors by name
};

struct ecapa_lid_context {
    ecapa_model model;
    int n_threads = 4;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context* weight_ctx = nullptr;
    std::string last_result;
};

// ===========================================================================
// Fbank (60-dim, 512-point FFT with bin interpolation to 400-point grid)
// ===========================================================================

static void compute_fbank(const float* pcm, int n_samples, std::vector<float>& features, int& n_frames,
                          const std::vector<float>& mel_fb_override, int n_fft_target, int n_mels_param) {
    const int sr = 16000, n_mels = n_mels_param, win_len = 400, hop = 160;
    const int N = n_fft_target, N_fft = 512;
    const float low_freq = 0.0f, high_freq = (float)sr / 2;

    int pad_len = N / 2;
    int n_padded = n_samples + 2 * pad_len;
    std::vector<float> pcm_padded(n_padded, 0.0f);
    memcpy(pcm_padded.data() + pad_len, pcm, n_samples * sizeof(float));
    for (int i = 0; i < pad_len; i++) {
        pcm_padded[pad_len - 1 - i] = pcm[std::min(i + 1, n_samples - 1)];
        pcm_padded[pad_len + n_samples + i] = pcm[std::max(n_samples - 2 - i, 0)];
    }

    n_frames = (n_padded - win_len) / hop + 1;
    if (n_frames <= 0) {
        n_frames = 0;
        return;
    }

    int bins = N / 2 + 1;
    std::vector<float> mel_fb;
    if (!mel_fb_override.empty() && (int)mel_fb_override.size() == n_mels * bins) {
        mel_fb = mel_fb_override;
    } else {
        mel_fb.resize(n_mels * bins, 0.0f);
        auto hz2mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
        auto mel2hz = [](float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };
        float ml = hz2mel(low_freq), mh = hz2mel(high_freq);
        std::vector<float> c(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            c[i] = mel2hz(ml + i * (mh - ml) / (n_mels + 1));
        for (int m = 0; m < n_mels; m++)
            for (int k = 0; k < bins; k++) {
                float f = (float)k * sr / N;
                if (f > c[m] && f <= c[m + 1] && c[m + 1] > c[m])
                    mel_fb[m * bins + k] = (f - c[m]) / (c[m + 1] - c[m]);
                else if (f > c[m + 1] && f < c[m + 2] && c[m + 2] > c[m + 1])
                    mel_fb[m * bins + k] = (c[m + 2] - f) / (c[m + 2] - c[m + 1]);
            }
    }

    std::vector<float> window(win_len);
    for (int i = 0; i < win_len; i++)
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win_len));

    features.resize(n_frames * n_mels);

    auto fft512 = [](float* r, float* im) {
        const int n = 512;
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

    const int fft_bins = N_fft / 2 + 1;
    std::vector<int> bin_lo(bins);
    std::vector<float> bin_frac(bins);
    for (int k = 0; k < bins; k++) {
        float fft_idx = (float)k * sr / N * N_fft / (float)sr;
        int lo = std::min((int)fft_idx, fft_bins - 2);
        bin_lo[k] = lo;
        bin_frac[k] = fft_idx - lo;
    }

    std::vector<float> fre(N_fft), fim(N_fft);
    for (int t = 0; t < n_frames; t++) {
        int off = t * hop;
        std::fill(fre.begin(), fre.end(), 0);
        std::fill(fim.begin(), fim.end(), 0);
        for (int i = 0; i < win_len; i++)
            fre[i] = pcm_padded[off + i] * window[i];
        fft512(fre.data(), fim.data());
        std::vector<float> fft_power(fft_bins);
        for (int j = 0; j < fft_bins; j++)
            fft_power[j] = fre[j] * fre[j] + fim[j] * fim[j];
        for (int m = 0; m < n_mels; m++) {
            float s = 0;
            for (int k = 0; k < bins; k++) {
                float pw = fft_power[bin_lo[k]] * (1 - bin_frac[k]) +
                           fft_power[std::min(bin_lo[k] + 1, fft_bins - 1)] * bin_frac[k];
                s += pw * mel_fb[m * bins + k];
            }
            features[t * n_mels + m] = 10.0f * log10f(std::max(s, 1e-10f));
        }
    }
    float global_max = *std::max_element(features.begin(), features.end());
    float floor = global_max - 80.0f;
    for (auto& v : features)
        if (v < floor)
            v = floor;
}

// ===========================================================================
// ggml graph helpers
// ===========================================================================

// Fused BN: x * (w/sqrt(v+eps)) + (b - m*w/sqrt(v+eps))
// Pre-compute scale and shift on CPU, apply as ggml_mul + ggml_add
static ggml_tensor* build_bn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, ggml_tensor* m,
                             ggml_tensor* v, ggml_tensor* eps) {
    if (!w)
        return x;
    // x: [T, C] → transpose to [C, T] for broadcasting, then back
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor* h = ggml_sub(ctx, xt, m);
    h = ggml_div(ctx, h, ggml_sqrt(ctx, ggml_add(ctx, v, eps)));
    h = ggml_mul(ctx, h, w);
    if (b)
        h = ggml_add(ctx, h, b);
    return ggml_cont(ctx, ggml_transpose(ctx, h));
}

// Conv1d(k=1) = matmul. Input [C_in, T], weight [C_in, C_out] (2D) or [1, C_in, C_out] (3D)
// Output [C_out, T]
// Conv1d(k=1) = matmul. Input x: [T, C_in], weight w: [C_in, C_out] → output [T, C_out]
static ggml_tensor* build_conv1d_k1(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    ggml_tensor* ww = (ggml_n_dims(w) > 2) ? ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]) : w;
    // x: [T, C_in] → transpose to [C_in, T]
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [C_in, T]
    // mul_mat(ww [C_in, C_out], xt [C_in, T]) → [C_out, T]
    ggml_tensor* h = ggml_mul_mat(ctx, ww, xt); // [C_out, T]
    // Add bias [C_out] — broadcasts over dim 1 (T)
    if (b)
        h = ggml_add(ctx, h, b);
    // Transpose back to [T, C_out]
    return ggml_cont(ctx, ggml_transpose(ctx, h));
}

// Conv1d with kernel > 1. Uses ggml_pad_reflect_1d for SpeechBrain compatibility.
// Input: [T, C_in], output: [T_out, C_out]
static ggml_tensor* build_conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                 int dilation) {
    int K = (int)w->ne[0];
    int pad = dilation * (K - 1) / 2;
    if (pad > 0)
        x = ggml_pad_reflect_1d(ctx, x, pad, pad);
    ggml_tensor* h = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
    if (b) {
        ggml_tensor* ht = ggml_cont(ctx, ggml_transpose(ctx, h));
        ht = ggml_add(ctx, ht, b);
        h = ggml_cont(ctx, ggml_transpose(ctx, ht));
    }
    return h;
}

// ===========================================================================
// Init
// ===========================================================================

extern "C" struct ecapa_lid_context* ecapa_lid_init(const char* model_path, int n_threads) {
    auto* ctx = new ecapa_lid_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    auto& m = ctx->model;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }
    m.n_mels = core_gguf::kv_u32(gctx, "ecapa.n_mels", 60);
    m.n_classes = core_gguf::kv_u32(gctx, "ecapa.n_classes", 107);
    m.n_fft_orig = core_gguf::kv_u32(gctx, "ecapa.n_fft", 400);
    m.fbank_bins = core_gguf::kv_u32(gctx, "ecapa.fbank_bins", 201);
    m.cls_type = core_gguf::kv_u32(gctx, "ecapa.cls_type", 0);
    m.lin_neurons = core_gguf::kv_u32(gctx, "ecapa.lin_neurons", 256);
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

    fprintf(stderr, "ecapa_lid: %d mels, %d classes, %d emb_dim, cls_type=%d, %zu labels\n", m.n_mels, m.n_classes,
            m.lin_neurons, m.cls_type, m.labels.size());

    ctx->backend = ggml_backend_init_best();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, ctx->backend, "ecapa-tdnn-lid", wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    ctx->weight_ctx = wl.ctx;
    ctx->buf = wl.buf;
    m.tensors = wl.tensors;

    // Load embedded filterbank to CPU
    {
        auto it = m.tensors.find("mel_filterbank");
        if (it != m.tensors.end()) {
            std::vector<float> raw;
            int n = (int)ggml_nelements(it->second);
            raw.resize(n);
            if (it->second->type == GGML_TYPE_F32)
                ggml_backend_tensor_get(it->second, raw.data(), 0, n * sizeof(float));
            m.mel_fb_embedded = raw;
            fprintf(stderr, "ecapa_lid: loaded filterbank [%d, %d]\n", m.n_mels, m.fbank_bins);
        }
    }

    ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, 32768, false, false);
    return ctx;
}

extern "C" void ecapa_lid_free(struct ecapa_lid_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->weight_ctx)
        ggml_free(ctx->weight_ctx);
    if (ctx->buf)
        ggml_backend_buffer_free(ctx->buf);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

// ===========================================================================
// Detect — builds ggml graph for entire forward pass
// ===========================================================================

extern "C" const char* ecapa_lid_detect(struct ecapa_lid_context* ctx, const float* samples, int n_samples,
                                        float* confidence) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    auto& m = ctx->model;
    auto& ts = m.tensors;

    constexpr int kMaxSamples = 16000 * 5;
    if (n_samples > kMaxSamples)
        n_samples = kMaxSamples;

    // Helper to get tensor by name
    auto G = [&](const std::string& name) -> ggml_tensor* {
        auto it = ts.find(name);
        return it != ts.end() ? it->second : nullptr;
    };

    // 1. Fbank on CPU
    std::vector<float> fbank;
    int T = 0;
    compute_fbank(samples, n_samples, fbank, T, m.mel_fb_embedded, m.n_fft_orig, m.n_mels);
    if (T <= 0)
        return nullptr;

    // Debug: optionally load reference fbank
    const char* ref_path = getenv("ECAPA_REF_FBANK");
    if (ref_path && *ref_path) {
        FILE* f = fopen(ref_path, "rb");
        if (f) {
            fread(fbank.data(), sizeof(float), T * m.n_mels, f);
            fclose(f);
            fprintf(stderr, "ecapa_lid: LOADED ref fbank from %s\n", ref_path);
        }
    }

    // Transpose to [n_mels, T] + mean normalize
    std::vector<float> x_data(m.n_mels * T);
    for (int c = 0; c < m.n_mels; c++) {
        float mean = 0;
        for (int t = 0; t < T; t++)
            mean += fbank[t * m.n_mels + c];
        mean /= T;
        for (int t = 0; t < T; t++)
            x_data[c * T + t] = fbank[t * m.n_mels + c] - mean;
    }

    // 2. Build ggml graph
    size_t mem = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(32768, false);
    std::vector<uint8_t> meta(mem);
    struct ggml_init_params gp = {mem, meta.data(), true};
    ggml_context* ctx0 = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input + constants
    ggml_tensor* inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, m.n_mels);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Epsilon constant for BN (must be allocated as input, not created with ggml_new_f32 in no_alloc context)
    ggml_tensor* eps_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(eps_t, "eps");
    ggml_set_input(eps_t);

    // Block 0: Conv1d(60→1024, k=5) + ReLU + BN
    ggml_tensor* h = build_conv1d(ctx0, inp, G("emb.blocks.0.conv.weight"), G("emb.blocks.0.conv.bias"), 1, 1);
    h = ggml_relu(ctx0, h);
    h = build_bn(ctx0, h, G("emb.blocks.0.bn.weight"), G("emb.blocks.0.bn.bias"), G("emb.blocks.0.bn.running_mean"),
                 G("emb.blocks.0.bn.running_var"), eps_t);

    // Blocks 1-3: SE-Res2Net
    // Each block: tdnn1(k=1) → Res2Net → tdnn2(k=1) → SE → residual
    std::vector<ggml_tensor*> block_outputs;
    for (int bi = 1; bi <= 3; bi++) {
        std::string bp = "emb.blocks." + std::to_string(bi);
        int dilation = bi + 1;
        ggml_tensor* residual = h;

        // tdnn1: Conv1d(1024→1024, k=1) + ReLU + BN
        ggml_tensor* h1 = build_conv1d_k1(ctx0, h, G(bp + ".tdnn1.conv.weight"), G(bp + ".tdnn1.conv.bias"));
        h1 = ggml_relu(ctx0, h1);
        h1 = build_bn(ctx0, h1, G(bp + ".tdnn1.bn.weight"), G(bp + ".tdnn1.bn.bias"), G(bp + ".tdnn1.bn.running_mean"),
                      G(bp + ".tdnn1.bn.running_var"), eps_t);

        // Res2Net: split 1024 channels into 8 sub-bands of 128
        // h1: [T, 1024] in ggml (ne[0]=T, ne[1]=1024)
        int sub_c = 128;
        int T_cur = (int)h1->ne[0];

        // Sub-band 0: pass through
        ggml_tensor* chunk0 = ggml_view_2d(ctx0, h1, T_cur, sub_c, h1->nb[1], 0);

        // Process sub-bands 1-7 with cumulative connections
        std::vector<ggml_tensor*> chunks;
        chunks.push_back(chunk0);
        ggml_tensor* prev = chunk0;

        for (int si = 1; si < 8; si++) {
            size_t offset = (size_t)si * sub_c * ggml_type_size(h1->type) * T_cur;
            ggml_tensor* chunk_i = ggml_view_2d(ctx0, h1, T_cur, sub_c, h1->nb[1], offset);

            ggml_tensor* sub_in;
            if (si == 1) {
                sub_in = chunk_i; // sub-band 1: no prev addition
            } else {
                sub_in = ggml_add(ctx0, chunk_i, prev);
            }

            // Conv1d(128→128, k=3, d=dilation) + ReLU + BN
            std::string sp = bp + ".res2net_block.blocks." + std::to_string(si - 1);
            ggml_tensor* sub_out =
                build_conv1d(ctx0, sub_in, G(sp + ".conv.weight"), G(sp + ".conv.bias"), 1, dilation);
            sub_out = ggml_relu(ctx0, sub_out);
            sub_out = build_bn(ctx0, sub_out, G(sp + ".bn.weight"), G(sp + ".bn.bias"), G(sp + ".bn.running_mean"),
                               G(sp + ".bn.running_var"), eps_t);

            chunks.push_back(sub_out);
            prev = sub_out;
        }

        // Concatenate sub-bands: [T, 128] × 8 → [T, 1024]
        ggml_tensor* r2_out = ggml_concat(ctx0, chunks[0], chunks[1], 1);
        for (int si = 2; si < 8; si++)
            r2_out = ggml_concat(ctx0, r2_out, chunks[si], 1);

        // tdnn2: Conv1d(1024→1024, k=1) + ReLU + BN
        ggml_tensor* h2 = build_conv1d_k1(ctx0, r2_out, G(bp + ".tdnn2.conv.weight"), G(bp + ".tdnn2.conv.bias"));
        h2 = ggml_relu(ctx0, h2);
        h2 = build_bn(ctx0, h2, G(bp + ".tdnn2.bn.weight"), G(bp + ".tdnn2.bn.bias"), G(bp + ".tdnn2.bn.running_mean"),
                      G(bp + ".tdnn2.bn.running_var"), eps_t);

        // SE: squeeze-excitation on tdnn2 output
        // Pool: mean over time → [1, 1024]
        ggml_tensor* se_pool = ggml_pool_1d(ctx0, h2, GGML_OP_POOL_AVG, T_cur, T_cur, 0);
        // conv1: 1024→128 + ReLU
        ggml_tensor* se1 =
            build_conv1d_k1(ctx0, se_pool, G(bp + ".se_block.conv1.conv.weight"), G(bp + ".se_block.conv1.conv.bias"));
        se1 = ggml_relu(ctx0, se1);
        // conv2: 128→1024 + sigmoid
        ggml_tensor* se2 =
            build_conv1d_k1(ctx0, se1, G(bp + ".se_block.conv2.conv.weight"), G(bp + ".se_block.conv2.conv.bias"));
        se2 = ggml_sigmoid(ctx0, se2);

        // Scale: h2 * se2 (broadcast se2 over time)
        h = ggml_mul(ctx0, h2, se2);

        // Residual
        h = ggml_add(ctx0, h, residual);
        block_outputs.push_back(h);
    }

    // MFA: concatenate block1-3 outputs + Conv1d(3072→3072, k=1) + ReLU + BN
    ggml_tensor* mfa_in = ggml_concat(ctx0, block_outputs[0], block_outputs[1], 1);
    mfa_in = ggml_concat(ctx0, mfa_in, block_outputs[2], 1);
    ggml_tensor* mfa = build_conv1d_k1(ctx0, mfa_in, G("emb.mfa.conv.weight"), G("emb.mfa.conv.bias"));
    mfa = ggml_relu(ctx0, mfa);
    mfa = build_bn(ctx0, mfa, G("emb.mfa.bn.weight"), G("emb.mfa.bn.bias"), G("emb.mfa.bn.running_mean"),
                   G("emb.mfa.bn.running_var"), eps_t);

    // ASP: Attentive Statistical Pooling
    // h_mean, h_std over time → cat with mfa → TDNN → tanh → conv → softmax → weighted mean+std
    // This is complex for ggml — mark mfa as output and do ASP + classifier on CPU
    ggml_set_name(mfa, "mfa_out");
    ggml_set_output(mfa);
    ggml_build_forward_expand(gf, mfa);

    // Allocate and compute graph
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "ecapa_lid: graph alloc failed\n");
        ggml_free(ctx0);
        return nullptr;
    }

    // Set input — ggml tensor [T, n_mels] is column-major: data[ic * T + t]
    // x_data is [n_mels, T] row-major: x_data[c * T + t]
    // For ggml column-major [ne[0]=T, ne[1]=IC]: data[ic * T + t] = x_data[ic * T + t]
    // They're the SAME layout! Just copy directly.
    ggml_backend_tensor_set(inp, x_data.data(), 0, m.n_mels * T * sizeof(float));
    float eps_val = 1e-5f;
    ggml_backend_tensor_set(eps_t, &eps_val, 0, sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ecapa_lid: graph compute failed\n");
        ggml_free(ctx0);
        return nullptr;
    }


    // Read MFA output: [T, 3072] in ggml
    ggml_tensor* mfa_t = ggml_graph_get_tensor(gf, "mfa_out");
    int T_mfa = (int)mfa_t->ne[0];
    int C_mfa = (int)mfa_t->ne[1]; // 3072
    std::vector<float> mfa_data(T_mfa * C_mfa);
    ggml_backend_tensor_get(mfa_t, mfa_data.data(), 0, T_mfa * C_mfa * sizeof(float));
    ggml_free(ctx0);

    // MFA data from ggml is column-major [ne[0]=T, ne[1]=C]: data[c*T+t]
    // Our CPU code expects [C, T]: data[c*T+t] — SAME layout! Just alias.
    std::vector<float>& mfa_ct = mfa_data;

    // ASP on CPU (small computation, not worth ggml overhead)
    std::vector<float> h_mean(C_mfa, 0), h_std(C_mfa, 0);
    for (int c = 0; c < C_mfa; c++) {
        float sum = 0, sum2 = 0;
        for (int t = 0; t < T_mfa; t++) {
            float v = mfa_ct[c * T_mfa + t];
            sum += v;
            sum2 += v * v;
        }
        h_mean[c] = sum / T_mfa;
        h_std[c] = sqrtf(std::max(sum2 / T_mfa - h_mean[c] * h_mean[c], 1e-10f));
    }

    // ASP input: [9216, T] = cat(mfa, mean_expanded, std_expanded)
    auto read_f32 = [](ggml_tensor* t, std::vector<float>& out) {
        if (!t) {
            out.clear();
            return;
        }
        int n = (int)ggml_nelements(t);
        out.resize(n);
        if (t->type == GGML_TYPE_F32)
            ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        else if (t->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(uint16_t));
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)tmp.data(), out.data(), n);
        }
    };

    // ASP TDNN + attention + pooling
    std::vector<float> asp_tdnn_w, asp_tdnn_b, asp_bn_w, asp_bn_b, asp_bn_m, asp_bn_v;
    read_f32(G("emb.asp.tdnn.conv.weight"), asp_tdnn_w);
    read_f32(G("emb.asp.tdnn.conv.bias"), asp_tdnn_b);
    read_f32(G("emb.asp.tdnn.bn.weight"), asp_bn_w);
    read_f32(G("emb.asp.tdnn.bn.bias"), asp_bn_b);
    read_f32(G("emb.asp.tdnn.bn.running_mean"), asp_bn_m);
    read_f32(G("emb.asp.tdnn.bn.running_var"), asp_bn_v);

    // TDNN: [9216→128, k=1]
    std::vector<float> asp_h(128 * T_mfa, 0);
    for (int i = 0; i < 128; i++) {
        for (int t = 0; t < T_mfa; t++) {
            double s = asp_tdnn_b.empty() ? 0 : asp_tdnn_b[i];
            for (int c = 0; c < C_mfa; c++) {
                s += mfa_ct[c * T_mfa + t] * asp_tdnn_w[i * 9216 + c];
                s += h_mean[c] * asp_tdnn_w[i * 9216 + C_mfa + c];
                s += h_std[c] * asp_tdnn_w[i * 9216 + 2 * C_mfa + c];
            }
            asp_h[i * T_mfa + t] = (float)s;
        }
    }
    // BN + tanh
    for (int c = 0; c < 128; c++) {
        float scale = asp_bn_w[c] / sqrtf(asp_bn_v[c] + 1e-5f);
        float shift = asp_bn_b[c] - asp_bn_m[c] * scale;
        for (int t = 0; t < T_mfa; t++)
            asp_h[c * T_mfa + t] = tanhf(asp_h[c * T_mfa + t] * scale + shift);
    }

    // Attention conv: [128→3072, k=1]
    std::vector<float> asp_conv_w, asp_conv_b;
    read_f32(G("emb.asp.conv.weight"), asp_conv_w);
    read_f32(G("emb.asp.conv.bias"), asp_conv_b);

    std::vector<float> attn(C_mfa * T_mfa, 0);
    for (int c = 0; c < C_mfa; c++) {
        for (int t = 0; t < T_mfa; t++) {
            double s = asp_conv_b.empty() ? 0 : asp_conv_b[c];
            for (int k = 0; k < 128; k++)
                s += asp_h[k * T_mfa + t] * asp_conv_w[c * 128 + k];
            attn[c * T_mfa + t] = (float)s;
        }
    }
    // Softmax over time
    for (int c = 0; c < C_mfa; c++) {
        float mx = attn[c * T_mfa];
        for (int t = 1; t < T_mfa; t++)
            if (attn[c * T_mfa + t] > mx)
                mx = attn[c * T_mfa + t];
        float sum = 0;
        for (int t = 0; t < T_mfa; t++) {
            attn[c * T_mfa + t] = expf(attn[c * T_mfa + t] - mx);
            sum += attn[c * T_mfa + t];
        }
        for (int t = 0; t < T_mfa; t++)
            attn[c * T_mfa + t] /= sum;
    }

    // Weighted mean + std
    std::vector<float> w_mean(C_mfa, 0), w_std(C_mfa, 0);
    for (int c = 0; c < C_mfa; c++) {
        float wm = 0, wm2 = 0;
        for (int t = 0; t < T_mfa; t++) {
            float a = attn[c * T_mfa + t], v = mfa_ct[c * T_mfa + t];
            wm += a * v;
            wm2 += a * v * v;
        }
        w_mean[c] = wm;
        w_std[c] = sqrtf(std::max(wm2 - wm * wm, 1e-10f));
    }

    // Pool: [6144] = cat(w_mean, w_std)
    std::vector<float> pool(6144);
    for (int c = 0; c < C_mfa; c++) {
        pool[c] = w_mean[c];
        pool[C_mfa + c] = w_std[c];
    }

    // ASP BN
    std::vector<float> aspbn_w, aspbn_b, aspbn_m, aspbn_v;
    read_f32(G("emb.asp_bn.norm.weight"), aspbn_w);
    read_f32(G("emb.asp_bn.norm.bias"), aspbn_b);
    read_f32(G("emb.asp_bn.norm.running_mean"), aspbn_m);
    read_f32(G("emb.asp_bn.norm.running_var"), aspbn_v);
    for (int c = 0; c < 6144; c++)
        pool[c] = (pool[c] - aspbn_m[c]) / sqrtf(aspbn_v[c] + 1e-5f) * aspbn_w[c] + aspbn_b[c];

    // FC: Linear(6144→lin_neurons)
    int lin_n = m.lin_neurons;
    std::vector<float> fc_w, fc_b;
    read_f32(G("emb.fc.conv.weight"), fc_w);
    read_f32(G("emb.fc.conv.bias"), fc_b);
    std::vector<float> emb(lin_n, 0);
    for (int i = 0; i < lin_n; i++) {
        double s = fc_b.empty() ? 0 : fc_b[i];
        for (int k = 0; k < 6144; k++)
            s += pool[k] * fc_w[i * 6144 + k];
        emb[i] = (float)s;
    }

    // Classifier — two types:
    // Type 0 (DNN/VoxLingua107): BN → Linear → BN → LeakyReLU → Linear
    // Type 1 (Cosine/CommonLanguage): normalize(emb) @ normalize(weight)^T
    int emb_dim = (int)emb.size();
    std::vector<float> logits(m.n_classes, 0);

    if (m.cls_type == 1) {
        // Cosine classifier: F.linear(F.normalize(emb), F.normalize(weight))
        std::vector<float> cls_w;
        read_f32(G("cls.weight"), cls_w); // [n_classes, emb_dim]
        // L2 normalize emb
        float emb_norm = 0;
        for (float v : emb)
            emb_norm += v * v;
        emb_norm = 1.0f / (sqrtf(emb_norm) + 1e-12f);
        for (float& v : emb)
            v *= emb_norm;
        // Cosine similarity with each class weight
        for (int i = 0; i < m.n_classes; i++) {
            float w_norm = 0;
            for (int k = 0; k < emb_dim; k++)
                w_norm += cls_w[i * emb_dim + k] * cls_w[i * emb_dim + k];
            w_norm = 1.0f / (sqrtf(w_norm) + 1e-12f);
            double s = 0;
            for (int k = 0; k < emb_dim; k++)
                s += emb[k] * cls_w[i * emb_dim + k] * w_norm;
            logits[i] = (float)s;
        }
    } else {
        // DNN classifier: BN(emb) → Linear(emb→512) → BN → LeakyReLU → Linear(512→n_classes)
        std::vector<float> cls_bn_w, cls_bn_b, cls_bn_m, cls_bn_v;
        read_f32(G("cls.bn.weight"), cls_bn_w);
        read_f32(G("cls.bn.bias"), cls_bn_b);
        read_f32(G("cls.bn.running_mean"), cls_bn_m);
        read_f32(G("cls.bn.running_var"), cls_bn_v);
        for (int c = 0; c < emb_dim; c++)
            emb[c] = (emb[c] - cls_bn_m[c]) / sqrtf(cls_bn_v[c] + 1e-5f) * cls_bn_w[c] + cls_bn_b[c];

        std::vector<float> cls_w1, cls_b1;
        read_f32(G("cls.DNN.block_0.linear.weight"), cls_w1);
        read_f32(G("cls.DNN.block_0.linear.bias"), cls_b1);
        int hidden = (int)cls_b1.size(); // 512
        std::vector<float> h1(hidden, 0);
        for (int i = 0; i < hidden; i++) {
            double s = cls_b1[i];
            for (int k = 0; k < emb_dim; k++)
                s += emb[k] * cls_w1[i * emb_dim + k];
            h1[i] = (float)s;
        }
        std::vector<float> cls_bn1_w, cls_bn1_b, cls_bn1_m, cls_bn1_v;
        read_f32(G("cls.DNN.block_0.bn.weight"), cls_bn1_w);
        read_f32(G("cls.DNN.block_0.bn.bias"), cls_bn1_b);
        read_f32(G("cls.DNN.block_0.bn.running_mean"), cls_bn1_m);
        read_f32(G("cls.DNN.block_0.bn.running_var"), cls_bn1_v);
        for (int c = 0; c < hidden; c++) {
            h1[c] = (h1[c] - cls_bn1_m[c]) / sqrtf(cls_bn1_v[c] + 1e-5f) * cls_bn1_w[c] + cls_bn1_b[c];
            h1[c] = h1[c] > 0 ? h1[c] : 0.01f * h1[c];
        }
        std::vector<float> cls_w2, cls_b2;
        read_f32(G("cls.out.w.weight"), cls_w2);
        read_f32(G("cls.out.w.bias"), cls_b2);
        for (int i = 0; i < m.n_classes; i++) {
            double s = cls_b2[i];
            for (int k = 0; k < hidden; k++)
                s += h1[k] * cls_w2[i * hidden + k];
            logits[i] = (float)s;
        }
    }

    // Softmax + argmax
    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0;
    for (auto& v : logits) {
        v = expf(v - mx);
        sum += v;
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
