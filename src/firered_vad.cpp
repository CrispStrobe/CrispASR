// firered_vad.cpp — FireRedVAD runtime (DFSMN-based VAD).
//
// The DFSMN is tiny (588K params) — runs entirely on CPU with simple loops.
// No ggml graph needed. Model loads in <10ms, inference in <50ms for 60s audio.

#include "firered_vad.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
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
// Model
// ===========================================================================

struct firered_vad_hparams {
    int R = 8;     // FSMN blocks
    int H = 256;   // hidden size
    int P = 128;   // projection size
    int N1 = 20;   // lookback context
    int N2 = 20;   // lookahead context
    int S1 = 1;    // lookback stride
    int S2 = 1;    // lookahead stride
    int idim = 80; // input dim (fbank bins)
    int odim = 1;  // output dim
};

struct fsmn_block {
    std::vector<float> fc1_w, fc1_b; // [H, P] + [H]
    std::vector<float> fc2_w;        // [P, H] (no bias)
    std::vector<float> lb_w, la_w;   // lookback/ahead [P, 1, N]
};

struct firered_vad_model {
    firered_vad_hparams hp;
    // Initial layers
    std::vector<float> fc1_w, fc1_b;       // [H, idim] + [H]
    std::vector<float> fc2_w, fc2_b;       // [P, H] + [P]
    std::vector<float> fsmn1_lb, fsmn1_la; // first FSMN lookback/ahead
    // FSMN blocks (R-1 = 7)
    std::vector<fsmn_block> blocks;
    // Final DNN + output
    std::vector<float> dnn_w, dnn_b; // [H, P] + [H]
    std::vector<float> out_w, out_b; // [1, H] + [1]
    // CMVN
    std::vector<float> cmvn_mean, cmvn_std;
};

struct firered_vad_context {
    firered_vad_model model;
};

// ===========================================================================
// Helpers
// ===========================================================================

static void read_f32(ggml_tensor* t, std::vector<float>& out) {
    int n = (int)ggml_nelements(t);
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
}

// Linear: out[i] = sum_j w[i*K+j] * x[j] + b[i], for each time step
static void cpu_linear(const float* x, const float* w, const float* b, float* out, int T, int K, int N) {
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++)
                s += x[t * K + k] * w[n * K + k];
            out[t * N + n] = (float)s + (b ? b[n] : 0.0f);
        }
    }
}

static void cpu_relu(float* x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0)
            x[i] = 0;
}

// FSMN: memory = x + lookback_conv(x) + lookahead_conv(x)
// Replicates PyTorch's Conv1d exactly:
//   lookback_filter: Conv1d(P,P, N1, stride=1, padding=(N1-1)*S1, dilation=S1, groups=P)
//   lookahead_filter: Conv1d(P,P, N2, stride=1, padding=(N2-1)*S2, dilation=S2, groups=P)
// Then: lookback = lb_conv(x)[:,:,:-(N1-1)*S1]  (trim right padding)
//       lookahead = F.pad(la_conv(x)[:,:,N2*S2:], (0, S2))  (trim left, pad right)
static void cpu_fsmn(const float* x, float* out, const float* lb_w, const float* la_w, int T, int P, int N1, int S1,
                     int N2, int S2) {
    // x layout: [T, P] row-major
    // Work in [P, T] layout for conv (transpose)
    std::vector<float> x_pt(P * T);
    for (int p = 0; p < P; p++)
        for (int t = 0; t < T; t++)
            x_pt[p * T + t] = x[t * P + p];

    // Lookback: Conv1d(padding=(N1-1)*S1, dilation=S1, groups=P)
    // Output length = T + 2*(N1-1)*S1 - S1*(N1-1) = T + (N1-1)*S1
    int lb_pad = (N1 - 1) * S1;
    int lb_out_len = T + lb_pad; // T + (N1-1)*S1
    std::vector<float> lb_conv(P * lb_out_len, 0.0f);
    for (int p = 0; p < P; p++) {
        for (int t_out = 0; t_out < lb_out_len; t_out++) {
            float s = 0;
            for (int k = 0; k < N1; k++) {
                int t_in = t_out - lb_pad + k * S1; // input index with padding offset
                if (t_in >= 0 && t_in < T)
                    s += lb_w[p * N1 + k] * x_pt[p * T + t_in];
            }
            lb_conv[p * lb_out_len + t_out] = s;
        }
    }
    // Trim: lookback[:,:,:-(N1-1)*S1] if (N1-1)*S1 > 0, else full
    // lb_out_len - lb_pad = T, so we take first T elements
    // → lb_trimmed[p][t] = lb_conv[p * lb_out_len + t] for t < T

    // Start with residual
    memcpy(out, x, T * P * sizeof(float));
    // Add lookback
    for (int p = 0; p < P; p++)
        for (int t = 0; t < T; t++)
            out[t * P + p] += lb_conv[p * lb_out_len + t];

    // Lookahead: Conv1d(padding=(N2-1)*S2, dilation=S2, groups=P)
    if (N2 > 0 && T > 1) {
        int la_pad = (N2 - 1) * S2;
        int la_out_len = T + la_pad;
        std::vector<float> la_conv(P * la_out_len, 0.0f);
        for (int p = 0; p < P; p++) {
            for (int t_out = 0; t_out < la_out_len; t_out++) {
                float s = 0;
                for (int k = 0; k < N2; k++) {
                    int t_in = t_out - la_pad + k * S2;
                    if (t_in >= 0 && t_in < T)
                        s += la_w[p * N2 + k] * x_pt[p * T + t_in];
                }
                la_conv[p * la_out_len + t_out] = s;
            }
        }
        // Python: memory += F.pad(lookahead[:, :, N2*S2:], (0, S2))
        // la_conv has la_out_len elements. Skip first N2*S2 elements.
        // Remaining: la_out_len - N2*S2 = T + (N2-1)*S2 - N2*S2 = T - S2
        // Then pad right by S2 → length T
        int skip = N2 * S2;
        for (int p = 0; p < P; p++)
            for (int t = 0; t < T; t++) {
                int la_idx = skip + t;
                if (la_idx < la_out_len)
                    out[t * P + p] += la_conv[p * la_out_len + la_idx];
                // else: padded with 0 (no-op)
            }
    }
}

// Compute 80-dim fbank (same as firered_asr)
static void compute_fbank_vad(const float* pcm, int n_samples, std::vector<float>& features, int& n_frames) {
    const int n_fft = 512, hop = 160, win = 400, n_mels = 80, sr = 16000;
    const float preemph = 0.97f, low_freq = 20.0f, high_freq = (float)sr / 2;

    n_frames = (n_samples - win) / hop + 1;
    if (n_frames <= 0) {
        n_frames = 0;
        return;
    }

    int bins = n_fft / 2 + 1;
    std::vector<float> mel_fb(n_mels * bins, 0.0f);
    {
        auto hz2mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };
        auto mel2hz = [](float m) { return 700.0f * (expf(m / 1127.0f) - 1.0f); };
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
    std::vector<float> window(win);
    for (int i = 0; i < win; i++) {
        float h = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (win - 1));
        window[i] = powf(h, 0.85f);
    }
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
    // FireRedVAD was trained with int16 fbank input.
    // Scale float32 (-1..1) to int16 range (-32768..32767) before fbank.
    const float scale_to_i16 = 32768.0f;

    for (int t = 0; t < n_frames; t++) {
        int off = t * hop;
        std::vector<float> fr(win);
        float dc = 0;
        for (int i = 0; i < win; i++) {
            fr[i] = ((off + i < n_samples) ? pcm[off + i] : 0.0f) * scale_to_i16;
            dc += fr[i];
        }
        dc /= win;
        for (int i = 0; i < win; i++)
            fr[i] -= dc;
        for (int i = win - 1; i > 0; i--)
            fr[i] -= preemph * fr[i - 1];
        fr[0] -= preemph * fr[0];
        std::fill(fre.begin(), fre.end(), 0);
        std::fill(fim.begin(), fim.end(), 0);
        for (int i = 0; i < win; i++)
            fre[i] = fr[i] * window[i];
        fft(fre.data(), fim.data(), n_fft);
        for (int m = 0; m < n_mels; m++) {
            float s = 0;
            for (int k = 0; k < bins; k++)
                s += (fre[k] * fre[k] + fim[k] * fim[k]) * mel_fb[m * bins + k];
            features[t * n_mels + m] = logf(std::max(s, 1.1920929e-7f));
        }
    }
}

// ===========================================================================
// Init / Free
// ===========================================================================

extern "C" struct firered_vad_context* firered_vad_init(const char* model_path) {
    auto* ctx = new firered_vad_context();
    auto& m = ctx->model;
    auto& hp = m.hp;

    // Read hparams
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        delete ctx;
        return nullptr;
    }
    hp.R = core_gguf::kv_u32(gctx, "firered_vad.R", hp.R);
    hp.H = core_gguf::kv_u32(gctx, "firered_vad.H", hp.H);
    hp.P = core_gguf::kv_u32(gctx, "firered_vad.P", hp.P);
    hp.N1 = core_gguf::kv_u32(gctx, "firered_vad.N1", hp.N1);
    hp.N2 = core_gguf::kv_u32(gctx, "firered_vad.N2", hp.N2);
    hp.S1 = core_gguf::kv_u32(gctx, "firered_vad.S1", hp.S1);
    hp.S2 = core_gguf::kv_u32(gctx, "firered_vad.S2", hp.S2);
    hp.idim = core_gguf::kv_u32(gctx, "firered_vad.idim", hp.idim);
    hp.odim = core_gguf::kv_u32(gctx, "firered_vad.odim", hp.odim);
    gguf_free(gctx);

    // Load weights
    ggml_backend_t backend = ggml_backend_init_best();
    if (!backend) {
        delete ctx;
        return nullptr;
    }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "firered_vad", wl)) {
        ggml_backend_free(backend);
        delete ctx;
        return nullptr;
    }

    auto get = [&](const char* name) -> ggml_tensor* {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Read all weights into CPU vectors
    auto rd = [&](const char* name, std::vector<float>& v) {
        ggml_tensor* t = get(name);
        if (t)
            read_f32(t, v);
    };

    rd("dfsmn.fc1.0.weight", m.fc1_w);
    rd("dfsmn.fc1.0.bias", m.fc1_b);
    rd("dfsmn.fc2.0.weight", m.fc2_w);
    rd("dfsmn.fc2.0.bias", m.fc2_b);
    rd("dfsmn.fsmn1.lookback_filter.weight", m.fsmn1_lb);
    rd("dfsmn.fsmn1.lookahead_filter.weight", m.fsmn1_la);

    int R_blocks = hp.R - 1; // first FSMN is separate
    m.blocks.resize(R_blocks);
    for (int i = 0; i < R_blocks; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc1.0.weight", i);
        rd(buf, m.blocks[i].fc1_w);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc1.0.bias", i);
        rd(buf, m.blocks[i].fc1_b);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc2.weight", i);
        rd(buf, m.blocks[i].fc2_w);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fsmn.lookback_filter.weight", i);
        rd(buf, m.blocks[i].lb_w);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fsmn.lookahead_filter.weight", i);
        rd(buf, m.blocks[i].la_w);
    }

    rd("dfsmn.dnns.0.weight", m.dnn_w);
    rd("dfsmn.dnns.0.bias", m.dnn_b);
    rd("out.weight", m.out_w);
    rd("out.bias", m.out_b);

    // CMVN
    rd("cmvn.mean", m.cmvn_mean);
    rd("cmvn.std", m.cmvn_std);

    // Clean up weight loading context
    core_gguf::free_weights(wl);
    ggml_backend_free(backend);

    return ctx;
}

extern "C" void firered_vad_free(struct firered_vad_context* ctx) {
    delete ctx;
}

// ===========================================================================
// Detect
// ===========================================================================

extern "C" int firered_vad_detect(struct firered_vad_context* ctx, const float* samples, int n_samples,
                                  struct firered_vad_segment** segments, int* n_segments, float threshold,
                                  float min_speech_sec, float min_silence_sec) {
    if (!ctx || !samples || n_samples <= 0)
        return -1;
    auto& m = ctx->model;
    auto& hp = m.hp;

    // Compute fbank
    std::vector<float> features;
    int n_frames = 0;
    compute_fbank_vad(samples, n_samples, features, n_frames);
    if (n_frames <= 0)
        return -1;

    // Apply CMVN
    if (!m.cmvn_mean.empty()) {
        for (int t = 0; t < n_frames; t++)
            for (int f = 0; f < hp.idim; f++)
                features[t * hp.idim + f] = (features[t * hp.idim + f] - m.cmvn_mean[f]) / m.cmvn_std[f];
    }

    // Forward pass
    int T = n_frames;

    // fc1: [T, 80] → [T, 256] + ReLU
    std::vector<float> h(T * hp.H);
    cpu_linear(features.data(), m.fc1_w.data(), m.fc1_b.data(), h.data(), T, hp.idim, hp.H);
    cpu_relu(h.data(), T * hp.H);

    fprintf(stderr, "  fc1[0,:5]=[%.2f,%.2f,%.2f,%.2f,%.2f]\n", h[0], h[1], h[2], h[3], h[4]);
    // fc2: [T, 256] → [T, 128] + ReLU
    std::vector<float> p(T * hp.P);
    cpu_linear(h.data(), m.fc2_w.data(), m.fc2_b.data(), p.data(), T, hp.H, hp.P);
    cpu_relu(p.data(), T * hp.P);
    fprintf(stderr, "  fc2[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", p[0], p[1], p[2], p[3], p[4]);

    // fsmn1
    std::vector<float> mem(T * hp.P);
    cpu_fsmn(p.data(), mem.data(), m.fsmn1_lb.data(), m.fsmn1_la.data(), T, hp.P, hp.N1, hp.S1, hp.N2, hp.S2);

    fprintf(stderr, "  fsmn1[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", mem[0], mem[1], mem[2], mem[3], mem[4]);

    // FSMN blocks
    std::vector<float> tmp_h(T * hp.H), tmp_p(T * hp.P), tmp_mem(T * hp.P);
    for (int i = 0; i < (int)m.blocks.size(); i++) {
        auto& b = m.blocks[i];
        // fc1: [T, P] → [T, H] + ReLU
        cpu_linear(mem.data(), b.fc1_w.data(), b.fc1_b.data(), tmp_h.data(), T, hp.P, hp.H);
        cpu_relu(tmp_h.data(), T * hp.H);
        // fc2: [T, H] → [T, P] (no bias, no ReLU)
        cpu_linear(tmp_h.data(), b.fc2_w.data(), nullptr, tmp_p.data(), T, hp.H, hp.P);
        // FSMN
        cpu_fsmn(tmp_p.data(), tmp_mem.data(), b.lb_w.data(), b.la_w.data(), T, hp.P, hp.N1, hp.S1, hp.N2, hp.S2);
        if (i == 0) {
            fprintf(stderr, "  b0.fc1[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f] nonzero=%d\n", tmp_h[0], tmp_h[1], tmp_h[2],
                    tmp_h[3], tmp_h[4], [&] {
                        int c = 0;
                        for (int j = 0; j < hp.H; j++)
                            if (tmp_h[j] != 0)
                                c++;
                        return c;
                    }());
            fprintf(stderr, "  b0.fc2[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", tmp_p[0], tmp_p[1], tmp_p[2], tmp_p[3],
                    tmp_p[4]);
            fprintf(stderr, "  b0.fsmn[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", tmp_mem[0], tmp_mem[1], tmp_mem[2],
                    tmp_mem[3], tmp_mem[4]);
        }
        // Skip connection
        for (int j = 0; j < T * hp.P; j++)
            mem[j] = tmp_mem[j] + mem[j];
        if (i == 0)
            fprintf(stderr, "  block0[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", mem[0], mem[1], mem[2], mem[3], mem[4]);
    }

    fprintf(stderr, "  after_blocks[0,:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", mem[0], mem[1], mem[2], mem[3], mem[4]);

    // DNN: [T, P] → [T, H] + ReLU
    cpu_linear(mem.data(), m.dnn_w.data(), m.dnn_b.data(), h.data(), T, hp.P, hp.H);
    cpu_relu(h.data(), T * hp.H);

    // Output: [T, H] → [T, 1] + sigmoid
    std::vector<float> probs(T);
    cpu_linear(h.data(), m.out_w.data(), m.out_b.data(), probs.data(), T, hp.H, 1);
    for (int t = 0; t < T; t++)
        probs[t] = 1.0f / (1.0f + expf(-probs[t]));

    // Debug: show probability stats
    float max_p = 0, mean_p = 0;
    int speech_frames = 0;
    for (int t = 0; t < T; t++) {
        if (probs[t] > max_p)
            max_p = probs[t];
        mean_p += probs[t];
        if (probs[t] > 0.3f)
            speech_frames++;
    }
    mean_p /= T;
    fprintf(stderr, "firered_vad: %d frames, max_prob=%.4f, mean_prob=%.4f, speech(>0.3)=%d\n", T, max_p, mean_p,
            speech_frames);
    fprintf(stderr, "  fbank[0,:3]=[%.2f,%.2f,%.2f] prob[0:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n", features[0], features[1],
            features[2], probs[0], probs[1], probs[2], probs[3], probs[4]);

    // Convert frame probabilities to segments
    float frame_sec = 0.01f; // 10ms per frame
    std::vector<firered_vad_segment> segs;
    bool in_speech = false;
    float seg_start = 0;
    int min_speech_frames = (int)(min_speech_sec / frame_sec);
    int min_silence_frames = (int)(min_silence_sec / frame_sec);

    // Simple thresholding with hysteresis
    int speech_count = 0, silence_count = 0;
    for (int t = 0; t < T; t++) {
        if (probs[t] >= threshold) {
            speech_count++;
            silence_count = 0;
            if (!in_speech && speech_count >= min_speech_frames) {
                in_speech = true;
                seg_start = (t - speech_count + 1) * frame_sec;
            }
        } else {
            silence_count++;
            speech_count = 0;
            if (in_speech && silence_count >= min_silence_frames) {
                in_speech = false;
                firered_vad_segment seg;
                seg.start_sec = seg_start;
                seg.end_sec = (t - silence_count + 1) * frame_sec;
                if (seg.end_sec - seg.start_sec >= min_speech_sec)
                    segs.push_back(seg);
            }
        }
    }
    if (in_speech) {
        firered_vad_segment seg;
        seg.start_sec = seg_start;
        seg.end_sec = T * frame_sec;
        segs.push_back(seg);
    }

    // Return segments
    *n_segments = (int)segs.size();
    if (segs.empty()) {
        *segments = nullptr;
        return 0;
    }
    *segments = (firered_vad_segment*)malloc(segs.size() * sizeof(firered_vad_segment));
    memcpy(*segments, segs.data(), segs.size() * sizeof(firered_vad_segment));
    return 0;
}
