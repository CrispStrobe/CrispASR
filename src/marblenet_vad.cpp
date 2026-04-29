// marblenet_vad.cpp — NVIDIA MarbleNet VAD runtime (1D separable CNN, 91.5K params).
//
// Architecture: 6 Jasper blocks (depthwise separable conv + BN-fused pointwise + ReLU)
// + Linear(128→2) + softmax. Input: 80-bin mel at 16kHz. Output: per-frame speech prob.
//
// Model: nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 (Apache-like, 6 languages).

#include "marblenet_vad.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
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

// ── Model ──────────────────────────────────────────────────────────────────

struct mbn_sub_block {
    ggml_tensor* dw_w = nullptr; // depthwise conv [C, 1, K]
    ggml_tensor* pw_w = nullptr; // pointwise conv [C_out, C_in, 1] (BN fused)
    ggml_tensor* pw_b = nullptr; // fused bias
};

struct mbn_block {
    int filters = 0;
    int kernel = 0;
    int stride = 1;
    int dilation = 1;
    int repeat = 1;
    bool residual = false;
    bool separable = false;
    std::vector<mbn_sub_block> subs;
    ggml_tensor* res_w = nullptr; // residual 1x1 conv (BN fused)
    ggml_tensor* res_b = nullptr;
};

struct mbn_model {
    int n_mels = 80;
    int n_fft = 512;
    int win_length = 400;
    int hop_length = 160;
    int n_blocks = 6;
    int num_classes = 2;
    float frame_ms = 10.0f; // hop_length / sr * 1000; stride-2 in block0 → 20ms output

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    ggml_tensor* mel_fb = nullptr;  // [80, 257]
    ggml_tensor* mel_win = nullptr; // [400]

    std::vector<mbn_block> blocks;

    ggml_tensor* dec_w = nullptr; // [2, 128]
    ggml_tensor* dec_b = nullptr; // [2]
};

struct marblenet_vad_context {
    mbn_model model;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
};

// ── Mel ────────────────────────────────────────────────────────────────────

static void mbn_fft_dft(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            float a = -2.0f * (float)M_PI * k * n / N;
            re += in[n] * cosf(a);
            im += in[n] * sinf(a);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

static void mbn_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    int h = N / 2;
    if (N != h * 2) {
        mbn_fft_dft(in, N, out);
        return;
    }
    float* ev = in + N;
    for (int i = 0; i < h; i++)
        ev[i] = in[2 * i];
    float* ev_f = out + 2 * N;
    mbn_fft(ev, h, ev_f);
    float* od = ev;
    for (int i = 0; i < h; i++)
        od[i] = in[2 * i + 1];
    float* od_f = ev_f + N;
    mbn_fft(od, h, od_f);
    for (int k = 0; k < h; k++) {
        float a = -2.0f * (float)M_PI * k / N;
        float re = cosf(a), im = sinf(a);
        float or_ = od_f[2 * k], oi = od_f[2 * k + 1];
        out[2 * k] = ev_f[2 * k] + re * or_ - im * oi;
        out[2 * k + 1] = ev_f[2 * k + 1] + re * oi + im * or_;
        out[2 * (k + h)] = ev_f[2 * k] - re * or_ + im * oi;
        out[2 * (k + h) + 1] = ev_f[2 * k + 1] - re * oi - im * or_;
    }
}

static std::vector<float> mbn_compute_mel(const float* pcm, int n_samples, const float* hann, int win_len,
                                          const float* fb, int n_mels, int n_fft, int hop, int* out_T) {
    const int n_freqs = n_fft / 2 + 1;
    // NeMo uses center=True: pad n_fft/2 zeros on each side
    int pad = n_fft / 2;
    int padded_len = n_samples + 2 * pad;
    std::vector<float> padded(padded_len, 0);
    memcpy(padded.data() + pad, pcm, n_samples * sizeof(float));

    int n_frames = (padded_len - n_fft) / hop + 1;
    if (n_frames <= 0) {
        *out_T = 0;
        return {};
    }

    std::vector<float> mel((size_t)n_mels * n_frames, 0);
    std::vector<float> si(4 * n_fft, 0), so(8 * n_fft, 0);

    for (int t = 0; t < n_frames; t++) {
        // Window (win_len) centered in n_fft-sized frame, zero-padded
        for (int i = 0; i < n_fft; i++)
            si[i] = 0;
        int frame_start = t * hop;
        for (int i = 0; i < win_len && (frame_start + i) < padded_len; i++)
            si[i] = padded[frame_start + i] * hann[i];
        mbn_fft(si.data(), n_fft, so.data());
        // Power spectrum
        for (int f = 0; f < n_freqs; f++) {
            float pw = so[2 * f] * so[2 * f] + so[2 * f + 1] * so[2 * f + 1];
            // Mel projection: fb is [n_mels, n_freqs]
            for (int m = 0; m < n_mels; m++)
                mel[(size_t)m * n_frames + t] += pw * fb[m * n_freqs + f];
        }
    }
    // Log + dither
    for (auto& v : mel)
        v = logf(v + 1e-5f);

    *out_T = n_frames;
    return mel; // [n_mels, T] row-major
}

// ── Init ───────────────────────────────────────────────────────────────────

extern "C" struct marblenet_vad_context* marblenet_vad_init(const char* path) {
    auto* ctx = new marblenet_vad_context();
    auto& m = ctx->model;

    struct gguf_init_params gp = {true, &m.ctx_w};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "marblenet_vad: failed to open '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    auto gu32 = [&](const char* k, uint32_t d) -> uint32_t {
        int64_t id = gguf_find_key(gctx, k);
        return id >= 0 ? gguf_get_val_u32(gctx, id) : d;
    };

    m.n_mels = (int)gu32("marblenet.n_mels", 80);
    m.n_fft = (int)gu32("marblenet.n_fft", 512);
    m.win_length = (int)gu32("marblenet.win_length", 400);
    m.hop_length = (int)gu32("marblenet.hop_length", 160);
    m.n_blocks = (int)gu32("marblenet.n_blocks", 6);
    m.num_classes = (int)gu32("marblenet.num_classes", 2);

    // Read block configs
    m.blocks.resize(m.n_blocks);
    for (int i = 0; i < m.n_blocks; i++) {
        char buf[64];
        auto gbi = [&](const char* f, uint32_t d) -> uint32_t {
            snprintf(buf, sizeof(buf), "marblenet.block.%d.%s", i, f);
            return gu32(buf, d);
        };
        auto& b = m.blocks[i];
        b.filters = (int)gbi("filters", 128);
        b.kernel = (int)gbi("kernel", 11);
        b.stride = (int)gbi("stride", 1);
        b.dilation = (int)gbi("dilation", 1);
        b.repeat = (int)gbi("repeat", 1);
        b.residual = gbi("residual", 0) != 0;
        b.separable = gbi("separable", 0) != 0;
        b.subs.resize(b.repeat);
    }
    gguf_free(gctx);

    // Load weights
    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, 4);

    struct gguf_init_params gp2 = {true, &m.ctx_w};
    gguf_context* gctx2 = gguf_init_from_file(path, gp2);

    size_t buf_size = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.ctx_w); t; t = ggml_get_next_tensor(m.ctx_w, t))
        buf_size += ggml_nbytes(t) + 64;
    m.buf_w = ggml_backend_alloc_buffer(ctx->backend, buf_size);
    ggml_tallocr alloc = ggml_tallocr_new(m.buf_w);
    for (ggml_tensor* t = ggml_get_first_tensor(m.ctx_w); t; t = ggml_get_next_tensor(m.ctx_w, t)) {
        ggml_tallocr_alloc(&alloc, t);
        size_t off = gguf_get_data_offset(gctx2) + gguf_get_tensor_offset(gctx2, gguf_find_tensor(gctx2, t->name));
        FILE* fp = fopen(path, "rb");
        if (!fp) return nullptr;
        fseek(fp, (long)off, SEEK_SET);
        std::vector<uint8_t> tmp(ggml_nbytes(t));
        size_t nr = fread(tmp.data(), 1, tmp.size(), fp);
        (void)nr;
        fclose(fp);
        ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size());
    }
    gguf_free(gctx2);

    auto get = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(m.ctx_w, name); };

    m.mel_fb = get("mel_filters");
    m.mel_win = get("mel_window");

    for (int i = 0; i < m.n_blocks; i++) {
        auto& b = m.blocks[i];
        char buf[64];
        for (int s = 0; s < b.repeat; s++) {
            snprintf(buf, sizeof(buf), "enc.%d.sub.%d.dw.weight", i, s);
            b.subs[s].dw_w = get(buf);
            snprintf(buf, sizeof(buf), "enc.%d.sub.%d.pw.weight", i, s);
            b.subs[s].pw_w = get(buf);
            snprintf(buf, sizeof(buf), "enc.%d.sub.%d.pw.bias", i, s);
            b.subs[s].pw_b = get(buf);
        }
        if (b.residual) {
            snprintf(buf, sizeof(buf), "enc.%d.res.weight", i);
            b.res_w = get(buf);
            snprintf(buf, sizeof(buf), "enc.%d.res.bias", i);
            b.res_b = get(buf);
        }
    }
    m.dec_w = get("decoder.weight");
    m.dec_b = get("decoder.bias");

    ggml_backend_t backends[1] = {ctx->backend};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, 1, 8192, false, false);

    fprintf(stderr, "marblenet_vad: %d blocks, %d classes, %d KB\n", m.n_blocks, m.num_classes, (int)(buf_size / 1024));
    return ctx;
}

// ── Forward ────────────────────────────────────────────────────────────────

static std::vector<float> mbn_forward(marblenet_vad_context* ctx, const float* mel, int T) {
    auto& m = ctx->model;
    const int n_mels = m.n_mels;

    size_t mem = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(4096, false);
    std::vector<uint8_t> meta(mem);
    ggml_init_params ip = {mem, meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    // Input mel [T, n_mels] for conv_1d
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, n_mels);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    ggml_tensor* h = mel_in;

    // Process each block
    for (int bi = 0; bi < m.n_blocks; bi++) {
        auto& b = m.blocks[bi];
        ggml_tensor* block_in = h; // for residual

        for (int si = 0; si < b.repeat; si++) {
            auto& sub = b.subs[si];

            if (b.separable && sub.dw_w) {
                // Depthwise conv: dw_w shape [C, 1, K] → use ggml_conv_1d_dw
                // Need input as [T, C] and kernel as [K, C] for conv_1d_dw
                int pad = (b.kernel - 1) * b.dilation / 2; // same padding
                h = ggml_pad_ext(ctx0, h, pad, pad, 0, 0, 0, 0, 0, 0);
                ggml_tensor* dw_f16 = ggml_cast(ctx0, sub.dw_w, GGML_TYPE_F16);
                h = ggml_conv_1d_dw(ctx0, dw_f16, h, b.stride, 0, b.dilation);
                // conv_1d_dw may return 3D; flatten to 2D
                if (ggml_n_dims(h) > 2)
                    h = ggml_reshape_2d(ctx0, h, h->ne[0], h->ne[1]);
            }

            if (sub.pw_w) {
                // Pointwise 1x1 conv (BN fused) — equivalent to Linear.
                // pw_w: [C_out, C_in, 1]. Squeeze to [C_out, C_in] for mul_mat.
                // h is [T, C_in]. Transpose to [C_in, T], mul_mat, transpose back.
                // pw_w: ggml ne[0]=1, ne[1]=C_in, ne[2]=C_out → reshape to [C_in, C_out]
                int pw_cin = (ggml_n_dims(sub.pw_w) >= 3) ? (int)sub.pw_w->ne[1] : (int)sub.pw_w->ne[0];
                int pw_cout = (ggml_n_dims(sub.pw_w) >= 3) ? (int)sub.pw_w->ne[2] : (int)sub.pw_w->ne[1];
                ggml_tensor* pw = ggml_reshape_2d(ctx0, sub.pw_w, pw_cin, pw_cout);
                h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [C_in, T]
                h = ggml_mul_mat(ctx0, pw, h);                // [C_out, T]
                if (sub.pw_b)
                    h = ggml_add(ctx0, h, sub.pw_b);
                h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [T, C_out]
            }

            // ReLU
            h = ggml_relu(ctx0, h);
        }

        // Residual connection (1x1 conv = Linear)
        if (b.residual && b.res_w) {
            int r_cin = (ggml_n_dims(b.res_w) >= 3) ? (int)b.res_w->ne[1] : (int)b.res_w->ne[0];
            int r_cout = (ggml_n_dims(b.res_w) >= 3) ? (int)b.res_w->ne[2] : (int)b.res_w->ne[1];
            ggml_tensor* rw = ggml_reshape_2d(ctx0, b.res_w, r_cin, r_cout);
            ggml_tensor* res = ggml_cont(ctx0, ggml_transpose(ctx0, block_in)); // [C_in, T]
            res = ggml_mul_mat(ctx0, rw, res);                                  // [C_out, T]
            if (b.res_b)
                res = ggml_add(ctx0, res, b.res_b);
            res = ggml_cont(ctx0, ggml_transpose(ctx0, res)); // [T, C_out]
            h = ggml_add(ctx0, h, res);
        }
    }

    // Decoder: h is [T_out, 128]. Transpose to [128, T_out], matmul with dec_w [2, 128].
    h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [128, T_out]
    h = ggml_mul_mat(ctx0, m.dec_w, h);           // [2, T_out]
    if (m.dec_b)
        h = ggml_add(ctx0, h, m.dec_b);

    // Softmax is done in post-processing
    ggml_set_name(h, "logits");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        ggml_free(ctx0);
        return {};
    }

    // Set mel input: need [T, n_mels] from our [n_mels, T] row-major mel
    std::vector<float> mel_t((size_t)T * n_mels);
    for (int m_idx = 0; m_idx < n_mels; m_idx++)
        for (int t = 0; t < T; t++)
            mel_t[(size_t)t * n_mels + m_idx] = mel[(size_t)m_idx * T + t];

    ggml_tensor* mt = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mt, mel_t.data(), 0, mel_t.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    int T_out = (int)out->ne[1];
    int nc = (int)out->ne[0]; // 2

    std::vector<float> logits_raw(nc * T_out);
    ggml_backend_tensor_get(out, logits_raw.data(), 0, logits_raw.size() * sizeof(float));
    ggml_free(ctx0);

    // Softmax over classes, extract speech probability (class 1)
    std::vector<float> probs(T_out);
    for (int t = 0; t < T_out; t++) {
        float l0 = logits_raw[t * nc + 0]; // non-speech
        float l1 = logits_raw[t * nc + 1]; // speech
        float mx = std::max(l0, l1);
        float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
        probs[t] = e1 / (e0 + e1);
    }
    return probs;
}

// ── Detect ─────────────────────────────────────────────────────────────────

extern "C" int marblenet_vad_detect(struct marblenet_vad_context* ctx, const float* samples, int n_samples,
                                    struct marblenet_vad_segment** segments, int* n_segments, float threshold,
                                    float min_speech_sec, float min_silence_sec) {
    if (!ctx || !samples || n_samples <= 0)
        return -1;
    auto& m = ctx->model;

    // Compute mel
    std::vector<float> hann(m.win_length), fb((size_t)m.n_mels * (m.n_fft / 2 + 1));
    if (m.mel_win)
        ggml_backend_tensor_get(m.mel_win, hann.data(), 0, m.win_length * sizeof(float));
    if (m.mel_fb)
        ggml_backend_tensor_get(m.mel_fb, fb.data(), 0, fb.size() * sizeof(float));

    int T_mel = 0;
    auto mel = mbn_compute_mel(samples, n_samples, hann.data(), m.win_length, fb.data(), m.n_mels, m.n_fft,
                               m.hop_length, &T_mel);
    if (mel.empty())
        return -1;

    auto probs = mbn_forward(ctx, mel.data(), T_mel);
    if (probs.empty())
        return -1;

    // Block 0 has stride=2, so output frame rate = 20ms (2 * hop_length/sr)
    float frame_sec = 0.02f; // 20ms per output frame

    // Extract segments with hysteresis
    float neg_thresh = std::max(threshold - 0.15f, 0.01f);
    int min_sp = (int)(min_speech_sec / frame_sec);
    int min_si = (int)(min_silence_sec / frame_sec);

    std::vector<marblenet_vad_segment> segs;
    bool in_speech = false;
    int start = 0, temp_end = 0;
    int nf = (int)probs.size();

    for (int i = 0; i < nf; i++) {
        if (probs[i] >= threshold && !in_speech) {
            in_speech = true;
            start = i;
            temp_end = 0;
        }
        if (probs[i] < neg_thresh && in_speech) {
            if (!temp_end)
                temp_end = i;
            if (i - temp_end >= min_si) {
                if (temp_end - start >= min_sp)
                    segs.push_back({start * frame_sec, temp_end * frame_sec});
                in_speech = false;
                temp_end = 0;
            }
        } else if (probs[i] >= threshold && temp_end) {
            temp_end = 0;
        }
    }
    if (in_speech && nf - start >= min_sp)
        segs.push_back({start * frame_sec, nf * frame_sec});

    *n_segments = (int)segs.size();
    if (segs.empty()) {
        *segments = nullptr;
    } else {
        *segments = (marblenet_vad_segment*)malloc(segs.size() * sizeof(marblenet_vad_segment));
        memcpy(*segments, segs.data(), segs.size() * sizeof(marblenet_vad_segment));
    }
    return 0;
}

extern "C" void marblenet_vad_free(struct marblenet_vad_context* ctx) {
    if (!ctx)
        return;
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->model.buf_w)
        ggml_backend_buffer_free(ctx->model.buf_w);
    if (ctx->model.ctx_w)
        ggml_free(ctx->model.ctx_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}
