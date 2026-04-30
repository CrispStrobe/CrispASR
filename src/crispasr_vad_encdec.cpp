// whisper_vad_encdec.cpp — Whisper-encoder + TransformerDecoder VAD runtime.
//
// Architecture: whisper-base encoder (6L, 512d, 8h, Conv1d front-end) +
// 2-layer TransformerDecoder (self-attn + cross-attn to encoder output) +
// frame_classifier (Linear 512→1 + sigmoid). Produces 1500 per-frame speech
// probabilities from 30s of 80-bin mel input.
//
// Model: TransWithAI/Whisper-Vad-EncDec-ASMR-onnx converted to GGUF.

#include "crispasr_vad_encdec.h"

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
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Model ──────────────────────────────────────────────────────────────────

struct wvad_enc_layer {
    ggml_tensor* attn_ln_w = nullptr; // self_attn_layer_norm.weight
    ggml_tensor* attn_ln_b = nullptr; // self_attn_layer_norm.bias
    ggml_tensor* q_w = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr; // no bias
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;      // out_proj.weight
    ggml_tensor* o_b = nullptr;      // out_proj.bias
    ggml_tensor* ffn_ln_w = nullptr; // final_layer_norm.weight
    ggml_tensor* ffn_ln_b = nullptr; // final_layer_norm.bias
    ggml_tensor* fc1_w = nullptr;
    ggml_tensor* fc1_b = nullptr;
    ggml_tensor* fc2_w = nullptr;
    ggml_tensor* fc2_b = nullptr;
};

struct wvad_dec_layer {
    // Self-attention (fused QKV via in_proj)
    ggml_tensor* sa_in_proj_w = nullptr; // [1536, 512]
    ggml_tensor* sa_in_proj_b = nullptr; // [1536]
    ggml_tensor* sa_o_w = nullptr;       // [512, 512]
    ggml_tensor* sa_o_b = nullptr;       // [512]
    ggml_tensor* norm1_w = nullptr;      // [512]
    ggml_tensor* norm1_b = nullptr;
    // Cross-attention to encoder
    ggml_tensor* ca_in_proj_w = nullptr; // [1536, 512]
    ggml_tensor* ca_in_proj_b = nullptr; // [1536]
    ggml_tensor* ca_o_w = nullptr;       // [512, 512]
    ggml_tensor* ca_o_b = nullptr;       // [512]
    ggml_tensor* norm2_b = nullptr;      // [512] (norm2 weight often missing in ONNX export)
    // FFN
    ggml_tensor* fc1_w = nullptr; // linear1
    ggml_tensor* fc1_b = nullptr;
    ggml_tensor* fc2_w = nullptr; // linear2
    ggml_tensor* fc2_b = nullptr;
    ggml_tensor* norm3_b = nullptr; // [512]
};

struct wvad_model {
    int n_enc_layers = 6;
    int d_model = 512;
    int n_heads = 8;
    int ffn_dim = 2048;
    int n_dec_layers = 2;
    int n_mels = 80;
    int n_frames = 1500;
    int frame_ms = 20;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Conv front-end
    ggml_tensor* conv1_w = nullptr; // [3, 80, 512]
    ggml_tensor* conv1_b = nullptr;
    ggml_tensor* conv2_w = nullptr; // [3, 512, 512]
    ggml_tensor* conv2_b = nullptr;
    ggml_tensor* pos_emb = nullptr; // [512, 1500]

    // Encoder
    std::vector<wvad_enc_layer> enc;
    ggml_tensor* enc_ln_w = nullptr; // encoder.layer_norm
    ggml_tensor* enc_ln_b = nullptr;

    // Decoder
    std::vector<wvad_dec_layer> dec;
    ggml_tensor* dec_pos_queries = nullptr; // [512, 1500, 1]

    // Frame classifier
    ggml_tensor* cls_w = nullptr; // [512, 1]
    ggml_tensor* cls_b = nullptr; // [1]

    // Mel
    ggml_tensor* mel_window = nullptr;
    ggml_tensor* mel_filters = nullptr;
};

struct whisper_vad_encdec_context {
    wvad_model model;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 4;
};

// ── FFT ────────────────────────────────────────────────────────────────────

static void wvad_dft(const float* in, int N, float* out) {
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

static void wvad_fft(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    int h = N / 2;
    if (N - h * 2 == 1) {
        wvad_dft(in, N, out);
        return;
    }
    float* ev = in + N;
    for (int i = 0; i < h; i++)
        ev[i] = in[2 * i];
    float* ev_f = out + 2 * N;
    wvad_fft(ev, h, ev_f);
    float* od = ev;
    for (int i = 0; i < h; i++)
        od[i] = in[2 * i + 1];
    float* od_f = ev_f + N;
    wvad_fft(od, h, od_f);
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

static void wvad_fft_wrap(const float* in, int N, float* out) {
    static thread_local std::vector<float> si, so;
    if ((int)si.size() < 4 * N)
        si.assign(4 * N, 0);
    if ((int)so.size() < 8 * N)
        so.assign(8 * N, 0);
    memcpy(si.data(), in, N * sizeof(float));
    wvad_fft(si.data(), N, so.data());
    memcpy(out, so.data(), 2 * N * sizeof(float));
}

// ── Mel (whisper-style, 80 bins) ──────────────────────────────────────────

static std::vector<float> wvad_compute_mel(const float* pcm, int n_samples, const float* hann, const float* fb,
                                           int n_fft, int hop, int n_mels) {
    const int n_freqs = n_fft / 2 + 1;
    // Center-pad
    int padded_len = n_samples + n_fft;
    std::vector<float> padded(padded_len, 0);
    memcpy(padded.data() + n_fft / 2, pcm, n_samples * sizeof(float));

    int n_frames = (padded_len - n_fft) / hop;
    std::vector<float> mel((size_t)n_mels * n_frames, 0);
    std::vector<float> fft_in(4 * n_fft, 0);
    std::vector<float> fft_out(8 * n_fft, 0);

    for (int t = 0; t < n_frames; t++) {
        // Window
        for (int i = 0; i < n_fft; i++)
            fft_in[i] = padded[t * hop + i] * hann[i];
        wvad_fft_wrap(fft_in.data(), n_fft, fft_out.data());
        // Power spectrum
        std::vector<float> power(n_freqs);
        for (int f = 0; f < n_freqs; f++)
            power[f] = fft_out[2 * f] * fft_out[2 * f] + fft_out[2 * f + 1] * fft_out[2 * f + 1];
        // Mel filterbank: fb is [n_freqs, n_mels]
        for (int m = 0; m < n_mels; m++) {
            double sum = 0;
            for (int f = 0; f < n_freqs; f++)
                sum += (double)power[f] * (double)fb[f * n_mels + m];
            mel[(size_t)m * n_frames + t] = (float)sum;
        }
    }
    // Log10 + clip + normalize (whisper style)
    float max_val = -1e20f;
    for (auto& v : mel) {
        v = std::log10(std::max(v, 1e-10f));
        max_val = std::max(max_val, v);
    }
    for (auto& v : mel) {
        v = std::max(v, max_val - 8.0f);
        v = (v + 4.0f) / 4.0f;
    }
    return mel; // [n_mels, n_frames] row-major
}

// ── Init ───────────────────────────────────────────────────────────────────

extern "C" struct whisper_vad_encdec_context* whisper_vad_encdec_init(const char* path) {
    auto* ctx = new whisper_vad_encdec_context();
    auto& m = ctx->model;

    // Read metadata
    struct gguf_init_params gp = {true, &m.ctx_w};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx) {
        fprintf(stderr, "whisper_vad_encdec: failed to open '%s'\n", path);
        delete ctx;
        return nullptr;
    }

    auto gu32 = [&](const char* k, uint32_t d) -> uint32_t {
        int64_t id = gguf_find_key(gctx, k);
        return id >= 0 ? gguf_get_val_u32(gctx, id) : d;
    };
    m.n_enc_layers = (int)gu32("whisper_vad.encoder_layers", 6);
    m.d_model = (int)gu32("whisper_vad.encoder_dim", 512);
    m.n_heads = (int)gu32("whisper_vad.encoder_heads", 8);
    m.ffn_dim = (int)gu32("whisper_vad.encoder_ffn_dim", 2048);
    m.n_dec_layers = (int)gu32("whisper_vad.decoder_layers", 2);
    m.n_mels = (int)gu32("whisper_vad.n_mels", 80);
    m.n_frames = (int)gu32("whisper_vad.n_frames", 1500);
    m.frame_ms = (int)gu32("whisper_vad.frame_duration_ms", 20);
    gguf_free(gctx);

    // Backend
    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Load weights
    ggml_init_params wip = {ggml_tensor_overhead() * 200 + 1024 * 1024, nullptr, true};
    m.ctx_w = ggml_init(wip);

    // Re-open and load all tensors
    struct gguf_init_params gp2 = {true, &m.ctx_w};
    gguf_context* gctx2 = gguf_init_from_file(path, gp2);
    if (!gctx2) {
        delete ctx;
        return nullptr;
    }

    size_t buf_size = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(m.ctx_w); t; t = ggml_get_next_tensor(m.ctx_w, t))
        buf_size += ggml_nbytes(t) + 64;
    m.buf_w = ggml_backend_alloc_buffer(ctx->backend, buf_size);
    ggml_tallocr alloc = ggml_tallocr_new(m.buf_w);
    for (ggml_tensor* t = ggml_get_first_tensor(m.ctx_w); t; t = ggml_get_next_tensor(m.ctx_w, t)) {
        ggml_tallocr_alloc(&alloc, t);
        size_t off = gguf_get_data_offset(gctx2) + gguf_get_tensor_offset(gctx2, gguf_find_tensor(gctx2, t->name));
        FILE* fp = fopen(path, "rb");
        if (!fp)
            return nullptr;
        fseek(fp, (long)off, SEEK_SET);
        std::vector<uint8_t> buf(ggml_nbytes(t));
        size_t nr = fread(buf.data(), 1, buf.size(), fp);
        (void)nr;
        fclose(fp);
        ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
    }
    gguf_free(gctx2);

    // Bind tensors by name
    auto get = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(m.ctx_w, name); };

    m.conv1_w = get("encoder.conv1.weight");
    m.conv1_b = get("encoder.conv1.bias");
    m.conv2_w = get("encoder.conv2.weight");
    m.conv2_b = get("encoder.conv2.bias");
    m.pos_emb = get("encoder.embed_positions.weight");
    m.enc_ln_w = get("encoder.layer_norm.weight");
    m.enc_ln_b = get("encoder.layer_norm.bias");

    m.enc.resize(m.n_enc_layers);
    for (int i = 0; i < m.n_enc_layers; i++) {
        char buf[128];
        auto g = [&](const char* s) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "encoder.layers.%d.%s", i, s);
            return get(buf);
        };
        auto& L = m.enc[i];
        L.attn_ln_w = g("self_attn_layer_norm.weight");
        L.attn_ln_b = g("self_attn_layer_norm.bias");
        L.q_w = g("self_attn.q_proj.weight");
        L.q_b = g("self_attn.q_proj.bias");
        L.k_w = g("self_attn.k_proj.weight");
        L.v_w = g("self_attn.v_proj.weight");
        L.v_b = g("self_attn.v_proj.bias");
        L.o_w = g("self_attn.out_proj.weight");
        L.o_b = g("self_attn.out_proj.bias");
        L.ffn_ln_w = g("final_layer_norm.weight");
        L.ffn_ln_b = g("final_layer_norm.bias");
        L.fc1_w = g("fc1.weight");
        L.fc1_b = g("fc1.bias");
        L.fc2_w = g("fc2.weight");
        L.fc2_b = g("fc2.bias");
    }

    m.dec.resize(m.n_dec_layers);
    for (int i = 0; i < m.n_dec_layers; i++) {
        char buf[128];
        auto g = [&](const char* s) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "decoder.decoder.layers.%d.%s", i, s);
            auto* t = get(buf);
            if (!t) {
                // Try without the double "decoder.decoder." prefix
                snprintf(buf, sizeof(buf), "decoder.layers.%d.%s", i, s);
                t = get(buf);
            }
            return t;
        };
        auto& L = m.dec[i];
        L.sa_in_proj_w = g("self_attn.in_proj_weight");
        L.sa_in_proj_b = g("self_attn.in_proj_bias");
        L.sa_o_w = g("self_attn.out_proj.weight");
        L.sa_o_b = g("self_attn.out_proj.bias");
        L.norm1_w = g("norm1.weight");
        L.norm1_b = g("norm1.bias");
        L.ca_in_proj_w = g("multihead_attn.in_proj_weight");
        L.ca_in_proj_b = g("multihead_attn.in_proj_bias");
        L.ca_o_w = g("multihead_attn.out_proj.weight");
        L.ca_o_b = g("multihead_attn.out_proj.bias");
        L.norm2_b = g("norm2.bias");
        L.fc1_w = g("linear1.weight");
        L.fc1_b = g("linear1.bias");
        L.fc2_w = g("linear2.weight");
        L.fc2_b = g("linear2.bias");
        L.norm3_b = g("norm3.bias");
    }

    m.dec_pos_queries = get("decoder.position_queries");
    m.cls_w = get("frame_classifier.weight");
    m.cls_b = get("frame_classifier.1.bias");
    m.mel_window = get("mel_window");
    m.mel_filters = get("mel_filters");

    // Scheduler
    ggml_backend_t backends[1] = {ctx->backend};
    ctx->sched = ggml_backend_sched_new(backends, nullptr, 1, 32768, false, false);

    fprintf(stderr, "whisper_vad_encdec: %dL enc + %dL dec, %dd, %d frames, %d ms/frame\n", m.n_enc_layers,
            m.n_dec_layers, m.d_model, m.n_frames, m.frame_ms);
    return ctx;
}

// ── Forward pass ───────────────────────────────────────────────────────────

struct wvad_result {
    std::vector<float> probs;   // frame-level speech probabilities
    std::vector<float> enc_emb; // encoder output [d * T] — reusable for whisper ASR
};

static wvad_result wvad_forward(whisper_vad_encdec_context* ctx, const float* mel_data, int n_frames) {
    auto& m = ctx->model;
    const int d = m.d_model;
    const int nh = m.n_heads;
    const int hd = d / nh;
    const int T = m.n_frames; // always 1500
    const float eps = 1e-5f;
    const float scale = 1.0f / sqrtf((float)hd);

    size_t mem = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(32768, false);
    std::vector<uint8_t> meta(mem);
    ggml_init_params ip = {mem, meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // Input mel [T*2, n_mels] — whisper pads mel to 2*n_ctx=3000 frames
    // but our mel is [n_mels, T_mel]. We need [T_mel, n_mels] for conv_1d.
    // Actually whisper conv_1d_ph expects [T, C] input where C=n_mels.
    ggml_tensor* mel_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 3000, m.n_mels);
    ggml_set_name(mel_in, "mel");
    ggml_set_input(mel_in);

    // Helper: ensure tensor is F32 (dequant Q4_K etc. for ops that need F32)
    auto f32 = [&](ggml_tensor* t) -> ggml_tensor* {
        return (t && t->type != GGML_TYPE_F32) ? ggml_cast(ctx0, t, GGML_TYPE_F32) : t;
    };

    // Conv1d front-end
    // ggml conv_1d_ph requires F16 kernel on CPU (im2col_f16 assert)
    ggml_tensor* conv1_w_f16 = ggml_cast(ctx0, m.conv1_w, GGML_TYPE_F16);
    ggml_tensor* h = ggml_conv_1d_ph(ctx0, conv1_w_f16, mel_in, 1, 1);
    if (m.conv1_b) {
        h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [C, T]
        h = ggml_add(ctx0, h, f32(m.conv1_b));
        h = ggml_cont(ctx0, ggml_transpose(ctx0, h));
    }
    h = ggml_gelu(ctx0, h);
    ggml_tensor* conv2_w_f16 = ggml_cast(ctx0, m.conv2_w, GGML_TYPE_F16);
    h = ggml_conv_1d_ph(ctx0, conv2_w_f16, h, 2, 1); // stride=2: 3000->1500
    if (m.conv2_b) {
        h = ggml_cont(ctx0, ggml_transpose(ctx0, h));
        h = ggml_add(ctx0, h, f32(m.conv2_b));
        h = ggml_cont(ctx0, ggml_transpose(ctx0, h));
    }
    h = ggml_gelu(ctx0, h);

    // h after conv2+bias is [T, d] from transpose back. It was already transposed in bias add.
    // After conv2 bias: h = [T, C] → transposed back from [C, T].
    // Actually after conv2 bias: we did trans→add→trans, so h is back to [T, C]=[1500, 512].
    // Need to transpose to [d, T] = [512, 1500].
    h = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [d, T]
    // pos_emb is [d, 1500]. If T != 1500 (e.g. from shorter audio), slice pos_emb.
    if (m.pos_emb) {
        int T_actual = (int)h->ne[1];
        if (T_actual == T) {
            h = ggml_add(ctx0, h, f32(m.pos_emb));
        } else {
            ggml_tensor* pe = ggml_view_2d(ctx0, m.pos_emb, d, T_actual, m.pos_emb->nb[1], 0);
            h = ggml_add(ctx0, h, pe);
        }
    }


    // ── Encoder (6 layers) ──
    for (int il = 0; il < m.n_enc_layers; il++) {
        auto& L = m.enc[il];
        ggml_tensor* residual = h;

        // Pre-attention LayerNorm
        h = ggml_norm(ctx0, h, eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, f32(L.attn_ln_w)), f32(L.attn_ln_b));

        // Q (with bias), K (no bias), V (with bias)
        ggml_tensor* Q = ggml_mul_mat(ctx0, L.q_w, h);
        if (L.q_b)
            Q = ggml_add(ctx0, Q, f32(L.q_b));
        ggml_tensor* K = ggml_mul_mat(ctx0, L.k_w, h);
        ggml_tensor* V = ggml_mul_mat(ctx0, L.v_w, h);
        if (L.v_b)
            V = ggml_add(ctx0, V, f32(L.v_b));

        // Reshape + permute for flash_attn: (hd, T, nh)
        Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q, hd, nh, T), 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, hd, nh, T), 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hd, nh, T), 0, 2, 1, 3));

        ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, ggml_cast(ctx0, K, GGML_TYPE_F16),
                                                ggml_cast(ctx0, V, GGML_TYPE_F16), nullptr, scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx0, attn, d, T);

        // Output projection + residual
        attn = ggml_mul_mat(ctx0, L.o_w, attn);
        if (L.o_b)
            attn = ggml_add(ctx0, attn, f32(L.o_b));
        h = ggml_add(ctx0, residual, attn);

        // FFN
        residual = h;
        h = ggml_norm(ctx0, h, eps);
        h = ggml_add(ctx0, ggml_mul(ctx0, h, f32(L.ffn_ln_w)), f32(L.ffn_ln_b));
        h = ggml_mul_mat(ctx0, L.fc1_w, h);
        if (f32(L.fc1_b))
            h = ggml_add(ctx0, h, f32(L.fc1_b));
        h = ggml_gelu(ctx0, h);
        h = ggml_mul_mat(ctx0, L.fc2_w, h);
        if (f32(L.fc2_b))
            h = ggml_add(ctx0, h, f32(L.fc2_b));
        h = ggml_add(ctx0, residual, h);
    }

    // Final encoder LayerNorm
    h = ggml_norm(ctx0, h, eps);
    h = ggml_add(ctx0, ggml_mul(ctx0, h, f32(m.enc_ln_w)), f32(m.enc_ln_b));
    // h = encoder output [d, T]

    ggml_tensor* enc_out = h;
    ggml_set_name(enc_out, "enc_out");
    ggml_set_output(enc_out);

    // ── Decoder (2 layers with self-attn + cross-attn) ──
    // Input: position queries [d, T, 1] → squeeze to [d, T]
    ggml_tensor* tgt = m.dec_pos_queries;
    if (ggml_n_dims(tgt) > 2)
        tgt = ggml_cont(ctx0, ggml_reshape_2d(ctx0, tgt, d, T));

    for (int il = 0; il < m.n_dec_layers; il++) {
        auto& L = m.dec[il];

        // ── Self-attention on target (position queries) ──
        ggml_tensor* residual = tgt;
        ggml_tensor* qkv = ggml_mul_mat(ctx0, L.sa_in_proj_w, tgt);
        if (L.sa_in_proj_b)
            qkv = ggml_add(ctx0, qkv, f32(L.sa_in_proj_b));
        // Split Q, K, V each [d, T]
        ggml_tensor* sa_Q = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, d, T, qkv->nb[1], 0));
        ggml_tensor* sa_K = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, d, T, qkv->nb[1], d * sizeof(float)));
        ggml_tensor* sa_V = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv, d, T, qkv->nb[1], 2 * d * sizeof(float)));

        sa_Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, sa_Q, hd, nh, T), 0, 2, 1, 3));
        sa_K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, sa_K, hd, nh, T), 0, 2, 1, 3));
        sa_V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, sa_V, hd, nh, T), 0, 2, 1, 3));

        ggml_tensor* sa = ggml_flash_attn_ext(ctx0, sa_Q, ggml_cast(ctx0, sa_K, GGML_TYPE_F16),
                                              ggml_cast(ctx0, sa_V, GGML_TYPE_F16), nullptr, scale, 0.0f, 0.0f);
        sa = ggml_reshape_2d(ctx0, sa, d, T);
        sa = ggml_mul_mat(ctx0, L.sa_o_w, sa);
        if (L.sa_o_b)
            sa = ggml_add(ctx0, sa, f32(L.sa_o_b));
        tgt = ggml_add(ctx0, residual, sa);

        // norm1
        if (L.norm1_w)
            tgt = ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, tgt, eps), f32(L.norm1_w)), f32(L.norm1_b));
        else if (f32(L.norm1_b))
            tgt = ggml_add(ctx0, ggml_norm(ctx0, tgt, eps), f32(L.norm1_b));

        // Cross-attention: Q from target, K/V from encoder.
        // Fused in_proj_w [ne0=d, ne1=3d]. Do full mul_mat then split output.
        residual = tgt;
        ggml_tensor* ca_q_all = ggml_mul_mat(ctx0, L.ca_in_proj_w, tgt);      // [3d, T]
        ggml_tensor* ca_kv_all = ggml_mul_mat(ctx0, L.ca_in_proj_w, enc_out); // [3d, T]
        if (L.ca_in_proj_b) {
            ca_q_all = ggml_add(ctx0, ca_q_all, f32(L.ca_in_proj_b));
            ca_kv_all = ggml_add(ctx0, ca_kv_all, f32(L.ca_in_proj_b));
        }
        // Q from tgt result [0..d), K from enc result [d..2d), V from enc result [2d..3d)
        ggml_tensor* ca_Q = ggml_cont(ctx0, ggml_view_2d(ctx0, ca_q_all, d, T, ca_q_all->nb[1], 0));
        ggml_tensor* ca_K =
            ggml_cont(ctx0, ggml_view_2d(ctx0, ca_kv_all, d, T, ca_kv_all->nb[1], (size_t)d * sizeof(float)));
        ggml_tensor* ca_V =
            ggml_cont(ctx0, ggml_view_2d(ctx0, ca_kv_all, d, T, ca_kv_all->nb[1], (size_t)2 * d * sizeof(float)));

        ca_Q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, ca_Q, hd, nh, T), 0, 2, 1, 3));
        ca_K = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, ca_K, hd, nh, T), 0, 2, 1, 3));
        ca_V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, ca_V, hd, nh, T), 0, 2, 1, 3));

        ggml_tensor* ca = ggml_flash_attn_ext(ctx0, ca_Q, ggml_cast(ctx0, ca_K, GGML_TYPE_F16),
                                              ggml_cast(ctx0, ca_V, GGML_TYPE_F16), nullptr, scale, 0.0f, 0.0f);
        ca = ggml_reshape_2d(ctx0, ca, d, T);
        ca = ggml_mul_mat(ctx0, L.ca_o_w, ca);
        if (L.ca_o_b)
            ca = ggml_add(ctx0, ca, f32(L.ca_o_b));
        tgt = ggml_add(ctx0, residual, ca);

        // norm2
        if (f32(L.norm2_b))
            tgt = ggml_add(ctx0, ggml_norm(ctx0, tgt, eps), f32(L.norm2_b));

        // FFN
        residual = tgt;
        ggml_tensor* ff = ggml_mul_mat(ctx0, L.fc1_w, tgt);
        if (f32(L.fc1_b))
            ff = ggml_add(ctx0, ff, f32(L.fc1_b));
        ff = ggml_relu(ctx0, ff); // PyTorch TransformerDecoder default is ReLU
        ff = ggml_mul_mat(ctx0, L.fc2_w, ff);
        if (f32(L.fc2_b))
            ff = ggml_add(ctx0, ff, f32(L.fc2_b));
        tgt = ggml_add(ctx0, residual, ff);

        // norm3
        if (f32(L.norm3_b))
            tgt = ggml_add(ctx0, ggml_norm(ctx0, tgt, eps), f32(L.norm3_b));
    }

    // ── Frame classifier: Linear(512→1) + sigmoid ──
    ggml_tensor* logits = ggml_mul_mat(ctx0, m.cls_w, tgt); // [1, T]
    if (m.cls_b)
        logits = ggml_add(ctx0, logits, f32(m.cls_b));
    // Sigmoid in post-processing (not in graph for numerical stability)

    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    // Run graph
    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "whisper_vad_encdec: graph alloc failed\n");
        ggml_free(ctx0);
        return {};
    }

    // Set mel input — need to transpose from [n_mels, T_mel] to [T_mel*2, n_mels]
    // and pad to 3000 frames (whisper convention: 30s = 3000 frames at hop=160)
    std::vector<float> mel_transposed(3000 * m.n_mels, 0);
    int T_mel = std::min(n_frames, 3000);
    for (int m_idx = 0; m_idx < m.n_mels; m_idx++)
        for (int t = 0; t < T_mel; t++)
            mel_transposed[(size_t)t * m.n_mels + m_idx] = mel_data[(size_t)m_idx * n_frames + t];

    ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    ggml_backend_tensor_set(mel_t, mel_transposed.data(), 0, mel_transposed.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "whisper_vad_encdec: graph compute failed\n");
        ggml_free(ctx0);
        return {};
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, "logits");
    int n_out = (int)(ggml_nelements(out));
    std::vector<float> logits_vec(n_out);
    ggml_backend_tensor_get(out, logits_vec.data(), 0, n_out * sizeof(float));

    // Read encoder output (for reuse by whisper ASR — encoder attention layers were fine-tuned despite freeze_encoder:true in config)
    ggml_tensor* enc_t = ggml_graph_get_tensor(gf, "enc_out");
    std::vector<float> enc_vec;
    if (enc_t) {
        int enc_n = (int)ggml_nelements(enc_t);
        enc_vec.resize(enc_n);
        ggml_backend_tensor_get(enc_t, enc_vec.data(), 0, enc_n * sizeof(float));
    }

    ggml_free(ctx0);

    // Sigmoid
    std::vector<float> probs(n_out);
    for (int i = 0; i < n_out; i++)
        probs[i] = 1.0f / (1.0f + expf(-logits_vec[i]));

    return {probs, enc_vec};
}

// ── Detect ─────────────────────────────────────────────────────────────────

extern "C" int whisper_vad_encdec_detect(struct whisper_vad_encdec_context* ctx, const float* samples, int n_samples,
                                         struct whisper_vad_encdec_segment** segments, int* n_segments, float threshold,
                                         float min_speech_sec, float min_silence_sec, float** probs_out,
                                         int* n_frames_out, float** encoder_out, int* encoder_out_size) {
    if (!ctx || !samples || n_samples <= 0)
        return -1;
    auto& m = ctx->model;

    // Compute mel
    std::vector<float> hann(400), fb(201 * 80);
    if (m.mel_window && m.mel_filters) {
        ggml_backend_tensor_get(m.mel_window, hann.data(), 0, 400 * sizeof(float));
        ggml_backend_tensor_get(m.mel_filters, fb.data(), 0, 201 * 80 * sizeof(float));
    } else {
        // Generate hann + filterbank
        for (int i = 0; i < 400; i++)
            hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / 400));
        // Minimal HTK mel filterbank
        auto h2m = [](double f) { return 2595.0 * log10(1.0 + f / 700.0); };
        auto m2h = [](double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); };
        double mlo = h2m(0), mhi = h2m(8000);
        std::vector<double> pts(82);
        for (int i = 0; i < 82; i++)
            pts[i] = m2h(mlo + (mhi - mlo) * i / 81);
        for (int mi = 0; mi < 80; mi++) {
            double lo = pts[mi], c = pts[mi + 1], hi = pts[mi + 2];
            if (hi <= lo)
                continue;
            double en = 2.0 / (hi - lo);
            for (int f = 0; f < 201; f++) {
                double freq = 16000.0 * f / 400;
                double v = 0;
                if (freq >= lo && freq <= c && c > lo)
                    v = (freq - lo) / (c - lo);
                else if (freq > c && freq <= hi && hi > c)
                    v = (hi - freq) / (hi - c);
                fb[f * 80 + mi] = (float)(v * en);
            }
        }
    }

    auto mel = wvad_compute_mel(samples, n_samples, hann.data(), fb.data(), 400, 160, 80);
    int T_mel = (int)(mel.size() / 80);

    // Forward pass
    auto fwd = wvad_forward(ctx, mel.data(), T_mel);
    if (fwd.probs.empty())
        return -1;
    auto& probs = fwd.probs;

    int nf = (int)probs.size();

    // Output raw probabilities if requested
    if (probs_out) {
        *probs_out = (float*)malloc(nf * sizeof(float));
        memcpy(*probs_out, probs.data(), nf * sizeof(float));
    }
    if (n_frames_out)
        *n_frames_out = nf;

    // Output encoder embeddings if requested (fine-tuned encoder output (NOT reusable for whisper ASR),
    // can be injected into whisper's cross-attention state to skip the ASR encoder pass)
    if (encoder_out && !fwd.enc_emb.empty()) {
        *encoder_out = (float*)malloc(fwd.enc_emb.size() * sizeof(float));
        memcpy(*encoder_out, fwd.enc_emb.data(), fwd.enc_emb.size() * sizeof(float));
    }
    if (encoder_out_size)
        *encoder_out_size = (int)fwd.enc_emb.size();

    // Extract segments with hysteresis
    float neg_thresh = std::max(threshold - 0.15f, 0.01f);
    int min_speech_frames = (int)(min_speech_sec * 1000 / m.frame_ms);
    int min_silence_frames = (int)(min_silence_sec * 1000 / m.frame_ms);

    std::vector<whisper_vad_encdec_segment> segs;
    bool in_speech = false;
    int start = 0, temp_end = 0;

    for (int i = 0; i < nf; i++) {
        if (probs[i] >= threshold && !in_speech) {
            in_speech = true;
            start = i;
            temp_end = 0;
        }
        if (probs[i] < neg_thresh && in_speech) {
            if (!temp_end)
                temp_end = i;
            if (i - temp_end >= min_silence_frames) {
                if (temp_end - start >= min_speech_frames)
                    segs.push_back({start * m.frame_ms / 1000.0f, temp_end * m.frame_ms / 1000.0f});
                in_speech = false;
                temp_end = 0;
            }
        } else if (probs[i] >= threshold && temp_end) {
            temp_end = 0;
        }
    }
    if (in_speech && nf - start >= min_speech_frames)
        segs.push_back({start * m.frame_ms / 1000.0f, nf * m.frame_ms / 1000.0f});

    // Output
    *n_segments = (int)segs.size();
    if (segs.empty()) {
        *segments = nullptr;
    } else {
        *segments = (whisper_vad_encdec_segment*)malloc(segs.size() * sizeof(whisper_vad_encdec_segment));
        memcpy(*segments, segs.data(), segs.size() * sizeof(whisper_vad_encdec_segment));
    }
    return 0;
}

// ── Free ───────────────────────────────────────────────────────────────────

extern "C" void whisper_vad_encdec_free(struct whisper_vad_encdec_context* ctx) {
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
